#!/usr/bin/env python3

import subprocess
import tempfile
from pathlib import Path
from glob import glob
import os
import random

import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from aniposelib.cameras import CameraGroup, Camera
import cv2

import numpy as np

from tqdm import tqdm
from collections import defaultdict
import pandas as pd

tempdir = 'tempframes'

# root = '/groups/voigts/voigtslab/outdoor/2025_10_20/data'
# root = '/groups/voigts/voigtslab/outdoor/2025_09_25_mouse_new_day3/data'

root = '/groups/voigts/voigtslab/outdoor/2026_04_10_mouse_left/data'
tracked_root = '/groups/karashchuk/karashchuklab/outdoor_analysis/2026_04_10_mouse_left'

calib_fname_init = os.path.join(tracked_root, 'calibration_vggt_init.toml')
calib_fname_out = os.path.join(tracked_root, 'calibration_adjusted.toml')
points_fname_out = os.path.join(tracked_root, 'points_3d.npz')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

def extract_first_frames(video_paths, output_dir=None):
    """
    Extract the first frame from a list of videos using ffmpeg.
    
    Args:
        video_paths: List of video file paths
        output_dir: Optional output directory. Uses temp folder if None.
    
    Returns:
        List of paths to extracted frame images
    """
    # Create temp directory if output_dir not specified
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="video_frames_")
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_paths = []
    
    for video_path in video_paths:
        video_path = Path(video_path)
        
        # Generate output filename
        output_filename = f"{video_path.stem}_frame.jpg"
        output_path = Path(output_dir) / output_filename
        
        # ffmpeg command to extract first frame
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-i', str(video_path),
            '-vframes', '1',  # Extract 1 frame
            '-q:v', '2',      # High quality (2-5 range, lower is better)
            '-y',             # Overwrite output file
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            output_paths.append(str(output_path))
            # print(f"Extracted frame: {output_path}")
        except subprocess.CalledProcessError as e:
            # print(f"Error processing {video_path}: {e.stderr.decode()}")
            output_paths.append(None)
    
    return output_paths



possible = sorted(glob(os.path.join(root, 'video_10_*.avi')))

templates = [p.replace('video_10', 'video_*') for p in possible]

# Pick at most 8 random templates if there are more than 8
if len(templates) > 8:
    templates = random.sample(templates, 8)

all_extrinsics = []
all_intrinsics = []
for template in templates:
    print(template)
    
    vidnames = sorted(glob(template))
    print(len(vidnames))

    # if len(vidnames) < 7:
    #     continue
    
    frames = extract_first_frames(vidnames, output_dir=tempdir)
    
    images = load_and_preprocess_images(frames).to(device)


    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, [512, 640])
        all_extrinsics.append(extrinsic[0])
        all_intrinsics.append(intrinsic[0])

all_extrinsics = torch.stack(all_extrinsics)
all_intrinsics = torch.stack(all_intrinsics)


datas = defaultdict(list)
fnames = glob(os.path.join(tracked_root, '*.pq'))
print('loading 2d points')
for fname in tqdm(fnames, ncols=70):
    basename = os.path.basename(fname)
    cname = os.path.basename(fname).split('_')[1]
    df = pd.read_parquet(fname)
    df['video'] = fname.replace('.pq', '')
    df['cam'] = cname
    df['framenum'] = np.arange(len(df))
    datas[cname].append(df)

cam_names = sorted(list(datas.keys()))

datas_combined = dict()
for k in datas.keys():
    datas_combined[k] = pd.concat(datas[k], ignore_index=True)

min_dt = min([d['timestamp'].min() for d in datas_combined.values()])
max_dt = max([d['timestamp'].max() for d in datas_combined.values()])
delta = pd.Timedelta(seconds=1/30.0)

stamps = pd.DataFrame({'timestamp': pd.date_range(min_dt, max_dt, freq=delta)})

frames_dict = defaultdict(list)
for cname in datas_combined.keys():
    d = datas_combined[cname].sort_values('timestamp').reset_index()
    frames_dict[cname] = pd.merge_asof(
        stamps, d, on='timestamp', direction='nearest', tolerance=pd.Timedelta(seconds=1/15.0)
    )

all_p2ds = []
scores = []
for cname in cam_names:
    p2d = frames_dict[cname][['x', 'y']].to_numpy()
    score = frames_dict[cname][['score']].to_numpy()[..., 0]
    all_p2ds.append(p2d)
    scores.append(score)
all_p2ds = np.array(all_p2ds)
scores = np.array(scores)

# handle output from new tracking pipeline
if np.nanmax(scores) > 1.1:
    scores = scores / np.nanpercentile(scores, 80)

from aniposelib.utils import select_matrices, mean_transform, mean_transform_robust

n_cams = len(cam_names)

def get_rtvec(M):
    rvec = torch.tensor(cv2.Rodrigues(M[:3, :3].detach().cpu().numpy())[0],
                        dtype=M.dtype, device=M.device).ravel()
    tvec = M[:3, 3].ravel()
    return rvec, tvec

K = np.array([
    [527.2260, 0, 316.4491],
    [0, 525.9586, 252.6574],
    [0, 0, 1]
])

dist = np.array([-0.3437, 0.1788, 0, 0, -0.0527])



cams = []
for i in range(n_cams):
    matrix = intrinsic[0, i]

    # rvecs = []
    # tvecs = []
    # for extrinsic in all_extrinsics:
    #     rvec, tvec = get_rtvec(extrinsic[i])
    #     rvecs.append(rvec)
    #     tvecs.append(tvec)
    # rvec = torch.median(torch.stack(rvecs), dim=0).values
    # tvec = torch.median(torch.stack(tvecs), dim=0).values

    L = all_extrinsics[:, i].detach().cpu().numpy()
    L_best = select_matrices(L)
    M_mean = mean_transform(L_best)
    M_mean = mean_transform_robust(L, M_mean, error=0.5)
    rvec, tvec = get_rtvec(torch.as_tensor(M_mean))

    cam = Camera(matrix=K, rvec=rvec, tvec=tvec, size=[640, 512], dist=dist,
                 name=cam_names[i])
    cams.append(cam)

# cgroup = CameraGroup.load('calibration_bundle.toml').to('cuda:0')
cgroup = CameraGroup(cams).to('cuda:0')
cgroup.dump(calib_fname_init)

p2d_cuda = torch.as_tensor(all_p2ds, device='cuda:0')
scores_cuda = torch.as_tensor(scores, device='cuda:0')
# p3d = cgroup.triangulate(p2d_cuda, progress=True)
p2d_cuda[scores_cuda < 0.95] = torch.nan

count = torch.sum(torch.isfinite(p2d_cuda[:, :, 0]), axis=0)
p2d_sub = p2d_cuda[:, count >= 2]

print(p2d_sub.shape)

err = cgroup.average_error(p2d_sub[:, :500])
print(err.item())

cgroup.bundle_adjust_iter(p2d_sub, verbose=True, only_extrinsics=True,
                          n_samp_iter=500, n_iters=10, n_samp_full=1000)

# cgroup.bundle_adjust_iter(p2d_sub, verbose=True, only_extrinsics=False,
#                           n_samp_iter=500, n_iters=4, n_samp_full=1000)

# cgroup.bundle_adjust_iter(p2d_sub, verbose=True, only_extrinsics=False)
cgroup.dump(calib_fname_out)

p2d_tri = torch.as_tensor(all_p2ds, device='cuda:0')
p2d_tri[scores_cuda < 0.90] = torch.nan

with torch.no_grad():
    p3d = cgroup.triangulate(p2d_tri, progress=True)

p3d_numpy = p3d.detach().cpu().numpy()
err = cgroup.reprojection_error(p3d, p2d_cuda, mean=True).detach().cpu().numpy()

start_time = str(stamps['timestamp'][0])
time_offset = (stamps['timestamp'] - stamps['timestamp'][0]) / pd.Timedelta(seconds=1.0)

np.savez_compressed(points_fname_out, p3d=p3d_numpy, err=err,
                    p2d=all_p2ds, scores=scores, start_time=start_time,
                    time_offset=time_offset, count=count.detach().cpu().numpy())

