#!/usr/bin/env python3

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate cameras using VGGT and SLEAP tracking data.")
    parser.add_argument("--root", default="/groups/voigts/voigtslab/outdoor/2026_04_10_mouse_left/data",
                        help="Directory containing source videos")
    parser.add_argument("--tracked-root", default="/groups/karashchuk/karashchuklab/outdoor_analysis/2026_04_10_mouse_left",
                        help="Directory containing tracking parquet files and calibration output")
    parser.add_argument("--tempdir", default="tempframes",
                        help="Directory for temporary extracted frames")
    parser.add_argument("--max-templates", type=int, default=8,
                        help="Maximum number of random time templates to use for VGGT")
    parser.add_argument("--camera-prefix", default="video_10",
                        help="Reference camera prefix used to discover time templates")
    return parser.parse_args()


def extract_first_frames(video_paths, output_dir=None):
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="video_frames_")
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_paths = []

    for video_path in video_paths:
        video_path = Path(video_path)
        output_filename = f"{video_path.stem}_frame.jpg"
        output_path = Path(output_dir) / output_filename

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            "-y",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            output_paths.append(str(output_path))
        except subprocess.CalledProcessError:
            output_paths.append(None)

    return output_paths


def main():
    args = parse_args()

    calib_fname_init = os.path.join(args.tracked_root, "calibration_vggt_init.toml")
    calib_fname_out = os.path.join(args.tracked_root, "calibration_adjusted.toml")
    points_fname_out = os.path.join(args.tracked_root, "points_3d.npz")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    possible = sorted(glob(os.path.join(args.root, f"{args.camera_prefix}_*.avi")))
    templates = [p.replace(args.camera_prefix, "video_*") for p in possible]

    if len(templates) > args.max_templates:
        templates = random.sample(templates, args.max_templates)

    all_extrinsics = []
    all_intrinsics = []
    for template in templates:
        print(template)

        vidnames = sorted(glob(template))
        print(len(vidnames))

        frames = extract_first_frames(vidnames, output_dir=args.tempdir)

        images = load_and_preprocess_images(frames).to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                images = images[None]
                aggregated_tokens_list, ps_idx = model.aggregator(images)

            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, [512, 640])
            all_extrinsics.append(extrinsic[0])
            all_intrinsics.append(intrinsic[0])

    all_extrinsics = torch.stack(all_extrinsics)
    all_intrinsics = torch.stack(all_intrinsics)

    datas = defaultdict(list)
    fnames = glob(os.path.join(args.tracked_root, "*.pq"))
    print("loading 2d points")
    for fname in tqdm(fnames, ncols=70):
        cname = os.path.basename(fname).split("_")[1]
        df = pd.read_parquet(fname)
        df["video"] = fname.replace(".pq", "")
        df["cam"] = cname
        df["framenum"] = np.arange(len(df))
        datas[cname].append(df)

    cam_names = sorted(list(datas.keys()))

    datas_combined = {k: pd.concat(v, ignore_index=True) for k, v in datas.items()}

    min_dt = min(d["timestamp"].min() for d in datas_combined.values())
    max_dt = max(d["timestamp"].max() for d in datas_combined.values())
    delta = pd.Timedelta(seconds=1 / 30.0)

    stamps = pd.DataFrame({"timestamp": pd.date_range(min_dt, max_dt, freq=delta)})

    frames_dict = defaultdict(list)
    for cname in datas_combined.keys():
        d = datas_combined[cname].sort_values("timestamp").reset_index()
        frames_dict[cname] = pd.merge_asof(
            stamps, d, on="timestamp", direction="nearest", tolerance=pd.Timedelta(seconds=1 / 15.0)
        )

    all_p2ds = []
    scores = []
    for cname in cam_names:
        p2d = frames_dict[cname][["x", "y"]].to_numpy()
        score = frames_dict[cname][["score"]].to_numpy()[..., 0]
        all_p2ds.append(p2d)
        scores.append(score)
    all_p2ds = np.array(all_p2ds)
    scores = np.array(scores)

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
        [0, 0, 1],
    ])
    dist = np.array([-0.3437, 0.1788, 0, 0, -0.0527])

    cams = []
    for i in range(n_cams):
        L = all_extrinsics[:, i].detach().cpu().numpy()
        L_best = select_matrices(L)
        M_mean = mean_transform(L_best)
        M_mean = mean_transform_robust(L, M_mean, error=0.5)
        rvec, tvec = get_rtvec(torch.as_tensor(M_mean))

        cam = Camera(matrix=K, rvec=rvec, tvec=tvec, size=[640, 512], dist=dist,
                     name=cam_names[i])
        cams.append(cam)

    cgroup = CameraGroup(cams).to("cuda:0")
    cgroup.dump(calib_fname_init)

    p2d_cuda = torch.as_tensor(all_p2ds, device="cuda:0")
    scores_cuda = torch.as_tensor(scores, device="cuda:0")
    p2d_cuda[scores_cuda < 0.95] = torch.nan

    count = torch.sum(torch.isfinite(p2d_cuda[:, :, 0]), axis=0)
    p2d_sub = p2d_cuda[:, count >= 2]

    print(p2d_sub.shape)

    err = cgroup.average_error(p2d_sub[:, :500])
    print(err.item())

    cgroup.bundle_adjust_iter(p2d_sub, verbose=True, only_extrinsics=True,
                              n_samp_iter=500, n_iters=10, n_samp_full=1000)

    cgroup.dump(calib_fname_out)

    p2d_tri = torch.as_tensor(all_p2ds, device="cuda:0")
    p2d_tri[scores_cuda < 0.90] = torch.nan

    with torch.no_grad():
        p3d = cgroup.triangulate(p2d_tri, progress=True)

    p3d_numpy = p3d.detach().cpu().numpy()
    err = cgroup.reprojection_error(p3d, p2d_cuda, mean=True).detach().cpu().numpy()

    start_time = str(stamps["timestamp"][0])
    time_offset = (stamps["timestamp"] - stamps["timestamp"][0]) / pd.Timedelta(seconds=1.0)

    np.savez_compressed(points_fname_out, p3d=p3d_numpy, err=err,
                        p2d=all_p2ds, scores=scores, start_time=start_time,
                        time_offset=time_offset, count=count.detach().cpu().numpy())


if __name__ == "__main__":
    main()
