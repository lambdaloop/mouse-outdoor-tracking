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
from aniposelib.utils import select_matrices, mean_transform, mean_transform_robust
import cv2

import numpy as np

from tqdm import tqdm
from collections import defaultdict
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate cameras using VGGT and SLEAP tracking data.")
    parser.add_argument("--source",
                        help="Directory containing source videos")
    parser.add_argument("--tracked",
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

    calib_fname_init = os.path.join(args.tracked, "calibration_vggt_init.toml")
    print(calib_fname_init)

    if os.path.exists(calib_fname_init):
        print("output file exists, exiting...")
        return
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    possible = sorted(glob(os.path.join(args.source, f"{args.camera_prefix}_*.avi")))
    templates = [p.replace(args.camera_prefix, "video_*") for p in possible]

    if len(templates) > args.max_templates:
        templates = random.sample(templates, args.max_templates)

    n_cams = max([len(glob(template)) for template in templates])
    print("detected cams:", n_cams)
    
    all_extrinsics = []
    all_intrinsics = []
    for template in templates:
        print(template)

        vidnames = sorted(glob(template))
        print(len(vidnames))
        if len(vidnames) < n_cams:
            continue

        cam_names = [v.split('_')[1] for v in vidnames]

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


if __name__ == "__main__":
    main()
