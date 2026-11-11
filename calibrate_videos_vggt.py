#!/usr/bin/env python3

import argparse
import subprocess
import tempfile
from pathlib import Path
from glob import glob
import os
import random
from collections import Counter

import numpy as np

from aniposelib.init import vggt

# Lab calibration of the thermal cameras (intrinsics are not estimated by VGGT here)
K = np.array([
    [527.2260, 0, 316.4491],
    [0, 525.9586, 252.6574],
    [0, 0, 1],
])
dist = np.array([-0.3437, 0.1788, 0, 0, -0.0527])


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
        
    possible = sorted(glob(os.path.join(args.source, f"{args.camera_prefix}_*.avi")))
    if not possible:
        raise FileNotFoundError(
            f"No videos matching '{args.camera_prefix}_*.avi' found in {args.source}"
        )

    template_camera_sets = []
    for path in possible:
        template = path.replace(args.camera_prefix, "video_*")
        vidnames = sorted(glob(template))
        camera_names = tuple(Path(v).name.split("_")[1] for v in vidnames)
        if camera_names:
            template_camera_sets.append((template, camera_names))

    # Sample only synchronized timestamps containing the same complete camera set.
    camera_set_counts = Counter(cameras for _, cameras in template_camera_sets)
    cam_names = max(
        camera_set_counts,
        key=lambda cameras: (len(cameras), camera_set_counts[cameras], cameras),
    )
    templates = [
        template for template, cameras in template_camera_sets if cameras == cam_names
    ]
    if args.max_templates < 2:
        raise ValueError("--max-templates must be at least 2 for camera pose selection")
    if len(templates) < 2:
        raise ValueError(
            "VGGT calibration requires at least two timestamps containing the same "
            f"complete camera set; found {len(templates)} for cameras {cam_names}"
        )
    if len(templates) > args.max_templates:
        templates = random.sample(templates, args.max_templates)

    n_cams = len(cam_names)
    print(f"detected cams: {n_cams}; using {len(templates)} complete timestamps")

    frame_sets = []
    for template in templates:
        print(template)

        vidnames = sorted(glob(template))
        current_cameras = tuple(Path(v).name.split("_")[1] for v in vidnames)
        if current_cameras != cam_names:
            raise RuntimeError(f"Camera set changed while processing {template}")

        frames = extract_first_frames(vidnames, output_dir=args.tempdir)
        if any(frame is None for frame in frames):
            print(f"  Skipping {template}: failed to extract at least one camera frame")
            continue
        frame_sets.append(frames)

    if len(frame_sets) < 2:
        raise RuntimeError(
            "VGGT calibration needs at least two successfully processed complete "
            f"timestamps; only {len(frame_sets)} were usable"
        )

    # VGGT inference (mode="crop" is the network input of vggt's load_and_preprocess_images),
    # then Sim(3) alignment of the per-template rigs and robust averaging.
    pred = vggt.predict(frame_sets, names=cam_names, mode="crop")
    agg = vggt.aggregate(pred)
    print("VGGT template spread: raw rig scale rel. spread "
          f"{agg.raw_scale_rel_spread:.3f}, median rotation dispersion "
          f"{np.median(agg.stats['rot_disp_deg']):.2f} deg")

    # VGGT only provides the extrinsics here (an init for bundle adjustment); the intrinsics
    # are the lab's calibration of the thermal cameras.
    cgroup = vggt.to_cameragroup(agg, K=K, dist=dist, sizes=[640, 512])
    cgroup.dump(calib_fname_init)


if __name__ == "__main__":
    main()
