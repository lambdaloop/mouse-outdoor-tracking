#!/usr/bin/env python3

import argparse
import subprocess
import tempfile
from pathlib import Path
from glob import glob
import os
import random

import torch

from aniposelib import CameraGroup
from aniposelib.init import vggt

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
    calib_fname_out = os.path.join(args.tracked, "calibration_adjusted.toml")
    points_fname_out = os.path.join(args.tracked, "points_3d.npz")

    possible = sorted(glob(os.path.join(args.source, f"{args.camera_prefix}_*.avi")))
    templates = [p.replace(args.camera_prefix, "video_*") for p in possible]

    if len(templates) > args.max_templates:
        templates = random.sample(templates, args.max_templates)

    frame_sets = []
    for template in templates:
        print(template)

        vidnames = sorted(glob(template))
        print(len(vidnames))

        frames = extract_first_frames(vidnames, output_dir=args.tempdir)
        frame_sets.append(frames)

    datas = defaultdict(list)
    fnames = glob(os.path.join(args.tracked, "*.pq"))
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

    # if np.nanmax(scores) > 1.1:
    #     scores = scores / np.nanpercentile(scores, 80)

    # VGGT inference (mode="crop" = vggt's own preprocessing), Sim(3)-aligned robust average
    # over the templates; the intrinsics are the lab's calibration of the thermal cameras.
    pred = vggt.predict(frame_sets, names=cam_names, mode="crop")
    agg = vggt.aggregate(pred)

    K = np.array([
        [527.2260, 0, 316.4491],
        [0, 525.9586, 252.6574],
        [0, 0, 1],
    ])
    dist = np.array([-0.3437, 0.1788, 0, 0, -0.0527])

    cgroup = vggt.to_cameragroup(agg, K=K, dist=dist, sizes=[640, 512])
    cgroup.dump(calib_fname_init)
    cgroup = cgroup.to("cuda:0" if torch.cuda.is_available() else "cpu")

    # numpy in -> numpy out
    p2d = all_p2ds.copy()
    p2d[scores < 0.95] = np.nan

    count = np.sum(np.isfinite(p2d[:, :, 0]), axis=0)
    p2d_sub = p2d[:, count >= 2]

    print(p2d_sub.shape)

    err = cgroup.average_error(p2d_sub[:, :500])
    print(float(err))

    # aniposelib >= 1.0: LM robust bundle adjustment (the Adam options n_samp_iter / n_samp_full
    # are gone; method="adam" restores the 0.9 code path)
    cgroup.bundle_adjust_iter(p2d_sub, verbose=True, only_extrinsics=True, n_iters=10)

    cgroup.dump(calib_fname_out)

    p2d_tri = all_p2ds.copy()
    p2d_tri[scores < 0.90] = np.nan

    p3d = cgroup.triangulate(p2d_tri, progress=True)
    err = cgroup.reprojection_error(p3d, p2d, mean=True)

    start_time = str(stamps["timestamp"][0])
    time_offset = (stamps["timestamp"] - stamps["timestamp"][0]) / pd.Timedelta(seconds=1.0)

    np.savez_compressed(points_fname_out, p3d=p3d, err=err,
                        p2d=all_p2ds, scores=scores, start_time=start_time,
                        time_offset=time_offset, count=count)


if __name__ == "__main__":
    main()
