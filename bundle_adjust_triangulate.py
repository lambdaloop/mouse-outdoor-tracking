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
    parser = argparse.ArgumentParser(description="Refine the VGGT calibration with bundle adjustment, then triangulate.")
    parser.add_argument("--root", default="/groups/voigts/voigtslab/outdoor/2026_04_10_mouse_left/data",
                        help="Directory containing source videos")
    parser.add_argument("--tracked-root", default="/groups/karashchuk/karashchuklab/outdoor_analysis/2026_04_10_mouse_left",
                        help="Directory containing tracking parquet files and calibration output")
    return parser.parse_args()


def main():
    args = parse_args()

    calib_fname_init = os.path.join(args.tracked_root, "calibration_vggt_init.toml")
    calib_fname_out = os.path.join(args.tracked_root, "calibration_adjusted.toml")
    points_fname_out = os.path.join(args.tracked_root, "points_3d.npz")

    if not os.path.exists(calib_fname_init):
        print("Need the following initial calibration file from VGGT:")
        print(calib_fname_init)
        print("exiting...")
        return

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

    # if np.nanmax(scores) > 1.1:
    #     scores = scores / np.nanpercentile(scores, 80)

    cgroup = CameraGroup.load(calib_fname_init).to("cuda:0")

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
