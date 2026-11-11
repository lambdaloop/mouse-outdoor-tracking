#!/usr/bin/env python3

import argparse
from glob import glob
import os

import torch

from aniposelib import CameraGroup

import numpy as np

from tqdm import tqdm
from collections import defaultdict
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Refine the VGGT calibration with bundle adjustment, then triangulate.")
    parser.add_argument("--tracked",
                        help="Directory containing tracking parquet files and calibration output")
    return parser.parse_args()


def main():
    args = parse_args()

    calib_fname_init = os.path.join(args.tracked, "calibration_vggt_init.toml")
    calib_fname_out = os.path.join(args.tracked, "calibration_adjusted.toml")
    points_fname_out = os.path.join(args.tracked, "points_3d.npz")

    if not os.path.exists(calib_fname_init):
        print("Need the following initial calibration file from VGGT:")
        print(calib_fname_init)
        print("exiting...")
        return

    datas = defaultdict(list)
    fnames = glob(os.path.join(args.tracked, "*.pq"))
    print("loading 2d points")
    for fname in tqdm(fnames, ncols=70):
        cname = os.path.basename(fname).split("_")[1]
        df = pd.read_parquet(fname)
        # Normalize legacy datetime columns and string timestamps to one timezone.
        # errors='coerce' safely identifies malformed rows from truncated timestamp files.
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce", utc=True
        )
        bad_timestamps = df["timestamp"].isna()
        if bad_timestamps.any():
            print(
                f"WARNING: dropping {int(bad_timestamps.sum())} rows with invalid "
                f"timestamps from {fname}"
            )
            df = df.loc[~bad_timestamps].copy()
        if df.empty:
            print(f"WARNING: skipping {fname}; no valid timestamps")
            continue
        df["video"] = fname.replace(".pq", "")
        df["cam"] = cname
        df["framenum"] = np.arange(len(df))
        datas[cname].append(df)

    # numeric camera order, matching CameraGroup.load (cam_2 before cam_10)
    cam_names = sorted(datas.keys(), key=lambda s: (len(s), s))

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

    cgroup = CameraGroup.load(calib_fname_init).to("cuda:0" if torch.cuda.is_available() else "cpu")

    # p2d_cuda = torch.as_tensor(all_p2ds, device="cuda:0")
    # scores_cuda = torch.as_tensor(scores, device="cuda:0")
    # p2d_cuda[scores_cuda < 0.95] = torch.nan

    count = np.sum(np.all(np.isfinite(all_p2ds), axis=-1), axis=0)
    p2d_sub = all_p2ds[:, count >= 2].copy()
    scores_sub = scores[:, count >= 2]
    # p2d_sub[scores_sub < 0.90] = np.nan

    print(p2d_sub.shape)

    err = cgroup.average_error(p2d_sub[:, :500])
    print(err.item())

    # aniposelib >= 1.0: Levenberg-Marquardt robust bundle adjustment by default (the old
    # Adam options n_samp_iter / n_samp_full are gone; method="adam" restores the 0.9 code path)
    cgroup.bundle_adjust_iter(p2d_sub, verbose=True, only_extrinsics=True, n_iters=16)

    cgroup.dump(calib_fname_out)

    # p2d_tri = all_p2ds.copy()
    # p2d_tri[scores < 0.90] = torch.nan

    with torch.no_grad():
        # p3d, _, _, _ = cgroup.triangulate_ransac(p2d_sub, progress=True)
        p3d = cgroup.triangulate(p2d_sub, progress=True)

    # numpy in -> numpy out
    err = cgroup.reprojection_error(p3d, p2d_sub, mean=True)

    p3d_out = np.full((all_p2ds.shape[1], 3), np.nan)
    p3d_out[count >= 2] = p3d

    err_out = np.full((all_p2ds.shape[1],), np.nan)
    err_out[count >= 2] = err
    
    start_time = str(stamps["timestamp"][0])
    time_offset = (stamps["timestamp"] - stamps["timestamp"][0]) / pd.Timedelta(seconds=1.0)

    np.savez_compressed(points_fname_out, p3d=p3d_out, err=err_out,
                        p2d=all_p2ds, scores=scores, start_time=start_time,
                        time_offset=time_offset, count=count)


if __name__ == "__main__":
    main()
