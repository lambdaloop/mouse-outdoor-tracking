#!/usr/bin/env python3
"""Robust calibration refinement + robust triangulation, on aniposelib-pytorch (>= 1.1).

Drop-in replacement for bundle_adjust_triangulate.py: same input (--tracked directory with
per-camera tracking parquet files and calibration_vggt_init.toml) and same outputs
(calibration_adjusted.toml, points_3d.npz with the same keys, plus some extra QC keys).

Steps
  1. Load tracking and align all cameras on a common time grid (default 25 Hz = camera rate,
     nearest frame within 20 ms, so no frame is used twice).
  2. Filter detections: blobs at the thermal sync block (track_point in CAM_CONFIGS of
     track_mouse_simple_gpu.py) and blobs outside the tracker's area range.
  3. Calibrate with aniposelib.calibrate_from_points (W2), starting from the VGGT init (used as
     the "reference" rig; SfM is skipped when it already explains the data): robust bundle
     adjustment with shrinking MSAC thresholds, leave-one-camera-out PnP resection, held-out-LOO
     gated refinement of focal length and k1, and removal of time windows with desynchronised
     cameras (times= is passed).  Cameras are loaded in numeric order (the aniposelib 1.x default).
  4. Robust triangulation with aniposelib.reconstruct_3d (W6): MSAC over camera pairs with a fixed
     inlier threshold (--tau px), no gap filling.
  5. QC flags from reconstruct_3d: number of agreeing cameras, static-artifact voxels, distance to a
     ground height-map fitted to the >=3-camera points. `p3d` keeps only points passing all of
     these ("good"); `p3d_all` keeps every point with >=2 agreeing cameras.

Output keys in points_3d.npz
  (same as before) p3d [N,3], err [N] (mean reprojection error over agreeing cameras, px),
      p2d [C,N,2], scores [C,N], start_time, time_offset [N] (s), count [N] (# cameras detecting)
  (extra) p3d_all, good, n_inliers, inliers [C,N], err_cams [C,N], static, surface_resid,
      p2d_kept [C,N] (detection filter), cam_names, loo (per-camera leave-one-out agreement at 8 px)
Extra file: calibration_report.json (verdict and report of calibrate_from_points / evaluate_calibration).
"""
import argparse
import json
import os
import time
import warnings
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd

import aniposelib
from aniposelib import CameraGroup
from aniposelib.cameras import CameraOrderWarning

_T0 = time.time()


def log(*args):
    """Print with elapsed time, flushed immediately (so cluster logs show progress live)."""
    el = time.time() - _T0
    print(f"[{int(el // 60):3d}m{int(el % 60):02d}s]", *args, flush=True)


# ============================================================================ data loading
def load_tracks(tracked):
    datas = defaultdict(list)
    fnames = sorted(glob(os.path.join(tracked, "*.pq")))
    log(f"reading {len(fnames)} tracking files")
    for i, fname in enumerate(fnames):
        if (i + 1) % 100 == 0 or i + 1 == len(fnames):
            log(f"  read {i + 1}/{len(fnames)} files")
        cname = os.path.basename(fname).split("_")[1]
        df = pd.read_parquet(fname, columns=["timestamp", "x", "y", "score"])
        # Normalize legacy datetime columns and string timestamps to one timezone;
        # errors='coerce' drops malformed rows from truncated timestamp files.
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        bad = df["timestamp"].isna()
        if bad.any():
            log(f"WARNING: dropping {int(bad.sum())} rows with invalid timestamps from {fname}")
            df = df.loc[~bad]
        if df.empty:
            log(f"WARNING: skipping {fname}; no valid timestamps")
            continue
        datas[cname].append(df)
    cam_names = sorted(datas.keys(), key=lambda s: (len(s), s))
    tracks = {k: pd.concat(v, ignore_index=True).sort_values("timestamp") for k, v in datas.items()}
    return cam_names, tracks


def align_tracks(cam_names, tracks, hz=25.0, tol=0.02):
    t0 = min(d["timestamp"].min() for d in tracks.values())
    t1 = max(d["timestamp"].max() for d in tracks.values())
    stamps = pd.DataFrame({"timestamp": pd.date_range(t0, t1, freq=pd.Timedelta(seconds=1 / hz))})
    p2d, scores = [], []
    for c in cam_names:
        log(f"  aligning camera {c}")
        m = pd.merge_asof(stamps, tracks[c][["timestamp", "x", "y", "score"]], on="timestamp",
                          direction="nearest", tolerance=pd.Timedelta(seconds=tol))
        p2d.append(m[["x", "y"]].to_numpy(dtype=np.float64))
        scores.append(m["score"].to_numpy(dtype=np.float64))
    time_offset = ((stamps["timestamp"] - stamps["timestamp"].iloc[0]) / pd.Timedelta(seconds=1.0)).to_numpy()
    return np.array(p2d), np.array(scores), stamps["timestamp"], time_offset


def detection_filter(p2d, scores, cam_names, arena, sync_rad=20.0, min_area=4, max_area=1200):
    """Keep mask [C,N]: drops detections near the sync block and outside the area range."""
    try:
        from track_mouse_simple_gpu import CAM_CONFIGS
        cfgs = CAM_CONFIGS.get(arena, {})
    except Exception as e:  # pragma: no cover
        log(f"WARNING: could not import CAM_CONFIGS ({e}); no sync-block filtering")
        cfgs = {}
    if not cfgs:
        log(f"WARNING: no camera configs for arena '{arena}'; no sync-block filtering")
    keep = np.isfinite(p2d[..., 0])
    for c, name in enumerate(cam_names):
        n0 = int(keep[c].sum())
        tp = cfgs.get(int(name) - 9, {}).get("track_point")  # same cam_id mapping as the tracker
        near = np.zeros(p2d.shape[1], bool)
        if tp is not None:
            near = np.linalg.norm(p2d[c] - np.array(tp, dtype=np.float64), axis=1) < sync_rad
        with np.errstate(invalid="ignore"):
            bad_area = (scores[c] < min_area) | (scores[c] >= max_area)
        keep[c] &= ~near & ~bad_area
        log(f"  cam {name}: {n0} detections, removed {int((near & np.isfinite(p2d[c, :, 0])).sum())} "
              f"at sync block, {n0 - int(keep[c].sum())} total")
    return keep


# ============================================================================ main
def parse_args():
    parser = argparse.ArgumentParser(description="Robust calibration refinement and robust triangulation.")
    parser.add_argument("--tracked", help="Directory containing tracking parquet files and calibration output")
    parser.add_argument("--arena", default="right-2026", help="Arena config (sync-block positions)")
    parser.add_argument("--hz", type=float, default=25.0, help="Output time grid rate (cameras run at 25 Hz)")
    parser.add_argument("--tol", type=float, default=0.02, help="Max time difference (s) to match a frame")
    parser.add_argument("--tau", type=float, default=10.0, help="Inlier threshold (px) for triangulation")
    parser.add_argument("--calib", default=None, help="Starting calibration (default: calibration_vggt_init.toml)")
    parser.add_argument("--max-calib-points", type=int, default=500_000,
                        help="calibrate on at most this many (random, seeded, time-sorted) timepoints; all timepoints "
                             "are still triangulated. Bounds GPU memory (the old script used the same 500k cap)")
    parser.add_argument("--init", default="reference", choices=["reference", "auto"],
                        help="'reference' (default): bundle adjustment starts from the given calibration and keeps its "
                             "coordinate frame and scale, as before; 'auto': aniposelib also tries SfM and picks the best "
                             "candidate (arbitrary gauge/scale)")
    parser.add_argument("--device", default=None, help="torch device for calibration/triangulation (default: cuda if available)")
    return parser.parse_args()


def jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def main():
    args = parse_args()
    t_start = time.time()
    device = args.device or "auto"
    log(f"robust_triangulation: tracked={args.tracked} arena={args.arena} device={device} "
        f"aniposelib {aniposelib.__version__}")

    calib_fname_init = args.calib or os.path.join(args.tracked, "calibration_vggt_init.toml")
    calib_fname_out = os.path.join(args.tracked, "calibration_adjusted.toml")
    points_fname_out = os.path.join(args.tracked, "points_3d.npz")
    report_fname_out = os.path.join(args.tracked, "calibration_report.json")

    if not os.path.exists(calib_fname_init):
        log("Need the following initial calibration file from VGGT:")
        log(calib_fname_init)
        log("exiting...")
        return

    log("== step 1/5: loading 2d points")
    cam_names, tracks = load_tracks(args.tracked)
    p2d, scores, stamps, time_offset = align_tracks(cam_names, tracks, hz=args.hz, tol=args.tol)
    count = np.isfinite(p2d[..., 0]).sum(0)
    log(f"cameras {cam_names}; {p2d.shape[1]} timepoints at {args.hz:g} Hz; "
        f"{int((count >= 2).sum())} with >= 2 detections")

    with warnings.catch_warnings():  # the numeric order is what we want; the files are written in it
        warnings.simplefilter("ignore", CameraOrderWarning)
        cgroup = CameraGroup.load(calib_fname_init)  # numeric camera order (cam_2 before cam_10)
    if cgroup.get_names() != cam_names:
        raise ValueError(f"calibration cameras {cgroup.get_names()} do not match tracking cameras {cam_names}")

    log("== step 2/5: filtering detections")
    keep = detection_filter(p2d, scores, cam_names, args.arena)
    p2d_f = np.where(keep[..., None], p2d, np.nan)
    sel = np.nonzero(np.isfinite(p2d_f[..., 0]).sum(0) >= 2)[0]
    P, t_sel = p2d_f[:, sel], time_offset[sel]
    if len(sel) > args.max_calib_points:
        sub = np.sort(np.random.default_rng(0).choice(len(sel), args.max_calib_points, replace=False))
        P_cal, t_cal = P[:, sub], t_sel[sub]
        log(f"calibrating on {len(sub)} of {len(sel)} timepoints (--max-calib-points)")
    else:
        P_cal, t_cal = P, t_sel

    log("== step 3/5: calibrating (aniposelib.calibrate_from_points)")
    cal = aniposelib.calibrate_from_points(P_cal, cgroup, init=args.init, times=t_cal, device=device)
    log(f"calibration verdict: {cal.verdict}; {cal.reasons}; init {cal.selected_init}; "
        f"intrinsics {cal.intrinsics.get('decision')}: {cal.intrinsics.get('reason')}")
    if cal.cgroup is None:
        raise RuntimeError(f"calibration failed: {cal.reasons}")
    cams = cal.cgroup
    cams.dump(calib_fname_out)
    log("saved", calib_fname_out)

    ev = aniposelib.evaluate_calibration(cams, P_cal, times=t_cal, device=device)
    loo = np.array([ev.per_camera[n].get("loo_agree@8px", np.nan) for n in cam_names], dtype=float)
    log("leave-one-out agreement @8px:", " ".join(f"{n}:{v:.2f}" for n, v in zip(cam_names, loo)))
    with open(report_fname_out, "w") as f:
        json.dump(jsonable(dict(calibrate_from_points=cal.to_dict(), evaluate_calibration=ev.to_dict())), f, indent=1)

    log(f"== step 4/5: robust triangulation of {len(sel)} timepoints (tau {args.tau:g} px)")
    R = aniposelib.reconstruct_3d(
        cams, P, times=t_sel, fill="none", device=device,
        options=dict(tau_px=args.tau, static_flag=True, surface_flag=True))

    N, C = p2d.shape[1], len(cam_names)
    p3d_all = np.full((N, 3), np.nan); p3d_all[sel] = R.points3d
    n_inl = np.zeros(N, np.int16); n_inl[sel] = R.n_views
    inliers = np.zeros((C, N), bool); inliers[:, sel] = R.inliers
    err_cams = np.full((C, N), np.nan, np.float32); err_cams[:, sel] = R.errors
    static = np.zeros(N, bool); static[sel] = R.static
    surf = np.zeros(N, np.float32)
    if R.surface_residual is not None:
        surf[:] = np.inf; surf[sel] = R.surface_residual
    else:
        log("WARNING: too few >=3-camera points for a ground model; skipping the surface check")
    off_surface = np.zeros(N, bool)
    if R.off_surface is not None:
        off_surface[sel] = R.off_surface
    with np.errstate(invalid="ignore", divide="ignore"):
        err = np.where(n_inl > 0, np.where(inliers, err_cams, 0).sum(0) / np.maximum(n_inl, 1), np.nan)

    log("== step 5/5: quality flags and saving")
    good = (n_inl >= 2) & ~static & ~off_surface
    p3d = np.where(good[:, None], p3d_all, np.nan)
    log(f"triangulated {int((n_inl >= 2).sum())} timepoints (>=2 agreeing cameras), "
        f"{int((n_inl >= 3).sum())} with >=3; static artifacts {int(static.sum())}; good {int(good.sum())}")
    if good.any():
        log(f"reprojection error of good points: median {np.nanmedian(err[good]):.2f} px, "
            f"p90 {np.nanpercentile(err[good], 90):.2f} px")

    np.savez_compressed(
        points_fname_out,
        p3d=p3d, err=err, p2d=p2d, scores=scores, start_time=str(stamps.iloc[0]),
        time_offset=time_offset, count=count,
        p3d_all=p3d_all, good=good, n_inliers=n_inl, inliers=inliers, err_cams=err_cams,
        static=static, surface_resid=surf.astype(np.float32), p2d_kept=keep,
        cam_names=np.array(cam_names), loo=loo, hz=args.hz, tau=args.tau,
    )
    log("saved", points_fname_out, f"({time.time() - t_start:.0f}s total)")


if __name__ == "__main__":
    main()
