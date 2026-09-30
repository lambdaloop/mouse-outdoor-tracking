#!/usr/bin/env python3
"""Robust calibration refinement + robust triangulation.

Drop-in replacement for bundle_adjust_triangulate.py: same input (--tracked directory with
per-camera tracking parquet files and calibration_vggt_init.toml) and same outputs
(calibration_adjusted.toml, points_3d.npz with the same keys, plus some extra QC keys).

Steps
  1. Load tracking and align all cameras on a common time grid (default 25 Hz = camera rate,
     nearest frame within 20 ms, so no frame is used twice).
  2. Filter detections: blobs at the thermal sync block (track_point in CAM_CONFIGS of
     track_mouse_simple_gpu.py) and blobs outside the tracker's area range.
  3. Calibrate, starting from the VGGT init. Cameras are loaded in numeric order: aniposelib's
     CameraGroup.load sorts keys as strings (cam_10 before cam_2), which shuffles >10 cameras.
       - robust bundle adjustment (Schur-complement Levenberg-Marquardt, Cauchy loss) on the
         multi-view inliers, with the inlier threshold tau shrinking 200 -> 5 px
       - leave-one-camera-out PnP-RANSAC resection between rounds (repairs cameras stuck in a
         bad local minimum)
       - final round also refines per-camera focal length and k1, k2
  4. Robust triangulation: for every timepoint, every pair of cameras that both detect the
     animal is a hypothesis; each is scored by its capped reprojection error in all observing
     cameras (MSAC); the best one's inliers are refit and refined by Gauss-Newton on the pixel
     error. Fully vectorized on GPU.
  5. QC flags: number of agreeing cameras, static-artifact voxels, distance to a ground
     height-map fitted to the >=3-camera points. `p3d` keeps only points passing all of these
     ("good"); `p3d_all` keeps every point with >=2 agreeing cameras.

Output keys in points_3d.npz
  (same as before) p3d [N,3], err [N] (mean reprojection error over agreeing cameras, px),
      p2d [C,N,2], scores [C,N], start_time, time_offset [N] (s), count [N] (# cameras detecting)
  (extra) p3d_all, good, n_inliers, inliers [C,N], err_cams [C,N], static, surface_resid,
      p2d_kept [C,N] (detection filter), cam_names, loo (per-camera leave-one-out agreement)
"""
import argparse
import itertools
import os
import time
from collections import defaultdict
from glob import glob

import cv2
import numpy as np
import pandas as pd
import toml
import torch
from torch.func import jvp

DT = torch.float64
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


# ============================================================================ cameras
def load_cams(fname, device):
    """Load an aniposelib toml in NUMERIC camera order (cam_0, cam_1, ..., cam_10, cam_11)."""
    d = toml.load(fname)
    keys = sorted([k for k in d if k.startswith("cam_")], key=lambda s: int(s.split("_")[1]))
    t = lambda x: torch.as_tensor(np.array(x, dtype=np.float64), dtype=DT, device=device)
    return dict(
        rvec=t([np.ravel(d[k]["rotation"]) for k in keys]),
        tvec=t([np.ravel(d[k]["translation"]) for k in keys]),
        K=t([d[k]["matrix"] for k in keys]),
        dist=t([np.pad(np.ravel(d[k]["distortions"]), (0, 5))[:5] for k in keys]),
        size=[list(d[k]["size"]) for k in keys],
    )


def save_cams(cams, fname, names):
    out = {}
    for i in range(cams["rvec"].shape[0]):
        out[f"cam_{i}"] = dict(
            name=str(names[i]),
            size=[int(x) for x in cams["size"][i]],
            matrix=cams["K"][i].detach().cpu().numpy().tolist(),
            distortions=cams["dist"][i].detach().cpu().numpy().tolist(),
            rotation=cams["rvec"][i].detach().cpu().numpy().tolist(),
            translation=cams["tvec"][i].detach().cpu().numpy().tolist(),
        )
    out["metadata"] = {"camera_order_note": "cam_<i> is camera index i in numeric order; "
                                            "do not load with a string-sorting loader"}
    with open(fname, "w") as f:
        toml.dump(out, f)


def rodrigues(rvec):
    """rvec [...,3] -> R [...,3,3] (differentiable)."""
    theta = torch.sqrt((rvec * rvec).sum(-1, keepdim=True) + 1e-30)
    k = rvec / theta
    Kx = torch.zeros(rvec.shape[:-1] + (3, 3), dtype=rvec.dtype, device=rvec.device)
    Kx[..., 0, 1] = -k[..., 2]; Kx[..., 0, 2] = k[..., 1]
    Kx[..., 1, 0] = k[..., 2];  Kx[..., 1, 2] = -k[..., 0]
    Kx[..., 2, 0] = -k[..., 1]; Kx[..., 2, 1] = k[..., 0]
    I = torch.eye(3, dtype=rvec.dtype, device=rvec.device).expand_as(Kx)
    return I + torch.sin(theta)[..., None] * Kx + (1 - torch.cos(theta))[..., None] * (Kx @ Kx)


def distort_normalized(x, y, k1, k2, p1, p2, k3):
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return xd, yd


def project(p3d, cams):
    """p3d [N,3] -> pixels [C,N,2], depth [C,N]. OpenCV model (k1,k2,p1,p2,k3)."""
    R = rodrigues(cams["rvec"])
    pc = torch.einsum("cij,nj->cni", R, p3d) + cams["tvec"][:, None, :]
    z = pc[..., 2]
    x, y = pc[..., 0] / z, pc[..., 1] / z
    dd = [cams["dist"][:, i, None] for i in range(5)]
    xd, yd = distort_normalized(x, y, *dd)
    K = cams["K"]
    u = K[:, None, 0, 0] * xd + K[:, None, 0, 1] * yd + K[:, None, 0, 2]
    v = K[:, None, 1, 1] * yd + K[:, None, 1, 2]
    return torch.stack([u, v], -1), z


def undistort(p2d, cams, n_iter=30):
    """Raw pixels [C,N,2] -> undistorted normalized coordinates [C,N,2]."""
    K = cams["K"]; dist = cams["dist"]
    fx, fy, cx, cy, sk = (K[:, None, 0, 0], K[:, None, 1, 1], K[:, None, 0, 2], K[:, None, 1, 2], K[:, None, 0, 1])
    y0 = (p2d[..., 1] - cy) / fy
    x0 = (p2d[..., 0] - cx - sk * y0) / fx
    k1, k2, p1, p2, k3 = [dist[:, i, None] for i in range(5)]
    x, y = x0.clone(), y0.clone()
    for _ in range(n_iter):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        x = (x0 - (2 * p1 * x * y + p2 * (r2 + 2 * x * x))) / rad
        y = (y0 - (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)) / rad
    return torch.stack([x, y], -1)


def reproj_err(p3d, p2d, cams):
    """[N,3], [C,N,2] -> pixel error [C,N] (inf if behind the camera)."""
    proj, z = project(p3d, cams)
    e = torch.linalg.norm(proj - p2d, dim=-1)
    return torch.where(z > 0, e, torch.full_like(e, float("inf")))


# ============================================================================ triangulation
def solve3(A, b):
    """Batched symmetric 3x3 solve via adjugate (elementwise; no cusolver batch limits)."""
    a00, a01, a02 = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    a11, a12, a22 = A[..., 1, 1], A[..., 1, 2], A[..., 2, 2]
    c00 = a11 * a22 - a12 * a12
    c01 = a02 * a12 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    c11 = a00 * a22 - a02 * a02
    c12 = a01 * a02 - a00 * a12
    c22 = a00 * a11 - a01 * a01
    det = a00 * c00 + a01 * c01 + a02 * c02
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return torch.stack([(c00 * b0 + c01 * b1 + c02 * b2) / det,
                        (c01 * b0 + c11 * b1 + c12 * b2) / det,
                        (c02 * b0 + c12 * b1 + c22 * b2) / det], -1)


def dlt_blocks(und, cams):
    """Per-camera DLT normal-matrix blocks [C,N,4,4]; the DLT of any camera subset is the
    sum of its blocks. Missing observations give zero blocks."""
    R = rodrigues(cams["rvec"])
    P = torch.cat([R, cams["tvec"][..., None]], -1)  # [C,3,4]
    valid = torch.isfinite(und[..., 0])
    u = torch.nan_to_num(und)
    ax = u[..., 0, None] * P[:, None, 2] - P[:, None, 0]
    ay = u[..., 1, None] * P[:, None, 2] - P[:, None, 1]
    M = ax[..., :, None] * ax[..., None, :] + ay[..., :, None] * ay[..., None, :]
    return M * valid[..., None, None]


def solve_blocks(M):
    """[...,4,4] -> 3D point: inhomogeneous least squares with X_h = [X, 1]."""
    return solve3(M[..., :3, :3], -M[..., :3, 3])


def refine_points_gn(X, p2d, cams, w, n_iter=2, damping=1e-6):
    """Weighted Gauss-Newton on pixel reprojection error, batched over points.
    Jacobian columns come from 3 forward-mode JVPs (points are independent)."""
    valid = torch.isfinite(p2d[..., 0]) & (w > 0)
    p2 = torch.nan_to_num(p2d)
    ww = torch.where(valid, w, torch.zeros_like(w))
    X = X.detach().clone()
    f = lambda x: project(x, cams)[0]
    eye = torch.eye(3, dtype=DT, device=X.device)
    for _ in range(n_iter):
        cols = []
        for k in range(3):
            r, jk = jvp(f, (X,), (eye[k].expand_as(X),))
            cols.append(jk)
        J = torch.stack(cols, -1)                       # [C,N,2,3]
        res = r - p2
        bad = ~(torch.isfinite(res).all(-1) & torch.isfinite(J).all(-1).all(-1))
        wk = torch.where(bad, torch.zeros_like(ww), ww)
        J = torch.nan_to_num(J) * wk[..., None, None].sqrt()
        res = torch.nan_to_num(res) * wk[..., None].sqrt()
        H = torch.einsum("cnki,cnkj->nij", J, J)
        g = torch.einsum("cnki,cnk->ni", J, res)
        H = H + (damping * H.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None] + 1e-12) * eye
        dx = solve3(H, g)
        X = torch.where(torch.isfinite(dx).all(-1)[:, None], X - dx, X)
    return X


def triangulate_robust(p2d, cams, tau=10.0, min_inliers=2, chunk=200_000, n_refine=2, progress=False):
    """Robust multi-view triangulation.

    p2d [C,N,2] raw pixels (NaN = missing). Every pair of cameras observing a point is a
    hypothesis (2-view DLT); score = sum over observing cameras of min(err^2, tau^2); the best
    hypothesis' inliers (err < tau) are refit, then refined with Gauss-Newton on the pixel error
    (inlier set re-evaluated after each refinement, never shrinking).

    Returns dict: p3d [N,3] (NaN if < min_inliers agree), inliers [C,N] bool, n_inl [N],
    err [C,N] pixel error of every observation w.r.t. the returned point.
    """
    device = cams["rvec"].device
    p2d = torch.as_tensor(p2d, dtype=DT, device=device)
    C, N, _ = p2d.shape
    pairs = list(itertools.combinations(range(C), 2))
    pa = torch.tensor([a for a, _ in pairs], device=device, dtype=torch.long)
    pb = torch.tensor([b for _, b in pairs], device=device, dtype=torch.long)
    out = torch.full((N, 3), float("nan"), dtype=DT, device=device)
    inl_out = torch.zeros((C, N), dtype=torch.bool, device=device)
    err_out = torch.full((C, N), float("nan"), dtype=DT, device=device)
    tau2 = tau * tau
    n_chunks = (N + chunk - 1) // chunk
    for ic, s0 in enumerate(range(0, N, chunk)):
        if progress:
            log(f"  triangulating chunk {ic + 1}/{n_chunks}")
        s1 = min(N, s0 + chunk)
        n = s1 - s0
        p2 = p2d[:, s0:s1]
        u = undistort(p2, cams)
        valid = torch.isfinite(p2[..., 0]) & torch.isfinite(u[..., 0])
        B = dlt_blocks(u, cams)
        si, ni = torch.nonzero(valid[pa] & valid[pb], as_tuple=True)
        if si.numel() == 0:
            continue
        Xh = solve_blocks(B[pa[si], ni] + B[pb[si], ni])
        eh = reproj_err(Xh, p2[:, ni], cams)
        vh = valid[:, ni]
        cost = torch.where(vh, torch.clamp(torch.nan_to_num(eh, nan=1e30, posinf=1e30) ** 2, max=tau2),
                           torch.zeros_like(eh)).sum(0)
        best_cost = torch.full((n,), float("inf"), dtype=DT, device=device).scatter_reduce(
            0, ni, cost, reduce="amin")
        is_best = cost <= best_cost[ni]
        H = len(ni)
        hid = torch.arange(H, device=device)
        first = torch.full((n,), H, dtype=torch.long, device=device).scatter_reduce(
            0, ni[is_best], hid[is_best], reduce="amin")
        pts = torch.nonzero(first < H)[:, 0]
        hsel = first[pts]
        inl = (eh[:, hsel] < tau) & vh[:, hsel]
        p2s = p2[:, pts]
        X = solve_blocks(torch.einsum("cn,cnij->nij", inl.to(DT), B[:, pts]))
        for _ in range(n_refine):
            X = refine_points_gn(X, p2s, cams, inl.to(DT), n_iter=2)
            inl2 = (reproj_err(X, p2s, cams) < tau) & valid[:, pts]
            inl = torch.where((inl2.sum(0) >= inl.sum(0))[None], inl2, inl)
        e = reproj_err(X, p2s, cams)
        good = inl.sum(0) >= min_inliers
        idx = s0 + pts
        out[idx] = torch.where(good[:, None], X, torch.full_like(X, float("nan")))
        inl_out[:, idx] = inl & good[None]
        err_out[:, idx] = torch.where(valid[:, pts], e, torch.full_like(e, float("nan")))
    return dict(p3d=out, inliers=inl_out, n_inl=inl_out.sum(0), err=err_out)


# ============================================================================ bundle adjustment
N_PARAMS = {"ext": 6, "full": 9}  # full = extrinsics + log focal scale + k1, k2


def cams_to_theta(cams, mode):
    th = [cams["rvec"], cams["tvec"]]
    if mode == "full":
        th += [torch.zeros_like(cams["rvec"][:, :1]), cams["dist"][:, :2].clone()]
    return torch.cat(th, -1)


def theta_to_cams(theta, cams0, mode):
    cams = dict(cams0)
    cams["rvec"] = theta[:, 0:3].clone()
    cams["tvec"] = theta[:, 3:6].clone()
    if mode == "full":
        K = cams0["K"].clone()
        s = torch.exp(theta[:, 6])
        K[:, 0, 0] *= s; K[:, 1, 1] *= s
        dist = cams0["dist"].clone(); dist[:, 0:2] = theta[:, 7:9]
        cams["K"], cams["dist"] = K, dist
    return cams


def project_obs(th, X, K0, d0, mode):
    """Per-observation projection: th [M,q], X [M,3] -> pixels [M,2], depth [M]."""
    pc = torch.einsum("mij,mj->mi", rodrigues(th[:, 0:3]), X) + th[:, 3:6]
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    k1, k2 = (th[:, 7], th[:, 8]) if mode == "full" else (d0[:, 0], d0[:, 1])
    xd, yd = distort_normalized(x, y, k1, k2, d0[:, 2], d0[:, 3], d0[:, 4])
    fx, fy = K0[:, 0, 0], K0[:, 1, 1]
    if mode == "full":
        s = torch.exp(th[:, 6]); fx, fy = fx * s, fy * s
    return torch.stack([fx * xd + K0[:, 0, 1] * yd + K0[:, 0, 2], fy * yd + K0[:, 1, 2]], -1), pc[:, 2]


def _residuals(theta, X, ci, pi, uv, cams0, mode, jac=True):
    th, x = theta[ci], X[pi]
    K0, d0 = cams0["K"][ci], cams0["dist"][ci]
    proj, z = project_obs(th, x, K0, d0, mode)
    r = proj - uv
    if not jac:
        return r, z
    f = lambda a, b: project_obs(a, b, K0, d0, mode)[0]
    Jc = []
    for k in range(th.shape[1]):
        t = torch.zeros_like(th); t[:, k] = 1
        Jc.append(jvp(lambda a: f(a, x), (th,), (t,))[1])
    Jp = []
    for k in range(3):
        t = torch.zeros_like(x); t[:, k] = 1
        Jp.append(jvp(lambda b: f(th, b), (x,), (t,))[1])
    return r, z, torch.stack(Jc, -1), torch.stack(Jp, -1)


def _robust_cost(r, z, sigma):
    c = sigma ** 2 * torch.log1p((r * r).sum(-1) / sigma ** 2)
    return torch.where((z > 0) & torch.isfinite(c), c, torch.full_like(c, 50.0 * sigma ** 2)).sum()


def lm_bundle(cams0, X0, ci, pi, uv, mode="ext", sigma=5.0, n_iter=25, lam=1e-3, fix_cam=0, chunk=200_000):
    """Schur-complement Levenberg-Marquardt on a Cauchy-robust reprojection cost.
    ci/pi [M]: camera/point index of each observation, uv [M,2] its pixel position."""
    device = X0.device
    C = cams0["rvec"].shape[0]
    theta = cams_to_theta(cams0, mode).clone()
    q = theta.shape[1]
    X = X0.clone()
    Np = X.shape[0]
    eye_q = torch.eye(q, dtype=DT, device=device)
    eye3 = torch.eye(3, dtype=DT, device=device)
    r, z = _residuals(theta, X, ci, pi, uv, cams0, mode, jac=False)
    cost = _robust_cost(r, z, sigma).item()
    hist = [cost]
    for _ in range(n_iter):
        r, z, Jc, Jp = _residuals(theta, X, ci, pi, uv, cams0, mode)
        e2 = (r * r).sum(-1)
        w = 1.0 / (1.0 + e2 / sigma ** 2)  # IRLS weights of the Cauchy loss
        ok = (z > 0) & torch.isfinite(e2) & torch.isfinite(Jc).all(-1).all(-1) & torch.isfinite(Jp).all(-1).all(-1)
        w = torch.where(ok, w, torch.zeros_like(w))
        r, Jc, Jp = torch.nan_to_num(r), torch.nan_to_num(Jc), torch.nan_to_num(Jp)
        Hcc = torch.zeros(C, q, q, dtype=DT, device=device).index_add_(0, ci, torch.einsum("m,mki,mkj->mij", w, Jc, Jc))
        gc = torch.zeros(C, q, dtype=DT, device=device).index_add_(0, ci, torch.einsum("m,mki,mk->mi", w, Jc, r))
        Hcp = torch.einsum("m,mki,mkj->mij", w, Jc, Jp)
        Hpp = torch.zeros(Np, 3, 3, dtype=DT, device=device).index_add_(0, pi, torch.einsum("m,mki,mkj->mij", w, Jp, Jp))
        gp = torch.zeros(Np, 3, dtype=DT, device=device).index_add_(0, pi, torch.einsum("m,mki,mk->mi", w, Jp, r))
        improved = False
        while lam <= 1e8:
            Hcc_d = Hcc + lam * Hcc.diagonal(dim1=-2, dim2=-1)[..., None] * eye_q + 1e-9 * eye_q
            Hpp_inv = torch.linalg.inv(Hpp + lam * Hpp.diagonal(dim1=-2, dim2=-1)[..., None] * eye3 + 1e-9 * eye3)
            S = torch.zeros(C * q, C * q, dtype=DT, device=device)
            Sv = S.view(C, q, C, q)
            for c in range(C):
                Sv[c, :, c, :] += Hcc_d[c]
            T = torch.einsum("mia,mab->mib", Hcp, Hpp_inv[pi])
            b = gc - torch.zeros(C, q, dtype=DT, device=device).index_add_(0, ci, torch.einsum("mib,mb->mi", T, gp[pi]))
            for s0 in range(0, Np, chunk):  # Schur complement, dense per point over cameras
                s1 = min(Np, s0 + chunk)
                m = (pi >= s0) & (pi < s1)
                A = torch.zeros(s1 - s0, C, q, 3, dtype=DT, device=device)
                A.index_put_((pi[m] - s0, ci[m]), Hcp[m], accumulate=True)
                AT = torch.einsum("ncia,nab->ncib", A, Hpp_inv[s0:s1])
                S -= torch.einsum("ncib,ndjb->cidj", AT, A).reshape(C * q, C * q)
            b = b.reshape(-1)
            if fix_cam is not None:  # gauge: freeze one camera's extrinsics
                fr = torch.zeros(C, q, dtype=torch.bool, device=device)
                fr[fix_cam, :6] = True
                fr = fr.reshape(-1)
                S[fr, :] = 0; S[:, fr] = 0; S[fr, fr] = 1.0; b[fr] = 0
            dc = torch.linalg.solve(S, b).reshape(C, q)
            Hpc_dc = torch.zeros(Np, 3, dtype=DT, device=device).index_add_(0, pi, torch.einsum("mia,mi->ma", Hcp, dc[ci]))
            dp = torch.einsum("nab,nb->na", Hpp_inv, gp - Hpc_dc)
            theta_new, X_new = theta - dc, X - dp
            r2, z2 = _residuals(theta_new, X_new, ci, pi, uv, cams0, mode, jac=False)
            new_cost = _robust_cost(r2, z2, sigma).item()
            if np.isfinite(new_cost) and new_cost < cost:
                theta, X, cost = theta_new, X_new, new_cost
                lam = max(lam / 3, 1e-7)
                improved = True
                break
            lam *= 5
        hist.append(cost)
        if not improved or abs(hist[-2] - hist[-1]) < 1e-7 * abs(hist[-1]):
            break
    return theta_to_cams(theta, cams0, mode), X, hist


def run_ba(cams, P, taus, mode="ext", max_pts=500_000, sig_frac=0.5, seed=0):
    """Bundle adjustment on the inliers of robust triangulation, for a schedule of thresholds."""
    for tau in taus:
        R = triangulate_robust(P, cams, tau=tau)
        ok = torch.nonzero(R["n_inl"] >= 2)[:, 0]
        if len(ok) < 100:
            log(f"  BA[{mode}] tau {tau}: only {len(ok)} consistent points, skipping")
            continue
        if len(ok) > max_pts:
            g = torch.Generator(device="cpu").manual_seed(seed)
            ok = ok[torch.randperm(len(ok), generator=g)[:max_pts].to(ok.device)].sort().values
        ci, pj = torch.nonzero(R["inliers"][:, ok], as_tuple=True)
        t0 = time.time()
        cams, _, hist = lm_bundle(cams, R["p3d"][ok].clone(), ci, pj, P[:, ok][ci, pj],
                                  mode=mode, sigma=tau * sig_frac)
        log(f"  BA[{mode}] tau {tau:g}: {len(ok)} points, {len(ci)} obs, "
              f"cost {hist[0]:.3e} -> {hist[-1]:.3e} ({len(hist) - 1} it, {time.time() - t0:.1f}s)")
    return cams


def resect_cameras(cams, P, tau_tri=5.0, tau_pnp=8.0):
    """For each camera: triangulate from the OTHER cameras, PnP-RANSAC this camera against
    those points, and keep the new pose if it explains more detections."""
    device = P.device
    cams = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in cams.items()}
    valid = torch.isfinite(P[..., 0])
    for c in range(P.shape[0]):
        Pm = P.clone(); Pm[c] = float("nan")
        idx = torch.nonzero(valid[c] & (torch.isfinite(Pm[..., 0]).sum(0) >= 2))[:, 0]
        R = triangulate_robust(Pm[:, idx], cams, tau=tau_tri)
        good = R["n_inl"] >= 2
        if good.sum() < 50:
            continue
        X = R["p3d"][good]; uv = P[c, idx[good]]
        n0 = int((reproj_err(X, P[:, idx[good]], cams)[c] < tau_pnp).sum().item())
        Xn = X.cpu().numpy(); uvn = uv.cpu().numpy()
        K = cams["K"][c].cpu().numpy(); dd = cams["dist"][c].cpu().numpy()
        ok, rv, tv, inl = cv2.solvePnPRansac(Xn, uvn, K, dd, reprojectionError=tau_pnp, iterationsCount=5000,
                                             confidence=0.9999, flags=cv2.SOLVEPNP_EPNP)
        n1 = 0 if inl is None else len(inl)
        upd = bool(ok and n1 > n0)
        if upd:
            rv, tv = cv2.solvePnPRefineLM(Xn[inl[:, 0]], uvn[inl[:, 0]], K, dd, rv, tv)
            cams["rvec"][c] = torch.as_tensor(rv.ravel(), dtype=DT, device=device)
            cams["tvec"][c] = torch.as_tensor(tv.ravel(), dtype=DT, device=device)
        log(f"  resect cam idx {c}: {len(Xn)} 2D-3D matches, inliers {n0} -> {n1}"
            f"{' (pose updated)' if upd else ' (kept)'}")
    return cams


def loo_agreement(cams, P, tau_tri=5.0, tau=8.0):
    """Per camera: fraction of its detections within tau px of the point triangulated
    robustly from the other cameras (calibration health metric)."""
    valid = torch.isfinite(P[..., 0]); out = []
    for c in range(P.shape[0]):
        Pm = P.clone(); Pm[c] = float("nan")
        idx = torch.nonzero(valid[c] & (torch.isfinite(Pm[..., 0]).sum(0) >= 2))[:, 0]
        R = triangulate_robust(Pm[:, idx], cams, tau=tau_tri)
        good = R["n_inl"] >= 2
        if not good.any():
            out.append(float("nan")); continue
        e = reproj_err(R["p3d"][good], P[:, idx[good]], cams)[c]
        out.append(round((e < tau).double().mean().item(), 3))
    return out


def calibrate(cams, P, cam_names):
    fmt = lambda loo: " ".join(f"{n}:{v:.2f}" for n, v in zip(cam_names, loo))
    log("leave-one-out agreement (init):", fmt(loo_agreement(cams, P)))
    for r, taus in enumerate([(200, 100, 50), (40, 20, 10, 5), (20, 10, 5)]):
        log(f"calibration round {r + 1}/3: bundle adjustment (tau {taus} px)")
        cams = run_ba(cams, P, taus, mode="ext")
        log(f"calibration round {r + 1}/3: camera resection")
        cams = resect_cameras(cams, P)
        log(f"leave-one-out agreement (round {r}):", fmt(loo_agreement(cams, P)))
    log("final bundle adjustment: extrinsics")
    cams = run_ba(cams, P, (20, 10, 5), mode="ext")
    log("final bundle adjustment: extrinsics + focal length + k1, k2")
    cams = run_ba(cams, P, (10, 5), mode="full")
    loo = loo_agreement(cams, P)
    log("leave-one-out agreement (final):", fmt(loo))
    return cams, loo


# ============================================================================ QC flags
def rig_scale(cams):
    """Median distance of camera centers from their centroid (the calibration's arbitrary unit)."""
    R = rodrigues(cams["rvec"])
    centers = -torch.einsum("cji,cj->ci", R, cams["tvec"])
    return torch.linalg.norm(centers - centers.mean(0), dim=-1).median().item()


# QC length scales as fractions of rig_scale (tuned on 2026_05_04_mouse_right, rig scale 4.47:
# voxel 0.05, height-map cell 0.2, max height residual 0.15)
STATIC_VOXEL_FRAC = 0.011
SURFACE_CELL_FRAC = 0.045
SURFACE_THRESH_FRAC = 0.034


def static_flags(p3d, time_offset, vs, min_count=1000, min_hours=5):
    """Points in 3D voxels occupied >= min_count frames over >= min_hours distinct hours (plus
    neighbouring voxels). The tracker is motion based (background adapts within ~1 s), so the
    animal cannot produce such voxels; they are static artifacts seen consistently by 2+ cams."""
    m = np.isfinite(p3d).all(1)
    out = np.zeros(len(p3d), bool)
    if m.sum() == 0:
        return out
    ii = np.nonzero(m)[0]
    u, inv, cnt = np.unique(np.floor(p3d[m] / vs).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    nh = pd.DataFrame({"v": inv, "h": (time_offset[ii] // 3600).astype(int)}).drop_duplicates() \
        .groupby("v").size().reindex(range(len(u)), fill_value=0).to_numpy()
    bad = set()
    for b in map(tuple, u[(cnt >= min_count) & (nh >= min_hours)]):
        for dv in itertools.product((-1, 0, 1), repeat=3):
            bad.add((b[0] + dv[0], b[1] + dv[1], b[2] + dv[2]))
    if bad:
        out[ii] = np.array([tuple(x) in bad for x in u])[inv]
    return out


def surface_residual(p3d, ref_mask, cell, min_n=20):
    """Distance along the ground normal to a height-map (median height per cell) fitted to the
    reference points; inf where the map has no data. Returns None if too few reference points."""
    q = p3d[ref_mask & np.isfinite(p3d).all(1)]
    if len(q) < 1000:
        return None
    mu = np.median(q, 0)
    Vt = np.linalg.svd(q - mu, full_matrices=False)[2]
    uvh = (q - mu) @ Vt.T
    lo = np.percentile(uvh[:, :2], 0.1, 0) - cell
    nb = np.ceil((np.percentile(uvh[:, :2], 99.9, 0) + cell - lo) / cell).astype(int)
    ij = np.clip(((uvh[:, :2] - lo) / cell).astype(int), 0, nb - 1)
    g = pd.DataFrame({"c": ij[:, 0] * nb[1] + ij[:, 1], "h": uvh[:, 2]}).groupby("c")["h"]
    med, n = g.median(), g.size()
    Hmap = np.full(nb.prod(), np.nan)
    keep = n[n >= min_n].index
    Hmap[keep] = med[keep]
    r = np.full(len(p3d), np.inf)
    fin = np.nonzero(np.isfinite(p3d).all(1))[0]
    a = (p3d[fin] - mu) @ Vt.T
    cij = np.floor((a[:, :2] - lo) / cell)
    inside = (cij >= 0).all(1) & (cij[:, 0] < nb[0]) & (cij[:, 1] < nb[1])
    cij = cij[inside].astype(np.int64)
    h = Hmap[cij[:, 0] * nb[1] + cij[:, 1]]
    rr = np.abs(a[inside, 2] - h)
    rr[~np.isfinite(h)] = np.inf
    r[fin[inside]] = rr
    return r


# ============================================================================ main
def parse_args():
    parser = argparse.ArgumentParser(description="Robust calibration refinement and robust triangulation.")
    parser.add_argument("--tracked", help="Directory containing tracking parquet files and calibration output")
    parser.add_argument("--arena", default="right-2026", help="Arena config (sync-block positions)")
    parser.add_argument("--hz", type=float, default=25.0, help="Output time grid rate (cameras run at 25 Hz)")
    parser.add_argument("--tol", type=float, default=0.02, help="Max time difference (s) to match a frame")
    parser.add_argument("--tau", type=float, default=10.0, help="Inlier threshold (px) for triangulation")
    parser.add_argument("--calib", default=None, help="Starting calibration (default: calibration_vggt_init.toml)")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"robust_triangulation: tracked={args.tracked} arena={args.arena} device={device}")

    calib_fname_init = args.calib or os.path.join(args.tracked, "calibration_vggt_init.toml")
    calib_fname_out = os.path.join(args.tracked, "calibration_adjusted.toml")
    points_fname_out = os.path.join(args.tracked, "points_3d.npz")

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

    cams = load_cams(calib_fname_init, device)
    if cams["rvec"].shape[0] != len(cam_names):
        raise ValueError(f"calibration has {cams['rvec'].shape[0]} cameras but tracking has {len(cam_names)}")

    log("== step 2/5: filtering detections")
    keep = detection_filter(p2d, scores, cam_names, args.arena)
    p2d_f = np.where(keep[..., None], p2d, np.nan)
    sel = np.nonzero(np.isfinite(p2d_f[..., 0]).sum(0) >= 2)[0]
    P = torch.as_tensor(p2d_f[:, sel], dtype=DT, device=device)

    with torch.no_grad():
        log("== step 3/5: calibrating (about 5 min on one GPU for 60 h of 12 cameras)")
        cams, loo = calibrate(cams, P, cam_names)
        save_cams(cams, calib_fname_out, cam_names)
        log("saved", calib_fname_out)

        log(f"== step 4/5: robust triangulation of {len(sel)} timepoints (tau {args.tau:g} px)")
        R = triangulate_robust(P, cams, tau=args.tau, progress=True)

    N, C = p2d.shape[1], len(cam_names)
    p3d_all = np.full((N, 3), np.nan); p3d_all[sel] = R["p3d"].cpu().numpy()
    n_inl = np.zeros(N, np.int16); n_inl[sel] = R["n_inl"].cpu().numpy()
    inliers = np.zeros((C, N), bool); inliers[:, sel] = R["inliers"].cpu().numpy()
    err_cams = np.full((C, N), np.nan, np.float32); err_cams[:, sel] = R["err"].cpu().numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        err = np.where(n_inl > 0, np.where(inliers, err_cams, 0).sum(0) / np.maximum(n_inl, 1), np.nan)

    log("== step 5/5: quality flags and saving")
    scale = rig_scale(cams)
    static = static_flags(p3d_all, time_offset, vs=STATIC_VOXEL_FRAC * scale)
    surf = surface_residual(p3d_all, (n_inl >= 3) & ~static, cell=SURFACE_CELL_FRAC * scale)
    if surf is None:
        log("WARNING: too few >=3-camera points for a ground model; skipping the surface check")
        surf = np.zeros(N)
    good = (n_inl >= 2) & ~static & ((n_inl >= 3) | (surf < SURFACE_THRESH_FRAC * scale))
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
        cam_names=np.array(cam_names), loo=np.array(loo), hz=args.hz, tau=args.tau,
    )
    log("saved", points_fname_out, f"({time.time() - t_start:.0f}s total)")


if __name__ == "__main__":
    main()
