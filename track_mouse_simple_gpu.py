#!/usr/bin/env ipython

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import argparse
import os
from glob import glob
from tqdm import tqdm

def track_video(video_path, cam_id=None, adapt_rate=0.97, threshold=15,
                min_size=4, max_size=1200, device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cam_configs = {
        1:  dict(track_point=[377, 73],  blank_x=range(-4, 5),   blank_y=range(-2, 4),
                 mask_regions=[dict(rows=(0,1))]),
        2:  dict(track_point=[292, 112], blank_x=range(-5, 9),   blank_y=range(-1, 4),
                 mask_regions=[dict(rows=(0,10)), dict(rows=(0,20), cols=(0,50))]),
        3:  dict(track_point=[231, 82],  blank_x=range(-4, 14),  blank_y=range(-5, 6)),
        4:  dict(track_point=[181, 206], blank_x=range(-8, 6),   blank_y=range(-4, 4)),
        5:  dict(track_point=[313, 119], blank_x=range(-5, 6),   blank_y=range(-2, 3),
                 mask_regions=[dict(rows=(0,50), cols=(500, None))]),
        6:  dict(track_point=[360, 123], blank_x=range(-8, 4),   blank_y=range(-2, 3),
                 mask_regions=[dict(rows=(0,70),  cols=(310, None)),
                                dict(rows=(0,120), cols=(450, None))]),
        7:  dict(track_point=[485, 190], blank_x=range(-8, 4),   blank_y=range(-2, 3)),
        8:  dict(track_point=[307, 127], blank_x=range(-6, 4),   blank_y=range(0,  4)),
        9:  dict(track_point=[580, 41],  blank_x=range(-14, 7),  blank_y=range(-4, 6),
                 mask_regions=[dict(rows=(0,50), cols=(0,50))]),
        10: dict(track_point=[617, 174], blank_x=range(-14, 7),  blank_y=range(-4, 6),
                 mask_regions=[dict(rows=(0,70), cols=(0,100)),
                                dict(rows=(0,40), cols=(0,250))]),
        11: dict(track_point=[28,  146], blank_x=range(-14, 7),  blank_y=range(-4, 6)),
        12: dict(track_point=[267, 104], blank_x=range(-14, 7),  blank_y=range(-4, 6)),
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"  {w}x{h}  frames={n_frames}  fps={fps:.1f}  device={device}")

    cfg          = cam_configs.get(cam_id, {})
    track_point  = cfg.get('track_point', None)
    blank_x      = list(cfg.get('blank_x', []))
    blank_y      = list(cfg.get('blank_y', []))
    mask_regions = cfg.get('mask_regions', [])

    spatial_mask = torch.ones((h, w), dtype=torch.float32, device=device)
    for region in mask_regions:
        rows = region.get('rows', None)
        cols = region.get('cols', None)
        r0 = rows[0] if rows else 0
        r1 = rows[1] if (rows and rows[1] is not None) else h
        c0 = cols[0] if cols else 0
        c1 = cols[1] if (cols and cols[1] is not None) else w
        spatial_mask[r0:r1, c0:c1] = 0.0

    sync_blank_mask = torch.ones((h, w), dtype=torch.float32, device=device)
    if track_point is not None and blank_x and blank_y:
        px, py = track_point
        for dy in blank_y:
            for dx in blank_x:
                ry = int(np.clip(py + dy, 0, h - 1))
                rx = int(np.clip(px + dx, 0, w - 1))
                sync_blank_mask[ry, rx] = 0.0

    box_kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device) / 9.0

    bg_ref     = None
    kal_xy     = torch.zeros(2, device=device)
    kal_radius = torch.tensor(10.0, device=device)

    frame_indices = []
    xs            = []
    ys            = []
    scores        = []   # blob area, analogous to peak confidence in predict_videos.py
    sync_vals     = []   # sync signal values

    frame_idx = 0
    with tqdm(total=n_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            vid_frame = torch.from_numpy(
                frame[:, :, 0].astype(np.float32)
            ).to(device)
            vid_frame = vid_frame * spatial_mask

            if bg_ref is None:
                bg_ref = vid_frame.clone()
                current_adapt = 0.0
            else:
                current_adapt = adapt_rate

            if track_point is not None:
                px, py = track_point
                sync_val = vid_frame[py, px].item()
            else:
                sync_val = float('nan')

            I = (vid_frame - bg_ref) * sync_blank_mask

            bg_ref = (1.0 - current_adapt) * vid_frame + current_adapt * bg_ref

            kal_radius = kal_radius + 3.0

            I_4d = I.unsqueeze(0).unsqueeze(0)
            I_sm = F.conv2d(I_4d, box_kernel, padding=1).squeeze()

            best_cx   = float('nan')
            best_cy   = float('nan')
            best_area = float('nan')

            if I_sm.max().item() > 20:
                binary_np = (I_sm > threshold).cpu().numpy().astype(np.uint8)
                n_labels, _, stats_np, centroids_np = \
                    cv2.connectedComponentsWithStats(binary_np)

                if n_labels > 1:
                    areas     = torch.from_numpy(
                        stats_np[1:, cv2.CC_STAT_AREA].astype(np.float32)
                    ).to(device)
                    centroids = torch.from_numpy(
                        centroids_np[1:].astype(np.float32)
                    ).to(device)

                    size_ok  = (areas >= min_size) & (areas < max_size)
                    dists    = torch.linalg.norm(centroids - kal_xy.unsqueeze(0), dim=1)
                    dist_ok  = dists < kal_radius
                    valid    = size_ok & dist_ok

                    if valid.any():
                        valid_areas     = areas[valid]
                        valid_centroids = centroids[valid]
                        best_idx        = torch.argmax(valid_areas)
                        best_cx         = valid_centroids[best_idx, 0].item()
                        best_cy         = valid_centroids[best_idx, 1].item()
                        best_area       = valid_areas[best_idx].item()
                        kal_xy          = valid_centroids[best_idx]
                        kal_radius      = torch.tensor(30.0, device=device)

            frame_indices.append(frame_idx)
            xs.append(best_cx)
            ys.append(best_cy)
            scores.append(best_area)
            sync_vals.append(sync_val)

            pbar.update(1)
            frame_idx += 1

    cap.release()

    df = pd.DataFrame({
        'frame': frame_indices,
        'x':     xs,
        'y':     ys,
        'score': scores,       # blob area in pixels; NaN when no detection
        'sync':  sync_vals,    # sync signal at track_point
    })
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPU thermal blob tracker')
    parser.add_argument('vidname', help='Camera number string, e.g. "13"')
    parser.add_argument('--source_path', type=str,
                        default='/groups/voigts/voigtslab/outdoor/2025_09_25_mouse_new_day3/data')
    parser.add_argument('--outdir',    type=str,    default='output')
    parser.add_argument('--cam_id',    type=int,    default=None,
                        help='Override cam_id (default: parsed from filename)')
    parser.add_argument('--adapt',     type=float,  default=0.97)
    parser.add_argument('--threshold', type=float,  default=15.0)
    parser.add_argument('--min_size',  type=int,    default=4)
    parser.add_argument('--max_size',  type=int,    default=1200)
    parser.add_argument('--device',    type=str,    default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    os.makedirs(args.outdir, exist_ok=True)

    fnames = sorted(glob(os.path.join(args.source_path,
                                      'video_{}_*.avi'.format(args.vidname))))
    if not fnames:
        raise FileNotFoundError(f"No videos matching video_{args.vidname}_*.avi "
                                f"in {args.source_path}")

    for ix_fname, fname in enumerate(fnames):
        print("")
        print("{}/{} - {}".format(ix_fname + 1, len(fnames), os.path.basename(fname)))

        outname = os.path.basename(fname).replace('.avi', '.pq')
        outpath = os.path.join(args.outdir, outname)

        if os.path.exists(outpath):
            continue
        # --- timestamps (same as predict_videos.py) ---
        # fname_csv = fname.replace('video_', 'timestamps_').replace('.avi', '.cvs')
        #
        # match timestamp file by ignoring seconds (video and timestamp can differ by a second)
        # e.g. video_17_2026-04-13T06_14_42.avi -> timestamps_17_2026-04-13T06_14_*.cv*
        vid_base = os.path.basename(fname)
        prefix = vid_base.replace('video_', 'timestamps_').rsplit('_', 1)[0]  # strip seconds + .avi
        matches = glob(os.path.join(args.source_path, prefix + '_*.cvs')) + \
                  glob(os.path.join(args.source_path, prefix + '_*.csv'))
        if not matches:
            print(f"  WARNING: no timestamp file found for {vid_base}, skipping")
            continue
        fname_csv = matches[0]
        
        tstamps = pd.read_csv(fname_csv, header=None, names=['timestamp'],
                              parse_dates=[0])
        tstamps['frame'] = np.arange(len(tstamps))

        # --- cam_id: parse from filename if not overridden ---
        if args.cam_id is not None:
            cam_id = args.cam_id
        else:
            tmp = os.path.basename(fname).split('_')
            cam_id = int(tmp[1]) - 9
            print(f"  cam_id={cam_id} (parsed from filename)")

        df = track_video(
            fname,
            cam_id     = cam_id,
            adapt_rate = args.adapt,
            threshold  = args.threshold,
            min_size   = args.min_size,
            max_size   = args.max_size,
            device     = device,
        )

        merged = pd.merge(tstamps, df, on='frame')
        merged.to_parquet(outpath)
        print(f"  saved -> {outpath}")
