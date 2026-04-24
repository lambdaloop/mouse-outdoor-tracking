#!/usr/bin/env python

import argparse
import cv2
import pandas as pd
import os
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Overlay tracked positions on a video.")
    parser.add_argument("basename", help="Video basename (without extension)")
    parser.add_argument("--source-path", default="/groups/voigts/voigtslab/outdoor/2026_04_10_mouse_right/data",
                        help="Directory containing source video")
    parser.add_argument("--tracked-root", default="outnew",
                        help="Directory containing tracking parquet files")
    parser.add_argument("--out-dir", default="vids",
                        help="Directory for output video")
    return parser.parse_args()


def main():
    args = parse_args()

    vidname = os.path.join(args.source_path, args.basename + ".avi")
    dname = os.path.join(args.tracked_root, args.basename + ".pq")
    outname = os.path.join(args.out_dir, args.basename + ".avi")

    print(vidname)
    print(outname)

    data = pd.read_parquet(dname)

    cap = cv2.VideoCapture(vidname)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(outname, fourcc, fps, (width, height))

    for frame_idx in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_data = data[data["frame"] == frame_idx]

        for _, row in frame_data.iterrows():
            if not pd.isna(row["x"]) and not pd.isna(row["y"]):
                x, y = int(row["x"]), int(row["y"])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        if not frame_data.empty and not pd.isna(frame_data["score"].iloc[0]):
            score_value = frame_data["score"].iloc[0]
            score_text = f"Score: {score_value:.2f}"

            text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = width - text_size[0] - 20
            text_y = height - 20

            cv2.putText(frame, score_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, score_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
