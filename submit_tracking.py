#!/usr/bin/env python3

import argparse
import os
import subprocess
from datetime import datetime
from glob import glob
from pathlib import Path

# Global timestamp variable, set once at module level
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit bsub jobs for tracking, calibration, and bundle adjustment."
    )
    parser.add_argument("--source", required=True, help="Directory with source .avi videos")
    parser.add_argument("--tracked", required=True, help="Directory for output files")
    parser.add_argument("--arena", default="right", help="Arena name for tracking configuration")
    return parser.parse_args()


def get_camera_numbers(source_dir, pattern="video_*_*.avi"):
    """
    Return sorted list of unique camera numbers (as strings) found in source_dir.
    Files are expected to be named e.g. video_10_2026-04-13T06_14_42.avi
    """
    avi_files = glob(os.path.join(source_dir, pattern))
    if not avi_files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {source_dir}"
        )
    cam_nums = set()
    for fpath in avi_files:
        stem = os.path.basename(fpath)
        parts = stem.split("_")
        if len(parts) >= 3:
            cam_nums.add(parts[1])  # e.g., "10" from video_10_...
    if not cam_nums:
        raise ValueError(f"Could not extract camera numbers from files in {source_dir}")
    return sorted(cam_nums)


def submit_bjob(job_name, command, dependency_ids=None):
    """
    Submit a bsub job with optional dependencies.
    Returns the job ID string.
    Uses the global timestamp to prefix the job name and to place logs
    into a subdirectory logs/{timestamp}/.
    """
    global timestamp
    os.makedirs(f"logs/{timestamp}", exist_ok=True)

    full_job_name = f"{timestamp}_{job_name}"
    bsub_cmd = [
        "bsub",
        "-J", full_job_name,
        "-n", "8",
        "-q", "gpu_l4",
        "-gpu", "num=1",
        "-W", "12:00",
        "-o", f"logs/{timestamp}/{job_name}.out",
        "-e", f"logs/{timestamp}/{job_name}.err",
    ]
    if dependency_ids:
        dep_expr = " && ".join(f"done({jid})" for jid in dependency_ids)
        bsub_cmd.extend(["-w", dep_expr])
    bsub_cmd.extend(["pixi", "run", "python"] + command)

    print(" ".join(bsub_cmd))
    result = subprocess.run(bsub_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError(f"bsub submission failed for {full_job_name}")
    # Parse job ID from output like "Job <123456> is submitted to queue <gpu_l4>."
    output = result.stdout.strip()
    print(output)
    job_id = None
    for word in output.split():
        if word.startswith("<") and word.endswith(">"):
            job_id = word.strip("<>")
            break
    if job_id is None:
        raise RuntimeError(f"Could not extract job ID from bsub output: {output}")
    return job_id


def main():
    args = parse_args()

    source_dir = args.source
    tracked_dir = args.tracked

    # Create tracked directory
    Path(tracked_dir).mkdir(parents=True, exist_ok=True)

    # Detect unique camera numbers
    cam_numbers = get_camera_numbers(source_dir)
    print(f"Detected cameras: {cam_numbers}")

    # Submit tracking jobs (one per camera)
    tracking_job_ids = []
    for cam_num in cam_numbers:
        job_name = f"track_cam_{cam_num}"
        command = [
            "track_mouse_simple_gpu.py",
            cam_num,                             # positional vidname argument
            "--source", source_dir,
            "--tracked", tracked_dir,
            "--arena", args.arena,
        ]
        jid = submit_bjob(job_name, command)
        tracking_job_ids.append(jid)

    # Submit calibration job (no dependencies – runs in parallel)
    calib_command = [
        "calibrate_videos_vggt.py",
        "--source", source_dir,
        "--tracked", tracked_dir,
    ]
    calib_job_id = submit_bjob("calibrate_vggt", calib_command)

    # Submit bundle adjustment job (dependent on all tracking + calibration)
    # Use default dependency expression
    all_deps = tracking_job_ids + [calib_job_id]
    bundle_command = [
        "bundle_adjust_triangulate.py",
        "--tracked", tracked_dir,
    ]
    bundle_job_id = submit_bjob("bundle_adjust", bundle_command, dependency_ids=all_deps)

    print("\nSubmitted jobs:")
    print(f"  Tracking jobs: {', '.join(tracking_job_ids)}")
    print(f"  Calibration job: {calib_job_id}")
    print(f"  Bundle adjustment job: {bundle_job_id} (depends on all above)")


if __name__ == "__main__":
    main()
