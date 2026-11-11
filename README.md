# outdoor mouse tracking pipeline

This is a pipeline for 3D tracking of mice in an outdoor arena at HHMI Janelia.

The pipeline is as follows:
1. `track_mouse_simple_gpu.py` - runs tracking by detecting movement in the thermal camera video
2. `calibrate_videos_vggt.py` - uses the VGGT network (`aniposelib.init.vggt`) to provide an initial calibration based on video frames
3. `robust_triangulation.py` - filters detections, refines the initial calibration with `aniposelib.calibrate_from_points` (robust bundle adjustment, resection, evidence-gated focal length/k1 refinement, dropping of desynchronised hours), then robustly triangulates the points with `aniposelib.reconstruct_3d` (ignoring cameras whose detections disagree; static-artifact and off-ground-surface flags). Writes `calibration_adjusted.toml`, `calibration_report.json` and `points_3d.npz`. (Replaces the older `bundle_adjust_triangulate.py`, kept for reference.)

The pipeline uses [aniposelib-pytorch](https://pypi.org/project/aniposelib-pytorch/) >= 1.1 (imports as `aniposelib`); it is installed by `pixi install` (see `pixi.toml`). Classic `aniposelib` is not used. Calibration files load with the numeric camera order (`cam_2` before `cam_10`).

Each of these scripts takes --source (video data folder) and --tracked (output folder) arguments. 

Steps 1 and 2 can be run in parallel, whereas step 3 depends on the first two.

To make it easy to process a session, we provide another script to easily submit the scripts on the Janelia cluster: `submit_tracking.py`
This script handles the submission of all 3 steps, adding dependency of step 3 on the first two.

Here's an example submission:

``` sh
python3 submit_tracking.py \
  --source /groups/voigts/voigtslab/outdoor/2026_04_10_mouse_right/data \
  --tracked /groups/karashchuk/karashchuklab/outdoor_analysis/2026_04_10_mouse_right \
  --arena right
```

(The arena parameter is needed for track_mouse_simple_gpu.py to know where the syncing thermal block is.) 


