# outdoor mouse tracking pipeline

This is a pipeline for 3D tracking of mice in an outdoor arena at HHMI Janelia.

The pipeline is as follows:
1. track_mouse_simple_gpu.py - runs tracking by detecting movement in the thermal camera video
2. calibrate_videos_vggt.py - uses VGGT network to provide an initial calibration based on video frames
3. bundle_adjust_triangulate.py - runs bundle adjustment on tracking to refine initial calibration and then triangulates the points

Each of these scripts takes --source (video data folder) and --tracked (output folder) arguments. 

Steps 1 and 2 can be run in parallel, whereas step 3 depends on the first two.

To make it easy to process a session, we provide another script to easily submit the scripts on the Janelia cluster: `submit_tracking.py`
This script handles the submission of all 3 steps, adding dependency of step 3 on the first two.

Here's an example submission:

``` sh
python submit_tracking.py \
  --source /groups/voigts/voigtslab/outdoor/2026_04_10_mouse_right/data \
  --tracked /groups/karashchuk/karashchuklab/outdoor_analysis/2026_04_10_mouse_right \
  --arena right
```

(The arena parameter is needed for track_mouse_simple_gpu.py to know where the syncing thermal block is.) 


