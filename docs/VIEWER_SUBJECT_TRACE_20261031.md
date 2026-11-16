# Tracing the people visible at 107.51s

Date: 2026-10-31, Asia/Saigon.

## Problem and correction

User reports the group is visible while exploring around the 107s camera, yet
cannot find it after going to 228s and exploring. They briefly wrote 117s, then
corrected it to 107s. An already-started 117s restoration diagnostic completed
separately; it is not used to explain the corrected observation. The earlier
107.51 -> 228.11 -> exact107.51 live test proved image repeatability, but did not
identify how these people are represented or occluded from the later position.

## Method

Created `scripts/trace-viewer-subject.py`. Loaded the exact active courtyard
checkpoint `results/realsense_20261030_111354_all30_120k/train/ckpts/final_post_update.pt`
and the same normalized dataset poses. Rendered RGB plus expected depth at
1280x720 using the regular CUDA rasterizer. Pixel (493,330) is on the torso of
the front standing person in the 107.51s image. Back-projected its median 7x7
expected depth to estimate a subject location. This is an estimate from rendered
depth, not a calibrated ground-truth point or semantic Gaussian annotation.

Target normalized coordinates: (0.04051027, 0.68209737, -0.05834709).
Rendered expected depth at the selected pixel: 0.42069060; alpha 0.99988395.

Selected every Gaussian whose mean lies within 0.10 normalized units of that
target: **1,447 fixed Gaussian IDs**. Saved their indices. The selection includes
people and some nearby background/floor; it is not a pure semantic people mask.
The isolated A render visibly includes the front standing people.

Rendered the full scene and this same fixed subset from the 228.11s camera
position, aiming the camera exactly at the estimated target. Also rendered
degree-zero SH and intermediate camera positions for diagnosis. Checkpoint and
viewer were not modified; isolation applies only to diagnostic renders.

## Findings

- The selected people-region Gaussians still render from the B camera when the
  rest of the model is hidden. They are fragmented and appear smaller from B;
  they were not deleted or unloaded because the replay reached 228s.
- In the full-scene B render aimed at the target, foreground geometry covers
  this region. The target is 0.927882 normalized units from B; the center ray's
  full-scene expected depth is only 0.171184, with alpha 0.999731. The isolated
  region's center expected depth is 0.942553, alpha 0.950598. This supports
  occlusion by nearer learned geometry in this tested B view.
- SH degree zero retains the standing people at A and still shows the
  foreground obstruction at B. Direction-dependent color alone is not needed
  to explain their absence from this full-scene B view.
- At 10% of the line from A to B, aiming at the target still shows the people;
  at 25%, a near surface fills the image. Straight-line movement through an
  unconstrained Gaussian scene can enter geometry; this is not a walkable route.
- Prior exact-restoration test remains valid: unchanged camera A at replay
  228.11s returns exactly the same image as A at 107.51s.

## Limits and interpretation

This identifies a concrete reason for absence in the tested 228s-position view:
foreground learned geometry blocks the fixed subject region. It does not measure
the camera positions used in the user's manual search or prove the cause for
every view they tried. It does not establish whether blockers are physically
correct surfaces, reconstruction artifacts, or a consequence of pose drift.
The fragmented appearance of the same region from B demonstrates that its
appearance is not a clean complete human model from all directions.

No retraining, deletion, pruning, checkpoint changes, viewer restart, or global
render-setting changes were performed. Original active viewer is preserved.

## Evidence and reproduction

`results/viewer_subject_trace_20261031/` contains full A/B diagnostic PNGs,
SH-degree-zero controls, isolated-region PNGs, interpolated-position PNGs,
`a_expected_depth.npy`, `subject_region_gaussian_ids.npy`, `subject_trace.json`,
and `visibility_comparison.png` (a labeled diagnostic figure; isolated B crop enlarged).

Run with the same MSVC/CUDA environment as `scripts/view-room.ps1`:

```powershell
.venv/Scripts/python.exe scripts/trace-viewer-subject.py --dataset data/realsense_20261030_111354_all30/rgbd/dataset --checkpoint results/realsense_20261030_111354_all30_120k/train/ckpts/final_post_update.pt --output results/viewer_subject_trace_20261031
```

Both CUDA diagnostic passes completed, saved finite numerical depths and rendered
images that were visually inspected. The second pass added the fixed-region
isolation and center-ray measurements. No scientific quality improvement is
claimed; these are diagnosis artifacts.
