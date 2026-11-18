# RealSense scene reconstruction experiment

Date: 2026-10-30 (Asia/Saigon). Source: `20261029_140739.db3`.

## Objective

Reconstruct the captured museum room as well as practical with 3D Gaussian
splatting on the local GTX 1660 SUPER (6 GiB). Keep reproducible commands,
checkpoints, view comparisons, quality measurements, and memory measurements.
The completed experiments and their limitations are recorded below.

## Final result

Created two usable 3DGS reconstructions on the local 6 GiB GPU:

| Model | Recovered views | Updates | Gaussians | Held-out PSNR / SSIM / LPIPS | Peak training PyTorch GiB |
| --- | ---: | ---: | ---: | --- | ---: |
| Wider view coverage: `full_scene_626_depth30k` | 626 / 694 selected | 30,000 | 999,133 | 18.8873 / 0.66191 / 0.53941 | 1.37098 |
| Sharper native-resolution alternative: `global_native_depth15k` | 151 / 694 selected | 15,000 | 699,658 | 22.6995 / 0.79375 / 0.43143 | 1.07923 |

**The primary model recovers 90.20% of selected camera views, not a measured
90.20% of room surfaces.** Its 68 excluded views are listed in `rgbd/report.json`.
The native alternative has cleaner detail in its recovered views. The two rows
use different images, resolutions, and test sets, so their scores do not rank
them on a shared benchmark. The broad model has 79 held-out images; native has 19.

Selected the broad model's 30k checkpoint because it has the best held-out PSNR
and LPIPS among its saved 7k/15k/30k checkpoints. SSIM is slightly higher at 15k.
Selection, exact checkpoint hash, and viewer starting camera are saved in
`results/realsense_20261029/selection.json`. Verified 30,000 actual updates,
30,000 recorded training-camera samples, finite checkpoint tensors, the live
Gaussian cap, and unchanged dataset binary hashes.

Full output paths:

- Broad preview / PLY: `results/realsense_20261029/full_preview/`.
- Native preview / PLY: `results/realsense_20261029/native_preview/`.
- Selected checkpoint:
  `results/realsense_20261029/full_scene_626_depth30k/ckpts/final_post_update.pt`.
- Native checkpoint:
  `results/realsense_20261029/global_native_depth15k/ckpts/final_post_update.pt`.

The exact final checkpoint includes the last optimizer update. The original
trainer's checkpoint/PLY snapshots are saved immediately before that update;
the preview exporter uses the exact post-update checkpoint for `scene.ply`.

### Verified outputs and hardware fit

- Broad checkpoint and **235,796,865-byte `scene.ply`** both reload with all
  999,133 splats and finite parameters. All rendered video pixels were finite.
- Broad comparison MP4: H.264, **1696 x 476, 4 FPS, 156.5 seconds**;
  173,299,285 bytes. Decoded first, middle, and last frames successfully.
  The even preview height is for video encoding; training/evaluation uses
  the dataset's 848 x 477 images.
- After training finished, 5 warm-up renders plus 50 timed captured views at
  **848 x 476** measured **154.65 FPS from mean latency**, **4.74 ms median**,
  **10.77 ms 90th percentile**, and **0.55328 GiB** peak PyTorch render allocation.
  This includes camera-tensor preparation and CUDA rendering, and excludes
  browser display, image/video encoding, and file I/O. Desktop applications
  remained open. The GPU snapshot was **1830 MiB total used of 6144 MiB**.
- Completed training on the actual GTX 1660 SUPER without an OOM. Peak training
  PyTorch allocation is allocator-tracked memory, not total device memory.
  Observed combined device use during concurrent training stayed below 6 GiB.
- Local viewer started successfully at **http://localhost:8080**, HTTP **200**,
  using `frame_000645.jpg` for the initial central-display view. Background
  launcher PID: **21628** (stop with `taskkill.exe /PID 21628 /T /F`).
  This status is a snapshot of this experiment session; use the launcher to
  restart it later. See `viewer_status.json` and `full_preview/verification.json`
  under `results/realsense_20261029/`.

Open the broad [comparison sheet](results/realsense_20261029/full_preview/comparison_sheet.jpg),
[comparison video](results/realsense_20261029/full_preview/capture_path.mp4), or
[3DGS PLY](results/realsense_20261029/full_preview/scene.ply).
The native [comparison sheet](results/realsense_20261029/native_preview/comparison_sheet.jpg)
and [PLY](results/realsense_20261029/native_preview/scene.ply) provide the sharper
alternative in its recovered views.

## Recording and starting state

- Original SQLite recording: 37,372,915,712 bytes; opened read-only.
- RealSense D455; 10,395 RGB and 10,395 depth messages, approximately 347 seconds.
- RGB: 1280 x 720, 30 FPS. Depth: 848 x 480, 30 FPS, Z16, 0.001 m/unit.
- Accelerometer and gyro messages and stream calibration/extrinsics are present.
  IMU streams were not fused; recovered poses use visual/depth constraints.
- Existing extraction: 694 RGB frames, approximately 2 FPS, selecting the
  sharpest of three nearby candidates. See
  `data/realsense_20261029_140739/capture_manifest.json` and `capture_preview.jpg`.
- A previous feature-extraction process had stopped around image 375/694.
  No reconstruction report, trained scene, or quality result existed at the start.
  Resumed extraction/matching using the existing database, preserving its log.
- Initial GPU use: about 960 MiB of 6144 MiB including desktop applications.
- Capture challenges visible in the contact sheet: glass display cases,
  reflective objects, overexposed windows, blank walls, and motion blur.

## Procedure and evidence

### Camera recovery (completed RGB attempts)

```powershell
.\.venv\Scripts\python.exe -u scripts/prepare-realsense-scene.py reconstruct --threads 8 --train-width 848
```

Log: `data/realsense_20261029_140739/reconstruction_resume.log`.
CPU SIFT, measured RGB intrinsics, one OPENCV camera with fitted forward
distortion, sequential matching with overlap 12 and quadratic overlap,
incremental COLMAP mapping, followed by pinhole undistortion.
The stored RGB inverse-distortion coefficients are not copied into COLMAP's
forward distortion model.

Initial result: three separate models, with 113, 20, and 24 registered images.
Largest component: **113/694 (16.28%)**, 11,484 points, mean reprojection error
0.603 pixels. This is a partial room reconstruction.

Added top-30 nonlocal image candidates from 512 RootSIFT visual words with
TF-IDF weighting, verified the matches in COLMAP, and retried mapping with a
4-degree initial triangulation threshold and 0.5-degree point threshold:

```powershell
.\.venv\Scripts\python.exe -u scripts/recover-realsense-global.py
```

Result: largest component **151/694 (21.76%)**, 14,505 points, 0.661-pixel mean
reprojection error; nine components (151, 18, 20, 39, 34, 25, 35, 22, 19 frames).
Evidence: `global_reconstruction_report.json`, `global.log`, `retrieval_pairs.txt`
under the capture data directory. RGB-only mapping still fragments the room.

### Sensor calibration and depth alignment (completed)

```powershell
.\.venv\Scripts\python.exe -u scripts/prepare-sensor-depth.py
```

- Decoded Z16 CDR messages without ROS; paired nearest recorded RGB/depth times,
  requiring <=20 ms separation, and saved header-time differences as evidence.
- Used reference-to-color extrinsics, column-major rotation, translation
  approximately (-0.059013, -0.000047, 0.000360) meters.
- Applied inverse Brown-Conrady calibration correctly by numerically inverting
  its raw-pixel-to-ray mapping for projection, then used COLMAP undistortion.
- Rejected invalid depth, values outside 0.3-6 m, and 3x3 depth discontinuities
  larger than max(3 cm, 2% of depth). Used nearest-surface z-buffering.
- Fitted metric scale from **training images only**, using sparse camera-Z /
  sensor camera-Z ratios. Scale: **2.208888 COLMAP units/meter**, 45,064 pairs.
- Median relative ratio error: **1.57%**; **98.67%** within 10%; 90th-percentile
  per-frame median scale deviation: **2.92%**. Gate passed.
- Saved up to 4096 sensor samples/view in `sensor_depth/*.npz`. The explicit
  `--sensor-depth-dir` loader rescales Z with the parser's normalization, so
  training targets and normalized camera positions share units.
- Evidence: `sensor_depth_report.json`, `depth_alignment.log`.

### First controlled splat comparison (completed)

Common settings: 113 registered views, 848-pixel undistorted width, every eighth
view held out (15 test views), 3000 updates, seed 42, batch one, packed rendering,
SH degree 2, degree interval 700, 250,000 Gaussian cap, refinement from 300 to
2400, opacity reset every 2000, cached CPU images. Sensor-depth weight: 0.01.
Recorded initialization hashes and camera-sequence hashes match across runs.

| Run | Held-out PSNR | SSIM | LPIPS (lower better) | Gaussians | Peak PyTorch GiB | Training-loop seconds* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rgb_baseline` | 20.6597 | 0.75302 | 0.45801 | 249,305 | 0.29065 | 59.36 |
| `depth_baseline` | 20.7741 | 0.75339 | 0.45379 | 249,348 | 0.30189 | 65.30 |

*Saved checkpoint statistics exclude final evaluation/export and process startup.
Read the JSON statistics for exact timings. These are single-seed results,
not evidence of a universal depth advantage. Visible render pairs recover the
room doorway and display shapes, with blur, missing moving people, and limited
detail. Baseline peak observed total device use was about 1.3 GiB, including
desktop applications; PyTorch allocations do not include all device memory.

Artifacts: `results/realsense_20261029/{rgb_baseline,depth_baseline}/` contains
`cfg.yml`, `experiment.json`, `stats`, checkpoint, PLY and held-out comparison
PNGs (captured RGB on the left, rendered RGB on the right).

The first `rgb_initial` attempt was interrupted after profiling a Windows
worker-spawn stall. No completed metrics are attributed to it. Added configurable
data-loader workers and used **zero workers** for the completed runs. Added
Open3D for CPU pose-graph processing and py-spy for the stall diagnosis.

### RGB-D pose recovery (completed)

```powershell
.\.venv\Scripts\python.exe -u scripts/recover-rgbd-scene.py extract
.\.venv\Scripts\python.exe -u scripts/recover-rgbd-scene.py register
.\.venv\Scripts\python.exe -u scripts/recover-rgbd-scene.py optimize
```

Uses calibrated depth-supported SIFT feature positions, 3D rigid RANSAC on
COLMAP-verified pairs, a maximum-confidence pose tree, and CPU pose-graph
optimization with loop rejection. This can handle rotation and weak parallax
where RGB triangulation failed. Geometry coverage is measured below; final
photometric rendering quality is measured separately.
Full-rate depth projection was resumed with a 2x depth grid and eight inverse
distortion iterations to reduce CPU time, retaining completed frames. Depth
values are selected, not averaged. RGB images use one common undistortion map.

The successive pose experiments were:

| Pose method | Largest connected component |
| --- | ---: |
| Verified epipolar pairs, 3D-to-3D depth RANSAC | 56 frames |
| All candidate pairs with metric RANSAC | 71 frames |
| Additional depth-backed PnP for holes in target depth | 71 frames |
| Additional bidirectional low-resolution dense RGB-D links | 79 frames |
| Metric-scaled COLMAP segment constraints plus RGB-D | 175 frames |
| Higher-rate original RGB-D frames bridging failed 2 FPS gaps | **626 frames** |

Additional stages, in order after the initial `register`:

```powershell
.\.venv\Scripts\python.exe -u scripts/recover-rgbd-scene.py odometry
.\.venv\Scripts\python.exe -u scripts/recover-rgbd-scene.py anchors
.\.venv\Scripts\python.exe -u scripts/recover-rgbd-scene.py bridge
.\.venv\Scripts\python.exe -u scripts/recover-rgbd-scene.py optimize
```

The higher-rate stage reads intermediate RGB/depth frames at approximately
10 FPS from the original recording without exporting the full 37 GB capture.
It accepted 113 of 127 attempted gap bridges, requiring small motion per step,
a nondegenerate information matrix and forward/backward agreement. The dense
stage also uses depth-warp consistency. Metric SfM scales were fitted separately
for each segment, with median relative scale errors between 0.53% and 2.87%.

Final graph: **626/694 views (90.20%)**, 1939 optimized edges, with remaining
components of 36, 26, 2, 2, 1 and 1 frames. Median / 90th / 99th-percentile
translation constraint residuals: **3.54 / 18.93 / 62.16 mm**; rotation residuals:
**0.159 / 0.794 / 2.417 degrees**. These indicate internal agreement, not ground
truth absolute accuracy. A chain through weakly textured areas can still drift.
Unregistered filenames are listed explicitly in `rgbd/report.json`.

The final dataset uses one 848 x 477 calibrated, undistorted pinhole camera,
180,000 depth-initialized points (2.5 cm voxel selection, capped), and up to
4096 sensor-depth training samples per frame. Initial geometry/colors use only
the training-view subset; the holdout is every eighth registered view.
Pose estimation itself uses the complete capture, including held-out cameras.

**Export correction:** the first attempted broad-scene training run loaded stale
56-view COLMAP binaries because the exporter wrote new text into a directory
containing old binaries. Stopped it and retained its artifacts as an interrupted
attempt (`full_scene_depth30k`), corrected the exporter to call `read_text`
explicitly, and asserted image/point counts. Reloaded the binary export and
verified **626 registered views and 180,000 points** before starting the replacement
`full_scene_626_depth30k` run. Do not attribute the earlier run's metrics to
626 views. The `report_*.json` files preserve earlier graph attempts; early
dataset exports reused the same directory and are superseded.

Evidence directory: `data/realsense_20261029_140739/rgbd/` contains the final
report, pose graph, poses, per-stage reports, edges, calibrated RGB/depth
samples, and the COLMAP-compatible training dataset. `rgbd_export_verified.log`
records the corrected export.

### Higher-detail splat training (completed)

The fallback `global_native_depth15k` run uses 151 RGB-SfM views, undistorted
images up to native width 1280 (actual 1278 x 716 or 1277 x 717), SH degree 3,
a 700,000 cap and 15,000 updates. Its depth
check used 61,678 training scale pairs: 97.93% within 10%, median relative error
0.679%, and 3.08% 90th-percentile frame scale deviation. Sensor samples were
transferred from 848 to native image coordinates using the two pinhole
calibration matrices, with camera-Z values unchanged.

Native dataset preparation is captured in `scripts/prepare-native-realsense.py`
(equivalent to the preparation command used here; refuses to overwrite existing
trained inputs). Native run completed: **22.6995 dB PSNR, 0.79375 SSIM,
0.43143 LPIPS, 699,658 Gaussians, 1.07923 GiB peak PyTorch allocated** across
19 held-out views. Some text and reflections remain blurry. Its 22,004-second
checkpoint timer includes the long pause and concurrent work described below.

The main `full_scene_626_depth30k` run uses the verified 626-view dataset:

```powershell
& scripts/test-small-dataset.ps1 -Steps 30000 `
  -DataDir data/realsense_20261029_140739/rgbd/dataset `
  -ResultDir results/realsense_20261029/full_scene_626_depth30k `
  -ShDegree 3 -ShDegreeInterval 1000 -RefineStart 500 -RefineStop 15000 `
  -ResetEvery 3000 -Strategy budget -GaussianBudget 1000000 `
  -RecordExperiment -DataWorkers 0 `
  -SensorDepthDir data/realsense_20261029_140739/rgbd/dataset/sensor_depth `
  -DepthWeight 0.01 -SplitMode long_axis -ClonePolicy split -EdgeWeight 0.25 `
  -Milestones @(7000,15000)
```

This uses the local budgeted long-axis split adaptation, with error/edge
attribution. The cap limits live splats, while rasterization and structural
updates also consume memory. Actual GPU fit must be measured, not inferred
from the count. Saves intermediate checkpoints/metrics at 7k and 15k, and a
checkpoint/PLY at 30k. These different datasets and resolutions are not a
controlled quality comparison with the initial 113-view ablation.

Both high-detail jobs ran concurrently for part of their duration. The native
run log contains a large wall-clock timing jump (elapsed approximately 5h49
near 4k updates); its cause was not determined. Do not interpret its elapsed
time as continuous GPU training cost. The broad run's checkpoint timer is
1700.21 seconds (28m20s), including concurrent-job contention, image loading,
and intermediate evaluation; it is not an isolated compute comparison.

First main-scene checkpoint (7000 updates): **742,767 Gaussians**, held-out
PSNR **18.1031 dB**, SSIM **0.63210**, LPIPS **0.67280**, peak PyTorch allocated
**1.00898 GiB**. Held-out images recover the central exhibit and room structure,
with visible blur and artifacts around glass, fine text, doors, and moving
visitors. Broader coverage makes these scores incomparable to the smaller
113-view ablation. The later checkpoints are recorded below.

At 15,000 updates: **999,133 Gaussians**, PSNR **18.8336 dB**, SSIM **0.66428**,
LPIPS **0.57745**, peak PyTorch allocated **1.37098 GiB**. Central-exhibit
details improved relative to 7k; doorway reflections and moving visitors still
produce artifacts. Gaussian growth stops here; optimization continues to 30k.

At 30,000 updates: **999,133 Gaussians**, PSNR **18.8873 dB**, SSIM **0.66191**,
LPIPS **0.53941**, peak training PyTorch allocated **1.37098 GiB**. PSNR largely
plateaued after 15k, while perceptual similarity continued to improve. Retained
the sharper native model as a complementary result rather than claiming more
registered views automatically produces better image detail.

Environment saved in `data/realsense_20261029_140739/environment.json`:
Windows 11, Python 3.12.10, PyTorch 2.9.1+cu128 / CUDA runtime 12.8,
pycolmap 4.2.1, Open3D 0.20.0, GTX 1660 SUPER 6 GiB. `pip check` passed.
`dataset_verification.json` records the 547/79 train/holdout split, exported
binary hashes, source-code hashes, and unchanged recording size/mtime.
Installed `plyfile` for actual PLY reload validation; exporting alone did not
require it. The native dataset contains two fitted pinhole calibrations for
the same physical RGB stream; the full RGB-D dataset uses one measured camera.

Native artifact verification completed: checkpoint tensors finite, exported
**165,120,765-byte PLY** reloaded with all 699,658 splats, and a decoded H.264
comparison video (2560 x 718, 4 FPS, 37.75 seconds). See
`results/realsense_20261029/native_preview/{scene.ply,capture_path.mp4,comparison_sheet.jpg,verification.json}`.
Measured native rendering was 71.0 FPS mean / 13.93 ms median at 1280 x 718,
**while the full-scene model was still training**; this is not an isolated
viewer-speed claim. Peak inference PyTorch allocation was 0.35643 GiB.
Native previews retain sharper central-display detail, with narrower recovered
view coverage than the 626-view reconstruction. Keep both as distinct options.

Adapter verification: `pytest tests/test_realsense_adapter.py -q`: **3 passed**.
Covers CDR decoding, metric camera Z and extrinsics, inverse-distortion round
trip, and rigid pose recovery with 25% synthetic outliers. Pytest also reports
an existing unknown `env` configuration-option warning.

### Comparison design used

1. Ran a capped-budget RGB baseline, batch size one, packed rasterization.
2. Validated alignment and metric-to-SfM scale before sensor-depth training.
3. Compared held-out PSNR/SSIM/LPIPS and image pairs, kept both coverage/detail
   options, and checked actual model reload and GPU rendering.

Held-out views assess interpolation within this capture. Their camera poses are
estimated jointly by COLMAP, so they are not an independent geometry test.
Do not interpret a splat as a watertight mesh or assume unseen surfaces exist.

## Limits of the result

The central exhibit and room structure are recognizable, but glass reflections,
overexposed windows, fine lettering, moving visitors, and low-light noise still
limit fidelity. Some doorway views are noticeably blurry or contain floaters.
Depth and pose-graph checks do not establish survey-grade metric accuracy.
The reconstruction does not fill missing views or unseen surfaces with verified
geometry. No separate hand-held video, scanner ground truth, or independent
capture was available to validate global shape.

## Opening the scene

From this repository in PowerShell:

```powershell
# Broad reconstruction; the launcher reads the selected completed checkpoint.
powershell -ExecutionPolicy Bypass -File scripts/view-realsense.ps1

# Sharper 151-view native-resolution alternative.
powershell -ExecutionPolicy Bypass -File scripts/view-realsense.ps1 -Model native
```

Open `http://localhost:8080`. Run one viewer at a time, or add `-Port 8081`
for the second. `-Background` starts it without a visible terminal window.
The launcher prints the process ID for stopping a background viewer.

The native preview is already available:
[comparison sheet](results/realsense_20261029/native_preview/comparison_sheet.jpg),
[comparison video](results/realsense_20261029/native_preview/capture_path.mp4),
[PLY](results/realsense_20261029/native_preview/scene.ply).
Each preview pair shows captured RGB on the left and the splat on the right.
The captured-pose videos run at 4 FPS (approximately twice the selected 2 FPS
capture sampling); missing temporal intervals jump rather than invent views.

## References

Depth backprojection and stream extrinsics follow the
[RealSense projection documentation](https://dev.realsenseai.com/docs/projection-in-realsense-sdk-2-0/).
Extrinsic direction was checked against
[librealsense device.cpp](https://github.com/realsenseai/librealsense/blob/master/src/device.cpp)
and [ros_writer.cpp](https://github.com/realsenseai/librealsense/blob/master/src/media/ros/ros_writer.cpp).
The existing trainer's `--depth-loss` normally uses projected COLMAP sparse
points, not recorded sensor depth; any sensor-depth run must explicitly use an adapter.
