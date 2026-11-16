# Clean a RealSense recording and create a 3DGS scene

From `D:\gsplat\gsplat` in PowerShell:

```powershell
.venv/Scripts/python.exe -u scripts/run-clean-realsense.py
```

Default sample: `20261030_111354.db3`. The runner prepares data, validates it,
trains 12,000 updates with sensor-depth supervision, exports and reloads a PLY,
and renders captured-image comparisons and a video. The original DB3 is read-only.
The existing portable CUDA/MSVC environment and Python environment are required.

For another recording, use fresh output directories:

```powershell
.venv/Scripts/python.exe -u scripts/run-clean-realsense.py `
  --recording my_capture.db3 --data data/my_capture_clean --result results/my_capture_clean
```

Preparation defaults: 3 FPS, width 848 before valid-border cropping, 0.3-6 m
depth, 20 ms maximum RGB/depth timestamp and header gaps. It selects the sharpest
acceptable candidate from up to three samples per interval. Exposure and blur
gates reject gross failures; images retain their observed exposure. Numerical
depth remains depth, not a colorized image.

For different preparation parameters, run preparation explicitly then the runner:

```powershell
.venv/Scripts/python.exe -u scripts/clean-realsense-scene.py all `
  --recording my_capture.db3 --output data/my_capture_clean --fps 5 --width 848
.venv/Scripts/python.exe -u scripts/run-clean-realsense.py --prepared `
  --recording my_capture.db3 --data data/my_capture_clean --result results/my_capture_clean
```

Stages: RGB selection/exposure gates -> RGB rectification and valid crop ->
depth masking/median denoise -> timestamp pairing -> calibrated depth alignment
and Z buffer -> COLMAP features/matching/poses -> depth-to-SfM scale gate ->
dataset validation -> depth-supervised 3DGS -> PLY reload and CUDA render QA.

If RGB-only coverage is below 90%, the runner also tries metric RGB-D pose
recovery using cleaned depth, robust 3D/PnP links and scale-gated SfM fragments.
Disconnected adjacent boundaries are also tried with every original recorded
RGB frame, using freshly cleaned/paired depth and bidirectional odometry gates.
It selects that dataset only if coverage improves and p90 pose-constraint
residuals are below 8 cm and 3 degrees. The initial RGB-only result is retained.
Selection is saved to `dataset_selection.json`. Explicit selection is available
with `--dataset-name dataset` or `--dataset-name rgbd/dataset`.

Recorded factory intrinsics/extrinsics are reused, with color inverse distortion
handled explicitly. IMU samples and per-frame associations are exported; they
are not fused into poses. Nearest timestamp pairing is not hardware synchronization.
No dynamic-object segmentation, temporal depth fusion or hole filling is applied.

Inspect these files in the data directory:

- `cleaning_report.json`, `rejected_frames.json`, `capture_manifest.json`:
  selection, thresholds, original timestamps and message provenance.
- `calibration_raw.json`, `calibration_clean.json`: source calibration and
  output pinhole camera, crop and metric stream transform.
- `clean_preview.jpg`, `depth_alignment_qa.jpg`, `clean_audit.json`:
  visual diagnostics and all-file numerical depth checks.
- `imu/`: decoded accelerometer/gyro CSVs, per-frame associations and checks.
- `reconstruction_report.json`: connected reconstruction coverage and errors.
- `sensor_depth_report.json`: metric-depth versus SfM scale consistency gate.
- `dataset/dataset_validation.json` or `rgbd/dataset/dataset_validation.json`:
  finite training samples, split and normalization for that specific dataset.
  Read `dataset_selection.json` in the result directory to identify the trained
  one; the original RGB-only validation report is retained separately.

In the result directory, inspect `status.json`, stage logs, `train/stats/`,
`preview/verification.json`, `preview/comparison_sheet.jpg`,
`preview/capture_path.mp4`, `preview/scene.ply` and `provenance.json`.
Success markers skip completed commands; changing settings requires a fresh result
directory. If a stage fails, inspect its log; the runner does not call that stage
successful or train past failed depth consistency checks.

Open the completed default sample interactively:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/view-clean-realsense.ps1 -Background
```

Then open `http://localhost:8083`. Other results can use `-ResultDir` and `-Port`.

Evaluation holds out every eighth registered image, with joint SfM pose estimates.
It measures image interpolation in this capture, not independent geometry accuracy.
Cleaning cannot recover clipped image detail, absent depth, moving leaves, water
reflections or surfaces never recorded. Read `experiment_realsense_cleaning.md`
for the actual sample findings and execution outcomes.

## Current full-rate run: every qualified recorded frame

For the same sample recording, the follow-up runner is:

```powershell
.venv/Scripts/python.exe -u scripts/run-realsense-all30.py
```

It checks all 6,996 recorded RGB frames individually, with no FPS-bin selection
or near-duplicate rejection. Each frame has an atomic decision/provenance record.
Only quality, synchronization and usable-depth failures reject a frame. Output
is separate under `data/realsense_20261030_111354_all30` and
`results/realsense_20261030_111354_all30`.

This run uses **20,000 updates and standard uncapped Gaussian densification**.
There is no maximum Gaussian count. Densification still follows its configured
schedule (refinement stops at update 6,000); the remaining updates optimize the
resulting splats. The completed older 3 FPS budget-strategy model is preserved.

The verified 3 FPS poses anchor the full-rate reconstruction of the exact same
recording. Additional poses are estimated from RGB-D matches/odometry, not
interpolated. Sparse pose-graph optimization handles the larger camera count.
The full-rate branch uses exact CUDA SIFT descriptor matching (L2 distance,
Lowe ratio and reciprocal matches), followed by metric geometric verification.
It retains all qualified frame files, reuses the original 88 heldout source views,
and excludes training within +/-0.2 seconds of their timestamps. Buffer views
are retained files, not rejected data.

The final side-by-side comparison uses identical heldout RGB targets. Changes
include frame density, strategy, Gaussian count policy and training length, so
results do not isolate an FPS-only effect or establish independent geometry
accuracy. Actual qualification, coverage and outcome counts are recorded in the
research log and result JSON files once their stages finish.

After the full-rate run completes, open its separate viewer:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/view-realsense-all30.ps1 -Background
```

Then open `http://localhost:8084`. The older 3 FPS scene remains on port 8083.
