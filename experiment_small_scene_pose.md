# Small stairs scene: pose reconstruction experiment

## 2026-10-30 — question and design

Problem: the courtyard splat loses sharp stairs, railings and plant detail.
Camera pose error was a hypothesis, not an established root cause. Test a small
section before changing the full reconstruction.

Selected `frame_000078.png` through `frame_000105.png` from the cleaned
`20261030_111354.db3` capture: 28 continuous views around the stairs. Selection
was visually checked in `results/realsense_small_pose_20261030/selection.jpg`.
This is a local view segment, not a segmented object; foliage and background
are still visible. No moving-object mask was invented or applied.

Outputs are separate in `data/realsense_small_pose_20261030` and
`results/realsense_small_pose_20261030`. Images/depth are linked read-only for
consumption; the source COLMAP database is opened read-only and selected cached
features/matches are copied into a new database. The DB3 and existing models
are not modified.

Two branches use identical source RGB and sampled sensor depth:

- `baseline`: inherited completed courtyard RGB-D camera poses.
- `refined`: fresh local exhaustive matching, incremental SfM, and robust
  bundle adjustment with fixed calibrated image intrinsics. No inherited pose
  anchors. Incremental SfM itself already performs bundle adjustment.

Both branches initialize from the same training-view depth/color sample IDs,
transformed using their respective poses; world coordinates and normalized
initial geometry therefore differ. Both use standard uncapped densification,
6,000 updates, batch size 1, seed 42, independent camera sampling, SH degree 3,
depth weight 0.01, and refinement through update 4,500. This compares pose
pipelines, including their associated geometry initialization; it does not
isolate extra bundle adjustment alone.

Explicit split: 23 training images, three heldout images (82, 90, 98), and two
buffer images (89, 99). Training frames within 0.3 seconds of a heldout image
are excluded. Pose reconstruction uses all selected images; evaluation is
image interpolation with jointly estimated cameras, not independent geometry.
The depth-based global scale fit uses exactly the 23 training images.
Before viewing fresh-run results, a static stairs diagnostic crop was specified
from reference image 90: pixel rectangle (620, 140)-(790, 320). Its metrics and
enlarged comparison supplement the full-image scores; they do not replace them.

## Pose reconstruction findings

- Fresh SfM registered all 28/28 views with 2,226 reconstructed sparse points.
- Additional robust bundle adjustment converged in 10 iterations in the final
  run; mean reprojection error changed from 0.74545 to 0.72358 pixels.
- Training-depth scale gate passed: 1,989 pairs, median relative error 3.98%,
  84.62% within 10%, frame-scale relative p90 9.38%. Scale before metricization
  was 2.0260 COLMAP units per meter. This is internal sensor agreement.
- Adjacent-motion differences between inherited and fresh cameras: median
  0.868 cm / 0.0335 degrees; p90 2.338 cm / 0.1594 degrees. This interval does
  not show a dramatic disagreement between the two pose reconstructions.
- On 31,192 geometric feature correspondences, pose-derived epipolar error
  p50/p90 changed from 0.31687/1.05096 to 0.30356/1.03478 pixels. This is a small
  improvement using correspondences available to reconstruction, not an
  independent validation result.
- Complementary neighboring-view depth warps did not improve: absolute depth
  error p50/p90 was 4.570/19.601 cm for inherited poses and 4.864/20.220 cm for
  fresh poses. Different overlap counts (43,601 vs 43,227), occlusions and
  unmasked foliage limit interpretation. Neither diagnostic proves the fresh
  poses are globally more accurate.
- Aligning the metric trajectories at their first common camera gives position
  differences p50/p90/max 8.12/17.20/24.92 cm and rotation differences
  0.138/0.396/0.902 degrees. Small per-step differences accumulate over the
  9.277-second segment. This is disagreement between estimates, not measured
  absolute error. See `trajectory_comparison.json` and `trajectories.png`.

## Problems and repairs

- First diagnostic shell command had nested quote parsing trouble; replaced it
  with a PowerShell here-string feeding Python.
- Initial extra BA hit its 100-iteration limit. Final run used a 300-iteration
  allowance and a finite function tolerance and converged after 10 iterations.
- Initial metric-model export failed because pycolmap requires the destination
  directory to exist. Added explicit directory creation and required both the
  pose report and exported model before considering reconstruction complete.
- Replaced the shared every-eighth scale-fitting helper with an explicit fit
  using this experiment's actual training names, avoiding heldout-depth use.
- PowerShell can return an error status for redirected native stderr despite
  Python completion; checked script status JSON and native training runner
  exit codes rather than treating COLMAP logging on stderr as a failed model.

## Reproduction

`.venv/Scripts/python.exe -u scripts/experiment-small-realsense-pose.py`

Stages are resumable; the script also accepts `--stage prepare`, `reconstruct`,
`diagnose`, `train`, `preview`, or `compare`. Training success markers check
the exact command. Changes to selection/settings should use fresh directories.

Execution and final comparison findings follow below after verification.

## Final results and decision

Both branches completed 6,000 updates. Independently rendered final post-update
checkpoints reproduced the trainer's results on exactly the same three targets:

| Pose branch | PSNR (higher) | SSIM (higher) | LPIPS (lower) | Gaussians |
|---|---:|---:|---:|---:|
| Inherited courtyard | 20.74396 | 0.602875 | 0.187114 | 1,718,623 |
| Fresh local SfM + BA | 20.71422 | 0.602454 | 0.183063 | 1,687,040 |

Fresh-minus-inherited: -0.02974 dB PSNR, -0.000421 SSIM, -0.004050 LPIPS.
This is mixed, with no convincing overall win. The static stairs crop was
worse on all three measures: 22.4137 -> 21.8583 dB, 0.75131 -> 0.73231 SSIM,
0.07560 -> 0.08223 LPIPS. Visual inspection of `stairs_detail.jpg` shows similar
blurred railing/stair details in both reconstructions. One full-image target
(frame 90) improved on all full-image metrics, while the other two lost PSNR;
that improvement does not imply its stairs crop improved.

Decision: keep the inherited-pose **small-scene control** as the default viewer
variant. Preserve the fresh local model as a completed trial. Neither replaces
the original full courtyard. This experiment weakens the claim that simply
rerunning local SfM/bundle adjustment will fix the visible problem. It does not
prove that camera poses are globally correct or irrelevant elsewhere.

The next isolated test should compare sensor-depth supervision against RGB-only
supervision on these same images/poses, then investigate exposure consistency
and reliable static-region depth if needed. Those tests were not executed here.
No claim that moving foliage, sensor noise, calibration or timing has been
individually corrected is supported by this experiment.

## Verification and artifacts

- `comparison.json` records exact per-view and stairs-crop metrics. Both
  branches have the identical camera sampling sequence, RGB file bytes and
  numerical sampled depth arrays; each completed exactly 6,000 updates.
- Source DB3 size and nanosecond modification time match the saved manifest.
  This run did not rehash the 25 GB DB3; preservation verification here is
  read-only processing plus matching source stat, not a new SHA256 comparison.
- Both final checkpoints and all preview pixels are finite. Both PLYs reloaded
  successfully: inherited 405,596,506 bytes; fresh 398,142,918 bytes.
- Both comparison videos fully decoded all 28 frames with finite pixels.
  See `video_verification.json`. These are captured-view path previews, not
  independent geometry validation or an interactive FPS benchmark.
- `matched_frame_000082.png.jpg`, `matched_frame_000090.png.jpg`, and
  `matched_frame_000098.png.jpg` show capture / inherited / fresh at identical
  target pixels. `stairs_detail.jpg` enlarges the preselected stairs crop.
- Both branch previews also contain `scene.ply`, `capture_path.mp4`, individual
  comparisons, and `verification.json`. Python compilation and PowerShell
  parsing passed; runtime reconstruction/training/export checks are the
  substantive verification for this experiment.
- Default small-scene viewer: `http://localhost:8085`, starting at frame 90.
  Restart with `powershell -ExecutionPolicy Bypass -File
  scripts/view-small-realsense.ps1 -Background`. To inspect the alternative,
  use `-Variant refined -Port 8086 -Background`.
