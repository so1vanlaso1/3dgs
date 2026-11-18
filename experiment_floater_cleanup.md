# RealSense floater cleanup trial

Date: 2026-10-30 (Asia/Saigon).

The user reported particles in the air and haze in the room reconstruction.
Tested post-training pruning on the latest 15 FPS model. The original checkpoint
and existing viewer were preserved. This trial does not retrain, change camera
poses, or change sensor-depth supervision.

## Result

Saved a mild pruned copy with **769,915 splats**, removing **228,051 (22.85%)**
of the original 997,966. This is an optional comparison model, not a confirmed
fix for the haze. Removed splats are not a measured count of actual floaters.

Visual inspection of nine identical-view comparisons shows that mild pruning
mostly preserves the original appearance, including substantial blur and bright
streaks near windows. It does not clearly eliminate the air-particle problem.
Stronger pruning introduces conspicuous dark gaps, mottling, and lost surfaces,
especially on floors and near doors. Do not use stronger deletion as a general
cleanup recommendation for this capture.

## What was tried

Loaded the source checkpoint:
`results/realsense_15fps_experiment/train30k/ckpts/final_post_update.pt`.

All filtering happens in the checkpoint's normalized coordinates:

1. Remove splats below an opacity threshold (sigmoid of stored opacity logits).
2. Remove splats whose largest axis scale exceeds a fraction of the dataset
   parser's scene scale. Axis scales are exponentials of stored log scales.
   Scene scale here is 1.3304576673, not a distance in meters.
3. Remove low-opacity spatial outliers: eighth-neighbor distance above the
   original cloud's 99.5th percentile, and opacity below 0.1. The distance
   threshold is 0.1085040938 in normalized coordinates. This outlier rule flags
   2,987 splats before overlaps with the other rules.

These are geometry/opacity heuristics. Dense floating clusters can survive, and
valid sparse surfaces or distant background splats can be deleted.

| Copy | Opacity minimum | Largest scale / scene scale | Remaining | Removed |
|---|---:|---:|---:|---:|
| Original | No filtering | No filtering | 997,966 | 0 |
| Mild | 0.005 | 0.10 | 769,915 | 228,051 |
| Balanced | 0.010 | 0.05 | 727,610 | 270,356 |
| Strong | 0.020 | 0.03 | 662,405 | 335,561 |

## Verification

Rendered all **79 existing held-out images** with identical poses, intrinsics,
resolution, SH degree, and black background for each copy. Also rendered the
viewer's initial camera, frame_004839.jpg, for a total of 80 views per model.

| Copy | PSNR, higher better | SSIM, higher better | LPIPS, lower better |
|---|---:|---:|---:|
| Original | 20.29982 | 0.69782 | 0.53138 |
| Mild | 20.20693 | 0.69637 | 0.53284 |
| Balanced | 19.81868 | 0.68677 | 0.54211 |
| Strong | 19.04496 | 0.66153 | 0.56247 |

The script selects the most pruning whose average PSNR drops at most 0.25 dB,
SSIM drops at most 0.01, and LPIPS rises at most 0.01. Only mild passed these
heuristic guards. All three mean metrics still got slightly worse with mild.
Only 18/79 views improved in PSNR; the worst view, frame_004044.jpg, lost
2.09 dB. Passing an average guard does not guarantee every view is preserved.

Verified finite source tensors and rendered pixels, exact tensor reload for
each saved checkpoint, finite PLY reload with the correct selected splat count,
and unchanged SHA-256 of the original checkpoint. Full per-view measurements,
thresholds, and hashes are in the JSON report.

No ground-truth empty-space labels are available. No assessment at the exact
free camera position from the user's screenshot was possible because that
camera transform was not recorded. Captured-view scores do not measure all
possible viewer angles.

## Files and commands

- [Comparison sheet](results/realsense_floater_cleanup_20261030/comparison_sheet.jpg):
  captured image, original, mild, balanced, strong, left to right.
- [Initial viewer camera comparison](results/realsense_floater_cleanup_20261030/comparison_frame_004839.jpg).
- [Measurements and hashes](results/realsense_floater_cleanup_20261030/report.json).
- Optional checkpoint: `results/realsense_floater_cleanup_20261030/mild.pt`.
- Optional PLY: `results/realsense_floater_cleanup_20261030/scene.ply`.
- Balanced and strong checkpoints are retained for comparison, not recommended.

Reproduce from the repository root:

```powershell
.\.venv\Scripts\python.exe -u scripts/clean-realsense-floaters.py
```

Open the mild copy at http://localhost:8082:

```powershell
.\scripts\view-realsense-cleaned.ps1 -Background
```

The existing original 15 FPS viewer uses http://localhost:8081. Both launchers
start at the same dataset camera, so compare before moving either camera.
Runtime status is saved separately in the cleanup output directory.

## Follow-up: disconnected Gaussian clusters

Tested the user's proposed removal of Gaussians that do not connect to others.
Both tests start from the original 997,966-splat 15 FPS checkpoint, not from
the earlier opacity-pruned copy. No opacity threshold is used in these tests.

First, grouped Gaussian centers into occupied voxels, connected the 26 touching
neighbor cells, and kept every component containing at least 500 original
splats. Keeping several large components avoids automatically deleting every
separate piece of furniture. This is approximate center connectivity, not exact
Gaussian ellipsoid intersection.

| Voxel width / scene scale | Removed splats | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| 0.016 (loose) | 29,050 | 19.54504 | 0.68397 | 0.54388 |
| 0.008 (medium) | 110,385 | 17.04873 | 0.56272 | 0.57232 |
| 0.004 (tight) | 282,197 | 14.02844 | 0.32425 | 0.61833 |

All settings failed the original image-quality guards. Sparse legitimate
surfaces also become disconnected when the connection distance is too small.

Second, tested a size-aware sparse graph. For each splat, checked its 12 nearest
centers. An edge exists if the centers are within the sum of the two largest
axis scales multiplied by a support multiplier (1, 2, or 3), capped at 0.04
times scene scale. The graph is treated as undirected. Retained all components
with at least 100 splats. Spheres enclosing the ellipsoids are only an overlap
approximation; they may bridge empty space, and 12-neighbor sampling can miss
some legitimate links. This is not a proof of physical surface connectivity.

| Support multiplier | Removed splats | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| 3 (loose) | 18,551 | 19.79230 | 0.69132 | 0.53652 |
| 2 (medium) | 33,710 | 19.78793 | 0.69125 | 0.53662 |
| 1 (tight) | 188,134 | 19.64201 | 0.68717 | 0.54000 |

All three settings also failed the guards. At multiplier 3, the largest
component contains 972,205 splats, approximately 97.4% of the entire model.
Connectedness alone is therefore a weak way to distinguish this model's haze
from its valid geometry. It does not establish that every splat in this large
component is valid, or that every removed component is a floater.

Each setting was rendered and measured on the same 79 held-out views, with
the initial viewer camera added to the comparison sheets. Checkpoint reloads,
finite pixels, and the original checkpoint hash were verified. Small synthetic
checks verified that both graph methods preserve two separate sufficiently
large clusters while removing an isolated singleton.

No candidate from either connectivity test was promoted. In these new output
directories, `scene.ply` is consequently the original fallback, not a cleaned
model. The explicitly named loose/medium/tight checkpoints are experimental
copies only; the existing viewers and original checkpoint are preserved.

- [Voxel comparison sheet](results/realsense_component_cleanup_20261030/comparison_sheet.jpg)
- [Voxel measurements](results/realsense_component_cleanup_20261030/report.json)
- [Size-aware comparison sheet](results/realsense_support_cleanup_20261030/comparison_sheet.jpg)
- [Size-aware shifted-view comparison](results/realsense_support_cleanup_20261030/freeview_comparison.jpg)
  shows original versus the loose trial, even though the trial failed the guards.
  These three camera offsets have no reference photos and do not reproduce the
  unknown camera transform in the user's original screenshot.
- [Size-aware measurements](results/realsense_support_cleanup_20261030/report.json)

Reproduce:

```powershell
.\.venv\Scripts\python.exe -u scripts/clean-realsense-floaters.py --components
.\.venv\Scripts\python.exe -u scripts/clean-realsense-floaters.py --support-components
```

For background on density-based clustering and its distance/minimum population
parameters, see the [Open3D clustering documentation](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html).
The experiments above use SciPy graph components rather than Open3D DBSCAN.

## Next experiment (not run)

A more targeted cleanup test would classify splats using multi-view free-space
evidence: only remove a suspected cluster if it lies consistently in front of
reliable measured surfaces in several relevant recorded depth views. Account
for coordinate transforms, depth holes, occlusions, reflective glass, and
alignment uncertainty. Connected artifacts can then be identified without
requiring that they be isolated from the rest of the scene. This method remains
unimplemented and unverified here.

This pruning trial is complete; the haze problem remains unresolved. The next
useful controlled trial is to retrain the same views and fixed poses with sensor
depth disabled, comparing against the existing depth weight of 0.01. If needed,
test 0.001 as a third condition. This has not been run in this trial. Fixed
RGB-D-derived poses still contain depth's influence even without depth loss.
If those comparisons retain the haze, inspect pose consistency and reconstruct
a smaller reliable RGB-only component before trying more aggressive pruning.
