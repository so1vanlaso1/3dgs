# Room improvement: steps 1–6 evidence and decisions

Date: 2026-11-03. Source: `20261030_134523.db3`. The user requested completion of the baseline, inspection, alignment, trajectory, multi-view depth and point-cloud stages. This report distinguishes completed diagnostics from gates that did not pass. No updated pose graph or cleaned training cloud has been validated.

## 1. Baseline and evaluation identities — verified

Recomputed the DB3 SHA-256 with `Get-FileHash`: `60e3f253051fc282a61af7f6c9c5eef40830a12d42566b9eba0992463115bc52`; it matches the baseline `provenance.json`. That file records a null initial hash, so matching it does not prove the source remained unchanged throughout the original training run. The 60 saved held-out image names, RGB/depth message IDs and timestamps matched the capture manifest. Nine saved RGB-camera inspection poses have finite 4x4 matrices, corresponding screenshots and an existing baseline checkpoint. These are nine recorded-camera views, not a separately recorded free-camera path. Full details and code hashes are in [verification provenance](../results/room_improve_multiview_diagnostic_20261103/verification_provenance.json).

## 2. Region inventory — available; revisit validation partial

The [existing inventory](ROOM_REGION_INVENTORY_20261103.md) covers the nine requested regions. It identifies, among other observations, a dark smeared area around 61 s, a missing moving visitor around 37 s, an exhibit and reflective cases. RGB-only revisit checks below identify candidate overlaps, but they do not yet establish independently validated 3D constraints or exhaustive coverage of 5–10 independent static structures.

## 3. RGB-depth alignment — inconclusive

`scripts/audit-room-transition-profile.py` defines an exploratory transition from the first observed 10% to 90% of the median difference between manually chosen interior depth strips. It counts missing samples in that band and compares the band to the strongest horizontal RGB gradient. The method does not fill gaps and the selected gradient may be texture or another surface. Development strips and scanlines were chosen after image inspection. The method is not a calibration error estimate, an absolute physical edge measurement, or a global acceptance test.

On 24 development rows from the box, kiosk and statue: 14 were ambiguous because their transition contained missing samples; 4 were rejected by the selected criteria; 6 were candidates without a detected hole. Even the latter have unverified edge correspondence. For instance, the kiosk depth band and strongest RGB gradient differ by several pixels; it is not established that they describe the same physical boundary. Data: [development rows](../results/room_improve_transition_profiles_20261103/development_rows.csv).

The algorithm was written before reviewing the three previously reserved frames. The reserved overlays were then reviewed and coordinates and rows chosen manually **after** inspection. On 15 rows, 7 had missing samples in the transition and 8 were rejected; none met the no-gap candidate rule. Frame 000004 only partially contains the kiosk and frame 000295 has limited usable depth in the chosen strips. These frames show related objects and views from the same capture, not independent scene or ground truth. [Reserved rows](../results/room_improve_transition_reserved_20261103/reserved_rows.csv); [depth projections](../results/room_improve_reserved_alignment_20261103/). All three projection runs reproduced existing cleaned-depth samples, which verifies processing consistency, not physical accuracy.

**Decision:** no consistent global correction is established; no alignment or time correction was applied. The gate remains open. The method was tested on synthetic shifted-depth, missing-depth and equal-depth examples (2 tests passed), but those tests do not prove real edge correspondence.

## 4. Camera trajectory — diagnostic only

`scripts/audit-room-revisits-rgb.py` used SIFT reciprocal ratio matching, fundamental-matrix RANSAC and a homography diagnostic on six pairs. The image-derived inliers were compared with the baseline pose's epipolar geometry; the median residuals below use the same fitted inlier set, not independent measurements. No masking guarantees static-only matches, and epipolar consistency alone does not establish correct translation scale or 3D geometry.

| Frames | Matches / fundamental inliers | Baseline pose median Sampson residual | Interpretation |
|---|---:|---:|---|
| 000089 / 000183 | 17 / 13 | 5.42 px | Match visualization contains obvious incorrect correspondences. Reject this fit as a pose constraint; its residual does not establish pose error. |
| 000110 / 000129 | 38 / 28 | 0.46 px | Some plausible overlap, concentrated on wall/display features; reflective or mismatched features remain possible. |
| 000204 / 000249 | 150 / 134 | 0.44 px | Many matches on the same exhibit/backdrop. Homography also has 134 inliers, suggesting a planar degeneracy risk. |
| 000249 / 000290 | 7 / not fitted | — | Too few reciprocal features for this diagnostic. |
| 000129 / 000143 | 57 / 43 | 0.32 px | Potential overlap, not an independent global geometry check. |
| 000204 / 000205 | 740 / 637 | 0.22 px | Nearby-frame control; no evidence of global consistency by itself. |

[Match images and machine-readable results](../results/room_improve_rgb_revisits_20261103/). These views could have contributed to the original reconstruction; matches were not masked, held out independently, or validated in depth. No candidate graph was optimized and no pose repair was accepted. More distributed static correspondences and an independent check are needed.

## 5. Multi-view depth consistency — screening completed, not validated filtering

`scripts/audit-room-depth-independent-groups.py` screened all 180,000 existing points using two disjoint groups of 30 prepared training frames each. Neither group includes any of the 60 held-out evaluation images. They are from the same recording, interleaved in time, and share poses and preprocessing; they are not independent physical ground truth. Sparse aligned depth samples and a two-pixel nearest-neighbor radius were used. Support means |point Z − depth Z| <= 0.08 m; free-space conflict means point Z is more than 0.15 m closer than the measured depth. These are exploratory thresholds, not validated error bounds.

In the development group, 604 points had at least three free-space observations and no support. Of those, the separate check group had at least one supporting observation for 198, at least one free-space observation for 452, and no matched sample for 73; the first two categories can overlap. This demonstrates conflicting evidence and incomplete observation, not that any points are verified floaters. No points were deleted. Full per-point arrays, frame identities and limitations: [multi-view report](../results/room_improve_multiview_diagnostic_20261103/report.json) and `evidence.npz`.

Reviewed [eight example point projections](../results/room_improve_multiview_examples_20261103/contact_sheet.jpg): four with check-group support and four with check-group conflicts. Examples were selected for spatial separation and for extreme residuals or close support; they are illustrative, not representative. Several project onto exhibit figures, reflective displays, or blurred views, so disagreement cannot be assigned to a floating point from these images alone. The original cloud was constructed from this capture, potentially including check-group frames: an apparently exact supporting observation may be a point's own source measurement, not independent confirmation. Example coordinates and source identities are in the adjacent `examples.json`.

## 6. Point-cloud output — diagnostic export, not a validated cleaned candidate

Exported [all 180,000 original points with diagnostic colors](../results/room_improve_multiview_diagnostic_20261103/diagnostic_all_points_colored.ply). Original positions are written as float32 and no point is removed. Red indicates 333 development-flagged points with a check observation and no check support; orange indicates 198 with check support; yellow indicates 73 without a check match. Other points retain their original color. Color labels are evidence categories, not claims of erroneous geometry. The PLY was reloaded and all vertex fields compared exactly with the written arrays.

**Decision:** do not use this diagnostic PLY as a cleaned initialization or claim it improves the baseline. A filtered/fused training candidate requires sufficiently validated alignment, poses, thresholds and independent retained-surface checks. No trained model, dataset, baseline cloud, camera graph or recording was modified; no training was run.

## Reproduction and verification

From the repository root, with fresh output directories, the executed commands were:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 20261030_134523.db3
.\.venv\Scripts\python.exe scripts/audit-room-rgbd-alignment.py --output results/room_improve_reserved_alignment_20261103 --names frame_000004.png frame_000145.png frame_000295.png
.\.venv\Scripts\python.exe scripts/audit-room-transition-profile.py
.\.venv\Scripts\python.exe scripts/check-room-reserved-transitions.py
.\.venv\Scripts\python.exe scripts/audit-room-revisits-rgb.py
.\.venv\Scripts\python.exe scripts/audit-room-depth-independent-groups.py
.\.venv\Scripts\python.exe scripts/export-room-diagnostic-cloud.py
.\.venv\Scripts\python.exe scripts/review-room-multiview-examples.py
.\.venv\Scripts\python.exe scripts/verify-room-steps-1-to-6.py
.\.venv\Scripts\python.exe -m pytest tests/test_room_transition_profile.py -q
```

Transition development ran before reserved overlays were opened; reserved projection extraction occurred earlier without visual inspection. Source-coordinate selection for the reserved profiles happened after those overlays were opened. OpenCV's RNG was seeded with 1234 for fundamental-matrix RANSAC. Analysis/export scripts compiled; two transition tests passed with one pre-existing pytest configuration warning about unknown option `env`. The script hashes and GPU information are recorded in verification provenance. An initial attempt to list the checkpoint under `ckpts` failed because it is under `train/ckpts`; the actual path was resolved from the inspection metadata and verified.
