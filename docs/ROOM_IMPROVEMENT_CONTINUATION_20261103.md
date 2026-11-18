# Existing room improvement: continuation

Date: 2026-11-03. User decision: **prioritize retaining the full room and clearly mark uncertain areas** if a fully verified clean room cannot be established. Source recording and baseline are unchanged.

**Later finding:** an independent RealSense SDK numerical cross-check identified an RGB inverse-Brown remapping discrepancy and led to a separate corrected-image, 477-view experimental RGB-D candidate. The calibration, trajectory and point-cloud status below describes the earlier baseline-based work; see the [SDK remap finding and candidate report](ROOM_SDK_REMAP_FINDING_20261103.md) for the newer evidence and limitations.

## 1. Repeatable free-camera baseline — done

[`render-room-free-camera-baseline.py`](../scripts/render-room-free-camera-baseline.py) rendered four short local sweeps around the baseline's recorded 43, 61, 68 and 110 second viewpoints: wide room, difficult doorway, exhibit and reflective display. Each has five camera poses, including four novel poses. Each sweep moves laterally up to 0.02 trainer-normalized units and yaws up to 2 degrees relative to its anchor. There are cuts between the four sweeps; this is **not** a continuous collision-checked walk around the room. These numerical movements are reproducible inspection choices, not guaranteed physically collision-free trajectories.

Artifacts: [20-view contact sheet](../results/room_improve_free_path_v02_20261103/contact_sheet.jpg), [inspection video](../results/room_improve_free_path_v02_20261103/inspection_segments.mp4), [exact matrices, intrinsics, settings and checkpoint hash](../results/room_improve_free_path_v02_20261103/path.json). All four center views are pixel-identical to the original saved baseline views; 20 video frames were saved and the video decoded. Candidate models with different normalization require a validated coordinate-frame alignment before using these world poses.

The first renderer attempt failed because the local CUDA toolkit was not configured in the process. The empty first output directory was retained. `scripts/run-room-free-path.ps1` applied the existing local MSVC/CUDA setup and `NUM_CHANNELS=3,4`; the separate v02 run succeeded. No baseline model was changed.

## 2. Expanded static structure inventory — done as an inventory

Reprojected original unfilled depth for 12 additional selected prepared frames in [structure projections](../results/room_improve_structure_projections_20261103/). Every frame reproduced the saved preprocessing samples. Added [ten annotated candidate structures](../results/room_improve_static_inventory_20261103/contact_sheet.jpg), with original message IDs, timestamps, rough regions and cautions in [inventory.json](../results/room_improve_static_inventory_20261103/inventory.json). They include the kiosk, black box, doorway, panel, exhibit backdrop, barrier post, figure, free-standing case, door column and a floor control. These regions contain background, glare or depth gaps; they are not ten validated alignment targets or independent objects at guaranteed distinct ranges.

Frame 000000 in the inventory is one of the 60 baseline held-out images, and previous diagnostics also inspected some held-out views; the held-out set is preserved against training, but these views should not be described as a completely blind evaluation set for human-guided choices. The 12 newly reprojected structure frames are not in that 60-image set. Validation targets should be frozen before future tuning, or a truly untouched evaluation subset should be reserved.

## 3. Calibration bookkeeping — consistent; physical alignment remains inconclusive

`scripts/audit-room-calibration-numerical.py` independently recomputes the scale/crop intrinsics from recorded values: max numerical difference `5.69e-14`, expected and saved dimensions both 834x463. Depth units both equal 0.001 m, and saved rotation/translation numerically match the recorded values under the existing column-major/depth-reference convention. RGB-depth timestamp gaps among 477 selected frames have median 7.244 ms and maximum 8.927 ms. [Full report](../results/room_improve_calibration_numerical_20261103/report.json).

These checks confirm bookkeeping, not physical transform direction, exposure synchronization, inverse-distortion model or RGB-depth boundary accuracy. The prior transition tests remain inconclusive; no correction, calibration pass, or evidence of a specific time error is claimed.

## 4. Camera trajectory — expanded check, no validated repair

`scripts/audit-room-revisits-split.py` examined 14 candidate RGB frame pairs; each pair's mutual-ratio SIFT matches were randomly split 60/40 **before** fitting the fundamental matrix. Ten pairs had enough matches for a fit; four did not. Matched points were not semantically masked, and check matches include outliers. Although several pairs have low median baseline epipolar residual, their tails can be large and some matches concentrate on planar or reflective features. For exhibit pairs 204/249 and 205/250, the fitting subset's homography inlier counts (79 and 72) nearly match fundamental inlier counts (82 and 73); that is a warning about planar ambiguity, not a proof of degeneracy. Other views and the original graph are not independent. Results and red check-match images: [split revisit checks](../results/room_improve_split_revisits_20261103/).

**Decision:** no globally validated pose correction. More spatially distributed static features and separate validation are still needed; no pose graph was changed.

## 5. Multi-view depth — disjoint source and check frames

`scripts/audit-room-disjoint-source-cloud.py` first built an experimental cloud from 30 source training frames and checked it using 30 separate training frames. This was useful for testing provenance, but its comparison with the original cloud did not establish full-room coverage. Its [results](../results/room_improve_disjoint_depth_cloud_20261103/) are retained.

To follow the user's full-room preference, `scripts/audit-room-full-source-cloud.py` sampled **387 source training frames** spread across the recording. It used 1,500 sampled depth points per frame, then selected one per 2.5 cm voxel and capped at 180,000. A disjoint group of 30 training frames was used only as depth-check frames; for every point, a check view must be more than five seconds from its original source frame. All 60 original held-out image identities are excluded from both groups. The source/check frames are distinct for depth sampling, but the original pose graph, calibration, scene and preprocessing are shared, and the graph may already use check images; they are not independent physical ground truth.

The separate sparse-depth check initially found 78,183 points with no match; sparse sampling limits coverage. We then reprojected **original unfilled depth** for all 30 check frames and repeated the check with exact rounded-pixel comparisons. A stricter second calculation uses only 3x3 neighborhoods with every projected pixel valid and local depth variation less than `max(0.03 m, 2% of depth)`. No holes were filled. For the 180,000-point experimental cloud:

| Check | No valid observation | At least one supporting view | At least 3 free-space conflicts and no support | At least 3 conflicts with support |
|---|---:|---:|---:|---:|
| Any valid unfilled projected depth | 34,141 | 47,569 | 3,162 | 1,032 |
| Stable 3x3 neighborhoods only | 50,612 | 41,453 | 1,596 | 296 |

Support and conflict use the existing exploratory 0.08 m and 0.15 m thresholds. Categories such as no observation, support and conflict are not all mutually exclusive; for example, some supported points also conflict elsewhere. Stable 3x3 evidence is a subset of the all-valid check. Depth occlusion, changing content, specular surfaces, source measurement errors and uncertain poses can all affect the labels. The stricter rule reduces comparisons near gaps and depth jumps but does not validate alignment or establish a point as incorrect.

[Eight examples](../results/room_improve_dense_conflict_examples_20261103/contact_sheet.jpg) show source and check projections selected for spatial spread and extreme conflicts; some include reflective displays, image blur and exhibit figures. These are illustrative rather than representative and do not justify automatic deletion. [Dense report and evidence](../results/room_improve_full_source_cloud_20261103/dense_check_report.json), [raw check projections](../results/room_improve_full_cloud_dense_checks_20261103/).

## 6. Whole-recording experimental point cloud — created, not accepted as improved

The 387-source-frame sample is available as [experimental colored point cloud](../results/room_improve_full_source_cloud_20261103/development_only_unfiltered.ply), with 180,000 points and original sampled RGB colors. A second [uncertainty-colored version](../results/room_improve_full_source_cloud_20261103/development_dense_stable_uncertainty.ply) keeps those same points: yellow means no stable check observation, gray a stable observation without support or three conflicts, green at least one support and fewer than three conflicts, orange both support and three or more conflicts, red three or more conflicts without support. These colors represent diagnostic evidence, not verified truth or a semantic mask.

Relative to the existing 180,000-point initialization, nearest-neighbor median distance is 3.35 cm in each direction and the original-to-candidate 90th percentile is 8.05 cm. This measures similarity of differently sampled point sets, **not** correctness, completeness or improvement. Sampling and voxel selection change point positions relative to the baseline even though no conflict-based deletion was used. The candidate is not established to cover every part of the room, and has not passed the RGB-depth or trajectory gates. It is not approved as a cleaned training initialization. [Coverage report](../results/room_improve_full_source_cloud_20261103/coverage_comparison.json).

[Four fixed-camera cloud comparisons](../results/room_improve_cloud_fixed_views_v02_20261103/contact_sheet.jpg) show prepared RGB, original and experimental point projections. Their one-pixel projection occupancy counts are similar in those views, but that only describes projected sampling and does not establish complete coverage or accurate geometry. The display expands points with 3x3 dilation, which can merge points and change colors. An initial one-pixel contact sheet was too dark to review at thumbnail size and is retained separately in `results/room_improve_cloud_fixed_views_20261103/`; the improved visualization is v02.

## Verification, provenance and next work

Checked that source/check/evaluation identities are disjoint; 200 sampled output points independently reproject to their recorded source samples with max `8.25e-13` px and `2.67e-15` m numerical differences. Reprojection is a provenance check, not physical accuracy. Both cloud PLYs were reloaded, and stable-3x3 observation counts were checked as subsets of all-valid counts. Thirty raw depth projections reproduced saved preprocessing values. Python scripts compiled; the two existing synthetic transition tests passed, with the existing pytest `env` configuration warning. [Verification](../results/room_improve_full_source_cloud_20261103/verification.json) and [code/artifact hashes](../results/room_improve_full_source_cloud_20261103/continuation_provenance.json).

Commands, run from the repository root with fresh output directories:

```powershell
.\scripts\run-room-free-path.ps1
.\.venv\Scripts\python.exe -u scripts/audit-room-rgbd-alignment.py --output results/room_improve_structure_projections_20261103 --names frame_000119.png frame_000131.png frame_000143.png frame_000179.png frame_000191.png frame_000228.png frame_000239.png frame_000252.png frame_000300.png frame_000347.png frame_000383.png frame_000443.png
.\.venv\Scripts\python.exe scripts/inventory-room-static-structures.py
.\.venv\Scripts\python.exe scripts/audit-room-calibration-numerical.py
.\.venv\Scripts\python.exe scripts/audit-room-revisits-split.py
.\.venv\Scripts\python.exe scripts/audit-room-disjoint-source-cloud.py
.\.venv\Scripts\python.exe scripts/compare-room-experimental-cloud.py
.\.venv\Scripts\python.exe scripts/audit-room-full-source-cloud.py
.\.venv\Scripts\python.exe scripts/compare-room-full-source-cloud.py
.\.venv\Scripts\python.exe scripts/verify-room-full-source-cloud.py
$roomCheckNames = 0..29 | ForEach-Object { 'frame_{0:D6}.png' -f (10 + 16 * $_) }
.\.venv\Scripts\python.exe -u scripts/audit-room-rgbd-alignment.py --output results/room_improve_full_cloud_dense_checks_20261103 --names $roomCheckNames
.\.venv\Scripts\python.exe scripts/audit-room-full-cloud-dense.py
.\.venv\Scripts\python.exe scripts/review-room-dense-conflicts.py
.\.venv\Scripts\python.exe scripts/compare-room-cloud-fixed-views.py
.\.venv\Scripts\python.exe scripts/write-room-continuation-provenance.py
.\.venv\Scripts\python.exe -m pytest tests/test_room_transition_profile.py -q
```

Next work: improve independent, distributed static correspondences and independently check calibration/alignment; inspect source and check views for flagged clusters while retaining uncertain areas. Decide thresholds and evaluate a filtered or fused candidate on separate regions only after those geometry checks. The currently exported experimental cloud is reviewable, but no point-cloud improvement has been verified. No training or original data/model modification occurred.
