# Improve the existing room reconstruction with the hardware available

Date: 2026-11-03. Workspace: `D:\gsplat\gsplat`.

**Constraint: use the existing DB3. A new recording is currently unavailable.**

This is an execution plan, not a report of completed improvements. The aim is a cleaner point cloud and more consistent free-view 3DGS using the RealSense D455 data already recorded and the GTX 1660 SUPER with 6 GB VRAM. No purchase is required. Time and careful inspection go into alignment and data consistency before longer training.

## 1. What we are improving

Primary source: `20261030_134523.db3`.

| Existing artifact | Meaning |
|---|---|
| `data/realsense_20261030_134523_clean` | Prepared images, calibration, depth and reconstruction reports |
| `data/realsense_20261030_134523_clean/rgbd/dataset` | Current selected RGB-D reconstruction |
| `results/realsense_20261030_134523_120k` | Completed baseline and diagnostics |
| 4,764 RGB frames and 4,764 depth frames | Recorded source data available for reprocessing |
| 477 selected images at 3 FPS | Current reconstruction input |
| 180,000 initialization points | Raw reconstructed cloud currently used by the point viewer |
| 417 training / 60 held-out views | Existing model's image split |
| 1,063,961 Gaussians after 120,000 updates | Saved trained model |

The earlier courtyard recording `20261030_111354.db3` is a separate scene. Its revisit diagnostics are useful methodological examples, but its poses, calibration, anchors and reported displacement must not be applied to this recording.

Success means fewer duplicated surfaces, less floating geometry, sharper retained detail and stable appearance along a fixed free-camera inspection path. Existing held-out image scores are secondary evidence alongside surface alignment and free-view inspection.

## 2. Work in this order

```text
Preserve baseline and evaluation views
    -> locate bad regions and useful revisits in the recorded frames
    -> check RGB-depth alignment
    -> independently test and, if needed, repair camera poses
    -> screen depth using multiple views
    -> rebuild initialization
    -> compare small, matched training runs
    -> increase image density or training budget only when justified
    -> export the best verified point cloud and Gaussian model
```

**Do not train past a failed alignment check.** More splats or training cannot establish that the input geometry is correct.

## 3. Preserve a reproducible baseline

1. Keep the original DB3 read-only and retain the completed model, prepared dataset and diagnostics.
2. Write every candidate into a fresh `data/room_improve_*` or `results/room_improve_*` directory. Do not run recovery or bridge scripts on the baseline: they write graph/dataset outputs.
3. Save the source SHA-256, current script hashes, parameters and GPU information in each experiment's provenance. The baseline already has a recorded source SHA-256 in `provenance.json`; compare against it.
4. Record the exact 60 held-out source-frame identities, timestamps and original message IDs. Filenames can change after resampling, so filenames alone are insufficient.
5. Save camera position, rotation, FOV and aspect for a small fixed inspection path. Use identical poses and render settings for before/after comparisons.
6. Include a wide room view, a difficult exhibit, a reflective area and a revisit if one exists. Do not choose only attractive viewpoints.

If camera estimates change, report fixed-world renders after putting candidates into the same coordinate frame, plus captured-view comparisons using each candidate's corresponding camera estimates. These answer different questions. Do not compare normalized coordinates directly when normalization has changed.

**Output:** baseline manifest, held-out frame list, inspection poses and baseline screenshots.

## 4. Inspect what the recording can actually provide

Review the existing selected RGB images, depth overlays and capture preview. Then inspect extra raw frames around each problematic area; the discarded 90% may contain sharper views, less occlusion or intermediate motion useful for registration.

Create a short table:

| Region / timestamp | Visible problem | Evidence to inspect | Next action |
|---|---|---|---|
| Example: display corner | Doubled edge | Same static corner from separate passes | Test pose agreement |
| Example: bright opening | Missing/unstable depth | RGB clipping and aligned depth | Exclude unreliable depth; retain useful RGB |
| Example: moving visitor | Ghost geometry | Frames before/during/after visitor | Select static observations or create masks |

Use real timestamps and screenshots when filling the table. Include examples of good regions as controls.

Select roughly 5–10 static structures across the room for alignment inspection. Prefer corners and objects at different depths rather than only a flat floor. This is a practical starting sample, not a statistical accuracy guarantee.

If there are no repeat views of a region, mark that limitation. Extra processing can use recorded evidence more effectively, but cannot observe an unseen backside or restore clipped RGB detail.

**Output:** region/timestamp inventory and candidate revisit pairs. No point deletion yet.

## 5. Verify RGB-depth alignment before using depth as evidence

The current pipeline already undistorts/crops RGB, applies the recorded depth-to-color transform, masks discontinuities and pairs frames by timestamp. Check these operations visually and numerically on this source rather than assuming passing bounds means correct geometry.

1. Inspect `calibration_raw.json`, `calibration_clean.json`, `depth_alignment_qa.jpg` and `cleaning_report.json`.
2. Overlay aligned depth boundaries on RGB at static corners, near objects and far walls, including frames during turns.
3. Check metric depth units, output intrinsics, crop offsets and transform direction. A camera crop must update the principal point.
4. Check whether mismatch grows during motion. Timestamp gaps currently reach about 8.93 ms; pairing is not hardware synchronization.
5. If consistent offsets appear, test a separate alignment/time-offset candidate using recorded data. Validate on frames excluded from selecting the correction.
6. Keep missing depth invalid. Do not make reflective glass or missing surfaces appear measured through aggressive filling.

The D455's published ideal range is 0.6–6 m. The current preprocessing accepts 0.3–6 m; that mask does not establish reliable measurements across that entire interval. Inspect the actual samples. A narrower range can be tested as a candidate, with the amount of discarded useful geometry reported. [RealSense D455](https://www.realsenseai.com/products/real-sense-depth-camera-d455f/)

SDK depth filtering is available, but filter settings require evaluation. Avoid applying temporal persistence blindly to moving-camera frames: check for stale depth and edge trails. Preserve unfiltered numerical depth for comparison. [RealSense filtering guidance](https://dev.realsenseai.com/docs/depth-post-processing-for-intel-realsense-depth-camera-d400-series/)

**Pass:** no systematic unexplained RGB-depth offset on the inspected static structures. Record remaining error versus distance and motion; select an acceptable tolerance based on the detail we want to preserve.

## 6. Check and repair the camera trajectory

This is the highest-priority geometry investigation, not a confirmed diagnosis of every defect.

Current evidence: RGB-only reconstruction registered 301/477 images in its largest model; RGB-D recovery connected 477/477. The graph's p90 fitted residual is about 5.15 cm and 0.924 degrees. Neither complete registration nor low fitted residuals proves globally correct poses.

For each useful revisit or overlapping region:

1. Match static features across separated times, excluding people, reflections and other changing content.
2. Require correspondences spread over the image and over multiple depths. A floor-only match is weak evidence for a complete room correction.
3. Estimate a candidate relative transform using robust geometry and valid depth where available.
4. Check reverse-direction agreement and nearby frame pairs. A candidate must agree with multiple neighboring observations rather than one convenient pair.
5. Reserve independent correspondences or nearby frames for validation. Do not validate only against the matches used to fit the transform.
6. Compare the candidate transform with the current trajectory. Investigate large disagreement; do not automatically reject a plausible loop just because the old graph places its endpoints far apart.
7. Add only supported constraints to a separate graph. Re-optimize and inspect all tested regions, including areas that were already good.
8. If appropriate, test bundle adjustment with verified image tracks and suitable camera constraints. Do not assume the existing RGB-D pose graph already performs a full global landmark bundle adjustment.

Recorded IMU is currently exported but not fused. Adding visual-inertial optimization is an optional later engineering project: timing, biases, calibration and coordinate conventions need validation. Do not make it a prerequisite for this first improvement attempt.

**Pass:** independent revisit checks improve or remain consistent; static surfaces agree across passes without creating new failures elsewhere. Report median and tail errors for each region, not only a single scene average.

**If this fails:** spend effort on intermediate recorded frames, correspondence review and constraints. Stop the training branch until the geometry is acceptable or explicitly restrict the reconstruction to a reliable subregion.

**Output:** candidate poses, accepted/rejected constraint evidence, independent alignment report and before/after screenshots.

## 7. Improve depth consistency and initialization

Only do this after the preceding checks. A point can appear inconsistent because its pose is wrong.

The current initialization samples individual depth maps, voxel-selects points and caps the result at 180,000. It does not require every retained point to have multi-view support.

Proposed replacement:

1. Transform valid depth samples using the validated poses.
2. Test their depth agreement in several suitable overlapping views.
3. Distinguish three cases: support near the observed surface; free-space conflict in front of a measured surface; occlusion behind a nearer surface. Occlusion and missing depth are not automatic rejection evidence.
4. Use a tolerance that accounts for distance, pose uncertainty and edges. Tune on development regions and validate elsewhere.
5. Conservatively flag repeatedly conflicting samples with little independent support. Preserve thin features and sparsely observed surfaces for manual review.
6. Fuse accepted samples into a voxel/surfel cloud, or test TSDF fusion for surfaces where appropriate. Use CPU processing, local regions and bounded memory. Do not assume the entire room must fit densely on the GPU.
7. Retain support counts, conflict counts, source frames and rejected-point reasons so filtering is auditable.
8. Compare against the existing initialization at similar point counts. Extra points alone do not demonstrate better geometry.

The existing 2,513 unsupported conflict candidates are investigation leads. They are not a validated list to delete. Previous component-based pruning on another recording reduced image quality; that outcome is a reason to test conservative alternatives, not proof that every filter fails here.

People can be handled by selecting unobstructed recorded views, or by masks consistently applied to feature estimation, depth fusion and training. The current main path has no general dynamic masking stage; mask support must be implemented and verified before claiming masked training. Areas hidden in every frame remain unknown.

**Pass:** fewer independently verified floating/duplicate structures, while retaining the static surfaces and thin details used as controls.

**Output:** original and candidate clouds, uncertainty/support information and fixed-view comparisons.

## 8. Run small, controlled training experiments on 6 GB

Train one model at a time. Stop other GPU-heavy reconstruction/rendering jobs during measurements. Keep the currently requested point viewer available unless it materially affects the measurement; record its state.

Suggested starting profile, to be measured on this machine:

| Setting | Initial choice | Reason |
|---|---|---|
| Image width | Existing 848 preparation width | Reuse known working image scale |
| Batch size | 1 | Existing launcher setting |
| Image caching | Off; zero data workers initially | Existing low-memory path |
| Training length | 12,000 updates | Early comparison before repeating 120,000 |
| Saved checkpoints | 3,000, 6,000 and 12,000 | Observe improvement/regression over time |
| SH degree | 3 | Match existing appearance capacity initially |
| Densification | Start 500, stop 6,000 | Controlled starting schedule already used locally |
| Gaussian budget | Test 400,000 cap with the existing budget strategy | Practical resource experiment, not a proven optimum |

The budget strategy changes densification behavior relative to the original default strategy. Use the same strategy and cap for every member of a controlled comparison. A capped candidate versus the old uncapped 120k model is an overall workflow comparison, not an isolated pose or depth ablation. A cap does not guarantee a particular VRAM peak.

Run sequentially, only as the evidence justifies:

| Experiment | Change | What it answers |
|---|---|---|
| T0 | Original prepared data under the new common short-run profile | Establish a fair training control |
| T1 | Validated/repaired poses; same source images and depth policy; regenerate initialization in those poses | Does corrected reconstruction improve the result? |
| T2 | T1 plus validated depth filtering/fusion initialization | Does the geometry cleanup help? |
| T3 | Best prepared data, depth supervision off versus weight 0.01 | Is depth supervision helping on that data? |
| T4 | Add suitable recorded frames; preserve exact evaluation targets | Does additional recorded view coverage help? |

T1 changes the reconstruction, including pose-dependent initialization; it does not isolate the effect of pose alone. T2 may bundle depth selection and fusion unless tested separately. Log that scope honestly. Skip steps whose prerequisite has no supported improvement.

The launcher defaults to holding out every eighth image. **When resampling frames, that default does not preserve the original evaluation set.** A source-identity split override and temporal exclusion must be implemented/verified before calling a higher-FPS experiment matched. Consider a +/-0.2-second exclusion around held-out images as a starting policy, and report remaining training coverage. Hold out whole revisit/trajectory portions separately when evaluating wider generalization.

If VRAM is insufficient, reduce the candidate cap first, then image resolution or SH degree. Change the common comparison profile for both runs and label it. Do not silently compare different resolutions or let other jobs distort timing.

**Pass:** candidate improves the declared geometry/visual target without unacceptable regression in held-out views or good control regions. Report GPU memory, time and Gaussian count as well as quality.

## 9. Commands already supported by the repository

Run from `D:\gsplat\gsplat` in PowerShell. These are examples for fresh candidate directories. Their arguments were checked against source; the improvement experiments have not been executed.

Prepare a separate 3 FPS candidate without starting training:

```powershell
.\.venv\Scripts\python.exe -u scripts/clean-realsense-scene.py clean `
  --recording 20261030_134523.db3 `
  --output data/room_improve_v01_3fps --fps 3 --width 848

.\.venv\Scripts\python.exe -u scripts/clean-realsense-scene.py reconstruct `
  --recording 20261030_134523.db3 `
  --output data/room_improve_v01_3fps
```

If RGB-D recovery is needed, these commands write into that candidate:

```powershell
.\.venv\Scripts\python.exe -u scripts/recover-clean-rgbd.py `
  --root data/room_improve_v01_3fps

.\.venv\Scripts\python.exe -u scripts/bridge-clean-rgbd.py `
  --root data/room_improve_v01_3fps

.\.venv\Scripts\python.exe -u scripts/clean-realsense-scene.py validate `
  --output data/room_improve_v01_3fps --dataset-name rgbd/dataset
```

Those scripts reproduce the existing recovery approach. They do not automatically implement the stronger independent loop checks or multi-view fusion described above. The validation command checks finite samples and the current split; it is not the geometric acceptance test.

After the alignment gates pass, a common 12k training profile can be run on a chosen dataset:

```powershell
& .\scripts\test-small-dataset.ps1 `
  -DataDir data/room_improve_v01_3fps/rgbd/dataset `
  -ResultDir results/room_improve_v01_depth_12k `
  -Steps 12000 -Milestones 3000,6000 `
  -ShDegree 3 -ShDegreeInterval 1000 `
  -RefineStart 500 -RefineStop 6000 -ResetEvery 3000 `
  -Strategy budget -GaussianBudget 400000 `
  -SensorDepthDir data/room_improve_v01_3fps/rgbd/dataset/sensor_depth `
  -DepthWeight 0.01 -RecordExperiment -DataWorkers 0 -NoCacheImages
```

For the no-depth comparison, use the identical dataset/settings and a fresh result directory; omit both `-SensorDepthDir` and `-DepthWeight`. This removes sensor-depth supervision while retaining the same depth-derived initialization, so it does not test removal of all depth information.

Use the same short profile on the original baseline dataset in a fresh result directory for T0. Do not overwrite the existing 120k run. Do not use the automatic long-run driver to bypass the manual geometry checks.

## 10. Only then try more of the existing frames

Start with targeted extra frames around turns, failed boundaries and useful revisits. They may provide more value than uniformly processing every frame.

For a broader sampling experiment, prepare a separate 5 FPS candidate before considering 10–15 FPS. These are trial levels, not guaranteed quality improvements. Increasing image count increases matching and pose-graph cost; the current small-graph solver should not automatically be scaled to thousands of frames. A sparse solver and graph-memory check are needed for a large branch.

Keep frame identities, calibration and source provenance consistent. Reuse anchors only from this exact recording and verify their correspondence. Re-estimate added camera poses from evidence rather than interpolating and calling them measured.

Changing FPS also changes optimization coverage per image at a fixed number of updates. Report both equal-update and, when useful, comparable-view-exposure budgets; neither is automatically equal compute. Track elapsed time.

Do not increase resolution, FPS, Gaussian cap and training steps together if the aim is to understand which improvement helped.

## 11. Acceptance checklist and research log

For every candidate, save:

- Source hash, frame identities, calibration, script hashes, full command and random seed.
- The problem being tested and one declared success criterion before running.
- Accepted/rejected frames, constraints and depth samples with reasons.
- Independent alignment evidence, including failure cases and uncertainty.
- Identical-view screenshots and a repeatable free-camera path.
- Held-out PSNR, SSIM and LPIPS where available, plus per-view failures.
- Elapsed time, peak GPU memory, final point/Gaussian count and any OOM/interruption.
- Decision: accepted, rejected, inconclusive or blocked; record why and the next action.

Example entry:

```text
Experiment:
Baseline and candidate paths:
Problem / hypothesis:
Changed factors:
Fixed factors / held-out source IDs:
Command and provenance:
Alignment result:
Depth / cloud result:
Held-out image result:
Free-view result:
Time / peak memory / count:
Failures and limitations:
Decision and next step:
```

Extend training only if saved checkpoints show useful progress. Test 20k–30k before repeating 120k. If quality stalls while duplicated geometry remains, return to the failed geometry check.

## 12. Optional final artistic point-cloud branch

After alignment and fusion are verified, a cleaned fused cloud can already be exported for viewing. If the statue video's surface-point look is desired, build and inspect a mesh from reliable fused geometry, sample colored surface points, and add Blender materials and a camera path. Work in chunks and use modest point density on 6 GB.

Meshing and artistic effects are separate deliverables. They can improve presentation without improving metric accuracy; retain the raw/corrected comparison evidence.

## First work session

1. Freeze the baseline frame list and inspection poses.
2. Inventory bad regions using existing previews and recorded frames.
3. Verify depth overlays on static structures.
4. Find and independently check distributed revisits in this specific recording.
5. Decide whether pose correction is supported; document the evidence before changing the graph.

These steps are the immediate priority. New training starts after they establish a defensible geometry candidate.

## Supporting project evidence

- [Current pipeline diagnosis](PIPELINE_DIAGNOSIS_20261102.md)
- [Video versus current pipeline](VIDEO_PIPELINE_COMPARISON_20261103.md)
- [Existing commands and preprocessing](realsense_clean_pipeline.md)
- [Research log](EXPERIMENT_LOG.md)
- [Latest baseline provenance](../results/realsense_20261030_134523_120k/provenance.json)
- [Latest RGB-D report](../data/realsense_20261030_134523_clean/rgbd/report.json)

Advanced pose validation, support-aware fusion, dynamic masking and a fixed-source split override are proposed engineering tasks, not completed capabilities. Expected gains must be measured; no numerical improvement is promised.
