# 3D Gaussian Splatting research progress report

**Coverage:** Documented experiments through 4 November 2026. Compiled on 6 November 2026 from saved project logs; experiments were not rerun. Links provide detailed settings, results, artifacts and limitations.

## 1. Objective and scope

The project extends the baseline gsplat repository in two main directions: improving Gaussian allocation and model efficiency, and reconstructing personal RealSense RGB-D recordings and ordinary videos on a GTX 1660 SUPER with 6 GB VRAM. Supporting work covers data preparation, sensor calibration, camera recovery, depth supervision, interactive viewing, reproducibility and diagnostics.

Problems investigated include large Gaussian counts, fragmented camera registration, blur and missing details, uncertain RGB–depth alignment, pose drift, floaters, reflective surfaces and moving people. Results below include rejected and inconclusive experiments. PSNR and SSIM are higher-is-better; LPIPS is lower-is-better. Held-out image quality does not alone prove accurate 3D geometry. Comparisons are restricted to their stated settings and views.

## 2. Gaussian allocation and densification

### 2.1 Budgeted allocation: A01–A03

**Problem.** Default densification can produce a large model. The goal was to reduce Gaussian count while retaining quality.

**Method.** Implemented BudgetStrategy while preserving the original strategy. A quadratic schedule limits growth toward a 700,000-Gaussian cap. Position-gradient eligibility and ranking were extended with contribution-weighted RGB reconstruction error and optional ground-truth edge detail. Signals are normalized and evidence is reset after structural growth. Existing pruning and optimizer bookkeeping are reused. The cap bounds live count, not temporary replacement memory. The method is inspired by Taming 3DGS but does not reproduce its complete method.

**Experiment.** Room, 272 training and 39 held-out images, 1,557 × 1,038, 7,000 updates, one seed; B00 was a saved baseline.

| Variant | Gaussians | PSNR | SSIM | LPIPS | Peak tensor GiB |
|---|---:|---:|---:|---:|---:|
| B00: original strategy | 841,017 | 30.0335 | 0.90206 | 0.21165 | 1.280 |
| A01: budget + gradient | 699,173 | 29.7823 | 0.89855 | 0.21555 | 1.101 |
| A02: + attributed error | 699,266 | 29.8630 | 0.90130 | 0.21299 | 1.104 |
| A03: + edge detail | 699,207 | 30.0166 | 0.90117 | 0.21405 | 1.116 |

Error attribution improved all three metrics over gradient-only selection. Edges improved PSNR over A02 while slightly worsening SSIM and LPIPS. A03 used 16.86% fewer Gaussians than B00 and came within 0.017 dB PSNR, but its SSIM and LPIPS remained slightly worse. This single-seed result does not establish equivalent quality.

### 2.2 A04: Long-Axis Split

**Method.** Combined A03 ranking with symmetric children placed along a parent's longest axis, modified scale and opacity, and copied rotation and spherical-harmonic appearance. Selected small parents are also split rather than cloned. A frozen-checkpoint diagnostic found less immediate disturbance in its tested setting; this does not prove faster recovery.

| Variant | Gaussians | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| B00 | 841,017 | 30.0335 | 0.90206 | 0.21165 |
| A03 | 699,207 | 30.0166 | 0.90117 | 0.21405 |
| A04 | 597,989 | 30.5058 | 0.90561 | 0.20486 |

Against B00, A04 used 28.90% fewer Gaussians, gained 0.4723 dB PSNR, and improved SSIM and LPIPS in this historical Room run. Several split properties changed together.

### 2.3 Multiscene and repeated-seed experiments

Frozen A04 was compared with B00 after 7,000 updates on three additional indoor scenes, one seed each.

| Scene | B00 → A04 Gaussians | Reduction | PSNR change | SSIM change | LPIPS change |
|---|---:|---:|---:|---:|---:|
| Counter | 749,489 → 569,520 | 24.01% | +0.2641 dB | +0.00336 | -0.00708 |
| Kitchen | 1,066,757 → 699,357 | 34.44% | +0.2599 dB | +0.00042 | -0.00153 |
| Bonsai | 1,017,191 → 695,971 | 31.58% | +0.1522 dB | +0.00093 | -0.00215 |

A later, separate campaign used three paired seeds and independent camera-sampling RNG, with matched initialization and camera sequences. Historical runs were excluded from its averages.

| Scene | Mean A04 − B00 PSNR | Paired 95% interval | Mean count reduction |
|---|---:|---:|---:|
| Room | +0.5688 dB | [0.3599, 0.7777] | 28.37% |
| Counter | +0.1361 dB | [0.0807, 0.1916] | 24.25% |
| Kitchen | +0.1297 dB | [-0.1458, 0.4052] | 35.27% |
| Bonsai | +0.0298 dB | [-0.7378, 0.7975] | 32.31% |

Room and Counter showed more consistent PSNR gains; Kitchen and Bonsai intervals include zero. Intervals from three seeds are imprecise, and outdoor generalization remains untested. The later completed campaign supersedes the earlier snapshot describing some work as skipped.

### 2.4 Ablation, budget sweep, tuning and failure analysis

**Ablation.** On Room, three seeds compared random versus long-axis operator bundles and cloning versus splitting small parents. Mean PSNR was 30.1286 dB for random/clone, 30.2922 for random/split, 30.4250 for long-axis/clone and 30.6327 for long-axis/split. Both the operator bundle and clone replacement were associated with gains. Placement, scale and opacity were not individually isolated.

**Budget sweep.** Room seed 42: caps 400k, 700k and 1M yielded final counts 399,842, 597,975 and 604,705 and PSNR 30.5461, 30.6007 and 30.6049. The cap is an upper bound, not a required final count.

**Tuning.** A development-view grid selected split distance 0.30 and opacity multiplier 0.40 using Room seed-42 PSNR. Confirmation PSNR changes were mixed and LPIPS worsened in all three reported checks; this is not an established untouched-test improvement.

**Visual failures.** Crop analysis found local regressions despite improved averages. A04 had higher full-view MSE on 20/35 Kitchen and 14/37 Bonsai views in the later comparison. Selected examples illustrate gains and failures rather than a random sample.

### 2.5 CCD-GS mechanism pilot

Bounded pose probes and split/no-split continuations tested a pose-corrected gradient-persistence score against simpler gradient, multiview and additional pose-compute controls. None of 36 checkpoint blocks passed the prespecified practical mechanism gate. Rankings sometimes changed, but a useful advantage beyond simpler controls was not established.

**Evidence:** [Experiment log](EXPERIMENT_LOG.md), [budget strategy](BUDGETED_DENSIFICATION.md), [additional scenes](MULTISCENE_A04.md), [completed campaign](../results/research_completion_20261027/REPORT.md), [CCD pilot](../results/research_completion_20261027/ccd/REPORT.md).

## 3. RealSense RGB-D pipeline

### 3.1 Ingestion, calibration and depth

Built DB3 extraction and RGB/depth synchronization, calibration and undistortion, depth-range and discontinuity filtering, sensor-depth sampling and training integration, depth-based initialization, and reconstruction, training, export and viewer scripts. Camera recovery uses depth-supported features, 3D RANSAC, PnP, dense odometry, SfM segment constraints, intermediate-frame bridging and robust pose-graph optimization. IMU was exported/associated but not fused into poses.

In a controlled 113-view experiment on recording 20261029_140739, after 3,000 updates on the same 15 held-out images, RGB-only PSNR / SSIM / LPIPS was 20.6597 / 0.75302 / 0.45801; depth supervision at weight 0.01 gave 20.7741 / 0.75339 / 0.45379. This is a small single-seed result.

### 3.2 Broad and native-resolution reconstructions

RGB-only mapping initially registered 113/694 selected views in its largest component; retrieval-assisted mapping reached 151/694. RGB-D links and higher-rate intermediate bridges yielded 626/694 connected views. An initial broad run loaded stale binary exports; it was stopped, and explicit text reading and image/point count checks were added before replacement training.

| Model | Registered views | Updates | Gaussians | PSNR / SSIM / LPIPS |
|---|---:|---:|---:|---|
| Broad scene | 626/694 | 30,000 | 999,133 | 18.8873 / 0.66191 / 0.53941 |
| Native-resolution alternative | 151/694 | 15,000 | 699,658 | 22.6995 / 0.79375 / 0.43143 |

The broad model offers greater registered-view coverage; the native alternative is sharper in its smaller region. Their images, resolutions and test sets differ, so these scores cannot directly rank them. Registered-view percentage is not percentage of physical room surfaces recovered.

### 3.3 Higher-rate 15 FPS experiment

Selected 5,204 frames while retaining original physical comparison images; rebuilt SfM anchors on 694 baseline observations, added RGB-D constraints and bridges, and used a sparse robust solver. An initial RGB-D-only graph fragmented; the final graph connected 5,201/5,204 frames. Both 30k-update models were evaluated on the same 79 original held-out images with a temporal exclusion buffer.

| Metric | 2 FPS baseline | 15 FPS pipeline | Change |
|---|---:|---:|---:|
| PSNR | 18.88727 | 20.29982 | +1.41256 dB |
| SSIM | 0.66191 | 0.69782 | +0.03591 |
| LPIPS | 0.53941 | 0.53138 | -0.00803 |

PSNR improved on 62/79 views, SSIM on 66/79 and LPIPS on 35/79. Matching and pose optimization changed as well, so the gain cannot be attributed only to FPS. Evaluation poses were jointly estimated, not independent ground truth.

### 3.4 Floater cleanup

Tested opacity, size and neighbor-outlier filters on copied checkpoints, followed by connectivity and spatial-support pruning. Mild filtering removed 228,051 of 997,966 Gaussians (22.85%), but PSNR decreased 20.29982 → 20.20693, SSIM 0.69782 → 0.69637 and LPIPS increased 0.53138 → 0.53284. Haze was not clearly eliminated; stronger pruning created gaps and lost surfaces. Connectivity trials also reduced held-out quality and were rejected. Flagged Gaussians are not confirmed floaters; the original remains preserved.

### 3.5 Courtyard: frame rate and training

A separate courtyard recording was processed at approximately 3 FPS and qualified 30 FPS with RGB-D recovery and depth supervision. The full-rate graph connected 6,996/6,996 qualified views. The following evaluations use the same 88 held-out RGB targets.

| Model | Updates | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| Earlier 3 FPS model | See original run | 20.2594 | 0.55901 | 0.50232 |
| Full-rate original poses | 20,000 | 19.9533 | 0.54451 | 0.51322 |
| Full-rate original poses | 120,000 | 20.8095 | 0.57712 | 0.44103 |

The 30 FPS 20k combination was worse than 3 FPS, but pose processing, densification and training conditions also changed. FPS alone is not isolated. The full-rate 20k run visited its 5,970 training images only three or four times each; its wall time included a long pause. The 120k result improved the image scores without independently validating geometry.

### 3.6 Small-scene pose refinement

Compared inherited courtyard poses with fresh local SfM and robust bundle adjustment on 28 continuous views. Both used the same RGB and sampled depth, 23 training views, three held-out views and 6,000 updates; each pose branch had its own transformed depth initialization.

Inherited versus fresh gave PSNR 20.74396 → 20.71422, SSIM 0.602875 → 0.602454 and LPIPS 0.187114 → 0.183063. A predefined stairs crop worsened on all three metrics. Reprojection error improved slightly, but depth-warp checks were mixed. The inherited control was retained; fresh local SfM/BA did not convincingly fix blur and does not prove the original poses correct.

**Evidence:** [RealSense experiment](../experiment_depth.md), [15 FPS methods](../results/realsense_15fps_experiment/METHODS.md), [15 FPS results](../results/realsense_15fps_experiment/REPORT.md), [floater cleanup](../experiment_floater_cleanup.md), [courtyard experiments](../experiment_realsense_cleaning.md), [small-scene trial](../experiment_small_scene_pose.md).

## 4. Viewer, people and courtyard revisits

Added capture-route replay, timeline interpolation, pause/free exploration, manual-camera refresh and point-cloud diagnostics. Exact-pose restoration and tracing people-region Gaussians supported viewpoint, occlusion and reconstruction effects in the tested views rather than replay time deleting people. A subsequent audit confirmed rendering at novel free-camera views; it did not establish that every artifact was fixed.

The courtyard was revisited once with people and once without. The original graph lacked direct constraints between approximately 107 and 216 seconds. A preliminary depth-supported revisit estimate strongly disagreed with the saved relative pose, but matches were concentrated on near-planar floor and are not ground truth. Straight-line camera interpolation can itself cross the floor, so appearance alone does not prove a second physical floor.

| Model, same 88 targets | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Original poses, 20k | 19.9533 | 0.54451 | 0.51322 |
| Original poses, 120k | 20.8095 | 0.57712 | 0.44103 |
| Corrected poses, unmasked, 20k | 18.7991 | 0.47879 | 0.60762 |
| People-preference trial, 20k | 18.4022 | 0.46992 | 0.61898 |

The corrected trial was blurrier and largely lost the group; the people-preference trial showed ghosted people and harmed late views. Both were rejected as replacements. Poses, initialization, split and/or update count differ, preventing single-cause attribution. A static model cannot faithfully show people both present and absent at different times without a dynamic treatment.

**Evidence:** [Capture replay](CAPTURE_REPLAY.md), [free-camera audit](VIEWER_FREE_CAMERA_20261031.md), [pose restoration](VIEWER_POSE_RESTORE_20261031.md), [subject trace](VIEWER_SUBJECT_TRACE_20261031.md), [revisit drift](REVISIT_DRIFT_20261102.md), [experiment log](EXPERIMENT_LOG.md).

## 5. New room recording and geometry investigation

### 5.1 Baseline and initial point cloud

For 20261030_134523.db3, selected 477/4,764 RGB frames at 3 FPS. RGB-only mapping registered 301/477 in its best component; RGB-D bridging connected 477/477. The 120k baseline used 417 training and 60 held-out views, depth weight 0.01 and ended with 1,063,961 Gaussians. Held-out PSNR / SSIM / LPIPS was 23.1133 / 0.79857 / 0.30898. The scene has recognizable broad shapes alongside blur, haze and missing detail.

A diagnostic screened the initial 180,000-point cloud and flagged 2,513 points with at least three free-space conflicts and no support in 60 sampled views. These are candidates, not verified errors: occlusion, poses, edges, moving objects and sampling affect the test. No points were deleted. Graph connectivity and fitted residuals do not independently establish trajectory accuracy.

### 5.2 Alignment and provenance

Checked the recording hash against baseline provenance, froze 60 held-out source identities and timestamps, and saved nine repeatable inspection views. The provenance did not include an initial hash, so this does not independently establish immutability during original training.

Examined unfilled depth overlays, manually selected RGB/depth edges, physical foreground/background boundaries, additional frames, depth-transition profiles, RGB-only revisits and disjoint view-group cloud checks. Saved preprocessing reproduced exactly, demonstrating pipeline consistency rather than physical calibration. Depth holes, reflections and ambiguous boundaries prevented a supported global timing or calibration correction. Diagnostic colors were not used as deletion labels.

### 5.3 SDK inverse-Brown mapping candidate

A numerical pyrealsense2 check exposed an RGB inverse-Brown mapping discrepancy. Sampled baseline coordinates differed from SDK deprojection by median 11.86 and maximum 19.38 raw pixels; an alternate mapping agreed within 0.029 pixels in the sampled check. Sampled depth deprojection and transform math agreed. This is a convention discrepancy relative to the SDK and recorded calibration, not independent physical ground truth or timing validation.

Prepared 477 separate candidate images, rebuilt RGB and RGB-D reconstructions with the same held-out identities and generated a separate 180,000-point diagnostic cloud. Candidate bridging reached 477/477 connected views. Some selected edge examples improved, while line and revisit checks were mixed or inconclusive. Both graphs use the same capture; connectivity does not establish room-wide accuracy.

### 5.4 Mesh and revisit repair trials

Built separate 180k- and 900k-point experimental source clouds and ball-pivoting meshes. The denser mesh contained 546,983 triangles and appeared more continuous in some approximate previews, but retained holes, jagged connections and smearing near exhibits and reflective surfaces. Missing depth does not prove empty space. No mesh or cleaned cloud was accepted for training.

A distributed search assessed 253 revisit pairs; most lacked enough matches, and the eight diagnostic pairs concentrated on repeated exhibit structures. An exploratory rigid-loop graph reduced its fitted loop residual, but another uncertain fitted comparison worsened and camera centers moved by up to 1.073 m. It was rejected for training and replacement.

**Decision.** The geometry gate remains open. Establish physically verified static correspondences across multiple depths and regions, then evaluate independent check constraints before accepting a correction. Additional capture may be needed if the recording lacks sufficient evidence. Original data, graph and checkpoint remain preserved.

**Evidence:** [Pipeline diagnosis](PIPELINE_DIAGNOSIS_20261102.md), [improvement workflow](EXISTING_CAPTURE_IMPROVEMENT_WORKFLOW.md), [alignment audit](ROOM_RGBD_ALIGNMENT_AUDIT_20261103.md), [steps 1–6](ROOM_STEPS_1_TO_6_PROGRESS_20261103.md), [SDK finding](ROOM_SDK_REMAP_FINDING_20261103.md), [distributed validation](ROOM_DISTRIBUTED_VALIDATION_20261103.md), [revisit repair](ROOM_REVISIT_REPAIR_TRIAL_20261103.md).

## 6. Ordinary video workflow

Added scripts/mp4-to-3dgs-prepare.py and scripts/train-mp4-scene.ps1 for frame extraction, RGB feature matching, COLMAP mapping and undistortion, training and viewing, with a registration gate. A 48-frame synthetic fixture registered 28/48 and failed the default 80% gate. At a diagnostic-only 50% threshold it completed preparation and a 1,000-step GPU training/viewer test. This verifies a fixture workflow, not real-video quality or the suitability of relaxing the gate.

For VID20261104003648.mp4 the logged job extracted 20,090 frames at 30 FPS and used CPU COLMAP. A tested run-specific override prepared 120,000 updates and a 1,000,000-Gaussian cap if training was reached. The last log did not establish successful registration or completed training; this is historical status, not live status.

The RealSense 15 FPS experiment used 694 reused observations as SfM anchors and depth-backed high-rate recovery, whereas the MP4 pipeline attempted RGB-only mapping on all selected images. This explains different workloads, not a controlled runtime benchmark.

**Evidence:** [MP4 workflow](MP4_TO_3DGS.md), [video comparison](VIDEO_PIPELINE_COMPARISON_20261103.md), [experiment log](EXPERIMENT_LOG.md).

## 7. Contributions and next steps

Contributions beyond baseline gsplat include budgeted and evidence-ranked allocation; long-axis splitting; repeated-seed, multiscene and ablation studies; a CCD pilot; RealSense extraction, alignment, depth supervision and RGB-D pose recovery; viewer and audit tools; reproducible evaluation and provenance; SDK mapping investigation; and an MP4 training workflow.

Next, validate calibration and poses using distributed static correspondences and independent checks at multiple depths. Once geometry is adequate, compare depth supervision, pose refinement, frame rate and per-view optimization one factor at a time with fixed source identities and held-out targets. Extend Gaussian studies to outdoor scenes and more seeds, isolate split parameters and control measurement load. Retain original recordings, checkpoints and rejected trials; do not treat diagnostic point flags or graph connectivity as validated geometry.

The main chronology is [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). The [later completed campaign](../results/research_completion_20261027/REPORT.md) records work after the earlier cutoff in [RESEARCH_COMPLETION.md](RESEARCH_COMPLETION.md); later results are used above. All numbers are transcribed from saved evidence and are not fresh measurements.
