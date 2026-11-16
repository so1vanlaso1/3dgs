# Next-step plan: DB3 reconstruction first; optimization and meshes later

Prepared 10 November 2026, Asia/Saigon. Repository: `D:\gsplat\gsplat`.

This is a planning and evidence-review deliverable. No reconstruction, training, mesh cleanup, dependency upgrade, or pipeline implementation was performed for this review. Findings, problems, proposed solutions, verification requirements, and the research review record are consolidated here so the deliverable is **one Markdown file**.

**User priority:** reconstruct `20261030_134523.db3` as well as the available observations support before working on general pipeline optimization or editable meshes. All active effort goes to calibration, camera poses, reliable depth, reconstruction training and verification of the resulting 3D view. There is no parallel public-Room mesh milestone or general performance campaign.

**Active sequence:** baseline and failure map → consistent RGB/depth projection → validated camera reconstruction → reliable initialization/depth → local Gaussian pilot → whole-recording reconstruction → final 3D-view comparison. Only after that sequence reaches its acceptance decision do mesh work and general optimization enter the schedule.

Start with [Problem 3: DB3 reconstruction](#3-how-to-reconstruct-the-db3-recording-as-well-as-possible) and its [revised timeline](#311-execution-sequence-and-deliverables). Problems 1 and 2 retain the earlier technical research as a deferred backlog; their numbering is retained for cross-references, not execution priority. Recording run identities, saving results and fixing a memory/runtime failure that blocks DB3 reconstruction remain necessary supporting work. A broad runner rewrite, profiling campaign, budget/compression sweep or mesh deliverable is not a prerequisite.

## 1. How to optimize the pipeline further

**Deferred until the DB3 reconstruction milestone.** The priorities and experiments in this section apply to the later optimization phase. During the active phase, implement only the corrections or minimal experiment controls needed to improve and evaluate the DB3 reconstruction itself.

### 1.1 What the project already establishes

There are two different scenes called “Room” in the project. They must remain distinct in every future experiment:

| Dataset | Current role | What its results establish |
|---|---|---|
| Public Mip-NeRF 360 Room, `data/room_factor2` | Controlled algorithm development with supplied camera reconstruction | Gaussian reduction/rendering comparisons and the latest Gaussian-derived mesh trial |
| Personal museum/room recording, `20261030_134523.db3` | Difficult real capture requiring estimated RGB-D geometry | An existing viewable reconstruction, diagnostic clouds and experimental sensor-derived meshes; physical geometry remains unresolved |

The completed A04 campaign already includes four scenes, three seeds, a split/clone factorial, a budget sweep and parameter confirmation. Do not spend the next phase rediscovering that work. Its Room budget sweep achieved 30.5461 dB at about 400k Gaussians versus 30.6007 dB at about 598k; raising the ceiling to 1M produced about 605k and 30.6049 dB. A ceiling alone did not force useful extra growth. These are A04 development results, not an A05-Abs budget sweep or evidence about the personal recording. [Completed campaign](../results/research_completion_20261027/REPORT.md)

A05-Abs improved PSNR and reduced Gaussian count in all six tested Room/Counter/Kitchen scene-seed pairs; five passed the predeclared practical gate. Counts fell approximately 24–33% relative to A04. The measured training times were similar, so count reduction must not be described as an established equivalent speedup. Counter seed 43 missed the +0.05 dB gate. The evaluation images were exposed during development, and changing the absolute-gradient signal and threshold together limits causal attribution. [A05-Abs results](A05_ABSGRAD_RESULTS_20261107.md)

**Keep A05-Abs with `surface_lambda=0` as the main public-scene candidate.** Keep A04 and the ordinary baseline as controls. Treat transfer to the DB3 scene as a new experiment. The curvature prior improved a smoothness proxy but did not consistently improve mesh topology. The residual-driven A05 fallback also failed its equal-count warm-start comparison; neither deserves priority over the known input-geometry problems. [Surface results](SCHOOL_COMPLETION_RESULTS_20261109.md), [residual A05 results](A05_RESEARCH_RESULTS_20261107.md)

### 1.2 Define optimization as an operating point

The objective is the best useful reconstruction within the available time and memory, not a single maximum PSNR or minimum Gaussian count. Record these separately:

| Objective | Required measurement | Common misleading substitute |
|---|---|---|
| Appearance | Per-view PSNR/SSIM/LPIPS, fixed difficult crops, local free-view inspection | Mean PSNR alone |
| Geometry | Withheld static correspondences, depth/surface consistency, reference geometry where available | Pose-graph fitted residual or smoothness alone |
| Coverage | Reliable observed surface coverage by room region, plus unknown regions | Registered-frame percentage |
| Training efficiency | Time to a predefined quality level, total stage time, peak CPU RAM and GPU memory | Final count alone |
| Viewer performance | End-to-end input-to-frame latency and sustained frame time at fixed resolution/path | CUDA-only FPS |
| Mesh usefulness | Surface fidelity, holes, unsupported bridges, editable export and reopen | Number of triangles or one connected component |

The machine currently reports a **GTX 1660 SUPER with 6,144 MiB VRAM**. The current driver reports 610.88; the old DB3 provenance reports 591.86. Repeat timing controls under the same current environment before claiming speedups. The saved personal-scene preview's roughly 242 FPS excludes display, encoding and image I/O; it is not interactive application FPS. [Preview verification](../results/realsense_20261030_134523_120k/preview/verification.json)

### 1.3 P0: make experiments reusable and comparable

**Problem:** many useful capabilities are isolated scripts with hard-coded dataset/output names. The SDK-consistent correction is not integrated into the normal cleaner. Some runners reuse stages by command equality, even though the same command can run against changed inputs or code. Current final post-update checkpoints contain splat weights but not a complete resume state.

**Proposed implementation:**

1. Define one versioned run specification containing recording identity, calibration/remap version, source-frame IDs, split, poses, initialization, strategy, loss settings, training schedule and extraction settings.
2. Divide the pipeline into explicit stages: inspect → rectify/pair → select/match → estimate/refine poses → validate geometry → initialize/fuse → train → evaluate → extract → edit/export.
3. Use a stage fingerprint built from input hashes, relevant configuration, source version and dependency versions. Reuse a stage only when the fingerprint and expected output checks match. A success marker or unchanged command text is insufficient.
4. Record dependency invalidation: changing remapping invalidates RGB features, matching, poses, point colors, initialization and downstream models; changing poses invalidates fusion and training; changing only mesh extraction does not require retraining.
5. Add a resumable checkpoint type containing splats, optimizer/scheduler states, densification accumulators, RNG states, camera replay position, completed updates and any pose/appearance modules. Keep export-only weights explicitly distinguished.
6. Make metric collection and exports load the same final post-update checkpoint. An intermediate historical checkpoint saved before an update is not automatically the model used for the subsequent evaluation.
7. Preserve raw and experimental artifacts; create a new run directory for every scientifically different candidate.

**Acceptance:** deliberately change one dependency and verify only the affected stages invalidate; interrupt/resume a short deterministic control and compare its camera sequence, structural events and final tensors within a declared tolerance. Export/reload must reproduce fixed-view renders. Implement these checks when implementing the runner, not by rerunning long campaigns now.

Relevant owners: `scripts/run-clean-realsense.py`, `scripts/run-new-recording-120k.py`, `scripts/run-research.ps1`, `scripts/research_a05.py`, `examples/simple_trainer.py`.

### 1.4 P1: profile before changing algorithms

Run one accepted public-scene control and, after Problem 3 passes locally, one personal-scene control. Record cold-start/compile time separately from warm execution.

Measure decoding, remapping, matching, pose optimization, initialization, image loading, host-to-device transfer, rasterization, backward pass, optimizer updates, densification/scoring, evaluation, checkpoint writing and meshing. Record peak process RAM, CUDA allocated/reserved memory and whole-device usage. Include resolution, active applications and the actual number of training rays/views.

Choose improvements from the measured bottleneck:

| Observed bottleneck | First intervention to test | Acceptance condition |
|---|---|---|
| Repeated DB3 decoding/remapping | Immutable decoded/rectified cache keyed by source and calibration | Identical pixels and source IDs; lower repeated-stage time |
| Matching too many near-duplicate views | Sequential matching with targeted revisit retrieval and adaptive keyframes | No loss of independently checked trajectory/coverage |
| Host RAM pressure | Zero-worker baseline; bounded image cache or shared/memory-mapped storage | No paging/OOM and identical camera replay |
| Renderer/optimizer memory | Benchmark packed mode, then compatible optimizer options individually | Measured memory benefit without unacceptable quality/time loss |
| Dense pose graph scaling | Sparse data structures and keyframe/submap optimization | Pose checks preserved; report solve time and convergence |
| Scoring/structural overhead | Adjust observation/event frequency only after profiling it | Same quality guardrails and measured time benefit |
| Mesh volume memory | Bounded local volumes or a validated sparse backend | Synthetic-control pass and no new seam/coverage defects |

The earlier four-worker Windows image cache exhausted available RAM; switching to zero workers changed camera sampling after the first epoch. The project already solved comparability using saved-camera replay. Preserve that solution when experimenting with loaders. `packed=True` is not automatically faster; the current trainer itself documents the memory/speed tradeoff. [Recorded worker failure and replay solution](SCHOOL_COMPLETION_RESULTS_20261109.md)

### 1.5 P2: small, ordered training experiments

Do these on fixed, accepted input geometry. Start with one development scene/seed, promote only useful candidates, and confirm on additional scenes/seeds. Do not run the whole Cartesian product.

| Order | Experiment | Keep fixed | Decision |
|---|---|---|---|
| O1 | Current baseline/A04/A05-Abs on the accepted personal subset | Images, split, initialization, cameras, updates, resolution and depth policy | Establish whether public-scene gains transfer |
| O2 | A05-Abs budgets: 400k and 700k first | O1 recipe and camera replay | Find lower count/time without unacceptable quality or coverage loss |
| O3 | Evaluate checkpoints at approximately 7k, 15k and 30k updates | One declared schedule and full resume state | Measure when development quality stops improving; extend only with evidence |
| O4 | Densification schedule relative to image exposure count | Total updates/budget and other settings | Test whether useful detail is missed when refinement ends early |
| O5 | Prepared 848-wide versus higher-resolution RGB | Source IDs, validated calibration/poses and comparison grid | Determine whether source detail justifies memory/time cost |
| O6 | Conservative contribution-based pruning plus short recovery | Same starting checkpoint and recovery cameras, no-prune control | Retain only if resource savings survive all quality/coverage checks |

The 1M budget is a conditional follow-up if 700k is binding and a detailed region demonstrably benefits. For O3, checkpoint evaluation measures progress within one schedule; a rescaled 7k schedule versus a rescaled 30k schedule is a different experiment. Report both updates and effective image exposures; using more frames with unchanged updates gives each frame fewer observations.

For O6, low opacity alone is inadequate evidence that a splat is disposable. Use observed contribution across multiple separated training views, protect thin structures and low-coverage areas, and retain a rollback copy. Save counts before pruning, after pruning and after recovery so regrowth cannot conceal the cost.

**Proposed engineering guardrails, to freeze before a sweep:** for a resource-saving candidate, require at least 15% reduction in the targeted resource, PSNR loss no worse than 0.10 dB, SSIM loss no worse than 0.001 and LPIPS increase no greater than 0.005, together with no material geometry/region-coverage regression. These are starting decision thresholds, not measured results or universal perceptual limits. For a pure quality claim, retain the existing +0.05 dB/0.001 SSIM/0.001 LPIPS practical gate and add geometry checks. Confirm finalists with at least two seeds; report failures and per-view changes.

### 1.6 P3: deployment and viewer optimization

After selecting the model, measure file size, load time, GPU use, steady frame time, first response after movement and final refinement time on the same replay and local exploration paths. Test compression or LOD only on a preserved derivative and compare reloaded renders with the selected model.

Separate viewer defects from scene defects: input events, camera matrices, viewport/FOV, model-to-camera normalization and reconnect behavior need checks, but a functioning viewer cannot repair a smeared learned surface. The existing project already has replay/free-camera diagnostics; extend those to the final candidate instead of rebuilding navigation first. Historical courtyard viewer results apply to that capture, not automatically to the personal museum scene. [Replay implementation history](CAPTURE_REPLAY.md), [free-camera audit](VIEWER_FREE_CAMERA_20261031.md)

### 1.7 Completion condition for pipeline optimization

Deliver a reproducible quality/resource table and a chosen default, plus optional fast/high-quality configurations if their tradeoffs are measured. Every output must identify its recording, camera/calibration version and checkpoint. The pipeline must stop or downgrade its claim when geometry validation fails. “Fully optimized” should mean meeting these declared requirements on the tested scenes and hardware, not proving a global optimum.

## 2. What to do next with the mesh part

**Deferred until the DB3 reconstruction milestone.** Do not start public-Room mesh cleanup, editor roundtrips, extraction sweeps, texture baking or surface-model research alongside the active reconstruction work. A quick cloud or fused-depth preview may be used to diagnose poses; it is a reconstruction diagnostic, not a mesh production milestone.

### 2.1 What exists and what is missing

| Artifact | Actual source | Current finding | Next action |
|---|---|---|---|
| `results/school_completion_replay_20261109/meshes_uniform/reduced/raw_mesh.ply` and `.obj` | A05-Abs public-Room Gaussian expected depth, 64 training cameras | 119,987 triangles; 1,162 components; 16,807 boundary edges; recognizable surfaces with holes and lost thin details | Clean a derivative and prove an editor roundtrip |
| Other three meshes in the same campaign | Baseline and surface-prior factorial arms | Prior does not consistently improve topology | Retain as matched controls |
| `results/room_improve_sdk_dense_test_mesh_20261103/room_experimental_mesh.ply` | Personal recording's sampled sensor-depth cloud, ball pivoting | 900,000 vertices, 546,983 triangles, 476,050 vertices unused by faces; large holes and fragmentation | Treat as diagnostic; repair input geometry before claiming improvement |

The sensor-derived mesh is **not** evidence of conversion from optimized Gaussians. Its ball-pivot radii are 4.5, 7.5 and 12 cm; reported maximum triangle-edge length is about 24 cm. Those settings can interpolate across gaps, and the current preview uses approximate painter rendering. These measurements do not identify every wrong face, but explain why adding points/triangles is not a sufficient quality criterion. [Sensor mesh report](../results/room_improve_sdk_dense_test_mesh_20261103/report.json)

The public-Room extractor currently uses `RGB+ED`, opacity >0.7, downsample factor 4, 64 training views and a bounded uniform TSDF capped at 192 voxels per axis. The cap means actual voxel size can be coarser than the requested `0.01 * scene_scale`. Expected depth can lie between multiple surfaces; high opacity alone does not make it a reliable surface intersection. The bounds can exclude exterior or peripheral scene content. [Current extractor](../scripts/gaussians_to_mesh.py)

### 2.2 M1: finish a useful editable mesh in the later mesh phase

Start this work after the DB3 reconstruction acceptance decision. Prefer the accepted personal-scene reconstruction for the eventual asset; the existing public-Room mesh remains a useful editor/extraction control if needed.

1. Freeze the raw A05-Abs mesh and its report/hash. Export a separately named working copy.
2. Audit finite vertices, duplicate/degenerate faces, unreferenced vertices, winding/normals, connected components by **surface area**, boundary-loop sizes, nonmanifold edges/vertices and self-intersections where the tool supports them.
3. Remove only invalid/redundant elements automatically. For small components, inspect position, source-view support and thin-object identity before removal. A chair leg or leaf can be small and valid.
4. Fill only small, locally supported holes. Keep doors, windows, missing backsides and large unobserved regions open or explicitly label a manually modeled completion. Never close the whole room simply to obtain a watertight score.
5. Test conservative decimation at 75% and 50% of the cleaned triangle count, one derivative at a time. Measure surface deviation, silhouettes and thin-feature retention; keep the best acceptable level rather than a predetermined polygon count.
6. Apply limited smoothing only where visibly needed. Compare before/after displacement and preserve edges; smoother does not imply more accurate.
7. Import into Blender or the intended design editor. Select and edit a small surface, save, close, reopen and verify the edit, orientation, units, normals and materials. Save before/after views and the exact cleanup recipe.
8. Deliver raw, cleaned and edited variants with a concise limitation note. PLY preserves existing vertex colors; the current OBJ exporter explicitly omits vertex colors, so a textured OBJ needs UVs/materials/texture files. GLB can be added as a viewer-friendly derivative after its import/export is verified.

**Acceptance:** successful real editing roundtrip; no invalid exported geometry; visibly retained important structures; documented remaining holes; topology and fidelity changes reported. An open room need not be globally watertight. Do not declare success solely because the component count decreases.

### 2.3 M2: improve extraction before adding another loss

Use the same fixed checkpoint and camera set so extraction effects are isolated from training effects.

| Test | Concrete change | Measurement and stop condition |
|---|---|---|
| Coverage | 64 versus 128 training cameras, then coverage-aware selection | Gains in supported surface coverage; stop if only redundancy/time increases |
| Sampling | Downsample 4 versus 2 | Thin-feature/silhouette improvement, memory and extraction time |
| Volume | Explicit common ROI; resolution 192 versus 256 if memory permits | Report actual voxel length, retained/excluded bounds and fidelity; 256³ has about 2.37× the cells of 192³ |
| Reliability | Add discontinuity, grazing-angle and cross-view depth/normal consistency masks | Fewer unsupported bridges without unacceptable missing coverage |
| Surface depth | Compare expected depth with a supported surface-depth alternative | Demonstrate reduced multi-surface blending using the same cameras; mark any required renderer work as new implementation |
| Color | Visibility-aware multi-view texture baking | Fewer seams/ghosts and preserved source provenance |

Increasing all of resolution, view count and mask strictness at once would make the source of improvement unclear. Run small ROI pilots, change one dimension, and promote only useful settings.

**Backend requirement:** the local Open3D 0.20 scalable backend previously returned no voxels even for a synthetic plane; the uniform backend passed. Keep the working uniform backend for near-term bounded tests. Before using a sparse/scalable or tensor backend for the whole room, reproduce the plane control in an isolated environment and add a two-surface/occlusion control and transform/units check. Do not assume a dependency upgrade fixes the recorded failure. Open3D documents the voxel/noise tradeoff and larger-scene advantages of sparse storage, but that is not validation of this local installation. [Local backend test](../tests/test_mesh_backend.py), [Open3D RGB-D integration](https://www.open3d.org/docs/release/tutorial/pipelines/rgbd_integration.html)

**Extractor contract to implement:** reconstruct the saved training configuration, split, normalization and active pose/appearance modules rather than creating a new Runner from mostly defaults. The present extractor is adequate evidence for its recorded fixed-pose campaign, but future pose-optimized or differently configured runs need their effective cameras restored. Verify rendered depth semantics: camera-axis Z versus Euclidean ray range, scale, intrinsics resizing and world-to-camera inversion using synthetic tests. Save the inverse normalization so personal-scene exports can recover meters; public SfM normalization alone does not establish physical units.

### 2.4 M3: compare two mesh branches after the DB3 geometry gate

Both branches should use the same accepted coordinate frame, ROI and validation observations, with clear source attribution:

**Branch A — sensor RGB-D TSDF.** Fuse accepted depth observations with validated poses and reliability masks. Preserve per-surface observation support. This is the fastest direct geometry baseline and a diagnostic for whether Gaussian training introduces new surface errors. It is not independent ground truth if it shares the same sensor/poses.

**Branch B — optimized Gaussian to mesh.** Fuse rendered training-view surface depth from the selected Gaussian model; exclude held-out depth/images from fusion. This directly addresses the research goal. Compare raw and identically cleaned versions with Branch A.

Interpret the result conditionally:

| Outcome | Next diagnosis |
|---|---|
| Both meshes show doubled walls or the same displaced structure | Revisit calibration, poses and coverage before changing meshing |
| Sensor mesh is coherent but Gaussian mesh blends surfaces | Investigate supervision, Gaussian surface representation and rendered-depth extraction |
| Gaussian mesh is coherent but sensor mesh has depth holes | Inspect sensor range/material limitations and whether multi-view RGB genuinely supports the recovered surface |
| Both are coherent but exports are unusable | Focus on topology cleanup, UVs, simplification and editor integration |

A hybrid sensor/Gaussian mesh is a later option. If used, label which surfaces are sensor-supported, image-derived or manually completed; do not present all surfaces as equally measured.

### 2.5 M4: conditional surface-model research

If poses and depth pass but Gaussian-derived geometry remains the limiting factor, run **one** controlled 2DGS comparison before broadening to several new methods. The repository already contains `examples/simple_trainer_2dgs.py` and `rasterization_2dgs`; this is a lower-integration-cost starting point than introducing an unrelated stack. The 2DGS method uses oriented disks and depth-distortion/normal-consistency objectives for surface reconstruction. Its publication results motivate a test, not a guarantee for this recording. [2DGS paper and code entry point](https://surfsplatting.github.io/)

First verify the local 2DGS path on a tiny known scene, then add the same split/provenance/replay support needed for a fair comparison. Do not assume A05's strategy or absolute-gradient handling transfers unchanged. The local trainer's normal regularizer starts at update 7,000 by default; a 7,000-update test could leave essentially no useful regularized training. Specify and record its schedule deliberately.

Compare accepted 3DGS, 3DGS plus a justified surface constraint, and 2DGS on the same validated geometry with both matched-update and measured time-to-quality reporting. Promote only if surface fidelity/completeness improves without unacceptable rendering/resource cost. The existing log-depth curvature loss remains a smoothness prior; it is not a physical planarity guarantee and has not earned default status.

### 2.6 Mesh evaluation and definition of done

Use three independent categories of evidence:

- **Structural/usability:** finite geometry, topology, boundary-loop distribution, component area distribution, file size, editing roundtrip and texture/material integrity.
- **Consistency:** mesh-rendered depth versus withheld reliable observations, normal consistency, silhouette agreement and supported coverage at fixed cameras. Report missing coverage alongside error so a tiny surviving surface cannot win by omitting difficult regions.
- **Physical accuracy where available:** independent scan or measured targets, with registration protocol, units and measurement uncertainty. Report precision/recall or F-score at thresholds justified by sensor/reference precision, as well as distance summaries. Do not report Chamfer distance to a cloud from the same pipeline as independent accuracy.

Completion means an editable asset with documented reliable regions and limitations, plus a measured comparison showing which reconstruction/extraction changes helped. It does not require inventing surfaces that were never observed.

## 3. How to reconstruct the DB3 recording as well as possible

### 3.1 Current evidence and prioritized diagnosis

Primary target: `20261030_134523.db3`, the indoor museum/room recording. The other courtyard recordings and the phone MP4 are separate datasets. This review inspected the DB3 read-only: it is a SQLite sensor container with RGB, depth, calibration, metadata, gyro and acceleration messages, not just an ordinary video.

| Finding | Evidence/status | Consequence |
|---|---|---|
| 4,764 color and 4,764 depth frames over about 159 seconds; 31,758 gyro and 15,905 acceleration samples | Direct read-only database query in this review | Preserve synchronized sensor content; do not reduce the task to MP4 extraction |
| Raw RGB 1280×720; depth 848×480; baseline prepared RGB 834×463; 477 selected frames | Saved calibration/manifest and current DB counts | There is more temporal/RGB detail to investigate, but more frames alone is not a fix |
| Baseline and SDK-candidate RGB-only largest components both registered 301/477 | Saved reconstruction diagnostics | Remapping alone did not solve registration completeness |
| RGB-D bridging connected 477/477 | Current baseline and candidate graph reports | Connectivity is established; physical correctness is not |
| Main cleaner still uses `invert_brown(ideal, coeff)` for the declared inverse-Brown model | Current source inspection | The SDK discrepancy is still relevant to normal reruns |
| Pinned SDK audit differed from the baseline by median 11.86 and maximum 19.38 raw RGB pixels on 63 samples; alternative roundtrip max about 0.026 pixels on 35 samples | Saved SDK audit, re-read in this review | Highest-priority concrete convention discrepancy; physical calibration still requires scene checks |
| Candidate and old trajectories differ after alignment; distributed revisit evidence is mixed/narrow | Saved pose/revisit reports | Do not call the SDK candidate globally corrected |
| The single-loop repair moved some centers by up to 1.073 m and worsened another uncertain check | Saved rejected trial | Do not reuse that graph as an accepted repair |
| Baseline used 120,000 updates, 417 training views, depth weight 0.01 and refinement ending at 6,000; final count 1,063,961 | Saved diagnosis/configuration and current preview verification | More training already occurred without resolving visible defects; no controlled step-count diagnosis exists |
| Saved evaluation: 23.1133 dB PSNR, 0.79857 SSIM, 0.30898 LPIPS on 60 jointly posed held-out views | Existing report | Interpolation performance, not independent geometry accuracy |
| Source depth is missing in parts of distant walls/doorway; reflective displays remain ambiguous | Existing depth review and contact sheets inspected again | Some areas may require RGB-only evidence, explicit uncertainty or new capture |

Sources: [pipeline diagnosis](PIPELINE_DIAGNOSIS_20261102.md), [SDK discrepancy](ROOM_SDK_REMAP_FINDING_20261103.md), [distributed validation](ROOM_DISTRIBUTED_VALIDATION_20261103.md), [rejected repair](ROOM_REVISIT_REPAIR_TRIAL_20261103.md), [baseline graph](../data/realsense_20261030_134523_clean/rgbd/report.json), [candidate graph](../data/room_improve_sdk_rectified_20261103/rgbd/report.json).

**Priority ranking:** resolve remapping semantics and prove useful local alignment; strengthen camera estimation/validation; build support-aware initialization and depth supervision; then tune training and extraction. Timing, global drift, dynamic content, weak parallax and inadequate exposure to each image remain hypotheses with different amounts of supporting evidence. No defensible percentage of blame can currently be assigned to them.

### 3.2 D0: freeze provenance, regions and evaluation

Create a new candidate namespace and retain the existing 120k model as the visual baseline. Record a fresh source SHA-256 before future processing; the saved hash is `60e3f253051fc282a61af7f6c9c5eef40830a12d42566b9eba0992463115bc52`. This review read the database but did not recompute its entire approximately 17.1 GB hash.

Freeze four categories of observations:

1. Geometry-fitting frames/features/depth used by calibration or pose estimation.
2. Geometry-check features/structures withheld from those fits, distributed by region and distance.
3. Appearance training/development images with explicit original message IDs.
4. Final evaluation observations reserved before further choices, preferably a new capture or reference when the old data have already been repeatedly inspected.

Retain the original 60 held-out image identities for historical comparison, but label them exposed development evaluation. They were excluded from splat training, not from all pose estimation or human-guided decisions. Splitting feature correspondences after a graph was already estimated does not create independent pose validation. Newly chosen buffered blocks from the old recording are prospective checks for subsequent work, not an untouched historical test set.

Use separate difficult regions: kiosk/map, wide room near 43 s, distant doorway near 61 s, exhibit/rails near 68 s, reflective case near 110 s and an end-of-route revisit. Include at least one matte, well-observed near/mid-distance region as a control. Reuse existing region inventories instead of starting an unrelated annotation effort. [Region/validation history](ROOM_IMPROVEMENT_CONTINUATION_20261103.md)

### 3.3 D1: unify and validate RGB/depth projection

**Problem:** a separate SDK-consistent RGB dataset exists, while the default cleaning path retains the disputed mapping; the candidate bridge replaces a function only within its process. This makes it easy to mix calibration conventions across stages.

**Implementation plan:**

1. Build one projection/remapping module shared by cleaning, candidate preparation, bridging, depth association, previews and validation. Select behavior from a versioned calibration model, not a hard-coded recording name or monkey patch.
2. Preserve the old mapping as a reproducibility mode. Use the exact pinned SDK audit as a numerical oracle for this recording, then expand coverage across center, corners, image borders and supported depth ranges.
3. Test both projection and deprojection, distortion model IDs, coefficient order, extrinsics direction, column/row layout, depth units, output scaling and crop principal-point shifts. For this calibration the saved crop is `(7, 7)`; use the actual map metadata, not a remembered horizontal-only correction.
4. Check mapping validity before sampling RGB. Project depth using a nearest-surface/Z-buffer rule, retain invalid pixels, and expose occlusion/disocclusion masks. Do not hole-fill calibration evidence.
5. Regenerate RGB, features, matches and poses in fresh outputs after a mapping change. Old poses or a trained checkpoint cannot simply be paired with new rectified images.
6. Validate real static boundaries on multiple objects, image locations, depths and edge orientations. Review physical correspondence manually; do not treat the nearest RGB texture edge as truth. Report signed offset and annotation uncertainty per structure.

**Gate:** a proposed numerical target is <0.1 raw RGB pixel discrepancy from the pinned SDK on the declared test domain, plus finite/in-bounds behavior where expected. This is a numerical compatibility gate only. For physical alignment, freeze a tolerance based on repeat annotations, depth-edge uncertainty and the resolution actually used. Require no coherent systematic shift across independent structures; explicitly leave unsupported far/glass regions unresolved.

The SDK documentation distinguishes distortion directions and stream coordinate transforms. It does not establish that a one-line Brown-function swap is universally correct across SDK versions or model declarations. Resolve the local implementation with pinned executable tests and recorded calibration, then test physical observations. [RealSense projection documentation](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)

### 3.4 D2: separate timing errors from calibration and motion

The approximately 7 ms RGB-depth pair gap is measurable; its effect is not yet isolated. Never apply a guessed global pixel shift to compensate for it.

1. Inspect image header timestamps, database timestamps, frame counters and available exposure metadata. Establish timestamp domains and monotonicity for RGB, depth and IMU; nearest database timestamps are not proof of simultaneous exposure.
2. Choose repeated observations of the same static boundaries during low and higher motion. Record location, depth and motion direction so content is not confused with a timing effect.
3. With calibration fixed, test a small, preregistered time-offset candidate range justified by timestamp evidence. Reassociate actual recorded observations and use an appropriate motion model; do not blend depth across moving occlusions as if it were measured.
4. Fit using development boundaries and evaluate on other structures/motion intervals. Reject timing changes that help one contour but worsen the rest or vary inconsistently by depth/image location.
5. Use gyro initially as a diagnostic for rotational motion. Full visual-inertial fusion is conditional on validated camera/IMU extrinsics, synchronization and bias/noise treatment; the existence of IMU messages does not supply a trusted trajectory.

**Output:** a documented accepted timing model or an explicit unresolved timing uncertainty, with high-motion pairs downweighted/excluded from geometry as justified. Do not block all useful low-motion local work while waiting for full inertial integration.

### 3.5 D3: use more frames strategically

Separate **tracking frames** from **Gaussian training frames**. Native-rate or 10–15 FPS observations can help track a fast transition, while a smaller selected set can provide diverse training views without repeating thousands of near-identical images.

Start by investigating the existing 44 bridge boundaries and several well-tracked controls. Use native-rate frames around failures, score blur/exposure/overlap, and select keyframes based on translation/parallax, new coverage and track survival. Do not reject a frame solely because a low-texture wall lowers its sharpness score. Preserve room coverage and explicitly report gaps.

For a pilot, compare the 3 FPS recipe with adaptive selection and a bounded higher-rate recipe. Keep all original evaluation source IDs excluded from training; add time buffers where densely adjacent views would make evaluation trivial. When comparing different image counts, report both equal-compute and comparable-exposure results rather than pretending equal update count gives equal training opportunity.

COLMAP recommends overlap, texture and translated viewpoints; it also provides sequential matching and additional matching options for incomplete reconstructions. Use those capabilities selectively after checking the installed version, rather than exhaustive matching of every native-rate pair. [COLMAP capture and reconstruction guidance](https://colmap.github.io/tutorial.html)

### 3.6 D4: repair the camera reconstruction, not just its graph connectivity

The current recovery exports relative-pose-graph results and sampled sensor points with empty image tracks; it is not a global joint refinement of all landmarks and image observations. Calling bundle adjustment on that exported trackless initialization will not create the missing constraints. Preserve or rebuild real tracks from the feature/matching stage.

Implement the following in order:

1. Rebuild RGB tracks on the accepted rectification. Use temporal neighbors plus candidate revisits; reject repeated-panel, moving-person and reflection matches after geometric and spatial-distribution checks.
2. Estimate local RGB-D motion with robust feature reprojection and reliable depth/point-to-plane terms. Test degeneracy: planar or narrowly clustered correspondences may not constrain all motion directions well.
3. Replace hard-coded bridge/SfM confidence proxies with measurement-derived information. `bridge-clean-rgbd.py` currently writes `inliers=100, error=.03`; scale-gated SfM edges use `inliers=1000`; the optimizer uses inlier-based isotropic information. These values do not represent measured anisotropic uncertainty. Propagate uncertainty through chained bridges and avoid counting correlated edges as independent evidence.
4. Add verified revisit constraints across different parts of the route. Evaluate each on withheld physical structures, not only the features used to estimate it. Use robust weighting or switchable/rejectable hypotheses so a false loop cannot warp the whole room.
5. Optimize the complete accepted measurement graph, then jointly refine poses and real landmarks with robust bundle adjustment. Initially hold verified intrinsics/extrinsics fixed; unlock tightly constrained calibration parameters only with independent support. RGB-only BA does not automatically add metric depth residuals: implement those explicitly or keep the metric alignment/factors separate and documented.
6. Re-evaluate metric scale and independent structure consistency after refinement. Small fitted residuals are necessary bookkeeping, not the acceptance test.
7. If a global graph remains unreliable, optimize overlapping local submaps and verify each merge. Preserve unresolved relative placement as uncertain instead of forcing a visually seamless whole room.

For graphs above 1,200 views, the current implementation changes to the sparse solver. Audit its residual weighting and rotation/translation scaling before promotion: raw radians and meters with one scalar confidence do not by themselves encode sensor uncertainty. Also remove avoidable dense adjacency/weight allocations before making full-rate reconstruction the default.

**Validation output:** per-region withheld reprojection and 3D separation, plane thickness/duplicate-surface diagnostics, revisit consistency, track length/coverage, accepted/rejected constraints and a trajectory overlay. Use a fixed alignment fitted without the check structures. A trajectory difference between two estimates is not an absolute trajectory error.

**Gate:** predeclare tolerances using sensor/annotation precision and scene needs; require improvement larger than repeat-measurement variability in the target region and no material worsening in the other checked regions. If reference measurements are unavailable, call the result improved consistency, not certified accuracy. If no distributed reliable constraints can be obtained, stop the global repair after the bounded pilot and take the fallback in D8.

### 3.7 D5: build trustworthy initialization and depth supervision

Current initialization samples up to 1,500 depth points per training frame, selects 2.5 cm voxels and caps the result at 180k. It does not require cross-view support. The exported training depth is sampled separately. This makes pose/depth reliability especially important.

Create confidence information for each retained observation based on valid range, local stability, occlusion/discontinuity proximity, incidence, source motion, cross-view agreement and correspondence/pose uncertainty. Keep categorical provenance alongside any numeric score. A confidence score is a model of evidence, not measured certainty.

Use three treatment classes:

- **Supported static geometry:** fuse/downsample with surface and thin-feature preservation, and use for stronger depth supervision.
- **Uncertain but potentially real content:** preserve in a separate layer or retain RGB-only influence with weak/no depth weight; keep its raw observations.
- **Confirmed dynamic/invalid observations:** exclude from static geometry using masks and a recorded reason. Inspect masks before use.

A no-support point is not automatically wrong. A free-space conflict depends on camera/depth reliability and occlusion; the previously colored diagnostic points are not deletion labels. Missing depth is unknown, not an empty-space measurement.

Extend `examples/datasets/colmap.py` to carry confidence/masks and update the trainer loss to normalize over valid weighted samples. The current loss uses sampled expected depth and a disparity-space L1 term with one global weight; the inspected path does not provide per-sample confidence weighting. Handle zero-valid-depth frames explicitly and keep depth units consistent through normalization.

With accepted fixed poses, compare:

| Arm | Initialization | Depth during training | Purpose |
|---|---|---|---|
| D5-A | Same frozen accepted cloud | Off | Appearance-only optimization control |
| D5-B | Same cloud | Current unweighted policy | Measure the current supervision effect |
| D5-C | Same cloud | Confidence-filtered/weighted policy | Test the proposed supervision change |

Only after this comparison, test improved initialization separately. D5-A is not an entirely RGB-only reconstruction if its fixed poses/cloud were derived from RGB-D; label it accurately. Select loss weights on development regions and freeze them before the confirmation run.

### 3.8 D6: train a small pilot, then expand

Train a matte/local exhibit subset and an adjacent difficult subset first. Use fresh directories, explicit dataset paths and matching initialization/camera replay for paired arms. Start with a bounded 7k-update screen after geometry checks; evaluate around 15k and 30k only if development quality is still improving. This is reconstruction-quality work, not the deferred resource-optimization sweep. Use one stable strategy initially and compare another only when a measured reconstruction defect motivates it; the entire O1–O6 matrix is not required now.

Evaluate captured views, held-out views, short lateral translations, small yaw changes and repeated views of the same structure. Keep the existing 43/61/68/110 s inspection anchors where applicable, but regenerate or align comparison paths carefully when poses change. The old tiny sweeps were in normalized units and were not a verified continuous collision-free walk. A rigid alignment cannot completely reconcile two non-rigidly differing trajectories.

For old/new rectifications, compare to the same original observations through a declared common evaluation mapping and intersection mask; report border/coverage changes. Do not claim a PSNR gain when the target image or effective crop changed. For pose changes, keep source IDs fixed and report the pose/evaluation convention as part of the pipeline comparison.

Promote the subset only if geometry checks, difficult-region views and resource limits all pass. Then expand through overlapping regions to the whole captured room. Report where the improvement stops; a successful local exhibit is not proof of a successful whole-room reconstruction.

### 3.9 D7: deliver and verify the best supported DB3 3D view

Deliver the selected Gaussian reconstruction, the matching camera/calibration configuration, source identities, world transform and an uncertainty/coverage map. Keep a whole-capture contextual view available while clearly distinguishing verified local surfaces from unresolved context. Compare it directly with the preserved 120k baseline and retain the best-supported candidate. Editable mesh extraction and packaging follow later and do not block this deliverable.

Final interactive validation should cover replay → pause → free movement → return, model reload, correct FOV/aspect and coordinate scale. Compare image artifacts at fixed poses separately from input latency or browser reconnect defects. Provide actual paths through observed free space; do not interpret the ability to fly through the viewer as evidence that unseen backsides are reconstructed.

### 3.10 D8: fallback when this recording cannot support the missing geometry

Do not spend repeated 120k runs trying to recover information absent from the recording. After the projection checks, review the local-pose pilot after roughly 3–5 focused working days. If it produces no credible distributed improvement, choose the next reconstruction route from the evidence below instead of moving on to mesh or optimization work. This is a decision checkpoint, not a deadline to abandon DB3 reconstruction:

| Limitation | Practical fallback |
|---|---|
| Local regions reconstruct well, global placement does not | Deliver validated local submaps with uncertain global relationships explicitly marked |
| Far surfaces lack depth but have good translated RGB views | Test an RGB multi-view reconstruction branch and validate it separately; metric scale still needs support |
| Glass/reflections remain ambiguous | Preserve appearance where possible, exclude unreliable depth from physical geometry, mark surfaces uncertain |
| Important surfaces have no useful observations | Recapture those regions or label any manual completion as modeled content |
| Scene-wide validation remains impossible | Keep the best existing full-room baseline, show measured local improvements and report the remaining limitation |

For a new capture, make a 20–30 second static matte control first and process it through calibration, poses and a quick fused surface before recording the entire room. Capture translated overlapping views at multiple heights, revisits of unique static objects, controlled exposure where practical and surfaces within a reliable sensor range. Avoid pure rotation and severe motion blur. Add removable textured targets and independently measured distances where permitted. Retain raw RGB/depth/IMU/calibration and a separate evaluation/reference pass. Confirm improvements on the control before expanding capture time.

### 3.11 Execution sequence and deliverables

**This timeline supersedes the earlier mixed reconstruction/mesh/optimization schedule.** It starts when implementation begins; Day 1 is not a claim that work has already started. Estimates are focused working time, not guaranteed elapsed runtime or improvement. Advance by evidence, not merely because an estimated day has passed.

| Phase | Suggested effort | Work | Concrete exit deliverable |
|---|---:|---|---|
| R0 — establish the target | 0.5–1 day | D0: preserve the current model, source identities and evaluation views; map defects at the fixed inspection regions | Baseline comparison package and frozen reconstruction checks |
| R1 — correct projection and check timing | 2–4 days | D1/D2: unify SDK-checked remapping; regenerate consistent inputs; validate static boundaries; investigate motion/timing only where evidence requires it | Consistent RGB-D input with accepted local alignment and explicit unresolved regions |
| R2 — recover reliable camera geometry | 3–5 days, then review | D3/D4: use higher-rate frames at failed transitions; rebuild real tracks; refine local poses and verified revisits; assess global connectivity separately from accuracy | Validated local/submap geometry and a supported route to whole-recording poses, or a concrete D8 fallback experiment |
| R3 — prove a local reconstruction improvement | 2–3 days | D5/D6: build support-aware initialization; compare depth off/current/confidence-weighted policies on fixed local geometry; train short controlled pilots | A local Gaussian reconstruction that improves the targeted defects without worsening the checked structures |
| R4 — reconstruct the whole captured scene | 3–5 days initially | Expand validated poses and training through overlapping regions; revisit failures; extend training only when measured quality improves | Best supported full-recording reconstruction, with weak regions and incomplete coverage identified |
| R5 — verify the final 3D view | 1–2 days | D7: compare old/new fixed views and local free movement; verify replay/pause/explore, reload and matching transforms; record remaining defects | Selected DB3 model, reproducible viewer launch/configuration and a before/after reconstruction report |

**Initial reconstruction allowance: approximately 12–20 focused working days.** Additional pose-recovery attempts or recapture can extend it. At each phase, publish the problem, candidate change, measured result and next decision. If a phase fails, retain the baseline and address that failure or execute a bounded D8 alternative; do not treat the schedule as permission to bypass a geometry gate.

**Reconstruction milestone required before later work:** the chosen model has consistent calibration and effective cameras, checked static geometry in the accepted regions, an old/new comparison over the full route including difficult regions, and a working freely navigable 3D view. Remaining missing/ambiguous surfaces must be documented. If this recording cannot support an acceptable whole-scene result, the milestone is a documented limitation and explicit decision on recapture or a partial deliverable—not a silent switch to another project task. There is no promise that every transparent or unseen surface can be recovered.

After that milestone, schedule the deferred work sequentially:

| Later phase | When | Work |
|---|---|---|
| Editable mesh | After R5 or an explicit decision to accept a limited reconstruction | M1–M3 using the accepted personal-scene geometry; extraction, cleanup and an editor roundtrip; estimate 3–5 working days for an initial asset, revising after inspection |
| General pipeline optimization | After the reconstruction and initial mesh quality baselines are frozen | Profile actual bottlenecks; select only useful O1–O6 experiments; improve reuse/resume/deployment; estimate an initial 3–5 day profiling and targeted-improvement cycle |
| Additional method research | Only if a specific remaining quality problem warrants it | M4/2DGS or other broader experiments with their own scope and acceptance criteria |

The relative order of those later phases can be reconsidered once reconstruction is complete; neither consumes the active DB3 schedule. If the 6 December deadline in the existing school schedule still applies, use the remaining time primarily for the DB3 result and its evaluation/report. Reduce or defer mesh/optimization scope rather than diverting effort from the user's stated reconstruction priority. Supervisor feedback can help communicate that revised scope. [Existing schedule evidence](SCHEDULE_PROGRESS_AUDIT_20261109.md)

**Next implementation action:** freeze the DB3 baseline and small-region checks, then unify the SDK-checked projection path. Do not launch mesh cleanup, public-scene benchmarking or a general optimization campaign first. This document revision does not start implementation.

### 3.12 Research review log and implementation map

#### Priority revision — 10 November 2026

**Problem:** the first timeline mixed DB3 repair with parallel public-Room mesh work and early optimization, which did not match the user's clarified priority. **Decision and solution:** dedicate the active schedule to the best supported reconstruction of the existing DB3; defer mesh production and general optimization until its acceptance decision. Replaced the mixed timeline with R0–R5, separated the current and later acceptance checklists, and retained only diagnostics/engineering necessary for reconstruction in the active scope. This revision changes the plan only; no code, data, model or experiment was changed.

#### Review performed for this plan

- Inspected the current Git worktree; it contains substantial modified/untracked research work. No existing research files were reverted, staged, committed or rewritten.
- Read current cleaner, recovery/bridge/pose-graph, candidate-remap, training/depth-loader, mesh-extraction and 2DGS code paths; consulted saved A04/A05, RGB-D, mesh and school results.
- Queried the target DB3 in SQLite read-only mode to confirm sensor topics, message counts, timestamp span and file size; did not decode every raw frame or recompute the full source hash.
- Re-read current baseline/candidate graph reports, SDK numerical results, baseline model verification and mesh provenance. Historical metrics in this plan are attributed to saved runs, not newly rerun experiments.
- Visually inspected the saved personal-scene capture/render sheet, personal dense ball-pivot mesh sheet and public-Room A05-Abs mesh/depth sheet. They show blur/ghosting in several personal-scene regions, fragmented/missing sensor-mesh surfaces, and recognizable but incomplete public-Room mesh geometry. These are selected saved views, not a complete new interactive inspection or physical accuracy test.
- Queried the current GPU/driver and checked primary RealSense, COLMAP, Open3D and 2DGS references for the proposed directions.
- Created only this plan. Its evidence and proposed solutions serve as this review's research log; no additional experiment-log file was changed.

#### Problems confirmed in current source and proposed ownership

| Finding/problem | Current owner | Proposed solution | Required verification |
|---|---|---|---|
| Normal cleaner and isolated SDK candidate use different remap routes | `scripts/clean-realsense-scene.py`, `scripts/prepare-room-sdk-rectified-candidate.py`, `scripts/bridge-room-sdk-candidate.py` | One versioned projection module and consistent regenerated dependencies | Pinned SDK numerical checks plus withheld physical boundaries |
| Success gates favor registration and fitted residuals | `scripts/run-clean-realsense.py`, `scripts/run-new-recording-120k.py` | Explicit structural, alignment, pose and surface validation states | A structurally connected but independently inconsistent fixture must not receive an accuracy pass |
| Bridge/SfM edges carry fixed confidence proxies | `scripts/bridge-clean-rgbd.py`, `scripts/recover-clean-rgbd.py`, `scripts/recover-rgbd-scene.py` | Measurement-derived uncertainty; robust complete-graph refinement | Synthetic bad-loop/degenerate cases and withheld region checks |
| Pose export lacks real landmark tracks for global BA | `scripts/recover-rgbd-scene.py` | Preserve/rebuild feature tracks before joint refinement | Track/reprojection audit; independent geometry checks |
| Sparse solver uses scalar weighting and shared treatment of radians/meters | `scripts/sparse-rgbd-pose-graph.py` | Whiten residuals by justified uncertainty and audit solver transition | Known-transform/noisy-constraint controls; matched local scene |
| Point initialization does not require multiview support | `scripts/recover-rgbd-scene.py` | Support-aware fusion with an uncertainty-preserving output | Coverage and support/conflict reporting on withheld observations |
| Depth supervision lacks per-sample confidence in the inspected path | `examples/datasets/colmap.py`, `examples/simple_trainer.py` | Confidence/mask transport and robust normalized loss | Invalid/empty-depth controls and D5 paired ablation |

## R0–R3 execution checkpoint: 10 November

The bounded R2/R3 campaign and integrity audit have executed; this is not a claim
that every scientific exit gate is closed. See
[measured R2/R3 results and artifacts](DB3_R2_R3_RESULTS_20261110.md) and
[machine phase decision](../results/db3_r2_r3_20261110/phase_decision.json).

- **R0:** frozen source/baseline integrity reverified; complete.
- **R1:** numerical projection, replacement inputs and timing work complete;
  independent precise physical alignment remains partial.
- **R2:** fresh real tracks, explicit metric bundle adjustment, 3/~10 FPS local
  comparisons and native bridge/revisit tests executed. The higher-rate matte
  and exhibit feature/depth checks pass conditionally, but actual exhibit
  structures remain unsupported/inconsistent. No globally verified merge.
  Concrete D8 separate-submap/local-reconstruction experiment executed; retain
  the original whole-room model as uncertain context.
- **R3:** completed the paired off/current/confidence 7k matrix, repaired a real
  zero-depth NaN-gradient failure and verified finite selected checkpoints. The
  matte local appearance/consistency follow-up passes. An additional 15k
  support/coverage candidate improves exhibit mean metrics and coarse depth
  checks, but still fails independently frozen statue/wall/floor/post checks;
  do not promote it or mark difficult-region acceptance complete.

Preserve both successful and negative gates. Resolve the failed physical support
and geometry checks before whole-recording expansion; no further full 120k run,
mesh work or threshold relaxation is justified by these local mean metrics.
| Weight exports are not complete resumable training state | `examples/simple_trainer.py` | Separate full-state resume checkpoint from deployment exports | Interrupted-versus-continuous replay comparison |
| Runner stage cache compares commands without full dependency identity | `scripts/run-clean-realsense.py`, `scripts/run-new-recording-120k.py` | Fingerprinted inputs/code/config plus output verification | Targeted dependency-change/reuse checks |
| Mesh extractor uses expected depth, coarse bounded defaults and a default-built Runner | `scripts/gaussians_to_mesh.py` | Restore exact run/cameras; measure actual sampling; validate surface depth/units | Synthetic geometry controls and fixed-checkpoint extraction comparison |

These are implementation facts or identified validation gaps. Their individual contribution to the visible DB3 defects is not yet experimentally isolated.

#### Log format for every future experiment

Record: question; observed failure; hypothesis; source/data/config hashes; exact command/environment; fit/development/check identities; changed factor; fixed factors; predicted effect; predeclared acceptance and stop criteria; outputs; per-region and overall metrics; runtime/memory; visual examples including failures; decision; limitations; next action. Append to the project experiment log and link the run's machine-readable manifest. Failed runs and rejected candidates remain part of the evidence.

#### Active DB3 reconstruction acceptance checklist

- [ ] One verified calibration/projection convention across every DB3 stage.
- [ ] Independently withheld geometry checks support accepted regions; unknown areas remain visible in reporting.
- [ ] Full-room context is preserved without presenting uncertain surfaces as accurate measurements.
- [ ] Claimed reconstruction/depth improvements are demonstrated with fixed-input or clearly attributed whole-pipeline comparisons, not inferred from public Room.
- [ ] Fixed captured, held-out and local free-view comparisons include the difficult regions and any regressions.
- [ ] The chosen DB3 model reloads and supports replay, pause and free exploration with matching cameras/transforms.
- [ ] Raw sources, baselines, rejected trials, transforms, splits and provenance remain recoverable.
- [ ] Final claims distinguish appearance, internal consistency, physical accuracy and unresolved capture limitations.
- [ ] R5 is complete, or an explicit decision records the acceptable partial result/recapture route before later work begins.

#### Deferred mesh and optimization acceptance checklist

- [ ] A Gaussian-derived mesh and any sensor-derived mesh are clearly distinguished.
- [ ] At least one cleaned mesh is edited, saved and reopened successfully in the intended editor.
- [ ] Manual surface completion is distinguished from measured/reconstructed geometry.
- [ ] Stage time, RAM/VRAM and actual viewer latency are measured on the current environment.
- [ ] Resource-saving changes preserve the accepted DB3 reconstruction and mesh quality within declared tolerances.



## R0–R3 re-audit: 11 November

See [the cleanup and verification report](DB3_R0_R3_CLEANUP_20261111.md). The final relevant suite passes 277 tests; immutable source/input/checkpoint/camera verification passes. Fail-closed acceptance preserves the recorded negative gates. The 90k cap causes a measured post-support loss, but uncapped/dense/tighter support variants do not resolve every exhibit structure. User scope remains the existing recording with explicit physical limits. R0 is complete; R1 precise physical alignment and R3 difficult-region acceptance remain open; R2 uses the verified separate-submap fallback without a global merge. Do not silently mark the scientific exits complete.

## R4 execution and partial-delivery checkpoint: 11 November

See [the R4 measured report](DB3_R4_RESULTS_20261111.md). Full-route screening
and the bounded D8 delivery are complete; an accepted improved whole-scene
reconstruction remains **unachieved**. Across all 477 selected identities, 132
of 317 eligible adjacent intervals pass local checks. All 185 failures were
retried with native observations; zero meet both link and endpoint gates.
Another 159 intervals are excluded by frozen reservations, not measured failures.
The independent audit passes for all 1,688 fitted inputs, source exclusions,
chain composition, sixteen raw dense-depth reproductions and delivery hashes.
The focused relevant suite passes 35 tests.

Deliver the preserved 120k-update whole-room context (all 477 cameras freshly
rendered finite, with matching frozen poses/normalization) and the separate
narrowly accepted matte candidate. Keep exhibit unpromoted; do not merge worlds
or launch longer training on failed geometry. Full-route uncertainties and
temporal support are mapped, without claiming physical surface coverage or
proving recovery impossible. R1 physical alignment, R2 distributed global
placement and R3 exhibit structures remain unresolved. R5 interactive verification
is still pending; this checkpoint does not mark it complete or start mesh work.

## R5 prerequisite review checkpoint: 11 November

[Fresh prerequisite review](DB3_R5_PREREQUISITE_REVIEW_20261111.md) verifies source/model integrity and executes one additional frozen camera-only direct-depth refinement. The matte control passes, but the difficult target still fails image alignment (3.050/7.401 px median/p90 against <2/<4 px). Existing physical, difficult-structure and global-placement gates remain open. R5 was not started under the user's prerequisite condition; no whole-scene model is promoted. This checkpoint preserves the available D8 partial delivery without converting viewer functionality into geometry acceptance.

## R5 operational start and recovery checkpoint: 12 November

The current request to unblock progress supersedes blocking every R5 activity.
R5 **operational verification has started** on the preserved delivery: real-browser
replay/pause/orbit/return/reconnect and live exact-camera restoration pass for the
context model. Use `scripts/run-db3-r5.ps1`, which verifies the selected delivery
before launching. Final reconstruction acceptance remains separate and incomplete;
this does not mark R4 improved whole-room reconstruction complete or constitute
the user's acceptance of a partial result as final.

New original-template and learned-correspondence recovery trials were executed
and rejected on measured evidence; no new model was promoted. See the
[recovery and R5 start report](DB3_R4_RECOVERY_R5_START_20261112.md) for exact
comparisons, source preservation, verification and remaining limitations.
