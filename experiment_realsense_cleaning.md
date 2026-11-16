# RealSense cleaning and scene reconstruction research log

Small-scene pose follow-up: see `experiment_small_scene_pose.md` for the isolated
28-view stairs experiment, pose diagnostics, matched training controls, failures
and final comparison. Outputs are separate from both completed courtyard models.

## 2026-10-30: new recording, requested end-to-end cleaning and 3DGS

- Input: `20261030_111354.db3`, 25,154,482,176 bytes; read-only access.
- Discovery: 6,996 RGB frames, 6,997 depth frames, 23,327 accelerometer and
  46,646 gyro samples; about 233.45 seconds. Both camera calibrations, depth
  scale (0.001 m/unit), and reference-to-stream extrinsics are recorded.
- Problem: older scripts hard-code the previous DB3; original COLMAP path
  rectifies RGB and aligns sensor depth after pose estimation. No exposure gate.
- Solution: new `scripts/clean-realsense-scene.py` performs exposure and blur
  rejection, sharp candidate selection, duplicate rejection, inverse-Brown RGB
  rectification with valid crop, depth range/invalid/discontinuity filtering,
  median denoising within valid neighborhoods, timestamp/header pairing gates,
  calibrated depth alignment, and nearest-Z collision handling before COLMAP.
- Recorded factory calibration is validated and reused, not independently fitted.
  IMU is decoded/exported and checked; no claim of visual-inertial pose fusion.
- Default sample: 3 FPS, width 848, near/far 0.3/6 m, pair tolerance 20 ms,
  sharpness floor 25 at 640x360. Thresholds are configurable and decisions logged.
- Reconstruction uses a fixed measured PINHOLE camera after rectification to
  preserve depth pixel geometry. Sensor scale is checked against training-view
  SfM observations before enabling depth supervision. Every eighth registered
  view is held out for image evaluation; jointly estimated poses mean this is
  not independent geometry validation.
- Raw RGB values are preserved except geometric resampling. Gross exposure
  failures are rejected rather than inventing clipped detail. No moving-object
  segmentation, temporal depth smoothing, hole filling or IMU pose integration.
- Reference: https://dev.realsenseai.com/docs/projection-in-realsense-sdk-2-0/
  (inverse Brown mapping, camera-Z depth, metric rigid stream transforms).

Execution results, failures, fixes and exported artifacts are appended below.

### Preparation and unit verification

- SQLite `PRAGMA quick_check` passed on the new 25 GB sample.
- Tests: `tests/test_realsense_cleaning.py` plus the existing adapter tests:
  **8 passed**. New coverage includes full-field inverse-distortion round trip,
  invalid/far-depth rejection, median noise removal, metric camera-Z after
  extrinsics, non-identity reference transforms, exposure/blur gates, and IMU
  CDR field offsets/units. Existing robust rigid-fit tests remain passing.
- Test environment issue: the default system pytest temporary directory denied
  access. Fixed invocation with `--basetemp .cache/pytest-realsense-cleaning`;
  no change to scientific code was needed. Existing unknown `env` pytest
  configuration warning remains unrelated.
- The new capture is an outdoor courtyard (stairs, plants, trees and buildings),
  confirmed by visual inspection of cleaned frame 100. This is different from
  the prior indoor recording; no prior poses or checkpoints are reused.
- Two previous-recording viewers are already running. They are left intact;
  training is capped to 500k Gaussians for the 6 GiB GPU. Render benchmarking
  will explicitly record that the existing viewers remain open.

### Cleaning results and numerical repair

- 701 selected frames from 6,996 RGB frames, covering the full 233-second capture
  at 3 FPS. No tested candidates crossed the conservative exposure/blur gates;
  selection is not evidence that every frame or pixel is artifact-free.
- Sharpness min/median/max: 213.52 / 2359.89 / 4376.24. RGB/depth timestamp gaps
  min/median/p95/max: 2.378 / 7.263 / 8.654 / 11.878 ms; header max also 11.878 ms.
- Valid source-depth fraction min/median/max: 6.37% / 42.90% / 94.93% after
  range/discontinuity masking. Missing depth is retained as missing, not filled.
- IMU: all 23,327 accel / 46,646 gyro samples decoded, finite and strictly
  increasing. Median sample spacing 10.014 / 5.007 ms. Median acceleration
  magnitude 9.696 m/s^2; median angular speed 0.340 rad/s. Exports are CSV.
- Discovered in all-file audit: occlusion grouping originally ran in float64,
  then saved float32 coordinates. Values near half-pixels could round into a
  previously occupied pixel after serialization. Fixed `clean_depth` to cast
  before Z-buffer grouping. Repaired the saved selected samples by re-Z-buffering
  float32 coordinates, keeping nearest Z; recorded exact counts in
  `depth_quantization_repair.json`. RGB and raw recording are unaffected.
- Source SHA256 measured during preparation:
  `f5815899f4aecc8ebcfd3e31a222bc9806856c24d7abcbdcd949e18a88370ca3`.
- Quantization repair affected 13 files, removing one duplicate selected sample
  each (13 total). All 701 saved depth files then passed finite/positive/bounds
  and unique-target-pixel checks, retaining 15,999-16,000 samples per frame.
- Frame-to-IMU associations exported for all 701 frames. Maximum nearest accel
  gap 6.059 ms; gyro gap 3.096 ms (header timestamps agree). This associates
  samples on the capture timeline, not hardware sync or inertial pose fusion.
- `depth_alignment_qa.jpg` provides eight RGB/depth-overlay diagnostics across
  the capture; inspect together with numerical scale checks, not as ground truth.

### RGB-only reconstruction and metric recovery

- RGB-only SfM produced five models (431, 38, 120, 93, 20 views; some overlap).
  Largest model covers 61.48% of the 701 selected frames, with 28,773 sparse
  points and mean reprojection error 0.773 pixels. Retained as a separate result.
- Sensor/SfM scale gate on that model passed: 32,777 training-view scale pairs,
  72.00% within 10% of the median scale, median relative error 5.28%, frame-scale
  relative p90 15.60%; 0.395682 COLMAP units per meter. This is internal scale
  consistency, not independent metric ground truth.
- Original model dataset validation passed for all 377 training / 54 holdout
  views, with finite images and positive finite aligned sensor-depth samples.
- Problem: good individual images and low reprojection error did not yield
  connected coverage of the whole capture. Added `recover-clean-rgbd.py`, using
  cleaned-frame COLMAP matches, aligned metric depth and robust 3D/PnP links.
  Scale-gated relative poses from each SfM fragment strengthen the pose graph.
  All five fragments passed their training-depth median scale error gate (<8%).
- Recovery never reopens the old recording and uses no prior-scene poses. It
  shares the previously tested robust rigid fitting and pose-graph exporter,
  but prepares inputs from the new calibrated cleaned images and depth.
- Ceres reports the requested 8 threads are bounded to its build maximum of 6;
  this is a performance warning, not a failed reconstruction.

- Initial metric SIFT/PnP graph still had components of 431, 230, 21 and smaller
  fragments, so depth alone did not automatically solve coverage. Its largest
  component had p90 translation/rotation constraint residuals 6.05 cm / 0.712 deg.
- Added `bridge-clean-rgbd.py`: reads the actual source path from the manifest,
  reuses the same depth cleaning and calibration, and attempts dense RGB-D
  odometry only at adjacent disconnected boundaries. Each step must pass depth
  warp agreement, motion bounds, and forward/backward consistency. No filling
  of invalid depth is used. It logs accepted and rejected links individually.
- First bridge attempt at roughly 15 FPS accepted 8/12 disconnected boundaries,
  raising coverage to 441/701 (62.91%); the important 449->450 boundary failed.
  Next attempt uses every recorded RGB frame (~30 FPS) to reduce inter-step
  motion. The original attempt reports and pre-bridge graph are retained.
- Full-rate retry accepted all three remaining component boundaries: 8->9,
  449->450 and 679->680. **701/701 selected views** now form one connected model.
  Optimized graph: 2,538 constraints, 40 gross loop hypotheses rejected;
  translation residual p50/p90/p99 = 0.30 / 5.91 / 15.56 cm;
  rotation residual p50/p90/p99 = 0.034 / 0.676 / 1.831 degrees.
  These are internal constraint residuals, not absolute reconstruction accuracy.
- Depth-backed training initialization uses 180,000 voxel-selected points from
  training views only. The metric RGB-D dataset preserves all cleaned frames,
  unlike the largest RGB-only fragment. Both datasets remain available.
- The reusable runner now includes full-rate boundary bridging after metric
  recovery. Running its automatic selection path on the prepared sample before
  training verifies this behavior rather than relying on manual-only recovery.

### Automatic reproduction and training

- `run-clean-realsense.py --prepared` reproduced the complete 701-view graph
  from the cleaned input without manual edge edits. Full-rate bridging accepted
  11/12 boundaries; the rejected 62->63 link was unnecessary because another
  accepted boundary already connected that small fragment. No forced pose link.
- Automatic dataset selection chose the RGB-D dataset for improved coverage and
  p90 residuals below 8 cm / 3 degrees. All 613 training / 88 heldout views passed
  dataset validation; all sensor-depth samples tested positive and finite.
- Final targeted tests, including the existing sparse pose-graph regression:
  **9 passed**, one pre-existing pytest `env` configuration warning.
- Training launched with 180k initialized Gaussians, 500k cap, 12,000 updates,
  SH degree up to 3, depth weight 0.01, batch size 1, packed rasterization,
  no cached RGB images, and refinement stopped at update 6,000. Exact command
  and stage logs are saved in `results/realsense_20261030_111354_clean/`.
- Environment saved in `environment.json`: Python 3.12.10, PyTorch 2.9.1+cu128,
  pycolmap 4.2.1, Open3D 0.20.0, NumPy 2.5.2, SciPy 1.18.1, OpenCV 5.0.0.93.
- Windows PowerShell renders some native stderr progress records as
  `System.Management.Automation.RemoteException`; these alone are not a failed
  training run. The runner uses native process exit codes plus verified output
  artifacts to determine success. Standalone preparation emitted COLMAP logs on
  stderr but completed all four stages and saved successful dataset validation.

### Trained scene and visual assessment

- 12,000 optimizer updates completed, final 499,559 Gaussians. Saved training
  loop time 488.95 seconds; reported peak PyTorch allocation 0.727 GiB (this is
  not total device/desktop VRAM). Final held-out evaluation:
  PSNR **20.2594 dB**, SSIM **0.55901**, LPIPS **0.50232** over 88 views.
  Initial evaluation: 12.8131 dB / 0.32694 / 0.97425. No matched baseline campaign
  was run to attribute gains to individual cleaning or recovery components.
- Loaded the exported comparison for frame 100: courtyard, tree, potted plants,
  stairs, railing and building are recognizable. The splat is visibly softer
  than captured RGB, especially foliage, railing edges and distant windows.
  This is a working first full-capture scene, not a photorealistic or artifact-free
  reconstruction. More iterations alone are not evidence that pose drift,
  moving foliage, reflective surfaces or missing depth have been corrected.
- Trained artifact: `train/ckpts/final_post_update.pt`; exported preview PLY:
  `preview/scene.ply` in `results/realsense_20261030_111354_clean/`.

### Final export, provenance and usage

- Runner completed successfully (exit 0; `status.json` = complete). Checkpoint
  tensors and every rendered video pixel are finite. Exported PLY reloaded with
  all **499,559** Gaussians; file size **117,897,401 bytes**.
- Comparison video fully decoded: **701 frames**, 1696x470 side-by-side RGB,
  4 FPS, **175.25 seconds**, 205,107,743 bytes. Capture is left, rendered splat
  right. Playback is a diagnostic at fixed 4 FPS, not original capture timing.
- Renderer QA: 50 benchmark views after 5 warmups, median 3.292 ms / p90 5.309 ms,
  mean-derived 266.1 FPS, peak render PyTorch allocation 0.332 GiB at 848x470.
  Scope excludes display, image loading and video encoding; previous viewers
  remained open, so this is not an isolated interactive-viewer FPS claim.
- Full source SHA256 after training/preview matches the preparation hash:
  `f5815899f4aecc8ebcfd3e31a222bc9806856c24d7abcbdcd949e18a88370ca3`.
  `provenance.json` saves both hashes, source identity, driver/GPU and code hashes.
- Local viewer launched on port 8083, with camera index 100 showing the courtyard.
  Background launcher PID 5872. Existing prior-recording viewers were preserved.
  HTTP root returned 200; listening server PID 2900.
- Open `http://localhost:8083`. Restart later with
  `powershell -ExecutionPolicy Bypass -File scripts/view-clean-realsense.ps1 -Background`.
- Reproduce with `.venv/Scripts/python.exe -u scripts/run-clean-realsense.py`.
  See `docs/realsense_clean_pipeline.md` for new recordings and stage artifacts.
- Completed scope: pre-COLMAP RGB/depth cleaning, factory-calibration handling,
  timestamp/header pairing and alignment, checked IMU export/associations,
  connected pose recovery, dataset checks, depth-supervised 3DGS training,
  PLY reload, all-view CUDA renders, decoded video and raw-source integrity.
  Explicit limits remain: no IMU pose fusion, no independent recalibration or
  ground-truth geometry test, no moving-object segmentation, no guarantee of
  artifact-free imagery or recovery of clipped/missing sensor information.

## Full recorded 30 FPS follow-up

- User requested every recorded frame, removing only unqualified frames. Added
  `--all-frames`, checking every source RGB message exactly once. Plain `--fps 30`
  is insufficient: timestamp-bin collisions can otherwise silently drop frames.
  Near-duplicate rejection is disabled in this mode; qualified similar frames stay.
- Extraction is separate at `data/realsense_20261030_111354_all30`; the 701-view
  dataset, checkpoint, source recording and viewer remain available.
- Tests after all-frame selection change: **10 passed**. Regression explicitly
  verifies that colliding timestamp bins retain every source ID in original order.
- Pose plan: same-recording calibrated 3 FPS relative poses anchor a sparse
  high-rate graph; additional poses use measured RGB-D correspondences and gated
  odometry, never pose interpolation. This is anchored densification, not an
  independent SfM reconstruction or geometry benchmark.
- Retain all qualified frame files. Hold out the same original 88 views and
  exclude training views within +/-0.2 seconds of those timestamps for comparison.
  These temporal-buffer views remain in the dataset, not deleted as bad frames.
- The user asked about the Gaussian hard limit: both completed 3 FPS training and
  the new planned full-rate training use `budget` with a **500,000** hard cap.
  Completed run: 499,559 splats. This is configurable and independent of frame
  count; equal optimizer updates/cap do not prove equal compute or convergence.
- User then explicitly requested removing the hard count cap and increasing
  training to **20,000 updates**. Full-rate runner now uses standard `default`
  densification, with no `cap_max`/Gaussian budget, instead of the budget strategy.
  The old 3 FPS model remains unchanged. Refinement still stops at step 6,000;
  this is a densification schedule boundary, not a Gaussian count cap. A GPU OOM
  would be reported as a failure, not silently worked around with a count cap.
- Comparison scope updated: 30 FPS + uncapped standard densification + 20k
  updates versus prior 3 FPS + capped budget strategy + 12k updates. It cannot
  isolate the effect of frame rate, Gaussian budget, strategy or training length.
- Sequential full-frame extraction was slower than practical for 6,996 frames.
  Interrupted that attempt explicitly (only its two Python wrapper/worker PIDs
  20756/16532), preserving its log and the prior scene. Added a four-worker,
  resumable all-frame extractor with per-frame atomic provenance/decision records.
  It retains identical quality/alignment gates and checks every source ID exactly
  once; BLAS/OpenCV threads are bounded per worker to avoid oversubscription.
  Fast PNG compression changes file encoding, not image pixels. Existing partial
  images lacking provenance sidecars are recomputed, not assumed valid.
- Four-worker extraction measured roughly 20 accepted frames/second after its
  integrity/metadata checks (first 601 frames in 31.8 seconds); no quality gate
  changes were needed. It uses per-source-message deterministic depth sampling.
- Final full-rate extraction: **6,996 accepted / 0 rejected / 6,996 source RGB
  frames accounted for**, with 6,996 provenance records, PNGs and aligned-depth
  NPZs. Processing took 341.83 seconds after initial integrity/metadata checks.
  No artificial rejections were introduced to force a smaller dataset.
- All-frame sharpness min/median/max: 86.98 / 2272.58 / 4376.24, above the
  configured minimum 25. Timestamp pair min/median/p95/max:
  2.001 / 7.262 / 8.636 / 14.721 ms, below the 20 ms gate. Valid source-depth
  fraction min/median/max: 5.98% / 43.50% / 95.20%; invalid depth remains missing.
- Split: **5,970 training / 88 original heldout / 938 temporal-buffer views**.
  All 6,996 files remain qualified and retained. Buffer views are withheld from
  optimization for evaluation isolation, not dropped by image/depth cleaning.
- Early identical-source RGB checks at original frames 0, 5, 20, 40 were
  byte-identical as decoded pixels despite different PNG compression settings.
- Feature extraction completed for every one of the 6,996 cleaned views.
- Matching optimization: full-rate poses use metric RGB-D RANSAC/PnP plus verified
  same-recording anchors, so the additional COLMAP epipolar-verification pass is
  redundant in this branch. Restarted matching with `skip_geometric_verification`
  enabled, reusing cached correspondences. The initial roughly 1,167-image
  matching attempt was interrupted (PIDs 15748/19272); its log is retained.
  Existing epipolar-inlier matches remain usable, but every accepted new pose
  still passes metric verification and the final graph quality/coverage gates.
- The installed pycolmap build reports `has_cuda=False`; CPU descriptor matching
  remained the bottleneck even after skipping epipolar verification. Benchmarked
  exact PyTorch CUDA descriptor matching on 21 cached pairs: retained 100% of the
  existing COLMAP matches in each tested pair, with additional candidates allowed
  only by standard distance/ratio/reciprocal criteria. Warm mean ~1.094 ms/pair
  in this small early-capture sample, excluding descriptor IO; not a whole-run
  throughput claim. CPU matching was explicitly interrupted again (15104/21220),
  preserving cached matches and its log.
- Added `match-clean-rgbd-cuda.py`: uint8 SIFT descriptors / 512, exact L2 distance
  <0.7, Lowe ratio <0.8, mutual nearest neighbor check. Full-rate neighbors up to
  four frames apart. Mandatory metric pose validation remains unchanged. Two
  new descriptor tests passed (ambiguous ratio, distance, reciprocal and empty
  candidate cases). This changes the candidate matcher, not the pose-quality gate.
- Full matching completed with **27,974** temporal-neighbor pairs: 12,524 cached
  native COLMAP pairs and 15,450 newly computed CUDA pairs, containing 10,196,070
  new candidate correspondences. Median new-pair descriptor-match duration
  1.304 ms, excluding descriptor IO; measured peak Torch allocation 0.0711 GiB.
  No assumption that a descriptor match alone is a verified geometric constraint.
- All-file depth audit passed on 6,996 frames: exactly 16,000 saved samples per
  frame, all finite/positive/in bounds and unique after Z buffering. Frame/IMU
  associations covered all 6,996 views: maximum accel gap 6.640 ms, gyro 5.422 ms.
- Recovery retained all 701 exact-source metric pose anchors and assembled
  29,276 metric constraints before optimization. The larger graph uses the
  existing sparse robust SE(3) solver. Full-rate dataset validation now requires
  a registered pose for every qualified frame, rather than silently training a
  subset of otherwise qualified retained files.
- All 6,996 qualified views form one connected graph. Sparse optimization
  converged by its `ftol` criterion (status 2, 21 function evaluations): weighted
  RMS 0.019337 -> 0.010461; robust cost 11.529 -> 4.708. This establishes local
  solver convergence/internal consistency, not global optimality or ground truth.
- Boundary bridging now returns without reoptimizing when every frame is already
  connected; this avoids repeating the same large solve and dataset export.
- Final full-rate pose result: **6,996/6,996 registered**, zero unregistered views,
  29,198 optimized constraints, 78 gross loop hypotheses rejected. Translation
  constraint p50/p90/p99 = 0.907 / 2.534 / 10.616 cm; rotation =
  0.127 / 0.338 / 0.982 degrees. Full coverage and configured residual gates pass.
- Input-anchor trajectory diagnostic after aligning the first common camera:
  701 anchors, translation difference p50/p90/max = 0.709 / 1.410 / 2.829 cm,
  rotation difference p50/p90/max = 0.072 / 0.256 / 0.568 degrees. Saved to
  `rgbd/anchor_consistency.json`; this compares to the baseline, not ground truth.
- Full-rate pre-training validation passed on all 5,970 training samples and
  retained all 88 heldout views. Saved `cfg.yml` confirms `max_steps: 20000` and
  `gsplat.strategy.default.DefaultStrategy`, with no `cap_max` parameter.
- Combined targeted regression suite after the full-rate changes: **12 passed**
  (3.54 seconds), with only the pre-existing unknown pytest `env` option warning.
- Initial heldout metrics before full-rate optimization: PSNR 12.8983 dB,
  SSIM 0.32836, LPIPS 0.97855, 180,000 initialized Gaussians. Compare final metrics
  on the same heldout pixels; no inference of component-wise causal gains.
- Uncapped training grew beyond the previous 500k ceiling: 789,274 Gaussians
  after the final scheduled refinement/pruning at step 5,900. Refinement stops
  at 6,000 as configured; optimization continues to 20,000. Around this stage
  total GPU snapshot was ~3.2 GiB, without adding an artificial count cap.
- Execution interruption: a requested 30-second wait returned after 9,334.3
  seconds of wall time. Training remained alive and resumed normal ~21 updates/s
  around update 15,900. The saved training elapsed time therefore includes a
  long pause and must not be presented as isolated GPU compute time or compared
  directly to the uninterrupted older run. No checkpoint restart was performed.
- Requested 20,000 updates completed continuously, ending with **789,274**
  Gaussians. Peak recorded PyTorch allocation 1.2273 GiB; total device snapshots
  were larger because they include other allocations/viewers. Wall elapsed
  10,189.14 seconds includes the long pause and is not active GPU compute time.
- Trainer heldout metrics: **PSNR 19.9533 dB, SSIM 0.54451, LPIPS 0.51322**.
  Earlier 3 FPS capped scene: 20.2594 / 0.55901 / 0.50232. This requested combined
  change did **not** improve these heldout metrics. Preserve both scenes; do not
  silently present more frames/points/updates as better or attribute the change
  to FPS alone. The same-pixel comparison/export verification follows below.
- Final same-pixel comparison independently reproduced the trainer results over
  all **88 identical heldout RGB targets**: 3 FPS = 20.259428 dB / 0.559007 /
  0.502320; all-30 = 19.953273 dB / 0.544511 / 0.513221. Deltas = -0.306155 dB
  PSNR, -0.014496 SSIM, +0.010901 LPIPS (LPIPS lower is better).
- Final experiment trace confirms **20,000 completed updates**, uncapped
  `DefaultStrategy`, and **all 5,970 training views used 3-4 times**. The 88
  heldout and 938 temporal-buffer views remain reconstructed and stored, with
  no photometric training on those views. All 6,996 qualified source frames
  remain in the connected scene; no raw frames were rejected by the gates.
- Checkpoint tensors and all rendered comparison pixels are finite. PLY reload
  succeeded for 789,274 splats; **186,270,141 bytes**. Video fully decoded all
  88 frames: H.264, 2502x488, 4 FPS, 22 seconds, 36,450,090 bytes. Its three
  panels are captured RGB / older 3 FPS / full qualified 30 FPS. FFmpeg emitted
  an initial rate-estimation warning; explicit output FPS and full decode checks
  confirm a valid 4 FPS video.
- Visual review of the matched stairs/courtyard image: both models retain
  recognizable layout and plants; floor, wall and fine foliage detail remain
  soft. The full-rate model is not a clear visual improvement. Keep the older
  model as the better measured option and the new model as the requested trial.
- Final source SHA256 still
  `f5815899f4aecc8ebcfd3e31a222bc9806856c24d7abcbdcd949e18a88370ca3`.
  `provenance.json` additionally hashes the full matching database, capture/
  calibration/split manifests, optimized edges/poses, both checkpoints and code.
- Runner finished with exit 0 / status complete. Full-rate viewer launched on
  `http://localhost:8084` (launcher PID 12300), camera index 1000. Older scene
  remains separate on port 8083. Restart the new viewer with:
  `powershell -ExecutionPolicy Bypass -File scripts/view-realsense-all30.ps1 -Background`.

## 2026-10-30: why the DB3 scene looks worse than downloaded scenes

### Pose reconstruction follow-up

- Code audit: `recover-clean-rgbd.py` estimates metric 3D RANSAC/PnP links and
  optionally imports lower-rate same-recording relative-pose anchors;
  `bridge-clean-rgbd.py` uses bidirectional RGB-D odometry at disconnected
  boundaries. `sparse-rgbd-pose-graph.py` optimizes camera transforms against
  relative-pose edges; it does not jointly optimize image feature tracks and
  3D landmarks as bundle adjustment would. Reusing those anchors alone cannot
  independently validate or remove their inherited systematic errors.
- Proposed pose improvement, not executed: use static-region feature tracks,
  calibrated image geometry, robust RGB/depth motion estimates, verified revisit
  matches, global graph optimization and subsequent bundle adjustment. Keep
  rectified/cropped intrinsics and aligned depth consistent if calibration changes.
  Estimate high-rate motion where needed to preserve overlap, while selecting
  a smaller set of useful training views separately.
- Validation should include unseen static feature correspondences, multi-view
  depth/surface alignment, duplicated surfaces, plausible trajectory and revisit
  consistency. A smooth trajectory or small graph residual alone is insufficient.
  Absolute pose correctness needs independent surveyed/ground-truth references;
  a common arbitrary world orientation/origin is harmless for 3DGS.
- Reference checked: https://colmap.github.io/tutorial.html (SfM, shared camera
  intrinsics, verified matching, overlapping translated views). No pose recovery
  or new training was run for this explanatory follow-up.

- Problem: recorded RGB, depth and IMU did not produce the visual quality of
  downloaded benchmark scenes. This audit inspected saved artifacts and code;
  no training, source recording or existing checkpoint was changed.
- Verified current configs for both courtyard runs enable `depth_loss: true`,
  `sensor_depth_dir`, and `depth_lambda: 0.01`. The current COLMAP loader reads
  those sensor samples. Older RGB-first notes are not a description of these
  completed runs. Depth is a weighted training constraint, not exact geometry.
- Full-rate cleaning report: median valid source depth after 0.3-6 m range and
  discontinuity masking is 43.50%, minimum 5.98%. These are source-depth-grid
  fractions, not RGB coverage. The remaining depth is missing/rejected; distant
  courtyard structures may be outside the configured range. Saved supervision
  is sampled to 16,000 valid aligned depth observations per image.
- Pose evidence: initial RGB SfM fragmented into five models. RGB-D recovery
  connected all frames, but connectivity and low internal graph residuals do
  not establish absolute pose accuracy. `pose_opt: false` in both trained
  configs means splat training does not refine the supplied camera poses.
  IMU was exported/associated, not fused. Median RGB/depth timestamp offset
  is 7.262 ms; motion-related alignment error remains a candidate, not a finding.
- Visual check: `preview/comparison_0100.jpg` in the clean result shows much
  sharper captured RGB than rendered stairs, railings, plants and background.
  Input resolution alone therefore cannot explain the lost detail. Pose error,
  view inconsistency, unmodeled foliage motion and insufficient optimization
  remain hypotheses; none was isolated by an ablation in this audit.
- Confirmed same-target comparison over 88 views: 3 FPS 20.2594 dB PSNR,
  0.55901 SSIM, 0.50232 LPIPS; all30 19.9533 / 0.54451 / 0.51322.
  More frames, uncapped densification and 20k steps together did not improve
  these metrics. All30 used 5,970 training views only 3-4 times each; its total
  step count does not establish per-view convergence. Several settings changed
  together, so this is not evidence that increasing FPS itself hurts quality.
- Downloaded room reference: `results/room_hq_gtx1660/cfg.yml` disables depth
  loss; saved val_step6999 reports 30.0335 dB / 0.90206 / 0.21165.
  These different-scene scores illustrate the observed gap but are not a
  controlled depth/no-depth or capture-device comparison. Exact downloaded
  scene intended by the user remains unspecified.
- Explanation: depth constrains distance but does not guarantee sharp color,
  accurate global camera poses, static content, or sufficient viewpoint coverage.
  The original 3DGS method starts from calibrated cameras and sparse points:
  https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ . Stereo depth also
  depends on texture, exposure and operating configuration:
  https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance/ .
- Proposed next experiment, not executed: select a compact static well-covered
  region; inspect/reduce pose and RGB/depth edge inconsistencies; compare RGB-only
  and confidence-filtered sensor-depth training with identical images, poses,
  strategy, updates and heldout targets. Separately test camera refinement and
  adequate per-view optimization. Preserve both completed scenes. A slower,
  overlapping multi-height capture of that region provides a simpler control.
