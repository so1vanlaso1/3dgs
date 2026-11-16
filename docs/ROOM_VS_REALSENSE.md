# Room versus RealSense: verified comparison, 2026-10-30

## Scope and problem

The user asked why the normal Room scene looks much cleaner than their RealSense scene. Compared the active Room baseline with the latest full courtyard recording, `20261030_111354.db3`, both its 3 FPS and all-frame runs. The earlier `20261029_140739.db3` is a separate capture and is noted separately. No reconstruction, training, or viewer changes were performed for this audit.

## Verified differences

| Property | Room baseline | Latest RealSense 3 FPS | Latest RealSense all-frame |
|---|---|---|---|
| Total registered views | 311 | 701 | 6,996 |
| Photometric training views | 272 | 613 | 5,970 |
| Evaluation views | 39 | 88 | Same 88 source targets |
| Temporal-buffer views excluded from training | None in standard split | None in standard split | 938 |
| Prepared image dimensions | 1557 x 1038 | 834 x 463 | 834 x 463 |
| Completed optimizer updates, batch size 1 | 7,000 | 12,000 | 20,000 |
| Average image selections per training view | 25.74 | 19.58 | 3.35; actual range 3-4 |
| Sensor depth supervision | Disabled | Enabled, weight 0.01 | Enabled, weight 0.01 |
| Camera optimization during splat training | Disabled | Disabled | Disabled |
| Densification | DefaultStrategy | BudgetStrategy, 500k cap | DefaultStrategy, uncapped |
| Final Gaussians | 841,017 | 499,559 | 789,274 |
| PSNR / SSIM / LPIPS | 30.03351 / 0.90206 / 0.21165 | 20.25943 / 0.55901 / 0.50232 | 19.95327 / 0.54451 / 0.51322 |

Room versus RealSense metrics use different scenes, resolutions and test sets. They describe reconstruction fidelity on each capture, not an isolated camera-device, depth-loss, or algorithm comparison. The two courtyard runs do use matched heldout targets, but several training settings changed simultaneously. Average selections are updates divided by training views, not proof of convergence or matched compute.

## Evidence and interpretation

1. Room preserves the benchmark's supplied COLMAP poses and 112,627 sparse initialization points. Initial RGB-only reconstruction of the selected RealSense frames produced five models; the largest contained 431/701 views. RGB-D recovery subsequently connected 701/701 and then 6,996/6,996. This establishes reconstruction difficulty. It does not measure final absolute pose error. The recovery graph optimizes relative camera constraints rather than jointly refining all image tracks and landmarks in global bundle adjustment. Both final courtyard configs keep supplied poses fixed during splat optimization; IMU associations were exported but not fused. Pose/calibration inconsistency is therefore a priority hypothesis, not a proven sole cause.
2. Visual inspection of `results/realsense_20261030_111354_clean/preview/comparison_0100.jpg` shows captured stairs, railings and leaves substantially sharper than the rendering. Room's `renders/val_step6999_0000.png` retains much more of its reference detail. Lost RealSense detail cannot be explained by sensor image resolution alone. Room does provide about 4.19 times as many prepared pixels per view.
3. Courtyard contains fine foliage and distant structures. Motion of foliage, changing appearance, reflections and incomplete multi-view coverage can violate consistent static-scene reconstruction; their individual contributions have not been measured here. Do not report observed foliage as proof that it moved.
4. Median valid source-depth fraction after masking is 43.50%, minimum 5.98%, with the configured 0.3-6 m range. This fraction refers to the source depth grid, not RGB image coverage. Supervision samples 16,000 aligned observations per frame. Depth is an incomplete weighted distance constraint, not a guarantee of accurate camera poses or sharp appearance. Median RGB/depth timestamp difference is 7.2624 ms; motion-related misalignment is a candidate requiring validation.
5. The full-rate run sees each training image only 3-4 times. More total steps are spread over many more views, while refinement ends at update 6,000. This motivates an optimization-budget experiment but does not establish undertraining as the sole cause. More splats did not yield better matched-view results.
6. A completed compact-region pose comparison also exists: 6,000 updates per branch and three heldout views. Baseline PSNR 20.74396 versus refined 20.71422; SSIM 0.60287 versus 0.60245; LPIPS 0.18711 versus 0.18306. Mixed results do not establish that the tested pose refinement solves the gap. Its two trajectories differ; neither is independent ground truth.
7. Earlier recording: the separate 15 FPS experiment registered 5,201/5,204 views and improved matched-view PSNR from 18.88727 to 20.29982 over its 2 FPS baseline. This is a different capture and should not be mixed with the courtyard's 701/6,996 counts.

## Recommended resolution, not executed

Use a compact static region with well-overlapping translated views as the diagnostic control. Validate feature reprojections, cross-view surface alignment, revisit consistency and calibrated RGB/depth alignment. Test stronger track-based pose refinement without assuming its success. With fixed images, poses, initialization, strategy and heldout targets, compare RGB-only against confidence-filtered depth supervision; separately test a larger per-view optimization budget. A fresh slow capture of static indoor objects from several heights would help separate capture difficulty from pipeline problems. Preserve all existing baselines and accept changes on matched-view image quality plus surface/trajectory checks.

## Source artifacts

- `results/room_hq_gtx1660/large_dataset_report.json` and `cfg.yml`.
- `data/realsense_20261030_111354_clean/reconstruction_report.json`, `calibration_clean.json`, `rgbd/dataset/dataset_validation.json`.
- `data/realsense_20261030_111354_all30/cleaning_report.json`, `calibration_clean.json`, `rgbd/dataset/dataset_validation.json`.
- `results/realsense_20261030_111354_all30/comparison.json`; both courtyard `train/cfg.yml` files.
- `results/realsense_small_pose_20261030/comparison.json`, `trajectory_comparison.json`.
- `results/realsense_15fps_experiment/REPORT.md`, `METHODS.md`.
- `experiment_realsense_cleaning.md` for earlier pipeline/code audit.
- Original method: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ .
- Capture and SfM guidance: https://colmap.github.io/tutorial.html .

The local artifact counts, configs, results and two comparison images were checked in this audit. Causal rankings remain hypotheses until controlled experiments isolate them.
