# Statue video versus current RealSense pipeline — 2026-11-03

## Scope and source confidence

User requested comparison with the pasted summary of Massimo Vendola's video, https://www.youtube.com/watch?v=YMbZvOLSK6E. Video-specific details below are attributed to that supplied summary; direct YouTube retrieval was throttled, so the exact camera edition, workstation, settings and frame count were not independently verified. Current-project facts were checked against saved artifacts and scripts. No reconstruction, training, filtering or viewer changes were performed for this comparison.

## Main finding

The two workflows have different outputs and processing stages. The supplied workflow makes a textured photogrammetry mesh, manually cleans it, samples new points on its surfaces, and renders a stylized animation. The currently requested local viewer displays 180,000 RGB-D initialization samples before Gaussian optimization. The saved local trained model instead contains 1,063,961 Gaussian primitives for novel-view rendering. Neither the raw depth cloud nor trained Gaussian centers are equivalent to a cleaned mesh resampled into artistic points.

## Comparison

| Stage | Supplied video summary | Current newest capture |
|---|---|---|
| Capture | Insta360 ONE RS 1-Inch camera; 360 video; elevated pole | RealSense D455 RGB, stereo depth and recorded IMU; source 20261030_134523.db3 |
| Image selection | 200 equirectangular frames | 4,764 RGB and 4,764 depth frames recorded; 477 sharp RGB frames selected at 3 FPS |
| Camera model | Spherical camera in Metashape | Undistorted/cropped pinhole images, prepared width 848; preview images 848 x 470 |
| Poses | Metashape image alignment | RGB COLMAP first; calibrated RGB-D feature constraints and pose-graph recovery because RGB coverage was low |
| Geometry | Reconstructed textured mesh | Back-projected depth samples transformed by estimated poses, voxel-selected and capped at 180,000 points |
| Cleanup | Manual deletion of unwanted/distorted mesh geometry | Exposure/sharpness/pairing checks; depth range, discontinuity and denoising masks; no manual surface cleanup or dynamic-object segmentation |
| Final representation | New points distributed on cleaned mesh faces, with texture colors | Current viewer: initialization points with RGB. Trained alternative: optimized anisotropic Gaussians with opacity and degree-3 spherical-harmonic appearance |
| Appearance | Blender materials, rim emission, custom effects, lighting and fog | Source RGB in Viser cloud; learned appearance in Gaussian rendering |
| Delivery | Authored camera path and Cycles-rendered animation | Interactive free-camera viewer and recorded captured-pose previews |
| Compute | i9-14900K and unspecified dual GPUs, according to summary | Saved run provenance: GTX 1660 SUPER, 6,144 MiB VRAM; Python/gsplat/CUDA and custom processing scripts |

## Why their result can appear cleaner

1. Points sampled from a mesh inherit that mesh's surface structure. They do not directly expose every noisy depth measurement. Meshing can also smooth detail or create incorrect surfaces; cleanliness is not accuracy evidence.
2. The supplied workflow explicitly removes artifacts before displaying points. The local raw cloud has not received comparable manual cleanup.
3. Elevated capture adds viewing angles of upper surfaces. A 360 camera observes many directions from each position, but cannot see through occluders or replace translated viewpoints.
4. Controlled camera paths, point radius, lighting, emission and fog change presentation. The video does not establish equal geometry quality under arbitrary free-camera inspection.
5. A larger workstation provides processing capacity. It does not repair missing views, inconsistent observations or inaccurate poses. No controlled hardware or software comparison is available here.

## Current verified weaknesses and limits

- RGB reconstruction's largest model registered 301/477 images (63.1%), with additional fragments. RGB-D recovery registered 477/477. Registration is camera connectivity, not surface completeness or trajectory accuracy.
- RGB-D graph report: 1,734 input constraints, 1,731 optimized; p90 translation residual 5.15 cm and rotation residual 0.924 degrees. These are fitted-constraint residuals, not ground-truth errors. This latest 477-view report names Open3D dense Levenberg-Marquardt; do not conflate it with the earlier full-rate sparse solver workflow.
- Initialization samples depth from individual views and does not require multi-view support for each point. Existing screening flags 2,513 points with at least three free-space conflicts and no sampled support. These are ambiguous diagnostic candidates, not validated deletion targets.
- Cleaning reports no dynamic segmentation, no IMU pose fusion, factory calibration without independent recalibration, and nearest-time pairing without hardware synchronization. Recorded IMU is not a fused visual-inertial trajectory.
- Trained configuration uses 417 training views and 60 held-out views, 120,000 updates, depth weight 0.01, fixed supplied poses and densification stopping at step 6,000. Final count is 1,063,961 Gaussians. More training has not been shown to resolve the suspected geometry problems.
- Existing diagnosis reports PSNR 23.1133 dB, SSIM 0.79857 and LPIPS 0.30898 on jointly reconstructed held-out cameras. These are image-quality results, not independent geometric accuracy. No matched metrics exist for the video.

## Practical solution by intended output

For an artistic point-cloud animation, a suitable extension is validated poses and depth -> multi-view depth fusion -> inspected/cleaned mesh -> textured surface sampling in Blender -> materials and camera animation. This is a proposed branch; no mesh fusion or Blender scene was created. Simply importing the current raw cloud and increasing point count would not reproduce the video's geometry preparation.

For better free-view 3DGS reconstruction, prioritize independent revisit/pose checks, RGB-depth alignment and multi-view depth consistency before another long training run. Existing bright openings, reflections, depth holes and people remain capture limitations. Matched controlled trials are needed before claiming Metashape, more frames, a new camera or more VRAM would solve them.

Do not select spherical camera calibration for the current D455 images. A 360 workflow requires corresponding calibration and a compatible downstream camera representation. A 360 camera's angular coverage is not a guarantee of greater subject detail or metric accuracy. Likewise, hardware alternatives listed in the supplied summary do not establish that LiDAR was used to reconstruct this particular statue.

## Evidence and external references

- data/realsense_20261030_134523_clean/capture_manifest.json and cleaning_report.json: selection, source, preprocessing, IMU and limitations.
- data/realsense_20261030_134523_clean/reconstruction_report.json and rgbd/report.json: registration, graph and point counts.
- results/realsense_20261030_134523_120k/train/cfg.yml, provenance.json and preview/verification.json: training, GPU, completed updates, saved model and preview dimensions.
- scripts/run-new-recording-120k.py and scripts/view-initial-point-cloud.py: actual pipeline and current point representation.
- docs/PIPELINE_DIAGNOSIS_20261102.md: prior evidence review, evaluation results and causal limits.
- Agisoft official spherical-image workflow: https://agisoft.freshdesk.com/support/solutions/articles/31000161406-processing-spherical-panoramas-in-agisoft-metashape
- Intel D455 specifications list operating range 0.6–6 m: https://www.intel.com/content/www/us/en/products/sku/205847/intel-realsense-depth-camera-d455/specifications.html . Local preprocessing accepts 0.3–6 m; that numerical mask should not be interpreted as a verified sensor accuracy range.

## Unresolved

Exact video hardware/settings are user-summary claims. No direct geometry comparison, mesh baseline, verified global trajectory, independently calibrated depth accuracy, matched Metashape test or new capture was produced. The relative contribution of capture, poses, depth, optimization and presentation remains unmeasured.
