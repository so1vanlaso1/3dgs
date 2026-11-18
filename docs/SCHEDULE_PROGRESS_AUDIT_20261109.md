# School schedule versus repository evidence

**Follow-up:** the user subsequently authorized closing Weeks 1–10 gaps. See [completion results and remaining qualifications](SCHOOL_COMPLETION_RESULTS_20261109.md). The table below preserves the initial audit, not the final catch-up state.

Assessment date: 9 November 2026, Asia/Saigon. Source schedule: `report/25d26565-488e-4e09-9426-1186840fbfda.xlsx`, sheet `Project Schedule`, A1:D15. Excel date serials were decoded using the 1899-12-30 epoch. Week 10 ended 8 November; Week 11 ends 15 November; final deadline is 6 December 2026.

## Scope and method

The schedule requires photographs -> optimized 3DGS -> editable polygon mesh, including rendering, geometry, count, timing, and editability evaluation. CCD-GS is a historical research direction, not evidence of the official school project title. The current progress report uses “3D Gaussian Splatting” / “Efficient Densification and RealSense Reconstruction.” The workbook does not state an official title.

Read the schedule, current reports, A04/A05 results, relevant implementation and experimental mesh report; checked that the dense experimental PLY and A05 confirmation summaries/audits exist. No training, evaluation, or software tests were rerun. Historical results below are attributed to saved evidence. Missing means not established by inspected repository evidence; supervisor activities could exist outside this repository. The workbook was unchanged.

## Milestones

| Week / ending | Assessment | Achieved and remaining |
|---|---|---|
| 1 / Sep 6 | Mostly complete | Repository, research scope and experiment logs exist. End-to-end editable-mesh acceptance criteria and weekly supervisor review records are not established. |
| 2 / Sep 13 | Partial | Extensive densification literature and source audits exist. A unified comparison covering geometry improvements and Poisson/Marching Cubes extraction is not established. |
| 3 / Sep 20 | Partial | Detailed CCD research specification, controls and ablations exist. A current proposal/system diagram integrating Gaussian reduction, surface constraints and Gaussian-to-mesh conversion is not established. |
| 4 / Sep 27 | Partial / external evidence needed | Public indoor benchmarks and personal RealSense recordings selected. No supervisor presentation/feedback approval evidence found. |
| 5 / Oct 4 | Mostly complete | Public and self-collected data preparation, pose recovery, splits and preprocessing scripts exist. Physical pose/calibration validity remains unresolved for the museum room; complete mesh data flow is missing. |
| 6 / Oct 11 | Complete for baseline research | B00 baseline runs, configurations, checkpoints, renders and metrics are documented; self-collected reconstructions also exist. This is the baseline gsplat implementation, not a separate reproduction of every original-paper condition. |
| 7 / Oct 18 | Partial | Multiscene baseline comparisons and visual failure analysis complete. Experimental ball-pivoting meshes exist from RGB-D source clouds, but no chosen, validated extraction route from optimized Gaussians is established. |
| 8 / Oct 25 | Substantially complete | Budgeted growth, attributed-error/edge ranking, Long-Axis Split and absolute-gradient variant reduce Gaussian count with controlled rendering comparisons. Count reduction does not independently identify redundant primitives or certify preserved surfaces. |
| 9 / Nov 1 | Partial | Sensor-depth supervision and geometry diagnostics implemented; small controlled depth experiment exists. Validated surface continuity constraints combined with the reduction method and geometric comparison are missing. |
| 10 / Nov 8 | Partial; scheduled deadline passed | A04 factorial ablation, budget/tuning studies and A05 tests exist. Reduction-versus-surface-constraint ablation and actual Gaussian-to-mesh conversion are not complete. RGB-D experimental PLY is not that conversion. |
| 11 / Nov 15 | Early trials only | Raw ball-pivot mesh exports exist. Validated denoising, smoothing, simplification, hole repair and a documented import/edit task remain unestablished. |
| 12 / Nov 22 | Partial preparation | Strong rendering/count studies on public scenes and separate RealSense experiments. A unified baseline/improved pipeline across public and personal scenes, reference-geometry evaluation, extraction timing and mesh usability are missing. |
| 13 / Nov 29 | Partial preparation | Progress report, figures, results and failure analyses exist. They do not constitute the full thesis or supervisor-reviewed final draft. |
| 14 / Dec 6 | Not established | Final photos-to-edited-mesh rerun, release package, final thesis, presentation and demonstration remain to be completed. |

## Main measured achievements

- A04: four public indoor scenes, three paired seeds, 7,000 updates; mean count reductions 28.37% Room, 24.25% Counter, 35.27% Kitchen, 32.31% Bonsai. Mean PSNR changes +0.5688, +0.1361, +0.1297, +0.0298 dB respectively; Kitchen and Bonsai intervals include zero. Factorial split/clone study and budget sweep exist. Source: `results/research_completion_20261027/REPORT.md`.
- A05 absolute-gradient candidate: Room, Counter, Kitchen, seeds 42/43; all six PSNR changes positive (+0.04191 to +0.25887 dB versus A04), counts reduced 24.33–32.59%; five of six predefined practical gates passed. Counter seed 43 missed the PSNR gate; small SSIM losses occur in some comparisons. Views were exposed during development, threshold and gradient policy changed together. Source: `docs/A05_ABSGRAD_RESULTS_20261107.md` and saved campaign summaries/audits.
- CCD mechanism study: zero of 36 blocks passed its practical gate. A completed negative experiment, not a successful CCD allocation method. First residual A05 also had negative equal-count warm-start results despite two promising single-seed end-to-end screens. Sources: progress report, A05 results.
- RealSense: capture extraction, RGB-D pose recovery, depth supervision, training, exports, replay/free-camera viewer, calibration and trajectory diagnostics. New room baseline: 417 training / 60 held-out views, 120k updates, 1,063,961 Gaussians, PSNR 23.1133. Geometry accuracy remains unverified. Source: `docs/RESEARCH_PROGRESS_REPORT_20261106.md`.
- Experimental dense mesh: 900,000 source-cloud vertices, 546,983 triangles, 423,950 vertices used by faces; holes, noise and uncertain geometry persist. The input is sampled RGB-D point data, not an extracted optimized Gaussian surface. Sources: `results/room_improve_sdk_dense_test_mesh_20261103/report.json`, `scripts/mesh-room-sdk-experimental.py`.

## Problem and next actions

Problem: substantial research effort has validated rendering/count tradeoffs, while the school deliverable also needs geometry and editable assets. Good PSNR and connected camera graphs do not close that gap.

1. By Nov 15: freeze the working reduction candidate and a small feasible scene; select and implement an explicit route from optimized Gaussians to a polygon mesh. Treat source-depth meshing as a separate baseline. Verify geometry/scale and record raw mesh defects. Document a current full pipeline and acceptance criteria.
2. By Nov 22: implement necessary mesh cleanup; open and edit the mesh in design software; run the same end-to-end baseline and candidate on selected public and personal scenes, with count, rendering, geometry where a reference exists, timing and mesh usability. If surface regularization is retained in scope, isolate its effect from reduction.
3. By Nov 29: assemble the full thesis from the existing evidence plus mesh results and obtain supervisor feedback.
4. By Dec 6: rerun representative photographs-to-mesh cases and package source, configurations, checkpoints, mesh samples, reproduction guide and demo.

Repository packaging risk: the working tree contains many modified/untracked research files, including current reports and A05 code/results documentation. Local existence does not establish a committed or distributed final submission.
