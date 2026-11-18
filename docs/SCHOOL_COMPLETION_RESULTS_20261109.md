# Schedule catch-up through 8 November 2026

Work executed 9 November 2026. This completes missing repository deliverables and a preliminary surface experiment; it does not backdate work, establish supervisor approval, or certify accurate geometry.

User-confirmed title: **Improving 3D Gaussian Splatting for efficient and geometrically consistent 3D reconstruction, followed by conversion into editable polygon meshes.** The user confirmed that the supervisor presentation has not occurred.

[Supervisor review package](../report/school_completion_20261109/README.md) includes the eight-slide editable briefing. Slides were rendered and inspected; the exported PPTX package was checked for eight slides and ZIP integrity. No PowerPoint application test is claimed.

## Week-by-week evidence

| Week / original ending | Repository deliverable now available | Remaining qualification |
|---|---|---|
| 1 / Sep 6 | [Objectives, scope, evaluation criteria, architecture and review routine](SCHOOL_PROJECT_PLAN_20261109.md); existing experiment log | Earlier weekly meetings are not asserted. |
| 2 / Sep 13 | [Seven-paper annotated matrix plus Poisson, Marching Cubes, TSDF and ball-pivot comparison](SCHOOL_LITERATURE_MATRIX_20261109.md) | Focused method review; not a reproduction of all papers. |
| 3 / Sep 20 | Current proposal, method rationale, Mermaid system diagram, baselines and ablation protocol in the project plan | Integration and empirical contribution; individual components have prior work. |
| 4 / Sep 27 | Dataset/capture inventory and an eight-slide supervisor briefing | **Presentation, supervisor feedback and incorporating that feedback remain outstanding.** |
| 5 / Oct 4 | [Current dataset QA](SCHOOL_DATASET_QA_20261109.md): four public sets and personal room, image hashes/dimensions, splits, finite/rigid poses, duplicate/blur screening | No exact duplicates found. Low-texture flags did not justify deleting benchmark frames. Personal room physical pose/calibration uncertainty remains documented. |
| 6 / Oct 11 | Existing B00 baseline checkpoints, renders, settings, metrics and setup scripts | gsplat implementation baseline; not a separate official-code reproduction. |
| 7 / Oct 18 | Existing four-scene benchmarks and failure analysis; newly executed optimized-Gaussian-to-mesh trial with chosen uniform TSDF extraction | Raw meshes have holes and fragments. A completed extraction trial is not a finished asset. |
| 8 / Oct 25 | Existing A01–A05 reduction code, checkpoints and multiscene/seed controls | Count reduction with preserved rendering is established within the reported scope; individual redundant primitives are not certified. |
| 9 / Nov 1 | Implemented an optional image-edge-aware log-depth curvature constraint; trained both baseline+constraint and reduction+constraint, checked stability and compared raw surfaces | **Preliminary geometry research only.** Curvature improves modestly, but topology is mixed and absolute geometric improvement is not established. |
| 10 / Nov 8 | Completed reduction x constraint factorial using matched cameras/initialization; evaluated lambda 0 versus 0.01; selected main candidate; four raw PLY/OBJ meshes and extraction/reproduction scripts | Fine parameter tuning and reference-geometry validation are not claimed. The constraint experiment does not justify replacing the current reference. |

## New measured results

Room, seed 42, 7,000 updates. Two verified archived controls and two new constraint runs. All four have matching initialization and 7,000-camera-sequence hashes. Each new arm's 39 evaluation target images match its control byte for byte. Initial image metrics agree within 0.0001. All new final model tensors are finite.

| Arm | PSNR | SSIM | LPIPS | Gaussians | Raw mesh triangles | Components |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 29.99812 | 0.901684 | 0.209726 | 831,326 | 119,613 | 1,241 |
| Baseline + constraint | 30.05953 | 0.902064 | 0.210307 | 827,764 | 114,598 | 1,131 |
| Reduced (A05-Abs) | 30.73701 | 0.909146 | 0.192058 | 452,485 | 119,987 | 1,162 |
| Reduced + constraint | 30.73650 | 0.909322 | 0.191455 | 450,398 | 119,400 | 1,216 |

The prior's rendering guardrails passed in both comparisons. On five fixed held-out views with identical all-arm opacity support, mean edge-aware log-depth curvature changed from 0.00541200 to 0.00514657 for baseline (-4.90%), and 0.00544199 to 0.00527539 for reduced (-3.06%). These are smoothness proxies, not physical accuracy. The reduced+constraint mesh had more components (1,216 versus 1,162) and boundary edges (17,700 versus 16,807), despite improved curvature. This prevents a general mesh-continuity improvement claim.

**Candidate decision:** retain A05-Abs, surface_lambda=0, as the main Week-10 model; retain surface_lambda=0.01 from update 3000 as an opt-in studied alternative. The validation choice is between these two settings, not an exhaustive weight search. Both model families and negative/mixed findings are preserved.

## Mesh extraction and visual inspection

Meshes use expected depth and RGB rendered from optimized Gaussian checkpoints at 64 training cameras. No sensor points or held-out depths are fused. Uniform TSDF uses a common cube around the shared SfM 1st–99th percentile bounds enlarged 20%, resolution capped at 192, target voxel 0.01*scene_scale and truncation four actual voxels. Actual bounds, resolution, voxel size, source hash, topology and timings are saved per arm.

Inspected all four five-view depth comparison sheets. Large room surfaces and furniture silhouettes are recognizable. Plant leaves/thin legs lose detail or fragment; holes remain near foreground edges, furniture and view boundaries. Some tabletop details merge into a simpler surface. The prior does not visibly eliminate these problems. These are initial polygon meshes suitable for the Week-10 extraction milestone; cleanup and editing validation remain Week 11 work.

Main candidate: [PLY](../results/school_completion_replay_20261109/meshes_uniform/reduced/raw_mesh.ply), [OBJ](../results/school_completion_replay_20261109/meshes_uniform/reduced/raw_mesh.obj), [five-view preview](../results/school_completion_replay_20261109/meshes_uniform/reduced/mesh_depth_comparison.jpg).

## Problems encountered and solutions

1. Four Windows workers duplicated the decoded image cache and reduced free memory to about 0.5 GB. Zero workers restored several GB of available memory. However, sampler progression differed after the first epoch. The first full run was rejected as a paired experiment and preserved under `results/school_completion_20261109`.
2. Added explicit saved-camera replay, verified the loader against all 7,000 archived indices and reran both prior arms. Final recorded hashes match the controls. The accepted campaign is `results/school_completion_replay_20261109/surface`.
3. The local Open3D 0.20 scalable TSDF returned zero voxels on a known flat-plane control. Uniform TSDF reconstructed that control and passed a regression test. Switched all four extractions to identical bounded uniform volumes. Failed scalable output is preserved under `meshes/`; accepted output is under `meshes_uniform/`.
4. Training source hashes match the frozen campaign snapshot. Extraction was corrected after training and has a separate source snapshot/hash under `extraction_evidence/`. No retraining was required for the backend correction.

## Verification and evidence

- 31 surface/budget/split tests passed after the trainer replay change; one independent known-plane TSDF test passed. A pre-existing pytest warning reports an unknown `env` option.
- Four 60-update training smoke arms completed. Smoke results are harness checks; final full runs use the later frozen replay implementation.
- Meshes have finite vertices, nonzero faces, and successful PLY save/reload count checks. OBJ exports also exist. No design-software edit test is claimed.
- [Full training/mesh table](../results/school_completion_replay_20261109/REPORT.md), [configuration/image/source audit](../results/school_completion_replay_20261109/analysis.json), [common-support curvature](../results/school_completion_replay_20261109/curvature_proxy.json), [execution protocol](../results/school_completion_replay_20261109/surface/protocol.json).

## Reproduce in fresh directories

```powershell
.\scripts\run-school-completion.ps1 -m pytest tests/test_surface.py tests/test_budget_strategy.py tests/test_long_axis_split.py tests/test_mesh_backend.py -q
.\scripts\run-school-completion.ps1 scripts/complete_school_pipeline.py --root results/school_completion_NEW
.\scripts\run-school-completion.ps1 scripts/measure_school_curvature.py --root results/school_completion_NEW
.\scripts\run-school-completion.ps1 scripts/analyze_school_completion.py --root results/school_completion_NEW
```

The generic pipeline now uses the corrected extractor directly under `meshes/`. The accepted historical catch-up run used `extract_school_meshes.py` after its first extraction failed, so its final mesh folder is `meshes_uniform/`. The analyzer supports both layouts. Full baseline/no-prior controls remain required archive inputs; their hashes are verified.

## What still needs action

Supervisor presentation and feedback are the outstanding non-repository milestone through Week 10. The technical deliverables now exist, but the research aim of reliably improved geometric reconstruction remains preliminary. Independent geometric validation, mesh cleaning and a real editing task are still necessary before claiming the final project outcome. Weeks 11–14 retain their original cleanup, full evaluation, thesis and demonstration work. Nothing here establishes a final submission package or a push to a remote repository.
