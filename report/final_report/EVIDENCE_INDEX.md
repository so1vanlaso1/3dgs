# Final report supplement

Open [the report](report/final_report/final_report.pdf), or start with [the Windows reproduction guide](report/final_report/REPRODUCE.md) and [the Room runner](report/final_report/reproduce_room.ps1). Extract the entire ZIP before running commands. The supplied source is a snapshot of the project files used for the report, including the CUDA headers needed by gsplat. The report links to the evidence IDs below inside the PDF. This index maps each ID to bundled files.

The public Room workflow starts from the official benchmark download. The source and environment instructions generate checkpoints locally. The bundle includes the saved Room mesh and edit for inspection. The raw museum recording, private inputs, full training checkpoints, and full archives of per-run renders remain in the project workspace. Historical records can contain original workspace paths and refer to those archives. The measured summaries, hashes and code linked below are included.

**E1. Public data and splits** — [Indoor dataset audit](docs/SCHOOL_DATASET_QA_20261109.md).

**E2. Local RGB-D preparation** — [Local fitting](docs/DB3_R2_R3_RESULTS_20261110.md) · [Full-route review](docs/DB3_R4_REVIEW_20261111.md) · [Source-following holdouts](docs/DB3_MOVING_HOLDOUT_RESULTS_20261111.md).

**E3. Development and attribution** — [Strategy code](gsplat/strategy/budget.py) · [A04 multi-seed campaign and split factorial](results/research_completion_20261027/REPORT.md) · [A04 run records](results/research_completion_20261027/all_runs.json) · [Absolute-gradient study](docs/A05_ABSGRAD_RESULTS_20261107.md) · [Seed 42 metrics](results/a05_absgrad_20261107/summary.json) · [Seed 43 metrics](results/a05_absgrad_s43_20261107/summary.json) · [Experiment log](docs/EXPERIMENT_LOG.md).

**E4. Original paired benchmarks** — [Room](results/week12_comparison_20261119/room_pair.json) · [Counter](results/week12_comparison_20261119/counter_pair.json) · [Kitchen](results/week12_comparison_20261119/kitchen_pair.json) · [Bonsai](results/week12_comparison_20261119/bonsai_pair.json) · [Matte](results/week12_comparison_20261119/matte_pair.json) · [Matte sensor-depth comparison](results/week12_comparison_20261119/review/matte/sensor_depth_consistency.json) · [Comparison runner](scripts/week12_comparison.py) · [Surface extractor](scripts/gaussians_to_mesh.py).

**E5. Surface ablation and controls** — [Factorial study and known-plane result](docs/SCHOOL_COMPLETION_RESULTS_20261109.md).

**E6. Original public Room edit** — [Cleanup](results/week11_public_room_20261119/cleanup.json) · [Independent reopen](results/week11_public_room_20261119/design_reopen.json) · [Cleanup code](scripts/week11_room_cleanup.py) · [Blender code](scripts/week11_room_blender.py). A complete saved asset from the fresh run is indexed under E10.

**E7. Secondary acquisition** — [Progress report](report/progress_report/research_progress.pdf) · [Ordinary-video workflow](docs/MP4_TO_3DGS.md).

**E8. Controlled Room replay** — [Study runner](scripts/final_report_study.py) · [Main protocol](results/final_report_study_20261119/protocol.json) · [Tight-cap protocol](results/final_report_ranking_20261119/protocol.json) · [Audit](results/final_report_study_20261119/final_audit.json) · [Verified outcomes and timing probes](report/final_report/evidence/room_study.json) · [Camera order](report/final_report/evidence/room_camera_order.json).

**E9. Fixed-checkpoint matte grid** — [Study runner](scripts/final_report_geometry.py) · [Grid results and shared-support depth](report/final_report/evidence/matte_grid.json).

**E10. Fresh public Room workflow** — [Data audit](report/final_report/evidence/room_input_audit.json) · [Pipeline audit](report/final_report/evidence/room_pipeline.json) · [Raw PLY](results/final_report_room_replay_20261119/raw/raw_mesh.ply) · [Extraction record](results/final_report_room_replay_20261119/raw/report.json) · [Cleaned PLY](results/final_report_room_replay_20261119/design/room_cleaned.ply) · [Cleanup record](results/final_report_room_replay_20261119/design/cleanup.json) · [Camera crop record](results/final_report_room_replay_20261119/design/silhouette_cameras.json) · [Saved Blender edit](results/final_report_room_replay_20261119/design/room_tabletop_edited.blend) · [Edit creation record](results/final_report_room_replay_20261119/design/design_create.json) · [Reopen measurements](results/final_report_room_replay_20261119/design/design_reopen.json).

**E11. Archived readback** — [Mesh topology and areas](report/final_report/evidence/topology_audit.json) · [Per-view PSNR and Bonsai sensitivity](report/final_report/evidence/per_view_rgb_audit.json).

**E12. Frozen Garden and Room follow-up** — [Runner](scripts/final_report_followup.py) · [Original protocol](results/final_report_followup_20261120/protocol.initial.json) · [Verified protocol](results/final_report_followup_20261120/protocol.json) · [Garden input manifest](data/garden_factor4/dataset_manifest.json) · [Paired check summaries](report/final_report/evidence/followup.json) · [Garden per-view PSNR](results/final_report_followup_20261120/garden_per_view.json).

**Presentation figures** — [Room quality and size](report/final_report/figures/slides/room_cap_tradeoff.png) · [Room learning curves](report/final_report/figures/slides/room_progress.png) · [Room seeds and outdoor evaluation](report/final_report/figures/slides/paired_followup.png) · [Main benchmark](report/final_report/figures/count_psnr.png) · [Split and gradient development plot](report/final_report/figures/ablation.png) · [Topology changes](report/final_report/figures/topology.png).

Verify the extracted files from the archive root in PowerShell:

```powershell
Get-Content MANIFEST.sha256 | ForEach-Object {
    $expected, $path = $_ -split '  ', 2
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $expected) {
        throw "Digest mismatch in $path"
    }
}
```
