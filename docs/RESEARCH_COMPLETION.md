# A04 research campaign: finalized completed work

Authorized 27 October 2026. Campaign root: `results/research_completion_20261027`.
The user subsequently requested skipping all unfinished work. Training and its
monitor were stopped; no automatic continuation remains.

**Final output:** [completed-results report](../results/research_completion_20261027/REPORT.md).
There are 21 completed runs and 10 matched pairs: three pairs each for Room,
Counter and Kitchen, and one for Bonsai. One additional completed Bonsai B00 run
is unpaired. The interrupted A04 run is preserved and excluded. The component
ablation, budget sweep, tuning and full CCD-GS study were skipped. Crop/failure
analysis and the prepared implementation remain available.

The completed runs used the GTX 1660 SUPER. Final status is written to
`status.json`; individual runs retain their command configuration, training log,
metrics, final post-update checkpoint, PLY, camera sequence, initialization hash,
growth history and verification record. Incomplete runs require inspection.

## Original planned A04 matrix (unfinished items skipped)

- Repetitions: Room, Counter, Kitchen, Bonsai × seeds42/43/44 × B00/A04 (24 runs).
- Component isolation: Room × three seeds × four cells (random/LAS operator ×
  keep clones/replace clones). Three A04 cells reuse the replication campaign;
  nine additional runs. Ranking and 700k cap are common to all four cells.
- Budget operating points: Room seed42, caps400k/700k/1000k (two additional runs).
- Tuning: Room seed42 at (distance, opacity) = (0.30,0.4), (0.30,0.8),
  (0.50,0.4), (0.50,0.8), with the existing (0.45,0.6) as reference.
  Choose the best PSNR with SSIM/LPIPS guardrails; confirm a changed winner on
  Room seeds43/44 and Counter seed42. Room is explicitly a development scene.
- All runs use 7,000 updates and the existing training/evaluation protocol.
  A separate DataLoader generator prevents structural RNG from changing camera order.
- Historical one-seed runs are preserved and excluded from the new seed averages.
- Crops: all four scenes at seed42; exhaustive fixed windows, largest gain,
  largest regression, largest residual; save all errors and selection coordinates.

The factorial isolates the LAS operator bundle from clone replacement. Separate
placement/scale/opacity causal effects are outside this factorial. Three seeds
give imprecise uncertainty; the budget sweep has one seed. Timing remains
system-load-sensitive. No outdoor generalization or untouched-test tuning claim.

## Prepared CCD-GS mechanism scope (full study skipped)

`scripts/ccd_mechanism.py` implements Stage 0 manifest loading, full-state
snapshots, bounded discarded probes, candidate lineage, identical continuations,
and gradient/pose controls. Tiny smoke studies tested the harness. The planned
H1/H2 scientific matrix was not executed; no CCD-GS efficacy finding follows.

The original plan and source snapshots are retained as provenance; planned rows
must not be presented as completed experiments.

## Current artifacts

- [Final campaign status](../results/research_completion_20261027/PROGRESS.md).
- [Current-run visual review](../results/research_completion_20261027/VISUAL_REVIEW.md).
- [Historical image crops and failure counts](../results/research_completion_20261027/historical_review/REPORT.md).
- [Written visual review](../results/research_completion_20261027/historical_review/VISUAL_REVIEW.md).

The report distinguishes completed runs from skipped work. Smoke-study
results are harness validation only and are excluded from scientific summaries.

## Continuation authorized 28 October 2026

The previously skipped campaign is running again under the original captured
protocol. The interrupted Bonsai seed43 A04 directory, log, and prior final
status/report are preserved in `results/research_completion_20261027/interrupted_20261027_archive/`.
Completed verified runs are reused; the interrupted run is restarted. Live
state is in `results/research_completion_20261027/status.json` and the
continuation logs. Results are not complete until the scheduled jobs,
verification, analysis, and CCD mechanism gate finish successfully. The prior
report describes the 27 October cutoff until analysis replaces it.
