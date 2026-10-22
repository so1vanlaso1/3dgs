# Budgeted densification experiment

`BudgetStrategy` changes the selection of clone/split parents in the core density
controller. It is inspired by [Taming 3DGS](https://humansensinglab.github.io/taming-3dgs/),
but does not reproduce its full scoring scheme, weighted sampling, opacity model,
or renderer optimizations. The original `default` strategy remains available.

## Allocation

- A quadratic count schedule limits net additions and reaches the cap at the last
  eligible refinement event. It uses the actual strict start/stop and reset-pause
  conditions. Insufficient eligible parents or subsequent pruning can leave the
  final count below the cap.
- Existing averaged positional gradients determine eligibility. Parents already
  below the opacity pruning threshold cannot consume growth slots.
- Every ten training steps during refinement, a detached contributor query reuses
  that training render's projections. It retains the four strongest `T * alpha`
  contributors for each pixel and attributes RGB L1 error to their global IDs.
- Error mass is averaged over contributing observations. Support from at least
  three sampled views receives full confidence; smaller observation counts are
  discounted. Gradient/error signals are scaled by their positive candidate
  median and clipped at twenty times that median.
- Candidates are ranked by the weighted sum of those signals, with deterministic
  top-K selection. `error_weight=0` gives the gradient-only budget control.
  Optional `edge_weight` adds attributed absolute ground-truth Laplacian detail.
- Existing clone/split geometry, optimizer-state updates, scene bookkeeping,
  opacity resets, and pruning are reused. Evidence resets after structural growth
  so children do not inherit their parent's attribution statistics.

The cap bounds live Gaussian count, not temporary memory during parameter
replacement. Extra scoring has a runtime and memory cost. Image error is a
heuristic for the need to grow, not a measured causal estimate of future loss
reduction. Only single-GPU classic 3DGS rendering is currently supported.

## Run on the local Windows setup

These commands train from the original SfM initialization for 7,000 updates at
1557 x 1038, using 272 training views and 39 held-out views. Results are separate
from `results/room_hq_gtx1660`; the budget launcher refuses an existing output
directory.

```powershell
# Gradient + contribution-weighted error
.\scripts\test-budget-dataset.ps1 -ResultDir results/room_budget700k_error

# Control: same count schedule, gradient-only selection
.\scripts\test-budget-dataset.ps1 -ErrorWeight 0 -ResultDir results/room_budget700k_gradient

# Optional edge term
.\scripts\test-budget-dataset.ps1 -EdgeWeight 0.25 -ResultDir results/room_budget700k_edges
```

Each run saves the configuration, model, PLY, validation renders and metrics. The
training statistics include every allocation event: scheduled target, eligible
parents, additions, pruning and resulting live count.

Compare completed experiments and check their artifacts/settings:

```powershell
.\.venv\Scripts\python.exe scripts/report-budget-experiment.py `
  --baseline results/room_hq_gtx1660 `
  --run error=results/room_budget700k_error `
  --run gradient=results/room_budget700k_gradient `
  --output results/budget_comparison.json
```

Use the JSON/Markdown report to compare all three quality metrics, count, runtime,
peak tensor allocation, and model size. A single seed cannot establish robust
quality equivalence. Training sampler settings match, but exact view order can
diverge as different split operations consume the random stream.

## Measured room results (2026-10-09)

All new variants completed 7,000 updates at the same image resolution, train/held-out
split and optimizer settings. The baseline is the existing saved room run, not a
fresh rerun. This is one run per variant with the trainer's fixed seed 42.

| Variant | Gaussians | PSNR | SSIM | LPIPS | Trainer seconds | Peak tensor GiB |
|---|---:|---:|---:|---:|---:|---:|
| Saved baseline | 841,017 | 30.0335 | 0.90206 | 0.21165 | 400.9 | 1.280 |
| Budget + gradient | 699,173 | 29.7823 | 0.89855 | 0.21555 | 385.4 | 1.101 |
| Budget + gradient + error | 699,266 | 29.8630 | 0.90130 | 0.21299 | 405.7 | 1.104 |
| Budget + gradient + error + edges (0.25) | 699,207 | 30.0166 | 0.90117 | 0.21405 | 409.7 | 1.116 |

The edge variant has 16.86% fewer Gaussians and PSNR within 0.017 dB of the saved
baseline. SSIM and LPIPS are slightly worse. Peak tensor allocation is 12.8% lower;
the trainer timer is 2.2% longer. Exact quality equivalence and repeatability across
seeds have not been established. Error attribution improved all three metrics over
the gradient-only budget control in these runs. Edges improved PSNR over error-only
selection, while its SSIM/LPIPS were slightly worse.

Artifacts and the full comparison report are under
`results/budget_allocation_experiment/`. Every checkpoint is finite, settings and
initial metrics match the baseline, allocation history stays within its budget,
and PLY counts match the checkpoints. Ten focused/existing strategy tests pass.
The edge checkpoint also passed a separate reload/evaluation: PSNR 30.0192,
SSIM 0.90113, LPIPS 0.21412, with the same 699,207 Gaussians. Its tiny difference
from the training validation is consistent with saving before the last optimizer
update. No additional training was performed during this reload check.

To reproduce the variant closest in PSNR, choose a new result directory:

```powershell
.\scripts\test-budget-dataset.ps1 -EdgeWeight 0.25 -ResultDir results/room_budget700k_edges
```

To view the already trained edge variant:

```powershell
.\scripts\view-room.ps1 `
  -Checkpoint results/budget_allocation_experiment/edges700k/ckpts/ckpt_6999_rank0.pt `
  -DataDir data/room_factor2
```

## Optional Long-Axis Split

`BudgetStrategy(split_mode="long_axis")` replaces cloning and random splitting
with two symmetric children. It uses the Improved-GS 3-sigma offset convention:
`+/-1.35*longest_scale`, longest-axis scale multiplier 0.55, other-axis multiplier
`sqrt(1-0.45**2)`, and activated opacity multiplier 0.6. Rotation and SH are copied.
The default `split_mode="random"` reproduces the existing budget variants.

```powershell
.\scripts\test-budget-dataset.ps1 -ErrorWeight 1 -EdgeWeight 0.25 `
  -SplitMode long_axis -ResultDir results/budget_allocation_experiment/longaxis700k
```

The trainer also accepts `--strategy.split-distance` (0 < value <= 0.5) and
`--strategy.opacity-reduction` (0 < value < 1). LAS rejects `revised_opacity=True`.
See [A04 in the experiment log](EXPERIMENT_LOG.md#16-a04-appearance-preserving-long-axis-split)
for protocol, evidence and results. This combines LAS with local candidate ranking;
it does not reproduce the paper's complete densification pipeline.

A04 completed 7,000 room updates with **597,989 Gaussians**, **30.5058 PSNR**,
**0.90561 SSIM**, and **0.20486 LPIPS**. Relative to A03, that is 14.48% fewer
Gaussians and +0.4891 dB PSNR, with SSIM and LPIPS also improved. Relative to
the original baseline it is 28.90% fewer Gaussians and +0.4723 dB PSNR. The cap
remained 700k; insufficient eligible candidates left the final count below it.
A paired frozen-checkpoint diagnostic found 29.31% less immediate RGB MSE than
the average of three random placements. Recovery-update savings are unmeasured;
these are one-scene, one-training-seed results.
The checkpoint reload passed at 30.5093 PSNR / 0.90558 SSIM / 0.20493 LPIPS,
with the same count. Nineteen strategy tests and all artifact checks passed.
