# 3DGS experiment log: baseline and budgeted Gaussian allocation

Recorded: **2026-10-09, Asia/Saigon**. Workspace: `D:\gsplat\gsplat`.

This record describes what was implemented, why it was tried, how each result was
obtained, and where its evidence is stored. It covers one saved baseline and four
completed advances/controls. It is intended as a source for a later project report.

## 1. Objective and experiment IDs

**Objective:** use fewer Gaussians while retaining approximately the same held-out
reconstruction quality at **7,000 optimizer updates**. Equal wall-clock time was
not a constraint; the cost of scoring was measured.

| ID | Method | Question | Status/date |
|---|---|---|---|
| B00 | Original `DefaultStrategy` | What quality/resources does the existing high-resolution room pipeline achieve? | Saved baseline, verified 2026-10-08 |
| A01 | Budget + gradient ranking | What happens when growth is limited to 700k and eligible parents are ranked by gradient? | Completed 2026-10-09 |
| A02 | A01 + contribution-weighted reconstruction error | Does allocating capacity to unresolved pixels improve the budgeted result? | Completed 2026-10-09 |
| A03 | A02 + ground-truth edge detail | Can additional detail awareness recover the PSNR lost at the smaller budget? | Completed 2026-10-09 |
| A04 | A03 + Long-Axis Split, replacing clones too | Does structured densification improve fidelity at the same 700k budget? | Completed and reload verified, 2026-10-09; see Section 16 |

The IDs describe the logical progression. **The original execution order was B00
historically, then A02, A01, A03, followed by an evaluation-only reload of A03.**
**A04 then ran with its own smoke, frozen-checkpoint diagnostic and reload**;
all A04 details and results are in Section 16.
The baseline was read from existing artifacts; it was not retrained during the
budget experiments. A01/A02/A03 each started from the original SfM initialization,
not from the previous variant's checkpoint.

Primary evidence:

- [Machine-readable comparison](../results/budget_allocation_experiment/comparison.json).
- [Generated comparison table](../results/budget_allocation_experiment/comparison.md).
- [Baseline verification report](../results/room_hq_gtx1660/large_dataset_report.json).
- [Implementation/run guide](BUDGETED_DENSIFICATION.md).

## 2. Shared dataset, training and evaluation settings

The dataset is the official **Mip-NeRF 360 room** scene, prepared from `images_2`.
The prepared directory already contains factor-two images, so the trainer uses
`data_factor=1`. Camera intrinsics were rescaled to the actual image dimensions;
source poses, points and colors were preserved. See the
[dataset manifest](../data/room_factor2/dataset_manifest.json) and
[preparation/verification guide](LARGE_DATASET_WIN.md).

| Setting | Value for B00/A01/A02/A03 |
|---|---|
| Dataset directory | `data/room_factor2` |
| Image resolution | 1557 x 1038 |
| Photographs | 311 |
| Training / held-out views | 272 / 39; `test_every=8` |
| Initialization | 112,627 SfM Gaussians; `init_type=sfm` |
| Initial opacity / scale multiplier | 0.1 / 1.0 |
| World normalization / global scale | Enabled / 1.0 |
| Updates / step scaler | 7,000 / 1.0 |
| Random seed | Trainer sets `42 + local_rank`; these single-GPU runs use 42 |
| Batch / image cache | 1 / enabled |
| Rasterization | Packed, classic 3DGS, pinhole camera |
| SH schedule | Maximum degree 3; increase every 1,000 steps |
| Reconstruction loss | 0.8 x RGB L1 + 0.2 x SSIM loss |
| Initial learning rates | Means 0.00016; scales 0.005; quaternions 0.001; opacity 0.05; SH0 0.0025; higher SH 0.000125 |
| Additional optimization | Pose/app optimization disabled; post-processing disabled |
| Regularization | Opacity 0; scale 0 |
| Rendering variants | Antialiasing, UT and eval3d disabled |
| Validation | At step indices 0 and 6999; final metrics average the same 39 held-out views |
| Quality metrics | PSNR and SSIM: higher is better; LPIPS/AlexNet: lower is better |
| Saved outputs | Checkpoint, PLY, configuration, validation renders and JSON statistics |

Density-control settings also match except for allocation and A04's split operator:

| Setting | Value |
|---|---:|
| Gradient eligibility threshold, `grow_grad2d` | 0.0002 |
| Clone/split scale threshold, `grow_scale3d` | 0.01 x scene scale |
| Low-opacity prune threshold, `prune_opa` | 0.005 |
| Large-scale prune threshold, `prune_scale3d` | 0.1 x scene scale |
| Refinement start / stop | 500 / 6000 |
| Refinement frequency | Every 100 steps |
| Opacity reset frequency | Every 3000 steps |
| Pause after reset | 0 |
| Absolute gradients / revised opacity | Both disabled |
| Screen-size refinement stop | 0; screen-size growth/pruning is disabled |

The strict implementation conditions produce **54 actual refinement events**:
step indices 600, 700, ..., 5900. Structural updates stop at index 6000; the final
1,000 updates optimize a fixed set of Gaussians. The actual reset during these
runs is at index 3000. The callback returns before the would-be reset at 6000.

The complete saved `cfg.yml` in each run is the authoritative parameter record.
Sampler settings and initial metrics match, but this does not mean the exact
camera order or all stochastic operations remain identical after growth diverges.

## 3. Environment and code provenance

Environment inspected when writing this record:

| Component | Observed value |
|---|---|
| OS/workflow | Windows, PowerShell, local `.venv` |
| GPU | NVIDIA GeForce GTX 1660 SUPER, 6144 MiB |
| Python | 3.12.10 |
| gsplat package | 1.6.0; local source changes described below |
| PyTorch | 2.9.1+cu128 |
| torchvision | 0.24.1+cu128 |
| NumPy | 2.5.2 |
| CUDA toolkit | 12.8.93, `.tools/cuda-12.8` |
| MSVC toolset directory | 14.38.33130 |
| Windows SDK | 10.0.22621.0 |
| NVIDIA driver at documentation time | 591.86 |
| Launcher build settings | `TORCH_CUDA_ARCH_LIST=7.5`, `MAX_JOBS=2`, `NUM_CHANNELS=3,4` |

These are current inspected versions, not a retrospectively captured package lock
or historical driver record for every run. The existing launchers import the
portable toolchain environment and reuse `.cache/torch_extensions`.

Repository HEAD at documentation time:

```text
c7f264cc260a5afad7b9f554217e96aed453abbb
```

The budget implementation is in working-tree changes/new files, **not in this
commit alone**. Retain those files with the experiment artifacts. There is no
per-run Git commit or start-of-run source archive for these experiments.

### Files added or modified for the budget experiments

| File | Change and purpose |
|---|---|
| [gsplat/strategy/budget.py](../gsplat/strategy/budget.py) | Added `BudgetStrategy`: count schedule, contributor/error/edge statistics, normalized candidate ranking and allocation history |
| [gsplat/strategy/__init__.py](../gsplat/strategy/__init__.py) | Exported the new strategy |
| [examples/simple_trainer.py](../examples/simple_trainer.py) | Added `budget` CLI configuration; collect detached image statistics before backward; restrict supported rendering modes; save budget/history diagnostics |
| [scripts/test-small-dataset.ps1](../scripts/test-small-dataset.ps1) | Added strategy, Gaussian-budget, error-weight and edge-weight arguments |
| [scripts/test-budget-dataset.ps1](../scripts/test-budget-dataset.ps1) | Added a launcher matching the room baseline settings, with protection against an existing result directory |
| [scripts/report-budget-experiment.py](../scripts/report-budget-experiment.py) | Added metric/resource comparison, configuration/artifact checks and optional checkpoint-reload comparison |
| [tests/test_budget_strategy.py](../tests/test_budget_strategy.py) | Added seven focused budget/attribution/state tests |
| [docs/BUDGETED_DENSIFICATION.md](BUDGETED_DENSIFICATION.md) | Added method description, commands and measured results |

There were **no new CUDA/C++ changes for these allocation experiments**. The new
strategy reuses the existing contributor query and existing duplicate/split/remove
operations. The default strategy remains selectable. This log is a subsequent
documentation addition, not another model-training change.

## 4. B00: saved high-resolution baseline

**Method:** `DefaultStrategy` averages positional gradient magnitudes across
training observations. Eligible small Gaussians are cloned and eligible large
Gaussians are split. There is no fixed count budget or top-K score allocation.
Opacity and scale rules prune Gaussians, and opacity resets follow the shared
schedule above.

**Role:** establish the reference quality, count, resource usage and artifacts.
This is the 7,000-step high-resolution room model, not the earlier 600-step
low-resolution room model. Results obtained at different evaluation resolutions
should not be substituted into this comparison.

**Equivalent launcher command for the saved settings:**

```powershell
Set-Location D:\gsplat\gsplat
.\scripts\test-large-dataset.ps1 -Steps 7000 -DataDir data/room_factor2 `
  -ResultDir results/room_hq_gtx1660
```

That output directory already contains the historical baseline. For a new
reproduction use a fresh name, such as `results/room_hq_gtx1660_repro01`; do not
overwrite the evidence above. The command is a reconstruction of the saved
configuration, not a claim that a shell transcript of the original invocation
was preserved.

**Result:** 841,017 Gaussians; PSNR 30.033510; SSIM 0.902064; LPIPS 0.211654;
trainer timer 400.940871 s; peak tensor allocation 1.279768 GiB.

**Evidence root:** [results/room_hq_gtx1660](../results/room_hq_gtx1660/).
The existing large-dataset report records 13 passed checks. Historical checkpoint
reload scored PSNR 30.035467, SSIM 0.902039, LPIPS 0.211727, at the same count.

## 5. Shared implementation for A01/A02/A03

### 5.1 Limit net additions with a count schedule

Let `S=112627`, `B=700000`, and `u` be completed eligible refinement events divided
by 54. The scheduled count is:

```text
C(u) = floor(S + (B - S) * (2*u - u*u)), capped at B
K = min(max(0, C(u) - current_count), eligible_parent_count)
```

This reaches the budget at the last eligible event, with smaller scheduled
increments toward the end. Both a clone and replacing one parent with two
children add one net Gaussian. If too few parents qualify, no extra parents are
forced into eligibility. Subsequent pruning can leave the count below the target.

Eligible parents must pass the existing gradient threshold and have opacity at
least 0.005. The opacity condition avoids spending slots on parents already due
for low-opacity pruning. This filter, the schedule and ranking are shared by all
three budget variants, so A01 versus B00 is not an isolated test of the schedule
alone.

### 5.2 Collect evidence from existing training renders

A02/A03 query the **four strongest contributors per pixel by `T * alpha`** every
ten training steps during refinement. These are detached statistics; there are
no additional optimizer updates or extra camera renders for scoring. The
contributor query itself is an extra rasterization traversal with a runtime and
memory cost.

In packed mode, query IDs identify projected entries. They are mapped through
`info["gaussian_ids"]` before global Gaussian accumulation. Invalid/padded IDs,
zero weights and masked pixels are excluded.

```text
error(pixel) = mean_RGB(abs(rendered_RGB - reference_RGB))
error_mass(g) = sum_over_sampled_views_and_retained_pixels(T * alpha * error)
observations(g) = number of sampled contributing observations
E(g) = error_mass(g) / max(observations(g), 1)
support(g) = sqrt(min(observations(g) / 3, 1))
```

Contribution mass is averaged by contributing observations, not by covered pixel
count. Repeated observations are not explicitly deduplicated by camera identity.
This statistic is a heuristic for useful growth, not a measured causal estimate
of the loss decrease that a particular clone/split will produce.

### 5.3 Normalize and rank

For each signal `x`, invalid values become zero; values are nonnegative. Divide by
the median of positive values among eligible candidates, then clip to 20. An
empty positive set gives a zero signal. Call this `N(x)`.

```text
score(g) = gradient_weight * N(average_gradient(g))
         + error_weight * support(g) * N(E(g))
         + edge_weight * support(g) * N(D(g))
```

Select at most `K` eligible parents using `torch.topk`. This implementation uses
ranking rather than the authors' weighted-sampling selection. Clone/split
geometry, optimizer-state management, scene bookkeeping, pruning and opacity
resets use existing operations. Error/edge/support evidence is reset at each
growth event so new children do not retain their parent's collected evidence.

Only single-GPU classic 3DGS is supported by the current trainer integration. The
cap bounds the live Gaussian count, not temporary memory while tensors are
replaced. Final quality still depends on the placement and subsequent optimization
of the selected children.

## 6. A01: budget plus gradient ranking

**Change from B00:** use the shared count schedule and eligible-parent filter;
select the strongest normalized positional-gradient candidates when growth slots
are limited. Error and edge collection are skipped.

**Purpose:** control for budgeted growth before evaluating extra image statistics.

| Parameter | Value |
|---|---:|
| `cap_max` | 700000 |
| `gradient_weight` | 1.0 |
| `error_weight` | 0.0 |
| `edge_weight` | 0.0 |
| `score_every` / `top_contributors` | 10 / 4, inactive because both image weights are zero |

**Training command used:**

```powershell
.\scripts\test-budget-dataset.ps1 -ErrorWeight 0 `
  -ResultDir results/budget_allocation_experiment/gradient700k `
  *> results\budget_allocation_experiment\gradient700k.log
```

The default launcher supplies 7,000 steps and the 700k budget. The output directory
now exists, so a reproduction must use a fresh directory and log filename.

**Result:** 699,173 Gaussians; PSNR 29.782270; SSIM 0.898554; LPIPS 0.215548;
385.392887 s; 1.100712 GiB peak tensor allocation.

**Interpretation:** count fell 16.8658%, but all three quality metrics worsened
against B00. PSNR loss was 0.251240 dB. The trainer timer was 3.88% shorter in this
comparison with the historical baseline; repeatable speedup has not been established.

**Evidence:** [run directory](../results/budget_allocation_experiment/gradient700k/)
and [training log](../results/budget_allocation_experiment/gradient700k.log).

## 7. A02: add contribution-weighted reconstruction error

**Change from A01:** set `error_weight=1.0`; use the attribution and observation
support described in Section 5. Edge weight remains zero. The gradient term
remains active, rather than replacing gradient eligibility with error alone.

**Purpose:** prefer eligible parents involved in currently unresolved image regions.

**Training command used:**

```powershell
.\scripts\test-budget-dataset.ps1 `
  -ResultDir results/budget_allocation_experiment/error700k `
  *> results\budget_allocation_experiment\error700k.log
```

Defaults here mean `gradient_weight=1`, `error_weight=1`, `edge_weight=0`, cap 700k,
score interval 10 and four retained contributors.

**Result:** 699,266 Gaussians; PSNR 29.863037; SSIM 0.901295; LPIPS 0.212985;
405.747246 s; 1.103704 GiB peak tensor allocation.

**Interpretation:** compared with A01, PSNR improved 0.080767 dB, SSIM improved
0.002742 and LPIPS decreased 0.002563. The trainer timer increased 20.354359 s.
Compared with B00, the count fell 16.8547%, while PSNR remained 0.170473 dB lower.
This was the best SSIM and LPIPS of the three budget variants, but not their best
PSNR. It did not exactly preserve baseline quality.

**Evidence:** [run directory](../results/budget_allocation_experiment/error700k/)
and [training log](../results/budget_allocation_experiment/error700k.log).

## 8. A03: add edge detail

**Change from A02:** set `edge_weight=0.25`; gradient and error weights remain 1.0.

The edge map is computed from mean-RGB ground-truth grayscale with replicate
padding and the absolute response of this Laplacian kernel:

```text
 0  1  0
 1 -4  1
 0  1  0
```

Its per-Gaussian edge mass uses the same retained blending contributions and is
averaged by contributing observations to obtain `D(g)` in the shared score.
This is an additional selection signal, not a change to the reconstruction loss.

**Purpose:** allocate some capacity to detail and edges while retaining gradient
and reconstruction-error guidance.

**Training command used:**

```powershell
.\scripts\test-budget-dataset.ps1 -ErrorWeight 1 -EdgeWeight 0.25 `
  -ResultDir results/budget_allocation_experiment/edges700k `
  *> results\budget_allocation_experiment\edges700k.log
```

**Result:** 699,207 Gaussians; PSNR 30.016626; SSIM 0.901174; LPIPS 0.214051;
409.676996 s; 1.116140 GiB peak tensor allocation.

**Interpretation:** PSNR improved 0.153589 dB over A02, but SSIM decreased 0.000121
and LPIPS increased 0.001066. Compared with B00: 16.8617% fewer Gaussians; PSNR
0.016884 dB lower; SSIM 0.000890 lower; LPIPS 0.002398 higher. Peak tensor allocation
was 12.79% lower and the trainer timer was 2.18% longer. This is the variant closest
to the baseline in PSNR; it is not an exact match across all quality metrics.

**Evidence:** [run directory](../results/budget_allocation_experiment/edges700k/)
and [training log](../results/budget_allocation_experiment/edges700k.log).

### Separate saved-checkpoint reload

The saved checkpoint was loaded into the original trainer's evaluation-only path:

```powershell
.\scripts\test-large-dataset.ps1 `
  -Checkpoint results/budget_allocation_experiment/edges700k/ckpts/ckpt_6999_rank0.pt `
  -ResultDir results/budget_allocation_experiment/edges700k_reload `
  *> results\budget_allocation_experiment\edges700k_reload.log
```

This did **not** add optimizer updates. Reload scored **30.019184 PSNR,
0.901129 SSIM, 0.214124 LPIPS**, with the same **699,207** Gaussians.

The trainer saves the checkpoint before the final optimizer update, while final
training validation runs afterward. The small difference is consistent with this
ordering. The reporter accepts absolute reload differences up to 0.02 dB PSNR,
0.001 SSIM and 0.001 LPIPS, with identical counts; A03 passed. Exact deltas and
tolerances are retained in `comparison.json` under `runs.edges.checkpoint_reload`.

Evidence: [reload metrics](../results/budget_allocation_experiment/edges700k_reload/stats/val_step6999.json)
and [reload log](../results/budget_allocation_experiment/edges700k_reload.log).

## 9. Consolidated results

### Quality, count and resources

| ID | Gaussians | PSNR (dB), up | SSIM, up | LPIPS/AlexNet, down | Trainer seconds | Peak tensor GiB | Checkpoint bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| B00 | 841,017 | 30.033510 | 0.902064 | 0.211654 | 400.940871 | 1.279768 | 198,482,997 |
| A01 | 699,173 | 29.782270 | 0.898554 | 0.215548 | 385.392887 | 1.100712 | 165,007,989 |
| A02 | 699,266 | 29.863037 | 0.901295 | 0.212985 | 405.747246 | 1.103704 | 165,029,813 |
| A03 | 699,207 | 30.016626 | 0.901174 | 0.214051 | 409.676996 | 1.116140 | 165,015,861 |

Displayed numbers are rounded; the source JSON retains the full stored precision.
These quality numbers are final in-memory training validation. The A03 reload
numbers are recorded separately and should not be mixed silently into this table.

### Differences from B00

| ID | Gaussian reduction | PSNR delta | SSIM delta | LPIPS delta | Trainer time change |
|---|---:|---:|---:|---:|---:|
| A01 | 16.8658% | -0.251240 | -0.003511 | +0.003895 | -3.88% |
| A02 | 16.8547% | -0.170473 | -0.000769 | +0.001332 | +1.20% |
| A03 | 16.8617% | -0.016884 | -0.000890 | +0.002398 | +2.18% |

### What the allocation history shows

Each budget run records 54 refinement events. Counts before pruning reached the
700,000 live-Gaussian cap; final counts were below it because pruning followed
growth. Growth was constrained by available slots at 29 events in A01, 28 in A02,
and 29 in A03. The scores therefore affected actual allocation.

The opacity reset at 3000 was followed by substantial pruning at 3100:

| ID | Pruned at 3100 | Live count after that event |
|---|---:|---:|
| A01 | 108,173 | 429,395 |
| A02 | 104,337 | 433,337 |
| A03 | 102,916 | 434,547 |

This explains the dip and subsequent regrowth in the count history. These runs
retain the original pruning/reset behavior; they are not strictly monotonic
constructive growth throughout training.

## 10. Artifact map: where to trace each number

| ID | Saved run root |
|---|---|
| B00 | `results/room_hq_gtx1660` |
| A01 | `results/budget_allocation_experiment/gradient700k` |
| A02 | `results/budget_allocation_experiment/error700k` |
| A03 | `results/budget_allocation_experiment/edges700k` |

For each root `R`:

| Relative file | What it establishes |
|---|---|
| `R/cfg.yml` | Full saved parameters and chosen strategy |
| `R/stats/val_step0000.json` | Initial quality/count; matching initialization check |
| `R/stats/val_step6999.json` | Final PSNR, SSIM, LPIPS, count and evaluation seconds/image |
| `R/stats/train_step6999_rank0.json` | Trainer timer and peak tensor allocation; budget runs also contain cap and growth history |
| `R/ckpts/ckpt_6999_rank0.pt` | Saved model before the last optimizer update |
| `R/ply/point_cloud_6999.ply` | Exported Gaussian scene |
| `R/renders/val_step6999_*.png` | Held-out visual comparisons; reference left, render right |
| `R/tb/` | TensorBoard training records |

In a budget run's `growth_history`, every entry contains `step`, `before`,
`target`, `eligible`, `added`, `pruned` and `after`. Accounting is
`after = before + added - pruned`. B00 did not collect this new structured history.

The aggregate report checks finite checkpoint tensors, shapes, step/count
consistency, PLY counts, shared settings and initial metrics. Budget histories are
checked against the schedule and cap. **All aggregate checks passed**, including
the A03 reload check. The baseline's earlier 13-check verification has its own
report and covers additional dataset/render/PLY checks.

## 11. Reproduce, regenerate reports and inspect the models

All commands run from `D:\gsplat\gsplat` and assume the existing environment/data.
To prepare the same dataset if absent:

```powershell
.\.venv\Scripts\python.exe scripts/download-room-small.py --image-factor 2
```

Training commands in Sections 4/6/7/8 describe the recorded configurations. Use
fresh result directories and fresh log filenames when retraining. The budget
launcher rejects existing output directories; the baseline launcher does not
provide that protection. To capture logs, create the parent directory first:

```powershell
New-Item -ItemType Directory -Path results/budget_repro01 -Force | Out-Null
.\scripts\test-budget-dataset.ps1 -Steps 7000 -GaussianBudget 700000 `
  -ErrorWeight 1 -EdgeWeight 0.25 -ResultDir results/budget_repro01/edges `
  *> results\budget_repro01\edges.log
```

Regenerate the comparison from the **existing completed runs** without training:

```powershell
.\.venv\Scripts\python.exe scripts/report-budget-experiment.py `
  --baseline results/room_hq_gtx1660 `
  --run gradient=results/budget_allocation_experiment/gradient700k `
  --run error=results/budget_allocation_experiment/error700k `
  --run edges=results/budget_allocation_experiment/edges700k `
  --reload edges=results/budget_allocation_experiment/edges700k_reload `
  --output results/budget_allocation_experiment/comparison.json
```

This updates the aggregate JSON and companion Markdown table; it does not modify
the individual training runs. To view A03 interactively:

```powershell
.\scripts\view-room.ps1 -DataDir data/room_factor2 `
  -Checkpoint results/budget_allocation_experiment/edges700k/ckpts/ckpt_6999_rank0.pt
```

The existing viewer defaults to port 8080. Use `-Port` if another instance occupies
that port. To evaluate a checkpoint without training, use the reload command in
Section 8 with a fresh output directory. `--ckpt` is evaluation-only; it is not a
resume/fine-tuning command.

### Tests already performed during implementation

```powershell
$env:NUM_CHANNELS = '3,4'
. .\.venv\Scripts\Activate-CUDA.ps1
python -m pytest tests/test_budget_strategy.py tests/test_strategy.py -q
```

Recorded outcome: **10 passed**, with one non-failing pytest configuration warning
about the unknown `env` option. Focused coverage includes schedule boundaries and
pauses, attributed error/padding/view counts, useful-parent selection, optimizer
state consistency, invalid settings, net split accounting, packed/dense CUDA
attribution agreement and masking. Existing strategy tests also passed.

The original `default --help` CLI and `git diff --check` also passed. Those test
outcomes are recorded from the implementation session; writing this log did not
retrain models or rerun the test suite. The earlier console output is not a
separately archived test log, so the command above is provided for a fresh check.

## 12. Measurement definitions and limits

- **Timer:** the stored training `ellipse_time` is elapsed wall time from the
  training loop's timer through the statistics save before the last optimizer
  update. It includes initial validation. For these runs it excludes dataset
  preparation/loading/cache and worker startup before that timer, final export,
  the final optimizer update and final validation. It is not total process time
  or an isolated GPU-kernel benchmark.
- **Memory:** `mem` is peak PyTorch tensor allocation in GiB as recorded before
  final evaluation. Driver/context, desktop and other applications are excluded.
  This is not a complete VRAM-use measurement.
- **Rendering speed:** `val_step6999.json` also contains an average evaluation
  time per image. No dedicated rendering-throughput benchmark was performed, so
  the count reduction alone does not establish a measured rendering speedup.
- **Sampling:** one run per variant, fixed starting seed 42. Split operations
  consume random numbers and can alter later sampler order; CUDA accumulation
  can also be nondeterministic. There are no confidence intervals or seed sweeps.
- **Baseline:** a historical saved run, not a fresh timed control. Shared saved
  settings and initial metrics were checked, but timing differences include
  possible variation in machine load and run conditions.
- **Quality:** A03 closely matches PSNR, while SSIM and LPIPS remain worse than
  B00. A02 is preferable among these budget runs if the priority is their measured
  SSIM/LPIPS. Exact equality or a statistically established quality-preserving
  method was not demonstrated.
- **Scoring approximation:** only the strongest four contributors are retained;
  weak contributors are omitted. This implementation does not collect the full
  Taming 3DGS score or perform its full optimization pipeline.
- **Scope:** one room scene at one resolution on one GPU. Other datasets,
  stronger count reductions, additional budgets, new split geometry, relocation,
  high-opacity Gaussians and CUDA scoring optimizations were not benchmarked here.

## 13. Wording supported by the evidence

Suggested report statement:

> On the Mip-NeRF 360 room scene at 1557 x 1038, the implemented budgeted
> gradient/error/edge allocation strategy produced 699,207 Gaussians after 7,000
> updates, compared with 841,017 in the saved default-strategy baseline. Gaussian
> count fell by 16.86%. PSNR was 30.0166 versus 30.0335 dB, while SSIM and LPIPS
> were slightly worse. Peak PyTorch tensor allocation decreased by 12.79%, and the
> recorded trainer timer increased by 2.18%. These are single-run observations.

For the ablation discussion: error attribution improved all three metrics over
the gradient-only budget control in these runs. Adding the edge term improved
PSNR further but slightly worsened SSIM/LPIPS relative to error-only selection.
Do not describe these results as identical quality, universal speedup, or a full
reproduction of the published method.

The research inspiration is [Taming 3DGS](https://humansensinglab.github.io/taming-3dgs/).
Its code is linked from that project page. Our implementation combines a related
budget-growth idea with reused training views, a simplified normalized score and
top-K ranking. Publication performance numbers are not the results of this local
experiment. Further method detail is in [BUDGETED_DENSIFICATION.md](BUDGETED_DENSIFICATION.md).

## 14. Source/data fingerprints at documentation time

These SHA-256 values identify the current files used for reproduction, not
archived hashes captured at the start of each historical run. Minor readability
cleanup occurred during the implementation session, so do not claim byte-identical
per-run source snapshots. A future source change should receive a new record.

```text
examples/simple_trainer.py
b613002c09c09ca7470066466a79b613c4cfdfab24ffe21c41dd053428d84334
gsplat/strategy/__init__.py
6e0c6411ce60da7fe526fff43cef12bb35dba1184cfa1a12cc0fb834c3fc6f96
gsplat/strategy/budget.py
585cbccf2579b33d7ac68543a7ebe54365fa8e738ea131535b95806c39378ce8
scripts/test-small-dataset.ps1
5962f845ac34e994c6fcbc540d80a6bb77e6709a685f5759f9ac63d1644fd7d0
scripts/test-budget-dataset.ps1
4633dd122b1738e7da01d35b0b49c390599314422aa8bda2c1fbe2385c0db9dc
scripts/report-budget-experiment.py
dab522443d88353577faf1d2a744ce9a603657e8271dae80bb687c3f7f780def
tests/test_budget_strategy.py
7724f768866cd1750dfe65688b7619d344b3a12c61daccbd0d999ed56803af4f
docs/BUDGETED_DENSIFICATION.md
cfc17d15f286aae888a8c981888be2d732278b878dd091dc48daab275a38fb21
scripts/test-large-dataset.ps1
27dc85411e8d76a3c770b5839129845a5338e205e3d260849a85f679ab2459a2
data/room_factor2/dataset_manifest.json
73c20c28c7e96ae5b87bcf7f5baf60ee573cb4dccd02a1bb88eca40ae24e1709
```

Check any current fingerprint with `Get-FileHash -Algorithm SHA256 -LiteralPath
<path>`. The manifest fingerprint identifies the preparation metadata, not every
image/point byte. `data/` and most `results/` are Git-ignored in this checkout;
retain them separately when preserving the experiment evidence. The code commit
alone does not preserve the datasets or trained models.

## 15. Template for the next advance

Copy this section and fill it in for each completed experiment. Keep unrun ideas
labelled as planned, and give every run a new output directory.

```text
Experiment ID / title:
Date and status (planned, running, completed, failed):
Question / hypothesis:
Reference experiment:
Exact code change and affected files:
Git commit plus working-tree patch/source snapshot:
Environment and dataset fingerprint:
Changed parameters:
Shared parameters and evaluation protocol:
Initialization (fresh SfM, warm start, resume, evaluation-only):
Exact command, working directory and output/log paths:
Actual optimizer updates / additional scoring or recovery work:
Result: Gaussian count, PSNR, SSIM, LPIPS, timer, memory, file size:
Differences from reference and interpretation:
Tests/checks and their saved evidence:
Checkpoint / PLY / config / raw statistics / renders:
Checkpoint-reload result if performed:
Limitations, failures and next decision:
Report wording supported by this experiment:
```

## 16. A04: appearance-preserving Long-Axis Split

**Date:** 2026-10-09, Asia/Saigon. **Status:** completed; training, tests, smoke,
immediate-disturbance diagnostic and separate checkpoint reload verified.

**Question:** can more structured densification improve the reconstruction at
A03's 700k budget and 7,000 updates? Reference: A03 (`edges700k`), with B00 as
the original quality/count reference. A04 starts from fresh SfM initialization.
Dataset, held-out split, resolution, optimizer and scoring settings match A03.

### Reference and geometry

Source: Deng et al., [Improving Densification in 3D Gaussian Splatting for
High-Fidelity Rendering](https://arxiv.org/html/2508.12313v1), Long-Axis Split,
Equation 9; checked against the authors' [implementation](https://github.com/XiaoBin2001/Improved-GS/blob/main/scene/gaussian_model_densification.py).

For parent scales `s=exp(log_scales)`, let `a=argmax(s)`, `R` be its quaternion
rotation, and `f=0.45`. The new centers are
`mu +/- R[:,a] * (3*f*s[a])`: the reference code uses the **3-sigma extent**,
so displacement is **1.35*s[a]**, not 0.45*s[a]. Child scale on the selected
axis is `(1-f)*s[a]=0.55*s[a]`; each other axis is multiplied by
`sqrt(1-f*f)=0.893028554975`. Child activated opacity is `0.6*sigmoid(logit)`.
Color/SH and rotation are inherited. Both children share the adjusted shape.
This approximates appearance; alpha blending is not exactly preserved.

Like the authors' method, this mode splits **all selected parents**, including
small parents previously cloned. Thus A03 vs A04 tests the densification
operator and the removal of cloning together. Candidate eligibility, top-K
ranking, edge/error weights and the count schedule remain the local A03 design.
This is a method adaptation, **not a full Improved-GS reproduction**: absolute
gradients, weighted sampling, recovery-aware pruning, multi-view updates and the
paper's growth curve are not added here. Paper-wide gains cannot be attributed
to this split operator alone.

### Code changes and compatibility

| File | Change |
|---|---|
| `gsplat/strategy/ops.py` | Optional `mode="long_axis"`; structured geometry and opacity; reuse optimizer/state/scene updates |
| `gsplat/strategy/budget.py` | `split_mode`, `split_distance`, `opacity_reduction`; LAS replaces cloning; record duplicate/split counts |
| `scripts/test-small-dataset.ps1` | `-SplitMode random/long_axis` forwarded for budget strategy |
| `scripts/test-budget-dataset.ps1` | Forward `-SplitMode` for identical room launch settings |
| `tests/test_long_axis_split.py` | Rotated axes, endpoint geometry, opacity, RNG, optimizer continuity, scene hooks, empty selection, validation, CUDA render/backward |
| `scripts/measure-split-disturbance.py` | Frozen-checkpoint paired split diagnostic on identical parents and four held-out views, without optimizer updates |
| `docs/BUDGETED_DENSIFICATION.md` | Usage and measured A04 findings |
| `docs/EXPERIMENT_LOG.md` | This experiment record |

Existing defaults retain random split plus cloning. `revised_opacity=True`
cannot be combined with LAS, which specifies its own opacity multiplier.
No CUDA/C++ changes are required. Parent optimizer moments are removed;
children start with zero moments and other parents retain theirs. The parent
is replaced by two children, adding one net Gaussian per selected parent.
Tied longest axes choose the first maximum. Structural ordering follows
the existing `rest, positive_children, negative_children` tensor layout.

LAS placement does not consume RNG. Consequently the shared seed does not
guarantee identical subsequent shuffled camera orders versus A03, whose
random split consumes RNG. Single-run final metrics cannot isolate recovery
updates or establish repeatable speedup.

### Training command and evidence

```powershell
Set-Location D:\gsplat\gsplat
.\scripts\test-budget-dataset.ps1 -ErrorWeight 1 -EdgeWeight 0.25 `
  -SplitMode long_axis `
  -ResultDir results/budget_allocation_experiment/longaxis700k `
  *> results\budget_allocation_experiment\longaxis700k.log
```

Defaults provide 7,000 updates, cap 700k, `f=0.45`, opacity multiplier 0.6,
error/gradient weights 1, edge weight 0.25, score interval 10, top contributors 4.
Use a fresh output directory for reproduction. The source snapshot and hashes
will be stored under `results/budget_allocation_experiment/longaxis_evidence/`.
Focused validation output: `results/budget_allocation_experiment/long_axis_tests.log`.

### Validation completed before training

```powershell
. .\.venv\Scripts\Activate-CUDA.ps1
$env:NUM_CHANNELS='3,4'
python -m pytest tests/test_long_axis_split.py tests/test_budget_strategy.py tests/test_strategy.py -q
.\scripts\test-small-dataset.ps1 -Strategy budget -SplitMode long_axis `
  -GaussianBudget 10000 -EdgeWeight 0.25 `
  -ResultDir results/budget_allocation_experiment/longaxis_smoke
```

**19 tests passed**, including CUDA tests. One existing pytest configuration
warning reports unknown option `env`; it does not fail validation. `pip check`
found no broken requirements. The synthetic 600-step smoke produced 524 Gaussians,
27.174973 PSNR, 0.959201 SSIM, 0.092084 LPIPS (fixture metrics, not room quality).
All eight artifact/budget checks passed. Evidence:
`longaxis_smoke.log`, `long_axis_tests.log`, and
`longaxis_evidence/smoke_checks.json` under `results/budget_allocation_experiment/`.

### Completed immediate-disturbance diagnostic

```powershell
. .\.venv\Scripts\Activate-CUDA.ps1
$env:NUM_CHANNELS='3,4'
python scripts/measure-split-disturbance.py `
  --checkpoint results/budget_allocation_experiment/edges700k/ckpts/ckpt_6999_rank0.pt `
  --output results/budget_allocation_experiment/longaxis_evidence/split_disturbance.json
```

This freezes A03's final checkpoint and selects up to 10,000 largest parents
among those with scale > `0.01*scene_scale` and opacity >= 0.005. Both split
operators receive **identical parent IDs**. The diagnostic compares RGB MSE
against the original render and the ground-truth PSNR change on four full-sized
held-out views (indices 0, 13, 26, 38). Random placement uses seeds 42, 43 and 44;
LAS uses its deterministic geometry. It measures split geometry alone: only
large parents qualify, so the removal of cloning is not part of this diagnostic.
It is not a replay of actual training splits, and performs no recovery updates.

| Operator | Random seed | Mean RGB MSE vs original render, lower is better | Mean PSNR change vs ground truth, dB |
|---|---:|---:|---:|
| Random split | 42 | 0.000484347 | -1.467235 |
| Random split | 43 | 0.000341089 | -1.180410 |
| Random split | 44 | 0.000442491 | -1.494085 |
| Random split average | Three placements | 0.000422642 | -1.380577 |
| Long-Axis Split | Deterministic | 0.000298755 | -1.060996 |

LAS produced **29.3126% less RGB disturbance MSE** than the random-placement
average and a **0.319581 dB smaller mean PSNR drop**. It also beat each of the
three random placements on these averaged measurements. Both operators still
disturbed the original reconstruction. This is direct evidence for smaller
immediate disturbance under this diagnostic, not a count of recovery updates.
Raw per-view values, the checkpoint SHA-256 and all 10,000 parent IDs are in
`longaxis_evidence/split_disturbance.json`; output is in `split_disturbance.log`.

### Final training result and comparisons

All rows below average the same **39 held-out room views at 1557 x 1038** after
7,000 updates. The original B00 and A03 rows are saved runs, not reruns during A04.

| ID | Gaussians | PSNR, up | SSIM, up | LPIPS, down | Trainer seconds | Peak tensor GiB |
|---|---:|---:|---:|---:|---:|---:|
| B00 | 841,017 | 30.033510 | 0.902064 | 0.211654 | 400.940871 | 1.279768 |
| A03 | 699,207 | 30.016626 | 0.901174 | 0.214051 | 409.676996 | 1.116140 |
| A04 | **597,989** | **30.505762** | **0.905610** | **0.204861** | **394.156893** | **1.061557** |

Compared with **A03**, A04 used **14.4761% fewer Gaussians**, improved PSNR by
**0.489136 dB** and SSIM by **0.004436**, and reduced LPIPS by **0.009190**.
The trainer timer was **3.7884% shorter** in this comparison.

Compared with **B00**, A04 used **28.8969% fewer Gaussians**, improved PSNR by
**0.472252 dB** and SSIM by **0.003546**, and reduced LPIPS by **0.006792**.
Peak tensor allocation decreased **17.0508%**; the timer was **1.6920% shorter**.
These are observed single-run timing differences, not a demonstrated repeatable
speedup. The timer has the exclusions described in Section 11.

The cap is **700,000**, not a forced target count. All **54** refinement events
ran; total selected/split parents were **597,632**, clones **0**, and Gaussian
prunes **112,270**. Initial 112,627 + net additions 597,632 - prunes 112,270 =
597,989. At the last growth event there were 593,651 Gaussians and 106,349 free
budget slots, but only 4,598 eligible parents. Splitting them and pruning 260
produced 597,989. Hence the final budget underfill is measured, not a launcher
error or a deliberately lowered cap. This is an equal-cap comparison, not an
equal-final-count comparison.

Checkpoint size: **141,128,565 bytes**. PyTorch peak allocation is separate from
sampled total GPU occupancy; the latter samples are retained in
`longaxis_evidence/gpu_samples.csv` and are not a continuous peak measurement.

### Artifact and configuration verification

The comparison reporter passed **all checks** against both B00 and A03:
finite/aligned checkpoint tensors, matching checkpoint/PLY/metric counts,
checkpoint step, budget bound, allocation accounting, matching shared training
and density thresholds, and matching initial model metrics. Additional checks
confirmed all PLY float properties finite, **39 full-resolution validation
renders**, matching A03 score weights/settings, every growth event using LAS
with no clones, correct refinement indices, and byte-identical saved source
snapshots. Seven additional checks passed in `additional_checks.json`.
The reproducing helper is retained as `longaxis_evidence/verify_artifacts.py`.

A preliminary auxiliary PLY check lacked `plyfile`; the completed check used
the existing NumPy binary-PLY approach without adding dependencies. This was
a validation-script import issue, not a training failure.

Evidence under `results/budget_allocation_experiment/`:

- `longaxis700k/cfg.yml`, `ckpts/ckpt_6999_rank0.pt`, `ply/point_cloud_6999.ply`.
- `longaxis700k/stats/val_step0000.json`, `val_step6999.json`, `train_step6999_rank0.json`.
- `longaxis700k/renders/val_step6999_*.png` and `longaxis700k.log`.
- `longaxis_evidence/comparison_baseline.json` and `.md` (B00/A03/A04).
- `longaxis_evidence/comparison_A03.json` and `.md` (direct A03/A04).
- `longaxis_evidence/additional_checks.json`, `environment.json`, `source_hashes.json`,
  `git_head.txt`, `source/`, `disturbance_source_hash.json`.

### Verified checkpoint reload

```powershell
.\scripts\test-large-dataset.ps1 `
  -Checkpoint results/budget_allocation_experiment/longaxis700k/ckpts/ckpt_6999_rank0.pt `
  -ResultDir results/budget_allocation_experiment/longaxis700k_reload `
  *> results\budget_allocation_experiment\longaxis700k_reload.log
```

This is evaluation-only and adds no optimizer updates. Reload scored
**30.509300 PSNR, 0.905576 SSIM, 0.204934 LPIPS**, at **597,989 Gaussians**.
Absolute differences from final training validation were 0.003538 dB,
0.000034 SSIM and 0.000072 LPIPS. All are within the reporter's unchanged
tolerances (0.02 dB, 0.001, 0.001); counts match. Checkpoint saving precedes the
last optimizer update, as described in Section 8. Both regenerated comparison
reports include the reload evidence and return `all_checks_pass=true`.

Final reporter commands (these replaced only A04's own comparison reports):

```powershell
.\.venv\Scripts\python.exe scripts/report-budget-experiment.py `
  --baseline results/room_hq_gtx1660 `
  --run A03=results/budget_allocation_experiment/edges700k `
  --run A04=results/budget_allocation_experiment/longaxis700k `
  --reload A04=results/budget_allocation_experiment/longaxis700k_reload `
  --output results/budget_allocation_experiment/longaxis_evidence/comparison_baseline.json
.\.venv\Scripts\python.exe scripts/report-budget-experiment.py `
  --baseline results/budget_allocation_experiment/edges700k `
  --run A04=results/budget_allocation_experiment/longaxis700k `
  --reload A04=results/budget_allocation_experiment/longaxis700k_reload `
  --output results/budget_allocation_experiment/longaxis_evidence/comparison_A03.json
```

Highest sampled total GPU occupancy during the training checks was **1,948 MiB**
out of 6,144 MiB. This is a sampled observation, not the true peak. The result's
checkpoint/PLY/renders have been retained with the logs and source snapshots.

### Supported interpretation and remaining limits

This adaptation improved all three held-out quality metrics with fewer Gaussians
in one room run. The frozen-checkpoint diagnostic independently showed less
immediate rendering disturbance than random splitting. Together these support
using LAS as a promising densification operator in this local pipeline.

The experiment **does not establish fewer recovery updates**: no recovery
trajectory, time-to-quality target or split-to-recovery interval was measured.
There is one training seed and one scene, and candidate/camera trajectories can
diverge. The main experiment also replaces cloning, so it does not isolate
long-axis placement from scale/opacity changes or the clone policy. No parameter
tuning, repeat training, extra recovery training or paper-wide reproduction
was performed. Existing defaults remain available for controlled comparisons.

## 17. A04 on three additional scenes: Counter, Kitchen and Bonsai

**Date:** 2026-10-22, Asia/Saigon. **Status:** six fresh 7,000-update training
runs completed; three A04 checkpoint reloads and all comparison/artifact checks
passed. See [full protocol, results and reproduction](MULTISCENE_A04.md).

Counter, Kitchen and Bonsai were chosen before observing their results. Each
baseline/A04 pair used official half-resolution images, the same per-scene SfM
initialization and every-eighth-image held-out split, and the existing Room
optimizer/loss/densification schedule. A04 retained the 700k cap, weights 1/1/0.25
and LAS settings 0.45/0.6 without tuning. No extra training or recovery updates
were added. Runs used the existing local GTX 1660 SUPER sequentially.

| Scene | B00 Gaussians | A04 Gaussians | Reduction | B00 PSNR | A04 PSNR | B00 SSIM | A04 SSIM | B00 LPIPS | A04 LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Counter | 749,489 | 569,520 | 24.01% | 27.5592 | 27.8234 | 0.88694 | 0.89029 | 0.19605 | 0.18897 |
| Kitchen | 1,066,757 | 699,357 | 34.44% | 29.7438 | 30.0037 | 0.91504 | 0.91546 | 0.11620 | 0.11467 |
| Bonsai | 1,017,191 | 695,971 | 31.58% | 30.2772 | 30.4295 | 0.92753 | 0.92846 | 0.15912 | 0.15697 |

Final in-memory held-out metrics are shown above; separate reload metrics and
their tolerances are in the per-scene JSON reports. On all three additional
scenes, A04 improved PSNR, SSIM and LPIPS while reducing Gaussian count. Across
the three scenes, unweighted mean count reduction was 30.01% and mean paired
PSNR gain was 0.2254 dB. Trainer time and peak tensor allocation also decreased
in each pair; time is a single-run observation with the existing timer scope.

Evidence root: `results/multiscene_a04_20261022`. It contains per-scene logs,
configs, checkpoint/PLY/renders, `comparison.json`, A04 reloads, combined
`summary.json`/`summary.md`, and `comparison.png`. The `evidence` folder contains
source snapshots/hashes, exact image-role/hash manifests, dataset preservation
checks, GPU information and package versions. All checked PLY/checkpoint values
are finite, counts agree, all 102 expected full-resolution held-out renders per
method are present, and campaign source hashes still match.

Scope remains one seed per method/scene and indoor scenes only. This extends the
earlier Room evidence; it does not isolate axis placement from opacity/scale and
clone-policy changes, establish uncertainty, or test CCD-GS pose probing. Repeat
seeds, outdoor evaluation and independent camera-sampling RNG remain priorities.

## 18. Repeated seeds and crop analysis, finalized at user request

**Date:** 27 October 2026. The user requested skipping all unfinished work. The campaign and monitor were stopped. Completed evidence: 21 full 7,000-update runs, 10 matched pairs, one completed unpaired Bonsai B00 seed43 run. The interrupted Bonsai A04 seed43 run is excluded.

These fresh runs use independent DataLoader sampling. Initialization and all 7,000 camera IDs match in each included pair. Historical results are preserved and excluded from these seed averages.

| Scene | Pairs | Mean PSNR gain | 95% paired t interval | Mean Gaussian reduction |
|---|---:|---:|---|---:|
| room | 3 | +0.5688 dB | [+0.3599, +0.7777] | 28.37% |
| counter | 3 | +0.1361 dB | [+0.0807, +0.1916] | 24.25% |
| kitchen | 3 | +0.1297 dB | [-0.1458, +0.4052] | 35.27% |
| bonsai | 1 | +0.3866 dB | Not estimated (one pair) | 32.74% |

Room and Counter have positive intervals in this small study. Kitchen's interval crosses zero and its SSIM changes are small and mixed. Bonsai remains one-pair evidence. Three-seed intervals are imprecise, conditional on this protocol and split; no population generalization claim follows.

Component ablations, multi-budget sweep, parameter tuning and the full CCD-GS mechanism matrix were **skipped**. Their implementations/plans and CCD smoke artifacts do not constitute scientific results. The split/budget/sampler checks reported 23 passed; the completed Room A04 seed42 checkpoint reload reproduced PSNR within 0.0000026 dB.

[Final report](../results/research_completion_20261027/REPORT.md), [full-precision statistics](../results/research_completion_20261027/statistics.json), [visual review](../results/research_completion_20261027/VISUAL_REVIEW.md), [status and skipped matrix](../results/research_completion_20261027/status.json). All raw completed-run artifacts, source snapshots and interrupted output are retained.
