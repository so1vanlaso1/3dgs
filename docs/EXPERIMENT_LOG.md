# 3DGS experiment log: baseline and budgeted Gaussian allocation

## Capture route replay, 2026-10-30

Problem: the result viewer lacked replay along the source recording route.
Solution: timestamp-based camera playback with play/pause, seeking, speed, loop
and an optional visible path, independently controlled per browser. The current
dataset supplies 5,201 poses over 346.8827193 seconds. Six targeted tests passed;
the viewer loaded 997,966 Gaussians and returned HTTP 200 on port 8086. Visual UI
verification remains incomplete because computer use stopped when Chrome's URL
could not be determined reliably. See [findings, implementation, usage and limits](CAPTURE_REPLAY.md).

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

## 2026-10-30: Room baseline image-count clarification

Question: How many photographs created the normal 3DGS Room scene? Verified both data/room_factor2/images and data/room_small/images contain 311 files. The documented standard split is 272 training photographs and 39 held-out evaluation photographs, consistent with examples/datasets/colmap.py (every eighth image held out). Thus 272 photographs train the scene; 311 is the complete dataset count. The two local versions use the same image count at different resolutions.


## 2026-10-30: Launch normal high-resolution Room viewer

User requested running the Room scene. Launched scripts/view-room.ps1 in background with data/room_factor2 and results/room_hq_gtx1660/ckpts/ckpt_6999_rank0.pt. Viewer process ID 5736, port 8080. Startup confirmed 841,017 Gaussians, 311 camera images, initial view DSCF4667.JPG, and HTTP 200 from http://localhost:8080. Opening the default browser was denied by the environment; the viewer URL is available to open manually. This loads the saved baseline checkpoint; no retraining was required. Logs: .cache/room-viewer.stdout.log and .cache/room-viewer.stderr.log.


## 2026-10-30: Stop other viewer instances

User requested stopping all other instances while keeping the normal Room viewer. Process inspection identified six other RealSense viewer instances listening on ports 8081-8086. Stopped their Python launcher/runtime processes: 12684, 14020, 14476, 16432, 5872, 2900, 12300, 25052, 23904, 21040, 24096, 9872. Kept Room processes 5736/1348 on port 8080. Verified only port 8080 remained listening in 8080-8099 and Room HTTP returned 200.


## 2026-10-30: Room versus RealSense dataset and quality audit

Problem: Room appears cleaner than the personal RealSense reconstruction. Compared Room with the latest courtyard 701-view and 6,996-view runs, and checked the earlier 15 FPS and compact-region pose trials. Verified resolution, splits, configs, matched metrics, incomplete depth, RGB SfM fragmentation, and actual full-rate view-use counts. Visually checked captured-versus-rendered courtyard and Room comparisons. Pose/scene consistency and optimization coverage are hypotheses requiring controlled tests; existing compact pose refinement did not clearly improve quality. Full findings, limitations, evidence and proposed diagnostic solution: docs/ROOM_VS_REALSENSE.md. No new training or reconstruction executed.

## 2026-10-30: Courtyard 120,000-step run

Problem: Test whether longer optimization improves the latest full-rate RealSense courtyard reconstruction. Started a fresh 120,000-update run using the existing 6,996-view prepared dataset and the original full-rate training settings, including seed 42, the same split, depth weight 0.01, fixed poses, and uncapped DefaultStrategy with refinement stopping at step 6,000. The original 20,000-step result is preserved. Runner: `scripts/run-courtyard-120k.ps1`; output and live status: `results/realsense_20261030_111354_all30_120k/`.

Interpretation: Changing max steps also changes the trainer's means learning-rate decay schedule, which is parameterized by max steps; this is an inherent consequence of the requested longer run. Completion, metrics, and comparison are pending. The previous 20,000-step training took about 2 hours 50 minutes; a sixfold estimate is roughly 17 hours, subject to machine load and training behavior. The runner evaluates the same heldout views and exports a preview after training.

## 2026-10-30: Courtyard 120k restart

Problem: The prior 120k run stopped with status failed at 22:33:09 and its training log ended around step 8,044; the cause is not established. No training process was active when checked. Solution: preserved the incomplete output in `results/realsense_20261030_111354_all30_120k_interrupted_20261030_223309/` and restarted `scripts/run-courtyard-120k.ps1` from initialization, with no checkpoint. Completion and quality remain unverified.


## 2026-10-30: Free memory for training

Problem: The user requested stopping background workloads to make RAM available for training. Initial inspection found no Python, COLMAP, viewer, or training process; Windows reported 9.25 GiB available of 15.95 GiB RAM and NVIDIA reported 442 MiB of 6,144 MiB GPU memory used. Stopped Windows Widgets and its WebView processes, Phone Link, Cross Device processes, VS Code Pylance language server, and GitHub Copilot runtime. Kept VS Code windows and terminals and Windows system/security services running. After cleanup, Windows reported 9.72 GiB available and GPU memory usage was 356 MiB. VS Code automatically restarted Pylance, so the gain is a snapshot and may change.

Finding: `results/realsense_20261030_111354_all30_120k/status.json` reports the restarted 120k run failed at 22:48:48 with pipeline exit code 1, before this cleanup. The training log ends near step 8,474 without a cause visible in its final lines. No training process was active during inspection. The failure cause remains unverified; cleanup does not establish that memory caused it. No training was started or stopped by this cleanup.

## 2026-10-30: Investigate reported training out-of-memory error

Question: Was the reported OOM caused by system RAM or GPU VRAM? Searched the restarted 120k run's `train.log` for CUDA OOM, MemoryError, allocation errors, and tracebacks; none were recorded. Its status shows exit code 1 and the log stops near step 8,474, so the available artifacts do not identify the memory type or establish that OOM caused this failure. No Windows System resource-exhaustion event 2004 was found in the previous three hours. After the run, Windows reported approximately 9.81 GiB available physical RAM and the GPU used 353 MiB of 6,144 MiB; these idle snapshots cannot establish peak usage during training. Next diagnostic: obtain the exact error and capture RAM/VRAM during a future run.

## 2026-10-30: Restart courtyard 120k training after memory cleanup

User requested running training again. Confirmed no existing trainer, preserved the failed 22:48:48 run at `results/realsense_20261030_111354_all30_120k_interrupted_20261030_224848/`, and started `scripts/run-courtyard-120k.ps1` fresh from initialization, runner PID 1148. The same dataset and 120,000-step settings are used; no checkpoint resume. Verified status `running` and progress beyond step 134. At startup GPU usage was 929 MiB / 6,144 MiB; this is not peak usage. Completion and the cause of the earlier failure remain unverified. Live status and log are in `results/realsense_20261030_111354_all30_120k/`.

## 2026-10-30: VS Code window out of memory

Problem: The user supplied a VS Code window crash reporting `oom`, code `-536870904`. VS Code's `main.log` independently records renderer OOMs at 22:48:25 and 23:04:12. This identifies the crashed VS Code renderer, but the logs examined do not establish what allocation triggered it, whether system RAM or a per-process limit was exhausted, or whether it caused either earlier training failure. The repository contains a 25,154,482,176-byte `.db3` file, but there is no evidence it was opened or caused the crash.

Current observation: the restarted 120k trainer and runner were still active after the second crash, with the training log around step 13,916. A later snapshot showed 8.11 GiB free of 15.95 GiB physical RAM and GPU use of 2,166 / 6,144 MiB; these are not crash-time peaks. Recovery suggested: reopen the window without restoring editors if it repeatedly crashes, preserving any needed unsaved work; avoid opening large recording files in the editor. If it recurs, isolate extensions and inspect renderer memory. These steps are recommendations, not a verified fix. No VS Code settings or training process were changed.

## 2026-10-30: Repeated VS Code renderer OOM diagnosis

Problem: Screenshot reports window termination with reason `oom`, code `-536870904`. Current inspection of `%APPDATA%\Code\logs\20261030T225152\main.log` confirms renderer OOMs at 23:04:12, 23:10:09, 23:13:58, and 23:16:06; the earlier session `20261030T223921` records another at 22:48:25. Extension hosts exit after the renderer failure in the first three latest entries, so those exits do not identify an extension as the cause.

Finding: The confirmed immediate cause is memory allocation failure in the VS Code window renderer. Exact trigger remains unproven: the available logs do not distinguish a renderer heap limit/leak from system memory pressure. Inspection snapshot showed about 3.59 GiB free physical RAM out of 15.95 GiB, and 12.27 GiB free virtual memory out of 29.45 GiB. No System resource-exhaustion event 2004 was returned for the previous day. These post-crash observations do not measure crash-time peaks or exclude system exhaustion. Two current Code processes each use about 1.24 GiB private memory; process roles were not mapped. Copilot database fallback and agent cancellation errors are present but no causal connection is established. The screenshot does not establish CUDA VRAM exhaustion or the cause of any trainer failure.

Recovery/next diagnostic: Reopen with Don't restore editors if repeated restoration triggers crashes, preserving any needed unsaved work. Use VS Code Process Explorer to identify renderer growth and isolate restored editors, chat/webview content, and extensions if needed. These remain diagnostic recommendations, not a verified repair. No application settings, extensions, or training processes were changed. Reference: https://github.com/microsoft/vscode/wiki/Performance-Issues

## 2026-10-30: Verify trainer continues independently of VS Code crashes

Question: How to know training can finish and produce results after the VS Code window crashes? Live inspection confirms Python trainer PID 14428 (venv launcher 4132) and runner PID 1148 remain active. Two log reads advanced from step 30,346 to 30,470 of 120,000, establishing continued progress, not merely a stale running status. Runner status remains train/running. Future completion cannot be guaranteed by current activity.

Output contract verified from scripts/run-courtyard-120k.ps1 and the live trainer command: checkpoint and PLY saves are configured at 120,000 steps only, with no intermediate checkpoint requested. Runner then performs finalization through scripts/run-realsense-all30.py, including comparison.json, preview/scene.ply, preview/capture_path.mp4, provenance.json, and viewer.json. It writes stage/status complete only after training and finalization exit successfully; failures write failed. Monitor advancing train.log while running, then confirm complete status and final artifacts. No trainer settings or processes changed.

## 2026-10-31: Open courtyard 120k scene

User requested running the 120k scene. Verified the 120,000-step training saved `train/ckpts/ckpt_119999_rank0.pt` and `final_post_update.pt` under `results/realsense_20261030_111354_all30_120k`. The overall runner status is failed because the subsequent finalization process lacked a CUDA toolkit/extension; this does not mean the completed training checkpoint is missing. Matched comparison and preview finalization remain incomplete.

Solution: launched the existing viewer with its MSVC/CUDA environment, loading `final_post_update.pt`, dataset `data/realsense_20261030_111354_all30/rgbd/dataset`, camera index 1000, port 8085, and output directory `results/realsense_20261030_111354_all30_120k/viewer`. Launcher PID 13468. Startup confirmed 705,300 Gaussians and 6,996 camera images; HTTP returned 200 from http://localhost:8085. Logs: `.cache/room-viewer.stdout.log` and `.cache/room-viewer.stderr.log`. No retraining performed. Browser rendering has not been visually inspected.

Reopen command: `powershell -ExecutionPolicy Bypass -File scripts/view-room.ps1 -Checkpoint results/realsense_20261030_111354_all30_120k/train/ckpts/final_post_update.pt -DataDir data/realsense_20261030_111354_all30/rgbd/dataset -CameraIndex 1000 -Port 8085 -OutputDir results/realsense_20261030_111354_all30_120k/viewer -Background`.

## 2026-10-31: People visible from one replay viewpoint but absent from another

User observation: a group of people appears along the recovered capture path but is absent when revisiting the same area later. Verified `examples/simple_viewer.py` loads fixed checkpoint splats and renders them with the current camera; `examples/capture_path.py` changes camera pose/FOV and replay labels, not Gaussian positions or existence. The replay timeline is a camera trajectory, not a dynamic scene reconstruction.

Interpretation (hypothesis, exact views not inspected): people present in only some training images violate the static-scene assumption. Optimization can retain incomplete/view-specific ghost geometry whose projected footprint, occlusion and direction-dependent SH color make people visible from some viewpoints but not others. Disappearance in the rendered image does not establish deletion of Gaussians. At exactly the same camera pose and render settings, a fixed checkpoint should reproduce the same image regardless of replay time. Diagnostic: compare the two exact viewpoints, then return to an identical pose; if needed inspect depth and SH degree zero to distinguish geometry/occlusion from directional color. No model edits or retraining performed. For a static clean reconstruction, a possible future controlled solution is masking transient people during training; actual motion reconstruction requires a dynamic representation. General reference: https://wild-gaussians.github.io/ .

## 2026-10-31: Reported frozen image during paused free exploration

User clarified desired behavior: pause recorded camera replay and freely navigate with a fresh render at each camera pose; reports image remains frozen when moving. This is distinct from fixed Gaussian geometry and should not be explained away as the static scene assumption. Verified nerfview/viewer.py registers client.camera.on_update to submit a move render using the current camera; CapturePlayback pause sets playing=False and tick then skips camera updates. Thus the intended code supports free exploration and no retraining is needed. Current viewer stderr contains no recorded error. A separate Render preview pause control also exists, so the exact control used is pending clarification. Live UI verification blocked: computer-use inventory reports native pipe and browser access host unavailable. No code or process changes made; actual cause remains unconfirmed.

### Follow-up: correct pause control confirmed

User confirms using Capture replay / Pause / explore. Inspected installed nerfview renderer, camera update subscription, Viser GUI callback behavior, and active viewer processes. Pause only stops path advancement; the render thread remains independent and responds to camera updates. The timeline assignment callbacks in this installed Viser execute synchronously, so the existing updating guard does not demonstrate an asynchronous feedback bug. Viewer processes 13468/13784 remain active; stderr has no reported exception. No root cause established and no speculative code modification applied. Need a visual observation of navigation or camera-pose/render telemetry to distinguish input/camera movement failure from stale rendering. Earlier static-geometry explanation is insufficient for the reported frozen image.

### 2026-10-31: Independent manual-camera refresh implemented and verified

A fresh browser reproduced correct manual rendering on the original server, so
the embedded-client freeze's exact cause remains unconfirmed. Added per-client
camera-state polling to GsplatViewer and the standalone main loop, independent
of CapturePlayback.tick: any changed pose, FOV or aspect queues a move render,
even without a camera callback. Unchanged cameras can settle to high quality.
Seven targeted tests and compilation passed. Initial tests encountered shared
temp-directory permissions; a unique workspace basetemp resolved the issue.
Restarted the same courtyard checkpoint on http://localhost:8085 (launcher
16076, serving child 16744); HTTP 200, empty stderr, 705,300 Gaussians loaded.
Visually verified play/pause/manual drag: timeline held at 5.719s while image and
rendered count changed (about 319,030 -> 327,194 -> 319,057 on reversal).
Full findings, limitations and screenshot reference: docs/CAPTURE_REPLAY.md.
The fallback does not repair blocked browser input or disconnected WebSockets;
the original VS Code tab needs a refresh after restart. No model edits/training.

## 2026-10-31: Exact 107.51s camera restored after 228.11s

Problem: user suspects people disappear even after returning to the same camera.
Added scripts/audit-viewer-pose-restore.py to exercise the live port 8085 server
with an independent client: timeline/camera A at 107.51s -> timeline/camera B at
228.11s -> exact camera A while timeline remains 228.11s. People reappear and the
JPEG bytes and decoded RGB pixels are identical (max/mean pixel difference 0).
Thus the suspected time-dependent rendering-state bug was not reproduced. Browser
input/display issues specific to the user's session remain unexcluded. Navigation
finding: drag uses an orbit camera about a target one normalized unit ahead, so
matching direction by dragging also moves camera position. Recorded A/B poses
differ by 0.668498 normalized units and 97.1324 degrees; B position with A direction
shows a pillar, not A's view. Do not claim this proves the user's manual coordinates
or the precise multi-view geometry of people. Initial diagnostic connection was
rejected for missing Viser version handshake; the matching subprotocol resolved it.
Evidence: results/viewer_pose_restore_20261031/report.json and four PNGs. Full
findings/reproduction/limitations: docs/VIEWER_POSE_RESTORE_20261031.md. No model,
viewer, render settings, or training changes; existing viewer left running.

## 2026-10-31: Trace actual people-region Gaussians from the 228s camera

User clarified 107s (briefly typed 117s). Exact camera restoration alone did not
explain visibility during manual exploration. Added scripts/trace-viewer-subject.py:
RGB+expected-depth backprojection of front person's torso at pixel493,330 in a
1280x720 107.51s render, then isolated 1,447 fixed Gaussian IDs within radius0.10
of the estimated world point. The same set remains renderable from 228.11s when
surrounding geometry is hidden, though fragmented. In the full 228-position view
aimed directly at that point, nearer geometry blocks it: target distance0.927882,
full-scene expected center depth0.171184, alpha0.999731; isolated subset center
depth0.942553, alpha0.950598. SH degree0 keeps the same visibility distinction.
This confirms occlusion in the tested view, not the cause of every manual view or
whether the obstruction is true geometry versus reconstruction error. Region is
not a pure semantic mask; includes nearby floor/background. No model/viewer edits.
Detailed evidence/limitations: docs/VIEWER_SUBJECT_TRACE_20261031.md. Artifacts:
results/viewer_subject_trace_20261031/. The already-started117s restoration audit
completed but is not used to explain the corrected107s report. Audit now accepts
--first-time and --second-time so its output names match actual timestamps.

## 2026-10-31: Fresh paused-camera audit and embedded browser check

Problem: user reports frozen views and poor quality off the recorded camera path.
Added scripts/audit-viewer-free-camera.py and tested current 8085 server with a
new paused WebSocket client: 107.51s pose, 5-degree in-place yaw, right offsets
0.01/0.03, forward offset0.03, exact original restoration (normalized units).
All four novel views changed; each received three frames, first-response latency
0.140-0.187s for novel views, final size2048x1152. Restored pixels exactly equal.
Visual comparison shows perspective changes and model distortion, not a frozen
original image. Existing camera-refresh fallback was already installed.
Native drag in user's VS Code embedded viewer visibly updated the image; controls
subsequently reset with new UUIDs, consistent with reconnect. Second paused drag
changed Rendered count115024->105025. Initial stale connection remains a candidate,
not an established root cause. Later wrong-window screenshot excluded as evidence.
No persistent renderer freeze reproduced; no new viewer patch/restart/training.
Recovery for a stale client: browser reload; geometry quality requires independent
pose/coverage/reconstruction work. Full evidence and limits:
docs/VIEWER_FREE_CAMERA_20261031.md; results/viewer_free_camera_20261031/report.json
and comparison.jpg. Audit executed successfully; existing model preserved.

## 2026-11-02: Reopen 120k courtyard viewer

Launched the saved final_post_update.pt checkpoint with scripts/view-room.ps1 on port 8085, camera index 1000. HTTP returned 200; viewer reported 705,300 Gaussians and 6,996 camera images. Background launcher PID 5112. This is a viewer restart, not training or completion of the previously failed finalization. Visual browser rendering was not checked.

## 2026-11-02: Recheck live 120k viewer updates

Problem: user reports the view may not update correctly and wants the most accurate scene at each viewpoint. Re-ran scripts/audit-viewer-free-camera.py against the live 8085 server, output results/viewer_free_camera_20261102. Four camera changes each produced three frames and a changed image; final images were 2048x1152 and restoring the original pose reproduced identical decoded pixels. First response for the four new views was 0.093-0.234 seconds. This tests an independent paused client, not the user's exact browser session or every trajectory. Inspected nerfview renderer: it uses lower resolution while moving, then refines at rest; static frames are JPEG quality 70. The fixed checkpoint supplies the same learned Gaussians for every viewpoint; camera-dependent projection changes, but time-varying objects are not reconstructed dynamically. No code, model, or process changes. If the user's client still freezes, exact browser/control and reproduction are needed; this audit does not prove their session works.

## 2026-11-02: Measure post-movement refinement frames

User reports no visible refinement after stopping. Earlier audit saved only the last frame and could not establish the refinement sequence. Added scripts/audit-viewer-refinement.py and recorded frame dimensions and arrival times in results/viewer_refinement_20261102/report.json. A fresh independent paused client received successive frames, including a 871x490 intermediate and a 2048x1152 final at 0.812 seconds for the forward test. Another case had a full-resolution first frame, so visual improvement need not always be obvious. Refinement increases render resolution and JPEG quality; it does not optimize or alter the learned Gaussians. The test does not establish whether the user's browser received or displayed the final frame, or whether the user expects geometry to improve. No model or viewer changes.

## 2026-11-02: Compare exact 107s and 218s capture replay views

User reports group appears at 107 seconds but not 218 seconds, thought to be approximately the same pose. Ran scripts/audit-viewer-pose-restore.py --first-time 107 --second-time 218 against port 8085; evidence results/viewer_pose_107_218_20261102/report.json and PNGs. Exact recorded poses differ by 0.89631 normalized position units and 55.20 degrees in orientation; normalized units are not established meters. The rendered 107s pose shows the group and the 218s pose shows a different view of the courtyard. Restoring the exact 107s camera while the replay timeline is at 218s yields pixel-identical 107s render: time itself does not remove the group in this test. Prior 228s subject trace showed occlusion in that specific aimed view, but was not repeated for 218s; do not claim 218s cause established. User's manually adjusted poses may differ from these exact recorded poses. No checkpoint or viewer code changes.

## 2026-11-02: Smooth capture timeline seeks

Problem: selecting a capture time jumped the camera immediately, making intermediate view changes impossible to observe. Changed examples/capture_path.py to interpolate current camera position, orientation (spherical interpolation), and field of view to the selected recorded pose over three seconds with eased motion. Seeks pause playback; repeated selections start from the current pose, and Pause cancels a transition. Return to start also moves smoothly. This is a direct virtual-camera transition, not traversal of the recorded path, and may cross scene geometry. No model changes. Seven capture/viewer tests passed. Live isolated 8086 test sent timeline 107s from an arbitrary camera and observed 43 successive position messages, ending exactly at the recorded 107s position, plus 44 background frames; actual browser perception was not visually tested. Restarted 120k viewer on port 8085, PID 15700, HTTP 200 and 6,996 capture poses loaded; reload browser to reconnect.

## 2026-11-02: Investigate courtyard revisit and apparent duplicate floor

29,276 saved pose edges contain no pair separated by 300 or more frames. SIFT revisit scan found frame 3206 vs 6470 with 262 ratio matches and 219 fundamental matrix inliers. Preliminary two-depth rigid RANSAC retained 29 of 55 depth correspondences (median 3.84 cm); its relative transform differs from saved graph by 13.65 m and 33.05 degrees. Matching concentrates on floor and the estimate requires independent non-coplanar/static validation. This supports a major revisit inconsistency, not proof of its root cause or a completed repair. Existing model and poses preserved. Full scope, evidence, repair requirements: docs/REVISIT_DRIFT_20261102.md and results/revisit_diagnostic_20261102/.

## 2026-11-02: Candidate revisit repair and trial

Problem: apparent duplicated courtyard across visits. `scripts/audit-courtyard-revisit.py` excludes an approximate group region and fits multiple RGB-D revisit transforms. Two candidate pairs (3176/6470, 3206/6470) pass preliminary checks; pair 3236/6470 is held out, but shares the later frame and is not independent of local pose errors. `static_revisit_pairs.json` includes inlier counts, depth distributions and transforms. Sparse keyframe correction uses the original optimized trajectory for local constraints and two revisits; the initial 100-evaluation solve hit its limit, so a separate rerun with up to 800 converged in 283 evaluations. Heldout relative-pose residual changed 13.794 m / 31.30 degrees to 0.175 m / 6.16 degrees. This is a limited relative measurement, not ground truth or sufficient validation of the entire reconstruction.

All 29,276 original graph edges assessed in `original_edge_validation.json`: median translation residual 0.00910 to 0.00915 m and p99 0.12456 to 0.12445 m; rotation p99 1.167 to 1.451 degrees. This is consistency with original constraints, not independent surface validation. Corrected poses and comparison figure are separate under `results/courtyard_revisit_repair_20261102/`; original checkpoint and poses are unchanged. Exported separate COLMAP dataset with all 6,996 cameras, original split/depth and 180,000 reinitialized points. Parser loaded all images and finite points. Static people treatment has not yet been chosen; current trial is unmasked.

Started a separate 20,000-step experimental training run using `scripts/run-courtyard-revisit-trial.ps1`, runner PID 1500. Status/log are under `results/courtyard_revisit_repair_20261102/trial/`. The matched-view comparison, static surface checks and final quality decision remain pending; running does not imply success or a repaired model. Candidate is globally constrained by only one revisit area and must not be called an established accurate trajectory.

Additional validation: retained heldout pair 3236/6470 has 44 fundamental-matrix-filtered RGB-D correspondences; median 3D separation under saved poses is 14.395 m and under candidate poses 0.220 m. Its 21 self-selected rigid inliers have medians 14.349 m and 0.195 m. The pair shares frame 6470 with constraints and inlier selection uses the pair, so these are local consistency checks, not independent global ground truth. Original 3206-to-6470 camera-center separation 13.830 m; candidate 0.423 m. Original 3206-to-6532 separation 12.232 m; candidate 1.932 m. `original_edge_validation.json` contains full percentiles. Diagnostic and export scripts compiled. Trial launch and auto-comparison watcher are running; completion and quality remain pending.

A further check used different image frames 3191 and 6455, neither used as a loop constraint. 23 of 41 depth pairs passed the preliminary rigid fit; candidate relative-pose disagreement decreased from 13.753 m / 32.315 degrees to 0.193 m / 4.794 degrees. This pair is temporally nearby and shares scene content, so it is not an independent global-accuracy benchmark. Artifacts: results/courtyard_revisit_repair_20261102/independent_pairs/.

User selected keeping people where possible; the experimental training therefore leaves them unmasked. This does not give a time-varying representation. Added meaningful rigid-fit and camera-pose-convention tests: 2 passed. The separate comparison watcher (PID 21384) waits for successful training, then renders 88 held-out views for the original 120k and corrected 20k models and compares them alongside saved original 20k results. It also saves four-panel source/original-20k/original-120k/corrected-20k previews near both revisits. Results remain pending.

## 2026-11-02: Corrected-pose trial result and follow-up

The 20,000-update unmasked corrected-pose trial completed with 466,697 Gaussians. Matched heldout evaluation on the same 88 targets: original 20k PSNR 19.9533 / SSIM .54451 / LPIPS .51322; original 120k 20.8095 / .57712 / .44103; corrected 20k 18.7991 / .47879 / .60762. Higher PSNR/SSIM and lower LPIPS are better. Different pose estimates and geometry initialization, plus sixfold update difference for the 120k model, limit causal interpretation. The first comparison failed because its environment lacked the venv Scripts directory in PATH, so Ninja was unavailable; fixed the watcher environment and reran successfully. Checkpoints and PLY exist; matched report and previews are under results/courtyard_revisit_repair_20261102/trial/.

Visual inspection of heldout frame_003197 shows the original models retain the group while the candidate is blurrier and the group largely disappears. In frame_006482 the candidate is also visibly softer than the original 120k model. Thus reject this trial as a replacement; it does not satisfy the user's request to keep the people. Aligning visits makes contradictory observations overlap, which is a plausible reason for suppression, but this trial does not isolate the cause. Original 120k scene remains the default.

Follow-up experiment: preserve the same heldout split and corrected poses, but exclude training frames 6300 through 6800 (426 training frames) from the late revisit where the group has left. This may favor their earlier appearance but also removes useful scene observations and is not a semantic mask. Export and separate training are underway. No dynamic-people capability is claimed.

People-preference candidate dataset export completed: 6,996 registered images, 180,000 depth-initialized points, 426 late-revisit frames removed from the training split, same 88 validation targets retained. Started separate 20k trial PID 19768 and comparison watcher PID 4124; result folder results/courtyard_revisit_repair_20261102/trial_people. Both are experimental and unverified while running.

## 2026-11-02: People-preference trial result

The people-preference trial completed 20,000 updates and 474,312 Gaussians. Same 88 heldout images: PSNR 18.4022, SSIM .46992, LPIPS .61898 versus original 120k 20.8095/.57712/.44103 and original 20k 19.9533/.54451/.51322. Excluding late views further lowers average fidelity relative to the unmasked corrected trial (18.7991/.47879/.60762). This is not a controlled comparison against the original 120k because update count, poses, initialization and train split differ.

At heldout frame 003197, the group reappears only as blurred ghosts; original 120k clearly retains them. Broad group crop PSNR/SSIM/LPIPS original 120k 22.3537/.65138/.26472 versus candidate 17.3554/.33158/.61674. At frame 006482, 120k PSNR 20.3361 and trial 11.2207: late-view exclusion severely harms that view. Thus reject the people-preference trial as a replacement: it does not adequately retain people and its overall quality is worse. Candidate viewer is available on 8087 for inspection, original remains 8085. Fixed its launcher PATH to include the venv Scripts directory after an initial Ninja-not-found failure; retry HTTP 200, 474,312 Gaussians and 6,996 poses. Comparison artifacts are under trial_people/matched_comparison.json and comparison_images/. Neither a validated full-scene pose correction nor successful people preservation has been established. Future work: stronger non-coplanar multi-revisit constraints and point/depth alignment, then explicitly separate moving content or mask only conflicting pixels rather than discarding entire viewpoints; validate before retraining at scale.

## 2026-11-02: New recording 120k run launched

Source: 20261030_134523.db3 (17,129,000,960 bytes). Started separate results/realsense_20261030_134523_120k and data/realsense_20261030_134523_clean using scripts/run-new-recording-120k.py --recording 20261030_134523.db3 --data data/realsense_20261030_134523_clean --result results/realsense_20261030_134523_120k --steps 120000. This uses the new recording's own calibration and pose reconstruction, 3 FPS qualifying-frame preparation, sensor depth weight 0.01, default uncapped densification and the previous 120k refinement schedule. Runner PID 11016 (venv launcher 21140); clean stage active at launch. The previous full-rate workflow requires anchors from the same recording, so no old-recording anchors were reused. Reconstruction/validation gates precede training; 120k updates and preview are pending and not yet verified. See status.json and stage logs for progress; inspect runner_error.log as well if the runner exits unexpectedly. Existing viewers remain active and consume GPU memory, which may affect training capacity.

At user request, stopped the two older viewer processes on ports 8085 and 8087 and their Python launcher processes. The new 120k pipeline was preserved; at the check it had progressed to RGB-D bridge preparation. No old checkpoints were deleted.

## 2026-11-02: New recording 120k result and viewer

The 120000-update run and preview finished; status complete. RGB-D reconstruction registered 477/477 selected 3 FPS frames (registration is not surface coverage). Preview verification reports 1,063,961 Gaussians, finite checkpoint tensors and video pixels, and successful PLY reload. Started the final_post_update.pt viewer on port 8083 using scripts/view-clean-realsense.ps1 -ResultDir results/realsense_20261030_134523_120k -Port 8083 -Background; launcher PID 24236. The preview benchmark context text is stale: the older viewers had been stopped at user request before training. HTTP availability checked; browser visuals not independently inspected.

## 2026-11-02: View pre-training point cloud

Started scripts/view-initial-point-cloud.py on localhost:8089 (launcher PID 4168), reading the RGB-D dataset sparse points through the same normalized COLMAP parser as the trainer. This is the depth-initialized reconstruction, before Gaussian optimization, not the trained Gaussian PLY. HTTP returned 200; browser visual appearance not independently inspected. Existing trained viewer remains on port 8083.

## 2026-11-02: Initial point cloud depth-consistency screening

User observed floating points. scripts/audit-initial-depth-consistency.py projects the 180,000 RGB-D initialization points into every eighth selected frame (60 views), comparing against nearest sampled aligned depth within two pixels. Support means within 8 cm in camera Z; free-space conflict means point is over 15 cm in front of the measured surface. 2,513 points have at least three free-space observations and zero supporting observations; 2,538 have at least three free-space observations and at least one support. These are diagnostic candidates, not confirmed errors: sparse samples, correlated views, pose error, dynamics and edges can cause conflicts. A density-only filter is unreliable: median fifth-neighbor distance 7.30 cm overall versus 8.31 cm among unsupported conflict candidates. No points or model were changed. Detailed counts, per-point evidence, views and thresholds are in results/realsense_20261030_134523_120k/point_cloud_audit/.

Follow-up: launched diagnostic overlay viewer on localhost:8090 via scripts/view-initial-point-audit.py (PID 20260). Sorted COLMAP point IDs explicitly align audit labels with trainer-normalized parser points. Red = 2,513 points with at least 3 free-space conflicts and no sampled support; orange = 2,538 with conflicts and support; other points keep source RGB. The colored points are intentionally larger for visibility. Inspected 8 spatially separated high-conflict examples with two source views each, in point_cloud_audit/candidate_source_sheet.jpg and candidate_examples.json. Several marked projections fall near exhibit statues, ropes and depth boundaries; visual review confirms these are ambiguous and does not validate automatic deletion or prove errors. Views are selected from the same screening set, not independent validation. Original cloud and trained model remain unchanged.

## 2026-11-02: New recording pipeline diagnosis

Reviewed input, reconstruction, initialization and 120k heldout/preview evidence; full diagnosis in docs/PIPELINE_DIAGNOSIS_20261102.md. RGB-only SfM registered 301/477 in best fragment; RGB-D graph registers 477/477 but graph residual is not independent accuracy. Of 1734 input pose edges only four span 30 or more selected indices; global pose quality needs independent revisit checks. Initial cloud uses single-view depth samples without a multiview support requirement, making depth and pose errors plausible floater sources. Source images show glare, bright openings and reflective objects. Mean heldout PSNR 23.1133, SSIM .79857, LPIPS .30898 on 60 jointly posed images, not free-view geometry. No controlled ablation or ground truth establishes relative blame; no new training or geometry changes made.

## 2026-11-03: Newest capture point cloud on port 8081

User requested all other viewers off and the newest room as a point cloud on port 8081. No Python viewer processes or listeners on ports 8080-8099 were present before launch. Started `scripts/view-initial-point-cloud.py --data data/realsense_20261030_134523_clean/rgbd/dataset --port 8081` using the project virtual environment. Startup reports 477 images and 180,000 RGB-D initialization points. HTTP returned 200 and port 8081 was the only listener in 8080-8099. This is the pre-training point cloud; browser visuals were not independently inspected. Logs: `.cache/initial-point-8081.stdout.log` and `.cache/initial-point-8081.stderr.log`.

## 2026-11-03: Compare statue photogrammetry video with newest RealSense pipeline

Problem: compare the supplied video pipeline/setup with the current room reconstruction and point cloud. Reviewed current manifests, reconstruction reports, RGB-D graph, saved training config, provenance and preview verification. Finding: supplied workflow uses spherical photos -> Metashape textured mesh -> manual cleanup -> newly sampled surface points -> Blender artistic rendering; current port-8081 cloud contains 180,000 depth initialization samples, while the saved 120k model has 1,063,961 optimized Gaussians. These are different representations and processing stages. Verified RGB-only 301/477 registration versus RGB-D 477/477; connectivity and fitted residuals do not establish geometric accuracy. Proposed solutions depend on goal: validated fusion/cleaned mesh and Blender for an artistic animation, or independent pose/depth validation before further 3DGS training for free-view quality. Video retrieval was throttled; its exact hardware/settings remain attributed to the supplied summary. No processing or viewer changes made. Detailed comparison, problems, proposed solutions, sources and limits: docs/VIDEO_PIPELINE_COMPARISON_20261103.md.

## 2026-11-03: Existing-capture improvement workflow on current hardware

User requested a Markdown improvement workflow and clarified that recapturing is currently unavailable. Created docs/EXISTING_CAPTURE_IMPROVEMENT_WORKFLOW.md, focused entirely on 20261030_134523.db3 and the GTX 1660 SUPER 6 GB. Plan prioritizes baseline/split preservation, recorded-frame inspection, RGB-depth checks, independent pose/revisit validation, conservative multi-view depth screening/fusion, then matched short training and targeted additional recorded frames. Includes pass conditions, existing CLI examples, a proposed capped 12k profile, research-log template, resource limits and explicit separation of existing tools from unimplemented validation/fusion/masking/split features. Commands were checked against script definitions; experiments were not run. No viewer, dataset or model changes made. The earlier courtyard diagnostics are not treated as evidence of the same displacement in this recording. Recapture is not a required step.

## 2026-11-03: Verify room recording baseline hash

Problem: establish whether the existing source recording matches the hash saved with the 120k baseline before improvement experiments. Ran `Get-FileHash -Algorithm SHA256 -LiteralPath '20261030_134523.db3'` from `D:\gsplat\gsplat`. The computed SHA-256 was `60e3f253051fc282a61af7f6c9c5eef40830a12d42566b9eba0992463115bc52`, matching `source_sha256` in `results/realsense_20261030_134523_120k/provenance.json`. Decision: source hash check passed. The provenance's `initial_source_sha256` and `source_hash_unchanged` are null, so this comparison does not independently establish that the recording remained unchanged throughout the original run. Next: preserve the held-out frame identities and fixed inspection poses.

## 2026-11-03: Freeze room baseline held-out source identities

Problem: resampling can change filenames and the default evaluation split, making future comparisons unmatched. Saved `docs/room_baseline_heldout_20261103.csv` with 60 evaluation indices, sorted registered indices, image names, original RGB and paired depth message IDs, database timestamps in nanoseconds, header timestamps and pairing gaps. Reconstructed the split from the baseline dataset's registered COLMAP images, sorted by filename, selecting zero-based indices divisible by 8, consistent with `examples/datasets/colmap.py` and the saved `train/cfg.yml` (`test_every: 8`, factor 1). No explicit split file exists in the baseline dataset. Checked 477 unique registered images against the capture manifest, all 60 selected image files exist, and all selected RGB/depth IDs and timestamps match the appropriate topics in the original DB3 opened read-only. Each selected RGB and depth ID is unique. The count agrees with `dataset_validation.json` (417 train / 60 held-out). CSV SHA-256: `4422155d086d005e120c8f2ab5b9c3f77739143df8d3e060589766ef892eb78d`. Decision: baseline source identities preserved; future resampling still requires a verified source-identity split override and temporal exclusion policy. Next: save fixed inspection poses and screenshots.

## 2026-11-03: Save user-selected baseline inspection views

Problem: improvements need repeatable visual comparisons. User selected replay times 30, 43, 61, 68, 83, 97, 99, 110 and 37 seconds, with wide, difficult and reflective labels. `scripts/save-room-inspection-baseline.py` selected the closest registered source frame to each requested time, using seconds relative to the first registered RGB timestamp, and rendered the baseline 120k checkpoint. Outputs: `results/room_improve_baseline_inspection_20261103/inspection_poses.json`, nine PNGs and `contact_sheet.jpg`. The JSON records exact RGB/depth IDs and timestamps, normalized camera-to-world matrices and position/rotation, intrinsics, FOV, aspect, dimensions and render settings. The closest selected replay times are 29.9692, 43.0185, 61.3072, 68.1485, 83.0333, 96.9831, 98.9857, 109.6982 and 36.9779 seconds respectively; maximum absolute offset 0.3072 seconds. All nine 848x470 PNGs and finite 4x4 poses verified; contact sheet visually inspected. Labels are user supplied; notably the 83-second image centers on the exhibit, so its suitability as a reflective-area control needs user review. These are recorded-camera views rather than manual free-camera viewpoints. Render uses degree 3, black background, packed rasterization and original normalized coordinate frame. If candidate normalization changes, align coordinate frames before fixed-world comparison. Initial run triggered an unintended CUDA rebuild from omitted `NUM_CHANNELS`; interrupted it and restarted with the viewer's `NUM_CHANNELS=3,4`. Rebuild completed and rendering succeeded. Original recording, dataset and checkpoint unchanged. Decision: baseline views saved; next inventory actual problem regions and inspect depth overlays and revisits.

## 2026-11-03: Inspect selected room views and depth samples

User requested RGB, Gaussian and depth comparisons. Created `scripts/inventory-room-inspection.py`, nine comparison sheets and source/neighbor metadata under `results/room_improve_region_inventory_20261103/`; inspected each and documented findings and candidate revisits in `docs/ROOM_REGION_INVENTORY_20261103.md`. At 61 s a dark smeared area appears near the doorway in the Gaussian render; at 37 s a visitor seen in RGB is largely absent in the static render; 43 s supplies a provisional room control and 110 s a reflective display target. 83 s primarily shows the exhibit. Proposed 37/43 s room and 30/61 s diorama overlaps require correspondence validation and are not accepted loop constraints. Existing stored depth consists of 16,000 samples per frame, not dense validity; color indicates distance, and overlay inspection cannot measure boundary alignment or prove depth absent. Prepared neighboring RGB frames were inspected, not additional raw DB3 frames. The alignment gate remains unpassed; next obtain dense unfilled aligned depth, inspect boundaries and extra source frames, and validate proposed revisits. No point deletion, pose correction or training was performed.

## 2026-11-03: Initial dense RGB-depth alignment audit

User requested the alignment check. `scripts/audit-room-rgbd-alignment.py` read six paired depth messages from the DB3 read-only and saved unfilled, unsmoothed metric projections, boundary overlays and metadata under `results/room_improve_rgbd_alignment_20261103/`. The selected replay times were 43.019, 61.307, 68.149 and 109.698 s plus higher single-sample gyro magnitude frames at 65.679 and 108.697 s. On each frame the baseline cleaned depth was recomputed and its 16,000 saved sample depths and pixel keys matched exactly (maximum depth difference 0 m), a pipeline reproduction check rather than independent calibration validation. Diagnostic projected coverage was 45.1–82.5% of prepared pixels, reflecting range, invalid pixels, sampling and occlusion, not accuracy. Far structure in the 43/61 s views often lacks projected samples; at 68 s some depth jumps visually trace exhibit silhouettes; glass at 110 s remains ambiguous. All six maps and images were checked for finite values, dimensions and recorded counts. The script's nearest-Canny-edge distances are exploratory and not valid alignment errors. High-motion views differ in scene content, so no motion-dependent error is established. Full method, links, observations and limitations: `docs/ROOM_RGBD_ALIGNMENT_AUDIT_20261103.md`. Decision: inconclusive; RGB-depth gate not passed. Next: measure independent signed offsets on selected non-reflective static edges at multiple depths and motion conditions; validate any correction on held-out views. No source/dataset/model changes, point deletion or training.

## 2026-11-03: Exploratory signed RGB-depth boundary measurements

User requested measurement and a motion comparison. Extended `scripts/audit-room-rgbd-alignment.py` to accept an explicit list and output directory; projected 14 prepared frames at 64.979–69.016 s in `results/room_improve_alignment_motion_20261103/`. Existing sampled preprocessing points reproduced on all 14. `scripts/measure-room-rgbd-edges.py` compares independently selected strongest RGB gradient and valid depth jump per horizontal scanline in manually chosen strips; sign is depth x minus RGB x, with left-sample convention. On one standing-statue head silhouette, development medians were -2 px (20/25 rows, gyro sample 0.599 rad/s), -4 px (21/25, 0.295), and -2 px (28/35, 0.108). Separate nearby 000203 had -2 px (21/31); 000198 had only 3/27 and a visually bad match. Candidate markers were reviewed; some select texture. A second backdrop test was rejected because selections land on different physical edges or surfaces. Ten regions, 418 rows, outputs and marker images verified; scanlines are correlated and not independent uncertainty estimates. No correction was fit, and held-out frames share the trajectory; no causal motion or calibration conclusion is justified. Findings, evidence and restrictions: `docs/ROOM_RGBD_EDGE_MEASUREMENTS_20261103.md`; raw rows and markers: `results/room_improve_edge_measurements_v02_20261103/`. Decision: alignment remains inconclusive; next use manually reviewed non-reflective correspondences on multiple structures, distances and orientations and independently test any candidate. No source, baseline data or model modified and no training.

## 2026-11-03: Manual review of additional RGB-depth structures

User requested the next alignment step. Inspected enlarged RGB/depth discontinuity grids of standing and seated statues, railing posts, backdrop and front exhibit region. `scripts/record-room-manual-edge-review.py` records visually selected locations assisted by Sobel/depth-jump responses and saves 11 annotated examples, data and grids in `results/room_improve_manual_edges_20261103/`. Five exploratory standing-head top samples across frames 197, 200 and 203 give signed depth-start minus RGB y values 0, -1, -3, 0, +1 px; these are one object, correlated, and frame 203 shares the trajectory. Three apparent right-post offsets and three seated-head measurements rejected due to broad/multiple depth transitions or unclear correspondence; the other post and backdrop provide no accepted independent constraint. Verified CSV arithmetic and all 11 marker images; script compiled. Detailed selection, rejection and limitations: `docs/ROOM_MANUAL_EDGE_REVIEW_20261103.md`. Decision: no global x/y, timing or calibration correction supported; alignment gate remains unresolved. Next search additional source frames for clear non-reflective edges at different distances and matched motion, reserving independent regions/frames for any candidate test; if evidence is absent, scope reliable depth rather than claim global success. No original data, model, graph or cloud modified.

## 2026-11-03: Search more recorded frames for alignment targets

User requested help with the next step. Reviewed a 40-view overview across the capture and reprojected paired original depth for 22 prepared views into `results/room_improve_target_projections_20261103/`. Identified black-box left edge in frame 000024 at 8.144 s, with paired projected depth about 2.2–2.3 m. In nine visually selected scanlines, eight unique depth-jump candidates had depth-start minus RGB x median 0 px and range -1 to +2; one had no qualifying jump. These rows are correlated, and strongest RGB/depth edge may not denote the same physical boundary. `scripts/extract-room-target-raw.py` extracted seven original RGB/depth pairs near that timestamp into `results/room_improve_target_raw_20261103/`; reference prepared RGB and separately reprojected depth matched exactly. `scripts/measure-room-raw-target.py` checked the same edge in all seven: per-frame candidate counts 4,3,1,7,5,0,1 of 8, medians +5.5,+3,+1,0,0,undefined,-3 px. Nearby strips were chosen after inspection; markers show potential different surfaces and validation is not independent. No timing or global calibration conclusion follows. Kiosk and distant doorway candidates remain ambiguous. Verified seven map/image dimensions, 56 records and exact reference depth equality; scripts compiled. Evidence, rejected cases, methods and scope: `docs/ROOM_ALIGNMENT_TARGET_SEARCH_20261103.md`. Decision: one additional candidate found, but global RGB-depth gate still inconclusive; no correction, pose changes, point deletion or training. Search is sampled, not exhaustive.

## 2026-11-03: Physical RGB-depth foreground and background review

User requested foreground/background verification and additional static targets. `scripts/review-room-physical-edges.py` annotated manually selected interiors and recorded their unfilled projected camera-Z in `results/room_improve_physical_edges_v02_20261103/`. For box RGB IDs 6487 and 6253, box medians 1.902 and 1.897 m versus wall/column 2.343 and 2.330 m support a foreground/background interpretation; at their boundaries, profiles show intermediate values and missing projections. Earlier single adjacent-jump offsets do not measure an exact physical silhouette. A separate kiosk pedestal has median 2.167 m versus wall 2.519 m, and a statue head has 1.184 m versus background props 2.080 m, but both edges have ambiguous transitions or gaps. Depth values are not independent range ground truth. Initial probe output had a mislabeled statue region and was retained as `results/room_improve_physical_edges_20261103/`; corrected positions are in v02. Reserved prepared frames 000004, 000145, 000295 by original RGB/depth identities for a possible future check; suitability untested, nearby scenes previously inspected, and they are separate from the original 60-image held-out list. Verified source IDs against capture manifest, four images, 200 profile entries (six missing), dimensions/finiteness, and script compilation. First `rg` wildcard invocation failed under PowerShell and was rerun using `-g`. Full methods and evidence: `docs/ROOM_PHYSICAL_EDGE_REVIEW_20261103.md`. Decision: alignment gate remains inconclusive; no fit or correction supported. Next, freeze an edge-transition method on development objects and use suitable reserved frames for subsequent checks, or document limited depth use if independent evidence remains inadequate. No source, baseline, graph, dataset or model changes; no training.

## 2026-11-03: Room improvement steps 1–6 audit and diagnostics

At user request, rechecked DB3 SHA-256 `60e3f253051fc282a61af7f6c9c5eef40830a12d42566b9eba0992463115bc52` against baseline provenance, verified all 60 held-out IDs/timestamps against manifest, nine finite recorded camera poses and screenshots, and baseline checkpoint. Baseline provenance lacks an initial hash, so original-run immutability is unproven. `scripts/audit-room-transition-profile.py` measured 24 preselected development scanlines as 10–90% depth transition bands: 14 had holes, 4 were rejected, 6 had no hole but correspondence unverified. Defined the algorithm before viewing reserved overlays; then selected their strips after viewing. `scripts/check-room-reserved-transitions.py` yielded 7 ambiguous and 8 rejected of 15 rows. Correspondence and calibration remain unresolved; no correction. Synthetic transition-shift and missing/equal-depth tests passed (2; pytest warns unknown config `env`).

`scripts/audit-room-revisits-rgb.py` assessed six SIFT/RANSAC pairs without relying on depth. Pair 89/183 had obvious wrong matches, 249/290 had too few, and 204/249 matched primarily the planar exhibit backdrop; no graph constraint accepted. Fitted-inlier pose residuals and match images are diagnostic only and do not independently verify trajectory. `scripts/audit-room-depth-independent-groups.py` screened all 180,000 points in two disjoint sets of 30 training views (no overlap with saved 60 held-out images). Of 604 points with >=3 development free-space conflicts and no support, 198 have check-group support, 452 have check-group free-space observations, and 73 have no check match; support and free-space categories overlap, and all evidence shares capture, poses and preprocessing. These are not verified deletion labels. Exported all 180,000 points to a diagnostic-only colored PLY, preserving all points, and verified reload. No cleaned or fused training cloud, pose changes or alignment correction has been validated; do not bypass gates. Full commands, evidence, scope and caveats: `docs/ROOM_STEPS_1_TO_6_PROGRESS_20261103.md`; outputs `results/room_improve_transition_profiles_20261103/`, `results/room_improve_transition_reserved_20261103/`, `results/room_improve_rgb_revisits_20261103/` and `results/room_improve_multiview_diagnostic_20261103/`. An incorrect `ckpts` listing path failed; verified the actual `train/ckpts` checkpoint from inspection metadata. Six scripts compiled. No baseline, source, graph or model changes; no training.

Reviewed eight spatially separated flagged-point examples using `scripts/review-room-multiview-examples.py`; outputs `results/room_improve_multiview_examples_20261103/`. Examples selected for extreme conflict or close support are illustrative, not representative; several land on exhibit figures, reflective displays or blurred views. Check-group support can include a point's own source observation, as the baseline cloud may use that group, and is not independent ground truth. This reinforces the no-deletion decision.

## 2026-11-03: Six-step continuation and full-room preference

User selected retaining the full room and clearly marking uncertain areas. Saved a repeatable baseline as four separate 5-view local lateral/yaw sweeps around 43, 61, 68 and 110 s, with cuts between regions; 20 poses and renders, contact sheet and decodable MP4 in `results/room_improve_free_path_v02_20261103/`. Each center is pixel-identical to the earlier recorded-view render; matrices, camera settings, checkpoint hash and limitations are saved. First launch failed because CUDA toolkit was not in process environment; left its output directory and used `scripts/run-room-free-path.ps1` to initialize project MSVC/CUDA and `NUM_CHANNELS=3,4` for a successful separate run.

Reprojected 12 additional source frames and prepared an annotated inventory of 10 candidate structures, not ten validated alignment targets; outputs `results/room_improve_structure_projections_20261103/` and `results/room_improve_static_inventory_20261103/`. Existing heldout frame 000000 is among inspected inventory frames and earlier work also reviewed some heldout views: original 60 IDs remain excluded from training, but these evaluations are not fully blind to human development. Numerical recalculation of K/crop agrees within 5.69e-14, units match .001 m, raw and saved rotation/translation match under the existing column-major convention, and all 477 selected RGB/depth gaps have median 7.244 and maximum 8.927 ms. These are bookkeeping checks, not proof of physical alignment, synchronization, or transform direction. `scripts/audit-room-revisits-split.py` tested 14 SIFT pairs, with feature train/check splits before fitting; 10 pairs were fit, four had insufficient matches, some remain planar or have large outlier tails. No independently validated pose correction.

Built `results/room_improve_full_source_cloud_20261103/` from 387 training-source frames, 1,500 depth samples per frame and 2.5 cm voxel selection, capped at 180,000. Reserved 30 separate training frames for depth checks, >5 s from each point's source; all 60 existing evaluation IDs excluded from source and check. Existing graph and calibration are shared and may use check frames; this is not independent physical ground truth. The initial sparse check found 78,183 without an observation. Reprojected all 30 original unfilled check depth frames and repeated: all-valid direct pixel comparison had 34,141 with no observation, 47,569 with support, 3,162 with >=3 free-space conflicts and no support, and 1,032 with both support and >=3 conflicts. Stricter 3x3 stable neighborhoods had 50,612, 41,453, 1,596 and 296 respectively; category overlap and correlated views apply. Eight extreme/selected visual examples include reflective displays, blur, and exhibit shapes; these do not verify deletion. Original-to-experimental nearest-neighbor median 3.35 cm and p90 8.05 cm describe sampling similarity, not geometry accuracy. Exported original-color candidate and dense-check uncertainty-colored cloud with all 180,000 sampled points. User preference is respected as intent, but actual full-room surface coverage and improved cleaning are unverified; no cleaned candidate accepted or training run.

Validated source/check/heldout separation, 200 point/source reprojections, PLY reload, stable-count subset, and 30 depth preprocessing reproduction checks. Python scripts compiled, two synthetic transition tests passed with existing pytest `env` warning. Full artifacts, script hashes, commands, user preference and findings: `docs/ROOM_IMPROVEMENT_CONTINUATION_20261103.md` and `results/room_improve_full_source_cloud_20261103/continuation_provenance.json`. The full-room geometry alignment and pose gates remain open; next inspect additional spatially distributed static correspondences and independent calibration evidence before filtering or training.

Compared original and experimental point projections at the four fixed recorded-camera views with `scripts/compare-room-cloud-fixed-views.py`. Initial one-pixel thumbnail sheet was too dark, so retained it and produced `results/room_improve_cloud_fixed_views_v02_20261103/` with 3x3 display-only dilation. One-pixel occupancy counts are similar, but do not establish accurate geometry or full surface coverage; dilation can merge points and alter colors. No new cloud acceptance claim.

## 2026-11-03: SDK inverse-Brown discrepancy and isolated candidate

Installed `pyrealsense2` 2.59.1.11053 to isolated `.cache/room_sdk_audit/install` (wheel SHA-256 `045dfae020a635ba0c2c27e79ea3fa844acb86101344f68ca4c4163beb94b721`), not the venv site-packages. `scripts/audit-room-realsense-sdk.py` compared 63 raw RGB coordinates against SDK inverse-Brown deprojection: baseline's implied Brown mapping differed median 11.86, max 19.38 raw RGB px; alternate inverse mapping differed max 0.029 px. On 35 inverse-remap round trips, existing output-to-input mapping differed up to 19.39 raw px from SDK, alternate mapping up to .026 px. Depth deprojection and transform math agreed within 4e-7 m in sampled cases. This is a convention discrepancy against SDK using same recorded calibration, not physical ground truth or timing validation; it supersedes any implication that K/crop checks validated the distortion mapping. Original K and transform bookkeeping were consistent, but RGB remap was not SDK-consistent.

Ten frames and four subsequently selected nearby frames were remapped from DB3 read-only; original preprocessing reproduced pixel exactly. Ten chosen pre-candidate but scenes are correlated and previous exposure precludes blind claims. Whole-image raw source-map displacement median 9.48, max 19.52 raw px. On three retrospectively chosen kiosk views, median old-to-depth-band distances were 5/5/7 prepared px vs 0/0/0 for SDK-consistent; holes, correlated scanlines and target selection limit conclusion. Additional statue frame 239 exploratory median 8 vs 1 px across five no-hole rows; nearby frame 243 had only one band; post and box markers select mixed edges and are rejected. No global alignment or timing pass.

Prepared all 477 SDK-consistent candidate RGB images in `data/room_improve_sdk_rectified_20261103/`, preserving source identities and original 60 heldout names; all images decoded and 10 matched separately generated candidates pixel-for-pixel. Original source/model untouched. RGB COLMAP stage completed 301/477 (15,145 points, in-sample mean reprojection .610 px; old RGB model 301/477 and .741 px, different detected features), though PowerShell returned exit 1 from redirected native stderr logs despite complete status and valid outputs. Candidate RGB-D graph with copied original target-K depth initially connected 345/477; retained prebridge snapshot and used candidate-specific SDK remap in intermediate-frame bridge; 44/44 boundaries accepted, producing 477/477 connected graph and 180,000 initialization points. A first attempt to copy snapshot into a non-existent directory created a single file; left it in place and made a correct snapshot directory. Registration/connectivity and graph residuals do not establish geometry accuracy.

First dataset validation failed on our explicit split key `test` versus loader's `val`; retained both failed files as `experiment_split.before_val_fix.json`, corrected only the key, reran validation: 477 registered, 417 train/60 val, exact original heldout identities, finite sample check passed. Both graphs use all frames for pose fitting; val is not independent pose truth. Rigid-fit camera alignment yielded heldout median/p90 positional differences .143/.712 m and orientation 2.63/4.47 degrees; these are differences, not truth. Candidate and baseline compared on identical corrected-image feature pairs showed mixed residuals; exhibit features are largely planar and graph inputs overlap check views.

Built independent-source candidate 180,000-point diagnostic cloud from 387 source frames; checked 30 disjoint source frames >5 s from each source. On stable raw projected 3x3 depth, 48,844 points lacked observations, 45,594 had some support, 1,198 had >=3 conflicts/no support, 268 had both >=3 conflicts and support. Matched 67,769 identical source depth samples across original/candidate clouds: any support 14,145 vs 15,629, >=3 conflicts/no support 602 vs 476, no observation 20,095 vs 19,083. This is exploratory: voxel-selection conditions the paired subset, check view projections change with poses, graph used same check views, calibration and source; cannot infer independent ground-truth improvement. Saved candidate-color and uncertainty-color clouds and captured/fixed-world comparisons; fixed-world alignment approximate and poses differ non-rigidly. No filtered cloud accepted, no training. Five meaningful numerical tests passed; pytest reports existing unknown `env` option. Full methodology, evidence, errors, caveats and paths: `docs/ROOM_SDK_REMAP_FINDING_20261103.md`.

Independent-of-depth exploratory straight-line check `scripts/audit-room-sdk-straight-lines.py` found slightly higher median line residual for SDK candidate on map-panel frames 001/002 (1.18 vs .98 px; .43 vs .30 px), similar on 003 (.61 vs .68), and mixed backdrop results. Markers show some texture or other structure rather than the intended line; selection retrospective and nearby rows correlated. This contrary/inconclusive evidence is retained in `results/room_improve_sdk_straight_lines_20261103/` and does not invalidate the numeric SDK discrepancy, but prevents a broad visual-improvement claim.

Follow-up `scripts/audit-room-sdk-3d-revisits.py` checked six RGB revisit pairs that are absent as direct edges in both graphs, using feature-split candidate correspondences and stable 3x3 dense depth under each mapping. Three pairs had zero jointly usable depth, 191/204 had only four (old/candidate median .337/.434 m), and near-planar exhibit pairs 204/249 and 205/250 had 42/43 pairs with old/candidate medians .064/.065 m and .046/.046 m. Thus these checks do not demonstrate improved pose geometry. Individual frames and related observations influenced graph optimization; candidate-based match filtering, possible mismatches and depth errors prevent treating them as external ground truth. Artifacts: `results/room_improve_sdk_3d_revisits_20261103/`.

Verified SDK candidate cloud source/check/heldout separation, 200 source-point reprojections and two PLY reloads with `scripts/verify-room-sdk-source-cloud.py`. Its first invocation failed because the reusable verifier expected the previous candidate's uncertainty filename; parameterized the PLY filenames and reran successfully. This verifies provenance, not physical calibration. Recorded updated SDK candidate script and artifact hashes in `results/room_improve_sdk_source_cloud_20261103/sdk_candidate_provenance.json`.

## 2026-11-03: User-provided statue workflow reference

The user supplied a summary of Massimo Vendola's statue video (https://www.youtube.com/watch?v=YMbZvOLSK6E). According to that summary, the workflow uses Agisoft Metashape to align spherical images and create a textured mesh, then Blender to manually remove unwanted mesh artifacts and distribute points on the mesh for an artistic animation. This is a user-provided description, not an independently verified video transcript. Our recording is D455 RGB-D rather than spherical imagery, and our experimental PLY is a sampled point cloud, not a textured mesh or trained Gaussian model. Blender-style cleanup may help presentation but does not independently establish camera-pose or calibration accuracy; meshing and resampling may obscure gaps or introduce interpolated surfaces. Keep original data and the unfiltered candidate, preserve uncertain room regions, and evaluate any separately saved cleanup against source observations and matched views before accepting it.

## 2026-11-03: Experimental room meshes

At the user's request generated separate colored test meshes using Open3D 0.20.0 CPU ball pivoting, radii 0.045/0.075/0.12 m, normals estimated within 0.15 m and oriented toward each depth sample's source camera. The baseline, original 180k candidate cloud and trained model remain untouched. `scripts/mesh-room-sdk-experimental.py` produced `results/room_improve_sdk_test_mesh_20261103/room_experimental_mesh.ply` and a second vertex-uncertainty-color version. The RGB mesh reloads with 180,000 vertices and 144,305 triangles; 112,105 vertices participate in faces. Visual comparison at four candidate camera views has extensive holes and fragmentation, so this is not a smooth or accepted room mesh. Initial attempt to inspect a guessed nonexistent builder script path failed; actual source was found and used.

To investigate density, `scripts/build-room-dense-mesh-source.py` sampled 8,000 aligned depth pixels from each of the same 387 development frames (3,096,000 samples), chose one measurement per 2.5 cm voxel, and capped to 900,000 points with seed 4913. It excludes the original 60 evaluation and 30 check frames as source frames, but the pose graph used all frames. No uncertainty evidence was computed for the dense sample and no points were cleaned. Ran `scripts/mesh-room-sdk-experimental.py` with `ROOM_MESH_SOURCE=results/room_improve_sdk_dense_mesh_source_20261103` and `ROOM_MESH_OUTPUT=results/room_improve_sdk_dense_test_mesh_20261103`; `room_experimental_mesh.ply` reloads with RGB vertex colors, 900,000 vertices, 546,983 triangles, 423,950 vertices used in faces; max triangle edge 0.240 m. The 97.7 s reported includes preview generation. Compared with the first trial, the four saved approximate captured-view previews show more continuous areas of wall and floor, but still have substantial holes, jagged/incorrect connections and smeared geometry, especially near exhibits and reflective surfaces. Camera poses, RGB/depth alignment, missing sensor depth and full-room physical coverage remain unverified. Previews use average triangle colors and a painter approximation, not per-pixel depth, so cannot reliably establish geometry accuracy or exact occlusion. The dense mesh has no uncertainty-color companion, and extra isolated vertices may be ignored by mesh viewers. Input/mesh hashes and counts are recorded in each output `report.json`; original RGB and both candidate mesh files are separate. Decision: experimental only, no accepted cleanup, training or claim of smooth full-room reconstruction.


## 2026-11-03: Distributed SDK revisit and mesh/source review

Expanded the SDK 3D revisit diagnostic to 253 pairs from 30 existing unfilled-depth frames, using index separations 32?192. 242 had too few matches, three lacked enough stable depth, and eight produced diagnostic pairs; check-point image hulls occupied only 0.55%?6.74%. Four candidate median 3D separations were lower and four higher; features were concentrated on display panels or the exhibit. The apparent 58/154 improvement 0.789 to 0.310 m is not accepted as physical evidence because of repeated display content and limited coverage. Reviewed four candidate mesh/cloud views alongside original RGB and unfilled-depth overlays: broad source-depth gaps, exhibit discontinuities, and reflective cases coincide with uncertain/fragmented regions. No blanket deletion or inferred surface fill. See `docs/ROOM_DISTRIBUTED_VALIDATION_20261103.md` and `results/room_improve_sdk_distributed_revisits_20261103/`. Decision: calibration/trajectory gate remains open; no supported new pose constraint, geometry repair, training or baseline modification. Next obtain spatially distributed validated static correspondences, or additional capture if unavailable, then separately optimize and assess before training.


## 2026-11-03: Existing room viewer and exploratory revisit graph trial

User wants a freely viewable full room using the existing recording. Launched the preserved 120k baseline on port 8090 with `scripts/view-clean-realsense.ps1`, HTTP 200 verified; no new interaction or full coverage claim. Reviewed distributed match sheet; manually reviewing the sheet is not per-correspondence physical validation. Fit separate rigid transforms with feature-level split in `scripts/audit-room-revisit-rigid-repair.py`. For exhibit pair 202/250 check residual median 0.110 to 0.011 m on 17 correlated checks; insufficient independent coverage. Ran `scripts/try-room-revisit-pose-repair.py`: simplified keyframe graph with this single fitted loop, retaining original trajectories. Loop residual 0.264 m/4.37 deg to 0.017 m/0.14 deg, but independent-of-loop fitted pair 202/234 residual worsened 0.865 m/9.64 deg to 1.005 m/10.13 deg; pair fits are uncertain and not ground truth. Median/max camera center displacement 0.105/1.073 m. Rejected candidate, no training or model replacement. Artifacts and limitations in `docs/ROOM_REVISIT_REPAIR_TRIAL_20261103.md` and `results/room_improve_sdk_rigid_repair_20261103/`. Need distributed verified physical correspondences; full-room completeness remains unverified.


## 2026-11-03: One-command RGB MP4 to Gaussian scene workflow

User requested a single runnable script for an MP4, with unknown phone intrinsics. Added `scripts/train-mp4-scene.ps1`, `scripts/mp4-to-3dgs-prepare.py`, and `docs/MP4_TO_3DGS.md`. Uses installed imageio-ffmpeg and pycolmap, single shared SIMPLE_RADIAL intrinsics estimated from RGB frames, sequential matching, undistortion, budgeted existing trainer, and background viewer. Default longest side 960, 2 fps, 7k steps, 500k splat cap. Each run writes isolated paths and logs; preparation stops under 80% registration, 20 images or 100 points. This is a heuristic and cannot prove room completeness or geometry quality. No sensor depth, accurate metric scale, 360 video, changing video crop or full unseen surfaces are supported.

Constructed test MP4 from 48 pre-existing room_small images (not user footage). Default 80% gate correctly rejected it: largest model 28/48 (58.3%). With diagnostic threshold 50%, preparation and undistortion completed; a 1,000-step GPU test produced checkpoint and PLY. Full wrapper test produced checkpoint `results/mp4_fixture_20261103_210710_c406d7/ckpts/ckpt_999_rank0.pt` and launched port 8092; startup took longer than the original 30-second check, which warned, but HTTP 200 was verified afterwards. Extended check to 60 seconds; long startup may still trigger a warning. Final revised image scaling and wrapper `-PrepareOnly` also completed with the diagnostic threshold. Fixture quality and artificially lowered threshold do not establish suitability for an actual MP4, and no user MP4 was provided. Existing 8090 room viewer preserved. Python script compiled and PowerShell wrapper ran in the existing environment.


## 2026-11-04: MP4 training duration and Gaussian cap

User requested 120,000 training steps and a maximum of 1,000,000 Gaussians while MP4 feature matching was already in progress. Active preparation PID 18884, source VID20261104003648.mp4, work directory `data/mp4_VID20261104003648_20261104_005941_5bab0b`, extracted 20,090 frames at user-configured 30 FPS. Running wrapper has already bound its old 7,000/500,000 values, so changing its source defaults alone cannot change that running instance. Added a schema-validated `training_override.json` next to the active dataset path and changed `test-small-dataset.ps1` to read that file at training launch, overriding steps/cap for that run without stopping matching. Changed future wrapper defaults to 120,000 / 1,000,000 and widened its step limit. Kept the existing 6,000-step densification cutoff; remaining updates optimize existing Gaussians. Test: a separate BuildOnly fixture read and printed the 120,000/1,000,000 override and reported CUDA_BACKEND_READY. First fixture creation command had a PowerShell here-string syntax error; corrected it with Python and reran. This tests reading and toolchain only; the active job has not yet passed registration or begun training, and completion or GPU memory fit is not established. 20,090 frames imply relatively few visits per image even at 120,000 updates; 30 FPS also makes preprocessing lengthy.


## 2026-11-04: MP4 versus DB3 preprocessing runtime explanation

Read-only inspection of running MP4 job `data/mp4_VID20261104003648_20261104_005941_5bab0b`: 20,090 frames extracted at 30 FPS; COLMAP uses CPU, up to eight threads, sequential overlap 12 with quadratic overlap. Its SQLite database had 20,090 image and keypoint rows and 236,985 match and two-view geometry rows at inspection; process remained present. Counts are not a completion percentage, and database contains rejected as well as usable match attempts. The earlier `20261030_134523.db3` room workflow used 477 selected frames at 3 FPS from 4,764 frames. The MP4 supplies about 42.1 times as many selected images, making work and matching substantially larger; captures and workflows differ, so this ratio is not a measured time multiplier. MP4 and DB3 are containers, not intrinsically fast/slow guarantees; DB3 provided sensor calibration and paired depth for that pipeline, while this MP4 pipeline estimates intrinsics and uses RGB only. No run or settings changed during this explanation.


## 2026-11-04 08:15: MP4 job elapsed time and progress check

Read-only status: job started 00:59 and remains active with about four CPU cores in use at the sample. 20,090 images/features and 236,985 pair rows; reconstruction directory created 01:52, and saved model 1 at 07:02 has 2,666 registered images and 68,337 points; model 0 has 8 registered and zero points. The running mapper may have additional unsaved progress; saved-model counts are not current global progress. No ETA can be inferred from saved outputs, and mapper allows up to five models. The acceptance gate requires a single model with at least 16,072/20,090 registered frames (80%), plus minimum points. If this is not achieved training will not start. Training time and GPU fit are also unmeasured for this video. No process interruption or input change.


## 2026-11-04: GPU availability for MP4 COLMAP mapping

User asked if the current reconstruction can use the GPU. Read-only check: active process 8260 still present; installed pycolmap 4.2.1 reports `has_cuda=False`, and incremental options expose `ba_use_gpu=False`. Script explicitly uses CPU for extraction/matching. GTX 1660 SUPER detected (6% utilization at the instant checked). COLMAP documentation describes GPU bundle adjustment when supported; actual compatibility, performance and memory requirements for a separate CUDA-enabled build on this device were not tested. Running process cannot be changed into a different installed build in place; current unsaved mapping progress must not be assumed resumable. No process or settings changed.


## 2026-11-04: Why the 15 FPS DB3 COLMAP stage was faster

Audited `results/realsense_15fps_experiment/METHODS.md`, `scripts/experiment-realsense-15fps.py`, and success markers. The 15 FPS recording selected 5,204 frames, but SfM anchors were rebuilt on only the 694 retained 2 FPS baseline frames using cached baseline SIFT observations/matches; COLMAP was not asked to map all 5,204. Saved stage elapsed times: feature extraction 207.42 s, pair matching 277.30 s, SfM anchors 61.06 s, final RGB-D optimization 303.54 s; these are separate stages and do not equal the full campaign runtime. High-rate pose coverage came from calibrated RGB-D constraints, odometry, bridges and sparse pose optimization, resulting in 5,201/5,204 cameras. The present MP4 has 20,090 frames and no sensor depth and attempts RGB-only incremental mapping on all selected frames. This explains the methodology difference, not a controlled runtime benchmark or a guarantee of what can be reconstructed from the MP4. No running job changed.

## 2026-11-07: LaTeX research progress report

Converted `docs/RESEARCH_PROGRESS_REPORT_20261106.md` into the standalone `report/progress_report/research_progress.tex`, preserving the CO4029 cover, institution, lecturer, class and team fields from `report/samplereport.tex`, and listing only Nguyen Vu Long. Bundled the supplied HCMUT logo and added build instructions. The report connects each improvement to its method and evaluation, covers the historical and repeated-seed A04 comparisons, RealSense recovery, courtyard and viewer trials, the room calibration and pose gate, and the MP4 workflow. The interval figure uses saved paired-seed statistics, and the historical table includes peak tensor allocation from the experiment record. The report distinguishes image-quality gains from physical geometry validation and preserves failed and inconclusive trials. Evidence cutoff is 4 November 2026. No experiment was rerun.

Native Windows pdfLaTeX was absent. Installed the required TeX Live packages in Ubuntu WSL and used two direct pdfLaTeX passes from the report folder after a nested-shell command lost its loop variable in the output filename. Fixed a duplicated cover page anchor, a table column's line spacing, and the references link target. The final build completed with no LaTeX warnings or overfull or underfull boxes. Rendered and inspected nine A4 pages with Poppler, checked sole authorship, page numbers, all seven PDF outline destinations, selected result values, 23 bibliography entries, and 20 repository evidence paths. Saved the verified PDF as `report/progress_report/research_progress.pdf` and removed scratch outputs.

## 2026-11-07: LaTeX references and table captions

Removed in-text citations and bibliography entries for repository Markdown evidence from the report and retained six external references. Moved the four numbered table captions below their tables and shortened them. Preserved comparison details in the surrounding prose and verified all numeric table values against the previous PDF. Moving the first caption initially pushed its table onto a separate page, resolved by placing the run setup in the results paragraph. Adjusted bibliography alignment to eliminate an underfull line. Two pdfLaTeX passes produced a clean nine-page PDF. Checked the caption positions, reference links and rendered layout before replacing the report PDF. No experiments were rerun.

## 2026-11-07: Expanded densification explanation

The original section compressed the growth schedule, candidate ranking, and split operation into one page. Expanded it into two pages organized around how many Gaussians to add, which parents qualify and rank highest, how Long-Axis Split places and initializes children, and why the historical A04 Room count ended below the cap. Added a worked budget example, a pixel-level RGB error attribution example, and a schematic of the symmetric split. Verified the schedule, ranking, observation discount, and child parameters against `gsplat/strategy/budget.py` and `gsplat/strategy/ops.py`. Clarified that the 29.31% diagnostic compared pre- and post-split renders before any recovery updates. The final A04 event reconciles as 593,651 live plus 4,598 additions minus 260 pruned, yielding 597,989 under the 700,000 cap.

Rebuilt the PDF with two pdfLaTeX passes. Verified ten pages, the rendered two-page section, the contents links, and unchanged historical table values and later section text apart from figure and page numbering. Removed build scratch outputs. No experiments were rerun.

## 2026-11-07: Dataset inventory and exploratory data analysis

Added a four-page dataset section ahead of the methods in the LaTeX report. It covers the four Mip-NeRF 360 indoor scenes (1,122 benchmark views), the one-eighth-resolution Room CCD-GS variant, the 48-view analytic-sphere fixture, three RealSense recordings and their pilot, native-resolution, broad, full-rate, pose, stairs and SDK-remap variants, the RGB-only workflow fixture, and the two sampling runs of the same user MP4. The audit distinguishes selected, registered, training, evaluation and withheld views, and avoids counting source reuse as a new capture. Benchmark sizes and sparse points come from local dataset manifests, pilot splits from the CCD-GS manifests, and RGB-D splits from registered image directories, dataset validation, split files and recorded camera sequences.

Read 844 image observations across 18 prepared RGB groups using up to 48 lexicographically ordered indices per group, sampled at evenly spaced integer positions from first to last. Converted to 8-bit grayscale, reduced long side to at most 384 using Pillow BOX, and calculated frame-mean luma, share of pixels at or above 250, and four-neighbor Laplacian variance on the interior pixels. The report records relevant p10/p50/p90 summaries. Kitchen was the brightest sampled benchmark and Bonsai had the widest luma spread. The analytic fixture has narrow luma spread and high-frequency checker edges at native resolution. The broader Oct. 29 reconstructions include darker frames than the pilot/native exports. The exhibit room had the highest median near-white share among the sampled RealSense inputs (4.47% baseline and 4.49% SDK remap). The near-white and Laplacian statistics are descriptive and reflect content and resampling.

Computed consecutive RGB gaps and available RGB/depth pairing and valid-source-depth quantiles from all frames in the capture manifests. Courtyard median valid depth was 42.9% at 3 FPS and 43.5% at 30 FPS, with 3.0% of full-rate frames below 20%. The exhibit room had 70.3% median and 40.7% p10 valid depth. For Oct. 29, sampled 48 filtered depth caches per rate from the corresponding rgbd/depth_all directories. Each sampled file held 12,000 retained points and the median per-frame depth was about 0.82 m at each rate. Those point counts cannot recover valid-pixel coverage. The 15 FPS split lists 4,743 training names, including three that lacked poses (frames 004853 through 004855). Intersecting with the 5,201 posed image files gives 4,740 usable training, 79 evaluation and 382 withheld frames. The 30k trainer sequence visited 4,740 distinct indices. The 2 FPS and 15 FPS broad runs averaged 54.8 and 6.3 updates per training view.

Recovered the real-video outcome from the final 4 November status.json files and checked each largest model's images.bin directly. At 2 FPS, 118/1,339 frames (8.8%) registered in the largest model. At 30 FPS, 4,072/20,090 (20.3%) did. Both failed the 80% gate before training, superseding the earlier pending status. Divided extracted frame IDs into ten consecutive nearly equal bins and counted registered IDs in each. The 2 FPS bins contain 0, 0, 0, 0, 0, 0, 19, 77, 22, 0 registered frames. The 30 FPS bins contain 0, 0, 0, 0, 0, 0, 286, 1,248, 950, 1,588. Both largest components begin around 7 min 39 s. The fixture's largest component covers frames 21--48 and passed only the diagnostic 50% threshold. Added the chart and updated the summary and MP4 results. No source data, split, model, or experimental run was changed.

Built the PDF with two pdfLaTeX passes. Checked all 14 rendered pages, eight outline destinations, six new table caption positions, seven external references, and selected EDA values. Verified the historical densification table against the previous PDF and removed build scratch outputs.

## 2026-11-07 — A05 direction and A04 eligibility audit

**Problem.** A04 uses RGB error and edge detail to rank parents only after they clear the original gradient and opacity gates. It was unclear how often the eligible pool constrained growth and whether raising the cap would address it.

**Finding.** Verified all 648 saved refinement events across the 12 completed A04 replication runs. Eligibility fell short of available scheduled slots at 117/120 events from steps 5000 through 5900. Room and Counter averaged 101,893 and 133,926 unused slots at finish. The first post-reset refinement accounted for 63.9–83.8% of each run's total pruning and coincided with the start of 3D scale pruning. Existing logs cannot separate the prune causes or establish the split utility of rejected parents.

**Proposed experiment.** Test a small residual-driven route into vacant A04 slots. Hold the split operator and budget fixed and compare short continuations with no extra splits and with an equal number of extra parents selected by relaxed gradient. The proposal records the analysis procedure, pilot gate, related work and follow-up baselines. No A05 training or code changes were made. [A05 audit and pilot plan](A05_RESEARCH_PLAN_20261107.md)

## 2026-11-07 — A05 implementation and three-scene evaluation

**Implementation.** Added opt-in residual and gradient fallback ranks to the budget strategy. Existing A04 growth takes priority. At steps 5000–5900, a fallback can select up to 512 opaque, below-threshold parents with positive attributed RGB error and contribution in at least three distinct training images. The existing scheduled target caps combined growth. The trainer passes the sampled image identities and records the new candidate and selection counts.

**Protocol adjustment.** The archived seed-42 A04 checkpoints store final weights. To obtain a matched-count intervention without training new late checkpoints, used a frozen-weight observation pass and paired 500-update fresh-Adam continuations at final weights. Ran a separate normal-trainer screen against the archived 7,000-update A04 controls to test repeated late growth. Selected Room and Counter for spare capacity and Kitchen for its nearly full cap.

**Validation.** Fixed the missing Windows CUDA environment by using the repository's research launcher. Budget and split tests passed 26/26. The Room continuation smoke exercised all four branches. Rechecked all 519 Counter and Kitchen image hashes against the original image manifest. [Implementation, protocols and results](A05_RESEARCH_RESULTS_20261107.md)

**Warm-start finding.** All three scenes completed. After 500 updates, residual selection was −0.01085, −0.01637 and −0.03395 dB versus the no-extra-split control on Room, Counter and Kitchen. It also trailed the equal-count gradient control on all three scenes. Most of the initial split disturbance recovered. The residual rule selected parents with median longest-axis standard deviations 3.9× to 9.1× those of the gradient control, an association measured by looking up the selected IDs in the source checkpoints.

**End-to-end finding.** All three normal-trainer runs completed 7,000 updates with matching initialization and camera sequence hashes against the archived A04 seed-42 controls. A05 changed PSNR by +0.00626 dB in Room, +0.05222 in Counter and +0.12737 in Kitchen. Counter and Kitchen also improved SSIM and LPIPS and passed the practical threshold. Room LPIPS worsened by 0.001002. A05 used 7,499, 5,810 and 4,261 additional live Gaussians, respectively. Treat the two scene gains as single-seed results at a common cap. The equal-count warm-start result did not support the residual ranking mechanism. [Results and limitations](A05_RESEARCH_RESULTS_20261107.md)

## 2026-11-07 — A05 alternative with absolute gradients

The user requested another approach after the first residual fallback. Reviewed AbsGS and the Recovery-Aware Pruning and multi-view update sections of Improving Densification. Selected the existing absolute-gradient renderer and strategy path, combined with A04's ranking and split operator, with the documented 0.0008 threshold. The hypothesis is that reducing pixel-gradient cancellation improves the eligible parent population. The residual fallback is disabled. Added a named native-trainer campaign variant and a CUDA test that exercises signed-gradient cancellation through both packed and dense budget-strategy paths. Fixed the three-scene seed-42 protocol and practical gate before training, with seed-43 confirmation if all three scenes pass. [Protocol and results](A05_ABSGRAD_RESULTS_20261107.md)

**Alternative validation.** All 28 budget and Long-Axis Split tests passed, including the packed and dense cancellation checks. Pytest emitted the existing unknown `env` config warning. The 60-update native smoke completed, added 5,530 Gaussians by Long-Axis Split, and verified a finite 118,157-Gaussian checkpoint under its 150k cap. The saved configuration confirms absolute gradients, threshold 0.0008 and fallback off. Started the frozen three-scene 7k campaign in `results/a05_absgrad_20261107` with source snapshots. Smoke rendering metrics are harness checks only.

**Room seed-42 result.** The absolute-gradient candidate completed 7,000 updates with matching initialization and training-camera hashes. PSNR 30.73701 versus A04 30.60068 (+0.13633 dB), SSIM +0.003950, LPIPS -0.010731, and 452,485 versus 597,975 Gaussians (-24.33%). It also exceeds the first residual A05's Room quality metrics. The practical gate passed. Counter and Kitchen remain pending, so this is a single-scene, single-seed finding.

**Counter seed-42 result.** The candidate completed with matching initialization and training-camera hashes. PSNR 27.83983 versus A04 27.74928 (+0.09056 dB), SSIM -0.000634, LPIPS -0.001025, and 381,228 versus 563,073 Gaussians (-32.30%). The SSIM decline is within the prospectively specified 0.001 tolerance, so the practical gate passed. The original residual A05 had slightly higher SSIM, while this alternative has higher PSNR and fewer Gaussians. Kitchen remains pending.

**Kitchen seed-42 result and confirmation decision.** Kitchen completed with PSNR 30.21228 versus A04 29.98970 (+0.22258 dB), SSIM -0.000247, LPIPS -0.000454, and 502,536 versus 695,179 Gaussians (-27.71%). Initialization and camera sequence hashes match. All three scenes passed the frozen practical gate. Started the predeclared seed-43 confirmation with the same 0.0008 threshold, cap and recipe, using archived A04 seed-43 controls. Seed-42 image and artifact audit is running separately.

**Seed-42 audit.** All 104 final evaluation target PNG halves are byte-for-byte identical to the paired A04 renders. Quantized PSNR improved on 30/39 Room, 23/30 Counter and 21/35 Kitchen views. All final checkpoint hashes, PLY counts, saved configuration flags and initial metrics passed the audit. All nine current training source hashes still match the campaign snapshot. Peak tensor allocation fell from 1.080 to 0.971 GiB in Room, 1.090 to 0.945 in Counter and 1.229 to 1.077 in Kitchen. Observed training timers were similar. Added the repeatable artifact/image audit at `scripts/audit_a05.py` and saved results alongside the training summaries.

**Configuration comparability.** Compared every shared saved configuration field for all seed-42 pairs. The only differences are the output directory, `strategy.absgrad` and `strategy.grow_grad2d`. New fields are the disabled fallback configuration, unused sensor-depth input, and explicit worker count 4. The archived trainer hard-coded the same four training workers and one evaluation worker. Added these checks to the artifact audit. This addresses drift between the archived A04 controls and current trainer configuration.

**Room seed-43 confirmation.** PSNR 30.81184 versus paired A04 30.72016 (+0.09168 dB), SSIM +0.003616, LPIPS -0.010074, and 442,656 versus 593,800 Gaussians (-25.45%). The same configuration passed the practical gate on Room in both seeds. Initialization and complete camera-sequence hashes matched. Counter and Kitchen confirmation runs remain pending.

**Seed-43 completion.** Counter gained +0.04191 dB PSNR, with SSIM -0.000633, LPIPS -0.000017, and 384,975 versus 571,091 Gaussians (-32.59%). It missed the predefined +0.05 dB gate. Kitchen gained +0.25887 dB, with SSIM +0.000158, LPIPS -0.000131, and 508,135 versus 699,121 Gaussians (-27.32%), passing the gate. Across both seeds all six PSNR comparisons improved and all six counts fell. Five of six passed the gate. The seed-43 audit verified configuration, source snapshots, initialization, camera sequences, finite checkpoint hashes, PLY counts and 104 identical target images. Quantized PSNR improved on 27/39 Room, 20/30 Counter and 24/35 Kitchen views. The exposed evaluation views and two seeds limit generalization. Absolute gradients and the threshold were changed together. [Full results](A05_ABSGRAD_RESULTS_20261107.md)

## 2026-11-09 — School schedule progress audit

### Follow-up: close Weeks 1–10 gaps

**Accepted execution and decision.** Both 7k prior runs completed in `results/school_completion_replay_20261109`, matching archived initialization and all camera indices. Baseline prior PSNR +0.06141 dB, SSIM +0.000380, LPIPS +0.000582; reduced prior PSNR -0.00050 dB, SSIM +0.000175, LPIPS -0.000603. Both rendering guardrails passed. Identical-support five-view curvature fell 4.90% and 3.06%, respectively. Mesh topology was mixed: reduced+prior had 1,216 components versus 1,162 and 17,700 boundary edges versus 16,807. Retain A05-Abs without the prior as the main candidate; no general geometry improvement is established.

**Extraction failure and correction.** The local Open3D 0.20 scalable TSDF produced zero voxels/triangles even on a known z=1 plane. Uniform TSDF recovered that plane and passed a new regression check. All four trained arms were extracted with identical bounded uniform settings and separately captured extraction source. Accepted raw PLY/OBJ files are in `meshes_uniform/`, with 114,598–119,987 triangles. All five-view sheets were inspected: major room structures are recognizable, but plant/thin furniture fragments, holes and merged tabletop detail persist. This closes the initial extraction trial, not mesh cleanup or accuracy validation. No sensor points were fused.

**Deliverables and verification.** Added the school project plan, seven-paper/extraction matrix, dataset QA, supervisor briefing and [completion report](SCHOOL_COMPLETION_RESULTS_20261109.md). 31 relevant training/loss tests plus one independent plane test passed. The supervisor meeting is explicitly pending per the user. The accepted output audit checks finite checkpoints, current training source hashes, matching initial/camera hashes, 39 identical evaluation targets per new arm, configuration differences, and finite mesh save/reload. Work was performed on 9 November to catch up through 8 November; it is not backdated.

**Worker-count pairing failure and solution.** The first full baseline+surface run finished with finite checkpoint and PSNR 30.00757, but the prospective hash check rejected pairing. Initialization matched B00 exactly; the first 272 camera indices matched, then the second epoch diverged. Zero-worker DataLoader recreation advances its seeded generator differently from the archived persistent-worker loader. Preserved `results/school_completion_20261109` as an unmatched diagnostic; its metrics must not be presented as a controlled surface effect. Added an optional saved-camera-sequence replay to the trainer and restarted in `results/school_completion_replay_20261109`, explicitly feeding all 7,000 archived training indices. This keeps the memory benefit of zero workers while making the comparison checkable. Final hashes remain the acceptance check.

The user authorized completing partial/missing work through 8 November. They supplied the school title: “Improving 3D Gaussian Splatting for efficient and geometrically consistent 3D reconstruction, followed by conversion into editable polygon meshes” and confirmed that the supervisor presentation has not occurred. Presentation/approval cannot be claimed from repository work.

Added a consolidated school project definition, pipeline diagram, evaluation criteria, dataset/capture plan and seven-paper literature/extraction matrix. Added read-only data QA and reviewed a candidate contact sheet; all four public sets and the personal room had no exact duplicate files, finite points/poses and valid rotation structure. Low sharpness scores were affected by scene texture; frozen splits were preserved.

Implemented an opt-in expected log-depth curvature prior with detached image-edge/opacity weights, and training-camera rendered-depth TSDF extraction from optimized Gaussians. Three new loss checks and existing budget/split tests passed (31 total). Four 60-update smoke arms completed; smoke is harness evidence only, and the final nonfinite-depth safeguard and zero-worker campaign setting supersede early smoke source snapshots. Started a frozen 7k Room 2x2 ablation with two archived no-prior controls and two fresh prior arms. New runs must match initialization and camera hashes; raw meshes use identical settings. Full results and limitations will be recorded separately.

Compared the supplied 14-week workbook with current source, reports and saved results. Week 10 ended 8 November; final deadline is 6 December. Baseline and Gaussian reduction research are substantially complete; surface consistency, optimized-Gaussian-to-mesh conversion, mesh editing validation and final packaging remain partial or missing. Experimental RGB-D ball-pivot meshes do not establish Gaussian-to-mesh conversion. A05 absolute gradients improved PSNR/count in six scene-seed comparisons, with five practical passes; CCD's 0/36 gate is a completed negative finding. No experiments rerun and workbook unchanged. Full evidence, milestone mapping, limitations and proposed next actions: [schedule progress audit](SCHEDULE_PROGRESS_AUDIT_20261109.md).

### Week 11 next-step review

**Finding.** The accepted main candidate is A05-Abs with `surface_lambda=0`. Gaussian-to-mesh extraction now produces inspectable raw PLY/OBJ assets, but visual review found holes, fragments and merged thin details; the curvature prior slightly improved smoothness proxies while worsening some topology counts. No independent geometric accuracy or design-software edit roundtrip is established. The eight-slide supervisor briefing exists, but the presentation and feedback remain outstanding. This status review used saved reports and artifacts only; no training, tests or mesh edits were run.

**Recommended solution.** Freeze the A05-Abs configuration and preserve its raw mesh. For Week 11, create a separately versioned cleaned derivative with a short, recorded cleanup recipe; import it in a mesh editor, make a simple geometry edit, save, close and reopen it, and record the tool/version plus before/after topology and screenshots. Keep cleanup claims to usability and visible defects unless independently validated geometry becomes available. Present the existing briefing and record supervisor decisions before expanding the research scope. Then use Week 12 for a matched baseline-versus-candidate end-to-end comparison on selected public and personal scenes, reporting render quality, Gaussian count, extraction/training time and mesh usability with calibration/reference limitations explicit.

### LaTeX progress report synchronized through 9 November

**Problem and evidence.** The reader-facing LaTeX report still declared a 4 November cutoff and used the earlier densification/RealSense subtitle. It omitted the completed residual A05 screen, six A05-Abs scene/seed pairs, the accepted Room reduction x surface-prior study, Gaussian-derived uniform-TSDF meshes and school catch-up. Its next-step table no longer matched the current school schedule. Read the A05 result documents, current completion results, accepted campaign `REPORT.md`, initial schedule audit with its follow-up, current loss implementation and report source. The root `report.tex` is a separate historical CCD-GS manuscript; the active progress report is `report/progress_report/research_progress.tex`.

**Solution.** Updated the active report cover to the user-confirmed full school title and 9 November evidence date. Preserved earlier A04/RealSense/video results. Added the negative equal-count residual finding, all six A05-Abs comparisons and the five-of-six practical gate, paired Room rendering and raw-mesh tables, curvature-proxy limitations, worker/camera-replay and TSDF-backend failures/corrections, accepted evidence locations, current candidate decision, technical Weeks 1–10 catch-up and remaining Weeks 11–14 milestones through 6 December. Supervisor presentation/feedback, geometry accuracy, cleanup/edit roundtrip and final packaging remain unestablished. Updated the report README and rebuilt its adjacent PDF. This revision uses saved research evidence; no training, extraction or research test suite was rerun.

**Layout finding and correction.** Expanded contents/summary initially split the summary from its table; A05 initially spilled a single line to an otherwise empty page. Gave the summary its own page, shortened the A05 interpretation without removing findings, and separated the surface experiment from extraction at a subsection boundary.

**Verification.** Compiled with Ubuntu WSL pdfLaTeX until cross-references and page counts stabilized. Final build is 19 physical pages (cover plus 18 numbered pages), with no LaTeX warnings, undefined citations/references or overfull/underfull boxes. Rendered pages with Poppler; inspected the complete initial render as contact sheets and the final changed summary/A05/surface/extraction pages individually. Tables and text are legible and contained within margins. Existing test counts in the report are attributed to the saved campaigns, not new tests. Refreshed `report/progress_report/research_progress.pdf`; unrelated modified/untracked research work was preserved.

### Local automatic LaTeX compilation guidance

**Finding/problem.** For the user's Overleaf-style local editing question, checked Ubuntu WSL commands and the Windows VS Code extension inventory. `pdflatex` exists at `/usr/bin/pdflatex`; `latexmk` was not found. The Windows inventory did not list Microsoft WSL or LaTeX Workshop. This does not establish the inventory of any remote VS Code extension host.

**Solution documented.** Added VS Code + WSL + LaTeX Workshop instructions to the report README: install `latexmk` in Ubuntu, open the report directory with `code .`, install the LaTeX extension in WSL, enable build-on-save and use the PDF preview. Verified the workflow/settings against the official VS Code WSL and LaTeX Workshop compile documentation. This turn did not install packages/extensions or change editor settings, and did not claim an interactive editor test.

### Automatic compilation setup executed

**Authorization and installation.** The user then requested completing the full setup. Installed Microsoft WSL extension 1.104.7 and LaTeX Workshop 10.20.0 in Windows VS Code; installed Ubuntu `latexmk` 4.87 using the WSL root user after noninteractive sudo required authentication. Apt installed one new package and did not upgrade the existing environment. Opened a dedicated `progress_report [WSL: Ubuntu]` editor window.

**Remote-extension finding and solution.** Windows `code --remote wsl+Ubuntu --install-extension` reported the already installed Windows extension without placing it in the new Ubuntu extension host. Explicitly used the initialized VS Code server's `bin/code-server --install-extension` command; the remote inventory then verified `james-yu.latex-workshop@10.20.0`.

**Configuration and validation.** Added report-scoped `.vscode/settings.json` and extension recommendations: one-second auto-save, build-on-file-change, PDF tab, explicit latexmk PDF recipe, SyncTeX and `build/` output. Ran that recipe successfully: 19-page PDF, SyncTeX output, all targets up to date. No report content or research experiment changed. Updated the README with the installed state and reopen command.

**Outstanding UI prerequisite.** The new VS Code WSL window is in Restricted Mode. Computer Use guidance prohibits changing in-app security settings, so the user was asked to trust this report folder themselves. The source is open. Automatic editor-triggered build and the live PDF preview must be verified after the user enables trust; successful CLI compilation alone is not that verification.

**Resolved and verified.** The user enabled workspace trust; observed Restricted Mode disappear and LaTeX Workshop controls become available. Opened the internal PDF viewer beside the source. Initial `onFileChange` did not build after Windows-side source writes under `/mnt/d`, and an editor save also did not start a build in that mode. A nonsemantic setup comment written externally also caused a stale-buffer save conflict during the whitespace check; reloaded the saved source before continuing. Changed the recipe trigger to `onSave`, reloaded the VS Code window to apply the configuration, and tested a whitespace-only editor edit with one-second auto-save. Extension log at `exthost2/output_logging_20261109T051820/1-LaTeX Workshop.log` records automatic build initiation, successful latexmk completion, zero parsed messages, PDF refresh and viewer-loaded confirmation. Final automatic build completed at 12:19:03 Asia/Saigon and produced the same 19-page report. Source and live PDF remain open side by side. README reflects the final verified settings, with `build/research_progress.pdf` as the live output. Research content was unchanged.

### Compile failure after cover edits and stale PDF diagnosis

**Problem/finding.** The user reported compilation ran without visible changes. Current source changed the lecturer to Nguyen Tuan Khoi and removed the ClassName/TeamName definitions while retaining their cover rows. The VS Code compiler log reports an undefined control sequence at ClassName; a separate continuous latexmk process also generated an errored intermediate PDF. The user had opened the adjacent `research_progress.pdf`, which was still the 12:09 copy, rather than the live `build/research_progress.pdf`.

**Correction.** Removed the two orphaned cover rows, preserving the user's lecturer and other edits. Stopped only the identified duplicate latexmk watcher (PID 3223) to avoid two builders writing the same auxiliary files; VS Code auto-save/build remains configured. Rebuilt with latexmk and halt-on-error successfully: 19 pages, no LaTeX warnings/errors. Rendered and inspected the corrected cover. Refreshed the adjacent distribution PDF once and opened the actual `build/` output in VS Code. Future live updates remain in `build/research_progress.pdf`; the adjacent copy is not automatically published by this recipe. No experiments changed.

### Remove report date markers, preserve Table 14

**Request/problem.** The user requested removing time marks throughout the report except Table 14. Calendar dates appeared on the cover, progress narrative, capture labels, subsection headings and visible evidence filenames. Week-based progress markers also appeared outside the schedule table.

**Solution.** Removed the report date/cutoff and calendar/week progress references, renamed capture-date labels to the indoor capture, and removed the date from the user-video display name. Replaced date-stamped displayed evidence paths with date-free hyperlink labels while preserving exact original file targets. Renamed the remaining-schedule heading without altering Table 14. Scientific durations, frame rates, optimization steps and bibliography publication years remain as measurements/reference metadata. Preserved the user's lecturer-name and cover edits.

**Validation.** Table 14's complete LaTeX environment was checked unchanged using SHA256 `5d65da3394df74f25d9fb5ca3093d9c7c9b70b1472f8a145c5414658334c1742`. A first build exposed an unbreakable evidence-link label; replaced it with readable prose and rebuilt. Final latexmk completed with 19 pages, no warnings/errors or overfull/underfull boxes. PDF text search found calendar/week markers only on physical page 18 containing Table 14. Rendered and visually inspected all pages as contact sheets. Refreshed both the live build PDF and adjacent reader copy. No research experiments were rerun.

## 2026-11-09: MP4 pipeline explanation from persisted runs

Problem: explain why the MP4 workflow can execute but the actual phone video did not reach Gaussian training. Rechecked both data/mp4_VID20261104003648_*/status.json files, preparation code, docs/MP4_TO_3DGS.md and the report's temporal-coverage analysis.

Finding: extraction and RGB-only COLMAP mapping ran. Largest-model registration was 118/1,339 (8.8%) at 2 FPS and 4,072/20,090 (20.3%) at 30 FPS. Both persisted states are failed because the required largest-model registration fraction is 80%; training was gated before launch. Multiple separate models were recovered. Reported temporal analysis places both largest components from about 7 min 39 s onward. This establishes incomplete reconstruction, not a uniquely identified capture defect. Blur, weak texture, rotation/insufficient parallax, dynamic content and changing intrinsics remain hypotheses, not verified root causes for this clip.

Explanation and proposed solution: MP4 supplies images; COLMAP estimates camera intrinsics, poses and sparse points for the existing RGB-only trainer. The earlier fixture verified the downstream path only after a diagnostic gate reduction (28/48 images, 50% threshold). Diagnose feature-track/pose continuity at failed transitions and shared-camera assumptions on the existing footage before a new full run. More Gaussian steps or a larger cap cannot fix missing poses; lowering the gate only admits a partial scene. No experiment, threshold or source data changed during this explanation.


## 2026-11-10 — DB3 reconstruction R0/R1 execution started

**Request and scope.** Execute R0 and R1 from `PIPELINE_MESH_DB3_NEXT_STEPS_20261110.md` for `20261030_134523.db3`, aiming at a better navigable 3D scene. Preserve existing raw inputs, 120k model and unrelated work. Mesh production, general optimization, R2 pose recovery and R3 training are outside this first execution.

**Problem/finding.** Cleaner and bridge duplicated the old inverse-Brown remap; the isolated SDK bridge monkey-patched a function. Pinned librealsense 2.59.1 `src/rs.cpp` also distinguishes ordinary Brown tangential evaluation from its inverse-Brown projection after radial scaling. The historical candidate is a close approximation, not the exact pinned formula. Source: https://github.com/realsenseai/librealsense/blob/v2.59.1/src/rs.cpp#L4012-L4164 .

**Implementation underway.** Added versioned shared `scripts/realsense_projection.py`; legacy mode reproduces historical inputs, SDK mode implements the pinned projection/deprojection. Cleaner records explicit version, refuses silently reusing legacy outputs as SDK, bridge reads calibration version, unknown versions fail. Shared depth alignment retains nearest-Z visibility and missing/collision masks, with no filling. Numerical oracle and regression controls are in `tests/test_realsense_projection.py`.

**Experiment controls frozen before review.** `scripts/run-db3-r0-r1.py freeze` hashes the full raw source and saves a hash-verified baseline checkpoint/camera/config/render package in `results/db3_r0_r1_20261110/`. Preserves the original 417/60 appearance split; it is exposed development evaluation. Prospective geometry checks use 1–2, 48–49 and 98–99 seconds, final blocks 23–25, 88–90 and 145–147 seconds, each with 0.5 s buffers. Exclude reserved native IDs from future fits/training; historical poses already used old observations. No independent historical test claim. R1 does not inspect final-block RGB.

**Acceptance/stop.** Require <0.1 raw RGB pixel SDK projection/roundtrip discrepancy on recorded-model image grid/borders/random points at 0.3/0.7/2/5.9 m; depth/extrinsics errors <1e-5 m. Regenerate all 477 selected images/depth in a fresh namespace with fixed original IDs/K/crop, exact historical depth-array comparison and explicit split. Physical alignment needs manually checked static boundaries with uncertainty; missing/glass/far regions remain unresolved. Timing is diagnostic until measured evidence supports a motion model; no guessed pixel/time shift. Stop promotion if a gate fails. Final results will be appended below.

**Execution findings and corrections.** Full raw SHA256 matched the saved 60e3f253… source; checkpoint and baseline cameras/config/render artifacts were copied and hash-verified. The first replacement preparation stopped at frame 49: testing border membership after float32 conversion removed one projected sample (282,687 -> 282,686) and changed the seeded 16k subsample. Restored double-precision membership gating before float32 grouping; preserved rejected output as `data/db3_r1_sdk_20261110_rejected_border_attempt/` with its failure manifest. The successful 477-frame run reproduced every baseline depth point/depth array exactly. A later synthetic edge audit also guards float32 rounding to an out-of-bounds pixel; no clamping or row wrapping is allowed.

**Environment issues.** First pytest invocation had 12 passes and one setup error because the existing system-temp pytest directory was inaccessible; reran using a run-local `--basetemp`. First baseline-render invocation omitted `.venv/Scripts` from PATH, so Ninja was undiscoverable; corrected launch PATH and successfully reloaded/rendered the preserved 1,063,961-splat checkpoint at seven source-ID anchors. No packages were installed or upgraded.

**Numerical result.** SDK 2.59.1 wheel SHA256 045dfae0… pinned; 12,100 samples per recorded stream over a border-inclusive grid/random set and four ranges. RGB maximum SDK projection discrepancy 0.000204627 raw px, maximum deprojection error 1.38738e-6 m; depth maximum projection discrepancy 0.000087959 px. Cropped K/output map, distortion IDs/coefficient order, depth scale and column-major extrinsics checks passed. This is compatibility, not independent physical calibration.

**Timing finding.** Parsed all RGB/depth/gyro/accel headers and all metadata. DB/header timestamps coincide; their median paired gap is -7.24715 ms (depth minus RGB). Both image metadata streams declare Global Time; device readout stamps differ by -0.033 to -0.034 ms, while device exposure-midpoint stamps differ by -0.431 to -12.423 ms, median -3.292 ms. Depth exposure varies from 962 to 24,944 us; RGB Actual Exposure is absent. The same 4,764 nearest pairs are obtained using exposure midpoints. No global 7 ms shift or frame reassociation is justified. SDK metadata definitions: https://github.com/realsenseai/librealsense/blob/v2.59.1/include/librealsense2/h/rs_frame.h#L15-L37 . Keep timing uncertainty, especially for motion.

**Visual/local result.** Saved source/capture-versus-baseline renders and 16 old/new/depth review views, excluding prospective final blocks from visual review. Reviewed kiosk and opaque-box outer contours move toward their observed raw-depth transition intervals: median interval distances 7 -> 0 px and 8 -> 0 px over three correlated rows each. Zero is interval overlap, not zero error; bands span up to 9/14 px and contain holes. Head control remains within a broad band. Rails are broad/fragmented; far doorway has only 14.8% valid projected ROI depth, wide doorway 44.1%; glass and exterior remain unresolved. Precise physical alignment gate does not pass. These exposed single-analyst observations support a local candidate convention, not validated whole-scene geometry.

**Final verification and decision.** `scripts/verify-db3-r0-r1.py` completed in 57.49 s: full raw SHA256 unchanged; baseline originals/copies and all 477 RGB/depth hashes unchanged; candidate hashes, exact depth-array equality, positive finite unique pixel samples, masks and crop/K/IDs passed. Final-code reprojection reproduced both RGB variants and unfilled depth in all 16 review views. Relevant tests: 17 passed, one existing unknown-config `env` warning (4.15 s). Future selected-frame sets are 377 train, 17 geometry checks and 28 prospective final, with 60 exposed historical evaluation identities; train is disjoint from all reserved sets. Successful preparation took 93.28 s; peak RAM/VRAM was not instrumented and no efficiency claim is made. Output sizes: new inputs 446.5 MiB, run/baseline/evidence 326.2 MiB, preserved failed attempt 45.4 MiB.

R0 complete; R1 code/input/timing work complete with precise physical gate partial. Keep the 120k model as contextual baseline. The SDK images are a supported candidate for bounded R2 local opaque-submap work, not an accepted whole-room calibration or new 3D scene. Future R2 must enforce native and selected-frame reservations, rebuild features/real tracks/poses and finish independent local geometry checks before any R3 Gaussian pilot. No mesh, training, commit/push or dependency upgrade occurred. Full findings/problems/solutions/commands/limits: [R0/R1 report](DB3_R0_R1_RESULTS_20261110.md). Machine decision: `results/db3_r0_r1_20261110/phase_decision.json`; complete source snapshots and integrity evidence: `results/db3_r0_r1_20261110/verification.json`.

## 2026-11-10 — DB3 R2/R3 bounded local reconstruction campaign

**Request and starting gate.** Execute R2/R3 and audit R0 through R3 cleanly. Current saved R0 integrity passed; R1's numerical/input/timestamp work passed but precise physical alignment remains partial. Do not relabel that unresolved physical gate complete. Preserve every baseline, raw recording, R1 artifact and unrelated workspace modification. Fresh experiment: `results/db3_r2_r3_20261110/`; fresh local inputs: `data/db3_r2_local_20261110/`. No full-recording 120k run, mesh or global merge is authorized by a successful local consistency check alone.

**Protocol frozen before fitting.** `protocol.json` declares matte 6–14 s and difficult exhibit/rails 64–72 s, 3 FPS versus bounded approximately 10 FPS tracking, fixed intrinsics, no old poses/features/matches, native-source reservation enforcement and historical appearance identities excluded from all mapping/depth fitting/training. Both windows avoid the prospective reserved blocks. A deterministic 70-pixel spatial-tile class is removed from COLMAP keypoints/descriptors before matching; separate OpenCV SIFT observations in that class supply prospective checks. Descriptor matches undergo a separately split RGB fundamental-matrix verification, never candidate-pose residual filtering. This is conditional feature/depth consistency on exposed development data, not external accuracy or independent physical annotations. The native rate is tracking only; Gaussian training keeps original selected training identities. RGB-D derived fixed geometry means the depth-off arm is appearance-only optimization, not an RGB-only reconstruction.

**Input and real-track result.** Read-only raw extraction prepared 186 RGB-D observations in 83.87 s with IDs, hashes, nearest recorded depth pairs, quality scores, source masks and no filling. Fresh COLMAP 4.2.1 fixed-calibration sequential matching/mapping produces real multi-view landmark tracks. Matte 3 FPS recovered 16/21 requested fitting frames; approximately 10 FPS recovered 92/92. Exhibit 3 FPS recovered 21/21; approximately 10 FPS recovered 88/88. A robust training-track median supplies metric scale; bounded sparse joint BA explicitly combines whitened pixel and camera-Z depth factors, fixing the first camera and calibration. Depth sigma is a declared conservative range/local-spread model, not an independently measured covariance. No hard-coded bridge inlier proxies or old graph edges enter these local fits.

**Implementation failures/fixes.** PyCOLMAP 4.2.1 exposes landmark IDs as reconstruction-map keys, not `Point3D.point3D_id`. Initial metric BA stopped after saving the fresh RGB metric model; corrected key access and resumed from that saved model without rebuilding or altering it. Logs preserve both failures. Repeated full-image morphology inside per-landmark depth lookup was needlessly costly; batched each image's lookups without changing the sampling rule. Evaluation localization initially repeated descriptors per observation; subsequent runs keep one descriptor per landmark. All four BA runs reached their bounded 45-evaluation budget (status 0), so fitted RMS is explicitly not a convergence or acceptance certificate. Separate check gates decide local usability.

**Initial withheld checks and decision.** Matte 3 FPS fails coverage (76.2%) despite small check residuals. Matte approximately 10 FPS passes local consistency: 386 common stable check matches over ten pairs, median/p90 separation 0.00834/0.03488 m, reprojection 0.468/1.349 px, eight low-error control pairs with no median worsening over 0.01 m. Exhibit 3 FPS passes: 705 matches/25 pairs, separation 0.01611/0.06596 m and reprojection 0.505/1.594 px; all 21 control pairs within the no-worsening allowance. Neither is a meaningful >0.01 m separation improvement over the original baseline. These gates support bounded local Gaussian experiments; they do not establish accurate global placement, whole-scene calibration or a completed R3 improvement.

**D5 implementation and controls.** Added optional per-observation sensor confidence transport with [0,1]/shape/finite validation, an opt-in normalized confidence-weighted disparity L1, and a differentiable zero for empty depth frames. The historical unweighted loss remains the default. Invalid predictions retain the historical zero-disparity penalty; invalid/absent sensor observations carry no confidence. `tests/test_sensor_depth_confidence.py`: 4 passed, one pre-existing pytest unknown `env` warning. Controls verify equivalence to unweighted loss for all-one weights, weight normalization, invalid/empty finite gradients, world-unit scaling and loader transport. No package upgrade occurred.

**R3 next gate.** On a passing local submap only, build a support-aware cloud from original fitting frames with at least two temporally separated agreeing views, range/stability/incidence/motion evidence and explicit missing/occluded/conflict flags. Keep uncertain raw observations in evidence; no unsupported point is declared physically false. Freeze identical cloud, camera replay, training IDs, sampling, seed and strategy across off/current-unweighted/confidence policies; run bounded 7k-update pilots and evaluate fixed localized historical development views plus physical depth/duplicate-surface checks. Final outcomes and clean phase audit follow below after execution.

### R2 completion and R3 development findings

**Higher-rate difficult submap.** Exhibit approximately 10 FPS recovered 88/88 cameras, retained 1,800 real landmarks with mean track length 28.61, and passed 705 withheld check observations/25 pairs: separation median/p90 0.01599/0.06487 m; reprojection 0.495/1.597 px; 21 low-error controls within 0.01 m worsening. Saved per-pair spread/planarity, real observations, all fitted/check residuals and local poses. The first-source-camera-aligned 3 FPS/10 FPS/contextual-baseline trajectory overlay is `local_trajectory_overlay.png`; alignment uses no check structures and is not absolute trajectory error. Visual check sheets mainly follow static graphics, exhibit and framing; people/glare remain in backgrounds. No claim every source pixel or individual correspondence is confirmed static.

**D3 failed-transition and revisit experiment.** Fresh native observations at historical boundaries 98, 345, 373, 451 and ordinary controls 30/204 were tested at selected endpoints, approximately 10 FPS and native approximately 30 FPS. All four failure-boundary routes remain unaccepted under the declared fitting/check support/coverage/conditioning gates; many higher-rate links fit but have fewer than 20 independent stable check matches. This is insufficient validation, not proof that those poses are wrong or recovery impossible. The exhibit control 204 passes all three recipes with endpoint median/p90 check separation approximately 0.017–0.019/0.075–0.078 m. Real feature-index tracks and per-edge anisotropic information are saved. Fresh distributed revisit candidates yield no globally accepted loop; two candidates involving reserved frame 266 were excluded before loading. No forced global merge or graph-connectivity accuracy claim.

**Uncertainty correction.** The initial diagnostic mixed a right-rotation Jacobian with an unpropagated covariance sum. Preserved it in `transitions_revisits/`; corrected the model to consistent left-SE(3) perturbations, verified its Jacobian against finite differences, propagated edge covariance through SE(3) adjoints, and multiplied the propagated sum by chain length as a conservative correlated-error bound. Corrected report: `transitions_revisits_v2/report.json`. These are assumed pixel/range uncertainty models, not calibrated sensor covariance. Accepted/rejected status did not justify global promotion. **Concrete D8 execution:** independently optimized, checked local submaps with unresolved inter-submap placement; preserve the whole-room 120k model only as context.

**Support-aware initialization.** Both local pilots freeze 90,000 sensor-cloud representatives from 1 cm voxels, selected by at least two temporally separated supporting fitting views, no observed free-space conflict, local stability/incidence and conservative motion/clipping masks. Missing and occluded observations remain unknown. Shared depth samples (up to 4,096 fitting-tile points/frame) carry evidence weights and categorical flags. Raw uncertain observations are retained in `dataset/evidence/` and original DB3. Original selected Gaussian training identities: 21 per region; three historical exposed evaluation identities each localized against frozen training landmarks using non-check RGB features, with no evaluation depth/map/training-pose refinement. Geometry, calibration, initialization and full 7k camera replay are identical within each three-arm study. The fused Gaussian initializer intentionally has empty tracks and is separate from the preserved real-track geometry model; it must never be sent to BA as though it contains landmark observations.

**Numerical failure found by actual checkpoint reload.** All first matte jobs reported 7,000 updates and finite image metrics, but the current unweighted depth checkpoint contained 1,487 Gaussians with NaN geometry parameters (4,461 mean components, 1,487 opacity logits, 5,948 quaternion components and 4,461 scale components). Rejected it and preserved its checkpoint/logs plus partial evaluation in `matte/pilot/current/` and `evaluation_rejected_nonfinite_current/`. Historical `depth_l1_loss` used `torch.where(pred>0,1/pred,0)`: both branches are evaluated and zero-depth reciprocals poison backward despite a finite forward value. The confidence implementation already guarded its reciprocal. Fixed unweighted loss by masking reciprocal inputs first; finite-positive loss values and the historical zero-disparity penalty remain unchanged. Added a regression with zero/negative/NaN rendered depths; 230 loss-related tests passed (one existing unknown pytest `env` warning). `current_safe/` repeats the exact initialization, camera replay, seed, samples and 7k updates; all 179,787 final Gaussians are finite. Added a recorded-run final-parameter guard so future non-finite results stop with `experiment_failed.json` instead of a successful experiment manifest. The initial broader suite before this loss fix passed 244 tests.

**Matte paired result, final checkpoints/common raw observation.** Baseline 21.458 dB / 0.80205 SSIM / 0.39394 LPIPS; off 28.966 / 0.88738 / 0.18933; repaired current 29.014 / 0.88797 / 0.18751; confidence 28.750 / 0.88569 / 0.19522. Baseline old-rectified renders are mapped through the same raw RGB observations to the SDK image; pilots share the same SDK targets/intersection. Source identities stay fixed, but baseline uses historical camera estimates and pilots use frozen-map localizations: this is a **local pipeline comparison**, not isolated training-method attribution. No best-crop or changed-target PSNR claim. All final selected checkpoints were reloaded; initialization and camera-replay hashes match exactly across arms.

**Failed original gates and bounded follow-up, frozen before confirmation.** None of the matte arms improves every target-view median sensor-depth error by >=0.01 m. Confidence also fails the original noninferiority gate versus repaired current/off (mean PSNR is 0.264 dB below repaired current). Preserve both negative findings. Repaired current worsens the frame32 target median by 0.01108 m; off worsens additional controls. Confidence is the sole arm within 0.01 m worsening of **every** checked matte target/control, while its target-region RGB PSNR improves by 3.805, 1.623 and 3.944 dB across the three fixed views. Selected confidence for a narrower, explicitly different confirmation: target appearance improvement (>=1 dB in each ROI/view), target/control median depth consistency within 0.01 m of baseline, coverage within 0.02, global appearance noninferiority, and visual/short-offset review. This does not turn the failed depth-gain or confidence-policy gates into passes. `selected_policy_confirmation.json` records the decision before any exhibit Gaussian training; exhibit is exposed development confirmation, not an untouched scene/test set. All three exhibit arms are run as controls with the repaired common implementation; selection stays frozen.

### R3 confirmation failure and bounded D6 continuation

**Completed common 7k matrix and tests.** Six valid 7,000-update jobs (two local regions x three depth policies), plus the preserved invalid unweighted matte job and its repaired rerun, completed. Every selected final checkpoint was independently reloaded and checked finite; matching initialization/replay hashes were verified. The final relevant suite after the reciprocal repair passes 245 tests. The isolated saved reproduction has identical legacy/fixed forward loss 1.0, but legacy zero-rendered-depth gradient `nan` versus repaired gradient 0.0.

**Matte narrower local result.** The selected confidence candidate passes all nine coarse view/region target-appearance/depth-consistency/coverage checks and the captured/small-offset visual review. This is a local appearance improvement with checked sensor consistency, not the failed original universal depth-gain gate. Source-relative rectangles do not follow the same physical object through camera rotation: coarse labels are inspection patches, not semantic segmentation/metrology. To strengthen the difficult-region confirmation, manually reviewed original exhibit captures and froze source-specific standing/kneeling statue, opaque post, wall/backdrop and floor rectangles before any exhibit Gaussian metrics were reviewed. Missing/glass/exterior/background motion remain unresolved; the source images and rectangle overlays are saved.

**Exhibit 7k fails confirmation.** Baseline 24.670 dB / 0.82852 SSIM / 0.22563 LPIPS; off 24.333 / 0.84897 / 0.15547; repaired current 24.245 / 0.85089 / 0.15455; confidence 24.332 / 0.84864 / 0.15372. Some figure/framing details appear clearer and LPIPS/SSIM improve, but confidence loses 0.339 dB versus baseline and violates opaque-surface alpha coverage and several physical-structure checks. Coarse region coverage drops about 2.3–8.2 percentage points; actual source-structure median depth controls include backdrop worsening 0.364 m, floor worsening 0.126 m, and smaller wall/post/standing-statue regressions in another view. The far first-view statue appearance targets also fail. Do not promote the exhibit submap from attractive captured views or mean feature residuals. Original reports, failed gates and seven 7k artifacts remain unchanged.

**D6 reason to continue, hypothesis and solution under test.** Exhibit confidence PSNR still improves 23.241 -> 24.332 dB from 3.5k to 7k. Expected-depth normalization and RGB alone can permit partially transparent, high-color splats at measured opaque surfaces; this is a plausible contributor, not a confirmed unique cause. Froze one additional 15k candidate in `supported_15k_protocol.json`: same fixed geometry, cloud, training identities, original 7k replay prefix, seed, strategy/cap and refinement stop; depth lambda 0.05 and an optional supported-sensor alpha penalty 0.02. Alpha prior is applied only at positive-confidence valid sensor observations; unknown/missing/zero-weight content is not made opaque. This adds an assumption that supported sensor surfaces should be opaque; glass/reflection applicability remains unresolved. New loss unit controls verify normalized weighting and finite zero/empty/unknown gradients (six sensor-confidence tests passed). No default behavior changes: alpha penalty defaults to zero. The 15k longer-compute/changed-objective combination is a bounded exposed-development pipeline experiment, not isolated attribution or a fair policy-superiority comparison to 7k controls. Exact same frozen coarse and actual source-structure thresholds apply. The failed 7k result is not relabeled. Final follow-up outcome is appended after reload/evaluation.

### Final 15k outcome, pretraining support diagnosis and acceptance

**Completed finite follow-up.** The supported 15k candidate finished in 409.79 s, with 179,865 finite final Gaussians. Final post-update reload: 25.047 dB / 0.85088 SSIM / 0.14043 LPIPS, versus contextual baseline 24.670 / 0.82852 / 0.22563. The 7.5k midpoint was 24.669 dB. Original initialization hash and exact 7k replay prefix match; total valid study updates are 42,000 common-matrix plus 15,000 follow-up, with a separate rejected unsafe 7k job retained. Longer compute and stronger depth/alpha objective are confounded; no isolated-method superiority claim. Final relevant suite after the supported-alpha addition: **246 passed**, one existing unknown pytest `env` warning. Checked changed files with `git diff --check`; only the existing Windows line-ending advisory appears.

**Coarse positive finding versus actual rejection.** The 15k exhibit passes the original coarse pipeline depth/global-appearance gate and fixes most alpha-coverage losses. Coarse target median depth improvements are 0.1894, 0.0287 and 0.0179 m, and all coarse controls stay within the 0.01 m non-worsening allowance. Nevertheless, the stricter fixed first-view target RGB gain is -0.638 dB, not >=+1 dB. Actual source-structure rectangles pass 8/15 checks: first-view standing/kneeling target gains -0.262/-1.161 dB, first-view backdrop/floor median-depth worsening 0.0739/0.0701 m, middle floor +0.0466 m, late wall/post +0.0209/0.0182 m. These are sensor-consistency differences, not certified physical errors. **Reject exhibit promotion.** Strong global PSNR/LPIPS or coarse gates do not override independently frozen structures. Both old 7k failures and new 15k positive/negative results remain saved; no thresholds changed after seeing them.

**Pretraining geometry/support diagnosis.** Projected the frozen supported sensor initializer with frozen evaluation poses at the same manually reviewed source rectangles, before considering its trained Gaussian depths. `exhibit/pilot/initialization_structure_diagnosis.json` uses nearest-Z observed points within 2 px, no filling. The first backdrop has only ~1% initial support; several floor controls have ~21–22%. At higher-coverage figure/post/wall regions, conditional median source-depth differences already range from roughly 0.02 to 0.47 m. Nearest-point/occlusion/edge contamination and factory-depth uncertainty limit this diagnostic; it is not a physical reference or unique pose/calibration diagnosis. It does show that feature-tile consistency did not validate every physical structure, and some lack of support/discrepancy predates training. Do not conclude opacity loss alone fixes the camera/depth geometry. The difficult submap's local feature gate was permission for a bounded pilot, not full physical-geometry acceptance.

**Best supported result and D8 decision.** Keep the matte 7k confidence scene as a local appearance/checked-consistency improvement (28.750 versus 21.458 dB; all nine declared coarse view/region follow-up checks and the exposed captured/small-offset visual review pass). Its original all-view >0.01 m depth-gain and confidence-policy-superiority claims remain unproven. Keep both exhibit candidates experimental/unpromoted, and retain the original whole-room 120k model for whole-route context. Do not merge local worlds or launch another full-recording 120k training run. R2's bounded real-track/metric-BA/bridge/revisit and concrete D8 experiment have executed; distributed reliable global placement remains absent. R3 has a supported matte local result but the difficult-region acceptance is still incomplete. R1's independent precise physical calibration remains partial, so **not every R0–R3 scientific exit can honestly be marked finished cleanly**.

**Delivery/remaining experiment.** Save each model's matching dataset/K, metric-before-normalization transform, source identities, actual timestamp replay, evidence flags, final checkpoint hashes and viewer configuration in `local_delivery.json` and each `pilot/camera_provenance.json`. Replay start/middle/end/FOV samples are checked offline; no new interactive browser or collision-free-path certification is claimed. The most informative next geometry experiment is to resolve opaque backdrop/floor/post support at the failed source views on the corrected inputs, with separate fitting/check structures, before another Gaussian expansion. If this recording cannot support those constraints, deliver the validated matte submap plus uncertain whole-room context under D8; independent physical/reference capture is still needed for accuracy claims. Final source/baseline integrity verification and machine phase decision follow in the campaign artifacts.

**Final clean integrity verification.** `scripts/audit_db3_r0_r3.py` completed successfully in 45.93 s. Full 17.129 GB DB3 SHA256 remains `60e3f253051fc282a61af7f6c9c5eef40830a12d42566b9eba0992463115bc52`; all frozen baseline originals/copies, 477 baseline RGB/depth hashes, all 477 R1 inputs and all 186 fresh local input hashes match. All native/historical reservations and spatial feature exclusions pass. The six selected 7k and one 15k final checkpoints are finite and <=180k; common 7k initializations/replays match; 15k initialization and replay prefix match. The rejected original unsafe 7k checkpoint stays excluded. Saved final source snapshots/hashes, `integrity_audit.json`, `phase_decision.json`, `local_delivery.json`, matched camera/dataset/viewer metadata and selected comparison sheets. Actual timestamp replay spans 7.910 s over 95 matte poses and 7.909 s over 91 exhibit poses. No package changes, full-scene retraining, mesh extraction, commit or push. Final report: [R2/R3 results](DB3_R2_R3_RESULTS_20261110.md); roadmap checkpoint updated to the actual partial scientific acceptance, without changing historical evidence or unrelated workspace work.


## 2026-11-11 — R0–R3 cleanup, immutable re-audit and structure-support follow-up

**Request and scope.** Recheck and clean up everything through R3 because problems remain. The user confirmed: use the existing recording and document unresolved physical limits. No external reference is assumed. Preserve the raw DB3, baselines, prior positive/negative experiments and unrelated dirty workspace files. Fresh evidence namespace: `results/db3_r0_r3_cleanup_20261111/`. Full findings, solutions, measurements, reproduction commands and phase checklist: [cleanup report](DB3_R0_R3_CLEANUP_20261111.md).

**Verification problems and fixes.** Paired acceptance silently skipped regions with <100 source samples; confirmation loaded frozen rules but used hardcoded constants; view matching depended on list order. Added shared fail-closed acceptance over the complete frozen view/region roster, with source identity checks, finite/range validation and protocol thresholds. Kept the original depth-gain claim separate from the narrower appearance follow-up and unequal-compute comparison. Both integrity scripts previously overwrote historical source snapshots/reports: they now use fresh namespaces and verify old snapshots rather than replacing them. Confirmation likewise refuses overwrites. Expanded R0–R3 audit binds the actual training sparse cloud, metric cameras, sampled depths, normalized camera provenance, viewer paths, checkpoint hashes and replay sequences to the saved evidence. No existing saved gate changed on recomputation; the repaired unsupported-region path is a latent false-promotion risk, not a claim of historical false promotion.

**Fresh validation.** Final broader suite: 277 passed in 45.09 s, one pre-existing unknown pytest `env` warning. Earlier broad/focused runs: 274 and 13 passed, respectively; the final count includes three additional acceptance regressions. R0/R1 verifier suite: 17 passed (overlaps the broader suite). Raw-source reproduction of all 16 R1 review remaps/unfilled depths, all 477 original/replacement inputs and exclusions passed. Expanded R0–R3 audit passed in 35.11 s: full raw SHA256 unchanged (`60e3f253051fc282a61af7f6c9c5eef40830a12d42566b9eba0992463115bc52`), 186 local input hashes/exclusions preserved, all six valid 7k and one 15k selected checkpoints finite and within cap, initialization/replay matching, 95/91 matching delivered cameras, 90k matching initializer points per region and 21 exactly reproduced sampled depth files per region. Historical snapshots remain hash-valid. Changed Python files compile; `git diff --check` passes with existing Windows line-ending advisories.

**Bounded support experiment and finding.** Seven frozen-pose variants separate 21-frame versus 88-frame fitting support, 1 cm voxelization, the 90k cap and an optional 5 cm support-agreement limit. No reserved/evaluation observation contributes fitting points or support; no evaluation depth adjusts any pose or model. Original initializer construction reproduces exactly. At the middle post, 90k selection has 85.40% conditional point support and 0.47045 m median depth discrepancy; restoring all 233,380 original supported voxels gives 100% and 0.03554 m. Thus the cap materially loses this structure's support. Dense supported fitting observations yield 100%/0.01672 m there, but only 52.50%/0.20542 m at the first backdrop. Tighter 5 cm agreement reduces backdrop support to 5.83% and retains 0.16692 m discrepancy. First/middle floors remain partially supported, and the late wall retains 0.07525 m discrepancy even with the tighter rule. These conditional nearest-point measurements change sample support and are not rendered-surface or absolute-accuracy claims. None resolves all structures. Both the initial six-variant run and the seven-variant isolation follow-up are retained. Historical initialization defaults were not changed; support-aware selection under the cap remains an explicit future bounded experiment.

**Acceptance and delivery.** Recomputed matte confirmation passes all nine coarse checks; exhibit 7k and 15k still fail coarse/physical-structure confirmation. Original paired pipeline/policy statuses are unchanged, including the 15k coarse pipeline pass that does not override failed fixed structures. Fresh visual review of saved contact sheets agrees with the previously reported matte improvement and exhibit limits; no new interactive or collision-free-path claim. Keep matte and uncertain full-room context separate; keep exhibit unpromoted. R0 is complete; R1 physical alignment remains partial; R2 local/D8 deliverables are verified but global geometry unaccepted; R3 difficult-region acceptance remains incomplete. Record these in `phase_decision.json`; do not mark all scientific exits complete. No new training, global merge, mesh, dependency change, commit or push occurred.

## 2026-11-11 — R4 full-route reconstruction expansion and D8 decision (in progress)

**Request.** Complete R4 thoroughly and correctly using the existing recording. Preserve prior baselines, failures, reservations and unrelated work. R4 is gated by R1 physical alignment, R2 global placement and R3 fixed difficult-structure checks; the roadmap explicitly permits a documented D8 partial result when scene-wide acceptance cannot be established.

**Predeclared experiment.** New `scripts/db3_r4_route.py` freezes all 477 R1 selected identities over the full recording, checks all eligible adjacent intervals with disjoint fit/check feature tiles, excludes every native/historical reservation, and retries every eligible failure using actual native observations, stopping each chain at its first failed edge. Endpoint checks are required for successful chains. 317 intervals are eligible. These are local consistency candidates, never global-pose or physical-accuracy acceptance. Graph connectivity cannot override the frozen exhibit structures or absent independently distributed placement constraints. No additional Gaussian training is justified until those gates pass. Fresh main namespace: `results/db3_r4_20261111_v2/`; original failed preflight namespace retained at `results/db3_r4_20261111/`.

**Implementation finding/fix.** The first execution stopped before producing any geometry result because R1 depth archives contain sparse `points`/`depths`, not dense `depth_m`. Replaced the incorrect dense-key assumption with nearest-surface reprojection of each original raw depth message using the same SDK-consistent calibration, preserving holes. Existing RGB and sampled-depth hashes are checked before use. Native retries are decoded read-only with header-pair checks. No source/prior result was changed. Four new fail-closed controls pass: connected-but-inaccurate graphs, missing/duplicate/unsupported roster entries, source reservations, and experiment authorization versus scientific acceptance.

**Fresh integrity result.** `scripts/audit_db3_r0_r3.py --output-dir results/db3_r4_20261111_integrity` passed in 36.05 seconds, including full raw SHA256, baseline/input/checkpoint/replay/transform checks and historical snapshot verification. Source remains `60e3f253051fc282a61af7f6c9c5eef40830a12d42566b9eba0992463115bc52`. Route screen and delivery verification are still running; final results follow below. No whole-scene success is inferred from this integrity pass.

**R4 delivery verification and additional engineering fixes.** Reloaded the preserved 120k-update baseline (1,063,961 Gaussians) and rendered all 477 original cameras over 158.690 s; every RGB/depth/alpha output is finite. Saved 20 captured/rendered pairs and all per-view measurements. Baseline mean PSNR is 26.6673 dB over all views (includes training), 23.1133 dB over 60 historical exposed evaluation identities, minimum 10.2131 dB. These are historical-projection contextual diagnostics, not new-model or SDK-comparison gains; alpha is not physical surface coverage. Matching baseline poses and normalization reproduce the frozen R0 values exactly. The separate matte checkpoint reloads with 179,785 finite Gaussians. Its own matching dataset/world is retained; no global merge is asserted. Recorded-timestamp offline replay samples are finite at start/middle/end; interactive validation remains R5.

The first GPU command omitted the venv executable directory from PATH, so installed Ninja was unavailable; the new runner includes the existing venv/CUDA/MSVC paths, without package changes. Packaging then finished rendering/manifest preparation in 22.11 s but its concurrent seal met a not-yet-created source_snapshot directory; added independent directory creation and a seal-only path that checks the existing complete evidence before hashing it. Original failed logs and the exact executed packaging source are preserved. Default pytest temp-directory permissions caused eight setup errors in the first focused suite (27 tests passed); a fresh workspace basetemp resolves them: 35 passed, one pre-existing pytest env warning. Python compilation, PowerShell parsing and tracked-log whitespace checks pass. No test assertion failure or scientific threshold change occurred. Native retries and their final independent source/exclusion/chain audit are still running; their final outcome follows.

### R4 final outcome — whole-route screen and independently verified partial delivery

**Measured result.** The full-route screen completed in 683.23 s. Frozen roster: 477 selected / 4,764 native observations; 377 selected observations are fitting-eligible. Of 317 eligible adjacent intervals, 132 pass local pair checks and 185 fail. All 185 failures received the declared native retry. Zero satisfy the complete gate: 108 stop at a failed short link; 77 complete their short-link chain, of which 72 lack 20 stable endpoint checks and five have enough endpoint matches but fail separation limits. The other 159 adjacent intervals are excluded by reservations, not failed geometry. Retained 1,688 distinct fitting inputs (1,605,067,577 bytes of input evidence); no evaluation/reserved observation enters a fit. Insufficient endpoint support is not a unique drift diagnosis or proof of impossibility.

**Independent final verification.** `scripts/verify_db3_r4.py --output-dir results/db3_r4_20261111_v2` passes in 31.01 s. Verified every input hash, fit/check tile mask and fit-only identity; complete selected/retry rosters; native transform composition; delivery artifact hashes; exact baseline camera/normalization equality to frozen R0; and sixteen temporally stratified unfilled dense-depth reproductions from the raw read-only DB. Original source/baseline/pilots and failed evidence remain intact. Final focused tests: 35 passed, one existing configuration warning. GPU delivery reload checks and full source integrity checks pass separately, without converting scientific failures into passes.

**Decision and remaining limits.** R4's bounded full-route expansion experiment and D8 delivery are finished, but a new accepted improved whole-scene reconstruction is NOT achieved. Keep the preserved baseline as uncertain full-recording context and the narrowly accepted matte improvement as a separate local world. Keep both exhibit candidates unpromoted. No new Gaussian training/extension, global BA/merge, mesh, package/dependency change, commit or push. Global reconstruction needs stronger checked overlap and unresolved exhibit backdrop/floor/wall/post geometry; this recording's impossibility is not established. Do not silently close R1 physical calibration, R2 global placement, R3 difficult-region acceptance or R5 interactive validation. The final report explains the explicit partial-result outcome, baseline per-view diagnostics, fixed-region visual findings, all engineering failures/fixes and exact reproduction/viewer commands: `docs/DB3_R4_RESULTS_20261111.md`. Roadmap checkpoint updated to match. Machine evidence, uncertainty inventory, temporal map, source snapshots, model/camera hashes and matching delivery live in `results/db3_r4_20261111_v2/`.

## 2026-11-11 — User-requested R4 quality review: whole-room goal unmet

**Problem/request.** User reports that R4 does not look good and asks for a check. Reviewed the current report, roadmap, phase decision, delivery manifest, route/retry code and saved per-view evidence rather than treating an integrity PASS as reconstruction success. Full review: `docs/DB3_R4_REVIEW_20261111.md`.

**Delivery finding.** R4 still delivers the original 120k full-room checkpoint plus the separate R3 matte model; it did not execute global bundle adjustment, whole-scene retraining or merging. Fresh read-only reanalysis verified all 50 sealed delivery artifact hashes and both model hashes. No corruption found in those checked artifacts. The raw-recording/dense-depth audit was not rerun. Integrity does not establish acceptable geometry or appearance.

**Visual/metric finding.** Inspected the full-route sheet and full-resolution capture/render pairs 0204 (~68 s) and 0389 (~130 s). Exhibit textures and fine boundaries soften; the later room view has pronounced smeared display edges and lost distant/person detail. Source blur, clipping and a person are also present, so this is not a unique-cause diagnosis. Saved PSNR reanalysis: training 417 views / 27.1786 dB; historical exposed evaluation 60 / 23.1133 dB; 13/60 evaluation views below 20 dB (descriptive threshold only). Combined mean 26.6673 dB hides poor individual views. Worst scores: 10.2131 dB at 8.144 s and 11.6881 dB at 149.346 s; those unsaved individual renders were not visually inspected. No new rendering or untouched-test claim.

**New experimental-design finding.** Of 317 eligible selected intervals, 132 pass and 185 fail; 159 reserved intervals are excluded, not failures. 176/185 failures have fewer than 20 depth-supported check matches. Native retry code changes the transform but reuses original endpoint descriptors, check masks, matching and depth filtering, so it cannot increase those endpoint check counts. All 77 completed chains have exactly the same count as their original pair; 72 remain count-insufficient and five fail residual thresholds with sufficient matches. Another 108 chains stop on a short-link failure. Thus zero recovered intervals does not mean 185 physically invalid transitions or that the recording is unrecoverable. Five sufficiently supported endpoint p90 errors are 0.15120, 0.15976, 0.16967, 0.12130 and 0.47636 m at boundaries 113, 126, 149, 332 and 449 respectively. Failure conditions can overlap; missing checks are not the only issue.

**Decision/proposed solution.** R4 screening and fallback packaging are complete; the user's whole-room reconstruction objective remains unmet. Return to bounded pose/overlap recovery with persistent multi-frame tracks, local camera/landmark refinement, and additional independently checked support on one failed transition plus a passing control. Freeze a new protocol and preserve all fit/check reservations; do not simply lower the check-count gate or repeat the same endpoint-limited retries. Inspect sufficiently supported residual failures separately. Require geometric and fixed-structure improvement before local Gaussian comparison and full-route expansion. This next experiment is proposed, not executed; viewer validation alone cannot resolve the reconstruction problem.

**Artifacts/verification.** `results/db3_r4_review_20261111/review.py` executed successfully and wrote `review.json`, including recomputed counts, PSNR distribution, worst-view identities, checked hashes and source evidence hashes. Original results/models and unrelated work preserved. No training, threshold/code change, global merge, mesh, dependency install, commit or push.

## 2026-11-11 — R4 persistent-track recovery: preregistered bounded experiment

**Request/problem.** User authorized the proposed recovery experiment after the R4 quality review. Original R4 native chaining cannot increase its reused endpoint match count; a whole-room improvement is still absent. Preserve all prior checkpoints, results, calibration and reservations.

**Frozen protocol before new results.** `results/db3_r4_track_recovery_20261111/protocol.json` fixes source identities for target boundary 362 (15 native views near 121 s; prior complete chain but only 12 endpoint checks), passing control 204 (11 views near 68 s), and large-tail diagnostic 449 (11 views near 150 s). Match SIFT descriptors across gaps 1/2/3/5 plus endpoints; construct persistent components, reject repeated-camera ambiguities and entire mixed fit/check components, retain tracks of length >=3. Validation features never enter camera/landmark optimization. Additional endpoint support comes from multi-frame check tracks, with first-last/first-middle/middle-last identities frozen before BA. This is additional correspondence evidence on exposed sensor data, not external physical truth.

**Method/acceptance.** Fixed first camera/intrinsics, up to 800 deterministic fitting tracks with two stable depth samples, robust joint pixel/depth BA capped at 60 function evaluations. Compare native chain, optimized cameras and direct endpoint estimate on identical new checks; also preserve original descriptor endpoint checks. Each new pair needs >=20 checks, >5% spatial hull, median/p90 separation <0.05/0.12 m, symmetric reprojection <2/4 px and positive depth. Require solver convergence and no >0.01 m fixed-check worsening; a pose improvement additionally requires >=0.01 m endpoint median or p90 gain. Support recovery and pose improvement are separate decisions. Unresolved physical structures/global placement still block training and whole-scene acceptance. No threshold tuning or reserved-frame fitting.

**Implementation/verification so far.** New `scripts/db3_r4_track_recovery.py`; source snapshots and input hashes frozen. Ten focused tests pass, including synthetic known-camera recovery, mixed-track/ambiguous-track rejection, indirect validation-track support, insufficient spatial support, fitting exclusion and prior R4 reservation/promotion controls. One pre-existing unknown pytest `env` configuration warning. Real-data run follows; no positive scientific result claimed yet.

**First-run findings and bounded follow-up protocol.** The first run completed all three cases. Persistent-SIFT check counts (endpoint/first-half/second-half): target 3/7/0, control 51/64/56, large-tail 17/29/24. Whole-component rejection of mixed fit/check identities preserves separation but loses many long tracks; target rejects 499 mixed components. No candidate passes the full gate. Target and large-tail solvers hit the 60-evaluation cap; the control converges but original endpoint median/p90 depth errors worsen from 0.01864/0.07569 m to 0.03073/0.09315 m. The large-tail original p90 stays about 0.476 m. No promotion or training.

A disclosed adaptive follow-up is frozen in `results/db3_r4_patch_followup_20261111/protocol.json` before new measurements: same source windows, original fitting tracks/initialization/objective, fixed 240-evaluation budget for every case. Obtain additional validation using first-frame corners tracked through all native images, forward/backward error <0.75 px, neighboring 11x11 patch NCC >=0.85, all observations remaining in original check tiles, no pose/residual filtering, and endpoint deduplication. Freeze checks before BA; preserve original SIFT/direct checks as regressions/vetoes. No check threshold relaxed, no new frame enters fitting. `scripts/db3_r4_patch_followup.py` added; two synthetic patch tests pass (known translation and blank image), existing pytest configuration warning only. This is exposed development evidence, not independent physical validation.

### R4 persistent-track recovery — final measured result and decision

**Executed results.** Both experiments finished. In the patch follow-up, endpoint/half-window stable-depth counts are target 0/0/0, control 36/37/38, late diagnostic 38/37/40. The target still reaches the 240-evaluation cap; control converges at 43 and late diagnostic at 64. Target starts with 269 corners but none survives all full-window fixed-tile/visibility/forward-backward/patch checks. The combined rejection is recorded; individual-rule causation was not measured. This is an experimental support limitation, not proof of absent scene information.

**Quantitative finding.** On identical patch endpoint checks, control chain vs refined median/p90 separation is 0.01717/0.06725 vs 0.02734/0.06596 m; late diagnostic is 0.01586/0.04929 vs 0.01706/0.05161 m. Reprojection improves: control median/p90 1.862/2.947 -> 0.448/1.515 px, late diagnostic 1.832/2.647 -> 0.671/1.341 px. The original chain already passes all patch pairs in both windows. No >=0.01 m endpoint metric gain is established. Control original checks still regress; late original descriptor p90 remains 0.47566 m. No full candidate passes, and no threshold or original failure was removed.

**Correspondence diagnosis.** Original boundary-449 errors include 0.854/0.690/0.556 m. Inspected full-resolution crops of all six worst original matches. The worst three visibly pair different repeated door details (different grid/panel corner; latch vs another edge; lower panel vs another edge). Saved all 23 original matches, coordinates, depths and errors plus labeled crops in `results/db3_r4_track_audit_20261111/`. This supports a correspondence-error contribution, not a unique explanation for every residual. New 38-match patch support is reported as a different frozen check set, not as an old-gate pass after deleting outliers.

**Verification.** Twelve focused tests pass across the two new modules and existing R4 safeguards; one existing pytest env-configuration warning. `scripts/audit_db3_r4_track_recovery.py` passes: 37 input hashes and source-role checks, six source snapshots, 28,504 fitting observations, 4,039 disjoint descriptor checks, 1,804 patch observations, deterministic track/check regeneration, valid camera transforms and independently recomputed metric residuals. Sealed 105 experiment artifacts. Original baseline and matte checkpoint hashes remain unchanged. Full raw-database hashing was not rerun. Scientific acceptance is distinct from this integrity pass.

**Decision/remaining problem.** The authorized bounded recovery experiment and disclosed follow-up are executed, but an improved whole-room reconstruction is NOT achieved. Preserve original models; no training, promotion, global merge, mesh, package change, commit or push. The new evidence rules out treating all large descriptor residuals as camera drift and shows that smaller image residuals can worsen metric checks. Fixed image-tile holdouts lose moving physical tracks; a subsequent experiment would need a new frozen source-identity track holdout, excluding its observations from both initialization and fitting, plus independently reviewed static structures and reliable depth. That subsequent experiment is proposed, not executed. R4 remains scientifically unmet. Full evidence/reproduction: `docs/DB3_R4_TRACK_RECOVERY_RESULTS_20261111.md`.

## 2026-11-11 — Corrected moving-source holdout reconstruction experiment

**Request/problem.** User explicitly requested execution of the corrected local experiment. Prior validation lost tracks by requiring a physical feature to remain inside fixed image tiles. Preserve prior results/models and native/historical source reservations.

**Frozen protocol.** `results/db3_moving_holdout_20261111/protocol.json` fixes the same difficult target (boundary 362, 15 native cameras, 120.678–121.145 s) and a passing interval inside the accepted matte region (boundary 36, 11 cameras, 12.015–12.348 s). Initial check corners still come from original check tiles, but their source identities may move anywhere in later images. Forward/backward <0.75 px, NCC>=0.85, patch/border/dedup criteria stay unchanged. Exclude every fitting feature within 12 px of every flow-valid reserved observation, including partial tracks, and propagate exclusions through entire descriptor components. Preserve all original tile exclusions. Refit native camera initialization from remaining fit-only descriptors; old poses are comparison references only. Freeze checks/exclusions/tracks before fitting.

**Two predeclared objectives.** Fixed first camera and intrinsics; joint camera/landmark fitting with pixel+depth or robust metric-3D residuals, same tracks/initializer, up to 800 fitting tracks, >=30 with two stable depths. Analytic sparse Rodrigues/projection Jacobian, parameter scaling by Jacobian, fixed 240 evaluations and 400 LSMR iterations. Keep existing metric/reprojection/support thresholds, no >0.01 m fixed-check worsening and >=0.01 m endpoint metric gain for a pose-improvement claim. Original endpoint checks remain recorded without deletion. Local support recovery and pose improvement are distinct; short-window support alone does not authorize whole-scene training.

**Implementation validation.** New `scripts/db3_moving_holdout.py`; five focused tests pass: analytic Jacobian versus finite differences (including rotation), known-camera recovery in both objectives, whole-component holdout propagation, tracked features crossing original tile boundaries, and no fabricated tracks on blank input. Existing pytest env-configuration warning only. Protocol/source snapshots frozen before real-data execution; results follow.

**Moving-holdout measured outcome.** Corrected tracking recovers 74 full-window target validation tracks, all crossing old tile boundaries, with 58/59/63 stable-depth checks on endpoint/half pairs. Matte has 112 tracks (63 crossing) and 104/103/107 checks. Both analytic-Jacobian objectives converge in both windows. Matte passes local support and noninferiority but shows no >=0.01 m metric gain. Target metric errors are small (old endpoint median/p90 0.01320/0.02315 m), while symmetric reprojection is 5.075/7.857 px. Pixel+depth refinement reaches 2.241/7.523 px but fails support and original-check noninferiority; metric refinement reaches 4.262/6.875 px and remains noninferior. The target's issue is now measurable rather than missing evidence. No pose-improvement or training promotion.

**Predeclared tracked-fitting follow-up.** `results/db3_tracked_fit_20261111/protocol.json` freezes reuse of the exact moving/original checks and thresholds. Replace SIFT fitting matches with directly tracked fitting corners, seed every third camera, reject original tiles and 12px moving reservations, discard entire tracks entering a reserved area, suppress duplicate seeds, and keep >=3-view tracks. Refit adjacent initializers from shared stable-depth track observations only; run the same two fixed refinement objectives. This directly addresses false fitting correspondences rather than changing checks. Two synthetic tests pass: known metric translation with reservation exclusion, and blank-image fail-closed initialization. This is a disclosed adaptive development follow-up, not untouched confirmation. Results follow.

### Corrected moving-source holdout experiment — completed result

**Implemented corrections.** Source-following validation tracks, 12px moving exclusions plus all original tiles, whole-component exclusion propagation, and fit-only reinitialization are implemented and executed. The disclosed tracked-fitting follow-up replaces potentially false SIFT fitting matches with persistent optical-flow corner tracks and refits every adjacent initializer. No original checks are deleted. Seven new focused tests pass, including analytic Jacobian validation and known metric camera recovery. Existing pytest env-configuration warning only.

**Results.** The difficult target now has 74 moving tracks, all crossing old tile boundaries, and 58/59/63 metric checks; the prior fixed-tile patch method yielded zero full-window tracks. Matte has 112 tracks and 104/103/107 checks. Follow-up fitting supplies 728 target and 793 matte tracks; all four optimizations converge. Target pixel+depth old-vs-new endpoint separation median/p90: 1.320/2.315 -> 1.680/2.434 cm; reprojection: 5.075/7.857 -> 3.244/7.494 px. Target metric arm: 1.385/2.256 cm and 4.997/8.100 px. Both target arms are noninferior on fixed metric checks but fail image alignment. Matte tracked pixel+depth: 1.376/3.223 cm and 0.529/1.906 px; matte tracked metric: 1.043/2.977 cm and 1.432/2.833 px. Both matte arms pass local consistency/noninferiority; no >=0.01 m endpoint metric gain is established. Lower fitting RMS uses a different fitting set and is not an independent accuracy gain.

**Visual finding/limits.** Reviewed six worst unchanged target checks; five lie on painting/details and one on the frame edge. Source crops show nearby corresponding visual features but do not establish subpixel truth. No residual or check was removed. Blur/appearance changes, tracking, depth, calibration/timing and pose remain possible contributors; no unique recording-quality diagnosis or physical-accuracy claim. New validation support proves the previous fixed-tile recipe discarded usable tracked observations, not that the target is now reconstructed correctly.

**Verification.** `scripts/audit_db3_moving_holdout.py` passes: 26 input image/feature pairs, reservations, seven source snapshots, 104 proper finite camera matrices, 11,572 tracked fitting observations, 2,342 moving check observations, reproduced exclusions and pre-fit seals, and independently recomputed metric/pixel residuals. Sealed 117 experiment artifacts. Original whole-room and matte model hashes unchanged; full raw database hashing not rerun. Scientific chart and unchanged check crops saved under `results/db3_moving_holdout_audit_20261111/`.

**Final decision.** The requested corrected local-geometry experiment and tracked-fitting follow-up are complete. The tracking/exclusion defect is fixed; matte control is supported; target color/depth/pose consistency remains insufficient. No new Gaussian appearance comparison or route expansion is justified by this failed target gate. Existing models, prior results and reservations preserved; no training, promotion, global merge, mesh, dependency change, commit or push. R4 whole-room improvement remains unmet. Full report and exact reproduction paths: `docs/DB3_MOVING_HOLDOUT_RESULTS_20261111.md`.

## 2026-11-11 — R5 prerequisite review and bounded direct-depth correction

**Request/problem.** User asks to start R5 after ensuring earlier work was done correctly, completing missing work first. Current source reports and machine results show completed R4 experiments but no accepted improved whole-room reconstruction. Target boundary 362 still fails the unchanged 2/4 px median/p90 image-alignment gate; R1 precise physical alignment, R2 global placement and R3 difficult structures remain open. R5 must not certify those unresolved gates.

**Fresh verification.** `scripts/audit_db3_moving_holdout.py --output-dir results/db3_r5_preflight_20261111/moving_holdout_audit` passes: 26 inputs, seven source snapshots, 104 poses, 11,572 fitting observations, 2,342 moving checks and 117 sealed artifacts; original delivered model hashes unchanged. Relevant geometry/acceptance/replay/viewer unit tests: 36 passed, one existing pytest `env` warning. This confirms recorded failures, not physical accuracy. Full raw hash not yet rerun.

**Finding/hypothesis and bounded corrective action.** Existing pixel+depth BA adjusts both cameras and latent landmarks, whereas acceptance projects measured sensor depths directly. A smaller landmark reprojection residual therefore need not improve measured-depth reprojection. This is an objective mismatch to test, not a proven implementation defect or root cause. Added `scripts/db3_direct_depth_refine.py`: one camera-only arm fitting symmetric measured-depth pixel projections plus metric separation, fixed intrinsics/first camera/depths, existing tracked fitting observations and independently reconstructed fit-only initializers. Keep all original tiles, moving-source reservations, frozen checks and thresholds. No hyperparameter sweep, check deletion, depth correction or calibration change. Execute target and matte control with the same fixed 240-evaluation budget. Source/input hashes and protocol must be frozen before measured results.

**Advance/stop rule.** Local consistency, noninferiority and meaningful metric gain remain separate. A successful short interval only justifies further regional checks, not whole-room training or final R5 acceptance. If this bounded correction fails, retain the prior models and record R5 blocked on reconstruction; do not substitute viewer operation for geometry acceptance. Results will be appended below.

### R5 prerequisite review — completed audits and direct-depth result

**Executed correction experiment.** Froze and ran `results/db3_direct_depth_20261111/`: one camera-only measured-depth reprojection/metric objective, exact existing fitting tracks/initializers and unchanged validation. Target uses 670 stable-depth fitting tracks / 11,654 pair constraints; matte 784 / 15,525. Both converge (target seven evaluations / 2.82 s; matte ten / 4.43 s). Target endpoint metric median/p90 is 1.696/2.485 cm and reprojection 3.050/7.401 px, versus old-chain 1.320/2.315 cm and 5.075/7.857 px. Target still fails the unchanged <2/<4 px image-alignment gate. Matte gives 1.418/3.218 cm and 0.499/1.910 px and passes local consistency. Both remain metric-noninferior; neither establishes the predeclared >=1 cm metric gain. No validation check was deleted. The objective mismatch hypothesis does not resolve the target failure and is not established as its root cause.

**Verification.** Two new tests pass (known camera rotation/translation recovery, reserved/insufficient-fit rejection), yielding 38 relevant passing tests across two invocations. Existing pytest `env` configuration warning only. `scripts/audit_db3_direct_depth.py` passes: frozen inputs/sources, fit-only source reservations, all 11,572 stored fitting observations outside original/moving exclusions, 27,179 fitting pair references, rebuilt initializers, proper poses, all unchanged check identities and independently recomputed validation errors. Original model hashes unchanged. Fresh full R0-R3 integrity audit passes in 61.46 s, including full raw DB3 hash `60e3f253051fc282a61af7f6c9c5eef40830a12d42566b9eba0992463115bc52`, baseline/R1 input preservation, pilot/replay/provenance and historical snapshots. Fresh moving-holdout audit also passes. Integrity is not scientific acceptance.

**Decision.** R5 was NOT started under the user's condition: R4 whole-room improvement and the earlier physical/structure/global-placement gates remain unmet after the bounded correction. Preserve the original whole-room context, separate matte model, all failed evidence and reservations. No new Gaussian training, model promotion, merge, mesh, dependency change, commit or push. Partial-delivery viewer testing is technically possible but cannot certify unresolved geometry. Further work must distinguish track/depth/calibration/timing/pose inconsistency using reviewed static structures before regional and global expansion; the recording's unrecoverability is not established.

**Artifacts/reproduction.** `docs/DB3_R5_PREREQUISITE_REVIEW_20261111.md` contains the exact commands, per-window comparisons, limitations and next requirements. Machine prerequisite status and fresh audits: `results/db3_r5_preflight_20261111/decision.json`, `r0_r3_integrity/`, `moving_holdout_audit/`, `direct_depth_audit/`. New protocol/results/source snapshots: `results/db3_direct_depth_20261111/`. Implementations: `scripts/db3_direct_depth_refine.py`, `scripts/audit_db3_direct_depth.py`; tests: `tests/test_db3_direct_depth_refine.py`. Prior result namespaces and unrelated workspace edits remain preserved.

## 2026-11-12 — R4 blocker diagnosis and recovery

**Request/problem.** User asks to solve the persistent R4 blocker and enable R5. Current owning reports confirm completed experiments but no accepted improved whole-room model. Prior direct-depth target error is 3.050/7.401 px median/p90; matte control passes. Existing results and dirty worktree are preserved.

**New hypothesis and frozen diagnostic.** Incremental optical-flow checks compare neighboring patches only, permitting accumulated template drift. Test direct anchor-to-endpoint matching of the exact same identities, initialized by the old tracks, alongside an RGB-only homography fitted only on the existing fitting tracks. A homography is a diagnostic for image consistency, not accepted 3D geometry. Record every check, failure and displacement without deleting original checks or changing thresholds. Both target and matte control use identical settings. `scripts/diagnose_db3_tracking_bias.py` freezes inputs/method before measurements in `results/db3_tracking_bias_20261112/protocol.json`.

**Reference.** OpenCV documents `OPTFLOW_USE_INITIAL_FLOW` and pyramidal LK in https://docs.opencv.org/4.7.0/dc/d6b/group__video__track.html. Using the chain as initial flow lets the direct match compare original image content without fitting a pose to validation. RealSense's depth-quality guidance distinguishes depth noise from absolute accuracy (https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance/); neither a local plane nor low fitted residual is independent truth.

**Diagnostic result and disclosed follow-up.** `db3_tracking_bias_20261112` shows direct pyramidal LK is unreliable over the target: only 21/74 endpoint tracks meet forward/backward <0.75px; median chain-to-direct difference is 20.48px. An unfiltered fit-only RGB epipolar diagnostic also retains a 6.96px p90 target error (matte 1.94px), so changing depth alone cannot explain away the target tail. This does not isolate tracking, reflections or rolling shutter. Inspecting full source images shows reflective display/painting regions and strong image motion. The global homography has only 23/124 target fitting inliers, so it is inappropriate as a single scene model.

**Follow-up before results.** Freeze `db3_anchor_patches_20261112`: affine ECC registers each exact old source patch directly to each frame around the old trajectory. Fixed 41px patch, bidirectional correlation>=.85, cycle<.75px, displacement<=12px, positive determinant and singular values (.75,1.3334). This handles small patch deformation while retaining original-template identity. Every old identity and every failed registration is reported; old gates/results stay intact. Diagnostic measurements on corrected locations are a new arm, not a retroactive old-check pass. Future fitting must rebuild exclusion masks around the larger patch support before these observations can validate a newly fitted camera.

**Template correction rejected.** The fixed affine-patch arm preserves every identity but only 2/74 target and 4/112 matte tracks satisfy the full-window quality checks. Its all-identity diagnostic endpoint remains 2.858/8.704px for target. It is not adequate validation or a recovery fix; no fitting/model promotion follows.

**Innovative correspondence branch, frozen before results.** `scripts/db3_learned_correspondence.py` uses pretrained LoFTR indoor_new (ScanNet), Kornia 0.8.2 and kornia_rs 0.1.10 installed only under `.tools/db3-matcher` with --no-deps; existing venv dependencies unchanged. Original model mirror URL returned 404; official maintainer model URL `https://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor_ds_new.ckpt` downloaded successfully. Protocol records exact weight and source hashes. Fixed 640x352 inference, confidence>=.8, unchanged native-coordinate geometry limits. Three endpoint/half pairs in both original windows. New checks occupy original check tiles in BOTH images, no residual filtering. New fitting excludes original tiles, all moving reservations and new checks, with component-wise propagation across 3px near-coincident cross-pair observations. First measure existing direct-depth cameras, then one fixed direct-depth fit using only new fitting correspondences and original fit-only initializer. Existing moving and descriptor checks remain separately reported as failures/vetoes. Learned matches are additional sensor-image evidence, not truth or an automatic promotion. References: https://github.com/zju3dv/LoFTR and https://www.kornia.org/tutorials/nbs/image_matching.html.

**Learned branch failures and measured result.** First run (`db3_learned_correspondence_20261112`) completed target with only 1/30/1 stable checks under confidence>=.8 and both-endpoint tile intersection, then failed on an empty matte match list because NumPy inferred float dtype for an empty fit mask. Preserved its exact source snapshot; fixed mask dtype to bool and added a regression test. Disclosed v2 uses the pretrained matcher's default .2 threshold and a source-following union split: if either endpoint occupies a check tile, both endpoints and connected components are excluded from fitting. Physical acceptance thresholds are unchanged. This is adaptive development; matcher confidence and split both changed, so their effects are not isolated.

V2 learned checks count target 546/481/604 and matte 348/451/334; both fits use 800 stable pair tracks and converge in nine evaluations. Target new-check endpoint improves from 3.471/8.190 to 3.004/7.032px with metric separation 1.834/3.559cm, but still fails. On the EXACT original moving checks, direct-depth -> learned-camera target is 3.050/7.401 -> 3.147/8.595px and matte is .499/1.910 -> 2.187/3.478px. Thus learned fitting worsens the retained checks and is rejected. New matcher checks also fail on the matte control, limiting any claim that they establish ground-truth error. No new Gaussian training, model promotion or checkpoint edits. These results do not prove the recording unrecoverable or isolate reflections, rolling shutter, calibration or timing as the root cause.

**Workflow correction and R5 start.** User's current instruction asks to enable R5. Separate starting operational verification from final scientific acceptance: unresolved R4 geometry blocks acceptance of an improved reconstruction but does not prevent testing the preserved delivery. `scripts/run-db3-r5.ps1` verifies the selected config/manifest/checkpoint seal and launches the matching context or matte world without requiring a false geometry pass. On 12 November, started the context viewer at http://localhost:8094 (process 16888) and used the computer-use skill/browser controls to verify visible playback (13.41s), pause at 29.625s, free orbit with capture time held fixed, return to time0 and browser reload. Both scene changes and remaining smearing/ghosting were visibly observed. Browser reload is client reconnect, not a second server checkpoint reload. Initial server startup freshly loaded the checkpoint. R5 is STARTED for operational verification; full R5 reconstruction acceptance remains open. The untouched original R4/R5 historical decisions are retained.

**Final verification and deliverables.** `scripts/audit_db3_recovery_20261112.py` passes: 149 source/input hashes, reconstructed learned check membership, fitting masks/reservations, unchanged original moving-check residuals, delivery artifact seals and both model hashes. Evaluation shares the tested repository geometry evaluator and does not establish physical truth. Live context A/B/A test at 12.015/120.678s restores identical pixels (max difference0); all four free-camera perturbations update images and the return matches exactly. Observed first-frame times .093-.172s in six protocol poses, not a general latency benchmark. Saved actual render evidence under `results/db3_r5_operational_20261112/`.

The relevant suite passes 23 tests. Initial invocation had 17 passes and six Windows temporary-directory permission errors; a fresh workspace `--basetemp results/db3_recovery_pytest_tmp_20261112` resolves those fixture errors. Existing unknown pytest env-option warning remains. Three new correspondence tests also rerun after optional-dependency skip handling. R5 context and matte `-CheckOnly` launch verification pass; matte has not received a new interactive browser test. Operational decision and17 evidence/source files sealed; first learned-run exception recorded alongside its unchanged partial result/source.

**Current outcome.** R5 operational verification is started and usable at http://localhost:8094; full R5 scientific acceptance remains false. The requested whole-room geometry fix is NOT achieved. No new checkpoint is selected, no old threshold/check is relaxed, and no final acceptance of partial delivery is attributed to the user. New report: `docs/DB3_R4_RECOVERY_R5_START_20261112.md`; launcher: `scripts/run-db3-r5.ps1`; machine status: `results/db3_r5_operational_20261112/decision.json`. Added this dated status to the reconstruction plan while preserving historical decisions. No commit, push, mesh, global merge or source-recording edit occurred. Existing worktree changes remain intact.


## 2026-11-16 — Research snapshot and Git ignore audit

**Scope.** Prepare the accumulated research code, tests, documentation, and report deliverables for a single fast-forward push to the existing `depth` branch. Preserve the local datasets, source capture, checkpoints, generated run evidence, and toolchain directories. Included the report sources, exported PDFs, schedule workbook, and supervisor briefing in the snapshot.

**Finding and correction.** The broad `data` and `results` rules matched generic names anywhere in the tree. Scoped those rules to the workspace data and results directories and the generated stage datasets so benchmark tables, fixtures, and stage manifests can be tracked. Added ignores for DB3 sidecars, local environment variants, and LaTeX intermediates. The 17,129,000,960-byte RealSense source remains ignored. No staged file exceeded 1 MiB. Normalized trailing whitespace and redundant terminal blank lines flagged by the staged diff check, preserving intended Markdown line breaks.

**Verification.** All 15 positive and negative ignore checks passed. Parsed all 156 changed Python files and 15 changed PowerShell scripts. The first focused pytest run had 91 passes and three CUDA rasterization failures because the plain shell had no configured gsplat CUDA toolkit. With the existing project CUDA/MSVC environment and extension cache, the same focused set passed all 94 tests. Pytest still reported the existing unknown `env` configuration option warning. This check did not rerun training or establish reconstruction acceptance.

## 2026-11-17 — DB3 artifact-removal literature and repository review

**Question.** Could noisy artifacts explain why the reconstruction from `20261030_134523.db3` has failed the whole-room geometry gates, and which of the three linked projects could help? Reviewed the papers, upstream source and repository history, the DB3 acceptance records and code, and saved capture/render evidence on `depth` at `3dfa0918`. Fresh `run-db3-r5.ps1 -CheckOnly` checks passed for both context and matte, including the selected checkpoint seals. This establishes availability of the preserved artifacts. The saved operational decision still has final reconstruction acceptance open.

**What appears in the recording.** Re-inspected the [37 s](../results/room_improve_region_inventory_20261103/037s_comparison.jpg), [61 s](../results/room_improve_region_inventory_20261103/061s_comparison.jpg) and [110 s](../results/room_improve_region_inventory_20261103/110s_comparison.jpg) comparisons. A visitor changes position near 37 s, the render around 61 s has a dark smear near the right-hand opening that is much less prominent in the prepared RGB, and illuminated reflections are already visible across the 110 s source images. At the actual failing interval (120.678–121.145 s, boundary 362), prepared RGB [94189](../results/db3_moving_holdout_20261111/target/inputs/94189.png), [94373](../results/db3_moving_holdout_20261111/target/inputs/94373.png) and [94553](../results/db3_moving_holdout_20261111/target/inputs/94553.png) show glare and changing reflections over the artwork and display, alongside fixed trim and opaque exhibit surfaces. Visual inspection identifies plausible classes of unreliable observations. Their overlap with the large-residual tracks has yet to be measured.

**Where the measured failure occurs.** The latest tried learned-camera fit yields 3.147/8.595 px median/p90 target endpoint error on the preserved moving checks. The preceding measured-depth camera fit yields 3.050/7.401 px on that target and 0.499/1.910 px on the matte control. The existing image bounds are <2 px median and <4 px p90, alongside support and metric checks on all three pairs. The earlier fit-only RGB essential-matrix diagnostic yields 6.960 px target p90 versus 1.940 px matte p90, with no sensor depth in the diagnostic. Together these show an upstream image-correspondence or camera consistency problem before Gaussian training, with depth error still potentially contributing to the RGB-D failure. Reflections, motion-related image degradation, calibration, timing, and pose error can produce overlapping symptoms. Rendered ghosting and floaters can be downstream effects, and their presence alone cannot explain the upstream failure. [Recovery and validation measurements](DB3_R4_RECOVERY_R5_START_20261112.md) · [Frozen direct-depth protocol](../results/db3_direct_depth_20261111/protocol.json) · [RGB-only diagnostic](../results/db3_tracking_bias_20261112/epipolar_diagnostic.json)

| Project | Upstream implementation inspected | Fit for this DB3 recording |
| --- | --- | --- |
| T-3DGS | `0416a51dcaaf44268117008fde0a4223ad8656a5`. RUP learns transient regions from training behavior with posed images. TMR refines masks using SAM-based video segmentation in the separate AutoVidSeg repo. Precomputed masks are also supported and matched to training images by sorted order. [Paper](https://arxiv.org/html/2412.00155) · [Code](https://github.com/Vadim200116/T-3DGS/tree/0416a51dcaaf44268117008fde0a4223ad8656a5) · [TMR](https://github.com/Vadim200116/AutoVidSeg) | Useful for human or other moving-object ghosts in a posed model. Reviewed temporal masks can also be reused as inputs to DB3 fitting. Poor poses or complex static surfaces can create high reconstruction residuals and false masks, especially around exhibit details. The released trainer takes calibrated poses as input. |
| Human-Aware Inpainting for 3D Gaussian Splatting | `be7192a0a118108944f7193c150cb83c0bf2cc8a`. YOLOv11 detects people, ZITS fills masked RGB for COLMAP pose and sparse initialization, original RGB is used with masked training, and region-restricted LightGaussian pruning removes residual splats. [Paper](https://norihi-k.com/paper/ismar25_liao.pdf) · [Supplement](https://norihi-k.com/paper/ismar25_liao_supp.pdf) · [Code](https://github.com/peggy30520/Human-aware-Inpainting-for-3DGS/tree/be7192a0a118108944f7193c150cb83c0bf2cc8a) | Its upstream masking idea is the closest fit for testing correspondence contamination. The repo targets actual people. Glass reflections and glare require their own reviewed regions. DB3 can exclude contaminated measured observations from fitting and initialization. Inpainted RGB could introduce synthetic features and cannot supply measured depth or geometric validation. |
| Clean-GS | The linked v1 uses mask projection, color/depth-buffer checks, and spatial or neighbor outlier filtering. `ba5c2fa97e9e4bec0510e25aee9f79f6bc8fa822` contains the historical three-stage implementation. Current `main` at `78a39e941c14df5857903b1112040d117f230ac4` follows v2, with Pareto-guided three-view selection and visibility-aware two-of-three voting. Current code expects calibrated COLMAP cameras, target masks, and a pretrained gsplat checkpoint or PLY. Its setup documents Linux/CUDA 12.8. [v1](https://arxiv.org/html/2601.00913v1) · [Historical code](https://github.com/smlab-niser/clean-gs/tree/ba5c2fa97e9e4bec0510e25aee9f79f6bc8fa822) · [v2](https://arxiv.org/abs/2601.00913v2) · [Current code](https://github.com/smlab-niser/clean-gs/tree/78a39e941c14df5857903b1112040d117f230ac4) | Suitable for an isolated exhibit export or a post-training floater review with trusted cameras. Object masks can discard walls, other exhibits, and surfaces outside the selected views. Gaussian deletion leaves the input camera/alignment problem unresolved. The existing [depth-consistency diagnostic](../scripts/audit-initial-depth-consistency.py) offers a starting point for ranking whole-room cleanup candidates with multi-view sensor evidence. Its original flags include points on or near real objects and thin ropes. |

**Integration points.** `scripts/db3_tracked_fit.py` already filters fit-only observations by spatial and temporal holdout reservations. A reviewed region mask can be applied at both seeding and subsequent tracked patch locations, while preserving those existing reservations. `examples/simple_trainer.py` accepts a keep mask for RGB L1 and zeros both images for SSIM. `examples/datasets/colmap.py` currently supplies a camera-keyed image mask and loads measurement confidence separately. Later per-image occluder masks need to follow the same rectification/crop as the RGB and explicitly gate the applicable depth samples or confidence and initialization. Reflection-only RGB exclusions and unreliable measured depth should be tracked separately. The trainer's existing RGB hook does not by itself apply a mask to depth supervision.

**Proposed first comparison.** Review the failing 120.678–121.145 s interval and the passing matte control at 12.015–12.348 s, with annotations based on source RGB/depth and without inspecting per-track residuals while drawing them. Tag moving occluders separately from glare/reflection and retain opaque frame, trim, and other trusted texture, including depth-valid support at different distances. Compare unmasked fitting with a people-only exclusion and a people-plus-reviewed-reflection exclusion using otherwise identical frame pairs, solver and depth policy. If support permits, add a spatially matched reduction of fitting tracks as a control for dropping data. Freeze annotation and fitting rules before running. Keep the original moving and descriptor checks in the report in full and add separately reserved opaque landmarks before fitting. Compare reprojection median/p90, depth agreement, retained track counts and spatial coverage on all three pairs. Improvement isolated to the fit would be insufficient. A drop on reserved static checks with adequate support and a stable matte control would justify more regional work. Retained opaque errors would prioritize a separate timing, motion, or calibration experiment. Use the 37 s visitor sequence for an appearance-only mask pilot if warranted, with fixed-camera views and thin-detail comparisons. These previously exposed windows support diagnosis; confirmation would require a separate reserved region or revisit.

**Outcome.** Review and visual inspection only. Direct loading of the ISMAR PDF by the browser tool exceeded its size cap, so the author's indexed PDF, supplement, and source repository were used. No upstream dependencies were installed, and no mask experiment, camera refit, training, or pruning was run. The three methods have different points of application and none has been benchmarked on this DB3 capture in this review.

## 2026-11-17 — Tested observation masks on DB3 camera fitting

**Problem and treatment.** Tested the pre-fit masking idea from Human-Aware Inpainting on the existing RGB-D pipeline. Used all 15 prepared target frames at 120.678–121.145 s and all 11 matte control frames at 12.015–12.348 s. Annotated source RGB before examining any new residuals. The only possible direct person in the target is a small foreground sliver. Kept its mask separate from the moving reflections and bright highlights. Interpolated three polygon keyframes in native coordinates, expanded the exclusions by 11 px at the rasterized patch locations, and rejected an entire archived track if any observation touched the applicable mask. Rebuilt the fit-only initialization and ran the existing measured-depth camera solver with fixed intrinsics and first camera. No RGB inpainting or depth substitution was used. Archived LK tracks still depend on the original images and their coarse-pyramid support.

**Controls and checks.** Froze four arms before fitting: unmasked, possible-person mask, possible-person plus glare/reflection masks, and a seeded track-count control matched by starting frame and lifespan group. The count control can differ in spatial coverage and stable-depth factors. Kept every original moving and descriptor check. Added separate, source-selected static candidates, reserving complete tracks and every fitting track within 12 px of any reserved observation in a frame. Candidate selection used the source annotations and stable-depth support, with no residual criterion. These are previously exposed observations, and some printed display details may be viewed through glass. The added target checks have only 14/14/26 track pairs. The endpoint and first half also have <5% hull coverage, so the added checks are diagnostic. Complete historical moving-check counts are 58/59/63 target and 104/103/107 matte.

| Trial | Target tracks available | Target endpoint reprojection median / p90 (px) | Matte endpoint reprojection median / p90 (px) |
| --- | ---: | ---: | ---: |
| Unmasked, both runs | 686 | 3.466 / 7.323 | 0.835 / 1.954 |
| Possible-person mask, both runs | 686 | 3.466 / 7.323 | 0.835 / 1.954 |
| Broad glare mask | 157 | Initialization incomplete | 1.384 / 2.971 |
| Broad count control | 157 | Initialization incomplete | 0.839 / 1.980 |
| Narrow glare mask | 464 | 3.420 / 7.617 | 0.851 / 1.988 |
| Narrow count control | 464 | 3.337 / 7.203 | 0.835 / 1.992 |

**Broad-mask failure and follow-up.** Broad masks covered an average 34.61% of the target image and removed 529 of 686 available tracks. Both broad target arms failed adjacent-frame initialization at 1→2, with 25 stable-depth matches under masking and 24 in the count control. The broad matte mask increased reprojection error relative to its count control, while both still pass on the three original moving-check pairs. After seeing the support failure, traced a tighter mask around the strongest visible glare and reflections. This was an adaptive follow-up using the same check IDs, unmasked tracks, solver, depth policy, and thresholds. The narrower mask covers 9.15% of the target image and retains 464 tracks, 441 of which contribute to the optimized depth pairs. The count control contributes 422 of its 464. The possible-person mask intersected zero available tracks in either window, so the equality with unmasked fitting is a null intervention. It does not test the confirmed visitor around 37 s.

**Result and decision.** Narrow masking did not improve the target's alignment consistently. The target half-pair p90s are 3.230/4.652 px under narrow masking versus 3.212/4.608 px unmasked. Endpoint metric separation median/p90 is 1.765/2.624 cm versus 1.721/2.526 cm unmasked. The added static endpoint p90 drops from 5.642 to 5.343 px while the median rises from 3.026 to 3.332 px, with only 14 checks. In the unmasked run, 12 frozen moving endpoint checks touch the narrow reflection regions and have a 9.015 px p90, compared with 6.582 px on 32 other checks and 4.677 px on 14 visually static candidates. Reflections identify some difficult observations, but excluding those fitting tracks did not recover alignment. The target still fails the original <2 px median and <4 px p90 limits. The matte control passes the original moving checks under all completed arms. Keep the masks as diagnostic evidence. No new Gaussian training, checkpoint edits, or promotion were performed. The root cause of the remaining image and depth disagreement is unresolved.

**Reproduction and verification.** Entry point [db3_observation_mask_trial.py](../scripts/db3_observation_mask_trial.py), with [broad](../scripts/db3_observation_mask_regions.json) and [narrow](../scripts/db3_observation_mask_narrow.json) annotations. Each run freezes source snapshots, input hashes, masks, assignments, and parameters before fitting. Use fresh output directories with `prepare --output-dir PATH`, then `run --output-dir PATH`. For the narrow preparation add `--reflection-override scripts/db3_observation_mask_narrow.json --reuse-holdout-from PATH_TO_BROAD_RUN`. The verification recomputed track membership and reservations, matched-count strata, actual fit references, and exact original/moving/static check scores from the saved poses using the repository evaluator. All 15 targeted tests passed with the existing pytest `env` warning. Both context and matte checkpoint seals passed the R5 launcher checks.

- Broad trial [results](../results/db3_observation_masks_20261117/result.json), [target overlay](../results/db3_observation_masks_20261117/target/mask_preview.jpg), [verification](../results/db3_observation_masks_20261117/verification.json).
- Narrow trial [results](../results/db3_observation_masks_narrow_20261117/result.json), [target overlay](../results/db3_observation_masks_narrow_20261117/target/mask_preview.jpg), [verification](../results/db3_observation_masks_narrow_20261117/verification.json).

## 2026-11-19 — Schedule audit and RealSense D455 room reconstruction

**Decision.** The strongest school result is the controlled Gaussian reduction and raw mesh extraction on public Room. The immediate delivery gap is the Week 11 cleaned mesh and demonstrated edit. For the personal scene, the 120k model can show the recorded route and the matte submap can show a local result. The evidence does not support metric room-wide placement, an improved whole-room mesh, or a unique explanation for the display failure. Work on the public mesh and a bounded personal-scene diagnosis can proceed independently.

### What the schedule requires and what exists

The supplied `C:/Users/hi/Downloads/Schedule_3DGS_EN.xlsx`, sheet `Project Schedule`, A1:D15, governs this comparison. Its SHA-256 is `a7d88d7d0f2cac55d199dc348d27167844f69d7577aeb9ea7c41a33384a54a80`. Read all fourteen rows from the attachment. The archived workbook differs in B1 and D14. Status below reflects evidence available on 19 November, including catch-up work completed after the original dates. Supervisor activity outside the repository is unverified.

| Week ending | Evidence of work done | Remaining work or qualification |
|---|---|---|
| 1 · Sep 6 | Objectives, evaluation criteria, pipeline outline, repository and experiment log exist. | Weekly review routine documented. Meeting history unverified. |
| 2 · Sep 13 | Seven-paper reading matrix and comparison of surface methods, including Poisson and Marching Cubes. | Reading deliverable evidenced. |
| 3 · Sep 20 | Proposal, system diagram, method choice, controls and ablation plan. | Proposal deliverable evidenced. |
| 4 · Sep 27 | Selected public/personal scenes, capture plan and eight-slide supervisor briefing. | Last documented status says presentation had not occurred. Presentation, feedback and revisions still need evidence. |
| 5 · Oct 4 | Image and pose checks, splits, preprocessing and architecture for four public sets and the personal room. | Low-texture flags were reviewed. Physical pose accuracy in the personal room remains open. |
| 6 · Oct 11 | B00 gsplat baseline, configuration, checkpoint, held-out renders, times and setup. | Identify the baseline implementation in the thesis. |
| 7 · Oct 18 | Multiscene benchmarks, failure analysis and uniform TSDF extraction from optimized public Room checkpoints. | Raw surfaces show lost thin structures and disconnected pieces. |
| 8 · Oct 25 | Controlled reduction studies. A04 cut mean counts by about 24–35% across four scenes. A05-Abs was tested against A04 on three scenes. | Rendering and count benefits are scoped to these settings. Five of six A05-Abs comparisons passed its practical gate. |
| 9 · Nov 1 | Surface regularizer integrated and tested with and without reduction. | Stable training and lower depth curvature. Effect on topology is mixed. Absolute accuracy is undetermined. |
| 10 · Nov 8 | Matched four-arm Room experiment, candidate settings, extraction script and four raw PLY/OBJ pairs. | Tuning covered two weights on one public scene in this factorial. Retain A05-Abs with surface weight zero as the working candidate. |
| 11 · Nov 15 | Raw exports and before-cleanup images are available. **Overdue.** | Reproducible cleanup, defects/detail comparison and an actual editor import, edit, save and reopen. |
| 12 · Nov 22 | Public rendering/count comparisons and extraction measurements exist. Personal RGB-D models, diagnostics and viewer evidence exist separately. **In progress.** | Matched baseline/candidate evaluation on public and personal scenes, a common results table, timed stages and mesh usability. Report coverage and reference availability per scene. |
| 13 · Nov 29 | Progress report, figures, method and failure analyses exist. | Assemble full thesis draft, final comparisons and reproducibility instructions. Record supervisor review. |
| 14 · Dec 6 | Component scripts, launchers, samples and briefing exist. | Representative input-to-edited-mesh rerun, packaged source/configs/checkpoints/assets, final thesis, presentation and demo. |

Sources for the table include the [project plan](SCHOOL_PROJECT_PLAN_20261109.md), [literature matrix](SCHOOL_LITERATURE_MATRIX_20261109.md), [dataset checks](SCHOOL_DATASET_QA_20261109.md), [supervisor materials](../report/school_completion_20261109/README.md), [four-scene research results](../results/research_completion_20261027/REPORT.md), [A05-Abs report](A05_ABSGRAD_RESULTS_20261107.md), [catch-up record](SCHOOL_COMPLETION_RESULTS_20261109.md) and [progress report](../report/progress_report/research_progress.pdf).

**Public mesh checkpoint.** Public Room and the museum DB3 are different scenes. On public Room, seed 42 at 7,000 updates, B00 reached 29.998 dB with 831,326 Gaussians. A05-Abs reached 30.737 dB with 452,485. All four PLY files match their recorded hashes and all four OBJ files are present. The reduced raw mesh has 119,987 triangles, 1,162 connected components and 16,807 boundary edges. Applying the tested surface prior raised the latter two counts to 1,216 and 17,700. In the five-view sheet, the large furniture remains recognizable while plant leaves and tabletop detail break up or merge. These give specific features to check after cleanup. The personal room also has experimental meshes made from RGB-D points. They use a different input and cannot fill the personal optimized-Gaussian mesh comparison. [Four-arm results](../results/school_completion_replay_20261109/REPORT.md) · [Reduced mesh preview](../results/school_completion_replay_20261109/meshes_uniform/reduced/mesh_depth_comparison.jpg) · [Personal depth-cloud mesh](../results/room_improve_sdk_dense_test_mesh_20261103/report.json)

### Where the DB3 pipeline works and where it breaks

1. **Capture and baseline.** Opened the original SQLite recording read-only. Device metadata identifies a RealSense D455, RGB8 at 1280 × 720 and Z16 at 848 × 480, both at 30 FPS, with 0.001 m per depth unit. The full 17,129,000,960-byte file still hashes to `60e3f253051fc282a61af7f6c9c5eef40830a12d42566b9eba0992463115bc52`. The baseline spans 158.69 seconds across 477 selected cameras. Its 417 training and 60 historical evaluation views produced a 120k-update model with 1,063,961 Gaussians and reported 23.11 dB PSNR. Those 60 views have been repeatedly inspected. Existing R5 tests verified replay and free-camera restoration. Both checkpoint seals passed again today. The context model retains its historical camera/input convention. The matte model has its own local coordinates. [Frozen baseline](DB3_R0_R1_RESULTS_20261110.md) · [Viewer and delivery evidence](DB3_R4_RECOVERY_R5_START_20261112.md)
2. **Projection and physical alignment.** A concrete software defect was addressed by reproducing the pinned RealSense SDK inverse-Brown projection and regenerating 477 candidate images. At the kiosk and opaque box the median distance from the RGB silhouette to the observed depth band fell from 7 and 8 px to overlap. The bands extend up to 9 and 14 px and include holes. They cannot certify two-pixel accuracy. Far doorways, glass, occlusions and thin rails have little or ambiguous depth. Keep missing depth unknown and use the matching projection, intrinsics, crop and camera poses throughout each comparison. [R1 numerical and physical checks](DB3_R0_R1_RESULTS_20261110.md)
3. **Local poses and global placement.** The 6–14 s matte submap registered 92/92 fitting views at about 10 FPS and supports a narrowly accepted local appearance/consistency result. Even the 64–72 s exhibit registered 88/88, yet its fixed statue/post, wall, backdrop and floor checks prevented promotion. Distributed revisits yielded zero accepted global loops. Recounting the saved edges gave 132 passes among 317 eligible intervals (41.6%). All 185 failures had native-rate retries. Of those, 108 failed a link, 72 completed the chain with insufficient endpoint checks, and five had enough endpoint matches but excessive error. None recovered under the full criteria. The other 159 intervals were reserved. These counts measure temporal links and say nothing directly about percentage of room surface reconstructed. [Local and revisit results](DB3_R2_R3_RESULTS_20261110.md) · [Route decisions](DB3_R4_RESULTS_20261111.md)
4. **Surface support and training.** The 90k initializer cap loses real support on one middle post. Removing the cap raised coverage there from 85.4% to 100% and reduced conditional median depth difference from 0.470 m to 0.036 m. On the first backdrop, dense supported input raised coverage to 52.5% while conditional median disagreement reached 0.205 m. The successful post change cannot be generalized across the exhibit. The 7k and 15k exhibit trials still failed fixed structure checks. Use coverage and residuals together on named structures before changing point selection, pruning floaters or increasing training length. [Structure-by-structure support audit](DB3_R0_R3_CLEANUP_20261111.md)
5. **Why the 120.7 s display still fails.** This is a separate window from the 64–72 s exhibit. Direct measured-depth pose fitting gave endpoint reprojection median/p90 of 3.050/7.401 px against limits of <2/<4 px. Matte passed at 0.499/1.910 px. Learned matching worsened the retained checks. A depth-free RGB epipolar check also had a 6.960 px target p90, so depth error alone is a weak explanation for the entire failure. The later masking study used additional reservations. Its own unmasked target scored 3.466/7.323 px, narrow glare masking 3.420/7.617 px and matched-count control 3.337/7.203 px. Broad masking lost initialization. The possible-person mask touched zero fitting tracks. Printed details behind glazing can track visually while the depth comes from another surface. That hypothesis requires checking which physical surface each observation belongs to. [Direct-depth trial](DB3_R5_PREREQUISITE_REVIEW_20261111.md) · [RGB and learned checks](DB3_R4_RECOVERY_R5_START_20261112.md) · [Narrow mask results](../results/db3_observation_masks_narrow_20261117/result.json)

### Fresh comparison of the two diagnostic windows

Joined each native RGB source ID in the 17 November narrow-mask protocol to the corresponding RGB message ID in the sensor timing table. All 4,764 timing IDs were unique. All 15 target and 11 control observations matched. Values below summarize the signed depth-minus-color midpoint gap and the existing focal-length × gyro-magnitude × absolute-gap estimate, in prepared-image pixels.

| Source window | Median midpoint gap | Median nearest gyro | Estimated rotation during gap, median / maximum | Depth exposure |
|---|---:|---:|---:|---:|
| Display, 120.678–121.145 s, 15 frames | −2.431 ms | 0.301 rad/s | 0.314 / 0.670 px | 4.961 ms |
| Matte, 12.015–12.348 s, 11 frames | −3.377 ms | 0.137 rad/s | 0.198 / 0.225 px | 6.853–7.025 ms |

The failing target had a smaller absolute median exposure gap. Its simple rotation scale is subpixel even though the endpoint reprojection error has a several-pixel tail. The two statistics describe different motions and must not be subtracted. This comparison weakens the case for a single constant timing shift as the first fix. It leaves translation, motion during exposure, calibration, tracking, occlusion and wrong-surface depths unresolved. Across the full recording the nearest midpoint pairing agreed with the existing pairing for every observation. The identified D455 uses global shutters for RGB and stereo depth. A row-wise rolling-shutter correction has little justification for this hardware. [Window IDs](../results/db3_observation_masks_narrow_20261117/protocol.json) · [Per-frame timing rows](../results/db3_r0_r1_20261110/sensor_timing_pairs.csv) · [RealSense D400-series datasheet, D455 features](https://www.realsenseai.com/wp-content/uploads/2026/03/RealSense-D400-Series-Datasheet-Mar-2026.pdf)

### Priorities and decisions for the next run

1. **Close Week 11 on public Room first.** Use the verified reduced PLY as a fixed input. Save a separate cleaned copy, the reproducible cleanup parameters, time and before/after counts for triangles, components, boundary and nonmanifold edges. Inspect the same table, plant and furniture silhouettes before and after simplification. In a design editor select and change one identifiable surface, save, close, reopen, and capture the persistent edit with its scale and orientation. This gives a deliverable even while personal-scene geometry remains open. [Raw PLY](../results/school_completion_replay_20261109/meshes_uniform/reduced/raw_mesh.ply)
2. **Run one discriminating DB3 test on the existing footage.** Start with the same display and matte windows and keep their original checks. Independently annotate repeated observations of opaque corners with unambiguous stable depth, distributed over image position and range. Mark glazing, depth jumps and unknowns explicitly. Obtain a second blind read or repeated annotation to estimate uncertainty. Freeze IDs and exclusion patches before fitting. The earlier added static target checks had only 14 endpoint pairs and under 5% hull coverage. Report insufficient support if the existing count/coverage gates cannot be met. [Source frames with mask overlays](../results/db3_observation_masks_narrow_20261117/target/mask_preview.jpg)
3. **Use the same landmarks to distinguish causes.** Record manual-vs-tracker displacement, RGB-only epipolar error, symmetric measured-depth reprojection, signed depth differences, spatial support and medians/p90 for endpoint and both half-pairs. Separate opaque and reflective observations and report missing depths. A reliable RGB match with a depth residual concentrated on glass motivates a visibility/depth association change. Large RGB-only errors on opaque points motivate checking correspondence and pose first. A repeatable signed error on opaque surfaces that varies with motion motivates a measured motion-compensation trial with clock/extrinsic checks. Retain the original guards, including <2/<4 px, and the matte control. Confirm any gain on a separately reserved region or revisit before attempting global placement or training. Preserve the original baseline and reservations, including the 377 fitting-eligible selected frames.
4. **By Nov 22, assemble the Week 12 comparison.** Put baseline and A05-Abs on identical inputs, camera sequence, split, training budget and extraction settings within each scene. Start with public Room and a clearly labeled personal matte subset, then report coverage of the remaining scheduled scenes and any unexecuted arms. Include PSNR/SSIM/LPIPS, count, training/extraction time, mesh defects, edit outcome and measured-depth coverage. Use physical error metrics only where an independent reference exists. The personal room's held-out sensor depth supports a consistency check. Document the depth-assisted personal input as a separate condition from the public photo-only pipeline. A local result leaves the full personal-room comparison partial.
5. **By Nov 29 and Dec 6, respectively,** assemble the thesis with these negative results and obtain supervisor feedback, then rerun and package a representative input-to-edited-mesh case with portable instructions, checkpoints or retrieval details, samples, presentation and demo. Large results live outside Git and the 17 November mask scripts/annotations were untracked at audit start. Include their provenance in the reproducibility package.

**Audit scope.** Read the supplied schedule, local reports, current saved machine results, images, raw metadata and relevant artifact inventories on branch `depth` at `3dfa0918`, including existing uncommitted November 17 work. Rehashed the full DB3 and four public PLYs, ran both R5 launcher seal checks, recounted the route and joined the per-window timing data. Historical training metrics, geometric fits and browser checks are attributed to the saved runs. This audit did not rerun training, refit cameras, test live playback, change the workbook or perform an editor round trip.

## 2026-11-19 — Week 11 public Room cleanup and design edit completed

Used the reduced A05-Abs public Room uniform-TSDF PLY from Week 10. Its SHA-256 matched the extraction report and a fresh topology recount before cleanup, then matched again after the run. SHA-256 `af370bb141609c651a48dc48fb7531c0574aa674657b370aa214d26d4c7e46fa`. The cleaned PLY is a separate binary file with vertex colors and recomputed normals. [Fixed source PLY](../results/school_completion_replay_20261109/meshes_uniform/reduced/raw_mesh.ply) · [Extraction report](../results/school_completion_replay_20261109/meshes_uniform/reduced/report.json) · [Cleaned PLY](../results/week11_public_room_20261119/room_cleaned.ply)

### Cleanup and timing

Open3D 0.20.0, NumPy 2.5.2, CPU AMD Ryzen 5 3500. Remove repeated-index and duplicate faces, then drop edge-connected components with **both** fewer than 8 faces and area below one quarter of a voxel squared (`0.00014630191656511053` scene units squared, from the extraction voxel size). This removed 341 components and 533 triangles. Compact unreferenced vertices. Run quadric decimation with target 65% of filtered faces (77,645 triangles), boundary weight 8 and Open3D's default unbounded maximum error. Clean duplicate and degenerate faces, compact again, recompute normals, save and reload. The initial decimator output contained 20 pairs with opposite winding. Blender's PLY importer dropped the redundant face from each pair. The final recipe removes these explicitly by sorted vertex triple, retaining one face per pair and giving both Open3D and Blender the same triangle count. An earlier comparison attempt corrected a local dataset camera-key mismatch.

| Stage | Vertices | Triangles | Components | Boundary edges | Nonmanifold edges |
|---|---:|---:|---:|---:|---:|
| Fixed raw PLY | 68,867 | 119,987 | 1,162 | 16,807 | 0 |
| After filtering | 67,705 | 119,454 | 821 | 15,674 | 0 |
| Cleaned PLY, reloaded | 45,508 | 77,624 | 748 | 13,386 | 0 |

Components are edge-connected triangle sets. Boundary and nonmanifold edges have one and more than two incident faces, respectively. Triangle count fell 35.3%. A single run spent 0.222 s in filtering, 0.618 s in simplification and final in-memory cleanup, and 0.019 s writing the PLY, for **0.860 s** total across those stages. Input loading, topology measurements, reload and view rendering are outside that timing. The clean PLY SHA-256 is `e8b7a542fdbb7c0914459c83e322501d6d8f374c2dca754c6b9e87761b83c9f7`. A fresh run reproduced the clean PLY, camera JSON and silhouette PNG byte for byte. [Cleanup parameters, timings and counts](../results/week11_public_room_20261119/cleanup.json)

### Matched silhouette inspection

Rendered raw and cleaned meshes at the same three held-out camera poses using the same intrinsics, crops and per-view depth color ranges. The photographs identify each subject. Hit gains and losses count first-hit coverage anywhere in each crop, including the background. Shared-hit depth deltas compare the two meshes in normalized scene units.

| Subject and view | Observation | Lost / gained hit pixels | Shared-hit p95 absolute depth delta |
|---|---|---:|---:|
| Coffee table, val 0 `DSCF4667.JPG` | Broad tabletop edge and visible right leg remain similar. The clutter blends into the top in both meshes. | 0 / 0 of 96,292 | 0.00145 |
| Potted plant, val 28 `DSCF4891.JPG` | Upper and lower leaf clusters remain recognizable as coarse shapes. Both have fragmented or merged leaves, with small changes around foliage and nearby holes. | 374 / 17 of 227,040 | 0.00092 |
| Armchair and footstool, val 19 `DSCF4819.JPG` | Chair back, arms and footstool outline remain recognizable. The thin footstool supports are already incomplete in the raw surface. | 4 / 6 of 151,548 | 0.00125 |

[Silhouette comparison](../results/week11_public_room_20261119/silhouette_comparison.png) · [Camera and crop records](../results/week11_public_room_20261119/silhouette_cameras.json)

### Blender edit and reopen verification

In portable Blender 4.5.4 LTS (build `b3efe983cc58`), imported the cleaned PLY at scale 1, forward Y and up Z. Using the Blender Python API, selected an edge-connected patch on the front-left coffee tabletop in `DSCF4667.JPG` and saved the selection as the `Coffee table | selected surface` vertex group. Selection used XY center `(0.006, 0.439)`, outer radius `0.056`, height band `(0.185, 0.216)`, and vertex normal Z greater than `0.55`. The `Tabletop lift (+Z)` shape key moved 14 vertices along +Z with peak displacement **0.040 normalized scene units** and cosine taper from radius `0.027` to `0.056`. The selected vertices span about `0.078 × 0.081` in XY. Their mean pre-edit normal is approximately `(-0.144, 0.146, 0.979)`. Teal vertex colors mark the edited patch in the viewport. Both original vertex colors and the geometry before the edit are available in the Blender file.

Saved the file with the shape key at `1.000` and exited Blender. A fresh Blender process loaded the `.blend` and verified that the saved group and the 14 evaluated vertex displacements matched the edit, with all displacement confined to the selected vertices. It also verified the 77,624 triangles and the identity object transform (scale `1, 1, 1`, Euler rotation `0°, 0°, 0°`, translation `0, 0, 0`). The viewport capture is from a separate interactive reopen and shows the active shape key, axis gizmo and transform panel. Blender creation and save took 0.395 s, and checks plus three renders took 2.038 s after loading. [Edited Blender project](../results/week11_public_room_20261119/room_tabletop_edited.blend) · [Reopened viewport capture](../results/week11_public_room_20261119/blender_reopened_viewport.jpg) · [Before](../results/week11_public_room_20261119/tabletop_before.png) · [After with original colors](../results/week11_public_room_20261119/tabletop_geometry_after_reopen.png) · [After with edit highlighted](../results/week11_public_room_20261119/tabletop_after_reopen.png) · [Reopen verification](../results/week11_public_room_20261119/design_reopen.json)

A fresh Blender replay matched the selection, evaluated displacement, object transform and camera crop. Blender emitted a small memory-block warning during shutdown after saving, and both creation processes exited successfully. All evidence images decoded successfully and the original-color before and after captures show the raised profile.

**Decision.** Week 11 is complete for public Room. This establishes a repeatable cleanup and a persistent local geometry edit. The cleaned surface still has open boundaries and residual fragments, especially near the plant. The scene scale remains relative. The before/after mesh comparison measures preservation in the three inspected views. Personal-scene geometry remains open.

### Reproduce

Run from the repository root with the fixed input and the prepared `data/room_factor2` camera dataset. Use a fresh output directory. [Cleanup script](../scripts/week11_room_cleanup.py) · [Blender script](../scripts/week11_room_blender.py)

```powershell
$roomRun = 'results/week11_public_room_replay'
$roomBlender = '.\.tools\blender-4.5.4-windows-x64\blender.exe'
.\.venv\Scripts\python.exe scripts/week11_room_cleanup.py --output $roomRun
& $roomBlender -b --factory-startup --python-exit-code 1 --python scripts/week11_room_blender.py -- create $roomRun
& $roomBlender -b --disable-autoexec "$roomRun/room_tabletop_edited.blend" --python-exit-code 1 --python scripts/week11_room_blender.py -- verify $roomRun
```

The run used the portable Windows x64 ZIP from the RWTH Aachen Blender mirror, checked against the release SHA-256 list (`0de55df1d99e4e7152605022cb648e795d5d49209c5c5c4889e1a19fb401a054`). [Blender 4.5 LTS release](https://www.blender.org/releases/4-5/) · [Archive checksum list](https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.5/blender-4.5.4.sha256)

## 2026-11-19 — Week 12 matched B00 and A05-Abs comparison

Week 12 compares the standard B00 densifier with the frozen A05-Abs policy at seed 42 and 7,000 updates per arm. The two arms within each scene share the prepared images, cameras, initialization, split and full training-camera order. Both use packed rendering, batch size 1, SH degree up to 3, refinement steps 500–6,000 at 100-step intervals, and opacity reset every 3,000 steps. B00 uses the standard signed-gradient policy. A05-Abs uses absolute gradients with threshold 0.0008, budgeted growth, attributed-error and edge ranking, Long-Axis Split, and fallback disabled. The scheduled Gaussian ceiling is part of the candidate policy and is 700,000 on the public scenes and 180,000 on the local matte subset. Training budgets refer to updates. The final primitive counts and runtimes are measured outcomes.

The public Mip-NeRF 360 inputs are photo-only prepared factor-2 images and SfM cameras/points with every eighth registered image held out. Room, Counter and Kitchen reuse the completed seed-42 RGB runs. Room also reuses the two matched uniform-TSDF extractions from November 9. The public views and the personal matte split were exposed during earlier development. [Public dataset preparation](../data/room_factor2/dataset_manifest.json) · [Public Room paired audit](../results/week12_comparison_20261119/room_pair.json)

The personal result covers only the museum DB3 **local matte subset**, where the 21 fitting RGB frames span 6.008–13.817 s of the capture and three evaluation RGB frames lie within that interval. The prepared map retains 95 registered cameras. The explicit split selects 24 for Gaussian training and evaluation and 21 for extraction. Both new matte arms start from the same frozen R2 joint local cameras and 90,000 supported RGB-D points, replay all 7,000 camera indices, and use the same confidence-weighted sensor-disparity L1 at weight 0.01 on fitting frames. Reserved RGB frames were localized from RGB against fixed landmarks. Their depth messages are reserved for the consistency check. The original full-route and exhibit results remain separate from this local test. [Matte commands and hashed inputs](../results/week12_comparison_20261119/matte_training_protocol.json) · [Frozen split](../results/db3_r2_r3_20261110/matte/pilot/dataset/experiment_split.json) · [Initialization](../results/db3_r2_r3_20261110/matte/pilot/initialization_report.json) · [Local pose provenance](../results/db3_r2_r3_20261110/matte/rate10/evaluation_localization.json)

Extraction fuses only Gaussian expected depth above alpha 0.7, using up to 64 evenly spaced training views at quarter linear resolution. It uses a shared uniform cube around the common 1st–99th percentile initializer bounds enlarged by 20%, target voxels of 0.01 × scene scale, maximum grid size 192, and truncation four actual voxels. Raw extraction is the endpoint for topology and the common editing task. In Blender 4.5.4 LTS, the task imports each raw PLY at unit transform, selects an edge-connected patch within four voxels of a corresponding ray hit, applies a tapered offset toward the same camera with a peak of half a voxel, and verifies the evaluated geometry in a fresh process after reopening the saved project. The first Room prototype passed a NumPy index to Blender's vertex-group API and stopped before saving. Converting indices to Python integers resolved the issue and both Room documents passed. [Extraction implementation](../scripts/gaussians_to_mesh.py) · [Blender edit implementation](../scripts/week12_blender_edit.py) · [Room import and retry log](../results/week12_comparison_20261119/review/room/B00_blender_create.log)

### Primary matched results

RGB metrics are mean held-out PSNR, SSIM and AlexNet LPIPS from the final trainer evaluation. Training time uses the trainer loop timer through the final update and includes the initial evaluation. Extraction time includes checkpoint/loader setup, render/fusion, mesh export and internal mesh checks. Training wall time and the training/extraction camera identities are in each pair audit. Timings are observed single runs on the local GTX 1660 SUPER and should be read descriptively.

| Scene and split | Arm | PSNR (dB) | SSIM | LPIPS ↓ | Gaussians | Train (s) | Extract (s) | Triangles | Components | Boundary edges | Nonmanifold edges |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Public Room, 272 / 39 | B00 | 29.99812 | 0.901684 | 0.209726 | 831,326 | 424.57 | 8.62 | 119,613 | 1,241 | 18,165 | 0 |
| Public Room, 272 / 39 | A05-Abs | 30.73701 | 0.909146 | 0.192058 | 452,485 | 394.80 | 8.16 | 119,987 | 1,162 | 16,807 | 0 |
| Personal matte, 21 / 3 | B00 | 28.87233 | 0.883266 | 0.182086 | 485,156 | 214.55 | 4.40 | 80,952 | 973 | 13,300 | 0 |
| Personal matte, 21 / 3 | A05-Abs | 28.89107 | 0.885599 | 0.189637 | 179,792 | 190.58 | 4.51 | 99,107 | 1,448 | 21,023 | 0 |

Public Room recovered the archived **+0.739 dB** and **45.57% fewer Gaussians**, with higher SSIM, lower LPIPS and fewer raw mesh components and boundary edges. Both meshes preserve the broad tabletop and armchair shapes, merge objects on the table, and break up thin plant leaves and footstool supports. Both edits reopened with 108 displaced vertices at the shared coffee-table patch. The Week 11 A05-Abs cleaned mesh and its original design edit are separate deliverables. [Room pair and hashes](../results/week12_comparison_20261119/room_pair.json) · [Room raw-mesh views](../results/week12_comparison_20261119/review/room/mesh_comparison.jpg) · [B00 Blender project](../results/week12_comparison_20261119/review/room/B00_edited.blend) · [A05-Abs Blender project](../results/week12_comparison_20261119/review/room/A05_absgrad_edited.blend)

On the matte subset, A05-Abs used **62.94% fewer Gaussians**, gained **0.019 dB** and raised SSIM by 0.00233. LPIPS increased by 0.00755. Its raw mesh added 475 components and 7,723 boundary edges. The exhibit panel outlines appear in both meshes, with blurred text, openings along the bright doorway and fragmented display-case edges. A05-Abs shows a larger gap at the left of the doorway in `rgb_0006487.png`. Both edits reopened with the blue exhibit panel patch intact. B00 displaced 97 vertices and A05-Abs displaced 84, with the same requested peak offset. [Matte pair and hashes](../results/week12_comparison_20261119/matte_pair.json) · [Matte raw-mesh views](../results/week12_comparison_20261119/review/matte/mesh_comparison.jpg) · [B00 Blender project](../results/week12_comparison_20261119/review/matte/B00_edited.blend) · [A05-Abs Blender project](../results/week12_comparison_20261119/review/matte/A05_absgrad_edited.blend)

### Held-out measured-depth consistency

Public scenes have no measured-depth coverage or independent reference mesh in this experiment. The matte condition used 58,967 positive-weight fitting depth samples out of 86,016 sampled points. All three reserved depth message IDs are disjoint from fitting-depth IDs, and initialization uses only fitting source IDs. Out of 1,158,426 held-out image pixels, 977,442 (84.38%) carried valid measured depth between 0.3 and 6 m. A 3×3 range filter retained 918,331 stable pixels (79.27% of all pixels). Per-view counts follow. [Per-frame source IDs, weights and depth statistics](../results/week12_comparison_20261119/review/matte/sensor_depth_consistency.json)

| Held-out RGB frame | Depth message | Valid depth pixels | Stable depth pixels | Stable pixels covered by both Gaussian arms | Stable pixels hit by both meshes |
|---|---:|---:|---:|---:|---:|
| `rgb_0006487.png` | 6478 | 268,799 | 246,093 | 246,019 | 242,820 |
| `rgb_0008463.png` | 8456 | 366,433 | 356,706 | 356,706 | 355,862 |
| `rgb_0010545.png` | 10536 | 342,210 | 315,532 | 314,167 | 282,729 |

Coverage below is each output's fraction of the stable measured pixels with alpha above 0.7 and valid Gaussian depth, or a positive mesh first hit. The discrepancy statistics compare predicted camera-Z with the held-out aligned sensor depth on the exact pairwise shared pixels. Metre values use the inherited local RGB-D scale. They measure agreement with the same device and estimated poses, and the experiment has no independent physical accuracy reference.

| Output | Arm | Stable measured-depth coverage | Pixels shared by the pair | Median signed discrepancy (cm) | Median absolute (cm) | P90 absolute (cm) | Mean absolute (cm) |
|---|---|---:|---:|---:|---:|---:|---:|
| Expected Gaussian depth | B00 | 99.95% | 916,892 | −6.64 | 7.37 | 21.62 | 11.60 |
| Expected Gaussian depth | A05-Abs | 99.88% | 916,892 | −7.33 | 7.97 | 21.25 | 11.20 |
| Raw mesh first hit | B00 | 97.30% | 881,411 | −8.32 | 8.38 | 20.20 | 11.69 |
| Raw mesh first hit | A05-Abs | 97.82% | 881,411 | −9.34 | 9.47 | 20.28 | 11.66 |

For matte mesh depth, median absolute discrepancy rose in all three views. P90 fell in the first and third views and rose in the second. The meshes hit 95.98% of stable measured pixels in common. The candidate mesh gains a small amount of measured-depth coverage while increasing fragmentation and worsening median agreement. These results support local Gaussian-count reduction, with mixed appearance and depth consistency. [Held-out depth records](../results/week12_comparison_20261119/review/matte/sensor_depth_consistency.json)

### Remaining scheduled scenes

All rows below use the public photo-only condition. Counter and Kitchen reuse both archived RGB runs and add fresh matched raw extractions and Blender checks. Bonsai reuses its archived B00 training and adds the previously missing A05-Abs arm, fresh paired extraction and Blender checks. The Bonsai candidate explicitly replays the archived B00 camera order with the same 7,000 updates and uses zero loader workers to stay within RAM. The archived B00 used four workers, which limits the interpretation of its training-time comparison. [Bonsai training protocol](../results/week12_comparison_20261119/bonsai_training_protocol.json)

| Scene and split | Arm | PSNR (dB) | SSIM | LPIPS ↓ | Gaussians | Train (s) | Extract (s) | Triangles | Components | Boundary edges | Nonmanifold edges |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Counter, 210 / 30 | B00 | 27.62975 | 0.888071 | 0.195367 | 742,755 | 441.71 | 13.28 | 158,759 | 2,606 | 33,559 | 0 |
| Counter, 210 / 30 | A05-Abs | 27.83983 | 0.889544 | 0.188124 | 381,228 | 432.75 | 10.72 | 163,497 | 2,443 | 33,701 | 0 |
| Kitchen, 244 / 35 | B00 | 29.91561 | 0.915642 | 0.116612 | 1,082,362 | 560.18 | 16.27 | 70,332 | 661 | 6,446 | 0 |
| Kitchen, 244 / 35 | A05-Abs | 30.21228 | 0.915189 | 0.115207 | 502,536 | 469.80 | 12.96 | 82,518 | 944 | 9,148 | 0 |
| Bonsai, 255 / 37 | B00 | 30.05508 | 0.923948 | 0.160054 | 1,020,333 | 449.61 | 16.65 | 322,809 | 5,364 | 66,045 | 0 |
| Bonsai, 255 / 37 | A05-Abs | 30.39079 | 0.928378 | 0.152843 | 463,678 | 398.83 | 16.83 | 320,840 | 4,859 | 64,084 | 0 |

Counter used 48.67% fewer Gaussians with A05-Abs. The main counter profile is visible in both meshes. The bowl rims, clustered objects, plant leaves and far right edge show merging and gaps. Both geometry edits reopened with 97 displaced vertices at the same scene patch near the fruit. [Counter pair audit](../results/week12_comparison_20261119/counter_pair.json) · [Counter raw-mesh views](../results/week12_comparison_20261119/review/counter/mesh_comparison.jpg) · [B00 Blender project](../results/week12_comparison_20261119/review/counter/B00_edited.blend) · [A05-Abs Blender project](../results/week12_comparison_20261119/review/counter/A05_absgrad_edited.blend)

Kitchen used 53.57% fewer Gaussians with A05-Abs. The toy bulldozer is recognizable in both meshes, with fused tread details, rounded thin parts and outer background areas removed by the extraction crop. The candidate has more components and boundary edges. Both toy-tread edits reopened, with 135 displaced vertices for B00 and 142 for A05-Abs. [Kitchen pair audit](../results/week12_comparison_20261119/kitchen_pair.json) · [Kitchen raw-mesh views](../results/week12_comparison_20261119/review/kitchen/mesh_comparison.jpg) · [B00 Blender project](../results/week12_comparison_20261119/review/kitchen/B00_edited.blend) · [A05-Abs Blender project](../results/week12_comparison_20261119/review/kitchen/A05_absgrad_edited.blend)

Bonsai used 54.56% fewer Gaussians with A05-Abs and gained 0.336 dB PSNR. Both meshes fuse individual blossoms into a few large surfaces and show extensive openings and broken bicycle detail in the background. A05-Abs has fewer components and boundary edges. Both edits near the planter base reopened, with 93 displaced vertices for B00 and 94 for A05-Abs. [Bonsai pair audit](../results/week12_comparison_20261119/bonsai_pair.json) · [Bonsai raw-mesh views](../results/week12_comparison_20261119/review/bonsai/mesh_comparison.jpg) · [B00 Blender project](../results/week12_comparison_20261119/review/bonsai/B00_edited.blend) · [A05-Abs Blender project](../results/week12_comparison_20261119/review/bonsai/A05_absgrad_edited.blend)

Across the four public scenes A05-Abs reduced count by 45.57–54.56% and raised PSNR by 0.210–0.739 dB. Mesh topology changed in both directions across scenes. The local editing task passed after reopening all ten projects across the public and matte conditions. Each pair audit matched the initialization hash, all 7,000 ordered camera indices, the final RGB evaluation targets pixel for pixel, and the list of extraction cameras. The first-update RGB metrics were within 0.00001 in every pair.

### Unexecuted arms and limits

The **full personal-room comparison remains partial**. Both matched full-route arms and both matched exhibit-only arms were unexecuted. The archived full-route context model and exhibit depth-policy experiments are available as separate evidence. They use other inputs, policies or schedules. A matched photo-only personal matte pair and a personal surface-prior pair were also unexecuted. Room already has the optional archived B00-plus-prior and A05-Abs-plus-prior ablation. The other public scenes used the two primary arms. All Week 12 results use one seed on development views. Independent geometric reference coverage is zero across the comparison. [Full-route delivery status](DB3_R4_RESULTS_20261111.md) · [Prior local experiments](DB3_R0_R3_CLEANUP_20261111.md) · [Optional Room surface-prior ablation](../results/school_completion_replay_20261109/REPORT.md)

### Reproduce and inspect

Run from the repository root with the prepared datasets and source campaign archives. The wrapper sets up the local CUDA/MSVC environment. Use a fresh output root to repeat training. The `extract` stage checks archive and run hashes, exact camera replay, RGB targets and split sizes before matching the TSDF settings and camera identities. The `edit` stage starts separate Blender processes for creation and reopening, checks topology counts and records the saved project hashes. [Comparison runner](../scripts/week12_comparison.py) · [Mesh review and depth evaluation](../scripts/week12_mesh_review.py) · [Blender round trip](../scripts/week12_blender_edit.py)

```powershell
$week12 = 'results/week12_comparison_replay'
.\scripts\run-school-completion.ps1 -u scripts/week12_comparison.py train --root $week12 --scene matte
.\scripts\run-school-completion.ps1 -u scripts/week12_comparison.py train --root $week12 --scene bonsai
foreach ($scene in @('room', 'matte', 'counter', 'kitchen', 'bonsai')) {
    .\scripts\run-school-completion.ps1 -u scripts/week12_comparison.py extract --root $week12 --scene $scene
    .\scripts\run-school-completion.ps1 -u scripts/week12_mesh_review.py --root $week12 --scene $scene
    .\.venv\Scripts\python.exe -u scripts/week12_comparison.py edit --root $week12 --scene $scene
    if ($LASTEXITCODE -ne 0) { throw "Blender round trip failed for $scene" }
}
```

## 2026-11-19 — Project closeout status and DB3 scope

The user deferred full-scene reconstruction from the personal DB3 recording and confirmed it is optional for current closeout. The depth-assisted matte experiment remains a bounded result. The technical evidence through Week 12 covers the matched four-scene public comparison and the editable public Room mesh. Public Room is the representative photo-to-edited-mesh case for the remaining deliverables. Geometry claims remain limited to observed defects and consistency because the comparison has no independent physical reference. [Public Room cleanup and edit](#2026-11-19--week-11-public-room-cleanup-and-design-edit-completed) · [Week 12 comparison](#2026-11-19--week-12-matched-b00-and-a05-abs-comparison)

**Remaining for submission.** By November 29, update the November 9 progress report into the full thesis draft with the Week 11 and 12 evidence, reproducibility instructions, and supervisor review. An eight-slide briefing exists, and the last documented status leaves presentation and feedback pending. By December 6, verify a representative public Room rerun from prepared photos through the edited mesh, package the source/configuration, checkpoint or retrieval instructions, and sample assets, and finish the thesis, presentation and demo. This status review used the schedule and repository evidence. [Progress report](../report/progress_report/README.md) · [Existing supervisor briefing](../report/school_completion_20261109/README.md) · [Schedule and deliverables](SCHEDULE_PROGRESS_AUDIT_20261109.md)


## 2026-11-19 — Illustrated final report prototype

Wrote the five-chapter [Markdown draft](../report/final_report/report_draft.md) first, then typeset the [editable LaTeX prototype](../report/final_report/final_report.tex) and [17-page PDF](../report/final_report/final_report.pdf). The manuscript follows the proposed academic structure, the university cover, running header and page format in [the report template](../report/samplereport.tex), and the project title and personnel confirmed in the existing [progress report](../report/progress_report/research_progress.tex). Public Room is the representative photo-to-edited-mesh case. The optional full-route museum reconstruction, independent geometry validation, clean release rehearsal and supervisor feedback remain open.

**Evidence and illustrations.** Reconciled the four public and one local matte Week 12 pair audits with the Week 11 cleanup and reopen records, the earlier paired-seed and surface-prior experiments, and the DB3 pose reviews. The manuscript contains seven tables, thirteen project figures and eleven literature references. The illustrations include an input-to-edit summary, Gaussian depth and split diagrams, scene photographs, identical held-out crops, raw-mesh failure examples, cleanup comparisons and the Blender edit. The four graphs show count versus PSNR, Room growth-event counts, sensitivity to seed, and signed changes in fragmentation. Figure organization draws on the quantitative/qualitative and ablation presentations in [AbsGS](https://arxiv.org/html/2404.10484v1) and [Improving Densification](https://arxiv.org/html/2508.12313v1). Research photographs and renders come from project archives. No training, mesh extraction or Blender experiments were rerun for this document.

**Measurement provenance.** Re-read all ten Gaussian checkpoint sizes and compared their SHA-256 hashes with the pair files. Computed render-call throughput as the reciprocal of the saved final mean synchronized render time. Peak tensor allocation and training time come from the verified training records. Reconstructed the growth graph from the recorded events and the seed comparison from the A04 and A05-Abs final metrics. The three reserved matte depth messages support same-device consistency. The public scenes have no independent surface reference in this experiment. [Week 12 pair records](../results/week12_comparison_20261119/) · [Room edit verification](../results/week11_public_room_20261119/design_reopen.json).

**Build and verification.** The earlier WSL build route could not start on this host, so Windows Tectonic compiled the native template. Corrected an unsupported dash, a wide table, a caption citation and a page break. Rendered and visually checked every final page, including the enlarged crops and relabeled cleanup panels. Verified the five chapters, all thirteen figure captions and local assets, seven tables, eleven bibliography entries, target evidence paths and the ten primary RGB/topology rows. Final build log had no overfull boxes, missing glyphs or unresolved citations/references. Rebuild from the final report directory with `tectonic final_report.tex`. PDF SHA-256 `c625901cc5d8d82fdb969e0f93240e1c31af23588c4e562c35da9551318bf97e`.

## 2026-11-19 — Final report review, controlled follow-up and reproduction

**Review and scope.** The feedback asked for a full technical report, implementation-level derivations and pseudocode, clearer novelty and claims, component-level controls, matched wall-clock and budget results, extraction sensitivity and a reproducible editing path. Preserved the 17-page prototype and its assets in `results/final_report_prototype_20261119`. Its recorded PDF digest matched before revision. Adopted the suggested report title *Budget-Aware Densification for Efficient 3D Gaussian Splatting: Surface Reconstruction and Editable Mesh Evaluation*. The project preserves the archived four-scene public and local matte evidence. Absolute accuracy and global museum reconstruction still require separate evidence.

**New protocol.** Fixed Room, seed 42, 7,000 updates, exact October B00 fitting-index replay, 39 identical validation targets and zero data workers. Reran B00 and the 700k A05-Abs reference, A05-Abs at 400k and 1M caps, RGB-error and edge score removals, and the other three cells of a signed/absolute gradient × 0.0002/0.0008 threshold factorial. The older three-seed A04 factorial already isolates the split operator bundle from clone replacement and the archived A04 tuning screen already tests offset/opacity. Held those choices fixed in the new gradient and ranking runs. [Study runner](../scripts/final_report_study.py) · [Frozen commands and source hashes](../results/final_report_study_20261119/protocol.json).

**Wall-clock measurements.** Added opt-in timed evaluation to the normal headless CUDA trainer. The two reference arms use synchronized probes at 90, 180 and 270 seconds, checked every ten updates with a five-second timing tolerance, plus update milestones. The stopwatch excludes interleaved validation and starts after initialization. Intermediate probes write statistics. Final evaluation still writes the comparison PNGs. The actual deadline samples were within 0.46 seconds of target. At the three deadlines A05-Abs gained 0.099, 0.261 and 0.513 dB with fewer Gaussians. At 90 seconds its SSIM and LPIPS were slightly worse. At 270 seconds B00 had 781,109 splats and 29.723 dB after 5,390 updates, while A05-Abs had 432,794 splats and 30.236 dB after 5,290 updates. At 7,000 matched updates the replay was 30.125 dB/834,493 Gaussians for B00 and 30.757 dB/452,409 for A05-Abs. This uses one seed and previously inspected evaluation views. Exact endpoint values differ from those in the original archive.

**Independent inputs and topology readback.** Fetched the official Room factor-two data into a fresh directory through ZIP byte ranges. SHA-256 comparisons matched all 311 photographs, the three upstream COLMAP binaries and all five prepared sparse-model files. Preserved the ordered indices and package versions with the report. Open3D readback confirmed all ten archived raw PLYs and the cleaned Room PLY against their recorded digests and counts. All eleven have open boundaries and fail vertex-manifoldness, despite zero edges incident to more than two triangles. All pass the orientability check. The largest edge-connected component accounts for 82.8–95.6% of raw mesh area and 94.1% after Room cleanup. Prior connected-patch edits remain a separate verified capability. [Fresh-input audit](../report/final_report/evidence/room_input_audit.json) · [Topology audit](../results/final_report_study_20261119/topology_audit.json).

**Per-view RGB audit.** Decoded the 147 archived final PNG pairs and verified all target halves against their recorded hashes. Candidate PSNR improved on 34/39 Room, 26/30 Counter, 21/35 Kitchen, 20/37 Bonsai and 1/3 matte views. The largest Bonsai gain was +13.317 dB on DSCF5589.JPG. The B00 rendering shows a broad dark cast. Removing this view for a sensitivity check changes the 8-bit mean from +0.337 to −0.023 dB. The primary float mean retains all views. Reported the paired distribution and illustrated the largest gain and regression separately. [Per-view evidence](../report/final_report/evidence/per_view_rgb_audit.json).

**Cap sweep and added ranking control.** The 400k A05-Abs cap constrained 33/54 events and yielded 399,819 Gaussians at 30.706 dB. The 700k and 1M ceilings never required selection among eligible parents, and final counts were 452,409 and 449,350 at 30.757 and 30.748 dB. In the nonbinding regime, changing scores cannot change which parents are selected if the candidate set is fixed. Added a second two-arm run at 400k, individually removing RGB-error and edge ranking while retaining the existing 400k full-score arm. The supplement uses the same training code and audited input order, with its own protocol and output root.

**Completed Room training.** All 11 arms finished. The final audit compared the frozen source snapshots, actual configurations, finite post-update checkpoints, exact initializer hashes, ordered 7,000-camera replay, first-update observations and 39 target RGB hashes. All budget arms recorded 54 events with count accounting and scheduled limits satisfied. The two timed arms reached each declared deadline. Scores below are native trainer results at update 7,000.

| Arm | PSNR (dB) | SSIM | LPIPS | Gaussians | Binding events |
| --- | ---: | ---: | ---: | ---: | ---: |
| B00 | 30.12485 | 0.902320 | 0.209753 | 834,493 | — |
| A05-Abs, 700k | 30.75720 | 0.909082 | 0.191663 | 452,409 | 0 |
| A05-Abs, 400k | 30.70610 | 0.908434 | 0.194161 | 399,819 | 33 |
| A05-Abs, 1M | 30.74806 | 0.909168 | 0.192005 | 449,350 | 0 |
| No RGB-error cue, 700k | 30.75036 | 0.908954 | 0.191726 | 453,277 | 0 |
| No edge cue, 700k | 30.71686 | 0.909172 | 0.191886 | 449,470 | 0 |
| Signed 0.0008, 700k | 29.66828 | 0.889442 | 0.240399 | 143,086 | 0 |
| Absolute 0.0002, 700k | 31.08402 | 0.917054 | 0.172353 | 699,645 | 54 |
| A04 signed 0.0002, 700k | 30.59667 | 0.905248 | 0.203047 | 603,070 | 4 |
| No RGB-error cue, 400k | 30.68800 | 0.907958 | 0.195019 | 399,817 | 33 |
| No edge cue, 400k | 30.73913 | 0.908342 | 0.194647 | 399,829 | 32 |

At the low and high thresholds, absolute aggregation raised PSNR by 0.487 and 1.089 dB respectively. In the absolute arm, raising the threshold reduced count by 247,236 and PSNR by 0.327 dB. The high-threshold signed control left substantial unused capacity and lost PSNR against B00. At 400k, removing the error cue lost 0.018 dB and removing the edge cue gained 0.033 dB, with slightly worse LPIPS in both. These are single-seed, development-view outcomes. The small binding-cap ranking differences do not support a general benefit for either cue. [Main audit](../results/final_report_study_20261119/final_audit.json) · [Tight-cap protocol](../results/final_report_ranking_20261119/protocol.json) · [Portable measured results](../report/final_report/evidence/room_study.json).

**TSDF resolution.** Added an optional fixed truncation argument to the uniform extractor. The default remains four realized voxels. The historical Room comparison now validates the extractor snapshot against its recorded SHA-256. The existing Room archive route passed its regression check. Reused the 192³ matte meshes and freshly generated 128³ and 256³ for each arm. Verified the same two checkpoint hashes, crop, 21 fitting cameras, quarter-resolution inputs, opacity threshold and 0.17255 normalized truncation distance. The depth audit jointly required valid predictions from both Gaussian renderers and all six meshes on the same three reserved sensor frames. It retained 856,383/918,331 stable pixels. The A05-Abs median absolute depth discrepancy exceeded B00 by 0.70 cm before fusion and 1.05–1.09 cm across the mesh resolutions. Within either arm, the resolution sweep moved the median by less than 0.05 cm. The candidate's open boundary edges grew from 9,061 at 128³ to 36,040 at 256³ while its largest component's share of area rose from 66.0% to 92.1%. Grid spacing strongly affected topology. The persistent depth gap is measured relative to the same-device sensor and poses. The shared-support results and diagnostic images preserve this distinction. [Grid protocol](../results/final_report_geometry_20261119/protocol.json) · [Grid results](../report/final_report/evidence/matte_grid.json).

**Fresh public Room rehearsal.** Used the fresh 452,409-Gaussian checkpoint from the controlled replay and the prepared inputs verified against the upstream download. The extractor fused 64 fitting cameras, produced 119,061 triangles, and wrote raw PLY and OBJ. Open3D warned that the OBJ export cannot include triangle normals. The readback and cleanup used the PLY. Generalized the Room cleanup input to accept the fresh extraction directory and verify its report and mesh hash. Cleanup saved 76,971 triangles, 779 components and 13,559 boundary edges. Blender 4.5.4 saved the design and verified it in a separate process. The 12-vertex patch retained a +0.0400000066 Z displacement in normalized units. Existing workstation software was used. A fresh installation on another machine was outside this rehearsal. [Raw extraction](../results/final_report_room_replay_20261119/raw/report.json) · [Reopened Blender project](../results/final_report_room_replay_20261119/design/room_tabletop_edited.blend) · [Pipeline audit](../report/final_report/evidence/room_pipeline.json).

**Regression checks.** Focused CUDA tests passed, 28 total, covering the budget strategy and Long-Axis Split. The repository's existing unknown pytest `env` option produced one warning. Python compilation and `git diff --check` passed. No change to the default TSDF truncation formula or the default trainer evaluation output was needed for the timed/grid diagnostics.

**Final manuscript and delivery.** Kept the university cover and main benchmark evidence. Rewrote the introduction, theory and methodology around research questions, camera and splat equations, score normalization and gating, scheduled top-k refinement, child and optimizer state initialization, TSDF and the editor workflow. Added the matched-time and count curves, factorial and tight-cap ranking results, all-view sensitivity, fixed-grid depth diagnostics, and objective-by-objective conclusions. The released reproduction guide includes clean Windows setup commands, the frozen fitting order, preparation, training, extraction, cleanup, Blender reopen and expected outputs. The verified fresh rehearsal used the configured workstation.

The Tectonic build produced **42 A4 pages, 35 of main content**, with 18 figures, 22 tables and 13 bibliography entries. Rendered every page using Poppler, checked the final layout, audited 36 local source links and 204 annotation rectangles, and confirmed the reported timing, cap, mesh and edit values are present. The final TeX log contains no overfull boxes, missing glyphs or unresolved citations/references. PDF SHA-256 `f54d9b9a55c60bbaebf2e4f4acf964925ceec5cba8726a13ad25ec665d68112d`. [Revised PDF](../report/final_report/final_report.pdf) · [Editable LaTeX](../report/final_report/final_report.tex) · [Reproduction guide](../report/final_report/REPRODUCE.md) · [Archived prototype](../results/final_report_prototype_20261119/final_report.pdf).

## 2026-11-20 — Final report LaTeX build check

Confirmed `final_report.tex` is the entry point and the existing section files, figures and generated table rows are sufficient to build the document. In PowerShell, run `tectonic final_report.tex` from `report/final_report` to write `final_report.pdf` beside the source. The sandbox's isolated package cache lacked the Tectonic bundle. With access to the normal Windows environment, Tectonic `0.17.0+20260731` compiled the report in a temporary output directory and reran TeX when its auxiliary file changed. Poppler verified 42 A4 pages. The TeX log had no errors, warnings, bad boxes or missing glyphs. Tectonic's console did emit a Fontconfig configuration diagnostic and exited successfully. The existing report PDF's SHA-256 still matched the previous entry. No report sources, evidence files or research runs were changed. [Build instructions](../report/final_report/REPRODUCE.md)

## 2026-11-20 — Final report review, seed replication and outdoor evaluation

**Review and decisions.** Applied the pasted final-report review to the 42-page baseline. Preserved the earlier PDF at `results/final_report_review_20261120/prior_final_report.pdf`. Centered the manuscript on the measured trade-off between held-out RGB quality and Gaussian count. The controlled Room arm at 700,000 admitted every eligible parent, so neither top-k selection nor budget enforcement directly excluded any parent in that run. The high absolute threshold saved 247,236 primitives and cost 0.327 dB relative to the lower absolute threshold. The lower threshold was also capacity-limited. The 400,000-cap result measures a binding count constraint, and the opposing ranking ablations leave general benefit unresolved. The established full policy remains the comparison reference. Preserved Bonsai's primary all-view mean and its leave-largest-gain-out sensitivity check.

**Frozen follow-up.** Selected the previously unused Garden scene from the same public dataset family and two more Room seeds before inspecting their new evaluation outputs. Prepared Garden factor four from the official Mip-NeRF 360 archive using byte ranges. The prepared model contains 185 photos at 1297×840 and 138,766 SfM points. Every eighth COLMAP-sorted image is held out. Room remains factor two with its exposed 39-view set. Froze Garden seed 42 and Room seeds 43 and 44, B00 and A05-Abs, 7,000 updates, zero workers, matched initial tensors and full fitting order within each pair. Used the existing independent sampler with each newly selected seed, kept signed B00 and the selected absolute policy fixed, and used count and PSNR as the primary comparison with SSIM and LPIPS also reported. No threshold, ranking, split or cap setting was selected from the new target results. [Runner](../scripts/final_report_followup.py) · [Frozen protocol](../results/final_report_followup_20261120/protocol.initial.json).

**Verifier recovery.** Garden B00 completed training and exported evaluation outputs. The first verifier assumed default-trainer statistics included the budget-only growth history and stopped with a missing-key error. Preserved the original runner and protocol. The correction only changed verification and resumption logic. Checked unchanged trainer code, frozen commands and dataset fingerprints. Read the existing B00 configuration and validated the completed checkpoint and evaluation, then continued the planned runs without repeating that arm. Its recovered record omits the outer process-wall timestamp.

**Room appendix rehearsal.** Implemented an executable PowerShell entry point with stage-by-stage and all-in-one invocations, explicit trainer/extractor arguments, fresh output checks and failure propagation. Traced the six commands from the wrapper. Ran preparation in an isolated unpacked supplement using the official Room archive. Its 311 photographs and five prepared sparse-model files matched the project's files by SHA-256. Ran extraction through the wrapper with the archived matched Room checkpoint and reproduced all 119,061 triangles and the exact raw PLY digest `9a0618fa997e4518fee80e09c5d53d77386036dbfdcba3e42f08c5ad6353b249`. Open3D again warned that the OBJ cannot include triangle normals. Used a disposable copy of that raw surface and a record with the isolated prepared input path to test cleanup, edit and independent Blender reopen from the unpacked supplement. The clean mesh reproduced 76,971 triangles, 779 components and 13,559 boundary edges, and the 12-vertex patch retained the +0.0400000066 normalized-Z lift. The new rehearsal used the workstation toolchain. The fresh package installation still relies on the documented Windows/CUDA setup. [Runner](../report/final_report/reproduce_room.ps1) · [New extraction record](../results/final_report_review_20261120/cli_replay/raw/report.json) · [Unpacked Blender verification](../results/final_report_review_20261120/smoke_extract/results/replay_smoke/design/design_reopen.json).

**Garden result.** The previously unused factor-four Garden scene has 161 fitting and 24 held-out photographs. B00 finished at 26.03173 dB, 0.808161 SSIM, 0.153030 LPIPS and 2,863,759 Gaussians. A05-Abs finished at 26.39678 dB, 0.816615 SSIM, 0.146150 LPIPS and 699,712 Gaussians. The 700,000 schedule bound all 54 candidate growth events. The paired result gained 0.36506 dB with 75.5667% fewer Gaussians. A separate hash-checked audit of the saved 8-bit panels found 23 of 24 gains, median +0.318 dB, and a mean of +0.335 dB after removing the largest gain. Garden tests outdoor transfer within the same dataset family with one seed and supplied SfM. No geometry or edit conclusion is drawn for Garden. [Per-view audit](../results/final_report_followup_20261120/garden_per_view.json).

**Additional Room seeds.** The seed-43 B00/A05-Abs endpoints were 30.05362/30.72942 dB and 830,408/450,776 Gaussians. At seed 44 they were 30.08517/30.80050 dB and 821,803/449,923 Gaussians. At seeds 43/44 the paired SSIM differences were +0.007003/+0.006729, and LPIPS differences were −0.017444/−0.018232. Including the controlled seed-42 result gives +0.67449 ± 0.04150 dB paired PSNR (mean and sample SD) and 45.5848% mean count reduction (0.2906 percentage-point sample SD). All three Room candidate runs admitted every eligible parent. The final audit checked all six new checkpoint digests, final reported metrics, trainer configurations, seed-paired initialization and exact camera-index arrays, consistent target pixel digests for all arms and identical Room targets against the seed-42 control. All three new paired runs passed. [Verified pairs](../report/final_report/evidence/followup.json) · [Campaign directory](../results/final_report_followup_20261120/).

**Report and packaging.** Added the new paired table, Garden sensitivity check and limitations. Retained the Bonsai distribution. Rebuilt the Room cap/threshold and time plots in vector format with larger labels and supplied high-resolution presentation PNGs, including the new paired comparison. Replaced relative PDF launch links with twelve internal Evidence Register destinations and a named supplementary ZIP. Its index maps the IDs to the verified records, source, public Room input audits and configuration, geometry and Blender asset. The official benchmark still supplies the photos and the source generates checkpoints. Standardized all 13 references with complete authors, publication data and clickable titles, including the published CVPR Findings paper. The cover, margins and section style follow the project's local HCMUT sample. Official CO4029 submission-format compliance was not independently verified through the faculty's sign-in portal.

**Caption and PDF verification.** Moved the refinement-event diagram caption beneath its diagram and used a shared below-image caption layout for the imported figures. Tectonic produced 44 A4 pages. A compact one-page contents layout removed the transient spillover page. The appendix preserved straight ASCII quotes for copying PowerShell commands. Rendered and inspected all pages with Poppler. Checked all 18 caption positions against the corresponding picture, plot or diagram in the final PDF. Validated 163 internal and 23 external link annotations, including 12 internal evidence destinations and no local-file launch actions. The TeX log had no warnings, missing glyphs or bad boxes. Tectonic printed the same Fontconfig console diagnostic and completed. Final PDF SHA-256 `7acf83a96aef0f765c18f43172dd5057acb2aa497a6affdeaf30297da287be1a`. [Final PDF](../report/final_report/final_report.pdf) · [Supplement](../report/final_report/final_report_supplement.zip) · [PDF structural audit](../results/final_report_review_20261120/final/qa_report.json).

**Supplement readback.** Verified SHA-256 for all 879 bundled files. Extracted the ZIP in a fresh directory, confirmed its included report matched the release digest, and rebuilt the 44-page A4 PDF from the extracted LaTeX source. The rebuild TeX log contained no warnings or layout errors. The archive was refreshed to include this readback record.

## 2026-11-20 — Final manuscript review and captions beneath every figure and table

**Starting point.** Reviewed the November 20 manuscript, its 14-week project schedule, the local HCMUT report template, the evidence register and the supplied measurements. The 44-page PDF already placed its 18 figure captions below their illustrations. All 23 table captions preceded their tables. The Room/Garden paired comparison plot existed in the supplement but was absent from the manuscript. Preserved the previous PDF and its digest in the [review directory](../results/final_report_caption_review_20261120/).

**Changes.** Placed each table body, caption and label in one unbreakable block, with the caption below the bottom rule. Included the paired Room/Garden plot in the follow-up results. Added linked lists of all figures and tables, concise list entries, explicit prose references and a Final Report cover designation. Expanded the dataset inventory to cover Garden and distinguished its factor-four protocol from the indoor factor-two protocol. Corrected the bundle-adjustment description, explained why the pipeline fuses rendered depth and runs Marching Cubes, and compared the input assumptions with Poisson reconstruction using the original publications. Revised the abstract and conclusion to identify the baseline, summarize repeatability, report the negative surface-prior result and distinguish local edit persistence from physical accuracy. The edits use existing measurements. No training or mesh extraction was rerun.

**Final-report assessment.** The written report meets the documented project's technical reporting goals: research questions and success criteria, cited prior methods, implementable algorithm and camera assumptions, baseline and controlled studies, resource costs, error analysis, reproduced mesh workflow, saved edit verification and a conclusion tied to each question. The compact-policy rendering gain is supported on the evaluated scenes, with the Bonsai outlier, development-view exposure and single outdoor scene disclosed. Three Room seeds support repeatability of the complete policy. Ranking benefit remains mixed at the binding cap. Mesh conversion and local editing passed. Independent geometric accuracy and whole-route museum reconstruction remain unresolved. The local gsplat default is the comparator. A separately reproduced authors' baseline, a watertight production asset and independently surveyed dimensions are outside the demonstrated work. The wider project schedule also requests checkpoint handoff, supervisor review and a final presentation/demo. The ZIP provides source, records and mesh/edit samples and documents how to regenerate checkpoints. The written review does not establish delivery or approval of those other milestones. The [faculty project portal](https://project.cse.hcmut.edu.vn/) requires sign-in, so this assessment uses the project schedule and local template and does not certify formal submission compliance.

**Verification.** Built with Tectonic, rendered all 46 A4 pages with Poppler and inspected the final page images. Condensed the follow-up discussion and conclusion after finding two spillover pages in an intermediate build. Checked the physical page, caption order and gap below every visual, including the vector plots and refinement diagram. All 23 table captions start 4.76 PDF points below their bottom rule. All 19 figures and 23 tables appear in the linked lists with matching printed page numbers and resolved destinations. Verified 232 internal and 25 external link annotations, 12 evidence destinations and 15 cited bibliography entries. Cross-checked all ten primary RGB rows, ten computation/storage rows, ten topology rows and four pooled sensor-discrepancy rows against the original records. The TeX log contains no unresolved references, bad boxes or missing characters. The host Fontconfig tool emitted its existing config-file diagnostic. Final PDF SHA-256 is `f5bf52ed6d1a41e4b4bacdb4cba666c8cfdaa0aab442b60a3bc61bf761a2fef3`. [Final PDF](../report/final_report/final_report.pdf) · [Caption audit](../results/final_report_caption_review_20261120/caption_audit.json) · [Reproduction supplement](../report/final_report/final_report_supplement.zip).

## 2026-11-20 — Standalone illustrated guide for first-time readers

**Deliverable.** Created [From photographs to an editable room](../report/final_report/project_guide.html) beside the final report. The guide follows the Room coffee table from camera calibration through Gaussian fitting, A05-Abs allocation, rendered-depth fusion, Marching Cubes and a reopened Blender edit. It introduces the vocabulary at first use and includes an interactive measured photo/render comparison and an illustrative two-splat opacity/depth calculation. The HTML embeds its images, styles and interaction code. Companion-file links provide the full report, supplement, implementation and reproduction instructions.

**Sources and interpretation.** Used the current 46-page report sources, benchmark tables, verified follow-up and geometry records, original Room evaluation panel and report figures. Checked the original 3DGS and Mip-NeRF 360 project pages for attribution. The text distinguishes the original five paired benchmarks from the Room replay, three-seed repeatability and frozen Garden test. It reports the Bonsai outlier, the mixed ranking ablations, cap binding differences, local scope of the D455 segment, same-device depth discrepancy and the limits of the edit check. This documentation task used archived measurements without rerunning training, extraction or Blender.

**Verification.** Both opening images use the two halves of the same A05-Abs final evaluation panel for DSCF4667.JPG. Compared the photograph half against the original prepared Room image and found zero difference in each RGB channel. Matched the ten count and PSNR endpoints of the primary table to the report source. In headless Chrome with networking offline, loaded the HTML from disk at 1440×940, 390×844 and 320×720, decoded every image, checked mobile image selection, page width, chapter anchors, unique IDs, table scrolling, expandable notes and mobile navigation, toggled the measured photograph/render and checked the opacity example at 0%, 60% and 100%. All checks passed, with eight valid companion paths and no observed JavaScript exceptions. Inspected desktop and mobile screenshots. Final HTML SHA-256 is `47155ce526159f13d77c149b1cc340d6fc22a1a687de481e56a98bdb958ef273`.

## 2026-11-20 — In-depth 3DGS and A05-Abs technical study guide

**Reader feedback and revision.** The initial HTML survey did not provide enough depth to learn how to derive or implement the method. Expanded the same [standalone study guide](../report/final_report/project_guide.html) into ten chapters. Added a notation primer, worked camera triangulation, metric scale, Gaussian parameterization, covariance projection with its Jacobian, rasterization, spherical harmonics, alpha compositing, expected depth, a complete numerical forward pass, signed backpropagation, Adam, parameter storage and the 7,000-update lifecycle. Worked examples also cover sensor supervision, TSDF integration, Marching Cubes interpolation, shared-support geometry evaluation, topology, cleanup and independent Blender verification.

**Method and evidence.** Read the local trainer, rendering path, default and budget strategies, split operation, surface prior, extractor, depth-review code, A04/A05 development records, controlled Room tables, grid sensitivity and current report. Traced B00 through A01–A04 and distinguished the residual-fallback A05 and earlier CCD-GS pilot from the selected A05-Abs. Derived the gate, per-pixel contributor attribution, support discount, positive eligible median scaling, count schedule, selection, child geometry and optimizer-state transfer. Added worked factorial contrasts, paired-seed interpretation, resource costs, and explicit limits on component attribution. Linked the execution path to the local implementation and added eight self-checks with worked answers.

**Teaching examples and records.** Added a live projection model with angle and depth controls, a five-parent selection example with error/edge switches and slot control, an explorer of the 54 actual original Room seed-42 growth events, and the report's controlled Room cap/threshold figure. Kept synthetic examples labeled separately from measured results. All figures, recorded event values and interactions are embedded. The core selection example also has a static HTML table.

**Checks.** Compared every embedded event field to the archived trainer history and independently recalculated all 54 targets, selection counts, pruning arithmetic and event-to-event continuity. Cross-checked all four factorial rows with the evidence table and all ten original count/PSNR endpoints with the report. In offline headless Chrome at 1440, 390 and 320 px widths, verified image decoding and responsive sources, chapter/topic anchors, all 19 local companion links, accessible table scrolling, all eight self-checks, the photo/render switch, compositing slider, numerical footprint responses, canvas drawing, selection changes and measured first/last event readouts. Inspected rendered desktop and phone examples. No browser exceptions observed. The task used existing experimental outputs, with no new training, extraction or Blender run. HTML SHA-256: `64eb6722f999e73c00d3e281f51ca133834901b3ac8b3b0923f70e50595749eb`.

## 2026-11-20 — iPhone Pro LiDAR feasibility for the DB3 reconstruction

**Question and evidence.** Can a second scan of the scene in `20261030_134523.db3` reduce geometric error? The failing display interval around 120.7 s previously scored 3.050/7.401 px median/p90 on reserved moving endpoint checks after direct-depth camera fitting, above the <2/<4 px limits. The matte control scored 0.499/1.910 px. The target also had a 6.960 px p90 on an RGB-only epipolar check, and the glare-masking trial still failed. The evidence points to upstream camera or correspondence inconsistency, with possible contributions from depth and surface association. A second capture could supply stable geometric anchors and measure drift in the old map. It could also add a shape estimate where the old recording lacks reliable depth. [Recovery results](DB3_R4_RECOVERY_R5_START_20261112.md) · [Mask trial](#2026-11-17--tested-observation-masks-on-db3-camera-fitting)

**Pilot capture.** If the same physical area is accessible, choose one small unchanged overlap near the display with a floor, wall corner and opaque display frame. Capture two short, slow LiDAR passes with overlapping views and a return to the starting corner. Export a point cloud in PLY with documented units, and record the phone model, iOS/app version, capture paths and scene changes. A cloud is sufficient for the first alignment test. For future iPhone-only Gaussian reconstruction or RGB-D pose fitting, preserve timestamped RGB, metric `sceneDepth`, confidence, intrinsics, image dimensions/orientation, camera-to-world transforms and tracking state. The existing gsplat sensor-depth loader expects per-image samples tied to calibrated camera poses and consistent world scale. A fused cloud can provide a surface prior or be projected with visibility checks after registration. [Apple scene depth](https://developer.apple.com/documentation/arkit/arconfiguration/framesemantics-swift.struct/scenedepth) · [Apple camera metadata](https://developer.apple.com/documentation/arkit/arcamera) · [Local depth loader](../examples/datasets/colmap.py)

**Pilot comparison.** Keep the original recording, dataset, baseline and splits. Check agreement between the two phone passes and take independent tape or laser measurements on several stable landmarks for scale and evaluation. Match unchanged opaque features on surfaces with different orientations and reserve other landmarks and a revisit before fitting. Fit a local rigid alignment after verifying units and scale. If the baseline is in arbitrary units, estimate one similarity transform on the designated fitting anchors. Refine only on stable overlapping surfaces. Inspect held-out surface residuals before using the scan as a low-weight pose constraint or depth prior. Compare the unassisted and assisted reconstructions on the same reserved DB3 checks and physical measurements, with the same crop and reporting of median/p90 error, common-support coverage and unmatched areas. A reduction in measured geometric errors beyond phone repeatability, without sacrificing coverage or degrading the matte control, would justify expanding the pilot. Agreement with the phone alone measures cross-device consistency. Record disagreement, glass, occlusions and changed objects as uncertain regions.

**Limits and status.** Published indoor tests on an iPhone 13 Pro found app-dependent tracking and doubled surfaces near loop ends. Apple also reports reduced depth confidence on highly reflective surfaces. Prioritize the opaque exhibit frame and nearby room structure. If the original area is inaccessible, paired captures in another room can test the workflow on that room. This entry records a feasibility review and proposed protocol. Phone export, measured improvement and the hybrid pipeline remain untested. [Indoor comparison against terrestrial laser scanning](https://doi.org/10.3390/geomatics3040030) · [Apple depth confidence](https://developer.apple.com/documentation/arkit/ardepthdata/confidencemap)

## 2026-11-20 — Related work on LiDAR-assisted 3D Gaussian Splatting

LiDAR can provide initial Gaussian positions, camera registration constraints or depth observations for training. The method remains a 3DGS variant when the scene is represented by 3D Gaussians and views are rendered by splatting those primitives. The original method initializes from sparse structure-from-motion points and optimizes the Gaussians from posed images. A raw LiDAR export is a point cloud. It can seed a Gaussian model, and the PLY extension can be used by both kinds of export. [Original 3DGS paper](https://doi.org/10.1145/3592433)

- **LetsGo (SIGGRAPH Asia 2024)** uses LiDAR point clouds for Gaussian initialization and depth regularization alongside photometric supervision. It uses a custom handheld LiDAR/IMU/camera rig. [Project and paper](https://zhaofuq.github.io/LetsGo/)
- **TCLC-GS (ECCV 2024)** uses geometry from LiDAR and camera data to initialize Gaussians and supervise rendered depth. It targets autonomous driving. [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7983_ECCV_2024_paper.php)
- **iPhone precedent.** The public lidar-gsplat repository implements iPhone LiDAR depth and confidence supervision. ScanNet++ independently documents synchronized RGB and LiDAR capture with an iPhone 13 Pro and registration to a separate laser scan. These examples show feasibility of the components. [iPhone code](https://github.com/theLodgeBots/lidar-gsplat) · [ScanNet++ capture and registration](https://openaccess.thecvf.com/content/ICCV2023/papers/Yeshwanth_ScanNet_A_High-Fidelity_Dataset_of_3D_Indoor_Scenes_ICCV_2023_paper.pdf)

The local DB3 matte comparison already uses RealSense depth supervision with a 3DGS renderer. A phone-assisted variant could use an aligned phone cloud to seed positions, or use its depth to guide optimization after aligning observations to the cameras. The failing DB3 display segment still needs pose validation before adding cross-device depth. This follow-up reviewed related work and the existing trainer entry point. No phone capture or hybrid training was run. [Local trainer](../examples/simple_trainer.py)

## 2026-11-20 — Located the DB3 room point-cloud file

The user requested the file. Located the existing [RGB point cloud](../results/room_improve_full_source_cloud_20261103/development_only_unfiltered.ply) from `20261030_134523.db3`. The [source-cloud record](ROOM_IMPROVEMENT_CONTINUATION_20261103.md) documents 387 source frames and the diagnostic sampling. Verified the binary PLY header contains 180,000 XYZ/RGB vertices and the file is 2,700,180 bytes. SHA-256 `87d88870a163200769d50a8c737fc348a68c59f03fdd9f22900a355800155a37`. Shared the existing export.

## 2026-11-20 — Verified a viewer for the DB3 room point cloud

The repository virtual environment has Open3D 0.20.0. Loaded the existing PLY as 180,000 RGB-colored points and verified rendering in a hidden 640 x 480 Open3D visualizer. The captured framebuffer contained 22,594 pixels with unequal RGB channels. To open an interactive viewer from the repository root, use the same virtual environment and call Open3D's `visualization.draw_geometries([cloud])` after `io.read_point_cloud`. No source data or export changes were made.

## 2026-11-20 — Measured raw wall scatter and saved-pose overlap in the DB3

**Capture and selection.** Opened `20261030_134523.db3` in SQLite read-only mode. Its freshly computed SHA-256 `60e3f253051fc282a61af7f6c9c5eef40830a12d42566b9eba0992463115bc52` matches the R0 baseline. Scouted every eighth prepared RGB frame, then marked three exposed development views of the same matte wall under a lit panel beside an open doorway before examining their depth. Frames 200, 208 and 240 occur 66.981, 69.651 and 80.030 s after the first selected RGB. The raw RGB/depth message pairs are 52341/52332, 54421/54414 and 62511/62506. Verified timestamps against the original messages and pair identities against the timing and R1 manifests. The manually selected rectangles exclude the lit panel, jamb, ropes and figures.

**Separate raw frames.** Deprojected native nonzero depth in RGB-selected wall interiors using recorded factory intrinsics and extrinsics. Retained original samples at their continuous 3D coordinates and measured perpendicular distances to a separately fitted robust plane in each frame. No smoothing, resampling or filling entered this measurement. The three patches have 5,099, 8,741 and 10,530 points and 99.4–99.8% projected fill. Median native camera-Z is 4.96, 4.66 and 4.68 m. Median absolute plane residuals are 59.1, 61.6 and 72.6 mm, and p90 residuals are 160.1, 159.9 and 188.5 mm. The scatter is spatially structured. The archived filtered and sampled baseline depths in the same ray windows still have p90 residuals of 157.4, 135.6 and 168.9 mm (325, 452 and 527 samples).

**Saved-pose overlap.** Measured points against the other frame's fitted wall plane in both directions, within the opposing patch's projected footprint inset 1 cm. The saved RGB-D directions retained 55–98% of points. The RGB-D poses exported in `rgbd/dataset/sparse` agree with `rgbd/poses.npz` for all three frames. The COLMAP comparison uses the original shared fragment `reconstruction/0`, with camera translations divided by the previously saved 0.8346629364 model units per metre. No scale, plane-constrained camera pose, ICP or other alignment was fitted to these views.

| Frame pair | Saved RGB-D median / p90 separation (mm) | Saved COLMAP median / p90 separation (mm) |
| --- | --- | --- |
| 200 / 208 | 85–90 / 213–227 | 85–89 / 210–224 |
| 200 / 240 | 141–149 / 286–319 | 138–147 / 282–315 |
| 208 / 240 | 102–109 / 246–267 | 102–109 / 246–267 |

Ranges give the two directions. On those exact RGB-D point subsets, own-plane p90 was 143–168 mm. With opposing planes p90 increased in all six directions. The fitted wall normals differ by 12.8–23.4 degrees after applying the RGB-D poses. The metric scale came from historical RGB-D observations and the saved RGB-D trajectory used scale-gated COLMAP constraints, so the two pose sets share evidence. Both retain inter-view disagreement. Depth shape bias and relative pose errors can both contribute to it.

**RGB–depth boundary.** Rectified native RGB with the pinned SDK projection and projected depth with nearest-Z collision handling and no hole filling. Across four manually marked jamb rows per view, compared unfilled horizontal depth profiles to the outer jamb-to-opening silhouette. For frames 208 and 240, qualifying near-to-far transitions spanned 11–22 px. Seven of the eight brackets contained the RGB mark and the eighth extended +4 to +15 px. Frame 200 had a gradual ramp and gaps, so none of its four rows gave a qualifying bracket. Archived exposure-midpoint gaps were −8.33, −6.25 and −2.83 ms, corresponding to rough rotation-only estimates of 1.05, 0.77 and 0.35 px. The jamb, outdoor depth, missing values and mixed samples limit calibration inference. This shows a coarse wall-to-opening transition with unresolved pixel-level alignment.

**Conclusion.** Substantial wall deformation exists within the input depth before fusion and survives the historical cleaner. Combining the views under either saved trajectory increases wall disagreement. One wall constrains displacement normal to the surface and tilt, with limited sensitivity to motion along the wall. It supplies no independent pose ground truth or room-wide attribution. The test therefore does not isolate a COLMAP contribution or establish the cause of every floater.

**Artifacts and replay.** Run `.venv/Scripts/python.exe scripts/db3_wall_diagnostic.py results/db3_wall_diagnostic_20261120/<new-directory>` from the repository root. Saved RGB scouting images and pre-depth edge marks in the result root. The first run in `analysis` used a ±35 mm display window that clipped much of the residual scatter. The final figure shows the full residuals, and the final run adds the historical cleaned comparison and matched-subset residuals. A fresh run in `replay` reproduced all six final files byte for byte, and the three native point arrays reproduce the recorded Z16 values at the exported pixel indices after unit conversion. The source, poses and baseline were consumed read-only. No reconstruction or training ran. [Measurements and camera matrices](../results/db3_wall_diagnostic_20261120/final/measurements.json) · [Raw points and RGB–depth profiles](../results/db3_wall_diagnostic_20261120/final/01_raw_depth_and_rgb_edges.png) · [Transformed wall overlap](../results/db3_wall_diagnostic_20261120/final/02_wall_overlap_saved_poses.png) · [Script](../scripts/db3_wall_diagnostic.py)

## 2026-11-20 — Same-wall range, low-motion repeatability and frozen-pose filtering

**Selection and controls.** Scouted prepared frames 192–246 and 396–440 at every second frame, plus native RGB near prepared frames 428–430. Tested 16 original RGB/depth pairs on the matte wall below the same lit panel. Used a small region defined relative to the four annotated RGB panel corners to follow the same physical interior, with approximate extrapolation in the last three partly clipped views. The panels are an RGB reference for the patch footprint. Doorway, floor and exhibit samples were excluded by the image region. Used original Z16, recorded calibration, and nearest DB3 playback-time depth within 20 ms. All ten prepared frames matched their archived depth IDs. Recorded observed values before any filtering, nearest-Z projected validity, coverage across RGB subregions, absolute point-to-plane residuals, median optical-axis Z, Euclidean camera-to-point range, and incidence estimates from the fitted planes. The narrower patch differs from the first wall experiment.

**Range and coverage.** For the three shortest observed ranges (3.36–3.49 m, optical-axis Z 2.53–2.68 m), raw p90 plane deviation was 38–50 mm with 38–61% projected pixel validity. Four views at 3.83–3.98 m (Z 3.04–3.47 m) had 58–67 mm p90 with 66–84% validity. The original three views at 4.65–5.34 m (Z 4.55–4.85 m) had 99–115 mm p90 with 99–100% validity in this smaller region. Later oblique views at 4.51–4.95 m had 40–113 mm p90 with 45–86% validity. Thus shorter observations can look flatter while returning fewer valid points. Camera angle, visibility and time vary with range, and the shortest apparent Z does not supply an approximately 2 m camera-to-wall range condition. No tape, laser or independent wall reference was available. [Range and coverage plot](../results/db3_wall_range_20261120/final/range_and_coverage.png) · [Annotated RGB regions](../results/db3_wall_range_20261120/final/wall_regions.png)

**Temporal evidence.** Searched native RGB ±180 ms around each reference. Kept distinct depth frames with RGB p90 displacement at most 1 px over at least 12 forward/backward-consistent local tracks and at least 70% of candidate tracks. Only one reference yielded another depth frame. RGB 110495/110519 were 33.0 ms apart, with p90 RGB motion 0.58 px across 48 tracks. At 468 common native depth pixels (85.1% of the reference patch's native samples), depth IDs 110486/110514 differ by 342 mm median absolute and 708 mm p90. The median signed change was +331 mm, with 480 mm p90 remaining after removing that median shift. Separately detrended spatial residuals had Pearson correlation 0.713. This is evidence of substantial change and persistent structure within one low-image-motion pair. True stationarity and a distance-specific temporal noise estimate remain unverified.

**Frozen-pose comparison on reserved observations.** Built two wall-only pooled plane models from frames 200, 208 and 240 using the unchanged metric RGB-D poses. The filtered arm takes a 3×3 native median only when at least five nonzero samples are within 40 mm of the valid center and never fills a hole. The final fit uses the same 1,118 native pixel IDs per frame in both arms. Of 3,354 fitted samples, 298 move after filtering. Scored raw depth from newly reserved frames within the common projected hull support, inset by 1 cm. The underlying historical poses may use these observations. The first run used separately drawn point subsamples. The final run pairs the exact pixels, and its results supersede the first run.

| Reserved frame | Shared footprint support | Raw fit p90 | Filtered fit p90 |
| --- | ---: | ---: | ---: |
| 232 | 98.1% | 124.0 mm | 124.2 mm |
| 234 | 87.7% | 110.4 mm | 110.8 mm |
| 429 | 27.8% | 79.6 mm | 78.9 mm |
| 422, 424, 426, 428 | 0–11.4% | Insufficient support | Insufficient support |

The filter changes held-out p90 by less than 1 mm on the three measurable views. The late oblique revisit frequently falls outside the fused region under the saved poses. This measures combined placement and depth-support failure, with no isolated pose attribution. The third arm requires a separately validated alternate pose trajectory spanning these exact fitting and evaluation frames. The archived 64–72 s exhibit candidate does not span the late revisit or even all three fit frames. The direct-depth follow-up validated only other short intervals, and the R4 trajectory did not pass global placement. No new pose adjustment was fitted. This interior patch experiment does not assess whether filtering preserves doorway geometry.

**Replay.** Run `.venv/Scripts/python.exe scripts/db3_wall_range.py results/db3_wall_range_20261120/<new-directory>`. A clean replay reproduced the final JSON, both PNGs and paired pooled-point NPZ byte for byte. Checked disjoint fitting and reserved frame lists, matched source counts, preserved raw missing-depth masks through filtering, and compiled the script. The source was opened read-only throughout. [Final measurements](../results/db3_wall_range_20261120/final/measurements.json) · [Script](../scripts/db3_wall_range.py)

## 2026-11-21 — RealSense D455 camera-pose input audit

Reviewed the DB3 camera-estimation code, saved 120k training configuration, local matte protocol and route checks in response to the sensor-use question. The pipeline uses D455 RGB feature tracks and calibrated depth observations. Initial COLMAP feature extraction, matching and mapping use RGB images. The historical 120k run consumed the metric RGB-D dataset: aligned depth contributes to relative pose constraints (3D alignment and PnP) and metric scaling of accepted SfM fragments in the pose graph. RGB-only SfM registered 301/477 selected frames in its largest fragment, and the RGB-D graph registered 477/477. Registration measures connectivity, with global accuracy still unverified. [Initial COLMAP and export code](../scripts/clean-realsense-scene.py#L309) · [RGB-D pose recovery](../scripts/recover-clean-rgbd.py) · [Saved configuration](../results/realsense_20261030_134523_120k/train/cfg.yml#L18)

The local R2 matte solve jointly refines camera poses and 3D landmarks with RGB reprojection errors and measured camera-axis depth residuals at valid feature observations. It fixes intrinsics and the first camera. The three reserved evaluation cameras are localized from RGB against the fixed local map. Their depth is reserved for consistency checks. [Joint RGB-D bundle adjustment](../scripts/db3_local_geometry.py#L220) · [Report methodology](../report/final_report/methodology.tex#L28)

Both gyro and accelerometer are present in the recording and exported. Gyro samples inform timing and motion diagnostics. Neither stream enters the audited pose objectives or supplies inertial priors. An exploratory keyframe graph tested one loop candidate. The later R2 distributed checks accepted zero global revisits, and R4 passed 132 of 317 eligible adjacent intervals without a full-route bundle adjustment or merge. Full-route loop closure and a globally validated room-scale trajectory remain unestablished. Public benchmark routes use supplied COLMAP cameras. This was a source and saved-artifact audit. No pose fitting, reconstruction or training was run. [IMU export](../scripts/clean-realsense-scene.py#L95) · [Exploratory revisit](../scripts/try-room-revisit-pose-repair.py) · [R2 revisit checks](../scripts/db3_transition_revisit.py#L136) · [R4 results](DB3_R4_RESULTS_20261111.md)

## 2026-11-21 — Controlled D455 IMU-assisted camera-pose experiment

The four arms kept all 477 selected cameras. Gyro-only B and exploratory full-IMU C changed the full-route trajectory and lowered its disagreement with integrated gyro. Neither increased the number of R4 direct interval checks passed relative to the RGB-D control A1. C improved alignment on the previously inspected wall patch, with mixed route-level evidence. Both inertial treatments had negligible effect on the matte control and failed to recover the display window. The experiment stopped at the pose and measured-point-cloud stage under the brief's validation gate. The 21-train/3-eval pose-only and reinitialized 7,000-update 3DGS runs and TSDF comparison were deferred.

### Frozen inputs and motion audit

- Used `20261030_134523.db3`, the matching original 477 RGB/depth identities, the serialized graph's 1,731 retained RGB-D and scale-gated SfM constraints, archived camera transforms, exact fitting factors and withheld checks for the two native-rate windows, and the frozen R4 and R2/R3 revisit rosters. Recomputed the 17,129,000,960-byte recording SHA-256 as `60e3f253051fc282a61af7f6c9c5eef40830a12d42566b9eba0992463115bc52`. The capture, original model and data products were read-only. [Frozen input hashes](../results/db3_imu_20261121/audit/audit_protocol.json)
- Matched IMU metadata to exports by exact playback timestamp, verified exported headers reproduced those playback stamps, and integrated on device time. All four modalities declare Global Time in the metadata. Their readout-to-global clock conversions show about 13.15 ms of drift over the capture and agree closely across modalities. Used RGB `Sensor Timestamp` at exposure midpoint and IMU `Frame Timestamp`. Kept the original paired depth IDs, without motion compensation. This establishes a usable common device-clock basis with residual sensor latency unverified. [Audit](../results/db3_imu_20261121/audit/audit.json) · [Synchronized arrays](../results/db3_imu_20261121/audit/synchronized_streams.npz) · [librealsense frame metadata definitions](https://github.com/realsenseai/librealsense/blob/v2.59.1/include/librealsense2/h/rs_frame.h#L15-L37)
- The streams contain 31,758 gyro and 15,905 accelerometer samples, with median steps of 5.006 and 9.998 ms. Six gyro gaps account for ten missing frame numbers. The largest gyro step is 20.024 ms. Both streams are finite and strictly increasing. Largest measured single-axis magnitudes are 6.192 rad/s and 22.771 m/s², below the D455 family nominal ranges of ±1,000°/s and ±4g. The recorded motion-correction setting was enabled. The 7.5/15 ms gyro/accel maximum-gap and boundary-bracketing rules yield gyro support on 469/476 intervals and accelerometer support on 475/476. The first interval is unbracketed for both sensors. Six intervals with gyro gaps receive no inertial factor. [All 476 intervals](../results/db3_imu_20261121/audit/selected_intervals.csv) · [D455 IMU ranges](https://www.realsenseai.com/wp-content/uploads/2026/03/RealSense-D400-Series-Datasheet-Mar-2026.pdf#page=94)
- Composed the recorded IMU-to-color rotation and lever arm, approximately (−28.8, −7.4, −15.7) mm in color coordinates. Gyro and accel have the same recorded extrinsics. The IMU intrinsics are identity with zero reported variances. No stationary segment was certified. The lowest corrected gyro p90 over scanned half-second windows was 0.087 rad/s. Estimated a constant gyro correction from 144 R4 fit-only rotations, withholding 73 calibration links by time block and excluding the test windows. The fitted bias in rad/s is (−0.00435, −0.00031, −0.00066). With the fixed 0 ms device-clock offset, reserved calibration rotation residual median/p95 were 0.244°/0.752° with the correction and 0.220°/0.706° with zero bias. The ±20 ms profile did not establish a unique delay across summary metrics. Kept the specified fitted bias and 0 ms device timing for the treatment. Effective rate uncertainty was at the 0.020 rad/s per-axis floor. It incorporates RGB-D/gyro disagreement and assumed floors. It does not establish sensor noise, accelerometer bias, or surveyed extrinsics. [Calibration profile](../results/db3_imu_20261121/audit/audit.json)

### Matched optimization

- A0 is the historical trajectory on the route and the archived direct-depth pose solution within each native-rate window. A1 uses the same RGB-D factors as the corresponding treatment arms, including archived graph information and confidence on the route and the exact 11,654 display/15,525 matte measurement factors locally. B adds covariance-scaled integrated relative gyro rotations. C adds preintegrated position, velocity and rotation with the color–IMU lever arm, optimized fixed-magnitude gravity and bias priors. C uses a constant bias within each short window and interpolated bias knots with a random walk on the full route. The accelerometer noise densities and bias priors are assumptions. The C results are exploratory. No new loop constraint was admitted.
- Used the installed SciPy sparse robust least-squares solver for all re-optimized arms after a GTSAM installation dry run found no compatible CPython 3.12 Windows wheel. Kept the first camera fixed and the camera initializer identical within each comparison. A1 converged in 3 evaluations on the route and moved the archived camera centers by 0.14 cm median and 0.31 cm p95, with rotation changes of 0.023° median and 0.085° p95. A1 locally reproduced A0 within 0.007 mm median translation in both windows. B and C converged in 34 and 75 evaluations on the route, with 469 inertial factors each. Relative to A1, route gyro normalized RMS fell from 2.327 to 0.618 and 0.409, respectively, while RGB-D normalized RMS rose from 0.258 to 0.282 and 0.403. The inertial arms moved camera centers by about 5.15 cm median and 40.55/36.87 cm p95. [Route protocol](../results/db3_imu_20261121/route_v2/protocol.json) · [Solver outputs and inertial state](../results/db3_imu_20261121/route_v2/full_route_optimization.json) · [Trajectory comparison](../results/db3_imu_20261121/route_comparison.png) · [GTSAM preintegration reference](https://borglab.github.io/gtsam/preintegratedimumeasurements/)

### Validation and measured geometry

| Full-route arm | Passed existing direct gates out of 317 | Prior failures now passing direct gates | Median / p90 separation across supported pair summaries (cm) | Median / p90 symmetric reprojection across supported pair summaries (px) |
| --- | ---: | ---: | ---: | ---: |
| A0 archived trajectory | 131 | 2 | 1.197 / 3.736 | 0.703 / 1.791 |
| A1 RGB-D control | 131 | 2 | 1.183 / 3.737 | 0.728 / 1.814 |
| B RGB-D + gyro | 131 | 2 | 1.284 / 3.831 | 0.912 / 1.843 |
| C RGB-D + full IMU | 130 | 1 | 1.210 / 3.776 | 0.811 / 1.766 |

Rebuilt the original check rosters from the archived feature/measurement arrays and reproduced 132/317 passes from the original local transforms and acceptance definitions. The table substitutes each global trajectory's relative endpoint poses and applies the same direct support and metric thresholds. The three optimized arms retained 129 original passes each. The 159 reserved boundaries were excluded, and only 141 eligible boundaries had at least 20 check matches. The full native retry rules require intermediate-camera estimates and were not rerun for the new trajectories. The archived route also shares capture observations with these checks. The metrics are development comparisons on matched observations with medians across pairs, with 20-second groups recorded separately. [All route interval scores, groups and revisits](../results/db3_imu_20261121/route_evaluation/route_checks.json)

Six frozen long-range candidates could be rescored and two stayed excluded by reservations. All six had 0–14 check matches against the minimum of 20. The candidate with 14 matches dropped from 39.9 cm median under A1 to 22.4 cm under B and 18.4 cm under C, still outside the 5 cm median threshold. Zero independent revisits were validated. Around 33.81 s, C reduced an A1 single step of 1.031 m to 0.146 m between frames 100 and 101. That interval has two check matches, insufficient to establish the corrected displacement.

| Local window | A1 endpoint reprojection median / p90 (px) | B | C | Local gates |
| --- | ---: | ---: | ---: | --- |
| Matte, 12.015–12.348 s | 0.499 / 1.910 | 0.499 / 1.910 | 0.500 / 1.911 | All pass |
| Display, 120.678–121.145 s | 3.050 / 7.401 | 3.052 / 7.403 | 3.049 / 7.398 | All fail |

The local checks use the original withheld moving and descriptor identities, on matching endpoints and both half-pairs. Bidirectional depth validation uses a fixed pixel grid in reserved tiles, stable 3×3 depth neighborhoods, pre-existing candidate-person and glare masks and an occlusion test. Dynamic and reflective masking remains approximate. Metrics compare the common supported source pixels across arms and record per-arm valid coverage. Across the six directions per window, the median of pair p90s was 4.138 cm with about 90.0% median coverage in matte, and 1.769–1.771 cm with about 76.9% coverage in the masked display area. The lower masked display depth number describes the stable visible subset. Its original RGB check still fails. Neither inertial arm met the existing 1 cm meaningful-gain criterion. [Matte checks and poses](../results/db3_imu_20261121/local_v2/matte_control/metrics.json) · [Display checks and poses](../results/db3_imu_20261121/local_v2/target/metrics.json)

Exported matched measured-depth clouds for A0/A1/B/C using fixed subsamples of the aligned, unfilled D455 depth, with 477,000 paired colored vertices per full-route cloud, 254,119 per matte cloud, and 322,446 per display cloud. For the previously inspected wall under the illuminated panel, held the RGB regions and native measured pixels fixed. Intersected opposing wall footprints across the four arms in each direction. On the same source points, C reduced point-to-opposing-plane p90 by 6.9–31.3 mm across all six directions. Absolute p90 still spans 198–288 mm in C. B improved four directions by 3.1–5.2 mm and worsened two by 0.5–2.4 mm. The wall check is a follow-up on exposed observations and provides no independent physical surface reference. [Matched wall measurements](../results/db3_imu_20261121/wall_same_points.json) · [Wall comparison figure](../results/db3_imu_20261121/wall_comparison.png) · [Route clouds and trajectories](../results/db3_imu_20261121/route_evaluation/)

### Replay and integrity

From the repository root, use the existing Windows virtual environment and a fresh output directory for each stage. The logged runs set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=2` for fitting.

```powershell
.venv/Scripts/python.exe scripts/db3_imu_audit.py results/db3_imu_replay/audit
.venv/Scripts/python.exe scripts/db3_imu_pose_experiment.py local results/db3_imu_replay/audit results/db3_imu_replay/local
.venv/Scripts/python.exe scripts/db3_imu_pose_experiment.py route results/db3_imu_replay/audit results/db3_imu_replay/route
```

The initial route run fitted A1/B, then stopped on a missing revisit source list. The next run added full-route C under the documented exploratory calibration assumptions and exposed a second missing-field case on the same two reserved revisits. Recovered by scoring the saved four-arm matrices unchanged with the reservation-aware reader. A full replay from the current entry points reproduced the audit JSON, coverage CSV and synchronized samples byte for byte, along with the route pose arrays, route measurement JSON and all four route PLYs. The first and refactored local runs reproduce A0/A1/B matrices exactly and agree on C within 2.70 × 10⁻⁶ per matrix entry after the bias sparsity change. Independent checks verified the recording hash, 21 frozen input hashes, sample coverage, the 477-pose roster and fixed gauge, identical point counts and RGB order across arms, and the original R4 threshold count. The determinant check allows the archived Open3D matrices their recorded precision. Synthetic stationary and constant-rate integrals, covariance and sample-gap checks passed. Nine focused repository tests passed, with the existing pytest warning about the unknown `env` option. [Verification output](../results/db3_imu_20261121/verification.json) · [Plot and wall-check script](../results/db3_imu_20261121/verify_and_plot.py)

## 2026-11-21 — D455 depth quality, measured-depth fusion and route diagnosis

**Scope.** Ran the archive experiments on `20261030_134523.db3` using its frozen RGB/depth identities, R2 matte poses and R4 checks. Read the recording in SQLite query-only mode and independently verified its SHA-256 as `60e3f253051fc282a61af7f6c9c5eef40830a12d42566b9eba0992463115bc52`. Development comparisons reuse exposed views and measurements. Physical error requires a surveyed reference. This run used existing poses and Gaussian meshes throughout.

### Phase 1 — Hardware measurement and archived geometry

- **1A remains physically blocked.** Both Windows present-device enumeration and the pinned RealSense SDK 2.59.1 found zero connected D455s. The live capture entry point reported the same condition before creating an output directory. No stationary distance data were collected and sensor acceptability remains open. Added `scripts/d455_stationary_depth.py` for a flat matte target covering the view with the camera on a fixed mount. Capture one repeatable 100-frame Z16 trial at each independently measured axial distance, nominally 0.7, 1, 2, 3 and 4 m as the setup allows. Record the reference uncertainty and account for the offset from the housing to the optical origin. It saves raw frames and actual active profile, scale, firmware, preset, emitter, exposure settings and available frame exposure metadata. Reports include valid fill, center bias, axis-intersection bias, per-pixel temporal standard deviation and MAD, plane RMS/p90, and outer-band minus center depth and plane residual. The comparison ROI spans the central 80% of each image dimension and requires 90% temporal pixel support for depth statistics. The synthetic 100-frame check recovered a known 9.5 mm bias, 0.5 mm MAD, 98.33% whole-image fill, and zero plane and edge error to numerical tolerance. [Stationary capture and analysis](../scripts/d455_stationary_depth.py) · [RealSense depth testing methodology](https://www.intel.com/content/dam/support/us/en/documents/emerging-technologies/intel-realsense-technology/RealSense_DepthQualityTesting.pdf)
- **1B numerical conventions verified.** Archived streams are RGB8 1280×720 and Z16 848×480 at nominal 30 Hz with 0.001 m per depth unit. Depth camera points use axial Z. The depth-to-color translation is (−59.013, −0.047, +0.360) mm and the recorded rotation convention is column-major. The rectified RGB and COLMAP model match at 834×463, focal lengths 428.4312/427.9930 px, principal point 419.2249/233.4870 px and crop (7,7). The COLMAP transform is world-to-camera. Back-projected points use its inverse for world placement and TSDF receives world-to-camera. Numeric round trips on known 3D points were within 5×10⁻¹⁶ m. Substituting Euclidean range for axial Z created up to 137 mm of error on the selected probes. A translated synthetic plane fused through Open3D raycast at the expected Z of 2.500 m in world coordinates and 2.000 m from its camera, with the expected RGB. The existing independent SDK audit found at most 0.000205 raw pixel projection difference and 0.000000187 m extrinsic transform difference. [Geometry audit](../results/db3_measured_depth_20261121_verified/geometry_audit.json) · [Pinned SDK audit](../results/db3_r0_r1_20261110/sdk_audit.json)
- **Recorded settings and timing.** Option values show preset 0, emitter enabled, laser power 150, and RGB and depth autoexposure enabled. Depth exposure metadata span 0.962–24.944 ms. The archived initialization and R4 depth preprocessing use stability/range filtering. The new raw cloud preserves finite in-range samples after projection and nearest-Z collision selection. No depth holes were filled. All 477 selected RGB/depth IDs matched the timing table. Across 4,764 native pairs, depth-minus-RGB playback gap median is −7.247 ms and exposure-midpoint gap median is −3.292 ms, with 5th–95th percentiles of −11.383 to −1.003 ms. [Sensor timing](../results/db3_r0_r1_20261110/sensor_timing_pairs.csv)
- **Physical alignment remains bounded.** The prior RGB silhouettes overlap projected depth bands at the kiosk and opaque box, with bands up to 9 and 14 px. On the difficult 4.66–4.96 m wall, only 8/12 jamb rows bracketed a measurable transition, with endpoint separations of 11–22 px. The known geometry checks verify implementation and direction. These exposed physical edge measurements support coarse alignment only and give no independent scale or extrinsic calibration estimate. [Reviewed opaque boundaries](DB3_R0_R1_RESULTS_20261110.md) · [Wall measurements](../results/db3_wall_diagnostic_20261120/final/measurements.json)

### Phase 2 — Direct measured-depth baseline

**Single frame.** Chose matte split training RGB 6227 using the RGB inspection to mark an opaque wall strip and the floor. Depth back-projection produced 294,981 colored points in camera coordinates. Native nonzero fill was 91.94% and rectified in-range fill was 76.39%. The missing area through the bright doorway is visible in the depth image. The wall patch at 1.832 m median Z had 98.62% fill, 16.6 mm plane RMS and 27.3 mm p90. The floor at 3.242 m had 99.88% fill, 8.0 mm RMS and 12.8 mm p90. The observed floor sample spans 1.73 m in camera X. The visible extent and planes have no tape-measure or surveyed reference. [Single-frame PLY](../results/db3_measured_depth_20261121_verified/single_frame.ply) · [RGB, depth and cloud](../results/db3_measured_depth_20261121_verified/single_frame.png)

**21-frame matte fusion.** Used the exact 21 archived training RGB/depth pairs and metric local COLMAP poses. Exported 7,244,905 raw colored points. The native 3×3 range/continuity gate requires neighbor spread below max(30 mm, 2% of depth), then applies a 3×3 median and rectified nearest-Z selection. This left 7,024,287 pre-voxel samples and 1,834,483 points after 1 cm voxel averaging. The CPU Open3D tensor TSDF integrates the filtered raster in meters with 1 cm voxels, 4 cm truncation and mesh extraction weight threshold 0.5. It produced 1,178,037 vertices and 2,184,169 triangles, with 12,708 components, 162,567 boundary edges and zero edges used by more than two faces. The largest component contains 71.82% of surface area. The very fine mesh exposes considerable fragmentation and retains holes. [Raw cloud](../results/db3_measured_depth_20261121_verified/multi_frame_raw.ply) · [Filtered cloud](../results/db3_measured_depth_20261121_verified/multi_frame_filtered.ply) · [Direct TSDF mesh](../results/db3_measured_depth_20261121_verified/direct_depth_mesh.ply) · [Protocol](../results/db3_measured_depth_20261121_verified/protocol.json)

**Initializer comparison.** The frozen Gaussian initializer has 90,000 points. Its nearest distances to the filtered direct cloud were 3.8/5.5 mm median/p90 and all points were within 2 cm. In the reverse direction the distances were 19.6/149.2 mm and 50.9% were within 2 cm. Distance is asymmetric because the initializer caps samples and requires multiview support and confidence. The extra direct-cloud points include surfaces, sparse measurements and outliers. The comparison does not designate them as additions or deletions. [Initializer PLY](../results/db3_measured_depth_20261121_verified/gaussian_initialization.ply) · [Numerical results](../results/db3_measured_depth_20261121_verified/matte_summary.json)

**Held-out mesh comparison.** Three reserved RGB-localized views and their sensor depth were withheld from integration. Raycast all four meshes in the same local metric frame and score the same 845,126 pixels where all four hit valid stable sensor depth. Per-mesh coverage uses all 918,331 stable pixels. The direct matched-grid arm uses the archived 192³ uniform volume, origin, 8.282 m side length, 0.17255 m truncation, 208×115 image dimensions, scaled intrinsics and training cameras. It uses nearest measured depth preserving zeros. The rendered depths retain their existing opacity gate. Thus the input depth source and validity mask differ while the extractor grid matches. [Side-by-side renders and discrepancy maps](../results/db3_measured_depth_20261121_verified/heldout_comparison.png) · [Matched grid PLY](../results/db3_measured_depth_20261121_verified/direct_depth_mesh_matched_uniform.ply) · [Matched grid protocol](../results/db3_measured_depth_20261121_verified/matched_uniform_protocol.json)

| Mesh | Stable sensor coverage | Median signed error (cm) | Median / p90 absolute error (cm) | Triangles | Boundary edges |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct D455, 1 cm | 97.25% | +0.12 | 3.79 / 8.90 | 2,184,169 | 162,567 |
| Direct D455, matched uniform grid | 94.56% | −1.16 | 3.40 / 9.66 | 55,341 | 6,027 |
| B00 rendered-depth mesh | 97.28% | −8.18 | 8.22 / 18.78 | 80,952 | 13,300 |
| A05-Abs rendered-depth mesh | 97.82% | −9.19 | 9.31 / 19.42 | 99,107 | 21,023 |

Direct depth retained a lower local discrepancy on the matched grid, at lower coverage. In the images the direct mesh leaves much of the open doorway and a reflective exhibit edge unobserved. The finer direct grid resolves more color detail on the panels and creates more open edges. The Gaussian meshes render coverage across some sensor gaps and show a larger nearward bias on the common observed pixels. This supports a local effect from the depth supplied to fusion. Residual pose, occlusion and sensor error also remain possible. Held-out D455 depth is a consistency reference with shared calibration and pose dependencies.

**Contiguous expansion.** Built independent runs from the intersection of original R4 passes and A0 passes. Found 12 uninterrupted runs of at least four views, with maximum length seven. Used the earliest run at each available length while skipping reservations. All three runs save raw and filtered clouds and a direct TSDF. A separate volume per run omitted its central view for the scores below. Predictions use fixed archived poses and the omitted view's filtered measured depth. Only the four-view example had a p90 under 5 cm. The longer examples show that passing local feature gates leaves dense disagreement. No route-wide fusion was attempted. [Route runs and full metrics](../results/db3_measured_depth_20261121_verified/route_segments.json)

| Selected indices | Views | Probe coverage | Probe median / p90 absolute error (cm) | Mesh triangles |
| --- | ---: | ---: | ---: | ---: |
| 9–12 | 4 | 99.17% | 0.94 / 3.07 | 288,653 |
| 306–311 | 6 | 87.90% | 1.41 / 14.82 | 1,985,347 |
| 201–207 | 7 | 92.51% | 4.18 / 14.33 | 2,641,126 |

### Phase 3 — Frozen route-pair error map and regional diagnosis

**Coverage and definitions.** Rebuilt SIFT match membership from archived descriptors using the original mutual ratio test and separate fit/check tiles. Computed RGB match and occupied tile counts and hull areas, original fit inlier fraction, fixed stable-depth support, paired depth fill/edge/unknown-neighbor coverage, signed exposure gap and motion, and relative meter/degree motion. Every interval also stores median/p90 3D separation and symmetric withheld reprojection for A0, A1, B and C. Reproduced the prior arm scores with the identical depth observations. Valid depth and stable non-edge depth are coverage proxies. Route-wide verified static-surface coverage is unavailable. The display inspection reports the portion covered in existing annotated regions. [CSV](../results/db3_measured_depth_route_20261121_verified/frame_pair_error_map.csv) · [Full per-pair JSON](../results/db3_measured_depth_route_20261121_verified/frame_pair_error_map.json) · [Trajectory and time plot](../results/db3_measured_depth_route_20261121_verified/trajectory_failure_map.png)

| Pose arm | Passed | Failed with usable evidence | Insufficient or invalid evidence |
| --- | ---: | ---: | ---: |
| A0 archived trajectory | 131 | 9 | 177 |
| A1 RGB-D control | 131 | 9 | 177 |
| B RGB-D + gyro | 131 | 9 | 177 |
| C RGB-D + full IMU | 130 | 10 | 177 |

The map covers 317 original eligible route intervals and shows 159 reservations separately. A usable direct check requires at least 20 stable-depth checks and the original direct fit hull and information gates. A pass additionally requires median below 5 cm and p90 below 12 cm. Original R4 local transforms plus native retries passed 132/317. A0 endpoint poses lose passes 93, 134 and 138 and gain 139 and 193. Intervals 93, 134 and 138 are the best-supported candidates for bounded pose refinement because the original local transforms pass the same frozen endpoints. Verify the gain against independent dense or physical checks before updating the trajectory. The 120–140 s block passes only 7/45, with 37 insufficient, while 80–100 s passes 23/33. This measures observation and support clustering, with topology and overlap of the physical room unresolved.

**Failure classification.** All 186 A0 nonpassing rows carry an explicit unresolved causal status, measured gate failures and relevant archived native retry outcomes. Among them, 64 had at least 20 RGB check matches but fewer than 20 stable-depth checks, 61 had under half aligned depth fill at an endpoint, five had sparse or concentrated RGB matching, 42 had a gyro timing estimate above 1 px at an endpoint, and 14 had more than 20° A0 endpoint rotation. Flags overlap. They expose depth, visual support and timing problems while leaving overlap, occlusion, surface identity, calibration, motion and pose drift unresolved where the data cannot separate them. The three local-transform comparisons support a relative pose discrepancy. The separate revisits still lack the required support for a verified global alignment.

**Display at 120.68–121.14 s.** Inspected the first, middle and last RGB/depth frames and overlaid their measured clouds under A0 and C. Glare ROI valid fill was 73.9–79.9%, versus 91.7–92.8% in visually static candidate regions, and stable non-edge fill was 64.8–73.2% versus 84.6–84.9%. Visible holes follow the bright reflection. Full-view clouds contain sparse distant returns as well as the main near display surfaces. The near overlay changes little between A0 and C. The original window endpoint RGB check still has p90 about 7.4 px across arms, while R4 interval 362 has insufficient stable-depth checks. The RGB-only audit and masking control support further examination of correspondence and physical surface identity. [Display RGB and depth](../results/db3_measured_depth_route_20261121_verified/display_rgb_and_depth.png) · [Display cloud overlay](../results/db3_measured_depth_route_20261121_verified/display_registered_clouds.png) · [Measurements](../results/db3_measured_depth_route_20261121_verified/regions.json)

**Wall under illuminated panel.** Inspected the fixed RGB ROIs, raw depth and measured-cloud overlays for selected frames 200, 208 and 240. The single-frame wall patches already have p90 plane residuals of 160, 160 and 188 mm. Registered overlap also remains broad. The saved common-footprint full-IMU comparison has 198–288 mm p90 against the opposing planes. This region has substantial disagreement before pose registration and a persistent cross-view component. Prioritize an independent range and surface test before using it as a pose refinement target. [Individual wall frames](../results/db3_wall_diagnostic_20261120/final/01_raw_depth_and_rgb_edges.png) · [Registered wall overlay](../results/db3_measured_depth_route_20261121_verified/wall_registered_clouds.png)

### Reproduction and verification

From the repository root in the existing Windows virtual environment, run with fresh destinations.

```powershell
$env:OMP_NUM_THREADS = '4'
.venv/Scripts/python.exe scripts/db3_measured_depth.py results/db3_measured_depth_replay
.venv/Scripts/python.exe scripts/db3_route_diagnostics.py results/db3_measured_depth_route_replay
```

For the archived SDK environment with a connected camera and target positioned at an independently measured distance, the stationary capture CLI takes the measured axial distance and its uncertainty in meters. For example, enter `--distance-m 1.000 --uncertainty-m 0.005` only for a 1.000 m reference with 5 mm uncertainty. Replay a saved stack by adding `--from-capture PATH_TO_RAW_DEPTH.NPZ`. The stationary program leaves the current camera options as recorded. With the pinned SDK install already in this workspace, expose it using `$env:PYTHONPATH = (Resolve-Path .cache/room_sdk_audit/install).Path` before running `.venv/Scripts/python.exe scripts/d455_stationary_depth.py RESULTS_DIRECTORY --distance-m MEASURED_DISTANCE --uncertainty-m MEASURED_UNCERTAINTY`.

The first legacy scalable Open3D 0.20 fusion produced no vertices. A synthetic plane reproduced that failure with float-meter and uint16 inputs at sampling strides 1, 2 and 4. Switched to the tensor CPU backend after verifying the translated plane and RGB. The first route reader assumed a container object for a saved list and stopped after producing the pair map. Corrected the reader to the archived list format. The first matched-grid attempt received float64 depth from scale multiplication. Explicit float32 fixed it, and both final entry points completed in fresh destinations. The final reproduction matched the 21-view raw PLY byte for byte and reproduced the earlier route segment counts and leave-one-out scores. Two complete route diagnoses produced identical CSV, JSON and regional measurement files. Reopened and checked every PLY for expected counts and finite vertices, validated triangle indices, checked image integrity and both frozen input hash manifests, and verified all 317 rows and 159 boundaries under reservation. Fifteen focused repository tests passed with the existing pytest `env` warning. The source recording hash was independently verified. [Verification manifest](../results/db3_measured_depth_20261121_verified/verification.json) · [Open3D tensor TSDF](https://www.open3d.org/docs/release/tutorial/t_reconstruction_system/integration.html)


## 2026-11-21 — Frozen matte Gaussian depth audit and common-mask TSDF

Ran G1 and G2 on the two Week 12 seed-42, 7,000-update checkpoints, the fixed local poses and the original 21 fitting and three RGB-localized reserved views. Verified disjoint fitting and reserved depth-message IDs and the archived checkpoint, split, sensor-sample, dense depth and RGB hashes. The experiment used the existing workstation CUDA and Open3D backend. Training, pose optimization and new captures were outside this run. The reference is aligned D455 axial depth with the archived camera poses. Physical surface accuracy remains unverified. [Frozen protocol](../results/db3_gaussian_depth_20261121/protocol.json) · [Experiment runner](../scripts/db3_gaussian_depth_audit.py).

### G1 — Bias in the rendered expected depth

Pooled the native camera-Z discrepancies on 6,606,674 fitting and 916,892 reserved pixels with stable measured depth and alpha above 0.7 in both arms. Signed discrepancy is rendered minus measured depth. The fitting-frame median is negative in all 21 views for both checkpoints. The reserved-frame aggregate reproduces the prior Week 12 result.

| Views | Arm | Signed median (cm) | Absolute median / p90 (cm) | RGB PSNR on shared pixels (dB) |
| --- | --- | ---: | ---: | ---: |
| 21 fitting | B00 | −5.02 | 5.76 / 18.73 | 37.90 |
| 21 fitting | A05-Abs | −5.82 | 6.58 / 21.08 | 37.52 |
| Three reserved | B00 | −6.64 | 7.37 / 21.62 | 29.55 |
| Three reserved | A05-Abs | −7.33 | 7.97 / 21.25 | 29.95 |

Picked RGB regions before viewing depth residuals. Floor and doorway are visible in 10 fitting views and one reserved view. Region and depth-bin results below use common support. Doorway includes valid edge samples. The JSON also provides a stable-only doorway score and per-view counts. The low-resolution and native mesh figures use separate, explicitly stored evaluation masks.

| Slice | Fitting / reserved pixels | Fitting signed median, B00 / A05-Abs (cm) | Reserved signed median, B00 / A05-Abs (cm) |
| --- | ---: | ---: | ---: |
| Wall or panel interior | 393,909 / 67,007 | −5.34 / −6.37 | −5.56 / −5.15 |
| Floor | 113,020 / 17,507 | −16.39 / −0.68 | −17.56 / −4.03 |
| Doorway boundary | 91,789 / 9,427 | −7.42 / −3.44 | −12.11 / −10.07 |
| Sensor depth 0.3–1.5 m | 31,369 / 0 | −1.19 / −1.33 | — |
| Sensor depth 1.5–3 m | 6,450,163 / 897,178 | −4.93 / −5.82 | −6.45 / −7.28 |
| Sensor depth 3–6 m | 125,142 / 19,714 | −35.73 / −13.06 | −35.74 / −17.19 |

A05-Abs floor absolute medians were 40.10 cm fitting and 35.61 cm reserved despite the smaller signed medians. Large errors in both directions can cancel in the signed statistic. On the 240,264 reserved pixels where both models' mean absolute RGB error was at most 2/255, depth medians remained −5.19 and −6.05 cm. Reserved medians in each model's alpha-above-0.95 bin were −6.78 and −7.41 cm. The low-opacity bin had only 495 B00 and 1,123 A05-Abs stable reserved pixels. High opacity and low RGB residual therefore coexist with a nearward offset.

Computed the alpha-weighted standard deviation of composited Gaussian center depths by rendering first and second moments. Verified the first moment against expected depth. In model-specific reserved bins, a spread of at least 10 cm corresponded to median absolute discrepancies of 11.91 cm for B00 and 9.49 cm for A05-Abs, versus 5.08 and 5.94 cm under 2 cm spread. Depth spread reflects the projected centers that contribute to a ray and also increases on slanted surfaces. It leaves the separate effects of layered geometry, center displacement and visibility weighting unresolved. Saved measured depth, expected depth, alpha, spread, signed depth residuals, and RGB residuals for all 24 images. [G1 measurements](../results/db3_gaussian_depth_20261121/g1_checkpoint_audit.json) · [Region overlays](../results/db3_gaussian_depth_20261121/g1_rgb_regions.jpg) · [Per-pixel evidence](../results/db3_gaussian_depth_20261121/g1_frames/).

**Loss and gradients.** Replayed the trainer's endpoint sampling and confidence-weighted inverse-depth L1 using the archived fitting samples and both endpoint configurations. Camera-Z targets use the parser's world transform scale, and the same factor cancels the inverse-depth normalization. Verified against an explicit inverse-meter expression on every fitting frame. Confidence-zero observations receive zero weight. Nonpositive or nonfinite predictions follow the existing zero-disparity branch, which gives no direct depth gradient through that prediction. All 58,967 positively weighted samples had finite positive predictions and alpha above 0.7 in both endpoints. On those supervised samples, signed medians were −4.83 cm (B00) and −5.51 cm (A05-Abs). The optional sensor alpha penalty was disabled in both saved configurations.

Differentiated the RGB objective and weighted depth objective separately on all 21 fitting frames. At depth weight 0.01, median per-frame depth-to-RGB gradient L2 ratios were 0.80%/0.64% for centers, 1.74%/1.66% for log-scales, 2.71%/2.82% for opacity logits and 1.55%/1.04% for rotations (B00/A05-Abs). Median RGB and weighted depth losses were 0.01542 and 0.000145 for B00, and 0.01650 and 0.000168 for A05-Abs. These are raw endpoint gradients before the optimizer and refinement policy. Their magnitudes motivate a depth-weight comparison. They do not establish gradient ratios over training or the cause of the geometric offset.

### G2 — Paired fusion with identical pixel support

Used 208×115 inputs, the same 21 normalized poses in the same order, one camera model and captured RGB array per view, and the archived uniform Open3D 192³ volume with 8.282404 normalized-unit side and 0.172550 truncation. In G2 the CUDA principal point is shifted by +0.5 in both axes to represent the same physical rays as the integer-centered D455/Open3D convention. G1 retains the training convention. Native filtered measured depths were sampled at nearest rectified pixels with zeros preserved. The mask intersected valid filtered D455 depth and both finite positive Gaussian depths with alpha above 0.7 and the same depth cutoff. All three arms integrated exactly 432,775/502,320 pixels across the 21 views. Inputs before and after masking, RGB, camera matrices and alpha were saved. On the common fitting lattice, pre-fusion B00 and A05-Abs signed expected-depth medians were −6.98 and −8.13 cm.

On the three reserved views, raycast the three mesh first hits against 835,369 common stable sensor pixels. Coverage below uses all 918,331 stable sensor pixels. Direct D455 had a lower absolute median in every reserved view, with a pooled median 58.9% lower than B00.

| TSDF input | Stable sensor coverage | Signed median (cm) | Absolute median / p90 (cm) | Triangles |
| --- | ---: | ---: | ---: | ---: |
| Filtered D455 depth | 94.56% | −1.18 | 3.39 / 9.67 | 54,957 |
| B00 expected depth | 93.23% | −8.21 | 8.25 / 18.43 | 55,621 |
| A05-Abs expected depth | 94.05% | −9.19 | 9.31 / 19.38 | 64,623 |

On 51,731 common low-resolution reserved pixels, using the same rays for checkpoint rendering and mesh first hits, absolute medians moved from 8.67 to 8.25 cm for B00 and from 9.50 to 9.30 cm for A05-Abs. The corresponding p90 discrepancies also fell. Per-view changes varied. This supports expected depth as the main source of the observed bias in the matched comparison. Fusion changed coverage and topology even with identical integration masks. Direct D455 had 6,153 boundary edges versus 9,313 for B00 and 13,383 for A05-Abs. [Input depth maps and masks](../results/db3_gaussian_depth_20261121/g2_inputs/) · [G2 measurements](../results/db3_gaussian_depth_20261121/g2_common_mask_tsdf.json) · [Comparison plot](../results/db3_gaussian_depth_20261121/depth_comparison.png).

**Decision.** G1 and G2 support proceeding to a B00-only matched training comparison of the existing objective and a stronger depth term, followed by robust sensor supervision and sensor-anchored constraints if warranted. Use the same initialization, fixed poses, camera replay and update budget. Predefine geometry, RGB and coverage gates before inspecting candidates. The three exposed reserved views can serve as a fixed reuse check, with an independent recording or reference needed for a confirmation claim. Scope for this run ends with the two frozen-checkpoint diagnostics. The separate route evidence remains 131 passed, nine supported failures and 177 insufficient or invalid intervals. The route diagnosis and suggested bounded checks on intervals 93, 134 and 138 remain in the preceding entry.

**Reproduction and corrections.** With the MSVC/CUDA 12.8 environment from the setup block in [the local pilot runner](../scripts/run-db3-local-pilot.ps1), set OMP_NUM_THREADS to 4 and run from the repository root.

~~~powershell
.venv/Scripts/python.exe -u scripts/db3_gaussian_depth_audit.py results/db3_gaussian_depth_replay
~~~

The first GPU probe omitted the virtual environment's Scripts directory from PATH, so the JIT could not locate the installed Ninja. The local setup recovered it. The first audit preflight compared every current source digest with the historical training protocol and stopped at the changed Week 12 driver. The final audit verifies training artifacts and data against archive hashes and records both digest pairs for the Week 12 driver and trainer. The renderer, loss, dataset loader and strategy source hashes match the training protocol, and the endpoint loss calculation was reconstructed from the archived configuration. Independent readback verified 62 artifact and 170 input hashes and recomputed the primary G1 and G2 statistics and integration masks from the arrays. The six normalized and local-metric PLYs were exported after validating the normalized meshes' triangle indices and finite vertices. Python compilation and git diff --check passed. The focused loss and sensor-confidence suite passed 26 tests with the existing pytest env warning. [Artifact hashes](../results/db3_gaussian_depth_20261121/artifacts.json) · [Independent readback](../results/db3_gaussian_depth_20261121/readback.json).


A fresh end-to-end replay verified all 62 files and reproduced 61 byte for byte, including every saved depth map, mask, mesh and G2 metric. The remaining G1 JSON agrees exactly on depth, opacity, and RGB statistics and agrees at reported precision on gradient ratios. The largest absolute per-frame difference was 5.17e-7 for the depth-to-RGB gradient norm ratio and 4.24e-5 for gradient cosine. [Replay readback](../results/db3_gaussian_depth_20261121/readback.json).

## 2026-11-21 — G3 controlled B00 depth-supervision pilot

**Protocol frozen before candidate training.** Repeat the Week 12 matte B00 training command at inverse-depth coefficients 0.01, 0.03, 0.10 and 0.30 (G3-A through G3-D). Keep the 90,000-point initializer, 21 training and three reserved development views, camera poses and 7,000-image replay, seed 42, RGB loss, default densification and Week 12 uniform TSDF extractor and settings. Add pre-update snapshots indexed by completed updates at 0, 100, 500, 501, 1,000, 2,000, 3,000, 3,001, 4,500, 5,901, 6,000 and 6,500, plus the existing post-update final at 7,000. The 501–1,000 interval spans the first densification at zero-based step 600. The 3,000–3,001 pair brackets the coincident opacity reset and densification. Compare native rendered axial Z and final mesh first hits to stable D455 depth. Report four-arm and pairwise common support, each arm's own coverage, full-frame RGB, separate term gradients and finite state checks.

**Development gates.** Require the 0.01 replay to agree with archived B00 within 0.25 cm on depth summaries, 0.15 dB PSNR, 0.005 SSIM, 0.5 percentage points coverage and 5% final Gaussian count on the matching checks. A stronger coefficient qualifies for a follow-up if, on pairwise common stable sensor pixels, both rendered depth and mesh first hits reduce pooled signed-median magnitude and pooled median absolute discrepancy by at least 1 cm, with p90 absolute discrepancy no higher and absolute medians lower in all three development views. Full-frame RGB may lose at most 0.5 dB PSNR and 0.01 SSIM. Each depth source may lose at most 1 percentage point coverage. Apply the RGB and coverage limits to the pooled and per-view results. Require finite saved parameters and probe gradients. These pilot thresholds are experiment decisions. The development views were previously inspected, and reliable confirmation requires independent observations. [Frozen protocol](../results/db3_g3_20261121/protocol.json) · [Training runner](../scripts/db3_g3_depth_sweep.py)

**Gradient scale correction.** G1 expressed the depth objective in inverse meters using the 0.9879909199 world-to-normalized scale. The trainer multiplies inverse normalized depth by its separate scene scale of 2.0813651241. Multiply the G1 weighted depth losses and depth-to-RGB gradient ratios by 2.1066642235 to recover the trainer scale. Gradient cosine, sensor discrepancy, RGB and coverage are unaffected. G3 uses the trainer scale. The initial G3-A probe on the first replayed camera reproduced the logged depth loss of 0.02529683 before coefficient weighting and the total loss of 0.21272545 after weighting.

### Final development comparison

All four arms completed 7,000 updates using the same 90,000-Gaussian state digest, seed, and exact sequence of 7,000 camera indices. Checked the saved configuration against the archive. Differences were the coefficient, output path and checkpoint schedule, plus the current trainer's inactive elapsed-time evaluation field. The archived extractor settings matched on the parser transform, all 21 integration cameras and their order, the 208×115 raster, 0.7 opacity threshold and 192³ uniform volume. Training took 228.1, 221.9, 225.8 and 227.1 s including setup and checkpoint I/O. These measurements are execution records. [Run protocol and outputs](../results/db3_g3_20261121/) · [Endpoint scores and extractor reports](../results/db3_g3_20261121/evaluation/summary.json)

The rendered-depth comparison shares 912,960 native pixels across the four arms. The mesh first-hit comparison shares 863,163. Coverage uses all 918,331 stable sensor pixels. RGB uses the mean of the three full-frame scores. Depth entries give signed median and median / p90 absolute discrepancy in centimeters, with negative signed values toward the camera.

| Arm | Coefficient | Render depth (cm) | Render coverage | Mesh depth (cm) | Mesh coverage | PSNR / SSIM | Gaussians |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G3-A | 0.01 | −6.34, 7.17 / 21.01 | 99.98% | −8.10, 8.17 / 18.81 | 97.68% | 28.910 / 0.8847 | 482,261 |
| G3-B | 0.03 | −3.97, 5.35 / 18.75 | 99.97% | −6.29, 6.43 / 16.96 | 97.56% | 28.507 / 0.8825 | 476,184 |
| G3-C | 0.10 | −1.96, 4.08 / 15.63 | 99.50% | −4.61, 5.08 / 15.28 | 95.46% | 28.570 / 0.8775 | 501,745 |
| G3-D | 0.30 | −1.68, 3.39 / 14.42 | 99.95% | −4.37, 4.75 / 14.80 | 98.25% | 28.577 / 0.8798 | 510,261 |

G3-A repeated the nearward bias, with −6.31 cm versus −6.65 cm for the archive on 917,713 shared render pixels. Render absolute p90 was 21.25 versus 21.67 cm and mesh absolute p90 was 19.46 versus 20.12 cm on the corresponding paired support. Both p90 differences exceed the 0.25 cm replay tolerance. Per-view depth, RGB and mesh coverage produced further replay misses. Pooled RGB and final count met their replay limits. The available committed trainer diff adds optional timed evaluation, which was inactive. The archive and current trainer and driver digests differ. The source of endpoint variation was unresolved.

All three stronger coefficients passed the pairwise geometry thresholds for both depth outputs, including lower absolute medians in every reserved view. G3-B lost 0.96 dB PSNR on `rgb_0006487.png`. G3-C lost 1.27 dB PSNR, 0.0207 SSIM, 1.28 percentage points of render coverage and 6.64 points of mesh coverage on `rgb_0010545.png`, with 2.22 points of pooled mesh coverage loss. G3-D had the best pooled depth agreement and the highest mesh coverage. It lost 0.58 dB PSNR on `rgb_0010545.png`, where paired rendered-depth p90 rose from 30.20 to 47.97 cm even as the absolute median fell from 13.69 to 8.61 cm. G3-D used 5.81% more Gaussians than G3-A. The LPIPS means were 0.1717, 0.1738, 0.1811 and 0.1814. The meshes contained 76,752/82,770/76,087/85,105 triangles and 9,166/11,744/12,223/15,217 boundary edges, in coefficient order.

### When the bias developed and how the loss behaved

On four-arm shared development support, signed G3-A error was −1.89 cm at initialization, −2.59 cm at 100 updates and −2.69 cm at 500, before the first split/prune step. On the same G3-A pixels before and after the 501–1,000 interval, signed error shifted from −2.71 to −3.27 cm. It reached −5.02 cm on four-arm support by 3,000 and −6.21 cm when densification stopped at 6,000, then reached −6.34 cm at 7,000. Fitting-view bias followed a similar pattern through densification, starting at −1.40 cm, reaching −4.05 cm at 3,000 and finishing at −4.96 cm. The snapshots locate early and later growth in the bias, and the interval comparisons include optimizer updates as well as densification. G3-D held the development signed median in the −0.86 to −1.76 cm range from 500 updates through the measured final trajectory with valid common support.

At 3,001 completed updates, the opacity reset reduced development coverage above 0.7 alpha to at most 0.0053% in every arm and left zero shared pixels. By the 4,500 snapshot, coverage had returned to at least 99.42%. Across 21 fitting probes per checkpoint, median weighted depth-to-RGB center gradient ratios just before the reset were 1.42/3.57/9.57/32.60%, rose immediately after to 8.60/20.81/89.39/282.45%, and finished at 1.69/3.06/9.56/31.57%. Final opacity-logit ratios were 5.92/5.88/12.35/34.17%. These are separate raw objective gradients, upstream of Adam and refinement. All saved parameter tensors, probe gradients and the 70 regular training-scalar samples per arm were finite.

The actual G3-C camera at zero-based step 3,001 was `rgb_0004823.png`. The recorded objective jumped to 14.460 with unweighted depth loss 140.04755. A replay of that saved pre-update state found that pixel (359, 101) contributed 99.95% of the depth term. Its predicted Z was 0.013 mm, alpha was 7.34 × 10⁻⁸, and the target was 1.830 m with confidence 0.1445. The weighted center gradient was about 12,044 times the RGB center gradient on that frame. This was a finite loss spike driven by a nearly transparent ray with a tiny positive prediction. The same camera's weighted depth loss was 0.000807 before the reset at 3,000 and 0.000568 at the 4,500 snapshot. [Trajectory plot](../results/db3_g3_20261121/evaluation/g3_trajectory.png) · [Training trajectory](../results/db3_g3_20261121/evaluation/trajectory.json) · [Per-view gradient probes](../results/db3_g3_20261121/evaluation/gradients.json) · [Trainer scalars](../results/db3_g3_20261121/evaluation/training_scalars.json) · [Opacity-reset outlier](../results/db3_g3_20261121/reset_outlier.json)

### Decision and replay

The pilot shows a dose response in agreement with the D455, with geometry gains from stronger supervision and a per-view RGB penalty in every candidate. Zero arms met the full frozen gate, and the strict archived-control replay also missed its tolerances. The 0.30 arm is the strongest geometry result of this sweep. Its render tail error and higher Gaussian count are relevant to the research objective. The reset spike identifies a robustness target for a later study of confidence and rendering support. This single seed and the three reused development views provide evidence of sensor consistency. Independent observations and physical calibration remain needed for a reliability or absolute accuracy claim. A05-Abs transfer remains pending.

Reproduce from the repository root with the MSVC/CUDA 12.8 environment in the setup block of [the local pilot runner](../scripts/run-db3-local-pilot.ps1). Use OMP_NUM_THREADS=4, TQDM_MININTERVAL=30 and a fresh output directory.

```powershell
.venv/Scripts/python.exe -u scripts/db3_g3_depth_sweep.py results/db3_g3_replay
.venv/Scripts/python.exe -u scripts/db3_g3_evaluate.py results/db3_g3_replay
```

The separate archive replay probe first omitted MSVC from PATH and stopped before rendering. It ran successfully with the same environment as training. The first independent comparison recalculated the masks and statistics but failed when serializing a NumPy count. Converting the count to a Python integer completed readback. Verified the final PLYs, saved frame maps, pairwise and four-arm depth statistics, camera sequence and initializer digests, and all 52 checkpoints. The focused depth and confidence suite passed 26 tests with the existing pytest `env` warning. Python compilation and git diff --check passed. [Independent readback](../results/db3_g3_20261121/evaluation/readback.json) · [Evaluation runner](../scripts/db3_g3_evaluate.py)

## 2026-11-21 — G4 robust, geometry-preserving RGB-D supervision

**Protocol recorded before candidate endpoints.** Use the frozen G3-A (0.01) and G3-D (0.30) checkpoints and meshes as G4-A and G4-B. The G3 readback pins the endpoint hashes. Repeat the seed-42 7,000-update matte training command and G3 checkpoint schedule for C, D and E. C rejects only nonfinite or sub-0.30 m predicted camera-Z before taking the reciprocal, using the existing dataset lower range limit. It retains the original evidence-weight denominator and uses no opacity cutoff. D adds a log1p penalty on the absolute inverse-depth residual with scale 0.025 inverse meters (approximately 10 cm at 2 m). It preserves the small-residual L1 slope and downweights large residual gradients. E repeats the selected C or D formulation at 0.10. Select by fewest original gate failures against A, then by smallest worst per-view p90 increase across render and mesh, then by smallest worst per-view PSNR loss, favoring C on an exact tie. Count per-step positively weighted and active sensor samples, probe the reset on the same saved camera/state, and score recovery of visible surface coverage and depth. Keep original G3 pooled and per-view gates and report per-view p90 separately, especially for rgb_0010545.png. [Frozen protocol](../results/db3_g4_20261121/protocol.json) · [Training runner](../scripts/db3_g4.py).

**Archive recovery.** Searched workspace snapshots, the final report supplement and Git blobs, including alternate line endings. Six DB3 snapshots match the historical trainer SHA-256. The Week 12 driver copies examined did not match its archived digest. The training protocol preserves the exact B00 argument list. Launched the matching archived trainer with that list, original checkpoint schedule and verified inputs using the existing workstation virtual environment. Redirected its output to a fresh directory and recorded the current package inventory. Exact historical environment identity and end-to-end driver replay remain unverified. The historical B00 comparison is a separate replay diagnostic.

Recovered the Week 12 driver from its original creation patch after the first trainer replay launched. Its exact bytes match the archived driver hash. Restored all nine archive-manifest sources to an isolated replay directory and located the original PowerShell launcher. The B00-only driver replay uses that launcher and the same virtual environment path. Windows SetupAPI records removal of NVIDIA 621.14 at 21:56 and installation of 621.24 at 22:11 local time on November 19. The archived B00 verification file was written at 12:08 that day. The historical training process did not pin its complete runtime. The 621.24 installation postdates training, so source and command agreement alone cannot establish an exact environment match or assign the cause of endpoint differences. [Source and environment provenance](../results/db3_g4_20261121/archive_provenance.json) · [Recovered driver](../results/db3_g4_20261121/archive_source/scripts/week12_comparison.py).

**Archived-source replay outcome.** Both isolated B00 repeats finished 7,000 updates with the archived initializer and exact camera sequence. The first repeat used the recovered trainer and recorded command directly. The second used the recovered driver, original launcher, complete nine-file source manifest, original save/eval schedule and original inputs. Driver-generated argv matched the archived B00 command after resolving input aliases and redirecting output. The original configuration matched apart from those paths. The recovered-driver result was 28.740 dB versus the archived 28.872 dB, with 468,822 versus 485,156 Gaussians. On pairwise common pixels, rendered absolute median/p90 were 7.03/21.11 versus 7.37/21.65 cm, and mesh median/p90 were 8.17/18.72 versus 8.36/20.05 cm. It missed 11 grouped replay checks, including both pooled depth checks and per-view RGB and mesh coverage. The direct-command repeat missed 10 grouped checks. Exact source recovery did not restore numerical interchangeability. The cause remains unresolved. All G4 attribution uses G4-A for control and G4-B for the strong reference. [Archive scores](../results/db3_g4_20261121/archive_scores.json) · [Driver-generated protocol](../results/db3_g4_20261121/driver_replay/matte_training_protocol.json) · [Replay environment](../results/db3_g4_20261121/driver_replay/environment.json).

**Selection before running E.** Both 0.30 candidates passed the original endpoint criteria against A. Across the six rendered and mesh per-view p90 comparisons, the smallest improvement was 0.85 cm with C and 1.26 cm with D. The worst PSNR losses were 0.382 and 0.232 dB, respectively. The frozen tie-break selected D's 0.025 inverse-meter log1p residual for E at 0.10. [Selection](../results/db3_g4_20261121/selection.json).

### G4 endpoint decision

All three new runs completed 7,000 updates from the same 90,000-point initialization and the same 7,000 camera indices. The table uses 916,603 stable pixels shared by all five renders and 868,406 shared mesh first hits. Each coverage denominator is all 918,331 stable sensor pixels. Full-frame PSNR and SSIM are averaged over the three development views. Depth entries are signed median and absolute median / p90 in cm relative to D455 axial depth. Decisions use **pairwise** shared support against A. A and B are the frozen G3-A and G3-D artifacts.

| Arm | Coefficient | Render signed, median / p90 (cm) | Mesh signed, median / p90 (cm) | Render / mesh coverage | PSNR / SSIM | Gaussians | Original endpoint gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| G4-A | 0.01 | −6.32, 7.20 / 21.20 | −8.16, 8.22 / 19.13 | 99.98% / 97.68% | 28.910 / 0.8847 | 482,261 | Control |
| G4-B | 0.30 | −1.71, 3.41 / 14.81 | −4.42, 4.79 / 14.91 | 99.95% / 98.25% | 28.577 / 0.8798 | 510,261 | Missed one per-view PSNR limit |
| G4-C | 0.30 | −1.48, 3.33 / 13.24 | −4.30, 4.74 / 14.87 | 99.93% / 98.38% | 28.655 / 0.8802 | 509,834 | Passed |
| G4-D | 0.30 | −1.32, 3.52 / 13.40 | −4.20, 4.73 / 14.62 | 99.95% / 97.63% | 28.798 / 0.8822 | 508,776 | Passed |
| G4-E | 0.10 | −2.39, 4.40 / 16.63 | −5.16, 5.50 / 15.76 | 99.98% / 96.21% | 28.763 / 0.8809 | 493,332 | Missed pooled and one per-view mesh coverage limit |

**Tail and RGB on rgb_0010545.png.** On pixels common to B and D, rendered p90 fell from 47.87 to 22.05 cm and mesh p90 from 22.25 to 17.03 cm. Full-frame PSNR rose from 26.90 to 27.31 dB, within 0.17 dB of A. On pixels common to A and D, render median/p90 went from 13.70/30.29 to 8.44/22.18 cm and mesh median/p90 from 13.74/23.86 to 10.47/17.37 cm. C also reduced the tail on this view. Both C and D reduced render and mesh absolute medians and p90 on every view against A and passed the pooled bias, depth, RGB, coverage and sampled finiteness checks. Against B on common support, D gave up 0.10 cm in pooled rendered absolute median and improved rendered p90 by 1.43 cm, mesh absolute median by 0.11 cm and mesh p90 by 0.82 cm. [Pairwise and five-arm scores](../results/db3_g4_20261121/final_scores.json) · [Comparison figure](../results/db3_g4_20261121/g4_comparison.png).

**Lower coefficient.** Relative to A, E lost 1.46 percentage points of pooled mesh coverage and 4.38 points on rgb_0010545.png. Its rendered p90 on that view increased from 30.38 to 33.90 cm on A/E support. Compared directly with D on the same pixels, E's pooled rendered median/p90 rose from 3.52/13.47 to 4.41/16.70 cm. Its mesh median/p90 rose from 4.77/14.84 to 5.56/16.05 cm. Mean full-frame PSNR moved from 28.798 to 28.763 dB. This 0.10 run offered no improvement in the measured RGB/geometry trade-off.

### Reset mechanism and recovery

Re-rendered the exact G3-C pre-update state and camera at step 3,001 with each loss at **the same 0.30 coefficient**. The original G3-C training coefficient was 0.10. Its saved weighted objective was reproduced after scaling back to 0.10. Of 1,211 positively weighted samples, the ray at (359, 101) had predicted depth 0.01315 mm, target 1.830 m and alpha 7.34e-8. Under legacy L1 it supplied 99.95% of the depth objective. Gradient ratios below are weighted depth / RGB L2 for Gaussian centers, before Adam. The opacity-only variant was evaluated on this frozen state.

| Loss on the identical state | Active samples | Weighted depth loss | Center gradient ratio | Outlier share |
| --- | ---: | ---: | ---: | ---: |
| Legacy inverse-depth L1 | 1,211 | 42.0142 | 36,131× | 99.95% |
| Alpha > 0.7 only | 0 | 0 | 0 | 0 |
| C, predicted Z ≥ 0.30 m | 1,210 | 0.02012 | 2.15× | 0 |
| D, same depth rule plus robust residual | 1,210 | 0.01098 | 0.94× | 0 |

The depth rule zeroed the pathological ray's derivative and retained 99.92% of supported samples on the historical spike. On each of the three **new** trajectories, all 19,656,796 positively weighted observations across 7,000 updates passed the rule, including every post-reset training observation. New endpoint differences therefore cannot be attributed to the guard rejecting training rays. The same-state test isolates the guard's effect where its threshold actually fires. The largest single-sample fraction of a frame's depth loss across each complete trajectory was 5.29% (C), 2.22% (D) and 1.17% (E). [Same-state gradients and counts](../results/db3_g4_20261121/same_state_reset.json).

At 3,001 updates, only 1, 1 and 0 of 58,967 fitting samples exceeded alpha 0.7 in C, D and E, while all 58,967 remained available for depth supervision. Development render coverage above alpha 0.7 collapsed to 0.00022%, 0.00403% and 0.00392%, respectively. By 4,500 updates it returned to 99.90%, 99.94% and 99.97%, with own-support absolute median discrepancies of 3.43, 3.61 and 4.62 cm. Finite checkpoints, loss telemetry at every update, and separate RGB/depth gradients at all 13 audit checkpoints across 21 fitting cameras support the recovery check. Final meshes used the original integration views, raster and TSDF settings. [C recovery](../results/db3_g4_20261121/G4-C_recovery.json) · [D recovery](../results/db3_g4_20261121/G4-D_recovery.json) · [E recovery](../results/db3_g4_20261121/G4-E_recovery.json).

**Decision.** Carry G4-D at 0.30 as the development candidate for independent replication. C also passed the frozen within-version thresholds and had a slightly lower pooled rendered absolute median. E failed the coverage gate. The archived-source replay still fails its strict endpoint tolerances, the seed is shared, and the three development views have been examined repeatedly. The measurements establish consistency with the aligned sensor on this split. Independent repeats and independently reserved observations are needed for a verified improvement claim. Physical accuracy remains unmeasured.

**Verification and reproduction.** Read back the three saved evaluation arrays and independently recomputed 184 per-view and pooled depth/coverage rows across the five-arm, A-paired, B-paired and D/E comparisons. Rechecked G3 control maps, full-frame aggregate RGB and Gaussian counts, input and checkpoint digests, sampled versus logged objectives, actual camera replay, and finite gradient tensors. The loss and sensor-confidence suites passed 236 tests with the existing pytest configuration warning. The G4-specific checks cover the 0.013 mm input, empty evidence, nonfinite predictions, safe gradients, world-to-meter scaling, preserved normalization, and agreement with the legacy loss on supported rays. [Readback and artifact hashes](../results/db3_g4_20261121/readback.json) · [Evaluations](../scripts/db3_g4_evaluate.py) · [Recovery probes](../scripts/db3_g4_recovery.py) · [Independent scorer](../scripts/db3_g4_readback.py).

For a fresh G4 run, execute the following from the repository root with the existing dataset and frozen G3 artifacts. The archive diagnostic can also be repeated with the restored source tree and original launcher using the command below. Its B00 mesh can be extracted with the original G3 extraction arguments after changing the checkpoint and destination; the archive evaluator reads the original mesh and replay mesh report from the corresponding folders.

~~~powershell
$r = 'results/db3_g4_reproduction'
$env:OMP_NUM_THREADS = '4'
./scripts/run-school-completion.ps1 -u scripts/db3_g4.py $r G4-C G4-D
./scripts/run-school-completion.ps1 -u scripts/db3_g4_evaluate.py $r select
./scripts/run-school-completion.ps1 -u scripts/db3_g4.py $r G4-E
./scripts/run-school-completion.ps1 -u scripts/db3_g4_evaluate.py $r final
./scripts/run-school-completion.ps1 -u scripts/db3_g4_recovery.py $r
./scripts/run-school-completion.ps1 -u scripts/db3_g4_readback.py $r
~~~

In a fresh shell with the original launcher defaults, use the following for the archival diagnostic in that reproduction folder.

~~~powershell
$r = 'results/db3_g4_reproduction'
./scripts/run-school-completion.ps1 -u scripts/db3_g4_archive_replay.py results/db3_g4_20261121/archive_source "$r/driver_replay"
./scripts/run-school-completion.ps1 -u scripts/gaussians_to_mesh.py --checkpoint "$r/driver_replay/replication/matte/B00_s42/ckpts/final_post_update.pt" --data results/db3_r2_r3_20261110/matte/pilot/dataset --output "$r/meshes/driver_replay" --views 64 --downsample 4 --voxel 0.01 --max-resolution 192
$env:OMP_NUM_THREADS = '4'
./scripts/run-school-completion.ps1 -u scripts/db3_g4_evaluate.py $r archive
~~~

## 2026-11-21 — G5 independent replication of robust depth supervision

**Protocol before endpoints.** Run five fresh paired seeds (43–47), with four contemporaneous B00-allocation arms per seed. A and B use legacy confidence-weighted inverse-depth L1 at 0.01 and 0.30. C adds the 0.30 m predicted-Z rule to B. D adds the G4-D log1p penalty with scale 0.025 inverse meters to C. Keep the existing 21 fitting cameras, their measured-depth input and frozen 90,000-point initializer, the same three development cameras, 7,000 updates and the G4 extraction settings. Freeze balanced shuffle sequences independently by seed and share each sequence across its four arms. Rotate arm order. Save pre/post-reset and recovery checkpoints and log each update's training objectives. Evaluate paired render and mesh discrepancies on common stable sensor pixels, full-frame PSNR/SSIM and coverage on all stable pixels. Apply the inherited pooled and per-view gates to D versus A and require each evaluated view's p90 to stay at or below A. Compare C versus B and D versus both B and C to distinguish the guard and robust penalty. Measure finiteness, loss peaks and coverage recovery to within 1 percentage point of pre-reset coverage and at least 99% by 4,500 updates. Freeze source, input and environment details in the protocol. The historical archive replay misses remain documented in G4. Generalization requires untouched localized observations, and A05-Abs transfer follows G5 acceptance. [Protocol](../results/db3_g5_20261121/protocol.json) · [Runner](../scripts/db3_g5.py)

**Untouched observations reserved before G5 scoring.** The original recording contains 69 eligible unused RGB-D pairs in 12 temporal strata after excluding R1 selected frames, R2 prepared observations, R4 loaded inputs and the local native trials. The 17.13 GB recording matched its original SHA-256. A supplementary roster comparison against R2 transition/revisit and R4 track-recovery inputs found zero selected RGB overlaps. The reservation selected each stratum's first RGB-only pose passing its frozen checks. Eight of 25 attempted poses passed, spanning 7.743–13.483 s. Each had 22–114 held-out landmark matches, 0.25–0.56 px median and 0.59–1.21 px p90 reprojection error. The two disjoint spatial fits differed by 0.22–1.91 cm and 0.04–0.52 degrees across accepted views. The fixed metric map supplied RGB landmarks. Sealed poses before reading reserved depth. Actual RGB/depth timestamp gaps were 7.02–9.03 ms. This extends evaluation to unused observations from the same recording and trajectory. The RealSense SDK reported zero connected devices, so no independent acquisition or physical measurements were added. [Reservation and provenance](../results/db3_g5_20261121/holdout/reservation.json) · [RGB-only localization checks](../results/db3_g5_20261121/holdout/localization.json) · [Sealed RGB-D test split](../results/db3_g5_20261121/holdout/frames.json) · [Scoring protocol](../results/db3_g5_20261121/evaluation_protocol.json)

**Pipeline check on the first paired seed.** The focused confidence/depth suite passed 31 tests with the existing pytest configuration warning. Replayed the scorer on sealed G4 maps and reproduced all six pooled A-paired results and 128 independent readback rows. The first finished G5 seed independently verified 416 scored depth/coverage rows across both splits. D passed development and recovery gates at seed 43. On the eight reserved views, D lowered paired pooled median/p90 by 3.06/7.55 cm for rendered depth and 3.12/5.07 cm for mesh depth. PSNR on view 7449 fell from 30.144 to 29.526 dB and missed the 0.5 dB limit. The pooled and per-view split comparisons use 2,573,425 stable sensor pixels on the new observations. The first-seed holdout result leaves the strict all-seed promotion gate unmet. Finish the full frozen series to measure recurrence and robust-loss attribution. [Seed 43 scores](../results/db3_g5_20261121/s43/evaluation/scores.json) · [Readback runner](../scripts/db3_g5_readback.py)

**Workstation interruption.** Windows entered low power at 03:32:40 and woke at 06:08:51 (Asia/Saigon) during s45/D. The same trainer process resumed, reached update 7,000, and produced its validated checkpoint and mesh. The NVIDIA driver remained 621.24 after wake. The recorded s45/D wall time includes about 2 h 36 min of sleep and is unsuitable for timing comparisons. Held a process-scoped Windows system-required power request from 06:10 through final scoring to deter further idle sleep. The request releases when the scoring wrapper exits.

### Five-seed endpoint results

All 20 fits completed 7,000 updates on Python 3.12.10, PyTorch 2.9.1+cu128, CUDA 12.8 and NVIDIA 621.24 on the GTX 1660 SUPER. Within each seed the four initializer digests and all 7,000 camera indices matched. The mesh audit matched all 21 integration views, parser transform, alpha threshold, TSDF bounds, voxel size and truncation to G4. Scores use pairwise common depth support, per-arm coverage on all stable pixels and full-frame RGB. The paired depth deltas below are later arm minus reference in cm, reported as mean ± sample SD over five seeds. In brackets is the number of seed pairs with lower discrepancy.

| Split | Contrast | Render absolute median | Render absolute p90 | Mesh absolute median | Mesh absolute p90 |
| --- | --- | ---: | ---: | ---: | ---: |
| Development | B − A | −3.57 ± 0.14 [5] | −6.07 ± 1.68 [5] | −3.28 ± 0.21 [5] | −3.34 ± 1.07 [5] |
| Development | C − B | +0.05 ± 0.04 [1] | +0.34 ± 0.82 [1] | −0.00 ± 0.09 [3] | +0.03 ± 0.94 [3] |
| Development | D − A | −3.42 ± 0.17 [5] | −5.65 ± 0.93 [5] | −3.27 ± 0.20 [5] | −3.50 ± 0.46 [5] |
| Development | D − B | +0.15 ± 0.04 [0] | +0.43 ± 0.92 [1] | +0.00 ± 0.08 [2] | −0.09 ± 0.63 [3] |
| Development | D − C | +0.10 ± 0.02 [0] | +0.05 ± 0.86 [2] | −0.00 ± 0.04 [2] | −0.24 ± 0.52 [4] |
| Unused observations | B − A | −3.01 ± 0.25 [5] | −8.21 ± 0.90 [5] | −2.90 ± 0.21 [5] | −4.81 ± 0.64 [5] |
| Unused observations | C − B | −0.01 ± 0.02 [2] | +0.01 ± 0.21 [3] | −0.02 ± 0.03 [4] | +0.02 ± 0.41 [4] |
| Unused observations | D − A | −2.95 ± 0.26 [5] | −7.91 ± 1.03 [5] | −2.89 ± 0.20 [5] | −4.82 ± 0.34 [5] |
| Unused observations | D − B | +0.07 ± 0.02 [0] | +0.30 ± 0.22 [0] | +0.02 ± 0.04 [2] | +0.04 ± 0.38 [3] |
| Unused observations | D − C | +0.07 ± 0.01 [0] | +0.28 ± 0.19 [1] | +0.03 ± 0.02 [0] | −0.07 ± 0.16 [3] |

**Reproducibility and attribution.** D improved all four pooled sensor-discrepancy metrics over A on both splits in 5/5 seeds. B did the same, so the 0.30 L1 control reproduces the pooled geometry gains. D had higher pooled rendered absolute median than both strong controls in all five seeds on both splits. Mesh effects between the 0.30 arms were small and mixed. D recovered mean PSNR relative to B and C by +0.186 and +0.143 dB on development and +0.104 and +0.095 dB on unused observations, respectively. C excluded zero evidence samples over five trajectories. D excluded one of 98,279,887 at zero-based step 3,005 in seed 44 on fitting frame rgb_0004823.png. These trajectories provide little opportunity for the threshold itself to affect fitting. [Paired summaries and directions](../results/db3_g5_20261121/summary.json).

**Generalization and trade-offs.** D passed the full development endpoint gate in seeds 43, 44 and 47. It passed the endpoint gate on unused observations in 0/5 seeds. D had mean PSNR/SSIM changes of −0.020 dB/−0.00151 on development and +0.054 dB/+0.00007 on the unused observations against A. Mean pooled render/mesh coverage changes were −0.017/+0.988 percentage points on development and +0.004/+0.466 points on unused observations. The view-specific losses still matter. The PSNR limit was 0.50 dB and the per-view p90 limit was zero increase. No SSIM rule failed for D.

| Seed | D minus A development exceptions | D minus A unused-observation exceptions | A count | B count | C count | D count |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 43 | Passed | 7449 PSNR −0.62 dB | 475,787 | 521,156 | 512,813 | 508,146 |
| 44 | Passed | 7449 PSNR −0.79 dB | 485,650 | 509,966 | 516,299 | 502,748 |
| 45 | 0010545 render p90 +0.28 cm | 7449 PSNR −0.70 dB | 484,387 | 520,997 | 515,553 | 512,507 |
| 46 | 0008463 PSNR −0.58 dB. 0010545 render p90 +4.19 cm and mesh coverage −4.46 pp. Pooled mesh coverage −1.54 pp. | 10649 render p90 +15.07 cm and mesh coverage −8.91 pp. Pooled mesh coverage −1.08 pp. | 482,112 | 513,302 | 518,822 | 511,168 |
| 47 | Passed | 7449 PSNR −0.98 dB. 10649 mesh p90 +0.46 cm. | 492,355 | 505,180 | 507,903 | 506,943 |

D increased count over A in 5/5 seeds, averaging +24,244 Gaussians and +5.02% ± 1.68 percentage points. It averaged 5,818 fewer Gaussians than B and 5,976 fewer than C. Mean endpoint counts were 484,058, 514,120, 514,278 and 508,302 for A, B, C and D. [Seed 43](../results/db3_g5_20261121/s43/evaluation/scores.json) · [Seed 44](../results/db3_g5_20261121/s44/evaluation/scores.json) · [Seed 45](../results/db3_g5_20261121/s45/evaluation/scores.json) · [Seed 46](../results/db3_g5_20261121/s46/evaluation/scores.json) · [Seed 47](../results/db3_g5_20261121/s47/evaluation/scores.json).

**Reset behavior.** Every arm in every seed passed the coverage recovery rule. D fell to 0.00011–0.00447% pooled render coverage after the reset at 3,001 completed updates, recovered 97.05–98.97% by 3,101 and reached 99.873–99.937% by 4,500. In the first 100 post-reset updates the peak weighted depth term was 0.0128–0.0159 for D, 0.0242–0.0357 for B and 0.0258–0.0300 for C. D had the lowest peak among the 0.30 arms in every seed in that interval. Over all 7,000 updates the largest single-sample loss shares across seeds were 3.33% for D and 11.52% for C. Scalar histories and checkpoint probes were finite. Coverage recovery measured from actual renders passed across all four formulations.

**Decision.** The pooled agreement improvement replicates on unused observations of this same capture. The robust penalty improves some RGB outcomes and suppresses reset-window loss peaks, with mixed geometry effects versus the two 0.30 controls. The recurrent PSNR loss on 7449 and the tail/coverage losses on 10649 fail the frozen acceptance criteria. The B00 × A05-Abs transfer remains gated. The archived-source B00 replay still falls outside its tolerance. A new acquisition and independent physical measurements remain pending, and the SDK again found zero connected RealSense devices at closeout. [Decision and sealed seed reports](../results/db3_g5_20261121/summary.json) · [Historical replay metrics](../results/db3_g4_20261121/archive_scores.json).

**Verification and replay.** Independently recomputed 2,080 pooled and per-view depth/coverage rows from saved layers and matched 20 trainer RGB and count endpoints. Verified training order, per-seed initializer identity, extracted mesh and checkpoint digests, stored phase telemetry and SHA-256 seals. The cached gsplat CUDA extension digest matched before and after the training campaign. The power request released after scoring and readback. Commands below use a fresh result directory and the same frozen dataset, original DB3 recording and G3/G4 recipes. The protocol retains each training and extraction command.

~~~powershell
$env:OMP_NUM_THREADS = '4'
$r = 'results/db3_g5_reproduction'
./scripts/run-school-completion.ps1 -u scripts/db3_g5.py $r
./scripts/run-school-completion.ps1 -u scripts/db3_g5_holdout.py "$r/holdout"
./scripts/run-school-completion.ps1 -u scripts/db3_g5_evaluate.py $r --freeze-only
./scripts/run-school-completion.ps1 -u scripts/db3_g5_evaluate.py $r
./scripts/run-school-completion.ps1 -u scripts/db3_g5_readback.py $r
~~~

[Readback verification](../results/db3_g5_20261121/readback.json) · [Frozen runtime and inputs](../results/db3_g5_20261121/protocol.json) · [Evaluator](../scripts/db3_g5_evaluate.py).

## 2026-11-21 — Training artifact storage audit and cleanup proposal

D: had 13.56 GiB free at inspection. The repository contained 29,516 result files totaling 101.45 GiB and 161,306 files under `data/` totaling 55.55 GiB by file length. Those folder totals count every hard-linked path. The separate original RealSense recording `20261030_134523.db3` is 15.95 GiB. Git tracks the report and its packaged evidence, and ignores the local results, datasets and recording. No Python trainer, Blender, FFmpeg or COLMAP process was visible at inspection.

The largest result folders were the October A04 research campaign at 30.87 GiB, the G5 paired-seed study at 12.38 GiB, and the controlled final-report Room study at 4.62 GiB. The final-report study runner reads the archived October B00 Room run. The research artifact verifier checks final rendered PNG sets, PLYs and saved models. The G3–G5 audit scripts and seals rely on named checkpoints, saved maps and mesh exports. Keep the final report and supplement, source recordings, prepared datasets under `data/`, unique measured-depth outputs, and research evidence for those studies and Week 11–12.

Scanned regular files of at least 256 KiB under `results/` and `data/`, grouping by size and comparing sampled bytes before full SHA-256 verification. NTFS file IDs and link counts distinguished separate copies from existing hard links. Found 813 content groups with multiple distinct NTFS files, representing 4.025 GiB of duplicate payload across the two trees. Another 10,520 groups consisted of names already sharing storage. The proposed file-level list was checked for tracked files, scope under `results/`, reparse paths, live file lengths and target overlap. Estimates below use reclaimable payload bytes and exclude filesystem allocation overhead.

| Proposed deletion | Exact scope and retention basis | Paths | Est. bytes freed |
| --- | --- | ---: | ---: |
| Temporary Room preparation copies | JPG and BIN files under `images/`, `source/` and `sparse/` in `results/final_report_prepared_input_20261119/` and `results/final_report_review_20261120/smoke_extract/data/room_factor2/`. All 1,260 paths matched the corresponding files under `data/room_factor2/` by SHA-256. The staged JSON records stay. | 1,260 | 783,584,934 |
| Unpacked smoke supplement | The 866 archive-member files under `results/final_report_review_20261120/smoke_extract/` match `supplement_smoke.zip` by SHA-256. Keep the ZIP and files created by the rehearsal after extraction, apart from the explicitly listed staged Room copies. | 866 | 33,812,128 |
| Trainer smoke outputs | PTs under `ckpts/`, PLYs under `ply/` and PNGs under `renders/` in `results/research_smoke_20261027/`, `results/a05_trainer_smoke_20261107/` and `results/a05_absgrad_smoke_20261107/`. These are test-run outputs. Keep configuration, metrics, logs, protocols and source snapshots. Viewing the original smoke models or renders would require rerunning the smoke test. | 243 | 851,887,161 |
| Room CLI rehearsal copies | In `results/final_report_review_20261120/cli_replay/`, the `candidate/ckpts/final_post_update.pt` matches the controlled Room A05-Abs seed 42 checkpoint. `raw/raw_mesh.obj` and `raw/raw_mesh.ply` match the released Room replay exports. Keep the CLI extraction record and comparison image. | 3 | 120,795,844 |
| **Deletion total** | **Includes existing hard-link accounting.** | **2,372** | **1,790,080,067 (1.667 GiB)** |

A further 561 separate copies totaling 2,796,509,027 bytes (2.604 GiB) are candidates for NTFS hard-link deduplication after the proposed deletions. All have matching SHA-256 content and another surviving path within `results/`. The largest groups include the G1/G2 Gaussian-depth replay (465,871,783 bytes), superseded measured-depth outputs (958,536,536 bytes across three directories), and identical RealSense checkpoint or cleanup exports. Keeping every filename preserves current manifests and script paths. Hard-linked names share any future in-place writes, so this option depends on treating the paired artifacts as frozen. The combined estimate is 4.272 GiB. The plan excludes hard-link changes to `data/`.

The October campaign and controlled report runs also contain 5.218 GiB of step-zero validation PNGs across four study directories. They are additional space only if the original visual evidence can be retired. Their aggregate size overlaps the duplicate audit, so it is excluded from both proposed totals. Preserve the pre-update and post-update checkpoints as distinct training states. The audit also found that the extracted final release differs from the current supplement in 16 files, so that extracted release is outside the deletion list.

The local audit lists are in `C:\Users\hi\AppData\Local\Temp\gsplat-storage-audit-20261121.json` and `C:\Users\hi\AppData\Local\Temp\gsplat-storage-cleanup-proposal-20261121.json`. The proposal SHA-256 is `da44bf4890c20ef39e50f9b975629d308f49d619f4bdd50b65ef8d6f7366c2c3`. This entry records the pre-deletion report. No training outputs were deleted, moved or relinked. Recheck file identities, hashes and active jobs against the approved scope immediately before any cleanup.

## Old run retirement proposal (2026-11-21)

The user clarified that the cleanup should target older artifacts that no longer serve the project, including unique files. This proposal replaces the duplicate-first deletion proposal. All targets are Git-ignored files beneath `results/`. Estimates count reclaimable bytes by NTFS file identity and link count. The recommendations assume the final report, current Room/Week 11-12 workflows, ongoing DB3 work and written records of past experiments are the retention priorities.

| Recommended retirement | Exact scope and effect | Paths | Estimated bytes freed |
| --- | --- | ---: | ---: |
| October A04 checkpoints saved before the final update | The 42 `ckpts/ckpt_6999_rank0.pt` files in runs listed by `results/research_completion_20261027/all_runs.json`. Verified each paired `final_post_update.pt` against its recorded SHA-256. Future runs can load the saved endpoints. Reloading the exact pre-final-update states would require reproducing them. | 42 | 7,056,380,146 |
| October A04 first-update validation images | `renders/val_step0_*.png` in those same 42 completed runs. Keep first-update metrics, camera orders, final validation images, Gaussian PLYs, crop sheets and all campaign records. The A04 verifier, Room replay and Week 12 input audit use final images or the JSON metrics. | 1,539 | 4,125,271,810 |
| Rejected school pairing and early smoke results | Saved models, Gaussian PLYs and validation PNGs within `school_completion_20261109/surface/` and `school_surface_smoke_20261109/replication/`. The recorded camera order diverged after index 271 in the rejected pairing. The accepted camera-matched experiment and meshes remain in `school_completion_replay_20261109/`. Keep the failed pairing record, logs, metrics, recipes and source snapshots. | 405 | 1,948,286,583 |
| Interrupted training renders | PNGs under `renders/` for the interrupted Bonsai A04 branch, both October 30 courtyard 120k interruptions, and the fragmented 15 FPS attempt's `train30k/`. Keep status files, logs, metrics, pose graphs and captured frames. The completed courtyard 120k checkpoint stays. | 269 | 310,901,505 |
| Prior trainer smoke outputs | PTs, PLYs and PNGs under the `ckpts/`, `ply/` and `renders/` subdirectories of `research_smoke_20261027/`, `a05_trainer_smoke_20261107/` and `a05_absgrad_smoke_20261107/`. Keep their validation summaries, configs, protocols and logs. | 243 | 851,887,161 |
| Rejected courtyard pose variants | The two checkpoints and PLY in each `results/courtyard_revisit_repair_20261102/{trial,trial_people}/train/`. Both variants were rejected as replacements for the original model. Keep the original and all matched comparisons, logs and previews. The two rejected variants' model viewers and live comparisons would require retraining. | 6 | 666,249,298 |
| **Recommended total** | | **2,504** | **14,958,976,503 (13.932 GiB)** |

An additional 5.523 GiB can be retired if the old experiments need only their recorded results. The nonselected RealSense small-pose `refined/` checkpoints and PLYs account for 1.483 GiB. The selected baseline remains viewable and the old `-Variant refined` path would require retraining. The original A05 residual-eligibility screen contributes 1.892 GiB in checkpoints, PLYs and validation PNGs within `a05_training_20261107/replication/`. This would retire direct replay of that screen. Its metrics, audit JSON and recipes stay. A05-Abs is a separate method and its models stay. The CCD-GS pilot's 36 `full_state.pt` replay checkpoints account for 2.147 GiB. Its 0/36 practical-gate outcome, scores and protocols stay. Replaying split/no-split continuations from the original saved states would require reproducing those states. The recommended and optional sets together free an estimated 19.454 GiB.

Keep the original RealSense recording, prepared input datasets, full Week 11-12 and final report artifacts, A04 final checkpoints and final validation images, A05-Abs, and G3-G5 checkpoint series. Earlier standalone A04 and RealSense checkpoints with documented viewer or comparison paths are outside the proposal. The approved cleanup should recheck the stored file identities, file lengths, timestamps, active jobs and final model digests before deleting.

The file-level proposal is `C:\Users\hi\AppData\Local\Temp\gsplat-old-run-cleanup-proposal-20261121.json` with SHA-256 `2deb7f9c929af410d4e7f279fb7718897a4d4d144638630aa966aa24dac79239`. Its 2,761 candidate paths were checked for Git tracking, scope, reparse points, and overlap. No files were deleted, moved or relinked.

## iPhone Stray Scanner exhibit reconstruction (2026-11-21)

Experiment A for the geometry-preserving Gaussian splatting project. The input is the untracked `710d9d02a3.zip` (269,190,096 bytes, SHA-256 `cd3d59c96c2c6cf0f820d96dca089cc991c43bc0856c2a0c12d51a8c8435785d`). The archive was retained unchanged. Its extracted files and derived outputs are under `results/stray_scanner_710d9d02a3/`. The scan shows a tiled exhibit area, a glass-topped case, display boards, thin wire mesh and a bright opening.

### Input and synchronization

- Found 3,756 depth PNGs, 3,756 confidence PNGs, 3,756 odometry rows with per-frame intrinsics, one HEVC RGB video, one final-frame camera matrix and an IMU CSV. No lens-distortion tables or tracking-state labels were present. IMU timestamps were audited. Exported poses were used without refitting.
- The default movie edit list discards the first RGB packet. FFprobe reported 3,756 packets and 3,755 presented images with the edit list enabled. Decoding with `-ignore_editlist 1` restores all 3,756 frames. Sorting underlying packet PTS into presentation order and fitting the 60 Hz video timebase to elapsed odometry time gave a slope of 0.999584994 and residuals between −0.0058 and +0.0039 ms. Frame indexing was retained only after checking that correspondence.
- Used the Stray Scanner [format specification](https://github.com/strayrobots/scanner/blob/main/docs/format.md) and [odometry encoder](https://github.com/strayrobots/scanner/blob/main/StrayScanner/Helpers/OdometryEncoder.swift). Stored quaternions are camera-to-world and already convert the ARKit camera axes to X-right, Y-down, Z-forward. The world frame remains ARKit Y-up. The pipeline scales depth millimetres to metres and scales each 1920×1440 RGB calibration to the 256×192 depth grid. Color was decoded in native landscape orientation and area-resampled to that grid.

### Reconstruction protocol

- Include confidence level 2 and depth from 0.25 to 5.00 m. Fit every sixth frame outside each reserved 1.1 s interval at phase [1.45, 2.55) s of a 4 s cycle. Score every ninth frame in the interval centers at phase [1.75, 2.25) s. The split, camera matrices and intrinsics are in `camera_split.npz`.
- Register and color samples from every second depth pixel in each image axis, then average within 10 mm world voxels. Compare 10 mm and 20 mm TSDF voxel sizes and 40 mm and 80 mm truncation distances using identical fit inputs. Keep triangle components of at least 100 faces and 0.0025 m². Leave observed boundaries open.
- Compare raycast Z-depth to high-confidence held-out LiDAR with signed error defined as mesh minus LiDAR. Report ray hit fraction and the fractions of all eligible reference pixels within 50 and 100 mm. Those fractions include misses in their denominators. Plane fits describe flatness on the reconstructed cloud and held-out scans use the same phone and pose stream, so neither supplies independent physical truth.

### Reproduction

From the repository root in PowerShell, using the existing `.venv` and FFmpeg on PATH. The first command assumes the output source directory has yet to be extracted.

```powershell
$out = 'results/stray_scanner_710d9d02a3'
Expand-Archive -LiteralPath 710d9d02a3.zip -DestinationPath "$out/source"
$scan = "$out/source/710d9d02a3"
ffprobe -v error -ignore_editlist 1 -select_streams v:0 -show_entries packet=pts_time -of csv=p=0 "$scan/rgb.mp4" | Sort-Object {[double]$_} | Set-Content -Encoding ascii "$out/video_pts_unedited.txt"
$env:OMP_NUM_THREADS='4'
.venv\Scripts\python.exe scripts/stray_scanner_reconstruct.py $scan $out --video-pts "$out/video_pts_unedited.txt"
```

### Results and geometry decision

The corrected run finished with Open3D 0.20.0, Python 3.12.10, NumPy 2.5.2, OpenCV 5.0.0, SciPy 1.18.1, trimesh 5.1.1 and FFmpeg 9.0.2. Initial runs diagnosed the MP4 edit list, an empty output from the Open3D legacy ScalableTSDFVolume backend in this environment, and JSON serialization of the plane coefficients. The successful run used the CPU tensor VoxelBlockGrid backend with 16-voxel blocks, a mesh extraction weight threshold of 1.5 and native serialization of the coefficients. Commands and output are in the reproduction block and `reconstruct.log`.

- Scan duration was 67.895 s. The estimated camera path was 19.83 m. Frame-to-frame translation had a 5.26 mm median and a 28.64 mm maximum. The largest rotation step was 5.07°. Quaternion norms deviated by at most 2.14e−7. The 6,781 IMU timestamps were strictly increasing and extended from +0.074 to +67.920 s relative to the first odometry timestamp. No IMU adjustment was applied.
- The fit used 453 frames. There were 53 scored frames in 17 reserved time blocks and 1,041 withheld frames in total. The minimum elapsed gap from a scored frame to any fitted frame was 0.317 s. The held-out check pooled 1,805,668 eligible high-confidence depth pixels. Views and poses come from the same session. The split has now been inspected and used to choose a mesh resolution.
- The 10 mm registered cloud has 573,710 colored points from 3,861,882 image samples. Of those voxels, 243,869 had one sample, including samples within the same frame, so voxel population alone is insufficient for anchor selection. The strongest fitted plane, identified visually as the floor, was inclined 0.24° from world horizontal. Its diagnostic residuals in a ±50 mm band had a 9.6 mm median and a 38.3 mm 95th percentile. Further near-parallel planes appeared around the floor and the wall. The residuals measure internal spread and the plane footprints do not establish physical dimensions.

| Metric | 10 mm TSDF | 20 mm TSDF |
| --- | ---: | ---: |
| Median signed error on hits | −1.39 mm | −3.31 mm |
| Median absolute error on hits | 6.31 mm | 9.14 mm |
| 95th percentile absolute error on hits | 29.02 mm | 55.14 mm |
| Eligible pixels with a ray hit | 99.84% | 99.79% |
| Eligible pixels within 50 mm | 98.01% | 93.39% |
| Eligible pixels within 100 mm | 99.14% | 98.88% |
| Weakest block within 50 mm | 84.78% near 10 s | 56.29% near 30 s |
| Vertices / faces | 564,363 / 1,092,631 | 120,832 / 235,409 |
| Retained components / boundary edges | 53 / 46,765 | 2 / 7,057 |

The 10 mm TSDF is the reference for this scan. Its main pre-filter component accounts for 96.63% of the total surface area. The mesh still has holes and disconnected parts, particularly around the wire grid, thin boundaries and surfaces with sparse coverage. The hit fractions refer only to eligible depth rays and cannot establish completeness in unobserved or low-confidence areas. Selected material boxes in four held-out frames were inspected in `material_regions.png` and evaluated in `material_regions.csv`. The 10 mm mesh was within 50 mm for 1,826/1,826 eligible floor pixels, 5,332/5,334 wood pixels and 3,251/3,251 glass-region pixels. The audited wire-grid/background box had **0/2,205 eligible high-confidence pixels**. Agreement on glass only shows that fusion reproduced the phone’s depth estimates in those boxes. It cannot distinguish the glass itself from visible material behind it or validate physical glass thickness.

Advance the well-observed opaque floor and cabinet panels as candidate anchor surfaces. Treat the glass, wire grid, silhouette boundaries and poorly observed surfaces as uncertain until independent geometry is available. The scope of this run was Experiment A and its geometry validation. Standard 3DGS, constrained Gaussian training, held-out RGB comparison and Gaussian counts remain pending. Signed depth is relative to LiDAR from the same device and independent physical surface error and completeness remain unmeasured.

### Artifacts and readback

- `registered_cloud.ply` is the metric colored point cloud. `tsdf_10mm.ply` and `tsdf_20mm.ply` are the matched meshes. `scene.glb` is the 10 mm model with vertex colors for Blender and glTF viewers. Readback confirmed the GLB and PLY have matching vertex positions, face indices and colors. The registered point cloud read back with its expected count and colors. The fit, withheld and scored frame arrays were checked for the required subset and disjointness relationships.
- `mesh_preview.png`, `geometry_preview.png` and `heldout_depth_preview.png` show the surface, registered cloud and held-out checks. `metrics.json` gives pose, plane, topology and depth metrics. Both `*_heldout.csv` files preserve per-frame counts.
- The ROI audit can be replayed after reconstruction with `.venv\Scripts\python.exe scripts/stray_scanner_material_audit.py $scan $out`.

| Output | SHA-256 |
| --- | --- |
| registered_cloud.ply | `5bd4950edf58619bff47a65060206bf9cb6a4c3d74eb76d4f3952ec788ef9733` |
| tsdf_10mm.ply | `8f8989772c9f1953b69497a8d07778a75d253a2512af4fa347ca902734bf9741` |
| tsdf_20mm.ply | `62bfb3c8a627860ca9e0c4d2281c4ed5320b66cf0291ad4bf61075f8143a7b38` |
| scene.glb | `8fccb4f60a5bb923c38c9f65956a4c954dc03e7b0f4db10f3a00057274aa405a` |
| metrics.json | `d40323e352df0a033583999187b05b177f4dee599fd50ed95dcd616da850455b` |
| camera_split.npz | `b372224a1d235a42f913cd8ee894af2de0933087f8c1445b589a1d59ace13fef` |

## Experiment B0/B1 — Stray Scanner exhibit Gaussian painter (2026-11-21)

Test H1 on the existing 10 mm `scene.glb`. B0 uses the standard gsplat 3D Gaussian renderer and freely optimized centres, rotations and 3D scales. Its fixed Gaussian count and mesh initialization are controls for this comparison. B1 fixes the same centres to sampled triangles, fixes each thin axis to the triangle normal, and learns two bounded tangent scales, opacity and spherical-harmonic color. Only the original scan's RGB supervised training. Both arms use the same 120,000 area-weighted samples, barycentric vertex colors, opacity initialization, camera sequence, seed 43 and 7,000 updates at 512×384. The SH degree increases from 0 to 3 at 1,000-update intervals. The RGB objective is 0.8 L1 and 0.2 DSSIM (implemented as 1 − SSIM). B1's tangent bounds are 0.5–2.5 times the initial tangent scale. There is no densification, pruning, A05-Abs allocation, depth loss, camera optimization or new mesh extraction in this pair. B2 and H2/H3 are outside this first comparison.

The script checks the GLB, registered cloud and split hashes against Experiment A and rederives all fit and held-out frame IDs from the original fusion rules. All 53 scored frames are in the 1,041-frame reserved set, and all 453 RGB training frames are the original fusion fit frames. The nearest temporal training frame is 0.317–0.617 s away. Forty-six scored views have a fit camera within 25 cm and 10° of full rotation, and 19 have a match more than four seconds away at those same pose thresholds. Pose proximity is a proxy for shared visibility. The existing holdout also informed mesh selection in Experiment A, so this is a same-scan held-out comparison with substantial view overlap and no independent validation of physical geometry.

RGB metrics are unweighted frame means over all 53 views. Depth uses gsplat's alpha-normalized expected projected Z at 256×192, with the same camera and pixel-centre convention as measured depth and GLB raycasting. Signed error is rendered minus reference. Eligible LiDAR pixels have confidence 2 and range 0.25–5.00 m. Report pooled error at alpha ≥0.90 on each arm's footprint and their intersection. The fraction within 50 mm over all eligible pixels includes missing and low-opacity renders in the denominator. Mesh-depth differences also use the previously audited floor and wood regions, with valid LiDAR and finite mesh hits. Report full-image alpha and whether the model covers those opaque regions. The same RGB rendering benchmark uses 53 views, five warm-up calls and three GPU-timed passes at 512×384. [gsplat rasterization](https://docs.gsplat.studio/main/apis/rasterization.html)

The protocol records exploratory appearance margins of at most 1 dB PSNR loss, 0.02 SSIM loss and 0.03 LPIPS increase for B1 versus B0. These are engineering decision margins for this run. Report the signed bias, absolute median and p95 in millimetres, coverage and mesh distances directly alongside those margins.

```powershell
$env:OMP_NUM_THREADS='4'
& scripts/run-school-completion.ps1 scripts/stray_painter.py results/stray_scanner_710d9d02a3 results/stray_painter_b01_replay
```

The default CUDA environment did not expose the toolkit. The existing school completion launcher loaded the local CUDA 12.8 toolchain and cached extension with PyTorch 2.9.1+cu128 on the GTX 1660 SUPER. The 12,000-Gaussian, five-update smoke run finished training, rendering and all metrics in both arms. Its transient NumPy read-only buffer warning was resolved by copying the decoded frames into owned arrays before training. The endpoint run has its own frozen protocol and logs. [Script](../scripts/stray_painter.py) · [Run directory](../results/stray_painter_b01_20261121/)

### Endpoint results

| Metric | B0 free geometry | B1 fixed surface |
| --- | ---: | ---: |
| PSNR | 20.385 dB | 17.438 dB |
| SSIM | 0.6532 | 0.5644 |
| AlexNet LPIPS | 0.2942 | 0.4519 |
| Mean full-image RGB alpha | 0.9412 | 0.9441 |
| Full-image RGB alpha ≥0.90 | 79.23% | 87.48% |
| LiDAR-pixel coverage at alpha ≥0.90 | 83.77% | 92.08% |
| LiDAR pixels within 50 mm of render, including misses | 71.06% | 86.63% |
| LiDAR signed median, own footprints | −15.22 mm | −13.46 mm |
| LiDAR absolute median / p95, own footprints | 18.97 / 102.02 mm | 15.39 / 53.97 mm |
| LiDAR signed median, shared footprint | −15.59 mm | −14.57 mm |
| LiDAR absolute median / p95, shared footprint | 19.24 / 102.88 mm | 16.63 / 56.61 mm |
| Opaque ROI mesh-depth coverage at alpha ≥0.90 | 86.33% | 89.65% |
| Opaque ROI render minus mesh signed median, shared footprint | −14.87 mm | −14.99 mm |
| Opaque ROI render vs mesh absolute median / p95, shared footprint | 15.77 / 49.79 mm | 15.18 / 23.45 mm |
| Nearest-mesh distance median / p95 | 2.824 / 14.071 mm | 0 / <0.001 mm |
| Centre displacement from initialization median / p95 / max | 7.04 / 20.08 / 121.79 mm | 0 / 0 / 0 mm |
| Gaussian count | 120,000 | 120,000 |
| GPU RGB render median / p95 | 0.985 / 1.321 ms | 1.574 / 2.948 ms |
| GPU RGB render throughput from mean frame time | 969 frames/s | 580 frames/s |
| Training time | 64.34 s | 57.45 s |

There were 1,805,668 eligible LiDAR pixels, with 1,457,232 scored on the intersection of the two render footprints. The floor and wood mesh comparison had 7,160 eligible pixels and 5,900 on the shared footprint. Direct GLB raycasting reproduced Experiment A's 1,802,823 hits and −1.39 mm signed median, 6.31 mm absolute median and 29.02 mm p95 against the phone depths. Both Gaussian renders had a systematic nearward bias on the opaque ROIs. B1's fixed centres reduced the tail substantially, while the typical apparent surface remained about 15 mm nearer than the mesh. Centre anchoring alone leaves the composited depth dependent on scale, opacity and overlaps.

B1 learned recognizable cabinet and floor appearance while keeping all centres fixed. It missed all three exploratory appearance margins, trailing by 2.95 dB PSNR and 0.0888 SSIM, with LPIPS 0.1577 higher. In the four visual examples, the fixed painter blurs fine texture and wire detail more, and leaves a dark unpainted opening in frame 2907 where B0 paints a blurry background. The PSNR difference was −3.13 dB in the 46 near-pose views and −1.74 dB in the seven other views, with SSIM and LPIPS also favoring B0 in both groups. This one-seed result demonstrates the basic painter behavior under fixed anchoring and falls short of the requested near-baseline appearance. The geometry and coverage improvements justify testing the soft anchor separately. H2 and H3 remain untested.

Readback checked every recorded input hash, finite training losses and learned parameters, 53 rendered files and depth maps per arm, common depth denominators, the training sequence and disjoint fit/reserved IDs. The B1 centres and quaternions exactly match initialization in the checkpoint. Barycentric readback placed all initial centres on their designated triangles to within 0.00025 mm of storage rounding. All B1 tangent scales stayed within bounds and every normal scale stayed fixed. Timing covers the renderer on the local GPU. Training used one seed and the full training run was performed once. [Protocol](../results/stray_painter_b01_20261121/main/protocol.json) · [Run log](../results/stray_painter_b01_20261121/main.log) · [Metrics](../results/stray_painter_b01_20261121/main/metrics.json) · [Per-frame depth](../results/stray_painter_b01_20261121/main/depth_frames.csv) · [Readback](../results/stray_painter_b01_20261121/main/readback.json) · [Visual comparison](../results/stray_painter_b01_20261121/main/heldout_comparison.jpg) · [B0 checkpoint](../results/stray_painter_b01_20261121/main/B0.pt) · [B1 checkpoint](../results/stray_painter_b01_20261121/main/B1.pt)

### Public checkpoint viewer (2026-11-21)

Loaded the final B0 and B1 checkpoints directly on the GTX 1660 SUPER. The Viser viewer presents live RGB renders of both checkpoints from the same interactive camera. It provides four captured viewpoints and uses ARKit world-up for display. The local service binds only to `127.0.0.1:8095`. The Viser relay issued an address, but public HTTPS returned 404 and the WebSocket upgrade returned 502. Winget verified the Cloudflare 2026.11.1 installer hash. The portable executable was extracted into the workspace and started a Cloudflare Quick Tunnel to the same loopback service. The verified public address for this session is [Exhibit Gaussian comparison](https://dozen-auburn-edit-whilst.trycloudflare.com). It is temporary and depends on the workstation and tunnel process.

Local HTTP and public HTTPS returned 200. Separate WebSocket clients on the local and public endpoints received both labeled renders at 1440×635 for the case and wire-grid viewpoints. The public and local JPEG hashes matched at both poses, the two poses produced different images, and the wire-grid button returned the expected captured camera position. Verified that all four pose buttons were present. [Viewer source](../scripts/view_stray_painter.py) · [Public verification](../results/stray_painter_b01_20261121/viewer_public_verification.json) · [Case render](../results/stray_painter_b01_20261121/viewer_public_000099.jpg) · [Viewer log](../results/stray_painter_b01_20261121/viewer.stdout.log) · [Tunnel log](../results/stray_painter_b01_20261121/cloudflared.stderr.log)

```powershell
& scripts/run-school-completion.ps1 scripts/view_stray_painter.py results/stray_painter_b01_20261121/main results/stray_scanner_710d9d02a3/camera_split.npz --port 8095
# Run in a second terminal while the viewer is active.
& .tools/painter-tunnel/package/cloudflared/cloudflared.exe tunnel --url http://127.0.0.1:8095 --no-autoupdate --protocol http2 --edge-ip-version 4
```

## Experiment B0/B1 — iPhone Stray Scanner rescan (`newscan.zip`, 2026-11-21)

The controlled rescan compares the previous free Gaussian and fixed-surface painter policies on the new capture. The original archive and prior run were retained. The new outputs use `results/iphone_rescan_v2/`. The archive is 798,949,584 bytes with SHA-256 `ea5f874767dea37a950a8accaca69b3c6caff186f6149fb3feb38aaff63e35a6`. Extracted directory ID is `e1cfcfffe4`.

### Source audit and frozen split

- The original RGB is HEVC at 1920×1440. The export contains 13,601 RGB packets, 13,601 odometry rows, 13,601 depth PNGs, 13,601 confidence PNGs, camera matrix and IMU CSVs. No distortion tables were present. Per-frame odometry intrinsics were used. The media counter spans 226.667 s while elapsed odometry spans 305.282 s. Of 13,600 odometry intervals, 4,713 exceed 25 ms. The first affine RGB-to-odometry clock check failed by as much as 22.39 s. The default movie edit list presents 13,600 images. Ignoring it restores all 13,601 and RGB PTS agree with frame index divided by 60 within 0.00034 ms. The MP4 is decoded in presentation order and in native raster orientation. The public Stray Scanner encoder uses a frame counter for PTS, while the depth and odometry are emitted from the same capture callback. [Format](https://github.com/strayrobots/scanner/blob/main/docs/format.md) · [Dataset encoder](https://github.com/strayrobots/scanner/blob/main/StrayScanner/Helpers/DatasetEncoder.swift) · [Video encoder](https://github.com/strayrobots/scanner/blob/main/StrayScanner/Helpers/VideoEncoder.swift)
- Independently checked ordinal pairing using forward-backward RGB tracks and LiDAR-backed relative-pose projection over the three capture thirds. On 7,429 / 4,337 / 3,441 common tracks, zero offset gave 0.288 / 0.381 / 0.880 px median reprojection error at 640×480. Every tested offset of ±2, ±6 or ±12 frames gave larger residuals in each third. The nearest tested alternatives gave 2.175 / 2.874 / 10.348 px. The path length is 79.14 m with maximum frame steps of 36.88 mm and 3.96°. Depth overlays show the cabinet, board and floor at corresponding image positions, with low-confidence gaps near the wire grid and reflective boundaries. [Preflight](../results/iphone_rescan_v2/dataset_audit/preflight.json) · [Alignment](../results/iphone_rescan_v2/dataset_audit/rgb_depth_alignment/flow_pose_lag.json)
- The six native text crops were inspected before model training. Frames 118 and 2277 show the same readable English print at two distances and poses. Frames 10022 and 11702 show overlapping print from the second panel. Frame 8275 shows the bilingual directory. Frame 11582 is a tighter view with visible softness on some lines. All six contain distinguishable letters. The crops and opaque-region boxes were frozen from the recorded images before evaluating reconstructions. [Native crops](../results/iphone_rescan_v2/dataset_audit/rgb_quality/) · [Region coordinates](../results/iphone_rescan_v2/splits/evaluation_regions.json)
- Fixed 0.8 s reserved blocks at phases [1.6, 2.4) in four-second odometry cycles. This reserves 2,712 source frames (19.94%). Every sixth frame outside those blocks yielded 1,815 common fusion/training views. The view closest to the centre of each block yielded 76 scored views spanning 2–302 s. The minimum separation from an integrated frame is 0.400 s. Exact lists were saved before fusion. A transient FFmpeg expression parsing failure on the contact-sheet selector was corrected by grouping terms. [Frozen split](../results/iphone_rescan_v2/splits/frame_split.json)

### Reconstruction and native-resolution preflight

Fused the 10 mm TSDF with 40 mm truncation and the previous 250–5,000 mm confidence-2 filter. Kept the original ARKit poses and the previous component filter. The cloud has 1,654,956 points from 17,063,764 observations. The GLB has 1,341,244 vertices and 2,567,910 triangles over 92.45 m². There are 215 retained components, 139,412 boundary edges and four nonmanifold edges. The dominant floor plane is inclined 0.53° to horizontal. Its 50 mm diagnostic band has 14.16 mm median and 40.56 mm p95 residual. These are internal cloud residuals. The orthographic previews recover the floor, cabinet and boards, while showing fragments and sparse boundary geometry around the doorway, glass and wire elements. The motion and depth checks use the same phone and ARKit poses as the reconstruction.

The 10 mm mesh raycast hit 99.81% of 2,874,584 eligible LiDAR pixels. Median signed mesh-minus-LiDAR error was −2.24 mm. Median / p95 absolute errors on hits were 7.32 / 42.97 mm, compared with 6.31 / 29.02 mm in the previous scan and split. The new mesh agrees within 50 mm on 95.92% of eligible pixels and within 100 mm on 98.97%. Two weak frames are 9782 at 178 s (65.73% within 50 mm) and 13502 at 302 s (69.22%). This is a larger scene and a different held-out distribution. No mesh resolution selection used the new score set. [Geometry metrics](../results/iphone_rescan_v2/geometry/metrics.json) · [Mesh](../results/iphone_rescan_v2/geometry/scene.glb) · [Depth checks](../results/iphone_rescan_v2/geometry/heldout_depth_preview.png)

Ran the eight-update, full 120,000-Gaussian smoke test in both arms at 1920×1440, including LPIPS, metric depth rendering and file output for three evaluation views. The PyTorch peak allocated memory was 500–529 MiB during the short training loop and about 919 MiB during evaluation. The full run uses the previous seed 43, fixed cameras, identical area-weighted surface initialization and frame order, 120,000 Gaussians and 7,000 updates in both arms. It uses the original RGB-only 0.8 L1 + 0.2 DSSIM objective with the same Adam settings and SH schedule. The training images come from an exact decoded RGB24 cache backed by a 15.68 GB memory-mapped file to fit the 6 GB GPU. The legacy experiment used 512×384. The new capture and delivered raster resolution both changed between experiments. [Smoke](../results/iphone_rescan_v2/smoke.log) · [Full run protocol](../results/iphone_rescan_v2/training/protocol.json)

The IMU log contains 30,490 rows and 18 non-increasing timestamp steps. No IMU refit was applied. All 39,996 pixels across the selected opaque interior boxes meet the confidence and range gate. In the two wire-grid boxes only 146/1,152 and 0/1,150 pixels meet that gate. The glass box has 299/460 eligible returns. Glass and wire regions are excluded from the opaque mesh-depth score. An eligible return on either material can represent foreground or background and still influence the original confidence-filtered TSDF. Of 76 scored views, 64 have a fit pose within 25 cm and 10° and 21 have such a fit pose more than four seconds away. [Pose overlap](../results/iphone_rescan_v2/training/split_audit.csv) · [Geometry hashes](../results/iphone_rescan_v2/geometry/sha256.json)

### Reproduction

Run from the repository root. Use an unused output directory and retain the same hand-marked `splits/evaluation_regions.json` (native RGB poster boxes and 256×192 depth material boxes) for this archive.

```powershell
$out = 'results/iphone_rescan_v2'
$scan = "$out/raw/e1cfcfffe4"
New-Item -ItemType Directory -Force "$out/dataset_audit", "$out/splits" | Out-Null
Get-FileHash newscan.zip | ConvertTo-Json | Set-Content "$out/dataset_audit/archive_sha256.json"
.venv\Scripts\python.exe -m zipfile -e newscan.zip "$out/raw"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,avg_frame_rate,nb_frames,duration -of json "$scan/rgb.mp4" | Set-Content "$out/dataset_audit/video.json"
ffprobe -v error -ignore_editlist 1 -select_streams v:0 -show_entries packet=pts_time -of csv=p=0 "$scan/rgb.mp4" | Sort-Object {[double]$_} | Set-Content -Encoding ascii "$out/dataset_audit/video_pts_unedited.csv"
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '2'
.venv\Scripts\python.exe scripts/iphone_rescan_audit.py $out $scan
.venv\Scripts\python.exe scripts/iphone_rescan_audit.py $out $scan --cache-native
.venv\Scripts\python.exe scripts/iphone_rescan_alignment.py $out $scan 118 838 8275 10022 11582 13142
.venv\Scripts\python.exe scripts/stray_scanner_reconstruct.py $scan "$out/geometry" --video-pts "$out/dataset_audit/video_pts_unedited.csv" --frame-split "$out/splits/frame_split.json" --rgb-frame-counter --only-10mm
& scripts/run-school-completion.ps1 scripts/iphone_rescan_painter.py $out $scan --smoke
& scripts/run-school-completion.ps1 scripts/iphone_rescan_painter.py $out $scan
& scripts/run-school-completion.ps1 scripts/iphone_rescan_compare.py $out $scan
& scripts/run-school-completion.ps1 scripts/iphone_rescan_readback.py $out
```

### Evaluation protocol

At updates 3,000, 5,000 and 7,000, score the 76 held-out RGB images at 1920×1440 and depth at 256×192. Report mean full-image PSNR, SSIM and AlexNet LPIPS by frame. Report PSNR, SSIM and signed gradient agreement on the six pre-marked native poster crops. Gradient agreement uses pixels in the upper quartile of captured luminance-gradient magnitudes and above the 0.02 magnitude floor. The signed dot product of rendered and captured gradients is normalized by the captured gradient energy. A ratio near zero indicates little source-aligned edge contrast at the recorded coordinates. The crop check does not require OCR.

Depth is alpha-normalized expected projected Z, in metres. Per-frame calibration was scaled from the calibrated depth grid to the RGB raster with the existing half-pixel convention. Score opaque coverage at alpha ≥0.90. Report LiDAR errors on each renderer's footprint and their intersection. The within-50 mm fraction includes pixels missing from the render. Compare render minus mesh on the manually identified opaque rectangles using the same eligible LiDAR and finite mesh masks. The fixed-anchor check reads back centres, orientations and scale limits from each checkpoint. Same-scan sensor agreement does not independently establish physical geometry.

The GLB and 10 mm PLY read back with identical vertices, triangle indices and RGB values. The registered cloud read back with 1,654,956 points and RGB. [Geometry readback](../results/iphone_rescan_v2/geometry/geometry_readback.json)



## 2026-11-21 — Viewer hosting stopped during training

Stopped the Stray Scanner Viser viewer on TCP port 8095, including its public sharing session, and the Cloudflare tunnel forwarding to it. Verified that port 8095 was no longer listening and found no other Python, Node, Cloudflare, or ngrok TCP listeners. The iPhone rescan process (PID 17672) was still running. Its log showed B0 reached 7,000 updates and entered held-out evaluation.

### Paired results

Both arms completed the planned native 1920×1440 run with 120,000 Gaussians, 7,000 updates and seed 43. They used the same initial positions, training sequence, camera poses and 76 scored frames. Training loops took 320.42 s for B0 and 308.96 s for B1. Scores at each update count use that checkpoint's own shared depth footprint. [Training log](../results/iphone_rescan_v2/training.log) · [All metrics](../results/iphone_rescan_v2/comparisons/summary.json)

| Updates | B0 PSNR / SSIM / LPIPS | B1 PSNR / SSIM / LPIPS | B0 / B1 shared LiDAR p95 |
| ---: | ---: | ---: | ---: |
| 3,000 | 17.42 / 0.7140 / 0.5911 | 16.25 / 0.6967 / 0.6289 | 118.94 / 86.32 mm |
| 5,000 | 18.09 / 0.7229 / 0.5689 | 16.75 / 0.7033 / 0.6188 | 167.89 / 81.06 mm |
| 7,000 | 18.35 / 0.7261 / 0.5549 | 16.94 / 0.7055 / 0.6129 | 165.28 / 79.27 mm |

| Metric at 7,000 updates | B0 free geometry | B1 fixed surface |
| --- | ---: | ---: |
| Poster PSNR / SSIM, mean over six crops | 14.026 / 0.49262 | 13.894 / 0.49268 |
| Poster edge-gradient retention | 0.000831 | 0.000333 |
| LiDAR coverage at alpha ≥0.90, own footprint | 96.48% | 98.68% |
| LiDAR pixels within 50 mm, including misses | 77.86% | 87.29% |
| LiDAR absolute median / p95, own footprints | 19.18 / 167.52 mm | 18.02 / 78.33 mm |
| LiDAR absolute median / p95, shared footprint | 19.16 / 165.28 mm | 18.01 / 79.27 mm |
| LiDAR signed median, shared footprint | −12.09 mm | −13.37 mm |
| Opaque ROI coverage against mesh, own footprint | 90.86% | 98.15% |
| Opaque ROI signed median versus mesh, shared footprint | −8.27 mm | −10.57 mm |
| Opaque ROI absolute median / p95 versus mesh, shared footprint | 9.43 / 36.66 mm | 10.98 / 39.28 mm |
| Nearest mesh distance median / p95 | 3.20 / 12.35 mm | 0 / <0.001 mm |
| Centre displacement median / p95 / max | 7.78 / 18.92 / 74.82 mm | 0 / 0 / 0 mm |

The endpoint shared footprints contain 2,753,653 LiDAR pixels and 36,202 opaque mesh pixels. B1 improved pooled depth coverage and cut the LiDAR p95 error by 52.0% on shared pixels. On the selected opaque regions, B1 nevertheless rendered a median 10.57 mm toward the camera relative to the mesh and increased the shared-footprint opaque p95 from 36.66 mm to 39.28 mm. The exact anchor locations and rendered depth therefore give different readings of geometry preservation. B1 missed the historical appearance margins by trailing B0 by 1.41 dB, 0.0206 SSIM and 0.0580 LPIPS. The 10 mm mesh has a lower held-out discrepancy than either renderer. [Material confidence and mesh errors](../results/iphone_rescan_v2/comparisons/material_confidence.csv) · [Opaque example](../results/iphone_rescan_v2/comparisons/depth_comparison/7000/013142.jpg)

### Poster outcome and decision

The recorded RGB resolves body text in the held-out images and in nearby training views. Neither run recovers readable body text at 7,000 updates. B0 retains some broad line structure on the directory and B1 loses more of it. Both have near-zero mean source-aligned edge contrast on the six marked crops. In the same exhibit paragraph, the old B0 at 512×384 at least retains distinct line bands. The compared poses, capture extent, frame sample and training resolution differ across the captures, so the historical comparison is qualitative. [Poster crop](../results/iphone_rescan_v2/comparisons/poster_crops/7000/002277.png) · [Directory crop](../results/iphone_rescan_v2/comparisons/poster_crops/7000/008275.png) · [Legacy crop](../results/iphone_rescan_v2/comparisons/legacy_poster_crops/001161.png) · [Training RGB check](../results/iphone_rescan_v2/dataset_audit/rgb_quality/002277_nearest_fit.jpg)

B2's entry criterion was unmet because B0 body text remained unreadable. At fixed count, area-weighted initialization is spread over 92.45 m² compared with 38.70 m² before. The nominal initial spacing increased from 17.96 to 27.76 mm and the initial tangent scale increased from 11.67 to 18.04 mm. An initialization diagnostic counting centres in the six crop projections within 10 mm of the first mesh hit found only 54–203 per crop. Increasing the fit set from 453 to 1,815 views also lowered the mean training exposures per fit frame from 15.45 to 3.86 with the same update budget. These are plausible contributors and this run does not isolate their effects. The next controlled test would change poster-region allocation in free B0 first, using training-only region annotations and matched data, resolution, optimizer and updates. Repeat under fixed anchors if B0 recovers the body text. [Projected initialization support](../results/iphone_rescan_v2/dataset_audit/rgb_quality/poster_support.json) · [Per-crop metrics](../results/iphone_rescan_v2/comparisons/poster_frames.csv)

Readback verified 164 recorded input hashes and the original ZIP hash. It checked all six checkpoints, all 456 RGB renders and the corresponding depth and opacity arrays. Every B1 checkpoint retains the exact initial centres and orientations, fixed normal thickness and bounded tangent scales. The result comes from one seed and the same phone supplies the geometry references. [Readback](../results/iphone_rescan_v2/comparisons/readback.json) · [B0 checkpoint](../results/iphone_rescan_v2/B0_standard/checkpoints/B0_7000.pt) · [B1 checkpoint](../results/iphone_rescan_v2/B1_fixed_painter/checkpoints/B1_7000.pt)

The command session reported status 1 without a diagnostic after writing both endpoint metric files. The comparison and readback commands exited successfully. All requested outputs are present and passed the checks above.

### Live comparison viewer

Started the 7,000-update B0 and B1 checkpoints in the synchronized viewer on loopback port 8096 at 2026-11-21 13:43:51 +07:00. A temporary Cloudflare HTTPS tunnel serves this viewer. The viewer starts at captured frame 2277 and offers the museum directory (8275), cabinet and floor (13142), and digital exhibit board (4436). The public and local routes returned HTTP 200 and delivered matching 1440×810 WebSocket render hashes at the poster and cabinet poses. The cabinet button set the expected camera pose. The first probe accepted a lower-resolution frame while the interactive renderer refined its image. Waiting for settled resolution gave matching public and local images. The tunnel stays live while the viewer and tunnel processes run on this PC. [Link record](../results/iphone_rescan_v2/live/url.txt) · [Verification](../results/iphone_rescan_v2/live/verification.json) · [Launch command](../results/iphone_rescan_v2/live/launch-viewer.ps1) · [Viewer log](../results/iphone_rescan_v2/live/viewer.stdout.log) · [Tunnel log](../results/iphone_rescan_v2/live/tunnel.stderr.log)

## Experiment A05-Abs uncapped — iPhone LiDAR rescan (2026-11-21)

Stopped the previous B0/B1 live viewer and its temporary HTTPS tunnel at 2026-11-21 13:52:18 +07:00, and confirmed port 8096 was free before launching training. Planned one run using the user's final choice of 500,000 area-weighted initial Gaussians. Reused the frozen iPhone rescan mesh, RGB cache, cameras, seed 43, 1,815 fit frames, saved 7,000-update camera sequence and 76 held-out views at 1920×1440. The archived A05-Abs rule uses a 0.0008 absolute-gradient threshold and Long-Axis Split with the existing opacity reset and pruning. The cap and its scheduled count targets are disabled. All eligible parents can split, so attribution and ranking are skipped when residual fallback is off. The initial density and allocation both change relative to the archived 120,000-Gaussian B0 arm. [Run directory](../results/iphone_rescan_v2/A05_absgrad_uncapped/) · [Training protocol](../results/iphone_rescan_v2/A05_absgrad_uncapped/protocol.json)

### Training and readback

The single training process launched at 13:52:18 +07:00 and exited successfully after 7,000 updates. On the 92.45 m² mesh, the 500,000 initial samples give nominal spacing of 13.60 mm, tangent scale of 8.84 mm and normal scale of 1.33 mm. The run retained the rescan's RGB-only 0.8 L1 + 0.2 DSSIM objective, fused Adam rates, SH schedule, and metric near/far planes of 0.05/20 m. It used the archived A05-Abs packed rasterization path and a fitted-camera scene radius of 3.118 m for scale-based pruning. Absolute gradients were tracked through refinement, with one opacity reset at zero-based step 3,000.

The 54 refinement events split all 624,583 eligible parents and pruned 122,500 Gaussians. Checkpoints contained 856,598 at update 3,000, 934,058 at 5,000 and **1,002,083** at 7,000. The peak before an event's pruning was 1,002,836. The first event after the opacity reset pruned 101,440 Gaussians. Training took 543.07 s with 1,401 MiB peak allocated tensors and 2,334 MiB peak reserved by PyTorch on the GTX 1660 SUPER (Torch 2.9.1+cu128, driver 621.24). The readback verified the recorded input and training-source hashes, all three checkpoint hashes and tensor shapes, finite losses, every event's count arithmetic, and the 76 endpoint RGB PNGs and depth/opacity arrays. The budget and strategy tests passed (22 tests with CUDA checks before launch, and 20 tests including the new uncapped growth case with three CUDA tests deselected during training). [Launch command](../results/iphone_rescan_v2/A05_absgrad_uncapped/launch-training.ps1) · [Training stdout](../results/iphone_rescan_v2/A05_absgrad_uncapped/training.stdout.log) · [Growth record](../results/iphone_rescan_v2/A05_absgrad_uncapped/growth.json) · [Final checkpoint](../results/iphone_rescan_v2/A05_absgrad_uncapped/checkpoints/A05_absgrad_uncapped_7000.pt)

### Endpoint comparison against archived B0

| Metric | B0, fixed 120,000 | A05-Abs, uncapped |
|---|---:|---:|
| Mean PSNR (dB) | 18.353 | 18.773 |
| Mean SSIM | 0.7261 | 0.7337 |
| Mean LPIPS Alex | 0.5549 | 0.4019 |
| Median native RGB render (ms) | 4.04 | 9.35 |
| LiDAR coverage at alpha ≥ 0.90 | 96.48% | 98.89% |
| LiDAR median / p95 absolute error on shared pixels (mm) | 19.22 / 167.77 | 15.80 / 250.08 |
| Opaque mesh median / p95 absolute error on shared pixels (mm) | 9.42 / 36.70 | 7.44 / 27.94 |
| Gaussian centre distance to the mesh, median / p95 (mm) | 3.20 / 12.35 | 5.91 / 167.10 |
| Mean source-aligned edge retention across six poster/directory crops | 0.0008 | 0.0602 |

Appearance improved on 56/76 views by PSNR, 46/76 by SSIM and 73/76 by LPIPS. The poster close-up recovered recognizable words and line shapes, with some body text still blurred. The directory recovered recognizable line structure and some large letters. The shared depth comparisons use 2,760,067 LiDAR pixels and 36,161 opaque mesh pixels. The denser model improved median depth agreement while increasing the LiDAR error tail. This single seed combines a denser starting sample with a different refinement policy and uses reference depth from the same phone capture. [Summary](../results/iphone_rescan_v2/A05_absgrad_uncapped/summary.json) · [Per-view RGB](../results/iphone_rescan_v2/A05_absgrad_uncapped/renders/7000/A05_absgrad_uncapped_frames.csv) · [Per-view depth](../results/iphone_rescan_v2/A05_absgrad_uncapped/renders/7000/depth_frames.csv) · [Poster close-up](../results/iphone_rescan_v2/A05_absgrad_uncapped/renders/7000/poster_crops/002277.png) · [Directory](../results/iphone_rescan_v2/A05_absgrad_uncapped/renders/7000/poster_crops/008275.png)

### Live viewer

Opened the new comparison in the Codex browser by 2026-11-21 14:04:27 +07:00 with A05-Abs first and archived B0 alongside it. Verified both loopback port 8096 and the replacement temporary HTTPS tunnel at 2026-11-21 14:05:09 +07:00. Both routes returned HTTP 200, exposed the four captured viewpoints, moved the camera to the requested cabinet pose, and returned identical 1440×810 render hashes at the poster and cabinet poses. The new link depends on the viewer and tunnel running on this PC. [Temporary URL record](../results/iphone_rescan_v2/A05_absgrad_uncapped/live/url.txt) · [Verification](../results/iphone_rescan_v2/A05_absgrad_uncapped/live/verification.json) · [Launch command](../results/iphone_rescan_v2/A05_absgrad_uncapped/live/launch-viewer.ps1)

## 2026-11-21 — iPhone rescan with 1 million initial Gaussians and 20,000 updates

Launched a fresh single-seed run at 17:57:59 ICT to test whether more Gaussians and a longer training schedule improve exhibit text readability. The baseline is the 7,000-update uncapped A05-Abs model with 500,000 initial Gaussians and 1,002,083 at the endpoint. This trial starts with 1,000,000 mesh samples and runs 20,000 updates. The initial count was chosen for this trial. The run uses the same native 1920×1440 inputs, frozen 1,815 fit and 76 held-out views, seed 43, and RGB loss. It extends the original camera permutation stream and uses the same uncapped Long-Axis Split strategy with refinement ending at 6,000 updates. The means learning-rate decay runs over the full training horizon. Checkpoints are scheduled at 3k, 5k, 7k, 10k, 15k, and 20k. [Launch script](../results/iphone_rescan_v2/A05_absgrad_uncapped_20k/launch-training.ps1)

The endpoint comparison will use matched native renders and the same six frozen poster and directory crops, with visual letter inspection alongside crop PSNR, SSIM, and source-aligned edge retention. The full held-out RGB and shared-footprint depth comparisons will check whether crops improve alongside the wider scene. The initial count and training schedule change together in this trial. The existing views have already been inspected.

Stopped the initial attempt after it reported 2,000 updates. A preflight check found that a shallow copy of the NumPy generator shared its state, so extending the camera sequence changed the starting random state for mesh sampling. The preserved attempt is linked above. Changed the copy to an independent generator and relaunched at 18:01:42 ICT. This preserves the original initialization RNG state while extending the camera sequence. The shared painter module's hash also changed relative to the earlier snapshot. The source diff is confined to conditional anchor-displacement reporting in evaluation and leaves sampling and rendering code identical. [Corrected launch](../results/iphone_rescan_v2/A05_absgrad_20k/launch-training.ps1)

For an additional readability check, froze a manual 100-word transcription of the English poster in frame 2277 before the corrected run produced any renders. Tesseract English OCR with 2× bicubic upsampling and page segmentation mode 6 made zero word errors on the recorded crop and 91 word edits per 100 reference words on the 7,000-update render. The same crop and OCR settings will be applied to the endpoint. This measures one English crop. The bilingual directory and other posters will be inspected visually and compared on the existing image metrics. [OCR script](../results/iphone_rescan_v2/A05_absgrad_20k/score_poster_ocr.py) · [Baseline OCR](../results/iphone_rescan_v2/A05_absgrad_20k/poster_002277_ocr_baseline.json)

### Training, readback, and poster results

The corrected run completed all 20,000 updates at 18:32 ICT in 1,841.91 s (30 min 42 s). The starting surface spacing was 9.62 mm compared with 13.60 mm before. Readback verified that the first 7,000 camera IDs matched the original sequence, that the entire 20,000-update sequence used fit views, and that the first 500,000 sampled face IDs matched the old initialization. There were 54 refinement events, 545,566 parent splits, and 197,757 prunes. The peak count before an event's pruning was 1,348,483. The six checkpoints held 1,296,756 at 3k, 1,285,692 at 5k, and 1,347,809 at each saved update from 7k onward. The final count is 34.50% above the baseline. Peak training memory reported by PyTorch was 1,851 MiB allocated and 3,046 MiB reserved on the GTX 1660 SUPER. Training and evaluation exited with code 0. Readback checked the protocol's input and training-source hashes, six checkpoint hashes and tensor contents, finite losses, growth arithmetic, 76 native RGB renders and the saved depth and opacity arrays. [Protocol](../results/iphone_rescan_v2/A05_absgrad_20k/protocol.json) · [Training log](../results/iphone_rescan_v2/A05_absgrad_20k/training.stdout.log) · [Evaluation command](../results/iphone_rescan_v2/A05_absgrad_20k/launch-evaluation.ps1) · [Final checkpoint](../results/iphone_rescan_v2/A05_absgrad_20k/checkpoints/A05_absgrad_uncapped_20000.pt)

| Metric on matched views | Archived A05-Abs at 7k | Denser A05-Abs at 20k |
| --- | ---: | ---: |
| Final Gaussians | 1,002,083 | 1,347,809 |
| Mean PSNR on 76 held-out views (dB) | 18.773 | 19.737 |
| Mean SSIM on 76 held-out views | 0.73373 | 0.74495 |
| Mean LPIPS Alex on 76 held-out views | 0.40185 | 0.34600 |
| Median native RGB render time (ms) | 9.35 | 10.23 |
| Mean crop PSNR on six crops (dB) | 15.809 | 16.139 |
| Mean crop SSIM on six crops | 0.54368 | 0.55049 |
| Mean source-aligned edge retention on six crops | 0.06019 | 0.08602 |
| OCR word error rate on the English poster at frame 2277 | 91% | 81% |
| LiDAR depth coverage at alpha ≥ 0.90, own footprint | 98.89% | 99.81% |
| LiDAR median / p95 absolute depth discrepancy on shared pixels (mm) | 15.80 / 251.55 | 20.91 / 222.55 |
| Opaque mesh median / p95 absolute depth discrepancy on shared pixels (mm) | 7.51 / 28.50 | 9.18 / 38.32 |

PSNR improved on 70/76 views, SSIM on 58/76, and LPIPS on 74/76. Source-aligned edge retention improved on five of the six crops, including frame 2277 (0.13283 → 0.19429) and the directory (0.08272 → 0.15940). The opening English paragraph has clearer words and the largest directory headings have more recognizable letters. The smaller text and second English panel remain difficult to read, and slanted streaks are visible in several crops. The 81% OCR word error rate measures one held-out English crop against the frozen transcript. The source image scored 0%. [English poster](../results/iphone_rescan_v2/A05_absgrad_20k/renders/20000/poster_crops/002277.png) · [Directory](../results/iphone_rescan_v2/A05_absgrad_20k/renders/20000/poster_crops/008275.png) · [Second English panel](../results/iphone_rescan_v2/A05_absgrad_20k/renders/20000/poster_crops/010022.png) · [OCR output](../results/iphone_rescan_v2/A05_absgrad_20k/poster_002277_ocr.json)

Within the denser run, mean crop edge retention rose from 0.04974 at 7k to 0.06576 at 10k, 0.08343 at 15k, and 0.08602 at 20k. Mean crop PSNR and SSIM peaked among these checkpoints at 15k (16.226 dB, 0.55460). At the matched 7k milestone, the denser run had lower mean crop edge retention than the archived model. The new means learning rate was already on its longer schedule at that milestone. The trial changes both density and schedule and uses one seed on previously inspected development views. [Crop trajectory](../results/iphone_rescan_v2/A05_absgrad_20k/renders/20000/poster_trajectory.csv)

The depth comparison used 2,839,931 shared LiDAR pixels and 39,580 shared opaque mesh pixels. LiDAR coverage increased and the shared LiDAR p95 discrepancy fell, while the LiDAR median and opaque-region discrepancies increased. The fraction of eligible LiDAR pixels within 50 mm on each model's own footprint fell from 82.32% to 75.58%. The 20k result is sharper for appearance, with only partial recovery of legible lettering and mixed depth agreement against same-capture references. Both models remain available. [Evaluation summary](../results/iphone_rescan_v2/A05_absgrad_20k/summary.json) · [Per-view RGB](../results/iphone_rescan_v2/A05_absgrad_20k/renders/20000/A05_absgrad_uncapped_frames.csv) · [Per-view depth](../results/iphone_rescan_v2/A05_absgrad_20k/renders/20000/depth_frames.csv)

### Local interactive preview

At 21:38:44 ICT, opened and verified the 20,000-step checkpoint alongside the 7,000-step A05-Abs baseline in the local comparison viewer. The checkpoint hash matched the training record. The viewer listens on loopback at http://127.0.0.1:8096 and starts at the poster close-up (frame 2277). The browser displayed both checkpoint renders. The local WebSocket check received 1440×810 frames at the poster and cabinet poses and verified the cabinet preset moved the camera to the saved capture pose. [Launch command](../results/iphone_rescan_v2/A05_absgrad_20k/live/launch-viewer.ps1) · [Verification](../results/iphone_rescan_v2/A05_absgrad_20k/live/verification.json)

## 2026-11-21 — Matched 20,000-update iPhone rescan comparison of B0 and A05-Abs

Reused the completed A05-Abs checkpoint and launched B0 at 21:42 ICT. Interpreted “B01 3DGS pipeline” as B0, the archived free-geometry standard arm of the B0/B1 painter experiment. B0 uses its original fixed-count training loop. Both arms use the exact same one million initial mesh samples, camera order, 1,815 fit views, 76 held-out views, native 1920×1440 RGB, seed 43, fixed camera poses, RGB loss, Adam learning rates, and SH schedule. B0 uses unpacked training rasterization and A05-Abs uses packed rasterization. A05-Abs adds absolute-gradient Long-Axis Split, pruning and an opacity reset. This tests the full refinement policy against B0 at equal initialization and updates. The final Gaussian counts can differ. The launcher checked the input hashes and read the initialization and camera sequence directly from A05-Abs. [Launch command](../results/iphone_rescan_v2/B0_matched_20k/launch-training.ps1) · [Training protocol](../results/iphone_rescan_v2/B0_matched_20k/protocol.json) · [Training log](../results/iphone_rescan_v2/B0_matched_20k/training.stdout.log)

The endpoint comparison will evaluate both checkpoints with the same renderer and metric code on the 76 frozen views. It will include native RGB scores, six frozen text crops and the same English OCR transcript, pooled sensor and opaque-mesh depth discrepancies over shared pixels, count, time and memory. The score views and depth references come from the same capture. This run uses one seed.

### Matched endpoint results

B0 completed 20,000 updates at 22:11 ICT in 1,742.44 s. Training and evaluation exited successfully. Readback verified the hashes and tensor contents of all six checkpoints per arm, all 20,000 finite losses per arm, the common protocol and 152 held-out depth and confidence files. Before the first A05 refinement, the 600 paired losses differed by 0.00000084 on average and at most 0.000014. Both endpoints were re-rendered with the same evaluation code and unpacked rasterization. A05's RGB metrics reproduced the archived values. The evaluation wrote 76 verified native RGB images per arm and finite depth and opacity arrays at 256×192. [B0 checkpoint](../results/iphone_rescan_v2/B0_matched_20k/checkpoints/B0_20000.pt) · [A05 checkpoint](../results/iphone_rescan_v2/A05_absgrad_20k/checkpoints/A05_absgrad_uncapped_20000.pt) · [Evaluation command](../results/iphone_rescan_v2/B0_matched_20k/launch-evaluation.ps1) · [Summary and checkpoint hashes](../results/iphone_rescan_v2/B0_matched_20k/summary.json)

| Metric at 20,000 updates | B0 standard | A05-Abs uncapped |
| --- | ---: | ---: |
| Mean PSNR (dB), 76 held-out views | 19.677 | 19.737 |
| Mean SSIM | 0.74167 | 0.74495 |
| Mean AlexNet LPIPS | 0.39585 | 0.34600 |
| Mean PSNR (dB) / SSIM, six text crops | 15.906 / 0.53545 | 16.139 / 0.55049 |
| Mean source-aligned edge retention, six text crops | 0.04465 | 0.08602 |
| OCR word error, English close-up at frame 2277 | 99% | 81% |
| LiDAR coverage with alpha ≥ 0.90, own footprint | 99.64% | 99.81% |
| LiDAR pixels within 50 mm, misses included | 81.69% | 75.58% |
| LiDAR median / p95 absolute depth discrepancy, shared pixels | 17.33 / 146.21 mm | 20.92 / 220.14 mm |
| Opaque mesh median / p95 absolute depth discrepancy, shared pixels | 7.58 / 26.32 mm | 9.15 / 38.26 mm |
| Gaussian centre distance to the mesh, median / p95 | 2.75 / 13.09 mm | 5.19 / 130.50 mm |
| Final Gaussian count | 1,000,000 | 1,347,809 |
| Median native RGB render time | 8.86 ms | 11.43 ms |
| Training time | 29 min 02 s | 30 min 42 s |
| Peak PyTorch allocation / reservation | 1,506 / 1,864 MiB | 1,851 / 3,046 MiB |

A05 lowered mean LPIPS by 12.6% and improved it on 72 of 76 views. PSNR rose by 0.060 dB and improved on 44 views. SSIM improved on 42 views. Average source-aligned text edge retention nearly doubled and improved in five of the six crops. In frame 2277, B0 blurs most letters into horizontal bands. A05 recovers much of the first paragraph's letter structure, and the lower paragraph remains visibly distorted. The recorded crop had 0% OCR word error. Frame 10022 lost text edge agreement with A05. A05's largest PSNR loss was 2.05 dB at frame 12422, which shows the back of the display and the aisle by the wire grid. [Poster close-up](../results/iphone_rescan_v2/B0_matched_20k/comparison/poster_crops/002277.png) · [Directory](../results/iphone_rescan_v2/B0_matched_20k/comparison/poster_crops/008275.png) · [Second poster](../results/iphone_rescan_v2/B0_matched_20k/comparison/poster_crops/010022.png) · [Frame 12422](../results/iphone_rescan_v2/B0_matched_20k/comparison/scene_012422.png) · [Paired RGB scores](../results/iphone_rescan_v2/B0_matched_20k/comparison/rgb_deltas.csv)

The shared depth comparison covers 2,861,635 eligible LiDAR pixels and 39,857 opaque mesh pixels. A05 increased the LiDAR p95 discrepancy by 50.6% despite slightly greater coverage. Its signed LiDAR median was 17.65 mm toward the camera versus 13.36 mm for B0. The Gaussian centre distance p95 was nearly ten times larger with A05. This is evidence of poorer agreement with the same-capture geometry references. This single-seed comparison tests the whole uncapped A05 policy, including its change in Gaussian count. For this capture A05 provides clearer rendered detail, while B0 gives closer depth agreement, fewer Gaussians, and faster rendering. [Per-frame depth scores](../results/iphone_rescan_v2/B0_matched_20k/comparison/depth_frames.csv)

Started the local matched viewer at [port 8097](http://127.0.0.1:8097). An HTTP and WebSocket readback confirmed both endpoint models loaded and produced distinct 1440×810 side-by-side renders for the poster and cabinet poses. The cabinet preset moved the shared camera to its recorded position. [Viewer launcher](../results/iphone_rescan_v2/B0_matched_20k/launch-viewer.ps1) · [Viewer verification](../results/iphone_rescan_v2/B0_matched_20k/viewer_verification.json)

## 2026-11-21 — Stray Scanner from-scratch capture (`mo.zip`) intake

Treated `mo.zip` as the new capture because it is the most recent scan archive in the workspace. It contains Stray Scanner capture `fadddc42db`, 1,019,565,942 bytes, with SHA-256 `2be5efeb8abbed4fc3dbe6c74d102afe07856ea5f9eb65975b411a69420a1871`. Verified contiguous frame IDs 0–22,371 across odometry, depth and confidence, plus 22,372 RGB packets in the 1920×1440 HEVC stream. Sampled depth and confidence PNG headers at the first, middle and last frames. All were 256×192 with 16-bit depth and 8-bit confidence. The archive also has the per-frame intrinsics in odometry, a camera matrix and an IMU CSV. [Stray Scanner format](https://github.com/strayrobots/scanner/blob/main/docs/format.md)

Odometry spans 617.815 s with strictly increasing timestamps. The RGB presentation timestamps span 372.967 s with `-ignore_editlist 1`. The default probe exposes one discard-marked RGB packet. The probe with `-ignore_editlist 1` exposes no discard-marked packets. Seven 60 Hz media ticks are absent, following zero-based RGB presentation ordinals 7239, 7316, 7528, 7535, 7546, 7561 and 7630. Those ordinals fall near 121–127 s on the odometry clock. The 116.667 ms maximum deviation from `ordinal / 60` exceeds the 0.01 ms tolerance in `scripts/iphone_rescan_audit.py` and the reconstruction script's frame-counter mode. The affine RGB-to-odometry fit also misses by up to 34.29 s. Equal counts establish a candidate ordinal pairing. The seven missing ticks alone leave the cause and the image-to-pose alignment unresolved. The largest adjacent camera translation is 0.346 m at odometry frame 16791, near 431.81 s. [Stray Scanner video encoder](https://github.com/strayrobots/scanner/blob/main/StrayScanner/Helpers/VideoEncoder.swift)

Recommendation for the geometry-preserving 3DGS pipeline. First decode in presentation order with `-ignore_editlist 1`, `-noautorotate` and passthrough frame timing. Verify the decoded frame count and test image/depth/pose alignment before, through and after the timestamp gaps. Inspect the large pose step and verify tracking on the capture as a whole. Use the exported ARKit poses and per-frame intrinsics as the initial calibration with depth converted from millimetres to metres. Pose refinement is conditional on the validation results. Review sharpness and coverage, freeze disjoint training and held-out blocks, and select native-resolution RGB keyframes. Fuse a confidence-filtered TSDF using fit views for mesh initialization, then check held-out sensor-depth agreement before training. A 10 mm voxel and confidence level 2 are the previous run's starting settings. The scope of this check was the archive and video metadata. Decoded alignment, fusion and training remain pending.

## 2026-11-21 — `mo.zip` from-scratch quality run, preprocessing and A05-Abs

The quality run uses `results/mo_quality_20261121/` and retains the original `mo.zip`. Decoded 22,372 RGB frames in native orientation, with `-ignore_editlist 1` and passthrough timing. Forward-backward image tracks with LiDAR depth and relative ARKit poses compared offsets −9 through +9. Zero offset gave the lowest median error in all eleven sampled bands. Before, around and after the seven missing RGB timestamp ticks, it gave 0.511, 0.347 and 0.747 px median error at 512×384 over 6,038, 14,634 and 2,430 common tracks. The eight route sections ranged from 0.360 to 1.897 px. The video timestamp discontinuities did not require inserting or dropping image frames. [Alignment audit](../results/mo_quality_20261121/dataset_audit/alignment.json)

Inspected trajectory extremes and source contact sheets. Most large translations had too few reliable tracks or elevated reprojection error around the event, including zero reliable tracks across the 0.346 m step at frame 16791. Excluded ±6 frame indices around eleven translation steps of at least 0.10 m and retained surrounding coverage. The capture includes printed signs, wood grain, carved surfaces, fabric, jewelry and glass cases. Reflections and dark surfaces often provide weak LiDAR or image-track evidence. Used the original pose trajectory and native per-frame calibration outside the excluded windows. This alignment experiment evaluates local pose consistency. It does not establish global map placement or physical metrology. [Tracking exclusions](../results/mo_quality_20261121/splits/tracking_exclusions.json) · [Source route](../results/mo_quality_20261121/dataset_audit/route_01.jpg)

Selected 3,657 sharp training/fusion keyframes using local 0.152 s time bins. Reserved 103 score views inside disjoint temporal blocks and assigned 77 to development and 26 to the final score. Cached all 3,760 selected 1920×1440 RGB frames with independently seekable lossless Zstandard compression. The cache is 7.76 GB. Fused a confidence-2, 0.25–5 m TSDF at 8 mm with 32 mm truncation from training views. Removed isolated mesh components below 12 triangles or 3 cm². [Frozen split](../results/mo_quality_20261121/splits/frame_split.json) · [Evaluation protocol](../results/mo_quality_20261121/splits/evaluation_protocol.json)

Planned 48,000 native RGB updates with A05-Abs uncapped absolute-gradient Long-Axis Split at threshold 0.0008 and an extended refinement window through update 16,000, subject to measured CUDA growth headroom on the 6 GB GPU. Starts from dense mesh sampling with 72% assigned by area and 28% biased toward recorded color contrast. The objective retains 0.8 RGB L1 plus 0.2 DSSIM. The frozen rule compares 20k, 32k, 40k and 48k checkpoints on the development views and selects by LPIPS among checkpoints within three percentage points of the best fraction of LiDAR pixels within 50 mm. The selected checkpoint will be scored on the 26 final views. Depth agreement is measured against the same source capture. [Training launcher](../results/mo_quality_20261121/launch-training.ps1)

The monolithic 8 mm Open3D volume integrated all 3,657 keyframes into 309,220 active blocks. Mesh extraction ended with Windows exit `0xC0000005` before writing a mesh. Re-ran the same fusion with 3.072 m XZ tiles and two-block halos. Export keeps each triangle in the tile containing its centroid. This bounds extraction memory while retaining the 8 mm voxel size and the original fit views. A native cache sample at frame 227 matched independent FFmpeg RGB bytes exactly. Defined 13 final surface crops from the recorded images before training. These cover carved stone, wood grain and floor, signage, cloth, footwear, glass and case edges, jewelry, a weathered chest, window lattice and ceiling trim. [Monolithic attempt](../results/mo_quality_20261121/fusion_monolithic_attempt.log) · [Tiled fusion](../results/mo_quality_20261121/fusion.log) · [Source-defined crops](../results/mo_quality_20261121/splits/evaluation_protocol.json)

### Trajectory and fusion correction

The tiled 8 mm ARKit-pose mesh finished with 39,508,912 triangles over 873.06 m². It hit 99.78% of 3,511,662 eligible LiDAR score pixels, but only 43.45% fell within 50 mm. The median and p95 absolute differences were 74.76 and 721.94 mm. Stopped the initial A05 launch before any training updates to improve this initialization. The large near-side discrepancies motivated tests of cross-pass trajectory drift and free-space evidence. [Original fusion audit](../results/mo_quality_20261121/geometry_arkit/metrics.json)

Used 591 training-only feature keyframes for a sparse SE(3) graph. Of 6,527 visual revisit candidates, 486 passed metric two-way PnP, depth and pose agreement checks. Retained 485 graph constraints for the second solve. Corrections are interpolated to the original frame times. Median per-frame position correction was 129.82 mm, and median rotation correction was 2.24°. Median fitted loop translation residual decreased from 60.91 to 12.06 mm. The independent development-view comparison below uses reserved views outside both pose fitting and TSDF integration. It measures consistency with LiDAR from the same capture. [Pose correction](../results/mo_quality_20261121/pose_refinement/metrics.json)

On the same 1,219 fit and 77 development views at 24 mm, correcting the poses improved within-50-mm agreement from 57.47% to 71.53%, and lowered median and p95 from 39.12/502.73 mm to 21.03/399.14 mm. Added free-space updates to already allocated blocks, only where a local patch of high-confidence depth was farther than the block by the conservative clearance. This improved 24 mm agreement to 75.68% and reduced median/p95 to 19.50/222.34 mm, with 99.69% coverage. This test uses high-confidence depth as visibility evidence. Glass and reflective cases can still produce ambiguous returns. [ARKit comparison](../results/mo_quality_20261121/pose_refinement/development_arkit.json) · [Corrected poses](../results/mo_quality_20261121/pose_refinement/development_refined.json) · [Free-space comparison](../results/mo_quality_20261121/pose_refinement/development_refined_carved.json)

With the same refined trajectory and conservative free-space test, the 24, 16, 12 and 10 mm candidates agreed within 50 mm on 75.68%, 77.60%, 78.82% and 78.15% of development pixels. The corresponding p95 differences were 222.34, 265.65, 233.39 and 208.23 mm. Selected 12 mm for its higher within-50-mm agreement while keeping p95 within 25% of the best candidate. The full fusion uses all 3,657 fit views and extracts surfaces at TSDF weight greater than 2.5. For native RGB training, initial geometry is sampled from the full-resolution tiles directly, without downsampling the source color frames or loading all mesh triangles into the training process. The 8 mm ARKit result is preserved for reference. [Geometry selection](../results/mo_quality_20261121/pose_refinement/geometry_selection.json) · [Refined run launcher](../results/mo_quality_20261121/launch-refined-run.ps1) · [Open3D TSDF integration kernel](https://github.com/isl-org/Open3D/blob/main/cpp/open3d/t/geometry/kernel/VoxelBlockGridImpl.h)

The full corrected 12 mm mesh finished at 23:45 ICT with eight spatial tiles, 4,481,343 vertices, 8,630,201 triangles and 433.76 m² of retained surface. Conservative free-space integration contributed 151,688 block updates. Across the original 103 reserved views, coverage was 99.82%, within-50-mm agreement was 78.75%, and median/p95 absolute discrepancy was 14.97/319.70 mm. The original ARKit 8 mm result had 99.78% coverage, 43.45% within 50 mm, and 74.76/721.94 mm median/p95. Pose corrections, voxel size and fusion policy all changed between these full meshes. Native A05-Abs initialization began with 1,850,000 samples from the corrected mesh. [Corrected mesh](../results/mo_quality_20261121/geometry/scene.glb) · [Full geometry scores](../results/mo_quality_20261121/geometry/metrics.json) · [Training log](../results/mo_quality_20261121/training.log)

At 00:06 ICT on 2026-11-22, the corrected A05-Abs run had completed 10,000 of 48,000 scheduled updates. The 3,000- and 10,000-update checkpoints held 2,273,390 and 2,766,285 Gaussians. Training keeps periodic atomic resume states with optimizer, growth statistics and RNG state. The first opacity reset ran at update 6,000. [Training progress](../results/mo_quality_20261121/training.log) · [Training settings](../results/mo_quality_20261121/A05_absgrad_quality/protocol.json)

### 2026-11-22 — Training recovery

After saving update 12,000 with 2,896,987 Gaussians, the original process made no new logged progress for about eight minutes. The GPU showed approximately 5,906 MiB used, 61 MiB free and 100% compute utilization during the stall. The precise driver or kernel cause was unconfirmed. Verified the optimizer snapshot archive CRC, stopped the stalled process and resumed from the saved model, optimizer, RNG, losses and unchanged camera sequence. Ended growth and further opacity resets at update 12,000 to stay within the measured GPU headroom. Kept the full 1920×1440 objective and the 48,000-update schedule. At 00:21 ICT the resumed run passed update 12,500 with finite loss. [Before resume](../results/mo_quality_20261121/training_before_resume.log) · [Continued training](../results/mo_quality_20261121/training.log)

### 2026-11-22 — Verified A05-Abs endpoint and scene exports

Training reached update 48,000 at 01:39 ICT. The final model holds 2,896,987 Gaussians. Across 114 refinement events it split 1,416,736 eligible parents and pruned 369,749 Gaussians. Training reported 6,292.89 seconds across both processes, 5,475 MiB peak allocated and 7,040 MiB peak reserved by PyTorch. Readback checked the frozen inputs, training sources, six checkpoint hashes and tensor shapes, growth arithmetic and all 48,000 finite losses. [Training record](../results/mo_quality_20261121/A05_absgrad_quality/training.json) · [Verified summary](../results/mo_quality_20261121/A05_absgrad_quality/summary.json)

| Update | Development PSNR (dB) | SSIM | LPIPS Alex ↓ | LiDAR pixels within 50 mm |
| ---: | ---: | ---: | ---: | ---: |
| 20,000 | 21.596 | 0.8454 | 0.3050 | 44.01% |
| 32,000 | 21.722 | 0.8467 | 0.2932 | 36.99% |
| 40,000 | 21.874 | 0.8487 | 0.2872 | 35.15% |
| 48,000 | 21.894 | 0.8491 | 0.2841 | 33.49% |

The frozen joint selection rule chose update 20,000. Update 48,000 gave the best appearance metrics, so it was also exported for appearance-first use. Both were chosen using the 77 development views and evaluated on the same 26 reserved final views. The fits use the same capture for RGB and depth, and the depth comparisons measure LiDAR agreement. [Joint selection PLY](../results/mo_quality_20261121/A05_absgrad_quality/scene.ply) · [Appearance PLY](../results/mo_quality_20261121/A05_absgrad_quality/scene_visual.ply)

| Final result | PSNR (dB) | SSIM | LPIPS Alex ↓ | LiDAR pixels within 50 mm | LiDAR median / p95 (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Corrected TSDF mesh | — | — | — | 81.31% | 12.52 / 376.09 |
| A05-Abs at 20,000 | 20.890 | 0.8265 | 0.3458 | 49.82% | 49.69 / 695.40 |
| A05-Abs at 48,000 | 21.533 | 0.8329 | 0.3212 | 40.65% | 70.94 / 729.21 |

Inspected the source-to-render panels and 13 predeclared crop pairs per checkpoint. At 48,000, wood grain and stone carving scored 33.23 and 23.96 dB compared with 31.04 and 22.83 dB at 20,000. Much of the Vietnamese sign body text is readable in the close-up, though edges are distorted. The 48,000-step sign crop retained 0.454 of the source-aligned Scharr gradient energy by the signed projection metric compared with 0.338 at 20,000. Small jewelry and labels, glass-case edges, and some oblique frame and ceiling views remain soft or distorted. The glass edge and jewelry crops have essentially no eligible LiDAR pixels within 50 mm of either rendered depth despite the nonzero high-confidence reference counts. Optical reflections and missing or ambiguous LiDAR returns affect comparisons on transparent surfaces. The corrected mesh gives the closest agreement to LiDAR among the delivered artifacts. [Appearance preview views 1–6](../results/mo_quality_20261121/A05_absgrad_quality/final_visual_comparison_1.jpg) · [Appearance preview views 7–12](../results/mo_quality_20261121/A05_absgrad_quality/final_visual_comparison_2.jpg) · [Surface metrics](../results/mo_quality_20261121/A05_absgrad_quality/surface_details_visual.csv)

Verified both 683,690,410-byte PLYs by loading all positions, log-scales, opacity logits, quaternions and spherical harmonics back and checking exact equality with the exported tensors. Rechecked both file hashes and opened all 52 final native renders and 26 surface-detail panels. Reopened the GLB and confirmed eight exported tiles and 8,630,201 triangles. The corrected run and the appearance export completed successfully. [Native final renders](../results/mo_quality_20261121/A05_absgrad_quality/evaluation/final/) · [Corrected TSDF mesh](../results/mo_quality_20261121/geometry/scene.glb)

### 2026-11-22 — Live 48,000-update scene

Started the interactive A05-Abs appearance checkpoint on loopback port 8098. The viewer loaded 2,896,987 Gaussians and offers seven captured viewpoints. The HTTP check returned 200, the checkpoint hash matched the training record, and the exhibit sign preset rendered in the in-app browser. Left the viewer process running and the browser tab open. Verified at 2026-11-22 06:33:50 +07:00. [Local viewer](http://127.0.0.1:8098) · [Viewer launcher](../results/mo_quality_20261121/live/launch-viewer.ps1) · [Verification](../results/mo_quality_20261121/live/verification.json)

## 2026-11-22 — High-quality training request for `NEWSCENCE.ZIP`

Intake checked the repository, the rest of the accessible D: drive, and the Windows user directory for the requested archive and likely spelling variants. Found two previously trained candidate inputs, `newscan.zip` (the iPhone rescan with a matched 20,000-update comparison) and `mo.zip` (Ancient Tombs of Sai Gon with a 48,000-update appearance checkpoint). The requested archive name did not match a local file. Requested the exact path or confirmation of a candidate. Training is pending dataset identification. No checkpoint or GPU job was created for this request.

### Confirmed input and native-resolution quality run

The user confirmed `newscan.zip`. Its SHA-256 `ea5f874767dea37a950a8accaca69b3c6caff186f6149fb3feb38aaff63e35a6` matches the earlier intake for capture `e1cfcfffe4`. Reused the verified 10 mm TSDF surface, native 1920×1440 RGB cache, ARKit poses and per-frame calibration, plus the frozen 1,815 fit and 76 held-out views. The original track audit found a zero-frame offset had the lowest reprojection discrepancy in each of three capture sections. [Archive record](../results/iphone_rescan_v2/dataset_audit/archive_sha256.json) · [Alignment audit](../results/iphone_rescan_v2/dataset_audit/rgb_depth_alignment/flow_pose_lag.json) · [Frozen split](../results/iphone_rescan_v2/splits/frame_split.json)

Launched a fresh seed-43 A05-Abs run at 06:50:20 ICT with 1,500,000 Gaussians and 48,000 scheduled native updates. Initialization samples 72% by mesh area and 28% by area-weighted color contrast. The objective is 0.8 RGB L1 + 0.2 DSSIM with packed rasterization and SH degree rising to 3. Absolute-gradient Long-Axis Split can refine through 16k, with opacity resets at 6k and 12k when refinement is active. The run stops growth early if reusable CUDA memory drops below 1,400 MiB or 32% of allocated memory, whichever is greater. Saves include milestones at 3k, 10k, 20k, 32k, 40k and 48k and atomic training resumes every 3k until the endpoint. Adapted the quality initializer to read the scan's single GLB and hash its uncompressed RGB cache. [Training launch](../results/iphone_rescan_v2/A05_absgrad_quality_48k/launch-training.ps1) · [Training log](../results/iphone_rescan_v2/A05_absgrad_quality_48k/training.stdout.log)

Frozen appearance selection compares 20k, 32k, 40k and 48k of the new run with the archived 20k A05-Abs baseline, using the lowest mean AlexNet LPIPS on the same 76 views. PSNR, SSIM, six source-defined text crops and English OCR at frame 2277 corroborate appearance. Depth coverage, within-50-mm agreement and shared-pixel residuals against the phone's LiDAR are reported separately. This is a single-seed comparison of training recipes. The 76 views exclude all training images, come from the same capture and have been used previously to judge model quality. No independent scan or physical ground truth is available.

At update 3,000, the run checkpoint held 1,736,607 Gaussians. The CUDA headroom guard stopped refinement at update 3,100 with 1,239 MiB available for growth. The trainer continues RGB and SH optimization at full resolution on the original 48,000-update schedule. [Training progress](../results/iphone_rescan_v2/A05_absgrad_quality_48k/training.stdout.log)

At 07:26 ICT the run completed update 20,000 with 1,744,170 Gaussians and no reported errors. Compared the saved camera sequences and confirmed the new run shares the exact first 20,000 camera IDs with the archived A05-Abs baseline. [Training log](../results/iphone_rescan_v2/A05_absgrad_quality_48k/training.stdout.log)

### Training completion and evaluation

The training process exited with status 0 at 08:15:57 ICT. It finished 48,000 updates in 5,084.01 s (84 min 44 s) and saved all six scheduled checkpoints. The endpoint holds 1,744,170 Gaussians. PyTorch reported 3,452.81 MiB peak allocated and 4,142 MiB peak reserved on the GTX 1660 SUPER. Evaluation began at 08:16:28 ICT. Before scoring, repeated OCR on the archived source and render using the original 100-word transcript and reproduced the 0% source and 81% archived-render word error rates. [Training record](../results/iphone_rescan_v2/A05_absgrad_quality_48k/training.json) · [Evaluation command](../results/iphone_rescan_v2/A05_absgrad_quality_48k/launch-evaluation.ps1)

Evaluation exited with status 0 at 08:19:22 ICT. Readback verified the input and source snapshot hashes, the frozen split and camera sequence, all 48,000 finite losses, and six checkpoint hashes and tensor shapes. At 25 refinement events, 248,194 eligible parents split and 4,024 Gaussians were pruned. The peak pre-prune count was 1,744,601. Evaluated and verified 304 new native RGB PNGs and their 256×192 depth and opacity arrays. [Readback and evaluation summary](../results/iphone_rescan_v2/A05_absgrad_quality_48k/summary.json)

| Checkpoint | Mean PSNR (dB) ↑ | Mean SSIM ↑ | AlexNet LPIPS ↓ | Mean text edge retention ↑ | English poster OCR word error ↓ | LiDAR within 50 mm ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Archived A05-Abs 20k | 19.737 | 0.74495 | 0.34600 | 0.08602 | 81% | 75.58% |
| Quality run 20k | 19.703 | 0.74444 | 0.34782 | 0.07116 | 99% | 76.30% |
| Quality run 32k | 19.910 | 0.74561 | 0.32877 | 0.09512 | 83% | 74.83% |
| Quality run 40k | 20.028 | 0.74633 | 0.32544 | 0.08016 | 95% | 74.61% |
| Quality run 48k | 20.058 | 0.74615 | 0.32149 | 0.09247 | 82% | 73.93% |

Selected 48k by the frozen appearance rule. Against the archived 20k model, it improved LPIPS on 74 of 76 views, PSNR on 60 and SSIM on 37. Mean LPIPS dropped by 7.1% and mean PSNR rose by 0.32 dB. Text-edge retention improved in all six crops. The 32k checkpoint scored highest on mean crop PSNR, SSIM and edge retention. Visual inspection of the 48k result shows recognizable large headings and preserved exhibit layouts, with fine lettering, wire edges and reflections still blurred or distorted. On the English close-up, OCR remained poor at 82% word error versus 81% at archived 20k and 0% on the source. [Selected views 1–6](../results/iphone_rescan_v2/A05_absgrad_quality_48k/selected_views_1.jpg) · [Selected views 7–12](../results/iphone_rescan_v2/A05_absgrad_quality_48k/selected_views_2.jpg) · [English poster comparison](../results/iphone_rescan_v2/A05_absgrad_quality_48k/poster_comparisons/002277.png) · [Directory comparison](../results/iphone_rescan_v2/A05_absgrad_quality_48k/poster_comparisons/008275.png) · [Per-crop scores](../results/iphone_rescan_v2/A05_absgrad_quality_48k/evaluation/poster_frames.csv)

The selected model's median native RGB render time was 11.91 ms versus 10.23 ms for archived 20k. Depth coverage rose from 99.81% to 99.92%, while the fraction of eligible LiDAR pixels within 50 mm decreased by 1.65 percentage points. On the 2,865,684 eligible LiDAR pixels shared by all five checkpoints, median/p95 absolute depth discrepancy rose from 20.91/221.05 mm to 22.99/243.28 mm. Across 39,967 shared opaque TSDF reference pixels, median/p95 changed from 9.15/38.17 mm to 9.59/28.44 mm. These references come from the same capture. [Per-frame depth scores](../results/iphone_rescan_v2/A05_absgrad_quality_48k/evaluation/depth_frames.csv)

Exported the selected degree-3 model to a 411,625,598-byte Gaussian PLY. Reopened the PLY and verified exact equality with the exported positions, scales, rotations, opacity logits and spherical harmonic coefficients. Independently rechecked the PLY SHA-256 `55d379dfc5786d5ff56d9b81119744a72dcd36b8d88d56369542cb88901fa17d`. [Scene PLY](../results/iphone_rescan_v2/A05_absgrad_quality_48k/scene.ply) · [48k checkpoint](../results/iphone_rescan_v2/A05_absgrad_quality_48k/checkpoints/A05_absgrad_uncapped_48000.pt) · [Evaluation source](../results/iphone_rescan_v2/A05_absgrad_quality_48k/evaluate.py)

### Local interactive preview

At 08:22:49 ICT, verified the 48,000-update checkpoint hash and started the interactive rescan viewer on loopback port 8099. The viewer loaded 1,744,170 Gaussians and returned HTTP 200. It exposes six saved viewpoints. In the in-app browser, verified that the exhibit overview and English poster presets render and left the viewer open on the English poster. [Local viewer](http://127.0.0.1:8099/) · [Launch script](../results/iphone_rescan_v2/A05_absgrad_quality_48k/live/launch-viewer.ps1) · [Verification](../results/iphone_rescan_v2/A05_absgrad_quality_48k/live/verification.json)

## 2026-11-22 — Tesla Reality View comparison

Tesla began rolling out FSD (Supervised) v15.0.1 in software build 2026.38.6.11. Reality View uses a virtual camera above the moving vehicle to generate a live view of its surroundings and shows planned turns and parking intent. Tesla describes the visualization as part of the end-to-end driving model. The launch materials reviewed describe the feature and its integration. They leave the production scene representation, camera-to-display latency, and metric-geometry accuracy unspecified. [Tesla announcement](https://x.com/Tesla/status/2124222447855743019) · [Vehicle release notes](https://www.notateslaapp.com/software-updates/version/2026.38.6.11/release-notes#tesla-reality-view) · [Ashok Elluswamy's launch explanation](https://x.com/aelluswamy/status/2124224924512641512)

At ICCV 2025, Elluswamy described Tesla's Generative Gaussian Splatting as producing Gaussians from production vehicle cameras, supporting moving objects and operating in about 220 ms without scene-specific initialization. He identified the limited range of viewpoints along typical driving trajectories as a challenge for traditional 3DGS. That disclosure establishes a technical connection to this project and explains how a learned model can address driving scenes. The v15 launch materials reviewed leave it unclear whether the shipping renderer uses that exact method. The 220 ms figure applies to the research system as described in 2025. [ICCV write-up by Ashok Elluswamy, mirrored](https://xitter.gq/aelluswamy/article/1981644831790379245)

The project comparator is the latest verified 48,000-update A05-Abs run on `newscan.zip`. It uses recorded iPhone RGB, LiDAR and ARKit poses to construct a 10 mm TSDF mesh. The mesh initializes explicit Gaussians optimized for that static exhibit using RGB supervision. The outputs are a reusable mesh and an exportable 1,744,170-Gaussian PLY that can be inspected from new viewpoints. Training took 84 min 44 s on a GTX 1660 SUPER. Once trained, median native RGB render time was 11.91 ms. That timing measures rendering an existing model and excludes capture, fusion and fitting. Reality View updates the displayed scene as the vehicle moves and its surroundings change. The available timings cover different stages and do not support a speed ranking. [Training and evaluation readback](../results/iphone_rescan_v2/A05_absgrad_quality_48k/summary.json) · [Exported PLY](../results/iphone_rescan_v2/A05_absgrad_quality_48k/scene.ply)

Both systems synthesize viewpoints away from the physical cameras. In this project, appearance and depth agreement are measurable separately. Across 76 held-out views from the same iPhone capture, the 48k run reduced mean LPIPS from 0.34600 to 0.32149 relative to archived 20k A05-Abs, while eligible LiDAR pixels within 50 mm fell from 75.58% to 73.93%. English poster OCR word error was 82% at 48k. These results quantify agreement with the same capture and leave independent physical accuracy open. The review provides related-work context for static RGB-D reconstruction and live automotive visualization. It uses existing project measurements and the public Tesla disclosures, with no shared dataset or new benchmark. [Depth scores](../results/iphone_rescan_v2/A05_absgrad_quality_48k/evaluation/depth_frames.csv) · [Poster scores](../results/iphone_rescan_v2/A05_absgrad_quality_48k/evaluation/poster_frames.csv)

### 2026-11-22 — Why learned reconstruction can run faster

The 48,000-update A05-Abs run fits Gaussian parameters for one exhibit by rendering sampled training views, comparing to the recorded RGB, and backpropagating the error. The 84 min 44 s is per-scene training. The median 11.91 ms is a native RGB render from the already fitted scene. [Training and evaluation summary](../results/iphone_rescan_v2/A05_absgrad_quality_48k/summary.json)

A feed-forward Gaussian reconstructor learns the image-to-Gaussian mapping during training and applies fixed network weights to a new scene. Thus it can replace thousands of scene-specific optimization steps with inference. pixelSplat provides a published example of this design. Tesla's 2025 description of its separate Generative Gaussian Splatting system is consistent with this approach and gives approximately 220 ms for the research system with no scene-specific initialization. Tesla's launch notes describe a continuously streamed visualization and give no camera-to-display latency or display frame rate. The 220 ms figure therefore cannot be used to derive the shipping UI frame rate. No new run was conducted for this follow-up. [pixelSplat paper](https://openaccess.thecvf.com/content/CVPR2024/html/Charatan_pixelSplat_3D_Gaussian_Splats_from_Image_Pairs_for_Scalable_Generalizable_CVPR_2024_paper.html) · [Tesla ICCV write-up](https://xitter.gq/aelluswamy/article/1981644831790379245) · [Vehicle release notes](https://www.notateslaapp.com/software-updates/version/2026.38.6.11/release-notes)

## 2026-11-24 — Test-time Baking paper review

Chin-Yang Lin et al., *TTB: Test-time Baking for Efficient Rendering of Decoder-only View Synthesis Models*, arXiv:2611.17331v1, posted 12 November 2026 (UTC), NeurIPS 2026 Spotlight. The listed affiliations are NVIDIA and National Yang Ming Chiao Tung University in Taiwan. [Paper and affiliations](https://arxiv.org/html/2611.17331v1) · [Project page](https://test-time-baking.github.io/)

TTB builds on a pretrained Decoder-only View Synthesis Model (DVSM). Given posed RGB images, it runs cross-view attention during scene setup and uses those outputs to supervise small per-scene MLPs. The MLP inputs come from camera-only context tokens, which match what is available when rendering a new pose. The full method fits these fast weights with three inner-loop updates using mean squared error. Once baked, the renderer uses the MLPs to synthesize RGB views. At fixed output resolution, the base variant's rendering cost and fast-weight size are independent of the number of input views. Scene setup still scales quadratically in the view count at fixed resolution. The paper reports 324 MB of FP32 fast weights for the base variant. The local-duplicate variant stores additional weight sets to increase capacity. [Method and appendices B–C](https://arxiv.org/html/2611.17331v1)

Author-reported DL3DV results on 140 held-out scenes with 64 posed context views at 544 × 960. DVSM and TTB timings are on an A100. Setup is per scene and FPS measures rendering after setup. [Benchmark and timing tables](https://arxiv.org/html/2611.17331v1) · [3DGS comparison](https://test-time-baking.github.io/)

| Method | Setup | Render FPS | PSNR (dB) | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| DVSM KV-cache | 60 s | 1 | 31.11 | 0.104 |
| TTB | 65 s | 23 | 30.20 | 0.131 |
| TTB local-duplicate | 65 s | 23 | 31.33 | 0.108 |
| TTB-minus | 7.5 s | 23 | 29.29 | 0.169 |
| 3DGS | >600 s | >50 | 27.25 | 0.193 |

The 23× figure refers to render throughput against DVSM at 64 views. With setup included, the paper estimates 69.3 s for standard TTB to render 100 novel views versus 160 s for DVSM. The authors report 18 GB peak inference memory for standard TTB with 32 input views at the benchmark resolution, including scene setup and rendering. This exceeds the 6 GB on the project's GTX 1660 SUPER. Feasibility at lower resolution or with fewer views remains unverified. The release provides CUDA inference, pretrained checkpoints, a posed-image input path, and a viewer that consumes a baked scene artifact. [Paper tables 14–15](https://arxiv.org/html/2611.17331v1#A3.SS6) · [Implementation](https://github.com/NVlabs/DVSM) · [Checkpoints](https://huggingface.co/nvidia/DVSM)

Project relevance: TTB is a candidate appearance baseline for the existing `newscan.zip` A05-Abs reconstruction. Its released renderer returns RGB from a latent scene state. The project's LiDAR TSDF and Gaussian PLY provide explicit geometry for inspection and depth evaluation. A matched appearance experiment would use the same posed context frames and held-out cameras, keep both renderers at the same scoring resolution, and record setup time, frame latency, peak VRAM, RGB metrics and poster OCR. A 64-view A05-Abs run would control for input count alongside the existing full-input project result. This entry records a literature review. No local TTB inference, checkpoint download, or matched comparison was performed. [A05-Abs run summary](../results/iphone_rescan_v2/A05_absgrad_quality_48k/summary.json) · [TTB inference interface](https://github.com/NVlabs/DVSM#inference)

### Why the A05-Abs result looks softer than the TTB demos

TTB starts from a model trained across scenes and uses a learned image decoder to synthesize RGB from posed photos. Its scene bake adapts pretrained features. A05-Abs fits explicit Gaussian positions, scales, opacities and colors to a single exhibit capture. The paper's demos and the project's held-out views differ in scene, camera path and display scale, so their apparent gap cannot establish how the two methods compare on this exhibit. [TTB paper](https://arxiv.org/html/2611.17331v1) · [TTB demos](https://test-time-baking.github.io/)

The `newscan.zip` quality run completed 48,000 updates. The CUDA headroom guard stopped Gaussian refinement at 3,100, after which optimization continued with the existing primitives. Relative to archived 20k A05-Abs, 48k improved mean PSNR from 19.737 to 20.058 dB and reduced LPIPS from 0.34600 to 0.32149 across 76 held-out views. On the English poster close-up, OCR word error was 0% in the source frame, 81% in the 20k render and 82% in the 48k render. The score-frame print is readable in the source, but that alone leaves its coverage and sharpness in the fit views unresolved. The experiment confirms a gain in overall appearance with little change in text legibility. [Run summary](../results/iphone_rescan_v2/A05_absgrad_quality_48k/summary.json) · [Poster scores](../results/iphone_rescan_v2/A05_absgrad_quality_48k/evaluation/poster_frames.csv)

Candidate explanations include subpixel pose misalignment around the print, limited sharp close-up coverage in fit views, or splats with footprints too broad to resolve letter strokes. Reflections and thin geometry present separate challenges. The existing evidence does not establish which factor dominates. A controlled diagnostic could use the frozen split to compare one poster crop in nearby fit views and held-out views. A sharp fit-view render with a soft held-out render would motivate checking viewpoint coverage and pose alignment. Soft renders even on sharp fit views would motivate checking projected splat footprints, pose alignment and whether longer refinement helps under a fixed memory budget. This follow-up analyzes existing results. No new model was fitted or tested.

### Whole-scene PSNR target for newscan

Read the saved full-frame, native-resolution PSNR scores for all 76 held-out views from the 48k A05-Abs run. Mean was 20.0584 dB, median was 19.9595 dB, and the range was 14.3937–28.2419 dB. 39 views were below 20 dB, three reached 25 dB, and zero reached 30 dB. Frames 9782 and 7315 were the minimum and maximum, respectively. Raising mean per-view PSNR to 30 dB on this fixed split and with the same RGB convention would require a 9.87-fold reduction in geometric-mean per-view MSE, computed as 10 raised to ((30 − mean PSNR) / 10). From 20k to 48k within the quality run, mean PSNR increased from 19.703 to 20.058 dB. The distribution shows errors across many whole-frame scores. [Saved per-view PSNR](../results/iphone_rescan_v2/A05_absgrad_quality_48k/evaluation/quality_48000/A05_absgrad_uncapped_frames.csv)

TTB's 30.20 dB standard result measures a different set of scenes and uses 544 × 960 scoring. It gives context for a target. It does not establish a 30 dB ceiling or expectation for this capture. A first whole-scene experiment would classify low- versus high-scoring views by pose reprojection, neighboring view overlap, exposure and motion blur, and compare A05-Abs on the same 76 cameras at native resolution and at a documented aspect-preserving resolution capped at a 960-pixel long edge. Keep resolution-specific metrics explicit. Then compare controlled pose/photometric correction and later budgeted splitting on the same fit/score split. Prioritize improvements that lift the lower-scoring views and reduce mean LPIPS, while reporting LiDAR depth agreement to track the existing geometry objective. This follow-up recomputed statistics from the archived evaluation. No new model, rescaled score or matched TTB run was measured. [TTB benchmark](https://arxiv.org/html/2611.17331v1)

### Refinement cutoff provenance

The assistant's 48k launch used quality mode with refinement scheduled up to step 16,000 and explicitly passed a 1,400 MiB growth headroom floor. At each eligible refinement check, the trainer compares free CUDA memory plus reusable PyTorch reserved memory with the greater of that floor and 32% of allocated memory. The runtime logged 1,239 MiB at step 3,100 and disabled refinement at that check. The 3,100 value was determined by the guard during training. Optimization continued through 48,000 updates. Verified the saved launch command, trainer and stdout for this clarification. [Launcher](../results/iphone_rescan_v2/A05_absgrad_quality_48k/launch-training.ps1) · [Trainer](../scripts/iphone_rescan_a05.py) · [Runtime log](../results/iphone_rescan_v2/A05_absgrad_quality_48k/training.stdout.log)

## 2026-11-24 — HD A05-Abs rerun with the memory guard disabled

Started a fresh seed-43 run on `newscan.zip` with the existing 1,815 fit views, 76 held-out views, calibrated poses, TSDF mesh and 1.5 million initial Gaussians. The trainer area-downsamples the native 1920 × 1440 RGB to 960 × 720, preserving the 4:3 field of view and scaling both rows of the RGB camera intrinsics by one half. Depth calibration stays at its existing resolution. Target training length is 48,000 updates. The launch passes `--growth-headroom-mib 0` and uses no Gaussian count cap. A05-Abs retains its scheduled refinement window through step 16,000. This run uses the 960-pixel long edge as an HD-scale setting inspired by TTB's benchmark. TTB adapts a pretrained renderer and has no corresponding Gaussian refinement schedule. The previous native-resolution result remains available as a baseline. [HD training launcher](../results/iphone_rescan_v2/A05_absgrad_hd720_48k/launch-training.ps1) · [Trainer](../scripts/iphone_rescan_a05.py)

Verified the new protocol reports 960 x 720 RGB, guard disabled, and no Gaussian count cap. The 48,000 training camera IDs have the same SHA-256 as the prior native run. At 4,500 updates the HD run had 1,901,307 Gaussians, confirming growth after the old 3,100 cutoff. Evaluation is queued to run after successful training and will score both 48k models on the same held-out cameras at each of 1920 x 1440 and 960 x 720. [Training output](../results/iphone_rescan_v2/A05_absgrad_hd720_48k/training.stdout.log) - [Protocol](../results/iphone_rescan_v2/A05_absgrad_hd720_48k/protocol.json)

## 2026-11-24 — Git checkout and host setup for the pipeline

Added root-specific ignore rules for `710d9d02a3.zip`, `mo.zip`, `newscan.zip`, and the local report template, plus a rule for compiled Python extensions on Windows. The existing rules cover sensor databases, datasets, results, toolchains, environments and caches. `git check-ignore` verified that local artifacts are excluded and the trainer, its dependencies, tests, CUDA sources and curated fixtures remain eligible for version control. No tracked files matched the ignore rules and none of the eligible untracked files exceeded 100 MiB. The trainer and evaluator both returned `--help` successfully in the existing Python environment. CUDA execution and setup on a second machine were outside this check.

The `git add` dry run included both training entry points and excluded the capture archives. Checked that the prepared HD-run inputs exist and that the 15,684,710,400 cache bytes match the dimensions and IDs for the 1,815 fit and 76 held-out frames.

Commit the pipeline source changes along with `.gitignore`. In particular, `scripts/iphone_rescan_a05.py` and its iPhone and Stray Scanner helpers were untracked at the time of this check. After pushing branch `new`, clone it with both CUDA/test submodules:

```powershell
git clone --recurse-submodules --branch new https://github.com/so1vanlaso1/3dgs.git
cd 3dgs
```

For Windows, use Python 3.12 and an x64 Visual Studio Developer PowerShell with a working NVIDIA driver, CUDA toolkit and MSVC C++ toolchain. The source machine uses CUDA 12.8 with PyTorch `2.9.1+cu128` and torchvision `0.24.1+cu128`. In the new environment install the matching CUDA wheels first, then the Python and CUDA-extension dependencies below. Set `CUDA_HOME` to the installed toolkit if it is outside the default discovery path. The local PowerShell launcher expects tools under `.tools/` and targets compute capability 7.5. The direct Python command uses the destination toolchain and lets PyTorch detect the destination GPU.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install setuptools wheel ninja numpy rich opencv-python-headless Pillow scipy torchmetrics==1.8.2 open3d==0.20.0 trimesh==5.1.1 zstandard==0.25.0
python -m pip install --no-build-isolation "git+https://github.com/rahul-goel/fused-ssim@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38"
$env:BUILD_NO_CUDA = '1'
python -m pip install --no-build-isolation -e .
Remove-Item Env:BUILD_NO_CUDA
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
```

The editable install makes this checkout importable and the CUDA backend builds on first use. Transfer the prepared `dataset_audit/`, `geometry/`, and `splits/` directories out of band, preserving their paths under `results/iphone_rescan_v2/`. The current `newscan.zip` run uses a 15.68 GB `dataset_audit/native_rgb.bin`. It also needs the cache frame list, preflight data, alignment audit, archive hashes, camera splits and mesh in those directories. Keep the original capture separately for preprocessing or depth evaluation. For result comparisons, also transfer the extracted capture and the chosen baseline's output directory.

Run the HD experiment from the clone root with a fresh output directory. The command repeats the current HD run's 960-pixel long edge, training schedule and disabled growth guard. Available VRAM on the destination determines how far uncapped refinement can run.

```powershell
python scripts/iphone_rescan_a05.py results/iphone_rescan_v2 results/iphone_rescan_v2/A05_absgrad_hd720_48k_repro --quality --initial-gaussians 1500000 --steps 48000 --rgb-long-edge 960 --growth-headroom-mib 0
```

The initial process became several times slower as Windows GPU memory use approached 6 GiB. It completed the atomic 12,000-update resume save and later reported update 12,500. Verified the saved archive structure, stopped that process, and resumed from update 12,000 in a fresh CUDA process with an allocator garbage collection threshold of 0.75 to reclaim unused cached blocks sooner. The run replays the unsaved updates. The camera sequence, Gaussian refinement schedule and disabled growth guard stay the same. Evaluation is queued behind the resumed process. [Initial training log](../results/iphone_rescan_v2/A05_absgrad_hd720_48k/training.stdout.log) - [Resume log](../results/iphone_rescan_v2/A05_absgrad_hd720_48k/training.resume.stdout.log)

Stopped the resumed HD training and its queued evaluation at the user's request. Verified both were inactive. The resumed trainer loaded the 12,000-update state before it was stopped. The latest durable resume remains at 12,000 updates, with regular checkpoints at 3,000 and 10,000. No final checkpoint, evaluation, or new PSNR result was produced. [Resume output](../results/iphone_rescan_v2/A05_absgrad_hd720_48k/training.resume.stdout.log)

## 2026-11-24 — NVIDIA TTB versus A05-Abs timing

NVIDIA and NYCU train TTB across scenes from a DVSM checkpoint using 64 A100s, with 100,000 updates at 340p and 10,000 each at 480p, 720p and 960p. The paper gives the schedule and a separate forward-plus-backward measurement of 90 s per iteration at 544 × 960 with 32 context views. Total training wall time is unreported, and extrapolating that measurement across the mixed-resolution schedule would be unsupported. A05-Abs used one GTX 1660 SUPER to fit one scene, completing 48,000 updates in 5,084.01 s (84 min 44 s) after the scan's TSDF mesh was already available. [TTB paper, section 4.1 and table 14](https://arxiv.org/html/2611.17331v1) · [A05-Abs training record](../results/iphone_rescan_v2/A05_absgrad_quality_48k/training.json)

Starting from 64 posed DL3DV views, standard TTB bakes a scene in 65 s on one A100, using the trained model and three inner-loop updates. It takes 18 s with 32 views. The completed A05-Abs `newscan.zip` fit used 1,815 views at 1920 × 1440 and is the corresponding step to make that scene renderable. The reported 64-view TTB setup is about 78× shorter than the A05-Abs fit. Once the scene is ready, TTB renders at 23 FPS at 960 × 544 (about 43.5 ms/frame). The A05-Abs median native RGB render is 11.91 ms across 76 held-out views (about 84 FPS from inverse median latency, 3.65× lower latency). This is a renderer measurement without viewer delivery or display. The source timings use different GPUs, captures, input counts, output resolutions and scene representations. The stopped 960 × 720 A05-Abs run provides no final timing. No matched timing or local TTB run was conducted. [TTB project and GPU](https://test-time-baking.github.io/) · [TTB tables 14–15](https://arxiv.org/html/2611.17331v1#A3.SS6) · [A05-Abs evaluation](../results/iphone_rescan_v2/A05_absgrad_quality_48k/summary.json)

## 2026-11-25 — Same-exhibit comparison with NVIDIA Test-time Baking

The experiment asks how the two pipelines render the same held-out views of the exhibit, including whether either preserves the poster lettering. TTB adapts a pretrained decoder to posed RGB views by fitting scene-specific MLP weights to cross-view attention outputs. A05-Abs fuses captured LiDAR into a TSDF mesh and uses that mesh to initialize explicit Gaussians that it fits to the scene's RGB. Both render novel RGB views. A05-Abs also yields a mesh and Gaussian PLY for geometry inspection, while the released TTB renderer yields RGB from a scene stored in network weights. [TTB paper](https://arxiv.org/html/2611.17331v1) · [A05-Abs evaluation](../results/iphone_rescan_v2/A05_absgrad_quality_48k/summary.json)

For the shared-view comparison, select 64 frames from the existing fit split and use their calibrated poses for both pipelines. Score the same 76 held-out cameras at a shared image size and field of view, with intrinsics adjusted for any resize or crop. Compare full-frame PSNR and LPIPS alongside fixed poster crops and OCR. Record scene setup time, render latency and peak VRAM, including LiDAR fusion in the full A05-Abs setup time and identifying the fit time separately. The existing 1,815-frame A05-Abs result is the reference for the full capture. A 64-view A05-Abs variant that uses the full-capture mesh still has access to LiDAR from outside the 64 views and should be identified as such. A matched RGB-only method comparison would require RGB-only A05-Abs initialization. The comparison remains unrun. [TTB inference implementation](https://github.com/NVlabs/DVSM#inference) · [A05-Abs evaluation](../results/iphone_rescan_v2/A05_absgrad_quality_48k/summary.json)

The 65-second TTB bake reuses a decoder trained across scenes on 64 A100s. It aggregates the new scene's posed photos with cross-view attention and uses three updates to fit the scene-specific weights to those features. A05-Abs starts with LiDAR-derived geometry and repeatedly renders RGB views and backpropagates to fit the explicit Gaussians over 48,000 updates. In the paper's 64-view DL3DV Table 1, standard TTB scores 30.20 dB, DVSM scores 31.11 dB, and baseline 3DGS scores 26.43 dB. The project page's separate motivation table lists 27.25 dB for 64-view 3DGS and 30.54 dB for roughly 300 views. Use the paper's Table 1 for its matched comparison. The relative contribution of learned priors, observation coverage and capture conditions to the gap with the exhibit's 20.06 dB is unknown. No new run was conducted. [TTB paper, Table 1 and section 4.1](https://arxiv.org/html/2611.17331v1) · [Project page](https://test-time-baking.github.io/) · [A05-Abs evaluation](../results/iphone_rescan_v2/A05_absgrad_quality_48k/summary.json)

### 2026-11-25 — PSNR gap and scoring-resolution check

Recomputed mean per-view RGB PSNR on the 76 archived 48k PNG renders against their cached native RGB references. At 1920 × 1440 the PNGs scored 20.0577 dB, with at most 0.0041 dB per-view deviation from the stored float scores. Area-downsampling both images to 960 × 720 in float, preserving 4:3 framing, raised the mean to 20.2457 dB (+0.1880 dB, per-view gains 0.0183–0.5223 dB). Changing scoring scale alone had a small effect on this model. This is an offline score of resized images from the same checkpoint. It does not measure model quality after training or rendering at the lower resolution. [Archived evaluation](../results/iphone_rescan_v2/A05_absgrad_quality_48k/evaluation/quality_48000/)

The earlier exhibit comparison held initialization, cameras and update count fixed at 20k. The B0 fixed-count control scored 19.677 dB and A05-Abs scored 19.737 dB, so replacing A05's refinement with that control did not recover the gap. B0 differs from the paper's original 3DGS baseline. The 48k run gained 0.355 dB from its own 20k checkpoint, with the memory guard ending refinement at update 3,100. These results motivate checking reprojection and RGB consistency in shared fit views, then comparing an original 3DGS run on the same exhibit split. The dominant source of the gap remains unresolved. [Matched 20k results](../results/iphone_rescan_v2/B0_matched_20k/summary.json) · [A05-Abs 48k results](../results/iphone_rescan_v2/A05_absgrad_quality_48k/summary.json) · [DL3DV Table 1](https://arxiv.org/html/2611.17331v1#S4.T1)
### 2026-11-25 — DVSM and TTB training recipe

The dense-view DVSM checkpoint learns across DL3DV scenes by taking posed context RGB, predicting RGB at target camera poses withheld from the context, and comparing predictions to the actual target photos using MSE plus a perceptual term weighted 0.2. DVSM samples 2, 4, 8, 16 or 32 context views and as many targets. TTB starts from the DVSM checkpoint and samples 16 or 32 context views and as many targets. During TTB training, each scene's fast weights take three inner-loop updates to approximate cross-view attention. The outer RGB rendering loss backpropagates through those three updates into the shared model. Both dense-view papers report 64 A100s and 100,000 updates at 340p followed by 10,000 each at 480p, 720p and 960p. At deployment, the shared model is fixed and only scene-specific weights are baked from the new posed photos. [DVSM paper](https://arxiv.org/html/2605.29891) · [TTB paper](https://arxiv.org/html/2611.17331v1)

### 2026-11-25 — Tesla Reality View architecture compared with TTB

Tesla's launch describes continuous video from a virtual camera above the vehicle. In a November 22 public explanation, Elon Musk described a roughly one-billion-parameter conditional mean-flow diffusion model that takes recent Universal Vision Encoder features shared with the driving system and uses the prior generated imagery to produce the next video chunk. This is additional production detail beyond the November 22 review of launch material. TTB fits per-scene MLP weights to cross-view attention features from posed photos, then reuses those weights for novel views of the captured scene. The common principle is learned novel-view synthesis. Tesla's disclosed runtime uses ongoing video generation, and the disclosure does not establish TTB-style per-scene weight fitting or that the ICCV 2025 Gaussian-splatting research ships in Reality View. Tesla's exact training recipe and a matched performance comparison remain unavailable. [Tesla launch](https://www.youtube.com/watch?v=eNBqkc7V1nU) · [Musk's technical explanation, mirrored](https://www.nitter.gq/elonmusk/status/2124458557525397780?sort=recency) · [TTB paper](https://arxiv.org/html/2611.17331v1)


### 2026-11-25 - DVSM TTB-Minus on newscan.zip

- Run directory [ttb_minus_20261125](../results/iphone_rescan_v2/ttb_minus_20261125/). Checked the archive SHA-256 against the existing dataset audit. Selected 16 RGB frames evenly by index from the frozen fit split and decoded those frames and the 76 existing held-out frames from the extracted RGB video. The context uses calibrated OpenCV camera-to-world rotations, translations, and intrinsics from the existing camera audit. The bake receives RGB and camera calibration only.
- Used NVIDIA DVSM revision `826be87c1a7c8ab3c0385b66b0ceaf8aa71f123b` and `DVSM-TTB-Minus-p8-DL3DV-960p` weights. The 1,021,910,832-byte weight file matched the published SHA-256 `0069f212531131a1655c58c50bc48a69606066dfa1756d879873c79c44945f86`. [Source revision](https://github.com/NVlabs/DVSM/tree/826be87c1a7c8ab3c0385b66b0ceaf8aa71f123b) · [Checkpoint](https://huggingface.co/nvidia/DVSM/tree/main/DVSM-TTB-Minus-p8-DL3DV-960p)
- The available GPU was a GeForce GTX 1660 SUPER with 6 GiB and compute capability 7.5. WSL2 could not start because virtualization was unavailable. Reused Windows PyTorch 2.9.1+cu128 and installed the additional Python packages into the experiment directory. Resolved the initial missing `regex` and incompatible `tokenizers` dependency. The local launcher disables compilation and uses chunked PyTorch scaled dot-product attention in place of the supplied Linux FlashAttention wheel. DVSM selects FP16 autocast on this GPU.
- The first input-preparation attempt decoded the PNGs and hit a shape error scaling intrinsics. Fixed the scale vector and reused all 92 decoded frames. The first bake used `--long_edge 480`, but DVSM's recursive image probe picked a package sample image beside the scene JSON and requested a 480 × 320 crop. Stopped that run during reconstruction. The full-frame retry uses `--target_w 480 --target_h 360`. [Context preparation](../results/iphone_rescan_v2/ttb_minus_20261125/prepare_scene.py) · [Windows launcher](../results/iphone_rescan_v2/ttb_minus_20261125/run_windows.py)

- The 16-view full-frame FP16 attempt exited and wrote a 324 MiB artifact in 206.863 s, with 6,163.7 MiB peak PyTorch allocation and 7,738.0 MiB reserved. The reload check found all three fast-weight matrices in all 12 blocks were entirely NaN. Quarantined that artifact as `rejected_fp16_nonfinite.recon.pt`. It provides no render or quality measurement.
- The retry keeps NVIDIA's weights and TTB-Minus update schedule, evaluates the fast-weight MSE update and Newton-Schulz iteration in FP32, and accumulates the gradient in 2,048-token batches. Its gradient matched PyTorch autograd on a 4,100-token check. The first compatibility smoke attempt hit an overly strict assertion on the checkpoint's gradient-checkpointing flag. The retry checks that gradients are disabled during inference. [FP32 bake shim](../results/iphone_rescan_v2/ttb_minus_20261125/fp32_bake.py)

- The FP32 smoke completed with 4 views at 240 × 176 in 14.522 s. The first FP32 run with 16 views completed in 89.014 s and rendered all 76 held-out pose IDs. A subsequent cache check found that the default FFmpeg MP4 decode differed from the audited RGB at those same IDs. Seven checked frames had mean absolute pixel differences of 4.52–22.44 on the 0–255 scale. The first bake and its scores are rejected for pose-matched comparison and archived under `rejected_default_decode`.
- The extracted `rgb.mp4` matches the ZIP-member size and CRC32 `70795cfb`. The corrected run exports all 92 selected images from the existing audited native RGB cache by frozen frame ID and checks each PNG against the cache pixel for pixel. Used the same camera calibration, view IDs, resolution, checkpoint and FP32 bake as before.

- A spot check decoded frame 0 with and without FFmpeg's `-ignore_editlist 1`. With the flag, the decoder output matched the audited cache byte for byte. The default decode had mean absolute error 16.83 on the 0–255 scale for that frame.
- The accepted 16-view run baked at 480 × 360 in **89.722 s**, including loading and saving. It used **4,137.7 MiB** peak PyTorch CUDA allocation and **4,946.0 MiB** peak reserved, and wrote **324 MiB** of finite fast weights across 12 model blocks. The artifact SHA-256 is `a9609f9c02d5b67a4ec5d9ee08f8128fd7184a0074d04364ed5f913deed81c98`. [Accepted bake log](../results/iphone_rescan_v2/ttb_minus_20261125/recon_fp32_480x360.log) · [Saved reconstruction](../results/iphone_rescan_v2/ttb_minus_20261125/newscan_16v_480.recon.pt)
- Reloaded the artifact and rendered all 76 held-out views at 480 × 360. TTB-Minus mean PSNR was **15.4984 dB**, with a median render time of **2.8364 s/view** after excluding the initial render and **1,734.2 MiB** peak PyTorch CUDA allocation. The archived 48k A05-Abs PNGs resized with Lanczos scored **20.5320 dB** on the same references. A05-Abs fit used 1,815 RGB views and LiDAR-derived initialization. The preview shows visible blur and missing detail in several TTB-Minus views. [Per-frame metrics](../results/iphone_rescan_v2/ttb_minus_20261125/heldout_metrics.json) · [Side-by-side preview](../results/iphone_rescan_v2/ttb_minus_20261125/heldout_preview.jpg)

- Started the Viser viewer on loopback port 8080 using the accepted artifact, `--max_res 448 --no_optim` and Python UTF-8 output. HTTP returned 200. A local WebSocket client verified two distinct 448 × 336 RGB frames after sending camera poses for source views 0 and 8. The viewer was left running. [Viewer startup log](../results/iphone_rescan_v2/ttb_minus_20261125/viewer.stdout.log) · [Camera movement check](../results/iphone_rescan_v2/ttb_minus_20261125/viewer_check.json)

### 2026-11-26 - AnyDesk access check

- Verified that the connected remote desktop was visible in the local AnyDesk window. The app displayed a five-minute session timeout. No remote input was sent, and shell access was not tested.

### 2026-11-26 - Native newscan quality run from a fresh clone

Added [the run entry point](../scripts/run_newscan_quality.py) and [archive preparation](../scripts/prepare_newscan.py). Starting from the audited `newscan.zip`, the pipeline freezes the original camera split, decodes 1,891 native RGB frames with the MP4 edit list disabled, checks RGB/pose alignment and builds the 10 mm TSDF. The 1.5 million initial Gaussians are sampled from the mesh. Gaussian positions, orientations, scales, opacities and spherical harmonics are optimized from the native 1920 × 1440 training images for 48,000 updates. The existing A05-Abs split/prune strategy has no Gaussian count cap and can refine until update 16,000, subject to the CUDA memory guard.

The [quality evaluator](../scripts/stray_quality_evaluate.py) scores the 20k, 32k, 40k and 48k checkpoints on 50 development views. It selects the best LPIPS among checkpoints within three percentage points of the highest fraction of eligible LiDAR pixels within 50 mm. It reports results on 26 separate final views and exports a Gaussian PLY. If a different checkpoint has the strongest appearance score, it also exports that checkpoint. The run directory contains `experiment_log.md` with the live stage and training progress, `run.log` with full output, and `model/summary.json` with scores. The RGB cache alone uses 15.68 GB.

On Ubuntu 24.04, install the system prerequisites with `sudo apt update` and `sudo apt install -y git build-essential python3.12 python3.12-venv ffmpeg`. Check the NVIDIA driver with `nvidia-smi` and the CUDA compiler with `nvcc --version`. If the driver works and the CUDA 12.8 toolkit is missing, add NVIDIA's repository for the host distribution and install its `cuda-toolkit-12-8` package. If the toolkit is installed under `/usr/local/cuda-12.8`, set `CUDA_HOME` to that directory and add `$CUDA_HOME/bin` to `PATH`. [Ubuntu Python package](https://packages.ubuntu.com/noble/python3.12-venv) · [NVIDIA CUDA 12.8 installation guide](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/)

From the Linux clone root, create and activate the Python environment:

```bash
git submodule update --init --recursive
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install setuptools wheel ninja numpy opencv-python-headless Pillow scipy torchmetrics==1.8.2 open3d==0.20.0 trimesh==5.1.1
python -m pip install --no-build-isolation 'git+https://github.com/rahul-goel/fused-ssim@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38'
BUILD_NO_CUDA=1 python -m pip install --no-build-isolation -e .
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 python scripts/run_newscan_quality.py ./newscan.zip results/newscan_quality
```

Pass `../newscan.zip` if the archive is one directory above the clone. The destination run directory must be new. To start another fit without rebuilding the dataset, pass `--prepared results/newscan_quality/dataset` and choose a new destination. [PyTorch CUDA wheels](https://pytorch.org/get-started/previous-versions/) · [Open3D 0.20.0 wheels](https://pypi.org/project/open3d/0.20.0/)

Local verification rebuilt the inputs from the archive. The frozen split, camera archive and RGB/pose alignment report matched their archived files byte for byte. The regenerated GLB had a different file hash and reproduced the mesh vertex count, triangle count and aggregate LiDAR agreement within 50 mm to the precision reported here. The first five native frames decoded with the inline FFmpeg filter matched the filter-file decode byte for byte. A 12,000-Gaussian, two-update CUDA smoke run completed training, evaluated all 76 held-out views and exported a PLY verified by readback. In that smoke run, 4,887 Gaussian centers moved more than 1 µm along the normal to their seed triangle, with a maximum normal displacement of 0.278 mm. The [smoke run log](../results/newscan_quality_smoke_20261126/experiment_log.md) contains the test metrics. The 48,000-update run and remote Linux setup are pending.

#### Remote CUDA build failure

The remote trace showed missing `glm/glm.hpp` and `glm/gtc/type_ptr.hpp`, plus missing `cuda/std/optional` and half-precision header errors while invoking `/usr/bin/nvcc` from Miniconda Python 3.13. The traceback does not report the installed toolkit or PyTorch build versions. Restore the pinned GLM files with `git submodule update --init --recursive` from the Git clone. Select the complete CUDA 12.8 toolkit through `CUDA_HOME` and `PATH`, and compare `nvcc --version` with `python -c 'import torch; print(torch.__version__, torch.version.cuda)'` in the intended environment. Prebuild the extension with `MAX_JOBS=2 python -c 'from gsplat.cuda._backend import _C; assert _C is not None'` before retrying. If the failed quality run already prepared the dataset, reuse it with `--prepared results/newscan_quality/dataset` and a fresh run directory. Remote recovery is unverified.

The remote follow-up showed that the GLM header is present and Miniconda already has PyTorch `2.9.1+cu128`. `/usr/local/cuda-12.8/bin/nvcc` does not exist on that host. The next step is to locate a CUDA 12.8 toolkit if it was installed elsewhere or install NVIDIA's `cuda-toolkit-12-8` package for the host distribution, then verify `nvcc` and build the extension in the existing environment. [NVIDIA Linux installation guide](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/index.html#package-manager-installation)

The next transcript shows both Git submodules checked out before the retry. The last Ninja invocation fails with exit code 127 because `/usr/local/cuda-12.8/bin/nvcc` is absent. The GLM and CUDA header diagnostics in the same paste belong to the earlier compile. Verify that the compiler executable runs before rebuilding the extension or restarting training.

The remote Ubuntu Noble `apt update` then failed because `ppa:octave/stable` returned HTTP 404 for its Release file. Remove that repository entry with `sudo add-apt-repository --remove --no-update ppa:octave/stable`, refresh APT, and retry installing `cuda-toolkit-12-8`. Verify the compiler before attempting the CUDA extension build. Installation remains unverified. [Ubuntu repository command](https://manpages.ubuntu.com/manpages/noble/man1/add-apt-repository.1.html)

The remote follow-up confirms that `nvcc` now runs and reports CUDA 12.8.93. The gsplat extension build started with `MAX_JOBS=2` and added `/usr/local/cuda-12.8/targets/x86_64-linux/include` to its include paths. The transcript ends during compilation. The extension import and quality training remain unverified.

For the higher-capacity remote machine, unset `MAX_JOBS` for future builds and let Ninja select the worker count. The quality launcher now accepts `--growth-headroom-mib 0`, which disables the trainer's CUDA memory cutoff for densification. It logs the setting, and local verification confirmed that the argument reaches the A05-Abs training command. The remote launch can omit the shell-level OpenMP and OpenBLAS overrides. The native-resolution 48,000-update remote run with the cutoff disabled is unverified.

#### 2026-11-26 - A05-Abs runner for the published DL3DV benchmark

Added [the DL3DV runner](../scripts/run_dl3dv_a05.py) to download the published benchmark scene inputs at a pinned dataset revision and use the authors' pinned 64-input and held-out frame lists. All 140 official scene IDs exist in the dataset catalog. The runner defaults to one scene and accepts `--all` for the full benchmark. It reads the Nerfstudio images and camera metadata, applies the released preprocessing conventions at 960 × 544, and triangulates Gaussian seed points exclusively from the 64 input RGB views. A05-Abs uses its existing RGB loss, optimizer and refinement strategy. It scores the final checkpoint on the published target views, exports a verified PLY and reports per-view and per-scene PSNR, SSIM and VGG LPIPS. The aggregate gives equal weight to each completed scene.

A CUDA smoke test used a synthetic posed scene with the first official scene's 64 input and 40 held-out indices and completed initialization, two training updates, per-view scoring and PLY readback. The official download returned HTTP 401 without dataset authorization. No official-scene quality score was produced. [Released view split](https://github.com/NVlabs/DVSM/blob/826be87c1a7c8ab3c0385b66b0ceaf8aa71f123b/configs/test_views_sampling/dl3dv_fold8_kmeans64.json) · [DL3DV benchmark data](https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark)

#### DL3DV refinement through the full fit

The DL3DV runner now schedules A05-Abs splitting and pruning through the final update by default. The Gaussian count has no fixed cap. The existing optional CUDA headroom guard can still end refinement early. Use `--refine-stop-iter 16000` to reproduce the original window. The configured cutoff is recorded and checked when reusing or resuming an output. The original schedule reserves later updates for fitting a fixed set of splats and bounds how long splitting, pruning, and opacity resets remain active.

Verified the default 48,000-step and explicit 16,000-step settings, and completed a two-update CUDA run with 40 scored synthetic targets and verified PLY export. In a separate matched 750-update synthetic fit, the full schedule recorded growth events at steps 600 and 700, while the arm stopping at 600 recorded none. No full DL3DV quality result has been measured under the changed schedule.
