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
