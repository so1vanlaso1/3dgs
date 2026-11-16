# A05 alternative using absolute gradients

## Research decision and frozen protocol

The first residual fallback improved two scenes in a single-seed training screen, but lost to equal-count gradient selection in the frozen-checkpoint intervention. Its selected parents were substantially larger. This motivates a different selection signal, addressing cancellation in positional gradients before the eligibility decision.

AbsGS identifies gradient cancellation as a reason that Gaussians covering detailed regions can fail to split. The local renderer already computes per-pixel absolute positional gradients and `BudgetStrategy` inherits their accumulation from `DefaultStrategy`. The trainer passes the strategy flag into rasterization. This experiment combines that existing signal with A04's budget, attributed-error and edge ranking, and Long-Axis Split. [AbsGS](https://arxiv.org/abs/2404.10484)

The candidate is named `A05_absgrad`. Use `absgrad=True` and `grow_grad2d=0.0008`, the value suggested in the local strategy documentation. Disable residual fallback. Use 7,000 updates, 700k cap, seed 42, the existing independent camera sampler, and the original Room, Counter and Kitchen images and evaluation split. Compare with archived A04 and the first A05. The threshold is fixed before seeing these results. Absolute gradient and threshold change together, so any gain is attributable to the combined policy. Evaluation views have already been exposed during development.

The absolute-gradient accumulator feeds both the eligibility threshold and the gradient component of A04's ranking. The attributed-error and edge components still use the same observations and weights. This is an AbsGS-inspired combination with the existing A04 pipeline.

A practical pass requires at least +0.05 dB PSNR versus A04, SSIM loss no greater than 0.001 and LPIPS increase no greater than 0.001. Record final Gaussian counts, peak tensor allocation, elapsed time, initialization hashes and camera-sequence hashes. Validate finite checkpoints and each growth-event cap. A 60-update smoke and CUDA cancellation tests precede full training. If all three scenes pass, repeat the frozen candidate at seed 43 against the corresponding archived A04 runs.

Also reviewed Recovery-Aware Pruning and late multi-view updates from the paper used by A04. Its pruning rule removes the lowest-opacity 20% at steps 3300 and 6300, while its multi-view updates follow a 30k-step schedule. Those interventions require schedule adaptations in this 7k pipeline. The absolute-gradient path offers an existing implementation and a concrete threshold for this trial. [Improving Densification](https://arxiv.org/html/2508.12313v1)

## Execution

All 28 budget and Long-Axis Split tests passed. The two new CUDA cases use a centered symmetric Gaussian, where the signed positional gradient falls below the growth threshold while the absolute gradient exceeds it. They exercise the packed and dense rendering paths through the budget strategy. The test run emitted the existing unknown `env` configuration warning.

The 60-update native smoke completed with 118,157 Gaussians, adding 5,530 through two Long-Axis Split events. Configuration, finite checkpoint and cap checks passed. The campaign stores source snapshots, commands and verified records. `scripts/audit_a05.py` checks the saved configuration, checkpoint and PLY counts, identical evaluation targets, and per-view PNG PSNR. PNG metrics are an independent quantized audit of the trainer's float metrics.

## Seed-42 results

All three full runs completed 7,000 updates and passed the frozen practical gate. Deltas are the absolute-gradient candidate minus the archived A04 run. Positive PSNR and SSIM are improvements. Negative LPIPS is an improvement.

| Scene | A04 PSNR | A05-Abs PSNR | Delta PSNR, dB | Delta SSIM | Delta LPIPS | A04 Gaussians | A05-Abs Gaussians | Count reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Room | 30.60068 | 30.73701 | +0.13633 | +0.003950 | -0.010731 | 597,975 | 452,485 | 24.33% |
| Counter | 27.74928 | 27.83983 | +0.09056 | -0.000634 | -0.001025 | 563,073 | 381,228 | 32.30% |
| Kitchen | 29.98970 | 30.21228 | +0.22258 | -0.000247 | -0.000454 | 695,179 | 502,536 | 27.71% |

Mean PSNR gain across the three scenes was +0.14982 dB. Counter and Kitchen had small SSIM declines within the specified tolerance. Relative to the first residual A05, PSNR improved by +0.13007, +0.03834 and +0.09521 dB for Room, Counter and Kitchen, with fewer Gaussians in all three. Room improved all quality metrics relative to that first variant. Counter and Kitchen traded a small SSIM decline for higher PSNR, lower LPIPS and lower count. [Float metrics](../results/a05_absgrad_20261107/summary.json)

All initial and complete camera-sequence hashes matched the paired controls. Initial metrics matched within 0.0000001, all final checkpoints were finite, and all 54 growth events per scene obeyed the scheduled cap and count arithmetic. The PLY vertex counts matched the recorded counts. All nine training source files still matched their captured hashes after the campaign. The image audit checked 104 identical ground-truth PNG halves. Quantized PSNR improved on 30/39 Room, 23/30 Counter and 21/35 Kitchen views. [Artifact and image audit](../results/a05_absgrad_20261107/render_and_protocol_audit.json)

| Scene | A04 peak tensors, GiB | A05-Abs peak tensors, GiB | A04 training timer, s | A05-Abs training timer, s | A05-Abs wall time, s |
|---|---:|---:|---:|---:|---:|
| Room | 1.080 | 0.971 | 404.7 | 394.8 | 472.4 |
| Counter | 1.090 | 0.945 | 432.4 | 432.8 | 507.7 |
| Kitchen | 1.229 | 1.077 | 478.1 | 469.8 | 547.2 |

Memory values measure peak tensor allocation. Timing comes from independent workstation runs. The smaller models reduced tensor memory, while observed training time was similar. Every late growth event remained eligibility-limited under the 700k ceiling.

## Seed-43 confirmation

All seed-42 scenes passed, triggering the predeclared confirmation. The identical policy completed against the corresponding archived A04 seed-43 runs.

| Scene | A04 PSNR | A05-Abs PSNR | Delta PSNR, dB | Delta SSIM | Delta LPIPS | A04 Gaussians | A05-Abs Gaussians | Count reduction | Practical pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| Room | 30.72016 | 30.81184 | +0.09168 | +0.003616 | -0.010074 | 593,800 | 442,656 | 25.45% | Yes |
| Counter | 27.73248 | 27.77439 | +0.04191 | -0.000633 | -0.000017 | 571,091 | 384,975 | 32.59% | No |
| Kitchen | 29.79324 | 30.05212 | +0.25887 | +0.000158 | -0.000131 | 699,121 | 508,135 | 27.32% | Yes |

PSNR increased and Gaussian counts fell in all six scene-seed comparisons. Five of six passed the predeclared gate. Counter seed 43 missed its +0.05 dB PSNR requirement by 0.00809 dB, though its SSIM and LPIPS met the guardrails. Counter's SSIM declined by approximately 0.00063 in both seeds. The three-scene mean paired PSNR gains were +0.14982 dB for seed 42 and +0.13082 dB for seed 43.

The second audit verified the saved configuration, source snapshots, checkpoint hashes, PLY counts, matching initialization and camera-sequence hashes, and all 104 matching target images. Quantized PSNR improved on 27/39 Room, 20/30 Counter and 24/35 Kitchen views. These are previously exposed development views, and the two seeds are an initial confirmation rather than a broad generalization test. The absolute gradient and its threshold changed together, and this comparison does not isolate their separate contributions. [Confirmation metrics](../results/a05_absgrad_s43_20261107/summary.json) · [Confirmation audit](../results/a05_absgrad_s43_20261107/render_and_protocol_audit.json)

## Reproduction

```powershell
.\scripts\run-research.ps1 -Mode test
.\scripts\run-research.ps1 -Mode a05-train -Variant absgrad -Smoke -OutputRoot results/a05_absgrad_smoke_20261107
.\scripts\run-research.ps1 -Mode a05-train -Variant absgrad -OutputRoot results/a05_absgrad_20261107
.\scripts\run-research.ps1 -Mode a05-train -Variant absgrad -Seed 43 -OutputRoot results/a05_absgrad_s43_20261107
.\.venv\Scripts\python.exe scripts/audit_a05.py --root results/a05_absgrad_20261107
.\.venv\Scripts\python.exe scripts/audit_a05.py --root results/a05_absgrad_s43_20261107
```

Choose fresh output directories when rerunning training.
