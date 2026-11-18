# A05 experiment — residual-driven split eligibility

**Date** 7 November 2026 (Asia/Saigon)\
**Scope** Room, Counter and Kitchen from Mip-NeRF 360. Room and Counter had the largest mean spare capacity in the A04 audit. Kitchen was close to the cap and is a contrasting case. [Prior audit and proposal](A05_RESEARCH_PLAN_20261107.md)

## Research question and implementation

A04's error and edge scores affect the ranking after the gradient and opacity gates. A05 asks whether residual evidence can select useful below-threshold parents for slots left by ordinary A04 growth. The rule uses the existing top-four contributor query and mean contribution-weighted L1 residual per contributing training view. A parent must be opaque enough for ordinary growth and have positive finite attributed error in aggregate with support from at least three distinct training images. Repeated samples of the same image count once in each fallback observation window. The gradient control uses the same supported pool and replaces the residual ranking with positional-gradient ranking.

Ordinary growth retains priority. The opt-in route starts at step 5000 and adds up to 512 parents at an event, bounded by the spare slots under the existing scheduled target. Long-Axis Split, opacity, pruning, optimizer updates, and the A04 ranking are shared in the tested configurations. With the fallback off, the trainer uses the original A04 rules. The growth history records the fallback pool size, camera count and actual fallback additions. [Budget strategy](../gsplat/strategy/budget.py) · [Trainer](../examples/simple_trainer.py)

The use of attributed error builds on Revising Densification, which ranks using the maximum attributed error across views. FastGS uses counts of high-error pixels across views for density control. A05 tests a narrower change to the eligibility rule of a fixed-budget A04 pipeline. [Revising Densification](https://arxiv.org/html/2404.06109v1) · [FastGS](https://arxiv.org/html/2511.04283v3)

## Actual protocols

**Warm-start split-utility test.** The archived A04 seed-42 checkpoints contain final splat weights but lack a resumable optimizer and sampler state at the proposed 5000/5800 checkpoints. The test instead forks one final verified checkpoint per scene. Score collection uses 100 training views at frozen weights and the same positional gradient normalization as the trainer. The contributor query runs every ten views. Each scene has a no-extra-split control, a replay of that control, and two branches splitting the same number of parents once, by residual or by gradient. All branches start with fresh Adam, the final A04 position learning rate held constant, and the same continuation cameras. Other learning rates match the training defaults. They run 500 updates without further structural changes. Metrics at updates 0, 100 and 500 use the same full-resolution evaluation split. These are exposed development views. This is a test of one split intervention at converged weights.

**End-to-end screen.** The normal trainer runs A05 from SfM with the 7,000-update seed-42 A04 recipe, a 700k cap, refinement through step 5900 and the fallback active for steps 5000–5900. Archived A04 seed-42 runs serve as paired controls, with exact initialization and full camera-sequence hash checks. Each method has one run per scene. Final Gaussian counts can differ under the common cap. This screen tests the combined late-growth policy. The equal-count gradient comparison comes from the warm-start experiment.

The warm-start advancement threshold is +0.05 dB PSNR over the equal-count gradient branch, a positive PSNR difference versus no extra splits beyond control replay variation, and SSIM/LPIPS guardrails of 0.001 against both. The end-to-end screen uses +0.05 dB and the same quality guardrails versus A04 as practical effect sizes.

## Warm-start results

All three scenes completed all four branches. Each split arm added exactly 512 live Gaussians. Values are means over the original evaluation views after 500 continuation updates. Delta columns compare A05 with the no-extra-split continuation.

| Scene | No extra splits PSNR | Gradient PSNR | A05 PSNR | Δ vs control, dB | Δ vs gradient, dB | Δ SSIM | Δ LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Room | 30.67601 | 30.67154 | 30.66515 | −0.01085 | −0.00638 | −0.000025 | +0.000057 |
| Counter | 27.73555 | 27.73173 | 27.71918 | −0.01637 | −0.01255 | −0.000119 | +0.000141 |
| Kitchen | 30.10578 | 30.10120 | 30.07184 | −0.03395 | −0.02937 | −0.000086 | +0.000177 |

A05 missed the practical gate on all three scenes. The maximum absolute control-replay PSNR difference was 0.000030 dB. The residual and gradient top-512 lists shared only 17, 11 and 16 parents for Room, Counter and Kitchen. The eligible fallback pools were 91,483, 159,933 and 325,781, respectively. The policy therefore exercised a genuinely different selection among plentiful candidates.

The immediate mean A05 loss versus no extra splits was −0.075, −0.111 and −0.211 dB. By update 500, most of the initial loss had recovered, with 8/39 Room, 4/30 Counter and 5/35 Kitchen views improving. The residual branch selected parents with a median attributed error approximately 35×, 27× and 23× that of the gradient branch in the respective scenes. At these trained weights, high attributed RGB error did not identify splits that improved average held-out rendering under this intervention. The test covers one seed and one final checkpoint per scene, with fresh optimizer state. [Summary and per-view records](../results/a05_warmstart_20261107/)

Matching the selected parent IDs back to the archived weights revealed a scale difference. The median longest-axis standard deviation of A05 parents was 5.8× the gradient control's in Room, 3.9× in Counter and 9.1× in Kitchen. Median opacity in the same order was 0.803, 0.765 and 0.756 for A05 versus 0.802, 0.719 and 0.650 for the gradient control. Summing attributed error across pixels can favor broadly contributing parents. The observed size difference is consistent with that mechanism, while this study does not isolate scale, opacity and residual in the split response.

## End-to-end results

All three A05 runs completed 7,000 updates. Deltas are A05 minus the archived A04 seed-42 run. Positive PSNR and SSIM are gains. Negative LPIPS is a gain.

| Scene | A04 PSNR | A05 PSNR | Δ PSNR, dB | Δ SSIM | Δ LPIPS | A04 → A05 Gaussians | Fallback splits | Practical pass |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Room | 30.60068 | 30.60694 | +0.00626 | +0.000075 | +0.001002 | 597,975 → 605,474 | 5,120 | No |
| Counter | 27.74928 | 27.80150 | +0.05222 | +0.000283 | −0.001005 | 563,073 → 568,883 | 5,120 | Yes |
| Kitchen | 29.98970 | 30.11707 | +0.12737 | +0.000201 | −0.000347 | 695,179 → 699,440 | 4,608 | Yes |

Counter and Kitchen exceeded the practical threshold at seed 42 and improved all three quality metrics. Room's PSNR difference was small and its LPIPS increased. Final Gaussian counts rose by 1.25%, 1.03% and 0.61% for Room, Counter and Kitchen. All runs stayed within the cap at every structural event. The extra split totals and final-count changes differ because normal growth and pruning evolve with the learned scene. [End-to-end metrics](../results/a05_training_20261107/summary.json)

Room and Counter made all ten batches of 512 fallback splits. At Kitchen's last event, 4,162 ordinary parents qualified and filled all 4,008 available slots, leaving zero fallback splits before the usual prune. The 104 saved final image pairs used identical target pixels. Recomputed 8-bit PSNR improved on 20/39 Room, 19/30 Counter and 27/35 Kitchen views. Peak tensor allocation rose by about 0.10 GiB in each scene. Timing was collected under ordinary workstation load. The per-view audit uses PNG precision, while the table uses the trainer's float-render metrics. [Render and protocol audit](../results/a05_training_20261107/render_and_protocol_audit.json)

The archived controls matched A05 on initialization and the full 7,000-view training sequences by hash. The initial render metrics agreed within 0.00001. Counts had already diverged slightly before the fallback began, by +752, +647 and +816 after the step-4900 event. The experiment compares independent runs at a common cap, with one seed per scene. The negative equal-count warm-start test leaves the cause of the Counter and Kitchen gains unresolved. Retain A04 as the reference and A05 as an opt-in experiment until the end-to-end gains are confirmed with repeated seeds and an equal-quota gradient arm.

## Validation and reproduction

The direct Windows invocation lacked the repository's CUDA toolkit environment. It passed 24 CPU tests and failed two GPU tests before the renderer could run. The repository launcher sets the local CUDA, MSVC and extension-cache paths. With that setup, all 26 budget and Long-Axis Split tests passed, including the packed rendering and split backward tests. The Room warm-start smoke completed all four branches with 512 extra splits per intervention. A 60-update normal-trainer smoke added 16 fallback splits in two events and verified the count, cap and final checkpoint. All three full A05 checkpoints were finite and their PLY vertex counts matched the verified final counts. Smoke metrics are harness checks.

```powershell
.\scripts\run-research.ps1 -Mode test
.\scripts\run-research.ps1 -Mode a05 -Smoke -OutputRoot results/a05_warmstart_smoke_new
.\scripts\run-research.ps1 -Mode a05 -OutputRoot results/a05_warmstart_new
.\scripts\run-research.ps1 -Mode a05-train -Smoke -OutputRoot results/a05_trainer_smoke_new
.\scripts\run-research.ps1 -Mode a05-train -OutputRoot results/a05_training_new
```

Use fresh output directories for new runs. The runner saves checkpoints or checkpoint identity as appropriate, exact commands or camera orders, per-view warm-start metrics, parent IDs, source hashes, and results. [Experiment runner](../scripts/research_a05.py)
