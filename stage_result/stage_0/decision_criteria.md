# Outcomes, practical targets, and prospective decision rules

**Declared:** 10 October 2026. **Version:** stage0-v1. These rules precede CCD-GS experiments. They provide a development decision policy, not a publication guarantee or a statistical preregistration with known power. Stage 1 measures feasibility/replay error; any subsequent changes are dated before the affected comparisons.

## 1. Primary comparison and outcome

The primary practical outcome is quality versus representation size. Compare CCD-GS with **ordinary density control plus the same real camera-pose optimizer**, then require the result to survive the stronger extra-pose-compute and retained-correction controls. A gain against vanilla fixed cameras alone does not satisfy the central question.

Use per-view RGB PSNR, SSIM, and LPIPS, averaged over development views within each scene. Report each scene and condition before aggregating. PSNR is the main scalar quality measure; SSIM/LPIPS are guardrails against a trade-off hidden by PSNR. Record exact LPIPS network/weights and SSIM implementation in Stage 1. Use the same uncropped evaluation region and camera policy for every internal method. Color-corrected metrics are secondary if ever enabled.

For each condition and seed, log final Gaussian count and quality across an exploratory budget grid. Compare curves and discrete operating points. Do not interpolate an unmeasured target quality and present that as an observed compactness result. A scene where the count cap prevents ordinary cloning requires an explicit common policy.

## 2. Initial practical target and “comparable”

At a declared reference operating point, development quality is comparable if all three hold:

```text
PSNR_CCD - PSNR_reference >= -0.20 dB
SSIM_CCD - SSIM_reference >= -0.005
LPIPS_CCD - LPIPS_reference <= +0.010
```

Initial compactness aspiration:

```text
N_final_CCD / N_final_reference <= 0.80
```

These are engineering margins chosen now. They are not predicted results. With PSNR based on the same MSE aggregation, 0.20 dB permits roughly 4.7% larger MSE; a mean of per-image PSNR is not exactly the PSNR of pooled mean MSE, so log the aggregation precisely. A single good average cannot conceal a severely degraded scene or region.

Check the target against the pose-enabled ordinary reference and report whether extra-pose/retained-correction controls meet it too. If they do, compactness alone cannot establish a counterfactual advantage. Report peak count, count trajectory, model bytes, time, and memory independently. A smaller final count after costly late pruning does not establish lower training cost.

For later repeated-seed decisions, “comparable” requires the mean differences to satisfy these margins and the one-sided paired uncertainty bounds to avoid crossing the adverse margin where the study has adequate repeat data. With only three seeds, uncertainty is weak; report that limitation and do not claim formal noninferiority from an underpowered pilot. Show individual runs.

## 3. Exploratory configurations, selected before outcomes

| Setting | Analytic fixture | Room development |
|---|---|---|
| Resolution | 128×128 | 389×259 |
| Seeds for decisive pilot comparisons | 0, 1, 2 | 0, 1, 2 |
| Proposed end-to-end step horizon | 3,000 initially | 7,000 initially |
| Candidate budget caps | 2,560; 5,120; 10,240 | 200,000; 300,000; 500,000 |
| Candidate checkpoints | 25%, 50%, 75% of the active split window | Same fractions |
| Utility samples | 24 candidates per checkpoint initially | Same, adjusted only from measured branch cost |
| Short continuation | 200 steps | 200 steps |
| Longer validation | 1,000 steps on a prespecified subset | Same |

These values allocate initial effort; they are not final-paper frozen settings. Use `DefaultStrategy` with a documented common cap/selection wrapper later. Stock `DefaultStrategy` does not enforce these caps merely because a JSON records them. Stage 1 fixes initial learning rates/density schedules and records their exact values; Stage 3 determines branch cost before expanding candidates. Change a horizon/budget for feasibility using a dated decision, never silently after selecting successful runs.

## 4. Controlled training-pose disturbances

Use clean, mild, and moderate conditions for both fixture families, with synthetic camera truth available only for the analytic fixture. Proposed mild bound: 0.5-degree rotation and translation radius 0.0025 scene scales. Proposed moderate bound: 2 degrees and translation radius 0.01 scene scales. These are starting artificial disturbances, not assertions about typical real calibration errors.

For each training view, sample independent isotropic rotation and translation directions from a fixed seed. Sample their magnitudes uniformly from zero to the declared bounds and apply a right-composed transform in the recorded convention. Rotation and translation have separate distributions/seeds and recorded realized values. Reuse directions when scaling mild/moderate severity to permit matched comparisons. Perturb training cameras only; hold development/test cameras and the initial point cloud fixed. Save every transform before a run.

The temporary probe starts from each method's actual current camera state. Probe limits do not automatically cover the whole moderate disturbance. A local failure under a larger error is informative; do not tune corruption to favor the method. Stage 1 checks projected overlap and baseline trainability on development data, then freezes the chosen severity conditions.

## 5. H1 diagnostic gate

First measure no-intervention replay and zero-update probe variation on identical checkpoint/candidate/view tuples. A zero-update probe should match original gradients within the measured numerical tolerance and should not advance training state. State mutation is an engineering failure, not a weak H1 result.

Proposed pilot evidence: reproducible rank displacement or top-quartile membership change exceeding five times the corresponding replay/zero-update variation, observed at two checkpoints and repeated on a fixed subset. Report Spearman/Kendall agreement between repeated diagnostic measurements, top-k overlap, gradient changes, and valid-observation rates. Evaluate stable-visibility cases separately.

If the numerical floor is effectively zero, any tiny rank swap is still not a practical result. Report its magnitude, near-tie prevalence, and top-k effect; H2 decides whether the changed ranking matters. Visibility loss, invalid probes, or radius changes may explain ranking changes and cannot be counted as pose-error removal.

## 6. H2 split-utility gate

Sample candidates using a recorded rule spanning original gradient, scale, opacity, and observation count. Include high-gradient and low/medium-gradient strata; match pose-sensitive/persistent candidates approximately on these covariates. Save the candidate IDs before branch outcomes are known. Utility is the mean development RGB MSE improvement of the split branch after matched continuation.

Primary pilot ranking statistic: mean utility of the top quartile (`k=6` for 24 candidates), evaluated on the same labeled candidate set for every ranking. Supplement with Spearman rank correlation, useful-split fraction above noise, and the full utility distribution. Do not select `k` from favorable plots.

Define normalized excess top-k utility over a simple comparator as:

```text
D = (mean(U_topk_CCD) - mean(U_topk_comparator))
    / mean(development_MSE_no_split for the shared candidate set)
```

Proposed practical signal: `D >= 0.005` and the raw utility difference exceeds five times the 95th-percentile absolute no-intervention branch-loss difference measured on matched replays. This is a 0.5% development-MSE effect criterion, not a p-value. Require directional advantage over both `g0` and `gplus` at two checkpoints, with a second fixture check and no systematic long-horizon reversal. Analyze uncertainty and attribute-stratified behavior; mixed evidence is inconclusive, not a pass.

Evaluate any fitted before/after score on candidate batches/checkpoints not used to fit it. Use checkpoint-clustered descriptive intervals in a pilot; candidates from one scene are correlated. Do not present thousands of candidates as thousands of independent scene replications. Report negative utilities, failures, insufficient visibility, and the number of candidates with reliable labels.

## 7. H3 allocation gate and resource matching

A pilot supports continuing the allocation-method framing only if the mechanism-validated score yields a repeatable quality/count improvement after ordinary pose refinement and the improvement remains against retained-correction and equal-compute extra-pose controls. Check at more than one budget and on clean/mild/moderate conditions. Do not require the 20% aspiration to declare a scientifically interesting mechanism; do not relax quality margins post hoc to advertise hitting it.

Equal-compute control gets the same measured time allowance for extra pose work as the diagnostic's total camera fitting/render/backward/scoring work, with the same hardware and warmup policy. Record actual time rather than assume five optimizer calls equal five probe operations. Both sides retain the common Gaussian-update policy. In equal-total-time comparisons, the CCD-GS arm may complete fewer ordinary iterations; report completed iterations and use explicit stop rules.

Proposed practical overhead target: probing adds no more than 20% to the pose-enabled ordinary training time at the initial frequency. This is a development aspiration. Exceeding it can still support a compactness study, but weakens efficiency framing and requires an explicit quality/count benefit. Benchmarking faster rendering requires separate warmed-up synchronized render measurements; count reduction is not that measurement.

## 8. Decision table

| Evidence at the early gate | Decision |
|---|---|
| H1 reproducible; H2 above simple signals/noise; H3 survives strong calibration controls | Continue to integrating the simplest score |
| H1 yes; `gplus` explains H2 | Simplify; reassess distinction from accepted correction |
| Utility ranking improves; extra/retained pose updates explain allocation gains | Reframe as calibration/scheduling research |
| Reliable mechanism result but large overhead | Narrow to compactness/diagnostic value; profile before efficiency wording |
| Mixed effects or weak replay reliability | One focused measurement revision/extension, with a dated question |
| No useful predictive information after reliable measurement and one focused revision | Stop expanding CCD-GS; preserve the negative study |

The pilot is a risk gate, not multi-scene publication evidence. A later final campaign must freeze method, bounds, score, budgets, camera policy, metrics, seeds, and comparison rules before reading untouched final test results.

## 9. Claim ledger

| Future sentence | Minimum supporting evidence | Current status |
|---|---|---|
| The probe measures pose sensitivity of growth signals | H1 plus state-isolation/visibility checks | Unmeasured |
| The signal predicts useful splits | H2 beyond original/corrected magnitude and covariates | Unmeasured |
| The method improves compactness | H3 quality/count curves with strong controls | Unmeasured |
| The method speeds training | Equal-total-time or synchronized full-training timing with all overhead | Unmeasured |
| The method speeds rendering | Independent fixed-hardware render benchmark | Unmeasured |
| The method improves camera accuracy | Synthetic pose truth and fixed gauge policy | Unmeasured |
| The method generalizes | Fresh scenes/families and repeated decisive runs | Unmeasured |

No contribution sentence from these rows is established by Stage 0.
