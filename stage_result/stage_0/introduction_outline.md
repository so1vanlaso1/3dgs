# Provisional introduction outline and working prose

**Date:** 10 October 2026. **Status:** argument draft before implementation; all proposed findings are conditional. No performance numbers or contribution claims have been established.

## 1. Suggested working title

**Do Temporary Pose Probes Predict Useful Gaussian Splits?**

“Calibration-Counterfactual Densification” remains a project name. An eventual method title should reflect the actual validated result. If permanent pose refinement explains the gains, a calibration/scheduling title is more accurate. If the study is negative, use an empirical-diagnostic title.

## 2. Paragraph-by-paragraph argument

| Paragraph | Job | Content to establish | Evidence/source | Wording limit |
|---|---|---|---|---|
| 1 | Introduce allocation | Gaussian growth determines representation size and detail; count changes affect resource use | Repository strategy; verified allocation literature in the matrix | No claim that count determines training/render speed by itself |
| 2 | Explain ambiguity | Image discrepancies can reflect camera error, scene error, appearance, visibility, or unfinished optimization | Hypothesis and controlled failure examples | Avoid saying every pose-sensitive gradient is unnecessary growth |
| 3 | Locate overlap | Existing error/consistency/conflict/predictive allocation and permanent pose refinement already address portions of this problem | [Related-work matrix](related_work_matrix.md) | No first camera-aware/multi-view/utility-aware claim |
| 4 | State the narrow question | Does response to a bounded copied-pose fit add information about later split utility? | Proposed specification | Explain that the correction is discarded in the diagnostic arm |
| 5 | Explain the essential validation | Matched split/no-split branches estimate conditional utility; controls isolate extra calibration work | Future H1/H2/H3 experiments | No causal calibration-error decomposition from a gradient change |
| 6 | Describe measured scope | Quality/count is primary; real/synthetic coverage, time/memory, failures, and uncertainty | Actual final evidence, when available | Write results only after Stage 7 |

## 3. Provisional gap paragraph

Existing density-control rules allocate primitives using image error, multi-view consistency, gradient conflict, or learned utility-related signals, while camera-refinement methods permanently adjust the reconstruction's calibration state. We investigate a narrower measurement: how an eligible Gaussian's growth signal responds to a bounded extrinsic-pose fit performed on independent camera state with the current scene values fixed. The question is whether this response predicts the benefit of splitting after subsequent optimization, beyond the original gradient, the corrected gradient alone, and the same computation spent on retained pose corrections. Its distinction must be established by split/no-split utility and allocation controls, rather than by camera sensitivity alone. The relevant overlap and version-specific qualifications are documented in the [verified matrix](related_work_matrix.md).

## 4. Working introduction prose

Adaptive growth is a consequential decision in Gaussian scene reconstruction. Adding capacity can recover detail, but the image discrepancies used to trigger growth do not uniquely identify a shortage of primitives. A discrepancy can also reflect inaccurate camera extrinsics, unfavorable visibility, appearance mismatch, or incomplete optimization of the current representation. A growth rule that responds to those effects may spend scene capacity without delivering a corresponding improvement on new views.

Camera error presents a particularly testable part of this ambiguity. Small changes to an image's pose alter projected Gaussian means and the loss gradients used in density control. It is therefore plausible that a temporary pose fit changes which candidates appear to deserve growth. The response is not automatically an explanation of the error: moving the camera can compensate for incorrect geometry, or change which primitive contributes to a pixel. Likewise, an error that survives the correction may still be resolved by optimizing existing primitives.

Our proposed investigation measures this response while keeping the actual training state fixed. For an eligible split candidate, we measure the normalized screen-space growth signal at the current pose, fit a bounded copied camera using a training image, and measure the same signal again. The diagnostic variant discards the camera correction. It asks whether the two measurements provide useful information for allocating later splits.

This question requires more than demonstrating that gradients decrease. We will estimate split utility with matched training branches: one branch performs an ordinary fixed split, the other omits it, and both continue from the same scene, camera, optimizer, and sampling state. The resulting development-view difference tests whether the diagnostic ranks beneficial structural changes. We will then compare allocation under common representation allowances and measured computation budgets, including controls that retain the correction or spend an equivalent allowance on ordinary camera refinement.

The practical target is improved reconstruction quality versus representation size. Training and rendering time remain separate outcomes because repeated probing adds computation and Gaussian count does not fully determine either cost. The investigation begins with a known-camera synthetic fixture and an official low-resolution room scene, before any larger final benchmark campaign. All performance statements remain provisional until mechanism and end-to-end comparisons are measured.

The opening background assertions above are the study's motivation. When assembling the manuscript, attach the verified papers in [the matrix](related_work_matrix.md) to the specific allocation/calibration claims and refresh the search; do not imply this prose itself establishes a new literature result.

## 5. Conditional contribution list to revisit after experiments

1. **If H1 passes:** a reproducible measurement of split-growth sensitivity to constrained extrinsic-pose interventions, with visibility and state-isolation limits.
2. **If H2 passes:** evidence that a specified response-based signal improves conditional split-utility prediction beyond original/corrected gradient baselines.
3. **If H3 passes across fresh scenes:** an allocation rule that improves the measured quality/count trade-off beyond strong permanent-calibration and growth controls, with its full overhead reported.

If only the first holds, present a diagnostic study. If only retained correction helps, present calibration/scheduling evidence. If none hold after reliable measurement, write the negative result. Do not list all three as unconditional contributions in the abstract.

## 6. Evidence slots for the future manuscript

- H1 figure: repeated original/corrected gradients, zero-update noise, rank changes, and visibility exclusions.
- H2 figure: top-k development split utility and utility distributions against original/corrected/random rankings, with horizon sensitivity.
- H3 figure: quality/count curves and quality/total-time curves against strong camera/allocation controls.
- Failure panel: a geometry-compensating pose fit, a changed-visibility observation, and a useful split incorrectly suppressed.
- Resource table: probe seconds, training seconds, final/peak counts, model bytes, and memory, with hardware and accounting policy.

No figure slot is filled with invented measurements. The [training-only fixture preview](evidence/synthetic_train_preview.png) documents data preparation, not a method result.
