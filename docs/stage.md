# CCD-GS: staged research and paper plan

Created: 10 October 2026. Based on [final.md](final.md), with [other_direction.md](other_direction.md) as background.

This is a plan for investigating Calibration-Counterfactual Densification (CCD-GS), not a report of completed experiments. Start with camera extrinsics only. Publication depends on the evidence; do not write the proposed benefits as established results.

## 1. The question your paper must answer

**Does pose-intervention scoring identify useful Gaussian growth better than spending the same resources on camera refinement and ordinary densification?**

Your proposed contribution is a bounded temporary pose probe that helps rank densification candidates. The primary practical claim to test is improved reconstruction quality versus Gaussian count. Training speed and rendering speed are separate measured outcomes, because probing adds work.

Test three hypotheses:

- **H1: diagnostic information.** A bounded pose probe changes candidate rankings reproducibly, beyond numerical variation.
- **H2: split usefulness.** The score predicts the benefit of splitting better than the original gradient and corrected-gradient magnitude alone.
- **H3: allocation value.** Using the score improves quality versus representation size beyond strong camera-refinement and allocation controls, with measured computation costs.

H1 alone is insufficient. H2 is mandatory for the mechanism claim. H3 is necessary for presenting CCD-GS as a useful reconstruction method.

## 2. Schedule and decision points

Assume about 12 weeks of consistent work for a prototype and initial paper draft. Reserve another 2–4 weeks if baseline reproduction, cloud access, or repeated experiments take longer. These are effort estimates, not a submission deadline.

| Stage | Suggested time | What you should achieve | What you should write |
|---|---|---|---|
| 0 | Week 1, days 1–2 | Fix the question, scope, and closest-work comparison | Research brief and related-work matrix |
| 1 | Week 1, days 3–5 | Reproducible baselines and trustworthy branch snapshots | Experimental protocol and baseline record |
| 2 | Week 2, days 1–3 | Isolated bounded pose probe and candidate measurements | Diagnostic method specification |
| 3 | Week 2, days 4–7 | Small split-utility study and first decision gate | Mechanism results and continue/revise decision |
| 4 | Weeks 3–4 | Integrate the simplest supported score into densification | Method section and algorithm |
| 5 | Weeks 5–6 | Establish whether gains survive essential controls | Pilot results and ablation draft |
| 6 | Weeks 7–9 | Run a frozen multi-scene benchmark campaign | Main tables and reproducibility details |
| 7 | Weeks 10–11 | Analyze uncertainty, failures, and actual claims | Results, discussion, and limitations |
| 8 | Week 12 | Assemble, audit, and revise the full paper | Submission-ready draft and artifact package |

Begin writing in Stage 0. Keep the abstract, contribution claims, and conclusion provisional until Stage 7.

## 3. Stage 0 — define the study before implementing it

**Goal:** turn the assessment in `final.md` into a narrow, falsifiable research specification.

### What you should do

1. Write a one-page brief containing the research question, H1–H3, primary outcome, and major risks: pose–scene ambiguity, visibility changes, optimization versus capacity, and probe overhead.
2. Limit the initial method to temporary extrinsic pose changes and split decisions. Defer exposure, intrinsics, distortion, rolling shutter, new CUDA kernels, and joint split/clone innovations.
3. Read and verify the closest sources named in `final.md`: Revising Densification, FastGS, SalientGS, Energy-GS, BSGS, HarmoGS, and BlitzGS. Verify their actual methods, versions, code availability, and evaluation protocols; the source document is a starting list, not a fresh literature verification.
4. Build a comparison matrix: pose refinement, multi-view scoring, temporary intervention, growth rule, pruning, budgets, and published evidence. Follow the corrected BlitzGS description in `final.md` and check it against the source.
5. Define the narrow gap in one paragraph. Avoid a first-ever claim. Explain what your intervention measures that permanent pose refinement or existing multi-view scores do not already measure.
6. Choose development data: one small synthetic scene with known camera poses and one low-resolution official room scene. Check data access and camera conventions before scheduling runs.
7. Separate training, development, and final test views. Fit probes using training views; use development views for score design and branch evaluation. Record view IDs. Never use final test views to choose the score or parameters.
8. Decide an initial practical target, such as comparable quality with fewer Gaussians. Define what “comparable” means on development data before the final campaign. A target such as 20% fewer Gaussians is an aspiration, not a predicted result or automatic paper acceptance rule.

### What you should write

- `docs/research_brief.md`: question, hypotheses, scope, decision criteria, and known risks.
- `docs/related_work_matrix.md`: verified source links, precise overlap, possible distinction, and verification dates.
- A provisional introduction outline: allocation problem → why pose errors might affect growth → proposed diagnostic → evidence needed.

**Stage complete when:** you can explain the proposed distinction from “refine poses more carefully, then densify normally,” and the data splits and primary comparison are recorded. If direct prior art already covers the mechanism, revise the question before substantial implementation.

## 4. Stage 1 — establish baselines and reliable experimental state

**Goal:** ensure subsequent changes can be attributed to the diagnostic rather than a changed pipeline.

### What you should do

1. Record the Git revision, local modifications, environment, GPU, dataset version, resolution, initialization, seed, loss, optimizer, and densification schedule. Keep unrelated experiments separate.
2. Run ordinary `DefaultStrategy` training on the small fixture, then on the low-resolution room scene. Check checkpoint loading and rendered output, and save quality/count/time/memory observations.
3. Run the same trainer with camera-pose optimization enabled. Keep initialization, training views, losses, and ordinary density control fixed.
4. For the synthetic fixture, create clean and controlled pose-perturbed conditions. Record rotation and translation disturbances separately, normalize translation to scene scale, and perturb training poses only. Keep the initial point cloud fixed for this first isolation study.
5. Use clean plus mild and moderate disturbance levels selected on development data. Save the exact perturbation transforms and random seeds. Do not select only disturbance levels where your method looks good.
6. Implement full experiment snapshots for matched branches. Include Gaussian tensors, actual camera state, all optimizer and scheduler states, densification accumulators, step, candidate IDs, Python/NumPy/PyTorch CPU and CUDA RNG states, and the subsequent view/sample sequence. Include appearance state if enabled.
7. Check restoration by running two no-intervention continuations from the same snapshot. They should agree within a documented numerical tolerance. Record any residual nondeterminism before measuring split effects.
8. Decide how to evaluate cameras consistently. Use the same held-out camera policy for every method; do not optimize final test poses from their test images. On synthetic data, evaluate at known cameras and report pose accuracy separately.

### Repository details to account for

- Existing camera adjustments are in `examples/utils.py` (`CameraOptModule`), and pose configuration/training are in `examples/simple_trainer.py`.
- `gsplat/strategy/default.py` accumulates and normalizes screen-space gradients and selects split/clone candidates. Match its gradient convention in diagnostic comparisons.
- The current trainer's `--ckpt` path is evaluation-only. Its saved scene checkpoint does not include full optimizer, scheduler, RNG, and strategy state. It is not sufficient for controlled training branches; add a separate full-state snapshot/restore path during implementation.
- Start from `DefaultStrategy` to isolate split decisions. Do not introduce an MCMC strategy change at the same time.

### What you should write

- `docs/experiment_protocol.md`: fixed splits, perturbations, camera evaluation policy, metrics, snapshots, and comparison rules.
- Append baseline observations to `docs/EXPERIMENT_LOG.md`; link to configurations, logs, checkpoints, and representative renders.
- Draft the experimental-setup subsection while these details are fresh.

**Stage complete when:** both baselines work, saved artifacts are inspectable, and no-intervention branch replay establishes a reliable comparison. Do not infer CCD-GS performance from the existing room baseline.

## 5. Stage 2 — build the smallest isolated pose diagnostic

**Goal:** measure how growth signals change under plausible pose interventions without changing the actual training state.

### What you should do

1. Sample candidates at several training checkpoints: early, middle, and later in the active densification period. Include high-gradient candidates and a range of scale, opacity, and observation counts.
2. Measure each candidate's original normalized screen-space growth signal, `g0`, in a sampled training view.
3. Copy the camera state into an independent probe. Keep scene parameter values fixed and optimize only the copied camera parameters for a small, fixed number of updates.
4. Bound the intervention using a rotation limit, scene-normalized translation limit, and a pose prior. Specify the coordinate convention and whether bounds constrain the total displacement or each update. Fit using broadly distributed image evidence, not just the candidate's pixels.
5. Measure the corrected signal, `g_plus`, using the same image/loss/gradient normalization. Holding scene values fixed means preventing scene updates; retain the autograd paths needed to measure screen-space gradients.
6. Log `g0`, `g_plus`, relative change, loss change, camera displacement, projected radius, opacity, and before/after visibility. Flag occlusion or visibility changes rather than labeling them as resolved calibration errors.
7. Reject nonfinite, out-of-bounds, or unreliable probes. Do not suppress weakly observed candidates just because they lack valid measurements.
8. Discard the temporary pose correction in the diagnostic variant. Verify that probe passes do not modify real camera values, Gaussian values, optimizer moments, densification accumulators, RNG progression, or the training sample sequence.
9. Repeat measurements with fixed samples and compare score variation with ordinary numerical variation. Include a zero-update probe as a sanity check.
10. Measure the full probe cost. Process views sequentially and release computation graphs to keep the local prototype within available VRAM.

### What you should write

- `docs/probe_spec.md`: inputs, outputs, bounds, fitting loss, number of updates, visibility checks, state isolation, and normalization.
- Draft the first method subsection defining `g0` and `g_plus` as diagnostic signals, not estimates of split benefit.
- Plot original versus corrected gradients and ranking changes for clean and perturbed settings. Show how many candidates were excluded by reliability checks.

**Stage complete when:** measurements are reproducible and state isolation is verified. A gradient reduction is evidence of sensitivity to the probe; it does not establish that a split would be unnecessary.

## 6. Stage 3 — test split usefulness and make the first decision

**Goal:** validate the central assumption before building an elaborate multi-view method.

### What you should do

1. Start with about 20–40 candidates per checkpoint across 2–3 checkpoints, adjusting the sample to measured branch cost. Include calibration-sensitive and persistent candidates matched approximately on original gradient, scale, opacity, and visibility. Save the sampling rule.
2. For each sampled candidate, restore one identical full-state snapshot into two branches. Apply one fixed split operation in branch A and omit it in branch B. Hold other structural changes fixed or disable them in both branches for this diagnostic window.
3. Give both branches the same continuation steps, sampled views, camera policy, and optimizer schedule. Apply the same state-expansion rule to newly created parameters in every split branch. Verify that unaffected optimizer state is retained.
4. Estimate utility from development views excluded from probe fitting. For example, define `utility = development_loss_no_split - development_loss_split`, so positive values indicate a useful split. Also report added Gaussian count and continuation length.
5. Use individual splits first where practical. Small groups are a cheaper alternative, but their utility belongs to the group and cannot be presented as a label for each member.
6. Compare rankings from original gradient, corrected-gradient magnitude alone, a persistence ratio, a magnitude-aware before/after score, and matched random ranking. A ratio alone can rank tiny useless gradients highly.
7. Evaluate rank correlation, top-k utility, and utility distributions. Account for original gradient, scale, opacity, and visibility through matched strata or a simple analysis. Do not treat thousands of observations from one scene as independent scene-level evidence.
8. Repeat a subset with a longer continuation to check whether short-horizon utility predicts later improvement. State the limits caused by candidate interactions and continuation length.
9. Run a small allocation pilot against ordinary pose optimization and a control that retains the same probe corrections and densifies normally. Give an extra-pose-update control an equivalent computation allowance. Include scoring overhead when comparing total time.

### What you should write

- `docs/split_utility_study.md`: sampling, snapshot replay, branch intervention, continuation policy, utility definition, plots, and limitations.
- A preliminary mechanism-results subsection, including negative or inconclusive observations.
- A dated gate decision in `docs/EXPERIMENT_LOG.md`: continue, revise once, or change the contribution.

### Two-week decision gate

| Finding | Action |
|---|---|
| Rankings change reproducibly; score predicts utility beyond simple signals; allocation pilot survives equal-compute pose controls | Continue to Stage 4 |
| Rankings change but corrected-gradient magnitude predicts utility equally well | Simplify to that signal and reassess whether a distinct contribution remains |
| Split utility improves but extra pose updates or retained corrections explain the pilot gain | Investigate calibration/scheduling; do not claim a counterfactual allocation advantage |
| Measurements are unreliable or the pilot is too small to distinguish effects | Extend this gate by up to one focused week; fix the identified measurement problem |
| No useful predictive signal remains after reliable measurement and a focused revision | Stop expanding CCD-GS; write the negative result and reconsider the research direction |

The two-week pilot is an early risk check, not definitive multi-scene proof. Set the practical effect criterion before reading the results, and interpret differences relative to replay noise and uncertainty. One favorable PSNR value is insufficient.

## 7. Stage 4 — integrate the simplest supported allocation rule

**Goal:** convert a validated diagnostic into controlled densification decisions.

### What you should do

1. Choose the simplest score supported by Stage 3. Specify exactly how original magnitude, corrected magnitude, relative change, and reliable observations enter the decision. Freeze the formula before the final benchmark campaign.
2. Add multi-view aggregation only now. Compare one view with a small fixed number of views; use observations reliable before and after correction. Define weights, aggregation, and minimum observation count explicitly.
3. Use the score to rank eligible split candidates under a fixed growth budget. Apply common scale, opacity, pruning, reset, and eligibility rules to all controls. Keep cloning unchanged initially, or evaluate it separately before expanding the claim to all growth.
4. Account for the net count increase of a split, and define how pruning and cloning affect the budget. A final-count cap alone does not match the full count trajectory or peak memory.
5. Define behavior for invalid probes, unseen candidates, and insufficient visibility. Use a recorded fallback rule and reevaluate suppressed candidates later to avoid permanent starvation.
6. Specify the real camera policy: ordinary camera refinement continues in every pose-enabled comparison; the CCD-GS variant discards the probe correction. Test retaining it as a separate variant.
7. Maintain stable candidate identity through split, clone, and pruning operations. Clear or remap score history deliberately; tensor row indices alone are not stable identities.
8. Verify zero-effect behavior when the diagnostic is disabled, correct score/history sizes after structural changes, complete branch restoration, and checkpoint evaluation. Prefer checks of these meaningful state invariants over tests that merely repeat the formula.
9. Profile views per probe, probe update count, probe frequency, memory, and time on the development fixtures. Reject configurations whose apparent benefit depends on excessive extra work.

### What you should write

- `docs/method_spec.md`: score equation, algorithm, state policy, budget accounting, fallback, and complexity.
- The method section and an algorithm box.
- A pipeline figure: ordinary optimization → eligible candidates → bounded pose probe → reliable score → budgeted split selection.

**Stage complete when:** the integrated method runs without state contamination, its decisions can be explained from logs, and its total overhead is measured. Do not add a second research contribution before the basic rule passes the controls.

## 8. Stage 5 — test whether the method survives strong controls

**Goal:** determine whether CCD-GS adds value beyond extra calibration compute, delayed growth, or simply fewer primitives.

### Controls you should implement

Use the same trainer for internal controls so renderer and loss differences do not obscure the intervention.

| Control | What it tests |
|---|---|
| Vanilla gsplat with ordinary density control | Reference workflow |
| Ordinary density control plus camera refinement | Whether enabling pose optimization explains the gain |
| Extra pose-only updates with a matched computation allowance | Whether extra calibration compute explains the gain |
| Retain probe corrections and densify normally | Whether applying the correction is sufficient |
| Corrected-gradient magnitude alone | Whether before/after comparison adds information |
| Multi-view scoring without the probe | Whether multi-view filtering explains the gain |
| Delayed densification | Whether waiting for a stable scene is sufficient |
| Higher threshold or matched random suppression | Whether reducing growth alone explains the gain |
| CCD-GS | Full proposed intervention |

### What you should do

1. Compare several Gaussian budgets on development scenes. Keep common initialization and camera treatment, and compare quality/count curves rather than a single favorable point.
2. Run separate fixed-budget and fixed-total-time comparisons. Count probe rendering, probe optimization, score computation, and allocation work in total training time. Also report optimizer steps; fixed steps do not imply equal compute.
3. Give controls a comparable development tuning allowance. Document their chosen schedules and thresholds before the final campaign.
4. Run clean, mild, and moderate perturbation settings. Report pose accuracy on synthetic scenes, accounting for the coordinate gauge; COLMAP poses in real scenes are reference estimates, not exact truth.
5. Repeat the decisive comparisons with at least three seeds if resources permit. Use matched seeds and saved perturbations. Examine scene-level consistency, not only an aggregate mean.
6. Ablate probe bounds, number of updates, frequency, view count, magnitude versus ratio, visibility filtering, and retained versus discarded corrections. Prioritize those that distinguish the mechanism; avoid a large parameter sweep without a question.
7. Inspect failures: immature scenes, changed occlusion, reflections, weakly observed areas, and camera changes that compensate for incorrect geometry.

### What you should write

- Pilot quality/count and quality/time plots with all essential controls.
- An ablation table and clean-data regression table.
- Draft explanations of which component helps and which controls remove the advantage.

**Stage complete when:** the simplest method shows a repeatable advantage over essential controls, or you have narrowed the claim to what the evidence supports. If the advantage disappears, revisit Stage 3 rather than launch a large benchmark campaign.

## 9. Stage 6 — run the frozen paper benchmark campaign

**Goal:** establish generalization beyond the local development scene.

### What you should do

1. Freeze the method, hyperparameters, splits, metrics, budget grid, and primary comparisons. Record the code revision and final experiment manifest before inspecting final test performance.
2. Plan at least six scenes across two dataset families as an initial scope target, including known-camera synthetic scenes and ordinary real reconstructions. Choose scenes for varied geometry, texture, visibility, and pose sensitivity, not observed method gains. Revise this target explicitly if resources constrain it.
3. Evaluate three settings: synthetic clean/controlled camera errors; real scenes with ordinary estimated poses; real scenes with reproducible training-pose disturbances. Include realistically imperfect initialization later if robustness becomes a central claim.
4. Use the stronger GPU/cloud for larger scenes and competitive baselines. Measure runtime comparisons on the same GPU/software environment. Record device differences; do not compare local timing against cloud timing as a method speedup.
5. Reproduce the closest available published allocation and pose methods selected from Stage 0. FastGS-style multi-view allocation and Energy-GS-style strong pose refinement are priorities to investigate. Establish protocol compatibility and reproduction quality before interpreting a gap.
6. Keep same-trainer internal controls distinct from official external reproductions. Label ports, approximations, unavailable methods, and configuration differences honestly; a simplified port is not automatically the published baseline.
7. Run multiple Gaussian budgets and time allowances. Repeat decisive comparisons with at least three seeds; report any single-seed exploratory results separately.
8. Save per-scene PSNR, SSIM, LPIPS, final and peak Gaussian counts, model size, total training time, probe overhead, peak VRAM, and pose error where meaningful.
9. If claiming faster rendering, measure renderer time separately using a fixed view/resolution/hardware protocol with warmup and synchronization. Gaussian count alone is not rendering-speed evidence.
10. Save complete configurations, logs, checkpoints, evaluation outputs, failure traces, and representative images. If a run fails or exceeds memory, retain the record and report the handling rule.
11. Evaluate final test views only with the frozen protocol. A test-driven method change requires explicit disclosure and a new untouched evaluation set where feasible.

### What you should write

- Main quality-versus-count and quality-versus-total-time figures.
- Per-scene result tables, repeated-seed summaries, and resource tables.
- A benchmark manifest linking every table row to its run and configuration.
- The final experimental-setup section and baseline reproduction notes.

**Stage complete when:** the planned experiment matrix is accounted for, results are traceable, and evidence extends beyond one scene. Missing baselines or dataset families must limit the claim rather than disappear from the report.

## 10. Stage 7 — determine what the paper can actually claim

**Goal:** explain the results and their limits without overstating the mechanism.

### What you should do

1. Revisit H1–H3 separately. State supported, unsupported, or inconclusive, and point to the corresponding evidence.
2. Analyze mechanism and final reconstruction together. Better utility ranking without an end-to-end gain, or a PSNR gain without useful ranking, requires a narrower explanation.
3. Report scene-level means and uncertainty for repeated seeds. Use paired comparisons where appropriate. Do not inflate confidence by treating correlated candidates or views as independent scenes.
4. Compare quality/count trade-offs and actual total time. If the method produces a compact model but takes longer to train, say so directly.
5. Check clean-data regressions, weak-texture regions, visibility failures, pose drift, and suppression starvation. Use matched image crops, error maps, and view settings; include failures alongside successes.
6. State the limits of bounded local probes, imperfect split-utility labels, point-cloud initialization, dataset coverage, and baseline reproduction.
7. Refresh the targeted literature search before settling the novelty wording. Update the closest-work matrix with new or changed sources.
8. Choose the paper framing and venue only after seeing the evidence and checking current venue requirements.

### What you should write

- Results organized around H1–H3, followed by resource costs and failures.
- A discussion of why useful gradients persist, when pose probes mislead, and whether retained corrections explain the gains.
- Limitations and an evidence-to-claim table.

| Evidence obtained | Defensible framing |
|---|---|
| Beats only vanilla training without pose refinement | Initial diagnostic study; insufficient for a strong allocation-method claim |
| Helps disturbed scenes but matches extra pose refinement | Calibration/scheduling investigation; counterfactual advantage unestablished |
| Predicts split utility and improves allocation across scenes | Focused densification-method contribution |
| Also survives strong pose/allocation baselines with manageable overhead | Stronger efficiency contribution, limited to measured settings |
| Reliable negative results | Graduation study or empirical analysis; reconsider a methods-paper framing |

**Stage complete when:** every contribution sentence has an experiment supporting it, and the title matches the observed result. Use “efficient” only with an explicit account of which resource improves.

## 11. Stage 8 — assemble and audit the paper

**Goal:** produce a complete, reviewable manuscript and reproducible artifact.

### What you should do

1. Assemble the introduction, related work, method, experiments, discussion, limitations, and conclusion from the stage drafts.
2. Write the abstract last. Include the problem, intervention, evaluation scope, strongest verified result, and main qualification. Use actual measured numbers only.
3. Limit the contribution list to findings supported by the final study. Keep the split-utility experiment visible in the main paper rather than burying the mechanism evidence in an appendix.
4. Check mathematical notation, gradient normalization, pose bounds, camera conventions, visibility handling, fallback behavior, budget accounting, and state restoration against the implementation.
5. Audit every plot/table against its source run. Check units, error bars, seed counts, averaging rules, hardware labels, and whether overhead is included.
6. Prepare an artifact README with environment setup, data preparation, frozen configurations, baseline commands, expected outputs, and full-state replay instructions. Run at least one documented reproduction path from the recorded configuration.
7. Prepare supplementary material for complete per-scene tables, additional failures, ablations, and baseline reproduction details. Observe dataset and code licenses when packaging artifacts.
8. Ask your supervisor or a technical reader to challenge the calibration controls, utility experiment, clean-data performance, and novelty claim. Revise the manuscript around the objections that remain.
9. Check the chosen venue's current format, page limit, anonymity, artifact policy, and deadline before submission.

### What you should write

- Full paper draft, bibliography, figures, tables, and supplementary material.
- Artifact README and final run manifest.
- A claim checklist mapping each contribution to evidence and known limitations.

**Stage complete when:** the manuscript is internally consistent, figures are reproducible, and remaining limitations are explicit. Twelve weeks may yield a strong initial draft; allow additional time for reproduction and review before calling it submission-ready.

## 12. Records to maintain throughout

Create the supporting files as their stages begin; the paths below are planned artifacts, not files already produced by this plan.

For every run, record:

```text
run_id, date, code_revision, config_path, environment, GPU
dataset, scene, resolution, train/dev/test_view_ids
seed, pose_perturbation_id, initialization_id
method, camera_policy, score_version, probe_settings
Gaussian_budget, step_limit, total_time_limit
PSNR, SSIM, LPIPS, pose_error_if_available
final_count, peak_count, model_bytes, peak_VRAM
total_training_seconds, probe_seconds, evaluation_seconds
checkpoint_path, full_snapshot_path, log_path, status
```

Keep evaluation time separate from training time, and use the same accounting policy for all methods. Each experiment-log entry should state the question, comparison, fixed conditions, result, interpretation, and next decision. Record negative results and failures as well as successful runs.

At the end of each week, write a short update: what the evidence changed, what remains uncertain, and the next experiment that will resolve it. Avoid selecting the next task only because it is easy to implement.

## 13. Your first working session

1. Create the one-page research brief using the question and H1–H3 above.
2. Start the closest-work matrix with the sources already named in `final.md`; read the methods before claiming a distinction.
3. Record development train/dev/test view IDs for one small synthetic scene and the official room fixture.
4. Save the unmodified trainer configuration and run the ordinary and pose-enabled baselines.
5. Check full-state snapshot requirements before starting the probe or split branches.

The first research milestone is trustworthy evidence that the probe predicts useful splits beyond simpler signals and equivalent camera-refinement compute. That milestone determines whether the rest of the paper plan should proceed.
