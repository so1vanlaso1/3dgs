# Detailed research specification

**Date:** 10 October 2026. **Version:** stage0-v1. **Purpose:** define a falsifiable study before implementing CCD-GS. Numerical settings below are prospective development choices; Stage 0 produced no training or mechanism measurements.

## 1. The unit of intervention

Let `G_t` denote all Gaussian values at training step `t`; `T_v` a camera-to-world transform; `K_v` the fixed camera intrinsics; and `L_v` the training image reconstruction loss. A candidate is one existing Gaussian that is eligible for a split under the shared density-control policy. A candidate observation consists of a candidate ID, checkpoint ID, training view ID, original camera state, probe settings, and before/after renderer measurements.

The first intervention changes one copied camera transform for one training view. It does not update Gaussian values, training camera embeddings, optimizers, strategy state, or the main view sequence. Structural growth occurs later in a separate allocation step. This separation is what permits attributing a measurement to the temporary pose intervention.

Use notation `g0(i,v,t)` and `gplus(i,v,t)` for growth signals before/after correction. They are diagnostic features. Do not call either a split-benefit estimate until evaluated against split/no-split branches. Use “pose-sensitive” for a large response; do not replace that phrase with “calibration-caused” without additional evidence.

## 2. Scope contract

| Component | Initial commitment | Reason |
|---|---|---|
| Images | Static RGB; full image at fixed resolution | Avoid introducing transient-object or crop differences into attribution |
| Scene | Existing gsplat Gaussians and renderer | Isolate density allocation from representation changes |
| Camera variables | Rotation and translation only | A small, interpretable intervention with available implementation hooks |
| Real camera policy | Ordinary pose optimization common to all pose-enabled controls | Test added diagnostic value after enabling refinement |
| Diagnostic camera policy | Independent copied pose; correction discarded | Separate observing a response from accepting a correction |
| Growth | Rank eligible split parents | Keep the structural operation common |
| Cloning/pruning | Shared ordinary rules, except controlled branches where both sides disable other structural edits | Prevent unrelated population changes from explaining split utility |
| Appearance | Disable trainable appearance/exposure modules in the initial study | Reduce compensating degrees of freedom |
| Gradient convention | Standard screen-space gradient, `absgrad=False`, same loss and normalization everywhere | Baseline eligibility and probe signal must have matching units |
| View aggregation | One view first; several training views only after mechanism validation | Avoid combining two unvalidated mechanisms |
| Kernels | Existing CUDA backend | Kernel engineering is outside the research question |

Deferred: focal length and principal point optimization, distortion estimation, exposure/color correction, rolling shutter, motion blur, dynamic scenes, learned survival policies, MCMC transitions, a new split operator, and a new clone operator. These may become future investigations only after the basic intervention survives its controls.

## 3. A bounded probe, specified provisionally

Use a right-composed update to the camera-to-world pose, `Tplus = T @ Delta`, matching the local camera-update convention. A Stage 2 implementation may use an SE(3) exponential parameterization or the existing 6D rotation module, but must bound the resulting rotation angle and translation displacement, not an arbitrary embedding norm. Pose parameterization is an implementation choice; intervention bounds and measured transforms are scientific commitments.

Proposed starting probe: five Adam pose-only updates, full training-image evidence, a pose-displacement prior, total rotation at most 1 degree, and total camera-center displacement at most 0.005 times the recorded scene scale. Limits constrain the total displacement from the starting camera, not each individual update. Record the bound hit rate and whether a projected/clipped step is used. Reject nonfinite updates. The ordinary camera optimizer's weight decay is not a substitute for explicit physical bounds.

An initial probe objective is

```text
Lprobe = Lphoto(G_t, T @ Delta, K_v; I_v)
       + lambda_pose * ((angle(Delta_R)/rotation_bound)^2
                      + (norm(center(Tplus)-center(T))/translation_bound)^2)
Lphoto = 0.8 * mean(abs(render - image)) + 0.2 * (1 - SSIM(render, image))
```

Start `lambda_pose=0.001` as a development proposal. Evaluate this choice on training/development data before freezing the implementation; record every revision. Use the photometric term alone to measure before/after screen-space growth gradients, so a new regularization term does not change the gradient comparison. The two measurements use identical image, resolution, renderer flags, Gaussian values, photometric loss, and batch-size convention.

The proposed bound sizes are deliberately conservative starting points, not estimates of the real room's pose uncertainty. A bound sweep and a zero-update control are later diagnostics. If the probe does not move appreciably, record the information limit instead of increasing the bound until CCD-GS wins.

## 4. Match the repository's gradient units

The checkout's [DefaultStrategy](../../gsplat/strategy/default.py) rescales projected-mean gradient components before accumulating their norm:

```text
q_x = grad_x * width / 2 * n_cameras
q_y = grad_y * height / 2 * n_cameras
per_observation_signal = sqrt(q_x^2 + q_y^2)
baseline_signal = accumulated_signal / max(observation_count, 1)
```

Packed rasterization uses renderer `gaussian_ids` to accumulate observations. The initial configuration has batch size one and standard gradients. Before/after comparison uses the same observed IDs and normalization. A single-view signal must not be presented as numerically identical to the strategy's accumulated multi-view average. Preserve both if a study compares them.

Holding scene values fixed means preventing scene optimization and value changes. It does not mean disabling every autograd path: projected means still need retained gradients. Clear diagnostic gradients on isolated state and prevent the probe passes from invoking ordinary densification hooks. Changing `requires_grad` indiscriminately can destroy the quantity being measured.

## 5. Candidate reliability and attribution

Log candidate ID, parent lineage, checkpoint, view, `g0`, `gplus`, photometric loss change, rotation angle, normalized camera-center displacement, projected radius, opacity, valid-observation count, and visibility before/after. Radius-positive visibility is only a geometric eligibility signal; it does not establish contribution through occlusion. Distinguish “projectable,” “contributing,” and “reliably comparable.”

Define a primary valid observation as finite measurements with a bounded correction and reliable contribution before and after. Initially flag any disappearance, newly appearing candidate, radius ratio outside `[0.5, 2]`, or large contribution change for separate analysis. Radius/contribution thresholds are provisional Stage 2 choices and require a log of exclusions. Do not set the post-change signal to zero for an invisible Gaussian and label it calibration-resolved.

An invalid observation cannot be evidence against growth. Use the original gradient as fallback and revisit eligibility later. Include invalid-probe and low-observation rates in every mechanism summary. Analyze rank results both on stable visibility observations and on the full population with the declared fallback, so favorable filtering cannot hide failures.

## 6. Features to compare before selecting a score

| Ranking | Definition or proposal | What it tests |
|---|---|---|
| Original | `g0` | Ordinary growth magnitude |
| Corrected magnitude | `gplus` | Whether a single corrected signal is sufficient |
| Ratio | `gplus / (g0 + epsilon)` | Response persistence; unstable for weak original signals |
| Magnitude-aware response | `g0 * clip(gplus/(g0+epsilon), 0, 1)` | Persistence with original magnitude; for strong signals largely reduces to capped corrected magnitude |
| Before/after feature model | Simple declared model using `g0`, `gplus`, and their difference | Whether retaining both measurements adds predictive information |
| Matched random | Randomized within original-gradient/scale/opacity/visibility strata | Whether any suppression with similar population statistics suffices |

Do not nominate the magnitude-aware example as a novel score: its algebraic overlap with corrected magnitude is a reason to test redundancy. Stage 3 selects the simplest supported rule. If only `gplus` works, revise the prospective contribution to a corrected-pose growth signal and reassess whether it adds anything beyond accepted correction. No formula is frozen for the final benchmark in Stage 0.

If a feature model is used, choose/tune it only on development study data and evaluate its ranking on held-back development candidate batches or checkpoints. Fitting and reporting the same utility labels is descriptive fit, not predictive validation. The six synthetic development views limit tuning capacity; start with simple fixed scores rather than a flexible model.

## 7. Split utility is an experimental label

For candidate `i` at checkpoint `t`, restore the same full training state into branches A/B. Split `i` in A using the ordinary fixed split operator; omit that split in B. Hold other structural operations fixed or disable them in both. Give both identical continuation views, camera policies, learning rates, continuation length, and unaffected optimizer state. Retain branch-independent random streams for the split itself and subsequent training. The branch that skips a random split must not accidentally receive a different view/random-augmentation sequence.

Define

```text
U(i,t,h) = mean_development_RGB_MSE(no_split after h steps)
         - mean_development_RGB_MSE(split after h steps)
```

Positive utility means lower development MSE with the split under that specific continuation. Begin with 200 steps and repeat a prespecified subset at 1,000 steps, subject to Stage 1 timing feasibility. Evaluate at the same fixed development cameras. A two-child split replacing one parent adds one net Gaussian; record actual operator behavior rather than assume it for every implementation.

The label is conditional on training state, operator, horizon, camera policy, and the other population. It is not an intrinsic “needed/not needed” truth for a Gaussian. Group splits have group utility; interactions prevent giving the group label to every member. A short-horizon negative split can become useful later, hence the longer-horizon check.

## 8. Essential comparisons

| Control | Treatment held common | Interpretation |
|---|---|---|
| Vanilla density control | Renderer, loss, seed, initialization | Reference pipeline; insufficient alone for the central claim |
| Ordinary density control with pose refinement | Real pose optimizer matches CCD-GS | Main camera-aware reference |
| Extra pose-only updates | Extra compute allowance matches measured probe work | Determines whether spending more on calibration explains gains |
| Retain probe corrections, ordinary growth | Same probe work and camera bounds | Directly challenges discarding the correction |
| Corrected-gradient ranking | Same fitting and gradient measurements | Tests value of before/after comparison |
| Multi-view scoring without intervention | Common sampled training views | Tests value beyond observing more views |
| Delayed densification | Matched schedule tuning allowance | Tests whether patience explains the effect |
| Threshold/random suppression | Matched split count or population trajectory where practical | Tests whether restricting growth explains compactness |
| CCD-GS | Diagnostic correction discarded | Proposed allocation treatment |

Use the same trainer for these internal controls. A scoring port is labeled a port, not an official reproduction. Published methods have different renderers, schedules, losses, initializers, and hardware; protocol compatibility must be established before importing their numbers into a comparison.

## 9. Budget accounting

For the initial integrated comparison, apply a common population cap and common cloning/pruning policy; CCD-GS changes the ordering of split candidates. Prioritize the same operator under the same available net-growth allowance. Log `N_before`, clones admitted, split parents selected, split children, parents removed, pruned count, and `N_after`. Reserve or account for cloning consistently so one arm does not spend the entire population allowance before the split ranking is considered.

Report count trajectories and peak count in addition to final count. Pruning late in training does not erase earlier representation cost. A final-count target does not automatically match training memory or total Gaussian-iteration workload. Initial exploratory budget grids are in [decision_criteria.md](decision_criteria.md); Stage 6 freezes the final campaign.

Use two comparison tracks: equal representation allowance and equal total training time. Equal optimizer step counts are not equal computation when one method probes. CUDA warmup/build and dataset preparation are recorded separately; all probe rendering, backward passes, camera fitting, and scoring count inside method training time. Evaluation remains separate and uses a fixed policy.

## 10. Scientific boundaries

The intervention is a constrained local diagnostic, not an identified causal decomposition of calibration and geometry. A camera change can lower photometric loss for an inaccurate scene. Synthetic pose truth helps inspect that ambiguity but does not eliminate it in real scenes. Camera motion also changes projection, covariance, overlap, and visibility. Those effects must be logged, not implicitly assigned to camera error.

The first fixture's surface point initialization is deliberately accurate and fixed across disturbed-pose conditions. This isolates pose effects but creates a privileged setting. The room's COLMAP reconstruction may use all capture images; record that conventional initialization provenance and do not claim strict training-only SfM. Neither scene is adequate for broad generalization. The final paper needs new public known-camera synthetic scenes and fresh real scenes beyond room.

Every claim remains conditional on [the decision criteria](decision_criteria.md). A reliable absence of advantage is a valid research outcome. A compact model that trains more slowly can support a compactness result if its quality and controls justify it, but not an unqualified speed claim.
