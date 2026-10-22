*I would choose the newer direction for a focused graduation project, but I would not yet call it a sufficiently established contribution for a research paper.* It has a plausible novelty gap and an achievable prototype. Its central assumption—whether temporary camera corrections reveal which Gaussian splits are unnecessary—still needs convincing experimental evidence.

I read [other_direction.md](D:/gsplat/gsplat/docs/other_direction.md) and [direction.md](D:/gsplat/gsplat/docs/direction.md), inspected the relevant implementation hooks, and checked related research available as of *9 October 2026*. This is a proposal and literature assessment; no CCD-GS experiments have been run.

| Aspect | My assessment |
|---|---|
| Importance of the problem | Strong: allocating scene capacity incorrectly is a meaningful problem |
| Scientific hypothesis | Plausible and testable, with substantial uncertainty |
| Novelty of the broad idea | Limited: calibration, densification, and multi-view consistency already overlap in prior work |
| Novelty of the exact proposed mechanism | Potentially defensible, but currently underspecified |
| Prototype feasibility | Good within your existing gsplat pipeline |
| Experimental difficulty | Substantial; separating allocation gains from calibration gains is the main challenge |
| Graduation-project suitability | Strong, provided scope and evaluation are controlled |
| Paper potential | Credible but conditional; stronger evidence is needed than the proposal currently suggests |

*The newer direction is a refinement of the old proposal, rather than a fundamentally different research idea.*

The old document’s first candidate already asks whether errors explainable by camera or exposure corrections should be discounted before densification. The new document narrows that to camera poses and measures changes in per-Gaussian growth signals.

| Dimension | Old first candidate | New CCD-GS proposal |
|---|---|---|
| Calibration variables | Pose and exposure | Pose initially |
| Main diagnostic | Residual after removing calibration-explainable error | Growth gradients before and after a temporary pose correction |
| Mathematical specification | Provides a local residual-projection equation | Leaves the persistence score undefined |
| Implementation burden | Potentially requires Jacobian estimation and error attribution | Can begin with ordinary pose optimization and gradient extraction |
| Research emphasis | Attribution of image error | Prediction of whether additional primitives are useful |

The new version is *better scoped and easier to prototype*. However, it loses some mathematical precision and some experimental safeguards present in the old document.

There is also a close mathematical connection: under a local linearization and squared-error loss, fitting a small camera correction and examining the remaining residual approximates the residual projection described in the old proposal. Measuring gradients afterward builds on that same principle.

Therefore, changing the title to “Calibration-Counterfactual Densification” does not itself add novelty. The contribution must come from the decision rule, its justification, and its demonstrated usefulness.

*The literature leaves room for a narrow contribution, but several ingredients are already established.*

| Closest work | What it already establishes | What CCD-GS would need to add |
|---|---|---|
| [Revising Densification, 2024](https://arxiv.org/abs/2404.06109) | Pixel-error-driven density control and primitive-count control | Evidence that accounting for plausible pose corrections improves allocation beyond ordinary error scoring |
| [FastGS, CVPR 2026](https://fastgs.github.io/) | Multi-view error scores that filter densification candidates and guide pruning | A useful calibration-conditioned signal beyond multi-view consistency |
| [SalientGS, 2026](https://arxiv.org/html/2607.11285v1) | Multi-view underfit/redundancy scoring with MCMC allocation and joint pose refinement | Explicit intervention on poses to evaluate candidate growth decisions |
| [Energy-GS, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Gao_Energy-GS_Image_Energy-guided_Pose_Alignment_Gaussian_Splatting_with_redesigned_pose_CVPR_2026_paper.pdf) | Redesigned pose-gradient flow and delaying densification until a later alignment stage | Local allocation improvements that remain useful after strong pose optimization and scheduling |
| [BSGS, 2025](https://arxiv.org/html/2510.12493v2) | Suppression of erroneous densification associated with camera-motion blur using adaptive thresholds | A pose-correction diagnostic for static-image reconstruction, rather than blur-specific scheduling |
| [HarmoGS, 2026](https://arxiv.org/html/2605.13073v1) | Conflict-modulated gradients for densification and handling cross-view inconsistencies | Separation of pose-sensitive growth signals from other gradient conflicts |

*FastGS should be added to the newer document’s closest-work discussion.* Its multi-view filtering is particularly relevant to your proposed second stage.

I would also correct the description of BlitzGS as “predicting unnecessary Gaussian creation.” The version inspected emphasizes contribution-guided simplification, density-control reweighting, distributed rendering, and view-dependent culling. That description is more accurate than presenting it as a split-utility predictor. [BlitzGS](https://arxiv.org/html/2605.13794v1)

I did not identify the exact proposed combination in the inspected sources. That supports investigating it; it does *not* establish a first-ever claim. A targeted search and method comparison cannot certify exhaustive novelty.

A defensible prospective contribution would be:

A bounded pose-intervention diagnostic that predicts the marginal benefit of Gaussian densification and improves allocation under matched representation and computation budgets.


That is substantially stronger than claiming “camera-aware densification” or “multi-view persistent errors,” which are too broad.

*The main scientific weakness is that a gradient change does not establish whether a split is useful.*

The proposal implicitly relies on this chain:

Pose correction reduces a Gaussian’s growth gradient → the gradient was caused by calibration error → splitting that Gaussian would have been unnecessary.


Neither implication is guaranteed.

Consider a shelf that the current scene represents in the wrong location. Moving the camera may align its rendered edge with the photograph and reduce the gradient. Yet the camera could now be less accurate, and the scene could still need additional capacity.

Conversely, an error that survives pose correction might come from reflections, moving objects, exposure differences, incorrect opacity, or poorly optimized existing Gaussians. Adding primitives may not be the appropriate response.

This produces several important risks:

- *Pose–scene ambiguity.* A camera update can compensate for an incorrect scene. Lower photometric loss does not prove better calibration.
- *Optimization versus capacity.* Persistent gradients may indicate that existing Gaussians need more optimization, rather than that more Gaussians are required.
- *Visibility changes.* A gradient can disappear because the corrected camera no longer sees the Gaussian or changes its occlusion, not because the underlying error is resolved.
- *A temporary solution may never become real.* If the probe finds a useful camera correction but discards it, suppressing growth assumes the actual optimizer will eventually achieve a similar correction.
- *Suppression can become self-reinforcing.* An immature scene can produce misleading pose corrections, which suppress useful growth and prevent the scene from improving.

These are research questions worth studying. They also explain why the proposal is promising without yet being convincing.

*The strongest reviewer objection will be: “Why not apply those camera corrections and then use ordinary densification?”*

Suppose your probe performs several pose updates and computes gradients at the corrected pose. A simple competitor can perform the same updates, retain the corrected poses, and densify normally.

If that competitor matches CCD-GS, the benefit may be better scheduling or additional optimization, with little justification for a separate counterfactual mechanism.

This means the following comparisons are essential:

| Control | What it rules out |
|---|---|
| Same trainer with ordinary pose optimization | Gains caused merely by enabling camera refinement |
| Extra pose-only updates with the same probe computation | Gains caused by giving calibration more compute |
| Delayed densification | Gains caused by waiting for a more stable scene |
| Higher densification threshold or matched random suppression | Gains caused merely by creating fewer Gaussians |
| Multi-view scoring without the camera probe | Gains caused entirely by multi-view filtering |
| Corrected-pose gradients alone | Whether comparing before/after signals adds value |
| Retain the probe’s pose corrections and densify normally | Whether temporary probing is better than using the correction directly |

The last two are missing from the newer proposal and are especially important.

*I would make the method more precise before treating it as a thesis contribution.*

A reasonable initial specification is:

1. Freeze the current scene’s parameter values during the probe.
2. Fit one bounded pose correction per sampled camera using broadly distributed image evidence.
3. Measure each candidate’s normalized growth signal before and after correction.
4. Aggregate observations across views in which the candidate has reliable visibility.
5. Use the resulting score to allocate a controlled growth budget.

For example, define

\[
g_{iv}^{0}
=
\left\|\frac{\partial L_v(G,T_v)}{\partial \mu^{2D}_{iv}}\right\|,
\qquad
g_{iv}^{+}
=
\left\|\frac{\partial L_v(G,T_v^{+})}{\partial \mu^{2D}_{iv}}\right\|.
\]

These represent a Gaussian’s screen-space growth signal before and after the probe. They are *diagnostic measurements*, not direct estimates of split benefit.

You would still need to specify:

- How large a “plausible” pose correction can be, with translation normalized to scene scale.
- How pose priors or geometric constraints prevent implausible compensation.
- Whether the score uses corrected magnitude, relative reduction, or both.
- How you handle weak gradients, changed visibility, and poorly observed Gaussians.
- Whether suppressed candidates are reconsidered later.
- Whether the real camera optimizer accepts, rejects, or independently approaches probe corrections.

A persistence ratio alone is insufficient: a tiny, useless gradient can remain perfectly persistent. The score needs to account for the amount of unresolved error or growth signal.

Also, “calibration” is broader than the initial scope. If you handle only extrinsic camera poses, explicitly say so; intrinsics, distortion, rolling shutter, and exposure remain outside the initial claim.

*For a paper, the split-utility experiment should be mandatory.*

The newer document labels short-horizon split experiments as optional. I disagree with that prioritization. They are your most direct evidence for the claimed mechanism.

At several intermediate training checkpoints:

1. Compute the proposed scores for sampled candidates.
2. Create matched training branches from identical scene, camera, optimizer, and random-number states.
3. Split a selected candidate or small candidate group in one branch; omit that split in the other.
4. Continue both branches under the same subsequent optimization schedule.
5. Measure the difference on development views excluded from fitting the probe.

The difference estimates the benefit of that split under the chosen continuation schedule.

You should then ask:

- Does CCD-GS rank useful splits better than the original gradient?
- Does it outperform corrected-gradient magnitude alone?
- Do calibration-sensitive candidates actually benefit less from splitting?
- Does this hold after accounting for original gradient, scale, opacity, and visibility?
- Does short-horizon usefulness predict longer-term improvement?

Testing a limited sample is sufficient initially; exhaustive branching would be expensive. Candidate interactions and continuation length should be acknowledged.

*If the score does not predict split benefit better, the proposed explanation is weak—even if one final PSNR result improves.*

*Your hardware and codebase are suitable for the initial investigation.*

I verified that your machine has a *GTX 1660 SUPER with 6 GB VRAM*. Your existing [room evaluation report](D:/gsplat/gsplat/results/room_hq_gtx1660/large_dataset_report.json) records 841,017 Gaussians and approximately 30.03 dB PSNR after the previous 7,000-step run. That demonstrates a usable development pipeline, not evidence for CCD-GS.

The checkout already provides:

- Camera-pose adjustments through [CameraOptModule](D:/gsplat/gsplat/examples/utils.py:27).
- Pose-optimization configuration in [simple_trainer.py](D:/gsplat/gsplat/examples/simple_trainer.py:217).
- Per-Gaussian gradient accumulation and split/clone selection in [DefaultStrategy](D:/gsplat/gsplat/gsplat/strategy/default.py:226).

A first implementation can plausibly stay within Python and existing renderer capabilities. You do not need to begin with a new CUDA rasterizer or a differentiable optimization layer.

The main engineering risks are maintaining independent probe state, comparing correctly normalized gradients, preserving Gaussian identities through structural changes, and preventing diagnostic passes from contaminating ordinary optimizer or densification statistics.

Process probe views sequentially and release their computation graphs. The number of camera parameters is small; repeated rendering and backpropagation are the likely costs.

The final benchmark campaign will be more demanding than the prototype. Stronger GPU access would help with larger scenes and reproducing competing methods, but I would validate the hypothesis locally before investing in that.

*The evaluation must distinguish practical usefulness from success on an artificial disturbance.*

I would use three experimental settings:

| Setting | Purpose |
|---|---|
| Synthetic scenes with known cameras and controlled pose errors | Establish whether the diagnostic responds to actual pose error |
| Real scenes with their ordinary estimated poses | Determine whether there is benefit in a normal reconstruction workflow |
| Real scenes with controlled pose perturbations | Measure robustness under reproducible disturbances |

For synthetic experiments, include both clean cameras and several disturbance levels. Measure rotation and translation errors separately, accounting for coordinate-frame ambiguity.

For real scenes, COLMAP poses are reference estimates, not exact ground truth. Perturbing them creates a controlled test but does not reproduce every property of real calibration errors. Eventually include a realistic imperfect-initialization setting if robustness is a central claim.

Keep the initial point cloud fixed for the first isolation experiment. Later, test imperfect initialization as well: noisy poses paired with an unusually accurate point cloud can make the problem easier than a real capture pipeline.

The main results should report:

- Quality against Gaussian count across several budgets.
- Quality against *total wall-clock time*, including probing.
- PSNR, SSIM, and LPIPS per scene.
- Final and peak Gaussian counts, model size, and peak memory.
- Pose accuracy where reference poses are available.
- Clean-data regressions and qualitative failures.
- Repeated seeds for the decisive comparisons.

A smaller model does not automatically imply faster training or rendering. Measure those outcomes separately.

Use a development split drawn from training data for score design and branch experiments. Keep final test views untouched. Your existing room result is useful for debugging, but a single scene cannot support a general methods claim.

*The project is stronger as a graduation topic than as a guaranteed publication plan.*

For a graduation project, this offers a coherent package: a concrete hypothesis, implementation, controlled experiments, visual results, resource analysis, and failure investigation. Even a carefully established negative result can have academic value, subject to your department’s requirements.

For publication, the outcome depends on what survives the controls:

| Outcome | My judgment |
|---|---|
| Beats only vanilla gsplat without pose optimization | Insufficient evidence for the claimed contribution |
| Helps noisy scenes but matches a stronger pose optimizer | Useful investigation; weak case for CCD-GS as a distinct method |
| Predicts split utility and improves quality/count trade-offs across scenes | Credible foundation for a focused methods paper |
| Adds consistent value over strong calibration and allocation baselines, with manageable cost | Substantially stronger publication case |
| Helps only extreme synthetic pose corruption | A narrow robustness result; broad efficiency claims would be difficult |

A major vision or graphics venue would require a particularly convincing mechanism and competitive evaluation. A focused workshop or specialist publication could be a more realistic initial target if the contribution is narrow but rigorously demonstrated. Venue suitability cannot be judged confidently from a proposal alone.

*I would retain the newer scope but change the research plan in five ways.*

1. *Restore the old document’s mathematical grounding and evaluation safeguards.* The new document should be a focused specification, not a shorter version that loses important controls.
2. *Make split-utility validation part of the core contribution.* Move it ahead of extensive multi-view engineering.
3. *Add FastGS and the stronger calibration controls.* Especially compare against retaining the same probe corrections.
4. *Make quality versus representation size the primary claim.* Treat training speed as a measured secondary outcome until probe costs are known.
5. *Use an explicit early decision gate.* Continue only if the diagnostic adds information and produces useful allocation decisions.

A sensible first gate is approximately two weeks of focused investigation, with three required findings:

- Pose probing changes candidate rankings reproducibly.
- Those changes predict split benefit better than simple alternatives.
- The advantage survives a baseline given equivalent pose-optimization computation.

For later experiments, gains such as 20% fewer Gaussians at comparable quality or a repeatable quality improvement at equal budget would be meaningful project targets. They are not acceptance thresholds or predicted results. Consistency, strong controls, and the explanation matter more than a chosen percentage.

The proposed 12-week schedule is plausible for a focused prototype and initial study if you can work consistently. Completing a strong publication campaign within that time is optimistic once baseline reproduction, failed ideas, multiple datasets, and writing are included.

*My decision as a judge would be: approve CCD-GS as a research investigation, with publication potential contingent on the early evidence.* I prefer its narrower scope over pursuing both directions in the old document. The decisive question should remain:

Does pose-intervention scoring identify useful Gaussian growth better than spending the same resources on camera refinement and ordinary densification?


A convincing answer to that question could support both a strong graduation project and a paper.