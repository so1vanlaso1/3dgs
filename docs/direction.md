**My strongest recommendation is to research how 3DGS decides when an image error actually requires more Gaussians.** A second promising direction is to optimize the slowest rendered views so that navigation has fewer frame-time spikes.

Both offer a specific contribution, measurable experiments, and a manageable graduation project. With access to a stronger GPU or cloud, you can develop locally and run the final comparisons on larger scenes.

However, **I cannot honestly guarantee that an idea has never appeared in any paper.** I searched relevant papers and preprints, including 2026 work, as of **9 October 2026**. The proposals below are gaps suggested by that search, not certified “first-ever” inventions.

| Candidate | Main improvement | Potential contribution | Main difficulty |
|---|---|---|---|
| **1. Densification conditioned on camera and exposure errors** | Better reconstruction with less unnecessary Gaussian growth | Estimate which errors remain after accounting for camera and exposure corrections, then use those errors to allocate Gaussians | Separating these effects reliably |
| **2. Optimization of slow-view rendering time** | More consistent interactive rendering | Optimize the distribution of rendering times across viewpoints, with a calibrated cost model and quality constraints | Producing gains beyond existing efficient pruning and rasterizers |

I would choose **Candidate 1 for a reconstruction-method paper**, and **Candidate 2 for a more contained graphics-systems project**.

Several attractive broad directions are already occupied:

| Broad idea | Existing work that prevents claiming it as new |
|---|---|
| Training within a memory budget | [Gaussians on a Diet, 2026](https://arxiv.org/abs/2604.20046) |
| Densifying according to reconstruction error | [Revising Densification in Gaussian Splatting, 2024](https://arxiv.org/abs/2404.06109) |
| Preserving coverage during pruning | [CoverPruneGS, ICML 2026](https://proceedings.mlr.press/v306/xiao26u.html) |
| Using validation to control Gaussian count | [VGNC, 2025](https://arxiv.org/abs/2504.14548) |
| Using conflicting gradients to guide densification | [HarmoGS, 2026 preprint](https://arxiv.org/abs/2605.13073) |
| Pruning according to computational cost | [RTGS, 2024 preprint](https://arxiv.org/abs/2407.00435) |

Consequently, your contribution needs to be more precise than “adaptive pruning,” “better densification,” or “faster 3DGS.”

**For Candidate 1, a working title is “Densify What Cannot Be Calibrated: Residual Attribution for Compact 3D Gaussian Splatting.”**

The research question is:

> Can we improve Gaussian allocation by discounting reconstruction errors that small camera-pose or exposure corrections can explain?

Imagine reconstructing a bookshelf. A blurry shelf edge could result from insufficient scene detail, slightly incorrect camera poses, or inconsistent brightness between photographs. These causes require different corrections. Your hypothesis is that densification sometimes spends scene capacity compensating for errors that should be corrected elsewhere.

The interaction between pose optimization and Gaussian structure is already recognized. **Energy-GS** changes Gaussian optimization and densification to stabilize joint pose estimation. **LighthouseGS** includes pose refinement and color correction. These are close prior works, so simply adding pose and exposure optimization would not establish novelty. [Energy-GS, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Energy-GS_Image_Energy-guided_Pose_Alignment_Gaussian_Splatting_with_redesigned_pose_CVPR_2026_paper.html), [LighthouseGS, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Han_LighthouseGS_Indoor_Structure-aware_3D_Gaussian_Splatting_for_Panorama-Style_Mobile_Captures_WACV_2026_paper.pdf).

Your proposed distinction is **an explicit estimate of calibration-explainable error used specifically to decide where to create new primitives**.

A practical first algorithm would work as follows:

1. **Warm up the reconstruction.** Start with standard 3DGS and conservative pose/exposure refinement. The diagnostic needs a reasonably useful initial scene.
2. **Estimate a small correction for each sampled camera.** Temporarily fix the Gaussian scene and estimate how much residual error a bounded pose and exposure update could explain.
3. **Compute the remaining residual.** Use this as evidence for densification.
4. **Attribute that residual to visible Gaussians.** Aggregate evidence across several co-visible training images so that one problematic image cannot dominate allocation.
5. **Spend a fixed growth budget.** Split or clone the highest-scoring candidates while keeping the same pruning and optimization machinery as the baseline.

A possible mathematical starting point is:

\[
r_{\mathrm{remaining}}
=
r-J_c(J_c^\top J_c+\lambda I)^{-1}J_c^\top r
\]

Here:

- \(r\) is the observed image minus the rendered image.
- \(J_c\) describes how rendered pixels change with small camera-pose and exposure adjustments.
- \(\lambda\) limits unstable corrections.

The equation approximately subtracts the component of the error explainable by those adjustments. You can estimate the small system using sampled pixels; a full image-sized matrix inverse is unnecessary.

**This residual is not guaranteed to represent geometry error.** It can still contain noise, reflections, occlusions, or other modeling errors. The research task is to determine whether it is a *better densification signal*.

Fit each camera correction across a sufficiently broad image region. Allowing a different “camera correction” around every Gaussian could incorrectly explain away genuine missing geometry.

The distinction from nearby methods should be documented explicitly:

| Closest work | Existing mechanism | Your proposed distinction |
|---|---|---|
| Revising Densification | Attributes pixel reconstruction error to primitives | Attributes error after estimating the portion explainable by calibration |
| Energy-GS | Redesigns pose-gradient flow and progressive alignment | Uses the remaining error to select local capacity increases |
| LighthouseGS | Refines poses and corrects appearance | Changes primitive allocation based on an explicit residual decomposition |
| HarmoGS | Handles conflicts between view-specific gradients | Separates parameter families that can explain the residual |

**I did not find this exact densification formulation in the sources inspected.** A full method-and-supplement comparison against these papers is still required before making a novelty claim.

Your experiments should establish three things.

First, demonstrate the failure mechanism. Starting with the same scene and training split, introduce controlled perturbations to training poses, exposure, and both together. Measure whether these perturbations change where Gaussians are created and whether that growth improves unseen-view reconstruction.

Second, test whether your rule improves allocation. Compare at equal Gaussian budgets and also at equal total training time.

Third, show that the benefit survives strong controls. The essential comparisons are:

- Standard 3DGS.
- The same trainer with pose and exposure refinement.
- That trainer with ordinary error-driven densification.
- That trainer with your remaining-residual densification.
- A control that simply delays densification or spends more iterations correcting poses.
- Relevant published methods, especially Energy-GS.

**The decisive comparison is against a baseline with the same pose/exposure optimizer.** Otherwise, an apparent gain could come entirely from calibration.

Reasonable *research targets*, rather than predicted results, would be:

| Measurement | Initial target |
|---|---|
| Novel-view quality under moderate capture errors | Approximately **+0.3–0.5 dB PSNR** at the same Gaussian budget, supported by SSIM/LPIPS |
| Representation efficiency | Approximately **20–30% fewer Gaussians** at comparable quality |
| Clean-data behavior | Little or no systematic regression |
| Diagnostic overhead | Ideally below **15% of total training time** |

These are project decision thresholds, not publication requirements.

Use synthetic scenes with known cameras to measure pose error accurately. Use real scenes such as Mip-NeRF 360 to evaluate practical reconstruction, but treat their COLMAP poses as reference estimates rather than exact ground truth. Keep test images untouched; any development validation images must come from the training portion.

The main risks are identifiable:

- After sufficiently good calibration, the proposed residual may become almost identical to the original residual.
- Camera corrections may absorb legitimate scene errors.
- Linearization may fail for large pose errors.
- Benefits may disappear when ordinary densification is delayed.

**The first two weeks should test these risks.** If the method cannot beat “correct poses more carefully before densifying,” it needs a different mechanism before becoming your main thesis contribution.

A successful paper would contribute a demonstrated allocation failure, a precise correction to the densification rule, and evidence that the correction improves reconstruction under matched resources.

**For Candidate 2, a working title is “Optimizing High-Percentile Frame Time in 3D Gaussian Splatting.”**

The research question is:

> Can we optimize a Gaussian scene so that its slowest viewpoints render faster while preserving reconstruction quality?

For an illustrative example, two methods might both average 60 FPS, while one regularly experiences much slower frames during close-ups. Average FPS alone would hide that difference.

The premise that Gaussian count is an incomplete speed predictor is already established. **RTGS** uses tile-intersection cost in pruning. **LODGE** selects level-of-detail thresholds using rendering workload. **HiGS** addresses workload imbalance inside the rasterizer. You must compare against these ideas, not claim to discover them. [RTGS](https://arxiv.org/html/2407.00435v1), [LODGE](https://arxiv.org/html/2505.23158v1), [HiGS, 2026 preprint](https://arxiv.org/abs/2606.00352).

The proposed contribution is narrower:

> Optimize the scene representation for high-percentile rendering time across a distribution of camera views, using measured hardware behavior and explicit quality constraints.

A practical implementation could begin with an already trained scene:

1. **Profile development viewpoints.** Record GPU frame time, visible Gaussian count, tile intersections, and workload concentration.
2. **Fit a lightweight latency predictor.** Determine which statistics predict expensive views on the target renderer and GPU.
3. **Identify expensive, low-benefit primitives or groups.** Estimate which structural changes could reduce slow-view cost with acceptable image degradation.
4. **Apply small pruning or scale-adjustment batches, then fine-tune.**
5. **Reprofile and repeat.** Rendering costs interact; removing an occluding Gaussian can expose additional work behind it. Do not assume individual cost savings simply add together.

One possible objective is:

\[
\min_G
\quad
\mathbb{E}_{v}[L_{\mathrm{image}}(G,v)]
+
\beta\,\mathrm{CVaR}_{0.95}\!\left(T(G,v)\right)
\]

Here \(T(G,v)\) is rendering time, and the CVaR term measures the average time among the slowest approximately 5% of sampled views. A memory or Gaussian-count limit can be added separately.

Neither CVaR nor latency prediction is new by itself. The contribution must be an effective scene-optimization method that meaningfully improves the quality–latency trade-off.

The evaluation should include:

| Measurement | Purpose |
|---|---|
| Median, P95 and P99 frame time | Expose slow-view behavior |
| Fraction of frames exceeding a chosen deadline | Measure practical interactivity |
| PSNR, SSIM and LPIPS | Detect quality sacrificed for speed |
| Quality of the worst-performing views | Prevent average quality from hiding damage |
| Model size and peak memory | Quantify resource use |
| Total optimization time | Account for profiling and fine-tuning overhead |

Use separate camera trajectories for optimization and final testing. Report image-quality metrics only where reference images exist. An interpolated trajectory without reference photographs can test speed, but cannot establish reconstruction accuracy.

An initial target could be **20–30% lower P95 frame time than a competitive cost-aware baseline**, with no more than approximately **0.2 dB average PSNR loss**. Again, this is an experimental target.

The critical controls are ordinary importance pruning, RTGS-style cost-aware pruning, and a simple baseline that merely gives extra weight to slow views. Test whether your optimized representation also helps a stronger rasterizer where compatible. If gains exist only on an inefficient renderer, the contribution becomes much narrower.

This candidate starts as **post-training optimization**: it can improve rendering and model size, but it cannot recover training time already spent. Any claim of faster scene creation must include the initial training, profiling, pruning, and recovery costs.

Its main novelty risk is that the method reduces to existing cost-aware pruning with a different weighting scheme. Merely reporting P95 instead of average FPS would be insufficient.

**For your graduation project, I would organize the work around an early decision rather than commit immediately to a proposed title.**

An illustrative 16-week schedule is:

| Period | Deliverable |
|---|---|
| Weeks 1–2 | Detailed prior-art comparison and a small experiment testing the proposed failure mechanism |
| Weeks 3–5 | Minimal implementation and decisive baseline comparisons |
| Weeks 6–9 | Evaluation across several scenes, resource settings and seeds |
| Weeks 10–12 | Ablations, failure cases and checks on additional scenes |
| Weeks 13–16 | Reproducible release, thesis, figures and paper draft |

Start with three diverse scenes for development. With stronger GPU access, expand the final evaluation across indoor and outdoor scenes and multiple datasets. Preserve image resolution, camera splits, initialization and evaluation conventions across comparisons. Use repeated seeds and report scene-level results, including failures.

A graduation project can succeed through a working method and a rigorous investigation even if the initial hypothesis is rejected. A publishable methods paper needs the additional evidence that the contribution is distinct from close prior work and consistently useful against competitive baselines.

**My choice for you would be Candidate 1, beginning with the question: “Does removing calibration-explainable error improve densification beyond simply optimizing calibration better?”** That experiment gives you a concrete, falsifiable starting point and a clear basis for deciding whether the direction deserves the rest of your project.