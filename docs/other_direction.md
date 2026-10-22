# My Recommended Research Direction for Your 3DGS Project

## Proposed title

CCD-GS: Calibration-Counterfactual Densification for Efficient 3D Gaussian Splatting

Recommended direction

I recommend that your research focus on improving Gaussian densification decisions by determining whether reconstruction errors genuinely require more Gaussians or can be explained by imperfect camera calibration.

Your implementation should extend your existing gsplat pipeline, without modifying the CUDA rasterizer initially.

## 1. The research problem

Traditional 3DGS uses screen-space gradients to decide when to split or clone Gaussians.

However, high gradients can arise from different causes:

- Insufficient Gaussian representation capacity.
- Inaccurate camera poses.
- Errors in existing Gaussian parameters.
- Inconsistent observations across views.

The problem: Standard densification can allocate additional Gaussians to regions where calibration improvements might resolve the error without increasing representation capacity.

Your research would investigate how to avoid this unnecessary growth.

## 2. The proposed solution

Standard gsplat optimization

Reconstruct scene and update Gaussians

Identify densification candidates

Use existing gradient-based criteria

New: Counterfactual camera probe

Temporarily adjust camera poses and reevaluate candidate gradients

New: Calibration-persistent score

Estimate whether a candidate's growth signal remains strong after plausible camera corrections, across multiple views

Persistent error

Prioritize densification

Calibration-sensitive

Deprioritize densification

## 3. Where the novelty would come from

Existing papers already cover much of the surrounding research:

| Existing method | Established direction                                         |
| --------------- | ------------------------------------------------------------- |
| SalientGS       | Multi-view residuals and camera optimization                  |
| Energy-GS       | Joint pose and Gaussian optimization                          |
| BSGS            | Suppressing inappropriate densification caused by motion blur |
| BlitzGS         | Predicting unnecessary Gaussian creation                      |

Your intended contribution is different: explicitly test how individual Gaussian growth signals change under hypothetical camera corrections before deciding whether to split.

This is a potential novelty gap, not yet a verified first-ever invention.

## 4. What you should implement

I recommend three development stages.

| Stage | Implementation                                                                                               | Priority              |
| ----- | ------------------------------------------------------------------------------------------------------------ | --------------------- |
| 1     | Temporary camera-pose probe and comparison of original versus adjusted gradients                             | Essential             |
| 2     | Multi-view calibration-persistent scoring integrated with gsplat's DefaultStrategy                         | Main contribution     |
| 3     | Optional short-horizon experiments to verify whether predicted useful splits actually improve reconstruction | Strengthens the paper |

Start with pose uncertainty only. Add exposure handling later if the initial idea succeeds.

## 5. What your experiments must demonstrate

Your primary goal should be:

Better reconstruction quality with fewer Gaussians, without excessive training overhead.

Evaluate:

- PSNR, SSIM and LPIPS.
- Gaussian count and model size.
- Total training time and peak VRAM.
- Performance under clean and perturbed camera poses.
- Whether your score predicts genuinely useful splits.

Most importantly, compare against standard gsplat with camera-pose optimization already enabled, along with delayed densification and error-based densification baselines.

This ensures that any improvement comes from your allocation method rather than simply better pose optimization.

## 6. Suggested research timeline

| Phase       | Goal                                                                                                   | Decision                     |
| ----------- | ------------------------------------------------------------------------------------------------------ | ---------------------------- |
| Weeks 1–2   | Establish baseline and test whether temporary pose corrections meaningfully change candidate gradients | Is the hypothesis plausible? |
| Weeks 3–4   | Evaluate whether gradient persistence predicts useful splitting                                        | Is the signal informative?   |
| Weeks 5–7   | Implement multi-view CCD-GS in gsplat                                                                  | Does it improve allocation?  |
| Weeks 8–10  | Compare strong baselines, perform ablations and evaluate efficiency                                    | Is it worth publishing?      |
| Weeks 11–12 | Complete larger experiments, failure analysis and paper drafting                                       | Finalize contributions       |

These are planning estimates, not guaranteed completion times.

## My final recommendation

## Focus on smarter Gaussian creation, not simply faster optimization.

YOUR RESEARCH HYPOTHESIS

Gaussian densification can be made more efficient by testing whether its optimization signal persists after plausible camera-pose corrections.

YOUR EXPECTED CONTRIBUTION

A calibration-counterfactual, multi-view densification mechanism that improves the quality–complexity trade-off of 3DGS.

YOUR MOST IMPORTANT FIRST EXPERIMENT

Determine whether Gaussians whose gradients disappear after temporary camera correction are less likely to benefit from splitting.

YOUR MAIN RISK

Existing camera optimization might already eliminate most of these errors, leaving too little additional benefit for your method.

My advice: Pursue CCD-GS as your primary direction, but treat the first two weeks as a validation gate. If the counterfactual signal meaningfully predicts unnecessary Gaussian growth, continue to the full method. If not, reconsider the mechanism before investing in a complete implementation.

That gives you a focused research question, a plausible novelty claim, and an achievable implementation path within your existing gsplat project.