# CCD-GS: Stage 0 research brief

**Specification:** stage0-v1, 10 October 2026, Asia/Saigon. **Status:** proposed study; no CCD-GS performance results. This is the compact brief requested by [Stage 0](../../docs/stage.md). Detailed commitments are in [study_specification.md](study_specification.md), [decision_criteria.md](decision_criteria.md), and [data_and_splits.md](data_and_splits.md).

## Question and proposed distinction

Does a bounded, temporary extrinsic-pose intervention identify useful Gaussian splits better than ordinary densification, corrected-gradient magnitude alone, or equivalent computation spent permanently refining cameras?

The proposed diagnostic measures a candidate's normalized screen-space growth gradient before and after fitting a small camera correction while holding the current scene values fixed. The diagnostic variant discards that correction. Its possible contribution is information about **subsequent split utility**, demonstrated by matched split/no-split training branches. A reduced gradient alone establishes neither camera accuracy nor an unnecessary split. Permanent pose refinement changes training state; this diagnostic asks whether the response to a constrained intervention adds allocation information at the same starting state.

## Hypotheses and falsifiers

| Hypothesis | Evidence required | Finding that defeats the intended claim |
|---|---|---|
| H1: diagnostic information | Candidate rankings change reproducibly beyond zero-update/replay variation | Changes disappear under replay or are mainly visibility changes |
| H2: split usefulness | Scores improve held-out development split-utility ranking over original and corrected gradients, with matched candidate attributes | Corrected magnitude or original gradient performs equally well; utility is noise |
| H3: allocation value | Better quality/count trade-offs survive ordinary pose refinement, retained corrections, and equal-compute pose controls | Fewer Gaussians explain everything, or retaining corrections/equivalent refinement matches the result |

H2 is mandatory. H1 by itself cannot justify an allocation-method claim. H3 is needed for practical method framing.

## Scope, outcomes, and data

Start with static RGB images, fixed intrinsics, temporary changes to camera rotation/translation, `DefaultStrategy`, and split selection. Keep cloning, pruning, opacity reset, rendering, losses, and initialization common. Defer exposure, intrinsics, distortion estimation, rolling shutter, new CUDA kernels, MCMC, and joint split/clone innovations. Multi-view aggregation follows single-view utility validation.

The primary outcome is reconstruction quality versus Gaussian count, compared against strong controls with the same camera policy. Initial aspiration: at least 20% fewer final Gaussians with development PSNR loss no greater than 0.20 dB, SSIM loss no greater than 0.005, and LPIPS increase no greater than 0.010. These are proposed engineering margins, not measured benefits or publication thresholds. Report total training time, probe cost, peak count/memory, and pose error separately.

Use the new analytic known-camera fixture, 128×128, 36/6/6 views, and official `room_small`, 389×259, 233/39/39 views. Exact IDs and hashes are recorded in [manifests](manifests/analytic_spheres_v1_splits.json). Probes and optimization use training images only; score design and branch utility use development views. Final test roles are prospectively locked. Previously evaluated room views make room a development scene, not untouched final paper evidence.

## Risks and first decision

Camera motion can compensate for wrong geometry; visibility can remove a gradient; persistent error can require optimization rather than capacity; discarded corrections may never be reached by real training; suppression can starve immature regions; probing adds rendering/backpropagation cost. Enforce bounded independent camera state, log visibility, retain fallbacks, measure split utility, and compare retained/equal-compute corrections.

Proceed to Stage 1 preparation. Do not expand the method until H1–H3 survive the small mechanism gate. Simplify if corrected magnitude suffices; reframe as calibration/scheduling if permanent refinement explains gains; preserve a reliable negative result if no predictive advantage remains.
