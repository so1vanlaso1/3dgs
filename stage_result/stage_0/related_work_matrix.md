# Verified closest-work matrix

**Verification date:** 10 October 2026, Asia/Saigon. Sources are primary papers and author-linked repositories. This matrix describes published methods; it does not report reproductions. `Not described` means absent from the inspected method, not proof that an idea has never appeared elsewhere. See [source_verification_notes.md](source_verification_notes.md) for access/version qualifications and [source_audit.json](evidence/source_audit.json) for metadata and pinned repository inventories.

## 1. Mechanism comparison

| Work and inspected version | Pose refinement | Multi-view scoring | Temporary pose intervention for allocation | Growth rule | Pruning | Budget/resource treatment | Published evidence |
|---|---|---|---|---|---|---|---|
| [Revising Densification, v1](https://arxiv.org/html/2404.06109v1), ECCV 2024 | Not described | Maximum attributed pixel error across views | Not described | Error-driven clone/split; clone-opacity correction | Smooth opacity decay; transmittance regularization | Global cap; per-event growth limit; baseline-matched counts | Mip-NeRF 360, Tanks & Temples, Deep Blending; quality and ablations |
| [FastGS, v3](https://arxiv.org/html/2511.04283v3), CVPR 2026 project | Not a central component | High/low-error footprint scores | Not described | Gradient eligibility plus view-consistent score | View-consistent pruning | Population reduction and compact rasterization boxes | Static benchmarks plus task extensions; quality/count/time/FPS |
| [SalientGS, v2](https://arxiv.org/html/2607.11285v2), ACM MM 2026 per paper | Joint photometric and reprojection refinement | Underfit/redundancy footprint attribution | Not described | Importance-guided MCMC birth/relocation | Capacity relocation and redundancy treatment | 1.5M cap | Three real benchmark families; end-to-end costs and pose study |
| [Energy-GS, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Energy-GS_Image_Energy-guided_Pose_Alignment_Gaussian_Splatting_with_redesigned_pose_CVPR_2026_paper.html) | Joint pose/scene optimization | Not a candidate-ranking contribution | Not described | Position-frozen gradient flow; delayed ordinary densification | No new allocation pruning rule established by this audit | Alignment and longer training schedules | Synthetic and Mip-NeRF 360; rendering and pose metrics |
| [BSGS, v2](https://arxiv.org/html/2510.12493v2), ACM MM 2025 per arXiv | Pose refinement then rigid-transform deblurring | Subframe gradient aggregation | Not the proposed static-camera diagnostic | Depth/stage-dependent thresholds | No distinct count-budget pruning contribution identified | Blur/trajectory reconstruction | Real-motion-blur and ExBluRF; image/deblur metrics and ablations |
| [HarmoGS, v2](https://arxiv.org/html/2605.13073v2), preprint | Not described | Dual-view gradient conflicts | Not described | Conflict-modulated view-space gradients | Conflict EMA-driven opacity decay | Shared densification schedule; no explicit hard cap identified | PhotoTourism, NeRF On-the-go, RobustNeRF |
| [BlitzGS, v2](https://arxiv.org/html/2605.13794v2), preprint | Not described | Contribution/visibility scores | No pose intervention described | Gradient reweighting **and learned spawn-survival veto** | Scheduled contribution-based simplification | Peak/final population; distributed workload and culling | City-scale benchmarks; spawn-gate ablations and replay/oracle analysis |
| [LeGS, v2](https://arxiv.org/html/2605.00408v2), additional close work | Not described | Sampled-view gradient/sensitivity inputs | No pose intervention described | Learned maintain/clone/split/prune policy | Learned prune action | Quality/count comparisons | Standard real benchmark families; reward/policy ablations |
| CCD-GS, proposed stage0-v1 | Common ordinary real-pose optimizer | One-view diagnostic first; aggregation deferred | Bounded copied extrinsics; scene values fixed; correction discarded | Utility-validated ranking of eligible split parents | Common ordinary policy | Equal representation and equal total-time tracks | **No results yet**; H1/H2/H3 required |

The CCD-GS row is a proposed specification, not a literature finding.

## 2. Protocol and release verification

| Work | Inspected evaluation commitments | Actual code status on verification date | Reproduction implication |
|---|---|---|---|
| Revising | Every eighth view held out; PSNR/SSIM/LPIPS; growth active through 27K; budgets matched to baseline | Dedicated official implementation not located in inspected primary links | A gsplat implementation would be an explicitly documented port |
| FastGS | 30K steps, ten scoring views, RTX 4090; distinguish Base/Big | [Official code](https://github.com/fastgs/FastGS) contains trainer and renderer sources | High-priority external allocation baseline; backend speed changes require separate reporting |
| SalientGS | 13 scenes, 30K joint steps, seed 42, RTX PRO 6000 Blackwell; includes SfM in time | [Official code](https://github.com/Six-Bit-TX/SalientGS) includes source and reproduction scripts | Match input/SfM accounting; end-to-end timing differs from fixed-initialization CCD-GS |
| Energy-GS | RTX 4090, PyTorch 2.6; 50K synthetic and 100K real steps; rendering and pose errors | [Author-linked repo](https://github.com/SkylerGao/ENGS) has README/license only at audited default-branch commit | Cannot currently reproduce from that public release; internal scheduling controls remain necessary |
| BSGS | RTX 3090; 21 subframes; sharper intermediate-frame evaluation | [Author-linked repo](https://github.com/wsxujm/bsgs) has README/license only at audited default-branch commit | Blur-specific overlap; unsuitable as a directly comparable sharp static-room baseline |
| HarmoGS | Mip-Splatting backbone and AsymGS-style schedules; PSNR/SSIM/LPIPS | No author-linked runnable repository located in inspected primary sources | Treat future simplified conflict control as a port/approximation |
| BlitzGS | Four A6000 GPUs; official city-scale splits; quality/time/peak/final counts | [Official repo](https://github.com/AkierRaee/BlitzGS) contains code | Check v2 gate presence and configurations before reproduction; code existence does not establish paper parity |
| LeGS | FastGS renderer; NVIDIA L20; standard benchmark protocol | [Linked code](https://github.com/AaronNZH/LeGS) contains source | Include in broader allocation review; differs from conditional pose probing |

Repository availability above was checked with recursive GitHub trees and recorded commit SHAs. No external method was installed or executed. Dataset split/resolution, appearance treatment, camera evaluation, seed counts, and LPIPS implementation must still be reconciled before comparing published numbers to new runs.

## 3. Narrow gap after fresh verification

The investigated gap is whether **the change in a candidate's growth signal under a small, discarded extrinsic-pose fit adds information about its subsequent split benefit**, after accounting for original gradient, corrected magnitude, visibility, and equivalent permanent pose refinement. Existing error, consistency, conflict, and survival scores already guide allocation; scene-frozen pose optimization is also established. The proposed distinction therefore rests on the conjunction of the measured response, explicit state isolation, matched development split/no-split utility, and strong calibration controls. No inspected method establishes that exact conjunction for static-image split allocation, but this targeted review cannot certify exhaustive novelty. Direct prior art or a result fully explained by corrected magnitude/permanent refinement would require revising the contribution.

## 4. Version-sensitive correction to the starting assessment

The instruction in `final.md` to describe BlitzGS as contribution-guided simplification is accurate for the inspected v1. It is incomplete for **v2**, which adds a survival-predicting spawn gate. Consequently, “predicting unnecessary Gaussian creation” and “using branch replay for densification” are too broad as novelty statements. CCD-GS must distinguish pose-conditioned split utility from survival labels and establish added value empirically. The new [v2 method and appendices](https://arxiv.org/html/2605.13794v2) are the relevant source for current wording.

SalientGS v2 also replaces some v1 experimental entries and clarifies released-code verification. Use the version-pinned paper and code record; do not copy v1 results into a current comparison. [SalientGS v2](https://arxiv.org/html/2607.11285v2).

## 5. Baseline priorities determined here

1. Same-trainer ordinary pose refinement, extra pose-only computation, retained corrections, corrected-gradient-only ranking, and matched growth suppression are the first priorities. They directly challenge the mechanism and are feasible without an external framework.
2. Investigate FastGS as the first external allocation baseline. Separate its scoring from renderer acceleration and do not call a Python score approximation the official method.
3. Use SalientGS to challenge broad claims about jointly refining poses and reallocating multi-view capacity. Its MCMC/front-end changes are beyond the first controlled study.
4. Investigate strong Energy-GS-style calibration/scheduling once an executable official release exists or a documented faithful reproduction is feasible. Its public repository is presently insufficient.
5. BlitzGS v2 and LeGS are additional challenges to broad utility/predictive-allocation wording. Their dataset/hardware/policy differences make them literature comparators first, not immediate local baselines.
6. BSGS and HarmoGS establish nuisance-aware density control in different observation settings. Their findings motivate failure analysis; they do not by themselves make sharp-room protocol comparisons fair.
