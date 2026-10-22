# Source verification notes and unresolved reproduction details

**Date:** 10 October 2026, Asia/Saigon. The inherited assessment was treated as a source list. Method/experiment sections were read, version histories checked, and linked repositories inspected. No benchmark numbers were reproduced. The complete metadata and code inventories are in [source_audit.json](evidence/source_audit.json).

## Reading and inference policy

Evidence levels are distinct: paper method described; publication metadata found; author-linked repository exists; source files present; installed successfully; experiment reproduced. Only the first four were investigated here. A README's “code available” claim does not establish executable availability. “Not located” is an audit limitation, not an assertion that no implementation exists.

Method distinctions in this package are our interpretation of the inspected sources. Negative novelty conclusions are necessarily limited. Every proposed performance improvement remains a hypothesis.

## Revising Densification

Inspected [arXiv v1, 9 April 2024](https://arxiv.org/html/2404.06109v1), Sections 3.2–3.5 and 4.1–4.2; [ECCV source](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08041.pdf) corroborates publication identity. Error attribution and count-controlled selection are established overlap. The investigated distinction is the pose intervention and later split-utility validation. A dedicated official release was not located in primary links; existing implementations that borrow ideas cannot automatically be labeled the official baseline.

## FastGS

Checked [latest arXiv v3, 6 December 2025](https://arxiv.org/html/2511.04283v3), Sections 4–5 and 9.1; the [author project](https://fastgs.github.io/) identifies CVPR 2026. Growth/pruning combine view scores with renderer acceleration. Clone and split gradient conventions differ in the published configuration, requiring care when comparing to standard-gradient gsplat. Record Base/Big, count trajectory, image resolution, kernel/backend, and total scoring cost. [Linked implementation](https://github.com/fastgs/FastGS) contains code; execution and metric parity remain unchecked.

## SalientGS

Inspected [v2, 15 July 2026](https://arxiv.org/html/2607.11285v2), Sections 3.3–3.4 and 4.1. The paper couples multi-view allocation with joint geometrically anchored pose refinement. Its v2 protocol/results differ from v1. [Code](https://github.com/Six-Bit-TX/SalientGS) includes reproduction scripts, but code presence is not a reproduction. Its all-stage timing includes feature extraction/matching/SfM; compare those separately from fixed-input CCD-GS training. The paper's conference label is author-reported metadata, not a separate proceedings audit.

## Energy-GS

Read the [CVPR accepted paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Gao_Energy-GS_Image_Energy-guided_Pose_Alignment_Gaussian_Splatting_with_redesigned_pose_CVPR_2026_paper.pdf), Sections 3.1–3.6 and 4. It freezes primitive positions, stabilizes pose-gradient flow, delays densification, and progressively exposes SVD image components. This is a strong challenge to calibration/scheduling explanations. The browser PDF endpoint failed; direct HTTP retrieval and local PDF parsing succeeded, recorded in [PDF access evidence](evidence/energy_pdf_access.json). The [linked repository](https://github.com/SkylerGao/ENGS) contains two files at its audited default-branch head, with no trainer code.

## BSGS

Read [v2, 17 October 2025](https://arxiv.org/html/2510.12493v2), Sections 3–4. Adaptive growth responds to blur depth/training stage; its trajectories and subframes solve a different observation model. Suppressing nuisance-induced growth is established overlap. Compare the static-pose diagnostic, not a general “camera-aware densification” claim. The arXiv abstract links [wsxujm/bsgs](https://github.com/wsxujm/bsgs); the audited public default branch contains README/license only. Do not confuse this with similarly named cryptographic BSGS repositories or other Gaussian papers.

## HarmoGS

Inspected [v2, 15 May 2026](https://arxiv.org/html/2605.13073v2), Sections 3.3–3.5 and 4.1. Its conflict-aware optimization/growth addresses cross-view disagreement with semantic reliability treatment; it does not establish a copied-pose response score. Distinguish a growth-gradient conflict from photometric camera sensitivity. An author-linked runnable release was not located in the paper/abstract checked here. Any later internal conflict baseline needs explicit approximation details and cannot inherit the published method's full robustness claim.

## BlitzGS

Compared [v1](https://arxiv.org/html/2605.13794v1) with [v2, 11 August 2026](https://arxiv.org/html/2605.13794v2), especially Section 3.2 and Appendices E/F. V2 adds a learned spawn-survival veto and replay/oracle analysis. Survival at the final population differs from positive marginal development reconstruction utility: pruning can remove a once-useful child, and a surviving child can be redundant. This distinction is conceptual, not evidence that CCD-GS wins. [Linked source](https://github.com/AkierRaee/BlitzGS) exists; release parity with v2 remains an implementation audit for later stages.

## Additional close work: LeGS

The targeted search and BlitzGS references exposed [LeGS v2, 11 May 2026](https://arxiv.org/html/2605.00408v2). Its learned density policy and Gaussian-removal sensitivity rewards challenge broad claims about utility-driven growth. Read Sections 3.2–3.5 and 4.1. Removal sensitivity is not identical to continued split/no-split utility under a pose probe. [Linked code](https://github.com/AaronNZH/LeGS) contains sources; no execution was attempted. Treat this as additional closest work, not an omitted baseline quietly deferred from the literature discussion.

## Search record

Queries used on 10 October 2026 included:

- `Gaussian splatting temporary camera pose intervention densification split utility counterfactual`
- `Gaussian densification calibration aware temporary pose probe`
- `Beyond Heuristics Learnable Density Control gaussian`
- Exact named-paper searches for official code and Energy-GS methods/protocols.

Only primary papers and author-linked projects/repos substantiate method claims. Secondary search hits were discovery leads. The search did not establish an exact pose-intervention/split-utility mechanism match. It did establish additional predictive-allocation overlap; the novelty language was revised accordingly.

## Remaining source checks before a final campaign

| Issue | Handling now | Later requirement |
|---|---|---|
| Revising/HarmoGS executable official release not located | Record unverified release availability | Recheck author sources; document any port |
| Energy-GS/BSGS placeholder public default branches | Do not schedule an official local reproduction | Recheck releases/branches and source parity |
| FastGS standard/absolute gradient and backend differences | Treat official baseline separately from same-trainer score controls | Pin exact source/configuration and reproduce protocol |
| SalientGS v1/v2 and timing differences | Use v2 for current protocol wording | Match input provenance and report SfM separately |
| BlitzGS code parity with v2 gate | Credit the paper mechanism; do not assume repository parity | Inspect gate implementation/configuration before running |
| All methods' exact held-out policies | Record published summaries, no score comparison yet | Resolve resolutions, metric implementation, held-out camera treatment, seeds, preprocessing, failed-run rules |

These limits do not block the narrow Stage 0 specification. They constrain which external results can be described as reproduced later.
