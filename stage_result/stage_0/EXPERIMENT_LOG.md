# Stage 0 research and preparation log

## 2026-10-10 — specification and evidence audit

**Question:** Is there a narrow, testable pose-intervention allocation study after accounting for the closest published work and the local data pipeline?

**Inputs:** `docs/stage.md`, `docs/final.md`, background direction documents, current parser/pose/trainer/strategy implementation, existing room/source data, primary literature and author-linked repositories.

**Actions actually executed:** read Stage 0 instructions and local implementation; refresh version histories and method/protocol comparisons; inspect recursive public repository trees at pinned commits; generate an analytic known-camera fixture; decode/audit both scenes; write disjoint view IDs/hashes and normalized camera manifests; check room source/prepared equivalence; verify source/document consistency and fixture reproducibility.

**Research decisions:** extrinsics and split selection only; one-view utility validation before multi-view aggregation; quality/count primary; require original/corrected-gradient comparisons and permanent/equal-compute pose controls; freeze prospective quality margins; use room only as a previously exposed development scene for paper claims.

**Literature correction:** BlitzGS v2 adds a learned survival gate and replay/oracle study beyond the inherited v1 description. Add LeGS to closest-work coverage. Narrow the question to pose-response information validated against continued split utility. Source release availability differs from paper claims for Energy-GS/BSGS; audited default branches contain no implementation code.

**Preparation observations:** analytic fixture has 48 views/1,280 points with 36 train, 6 development, 6 reserved test; room has 311 views/112,627 points with 233 train, 39 development, 39 reserved test. Both parse successfully as pinhole datasets. Actual numeric checks are in [data_audit.json](evidence/data_audit.json); complete view identities are in manifests.

**Interpretation:** the inspected literature supports investigating the narrow question but does not certify novelty. Data and conventions are available for development. No H1/H2/H3 finding has been measured. The practical target and margins are prospective decisions.

**Gate decision:** Stage 0 specification complete; proceed to Stage 1 baseline/snapshot preparation. Do not begin a broad benchmark campaign or advertise CCD-GS benefits. The stock trainer still needs manifest-aware three-way loading and full-state replay before later experiments.

**Artifacts:** [brief](research_brief.md), [detailed specification](study_specification.md), [matrix](related_work_matrix.md), [source audit notes](source_verification_notes.md), [data protocol](data_and_splits.md), [decision criteria](decision_criteria.md), [introduction outline](introduction_outline.md), [handoff](stage_1_handoff.md), and machine-readable evidence/manifests.

**Not executed:** new baseline training, pose probe optimization, split/no-split branches, integrated allocation, external method reproduction, or final test model scoring. There are no CCD-GS numerical results to report.
