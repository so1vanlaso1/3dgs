# Stage 0 — completed research specification

**Completed:** 10 October 2026, Asia/Saigon. **Plan followed:** [docs/stage.md, Stage 0](../../docs/stage.md). **Output location:** all new research documents, preparation scripts, fixture files, manifests, and evidence are under `stage_result/stage_0`.

Stage 0 is complete as a specification/data-preparation milestone. The research question is narrow and falsifiable, closest-work versions/releases have been checked, exact view identities are recorded, and the primary comparison/quality margins are defined. **No CCD-GS experiments or performance claims are established.**

## Read in this order

| Artifact | Contents |
|---|---|
| [Research brief](research_brief.md) | Compact question, H1–H3, scope, target, risks, and decision |
| [Detailed study specification](study_specification.md) | Intervention semantics, gradient normalization, candidate validity, score alternatives, branch utility, controls, budgets |
| [Related-work matrix](related_work_matrix.md) | All seven requested papers plus LeGS; pose/scoring/growth/pruning/budget/evidence comparison |
| [Source verification notes](source_verification_notes.md) | Version-sensitive corrections, actual code availability, reading scope, search record, remaining reproduction limits |
| [Data and splits](data_and_splits.md) | Chosen scenes, exact split rule/IDs, camera conventions, access checks, exposure policy |
| [Decision criteria](decision_criteria.md) | Comparable-quality margins, compactness aspiration, proposed settings, H1/H2/H3 gates, claim ledger |
| [Introduction outline](introduction_outline.md) | Argument structure, provisional prose, conditional contributions, evidence slots |
| [Stage 1 handoff](stage_1_handoff.md) | Repository hooks and baseline/snapshot work required next |
| [Research log](EXPERIMENT_LOG.md) | Executed actions, observations, interpretation, and completion decision |

## Important findings

- BlitzGS v2 includes a learned spawn-survival gate and replay/oracle analysis. The inherited v1 description therefore needed a version-aware update; broad predictive-allocation novelty claims are inappropriate. See the [verified comparison](related_work_matrix.md).
- Energy-GS and BSGS author-linked public default branches contain only README/license files at the recorded commits. Source presence/runnability was not established for those releases. See [the source audit](evidence/source_audit.json).
- The room's prior held-out views have already been evaluated. Room remains a development scene for paper claims; later final evidence requires fresh scenes.
- The stock trainer has a two-way split and evaluation-only checkpoint loading. Stage 1 must implement three-way manifests and full-state replay before controlled experiments.

## Prepared data and machine-readable evidence

| Item | Artifact |
|---|---|
| Synthetic data with exact camera truth | [Dataset manifest](data/analytic_spheres_v1/dataset_manifest.json) |
| Synthetic 36/6/6 split | [Split IDs](manifests/analytic_spheres_v1_splits.json), [all view identities/hashes](manifests/analytic_spheres_v1_views.csv) |
| Room 233/39/39 split | [Split IDs](manifests/room_small_splits.json), [all view identities/hashes](manifests/room_small_views.csv) |
| Camera metadata | [Synthetic](manifests/analytic_spheres_v1_cameras.json), [room](manifests/room_small_cameras.json) |
| Data integrity/source equivalence | [Data audit](evidence/data_audit.json) |
| Source versions/repository inventories | [Source audit](evidence/source_audit.json) |
| PDF access fallback | [Energy-GS PDF verification](evidence/energy_pdf_access.json) |
| Source checkout/context hashes | [Workspace provenance](evidence/workspace_provenance.json) |
| Prospective choices | [Study decisions](manifests/study_decisions.json) |
| Package integrity | [Validation report](evidence/validation_report.json), [artifact hashes](manifests/artifact_manifest.json) |
| Training-only synthetic visual check | [Preview](evidence/synthetic_train_preview.png) |

## Stage 0 completion checklist

| Requirement in `stage.md` | Completed artifact/evidence |
|---|---|
| Brief with question, H1–H3, outcome, risks | Research brief and detailed specification |
| Initial extrinsics/split scope; deferred topics | Scope contract |
| Verify all seven named sources/methods/versions/code/protocols | Matrix, notes, source audit; unverified releases explicitly identified |
| Pose/multi-view/intervention/growth/pruning/budget/evidence matrix | Related-work matrix |
| Narrow gap without a first-ever claim | Gap paragraph and revised predictive-allocation distinction |
| Choose/access-check synthetic and official low-resolution room | Generated fixture and executed data audit |
| Separate train/development/test and record every view ID | CSV/JSON manifests; prospective exposure policy |
| Define practical target and comparable quality before final runs | Decision criteria and machine-readable choices |
| Provisional introduction outline | Introduction outline and working prose |

The completion decision means the study can proceed to trustworthy baseline preparation. It does not mean the proposed mechanism has been validated or that every external baseline is runnable.

## Reproduce the preparation and audit

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe stage_result/stage_0/scripts/prepare_stage0_data.py
.\.venv\Scripts\python.exe stage_result/stage_0/scripts/verify_preparation_replay.py
.\.venv\Scripts\python.exe stage_result/stage_0/scripts/validate_stage0.py
```

To refresh current online source availability deliberately:

```powershell
.\.venv\Scripts\python.exe stage_result/stage_0/scripts/audit_sources.py
```

Source refreshes can change repository heads/version histories; preserve the current record if comparing later literature states. Fixture generation is deterministic and reads the existing official room/source folders. Dataset/repository audits do not run training or external code. The downloaded PDF parser wheel is stored locally in `evidence` for the documented PDF fallback; it was imported directly and was not installed into the project environment.
