# R5 prerequisite review — reconstruction gate remains open

Date: 11 November 2026, Asia/Saigon.

**R5 was not started.** The user requested R5 conditional on earlier work being
correct and complete. Fresh audits confirm that the previous experiments were
executed consistently, but their scientific acceptance gates remain unsatisfied.
An additional bounded camera-refinement experiment also fails the difficult
target. No accepted improved whole-room model exists to validate as final.

This result does not establish that the recording is unrecoverable. The existing
D8 partial delivery remains available: the unchanged full-room context and a
separate narrowly accepted matte model. Viewer-only verification of that partial
delivery is possible, but would not fulfill the unresolved reconstruction gates.

## Prerequisite status

| Phase | Current evidence and remaining requirement |
|---|---|
| R0 | Source and baseline preserved; fresh full-recording hash and artifact audit pass. |
| R1 | Numerical projection/input checks pass; precise physical alignment remains partial. |
| R2 | Local submaps and separate-world fallback exist; distributed global placement remains unaccepted. |
| R3 | Matte local improvement retained; difficult exhibit structures remain unaccepted. |
| R4 | Bounded experiments completed; no accepted improved whole-room reconstruction or new global model. |
| R5 | Not started under the requested prerequisite condition; no interactive pass claimed. |

## Fresh verification

- R0–R3 integrity audit passes in 61.46 seconds, including the entire
  17,129,000,960-byte raw DB3 hash, baseline/R1 inputs, source reservations,
  model finiteness, initialization/replay consistency and historical snapshots.
  The raw SHA256 remains
  `60e3f253051fc282a61af7f6c9c5eef40830a12d42566b9eba0992463115bc52`.
- The moving-holdout audit passes again: 26 input image/feature pairs,
  seven source snapshots, 104 poses, 11,572 fitting observations, 2,342 moving
  validation observations and 117 sealed artifacts. Validation residuals and
  fitting exclusions reproduce; original delivered checkpoint hashes match.
- 36 relevant geometry, acceptance, replay and viewer unit tests pass.
  Two additional direct-depth tests pass: known metric camera recovery and
  rejection of reserved/insufficient fitting observations. The existing pytest
  unknown `env` configuration warning remains. These are 38 tests across two
  invocations, not an interactive viewer test or a full repository suite.

## Bounded corrective experiment

The previous pixel-plus-depth solver adjusts latent landmarks along with cameras.
Acceptance instead projects the measured sensor depths. A low fitted landmark
residual therefore does not necessarily imply good measured-depth reprojection.
This is a methodological hypothesis, not a demonstrated code defect or unique
cause of the reconstruction failure.

Before measuring results, froze a single camera-only refinement arm in
`results/db3_direct_depth_20261111/protocol.json`. It uses symmetric measured-depth
pixel residuals and metric separation, keeping sensor depths, intrinsics and the
first camera fixed. It reuses the existing tracked fitting observations and
rebuilds their fit-only initialization. All original spatial exclusions, moving
source reservations, validation identities and acceptance thresholds are retained.
No validation observation participates in fitting; no check is deleted.

The same fixed solver settings and 240-evaluation cap apply to the target and
matte control. There is no parameter sweep. This is adaptive development on
previously exposed data, not untouched confirmation or independent metrology.

| Window / method | Endpoint metric median / p90 (cm) | Endpoint reprojection median / p90 (px) | Local consistency |
|---|---:|---:|---|
| Target, old chain | 1.320 / 2.315 | 5.075 / 7.857 | Fail |
| Target, prior tracked pixel+depth | 1.680 / 2.434 | 3.244 / 7.494 | Fail |
| Target, direct measured-depth refinement | 1.696 / 2.485 | 3.050 / 7.401 | **Fail** |
| Matte, old chain | 1.102 / 3.099 | 1.942 / 3.100 | Pass |
| Matte, direct measured-depth refinement | 1.418 / 3.218 | 0.499 / 1.910 | Pass |

The target keeps 58/59/63 checks on its endpoint/half-window pairs; matte keeps
104/103/107. Acceptance requires all three pairs to pass, including <2 px median
and <4 px p90 reprojection, sufficient count/spatial support, positive depth and
the existing metric bounds. Both solvers converge: target 7 evaluations / 2.82 s,
matte 10 / 4.43 s. Both are noninferior under the existing 1 cm metric tolerance,
including unchanged original descriptor checks, but neither establishes the
predeclared 1 cm metric improvement. The target still fails image alignment.

The direct arm uses 670 target and 784 matte fitting tracks with valid depth pairs,
yielding 11,654 and 15,525 pair constraints. These are correlated observations of
tracks, not that many independent scene structures. Fewer participating tracks
than the stored 728/793 reflects stable-depth pair requirements, not validation
residual filtering.

An additional audit independently reproduces every saved validation residual,
checks exact retained validation identities, source reservations, all fitting
exclusions, proper camera transforms, fit-only initializers, input/source hashes
and unchanged delivered model hashes. It verifies all 27,179 fitting pair
references and 11,572 stored fitting observations. Integrity passes; scientific
acceptance does not.

## Decision and remaining work

Retain the negative result and the existing models. The tested objective change
does not resolve target alignment; do not promote it or use it to launch Gaussian
training, global merging, mesh work or final R5 acceptance. No original data,
checkpoints, thresholds or previous result namespace was changed.

Further work needs to distinguish correspondence error from sensor-depth,
calibration/timing and pose disagreement using reviewed static structures and
fitting-only diagnostics. A successful target interval would still need regional
structure checks and distributed pose/overlap validation before whole-room
expansion. Repeating longer training or only testing viewer controls cannot close
these gates. No unique cause or proof of unavailable scene information is claimed.

## Reproduction and evidence

Recorded namespaces refuse overwrite. Use fresh output namespaces to rerun:

```powershell
.\.venv\Scripts\python.exe scripts/audit_db3_r0_r3.py --output-dir results/NEW_r0_r3_audit
.\.venv\Scripts\python.exe scripts/audit_db3_moving_holdout.py --output-dir results/NEW_moving_audit
.\.venv\Scripts\python.exe scripts/db3_direct_depth_refine.py prepare --output-dir results/NEW_direct_depth
.\.venv\Scripts\python.exe scripts/db3_direct_depth_refine.py run --output-dir results/NEW_direct_depth
.\.venv\Scripts\python.exe scripts/audit_db3_direct_depth.py --run-dir results/NEW_direct_depth --output-dir results/NEW_direct_audit
```

The follow-up explicitly references the preserved moving-holdout/tracked-fitting
parent namespaces. It must not be presented as a new independent capture or split.

- [Prerequisite decision](../results/db3_r5_preflight_20261111/decision.json)
- [Fresh R0–R3 integrity audit](../results/db3_r5_preflight_20261111/r0_r3_integrity/integrity_audit.json)
- [Fresh moving-holdout audit](../results/db3_r5_preflight_20261111/moving_holdout_audit/audit.json)
- [Frozen direct-depth protocol](../results/db3_direct_depth_20261111/protocol.json)
- [Direct-depth results](../results/db3_direct_depth_20261111/result.json)
- [Independent direct-depth audit](../results/db3_r5_preflight_20261111/direct_depth_audit/audit.json)
- [Prior corrected reconstruction report](DB3_MOVING_HOLDOUT_RESULTS_20261111.md)
