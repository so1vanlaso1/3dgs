# R4 bounded pose recovery — executed, no reconstruction candidate promoted

Date: 11 November 2026, Asia/Saigon.

**Outcome:** implemented and ran persistent-track camera/landmark refinement on
three windows, then one disclosed patch-tracking follow-up. No candidate meets
the complete frozen improvement criteria. The existing room and matte models
remain unchanged; no new Gaussian training or whole-scene merge was started.

The useful new result is a distinction between correspondence failures and
camera errors: the late transition has substantially better independent-of-fit
patch support than its original descriptor checks suggest. Its existing chained
poses already pass those patch checks. Refinement improves pixel alignment but
does not establish a meaningful metric improvement.

## Frozen windows and method

| Case | Boundary | Time interval | Native cameras | Reason selected |
|---|---:|---|---:|---|
| Count-limited target | 362 | 120.678–121.145 s | 15 | Complete native chain, only 12 original endpoint checks |
| Passing control | 204 | 68.149–68.482 s | 11 | Original R4 pair accepted |
| Large-tail diagnostic | 449 | 149.680–150.013 s | 11 | Original endpoint p90 approximately 0.476 m |

Selection used exposed prior results and was frozen before new measurements.
All 37 input identities remain eligible fitting/development observations; no
historical evaluation, native geometry/final reservation or buffer enters fit.

The first protocol constructs real persistent SIFT tracks across temporal gaps
1, 2, 3 and 5 plus endpoints. It rejects entire ambiguous components with two
features in one camera and entire components spanning both fitting/check tiles.
Tracks require at least three observations. At most 800 deterministic fitting
tracks with at least two stable depth observations enter joint robust pixel/depth
camera/landmark optimization. Calibration and the first camera are fixed.

Validation uses disjoint check tracks at first–last, first–middle and
middle–last cameras, frozen before optimization. It also retains the original
direct endpoint descriptor checks. Each new pair needs at least 20 stable-depth
matches, hull coverage >5% in each image, median/p90 separation <0.05/0.12 m,
median/p90 symmetric reprojection <2/4 px, and positive predicted depth.
Refinement must converge, avoid >0.01 m worsening on fixed metric checks, and
improve endpoint median or p90 by at least 0.01 m to count as a pose improvement.
These checks are local consistency, not physically calibrated ground truth.

## First experiment: persistent SIFT tracks

| Case | Persistent fitting/check tracks | New check counts: endpoints / halves | Optimizer | Result |
|---|---|---|---|---|
| Target | 798 / 129 | 3 / 7 / 0 | 60-evaluation cap | Insufficient support, no convergence, metric regression |
| Control | 2,654 / 371 | 51 / 64 / 56 | Converged at 43 | New checks pass, original depth checks worsen |
| Large tail | 1,606 / 275 | 17 / 29 / 24 | 60-evaluation cap | Endpoint count insufficient; original large tail remains |

The selected fitting subset can be smaller than the persistent-track count
because it additionally requires two stable depth observations and caps at 800.
No candidate passed the complete criteria. The control's original endpoint
median/p90 worsened from 0.01864/0.07569 m to 0.03073/0.09315 m, despite much
better reprojection. Lower fitted or pixel residuals therefore do not establish
better metric geometry.

## Disclosed follow-up: patch tracks and fixed larger solver budget

The follow-up protocol was frozen after the first results, before its own
measurements. It keeps the same fitting data, initialization, objective and
acceptance thresholds, with a fixed 240-evaluation cap for every case. It adds
first-frame corners tracked through all native images using forward/backward
optical flow (<0.75 px) and normalized neighboring-patch correlation (>=0.85).
Every tracked observation must remain within the original check tiles; no
fitted pose or metric residual filters the validation tracks. Existing checks
are preserved as regressions/vetoes.

| Case | New patch check counts: endpoints / halves | Optimizer | Full outcome |
|---|---|---|---|
| Target | 0 / 0 / 0 | Still at 240-evaluation cap | No full-window support; no promotion |
| Control | 36 / 37 / 38 | Converged at 43 | Patch checks pass, original metric regression remains |
| Large tail | 38 / 37 / 40 | Converged at 64 | Patch checks pass; no meaningful metric gain; original descriptor gate still fails |

Endpoint measurements on the **same newly frozen patch identities**:

| Case | Chain median/p90 separation | Refined median/p90 separation | Chain → refined median/p90 reprojection |
|---|---|---|---|
| Control | 0.01717 / 0.06725 m | 0.02734 / 0.06596 m | 1.862 / 2.947 → 0.448 / 1.515 px |
| Large tail | 0.01586 / 0.04929 m | 0.01706 / 0.05161 m | 1.832 / 2.647 → 0.671 / 1.341 px |

The original chain already passes all three patch pairs for both windows.
Refinement improves image alignment but does not improve endpoint metric
separation by 0.01 m. The late window's original descriptor p90 remains
0.47566 m after refinement and is **not** erased by the better patch subset.

The target begins with 269 check-tile corners but loses every full-window track
under the combined fixed-tile, forward/backward, visibility and patch-consistency
rules. This outcome does not separate which rule is dominant and does not prove
the scene lacks usable observations. Fixed image tiles are a limitation for
tracking physical features that move across tile boundaries. No tile condition
or acceptance threshold was relaxed in these runs.

## Original-match inspection at the late transition

The original endpoint checks contain three extreme separations: 0.854, 0.690 and
0.556 m. Full-resolution crop inspection shows different repeated door details
being paired: a grid intersection with a different panel corner, a latch area
with a door-edge detail, and a lower panel with another door-edge detail.
These are visible correspondence problems. Other residuals can still reflect
depth or pose problems; this inspection does not validate the entire window.

[Original match crops, worst six](../results/db3_r4_track_audit_20261111/boundary_449_original_match_crops.png)

All 23 original matches and their residuals are retained in
[the match inventory](../results/db3_r4_track_audit_20261111/boundary_449_original_matches.json).
No post-hoc deletion was used to change acceptance. The 38 patch correspondences
are a different explicitly reported check set, not a recalculation of the old
p90 after removing outliers.

## Verification and decision

Twelve focused tests pass: known-camera synthetic BA recovery, held-out track
exclusion, ambiguous/mixed-component rejection, indirect endpoint support,
spatial support failure, known-translation patch tracking, blank-image behavior,
and existing reservation/promotion safeguards. The existing unknown pytest
`env` configuration warning remains.

The separate artifact audit passes. It verifies 37 cached/new input hashes,
source reservations, six source snapshots, 28,504 fitting-track observations,
4,039 disjoint descriptor-check observations, 1,804 patch observations, fixed
check reconstruction, proper finite camera transforms, numerical metric residual
recomputation, and unchanged hashes of the original baseline and matte models.
It seals 105 experiment artifacts. This is an integrity audit; it does not
certify physical accuracy. The complete raw database hash was not rerun.

**Decision:** retain the existing models and keep R4 whole-room acceptance
unmet. These bounded trials do not justify training or expansion. The experiment
requested by the user has been executed; an improved whole-room result has not
been delivered.

The concrete remaining work is to design source-identity track holdouts that
follow features while excluding their observations from fitting, rather than
requiring a physical feature to remain in fixed image tiles. That would require
a new frozen split and fitting initialization, preserving all existing reserved
observations. Independently reviewed static structures and consistent depth
support are still needed, particularly where pixel and metric objectives
disagree. This follow-on design is not implemented here. Do not substitute
longer Gaussian training or an increased optimizer budget for those checks.

## Artifacts and reproduction

- [First protocol](../results/db3_r4_track_recovery_20261111/protocol.json), [first results](../results/db3_r4_track_recovery_20261111/result.json)
- [Follow-up protocol](../results/db3_r4_patch_followup_20261111/protocol.json), [follow-up results](../results/db3_r4_patch_followup_20261111/result.json)
- [Artifact audit](../results/db3_r4_track_audit_20261111/audit.json)
- Implementation: `scripts/db3_r4_track_recovery.py`, `scripts/db3_r4_patch_followup.py`, `scripts/audit_db3_r4_track_recovery.py`

The original-run commands, already executed, were:

```powershell
.\.venv\Scripts\python.exe scripts/db3_r4_track_recovery.py prepare --output-dir results/db3_r4_track_recovery_20261111
.\.venv\Scripts\python.exe scripts/db3_r4_track_recovery.py run --output-dir results/db3_r4_track_recovery_20261111
.\.venv\Scripts\python.exe scripts/db3_r4_patch_followup.py prepare --output-dir results/db3_r4_patch_followup_20261111
.\.venv\Scripts\python.exe scripts/db3_r4_patch_followup.py run --output-dir results/db3_r4_patch_followup_20261111
.\.venv\Scripts\python.exe scripts/audit_db3_r4_track_recovery.py --output-dir results/db3_r4_track_audit_20261111
```

These namespaces now exist and are protected against overwrite. The first
script accepts a fresh output namespace; the follow-up and audit explicitly
reference the recorded first-run namespace as their historical input. Any new
campaign must freeze its own source paths/protocol instead of rewriting these
artifacts. No commit, push, package installation, mesh or model merge occurred.
