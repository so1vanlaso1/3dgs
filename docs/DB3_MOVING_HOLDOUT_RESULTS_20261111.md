# Corrected local reconstruction experiment — tracking fixed, target still unaccepted

Date: 11 November 2026, Asia/Saigon.

**Outcome:** implemented source-following validation holdouts, rebuilt camera
initialization without those observations, and executed two camera/landmark
refinement objectives. A disclosed follow-up also replaced SIFT fitting matches
with tracked image corners. The matte control passes local consistency. The
difficult target now has sufficient measured support but still fails image
alignment. No new Gaussian model is promoted or trained.

This is a completed bounded local-geometry experiment. It does not deliver an
improved whole-room reconstruction, and it does not establish that the recording
is unusable. The geometry prerequisite for the requested target appearance
comparison was not met, so no new Gaussian before/after appearance claim is made.

## Fixed scope and corrected implementation

| Window | Source interval | Cameras | Role |
|---|---|---:|---|
| Target, boundary 362 | 120.678–121.145 s | 15 | Same difficult interval as the prior track-recovery experiment |
| Matte control, boundary 36 | 12.015–12.348 s | 11 | Passing interval inside the previously accepted matte region |

These are short local consistency windows, not full-scene or full-submap
reconstructions. Source identities were frozen before each experiment. All
historical/native geometry/final/buffer reservations remain excluded from fitting.

The previous method required a validation feature to remain inside fixed image
tiles. The corrected method seeds original check tiles in the first frame, then
follows that source feature across later image locations. Forward/backward flow,
patch correlation, borders and deduplication retain their prior thresholds; no
camera residual is used to select checks.

Before any fitting, the implementation excludes all original check tiles and
every fitting feature within 12 pixels of a reserved tracked observation. It
also excludes flow-valid partial tracks and propagates exclusions through whole
descriptor-linked components. Every native camera initializer is refitted from
the remaining fitting observations. Old cameras are comparison references only.
Tracks, exclusion masks and validation identities are sealed before optimization.

Both fixed objectives jointly optimize cameras and landmarks with fixed
intrinsics and first camera: robust pixel-plus-depth, and robust metric-3D
alignment. Both use a tested analytic sparse Jacobian, Jacobian parameter
scaling, and the same fixed solver budget. Validation requires >=20 checks per
pair, >5% spatial hull, metric median/p90 <0.05/0.12 m, symmetric reprojection
median/p90 <2/4 pixels, convergence and finite positive-depth output. The existing
0.01 m noninferiority and meaningful-metric-gain criteria remain unchanged.

## Validation support recovered

| Window | Full-window validation tracks | Tracks crossing old tile boundaries | Stable-depth checks: endpoint / first half / second half |
|---|---:|---:|---|
| Target | 74 | 74 | 58 / 59 / 63 |
| Matte | 112 | 63 | 104 / 103 / 107 |

The earlier fixed-tile patch experiment had **zero** full-window target tracks.
Recovering 74 tracks demonstrates an implementation limitation in that earlier
validation recipe; it does not imply new information was added to the recording.
All 74 target tracks would cross the old check-tile boundary at least once.

Check identities remain separate from fitting observations. They still share
images, sensor noise and calibration. In particular, patch overlap or correlated
surface features do not create independent physical reference measurements.

## First run: corrected exclusions with SIFT fitting tracks

Both objectives converge in both windows. The target's pixel-plus-depth arm
fails local image-alignment and original-check noninferiority criteria; the
metric arm is noninferior but still fails image alignment. Both matte arms pass
local consistency and noninferiority. None establishes the preregistered
>=0.01 m endpoint metric improvement over the existing chained estimate.

The target endpoint reprojection median/p90 changes from 5.075/7.857 pixels in
the old chain to 2.241/7.523 pixels for pixel-plus-depth and 4.262/6.875 pixels for
metric alignment. The target is therefore measurable but unresolved.

## Follow-up: fitting physical image tracks instead of descriptor matches

Before computing follow-up outcomes, froze the same source windows, validation
identities, acceptance thresholds and two objectives. New fitting corners are
seeded every third frame and tracked through native images with the same
forward/backward and appearance checks. Original check tiles and moving
reservations remain excluded. Entire fitting tracks entering reserved regions
are discarded, and seeds near previously tracked observations are suppressed.
Initial poses are rebuilt from shared stable-depth fitting tracks.

This produces 728 target fitting tracks and 793 matte fitting tracks. The
target fitting residual RMS for pixel-plus-depth is about 2.11, versus 18.99
with descriptor fitting. These are normalized residuals on **different fitting
sets**, so that difference is diagnostic and is not an independent accuracy gain.
All four follow-up optimizations converge; both target arms become noninferior
on the fixed metric checks, but still fail image alignment.

Endpoint results on exactly the same moving validation correspondences:

| Window / method | Metric median / p90 | Reprojection median / p90 | Local consistency |
|---|---|---|---|
| Target, old chain | 1.320 / 2.315 cm | 5.075 / 7.857 px | Fail |
| Target, tracked pixel+depth | 1.680 / 2.434 cm | 3.244 / 7.494 px | Fail |
| Target, tracked metric | 1.385 / 2.256 cm | 4.997 / 8.100 px | Fail |
| Matte, old chain | 1.102 / 3.099 cm | 1.942 / 3.100 px | Pass |
| Matte, tracked pixel+depth | 1.376 / 3.223 cm | 0.529 / 1.906 px | Pass |
| Matte, tracked metric | 1.043 / 2.977 cm | 1.432 / 2.833 px | Pass |

The matte results confirm that the pipeline can obtain supported local geometry
on this recording. They are not a fresh Gaussian comparison against the prior
matte model. The target's centimeter-scale depth consistency does not satisfy
the pixel-alignment requirement. No threshold was relaxed to make it pass.

[Target metric/pixel comparison chart](../results/db3_moving_holdout_audit_20261111/target_geometry_comparison.png)

## Visual inspection and interpretation

Reviewed the six highest-error unchanged target correspondences in the
source-image crops. Five are on the painting/detail region and one is on its
frame edge. The pairs appear to track nearby corresponding visual features;
the crops do not establish subpixel correspondence truth. Source blur/appearance
changes and errors from tracking, depth, calibration, timing or pose remain
possible contributors. No unique cause has been isolated, and the check set's
surface clustering limits generalization.

[Unchanged target check crops](../results/db3_moving_holdout_audit_20261111/target_check_crops.png)

None of these checks was deleted after inspection. The scientific conclusion
is that this target still cannot be accepted under the combined requirements,
not that the whole recording is bad, or that depth agreement alone proves the
room geometry is correct.

## Verification, artifacts and final decision

Seven new focused tests pass: analytic Jacobian versus numerical differentiation,
known-camera recovery in both objective arms, component-wise exclusion,
validation features crossing tiles, known-translation tracked fitting with
reserved observations excluded, and blank-input handling. One existing pytest
`env` configuration warning remains.

A separate audit passes, verifying 26 input image/feature pairs and source
reservations, seven frozen source snapshots, 104 saved camera matrices, 11,572
tracked fitting observations, and 2,342 moving validation observations. It
reproduces exclusion masks, checks the pre-fit seals, confirms fitting
observations lie outside original tiles and moving 12-pixel reservations, and
independently recomputes metric and projection residuals. It seals 117 experiment
artifacts. Both original delivered checkpoint hashes remain unchanged. The
complete raw recording hash was not recomputed in this turn.

**Decision:** keep the implemented tracking/exclusion fixes and all measured
candidate artifacts. Preserve the existing room and matte models. The matte
control passes local consistency, while target alignment blocks a new local
Gaussian appearance comparison and route expansion. No whole-scene improvement,
new Gaussian training, global merge, mesh, package installation, commit or push
is claimed. R4's reconstruction objective remains unmet.

- [Moving-holdout protocol](../results/db3_moving_holdout_20261111/protocol.json) and [results](../results/db3_moving_holdout_20261111/result.json)
- [Tracked-fitting protocol](../results/db3_tracked_fit_20261111/protocol.json) and [results](../results/db3_tracked_fit_20261111/result.json)
- [Audit](../results/db3_moving_holdout_audit_20261111/audit.json)
- Implementations: `scripts/db3_moving_holdout.py`, `scripts/db3_tracked_fit.py`, `scripts/audit_db3_moving_holdout.py`

Executed commands (recorded namespaces refuse overwrite):

```powershell
.\.venv\Scripts\python.exe scripts/db3_moving_holdout.py prepare --output-dir results/db3_moving_holdout_20261111
.\.venv\Scripts\python.exe scripts/db3_moving_holdout.py run --output-dir results/db3_moving_holdout_20261111
.\.venv\Scripts\python.exe scripts/db3_tracked_fit.py prepare --output-dir results/db3_tracked_fit_20261111
.\.venv\Scripts\python.exe scripts/db3_tracked_fit.py run --output-dir results/db3_tracked_fit_20261111
.\.venv\Scripts\python.exe scripts/audit_db3_moving_holdout.py --output-dir results/db3_moving_holdout_audit_20261111
```

Follow-up and audit scripts explicitly reference the recorded parent experiment.
A new campaign needs a newly frozen parent namespace, not overwritten results.
