# R4 recovery experiments and R5 operational start

Date: 12 November 2026, Asia/Saigon.

**R5 operational verification has started and the preserved context viewer works.
The whole-room reconstruction defect is not solved.** No new reconstruction is
accepted, and no old failure has been converted into a pass.

## Why progress was blocked

The earlier work was executed, but its outputs did not meet the scientific exit
conditions. R1 physical alignment, R2 global placement and R3 difficult structures
remain unresolved. R4's delivery is the original room plus a separate matte model,
not an improved globally consistent reconstruction.

The previous prerequisite review blocked R5 entirely under its earlier conditional
request. The current request permits a useful distinction: **start testing the
existing delivery now**, while keeping final reconstruction acceptance blocked.
This is a workflow correction, not permission to weaken geometric thresholds or
claim the user has accepted a partial reconstruction as the final outcome.

## Recovery work actually executed

All experiments preserve the original source windows, sensor inputs, models and
old check coordinates. New protocols and namespaces retain unsuccessful results.

1. **Separate tracking from depth.** Fit RGB-only epipolar geometry on fitting
   observations; examine every old check without geometric filtering. Target
   essential-matrix check median/p90 is 0.675/6.960px; the depth-free diagnostic
   still has a large tail. It does not isolate tracking, reflection, rolling
   shutter, calibration or camera error. Direct first-to-last LK is also
   unreliable: only 21/74 target identities meet its forward/backward criterion.
2. **Anchor patches to their original appearance.** Independent bidirectional
   affine ECC handles patch deformation without accumulating neighboring-frame
   drift. Fixed 41px patches, correlation >=0.85, cycle <0.75px, displacement
   <=12px and bounded affine scales. Only 2/74 target and 4/112 matte identities
   satisfy the complete window. Reject this arm; it cannot supply sufficient
   validation. Its synthetic known-affine-motion test passes.
3. **Learned coarse-to-fine correspondences.** Run pretrained LoFTR indoor_new on
   the target and matte control. Preserve old checks and add source-following
   check identities in excluded regions. Rebuild fitting exclusions, propagate
   them through cross-pair components, freeze observations, then fit three
   cameras with the existing direct-depth objective. This supplies hundreds of
   checks but does not pass the geometric limits.

The first learned run used confidence >=0.8 and required both endpoints to remain
in excluded tiles. Target support was only 1/30/1 checks; an empty matte matching
list then exposed an empty-array boolean dtype bug. The failed source/run are
preserved. Fixed the dtype and added a regression test. A disclosed v2 uses the
pretrained matcher's default 0.2 confidence and follows a check source whenever
either endpoint is in an excluded tile. Both corresponding locations and linked
fitting components are excluded. This changes measurement support, not the
physical error limits. It is adaptive development, not untouched confirmation.

### Comparable results on the unchanged original moving checks

| Case | Prior direct-depth camera median/p90 | Learned-fit camera median/p90 | Decision |
|---|---:|---:|---|
| Target endpoint, 58 checks | 3.050 / 7.401px | 3.147 / 8.595px | Reject |
| Matte endpoint, 104 checks | 0.499 / 1.910px | 2.187 / 3.478px | Reject |

The unchanged limits are median <2px and p90 <4px, alongside metric, count,
spatial-support and positive-depth requirements. Learned camera fitting worsens
these retained checks, including the control. It is not a successful pose fix.

On the **different new learned checks**, target support is 546/481/604 points
across endpoint/half pairs. Its endpoint changes from 3.471/8.190px to
3.004/7.032px after fitting, still failing; metric median/p90 is 1.834/3.559cm.
Matte has 348/451/334 new checks and also fails. Both optimizations converge in
nine evaluations with 800 stable pair tracks. Pair counts and learned confidence
are not independent physical ground truth. The control failure limits any claim
that LoFTR uniquely diagnoses the old tracker or establishes true sensor error.

## R5 work verified

- Fresh server startup loads the unchanged 1,063,961-Gaussian room checkpoint
  with its own historical dataset, 477 poses and 158.69-second recorded timeline.
- Real browser actions verified playback, pause at 29.625s, free orbit while
  the timeline stays fixed, return to time zero and browser reload/reconnect.
- A live protocol test moved between 12.015s and 120.678s, then restored the
  first camera with the timeline at the other time. Final pixels are identical:
  maximum difference **0**, showing that timeline position does not secretly
  substitute a different view after camera restoration.
- At 120.678s, all four tested rotation/translation perturbations change the
  render; returning to the reference gives identical pixels. First received
  frames take 0.093–0.172s in this six-pose test. This is one local protocol
  measurement, not a comprehensive interactive latency benchmark.
- Render dimensions are consistently 2048x1152 for the protocol client's
  16:9 viewport; capture-derived vertical FOV is about 0.99165 radians.
- Existing blur/ghosting remains visible. Small camera perturbations do not
  establish collision-free navigation, unseen surface coverage or metric truth.

The matte launcher/configuration/checkpoint hashes were verified; matte browser
interaction was not newly exercised. Browser reload does not mean a second
checkpoint load in the server. Final R5 appearance/geometry acceptance and the
full before/after improved reconstruction report remain incomplete.

![Actual context viewer camera checks](../results/db3_r5_operational_20261112/context_free_camera/comparison.jpg)

## Reproduce and use

From `D:\gsplat\gsplat`:

```powershell
# Verify the selected immutable delivery without starting a server.
.\scripts\run-db3-r5.ps1 -CheckOnly
.\scripts\run-db3-r5.ps1 -Model matte -CheckOnly

# Context is already running on 8094. For a later fresh launch:
.\scripts\run-db3-r5.ps1 -Model context -Port 8094 -Background
# Matte is a separate coordinate system, not a globally aligned replacement.
.\scripts\run-db3-r5.ps1 -Model matte -Port 8095
```

Open [the context viewer](http://localhost:8094). The launcher verifies the
config, manifest and selected checkpoint against the preserved delivery seal.
It cannot grant reconstruction acceptance.

The optional learned branch uses isolated dependencies, leaving the existing
virtual environment's packages unchanged:

```powershell
.\.venv\Scripts\python.exe -m pip install --target .tools/db3-matcher --no-deps kornia==0.8.2 kornia_rs==0.1.10
.\.venv\Scripts\python.exe scripts/db3_learned_correspondence.py prepare --output-dir results/NEW_learned
.\.venv\Scripts\python.exe scripts/db3_learned_correspondence.py run --output-dir results/NEW_learned
```

The pretrained checkpoint is cached at `.cache/db3-models/loftr_indoor_ds_new.ckpt`;
its exact SHA256 and implementation hashes are frozen in the run protocol.
The script uses the recorded original parent namespaces. Source snapshots permit
inspection of the earlier failed version. New runs refuse output overwrite.

## Verification and limits

The recovery audit verifies 149 input/source hashes, reconstructs learned check
membership and fitting exclusions, reproduces new and unchanged moving-check
residuals, and verifies the original delivery artifacts and both model hashes.
The numerical residual evaluator is shared with the tested repository evaluator;
this audit is not independent metrology. The 17GB raw database hash was not
rerun because source/input/model preservation was already established and no
source writes were performed.

The relevant suite passes **23 tests**, including known affine patch recovery,
blank-image rejection, empty matcher input, transitive holdout exclusion,
camera recovery, reservation safeguards, replay and camera-refresh behavior.
The first suite invocation had six setup errors because Windows denied access
to pytest's default temporary directory. Re-running with a fresh workspace
`--basetemp` passed; no application fix was needed. One existing unknown `env`
configuration warning remains.

No full-room training, model merge, mesh, threshold relaxation, commit or push
was performed. R5 is started; R4's requested whole-room improvement is still
unachieved. A next reconstruction experiment must change the image-formation
or sensor-motion model, or validate a different supported regional route;
another unconstrained increase in training iterations has no evidence here.
This report does not prove that recapture is necessary.

## Evidence and method references

- [Tracking diagnostic](../results/db3_tracking_bias_20261112/result.json)
- [Depth-free epipolar diagnostic](../results/db3_tracking_bias_20261112/epipolar_diagnostic.json)
- [Rejected affine-template arm](../results/db3_anchor_patches_20261112/result.json)
- [Learned protocol](../results/db3_learned_correspondence_v2_20261112/protocol.json)
- [Learned results](../results/db3_learned_correspondence_v2_20261112/result.json)
- [Recovery audit](../results/db3_recovery_audit_20261112/audit.json)
- [Exact camera restoration](../results/db3_r5_operational_20261112/context_pose_restore/report.json)
- [Free-camera evidence](../results/db3_r5_operational_20261112/context_free_camera/report.json)
- [Current R5 decision](../results/db3_r5_operational_20261112/decision.json)
- [OpenCV tracking/ECC API](https://docs.opencv.org/4.7.0/dc/d6b/group__video__track.html)
- [Original LoFTR implementation](https://github.com/zju3dv/LoFTR)
- [Kornia LoFTR usage and indoor sizing guidance](https://www.kornia.org/tutorials/nbs/image_matching.html)
