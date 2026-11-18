# Exact camera restoration investigation

Date: 2026-10-31 (Asia/Saigon).

## Reported problem

People visible at replay time 107.51s could not be found after seeking to
228.11s and manually matching the viewpoint. Test whether the renderer changes
its result when an identical camera is restored at a different replay time.

## Controlled live test

Ran `scripts/audit-viewer-pose-restore.py` against the existing port 8085 server,
using a separate WebSocket client and the installed Viser protocol. This exercises
the live capture GUI handlers, camera message handler, nerfview rendering thread,
CUDA Gaussian rendering, and transmitted JPEG output. It does not simulate browser
mouse input or inspect the user's original embedded browser session.

The script loads the same normalized COLMAP poses and timestamp interpolation
as the viewer. It sends complete camera position, quaternion, FOV and image aspect.
Sequence, with four seconds for each camera's render to settle:

1. Set replay time 107.51s and camera A to its exact corresponding pose.
2. Set replay time 228.11s and camera B to its exact corresponding pose.
3. Leave replay time at 228.11s and restore camera A exactly.
4. Leave replay time at 228.11s; test camera B position with camera A orientation.

The version handshake initially rejected the diagnostic client (missing Viser
subprotocol); supplying `viser-v1.1.1`, matching the installed server, resolved it.
No package installation or viewer restart was required.

## Results

- The people are clearly visible in A, and reappear when A is restored at 228.11s.
- A before/after has **identical JPEG bytes and identical decoded RGB pixels**:
  max pixel difference 0, mean pixel difference 0, at 2048 x 1152 resolution.
- JPEG SHA256 for both: `33d5501dce83b3f1e2bfa0b5f985d2ac81d34cd2750ab822d00c899ade2cc760`.
- No time-dependent disappearance or persistent renderer-state change was
  reproduced by this test. This does not exclude a browser display/input issue
  specific to the user's session or an intermittent problem.
- Recorded camera A and B differ by 0.668498 normalized world units in position
  and 97.1324 degrees in orientation. Undoing parser normalization gives a
  position distance of 9.12059 source reconstruction units; no additional metric
  scale validation was performed in this investigation.
- At B's position, setting A's direction does not reproduce A's image: a nearby
  pillar fills much of the image. Same direction alone is not the same camera.

## Navigation finding and limits

`CapturePlayback.apply` places the orbit target at `position + rotation[:, 2]`,
one normalized world unit ahead. Viser uses orbit controls: turning by dragging
rotates the camera about its target, moving the camera's world position. It does
not simply turn in place. The installed `CameraControls.tsx` also uses
`cameraControls.rotate` for arrow keys. This explains why matching direction by
eye is not a reliable way to reproduce an earlier pose. It is a concrete
navigation behavior, but the user's manual camera coordinates were not recorded,
so it is not proven to be the complete cause of their observation.

No claim is made here that the people form complete, physically correct 3D bodies.
The test establishes repeatability of this particular rendered view, not correct
multi-view reconstruction of moving subjects. No model/renderer changes were
made because this controlled test did not reproduce the suspected bug.

## Artifacts and reproduction

Evidence directory: `results/viewer_pose_restore_20261031/`.

- `report.json`: exact cameras, hashes, frame counts, dimensions and comparisons.
- `a_107.png`: people visible at replay 107.51s.
- `b_228.png`: camera B at replay 228.11s.
- `a_restored_at_228.png`: exact camera A restored while replay remains 228.11s.
- `b_position_a_direction.png`: B's position with A's direction/FOV.

```powershell
.venv/Scripts/python.exe scripts/audit-viewer-pose-restore.py --dataset data/realsense_20261030_111354_all30/rgbd/dataset --output results/viewer_pose_restore_20261031
```

Keep port 8085's viewer running first. The audit creates an independent client;
capture time and camera changes apply to that client. Its connection closes at
the end. Global render settings are not changed by the audit.
