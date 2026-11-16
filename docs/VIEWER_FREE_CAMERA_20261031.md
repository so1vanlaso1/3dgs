# Paused free-camera rendering audit

Date: 2026-10-31 (Asia/Saigon).

## Problem

User reports the scene seems frozen to a recorded camera view, and looks poor
when navigating away. Distinguish failure to refresh from novel-view model errors.

## Current runtime and method

Verified running Python processes 16076/16744 serve port 8085 with
`results/realsense_20261030_111354_all30_120k/train/ckpts/final_post_update.pt`
and `data/realsense_20261030_111354_all30/rgbd/dataset`. Loaded model has
705,300 Gaussians. Existing camera polling fallback was already present before
this audit; no viewer-code changes or restart were performed here.

Added `scripts/audit-viewer-free-camera.py`. An independent Viser WebSocket
client defaults to paused; no replay GUI controls are touched. Starting at the
107.51s interpolated capture pose (nearest frame_003221.png), send a 5-degree
in-place yaw, local-right translations of 0.01 and 0.03 normalized world units,
and a local-forward translation of 0.03. Restore the initial full pose afterward.
Receive for five seconds per pose, saving the final decoded RGB frame and hashes.
These offsets are normalized world units, not verified meters.

## Verified results

- All four novel views return changed images. Each pose receives three frames.
- First frame after each novel camera message arrives in 0.140-0.187 seconds.
  This is first-response latency, not a claim about sustained FPS or final-quality latency.
- Final frames are all 2048 x 1152. Original/restored pixels are exactly equal.
- Mean absolute RGB changes versus original, on 0-255 scale: yaw 39.5700;
  right 0.01: 24.4283; right 0.03: 36.2457; forward 0.03: 27.4963.
  These measure image change, not reconstruction accuracy.
- Visually inspected comparison: perspective/occlusion changes with movement;
  surfaces and people show distortion/smearing, especially with lateral offset.
  The viewer is not merely displaying the original camera image unchanged.
- The code sends current camera matrices and the complete fixed Gaussian scene
  into rasterization on every render. Replay changes camera pose, not model identity.

## Embedded browser check

Inspected the user's VS Code integrated Viser at localhost:8085. Initial controls
showed paused at 106.29s, rendered count 165, with a dim viewport/header pause icon.
After a native drag in the viewport, a new image was visibly displayed. Later UI
inspection showed freshly generated control UUIDs and the initial paused message
with timeline 0, consistent with a client reconnection. A second small drag changed
the Rendered count from 115,024 to 105,025 while replay remained paused.

This is evidence of functioning input/render updates after interaction, and a
possible stale/reconnected client initially. It does not establish why that
connection changed, nor prove that a browser bug caused all prior symptoms.
Another later screenshot unexpectedly showed the foreground chat rather than the
target viewport; it is excluded as visual evidence. No further UI inputs were made.
Server stderr was empty; stdout contains an undated version-handshake rejection
from an unknown client, which cannot be attributed to the user's current session.

## Conclusion and solution boundary

No persistent camera-refresh/rendering bug reproduced. Current backend renders
novel poses; embedded UI responds to manual camera movement. Novel-view distortion
is present in the returned renders themselves and cannot be corrected simply by
requesting another render of the same pose. Pose accuracy, capture coverage, scene
motion and learned geometry remain candidates; this audit does not isolate them.

For a temporary stale session, reloading localhost:8085 in the browser is a
reasonable recovery step, not a proven permanent fix. A clean reconstruction must
be evaluated across nearby held-out poses; free navigation does not guarantee
accurate surfaces from unobserved directions. No retraining, pruning, checkpoint
modification or speculative viewer patch was performed.

## Reproduction and artifacts

```powershell
.venv/Scripts/python.exe scripts/audit-viewer-free-camera.py --dataset data/realsense_20261030_111354_all30/rgbd/dataset --output results/viewer_free_camera_20261031
```

Evidence: `results/viewer_free_camera_20261031/report.json`, six full-resolution
PNGs, and `comparison.jpg`. Audit exits unsuccessfully if novel views fail to
change, no frame arrives, dimensions fail to settle, or exact restoration differs.
