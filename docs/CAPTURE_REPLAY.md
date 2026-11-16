# Capture route replay

## 2026-10-31: Manual viewpoint refresh fallback

Reported problem: the embedded viewer appeared frozen after Capture replay /
Pause / explore. A fresh in-app browser on the same port rendered camera moves
both before playback and after pause, even before the patch. Therefore the
original embedded-client failure has not been reproduced or assigned a proven
root cause; a missed camera callback is a hypothesis, not an established finding.

Added `GsplatViewer.refresh_camera_views()` and call it from the standalone
viewer loop independently of playback, at up to 15 checks per second. Changes
to each client's position, orientation, FOV or aspect submit a current-camera
move task even if the ordinary event callback was missed. Unchanged cameras
do not trigger polling renders, allowing nerfview's normal full-quality static
refinement. Disconnects discard snapshots. This fallback requires camera state
to reach the server; it cannot repair a disconnected browser or blocked input.

Verification: 7 targeted tests passed (camera-refresh regression and existing
capture-path suite); Python compilation passed. Initial pytest attempt hit a
Windows permission error in the shared temp folder; reran successfully with
`--basetemp .cache/pytest-camera-refresh-20261031 -p no:cacheprovider`.
The remaining warning is the existing unknown pytest `env` option.

Restarted only the active courtyard viewer on port 8085, loading the same
705,300-Gaussian checkpoint and 6,996-camera dataset. New launcher PID 16076,
serving child PID 16744. HTTP 200 and empty stderr verified. After restart,
browser play -> pause -> drag visibly changed the image while capture time
remained 5.719 seconds. Rendered count changed from approximately 319,030 to
327,194, then back to 319,057 on reversing the drag. Screenshot:
`.cache/paused-viewer-verified-20261031.png`. The verified browser is left open.
The original VS Code embedded tab could not be operated reliably because native
input reported another foreground window; refresh/reopen that tab to reconnect.
No checkpoint or Gaussian geometry was changed. This is a tested robustness
improvement, not proof of the cause of the original embedded-client freeze.

Date: 2026-10-30, Asia/Saigon.

## Problem and findings

The user requested travelling through the Gaussian result along the route taken
in the source recording. The viewer previously used only one saved camera as an
initial view. It already loads normalized COLMAP cameras; reusing that same parser
keeps the replay in the checkpoint's default training coordinate system.

The current 15 FPS dataset has 5,201 registered cameras. Its ancestor
`capture_manifest.json` provides original nanosecond timestamps for every pose.
The route spans 346.8827193 seconds, from `frame_000000.jpg` to
`frame_005203.jpg`.

## Solution

`examples/capture_path.py` orders cameras by timestamp, interpolates position
linearly and orientation with quaternion SLERP, and preserves camera roll and
vertical FOV. Timestamp differences are calculated as integers before conversion
to seconds. `examples/simple_viewer.py` now attaches replay controls whenever
`--camera_data_dir` supplies at least two cameras.

Each browser receives independent **Capture replay** controls: **Play capture**,
**Pause / explore**, **Return to start**, a seconds timeline, speed (0.25x to 2x),
loop and optional visible path. Pausing leaves the camera free for navigation.
Return to start pauses at the first pose; playing from the end restarts playback.
The existing selected initial view is preserved until Play or seeking.

Use the existing launcher; new processes automatically include replay:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/view-realsense-15fps.ps1 -Background
```

Open the URL actually printed by Viser. Expand **Capture replay** and click
**Play capture**. Drag **Capture time (s)** to seek, or **Pause / explore** to look
around. Existing viewer processes must restart to load new code.

Optional CLI arguments: `--capture_manifest <path>` explicitly selects timestamps;
otherwise the dataset and up to three ancestors are searched. If no manifest
covers all registered names, the UI labels timing as estimated, using
`--capture_fps` (default 15). Numeric frame suffixes preserve skipped intervals
in that fallback. Invalid/duplicate timestamps and explicit incomplete manifests
raise errors.

## Verification

- `python -m pytest tests/test_capture_path.py -q`: **6 passed**. Coverage includes
  timestamp precision/order and precedence over FPS, skipped-frame timing, SLERP,
  endpoint clamping, invalid/incomplete manifests, play/pause/seek/end/restart/loop,
  route visibility and independent clients. GUI lifecycle checks use test doubles.
- Python compilation passed for the viewer and helper. The actual dataset loaded
  successfully with recorded timestamps for all 5,201 poses.
- A separate viewer was launched with requested port 8083. Viser found it occupied
  and selected **http://localhost:8086**. Startup loaded the selected
  `final_post_update.pt` with **997,966 Gaussians** and the replay path. HTTP **200**
  was verified on port 8086. Existing viewers were left running.
- Visual playback verification is **incomplete**. The browser access host was
  unavailable. Native Chrome fallback then ended automatically because the
  computer-use tool could not confidently determine its current URL. This is a
  tool policy stop, not an observed viewer error. Browser controls, rendered
  alignment and smoothness remain unverified visually.
- Pytest reported environment warnings for an unknown `env` configuration option
  and an unwritable pytest cache; all six tests passed.

## Limits

This follows estimated reconstruction poses, not ground-truth motion. Missing
poses are interpolated; drift and Gaussian artifacts remain. Playback advances
by wall-clock time and requests 15 camera updates per second; rendering may be
slower and skip displayed views. Browser aspect ratio may differ from the source
video: vertical FOV is preserved, but exact video framing is not guaranteed.
The checkpoint must use the parser's default normalized world, as the existing
RealSense viewer does. No retraining, Gaussian edits or video export was done.
