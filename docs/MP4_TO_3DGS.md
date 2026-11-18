# Train a 3D Gaussian scene from an MP4

From PowerShell in the repository root, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/train-mp4-scene.ps1
```

Select the MP4 in the file picker. Keep the window open. The script extracts 30 frames/second by default, resizes the longest side to at most 960 pixels, estimates a shared SIMPLE_RADIAL camera with CPU COLMAP, uses sequential matching, undistorts the largest model, trains 120,000 RGB-only steps with a 1,000,000 Gaussian cap, and attempts to start a viewer on port 8091. Every run gets separate timestamped data and results folders. No source video is modified.

It stops before training if fewer than 20 images or 100 sparse points are reconstructed, or fewer than 80% of frames register in the largest model. This gate is a screening heuristic: passing it does not establish trajectory accuracy or full scene coverage. Inspect the 3D scene and COLMAP results, especially reflective, dynamic and unobserved regions. Intrinsics are estimated, not known calibration; shared intrinsics assume one lens with stable zoom and crop. The original video contains no measured depth in this workflow, scale is arbitrary and unseen surfaces cannot be reconstructed reliably.

Alternative example:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/train-mp4-scene.ps1 -Video "D:\videos\scene.mp4" -Fps 2 -Width 960 -Steps 7000 -Port 8091
```

Use `-PrepareOnly` for reconstruction without training. Each run saves preparation `status.json` and `ffmpeg.log` in its data folder; the results folder contains `pipeline.log`, and after training `scene.json` and `start-viewer.ps1`. If port 8091 is already occupied, training remains saved; restart using another port with `scripts/view-room.ps1`. If COLMAP fails, inspect the status and source video for blur, lack of overlap or mostly rotational movement. Lowering `-MinRegistration` can train an incomplete model and is intended for diagnostic experiments, not evidence of coverage.

The script uses the existing `.venv` pycolmap and imageio-ffmpeg packages, local CUDA/MSVC toolchain and existing trainer. It requires the existing CUDA environment. It is for ordinary perspective video; 360/equirectangular footage needs a different preparation workflow. Image stabilization and lens switching can violate the single-intrinsics assumption.

The 30 FPS default can generate very large datasets and make matching slow; adjust `-Fps` when appropriate. Training refines Gaussians until step 6,000; the rest of the steps optimize the existing set. The budget is a maximum, not a promise that training reaches that count or fits in every 6 GB GPU run.
