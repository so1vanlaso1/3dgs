# Small GPU pipeline test on Windows

Run from the repository root:

```powershell
.\scripts\test-small-dataset.ps1
```

The launcher uses the local `.venv`, portable MSVC tools in `.tools/msvc`,
and CUDA 12.8 in `.tools/cuda-12.8`. It targets the GTX 1660 SUPER (CUDA
architecture 7.5). The first run builds the native CUDA extension; later
runs reuse the build cache.

The existing `data/smoke_colmap` fixture contains 12 synthetic 128 x 128
images, camera poses, and 128 colored 3D points. The standard COLMAP trainer
uses 10 training views and two held-out validation views. This is an
integration smoke test; reconstruction quality on real scenes needs a
separate dataset.

Settings: 600 steps, batch size one, packed rasterization, spherical
harmonics up to degree one, and a short densification window. The CUDA
extension is built for RGB and RGB+depth (three and four channels).

Outputs go to `results/smoke_gtx1660`:

- `ckpts/ckpt_599_rank0.pt`: saved model
- `ply/point_cloud_599.ply`: exported Gaussians
- `renders/val_step599_*.png`: reference/render comparisons
- `stats/train_step0599_rank0.json`: training time, peak PyTorch GPU memory,
  and Gaussian count
- `stats/val_step0599.json`: held-out PSNR, SSIM, and LPIPS

To check checkpoint loading and evaluation:

```powershell
.\scripts\test-small-dataset.ps1 `
    -Checkpoint results/smoke_gtx1660/ckpts/ckpt_599_rank0.pt `
    -ResultDir results/smoke_gtx1660_reload
```

To change the step count or use another existing COLMAP dataset:

```powershell
.\scripts\test-small-dataset.ps1 -Steps 200 -ResultDir results/smoke_200
.\scripts\test-small-dataset.ps1 -DataDir data/another_scene -ResultDir results/another_scene
```

The memory footprint of another dataset depends on its image resolution,
initial point count, and subsequent densification. The launcher uses image
factor one, so prepare small images before passing a different dataset.

## Verified result (2026-10-08)

On the GTX 1660 SUPER with 6 GB VRAM, the 600-step training loop completed
in about 28 seconds, excluding interpreter startup and the first CUDA
build. Densification and pruning grew the model from 128 to 557 Gaussians.
Held-out PSNR improved from 19.03 to 26.25 dB; SSIM improved from 0.0947 to
0.9509. Checkpoint reload produced PSNR 26.25 dB and SSIM 0.9513.

`nvidia-smi` sampled 2748 MiB of total device memory use during the run,
including CUDA/driver and desktop overhead. The trainer separately reports
36.54 MiB of peak PyTorch tensor allocation before final evaluation. The
device sample is an observation, not a continuous peak measurement.

All checkpoint and PLY values were finite, and the PLY contained 557
vertices. Two focused GPU regression tests passed: parallel rasterizer
backward with unused normals, and spherical harmonics with empty degree-zero
coefficients. A summary is saved in `results/smoke_gtx1660/smoke_report.json`.

The initial build required Windows portability fixes for device math and
arrays, nested CUDA lambdas, bit counting, a function signature, and CUB
sorting linkage. The compiled extension is cached in `.cache/torch_extensions`.

## Real room dataset

Download the `room` scene from the official
[Mip-NeRF 360 dataset](https://jonbarron.info/mipnerf360/) and test it:

```powershell
.\.venv\Scripts\python.exe scripts/download-room-small.py
.\scripts\test-small-dataset.ps1 -DataDir data/room_small -ResultDir results/room_gtx1660
```

The downloader uses HTTP byte ranges to fetch only `room/images_8` and
`room/sparse/0`, totaling 55.75 MiB of compressed scene data from the
11.67 GiB archive. ZIP CRC checks verify downloaded files. It prepares
311 real photographs at 389 x 259 pixels and all 112,627 COLMAP points.
The camera intrinsics are rescaled to the actual reduced image dimensions;
camera poses and points are preserved. The prepared dataset uses image
factor one with the existing launcher. Original downloaded files are kept
in `data/room_small/source`; subsequent downloads reuse them.

The standard split has 272 training images and 39 held-out images. A
600-step run on the GTX 1660 SUPER completed successfully on 2026-10-08:

| Metric | Initial evaluation | After 600 steps |
| --- | ---: | ---: |
| PSNR | 12.97 dB | 24.86 dB |
| SSIM | 0.3975 | 0.8343 |
| LPIPS (AlexNet, lower is better) | 0.8408 | 0.1970 |
| Gaussians | 112,627 | 142,067 |

The training loop including its initial and final evaluation took about
31 seconds, excluding interpreter startup, dataset loading, and initial
Windows worker startup. Peak PyTorch GPU tensor allocation before final
evaluation was 105.41 MiB; this excludes CUDA/driver and desktop memory.
This short, low-resolution run verifies the pipeline; longer training and
larger images can improve reconstruction detail.

For the higher-resolution 1557 x 1038 dataset and 7,000-step training run,
see [the larger dataset test](LARGE_DATASET_WIN.md). It uses the same camera
views at about 16 times the image pixel count and evaluates the earlier
checkpoint at that resolution for comparison.

Outputs are in `results/room_gtx1660`:

- `ckpts/ckpt_599_rank0.pt`: saved model (about 12.47 MiB).
- `ply/point_cloud_599.ply`: exported Gaussians (about 12.47 MiB).
- `renders/val_step599_*.png`: 39 comparisons, reference on the left and
  reconstruction on the right.
- `stats/`: baseline/final quality and training memory/timing measurements.

Reload and evaluate the saved checkpoint with:

```powershell
.\scripts\test-small-dataset.ps1 -DataDir data/room_small `
    -ResultDir results/room_gtx1660_reload `
    -Checkpoint results/room_gtx1660/ckpts/ckpt_599_rank0.pt
```

Checkpoint reload was verified: PSNR 24.85 dB, SSIM 0.8341, and LPIPS
0.1971. The small difference from the final training evaluation occurs
because the trainer saves before its final optimizer update. The checkpoint
and PLY contain only finite values, and the PLY vertex count matches the
142,067 saved Gaussians. Source camera poses and all 3D points were checked
against the prepared dataset; the rescaled intrinsics also passed verification.
The complete result is saved in `results/room_gtx1660/room_report.json`.

## Explore the room in 3D

```powershell
.\scripts\view-room.ps1
```

Open <http://localhost:8080> in Chrome or Edge. The launcher loads the saved
checkpoint without training and starts at the first dataset camera inside
the room. Drag to rotate and scroll to zoom. Keep the terminal running;
press Ctrl+C there to stop. If port 8080 is occupied, pass `-Port 8081`.
For a background process, pass `-Background`; the launcher prints the
process ID and the command to stop its entire process tree.

Live rendering was verified for RGB, accumulated depth, expected depth, and
alpha. The viewer uses RGB+depth kernels for depth views, matching this
project's small CUDA build with three and four supported channels.
