# Higher-resolution room pipeline test on Windows

This test uses the official [Mip-NeRF 360 room scene](https://jonbarron.info/mipnerf360/)
at image factor two: 311 photographs at 1557 x 1038 pixels, 272 training
views, 39 held-out views, and all 112,627 COLMAP initialization points.
The previous room test used 389 x 259 images. The new dataset has about
16 times more pixels per image and 5.39 times the compressed bytes.
The number of camera views is the same; this increases dataset size through
resolution rather than combining unrelated scenes.

The downloader fetches about 300.39 MiB of the official ZIP using byte
ranges, verifies extracted files using ZIP CRC checks, preserves camera
poses and points, and rescales intrinsics to the actual image dimensions.
These are source photographs at higher resolution, rather than enlarged
copies of the previous low-resolution images.

Run from the repository root using the existing local Windows toolchain:

```powershell
.\.venv\Scripts\python.exe scripts/download-room-small.py --image-factor 2
.\scripts\test-large-dataset.ps1
```

Training uses 7,000 steps, full prepared resolution (`data_factor=1`), batch
size one, packed rasterization, and spherical harmonics up to degree three.
Gaussian refinement runs from step 500 through step 5,999, every 100 steps,
with opacity resets every 3,000 steps. Images are cached in host memory.
The launcher reuses the CUDA extension and Windows toolchain from the small
test. It writes outputs to `results/room_hq_gtx1660`.

Evaluate the earlier low-resolution-trained model on these same
higher-resolution held-out images:

```powershell
.\scripts\test-small-dataset.ps1 -DataDir data/room_factor2 `
    -ResultDir results/room_hq_old_checkpoint `
    -Checkpoint results/room_gtx1660/ckpts/ckpt_599_rank0.pt
```

Check loading and evaluation of the new saved checkpoint:

```powershell
.\scripts\test-large-dataset.ps1 -ResultDir results/room_hq_gtx1660_reload `
    -Checkpoint results/room_hq_gtx1660/ckpts/ckpt_6999_rank0.pt
.\.venv\Scripts\python.exe scripts/verify-large-dataset.py
```

The verifier checks image dimensions, source camera poses and 3D points,
rescaled intrinsics, finite checkpoint and PLY values, vertex count,
degree-three coefficient shapes, all 39 saved validation renders, improved
quality over the earlier model at the same evaluation resolution, and
checkpoint reload agreement. It writes
`results/room_hq_gtx1660/large_dataset_report.json` and fails if a check fails.

Outputs:

- `ckpts/ckpt_6999_rank0.pt`: trained model.
- `ply/point_cloud_6999.ply`: exported Gaussian scene.
- `renders/val_step6999_*.png`: reference on the left, reconstruction on the right.
- `stats/`: initial and final quality, training timing, and GPU allocation.
- `large_dataset_report.json`: checked results and artifact paths.

## Verified result (2026-10-08)

The 7,000-step run completed on the GTX 1660 SUPER. Both models below were
evaluated on the same 39 held-out images at 1557 x 1038; the earlier model
was trained at low resolution for only 600 steps.

| Metric | Earlier checkpoint at high resolution | New 7,000-step model |
| --- | ---: | ---: |
| PSNR (higher is better) | 23.70 dB | 30.03 dB |
| SSIM (higher is better) | 0.7853 | 0.9021 |
| LPIPS/AlexNet (lower is better) | 0.4292 | 0.2117 |
| Gaussians | 142,067 | 841,017 |

The complete training loop, including its initial and final evaluations,
took about 7 minutes 8 seconds. Dataset loading, image caching, interpreter
startup, and Windows worker startup are additional. The saved training
statistic is 400.94 seconds, which excludes final evaluation and export.
Peak PyTorch tensor allocation before final evaluation was 1.28 GiB
(1,310.48 MiB); it excludes CUDA/driver and desktop memory.

All 13 verification checks passed. Checkpoint reload scored 30.035 dB PSNR,
0.9020 SSIM, and 0.2117 LPIPS. The checkpoint and PLY each occupy about
189.29 MiB and contain finite values. The PLY has exactly 841,017 vertices.
All 39 final comparison images are present at 3114 x 1038 pixels. A visual
inspection of the first comparison confirmed a coherent reconstruction;
some fine surface textures remain softer than the reference photograph.

The report is in `results/room_hq_gtx1660/large_dataset_report.json`.

Explore the new scene interactively:

```powershell
.\scripts\view-room.ps1 -DataDir data/room_factor2 `
    -Checkpoint results/room_hq_gtx1660/ckpts/ckpt_6999_rank0.pt
```

Then open <http://localhost:8080>. The existing viewer launcher accepts a
different `-Port` if needed. The checkpoint stores degree-three coefficients;
the viewer detects the saved degree automatically.

To change the training length, pass `-Steps` and a fresh `-ResultDir` to
`test-large-dataset.ps1`. To download full-resolution source images, use
`--image-factor 1`; they are prepared in `data/room_factor1`. Pass that
directory through `-DataDir`. Their larger memory and compute requirements
have not been tested by this factor-two run.
