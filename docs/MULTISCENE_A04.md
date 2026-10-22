# A04: evaluation on three additional scenes

Campaign started 22 October 2026 (Asia/Saigon). This extends the earlier Room
development experiment to **Counter, Kitchen and Bonsai**, chosen before seeing
their results. These are three additional scenes, not three partitions of Room.

## Question and frozen methods

Does the combined A04 densification method retain its quality/count advantage on
other indoor scenes under the same 7,000-update protocol?

- **B00:** the existing `DefaultStrategy`, without a Gaussian cap.
- **A04:** the existing `BudgetStrategy` with a 700,000-Gaussian cap, gradient/error/
  edge weights 1/1/0.25, four contributors per pixel, observation every ten steps,
  and Long-Axis Split (`split_distance=0.45`, `opacity_reduction=0.6`). All selected
  parents are split, including those that B00 would clone.

No method parameters are tuned against the new scene results. Both methods start
from each scene's original SfM points; no run continues from Room or another
variant. Source snapshots and SHA-256 hashes are captured in the campaign folder.

## Data and paired training protocol

Official source: `https://storage.googleapis.com/gresearch/refraw360/360_v2.zip`.
The downloader requests only each selected scene's `images_2` and `sparse/0` files;
it checks HTTP byte ranges and ZIP CRCs. Intrinsics are rescaled to actual image
dimensions, while original poses and points are retained. Prepared data uses
`data_factor=1`; no second downsampling is applied.

| Scene | Images | Training | Held out | Resolution | Initial Gaussians |
|---|---:|---:|---:|---|---:|
| Counter | 240 | 210 | 30 | 1558 x 1038 | 155,767 |
| Kitchen | 279 | 244 | 35 | 1558 x 1039 | 241,367 |
| Bonsai | 292 | 255 | 37 | 1559 x 1039 | 206,613 |

Views follow the existing parser: sort image names and reserve indices divisible
by eight. Exact names, roles and image hashes are recorded in
`results/multiscene_a04_20261022/evidence/protocol.json`.

Common settings: 7,000 updates; seed 42; batch size one; packed rendering; image
caching; SH degree 3 with increments every 1,000 updates; loss
`0.8 * RGB L1 + 0.2 * SSIM loss`; original learning rates; no pose, appearance or
post-processing optimization. Densification runs at indices 600, 700, ..., 5900;
the final 1,000 updates optimize a fixed Gaussian set. Opacity reset occurs at
index 3000. Other density-control thresholds match the earlier Room protocol.

Runs execute sequentially on the local GTX 1660 SUPER. This is an equal-update
comparison, not equal wall-clock compute. Other applications can affect time;
quality and count are the primary outcomes.

## Measurements and validation

For each method/scene, report final in-memory mean held-out PSNR, SSIM and
LPIPS/AlexNet, Gaussian count, trainer time and peak PyTorch tensor allocation.
Timing excludes dataset/worker startup, final export and final validation; peak
tensor allocation excludes CUDA context and other applications.

The existing comparison utility checks matched configurations and initial
metrics, finite checkpoint tensors, checkpoint/PLY counts and A04 growth-budget
accounting. A04 is also evaluated from its saved checkpoint. The summary script
checks finite binary PLY values, all expected full-resolution held-out renders,
and unchanged captured source hashes. Checkpoints precede the final optimizer
update, so reload and final in-memory metrics need not be identical.

Per-scene results and checks are written to
`results/multiscene_a04_20261022/<scene>/comparison.json`. The combined verified
table, plots and machine-readable summary are generated only after all three
paired comparisons finish:

- `results/multiscene_a04_20261022/summary.md`
- `results/multiscene_a04_20261022/summary.json`
- `results/multiscene_a04_20261022/comparison.png`

## Reproduction

Run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/download-room-small.py --scene counter --image-factor 2
.\.venv\Scripts\python.exe scripts/download-room-small.py --scene kitchen --image-factor 2
.\.venv\Scripts\python.exe scripts/download-room-small.py --scene bonsai --image-factor 2
.\.venv\Scripts\python.exe scripts/summarize-multiscene.py --capture
.\scripts\run-multiscene-comparison.ps1
.\.venv\Scripts\python.exe scripts/summarize-multiscene.py
```

Use a new `-OutputRoot` / `--root` for a new campaign; preserve existing evidence.
The launcher refuses to overwrite incomplete run directories and skips completed
runs when continuing the same campaign.

## Interpretation boundaries

There is one seed per method/scene, no confidence interval and no outdoor scene.
Shared seed does not guarantee identical subsequent camera order because random
splitting consumes RNG differently. A04 changes placement, scale, opacity and
clone replacement together; this campaign evaluates their combined behavior.
Prior Room results remain historical development evidence and are not pooled
into a fresh-scene average. This campaign does not implement or test CCD-GS pose
probes, nor does it establish faster recovery after splitting.

## Completed results

All six training runs completed, and all three A04 checkpoint reloads passed.
The combined summary additionally verified finite PLY values, matching vertex
counts, all 102 expected held-out images per method (30 + 35 + 37), full image
dimensions and unchanged captured source hashes. Dataset pose/point preservation
and intrinsics checks also passed for all scenes.

| Scene | Method | Gaussians | PSNR (dB), up | SSIM, up | LPIPS, down | Trainer seconds | Peak tensor GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| Counter | B00 | 749,489 | 27.5592 | 0.88694 | 0.19605 | 465.6 | 1.163 |
| Counter | A04 | 569,520 | 27.8234 | 0.89029 | 0.18897 | 449.4 | 1.073 |
| Kitchen | B00 | 1,066,757 | 29.7438 | 0.91504 | 0.11620 | 573.2 | 1.647 |
| Kitchen | A04 | 699,357 | 30.0037 | 0.91546 | 0.11467 | 496.1 | 1.202 |
| Bonsai | B00 | 1,017,191 | 30.2772 | 0.92753 | 0.15912 | 455.3 | 1.543 |
| Bonsai | A04 | 695,971 | 30.4295 | 0.92846 | 0.15697 | 418.0 | 1.156 |

| Scene | Fewer Gaussians | PSNR change | SSIM change | LPIPS change |
|---|---:|---:|---:|---:|
| Counter | 24.01% | +0.2641 dB | +0.00336 | -0.00708 |
| Kitchen | 34.44% | +0.2599 dB | +0.00042 | -0.00153 |
| Bonsai | 31.58% | +0.1522 dB | +0.00093 | -0.00215 |

Across these three scenes, A04 improved all three mean held-out quality metrics
while using fewer Gaussians at the same 7,000-update budget. The unweighted mean
of per-scene PSNR gains is approximately **0.2254 dB**, and the unweighted mean
of per-scene count reductions is approximately **30.01%**. These are descriptive
averages of the three scenes, not confidence intervals or pooled per-image
metrics. Kitchen's SSIM/LPIPS gains are small, so their repeatability needs seed
replication. Trainer time and peak tensor allocation were lower in every pair;
time measurements do not establish a repeatable speedup under controlled load.

The findings extend the original Room result to three further indoor scenes.
They support the combined A04 method under this protocol; they do not identify
the independent effects of ranking, placement, scale, opacity or clone policy.
Next priorities are repeated seeds, an outdoor scene, and an ablation separating
Long-Axis Split from clone replacement, with camera sampling RNG controlled
independently of structural operations.

### A04 saved-checkpoint evaluation

The table above uses final in-memory validation. Separate reload results are:

| Scene | Reload PSNR | Reload SSIM | Reload LPIPS |
|---|---:|---:|---:|
| Counter | 27.819046 | 0.890242 | 0.189025 |
| Kitchen | 29.990192 | 0.915457 | 0.114703 |
| Bonsai | 30.434795 | 0.928492 | 0.156991 |

All reload metric deltas meet the pre-existing tolerances: 0.02 dB PSNR,
0.001 SSIM and 0.001 LPIPS; Gaussian counts match exactly. See each
`<scene>/comparison.json` for full-precision reload metrics and deltas.

![A04 count reduction and PSNR gain](../results/multiscene_a04_20261022/comparison.png)
