# A05 research proposal — residual-driven split eligibility

**Date** 7 November 2026 (Asia/Saigon)\
**Status** The archive audit and original proposed protocol are preserved below. See [A05 execution](A05_RESEARCH_RESULTS_20261107.md) for the implemented variant, actual protocol and results.\
**Scope** The static-scene A01–A04 density-control pipeline on Mip-NeRF 360.

## Problem

A01 controls Gaussian growth and ranks by gradient. A02 adds attributed RGB error, A03 adds edge detail, and A04 changes how selected parents are split. A04 still requires a parent to clear the position-gradient and opacity thresholds before any of those ranking signals can act. The saved A04 configuration uses `grow_grad2d=0.0002`, `prune_opa=0.005` and disables the screen-size gate. In a late refinement window, a Gaussian can contribute to persistent rendering errors and still be ineligible because its positional gradient is small. Whether splitting such a Gaussian helps is the open question. [Growth selection](../gsplat/strategy/budget.py) · [A01–A04 history](EXPERIMENT_LOG.md)

The later completed A04 campaign supersedes the earlier stopped snapshot. Its paired replications, Room split/clone ablation and budget sweep are available. Room and Counter had more consistent PSNR gains than Kitchen and Bonsai. The Room seed-42 sweep from 700k to a 1M cap changed the final count from 597,975 to 604,705 and PSNR from 30.6007 to 30.6049 dB. A higher ceiling alone produced little change in that run. The split-parameter grid had mixed PSNR confirmation and worsened LPIPS in all three confirmation comparisons. The CCD-GS score passed 0 of 36 checkpoint blocks under the practical gate. [A04 campaign](../results/research_completion_20261027/REPORT.md) · [CCD pilot](../results/research_completion_20261027/ccd/REPORT.md)

## Findings from the saved growth histories

Audited all 12 verified A04 replication runs, with three seeds per scene. Each run had 54 refinement events at steps 600 through 5900. At every event, `added` equaled the smaller of `eligible` and available scheduled slots (`target - before`), and `after = before + added - pruned`. The next event's `before` matched the previous event's `after`. Growth histories matched the copied records in `verified.json`, and all 12 A04/B00 pairs matched initialization and training-camera sequence hashes.

An event was eligibility-limited when `eligible < target - before` **before pruning**. Counts below pool events descriptively across seeds. The mean spare count compares final live Gaussians to the 700k ceiling.

| Scene | Limited through step 3000 | Limited from step 3100 | Limited at steps 5000–5900 | Mean final Gaussians | Mean spare at finish | Eligible at step 5900 |
|---|---:|---:|---:|---:|---:|---:|
| Room | 63/75 | 87/87 | 30/30 | 598,107 | 101,893 | 2,944–3,368 |
| Counter | 60/75 | 87/87 | 30/30 | 566,074 | 133,926 | 2,410–2,963 |
| Kitchen | 14/75 | 84/87 | 27/30 | 697,975 | 2,025 | 3,481–5,606 |
| Bonsai | 3/75 | 87/87 | 30/30 | 689,408 | 10,592 | 4,487–5,737 |

- Eligibility limited **117/120 late events**. Room and Counter had substantial unused headroom and frequently ran short of eligible parents even before the reset. Kitchen often came close to the cap by the end despite the late shortfalls.
- Step 3100 alone accounted for **63.9–83.8%** of all pruning within each run. Step 3000 resets opacity after refinement, and step 3100 is also the first event at which 3D scale pruning is active. The logs contain the combined prune count. Separate opacity-only, scale-only and overlap counters would distinguish their roles.
- The histories report the combined eligibility decision. They cannot separate low gradient from low opacity or show whether rejected parents carry error worth addressing. Nor do they tell us whether an unused slot would improve a held-out view.

Evidence is in `results/research_completion_20261027/replication/<scene>/A04_s<seed>/stats/train_step6999_rank0.json` for seeds 42, 43 and 44. [Replication files](../results/research_completion_20261027/replication/) · [Reset and prune callbacks](../gsplat/strategy/default.py)

## Proposed solution

Keep the A04 ranking, scheduled budget, opacity floor, pruning, and Long-Axis Split. At late refinement events with empty slots, allow a small number of visible parents below the ordinary gradient threshold to qualify when they have high attributed RGB error across distinct training cameras. Use the existing top-contributor projection query. The initial score is the mean contribution-weighted absolute RGB error across distinct contributing views, with at least three distinct views required. Keep the support requirement identical in the relaxed-gradient control. Log rejected parents by gradient, opacity, support and overlaps. A camera ID counts once in the experimental score window even if the sampler draws it repeatedly.

Limit the initial change to one refinement event per branch. Start with 512 additional parents, bounded by the shared candidate pool and slots left by ordinary growth before pruning. Set the count before inspecting continuation metrics. Apply the same cap in every branch. A large raw RGB error can reflect a reflection, registration error or an already-correct thin feature, so selection diagnostics need actual split utility.

## Smallest useful pilot

1. Capture complete A04 state for Room and Counter, seed 42, at the scheduled steps 5000 and 5800 before refinement. Save parameters, optimizer state, RNG, camera sequence and candidate evidence. Derive candidates using the original render and gradient conventions. Retain exact camera IDs and per-parent evidence for the experimental score.
2. Fork each checkpoint into three branches. Run the common A04 event in all branches, then apply the extra splits to disjoint, surviving below-threshold parents. Use the same number of extra splits and the same supported candidate pool in the last two branches.

   | Branch | Extra splits after the common A04 event |
   |---|---|
   | A04 control | None |
   | Relaxed-gradient control | Highest positional gradient in the supported below-threshold pool |
   | A05 | Highest attributed RGB error in the same pool |

3. Run 100 and 500 identical subsequent optimizer updates with additional structural changes disabled. Replay the A04 control to measure numerical variation. Log PSNR, SSIM, LPIPS and per-view RGB MSE at the intervention, at 100 updates and at 500 updates. Inspect a predeclared set of detail and low-texture crops from both scenes. Previously inspected evaluation views serve as development views. Keep them out of score collection and optimization.
4. Advance only if the residual branch consistently helps relative to the no-extra-split control and beats the equal-count gradient control at 500 updates by more than replay variation on both scenes. Use a provisional practical gate of at least +0.05 dB mean PSNR over the gradient control within each scene, with both checkpoints directionally positive, SSIM loss at most 0.001, and LPIPS increase at most 0.001. These are proposed engineering thresholds. Record ties, reversals and failures. If the gradient control matches A05, retain the simpler explanation that admitting more candidates drove the change.

If the pilot passes, run the frozen policy for 7,000 updates against A04 and relaxed-gradient A04 over three paired seeds. Keep the 700k ceiling and shared training sequences. Report actual final counts and matched-count operating points, since different pruning can make equal caps end at different sizes. Include a fresh outdoor scene such as Garden with the split frozen before inspecting results. Log total runtime, peak GPU allocation, final count and metrics alongside per-view changes. Add the repository's `MCMCStrategy` as a stronger external method baseline on the same data and update budget, with matched-count points where feasible. [3DGS as MCMC](https://proceedings.neurips.cc/paper_files/paper/2024/hash/93be245fce00a9bb2333c17ceae4b732-Abstract-Conference.html)

## Related methods and other directions

- **Pixel error and multiview support.** Revising Densification already uses pixel-error-driven growth and primitive-count control. FastGS uses multiview error scores with gradient-based candidate selection. The scoped A05 question is whether below-threshold parents use otherwise empty A04 slots better than a relaxed gradient rule at equal count. A positive pilot would establish a local result on this pipeline. [Revising Densification](https://arxiv.org/abs/2404.06109) · [FastGS](https://arxiv.org/html/2511.04283v3)
- **Absolute gradients.** The local trainer and strategy already support `--strategy.absgrad`. AbsGS targets cancellation in projected positional gradients. Use it as an inexpensive additional comparison with a calibrated threshold if the pilot suggests standard gradients miss important detail. A direct toggle changes the gradient scale as well as the eligible population. [AbsGS](https://arxiv.org/abs/2404.10484)
- **Post-reset pruning.** If pruning diagnostics show that the first post-reset event drops useful parents, test recovery-aware opacity pruning while retaining a separate scale check. Keep the growth schedule fixed for that comparison so changed refinement timing cannot explain the result. Recovery-aware pruning already appears in the Long-Axis Split line of work. [Improving Densification in 3D Gaussian Splatting](https://arxiv.org/abs/2508.12313)

## Recheck the audit

Run the following Python from the repository root. It reads the verified training statistics and prints the three eligibility-limited counts, final mean count, last eligible range and pruning-share range by scene.

```python
import json
from pathlib import Path
from statistics import mean

root = Path("results/research_completion_20261027/replication")
for scene in ("room", "counter", "kitchen", "bonsai"):
    histories = []
    for seed in (42, 43, 44):
        run = root / scene / f"A04_s{seed}"
        verified = json.loads((run / "verified.json").read_text(encoding="utf-8"))
        rows = verified["train"]["growth_history"]
        assert [row["step"] for row in rows] == list(range(600, 6000, 100))
        assert all(
            row["after"] == row["before"] + row["added"] - row["pruned"]
            and row["added"] == min(row["eligible"], row["target"] - row["before"])
            for row in rows
        )
        histories.append(rows)
    limited = [
        row["step"]
        for rows in histories
        for row in rows
        if row["eligible"] < row["target"] - row["before"]
    ]
    final = round(mean(rows[-1]["after"] for rows in histories))
    last_eligible = [rows[-1]["eligible"] for rows in histories]
    reset_share = [
        next(row["pruned"] for row in rows if row["step"] == 3100)
        / sum(row["pruned"] for row in rows)
        for rows in histories
    ]
    print(
        scene,
        sum(step <= 3000 for step in limited),
        sum(step >= 3100 for step in limited),
        sum(step >= 5000 for step in limited),
        final,
        (min(last_eligible), max(last_eligible)),
        (min(reset_share), max(reset_share)),
    )
```
