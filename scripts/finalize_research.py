"""Finalize completed research records after the user's request to skip the rest.

CPU analysis only: this script never trains or evaluates a model on the GPU.
"""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_research import crop_analysis, interval


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('results/research_completion_20261027'))
    args=cli.parse_args(); root=args.root
    protocol=json.loads((root/'protocol.json').read_text())
    for path,expected in protocol['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected, f'Training source changed: {path}'
    records=[json.loads(p.read_text()) for p in sorted(root.glob('*/*/*/verified.json'))]
    assert records and all(r['checks_pass'] for r in records)
    index={(r['job']['scene'],r['job']['seed'],r['job']['method']):r for r in records}
    statistics={}; paired=[]; used=set()
    for scene in ('room','counter','kitchen','bonsai'):
        rows=[]
        for seed in (42,43,44):
            b=index.get((scene,seed,'B00')); a=index.get((scene,seed,'A04'))
            if a is None or b is None: continue
            assert a['initial_sha256']==b['initial_sha256']
            assert a['camera_sha256']==b['camera_sha256']
            used.update((a['job']['id'],b['job']['id']))
            row=dict(scene=scene,seed=seed,
                **{k:a['metrics'][k]-b['metrics'][k] for k in ('psnr','ssim','lpips')},
                reduction=100*(1-a['metrics']['num_GS']/b['metrics']['num_GS']),
                baseline_count=b['metrics']['num_GS'],a04_count=a['metrics']['num_GS'])
            rows.append(row); paired.append(row)
        statistics[scene]={}
        for key in ('psnr','ssim','lpips','reduction'):
            s=interval([r[key] for r in rows])
            if s['n']==1: s['sd']=None
            statistics[scene][key]=s
    unpaired=[r['job']['id'] for r in records if r['job']['id'] not in used]
    completed={r['job']['id'] for r in records}
    skipped=[j for j in protocol['jobs'] if j['id'] not in completed]
    old_status=json.loads((root/'status.json').read_text())
    interrupted=old_status.get('interrupted_run') or old_status.get('current',{}).get('id')
    final_status=dict(status='finished_early_at_user_request',
        finished_local=old_status.get('finished_local') or datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7))).isoformat(),
        completed_training_runs=len(records),completed_pairs=len(paired),
        complete_three_seed_scenes=[s for s in statistics if statistics[s]['psnr']['n']==3],
        completed_unpaired_runs=unpaired,interrupted_run=interrupted,
        skipped_scheduled_training_runs=skipped,full_ccd_study='skipped',
        note='User requested skipping all unfinished work. No automatic continuation remains.')
    write(root/'status.json',final_status)
    write(root/'all_runs.json',records)
    write(root/'statistics.json',dict(paired=statistics,paired_runs=paired,unpaired_runs=unpaired))
    with (root/'runs.csv').open('w',newline='',encoding='utf-8') as stream:
        rows=[dict(id=r['job']['id'],scene=r['job']['scene'],seed=r['job']['seed'],method=r['job']['method'],
                   **r['metrics'],wall_seconds=r['wall_seconds'],checkpoint_bytes=r['checkpoint_bytes']) for r in records]
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (root/'paired_runs.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
    figures=root/'figures'; figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,key,title,ylabel in zip(axes,('psnr','reduction'),
                                 ('Paired held-out PSNR gain','Gaussian-count reduction'),
                                 ('A04 minus B00 (dB)','Fewer Gaussians (%)')):
        for i,(scene,stats) in enumerate(statistics.items()):
            s=stats[key]
            if s['ci95'] is not None:
                ax.errorbar(i,s['mean'],yerr=[[s['mean']-s['ci95'][0]],[s['ci95'][1]-s['mean']]],fmt='o',capsize=5,color='#2166ac')
            else: ax.scatter([i],[s['mean']],facecolors='none',edgecolors='#2166ac',s=65)
            offsets=np.linspace(-.07,.07,s['n']) if s['n']>1 else np.zeros(1)
            ax.scatter(np.full(s['n'],i)+offsets,s['values'],s=22,color='#dc7633',zorder=3)
        ax.axhline(0,color='gray',lw=1); ax.grid(axis='y',alpha=.2)
        ax.set_xticks(range(4),[f'{scene.title()}\nn={statistics[scene][key]["n"]}' for scene in statistics])
        ax.set_ylabel(ylabel); ax.set_title(title)
    fig.suptitle('Completed pairs only: seeds, means and 95% t intervals; Bonsai has one pair')
    fig.savefig(figures/'completed_comparisons.png',dpi=180); plt.close(fig)
    crops=crop_analysis(root,records)
    write(root/'statistics.json',dict(paired=statistics,paired_runs=paired,unpaired_runs=unpaired,crops=crops))
    lines=['# Completed A04 experiments — final report','',
        '**Finished early at the user’s request on 27 October 2026. All remaining runs were stopped or skipped.**','',
        f'{len(records)} complete training runs form {len(paired)} matched pairs. Room, Counter and Kitchen have three pairs each; Bonsai has one. A completed Bonsai B00 seed43 run is retained as an unpaired result. Its interrupted A04 counterpart is excluded from all comparisons.','',
        '## Completed versus skipped','',
        '| Work | Final status |','|---|---|',
        '| Several seeds | Completed on Room, Counter and Kitchen (42/43/44). Bonsai has seed42 only. |',
        '| Split/clone component ablation | Code prepared; experiments skipped. |',
        '| Several Gaussian budgets | Runner prepared; sweep skipped. |',
        '| Image crops and failure analysis | Completed for all four scenes at seed42, with exhaustive window/view errors. Historical visual review also completed. |',
        '| Split-distance/opacity tuning | Grid prepared; experiments skipped. No better setting claimed. |',
        '| CCD-GS mechanism | Snapshot/probe/branch harness smoke-tested; full scientific study skipped. No CCD-GS efficacy conclusion. |','',
        '## Matched protocol','',
        'Fresh 7,000-update runs at the existing official half-resolution data, same SfM initialization per pair, same train/held-out split, optimizer/loss/SH and density schedule. B00 uses DefaultStrategy without a count cap. A04 uses the existing 700,000 cap, gradient/error/edge weights 1/1/0.25 and LAS distance 0.45 / opacity multiplier 0.6. Pose, appearance and post-processing optimization are disabled.','',
        'The experiment adds an independent DataLoader generator. Initial-parameter and complete 7,000-camera-sequence hashes match in every included pair. Historical runs used a different sampler and are not pooled into the new seed averages.','',
        '## Paired results','',
        'Deltas are A04 minus B00. Lower LPIPS is better. Three-seed values show mean ± sample SD and paired Student-t 95% intervals (df=2). The intervals are imprecise with only three seeds and do not establish population-level generalization. A single Bonsai pair has no uncertainty estimate.','',
        '| Scene | Pairs | PSNR delta (dB), mean ± SD | PSNR 95% interval | SSIM delta | LPIPS delta | Fewer Gaussians |',
        '|---|---:|---:|---|---:|---:|---:|']
    for scene,stats in statistics.items():
        ps=stats['psnr']; ci=ps['ci95']
        gain=f'{ps["mean"]:+.4f}' if ps['sd'] is None else f'{ps["mean"]:+.4f} ± {ps["sd"]:.4f}'
        ci_text='Not estimated (one pair)' if ci is None else f'[{ci[0]:+.4f}, {ci[1]:+.4f}]'
        lines.append(f'| {scene.title()} | {ps["n"]} | {gain} | {ci_text} | {stats["ssim"]["mean"]:+.5f} | {stats["lpips"]["mean"]:+.5f} | {stats["reduction"]["mean"]:.2f}% |')
    lines+=['','![Completed comparisons and uncertainty](figures/completed_comparisons.png)','',
        'All ten completed pairs improved mean held-out PSNR and reduced Gaussian count. Room and Counter have positive paired PSNR intervals in this small study. Kitchen’s PSNR interval crosses zero; its SSIM changes are small and mixed. The Bonsai result remains one-seed evidence. These results support a scoped quality/size finding for the combined A04 method; they do not isolate its components.','',
        '## Individual matched pairs','',
        '| Scene | Seed | PSNR delta | SSIM delta | LPIPS delta | B00 count | A04 count | Reduction |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in paired:
        lines.append(f'| {r["scene"]} | {r["seed"]} | {r["psnr"]:+.4f} | {r["ssim"]:+.5f} | {r["lpips"]:+.5f} | {r["baseline_count"]:,} | {r["a04_count"]:,} | {r["reduction"]:.2f}% |')
    lines+=['','## Image crops and failures','',
        'New-campaign seed42 images are scanned exhaustively with 192×192 windows at stride 96, including boundaries. Each scene shows largest gain, largest regression, largest remaining residual, and largest gain among interior high-edge-density windows. These deliberately selected examples use image outcomes; they are illustrative. All windows, views, coordinates and selection rules are in `qualitative/`. Error heatmaps use the same scale. Windows overlap and are not independent statistical replicates.','',
        'The errors below are recomputed from saved 8-bit PNGs; trainer metrics use floating-point renders. Improvement in average PSNR does not imply every view or region improves.','',
        '| Scene | Full views with higher A04 RGB MSE | Windows with higher A04 RGB MSE |',
        '|---|---:|---:|']
    for scene,s in crops.items(): lines.append(f'| {scene.title()} | {s["views_worse"]}/{s["views"]} | {s["tiles_worse"]}/{s["tiles"]} |')
    for scene in crops: lines+=['',f'### {scene.title()}','',f'![{scene} selected crops](qualitative/{scene}_contact_sheet.png)']
    lines+=['','[Written visual review of these completed runs](VISUAL_REVIEW.md). A separate [historical render review](historical_review/VISUAL_REVIEW.md) is retained; its observations are separate from the new results.','',
        '## Verification and artifacts','',
        '- All 21 included runs completed their full 7,000 updates and passed the recorded finite-checkpoint, final-count and applicable budget-history checks. Every included pair has matching initialization and camera-sequence hashes.',
        '- The Room seed42 final post-update checkpoint was re-evaluated over all 39 held-out views. PSNR differed by 0.0000026 dB; SSIM/LPIPS differences were below 0.00000003. Other new checkpoint reloads were skipped when the user stopped remaining work.',
        '- Targeted split/budget/sampler tests: 23 passed, with one existing pytest configuration warning. CCD smoke studies verify state isolation and branch plumbing only; their tiny horizons are excluded from research claims.',
        '- `runs.csv` contains all completed runs; `paired_runs.csv` and `statistics.json` contain full-precision paired results. Per-run `verified.json`, configs, logs, checkpoints, PLYs and renders remain available.',
        '- `protocol.json` preserves the originally planned matrix. `status.json` records the user-requested cutoff, skipped jobs and interrupted run. Source snapshots, hashes, environment versions and validation summaries are under `evidence/`.',
        '- Training used equal optimizer-update counts, with variable computation and system load. Wall times are recorded but do not establish a controlled speedup. All four natural scenes are indoor scenes. No component-cause, tuning, budget-curve or CCD-GS efficacy claim is made.','',
        '## Completed-run appendix','',
        '| Run | PSNR | SSIM | LPIPS | Gaussians |', '|---|---:|---:|---:|---:|']
    for r in records:
        m=r['metrics']; lines.append(f'| {r["job"]["id"]} | {m["psnr"]:.4f} | {m["ssim"]:.5f} | {m["lpips"]:.5f} | {m["num_GS"]:,} |')
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'PROGRESS.md').write_text('# Campaign finished at the user’s request\n\nNo further training is running.\n\n[Final completed-results report](REPORT.md)\n\n'+
        f'{len(records)} completed training runs; {len(paired)} matched pairs; {len(skipped)} scheduled jobs stopped/skipped. Full CCD-GS study skipped.\n',encoding='utf-8')
    if interrupted and (root/interrupted).exists():
        write(root/interrupted/'INTERRUPTED_AT_USER_REQUEST.json',dict(reason='User requested skipping all unfinished work.',excluded_from_comparisons=True))
    print(json.dumps(dict(completed=len(records),pairs=len(paired),statistics=statistics,crops=crops),indent=2))


if __name__=='__main__': main()
