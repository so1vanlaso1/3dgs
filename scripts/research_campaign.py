"""Resume-safe, prospectively specified A04 replication and ablation campaign."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
SCENES = ('room', 'counter', 'kitchen', 'bonsai')
SEEDS = (42, 43, 44)
SOURCES = ('examples/simple_trainer.py', 'gsplat/strategy/budget.py',
           'gsplat/strategy/ops.py', 'gsplat/strategy/default.py',
           'scripts/research_campaign.py', 'scripts/run-research.ps1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def job(scene, seed, method='A04', cap=700000, mode='long_axis', policy='split',
        distance=.45, opacity=.6, phase='replication'):
    return dict(scene=scene, seed=seed, method=method, cap=cap, mode=mode,
                policy=policy, distance=distance, opacity=opacity, phase=phase,
                id=f'{phase}/{scene}/{method}_s{seed}')


def plan():
    jobs = [job(s, seed, m) for s in SCENES for seed in SEEDS for m in ('B00','A04')]
    for seed in SEEDS:
        for mode, policy, label in [('random','clone','random_clone'),
                                    ('random','split','random_all'),
                                    ('long_axis','clone','LAS_clone')]:
            jobs.append(job('room', seed, label, mode=mode, policy=policy, phase='ablation'))
    for cap in (400000, 1000000):
        jobs.append(job('room', 42, f'A04_cap{cap}', cap=cap, phase='budgets'))
    for d, o in ((.30,.4), (.30,.8), (.50,.4), (.50,.8)):
        jobs.append(job('room', 42, f'LAS_d{d:.2f}_o{o:.2f}', distance=d, opacity=o, phase='tuning'))
    return jobs


def command(j, root, steps):
    default = j['method'] == 'B00'
    args = [sys.executable, '-u', 'examples/simple_trainer.py',
            'default' if default else 'budget', '--seed', str(j['seed']),
            '--independent-sampling', '--record-experiment',
            '--data-dir', f'data/{j["scene"]}_factor2', '--data-factor', '1',
            '--result-dir', str(root / j['id']), '--max-steps', str(steps),
            '--eval-steps', '1', str(steps), '--save-steps', str(steps),
            '--save-ply', '--ply-steps', str(steps), '--disable-viewer', '--disable-video',
            '--batch-size','1','--packed','--sh-degree','3','--sh-degree-interval','1000',
            '--cache-images', '--strategy.refine-start-iter', '500' if steps == 7000 else '10',
            '--strategy.refine-stop-iter', '6000' if steps == 7000 else str(steps-10),
            '--strategy.refine-every', '100' if steps == 7000 else '20',
            '--strategy.reset-every', '3000']
    if not default:
        args += ['--strategy.cap-max', str(j['cap']), '--strategy.error-weight','1',
                 '--strategy.edge-weight','.25', '--strategy.split-mode',j['mode'],
                 '--strategy.clone-policy',j['policy'],
                 '--strategy.split-distance', str(j['distance']),
                 '--strategy.opacity-reduction',str(j['opacity'])]
    return args


def verify(root, j, steps):
    import numpy as np
    import torch
    run = root / j['id']
    exp = json.loads((run/'experiment.json').read_text())
    metrics = json.loads((run/f'stats/val_step{steps-1:04d}.json').read_text())
    train = json.loads((run/f'stats/train_step{steps-1:04d}_rank0.json').read_text())
    ckpt = torch.load(run/'ckpts/final_post_update.pt', map_location='cpu', weights_only=True)
    tensors = ckpt['splats']
    count = len(tensors['means'])
    assert all(torch.isfinite(t).all() for t in tensors.values())
    assert count == metrics['num_GS'] == exp['final_count']
    assert exp['completed_updates'] == steps and len(exp['camera_sequence']) == steps
    assert np.isfinite([metrics[k] for k in ('psnr','ssim','lpips')]).all()
    if j['method'] != 'B00':
        assert count <= j['cap']
        assert all(h['after'] <= h['target'] for h in train['growth_history'])
    return dict(job=j, metrics=metrics, initial_sha256=exp['initial_sha256'],
                camera_sha256=exp['camera_sequence_sha256'],
                train=train, final_checkpoint_sha256=sha(run/'ckpts/final_post_update.pt'),
                checkpoint_bytes=(run/'ckpts/final_post_update.pt').stat().st_size,
                parameter_bytes=sum(t.numel()*t.element_size() for t in tensors.values()),
                checks_pass=True)


def execute(j, root, steps):
    run = root / j['id']
    completion = run/'verified.json'
    if completion.exists():
        return json.loads(completion.read_text())
    if run.exists():
        raise RuntimeError(f'Incomplete run must be inspected before retry: {run}')
    run.parent.mkdir(parents=True, exist_ok=True)
    args = command(j, root, steps)
    print('START', j['id'], flush=True)
    write(root/'status.json', dict(status='training', current=j, started_utc=datetime.now(timezone.utc).isoformat()))
    tic = time.perf_counter()
    with (run.parent/f'{run.name}.log').open('w',encoding='utf-8') as log:
        result = subprocess.run(args,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        write(root/'status.json',dict(status='failed',current=j,exit_code=result.returncode))
        raise RuntimeError(f'Failed {j["id"]}; inspect its log')
    record = verify(root,j,steps)
    record['wall_seconds'] = time.perf_counter()-tic
    write(completion,record)
    print('DONE', j['id'], json.dumps(record['metrics']), flush=True)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    os.chdir(REPO)
    root=args.root
    root.mkdir(parents=True,exist_ok=True)
    jobs=plan()
    steps=7000
    if args.smoke:
        jobs=[job('room',42,cap=150000)]
        steps=60
    protocol_path=root/'protocol.json'
    if not protocol_path.exists():
        evidence=root/'evidence'
        hashes={}
        for name in SOURCES:
            target=evidence/'source'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(name,target)
            hashes[name]=sha(name)
        write(protocol_path, dict(created_utc=datetime.now(timezone.utc).isoformat(),
            local_date='2026-10-27',seeds=SEEDS,steps=steps,jobs=jobs,source_sha256=hashes,
            ai_usage_sha256=sha('AI_USAGE.md'),
            tuning_policy='Room held-out views are development only; choose highest PSNR among guardrail-compliant settings, then confirm on seeds43/44 and Counter seed42; no untouched test claim.',
            limitations=['Budget sweep: Room seed42 only; component ablation: Room three seeds.',
                         'Three-seed confidence intervals are descriptive and imprecise.',
                         'LAS operator bundles placement, scale and opacity; factorial isolates it from clone replacement.',
                         'Historical runs excluded from new seed averages because camera sampling changed.']))
    protocol=json.loads(protocol_path.read_text())
    assert protocol['jobs']==jobs and protocol['steps']==steps
    assert all(sha(p)==h for p,h in protocol['source_sha256'].items()), 'Core source changed after capture'
    records=[execute(j,root,steps) for j in jobs]
    if not args.smoke:
        baseline=next(r for r in records if r['job']['id']=='replication/room/A04_s42')
        candidates=[r for r in records if r['job']['phase']=='tuning']+[baseline]
        valid=[r for r in candidates if r['metrics']['ssim']>=baseline['metrics']['ssim']-.005
               and r['metrics']['lpips']<=baseline['metrics']['lpips']+.010]
        winner=max(valid,key=lambda r:r['metrics']['psnr'])
        write(root/'tuning_selection.json',dict(winner=winner['job'],
              selection_metrics=winner['metrics'],selected_after_all_grid_runs=True,
              policy=protocol['tuning_policy']))
        if winner['job']['phase']=='tuning':
            for scene,seed in [('room',43),('room',44),('counter',42)]:
                j=job(scene,seed,'tuned_A04',distance=winner['job']['distance'],
                      opacity=winner['job']['opacity'],phase='confirmation')
                records.append(execute(j,root,steps))
    write(root/'all_runs.json',records)
    assert sha('AI_USAGE.md')==protocol['ai_usage_sha256']
    if not args.smoke:
        subprocess.run([sys.executable,'scripts/analyze_research.py','--root',str(root)],check=True)
        write(root/'status.json',dict(status='ccd_mechanism',completed_a04_runs=len(records)))
        subprocess.run([sys.executable,'-u','scripts/ccd_mechanism.py','--root',str(root/'ccd')],check=True)
        subprocess.run([sys.executable,'scripts/analyze_research.py','--root',str(root)],check=True)
    write(root/'status.json',dict(status='complete',completed_a04_runs=len(records),smoke=args.smoke))
    print('CAMPAIGN_COMPLETE',flush=True)


if __name__=='__main__':
    main()
