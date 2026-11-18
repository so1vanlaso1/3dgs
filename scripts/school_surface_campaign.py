"""Frozen 2x2 reduction x surface-prior ablation, Room seed 42."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from research_campaign import REPO, command, job, sha, verify, write


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    steps = 60 if a.smoke else 7000
    if not a.smoke:
        archived = REPO/'results/research_completion_20261027/replication/room/B00_s42/experiment.json'
        sequence = json.loads(archived.read_text())['camera_sequence']
        assert len(sequence) == steps
        write(root/'camera_sequence.json', sequence)
    jobs, commands = [], []
    for reduced, surface in ((False, False), (False, True), (True, False), (True, True)):
        if not a.smoke and not surface:
            continue
        label = ('reduced' if reduced else 'baseline') + ('_surface' if surface else '')
        spec = job('room', 42, 'A05' if reduced else 'B00', cap=150000 if a.smoke else 700000)
        spec['id'] = 'replication/room/' + label
        args = command(spec, root, steps)
        # Avoid four Windows workers duplicating the decoded-image cache.
        args += ['--data-workers', '0']
        if not a.smoke:
            args += ['--camera-sequence-path', str(root/'camera_sequence.json')]
        if reduced:
            args += ['--strategy.absgrad', '--strategy.grow-grad2d', '.0008']
        if surface:
            args += ['--surface-lambda', '.01', '--surface-start', '20' if a.smoke else '3000']
        jobs.append(spec)
        commands.append(args)
    sources = ['examples/simple_trainer.py', 'gsplat/surface.py', 'gsplat/strategy/budget.py',
               'gsplat/strategy/ops.py', 'scripts/school_surface_campaign.py',
               'scripts/research_campaign.py', 'scripts/gaussians_to_mesh.py']
    for name in sources:
        dest = root / 'evidence/source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, dest)
    write(root/'protocol.json', dict(steps=steps, jobs=jobs, commands=commands,
        source_sha256={n:sha(REPO/n) for n in sources}, surface_lambda=.01,
        intent='Ablate rendering and surface proxies; no geometric ground truth available for Room.',
        selection='Fixed weight, no tuning. Reject replacement if PSNR loss > .2 dB, SSIM loss > .005 or LPIPS increase > .01 versus matching no-prior arm.',
        limitations='One indoor scene and one seed; exposed development split; smoothness is not geometry accuracy. RGB+ED is enabled in prior arms.'))
    records = {}
    if not a.smoke:
        for label, archive in (
            ('baseline','results/research_completion_20261027/replication/room/B00_s42'),
            ('reduced','results/a05_absgrad_20261107/replication/room/A05_absgrad_s42')):
            path = REPO/archive/'verified.json'
            rec = json.loads(path.read_text())
            assert sha(REPO/archive/'ckpts/final_post_update.pt') == rec['final_checkpoint_sha256']
            rec['archived_control'] = str(path)
            records[label] = rec
        assert records['baseline']['initial_sha256'] == records['reduced']['initial_sha256']
        assert records['baseline']['camera_sha256'] == records['reduced']['camera_sha256']
    for spec, args in zip(jobs, commands):
        label = spec['id'].split('/')[-1]
        write(root/'status.json', dict(state='training', arm=label))
        print('START', label, flush=True)
        start = time.perf_counter()
        with (root/(label+'.log')).open('w',encoding='utf8') as log:
            subprocess.run(args, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        rec = verify(root, spec, steps)
        rec['wall_seconds'] = time.perf_counter()-start
        if records:
            first = next(iter(records.values()))
            assert all(rec[k] == first[k] for k in ('initial_sha256','camera_sha256'))
        write(root/spec['id']/'verified.json', rec)
        records[label] = rec
        write(root/'summary.json', records)
        print('DONE',label,rec['metrics'],flush=True)
    write(root/'status.json',dict(state='training_complete',smoke=a.smoke))


if __name__ == '__main__':
    main()
