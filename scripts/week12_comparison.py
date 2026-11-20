"""Run and audit the Week 12 matched B00/A05-Abs comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from research_campaign import REPO, command, job, sha, verify, write


STEPS = 7000
SEED = 42
MATTE = REPO / 'results/db3_r2_r3_20261110/matte/pilot'
PUBLIC = REPO / 'results/research_completion_20261027'
ABS = REPO / 'results/a05_absgrad_20261107'
ROOM_MESHES = REPO / 'results/school_completion_replay_20261109/meshes_uniform'
ROOM_EXTRACTOR_SHA256 = '9e31061a0453e76e5a6fc4305728d3128ad5b328b3b203a1707f342a4d2c67ee'
ARMS = ('B00', 'A05_absgrad')
SCENES = ('room', 'matte', 'counter', 'kitchen', 'bonsai')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def spec(scene, arm):
    return job(scene, SEED, method=arm, cap=180000 if scene == 'matte' else 700000)


def data_path(scene):
    return MATTE / 'dataset' if scene == 'matte' else REPO / f'data/{scene}_factor2'


def run_path(root, scene, arm):
    if scene == 'matte' or (scene == 'bonsai' and arm == 'A05_absgrad'):
        campaign = root
    else:
        campaign = PUBLIC if arm == 'B00' else ABS
    return campaign / spec(scene, arm)['id']


def train(root, scene):
    arms = ARMS if scene == 'matte' else ('A05_absgrad',)
    commands = {}
    for arm in arms:
        args = command(spec(scene, arm), root, STEPS)
        if arm == 'A05_absgrad':
            args += ['--strategy.absgrad', '--strategy.grow-grad2d', '0.0008']
        if scene == 'matte':
            dataset = data_path(scene)
            args[args.index('--data-dir') + 1] = str(dataset)
            args += ['--data-workers', '0', '--camera-sequence-path', str(MATTE / 'camera_sequence.json'),
                     '--depth-loss', '--depth-confidence', '--sensor-depth-dir', str(dataset / 'depth'),
                     '--depth-lambda', '0.01']
        else:
            baseline = run_path(root, scene, 'B00')
            replay = root / 'bonsai_camera_sequence.json'
            sequence = read(baseline / 'experiment.json')['camera_sequence']
            assert len(sequence) == STEPS
            if replay.exists():
                assert sequence == read(replay)
            else:
                write(replay, sequence)
            args += ['--data-workers', '0', '--camera-sequence-path', str(replay)]
        commands[arm] = args
    protocol = root / f'{scene}_training_protocol.json'
    if not protocol.exists():
        dataset = data_path(scene)
        inputs = {}
        if scene == 'matte':
            paths = list((dataset / 'sparse').glob('*.bin')) + list((dataset / 'images').glob('*.png'))
            paths += list((dataset / 'depth').glob('*.npz'))
            paths += [dataset / 'experiment_split.json', MATTE / 'camera_sequence.json', MATTE / 'initialization_report.json']
            inputs = {str(path.relative_to(REPO)): sha(path) for path in sorted(paths)}
        sources = ('scripts/week12_comparison.py', 'scripts/research_campaign.py',
                   'examples/simple_trainer.py', 'examples/datasets/colmap.py', 'gsplat/losses.py',
                   'gsplat/rendering.py', 'gsplat/strategy/default.py', 'gsplat/strategy/budget.py',
                   'gsplat/strategy/ops.py')
        write(protocol, dict(seed=SEED, updates=STEPS, commands=commands, input_sha256=inputs,
                             source_sha256={name: sha(REPO / name) for name in sources}))
    else:
        assert read(protocol)['commands'] == commands

    for arm, args in commands.items():
        run = run_path(root, scene, arm)
        if (run / 'verified.json').exists():
            continue
        if run.exists():
            raise FileExistsError(f'Inspect and preserve incomplete run {run}')
        print('TRAIN', scene, arm, flush=True)
        started = time.perf_counter()
        run.parent.mkdir(parents=True, exist_ok=True)
        with run.with_suffix('.log').open('w', encoding='utf-8') as log:
            subprocess.run(args, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        wall = time.perf_counter() - started
        record = verify(root, spec(scene, arm), STEPS)
        record['wall_seconds'] = wall
        write(run / 'verified.json', record)
        print('VERIFIED', scene, arm, record['metrics'], flush=True)


def audit_pair(root, scene):
    import numpy as np
    from PIL import Image
    import yaml

    sys.path.insert(0, str(REPO / 'examples'))
    from datasets.colmap import Dataset, Parser

    result = {}
    configs, sequences, targets = [], [], []
    parser = Parser(str(data_path(scene)), factor=1, normalize=True, test_every=8)
    split = {name: [parser.image_names[i] for i in Dataset(parser, split=name).indices]
             for name in ('train', 'val')}
    assert not set(split['train']).intersection(split['val'])
    for arm in ARMS:
        run = run_path(root, scene, arm)
        record = read(run / 'verified.json')
        cfg = yaml.load((run / 'cfg.yml').read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
        exp = read(run / 'experiment.json')
        assert exp['completed_updates'] == STEPS and exp['seed'] == SEED
        assert (REPO / cfg['data_dir']).resolve() == data_path(scene)
        assert cfg['data_factor'] == '1' and cfg['test_every'] == '8'
        assert exp['initial_sha256'] == record['initial_sha256']
        sequence = exp['camera_sequence']
        assert len(sequence) == STEPS and set(sequence) == set(range(len(split['train'])))
        assert hashlib.sha256(np.asarray(sequence, dtype=np.int64).tobytes()).hexdigest() == record['camera_sha256']
        checkpoint = run / 'ckpts/final_post_update.pt'
        assert sha(checkpoint) == record['final_checkpoint_sha256']
        assert record['metrics'] == read(run / 'stats/val_step6999.json')
        images = sorted((run / 'renders').glob('val_step6999_*.png'))
        assert len(images) == len(split['val'])
        digests = []
        for png in images:
            with Image.open(png) as image:
                assert image.width % 2 == 0
                target = image.crop((0, 0, image.width // 2, image.height))
                digests.append(hashlib.sha256(target.tobytes()).hexdigest())
        strategy = cfg.pop('strategy')
        assert strategy['absgrad'] == ('false' if arm == 'B00' else 'true')
        assert strategy['grow_grad2d'] == ('0.0002' if arm == 'B00' else '0.0008')
        assert {key: strategy[key] for key in ('refine_start_iter', 'refine_stop_iter', 'refine_every', 'reset_every')} == {
            'refine_start_iter': '500', 'refine_stop_iter': '6000', 'refine_every': '100', 'reset_every': '3000',
        }
        if arm == 'A05_absgrad':
            assert strategy['cap_max'] == str(spec(scene, arm)['cap'])
            assert strategy['split_mode'] == 'long_axis' and strategy['clone_policy'] == 'split'
            assert strategy['fallback_rank'] == 'off'
            assert all(row['after'] <= row['target'] for row in record['train']['growth_history'])
            assert sum(row['added'] for row in record['train']['growth_history']) > 0
        assert cfg['depth_loss'] == ('true' if scene == 'matte' else 'false')
        cfg.pop('result_dir')
        # The archived October trainer fixed four workers in the loader and
        # had no sensor-depth setting. The November candidate exposes both.
        cfg.setdefault('data_workers', '4')
        cfg.setdefault('sensor_depth_dir', 'null')
        cfg.setdefault('depth_confidence', 'false')
        cfg.setdefault('camera_sequence_path', 'null')
        cfg.setdefault('sensor_alpha_lambda', '0.0')
        cfg.setdefault('surface_lambda', '0.0')
        assert cfg['surface_lambda'] == cfg['sensor_alpha_lambda'] == '0.0'
        if scene == 'matte':
            assert cfg['depth_confidence'] == 'true' and cfg['depth_lambda'] == '0.01'
            assert Path(cfg['sensor_depth_dir']) == data_path(scene) / 'depth'
        configs.append(cfg)
        sequences.append(sequence)
        targets.append(digests)
        result[arm] = dict(source=str(run.relative_to(REPO)), strategy=strategy, metrics=record['metrics'],
                           metrics_after_first_update=read(run / 'stats/val_step0000.json'),
                           train_seconds=record['train']['ellipse_time'], wall_seconds=record['wall_seconds'],
                           initial_sha256=record['initial_sha256'], camera_sha256=record['camera_sha256'],
                           checkpoint_sha256=record['final_checkpoint_sha256'])

    differences = {key: [configs[0].get(key), configs[1].get(key)] for key in configs[0].keys() | configs[1].keys()
                   if configs[0].get(key) != configs[1].get(key)}
    deviations = {}
    if scene == 'bonsai':
        # Camera replay keeps the recorded order even with the lower-memory loader.
        assert differences.pop('data_workers') == ['4', '0']
        assert Path(differences.pop('camera_sequence_path')[1]) == root / 'bonsai_camera_sequence.json'
        assert sequences[0] == read(root / 'bonsai_camera_sequence.json')
        deviations = dict(data_workers=[4, 0], camera_sampler=['seeded loader', 'exact replay of B00'])
        # Surface loss is disabled for both arms.
        assert configs[0]['surface_lambda'] == configs[1]['surface_lambda'] == '0.0'
        differences.pop('surface_start', None)
    if differences:
        raise ValueError(f'{scene} meaningful non-strategy config differences: {differences}')
    assert sequences[0] == sequences[1] and targets[0] == targets[1]
    assert result['B00']['initial_sha256'] == result['A05_absgrad']['initial_sha256']
    baseline, candidate = (result[arm]['metrics_after_first_update'] for arm in ARMS)
    assert baseline['num_GS'] == candidate['num_GS']
    assert all(abs(baseline[key] - candidate[key]) < 1e-5 for key in ('psnr', 'ssim', 'lpips'))
    if scene == 'matte':
        assert sequences[0] == read(MATTE / 'camera_sequence.json')
        protocol = read(root / 'matte_training_protocol.json')
        assert all(sha(REPO / name) == digest for name, digest in protocol['input_sha256'].items())
    return dict(scene=scene, split=split, loader_differences=deviations,
                ordered_training_frames=[split['train'][i] for i in sequences[0]],
                validation_rgb_sha256=targets[0], arms=result)


def mesh_path(root, scene, arm):
    if scene == 'room':
        return ROOM_MESHES / ('baseline' if arm == 'B00' else 'reduced')
    return root / 'meshes' / scene / arm


def extract_pair(root, scene, pair):
    if scene == 'room':
        archived_extractor = ROOM_MESHES.parent / 'extraction_evidence/gaussians_to_mesh.py'
        assert sha(archived_extractor) == ROOM_EXTRACTOR_SHA256
    reports = []
    for arm in ARMS:
        folder = mesh_path(root, scene, arm)
        if not (folder / 'report.json').exists():
            if scene == 'room' or folder.exists():
                raise FileNotFoundError(f'Expected a complete extraction at {folder}')
            folder.parent.mkdir(parents=True, exist_ok=True)
            print('EXTRACT', scene, arm, flush=True)
            with folder.with_suffix('.log').open('w', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-u', 'scripts/gaussians_to_mesh.py',
                                '--checkpoint', str(run_path(root, scene, arm) / 'ckpts/final_post_update.pt'),
                                '--data', str(data_path(scene)), '--output', str(folder),
                                '--views', '64', '--downsample', '4', '--voxel', '0.01', '--max-resolution', '192'],
                               cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        record = read(folder / 'report.json')
        assert sha(folder / 'raw_mesh.ply') == record['mesh_sha256']
        assert record['checkpoint_sha256'] == pair['arms'][arm]['checkpoint_sha256']
        assert record['gaussian_count'] == pair['arms'][arm]['metrics']['num_GS']
        reports.append(record)
    for key in ('parser_transform', 'scene_scale', 'tsdf_backend', 'resolution', 'box_origin', 'box_length',
                'voxel_length', 'sdf_trunc', 'alpha_threshold'):
        assert reports[0][key] == reports[1][key], (scene, key)
    ids = [[row['train_index'] for row in rec['training_views']] for rec in reports]
    assert ids[0] == ids[1] and len(ids[0]) == min(64, len(pair['split']['train']))
    pair['extraction_views'] = [pair['split']['train'][i] for i in ids[0]]
    for arm, record in zip(ARMS, reports):
        pair['arms'][arm]['mesh'] = {**record, 'folder': str(mesh_path(root, scene, arm).relative_to(REPO))}


def edit_pair(root, scene):
    folder = root / 'review' / scene
    pair = read(root / f'{scene}_pair.json')
    blender = REPO / '.tools/blender-4.5.4-windows-x64/blender.exe'
    for arm in ARMS:
        document = folder / f'{arm}_edited.blend'
        output = folder / f'{arm}_edit_verification.json'
        if not output.exists():
            for action in ('create', 'verify'):
                if action == 'create' and document.exists():
                    continue
                before = ['--factory-startup'] if action == 'create' else ['--disable-autoexec', str(document)]
                with (folder / f'{arm}_blender_{action}.log').open('a', encoding='utf-8') as log:
                    subprocess.run([str(blender), '-b', *before, '--python-exit-code', '1', '--python',
                                    'scripts/week12_blender_edit.py', '--', action, str(folder), arm],
                                   cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = read(output)
        assert sha(document) == report['document_sha256'] and report['verified_after_reopen']
        mesh = pair['arms'][arm]['mesh']
        assert report['triangles'] == mesh['triangles'] and report['vertices'] == mesh['vertices']
        print('EDIT VERIFIED', scene, arm, report['selected_vertices'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('train', 'extract', 'edit'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--scene', required=True, choices=SCENES)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == 'train':
        if args.scene not in ('matte', 'bonsai'):
            parser.error('Only matte and Bonsai require new training')
        train(root, args.scene)
    elif args.stage == 'extract':
        pair = audit_pair(root, args.scene)
        extract_pair(root, args.scene, pair)
        write(root / f'{args.scene}_pair.json', pair)
        print('AUDITED', args.scene, flush=True)
    else:
        edit_pair(root, args.scene)


if __name__ == '__main__':
    main()
