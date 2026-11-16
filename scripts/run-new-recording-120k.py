"""Run cleaning/reconstruction, validated depth training, and reload/render QA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recording', type=Path, default=Path('20261030_111354.db3'))
    ap.add_argument('--data', type=Path, default=Path('data/realsense_20261030_111354_clean'))
    ap.add_argument('--result', type=Path, default=Path('results/realsense_20261030_111354_clean'))
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--prepared', action='store_true', help='Use separately completed preparation; still validate it')
    ap.add_argument('--dataset-name', choices=['auto', 'dataset', 'rgbd/dataset'], default='auto')
    args = ap.parse_args()
    os.chdir(REPO)
    out = args.result.resolve(); out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', PYTHONUNBUFFERED='1',
               TORCH_HOME=str(REPO/'.cache/torch'), TORCH_EXTENSIONS_DIR=str(REPO/'.cache/torch_extensions'),
               TORCH_CUDA_ARCH_LIST='7.5', NUM_CHANNELS='3,4', MAX_JOBS='2', PYTHONPATH=str(REPO))
    def write(name, value):
        (out/name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    def run(stage, command):
        marker = out/(stage+'_success.json')
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved['command'] != list(map(str, command)):
                raise ValueError(f'Changed command for completed {stage}; use a different result directory')
            print('SKIP completed', stage, flush=True)
            return
        write('status.json', dict(stage=stage, status='running', command=list(map(str, command)), pid=os.getpid()))
        print('RUN', stage, flush=True)
        start = time.time()
        with (out/(stage+'.log')).open('a', encoding='utf-8') as log:
            result = subprocess.run(list(map(str, command)), env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write('status.json', dict(stage=stage, status='failed', exit_code=result.returncode))
            raise SystemExit(result.returncode)
        write(marker.name, dict(elapsed_seconds=time.time()-start, command=list(map(str, command))))
    if not args.prepared:
        for stage in ('clean', 'reconstruct'):
            run(stage, [sys.executable, '-u', 'scripts/clean-realsense-scene.py', stage, '--recording', args.recording, '--output', args.data])
    run('clean_audit', [sys.executable, '-u', 'scripts/audit-clean-realsense.py', '--root', args.data])
    report = json.loads((args.data/'reconstruction_report.json').read_text())
    selected = args.dataset_name
    selection_reason = 'Explicit dataset requested'
    if selected == 'auto':
        selected = 'dataset'
        selection_reason = 'RGB SfM meets coverage gate'
        if report['coverage_fraction'] < .9:
            run('recover_rgbd', [sys.executable, '-u', 'scripts/recover-clean-rgbd.py', '--root', args.data])
            run('bridge_rgbd', [sys.executable, '-u', 'scripts/bridge-clean-rgbd.py', '--root', args.data])
            metric = json.loads((args.data/'rgbd/report.json').read_text())
            if (metric['coverage_fraction'] > report['coverage_fraction'] and
                metric['translation_residual_percentiles_m'][1] < .08 and
                metric['rotation_residual_percentiles_degrees'][1] < 3.):
                selected = 'rgbd/dataset'
                selection_reason = 'Metric RGB-D improves coverage and passes p90 pose-constraint residual gates'
            else:
                selection_reason = 'RGB-D did not improve coverage with acceptable residuals; retained RGB SfM'
    dataset = args.data/selected
    if selected == 'dataset':
        if report['registered_frames'] < 30 or report['mean_reprojection_error'] > 2:
            raise RuntimeError('Reconstruction quality gate failed; inspect before training')
        run('depth', [sys.executable, '-u', 'scripts/clean-realsense-scene.py', 'depth', '--output', args.data])
    else:
        metric = json.loads((args.data/'rgbd/report.json').read_text())
        if (metric['registered_frames'] < 30 or metric['translation_residual_percentiles_m'][1] >= .08 or
            metric['rotation_residual_percentiles_degrees'][1] >= 3.):
            raise RuntimeError('RGB-D pose quality gate failed')
    write('dataset_selection.json', dict(dataset=str(dataset.resolve()), method=selected, reason=selection_reason))
    run('validate', [sys.executable, '-u', 'scripts/clean-realsense-scene.py', 'validate', '--output', args.data, '--dataset-name', selected])
    run('train', ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', 'scripts/test-small-dataset.ps1',
          '-Steps', args.steps, '-DataDir', dataset, '-ResultDir', out/'train',
          '-ShDegree', '3', '-ShDegreeInterval', '1000', '-RefineStart', '500', '-RefineStop', '6000',
          '-ResetEvery', '3000', '-Strategy', 'default',
          '-RecordExperiment', '-DataWorkers', '0', '-NoCacheImages',
          '-SensorDepthDir', dataset/'sensor_depth', '-DepthWeight', '.01'])
    # Import the same portable compiler/CUDA environment as the training launcher.
    batch = REPO/'.tools/msvc/setup_x64.bat'
    text = subprocess.check_output(f'call "{batch}" >nul && set', shell=True, text=True)
    for line in text.splitlines():
        if '=' in line and not line.startswith('='):
            k, v = line.split('=', 1); env[k] = v
    env.update(CUDA_HOME=str(REPO/'.tools/cuda-12.8'), CUDA_PATH=str(REPO/'.tools/cuda-12.8'))
    env['PATH'] = str(REPO/'.venv/Scripts')+os.pathsep+str(REPO/'.tools/cuda-12.8/bin')+os.pathsep+env['PATH']
    checkpoint = out/'train/ckpts/final_post_update.pt'
    run('preview', [sys.executable, '-u', 'scripts/inspect-realsense-scene.py', '--ckpt', checkpoint,
          '--data-dir', dataset, '--output', out/'preview', '--width', '848',
          '--benchmark-context', 'Existing previous-recording viewers remain open; no other training job launched by this runner'])
    write('status.json', dict(stage='provenance', status='running'))
    manifest = json.loads((args.data/'capture_manifest.json').read_text())
    with Path(manifest['source']).open('rb') as stream:
        source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    initial = args.data/'source_hash_initial.json'
    if initial.exists():
        initial_hash = json.loads(initial.read_text(encoding='utf-8-sig'))['Hash'].lower()
        if initial_hash != source_hash:
            raise RuntimeError('Raw recording SHA256 changed during pipeline')
    else:
        initial_hash = None
    files = ['scripts/clean-realsense-scene.py', 'scripts/audit-clean-realsense.py',
             'scripts/run-clean-realsense.py', 'scripts/recover-clean-rgbd.py', 'scripts/bridge-clean-rgbd.py',
             'scripts/recover-rgbd-scene.py', 'scripts/sparse-rgbd-pose-graph.py', 'scripts/test-small-dataset.ps1',
             'scripts/inspect-realsense-scene.py', 'examples/simple_trainer.py', 'examples/datasets/colmap.py']
    write('provenance.json', dict(source_sha256=source_hash,
          initial_source_sha256=initial_hash, source_hash_unchanged=source_hash == initial_hash if initial_hash else None,
          code_sha256={s: hashlib.sha256((REPO/s).read_bytes()).hexdigest() for s in files},
          gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip(),
          source=manifest['source'], completed_updates=args.steps))
    write('viewer.json', dict(checkpoint=str(checkpoint), dataset=str(dataset.resolve()), camera_index=0))
    write('status.json', dict(stage='complete', status='complete', updated_epoch=time.time()))
    print('CLEAN_SCENE_COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
