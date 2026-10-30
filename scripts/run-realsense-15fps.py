"""Sequential, logged 15 FPS experiment. Resume only successfully finished stages."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

repo=Path(__file__).resolve().parents[1]
root=repo/'data/realsense_20261029_140739_15fps'
out=repo/'results/realsense_15fps_experiment'
out.mkdir(parents=True,exist_ok=True)
os.chdir(repo)
env=os.environ.copy()
env.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',PYTHONUNBUFFERED='1',
           TORCH_HOME=str(repo/'.cache/torch'),TORCH_EXTENSIONS_DIR=str(repo/'.cache/torch_extensions'),
           CUDA_HOME=str(repo/'.tools/cuda-12.8'),CUDA_PATH=str(repo/'.tools/cuda-12.8'),
           TORCH_CUDA_ARCH_LIST='7.5',NUM_CHANNELS='3,4',PYTHONPATH=str(repo))
env['PATH']=str(repo/'.tools/cuda-12.8/bin')+os.pathsep+env['PATH']


def state(stage,status,**kw):
    (out/'status.json').write_text(json.dumps(dict(stage=stage,status=status,updated_epoch=time.time(),pid=os.getpid(),**kw),indent=2))
    print(stage,status,kw,flush=True)


def run(stage,command):
    marker=out/(stage+'_success.json')
    if marker.exists():
        print('SKIP completed',stage,flush=True)
        return
    state(stage,'running',command=command)
    start=time.time()
    with (out/(stage+'.log')).open('a',encoding='utf-8') as log:
        proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
    if proc.returncode:
        state(stage,'failed',exit_code=proc.returncode,log=str(out/(stage+'.log')))
        raise SystemExit(proc.returncode)
    marker.write_text(json.dumps(dict(elapsed_seconds=time.time()-start,command=command),indent=2))


state('await_extraction','running')
for attempt in range(180):
    if (root/'capture_manifest.json').exists():
        break
    time.sleep(5)
else:
    state('await_extraction','failed',reason='Extraction did not finish within 15 minutes')
    raise SystemExit(1)

for stage in ('split','features','match'):
    run(stage,[sys.executable,'-u','scripts/experiment-realsense-15fps.py',stage])
for stage in ('extract','register','odometry'):
    run('rgbd_'+stage,[sys.executable,'-u','scripts/recover-rgbd-scene.py',stage,'--root',str(root)])
run('sfm_anchors',[sys.executable,'-u','scripts/experiment-realsense-15fps.py','sfm_anchors'])
for stage in ('anchors','bridge','optimize'):
    run('rgbd_'+stage,[sys.executable,'-u','scripts/recover-rgbd-scene.py',stage,'--root',str(root)])
run('validate_dataset',[sys.executable,'-u','scripts/compare-realsense-15fps.py','validate'])
run('train30k',['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File','scripts/test-small-dataset.ps1',
    '-Steps','30000','-DataDir',str(root/'rgbd/dataset'),'-ResultDir',str(out/'train30k'),
    '-ShDegree','3','-ShDegreeInterval','1000','-RefineStart','500','-RefineStop','15000',
    '-ResetEvery','3000','-Strategy','budget','-GaussianBudget','1000000',
    '-RecordExperiment','-DataWorkers','0','-NoCacheImages',
    '-SensorDepthDir',str(root/'rgbd/dataset/sensor_depth'),'-DepthWeight','0.01',
    '-SplitMode','long_axis','-ClonePolicy','split','-EdgeWeight','0.25'])
run('compare',[sys.executable,'-u','scripts/compare-realsense-15fps.py','compare'])
state('complete','complete')
