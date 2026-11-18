"""Keep every qualified source frame, estimate its pose, train and verify 3DGS.

Reuses verified same-recording 3 FPS relative-pose anchors; all additional poses
come from cleaned RGB-D feature matches and gated odometry, never interpolation.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared', action='store_true')
    ap.add_argument('--await-clean', action='store_true', help='Wait for a separately running extractor to publish its complete manifest')
    ap.add_argument('--data', type=Path, default=Path('data/realsense_20261030_111354_all30'))
    ap.add_argument('--base', type=Path, default=Path('data/realsense_20261030_111354_clean'))
    ap.add_argument('--result', type=Path, default=Path('results/realsense_20261030_111354_all30'))
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--stage', choices=['run', 'features', 'match', 'split', 'finalize'], default='run')
    a = ap.parse_args(); os.chdir(REPO)
    a.data=a.data.resolve(); a.base=a.base.resolve(); a.result=a.result.resolve()
    a.result.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', PYTHONUNBUFFERED='1', PYTHONPATH=str(REPO),
               TORCH_HOME=str(REPO/'.cache/torch'), TORCH_EXTENSIONS_DIR=str(REPO/'.cache/torch_extensions'),
               TORCH_CUDA_ARCH_LIST='7.5', NUM_CHANNELS='3,4', MAX_JOBS='2')
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    def run(stage, command):
        command=list(map(str,command)); marker=a.result/(stage+'_success.json')
        if marker.exists():
            if json.loads(marker.read_text())['command'] != command:
                raise ValueError('Changed command for completed '+stage)
            print('SKIP',stage,flush=True); return
        write(a.result/'status.json',dict(stage=stage,status='running',pid=os.getpid(),command=command))
        print('RUN',stage,flush=True); started=time.time()
        with (a.result/(stage+'.log')).open('a',encoding='utf-8') as log:
            proc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:
            write(a.result/'status.json',dict(stage=stage,status='failed',exit_code=proc.returncode))
            raise SystemExit(proc.returncode)
        write(marker,dict(command=command,elapsed_seconds=time.time()-started))
    if a.await_clean and a.stage=='run':
        write(a.result/'status.json',dict(stage='await_clean',status='running',pid=os.getpid()))
        while not (a.data/'capture_manifest.json').exists():
            time.sleep(5)
    if a.stage == 'features':
        import numpy as np
        import pycolmap as p
        cal=json.loads((a.data/'calibration_clean.json').read_text()); K=np.array(cal['K'])
        reader=p.ImageReaderOptions();reader.camera_model='PINHOLE'
        reader.camera_params=','.join(map(str,[K[0,0],K[1,1],K[0,2],K[1,2]]))
        opt=p.FeatureExtractionOptions();opt.num_threads=8;opt.max_image_size=848
        opt.sift.max_num_features=6000;opt.sift.first_octave=0
        p.extract_features(a.data/'colmap.db',a.data/'images',camera_mode=p.CameraMode.SINGLE,
                           reader_options=reader,extraction_options=opt,device=p.Device.cpu)
        return
    if a.stage == 'match':
        proc=subprocess.run([sys.executable,'-u',str(REPO/'scripts/match-clean-rgbd-cuda.py'),
                             '--root',str(a.data),'--report',str(a.result/'matching_policy.json')])
        if proc.returncode:raise SystemExit(proc.returncode)
        return
    if a.stage == 'split':
        import numpy as np
        import pycolmap as p
        baseline=json.loads((a.base/'capture_manifest.json').read_text())
        manifest=json.loads((a.data/'capture_manifest.json').read_text())
        original={f['name']:f for f in baseline['frames']}
        current={f['message_id']:f for f in manifest['frames']}
        model=p.Reconstruction(a.base/'rgbd/dataset/sparse')
        heldout=[original[n] for n in sorted(im.name for im in model.images.values())[::8]]
        missing=[f['message_id'] for f in heldout if f['message_id'] not in current]
        if missing:
            raise ValueError(f'Original heldout views rejected by cleaning: {missing}')
        times=np.array([f['timestamp_ns'] for f in heldout],dtype=np.int64)
        val=[current[f['message_id']]['name'] for f in heldout]
        train=[f['name'] for f in manifest['frames'] if np.min(np.abs(times-f['timestamp_ns'])) > 200_000_000]
        excluded=[f['name'] for f in manifest['frames'] if f['name'] not in set(train)|set(val)]
        write(a.data/'experiment_split.json',dict(train=train,val=val,excluded=excluded,
              policy='Same 88 original 3 FPS heldout frames; retain all qualified files, exclude training within +/-0.2 seconds of heldout timestamps',
              baseline_source_ids=[f['message_id'] for f in heldout]))
        print('SPLIT',len(train),len(val),'temporal buffer',len(excluded),flush=True)
        return
    if a.stage == 'finalize':
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw
        import numpy as np
        import torch
        sys.path.insert(0,str(REPO/'examples'))
        from datasets.colmap import Parser
        from gsplat.rendering import rasterization
        from gsplat.exporter import export_splats,load_ply_to_splats
        checkpoint=a.result/'train/ckpts/final_post_update.pt'
        experiment=json.loads((a.result/'train/experiment.json').read_text())
        cfg_text=(a.result/'train/cfg.yml').read_text()
        if experiment['completed_updates']!=a.steps or 'DefaultStrategy' not in cfg_text or 'cap_max:' in cfg_text:
            raise ValueError('Saved training configuration does not match requested uncapped strategy/update count')
        ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for v in ckpt['splats'].values())
        preview=a.result/'preview';preview.mkdir(exist_ok=True)
        export_splats(**{k:ckpt['splats'][k] for k in ('means','scales','quats','opacities','sh0','shN')},format='ply',save_to=str(preview/'scene.ply'))
        reloaded=load_ply_to_splats(str(preview/'scene.ply'))
        assert len(reloaded['means'])==len(ckpt['splats']['means']) and all(torch.isfinite(v).all() for v in reloaded.values())
        del reloaded
        splats={k:v.cuda() for k,v in ckpt['splats'].items()}
        colors=torch.cat([splats['sh0'],splats['shN']],dim=1)
        parser=Parser(str(a.data/'rgbd/dataset'),factor=1,normalize=True,load_exposure=False)
        source=json.loads((a.data/'capture_manifest.json').read_text())
        source_by_name={f['name']:f for f in source['frames']}
        split=json.loads((a.data/'experiment_split.json').read_text())
        camera_uses=Counter(experiment['camera_sequence'])
        if len(camera_uses)!=len(split['train']):
            raise ValueError('Some qualified training views were never sampled')
        # Compare identical heldout source images; do not insert blank baseline
        # panels for frames absent from the lower-rate capture selection.
        selected=sorted(set(split['val']))
        idx={n:i for i,n in enumerate(parser.image_names)}
        base_manifest=json.loads((a.base/'capture_manifest.json').read_text())
        base_by_id={f['message_id']:f['name'] for f in base_manifest['frames']}
        base_parser=Parser(str(a.base/'rgbd/dataset'),factor=1,normalize=True,load_exposure=False)
        base_idx={n:i for i,n in enumerate(base_parser.image_names)}
        base_ckpt=torch.load(REPO/'results/realsense_20261030_111354_clean/train/ckpts/final_post_update.pt',map_location='cpu',weights_only=True)
        old={k:v.cuda() for k,v in base_ckpt['splats'].items()}
        old_colors=torch.cat([old['sh0'],old['shN']],dim=1)
        from torchmetrics.image import PeakSignalNoiseRatio,StructuralSimilarityIndexMeasure
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        metrics=dict(psnr=PeakSignalNoiseRatio(data_range=1.).cuda(),ssim=StructuralSimilarityIndexMeasure(data_range=1.).cuda(),
                     lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda())
        def render(n,par,ids,s,sh):
            i=ids[n]; cid=par.camera_ids[i];w,h=par.imsize_dict[cid]
            pred,_,_=rasterization(s['means'],s['quats'],torch.exp(s['scales']),torch.sigmoid(s['opacities']),sh,
                  torch.from_numpy(np.linalg.inv(par.camtoworlds[i])[None]).float().cuda(),
                  torch.from_numpy(par.Ks_dict[cid][None]).float().cuda(),w,h,sh_degree=3,packed=True)
            pred=pred[0].clamp(0,1);assert torch.isfinite(pred).all();return pred
        rows=[];sheet=Image.new('RGB',(1272,8*260),'#202020')
        chosen=set(np.linspace(0,len(selected)-1,8,dtype=int).tolist())
        video=imageio.get_writer(preview/'capture_path.mp4',fps=4,macro_block_size=1,codec='libx264',quality=8)
        with torch.no_grad():
            for j,name in enumerate(selected):
                im=np.array(Image.open(a.data/'images'/name))
                target=torch.from_numpy(im.copy()).float().cuda()/255
                pred=render(name,parser,idx,splats,colors)
                old_name=base_by_id.get(source_by_name[name]['message_id'])
                prior=render(old_name,base_parser,base_idx,old,old_colors) if old_name in base_idx else None
                if name in split['val']:
                    assert prior is not None
                    assert np.array_equal(im,np.array(Image.open(base_parser.image_paths[base_idx[old_name]])))
                    row=dict(name=name,source_id=source_by_name[name]['message_id'])
                    for label,p in [('3fps',prior),('all30',pred)]:
                        row[label]={k:float(tool(p.permute(2,0,1)[None],target.permute(2,0,1)[None])) for k,tool in metrics.items()}
                    rows.append(row)
                pixels=(pred.cpu().numpy()*255).astype('u1')
                pieces=[im,(prior.cpu().numpy()*255).astype('u1') if prior is not None else np.zeros_like(im),pixels]
                canvas=Image.fromarray(np.concatenate(pieces,axis=1))
                caption=Image.new('RGB',(canvas.width,canvas.height+25),'#202020');caption.paste(canvas,(0,25))
                ImageDraw.Draw(caption).text((5,5),'Captured RGB | 3 FPS (black when no original source view) | all qualified 30 FPS',fill='white')
                if caption.height%2:
                    even=Image.new('RGB',(caption.width,caption.height+1));even.paste(caption);caption=even
                video.append_data(np.array(caption))
                if j in chosen:
                    thumb=caption.copy();thumb.thumbnail((1272,255));sheet.paste(thumb,(0,sorted(chosen).index(j)*260))
                    caption.save(preview/f'comparison_{j:04d}.jpg',quality=94)
        video.close();sheet.save(preview/'comparison_sheet.jpg',quality=94)
        report=dict(heldout_views=len(rows),per_view=rows,matched_pixel_targets=True,
                    mean_metrics={label:{k:float(np.mean([r[label][k] for r in rows])) for k in metrics} for label in ('3fps','all30')},
                    scene_gaussians=len(splats['means']),ply_reload_verified=True,finite_tensors=True,finite_rendered_pixels=True,
                    rendered_views=len(selected),pose_anchor_scope='Uses same-recording baseline relative poses; not an independent reconstruction or geometry benchmark',
                    completed_updates=experiment['completed_updates'],uncapped_default_strategy_verified=True,
                    distinct_training_views_used=len(camera_uses),minimum_updates_per_training_view=min(camera_uses.values()),
                    maximum_updates_per_training_view=max(camera_uses.values()),
                    compute_scope=f'Full-rate run: {a.steps} updates with standard uncapped densification; baseline: 12000 updates with 500k cap. Combined change, not an FPS-only ablation')
        reader=imageio.get_reader(preview/'capture_path.mp4');decoded=sum(1 for _ in reader);reader.close()
        assert decoded==len(selected);report['decoded_video_frames']=decoded
        write(a.result/'comparison.json',report)
        original_provenance=json.loads((REPO/'results/realsense_20261030_111354_clean/provenance.json').read_text())
        with Path(source['source']).open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        assert digest==original_provenance['source_sha256']
        inputs=[a.data/'colmap.db',a.data/'capture_manifest.json',a.data/'calibration_clean.json',a.data/'experiment_split.json',
                a.data/'rgbd/edges.json',a.data/'rgbd/poses.npz',checkpoint,
                REPO/'results/realsense_20261030_111354_clean/train/ckpts/final_post_update.pt']
        input_hashes={}
        for path in inputs:
            with path.open('rb') as stream:input_hashes[str(path.relative_to(REPO))]=hashlib.file_digest(stream,'sha256').hexdigest()
        write(a.result/'provenance.json',dict(source_sha256=digest,source_hash_unchanged=True,input_sha256=input_hashes,
              code_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),REPO/'scripts/match-clean-rgbd-cuda.py',REPO/'scripts/extract-realsense-all30.py',REPO/'scripts/clean-realsense-scene.py',REPO/'scripts/recover-clean-rgbd.py',REPO/'scripts/bridge-clean-rgbd.py',REPO/'scripts/recover-rgbd-scene.py',REPO/'scripts/sparse-rgbd-pose-graph.py',REPO/'scripts/test-small-dataset.ps1',REPO/'examples/simple_trainer.py']}))
        write(a.result/'viewer.json',dict(checkpoint=str(checkpoint),dataset=str(a.data/'rgbd/dataset'),camera_index=1000))
        print('FULL30_VERIFIED',report['mean_metrics'],flush=True);return
    if not a.prepared:
        if not (a.data/'calibration_clean.json').exists():
            # The calibrated 3 FPS sample has exactly the same source and output
            # camera. Verify identity before reusing its rectification metadata.
            baseline=json.loads((a.base/'capture_manifest.json').read_text())
            source=Path(baseline['source']);stat=source.stat()
            if (stat.st_size,stat.st_mtime_ns)!=(baseline['source_bytes'],baseline['source_mtime_ns']):
                raise ValueError('Baseline source changed')
            a.data.mkdir(parents=True,exist_ok=True)
            for name in ('calibration_clean.json','calibration_raw.json'):
                (a.data/name).write_bytes((a.base/name).read_bytes())
        run('clean',[sys.executable,'-u','scripts/extract-realsense-all30.py','--root',a.data])
    for stage in ('split','features','match'):
        run(stage,[sys.executable,'-u',__file__,'--stage',stage,'--data',a.data,'--base',a.base,'--result',a.result])
    run('audit',[sys.executable,'-u','scripts/audit-clean-realsense.py','--root',a.data])
    run('recover',[sys.executable,'-u','scripts/recover-clean-rgbd.py','--root',a.data,'--anchor-root',a.base])
    run('bridge',[sys.executable,'-u','scripts/bridge-clean-rgbd.py','--root',a.data])
    report=json.loads((a.data/'rgbd/report.json').read_text())
    write(a.result/'coverage_gate.json',report)
    if report['coverage_fraction'] < .99 or report['translation_residual_percentiles_m'][1] >= .08 or report['rotation_residual_percentiles_degrees'][1] >= 3:
        raise ValueError('Full-rate coverage/pose gate failed; inspect before training')
    run('validate',[sys.executable,'-u','scripts/clean-realsense-scene.py','validate','--output',a.data,'--dataset-name','rgbd/dataset'])
    run('train',['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File','scripts/test-small-dataset.ps1',
         '-Steps',a.steps,'-DataDir',a.data/'rgbd/dataset','-ResultDir',a.result/'train',
         '-ShDegree','3','-ShDegreeInterval','1000','-RefineStart','500','-RefineStop','6000','-ResetEvery','3000',
         '-Strategy','default','-RecordExperiment','-DataWorkers','0','-NoCacheImages',
         '-SensorDepthDir',a.data/'rgbd/dataset/sensor_depth','-DepthWeight','.01'])
    compiler=subprocess.check_output(f'call "{REPO/".tools/msvc/setup_x64.bat"}" >nul && set',shell=True,text=True)
    for line in compiler.splitlines():
        if '=' in line and not line.startswith('='):
            k,v=line.split('=',1);env[k]=v
    env.update(CUDA_HOME=str(REPO/'.tools/cuda-12.8'),CUDA_PATH=str(REPO/'.tools/cuda-12.8'))
    env['PATH']=str(REPO/'.venv/Scripts')+os.pathsep+str(REPO/'.tools/cuda-12.8/bin')+os.pathsep+env['PATH']
    run('finalize',[sys.executable,'-u',__file__,'--stage','finalize','--data',a.data,'--base',a.base,'--result',a.result,'--steps',a.steps])
    write(a.result/'status.json',dict(stage='complete',status='complete',updated_epoch=time.time()))


if __name__=='__main__':
    main()
