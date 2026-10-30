"""Validate split isolation and compare identical source frames across reconstructions."""
import argparse
import json
import os
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'examples'))
ROOT=REPO/'data/realsense_20261029_140739_15fps'
BASE=REPO/'data/realsense_20261029_140739'
OUT=REPO/'results/realsense_15fps_experiment'
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('TORCH_HOME',str(REPO/'.cache/torch'))


def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')


def validate():
    import hashlib
    import numpy as np
    from datasets.colmap import Parser,Dataset
    manifest=json.loads((ROOT/'capture_manifest.json').read_text())
    split=json.loads((ROOT/'experiment_split.json').read_text())
    p=Parser(str(ROOT/'rgbd/dataset'),factor=1,normalize=True,load_exposure=False)
    train=Dataset(p,split='train',load_depths=True,sensor_depth_dir=str(ROOT/'rgbd/dataset/sensor_depth'))
    val=Dataset(p,split='val')
    viewer_name=next(f['name'] for f in manifest['frames'] if f['baseline_name']=='frame_000645.jpg')
    assert viewer_name in p.image_names, 'Reconstruction omits the baseline viewer region; do not train a replacement comparison'
    tn={p.image_names[i] for i in train.indices};vn={p.image_names[i] for i in val.indices}
    assert not tn&vn and tn<=set(split['train']) and vn<=set(split['val'])
    frames={f['name']:f for f in manifest['frames']}
    tv=np.array([frames[n]['time_seconds'] for n in vn])
    min_gap=min(float(np.abs(tv-frames[n]['time_seconds']).min()) for n in tn)
    assert min_gap>.2
    for i in np.linspace(0,len(train)-1,8,dtype=int):
        sample=train[int(i)]
        assert np.isfinite(sample['image'].numpy()).all()
        assert np.isfinite(sample['depths'].numpy()).all()
    source=Path(manifest['source']).stat()
    assert source.st_size==manifest['source_bytes'] and source.st_mtime_ns==manifest['source_mtime_ns']
    baseline_selection=json.loads((REPO/'results/realsense_20261029/selection.json').read_text())
    with (REPO/baseline_selection['checkpoint']).open('rb') as stream:
        baseline_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    assert baseline_hash==baseline_selection['checkpoint_sha256']
    report=dict(views=len(p.image_names),train_views=len(tn),heldout_views=len(vn),
                original_heldout_views=79,minimum_train_to_heldout_gap_seconds=min_gap,
                source_stat_unchanged=True,baseline_checkpoint_sha256=baseline_hash,finite_sample_images_and_depth=True,
                normalization_transform=p.transform.tolist())
    write('dataset_validation.json',report)
    print('DATASET_VALID',report,flush=True)


def compare():
    import hashlib
    import subprocess
    if sys.platform=='win32':
        # The training PowerShell process owns its MSVC environment; it does
        # not propagate that environment back to this comparison process.
        batch=REPO/'.tools/msvc/setup_x64.bat'
        envtext=subprocess.check_output(f'call "{batch}" >nul && set',shell=True,text=True)
        for line in envtext.splitlines():
            if '=' in line and not line.startswith('='):
                key,value=line.split('=',1);os.environ[key]=value
        os.environ['CUDA_HOME']=str(REPO/'.tools/cuda-12.8')
        os.environ['CUDA_PATH']=os.environ['CUDA_HOME']
        os.environ['PATH']=str(REPO/'.venv/Scripts')+os.pathsep+str(REPO/'.tools/cuda-12.8/bin')+os.pathsep+os.environ['PATH']
        os.environ.setdefault('TORCH_EXTENSIONS_DIR',str(REPO/'.cache/torch_extensions'))
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST','7.5')
        os.environ.setdefault('NUM_CHANNELS','3,4')
        os.environ.setdefault('MAX_JOBS','2')
    import math
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from datasets.colmap import Parser
    from gsplat.rendering import rasterization
    from gsplat.exporter import export_splats,load_ply_to_splats
    from torchmetrics.image import PeakSignalNoiseRatio,StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    torch.set_num_threads(8)
    manifest=json.loads((ROOT/'capture_manifest.json').read_text())
    mapping={f['baseline_name']:f['name'] for f in manifest['frames'] if f['baseline_name']}
    split=json.loads((ROOT/'experiment_split.json').read_text())
    b=Parser(str(BASE/'rgbd/dataset'),factor=1,normalize=True,load_exposure=False)
    n=Parser(str(ROOT/'rgbd/dataset'),factor=1,normalize=True,load_exposure=False)
    bi={s:i for i,s in enumerate(b.image_names)};ni={s:i for i,s in enumerate(n.image_names)}
    common=[s for s in split['baseline_val_names'] if s in bi and mapping[s] in ni]
    assert common
    selection=common.copy()
    if 'frame_000645.jpg' in bi and mapping['frame_000645.jpg'] in ni:
        selection.append('frame_000645.jpg')
    metric_tools=dict(psnr=PeakSignalNoiseRatio(data_range=1.).cuda(),
                      ssim=StructuralSimilarityIndexMeasure(data_range=1.).cuda(),
                      lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda())
    targets={}
    for name in selection:
        a=np.asarray(Image.open(b.image_paths[bi[name]]))[:,:,:3]
        z=np.asarray(Image.open(n.image_paths[ni[mapping[name]]]))[:,:,:3]
        assert np.array_equal(a,z),f'Comparison RGB differs for {name}'
        targets[name]=torch.from_numpy(a.copy()).float().cuda()/255.
    rows={s:dict(baseline_name=s,new_name=mapping[s],heldout=s in common) for s in selection}
    preview=OUT/'comparison';preview.mkdir(exist_ok=True)
    allpixels={}
    models=[('2fps',b,bi,REPO/'results/realsense_20261029/full_scene_626_depth30k/ckpts/final_post_update.pt'),
            ('15fps',n,ni,OUT/'train30k/ckpts/final_post_update.pt')]
    model_info={}
    with torch.no_grad():
        for label,parser,indices,path in models:
            ckpt=torch.load(path,map_location='cpu',weights_only=True)
            assert all(torch.isfinite(v).all() for v in ckpt['splats'].values())
            splats={k:v.cuda() for k,v in ckpt['splats'].items()}
            colors=torch.cat([splats['sh0'],splats['shN']],dim=1)
            degree=int(math.sqrt(colors.shape[1])-1)
            allpixels[label]={}
            for j,name in enumerate(selection):
                index=indices[name if label=='2fps' else mapping[name]]
                cid=parser.camera_ids[index];w,h=parser.imsize_dict[cid]
                K=torch.from_numpy(parser.Ks_dict[cid][None]).float().cuda()
                view=torch.from_numpy(np.linalg.inv(parser.camtoworlds[index])[None]).float().cuda()
                pred,_,_=rasterization(splats['means'],splats['quats'],torch.exp(splats['scales']),
                    torch.sigmoid(splats['opacities']),colors,view,K,w,h,sh_degree=degree,packed=True)
                pred=pred[0].clamp(0,1)
                assert torch.isfinite(pred).all()
                a=pred.permute(2,0,1)[None];t=targets[name].permute(2,0,1)[None]
                rows[name][label]={k:float(tool(a,t)) for k,tool in metric_tools.items()}
                allpixels[label][name]=(pred.cpu().numpy()*255).astype('uint8')
                if j%20==0: print('COMPARE',label,j,len(selection),flush=True)
            model_info[label]=dict(checkpoint=str(path),step=ckpt['step'],gaussians=len(splats['means']),finite_tensors=True)
            if label=='15fps':
                ply=OUT/'scene_15fps.ply'
                export_splats(**{k:ckpt['splats'][k] for k in ('means','scales','quats','opacities','sh0','shN')},format='ply',save_to=str(ply))
                reloaded=load_ply_to_splats(str(ply))
                assert len(reloaded['means'])==len(splats['means']) and all(torch.isfinite(v).all() for v in reloaded.values())
                model_info[label]['ply_reload_verified']=True
            del splats,ckpt,colors
            torch.cuda.empty_cache()
    writer=imageio.get_writer(preview/'matched_views.mp4',fps=4,macro_block_size=1,codec='libx264',quality=8)
    chosen=[common[i] for i in np.linspace(0,len(common)-1,min(8,len(common)),dtype=int)]
    if selection[-1] not in common: chosen.append(selection[-1])
    sheet=Image.new('RGB',(1272,len(chosen)*260),'#202020')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    for name in selection:
        target=(targets[name].cpu().numpy()*255).astype('uint8')
        canvas=np.concatenate([target,allpixels['2fps'][name],allpixels['15fps'][name]],axis=1)
        im=Image.fromarray(canvas)
        title=Image.new('RGB',(im.width,im.height+30),'#202020');title.paste(im,(0,30))
        d=ImageDraw.Draw(title)
        for x,label in enumerate(['Captured RGB','2 FPS - 30k updates','15 FPS - 30k updates']):
            d.text((x*target.shape[1]+10,3),label+' | '+name,fill='white',font=font)
        # H.264 needs even dimensions. The added border does not enter metrics.
        even=Image.new('RGB',(title.width+title.width%2,title.height+title.height%2));even.paste(title)
        writer.append_data(np.asarray(even))
        if name in chosen:
            title.save(preview/(name[:-4]+'.png'))
            thumb=title.copy();thumb.thumbnail((1272,255))
            sheet.paste(thumb,(0,chosen.index(name)*260))
    writer.close();sheet.save(preview/'comparison_sheet.jpg',quality=94)
    means={label:{key:float(np.mean([rows[s][label][key] for s in common])) for key in metric_tools} for label in ('2fps','15fps')}
    delta={key:means['15fps'][key]-means['2fps'][key] for key in metric_tools}
    report=dict(common_heldout_views=len(common),original_heldout_views=79,models=model_info,mean_metrics=means,
                delta_15fps_minus_2fps=delta,per_view=list(rows.values()),
                viewpoint_policy='Exact same source RGB message per pair; each model uses its independently rebuilt pose for that capture. World coordinates differ.',
                training_policy='30k updates, same 1M Gaussian cap/resolution/depth weight; 15 FPS has more training views and fewer passes per view.',
                limits='Estimated cameras, not ground truth; reflections/motion/blur persist. Equal updates do not imply equal total reconstruction cost.')
    report['pose_reconstruction']=json.loads((ROOT/'rgbd/report.json').read_text())
    report['dataset_validation']=json.loads((OUT/'dataset_validation.json').read_text())
    report['quality_wins']={key:sum((rows[s]['15fps'][key]-rows[s]['2fps'][key])*(1 if key!='lpips' else -1)>0 for s in common) for key in metric_tools}
    report['completed_stage_seconds']={f.stem.removesuffix('_success'):json.loads(f.read_text())['elapsed_seconds'] for f in OUT.glob('*_success.json')}
    report['source_code_sha256']={}
    for source_name in ['examples/datasets/colmap.py','examples/simple_trainer.py',
                        'scripts/experiment-realsense-15fps.py','scripts/recover-rgbd-scene.py',
                        'scripts/sparse-rgbd-pose-graph.py','scripts/compare-realsense-15fps.py',
                        'scripts/test-small-dataset.ps1','scripts/run-realsense-15fps.py']:
        with (REPO/source_name).open('rb') as stream:
            report['source_code_sha256'][source_name]=hashlib.file_digest(stream,'sha256').hexdigest()
    stat=Path(manifest['source']).stat()
    assert stat.st_size==manifest['source_bytes'] and stat.st_mtime_ns==manifest['source_mtime_ns']
    report['source_stat_unchanged_after_training']=True
    write('comparison.json',report)
    viewer_name=mapping['frame_000645.jpg'] if mapping['frame_000645.jpg'] in ni else mapping[common[len(common)//2]]
    write('viewer.json',dict(checkpoint=str(OUT/'train30k/ckpts/final_post_update.pt'),
                            dataset=str(ROOT/'rgbd/dataset'),camera_index=ni[viewer_name],image_name=viewer_name))
    md=['# 15 FPS experiment','',f'Extracted 5,204 frames at 15 FPS; rebuilt {len(n.image_names)} cameras. Baseline: {len(b.image_names)} cameras from 694 selected frames.','',
        f'Both models trained for 30,000 updates. Compared {len(common)} of the original 79 held-out source frames.','',
        '| Metric | 2 FPS | 15 FPS | Difference |','|---|---:|---:|---:|']
    for key in metric_tools:
        md.append(f'| {key.upper()} | {means["2fps"][key]:.5f} | {means["15fps"][key]:.5f} | {delta[key]:+.5f} |')
    md+=['',f'15 FPS improves PSNR on {report["quality_wins"]["psnr"]}/{len(common)} common views; SSIM on {report["quality_wins"]["ssim"]}/{len(common)}; LPIPS on {report["quality_wins"]["lpips"]}/{len(common)}.','',report['viewpoint_policy'],'',report['training_policy'],'',report['limits'],'',
         'See [experiment methods](METHODS.md) for sampling, temporal exclusions, pose solver, and comparison limits.','',
         '[Comparison sheet](comparison/comparison_sheet.jpg)',
         '[Matched-view video](comparison/matched_views.mp4)',
         '[Detailed measurements](comparison.json)']
    (OUT/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('COMPARISON_COMPLETE',means,delta,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['validate','compare'])
    globals()[ap.parse_args().stage]()
