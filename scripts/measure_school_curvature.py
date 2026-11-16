"""Fixed-view log-depth curvature proxy, not ground-truth geometry accuracy."""
import gc
import json
from pathlib import Path
import numpy as np
import torch
from research_campaign import REPO, write, sha
from simple_trainer import Config, Runner
from gsplat.scene import GaussianScene
from gsplat.stage import Stage
from gsplat.surface import surface_depth_loss

import argparse
p=argparse.ArgumentParser(__doc__)
p.add_argument('--root',type=Path,default=REPO/'results/school_completion_20261109')
root=p.parse_args().root.resolve()
records=json.loads((root/'surface/summary.json').read_text())
saved=root/'curvature_proxy.json'
result={}
depth_samples={}
rgb_samples={}
valid_samples={}
with torch.no_grad():
    for label,rec in records.items():
        folder=Path(rec['archived_control']).parent if 'archived_control' in rec else root/'surface'/rec['job']['id']
        cfg=Config(data_dir=str(REPO/'data/room_factor2'),data_factor=1,
            result_dir=str(root/'curvature_loaders'/label),disable_viewer=True,disable_video=True,
            packed=True,data_workers=0,load_exposure=False)
        runner=Runner(0,0,1,cfg)
        ckpt=torch.load(folder/'ckpts/final_post_update.pt',map_location=runner.device,weights_only=True)
        for key in runner.splats: runner.splats[key].data=ckpt['splats'][key]
        runner.scene=GaussianScene.from_splats(runner.splats,id='scene')
        runner.splats=runner.scene.splats
        runner.stage=Stage()
        runner.stage.add_scene(runner.scene,runner.rasterize_splats)
        rows=[]
        depth_samples[label],valid_samples[label]={},{}
        for index in np.linspace(0,len(runner.valset)-1,5).astype(int):
            sample=runner.valset[int(index)]
            h,w=sample['image'].shape[:2]
            out,alpha,_=runner.stage.render(runner.scene.id,
                camtoworlds=sample['camtoworld'].to(runner.device)[None],
                Ks=sample['K'].to(runner.device)[None],width=w,height=h,
                sh_degree=3,render_mode='RGB+ED')
            rgb=sample['image'].to(runner.device)[None].float()/255
            value=surface_depth_loss(out[...,3:4],rgb,alpha,sample.get('mask',None))
            depth_samples[label][int(index)]=out[0,::4,::4,3].cpu().numpy().copy()
            valid_samples[label][int(index)]=(alpha[0,::4,::4,0]>.7).cpu().numpy().copy()
            rgb_samples[int(index)]=rgb[0,::4,::4].cpu().numpy().copy()
            rows.append(dict(val_index=int(index),curvature=float(value),
                opaque_fraction=float((alpha>.7).float().mean())))
        result[label]=dict(mean_curvature=float(np.mean([r['curvature'] for r in rows])),views=rows,
            checkpoint_sha256=rec['final_checkpoint_sha256'])
        write(root/'curvature_proxy.json',dict(arms=result,
            limitations='Same five exposed held-out views. Lower edge-aware log-depth curvature can reflect oversmoothing; opacity masks can differ. This is not geometric accuracy.'))
        print(label,result[label]['mean_curvature'],flush=True)
        runner.writer.close()
        del runner,ckpt,out,alpha,rgb,sample
        gc.collect()
        torch.cuda.empty_cache()
common={}
for index,rgb in rgb_samples.items():
    valid=np.logical_and.reduce([valid_samples[k][index] & np.isfinite(depth_samples[k][index]) &
                                (depth_samples[k][index]>0) for k in result])
    for label in result:
        z=np.log(np.maximum(depth_samples[label][index],1e-5))
        terms=[]
        for axis in (0,1):
            a,b,c=[slice(None)]*2,[slice(None)]*2,[slice(None)]*2
            a[axis],b[axis],c[axis]=slice(None,-2),slice(1,-1),slice(2,None)
            a,b,c=tuple(a),tuple(b),tuple(c)
            edge=np.abs(rgb[a]-rgb[b]).mean(-1)+np.abs(rgb[c]-rgb[b]).mean(-1)
            weight=np.exp(-10*edge)*(valid[a]&valid[b]&valid[c])
            terms.append(float((np.abs(z[a]-2*z[b]+z[c])*weight).sum()/max(weight.sum(),1)))
        common.setdefault(label,[]).append(dict(val_index=index,curvature=float(np.mean(terms)),common_fraction=float(valid.mean())))
for label in result:
    result[label]['common_support_views']=common[label]
    result[label]['common_support_mean_curvature']=float(np.mean([r['curvature'] for r in common[label]]))
write(saved,dict(arms=result,source_sha256=sha(Path(__file__)),
    limitations='Five exposed held-out views. Both own-alpha and identical all-arm opacity support are reported. Lower curvature can be oversmoothing; neither is geometric accuracy.'))
