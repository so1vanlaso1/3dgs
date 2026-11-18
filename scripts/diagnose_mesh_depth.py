import numpy as np
import torch
from simple_trainer import Config,Runner
from gsplat.scene import GaussianScene
from gsplat.stage import Stage
from research_campaign import REPO
runner=Runner(0,0,1,Config(data_dir=str(REPO/'data/room_factor2'),data_factor=1,
    result_dir=str(REPO/'results/school_mesh_depth_diagnostic_20261109'),disable_viewer=True,packed=True,load_exposure=False))
ckpt=torch.load(REPO/'results/research_completion_20261027/replication/room/B00_s42/ckpts/final_post_update.pt',map_location=runner.device,weights_only=True)
for k in runner.splats:runner.splats[k].data=ckpt['splats'][k]
runner.scene=GaussianScene.from_splats(runner.splats,id='scene')
runner.splats=runner.scene.splats
runner.stage=Stage()
runner.stage.add_scene(runner.scene,runner.rasterize_splats)
with torch.no_grad():
    s=runner.trainset[0]
    h,w=s['image'].shape[:2]
    K=s['K'].clone();K[0]*=(w//4)/w;K[1]*=(h//4)/h
    out,a,_=runner.stage.render(runner.scene.id,camtoworlds=s['camtoworld'].to(runner.device)[None],Ks=K.to(runner.device)[None],width=w//4,height=h//4,sh_degree=3,render_mode='RGB+ED')
    for key,x in [('depth',out[0,...,3]),('alpha',a[0,...,0]),('means',runner.splats['means'])]:
        print(key,tuple(x.shape),torch.quantile(x.flatten(),torch.tensor([0,.01,.5,.99,1],device=x.device)).tolist(),flush=True)
    np.savez(REPO/'results/school_mesh_depth_diagnostic_20261109/view.npz',depth=out[0,...,3].cpu().numpy(),alpha=a[0,...,0].cpu().numpy(),K=K.numpy(),extrinsic=np.linalg.inv(s['camtoworld'].numpy()))
