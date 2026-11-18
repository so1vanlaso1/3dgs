"""Compare candidate and saved 20k baseline on the same held-out RGB images."""
import json
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
sys.path.insert(0,str(Path('examples').resolve()))
from datasets.colmap import Parser
from gsplat.rendering import rasterization

ap=argparse.ArgumentParser();ap.add_argument('--prefer-people',action='store_true');args=ap.parse_args()
data=Path('data/courtyard_revisit_people_20261102' if args.prefer_people else 'data/courtyard_revisit_repair_20261102')
out=Path('results/courtyard_revisit_repair_20261102/trial_people' if args.prefer_people else 'results/courtyard_revisit_repair_20261102/trial')
baseline=json.loads(Path('results/realsense_20261030_111354_all30/comparison.json').read_text())
parser=Parser(str(data),factor=1,normalize=True,load_exposure=False)
ids={name:i for i,name in enumerate(parser.image_names)}
old_parser=Parser('data/realsense_20261030_111354_all30/rgbd/dataset',factor=1,normalize=True,load_exposure=False)
old_ids={name:i for i,name in enumerate(old_parser.image_names)}
ckpt=torch.load(out/'train/ckpts/final_post_update.pt',map_location='cpu',weights_only=True)
assert all(torch.isfinite(v).all() for v in ckpt['splats'].values())
s={k:v.cuda() for k,v in ckpt['splats'].items()}; colors=torch.cat([s['sh0'],s['shN']],dim=1)
old_ckpt=torch.load('results/realsense_20261030_111354_all30/train/ckpts/final_post_update.pt',map_location='cpu',weights_only=True)
old={k:v.cuda() for k,v in old_ckpt['splats'].items()}; old_colors=torch.cat([old['sh0'],old['shN']],dim=1)
long_ckpt=torch.load('results/realsense_20261030_111354_all30_120k/train/ckpts/final_post_update.pt',map_location='cpu',weights_only=True)
long_scene={k:v.cuda() for k,v in long_ckpt['splats'].items()};long_colors=torch.cat([long_scene['sh0'],long_scene['shN']],dim=1)
metrics=dict(psnr=PeakSignalNoiseRatio(data_range=1.).cuda(),
    ssim=StructuralSimilarityIndexMeasure(data_range=1.).cuda(),
    lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda())
rows=[];people_crop=None;preview=out/'comparison_images';preview.mkdir(exist_ok=True)
with torch.no_grad():
    for index,original in enumerate(baseline['per_view']):
        name=original['name']; i=ids[name]; cid=parser.camera_ids[i];w,h=parser.imsize_dict[cid]
        rgb=np.array(Image.open(data/'images'/name).convert('RGB'))
        target=torch.from_numpy(rgb.copy()).float().cuda()/255
        pred,_,_=rasterization(s['means'],s['quats'],torch.exp(s['scales']),torch.sigmoid(s['opacities']),colors,
            torch.from_numpy(np.linalg.inv(parser.camtoworlds[i])[None]).float().cuda(),
            torch.from_numpy(parser.Ks_dict[cid][None]).float().cuda(),w,h,sh_degree=3,packed=True)
        pred=pred[0].clamp(0,1);assert torch.isfinite(pred).all()
        values={k:float(m(pred.permute(2,0,1)[None],target.permute(2,0,1)[None])) for k,m in metrics.items()}
        oi=old_ids[name];ocid=old_parser.camera_ids[oi]
        long_pred,_,_=rasterization(long_scene['means'],long_scene['quats'],torch.exp(long_scene['scales']),torch.sigmoid(long_scene['opacities']),long_colors,
            torch.from_numpy(np.linalg.inv(old_parser.camtoworlds[oi])[None]).float().cuda(),
            torch.from_numpy(old_parser.Ks_dict[ocid][None]).float().cuda(),w,h,sh_degree=3,packed=True)
        long_pred=long_pred[0].clamp(0,1);assert torch.isfinite(long_pred).all()
        long_values={k:float(m(long_pred.permute(2,0,1)[None],target.permute(2,0,1)[None])) for k,m in metrics.items()}
        rows.append(dict(name=name,baseline=original['all30'],baseline_120k=long_values,candidate=values))
        if index in (0,20,40,60,87) or name in ('frame_003197.png','frame_006482.png'):
            pixels=(pred.cpu().numpy()*255).astype('u1')
            old_pred,_,_=rasterization(old['means'],old['quats'],torch.exp(old['scales']),torch.sigmoid(old['opacities']),old_colors,
                torch.from_numpy(np.linalg.inv(old_parser.camtoworlds[oi])[None]).float().cuda(),
                torch.from_numpy(old_parser.Ks_dict[ocid][None]).float().cuda(),w,h,sh_degree=3,packed=True)
            old_pixels=(old_pred[0].clamp(0,1).cpu().numpy()*255).astype('u1')
            long_pixels=(long_pred.cpu().numpy()*255).astype('u1')
            if name=='frame_003197.png':
                # Preselected broad image region around the visible group.
                crop=(slice(160,330),slice(320,500))
                people_crop={'frame':name,'xyxy':[320,160,500,330], 'metrics':{}}
                for label,rendered in [('original_20k',old_pred[0].clamp(0,1)),('original_120k',long_pred),('corrected_20k',pred)]:
                    a=rendered[crop].permute(2,0,1)[None]; b=target[crop].permute(2,0,1)[None]
                    people_crop['metrics'][label]={k:float(metric(a,b)) for k,metric in metrics.items()}
            panel=Image.fromarray(np.concatenate([rgb,old_pixels,long_pixels,pixels],axis=1))
            labeled=Image.new('RGB',(panel.width,panel.height+28),'#202020');labeled.paste(panel,(0,28))
            draw=ImageDraw.Draw(labeled)
            for k,label in enumerate(('Capture','Original 20k','Original 120k','Corrected 20k')):draw.text((k*w+8,7),label,fill='white')
            labeled.save(preview/f'{name}.jpg',quality=92)
        if index%10==0:print('COMPARE',index,'/',len(baseline['per_view']),flush=True)
assert len(rows)==88
mean={label:{m:float(np.mean([r[label][m] for r in rows])) for m in metrics} for label in ('baseline','baseline_120k','candidate')}
report=dict(heldout_views=len(rows),mean_metrics=mean,
    deltas={m:mean['candidate'][m]-mean['baseline'][m] for m in metrics},per_view=rows,
    people_region=people_crop,
    scope='Same 88 held-out RGB images; original 20k and corrected 20k have different poses and initialization. Original 120k has six times the updates. Original 20k metrics from saved independently rendered comparison.json. People-preference trial also excludes late training images. Not a controlled pose-only ablation or independent geometry ground truth.')
(out/'matched_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('MATCHED_COMPARISON',json.dumps(mean),flush=True)
