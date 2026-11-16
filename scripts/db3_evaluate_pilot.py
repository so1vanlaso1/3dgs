"""Reload final R3 checkpoints; common-observation RGB and sensor-depth checks."""
import argparse
import importlib.util
import math
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'examples')]
s=importlib.util.spec_from_file_location('local',ROOT/'scripts/db3_local_geometry.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
s=importlib.util.spec_from_file_location('validation',ROOT/'scripts/db3_validate_local.py');v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
from datasets.colmap import Parser
from gsplat.rendering import rasterization
from db3_acceptance import paired_decisions


def evaluation_protocol(region):
    # Image-relative regions are development inspection regions, not physical
    # metrology annotations. Report each view separately to reveal support gaps.
    regions=({'wall_text_target':[.07,.10,.47,.78], 'opaque_panel_control':[.76,.40,.94,.90],
              'lower_wall_control':[.12,.75,.70,.94]} if region=='matte' else
             {'exhibit_target':[.28,.28,.57,.89], 'posts_control':[.61,.63,.82,.99],
              'floor_control':[.65,.82,.96,.98]})
    return dict(regions_normalized_xyxy=regions,depth='Expected camera-Z / parser normalization scale; valid observed source depth only; missing is unknown',
        common_mapping='Baseline old-rectified render mapped through raw RGB correspondence to SDK rectification. All arms evaluated on same valid intersection. Source IDs fixed; baseline uses historical pose, pilots use fixed localized local pose.',
        geometry='No independent reference. Sensor-depth error, alpha coverage, Gaussian layer offsets and free-view appearance are consistency/defect diagnostics.',
        novel_view_offsets_m=[-.03,.03],novel_yaw_degrees=[-3,3],
        promotion=m.read(m.RUN/'protocol.json')['R3']['promotion'])


def load_splats(path):
    state=torch.load(path,map_location='cpu',weights_only=True)['splats']
    assert all(torch.isfinite(x).all() for x in state.values())
    return {k:x.cuda() for k,x in state.items()}


@torch.no_grad()
def render(splats,C,K,w,h):
    colors=torch.cat((splats['sh0'],splats['shN']),dim=1);degree=int(math.sqrt(colors.shape[1])-1)
    rgb,alpha,_=rasterization(splats['means'],splats['quats'],splats['scales'].exp(),splats['opacities'].sigmoid(),colors,
        torch.from_numpy(np.linalg.inv(C)[None]).float().cuda(),torch.from_numpy(K[None]).float().cuda(),w,h,
        sh_degree=degree,packed=True,render_mode='RGB+ED')
    return rgb[0,:,:,:3].clamp(0,1).cpu().numpy(),rgb[0,:,:,3].cpu().numpy(),alpha[0,:,:,0].cpu().numpy()


def mask_box(box,h,w):
    x0,y0,x1,y1=box;mask=np.zeros((h,w),bool);mask[int(y0*h):int(y1*h),int(x0*w):int(x1*w)]=True;return mask


def errors(image,target,depth,gt,alpha,mask):
    stable=np.isfinite(gt)&(gt>.3)&(gt<6)
    valid=stable&mask
    if not valid.any():return dict(source_samples=0)
    # Penalize unsupported predicted surfaces; never make holes disappear by
    # evaluating only the highest alpha intersection of an arm.
    good=np.isfinite(depth)&(depth>.1)&(alpha>.1)
    delta=np.where(good,np.abs(depth-gt),6.)[valid]
    rgbmask=mask&(target.max(2)<.98)&(target.max(2)>.03)
    mse=np.mean((image[rgbmask]-target[rgbmask])**2) if rgbmask.any() else 1.
    return dict(source_samples=int(valid.sum()),source_fraction=float(valid.sum()/max(1,mask.sum())),
        depth_absolute_error_m_median_p90=[float(x) for x in np.percentile(delta,[50,90])],
        depth_mae_m=float(delta.mean()),depth_within_5cm_fraction=float(np.mean(delta<.05)),
        observed_alpha_coverage=float(np.mean(alpha[valid]>.8)),rgb_psnr_db=float(-10*np.log10(max(mse,1e-12))))


def freeze_protocol(region):
    out=m.RUN/region/'pilot';path=out/'evaluation_protocol.json'
    if path.exists():raise FileExistsError('Evaluation protocol already frozen')
    m.write(path,evaluation_protocol(region))


@torch.no_grad()
def evaluate(region,candidate_dir=None,steps=7000):
    from torchmetrics.functional.image import structural_similarity_index_measure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    out=m.RUN/region/'pilot';suffix='_supported_15k' if candidate_dir else ''
    evidence=out/('evaluation'+suffix);evidence.mkdir(exist_ok=False)
    protocol=m.read(out/'evaluation_protocol.json');dataset=out/'dataset'
    parser=Parser(str(dataset),factor=1,normalize=True,load_exposure=False)
    old_parser=Parser(str(m.ROOT/'data/realsense_20261030_134523_clean/rgbd/dataset'),factor=1,normalize=True,load_exposure=False)
    split=m.read(dataset/'experiment_split.json');scale=float(np.linalg.norm(parser.transform[:3,0]));old_scale=float(np.linalg.norm(old_parser.transform[:3,0]))
    cal=m.read(m.DATA/'calibration_clean.json');K=np.array(cal['K']);w,h=cal['width'],cal['height']
    yy,xx=np.mgrid[:h,:w];pixels=np.c_[xx.ravel(),yy.ravel()]
    old_pixels=v.legacy_pixels(pixels,K,cal).reshape(h,w,2).astype('f4')
    common=(old_pixels[:,:,0]>=1)&(old_pixels[:,:,0]<w-2)&(old_pixels[:,:,1]>=1)&(old_pixels[:,:,1]<h-2)
    np.savez_compressed(evidence/'common_mapping.npz',old_pixel_xy=old_pixels,valid_intersection=common)
    rows={f['name']:f for f in m.read(m.DATA/'manifest.json')['frames']}
    frozen=m.read(m.R1/'baseline_manifest.json');base=next(x for x in frozen['files'] if x['source'].endswith('final_post_update.pt'))
    current_directory='current_safe' if (out/'current_safe').exists() else 'current'
    checkpoint_paths={'baseline':m.R1/'baseline'/base['source'],**{a:out/(current_directory if a=='current' else (candidate_dir if a=='confidence' and candidate_dir else a))/'ckpts/final_post_update.pt' for a in ('off','current','confidence')}}
    lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda();results={};panels={n:[] for n in split['val']};experiments={}
    for arm,path in checkpoint_paths.items():
        splats=load_splats(path);arm_results=[];global_rgb=[]
        if arm!='baseline':
            directory=current_directory if arm=='current' else (candidate_dir if arm=='confidence' and candidate_dir else arm)
            exp=m.read(out/directory/'experiment.json');expected=steps if arm=='confidence' else 7000
            assert exp['completed_updates']==expected and len(exp['camera_sequence'])==expected
            experiments[arm]=dict(initial_sha256=exp['initial_sha256'],camera_sequence_sha256=exp['camera_sequence_sha256'],completed_updates=exp['completed_updates'],final_count=len(splats['means']))
        for name in split['val']:
            i=parser.image_names.index(name);C=parser.camtoworlds[i];f=rows[name]
            if arm=='baseline':
                j=old_parser.image_names.index(f['selected_name']);rgb,depth,alpha=render(splats,old_parser.camtoworlds[j],K,w,h)
                rgb=cv2.remap(rgb,old_pixels[:,:,0],old_pixels[:,:,1],cv2.INTER_LINEAR)
                depth=cv2.remap(depth/old_scale,old_pixels[:,:,0],old_pixels[:,:,1],cv2.INTER_LINEAR)
                alpha=cv2.remap(alpha,old_pixels[:,:,0],old_pixels[:,:,1],cv2.INTER_LINEAR)
            else:
                rgb,depth,alpha=render(splats,C,K,w,h);depth/=scale
            target=np.array(Image.open(m.DATA/'images'/name))/255.
            gt=np.load(m.DATA/'depth'/(name+'.npz'))['depth_m']
            gt_valid=gt>.3;lo=cv2.erode(gt,np.ones((3,3),'u1'));hi=cv2.dilate(gt,np.ones((3,3),'u1'))
            stable=(lo>.3)&(hi<6)&((hi-lo)<np.maximum(.03,.02*gt));gt=np.where(stable,gt,0)
            Image.fromarray((rgb*255).astype('u1')).save(evidence/f'{arm}_{name}')
            np.savez_compressed(evidence/f'{arm}_{name}.npz',depth_m=depth,alpha=alpha)
            regions={label:errors(rgb,target,depth,gt,alpha,common&mask_box(box,h,w)) for label,box in protocol['regions_normalized_xyxy'].items()}
            # Same rectangular intersection for full-image learned metrics; image
            # corners outside the intersection are identically black in all arms.
            p=torch.from_numpy(np.where(common[:,:,None],rgb,0)).permute(2,0,1)[None].float().cuda()
            q=torch.from_numpy(np.where(common[:,:,None],target,0)).permute(2,0,1)[None].float().cuda()
            mse=(p-q).square().sum()/(common.sum()*3)
            metrics=dict(psnr=float(-10*torch.log10(mse)),ssim=float(structural_similarity_index_measure(p,q,data_range=1.)),lpips=float(lpips(p,q)))
            global_rgb.append(metrics);arm_results.append(dict(name=name,original_source=f['selected_name'],source_id=f['rgb_message_id'],regions=regions,appearance=metrics))
            thumbnail=Image.fromarray((rgb*255).astype('u1'));thumbnail.thumbnail((417,232));panels[name].append((arm,thumbnail))
            if arm!='baseline':
                # Gaussian layer residuals in the source camera: diagnostic only.
                W=np.linalg.inv(C);xyz=splats['means'].cpu().numpy();q3=xyz@W[:3,:3].T+W[:3,3]
                uv=q3[:,:2]/np.maximum(q3[:,2:3],.01)*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
                sd,sv,_=m.sample_depth(gt,uv);sv &= (q3[:,2]/scale>.3)&(splats['opacities'].sigmoid().cpu().numpy().reshape(-1)>.05)
                offsets=(q3[:,2]/scale-sd)[sv]
                arm_results[-1]['gaussian_source_surface_offset_m_median_p90']=[float(x) for x in np.percentile(np.abs(offsets),[50,90])] if len(offsets) else None
            if arm=='confidence':
                for amount in (-.03,.03):
                    novel=C.copy();novel[:3,3]+=C[:3,0]*amount*scale
                    nr,_,_=render(splats,novel,K,w,h);Image.fromarray((nr*255).astype('u1')).save(evidence/f'novel_x{amount:+.2f}_{name}')
                for yaw in (-3.,3.):
                    angle=np.deg2rad(yaw);R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
                    novel=C.copy();novel[:3,:3]=C[:3,:3]@R
                    nr,_,_=render(splats,novel,K,w,h);Image.fromarray((nr*255).astype('u1')).save(evidence/f'novel_yaw{yaw:+.0f}_{name}')
        results[arm]=dict(checkpoint_sha256=m.sha(path),gaussians=len(splats['means']),views=arm_results,
            mean_appearance={k:float(np.mean([r[k] for r in global_rgb])) for k in ('psnr','ssim','lpips')})
        del splats;torch.cuda.empty_cache();print('EVALUATED',region,arm,results[arm]['mean_appearance'],flush=True)
    assert len({e['initial_sha256'] for e in experiments.values()})==1
    if not candidate_dir:
        assert len({e['camera_sequence_sha256'] for e in experiments.values()})==1
    else:
        candidate_replay=m.read(out/candidate_dir/'experiment.json')['camera_sequence']
        assert candidate_replay[:7000]==m.read(out/'off/experiment.json')['camera_sequence']
    # Evaluate the full frozen source roster. Missing or unsupported regions fail.
    decisions=paired_decisions(results,split['val'],protocol['regions_normalized_xyxy'],matched_compute=not bool(candidate_dir))
    gate=decisions['gates'];pipeline=decisions['pipeline_against_baseline']
    target_changes=decisions['target_changes'];control_changes=decisions['control_changes']
    m.write(out/('paired_evaluation'+suffix+'.json'),dict(region=region,results=results,experiments=experiments,gates=gate,
        target_changes=target_changes,control_changes=control_changes,
        quantitative_promotion_pass=(all(x for k,x in gate.items() if k!='visual_review_completed') if not candidate_dir else None),pipeline_against_baseline=pipeline,
        candidate_directory=candidate_dir or 'confidence',candidate_steps=steps,unequal_compute_to_7000_policy_controls=bool(candidate_dir),
        absolute_accuracy=False,whole_room_improvement=False,protocol_sha256=m.sha(out/'evaluation_protocol.json')))
    sheet=Image.new('RGB',(417*5,262*len(panels)),'white');draw=ImageDraw.Draw(sheet)
    for j,(name,images) in enumerate(panels.items()):
        target=Image.open(m.DATA/'images'/name);target.thumbnail((417,232));all_images=[('captured',target),*images]
        for k,(label,im) in enumerate(all_images):
            sheet.paste(im,(k*417,j*262+30));draw.text((k*417+5,j*262+7),f'{label} {rows[name]["selected_name"]}',fill='black')
    sheet.save(evidence/'paired_contact_sheet.jpg',quality=94)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['freeze','evaluate']);p.add_argument('--region',choices=['matte','exhibit'],required=True);p.add_argument('--candidate-dir',choices=['confidence_supported_15k']);p.add_argument('--steps',type=int,default=7000);a=p.parse_args()
    if a.stage=='freeze':freeze_protocol(a.region)
    else:evaluate(a.region,a.candidate_dir,a.steps)
