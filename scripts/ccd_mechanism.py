"""CCD-GS H1/H2 pilot with full-state replay and matched branch controls.

This is a mechanism study, not an end-to-end CCD-GS allocation implementation.
All RGB optimization is train-only; utility uses the locked development split.
"""
import argparse
import csv
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'examples'),str(REPO)]
from datasets.colmap import Dataset, Parser
from simple_trainer import create_splats_with_optimizers
from utils import CameraOptModule, set_random_seed
from gsplat.losses import ssim_loss
from gsplat.rendering import rasterization
from gsplat.strategy import BudgetStrategy
from gsplat.strategy.ops import split


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8'); temp.replace(path)


def cpu(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)


def device_tree(value):
    if torch.is_tensor(value): return value.cuda()
    if isinstance(value,dict): return {k:device_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [device_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(device_tree(v) for v in value)
    return copy.deepcopy(value)


def digest(value):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            h.update(str((str(v.dtype),tuple(v.shape))).encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,np.ndarray): h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str): h.update(str(k).encode()); visit(v[k])
        elif isinstance(v,(tuple,list)):
            for x in v: visit(x)
        else: h.update(repr(v).encode())
    visit(value); return h.hexdigest()


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def rigid(rotation,translation):
    """Right-composed SE(3) with rotation-vector exponential, no small-angle divide."""
    x,y,z=rotation.unbind(-1)
    zero=x*0
    skew=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero],-1).reshape(*rotation.shape[:-1],3,3)
    matrix=torch.eye(4,device=rotation.device,dtype=rotation.dtype).repeat(*rotation.shape[:-1],1,1)
    matrix[...,:3,:3]=torch.matrix_exp(skew); matrix[...,:3,3]=translation
    return matrix


def pose_error(pose,reference,scale):
    relative=pose[...,:3,:3].transpose(-1,-2)@reference[...,:3,:3]
    cosine=((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
    return dict(rotation_degrees=float(torch.rad2deg(torch.acos(cosine)).mean()),
                translation_scene_scale=float((pose[...,:3,3]-reference[...,:3,3]).norm(dim=-1).mean()/scale))


class Lineage:
    def __init__(self,n):
        self.ids=torch.arange(n,device='cuda',dtype=torch.long); self.next=n; self.events=[]

    def children(self,parents):
        ids=torch.arange(self.next,self.next+len(parents),device='cuda'); self.next+=len(parents)
        self.events.append(dict(children=ids.cpu().tolist(),parents=parents.cpu().tolist()))
        return ids

    def on_duplicate(self,sel): self.ids=torch.cat([self.ids,self.children(self.ids[sel])])
    def on_split(self,sel,rest): self.ids=torch.cat([self.ids[rest],self.children(self.ids[sel].repeat(2))])
    def on_remove(self,mask): self.ids=self.ids[~mask]


class Study:
    def __init__(self,scene,seed,condition,smoke=False):
        self.scene=scene; self.seed=seed; self.condition=condition; self.smoke=smoke
        set_random_seed(seed)
        manifest_path=REPO/f'stage_result/stage_0/manifests/{scene}_splits.json'
        self.manifest=json.loads(manifest_path.read_text())
        self.parser=Parser(str(REPO/self.manifest['data_path_relative_to_repo']),factor=1,normalize=True,test_every=8,load_exposure=False)
        self.train=Dataset(self.parser,split='train'); self.dev=Dataset(self.parser,split='val')
        self.train.indices=np.asarray(self.manifest['parser_indices']['train'])
        self.dev.indices=np.asarray(self.manifest['parser_indices']['development'])
        for role in ('train','development','test'):
            actual=[f'{scene}:{self.parser.image_names[i]}' for i in self.manifest['parser_indices'][role]]
            assert actual==self.manifest['view_ids'][role], 'Manifest/parser identity mismatch'
        test=set(self.manifest['parser_indices']['test'])
        assert not(set(self.train.indices)&set(self.dev.indices)|set(self.train.indices)&test|set(self.dev.indices)&test)
        self.train.preload_images(); self.dev.preload_images()
        self.scale=float(self.parser.scene_scale*1.1)
        self.clean=torch.from_numpy(self.parser.camtoworlds[self.train.indices]).float().cuda()
        g=np.random.default_rng(seed+991)
        rv=g.normal(size=(len(self.train),3)); rv/=np.linalg.norm(rv,axis=-1,keepdims=True)
        tv=g.normal(size=(len(self.train),3)); tv/=np.linalg.norm(tv,axis=-1,keepdims=True)
        rm=g.uniform(size=(len(self.train),1)); tm=g.uniform(size=(len(self.train),1))
        deg,translation={'clean':(0.,0.),'mild':(.5,.0025),'moderate':(2.,.01)}[condition]
        self.disturbance=rigid(torch.tensor(rv*rm*np.deg2rad(deg),device='cuda',dtype=torch.float32),
                               torch.tensor(tv*tm*translation*self.scale,device='cuda',dtype=torch.float32))
        self.cameras=self.clean@self.disturbance
        self.total=3000 if scene=='analytic_spheres_v1' else 7000
        self.checkpoints=[750,1500] if scene=='analytic_spheres_v1' else [1750,3500]
        if smoke: self.total=60; self.checkpoints=[20,40]
        self.params,self.opts=create_splats_with_optimizers(self.parser,scene_scale=self.scale,sh_degree=3)
        self.initial_sha256=digest(self.params.state_dict())
        self.pose=CameraOptModule(len(self.train)).cuda(); self.pose.zero_init()
        self.pose_opt=torch.optim.Adam(self.pose.parameters(),lr=1e-5,weight_decay=1e-6)
        self.schedulers=[torch.optim.lr_scheduler.ExponentialLR(self.opts['means'],gamma=.01**(1/self.total)),
                         torch.optim.lr_scheduler.ExponentialLR(self.pose_opt,gamma=.01**(1/self.total))]
        self.strategy=BudgetStrategy(cap_max=5120 if scene=='analytic_spheres_v1' else 200000,
            gradient_weight=1,error_weight=0,edge_weight=0,clone_policy='clone',split_mode='random',
            refine_start_iter=100 if not smoke else 5,
            refine_stop_iter=2500 if scene=='analytic_spheres_v1' else 6000,
            refine_every=100 if not smoke else 10,reset_every=3000)
        self.strategy_state=self.strategy.initialize_state(self.scale)
        self.lineage=Lineage(len(self.params['means'])); self.step=0
        g=np.random.default_rng(seed+104729)
        # Full future schedule is saved; branches never consume shared RNG to sample views.
        self.sequence=np.concatenate([g.permutation(len(self.train)) for _ in range((self.total+1500)//len(self.train)+2)])
        self.mapping=[self.parser.image_names[i] for i in self.train.indices]

    def frame(self,item,dev=False):
        frame=(self.dev if dev else self.train)[item]
        return dict(camera=frame['camtoworld'].cuda()[None] if dev else self.cameras[item:item+1],
                    K=frame['K'].cuda()[None],pixels=frame['image'].cuda().float()[None]/255,item=item)

    def current_camera(self,item):
        return self.pose(self.cameras[item:item+1],torch.tensor([item],device='cuda'))

    def render(self,frame,camera=None):
        h,w=frame['pixels'].shape[1:3]
        colors,alpha,info=rasterization(means=self.params['means'],quats=self.params['quats'],
            scales=self.params['scales'].exp(),opacities=self.params['opacities'].sigmoid(),
            colors=torch.cat([self.params['sh0'],self.params['shN']],1),
            sh_degree=min(self.step//1000,3),viewmats=torch.linalg.inv(camera if camera is not None else frame['camera']),
            Ks=frame['K'],width=w,height=h,packed=True,near_plane=.01,far_plane=1e10)
        return colors,info

    @staticmethod
    def loss(rgb,gt):
        return .8*(rgb-gt).abs().mean()+.2*ssim_loss(rgb.permute(0,3,1,2),gt.permute(0,3,1,2))

    def train_step(self,structural=True):
        item=int(self.sequence[self.step]); frame=self.frame(item)
        rgb,info=self.render(frame,self.current_camera(item))
        if structural: self.strategy.step_pre_backward(self.params,self.opts,self.strategy_state,self.step,info)
        loss=self.loss(rgb,frame['pixels']); loss.backward()
        for opt in [*self.opts.values(),self.pose_opt]: opt.step(); opt.zero_grad(set_to_none=True)
        for scheduler in self.schedulers: scheduler.step()
        if structural:
            self.strategy.step_post_backward(self.params,self.opts,self.strategy_state,self.step,info,packed=True,scene=self.lineage)
        self.step+=1
        return float(loss.detach())

    def snapshot(self):
        return cpu(dict(params=self.params.state_dict(),optimizers={k:o.state_dict() for k,o in self.opts.items()},
            pose=self.pose.state_dict(),pose_optimizer=self.pose_opt.state_dict(),
            schedulers=[s.state_dict() for s in self.schedulers],strategy=vars(self.strategy),strategy_state=self.strategy_state,
            step=self.step,ids=self.lineage.ids,lineage=self.lineage.events,next_id=self.lineage.next,
            cameras=self.cameras,disturbance=self.disturbance,sample_sequence=self.sequence,
            item_to_view=self.mapping,manifest=self.manifest,rng=rng_state()))

    def restore(self,snapshot):
        self.params=torch.nn.ParameterDict({k:torch.nn.Parameter(v.cuda().clone()) for k,v in snapshot['params'].items()})
        self.opts={}
        for k,state in snapshot['optimizers'].items():
            opt=torch.optim.Adam([self.params[k]],lr=state['param_groups'][0]['lr'],eps=1e-15,fused=True)
            opt.load_state_dict(copy.deepcopy(state)); self.opts[k]=opt
        self.pose.load_state_dict(device_tree(snapshot['pose']))
        self.pose_opt=torch.optim.Adam(self.pose.parameters(),lr=1e-5,weight_decay=1e-6)
        self.pose_opt.load_state_dict(copy.deepcopy(snapshot['pose_optimizer']))
        self.schedulers=[torch.optim.lr_scheduler.ExponentialLR(self.opts['means'],gamma=.01**(1/self.total)),
                         torch.optim.lr_scheduler.ExponentialLR(self.pose_opt,gamma=.01**(1/self.total))]
        for scheduler,state in zip(self.schedulers,snapshot['schedulers']): scheduler.load_state_dict(state)
        self.strategy=BudgetStrategy(**snapshot['strategy']); self.strategy_state=device_tree(snapshot['strategy_state'])
        self.step=snapshot['step']; self.lineage=Lineage(0)
        self.lineage.ids=snapshot['ids'].cuda().clone(); self.lineage.next=snapshot['next_id']; self.lineage.events=copy.deepcopy(snapshot['lineage'])
        self.cameras=snapshot['cameras'].cuda().clone(); self.disturbance=snapshot['disturbance'].cuda().clone()
        self.sequence=snapshot['sample_sequence'].copy(); self.mapping=snapshot['item_to_view'].copy()
        restore_rng(snapshot['rng'])

    @torch.no_grad()
    def evaluate(self):
        errors=[]
        for i in range(len(self.dev)):
            frame=self.frame(i,dev=True); rgb,_=self.render(frame)
            errors.append(float((rgb.clamp(0,1)-frame['pixels']).square().mean()))
        return dict(mse=float(np.mean(errors)),mean_psnr=float(np.mean(-10*np.log10(np.maximum(errors,1e-12)))),per_view_mse=errors)

    def gradient(self,item,camera,projection=False):
        frame=self.frame(item); rgb,info=self.render(frame,camera)
        loss=self.loss(rgb,frame['pixels'])
        grad=torch.autograd.grad(loss,info['means2d'])[0].detach().clone()
        grad[:,0]*=info['width']/2; grad[:,1]*=info['height']/2
        n=len(self.params['means']); result=torch.zeros(n,device='cuda'); radii=torch.zeros_like(result)
        ids=info['gaussian_ids'].long()
        result.index_add_(0,ids,grad.norm(dim=-1)); radii[ids]=info['radii'].max(-1).values.float()/max(info['width'],info['height'])
        if projection:
            centers=torch.zeros(n,2,device='cuda'); centers[ids]=info['means2d'].detach()
            return result,radii,float(loss.detach()),centers
        return result,radii,float(loss.detach())

    def probe(self,item,updates=5):
        camera=self.current_camera(item).detach(); frame=self.frame(item)
        delta=torch.nn.Parameter(torch.zeros(1,6,device='cuda'))
        opt=torch.optim.Adam([delta],lr=.002)
        rotation_bound=math.radians(1); translation_bound=.005*self.scale
        torch.cuda.synchronize(); tic=time.perf_counter()
        for _ in range(updates):
            adjusted=camera@rigid(delta[:,:3],delta[:,3:])
            rgb,_=self.render(frame,adjusted)
            photo=self.loss(rgb,frame['pixels'])
            prior=.001*((delta[:,:3]/rotation_bound).square().mean()+(delta[:,3:]/translation_bound).square().mean())
            delta.grad=torch.autograd.grad(photo+prior,delta)[0]; opt.step(); opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                delta[:,:3]*=(rotation_bound/delta[:,:3].norm(dim=-1,keepdim=True).clamp_min(1e-12)).clamp_max(1)
                delta[:,3:]*=(translation_bound/delta[:,3:].norm(dim=-1,keepdim=True).clamp_min(1e-12)).clamp_max(1)
        adjusted=(camera@rigid(delta[:,:3],delta[:,3:])).detach()
        assert float(delta[:,:3].detach().norm())<=rotation_bound*1.00001
        assert float(delta[:,3:].detach().norm())<=translation_bound*1.00001
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-tic
        return adjusted,dict(seconds=elapsed,updates=updates,delta=delta.detach().cpu().tolist(),
            displacement=pose_error(adjusted,camera,self.scale),
            reference_before=pose_error(camera,self.clean[item:item+1],self.scale),
            reference_after=pose_error(adjusted,self.clean[item:item+1],self.scale))

    def split_ids(self,ids,seed):
        if not len(ids): return
        original_rng=rng_state(); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        mask=torch.zeros(len(self.params['means']),dtype=torch.bool,device='cuda'); mask[ids]=True
        split(self.params,self.opts,self.strategy_state,mask,mode='random',scene=self.lineage)
        restore_rng(original_rng)

    def branch(self,snapshot,ids,horizons,seed,delayed=False):
        self.restore(snapshot)
        if not delayed: self.split_ids(ids,seed)
        measurements={}; started=time.perf_counter()
        for index in range(max(horizons)):
            if delayed and index==max(horizons)//2: self.split_ids(ids,seed)
            self.train_step(structural=False)
            if index+1 in horizons:
                measurements[str(index+1)]=self.evaluate()
                with torch.no_grad():
                    measurements[str(index+1)]['pose_reference_error']=pose_error(
                        self.pose(self.cameras,torch.arange(len(self.train),device='cuda')),self.clean,self.scale)
        torch.cuda.synchronize()
        return dict(measurements=measurements,seconds=time.perf_counter()-started,
                    final_count=len(self.params['means']),final_parameter_digest=digest(self.params.state_dict()))


def correlation(a,b):
    if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0:
        return None
    value=float(spearmanr(a,b).statistic)
    return value if np.isfinite(value) else None


def candidate_set(study,g0,gplus,r0,rplus,multi,count,seed):
    scale=study.params['scales'].exp().max(-1).values.detach()
    opacity=study.params['opacities'].sigmoid().detach()
    valid=(g0>0)&(r0>0)&(rplus>0)&(scale>.01*study.scale)&(opacity>=.005)
    ids=valid.nonzero().flatten().cpu().numpy()
    if len(ids)<count:
        # Mechanism labels remain meaningful for smaller parents, but record it.
        valid=(g0>0)&(r0>0)&(rplus>0)&(opacity>=.005); ids=valid.nonzero().flatten().cpu().numpy()
        selection='Visible parents spanning all scales because fewer than requested normally-split parents were visible.'
    else: selection='Visible large parents satisfying the split scale criterion, sampled across g0 quartiles including low-gradient controls.'
    count=min(count,len(ids)); rng=np.random.default_rng(seed)
    ordered=ids[np.argsort(g0[ids].cpu().numpy(),kind='stable')]
    strata=np.array_split(ordered,4); selected=[]
    for group in strata:
        take=min(len(group),count//4)
        selected.extend(rng.choice(group,take,replace=False).tolist())
    rest=np.setdiff1d(ids,np.asarray(selected,dtype=int))
    if len(selected)<count: selected.extend(rng.choice(rest,count-len(selected),replace=False).tolist())
    return selected,selection


def checkpoint_study(study,snapshot,folder,smoke=False):
    report_path=folder/'result.json'
    if report_path.exists(): return json.loads(report_path.read_text())
    folder.mkdir(parents=True,exist_ok=True)
    short,long=(5,10) if smoke else (200,1000)
    study.restore(snapshot)
    before_hash=digest(study.snapshot())
    anchor=int(study.sequence[study.step])
    camera=study.current_camera(anchor).detach()
    torch.cuda.synchronize(); tic=time.perf_counter()
    g0,r0,loss0,centers0=study.gradient(anchor,camera,projection=True)
    corrected,probe=study.probe(anchor)
    gplus,rplus,lossplus,centersplus=study.gradient(anchor,corrected,projection=True)
    torch.cuda.synchronize(); probe_budget=time.perf_counter()-tic
    zero,rzero,_=study.gradient(anchor,study.probe(anchor,0)[0])
    repeated,_=study.probe(anchor)
    grepeat,_,_=study.gradient(anchor,repeated)
    torch.cuda.synchronize(); validation_seconds=time.perf_counter()-tic-probe_budget
    multi=torch.zeros_like(g0)
    for item in [anchor,*[int(x) for x in study.sequence[study.step+1:study.step+4]]]:
        multi+=study.gradient(item,study.current_camera(item).detach())[0]/4
    assert digest(study.snapshot())==before_hash, 'Diagnostic mutated training state'
    candidates,selection=candidate_set(study,g0,gplus,r0,rplus,multi,4 if smoke else 24,study.seed+study.step)
    if len(candidates)<4: raise RuntimeError('Insufficient visible candidates for mechanism labels')
    selected=np.asarray(candidates)
    scores=dict(original=g0[selected].cpu().tolist(),corrected=gplus[selected].cpu().tolist(),
        ccd=(g0[selected]*torch.clamp(gplus[selected]/(g0[selected]+1e-12),0,1)).cpu().tolist(),
        persistence=(gplus[selected]/(g0[selected]+1e-12)).cpu().tolist(),
        multiview=multi[selected].cpu().tolist())
    top=max(1,len(candidates)//4)
    original_top=set(np.argsort(-np.asarray(scores['original']),kind='stable')[:top])
    corrected_top=set(np.argsort(-np.asarray(scores['corrected']),kind='stable')[:top])
    repeated_top=set(np.argsort(-grepeat[selected].cpu().numpy(),kind='stable')[:top])
    h1=dict(state_unchanged=True,zero_max_abs=float((g0-zero).abs().max()),
        repeated_max_abs=float((gplus-grepeat).abs().max()),
        max_gradient_change=float((gplus-g0).abs().max()),
        original_corrected_spearman=correlation(scores['original'],scores['corrected']),
        repeat_spearman=correlation(gplus[selected].cpu().numpy(),grepeat[selected].cpu().numpy()),
        original_corrected_topk_overlap=len(original_top&corrected_top)/top,
        corrected_repeat_topk_overlap=len(corrected_top&repeated_top)/top,
        visible_before_count=int((r0>0).sum()),visible_after_count=int((rplus>0).sum()),
        common_visibility_count=int(((r0>0)&(rplus>0)).sum()),
        stable_radius_fraction=float(((rplus[selected]-r0[selected]).abs()/r0[selected].clamp_min(1e-12)<.1).float().mean()),
        loss_before=loss0,loss_after=lossplus,probe=probe,measured_diagnostic_seconds=probe_budget,
        repeat_validation_seconds=validation_seconds)
    info=dict(scene=study.scene,seed=study.seed,condition=study.condition,checkpoint=snapshot['step'],
        manifest_counts=study.manifest['counts'],anchor_item=anchor,anchor_view=study.mapping[anchor],
        snapshot_digest=before_hash,candidates=candidates,
        candidate_stable_ids=snapshot['ids'][selected].tolist(),selection=selection,scores=scores,h1=h1,
        attributes=dict(scale=study.params['scales'].detach().exp().max(-1).values[selected].cpu().tolist(),
                        opacity=study.params['opacities'].detach().sigmoid()[selected].cpu().tolist(),
                        projection_observations=snapshot['strategy_state']['count'][selected].tolist(),
                        projected_displacement_pixels=(centersplus[selected]-centers0[selected]).norm(dim=-1).cpu().tolist(),
                        radius_before=r0[selected].cpu().tolist(),radius_after=rplus[selected].cpu().tolist()))
    with torch.no_grad():
        info['warmup_reference_pose_error']=pose_error(study.pose(study.cameras,torch.arange(len(study.train),device='cuda')),study.clean,study.scale)
    write(folder/'diagnostic.json',info)
    del g0,gplus,zero,rzero,grepeat,multi,centers0,centersplus
    replay1=study.branch(snapshot,[],[short,long],study.seed)
    replay2=study.branch(snapshot,[],[short,long],study.seed)
    noise=abs(replay1['measurements'][str(short)]['mse']-replay2['measurements'][str(short)]['mse'])
    info['replay']=dict(first=replay1,second=replay2,short_mse_difference=noise,
        parameter_digest_equal=replay1['final_parameter_digest']==replay2['final_parameter_digest'])
    candidate_path=folder/'candidate_results.json'
    labels=json.loads(candidate_path.read_text()) if candidate_path.exists() else []
    for index,parent in enumerate(candidates):
        if index<len(labels): continue
        horizons=[short,long] if index%4==0 else [short]
        split_seed=study.seed*1000000+snapshot['step']*100+index
        result=study.branch(snapshot,[parent],horizons,split_seed)
        utility={h:replay1['measurements'][h]['mse']-m['mse'] for h,m in result['measurements'].items()}
        labels.append(dict(index=index,parent=parent,stable_id=info['candidate_stable_ids'][index],split_seed=split_seed,utility=utility,**result))
        write(candidate_path,labels)
        print('CCD_LABEL',study.scene,study.seed,study.condition,snapshot['step'],index+1,'/',len(candidates),flush=True)
    k=max(1,len(candidates)//4)
    utilities=np.array([r['utility'][str(short)] for r in labels])
    rankings={name:np.argsort(-np.asarray(values),kind='stable')[:k].tolist() for name,values in scores.items()}
    ranking_stats={name:dict(topk_indices=indices,topk_mean_utility=float(utilities[indices].mean()),
        utility_spearman=correlation(values,utilities)) for (name,values),indices in zip(scores.items(),rankings.values())}
    random_top=np.random.default_rng(study.seed+study.step+391).choice(len(candidates),k,replace=False).tolist()
    rankings['random']=random_top
    group_path=folder/'group_controls.json'
    groups=json.loads(group_path.read_text()) if group_path.exists() else {}
    for name in ['original','corrected','ccd','multiview','random','delayed','retained','equal_compute_pose']:
        if name in groups: continue
        intervention=snapshot
        indices=rankings['original' if name in ('delayed','retained','equal_compute_pose') else name]
        extra_seconds=0.
        if name in ('retained','equal_compute_pose'):
            study.restore(snapshot)
            torch.cuda.synchronize(); extra_start=time.perf_counter()
            if name=='retained':
                # Accept the same physical correction as an update of the base
                # camera, preserving pose embedding and optimizer moments.
                delta=rigid(torch.tensor(probe['delta'],device='cuda')[:,:3],torch.tensor(probe['delta'],device='cuda')[:,3:])
                current=study.current_camera(anchor).detach()
                actual=torch.linalg.solve(study.cameras[anchor:anchor+1],current)
                study.cameras[anchor:anchor+1]=study.cameras[anchor:anchor+1]@actual@delta@torch.linalg.inv(actual)
                extra_iterations=5
            else:
                extra_iterations=0
                while time.perf_counter()-extra_start<probe_budget or extra_iterations==0:
                    frame=study.frame(anchor); rgb,_=study.render(frame,study.current_camera(anchor))
                    loss=study.loss(rgb,frame['pixels'])
                    study.pose.embeds.weight.grad=torch.autograd.grad(loss,study.pose.embeds.weight)[0]
                    study.pose_opt.step(); study.pose_opt.zero_grad(set_to_none=True)
                    extra_iterations+=1
                    torch.cuda.synchronize()
            extra_score=study.gradient(anchor,study.current_camera(anchor).detach())[0][selected].cpu().numpy()
            indices=np.argsort(-extra_score,kind='stable')[:k].tolist()
            torch.cuda.synchronize(); extra_seconds=time.perf_counter()-extra_start
            if name=='retained': extra_seconds+=probe['seconds']
            intervention=study.snapshot()
        parents=[candidates[i] for i in indices]
        result=study.branch(intervention,parents,[short],study.seed*100000+snapshot['step'],delayed=name=='delayed')
        groups[name]=dict(selected=parents,extra_seconds=extra_seconds,
             extra_iterations=extra_iterations if name in ('retained','equal_compute_pose') else 0,
             utility=replay1['measurements'][str(short)]['mse']-result['measurements'][str(short)]['mse'],**result)
        write(group_path,groups)
    base=replay1['measurements'][str(short)]['mse']
    ccd=ranking_stats['ccd']['topk_mean_utility']
    excess={name:(ccd-ranking_stats[name]['topk_mean_utility'])/max(base,1e-12) for name in ('original','corrected','multiview')}
    info.update(labels=labels,ranking_stats=ranking_stats,group_controls=groups,
        normalized_excess_utility=excess,short_horizon=short,long_horizon=long,
        baseline_development_mse=base,noise_floor=noise,
        primary_threshold=.005,per_checkpoint_practical_gate=all(v>=.005 for v in excess.values()) and
            all(ccd-ranking_stats[name]['topk_mean_utility']>5*noise for name in excess))
    write(report_path,info)
    return info


def summarize(root,records,smoke):
    pooled_noise95=float(np.quantile([r['noise_floor'] for r in records],.95)) if records else 0.
    for r in records:
        r['practical_gate_pooled_noise']=all(v>=.005 for v in r['normalized_excess_utility'].values()) and all(
            r['ranking_stats']['ccd']['topk_mean_utility']-r['ranking_stats'][name]['topk_mean_utility']>5*pooled_noise95
            for name in ('original','corrected','multiview'))
    long_comparisons={}; strong_comparisons={}
    for r in records:
        key=f'{r["scene"]}/checkpoint{r["checkpoint"]}'
        indices=[i for i,label in enumerate(r['labels']) if str(r['long_horizon']) in label['utility']]
        long_k=max(1,len(indices)//4)
        long_base=r['replay']['first']['measurements'][str(r['long_horizon'])]['mse']
        utilities={i:r['labels'][i]['utility'][str(r['long_horizon'])] for i in indices}
        top_util={}
        for name in ('ccd','original','corrected'):
            top=sorted(indices,key=lambda i:r['scores'][name][i],reverse=True)[:long_k]
            top_util[name]=float(np.mean([utilities[i] for i in top]))
        long_comparisons.setdefault(key,[]).append({name:(top_util['ccd']-top_util[name])/max(long_base,1e-12) for name in ('original','corrected')})
        groups=r['group_controls']
        strong_comparisons.setdefault(key,[]).append({name:(groups['ccd']['utility']-groups[name]['utility'])/max(r['baseline_development_mse'],1e-12)
            for name in ('retained','equal_compute_pose')})
    long_means={key:{name:float(np.mean([row[name] for row in values])) for name in ('original','corrected')}
                for key,values in long_comparisons.items()}
    strong_means={key:{name:float(np.mean([row[name] for row in values])) for name in ('retained','equal_compute_pose')}
                  for key,values in strong_comparisons.items()}
    no_long_reversal=all(value>=0 for row in long_means.values() for value in row.values())
    strong_survive=all(value>0 for row in strong_means.values() for value in row.values())
    lines=['# CCD-GS mechanism pilot','',
        'Executed bounded pose probes and conditional split/no-split continuations. The primary CCD score was fixed prospectively as g0 * clip(gplus/(g0+1e-12),0,1). It is algebraically close to capped corrected magnitude; an advantage beyond corrected magnitude must be measured.', '',
        'All warmups and continuations include the same ordinary camera optimizer. Probe corrections are discarded in the primary diagnostic. Additional controls retain the correction or spend the measured diagnostic wall time on ordinary pose-only updates. Utility is mean development RGB MSE(no split) minus MSE(split). Gaussian growth is disabled during the continuation except the declared intervention.', '',
        '| Scene | Condition | Seed | Checkpoint | Candidates | Replay MSE difference | Excess vs g0 | Excess vs gplus | Excess vs multi-view | Gate |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in records:
        d=r['normalized_excess_utility']
        lines.append(f'| {r["scene"]} | {r["condition"]} | {r["seed"]} | {r["checkpoint"]} | {len(r["candidates"])} | {r["noise_floor"]:.3g} | {d["original"]:+.5f} | {d["corrected"]:+.5f} | {d["multiview"]:+.5f} | {r["practical_gate_pooled_noise"]} |')
    complete=len(records)==36 and all(len(r['candidates'])==24 for r in records) and not smoke
    passed=sum(r['practical_gate_pooled_noise'] for r in records)
    lines+=['',f'Completed checkpoint blocks: {len(records)}. Practical gate passed: {passed}/{len(records)}. Pooled 95th-percentile absolute replay MSE difference: {pooled_noise95:.6g}.', '',
        'The prospective practical gate requires normalized excess top-quartile utility >=0.005 over original and corrected gradients, an effect above five times replay noise, two checkpoint checks, a second fixture, and no systematic long-horizon reversal. Passing only isolated blocks does not establish the mechanism.', '',
        '## Outcome','']
    changed=sum(r['h1']['original_corrected_topk_overlap']<1 for r in records)
    redundant=sum(set(r['ranking_stats']['ccd']['topk_indices'])==set(r['ranking_stats']['corrected']['topk_indices']) for r in records)
    lines += [f'H1: the probe changed original-versus-corrected top-quartile membership in{changed}/{len(records)} blocks. The fixed CCD score selected the same top quartile as corrected magnitude in{redundant}/{len(records)} blocks. Ranking changes alone do not establish useful splits.', '']
    if smoke: outcome='Smoke validation only. These tiny horizons are not scientific evidence.'
    elif not complete: outcome='Incomplete candidate coverage; no mechanism conclusion.'
    elif passed==len(records) and not no_long_reversal: outcome='Short-horizon practical gains did not survive the prespecified long-horizon subset consistently. The mechanism gate is not established.'
    elif passed==len(records) and not strong_survive: outcome='The conditional short-horizon ranking gate passed, but retained corrections or extra pose computation explain comparable or better group gains. A distinct CCD allocation advantage is not established.'
    elif passed==len(records): outcome='The fixed score supports a mechanism advantage within this pilot, including the longer subset and strong pose controls. An end-to-end allocation campaign is still required for H3 or a full CCD-GS method claim.'
    else: outcome='The fixed CCD persistence score did not satisfy the prospective mechanism gate consistently. This pilot does not establish a useful split-ranking advantage beyond simpler controls. Preserve the negative/mixed evidence; further score design would need a separate validation campaign.'
    lines+=[outcome,'','## Long-horizon and strong controls','',
        'Long continuations (1,000 updates) use the prespecified every-fourth candidate subset, selected before utility outcomes. The subset is small; all labels are provided. Group controls split the same number of selected parents and use a shared split seed. Their joint effect is not assumed to equal the sum of single-parent effects.','',
        '| Scene | Condition | Seed | Checkpoint | CCD group utility | g0 group | gplus group | retained correction | equal-compute pose |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        g=r['group_controls']; lines.append(f'| {r["scene"]} | {r["condition"]} | {r["seed"]} | {r["checkpoint"]} | {g["ccd"]["utility"]:+.6g} | {g["original"]["utility"]:+.6g} | {g["corrected"]["utility"]:+.6g} | {g["retained"]["utility"]:+.6g} | {g["equal_compute_pose"]["utility"]:+.6g} |')
    lines+=['','Normalized CCD utility excess below averages the three seeds and three severity conditions within each scene/checkpoint. These are descriptive aggregates; individual severity/seed outcomes above remain primary. Long-subset top-k is floor(n/4), at least1, chosen from the prespecified every-fourth candidates. Positive values favor the CCD score/group.','',
            '| Scene/checkpoint | Long excess vs g0 | Long excess vs gplus | Group excess vs retained pose | Group excess vs extra pose |',
            '|---|---:|---:|---:|---:|']
    for key,row in long_means.items():
        strong=strong_means[key]
        lines.append(f'| {key} | {row["original"]:+.6g} | {row["corrected"]:+.6g} | {strong["retained"]:+.6g} | {strong["equal_compute_pose"]:+.6g} |')
    flat=[]
    for r in records:
        for index,label in enumerate(r['labels']):
            flat.append(dict(scene=r['scene'],seed=r['seed'],condition=r['condition'],checkpoint=r['checkpoint'],
                parent=label['parent'],stable_id=label['stable_id'],split_seed=label['split_seed'],
                **{name:values[index] for name,values in r['scores'].items()},
                **{name:values[index] for name,values in r['attributes'].items()},
                short_utility=label['utility'][str(r['short_horizon'])],
                long_utility=label['utility'].get(str(r['long_horizon']),''),
                baseline_development_mse=r['baseline_development_mse']))
    if flat:
        with (root/'candidate_labels.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    if complete:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.stats import rankdata
        fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
        for row,scene in enumerate(('analytic_spheres_v1','room_small')):
            for column,name in enumerate(('original','corrected','ccd')):
                ax=axes[row,column]
                for condition,color in [('clean','#2166ac'),('mild','#b8860b'),('moderate','#b2182b')]:
                    rs=[r for r in records if r['scene']==scene and r['condition']==condition]
                    xs=[]; ys=[]
                    for r in rs:
                        xs.extend((rankdata(r['scores'][name])-1)/(len(r['candidates'])-1))
                        ys.extend([label['utility'][str(r['short_horizon'])]/r['baseline_development_mse'] for label in r['labels']])
                    ax.scatter(xs,ys,s=9,alpha=.45,label=condition,color=color)
                ax.axhline(0,color='gray',lw=1); ax.set_title(f'{scene}: {name}')
                ax.set_xlabel('Within-block score percentile'); ax.set_ylabel('Utility / no-split development MSE')
        axes[0,0].legend(fontsize=8)
        fig.savefig(root/'ranking_utility.png',dpi=180); plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for ax,scene in zip(axes,('analytic_spheres_v1','room_small')):
            rows=[r for r in flat if r['scene']==scene and r['long_utility']!='']
            ax.scatter([r['short_utility']/r['baseline_development_mse'] for r in rows],
                       [r['long_utility']/r['baseline_development_mse'] for r in rows],s=12,alpha=.6)
            ax.axhline(0,color='gray',lw=1); ax.axvline(0,color='gray',lw=1)
            ax.set_xlabel('Normalized 200-update utility'); ax.set_ylabel('Normalized 1,000-update utility'); ax.set_title(scene)
        fig.savefig(root/'short_long_utility.png',dpi=180); plt.close(fig)
        lines+=['','![Rank versus split utility](ranking_utility.png)','',
                '![Short versus long continuation utility](short_long_utility.png)','']
    lines+=['','## Scope and implementation','',
        '- Two Stage 0 fixtures, three seeds, clean/mild/moderate train-only pose perturbations, two checkpoints, 24 individual candidates per block, 200-update continuations and a 1,000-update subset.',
        '- Full snapshots include Gaussian/camera tensors, all optimizer moments, schedulers, density accumulators, stable IDs/lineage, Python/NumPy/Torch CPU/CUDA RNG, current step, manifest identities and the future camera sequence.',
        '- Independent no-intervention replays quantify the actual numerical floor; diagnostics assert unchanged training state. Gradient normalization matches DefaultStrategy image-dimension/camera normalization.',
        '- Warmup uses a capped gradient-only DefaultStrategy-derived BudgetStrategy: random split and ordinary clones. Caps are5120 (synthetic) and200000 (Room); this is an internal common control, not an external method reproduction.',
        '- Prospective feasibility scope: checkpoints750/1500 for the3k synthetic schedule and1750/3500 for the7k Room schedule. The third planned checkpoint and end-to-end allocation study are deferred until the two-checkpoint mechanism gate succeeds. These choices are written before outcomes.',
        '- Candidate selection is stratified over original-gradient quartiles among visible ordinary-split parents; blocks with too few use all scales and explicitly record that change. Candidate attributes, radius changes and per-label random split seeds are logged.',
        '- Low-gradient control candidates are deliberately included; satisfying the scale criterion does not mean the ordinary gradient threshold would select that parent. Individual split branches add one Gaussian, while no-split branches keepN. Group controls each add the samek. The common count cap applies to warmup; these marginal branch interventions are permitted above that cap.',
        '- Final-test RGB is not used for optimization, score selection or quality evaluation. The existing parser decodes its first image to inspect dimensions only. Room is previously exposed development data. Analytic camera poses provide reference truth; COLMAP poses on Room are reference estimates, not known real truth.',
        '- No outdoor benchmark, novel-method claim, full allocation policy, or training-speed claim follows from this mechanism pilot. Probe fitting, diagnostics, branches and controls have separately recorded wall times.',
        '- This script measures a fixed score and a secondary pure persistence ratio; fitting a new score to these utility labels would require held-back validation.']
    lines+=['- Each candidate has one prespecified random split draw; the utility label includes placement randomness. The three training seeds and replay controls do not fully characterize within-parent split-draw variance. A failed pilot gate limits evidence for this fixed score and protocol; it does not rule out all pose-conditioned signals.']
    lines+=['- Undefined rank correlations (for example, constant utility labels) are recorded as null rather than zero.']
    write(root/'summary.json',dict(complete=complete,checkpoint_blocks=len(records),practical_blocks_passed=passed,pooled_replay_noise95=pooled_noise95,
          no_systematic_long_reversal=no_long_reversal,strong_group_controls_survive=strong_survive,
          long_excess_by_scene_checkpoint=long_means,strong_group_excess_by_scene_checkpoint=strong_means,
          outcome=outcome,records=[str(Path(r['scene'])/f'seed{r["seed"]}_{r["condition"]}'/f'checkpoint{r["checkpoint"]}'/'result.json') for r in records]))
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    cli=argparse.ArgumentParser(description=__doc__); cli.add_argument('--root',type=Path,required=True); cli.add_argument('--smoke',action='store_true')
    args=cli.parse_args(); root=args.root; root.mkdir(parents=True,exist_ok=True)
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dependencies=['examples/simple_trainer.py','examples/utils.py','examples/datasets/colmap.py',
                  'gsplat/strategy/default.py','gsplat/strategy/budget.py','gsplat/strategy/ops.py']
    protocol=dict(source_sha256=source_hash,dependency_sha256={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in dependencies},
        scenes=['analytic_spheres_v1','room_small'],seeds=[0,1,2],
        conditions=['clean','mild','moderate'],checkpoints={'analytic_spheres_v1':[750,1500],'room_small':[1750,3500]},
        candidate_count=24,short=200,long=1000,long_subset='every fourth candidate before outcomes',
        primary_score='g0*clip(gplus/(g0+1e-12),0,1)',secondary_score='gplus/(g0+1e-12)',
        long_analysis='Within prespecified long subset, rank top floor(n/4), min1; require no negative mean normalized excess over g0/gplus within either scene/checkpoint, across seeds/severities.',
        strong_control_analysis='Report joint top-k split utility and CCD-minus-retained/extra-pose normalized excess within each scene/checkpoint. A failed practical short gate cannot be rescued by these secondary aggregates.',
        probe_updates=5,rotation_bound_degrees=1,translation_bound_scene_scale=.005,
        prospective_scope_revision='Two checkpoints and one cap per fixture for the H1/H2 gate; no allocation campaign before a supported mechanism.',smoke=args.smoke)
    if (root/'protocol.json').exists(): assert json.loads((root/'protocol.json').read_text())==protocol, 'CCD source/protocol changed after capture'
    else: write(root/'protocol.json',protocol)
    records=[]
    initial_hashes={}
    for scene in protocol['scenes'][:1] if args.smoke else protocol['scenes']:
        for seed in [0] if args.smoke else protocol['seeds']:
            for condition in ['mild'] if args.smoke else protocol['conditions']:
                folder=root/scene/f'seed{seed}_{condition}'
                study=Study(scene,seed,condition,args.smoke)
                key=(scene,seed)
                if key in initial_hashes: assert initial_hashes[key]==study.initial_sha256, 'Pose condition changed Gaussian initialization'
                initial_hashes[key]=study.initial_sha256
                write(folder/'data_and_disturbance.json',dict(manifest=study.manifest,
                    item_to_view=study.mapping,initial_parameter_sha256=study.initial_sha256,
                    sample_sequence_sha256=hashlib.sha256(study.sequence.tobytes()).hexdigest(),
                    exact_train_perturbations=study.disturbance.cpu().tolist(),
                    reference_pose_error=pose_error(study.cameras,study.clean,study.scale)))
                for checkpoint in study.checkpoints:
                    target=folder/f'checkpoint{checkpoint}'; saved=target/'full_state.pt'
                    if saved.exists(): snapshot=torch.load(saved,map_location='cpu',weights_only=False)
                    else:
                        while study.step<checkpoint: study.train_step()
                        snapshot=study.snapshot(); target.mkdir(parents=True,exist_ok=True); torch.save(snapshot,saved)
                    result=checkpoint_study(study,snapshot,target,args.smoke); records.append(result)
                    # Branch experiments end at a later step; resume the common warmup.
                    study.restore(snapshot)
                    write(root/'status.json',dict(status='running',completed_blocks=len(records),scene=scene,seed=seed,condition=condition,checkpoint=checkpoint))
                    summarize(root,records,args.smoke)
                del study; torch.cuda.empty_cache()
    summarize(root,records,args.smoke)
    if not args.smoke:
        assert json.loads((root/'summary.json').read_text())['complete'], 'Mechanism matrix has incomplete candidate coverage'
    write(root/'status.json',dict(status='complete',completed_blocks=len(records),smoke=args.smoke))
    print('CCD_MECHANISM_COMPLETE',len(records),flush=True)


if __name__=='__main__': main()
