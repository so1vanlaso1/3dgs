"""Reload selected D8 models; render every preserved full-route camera.

Baseline measurements use its own historical images/cameras. They are diagnostic,
not an SDK-vs-legacy comparison or a new reconstructed whole-room model.
"""
import argparse
import importlib.util
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'examples')]
s=importlib.util.spec_from_file_location('evaluation',ROOT/'scripts/db3_evaluate_pilot.py')
e=importlib.util.module_from_spec(s);s.loader.exec_module(e)
m=e.m
import numpy as np
import torch
from PIL import Image, ImageDraw
from datasets.colmap import Parser
from capture_path import CapturePath


@torch.no_grad()
def main(out):
    assert (out/'protocol.json').exists()
    dest=out/'delivery';dest.mkdir(exist_ok=False);(dest/'full_route_views').mkdir()
    started=time.time()
    frozen=m.read(m.R1/'baseline_manifest.json')
    assert m.read(out/'protocol.json')['source_sha256']==frozen['source_sha256']
    record=next(f for f in frozen['files'] if f['source'].endswith('final_post_update.pt'))
    ckpt=ROOT/record['source'];assert m.sha(ckpt)==record['sha256']
    dataset=ROOT/'data/realsense_20261030_134523_clean/rgbd/dataset'
    parser=Parser(str(dataset),factor=1,normalize=True,load_exposure=False)
    baseline=m.read(dataset.parent.parent/'capture_manifest.json');byname={f['name']:f for f in baseline['frames']}
    origin=min(f['timestamp_ns'] for f in byname.values())
    times=np.array([(byname[n]['timestamp_ns']-origin)/1e9 for n in parser.image_names])
    assert len(times)==477 and set(parser.image_names)==set(byname)
    path=CapturePath(parser)
    assert path.timing_source.startswith('Recorded timestamps')
    replay_samples=[]
    for at in (0,path.duration/2,path.duration):
        position,rotation,fov,name=path.sample(at)
        assert np.isfinite(position).all() and np.isfinite(rotation).all() and np.isfinite(fov)
        replay_samples.append(dict(seconds=at,position=position.tolist(),rotation_matrix=rotation.tolist(),image_name=name,vertical_fov_radians=float(fov)))
    requests=sorted(set([0,10,43,61,68,110,157,*range(0,159,10)]))
    anchors={int(np.argmin(abs(times-at))) for at in requests}
    splits=m.read(m.INPUT/'reconstruction_split.json');eval_names=set(splits['historical_eval'])
    splats=e.load_splats(ckpt);assert torch.isfinite(splats['scales'].exp()).all()
    baseline_count=len(splats['means'])
    metrics=[];panels=[]
    for i,name in enumerate(parser.image_names):
        cid=parser.camera_ids[i];w,h=parser.imsize_dict[cid];K=parser.Ks_dict[cid]
        rgb,depth,alpha=e.render(splats,parser.camtoworlds[i],K,w,h)
        assert np.isfinite(rgb).all() and np.isfinite(depth).all() and np.isfinite(alpha).all()
        target=np.asarray(Image.open(dataset.parent.parent/'images'/name).convert('RGB'),dtype='f4')/255
        assert target.shape==rgb.shape
        mse=float(np.mean((rgb-target)**2))
        row=dict(name=name,source_id=byname[name]['message_id'],seconds=float(times[i]),
            historical_exposed_eval=name in eval_names,rgb_psnr_db=float(-10*np.log10(max(mse,1e-12))),
            mean_alpha=float(alpha.mean()),fraction_pixels_alpha_gt_08=float((alpha>.8).mean()),
            finite_render=True)
        if i in anchors:
            im=Image.fromarray((rgb*255).astype('uint8'));capture=Image.fromarray((target*255).astype('uint8'))
            im.save(dest/'full_route_views'/f'{i:04d}_render.jpg',quality=94)
            capture.save(dest/'full_route_views'/f'{i:04d}_capture.jpg',quality=94)
            panel=Image.new('RGB',(1000,305),'white');d=ImageDraw.Draw(panel)
            d.text((5,5),f'{times[i]:.3f}s | {name} | observed / preserved context | PSNR {row["rgb_psnr_db"]:.2f}',fill='black')
            capture.thumbnail((500,278));im.thumbnail((500,278));panel.paste(capture,(0,27));panel.paste(im,(500,27));panels.append(panel)
            row['saved_render']=f'full_route_views/{i:04d}_render.jpg'
        metrics.append(row)
        if (i+1)%50==0:print('FULL ROUTE RELOAD/RENDER',i+1,'/477',flush=True)
    del splats;torch.cuda.empty_cache()
    sheet=Image.new('RGB',(1000,sum(p.height for p in panels)),'white')
    for i,panel in enumerate(panels):sheet.paste(panel,(0,i*305))
    sheet.save(dest/'full_route_comparison.jpg',quality=92)
    m.write(dest/'baseline_viewer.json',dict(checkpoint=str(ckpt.relative_to(ROOT)),dataset=str(dataset.relative_to(ROOT)),camera_index=0,
        scope='Preserved whole-route context; uncertain physical geometry; historical projection only; do not pair with R1 SDK images'))
    m.write(dest/'baseline_camera_provenance.json',dict(checkpoint_sha256=record['sha256'],
        normalization_transform=parser.transform.tolist(),image_names=parser.image_names,
        camera_to_world_normalized=parser.camtoworlds.tolist(),
        K={str(k):v.tolist() for k,v in parser.Ks_dict.items()},image_sizes={str(k):v for k,v in parser.imsize_dict.items()},
        source_id_by_image={n:byname[n]['message_id'] for n in parser.image_names},
        metric_before_normalization=True,replay_duration_seconds=path.duration,replay_samples=replay_samples,
        projection='historical baseline convention, independently retained; not interchangeable with SDK local inputs'))
    summary={}
    for label,subset in [('all',[r for r in metrics]),('historical_exposed_eval',[r for r in metrics if r['historical_exposed_eval']])]:
        summary[label]=dict(views=len(subset),mean_psnr_db=float(np.mean([r['rgb_psnr_db'] for r in subset])),
            min_psnr_db=min(r['rgb_psnr_db'] for r in subset),mean_alpha=float(np.mean([r['mean_alpha'] for r in subset])))
    m.write(dest/'full_route_metrics.json',dict(views=metrics,summary=summary,
        scope='Fresh final-checkpoint reload at every preserved camera. Historical target/projection, mixed training and exposed evaluation.',
        alpha_is_not_surface_coverage=True,physical_accuracy_established=False,new_model_improvement_claim=False))
    local=m.read(m.RUN/'local_delivery.json')['submaps']['matte']
    assert local['declared_local_followup_gate_pass']
    assert m.sha(ROOT/local['checkpoint'])==local['checkpoint_sha256']
    local_splats=e.load_splats(ROOT/local['checkpoint']);local_count=len(local_splats['means'])
    assert torch.isfinite(local_splats['scales'].exp()).all()
    del local_splats;torch.cuda.empty_cache()
    for name in ('viewer.json','camera_provenance.json','selected_comparison.jpg'):
        shutil.copyfile(m.RUN/'matte/pilot'/name,dest/('matte_'+name))
    shutil.copyfile(ROOT/'results/db3_r0_r3_cleanup_20261111/matte/selected_policy_confirmation_result.json',dest/'matte_acceptance.json')
    m.write(dest/'manifest.json',dict(source_sha256=frozen['source_sha256'],
        whole_route=dict(checkpoint=str(ckpt.relative_to(ROOT)),checkpoint_sha256=record['sha256'],gaussians=baseline_count,
            matching_dataset=str(dataset.relative_to(ROOT)),camera_count=len(metrics),all_renders_finite=True,selected_role='uncertain full-recording context'),
        matte=dict(**local,gaussians=local_count,selected_role='accepted narrow local appearance/consistency follow-up'),
        global_alignment=None,worlds_must_remain_separate=True,exhibit_promoted=False,
        R4_delivery='D8 partial result; no new accepted full-scene reconstruction',R5_interactive_verification='pending',
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)),
        elapsed_seconds=time.time()-started))
    seal(out)
    print('DELIVERY VERIFIED',summary,flush=True)


def seal(out):
    dest=out/'delivery';manifest=m.read(dest/'manifest.json')
    metrics=m.read(dest/'full_route_metrics.json')
    assert manifest['whole_route']['camera_count']==len(metrics['views'])==477
    assert all(r['finite_render'] for r in metrics['views'])
    assert len({r['source_id'] for r in metrics['views']})==477
    assert m.sha(ROOT/manifest['whole_route']['checkpoint'])==manifest['whole_route']['checkpoint_sha256']
    assert m.sha(ROOT/manifest['matte']['checkpoint'])==manifest['matte']['checkpoint_sha256']
    for name in ('baseline_camera_provenance.json','baseline_viewer.json','matte_camera_provenance.json',
                 'matte_viewer.json','matte_acceptance.json','matte_selected_comparison.jpg','full_route_comparison.jpg'):
        assert (dest/name).is_file()
    dst=out/'source_snapshot'/Path(__file__).name;dst.parent.mkdir(exist_ok=True)
    if dst.exists():raise FileExistsError(dst)
    shutil.copyfile(__file__,dst)
    m.write(dest/'artifact_hashes.json',{str(p.relative_to(dest)):m.sha(p) for p in dest.rglob('*') if p.is_file()})
    print('DELIVERY SEALED',len(metrics['views']),'verified views',flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(__doc__);a.add_argument('--output-dir',type=Path,required=True)
    a.add_argument('--seal-existing',action='store_true')
    args=a.parse_args()
    (seal if args.seal_existing else main)(args.output_dir)
