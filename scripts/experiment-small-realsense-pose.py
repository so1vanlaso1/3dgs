"""Small stairs pose experiment; separate outputs and matched training controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import pycolmap as p
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO/'data/realsense_20261030_111354_clean'
ROOT = REPO/'data/realsense_small_pose_20261030'
OUT = REPO/'results/realsense_small_pose_20261030'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def run(stage, command, env=None):
    marker = OUT/(stage+'_success.json')
    command = list(map(str, command))
    if marker.exists():
        assert json.loads(marker.read_text())['command'] == command
        return
    write(OUT/'status.json', dict(stage=stage, status='running', pid=os.getpid()))
    started = time.time()
    with (OUT/(stage+'.log')).open('a', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{stage} failed ({result.returncode}); see {stage}.log')
    write(marker, dict(command=command, elapsed_seconds=time.time()-started))


def prepare():
    if (ROOT/'selection.json').exists(): return
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE/'capture_manifest.json').read_text())
    frames = manifest['frames'][78:106]
    names = {f['name'] for f in frames}
    for dirname in ('images', 'aligned_depth_meters'):
        (ROOT/dirname).mkdir(exist_ok=True)
        for name in names:
            filename = name if dirname == 'images' else name+'.npz'
            target = ROOT/dirname/filename
            if not target.exists(): os.link(SOURCE/dirname/filename, target)
    manifest['frames'] = frames
    manifest['parameters']['all_frames'] = False
    write(ROOT/'capture_manifest.json', manifest)
    (ROOT/'calibration_clean.json').write_bytes((SOURCE/'calibration_clean.json').read_bytes())
    # Copy cached descriptors, never mutate the source database.
    with sqlite3.connect((SOURCE/'colmap.db').resolve().as_uri()+'?mode=ro', uri=True) as src, sqlite3.connect(ROOT/'colmap.db') as dst:
        tables = src.execute("select name,sql from sqlite_master where type='table' and name!='sqlite_sequence'").fetchall()
        ids = {iid for iid, name in src.execute('select image_id,name from images') if name in names}
        for table, schema in tables:
            dst.execute(schema)
            if table in ('images', 'keypoints', 'descriptors'):
                rows = src.execute(f"select * from {table} where image_id in ({','.join('?' for _ in ids)})", tuple(ids)).fetchall()
            elif table in ('matches', 'two_view_geometries'):
                rows = [r for r in src.execute(f'select * from {table}') if r[0]//2147483647 in ids and r[0]%2147483647 in ids]
            else:
                rows = src.execute(f'select * from {table}').fetchall()
            if rows: dst.executemany(f"insert into {table} values ({','.join('?' for _ in rows[0])})", rows)
    selected = dict(frames=frames, source=str(SOURCE), reason='Contiguous stairs/nearby wall interval, inspected contact sheet; no manual moving-object masks',
                    source_db3_stat=dict(bytes=Path(manifest['source']).stat().st_size, mtime_ns=Path(manifest['source']).stat().st_mtime_ns))
    write(ROOT/'selection.json', selected)
    print('PREPARED', len(frames), 'frames', flush=True)


def reconstruct():
    if (ROOT/'pose_report.json').exists() and (ROOT/'local_metric_sfm/images.bin').exists(): return
    matching = p.FeatureMatchingOptions(); matching.num_threads = 6
    p.match_exhaustive(ROOT/'colmap.db', matching_options=matching, device=p.Device.cpu)
    options = p.IncrementalPipelineOptions()
    options.num_threads = 6; options.random_seed = 42; options.min_model_size = 10
    options.ba_refine_focal_length = False; options.ba_refine_principal_point = False; options.ba_refine_extra_params = False
    options.mapper.abs_pose_refine_focal_length = False; options.mapper.abs_pose_refine_extra_params = False
    options.mapper.init_min_tri_angle = 8
    (ROOT/'reconstruction').mkdir(exist_ok=True)
    models = p.incremental_mapping(ROOT/'colmap.db', ROOT/'images', ROOT/'reconstruction', options=options)
    if not models: raise RuntimeError('No local SfM model')
    model = max(models.values(), key=lambda m:m.num_reg_images())
    before = model.compute_mean_reprojection_error()
    ba = p.BundleAdjustmentOptions()
    ba.refine_focal_length = False; ba.refine_principal_point = False; ba.refine_extra_params = False
    ba.ceres.loss_function_type = p.LossFunctionType.SOFT_L1
    ba.ceres.solver_options.num_threads = 6; ba.ceres.solver_options.max_num_iterations = 300
    ba.ceres.solver_options.function_tolerance = 1e-6
    p.bundle_adjustment(model, ba)
    after = model.compute_mean_reprojection_error()
    (ROOT/'dataset/sparse').mkdir(parents=True, exist_ok=True)
    model.write(ROOT/'dataset/sparse')
    selected = json.loads((ROOT/'selection.json').read_text())['frames']
    names = sorted(im.name for im in model.images.values())
    if len(names) != len(selected):
        raise RuntimeError(f'Local SfM registered only {len(names)}/{len(selected)}; inspect before training')
    # Same heldout image names, with direct neighbors excluded from training.
    val = names[4::8]
    times = {f['name']: f['timestamp_ns']/1e9 for f in selected}
    train = [n for n in names if n not in val and all(abs(times[n]-times[v]) > .3 for v in val)]
    split = dict(train=train, val=val, buffer=[n for n in names if n not in train and n not in val],
                 scope='Image interpolation; jointly reconstructed poses, no independent geometry ground truth')
    if len(train)<15 or len(val)<2: raise RuntimeError('Insufficient small-scene split')
    # Scale fit uses precisely this experiment's training set, never heldout depth.
    ratios=[]; per_frame=[]
    for im in model.images.values():
        if im.name not in train: continue
        with np.load(ROOT/'aligned_depth_meters'/(im.name+'.npz')) as dep:
            pixels,z=dep['points'],dep['depths']
        observed=[]; world=[]
        for point in im.points2D:
            if point.has_point3D():
                observed.append(point.xy);world.append(model.points3D[point.point3D_id].xyz)
        if not world: continue
        dist,nn=cKDTree(pixels).query(np.asarray(observed))
        W=im.cam_from_world().matrix(); camera=np.asarray(world)@W[:,:3].T+W[:,3]
        good=(dist<1.5)&(camera[:,2]>0)
        r=camera[good,2]/z[nn[good]]
        ratios.extend(r.tolist())
        if len(r)>=20: per_frame.append(float(np.median(r)))
    scale=float(np.median(ratios));err=np.abs(np.asarray(ratios)/scale-1)
    scale_report=dict(colmap_units_per_meter=scale,training_scale_pairs=len(ratios),median_relative_error=float(np.median(err)),
        scale_inlier_fraction_10pct=float(np.mean(err<.1)),frame_scale_relative_p90=float(np.percentile(np.abs(np.asarray(per_frame)/scale-1),90)),fit_scope='Explicit small-experiment training views only')
    scale_report['passed']=len(ratios)>=500 and scale_report['scale_inlier_fraction_10pct']>.65 and scale_report['frame_scale_relative_p90']<.2
    write(ROOT/'sensor_depth_report.json',scale_report)
    if not scale_report['passed']: raise RuntimeError('Local depth/scale gate failed')
    model.transform(p.Sim3d(1/scale, p.Rotation3d(), np.zeros(3)))
    baseline = p.Reconstruction(SOURCE/'rgbd/dataset/sparse')
    base = {im.name: im for im in baseline.images.values()}
    fresh = {im.name: im for im in model.images.values()}
    pose_sets = {}
    for tag, images in [('baseline', base), ('refined', fresh)]:
        poses = {}
        for name in names:
            W = np.eye(4); W[:3] = images[name].cam_from_world().matrix()
            poses[name] = np.linalg.inv(W)
        pose_sets[tag] = poses
        export_dataset(ROOT/tag, names, poses, split)
    # Relative movement diagnostic, no world-coordinate alignment assumptions.
    deltas = []
    for a,b in zip(names[:-1],names[1:]):
        rel = [np.linalg.inv(pose_sets[t][b])@pose_sets[t][a] for t in ('baseline','refined')]
        delta = np.linalg.inv(rel[0])@rel[1]
        deltas.append(dict(a=a,b=b,translation_m=float(np.linalg.norm(delta[:3,3])),rotation_deg=float(np.rad2deg(Rotation.from_matrix(delta[:3,:3]).magnitude()))))
    (ROOT/'local_metric_sfm').mkdir(exist_ok=True)
    model.write(ROOT/'local_metric_sfm')
    write(ROOT/'pose_report.json', dict(registered=len(names), selected=len(selected), points=model.num_points3D(),
          local_sfm_reprojection_before_extra_ba=before, local_sfm_reprojection_after_extra_ba=after,
          split=split, scale=scale, adjacent_pose_differences=deltas,
          method='Fresh local exhaustive matching + incremental SfM + robust BA; fixed calibrated intrinsics; no inherited pose anchors',
          limitations='No manual motion masks, no independent metric ground truth; local SfM already includes BA'))
    print('LOCAL_POSES_READY', len(names), 'reprojection', after, flush=True)


def export_dataset(root, names, poses, split):
    cal = json.loads((ROOT/'calibration_clean.json').read_text()); K = np.array(cal['K'])
    for d in ('images','sparse','sensor_depth'): (root/d).mkdir(parents=True, exist_ok=True)
    write(root/'experiment_split.json', split)
    (root/'sparse/cameras.txt').write_text(f"1 PINHOLE {cal['width']} {cal['height']} {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}\n")
    lines=[]; xyz=[]; rgb=[]; rng=np.random.default_rng(42)
    for j,name in enumerate(names):
        if not (root/'images'/name).exists(): os.link(ROOT/'images'/name, root/'images'/name)
        W=np.linalg.inv(poses[name]); q=Rotation.from_matrix(W[:3,:3]).as_quat(); t=W[:3,3]
        lines.append(f'{j+1} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} 1 {name}\n\n')
        with np.load(ROOT/'aligned_depth_meters'/(name+'.npz')) as dep:
            pixels,z=dep['points'].copy(),dep['depths'].copy()
        # Identical sampled source pixels in both branches; only poses differ.
        keep=rng.choice(len(z),min(4096,len(z)),replace=False)
        np.savez_compressed(root/'sensor_depth'/(name+'.npz'),points=pixels[keep],depths=z[keep])
        if name in split['train']:
            pixels,z=pixels[keep],z[keep]
            cam=np.column_stack(((pixels[:,0]-K[0,2])*z/K[0,0],(pixels[:,1]-K[1,2])*z/K[1,1],z))
            xyz.append(cam@poses[name][:3,:3].T+poses[name][:3,3])
            image=np.asarray(Image.open(ROOT/'images'/name)); px=np.rint(pixels).astype(int)
            rgb.append(image[px[:,1],px[:,0]])
    (root/'sparse/images.txt').write_text(''.join(lines))
    xyz=np.concatenate(xyz);rgb=np.concatenate(rgb)
    # Same source sample count, no pose-dependent voxel reduction.
    with (root/'sparse/points3D.txt').open('w') as stream:
        for i,(pt,color) in enumerate(zip(xyz,rgb)):
            stream.write(f'{i+1} {pt[0]} {pt[1]} {pt[2]} {color[0]} {color[1]} {color[2]} 0\n')
    m=p.Reconstruction(); m.read_text(root/'sparse'); m.write(root/'sparse')


def train(steps):
    for tag in ('baseline','refined'):
        data=ROOT/tag
        run('train_'+tag, ['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',REPO/'scripts/test-small-dataset.ps1',
            '-Steps',steps,'-DataDir',data,'-ResultDir',OUT/tag,'-ShDegree',3,'-ShDegreeInterval',1000,
            '-RefineStart',500,'-RefineStop',4500,'-ResetEvery',3000,'-Strategy','default',
            '-RecordExperiment','-DataWorkers',0,'-SensorDepthDir',data/'sensor_depth','-DepthWeight',.01])


def diagnose():
    """Compare pose-derived epipolar errors and separate sensor-depth warps."""
    names=json.loads((ROOT/'pose_report.json').read_text())['split']
    cal=json.loads((ROOT/'calibration_clean.json').read_text()); K=np.asarray(cal['K']); invK=np.linalg.inv(K)
    reports={}
    with sqlite3.connect((ROOT/'colmap.db').as_uri()+'?mode=ro',uri=True) as db:
        image_names=dict(db.execute('select image_id,name from images'))
        keys={i:np.frombuffer(blob,'f4').reshape(n,c)[:,:2].copy() for i,n,c,blob in db.execute('select image_id,rows,cols,data from keypoints')}
        pairs=db.execute('select pair_id,rows,data from two_view_geometries where rows>=30').fetchall()
    for tag in ('baseline','refined'):
        model=p.Reconstruction(ROOT/tag/'sparse'); images={im.name:im for im in model.images.values()}
        epi=[];depth_errors=[];pair_rows=[]
        for pid,n,blob in pairs:
            a,b=pid//2147483647,pid%2147483647; na,nb=image_names[a],image_names[b]
            ia=int(na[6:12]);ib=int(nb[6:12])
            if abs(ia-ib)>4:continue
            Wa=np.eye(4);Wb=np.eye(4);Wa[:3]=images[na].cam_from_world().matrix();Wb[:3]=images[nb].cam_from_world().matrix()
            T=Wb@np.linalg.inv(Wa); t=T[:3,3]
            cross=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
            F=invK.T@cross@T[:3,:3]@invK
            match=np.frombuffer(blob,'u4').reshape(n,2)
            x=np.column_stack((keys[a][match[:,0]],np.ones(n)));y=np.column_stack((keys[b][match[:,1]],np.ones(n)))
            Fx=x@F.T;Fty=y@F; e=np.abs(np.sum(y*Fx,axis=1))/np.sqrt(np.sum(Fx[:,:2]**2,axis=1)+np.sum(Fty[:,:2]**2,axis=1)+1e-20)
            epi.extend(e.tolist())
            with np.load(ROOT/'aligned_depth_meters'/(na+'.npz')) as dep: px,z=dep['points'][::8],dep['depths'][::8]
            cam=np.column_stack((px,np.ones(len(px))))@invK.T*z[:,None];cam=cam@T[:3,:3].T+t
            projected=cam@K.T;uv=projected[:,:2]/projected[:,2:3]
            with np.load(ROOT/'aligned_depth_meters'/(nb+'.npz')) as dep:
                dist,nn=cKDTree(dep['points']).query(uv);z_other=dep['depths'][nn]
            valid=(dist<1.5)&(cam[:,2]>0)
            err=np.abs(cam[valid,2]-z_other[valid])
            depth_errors.extend(err.tolist())
            pair_rows.append(dict(a=na,b=nb,feature_matches=n,epipolar_median_pixels=float(np.median(e)),depth_overlap_samples=int(valid.sum()),depth_abs_median_m=float(np.median(err)) if len(err) else None))
        reports[tag]=dict(epipolar_count=len(epi),epipolar_pixels_p50_p90=np.percentile(epi,[50,90]).tolist(),
            depth_overlap_count=len(depth_errors),depth_abs_error_m_p50_p90=np.percentile(depth_errors,[50,90]).tolist(),pairs=pair_rows)
    reports['scope']='Epipolar residuals use matches also available to SfM, not independent validation. Depth warp is a complementary check, includes occlusions and unmasked foliage; overlap counts differ with poses.'
    write(OUT/'pose_diagnostics.json',reports)


def compare():
    import torch
    from PIL import ImageDraw
    sys.path.insert(0,str(REPO/'examples'))
    from datasets.colmap import Parser
    from gsplat.rendering import rasterization
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    import math
    split=json.loads((ROOT/'refined/experiment_split.json').read_text())
    names=split['val']; canvases={n:[] for n in names}; report={}
    metrics=dict(psnr=PeakSignalNoiseRatio(data_range=1.).cuda(),ssim=StructuralSimilarityIndexMeasure(data_range=1.).cuda(),
                 lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda())
    traces={}
    for tag in ('baseline','refined'):
        parser=Parser(str(ROOT/tag),factor=1,normalize=True,load_exposure=False)
        ckpt=torch.load(OUT/tag/'ckpts/final_post_update.pt',map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for v in ckpt['splats'].values())
        splats={k:v.cuda() for k,v in ckpt['splats'].items()};colors=torch.cat((splats['sh0'],splats['shN']),dim=1)
        per_view=[]
        with torch.no_grad():
            for name in names:
                i=parser.image_names.index(name);cid=parser.camera_ids[i];w,h=parser.imsize_dict[cid]
                view=torch.from_numpy(np.linalg.inv(parser.camtoworlds[i])[None]).float().cuda()
                intr=torch.from_numpy(parser.Ks_dict[cid][None]).float().cuda()
                rgb,_,_=rasterization(splats['means'],splats['quats'],splats['scales'].exp(),splats['opacities'].sigmoid(),colors,view,intr,w,h,
                    sh_degree=int(math.sqrt(colors.shape[1])-1),packed=True)
                assert torch.isfinite(rgb).all()
                canvases[name].append(Image.fromarray((rgb[0].clamp(0,1).cpu().numpy()*255).astype('uint8')))
                target=torch.from_numpy(np.asarray(Image.open(ROOT/'images'/name).convert('RGB')).copy()).float().cuda()[None].permute(0,3,1,2)/255
                prediction=rgb.clamp(0,1).permute(0,3,1,2)
                row=dict(name=name,**{k:float(metric(prediction,target)) for k,metric in metrics.items()})
                for metric in metrics.values():metric.reset()
                if name=='frame_000090.png':
                    # Defined from the captured reference before fresh-run results.
                    row['stairs_roi_xyxy']=[620,140,790,320]
                    row['stairs_roi']={k:float(metric(prediction[:,:,140:320,620:790],target[:,:,140:320,620:790])) for k,metric in metrics.items()}
                    for metric in metrics.values():metric.reset()
                per_view.append(row)
        final=sorted((OUT/tag/'stats').glob('val_step*.json'))[-1]
        report[tag]=dict(**{k:float(np.mean([v[k] for v in per_view])) for k in metrics},per_view=per_view,
                         num_GS=len(splats['means']),trainer_stats=json.loads(final.read_text()))
        traces[tag]=json.loads((OUT/tag/'experiment.json').read_text())
        del splats,colors,ckpt
        torch.cuda.empty_cache()
    for name,rendered in canvases.items():
        reference=Image.open(ROOT/'images'/name).convert('RGB');w,h=reference.size
        canvas=Image.new('RGB',(w*3,h+28));d=ImageDraw.Draw(canvas)
        for j,(im,label) in enumerate(zip([reference]+rendered,['Captured RGB','Inherited courtyard poses','Fresh local SfM + BA'])):
            canvas.paste(im,(j*w,28));d.text((j*w+8,8),label,fill='white')
        canvas.save(OUT/('matched_'+name+'.jpg'),quality=95)
        if name=='frame_000090.png':
            crop=Image.new('RGB',(510*3,540+28));cd=ImageDraw.Draw(crop)
            for j,(im,label) in enumerate(zip([reference]+rendered,['Captured stairs','Inherited courtyard poses','Fresh local SfM + BA'])):
                crop.paste(im.crop((620,140,790,320)).resize((510,540),Image.Resampling.NEAREST),(j*510,28));cd.text((j*510+8,8),label,fill='white')
            crop.save(OUT/'stairs_detail.jpg',quality=95)
    report.update(heldout_names=names,matched_targets=True,scope='Small scene only; 3 heldout images, joint pose estimation. Same RGB/depth samples, optimizer settings and source initialization samples; poses and resulting world-space initialization differ.')
    assert traces['baseline']['camera_sequence_sha256']==traces['refined']['camera_sequence_sha256']
    assert traces['baseline']['completed_updates']==traces['refined']['completed_updates']
    report['completed_updates_per_branch']=traces['baseline']['completed_updates']
    report['identical_camera_sequence_verified']=True
    for name in sorted(p.name for p in (ROOT/'images').glob('*.png')):
        assert hashlib.sha256((ROOT/'baseline/images'/name).read_bytes()).digest()==hashlib.sha256((ROOT/'refined/images'/name).read_bytes()).digest()
        with np.load(ROOT/'baseline/sensor_depth'/(name+'.npz')) as a,np.load(ROOT/'refined/sensor_depth'/(name+'.npz')) as b:
            assert np.array_equal(a['points'],b['points']) and np.array_equal(a['depths'],b['depths'])
    report['identical_rgb_and_depth_samples_verified']=True
    manifest=json.loads((ROOT/'capture_manifest.json').read_text());stat=Path(manifest['source']).stat()
    assert (stat.st_size,stat.st_mtime_ns)==(manifest['source_bytes'],manifest['source_mtime_ns'])
    report['source_db3_stat_unchanged']=True
    report['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a,b=report['baseline'],report['refined']
    report['delta_refined_minus_baseline']={k:b[k]-a[k] for k in ('psnr','ssim','lpips')}
    report['selected']='refined' if b['psnr']>a['psnr'] and b['ssim']>a['ssim'] and b['lpips']<a['lpips'] else 'baseline'
    write(OUT/'comparison.json',report)
    for tag in ('baseline','refined'):
        write(OUT/tag/'viewer.json',dict(checkpoint=str(OUT/tag/'ckpts/final_post_update.pt'),dataset=str(ROOT/tag),camera_index=12))
    selected=report['selected']
    write(OUT/'viewer.json',dict(checkpoint=str(OUT/selected/'ckpts/final_post_update.pt'),dataset=str(ROOT/selected),camera_index=12,selected=selected))
    print('SMALL_SCENE_COMPARISON',report,flush=True)


def preview():
    env=os.environ.copy()
    text=subprocess.check_output(f'call "{REPO/".tools/msvc/setup_x64.bat"}" >nul && set',shell=True,text=True)
    for line in text.splitlines():
        if '=' in line and not line.startswith('='):
            k,v=line.split('=',1);env[k]=v
    env.update(CUDA_HOME=str(REPO/'.tools/cuda-12.8'),CUDA_PATH=str(REPO/'.tools/cuda-12.8'),
               TORCH_EXTENSIONS_DIR=str(REPO/'.cache/torch_extensions'),TORCH_HOME=str(REPO/'.cache/torch'),
               TORCH_CUDA_ARCH_LIST='7.5',NUM_CHANNELS='3,4',PYTHONPATH=str(REPO))
    env['PATH']=str(REPO/'.venv/Scripts')+os.pathsep+str(REPO/'.tools/cuda-12.8/bin')+os.pathsep+env['PATH']
    for tag in ('baseline','refined'):
        run('preview_'+tag,[sys.executable,'-u',REPO/'scripts/inspect-realsense-scene.py',
            '--ckpt',OUT/tag/'ckpts/final_post_update.pt','--data-dir',ROOT/tag,'--output',OUT/tag/'preview'],env)
    run('compare',[sys.executable,'-u',__file__,'--stage','compare'],env)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','reconstruct','diagnose','train','preview','compare','all'],default='all');ap.add_argument('--steps',type=int,default=6000)
    a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    os.environ.update(OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6')
    try:
        for stage,func in [('prepare',prepare),('reconstruct',reconstruct),('diagnose',diagnose),('train',lambda:train(a.steps)),('preview',preview),('compare',compare)]:
            if stage=='compare' and a.stage=='all':continue
            if a.stage in (stage,'all'):
                write(OUT/'status.json',dict(stage=stage,status='running',pid=os.getpid()));func()
        write(OUT/'status.json',dict(stage=a.stage,status='complete'))
    except Exception as e:
        write(OUT/'status.json',dict(stage=a.stage,status='failed',error=str(e)));raise


if __name__=='__main__':main()
