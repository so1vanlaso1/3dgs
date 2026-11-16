"""Extract an initial polygon mesh by TSDF fusion of Gaussian-rendered depth.

Uses only training cameras and optimized checkpoint weights. No sensor depth,
COLMAP points or held-out images are fused. World coordinates match training
normalization; they are not necessarily meters.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import open3d as o3d
import torch
from research_campaign import REPO, sha, write
sys.path[:0] = [str(REPO/'examples'), str(REPO)]
from simple_trainer import Config, Runner
from gsplat.scene import GaussianScene
from gsplat.stage import Stage


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--views', type=int, default=64)
    p.add_argument('--downsample', type=int, default=4)
    p.add_argument('--voxel', type=float, default=.01, help='Fraction of scene scale')
    p.add_argument('--max-resolution',type=int,default=192)
    a = p.parse_args()
    if a.views < 2 or a.downsample < 1 or a.voxel <= 0:
        p.error('views>=2, downsample>=1, voxel>0 required')
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cfg = Config(data_dir=str(a.data), data_factor=1, result_dir=str(a.output/'loader'),
                 disable_viewer=True, disable_video=True, packed=True, data_workers=0)
    runner = Runner(0,0,1,cfg)
    ckpt = torch.load(a.checkpoint, map_location=runner.device, weights_only=True)
    assert all(torch.isfinite(t).all() for t in ckpt['splats'].values())
    for key in runner.splats:
        runner.splats[key].data = ckpt['splats'][key]
    runner.scene = GaussianScene.from_splats(runner.splats, id='scene')
    runner.splats = runner.scene.splats
    runner.stage = Stage()
    runner.stage.add_scene(runner.scene, runner.rasterize_splats)
    # The local Open3D 0.20 scalable backend produced zero voxels even for a
    # synthetic fronto-parallel plane. The uniform backend passes that control.
    low,high=np.percentile(runner.parser.points,[1,99],axis=0)
    center=(low+high)/2
    length=float(np.max(high-low)*1.2)
    resolution=min(a.max_resolution,int(np.ceil(length/(a.voxel*runner.scene_scale))))
    if resolution<16: raise ValueError('TSDF resolution must be at least 16')
    voxel=length/resolution
    origin=center-length/2
    volume = o3d.pipelines.integration.UniformTSDFVolume(length=length,resolution=resolution,
        sdf_trunc=4*voxel,origin=origin,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    ids = np.unique(np.linspace(0,len(runner.trainset)-1,min(a.views,len(runner.trainset))).astype(int))
    records = []
    for idx in ids:
        sample = runner.trainset[int(idx)]
        h0,w0 = sample['image'].shape[:2]
        h,w = h0//a.downsample,w0//a.downsample
        K = sample['K'].clone().numpy()
        K[0] *= w/w0
        K[1] *= h/h0
        cam = sample['camtoworld'].to(runner.device)[None]
        out, alpha, _ = runner.stage.render(runner.scene.id,camtoworlds=cam,
            Ks=torch.from_numpy(K).to(runner.device)[None],width=w,height=h,
            sh_degree=3,near_plane=.01,far_plane=1e10,render_mode='RGB+ED')
        image = out[0,...,:3].clamp(0,1).cpu().numpy()
        depth = out[0,...,3].cpu().numpy().copy()
        good = (alpha[0,...,0].cpu().numpy()>.7) & np.isfinite(depth) & (depth>0)
        depth[~good] = 0
        if 'mask' in sample:
            mask = sample['mask'].numpy()
            import cv2
            depth[~cv2.resize(mask.astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)] = 0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image((image*255).astype('uint8')),
            o3d.geometry.Image(depth.astype('float32')),depth_scale=1,
            depth_trunc=8*runner.scene_scale,convert_rgb_to_intensity=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w,h,K[0,0],K[1,1],K[0,2],K[1,2])
        volume.integrate(rgbd,intrinsic,np.linalg.inv(sample['camtoworld'].numpy()))
        records.append(dict(train_index=int(idx),image_id=int(sample['image_id']), valid_fraction=float(good.mean())))
        print('FUSED',idx,flush=True)
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    if len(mesh.triangles) == 0:
        raise RuntimeError('No triangles extracted; inspect depth/scale before changing thresholds')
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    assert np.isfinite(vertices).all()
    path = a.output/'raw_mesh.ply'
    assert o3d.io.write_triangle_mesh(str(path),mesh)
    # A polygon OBJ also permits import in standard design tools (colors remain in PLY).
    assert o3d.io.write_triangle_mesh(str(a.output/'raw_mesh.obj'),mesh,write_vertex_colors=False)
    reloaded = o3d.io.read_triangle_mesh(str(path))
    assert len(reloaded.vertices)==len(vertices) and len(reloaded.triangles)==len(faces)
    edges = np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    _,counts = np.unique(edges,axis=0,return_counts=True)
    _,components,_ = mesh.cluster_connected_triangles()
    # Independent mesh rasterization checks representation agreement, not truth.
    import cv2
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    checks, panels = [], []
    for index in np.linspace(0,len(runner.valset)-1,5).astype(int):
        sample = runner.valset[int(index)]
        h0,w0 = sample['image'].shape[:2]
        h,w = h0//a.downsample,w0//a.downsample
        K = sample['K'].numpy().copy()
        K[0] *= w/w0
        K[1] *= h/h0
        extrinsic = np.linalg.inv(sample['camtoworld'].numpy())
        rays=o3d.t.geometry.RaycastingScene.create_rays_pinhole(K,extrinsic,w,h)
        mesh_depth=ray_scene.cast_rays(rays)['t_hit'].numpy()
        out,alpha,_=runner.stage.render(runner.scene.id,
            camtoworlds=sample['camtoworld'].to(runner.device)[None],
            Ks=torch.from_numpy(K).to(runner.device)[None],width=w,height=h,
            sh_degree=3,render_mode='RGB+ED')
        expected=out[0,...,3].cpu().numpy()
        valid=(alpha[0,...,0].cpu().numpy()>.7)&np.isfinite(expected)&(expected>0)
        hit=np.isfinite(mesh_depth)&(mesh_depth>0)
        both=hit&valid
        diff=np.abs(expected[both]-mesh_depth[both])
        checks.append(dict(val_index=int(index),opaque_pixels=int(valid.sum()),
            mesh_coverage_on_opaque=float(both.sum()/max(valid.sum(),1)),
            median_depth_disagreement=float(np.median(diff)) if len(diff) else None,
            p95_depth_disagreement=float(np.percentile(diff,95)) if len(diff) else None))
        low,high=np.percentile(expected[valid],[5,95]) if valid.any() else (0,1)
        def color_depth(d,good):
            d=np.where(good,d,low)
            color=cv2.applyColorMap((np.clip((d-low)/max(high-low,1e-6),0,1)*255).astype('uint8'),cv2.COLORMAP_TURBO)
            color[~good]=0
            return cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
        rgb=(out[0,...,:3].clamp(0,1).cpu().numpy()*255).astype('uint8')
        panels.append(np.concatenate([rgb,color_depth(expected,valid),color_depth(mesh_depth,hit)],axis=1))
    from PIL import Image
    Image.fromarray(np.concatenate(panels,axis=0)).save(a.output/'mesh_depth_comparison.jpg')
    write(a.output/'report.json',dict(method='TSDF fusion of expected depth rendered from optimized 3D Gaussians',
        checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=sha(a.checkpoint),
        gaussian_count=len(runner.splats['means']),data=str(a.data.resolve()),
        parser_transform=runner.parser.transform.tolist(),scene_scale=float(runner.scene_scale),
        tsdf_backend='uniform',resolution=resolution,box_origin=origin.tolist(),box_length=length,
        crop_policy='Cube around common SfM 1st–99th percentile bounds, enlarged 20%; unbounded exterior excluded',
        voxel_length=float(voxel),sdf_trunc=float(4*voxel),alpha_threshold=.7,
        training_views=records,vertices=len(vertices),triangles=len(faces),
        boundary_edges=int((counts==1).sum()),nonmanifold_edges=int((counts>2).sum()),
        components=len(components),bounds=[vertices.min(0).tolist(),vertices.max(0).tolist()],
        heldout_self_consistency=checks,
        elapsed_seconds=time.perf_counter()-started,mesh_sha256=sha(path),
        limitations='Expected depth can blend surfaces. Training-camera fusion; no geometry ground truth. Raw mesh may have holes/disconnected surfaces. Normalized scene units, not certified metric scale.'))
    print('MESH_COMPLETE',len(vertices),len(faces),flush=True)


if __name__ == '__main__':
    main()
