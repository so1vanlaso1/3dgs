"""Build a reversible, conservative ball-pivot mesh for visual inspection only."""
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / os.environ.get('ROOM_MESH_SOURCE', 'results/room_improve_sdk_source_cloud_20261103')
DATA = ROOT / 'data/room_improve_sdk_rectified_20261103'
OUT = ROOT / os.environ.get('ROOM_MESH_OUTPUT', 'results/room_improve_sdk_test_mesh_20261103')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    src = SRC / 'development_only_unfiltered.ply'
    pc = o3d.io.read_point_cloud(str(src))
    original = np.asarray(pc.points).copy()
    assert len(original) > 0 and pc.has_colors()
    meta = json.loads((SRC / 'report.json').read_text(encoding='utf8'))
    with np.load(SRC / 'evidence.npz') as f:
        assert np.allclose(original, f['points_m'], atol=2e-6)
        source = f['source_group_index'].copy()
    model = pycolmap.Reconstruction(DATA / 'rgbd/dataset/sparse')
    imgs = {im.name: im for im in model.images.values()}
    centers = []
    for name in meta['development_images']:
        T = imgs[name].cam_from_world().matrix()
        centers.append(-T[:,:3].T @ T[:,3])
    center = np.asarray(centers)[source]
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
    normals = np.asarray(pc.normals)
    normals[np.sum(normals * (center - original), axis=1) < 0] *= -1
    pc.normals = o3d.utility.Vector3dVector(normals)
    radii = [0.045, 0.075, 0.12]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc, o3d.utility.DoubleVector(radii))
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    assert np.allclose(vertices, original) and len(faces) > 0
    edges = np.concatenate([np.linalg.norm(vertices[faces[:,i]] - vertices[faces[:,(i+1)%3]], axis=1) for i in range(3)])
    used = np.unique(faces)
    mesh.compute_vertex_normals()
    main_ply = OUT / 'room_experimental_mesh.ply'
    assert o3d.io.write_triangle_mesh(str(main_ply), mesh, write_ascii=False, write_vertex_colors=True)
    uncertainty_src = SRC / 'development_dense_stable_uncertainty.ply'
    uncertainty_hash = None
    if uncertainty_src.exists():
        uncertainty = o3d.io.read_point_cloud(str(uncertainty_src))
        assert np.allclose(np.asarray(uncertainty.points), original) and uncertainty.has_colors()
        mesh.vertex_colors = uncertainty.colors
        uncertainty_ply = OUT / 'room_experimental_mesh_uncertainty.ply'
        assert o3d.io.write_triangle_mesh(str(uncertainty_ply), mesh, write_ascii=False, write_vertex_colors=True)
        uncertainty_hash = digest(uncertainty_ply)
    check = o3d.io.read_triangle_mesh(str(main_ply))
    assert len(check.vertices) == len(vertices) and len(check.triangles) == len(faces) and check.has_vertex_colors()
    # CPU painter's algorithm: approximate preview, not a visibility/accuracy measurement.
    colors = np.asarray(pc.colors)
    cam = next(iter(model.cameras.values()))
    fx, fy, cx, cy = cam.params
    views = json.loads((ROOT / 'results/room_improve_baseline_inspection_20261103/inspection_poses.json').read_text())['views']
    chosen = [v for v in views if v['requested_seconds'] in (43,61,68,110)]
    panels = []
    for v in chosen:
        name = v['image_name']; T = imgs[name].cam_from_world().matrix()
        q = vertices @ T[:,:3].T + T[:,3]
        z = q[:,2]
        zsafe = np.maximum(z, 0.01)
        uv = np.stack([fx*q[:,0]/zsafe+cx, fy*q[:,1]/zsafe+cy], axis=1)
        valid = (z[faces] > .3).all(axis=1) & (z[faces] < 8).all(axis=1)
        valid &= (np.abs(uv[faces]) < 10000).all(axis=(1,2))
        f = faces[valid]
        depth = z[f].mean(axis=1)
        f = f[np.argsort(-depth)]
        canvas = np.zeros((cam.height, cam.width, 3), np.uint8)
        for tri in f:
            poly = np.rint(uv[tri]).astype(np.int32)
            if (poly[:,0].max()<0 or poly[:,0].min()>=cam.width or poly[:,1].max()<0 or poly[:,1].min()>=cam.height):
                continue
            color = (colors[tri].mean(axis=0)*255).astype(np.uint8)
            cv2.fillConvexPoly(canvas, poly, tuple(int(c) for c in color), lineType=cv2.LINE_8)
        rgb = Image.open(DATA/'images'/name).convert('RGB')
        panel = Image.new('RGB', (cam.width*2, cam.height+30), 'white')
        panel.paste(rgb, (0,30)); panel.paste(Image.fromarray(canvas), (cam.width,30))
        draw=ImageDraw.Draw(panel); draw.text((4,7), f'{v["requested_seconds"]}s candidate RGB',fill='black'); draw.text((cam.width+4,7),'experimental mesh - approximate preview',fill='black')
        panel.save(OUT/f'{v["requested_seconds"]:03d}s.png')
        panels.append(panel.resize((1018, 306)))
    sheet=Image.new('RGB',(1018,306*len(panels)),'white')
    for i,panel in enumerate(panels): sheet.paste(panel,(0,i*306))
    sheet.save(OUT/'captured_views_contact_sheet.jpg',quality=92)
    report = dict(method='Open3D ball pivoting; normals from 15 cm / 30 neighbors oriented towards each point source camera',radii_m=radii,
      input_sha256=digest(src), vertex_count=len(vertices), triangle_count=len(faces), vertices_used_by_faces=len(used),
      unused_vertices=len(vertices)-len(used),edge_length_m_percentiles=dict(zip(['p50','p90','p99','max'],map(float,np.percentile(edges,[50,90,99,100])))),
      rgb_mesh_sha256=digest(main_ply), uncertainty_mesh_sha256=uncertainty_hash,open3d_version=o3d.__version__,seconds=round(time.perf_counter()-start,1),
      caveats='Experimental mesh from sampled point cloud, unverified poses/calibration; surfaces interpolate between points, may bridge unrelated surfaces, no ground truth; no points cleaned. Uncertainty is vertex diagnostic color, not validated surface confidence. Preview uses average vertex color and triangle painter sorting, not per-pixel z buffering; occlusion can be wrong.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
