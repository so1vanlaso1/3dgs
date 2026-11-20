"""Review matched raw meshes and compare the matte pair with held-out depth."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

from research_campaign import REPO, sha, write
from week12_comparison import ARMS, MATTE, SCENES, data_path, read, run_path

sys.path.insert(0, str(REPO / 'examples'))
from datasets.colmap import Dataset, Parser


DB3 = REPO / 'data/db3_r2_local_20261110'
VIEWS = {'room': (0, 19, 28), 'matte': (0, 1, 2)}
ANCHORS = {'matte': (0.86, 0.56)}


def render_mesh(mesh, scene, rays):
    result = scene.cast_rays(rays)
    depth = result['t_hit'].numpy()
    hit = np.isfinite(depth) & (depth > 0)
    faces = np.asarray(mesh.triangles)[result['primitive_ids'].numpy()[hit]]
    uv = result['primitive_uvs'].numpy()[hit]
    weights = np.column_stack((1 - uv.sum(axis=1), uv))
    colors = np.asarray(mesh.vertex_colors)
    image = np.full((*depth.shape, 3), 24, dtype=np.uint8)
    image[hit] = np.clip(np.einsum('ij,ijk->ik', weights, colors[faces]) * 255, 0, 255)
    return depth, hit, image


def matte_sources(pair):
    manifest = read(DB3 / 'manifest.json')
    rows = {row['name']: row for row in manifest['frames']}
    split = pair['split']
    init = read(MATTE / 'initialization_report.json')
    geometry = Path(init['geometry'])
    fit = {row['depth_message_id'] for row in read(geometry / 'input.json')['frames']}
    eval_depth_ids = {rows[name]['depth_message_id'] for name in split['val']}
    assert len(eval_depth_ids) == len(split['val']) and not (fit & eval_depth_ids)
    training_files = {path.name.removesuffix('.npz') for path in (MATTE / 'dataset/depth').glob('*.npz')}
    assert training_files == set(split['train'])
    localization = read(geometry / 'evaluation_localization.json')
    assert {row['name'] for row in localization['frames'] if row['accepted']} == set(split['val'])
    with np.load(MATTE / 'supported_initialization.npz') as points:
        assert len(points['xyz']) == init['initial_points']
        assert set(points['source_id']) <= {rows[name]['rgb_message_id'] for name in split['train']}
    source = {}
    for name in split['val']:
        row = rows[name]
        path = DB3 / 'depth' / f'{name}.npz'
        assert row['role'] == 'eval' and not row['original_train']
        assert sha(path) == row['depth_sha256'] and sha(DB3 / 'images' / name) == row['rgb_sha256']
        source[name] = row
    training = []
    for name in split['train']:
        with np.load(MATTE / 'dataset/depth' / f'{name}.npz') as samples:
            training.append(dict(name=name, sampled=len(samples['confidence']),
                                 positive_weight=int((samples['confidence'] > 0).sum())))
    return dict(held_out_sources=source, training_depth=training,
                localization=localization['convention'],
                fitting_depth_message_ids=sorted(fit), initial_points=init['initial_points'],
                source_recording_sha256=manifest['source_sha256'],
                extraction_excludes_sensor_depth=True, metrology_reference=False,
                thresholds=dict(sensor_range_m=[0.3, 6], neighborhood='3x3 range < max(0.03 m, 2% of center depth)',
                                gaussian_alpha_strictly_above=0.7))


def sensor_masks(measured):
    valid = np.isfinite(measured) & (measured > 0.3) & (measured < 6)
    z = np.where(valid, measured, 0).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.uint8)
    nearby_min, nearby_max = cv2.erode(z, kernel), cv2.dilate(z, kernel)
    stable = valid & (nearby_min > 0.3) & (nearby_max - nearby_min < np.maximum(0.03, 0.02 * z))
    stable[[0, -1]] = False
    stable[:, [0, -1]] = False
    return valid, stable


def depth_statistics(delta):
    if not len(delta):
        return None
    absolute = np.abs(delta)
    return dict(pixels=len(delta), median_signed_m=float(np.median(delta)),
                median_abs_m=float(np.median(absolute)), p90_abs_m=float(np.percentile(absolute, 90)),
                mae_m=float(absolute.mean()), within_5cm=float((absolute <= 0.05).mean()))


def compare_depth(measured, sources, predictions, per_arm_errors):
    valid, stable = sensor_masks(measured)
    summary = dict(pixels=measured.size, valid_sensor_pixels=int(valid.sum()), stable_sensor_pixels=int(stable.sum()),
                   arms={arm: {} for arm in ARMS}, joint_pixels={})
    for kind, model in predictions.items():
        masks = {arm: stable & model[arm][1] for arm in ARMS}
        common = masks[ARMS[0]] & masks[ARMS[1]]
        summary['joint_pixels'][kind] = int(common.sum())
        for arm in ARMS:
            estimate, eligible = model[arm]
            delta = estimate[common] - measured[common]
            per_arm_errors[kind][arm].append(delta)
            summary['arms'][arm][kind] = dict(stable_sensor_pixels_covered=int(masks[arm].sum()),
                                              valid_sensor_pixels_covered=int((valid & eligible).sum()),
                                              depth_consistency_on_joint_pixels=depth_statistics(delta))
    return dict(**sources, **summary)


def make_edit_plan(scene, hit, depths, rays, camera, record):
    height, width = hit.shape
    x, y = (np.asarray(ANCHORS.get(scene, (0.5, 0.52))) * (width, height)).astype(int)
    if scene == 'room':
        position = np.array([0.006, 0.439, 0.20, 1.0])
        q = np.linalg.inv(camera['camtoworld']) @ position
        p = camera['K'] @ q[:3]
        x, y = np.rint(p[:2] / p[2]).astype(int)
    interior = cv2.erode(hit.astype(np.uint8), np.ones((11, 11), dtype=np.uint8)).astype(bool)
    ys, xs = np.nonzero(interior)
    if not len(xs):
        raise ValueError('No shared mesh patch in the first inspection camera')
    index = np.argmin((xs - x)**2 + (ys - y)**2)
    x, y = int(xs[index]), int(ys[index])
    ray = rays[y, x]
    direction = -ray[3:] / np.linalg.norm(ray[3:])
    voxel = record['voxel_length']
    anchors = {arm: (ray[:3] + ray[3:] * depths[arm][y, x]).tolist() for arm in ARMS}
    return dict(pixel_xy=[x, y], image=camera['name'], image_size=[width, height], anchors=anchors,
                toward_camera_unit_vector=direction.tolist(), selection_radius=4 * voxel, peak_offset=0.5 * voxel,
                camera_to_world_cv=camera['camtoworld'].tolist(), K=camera['K'].tolist())


def review(root, scene):
    pair = read(root / f'{scene}_pair.json')
    folder = root / 'review' / scene
    folder.mkdir(parents=True, exist_ok=True)
    parser = Parser(str(data_path(scene)), factor=1, normalize=True, test_every=8)
    dataset = Dataset(parser, split='val')
    meshes, extract, ray_scenes = {}, {}, {}
    for arm in ARMS:
        extract[arm] = pair['arms'][arm]['mesh']
        path = REPO / extract[arm]['folder'] / 'raw_mesh.ply'
        assert sha(path) == extract[arm]['mesh_sha256']
        meshes[arm] = o3d.io.read_triangle_mesh(str(path))
        ray_scenes[arm] = o3d.t.geometry.RaycastingScene()
        ray_scenes[arm].add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(meshes[arm]))
    depth_protocol = matte_sources(pair) if scene == 'matte' else None
    splats = {}
    if depth_protocol:
        from db3_evaluate_pilot import load_splats, render

        splats = {arm: load_splats(run_path(root, scene, arm) / 'ckpts/final_post_update.pt') for arm in ARMS}

    depth_checks, inspections, panels = [], [], []
    errors = {kind: {arm: [] for arm in ARMS} for kind in ('Gaussian', 'mesh')}
    views = VIEWS.get(scene, (0, len(dataset)//2, len(dataset)-1))
    for index in views:
        sample = dataset[index]
        photo = sample['image'].numpy().astype(np.uint8)
        k, c2w = sample['K'].numpy(), sample['camtoworld'].numpy()
        name = parser.image_names[dataset.indices[index]]
        assert name == pair['split']['val'][index]
        h, w = photo.shape[:2]
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(k, np.linalg.inv(c2w), w, h)
        depths, hits, renders = {}, {}, {}
        for arm in ARMS:
            depths[arm], hits[arm], renders[arm] = render_mesh(meshes[arm], ray_scenes[arm], rays)
        joint_hit = hits[ARMS[0]] & hits[ARMS[1]]
        if index == views[0]:
            camera = dict(K=k, camtoworld=c2w, name=name)
            plan = make_edit_plan(scene, joint_hit, depths, rays.numpy(), camera, extract[ARMS[0]])
            plan['mesh_files'] = {arm: extract[arm]['folder'] + '/raw_mesh.ply' for arm in ARMS}
            plan['mesh_sha256'] = {arm: extract[arm]['mesh_sha256'] for arm in ARMS}
            write(folder / 'edit_plan.json', plan)

        inspection = dict(image=name, pixels=w*h, both_mesh_hits=int(joint_hit.sum()),
                          mesh_hits={arm: int(hits[arm].sum()) for arm in ARMS})
        inspections.append(inspection)
        thumb_width = 460
        thumb_height = round(thumb_width * h / w)
        panel = Image.new('RGB', (3 * thumb_width, thumb_height + 29), (26, 26, 26))
        label = ImageDraw.Draw(panel)
        for column, (caption, rgb) in enumerate(zip((f'RGB    {name}', 'B00 raw mesh', 'A05-Abs raw mesh'),
                                                   (photo, *(renders[arm] for arm in ARMS)))):
            left = column * thumb_width
            panel.paste(Image.fromarray(rgb).resize((thumb_width, thumb_height), Image.Resampling.LANCZOS), (left, 29))
            label.text((left + 8, 8), caption, fill='white')
            if index == views[0]:
                x, y = plan['pixel_xy']
                px, py = left + x * thumb_width / w, 29 + y * thumb_height / h
                label.ellipse((px-6, py-6, px+6, py+6), outline='#ffce4e', width=2)
        panels.append(panel)

        if depth_protocol:
            with np.load(DB3 / 'depth' / f'{name}.npz') as sensor:
                measured = sensor['depth_m']
            assert measured.shape == (h, w)
            assert np.allclose(k, read(DB3 / 'calibration_clean.json')['K'], atol=0.001)
            scale = float(np.linalg.norm(parser.transform[:3, 0]))
            predictions = {'mesh': {arm: (depths[arm] / scale, hits[arm]) for arm in ARMS}, 'Gaussian': {}}
            for arm in ARMS:
                _, depth, alpha = render(splats[arm], c2w, k, w, h)
                predictions['Gaussian'][arm] = (depth / scale, (alpha > 0.7) & np.isfinite(depth) & (depth > 0))
            source = depth_protocol['held_out_sources'][name]
            depth_checks.append(compare_depth(measured, dict(image=name, depth_message_id=source['depth_message_id']),
                                              predictions, errors))

    overview = Image.new('RGB', (panels[0].width, sum(panel.height for panel in panels)))
    top = 0
    for panel in panels:
        overview.paste(panel, (0, top))
        top += panel.height
    overview.save(folder / 'mesh_comparison.jpg', quality=93)
    write(folder / 'mesh_review.json', dict(scene=scene, inspections=inspections))
    if depth_protocol:
        pooled = {kind: {arm: depth_statistics(np.concatenate(rows)) for arm, rows in arms.items()}
                  for kind, arms in errors.items()}
        write(folder / 'sensor_depth_consistency.json', dict(protocol=depth_protocol, views=depth_checks,
                                                            pooled_consistency_on_joint_pixels=pooled))
    print('REVIEWED', scene, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, required=True)
    cli.add_argument('--scene', choices=SCENES, required=True)
    options = cli.parse_args()
    review(options.root.resolve(), options.scene)
