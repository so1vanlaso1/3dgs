"""Apply an identical local deformation task to each mesh and verify saved geometry."""

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
import numpy as np


REPO = Path(__file__).resolve().parents[1]
GROUP = 'Selected surface patch'
EDIT = 'Surface patch offset toward camera'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def coords(vertices):
    positions = np.empty(3 * len(vertices), dtype=np.float32)
    vertices.foreach_get('co', positions)
    return positions.reshape(-1, 3)


def patch(mesh, center, radius):
    positions = coords(mesh.vertices)
    distances = np.linalg.norm(positions - center, axis=1)
    allowed = set(np.flatnonzero(distances < radius).tolist())
    if not allowed:
        raise ValueError('No vertices around the shared inspection ray')
    neighbors = {i: [] for i in allowed}
    for edge in mesh.edges:
        u, v = edge.vertices
        if u in allowed and v in allowed:
            neighbors[u].append(v)
            neighbors[v].append(u)
    seed = min(allowed, key=lambda i: (distances[i], i))
    selected = {seed}
    queue = deque([seed])
    while queue:
        for neighbor in neighbors[queue.popleft()]:
            if neighbor not in selected:
                selected.add(neighbor)
                queue.append(neighbor)
    if len(selected) < 8:
        raise ValueError(f'Expected a connected patch of at least eight vertices, got {len(selected)}')
    indices = sorted(selected)
    weights = 0.5 + 0.5 * np.cos(np.pi * distances[indices] / radius)
    return indices, weights / weights.max()


def create(folder, arm, plan):
    result = folder / f'{arm}_edited.blend'
    if result.exists():
        raise FileExistsError(result)
    source = REPO / plan['mesh_files'][arm]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == plan['mesh_sha256'][arm]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.ply_import(filepath=str(source), global_scale=1, forward_axis='Y', up_axis='Z', import_colors='SRGB')
    obj = bpy.context.object
    obj.name = f'{arm} raw mesh'
    basis = coords(obj.data.vertices)
    indices, weights = patch(obj.data, plan['anchors'][arm], plan['selection_radius'])
    group = obj.vertex_groups.new(name=GROUP)
    obj.shape_key_add(name='Basis')
    edited = obj.shape_key_add(name=EDIT)
    shift = np.asarray(plan['toward_camera_unit_vector']) * plan['peak_offset']
    for i, weight in zip(indices, weights):
        group.add([i], float(weight), 'REPLACE')
        edited.data[i].co += Vector(shift * weight)
        obj.data.vertices[i].select = True
    edited.value = 1.0
    obj.active_shape_key_index = 1
    obj['Source mesh SHA-256'] = plan['mesh_sha256'][arm]
    obj['Basis coordinates SHA-256'] = hashlib.sha256(basis.tobytes()).hexdigest()
    obj['Inspection frame'] = plan['image']
    obj['Anchor pixel XY'] = plan['pixel_xy']
    obj['Selection radius'] = plan['selection_radius']
    obj['Peak offset'] = plan['peak_offset']
    obj['Offset direction world XYZ'] = plan['toward_camera_unit_vector']

    rotation = Matrix(plan['camera_to_world_cv']).to_3x3() @ Matrix.Diagonal(Vector((1, -1, -1)))
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue
            view = area.spaces.active
            view.region_3d.view_rotation = rotation.to_quaternion()
            view.region_3d.view_location = plan['anchors'][arm]
            view.region_3d.view_distance = 8 * plan['selection_radius']
            view.shading.color_type = 'VERTEX'
            view.shading.show_cavity = True
    obj.data.update()
    bpy.ops.wm.save_as_mainfile(filepath=str(result))


def verify(folder, arm, plan):
    document = folder / f'{arm}_edited.blend'
    obj = bpy.data.objects[f'{arm} raw mesh']
    assert Path(bpy.data.filepath).resolve() == document.resolve()
    assert obj['Source mesh SHA-256'] == plan['mesh_sha256'][arm]
    assert obj.matrix_world == Matrix.Identity(4)
    basis = coords(obj.data.shape_keys.key_blocks['Basis'].data)
    assert hashlib.sha256(basis.tobytes()).hexdigest() == obj['Basis coordinates SHA-256']
    assert obj.data.shape_keys.key_blocks[EDIT].value == 1.0
    assert all(len(face.vertices) == 3 for face in obj.data.polygons)
    indices, weights = patch(obj.data, plan['anchors'][arm], plan['selection_radius'])
    group = obj.vertex_groups[GROUP]
    saved_indices = [vertex.index for vertex in obj.data.vertices if any(item.group == group.index for item in vertex.groups)]
    assert saved_indices == indices
    np.testing.assert_allclose([group.weight(i) for i in indices], weights, atol=1e-7)
    evaluation = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    delta = coords(evaluation.data.vertices) - basis
    expected = np.zeros_like(delta)
    expected[indices] = weights[:, None] * np.asarray(plan['toward_camera_unit_vector']) * plan['peak_offset']
    np.testing.assert_allclose(delta, expected, rtol=0, atol=5e-7)
    assert not np.any(np.delete(delta, indices, axis=0))
    report = dict(blender=bpy.app.version_string, document=document.name, vertices=len(basis),
                  triangles=len(obj.data.polygons), selected_vertices=len(indices),
                  max_evaluated_displacement=float(np.linalg.norm(delta, axis=1).max()),
                  peak_requested=plan['peak_offset'], unchanged_outside_selection=True,
                  object_transform_identity=True, surface_selection_connected=True,
                  saved_shape_key_active=True, verified_after_reopen=True,
                  document_sha256=hashlib.sha256(document.read_bytes()).hexdigest())
    (folder / f'{arm}_edit_verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('REOPENED', arm, report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('create', 'verify'))
    parser.add_argument('folder', type=Path)
    parser.add_argument('arm', choices=('B00', 'A05_absgrad'))
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    plan = read(args.folder / 'edit_plan.json')
    (create if args.action == 'create' else verify)(args.folder.resolve(), args.arm, plan)
