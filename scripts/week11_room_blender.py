"""Create or reopen the Week 11 Room design document in Blender 4.5 LTS."""

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import bpy
from mathutils import Matrix, Vector
import numpy as np


OBJECT = "Room | cleaned mesh"
PATCH = "Coffee table | selected surface"
LIFT = "Tabletop lift (+Z)"
BASE_COLOR = "PLY vertex colors"
EDIT_COLOR = "Tabletop edit preview"
CENTER = (0.006, 0.439)
INNER_RADIUS = 0.027
OUTER_RADIUS = 0.056
HEIGHT_RANGE = (0.185, 0.216)
MIN_NORMAL_Z = 0.55
LIFT_DISTANCE = 0.040
TEAL = (0.025, 0.58, 0.44, 1.0)
CROP = (36, 285, 376, 245)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coordinates(points):
    data = np.empty(len(points) * 3, dtype=np.float32)
    points.foreach_get("co", data)
    return data.reshape(-1, 3)


def select_surface(obj, vertices):
    radial = np.linalg.norm(vertices[:, :2] - CENTER, axis=1)
    candidate = (radial < OUTER_RADIUS) & (vertices[:, 2] > HEIGHT_RANGE[0]) & (vertices[:, 2] < HEIGHT_RANGE[1])
    normals_z = np.array([vertex.normal.z for vertex in obj.data.vertices])
    candidate &= normals_z > MIN_NORMAL_Z
    allowed = set(np.flatnonzero(candidate).tolist())
    seed = min(allowed, key=radial.__getitem__)
    adjacent = {i: [] for i in allowed}
    for edge in obj.data.edges:
        u, v = edge.vertices
        if u in allowed and v in allowed:
            adjacent[u].append(v)
            adjacent[v].append(u)
    selected = {seed}
    queue = deque([seed])
    while queue:
        for vertex in adjacent[queue.popleft()]:
            if vertex not in selected:
                selected.add(vertex)
                queue.append(vertex)
    indices = sorted(selected)
    taper = np.clip((radial[indices] - INNER_RADIUS) / (OUTER_RADIUS - INNER_RADIUS), 0, 1)
    weights = 0.5 + 0.5 * np.cos(math.pi * taper)
    if len(indices) < 8 or not np.isclose(weights.max(), 1.0):
        raise RuntimeError("The connected tabletop selection did not match the expected patch")
    return indices, weights


def set_vertex_display(mesh, name):
    attrs = mesh.color_attributes
    attrs.active_color = attrs[name]
    attrs.render_color_index = attrs.find(name)


def setup_camera(scene, camera_info):
    matrix = Matrix(camera_info["camera_to_world_cv"]) @ Matrix.Diagonal(Vector((1, -1, -1, 1)))
    x, y, width, height = CROP
    k = camera_info["K"]
    camera = bpy.data.cameras.new("Coffee table crop | DSCF4667")
    camera.sensor_fit = "HORIZONTAL"
    camera.lens = camera.sensor_width * k[0][0] / width
    camera.shift_x = (width / 2 - (k[0][2] - x)) / width
    camera.shift_y = (k[1][2] - y - height / 2) / width
    camera.passepartout_alpha = 0.85
    camera_object = bpy.data.objects.new(camera.name, camera)
    scene.collection.objects.link(camera_object)
    camera_object.matrix_world = matrix
    scene.camera = camera_object

    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 3 * width
    scene.render.resolution_y = 3 * height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_y = k[0][0] / k[1][1]
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "VERTEX"
    shading.show_cavity = True
    shading.cavity_type = "BOTH"


def create(folder, document):
    if document.exists():
        raise FileExistsError(document)
    started = time.perf_counter()
    source = folder / "room_cleaned.ply"
    report = json.loads((folder / "cleanup.json").read_text(encoding="utf-8"))
    if sha256(source) != report["output_sha256"]:
        raise RuntimeError("Cleaned mesh does not match the cleanup report")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.ply_import(filepath=str(source), global_scale=1, forward_axis="Y", up_axis="Z", import_colors="SRGB")
    obj = bpy.context.object
    obj.name = OBJECT
    mesh = obj.data
    vertices = coordinates(mesh.vertices)
    stats = report["after_simplify_reloaded"]
    np.testing.assert_allclose([vertices.min(0), vertices.max(0)], stats["bounds"], atol=1e-7)
    assert len(mesh.polygons) == stats["triangles"] and len(vertices) == stats["vertices"]

    indices, weights = select_surface(obj, vertices)
    group = obj.vertex_groups.new(name=PATCH)
    obj.shape_key_add(name="Basis")
    lift = obj.shape_key_add(name=LIFT)
    source_colors = mesh.color_attributes.active_color
    assert source_colors.domain == "POINT"
    source_colors.name = BASE_COLOR
    preview_colors = mesh.color_attributes.new(name=EDIT_COLOR, type="FLOAT_COLOR", domain="POINT")
    colors = np.empty(len(vertices) * 4, dtype=np.float32)
    source_colors.data.foreach_get("color", colors)
    colors = colors.reshape(-1, 4)
    colors[indices] = 0.2 * colors[indices] + 0.8 * np.array(TEAL)
    preview_colors.data.foreach_set("color", colors.ravel())
    for i, weight in zip(indices, weights):
        group.add([i], float(weight), "REPLACE")
        lift.data[i].co.z += LIFT_DISTANCE * weight
        mesh.vertices[i].select = True
    lift.value = 1.0
    obj.active_shape_key_index = 1
    set_vertex_display(mesh, EDIT_COLOR)
    mesh.update()

    scene = bpy.context.scene
    scene.unit_settings.system = "NONE"
    camera_info = json.loads((folder / "silhouette_cameras.json").read_text(encoding="utf-8"))[0]
    setup_camera(scene, camera_info)
    obj["Surface"] = "Front-left patch of the coffee tabletop, seen in DSCF4667"
    obj["Axes and scale"] = "Source XYZ, +Z up, unit scale, normalized scene units"
    obj["Peak lift (+Z)"] = LIFT_DISTANCE
    obj["Cleaned PLY SHA256"] = report["output_sha256"]

    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                space.shading.color_type = "VERTEX"
                space.shading.show_cavity = True
                space.shading.cavity_type = "BOTH"
                space.overlay.show_floor = False
                space.overlay.show_axis_x = False
                space.overlay.show_axis_y = False
                space.show_region_ui = True
                space.region_3d.view_perspective = "CAMERA"
                space.region_3d.view_camera_zoom = 20
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(document))
    write_json(folder / "design_create.json", {"create_and_save_seconds": time.perf_counter() - started,
                                               "document_sha256": sha256(document)})


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def verify_and_capture(folder, document):
    if Path(bpy.data.filepath).resolve() != document or bpy.context.scene is None:
        raise RuntimeError("Start a fresh Blender process with the saved .blend as its input")
    started = time.perf_counter()
    scene = bpy.context.scene
    obj = bpy.data.objects[OBJECT]
    mesh = obj.data
    group = obj.vertex_groups[PATCH]
    report = json.loads((folder / "cleanup.json").read_text(encoding="utf-8"))
    creation = json.loads((folder / "design_create.json").read_text(encoding="utf-8"))
    document_hash = sha256(document)
    assert creation["document_sha256"] == document_hash
    assert obj["Cleaned PLY SHA256"] == report["output_sha256"]
    assert scene.unit_settings.system == "NONE" and tuple(obj.scale) == (1, 1, 1)
    assert obj.matrix_world == Matrix.Identity(4)
    key = mesh.shape_keys.key_blocks[LIFT]
    assert key.value == 1 and mesh.color_attributes.active_color.name == EDIT_COLOR
    original = coordinates(mesh.vertices)
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    displacement = coordinates(evaluated.data.vertices) - original
    moved = np.flatnonzero(np.linalg.norm(displacement, axis=1) > 1e-7)
    indices, weights = select_surface(obj, original)
    stored_selection = [vertex.index for vertex in mesh.vertices if any(g.group == group.index for g in vertex.groups)]
    np.testing.assert_array_equal(moved, indices)
    np.testing.assert_array_equal(stored_selection, indices)
    np.testing.assert_allclose([group.weight(i) for i in indices], weights, atol=1e-7)
    np.testing.assert_allclose(displacement[moved, :2], 0, atol=1e-7)
    np.testing.assert_allclose(displacement[moved, 2], LIFT_DISTANCE * weights, atol=1e-7)
    assert len(evaluated.data.polygons) == report["after_simplify_reloaded"]["triangles"]

    for value, attr, filename in [(1, EDIT_COLOR, "tabletop_after_reopen.png"),
                                  (0, BASE_COLOR, "tabletop_before.png"),
                                  (1, BASE_COLOR, "tabletop_geometry_after_reopen.png")]:
        key.value = value
        set_vertex_display(mesh, attr)
        scene.render.filepath = str(folder / filename)
        bpy.ops.render.render(write_still=True)
    assert sha256(document) == document_hash
    base = original[moved]
    normal = np.mean([tuple(mesh.vertices[i].normal) for i in moved], axis=0)
    normal /= np.linalg.norm(normal)
    write_json(folder / "design_reopen.json", {
        "blender": bpy.app.version_string, "build_hash": bpy.app.build_hash.decode(),
        "document": document.name, "document_sha256": document_hash,
        "check": "Opened in a separate Blender process after the creation process exited",
        "surface": obj["Surface"], "vertices_moved": len(moved), "triangles": len(mesh.polygons),
        "selection": {"vertex_group": PATCH, "xy_center": CENTER, "inner_radius": INNER_RADIUS,
                      "outer_radius": OUTER_RADIUS, "height_range": HEIGHT_RANGE, "minimum_normal_z": MIN_NORMAL_Z,
                      "connectivity": "Edge connected from nearest XY vertex", "falloff": "Cosine"},
        "geometry": {"shape_key": LIFT, "saved_value": 1.0,
                     "peak_displacement_xyz": [0, 0, float(displacement[moved, 2].max())],
                     "surface_bounds_before_xyz": [base.min(0).tolist(), base.max(0).tolist()],
                     "mean_vertex_normal_before_xyz": normal.tolist()},
        "coordinates": {"units": "Normalized source scene units", "up": "+Z",
                        "object_scale_xyz": list(obj.scale), "object_rotation_euler_deg": [0, 0, 0],
                        "object_translation_xyz": list(obj.location), "editor_import": "Forward Y / Up Z, scale 1"},
        "camera_crop_xywh": CROP, "verify_and_capture_seconds": time.perf_counter() - started,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "verify"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    folder = args.folder.resolve()
    document = folder / "room_tabletop_edited.blend"
    if args.mode == "create":
        create(folder, document)
    else:
        verify_and_capture(folder, document)


if __name__ == "__main__":
    main()
