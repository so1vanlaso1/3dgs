"""Clean the fixed Week 10 public Room mesh and compare held-out silhouettes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/school_completion_replay_20261109/meshes_uniform/reduced"
SOURCE_SHA256 = "af370bb141609c651a48dc48fb7531c0574aa674657b370aa214d26d4c7e46fa"
DATA = ROOT / "data/room_factor2"
MIN_COMPONENT_FACES = 8
MAX_SPECK_AREA_VOXELS_SQUARED = 0.25
TRIANGLE_FRACTION = 0.65
BOUNDARY_WEIGHT = 8.0
VIEWS = (
    (0, "Coffee table", (19, 125, 200, 258)),
    (28, "Potted plant", (108, 0, 328, 258)),
    (19, "Armchair and footstool", (20, 17, 239, 190)),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def topology(mesh):
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, incidence = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    _, counts, _ = mesh.cluster_connected_triangles()
    return {
        "vertices": len(vertices), "triangles": len(faces), "components": len(counts),
        "boundary_edges": int((incidence == 1).sum()),
        "nonmanifold_edges": int((incidence > 2).sum()),
        "surface_area": mesh.get_surface_area(),
        "bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
    }


def remove_repeated_faces(mesh):
    """Remove duplicate vertex triples, including reversed winding."""
    mesh.remove_degenerate_triangles()
    faces = np.asarray(mesh.triangles)
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    remove = np.ones(len(faces), dtype=bool)
    remove[first] = False
    mesh.remove_triangles_by_mask(remove)
    return int(remove.sum())


def cleanup(output, source_folder=SOURCE):
    source = source_folder / "raw_mesh.ply"
    prior = json.loads((source_folder / "report.json").read_text(encoding="utf-8"))
    expected_hash = SOURCE_SHA256 if source_folder == SOURCE else prior["mesh_sha256"]
    if sha256(source) != expected_hash or prior["mesh_sha256"] != expected_hash:
        raise RuntimeError("Room input does not match the extraction record")
    if Path(prior["data"]).resolve() != DATA.resolve():
        raise ValueError("The cleanup recipe requires the prepared public Room scene")
    raw = o3d.io.read_triangle_mesh(str(source))
    before = topology(raw)
    for key in ("vertices", "triangles", "components", "boundary_edges", "nonmanifold_edges"):
        if before[key] != prior[key]:
            raise RuntimeError(f"Input {key} disagrees with the extraction record")

    start = time.perf_counter()
    mesh = o3d.geometry.TriangleMesh(raw)
    duplicates_before = remove_repeated_faces(mesh)
    labels, counts, areas = map(np.asarray, mesh.cluster_connected_triangles())
    area_limit = MAX_SPECK_AREA_VOXELS_SQUARED * prior["voxel_length"] ** 2
    specks = (counts < MIN_COMPONENT_FACES) & (areas < area_limit)
    mesh.remove_triangles_by_mask(specks[labels])
    mesh.remove_unreferenced_vertices()
    filter_seconds = time.perf_counter() - start
    filtered = topology(mesh)

    target = round(filtered["triangles"] * TRIANGLE_FRACTION)
    start = time.perf_counter()
    mesh = mesh.simplify_quadric_decimation(target, boundary_weight=BOUNDARY_WEIGHT)
    duplicates_after = remove_repeated_faces(mesh)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    simplify_seconds = time.perf_counter() - start
    result = output / "room_cleaned.ply"
    start = time.perf_counter()
    if not o3d.io.write_triangle_mesh(str(result), mesh):
        raise RuntimeError("Could not save the cleaned PLY")
    write_seconds = time.perf_counter() - start
    saved = o3d.io.read_triangle_mesh(str(result))
    after = topology(saved)
    if (after != topology(mesh) or after["nonmanifold_edges"] > before["nonmanifold_edges"]
            or not saved.has_vertex_colors() or not saved.has_vertex_normals()
            or sha256(source) != expected_hash):
        raise RuntimeError(f"Cleaned PLY verification failed: {after}")
    write_json(output / "cleanup.json", {
        "input": (source.relative_to(ROOT) if source.is_relative_to(ROOT) else source).as_posix(),
        "input_sha256": expected_hash,
        "output": result.name, "output_sha256": sha256(result),
        "open3d": o3d.__version__, "numpy": np.__version__,
        "coordinates": "Input axes and normalized scene units; no spatial transform",
        "definitions": {"components": "edge-connected triangles",
                        "boundary_edges": "one incident face", "nonmanifold_edges": "more than two incident faces"},
        "parameters": {
            "sequence": ["remove degenerate and duplicate face triples regardless of winding", "filter disconnected specks",
                         "compact vertices", "quadric decimation", "remove degenerate and duplicate face triples regardless of winding",
                         "compact vertices", "recompute vertex normals", "write binary PLY"],
            "duplicate_faces_removed_before_decimation": duplicates_before,
            "duplicate_faces_removed_after_decimation": duplicates_after,
            "speck_rule": "Both strict upper limits must hold",
            "speck_face_limit": MIN_COMPONENT_FACES, "speck_area_limit": area_limit,
            "removed_speck_components": int(specks.sum()),
            "removed_speck_triangles": int(counts[specks].sum()),
            "removed_speck_area": float(areas[specks].sum()),
            "quadric_target_fraction_of_filtered": TRIANGLE_FRACTION,
            "quadric_target_triangles": target, "quadric_boundary_weight": BOUNDARY_WEIGHT,
            "quadric_max_error": "Open3D default (unbounded)",
        },
        "seconds": {"filter": filter_seconds, "simplify": simplify_seconds, "ply_write": write_seconds,
                    "total_operations_and_write": filter_seconds + simplify_seconds + write_seconds},
        "before": before, "after_filter": filtered, "after_simplify_reloaded": after,
    })
    return raw, saved, prior


def raycast(mesh, rays):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    result = scene.cast_rays(rays)
    depth = result["t_hit"].numpy()
    hit = np.isfinite(depth)
    faces = np.asarray(mesh.triangles)[result["primitive_ids"].numpy()[hit]]
    uv = result["primitive_uvs"].numpy()[hit]
    weights = np.column_stack((1 - uv.sum(axis=1), uv))
    rgb = np.full((*depth.shape, 3), 22, dtype=np.uint8)
    rgb[hit] = np.clip(np.einsum("ij,ijk->ik", weights, np.asarray(mesh.vertex_colors)[faces]) * 255, 0, 255)
    return depth, rgb


def depth_image(depth, low, high):
    hit = np.isfinite(depth)
    scaled = np.clip((np.where(hit, depth, low) - low) / (high - low) * 255, 0, 255)
    colors = cv2.cvtColor(cv2.applyColorMap(scaled.astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    colors[~hit] = 22
    return colors


def compare(output, raw, cleaned, prior):
    sys.path.insert(0, str(ROOT / "examples"))
    from datasets.colmap import Dataset, Parser

    parser = Parser(str(DATA), factor=1, normalize=True, test_every=8)
    np.testing.assert_allclose(parser.transform, prior["parser_transform"], atol=1e-8)
    dataset = Dataset(parser, split="val")
    cameras = []
    panels = []
    for val_index, name, box in VIEWS:
        sample = dataset[val_index]
        photo = sample["image"].numpy().astype(np.uint8)
        source_height, source_width = photo.shape[:2]
        width, height = 2 * (source_width // 4), 2 * (source_height // 4)
        k = sample["K"].numpy().copy()
        k[0] *= width / source_width
        k[1] *= height / source_height
        c2w = sample["camtoworld"].numpy()
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(k, np.linalg.inv(c2w), width, height)
        raw_depth, raw_rgb = raycast(raw, rays)
        new_depth, new_rgb = raycast(cleaned, rays)
        photo = cv2.resize(photo, (width, height), interpolation=cv2.INTER_AREA)
        x0, y0, x1, y1 = (2 * value for value in box)
        crop = np.s_[y0:y1, x0:x1]
        d0, d1 = raw_depth[crop], new_depth[crop]
        h0, h1 = np.isfinite(d0), np.isfinite(d1)
        combined = h0 & h1
        loss, gain = h0 & ~h1, h1 & ~h0
        delta = np.abs(d1[combined] - d0[combined])
        low, high = np.percentile(d0[h0], (3, 97))
        panels.append((name, val_index, [photo[crop], raw_rgb[crop], new_rgb[crop],
                                        depth_image(d0, low, high), depth_image(d1, low, high)]))
        image_name = parser.image_names[int(dataset.indices[val_index])]
        cameras.append({"subject": name, "val_index": val_index, "image_name": image_name,
                        "width": width, "height": height, "K": k.tolist(), "camera_to_world_cv": c2w.tolist(),
                        "crop_xyxy": [x0, y0, x1, y1], "depth_color_range": [float(low), float(high)],
                        "crop_pixels": d0.size, "mesh_hit_lost_pixels": int(loss.sum()),
                        "mesh_hit_gained_pixels": int(gain.sum()),
                        "shared_hit_median_absolute_depth_delta": float(np.median(delta)),
                        "shared_hit_p95_absolute_depth_delta": float(np.percentile(delta, 95))})
    write_json(output / "silhouette_cameras.json", cameras)

    tile_width, tile_height, header = 292, 328, 56
    sheet = Image.new("RGB", (tile_width * 5, header + tile_height * len(panels)), (22, 24, 29))
    draw = ImageDraw.Draw(sheet)
    regular = ImageFont.load_default(size=16)
    bold = ImageFont.load_default(size=19)
    draw.text((12, 7), "Public Room | Week 11 silhouette check", font=bold, fill="white")
    for index, label in enumerate(("Held-out photo", "Raw vertex colors", "Cleaned vertex colors", "Raw mesh depth", "Cleaned mesh depth")):
        draw.text((index * tile_width + 12, 35), label, font=regular, fill="#d8dbe5")
    for index, (name, val_index, images) in enumerate(panels):
        y = header + index * tile_height
        draw.text((12, y + 8), f"{name}  |  validation view {val_index}", font=bold, fill="white")
        for column, array in enumerate(images):
            panel = Image.fromarray(array)
            panel.thumbnail((tile_width - 16, tile_height - 40), Image.Resampling.LANCZOS)
            sheet.paste(panel, (column * tile_width + (tile_width - panel.width) // 2, y + 35))
    sheet.save(output / "silhouette_comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE, help="Room extraction directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    compare(args.output, *cleanup(args.output, args.source.resolve()))


if __name__ == "__main__":
    main()
