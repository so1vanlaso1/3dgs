"""Fuse selected Stray Scanner views and score the mesh on reserved frames."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation

from iphone_rescan_alignment import NativeFrames, rgb_depth
from stray_quality_preflight import write_json
from stray_quality_visibility import carve_existing
from stray_scanner_reconstruct import columns, evaluate, read_csv


def make_grid(voxel, blocks):
    return o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3d.core.float32,) * 3,
        attr_channels=(1, 1, 3), voxel_size=voxel, block_resolution=8, block_count=blocks,
        device=o3d.core.Device("CPU:0"),
    )


def fuse(run, source, output, voxel, poses, carve):
    output.mkdir(parents=True, exist_ok=True)
    audit = run / "dataset_audit"
    preflight = json.loads((audit / "preflight.json").read_text())
    size = preflight["native_rgb_size"]
    images = NativeFrames(audit, size)
    ids = json.loads((run / "splits/frame_split.json").read_text())
    fit, heldout = np.array(ids["fit"]), np.array(ids["heldout"])
    if len(np.intersect1d(fit, ids["withheld"])) or not set(heldout) <= set(ids["withheld"]):
        raise ValueError("Training/fusion and held-out splits overlap")
    if images.ids != sorted(ids["fit"] + ids["heldout"]):
        raise ValueError("Native RGB cache does not match fusion and held-out views")
    odometry = read_csv(source / "odometry.csv")
    xyz = columns(odometry, "x", "y", "z")
    time = columns(odometry, "timestamp").ravel()
    orientation = Rotation.from_quat(columns(odometry, "qx", "qy", "qz", "qw")).as_matrix()
    if poses is not None:
        with np.load(poses) as corrected:
            xyz, orientation = corrected["positions_m"], corrected["camera_to_world_rotation"]
        if xyz.shape != (len(odometry), 3) or orientation.shape != (len(odometry), 3, 3):
            raise ValueError("Corrected poses do not cover the Stray Scanner capture")
    native_intrinsics = columns(odometry, "fx", "fy", "cx", "cy")
    intrinsics = native_intrinsics * ([256 / size[0], 192 / size[1]] * 2)
    camera = dict(positions_m=xyz, camera_to_world_rotation=orientation,
                  intrinsics_depth_px=intrinsics, intrinsics_native_px=native_intrinsics, timestamps_s=time, **ids)
    np.savez_compressed(output / "camera_split.npz", **camera)
    time = time - time[0]
    u, v = np.meshgrid(np.arange(256), np.arange(192))
    tile_blocks = 48
    halo = 2
    tile_span = tile_blocks * 8 * voxel
    locator = make_grid(voxel, 1024)
    fit_pixels, tiles = 0, defaultdict(list)
    for n, frame in enumerate(fit, 1):
        depth, confidence = rgb_depth(source, int(frame))
        valid = (confidence == 2) & (depth >= 250) & (depth <= 5000)
        fit_pixels += int(valid.sum())
        if valid.any():
            z = o3d.t.geometry.Image(np.where(valid, depth * np.float32(.001), np.float32(0)))
            fx, fy, cx, cy = intrinsics[frame]
            k = o3d.core.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=o3d.core.float64)
            view = np.eye(4)
            view[:3, :3] = orientation[frame].T
            view[:3, 3] = -orientation[frame].T @ xyz[frame]
            extrinsic = o3d.core.Tensor(view)
            coords = locator.compute_unique_block_coordinates(z, k, extrinsic, 1., 5.001, 4.).numpy()
            xz = coords[:, [0, 2]]
            lower = (xz.min(0) - halo) // tile_blocks
            upper = (xz.max(0) + halo) // tile_blocks
            for x in range(lower[0], upper[0] + 1):
                for zz in range(lower[1], upper[1] + 1):
                    start = np.array([x, zz]) * tile_blocks
                    inside = np.all((xz >= start - halo) & (xz < start + tile_blocks + halo), axis=1)
                    if inside.any():
                        tiles[x, zz].append((int(frame), coords[inside]))
        if n % 600 == 0 or n == len(fit):
            print(f"Indexed {n:,}/{len(fit):,} frames into {len(tiles)} overlapping fusion tiles.", flush=True)
    del locator
    fused = trimesh.Scene()
    mesh = o3d.geometry.TriangleMesh()
    tile_directory = output / "surface_tiles"
    tile_directory.mkdir(exist_ok=True)
    tile_files = []
    retained_area, removed_area, components, carved_blocks = 0., 0., 0, 0
    for index, (tile, views) in enumerate(sorted(tiles.items()), 1):
        volume = make_grid(voxel, 8192)
        for frame, coords in views:
            depth, confidence = rgb_depth(source, frame)
            valid = (confidence == 2) & (depth >= 250) & (depth <= 5000)
            z = o3d.t.geometry.Image(np.where(valid, depth * np.float32(.001), np.float32(0)))
            color = cv2.resize(images[frame], (256, 192), interpolation=cv2.INTER_AREA)
            color = o3d.t.geometry.Image(color.astype(np.float32) / 255)
            fx, fy, cx, cy = intrinsics[frame]
            k = o3d.core.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=o3d.core.float64)
            view = np.eye(4)
            view[:3, :3] = orientation[frame].T
            view[:3, 3] = -orientation[frame].T @ xyz[frame]
            volume.integrate(o3d.core.Tensor(coords), z, color, k, k, o3d.core.Tensor(view), 1., 5.001, 4.)
        block_count = volume.hashmap().size()
        if carve and block_count:
            carved_blocks += carve_existing(volume, source, fit[::9], xyz, orientation, intrinsics, voxel)
        piece = volume.extract_triangle_mesh(weight_threshold=2.5 if poses is not None else 1.5).to_legacy()
        del volume
        if piece.has_triangles():
            piece.remove_degenerate_triangles()
            piece.remove_duplicated_triangles()
            labels, counts, areas = map(np.asarray, piece.cluster_connected_triangles())
            keep = (counts >= 12) & (areas >= .0003)
            vertices = np.asarray(piece.vertices)
            face_vertices = vertices[np.asarray(piece.triangles)]
            centroid = face_vertices.mean(1)
            owner = np.floor(centroid[:, [0, 2]] / tile_span + 1e-7).astype(int)
            own = np.all(owner == tile, axis=1)
            piece.remove_triangles_by_mask(~(own & keep[labels]))
            piece.remove_unreferenced_vertices()
            if piece.has_triangles():
                piece.compute_vertex_normals()
                retained_area += piece.get_surface_area()
                components += int(len(np.unique(labels[own & keep[labels]])))
                removed_area += float(areas[~keep].sum())
                model = trimesh.Trimesh(vertices=piece.vertices, faces=piece.triangles,
                                        vertex_normals=piece.vertex_normals,
                                        vertex_colors=np.rint(np.clip(np.asarray(piece.vertex_colors), 0, 1) * 255).astype(np.uint8),
                                        process=False)
                name = f"tile_{tile[0]}_{tile[1]}"
                fused.add_geometry(model, geom_name=name)
                tile_file = tile_directory / f"{name}.ply"
                if not o3d.io.write_triangle_mesh(str(tile_file), piece, write_vertex_normals=False):
                    raise OSError(f"Could not write {tile_file}")
                tile_files.append(str(tile_file.relative_to(output)))
                mesh += piece
        print(f"Extracted tile {index}/{len(tiles)} at {tile}, {len(views):,} views, {block_count:,} blocks.", flush=True)
    if not mesh.has_triangles():
        raise ValueError("Fusion produced no supported surfaces")
    fused.export(output / "scene.glb")
    del fused
    write_json(output / "surface_tiles.json", tile_files)
    print("Scoring the mesh against held-out LiDAR.", flush=True)
    directions, depths, masks = {}, {}, {}
    for frame in heldout:
        depth, confidence = rgb_depth(source, int(frame))
        fx, fy, cx, cy = intrinsics[frame]
        depths[frame] = depth
        masks[frame] = (confidence == 2) & (depth >= 250) & (depth <= 5000)
        directions[frame] = np.stack(((u - cx) / fx, (v - cy) / fy, np.ones_like(u)), axis=-1)
    scores, view_rows, rendered = evaluate(mesh, heldout, depths, masks, directions, orientation, xyz, time, time // 6)
    with (output / "heldout.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(view_rows[0]))
        writer.writeheader()
        writer.writerows(view_rows)
    np.savez_compressed(output / "heldout_mesh_depth.npz", frames=heldout,
                        depth_m=np.stack([rendered[frame] for frame in heldout]))
    metrics = {"voxel_m": voxel, "sdf_truncation_m": 4 * voxel, "fusion_views": len(fit),
               "fit_depth_observations": fit_pixels, "tiles": len(tiles), "tile_core_m": tile_span, "tile_halo_blocks": halo,
               "vertices": len(mesh.vertices), "triangles": len(mesh.triangles), "mesh_area_m2": retained_area,
               "retained_tile_components": components, "removed_tile_area_m2_including_halos": removed_area,
               "pose_source": str(poses) if poses else "recorded ARKit", "carved_block_updates": carved_blocks, "heldout": scores}
    write_json(output / "metrics.json", metrics)
    print(json.dumps({key: value for key, value in metrics.items() if key != "heldout"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--voxel", type=float, default=.008)
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--carve", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    fuse(args.run, args.source, args.output, args.voxel, args.poses, args.carve)
