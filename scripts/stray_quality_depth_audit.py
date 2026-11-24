"""Compare candidate camera poses using a matched fit-only depth reconstruction."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from iphone_rescan_alignment import rgb_depth
from stray_quality_preflight import write_json
from stray_quality_visibility import carve_existing
from stray_scanner_reconstruct import columns, evaluate, read_csv


def audit(run, source, output, candidate, voxel, carve):
    cfg = json.loads((run / "splits/evaluation_protocol.json").read_text())
    train = json.loads((run / "splits/train_frames.json").read_text())[::3]
    development = cfg["development"]
    rows = read_csv(source / "odometry.csv")
    time = columns(rows, "timestamp").ravel()
    time -= time[0]
    k = columns(rows, "fx", "fy", "cx", "cy") * (256 / 1920)
    xyz = columns(rows, "x", "y", "z")
    rotation = Rotation.from_quat(columns(rows, "qx", "qy", "qz", "qw")).as_matrix()
    if candidate:
        with np.load(candidate) as data:
            xyz, rotation = data["positions_m"], data["camera_to_world_rotation"]
    volume = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight"), attr_dtypes=(o3d.core.float32,) * 2, attr_channels=(1, 1),
        voxel_size=voxel, block_resolution=8, block_count=16384, device=o3d.core.Device("CPU:0"),
    )
    for frame in train:
        depth, confidence = rgb_depth(source, frame)
        mask = (confidence == 2) & (depth >= 250) & (depth <= 5000)
        if not mask.any():
            continue
        depth = o3d.t.geometry.Image(np.where(mask, depth * np.float32(.001), np.float32(0)))
        fx, fy, cx, cy = k[frame]
        calibration = o3d.core.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=o3d.core.float64)
        view = np.eye(4)
        view[:3, :3], view[:3, 3] = rotation[frame].T, -rotation[frame].T @ xyz[frame]
        view = o3d.core.Tensor(view)
        blocks = volume.compute_unique_block_coordinates(depth, calibration, view, 1., 5.001, 4.)
        volume.integrate(blocks, depth, calibration, view, 1., 5.001, 4.)
    free_space_blocks = carve_existing(volume, source, train[::3], xyz, rotation, k, voxel) if carve else 0
    mesh = volume.extract_triangle_mesh(weight_threshold=1.5).to_legacy()
    del volume
    x, y = np.meshgrid(np.arange(256), np.arange(192))
    depths, masks, dirs = {}, {}, {}
    for frame in development:
        depth, confidence = rgb_depth(source, frame)
        depths[frame] = depth
        masks[frame] = (confidence == 2) & (depth >= 250) & (depth <= 5000)
        fx, fy, cx, cy = k[frame]
        dirs[frame] = np.stack(((x - cx) / fx, (y - cy) / fy, np.ones_like(x)), axis=-1)
    report, _, _ = evaluate(mesh, development, depths, masks, dirs, rotation, xyz, time, time // 6)
    results = {"fit_views": len(train), "voxel_m": voxel, "triangles": len(mesh.triangles), "carved_blocks": free_space_blocks,
               "surface_area_m2": mesh.get_surface_area(), "scores": report}
    write_json(output, results)
    print(json.dumps({"abs_mm": report["absolute_mm"], "within_50mm": report["absolute_50mm_fraction_of_reference"],
                      "coverage": report["hit_fraction_of_reference"], "triangles": len(mesh.triangles)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--voxel", default=.024, type=float)
    parser.add_argument("--carve", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    audit(args.run, args.source, args.output, args.poses, args.voxel, args.carve)
