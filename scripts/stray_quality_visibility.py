"""Integrate conservative free-space evidence into already allocated TSDF blocks."""

import cv2
import numpy as np
import open3d as o3d

from iphone_rescan_alignment import rgb_depth


def carve_existing(volume, source, frames, xyz, orientation, intrinsics, voxel):
    hashmap = volume.hashmap()
    indices = hashmap.active_buf_indices().to(o3d.core.int64)
    keys = hashmap.key_tensor()[indices].numpy().copy()
    centers = (keys.astype(np.float32) + .5) * (8 * voxel)
    bounds = centers.min(0), centers.max(0)
    patch = np.ones((5, 5), np.uint8)
    clearance = max(.14, np.sqrt(3) * 4.5 * voxel + 4 * voxel)
    updated_blocks = 0
    for frame in frames:
        position, rotation = xyz[frame], orientation[frame]
        distance = np.maximum(np.maximum(bounds[0] - position, position - bounds[1]), 0.)
        if np.linalg.norm(distance) > 5:
            continue
        camera = (centers - position.astype(np.float32)) @ rotation.astype(np.float32)
        z = camera[:, 2]
        usable = np.flatnonzero((z > .18) & (z < 5 - clearance))
        fx, fy, cx, cy = intrinsics[frame]
        u = np.rint(camera[usable, 0] / z[usable] * fx + cx).astype(int)
        v = np.rint(camera[usable, 1] / z[usable] * fy + cy).astype(int)
        image = (u >= 3) & (u < 253) & (v >= 3) & (v < 189)
        usable, u, v = usable[image], u[image], v[image]
        if not len(usable):
            continue
        depth, confidence = rgb_depth(source, int(frame))
        measured_depth = np.where((confidence == 2) & (depth >= 250) & (depth <= 5000), depth, 0)
        local_minimum = cv2.erode(measured_depth, patch)[v, u] * .001
        chosen = usable[local_minimum - z[usable] >= clearance]
        if not len(chosen):
            continue
        calibration = o3d.core.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=o3d.core.float64)
        view = np.eye(4)
        view[:3, :3], view[:3, 3] = rotation.T, -rotation.T @ position
        z_image = o3d.t.geometry.Image(measured_depth.astype(np.float32) * .001)
        volume.integrate(o3d.core.Tensor(keys[chosen]), z_image, calibration, o3d.core.Tensor(view), 1., 5.001, 4.)
        updated_blocks += len(chosen)
    return updated_blocks
