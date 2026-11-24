"""Reconstruct and audit a Stray Scanner RGB-D capture in ARKit world metres."""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def percentiles(x):
    return dict(zip(("min", "p05", "median", "p95", "max"),
                    np.percentile(x, [0, 5, 50, 95, 100]).tolist())) if len(x) else None


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, skipinitialspace=True))


def columns(rows, *names):
    return np.array([[float(row[name]) for name in names] for row in rows])


def voxel_sums(points, colors, voxel):
    keys = np.floor(points / voxel).astype(np.int32)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    values = np.c_[points, colors, np.ones(len(points))]
    sums = np.column_stack([np.bincount(inverse, weights=col) for col in values.T])
    return unique, sums


def rays_for(frame, directions, poses, positions):
    world_dirs = directions[frame] @ poses[frame].T
    origins = np.broadcast_to(positions[frame], world_dirs.shape)
    return np.concatenate((origins, world_dirs), axis=-1).astype(np.float32)


def clean_mesh(mesh):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    labels, counts, areas = mesh.cluster_connected_triangles()
    counts, areas = np.asarray(counts), np.asarray(areas)
    keep = (counts >= 100) & (areas >= 0.0025)
    mesh.remove_triangles_by_mask(~keep[np.asarray(labels)])
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    stats = {
        "raw_components": len(counts), "retained_components": int(keep.sum()),
        "removed_area_m2": float(areas[~keep].sum()),
        "retained_area_m2": float(areas[keep].sum()),
        "largest_component_fraction_of_area": float(areas.max() / areas.sum()),
        "vertices": len(mesh.vertices), "triangles": len(mesh.triangles),
        "edge_manifold_allow_boundary": mesh.is_edge_manifold(allow_boundary_edges=True),
    }
    faces = np.asarray(mesh.triangles)
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]])), axis=1)
    _, use = np.unique(edges, axis=0, return_counts=True)
    stats.update(boundary_edges=int((use == 1).sum()), nonmanifold_edges=int((use > 2).sum()))
    return stats


def measure_planes(cloud, seed=41):
    """Report major plane fits and residuals within a 5 cm band of each plane."""
    o3d.utility.random.seed(seed)
    planes = []
    remainder = cloud
    for _ in range(6):
        if len(remainder.points) < 500:
            break
        model, indices = remainder.segment_plane(0.02, 3, 700, 0.999)
        normal = np.asarray(model[:3])
        candidates = np.asarray(remainder.points)
        points = candidates[indices]
        if len(points) < 500:
            break
        residual = candidates @ normal + model[3]
        # Limit the diagnostic to the inlier footprint with a small margin.
        _, _, basis = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
        footprint = points @ basis[:2].T
        coords = candidates @ basis[:2].T
        keep = np.all((coords >= footprint.min(0) - .03) & (coords <= footprint.max(0) + .03), axis=1)
        planes.append({
            "equation": np.asarray(model).tolist(), "inlier_points_at_20mm": len(indices),
            "inclination_from_horizontal_deg": float(np.degrees(np.arccos(np.clip(abs(normal[1]), 0, 1)))),
            "centroid_m": points.mean(axis=0).tolist(),
            "inlier_extent_m": np.ptp(footprint, axis=0).tolist(),
            "candidate_band_abs_error_mm": percentiles(np.abs(residual[keep & (np.abs(residual) <= .05)]) * 1000),
        })
        remainder = remainder.select_by_index(indices, invert=True)
    return planes


def evaluate(mesh, heldout, depths, valid, directions, rotations, positions, times, blocks):
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rows, all_errors, previews = [], [], {}
    for index in heldout:
        rays = rays_for(index, directions, rotations, positions)
        result = scene.cast_rays(o3d.core.Tensor(rays), nthreads=4)
        predicted = result["t_hit"].numpy()
        reference = depths[index].astype(np.float32) / 1000
        mask = valid[index]
        hits = mask & np.isfinite(predicted)
        signed = (predicted[hits] - reference[hits]) * 1000
        all_errors.append(signed)
        count = int(mask.sum())
        rows.append({
            "frame": int(index), "seconds": float(times[index]), "block": int(blocks[index]),
            "reference_pixels": count, "hit_pixels": len(signed),
            "absolute_50mm_pixels": int((np.abs(signed) <= 50).sum()),
            "absolute_100mm_pixels": int((np.abs(signed) <= 100).sum()),
            "coverage": len(signed) / count if count else None,
            "signed_median_mm": float(np.median(signed)) if len(signed) else None,
            "absolute_median_mm": float(np.median(np.abs(signed))) if len(signed) else None,
        })
        previews[index] = predicted
    signed = np.concatenate(all_errors)
    summary = {
        "frames": len(rows),
        "valid_reference_pixels": sum(row["reference_pixels"] for row in rows),
        "hit_pixels": len(signed),
        "signed_mm": percentiles(signed),
        "absolute_mm": percentiles(np.abs(signed)),
        "blocks": [],
    }
    for block in sorted(set(row["block"] for row in rows)):
        group = [r for r in rows if r["block"] == block]
        refs = sum(r["reference_pixels"] for r in group)
        if refs:
            summary["blocks"].append({
                "block": block, "seconds": group[len(group) // 2]["seconds"],
                "coverage": sum(r["hit_pixels"] for r in group) / refs,
                "within_50mm": sum(r["absolute_50mm_pixels"] for r in group) / refs,
            })
    total = summary["valid_reference_pixels"]
    for key in ("hit", "absolute_50mm", "absolute_100mm"):
        summary[key + "_fraction_of_reference"] = sum(row[key + "_pixels"] for row in rows) / total
    return summary, rows, previews


def make_previews(output, mesh, cloud, positions, heldout, depths, valid, rgb, cases, times):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    # Paint higher points first so room features below ceilings remain visible.
    order = np.argsort(points[:, 1])[::-1]
    axes[0].scatter(points[order, 0], -points[order, 2], c=colors[order], s=.55, rasterized=True)
    axes[0].plot(positions[:, 0], -positions[:, 2], color="#f3b32d", lw=1)
    axes[0].scatter(positions[heldout, 0], -positions[heldout, 2], c="#fa6a4b", s=9)
    order = np.argsort(points[:, 2])
    axes[1].scatter(points[order, 0], points[order, 1], c=colors[order], s=.55, rasterized=True)
    for axis, ylabel in zip(axes, ("−Z (m)", "Y (m)")):
        axis.set(xlabel="X (m)", ylabel=ylabel, aspect="equal", facecolor="#e8edf1")
    axes[0].set_title("Top view · scan path and held-out poses")
    axes[1].set_title("Elevation · registered LiDAR point cloud")
    fig.savefig(output / "geometry_preview.png", dpi=165)
    plt.close(fig)

    sample = heldout[np.linspace(0, len(heldout) - 1, 5, dtype=int)]
    fig, axes = plt.subplots(len(sample), 4, figsize=(12, 11), constrained_layout=True)
    for row, index in enumerate(sample):
        observed = np.where(valid[index], depths[index] / 1000., np.nan)
        rendered = np.where(np.isfinite(cases[index]), cases[index], np.nan)
        axes[row, 0].imshow(rgb[index])
        axes[row, 0].set_title(f"Frame {index} · {times[index]:.1f} s", fontsize=10)
        axes[row, 1].imshow(observed, vmin=.25, vmax=4, cmap="viridis")
        axes[row, 2].imshow(rendered, vmin=.25, vmax=4, cmap="viridis")
        image = axes[row, 3].imshow((rendered - observed) * 1000, vmin=-150, vmax=150, cmap="RdBu_r")
        for axis in axes[row]:
            axis.axis("off")
    for j, title in enumerate(("Recorded depth", "Mesh raycast depth", "Mesh minus recorded")):
        axes[0, j + 1].set_title(title, fontsize=10)
    fig.colorbar(image, ax=axes[:, 3], label="Signed depth error (mm)", shrink=.7)
    fig.savefig(output / "heldout_depth_preview.png", dpi=145)
    plt.close(fig)

    # Blender and glTF viewers can read this vertex-coloured mesh.
    import trimesh
    vertices = np.asarray(mesh.vertices).copy()
    # glTF and the native export here both use Y up.
    model = trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.triangles),
                            vertex_colors=np.rint(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8),
                            vertex_normals=np.asarray(mesh.vertex_normals), process=False)
    model.export(output / "scene.glb")
    render_overview(mesh, output / "mesh_preview.png")


def render_overview(mesh, output):
    """Render two orthographic views from above in the metric world frame."""
    from PIL import Image, ImageDraw

    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    points = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    low, high = np.quantile(points, [.015, .985], axis=0)
    center = (low + high) / 2
    span = np.linalg.norm(high - low)
    width, height = 760, 620
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(1, -1, height))
    sheet = Image.new("RGB", (2 * width, height + 36), "#16222d")
    title = ImageDraw.Draw(sheet)
    for panel, direction in enumerate(([.55, -.65, -.65], [-.65, -.7, .6])):
        forward = np.array(direction, dtype=float)
        forward /= np.linalg.norm(forward)
        across = np.cross(forward, [0., 1., 0.])
        across /= np.linalg.norm(across)
        up = np.cross(across, forward)
        projected_x = (points - center) @ across
        projected_y = (points - center) @ up
        scale = max(np.percentile(np.abs(projected_x), 99) * 1.15,
                    np.percentile(np.abs(projected_y), 99) * 1.15 * width / height)
        origin = (center - span * forward + x[..., None] * across * scale
                  + y[..., None] * up * scale * height / width)
        dirs = np.broadcast_to(forward, origin.shape)
        rays = o3d.core.Tensor(np.concatenate((origin, dirs), axis=-1).astype(np.float32))
        hits = scene.cast_rays(rays, nthreads=4)
        mask = np.isfinite(hits["t_hit"].numpy())
        uv = hits["primitive_uvs"].numpy()[mask]
        triangle = faces[hits["primitive_ids"].numpy()[mask].astype(int)]
        weights = np.c_[1 - uv.sum(1), uv]
        rgb = (colors[triangle] * weights[..., None]).sum(axis=1)
        normals = hits["primitive_normals"].numpy()[mask]
        normal_weight = np.abs(normals @ np.array([.28, .93, .22]))
        rgb *= (.62 + .38 * normal_weight)[:, None]
        canvas = np.broadcast_to([22, 34, 45], (height, width, 3)).astype(np.uint8).copy()
        canvas[mask] = np.rint(rgb.clip(0, 1) * 255).astype(np.uint8)
        sheet.paste(Image.fromarray(canvas), (panel * width, 36))
        title.text((18 + panel * width, 11), f"TSDF surface | View {panel + 1}", fill="#f3f6f9")
    sheet.save(output)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Extracted scan directory containing odometry.csv")
    parser.add_argument("output", type=Path)
    parser.add_argument("--video-pts", type=Path, required=True)
    parser.add_argument("--frame-split", type=Path, help="Predeclared fit, withheld and heldout frame IDs")
    parser.add_argument("--only-10mm", action="store_true")
    parser.add_argument("--rgb-frame-counter", action="store_true",
                        help="Validate RGB PTS against the exporter's 60 Hz frame counter")
    args = parser.parse_args()
    source, output = args.source, args.output
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    odometry = read_csv(source / "odometry.csv")
    ids = np.array([int(row["frame"]) for row in odometry])
    if not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError("Odometry frame indices must be contiguous and zero-based")
    timestamps = columns(odometry, "timestamp")[:, 0]
    times = timestamps - timestamps[0]
    pts = np.loadtxt(args.video_pts, delimiter=",", usecols=0)
    if np.any(np.diff(times) <= 0) or np.any(np.diff(pts) <= 0) or len(pts) != len(ids):
        raise ValueError("Odometry timestamps or RGB count do not match")
    clock_scale, clock_offset = np.polyfit(times, pts, 1)
    timing = (pts - clock_scale * times - clock_offset) * 1000
    counter_error = (pts - pts[0] - ids / 60) * 1000
    if args.rgb_frame_counter and np.max(np.abs(counter_error)) > .01:
        raise ValueError("RGB PTS does not match the 60 Hz frame counter")
    if not args.rgb_frame_counter and np.max(np.abs(timing)) > 8:
        raise ValueError("RGB timestamps differ from odometry by more than 8 ms")
    xyz = columns(odometry, "x", "y", "z")
    quat = columns(odometry, "qx", "qy", "qz", "qw")
    intrinsics = columns(odometry, "fx", "fy", "cx", "cy")
    norms = np.linalg.norm(quat, axis=1)
    if not np.isfinite(xyz).all() or not np.isfinite(intrinsics).all() or np.max(np.abs(norms - 1)) > 1e-3:
        raise ValueError("Invalid camera pose or intrinsics")
    rotations = Rotation.from_quat(quat)
    orientation = rotations.as_matrix()
    imu = columns(read_csv(source / "imu.csv"), "timestamp")[:, 0]
    video = source / "rgb.mp4"
    video_info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height", "-of", "json", str(video),
    ]))["streams"][0]
    rgb_width, rgb_height = video_info["width"], video_info["height"]

    expected = {f"{i:06}.png" for i in ids}
    for name in ("depth", "confidence"):
        if {p.name for p in (source / name).glob("*.png")} != expected:
            raise ValueError(f"{name} filenames do not match odometry indices")
    width, height = 256, 192
    intrinsics *= [width / rgb_width, height / rgb_height] * 2
    if np.any(intrinsics[:, :2] <= 0):
        raise ValueError("Invalid focal length")
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    blocks = (times // 4).astype(int)
    phase = times % 4
    if args.frame_split is None:
        withheld = (phase >= 1.45) & (phase < 2.55)
        fit = ids[(ids % 6 == 0) & ~withheld]
        heldout = ids[(phase >= 1.75) & (phase < 2.25) & (ids % 9 == 0)]
    else:
        frozen = json.loads(args.frame_split.read_text(encoding="utf-8"))
        fit, heldout = (np.unique(frozen[key]) for key in ("fit", "heldout"))
        reserved = np.array(frozen["withheld"])
        if (not len(fit) or not len(heldout) or not np.isin(np.r_[fit, heldout, reserved], ids).all()
                or not np.isin(heldout, reserved).all() or np.intersect1d(fit, reserved).size):
            raise ValueError("Invalid frozen fit, reserved or evaluation frame IDs")
        withheld = np.isin(ids, reserved)
    selected = set(fit) | set(heldout)
    fit_set = set(fit)
    np.savez_compressed(output / "camera_split.npz", fit=fit, withheld=ids[withheld], heldout=heldout,
                        timestamps_s=timestamps, positions_m=xyz, camera_to_world_rotation=orientation,
                        intrinsics_depth_px=intrinsics)
    setups = {"tsdf_10mm": (.01, .04)}
    if not args.only_10mm:
        setups["tsdf_20mm"] = (.02, .08)
    f32 = o3d.core.Dtype.Float32
    volumes = {
        name: o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"), attr_dtypes=(f32, f32, f32),
            attr_channels=((1,), (1,), (3,)), voxel_size=voxel, block_resolution=16,
            block_count=2048, device=o3d.core.Device("CPU:0"))
        for name, (voxel, _) in setups.items()
    }
    voxel_bins, colors, depths, masks, directions = {}, {}, {}, {}, {}
    confidence_counts = np.zeros(3, dtype=np.int64)
    accepted_counts = {}
    frame_bytes = width * height * 3
    decoder = subprocess.Popen([
        "ffmpeg", "-v", "error", "-ignore_editlist", "1", "-noautorotate",
        "-i", str(video), "-map", "0:v:0", "-vf", f"scale={width}:{height}:flags=area",
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ], stdout=subprocess.PIPE)
    for index in ids:
        image = decoder.stdout.read(frame_bytes)
        if len(image) != frame_bytes:
            raise ValueError(f"RGB decode failed at frame {index}")
        file = f"{index:06}.png"
        depth = cv2.imread(str(source / "depth" / file), cv2.IMREAD_UNCHANGED)
        confidence = cv2.imread(str(source / "confidence" / file), cv2.IMREAD_UNCHANGED)
        if (depth is None or confidence is None or depth.shape != (height, width)
                or confidence.shape != depth.shape or depth.dtype != np.uint16
                or confidence.dtype != np.uint8 or confidence.max() > 2):
            raise ValueError(f"Invalid depth/confidence at frame {index}")
        confidence_counts += np.bincount(confidence.ravel(), minlength=3)
        if index not in selected:
            continue
        color = np.frombuffer(image, dtype=np.uint8).reshape(height, width, 3)
        valid = (depth >= 250) & (depth <= 5000) & (confidence == 2)
        accepted_counts[int(index)] = int(valid.sum())
        fx, fy, cx, cy = intrinsics[index]
        direction = np.stack(((u - cx) / fx, (v - cy) / fy, np.ones_like(u)), axis=-1)
        if index not in fit_set:
            depths[index], colors[index], masks[index], directions[index] = depth, color, valid, direction
            continue
        depth_image = o3d.t.geometry.Image(np.where(valid, depth * np.float32(.001), np.float32(0)))
        color_image = o3d.t.geometry.Image(color.astype(np.float32) / 255)
        K = o3d.core.Tensor(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = orientation[index].T
        world_to_camera[:3, 3] = -orientation[index].T @ xyz[index]
        extrinsic = o3d.core.Tensor(world_to_camera)
        for name, volume in volumes.items():
            voxel, truncation = setups[name]
            margin = truncation / voxel
            coords = volume.compute_unique_block_coordinates(depth_image, K, extrinsic, 1., 5.001, margin)
            volume.integrate(coords, depth_image, color_image, K, K, extrinsic, 1., 5.001, margin)

        # Sample every second pixel and pool voxel sums in batches.
        sample = valid[::2, ::2]
        camera_points = direction[::2, ::2][sample] * (depth[::2, ::2][sample, None] / 1000.)
        world_points = camera_points @ orientation[index].T + xyz[index]
        batch = int(index) // 120
        voxel_bins.setdefault(batch, []).append((world_points, color[::2, ::2][sample]))
        if index % 300 == 0:
            print(f"Integrated through frame {index}/{len(ids)}", flush=True)
    if decoder.stdout.read(1):
        raise ValueError("RGB has more decodable frames than odometry")
    if decoder.wait():
        raise ValueError("FFmpeg RGB decode failed")
    print("Writing registered point cloud", flush=True)
    key_chunks, sum_chunks = [], []
    for chunk in voxel_bins.values():
        xyz_chunk, rgb_chunk = (np.concatenate(part) for part in zip(*chunk))
        keys, sums = voxel_sums(xyz_chunk, rgb_chunk, .01)
        key_chunks.append(keys)
        sum_chunks.append(sums)
    _, inverse = np.unique(np.concatenate(key_chunks), axis=0, return_inverse=True)
    sums = np.concatenate(sum_chunks)
    fused = np.column_stack([np.bincount(inverse, weights=col) for col in sums.T])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(fused[:, :3] / fused[:, 6:])
    cloud.colors = o3d.utility.Vector3dVector(fused[:, 3:6] / (255 * fused[:, 6:]))
    if not o3d.io.write_point_cloud(str(output / "registered_cloud.ply"), cloud):
        raise IOError("Could not write registered point cloud")
    print("Fitting diagnostic planes", flush=True)
    planes = measure_planes(cloud.voxel_down_sample(.025))

    report = {
        "rgb_frames": len(ids), "rgb_resolution": [rgb_width, rgb_height], "depth_resolution": [width, height],
        "duration_s": float(times[-1]), "rgb_minus_odometry_ms_after_clock_fit": percentiles(timing),
        "rgb_clock_scale": float(clock_scale), "rgb_clock_offset_s": float(clock_offset),
        "frame_delta_ms": percentiles(np.diff(timestamps) * 1000),
        "translation_step_mm": percentiles(np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1000),
        "rotation_step_deg": percentiles(np.degrees((rotations[:-1].inv() * rotations[1:]).magnitude())),
        "max_quaternion_norm_error": float(np.max(abs(norms - 1))),
        "camera_path_length_m": float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
        "camera_bounds_m": [xyz.min(0).tolist(), xyz.max(0).tolist()],
        "imu_rows": len(imu), "imu_span_relative_to_odometry_s": (imu[[0, -1]] - timestamps[0]).tolist(),
        "imu_nonpositive_steps": int((np.diff(imu) <= 0).sum()),
        "confidence_counts": confidence_counts.tolist(), "fit_frames": len(fit),
        "heldout_gap_frames": int(withheld.sum()), "heldout_frames": len(heldout),
        "min_gap_from_fit_s": float(np.abs(times[heldout, None] - times[fit]).min()),
        "valid_fit_pixels_per_frame": percentiles([accepted_counts[i] for i in fit]),
        "point_cloud": {"points": len(cloud.points), "voxel_m": .01,
                        "point_observations": int(fused[:, 6].sum()),
                        "single_observation_voxels": int((fused[:, 6] == 1).sum()),
                        "bounds_m": [cloud.get_min_bound().tolist(), cloud.get_max_bound().tolist()],
                        "planes": planes},
        "meshes": {},
    }
    if args.rgb_frame_counter:
        report.update(rgb_index_validation="60 Hz export frame counter",
                      rgb_counter_error_ms=percentiles(counter_error))
    for name, volume in volumes.items():
        print(f"Validating {name}", flush=True)
        mesh = volume.extract_triangle_mesh(weight_threshold=1.5).to_legacy()
        if not mesh.has_triangles():
            raise ValueError(f"{name} did not produce a surface")
        np.clip(np.asarray(mesh.vertex_colors), 0, 1, out=np.asarray(mesh.vertex_colors))
        stats = clean_mesh(mesh)
        stats["voxel_m"], stats["sdf_trunc_m"] = setups[name]
        if not o3d.io.write_triangle_mesh(str(output / f"{name}.ply"), mesh):
            raise IOError(f"Could not write {name}")
        summary, rows, previews = evaluate(mesh, heldout, depths, masks, directions, orientation, xyz, times, blocks)
        report["meshes"][name] = {"geometry": stats, "heldout": summary}
        with (output / f"{name}_heldout.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if name == "tsdf_10mm":
            make_previews(output, mesh, cloud, xyz, heldout, depths, masks, colors, previews, times)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Reconstruction complete", flush=True)


if __name__ == "__main__":
    main()




