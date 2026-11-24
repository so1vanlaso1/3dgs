"""Replay the measured D455 depth baseline without changing archived poses."""

import argparse
from contextlib import closing
import csv
from itertools import groupby
from pathlib import Path
import sqlite3
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import open3d as o3d
from PIL import Image
import pycolmap
from scipy.spatial import cKDTree

import db3_local_geometry as db3
from db3_wall_diagnostic import plane_fit
from week12_mesh_review import depth_statistics, matte_sources, render_mesh, sensor_masks


ROOT = db3.ROOT
PILOT = db3.RUN / "matte/pilot"
WEEK12 = ROOT / "results/week12_comparison_20261119/matte_pair.json"
IMU = ROOT / "results/db3_imu_20261121"
RECORDING = ROOT / "20261030_134523.db3"
CAL = db3.INPUT / "calibration_clean.json"
TIMING = db3.R1 / "sensor_timing_pairs.csv"
VOXEL = .01
TRUNCATION = .04
SINGLE = "rgb_0006227.png"
PLANES = {"Plain wall": (100, 76, 159, 411), "Floor": (235, 394, 440, 450)}


def write_cloud(path, points, color):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(color.astype(float) / 255)
    assert o3d.io.write_point_cloud(str(path), cloud)
    return cloud


def world(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def blob(db, mid, stamp):
    actual, data = db.execute("SELECT timestamp,data FROM messages WHERE id=?", (mid,)).fetchone()
    assert actual == stamp
    return data


def frame(db, src, path, cal, pose, local):
    rgb_id = src["rgb_message_id"] if local else src["message_id"]
    depth_id = src["depth_message_id"]
    raw, _ = db3.clean.adapter.decode_depth(blob(db, depth_id, src["depth_timestamp_ns"]))
    assert db3.sha(path) == src["rgb_sha256"]
    rgb = np.asarray(Image.open(path).convert("RGB"))
    k = np.array(cal["K"])
    args = (raw, cal["depth_units"], cal["raw_depth"], cal["depth_to_color_R"],
            cal["depth_to_color_t_m"], k, cal["width"], cal["height"])
    uv, z, _, evidence = db3.projection.align_depth(*args, filtered=False)
    _, _, fraction, filtered = db3.projection.align_depth(*args, filtered=True)
    assert rgb.shape[:2] == evidence["depth_m"].shape
    # Preserve fractional projected pixels in the raw cloud. TSDF needs a raster.
    x, y = np.rint(uv).astype(int).T
    points = db3.backproject(uv, z, k)
    dz = filtered["depth_m"]
    ty, tx = np.nonzero(dz)
    return dict(name=src["name"], rgb_id=rgb_id, depth_id=depth_id, pose=pose, rgb=rgb,
                raw=raw, measured=evidence["depth_m"], filtered=dz, points=points, color=rgb[y, x],
                filtered_points=db3.backproject(np.c_[tx, ty], dz[ty, tx], k),
                filtered_color=rgb[ty, tx], raw_fill=float((raw > 0).mean()),
                native_filtered_fraction=fraction, collisions=evidence["discarded_collisions"])


def volume_for(frames, k):
    intrinsic = o3d.core.Tensor(k)
    f32 = o3d.core.Dtype.Float32
    volume = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"), attr_dtypes=(f32, f32, f32),
        attr_channels=((1,), (1,), (3,)), voxel_size=VOXEL, block_resolution=16,
        block_count=2048, device=o3d.core.Device("CPU:0"))
    for item in frames:
        depth = o3d.t.geometry.Image(item["filtered"])
        color = o3d.t.geometry.Image(item["rgb"].astype("f4") / 255)
        extrinsic = o3d.core.Tensor(np.linalg.inv(item["pose"]))
        blocks = volume.compute_unique_block_coordinates(depth, intrinsic, extrinsic, 1., 6., TRUNCATION/VOXEL)
        volume.integrate(blocks, depth, color, intrinsic, intrinsic, extrinsic, 1., 6., TRUNCATION/VOXEL)
    return volume


def mesh_stats(mesh):
    faces = np.asarray(mesh.triangles)
    assert len(faces)
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, uses = np.unique(edges, axis=0, return_counts=True)
    _, counts, areas = map(np.asarray, mesh.cluster_connected_triangles())
    vertices = np.asarray(mesh.vertices)
    return dict(vertices=len(vertices), triangles=len(faces), components=len(counts),
                area_m2=float(areas.sum()), largest_component_area_fraction=float(areas.max()/areas.sum()),
                boundary_edges=int((uses == 1).sum()), nonmanifold_edges=int((uses > 2).sum()),
                bounds_m=np.stack((vertices.min(axis=0), vertices.max(axis=0))).tolist())


def ray_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def fuse(frames, k, out, raw=True):
    start = time.perf_counter()
    out.mkdir(exist_ok=True)
    raw_count = sum(len(f["points"]) for f in frames)
    if raw:
        write_cloud(out / "multi_frame_raw.ply",
                    np.concatenate([world(f["points"], f["pose"]) for f in frames]),
                    np.concatenate([f["color"] for f in frames]))
    points = np.concatenate([world(f["filtered_points"], f["pose"]) for f in frames])
    colors = np.concatenate([f["filtered_color"] for f in frames])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(float) / 255)
    cloud = cloud.voxel_down_sample(VOXEL)
    assert o3d.io.write_point_cloud(str(out / "multi_frame_filtered.ply"), cloud)
    volume = volume_for(frames, k)
    mesh = volume.extract_triangle_mesh(weight_threshold=.5).to_legacy().compute_vertex_normals()
    np.clip(np.asarray(mesh.vertex_colors), 0, 1, out=np.asarray(mesh.vertex_colors))
    assert o3d.io.write_triangle_mesh(str(out / "direct_depth_mesh.ply"), mesh)
    result = dict(frames=[{key: f[key] for key in ("name", "rgb_id", "depth_id", "raw_fill",
                  "native_filtered_fraction", "collisions")} | {
                      "raw_aligned_points": len(f["points"]),
                      "filtered_aligned_points": len(f["filtered_points"]),
                      "color_camera_to_world": f["pose"].tolist()} for f in frames],
                  raw_points=raw_count, filtered_prevoxel=len(points), filtered_points=len(cloud.points),
                  mesh=mesh_stats(mesh), seconds=time.perf_counter() - start)
    return mesh, np.asarray(cloud.points), result


def single_frame(item, k, out):
    write_cloud(out / "single_frame.ply", item["points"], item["color"])
    z = item["measured"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), layout="constrained")
    axes[0].imshow(item["rgb"])
    axes[0].set_title("Rectified RGB with plane regions")
    artist = axes[1].imshow(np.ma.masked_equal(z, 0), vmin=.7, vmax=5)
    axes[1].set_title("Measured axial depth")
    fig.colorbar(artist, ax=axes[1], label="Meters")
    points, colors = item["points"][::25], item["color"][::25]
    axes[2].scatter(points[:, 0], points[:, 2], c=colors/255, s=.6)
    axes[2].set(title="Single-frame cloud from above", xlabel="Camera X (m)", ylabel="Camera Z (m)",
                aspect="equal")
    measures = {}
    uv = item["points"][:, :2]/item["points"][:, 2:] * [k[0, 0], k[1, 1]] + k[:2, 2]
    for label, (x0, y0, x1, y1) in PLANES.items():
        axes[0].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, edgecolor="yellow", fill=False))
        take = ((uv >= [x0, y0]) & (uv < [x1, y1])).all(axis=1)
        points = item["points"][take]
        center, normal = plane_fit(points)
        error = (points - center) @ normal
        measures[label] = dict(roi_xyxy=[x0, y0, x1, y1], range_m=float(np.median(points[:, 2])),
                               aligned_fill=float((z[y0:y1, x0:x1]>0).mean()),
                               observed_extent_m=np.ptp(points, axis=0).tolist(),
                               orthogonal_plane_rms_m=float(np.sqrt(np.mean(error**2))),
                               orthogonal_plane_p90_m=float(np.percentile(abs(error), 90)))
    for ax in axes[:2]:
        ax.axis("off")
    fig.savefig(out / "single_frame.png", dpi=170)
    plt.close(fig)
    return dict(name=item["name"], points=len(item["points"]), native_fill=item["raw_fill"],
                aligned_fill=float((z>0).mean()), planes=measures)


def compare_initialization(points, out):
    path = PILOT / "supported_initialization.npz"
    with np.load(path) as data:
        xyz, rgb = data["xyz"], data["rgb"]
    write_cloud(out / "gaussian_initialization.ply", xyz, rgb)
    result = {}
    for label, x, y in (("filtered_to_initializer", points, xyz), ("initializer_to_filtered", xyz, points)):
        dist, _ = cKDTree(y).query(x, workers=4)
        result[label] = dict(p50_p90_m=np.percentile(dist, [50, 90]).tolist(),
                             within_2cm=float((dist <= .02).mean()))
    return dict(initializer_points=len(xyz), initializer_sha256=db3.sha(path), distances=result)


def matched_uniform_mesh(training, pair, k, out):
    fields = ("parser_transform", "resolution", "box_origin", "box_length", "sdf_trunc", "scene_scale")
    left, right = (row["mesh"] for row in pair["arms"].values())
    assert all(left[key] == right[key] for key in fields)
    transform = np.array(left["parser_transform"])
    scale = np.linalg.norm(transform[:3, 0])
    volume = o3d.pipelines.integration.UniformTSDFVolume(
        length=left["box_length"], resolution=left["resolution"], origin=np.array(left["box_origin"]),
        sdf_trunc=left["sdf_trunc"], color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    h0, w0 = training[0]["filtered"].shape
    h, w = h0//4, w0//4
    small_k = k.copy()
    small_k[0] *= w/w0
    small_k[1] *= h/h0
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, small_k[0, 0], small_k[1, 1], small_k[0, 2], small_k[1, 2])
    y, x = np.ix_(np.rint(np.arange(h)*h0/h).astype(int), np.rint(np.arange(w)*w0/w).astype(int))
    for frame in training:
        depth = frame["filtered"][y, x] * np.float32(scale)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(frame["rgb"][y, x]), o3d.geometry.Image(depth),
            depth_scale=1., depth_trunc=8*left["scene_scale"], convert_rgb_to_intensity=False)
        pose = transform @ frame["pose"]
        pose[:3, :3] /= scale
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
    mesh = volume.extract_triangle_mesh().transform(np.linalg.inv(transform)).compute_vertex_normals()
    assert o3d.io.write_triangle_mesh(str(out / "direct_depth_mesh_matched_uniform.ply"), mesh)
    db3.write(out / "matched_uniform_protocol.json", {key: left[key] for key in fields} |
              dict(input_dimensions=[w, h], K=small_k.tolist(), same_training_frames=len(training),
                   output_units="Local metric world",
                   resized_depth="Nearest rectified pixel, preserving zeros; 0.3–6 m upstream range gate",
                   alpha_gate="Applies to Gaussian renders only"))
    return mesh


def compare_meshes(mesh, matched, pair, evaluations, k, out):
    meshes = {"Direct D455": mesh, "Direct matched grid": matched}
    for arm, row in pair["arms"].items():
        record = row["mesh"]
        path = ROOT / record["folder"] / "raw_mesh.ply"
        assert db3.sha(path) == record["mesh_sha256"]
        meshes[arm] = o3d.io.read_triangle_mesh(str(path)).transform(
            np.linalg.inv(np.array(record["parser_transform"])))
    scenes = {name: ray_scene(value) for name, value in meshes.items()}
    frames, errors = [], {name: [] for name in meshes}
    fig, axes = plt.subplots(len(evaluations), 6, figsize=(17, 5.5), layout="constrained")
    for index, item in enumerate(evaluations):
        z = item["measured"]
        _, stable = sensor_masks(z)
        # Open3D's convenience rays use pixel centers at x+.5, y+.5. The repo
        # deprojects at integer pixel centers, so use the matching rays below.
        y, x = np.indices(z.shape)
        directions = db3.backproject(np.c_[x.ravel(), y.ravel()], np.ones(z.size), k)
        pose = item["pose"]
        ray_array = np.concatenate((np.broadcast_to(pose[:3, 3], directions.shape),
                                    directions @ pose[:3, :3].T), axis=1)
        rays = o3d.core.Tensor(ray_array.reshape(*z.shape, 6).astype("f4"))
        panels = {name: render_mesh(value, scenes[name], rays) for name, value in meshes.items()}
        common = stable & np.logical_and.reduce([mask for _, mask, _ in panels.values()])
        row = dict(name=item["name"], stable_pixels=int(stable.sum()), common_pixels=int(common.sum()), variants={})
        axes[index, 0].imshow(item["rgb"])
        for col, (name, (depth, mask, color)) in enumerate(panels.items(), 1):
            errors[name].append(depth[common] - z[common])
            row["variants"][name] = dict(covered=int((stable & mask).sum()),
                                         common=depth_statistics(errors[name][-1]))
            axes[index, col].imshow(color)
        direct = panels["Direct D455"][0]
        rendered = axes[index, -1].imshow(np.where(common, (direct-z)*100, np.nan), cmap="RdBu_r", vmin=-12, vmax=12)
        for ax in axes[index]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[index, 0].set_ylabel(str(item["rgb_id"]))
        frames.append(row)
    for ax, name in zip(axes[0], ("Held-out RGB", *panels, "Direct minus sensor (cm)"), strict=True):
        ax.set_title(name)
    fig.colorbar(rendered, ax=axes[:, -1], shrink=.72, extend="both")
    fig.savefig(out / "heldout_comparison.png", dpi=175)
    plt.close(fig)
    total = sum(row["stable_pixels"] for row in frames)
    return dict(stable_pixels=total, common_pixels=sum(row["common_pixels"] for row in frames),
                frames=frames, aggregate={
                    name: dict(coverage=sum(row["variants"][name]["covered"] for row in frames)/total,
                               common=depth_statistics(np.concatenate(errors[name])), topology=mesh_stats(value))
                    for name, value in meshes.items()})


def alignment_check(cal, matte_model, item, db):
    k = np.array(cal["K"])
    probe = np.array([[-.32, -.19, .7], [.4, -.33, 1.], [.58, .32, 2.], [-.73, .42, 3.], [0., 0., 4.]])
    r, t = np.asarray(cal["depth_to_color_R"]), np.asarray(cal["depth_to_color_t_m"])
    raw_info = cal["raw_depth"]
    native_uv = db3.projection.project(probe, raw_info)
    restored = db3.projection.deproject(native_uv, probe[:, 2], raw_info)
    color = restored @ r.T + t
    rgb_uv = color @ k.T
    rect_uv = rgb_uv[:, :2]/rgb_uv[:, 2:]
    recovered = db3.backproject(rect_uv, color[:, 2], k)
    pose = item["pose"]
    wrong_range = db3.backproject(rect_uv, np.linalg.norm(color, axis=1), k)
    for camera in matte_model.cameras.values():
        assert camera.width == cal["width"] and camera.height == cal["height"]
        np.testing.assert_allclose(camera.params, [k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
    with TIMING.open(newline="", encoding="utf-8") as f:
        timing = list(csv.DictReader(f))
    selected = db3.read(db3.INPUT / "capture_manifest.json")["frames"]
    timing_ids = {int(row["rgb_message_id"]): int(row["depth_message_id"]) for row in timing}
    assert all(timing_ids[row["message_id"]] == row["depth_message_id"] for row in selected)
    sdk = db3.read(db3.R1 / "sdk_audit.json")
    topics = db.execute("SELECT id,name FROM topics ORDER BY id").fetchall()
    config = {}
    for tid, name in topics:
        if "/option/" in name.lower() and name.endswith("/value") and any(
                s in name.lower() for s in ("preset", "emit", "exposure", "laser")):
            config[name] = [db3.clean.capture.CDR(value).string() for (value,) in db.execute(
                "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,))]
    physical = db3.read(ROOT / "results/db3_wall_diagnostic_20261120/final/measurements.json")
    return dict(K=k.tolist(), dimensions=[cal["width"], cal["height"]], crop_xy=cal["crop_xy"],
                recorded_sensor_options=config, recorded_profiles={"rgb": "RGB8 1280x720 @ 30 Hz", "depth": "Z16 848x480 @ 30 Hz"},
                depth_scale_m=cal["depth_units"], depth_to_color_R=r.tolist(), depth_to_color_t_m=t.tolist(),
                native_z_round_trip_max_m=float(abs(restored-probe).max()),
                rectified_z_round_trip_max_m=float(abs(recovered-color).max()),
                world_inverse_round_trip_max_m=float(abs(world(world(color, pose), np.linalg.inv(pose))-color).max()),
                ray_distance_used_as_z_max_error_m=float(np.linalg.norm(wrong_range-color, axis=1).max()),
                sdk_crosscheck=sdk, matched_selected_pairs=len(selected), native_timing_rows=len(timing),
                timing_quantiles_ms={key: np.percentile([float(row[key]) for row in timing],
                                                        [0, 5, 50, 95, 100]).tolist()
                                     for key in ("signed_playback_depth_minus_rgb_ms",
                                                  "signed_exposure_midpoint_depth_minus_rgb_ms")},
                depth_exposure_us_range=[min(int(row["depth_exposure_us"]) for row in timing),
                                         max(int(row["depth_exposure_us"]) for row in timing)],
                opaque_wall_physical_edge_checks=[dict(frame=row["id"], edges=row["edges"])
                                                  for row in physical["frames"]],
                surveyed_reference_available=False)


def route_segments(db, cal, k, out):
    scores = db3.read(IMU / "route_evaluation/route_checks.json")["adjacent"]
    with np.load(IMU / "route_v2/full_route_camtoworlds.npz") as data:
        poses = data["A0"]
    manifest = db3.read(db3.INPUT / "capture_manifest.json")["frames"]
    passing = [row["boundary"] for row in scores if row["archived_pass"] and row["same_direct_fit_support"]
               and row["arms"]["A0"]["endpoint_pass"]]
    runs = []
    for _, group in groupby(enumerate(passing), lambda pair: pair[1]-pair[0]):
        indices = [value for _, value in group]
        if len(indices) >= 3:
            runs.append(list(range(indices[0], indices[-1]+2)))
    summaries = []
    for size in sorted({len(run) for run in runs}):
        indices = next(run for run in runs if len(run) == size)
        label = f"route_{indices[0]:03d}_{indices[-1]:03d}"
        frames = [frame(db, manifest[i], db3.INPUT / "images" / manifest[i]["name"], cal,
                        poses[i], False) for i in indices]
        folder = out / label
        mesh, _, summary = fuse(frames, k, folder)
        # Leave the central source out of a separate volume to quantify local prediction.
        middle = len(frames)//2
        holdback = frames[middle]
        volume = volume_for(frames[:middle] + frames[middle+1:], k)
        scene = ray_scene(volume.extract_triangle_mesh(weight_threshold=.5).to_legacy())
        z = holdback["filtered"]
        y, x = np.nonzero(z)
        view = db3.backproject(np.c_[x, y], np.ones(len(x)), k)
        pose = holdback["pose"]
        rays = np.c_[np.broadcast_to(pose[:3, 3], view.shape), view @ pose[:3, :3].T]
        pred = scene.cast_rays(o3d.core.Tensor(rays.astype("f4")))["t_hit"].numpy()
        keep = np.isfinite(pred) & (pred>0)
        summary["central_view_leave_one_out"] = dict(name=holdback["name"], coverage=float(keep.mean()),
                                                     errors=depth_statistics(pred[keep] - z[y[keep], x[keep]]))
        summary["indices"] = indices
        summary["directory"] = label
        db3.write(folder / "summary.json", summary)
        summaries.append(summary)
        print("ROUTE FUSION", label, summary["central_view_leave_one_out"], flush=True)
    return dict(runs_with_four_or_more_views=runs, demonstrated=summaries)


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    cal = db3.read(CAL)
    k = np.array(cal["K"])
    assert cal == db3.read(db3.DATA / "calibration_clean.json")
    pair = db3.read(WEEK12)
    matte_sources(pair)
    split = db3.read(PILOT / "dataset/experiment_split.json")
    assert pair["split"] == split and len(split["train"]) == 21 and len(split["val"]) == 3
    model = pycolmap.Reconstruction(PILOT / "dataset/sparse")
    poses = {im.name: np.linalg.inv(np.vstack((im.cam_from_world().matrix(), [0, 0, 0, 1])))
             for im in model.images.values()}
    manifest = {row["name"]: row for row in db3.read(db3.DATA / "manifest.json")["frames"]}
    frozen_paths = [Path(__file__), CAL, WEEK12, TIMING, PILOT / "dataset/experiment_split.json",
                    db3.DATA / "manifest.json", db3.INPUT / "capture_manifest.json",
                    IMU / "route_v2/full_route_camtoworlds.npz",
                    IMU / "route_evaluation/route_checks.json", db3.R1 / "sdk_audit.json"]
    frozen_paths.extend((PILOT / "dataset/sparse").glob("*.bin"))
    db3.write(out / "protocol.json", dict(inputs={str(p.relative_to(ROOT)): db3.sha(p) for p in frozen_paths},
              recording_sha256_from_manifest=db3.read(db3.INPUT / "capture_manifest.json")["source_sha256"],
              open3d=o3d.__version__, backend="CPU tensor VoxelBlockGrid", voxel_m=VOXEL, tsdf_truncation_m=TRUNCATION,
              range_m=[.3, 6.], volume_unit_resolution=16, mesh_weight_threshold=.5, split=split,
              raw="Native range-gated, rectified color projection with nearest-Z collision selection. No smoothing or filling.",
              filtered="Native 3x3 finite-range and max(30 mm, 2%) continuity gate, 3x3 median, z-buffer. Point cloud adds 1 cm voxel averaging.",
              expansion="Local A0 plus original R4 pass intersection, contiguous, earliest segment at each length >=4. No merge or pose fitting."))
    with closing(sqlite3.connect(RECORDING.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        training, validation = ([frame(db, manifest[name], db3.DATA / "images" / name,
                                       cal, poses[name], True) for name in split[role]] for role in ("train", "val"))
        assert {f["depth_id"] for f in training}.isdisjoint(f["depth_id"] for f in validation)
        # Score held-out against exactly the archived stable-depth reference used in Week 12.
        for f in validation:
            path = db3.DATA / "depth" / (f["name"] + ".npz")
            assert db3.sha(path) == manifest[f["name"]]["depth_sha256"]
            with np.load(path) as data:
                f["measured"] = data["depth_m"]
        selected = next(f for f in training if f["name"] == SINGLE)
        db3.write(out / "geometry_audit.json", alignment_check(cal, model, selected, db))
        one = single_frame(selected, k, out)
        mesh, points, summary = fuse(training, k, out)
        matched = matched_uniform_mesh(training, pair, k, out)
        summary.update(single_frame=one, initialization=compare_initialization(points, out),
                       heldout=compare_meshes(mesh, matched, pair, validation, k, out))
        db3.write(out / "matte_summary.json", summary)
        print("MATTE", summary["mesh"], flush=True)
        db3.write(out / "route_segments.json", route_segments(db, cal, k, out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output.resolve())
