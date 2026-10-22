"""Create a known-camera development fixture and audit the existing official room.

Run from the repository root with .venv/Scripts/python.exe. All outputs remain
inside stage_result/stage_0. This script prepares data; it never trains a model.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from datasets.colmap import Parser, Dataset


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


SPHERES = [
    (np.array([-0.55, 0.0, 0.48]), 0.48, np.array([0.9, 0.22, 0.12])),
    (np.array([0.55, 0.2, 0.34]), 0.34, np.array([0.12, 0.5, 0.88])),
    (np.array([0.15, -0.65, 0.23]), 0.23, np.array([0.2, 0.8, 0.28])),
]


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def look_at(center):
    # OpenCV/COLMAP camera: x right, y down, z forward. World z is up.
    forward = unit(np.array([0.0, 0.0, 0.35]) - center)
    right = unit(np.cross(forward, np.array([0.0, 0.0, 1.0])))
    down = np.cross(forward, right)
    T = np.eye(4)
    T[:3, :3] = np.stack([right, down, forward], axis=1)
    T[:3, 3] = center
    return T


def render(T, K, size=128):
    # Four deterministic subpixel samples. Geometry is analytic, independent
    # of gsplat; no learned model or evaluation image informs construction.
    y, x = np.mgrid[:size, :size]
    accum = np.zeros((size, size, 3), dtype=np.float64)
    for ox, oy in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]:
        rays = unit(np.stack([(x + ox - K[0, 2]) / K[0, 0],
                              (y + oy - K[1, 2]) / K[1, 1], np.ones_like(x)], -1))
        rays = rays @ T[:3, :3].T
        origin = T[:3, 3]
        best = np.full((size, size), np.inf)
        rgb = np.broadcast_to([0.04, 0.06, 0.09], (size, size, 3)).copy()
        # Bounded checker floor, opaque surfaces and fixed world illumination.
        floor_t = -origin[2] / np.where(abs(rays[..., 2]) > 1e-12, rays[..., 2], 1e-12)
        hit = origin + floor_t[..., None] * rays
        floor = (floor_t > 0) & (abs(hit[..., 0]) < 1.8) & (abs(hit[..., 1]) < 1.8)
        checker = (np.floor(hit[..., 0] * 5) + np.floor(hit[..., 1] * 5)) % 2
        floor_rgb = np.where(checker[..., None] == 0, [0.7, 0.7, 0.63], [0.22, 0.25, 0.28])
        rgb[floor] = floor_rgb[floor]
        best[floor] = floor_t[floor]
        for center, radius, base in SPHERES:
            oc = origin - center
            b = rays @ oc
            disc = b * b - (oc @ oc - radius * radius)
            distance = -b - np.sqrt(np.maximum(disc, 0))
            valid = (disc >= 0) & (distance > 0) & (distance < best)
            p = origin + distance[..., None] * rays
            normal = (p - center) / radius
            shade = 0.35 + 0.65 * np.maximum(normal @ unit(np.array([-0.4, -0.6, 1.0])), 0)
            stripes = 0.75 + 0.25 * (np.sin(18 * p[..., 2] + 7 * p[..., 0]) > 0)
            sphere_rgb = base * shade[..., None] * stripes[..., None]
            rgb[valid] = sphere_rgb[valid]
            best[valid] = distance[valid]
        accum += rgb
    return np.rint(np.clip(accum / 4, 0, 1) * 255).astype(np.uint8)


def synthetic():
    dest = OUT / "data" / "analytic_spheres_v1"
    images, sparse = dest / "images", dest / "sparse" / "0"
    images.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)
    K = np.array([[115., 0., 64.], [0., 115., 64.], [0., 0., 1.]])
    rec = pycolmap.Reconstruction()
    rec.add_camera_with_trivial_rig(pycolmap.Camera(camera_id=1, model="PINHOLE",
        width=128, height=128, params=[115., 115., 64., 64.]))
    poses = []
    for i in range(48):
        theta = 2 * np.pi * i / 48
        center = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.5 + 0.25 * np.sin(2 * theta)])
        T = look_at(center)
        poses.append(T)
        name = f"view_{i:03d}.png"
        Image.fromarray(render(T, K)).save(images / name)
        W = np.linalg.inv(T)
        im = pycolmap.Image(name=name, camera_id=1, image_id=i + 1)
        rec.add_image_with_trivial_frame(im, pycolmap.Rigid3d(W[:3, :4]))
    # A fixed coarse surface prior; independent of train/dev/test images.
    # Empty tracks intentionally mean this is not a bundle-adjustment fixture.
    rng = np.random.default_rng(20261010)
    for center, radius, base in SPHERES:
        normals = unit(rng.normal(size=(256, 3)))
        for n in normals:
            rec.add_point3D(center + radius * n, pycolmap.Track(), np.rint(base * 255).astype(np.uint8))
    for xy in rng.uniform(-1.8, 1.8, (512, 2)):
        col = [179, 179, 161] if int(np.floor(xy[0] * 5) + np.floor(xy[1] * 5)) % 2 == 0 else [56, 64, 71]
        rec.add_point3D(np.r_[xy, 0.], pycolmap.Track(), np.array(col, dtype=np.uint8))
    rec.write(str(sparse))
    rec.write_text(str(sparse))
    np.savez(dest / "ground_truth_cameras.npz", camtoworlds=np.stack(poses), K=K)
    save_json(dest / "dataset_manifest.json", {
        "scene": "analytic_spheres_v1", "version": 1, "generator": "scripts/prepare_stage0_data.py",
        "generator_sha256": digest(Path(__file__)), "seed": 20261010,
        "images": 48, "width": 128, "height": 128, "points": 1280,
        "camera_convention": "OpenCV camera-to-world; world z up; camera x right, y down, z forward",
        "camera_truth": "analytic camera used to generate each image, not an SfM estimate",
        "raster": "4 subpixel ray samples, opaque spheres and bounded checker floor, fixed world lighting, uint8",
        "initialization": "fixed independent analytic surface samples; no image-based fitting; empty feature tracks",
        "limitation": "toy mechanism development fixture, not a public reconstruction benchmark; accurate surface prior is deliberately privileged",
    })
    # Only training views in the human-inspected preview.
    ids = [1, 9, 17, 25, 33, 41]
    sheet = Image.new("RGB", (3 * 128, 2 * 148), "white")
    draw = ImageDraw.Draw(sheet)
    for j, i in enumerate(ids):
        x, y = (j % 3) * 128, (j // 3) * 148
        with Image.open(images / f"view_{i:03d}.png") as im:
            sheet.paste(im, (x, y))
        draw.text((x + 5, y + 130), f"train view_{i:03d}", fill="black")
    sheet.save(OUT / "evidence" / "synthetic_train_preview.png")
    return dest


def audit_scene(scene, path):
    rec = pycolmap.Reconstruction(str(path / "sparse" / "0"))
    parser = Parser(str(path), factor=1, normalize=True, test_every=8)
    im_by_name = {im.name: im for im in rec.images.values()}
    rows = []
    cameras = []
    max_inverse_error = max_rot_error = max_det_error = 0.
    for i, name in enumerate(parser.image_names):
        role = "test" if i % 8 == 0 else "development" if i % 8 == 4 else "train"
        im = im_by_name[name]
        image_path = Path(parser.image_paths[i])
        with Image.open(image_path) as pixels:
            size = pixels.size
            pixels.verify()
        C = parser.camtoworlds[i]
        W = np.linalg.inv(C)
        max_inverse_error = max(max_inverse_error, float(np.max(abs(W @ C - np.eye(4)))))
        max_rot_error = max(max_rot_error, float(np.max(abs(C[:3, :3].T @ C[:3, :3] - np.eye(3)))))
        max_det_error = max(max_det_error, abs(float(np.linalg.det(C[:3, :3])) - 1))
        rows.append({"scene": scene, "view_id": f"{scene}:{name}", "sorted_index": i,
            "colmap_image_id": int(im.image_id), "camera_id": int(im.camera_id),
            "image_name": name, "split": role, "width": size[0], "height": size[1],
            "image_sha256": digest(image_path)})
        cameras.append({"view_id": f"{scene}:{name}", "sorted_index": i,
            "camtoworld_normalized": C.tolist(), "K": parser.Ks_dict[int(im.camera_id)].tolist()})
    base = OUT / "manifests"
    base.mkdir(exist_ok=True)
    with (base / f"{scene}_views.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    split = {role: [r["view_id"] for r in rows if r["split"] == role]
             for role in ["train", "development", "test"]}
    split_indices = {role: [r["sorted_index"] for r in rows if r["split"] == role]
                     for role in split}
    save_json(base / f"{scene}_splits.json", {
        "version": "stage0-v1", "declared_date_local": "2026-10-10", "timezone": "Asia/Saigon",
        "data_path_relative_to_repo": path.relative_to(ROOT).as_posix(),
        "rule": "lexicographic parser index % 8: test=0, development=4, train=all other residues",
        "counts": {k: len(v) for k, v in split.items()}, "view_ids": split,
        "parser_indices": split_indices, "final_test_policy": "locked prospectively; no Stage 0 model scoring",
        "previous_exposure": "none before this fixture generation" if scene == "analytic_spheres_v1" else
            "entire scene used in prior training; test=0 views evaluated previously; room is development-only for paper claims",
    })
    save_json(base / f"{scene}_cameras.json", {"normalization_transform": parser.transform.tolist(),
        "scene_scale": float(parser.scene_scale), "views": cameras})
    train = Dataset(parser, "train")
    train.indices = np.array(split_indices["train"])
    dev = Dataset(parser, "val")
    dev.indices = np.array(split_indices["development"])
    sample_train = train[0]
    sample_dev = dev[0]
    report = {"scene": scene, "registered_images": rec.num_reg_images(), "points": rec.num_points3D(),
        "splits": {k: len(v) for k, v in split.items()}, "all_images_decodable": True,
        "camera_models": sorted({str(c.model_name) for c in rec.cameras.values()}),
        "image_dimensions": sorted({(r["width"], r["height"]) for r in rows}),
        "max_inverse_identity_error": max_inverse_error, "max_rotation_orthogonality_error": max_rot_error,
        "max_rotation_determinant_error": max_det_error, "scene_scale_normalized": float(parser.scene_scale),
        "all_camera_values_finite": bool(np.isfinite(parser.camtoworlds).all()),
        "all_point_values_finite": bool(np.isfinite(parser.points).all()),
        "train_sample_shape": list(sample_train["image"].shape), "dev_sample_shape": list(sample_dev["image"].shape),
        "train_dev_loading_verified": True, "trainer_default_three_way_split": False,
        "source_files_sha256": {p.name: digest(p) for p in sorted((path / "sparse" / "0").glob("*.bin"))},
    }
    if scene == "analytic_spheres_v1":
        with np.load(path / "ground_truth_cameras.npz") as truth:
            raw_w2c = np.tile(np.eye(4), (len(parser.image_names), 1, 1))
            raw_w2c[:, :3, :4] = np.stack([im_by_name[n].cam_from_world().matrix()
                                         for n in parser.image_names])
            raw_poses = np.linalg.inv(raw_w2c)
            truth_error = float(np.max(abs(raw_poses - truth["camtoworlds"])))
        assert truth_error < 1e-10
        report["camera_truth_export_max_error"] = truth_error
    assert max_rot_error < 1e-10 and max_det_error < 1e-10
    assert all(set(split[a]).isdisjoint(split[b]) for a, b in [("train", "development"), ("train", "test"), ("development", "test")])
    return report


def room_source_check():
    prepared = pycolmap.Reconstruction(str(ROOT / "data/room_small/sparse/0"))
    source = pycolmap.Reconstruction(str(ROOT / "data/room_small/source/sparse/0"))
    pose_error = point_error = calibration_error = 0.
    for image_id, im in prepared.images.items():
        pose_error = max(pose_error, float(np.max(abs(im.cam_from_world().matrix() - source.images[image_id].cam_from_world().matrix()))))
    for pid, point in prepared.points3D.items():
        point_error = max(point_error, float(np.max(abs(point.xyz - source.points3D[pid].xyz))))
    for cid, cam in prepared.cameras.items():
        original = source.cameras[cid]
        expected = pycolmap.Camera(original.todict())
        expected.rescale(cam.width, cam.height)
        calibration_error = max(calibration_error, float(np.max(abs(expected.params - cam.params))))
    assert pose_error < 1e-10 and point_error < 1e-10 and calibration_error < 1e-10
    return {"source_manifest": json.loads((ROOT / "data/room_small/dataset_manifest.json").read_text()),
        "prepared_poses_max_difference_from_source": pose_error,
        "prepared_points_max_difference_from_source": point_error,
        "rescaled_intrinsics_max_difference": calibration_error,
        "source_provenance": "local official-download record and source/prepared equivalence; no fresh archive CRC download in Stage 0"}


if __name__ == "__main__":
    (OUT / "evidence").mkdir(parents=True, exist_ok=True)
    fixture = synthetic()
    report = {"date_local": "2026-10-10", "scenes": [audit_scene("analytic_spheres_v1", fixture),
        audit_scene("room_small", ROOT / "data/room_small")], "room_source_check": room_source_check()}
    save_json(OUT / "evidence" / "data_audit.json", report)
    print(json.dumps({r["scene"]: r["splits"] for r in report["scenes"]}, indent=2))
