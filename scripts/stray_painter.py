"""Matched free-geometry and fixed-surface Gaussian training on the exhibit scan."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import cv2
from fused_ssim import fused_ssim
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
import torch
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import trimesh

from gsplat import rasterization
from stray_scanner_material_audit import REGIONS


WIDTH, HEIGHT = 512, 384
DEPTH_WIDTH, DEPTH_HEIGHT = 256, 192
ALPHA = 0.90
PREVIEWS = (99, 954, 1800, 2907)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor(array):
    return torch.as_tensor(array, device="cuda", dtype=torch.float32)


def decode(video, selected, total):
    rgb = {}
    frame_size = WIDTH * HEIGHT * 3
    with subprocess.Popen([
        "ffmpeg", "-v", "error", "-ignore_editlist", "1", "-noautorotate",
        "-i", str(video), "-map", "0:v:0", "-vf", f"scale={WIDTH}:{HEIGHT}:flags=area",
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ], stdout=subprocess.PIPE) as decoder:
        for index in range(total):
            data = decoder.stdout.read(frame_size)
            if len(data) != frame_size:
                raise ValueError(f"RGB ended at frame {index}")
            if index in selected:
                rgb[index] = np.frombuffer(data, np.uint8).reshape(HEIGHT, WIDTH, 3).copy()
        if decoder.stdout.read(1) or decoder.wait():
            raise ValueError("RGB stream does not match the pose log")
    return rgb


def split_audit(split, *, original=True):
    fit, heldout = split["fit"], split["heldout"]
    time_s = split["timestamps_s"] - split["timestamps_s"][0]
    xyz, rotation = split["positions_m"], split["camera_to_world_rotation"]
    phase = time_s % 4
    ids = np.arange(len(phase))
    reserved = (phase >= 1.45) & (phase < 2.55)
    if original and (len(fit) != 453 or len(heldout) != 53
            or not np.array_equal(split["withheld"], ids[reserved])
            or not np.array_equal(fit, ids[(ids % 6 == 0) & ~reserved])
            or not np.array_equal(heldout, ids[(ids % 9 == 0) & (phase >= 1.75) & (phase < 2.25)])):
        raise ValueError("Camera split differs from the exhibit TSDF construction rules")
    if np.intersect1d(fit, split["withheld"]).size or not np.isin(heldout, split["withheld"]).all():
        raise ValueError("Evaluation views must belong to disjoint reserved blocks")
    rows = []
    for frame in heldout:
        gap = np.abs(time_s[fit] - time_s[frame])
        distance = np.linalg.norm(xyz[fit] - xyz[frame], axis=-1)
        angle = np.degrees(Rotation.from_matrix(rotation[fit] @ rotation[frame].T).magnitude())
        same_direction = angle <= 10
        near = (distance < .25) & same_direction
        nearby = np.where(same_direction)[0]
        nearest_pose = int(nearby[distance[nearby].argmin()]) if len(nearby) else None
        nearest_time = int(gap.argmin())
        rows.append({
            "frame": int(frame), "seconds": float(time_s[frame]), "block": int(time_s[frame] // 4),
            "time_neighbor": int(fit[nearest_time]), "time_gap_s": float(gap[nearest_time]),
            "time_neighbor_distance_m": float(distance[nearest_time]),
            "time_neighbor_rotation_deg": float(angle[nearest_time]),
            "pose_neighbor_within_10deg": int(fit[nearest_pose]) if nearest_pose is not None else None,
            "pose_neighbor_distance_m": float(distance[nearest_pose]) if nearest_pose is not None else None,
            "pose_neighbor_rotation_deg": float(angle[nearest_pose]) if nearest_pose is not None else None,
            "fit_neighbors_within_25cm_and_10deg": int(near.sum()),
            "revisits_over_4s_within_25cm_and_10deg": int((near & (gap > 4)).sum()),
        })
    return rows


def initialize(mesh, count, rng):
    faces = rng.choice(len(mesh.faces), count, p=mesh.area_faces / mesh.area)
    bary = rng.random((count, 2))
    bary = np.where(bary.sum(axis=1, keepdims=True) > 1, 1 - bary, bary)
    bary = np.column_stack((1 - bary.sum(axis=1), bary))
    vertices = mesh.faces[faces]
    points = np.einsum("ni,nij->nj", bary, mesh.vertices[vertices])
    normals = mesh.face_normals[faces]
    reference = np.eye(3)[np.abs(normals).argmin(axis=-1)]
    tangent = np.cross(reference, normals)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    rotation = np.stack((tangent, np.cross(normals, tangent), normals), axis=-1)
    xyzw = Rotation.from_matrix(rotation).as_quat()
    rgb = np.einsum("ni,nij->nj", bary, mesh.visual.vertex_colors[vertices, :3]) / 255
    return {
        "points": points.astype(np.float32), "quats": xyzw[:, [3, 0, 1, 2]].astype(np.float32),
        "rgb": rgb.astype(np.float32), "faces": faces, "barycentric": bary,
    }


def camera_arrays(split, size=(WIDTH, HEIGHT)):
    rotations = split["camera_to_world_rotation"].transpose(0, 2, 1)
    views = np.repeat(np.eye(4)[None], len(rotations), axis=0)
    views[:, :3, :3] = rotations
    views[:, :3, 3] = -(rotations @ split["positions_m"][..., None])[..., 0]
    ks = np.repeat(np.eye(3)[None], len(rotations), axis=0)
    fx, fy, cx, cy = split["intrinsics_depth_px"].T
    ks[:, 0, 0], ks[:, 1, 1] = fx, fy
    # Integer depth pixel indices in the reconstruction correspond to half-pixel
    # centers in the Gaussian rasterizer. Scale calibrated centers with the raster.
    ks[:, 0, 2], ks[:, 1, 2] = cx + .5, cy + .5
    high_res = ks.copy()
    if "intrinsics_native_px" in split:
        fx, fy, cx, cy = split["intrinsics_native_px"].T
        high_res[:, 0, 0], high_res[:, 1, 1] = fx, fy
        high_res[:, 0, 2], high_res[:, 1, 2] = cx + .5, cy + .5
    else:
        high_res[:, 0] *= size[0] / DEPTH_WIDTH
        high_res[:, 1] *= size[1] / DEPTH_HEIGHT
    return tensor(views), tensor(high_res), tensor(ks)


def render(state, view, k, width, height, degree, depth=False):
    points, quats, scales, opacity, sh = state
    image, alpha, _ = rasterization(
        points, quats, scales, opacity, sh, view[None], k[None], width, height,
        sh_degree=degree, packed=False, near_plane=.05, far_plane=20.,
        render_mode="RGB+ED" if depth else "RGB",
    )
    return image, alpha


def train(arm, init, spacing, views, ks, images, sequence, out, *,
          size=(WIDTH, HEIGHT), checkpoint_steps=()):
    fixed = arm == "B1"
    centers = tensor(init["points"]).requires_grad_(not fixed)
    quats = tensor(init["quats"]).requires_grad_(not fixed)
    plane_scale = .65 * spacing
    normal_scale = min(.002, .15 * plane_scale)
    log_scales = tensor(np.tile(np.log([plane_scale, plane_scale, normal_scale]),
                               (len(centers), 1)))
    scale_parameters = log_scales[:, :2].contiguous() if fixed else log_scales
    scale_parameters.requires_grad_()
    opacity = torch.full((len(centers),), math.log(.3 / .7), device="cuda", requires_grad=True)
    dc = ((tensor(init["rgb"]) - .5) / .28209479177387814)[:, None].requires_grad_()
    rest = torch.zeros((len(centers), 15, 3), device="cuda", requires_grad=True)
    groups = [
        {"params": [scale_parameters], "lr": .005}, {"params": [opacity], "lr": .05},
        {"params": [dc], "lr": .0025}, {"params": [rest], "lr": .0025 / 20},
    ]
    if not fixed:
        groups.extend([{"params": [quats], "lr": .001}, {"params": [centers], "lr": .00016}])
    optimizer = torch.optim.Adam(groups, eps=1e-15, fused=True)

    def state():
        if fixed:
            in_plane = scale_parameters.clamp(math.log(.5 * plane_scale), math.log(2.5 * plane_scale))
            scales = torch.cat((in_plane, log_scales[:, 2:]), dim=1).exp()
        else:
            scales = scale_parameters.exp()
        return centers, quats, scales, opacity.sigmoid(), torch.cat((dc, rest), dim=1)

    losses = torch.empty(len(sequence), device="cuda")
    start = time.perf_counter()
    for step, frame in enumerate(sequence):
        degree = min(step // 1000, 3)
        if not fixed:
            optimizer.param_groups[-1]["lr"] = .00016 * .01 ** (step / max(1, len(sequence) - 1))
        optimizer.zero_grad(set_to_none=True)
        predicted, _ = render(state(), views[frame], ks[frame], *size, degree)
        target = images[frame][None].float() / 255
        l1 = (predicted - target).abs().mean()
        ssim = fused_ssim(predicted.permute(0, 3, 1, 2).contiguous(),
                          target.permute(0, 3, 1, 2).contiguous(), padding="valid")
        loss = .8 * l1 + .2 * (1 - ssim)
        loss.backward()
        optimizer.step()
        losses[step] = loss.detach()
        if step + 1 in checkpoint_steps:
            with torch.no_grad():
                save_checkpoint(state(), degree, out / f"{arm}_{step + 1:04d}.pt")
        if (step + 1) % 500 == 0 or step + 1 == len(sequence):
            print(f"{arm} {step + 1}/{len(sequence)} loss={loss.item():.5f} "
                  f"elapsed={time.perf_counter() - start:.1f}s", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    final = tuple(value.detach() for value in state())
    if not torch.isfinite(losses).all() or not all(torch.isfinite(value).all() for value in final):
        raise FloatingPointError(f"{arm} produced nonfinite training state")
    degree = min((len(sequence) - 1) // 1000, 3)
    save_checkpoint(final, degree, out / f"{arm}.pt")
    np.save(out / f"{arm}_loss.npy", losses.cpu().numpy())
    return final, elapsed, degree


def save_checkpoint(state, degree, path):
    keys = ("means", "quats", "scales", "opacities", "sh")
    torch.save(dict(zip(keys, (value.detach().cpu() for value in state))) | {"sh_degree": degree}, path)


def mesh_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(mesh.vertices.astype(np.float32)),
                        o3d.core.Tensor(mesh.faces.astype(np.uint32)))
    return scene


def references(scene, split, audits, source, *, regions=REGIONS, expected_pixels=1805668):
    u, v = np.meshgrid(np.arange(DEPTH_WIDTH), np.arange(DEPTH_HEIGHT))
    frames = {}
    for row in audits:
        frame = row["frame"]
        fx, fy, cx, cy = split["intrinsics_depth_px"][frame]
        directions = np.stack(((u - cx) / fx, (v - cy) / fy, np.ones_like(u)), axis=-1)
        origins = np.broadcast_to(split["positions_m"][frame], directions.shape)
        directions = directions @ split["camera_to_world_rotation"][frame].T
        rays = np.concatenate((origins, directions), axis=-1).astype(np.float32)
        mesh_z = scene.cast_rays(o3d.core.Tensor(rays), nthreads=4)["t_hit"].numpy()
        depth = cv2.imread(str(source / "depth" / f"{frame:06}.png"), cv2.IMREAD_UNCHANGED)
        conf = cv2.imread(str(source / "confidence" / f"{frame:06}.png"), cv2.IMREAD_UNCHANGED)
        eligible = (depth >= 250) & (depth <= 5000) & (conf == 2)
        opaque = np.zeros_like(eligible)
        for material, (x0, y0, x1, y1) in regions.get(frame, {}).items():
            if material in {"wood", "floor", "board", "cabinet"}:
                opaque[y0:y1, x0:x1] = True
        frames[frame] = (depth.astype(np.float32) * .001, eligible, mesh_z, opaque)
    if expected_pixels is not None and sum(eligible.sum() for _, eligible, _, _ in frames.values()) != expected_pixels:
        raise ValueError("The original held-out eligible depth pixel count has changed")
    return frames


def error_stats(errors):
    if not len(errors):
        return {"signed_median_mm": None, "abs_median_mm": None, "abs_p95_mm": None}
    return {"signed_median_mm": float(np.median(errors) * 1000),
            "abs_median_mm": float(np.median(abs(errors)) * 1000),
            "abs_p95_mm": float(np.percentile(abs(errors), 95) * 1000)}


@torch.inference_mode()
def evaluate(arm, state, degree, views, ks, depth_ks, rgb, refs, lpips, scene, init, out, *,
             size=(WIDTH, HEIGHT), preview_frames=PREVIEWS, png_compression=6):
    rows, maps, timings, panels = [], {}, [], {}
    frame_ids = list(refs)
    render_dir = out / arm
    render_dir.mkdir(exist_ok=True)
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in frame_ids]
    for frame in frame_ids[:5]:
        render(state, views[frame], ks[frame], *size, degree)
    for _ in range(3):
        torch.cuda.synchronize()
        for frame, (start, end) in zip(frame_ids, events):
            start.record()
            render(state, views[frame], ks[frame], *size, degree)
            end.record()
        torch.cuda.synchronize()
        timings.extend(start.elapsed_time(end) for start, end in events)
    for frame, (sensor, eligible, mesh_z, opaque) in refs.items():
        pred, a = render(state, views[frame], ks[frame], *size, degree)
        pred = pred.clamp(0, 1)
        target = tensor(rgb[frame])[None] / 255
        x, y = pred.permute(0, 3, 1, 2).contiguous(), target.permute(0, 3, 1, 2).contiguous()
        image = Image.fromarray((pred[0] * 255).round().byte().cpu().numpy())
        image.save(render_dir / f"{frame:06}.png", compress_level=png_compression)
        if frame in preview_frames:
            panels[frame] = image
        depth, alpha = render(state, views[frame], depth_ks[frame], DEPTH_WIDTH,
                              DEPTH_HEIGHT, degree, depth=True)
        z, alpha = depth[0, ..., 3].cpu().numpy(), alpha[0, ..., 0].cpu().numpy()
        valid = (alpha >= ALPHA) & np.isfinite(z) & (z > 0)
        score = eligible & valid
        errors = z[score] - sensor[score]
        maps[frame] = z, alpha
        rows.append({
            "frame": frame, "psnr_db": float(-10 * torch.log10((x - y).square().mean())),
            "ssim": float(structural_similarity_index_measure(x, y, data_range=1.)),
            "lpips_alex": float(lpips(x, y)), "mean_rgb_alpha": float(a.mean()),
            "rgb_alpha_ge_090": float((a >= ALPHA).float().mean()),
            "eligible_depth_pixels": int(eligible.sum()), "covered_depth_pixels": int(score.sum()),
            "within_50mm_pixels": int((abs(errors) <= .05).sum()), **error_stats(errors),
        })
    save_rows(out / f"{arm}_frames.csv", rows)
    xyz = state[0].cpu().numpy()
    timings = np.asarray(timings)
    summary = {key: float(np.mean([row[key] for row in rows])) for key in
               ("psnr_db", "ssim", "lpips_alex", "mean_rgb_alpha", "rgb_alpha_ge_090")}
    summary.update({
        "gaussians": len(xyz), "render_rgb_ms_median": float(np.median(timings)),
        "render_rgb_ms_p95": float(np.percentile(timings, 95)),
        "render_rgb_fps_from_mean": float(1000 / timings.mean()),
    })
    if scene is not None:
        distance = scene.compute_distance(o3d.core.Tensor(xyz), nthreads=4).numpy()
        summary["mesh_distance_p50_p95_max_mm"] = (np.percentile(distance, [50, 95, 100]) * 1000).tolist()
    if init is not None:
        displacement = np.linalg.norm(xyz - init["points"], axis=1)
        if arm == "B1" and (np.any(displacement != 0)
                            or not np.array_equal(state[1].cpu().numpy(), init["quats"])):
            raise ValueError("Fixed painters changed anchor positions or orientation")
        summary["anchor_displacement_p50_p95_max_mm"] = (
            np.percentile(displacement, [50, 95, 100]) * 1000
        ).tolist()
    return summary, maps, panels


def depth_comparison(refs, maps, out):
    records = {}
    rows = []
    for frame, (sensor, eligible, mesh_z, opaque) in refs.items():
        frame_maps = {arm: views[frame] for arm, views in maps.items()}
        coverage = {arm: (a >= ALPHA) & np.isfinite(z) & (z > 0)
                    for arm, (z, a) in frame_maps.items()}
        common = np.logical_and.reduce(list(coverage.values()))
        mesh_eligible = np.isfinite(mesh_z) & (mesh_z > 0)
        for arm, (z, _) in frame_maps.items():
            for label, reference, mask, covered in (
                ("sensor_own", sensor, eligible, coverage[arm]),
                ("sensor_common", sensor, eligible, common),
                ("opaque_mesh_own", mesh_z, mesh_eligible & eligible & opaque, coverage[arm]),
                ("opaque_mesh_common", mesh_z, mesh_eligible & eligible & opaque, common),
            ):
                hit = mask & covered
                errors, count = z[hit] - reference[hit], int(mask.sum())
                records.setdefault((arm, label), []).append((errors, count))
                if count:
                    rows.append({
                        "frame": frame, "arm": arm, "reference": label, "eligible": count,
                        "covered": len(errors), "within_50mm": int((abs(errors) <= .05).sum()),
                        **error_stats(errors),
                    })
    save_rows(out / "depth_frames.csv", rows)
    summary = {arm: {} for arm in maps}
    for (arm, label), values in records.items():
        errors = np.concatenate([e for e, _ in values])
        count = sum(n for _, n in values)
        summary[arm][label] = {
            "eligible": count, "covered": len(errors), "coverage": len(errors) / count if count else None,
            "within_50mm_fraction_of_eligible": float((abs(errors) <= .05).sum() / count) if count else None,
            **error_stats(errors),
        }
    return summary


def mesh_readback(refs):
    errors = []
    for sensor, mask, mesh_z, _ in refs.values():
        hits = mask & np.isfinite(mesh_z)
        errors.append(mesh_z[hits] - sensor[hits])
    return {"hits": sum(len(e) for e in errors), **error_stats(np.concatenate(errors))}


def preview(rgb, panels, out):
    header = 22
    sheet = Image.new("RGB", (WIDTH * 3, (HEIGHT + header) * len(PREVIEWS)), "#1d252b")
    draw = ImageDraw.Draw(sheet)
    for row, frame in enumerate(PREVIEWS):
        for col, (label, image) in enumerate((("Camera RGB", Image.fromarray(rgb[frame])),
                                              ("B0 Free geometry", panels["B0"][frame]),
                                              ("B1 Fixed surface", panels["B1"][frame]))):
            x, y = col * WIDTH, row * (HEIGHT + header)
            draw.text((x + 8, y + 4), f"{frame:06}   {label}", fill="white")
            sheet.paste(image, (x, y + header))
    sheet.save(out / "heldout_comparison.jpg", quality=94)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=7000)
    parser.add_argument("--gaussians", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    out, geometry = args.output, args.geometry
    out.mkdir(exist_ok=True, parents=True)
    if (out / "protocol.json").exists():
        raise FileExistsError(f"Choose an unused result directory: {out}")
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    source = geometry / "source/710d9d02a3"
    files = [geometry / key for key in ("scene.glb", "camera_split.npz", "registered_cloud.ply",
                                       "tsdf_10mm.ply", "metrics.json")]
    files.extend(source / key for key in ("rgb.mp4", "odometry.csv"))
    split = dict(np.load(geometry / "camera_split.npz"))
    audits = split_audit(split)
    save_rows(out / "split_audit.csv", audits)
    fit = split["fit"]
    sequence = np.concatenate([rng.permutation(fit) for _ in range(math.ceil(args.steps / len(fit)))])[:args.steps]
    np.save(out / "training_sequence.npy", sequence)
    mesh = trimesh.load(geometry / "scene.glb", force="mesh", process=False)
    if len(mesh.faces) != 1_092_631 or mesh.visual.kind != "vertex":
        raise ValueError("Unexpected geometry or vertex color data in scene.glb")
    init = initialize(mesh, args.gaussians, rng)
    np.savez_compressed(out / "initialization.npz", **init)
    spacing = math.sqrt(mesh.area / args.gaussians)
    for frame in split["heldout"]:
        files.extend(source / key / f"{frame:06}.png" for key in ("depth", "confidence"))
    input_hashes = {str(path): sha(path) for path in files}
    expected_hashes = {
        "scene.glb": "8fccb4f60a5bb923c38c9f65956a4c954dc03e7b0f4db10f3a00057274aa405a",
        "camera_split.npz": "b372224a1d235a42f913cd8ee894af2de0933087f8c1445b589a1d59ace13fef",
        "registered_cloud.ply": "5bd4950edf58619bff47a65060206bf9cb6a4c3d74eb76d4f3952ec788ef9733",
    }
    for name, expected in expected_hashes.items():
        if input_hashes[str(geometry / name)] != expected:
            raise ValueError(f"{name} differs from the original Experiment A output")
    protocol = {
        "steps": args.steps, "gaussians": args.gaussians, "seed": args.seed, "fit": fit.tolist(),
        "heldout": split["heldout"].tolist(), "rgb_size": [WIDTH, HEIGHT],
        "depth_size": [DEPTH_WIDTH, DEPTH_HEIGHT], "alpha_threshold": ALPHA,
        "spacing_m": spacing, "initial_tangent_scale_m": .65 * spacing,
        "initial_normal_scale_m": min(.002, .15 * .65 * spacing),
        "B1_tangent_scale_bounds_m": [.5 * .65 * spacing, 2.5 * .65 * spacing],
        "appearance_margins_B1_vs_B0": {"max_psnr_drop_db": 1., "max_ssim_drop": .02,
                                        "max_lpips_increase": .03},
        "objective": "0.8 L1 + 0.2 (1 - SSIM), RGB only; degree 0 to 3 by 1000 updates",
        "allocation": "identical area-weighted mesh samples and vertex colors; fixed count",
        "baseline": "free Gaussian centres, rotations and scales, matched initialization",
        "inputs_sha256": input_hashes,
        "initialization_sha256": sha(out / "initialization.npz"),
        "training_sequence_sha256": sha(out / "training_sequence.npy"),
        "script_sha256": sha(Path(__file__)),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "gpu_details": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version,memory.total",
                                                "--format=csv,noheader"], text=True).strip(),
    }
    save_json(out / "protocol.json", protocol)
    shutil.copy2(__file__, out / "source_snapshot.py")
    print("Frozen protocol. Decoding RGB and raycasting the existing GLB.", flush=True)
    rgb = decode(source / "rgb.mp4", set(fit) | set(split["heldout"]), len(split["timestamps_s"]))
    images = {int(i): torch.from_numpy(rgb[i]).to("cuda") for i in fit}
    views, ks, depth_ks = camera_arrays(split)
    scene = mesh_scene(mesh)
    refs = references(scene, split, audits, source)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda().eval()
    summaries, all_maps, panels, trained = {}, {}, {}, {}
    for arm in ("B0", "B1"):
        trained[arm] = train(arm, init, spacing, views, ks, images, sequence, out)
    for arm, (state, elapsed, degree) in trained.items():
        summaries[arm], all_maps[arm], panels[arm] = evaluate(
            arm, state, degree, views, ks, depth_ks, rgb, refs, lpips, scene, init, out)
        summaries[arm]["training_seconds"] = elapsed
        np.savez_compressed(out / f"{arm}_depth.npz", frames=split["heldout"],
                            z_m=np.stack([z for z, _ in all_maps[arm].values()]),
                            alpha=np.stack([a for _, a in all_maps[arm].values()]))
    save_json(out / "metrics.json", {"arms": summaries, "mesh_sensor_readback": mesh_readback(refs),
                                     "depth": depth_comparison(refs, all_maps, out)})
    preview(rgb, panels, out)
    print(f"Finished B0 and B1. Results in {out}.", flush=True)


if __name__ == "__main__":
    main()
