"""Run G1 frozen matte checkpoint diagnostics and G2 paired uniform TSDF fusion."""

import argparse
from collections import defaultdict
from contextlib import closing
from pathlib import Path
import sqlite3

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

import db3_measured_depth as measured
from db3_evaluate_pilot import load_splats
from gsplat.losses import confidence_depth_l1_loss, ssim_loss
from gsplat.rendering import rasterization
from week12_mesh_review import Dataset, Parser, depth_statistics, matte_sources, sensor_masks


db3 = measured.db3
ROOT, PILOT, PAIR_PATH = measured.ROOT, measured.PILOT, measured.WEEK12
DEPTH_DIR = db3.DATA / "depth"
ARMS = ("B00", "A05_absgrad")
ALPHA = .7
DEPTH_LAMBDA = .01
SIZE = (208, 115)
REGION_COLORS = {"wall_or_panel": "yellow", "floor": "cyan", "doorway_boundary": "magenta"}

# Native rectified RGB coordinates picked from the source photographs before
# viewing model residuals. Wall/panel and floor boxes select interior patches.
WALL = [
    (65, 235, 235, 355), (30, 220, 220, 315), (30, 220, 220, 315),
    (51, 343, 192, 427), (20, 355, 182, 426), (628, 101, 672, 296),
    (525, 96, 580, 322), (474, 96, 526, 322), (348, 104, 402, 318),
    (194, 105, 250, 319), (175, 105, 215, 319), (205, 268, 405, 388),
    (151, 265, 407, 396), (130, 271, 368, 418), (132, 271, 366, 418),
    (81, 275, 342, 420), (45, 270, 279, 411), (51, 270, 216, 408),
    (15, 285, 175, 407), (17, 280, 148, 412), (119, 305, 302, 421),
    (250, 272, 400, 396), (184, 264, 353, 400), (30, 283, 163, 383),
]
FLOOR = {
    4823: (802, 339, 829, 450), 5317: (519, 339, 728, 444),
    5341: (518, 339, 729, 444), 5707: (350, 327, 539, 443),
    5863: (326, 337, 527, 442), 6227: (235, 335, 449, 437),
    6487: (155, 353, 400, 447), 6747: (40, 357, 315, 449),
    7007: (14, 358, 182, 442), 7397: (5, 361, 37, 441),
    7529: (4, 366, 24, 435),
}
# Endpoints of the visible jambs. Score a 25-pixel-wide band on each side
# together, including the sensor-valid edge samples omitted by the stable mask.
JAMBS = {
    4823: [(774, 20, 753, 445)],
    5317: [(504, 20, 498, 445), (778, 20, 747, 444)],
    5341: [(505, 20, 495, 445), (776, 20, 745, 444)],
    5707: [(344, 20, 329, 447), (591, 20, 550, 423)],
    5863: [(305, 20, 300, 447), (574, 20, 537, 428)],
    6227: [(220, 20, 207, 450), (486, 20, 462, 429)],
    6487: [(137, 20, 144, 444), (428, 20, 404, 423)],
    6747: [(339, 20, 329, 412)], 7007: [(204, 20, 198, 413)],
    7397: [(44, 20, 50, 450)], 7529: [(28, 20, 37, 432)],
}


def regions(names):
    assert len(names) == len(WALL)
    return {
        name: dict(wall_or_panel=box, floor=FLOOR.get(int(name[4:-4])),
                   doorway_boundary=JAMBS.get(int(name[4:-4]), []))
        for name, box in zip(names, WALL, strict=True)
    }


def roi_masks(annotation, shape):
    masks = {}
    for region in ("wall_or_panel", "floor"):
        box = annotation[region]
        mask = np.zeros(shape, dtype=bool)
        if box:
            x0, y0, x1, y1 = box
            mask[y0:y1, x0:x1] = True
        masks[region] = mask
    jamb = np.zeros(shape, dtype=np.uint8)
    for x0, y0, x1, y1 in annotation["doorway_boundary"]:
        cv2.line(jamb, (x0, y0), (x1, y1), 1, thickness=25)
    return masks | {"doorway_boundary": jamb.astype(bool)}


def render(splats, c2w, k, w, h, scale, *, moments=False, sh_degree=3):
    """Use the audited trainer's classic, packed CUDA renderer."""
    c2w = torch.as_tensor(c2w, device="cuda", dtype=torch.float32)[None]
    view = torch.linalg.inv_ex(c2w).inverse
    options = dict(viewmats=view, Ks=torch.as_tensor(k, device="cuda", dtype=torch.float32)[None],
                   width=w, height=h, near_plane=.01, far_plane=1e10, packed=True)
    geometry = (splats["means"], splats["quats"], splats["scales"].exp(), splats["opacities"].sigmoid())
    sh = torch.cat((splats["sh0"], splats["shN"]), dim=1)
    rgba_z, alpha, _ = rasterization(*geometry, sh, sh_degree=sh_degree, render_mode="RGB+ED", **options)
    spread = None
    if moments:
        with torch.no_grad():
            # Accumulate first and second axial depth moments with the same
            # compositing weights. A broad profile can also be a slanted surface.
            z = (geometry[0] @ view[0, 2, :3] + view[0, 2, 3]) / scale
            features = torch.stack((z, z.square(), torch.zeros_like(z)), dim=-1)
            raw, moment_alpha, _ = rasterization(*geometry, features, **options)
            torch.testing.assert_close(moment_alpha, alpha)
            mean = raw[..., 0] / alpha[..., 0].clamp_min(1e-10)
            support = alpha[..., 0] > .05
            torch.testing.assert_close(mean[support], rgba_z[..., 3][support] / scale,
                                       atol=1e-4, rtol=1e-5)
            variance = raw[..., 1] / alpha[..., 0].clamp_min(1e-10) - mean.square()
            spread = variance.clamp_min(0).sqrt()[0].cpu().numpy()
    return rgba_z, alpha, spread


def psnr(mse):
    return float(-10 * np.log10(max(float(mse), 1e-12)))


def summarize(delta, rgb_mae, rgb_mse):
    row = depth_statistics(delta)
    if row is not None:
        row.update(rgb_mae=float(rgb_mae.mean()), rgb_psnr_db=psnr(rgb_mse.mean()),
                   nearward_fraction=float((delta < 0).mean()))
    return row


def gradient_probe(splats, rgba_z, alpha, sample, scale, *, depth_lambda=DEPTH_LAMBDA,
                   training_scene_scale=None):
    """Differentiate both terms separately at the frozen checkpoint, without updates."""
    target = sample["image"][None].cuda().float() / 255
    h, w = target.shape[1:3]
    xy = sample["points"].cuda()
    grid = (xy / xy.new_tensor([w-1, h-1]) * 2 - 1)[None, :, None]
    sampled_depth = F.grid_sample(rgba_z[..., 3:].permute(0, 3, 1, 2), grid,
                                  align_corners=True).reshape(1, -1)
    sampled_alpha = F.grid_sample(alpha.permute(0, 3, 1, 2), grid,
                                  align_corners=True).reshape(1, -1)
    conf = sample["depth_confidence"][None].cuda()
    sensor = sample["depths"][None].cuda()
    supported = conf > 0
    color = rgba_z[..., :3]
    loss_rgb = torch.lerp(F.l1_loss(color, target),
                          ssim_loss(color.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2)), .2)
    loss_depth = confidence_depth_l1_loss(sampled_depth, sensor, conf, scene_scale=scale)
    # Verify the trained disparity expression in explicit inverse meters.
    valid = torch.isfinite(sampled_depth) & (sampled_depth > 0)
    safe_depth = torch.where(valid, sampled_depth.clamp_min(1e-8), 1)
    inv_pred_m = torch.where(valid, scale / safe_depth, 0)
    explicit = ((inv_pred_m - scale / sensor).abs() * conf).sum() / conf.sum()
    torch.testing.assert_close(loss_depth, explicit, atol=1e-7, rtol=1e-5)
    count = int(supported.sum())
    shared = ("means", "scales", "opacities", "quats")
    params = [splats[key] for key in shared]
    grad_rgb = torch.autograd.grad(loss_rgb, params, retain_graph=True)
    # The trainer scales inverse normalized Z by its camera-extent scale.
    # `scale` is the separate world-to-normalized conversion used for meters.
    loss_factor = 1.0 if training_scene_scale is None else training_scene_scale / scale
    weighted_depth = depth_lambda * loss_factor * loss_depth
    grad_depth = torch.autograd.grad(weighted_depth, params)
    gradients = {}
    for key, rgb, depth in zip(shared, grad_rgb, grad_depth, strict=True):
        assert torch.isfinite(rgb).all() and torch.isfinite(depth).all()
        rgb_norm, depth_norm = float(rgb.norm()), float(depth.norm())
        gradients[key] = dict(rgb_l2=rgb_norm, weighted_depth_l2=depth_norm,
                              depth_to_rgb_l2=depth_norm / rgb_norm if rgb_norm else None,
                              cosine=float((rgb * depth).sum()) / (rgb_norm * depth_norm)
                              if rgb_norm and depth_norm else None)
    take = supported & valid
    data = dict(positive_confidence=count, invalid_prediction=int((supported & ~valid).sum()),
                alpha_le_0_7=int((supported & (sampled_alpha <= ALPHA)).sum()),
                rgb_objective=float(loss_rgb.detach()), inverse_m_l1=float(loss_depth.detach()),
                weighted_depth=float(weighted_depth.detach()),
                gradients=gradients)
    residuals = ((sampled_depth - sensor) / scale).detach()[take].cpu().numpy()
    data["positive_confidence_valid_depth"] = depth_statistics(residuals)
    return data, residuals


def checkpoint_audit(frames, splats, out, scale):
    folder = out / "g1_frames"
    folder.mkdir()
    rows, pooled, sample_residuals = [], defaultdict(list), defaultdict(list)
    for frame in frames:
        name, sample, role = (frame[key] for key in ("name", "sample", "split"))
        photo, depth = frame["rgb"], frame["depth"]
        target = photo.astype(np.float32) / 255
        h, w = depth.shape
        valid, stable = sensor_masks(depth)
        k, c2w = frame["K"], frame["c2w"]
        predictions, gradients = {}, {}
        for arm, model in splats.items():
            with torch.set_grad_enabled(role == "train"):
                pixels, alpha, spread = render(model, c2w, k, w, h, scale, moments=True)
                if role == "train":
                    gradients[arm], residuals = gradient_probe(model, pixels, alpha, sample, scale)
                    sample_residuals[arm].append(residuals)
            color = pixels[0, ..., :3].detach().clamp(0, 1).cpu().numpy()
            residual = abs(color - target)
            predictions[arm] = dict(depth_m=pixels[0, ..., 3].detach().cpu().numpy() / scale,
                                    alpha=alpha[0, ..., 0].detach().cpu().numpy(),
                                    spread_m=spread, rgb=np.round(color * 255).astype(np.uint8),
                                    rgb_mae=residual.mean(axis=-1), rgb_mse=(residual**2).mean(axis=-1))
        eligible = {arm: np.isfinite(p["depth_m"]) & (p["depth_m"] > 0)
                    for arm, p in predictions.items()}
        paired = np.logical_and.reduce([eligible[arm] & (p["alpha"] > ALPHA)
                                       for arm, p in predictions.items()])
        masks = roi_masks(frame["regions"], depth.shape)
        wall, floor, jamb = (masks[key] for key in REGION_COLORS)
        groups = {
            "stable_alpha_gt_0.7_both": stable & paired,
            "wall_or_panel": stable & paired & wall & ~jamb,
            "floor": stable & paired & floor & ~jamb,
            "doorway_boundary": valid & paired & jamb,
            "doorway_boundary_stable": stable & paired & jamb,
            "range_0.3_to_1.5m": stable & paired & (depth < 1.5),
            "range_1.5_to_3m": stable & paired & (depth >= 1.5) & (depth < 3),
            "range_3_to_6m": stable & paired & (depth >= 3),
        }
        row = dict(name=name, split=role, sensor_valid=int(valid.sum()), stable=int(stable.sum()),
                   both_opaque=int((stable & paired).sum()), arms={})
        arrays = dict(target_rgb=photo, measured_m=depth, sensor_valid=valid, sensor_stable=stable,
                      paired_gaussian_support=paired, **masks)
        for arm, p in predictions.items():
            z, alpha, spread = (p[key] for key in ("depth_m", "alpha", "spread_m"))
            keep = stable & eligible[arm]
            opaque = keep & (alpha > ALPHA)
            view_groups = groups | {
                "alpha_0.05_to_0.7": keep & (alpha > .05) & (alpha <= ALPHA),
                "alpha_0.7_to_0.95": opaque & (alpha <= .95),
                "alpha_above_0.95": opaque & (alpha > .95),
                "spread_under_2cm": opaque & (spread < .02),
                "spread_2_to_10cm": opaque & (spread >= .02) & (spread < .10),
                "spread_at_least_10cm": opaque & (spread >= .10),
            }
            stats = {}
            delta = z - depth
            for group, mask in view_groups.items():
                values = np.stack((delta[mask], p["rgb_mae"][mask], p["rgb_mse"][mask]))
                stats[group] = summarize(*values)
                pooled[role, arm, group].append(values)
            row["arms"][arm] = dict(opaque_coverage_px=int(opaque.sum()),
                                     positive_depth_px=int(keep.sum()),
                                     rgb_psnr_db=psnr(p["rgb_mse"].mean()), groups=stats,
                                     training_gradient_probe=gradients.get(arm))
            arrays.update({f"{arm}_{key}": value for key, value in p.items()})
            arrays[f"{arm}_signed_m"] = np.where(valid & eligible[arm] & (alpha > .05), delta, np.nan)
        np.savez_compressed(folder / f"{name[:-4]}.npz", **arrays)
        rows.append(row)
        print("G1", role, name, row["both_opaque"], flush=True)

    summary = {role: {arm: {} for arm in ARMS} for role in ("train", "val")}
    for (role, arm, name), chunks in pooled.items():
        summary[role][arm][name] = summarize(*np.concatenate(chunks, axis=1))
    gradient_summary = {}
    for arm in ARMS:
        probes = [row["arms"][arm]["training_gradient_probe"] for row in rows if row["split"] == "train"]
        gradient_summary[arm] = dict(
            positive_confidence=sum(p["positive_confidence"] for p in probes),
            invalid_prediction=sum(p["invalid_prediction"] for p in probes),
            alpha_le_0_7=sum(p["alpha_le_0_7"] for p in probes),
            supervised_depth=depth_statistics(np.concatenate(sample_residuals[arm])),
            median_rgb_objective=float(np.median([p["rgb_objective"] for p in probes])),
            median_weighted_depth_objective=float(np.median([p["weighted_depth"] for p in probes])),
            parameter_gradients={key: {metric: float(np.median([
                p["gradients"][key][metric] for p in probes])) for metric in p_metrics}
                for key, p_metrics in probes[0]["gradients"].items()})
    db3.write(out / "g1_checkpoint_audit.json", dict(pooled=summary, frames=rows,
                                                      gradient_summary=gradient_summary))
    return rows, summary


def small_view(frame):
    w, h = SIZE
    h0, w0 = frame["depth"].shape
    k = frame["K"].copy()
    k[0] *= w/w0
    k[1] *= h/h0
    # Map the common low-resolution ray lattice back to the source raster.
    selection = np.ix_(np.rint(np.arange(h)*h0/h).astype(int), np.rint(np.arange(w)*w0/w).astype(int))
    render_k = k.copy()
    # CUDA integrates a pixel at (x+.5, y+.5); D455 and Open3D TSDF
    # address integer centers. This translation preserves the physical rays.
    render_k[:2, 2] += .5
    return k, render_k, selection


def volume(config):
    return o3d.pipelines.integration.UniformTSDFVolume(
        length=config["box_length"], resolution=config["resolution"],
        origin=np.array(config["box_origin"]), sdf_trunc=config["sdf_trunc"],
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)


def mesh_depth(scene, shape, k, c2w):
    y, x = np.indices(shape)
    directions = db3.backproject(np.c_[x.ravel(), y.ravel()], np.ones(x.size), k)
    rays = np.c_[np.broadcast_to(c2w[:3, 3], directions.shape), directions @ c2w[:3, :3].T]
    return scene.cast_rays(o3d.core.Tensor(rays.reshape(*shape, 6).astype("f4")))["t_hit"].numpy()


def evaluate_fusion(frames, scenes, splats, scale, out, *, low_res):
    rows, pooled = [], defaultdict(list)
    for frame in frames:
        k, c2w, depth = frame["K"], frame["c2w"], frame["depth"]
        _, stable = sensor_masks(depth)
        if low_res:
            k, render_k, ix = small_view(frame)
            depth, stable = depth[ix], stable[ix]
        predictions = {key: mesh_depth(scene, depth.shape, k, c2w) / scale
                       for key, scene in scenes.items()}
        eligible = {key: np.isfinite(z) & (z > 0) for key, z in predictions.items()}
        if low_res:
            for key, model in splats.items():
                rgba_z, alpha, _ = render(model, c2w, render_k, *SIZE, scale)
                z = rgba_z[0, ..., 3].cpu().numpy() / scale
                name = f"{key}_pre_fusion"
                predictions[name] = z
                eligible[name] = np.isfinite(z) & (z > 0) & (alpha[0, ..., 0].cpu().numpy() > ALPHA)
        common = stable & np.logical_and.reduce(list(eligible.values()))
        stats = {}
        for key, z in predictions.items():
            delta = z[common] - depth[common]
            stats[key] = dict(coverage_px=int((eligible[key] & stable).sum()),
                              error=depth_statistics(delta))
            pooled[key].append(delta)
        rows.append(dict(name=frame["name"], stable=int(stable.sum()), common=int(common.sum()), variants=stats))
        np.savez_compressed(out / f"heldout_{frame['name'][:-4]}_{'small' if low_res else 'native'}.npz",
                            sensor_m=depth, sensor_stable=stable, common=common,
                            **{f"{key}_depth_m": value for key, value in predictions.items()})
    total = sum(row["stable"] for row in rows)
    return dict(per_frame=rows, stable=total, common=sum(row["common"] for row in rows),
                pooled={key: dict(error=depth_statistics(np.concatenate(deltas)), coverage=sum(
                    row["variants"][key]["coverage_px"] for row in rows) / total)
                    for key, deltas in pooled.items()})


@torch.no_grad()
def matched_fusion(frames, splats, scale, config, out):
    folder = out / "g2_inputs"
    folder.mkdir()
    keys = ("D455", *ARMS)
    volumes = {key: volume(config) for key in keys}
    depth_trunc = 8 * config["scene_scale"]
    records, errors = [], defaultdict(list)
    with closing(sqlite3.connect(measured.RECORDING.as_uri() + "?mode=ro", uri=True)) as recording:
        recording.execute("PRAGMA query_only=ON")
        for frame in (f for f in frames if f["split"] == "train"):
            name = frame["name"]
            k, render_k, ix = small_view(frame)
            raw_depth, _ = db3.clean.adapter.decode_depth(measured.blob(
                recording, frame["source"]["depth_message_id"], frame["source"]["depth_timestamp_ns"]))
            cal = frame["calibration"]
            _, _, _, filtered = db3.projection.align_depth(
                raw_depth, cal["depth_units"], cal["raw_depth"], cal["depth_to_color_R"],
                cal["depth_to_color_t_m"], frame["K"], cal["width"], cal["height"], filtered=True)
            z = {"D455": filtered["depth_m"][ix] * np.float32(scale)}
            rgb, alphas = frame["rgb"][ix], {}
            c2w = frame["c2w"]
            for arm, model in splats.items():
                pixels, alpha, _ = render(model, c2w, render_k, *SIZE, scale)
                z[arm] = pixels[0, ..., 3].cpu().numpy()
                alphas[arm] = alpha[0, ..., 0].cpu().numpy()
            masks = {key: np.isfinite(d) & (d > 0) & (d < depth_trunc) for key, d in z.items()}
            for arm in ARMS:
                masks[arm] &= alphas[arm] > ALPHA
            common = np.logical_and.reduce(list(masks.values()))
            assert common.any()
            depths = {key: np.ascontiguousarray(np.where(common, d, 0), dtype="f4") for key, d in z.items()}
            intrinsic = o3d.camera.PinholeCameraIntrinsic(*SIZE, k[0, 0], k[1, 1], k[0, 2], k[1, 2])
            extrinsic = np.linalg.inv(c2w.astype(np.float64))
            color_image = o3d.geometry.Image(np.ascontiguousarray(rgb))
            for key, data in depths.items():
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_image, o3d.geometry.Image(data), depth_scale=1., depth_trunc=depth_trunc,
                    convert_rgb_to_intensity=False)
                assert np.array_equal(np.asarray(rgbd.depth) > 0, common)
                volumes[key].integrate(rgbd, intrinsic, extrinsic)
            np.savez_compressed(folder / f"{name[:-4]}.npz", K=k, K_rasterizer=render_k,
                                world_to_camera=extrinsic, rgb=rgb, common=common,
                                **{f"{key}_unmasked_z": d for key, d in z.items()},
                                **{f"{key}_integrated_z": d for key, d in depths.items()},
                                **{f"{key}_alpha": a for key, a in alphas.items()})
            deltas = {key: (z[key][common] - z["D455"][common]) / scale for key in ARMS}
            records.append(dict(name=name, source_depth_id=frame["source"]["depth_message_id"],
                                input_valid={key: int(mask.sum()) for key, mask in masks.items()},
                                common=int(common.sum()), pre_fusion={key: depth_statistics(d)
                                for key, d in deltas.items()}))
            for key, delta in deltas.items():
                errors[key].append(delta)
            print("G2 fused", name, int(common.sum()), flush=True)

    scenes, meshes = {}, {}
    transform = np.array(config["parser_transform"])
    for key, grid in volumes.items():
        mesh = grid.extract_triangle_mesh().compute_vertex_normals()
        ply = out / f"g2_{key}_normalized.ply"
        assert o3d.io.write_triangle_mesh(str(ply), mesh)
        reloaded = o3d.io.read_triangle_mesh(str(ply))
        vertices, faces = np.asarray(reloaded.vertices), np.asarray(reloaded.triangles)
        assert len(vertices) == len(mesh.vertices) and np.array_equal(faces, np.asarray(mesh.triangles))
        assert np.isfinite(vertices).all() and faces.min() >= 0 and faces.max() < len(vertices)
        scenes[key] = measured.ray_scene(reloaded)
        metric = reloaded.transform(np.linalg.inv(transform)).compute_vertex_normals()
        metric_path = out / f"g2_{key}_metric.ply"
        assert o3d.io.write_triangle_mesh(str(metric_path), metric)
        meshes[key] = dict(topology=measured.mesh_stats(metric), normalized_sha256=db3.sha(ply),
                           metric_sha256=db3.sha(metric_path))
    heldout = [f for f in frames if f["split"] == "val"]
    native = evaluate_fusion(heldout, scenes, splats, scale, out, low_res=False)
    small = evaluate_fusion(heldout, scenes, splats, scale, out, low_res=True)
    result = dict(integration=records, pre_fusion={key: depth_statistics(np.concatenate(deltas))
                  for key, deltas in errors.items()}, meshes=meshes, heldout_native=native,
                  heldout_small_with_pre_fusion=small)
    db3.write(out / "g2_common_mask_tsdf.json", result)
    return result


def figures(frames, g1, g2, out):
    contact = Image.new("RGB", (1251, 8 * 248), "#202020")
    labels = ImageDraw.Draw(contact)
    for i, frame in enumerate(sorted(frames, key=lambda f: f["name"])):
        x, y = i % 3 * 417, i // 3 * 248
        source = Image.fromarray(frame["rgb"])
        draw = ImageDraw.Draw(source)
        for region in ("wall_or_panel", "floor"):
            box = frame["regions"][region]
            if box:
                draw.rectangle(box, outline=REGION_COLORS[region], width=3)
        for line in frame["regions"]["doorway_boundary"]:
            draw.line(line, fill=REGION_COLORS["doorway_boundary"], width=5)
        contact.paste(source.resize((417, 231)), (x, y + 17))
        labels.text((x+4, y+2), f"{frame['name']}  {frame['split']}", fill="white")
    contact.save(out / "g1_rgb_regions.jpg", quality=92)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    group = "stable_alpha_gt_0.7_both"
    labels = ["B00", "A05-Abs"]
    x = np.arange(2)
    for offset, role, label in ((-.18, "train", "21 fitting views"), (.18, "val", "3 reserved views")):
        values = [g1[role][arm][group]["median_signed_m"] * 100 for arm in ARMS]
        axes[0].bar(x + offset, values, .35, label=label)
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Signed discrepancy (cm)",
                title="Expected depth before fusion")
    axes[0].legend()
    x = np.arange(3)
    for offset, stat, label in ((-.18, "median_abs_m", "Median"), (.18, "p90_abs_m", "90th percentile")):
        values = [g2["heldout_native"]["pooled"][key]["error"][stat] * 100 for key in ("D455", *ARMS)]
        axes[1].bar(x + offset, values, .35, label=label)
    axes[1].set(xticks=x, xticklabels=["D455", *labels], ylabel="Absolute discrepancy (cm)",
                title="Common-mask TSDF on reserved views")
    axes[1].legend()
    fig.savefig(out / "depth_comparison.png", dpi=165)
    plt.close(fig)


def run(out):
    pair = db3.read(PAIR_PATH)
    source_audit = matte_sources(pair)
    data = PILOT / "dataset"
    calibration = db3.read(measured.CAL)
    split = pair["split"]
    assert db3.read(data / "experiment_split.json") == split
    assert (len(split["train"]), len(split["val"])) == (21, 3)
    names = sorted(split["train"] + split["val"])
    annotations = regions(names)
    archived = db3.read(PAIR_PATH.parent / "matte_training_protocol.json")
    manifest_path = db3.DATA / "manifest.json"
    manifest = {row["name"]: row for row in db3.read(manifest_path)["frames"]}
    parser = Parser(str(data), factor=1, normalize=True, test_every=8)
    frames, hashes = [], {}
    for path in (Path(__file__), PAIR_PATH, manifest_path, measured.CAL):
        hashes[str(path.relative_to(ROOT))] = db3.sha(path)
    for relative, digest in archived["input_sha256"].items():
        assert db3.sha(ROOT / relative) == digest, relative
        hashes[relative] = digest
    source_drift = {}
    for relative, digest in archived["source_sha256"].items():
        actual = db3.sha(ROOT / relative)
        hashes[relative] = actual
        if actual != digest:
            source_drift[relative] = dict(archived=digest, current=actual)
    assert set(source_drift) <= {"scripts/week12_comparison.py", "examples/simple_trainer.py"}
    for path in (Path(measured.__file__), ROOT / "scripts/db3_evaluate_pilot.py",
                 ROOT / "scripts/week12_mesh_review.py", ROOT / "scripts/db3_local_geometry.py",
                 ROOT / "gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchFwd.cu"):
        hashes[str(path.relative_to(ROOT))] = db3.sha(path)
    for role in ("train", "val"):
        dataset = Dataset(parser, split=role, load_depths=(role == "train"),
                          sensor_depth_dir=str(data / "depth") if role == "train" else None)
        loader_names = [parser.image_names[index] for index in dataset.indices]
        assert set(loader_names) == set(split[role])
        for index, name in enumerate(loader_names):
            sample = dataset[index]
            assert "mask" not in sample
            source = manifest[name]
            path = DEPTH_DIR / f"{name}.npz"
            assert db3.sha(path) == source["depth_sha256"], name
            hashes[str(path.relative_to(ROOT))] = source["depth_sha256"]
            with np.load(path) as stored:
                depth = stored["depth_m"]
            rgb = sample["image"].numpy().astype(np.uint8)
            k = np.array(calibration["K"])
            np.testing.assert_allclose(sample["K"].numpy(), k, atol=1e-4)
            assert depth.shape == rgb.shape[:2] == (463, 834)
            frames.append(dict(name=name, split=role, sample=sample, depth=depth, rgb=rgb, K=k,
                               c2w=sample["camtoworld"].numpy(), source=source,
                               regions=annotations[name], calibration=calibration))
    assert {f["source"]["depth_message_id"] for f in frames if f["split"] == "train"}.isdisjoint(
        f["source"]["depth_message_id"] for f in frames if f["split"] == "val")

    config = pair["arms"]["B00"]["mesh"]
    scale = float(np.linalg.norm(parser.transform[:3, 0]))
    assert np.array_equal(parser.transform, config["parser_transform"])
    paths = {}
    for arm in ARMS:
        record = pair["arms"][arm]
        paths[arm] = Path(record["mesh"]["checkpoint"])
        digest = db3.sha(paths[arm])
        assert digest == record["checkpoint_sha256"]
        hashes[str(paths[arm].relative_to(ROOT))] = digest
        cfg = paths[arm].parents[1] / "cfg.yml"
        lines = set(cfg.read_text(encoding="utf-8").splitlines())
        assert {"depth_loss: true", "depth_confidence: true", "depth_lambda: 0.01",
                "ssim_lambda: 0.2", "near_plane: 0.01", "far_plane: 10000000000.0",
                "camera_model: pinhole", "packed: true", "antialiased: false",
                "post_processing: null", "sensor_alpha_lambda: 0.0", "pose_opt: false",
                "sh_degree: 3", "random_bkgd: false", "app_opt: false",
                "with_ut: false", "with_eval3d: false", "patch_size: null"} <= lines
        hashes[str(cfg.relative_to(ROOT))] = db3.sha(cfg)
        for key in ("parser_transform", "scene_scale", "resolution", "box_origin", "box_length", "sdf_trunc"):
            assert record["mesh"][key] == config[key]
    # Match the previous measured-depth baseline's camera order.
    frame_by_name = {frame["name"]: frame for frame in frames}
    frames = [frame_by_name[name] for name in split["train"] + split["val"]]
    out.mkdir(parents=True, exist_ok=False)
    db3.write(out / "protocol.json", dict(
        inputs=hashes, versions=dict(torch=torch.__version__, open3d=o3d.__version__,
                                     gpu=torch.cuda.get_device_name()),
        source_audit=source_audit, training_source_drift=source_drift,
        split=split, rgb_regions=annotations,
        g1="Native raster and trainer intrinsics. Joint stable D455 support and alpha > 0.7 in both models. "
           "Doorway bands also scored on valid sensor depth before 3x3 stability rejection. "
           "Alpha and weighted projected center-depth spread bins are arm-specific.",
        g2="One common measured/B00/A05-Abs mask per fitting frame, captured RGB in every arm. "
           "Native filtered sensor depth, nearest source pixel preserving zero. A +0.5 CUDA principal "
           "point translation represents the same integer-centered Open3D rays. "
           "Same camera matrices, mask, grid, depth cutoff and Open3D TSDF in every arm.",
        g2_grid={key: config[key] for key in ("parser_transform", "scene_scale", "resolution",
                 "box_origin", "box_length", "sdf_trunc")}, sensor_world_to_normalized_scale=scale,
        g2_size=SIZE, g2_inputs_units="Normalized camera Z; metric units after dividing by parser scale",
        camera_frames=[dict(name=f["name"], K=f["K"].tolist(), c2w_normalized=f["c2w"].tolist())
                       for f in frames], same_device_consistency_only=True))
    splats = {arm: load_splats(path) for arm, path in paths.items()}
    for model in splats.values():
        for key in ("means", "quats", "scales", "opacities"):
            model[key].requires_grad_(True)
    _, g1 = checkpoint_audit(frames, splats, out, scale)
    g2 = matched_fusion(frames, splats, scale, config, out)
    figures(frames, g1, g2, out)
    db3.write(out / "artifacts.json", {str(p.relative_to(out)): db3.sha(p)
              for p in sorted(out.rglob("*")) if p.is_file()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output.resolve())
