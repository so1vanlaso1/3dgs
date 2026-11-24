"""Measure G3 checkpoint trajectories, gradient balance and final mesh agreement."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
from torchmetrics.functional.image import structural_similarity_index_measure
import yaml

from db3_g3_depth_sweep import ARCHIVE, CHECKPOINT_UPDATES, COEFFICIENTS, GATES, REPLAY_TOLERANCES
from db3_gaussian_depth_audit import (ALPHA, DEPTH_DIR, PAIR_PATH, PILOT, ROOT, Dataset, Parser, db3,
                                     depth_statistics, gradient_probe, load_splats, matte_sources,
                                     psnr, render, sensor_masks)
from week12_comparison import read, sha, write


ARMS = tuple(COEFFICIENTS)
BASELINE = ARMS[0]
REFERENCE = "Archived B00"
UPDATES = (*CHECKPOINT_UPDATES, 7000)
GRADIENT_PARAMS = ("means", "scales", "opacities", "quats")
DEPTH_KEYS = ("median_signed_m", "median_abs_m", "p90_abs_m")


def load_frames(pair):
    source_audit = matte_sources(pair)
    assert source_audit["initial_points"] == 90000
    folder = PILOT / "dataset"
    parser = Parser(str(folder), factor=1, normalize=True, test_every=8)
    manifest = {row["name"]: row for row in read(db3.DATA / "manifest.json")["frames"]}
    frames, hashes = {}, {}
    for split in ("train", "val"):
        dataset = Dataset(parser, split=split, load_depths=split == "train",
                          sensor_depth_dir=str(folder / "depth") if split == "train" else None)
        names = [parser.image_names[i] for i in dataset.indices]
        assert set(names) == set(pair["split"][split]) and len(names) == (21 if split == "train" else 3)
        frames[split] = []
        for index, name in enumerate(names):
            path = DEPTH_DIR / f"{name}.npz"
            hashes[str(path.relative_to(ROOT))] = sha(path)
            assert hashes[str(path.relative_to(ROOT))] == manifest[name]["depth_sha256"]
            with np.load(path) as archive:
                depth = archive["depth_m"]
            sample = dataset[index]
            assert "mask" not in sample and sample["image"].shape[:2] == depth.shape
            _, stable = sensor_masks(depth)
            frames[split].append(dict(name=name, sample=sample, depth=depth, stable=stable,
                                      K=sample["K"].numpy(), pose=sample["camtoworld"].numpy()))
    assert np.array_equal(parser.transform, pair["arms"]["B00"]["mesh"]["parser_transform"])
    scale = float(np.linalg.norm(parser.transform[:3, 0]))
    return frames, scale, source_audit, hashes


def model_maps(checkpoint, frame_sets, scale, degree, coefficient, *, training_scene_scale=None):
    model = load_splats(checkpoint)
    for key in GRADIENT_PARAMS:
        model[key].requires_grad_(coefficient is not None)
    predictions, gradients = {}, []
    for frame in frame_sets:
        height, width = frame["depth"].shape
        probe = coefficient is not None and "depth_confidence" in frame["sample"]
        with torch.set_grad_enabled(probe):
            rgba_z, alpha, _ = render(model, frame["pose"], frame["K"], width, height, scale,
                                      sh_degree=degree)
            if probe:
                data, _ = gradient_probe(model, rgba_z, alpha, frame["sample"], scale,
                                         depth_lambda=coefficient, training_scene_scale=training_scene_scale)
                gradients.append(dict(name=frame["name"], **data))
        with torch.no_grad():
            assert torch.isfinite(rgba_z).all() and torch.isfinite(alpha).all(), frame["name"]
            target = frame["sample"]["image"].cuda().float() / 255
            color = rgba_z[..., :3].clamp(0, 1)
            mse = (target - color[0]).square().mean().item()
            ssim = structural_similarity_index_measure(color.permute(0, 3, 1, 2),
                                                       target.permute(2, 0, 1)[None], data_range=1.).item()
            z = rgba_z[0, ..., 3].cpu().numpy() / scale
            a = alpha[0, ..., 0].cpu().numpy()
            predictions[frame["name"]] = dict(depth=z, valid=(z > 0) & (a > ALPHA),
                                               rgb=dict(psnr_db=psnr(mse), ssim=ssim), alpha=a,
                                               color=(color[0] * 255).byte().cpu().numpy())
    return predictions, dict(gaussians=len(model["means"]), frames=gradients)


def mesh_maps(path, frames, scale):
    mesh = o3d.io.read_triangle_mesh(str(path))
    triangles = np.asarray(mesh.triangles)
    assert len(triangles) and triangles.min() >= 0 and triangles.max() < len(mesh.vertices)
    assert np.isfinite(np.asarray(mesh.vertices)).all()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    predictions = {}
    for frame in frames:
        h, w = frame["depth"].shape
        rays = scene.create_rays_pinhole(frame["K"], np.linalg.inv(frame["pose"]), w, h)
        z = scene.cast_rays(rays)["t_hit"].numpy() / scale
        predictions[frame["name"]] = dict(depth=z, valid=np.isfinite(z) & (z > 0))
    return predictions


def compare(frames, maps):
    per_view = {}
    delta = {arm: [] for arm in maps}
    for frame in frames:
        name, stable = frame["name"], frame["stable"]
        masks = {arm: stable & maps[arm][name]["valid"] for arm in maps}
        shared = np.logical_and.reduce(list(masks.values()))
        per_view[name] = dict(stable_pixels=int(stable.sum()), shared_pixels=int(shared.sum()), arms={})
        for arm, mask in masks.items():
            prediction = maps[arm][name]
            error = prediction["depth"][shared] - frame["depth"][shared]
            delta[arm].append(error)
            per_view[name]["arms"][arm] = dict(covered_pixels=int(mask.sum()),
                coverage=float(mask.sum() / stable.sum()), error=depth_statistics(error),
                rgb=prediction.get("rgb"))
    total = sum(row["stable_pixels"] for row in per_view.values())
    pooled = {}
    for arm, errors in delta.items():
        rows = [row["arms"][arm] for row in per_view.values()]
        rgb = rows[0]["rgb"]
        pooled[arm] = dict(coverage=sum(row["covered_pixels"] for row in rows) / total,
                           error=depth_statistics(np.concatenate(errors)),
                           rgb={key: float(np.mean([row["rgb"][key] for row in rows]))
                                for key in rgb} if rgb is not None else None)
    return dict(per_view=per_view, pooled=pooled, stable_pixels=total,
                shared_pixels=sum(row["shared_pixels"] for row in per_view.values()))


def agreement(frames, maps):
    return dict(all=compare(frames, maps),
                paired={arm: compare(frames, {BASELINE: maps[BASELINE], arm: maps[arm]})
                        for arm in maps if arm != BASELINE})


def compare_rows(comparison):
    yield "pooled", comparison["pooled"]
    for name, view in comparison["per_view"].items():
        yield name, view["arms"]


def replay_check(reference, counts, baseline=BASELINE, archived=REFERENCE):
    limits = REPLAY_TOLERANCES
    failures = []
    for kind, comparison in reference.items():
        for view, rows in compare_rows(comparison):
            a, b = rows[baseline], rows[archived]
            if not a["error"] or any(100 * abs(a["error"][key] - b["error"][key]) >
                                    limits["max_depth_stat_difference_cm"] for key in DEPTH_KEYS):
                failures.append([kind, view, "depth"])
            if 100 * abs(a["coverage"] - b["coverage"]) > limits["max_coverage_difference_pp"]:
                failures.append([kind, view, "coverage"])
            if a["rgb"] and (abs(a["rgb"]["psnr_db"] - b["rgb"]["psnr_db"]) > limits["max_psnr_difference_db"] or
                             abs(a["rgb"]["ssim"] - b["rgb"]["ssim"]) > limits["max_ssim_difference"]):
                failures.append([kind, view, "RGB"])
    if abs(counts[baseline] / counts[archived] - 1) > limits["max_count_relative_difference"]:
        failures.append(["render", "global", "count"])
    return dict(passed=not failures, failures=failures)


def candidate_checks(rendered, meshes, replay, baseline=BASELINE):
    results = {}
    for arm in rendered:
        failures = []
        for kind, paired in (("render", rendered), ("mesh", meshes)):
            for view, rows in compare_rows(paired[arm]):
                a, b = rows[baseline], rows[arm]
                ea, eb = a["error"], b["error"]
                if ea is None or eb is None:
                    failures.append([kind, view, "no shared depth"])
                elif view == "pooled":
                    if 100 * (abs(ea["median_signed_m"]) - abs(eb["median_signed_m"])) < GATES["min_signed_bias_gain_cm"]:
                        failures.append([kind, view, "signed median"])
                    if 100 * (ea["median_abs_m"] - eb["median_abs_m"]) < GATES["min_absolute_median_gain_cm"]:
                        failures.append([kind, view, "absolute median"])
                    if 100 * (eb["p90_abs_m"] - ea["p90_abs_m"]) > GATES["max_absolute_p90_increase_cm"]:
                        failures.append([kind, view, "absolute p90"])
                elif eb["median_abs_m"] >= ea["median_abs_m"]:
                    failures.append([kind, view, "absolute median"])
                if 100 * (a["coverage"] - b["coverage"]) > GATES[f"max_{kind}_coverage_loss_pp"]:
                    failures.append([kind, view, "coverage"])
                if a["rgb"]:
                    for key, gate in (("psnr_db", "max_psnr_loss_db"), ("ssim", "max_ssim_loss")):
                        if a["rgb"][key] - b["rgb"][key] > GATES[gate]:
                            failures.append([kind, view, key])
        results[arm] = dict(passed_candidate_gates=not failures, eligible_for_followup=not failures and replay["passed"],
                            failures=failures)
    return results


def validate_run(out, pair):
    base = pair["arms"]["B00"]["mesh"]
    checkpoint = Path(base["checkpoint"])
    original_config = yaml.load((checkpoint.parents[1] / "cfg.yml").read_text(), Loader=yaml.BaseLoader)
    reference_mesh = ROOT / base["folder"] / "raw_mesh.ply"
    assert sha(checkpoint) == base["checkpoint_sha256"] and sha(reference_mesh) == base["mesh_sha256"]
    reports = {}
    for arm, coefficient in COEFFICIENTS.items():
        cfg = yaml.load((out / arm / "cfg.yml").read_text(), Loader=yaml.BaseLoader)
        assert float(cfg["depth_lambda"]) == coefficient and cfg["timed_eval_seconds"] == []
        ignored = {"depth_lambda", "result_dir", "save_steps", "timed_eval_seconds"}
        assert {key: value for key, value in cfg.items() if key not in ignored} == {
            key: value for key, value in original_config.items() if key not in ignored}, arm
        folder = out / "meshes" / arm
        report = read(folder / "report.json")
        for key in ("parser_transform", "scene_scale", "tsdf_backend", "resolution", "box_origin", "box_length",
                    "voxel_length", "sdf_trunc", "alpha_threshold"):
            assert report[key] == base[key], (arm, key)
        assert [(r["train_index"], r["image_id"]) for r in report["training_views"]] == [
            (r["train_index"], r["image_id"]) for r in base["training_views"]]
        assert sha(folder / "raw_mesh.ply") == report["mesh_sha256"]
        reports[arm] = report
    return checkpoint, reference_mesh, reports


def checkpoint(out, arm, updates):
    filename = "final_post_update.pt" if updates == 7000 else f"ckpt_{updates}_rank0.pt"
    return out / arm / "ckpts" / filename


def training_scalars(out, frames, probes):
    keys = ("loss", "l1loss", "ssimloss", "depthloss", "num_GS", "mem")
    history = {}
    for arm, coefficient in COEFFICIENTS.items():
        run = out / arm
        accumulator = EventAccumulator(str(run / "tb"), size_guidance={"scalars": 0})
        accumulator.Reload()
        history[arm] = {}
        for key in keys:
            events = accumulator.Scalars(f"train/{key}")
            assert [event.step for event in events] == list(range(0, 7000, 100))
            assert all(np.isfinite(event.value) for event in events), (arm, key)
            history[arm][key] = {str(event.step): event.value for event in events}
        initial_frame = frames[read(run / "experiment.json")["camera_sequence"][0]]["name"]
        probe = next(row for row in probes["0"][arm]["frames"] if row["name"] == initial_frame)
        np.testing.assert_allclose(probe["weighted_depth"] / coefficient, history[arm]["depthloss"]["0"],
                                   rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(probe["rgb_objective"] + probe["weighted_depth"], history[arm]["loss"]["0"],
                                   rtol=1e-4, atol=1e-6)
    return history


def figures(out, history, probes):
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.3), layout="constrained")
    fig.suptitle("G3 matte depth supervision on fixed views")
    for arm, coefficient in COEFFICIENTS.items():
        label = f"{arm} ({coefficient:.2f})"
        dev = [history[str(step)]["val"]["all"]["pooled"][arm] for step in UPDATES]
        axes[0, 0].plot(UPDATES, [100 * row["error"]["median_signed_m"] if row["error"] else np.nan
                                  for row in dev], ".-", label=label)
        axes[0, 1].plot(UPDATES, [100 * row["error"]["median_abs_m"] if row["error"] else np.nan
                                  for row in dev], ".-")
        axes[0, 2].plot(UPDATES, [100 * row["coverage"] for row in dev], ".-")
        axes[1, 0].plot(UPDATES, [row["rgb"]["psnr_db"] for row in dev], ".-")
        trajectory = [probes[str(step)][arm] for step in UPDATES]
        axes[1, 1].plot(UPDATES, [row["gaussians"] / 1000 for row in trajectory], ".-")
        axes[1, 2].plot(UPDATES, [100 * np.median([f["gradients"]["means"]["depth_to_rgb_l2"]
                                                  for f in row["frames"]]) for row in trajectory], ".-")
    labels = ("Signed depth median (cm)", "Absolute depth median (cm)", "Valid depth coverage (%)",
              "Full-frame PSNR (dB)", "Gaussians (thousands)", "Depth/RGB center gradient (%)")
    for axis, label in zip(axes.flat, labels, strict=True):
        axis.set(xlabel="Completed updates", ylabel=label)
        for boundary in (600, 3000, 6000):
            axis.axvline(boundary, color="0.82", zorder=0)
        axis.grid(alpha=.2)
    axes[0, 0].axhline(0, color="0.5", linewidth=.8)
    axes[0, 0].legend(frameon=False)
    axes[1, 2].set_yscale("log")
    fig.savefig(out / "g3_trajectory.png", dpi=160)
    plt.close(fig)


def evaluate(out):
    frozen = read(out / "protocol.json")
    assert frozen["gates"] == GATES and frozen["checkpoint_updates"] == list(UPDATES)
    for relative, digest in frozen["input_sha256"].items():
        assert sha(ROOT / relative) == digest, relative
    pair = read(PAIR_PATH)
    reference_checkpoint, reference_mesh, reports = validate_run(out, pair)
    frames, scale, source_audit, hashes = load_frames(pair)
    training_scene_scale = reports[BASELINE]["scene_scale"]
    folder = out / "evaluation"
    folder.mkdir(exist_ok=False)
    history, probes, event_checks, event_before = {}, {}, {}, {}
    for updates in UPDATES:
        key = str(updates)
        degree = min(3, max(0, updates - 1) // 1000)
        maps, probes[key] = {}, {}
        for arm, coefficient in COEFFICIENTS.items():
            path = checkpoint(out, arm, updates)
            hashes[str(path.relative_to(out))] = sha(path)
            maps[arm], probes[key][arm] = model_maps(path, frames["train"] + frames["val"],
                                                   scale, degree, coefficient,
                                                   training_scene_scale=training_scene_scale)
        history[key] = {split: agreement(views, maps) for split, views in frames.items()}
        if updates in (500, 501, 3000):
            event_before[updates] = maps
        for before, after in ((500, 501), (501, 1000), (3000, 3001)):
            if updates == after:
                previous = event_before.pop(before)
                event_checks[f"{before}-{after}"] = {
                    split: {arm: compare(views, {"before": previous[arm], "after": maps[arm]})
                            for arm in ARMS} for split, views in frames.items()}
        write(folder / "trajectory.json", history)
        write(folder / "gradients.json", probes)
        print(f"Scored {updates} completed updates", flush=True)
    dev = frames["val"]
    mesh = {arm: mesh_maps(out / "meshes" / arm / "raw_mesh.ply", dev, scale) for arm in ARMS}
    mesh_scores = agreement(dev, mesh)
    write(folder / "training_scalars.json", training_scalars(out, frames["train"], probes))
    archive_maps, archive_count = model_maps(reference_checkpoint, dev, scale, 3, None)
    reference = dict(render=compare(dev, {BASELINE: maps[BASELINE], REFERENCE: archive_maps}),
                     mesh=compare(dev, {BASELINE: mesh[BASELINE], REFERENCE: mesh_maps(reference_mesh, dev, scale)}))
    counts = {arm: probes["7000"][arm]["gaussians"] for arm in ARMS}
    counts[REFERENCE] = archive_count["gaussians"]
    replay = replay_check(reference, counts)
    candidates = candidate_checks(history["7000"]["val"]["paired"], mesh_scores["paired"], replay)
    for frame in dev:
        name = frame["name"]
        arrays = dict(sensor_m=frame["depth"], stable=frame["stable"],
                      target=frame["sample"]["image"].numpy().astype(np.uint8), K=frame["K"], pose=frame["pose"])
        for arm in ARMS:
            for kind, records in (("render", maps), ("mesh", mesh)):
                arrays.update({f"{arm}_{kind}_{key}": value for key, value in records[arm][name].items()
                               if key != "rgb"})
        np.savez_compressed(folder / f"{Path(name).stem}_final.npz", **arrays)
    write(folder / "summary.json", dict(final_counts=counts, render=history["7000"], mesh=mesh_scores,
           mesh_reports=reports, replay_reference=reference, replay=replay, candidates=candidates,
           trainer_rgb={arm: read(out / arm / "stats/val_step6999.json") for arm in ARMS},
           event_checks=event_checks, source_audit=source_audit, evaluation_input_sha256=hashes,
           gradients=dict(world_to_normalized_scale=scale, training_scene_scale=training_scene_scale,
                          evaluation_sh_degree={str(step): min(3, max(0, step - 1) // 1000) for step in UPDATES}),
           evaluation_source_sha256={str(path.relative_to(ROOT)): sha(path) for path in (
               Path(__file__), ROOT / "scripts/db3_gaussian_depth_audit.py", ROOT / "scripts/week12_mesh_review.py")},
           versions=dict(torch=str(torch.__version__), open3d=o3d.__version__, gpu=torch.cuda.get_device_name())))
    figures(folder, history, probes)
    print(f"Replay {replay}; candidate gates {candidates}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    evaluate(parser.parse_args().output.resolve())
