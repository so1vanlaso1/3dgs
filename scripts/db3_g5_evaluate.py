"""Score matched G5 endpoints and opacity-reset recovery on locked observations."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch

from datasets.normalize import transform_cameras
from db3_g3_evaluate import PAIR_PATH, candidate_checks, compare, compare_rows, load_frames, model_maps
from db3_g4_evaluate import config, endpoint
from db3_g5 import ARMS, FINAL, G3, SEEDS, UPDATES, verify_run
from db3_g5_holdout import previous_observations
from week12_comparison import REPO, read, sha, write
from week12_mesh_review import sensor_masks


EXCLUDED_CONFIG = {"seed", "result_dir", "camera_sequence_path", "save_steps", "save_ply",
                   "ply_steps", "tb_every", "depth_lambda", "sensor_min_pred_depth_m",
                   "sensor_residual_scale_inv_m"}
DEPTH_KEYS = ("median_abs_m", "p90_abs_m")


def freeze(out):
    sources = [Path(__file__), Path(__file__).with_name("db3_g5_holdout.py"),
               REPO / "scripts/week12_mesh_review.py"]
    plan = dict(
        training_protocol_sha256=sha(out / "protocol.json"),
        source_sha256={str(p.relative_to(REPO)): sha(p) for p in sources},
        paired=list(map(list, combinations(ARMS, 2))),
        scores="Each comparison uses stable measured pixels with depth in both arms; "
               "coverage uses all stable measured pixels; PSNR/SSIM use entire RGB images. "
               "Pool depth over pixels within seed; summarize deltas across seeds with equal weights.",
        attribution="Check D minus B and D minus C separately. Report paired direction counts for "
                    "render and mesh absolute median and p90. Call the robust arm consistently "
                    "superior on a metric only if its discrepancy is lower in all five seeds.",
        ready_rule="Require every D vs A endpoint gate to pass on development and on the qualified "
                   "holdout and every D recovery to pass. The archive replay remains a separate "
                   "limitation. Holdout frames from this recording do not establish new-session "
                   "generalization or physical accuracy.",
    )
    path = out / "evaluation_protocol.json"
    if path.exists():
        assert read(path) == plan, "Frozen evaluation protocol changed"
    else:
        write(path, plan)
    return plan


def load_holdout(out, transform):
    folder = out / "holdout"
    manifest_path = folder / "frames.json"
    if not manifest_path.exists():
        return [], dict(qualified=False, reason="No localized holdout manifest")
    manifest = read(manifest_path)
    for key in ("localization", "reservation"):
        assert sha(folder / f"{key}.json") == manifest[f"{key}_sha256"]
    rows = read(folder / "localization.json")["accepted"]
    rgb_excluded, depth_excluded, _, _ = previous_observations()
    rgb, depths, frames = set(), set(), []
    K = np.array(manifest["K"], dtype="f4")
    for row in manifest["frames"]:
        mid, did = row["rgb_message_id"], row["depth_message_id"]
        assert mid not in rgb | rgb_excluded and did not in depths | depth_excluded
        rgb.add(mid)
        depths.add(did)
        validated = next(f for f in rows if f["rgb_message_id"] == mid)
        assert validated["accepted"] and all(row[k] == validated[k] for k in validated)
        name = f"{mid}.png"
        rgb_path, depth_path = folder / name, folder / f"{mid}_depth.npz"
        assert sha(rgb_path) == row["rgb_sha256"] and sha(depth_path) == row["depth_sha256"]
        with np.load(depth_path) as measured, Image.open(rgb_path) as photo:
            depth, color = measured["depth_m"], np.array(photo.convert("RGB"))
        assert color.shape[:2] == depth.shape
        _, stable = sensor_masks(depth)
        assert stable.any()
        poses = [row["world_to_camera"], *row["split_world_to_camera"]]
        pose, *alternate_poses = transform_cameras(transform, np.linalg.inv(np.array(poses))).astype("f4")
        frames.append(dict(name=name, sample={"image": torch.from_numpy(color)}, depth=depth,
                           stable=stable, K=K, pose=pose, alternate_poses=alternate_poses))
    return frames, dict(qualified=manifest["qualified"], views=len(frames),
                        capture_span_s=manifest["capture_span_s"],
                        manifest_sha256=sha(manifest_path))


def scalars(run, coefficient, guarded, out):
    events = EventAccumulator(str(run / "tb"), size_guidance={"scalars": 0})
    events.Reload()
    tags = ["train/loss", "train/depthloss", "train/l1loss", "train/ssimloss", "train/num_GS"]
    if guarded:
        tags += ["sensor/evidence_samples", "sensor/active_samples", "sensor/largest_loss_share",
                 "sensor/depth_loss"]
    arrays = {}
    for tag in tags:
        records = events.Scalars(tag)
        assert [rec.step for rec in records] == list(range(7000)), (run, tag)
        values = np.array([rec.value for rec in records])
        assert np.isfinite(values).all(), (run, tag)
        arrays[tag.replace("/", "_")] = values
    np.savez_compressed(out, **arrays)
    if guarded:
        np.testing.assert_array_equal(arrays["train_depthloss"], arrays["sensor_depth_loss"])
        evidence = arrays["sensor_evidence_samples"]
        active = arrays["sensor_active_samples"]
        assert (active <= evidence).all() and (evidence > 0).all()
    result = {}
    for start, stop in ((0, 7000), (3000, 3001), (3001, 3011), (3001, 3101), (3101, 4501)):
        window = slice(start, stop)
        depth = arrays["train_depthloss"][window] * coefficient
        summary = dict(peak_weighted_depth=float(depth.max()), peak_at_step=start + int(depth.argmax()),
                       median_weighted_depth=float(np.median(depth)),
                       peak_objective=float(arrays["train_loss"][window].max()))
        if guarded:
            summary.update(guard_rejections=int((evidence[window] - active[window]).sum()),
                           evidence=int(evidence[window].sum()),
                           peak_sample_share=float(arrays["sensor_largest_loss_share"][window].max()))
        result[f"{start}:{stop}"] = summary
    return result


def recovery(run, evidence, arm, dev, scale, final_maps, seal):
    evidence.mkdir(parents=True, exist_ok=True)
    windows = scalars(run, ARMS[arm][0], ARMS[arm][1] > 0, evidence / f"{arm}_scalars.npz")
    snapshots = {}
    for step in UPDATES:
        if step == 7000:
            maps = final_maps
        else:
            name = f"ckpt_{step}_rank0.pt"
            path = run / "ckpts" / name
            assert sha(path) == seal["checkpoint_sha256"][name]
            maps, _ = model_maps(path, dev, scale, 3, None)
        snapshots[str(step)] = compare(dev, {arm: maps})
    cover = {step: score["pooled"][arm]["coverage"] for step, score in snapshots.items()}
    passed = cover["4500"] >= .99 and cover["4500"] >= cover["3000"] - .01
    return dict(passed=passed, pooled_coverage=cover, snapshots=snapshots, windows=windows,
                scalar_sha256=sha(evidence / f"{arm}_scalars.npz"))


def score(frames, maps, p90_limit):
    kinds = "render", "mesh"
    paired = {a + b: {kind: compare(frames, {c: maps[c][kind] for c in (a, b)})
                      for kind in kinds} for a, b in combinations(ARMS, 2)}
    checks = candidate_checks({a: paired["A" + a]["render"] for a in "BCD"},
                              {a: paired["A" + a]["mesh"] for a in "BCD"}, {"passed": True}, "A")
    for arm, check in checks.items():
        failures = check["failures"]
        for kind, comparison in paired["A" + arm].items():
            for view, rows in compare_rows(comparison):
                if view == "pooled" or rows["A"]["error"] is None:
                    continue
                tail_change = 100 * (rows[arm]["error"]["p90_abs_m"] - rows["A"]["error"]["p90_abs_m"])
                if tail_change > p90_limit:
                    failures.append([kind, view, "absolute p90"])
        checks[arm] = dict(passed=not failures, failures=failures)
    return dict(pairwise=paired, all_arm={kind: compare(frames, {a: maps[a][kind] for a in ARMS})
                                          for kind in kinds}, checks_against_A=checks)


def save_maps(out, frames, maps):
    paths = {}
    for f in frames:
        name = f["name"]
        layers = {"stable": f["stable"], "sensor": f["depth"]}
        for arm in ARMS:
            for kind in ("render", "mesh"):
                prediction = maps[arm][kind][name]
                stem = f"{arm}_{kind}"
                layers[stem] = prediction["depth"]
                layers[stem + "_valid"] = prediction["valid"]
        path = out / f"{Path(name).stem}_maps.npz"
        np.savez_compressed(path, **layers)
        paths[name] = sha(path)
    return paths


def evaluate_seed(out, label, dev, holdout, scale, mesh_reference, protocol):
    counts, maps, recovery_reports, seals = {}, {}, {}, {}
    evidence = out / label / "evaluation"
    baseline_cfg = config(G3 / "G3-A")
    sequence = read(out / f"{label}_cameras.json")
    assert sha(out / f"{label}_cameras.json") == protocol["camera_sha256"][label]
    for arm in ARMS:
        run = out / label / arm
        cfg = config(run)
        assert tuple(float(cfg[k]) for k in
                     ("depth_lambda", "sensor_min_pred_depth_m", "sensor_residual_scale_inv_m")) == ARMS[arm]
        assert cfg["tb_every"] == "1"
        assert {k: v for k, v in cfg.items() if k not in EXCLUDED_CONFIG} == {
            k: v for k, v in baseline_cfg.items() if k not in EXCLUDED_CONFIG}, (label, arm)
        seal = read(run / "verified.json")
        verified = verify_run(run, int(label[1:]), sequence)
        assert all(seal[key] == verified[key] for key in verified)
        assert seal["mesh_sha256"] == sha(run / "mesh/raw_mesh.ply")
        seals[arm] = seal["experiment"]["initial_sha256"]
        result, counts[arm] = endpoint(run, run / "mesh", dev + holdout, scale, mesh_reference)
        assert counts[arm] == seal["experiment"]["final_count"]
        maps[arm] = result
        recovery_reports[arm] = recovery(run, evidence, arm, dev, scale, result["render"], seal)
    assert len(set(seals.values())) == 1
    report = dict(counts=counts, recovery=recovery_reports, matched_initializer=seals["A"])
    for split, frames in (("development", dev), ("holdout", holdout)):
        if frames:
            report[split] = score(frames, maps, protocol["extra_per_view_p90_limit_cm"])
            report[split]["maps_sha256"] = save_maps(evidence, frames, maps)
    write(evidence / "scores.json", report)
    print(f"{label}: {counts}; development failures "
          f"{report['development']['checks_against_A']['D']['failures']}", flush=True)
    return report


def statistics(values):
    defined = [value for value in values if value is not None]
    return dict(mean=float(np.mean(defined)) if defined else None,
                sd=float(np.std(defined, ddof=1)) if len(defined) > 1 else None,
                values=values, n_negative=sum(value < 0 for value in defined),
                n_equal=sum(value == 0 for value in defined), n_defined=len(defined))


def depth_delta(rows, a, b, metric):
    if rows[b]["error"] is None or rows[a]["error"] is None:
        return None
    return 100 * (rows[b]["error"][metric] - rows[a]["error"][metric])


def worst_tail_delta(comparison, a, b):
    deltas = [depth_delta(row["arms"], a, b, "p90_abs_m")
              for row in comparison["per_view"].values()]
    return None if None in deltas else max(deltas)


def summarize(seeds, split):
    summary = {}
    for a, b in combinations(ARMS, 2):
        pair = a + b
        summary[pair] = {}
        for kind in ("render", "mesh"):
            scores = [report[split]["pairwise"][pair][kind] for report in seeds.values()]
            metrics = {}
            for metric in DEPTH_KEYS:
                metrics[metric + "_delta_cm"] = statistics([
                    depth_delta(row["pooled"], a, b, metric) for row in scores])
            metrics["coverage_delta_pp"] = statistics([
                100 * (row["pooled"][b]["coverage"] - row["pooled"][a]["coverage"]) for row in scores])
            metrics["worst_view_p90_delta_cm"] = statistics([
                worst_tail_delta(row, a, b) for row in scores])
            if kind == "render":
                for metric in ("psnr_db", "ssim"):
                    metrics[metric + "_delta"] = statistics([
                        row["pooled"][b]["rgb"][metric] - row["pooled"][a]["rgb"][metric] for row in scores])
            summary[pair][kind] = metrics
        summary[pair]["gaussian_count_delta"] = statistics([
            report["counts"][b] - report["counts"][a] for report in seeds.values()])
        summary[pair]["gaussian_count_change_pct"] = statistics([
            100 * (report["counts"][b] / report["counts"][a] - 1.) for report in seeds.values()])
    return summary


def evaluate(out):
    plan = freeze(out)
    protocol = read(out / "protocol.json")
    for relative, expected in {**protocol["source_sha256"], **protocol["input_sha256"]}.items():
        assert sha(REPO / relative) == expected, relative
    pair = read(PAIR_PATH)
    frames, scale, _, depth_hashes = load_frames(pair)
    reference = read(G3 / "meshes/G3-A/report.json")
    holdout, holdout_info = load_holdout(out, np.array(reference["parser_transform"]))
    seeds = {f"s{seed}": evaluate_seed(out, f"s{seed}", frames["val"], holdout, scale, reference, protocol)
             for seed in SEEDS}
    summary = {split: summarize(seeds, split) for split in ("development", "holdout")
               if all(split in report for report in seeds.values())}
    gates = {label: {"development": r["development"]["checks_against_A"]["D"]["passed"],
                     "holdout": bool(holdout_info["qualified"]) and r["holdout"]["checks_against_A"]["D"]["passed"],
                     "recovery": r["recovery"]["D"]["passed"]} for label, r in seeds.items()}
    write(out / "summary.json", dict(seed_order=list(seeds), paired=summary, d_gate_by_seed=gates,
                                    eligible_for_A05_abs_transfer=all(all(g.values()) for g in gates.values()),
                                    holdout=holdout_info,
                                    depth_sha256=depth_hashes, evaluation_protocol=plan,
                                    seed_scores_sha256={label: sha(out / label / "evaluation/scores.json")
                                                        for label in seeds}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--freeze-only", action="store_true")
    cli = parser.parse_args()
    output = cli.output.resolve()
    if cli.freeze_only:
        freeze(output)
    else:
        evaluate(output)
