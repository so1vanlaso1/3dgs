"""Three-scene A05 split-utility experiment from archived A04 weights."""

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from research_campaign import REPO, command, job, sha, verify, write

sys.path[:0] = [str(REPO / "examples"), str(REPO)]
from simple_trainer import Config, Runner
from gsplat.losses import ssim_loss
from gsplat.scene import GaussianScene
from gsplat.stage import Stage
from gsplat.strategy import BudgetStrategy
from gsplat.strategy.ops import split


SCENES = ("room", "counter", "kitchen")
BRANCHES = ("control", "gradient", "residual", "control_replay")
HORIZONS = (0, 100, 500)
SEED = 42
ARCHIVE = REPO / "results/research_completion_20261027/replication"


def render(runner, sample):
    pixels = sample["image"].to(runner.device)[None].float() / 255
    height, width = pixels.shape[1:3]
    mask = sample.get("mask")
    if mask is not None:
        mask = mask.to(runner.device)[None]
    colors, _, info = runner.stage.render(
        runner.scene.id,
        camtoworlds=sample["camtoworld"].to(runner.device)[None],
        Ks=sample["K"].to(runner.device)[None],
        width=width,
        height=height,
        sh_degree=3,
        near_plane=runner.cfg.near_plane,
        far_plane=runner.cfg.far_plane,
        masks=mask,
    )
    return colors, pixels, info, mask


def objective(colors, pixels, mask):
    if mask is None:
        l1 = (colors - pixels).abs().mean()
    else:
        l1 = (colors[mask] - pixels[mask]).abs().mean()
        colors = colors * mask[..., None]
        pixels = pixels * mask[..., None]
    return torch.lerp(l1, ssim_loss(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)), 0.2)


def probe(runner, strategy, indices):
    params = runner.splats
    state = strategy.initialize_state(runner.scene_scale)
    for step, index in enumerate(indices, start=5001):
        colors, pixels, info, mask = render(runner, runner.trainset[index])
        strategy.step_pre_backward(params, runner.optimizers, state, step, info)
        strategy.observe_image(params, state, step, info, colors, pixels, mask, camera_ids=[index])
        objective(colors, pixels, mask).backward()
        with torch.no_grad():
            strategy._update_state(params, state, info, packed=True)
        for opt in runner.optimizers.values():
            opt.zero_grad(set_to_none=True)

    gradients = state["grad2d"] / state["count"].clamp_min(1)
    opaque = params["opacities"].detach().sigmoid().flatten() >= strategy.prune_opa
    eligible = (gradients > strategy.grow_grad2d) & opaque
    pool, residual = strategy._fallback_candidates(state, gradients, eligible, opaque)
    count = min(strategy.fallback_max, strategy.cap_max - len(gradients), len(pool))
    if count <= 0:
        raise RuntimeError("No budget or supported below-threshold candidates at this checkpoint")
    selected = {
        label: pool[score[pool].topk(count, sorted=False).indices].cpu()
        for label, score in (("gradient", gradients), ("residual", residual))
    }
    details = {
        "base_count": len(gradients),
        "above_threshold_and_opaque": int(eligible.sum()),
        "below_or_at_threshold_and_opaque": int(((gradients <= strategy.grow_grad2d) & opaque).sum()),
        "below_opacity_floor": int((~opaque).sum()),
        "distinct_scored_views": len(state["fallback_seen"]),
        "fallback_pool": len(pool),
        "split_count": count,
        "overlap": len(set(selected["gradient"].tolist()) & set(selected["residual"].tolist())),
        "parents": {},
    }
    for label, ids in selected.items():
        details["parents"][label] = {
            "ids": ids.tolist(),
            "gradient": gradients[ids].cpu().tolist(),
            "residual": residual[ids].cpu().tolist(),
            "views": state["fallback_views"][ids].cpu().tolist(),
        }
    return selected, details


@torch.no_grad()
def evaluate(runner, indices):
    rows = []
    for i in indices:
        colors, pixels, _, _ = render(runner, runner.valset[i])
        colors = colors.clamp(0, 1).permute(0, 3, 1, 2)
        pixels = pixels.permute(0, 3, 1, 2)
        name = str(runner.parser.image_names[runner.valset.indices[i]])
        rows.append({
            "name": name,
            "psnr": runner.psnr(colors, pixels).item(),
            "ssim": runner.ssim(colors, pixels).item(),
            "lpips": runner.lpips(colors, pixels).item(),
            "mse": (colors - pixels).square().mean().item(),
        })
    return {
        "mean": {k: float(np.mean([row[k] for row in rows])) for k in ("psnr", "ssim", "lpips", "mse")},
        "views": rows,
        "gaussians": len(runner.splats["means"]),
    }


def restore(runner, weights, lrs):
    for key, param in runner.splats.items():
        param.data = weights[key].to(runner.device)
    for key, opt in runner.optimizers.items():
        opt.state.clear()
        opt.param_groups[0]["lr"] = lrs[key]
    runner.scene = GaussianScene.from_splats(runner.splats, id="scene")
    runner.splats = runner.scene.splats
    runner.stage = Stage()
    runner.stage.add_scene(runner.scene, runner.rasterize_splats)


def branch(runner, weights, lrs, parents, strategy, sequence, eval_indices, horizons):
    restore(runner, weights, lrs)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    initial = len(weights["means"])
    extra = 0 if parents is None else len(parents)
    if extra:
        mask = torch.zeros(initial, device=runner.device, dtype=torch.bool)
        mask[parents] = True
        split(
            runner.splats, runner.optimizers, {}, mask, scene=runner.scene,
            mode=strategy.split_mode, split_distance=strategy.split_distance,
            opacity_reduction=strategy.opacity_reduction,
        )
    assert len(runner.splats["means"]) == initial + extra <= strategy.cap_max
    results = {0: evaluate(runner, eval_indices)}
    for step, index in enumerate(sequence, start=1):
        colors, pixels, _, mask = render(runner, runner.trainset[index])
        objective(colors, pixels, mask).backward()
        for opt in runner.optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if step in horizons:
            results[step] = evaluate(runner, eval_indices)
    assert all(torch.isfinite(p).all() for p in runner.splats.values())
    return {
        "metrics": results,
        "wall_seconds": perf_counter() - started,
        "peak_tensor_mib": torch.cuda.max_memory_allocated() / 2**20,
    }


def train_order(size, length, seed):
    rng = np.random.default_rng(seed)
    order = []
    while len(order) < length:
        order.extend(rng.permutation(size).tolist())
    return order[:length]


def run_scene(root, scene, smoke):
    folder = root / scene
    source = ARCHIVE / scene / f"A04_s{SEED}"
    checkpoint = source / "ckpts/final_post_update.pt"
    verified = json.loads((source / "verified.json").read_text())
    checkpoint_hash = sha(checkpoint)
    assert checkpoint_hash == verified["final_checkpoint_sha256"]
    strategy = BudgetStrategy(
        fallback_rank="residual", refine_stop_iter=6000,
        split_mode="long_axis", clone_policy="split", edge_weight=0.25,
    )
    cfg = Config(
        seed=SEED, data_dir=str(REPO / "data" / f"{scene}_factor2"), data_factor=1,
        result_dir=str(folder), disable_viewer=True, disable_video=True, packed=True,
        # Use the last A04 means LR and hold it fixed throughout the continuation.
        means_lr=Config().means_lr * 0.01, strategy=strategy,
    )
    runner = Runner(0, 0, 1, cfg)
    runner.trainset.preload_images()
    runner.valset.preload_images()
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)["splats"]
    assert len(weights["means"]) == verified["metrics"]["num_GS"]
    lrs = {k: opt.param_groups[0]["lr"] for k, opt in runner.optimizers.items()}
    restore(runner, weights, lrs)
    observation_order = train_order(len(runner.trainset), 100, SEED + 5000)
    horizons = (0, 2, 5) if smoke else HORIZONS
    continuation_order = train_order(len(runner.trainset), horizons[-1], SEED + 7000)
    evaluation = list(range(2 if smoke else len(runner.valset)))
    selected, probe_details = probe(runner, strategy, observation_order)
    names = runner.parser.image_names
    write(folder / "probe.json", {
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "training_names": [str(names[i]) for i in runner.trainset.indices],
        "eval_names": [str(names[runner.valset.indices[i]]) for i in evaluation],
        "observation_order": observation_order, "continuation_order": continuation_order,
        "learning_rates": lrs, "probe": probe_details,
    })
    print(scene, "pool", probe_details["fallback_pool"], "split", probe_details["split_count"],
          "overlap", probe_details["overlap"], flush=True)
    outcomes = {}
    for label in BRANCHES:
        result = branch(runner, weights, lrs, selected.get(label), strategy,
                        continuation_order, evaluation, horizons)
        outcomes[label] = result
        write(folder / f"{label}.json", result)
        print(scene, label, result["metrics"][horizons[-1]]["mean"], flush=True)
    return outcomes


def summarize(root, results, smoke):
    final = 5 if smoke else HORIZONS[-1]
    summary = {}
    for scene, outcomes in results.items():
        endpoints = {label: outcome["metrics"][final] for label, outcome in outcomes.items()}
        assert endpoints["residual"]["gaussians"] == endpoints["gradient"]["gaussians"]
        for endpoint in endpoints.values():
            assert all(math.isfinite(value) for value in endpoint["mean"].values())
            assert [v["name"] for v in endpoint["views"]] == [
                v["name"] for v in endpoints["control"]["views"]
            ]
        scores = {label: endpoint["mean"] for label, endpoint in endpoints.items()}
        delta = {
            label: {k: scores["residual"][k] - scores[label][k] for k in scores[label]}
            for label in ("control", "gradient")
        }
        replay_psnr = abs(scores["control"]["psnr"] - scores["control_replay"]["psnr"])
        passes = (
            delta["control"]["psnr"] > replay_psnr
            and delta["gradient"]["psnr"] >= 0.05
            and delta["gradient"]["psnr"] > replay_psnr
            and all(d["ssim"] >= -0.001 and d["lpips"] <= 0.001 for d in delta.values())
        )
        summary[scene] = {
            "endpoints": scores, "residual_delta": delta, "control_replay_psnr": replay_psnr,
            "gaussians": {k: v["gaussians"] for k, v in endpoints.items()}, "passes": passes,
        }
    write(root / "summary.json", {"smoke": smoke, "horizon": final, "scenes": summary})


def train_campaign(root, smoke, variant="residual", seed=SEED):
    """Screen an A05 policy through the normal trainer and A04 schedule."""
    steps = 60 if smoke else 7000
    scenes = SCENES[:1] if smoke else SCENES
    method = {"residual": "A05", "absgrad": "A05_absgrad", "a04": "A04"}[variant]
    jobs = [job(scene, seed, method=method, cap=150000 if smoke else 700000) for scene in scenes]
    commands = []
    for spec in jobs:
        args = command(spec, root, steps)
        if variant == "residual":
            args += ["--strategy.fallback-rank", "residual"]
            if smoke:
                args += [
                    "--strategy.fallback-start-iter", "20", "--strategy.fallback-max", "8",
                    "--strategy.score-every", "2", "--strategy.grow-grad2d", "1",
                ]
        elif variant == "absgrad":
            args += ["--strategy.absgrad", "--strategy.grow-grad2d", "0.0008"]
        commands.append(args)
    sources = (
        "examples/simple_trainer.py", "examples/datasets/colmap.py",
        "gsplat/strategy/budget.py", "gsplat/strategy/default.py",
        "gsplat/strategy/ops.py", "gsplat/rendering.py",
        "scripts/research_a05.py", "scripts/research_campaign.py", "scripts/run-research.ps1",
    )
    for name in sources:
        snapshot = root / "evidence/source" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, snapshot)
    write(root / "protocol.json", {
        "steps": steps, "seed": seed, "variant": variant, "smoke": smoke,
        "jobs": jobs, "commands": commands,
        "source_hashes": {name: sha(REPO / name) for name in sources},
        "control": str(ARCHIVE), "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "gate": "PSNR gain >= 0.05 dB, SSIM loss <= 0.001 and LPIPS increase <= 0.001 versus A04",
        "absgrad_reference": "https://arxiv.org/abs/2404.10484",
        "absgrad_threshold_source": "gsplat/strategy/default.py documentation recommends 0.0008",
        "limitations": "Exposed development views, independent runs at a common cap. Absolute gradient and threshold change together.",
    })
    summary = {}
    for spec, args in zip(jobs, commands):
        scene = spec["scene"]
        folder = root / spec["id"]
        log = root / f"{scene}.log"
        write(root / "status.json", {"state": "running", "scene": scene})
        print("TRAIN", scene, flush=True)
        started = perf_counter()
        try:
            with log.open("w", encoding="utf-8") as output:
                subprocess.run(args, cwd=REPO, stdout=output, stderr=subprocess.STDOUT, check=True)
            record = verify(root, spec, steps)
            record["wall_seconds"] = perf_counter() - started
            extra = 0
            for row in record["train"]["growth_history"]:
                fallback = row.get("fallback_added", 0)
                slots = row["target"] - row["before"]
                assert row["added"] == min(slots, row["eligible"]) + fallback <= slots
                assert row["after"] == row["before"] + row["added"] - row["pruned"]
                assert 0 <= fallback <= min(8 if smoke else 512, row.get("fallback_candidates", 0))
                extra += fallback
            record["fallback_total"] = extra
            write(folder / "verified.json", record)
            comparison = {"a05": record["metrics"], "variant": variant, "fallback_added": extra}
            if not smoke:
                archive = ARCHIVE / scene / f"A04_s{seed}" / "verified.json"
                control = json.loads(archive.read_text(encoding="utf-8"))
                for key in ("initial_sha256", "camera_sha256"):
                    assert record[key] == control[key], f"{scene} mismatch in {key}"
                comparison["a04"] = control["metrics"]
                comparison["a04_record_sha256"] = sha(archive)
                comparison["delta"] = {
                    k: record["metrics"][k] - control["metrics"][k]
                    for k in ("psnr", "ssim", "lpips", "num_GS")
                }
                delta = comparison["delta"]
                comparison["practical_pass"] = (
                    delta["psnr"] >= 0.05 and delta["ssim"] >= -0.001 and delta["lpips"] <= 0.001
                )
            elif variant == "residual":
                assert extra > 0, "Smoke did not exercise fallback growth"
            if variant != "residual":
                assert extra == 0
                assert sum(row["added"] for row in record["train"]["growth_history"]) > 0
            summary[scene] = comparison
            write(root / "summary.json", {"smoke": smoke, "variant": variant, "seed": seed, "scenes": summary})
            print("DONE", scene, comparison, flush=True)
        except Exception as error:
            write(root / "status.json", {"state": "failed", "scene": scene, "error": str(error)})
            raise
    write(root / "status.json", {"state": "complete", "scenes": scenes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", action="store_true", help="Run the normal end-to-end trainer")
    parser.add_argument("--variant", choices=("residual", "absgrad", "a04"), default="residual")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=SEED)
    args = parser.parse_args()
    if not args.train and (args.variant != "residual" or args.seed != SEED):
        parser.error("Variant and seed selection require --train")
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    if args.train:
        train_campaign(args.root, args.smoke, args.variant, args.seed)
        return
    scenes = SCENES[:1] if args.smoke else SCENES
    sources = (Path(__file__), REPO / "examples/simple_trainer.py",
               REPO / "gsplat/strategy/budget.py", REPO / "gsplat/strategy/ops.py")
    write(args.root / "protocol.json", {
        "scenes": scenes, "seed": SEED, "branches": BRANCHES, "smoke": args.smoke,
        "source_hashes": {str(p.relative_to(REPO)): sha(p) for p in sources},
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "kind": "final A04 weights, frozen observation, matched fresh Adam continuations",
        "gate": "+0.05 dB against gradient and improvement against control above replay noise, SSIM and LPIPS within 0.001 of both",
    })
    results = {}
    for scene in scenes:
        write(args.root / "status.json", {"state": "running", "scene": scene})
        try:
            results[scene] = run_scene(args.root, scene, args.smoke)
            summarize(args.root, results, args.smoke)
        except Exception as error:
            write(args.root / "status.json", {"state": "failed", "scene": scene, "error": str(error)})
            raise
    write(args.root / "status.json", {"state": "complete", "scenes": scenes})


if __name__ == "__main__":
    main()
