"""Replay the Room training order for the final report's controlled A05-Abs study."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

from research_campaign import REPO, command, job, sha, verify, write

STEPS = 7000
DEADLINES = (90, 180, 270)
UPDATES = (1, 1000, 2000, 3000, 4000, 5000, 6000, STEPS)
BASELINE = REPO / "results/research_completion_20261027/replication/room/B00_s42"
ARMS = {
    "B00": {},
    "A05_absgrad": {},
    "cap400k": {"cap": 400000},
    "cap1000k": {"cap": 1000000},
    "no_error": {"error_weight": 0},
    "no_edge": {"edge_weight": 0},
    "signed_high": {"absolute": False},
    "abs_low": {"threshold": 0.0002},
    "A04": {"absolute": False, "threshold": 0.0002},
}
TIGHT_RANKING_ARMS = {
    "no_error_400k": {"cap": 400000, "error_weight": 0},
    "no_edge_400k": {"cap": 400000, "edge_weight": 0},
}
SOURCES = (
    "scripts/final_report_study.py", "scripts/research_campaign.py",
    "scripts/run-school-completion.ps1", "examples/simple_trainer.py",
    "examples/datasets/colmap.py", "gsplat/rendering.py", "gsplat/strategy/default.py",
    "gsplat/strategy/budget.py", "gsplat/strategy/ops.py",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def target_hashes(run):
    from PIL import Image

    result = {}
    for path in sorted((run / "renders").glob("val_step6999_*.png")):
        with Image.open(path) as image:
            target = image.crop((0, 0, image.width // 2, image.height))
            result[path.name] = hashlib.sha256(target.tobytes()).hexdigest()
    if len(result) != 39:
        raise ValueError(f"Expected 39 final evaluation targets in {run}, found {len(result)}")
    return result


def make_plan(root, arms):
    jobs = {}
    for arm, changes in arms.items():
        spec = job("room", 42, method=arm, cap=changes.get("cap", 700000))
        args = command(spec, root, STEPS)
        args += ["--data-workers", "0", "--camera-sequence-path", str(root / "camera_order.json")]
        if arm != "B00":
            if changes.get("absolute", True):
                args += ["--strategy.absgrad"]
            args += ["--strategy.grow-grad2d", str(changes.get("threshold", 0.0008))]
            for setting in ("error_weight", "edge_weight"):
                if setting in changes:
                    flag = "--strategy." + setting.replace("_", "-")
                    args[args.index(flag) + 1] = str(changes[setting])
        if arm in {"B00", "A05_absgrad"}:
            start = args.index("--eval-steps") + 1
            args[start:start + 2] = list(map(str, UPDATES))
            args += ["--timed-eval-seconds", *map(str, DEADLINES)]
        jobs[arm] = {"spec": spec, "command": args, "changes": changes}
    return jobs


def validate_probes(run, final_count):
    updates = [read(run / f"stats/val_step{step - 1:04d}.json") for step in UPDATES]
    deadlines = sorted(
        (read(path) for path in (run / "stats").glob("time*_step*.json")),
        key=lambda row: row["target_seconds"],
    )
    if [row["completed_updates"] for row in updates] != list(UPDATES):
        raise ValueError(f"Incorrect update probes in {run}")
    if [row["target_seconds"] for row in deadlines] != list(DEADLINES):
        raise ValueError(f"Missing elapsed-time probes in {run}")
    if updates[-1]["num_GS"] != final_count:
        raise ValueError(f"Final probe count differs from the checkpoint in {run}")
    for row in updates + deadlines:
        if not all(math.isfinite(row[k]) for k in ("psnr", "ssim", "lpips", "train_seconds")):
            raise ValueError(f"Nonfinite probe in {run}: {row}")
    if any(not 0 <= row["train_seconds"] - row["target_seconds"] <= 5 for row in deadlines):
        raise ValueError(f"Elapsed-time probe missed the 5-second tolerance in {run}")
    if any(a["train_seconds"] >= b["train_seconds"] for a, b in zip(updates, updates[1:])):
        raise ValueError(f"Update probes have non-increasing times in {run}")
    return {"updates": updates, "timed": deadlines}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tight-ranking", action="store_true",
                        help="Test the two ranking removals at the 400,000-Gaussian cap.")
    parser.add_argument("--arm", choices=(*ARMS, *TIGHT_RANKING_ARMS), action="append",
                        help="Execute selected arms in protocol order. Default runs all arms.")
    args = parser.parse_args()
    arms = TIGHT_RANKING_ARMS if args.tight_ranking else ARMS
    if args.arm and not set(args.arm) <= arms.keys():
        parser.error("--arm must belong to the selected design")
    root = args.root.resolve()
    archive = read(BASELINE / "experiment.json")
    reference_targets = target_hashes(BASELINE)
    jobs = make_plan(root, arms)
    protocol = {
        "scene": "room", "seed": 42, "updates": STEPS, "update_probes": UPDATES,
        "time_targets_s": DEADLINES, "time_tolerance_s": 5,
        "reference": str(BASELINE), "reference_initial_sha256": archive["initial_sha256"],
        "reference_camera_sha256": archive["camera_sequence_sha256"],
        "validation_target_hashes": reference_targets,
        "sources": {name: sha(REPO / name) for name in SOURCES}, "arms": jobs,
    }
    protocol_path = root / "protocol.json"
    order_path = root / "camera_order.json"
    if protocol_path.exists():
        if read(protocol_path) != json.loads(json.dumps(protocol)) or read(order_path) != archive["camera_sequence"]:
            raise ValueError("The protocol, source or camera order changed. Use a fresh output root.")
    else:
        for name in SOURCES:
            dest = root / "source" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, dest)
        write(order_path, archive["camera_sequence"])
        write(protocol_path, protocol)

    summary = {}
    for arm, entry in jobs.items():
        spec = entry["spec"]
        run = root / spec["id"]
        done = run / "verified.json"
        if done.exists():
            record = read(done)
            if sha(run / "ckpts/final_post_update.pt") != record["final_checkpoint_sha256"]:
                raise ValueError(f"Saved checkpoint differs from {done}")
        elif args.arm is None or arm in args.arm:
            if run.exists():
                raise FileExistsError(f"Preserve and inspect incomplete run {run} before retrying")
            run.parent.mkdir(parents=True, exist_ok=True)
            write(root / "status.json", {"state": "training", "arm": arm})
            print(f"TRAIN {arm}", flush=True)
            started = time.perf_counter()
            with run.with_suffix(".log").open("w", encoding="utf-8") as log:
                subprocess.run(entry["command"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
            wall = time.perf_counter() - started
            record = verify(root, spec, STEPS)
            record["process_wall_seconds"] = wall
            observed = read(run / "experiment.json")
            if (record["initial_sha256"] != archive["initial_sha256"]
                or record["camera_sha256"] != archive["camera_sequence_sha256"]
                or observed["camera_sequence"] != archive["camera_sequence"]
                or target_hashes(run) != reference_targets):
                raise ValueError(f"Initialization, fitting camera order or RGB targets differ in {run}")
            if arm in {"B00", "A05_absgrad"}:
                record["probes"] = validate_probes(run, record["metrics"]["num_GS"])
            if arm != "B00":
                growth = record["train"]["growth_history"]
                if len(growth) != 54 or not all(
                    h["added"] == min(max(h["target"] - h["before"], 0), h["eligible"])
                    and h["after"] == h["before"] + h["added"] - h["pruned"]
                    and h["duplicated"] == 0 and h["split"] == h["added"]
                    for h in growth
                ):
                    raise ValueError(f"Invalid budget accounting in {run}")
            write(done, record)
            print(f"VERIFIED {arm} {record['metrics']}", flush=True)
        else:
            continue
        summary[arm] = {key: record[key] for key in (
            "metrics", "initial_sha256", "camera_sha256", "final_checkpoint_sha256",
            "checkpoint_bytes", "process_wall_seconds",
        )}
        if "probes" in record:
            summary[arm]["probes"] = record["probes"]
        write(root / "summary.json", summary)
    write(root / "status.json", {"state": "complete" if len(summary) == len(arms) else "selected_arms_complete",
                                 "completed": list(summary)})


if __name__ == "__main__":
    main()
