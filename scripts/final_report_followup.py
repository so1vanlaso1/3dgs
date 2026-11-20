"""Run the frozen, paired seed replication and outdoor evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from final_report_study import SOURCES, STEPS
from research_campaign import REPO, command, job, sha, verify, write


PAIRS = (("garden", 4, 42), ("room", 2, 43), ("room", 2, 44))
ARMS = ("B00", "A05_absgrad")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def plan_pair(root, scene, factor, seed):
    dataset = REPO / f"data/{scene}_factor{factor}"
    images = sorted((dataset / "images").glob("*.jpg"))
    holdout = images[::8]
    if not holdout:
        raise FileNotFoundError(f"Missing benchmark images in {dataset}")
    jobs = {}
    for arm in ARMS:
        spec = job(scene, seed, method=arm)
        args = command(spec, root, STEPS)
        args[args.index("--data-dir") + 1] = str(dataset)
        args.remove("--save-ply")
        ply_index = args.index("--ply-steps")
        del args[ply_index:ply_index + 2]
        args += ["--data-workers", "0"]
        if arm != "B00":
            args += ["--strategy.absgrad", "--strategy.grow-grad2d", "0.0008"]
        jobs[arm] = {"spec": spec, "command": args}
    return {
        "scene": scene, "factor": factor, "seed": seed, "steps": STEPS,
        "fitting_images": len(images) - len(holdout),
        "held_out_images": [path.name for path in holdout],
        "prepared_model_sha256": {
            name: sha(dataset / "sparse/0" / name)
            for name in ("cameras.bin", "images.bin", "points3D.bin")
        },
        "jobs": jobs,
    }


def targets(run):
    from PIL import Image

    result = {}
    for path in sorted((run / "renders").glob("val_step6999_*.png")):
        with Image.open(path) as image:
            rgb = image.crop((0, 0, image.width // 2, image.height))
            result[path.name] = hashlib.sha256(rgb.tobytes()).hexdigest()
    return result


def run_arm(root, entry):
    spec = entry["spec"]
    run = root / spec["id"]
    done = run / "verified.json"
    if done.exists():
        record = read(done)
        if sha(run / "ckpts/final_post_update.pt") != record["final_checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed in {run}")
        return record
    wall_seconds = None
    if run.exists():
        import yaml

        if not (run / "ckpts/final_post_update.pt").is_file():
            raise FileExistsError(f"Preserve and inspect the incomplete run in {run}")
        config = yaml.load((run / "cfg.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        argv = entry["command"]
        expected = {
            "data_dir": argv[argv.index("--data-dir") + 1], "seed": str(spec["seed"]),
            "max_steps": str(STEPS), "data_workers": "0", "test_every": "8",
            "independent_sampling": "true", "result_dir": str(run),
        }
        strategy = config["strategy"]
        expected_strategy = {"absgrad": "false", "grow_grad2d": "0.0002"} if spec["method"] == "B00" else {
            "absgrad": "true", "grow_grad2d": "0.0008", "cap_max": str(spec["cap"]),
            "split_mode": "long_axis", "clone_policy": "split",
        }
        if any(config[name] != value for name, value in expected.items()) or any(
            strategy[name] != value for name, value in expected_strategy.items()
        ):
            raise ValueError(f"Saved configuration differs from the frozen command in {run}")
        print(f"AUDIT EXISTING {spec['id']}", flush=True)
    else:
        run.parent.mkdir(parents=True, exist_ok=True)
        print(f"TRAIN {spec['id']}", flush=True)
        started = time.perf_counter()
        with run.with_suffix(".log").open("w", encoding="utf-8") as log:
            subprocess.run(entry["command"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        wall_seconds = time.perf_counter() - started
    record = verify(root, spec, STEPS)
    history = record["train"].get("growth_history", [])
    if spec["method"] != "B00" and (
        len(history) != 54
        or any(row["added"] != min(row["eligible"], max(0, row["target"] - row["before"]))
               or row["after"] != row["before"] + row["added"] - row["pruned"]
               or row["split"] != row["added"] or row["duplicated"] != 0 for row in history)
    ):
        raise ValueError(f"Invalid scheduled growth accounting in {run}")
    record["binding_events"] = sum(row["eligible"] > row["added"] for row in history) if spec["method"] != "B00" else None
    record["target_rgb_sha256"] = targets(run)
    record["process_wall_seconds"] = wall_seconds
    write(done, record)
    print(f"VERIFIED {spec['id']} {record['metrics']}", flush=True)
    return record


def compare(pair, records):
    baseline, candidate = (records[arm] for arm in ARMS)
    for key in ("initial_sha256", "camera_sha256", "target_rgb_sha256"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Pair mismatch for {key} in {pair['scene']}, seed {pair['seed']}")
    if len(baseline["target_rgb_sha256"]) != len(pair["held_out_images"]):
        raise ValueError("The final render count differs from the frozen holdout")
    left, right = baseline["metrics"], candidate["metrics"]
    return {
        "scene": pair["scene"], "factor": pair["factor"], "seed": pair["seed"],
        "fitting_images": pair["fitting_images"], "held_out_images": pair["held_out_images"],
        "initial_sha256": baseline["initial_sha256"], "camera_sha256": baseline["camera_sha256"],
        "target_rgb_sha256": baseline["target_rgb_sha256"],
        "arms": {arm: {key: record[key] for key in ("metrics", "binding_events", "final_checkpoint_sha256",
                 "checkpoint_bytes", "process_wall_seconds")} for arm, record in records.items()},
        "delta": {key: right[key] - left[key] for key in ("psnr", "ssim", "lpips")},
        "gaussian_reduction_fraction": 1 - right["num_GS"] / left["num_GS"],
        "higher_psnr_with_fewer_gaussians": right["psnr"] > left["psnr"] and right["num_GS"] < left["num_GS"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sources = (*SOURCES, "scripts/final_report_followup.py", "scripts/download-room-small.py")
    pairs = [plan_pair(root, *pair) for pair in PAIRS]
    protocol = {
        "selection": "Garden outdoor scene and Room seeds 43 and 44 fixed before training this follow-up.",
        "rules": "No tuning on the new evaluation. Report all paired metrics; count and PSNR determine the direction of the trade-off.",
        "holdout": "Every eighth image in COLMAP name order. Supplied SfM may use all photos.",
        "sources": {name: sha(REPO / name) for name in sources},
        "pairs": pairs,
    }
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        if read(protocol_path) != protocol:
            raise ValueError("Frozen protocol changed. Use a fresh results directory.")
    else:
        for name in sources:
            target = root / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, target)
        write(protocol_path, protocol)

    summary = []
    for pair in pairs:
        records = {}
        for arm in ARMS:
            entry = pair["jobs"][arm]
            write(root / "status.json", {"state": "running", "arm": entry["spec"]["id"]})
            records[arm] = run_arm(root, entry)
        summary.append(compare(pair, records))
        write(root / "summary.json", summary)
    write(root / "status.json", {"state": "complete", "pairs": len(summary)})


if __name__ == "__main__":
    main()
