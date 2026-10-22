"""Check deterministic fixture regeneration and capture workspace provenance."""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout.strip()


if __name__ == "__main__":
    fixture = OUT / "data" / "analytic_spheres_v1"
    before = {p.relative_to(fixture).as_posix(): sha(p) for p in sorted(fixture.rglob("*")) if p.is_file()}
    output = run(sys.executable, str(OUT / "scripts" / "prepare_stage0_data.py"))
    after = {p.relative_to(fixture).as_posix(): sha(p) for p in sorted(fixture.rglob("*")) if p.is_file()}
    differing = [k for k in sorted(before.keys() | after.keys()) if before.get(k) != after.get(k)]
    report = {"date_local": "2026-10-10", "files_compared": len(before),
        "all_fixture_files_identical": not differing, "differing_paths": differing,
        "scope": "byte-for-byte analytic images, camera truth, COLMAP models, and dataset manifest after rerunning identical generator",
        "preparation_stdout": output}
    (OUT / "evidence" / "fixture_reproducibility.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    tracked_inputs = ["docs/stage.md", "docs/final.md", "docs/other_direction.md", "docs/direction.md",
        "examples/datasets/colmap.py", "examples/utils.py", "examples/simple_trainer.py", "gsplat/strategy/default.py",
        "scripts/download-room-small.py", "docs/SMALL_DATASET_WIN.md"]
    try:
        gpu = run("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader")
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        gpu = f"not available: {type(exc).__name__}"
    provenance = {"date_local": "2026-10-10", "timezone": "Asia/Saigon",
        "revision": run("git", "rev-parse", "HEAD"),
        "initial_status_observed_before_stage0_writes": ["A  docs/direction.md", "?? docs/final.md", "?? docs/other_direction.md", "?? docs/stage.md"],
        "current_status_after_stage0_writes": run("git", "status", "--short").splitlines(),
        "python_version": platform.python_version(), "python_executable": sys.executable,
        "platform": platform.platform(), "gpu_read_only_query": gpu,
        "input_sha256": {path: sha(ROOT / path) for path in tracked_inputs},
        "production_code_changed_by_stage0": False,
        "training_executed_in_stage0": False,
        "existing_room_modified_by_stage0": False}
    (OUT / "evidence" / "workspace_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    assert not differing, differing
    print(json.dumps({"fixture_replay": "passed", "files_identical": len(before)}, indent=2))
