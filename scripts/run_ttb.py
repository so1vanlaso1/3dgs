"""Set up NVIDIA Test-Time Baking and bake an RGB video in one command."""

import argparse
import os
import shutil
import subprocess
import sys
import venv
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".tools" / "ttb"
DVSM_REVISION = "826be87c1a7c8ab3c0385b66b0ceaf8aa71f123b"
PI3_REVISION = "a65e87ee9fb84ee4fd7a15ff8d8a26d16119da99"
DEPENDENCIES = (
    "numpy==1.26.4", "opencv-python-headless==4.11.0.86",
    "transformers==4.57.3", "huggingface_hub>=0.34,<1",
    "einops", "imageio[pyav]", "Pillow", "safetensors", "yacs", "scipy",
    f"git+https://github.com/yyfz/Pi3.git@{PI3_REVISION}",
)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".wmv"}


def run(command, log):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    with subprocess.Popen(
        [str(arg) for arg in command], cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace",
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise RuntimeError(f"{command[0]} exited with status {process.returncode}. See {log.name}")


def setup(cuda_wheel, log):
    cache = CACHE / cuda_wheel
    environment = cache / "venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    source = CACHE / "DVSM"
    if not (source / ".git").exists():
        run(["git", "clone", "https://github.com/NVlabs/DVSM.git", source], log)
    revision = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
    if revision != DVSM_REVISION:
        run(["git", "-C", source, "fetch", "origin", DVSM_REVISION], log)
        run(["git", "-C", source, "checkout", "--detach", DVSM_REVISION], log)

    if not python.is_file():
        venv.EnvBuilder().create(environment)
    stamp = cache / "dependencies.txt"
    requirements = (f"torch==2.7.1+{cuda_wheel}", f"torchvision==0.22.1+{cuda_wheel}", *DEPENDENCIES)
    manifest = "\n".join(requirements) + "\n"
    if not stamp.is_file() or stamp.read_text(encoding="utf-8") != manifest:
        run([python, "-m", "ensurepip", "--upgrade"], log)
        pip = [python, "-m", "pip", "install", "--disable-pip-version-check"]
        run([*pip, "--index-url", f"https://download.pytorch.org/whl/{cuda_wheel}", *requirements[:2]], log)
        run([*pip, *DEPENDENCIES], log)
        run([python, "-m", "pip", "check"], log)
        stamp.write_text(manifest, encoding="utf-8")
    return python, source


def find_capture(capture):
    if capture is None:
        capture = next(
            (p for p in (ROOT / "newscan.zip", ROOT.parent / "newscan.zip") if p.is_file()),
            None,
        )
        if capture is None:
            raise ValueError("Pass the path to your RGB video or newscan.zip.")
    capture = capture.expanduser().resolve()
    if not capture.is_file() or capture.suffix.lower() not in VIDEO_EXTENSIONS | {".zip"}:
        raise ValueError(f"Expected an RGB video or ZIP file, got {capture}")
    if capture.suffix.lower() == ".zip":
        with zipfile.ZipFile(capture) as archive:
            matches = [member.filename for member in archive.infolist()
                       if not member.is_dir() and Path(member.filename).name.lower() == "rgb.mp4"]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one rgb.mp4 in {capture}, found {len(matches)}")
        return capture, matches[0]
    return capture, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, nargs="?", help="RGB video or ZIP with rgb.mp4 (defaults to newscan.zip)")
    parser.add_argument("--out", type=Path, help="New output directory (defaults to a timestamped results/ttb directory)")
    parser.add_argument("--views", type=int, help="Maximum evenly sampled video frames")
    parser.add_argument("--long-edge", type=int, help="TTB input long edge in pixels")
    parser.add_argument("--pose-size", type=int, help="Pi3 input long edge in pixels")
    parser.add_argument("--cuda-wheel", choices=("cu126", "cu128"), default="cu126" if os.name == "nt" else "cu128")
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        parser.error("Use Python 3.10 through 3.12.")
    if shutil.which("git") is None:
        parser.error("Git must be on PATH.")
    if any(value is not None and value < minimum for value, minimum in
           ((args.views, 2), (args.long_edge, 16), (args.pose_size, 28))):
        parser.error("Use at least 2 views, a 16-pixel long edge and a 28-pixel pose size.")
    try:
        capture, member = find_capture(args.capture)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out = (args.out or ROOT / "results" / "ttb" / f"{capture.stem}_{timestamp}").expanduser().resolve()
        out.mkdir(parents=True)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))

    CACHE.mkdir(parents=True, exist_ok=True)
    experiment = out / "experiment_log.md"
    experiment.write_text(
        f"# NVIDIA full Test-Time Baking\n\n"
        f"- Input: `{capture}`\n- CUDA wheel: `{args.cuda_wheel}`\n"
        f"- DVSM source: `{DVSM_REVISION}`\n- Pi3 source: `{PI3_REVISION}`\n",
        encoding="utf-8",
    )
    try:
        with (out / "run.log").open("w", encoding="utf-8") as log:
            python, source = setup(args.cuda_wheel, log)
            command = [python, ROOT / "scripts" / "ttb_bake.py", source, CACHE, capture, out]
            for flag, value in (("--member", member), ("--views", args.views),
                                ("--long-edge", args.long_edge), ("--pose-size", args.pose_size)):
                if value is not None:
                    command.extend([flag, value])
            run(command, log)
    except (Exception, KeyboardInterrupt) as exc:
        with experiment.open("a", encoding="utf-8") as log:
            log.write(f"\n- Run failed: {exc or type(exc).__name__}\n")
        print(f"TTB did not finish. See {out / 'run.log'}", file=sys.stderr)
        return 1
    print(f"Reconstruction: {out / 'scene.recon.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
