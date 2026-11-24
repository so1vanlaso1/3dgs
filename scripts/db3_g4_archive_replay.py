"""Run the B00 arm of the recovered Week 12 driver with the original launcher."""

import argparse
import importlib.util
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch

from week12_comparison import REPO, read, sha, write


def run(source, out):
    expected = read(REPO/"results/week12_comparison_20261119/matte_training_protocol.json")
    for name, digest in expected["source_sha256"].items():
        assert sha(source/name) == digest, name
    # The recovered driver imports its repository root from research_campaign.
    # Load that copy in this process; its child launches the archived trainer normally.
    sys.path.insert(0, str(source/"scripts"))
    sys.modules.pop("research_campaign", None)
    spec = importlib.util.spec_from_file_location("archived_week12", source/"scripts/week12_comparison.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    assert driver.REPO == source
    driver.ARMS = ("B00",)
    env_keys = ("CUDA_PATH", "CUDA_HOME", "MAX_JOBS", "TORCH_CUDA_ARCH_LIST", "NUM_CHANNELS",
                "OMP_NUM_THREADS", "PYTHONPATH", "TORCH_EXTENSIONS_DIR", "TORCH_HOME")
    write(out/"environment.json", dict(
        python=sys.version, platform=platform.platform(), torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version","--format=csv,noheader"],text=True).strip(),
        launcher_sha256=sha(REPO/"scripts/run-school-completion.ps1"),
        env={k:os.environ.get(k) for k in env_keys},
        packages=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True).splitlines()))
    driver.train(out,"matte")
    actual = read(out/"matte_training_protocol.json")
    for key in ("seed","updates","input_sha256","source_sha256"):
        assert actual[key] == expected[key], key
    args = actual["commands"]["B00"].copy()
    old = expected["commands"]["B00"]
    for flag in ("--result-dir","--sensor-depth-dir","--data-dir","--camera-sequence-path"):
        pos = args.index(flag)+1
        if flag != "--result-dir":
            assert Path(args[pos]).samefile(old[pos]), flag
        args[pos] = old[pos]
    assert args == old
    print("Verified archived driver, sources, training command, inputs and checkpoint schedule")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())
