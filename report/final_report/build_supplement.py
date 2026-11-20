"""Package the report, indexed measurements and standalone Room reproduction sources."""

import argparse
import hashlib
from pathlib import Path
import re
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile


REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[1]
ARCHIVE_NAME = "final_report_supplement.zip"


def digest(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def collect():
    index = REPORT / "EVIDENCE_INDEX.md"
    files = {"EVIDENCE_INDEX.md": index}
    for name in re.findall(r"\]\(([^)]+)\)", index.read_text(encoding="utf-8")):
        files[name] = ROOT / name
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "gsplat", "examples", "README.md", "LICENSE",
         "MANIFEST.in", "pyproject.toml", "setup.py", "config.yaml"],
        cwd=ROOT, text=True,
    ).split("\0")
    for name in tracked:
        if name and (ROOT / name).is_file():
            files[name] = ROOT / name
    glm = ROOT / "gsplat/cuda/csrc/third_party/glm"
    glm_paths = [glm / "copying.txt", *(p for p in (glm / "glm").rglob("*") if p.is_file())]
    if not (glm / "glm/glm.hpp").is_file():
        raise FileNotFoundError("Initialize the pinned GLM submodule before packaging")
    report_paths = [
        *REPORT.glob("*.tex"), REPORT / "build_evidence.py",
        *(p for folder in ("evidence", "figures") for p in (REPORT / folder).rglob("*") if p.is_file()),
    ]
    script_paths = [
        ROOT / "scripts" / name for name in (
            "download-room-small.py", "research_campaign.py", "run-school-completion.ps1",
            "final_report_followup.py", "week12_mesh_review.py",
        )
    ]
    for path in glm_paths + report_paths + script_paths:
        files[path.relative_to(ROOT).as_posix()] = path
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing supplement files: {missing}")
    return files


def build(destination):
    files = collect()
    staged = destination.with_suffix(".zip.tmp")
    with ZipFile(staged, "w", compression=ZIP_DEFLATED) as archive:
        manifest = []
        for name, path in sorted(files.items()):
            with path.open("rb") as stream:
                manifest.append(f"{digest(stream)}  {name}")
            archive.write(path, name)
        archive.writestr("MANIFEST.sha256", "\n".join(manifest) + "\n")
    with ZipFile(staged) as archive:
        for line in archive.read("MANIFEST.sha256").decode("utf-8").splitlines():
            expected, name = line.split("  ", 1)
            with archive.open(name) as stream:
                if digest(stream) != expected:
                    raise ValueError(f"Invalid archived digest for {name}")
    staged.replace(destination)
    print(f"Packaged and verified {len(files)} files at {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT / ARCHIVE_NAME)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
