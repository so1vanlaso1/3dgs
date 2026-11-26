"""Build the frozen A05-Abs training inputs from the original newscan.zip."""

import argparse
from hashlib import file_digest
from pathlib import Path
from shutil import copyfile, which
import subprocess
import sys
from zipfile import ZipFile

import cv2

from iphone_rescan_alignment import check
from iphone_rescan_audit import audit, write_json


ARCHIVE_SHA256 = "ea5f874767dea37a950a8accaca69b3c6caff186f6149fb3feb38aaff63e35a6"
CAPTURE = "e1cfcfffe4"
SCRIPTS = Path(__file__).resolve().parent


def prepare(archive, output):
    for tool in ("ffmpeg", "ffprobe"):
        if which(tool) is None:
            raise SystemExit(f"{tool} must be on PATH")
    with archive.open("rb") as file:
        digest = file_digest(file, "sha256").hexdigest()
    if digest != ARCHIVE_SHA256:
        raise SystemExit(f"{archive} does not match the audited newscan.zip (SHA-256 {digest})")

    output.mkdir(parents=True, exist_ok=False)
    metadata = output / "dataset_audit"
    metadata.mkdir()
    write_json(metadata / "archive_sha256.json", {
        "Algorithm": "SHA256", "Hash": digest.upper(), "Path": str(archive.resolve()),
    })
    print(f"Extracting {archive} to {output / 'raw'}", flush=True)
    with ZipFile(archive) as capture:
        capture.extractall(output / "raw")
    source = output / "raw" / CAPTURE
    video = str(source / "rgb.mp4")

    pts = subprocess.check_output([
        "ffprobe", "-v", "error", "-ignore_editlist", "1", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time", "-of", "csv=p=0", video,
    ], text=True)
    timestamps = sorted(pts.splitlines(), key=float)
    (metadata / "video_pts_unedited.csv").write_text(
        "\n".join(timestamps) + "\n", encoding="ascii",
    )
    cv2.setNumThreads(2)
    print("Building the frozen split and native RGB cache", flush=True)
    audit(output, source, 1920, 1440, cache=True)
    copyfile(SCRIPTS / "newscan_evaluation_regions.json", output / "splits/evaluation_regions.json")
    print("Auditing RGB and camera alignment", flush=True)
    check(output, source, [118, 838, 8275, 10022, 11582, 13142])
    print("Fusing the 10 mm TSDF and exporting cameras", flush=True)
    subprocess.run([
        sys.executable, str(SCRIPTS / "stray_scanner_reconstruct.py"), str(source),
        str(output / "geometry"), "--video-pts", str(metadata / "video_pts_unedited.csv"),
        "--frame-split", str(output / "splits/frame_split.json"),
        "--rgb-frame-counter", "--only-10mm",
    ], check=True)
    print(f"A05-Abs inputs ready in {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path, help="New directory for prepared data")
    args = parser.parse_args()
    prepare(args.archive, args.output)
