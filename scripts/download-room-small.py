"""Fetch official room photographs and COLMAP at a selected resolution."""

import argparse
import io
import json
import zipfile
from pathlib import Path

import pycolmap
import requests
from PIL import Image

URL = "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip"


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.session = requests.Session()
        response = self.session.head(url, timeout=60)
        response.raise_for_status()
        self.size = int(response.headers["Content-Length"])
        self.url, self.position, self.regions = url, 0, []

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        return self.position

    def tell(self):
        return self.position

    def fetch(self, start, length):
        end = min(start + length, self.size) - 1
        with self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120, stream=True) as response:
            response.raise_for_status()
            if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                raise RuntimeError("Server did not honor the requested byte range")
            data = response.content
        if len(data) != end - start + 1:
            raise RuntimeError("Incomplete byte range download")
        self.regions.append((start, data))
        return data

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        for start, data in self.regions:
            offset = self.position - start
            if 0 <= offset and offset + size <= len(data):
                self.position += size
                return data[offset:offset + size]
        data = self.fetch(self.position, max(size, min(65536, self.size - self.position)))
        self.position += size
        return data[:size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-factor", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = (args.output_dir or Path("data/room_small" if args.image_factor == 8 else f"data/room_factor{args.image_factor}")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "dataset_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["source_image_factor"] != args.image_factor:
        raise ValueError("Use a different output directory for each image factor")
    image_folder = "images" if args.image_factor == 1 else f"images_{args.image_factor}"
    image_prefix = f"room/{image_folder}/"
    source = output / "source"
    remote = RemoteZip(URL)
    with zipfile.ZipFile(remote) as archive:
        selected = [entry for entry in archive.infolist() if not entry.is_dir() and
                    (entry.filename.startswith(image_prefix) or entry.filename.startswith("room/sparse/0/"))]
        compressed_bytes = sum(entry.compress_size for entry in selected)
        print(f"Downloading {len(selected)} files: {compressed_bytes / 1024**2:.2f} MiB instead of {remote.size / 1024**3:.2f} GiB", flush=True)
        for prefix in (image_prefix, "room/sparse/0/"):
            entries = [entry for entry in selected if entry.filename.startswith(prefix)]
            if all((source / entry.filename.removeprefix("room/")).exists() for entry in entries):
                print(f"Reusing {prefix}", flush=True)
                continue
            start = min(entry.header_offset for entry in entries)
            end = max(entry.header_offset + 30 + len(entry.filename.encode()) + len(entry.extra) + entry.compress_size + 128 for entry in entries)
            print(f"Fetching {prefix}: {(end - start) / 1024**2:.2f} MiB", flush=True)
            remote.fetch(start, end - start)
            for entry in entries:
                target = source / entry.filename.removeprefix("room/")
                target.parent.mkdir(parents=True, exist_ok=True)
                # ZipFile checks the decompressed data against the archive CRC.
                target.write_bytes(archive.read(entry))

    reconstruction = pycolmap.Reconstruction(str(source / "sparse/0"))
    image_dir = output / "images"
    image_dir.mkdir(exist_ok=True)
    dimensions = {}
    source_images = {path.stem: path for path in (source / image_folder).iterdir() if path.is_file()}
    for image in reconstruction.images.values():
        image_path = source_images[Path(image.name).stem]
        with Image.open(image_path) as pixels:
            dimensions[image.camera_id] = pixels.size
        target = image_dir / image_path.name
        if not target.exists():
            target.hardlink_to(image_path)
        image.name = image_path.name
    for camera_id, (width, height) in dimensions.items():
        reconstruction.cameras[camera_id].rescale(width, height)
    sparse = output / "sparse/0"
    sparse.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(sparse))
    manifest = {
        "source_url": URL, "scene": "room", "source_image_factor": args.image_factor,
        "images": reconstruction.num_reg_images(), "points": reconstruction.num_points3D(),
        "image_dimensions": dimensions, "compressed_dataset_bytes": compressed_bytes,
        "archive_bytes": remote.size,
        "preparation": f"Official {image_folder}; camera intrinsics rescaled to actual dimensions; all poses and points preserved. Use data_factor=1.",
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
