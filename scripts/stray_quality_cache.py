"""Cache selected native Stray Scanner RGB views losslessly with random access."""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from zstandard import ZstdCompressor

from iphone_rescan_audit import expression_sum, reserve_ranges


def build_cache(run, source):
    audit = run / "dataset_audit"
    preflight = json.loads((audit / "preflight.json").read_text())
    width, height = preflight["native_rgb_size"]
    full = width * height * 3
    split = json.loads((run / "splits/frame_split.json").read_text())
    frames = sorted(split["fit"] + split["heldout"])
    groups = reserve_ranges(np.array(frames))
    commands = [f"between(n,{g[0]},{g[-1]})" if len(g) > 1 else f"eq(n,{g[0]})" for g in groups]
    select = audit / "native_rgb.filter"
    select.write_text(f"select='{expression_sum(commands)}'", encoding="ascii")
    path = audit / "native_rgb.zst"
    partial = path.with_suffix(".partial")
    offsets = [0]

    def compress(blob):
        return ZstdCompressor(level=1, write_checksum=True).compress(blob)

    args = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "5", "-ignore_editlist", "1", "-noautorotate",
            "-i", str(source / "rgb.mp4"), "-map", "0:v:0", "-/filter:v", str(select), "-filter_threads", "2",
            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    with subprocess.Popen(args, stdout=subprocess.PIPE) as decoder, partial.open("wb") as output, ThreadPoolExecutor(2) as pool:
        try:
            for first in range(0, len(frames), 8):
                blobs = [decoder.stdout.read(full) for _ in frames[first:first + 8]]
                if any(len(blob) != full for blob in blobs):
                    raise ValueError(f"Short native RGB decode from cached row {first}")
                for block in pool.map(compress, blobs):
                    output.write(block)
                    offsets.append(offsets[-1] + len(block))
                if first % 400 == 0:
                    print(f"Native RGB cache {len(offsets)-1:,}/{len(frames):,}", flush=True)
            if decoder.stdout.read(1) or decoder.wait() != 0:
                raise ValueError("Unexpected FFmpeg exit status or native frame count")
        except BaseException:
            decoder.kill()
            raise
    partial.replace(path)
    np.save(audit / "native_rgb_offsets.npy", np.array(offsets, dtype=np.int64))
    (audit / "native_rgb_frames.json").write_text(json.dumps(frames) + "\n", encoding="utf-8")
    print(f"Cached {len(frames):,} native views in {offsets[-1] / 1e9:.2f} GB.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    build_cache(args.run, args.source)
