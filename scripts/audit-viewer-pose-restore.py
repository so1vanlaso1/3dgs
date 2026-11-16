"""Exercise the running viewer with exact camera A/B/A messages; save evidence."""
import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import msgspec
import numpy as np
import zstandard
import viser
from PIL import Image
from scipy.spatial.transform import Rotation
from websockets.sync.client import connect

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from capture_path import CapturePath
from datasets.colmap import Parser


def decode(raw):
    n = int.from_bytes(raw[8:16], "little")
    payload = msgspec.msgpack.decode(zstandard.ZstdDecompressor().decompress(raw[16:16+n]))
    buffers, offset = [], 16+n
    for length in payload["binaryBufferLengths"]:
        offset += (-offset) % 8
        buffers.append(raw[offset:offset+length])
        offset += length
    for message in payload["messages"]:
        for key, value in message.items():
            if isinstance(value, dict) and "__binary_index" in value:
                message[key] = buffers[value["__binary_index"]]
    return payload["messages"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--url", default="ws://localhost:8085")
    ap.add_argument("--output", required=True)
    ap.add_argument("--first-time", type=float, default=107.51)
    ap.add_argument("--second-time", type=float, default=228.11)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    parser = Parser(args.dataset, normalize=True)
    path = CapturePath(parser)
    a, b = path.sample(args.first_time), path.sample(args.second_time)
    report = {"dataset": args.dataset, "url": args.url, "poses": {}, "renders": {}}
    for name, pose in ((str(args.first_time), a), (str(args.second_time), b)):
        p, r, fov, frame = pose
        report["poses"][name] = dict(position=p.tolist(), rotation=r.tolist(), fov=fov, frame=frame)
    report["position_distance_normalized"] = float(np.linalg.norm(a[0]-b[0]))
    report["position_distance_source_units"] = float(np.linalg.norm(a[0]-b[0]) / np.linalg.norm(parser.transform[:3, 0]))
    report["orientation_difference_degrees"] = float(np.rad2deg(Rotation.from_matrix(a[1].T @ b[1]).magnitude()))
    images, controls = {}, {}
    with connect(args.url, subprotocols=[f"viser-v{viser.__version__}"], max_size=64*1024*1024) as ws:
        def receive_for(seconds):
            end, backgrounds = time.monotonic()+seconds, []
            while time.monotonic() < end:
                try:
                    raw = ws.recv(timeout=min(0.25, max(0.001, end-time.monotonic())))
                except TimeoutError:
                    continue
                for message in decode(raw):
                    if message["type"] == "BackgroundImageMessage" and message.get("rgb_data"):
                        backgrounds.append(message["rgb_data"])
                    if message["type"] == "GuiSliderMessage":
                        label = message.get("props", {}).get("label")
                        controls[label] = message["uuid"]
            return backgrounds

        def send_pose(pose):
            p, r, fov, _ = pose
            q = Rotation.from_matrix(r).as_quat()
            ws.send(msgspec.msgpack.encode(dict(type="ViewerCameraMessage",
                wxyz=[float(q[3]), *q[:3].tolist()], position=p.tolist(), fov=fov,
                near=0.01, far=1000., image_height=720, image_width=1280,
                look_at=(p+r[:, 2]).tolist(), up_direction=(-r[:, 1]).tolist())))

        send_pose(a)
        receive_for(5)
        if "Capture time (s)" not in controls:
            raise RuntimeError("Capture replay timeline was not found; cannot test replay time independently")
        a_label = f"a_{args.first_time:g}"
        b_label = f"b_{args.second_time:g}"
        restored_label = f"a_restored_at_{args.second_time:g}"
        for label, seconds, pose in ((a_label, args.first_time, a), (b_label, args.second_time, b),
                                      (restored_label, args.second_time, a),
                                      ("b_position_a_direction", args.second_time, (b[0], a[1], a[2], b[3]))):
            if "Capture time (s)" in controls:
                ws.send(msgspec.msgpack.encode(dict(type="GuiUpdateMessage",
                    uuid=controls["Capture time (s)"], updates={"value": seconds})))
                receive_for(0.5)
            send_pose(pose)
            frames = receive_for(4)
            if not frames:
                raise RuntimeError(f"No render received for {label}")
            image = np.asarray(Image.open(io.BytesIO(frames[-1])).convert("RGB"))
            images[label] = image
            Image.fromarray(image).save(out / f"{label}.png")
            report["renders"][label] = dict(frames_received=len(frames), shape=list(image.shape),
                jpeg_sha256=hashlib.sha256(frames[-1]).hexdigest(),
                pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest())
        difference = np.abs(images[a_label].astype(float)-images[restored_label].astype(float))
        report["exact_restore"] = dict(equal=bool(np.array_equal(images[a_label], images[restored_label])),
            max_abs_pixel_difference=float(difference.max()), mean_abs_pixel_difference=float(difference.mean()))
        report["timeline_control_found"] = "Capture time (s)" in controls
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["exact_restore"]["equal"]:
        raise SystemExit("Exact restoration produced different final pixels; inspect saved evidence")


if __name__ == "__main__":
    main()
