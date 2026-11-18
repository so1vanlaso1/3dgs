"""Probe live paused rendering at novel camera offsets, without changing the model."""
import argparse
import hashlib
import importlib.util
import io
import json
import time
from pathlib import Path

import msgspec
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from websockets.sync.client import connect
import viser

spec = importlib.util.spec_from_file_location("pose_audit", Path(__file__).with_name("audit-viewer-pose-restore.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--url", default="ws://localhost:8085")
    ap.add_argument("--seconds", type=float, default=107.51)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    parser = audit.Parser(args.dataset, normalize=True)
    p, r, fov, name = audit.CapturePath(parser).sample(args.seconds)
    poses = [("reference", p, r),
             ("turn_right_5deg", p, r @ Rotation.from_euler("y", 5, degrees=True).as_matrix()),
             ("move_right_0.01", p + 0.01*r[:, 0], r),
             ("move_right_0.03", p + 0.03*r[:, 0], r),
             ("move_forward_0.03", p + 0.03*r[:, 2], r),
             ("reference_restored", p, r)]
    report = dict(url=args.url, dataset=args.dataset, source_frame=name,
                  replay="New client defaults to paused; no replay controls changed",
                  translation_units="normalized world units, not meters", renders={})
    images = {}
    with connect(args.url, subprotocols=[f"viser-v{viser.__version__}"], max_size=64*1024*1024) as ws:
        for label, position, rotation in poses:
            q = Rotation.from_matrix(rotation).as_quat()
            ws.send(msgspec.msgpack.encode(dict(type="ViewerCameraMessage", wxyz=[float(q[3]), *q[:3].tolist()],
                position=position.tolist(), fov=fov, near=0.01, far=1000., image_height=720, image_width=1280,
                look_at=(position+rotation[:, 2]).tolist(), up_direction=(-rotation[:, 1]).tolist())))
            start = time.monotonic()
            frames = []
            while time.monotonic()-start < 5:
                try:
                    raw = ws.recv(timeout=0.25)
                except TimeoutError:
                    continue
                for message in audit.decode(raw):
                    if message["type"] == "BackgroundImageMessage" and message.get("rgb_data"):
                        frames.append((time.monotonic()-start, message["rgb_data"]))
            if not frames:
                raise RuntimeError(f"No frame for {label}")
            pixels = np.asarray(Image.open(io.BytesIO(frames[-1][1])).convert("RGB"))
            images[label] = pixels
            Image.fromarray(pixels).save(out / f"{label}.png")
            report["renders"][label] = dict(position=position.tolist(), rotation=rotation.tolist(),
                frames_received=len(frames), frame_sequence=[dict(seconds=round(t,3), size=list(Image.open(io.BytesIO(data)).size), bytes=len(data)) for t,data in frames], first_frame_seconds=frames[0][0], shape=list(pixels.shape),
                sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
    base = images["reference"]
    for label, pixels in images.items():
        if pixels.shape != base.shape:
            raise RuntimeError("Frames did not settle to identical dimensions")
        report["renders"][label]["mean_abs_change_255"] = float(np.abs(pixels.astype(float)-base).mean())
    report["restored_exactly"] = bool(np.array_equal(base, images["reference_restored"]))
    report["all_novel_views_changed"] = all(not np.array_equal(base, images[label]) for label, _, _ in poses[1:-1])
    canvas = Image.new("RGB", (960, 3*294), "#181818")
    draw = ImageDraw.Draw(canvas)
    for i, (label, pixels) in enumerate(images.items()):
        x, y = (i % 2)*480, (i // 2)*294
        canvas.paste(Image.fromarray(pixels).resize((480, 270)), (x, y+24))
        draw.text((x+8, y+5), label, fill="white")
    canvas.save(out / "comparison.jpg")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["restored_exactly"] or not report["all_novel_views_changed"]:
        raise SystemExit("Camera refresh check failed; inspect evidence")


if __name__ == "__main__":
    main()
