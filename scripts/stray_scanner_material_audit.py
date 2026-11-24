"""Audit hand-marked material regions in Stray Scanner exhibit 710d9d02a3."""
import argparse
import csv
from pathlib import Path
import subprocess

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

from stray_scanner_reconstruct import rays_for


# Native 256 x 192 RGB-D raster. Boxes exclude visible surface boundaries
# where possible. In the grid region each ray may hit a wire or background.
REGIONS = {
    99: {"floor": (230, 119, 249, 163), "wood": (168, 100, 197, 130),
         "glass": (110, 106, 128, 141)},
    954: {"floor": (226, 43, 248, 88), "wood": (153, 54, 202, 120),
          "glass": (110, 55, 135, 124)},
    1800: {"wire_grid_and_background": (181, 117, 226, 166)},
    2907: {"wood": (160, 131, 201, 161), "glass": (97, 129, 125, 161)},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source, output = args.source, args.output
    split = np.load(output / "camera_split.npz")
    if not set(REGIONS).issubset(split["heldout"]):
        raise ValueError("Material regions must use held-out frames")
    width, height = 256, 192
    rgb = subprocess.check_output([
        "ffmpeg", "-v", "error", "-ignore_editlist", "1", "-noautorotate",
        "-i", str(source / "rgb.mp4"), "-map", "0:v:0", "-vf",
        "select=" + "+".join(f"eq(n\\,{i})" for i in REGIONS) + ",scale=256:192:flags=area",
        "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ])
    rgb = np.frombuffer(rgb, dtype=np.uint8).reshape(len(REGIONS), height, width, 3)
    sheet = Image.new("RGB", (2 * width * 2, 2 * (height * 2 + 24)), "#142733")
    colors = {"floor": "#3ff2db", "wood": "#b7fb66", "glass": "#ffd473",
              "wire_grid_and_background": "#edb5ff"}
    for panel, (frame, boxes) in enumerate(REGIONS.items()):
        view = Image.fromarray(rgb[panel]).resize((2 * width, 2 * height))
        draw = ImageDraw.Draw(view)
        for name, (x0, y0, x1, y1) in boxes.items():
            draw.rectangle((2*x0, 2*y0, 2*(x1-1), 2*(y1-1)), outline=colors[name], width=2)
            draw.text((2*x0, 2*y0-14), name, fill=colors[name], stroke_width=2, stroke_fill="black")
        origin_x = panel % 2 * 2 * width
        origin_y = panel // 2 * (2 * height + 24)
        ImageDraw.Draw(sheet).text((origin_x + 10, origin_y + 7), f"Frame {frame}", fill="white")
        sheet.paste(view, (origin_x, origin_y + 24))
    sheet.save(output / "material_regions.png")

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    rows = []
    for model in ("tsdf_10mm", "tsdf_20mm"):
        mesh = o3d.io.read_triangle_mesh(str(output / f"{model}.ply"))
        scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        for frame, boxes in REGIONS.items():
            fx, fy, cx, cy = split["intrinsics_depth_px"][frame]
            direction = np.stack(((u - cx) / fx, (v - cy) / fy, np.ones_like(u)), axis=-1)
            rays = rays_for(frame, {frame: direction}, split["camera_to_world_rotation"], split["positions_m"])
            predicted = scene.cast_rays(o3d.core.Tensor(rays), nthreads=4)["t_hit"].numpy()
            depth = cv2.imread(str(source / "depth" / f"{frame:06}.png"), -1)
            conf = cv2.imread(str(source / "confidence" / f"{frame:06}.png"), -1)
            for material, (x0, y0, x1, y1) in boxes.items():
                area = np.s_[y0:y1, x0:x1]
                valid = (conf[area] == 2) & (depth[area] >= 250) & (depth[area] <= 5000)
                hit = valid & np.isfinite(predicted[area])
                signed = predicted[area][hit] * 1000 - depth[area][hit]
                rows.append({
                    "model": model, "frame": frame, "material": material,
                    "x0": x0, "y0": y0, "x1_exclusive": x1, "y1_exclusive": y1,
                    "pixels": valid.size, "eligible": int(valid.sum()), "hits": int(hit.sum()),
                    "within_50mm": int((abs(signed) <= 50).sum()),
                    "signed_median_mm": float(np.median(signed)) if len(signed) else None,
                    "absolute_median_mm": float(np.median(abs(signed))) if len(signed) else None,
                })
    with (output / "material_regions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

