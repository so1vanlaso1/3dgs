"""Save RGB/depth alignment QA panels and audit every cleaned depth file."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('data/realsense_20261030_111354_clean'))
    ap.add_argument('--repair-quantization', action='store_true', help='Re-Z-buffer serialized float32 samples; log every correction')
    a = ap.parse_args()
    manifest = json.loads((a.root/'capture_manifest.json').read_text())
    cal = json.loads((a.root/'calibration_clean.json').read_text())
    width, height = cal['width'], cal['height']
    indices = set(np.linspace(0, len(manifest['frames'])-1, 8, dtype=int).tolist())
    sheet = Image.new('RGB', (1280, 8*380), '#202020')
    draw = ImageDraw.Draw(sheet)
    counts, p95, repaired = [], [], []
    for index, f in enumerate(manifest['frames']):
        with np.load(a.root/'aligned_depth_meters'/(f['name']+'.npz')) as d:
            p, z = d['points'], d['depths']
        assert np.isfinite(p).all() and np.isfinite(z).all() and len(z) >= 1500
        assert (z > 0).all() and (p[:, 0] >= 0).all() and (p[:, 0] < width).all()
        assert (p[:, 1] >= 0).all() and (p[:, 1] < height).all()
        keys = np.rint(p[:, 1]).astype(int)*width+np.rint(p[:, 0]).astype(int)
        if len(keys) != len(np.unique(keys)) and a.repair_quantization:
            order = np.lexsort((z, keys))
            _, first = np.unique(keys[order], return_index=True)
            keep = order[first]
            repaired.append(dict(name=f['name'], removed_duplicate_samples=len(z)-len(keep)))
            p, z, keys = p[keep], z[keep], keys[keep]
            np.savez_compressed(a.root/'aligned_depth_meters'/(f['name']+'.npz'), points=p, depths=z)
        assert len(keys) == len(np.unique(keys)), 'Z buffer produced duplicate target pixels'
        counts.append(len(z)); p95.append(float(np.percentile(z, 95)))
        if index not in indices:
            continue
        image = np.array(Image.open(a.root/'images'/f['name']))
        overlay = image.copy()
        color = cv2.applyColorMap((255*(1-np.clip((z-.3)/5.7, 0, 1))).astype('u1'), cv2.COLORMAP_TURBO)[:, 0, ::-1]
        pix = np.rint(p).astype(int)
        overlay[pix[:, 1], pix[:, 0]] = (image[pix[:, 1], pix[:, 0]]*.25+color*.75).astype('u1')
        pair = Image.fromarray(np.concatenate([image, overlay], axis=1))
        pair.thumbnail((1280, 355))
        row = sorted(indices).index(index)
        sheet.paste(pair, (0, row*380+25))
        draw.text((5, row*380+4), f"{f['name']} | RGB / depth overlay | gap {f['pair_delta_seconds']*1000:.2f} ms | Z .3-6m", fill='white')
    sheet.save(a.root/'depth_alignment_qa.jpg', quality=94)
    if a.repair_quantization:
        (a.root/'depth_quantization_repair.json').write_text(json.dumps(dict(
            reason='Float64-to-float32 half-pixel rounding can merge formerly distinct target pixels',
            policy='Re-Z-buffer in serialized float32 coordinates; keep nearest positive camera Z',
            changed_files=len(repaired), removed_samples=sum(r['removed_duplicate_samples'] for r in repaired), frames=repaired), indent=2))
    report = dict(checked_frames=len(counts), all_depth_samples_finite_positive_in_bounds=True,
                  all_target_depth_pixels_unique=True, saved_samples_min=min(counts), saved_samples_max=max(counts),
                  p95_camera_z_meters_range=[min(p95), max(p95)],
                  visual_qa='depth_alignment_qa.jpg; sparse samples overlaid, depth is not dense or hole-filled',
                  interpretation='Numerical consistency and visual diagnostic; no independent calibration or depth ground truth')
    imu_pairs = {}
    for kind in ('accel', 'gyro'):
        path = a.root/'imu'/f'{kind}.csv'
        if not path.exists():
            continue
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        stamps = np.array([int(r['timestamp_ns']) for r in rows], dtype=np.int64)
        associated = []
        for f in manifest['frames']:
            j = int(np.searchsorted(stamps, f['timestamp_ns']))
            j = min([max(0, j-1), min(len(stamps)-1, j)], key=lambda k: abs(int(stamps[k])-f['timestamp_ns']))
            row = rows[j]
            gap = abs(int(stamps[j])-f['timestamp_ns'])/1e9
            if gap > .02:
                raise ValueError(f'IMU/RGB gap exceeds 20 ms: {kind}, {f["name"]}, {gap}')
            associated.append(dict(name=f['name'], imu_message_id=row['message_id'], gap_seconds=gap,
                                   header_gap_seconds=abs(float(row['header_seconds'])-f['header_seconds']),
                                   x=float(row['x']), y=float(row['y']), z=float(row['z'])))
        with (a.root/'imu'/f'{kind}_frame_associations.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(associated[0]))
            writer.writeheader(); writer.writerows(associated)
        imu_pairs[kind] = dict(associated_frames=len(associated), max_gap_seconds=max(r['gap_seconds'] for r in associated),
                              max_header_gap_seconds=max(r['header_gap_seconds'] for r in associated),
                              method='Nearest timestamp association; no pose fusion')
    report['imu_frame_associations'] = imu_pairs
    (a.root/'clean_audit.json').write_text(json.dumps(report, indent=2))
    print('CLEAN_AUDIT', report, flush=True)


if __name__ == '__main__':
    main()
