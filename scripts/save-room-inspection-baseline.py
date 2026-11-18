"""Save recorded-camera baseline renders and exact camera metadata."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples'))
from datasets.colmap import Parser
from gsplat.rendering import rasterization

REQUESTS = [(30, 'wide'), (43, 'wide'), (61, 'difficult'), (68, 'difficult'),
            (83, 'reflective'), (97, 'wide'), (99, 'wide'),
            (110, 'reflective and difficult'), (37, 'difficult and reflective')]
DATA = ROOT / 'data/realsense_20261030_134523_clean'
DATASET = DATA / 'rgbd/dataset'
CHECKPOINT = ROOT / 'results/realsense_20261030_134523_120k/train/ckpts/final_post_update.pt'
OUT = ROOT / 'results/room_improve_baseline_inspection_20261103'

@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=False)
    parser = Parser(str(DATASET), factor=1, normalize=True, load_exposure=False)
    frames = {f['name']: f for f in json.loads((DATA / 'capture_manifest.json').read_text(encoding='utf-8'))['frames']}
    timestamps = np.array([frames[n]['timestamp_ns'] for n in parser.image_names], dtype=np.int64)
    times = (timestamps - timestamps.min()) / 1e9
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    s = {k: v.cuda() for k, v in ckpt['splats'].items()}
    colors = torch.cat((s['sh0'], s['shN']), dim=1)
    degree = int(math.sqrt(colors.shape[1]) - 1)
    results = []
    for requested, label in REQUESTS:
        i = int(np.argmin(np.abs(times - requested)))
        name = parser.image_names[i]
        f = frames[name]
        cid = parser.camera_ids[i]
        oldw, oldh = parser.imsize_dict[cid]
        width = 848
        height = max(2, 2 * round(oldh * width / oldw / 2))
        K = parser.Ks_dict[cid].copy()
        K[0] *= width / oldw
        K[1] *= height / oldh
        pose = parser.camtoworlds[i]
        rgb, _, _ = rasterization(s['means'], s['quats'], torch.exp(s['scales']),
            torch.sigmoid(s['opacities']), colors,
            torch.from_numpy(np.linalg.inv(pose)[None]).float().cuda(),
            torch.from_numpy(K[None]).float().cuda(), width, height,
            sh_degree=degree, packed=True)
        pixels = rgb[0].clamp(0, 1).cpu().numpy()
        if not np.isfinite(pixels).all():
            raise ValueError(f'Nonfinite render: {name}')
        filename = f'{requested:03d}s_{name}'
        Image.fromarray((pixels * 255).astype('uint8')).save(OUT / filename)
        entry = dict(requested_seconds=requested, label=label, actual_replay_seconds=float(times[i]),
            difference_seconds=float(times[i]-requested), image_name=name,
            rgb_message_id=f['message_id'], rgb_timestamp_ns=f['timestamp_ns'],
            depth_message_id=f['depth_message_id'], depth_timestamp_ns=f['depth_timestamp_ns'],
            camera_to_world_normalized=pose.tolist(), position_normalized=pose[:3,3].tolist(),
            rotation_camera_to_world=pose[:3,:3].tolist(), intrinsics=K.tolist(),
            vertical_fov_radians=2*math.atan(height/(2*K[1,1])), aspect=width/height,
            width=width, height=height, render_file=filename)
        results.append(entry)
        print(requested, name, round(times[i], 4), filename, flush=True)
    report = dict(source='20261030_134523.db3', checkpoint=str(CHECKPOINT), dataset=str(DATASET),
        timing='seconds relative to the first registered RGB timestamp, matching CapturePath',
        selection='nearest registered RGB frame', coordinate_system='trainer normalized camera-to-world',
        render_settings=dict(width=848, height=height, sh_degree=degree, packed=True,
                             background='black (rasterization default)'), views=results)
    (OUT/'inspection_poses.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    sheet = Image.new('RGB', (3*424, 3*(235+30)), 'white')
    draw = ImageDraw.Draw(sheet)
    for j, view in enumerate(results):
        x, y = (j % 3)*424, (j // 3)*265
        thumb = Image.open(OUT/view['render_file']).resize((424,235))
        sheet.paste(thumb, (x,y+30))
        draw.text((x+5,y+6), f"{view['requested_seconds']}s: {view['label']} (actual {view['actual_replay_seconds']:.2f}s)", fill='black')
    sheet.save(OUT/'contact_sheet.jpg', quality=94)

if __name__ == '__main__':
    main()
