"""Build visual evidence for the room's selected recorded-camera views."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/realsense_20261030_134523_clean'
BASE = ROOT / 'results/room_improve_baseline_inspection_20261103'
OUT = ROOT / 'results/room_improve_region_inventory_20261103'

def label(im, caption):
    tile = Image.new('RGB', (417, 260), 'white')
    tile.paste(im.resize((417, 232)), (0, 28))
    ImageDraw.Draw(tile).text((4, 5), caption, fill='black')
    return tile

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    views = json.loads((BASE/'inspection_poses.json').read_text(encoding='utf-8'))['views']
    frames = json.loads((DATA/'capture_manifest.json').read_text(encoding='utf-8'))['frames']
    lookup = {f['name']: i for i, f in enumerate(frames)}
    rows=[]
    for view in views:
        i = lookup[view['image_name']]
        frame = frames[i]
        rgb = Image.open(DATA/'images'/frame['name']).convert('RGB')
        render = Image.open(BASE/view['render_file']).convert('RGB')
        with np.load(DATA/'aligned_depth_meters'/(frame['name']+'.npz')) as d:
            xy, z = d['points'], d['depths']
        assert len(xy)==len(z) and np.isfinite(xy).all() and np.isfinite(z).all()
        # The stored points are a subsample, not a dense depth map. Color denotes depth, not error.
        q = np.clip((z - 0.3)/5.7,0,1)
        cmap = cv2.applyColorMap((q*255).astype('uint8').reshape(-1,1), cv2.COLORMAP_TURBO).reshape(-1,3)[:,::-1]
        overlay=np.asarray(rgb).copy()
        for (x,y),color in zip(xy,cmap):
            xx, yy = round(float(x)), round(float(y))
            if 0 <= xx < overlay.shape[1] and 0 <= yy < overlay.shape[0]:
                overlay[yy,xx]=color
        before, after = frames[max(i-3,0)], frames[min(i+3,len(frames)-1)]
        imgs=[label(rgb,'Prepared source RGB'),label(render,'120k Gaussian render'),
              label(Image.fromarray(overlay),'Sampled depth overlay: color is distance'),
              label(Image.open(DATA/'images'/before['name']).convert('RGB'),f"Earlier: {before['name']}"),
              label(Image.open(DATA/'images'/after['name']).convert('RGB'),f"Later: {after['name']}")]
        canvas=Image.new('RGB',(417*3, 260*2),'white')
        for j,img in enumerate(imgs): canvas.paste(img,((j%3)*417,(j//3)*260))
        name=f"{view['requested_seconds']:03d}s_comparison.jpg"
        canvas.save(OUT/name,quality=94)
        rows.append(dict(requested_seconds=view['requested_seconds'],label=view['label'],
            image_name=frame['name'],actual_replay_seconds=view['actual_replay_seconds'],
            rgb_message_id=frame['message_id'],timestamp_ns=frame['timestamp_ns'],
            comparison=name,earlier_name=before['name'],earlier_offset_s=(before['timestamp_ns']-frame['timestamp_ns'])/1e9,
            later_name=after['name'],later_offset_s=(after['timestamp_ns']-frame['timestamp_ns'])/1e9,
            depth_samples=len(z), sampled_depth_percentiles_m=np.percentile(z,[10,50,90]).tolist(),
            alignment_pair_gap_ms=frame['pair_delta_seconds']*1000))
    (OUT/'inventory.json').write_text(json.dumps({'views':rows,
        'depth_note':'Stored sampled aligned points; color indicates distance. Not a dense validity map or measured alignment error.',
        'neighbor_note':'Frames three selected indices before and after, not raw 30 FPS frames.'},indent=2),encoding='utf-8')
    print('Saved',len(rows),'comparison sheets')

if __name__ == '__main__':
    main()
