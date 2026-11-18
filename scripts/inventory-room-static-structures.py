"""Build a visually annotated, tentative structure inventory from saved views."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/realsense_20261030_134523_clean'
OUT=ROOT/'results/room_improve_static_inventory_20261103'
PROJS=[ROOT/'results/room_improve_structure_projections_20261103',ROOT/'results/room_improve_target_projections_20261103']
# Manually chosen after visual inspection. Bounds cover objects and backgrounds, not masks.
CASES=[
 ('kiosk_pedestal',0,(165,315,340,463),'static opaque pedestal; screen can reflect; prior edge has holes'),
 ('black_box',143,(200,335,287,462),'static box; adjacent wall; selected boundary has broad transition'),
 ('doorway_frame',131,(240,80,420,423),'static frame; bright exterior clipped; distant depth unreliable'),
 ('wall_display_panel',131,(435,160,545,355),'panel is mostly planar; glare and texture can cause false edges'),
 ('exhibit_backdrop',191,(455,130,628,355),'stationary painted backdrop; many matches approximately planar'),
 ('barrier_post',191,(590,310,632,450),'static thin post; depth edges and occlusion are difficult'),
 ('standing_figure',239,(130,100,305,385),'static exhibit figure; curved surface and silhouette holes'),
 ('freestanding_case',443,(190,0,600,463),'static enclosure; glass/reflections and very near depth need caution'),
 ('door_column',443,(610,0,710,463),'static door column; clipped exterior and occlusions'),
 ('floor_control',191,(160,360,400,460),'static floor control; mostly planar, weak 3D constraint alone'),
]
def main():
    OUT.mkdir(parents=True,exist_ok=False);manifest=json.loads((DATA/'capture_manifest.json').read_text());by={f['name']:f for f in manifest['frames']};first=manifest['frames'][0]['timestamp_ns'];rows=[]
    for label,num,box,note in CASES:
        name=f'frame_{num:06}.png';source=next(p/(name+'.dense_depth.npz') for p in PROJS if (p/(name+'.dense_depth.npz')).exists())
        with np.load(source) as f:z=f['depth_m']
        image=Image.open(DATA/'images'/name).convert('RGB');draw=ImageDraw.Draw(image);draw.rectangle(box,outline='yellow',width=4)
        image.save(OUT/(label+'.png'));x0,y0,x1,y1=box;sub=z[y0:y1,x0:x1];valid=sub[sub>0];f=by[name]
        rows.append(dict(label=label,image_name=name,replay_seconds=(f['timestamp_ns']-first)/1e9,rgb_message_id=f['message_id'],depth_message_id=f['depth_message_id'],
            rgb_timestamp_ns=f['timestamp_ns'],depth_timestamp_ns=f['depth_timestamp_ns'],roi_xyxy=box,projected_valid_fraction=float((sub>0).mean()),
            median_valid_camera_z_m=float(np.median(valid)) if valid.size else None,notes=note,annotated_image=label+'.png'))
    sheet=Image.new('RGB',(2*626,5*375),'white');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(rows):
        x=i%2*626;y=i//2*375;draw.text((x+5,y+7),f"{r['label']} at {r['replay_seconds']:.2f}s ({r['image_name']})",fill='black')
        sheet.paste(Image.open(OUT/r['annotated_image']).resize((626,347)),(x,y+28))
    sheet.save(OUT/'contact_sheet.jpg',quality=92)
    (OUT/'inventory.json').write_text(json.dumps(dict(structures=rows,limitations='Rectangles include backgrounds and multiple surfaces. ROI depth valid fraction and median are not object accuracy or masks. Candidate static identity, selected after inspection. Reflective/dynamic elements require exclusion.'),indent=2),encoding='utf-8')
    print('saved',len(rows),'structures')
if __name__=='__main__':main()
