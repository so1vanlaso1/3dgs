"""Enlarge fixed SDK remap and depth overlays with coordinate grids."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'results/room_improve_sdk_remap_comparison_20261103';DEP=ROOT/'results/room_improve_sdk_candidate_depth_20261103';OUT=ROOT/'results/room_improve_sdk_crop_review_20261103'
CASES=[('development',3,(90,350,260,463)),('check',5,(0,350,130,463)),('development',27,(100,350,310,463)),('check',29,(0,350,185,463)),('development',191,(350,250,650,450)),('check',195,(320,230,630,430)),('development',239,(100,90,310,300)),('check',243,(70,80,300,300))]
def main():
    OUT.mkdir(parents=True,exist_ok=False);rows=[]
    for group,num,box in CASES:
        n=f'frame_{num:06}.png';x0,y0,x1,y1=box;size=((x1-x0)*3,(y1-y0)*3);panel=Image.new('RGB',(size[0]*2,size[1]+35),'white');draw=ImageDraw.Draw(panel)
        with np.load(DEP/(n+'.dense_depth.npz')) as f:z=f['depth_m']
        for j,typ in enumerate(('baseline','sdk_consistent')):
            rgb=np.asarray(Image.open(SRC/f'{group}_{n}.{typ}.png').convert('RGB')).copy();a,b=z[:, :-1],z[:,1:];edge=(a>0)&(b>0)&(abs(a-b)>np.maximum(.06,.025*np.minimum(a,b)));rgb[:,:-1][edge]=[255,0,255]
            panel.paste(Image.fromarray(rgb[y0:y1,x0:x1]).resize(size,Image.Resampling.NEAREST),(j*size[0],35));draw.text((j*size[0]+4,8),typ+'; magenta = adjacent valid depth jump',fill='black')
            for x in range((x0//20+1)*20,x1,20):
                xx=j*size[0]+(x-x0)*3;draw.line((xx,35,xx,size[1]+35),fill=(255,255,0),width=1);draw.text((xx+2,37),str(x),fill=(255,255,0))
            for y in range((y0//20+1)*20,y1,20):
                yy=(y-y0)*3+35;draw.line((j*size[0],yy,(j+1)*size[0],yy),fill=(255,255,0),width=1);draw.text((j*size[0]+2,yy+1),str(y),fill=(255,255,0))
        name=f'{group}_{num:06}.png';panel.save(OUT/name);rows.append(dict(group=group,frame=num,crop=box,file=name))
    (OUT/'report.json').write_text(json.dumps(dict(crops=rows,note='Crops/coordinates selected after candidate inspection; magenta is depth jump, not proven corresponding RGB edge. Check frames share nearby scenes.'),indent=2),encoding='utf-8');print('saved',len(rows),'crops')
if __name__=='__main__':main()
