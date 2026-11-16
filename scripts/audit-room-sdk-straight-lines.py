"""Compare curvature of manually selected architectural edges after RGB remapping."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/room_improve_sdk_straight_lines_20261103';SRC=ROOT/'results/room_improve_sdk_remap_comparison_20261103';EXTRA=ROOT/'results/room_improve_sdk_remap_extra_20261103'
# Selected after viewing remaps. Bands may include texture; manual review required.
CASES=[('map_panel',1,EXTRA,'nearby_check',(365,405),(30,340)),('map_panel',2,EXTRA,'nearby_check',(350,390),(30,340)),('map_panel',3,SRC,'development',(320,360),(30,340)),('exhibit_backdrop',191,SRC,'development',(440,475),(45,245)),('exhibit_backdrop',195,SRC,'check',(400,445),(45,245))]
def main():
    OUT.mkdir(parents=True,exist_ok=False);rows=[];summaries=[]
    for label,num,root,group,(x0,x1),(y0,y1) in CASES:
        name=f'frame_{num:06}.png';panel=Image.new('RGB',((x1-x0)*4*2,(y1-y0)*2+35),'white');draw=ImageDraw.Draw(panel)
        for j,mode in enumerate(('baseline','sdk_consistent')):
            img=np.asarray(Image.open(root/f'{group}_{name}.{mode}.png').convert('RGB'));gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY);grad=np.abs(cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3));ys=np.arange(y0,y1,4);xs=np.array([x0+int(np.argmax(grad[y,x0:x1])) for y in ys]);strength=grad[ys,xs];keep=strength>60;coef=np.polyfit(ys[keep],xs[keep],1) if keep.sum()>=8 else None
            panel.paste(Image.fromarray(img[y0:y1,x0:x1]).resize(((x1-x0)*4,(y1-y0)*2)),(j*(x1-x0)*4,35));draw.text((j*(x1-x0)*4+3,8),mode,fill='black')
            for yy,xx,g in zip(ys,xs,keep):
                px=j*(x1-x0)*4+(xx-x0)*4;py=35+(yy-y0)*2;draw.ellipse((px-2,py-2,px+2,py+2),fill='cyan' if g else 'red')
                rows.append(dict(label=label,frame=name,group=group,mapping=mode,y=int(yy),x=int(xx),gradient=float(grad[yy,xx]),kept=bool(g),line_residual_px=float(abs(xx-np.polyval(coef,yy))) if g and coef is not None else ''))
            residual=np.abs(xs[keep]-np.polyval(coef,ys[keep])) if coef is not None else np.array([])
            summaries.append(dict(label=label,frame=name,group=group,mapping=mode,rows=int(keep.sum()),median_px=float(np.median(residual)) if len(residual) else None,p90_px=float(np.percentile(residual,90)) if len(residual) else None))
        panel.save(OUT/f'{label}_{num:06}.png')
    with (OUT/'rows.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (OUT/'report.json').write_text(json.dumps(dict(summary=summaries,note='Bands and lines selected after viewing both variants. Strongest x-gradient per row; linear fit on same rows, no independent ground truth. Lines must correspond to same physical straight structure; texture, painting, reflections, motion and parallax can invalidate results. Rows correlated; do not generalize.'),indent=2),encoding='utf-8')
    for q in summaries:print(q)
if __name__=='__main__':main()
