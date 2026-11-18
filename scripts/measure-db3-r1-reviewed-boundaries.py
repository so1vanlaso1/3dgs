"""Record analyst-reviewed physical candidates and uncertainty, including failures.

Intervals were read from the RGB crops after side-by-side inspection. They are
not blind independent annotations. No correction is fitted to these observations.
Depth transition bands are an uncertainty diagnostic, never geometric truth.
"""
import importlib.util
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('r1',ROOT/'scripts/run-db3-r0-r1.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
OUT=m.RUN/'alignment_review'
# frame, static physical boundary, axis, scan coordinate, old RGB interval,
# SDK RGB interval, two depth-interior intervals, transition search interval.
READINGS=[
    (3,'kiosk pedestal left outer contour','x',390,(144,145),(136,137),(118,128),(145,155),(128,145)),
    (3,'kiosk pedestal left outer contour','x',410,(141,142),(132,133),(112,122),(140,150),(122,140)),
    (3,'kiosk pedestal left outer contour','x',430,(147,148),(135,136),(118,128),(145,155),(128,145)),
    (27,'opaque box left outer contour','x',390,(147,148),(137,138),(118,128),(145,155),(128,145)),
    (27,'opaque box left outer contour','x',410,(149,150),(139,141),(118,128),(147,157),(128,147)),
    (27,'opaque box left outer contour','x',430,(150,151),(140,141),(120,130),(147,157),(130,147)),
    (203,'standing exhibit head top','y',350,(147,148),(146,147),(135,143),(153,159),(143,153)),
]


def main():
    m.write(OUT/'reviewed_rgb_annotations.json',dict(
        provenance='One analyst visual interval reading from saved coordinate-grid crops, assisted by scanline gradients. Both remaps and depth were exposed during review; no blind/repeated independent annotation claim.',
        annotations=[dict(frame=f'frame_{n:06d}.png',physical_feature=f,axis=a,scan_coordinate=y,
            old_rgb_interval_px=o,sdk_rgb_interval_px=c,depth_interiors=[l,r],depth_search=search)
            for n,f,a,y,o,c,l,r,search in READINGS]))
    rows=[]
    for n,feature,axis,fixed,old,new,left,right,search in READINGS:
        name=f'frame_{n:06d}.png'
        with np.load(OUT/(name+'.raw_depth.npz')) as a:z=a['depth_m']
        line=z[fixed,:] if axis=='x' else z[:,fixed]
        l=line[left[0]:left[1]];r=line[right[0]:right[1]];l=l[l>0];r=r[r>0]
        lo,hi=search;x=np.arange(lo,hi);values=line[lo:hi];zl,zr=np.median(l),np.median(r)
        fraction=(values-zl)/(zr-zl);valid=values>0
        a=x[valid&(fraction>=.1)];b=x[valid&(fraction>=.9)]
        assert len(a) and len(b) and abs(zl-zr)>.15
        start=int(a[0]);end=int(b[b>=start][0]);holes=int((line[start:end+1]<=0).sum())
        entry=dict(frame=name,feature=feature,axis=axis,scan_coordinate=fixed,
            old_rgb_interval_px=old,sdk_rgb_interval_px=new,depth_10_90_observed_band_px=[start,end],
            band_width_px=end-start,missing_samples_inside_band=holes,interior_depths_m=[float(zl),float(zr)])
        for label,rgb in [('old',old),('sdk',new)]:
            entry[label+'_signed_depth_band_minus_rgb_interval_px']=[start-rgb[1],end-rgb[0]]
            entry[label+'_distance_between_observed_intervals_px']=max(start-rgb[1],rgb[0]-end,0)
        entry['decision']='coarse interval consistency only; precise gate unresolved (broad band/holes or unreplicated annotation)'
        rows.append(entry)
    summaries=[]
    for feature in dict.fromkeys(r['feature'] for r in rows):
        group=[r for r in rows if r['feature']==feature]
        summaries.append(dict(feature=feature,rows=len(group),
            old_median_distance_to_depth_band_px=float(np.median([r['old_distance_between_observed_intervals_px'] for r in group])),
            sdk_median_distance_to_depth_band_px=float(np.median([r['sdk_distance_between_observed_intervals_px'] for r in group])),
            max_depth_band_width_px=max(r['band_width_px'] for r in group),rows_with_missing_depth=sum(r['missing_samples_inside_band']>0 for r in group)))
    m.write(OUT/'reviewed_boundary_measurements.json',dict(rows=rows,summary=summaries,
        tolerance_policy='A precise <2 px local physical gate requires narrow, valid transitions and replicated annotations; current bands are wider or have holes. Interval distance 0 means overlap, not zero true alignment error.',
        local_decision='SDK mode improves coarse near opaque kiosk/box interval compatibility; use as R2 candidate. Do not certify calibration or every scene region. No evidence here supports a global pixel shift.',
        rejected_or_unresolved=[
            dict(region='145 box/wall',reason='rotation-gap diagnostic 1.18 px; not low motion, ambiguous transition'),
            dict(region='197 and 200 head',reason='rotation-gap diagnostic 2.03/1.05 px; narrow timing effect not separable from edge uncertainty'),
            dict(region='203 head',reason='broader depth transition, missing samples; descriptive control only'),
            dict(region='204 rails',reason='Depth shape broader/fragmented relative to thin post; invalid correspondence for fitting'),
            dict(region='131/183 doorway',reason='Only 44.1%/14.8% valid projected depth in declared ROIs; distant/exterior unknown'),
            dict(region='329/443 case',reason='Glass/reflections, near-surface holes and ambiguous background returns'),
            dict(region='471 end',reason='Partial exterior coverage and depth holes; baseline context retained')],
        precise_physical_alignment_gate_pass=False))
    # Compact deliverable at matched coordinates: this is input alignment, not a
    # new Gaussian reconstruction. Use nearest-neighbor enlargement for evidence.
    sheet=Image.new('RGB',(1050,570),'white');d=ImageDraw.Draw(sheet)
    for j,(num,box,label) in enumerate([(3,(110,380,160,445),'Kiosk'),(27,(120,380,165,445),'Opaque box')]):
        x0,y0,x1,y1=box
        for k,(root,title) in enumerate([(m.BASE,'Historical RGB'),(m.DATA,'SDK RGB')]):
            im=Image.open(root/'images'/f'frame_{num:06d}.png').convert('RGB').crop(box).resize((250,325),Image.Resampling.NEAREST)
            x=j*525+k*255;sheet.paste(im,(x,55));d.text((x+4,7),label+' | '+title,fill='black')
            with np.load(OUT/f'frame_{num:06d}.png.raw_depth.npz') as a:z=a['depth_m']
            for r in [r for r in rows if r['frame']==f'frame_{num:06d}.png']:
                yy=55+(r['scan_coordinate']-y0)*5
                # RGB annotation cyan, observed depth band magenta.
                interval=r[('old' if k==0 else 'sdk')+'_rgb_interval_px']
                rx=x+(sum(interval)/2-x0)*250/(x1-x0)
                d.ellipse((rx-3,yy-3,rx+3,yy+3),fill='cyan')
                for bx in r['depth_10_90_observed_band_px']:
                    xx=x+(bx-x0)*250/(x1-x0);d.ellipse((xx-3,yy-3,xx+3,yy+3),fill='magenta')
        summary=summaries[j]
        d.text((j*525+5,400),f"Median distance to observed depth band: {summary['old_median_distance_to_depth_band_px']:g} -> {summary['sdk_median_distance_to_depth_band_px']:g} px",fill='black')
        d.text((j*525+5,425),'Cyan: reviewed RGB interval midpoint',fill='black')
        d.text((j*525+5,448),'Magenta: 10-90% depth band endpoints',fill='black')
    d.text((5,490),'Input alignment comparison only. Broad/missing depth limits precision.',fill='black')
    d.text((5,515),'Zero distance means overlapping intervals; it does not establish zero calibration error.',fill='black')
    d.text((5,540),'Same exposed recording; no fitted correction or new trained model.',fill='black')
    sheet.save(OUT/'local_input_comparison.png')
    print(summaries,flush=True)


if __name__=='__main__':main()
