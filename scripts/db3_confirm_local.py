"""Evaluate the predeclared selected policy; never reselect after confirmation."""
import argparse
import importlib.util
from pathlib import Path
import cv2
import numpy as np
from db3_acceptance import compare_views, region_comparison
from PIL import Image, ImageDraw

s=importlib.util.spec_from_file_location('evaluator',Path(__file__).with_name('db3_evaluate_pilot.py'))
e=importlib.util.module_from_spec(s);s.loader.exec_module(e);m=e.m


def freeze_structures():
    """Manually reviewed original captures only, before exhibit Gaussian results."""
    p=m.RUN/'confirmation_structure_regions.json'
    if p.exists():raise FileExistsError(p)
    regions={
      'rgb_0050129.png':{'standing_exhibit_target':[522,216,601,318], 'kneeling_exhibit_target':[503,284,577,365],
        'opaque_backdrop_control':[491,25,600,75], 'post_control':[617,371,639,424], 'floor_control':[271,390,401,444]},
      'rgb_0052341.png':{'standing_exhibit_target':[332,154,408,277], 'kneeling_exhibit_target':[327,279,418,393],
        'opaque_wall_control':[538,35,620,120], 'post_control':[548,348,568,444], 'floor_control':[667,367,796,435]},
      'rgb_0054421.png':{'standing_exhibit_target':[350,96,462,288], 'kneeling_exhibit_target':[405,298,558,460],
        'opaque_wall_control':[541,177,622,226], 'post_control':[696,256,720,329], 'floor_control':[578,278,641,304]}}
    m.write(p,dict(frozen_before_exhibit_Gaussian_metrics_review=True,source='Original SDK-rectified captures; single-analyst image review, no fitted model/render used',
        regions_xyxy=regions,min_stable_depth_samples=100,
        claim='Source-view statue interiors and opaque post/wall/floor controls. Depth-edge holes and external metric accuracy remain unverified. Rectangles are inspection regions, not semantic segmentation or reference metrology.',
        thresholds=m.read(m.RUN/'selected_policy_confirmation.json')['gate']))
    for name,boxes in regions.items():
        image=Image.open(m.DATA/'images'/name).convert('RGB');draw=ImageDraw.Draw(image)
        for label,box in boxes.items():
            color='#ff6600' if 'target' in label else '#00ff99'
            draw.rectangle(box,outline=color,width=2);draw.text((box[0],max(0,box[1]-12)),label,fill=color)
        image.save(m.RUN/f'confirmation_regions_{name}')


def confirm(region,supported=False,output_dir=None):
    suffix='_supported_15k' if supported else ''
    out=m.RUN/region/'pilot';paired=m.read(out/('paired_evaluation'+suffix+'.json'));selection=m.read(m.RUN/'selected_policy_confirmation.json')
    evidence=out/('evaluation'+suffix)
    arm=selection['selected_policy'];assert arm=='confidence'
    baseline=paired['results']['baseline'];candidate=paired['results'][arm]
    rules=selection['gate']
    destination=Path(output_dir) if output_dir else out
    result_path=destination/('selected_policy_confirmation_result'+suffix+'.json')
    if result_path.exists():raise FileExistsError('Preserve confirmation; select a fresh output directory')
    destination.mkdir(parents=True,exist_ok=True)
    split=m.read(out/'dataset/experiment_split.json')
    protocol=m.read(out/'evaluation_protocol.json')
    compared=compare_views(candidate,baseline,split['val'],protocol['regions_normalized_xyxy'],rules)
    rows=compared['rows'];appearance=compared['appearance_noninferior']
    physical=[]
    if region=='exhibit':
        structures=m.read(m.RUN/'confirmation_structure_regions.json')
        common=np.load(evidence/'common_mapping.npz')['valid_intersection']
        for name,boxes in structures['regions_xyxy'].items():
            gt=np.load(m.DATA/'depth'/(name+'.npz'))['depth_m'];h,w=gt.shape
            lo=cv2.erode(gt,np.ones((3,3),'u1'));hi=cv2.dilate(gt,np.ones((3,3),'u1'))
            gt=np.where((lo>.3)&(hi<6)&((hi-lo)<np.maximum(.03,.02*gt)),gt,0)
            target=np.array(Image.open(m.DATA/'images'/name))/255.
            cs={a:np.load(evidence/f'{a}_{name}.npz') for a in ('baseline',arm)}
            rgbs={a:np.array(Image.open(evidence/f'{a}_{name}'))/255. for a in ('baseline',arm)}
            for label,(x0,y0,x1,y1) in boxes.items():
                mask=np.zeros((h,w),bool);mask[y0:y1,x0:x1]=True;mask &= common
                recs={a:e.errors(rgbs[a],target,cs[a]['depth_m'],gt,cs[a]['alpha'],mask) for a in cs}
                a,b=recs[arm],recs['baseline']
                physical.append(dict(view=name,region=label,measurements=recs,
                    **region_comparison(a,b,label,rules,structures['min_stable_depth_samples'])))
    result=dict(region=region,selected_policy=paired.get('candidate_directory',arm),selection_frozen_before_confirmation=True,
        exposed_development_followup=supported,
        coarse_regions=rows,manual_source_structure_regions=physical,appearance_noninferior=appearance,
        coarse_quantitative_pass=bool(rows) and all(r['pass_gate'] for r in rows) and appearance,
        manual_structure_quantitative_pass=bool(physical) and all(r['pass_gate'] for r in physical) if region=='exhibit' else None,
        original_depth_improvement_gate_pass=paired['pipeline_against_baseline'][arm]['quantitative_pass'],
        confidence_policy_beats_other_arms=paired['quantitative_promotion_pass'],
        visual_review_complete=False,absolute_geometry_accuracy=False,whole_recording_accepted=False)
    result['input_sha256']={str(p.relative_to(m.ROOT)):m.sha(p) for p in
        (out/('paired_evaluation'+suffix+'.json'),m.RUN/'selected_policy_confirmation.json',out/'evaluation_protocol.json',out/'dataset/experiment_split.json')}
    m.write(result_path,result)
    print('CONFIRMATION',region,'coarse',result['coarse_quantitative_pass'],'structures',result['manual_structure_quantitative_pass'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['freeze','confirm']);p.add_argument('--region',choices=['matte','exhibit'],default='exhibit');p.add_argument('--supported',action='store_true');p.add_argument('--output-dir',type=Path);a=p.parse_args()
    if a.stage=='freeze':freeze_structures()
    else:confirm(a.region,a.supported,a.output_dir)
