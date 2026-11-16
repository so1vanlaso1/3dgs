"""Save matching local viewer/camera/provenance metadata without global merging."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'examples')]
s=importlib.util.spec_from_file_location('local',ROOT/'scripts/db3_local_geometry.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
from datasets.colmap import Parser
from capture_path import CapturePath


def main():
    source=m.read(m.DATA/'manifest.json');byname={f['name']:f for f in source['frames']}
    records={}
    for region in ('matte','exhibit'):
        out=m.RUN/region/'pilot';dataset=out/'dataset';parser=Parser(str(dataset),factor=1,normalize=True,load_exposure=False)
        frames=[dict(byname[n],message_id=byname[n]['rgb_message_id']) for n in parser.image_names]
        m.write(dataset/'capture_manifest.json',dict(source=source['source'],source_sha256=source['source_sha256'],frames=frames,
            note='Local submap only. Timestamp replay includes geometry-only tracking views and evaluation localizations, not only Gaussian training views.'))
        replay=CapturePath(parser);assert replay.timing_source.startswith('Recorded timestamps')
        for t in (0,replay.duration/2,replay.duration):assert np.isfinite(replay.sample(t)[0]).all()
        selected='confidence';suffix=''
        original_confirmation=m.read(out/'selected_policy_confirmation_result.json')
        prior_visual=m.read(m.RUN/'visual_review_7000.json')
        accepted=bool(region=='matte' and original_confirmation['coarse_quantitative_pass'] and prior_visual['matte_local_visual_gate'])
        followup=out/'selected_policy_confirmation_result_supported_15k.json'
        if region=='exhibit' and followup.exists():
            confirmation=m.read(followup)
            review_path=m.RUN/'visual_review_supported15k.json'
            visual_ok=review_path.exists() and m.read(review_path).get('promotion_visual_gate',False)
            if confirmation['coarse_quantitative_pass'] and confirmation['manual_structure_quantitative_pass'] and visual_ok:
                selected='confidence_supported_15k';suffix='_supported_15k';accepted=True
        checkpoint=out/selected/'ckpts/final_post_update.pt'
        m.write(out/'viewer.json',dict(checkpoint=str(checkpoint.relative_to(ROOT)),dataset=str(dataset.relative_to(ROOT)),camera_index=0,
            source_sha256=source['source_sha256'],scope='Local appearance/consistency candidate; independent submap world; unresolved global placement and physical accuracy',
            declared_local_followup_gate_pass=accepted,
            verification='Final checkpoint reload and offline pose/FOV/replay samples; no new browser interaction audit'))
        m.write(out/'camera_provenance.json',dict(normalization_transform=parser.transform.tolist(),
            image_names=parser.image_names,camera_to_world_normalized=parser.camtoworlds.tolist(),
            K={str(k):v.tolist() for k,v in parser.Ks_dict.items()},
            source_id_by_image={n:byname[n]['rgb_message_id'] for n in parser.image_names},
            train=m.read(dataset/'experiment_split.json')['train'],eval=m.read(dataset/'experiment_split.json')['val'],
            projection_version=m.projection.SDK,metric_before_normalization=True,world_relationship_to_other_submap=None,
            replay_seconds=replay.duration,replay_uses_actual_timestamps=True))
        paired=m.read(out/('paired_evaluation'+suffix+'.json'));evidence=out/('evaluation'+suffix);panels=[]
        for view in paired['results']['confidence']['views']:
            name=view['name'];row=Image.new('RGB',(1251,260),'white');draw=ImageDraw.Draw(row)
            for i,(label,path) in enumerate([('Captured',m.DATA/'images'/name),('Preserved 120k baseline',evidence/f'baseline_{name}'),('Local confidence candidate',evidence/f'confidence_{name}')]):
                im=Image.open(path);im.thumbnail((417,232));row.paste(im,(i*417,28));draw.text((i*417+5,7),f'{label} | {byname[name]["selected_name"]}',fill='black')
            panels.append(row)
        sheet=Image.new('RGB',(1251,len(panels)*260),'white')
        for i,row in enumerate(panels):sheet.paste(row,(0,i*260))
        sheet.save(out/'selected_comparison.jpg',quality=94)
        records[region]=dict(checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=m.sha(checkpoint),dataset=str(dataset.relative_to(ROOT)),
            normalization_sha256=m.sha(out/'camera_provenance.json'),viewer_config=str((out/'viewer.json').relative_to(ROOT)),
            replay_poses=len(replay.names),replay_duration_seconds=replay.duration,
            declared_local_followup_gate_pass=accepted,selected_variant=selected,
            unpromoted_followup_checkpoint=(str((out/'confidence_supported_15k/ckpts/final_post_update.pt').relative_to(ROOT))
                if region=='exhibit' and (out/'confidence_supported_15k/ckpts/final_post_update.pt').exists() and not accepted else None),
            novel_views='evaluation/novel_x* and novel_yaw*: small perturbations for inspection, not certified free-space paths')
    m.write(m.RUN/'local_delivery.json',dict(submaps=records,global_merge=None,baseline_preserved=True,
        coverage_and_uncertainty='Only declared local windows/check support. Missing/far/glass/dynamic backgrounds/thin-edge holes and physical accuracy remain unresolved. Per-observation uncertainty flags retained in each dataset/evidence.',
        all_R0_R3_acceptance_gates_complete=False))


if __name__=='__main__':main()
