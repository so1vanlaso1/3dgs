"""Summarize saved surface factorial results without reselecting settings."""
import json
from pathlib import Path
import yaml
import numpy as np
from PIL import Image
from research_campaign import REPO, sha, write

import argparse
p=argparse.ArgumentParser(__doc__)
p.add_argument('--root',type=Path,default=REPO/'results/school_completion_20261109')
root=p.parse_args().root.resolve()
records=json.loads((root/'surface/summary.json').read_text())
protocol=json.loads((root/'surface/protocol.json').read_text())
assert all(sha(REPO/name)==value for name,value in protocol['source_sha256'].items() if name!='scripts/gaussians_to_mesh.py')
summary={}
mesh_root=root/('meshes_uniform' if (root/'meshes_uniform').exists() else 'meshes')
for label,rec in records.items():
    mesh=json.loads((mesh_root/label/'report.json').read_text())
    assert mesh['checkpoint_sha256']==rec['final_checkpoint_sha256']
    assert mesh['gaussian_count']==rec['metrics']['num_GS']
    summary[label]=dict(render=rec['metrics'],mesh=mesh)
deltas={}
for base in ('baseline','reduced'):
    before,after=records[base],records[base+'_surface']
    delta={k:after['metrics'][k]-before['metrics'][k] for k in ('psnr','ssim','lpips','num_GS')}
    delta['render_guardrails_pass']=delta['psnr']>=-.2 and delta['ssim']>=-.005 and delta['lpips']<=.01
    folders=[]
    for rec in (before,after):
        folder=Path(rec['archived_control']).parent if 'archived_control' in rec else root/'surface'/rec['job']['id']
        folders.append(folder)
    configs=[yaml.load((folder/'cfg.yml').read_text(),Loader=yaml.BaseLoader) for folder in folders]
    initial=[json.loads((folder/'stats/val_step0000.json').read_text()) for folder in folders]
    initial_delta={k:initial[1][k]-initial[0][k] for k in ('psnr','ssim','lpips')}
    assert all(abs(v)<.0001 for v in initial_delta.values()),initial_delta
    images=sorted((folders[0]/'renders').glob('val_step6999_*.png'))
    assert len(images)==39
    for image in images:
        a=np.asarray(Image.open(image))
        b=np.asarray(Image.open(folders[1]/'renders'/image.name))
        assert a.shape==b.shape and np.array_equal(a[:,:a.shape[1]//2],b[:,:b.shape[1]//2])
    delta['identical_target_images']=len(images)
    delta['initial_metric_delta']=initial_delta
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events=EventAccumulator(str(folders[1]/'tb'))
    events.Reload()
    values=events.Scalars('train/surface_loss')
    assert len(values)==40 and np.isfinite([v.value for v in values]).all()
    delta['logged_surface_penalty']={'samples':len(values),'first':values[0].value,'last':values[-1].value}
    allowed={'result_dir','data_workers','surface_lambda','surface_start','camera_sequence_path'}
    changes={key:[configs[0][key],configs[1][key]] for key in configs[0].keys()&configs[1].keys()
             if configs[0][key]!=configs[1][key]}
    assert not changes.keys()-allowed,changes
    delta['shared_configuration_differences']=changes
    delta['new_configuration_fields']={k:v for k,v in configs[1].items() if k not in configs[0]}
    deltas[base]=delta
write(root/'analysis.json',dict(arms=summary,prior_deltas=deltas,training_source_hashes_match=True,
    extraction_source_sha256=sha(REPO/'scripts/gaussians_to_mesh.py'),
    extraction_revision='Changed scalable to uniform TSDF after synthetic plane showed empty scalable integration; trained sources unchanged.',
    limitations='Room one seed, exposed evaluation views, archived controls. Mesh coverage/agreement/topology are not geometric ground truth.'))
lines=['# School surface and mesh results','',
       'Executed 9 November 2026. Full Room seed-42 factorial, 7,000 updates. Two verified archived no-prior controls and two new prior arms; matching initialization and camera-order hashes.','',
       '| Arm | PSNR | SSIM | LPIPS | Gaussians | Mesh vertices | Triangles | Boundary edges | Components |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for label,rec in summary.items():
    r,m=rec['render'],rec['mesh']
    lines.append(f"| {label} | {r['psnr']:.5f} | {r['ssim']:.6f} | {r['lpips']:.6f} | {r['num_GS']} | {m['vertices']} | {m['triangles']} | {m['boundary_edges']} | {m['components']} |")
lines+=['','## Surface-prior effects','']
for base,d in deltas.items():
    lines.append(f"- {base}: PSNR {d['psnr']:+.5f} dB, SSIM {d['ssim']:+.6f}, LPIPS {d['lpips']:+.6f}; render guardrails {'pass' if d['render_guardrails_pass'] else 'FAIL'}.")
lines+=['','## Interpretation limits','',
    'Raw meshes were extracted from the optimized Gaussian checkpoints, not sensor point clouds. Five held-out views are used only to compare mesh depth with Gaussian depth; training views alone are fused. A common bounded uniform TSDF excludes the exterior of an enlarged percentile SfM box. Neither agreement nor fewer components/boundary edges proves more accurate geometry. Expected depth can interpolate through multiple surfaces. Raw meshes retain holes, noise and disconnected pieces. No untouched-test or broad generalization claim is justified.',
    '', f'Compare previews under {mesh_root.name}/<arm>/mesh_depth_comparison.jpg. Columns: Gaussian RGB, Gaussian expected depth, independently raycast mesh depth. Depth colors share the 5th–95th percentile range of that view; black indicates no accepted depth.',
    '', 'Source/configuration checks and full metrics: analysis.json. Replacement of the reference model requires an explicit decision after visual review, not merely a passed RGB guardrail.']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(deltas,indent=2))
