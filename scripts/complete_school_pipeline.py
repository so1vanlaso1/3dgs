"""Run the frozen surface ablation, then extract every arm's initial mesh."""
from pathlib import Path
import subprocess
import sys
from research_campaign import REPO, write

import argparse
p=argparse.ArgumentParser(__doc__)
p.add_argument('--root',type=Path,default=REPO/'results/school_completion_20261109')
a=p.parse_args()
root=a.root.resolve()
root.mkdir(exist_ok=False)
campaign=root/'surface'
subprocess.run([sys.executable,'-u','scripts/school_surface_campaign.py','--root',str(campaign)],check=True,cwd=REPO)
checkpoints={
 'baseline':REPO/'results/research_completion_20261027/replication/room/B00_s42/ckpts/final_post_update.pt',
 'reduced':REPO/'results/a05_absgrad_20261107/replication/room/A05_absgrad_s42/ckpts/final_post_update.pt',
 'baseline_surface':campaign/'replication/room/baseline_surface/ckpts/final_post_update.pt',
 'reduced_surface':campaign/'replication/room/reduced_surface/ckpts/final_post_update.pt',
}
for label,ckpt in checkpoints.items():
    write(root/'status.json',dict(state='extracting_mesh',arm=label))
    print('EXTRACT',label,flush=True)
    with (root/(label+'_mesh.log')).open('w',encoding='utf8') as log:
        subprocess.run([sys.executable,'-u','scripts/gaussians_to_mesh.py',
            '--checkpoint',str(ckpt),'--data','data/room_factor2',
            '--output',str(root/'meshes'/label)],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=REPO)
write(root/'status.json',dict(state='complete',arms=list(checkpoints)))
print('PIPELINE_COMPLETE',flush=True)
