"""Run identical corrected bounded TSDF extraction on all four saved arms."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import json
from research_campaign import REPO,sha,write
p=argparse.ArgumentParser(__doc__)
p.add_argument('--root',type=Path,required=True)
a=p.parse_args()
root=a.root.resolve()
records=json.loads((root/'surface/summary.json').read_text())
dest=root/'extraction_evidence/gaussians_to_mesh.py'
dest.parent.mkdir(exist_ok=True)
shutil.copy2(REPO/'scripts/gaussians_to_mesh.py',dest)
for label,rec in records.items():
    out=root/'meshes_uniform'/label
    write(root/'status.json',dict(state='extracting_mesh_uniform',arm=label))
    folder=Path(rec['archived_control']).parent if 'archived_control' in rec else root/'surface'/rec['job']['id']
    with (root/(label+'_uniform_mesh.log')).open('w',encoding='utf8') as log:
        subprocess.run([sys.executable,'-u','scripts/gaussians_to_mesh.py',
            '--checkpoint',str(folder/'ckpts/final_post_update.pt'),'--data','data/room_factor2',
            '--output',str(out)],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=REPO)
    print('EXTRACTED',label,flush=True)
write(root/'status.json',dict(state='complete',extraction_source_sha256=sha(dest)))
