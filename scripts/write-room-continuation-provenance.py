"""Record code and primary artifact hashes for the room continuation."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/room_improve_full_source_cloud_20261103'
FILES=['scripts/render-room-free-camera-baseline.py','scripts/run-room-free-path.ps1','scripts/inventory-room-static-structures.py',
 'scripts/audit-room-calibration-numerical.py','scripts/audit-room-revisits-split.py','scripts/audit-room-disjoint-source-cloud.py',
 'scripts/audit-room-full-source-cloud.py','scripts/compare-room-experimental-cloud.py','scripts/compare-room-full-source-cloud.py',
 'scripts/verify-room-full-source-cloud.py','scripts/audit-room-full-cloud-dense.py','scripts/review-room-dense-conflicts.py','scripts/compare-room-cloud-fixed-views.py',
 'scripts/write-room-continuation-provenance.py','tests/test_room_transition_profile.py']
ARTIFACTS=['results/room_improve_free_path_v02_20261103/path.json','results/room_improve_static_inventory_20261103/inventory.json',
 'results/room_improve_split_revisits_20261103/report.json','results/room_improve_full_source_cloud_20261103/development_only_unfiltered.ply',
 'results/room_improve_full_source_cloud_20261103/development_dense_stable_uncertainty.ply','results/room_improve_full_source_cloud_20261103/dense_check_report.json',
 'results/room_improve_cloud_fixed_views_v02_20261103/report.json']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    out=dict(source='20261030_134523.db3',source_sha256='60e3f253051fc282a61af7f6c9c5eef40830a12d42566b9eba0992463115bc52',
       source_hash_note='Previously recomputed in same research session; not recomputed by this script.',
       code_sha256={f:sha(ROOT/f) for f in FILES},artifact_sha256={f:sha(ROOT/f) for f in ARTIFACTS},
       gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),
       user_preference='Keep full room and clearly mark uncertain areas.',
       outcome='Experimental artifacts only. Alignment, pose correctness and filtered/fused cloud not validated; no training.')
    (OUT/'continuation_provenance.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print('wrote',len(FILES),'code and',len(ARTIFACTS),'artifact hashes')
if __name__=='__main__':main()
