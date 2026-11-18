"""Verify baseline identities and record this audit's reproducibility metadata."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/room_improve_multiview_diagnostic_20261103'
DATA=ROOT/'data/realsense_20261030_134523_clean'
SCRIPTS=['audit-room-rgbd-alignment.py','audit-room-transition-profile.py','check-room-reserved-transitions.py',
         'audit-room-revisits-rgb.py','audit-room-depth-independent-groups.py','export-room-diagnostic-cloud.py',
         'review-room-multiview-examples.py','verify-room-steps-1-to-6.py']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=json.loads((DATA/'capture_manifest.json').read_text());by={f['name']:f for f in manifest['frames']}
    with (ROOT/'docs/room_baseline_heldout_20261103.csv').open(newline='',encoding='utf-8') as f:held=list(csv.DictReader(f))
    assert len(held)==60 and len({r['image_name'] for r in held})==60
    for r in held:
        x=by[r['image_name']]
        assert int(r['rgb_message_id'])==x['message_id'] and int(r['depth_message_id'])==x['depth_message_id']
        assert int(r['rgb_timestamp_ns'])==x['timestamp_ns'] and int(r['depth_timestamp_ns'])==x['depth_timestamp_ns']
    poses=json.loads((ROOT/'results/room_improve_baseline_inspection_20261103/inspection_poses.json').read_text());assert len(poses['views'])==9
    for v in poses['views']:
        assert np.asarray(v['camera_to_world_normalized']).shape==(4,4) and np.isfinite(v['camera_to_world_normalized']).all()
        assert (ROOT/'results/room_improve_baseline_inspection_20261103'/v['render_file']).is_file()
    assert Path(poses['checkpoint']).is_file()
    r=json.loads((OUT/'report.json').read_text());a=set(r['views']['development']);b=set(r['views']['check']);h={x['image_name'] for x in held}
    assert not a&b and not a&h and not b&h
    for p in (ROOT/'results/room_improve_reserved_alignment_20261103').glob('*.dense_depth.npz'):
        with np.load(p) as f:assert f['depth_m'].shape==(463,834) and np.isfinite(f['depth_m']).all()
    out=dict(baseline_heldout_count=len(held),manifest_identity_checks='pass',saved_pose_count=9,pose_and_render_checks='pass',
        checkpoint=str(poses['checkpoint']),training_frame_group_sizes=[len(a),len(b)],heldout_overlap_counts=[len(a&h),len(b&h)],
        group_overlap=len(a&b),source_sha256_recorded_and_rechecked='60e3f253051fc282a61af7f6c9c5eef40830a12d42566b9eba0992463115bc52',
        heldout_sha256=sha(ROOT/'docs/room_baseline_heldout_20261103.csv'),
        scripts_sha256={s:sha(ROOT/'scripts'/s) for s in SCRIPTS},
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),
        note='DB3 hash independently computed using PowerShell Get-FileHash and manually matched to provenance; initial baseline provenance has null initial hash.')
    (OUT/'verification_provenance.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
