"""Record hashes and run metadata for the isolated SDK candidate."""
import hashlib
import json
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/room_improve_sdk_source_cloud_20261103'
scripts=['audit-room-realsense-sdk.py','compare-room-sdk-remap.py','compare-room-sdk-remap-extra.py','review-room-sdk-crops.py',
 'measure-room-sdk-edges.py','measure-room-sdk-additional-edges.py','audit-room-sdk-straight-lines.py','audit-room-sdk-3d-revisits.py','prepare-room-sdk-rectified-candidate.py','initialize-room-sdk-reconstruction.py',
 'prepare-room-sdk-rgbd-inputs.py','bridge-room-sdk-candidate.py','fix-room-sdk-split.py','compare-room-sdk-poses.py',
 'compare-room-sdk-paired-epipolar.py','audit-room-sdk-source-cloud.py','audit-room-sdk-cloud-dense.py',
 'audit-room-full-source-cloud.py','audit-room-full-cloud-dense.py','verify-room-full-source-cloud.py','verify-room-sdk-source-cloud.py','compare-room-sdk-paired-depth.py','compare-room-sdk-cloud-views.py',
 'write-room-sdk-provenance.py','clean-realsense-scene.py','recover-clean-rgbd.py','bridge-clean-rgbd.py','recover-rgbd-scene.py']
artifacts=['data/room_improve_sdk_rectified_20261103/candidate_manifest.json','data/room_improve_sdk_rectified_20261103/rgbd/report.json',
 'data/room_improve_sdk_rectified_20261103/rgbd/dataset/experiment_split.json','results/room_improve_sdk_calibration_v03_20261103/report.json',
 'results/room_improve_sdk_pose_comparison_20261103/report.json','results/room_improve_sdk_paired_depth_20261103/report.json',
 'results/room_improve_sdk_3d_revisits_20261103/report.json','results/room_improve_sdk_straight_lines_20261103/report.json',
 'results/room_improve_sdk_source_cloud_20261103/development_only_unfiltered.ply','results/room_improve_sdk_source_cloud_20261103/development_dense_stable_uncertainty.ply']
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def main():
    d=dict(source='20261030_134523.db3',source_sha256='60e3f253051fc282a61af7f6c9c5eef40830a12d42566b9eba0992463115bc52',
      source_hash_note='Rechecked earlier in this research session; not recomputed by this script.',
      sdk_wheel_sha256='045dfae020a635ba0c2c27e79ea3fa844acb86101344f68ca4c4163beb94b721',sdk_version='2.59.1.11053',
      scripts_sha256={s:sha(ROOT/'scripts'/s) for s in scripts},tests_sha256={s:sha(ROOT/'tests'/s) for s in ('test_room_sdk_pose_comparison.py','test_room_sdk_remap.py','test_room_transition_profile.py')},artifacts_sha256={a:sha(ROOT/a) for a in artifacts},
      gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),
      user_preference='Retain full room and identify uncertainty',status='Provisional RGB-D candidate; no calibration/pose acceptance or training.')
    (OUT/'sdk_candidate_provenance.json').write_text(json.dumps(d,indent=2),encoding='utf-8');print('recorded',len(scripts),'scripts and',len(artifacts),'artifacts')
if __name__=='__main__':main()
