"""Recompute bookkeeping checks of saved calibration; not physical validation."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/realsense_20261030_134523_clean';OUT=ROOT/'results/room_improve_calibration_numerical_20261103'
def fields(s):return dict(f.split('=',1) for f in s.split(';'))
def main():
    OUT.mkdir(parents=True,exist_ok=False);raw=json.loads((DATA/'calibration_raw.json').read_text());cal=json.loads((DATA/'calibration_clean.json').read_text());manifest=json.loads((DATA/'capture_manifest.json').read_text())
    color=fields(raw['/device_0/sensor_1/Color_0/camera_info']);depth=fields(raw['/device_0/sensor_0/Depth_0/camera_info']);w=manifest['parameters']['width'];h=round(int(color['height'])*w/int(color['width']));cx,cy=cal['crop_xy']
    expected=np.array([[float(color['fx'])*w/int(color['width']),0,float(color['ppx'])*w/int(color['width'])-cx],
      [0,float(color['fy'])*h/int(color['height']),float(color['ppy'])*h/int(color['height'])-cy],[0,0,1]])
    K=np.asarray(cal['K']);rawtf=fields(raw['/device_0/sensor_1/Color_0/tf/ref_0']);R=np.array(list(map(float,rawtf['rotation'].split(',')))).reshape(3,3,order='F');t=np.array(list(map(float,rawtf['translation'].split(','))))
    savedR=np.array(cal['depth_to_color_R']);savedt=np.array(cal['depth_to_color_t_m']);gaps=np.array([abs(f['timestamp_ns']-f['depth_timestamp_ns'])/1e6 for f in manifest['frames']]);header=np.array([abs(f['header_seconds']-f['depth_header_seconds'])*1000 for f in manifest['frames']])
    report=dict(expected_K=expected.tolist(),saved_K=K.tolist(),max_intrinsic_difference=float(np.max(abs(expected-K))),
      expected_dimensions=[w-2*cx,h-2*cy],saved_dimensions=[cal['width'],cal['height']],raw_depth_units=float(raw['/device_0/sensor_0/option/Depth_Units/value']),saved_depth_units=cal['depth_units'],
      depth_distortion_coeffs=depth['coeffs'],rgb_distortion_model=color['model'],rotation_storage_assumption='column-major as used by baseline parser; depth is reference',
      max_rotation_difference=float(np.max(abs(savedR-R))),max_translation_difference_m=float(np.max(abs(savedt-t))),rotation_orthonormal_error=float(np.max(abs(savedR.T@savedR-np.eye(3)))),rotation_determinant=float(np.linalg.det(savedR)),
      frame_count=len(gaps),timestamp_gap_ms_percentiles=np.percentile(gaps,[0,50,90,100]).tolist(),header_gap_ms_percentiles=np.percentile(header,[0,50,90,100]).tolist(),
      caveat='Bookkeeping checks reproduce recorded values and baseline conventions; they do not establish coordinate transform direction, physical sensor calibration, RGB-depth synchronization, edge accuracy, or applicability of inverse Brown model. Different exposure times are not recovered by timestamps alone.')
    assert report['max_intrinsic_difference']<1e-9 and report['expected_dimensions']==report['saved_dimensions']
    assert report['max_rotation_difference']<1e-9 and report['max_translation_difference_m']<1e-9
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
