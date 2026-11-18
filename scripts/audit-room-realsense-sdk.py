"""Cross-check calibration math against isolated librealsense Python SDK."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.cache/room_sdk_audit/install'))
import pyrealsense2 as rs
spec=importlib.util.spec_from_file_location('clean',ROOT/'scripts/clean-realsense-scene.py');clean=importlib.util.module_from_spec(spec);spec.loader.exec_module(clean)
DATA=ROOT/'data/realsense_20261030_134523_clean';OUT=ROOT/'results/room_improve_sdk_calibration_v03_20261103'

def intr(d,model):
    i=rs.intrinsics();i.width=int(d['width']);i.height=int(d['height']);i.fx=float(d['fx']);i.fy=float(d['fy']);i.ppx=float(d['ppx']);i.ppy=float(d['ppy']);i.model=model;i.coeffs=[float(c) for c in d['coeffs'].split(',')];return i
def main():
    OUT.mkdir(parents=True,exist_ok=False);cal=json.loads((DATA/'calibration_clean.json').read_text());raw=json.loads((DATA/'calibration_raw.json').read_text());color=cal['raw_color'];depth=cal['raw_depth'];ci=intr(color,rs.distortion.inverse_brown_conrady);di=intr(depth,rs.distortion.brown_conrady)
    xs=np.linspace(0,ci.width-1,9);ys=np.linspace(0,ci.height-1,7);pixels=np.array([(x,y) for y in ys for x in xs]);norm=(pixels-[float(color['ppx']),float(color['ppy'])])/[float(color['fx']),float(color['fy'])];coeff=np.array(list(map(float,color['coeffs'].split(','))))
    local=clean.brown(norm,coeff);sdk=np.array([rs.rs2_deproject_pixel_to_point(ci,[float(x),float(y)],1.)[:2] for x,y in pixels]);difference=np.linalg.norm((local-sdk)*[float(color['fx']),float(color['fy'])],axis=1)
    alternate=clean.invert_brown(norm,coeff);alt_difference=np.linalg.norm((alternate-sdk)*[float(color['fx']),float(color['fy'])],axis=1)
    ideals=np.array([(x,y) for y in np.linspace(-.5,.5,5) for x in np.linspace(-.8,.8,7)]);inverse=clean.invert_brown(ideals,coeff);rawpixel=inverse*[float(color['fx']),float(color['fy'])]+[float(color['ppx']),float(color['ppy'])];back=np.array([rs.rs2_deproject_pixel_to_point(ci,[float(x),float(y)],1.)[:2] for x,y in rawpixel]);roundtrip=np.linalg.norm((back-ideals)*[float(color['fx']),float(color['fy'])],axis=1)
    alternative_pixel=clean.brown(ideals,coeff)*[float(color['fx']),float(color['fy'])]+[float(color['ppx']),float(color['ppy'])];altback=np.array([rs.rs2_deproject_pixel_to_point(ci,[float(x),float(y)],1.)[:2] for x,y in alternative_pixel]);alt_roundtrip=np.linalg.norm((altback-ideals)*[float(color['fx']),float(color['fy'])],axis=1)
    tf=clean.fields(raw['/device_0/sensor_1/Color_0/tf/ref_0']);ext=rs.extrinsics();ext.rotation=list(map(float,tf['rotation'].split(',')));ext.translation=list(map(float,tf['translation'].split(',')))
    depthpixels=np.array([(x,y) for y in (0,239,479) for x in (0,423,847)]);depths=(.7,2.,5.);errors=[];trans=[]
    for uv in depthpixels:
        for z in depths:
            sdkpoint=np.array(rs.rs2_deproject_pixel_to_point(di,[float(uv[0]),float(uv[1])],z));localpoint=np.array([(uv[0]-float(depth['ppx']))/float(depth['fx'])*z,(uv[1]-float(depth['ppy']))/float(depth['fy'])*z,z]);errors.append(np.linalg.norm(sdkpoint-localpoint))
            sdkx=np.array(rs.rs2_transform_point_to_point(ext,sdkpoint.tolist()));localx=np.array(cal['depth_to_color_R'])@localpoint+np.array(cal['depth_to_color_t_m']);trans.append(np.linalg.norm(sdkx-localx))
    wheel=next((ROOT/'.cache/room_sdk_audit').glob('pyrealsense2-*.whl'));sha=hashlib.sha256(wheel.read_bytes()).hexdigest()
    report=dict(sdk_version=rs.__version__,wheel=wheel.name,wheel_sha256=sha,rgb_raw_pixel_samples=len(pixels),rgb_sdk_deproject_vs_baseline_px_percentiles=np.percentile(difference,[50,90,100]).tolist(),
      rgb_sdk_deproject_vs_alternate_inverse_px_percentiles=np.percentile(alt_difference,[50,90,100]).tolist(),
      inverse_remap_grid_samples=len(ideals),sdk_roundtrip_vs_target_px_percentiles=np.percentile(roundtrip,[50,90,100]).tolist(),depth_samples=len(errors),depth_deprojection_error_m_max=float(max(errors)),
      alternate_sdk_roundtrip_vs_target_px_percentiles=np.percentile(alt_roundtrip,[50,90,100]).tolist(),
      depth_to_color_transform_error_m_max=float(max(trans)),
      method='SDK color inverse-Brown deprojection compared with local brown mapping; SDK deprojection of local inverse mapping compared with target; SDK depth deprojection and SDK raw extrinsics transform compared with saved transform.',
      caveat='Uses same stored calibration and model declarations, no camera/device and no measured ground truth. Confirms internal agreement with this SDK implementation only; not transform-frame semantics in the recording, exposure timing, scene edge alignment, or accuracy.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
