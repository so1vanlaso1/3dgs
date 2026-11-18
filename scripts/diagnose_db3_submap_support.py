"""Separate initial fused support from trained Gaussian depth on fixed check ROIs."""
import importlib.util
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree

s=importlib.util.spec_from_file_location('local',Path(__file__).with_name('db3_local_geometry.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def main():
    out=m.RUN/'exhibit/pilot';cloud=np.load(out/'supported_initialization.npz')['xyz']
    cam=np.load(m.RUN/'exhibit/rate10/checked_poses.npz');poses=dict(zip(cam['names'],cam['world_to_camera']))
    K=np.array(m.read(m.DATA/'calibration_clean.json')['K']);regions=m.read(m.RUN/'confirmation_structure_regions.json')['regions_xyxy']
    records=[]
    for name,boxes in regions.items():
        W=poses[name];q=cloud@W[:3,:3].T+W[:3,3];good=(q[:,2]>.3)&(q[:,2]<6);q=q[good]
        uv=q[:,:2]/q[:,2:3]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
        p,z,ev=m.projection.zbuffer(uv,q[:,2],834,463);tree=cKDTree(p)
        gt=np.load(m.DATA/'depth'/(name+'.npz'))['depth_m'];lo=cv2.erode(gt,np.ones((3,3),'u1'));hi=cv2.dilate(gt,np.ones((3,3),'u1'))
        stable=(lo>.3)&(hi<6)&((hi-lo)<np.maximum(.03,.02*gt))
        for label,(x0,y0,x1,y1) in boxes.items():
            yy,xx=np.mgrid[y0:y1:2,x0:x1:2];pixels=np.c_[xx.ravel(),yy.ravel()];valid=stable[pixels[:,1],pixels[:,0]];pixels=pixels[valid]
            distance,nn=tree.query(pixels);covered=distance<=2.
            err=np.abs(z[nn[covered]]-gt[pixels[covered,1],pixels[covered,0]])
            records.append(dict(view=name,region=label,stable_source_samples=len(pixels),initial_support_samples=int(covered.sum()),
                fraction_with_initial_point_within_2px=float(covered.mean()) if len(covered) else 0.,
                initial_depth_error_m_median_p90=[float(v) for v in np.percentile(err,[50,90])] if len(err) else None))
    m.write(out/'initialization_structure_diagnosis.json',dict(regions=records,
        convention='Frozen metric supported initializer projected with frozen evaluation localizations; nearest-Z observed cloud points, no hole filling; check source sample must lie within 2px of a projected point. Coverage conditional, not a surface rendering.',
        interpretation='A pretraining discrepancy prevents uniquely attributing trained depth error to Gaussian optimization. Low support is unknown, not confirmed wrong geometry. No absolute accuracy or sensor calibration inference.'))
    print(records,flush=True)


if __name__=='__main__':main()
