"""RGB-only revisit diagnostics against existing poses; never modifies a graph."""
import json
from pathlib import Path
import cv2
import numpy as np
import pycolmap

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/realsense_20261030_134523_clean/rgbd/dataset'
OUT=ROOT/'results/room_improve_rgb_revisits_20261103'
PAIRS=[(89,183),(110,129),(204,249),(249,290),(129,143),(204,205)]

def skew(t):
    x,y,z=t;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])

def residual(F,a,b):
    a=np.c_[a,np.ones(len(a))];b=np.c_[b,np.ones(len(b))];fa=a@F.T;ftb=b@F
    denom=np.sqrt(fa[:,0]**2+fa[:,1]**2+ftb[:,0]**2+ftb[:,1]**2)
    return np.abs(np.sum(b*fa,axis=1))/np.maximum(denom,1e-12)

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    model=pycolmap.Reconstruction(DATA/'sparse'); images={i.name:i for i in model.images.values()}; cam=next(iter(model.cameras.values()))
    fx,fy,cx,cy=cam.params;K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    sift=cv2.SIFT_create(nfeatures=6000); matcher=cv2.BFMatcher(); cache={};rows=[]
    for n in sorted(set(n for p in PAIRS for n in p)):
        name=f'frame_{n:06d}.png';im=cv2.imread(str(DATA/'images'/name));assert im is not None
        k,d=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None); cache[n]=(im,k,d,images[name])
    for a,b in PAIRS:
        im1,k1,d1,p1=cache[a];im2,k2,d2,p2=cache[b]
        fw=matcher.knnMatch(d1,d2,k=2);rev=matcher.knnMatch(d2,d1,k=2)
        back={m.queryIdx:m.trainIdx for pair in rev if len(pair)==2 for m,n in [pair] if m.distance < .75*n.distance}
        good=[m for pair in fw if len(pair)==2 for m,n in [pair] if m.distance < .75*n.distance and back.get(m.trainIdx)==m.queryIdx]
        row=dict(frame_a=a,frame_b=b,mutual_ratio_matches=len(good),warning='Unmasked features may include moving objects, reflections, repeats or planar regions. Same images may have supported original reconstruction; not independent ground truth.')
        if len(good)<12:row['decision']='insufficient mutual features';rows.append(row);continue
        xy1=np.array([k1[m.queryIdx].pt for m in good],dtype=np.float64);xy2=np.array([k2[m.trainIdx].pt for m in good],dtype=np.float64)
        cv2.setRNGSeed(1234); F,mask=cv2.findFundamentalMat(xy1,xy2,cv2.FM_RANSAC,1.5,.999)
        H,hm=cv2.findHomography(xy1,xy2,cv2.RANSAC,3.0)
        t1=p1.cam_from_world().matrix();t2=p2.cam_from_world().matrix();T1=np.eye(4);T2=np.eye(4);T1[:3]=t1;T2[:3]=t2;T=T2@np.linalg.inv(T1)
        if F is None or mask is None or F.shape!=(3,3):row['decision']='fundamental fit failed';rows.append(row);continue
        keep=mask.ravel().astype(bool); fpose=np.linalg.inv(K).T@skew(T[:3,3])@T[:3,:3]@np.linalg.inv(K)
        ep=residual(fpose,xy1,xy2);fit=residual(F,xy1,xy2)
        hull=cv2.convexHull(xy1[keep].astype('float32')) if keep.sum()>=3 else None
        row.update(fundamental_inliers=int(keep.sum()),homography_inliers=int(hm.sum()) if hm is not None else 0,
           image1_inlier_hull_fraction=float(cv2.contourArea(hull)/(im1.shape[0]*im1.shape[1])) if hull is not None else 0,
           baseline_camera_center_separation=float(np.linalg.norm(T[:3,3])),
           fitted_f_inlier_sampson_px_median=float(np.median(fit[keep])) if keep.any() else None,
           baseline_pose_sampson_px_median=float(np.median(ep[keep])) if keep.any() else None,
           baseline_pose_sampson_px_p90=float(np.percentile(ep[keep],90)) if keep.any() else None,
           decision='diagnostic only; no accepted constraint')
        show=[m for m,k in zip(good,keep) if k][:100];image=cv2.drawMatches(im1,k1,im2,k2,show,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(str(OUT/f'{a:06d}_{b:06d}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,90]);rows.append(row)
        print(a,b,len(good),row['fundamental_inliers'],round(row['baseline_pose_sampson_px_median'],2),flush=True)
    (OUT/'report.json').write_text(json.dumps(dict(method='SIFT ratio 0.75 plus reciprocal matching; fundamental RANSAC 1.5 px; homography RANSAC 3 px. Pose residual on same F-selected inliers, not independent ground truth.',pairs=rows),indent=2),encoding='utf-8')

if __name__=='__main__':main()
