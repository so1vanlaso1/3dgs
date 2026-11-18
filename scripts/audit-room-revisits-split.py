"""Train/check feature split for RGB-only candidate revisit pairs."""
import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np
import pycolmap

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/realsense_20261030_134523_clean/rgbd/dataset';OUT=ROOT/'results/room_improve_split_revisits_20261103'
spec=importlib.util.spec_from_file_location('revisit',ROOT/'scripts/audit-room-revisits-rgb.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
PAIRS=[(110,129),(111,130),(119,143),(129,143),(131,143),(143,179),(179,191),(191,204),(204,249),(205,250),(239,300),(249,290),(252,300),(300,329)]
def main():
    OUT.mkdir(parents=True,exist_ok=False);model=pycolmap.Reconstruction(DATA/'sparse');images={i.name:i for i in model.images.values()};cam=next(iter(model.cameras.values()));fx,fy,cx,cy=cam.params;K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    sift=cv2.SIFT_create(nfeatures=6000);matcher=cv2.BFMatcher();cache={};rows=[]
    for n in sorted(set(v for pair in PAIRS for v in pair)):
        name=f'frame_{n:06}.png';img=cv2.imread(str(DATA/'images'/name));kp,desc=sift.detectAndCompute(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),None);cache[n]=(img,kp,desc,images[name])
    for a,b in PAIRS:
        ia,ka,da,pa=cache[a];ib,kb,db,pb=cache[b];fw=matcher.knnMatch(da,db,k=2);rv=matcher.knnMatch(db,da,k=2)
        back={m.queryIdx:m.trainIdx for pair in rv if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance}
        good=[m for pair in fw if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance and back.get(m.trainIdx)==m.queryIdx]
        row=dict(a=a,b=b,mutual_matches=len(good))
        if len(good)<25:row['status']='too few matches';rows.append(row);print(a,b,row['status'],len(good));continue
        x=np.array([ka[m.queryIdx].pt for m in good]);y=np.array([kb[m.trainIdx].pt for m in good]);rng=np.random.default_rng(a*10000+b);perm=rng.permutation(len(good));n=max(15,int(.6*len(good)));train=perm[:n];test=perm[n:]
        cv2.setRNGSeed(1234);F,mask=cv2.findFundamentalMat(x[train],y[train],cv2.FM_RANSAC,1.5,.999);H,hm=cv2.findHomography(x[train],y[train],cv2.RANSAC,3.)
        if F is None or F.shape!=(3,3):row['status']='fundamental fit failed';rows.append(row);continue
        t1=np.eye(4);t2=np.eye(4);t1[:3]=pa.cam_from_world().matrix();t2[:3]=pb.cam_from_world().matrix();T=t2@np.linalg.inv(t1);Fp=np.linalg.inv(K).T@base.skew(T[:3,3])@T[:3,:3]@np.linalg.inv(K)
        fit=base.residual(F,x[test],y[test]);pose=base.residual(Fp,x[test],y[test]);
        row.update(status='diagnostic',fit_matches=len(train),fit_f_inliers=int(mask.sum()),fit_h_inliers=int(hm.sum()) if hm is not None else 0,
          check_matches=len(test),check_fitted_median_px=float(np.median(fit)),check_fitted_p90_px=float(np.percentile(fit,90)),
          check_pose_median_px=float(np.median(pose)),check_pose_p90_px=float(np.percentile(pose,90)))
        # Both train and check correspondences are shown; red lines denote check matches.
        show=[good[i] for i in test[:80]];vis=cv2.drawMatches(ia,ka,ib,kb,show,None,matchColor=(0,0,255),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(str(OUT/f'{a:06}_{b:06}_check.jpg'),vis);rows.append(row)
        print(a,b,len(good),'check',len(test),'baseline median',round(float(np.median(pose)),2),flush=True)
    (OUT/'report.json').write_text(json.dumps(dict(pairs=rows,method='SIFT mutual ratio .75; deterministic 60/40 feature split per pair before F fitting. RANSAC 1.5 px; check matches are not filtered by either model. Same images may influence original graph; no explicit dynamic/reflective mask. Metrics are Sampson residuals, not ground truth or metric pose accuracy.',seed='a*10000+b for split; OpenCV RANSAC 1234'),indent=2),encoding='utf-8')
if __name__=='__main__':main()
