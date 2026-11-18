"""Compare pose graphs on same candidate-image feature correspondences."""
import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np
import pycolmap

ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'data/realsense_20261030_134523_clean';NEW=ROOT/'data/room_improve_sdk_rectified_20261103';OUT=ROOT/'results/room_improve_sdk_paired_epipolar_20261103'
spec=importlib.util.spec_from_file_location('clean',ROOT/'scripts/clean-realsense-scene.py');clean=importlib.util.module_from_spec(spec);spec.loader.exec_module(clean)
spec2=importlib.util.spec_from_file_location('revisit',ROOT/'scripts/audit-room-revisits-rgb.py');base=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(base)
PAIRS=[(110,129),(111,130),(119,143),(129,143),(131,143),(179,191),(191,204),(204,249),(205,250),(239,300),(249,290),(252,300),(300,329)]
def pose(im):
    w=np.eye(4);w[:3]=im.cam_from_world().matrix();return w
def Fpose(a,b,K):
    T=pose(b)@np.linalg.inv(pose(a));return np.linalg.inv(K).T@base.skew(T[:3,3])@T[:3,:3]@np.linalg.inv(K)
def to_old(uv,K,coeff):
    xy=(uv-[K[0,2],K[1,2]])/[K[0,0],K[1,1]]
    # candidate ideal -> raw via Brown -> original ideal via Brown
    return clean.brown(clean.brown(xy,coeff),coeff)*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
def main():
    OUT.mkdir(parents=True,exist_ok=False);old=pycolmap.Reconstruction(OLD/'rgbd/dataset/sparse');new=pycolmap.Reconstruction(NEW/'rgbd/dataset/sparse');oi={i.name:i for i in old.images.values()};ni={i.name:i for i in new.images.values()};cal=json.loads((NEW/'calibration_clean.json').read_text());K=np.array(cal['K']);coeff=np.array(list(map(float,cal['raw_color']['coeffs'].split(','))))
    sift=cv2.SIFT_create(nfeatures=6000);matcher=cv2.BFMatcher();cache={};rows=[]
    for n in sorted(set(x for p in PAIRS for x in p)):
        name=f'frame_{n:06}.png';im=cv2.imread(str(NEW/'images'/name));kp,d=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None);cache[n]=(im,kp,d,name)
    for a,b in PAIRS:
        ia,ka,da,na=cache[a];ib,kb,db,nb=cache[b];fw=matcher.knnMatch(da,db,k=2);rv=matcher.knnMatch(db,da,k=2);back={m.queryIdx:m.trainIdx for pair in rv if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance}
        good=[m for pair in fw if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance and back.get(m.trainIdx)==m.queryIdx];row=dict(a=a,b=b,matches=len(good))
        if len(good)<25:row['status']='insufficient matches';rows.append(row);continue
        x=np.array([ka[m.queryIdx].pt for m in good]);y=np.array([kb[m.trainIdx].pt for m in good]);perm=np.random.default_rng(a*10000+b).permutation(len(good));n=int(.6*len(good));train=perm[:n];test=perm[n:]
        cv2.setRNGSeed(1234);F,mask=cv2.findFundamentalMat(x[train],y[train],cv2.FM_RANSAC,1.5,.999)
        if F is None or F.shape!=(3,3):row['status']='fundamental fit failed';rows.append(row);continue
        oldx=to_old(x,K,coeff);oldy=to_old(y,K,coeff);ea=base.residual(Fpose(oi[na],oi[nb],K),oldx[test],oldy[test]);eb=base.residual(Fpose(ni[na],ni[nb],K),x[test],y[test]);ef=base.residual(F,x[test],y[test]);consistent=ef<1.5
        def metrics(v):return [float(z) for z in np.percentile(v,[50,90])]
        row.update(status='diagnostic',fit_matches=len(train),check_matches=len(test),fit_inliers=int(mask.sum()),check_f_consistent=int(consistent.sum()),
          old_all_median_p90_px=metrics(ea),candidate_all_median_p90_px=metrics(eb),
          old_f_consistent_median_p90_px=metrics(ea[consistent]) if consistent.any() else None,candidate_f_consistent_median_p90_px=metrics(eb[consistent]) if consistent.any() else None)
        show=[good[i] for i,ok in zip(test,consistent) if ok][:80];vis=cv2.drawMatches(ia,ka,ib,kb,show,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS);cv2.imwrite(str(OUT/f'{a:06}_{b:06}.jpg'),vis)
        rows.append(row);print(a,b,'check',len(test),'F-consistent',int(consistent.sum()),'old',round(float(np.median(ea)),2),'new',round(float(np.median(eb)),2),flush=True)
    (OUT/'report.json').write_text(json.dumps(dict(pairs=rows,method='Same candidate-image SIFT matches compared against each graph; corrected coordinates mapped continuously to old rectification via Brown twice. Feature split 60/40 before F fitting, check F-consistent uses candidate-derived F. No semantic masking.',
      caveat='Both graphs use overlapping source frames, including all check views; feature check split is not independent pose-graph validation. Mappings use SDK interpretation and fitted local Brown math; rounding/interpolation and feature localization differ. No metric depth, physical surface or global truth. F-consistent subset selected by candidate F; may favor candidate. Planarity, reflections, moving people and outliers remain.'),indent=2),encoding='utf-8')
if __name__=='__main__':main()
