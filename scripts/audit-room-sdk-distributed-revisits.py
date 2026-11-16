"""Evaluate unused direct revisit pairs using 3D depth correspondences."""
import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np
import pycolmap

ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'data/realsense_20261030_134523_clean';NEW=ROOT/'data/room_improve_sdk_rectified_20261103';DEP=ROOT/'results/room_improve_full_cloud_dense_checks_20261103';OUT=ROOT/'results/room_improve_sdk_distributed_revisits_20261103'
s=importlib.util.spec_from_file_location('paired',ROOT/'scripts/compare-room-sdk-paired-epipolar.py');base=importlib.util.module_from_spec(s);s.loader.exec_module(base)
FRAMES=list(range(10,475,16))
PAIRS=[(a,b) for i,a in enumerate(FRAMES) for b in FRAMES[i+1:] if 32 <= b-a <= 192]
def model(root):
    m=pycolmap.Reconstruction(root/'rgbd/dataset/sparse');return {i.name:i for i in m.images.values()}
def sample(z,uv,K):
    xy=np.rint(uv).astype(int);x=xy[:,0];y=xy[:,1];inside=(x>=0)&(x<z.shape[1])&(y>=0)&(y<z.shape[0]);x=np.clip(x,0,z.shape[1]-1);y=np.clip(y,0,z.shape[0]-1)
    low=cv2.erode(z,np.ones((3,3),np.uint8));high=cv2.dilate(z,np.ones((3,3),np.uint8));d=z[y,x];valid=inside&(low[y,x]>.3)&(high[y,x]<6)&((high[y,x]-low[y,x])<np.maximum(.03,d*.02))
    xyz=np.c_[(uv[:,0]-K[0,2])*d/K[0,0],(uv[:,1]-K[1,2])*d/K[1,1],d];return xyz,valid
def world(im,xyz):
    W=np.linalg.inv(base.pose(im));return xyz@W[:3,:3].T+W[:3,3]
def main():
    OUT.mkdir(parents=True,exist_ok=False);oi=model(OLD);ni=model(NEW);cal=json.loads((NEW/'calibration_clean.json').read_text());K=np.array(cal['K']);coeff=np.array(list(map(float,cal['raw_color']['coeffs'].split(','))));matcher=cv2.BFMatcher();sift=cv2.SIFT_create(nfeatures=3000);cache={}
    for n in sorted({v for p in PAIRS for v in p}):
        name=f'frame_{n:06}.png';im=cv2.imread(str(NEW/'images'/name));kp,d=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None)
        with np.load(DEP/(name+'.dense_depth.npz')) as f:z=f['depth_m']
        cache[n]=(name,im,kp,d,z)
    graph=[]
    for root in (OLD,NEW):graph.append({(min(e['i'],e['j']),max(e['i'],e['j'])) for e in json.loads((root/'rgbd/edges.json').read_text())})
    rows=[]
    for a,b in PAIRS:
        na,ia,ka,da,za=cache[a];nb,ib,kb,db,zb=cache[b];fw=matcher.knnMatch(da,db,k=2);rv=matcher.knnMatch(db,da,k=2);back={m.queryIdx:m.trainIdx for pair in rv if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance};matches=[m for pair in fw if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance and back.get(m.trainIdx)==m.queryIdx]
        row=dict(a=a,b=b,matches=len(matches),direct_edge_original=(a,b) in graph[0],direct_edge_candidate=(a,b) in graph[1]);
        if row['direct_edge_original'] or row['direct_edge_candidate']: row['reason']='direct graph edge'; rows.append(row); continue
        if len(matches)<25:row['reason']='too few matches';rows.append(row);continue
        xa=np.array([ka[m.queryIdx].pt for m in matches]);xb=np.array([kb[m.trainIdx].pt for m in matches]);perm=np.random.default_rng(a*10000+b).permutation(len(matches));n=int(.6*len(matches));train=perm[:n];check=perm[n:];cv2.setRNGSeed(1234);F,mask=cv2.findFundamentalMat(xa[train],xb[train],cv2.FM_RANSAC,1.5,.999)
        if F is None or F.shape!=(3,3):row['reason']='F fit failed';rows.append(row);continue
        fcheck=base.base.residual(F,xa[check],xb[check])<1.5;check=check[fcheck]
        nax,nva=sample(za,xa[check],K);nbx,nvb=sample(zb,xb[check],K);oax,ova=sample(za,base.to_old(xa[check],K,coeff),K);obx,ovb=sample(zb,base.to_old(xb[check],K,coeff),K);valid=nva&nvb&ova&ovb;check=check[valid]
        row.update(fit_matches=len(train),check_f_consistent=int(fcheck.sum()),stable_depth_pairs=len(check))
        if len(check)<3:row['reason']='fewer than 3 jointly stable check-depth correspondences';rows.append(row);continue
        old_error=np.linalg.norm(world(oi[na],oax[valid])-world(oi[nb],obx[valid]),axis=1);new_error=np.linalg.norm(world(ni[na],nax[valid])-world(ni[nb],nbx[valid]),axis=1)
        coord=xa[check];hull=cv2.convexHull(coord.astype('f4')) if len(check)>=3 else None
        q=lambda x:[float(v) for v in np.percentile(x,[50,90])]
        row.update(reason='diagnostic only',old_distance_m_median_p90=q(old_error),candidate_distance_m_median_p90=q(new_error),candidate_better_count=int(np.sum(new_error<old_error)),image_a_hull_fraction=float(cv2.contourArea(hull)/(ia.shape[0]*ia.shape[1])))
        selected=[matches[i] for i in check];vis=cv2.drawMatches(ia,ka,ib,kb,selected,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS);cv2.imwrite(str(OUT/f'{a:06}_{b:06}.jpg'),vis)
        np.savez_compressed(OUT/f'{a:06}_{b:06}.npz',old_error_m=old_error,candidate_error_m=new_error,xa=xa[check],xb=xb[check]);rows.append(row)
        print(a,b,'stable',len(check),'old',round(np.median(old_error),3),'candidate',round(np.median(new_error),3),flush=True)
    (OUT/'report.json').write_text(json.dumps(dict(pairs=rows,method='Corrected RGB SIFT mutual matches; 60/40 split before candidate F-RANSAC; F-consistent check matches only; same underlying raw features mapped to baseline RGB coordinates; require stable 3x3 original projected depth under both versions; compare each graph world 3D distances.',
      limitations='These pairs are not direct edges in either stored graph but their individual frames and related image/depth observations influenced pose graphs. Candidate F selects correspondences; depth alignment, poses, reflectance, moving objects, similar textures and discontinuities are uncertain. RGB feature mapping approximate. Spatial extent and 3D depths may be degenerate. No external ground truth or 3D transform fitted here; selected matches not independent scenes.'),indent=2),encoding='utf-8')
if __name__=='__main__':main()
