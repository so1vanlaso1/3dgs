"""Metric RGB-D registration, robust pose graph, and a low-memory gsplat dataset.

Uses recorded calibration, SIFT correspondences already verified by COLMAP,
depth-supported 3D RANSAC, and Open3D CPU pose graph optimization.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','8')
import argparse
import importlib.util
import json
import sqlite3
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from PIL import Image

spec = importlib.util.spec_from_file_location('depth_adapter',Path(__file__).with_name('prepare-sensor-depth.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def inverse_brown(ideal, coeff):
    raw = ideal.copy()
    k1,k2,p1,p2,k3 = coeff
    for _ in range(8):
        x,y = raw.T
        r2 = x*x+y*y
        radial = 1+k1*r2+k2*r2*r2+k3*r2*r2*r2
        tang = np.stack((2*p1*x*y+p2*(r2+2*x*x),p1*(r2+2*y*y)+2*p2*x*y),axis=1)
        raw = (ideal-tang)/radial[:,None]
    return raw


def fit(a,b):
    ac,bc = a.mean(axis=0),b.mean(axis=0)
    u,s,vt = np.linalg.svd((a-ac).T @ (b-bc))
    r = vt.T @ u.T
    if np.linalg.det(r)<0:
        vt[-1] *= -1
        r = vt.T @ u.T
    t = bc-r@ac
    T = np.eye(4)
    T[:3,:3],T[:3,3] = r,t
    return T


def robust_fit(a,b,rng):
    if len(a)<25:
        return None
    # Metric threshold combines sensor noise and subpixel matching error.
    threshold = .025 + .012*np.minimum(a[:,2],b[:,2])
    best = np.zeros(len(a),bool)
    for _ in range(160):
        idx = rng.choice(len(a),3,replace=False)
        if np.linalg.norm(np.cross(a[idx[1]]-a[idx[0]],a[idx[2]]-a[idx[0]])) < .0002:
            continue
        T = fit(a[idx],b[idx])
        good = np.linalg.norm(a@T[:3,:3].T+T[:3,3]-b,axis=1)<threshold
        if good.sum()>best.sum():
            best = good
            if best.mean()>.85:
                break
    if best.sum()<25 or best.mean()<.50:
        return None
    for _ in range(3):
        T = fit(a[best],b[best])
        residual = np.linalg.norm(a@T[:3,:3].T+T[:3,3]-b,axis=1)
        best = residual<threshold
    if best.sum()<25 or best.mean()<.5:
        return None
    if np.linalg.norm(np.std(a[best],axis=0)) < .12:
        return None
    return T,int(best.sum()),float(np.median(residual[best]))


def extract(root):
    out = root/'rgbd'
    out.mkdir(exist_ok=True)
    (out/'images').mkdir(exist_ok=True)
    (out/'depth_all').mkdir(exist_ok=True)
    (out/'features').mkdir(exist_ok=True)
    cal = json.loads((root/'calibration.json').read_text())
    c = mod.fields(next(v for k,v in cal.items() if k.endswith('/Color_0/camera_info')))
    d = mod.fields(next(v for k,v in cal.items() if k.endswith('/Depth_0/camera_info')))
    coeff = np.array([float(v) for v in c['coeffs'].split(',')])
    K = np.array([[float(c['fx']),0,float(c['ppx'])],[0,float(c['fy']),float(c['ppy'])],[0,0,1]])
    w,h = 848,477
    targetK = K.copy()
    targetK[:2] *= w/1280
    yy,xx = np.mgrid[:h,:w]
    ideal = np.stack(((xx.ravel()-targetK[0,2])/targetK[0,0],(yy.ravel()-targetK[1,2])/targetK[1,1]),axis=1)
    raw = inverse_brown(ideal,coeff)*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
    maps = raw.reshape(h,w,2).astype('f4')
    frames = json.loads((root/'capture_manifest.json').read_text())['frames']
    rng = np.random.default_rng(42)
    with sqlite3.connect(Path('20261029_140739.db3').resolve().as_uri()+'?mode=ro',uri=True) as db, sqlite3.connect(root/'colmap.db') as featuredb:
        topics = {n:i for i,n in db.execute('select id,name from topics')}
        def get_string(suffix):
            tid = next(i for n,i in topics.items() if n.endswith(suffix))
            return mod.capture.CDR(db.execute('select data from messages where topic_id=? limit 1',(tid,)).fetchone()[0]).string()
        ext = mod.fields(get_string('/Color_0/tf/ref_0'))
        R = np.array([float(v) for v in ext['rotation'].split(',')]).reshape(3,3,order='F')
        t = np.array([float(v) for v in ext['translation'].split(',')])
        tid = next(i for n,i in topics.items() if n.endswith('/Depth_0/image/data'))
        times = db.execute('select id,timestamp from messages where topic_id=? order by timestamp',(tid,)).fetchall()
        stamps = np.array([t for i,t in times],dtype=np.int64)
        rows = {n:(i,nr,nc,b) for i,n,nr,nc,b in featuredb.execute('select i.image_id,i.name,k.rows,k.cols,k.data from images i join keypoints k on k.image_id=i.image_id')}
        records=[]
        for index,f in enumerate(frames):
            name=f['name']
            record=dict(name=name,image_id=rows[name][0])
            records.append(record)
            if (out/'features'/(name+'.npz')).exists():
                continue
            j=np.searchsorted(stamps,f['timestamp_ns'])
            j=min([max(0,j-1),min(j,len(times)-1)],key=lambda k:abs(int(stamps[k])-f['timestamp_ns']))
            dt=abs(int(stamps[j])-f['timestamp_ns'])/1e9
            if dt>.02:
                raise ValueError((name,dt))
            blob=db.execute('select data from messages where id=?',(times[j][0],)).fetchone()[0]
            depth,header=mod.decode_depth(blob)
            z=depth*.001
            low=cv2.erode(z,np.ones((3,3),np.uint8))
            high=cv2.dilate(z,np.ones((3,3),np.uint8))
            valid=(low>.3)&(high<6)&((high-low)<np.maximum(.03,z*.02))
            # A 2x depth grid keeps subpixel-to-few-pixel RGB coverage while
            # reducing CPU projection work fourfold; no depth values are averaged.
            valid[1::2,:]=False
            valid[:,1::2]=False
            y,x=np.nonzero(valid)
            z=z[valid]
            xyz=np.stack(((x-float(d['ppx']))/float(d['fx'])*z,(y-float(d['ppy']))/float(d['fy'])*z,z),axis=1)@R.T+t
            raw=inverse_brown(xyz[:,:2]/xyz[:,2:3],coeff)*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
            i,nr,nc,b=rows[name]
            keys=np.frombuffer(b,'f4').reshape(nr,nc)[:,:2]
            dist,nearest=cKDTree(raw).query(keys)
            keyxyz=xyz[nearest].astype('f4')
            keyxyz[dist>2]=np.nan
            pixels=xyz[:,:2]/xyz[:,2:3]*[targetK[0,0],targetK[1,1]]+[targetK[0,2],targetK[1,2]]
            valid=(pixels[:,0]>=1)&(pixels[:,0]<w-1)&(pixels[:,1]>=1)&(pixels[:,1]<h-1)&(xyz[:,2]>0)
            pixels,z=pixels[valid],xyz[valid,2]
            key=np.rint(pixels[:,1]).astype(int)*w+np.rint(pixels[:,0]).astype(int)
            order=np.lexsort((z,key))
            _,first=np.unique(key[order],return_index=True)
            keep=order[first]
            keep=keep[rng.choice(len(keep),min(12000,len(keep)),replace=False)]
            np.savez_compressed(out/'depth_all'/(name+'.npz'),points=pixels[keep].astype('f4'),depths=z[keep].astype('f4'))
            im=np.asarray(Image.open(root/'raw_images'/name))
            im=cv2.remap(im,maps[:,:,0],maps[:,:,1],cv2.INTER_LINEAR)
            Image.fromarray(im).save(out/'images'/name,quality=96,subsampling=0)
            np.savez_compressed(out/'features'/(name+'.npz'),xyz=keyxyz,pair_delta_seconds=dt,header_delta_seconds=abs(header-f['header_seconds']))
            if index%25==0:
                print('RGBD extracted',index+1,len(frames),flush=True)
    (out/'calibration.json').write_text(json.dumps(dict(width=w,height=h,K=targetK.tolist(),frames=records),indent=2))


def register(root):
    out=root/'rgbd'
    cal=json.loads((out/'calibration.json').read_text())
    records=cal['frames']
    ids={f['image_id']:j for j,f in enumerate(records)}
    xyz=[np.load(out/'features'/(f['name']+'.npz'))['xyz'] for f in records]
    calibration=json.loads((root/'calibration.json').read_text())
    c=mod.fields(next(v for k,v in calibration.items() if k.endswith('/Color_0/camera_info')))
    K=np.array([[float(c['fx']),0,float(c['ppx'])],[0,float(c['fy']),float(c['ppy'])],[0,0,1]])
    coeff=np.array([float(v) for v in c['coeffs'].split(',')])
    rng=np.random.default_rng(42)
    edges=[]
    with sqlite3.connect(root/'colmap.db') as db:
        keys={i:np.frombuffer(b,'f4').reshape(n,c)[:,:2].astype('f8') for i,n,c,b in db.execute('select image_id,rows,cols,data from keypoints')}
        # Metric 3D verification also accepts low-parallax pairs rejected by
        # epipolar estimation. Prefer existing geometric inliers where present.
        candidates={r[0]:r for r in db.execute('select pair_id,rows,data from matches where rows>=30')}
        candidates.update({r[0]:r for r in db.execute('select pair_id,rows,data from two_view_geometries where rows>=30')})
        rows=list(candidates.values())
    rows.sort(key=lambda row: abs(ids.get(row[0]//2147483647,0)-ids.get(row[0]%2147483647,0)))
    for k,(pid,n,b) in enumerate(rows):
        a,bid=pid//2147483647,pid%2147483647
        if a not in ids or bid not in ids:
            continue
        i,j=ids[a],ids[bid]
        matches=np.frombuffer(b,np.uint32).reshape(n,2)
        x,y=xyz[i][matches[:,0]],xyz[j][matches[:,1]]
        good=np.isfinite(x).all(axis=1)&np.isfinite(y).all(axis=1)
        result=robust_fit(x[good],y[good],rng)
        if result is None:
            # PnP only needs source depth, unlike 3D-3D alignment; useful when
            # the target depth has holes or a narrow common depth field of view.
            srcgood=np.isfinite(x).all(axis=1)
            raw=(keys[bid][matches[:,1]]-[K[0,2],K[1,2]])/[K[0,0],K[1,1]]
            u,v=raw.T;r2=u*u+v*v;k1,k2,p1,p2,k3=coeff
            radial=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
            ideal=np.stack((u*radial+2*p1*u*v+p2*(r2+2*u*u),v*radial+p1*(r2+2*v*v)+2*p2*u*v),axis=1)
            pix=ideal*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
            if srcgood.sum()>=25:
                ok,rv,tv,inliers=cv2.solvePnPRansac(x[srcgood],pix[srcgood],K,None,iterationsCount=300,reprojectionError=3,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
                if ok and inliers is not None and len(inliers)>=25 and len(inliers)/srcgood.sum()>.45:
                    ii=inliers[:,0]
                    ok,rv,tv=cv2.solvePnP(x[srcgood][ii],pix[srcgood][ii],K,None,rv,tv,True,flags=cv2.SOLVEPNP_ITERATIVE)
                    T=np.eye(4);T[:3,:3]=cv2.Rodrigues(rv)[0];T[:3,3]=tv[:,0]
                    errors=np.linalg.norm(x[good]@T[:3,:3].T+T[:3,3]-y[good],axis=1)
                    # Require agreement wherever both sensors have valid depth.
                    if len(errors)<10 or np.median(errors)<.10:
                        result=T,len(ii),float(np.median(errors)) if len(errors) else .05
        if result:
            T,nin,err=result
            edges.append(dict(i=i,j=j,T=T.tolist(),inliers=nin,error=err))
        if k%500==0:
            print('RGBD matched',k,len(rows),'edges',len(edges),flush=True)
    (out/'edges.json').write_text(json.dumps(edges))
    print('RGBD_EDGES',len(edges),flush=True)


def optimize(root):
    import open3d as o3d
    import pycolmap
    out=root/'rgbd'
    cal=json.loads((out/'calibration.json').read_text())
    records=cal['frames']
    edges=json.loads((out/'edges.json').read_text())
    n=len(records)
    graph=np.zeros((n,n))
    for e in edges:
        graph[e['i'],e['j']]=graph[e['j'],e['i']]=1
    nc,labels=connected_components(csr_matrix(graph),directed=False)
    components=sorted([(int((labels==c).sum()),c) for c in range(nc)],reverse=True)
    selected=np.flatnonzero(labels==components[0][1])
    index={int(i):j for j,i in enumerate(selected)}
    edges=[e for e in edges if e['i'] in index and e['j'] in index]
    # Strongest edges initialize a maximum-confidence tree.
    weights=np.zeros((len(selected),len(selected)))
    lookup={}
    for k,e in enumerate(edges):
        i,j=index[e['i']],index[e['j']]
        weights[i,j]=weights[j,i]=(1+e['error']*30)/e['inliers']
        lookup[i,j]=lookup[j,i]=k
    tree=minimum_spanning_tree(csr_matrix(weights)).tocoo()
    adjacency=[[] for _ in selected]
    tree_edges=set()
    for i,j in zip(tree.row,tree.col):
        k=lookup[i,j]
        tree_edges.add(k)
        adjacency[i].append((j,k));adjacency[j].append((i,k))
    poses=[None]*len(selected)
    poses[0]=np.eye(4)
    todo=[0]
    for i in todo:
        for j,k in adjacency[i]:
            if poses[j] is not None:
                continue
            e=edges[k]
            T=np.array(e['T'])
            if index[e['i']]!=i:
                T=np.linalg.inv(T)
            poses[j]=poses[i]@np.linalg.inv(T)
            todo.append(j)
    pg=o3d.pipelines.registration.PoseGraph()
    for pose in poses:
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose))
    rejected=0
    for k,e in enumerate(edges):
        i,j=index[e['i']],index[e['j']]
        T=np.array(e['T'])
        predicted=np.linalg.inv(poses[j])@poses[i]
        discrepancy=np.linalg.inv(T)@predicted
        # Reject loop hypotheses that disagree grossly with the strongest tree.
        if k not in tree_edges and (np.linalg.norm(discrepancy[:3,3])>1 or Rotation.from_matrix(discrepancy[:3,:3]).magnitude()>.4):
            rejected+=1
            continue
        info=np.eye(6)*min(1000,e['inliers'])
        pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(i,j,T,info,k not in tree_edges))
    solver_report={'solver':'Open3D dense Levenberg-Marquardt'}
    if len(selected)>1200:
        sparse_spec=importlib.util.spec_from_file_location('sparse_pose_graph',Path(__file__).with_name('sparse-rgbd-pose-graph.py'))
        sparse_mod=importlib.util.module_from_spec(sparse_spec);sparse_spec.loader.exec_module(sparse_mod)
        optimized,solver_report=sparse_mod.optimize(poses,list(pg.edges))
        for node,pose in zip(pg.nodes,optimized):
            node.pose=pose
    else:
        o3d.pipelines.registration.global_optimization(pg,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=.07,edge_prune_threshold=.25,preference_loop_closure=1,reference_node=0))
    poses=np.stack([node.pose for node in pg.nodes])
    np.savez(out/'poses.npz',indices=selected,camtoworlds=poses)
    o3d.io.write_pose_graph(str(out/'pose_graph.json'),pg)
    dataset=out/'dataset'
    (dataset/'images').mkdir(parents=True,exist_ok=True)
    (dataset/'sparse').mkdir(exist_ok=True)
    (dataset/'sensor_depth').mkdir(exist_ok=True)
    K=np.array(cal['K']);w,h=cal['width'],cal['height']
    (dataset/'sparse/cameras.txt').write_text(f'1 PINHOLE {w} {h} {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}\n')
    image_lines=[]
    world_points=[];colors=[]
    split_path=root/'experiment_split.json'
    explicit_train=set(json.loads(split_path.read_text())['train']) if split_path.exists() else None
    if split_path.exists():
        (dataset/'experiment_split.json').write_bytes(split_path.read_bytes())
    rng=np.random.default_rng(42)
    for j,i in enumerate(selected):
        f=records[i];name=f['name']
        src=out/'images'/name;dst=dataset/'images'/name
        if not dst.exists():
            os.link(src,dst)
        src=out/'depth_all'/(name+'.npz');dst=dataset/'sensor_depth'/(name+'.npz')
        with np.load(src) as a:
            p,z=a['points'],a['depths']
        keep=rng.choice(len(z),min(4096,len(z)),replace=False)
        np.savez_compressed(dst,points=p[keep],depths=z[keep])
        w2c=np.linalg.inv(poses[j]);q=Rotation.from_matrix(w2c[:3,:3]).as_quat();t=w2c[:3,3]
        image_lines.append(f'{j+1} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} 1 {name}\n\n')
        # Training views only initialize geometry and color; same test split as loader.
        is_training = name in explicit_train if explicit_train is not None else j%8!=0
        if is_training:
            keep=rng.choice(len(z),min(1500,len(z)),replace=False)
            p,z=p[keep],z[keep]
            camera=np.column_stack(((p[:,0]-K[0,2])/K[0,0]*z,(p[:,1]-K[1,2])/K[1,1]*z,z))
            world_points.append(camera@poses[j,:3,:3].T+poses[j,:3,3])
            im=np.asarray(Image.open(out/'images'/name));pix=np.rint(p).astype(int)
            colors.append(im[pix[:,1],pix[:,0]])
    (dataset/'sparse/images.txt').write_text(''.join(image_lines))
    points=np.concatenate(world_points);rgb=np.concatenate(colors)
    # Centimeter-scale voxel selection, capped to keep initialization inexpensive.
    _,keep=np.unique(np.floor(points/.025).astype(np.int32),axis=0,return_index=True)
    if len(keep)>180000:
        keep=rng.choice(keep,180000,replace=False)
    with (dataset/'sparse/points3D.txt').open('w') as stream:
        for j,k in enumerate(keep):
            x,y,z=points[k];r,g,b=rgb[k]
            stream.write(f'{j+1} {x} {y} {z} {r} {g} {b} 0\n')
    # Explicit text read: reusing a directory containing old binaries would
    # otherwise silently prefer those binaries over the newly written text.
    model=pycolmap.Reconstruction()
    model.read_text(dataset/'sparse')
    if model.num_reg_images()!=len(selected) or model.num_points3D()!=len(keep):
        raise RuntimeError('Exported COLMAP dataset does not match optimized graph')
    model.write(dataset/'sparse')
    translation_errors=[];rotation_errors=[]
    for edge in pg.edges:
        predicted=np.linalg.inv(poses[edge.target_node_id])@poses[edge.source_node_id]
        delta=np.linalg.inv(edge.transformation)@predicted
        translation_errors.append(np.linalg.norm(delta[:3,3]))
        rotation_errors.append(np.degrees(Rotation.from_matrix(delta[:3,:3]).magnitude()))
    report=dict(method='calibrated RGB-D SIFT 3D RANSAC + CPU pose graph',pose_graph_solver=solver_report,selected_frames=n,
        registered_frames=len(selected),coverage_fraction=len(selected)/n,component_sizes=[c[0] for c in components],
        edges_before_pruning=len(edges),edges_optimized=len(pg.edges),gross_loops_rejected=rejected,
        initialized_points=len(keep),metric_units='meters before trainer normalization',
        translation_residual_percentiles_m=np.percentile(translation_errors,[50,90,99]).tolist(),
        rotation_residual_percentiles_degrees=np.percentile(rotation_errors,[50,90,99]).tolist(),
        unregistered_frames=[records[i]['name'] for i in range(n) if i not in index])
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print('RGBD_COMPLETE',report,flush=True)


def odometry(root):
    """Try dense RGB-D odometry only for missing immediate-neighbor links.

    Depth samples are reduced to a 212x119 grid with nearest fill <=2 pixels.
    Bidirectional consistency and independent depth-warp checks gate each link.
    """
    import open3d as o3d
    from scipy.ndimage import distance_transform_edt
    out=root/'rgbd'
    cal=json.loads((out/'calibration.json').read_text())
    records=cal['frames'];K=np.array(cal['K']);K[:2]/=4
    w,h=212,119
    intrinsic=o3d.camera.PinholeCameraIntrinsic(w,h,K[0,0],K[1,1],K[0,2],K[1,2])
    option=o3d.pipelines.odometry.OdometryOption()
    option.depth_min=.3;option.depth_max=6;option.depth_diff_max=.08
    option.iteration_number_per_pyramid_level=o3d.utility.IntVector([30,20,10])
    def load(i):
        name=records[i]['name']
        with np.load(out/'depth_all'/(name+'.npz')) as a:
            p,z=a['points']/4,a['depths']
        xy=np.rint(p).astype(int)
        valid=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        xy,z=xy[valid],z[valid]
        depth=np.full((h,w),np.inf,dtype='f4')
        np.minimum.at(depth,(xy[:,1],xy[:,0]),z)
        valid=np.isfinite(depth)
        distance,idx=distance_transform_edt(~valid,return_indices=True)
        depth=depth[tuple(idx)]
        depth[distance>2]=0
        image=np.asarray(Image.open(out/'images'/name).resize((w,h),Image.Resampling.BILINEAR)).copy()
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(image),o3d.geometry.Image(depth),depth_scale=1,depth_trunc=6,convert_rgb_to_intensity=True)
        return rgbd,depth
    def check(T,a,b):
        yy,xx=np.nonzero(a>0)
        z=a[yy,xx]
        xyz=np.stack(((xx-K[0,2])/K[0,0]*z,(yy-K[1,2])/K[1,1]*z,z),axis=1)
        xyz=xyz@T[:3,:3].T+T[:3,3]
        p=xyz[:,:2]/xyz[:,2:3]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
        p=np.rint(p).astype(int)
        valid=(p[:,0]>=0)&(p[:,0]<w)&(p[:,1]>=0)&(p[:,1]<h)&(xyz[:,2]>.3)
        p=p[valid];xyz=xyz[valid]
        measured=b[p[:,1],p[:,0]]
        valid=measured>0
        errors=np.abs(xyz[valid,2]-measured[valid])
        return len(errors)>1500 and np.mean(errors<.08)>.65 and np.median(errors)<.05
    edges=json.loads((out/'edges.json').read_text())
    existing={tuple(sorted((e['i'],e['j']))) for e in edges}
    accepted=[];rejected=[]
    source,adepth=load(0)
    jac=o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
    for i in range(len(records)-1):
        target,bdepth=load(i+1)
        if (i,i+1) not in existing:
            ok,T,info=o3d.pipelines.odometry.compute_rgbd_odometry(source,target,intrinsic,np.eye(4),jac,option)
            good=ok and np.linalg.norm(T[:3,3])<.65 and Rotation.from_matrix(T[:3,:3]).magnitude()<.6 and check(T,adepth,bdepth)
            if good:
                ok,back,_=o3d.pipelines.odometry.compute_rgbd_odometry(target,source,intrinsic,np.linalg.inv(T),jac,option)
                delta=back@T
                good=ok and np.linalg.norm(delta[:3,3])<.04 and Rotation.from_matrix(delta[:3,:3]).magnitude()<.035 and check(back,bdepth,adepth)
            if good:
                e=dict(i=i,j=i+1,T=T.tolist(),inliers=40,error=.03,method='dense_rgbd_bidirectional')
                edges.append(e);accepted.append(i)
            else:
                rejected.append(i)
        source,adepth=target,bdepth
        if i%25==0:
            print('Dense RGBD',i,'accepted',len(accepted),'rejected',len(rejected),flush=True)
    (out/'edges.json').write_text(json.dumps(edges))
    (out/'odometry_report.json').write_text(json.dumps(dict(accepted=accepted,rejected=rejected),indent=2))
    print('ODOMETRY_COMPLETE',len(accepted),len(rejected),flush=True)


def anchors(root):
    """Join RGB-D fragments using independently recovered metric-scaled SfM segments."""
    import pycolmap
    out=root/'rgbd'
    records=json.loads((out/'calibration.json').read_text())['frames']
    indices={f['name']:i for i,f in enumerate(records)}
    edges=json.loads((out/'edges.json').read_text())
    reports=[]
    for directory in sorted((root/'reconstruction_global').iterdir()):
        model=pycolmap.Reconstruction(directory)
        ratios=[]
        images=sorted(model.images.values(),key=lambda im:im.name)
        for im in images:
            xyz=np.load(out/'features'/(im.name+'.npz'))['xyz']
            valid=[(j,p.point3D_id) for j,p in enumerate(im.points2D) if p.has_point3D() and np.isfinite(xyz[j]).all()]
            if not valid:
                continue
            world=np.array([model.points3D[pid].xyz for j,pid in valid])
            W=np.asarray(im.cam_from_world().matrix())
            z=(world@W[:3,:3].T+W[:3,3])[:,2]
            depth=np.array([xyz[j,2] for j,pid in valid])
            good=(z>0)&(depth>.3)
            ratios.extend((z[good]/depth[good]).tolist())
        ratios=np.array(ratios);scale=float(np.median(ratios))
        error=float(np.median(np.abs(ratios/scale-1)))
        passed=len(ratios)>100 and error<.08
        reports.append(dict(model=directory.name,images=len(images),scale=scale,median_relative_scale_error=error,passed=passed))
        if not passed:
            continue
        poses=[]
        for im in images:
            W=np.eye(4);W[:3]=im.cam_from_world().matrix()
            pose=np.linalg.inv(W);pose[:3,3]/=scale
            poses.append(pose)
        for j in range(len(images)):
            for delta in (1,4):
                if j+delta>=len(images):
                    continue
                i,k=indices[images[j].name],indices[images[j+delta].name]
                T=np.linalg.inv(poses[j+delta])@poses[j]
                edges.append(dict(i=i,j=k,T=T.tolist(),inliers=1000,error=max(.005,error),method='metric_colmap_'+directory.name))
    (out/'edges.json').write_text(json.dumps(edges))
    (out/'anchor_report.json').write_text(json.dumps(reports,indent=2))
    print('ANCHORS_COMPLETE',reports,flush=True)


def bridge(root):
    """Use original higher-rate RGB-D frames to bridge failed 0.5s odometry links."""
    import open3d as o3d
    from scipy.ndimage import distance_transform_edt
    out=root/'rgbd'
    failed=json.loads((out/'odometry_report.json').read_text())['rejected']
    frames=json.loads((root/'capture_manifest.json').read_text())['frames']
    calibration=json.loads((root/'calibration.json').read_text())
    c=mod.fields(next(v for k,v in calibration.items() if k.endswith('/Color_0/camera_info')))
    d=mod.fields(next(v for k,v in calibration.items() if k.endswith('/Depth_0/camera_info')))
    coeff=np.array([float(v) for v in c['coeffs'].split(',')])
    K=np.array([[float(c['fx']),0,float(c['ppx'])],[0,float(c['fy']),float(c['ppy'])],[0,0,1]])
    smallK=K.copy();smallK[:2]/=1280/212
    w,h=212,119
    yy,xx=np.mgrid[:h,:w]
    rays=np.stack(((xx.ravel()-smallK[0,2])/smallK[0,0],(yy.ravel()-smallK[1,2])/smallK[1,1]),axis=1)
    raw=inverse_brown(rays,coeff)*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
    maps=raw.reshape(h,w,2).astype('f4')
    intrinsic=o3d.camera.PinholeCameraIntrinsic(w,h,smallK[0,0],smallK[1,1],smallK[0,2],smallK[1,2])
    opt=o3d.pipelines.odometry.OdometryOption();opt.depth_min=.3;opt.depth_max=6;opt.depth_diff_max=.08
    opt.iteration_number_per_pyramid_level=o3d.utility.IntVector([25,15,10])
    jac=o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
    edges=json.loads((out/'edges.json').read_text());accepted=[];rejected=[]
    with sqlite3.connect(Path('20261029_140739.db3').resolve().as_uri()+'?mode=ro',uri=True) as db:
        topics={n:i for i,n in db.execute('select id,name from topics')}
        def rows_for(suffix):
            tid=next(i for n,i in topics.items() if n.endswith(suffix))
            return db.execute('select id,timestamp from messages where topic_id=? order by timestamp',(tid,)).fetchall()
        rgbrows=rows_for('/Color_0/image/data');drows=rows_for('/Depth_0/image/data')
        rgbtimes=np.array([t for i,t in rgbrows],dtype=np.int64);dtimes=np.array([t for i,t in drows],dtype=np.int64)
        tid=next(i for n,i in topics.items() if n.endswith('/Color_0/tf/ref_0'))
        ext=mod.fields(mod.capture.CDR(db.execute('select data from messages where topic_id=? limit 1',(tid,)).fetchone()[0]).string())
        R=np.array([float(v) for v in ext['rotation'].split(',')]).reshape(3,3,order='F');t=np.array([float(v) for v in ext['translation'].split(',')])
        def nearest(stamps,target):
            j=np.searchsorted(stamps,target)
            return min([max(0,j-1),min(j,len(stamps)-1)],key=lambda k:abs(int(stamps[k])-int(target)))
        def load(timestamp):
            j=nearest(rgbtimes,timestamp);k=nearest(dtimes,rgbtimes[j])
            rgb=mod.capture.CDR(db.execute('select data from messages where id=?',(rgbrows[j][0],)).fetchone()[0]).image()[0]
            blob=db.execute('select data from messages where id=?',(drows[k][0],)).fetchone()[0]
            depth,_=mod.decode_depth(blob);depth=depth[::2,::2]*.001
            yy,xx=np.mgrid[:depth.shape[0],:depth.shape[1]];yy*=2;xx*=2
            valid=(depth>.3)&(depth<6)
            z=depth[valid];x=xx[valid];y=yy[valid]
            xyz=np.stack(((x-float(d['ppx']))/float(d['fx'])*z,(y-float(d['ppy']))/float(d['fy'])*z,z),axis=1)@R.T+t
            pix=xyz[:,:2]/xyz[:,2:3]*[smallK[0,0],smallK[1,1]]+[smallK[0,2],smallK[1,2]]
            pix=np.rint(pix).astype(int);valid=(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
            pix=pix[valid];z=xyz[valid,2]
            result=np.full((h,w),np.inf,dtype='f4');np.minimum.at(result,(pix[:,1],pix[:,0]),z)
            dist,idx=distance_transform_edt(~np.isfinite(result),return_indices=True)
            result=result[tuple(idx)];result[dist>1]=0
            color=cv2.remap(rgb,maps[:,:,0],maps[:,:,1],cv2.INTER_LINEAR)
            return o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.ascontiguousarray(color)),o3d.geometry.Image(result),depth_scale=1,depth_trunc=6,convert_rgb_to_intensity=True)
        for count,i in enumerate(failed):
            start,stop=frames[i]['timestamp_ns'],frames[i+1]['timestamp_ns']
            timestamps=np.linspace(start,stop,max(3,int((stop-start)/1e8)+1)).astype(np.int64)
            source=load(timestamps[0]);total=np.eye(4);good=True
            for timestamp in timestamps[1:]:
                target=load(timestamp)
                ok,T,info=o3d.pipelines.odometry.compute_rgbd_odometry(source,target,intrinsic,np.eye(4),jac,opt)
                eigen=np.linalg.eigvalsh(info)
                if not ok or eigen[0]/max(eigen[-1],1e-9)<1e-5 or np.linalg.norm(T[:3,3])>.25 or Rotation.from_matrix(T[:3,:3]).magnitude()>.25:
                    good=False;break
                ok,back,_=o3d.pipelines.odometry.compute_rgbd_odometry(target,source,intrinsic,np.linalg.inv(T),jac,opt)
                delta=back@T
                if not ok or np.linalg.norm(delta[:3,3])>.025 or Rotation.from_matrix(delta[:3,:3]).magnitude()>.02:
                    good=False;break
                total=T@total;source=target
            if good and np.linalg.norm(total[:3,3])<.8:
                edges.append(dict(i=i,j=i+1,T=total.tolist(),inliers=30,error=.04,method='higher_rate_rgbd_bridge'));accepted.append(i)
            else:
                rejected.append(i)
            if count%10==0:
                print('RGBD bridge',count,len(failed),'accepted',len(accepted),flush=True)
    (out/'edges.json').write_text(json.dumps(edges))
    (out/'bridge_report.json').write_text(json.dumps(dict(accepted=accepted,rejected=rejected),indent=2))
    print('BRIDGE_COMPLETE',len(accepted),len(rejected),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['extract','register','optimize','odometry','anchors','bridge'])
    ap.add_argument('--root',type=Path,default=Path('data/realsense_20261029_140739'))
    args=ap.parse_args()
    globals()[args.stage](args.root)
