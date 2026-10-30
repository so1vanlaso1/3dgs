"""Sparse robust SE(3) pose-graph optimization for multi-thousand-frame captures."""
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation


def optimize(poses, edges, max_nfev=80):
    poses=np.array(poses,dtype=float,copy=True)
    n=len(poses)
    source=np.array([e.source_node_id for e in edges],dtype=int)
    target=np.array([e.target_node_id for e in edges],dtype=int)
    measurement=np.stack([e.transformation for e in edges])
    confidence=np.sqrt(np.array([np.trace(e.information)/6 for e in edges])/100)
    x0=np.concatenate([Rotation.from_matrix(poses[1:,:3,:3]).as_rotvec(),poses[1:,:3,3]],axis=1).ravel()
    fixedR=poses[0,:3,:3].copy();fixedt=poses[0,:3,3].copy()
    observedRinv=measurement[:,:3,:3].transpose(0,2,1)
    def unpack(x):
        q=x.reshape(n-1,6)
        R=np.concatenate([fixedR[None],Rotation.from_rotvec(q[:,:3]).as_matrix()])
        t=np.concatenate([fixedt[None],q[:,3:]])
        return R,t
    count=0;last=time.monotonic()
    def residual(x):
        nonlocal count,last
        R,t=unpack(x)
        inv_target=R[target].transpose(0,2,1)
        predictedR=inv_target@R[source]
        predictedt=np.einsum('nij,nj->ni',inv_target,t[source]-t[target])
        deltaR=observedRinv@predictedR
        deltat=np.einsum('nij,nj->ni',observedRinv,predictedt-measurement[:,:3,3])
        out=np.concatenate([Rotation.from_matrix(deltaR).as_rotvec(),deltat],axis=1)*confidence[:,None]
        count+=1
        if time.monotonic()-last>20:
            print('SPARSE_GRAPH evaluations',count,'RMS',float(np.sqrt(np.mean(out*out))),flush=True)
            last=time.monotonic()
        return out.ravel()
    row=[];col=[]
    for nodes in (source,target):
        valid=nodes>0
        r=np.arange(len(edges))[valid,None,None]*6+np.arange(6)[None,:,None]
        c=(nodes[valid,None,None]-1)*6+np.arange(6)[None,None,:]
        r,c=np.broadcast_arrays(r,c)
        row.extend(r.ravel());col.extend(c.ravel())
    sparsity=coo_matrix((np.ones(len(row)),(row,col)),shape=(len(edges)*6,(n-1)*6)).tocsr()
    before=residual(x0)
    result=least_squares(residual,x0,jac_sparsity=sparsity,method='trf',tr_solver='lsmr',
        loss='soft_l1',f_scale=.03,max_nfev=max_nfev,ftol=1e-5,xtol=1e-6,gtol=1e-5,
        tr_options={'maxiter':150},verbose=1)
    R,t=unpack(result.x)
    poses[:,:3,:3]=R;poses[:,:3,3]=t
    after=residual(result.x)
    assert np.isfinite(poses).all()
    stats=dict(solver='scipy sparse SE3 least_squares, soft_l1 scale=0.03; first camera fixed',
               nfev=result.nfev,status=result.status,message=result.message,
               initial_weighted_rms=float(np.sqrt(np.mean(before**2))),
               final_weighted_rms=float(np.sqrt(np.mean(after**2))),
               cost=float(result.cost),optimality=float(result.optimality))
    print('SPARSE_GRAPH_COMPLETE',stats,flush=True)
    return poses,stats
