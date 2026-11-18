"""Check metric pose recovery, transform direction, and fixed-camera gauge."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation


def test_sparse_pose_graph_recovers_consistent_loop():
    path=Path(__file__).resolve().parents[1]/'scripts/sparse-rgbd-pose-graph.py'
    spec=importlib.util.spec_from_file_location('sparse_graph_test',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    rng=np.random.default_rng(42)
    n=24
    truth=np.repeat(np.eye(4)[None],n,axis=0)
    angle=np.arange(n)*.15
    truth[:,:3,:3]=Rotation.from_euler('z',angle[:,None]).as_matrix()
    truth[:,:3,3]=np.column_stack([np.sin(angle),1-np.cos(angle),angle*.1])
    edges=[]
    for i in range(n):
        for delta in (1,4,10):
            j=(i+delta)%n
            edges.append(SimpleNamespace(source_node_id=i,target_node_id=j,
                transformation=np.linalg.inv(truth[j])@truth[i],information=np.eye(6)*100))
    initial=truth.copy()
    initial[1:,:3,3]+=rng.normal(0,.04,(n-1,3))
    initial[1:,:3,:3]=Rotation.from_rotvec(rng.normal(0,.02,(n-1,3))).as_matrix()@initial[1:,:3,:3]
    recovered,stats=mod.optimize(initial,edges,max_nfev=40)
    np.testing.assert_allclose(recovered,truth,atol=1e-4)
    np.testing.assert_array_equal(recovered[0],initial[0])
    assert stats['final_weighted_rms']<stats['initial_weighted_rms']*.01
