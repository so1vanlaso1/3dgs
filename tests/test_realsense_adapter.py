"""Geometry checks for the recording-specific offline depth adapters."""
import importlib.util
from pathlib import Path
import struct

import numpy as np
import pycolmap


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def depth_blob(depth):
    b = bytearray(b'\x00\x01\x00\x00')
    def u32(n):
        b.extend(b'\x00'*((-(len(b)-4))%4))
        b.extend(struct.pack('<I',n))
    def string(s):
        data=s.encode()+b'\x00'
        u32(len(data));b.extend(data)
    u32(12);u32(500000000);string('depth')
    u32(depth.shape[0]);u32(depth.shape[1]);string('16UC1')
    b.append(0);u32(depth.shape[1]*2);u32(depth.nbytes);b.extend(depth.tobytes())
    return bytes(b)


def test_depth_cdr_and_camera_z_translation():
    m=script('prepare-sensor-depth.py')
    blob=depth_blob(np.full((8,10),1000,dtype='<u2'))
    depth,t=m.decode_depth(blob)
    assert t==12.5
    np.testing.assert_array_equal(depth,1000)
    intr=dict(fx='10',fy='10',ppx='5',ppy='4',coeffs='0,0,0,0,0')
    camera=pycolmap.Camera(model='PINHOLE',width=10,height=8,params=[10,10,5,4])
    pixels,z,_=m.align(blob,intr,intr,np.eye(3),np.array([-.05,0,.1]),camera,camera,.001)
    assert len(z)>10
    np.testing.assert_allclose(z,1.1,atol=1e-6)
    # Undo the target camera projection and extrinsic to recover source pixels.
    source_x=((pixels[:,0]-5)/10*z+.05)*10+5
    np.testing.assert_allclose(source_x,np.rint(source_x),atol=1e-5)


def test_rgbd_ransac_rejects_outliers_and_recovers_metric_transform():
    m=script('recover-rgbd-scene.py')
    rng=np.random.default_rng(9)
    a=rng.uniform([-.8,-.6,1],[.8,.6,3],(180,3))
    theta=.13
    r=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    t=np.array([.12,-.02,.04])
    b=a@r.T+t+rng.normal(0,.002,a.shape)
    b[:45]=rng.uniform(-2,2,(45,3))
    T,n,error=m.robust_fit(a,b,rng)
    assert n>=130 and error<.01
    np.testing.assert_allclose(T[:3,:3],r,atol=.003)
    np.testing.assert_allclose(T[:3,3],t,atol=.004)


def test_inverse_brown_round_trip():
    m=script('recover-rgbd-scene.py')
    ideal=np.random.default_rng(1).uniform(-.9,.9,(100,2))
    coeff=np.array([-.055945,.064586,-.000253,.000818,-.021141])
    raw=m.inverse_brown(ideal,coeff)
    x,y=raw.T;r2=x*x+y*y;k1,k2,p1,p2,k3=coeff
    f=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
    recovered=np.stack((x*f+2*p1*x*y+p2*(r2+2*x*x),y*f+p1*(r2+2*y*y)+2*p2*x*y),axis=1)
    np.testing.assert_allclose(recovered,ideal,atol=1e-7)
