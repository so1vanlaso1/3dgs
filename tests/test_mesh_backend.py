import numpy as np
import open3d as o3d


def test_uniform_tsdf_recovers_known_plane():
    rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.full((100,100,3),100,np.uint8)),
        o3d.geometry.Image(np.ones((100,100),np.float32)),
        depth_scale=1,depth_trunc=3,convert_rgb_to_intensity=False)
    volume=o3d.pipelines.integration.UniformTSDFVolume(length=4,resolution=128,
        sdf_trunc=.08,color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        origin=np.array([-2.,-2.,0.]))
    volume.integrate(rgbd,o3d.camera.PinholeCameraIntrinsic(100,100,100,100,50,50),np.eye(4))
    mesh=volume.extract_triangle_mesh()
    xyz=np.asarray(mesh.vertices)
    assert len(mesh.triangles)>100 and np.isfinite(xyz).all()
    assert np.max(np.abs(xyz[:,2]-1))<.04
