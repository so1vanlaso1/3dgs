"""Check candidate provenance, disjointness, depth-source reprojection and PLY."""
import csv
import json
from pathlib import Path
import numpy as np
import pycolmap
from scipy.spatial import cKDTree
from plyfile import PlyData

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/realsense_20261030_134523_clean';OUT=ROOT/'results/room_improve_full_source_cloud_20261103'
PLY_FILES=('development_only_unfiltered.ply','development_only_uncertainty_colors.ply')
def main():
    r=json.loads((OUT/'report.json').read_text());dev=r['development_images'];check=r['check_images']
    with (ROOT/'docs/room_baseline_heldout_20261103.csv').open(newline='',encoding='utf-8') as f:held={x['image_name'] for x in csv.DictReader(f)}
    assert len(dev)==387 and len(check)==30 and not set(dev)&set(check) and not held&set(dev) and not held&set(check)
    with np.load(OUT/'evidence.npz') as f:pts=f['points_m'];source=f['source_group_index'];counts=f['check_counts'];colors=f['colors']
    assert len(pts)==180000 and len(source)==len(pts) and np.isfinite(pts).all() and source.min()>=0 and source.max()<len(dev)
    model=pycolmap.Reconstruction(DATA/'rgbd/dataset/sparse');images={i.name:i for i in model.images.values()};cam=next(iter(model.cameras.values()));fx,fy,cx,cy=cam.params
    ix=np.random.default_rng(888).choice(len(pts),200,replace=False);maxpixel=0.;maxdepth=0.
    for group in np.unique(source[ix]):
        sel=ix[source[ix]==group];im=images[dev[int(group)]];tf=im.cam_from_world().matrix();q=pts[sel]@tf[:,:3].T+tf[:,3];z=q[:,2];uv=np.c_[fx*q[:,0]/z+cx,fy*q[:,1]/z+cy]
        with np.load(DATA/'aligned_depth_meters'/(im.name+'.npz')) as d:xy=d['points'];depth=d['depths']
        dist,ind=cKDTree(xy).query(uv);maxpixel=max(maxpixel,float(dist.max()));maxdepth=max(maxdepth,float(abs(z-depth[ind]).max()))
    assert maxpixel<.001 and maxdepth<.00001
    for name in PLY_FILES:
        ply=PlyData.read(OUT/name)['vertex'].data;assert len(ply)==len(pts) and np.array_equal(ply['x'],pts[:,0].astype('f4'))
    assert (counts[:,1:]<=counts[:,:1]).all()
    report=dict(source_frames=len(dev),check_frames=len(check),heldout_frames=len(held),source_check_overlap=0,heldout_overlap=0,points=len(pts),
        sample_point_source_checks=len(ix),max_source_pixel_difference=maxpixel,max_source_z_difference_m=maxdepth,
        ply_reload='both 180000 vertices, x coordinate exact in float32',metric='Source reprojection verifies provenance and code path, not physical alignment or geometry.',seed=888)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
