"""Source-ID-anchored local trajectory overlay; differences are not pose error."""
import importlib.util
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pycolmap

s=importlib.util.spec_from_file_location('local',Path(__file__).with_name('db3_local_geometry.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def cameras(path):
    model=pycolmap.Reconstruction(path);result={}
    for im in model.images.values():
        W=np.eye(4);W[:3]=im.cam_from_world().matrix();result[im.name]=np.linalg.inv(W)
    return result


def main():
    rows=m.read(m.DATA/'manifest.json')['frames'];byname={f['name']:f for f in rows}
    old=cameras(m.ROOT/'data/realsense_20261030_134523_clean/rgbd/dataset/sparse')
    fig=plt.figure(figsize=(12,5.5));records={}
    for i,region in enumerate(('matte','exhibit')):
        ax=fig.add_subplot(1,2,i+1,projection='3d');models={rate:cameras(m.RUN/region/f'rate{rate}/metric_joint') for rate in (3,10)}
        selected=sorted([n for n in models[3] if n in models[10] and byname[n]['selected_name']],key=lambda n:byname[n]['timestamp_ns'])
        anchor=selected[0];target=models[10][anchor]
        aligned={}
        for label,poses,first,color in [('3 FPS',models[3],models[3][anchor],'#c27813'),('10 FPS',models[10],target,'#176c9c')]:
            T=target@np.linalg.inv(first);names=sorted(poses,key=lambda n:byname[n]['timestamp_ns'])
            xyz=np.array([(T@poses[n])[:3,3] for n in names]);ax.plot(*xyz.T,label=label,color=color,linewidth=1.5)
            aligned[label]=dict(names=names,centers_m=xyz.tolist(),alignment=T.tolist())
        old_names=[byname[n]['selected_name'] for n in selected];T=target@np.linalg.inv(old[byname[anchor]['selected_name']])
        xyz=np.array([(T@old[n])[:3,3] for n in old_names]);ax.plot(*xyz.T,ls='--',color='#666666',label='Context baseline, common IDs')
        ax.set(title=region.capitalize(),xlabel='Local X (m)',ylabel='Local Y (m)',zlabel='Local Z (m)');ax.legend(fontsize=8)
        records[region]=dict(anchor_source_id=byname[anchor]['rgb_message_id'],aligned=aligned,
            baseline_common_names=old_names,baseline_centers_m=xyz.tolist(),baseline_alignment=T.tolist())
    fig.suptitle('Rigid first-camera alignment only; trajectory differences are not absolute errors',fontsize=11)
    fig.tight_layout();fig.savefig(m.RUN/'local_trajectory_overlay.png',dpi=160);plt.close(fig)
    m.write(m.RUN/'trajectory_overlay.json',dict(regions=records,alignment='First common source-ID camera only; no check structures fitted; independent local worlds remain unmerged'))


if __name__=='__main__':main()
