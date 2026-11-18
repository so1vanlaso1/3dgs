"""Freeze current baseline at R0 source-ID anchors, including start/end context."""
import importlib.util
import math
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'examples')]
spec=importlib.util.spec_from_file_location('r0',ROOT/'scripts/run-db3-r0-r1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from datasets.colmap import Parser
from gsplat.rendering import rasterization


@torch.no_grad()
def main():
    out=m.RUN/'fixed_views';out.mkdir(exist_ok=False)
    parser=Parser(str(m.BASE/'rgbd/dataset'),factor=1,normalize=True,load_exposure=False)
    frozen=m.read(m.RUN/'baseline_manifest.json');frozen_ckpt=next(x for x in frozen['files'] if x['source'].endswith('final_post_update.pt'))
    ckpt=m.RUN/'baseline'/frozen_ckpt['source']
    assert m.sha(ckpt)==frozen_ckpt['sha256']
    s=torch.load(ckpt,map_location='cpu',weights_only=True)['splats']
    assert all(torch.isfinite(v).all() for v in s.values())
    s={k:v.cuda() for k,v in s.items()};colors=torch.cat([s['sh0'],s['shN']],1);degree=int(math.sqrt(colors.shape[1])-1)
    frames={f['name']:f for f in m.read(m.BASE/'capture_manifest.json')['frames']}
    origin=min(f['timestamp_ns'] for f in frames.values());times=np.array([(frames[n]['timestamp_ns']-origin)/1e9 for n in parser.image_names])
    requests=[(0,'kiosk/map'),(10,'near/mid opaque control'),(43,'wide room'),(61,'distant doorway'),(68,'exhibit/rails'),(110,'reflective case'),(157,'end-of-route context')]
    rows=[];sheet=Image.new('RGB',(848,7*265),'white');draw=ImageDraw.Draw(sheet)
    for row,(sec,label) in enumerate(requests):
        i=int(np.argmin(abs(times-sec)));name=parser.image_names[i];f=frames[name];cid=parser.camera_ids[i];ow,oh=parser.imsize_dict[cid]
        w,h=848,470;K=parser.Ks_dict[cid].copy();K[0]*=w/ow;K[1]*=h/oh;pose=parser.camtoworlds[i]
        rgb,alpha,_=rasterization(s['means'],s['quats'],s['scales'].exp(),s['opacities'].sigmoid(),colors,
            torch.from_numpy(np.linalg.inv(pose)[None]).float().cuda(),torch.from_numpy(K[None]).float().cuda(),w,h,sh_degree=degree,packed=True)
        pixels=rgb[0].clamp(0,1).cpu().numpy();assert np.isfinite(pixels).all()
        im=Image.fromarray((pixels*255).astype('uint8'));im.save(out/f'{sec:03d}s_render.png')
        capture=Image.open(m.BASE/'images'/name).convert('RGB').resize((w,h),Image.Resampling.BILINEAR);capture.save(out/f'{sec:03d}s_capture.png')
        sheet.paste(capture.resize((424,235)),(0,row*265+30));sheet.paste(im.resize((424,235)),(424,row*265+30))
        draw.text((5,row*265+7),f'{label}: {times[i]:.3f}s {name} | capture / preserved 120k model',fill='black')
        rows.append(dict(region=label,requested_seconds=sec,actual_seconds=float(times[i]),image_name=name,
            rgb_message_id=f['message_id'],depth_message_id=f['depth_message_id'],camera_to_world_normalized=pose.tolist(),
            K=K.tolist(),width=w,height=h,render=f'{sec:03d}s_render.png',capture=f'{sec:03d}s_capture.png',
            mean_abs_rgb_difference=float(abs(pixels-np.asarray(capture)/255.).mean())))
        print(label,name,flush=True)
    sheet.save(out/'capture_render_sheet.jpg',quality=94)
    m.write(out/'cameras.json',dict(checkpoint_sha256=frozen_ckpt['sha256'],gaussians=len(s['means']),
        normalization_transform=parser.transform.tolist(),all_image_names=parser.image_names,
        all_camera_to_world_normalized=parser.camtoworlds.tolist(),views=rows,
        render_settings=dict(sh_degree=degree,packed=True,background='black'),
        scope='Source-ID-anchored baseline. No candidate training and no geometry accuracy inference.'))


if __name__=='__main__':main()
