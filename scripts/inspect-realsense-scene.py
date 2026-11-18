"""Reload a splat, validate tensors, benchmark CUDA rendering, and save a preview."""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
import torch
from datasets.colmap import Parser
from gsplat.rendering import rasterization
from gsplat.exporter import export_splats, load_ply_to_splats


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt',type=Path,required=True)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--width',type=int,default=848)
    ap.add_argument('--benchmark-context',default='No other reconstruction jobs running; desktop applications remain open')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    ckpt=torch.load(args.ckpt,map_location='cpu',weights_only=True)
    finite=all(torch.isfinite(v).all().item() for v in ckpt['splats'].values())
    if not finite:
        raise ValueError('Nonfinite checkpoint tensors')
    ply=args.output/'scene.ply'
    export_splats(**{k:ckpt['splats'][k] for k in ('means','scales','quats','opacities','sh0','shN')},
                  format='ply',save_to=str(ply))
    reloaded=load_ply_to_splats(str(ply))
    ply_ok=len(reloaded['means'])==len(ckpt['splats']['means']) and all(torch.isfinite(v).all().item() for v in reloaded.values())
    if not ply_ok:
        raise ValueError('PLY reload validation failed')
    del reloaded
    splats={k:v.cuda() for k,v in ckpt['splats'].items()}
    n=len(splats['means'])
    scales=torch.exp(splats['scales']);opacities=torch.sigmoid(splats['opacities'])
    colors=torch.cat((splats['sh0'],splats['shN']),dim=1)
    degree=int(math.sqrt(colors.shape[1])-1)
    parser=Parser(str(args.data_dir),factor=1,normalize=True,load_exposure=False)
    def render(i):
        cid=parser.camera_ids[i]
        oldw,oldh=parser.imsize_dict[cid]
        width=args.width;height=max(2,2*round(oldh*width/oldw/2))
        K=parser.Ks_dict[cid].copy();K[0]*=width/oldw;K[1]*=height/oldh
        view=torch.from_numpy(np.linalg.inv(parser.camtoworlds[i])[None]).float().cuda()
        K=torch.from_numpy(K[None]).float().cuda()
        rgb,alpha,_=rasterization(splats['means'],splats['quats'],scales,opacities,colors,view,K,width,height,sh_degree=degree,packed=True)
        return rgb[0].clamp(0,1),width,height
    for _ in range(5):
        render(0)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    latencies=[]
    for j in range(50):
        i=int(j*(len(parser.image_names)-1)/49)
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();render(i);end.record();torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))
    selected=set(np.linspace(0,len(parser.image_names)-1,8,dtype=int).tolist())
    sheet=Image.new('RGB',(1280,4*200),'#222222');draw=ImageDraw.Draw(sheet)
    writer=imageio.get_writer(args.output/'capture_path.mp4',fps=4,macro_block_size=1,codec='libx264',quality=8)
    for i,name in enumerate(parser.image_names):
        rgb,width,height=render(i)
        pixels=rgb.cpu().numpy()
        if not np.isfinite(pixels).all():
            raise ValueError(f'Nonfinite rendered pixels in {name}')
        generated=(pixels*255).astype('uint8')
        reference=np.asarray(Image.open(parser.image_paths[i]).resize((width,height),Image.Resampling.LANCZOS))
        canvas=np.concatenate((reference,generated),axis=1)
        writer.append_data(canvas)
        if i in selected:
            j=sorted(selected).index(i)
            thumb=Image.fromarray(canvas);thumb.thumbnail((640,180))
            x=(j%2)*640;y=(j//2)*200
            sheet.paste(thumb,(x,y+20));draw.text((x+4,y+3),name+' | capture left / splat right',fill='white')
            Image.fromarray(canvas).save(args.output/f'comparison_{i:04d}.jpg',quality=94)
    writer.close();sheet.save(args.output/'comparison_sheet.jpg',quality=94)
    peak=torch.cuda.max_memory_allocated()/1024**3
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader'],text=True).strip()
    report=dict(checkpoint=str(args.ckpt.resolve()),dataset=str(args.data_dir.resolve()),step=ckpt['step'],
        finite_tensors=finite,finite_video_pixels=True,ply_reload_verified=ply_ok,ply_bytes=ply.stat().st_size,
        gaussians=n,sh_degree=degree,registered_views=len(parser.image_names),
        render_width=width,render_height=height,benchmark_frames=50,warmup_frames=5,
        render_ms_median=float(np.median(latencies)),render_ms_p90=float(np.percentile(latencies,90)),
        render_fps_from_mean=1000/float(np.mean(latencies)),peak_render_allocated_GiB=peak,
        gpu_snapshot=gpu,benchmark_scope='CUDA rasterization plus camera tensor preparation; excludes display, image read/write and video encoding',
        benchmark_context=args.benchmark_context,
        video='Captured poses in filename order, 4 frames/s; capture left, rendered splat right. Disconnected temporal gaps jump between views.')
    (args.output/'verification.json').write_text(json.dumps(report,indent=2))
    print('RELOAD_RENDER_VERIFIED',json.dumps(report),flush=True)


if __name__=='__main__':
    main()
