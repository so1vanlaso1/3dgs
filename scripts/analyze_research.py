"""Tables, paired uncertainty, budget plots, objective crop selection and failures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import t


def verify_artifacts(root,records):
    """Check full exports and evaluate four final post-update saved models."""
    protocol=json.loads((root/'protocol.json').read_text())
    for path,expected in protocol['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected, f'Source changed: {path}'
    for r in records:
        run=root/r['job']['id']
        expected_count={'room':39,'counter':30,'kitchen':35,'bonsai':37}[r['job']['scene']]
        images=sorted((run/'renders').glob('val_step6999_*.png'))
        assert len(images)==expected_count
        source=next((Path('data')/f'{r["job"]["scene"]}_factor2'/'images').iterdir())
        with Image.open(source) as im: width,height=im.size
        for path in images:
            with Image.open(path) as im: assert im.size==(width*2,height)
        with (run/'ply/point_cloud_6999.ply').open('rb') as stream:
            for _ in range(256):
                line=stream.readline().decode('ascii').strip()
                if line.startswith('element vertex '): assert int(line.split()[-1])==r['metrics']['num_GS']
                if line=='end_header': break
            else: raise ValueError('Invalid PLY header')
    import torch
    sys.path.insert(0,str(Path('examples').resolve()))
    from datasets.colmap import Dataset, Parser
    from gsplat.rendering import rasterization
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    results={}
    reload_scenes=sorted({r['job']['scene'] for r in records if r['job']['method']=='A04' and r['job']['seed']==42 and r['job']['phase']=='replication'})
    for scene in reload_scenes:
        record=next(r for r in records if r['job']['id']==f'replication/{scene}/A04_s42')
        run=root/record['job']['id']; saved=run/'checkpoint_reload.json'
        if saved.exists():
            result=json.loads(saved.read_text())
            assert result['checkpoint_sha256']==record['final_checkpoint_sha256']
            results[scene]=result; continue
        metrics=[PeakSignalNoiseRatio(data_range=1.).cuda(),StructuralSimilarityIndexMeasure(data_range=1.).cuda(),
                 LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda()]
        parser=Parser(f'data/{scene}_factor2',factor=1,normalize=True,test_every=8,load_exposure=False)
        dataset=Dataset(parser,split='val')
        params={k:v.cuda() for k,v in torch.load(run/'ckpts/final_post_update.pt',weights_only=True,map_location='cpu')['splats'].items()}
        values=[]
        with torch.no_grad():
            for item in range(len(dataset)):
                frame=dataset[item]; gt=frame['image'].cuda().float()[None]/255
                h,w=gt.shape[1:3]
                rgb,_,_=rasterization(means=params['means'],quats=params['quats'],scales=params['scales'].exp(),
                    opacities=params['opacities'].sigmoid(),colors=torch.cat([params['sh0'],params['shN']],1),
                    sh_degree=3,viewmats=torch.linalg.inv_ex(frame['camtoworld'].cuda()[None]).inverse,
                    Ks=frame['K'].cuda()[None],width=w,height=h,packed=True,near_plane=.01,far_plane=1e10)
                pred=rgb.clamp(0,1).permute(0,3,1,2); target=gt.permute(0,3,1,2)
                values.append([float(metric(pred,target)) for metric in metrics])
        reloaded=dict(zip(('psnr','ssim','lpips'),np.asarray(values).mean(0).tolist()))
        delta={k:reloaded[k]-record['metrics'][k] for k in reloaded}
        assert all(abs(delta[k])<tol for k,tol in [('psnr',.02),('ssim',.001),('lpips',.001)])
        result=dict(metrics=reloaded,delta=delta,checkpoint_sha256=record['final_checkpoint_sha256'],all_checks_pass=True)
        saved.write_text(json.dumps(result,indent=2)+'\n'); results[scene]=result
        del params,metrics; torch.cuda.empty_cache()
    (root/'artifact_verification.json').write_text(json.dumps(dict(all_checks_pass=True,runs=len(records),reloads=results),indent=2)+'\n')
    print(f'VERIFIED {len(records)} artifact sets and {len(results)} final post-update checkpoint reloads.')


def interval(values):
    values=np.asarray(values,dtype=float)
    mean=float(values.mean())
    sd=float(values.std(ddof=1)) if len(values)>1 else 0.
    half=float(t.ppf(.975,len(values)-1)*sd/np.sqrt(len(values))) if len(values)>1 else None
    return dict(n=len(values),mean=mean,sd=sd,ci95=None if half is None else [mean-half,mean+half],values=values.tolist())


def save_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def crop_analysis(root, records):
    out=root/'qualitative'; out.mkdir(exist_ok=True)
    view_rows=[]; tile_rows=[]; selected=[]; summaries={}
    font_path=Path('C:/Windows/Fonts/arial.ttf')
    font=ImageFont.truetype(str(font_path),18) if font_path.exists() else ImageFont.load_default()
    for scene in ('room','counter','kitchen','bonsai'):
        b=next(r for r in records if r['job']['id']==f'replication/{scene}/B00_s42')
        a=next(r for r in records if r['job']['id']==f'replication/{scene}/A04_s42')
        bp=Path(b['render_path']) if 'render_path' in b else root/b['job']['id']/'renders'
        ap=Path(a['render_path']) if 'render_path' in a else root/a['job']['id']/'renders'
        all_tiles=[]
        for path in sorted(bp.glob('val_step6999_*.png')):
            bb=np.asarray(Image.open(path).convert('RGB')); aa=np.asarray(Image.open(ap/path.name).convert('RGB'))
            width=bb.shape[1]//2
            assert np.array_equal(bb[:,:width],aa[:,:width]), 'GT panels differ'
            gt=bb[:,:width].astype(np.float64)/255
            br=bb[:,width:].astype(np.float64)/255; ar=aa[:,width:].astype(np.float64)/255
            be=np.square(br-gt).mean(-1); ae=np.square(ar-gt).mean(-1)
            gy,gx=np.gradient(gt.mean(-1)); detail=(np.abs(gx)+np.abs(gy))>.025
            view_rows.append(dict(scene=scene,view=path.name,b00_mse=float(be.mean()),a04_mse=float(ae.mean()),
                b00_psnr=float(-10*np.log10(max(be.mean(),1e-12))),
                a04_psnr=float(-10*np.log10(max(ae.mean(),1e-12))),
                mse_improvement=float((be-ae).mean())))
            size=192
            xs=sorted(set(list(range(0,width-size+1,96))+[width-size]))
            ys=sorted(set(list(range(0,len(gt)-size+1,96))+[len(gt)-size]))
            for y in ys:
                for x in xs:
                    bm=float(be[y:y+size,x:x+size].mean()); am=float(ae[y:y+size,x:x+size].mean())
                    row=dict(scene=scene,view=path.name,x=x,y=y,size=size,b00_mse=bm,a04_mse=am,mse_improvement=bm-am,
                             gt_edge_fraction=float(detail[y:y+size,x:x+size].mean()),width=width,height=len(gt))
                    all_tiles.append(row); tile_rows.append(row)
        scene_views=[v for v in view_rows if v['scene']==scene]
        summaries[scene]=dict(views=len(scene_views),views_worse=sum(v['mse_improvement']<0 for v in scene_views),
            tiles=len(all_tiles),tiles_worse=sum(v['mse_improvement']<0 for v in all_tiles),
            tile_mse_delta_quantiles=np.quantile([v['mse_improvement'] for v in all_tiles],[0,.1,.25,.5,.75,.9,1]).tolist())
        rows=[]
        interior=[r for r in all_tiles if r['x']>=96 and r['y']>=96 and r['x']+r['size']<=r['width']-96 and r['y']+r['size']<=r['height']-96]
        threshold=np.quantile([r['gt_edge_fraction'] for r in interior],.75)
        textured=[r for r in interior if r['gt_edge_fraction']>=threshold]
        choices=[('largest_gain',max(all_tiles,key=lambda r:r['mse_improvement'])),
                 ('interior_detail_gain',max(textured,key=lambda r:r['mse_improvement'])),
                 ('largest_regression',min(all_tiles,key=lambda r:r['mse_improvement'])),
                 ('largest_remaining_error',max(all_tiles,key=lambda r:r['a04_mse']))]
        for label,pick in choices:
            selected.append(dict(category=label,**pick))
            raw_b=np.asarray(Image.open(bp/pick['view']).convert('RGB'))
            raw_a=np.asarray(Image.open(ap/pick['view']).convert('RGB'))
            width=raw_b.shape[1]//2; x=pick['x']; y=pick['y']; size=pick['size']
            gt=raw_b[:,:width]; bimg=raw_b[:,width:]; aimg=raw_a[:,width:]
            full=Image.fromarray(gt); draw=ImageDraw.Draw(full); draw.rectangle((x,y,x+size,y+size),outline='red',width=5)
            images=[full,Image.fromarray(gt[y:y+size,x:x+size]),Image.fromarray(bimg[y:y+size,x:x+size]),
                    Image.fromarray(aimg[y:y+size,x:x+size])]
            for img in (bimg,aimg):
                err=np.abs(img.astype(float)-gt.astype(float)).mean(-1)/255
                # Fixed shared [0,.25] scale in both error panels and all crops.
                heat=(plt.get_cmap('magma')(np.clip(err[y:y+size,x:x+size]/.25,0,1))[:,:,:3]*255).astype('uint8')
                images.append(Image.fromarray(heat))
            canvas=Image.new('RGB',(6*256,320),'white'); d=ImageDraw.Draw(canvas)
            labels=['Location','Ground truth','B00','A04','B00 abs error','A04 abs error']
            for i,(im,title) in enumerate(zip(images,labels)):
                im.thumbnail((252,252),Image.Resampling.LANCZOS)
                canvas.paste(im,(i*256+(256-im.width)//2,32))
                d.text((i*256+8,7),title,fill='black',font=font)
            text=f'{scene} | {label.replace("_", " ")} | {pick["view"]} | ({x},{y},{size},{size}) | MSE B00={pick["b00_mse"]:.6f}, A04={pick["a04_mse"]:.6f}'
            d.text((8,291),text,fill='black',font=font)
            canvas.save(out/f'{scene}_{label}.png'); rows.append(canvas)
        sheet=Image.new('RGB',(1536,1280),'white')
        for i,img in enumerate(rows): sheet.paste(img,(0,i*320))
        sheet.save(out/f'{scene}_contact_sheet.png')
    save_csv(out/'per_view.csv',view_rows); save_csv(out/'all_windows.csv',tile_rows)
    (out/'selection.json').write_text(json.dumps(dict(selected=selected,summaries=summaries,
        selection='Exhaustive192x192 windows, stride96 plus boundaries. Gain/regression/residual extrema plus largest gain among interior windows in the top quartile of GT edge fraction. Interior margin96; gray gradient L1 threshold0.025. Seed42 only. Selection is illustrative and uses image outcomes, not independent evaluation.',
        precision='MSE recomputed on saved 8-bit PNGs; trainer metrics use floating-point renders.',
        error_scale='Mean absolute RGB error mapped identically from [0,0.25] to magma.'),indent=2)+'\n')
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--historical-crops',action='store_true')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args(); root=args.root
    if args.verify_only:
        records=[json.loads(path.read_text()) for path in root.glob('*/*/*/verified.json')]
        verify_artifacts(root,records); return
    if args.historical_crops:
        records=[]
        for scene in ('room','counter','kitchen','bonsai'):
            for method in ('B00','A04'):
                path=(Path('results/room_hq_gtx1660') if method=='B00' else Path('results/budget_allocation_experiment/longaxis700k')) if scene=='room' else Path(f'results/multiscene_a04_20261022/{scene}/{method}')
                records.append(dict(job=dict(id=f'replication/{scene}/{method}_s42'),render_path=str(path/'renders')))
        target=root/'historical_review'; target.mkdir(parents=True,exist_ok=True)
        summaries=crop_analysis(target,records)
        lines=['# Crop and failure review: historical one-seed runs','',
               'These crops analyze the preserved original B00/A04 Room run and the22 October three-scene campaign. They are separate from the new repeated-seed experiments. Crops show measured gain/regression extrema and remaining residuals; image content alone does not identify their cause.','',
               'All windows and view errors, exact crop locations and selection rules are saved in `qualitative/`. Error images use a shared scale. Overlapping windows are correlated and are not statistical replicates.', '',
               '[Written visual observations](VISUAL_REVIEW.md). Interior detail crops use high-edge-density windows with a96-pixel margin; selection is illustrative.','']
        for scene,s in summaries.items():
            lines += [f'## {scene}', '',f'A04 has larger saved-PNG RGB MSE on {s["views_worse"]}/{s["views"]} full views and {s["tiles_worse"]}/{s["tiles"]} scanned windows.', '',f'![Gain, regression and remaining error](qualitative/{scene}_contact_sheet.png)','']
        (target/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print(json.dumps(summaries,indent=2)); return
    records=json.loads((root/'all_runs.json').read_text())
    for record in records: assert record['checks_pass']
    verify_artifacts(root,records)
    figures=root/'figures'; figures.mkdir(exist_ok=True)
    lines=['# A04 research completion campaign','',
           'Measured campaign: 7,000 updates, four scenes, three seeds (42/43/44), independent camera-sampling RNG. Historical runs are excluded from the seed averages.',
           '', '## Repeated-seed comparison', '',
           'Values below are paired A04 minus B00, mean ± sample SD over three seeds. Intervals are paired Student-t 95% intervals (df=2); they are imprecise descriptive intervals, not broad generalization evidence.', '',
           '| Scene | PSNR delta (dB) | 95% interval | SSIM delta | LPIPS delta | Gaussian reduction |',
           '|---|---:|---|---:|---:|---:|']
    statistics={}; points=[]
    for scene in ('room','counter','kitchen','bonsai'):
        pairs=[]
        for seed in (42,43,44):
            b=next(r for r in records if r['job']['id']==f'replication/{scene}/B00_s{seed}')
            a=next(r for r in records if r['job']['id']==f'replication/{scene}/A04_s{seed}')
            assert b['initial_sha256']==a['initial_sha256'] and b['camera_sha256']==a['camera_sha256']
            for r in records:
                if r['job']['scene']==scene and r['job']['seed']==seed:
                    assert r['initial_sha256']==a['initial_sha256'] and r['camera_sha256']==a['camera_sha256']
            pairs.append({k:a['metrics'][k]-b['metrics'][k] for k in ('psnr','ssim','lpips')})
            pairs[-1]['reduction']=100*(1-a['metrics']['num_GS']/b['metrics']['num_GS'])
        stats={k:interval([r[k] for r in pairs]) for k in pairs[0]}; statistics[scene]=stats
        ps=stats['psnr']; ci=ps['ci95']
        lines.append(f'| {scene} | {ps["mean"]:+.4f} ± {ps["sd"]:.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {stats["ssim"]["mean"]:+.5f} | {stats["lpips"]["mean"]:+.5f} | {stats["reduction"]["mean"]:.2f}% |')
        points.append(ps)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for i,(scene,stats) in enumerate(statistics.items()):
        s=stats['psnr']; ax.errorbar(i,s['mean'],yerr=[[s['mean']-s['ci95'][0]],[s['ci95'][1]-s['mean']]],fmt='o',capsize=5,color='#2166ac')
        ax.scatter(np.full(3,i)+np.array([-.07,0,.07]),s['values'],s=22,color='#dc7633')
    ax.axhline(0,color='gray',lw=1); ax.set_xticks(range(4),list(statistics)); ax.set_ylabel('Paired PSNR gain (dB)'); ax.set_title('Three seeds: mean, individual runs and 95% t interval')
    fig.savefig(figures/'seed_uncertainty.png',dpi=180); plt.close(fig)
    lines+=['','## Split operator × clone replacement ablation','',
            'Room, three seeds, same budget/ranking/camera sequence. The LAS operator includes its placement, scale and opacity changes. This factorial isolates that operator bundle from replacing normally cloned parents.', '',
            '| Operator | Small-parent policy | PSNR mean ± SD | SSIM mean | LPIPS mean | Gaussians mean |', '|---|---|---:|---:|---:|---:|']
    cells={}
    for name,mode,policy in [('random_clone','random','clone'),('random_all','random','split'),('LAS_clone','long_axis','clone'),('A04','long_axis','split')]:
        runs=[r for r in records if r['job']['scene']=='room' and r['job']['method']==name and r['job']['phase'] in ('ablation','replication')]
        cells[name]=runs
        s=interval([r['metrics']['psnr'] for r in runs])
        lines.append(f'| {mode} | {policy} | {s["mean"]:.4f} ± {s["sd"]:.4f} | {np.mean([r["metrics"]["ssim"] for r in runs]):.5f} | {np.mean([r["metrics"]["lpips"] for r in runs]):.5f} | {np.mean([r["metrics"]["num_GS"] for r in runs]):,.0f} |')
    def paired(x,y):
        return interval([next(r for r in cells[x] if r['job']['seed']==seed)['metrics']['psnr']-
                         next(r for r in cells[y] if r['job']['seed']==seed)['metrics']['psnr'] for seed in (42,43,44)])
    effects={'LAS_with_clones':paired('LAS_clone','random_clone'),
             'LAS_with_all_split':paired('A04','random_all'),
             'clone_replacement_with_random':paired('random_all','random_clone'),
             'clone_replacement_with_LAS':paired('A04','LAS_clone')}
    effects['operator_clone_interaction']=interval(np.asarray(effects['LAS_with_all_split']['values'])-np.asarray(effects['LAS_with_clones']['values']))
    lines+=['','Paired PSNR effects (dB):','']+[f'- {name}: {s["mean"]:+.4f} ± {s["sd"]:.4f}; values {s["values"]}' for name,s in effects.items()]
    lines+=['','## Gaussian-budget operating points','',
            'Room, seed42. These are measured caps and final counts; the cap is an upper bound. The other settings are held fixed. No seed uncertainty estimate for this sweep.','',
            '| Cap | Final count | PSNR | SSIM | LPIPS | Checkpoint MiB | Wall seconds |','|---:|---:|---:|---:|---:|---:|---:|']
    budget_runs=[r for r in records if r['job']['phase']=='budgets']+[next(r for r in records if r['job']['id']=='replication/room/A04_s42')]
    budget_runs.sort(key=lambda r:r['job']['cap'])
    for r in budget_runs:
        m=r['metrics']; lines.append(f'| {r["job"]["cap"]:,} | {m["num_GS"]:,} | {m["psnr"]:.4f} | {m["ssim"]:.5f} | {m["lpips"]:.5f} | {r["checkpoint_bytes"]/2**20:.2f} | {r["wall_seconds"]:.1f} |')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,key in zip(axes,('psnr','ssim','lpips')):
        ax.plot([r['metrics']['num_GS']/1e3 for r in budget_runs],[r['metrics'][key] for r in budget_runs],'o-')
        for r in budget_runs: ax.annotate(f'{r["job"]["cap"]//1000}k cap',(r['metrics']['num_GS']/1e3,r['metrics'][key]),xytext=(4,5),textcoords='offset points',fontsize=8)
        ax.set_xlabel('Final Gaussians (thousands)'); ax.set_ylabel(key.upper()); ax.grid(alpha=.2)
    fig.savefig(figures/'budget_tradeoff.png',dpi=180); plt.close(fig)
    lines+=['','## Parameter tuning','',
            'Five settings: existing (0.45,0.6) plus a prespecified 2×2 grid at distances 0.30/0.50 and opacity multipliers 0.4/0.8. Selection uses previously exposed Room development views. It does not constitute untouched-test improvement evidence.','',
            '| Distance | Opacity | Room seed42 PSNR | SSIM | LPIPS | Gaussians |','|---:|---:|---:|---:|---:|---:|']
    tuning=[r for r in records if r['job']['phase']=='tuning']+[next(r for r in records if r['job']['id']=='replication/room/A04_s42')]
    for r in tuning:
        j=r['job']; m=r['metrics']; lines.append(f'| {j["distance"]:.2f} | {j["opacity"]:.2f} | {m["psnr"]:.4f} | {m["ssim"]:.5f} | {m["lpips"]:.5f} | {m["num_GS"]:,} |')
    if (root/'tuning_selection.json').exists():
        selection=json.loads((root/'tuning_selection.json').read_text()); choice=selection['winner']
        reference=next(r for r in records if r['job']['id']=='replication/room/A04_s42')
        lines+=['',f'Selected setting: distance={choice["distance"]}, opacity={choice["opacity"]}. Room seed42 count: {selection["selection_metrics"]["num_GS"]:,} versus reference {reference["metrics"]["num_GS"]:,}. Selection optimizes development PSNR under a common cap; it does not by itself demonstrate an improvement at equal final count.', '']
    for r in records:
        if r['job']['phase']=='confirmation':
            ref=next(s for s in records if s['job']['id']==f'replication/{r["job"]["scene"]}/A04_s{r["job"]["seed"]}')
            lines.append(f'- Frozen-setting confirmation {r["job"]["scene"]}, seed{r["job"]["seed"]}: PSNR delta {r["metrics"]["psnr"]-ref["metrics"]["psnr"]:+.4f} dB, LPIPS delta {r["metrics"]["lpips"]-ref["metrics"]["lpips"]:+.5f}.')
    crops=crop_analysis(root,records)
    lines+=['','## Image crops and failures','',
            'Selection exhaustively scans192×192 windows with stride96 and includes boundaries. Each scene shows the largest gain, largest regression, largest remaining A04 error, and largest gain among interior high-edge-density windows. These are deliberately selected examples. Full per-view/window tables and exact crop coordinates are saved in `qualitative/`. PNG-derived RGB errors have8-bit precision; native trainer metrics above use float renders.','',
            'A separate [written visual review of the preserved historical runs](historical_review/VISUAL_REVIEW.md) describes visible texture gains and failures. Historical observations are not pooled into the new seed results.','']
    for scene,s in crops.items():
        lines += [f'- {scene}: A04 has larger MSE on {s["views_worse"]}/{s["views"]} full views and {s["tiles_worse"]}/{s["tiles"]} overlapping windows.',
                  f'![{scene} gain, regression and residual](qualitative/{scene}_contact_sheet.png)','']
    ccd=root/'ccd/REPORT.md'
    lines+=['## CCD-GS mechanism study','',
            '[Mechanism report](ccd/REPORT.md)' if ccd.exists() else 'Mechanism study running; no outcome claimed yet.', '',
            '## Verification and scope','',
            'Every training run has a finite final checkpoint, matching final count, completed-update record, and recorded source/configuration. Initial-parameter and full 7,000-camera-sequence hashes match within every scene/seed comparison. Budget histories are checked against scheduled targets. `protocol.json`, per-run `verified.json`, `all_runs.json`, and `statistics.json` retain full precision. Timing includes startup/evaluation/export and is affected by other applications; these measurements do not establish controlled speedup.', '',
            'Four indoor scenes and three seeds support a scoped empirical report. No outdoor result, independent LAS placement-only effect, statistical population claim, or CCD-GS end-to-end allocation result is implied.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'statistics.json').write_text(json.dumps(dict(paired=statistics,factorial_effects=effects,crops=crops),indent=2)+'\n')
    flat=[dict(id=r['job']['id'],scene=r['job']['scene'],seed=r['job']['seed'],phase=r['job']['phase'],
               method=r['job']['method'],**r['metrics'],wall_seconds=r['wall_seconds'],checkpoint_bytes=r['checkpoint_bytes']) for r in records]
    save_csv(root/'runs.csv',flat)
    print(f'ANALYZED {len(records)} runs; crop panels and REPORT.md written.')


if __name__=='__main__': main()
