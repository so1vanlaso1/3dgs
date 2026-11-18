"""Prune copies of a trained scene and evaluate identical held-out cameras.

Run from the repository with .venv/Scripts/python.exe. No retraining or
ground-truth empty-space classification is performed.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'examples'))
OUT = REPO / 'results/realsense_floater_cleanup_20261030'
SOURCE = REPO / 'results/realsense_15fps_experiment/train30k/ckpts/final_post_update.pt'
DATA = REPO / 'data/realsense_20261029_140739_15fps/rgbd/dataset'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--components', action='store_true', help='Remove small disconnected voxel components instead of opacity/scale pruning.')
    ap.add_argument('--support-components', action='store_true', help='Use scale-aware sparse support connectivity instead of voxels.')
    args = ap.parse_args()
    global OUT
    if args.support_components:
        args.components = True
        OUT = REPO / 'results/realsense_support_cleanup_20261030'
    elif args.components:
        OUT = REPO / 'results/realsense_component_cleanup_20261030'
    OUT.mkdir(exist_ok=True)
    if sys.platform == 'win32':
        batch = REPO / '.tools/msvc/setup_x64.bat'
        lines = subprocess.check_output(f'call "{batch}" >nul && set', shell=True, text=True)
        for line in lines.splitlines():
            if '=' in line and not line.startswith('='):
                key, value = line.split('=', 1)
                os.environ[key] = value
    for key, value in dict(CUDA_HOME=str(REPO / '.tools/cuda-12.8'),
                           CUDA_PATH=str(REPO / '.tools/cuda-12.8'),
                           TORCH_EXTENSIONS_DIR=str(REPO / '.cache/torch_extensions'),
                           TORCH_HOME=str(REPO / '.cache/torch'),
                           TORCH_CUDA_ARCH_LIST='7.5', NUM_CHANNELS='3,4',
                           MAX_JOBS='2', OMP_NUM_THREADS='8').items():
        os.environ[key] = value
    os.environ['PATH'] = str(REPO / '.venv/Scripts') + os.pathsep + str(REPO / '.tools/cuda-12.8/bin') + os.pathsep + os.environ['PATH']
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from scipy.spatial import cKDTree
    from datasets.colmap import Parser
    from gsplat.rendering import rasterization
    from gsplat.exporter import export_splats, load_ply_to_splats
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    torch.set_num_threads(8)
    source_hash = sha(SOURCE)
    ckpt = torch.load(SOURCE, map_location='cpu', weights_only=True)
    s = ckpt['splats']
    assert all(torch.isfinite(v).all() for v in s.values())
    n = len(s['means'])
    parser = Parser(str(DATA), factor=1, normalize=True, load_exposure=False)
    opacity = s['opacities'].sigmoid().numpy()
    scale = s['scales'].exp().max(1).values.numpy()
    masks = {'original': np.ones(n, dtype=bool)}
    report = dict(source=str(SOURCE), source_sha256=source_hash, dataset=str(DATA),
                  source_gaussians=n, scene_scale=float(parser.scene_scale),
                  method='support_components' if args.support_components else ('voxel_components' if args.components else 'opacity_scale_isolation'), models={})
    if args.components:
        import runpy
        helpers = runpy.run_path(str(REPO / 'scripts/splat-connectivity.py'))
        component_filter = helpers['component_filter']
        configs = {'original': None, 'loose': 3., 'medium': 2., 'tight': 1.} if args.support_components else {'original': None, 'loose': .016, 'medium': .008, 'tight': .004}
        for label, fraction in configs.items():
            detail = {}
            if fraction:
                if args.support_components:
                    masks[label], detail = helpers['support_component_filter'](s['means'].numpy(), s['scales'].exp().numpy(), parser.scene_scale, fraction)
                else:
                    masks[label], detail = component_filter(s['means'].numpy(), fraction*parser.scene_scale, 500)
                    detail['voxel_size_scene_fraction'] = fraction
            report['models'][label] = dict(gaussians=int(masks[label].sum()), removed=int(n-masks[label].sum()), connectivity=detail)
            print(label, report['models'][label], flush=True)
    else:
        print('Computing 8th-neighbor distances', flush=True)
        distances = cKDTree(s['means'].numpy()).query(s['means'].numpy(), k=9, workers=8)[0][:, -1]
        isolation_limit = float(np.quantile(distances, .995))
        isolated = (distances > isolation_limit) & (opacity < .1)
        configs = {'original': None, 'mild': (.005, .1), 'balanced': (.01, .05), 'strong': (.02, .03)}
        report['isolation'] = dict(neighbor=8, distance_threshold=isolation_limit, percentile=99.5, opacity_below=.1, removed_if_combined=int(isolated.sum()))
        for label, thresholds in configs.items():
            if thresholds is not None:
                op, sz = thresholds
                masks[label] = (opacity >= op) & (scale <= sz * parser.scene_scale) & ~isolated
            report['models'][label] = dict(gaussians=int(masks[label].sum()), removed=int(n-masks[label].sum()), opacity_min=thresholds[0] if thresholds else None, max_axis_scale_scene_fraction=thresholds[1] if thresholds else None)
            print(label, report['models'][label], flush=True)
    candidates = [label for label in configs if label != 'original']
    split = json.loads((DATA.parent.parent / 'experiment_split.json').read_text())
    names = [name for name in parser.image_names if name in set(split['val'])]
    assert len(names) == 79
    name_to_index = {v: i for i, v in enumerate(parser.image_names)}
    chosen = [names[i] for i in np.linspace(0, len(names)-1, 8, dtype=int)]
    viewer = json.loads((REPO / 'results/realsense_15fps_experiment/viewer.json').read_text())
    if viewer['image_name'] not in chosen:
        chosen.append(viewer['image_name'])
    all_names = names + [v for v in chosen if v not in names]
    previews = {name: {} for name in chosen}
    ssim = StructuralSimilarityIndexMeasure(data_range=1.).cuda()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
    with torch.no_grad():
        for label in configs:
            mask = torch.from_numpy(masks[label])
            splats = {k: v[mask].cuda() for k, v in s.items()}
            colors = torch.cat([splats['sh0'], splats['shN']], 1)
            rows = []
            for j, name in enumerate(all_names):
                i = name_to_index[name]
                cid = parser.camera_ids[i]
                w, h = parser.imsize_dict[cid]
                view = torch.from_numpy(np.linalg.inv(parser.camtoworlds[i])[None]).float().cuda()
                K = torch.from_numpy(parser.Ks_dict[cid][None]).float().cuda()
                pred, _, _ = rasterization(splats['means'], splats['quats'], splats['scales'].exp(),
                    splats['opacities'].sigmoid(), colors, view, K, w, h,
                    sh_degree=int(math.sqrt(colors.shape[1])-1), packed=True,
                    near_plane=.01, far_plane=1e10)
                assert torch.isfinite(pred).all()
                pred = pred[0].clamp(0, 1)
                target_array = np.asarray(Image.open(parser.image_paths[i]).convert('RGB')).copy()
                target = torch.from_numpy(target_array).float().cuda()/255
                if name in names:
                    a, b = pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
                    rows.append(dict(name=name, psnr=float(-10*torch.log10((pred-target).square().mean())),
                                     ssim=float(ssim(a, b)), lpips=float(lpips(a, b))))
                    ssim.reset()
                    lpips.reset()
                if name in chosen:
                    previews[name][label] = Image.fromarray((pred.cpu().numpy()*255).astype('uint8'))
                    previews[name]['target'] = Image.fromarray(target_array)
                if j % 20 == 0:
                    print('RENDER', label, j, len(all_names), flush=True)
            report['models'][label]['metrics'] = {key: float(np.mean([r[key] for r in rows]))
                                                  for key in ('psnr', 'ssim', 'lpips')}
            report['models'][label]['per_view'] = rows
            del splats, colors
            torch.cuda.empty_cache()
    # Retain candidates only if average SSIM/LPIPS and PSNR degradation are small.
    base = report['models']['original']['metrics']
    eligible = []
    for label in candidates:
        m = report['models'][label]['metrics']
        passes = m['psnr'] >= base['psnr']-.25 and m['ssim'] >= base['ssim']-.01 and m['lpips'] <= base['lpips']+.01
        report['models'][label]['quality_guard_pass'] = passes
        if passes:
            eligible.append(label)
    selected = eligible[-1] if eligible else 'original'
    report.update(selected=selected, evaluated_heldout_views=len(names), finite_render_pixels=True,
                  selection_policy='Most pruning passing mean PSNR loss <=0.25 dB, SSIM loss <=0.01, LPIPS rise <=0.01; heuristic, not proof of floater removal.')
    for label in candidates:
        candidate = dict(step=ckpt['step'], splats={k: v[torch.from_numpy(masks[label])].clone() for k, v in s.items()})
        path = OUT / f'{label}.pt'
        torch.save(candidate, path)
        reloaded = torch.load(path, map_location='cpu', weights_only=True)
        assert all(torch.equal(candidate['splats'][k], reloaded['splats'][k]) for k in s)
        report['models'][label]['checkpoint'] = str(path)
        report['models'][label]['sha256'] = sha(path)
        report['models'][label]['reload_verified'] = True
    selected_path = SOURCE if selected == 'original' else OUT / f'{selected}.pt'
    selected_splats = torch.load(selected_path, map_location='cpu', weights_only=True)['splats']
    export_splats(**selected_splats, format='ply', save_to=str(OUT / 'scene.ply'))
    loaded = load_ply_to_splats(str(OUT / 'scene.ply'))
    assert len(loaded['means']) == len(selected_splats['means'])
    assert all(torch.isfinite(v).all() for v in loaded.values())
    report['ply_reload_verified'] = True
    report['original_unchanged'] = sha(SOURCE) == source_hash
    assert report['original_unchanged']
    labels = ['target', 'original'] + candidates
    sheet = Image.new('RGB', (5*360, len(chosen)*225), '#202020')
    draw = ImageDraw.Draw(sheet)
    for row, name in enumerate(chosen):
        full = Image.new('RGB', (5*848, 507), '#202020')
        fd = ImageDraw.Draw(full)
        for col, label in enumerate(labels):
            im = previews[name][label]
            full.paste(im, (col*848, 30))
            fd.text((col*848+8, 5), label+' | '+name, fill='white')
            small = im.copy(); small.thumbnail((360, 203))
            sheet.paste(small, (col*360, row*225+22))
            draw.text((col*360+4, row*225+3), label+' | '+name, fill='white')
        full.save(OUT / f'comparison_{name[:-4]}.jpg', quality=95)
    sheet.save(OUT / 'comparison_sheet.jpg', quality=94)
    # Same camera offsets for original/selected; these have no reference photos.
    if args.components:
        freeview_candidate = selected if selected != 'original' else candidates[0]
        free_sheet = Image.new('RGB', (1696, 3*507), '#202020')
        fd = ImageDraw.Draw(free_sheet)
        i = name_to_index[viewer['image_name']]
        cid = parser.camera_ids[i]
        w, h = parser.imsize_dict[cid]
        offsets = [(0, -.12, -.12), (-.15, 0, -.08), (.15, 0, -.08)]
        with torch.no_grad():
            for col, (label, mask_array) in enumerate([('original', masks['original']), (freeview_candidate, masks[freeview_candidate])]):
                sm = {k: v[torch.from_numpy(mask_array)].cuda() for k, v in s.items()}
                colors = torch.cat([sm['sh0'], sm['shN']], 1)
                K = torch.from_numpy(parser.Ks_dict[cid][None]).float().cuda()
                for row, offset in enumerate(offsets):
                    camera = parser.camtoworlds[i].copy()
                    camera[:3, 3] += camera[:3, :3] @ (np.asarray(offset)*parser.scene_scale)
                    view = torch.from_numpy(np.linalg.inv(camera)[None]).float().cuda()
                    rgb, _, _ = rasterization(sm['means'], sm['quats'], sm['scales'].exp(), sm['opacities'].sigmoid(), colors, view, K, w, h, sh_degree=3, packed=True, near_plane=.01, far_plane=1e10)
                    assert torch.isfinite(rgb).all()
                    im = Image.fromarray((rgb[0].clamp(0,1).cpu().numpy()*255).astype('uint8'))
                    free_sheet.paste(im, (col*848, row*507+30))
                    fd.text((col*848+8, row*507+5), f'{label} | offset {offset} x scene scale; no reference photo', fill='white')
                del sm, colors
                torch.cuda.empty_cache()
        free_sheet.save(OUT / 'freeview_comparison.jpg', quality=95)
        report['freeview_offsets_camera_axes_scene_scale'] = offsets
        report['freeview_candidate'] = freeview_candidate
    report['viewer'] = dict(checkpoint=str(selected_path), dataset=str(DATA),
                            camera_index=viewer['camera_index'])
    (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('COMPLETE', selected, {k:v['metrics'] for k,v in report['models'].items()}, flush=True)


if __name__ == '__main__':
    main()
