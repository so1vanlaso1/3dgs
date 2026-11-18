"""Extract MP4 frames and create an RGB-only COLMAP dataset for gsplat."""
import argparse, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
import imageio_ffmpeg
import pycolmap as p

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--video',required=True,type=Path);ap.add_argument('--work',required=True,type=Path)
    ap.add_argument('--fps',type=float,default=2);ap.add_argument('--width',type=int,default=960);ap.add_argument('--min-registration',type=float,default=.8)
    ap.add_argument('--threads',type=int,default=min(8,os.cpu_count() or 4));args=ap.parse_args()
    video=args.video.resolve();work=args.work.resolve();work.mkdir(parents=True,exist_ok=True);status=work/'status.json'
    report={'video':str(video),'fps':args.fps,'max_image_side':args.width,'depth_used':False,'camera_model':'SIMPLE_RADIAL','camera_mode':'SINGLE','state':'starting','stages':{}}
    try:
        if not video.is_file():raise ValueError(f'Video not found: {video}')
        if args.fps<=0 or args.width<320 or not 0<args.min_registration<=1:raise ValueError('Invalid fps, width, or registration threshold')
        images=work/'raw_images';images.mkdir(exist_ok=False)
        report['source_bytes']=video.stat().st_size
        print('STAGE: Extract video frames',flush=True)
        # Scale the longest side down only; preserve aspect ratio and even dimensions.
        filt=f"fps={args.fps},scale=w='trunc(iw*min(1,{args.width}/max(iw,ih))/2)*2':h='trunc(ih*min(1,{args.width}/max(iw,ih))/2)*2'"
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-nostdin','-i',str(video),'-vf',filt,'-q:v','2',str(images/'frame_%06d.jpg')]
        with (work/'ffmpeg.log').open('w',encoding='utf8') as log: subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        frames=list(images.glob('*.jpg'));report['stages']['extraction']={'frames':len(frames),'command':command};write(status,report)
        if len(frames)<20:raise RuntimeError(f'Only {len(frames)} frames extracted; at least 20 are required. Increase fps or use a longer video.')
        print('STAGE: Extract COLMAP features (CPU)',flush=True)
        reader=p.ImageReaderOptions();reader.camera_model='SIMPLE_RADIAL'
        feat=p.FeatureExtractionOptions();feat.num_threads=args.threads;feat.max_image_size=args.width;feat.sift.max_num_features=4096;feat.sift.first_octave=0
        database=work/'colmap.db';p.extract_features(database,images,camera_mode=p.CameraMode.SINGLE,reader_options=reader,extraction_options=feat,device=p.Device.cpu)
        print('STAGE: Match sequential frames (CPU)',flush=True)
        match=p.FeatureMatchingOptions();match.num_threads=args.threads;pair=p.SequentialPairingOptions();pair.overlap=12;pair.quadratic_overlap=True
        p.match_sequential(database,matching_options=match,pairing_options=pair,device=p.Device.cpu)
        print('STAGE: Reconstruct camera positions and points',flush=True)
        opts=p.IncrementalPipelineOptions();opts.num_threads=args.threads;opts.random_seed=42;opts.max_num_models=5;opts.min_model_size=20
        sparse=work/'reconstruction';sparse.mkdir();models=p.incremental_mapping(database,images,sparse,options=opts)
        summary=[dict(id=int(i),registered_images=m.num_reg_images(),points3D=m.num_points3D(),mean_reprojection_error=m.compute_mean_reprojection_error()) for i,m in models.items()]
        report['stages']['reconstruction']={'models':summary};write(status,report)
        if not models:raise RuntimeError('COLMAP could not reconstruct a camera path. Check video overlap, motion blur, and texture.')
        best_id,best=max(models.items(),key=lambda kv:kv[1].num_reg_images());count=best.num_reg_images();fraction=count/len(frames)
        report['stages']['reconstruction'].update(selected_model=int(best_id),registered_images=count,registration_fraction=fraction,required_fraction=args.min_registration)
        if count<20 or best.num_points3D()<100 or fraction<args.min_registration:
            raise RuntimeError(f'Only {count}/{len(frames)} frames registered in the largest model ({fraction:.1%}); require at least {args.min_registration:.0%}, 20 images and 100 points. Inspect the reconstruction before training.')
        print(f'STAGE: Undistort {count}/{len(frames)} registered images',flush=True)
        dataset=work/'dataset';dataset.mkdir();und=p.UndistortCameraOptions();und.max_image_size=args.width
        names=sorted(im.name for im in best.images.values())
        p.undistort_images(dataset,sparse/str(best_id),images,image_names=names,undistort_options=und,num_threads=args.threads)
        report['state']='prepared';report['dataset']=str(dataset);report['limitations']='RGB only; SfM scale is arbitrary; registration is not proof of full coverage or camera accuracy; static scene and approximate shared intrinsics assumed.';write(status,report)
        print('PREPARATION COMPLETE:',dataset,flush=True)
    except Exception as e:
        report['state']='failed';report['error']=str(e);write(status,report);traceback.print_exc();return 1
    return 0
if __name__=='__main__':sys.exit(main())
