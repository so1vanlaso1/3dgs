"""Export native-width COLMAP images and transfer calibrated depth samples."""
import argparse
from pathlib import Path
import numpy as np
import pycolmap as p


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('data/realsense_20261029_140739'))
    ap.add_argument('--output-name',default='global_native_dataset')
    args=ap.parse_args();root=args.root;destination=root/args.output_name
    if destination.exists():
        raise RuntimeError('Output exists; choose a different --output-name to preserve trained inputs')
    old=p.Reconstruction(root/'global_dataset/sparse')
    model=p.Reconstruction(root/'reconstruction_global/0')
    options=p.UndistortCameraOptions();options.max_image_size=1280
    p.undistort_images(destination,root/'reconstruction_global/0',root/'raw_images',
        image_names=sorted(im.name for im in model.images.values()),undistort_options=options,num_threads=8)
    new=p.Reconstruction(destination/'sparse')
    images={im.name:im for im in new.images.values()}
    output=destination/'sensor_depth';output.mkdir()
    for im in old.images.values():
        A=np.asarray(old.cameras[im.camera_id].calibration_matrix())
        B=np.asarray(new.cameras[images[im.name].camera_id].calibration_matrix())
        with np.load(root/'global_sensor_depth'/(im.name+'.npz')) as a:
            rays=(a['points']-[A[0,2],A[1,2]])/[A[0,0],A[1,1]]
            points=rays*[B[0,0],B[1,1]]+[B[0,2],B[1,2]]
            np.savez_compressed(output/(im.name+'.npz'),points=points.astype('f4'),depths=a['depths'])
    print('NATIVE_DATASET_READY',new.num_reg_images(),flush=True)


if __name__=='__main__':
    main()
