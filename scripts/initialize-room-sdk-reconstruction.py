"""Prepare candidate-only metadata for RGB pose reconstruction."""
import json
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'data/realsense_20261030_134523_clean';CAND=ROOT/'data/room_improve_sdk_rectified_20261103'
def main():
    candidate=json.loads((CAND/'candidate_manifest.json').read_text());original=json.loads((SOURCE/'capture_manifest.json').read_text());assert len(candidate['frames'])==len(original['frames'])==477
    for name in ('capture_manifest.json','calibration_clean.json'):
        dest=CAND/name
        if dest.exists():raise FileExistsError(dest)
        shutil.copyfile(SOURCE/name,dest)
    (CAND/'reconstruction_input_note.json').write_text(json.dumps(dict(images='SDK-consistent candidate images',capture_manifest='Copied from original to preserve source IDs and original selection; image pixel mapping differs.',
        calibration='same output pinhole K and dimensions; corrected candidate remap',depth='Not copied; no depth supervision or pose graph is implied.',
        holdout='candidate_manifest.json and docs/room_baseline_heldout_20261103.csv preserve 60 source IDs; do not use default split after model registration changes.',
        limitation='Reconstruction processes all images to estimate poses; that does not mean holdout is blind to trajectory estimation or inspected imagery.'),indent=2),encoding='utf-8')
    print('candidate metadata ready; existing baseline untouched')
if __name__=='__main__':main()
