"""Fix candidate split key while preserving the failed versions."""
import json
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'data/room_improve_sdk_rectified_20261103'
for path in (ROOT/'experiment_split.json',ROOT/'rgbd/dataset/experiment_split.json'):
    before=path.with_name('experiment_split.before_val_fix.json')
    if before.exists():raise FileExistsError(before)
    data=json.loads(path.read_text());assert 'test' in data and 'val' not in data
    shutil.copyfile(path,before);data['val']=data.pop('test');path.write_text(json.dumps(data,indent=2),encoding='utf-8')
    assert len(data['train'])==417 and len(data['val'])==60 and not set(data['train'])&set(data['val'])
print('corrected candidate split key: train 417 / val 60')
