"""Recompute DB3 decisions from saved measurements without replacing evidence."""
import argparse
import hashlib
import json
from pathlib import Path
from db3_acceptance import paired_decisions

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'results/db3_r2_r3_20261110'


def main(destination):
    if destination.exists():raise FileExistsError(destination)
    records={}
    for region,suffix in [('matte',''),('exhibit',''),('exhibit','_supported_15k')]:
        out=CAMPAIGN/region/'pilot';path=out/f'paired_evaluation{suffix}.json'
        read=lambda p:json.loads(p.read_text(encoding='utf-8'))
        old=read(path);protocol=read(out/'evaluation_protocol.json');split=read(out/'dataset/experiment_split.json')
        assert hashlib.sha256((out/'evaluation_protocol.json').read_bytes()).hexdigest()==old['protocol_sha256']
        check=paired_decisions(old['results'],split['val'],protocol['regions_normalized_xyxy'],matched_compute=not bool(suffix))
        changes={arm:dict(old=old['pipeline_against_baseline'][arm]['quantitative_pass'],
                          rechecked=rec['quantitative_pass']) for arm,rec in check['pipeline_against_baseline'].items()}
        records[region+suffix]=dict(input=str(path.relative_to(ROOT)),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            pipeline_status=changes,policy_gate_unchanged=old['quantitative_promotion_pass']==check['quantitative_promotion_pass'],
            decisions=check)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(dict(records=records,
        scope='Recomputed acceptance only; original rendered measurements unchanged. Unsupported checks fail. Source roster fixed.'),indent=2),encoding='utf-8')
    for name,rec in records.items():print(name,rec['pipeline_status'],'policy unchanged',rec['policy_gate_unchanged'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
