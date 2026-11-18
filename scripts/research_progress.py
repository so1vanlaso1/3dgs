"""Read-only campaign monitor; writes an inspectable progress table."""
import argparse
import json
from pathlib import Path
import re
import time


def snapshot(root):
    status=json.loads((root/'status.json').read_text())
    result=dict(status=status['status'],verified_runs=len(list(root.glob('*/*/*/verified.json'))))
    current=status.get('current',{}).get('id')
    if current:
        result['current']=current
        path=root/Path(current).parent/(Path(current).name+'.log')
        if path.exists():
            with path.open('rb') as stream:
                stream.seek(max(0,path.stat().st_size-65536)); tail=stream.read().decode('utf-8',errors='replace')
            matches=re.findall(r'(\d+)/7000',tail)
            if matches: result['updates']=int(matches[-1])
    ccd=root/'ccd/status.json'
    if ccd.exists(): result['ccd']=json.loads(ccd.read_text())
    return result


def refresh(root):
    protocol=json.loads((root/'protocol.json').read_text())
    status=json.loads((root/'status.json').read_text())
    records=[json.loads(p.read_text()) for p in root.glob('*/*/*/verified.json')]
    if status['status'].startswith('finished_'):
        (root/'PROGRESS.md').write_text('# Campaign finished at the user’s request\n\nNo further training is running.\n\n[Final completed-results report](REPORT.md)\n\n'+f'{len(records)} completed runs; all unfinished work skipped.\n',encoding='utf-8')
        return status['status'],len(records),None
    rows=['# Research campaign progress','',f'Status: **{status["status"]}**. Verified A04/baseline runs: **{len(records)}/{len(protocol["jobs"])}** planned runs, plus up to3 tuning confirmations.', '',
          'The complete campaign also includes36 CCD-GS checkpoint blocks after A04 training. This is a live status page; completed rows are measurements, pending rows have no result.','',
          '| Run | PSNR | SSIM | LPIPS | Gaussians | Wall seconds |','|---|---:|---:|---:|---:|---:|']
    for record in sorted(records,key=lambda r:r['job']['id']):
        m=record['metrics']; rows.append(f'| {record["job"]["id"]} | {m["psnr"]:.4f} | {m["ssim"]:.5f} | {m["lpips"]:.5f} | {m["num_GS"]:,} | {record["wall_seconds"]:.1f} |')
    current=status.get('current',{}).get('id')
    if current: rows+=['',f'Current run: `{current}`. Its training log is beside the run folder.']
    ccd=root/'ccd/status.json'
    if ccd.exists(): rows+=['',f'CCD-GS: {json.loads(ccd.read_text())}']
    (root/'PROGRESS.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    return status['status'],len(records),current


def main():
    cli=argparse.ArgumentParser(); cli.add_argument('--root',type=Path,required=True); cli.add_argument('--watch',action='store_true'); cli.add_argument('--json',action='store_true')
    args=cli.parse_args(); previous=None
    if args.json: print(json.dumps(snapshot(args.root))); return
    while True:
        try:
            result=refresh(args.root)
            if result!=previous: print(result,flush=True); previous=result
            if not args.watch or result[0] in ('complete','failed') or result[0].startswith('finished_'): break
        except (FileNotFoundError,json.JSONDecodeError):
            if not args.watch: raise
        time.sleep(30)


if __name__=='__main__': main()
