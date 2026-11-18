"""Distinguish playback/header, readout and exposure-midpoint timestamp evidence."""
import csv
import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('r1',ROOT/'scripts/run-db3-r0-r1.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def fields(text):
    return dict(v.split('=',1) for v in text.split(';') if '=' in v)


def main():
    metadata=m.read(m.RUN/'timing_metadata.json')
    with (m.RUN/'timing_pairs.csv').open(newline='',encoding='utf-8') as f: pairs=list(csv.DictReader(f))
    sources={}
    for kind in ('Color','Depth'):
        topic=next(n for n in metadata if f'/{kind}_0/image/' in n)
        entries=metadata[topic]
        assert len(entries)==4764
        sources[kind]={int(e['timestamp_ns']):dict(fields(e['text']),metadata_message_id=e['message_id']) for e in entries}
        for key in ('Frame Counter','Frame Timestamp','Sensor Timestamp'):
            values=np.array([int(fields(e['text'])[key]) for e in entries])
            assert np.all(np.diff(values)>0), (kind,key)
        assert set(fields(e['text'])['timestamp_domain'] for e in entries)=={'Global Time'}
    with m.connect() as db:
        dep_stamps=dict(db.execute("select id,timestamp from messages where topic_id=(select id from topics where name like '%/Depth_0/image/data')"))
    rows=[];nearest_different=0
    depth_items=sorted(sources['Depth'].items(),key=lambda item:int(item[1]['Sensor Timestamp']))
    ds=np.array([int(d['Sensor Timestamp']) for _,d in depth_items])
    for pair in pairs:
        c=sources['Color'][int(pair['rgb_timestamp_ns'])]
        d=sources['Depth'][dep_stamps[int(pair['depth_message_id'])]]
        ct=int(c['Sensor Timestamp']); j=np.searchsorted(ds,ct);j=min(max(j,1),len(ds)-1)
        if abs(ds[j-1]-ct)<=abs(ds[j]-ct):j-=1
        nearest_different+=int(depth_items[j][0]!=dep_stamps[int(pair['depth_message_id'])])
        rows.append(dict(rgb_message_id=int(pair['rgb_message_id']),depth_message_id=int(pair['depth_message_id']),
            color_metadata_message_id=c['metadata_message_id'],depth_metadata_message_id=d['metadata_message_id'],
            signed_playback_depth_minus_rgb_ms=float(pair['signed_depth_minus_rgb_ms']),
            signed_global_metadata_depth_minus_rgb_ms=float(d['timestamp'])-float(c['timestamp']),
            signed_device_readout_depth_minus_rgb_ms=(int(d['Frame Timestamp'])-int(c['Frame Timestamp']))/1000,
            signed_exposure_midpoint_depth_minus_rgb_ms=(int(d['Sensor Timestamp'])-ct)/1000,
            depth_exposure_us=int(d['Actual Exposure']),
            depth_readout_minus_midpoint_us=int(d['Frame Timestamp'])-int(d['Sensor Timestamp']),
            color_readout_minus_midpoint_us=int(c['Frame Timestamp'])-ct,
            color_minus_depth_counter=int(c['Frame Counter'])-int(d['Frame Counter']),
            nearest_gyro_rad_s=float(pair['nearest_gyro_rad_s']),
            estimated_rotation_during_sensor_gap_px=428.4311968375*float(pair['nearest_gyro_rad_s'])*abs(int(d['Sensor Timestamp'])-ct)/1e6))
    with (m.RUN/'sensor_timing_pairs.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary={key:np.percentile([r[key] for r in rows],[0,50,95,100]).tolist() for key in rows[0] if key.endswith(('_ms','_us','_px'))}
    m.write(m.RUN/'sensor_timing_report.json',dict(pairs=len(rows),metadata_join='Exact image/metadata database timestamp; device frame counters and timestamps strictly monotonic',
        timestamp_domain='Global Time declared on every RGB/depth metadata message',
        percentiles_min_median_p95_max=summary,counter_differences=sorted(set(r['color_minus_depth_counter'] for r in rows)),
        nearest_sensor_midpoint_pair_differs_count=nearest_different,
        metadata_definitions_reference='https://github.com/realsenseai/librealsense/blob/v2.59.1/include/librealsense2/h/rs_frame.h#L15-L37',
        interpretation='Playback/header separation is not exposure-midpoint separation. SDK metadata defines Frame Timestamp as device readout/transmit start and Sensor Timestamp as exposure midpoint. Retain same observed pairs: every nearest midpoint association agrees with playback pairing.',
        decision='No guessed 7 ms shift or neighboring-frame reassociation. Keep per-pair sensor midpoint gap and gyro as uncertainty; future motion compensation requires accepted motion/extrinsics. High-motion samples excluded from local physical alignment acceptance.',
        limitations='Device metadata is evidence, not independent synchronization calibration. Color Actual Exposure is absent; source option value is not per-frame exposure. Readout/sensor fields and gyro clocks not independently cross-calibrated. Rotation-only displacement scale omits translation/rolling shutter.'))
    print(summary,flush=True)


if __name__=='__main__':main()
