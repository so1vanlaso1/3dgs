"""Fail-closed source-ID comparisons for the frozen DB3 acceptance rules."""
import math


def appearance_noninferior(candidate, reference, rules=None):
    rules = rules or {}
    limits = {'psnr': rules.get('global_psnr_max_drop_db', .2),
              'ssim': rules.get('global_ssim_max_drop', .005),
              'lpips': rules.get('global_lpips_max_increase', .01)}
    if not all(math.isfinite(x.get(k, float('nan'))) for x in (candidate, reference) for k in limits):
        return False
    return all(candidate[k] >= reference[k] - limits[k] if k != 'lpips'
               else candidate[k] <= reference[k] + limits[k] for k in limits)


def region_comparison(candidate, reference, label, rules=None, min_samples=100):
    rules = rules or {}
    required = ('rgb_psnr_db', 'observed_alpha_coverage')
    for rec in (candidate, reference):
        if not math.isfinite(rec.get('source_samples', float('nan'))) or rec.get('source_samples', 0) < min_samples:
            return dict(pass_gate=False, reason='insufficient sensor support')
        depth = rec.get('depth_absolute_error_m_median_p90', [])
        if len(depth) != 2 or not all(math.isfinite(v) for v in depth) or not all(
                math.isfinite(rec.get(k, float('nan'))) for k in required):
            return dict(pass_gate=False, reason='missing or nonfinite measurements')
        if not 0 <= rec['observed_alpha_coverage'] <= 1 or min(depth)<0:
            return dict(pass_gate=False, reason='invalid measurement range')
    change = candidate['depth_absolute_error_m_median_p90'][0] - reference['depth_absolute_error_m_median_p90'][0]
    gain = candidate['rgb_psnr_db'] - reference['rgb_psnr_db']
    coverage = candidate['observed_alpha_coverage'] - reference['observed_alpha_coverage']
    depth_limit = rules.get('each_target_and_control_median_depth_max_worsening_m', .01)
    if 'target' in label and 'target_depth_gain_m' in rules:
        depth_limit = -rules['target_depth_gain_m']
    gates = dict(depth=change <= depth_limit,
                 coverage=coverage >= -rules.get('alpha_coverage_max_drop', .02),
                 target_appearance=('target' not in label or gain >= rules.get('each_target_ROI_RGB_PSNR_gain_db', -float('inf'))))
    return dict(depth_median_change_m=change, ROI_psnr_gain_db=gain, coverage_change=coverage,
                gates=gates, pass_gate=all(gates.values()))


def compare_views(candidate, reference, expected_views, expected_regions, rules=None):
    """Never drop unsupported/missing views or match them by incidental list order."""
    expected_views = list(expected_views)
    if not expected_views or len(set(expected_views)) != len(expected_views) or not expected_regions:
        raise ValueError('Expected nonempty, unique frozen views and regions')
    indexed = []
    for result in (candidate, reference):
        views = result['views']
        names = [v['name'] for v in views]
        if len(set(names)) != len(names) or set(names) != set(expected_views):
            raise ValueError('Evaluation identities differ from frozen split')
        indexed.append({v['name']: v for v in views})
    rows = []
    for name in expected_views:
        for key in ('source_id','original_source'):
            a,b=(index[name].get(key) for index in indexed)
            if a!=b:
                raise ValueError('Source metadata differs for matched evaluation view')
        for label in expected_regions:
            a, b = (index[name]['regions'].get(label, {}) for index in indexed)
            rows.append(dict(view=name, region=label, **region_comparison(a, b, label, rules)))
    appearance = appearance_noninferior(candidate['mean_appearance'], reference['mean_appearance'], rules)
    return dict(rows=rows, appearance_noninferior=appearance,
                quantitative_pass=all(r['pass_gate'] for r in rows) and appearance)


def paired_decisions(results, expected_views, expected_regions, matched_compute=True):
    rules={'target_depth_gain_m': .01}
    comparisons={a:compare_views(results['confidence'],results[a],expected_views,expected_regions,rules)
                 for a in ('off','current')}
    targets=[];controls=[]
    for reference,comparison in comparisons.items():
        for row in comparison['rows']:
            item=dict(view=row['view'],region=row['region'],reference=reference,
                      median_depth_error_change_m=row.get('depth_median_change_m'),pass_gate=row['pass_gate'])
            (targets if 'target' in row['region'] else controls).append(item)
    all_rows=[row for c in comparisons.values() for row in c['rows']]
    supported=all('gates' in row for row in all_rows)
    gate=dict(equal_initialization_and_replay=matched_compute,completed_all_7000_updates=matched_compute,
        appearance_noninferior=all(c['appearance_noninferior'] for c in comparisons.values()),
        meaningful_target_gain_every_checked_view=bool(targets) and all(r['median_depth_error_change_m'] is not None and r['median_depth_error_change_m']<=-.01 for r in targets),
        controls_no_material_worsening=bool(controls) and all(r['median_depth_error_change_m'] is not None and r['median_depth_error_change_m']<=.01 for r in controls),
        coverage_no_material_worsening=supported and all(r['gates']['coverage'] for r in all_rows),
        all_frozen_regions_supported=supported,visual_review_completed=False)
    pipeline={}
    for arm in ('off','current','confidence'):
        compared=compare_views(results[arm],results['baseline'],expected_views,expected_regions,rules)
        rows=compared['rows'];target=[r for r in rows if 'target' in r['region']];control=[r for r in rows if 'target' not in r['region']]
        supported=all('gates' in r for r in rows)
        gates=dict(appearance_noninferior=compared['appearance_noninferior'],
            target_gain=bool(target) and all(r.get('gates',{}).get('depth',False) for r in target),
            controls_not_worse=bool(control) and all(r.get('gates',{}).get('depth',False) for r in control),
            coverage=supported and all(r['gates']['coverage'] for r in rows),all_frozen_regions_supported=supported)
        changes=lambda group:[dict(view=r['view'],region=r['region'],change_m=r.get('depth_median_change_m'),pass_gate=r['pass_gate']) for r in group]
        pipeline[arm]=dict(gates=gates,quantitative_pass=all(gates.values()),target_changes=changes(target),control_changes=changes(control),complete_region_comparison=compared)
    return dict(gates=gate,target_changes=targets,control_changes=controls,pipeline_against_baseline=pipeline,
        quantitative_promotion_pass=all(v for k,v in gate.items() if k!='visual_review_completed') if matched_compute else None)
