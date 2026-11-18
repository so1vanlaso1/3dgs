"""Regression controls against accidental DB3 false promotion."""
import copy
import pytest
from scripts.db3_acceptance import compare_views, region_comparison, paired_decisions


def measurement(depth=.1, samples=100, psnr=25.):
    return dict(source_samples=samples,depth_absolute_error_m_median_p90=[depth,depth*2],
                rgb_psnr_db=psnr,observed_alpha_coverage=.95)


def result():
    return dict(views=[dict(name=n,regions={'target':measurement(), 'control':measurement()}) for n in ('a','b')],
                mean_appearance=dict(psnr=25.,ssim=.8,lpips=.2))


def test_unsupported_region_cannot_disappear_from_promotion():
    reference=result();candidate=copy.deepcopy(reference)
    candidate['views'][1]['regions']['target']=measurement(samples=99)
    check=compare_views(candidate,reference,['a','b'],['target','control'])
    assert len(check['rows'])==4 and not check['quantitative_pass']


def test_missing_region_fails_and_reordered_views_match_by_identity():
    a=result();b=result();b['views'].reverse()
    assert compare_views(a,b,['a','b'],['target','control'])['quantitative_pass']
    del b['views'][0]['regions']['control']
    assert not compare_views(a,b,['a','b'],['target','control'])['quantitative_pass']


@pytest.mark.parametrize('names',[['a','a'],['a'],['a','c']])
def test_wrong_or_duplicate_source_roster_rejected(names):
    a=result();a['views']=[dict(name=n,regions={}) for n in names]
    with pytest.raises(ValueError):compare_views(a,result(),['a','b'],['target','control'])


def test_nan_measurements_and_empty_expected_roster_fail():
    assert not region_comparison(measurement(depth=float('nan')),measurement(),'target')['pass_gate']
    with pytest.raises(ValueError):compare_views(result(),result(),[],['target'])


def test_paired_gate_requires_every_region_and_preserves_unequal_compute():
    results={arm:result() for arm in ('baseline','off','current','confidence')}
    for view in results['confidence']['views']:view['regions']['target']['depth_absolute_error_m_median_p90']=[.08,.16]
    args=(results,['a','b'],['target','control'])
    assert paired_decisions(*args)['quantitative_promotion_pass']
    assert paired_decisions(*args,matched_compute=False)['quantitative_promotion_pass'] is None
    results['confidence']['views'][1]['regions']['control']['source_samples']=99
    check=paired_decisions(*args)
    assert not check['quantitative_promotion_pass']
    assert not check['pipeline_against_baseline']['confidence']['quantitative_pass']


def test_source_identity_mismatch_and_nonfinite_appearance_rejected():
    a=result();b=result();a['views'][0]['source_id']=12;b['views'][0]['source_id']=13
    with pytest.raises(ValueError):compare_views(a,b,['a','b'],['target'])
    a=result();b=result();a['mean_appearance']['psnr']=float('nan')
    assert not compare_views(a,b,['a','b'],['target','control'])['quantitative_pass']


def test_invalid_sample_count_and_coverage_rejected():
    assert not region_comparison(measurement(samples=float('nan')),measurement(),'control')['pass_gate']
    a=measurement();a['observed_alpha_coverage']=1.5
    assert not region_comparison(a,measurement(),'control')['pass_gate']


def test_frozen_thresholds_are_used_and_original_depth_gain_remains_separate():
    a=measurement(depth=.105,psnr=27.);b=measurement()
    assert region_comparison(a,b,'target',{'each_target_ROI_RGB_PSNR_gain_db':1.})['pass_gate']
    assert not region_comparison(a,b,'target',{'target_depth_gain_m':.01})['pass_gate']
    assert not region_comparison(a,b,'target',{'each_target_and_control_median_depth_max_worsening_m':.001})['pass_gate']
    assert not region_comparison(a,b,'target',{'each_target_ROI_RGB_PSNR_gain_db':3.})['pass_gate']
