"""Do not promote perfect connectivity with missing independent geometry evidence."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('r4',Path(__file__).resolve().parents[1]/'scripts/db3_r4_route.py')
r4=importlib.util.module_from_spec(spec);spec.loader.exec_module(r4)


def test_connected_inaccurate_graph_is_rejected():
    edges=[dict(boundary=i,accepted=True) for i in range(4)]
    for structures,global_ok in [(False,True),(True,False),(False,False)]:
        decision=r4.expansion_decision(edges,range(4),structures_pass=structures,global_constraints_pass=global_ok)
        assert not decision['whole_scene_training_authorized_by_evidence']


def test_missing_duplicate_and_unsupported_edges_fail_closed():
    for edges in [[],[dict(boundary=0,accepted=True)],
                  [dict(boundary=0,accepted=True)]*2,
                  [dict(boundary=0,accepted=True),dict(boundary=1,accepted=False)],
                  [dict(boundary=0,accepted=True),dict(boundary=1)]]:
        assert not r4.expansion_decision(edges,[0,1],structures_pass=True,global_constraints_pass=True)['whole_scene_training_authorized_by_evidence']


def test_reservations_and_historical_evaluation_never_fit():
    reservations={1:dict(role='fit_development'),2:dict(role='geometry_check'),3:dict(role='final_prospective'),4:dict(role='buffer')}
    assert r4.eligible(dict(message_id=1),reservations,set())
    assert not r4.eligible(dict(message_id=1),reservations,{1})
    for mid in (2,3,4):assert not r4.eligible(dict(rgb_message_id=mid),reservations,set())


def test_pass_only_authorizes_experiment_not_scientific_acceptance():
    decision=r4.expansion_decision([dict(boundary=0,accepted=True)],[0],structures_pass=True,global_constraints_pass=True)
    assert decision['whole_scene_training_authorized_by_evidence']
    assert not decision['whole_scene_scientifically_accepted']
