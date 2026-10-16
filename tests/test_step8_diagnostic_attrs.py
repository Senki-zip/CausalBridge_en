"""Diagnostic metadata must not be recopied per numerical TF group."""
import copy

import numpy as np
import pandas as pd
import pytest

from CausalBridge import perturbation as p


class CountedDiagnostic:
    def __init__(self, calls):
        self.calls = calls
        self.mutable = {"nested": ["unchanged", {"origin": "checkpoint"}]}

    def __deepcopy__(self, memo):
        self.calls.append(1)
        result = type(self)(self.calls)
        result.mutable = copy.deepcopy(self.mutable, memo)
        memo[id(self)] = result
        return result


def attach_diagnostics(frame):
    calls = []
    frame.attrs = {
        "tf_mapping_diagnostics": [{"tf": f"TF{i}", "candidates": ["a", "b"],
                                    "details": {"status": "kept", "score": .5}}
                                   for i in range(4000)],
        "provenance": {"schema": "test", "export": {"retain": True, "ids": [1, 2]}},
        "probe": CountedDiagnostic(calls),
    }
    attrs = frame.attrs
    diagnostic = attrs["tf_mapping_diagnostics"]
    saved = copy.deepcopy({k: v for k, v in attrs.items() if k != "probe"})
    probe, mutable = attrs["probe"], copy.deepcopy(attrs["probe"].mutable)

    def assert_unchanged():
        assert frame.attrs is attrs
        assert frame.attrs["tf_mapping_diagnostics"] is diagnostic
        assert frame.attrs["probe"] is probe
        assert probe.mutable == mutable
        assert {k: v for k, v in frame.attrs.items() if k != "probe"} == saved
    return calls, assert_unchanged


def assert_result_equal(expected, actual):
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)
    elif isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(expected, actual, check_exact=True)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(expected, actual)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert_result_equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            assert_result_equal(left, right)
    elif isinstance(expected, float) and np.isnan(expected):
        assert np.isnan(actual)
    else:
        assert expected == actual




@pytest.mark.parametrize("depth", [3, "unlimited"])
def test_hybrid_simulation_preserves_caller_attrs_and_all_outputs(depth):
    from test_saturating_transfer import fixture_data
    from CausalBridge import saturating_transfer as t
    r,a,pt,edges,tf,cfg=fixture_data()
    cfg['perturbation']['subnetwork_depth']=depth
    table,_,genes=t.fit_peak_gene(r,a,pt,edges,cfg);tfs=t.fit_tf_peak(r,a,pt,tf,cfg)
    kwargs=dict(transfer_models=genes,tf_model_registry=tfs)
    plain=p.propagate_perturbation(['A'],r,a,pt,edges,table,tf,cfg,**kwargs)
    calls,unchanged=attach_diagnostics(tf)
    rich=p.propagate_perturbation(['A'],r,a,pt,edges,table,tf,cfg,**kwargs)
    pd.testing.assert_frame_equal(plain['perturbation_results'],rich['perturbation_results'])
    np.testing.assert_array_equal(plain['rna_state_history'],rich['rna_state_history'])
    # Hybrid topology validation makes a bounded four copies before simulation;
    # unlike the deleted joint path, its fixed setup overhead is not one copy.
    assert len(calls)<=4
    unchanged()
