"""Reference means stay bounded even when a floating-point reduction rounds up."""
import copy

import numpy as np
import pytest

from CausalBridge import run, saturating_hybrid as core


@pytest.mark.parametrize('family', ['nn', 'saturation', 'spline'])
@pytest.mark.parametrize('n,upper', [(37, .3), (50, .7), (12, .1)])
def test_constant_upper_boundary_local_reference_roundtrip(family, n, upper):
    rows = np.arange(n)
    provenance = dict(relation_type='saturating_hybrid_tf_peak', local_lag_max=0,
                      response_family=family)
    if family == 'nn':
        provenance['nn_response_config'] = dict(epochs=3)
    if family == 'spline':
        provenance['spline_response_config'] = dict(max_nfev=100, restarts=1)
    data = core.TargetData('P', (('TF', 'local'),),
        np.linspace(.1, 1., n)[:, None], np.full(n, upper), rows[:, None], rows,
        ('b',)*n, 'rna_log1p_cp10k', 'normalized_atac_log1p', provenance,
        support_rows=rows, route_supported=np.ones((n, 1), dtype=bool))
    assert data.y.mean() > upper  # Exercise the reduction roundoff, not training.
    model = core.fit_target(data, (0., upper))
    assert 0. <= model.diagnostics['single_WT_reference_means'][0] <= upper
    loaded = core.TargetModel.from_bundle(model.to_bundle())
    assert loaded.fingerprint == model.fingerprint
    assert np.array_equal(core.predict_delta(loaded, data.X, data.X).values, np.zeros(n))
    # Invalid serialized references must still be rejected, including one ULP.
    bundle = copy.deepcopy(model.to_bundle())
    bundle['diagnostics']['single_WT_reference_means'] = [np.nextafter(upper, np.inf)]
    bundle['fingerprint'] = core._digest({k:v for k,v in bundle.items() if k != 'fingerprint'})
    with pytest.raises(ValueError, match='invalid local single WT references'):
        core.TargetModel.from_bundle(bundle)


def test_new_single_quality_rule_does_not_reuse_reference_only_alias(monkeypatch):
    key = 'CausalBridge.saturating_hybrid.fit_target'
    before = run._numerical_source_hash(core.fit_target)
    assert before != run._WT_REFERENCE_SOURCE_COMPATIBILITY[key][1]
    original = run.inspect.getsource
    source = original(core.fit_target)
    source = source.replace('float(np.clip(s.predict', 'float(np.clip(2*s.predict')
    monkeypatch.setattr(run.inspect, 'getsource', lambda obj: source if obj is core.fit_target else original(obj))
    assert run._numerical_source_hash(core.fit_target) != before


@pytest.mark.parametrize('family', ['nn', 'spline', 'saturation'])
@pytest.mark.parametrize('continuous', [True, False])
def test_nested_response_contract_tracks_new_rule_and_future_changes(family, continuous, monkeypatch):
    from CausalBridge import saturating_transfer as transfer
    signature = transfer._source_signature(transfer.TF_TYPE, continuous, family)
    key = 'CausalBridge.saturating_hybrid.fit_target'
    assert signature['functions'][key] == run._numerical_source_hash(core.fit_target)
    assert signature['functions'][key] != run._WT_REFERENCE_SOURCE_COMPATIBILITY[key][1]
    original = run.inspect.getsource
    changed = original(core.fit_target).replace('float(np.clip(s.predict', 'float(np.clip(2*s.predict')
    monkeypatch.setattr(run.inspect, 'getsource', lambda obj: changed if obj is core.fit_target else original(obj))
    assert transfer._source_signature(transfer.TF_TYPE, continuous, family)['functions'][key] != signature['functions'][key]
