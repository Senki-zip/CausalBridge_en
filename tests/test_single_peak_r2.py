"""Single-edge quality filtering precedes alpha/group fitting and is audited."""
from dataclasses import replace
import copy

import numpy as np
import pytest

from CausalBridge import saturating_hybrid as core, saturating_transfer as transfer
from test_saturating_hybrid import data


def inputs(threshold=.3):
    original = data()
    X = np.repeat(original.X[:, :1], 4, axis=1)
    return replace(original, input_keys=tuple((f'p{i}',0) for i in range(4)), X=X,
        source_bins=np.repeat(original.source_bins[:,:1],4,axis=1),
        route_supported=np.ones(X.shape,dtype=bool),selected_lag=np.zeros(X.shape,dtype=int),
        support_rows=np.arange(len(X)),
        provenance=dict(relation_type=transfer.GENE_TYPE, single_min_r2=threshold))


def install_quality(monkeypatch, values):
    real = core.fit_single
    iterator = iter(values)
    def fit(*args, **kwargs):
        single = real(*args, **kwargs)
        return replace(single, metrics=dict(single.metrics, training_r2=next(iterator)))
    monkeypatch.setattr(core,'fit_single',fit)


def test_filter_before_alpha_and_exact_threshold_roundtrip(monkeypatch):
    d = inputs()
    install_quality(monkeypatch,[.8,.3,.1,None])
    model = core.fit_target(d,(0.,1.),core.HybridConfig(min_group_rows=1000))
    assert model.input_keys == (('p0',0),('p1',0))
    np.testing.assert_allclose(model.weights,[.5,.5])
    excluded=model.provenance['excluded_single_relations']['records']
    assert [r['status'] for r in excluded] == ['low_single_r2']*2
    assert [r['training_r2'] for r in excluded] == [.1,None]
    assert model.provenance['excluded_single_relations']['original_input_count']==4
    assert core.TargetModel.from_bundle(model.to_bundle()).fingerprint==model.fingerprint
    bad=copy.deepcopy(model.to_bundle())
    bad['provenance']['single_min_r2']=.9
    bad['fingerprint']=core._digest({k:v for k,v in bad.items() if k!='fingerprint'})
    with pytest.raises(ValueError,match='retained single model'):
        core.TargetModel.from_bundle(bad)


def test_all_filtered_is_explicit_no_information(monkeypatch):
    install_quality(monkeypatch,[.1,.2,None,.29])
    with pytest.raises(core.TargetFitFailure) as exc:
        core.fit_target(inputs(),(0.,1.))
    diagnostic=exc.value.diagnostics
    assert diagnostic['status']=='no_information'
    assert len(diagnostic['excluded_single_relations']['records'])==4
    core.validate_excluded_relations(diagnostic,'target',())


def test_disabled_policy_keeps_low_r2_models(monkeypatch):
    install_quality(monkeypatch,[.1,.2,None,.29])
    model=core.fit_target(inputs(None),(0.,1.),core.HybridConfig(min_group_rows=1000))
    assert len(model.singles)==4


@pytest.mark.parametrize('value',[True,'0.3',-.1,1.1,float('nan')])
def test_bad_threshold(value):
    with pytest.raises(ValueError):
        transfer.single_peak_min_r2({'atac_to_rna':{'single_min_r2':value}})


def test_policy_enters_only_peak_gene_contract_and_context():
    cfg={'atac_to_rna':{'model_mode':'saturating_hybrid','single_min_r2':.3}}
    assert transfer.response_contract(cfg,transfer.GENE_TYPE)['single_min_r2']==.3
    assert 'single_min_r2' not in transfer.response_contract(cfg,transfer.TF_TYPE)
    assert transfer.response_context({},cfg,transfer.GENE_TYPE)['single_min_r2']==.3
    assert 'single_min_r2' not in transfer.response_context({},cfg,transfer.TF_TYPE)


@pytest.mark.parametrize('family',['nn','spline','saturation'])
def test_real_adapter_family_applies_policy_before_packing(family):
    from test_saturating_transfer import fixture_data
    r,a,pt,edges,tf,cfg=fixture_data()
    cfg['atac_to_rna'].update(single_min_r2=1.,response_model=family)
    cfg['atac_to_rna']['nn_response']={'epochs':3}
    cfg['atac_to_rna']['spline_response']={'max_nfev':100,'restarts':1}
    _,_,registry=transfer.fit_peak_gene(r,a,pt,edges,cfg)
    transfer.validate_registry(registry)
    assert registry['context']['single_min_r2']==1.
    assert registry['failures']
    for bundle in registry['targets'].values():
        model=core.TargetModel.from_bundle(bundle)
        assert all(s.metrics['training_r2'] >= 1. for s in model.singles)
    for diagnostic in registry['failures'].values():
        assert diagnostic['excluded_single_relations']['records']
        assert all(x['status']=='low_single_r2' for x in diagnostic['excluded_single_relations']['records'])
