"""CPU-only bounded spline math, strict codecs, independent-layer integration."""
import copy
import json
import warnings
from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from CausalBridge import spline_hybrid as spline, saturating_hybrid as core
from CausalBridge import saturating_transfer as transfer, run, io, binning_adapter
from test_saturating_transfer import fixture_data


@pytest.mark.parametrize("shape", ["monotone", "nonmonotone"])
def test_real_fit_bounds_C1_tails_and_objective(shape):
    x = np.linspace(.2, .9, 38)
    y = .1 + .7 * (x - x.min()) / np.ptp(x) if shape == "monotone" else .15 + .65*np.sin((x-.2)/.7*np.pi)**2
    cfg = spline.SplineConfig()
    m = spline.fit_single(x, y, (0, 1), cfg)
    assert m.metrics["training_r2"] > .95
    assert m.metrics["solver_converged"] is True and m.metrics["ridge"] == 0
    assert len(m.coefficients) == 6 and len(m.metrics["starts"]) == 3
    grid = np.r_[np.linspace(-100, 100, 1000), -1e200, 1e200]
    predicted = m.predict(grid)
    assert np.isfinite(predicted).all() and (predicted >= 0).all() and (predicted <= 1).all()
    for join in m.x_range:
        eps = 1e-7
        value = m.predict(np.array([join-eps, join, join+eps]))
        assert abs((value[1]-value[0])/eps-(value[2]-value[1])/eps) < 1e-3
    # Exact implemented residual definition, including no extra ridge.
    normalized_mse = np.mean((m.predict(x)-y)**2)
    penalty = np.sum((spline._penalty(m.knots, cfg.tail_scale)@m.coefficients)**2)
    assert m.metrics["best_objective"] == pytest.approx(normalized_mse+cfg.smoothness*penalty)
    assert m.predict([-1e200])[0] == pytest.approx(m.predict([-100])[0])
    assert m.predict([1e200])[0] == pytest.approx(m.predict([100])[0])
    if shape == "nonmonotone":
        assert m.predict([.55])[0] > max(m.predict([.2, .9]))


def test_green_design_matches_independent_basis_and_deduplicates_knots():
    x = np.linspace(.3, .9, 31)
    knots = spline._knots(x, 2)
    z = (x-x.min())/np.ptp(x)
    from scipy.interpolate import BSpline
    basis = BSpline(knots, np.eye(6), 3)
    np.testing.assert_array_equal(spline._design(x, (x.min(),x.max()), knots, .5), basis(z))
    q = np.array([-.4, .3, .9, 1.6])
    B = spline._design(q, (.3,.9), knots, .5)
    assert B[0] == pytest.approx(basis(0)+.5*basis.derivative()(0)*np.expm1(((q[0]-.3)/.6)/.5))
    assert B[-1] == pytest.approx(basis(1)-.5*basis.derivative()(1)*np.expm1(-(((q[-1]-.3)/.6)-1)/.5))
    repeated = np.r_[np.zeros(20), np.ones(20)]
    knots = spline._knots(repeated, 2)
    assert np.array_equal(knots, np.r_[np.zeros(4),np.ones(4)])


@pytest.mark.parametrize("change", [dict(smoothness=-1),dict(smoothness=True),dict(smoothness=np.inf),
    dict(tail_scale=0),dict(tail_scale=np.nan),dict(internal_knots=True),dict(internal_knots=-1),
    dict(max_nfev=0),dict(restarts=0),dict(seed=-1),dict(degree=4),dict(ridge=.1)])
def test_config_rejects_invalid_or_unknown(change):
    with pytest.raises(ValueError):
        spline.SplineConfig.from_dict(change)


@pytest.mark.parametrize("x,y,bounds,status", [
    (np.ones(20),np.linspace(.1,.7,20),(0,1),"constant_input"),
    (np.linspace(0,1,20),np.zeros(20),(0,1),"constant_output"),
    (np.linspace(0,1,20),np.ones(20),(1,1),"constant_output")])
def test_constant_noinformation_and_exact_boundary(x,y,bounds,status):
    m=spline.fit_single(x,y,bounds)
    assert m.metrics["status"]==status
    assert np.array_equal(m.predict([-10,0,10]),np.repeat(y.mean(),3))
    clone=spline.SplineSingleModel(**{k:v for k,v in m.to_dict().items() if k!='response_type'})
    np.testing.assert_array_equal(clone.predict([-10,0,10]),m.predict([-10,0,10]))


def target_data():
    x=np.linspace(0,1,36);X=np.column_stack([x,x*x])
    return core.TargetData('g',(('p',0),('q',0)),X,.1+.7*x,np.tile(np.arange(36)[:,None],(1,2)),
        np.arange(36),('b',)*36,'raw_atac','raw_atac',
        dict(response_family='spline',spline_response_config=spline.SplineConfig().to_dict()),cell_weights=np.eye(36))


def test_strict_target_roundtrip_frozen_alpha_groups_and_wrong_family():
    d=target_data();m=core.fit_target(d,(0,1));bundle=m.to_bundle()
    assert bundle['schema']==spline.SCHEMA
    assert all(a.get('reason')!='multivariate_spline_unsupported' for a in m.diagnostics['group_attempts'])
    clone=core.TargetModel.from_bundle(bundle)
    np.testing.assert_array_equal(core.predict_target(m,d.X).values,core.predict_target(clone,d.X).values)
    query=d.X.copy();query[:,0]=0
    old=np.column_stack([s.predict(d.X[:,i]) for i,s in enumerate(m.singles)])@m.weights
    new=np.column_stack([s.predict(query[:,i]) for i,s in enumerate(m.singles)])@m.weights
    np.testing.assert_allclose(core.predict_delta(m,d.X,query).values,new-old,atol=1e-15)
    assert clone.weights==m.weights
    for change in ['family','schema','coefficients','tail','component_type','config','units','groups']:
        bad=copy.deepcopy(bundle)
        if change=='family':bad['provenance']['response_family']='nn'
        elif change=='schema':bad['schema']='saturating_hybrid_target_v1'
        elif change=='coefficients':bad['singles'][0]['coefficients'].append(0)
        elif change=='tail':bad['singles'][0]['tail_scale']=.25
        elif change=='component_type':bad['singles'][0]['response_type']='nn'
        elif change=='config':bad['provenance']['spline_response_config']['smoothness']=.1
        elif change=='units':bad['singles'][0]['output_unit']='rna_log1p_cp10k'
        else:bad['groups']=[{}]
        bad['fingerprint']=core._digest({k:v for k,v in bad.items() if k!='fingerprint'})
        with pytest.raises(ValueError):core.TargetModel.from_bundle(bad)


def test_exhausted_spline_is_explicit_failure_even_with_bestfinite_policy():
    x=np.linspace(0,1,40);d=target_data()
    with pytest.raises(FloatingPointError,match='no successful finite candidate'):
        spline.fit_single(x,.1+.7*x,(0,1),spline.SplineConfig(max_nfev=1))
    provenance=dict(d.provenance,budget_exhausted_policy='best_finite',
        spline_response_config=spline.SplineConfig(max_nfev=1).to_dict())
    with pytest.raises(core.TargetFitFailure) as exc:core.fit_target(replace(d,provenance=provenance),(0,1))
    assert exc.value.diagnostics['status']=='no_information'
    assert all(r['status']=='failed' for r in exc.value.diagnostics['excluded_single_relations']['records'])


@pytest.mark.parametrize('gene_family,tf_family',[('spline','spline'),('spline','saturation'),('saturation','spline'),('spline','nn'),('nn','spline')])
def test_both_layer_adapters_reporting_checkpoint_and_units(tmp_path,gene_family,tf_family):
    r,a,pt,edges,tf,cfg=fixture_data()
    cfg['atac_to_rna'].update(response_model=gene_family,nn_response=dict(epochs=20))
    cfg['rna_to_atac']=dict(response_model=tf_family,nn_response=dict(epochs=20))
    cfg['output']=dict(dir=str(tmp_path/'with spaces/gene_ids'),checkpoint_dir=None,save_intermediate=False)
    table,_,gr=transfer.fit_peak_gene(r,a,pt,edges,cfg)
    tr=transfer.fit_tf_peak(r,a,pt,tf,cfg)
    for kind,reg,family,layer in [(transfer.GENE_TYPE,gr,gene_family,'peak_gene'),(transfer.TF_TYPE,tr,tf_family,'tf_peak')]:
        transfer.validate_registry(reg,expected_type=kind,expected_schema=transfer.registry_schema(kind,cfg))
        if family=='spline':
            assert reg['schema'].startswith('hybrid_spline_')
            assert reg['context']['spline_response_config']==spline.SplineConfig().to_dict()
            _,weights,curves=run._saturating_registry_reports(reg,layer)
            assert set(curves.response_family)=={'spline'}
            assert curves.b.isna().all() and curves.spline_coefficients_json.map(json.loads).map(len).ge(4).all()
            assert not weights.empty
        bad=copy.deepcopy(reg);bad['schema']='wrong';bad['fingerprint']=core._digest({k:v for k,v in bad.items() if k!='fingerprint'})
        with pytest.raises(ValueError):transfer.validate_registry(bad)
    run._validate_step5_checkpoint({'transfer_models':gr},'saturating_hybrid',config=cfg)
    other=copy.deepcopy(cfg);other['atac_to_rna']['response_model']='nn' if gene_family!='nn' else 'spline'
    with pytest.raises(RuntimeError,match='使用新的checkpoint_dir'):
        run._validate_step5_checkpoint({'transfer_models':gr},'saturating_hybrid',config=other)
    assert run._joint_runtime_config(cfg)['output']['dir']==cfg['output']['dir']
    assert run._checkpoint_dir(cfg)==Path(cfg['output']['dir'])/'checkpoints'
    # Real propagation/serialization without a production pipeline.
    from CausalBridge.saturating_perturbation import propagate_saturating
    output=propagate_saturating(['A'],r,a,pt,edges,table,tf,cfg,transfer_models=gr,tf_model_registry=tr,cluster_label='0')
    assert np.isfinite(output['saturating_endpoints'].query("status == 'ok'").delta_rna).all()
    reports=run._saturating_registry_reports(gr,'peak_gene')
    cfg['output']['export_details']=True
    io.save_results(dict(saturating_gene_metrics=reports[0],saturating_response_curves=reports[2]),cfg)
    assert (Path(cfg['output']['dir'])/'saturating_response_curves.csv').exists()


def test_inactive_settings_and_source_signature(monkeypatch):
    cfg=dict(atac_to_rna=dict(model_mode='saturating_hybrid',response_model='spline'),rna_to_atac=dict(response_model='spline'))
    contract=run._step5_model_config(cfg)
    cfg['atac_to_rna']['spline_response']=dict(smoothness=1e-5)
    assert run._step5_model_config(cfg)!=contract
    cfg['atac_to_rna']['response_model']='nn'
    before=run._step5_model_config(cfg);cfg['atac_to_rna']['spline_response']=dict(unknown=True)
    assert run._step5_model_config(cfg)==before
    sig=transfer._source_signature(transfer.GENE_TYPE,response_family='spline')
    assert sig['core_schema']==spline.SCHEMA
    assert 'CausalBridge.spline_hybrid._design' in sig['functions']
    assert 'torch' not in sig['versions']


def test_spawn_matches_serial_both_spline_layers():
    r,a,pt,edges,tf,cfg=fixture_data()
    cfg['atac_to_rna']['response_model']='spline';cfg['rna_to_atac']=dict(response_model='spline')
    serial=transfer.fit_peak_gene(r,a,pt,edges,cfg)[2]
    serial_tf=transfer.fit_tf_peak(r,a,pt,tf,cfg)
    cfg['atac_to_rna']['saturating_hybrid']['workers']=2
    assert transfer.fit_peak_gene(r,a,pt,edges,cfg)[2]==serial
    assert transfer.fit_tf_peak(r,a,pt,tf,cfg)==serial_tf


@pytest.mark.parametrize('smooth',['hard','gaussian'])
def test_continuous_runtime_checkpoint_recovery_and_family_rejection(tmp_path,monkeypatch,smooth):
    from test_atac_normalization import continuous_fixture
    from CausalBridge import kinetics, perturbation
    from test_nn_hybrid import assert_pipeline_registry_contracts
    r,a,pt,edges,cfg=continuous_fixture(tmp_path,smooth)
    cfg['atac_to_rna']['response_model']='spline';cfg['rna_to_atac']['response_model']='spline'
    monkeypatch.setattr(kinetics,'_scan_peaks_with_jaspar',lambda *a,**k:{'chr1:10-20':['TF']})
    table,gr=kinetics.fit_atac_to_rna(r,a,pt,edges,cfg)
    tr={}
    screen=kinetics.fit_rna_to_atac(r,a,pt,edges,cfg,tf_model_registry=tr,
        motif_resources={'motif_db':{'synthetic':{'tf_name':'TF'}},'tf_database':frozenset({'TF'}),'resource_cache_dir':str(tmp_path)})
    assert_pipeline_registry_contracts(cfg,gr,tr)
    assert gr['schema']=='hybrid_spline_peak_gene_registry_v2'
    assert tr['schema']=='hybrid_spline_tf_peak_local_registry_v3'
    frozen=copy.deepcopy((gr,tr))
    output=perturbation.propagate_perturbation(['TF'],r,a,pt,edges,table,screen,cfg,
        transfer_models=gr,tf_model_registry=tr)
    assert output['effect_schema_version']=='bin_saturating_effect_v1'
    assert not output['perturbation_results'].empty
    zero=copy.deepcopy(cfg);zero['perturbation']['ko_strength']=0
    noop=perturbation.propagate_perturbation(['TF'],r,a,pt,edges,table,screen,zero,
        transfer_models=gr,tf_model_registry=tr)
    assert all(abs(x)==0 for x in noop['delta_rna'].values())
    assert (gr,tr)==frozen


def test_spline_helper_change_invalidates_signature(monkeypatch):
    before=transfer._source_signature(transfer.GENE_TYPE,response_family='spline')
    monkeypatch.setattr(spline,'_design',alternative_design)
    after=transfer._source_signature(transfer.GENE_TYPE,response_family='spline')
    assert before!=after


def alternative_design(x,x_range,knots,tail_scale):
    raise AssertionError('signature only, never evaluated')


@pytest.mark.parametrize('tiny', [8.4e-131, 3.27e-25, 2.94e-16, 1e-12])
def test_only_numerically_degenerate_knots_are_repaired(tiny):
    x = np.r_[0., np.full(20, tiny), np.linspace(.01, 1, 21)]
    y = .15 + .65 * np.sin(np.pi*x)**2
    cfg = spline.SplineConfig()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        model = spline.fit_single(x, y, (0, 1), cfg)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    repair = model.metrics['knot_repair']
    np.testing.assert_array_equal(repair['original_knots'], spline._knots(x, 2))
    assert len(model.knots) < len(repair['original_knots'])
    assert model.config == cfg.to_dict()
    assert np.isfinite(model.metrics['best_objective'])
    assert model.metrics['best_objective'] < .1
    clone = spline.SplineSingleModel(**{k:v for k,v in model.to_dict().items() if k != 'response_type'})
    query = np.linspace(-1, 2, 200)
    np.testing.assert_array_equal(model.predict(query), clone.predict(query))
    assert ((model.predict(query) >= 0) & (model.predict(query) <= 1)).all()
    for join in model.x_range:
        h = 1e-7
        v = model.predict([join-h, join, join+h])
        assert abs((v[2]-2*v[1]+v[0])/h) < 1e-3
    bad = model.to_dict(); bad.pop('response_type')
    bad['metrics']['knot_repair']['repaired_knots'][4] = .2
    with pytest.raises(ValueError, match='repair'):
        spline.SplineSingleModel(**bad)


def test_healthy_matrices_and_budget_failure_do_not_repair(monkeypatch):
    x = np.r_[np.zeros(10), np.linspace(.02, 1, 32)]
    cfg = spline.SplineConfig(max_nfev=1)
    knots = spline._knots(x, cfg.internal_knots)
    B, P, used, repair = spline._single_matrices(x, (x.min(), x.max()), knots, cfg)
    assert repair is None
    np.testing.assert_array_equal(used, knots)
    np.testing.assert_array_equal(B, spline._design(x, (x.min(), x.max()), knots, cfg.tail_scale))
    np.testing.assert_array_equal(P, np.sqrt(cfg.smoothness)*spline._penalty(knots, cfg.tail_scale))
    calls = []
    original = spline.least_squares
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(spline, 'least_squares', counted)
    with pytest.raises(spline.SplineFitFailure):
        spline.fit_single(x, .1+.7*x, (0, 1), cfg)
    assert len(calls) == cfg.restarts


@pytest.mark.parametrize('bad', ['objective', 'jacobian', 'warning'])
def test_numerically_invalid_success_is_not_accepted(monkeypatch, bad):
    def invalid_success(fun, start, jac, **kwargs):
        if bad == 'warning':
            warnings.warn('overflow encountered in solver', RuntimeWarning)
        return SimpleNamespace(success=True, status=3, nfev=1, x=start,
            fun=np.full(10, 1e200) if bad == 'objective' else fun(start),
            jac=np.full((10, len(start)), np.nan) if bad == 'jacobian' else jac(start))
    monkeypatch.setattr(spline, 'least_squares', invalid_success)
    with pytest.raises(spline.SplineFitFailure) as exc:
        spline.fit_single(np.linspace(0, 1, 30), np.linspace(.1, .9, 30), (0, 1))
    assert all(not s['finite'] for s in exc.value.starts)
    if bad == 'warning':
        assert all(s['numeric_warnings'] for s in exc.value.starts)


def test_numeric_helpers_are_in_source_signature():
    signature = transfer._source_signature(transfer.TF_TYPE, response_family='spline')
    for name in ('_single_matrices', '_finite_objective', '_numerical_knot_failure'):
        assert 'CausalBridge.spline_hybrid.' + name in signature['functions']


def test_spline_subset_warm_recovery_and_config_change_rejects_without_overwrite(tmp_path,monkeypatch):
    from test_atac_normalization import continuous_fixture
    from CausalBridge import kinetics
    r,a,pt,edges,cfg=continuous_fixture(tmp_path)
    cfg['pseudotime'].update(n_neighbors=10,method='dpt')
    cfg['granger']={'reference_bins':24,'min_composite_score':.3}
    cfg['atac_to_rna']['response_model']='spline';cfg['rna_to_atac']['response_model']='spline'
    resources={'motif_db':{'synthetic':{'tf_name':'TF','pwm':np.full((4,6),.25)}},
        'tf_database':frozenset({'TF'}),'resource_cache_dir':str(tmp_path/'cache')}
    monkeypatch.setattr(run,'infer_pseudotime',lambda *a,**k:pt.copy(deep=True))
    monkeypatch.setattr(run,'granger_test',lambda *a,**k:edges.assign(is_causal=True))
    monkeypatch.setattr(run,'_prepare_adaptive_motif_resources',lambda *a,**k:resources)
    monkeypatch.setattr(kinetics,'_scan_peaks_with_jaspar',lambda *a,**k:{'chr1:10-20':['TF']})
    first=run._run_modeling_for_subset(r,a,['TF'],{'TF':'TF'},cfg,suffix='cluster_0')
    meta=first['saturating_context_metadata']
    assert meta['gene_response_model']==meta['tf_response_model']=='spline'
    assert meta['response_settings']['peak_gene']['spline_response']['smoothness']==1e-6
    directory=run._checkpoint_dir(cfg,'cluster_0');before={p.name:p.read_bytes() for p in directory.glob('*.pkl')}
    monkeypatch.setattr(run,'fit_atac_to_rna',lambda *a,**k:pytest.fail('warm spline refit'))
    warm=run._run_modeling_for_subset(r,a,['TF'],{'TF':'TF'},cfg,suffix='cluster_0')
    pd.testing.assert_frame_equal(warm['perturbation_results'],first['perturbation_results'])
    changed=copy.deepcopy(cfg);changed['atac_to_rna']['spline_response']={'smoothness':1e-5}
    with pytest.raises(RuntimeError,match='incompatible.*使用新的checkpoint_dir'):
        run._run_modeling_for_subset(r,a,['TF'],{'TF':'TF'},changed,suffix='cluster_0')
    assert before=={p.name:p.read_bytes() for p in directory.glob('*.pkl')}


def test_config_defaults_and_supplied_config_select_spline_without_path_change(tmp_path):
    import yaml
    supplied=yaml.safe_load((Path(__file__).resolve().parents[1]/'config.yaml').read_text())
    assert supplied['atac_to_rna']['response_model'] in ('spline','nn','saturation')
    assert supplied['rna_to_atac']['response_model'] in ('spline','nn','saturation')
    for layer in ['atac_to_rna','rna_to_atac']:
        assert spline.SplineConfig.from_dict(supplied[layer]['spline_response'])==spline.SplineConfig()
    for name in ['rna','atac']:(tmp_path/name).touch()
    path=tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(dict(input=dict(rna_h5ad=str(tmp_path/'rna'),atac_h5ad=str(tmp_path/'atac')),
        atac_to_rna=dict(model_mode='saturating_hybrid',response_model='spline'),
        rna_to_atac=dict(response_model='spline'),output=dict(dir='explicit/gene_ids/',checkpoint_dir=None))))
    loaded=io.load_config(path)
    assert loaded['atac_to_rna']['spline_response']==spline.SplineConfig().to_dict()
    assert run._joint_runtime_config(loaded)['output']['dir']=='explicit/gene_ids/'
