"""Synthetic-only numerical gate; no production dispatch or real-data training."""
import copy
import json
import time
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from CausalBridge.saturating_hybrid import (
    HybridConfig, TargetData, CellSupport, SingleModel, TargetModel,
    fit_single, fit_target, predict_target, predict_delta, rna_output_bounds,
    validate_cell_support, _digest,
)


def data(X=None, y=None, weights=None, rows=None, branches=None, keys=None):
    rng = np.random.default_rng(23)
    if X is None:
        x = rng.uniform(.1, .9, 48)
        X = np.column_stack([x, x])
    else:
        X = np.asarray(X)
    n, p = X.shape
    keys = tuple((f"p{i}", 0) for i in range(p)) if keys is None else keys
    rows = np.arange(n) if rows is None else np.asarray(rows)
    y = .5 + .3*(X[:,0]-.5)/(.2+abs(X[:,0]-.5)) if y is None else np.asarray(y)
    return TargetData("target", keys, X, y,
        rows[:,None]-np.array([k[1] for k in keys]), rows,
        tuple(["b"]*n) if branches is None else tuple(branches),
        "raw_atac", "raw_atac", {"source":"synthetic","seed":23}, cell_weights=weights)


def disjoint_weights(n, cells_per_row=4):
    w = np.zeros((n, n*cells_per_row))
    for row in range(n):
        w[row,row*cells_per_row:(row+1)*cells_per_row] = 1/cells_per_row
    return w


@pytest.fixture(scope="module")
def baseline():
    d = data()
    return d, fit_target(d, (0.,1.))


@pytest.fixture(scope="module")
def grouped():
    rng = np.random.default_rng(37)
    X = rng.uniform(.1,.9,(64,3))
    u = (X[:,:2]-X[:,:2].mean(axis=0))/X[:,:2].std(axis=0)
    X[:,2] = .5  # Zero-score outside input still must be covered if changed.
    index = 2*u[:,0]*u[:,1]+.8*u[:,0]+.5*u[:,1]
    y = .5+.45*index/(1+np.abs(index))
    d = data(X,y,disjoint_weights(64))
    # Explicit experimental test setting: weak ridge, not an invented gain metric.
    cfg = HybridConfig(ridge=.001, min_query_cell_ess=1.)
    m = fit_target(d,(0.,1.),cfg)
    assert m.groups, m.diagnostics
    return d,m,CellSupport(d.cell_weights)


@pytest.mark.parametrize("settings", [
    {"workers":0},{"workers":True},{"workers":1.5},{"max_nfev":3001},{"max_nfev":1.5},
    {"max_nfev":True},{"det_starts":4},{"max_groups":3},{"polynomial_degree":1},
    {"ridge":float("nan")},{"ridge":float("inf")},{"ridge":True},{"ridge":-.1},
    {"group_sizes":(2,2)},{"group_sizes":(4,)},{"group_sizes":()},
    {"min_group_cell_ess":0},{"support_radius_quantile":0},
    {"support_radius_quantile":1.1},{"max_geometry_condition":.5},
    {"min_penalized_gain":1},{"restart_prediction_spread_max":2},
])
def test_config_rejects_unsupported_or_nonfinite(settings):
    with pytest.raises(ValueError):HybridConfig(**settings)


def test_config_json_frozen_known_keys():
    cfg = HybridConfig()
    assert HybridConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    with pytest.raises(FrozenInstanceError):cfg.ridge=.2
    with pytest.raises(ValueError):HybridConfig.from_dict({"unknown":1})


@pytest.mark.parametrize("amplitude", [.3,-.3])
def test_single_increasing_decreasing_bounded_extreme_and_ko(amplitude):
    x = np.linspace(.1,1.,48)
    y = .5+amplitude*(x-.4)/(.2+np.abs(x-.4))
    model = fit_single(x,y,(0.,1.))
    query = np.array([-1e300,-100.,0.,.2,.5,1.,100.,1e300])
    result = model.predict(query)
    assert np.isfinite(result).all() and result.min()>=0 and result.max()<=1
    assert amplitude*model.a>0
    assert model.metrics["training_rmse"] < .005
    assert model.metrics["n_rows"] == 48
    assert model.metrics["evaluation"] == "all_WT_training_rows_not_validation"
    np.testing.assert_allclose(model.predict(x),y,atol=.005)


def test_single_constant_input_output_and_degenerate_bounds():
    x=np.linspace(0.,1.,24)
    for xx,y,bounds in [(x,np.full(24,.4),(0.,1.)),(np.ones(24),x,(0.,1.)),(x,np.zeros(24),(0.,0.))]:
        m=fit_single(xx,y,bounds)
        assert m.a==0 and m.s>0 and m.metrics["identifiable"] is False
        assert np.isfinite(m.predict(np.array([0.,1e300]))).all()
    with pytest.raises(ValueError):fit_single(x,x,(0.,.5))
    with pytest.raises(ValueError):fit_single(x[:3],x[:3],(0.,1.))


def test_budget_exhaustion_is_not_labeled_success():
    x=np.linspace(.1,1,24);y=.5+.3*(x-.2)/(.03+abs(x-.2))
    m=fit_single(x,y,(0.,1.),HybridConfig(max_nfev=1))
    assert m.metrics["status"]=="budget_exhausted" and m.metrics["termination_status"]==0


def test_rna_bounds_units_and_negative_domain():
    assert rna_output_bounds(np.array([.2,.6]))==pytest.approx((0.,.7))
    with pytest.raises(ValueError):rna_output_bounds(np.array([-.1,.2]))
    d=data()
    with pytest.raises(ValueError):replace(d,input_unit="CP10k")
    with pytest.raises(ValueError):replace(d,output_unit="RNA")
    with pytest.raises(ValueError):replace(d,X=np.full(d.X.shape,-.1))
    with pytest.raises(ValueError):replace(d,X=np.full(d.X.shape,1.1))
    with pytest.raises(ValueError):fit_target(d,(0.,.8))


def test_target_data_causal_alignment_and_ownership():
    d=data()
    with pytest.raises(ValueError):replace(d,input_keys=(("p",0),("p",0)))
    with pytest.raises(ValueError):replace(d,input_keys=(("p",True),("q",0)))
    with pytest.raises(ValueError):replace(d,source_bins=d.source_bins+1)
    with pytest.raises(ValueError):replace(d,target_rows=d.target_rows.astype(float))
    with pytest.raises(ValueError):replace(d,y=np.r_[np.nan,d.y[1:]])
    with pytest.raises(ValueError):replace(d,branch_ids=("b",))
    with pytest.raises(ValueError):replace(d,support_rows=np.zeros(len(d.y),dtype=int))
    with pytest.raises(ValueError):d.X[0,0]=.3
    with pytest.raises(ValueError):d.X.setflags(write=True)
    with pytest.raises(TypeError):d.provenance["source"]="changed"


def test_scores_frozen_from_wt_predictions_and_single_change_not_renormalized(baseline):
    d,m=baseline
    P=np.column_stack([s.predict(d.X[:,i]) for i,s in enumerate(m.singles)])
    expected=np.mean(np.abs(np.diff(P,axis=0)),axis=0)
    np.testing.assert_allclose(m.scores,expected,rtol=0,atol=0)
    np.testing.assert_allclose(m.weights,expected/expected.sum())
    assert min(m.weights)>=0 and sum(m.weights)==pytest.approx(1.)
    new=d.X.copy();new[:,0]=0.
    answer=predict_delta(m,d.X,new)
    expected=m.weights[0]*(m.singles[0].predict(new[:,0])-m.singles[0].predict(d.X[:,0]))
    np.testing.assert_allclose(answer.values,expected,atol=1e-15)
    assert answer.route=="singles" and m.weights[0]<1.
    assert answer.diagnostics["active_input_renormalization"] is False
    np.testing.assert_allclose(predict_target(m,new,wt_X=d.X).values-
                               predict_target(m,d.X).values,answer.values)


def test_adjacent_scores_do_not_bridge_gaps_or_branches():
    X=np.random.default_rng(17).uniform(.1,.9,(24,2))
    rows=np.r_[np.arange(12),np.arange(20,32)]
    branches=("a",)*8+("b",)*16
    d=data(X,rows=rows,branches=branches)
    m=fit_target(d,(0.,1.))
    P=np.column_stack([s.predict(X[:,i]) for i,s in enumerate(m.singles)])
    mask=(np.diff(rows)==1)&(np.array(branches[1:])==np.array(branches[:-1]))
    assert mask.sum()==21
    np.testing.assert_allclose(m.scores,np.mean(abs(np.diff(P,axis=0)[mask]),axis=0))
    assert m.diagnostics["n_adjacent_pairs"]==21


def test_lagged_keys_and_score_zero_no_uniform_weights():
    d=data(rows=np.arange(2,50),keys=(("same",0),("same",2)))
    m=fit_target(d,(0.,1.))
    assert m.input_keys==(("same",0),("same",2))
    d=replace(d,X=np.full(d.X.shape,.5))
    m=fit_target(d,(0.,1.))
    assert m.status=="no_information" and m.weights==(0.,0.) and not m.groups
    result=predict_delta(m,d.X,np.zeros_like(d.X))
    assert result.route=="no_information" and np.array_equal(result.values,np.zeros(len(d.y)))
    np.testing.assert_allclose(predict_target(m,d.X).values,d.y.mean())


def test_no_adjacent_rows_no_information():
    d=data(rows=np.arange(0,96,2))
    m=fit_target(d,(0.,1.))
    assert m.status=="no_information" and m.diagnostics["n_adjacent_pairs"]==0


def test_constant_rna_cap_roundoff_and_constant_r2_none():
    d=replace(data(y=np.full(48,.4)),output_unit="log1p_rna_log1p_cp10k")
    m=fit_target(d,rna_output_bounds(d.y))
    assert m.bounds==(0.,.4) and m.wt_mean<=.4
    assert m.status=="no_information" and all(s.metrics["training_r2"] is None for s in m.singles)
    result=predict_target(m,d.X)
    assert result.values.max()<=.4


def test_shared_cell_support_validation_and_ess_not_fake_rows():
    w=disjoint_weights(48)
    support=CellSupport(w)
    with pytest.raises(ValueError):CellSupport(w,"0"*64)
    with pytest.raises(ValueError):CellSupport(w*2)
    with pytest.raises(ValueError):CellSupport(np.zeros((48,3)))
    with pytest.raises(ValueError):support.weights[0,0]=0
    # Forty-eight identical kernels contain one effective row, not 48.
    redundant=np.full((48,64),1/64)
    m=fit_target(data(weights=redundant),(0.,1.))
    assert not m.groups
    attempt=m.diagnostics["group_attempts"][0]
    assert attempt["pooled_unique_cell_ess"]==pytest.approx(64.)
    assert attempt["kernel_effective_rows"]==pytest.approx(1.)
    assert attempt["reason"]=="kernel_effective_rows"


def test_low_unique_cell_ess_fails_group():
    w=np.full((48,4),.25)
    m=fit_target(data(weights=w),(0.,1.))
    assert m.diagnostics["group_attempts"][0]["reason"]=="training_cell_ess"


def test_missing_training_support_and_insufficient_rows_are_explicit(baseline):
    _,m=baseline
    assert m.diagnostics["group_attempts"][0]["reason"]=="training_cell_support_missing"
    x=np.linspace(.1,.9,12)
    d=data(X=np.column_stack([x,x]),weights=disjoint_weights(12))
    m=fit_target(d,(0.,1.))
    assert m.diagnostics["group_attempts"][0]["reason"]=="insufficient_group_rows"


def test_collinear_group_rejected_without_pca():
    x=np.linspace(.1,.9,48);X=np.column_stack([x,x])
    m=fit_target(data(X,weights=disjoint_weights(48)),(0.,1.))
    assert not m.groups
    assert m.diagnostics["group_attempts"][0]["reason"]=="rank_deficient_or_ill_conditioned_geometry"


def test_real_synthetic_group_gain_and_restart_evidence(grouped):
    d,m,_=grouped
    assert len(m.groups)<=2 and all(len(g.indices) in (2,3) for g in m.groups)
    for g in m.groups:
        assert g.diagnostics["penalized_gain"]>=m.config.min_penalized_gain
        assert g.diagnostics["restart_prediction_spread"]<=m.config.restart_prediction_spread_max
        assert g.diagnostics["stability_evidence"]=="deterministic_restarts_only_not_validation"
        assert g.diagnostics["n_rows"]==len(d.y)
        assert g.diagnostics["kernel_effective_rows"]==pytest.approx(64.)
        assert g.diagnostics["pooled_unique_cell_ess"]==pytest.approx(256.)


def test_group_is_whole_target_replacement_once_and_no_corrections(grouped):
    d,m,support=grouped
    chosen=sorted(m.groups,key=lambda g:(-g.diagnostics["penalized_gain"],g.indices))[0]
    new=d.X.copy();new[:,chosen.indices[0]]+=1e-5
    result=predict_delta(m,d.X,new,cell_support=support)
    assert result.route=="group"
    group=next(g for g in m.groups if g.indices==result.group_indices)
    expected=group.predict(new[:,group.indices])-group.predict(d.X[:,group.indices])
    np.testing.assert_array_equal(result.values,expected)
    assert result.diagnostics["group_corrections_added"] is False
    assert result.diagnostics["route_scope"]=="whole_query_batch"
    predicted=predict_target(m,new,wt_X=d.X,cell_support=support)
    np.testing.assert_array_equal(predicted.values,group.predict(new[:,group.indices]))
    assert predicted.values.min()>=0 and predicted.values.max()<=1


def test_missing_query_support_fallback_and_support_fingerprint_mismatch(grouped):
    d,m,_=grouped
    new=d.X.copy();new[:,0]+=.001
    result=predict_delta(m,d.X,new)
    assert result.route=="singles"
    assert any(v["reason"]=="query_cell_support_missing" for v in result.diagnostics["group_fallbacks"])
    other=CellSupport(disjoint_weights(64,8))
    with pytest.raises(ValueError,match="fingerprint"):predict_delta(m,d.X,new,cell_support=other)


def test_query_geometry_ood_and_wt_support_both_checked(grouped):
    d,m,support=grouped
    new=d.X.copy();new[:,:2]=0.
    result=predict_delta(m,d.X,new,cell_support=support)
    assert result.route=="singles"
    assert any(v["reason"]=="query_geometry_out_of_support" for v in result.diagnostics["group_fallbacks"])
    assert result.diagnostics["saturation_out_of_WT_support"]
    result=predict_delta(m,new,d.X,cell_support=support)
    assert result.route=="singles" # WT side is unsupported too.


def test_query_cell_ess_gate_no_rowcount_substitution(grouped):
    d,m,support=grouped
    strict=replace(m,config=replace(m.config,min_query_cell_ess=256.))
    new=d.X.copy();new[:,0]+=1e-6
    result=predict_delta(strict,d.X,new,cell_support=support)
    assert result.route=="singles"
    assert any(v["reason"]=="query_cell_ess" for v in result.diagnostics["group_fallbacks"])


def test_uncovered_input_union_whole_batch_falls_back(grouped):
    d,m,support=grouped
    two=next((g for g in m.groups if len(g.indices)==2),None)
    assert two is not None
    only_two=replace(m,groups=(two,))
    new=d.X.copy();new[0,0]+=.001;new[1,1]+=.001;new[2,2]+=.001
    result=predict_delta(only_two,d.X,new,cell_support=support)
    assert result.route=="singles" and not result.group_indices
    assert result.diagnostics["group_fallbacks"][0]["reason"]=="changed_inputs_not_covered"


def test_query_branch_local_unknown_branch_fallback(grouped):
    d,m,support=grouped
    new=d.X[:1].copy();new[:,0]+=1e-5
    result=predict_delta(m,d.X[:1],new,branch_ids=("unseen",),cell_support=support)
    assert result.route=="singles"
    assert any(v["reason"]=="query_branch_unsupported" for v in result.diagnostics["group_fallbacks"])


def test_true_top_three_group_and_three_changed_inputs_replacement(monkeypatch):
    rng=np.random.default_rng(18)
    X=rng.uniform(.1,.9,(96,3));u=(X-X.mean(axis=0))/X.std(axis=0)
    index=u[:,0]+.3*u[:,1]+.5*u[:,2]+1.5*u[:,0]*u[:,1]+2*u[:,1]*u[:,2]+.3*u[:,0]*u[:,2]
    y=.5+.45*index/(1+abs(index))
    d=data(X,y,disjoint_weights(96))
    # Isolate the group solver/routing gate with explicit successful singles.
    from CausalBridge import saturating_hybrid as core
    def successful_single(x,y,bounds,config):
        predicted=core._single_values(.5,.1,.5,.2,x,bounds)
        return SingleModel(.5,.1,.5,.2,bounds,(float(x.min()),float(x.max())),
            dict(core._metrics(y,predicted),status="ok",termination_status=1,nfev=1))
    monkeypatch.setattr(core,"fit_single",successful_single)
    m=fit_target(d,(0.,1.),HybridConfig(ridge=.001,min_query_cell_ess=1.))
    triple=next((g for g in m.groups if len(g.indices)==3),None)
    assert triple is not None,m.diagnostics
    assert triple.diagnostics["effective_rows_required"]==20
    new=X+1e-6
    result=predict_delta(m,X,new,cell_support=CellSupport(d.cell_weights))
    assert result.route=="group" and len(result.group_indices)==3
    np.testing.assert_array_equal(result.values,triple.predict(new[:,triple.indices])-triple.predict(X[:,triple.indices]))


def test_polynomial_parameter_count_effective_row_floor(monkeypatch):
    n=64
    w=.23*np.eye(n)+.77*np.full((n,n),1/n)
    X=np.random.default_rng(27).uniform(.1,.9,(n,3))
    d=data(X,weights=w)
    from CausalBridge import saturating_hybrid as core
    def successful_single(x,y,bounds,config):
        predicted=core._single_values(.5,.1,.5,.2,x,bounds)
        return SingleModel(.5,.1,.5,.2,bounds,(float(x.min()),float(x.max())),
            dict(core._metrics(y,predicted),status="ok",termination_status=1,nfev=1))
    monkeypatch.setattr(core,"fit_single",successful_single)
    m=fit_target(d,(0.,1.))
    attempt=next(a for a in m.diagnostics["group_attempts"] if len(a["indices"])==3)
    assert 12<attempt["kernel_effective_rows"]<20
    assert attempt["effective_rows_required"]==20 and attempt["reason"]=="kernel_effective_rows"


def test_group_solver_budget_and_gain_gates(grouped):
    d,_,_=grouped
    exhausted=fit_target(d,(0.,1.),HybridConfig(max_nfev=1,ridge=.001))
    assert not exhausted.groups
    assert exhausted.status=="no_information"
    assert not exhausted.diagnostics["group_attempts"]
    assert len(exhausted.provenance["excluded_single_relations"]["records"])==2
    strict=fit_target(d,(0.,1.),HybridConfig(ridge=.001,min_penalized_gain=.99))
    assert not strict.groups
    assert strict.diagnostics["group_attempts"][0]["reason"]=="insufficient_penalized_gain"


def test_shared_support_load_validation_recomputed_and_no_dataset_copy(grouped):
    _,m,support=grouped
    restored=TargetModel.from_bundle(m.to_bundle(),cell_support=support)
    assert restored.fingerprint==m.fingerprint
    evidence=validate_cell_support(restored,support)
    assert evidence["kernel_effective_rows"]==pytest.approx(64.)
    corrupted=m.to_bundle()
    corrupted["groups"][0]["diagnostics"]["pooled_unique_cell_ess"]+=1
    corrupted=resign(corrupted)
    # Reference-only loading is possible, but supplied evidence must validate.
    TargetModel.from_bundle(corrupted)
    with pytest.raises(ValueError,match="cell evidence"):
        TargetModel.from_bundle(corrupted,cell_support=support)


def test_independent_branch_local_geometry_not_hidden_by_global_rank():
    rng=np.random.default_rng(19)
    first=rng.uniform(.1,.9,(24,2));t=np.linspace(.1,.9,24)
    X=np.vstack([first,np.column_stack([t,t])])
    d=data(X,weights=disjoint_weights(48),branches=("a",)*24+("b",)*24)
    m=fit_target(d,(0.,1.))
    assert not m.groups
    assert m.diagnostics["group_attempts"][0]["reason"]=="rank_deficient_branch_geometry"


def test_exact_zero_change_and_declared_changes_protection(baseline):
    d,m=baseline
    result=predict_delta(m,d.X,d.X)
    assert result.route=="no_change" and np.array_equal(result.values,np.zeros(len(d.y)))
    new=d.X.copy();new[:,0]=0
    with pytest.raises(ValueError):predict_delta(m,d.X,new,changed_inputs=[m.input_keys[1]])
    with pytest.raises(ValueError):predict_delta(m,d.X,new,changed_inputs=[("absent",0)])
    with pytest.raises(ValueError):predict_delta(m,d.X[:1],new[:1])
    with pytest.raises(ValueError):predict_delta(m,d.X,np.full(d.X.shape,np.inf))
    with pytest.raises(ValueError):predict_delta(m,d.X,np.full(d.X.shape,1.1))


def test_nonnegative_rna_extreme_saturated_single_fallback():
    d=data()
    d=replace(d,input_unit="rna_log1p_cp10k",output_unit="log1p_rna_log1p_cp10k")
    m=fit_target(d,rna_output_bounds(d.y))
    keep=[d.input_keys.index(key) for key in m.input_keys]
    d=replace(d,input_keys=m.input_keys,X=d.X[:,keep],source_bins=d.source_bins[:,keep])
    new=np.full(d.X.shape,1e300)
    result=predict_target(m,new,wt_X=d.X)
    assert np.isfinite(result.values).all() and result.values.min()>=0 and result.values.max()<=m.bounds[1]
    assert result.diagnostics["saturation_out_of_WT_support"]


@pytest.mark.parametrize("use_group",[False,True])
def test_json_bundle_roundtrip_fingerprint_and_immutable_models(baseline,grouped,use_group):
    d,m=grouped[:2] if use_group else baseline
    bundle=json.loads(json.dumps(m.to_bundle(),allow_nan=False))
    assert "cell_weights" not in bundle
    restored=TargetModel.from_bundle(bundle)
    assert restored.fingerprint==m.fingerprint and restored.to_bundle()==m.to_bundle()
    new=d.X.copy();new[:,0]*=.8
    np.testing.assert_array_equal(predict_delta(restored,d.X,new).values,predict_delta(m,d.X,new).values)
    with pytest.raises(FrozenInstanceError):restored.weights=(1.,0.)
    with pytest.raises(TypeError):restored.diagnostics["alpha_frozen"]=False
    with pytest.raises(ValueError):restored.source_bins[0,0]=100
    if restored.groups:
        with pytest.raises(ValueError):restored.groups[0].coefficients[0]=3


def resign(bundle):
    bundle.pop("fingerprint",None)
    bundle["fingerprint"]=_digest(bundle)
    return bundle


def test_resigned_finite_scores_overflow_total_rejected(baseline):
    _,m=baseline
    bundle=m.to_bundle()
    bundle.update(scores=[1e308,1e308],weights=[0.,0.],status="ok")
    with pytest.raises(ValueError,match="WT score total must be finite"):
        TargetModel.from_bundle(resign(bundle))


@pytest.mark.parametrize("weights",[[0.,0.],[.4,.4],[.6,.6]])
def test_resigned_informative_weights_wrong_sum_rejected(baseline,weights):
    _,m=baseline
    bundle=m.to_bundle()
    bundle.update(scores=[1.,1.],weights=weights,status="ok")
    with pytest.raises(ValueError,match="informative weights must sum to one"):
        TargetModel.from_bundle(resign(bundle))


def test_resigned_no_information_zero_weights_exception_preserved(baseline):
    _,m=baseline
    bundle=m.to_bundle()
    bundle.update(scores=[0.,0.],weights=[0.,0.],status="no_information")
    restored=TargetModel.from_bundle(resign(bundle))
    assert restored.status=="no_information" and restored.weights==(0.,0.)
    bundle=restored.to_bundle()
    bundle["weights"]=[1e-13,0.]
    with pytest.raises(ValueError,match="no-information weights must be zero"):
        TargetModel.from_bundle(resign(bundle))


def test_fit_target_rejects_score_aggregate_overflow(monkeypatch,baseline):
    from CausalBridge import saturating_hybrid as core
    d,_=baseline
    original_mean=np.mean
    def overflowing_scores(values,*args,**kwargs):
        if np.shape(values)==(len(d.y)-1,len(d.input_keys)) and kwargs.get("axis")==0:
            return np.array([1e308,1e308])
        return original_mean(values,*args,**kwargs)
    # Controlled score-reduction injection only; not an actual-data fit claim.
    monkeypatch.setattr(core.np,"mean",overflowing_scores)
    with pytest.raises(ValueError,match="WT score total must be finite"):
        fit_target(d,(0.,1.))


@pytest.mark.parametrize("mutation",[
    lambda b:b.update(schema="old_schema"),
    lambda b:b.update(weights=[1.,0.]),
    lambda b:b.update(scores=[-1.,1.]),
    lambda b:b.update(status="no_information"),
    lambda b:b.update(input_keys=[["p",0],["p",0]]),
    lambda b:b.update(input_unit="unknown"),
    lambda b:b.update(bounds=[0.,.8]),
    lambda b:b["singles"][0].update(s=0.),
    lambda b:b["singles"][0].update(a=100.),
    lambda b:b["singles"][0]["metrics"].update(n_rows=1),
    lambda b:b["config"].update(workers=2),
    lambda b:b.update(source_bins=[[0,0]]),
    lambda b:b.update(provenance={"bad":float("nan")}),
    lambda b:b["diagnostics"].update(alpha_frozen=False),
    lambda b:b["diagnostics"].update(active_input_renormalization=True),
])
def test_resigned_corrupt_bundle_rejected(baseline,mutation):
    _,m=baseline
    bundle=copy.deepcopy(m.to_bundle());mutation(bundle)
    with pytest.raises(ValueError):TargetModel.from_bundle(resign(bundle))


def test_unsigned_mutation_rejected_and_group_corruption_rejected(grouped):
    _,m,_=grouped
    bundle=m.to_bundle();bundle["wt_mean"]+=.01
    with pytest.raises(ValueError,match="fingerprint"):TargetModel.from_bundle(bundle)
    mutations=[lambda g:g.update(scale=[0.]*len(g["scale"])),
               lambda g:g.update(coefficients=[1.]),
               lambda g:g.update(radius_by_branch={"b":0.}),
               lambda g:g.update(radius_by_branch={"b":1e10}),
               lambda g:g["diagnostics"].update(penalized_gain=-1),
               lambda g:g["diagnostics"].update(restart_prediction_spread=-1),
               lambda g:g.update(coefficients=[1000.]*len(g["coefficients"])),
               lambda g:g.update(support_rows=[0]*len(g["support_rows"]))]
    for mutation in mutations:
        bundle=copy.deepcopy(m.to_bundle());mutation(bundle["groups"][0])
        with pytest.raises(ValueError):TargetModel.from_bundle(resign(bundle))


def test_deterministic_full_wt_fit_and_synthetic_50_single_budget():
    d=data()
    m1=fit_target(d,(0.,1.));m2=fit_target(d,(0.,1.))
    assert m1.to_bundle()==m2.to_bundle()
    x=np.linspace(0.,1.,32);y=.5+.3*(x-.4)/(.2+abs(x-.4))
    begin=time.monotonic()
    for _ in range(50):fit_single(x,y,(0.,1.))
    elapsed=time.monotonic()-begin
    print(f"50 synthetic singles: {elapsed:.3f}s")
    assert elapsed<30.
