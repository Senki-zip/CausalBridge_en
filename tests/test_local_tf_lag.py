"""Per-target-bin routes, masks, and replay contracts, using small WT fixtures."""
from dataclasses import replace
from types import SimpleNamespace
import copy

import numpy as np
import pandas as pd
import pytest

from CausalBridge import kinetics, saturating_hybrid as core
from CausalBridge import saturating_transfer as transfer
from CausalBridge import saturating_perturbation as engine


def screen(x, y, maxlag=8, boundaries=None):
    return kinetics._fit_tf_peak_regression({"P":["TF"]},
        SimpleNamespace(var_names=["TF"]),SimpleNamespace(var_names=["P"]),
        np.asarray(x)[:,None],np.asarray(y)[:,None],max_lag=maxlag,
        branch_boundaries=boundaries,min_abs_contribution=.01)[0]


def test_every_bin_including_branch_starts_has_own_route():
    x = np.r_[np.linspace(.1,2,12),np.linspace(4,1,12)]
    edge = screen(x,.1+.1*x,10,[(0,12),(12,24)])
    assert edge["lag_mode"] == "local" and edge["tf_lag"] is None
    assert len(edge["local_routes"]) == 24
    assert all(r["source_bin"] == r["target_bin"] and r["selected_lag"] == 0
               and r["status"] == "supported" for r in edge["local_routes"])


def test_no_supported_lag_is_explicit_without_forced_zero():
    edge = screen(np.ones(20),np.linspace(.1,.8,20))
    assert len(edge["local_routes"]) == 20
    assert all(r["source_bin"] == r["selected_lag"] == -1 and
               r["status"] == "no_supported_lag" for r in edge["local_routes"])
    assert all(r["n_evidence_rows"] == 12 and r["n_stat_tests"] == 0
               for r in edge["local_routes"])
    assert [r["n_lag_attempts"] for r in edge["local_routes"]] == [min(b,8)+1 for b in range(20)]


@pytest.mark.parametrize("n",[5,20])
def test_failed_pearson_routes_preserve_evidence_without_selected_statistics(monkeypatch,n):
    calls = []
    def reject(x, y):
        calls.append(len(x))
        return -.49, .105
    monkeypatch.setattr(kinetics,"pearsonr",reject)
    routes = screen(np.arange(float(n)),np.linspace(.1,.8,n),2)["local_routes"]
    assert calls and max(calls) == min(n,12)
    assert sum(r["n_stat_tests"] for r in routes) == len(calls)
    for b, route in enumerate(routes):
        assert route["status"] == "no_supported_lag"
        assert route["source_bin"] == route["selected_lag"] == -1
        assert route["n_evidence_rows"] == min(n,12)
        assert route["n_lag_attempts"] == route["n_stat_tests"] == min(b,2)+1
        assert route["weight"] is route["p_value"] is route["abs_contribution"] is None


def test_insufficient_evidence_counts_valid_pairs_without_a_statistical_test(monkeypatch):
    def unexpected(*args):
        pytest.fail("Pearson needs at least three valid pairs")
    monkeypatch.setattr(kinetics,"pearsonr",unexpected)
    routes = screen([1.,2.,np.nan],[.1,.2,.3],1)["local_routes"]
    assert [r["n_evidence_rows"] for r in routes] == [2,2,2]
    assert [r["n_lag_attempts"] for r in routes] == [1,2,1]
    assert all(r["n_stat_tests"] == 0 and r["status"] == "no_supported_lag" for r in routes)
    unavailable = screen([np.nan,np.nan],[.1,.2],1)["local_routes"]
    assert all(r["n_evidence_rows"] == r["n_lag_attempts"] == r["n_stat_tests"] == 0
               for r in unavailable)


def test_supported_evidence_count_belongs_to_selected_lag(monkeypatch):
    def accept_shorter(x, y):
        return 1., .5 if len(x) == 5 else .001
    monkeypatch.setattr(kinetics,"pearsonr",accept_shorter)
    route = screen(np.arange(5.),np.linspace(.1,.8,5),1)["local_routes"][1]
    assert route["status"] == "supported" and route["selected_lag"] == 1
    assert route["n_evidence_rows"] == 4  # rejected lag0 had five valid pairs
    assert route["n_lag_attempts"] == route["n_stat_tests"] == 2


def test_true_delay_only_looks_back_from_each_bin():
    rng = np.random.default_rng(7)
    x = rng.uniform(.1,1,50)
    y = np.r_[.44,.33,x[:-2]]
    edge = screen(x,y,3)
    routes = edge["local_routes"]
    assert routes[0]["selected_lag"] in (-1,0)
    assert sum(r["selected_lag"] == 2 for r in routes) > 35
    for r in routes:
        if r["status"] == "supported":
            assert 0 <= r["source_bin"] <= r["target_bin"]
            assert r["target_bin"]-r["source_bin"] == r["selected_lag"] <= 3


def test_nearest_window_includes_current_bin_and_earlier_distance_tie(monkeypatch):
    calls = []
    def pearson(x,y):
        calls.append(x.copy())
        return 1.,.001
    monkeypatch.setattr(kinetics,"pearsonr",pearson)
    x = np.arange(20,dtype=float)
    edge = screen(x,.1+.01*x)
    assert len(calls) == 20  # one independent first-lag decision per target
    assert np.array_equal(calls[10],x[4:16])
    assert all(r["n_evidence_rows"] == 12 for r in edge["local_routes"])


def test_supported_lag_changes_with_local_wt_relationship():
    rng = np.random.default_rng(37)
    x = rng.uniform(.1,1.,100)
    y = x.copy(); y[50:] = x[48:-2]
    routes = screen(x,y,3)["local_routes"]
    assert all(r["selected_lag"] == 0 for r in routes[:40])
    assert sum(r["selected_lag"] == 2 for r in routes[60:]) > 30


def local_data():
    n = 24
    rows = np.arange(n)
    source = np.column_stack([rows, np.maximum(rows-rows%3,0)])
    supported = np.ones(source.shape,dtype=bool)
    supported[[3,8,15],0] = False
    supported[[0,1,6,7],1] = False
    source[~supported] = -1
    x = np.column_stack([.2+rows/30, .1+np.sin(rows/4)**2])
    y = .15+.3*x[:,0]+.2*x[:,1]
    x[~supported] = 0
    return core.TargetData("P",(("TF","local"),("U","local")),x,y,source,rows,
        ("b",)*n,"rna_log1p_cp10k","raw_atac",
        dict(relation_type=transfer.TF_TYPE,local_lag_max=3),route_supported=supported)


@pytest.mark.parametrize("family",["saturation","spline","nn"])
def test_supported_fit_scores_masks_and_strict_roundtrip(family):
    data = local_data()
    if family == "spline":
        from CausalBridge.spline_hybrid import SplineConfig
        data = replace(data,provenance=dict(data.provenance,response_family=family,
            spline_response_config=SplineConfig(max_nfev=300,restarts=1).to_dict()))
    if family == "nn":
        data = replace(data,provenance=dict(data.provenance,response_family=family,
            nn_response_config=dict(epochs=3)))
    model = core.fit_target(data,(0.,1.),core.HybridConfig(max_nfev=1000,min_group_rows=100))
    assert [s.metrics["n_rows"] for s in model.singles] == [21,20]
    assert model.target_rows[0] == 0 and len(model.target_rows) == 24
    predicted = core._single_predictions(model.singles,data.X,model.route_supported,
        model.diagnostics["single_WT_reference_means"])
    expected = (abs(np.diff(predicted,axis=0)) * (data.route_supported[1:] & data.route_supported[:-1])).sum(axis=0)/23
    assert np.allclose(model.scores, expected)
    loaded = core.TargetModel.from_bundle(model.to_bundle())
    assert loaded.fingerprint == model.fingerprint
    new = data.X.copy(); new[~data.route_supported] = 10
    answer = core.predict_delta(loaded,data.X,new)
    assert np.array_equal(answer.values,np.zeros(24))
    with pytest.raises(ValueError,match="input_support"):
        core.predict_delta(loaded,data.X[:1],new[:1],branch_ids=("b",))
    with pytest.raises(ValueError,match="unsupported frozen route"):
        core.predict_delta(loaded,data.X,new,input_support=np.ones(data.X.shape,dtype=bool))
    for field in ("source_bins","selected_lag","route_supported"):
        bundle = copy.deepcopy(model.to_bundle())
        if field == "route_supported":
            bundle[field][3][0] = True
        else:
            bundle[field][0][0] = 99
        bundle["fingerprint"] = core._digest({k:v for k,v in bundle.items() if k != "fingerprint"})
        with pytest.raises(ValueError):
            core.TargetModel.from_bundle(bundle)


def test_masked_sources_never_index_last_or_future_bin():
    data = local_data()
    model = core.fit_target(data,(0.,1.),core.HybridConfig(max_nfev=1000,min_group_rows=100))
    engine._bind_rows(model,dict(bin_to_support=list(range(24))))
    states = np.arange(48).reshape(24,2).astype(float)
    X = engine._inputs(model,np.arange(24),states,{"TF":0,"U":1})
    assert np.array_equal(X[~data.route_supported],np.zeros((~data.route_supported).sum()))
    for i in range(2):
        mask = data.route_supported[:,i]
        assert np.array_equal(X[mask,i],states[data.source_bins[mask,i],i])
    mask = engine._input_support(model,np.arange(24),True)
    lags = np.where(data.route_supported,data.target_rows[:,None]-data.source_bins,100)
    assert np.array_equal(mask,data.route_supported & (lags==lags.min(axis=1)[:,None]))


def test_group_fit_receives_only_common_supported_rows(monkeypatch):
    data = local_data()
    captured = []
    def fit_group(group_data, indices, baseline, bounds, cfg):
        captured.append((group_data,indices,baseline))
        assert group_data.route_supported[:,indices].all()
        return None, dict(indices=list(indices),accepted=False,reason="test_capture")
    monkeypatch.setattr(core,"_fit_group",fit_group)
    model = core.fit_target(data,(0.,1.),core.HybridConfig(max_nfev=1000))
    assert len(captured) == 1
    common = data.route_supported.all(axis=1)
    assert np.array_equal(captured[0][0].target_rows,data.target_rows[common])
    predictions = core._single_predictions(model.singles,data.X,data.route_supported,
        model.diagnostics["single_WT_reference_means"])
    assert np.allclose(captured[0][2],(predictions@model.weights)[common])


def test_all_unsupported_edge_is_audited_without_fit_or_fake_response():
    data = local_data()
    data = replace(data,source_bins=np.full(data.X.shape,-1,dtype=int),
        route_supported=np.zeros(data.X.shape,dtype=bool),selected_lag=np.full(data.X.shape,-1,dtype=int))
    with pytest.raises(core.TargetFitFailure) as error:
        core.fit_target(data,(0.,1.))
    records = error.value.diagnostics["excluded_single_relations"]["records"]
    assert {r["status"] for r in records} == {"no_supported_lag"}


def test_accepted_local_group_subset_roundtrip_and_query_mask():
    from test_saturating_hybrid import disjoint_weights
    rng = np.random.default_rng(37)
    X = rng.uniform(.1,.9,(64,3))
    u = (X[:,:2]-X[:,:2].mean(axis=0))/X[:,:2].std(axis=0)
    X[:,2] = .5
    index = 2*u[:,0]*u[:,1]+.8*u[:,0]+.5*u[:,1]
    y = .5+.45*index/(1+np.abs(index))
    mask = np.ones(X.shape,dtype=bool)
    mask[:2,0] = False; mask[2:4,1] = False
    rows = np.arange(64)
    source = np.broadcast_to(rows[:,None],X.shape).copy(); source[~mask] = -1
    data = core.TargetData("P",tuple((tf,"local") for tf in ("TF","U","V")),X,y,source,rows,
        ("b",)*64,"rna_log1p_cp10k","raw_atac",dict(relation_type=transfer.TF_TYPE,local_lag_max=0,budget_exhausted_policy="best_finite"),
        cell_weights=disjoint_weights(64),route_supported=mask)
    model = core.fit_target(data,(0.,1.),core.HybridConfig(ridge=.001,min_query_cell_ess=1.))
    assert model.groups, model.diagnostics
    support = core.CellSupport(data.cell_weights)
    model = core.TargetModel.from_bundle(model.to_bundle(),cell_support=support)
    group = model.groups[0]
    common = mask[:,group.indices].all(axis=1)
    assert len(group.support_rows) == common.sum() < 64
    old = X[common]; new = old.copy(); new[:,group.indices[0]] += 1e-5
    result = core.predict_delta(model,old,new,branch_ids=("b",)*len(old),target_rows=rows[common],
        input_support=mask[common],cell_support=support)
    assert result.route == "group"
    reduced = mask[common].copy(); reduced[0,group.indices[0]] = False
    result = core.predict_delta(model,old,new,branch_ids=("b",)*len(old),target_rows=rows[common],
        input_support=reduced,cell_support=support)
    assert result.route == "singles"
    assert result.values[0] == 0


def test_local_keys_rejected_for_peak_gene_and_future_sources_rejected():
    data = local_data()
    with pytest.raises(ValueError,match="reserved"):
        replace(data,provenance=dict(relation_type=transfer.GENE_TYPE,local_lag_max=3))
    source = data.source_bins.copy(); source[0,0] = 1
    with pytest.raises(ValueError,match="source"):
        replace(data,source_bins=source)


@pytest.mark.parametrize("branches",[1,2])
def test_local_registry_public_simulation_keeps_all_branch_starts(branches):
    from test_saturating_transfer import fixture_data
    r,a,pt,edges,_,cfg = fixture_data(32,branches)
    snapshot = transfer._snapshot(r,a,pt,cfg)
    records = kinetics._fit_tf_peak_regression({"p0":["A"]},r,a,snapshot["R"],snapshot["A"],
        max_lag=8,branch_boundaries=snapshot["boundaries"],min_abs_contribution=.01)
    frame = pd.DataFrame(records)
    _,_,genes = transfer.fit_peak_gene(r,a,pt,edges,cfg)
    tfs = transfer.fit_tf_peak(r,a,pt,frame,cfg)
    assert tfs["schema"] == "hybrid_saturation_tf_peak_local_registry_v3"
    model = core.TargetModel.from_bundle(tfs["targets"]["p0"])
    assert len(model.target_rows) == 32
    assert (model.target_rows == 0).sum() == branches
    result = engine.propagate_saturating(["A"],r,a,pt,edges,None,frame,cfg,
        transfer_models=genes,tf_model_registry=tfs)
    assert not result["atac_unavailable_mask"].any()
    contribution = result["saturating_tf_peak_contributions"]
    assert (contribution.source_bin <= contribution.bin).all()
    assert np.array_equal(contribution.tf_lag,contribution.bin-contribution.source_bin)
