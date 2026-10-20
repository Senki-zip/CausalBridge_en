import copy
import json
from dataclasses import replace
from types import SimpleNamespace
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from CausalBridge import saturating_perturbation as pert
from CausalBridge import saturating_transfer as transfer
from CausalBridge import saturating_hybrid as core
from CausalBridge.saturating_projection import cell_l0_from_saturating_payload,SCHEMA
from test_saturating_transfer import fitted,fixture_data


def execute(f,config=None,targets=("A",)):
    r,a,pt,e,tf,cfg,table,metrics,genes,tfs=f
    return pert.propagate_saturating(targets,r,a,pt,e,table,tf,cfg if config is None else config,
        transfer_models=genes,tf_model_registry=tfs)


def test_partial_ko_is_cell_raw_cp10k_not_l1_multiplier(fitted):
    out=execute(fitted)
    payload=out["projection_inputs"][0]
    assert payload["effect_schema_version"]==SCHEMA
    source=fitted[0].X.toarray()
    expected=np.log1p(np.expm1(source[:,0])*.5)
    np.testing.assert_allclose(payload["direct_cell_states"][:,0],expected)
    assert not np.allclose(expected,source[:,0]*.5)
    ids,genes,cp=cell_l0_from_saturating_payload(payload)
    np.testing.assert_allclose(cp[:,0],np.expm1(source[:,0])*.5,rtol=1e-12)
    assert len(ids)==80 and genes==("A","B","C")
    endpoint=out["saturating_endpoints"].set_index("affected_gene").loc["A"]
    assert endpoint.delta_rna==pytest.approx(-1.)
    assert out["saturating_context_metadata"]["alpha_frozen"] is True


def test_completion_summary_public_gene_and_unique_relations(fitted, caplog):
    r,a,pt,e,tf,cfg,table,_,genes,tfs=fitted
    with caplog.at_level("INFO", logger=pert.__name__):
        out=pert.propagate_saturating(["A"],r,a,pt,e,table,tf,cfg,
            transfer_models=genes,tf_model_registry=tfs,cluster_label="cluster_0")
    public=out["perturbation_results"]
    assert len(public)>0 and "A" not in set(public.affected_gene)
    assert f"扰动完成 [cluster_0/A]: 扰动基因={public.affected_gene.nunique()}" in caplog.text
    assert f"结果行={len(public)}" in caplog.text
    tf_rows=out["saturating_tf_peak_contributions"]
    gene_rows=out["saturating_peak_attributions"]
    assert len(tf_rows)>len(tf_rows[["tf_gene","peak_id"]].drop_duplicates())
    tf_count=len(tf_rows.loc[np.isfinite(tf_rows.delta_atac_raw)&(tf_rows.delta_atac_raw!=0),
        ["tf_gene","peak_id"]].drop_duplicates())
    gene_count=len(gene_rows.loc[np.isfinite(gene_rows.delta_l2)&(gene_rows.delta_l2!=0),
        ["peak_id","gene"]].drop_duplicates())
    assert f"TF→peak={tf_count}; peak→gene={gene_count}" in caplog.text


def test_zero_ko_summary_does_not_count_fitted_network(fitted, caplog):
    cfg=copy.deepcopy(fitted[5]);cfg["perturbation"]["ko_strength"]=0
    assert fitted[-2]["targets"] and fitted[-1]["targets"]
    with caplog.at_level("INFO", logger=pert.__name__):
        out=execute(fitted,cfg)
    assert out["perturbation_results"].empty
    assert "扰动基因=0; 质量排除=0; 不可用=0（基因）/0（peak）; 结果行=0" in caplog.text
    assert "TF→peak=0; peak→gene=0" in caplog.text


def test_quality_pruned_completion_summary(fitted, caplog):
    cfg=copy.deepcopy(fitted[5]);cfg["perturbation"]["min_r2_threshold"]=1.1
    with caplog.at_level("INFO", logger=pert.__name__):
        out=execute(fitted,cfg)
    assert out["perturbation_results"].empty
    assert "扰动基因=0; 质量排除=2; 不可用=0（基因）/0（peak）" in caplog.text
    assert "peak→gene=0" in caplog.text


def test_summary_filters_failed_nonfinite_and_zero_contributions(caplog):
    public=pd.DataFrame(dict(affected_gene=["B","A","C","D","E","F","G"],
        status=["ok","ok","ok","ok","unavailable","ok","ok"],
        selected=[True,True,True,False,True,True,True],
        direct_target=[False,True,False,False,False,False,False],
        delta_rna=[.5,1.,np.inf,.5,np.nan,.1,0.]))
    support=pd.DataFrame(dict(bin=[0,1,2,0,1,2],target_id=["p0"]*3+["B"]*3,
        relation_type=[transfer.TF_TYPE]*3+[transfer.GENE_TYPE]*3,
        status=["ok","ok","numerical_failure"]*2))
    result=dict(target_gene="A",perturbation_results=public,excluded_gene_ids=["Q"],
        rna_unavailable_mask=np.array([[False,True],[False,False]]),
        atac_unavailable_mask=np.array([[True],[False]]),saturating_support_diagnostics=support,
        saturating_tf_peak_contributions=pd.DataFrame(dict(bin=[0,1,2,0,0,0],
            tf_gene=["A","A","failed","nan","infinite","zero"],peak_id=["p0"]*6,
            delta_atac_raw=[.2,.3,.4,np.nan,np.inf,0.])),
        saturating_peak_attributions=pd.DataFrame(dict(bin=[0,1,2,0,0],gene=["B"]*5,
            peak_id=["p0","p0","failed","nan","zero"],delta_l2=[.2,.3,.4,np.nan,0.])))
    with caplog.at_level("INFO", logger=pert.__name__):
        pert._log_intervention_summary(result,"cluster_0",.1)
    assert "扰动基因=1; 质量排除=1; 不可用=1（基因）/1（peak）; 结果行=7" in caplog.text
    assert "TF→peak=1; peak→gene=1" in caplog.text


def test_full_and_zero_ko_exact_direct_clamp(fitted):
    for strength in (0.,1.):
        cfg=copy.deepcopy(fitted[5]);cfg["perturbation"]["ko_strength"]=strength
        out=execute(fitted,cfg);payload=out["projection_inputs"][0]
        _,_,cp=cell_l0_from_saturating_payload(payload)
        if strength==1:
            assert np.array_equal(cp[:,0],np.zeros(len(cp)))
            assert np.array_equal(payload["perturbed_bin_states"][:,0],np.zeros(20))
        else:
            np.testing.assert_allclose(cp,np.expm1(fitted[0].X.toarray()),rtol=1e-12)
            np.testing.assert_array_equal(out["atac_state_history"],out["control_atac_states"])


def test_physical_atac_and_rna_caps_frozen_weights_and_screen_only(fitted):
    before=copy.deepcopy(fitted[-2])
    first=execute(fitted)
    changed=list(fitted);changed[4]=changed[4].copy();changed[4].weight*=1e20
    second=execute(changed)
    np.testing.assert_array_equal(first["rna_state_history"],second["rna_state_history"])
    assert fitted[-2]==before
    assert first["rna_state_history"].min()>=0
    assert first["atac_state_history"].min()>=0 and first["atac_state_history"].max()<=1
    assert (first["saturating_support_diagnostics"].route=="weighted_singles").all()


def test_peak_relative_zero_and_absolute_raw_clamp(fitted):
    r,a,pt,e,tf,cfg,table,_,genes,tfs=fitted
    for mode,strength in (("relative",-1.),("absolute",.3)):
        c=copy.deepcopy(cfg);c["perturbation"]["peak_ko"]={"mode":mode,"strength":strength}
        out=pert.perturb_peak_saturating("p0",r,a,pt,e,table,tf,c,
            transfer_models=genes,tf_model_registry=tfs)
        np.testing.assert_allclose(out["atac_state_history"][:,0],0. if mode=="relative" else .3)
        assert out["target_gene"]=="PEAK:p0"
        assert not out["projection_inputs"]["target_genes"]


def test_quality_pruned_gene_stays_wt_and_targets_proxy_rejected(fitted):
    cfg=copy.deepcopy(fitted[5]);cfg["perturbation"]["min_r2_threshold"]=1.1
    out=execute(fitted,cfg)
    np.testing.assert_array_equal(out["rna_state_history"][:,1:],out["control_rna_states"][:,1:])
    assert set(out["excluded_gene_ids"])=={"B","C"}
    r,a,pt,e,tf,cfg,table,_,genes,tfs=fitted
    with pytest.raises(ValueError,match="TARGET"):
        pert.propagate_saturating(["A"],r,a,pt,e,table,tf,cfg,transfer_models=genes,
            tf_model_registry=tfs,gene_types={"A":"TARGET"})


def test_changed_data_or_configuration_rejects_registry(fitted):
    changed=list(fitted);changed[0]=changed[0].copy();changed[0].X=changed[0].X*1.01
    with pytest.raises(ValueError,match="WT context"):execute(changed)
    cfg=copy.deepcopy(fitted[5]);cfg["atac_to_rna"]["saturating_hybrid"]={"ridge":.2}
    with pytest.raises(ValueError,match="WT context/config"):execute(fitted,cfg)


def test_joint_and_independent_target_payloads(fitted):
    joint=execute(fitted,targets=("A","B"))
    assert len(joint["projection_inputs"])==1
    cfg=copy.deepcopy(fitted[5]);cfg["perturbation"]["joint_knockout"]=False
    separate=execute(fitted,cfg,targets=("A","B"))
    assert len(separate["projection_inputs"])==2
    assert [tuple(p["target_genes"]) for p in separate["projection_inputs"]]==[("A",),("B",)]


def test_invalid_gene_and_peak_strength_rejected(fitted):
    cfg=copy.deepcopy(fitted[5]);cfg["perturbation"]["ko_strength"]=1.5
    with pytest.raises(ValueError,match="ko_strength"):execute(fitted,cfg)


@pytest.fixture(scope="module")
def grouped_fit():
    rng=np.random.default_rng(37);bins=64;per=20
    grid=np.linspace(.15,.85,8)
    X=np.array([(a,d) for a in grid for d in grid])[rng.permutation(bins)]
    u=(X-X.mean(axis=0))/X.std(axis=0)
    z=u[:,0]+u[:,1]+.5*u[:,0]*u[:,1]
    raw=.5+.45*z/(1+abs(z));counts=np.rint(raw*per).astype(int)
    A=np.zeros((bins*per,1))
    for b,n in enumerate(counts):A[b*per:b*per+n,0]=1
    means=counts/per;previous=np.r_[means[0],means[:-1]]
    B=np.expm1(.5+.15*(previous-.5)/(.2+abs(previous-.5)))
    R=np.column_stack([np.repeat(X,per,axis=0),np.repeat(B,per)])
    ids=[f"g{i}" for i in range(bins*per)]
    rna=ad.AnnData(sparse.csr_matrix(R),obs=pd.DataFrame(index=ids),var=pd.DataFrame(index=["A","D","B"]))
    atac=ad.AnnData(sparse.csr_matrix(A),obs=rna.obs.copy(),
        var=pd.DataFrame(dict(chr=["1"],start=[10],end=[20]),index=["p0"]))
    pt=pd.DataFrame(dict(bin=np.repeat(np.arange(bins),per),
        pseudotime=np.repeat(np.linspace(0,1,bins),per),branch="0"),index=ids)
    edges=pd.DataFrame(dict(gene=["B"],peak_id=["p0"]))
    tf=pd.DataFrame(dict(tf_gene=["A","D"],peak_id=["p0","p0"],tf_lag=[0,0],weight=[-999.,999.]))
    cfg=fixture_data()[-1]
    cfg["atac_to_rna"]["saturating_hybrid"]={"ridge":.001}
    cfg["perturbation"]["ko_strength"]=.001
    table,metrics,genes=transfer.fit_peak_gene(rna,atac,pt,edges,cfg)
    tfs=transfer.fit_tf_peak(rna,atac,pt,tf,cfg)
    model=core.TargetModel.from_bundle(tfs["targets"]["p0"])
    assert model.groups,model.diagnostics
    return rna,atac,pt,edges,tf,cfg,table,metrics,genes,tfs


def test_real_group_route_locked_for_whole_trajectory(grouped_fit):
    out=execute(grouped_fit,targets=("A","D"))
    metadata=out["saturating_context_metadata"]
    assert metadata["route_passes"]==2 and metadata["route_replay_count"]==0,(grouped_fit[-1]["target_metrics"],metadata)
    assert "saturating_hybrid_tf_peak:p0" in metadata["locked_groups"]
    support=out["saturating_support_diagnostics"]
    tf=support[support.relation_type==transfer.TF_TYPE]
    assert set(tf.route)=={"group"}
    assert set(out["saturating_tf_peak_contributions"].method)=={"ordered_telescoping_not_causal"}


def test_group_support_loss_replays_entire_intervention_not_spliced(grouped_fit,monkeypatch):
    actual_predict=core.predict_delta
    proxy=SimpleNamespace(**core.__dict__)
    injected=[]
    def lose_support(model,old,new,**kwargs):
        answer=actual_predict(model,old,new,**kwargs)
        if len(old)==1 and kwargs.get("cell_support") is not None and answer.route=="group" and not injected:
            injected.append(True)
            return replace(answer,route="singles",group_indices=())
        return answer
    # Adapter-only routing fault injection; original core/source fingerprints stay unchanged.
    proxy.predict_delta=lose_support;monkeypatch.setattr(pert,"core",proxy)
    out=execute(grouped_fit,targets=("A","D"))
    assert injected and out["saturating_context_metadata"]["route_replay_count"]==1
    assert out["saturating_context_metadata"]["route_passes"]==3
    assert not out["saturating_context_metadata"]["locked_groups"]
    assert set(out["saturating_support_diagnostics"].route)=={"weighted_singles"}
    def force_weighted(model,old,new,**kwargs):
        kwargs["cell_support"]=None
        return actual_predict(model,old,new,**kwargs)
    proxy.predict_delta=force_weighted
    baseline=execute(grouped_fit,targets=("A","D"))
    np.testing.assert_array_equal(out["rna_state_history"],baseline["rna_state_history"])
    np.testing.assert_array_equal(out["atac_state_history"],baseline["atac_state_history"])


def test_branch_warmup_unchanged_no_cross_branch_carry():
    r,a,pt,e,tf,cfg=fixture_data(32,2)
    table,metrics,genes=transfer.fit_peak_gene(r,a,pt,e,cfg)
    tfs=transfer.fit_tf_peak(r,a,pt,tf,cfg)
    f=(r,a,pt,e,tf,cfg,table,metrics,genes,tfs)
    out=execute(f)
    np.testing.assert_array_equal(out["rna_state_history"][[0,16],1:],out["control_rna_states"][[0,16],1:])


def test_resigned_wrong_training_quality_rejected_against_actual_wt(fitted):
    changed=list(fitted);registry=copy.deepcopy(changed[-2])
    registry["target_metrics"]["B"]["train_r2"]=.99999
    registry.pop("fingerprint");registry["fingerprint"]=core._digest(registry)
    changed[-2]=registry
    with pytest.raises(ValueError,match="quality differs from frozen WT"):execute(changed)


def test_screen_topology_not_weights_must_match_fitted_models(fitted):
    changed = list(fitted)
    changed[4] = changed[4].iloc[:0].copy()
    with pytest.raises(ValueError, match="screening topology"):
        execute(changed)
    changed[4] = fitted[4].copy(); changed[4].loc[0, "tf_lag"] = 2
    with pytest.raises(ValueError, match="input keys/lags"):
        execute(changed)


def test_min_lag_filter_masks_later_inputs_without_refitting_or_renormalizing():
    r, a, pt, edges, _, cfg = fixture_data()
    tf = pd.DataFrame({"tf_gene": ["A", "B"], "peak_id": ["p0", "p0"],
                       "tf_lag": [0, 1], "weight": [999., -999.]})
    table, metrics, genes = transfer.fit_peak_gene(r, a, pt, edges, cfg)
    tfs = transfer.fit_tf_peak(r, a, pt, tf, cfg)
    before = copy.deepcopy(tfs)
    args = (r, a, pt, edges, tf, cfg, table, metrics, genes, tfs)
    filtered = copy.deepcopy(cfg); filtered["perturbation"]["min_lag_per_peak_filter"] = True
    out = execute(args, filtered, targets=("B",))
    np.testing.assert_array_equal(out["atac_state_history"], out["control_atac_states"])
    unfiltered = copy.deepcopy(cfg); unfiltered["perturbation"]["min_lag_per_peak_filter"] = False
    other = execute(args, unfiltered, targets=("B",))
    assert not np.array_equal(other["atac_state_history"], other["control_atac_states"])
    assert tfs == before


def test_closure_work_budget_is_consumed(fitted):
    cfg = copy.deepcopy(fitted[5]); cfg["perturbation"]["subnetwork_unlimited_max_work_edges"] = 1
    with pytest.raises(ValueError, match="work-edge budget.*budget=1, scanned=2"):
        execute(fitted, cfg)


@pytest.fixture(scope="module")
def lag_failure_fit():
    r, a, pt, _, _, cfg = fixture_data()
    r.var_names = ["D", "A", "B"]
    a.X = sparse.hstack([a.X[:, 0], a.X[:, 0]], format="csr")
    edges = pd.DataFrame({"gene": ["A", "B"], "peak_id": ["p1", "p1"]})
    tf = pd.DataFrame({"tf_gene": ["A", "B", "D"], "peak_id": ["p0", "p0", "p1"],
                       "tf_lag": [0, 1, 0], "weight": [999., -999., 999.]})
    cfg["perturbation"]["min_lag_per_peak_filter"] = True
    table, metrics, genes = transfer.fit_peak_gene(r, a, pt, edges, cfg)
    tfs = transfer.fit_tf_peak(r, a, pt, tf, cfg)
    assert all(v["status"] == "ok" for v in genes["target_metrics"].values())
    assert all(v["status"] == "ok" for v in tfs["target_metrics"].values())
    return r, a, pt, edges, tf, cfg, table, metrics, genes, tfs


@pytest.mark.parametrize("failed_gene,tainted", [("B", False), ("A", True)])
def test_only_enabled_lag_input_failures_taint_tf_peak(lag_failure_fit, monkeypatch, failed_gene, tainted):
    f = lag_failure_fit
    baseline = execute(f, targets=("D",))
    frozen = copy.deepcopy(f[-1])
    model = core.TargetModel.from_bundle(f[-1]["targets"]["p0"])
    assert model.input_keys == (("A", 0), ("B", 1))
    assert 0 < model.weights[0] < 1 and 0 < model.weights[1] < 1
    actual_predict = pert._SinglesPlan.predict
    injected = []
    def fail_gene(plan, position, new):
        m = plan.model
        old = plan.old[position:position+1]
        if m.target_id == failed_gene and m.output_unit == "log1p_rna_log1p_cp10k" and len(old) == 1:
            injected.append(True)
            raise FloatingPointError("synthetic gene query failure")
        return actual_predict(plan, position, new)
    monkeypatch.setattr(pert._SinglesPlan, "predict", fail_gene)
    out = execute(f, targets=("D",))
    assert len(injected) == 19
    assert out["rna_unavailable_mask"][1:, ["D", "A", "B"].index(failed_gene)].all()
    assert int(out["atac_unavailable_mask"][:, 0].sum()) == (19 if tainted else 0)
    np.testing.assert_array_equal(out["atac_state_history"][:, 1], baseline["atac_state_history"][:, 1])
    if not tainted:
        np.testing.assert_array_equal(out["atac_state_history"][:, 0], baseline["atac_state_history"][:, 0])
        old = baseline["control_rna_states"][1:, 1]
        new = baseline["rna_state_history"][1:, 1]
        single_delta = model.singles[0].predict(new) - model.singles[0].predict(old)
        expected = np.clip(baseline["control_atac_states"][1:, 0] + model.weights[0] * single_delta, 0, 1)
        np.testing.assert_allclose(out["atac_state_history"][1:, 0], expected, rtol=1e-12, atol=1e-12)
        renormalized = np.clip(baseline["control_atac_states"][1:, 0] + single_delta, 0, 1)
        assert np.max(abs(expected - renormalized)) > 1e-5
        assert set(out["saturating_tf_peak_contributions"].query("peak_id == 'p0'").tf_gene) == {"A"}
    else:
        support = out["saturating_support_diagnostics"]
        failures = support[(support.target_id == "p0") & (support.status == "unavailable_upstream")]
        assert len(failures) == 19 and set(failures.layer) == {"tf_peak"}
        assert all(json.loads(v) == [["A", 0]] for v in failures.enabled_input_features_json)
        assert all(json.loads(v) == ["A:numerical_failure"] for v in failures.unavailable_origins_json)
        assert np.isnan(out["delta_atac"]["p0"])
    assert f[-1] == frozen


@pytest.fixture(scope="module")
def independent_peak_fit():
    r, a, pt, edges, _, cfg = fixture_data()
    tf = pd.DataFrame({"tf_gene": ["A", "A"], "peak_id": ["p0", "p1"],
                       "tf_lag": [0, 0], "weight": [999., -999.]})
    cfg["perturbation"]["min_r2_threshold"] = 1.1  # Pruned gene models cannot invalidate canonical RNA.
    cfg["perturbation"]["cell_projection"]["enabled"] = False
    table, metrics, genes = transfer.fit_peak_gene(r, a, pt, edges, cfg)
    tfs = transfer.fit_tf_peak(r, a, pt, tf, cfg)
    assert all(v["status"] == "ok" for v in tfs["target_metrics"].values())
    return r, a, pt, edges, tf, cfg, table, metrics, genes, tfs


@pytest.mark.parametrize("partial", [False, True])
def test_peak_failure_abstains_in_aggregate_and_durable_projection_disabled_audit(independent_peak_fit, monkeypatch, tmp_path, partial):
    from CausalBridge import run, io
    f = independent_peak_fit
    baseline = execute(f)
    reference = transfer._snapshot(f[0], f[1], f[2], f[5])["R"][5, 0]
    actual_predict = pert._SinglesPlan.predict
    injected = []
    def fail_peak(plan, position, new):
        m = plan.model
        old = plan.old[position:position+1]
        if m.target_id == "p0" and m.output_unit == "raw_atac" and len(old) == 1 and (
                not partial or old[0, 0] == reference):
            injected.append(True)
            raise FloatingPointError("synthetic peak query failure")
        return actual_predict(plan, position, new)
    monkeypatch.setattr(pert._SinglesPlan, "predict", fail_peak)
    out = execute(f)
    expected_count = 1 if partial else 20
    assert len(injected) == expected_count == int(out["atac_unavailable_mask"][:, 0].sum())
    assert not out["atac_unavailable_mask"][:, 1].any()
    assert np.isfinite(out["atac_state_history"]).all()  # Finite internal placeholders are not estimates.
    failed = out["atac_unavailable_mask"][:, 0]
    np.testing.assert_array_equal(out["atac_state_history"][failed, 0], out["control_atac_states"][failed, 0])
    assert np.isnan(out["delta_atac"]["p0"]) and np.isnan(out["atac_z_score"]["p0"])
    assert np.isfinite(out["delta_atac"]["p1"])
    assert out["delta_atac"]["p1"] == baseline["delta_atac"]["p1"]
    assert out["atac_z_score"]["p1"] == baseline["atac_z_score"]["p1"]
    assert abs(out["delta_atac"]["p1"]) > 1e-5
    np.testing.assert_array_equal(out["rna_state_history"], baseline["rna_state_history"])
    assert not out["projection_inputs"][0]["invalid_state"]  # ATAC-only failure does not block valid RNA.
    audit = out["saturating_support_diagnostics"].query("status == 'numerical_failure'")
    assert len(audit) == expected_count and set(audit.layer) == {"tf_peak"}
    assert set(audit.target_id) == {"p0"} and set(audit.branch) == {"0"}
    assert all(json.loads(v) == [["A", 0]] for v in audit.input_features_json)
    assert all(json.loads(v) == ["p0:numerical_failure"] for v in audit.unavailable_origins_json)
    row = out["atac_changes"].set_index("peak_id").loc["p0"]
    assert row.status == "unavailable" and np.isnan(row.delta_accessibility) and np.isnan(row.z_score)
    assert not row.is_significant and row.unavailable_bin_count == expected_count
    assert (row["chr"], row.start, row.end) == ("1", 10, 20)
    context = dict(out, cluster="fault")
    tables = run._collect_saturating_tables([context])
    cfg = copy.deepcopy(f[5]); cfg["output"] = {"dir": str(tmp_path), "save_intermediate": False, "export_details": True}
    io.save_results(dict(tables, atac_changes=out["atac_changes"]), cfg)
    directory = tmp_path
    saved = pd.read_csv(directory / "saturating_support_diagnostics.csv")
    saved_failures = saved.query("status == 'numerical_failure'")
    assert len(saved_failures) == expected_count and set(saved_failures.source_context) == {"fault"}
    assert all(json.loads(v) == ["p0:numerical_failure"] for v in saved_failures.unavailable_origins_json)
    peak_csv = pd.read_csv(directory / "atac_changes.csv").set_index("peak_id").loc["p0"]
    assert peak_csv.status == "unavailable" and np.isnan(peak_csv.delta_accessibility) and np.isnan(peak_csv.z_score)
