"""Frozen-denominator ownership and coherent continuous public model integration."""
import copy
import pickle
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from CausalBridge import atac_normalization as norm, kinetics, perturbation, run, io
from CausalBridge import saturating_transfer as transfer, saturating_hybrid as core
from CausalBridge.preprocess import preprocess_atac


def config():
    return {"atac_to_rna": {"model_mode": "saturating_hybrid", "log_normalize_rna": True},
        "preprocess": {"atac": {"binarize": True}, "atac_depth_normalization": {
            "enabled": True, "target_sum": 10000., "output_cap_expansion": .25}}}


def counts(values, csr=True):
    x = np.asarray(values, dtype=float)
    return ad.AnnData(sparse.csr_matrix(x) if csr else x,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(len(x))]),
        var=pd.DataFrame(index=[f"p{i}" for i in range(x.shape[1])]))


@pytest.mark.parametrize("csr", [True, False])
def test_full_input_denominator_survives_qc_axes_and_no_alias(csr):
    raw = counts([[1, 100], [2, 200], [0, 50]], csr)
    raw.obs["nCount_atac"] = [101., 1., 50.]
    original = raw.copy()
    prepared = norm.prepare_atac_counts(raw)
    settings = norm.validate_atac_depth_config(config())
    selected = prepared[[1, 0], :1].copy()
    selected.X = (selected.X > 0).astype(float)
    selected = norm.finalize_atac_qc(selected, prepared)
    view = norm.depth_log1p_model_view(selected, settings)
    expected = np.log1p(10000 * np.array([[2.], [1.]]) / np.array([[202.], [101.]]))
    np.testing.assert_allclose(view.X.toarray(), expected, atol=1e-12)
    np.testing.assert_array_equal(view.obs[norm.DEPTH], [202., 101.])
    np.testing.assert_array_equal(view.layers[norm.RAW_LAYER].toarray(), [[2.], [1.]])
    assert norm.atac_representation(view)["source_shape"] == [3, 2]
    assert view.uns[norm.META]["diagnostics"]["nCount_atac_mismatches"] == 1
    assert norm.META not in raw.uns and norm.RAW_LAYER not in raw.layers
    np.testing.assert_array_equal(raw.X.toarray() if csr else raw.X, original.X.toarray() if csr else original.X)
    view.layers[norm.RAW_LAYER].data[:] = 0
    np.testing.assert_array_equal(prepared.layers[norm.RAW_LAYER].toarray(), [[1,100],[2,200],[0,50]])


def test_depth_normalization_scale_invariance_and_zero_depth_rejected_for_fit():
    prepared = norm.prepare_atac_counts(counts([[1, 2], [10, 20], [0, 0]]))
    view = norm.depth_log1p_model_view(prepared, norm.validate_atac_depth_config(config()))
    np.testing.assert_allclose(view.X.toarray()[0], view.X.toarray()[1])
    np.testing.assert_array_equal(view.X.toarray()[2], [0.,0.])
    with pytest.raises(ValueError, match="zero-depth"):
        norm.atac_representation(view)
    with pytest.raises(ValueError, match="already normalized"):
        norm.depth_log1p_model_view(view, norm.validate_atac_depth_config(config()))


@pytest.mark.parametrize("value", [-1, .1, np.nan, np.inf])
def test_invalid_original_counts_rejected(value):
    with pytest.raises(ValueError, match="integer input counts"):
        norm.prepare_atac_counts(counts([[value, 1]]))


def test_sparse_duplicates_canonicalized_without_mutation_and_raw_conflicts():
    x = sparse.csr_matrix((np.array([1.,2.,4.]), np.array([0,0,1]), np.array([0,3])), shape=(1,2))
    raw = ad.AnnData(x,obs=pd.DataFrame(index=["c"]),var=pd.DataFrame(index=["p0","p1"]))
    prepared = norm.prepare_atac_counts(raw)
    assert raw.X.nnz == 3 and prepared.layers[norm.RAW_LAYER].nnz == 2
    assert prepared.obs[norm.DEPTH].iloc[0] == 7.
    raw.layers["raw"] = sparse.csr_matrix([[3.,5.]])
    with pytest.raises(ValueError, match="conflicts"):
        norm.prepare_atac_counts(raw)


def test_tampered_depth_or_declared_transform_rejected():
    view = norm.depth_log1p_model_view(norm.prepare_atac_counts(counts([[1,100],[2,50]])),
                                      norm.validate_atac_depth_config(config()))
    changed = view.copy(); changed.obs[norm.DEPTH] *= 2
    with pytest.raises(ValueError, match="modified"):
        norm.atac_representation(changed)
    changed = view.copy(); changed.X = (changed.X>0).astype(float)
    with pytest.raises(ValueError, match="differs"):
        norm.atac_representation(changed)


@pytest.mark.parametrize("section", [{"enabled":"true"}, {"target_sum":True}, {"target_sum":0},
    {"target_sum":np.inf}, {"output_cap_expansion":-.1}, {"output_cap_expansion":np.nan}, {"unknown":1}])
def test_strict_config_rejects_before_input_checks(section):
    cfg = config(); cfg["preprocess"]["atac_depth_normalization"].update(section)
    with pytest.raises(ValueError):
        io._validate_config(cfg)


@pytest.mark.parametrize("mode", ["legacy_pair_nn", "joint_gene"])
def test_enabled_wrong_mode_rejected(mode):
    cfg = config(); cfg["atac_to_rna"]["model_mode"] = mode
    with pytest.raises(ValueError, match="requires saturating_hybrid"):
        norm.validate_atac_depth_config(cfg)


def test_requires_binary_reference_and_disabled_contract_settings_ignored():
    from CausalBridge.binning_adapter import _model_config
    cfg = config(); cfg["preprocess"]["atac"]["binarize"] = False
    with pytest.raises(ValueError, match="binary reference"):
        norm.validate_atac_depth_config(cfg)
    cfg["preprocess"]["atac_depth_normalization"]["enabled"] = False
    before = _model_config(cfg), run._step5_model_config(cfg)
    cfg["preprocess"]["atac_depth_normalization"].update(target_sum=1000000., output_cap_expansion=2.)
    assert before == (_model_config(cfg),run._step5_model_config(cfg))
    legacy = preprocess_atac(counts([[2,3],[4,0]]),min_cells_per_peak=0,min_peaks_per_cell=0,binarize=False)
    np.testing.assert_array_equal(legacy.X.toarray(), [[2,3],[4,0]])


def test_real_reference_unchanged_and_qc_once(monkeypatch):
    from test_reference_clustering import inputs
    from CausalBridge import reference_clustering as reference
    rna, atac, cfg = inputs()
    atac.X = atac.X * 3
    cfg["atac_to_rna"] = {"model_mode":"saturating_hybrid"}
    before, binary = reference.build_reference_clustering(rna,atac,["G1"],cfg)
    enabled = copy.deepcopy(cfg)
    enabled["preprocess"]["atac_depth_normalization"] = config()["preprocess"]["atac_depth_normalization"]
    calls=[]; actual=reference.preprocess_atac
    def once(*args,**kwargs):
        calls.append(1); return actual(*args,**kwargs)
    monkeypatch.setattr(reference,"preprocess_atac",once)
    after, qc = reference.build_reference_clustering(rna,atac,["G1"],enabled)
    assert calls == [1]
    assert before.uns["clustering_reference"] == after.uns["clustering_reference"]
    np.testing.assert_array_equal(before.obs.leiden,after.obs.leiden)
    np.testing.assert_array_equal(before.obsp["connectivities"].toarray(),after.obsp["connectivities"].toarray())
    np.testing.assert_array_equal(binary.X.toarray(),qc.X.toarray())
    model=norm.depth_log1p_model_view(qc,norm.validate_atac_depth_config(enabled))
    assert model.obs_names.equals(qc.obs_names) and model.var_names.equals(qc.var_names)
    assert model.X.max()>1 and norm.RAW_LAYER in model.layers


def continuous_fixture(tmp_path, smooth="hard"):
    from test_saturating_pipeline import public_fixture
    rna, old_atac, pt, edges, cfg = public_fixture(smooth)
    raw = ad.AnnData(sparse.hstack([old_atac.X, sparse.csr_matrix(np.full((96,1),100.))],format="csr"),
        obs=old_atac.obs.copy(),var=pd.DataFrame({"chr":["chr1","chr1"],"start":[10,30],"end":[20,40]},
                                               index=["chr1:10-20","background"]))
    cfg["preprocess"] = copy.deepcopy(config()["preprocess"])
    cfg["output"].update(dir=str(tmp_path),checkpoint_dir=str(tmp_path/"checkpoints"),cache_dir=str(tmp_path/"cache"))
    prepared=norm.prepare_atac_counts(raw)
    qc=preprocess_atac(prepared,min_cells_per_peak=0,min_peaks_per_cell=0,binarize=True)
    # A discarded full-input peak still contributes to the frozen denominator.
    view=norm.depth_log1p_model_view(norm.finalize_atac_qc(qc[:,:1].copy(),prepared),norm.validate_atac_depth_config(cfg))
    return rna,view,pt,edges,cfg


@pytest.mark.parametrize("smooth", ["hard", "gaussian"])
def test_real_public_continuous_fit_units_caps_and_raw_interventions(tmp_path,monkeypatch,smooth):
    rna,atac,pt,edges,cfg=continuous_fixture(tmp_path,smooth)
    monkeypatch.setattr(kinetics,"_scan_peaks_with_jaspar",lambda *a,**k:{"chr1:10-20":["TF"]})
    assert not hasattr(kinetics, "_scale_atac")
    table,genes=kinetics.fit_atac_to_rna(rna,atac,pt,edges,cfg)
    tfs={}
    screen=kinetics.fit_rna_to_atac(rna,atac,pt,edges,cfg,tf_model_registry=tfs,
        motif_resources={"motif_db":{"synthetic":{"tf_name":"TF"}},"tf_database":frozenset({"TF"}),"resource_cache_dir":str(tmp_path)})
    assert genes["schema"]==transfer.GENE_SCHEMA_V2 and tfs["schema"]==transfer.TF_SCHEMA_V2
    assert genes["input_unit"]==norm.UNIT and tfs["output_unit"]==norm.UNIT
    assert genes["targets"]["G"]["schema"]==core.CONTINUOUS_SCHEMA
    transfer.validate_registry(genes,expected_schema=transfer.GENE_SCHEMA_V2)
    transfer.validate_registry(tfs,expected_schema=transfer.TF_SCHEMA_V2)
    snapshot=transfer._snapshot(rna,atac,pt,cfg)
    np.testing.assert_allclose(snapshot["A"],snapshot["operator"]@atac.X.toarray())
    tf=core.TargetModel.from_bundle(tfs["targets"]["chr1:10-20"])
    assert tf.bounds==norm.atac_output_bounds(snapshot["A"][:,0],snapshot["context"]["atac_representation"])
    assert tf.bounds[1]>1 and tf.bounds[1]<=np.log1p(10000.)
    frozen=copy.deepcopy((genes,tfs));kwargs=dict(transfer_models=genes,tf_model_registry=tfs)
    gene=perturbation.propagate_perturbation(["TF"],rna,atac,pt,edges,table,screen,cfg,**kwargs)
    assert not gene["perturbation_results"].empty
    assert gene["atac_state_history"].max()>1
    peak=perturbation.perturb_peak("chr1:10-20",rna,atac,pt,edges,table,screen,cfg,**kwargs)
    expected=snapshot["operator"]@np.log1p(np.expm1(atac.X.toarray())*.5)
    np.testing.assert_allclose(peak["atac_state_history"],expected,rtol=1e-12,atol=1e-12)
    assert not np.allclose(expected,snapshot["A"]*.5)
    assert set(peak["atac_changes"].atac_state_unit)=={norm.UNIT}
    for amount in (-1.,0.):
        zero=copy.deepcopy(cfg);zero["perturbation"]["peak_ko"]["strength"]=amount
        out=perturbation.perturb_peak("chr1:10-20",rna,atac,pt,edges,table,screen,zero,**kwargs)
        if amount==0:
            np.testing.assert_array_equal(out["atac_state_history"],snapshot["A"])
        else:
            np.testing.assert_array_equal(out["atac_state_history"],np.zeros_like(snapshot["A"]))
    absolute=copy.deepcopy(cfg);absolute["perturbation"]["peak_ko"].update(mode="absolute",strength=2.)
    out=perturbation.perturb_peak("chr1:10-20",rna,atac,pt,edges,table,screen,absolute,**kwargs)
    np.testing.assert_allclose(out["atac_state_history"],snapshot["operator"]@np.full((96,1),2.),atol=1e-12)
    high=copy.deepcopy(cfg);high["perturbation"]["peak_ko"]["strength"]=10000.
    clipped=perturbation.perturb_peak("chr1:10-20",rna,atac,pt,edges,table,screen,high,**kwargs)
    assert clipped["saturating_context_metadata"]["direct_peak_physical_clip_cells"]>0
    assert clipped["atac_state_history"].max()<=np.log1p(10000.)
    zero=copy.deepcopy(cfg);zero["perturbation"]["ko_strength"]=0
    out=perturbation.propagate_perturbation(["TF"],rna,atac,pt,edges,table,screen,zero,**kwargs)
    np.testing.assert_array_equal(out["atac_state_history"],snapshot["A"])
    np.testing.assert_array_equal(out["rna_state_history"],snapshot["R"])
    assert (genes,tfs)==frozen
    assert core.TargetModel.from_bundle(tf.to_bundle()).to_bundle()==tf.to_bundle()
    changed=copy.deepcopy(cfg);changed["preprocess"]["atac_depth_normalization"]["target_sum"]*=2
    with pytest.raises(ValueError,match="settings differ"):
        perturbation.propagate_perturbation(["TF"],rna,atac,pt,edges,table,screen,changed,**kwargs)
    binary=copy.deepcopy(genes);binary["schema"]=binary["schema_version"]=transfer.GENE_SCHEMA
    binary["fingerprint"]=core._digest({k:v for k,v in binary.items() if k!="fingerprint"})
    with pytest.raises(ValueError,match="schema"):
        transfer.validate_registry(binary)


def test_namespace_idempotence_and_continuous_fit_contract(tmp_path):
    cfg=config();cfg["output"]={"dir":str(tmp_path/"with spaces"),"checkpoint_dir":str(tmp_path/"custom cp")}
    effective=run._joint_runtime_config(cfg)
    assert effective["output"]["dir"] == cfg["output"]["dir"]
    assert run._joint_runtime_config(effective)==effective
    assert run._checkpoint_dir(effective) == Path(cfg["output"]["checkpoint_dir"])
    signature=run._step5_model_config(cfg)
    changed=copy.deepcopy(cfg);changed["preprocess"]["atac_depth_normalization"]["output_cap_expansion"]*=2
    assert run._step5_model_config(changed)!=signature
    gtf=tmp_path/"genes.gtf";gtf.write_text('1\tr\tgene\t1\t2\t.\t+\t.\tgene_id "G"; gene_name "G";\n')
    cfg["input"]={"gene_identity_mode":"gtf_gene_id","annotation_gtf":str(gtf)}
    effective=run._joint_runtime_config(cfg)
    assert effective["output"]["dir"] == cfg["output"]["dir"]
    assert effective["output"].get("checkpoint_dir") == cfg["output"].get("checkpoint_dir")
    assert run._joint_runtime_config(effective)==effective


def test_frozen_pre_continuous_v1_bundle_exact_roundtrip():
    # Captured from the completed Phase-3 binary synthetic smoke before this feature.
    payload=json.loads((Path(__file__).parent/"saturating_v1_target_fixture.json").read_text())
    model=core.TargetModel.from_bundle(payload)
    assert model.to_bundle()==payload
    assert model.fingerprint=="329275aa0bd26fc8f604f5999e689fce15aa856654880bc9a9f4a96e1b4a9b40"
    changed=copy.deepcopy(payload);changed["schema"]=core.CONTINUOUS_SCHEMA
    changed["fingerprint"]=core._digest({k:v for k,v in changed.items() if k!="fingerprint"})
    with pytest.raises(ValueError,match="schema/unit"):
        core.TargetModel.from_bundle(changed)


def test_input_provenance_distinguishes_depth_source_but_cap_is_fit_only():
    raw=counts([[1,100],[2,200]])
    settings=norm.validate_atac_depth_config(config())
    first=norm.depth_log1p_model_view(norm.prepare_atac_counts(raw),settings)
    second=norm.depth_log1p_model_view(norm.prepare_atac_counts(counts([[2,200],[4,400]])),settings)
    np.testing.assert_array_equal(first.X.toarray(),second.X.toarray())
    assert norm.atac_input_contract(first)!=norm.atac_input_contract(second)
    settings["output_cap_expansion"] = .5
    different_cap=norm.depth_log1p_model_view(norm.prepare_atac_counts(raw),settings)
    assert norm.atac_input_contract(first)==norm.atac_input_contract(different_cap)


@pytest.mark.parametrize("smooth",["hard","gaussian"])
def test_continuous_subset_warm_v2_and_exports(tmp_path,monkeypatch,smooth):
    rna,atac,pt,edges,cfg=continuous_fixture(tmp_path,smooth)
    cfg["pseudotime"].update(n_neighbors=10,method="dpt")
    cfg["granger"]={"reference_bins":24,"min_composite_score":.3}
    resources={"motif_db":{"synthetic":{"tf_name":"TF","pwm":np.full((4,6),.25)}},
        "tf_database":frozenset({"TF"}),"resource_cache_dir":str(tmp_path/"cache")}
    monkeypatch.setattr(run,"infer_pseudotime",lambda *a,**k:pt.copy(deep=True))
    monkeypatch.setattr(run,"granger_test",lambda *a,**k:edges.assign(is_causal=True))
    monkeypatch.setattr(run,"_prepare_adaptive_motif_resources",lambda *a,**k:resources)
    monkeypatch.setattr(kinetics,"_scan_peaks_with_jaspar",lambda *a,**k:{"chr1:10-20":["TF"]})
    first=run._run_modeling_for_subset(rna,atac,["TF"],{"TF":"TF"},cfg,suffix="continuous")
    directory=run._checkpoint_dir(run._joint_runtime_config(cfg),"continuous")
    assert directory == tmp_path / "checkpoints" / "continuous"
    saved={stage:(directory/f"{stage}.pkl").read_bytes() for stage in
           ("step4_granger","step5_atac_to_rna","step6_rna_to_atac")}
    step5=pickle.loads(saved["step5_atac_to_rna"])
    step6=pickle.loads(saved["step6_rna_to_atac"])
    assert step5["transfer_models"]["schema"]==transfer.GENE_SCHEMA_V2
    assert step6["tf_model_registry"]["schema"]==transfer.TF_SCHEMA_V2
    with pytest.raises(RuntimeError,match="incompatible"):
        run._validate_step5_checkpoint(step5,"saturating_hybrid")  # Binary expectation must not consume v2.
    warm_cfg = copy.deepcopy(cfg)
    warm_cfg["atac_to_rna"].setdefault("saturating_hybrid", {})["workers"] = 4
    warm=run._run_modeling_for_subset(rna,atac,["TF"],{"TF":"TF"},warm_cfg,suffix="continuous")
    pd.testing.assert_frame_equal(warm["perturbation_results"],first["perturbation_results"])
    assert all((directory/f"{stage}.pkl").read_bytes()==data for stage,data in saved.items())
    io.save_results(first,cfg)
    output=tmp_path
    assert (output/"saturating_contexts.json").is_file()
    context=json.loads((output/"saturating_contexts.json").read_text())[0]
    assert context["atac_representation"]["unit"]==norm.UNIT
    tf_metrics=pd.read_csv(output/"saturating_tf_metrics.csv")
    assert set(tf_metrics.output_unit)=={norm.UNIT}


def continuous_refit(fitted):
    """Use a discarded background peak to keep the full raw depth exactly 100."""
    rna, old, pt, edges, screen, cfg = fitted[:6]
    cfg = copy.deepcopy(cfg)
    cfg["preprocess"] = copy.deepcopy(config()["preprocess"])
    background = 100 - np.asarray(old.X.sum(axis=1)).ravel()
    raw = ad.AnnData(sparse.hstack([old.X, sparse.csr_matrix(background[:, None])], format="csr"),
        obs=old.obs.copy(), var=pd.concat([old.var, pd.DataFrame(index=["discarded_background"])]))
    prepared = norm.prepare_atac_counts(raw)
    qc = preprocess_atac(prepared, min_cells_per_peak=0, min_peaks_per_cell=0, binarize=True)
    atac = norm.depth_log1p_model_view(norm.finalize_atac_qc(qc[:, :old.n_vars].copy(), prepared), norm.validate_atac_depth_config(cfg))
    np.testing.assert_array_equal(atac.obs[norm.DEPTH], np.full(old.n_obs, 100.))
    assert atac.X.max() > 1
    table, metrics, genes = transfer.fit_peak_gene(rna, atac, pt, edges, cfg)
    tfs = transfer.fit_tf_peak(rna, atac, pt, screen, cfg)
    return rna, atac, pt, edges, screen, cfg, table, metrics, genes, tfs


@pytest.fixture(scope="module")
def continuous_group_fit():
    from test_saturating_perturbation import grouped_fit
    return continuous_refit(grouped_fit.__wrapped__())


def test_continuous_real_group_codec_and_fixed_route(continuous_group_fit):
    from test_saturating_perturbation import execute
    fitted = continuous_group_fit
    model = core.TargetModel.from_bundle(fitted[-1]["targets"]["p0"])
    assert model.groups and model.output_unit == norm.UNIT
    assert model.to_bundle()["schema"] == core.CONTINUOUS_SCHEMA
    bundle = model.to_bundle()
    assert all(single["output_unit"] == norm.UNIT and single["schema"] == core.CONTINUOUS_SINGLE_SCHEMA
               for single in bundle["singles"])
    assert all(group["output_unit"] == norm.UNIT and group["schema"] == core.CONTINUOUS_GROUP_SCHEMA
               for group in bundle["groups"])
    assert core.TargetModel.from_bundle(model.to_bundle()).to_bundle() == model.to_bundle()
    result = execute(fitted, targets=("A", "D"))
    assert result["saturating_context_metadata"]["route_passes"] == 2
    assert result["saturating_context_metadata"]["locked_groups"]
    support = result["saturating_support_diagnostics"]
    assert set(support.loc[support.relation_type == transfer.TF_TYPE, "route"]) == {"group"}
    assert result["atac_state_history"].max() > 1
    assert result["atac_state_history"].max() <= model.bounds[1]
    zero = copy.deepcopy(fitted[5]); zero["perturbation"]["ko_strength"] = 0.
    noop = execute(fitted, zero, targets=("A", "D"))
    np.testing.assert_array_equal(noop["atac_state_history"], noop["control_atac_states"])


@pytest.fixture(scope="module")
def continuous_lag_failure_fit():
    from test_saturating_perturbation import lag_failure_fit
    return continuous_refit(lag_failure_fit.__wrapped__())


@pytest.mark.parametrize("failed_gene,tainted", [("B", False), ("A", True)])
def test_continuous_only_enabled_lag_failures_taint_and_abstain(
        continuous_lag_failure_fit, monkeypatch, failed_gene, tainted, tmp_path):
    from types import SimpleNamespace
    from CausalBridge import saturating_perturbation as simulation
    from test_saturating_perturbation import execute
    fitted = continuous_lag_failure_fit
    baseline = execute(fitted, targets=("D",))
    model = core.TargetModel.from_bundle(fitted[-1]["targets"]["p0"])
    frozen = copy.deepcopy(fitted[-1])
    proxy = SimpleNamespace(**core.__dict__)
    def fail_gene(m, old, new, **kwargs):
        if m.target_id == failed_gene and m.output_unit == "log1p_rna_log1p_cp10k" and len(old) == 1:
            raise FloatingPointError("synthetic continuous query failure")
        return core.predict_delta(m, old, new, **kwargs)
    proxy.predict_delta = fail_gene
    monkeypatch.setattr(simulation, "core", proxy)
    result = execute(fitted, targets=("D",))
    assert result["rna_unavailable_mask"][1:, ["D", "A", "B"].index(failed_gene)].all()
    assert int(result["atac_unavailable_mask"][:, 0].sum()) == (19 if tainted else 0)
    np.testing.assert_array_equal(result["atac_state_history"][:, 1], baseline["atac_state_history"][:, 1])
    if tainted:
        assert np.isnan(result["delta_atac"]["p0"])
        failures = result["saturating_support_diagnostics"].query(
            "target_id == 'p0' and status == 'unavailable_upstream'")
        assert len(failures) == 19
        assert all(json.loads(v) == [["A", 0]] for v in failures.enabled_input_features_json)
        assert all(json.loads(v) == ["A:numerical_failure"] for v in failures.unavailable_origins_json)
    else:
        single = model.singles[0]
        delta = single.predict(baseline["rna_state_history"][1:, 1]) - single.predict(baseline["control_rna_states"][1:, 1])
        expected = np.clip(baseline["control_atac_states"][1:, 0] + model.weights[0]*delta, *model.bounds)
        np.testing.assert_allclose(result["atac_state_history"][1:, 0], expected, rtol=1e-12, atol=1e-12)
        assert set(result["saturating_tf_peak_contributions"].query("peak_id == 'p0'").tf_gene) == {"A"}
    assert fitted[-1] == frozen
    cfg = copy.deepcopy(fitted[5]); cfg["output"] = {"dir": str(tmp_path), "save_intermediate": False}
    io.save_results(result, cfg)
    exported = pd.read_csv(tmp_path / "saturating_support_diagnostics.csv")
    assert "unavailable_origins_json" in exported
    assert (exported.status == "numerical_failure").any()


@pytest.mark.parametrize("field,value", [
    ("source_counts_sha256", "z"*64), ("source_shape", [1, 1]),
    ("denominator_kind", "post_qc_depth"), ("retained_peak_axis_sha256", "0"*64),
    ("unexpected", "extra")])
def test_resigned_v2_representation_corruption_rejected(continuous_group_fit, field, value):
    registry = copy.deepcopy(continuous_group_fit[-1])
    registry["context"]["atac_representation"][field] = value
    registry["context"]["fingerprint"] = core._digest({k:v for k,v in registry["context"].items() if k != "fingerprint"})
    registry["fingerprint"] = core._digest({k:v for k,v in registry.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="continuous ATAC"):
        transfer.validate_registry(registry)


def test_resigned_v2_root_audit_rejects_normalized_signal(continuous_group_fit):
    registry = copy.deepcopy(continuous_group_fit[-1])
    registry["root_atac_reference"] = [2.]
    registry["fingerprint"] = core._digest({k:v for k,v in registry.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="fields mismatch"):
        transfer.validate_registry(registry)


@pytest.mark.parametrize("csr", [True, False])
def test_source_counts_halved_before_normalization_rejected(csr):
    prepared = norm.prepare_atac_counts(counts([[2, 100], [4, 200]], csr))
    prepared.layers[norm.RAW_LAYER] *= .5
    with pytest.raises(ValueError, match="source counts were modified"):
        norm.depth_log1p_model_view(prepared, norm.validate_atac_depth_config(config()))
    with pytest.raises(ValueError, match="source counts were modified"):
        preprocess_atac(prepared, min_cells_per_peak=0, min_peaks_per_cell=0)


def test_doubled_subset_depth_with_regenerated_row_proofs_rejected():
    prepared = norm.prepare_atac_counts(counts([[2, 100], [4, 200]]))
    selected = prepared[:, :1].copy()
    selected.obs[norm.DEPTH] *= 2
    selected.obs[norm.ROW_PROOF] = [norm._row_proof(c, d, selected.uns[norm.META]["source_counts_sha256"])
        for c, d in zip(selected.obs_names, selected.obs[norm.DEPTH])]
    with pytest.raises(ValueError, match="complete full-input"):
        norm.depth_log1p_model_view(selected, norm.validate_atac_depth_config(config()))
    with pytest.raises(ValueError, match="differs from full-input"):
        norm.finalize_atac_qc(selected, prepared)
    selected = norm.finalize_atac_qc(prepared[:, :1].copy(), prepared)
    selected.obs[norm.DEPTH] *= 2
    selected.obs[norm.ROW_PROOF] = [norm._row_proof(c, d, selected.uns[norm.META]["source_counts_sha256"])
        for c, d in zip(selected.obs_names, selected.obs[norm.DEPTH])]
    with pytest.raises(ValueError, match="source evidence"):
        norm.depth_log1p_model_view(selected, norm.validate_atac_depth_config(config()))


def test_full_depth_hash_replacement_rejected_at_every_boundary(continuous_group_fit):
    prepared = norm.prepare_atac_counts(counts([[2, 100], [4, 200]]))
    changed = prepared.copy(); changed.uns[norm.META]["source_depth_sha256"] = "0"*64
    with pytest.raises(ValueError, match="depth source evidence"):
        norm.depth_log1p_model_view(changed, norm.validate_atac_depth_config(config()))
    qc = preprocess_atac(prepared, min_cells_per_peak=0, min_peaks_per_cell=0)
    view = norm.depth_log1p_model_view(qc, norm.validate_atac_depth_config(config()))
    view.uns[norm.META]["source_depth_sha256"] = "0"*64
    with pytest.raises(ValueError, match="depth source evidence"):
        norm.atac_representation(view)
    registry = copy.deepcopy(continuous_group_fit[-1])
    registry["context"]["atac_representation"]["source_depth_sha256"] = "0"*64
    registry["context"]["fingerprint"] = core._digest({k:v for k,v in registry["context"].items() if k != "fingerprint"})
    registry["fingerprint"] = core._digest({k:v for k,v in registry.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="depth source evidence"):
        transfer.validate_registry(registry)


@pytest.mark.parametrize("csr", [True, False])
def test_checked_qc_cell_subsets_roundtrip_and_count_integrity(tmp_path, csr):
    prepared = norm.prepare_atac_counts(counts([[2, 100, 0], [4, 200, 0], [0, 0, 0], [6, 0, 10]], csr))
    qc = preprocess_atac(prepared, min_cells_per_peak=2, min_peaks_per_cell=1)
    assert qc.shape == (3, 2)
    np.testing.assert_array_equal(qc.obs[norm.DEPTH], [102., 204., 16.])
    assert list(qc.uns[norm.META]["source_full_depth"]) == [102., 204., 0., 16.]
    selected = qc[[2, 0]].copy()
    model = norm.depth_log1p_model_view(selected, norm.validate_atac_depth_config(config()))
    before = norm.atac_representation(model)
    assert before["retained_counts_sha256"] == norm._matrix_hash(model.layers[norm.RAW_LAYER])
    assert before["source_shape"] == [4, 3]
    assert before["source_full_depth"] == [102., 204., 0., 16.]
    model.write_h5ad(tmp_path / "checked.h5ad")
    restored = ad.read_h5ad(tmp_path / "checked.h5ad")
    assert norm.atac_representation(restored) == before
    cluster = restored[[1]].copy()
    assert norm.atac_representation(cluster)["source_depth_sha256"] == before["source_depth_sha256"]
    tampered = selected.copy(); tampered.layers[norm.RAW_LAYER] *= .5
    with pytest.raises(ValueError, match="integer input counts|count layer was modified"):
        norm.depth_log1p_model_view(tampered, norm.validate_atac_depth_config(config()))
    # Preserve integer counts and row sums; the row digest still catches edits.
    tampered = qc.copy(); matrix = tampered.layers[norm.RAW_LAYER].toarray()
    matrix[0] = [4., 98.]; tampered.layers[norm.RAW_LAYER] = sparse.csr_matrix(matrix)
    with pytest.raises(ValueError, match="count layer was modified"):
        norm.depth_log1p_model_view(tampered, norm.validate_atac_depth_config(config()))
    with pytest.raises(ValueError, match="peak axis changed"):
        norm.depth_log1p_model_view(qc[:, :1].copy(), norm.validate_atac_depth_config(config()))
    # Legitimate peak selection uses the original source, never unchecked data.
    reduced = norm.finalize_atac_qc(qc[[2, 0], :1].copy(), prepared)
    norm.atac_representation(norm.depth_log1p_model_view(reduced, norm.validate_atac_depth_config(config())))


def test_duplicate_sparse_source_checked_against_canonical_subset():
    matrix = sparse.csr_matrix((np.array([2., 4., 100.]), np.array([0, 0, 1]), np.array([0, 3])), shape=(1, 2))
    raw = ad.AnnData(matrix, obs=pd.DataFrame(index=["c"]), var=pd.DataFrame(index=["p", "background"]))
    prepared = norm.prepare_atac_counts(raw)
    qc = norm.finalize_atac_qc(prepared[:, :1].copy(), prepared)
    model = norm.depth_log1p_model_view(qc, norm.validate_atac_depth_config(config()))
    np.testing.assert_allclose(model.X.toarray(), [[np.log1p(10000*6/106)]])
    assert raw.X.nnz == 3
    assert norm.atac_representation(model)["source_full_depth"] == [106.]
