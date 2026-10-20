"""Wiring contracts; stub adapters deliberately isolate this lane's dispatch."""
import copy
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from CausalBridge import kinetics, perturbation, run, io, binning_adapter
from CausalBridge._wt_aggregation import validate_model_mode
from CausalBridge.saturating_hybrid import HybridConfig
from CausalBridge.saturating_transfer import saturating_model_settings, response_contract, response_settings, GENE_TYPE, TF_TYPE


@pytest.fixture
def adapters(monkeypatch):
    transfer = ModuleType("CausalBridge.saturating_transfer")
    transfer.validate_saturating_config = lambda cfg: HybridConfig.from_dict(cfg.get("atac_to_rna", {}).get("saturating_hybrid", {}))
    transfer.saturating_model_settings = saturating_model_settings
    transfer.response_contract = response_contract
    transfer.response_settings = response_settings
    transfer.GENE_TYPE = GENE_TYPE
    transfer.TF_TYPE = TF_TYPE
    def validate(registry, *, expected_type=None, expected_schema=None):
        if not isinstance(registry, dict) or registry.get("type") != expected_type or registry.get("schema") != expected_schema:
            raise ValueError("incompatible saturating registry; rerun Step 5/6")
        return registry
    transfer.validate_registry = validate
    def validate_configured(registry, kind, cfg):
        from CausalBridge.atac_normalization import expected_registry_schema
        return validate(registry, expected_type=kind,
            expected_schema=expected_registry_schema(kind, cfg))
    transfer.validate_configured_registry = validate_configured
    transfer.fit_tf_peak = lambda *a, **k: registry("tf_peak")
    transfer._snapshot = lambda r, a, *args: {"R": r.X, "A": a.X}
    engine = ModuleType("CausalBridge.saturating_perturbation")
    projection = ModuleType("CausalBridge.saturating_projection")
    for module in (transfer, engine, projection):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return transfer, engine, projection


def config():
    return {"atac_to_rna": {"model_mode": "saturating_hybrid", "joint": {"workers": 24}},
            "rna_to_atac": {"time_lag": False, "tf_expression_filter": False},
            "pseudotime": {"n_bins": 16}, "perturbation": {"ko_strength": .5},
            "output": {"dir": "results", "checkpoint_dir": "custom", "cache_dir": "cache", "save_intermediate": False}}


def registry(layer):
    return {"type": "saturating_hybrid_" + layer,
            "schema": "hybrid_saturation_tf_peak_local_registry_v3" if layer == "tf_peak" else "saturating_" + layer + "_registry_v1"}


def test_mode_and_invalid_mode():
    assert validate_model_mode(config()) == 'saturating_hybrid'
    assert validate_model_mode({}) == 'saturating_hybrid'
    omitted = {"atac_to_rna": {"response_model": "nn", "nn_response": {"epochs": 17}},
               "rna_to_atac": {"response_model": "spline"}}
    assert run._step5_model_config({})["model_mode"] == "saturating_hybrid"
    assert run._step5_model_config(omitted)["nn_response"]["epochs"] == 17
    assert run._step6_model_config(omitted)["response_model"] == "spline"
    with pytest.raises(ValueError,match='removed'):
        validate_model_mode({"atac_to_rna": {"model_mode": "joint_gene"}})
    with pytest.raises(ValueError):
        validate_model_mode({"atac_to_rna": {"model_mode": "foo"}})


@pytest.mark.parametrize("mode", ["legacy_pair_nn"])
def test_no_hybrid_registry_in_old_mode(mode):
    cfg = config(); cfg["atac_to_rna"]["model_mode"] = mode
    with pytest.raises(ValueError, match="removed"):
        perturbation.propagate_perturbation(["TF"], None, None, None, None, None, None,
            cfg, tf_model_registry=registry("tf_peak"))


def test_raw_step5_dispatch(adapters, monkeypatch):
    transfer, _, _ = adapters
    raw_rna, raw_atac, pt, edges = object(), object(), {"bin": [0]}, object()
    root = np.array([.2])
    calls = []
    def fit(*args, **kwargs):
        calls.append((args, kwargs))
        return pd.DataFrame({"alpha": [.7]}), pd.DataFrame(), registry("peak_gene")
    transfer.fit_peak_gene = fit
    frame, models = kinetics.fit_atac_to_rna(raw_rna, raw_atac, pt, edges, config())
    assert calls[0][0][:4] == (raw_rna, raw_atac, pt, edges)
    assert models == registry("peak_gene")
    assert not frame.attrs


def test_tf_registry_required_before_scanning(adapters, monkeypatch):
    monkeypatch.setattr(kinetics, "_load_motif_database", lambda *a, **k: pytest.fail("scan"))
    with pytest.raises(ValueError, match="caller-owned"):
        kinetics.fit_rna_to_atac(None, None, None, None, config())


def test_tf_registry_transactional_and_no_attrs(adapters, monkeypatch):
    transfer, _, _ = adapters
    rna = SimpleNamespace(X=np.ones((16, 1)), n_obs=16, var_names=pd.Index(["TF"]), uns={})
    atac = SimpleNamespace(X=np.ones((16, 1)), n_obs=16, var_names=pd.Index(["P"]))
    pt = pd.DataFrame({"pseudotime": np.arange(16), "bin": np.arange(16)})
    monkeypatch.setattr(kinetics, "_adaptive_rebin", lambda *a, **k: (np.arange(16), 16))
    monkeypatch.setattr(kinetics, "_bin_matrix", lambda matrix, *a, **k: matrix)
    monkeypatch.setattr(kinetics, "_load_motif_database", lambda *a, **k: None)
    monkeypatch.setattr(kinetics, "_assign_tfs_fallback", lambda *a, **k: {"P": ["TF"]})
    monkeypatch.setattr(kinetics, "_fit_tf_peak_regression", lambda *a, **k: [
        {"peak_id": "P", "tf_gene": "TF", "tf_lag": 0, "weight": .9, "p_value": .01}])
    caller = {"previous": True}
    def fail(*a, **k):
        return {"type": "nn"}
    transfer.fit_tf_peak = fail
    with pytest.raises(ValueError):
        kinetics.fit_rna_to_atac(rna, atac, pt, pd.DataFrame(), config(), tf_model_registry=caller)
    assert caller == {"previous": True}
    def fit(*args, **kwargs):
        assert args[0] is rna and args[1] is atac
        assert set(args[3].screening_status) == {"local_pearson_routes"}
        return registry("tf_peak")
    transfer.fit_tf_peak = fit
    result = kinetics.fit_rna_to_atac(rna, atac, pt, pd.DataFrame(), config(), tf_model_registry=caller)
    assert caller == registry("tf_peak") and not result.attrs


@pytest.mark.parametrize("peak", [False, True])
def test_half_ko_dispatch_never_linear(adapters, peak):
    transfer, engine, _ = adapters
    seen = []
    def simulate(*args, **kwargs):
        # The engine now owns the single strict gate for direct and public calls.
        transfer.validate_configured_registry(kwargs["transfer_models"], GENE_TYPE, args[7])
        transfer.validate_configured_registry(kwargs["tf_model_registry"], TF_TYPE, args[7])
        seen.append((args, kwargs))
        return {"raw_ko_strength": args[7]["perturbation"]["ko_strength"]}
    engine.propagate_saturating = simulate
    engine.perturb_peak_saturating = simulate
    function = perturbation.perturb_peak if peak else perturbation.propagate_perturbation
    result = function("P" if peak else ["TF"], None, None, None, None, None,
        pd.DataFrame({"weight": [1e9]}), config(), transfer_models=registry("peak_gene"),
        tf_model_registry=registry("tf_peak"), cluster_label="cluster_1")
    assert result["raw_ko_strength"] == .5
    assert seen[0][1]["cluster_label"] == "cluster_1"
    with pytest.raises(ValueError, match="rerun"):
        function("P" if peak else ["TF"], None, None, None, None, None, None, config(),
                 transfer_models=registry("peak_gene"))


def test_projection_entry_raw_decoder(adapters):
    _, _, projection = adapters
    payload = {"effect_schema_version": "bin_saturating_effect_v1"}
    expected = (pd.DataFrame(), np.array([[100.]]), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    projection.project_saturating_inputs = lambda *args: expected
    projection.cell_l0_from_saturating_payload = lambda p: (["c"], ("G",), expected[1])
    assert run._project_bin_forward_inputs(payload, config(), "c", "TF") is expected
    assert run._cell_l0_from_saturating_payload(payload)[2][0, 0] == 100.


def test_namespace_and_inactive_settings(adapters):
    cfg = config(); before = copy.deepcopy(cfg)
    effective = run._joint_runtime_config(cfg)
    assert effective["output"]["dir"] == "results"
    assert effective["output"]["checkpoint_dir"] == "custom"
    assert run._joint_runtime_config(effective) == effective and cfg == before
    fit = run._step5_model_config(cfg)
    cfg["atac_to_rna"]["joint"]["workers"] = 999
    cfg["atac_to_rna"]["root_atac_floor"] = .99
    assert run._step5_model_config(cfg) == fit
    cfg["atac_to_rna"]["model_mode"] = "legacy_pair_nn"
    with pytest.raises(ValueError, match="removed"):
        run._step5_model_config(cfg)
    with pytest.raises(ValueError, match="removed"):
        binning_adapter._model_config(cfg)


@pytest.mark.parametrize("field,value", [("workers", 0), ("max_nfev", True),
    ("max_nfev", 3001), ("det_starts", 4), ("ridge", np.inf), ("rna_cap_lambda", np.nan),
    ("budget_exhausted_policy", "unknown")])
def test_strict_new_config(adapters, field, value):
    cfg = config(); cfg["atac_to_rna"]["saturating_hybrid"] = {field: value}
    with pytest.raises(ValueError):
        run._step5_model_config(cfg)


def test_defaults_and_user_selection_preserved(tmp_path):
    rna = tmp_path / "rna"; atac = tmp_path / "atac"
    rna.touch(); atac.touch()
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"input": {"rna_h5ad": str(rna), "atac_h5ad": str(atac)}}))
    cfg = io.load_config(path)
    assert cfg["atac_to_rna"]["model_mode"] == "saturating_hybrid"
    assert cfg["atac_to_rna"]["saturating_hybrid"] == HybridConfig().to_dict()
    # User config is editable; validate explicit choices using isolated inputs.
    for mode in ("saturating_hybrid",):
        path.write_text(yaml.safe_dump({"input": {"rna_h5ad": str(rna), "atac_h5ad": str(atac)},
            "atac_to_rna": {"model_mode": mode}}))
        selected = io.load_config(path)
        assert selected["atac_to_rna"]["model_mode"] == mode
        assert selected["atac_to_rna"]["saturating_hybrid"] == HybridConfig().to_dict()


def test_step4_sources_mode_independent():
    assert run._stage_numerical_signatures("step4_granger", "saturating_hybrid") == run._stage_numerical_signatures("step4_granger")


def test_old_checkpoint_cannot_pair_with_new_gene_mode(adapters):
    with pytest.raises(RuntimeError, match="使用新的checkpoint_dir"):
        run._validate_step5_checkpoint({"transfer_models": {"type": "nn"}}, "saturating_hybrid")


def test_summary_io_isolated_and_no_model_exports(tmp_path):
    cfg = config(); cfg["output"]["dir"] = str(tmp_path)
    frame = pd.DataFrame({"gene": ["G"], "train_r2": [.8]})
    results = {"saturating_gene_metrics": frame, "saturating_context_metadata": {"context": "c"}}
    io.save_results(results, cfg)
    assert (tmp_path / "saturating_gene_metrics.csv").is_file()
    assert (tmp_path / "saturating_contexts.json").is_file()
    assert frame.columns.tolist() == ["gene", "train_r2"]


def test_real_fit_checkpoint_recovery_and_missing_tf_registry(tmp_path, monkeypatch):
    """Real fitting/checkpoints; only discovery and simulation are controlled."""
    import pickle
    from test_saturating_transfer import fixture_data
    from CausalBridge import saturating_transfer
    rna, atac, pt, edges, screen, cfg = fixture_data(bins=50)
    rna.var_names=['KO','B','C'];screen.tf_gene=screen.tf_gene.replace({'A':'KO'})
    # This pipeline fixture emulates the current per-bin Step 6 screen.
    screen["lag_mode"] = "local"
    screen["local_lag_max"] = 1
    screen["local_lag_window_rows"] = 12
    screen["local_routes"] = [[dict(target_bin=b,source_bin=b-int(lag) if b>=lag else -1,
        selected_lag=int(lag) if b>=lag else -1,status="supported" if b>=lag else "no_supported_lag")
        for b in range(50)] for lag in screen.tf_lag]
    screen["tf_lag"] = None
    cfg['pseudotime'].update(n_bins=50,n_neighbors=5,method='dpt')
    cfg['granger']={'reference_bins':50,'min_composite_score':.3}
    cfg['output']=dict(dir=str(tmp_path),checkpoint_dir=str(tmp_path/'checkpoints'),cache_dir=str(tmp_path/'cache'))
    cfg["atac_to_rna"]["model_mode"] = "saturating_hybrid"
    cfg["atac_to_rna"]["joint"] = {"workers": 24, "estimator": "ignored_invalid"}
    cfg["perturbation"]["cell_projection"]["enabled"] = False
    monkeypatch.setattr(run, "infer_pseudotime", lambda *a, **k: pt.copy(deep=True))
    monkeypatch.setattr(run, "granger_test", lambda *a, **k: edges.copy())
    monkeypatch.setattr(run, "build_causal_grn", lambda *a, **k: edges.copy())
    monkeypatch.setattr(run, "_prepare_adaptive_motif_resources", lambda *a, **k: {
        "motif_db": None, "tf_database": frozenset(), "resource_cache_dir": str(tmp_path)})
    def screened_fit(*args, **kwargs):
        kwargs["tf_model_registry"].update(saturating_transfer.fit_tf_peak(*args[:3], screen, args[4]))
        return screen.copy()
    monkeypatch.setattr(run, "fit_rna_to_atac", screened_fit)
    def simulate(*args, **kwargs):
        saturating_transfer.validate_registry(kwargs["tf_model_registry"], expected_type="saturating_hybrid_tf_peak")
        return {"perturbation_results": pd.DataFrame(), "projection_inputs": []}
    monkeypatch.setattr(run, "propagate_perturbation", simulate)
    first = run._run_modeling_for_subset(rna, atac, ["KO"], {"KO": "TF"}, cfg)
    assert first["model_mode"] == "saturating_hybrid"
    assert len(first["saturating_gene_metrics"]) == 2
    assert len(first["saturating_tf_metrics"]) == 2
    assert set(first["saturating_response_weights"].layer) == {"peak_gene", "tf_peak"}
    directory = run._checkpoint_dir(run._joint_runtime_config(cfg))
    bytes_before = {name: (directory / (name + ".pkl")).read_bytes() for name in
                    ("step4_granger", "step5_atac_to_rna", "step6_rna_to_atac")}
    monkeypatch.setattr(run, "fit_atac_to_rna", lambda *a, **k: pytest.fail("warm refit"))
    monkeypatch.setattr(run, "fit_rna_to_atac", lambda *a, **k: pytest.fail("warm TF refit"))
    changed = copy.deepcopy(cfg)
    changed["perturbation"]["ko_strength"] = .5
    changed["perturbation"]["min_r2_threshold"] = .9
    warm = run._run_modeling_for_subset(rna, atac, ["KO"], {"KO": "TF"}, changed)
    pd.testing.assert_frame_equal(first["saturating_gene_metrics"], warm["saturating_gene_metrics"])
    assert all((directory / (name + ".pkl")).read_bytes() == data for name, data in bytes_before.items())
    step6 = pickle.loads(bytes_before["step6_rna_to_atac"])
    step6.pop("tf_model_registry")
    (directory / "step6_rna_to_atac.pkl").write_bytes(pickle.dumps(step6))
    with pytest.raises((ValueError, RuntimeError), match="registry"):
        run._run_modeling_for_subset(rna, atac, ["KO"], {"KO": "TF"}, cfg)


def test_real_enabled_projection_uses_raw_half_ko():
    cp = np.column_stack((np.linspace(10, 100, 24), np.linspace(5, 50, 24)**1.2))
    cells = np.log1p(cp)
    factors = np.ones_like(cp); factors[:, 0] = .5
    direct = np.log1p(cp * factors)
    operator = np.zeros((2, 24)); operator[0, :12] = 1/12; operator[1, 12:] = 1/12
    memberships = np.repeat(np.eye(2), 12, axis=0)
    payload = dict(effect_schema_version="bin_saturating_effect_v1",
        intervention_unit="cp10k_before_log1p_and_aggregation",
        state_unit="rna_log1p_cp10k", native_effect_unit="delta_log1p_rna_log1p_cp10k",
        control_cell_states=cells, direct_cell_states=direct, direct_cell_cp10k_factors=factors,
        control_bin_states=operator @ cells, direct_bin_states=operator @ direct,
        perturbed_bin_states=operator @ direct, native_bin_effect=np.zeros((2, 2)),
        cell_bin_weights=memberships, bin_cell_operator=operator,
        features=("TF", "G"), cell_ids=[f"c{i}" for i in range(24)],
        target_genes=("TF",), branch=["b"]*24, bins=np.repeat([0, 1], 12),
        pseudotime=np.linspace(0, 1, 24), bin_branch_ids=["b", "b"], bin_plan_centers=[.25, .75])
    cfg = config(); cfg["perturbation"]["cell_projection"] = {"enabled": True, "n_neighbors": 4}
    projected, states, history, metadata, _ = run._project_bin_forward_inputs(payload, cfg, "c", "TF")
    assert len(projected) == 24 and len(history) == 4
    assert set(metadata.effect_schema_version) == {"bin_saturating_effect_v1"}
    _, genes, reconstructed = run._cell_l0_from_saturating_payload(payload)
    assert genes == ("TF", "G")
    np.testing.assert_allclose(reconstructed[:, 0], cp[:, 0] * .5)
    np.testing.assert_allclose(np.expm1(states["TF"]), cp[:, 0] * .5)
    merge_payload = dict(payload, target_genes=(), target_gene="PEAK:P")
    endpoints = pd.DataFrame({"target_gene": ["PEAK:P"], "affected_gene": ["TF"],
        "selected": [True], "direct_target": [False], "status": ["ok"]})
    merged, diagnostics = run._saturating_merge_contexts([
        {"cluster": "c", "projection_inputs": [merge_payload], "saturating_endpoints": endpoints}],
        {"c": {"n_cells": 24, "original_l0_mean": cp.mean(axis=0)}}, ["TF", "G"])
    assert diagnostics.empty
    assert merged.delta_rna_signed.iloc[0] == pytest.approx(-1.)


def public_fixture(smooth):
    import anndata as ad
    from scipy import sparse
    bins, per = 24, 4
    counts = np.tile(np.repeat([1, 2, 3], 4), 2)
    probability = counts / per
    z = (probability - .5) / .4
    tf = 2 + .4 * z / (1 - abs(z))
    previous = np.concatenate([np.r_[branch[0], branch[:-1]] for branch in probability.reshape(2, 12)])
    gene_l2 = .5 + .15 * (previous - .5) / (.2 + abs(previous - .5))
    r = np.column_stack((np.repeat(tf, per) + np.tile([-.03, -.01, .01, .03], bins),
                         np.repeat(np.expm1(gene_l2), per)))
    a = np.zeros((bins * per, 1))
    for b, count in enumerate(counts):
        a[b*per:b*per+count, 0] = 1
    ids = pd.Index([f"public_c{i}" for i in range(bins * per)])
    rna = ad.AnnData(sparse.csr_matrix(r), obs=pd.DataFrame(index=ids), var=pd.DataFrame(index=["TF", "G"]))
    atac = ad.AnnData(sparse.csr_matrix(a), obs=rna.obs.copy(),
        var=pd.DataFrame({"chr": ["chr1"], "start": [10], "end": [20]}, index=["chr1:10-20"]))
    pt = pd.DataFrame({"bin": np.repeat(np.arange(bins), per),
        "pseudotime": np.repeat(np.tile(np.linspace(0, 1, 12), 2), per),
        "branch": np.repeat(["a", "b"], 48)}, index=ids)
    pt.attrs["branch_boundaries"] = [(0, 12), (12, 24)]
    edges = pd.DataFrame({"gene": ["G"], "peak_id": ["chr1:10-20"]})
    cfg = config()
    cfg["atac_to_rna"].update(log_normalize_rna=True, saturating_hybrid={"max_nfev": 100, "det_starts": 2})
    cfg["pseudotime"].update(n_bins=bins, min_cells_per_bin=per, smooth_kernel=smooth)
    cfg["rna_to_atac"].update(tf_peak_pvalue_threshold=.05)
    cfg["perturbation"].update(min_r2_threshold=0., propagation_rounds=1,
        joint_knockout=True, subnetwork_depth="unlimited", cell_projection={"enabled": True, "n_neighbors": 4})
    cfg["perturbation"]["peak_ko"] = {"mode": "relative", "strength": -.5, "depth": 1, "match": "exact"}
    return rna, atac, pt, edges, cfg


@pytest.mark.parametrize("smooth", ["hard", "gaussian"])
def test_real_public_fit_simulate_project_and_merge(smooth, monkeypatch):
    """Only external motif discovery is mocked; both biological fits/engines are real."""
    from CausalBridge import saturating_transfer as transfer, saturating_hybrid as core
    from CausalBridge.saturating_projection import cell_l0_from_saturating_payload
    rna, atac, pt, edges, cfg = public_fixture(smooth)
    monkeypatch.setattr(kinetics, "_scan_peaks_with_jaspar", lambda *a, **k: {"chr1:10-20": ["TF"]})
    motif_resources = {"motif_db": {"synthetic": {"tf_name": "TF"}},
                       "tf_database": frozenset({"TF"}), "resource_cache_dir": "/tmp/opencode"}
    original_rna = rna.X.copy(); original_atac = atac.X.copy(); original_pt = pt.copy(deep=True)
    table, genes = kinetics.fit_atac_to_rna(rna, atac, pt, edges, cfg)
    tfs = {}
    screen = kinetics.fit_rna_to_atac(rna, atac, pt, edges, cfg,
        tf_model_registry=tfs, motif_resources=motif_resources)
    transfer.validate_registry(genes, expected_type=transfer.GENE_TYPE)
    transfer.validate_registry(tfs, expected_type=transfer.TF_TYPE)
    assert set(tfs["targets"]) == {"chr1:10-20"} and set(genes["targets"]) == {"G"}
    assert set(screen.screening_status) == {"local_pearson_routes"} and not screen.attrs
    assert genes["target_metrics"]["G"]["status"] == "ok"
    assert tfs["target_metrics"]["chr1:10-20"]["status"] == "ok"
    before = copy.deepcopy((genes, tfs))
    for registry in (genes, tfs):
        for bundle in registry["targets"].values():
            model = core.TargetModel.from_bundle(bundle)
            np.testing.assert_allclose(model.weights, np.asarray(model.scores) / sum(model.scores))
            assert len(model.target_rows) >= 12
        assert len(registry["cell_support"]["weights"]) == 24
        assert len(registry["cell_support"]["weights"][0]) == 96
    kwargs = {"transfer_models": genes, "tf_model_registry": tfs, "cluster_label": "public"}
    out = perturbation.propagate_perturbation(["TF"], rna, atac, pt, edges, table, screen, cfg, **kwargs)
    assert not out["perturbation_results"].empty and set(out["perturbation_results"].affected_gene) == {"G"}
    payload = out["projection_inputs"][0]
    _, _, cp = cell_l0_from_saturating_payload(payload)
    np.testing.assert_allclose(cp[:, 0], np.expm1(rna.X.toarray()[:, 0]) * .5, rtol=1e-12)
    assert not np.allclose(payload["direct_cell_states"][:, 0], rna.X.toarray()[:, 0] * .5)
    # Lag-one branch warmup cannot consume the previous branch's final bin.
    np.testing.assert_array_equal(out["rna_state_history"][[0, 12], 1], out["control_rna_states"][[0, 12], 1])
    assert not out["local_events"].duplicated(["relation_type", "target_id", "bin"]).any()
    attribution = out["saturating_peak_attributions"].groupby("bin").delta_l2.sum()
    events = out["local_events"].query("target_id == 'G'").set_index("bin").delta_native
    np.testing.assert_allclose(attribution, events.reindex(attribution.index), atol=1e-12)
    projected, _, _, metadata, _ = run._project_bin_forward_inputs(payload, cfg, "public", "TF")
    assert len(projected) == 96 and set(metadata.effect_schema_version) == {"bin_saturating_effect_v1"}
    context = {"cluster": "public", "projection_inputs": [payload], "saturating_endpoints": out["saturating_endpoints"]}
    merged, diagnostic = run._saturating_merge_contexts([context],
        {"public": {"n_cells": 96, "original_l0_mean": np.expm1(rna.X.toarray()).mean(axis=0)}}, ["TF", "G"])
    assert diagnostic.empty and len(merged) == 1
    actual = out["saturating_endpoints"].set_index("affected_gene").loc["G"].delta_rna
    assert merged.delta_rna_signed.iloc[0] == pytest.approx(actual)
    altered_screen = screen.copy(); altered_screen["weight"] *= -1e30
    repeated = perturbation.propagate_perturbation(["TF"], rna, atac, pt, edges, table, altered_screen, cfg, **kwargs)
    np.testing.assert_array_equal(out["rna_state_history"], repeated["rna_state_history"])
    peak = perturbation.perturb_peak("chr1:10-20", rna, atac, pt, edges, table, screen, cfg, **kwargs)
    np.testing.assert_allclose(peak["atac_state_history"], peak["control_atac_states"] * .5)
    assert peak["peak_id"] == "chr1:10-20" and not peak["perturbation_results"].empty
    zero_cfg = copy.deepcopy(cfg); zero_cfg["perturbation"]["ko_strength"] = 0.
    zero = perturbation.propagate_perturbation(["TF"], rna, atac, pt, edges, table, screen, zero_cfg, **kwargs)
    np.testing.assert_array_equal(zero["rna_state_history"], zero["control_rna_states"])
    np.testing.assert_array_equal(zero["atac_state_history"], zero["control_atac_states"])
    _, zero_states, _, _, _ = run._project_bin_forward_inputs(zero["projection_inputs"][0], cfg, "public", "TF")
    np.testing.assert_array_equal(zero_states[["TF", "G"]], rna.X.toarray())
    assert (genes, tfs) == before
    np.testing.assert_array_equal(rna.X.toarray(), original_rna.toarray())
    np.testing.assert_array_equal(atac.X.toarray(), original_atac.toarray())
    pd.testing.assert_frame_equal(pt, original_pt)


@pytest.mark.parametrize("smooth", ["hard", "gaussian"])
@pytest.mark.parametrize("ko_type", ["gene", "peak"])
def test_real_subset_runner_warm_checkpoint_and_exports(tmp_path, monkeypatch, smooth, ko_type):
    """Frozen synthetic trajectory/graph and motif candidates; no model/engine mocks."""
    import pickle
    from CausalBridge.saturating_transfer import validate_registry, GENE_TYPE, TF_TYPE
    rna, atac, pt, edges, cfg = public_fixture(smooth)
    cfg["output"].update(dir=str(tmp_path), checkpoint_dir=str(tmp_path / "checkpoints"), cache_dir=str(tmp_path / "cache"))
    cfg["pseudotime"].update(n_neighbors=10, method="dpt")
    cfg["granger"] = {"reference_bins": 24, "min_composite_score": .3}
    cfg["perturbation"].update(knockout_type=ko_type)
    cfg["perturbation"]["peak_ko"]["peak_id"] = "chr1:10-20"
    resources = {"motif_db": {"synthetic": {"tf_name": "TF", "pwm": np.full((4, 6), .25)}},
        "tf_database": frozenset({"TF"}), "resource_cache_dir": str(tmp_path / "cache")}
    monkeypatch.setattr(run, "infer_pseudotime", lambda *a, **k: pt.copy(deep=True))
    monkeypatch.setattr(run, "granger_test", lambda *a, **k: edges.assign(is_causal=True))
    monkeypatch.setattr(run, "_prepare_adaptive_motif_resources", lambda *a, **k: resources)
    monkeypatch.setattr(kinetics, "_scan_peaks_with_jaspar", lambda *a, **k: {"chr1:10-20": ["TF"]})
    result = run._run_modeling_for_subset(rna, atac, ["TF"], {"TF": "TF"}, cfg, suffix="synthetic")
    assert result["model_mode"] == "saturating_hybrid" and len(result["cell_projection"]) == 96
    assert not result["perturbation_results"].empty
    assert set(result["saturating_response_weights"].layer) == {"peak_gene", "tf_peak"}
    if ko_type == "peak":
        assert result["peak_id"] == "chr1:10-20"
        assert result["delta_accessibility"] < 0
    directory = run._checkpoint_dir(run._joint_runtime_config(cfg), "synthetic")
    checkpoints = {name: (directory / f"{name}.pkl").read_bytes() for name in
        ("step4_granger", "step5_atac_to_rna", "step6_rna_to_atac")}
    step5 = pickle.loads(checkpoints["step5_atac_to_rna"])
    step6 = pickle.loads(checkpoints["step6_rna_to_atac"])
    validate_registry(step5["transfer_models"], expected_type=GENE_TYPE)
    validate_registry(step6["tf_model_registry"], expected_type=TF_TYPE)
    assert not step6["tf_weights"].attrs
    warm = run._run_modeling_for_subset(rna, atac, ["TF"], {"TF": "TF"}, cfg, suffix="synthetic")
    pd.testing.assert_frame_equal(warm["perturbation_results"], result["perturbation_results"])
    pd.testing.assert_frame_equal(warm["cell_projection"], result["cell_projection"])
    assert all((directory / f"{name}.pkl").read_bytes() == content for name, content in checkpoints.items())
    cfg["output"]["export_details"] = True
    io.save_results(result, cfg)
    output = tmp_path
    assert len(pd.read_csv(output / "saturating_response_weights.csv")) == 2
    assert (output / "perturbation_results.csv").is_file()
    assert (output / "saturating_contexts.json").is_file()
