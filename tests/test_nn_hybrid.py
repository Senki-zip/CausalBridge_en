"""Real CPU fits: independence, support, codecs and mixed-family adapters."""
import builtins
import copy
from dataclasses import replace
import numpy as np
import pytest
from scipy.special import expit
from CausalBridge import nn_hybrid as nn, saturating_hybrid as core
from CausalBridge import saturating_transfer as transfer, kinetics, run
from CausalBridge.saturating_perturbation import perturb_peak_saturating
from test_saturating_transfer import fixture_data


def data(X, y, cfg=None):
    n, p = X.shape
    return core.TargetData("target", tuple((f"p{i}", 0) for i in range(p)), X, y,
        np.tile(np.arange(n)[:, None], (1, p)), np.arange(n), ("b",)*n,
        "raw_atac", "raw_atac", dict(response_family="nn", nn_response_config=(cfg or nn.NNConfig()).to_dict()),
        cell_weights=np.eye(n))


def assert_pipeline_registry_contracts(cfg, genes, tfs):
    """Exercise fresh Step 5 and resumed Steps 5/6 using real fitted registries."""
    context = genes["context"]
    adapter = dict(binning_model_contract_domain="saturating-hybrid-v1",
                   n_bins=context["n_bins"], bin_plan_centers=context["bin_centers"],
                   branch_boundaries=context["branch_boundaries"])
    run._validate_step5_binning_bundle(genes, adapter, len(context["peak_ids"]), context["peak_ids"])
    expected = dict(saturating_data_fingerprint=context["data_fingerprint"],
                    rna_gene_ids=context["gene_ids"], atac_peak_ids=context["peak_ids"],
                    atac_n_vars=len(context["peak_ids"]), adapter_contract=adapter,
                    stages={"step5_atac_to_rna": {"atac_to_rna": run._step5_model_config(cfg)},
                            "step6_rna_to_atac": {"rna_to_atac": run._step6_model_config(cfg)}})
    for stage, registry_key, registry in (("step5_atac_to_rna", "transfer_models", genes),
                                          ("step6_rna_to_atac", "tf_model_registry", tfs)):
        checkpoint = dict(adaptive_checkpoint=run._stage_checkpoint_metadata(expected, stage, None),
                          causal_edges=[], transfer_functions=[],  tf_weights=[],
                          **{registry_key: registry})
        run._validate_adaptive_checkpoint(checkpoint, expected, stage, None)
        other = expected
        # NN registries intentionally omit the saturation-only budget policy.
        if registry["context"].get("response_family") == "nn":
            other = copy.deepcopy(expected)
            other["stages"]["step5_atac_to_rna"]["atac_to_rna"]["saturating_hybrid"]["budget_exhausted_policy"] = "best_finite"
            checkpoint["adaptive_checkpoint"] = run._stage_checkpoint_metadata(other, stage, None)
            run._validate_adaptive_checkpoint(checkpoint, other, stage, None)
        bad = copy.deepcopy(registry)
        saturation_schema = transfer.GENE_SCHEMA if registry_key == "transfer_models" else transfer.TF_SCHEMA
        bad["schema"] = bad["schema_version"] = saturation_schema if registry["context"].get("response_family") == "nn" else "hybrid_nn_peak_gene_registry_v1"
        bad["fingerprint"] = core._digest({k: v for k, v in bad.items() if k != "fingerprint"})
        checkpoint[registry_key] = bad
        with pytest.raises(RuntimeError, match="registry"):
            run._validate_adaptive_checkpoint(checkpoint, other, stage, None)
        if registry_key == "transfer_models":
            with pytest.raises(ValueError, match="schema"):
                run._validate_step5_binning_bundle(bad, adapter, len(context["peak_ids"]))
        mismatched = copy.deepcopy(expected)
        layer = "atac_to_rna" if registry_key == "transfer_models" else "rna_to_atac"
        mismatched["stages"][stage][layer]["response_model"] = "saturation" if registry["context"].get("response_family") == "nn" else "nn"
        checkpoint[registry_key] = registry
        checkpoint["adaptive_checkpoint"] = run._stage_checkpoint_metadata(mismatched, stage, None)
        with pytest.raises(RuntimeError, match="contract mismatch"):
            run._validate_adaptive_checkpoint(checkpoint, mismatched, stage, None)
    with pytest.raises(ValueError, match="registry"):
        run._validate_step5_binning_bundle({}, adapter, len(context["peak_ids"]))


@pytest.mark.parametrize("shape", ["sigmoid", "nonmonotone"])
def test_independent_batched_networks(shape):
    x = np.linspace(0, 1, 96)
    y = expit(5*np.tanh(3*(x-.5))) if shape == "sigmoid" else .2+.6*np.sin(np.pi*x)**2
    X = np.column_stack([x, x**2, np.ones(len(x))])
    keys = (("a", 0), ("b", 1), ("constant", 0))
    cfg = nn.NNConfig(epochs=300, l2=.0001)
    import torch
    threads = torch.get_num_threads()
    batched = nn.fit_singles(X, y, keys, (0, 1), cfg)
    reordered = nn.fit_singles(X[:, [1, 0]], y, keys[1::-1], (0, 1), cfg)
    for i in range(2):
        single = nn.fit_singles(X[:, i:i+1], y, (keys[i],), (0, 1), cfg)[0]
        np.testing.assert_allclose(single.W, batched[i].W, rtol=0, atol=1e-12)
        np.testing.assert_allclose(reordered[1-i].predict(X[:, i]), single.predict(X[:, i]), rtol=0, atol=1e-12)
        assert batched[i].metrics["training_status"] == "fixed_epochs_completed"
        assert batched[i].metrics["solver_converged"] is False
        assert batched[i].metrics["n_rows"] == len(y)
    assert batched[0].metrics["training_r2"] > .75
    assert batched[2].metrics["status"] == "constant_input"
    assert torch.get_num_threads() == threads


def test_real_group_acceptance_routing_and_codec():
    rng = np.random.default_rng(12)
    X = rng.uniform(.02, .98, (160, 2))
    y = expit(4*np.tanh(5*(X[:, 0]+X[:, 1]-1)))
    cfg = nn.NNConfig(epochs=300, l2=.00001)
    wt = data(X, y, cfg)
    model = core.fit_target(wt, (0, 1), core.HybridConfig(min_penalized_gain=.01))
    assert model.groups and isinstance(model.groups[0], nn.NNGroupModel)
    assert model.groups[0].n_parameters == 17
    assert model.groups[0].diagnostics["effective_rows_required"] == 34
    bundle = model.to_bundle()
    assert bundle["schema"] == nn.SCHEMA
    loaded = core.TargetModel.from_bundle(bundle)
    assert loaded.to_bundle() == bundle
    # Each WT bin represents multiple cells: even a near-exact neighbor must
    # satisfy the unchanged local query ESS gate (default 10 cells).
    query_support = core.CellSupport(np.kron(np.eye(160), np.ones((1, 16))/16))
    wt = replace(wt, cell_weights=query_support.weights, cell_support_ref="")
    loaded = core.TargetModel.from_bundle(core.fit_target(wt, (0, 1), model.config).to_bundle())
    prediction = core.predict_target(loaded, X[:1]+1e-6, wt_X=X[:1], branch_ids=("b",), changed_inputs=loaded.input_keys,
                                     cell_support=query_support)
    assert prediction.route == "group"
    no_op = core.predict_delta(loaded, X[:1], X[:1], branch_ids=("b",))
    np.testing.assert_array_equal(no_op.values, np.zeros(1))
    bad = copy.deepcopy(bundle)
    bad["singles"][0]["response_type"] = "saturation"
    bad["fingerprint"] = core._digest({k:v for k,v in bad.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="family"):
        core.TargetModel.from_bundle(bad)
    # A real NN remains a real NN during inference, with no Torch imports.
    import_original = builtins.__import__
    def reject_torch(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("prediction imported Torch")
        return import_original(name, *args, **kwargs)
    from unittest.mock import patch
    with patch("builtins.__import__", reject_torch):
        loaded = core.TargetModel.from_bundle(bundle)
        values = loaded.groups[0].predict(X)
        assert np.all((values >= 0) & (values <= 1))


@pytest.mark.parametrize("mode", ["collinear", "low_ess", "row_floor"])
def test_group_rejection(mode):
    rng = np.random.default_rng(42)
    X = rng.uniform(.1, .9, (160, 2))
    if mode == "collinear":
        X[:, 1] = X[:, 0]
    if mode == "row_floor":
        X = X[:30]
    y = expit(4*(X[:, 0]+X[:, 1]-1))
    wt = data(X, y, nn.NNConfig(epochs=60))
    if mode == "low_ess":
        wt = replace(wt, cell_weights=np.full((160, 4), .25), cell_support_ref="")
    model = core.fit_target(wt, (0, 1))
    assert not model.groups


@pytest.mark.parametrize("gene_family,tf_family", [("nn", "nn"), ("nn", "saturation"), ("saturation", "nn"), ("saturation", "saturation")])
@pytest.mark.parametrize("smooth", ["hard", "gaussian"])
def test_public_fits_and_intervention(monkeypatch, gene_family, tf_family, smooth):
    r, a, pt, edges, _, cfg = fixture_data(bins=64, branches=2)
    cfg["pseudotime"].update(smooth_kernel=smooth, n_bins=64)
    cfg["atac_to_rna"].update(response_model=gene_family, nn_response=dict(epochs=60))
    cfg["rna_to_atac"] = dict(response_model=tf_family, nn_response=dict(epochs=60), time_lag=False,
                             tf_expression_filter=False)
    monkeypatch.setattr(kinetics, "_load_motif_database", lambda *a: {"mock": object()})
    monkeypatch.setattr(kinetics, "_scan_peaks_with_jaspar", lambda *a, **kw: {"p0": ["A"], "p1": ["A"]})
    # Only external motif inputs are mocked: Pearson screen and both fitters run.
    table, genes = kinetics.fit_atac_to_rna(r, a, pt, edges, cfg)
    tfs = {}
    screened = kinetics.fit_rna_to_atac(r, a, pt, edges, cfg,  tf_model_registry=tfs)
    assert_pipeline_registry_contracts(cfg, genes, tfs)
    assert genes["targets"] and tfs["targets"] and len(screened)
    for registry, family in ((genes, gene_family), (tfs, tf_family)):
        transfer.validate_registry(registry)
        for bundle in registry["targets"].values():
            assert bundle["provenance"].get("response_family", "saturation") == family
            assert all(s.get("response_type", "saturation") == family for s in bundle["singles"])
    out = perturb_peak_saturating("p0", r, a, pt, edges, table, screened, cfg,
                                 transfer_models=genes, tf_model_registry=tfs, )
    assert out["effect_schema_version"] == "bin_saturating_effect_v1"
    cfg["perturbation"]["peak_ko"] = dict(mode="relative", strength=0)
    noop = perturb_peak_saturating("p0", r, a, pt, edges, table, screened, cfg,
                                  transfer_models=genes, tf_model_registry=tfs, )
    np.testing.assert_array_equal(list(noop["delta_rna"].values()), np.zeros(len(noop["delta_rna"])))


def test_spawn_matches_serial():
    r, a, pt, edges, _, cfg = fixture_data(bins=64)
    cfg["atac_to_rna"].update(response_model="nn", nn_response=dict(epochs=60))
    serial = transfer.fit_peak_gene(r, a, pt, edges, cfg)[2]
    cfg["atac_to_rna"]["saturating_hybrid"]["workers"] = 2
    parallel = transfer.fit_peak_gene(r, a, pt, edges, cfg)[2]
    assert parallel == serial


@pytest.mark.parametrize("options", [{"epochs": True}, {"hidden_width": 0}, {"lr": 0}, {"l2": -1}, {"seed": -1}, {"restarts": 4}, {"unknown": 1}, {"lr": np.inf}])
def test_strict_config(options):
    with pytest.raises(ValueError):
        nn.NNConfig.from_dict(options)


def test_inactive_contract_and_namespaces():
    cfg = dict(atac_to_rna=dict(model_mode="saturating_hybrid"), rna_to_atac={}, output=dict(dir="results", checkpoint_dir="checkpoints"))
    old = run._step5_model_config(cfg)
    cfg["atac_to_rna"]["nn_response"] = dict(epochs=17)
    assert run._step5_model_config(cfg) == old
    cfg["atac_to_rna"]["response_model"] = "nn"
    runtime = run._joint_runtime_config(cfg)
    assert runtime["output"]["dir"] == cfg["output"]["dir"]
    assert run._joint_runtime_config(runtime)["output"] == runtime["output"]


@pytest.mark.parametrize("gene_family,tf_family", [("nn", "nn"), ("nn", "saturation"), ("saturation", "nn"), ("saturation", "saturation")])
@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("checkpoint_dir", [None, "custom/gene_ids/user/checkpoints"])
def test_identity_layer_namespace_idempotence(gene_family, tf_family, continuous, checkpoint_dir):
    from pathlib import Path
    cfg = dict(input=dict(gene_identity_mode="gtf_gene_id", annotation_gtf=__file__),
               atac_to_rna=dict(model_mode="saturating_hybrid", response_model=gene_family),
               rna_to_atac=dict(response_model=tf_family),
               preprocess=dict(atac_depth_normalization=dict(enabled=continuous)),
               output=dict(dir="custom/saturating_hybrid/user/results", checkpoint_dir=checkpoint_dir))
    runtime = run._joint_runtime_config(cfg)
    assert runtime["output"]["dir"] == cfg["output"]["dir"]
    assert runtime["output"]["checkpoint_dir"] == checkpoint_dir
    assert run._checkpoint_dir(runtime) == Path(checkpoint_dir or "custom/saturating_hybrid/user/results/checkpoints")
    assert run._joint_runtime_config(runtime) == runtime
    assert cfg["output"]["dir"] == "custom/saturating_hybrid/user/results"


def test_inactive_invalid_settings_and_legacy_contracts():
    from CausalBridge.binning_adapter import _model_config
    for mode in ("saturating_hybrid",):
        cfg = dict(atac_to_rna=dict(model_mode=mode), rna_to_atac={})
        before = (_model_config(cfg), run._step5_model_config(cfg), run._step6_model_config(cfg))
        for layer in ("atac_to_rna", "rna_to_atac"):
            cfg[layer]["nn_response"] = dict(epochs=False, unknown="ignored")
            if mode != "saturating_hybrid":
                cfg[layer]["response_model"] = "unused"
        assert (_model_config(cfg), run._step5_model_config(cfg), run._step6_model_config(cfg)) == before
    cfg["atac_to_rna"].update(model_mode="saturating_hybrid", response_model="nn")
    with pytest.raises(ValueError):
        transfer.validate_saturating_config(cfg)


@pytest.mark.parametrize("field,value", [
    ("input_unit", "rna_log1p_cp10k"), ("W", [[1., 2.]]),
    ("metrics.n_rows", True), ("metrics.training_rmse", True),
    ("metrics.epochs", True), ("metrics.restarts", True),
    ("metrics.best_objective", -1.), ("metrics.solver_converged", True),
])
def test_resigned_nn_corruption_rejected(field, value):
    x = np.linspace(.1, .9, 32)
    model = core.fit_target(data(x[:, None], expit(4*(x-.5)), nn.NNConfig(epochs=10)), (0, 1))
    bundle = model.to_bundle()
    node = bundle["singles"][0]
    parts = field.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    bundle["fingerprint"] = core._digest({k:v for k,v in bundle.items() if k != "fingerprint"})
    with pytest.raises(ValueError):
        core.TargetModel.from_bundle(bundle)


@pytest.mark.parametrize("gene_family,tf_family", [("nn", "nn"), ("nn", "saturation"), ("saturation", "nn"), ("saturation", "saturation")])
def test_continuous_public_fits(tmp_path, monkeypatch, gene_family, tf_family):
    from test_atac_normalization import continuous_fixture
    from CausalBridge import atac_normalization as norm
    from CausalBridge.saturating_perturbation import propagate_saturating
    r, a, pt, edges, cfg = continuous_fixture(tmp_path)
    cfg["atac_to_rna"].update(response_model=gene_family, nn_response=dict(epochs=40))
    cfg["atac_to_rna"]["saturating_hybrid"]["budget_exhausted_policy"] = "best_finite"
    cfg["rna_to_atac"].update(response_model=tf_family, nn_response=dict(epochs=40))
    monkeypatch.setattr(kinetics, "_scan_peaks_with_jaspar", lambda *a, **k: {"chr1:10-20": ["TF"]})
    table, genes = kinetics.fit_atac_to_rna(r, a, pt, edges, cfg)
    tfs = {}
    screen = kinetics.fit_rna_to_atac(r, a, pt, edges, cfg,  tf_model_registry=tfs,
        motif_resources={"motif_db": {"synthetic": {"tf_name": "TF"}}, "tf_database": frozenset({"TF"}), "resource_cache_dir": str(tmp_path)})
    assert_pipeline_registry_contracts(cfg, genes, tfs)
    assert genes["targets"] and tfs["targets"]
    assert genes["input_unit"] == tfs["output_unit"] == norm.UNIT
    for registry, family in ((genes, gene_family), (tfs, tf_family)):
        transfer.validate_registry(registry)
        assert ("budget_exhausted_policy" in registry) == (family == "saturation")
        for bundle in registry["targets"].values():
            model = core.TargetModel.from_bundle(bundle)
            assert model.to_bundle() == bundle
            assert all(isinstance(s, nn.NNSingleModel) == (family == "nn") for s in model.singles)
    cfg["perturbation"]["ko_strength"] = 0
    out = propagate_saturating(["TF"], r, a, pt, edges, table, screen, cfg,
        transfer_models=genes, tf_model_registry=tfs, )
    np.testing.assert_array_equal(out["atac_state_history"], out["control_atac_states"])
    np.testing.assert_array_equal(out["rna_state_history"], out["control_rna_states"])
    _, weights, curves = run._saturating_registry_reports(genes, "peak_gene")
    assert len(weights) and len(curves)
    if gene_family == "nn":
        assert curves["b"].isna().all() and curves["nn_weights_json"].notna().all()


def test_numeric_failed_edge_is_pruned(monkeypatch):
    x = np.linspace(.1, .9, 64)
    wt = data(np.column_stack([x, x**2]), expit(4*(x-.5)), nn.NNConfig(epochs=60))
    original = nn._train
    def corrupt(*args):
        result = original(*args)
        result[0][0] = None
        return result
    monkeypatch.setattr(nn, "_train", corrupt)
    model = core.fit_target(wt, (0, 1))
    assert len(model.singles) == 1
    assert model.provenance["excluded_single_relations"]["records"][0]["status"] == "failed"
