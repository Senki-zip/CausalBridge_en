"""Synthetic pipeline wiring: no WT datasets, downloads or expensive NN fit."""
from copy import deepcopy
from dataclasses import replace
import json
import pickle

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import yaml

from CausalBridge import run, io, granger, kinetics, perturbation


def inputs(tmp_path):
    rng = np.random.default_rng(77)
    ids = pd.Index([f"cell{i}" for i in range(200)])
    rna = ad.AnnData(sparse.csr_matrix(rng.uniform(.2, 3, (200, 3))),
        obs=pd.DataFrame(index=ids), var=pd.DataFrame({"chr": ["1"]*3,
        "start": [10, 20, 30], "end": [11, 21, 31]}, index=["KO", "TF", "G"]))
    atac = ad.AnnData(sparse.csr_matrix(rng.binomial(1, .5, (200, 2))),
        obs=pd.DataFrame(index=ids), var=pd.DataFrame({"chr": ["1"]*2,
        "start": [10, 20], "end": [12, 22]}, index=["p", "q"]))
    cfg = {"pseudotime": {"n_neighbors": 10, "n_bins": 10, "min_cells_per_bin": 5,
        "adaptive_binning": {"enabled": True}, "smooth_kernel": "gaussian"},
        "granger": {"distance_thresh": 1000, "significance_threshold": .05,
                    "correction_method": "fdr_bh", "min_effect_size": .01,
                    "soft_threshold": True, "min_composite_score": 0},
        "atac_to_rna": {"model_mode": "saturating_hybrid", "log_normalize_rna": True},
        "rna_to_atac": {"max_lag": 1, "tf_expression_filter": False},
        "perturbation": {"ko_strength": .1, "joint_knockout": True,
                         "subnetwork_depth": 2, "cell_projection": {"enabled": True}},
        "output": {"dir": str(tmp_path / "results"), "cache_dir": str(tmp_path / "cache"),
                   "save_intermediate": True}}
    return rna, atac, cfg


def controlled_infer(rna, atac, config, *, defer_binning, joint_repr):
    assert defer_binning and joint_repr.shape[0] == rna.n_obs
    return pd.DataFrame({"pseudotime": np.linspace(0, 1, rna.n_obs),
                         "branch": ["a"] * rna.n_obs}, index=rna.obs_names)


def install_stubs(monkeypatch):
    monkeypatch.setattr(run, "infer_pseudotime", controlled_infer)
    monkeypatch.setattr(kinetics, "_load_motif_database", lambda *args, **kwargs: {})
    seen = []
    def step4(r, a, pt, cfg, *, bin_plan):
        seen.append((4, bin_plan, cfg))
        return pd.DataFrame({"peak_id": ["p"], "gene": ["TF"], "is_causal": [True]})
    from CausalBridge import saturating_transfer as t
    def registry(kind, r, a, pt, cfg, plan):
        snapshot = t._snapshot(r, a, pt, cfg, plan)
        snapshot["context"] = t.response_context(snapshot["context"], cfg, kind)
        return t._registry(kind, snapshot, t.validate_saturating_config(cfg), {}, {}, {})
    def step5(r, a, pt, edges, cfg, *, granger_results, bin_plan):
        seen.append((5, bin_plan, cfg))
        return pd.DataFrame({"peak_id": ["p"], "gene": ["TF"]}), registry(t.GENE_TYPE, r, a, pt, cfg, bin_plan)
    def step6(r, a, pt, edges, cfg, *, target_gene_types, rna_raw, bin_plan, motif_resources, raw_scan_cache, tf_model_registry):
        assert raw_scan_cache["base_cache_dir"] == cfg["output"]["cache_dir"]
        seen.append((6, bin_plan, cfg))
        tf_model_registry.update(registry(t.TF_TYPE, r, a, pt, cfg, bin_plan))
        return pd.DataFrame({"peak_id": ["p"], "tf_gene": ["KO"], "weight": [.1]})
    def step8(targets, r, a, pt, edges, transfer, tf, cfg, *, transfer_models, gene_types, bin_plan, tf_model_registry, cluster_label):
        t.validate_registry(transfer_models)
        t.validate_registry(tf_model_registry)
        seen.append((8, bin_plan, cfg))
        return {"perturbation_results": pd.DataFrame(), "projection_inputs": []}
    for name, function in (("granger_test", step4), ("fit_atac_to_rna", step5),
                           ("fit_rna_to_atac", step6), ("propagate_perturbation", step8)):
        monkeypatch.setattr(run, name, function)
    return seen


def test_fit_once_identity_all_stages_namespace_resume_and_io(tmp_path, monkeypatch):
    rna, atac, cfg = inputs(tmp_path)
    original = deepcopy(cfg)
    seen = install_stubs(monkeypatch)
    fit = run.fit_context_binning
    fits = []
    def fitting(*args, **kwargs):
        context = fit(*args, **kwargs)
        fits.append(context)
        return context
    monkeypatch.setattr(run, "fit_context_binning", fitting)
    result = run._run_modeling_for_subset(rna, atac, ["KO"], {"KO": "TF"}, cfg, suffix="one")
    assert len(fits) == 1
    assert [stage for stage, _, _ in seen] == [4, 5, 6, 8]
    assert all(plan is fits[0].plan for _, plan, _ in seen)
    assert cfg == original
    assert (tmp_path / "results" / "checkpoints" / "one" / "step4_granger.pkl").exists()
    assert not (tmp_path / "results" / "checkpoints" / "adaptive_binning").exists()
    assert seen[-1][2]["output"]["cache_dir"] == cfg["output"]["cache_dir"]
    assert seen[0][2]["granger"]["reference_bins"] == fits[0].plan.n_bins
    metadata = result["binning_metadata"]
    json.dumps(metadata, allow_nan=False)
    directory = metadata["diagnostic_directory"]
    assert (run.Path(directory) / "binning_metadata.json").exists()
    assert (run.Path(directory) / "binning_bins.csv").exists()
    io.save_results({"cluster_results": [result, result]}, cfg)
    assert len(json.loads((tmp_path / "results" / "binning_contexts.json").read_text())) == 2
    namespace = run._adaptive_namespace(fits[0], "one")
    for stage in ("step4_granger", "step5_atac_to_rna", "step6_rna_to_atac"):
        checkpoint = run._load_checkpoint(cfg, stage, namespace)
        assert checkpoint["adaptive_checkpoint"]["schema_version"] == run.ADAPTIVE_CHECKPOINT_SCHEMA
        assert "bin_plan" not in checkpoint
    seen.clear()
    run._run_modeling_for_subset(rna, atac, ["KO"], {"KO": "TF"}, cfg, suffix="one")
    assert [stage for stage, _, _ in seen] == [8]
    assert len(fits) == 2  # exactly once per invocation, even when recovering models


@pytest.mark.parametrize("change", ["expression", "time", "model", "targets", "raw"])
def test_changed_contract_rejects_checkpoint_outside_load_catch(tmp_path, monkeypatch, change):
    rna, atac, cfg = inputs(tmp_path)
    seen = install_stubs(monkeypatch)
    raw = rna.copy()
    run._run_modeling_for_subset(rna, atac, ["KO"], {"KO": "TF"}, cfg, rna_raw=raw)
    seen.clear()
    targets = ["KO"]
    if change == "expression":
        rna.X.data[0] += .1
    elif change == "time":
        def changed(*args, **kwargs):
            frame = controlled_infer(*args, **kwargs)
            frame["pseudotime"] = frame.pseudotime**2
            return frame
        monkeypatch.setattr(run, "infer_pseudotime", changed)
    elif change == "model":
        cfg["atac_to_rna"]["saturating_hybrid"] = {"ridge": .123}
    elif change == "targets":
        targets = ["TF"]
    else:
        cfg["rna_to_atac"]["tf_expression_filter"] = True
        raw.var_names = ["KO", "TF", "OTHER"]
    expected_stage = {"expression": "step4_granger", "time": "step4_granger",
                      "model": "step5_atac_to_rna", "targets": "step6_rna_to_atac",
                      "raw": "step6_rna_to_atac"}[change]
    with pytest.raises(RuntimeError, match=expected_stage + ".*使用新的checkpoint_dir"):
        run._run_modeling_for_subset(rna, atac, targets, {"KO": "TF"}, cfg, rna_raw=raw)
    assert not seen


def test_missing_checkpoint_metadata_and_upstream_mismatch(tmp_path, monkeypatch):
    rna, atac, cfg = inputs(tmp_path)
    install_stubs(monkeypatch)
    result = run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg)
    metadata = result["binning_metadata"]["checkpoint_metadata"]
    with pytest.raises(RuntimeError, match="missing or mismatched"):
        run._validate_adaptive_checkpoint({}, metadata, "step4_granger", None)
    good = {"adaptive_checkpoint": dict(metadata, stage="step5_atac_to_rna", upstream_fingerprint="old")}
    with pytest.raises(RuntimeError):
        run._validate_adaptive_checkpoint(good, metadata, "step5_atac_to_rna", "changed")
    # Corrupt the metadata of a readable enabled checkpoint: no silent retry.
    paths = [tmp_path / "results" / "checkpoints" / "step4_granger.pkl"]
    with paths[0].open("wb") as handle:
        pickle.dump({"granger_results": pd.DataFrame()}, handle)
    with pytest.raises(RuntimeError, match="使用新的checkpoint_dir"):
        run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg)


def test_invalid_plan_aborts_before_stages_and_shuffle_before_embedding(tmp_path, monkeypatch):
    rna, atac, cfg = inputs(tmp_path)
    seen = install_stubs(monkeypatch)
    original = run.fit_context_binning
    def invalid(*args, **kwargs):
        context = original(*args, **kwargs)
        return replace(context, plan=replace(context.plan, cell_valid=np.zeros(rna.n_obs, bool)))
    monkeypatch.setattr(run, "fit_context_binning", invalid)
    with pytest.raises(ValueError, match="uncovered"):
        run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg)
    assert not seen
    cfg["pseudotime"]["ablation_shuffle_pseudotime"] = True
    monkeypatch.setattr(run, "prepare_wt_embedding", lambda *args: pytest.fail("embedding before shuffle validation"))
    with pytest.raises(ValueError, match="shuffled"):
        run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg)




def test_io_defaults_and_unknown_settings(tmp_path):
    for name in ("rna.h5ad", "atac.h5ad"):
        (tmp_path / name).touch()
    base = {"input": {"rna_h5ad": str(tmp_path / "rna.h5ad"), "atac_h5ad": str(tmp_path / "atac.h5ad")}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(base))
    assert io.load_config(str(path))["pseudotime"]["adaptive_binning"]["enabled"] is False
    assert io.load_config(str(path))["pseudotime"]["adaptive_binning"]["state_modality"] == "joint"
    for settings in ({"enabled": True, "unknown": 1}, {"embedding": {"unknown": 1}}, {"state_budget": -1},
                     {"bandwidth_selection": "cells"}, {"boundary": "bad"},
                     {"state_modality": True}, {"enabled": True, "state_modality": "atac"}):
        base["pseudotime"] = {"adaptive_binning": settings}
        path.write_text(yaml.safe_dump(base))
        with pytest.raises(ValueError):
            io.load_config(str(path))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("boundary", ["split_noise", "none"])
def test_io_ess_strategy_direct_keys(tmp_path, enabled, boundary):
    for name in ("rna.h5ad", "atac.h5ad"):
        (tmp_path / name).touch()
    settings = {"enabled": enabled, "bandwidth_selection": "ess",
                "boundary": boundary, "state_modality": "rna"}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "input": {"rna_h5ad": str(tmp_path / "rna.h5ad"),
                  "atac_h5ad": str(tmp_path / "atac.h5ad")},
        "pseudotime": {"adaptive_binning": settings}}))
    loaded = io.load_config(str(path))["pseudotime"]["adaptive_binning"]
    assert {key: loaded[key] for key in settings} == settings


def test_state_mode_keeps_joint_trajectory_and_shared_plan(tmp_path, monkeypatch):
    rna, atac, cfg = inputs(tmp_path)
    seen = install_stubs(monkeypatch)
    captured = []
    def infer(*args, **kwargs):
        captured.append(kwargs["joint_repr"].copy())
        return controlled_infer(*args, **kwargs)
    monkeypatch.setattr(run, "infer_pseudotime", infer)
    results = []
    for mode in ("joint", "rna"):
        cfg["pseudotime"]["adaptive_binning"]["state_modality"] = mode
        seen.clear()
        results.append(run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg, suffix=mode))
        assert [stage for stage, _, _ in seen] == [4, 5, 6, 8]
        assert all(plan is seen[0][1] for _, plan, _ in seen)
        assert results[-1]["binning_metadata"]["state_embedding"]["state"]["shape"][1] == (5 if mode == "joint" else 3)
        metadata = results[-1]["binning_metadata"]
        persisted = json.loads((run.Path(metadata["diagnostic_directory"]) / "binning_metadata.json").read_text())
        assert persisted["state_embedding"] == metadata["state_embedding"]
        assert persisted["state_embedding"]["state_modality"] == mode
    pd.testing.assert_frame_equal(captured[0], captured[1])
    assert captured[0].shape[1] == 5  # actual 3 RNA + 2 ATAC, not assumed 5+5




def test_peak_entry_receives_identical_plan_and_preserved_root(tmp_path, monkeypatch):
    rna, atac, cfg = inputs(tmp_path)
    cfg["perturbation"].update(knockout_type="peak", peak_ko={"peak_id": "p", "depth": 1})
    seen = install_stubs(monkeypatch)
    def peak(peak_id, r, a, pt, edges, transfer, weights, config, *, transfer_models, bin_plan, tf_model_registry, cluster_label):
        assert peak_id == "p"
        seen.append((8, bin_plan, config))
        return {"perturbation_results": pd.DataFrame()}
    monkeypatch.setattr(run, "perturb_peak", peak)
    run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg)
    assert [stage for stage, _, _ in seen] == [4, 5, 6, 8]
    assert all(plan is seen[0][1] for _, plan, _ in seen)


@pytest.mark.parametrize("custom_root", [False, True])
@pytest.mark.parametrize("suffix", [None, "cluster_7"])
def test_enabled_established_checkpoint_layout_and_warm_restore(tmp_path, monkeypatch, custom_root, suffix):
    rna, atac, cfg = inputs(tmp_path)
    if custom_root:
        cfg["output"]["checkpoint_dir"] = str(tmp_path / "given_checkpoints")
    seen = install_stubs(monkeypatch)
    run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg, suffix=suffix)
    root = tmp_path / "given_checkpoints" if custom_root else tmp_path / "results" / "checkpoints"
    context = root / suffix if suffix else root
    assert run._checkpoint_dir(cfg, suffix) == context
    for stage in ("step4_granger", "step5_atac_to_rna", "step6_rna_to_atac"):
        assert (context / (stage + ".pkl")).exists()
    assert not (root / "adaptive_binning").exists()
    seen.clear()
    run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg, suffix=suffix)
    assert [stage for stage, _, _ in seen] == [8]


@pytest.mark.parametrize("custom_root", [False, True])
def test_existing_legacy_checkpoint_same_path_rejected_not_overwritten(tmp_path, monkeypatch, custom_root):
    rna, atac, cfg = inputs(tmp_path)
    if custom_root:
        cfg["output"]["checkpoint_dir"] = str(tmp_path / "given_checkpoints")
    seen = install_stubs(monkeypatch)
    run._save_checkpoint(cfg, "step4_granger", {"granger_results": pd.DataFrame(), "legacy": True}, "cluster_7")
    path = run._checkpoint_dir(cfg, "cluster_7") / "step4_granger.pkl"
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="使用新的checkpoint_dir"):
        run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg, suffix="cluster_7")
    assert path.read_bytes() == before and not seen
    assert not (path.parent / "step5_atac_to_rna.pkl").exists()


@pytest.mark.parametrize("resource", ["genome_edit", "genome_available", "motif", "mapping", "tf_database"])
def test_resource_contents_or_availability_reject_recovery_change_candidate_namespace(tmp_path, monkeypatch, resource):
    rna, atac, cfg = inputs(tmp_path)
    seen = install_stubs(monkeypatch)
    genome = tmp_path / "genome.fa"
    cfg["input"] = {"genome_fasta": str(genome)}
    installation = tmp_path / "site"
    motif_dir = installation / "data" / "motif_databases"
    motif_dir.mkdir(parents=True)
    monkeypatch.setattr(kinetics, "get_paths", lambda: {"purelib": str(installation)})
    cache = tmp_path / "cache"
    cache.mkdir()
    if resource != "genome_available":
        genome.write_text(">tiny\nAAAA\n")
    files = {"motif": motif_dir / "gimme.vertebrate.v5.0.pfm",
             "mapping": motif_dir / "gimme.vertebrate.v5.0.motif2factors.txt",
             "tf_database": cache / "tf_database.txt"}
    if resource in files:
        files[resource].write_text("before\n")
    old = run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg)
    old_metadata = old["binning_metadata"]["checkpoint_metadata"]
    old_candidate_dir = seen[0][2]["output"]["cache_dir"]
    seen.clear()
    (files[resource] if resource in files else genome).write_text("after\n")
    with pytest.raises(RuntimeError, match="step6_rna_to_atac.*resources.*使用新的checkpoint_dir"):
        run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg)
    assert not seen
    # Resource fingerprints reject recovery while configured cache paths stay literal.
    cfg["output"]["dir"] = str(tmp_path / "fresh")
    fresh = run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg)
    new_metadata = fresh["binning_metadata"]["checkpoint_metadata"]
    assert new_metadata["resource_identity"] != old_metadata["resource_identity"]
    assert new_metadata["model_contract_fingerprint"] != old_metadata["model_contract_fingerprint"]
    assert seen[0][2]["output"]["cache_dir"] == old_candidate_dir


@pytest.mark.parametrize("damage", ["plan", "policy", "data", "model", "unreadable"])
def test_recovered_step5_inner_bundle_and_root_fail_before_step6_or_nn_loader(tmp_path, monkeypatch, damage):
    rna, atac, cfg = inputs(tmp_path)
    seen = install_stubs(monkeypatch)
    run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg)
    path = tmp_path / "results" / "checkpoints" / "step5_atac_to_rna.pkl"
    if damage == "unreadable":
        path.write_bytes(b"not a pickle")
    else:
        with path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        key = {"plan": "bin_plan_fingerprint", "policy": "bin_policy_fingerprint",
               "data": "binning_data_fingerprint", "model": "binning_model_contract_fingerprint"}[damage]
        checkpoint["transfer_models"]["binning_metadata"][key] = "wrong-inner-hash"
        with path.open("wb") as handle:
            pickle.dump(checkpoint, handle)
    seen.clear()
    with pytest.raises(RuntimeError, match="使用新的checkpoint_dir"):
        run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg)
    assert not seen  # Step4 recovered; Step6 and perturbation never execute
