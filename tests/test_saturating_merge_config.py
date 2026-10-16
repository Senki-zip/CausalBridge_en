import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from CausalBridge import binning_adapter, io, run


def write_config(tmp_path, policy=None, include=False):
    rna = tmp_path / "rna.h5ad"; rna.touch()
    atac = tmp_path / "atac.h5ad"; atac.touch()
    cfg = {"input": {"rna_h5ad": str(rna), "atac_h5ad": str(atac)}}
    if include:
        cfg["perturbation"] = {"aggregation_policy": policy}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_omitted_policy_preserves_library_and_mode_defaults(tmp_path):
    cfg = io.load_config(write_config(tmp_path))
    assert cfg["perturbation"]["aggregation_policy"] == "strict_complete"
    assert cfg["atac_to_rna"]["model_mode"] == "saturating_hybrid"


@pytest.mark.parametrize("policy", ["strict_complete", "valid_cell_means"])
def test_known_policy_loads(tmp_path, policy):
    cfg = io.load_config(write_config(tmp_path, policy, include=True))
    assert cfg["perturbation"]["aggregation_policy"] == policy


@pytest.mark.parametrize("policy", ["unknown", "valid_cells_means", None, True, 1, [], {}])
def test_unknown_policy_rejected_upfront(tmp_path, policy):
    with pytest.raises(ValueError, match=r"perturbation\.aggregation_policy.*strict_complete.*valid_cell_means"):
        io.load_config(write_config(tmp_path, policy, include=True))


def test_current_config_enables_repair_and_only_policy_changes_contract(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = io.load_config(path)
    assert cfg["perturbation"]["aggregation_policy"] == "valid_cell_means"
    original = copy.deepcopy(cfg)
    omitted = copy.deepcopy(cfg)
    omitted["perturbation"].pop("aggregation_policy")
    strict = copy.deepcopy(cfg)
    strict["perturbation"]["aggregation_policy"] = "strict_complete"
    # Full adapter contract, including the formerly retained perturbation section.
    contract = binning_adapter._model_config(omitted)
    assert binning_adapter._model_config(strict) == contract
    assert binning_adapter._model_config(cfg) == contract
    assert run._step5_model_config(cfg) == run._step5_model_config(omitted)
    # Real stage source signatures and stage metadata, synthetic matrices only.
    model_cfg = copy.deepcopy(cfg)
    model_cfg["input"]["gene_identity_mode"] = "legacy"
    data = SimpleNamespace(X=np.ones((2, 1)), obs_names=pd.Index(["c0", "c1"]),
        var_names=pd.Index(["G"]), var=pd.DataFrame(index=["G"]), uns={})
    plan = SimpleNamespace(fingerprint="plan", cell_ids=np.array(["c0", "c1"]),
        times=np.array([0., 1.]), branches=np.array(["b", "b"]), centers=np.array([.5]))
    context = SimpleNamespace(plan=plan, policy=SimpleNamespace(fingerprint="policy", branch_boundaries=[]),
        metadata={}, model_contract_fingerprint="adapter")
    monkeypatch.setattr(run, "_resource_identity", lambda *a: {})
    metadata = run._adaptive_checkpoint_metadata(context, model_cfg, ["G"], {"G": "TF"},
        rna=data, atac=data, n_peaks=1)
    model_cfg["perturbation"].pop("aggregation_policy")
    prior = run._adaptive_checkpoint_metadata(context, model_cfg, ["G"], {"G": "TF"},
        rna=data, atac=data, n_peaks=1)
    for stage in ("step5_atac_to_rna", "step6_rna_to_atac"):
        assert metadata["stages"][stage] == prior["stages"][stage]
        assert run._provenance_fingerprint(metadata["stages"][stage]) == run._provenance_fingerprint(prior["stages"][stage])
    assert cfg == original


def test_merge_export_retains_scope_and_same_subset_means(tmp_path):
    endpoints = pd.DataFrame([dict(source_context="a", target_gene="KO", affected_gene="G",
        control_mean_cp10k=2., perturbed_mean_cp10k=8., status="ok", selected=True, direct_target=False)])
    merged, diagnostic = run._aggregate_saturating_cell_means(endpoints,
        {"a": {"n_cells": 2}, "missing": {"n_cells": 6}})
    io.save_results({"perturbation_merged": merged, "saturating_aggregation_diagnostics": diagnostic},
        {"atac_to_rna": {"model_mode": "saturating_hybrid"},
         "output": {"dir": str(tmp_path), "save_intermediate": False}})
    saved = pd.read_csv(tmp_path / "perturbation_merged.csv")
    assert saved.iloc[0].aggregation_scope == "partial"
    assert saved.iloc[0].aggregation_used_cells == 2
    assert saved.iloc[0].aggregation_total_cells == 8
    assert saved.iloc[0].aggregation_cell_fraction == .25
    assert saved.iloc[0].control_mean_cp10k == 2.
    assert saved.iloc[0].perturbed_mean_cp10k == 8.
    assert saved.iloc[0].delta_rna_signed == pytest.approx(2.)
