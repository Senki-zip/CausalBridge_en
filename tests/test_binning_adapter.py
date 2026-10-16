"""Integration contract tests; numerical state-policy oracles live separately."""
from copy import deepcopy
from dataclasses import replace
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from CausalBridge.adaptive_binning import build_plan
from CausalBridge.binning_adapter import (
    aggregate_with_plan, fit_context_binning, persist_binning_diagnostics,
    prepare_wt_embedding, require_bin_plan, validate_bin_plan, select_state_embedding,
)


def inputs(n=240, features=12):
    rng = np.random.default_rng(31)
    ids = pd.Index([f"wt{i}" for i in range(n)])
    rna = ad.AnnData(sparse.csr_matrix(rng.normal(size=(n, features))), obs=pd.DataFrame(index=ids))
    atac = ad.AnnData(sparse.csr_matrix(rng.normal(size=(n, features + 2))), obs=pd.DataFrame(index=ids))
    pt = pd.DataFrame({"pseudotime": np.tile(np.linspace(0, 1, n//2), 2),
                       "branch": np.repeat(["a", "b"], n//2)}, index=ids)
    cfg = {"pseudotime": {"n_bins": 6, "min_cells_per_bin": 5,
            "adaptive_binning": {"enabled": True}}, "granger": {"max_lag": 1}}
    return rna, atac, pt, cfg


def context(rna, atac, pt, cfg):
    embedding, metadata = prepare_wt_embedding(rna, atac, {})
    return fit_context_binning(rna, atac, pt, cfg, embedding=embedding, embedding_metadata=metadata)


def test_require_and_strict_order():
    rna, atac, pt, cfg = inputs()
    require_bin_plan({}, None)
    with pytest.raises(ValueError, match="required"):
        require_bin_plan(cfg, None)
    ctx = context(rna, atac, pt, cfg)
    require_bin_plan(cfg, ctx.plan)
    validate_bin_plan(ctx.plan, pt, rna.obs_names)
    for ids in (rna.obs_names[::-1], ["duplicate"] * len(pt)):
        with pytest.raises(ValueError):
            validate_bin_plan(ctx.plan, pt, ids)
    with pytest.raises(ValueError):
        validate_bin_plan(ctx.plan, pt.iloc[::-1], rna.obs_names)
    with pytest.raises(TypeError):
        validate_bin_plan(None, pt, rna.obs_names)


@pytest.mark.parametrize("column", ["pseudotime", "bin", "branch"])
def test_mutated_metadata_rejected(column):
    rna, atac, pt, cfg = inputs()
    ctx = context(rna, atac, pt, cfg)
    changed = pt.copy()
    changed.loc[changed.index[0], column] = "wrong" if column == "branch" else 99
    with pytest.raises(ValueError, match=column):
        validate_bin_plan(ctx.plan, changed, rna.obs_names)
    changed = pt.copy()
    changed.attrs["branch_boundaries"] = ((0, 99),)
    with pytest.raises(ValueError, match="branch_boundaries"):
        validate_bin_plan(ctx.plan, changed, rna.obs_names)


def test_embedding_sparse_feature_cap_and_equal_blocks(monkeypatch):
    rna, atac, _, _ = inputs(n=40, features=600)
    original = sparse.csr_matrix.toarray
    calls = []
    def guarded(self, *args, **kwargs):
        assert self.shape[1] <= 512, "full sparse input densified"
        calls.append(self.shape)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(sparse.csr_matrix, "toarray", guarded)
    embedding, metadata = prepare_wt_embedding(rna, atac, {})
    assert calls == [(40, 512), (40, 512)]
    assert embedding.shape == (40, 10)
    assert metadata["dense_elements"] == 40960
    for block in (embedding.iloc[:, :5], embedding.iloc[:, 5:]):
        assert np.mean(np.sum(block.to_numpy()**2, axis=1)) == pytest.approx(1)
        np.testing.assert_allclose(block.mean(axis=0), 0, atol=1e-14)
    calls.clear()
    with pytest.raises(MemoryError, match="max_dense_elements"):
        prepare_wt_embedding(rna, atac, {"max_dense_elements": 100})
    assert not calls


def test_alignment_constant_and_rank():
    rna, atac, _, _ = inputs()
    with pytest.raises(ValueError, match="order"):
        prepare_wt_embedding(rna, atac[::-1], {})
    atac.obs_names = ["same"] * atac.n_obs
    with pytest.raises(ValueError, match="unique"):
        prepare_wt_embedding(rna, atac, {})
    rna, atac, _, _ = inputs()
    rna.X = sparse.csr_matrix(np.ones(rna.shape))
    with pytest.raises(ValueError, match="constant"):
        prepare_wt_embedding(rna, atac, {})
    rna.X = sparse.csr_matrix(np.outer(np.arange(rna.n_obs), np.arange(rna.n_vars)))
    embedding, metadata = prepare_wt_embedding(rna, atac, {})
    assert metadata["modalities"]["rna"]["n_components"] == 1
    assert metadata["modalities"]["rna"]["column_slice"] == [0, 1]
    assert metadata["modalities"]["atac"]["column_slice"] == [1, 6]
    pd.testing.assert_frame_equal(select_state_embedding(embedding, metadata, "rna"), embedding.iloc[:, :1])
    assert embedding.shape[1] == 6


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("value", [True, False, None, 1, [], "RNA", "atac", "unknown"])
def test_state_modality_strict_validation(enabled, value):
    with pytest.raises(ValueError, match="state_modality"):
        require_bin_plan({"pseudotime": {"adaptive_binning": {
            "enabled": enabled, "state_modality": value}}}, object())


@pytest.mark.parametrize("damage", ["missing", "gap", "overlap", "count", "width", "bool"])
def test_rna_requires_exact_block_metadata(damage):
    rna, atac, _, _ = inputs()
    embedding, metadata = prepare_wt_embedding(rna, atac, {})
    assert select_state_embedding(embedding, {}, "joint") is embedding
    if damage == "missing":
        metadata = {}
    elif damage == "gap":
        metadata["modalities"]["atac"]["column_slice"][0] += 1
    elif damage == "overlap":
        metadata["modalities"]["rna"]["column_slice"][0] = -1
    elif damage == "count":
        metadata["modalities"]["rna"]["n_components"] = 2
    elif damage == "width":
        embedding = embedding.iloc[:, :-1]
    else:
        metadata["modalities"]["rna"]["column_slice"][0] = False
    with pytest.raises(ValueError, match="column_slice"):
        select_state_embedding(embedding, metadata, "rna")


def test_rna_state_invariant_to_atac_with_joint_control():
    rna, atac, pt, cfg = inputs()
    # Fixed times and exact prepared RNA block; replace only the ATAC PCs.
    embedding, metadata = prepare_wt_embedding(rna, atac, {"n_components": 2})
    changed = embedding.copy()
    changed.iloc[:, 2:] = np.where((np.arange(len(pt)) % 120)[:, None] < 60, -100., 100.)
    fitted = {}
    for mode in ("rna", "joint"):
        cfg["pseudotime"]["adaptive_binning"]["state_modality"] = mode
        fitted[mode] = [fit_context_binning(rna, atac, pt.copy(), cfg,
            embedding=e, embedding_metadata=metadata) for e in (embedding, changed)]
    a, b = fitted["rna"]
    assert a.policy.fingerprint == b.policy.fingerprint
    np.testing.assert_array_equal(a.plan.bandwidths, b.plan.bandwidths)
    for key in a.plan.bandwidth_diagnostics:
        np.testing.assert_array_equal(a.plan.bandwidth_diagnostics[key], b.plan.bandwidth_diagnostics[key])
    from CausalBridge.binning_adapter import _primitive
    assert _primitive(a.policy.boundary_diagnostics) == _primitive(b.policy.boundary_diagnostics)
    assert a.metadata["state_embedding"]["state"] == b.metadata["state_embedding"]["state"]
    assert a.metadata["state_embedding"]["joint"] != b.metadata["state_embedding"]["joint"]
    assert fitted["joint"][0].policy.fingerprint != fitted["joint"][1].policy.fingerprint
    assert not np.array_equal(fitted["joint"][0].plan.bandwidth_diagnostics["score"],
                              fitted["joint"][1].plan.bandwidth_diagnostics["score"], equal_nan=True)
    for data in (rna, atac):
        assert aggregate_with_plan(a.plan, data.X, cell_ids=data.obs_names).shape == (a.plan.n_bins, data.n_vars)


@pytest.mark.parametrize("settings", [{"mystery": 1}, {"max_features": 513},
                                      {"random_seed": 43}, {"n_components": 0}])
def test_unknown_or_unbounded_embedding(settings):
    rna, atac, _, _ = inputs()
    with pytest.raises(ValueError):
        prepare_wt_embedding(rna, atac, settings)


def test_unknown_selector_and_shuffle_no_mutation():
    rna, atac, pt, cfg = inputs()
    embedding, metadata = prepare_wt_embedding(rna, atac, {})
    for mutate in (lambda c: c["pseudotime"]["adaptive_binning"].update(unknown=True),
                   lambda c: c["pseudotime"]["adaptive_binning"].update(embedding={"unknown": 1}),
                   lambda c: c["pseudotime"].update(ablation_shuffle_pseudotime=True)):
        bad = deepcopy(cfg)
        mutate(bad)
        with pytest.raises(ValueError):
            fit_context_binning(rna, atac, pt, bad, embedding=embedding, embedding_metadata=metadata)
        assert "bin" not in pt


def test_context_rejects_invalid_support_and_embedding_order():
    rna, atac, pt, cfg = inputs(n=8)
    embedding, metadata = prepare_wt_embedding(rna, atac, {})
    with pytest.raises(ValueError, match="invalid bins"):
        fit_context_binning(rna, atac, pt, cfg, embedding=embedding, embedding_metadata=metadata)
    assert "bin" not in pt
    rna, atac, pt, cfg = inputs()
    embedding, metadata = prepare_wt_embedding(rna, atac, {})
    with pytest.raises(ValueError, match="embedding IDs"):
        fit_context_binning(rna, atac, pt, cfg, embedding=embedding.iloc[::-1], embedding_metadata=metadata)
    assert "bin" not in pt


def test_low_reliability_warns_without_imputation():
    rna, atac, pt, cfg = inputs()
    cfg["pseudotime"]["adaptive_binning"]["target_ess"] = 1e9
    with pytest.warns(RuntimeWarning, match="low-reliability"):
        ctx = context(rna, atac, pt, cfg)
    assert ctx.plan.valid.all() and not ctx.plan.reliable.any()
    assert np.isfinite(aggregate_with_plan(ctx.plan, rna.X, cell_ids=rna.obs_names)).all()


def test_context_shared_plan_attrs_determinism_and_frozen_query():
    rna, atac, pt, cfg = inputs()
    original = deepcopy(cfg)
    ctx = context(rna, atac, pt, cfg)
    repeat = context(rna, atac, pt.copy(), cfg)
    assert cfg == original
    assert ctx.plan.valid.all() and ctx.plan.cell_valid.all()
    assert ctx.policy.fingerprint == repeat.policy.fingerprint
    assert ctx.model_contract_fingerprint == repeat.model_contract_fingerprint
    assert pt.attrs["branch_boundaries"] == ctx.policy.branch_boundaries
    json.dumps(ctx.metadata, allow_nan=False)
    json.dumps(pt.attrs, allow_nan=False)
    np.testing.assert_array_equal(pt["bin"], ctx.plan.labels)
    for data in (rna, atac):
        output = aggregate_with_plan(ctx.plan, data.X, cell_ids=data.obs_names)
        assert output.shape == (ctx.plan.n_bins, data.n_vars)
        assert np.isfinite(output).all()
    frozen = build_plan(ctx.policy, [.2, .8], ["q1", "q2"], ["a", "b"])
    np.testing.assert_array_equal(frozen.bandwidths, ctx.plan.bandwidths)
    assert frozen.policy_fingerprint == ctx.policy.fingerprint
    with pytest.raises(ValueError, match="invalid bins"):
        aggregate_with_plan(frozen, np.ones((2, 1)), cell_ids=frozen.cell_ids)
    invalid = replace(ctx.plan, cell_valid=np.zeros(len(pt), bool))
    with pytest.raises(ValueError, match="uncovered"):
        aggregate_with_plan(invalid, rna.X, cell_ids=rna.obs_names)
    with pytest.raises(ValueError):
        aggregate_with_plan(ctx.plan, rna.X, cell_ids=rna.obs_names[::-1])


def test_fingerprints_include_data_coordinates_raw_targets_model(tmp_path):
    rna, atac, pt, cfg = inputs()
    targets = tmp_path / "targets.txt"
    targets.write_text("Gene1\n")
    cfg["input"] = {"target_genes": str(targets)}
    baseline = context(rna, atac, pt, cfg)
    changed = deepcopy(cfg)
    changed["granger"]["max_lag"] = 2
    other = context(rna, atac, pt.copy(), changed)
    assert other.data_fingerprint == baseline.data_fingerprint
    assert other.model_contract_fingerprint != baseline.model_contract_fingerprint
    changed = deepcopy(cfg)
    changed["output"] = {"dir": "/not/an/input"}
    assert context(rna, atac, pt.copy(), changed).model_contract_fingerprint == baseline.model_contract_fingerprint
    targets.write_text("Gene2\n")
    assert context(rna, atac, pt.copy(), cfg).model_contract_fingerprint != baseline.model_contract_fingerprint
    for mutate in (lambda a: a.X.data.__setitem__(0, a.X.data[0]+1),
                   lambda a: a.var.__setitem__("start", np.arange(a.n_vars)),
                   lambda a: a.layers.__setitem__("raw", a.X.copy())):
        altered = rna.copy()
        mutate(altered)
        assert context(altered, atac, pt.copy(), cfg).data_fingerprint != baseline.data_fingerprint


def test_diagnostics_explicit_write_only(tmp_path):
    rna, atac, pt, cfg = inputs()
    ctx = context(rna, atac, pt, cfg)
    assert not list(tmp_path.iterdir())
    before = ctx.plan.raw_kernel.copy()
    refs = persist_binning_diagnostics(ctx, tmp_path / "diagnostics")
    payload = json.loads((tmp_path / "diagnostics" / refs["metadata"]).read_text())
    bins = pd.read_csv(tmp_path / "diagnostics" / refs["bins"])
    assert payload["model_contract_fingerprint"] == ctx.model_contract_fingerprint
    assert len(bins) == ctx.plan.n_bins
    assert {"center", "h", "ess", "valid", "state_coverage"} <= set(bins)
    np.testing.assert_array_equal(before, ctx.plan.raw_kernel)
    assert not ctx.plan.raw_kernel.flags.writeable
