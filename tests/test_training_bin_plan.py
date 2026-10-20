"""Real aggregation contracts at each training entry point, without NN/motif work."""
from copy import deepcopy
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from CausalBridge import granger, kinetics
from CausalBridge.binning_adapter import aggregate_with_plan, fit_context_binning, prepare_wt_embedding
from CausalBridge.adaptive_binning import build_plan


def training_inputs():
    rng = np.random.default_rng(91)
    n = 200
    ids = pd.Index([f"c{i}" for i in range(n)])
    rna = ad.AnnData(sparse.csr_matrix(rng.poisson(4, (n, 2))),
                     obs=pd.DataFrame(index=ids),
                     var=pd.DataFrame({"chr": ["chr1"]*2, "start": [110, 210], "end": [120, 220]},
                                      index=["TF1", "Gene2"]))
    atac = ad.AnnData(sparse.csr_matrix(rng.binomial(1, .4, (n, 2))),
                      obs=pd.DataFrame(index=ids),
                      var=pd.DataFrame({"chr": ["chr1"]*2, "start": [100, 200], "end": [130, 230]},
                                       index=["p1", "p2"]))
    pt = pd.DataFrame({"pseudotime": np.tile(np.linspace(0, 1, n//2), 2),
                       "branch": np.repeat(["a", "b"], n//2)}, index=ids)
    cfg = {"pseudotime": {"n_bins": 20, "min_cells_per_bin": 5, "smooth_kernel": "gaussian",
            "adaptive_binning": {"enabled": True}},
           "granger": {"distance_thresh": 1000, "max_lag": 1, "correction_method": "fdr_bh",
                       "significance_threshold": .05, "min_effect_size": .01},
           "atac_to_rna": {"model_mode": "saturating_hybrid", "log_normalize_rna": True},
           "rna_to_atac": {"max_lag": "auto", "tf_expression_filter": False}}
    embedding, metadata = prepare_wt_embedding(rna, atac, {})
    ctx = fit_context_binning(rna, atac, pt, cfg, embedding=embedding, embedding_metadata=metadata)
    edges = pd.DataFrame({"peak_id": ["p1"], "gene": ["TF1"], "is_causal": [True]})
    # The supplied plan, not a later nominal config bin count, controls training.
    cfg["pseudotime"]["n_bins"] = 999
    return rna, atac, pt, cfg, ctx, edges


def forbid_legacy(monkeypatch, n_cells):
    def forbidden(*args, **kwargs):
        pytest.fail("legacy binning/kernel helper was called with a supplied plan")
    monkeypatch.setattr(granger, "_adaptive_rebin", forbidden)
    monkeypatch.setattr(kinetics, "_adaptive_rebin", forbidden)
    monkeypatch.setattr(kinetics, "_bin_matrix", forbidden)
    original = sparse.csr_matrix.toarray
    def bounded(self, *args, **kwargs):
        assert self.shape[0] < n_cells, "full cell matrix densified"
        return original(self, *args, **kwargs)
    monkeypatch.setattr(sparse.csr_matrix, "toarray", bounded)




def test_real_granger_plan_candidates_and_no_rebin(monkeypatch):
    rna, atac, pt, cfg, ctx, _ = training_inputs()
    before = pt.copy(deep=True)
    attrs = deepcopy(pt.attrs)
    expected = aggregate_with_plan(ctx.plan, rna.X, cell_ids=rna.obs_names)
    forbid_legacy(monkeypatch, rna.n_obs)
    np.testing.assert_allclose(granger._bin_expression(rna, pt, ctx.plan.n_bins, bin_plan=ctx.plan), expected)
    result = granger.granger_test(rna, atac, pt, cfg, bin_plan=ctx.plan)
    assert len(result) == 4
    assert {"peak_id", "gene", "peak_chr", "peak_start", "peak_end", "gene_chr", "gene_tss",
            "F_statistic", "p_value", "p_adj", "delta_r2", "is_causal"} <= set(result)
    assert set(zip(result.peak_id, result.gene)) == {("p1", "TF1"), ("p1", "Gene2"),
                                                           ("p2", "TF1"), ("p2", "Gene2")}
    pd.testing.assert_frame_equal(pt, before)
    assert pt.attrs == attrs






@pytest.mark.parametrize("entry", ["granger", "atac_rna", "rna_atac"])
@pytest.mark.parametrize("error", ["missing", "axis", "labels", "times"])
def test_entry_contract_errors_before_training(entry, error, monkeypatch):
    rna, atac, pt, cfg, ctx, edges = training_inputs()
    plan = ctx.plan
    if error == "missing":
        plan = None
    elif error == "axis":
        atac = atac[::-1].copy()
    elif error == "labels":
        pt.loc[pt.index[0], "bin"] = 99
    else:
        pt.loc[pt.index[0], "pseudotime"] = 99.
    forbid_legacy(monkeypatch, rna.n_obs)
    with pytest.raises(ValueError):
        if entry == "granger":
            granger.granger_test(rna, atac, pt, cfg, bin_plan=plan)
        elif entry == "atac_rna":
            kinetics.fit_atac_to_rna(rna, atac, pt, edges, cfg, bin_plan=plan)
        else:
            kinetics.fit_rna_to_atac(rna, atac, pt, edges, cfg, tf_model_registry={}, bin_plan=plan)


def test_bin_expression_rejects_wrong_count():
    rna, _, pt, _, ctx, _ = training_inputs()
    with pytest.raises(ValueError, match="n_bins"):
        granger._bin_expression(rna, pt, ctx.plan.n_bins+1, bin_plan=ctx.plan)




def test_deferred_pseudotime_uses_aligned_data_embedding(monkeypatch):
    rna, atac, _, cfg, _, _ = training_inputs()
    embedding, _ = prepare_wt_embedding(rna, atac, {})
    cfg["pseudotime"].update(method="dpt", root_cell_method="marker", root_cell_marker="TF1")
    def forbidden(*args, **kwargs):
        pytest.fail("dense joint builder or legacy rebin called")
    monkeypatch.setattr(granger, "_build_joint_representation", forbidden)
    monkeypatch.setattr(granger, "_adaptive_rebin", forbidden)
    def dispatch(method, r, joint, pcfg, config):
        assert method == "dpt" and pcfg["root_cell_method"] == "marker"
        np.testing.assert_allclose(joint, embedding.to_numpy())
        assert granger._find_root_cell(r, "TF1", method="marker") == int(np.argmax(np.asarray(r.X[:, 0].todense()).ravel()))
        return np.linspace(0, 1, r.n_obs), np.repeat(["a", "b"], r.n_obs//2)
    monkeypatch.setattr(granger, "_dispatch_pseudotime_method", dispatch)
    result = granger.infer_pseudotime(rna, atac, cfg, defer_binning=True, joint_repr=embedding)
    assert result.index.equals(rna.obs_names)
    assert "bin" not in result and set(result.branch) == {"a", "b"}
    assert result.attrs["branch_boundaries"] is None
    with pytest.raises(ValueError, match="order"):
        granger.infer_pseudotime(rna, atac[::-1], cfg, defer_binning=True, joint_repr=embedding)
    with pytest.raises(ValueError, match="order"):
        granger.infer_pseudotime(rna, atac, cfg, defer_binning=True, joint_repr=embedding.iloc[::-1])
    with pytest.raises(ValueError, match="finite"):
        granger.infer_pseudotime(rna, atac, cfg, defer_binning=True, joint_repr=np.full((rna.n_obs, 2), np.nan))


def test_no_attrs_granger_actual_lag_pairs_use_plan_not_frame(monkeypatch):
    rna, atac, pt, cfg, ctx, _ = training_inputs()
    pt.attrs.clear()
    original = pt.copy(deep=True)
    bounds = ctx.policy.branch_boundaries
    assert bounds == ((0, 10), (10, 20))
    lag_pairs = granger._branch_aware_lag_pairs
    captured = []
    def pairs(x, y, *, lag, branch_boundaries):
        assert branch_boundaries == bounds
        left, right = lag_pairs(x, y, lag=lag, branch_boundaries=branch_boundaries)
        expected_left = np.concatenate([x[a:b-1] for a, b in bounds])
        expected_right = np.concatenate([y[a+1:b] for a, b in bounds])
        np.testing.assert_allclose(left, expected_left)
        np.testing.assert_allclose(right, expected_right)
        assert len(left) == len(right) == 18  # global slicing would produce 19
        captured.append(True)
        return left, right
    monkeypatch.setattr(granger, "_branch_aware_lag_pairs", pairs)
    result = granger.granger_test(rna, atac, pt, cfg, bin_plan=ctx.plan)
    assert len(captured) == 2 and not result.empty
    pd.testing.assert_frame_equal(pt, original)
    assert pt.attrs == {}




def test_no_attrs_tf_actual_regression_rows_and_root_reuse(monkeypatch):
    rna, atac, pt, cfg, ctx, edges = training_inputs()
    pt.attrs.clear()
    original = pt.copy(deep=True)
    cfg["rna_to_atac"]["max_lag"] = 1
    r_bins = aggregate_with_plan(ctx.plan, rna.X, cell_ids=rna.obs_names)
    a_bins = aggregate_with_plan(ctx.plan, atac.X, cell_ids=atac.obs_names)
    monkeypatch.setattr(kinetics, "_load_motif_database", lambda *args: {})
    monkeypatch.setattr(kinetics, "_scan_peaks_with_jaspar", lambda *args: {"p1": ["TF1"]})
    calls = []
    # Each target bin tests lag 0 then lag 1, using valid pairs from its own
    # branch. These ten-bin branches fit entirely inside the 12-row window.
    expected_pairs = []
    for start, end in ctx.policy.branch_boundaries:
        for target in range(start, end):
            for lag in range(min(1, target-start)+1):
                times = np.array([b for b in range(start+lag, end)
                    if ctx.plan.valid[b] and ctx.plan.valid[b-lag]])
                expected_pairs.append((r_bins[times-lag, 0], a_bins[times, 0]))
    def pearson(x, y):
        expected_x, expected_y = expected_pairs[len(calls)]
        assert len(x) == len(y) == len(expected_x)
        np.testing.assert_allclose(x, expected_x)
        np.testing.assert_allclose(y, expected_y)
        calls.append(True)
        return 0., 1.  # force both lag candidates through the real regressor
    monkeypatch.setattr(kinetics, "pearsonr", pearson)
    kinetics.fit_rna_to_atac(rna, atac, pt, edges, cfg, tf_model_registry={}, bin_plan=ctx.plan)
    assert len(calls) == len(expected_pairs) == 38
    pd.testing.assert_frame_equal(pt, original)
    assert pt.attrs == {}


@pytest.mark.parametrize("entry", ["granger", "atac_rna", "rna_atac"])
def test_genuine_unusable_query_plan_rejected_before_stage_work(monkeypatch, entry):
    rna, atac, _, cfg, ctx, edges = training_inputs()
    rna, atac = rna[:2].copy(), atac[:2].copy()
    plan = build_plan(ctx.policy, [.2, .8], rna.obs_names, ["a", "a"])
    assert not plan.valid.all()
    pt = pd.DataFrame({"pseudotime": plan.times, "branch": plan.branches,
                       "bin": plan.labels}, index=rna.obs_names)
    forbid_legacy(monkeypatch, rna.n_obs)
    with pytest.raises(ValueError, match="invalid bins"):
        if entry == "granger":
            granger.granger_test(rna, atac, pt, cfg, bin_plan=plan)
        elif entry == "atac_rna":
            kinetics.fit_atac_to_rna(rna, atac, pt, edges, cfg, bin_plan=plan)
        else:
            kinetics.fit_rna_to_atac(rna, atac, pt, edges, cfg, tf_model_registry={}, bin_plan=plan)


@pytest.mark.parametrize("entry", ["granger", "atac_rna", "rna_atac"])
def test_wrong_present_branch_attrs_are_not_repaired(monkeypatch, entry):
    rna, atac, pt, cfg, ctx, edges = training_inputs()
    pt.attrs["branch_boundaries"] = ((0, 20),)
    forbid_legacy(monkeypatch, rna.n_obs)
    with pytest.raises(ValueError, match="branch_boundaries"):
        if entry == "granger":
            granger.granger_test(rna, atac, pt, cfg, bin_plan=ctx.plan)
        elif entry == "atac_rna":
            kinetics.fit_atac_to_rna(rna, atac, pt, edges, cfg, bin_plan=ctx.plan)
        else:
            kinetics.fit_rna_to_atac(rna, atac, pt, edges, cfg, tf_model_registry={}, bin_plan=ctx.plan)
