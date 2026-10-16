"""Production Granger statistics against independent OLS residual vectors."""
from copy import deepcopy

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import stats
import statsmodels.api as sm

from CausalBridge import granger


def run_pair(monkeypatch, y, x, *, lag0=False, soft=False):
    """Supply known binned signals while exercising the real statistical path."""
    n = len(y)
    ids = pd.Index([f"c{i}" for i in range(n)])
    coords = {"chr": ["chr1"], "start": [100], "end": [200]}
    rna = ad.AnnData(np.asarray(y)[:, None], obs=pd.DataFrame(index=ids),
                     var=pd.DataFrame(coords, index=["gene"]))
    atac = ad.AnnData(np.asarray(x)[:, None], obs=pd.DataFrame(index=ids),
                      var=pd.DataFrame(coords, index=["peak"]))
    pt = pd.DataFrame({"pseudotime": np.linspace(0, 1, n)}, index=ids)
    cfg = {"pseudotime": {"n_bins": n, "min_cells_per_bin": 1},
           "granger": {"distance_thresh": 1000, "ablation_lag0": lag0,
                       "max_lag": 1, "correction_method": "fdr_bh",
                       "significance_threshold": .05, "min_effect_size": .1,
                       "reference_bins": 50, "soft_threshold": soft,
                       "min_composite_score": .3, "composite_weights": [.4, .6]}}
    before = deepcopy(cfg)
    monkeypatch.setattr(granger, "_adaptive_rebin", lambda *a, **k: (np.arange(n), n))
    monkeypatch.setattr(granger, "_bin_expression", lambda data, *a, **k: data.X)
    monkeypatch.setattr(granger, "_generate_candidate_pairs", lambda *a: [("peak", "gene")])
    result = granger.granger_test(rna, atac, pt, cfg)
    assert cfg == before
    return result, cfg["granger"]


@pytest.mark.parametrize("lag0", [False, True])
@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize("noise", [.8, .02])
def test_actual_granger_matches_independent_ols(monkeypatch, lag0, soft, noise):
    rng = np.random.default_rng(7421)
    n = 41
    x = rng.normal(size=n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = .5 + .3*y[i-1] + .7*x[i if lag0 else i-1] + noise*rng.normal()
    result, cfg = run_pair(monkeypatch, y, x, lag0=lag0, soft=soft)
    assert len(result) == 1
    target = y if lag0 else y[1:]
    reduced_design = np.ones((len(target), 1))
    if not lag0:
        reduced_design = np.column_stack([reduced_design, y[:-1]])
    full_design = np.column_stack([reduced_design, x if lag0 else x[:-1]])
    reduced = sm.OLS(target, reduced_design).fit()
    full = sm.OLS(target, full_design).fit()
    # Independently calculate SSE from residual vectors, then compare statsmodels.
    sr = float(reduced.resid @ reduced.resid)
    sf = float(full.resid @ full.resid)
    df = len(target) - full_design.shape[1]
    expected_f = (sr - sf) / (sf / df)
    expected_p = stats.f.sf(expected_f, 1, df)
    np.testing.assert_allclose(full.compare_f_test(reduced)[:2], [expected_f, expected_p], rtol=1e-10)
    row = result.iloc[0]
    np.testing.assert_allclose(row.F_statistic, expected_f, rtol=1e-10)
    np.testing.assert_allclose([row.p_value, row.p_adj], expected_p, rtol=1e-10, atol=0)
    effect = 1 - sf/sr
    np.testing.assert_allclose(row.delta_r2, effect, rtol=1e-10)
    np.testing.assert_allclose(row.atac_coef, full.params[-1], rtol=1e-10)
    # Preserve the historical df correction even for the lag-0 ablation.
    correction = ((cfg["reference_bins"] - 3) / max(len(target) - 3, 1)) ** 2
    sig = cfg["significance_threshold"] / correction
    effect_threshold = cfg["min_effect_size"] * correction
    score = .4*np.clip(1-expected_p/sig, 0, 1) + .6*np.clip(effect/effect_threshold, 0, 2)/2
    np.testing.assert_allclose(row.composite_score, score, rtol=1e-10)
    assert bool(row.is_causal) == (score >= .3 if soft else expected_p < sig and effect > effect_threshold)
    assert set(result) == {"peak_id", "gene", "peak_chr", "peak_start", "peak_end",
                           "gene_chr", "gene_tss", "F_statistic", "p_value", "delta_r2",
                           "atac_coef", "pearson_r", "p_adj", "composite_score", "is_causal"}


@pytest.mark.parametrize("model", ["reduced", "full"])
def test_rank_deficient_empty_residuals_still_skip(monkeypatch, model):
    rng = np.random.default_rng(12)
    y = rng.normal(size=30)
    x = rng.normal(size=30)
    if model == "reduced":
        # Constant lagged RNA but a varying final target: reduced design rank 1.
        y = np.ones(30)
        y[-1] = 2
    else:
        x[:-1] = y[:-1]  # Full ATAC column duplicates lagged RNA.
    result, _ = run_pair(monkeypatch, y, x)
    assert result.empty


@pytest.mark.parametrize("lag0", [False, True])
def test_existing_sse_cutoff_still_skips_near_perfect_fit(monkeypatch, lag0):
    rng = np.random.default_rng(63)
    x = rng.normal(size=30)
    y = np.zeros(30)
    if lag0:
        y = 1 + 2*x
    else:
        for i in range(1, 30):
            y[i] = 1 + .2*y[i-1] + 2*x[i-1]
    result, _ = run_pair(monkeypatch, y, x, lag0=lag0)
    assert result.empty
