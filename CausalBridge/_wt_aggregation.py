"""Shared frozen-WT aggregation, independent of response family."""
import numpy as np
from scipy import sparse


def validate_model_mode(config):
    mode = config.get("atac_to_rna", {}).get("model_mode", "saturating_hybrid")
    if mode == "joint_gene":
        raise ValueError("joint_gene mode has been removed; use saturating_hybrid with response_model=spline")
    if mode != "saturating_hybrid":
        raise ValueError("atac_to_rna.model_mode must be saturating_hybrid; legacy_pair_nn has been removed. Refit with response_model=nn, spline, or saturation")
    return mode


def _weights(pseudotime_df, n_bins, config, bin_plan):
    """Recover exact legacy raw kernels without a cell-by-cell identity matrix."""
    times = pseudotime_df["pseudotime"].to_numpy(dtype=float)
    labels = pseudotime_df["bin"].to_numpy(dtype=int)
    if bin_plan is not None:
        from .binning_adapter import validate_bin_plan
        validate_bin_plan(bin_plan, pseudotime_df, pseudotime_df.index)
        return sparse.csr_matrix(bin_plan.raw_kernel, copy=True), np.asarray(bin_plan.centers)
    pcfg = config.get("pseudotime", {})
    mode = pcfg.get("smooth_kernel", "hard")
    if mode == "hard":
        weights = sparse.csr_matrix((np.ones(len(times)), (labels, np.arange(len(times)))), shape=(n_bins, len(times)))
        centers = np.array([np.median(times[labels == b]) if np.any(labels == b) else np.nan for b in range(n_bins)])
        return weights, centers
    if mode != "gaussian":
        raise ValueError("WT aggregation supports hard or gaussian kernels")
    span = times.max() - times.min()
    span = span if span > 0 else 1.
    sigma = span / n_bins * max(1., pcfg.get("min_cells_per_bin", 5) / max(len(times) / n_bins, 1))
    centers = np.linspace(times.min() + span / n_bins / 2, times.max() - span / n_bins / 2, n_bins)
    return sparse.csr_matrix(np.exp(-.5 * ((centers[:, None] - times) / sigma) ** 2)), centers


def _full_wt_aggregates(kernel, atac, rna, bin_plan, *, smooth):
    """Use the plan operator, or unchanged legacy hard/Gaussian means."""
    mass = np.asarray(kernel.sum(axis=1)).ravel()
    valid = mass > 0
    if bin_plan is not None:
        operator = bin_plan.aggregate_weights
    else:
        operator = kernel.copy()
        denominator = mass + (1e-10 if smooth == "gaussian" else 0.)
        operator.data /= np.repeat(denominator, np.diff(operator.indptr))
    results = []
    for matrix in (atac, rna):
        result = operator @ matrix
        results.append(result.toarray() if sparse.issparse(result) else np.asarray(result))
    return *results, valid
