"""
CausalBridge perturbation simulation engine

Performs in silico gene knockout on an already-built causal model.

Core algorithm: per-bin forward propagation (bin_forward)

  Initialization:
    Gene G is knocked out → R_G(t) is zeroed at every pseudotime point
    If G is not a TF → map the knockout onto the affected TFs via co-expression

  Traverse forward along the pseudotime axis through each bin (t = max_lag → n_bins-1):
    [RNA→ATAC] TF(t - lag[tf]) changes → peak(t) accessibility changes (Pearson weight: r·sd(Y)/sd(X))
    [ATAC→RNA] peak(t) changes → gene(t) expression changes (NN difference; falls back to gain × ΔA linear when it fails)

  Key properties:
    - Time lags are governed by TF-specific lags; each edge uses its first significant lag
    - The forward direction + lag naturally guarantee causal ordering (ATAC[t-1]→RNA[t])
    - Feedback is naturally preserved: TF changes affect downstream TFs through ATAC→gene, which then propagate in later bins
"""

import numpy as np
import pandas as pd
from scipy.sparse import issparse
import logging
from collections import deque
from typing import Dict, List, Tuple, Optional

from .granger import _bin_expression
from .kinetics import _adaptive_rebin, _compute_root_atac, _scale_atac, _unscale_atac
from .nn_transfer import _load_nn_from_transfer_models, _nn_predict_delta_r

logger = logging.getLogger(__name__)

RNA_L1_FLOOR = 0.0


def _floor_rna_l1(values):
    """Apply the single physical L1 RNA boundary without affecting propagation."""
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr) & (arr < RNA_L1_FLOOR)
    out = arr.copy()
    out[mask] = RNA_L1_FLOOR
    return out, int(mask.sum())


def _reportable_effect(delta, min_delta: float) -> bool:
    """Whether a simulated effect is a usable value for result tables.

    Invalid endpoints (``NaN``) mean the counterfactual state could not be
    evaluated — not that the effect is zero.  ``abs(nan) < threshold`` is
    ``False``, so a bare magnitude check would silently let them through;
    require an explicit finiteness gate.
    """
    return bool(np.isfinite(delta)) and abs(float(delta)) >= max(float(min_delta), 1e-6)


def _read_nn_rna_log_normalized(transfer_models: dict, config: dict) -> bool:
    """Read NN metadata and reject checkpoints trained with another RNA scale."""
    data = transfer_models.get("data", {})
    if "rna_log_normalized" not in data:
        raise RuntimeError("NN checkpoint lacks rna_log_normalized metadata; rerun Step 5")
    saved = bool(data["rna_log_normalized"])
    current = bool(config.get("atac_to_rna", {}).get("log_normalize_rna", True))
    if saved != current:
        raise RuntimeError(
            "NN checkpoint rna_log_normalized metadata disagrees with current "
            "atac_to_rna.log_normalize_rna; rerun Step 5"
        )
    return saved


def _nn_delta_or_linear(
    predictor, predictor_args: tuple, delta_a: float, gain: float,
    corrected_gain: float = 0.0,
) -> Tuple[float, str]:
    """Use NN prediction when available, otherwise use the edge gain."""
    try:
        delta_r = predictor(*predictor_args)
        if corrected_gain != 0 and delta_r * corrected_gain * delta_a < 0:
            delta_r = -delta_r
        return delta_r, "nn"
    except Exception:
        return gain * delta_a, "linear"


# ============================================================================
def _delta_r_log_to_raw(delta_r_log: float, r_orig_raw: float) -> float:
    """Convert delta_r from log1p space back to raw count space.

    When the NN is trained on log1p(RNA), delta_r = log1p(r_new) - log1p(r_orig);
    it must be converted to r_new - r_orig before it can be accumulated onto
    the raw count matrix.
    """
    r_new = np.expm1(np.log1p(max(r_orig_raw, 0.0)) + delta_r_log)
    return r_new - r_orig_raw


# ============================================================================
# Main entry: batch simulation
# ============================================================================

def propagate_perturbation(
    target_genes: List[str],
    rna_adata,
    atac_adata,
    pseudotime_df,
    granger_edges: pd.DataFrame,
    transfer_functions: pd.DataFrame,
    tf_peak_weights: pd.DataFrame,
    config: dict,
    transfer_models: Optional[dict] = None,
    gene_types: Optional[dict] = None,
    root_atac: Optional[np.ndarray] = None,
) -> dict:
    """
    Batch-simulate knockouts of multiple genes and return a summary result table.

    Perturbation mode: bin_forward — simulate forward bin by bin along the
    pseudotime axis, respecting TF-specific time lags.

    Returns
    -------
    dict
        "perturbation_results": DataFrame (net effect per gene)
        "pathway_edges": DataFrame (per-edge TF→peak→gene contributions)
    """
    if gene_types is None:
        gene_types = {}

    # Transfer models are produced by the NN fitter; linear gains remain the
    # per-edge fallback when an NN prediction is unavailable.
    nn_model = None
    nn_peak_to_idx = None
    nn_gene_to_idx = None
    nn_rna_log_normalized = bool(config.get("atac_to_rna", {}).get("log_normalize_rna", True))
    if transfer_models is not None:
        if transfer_models.get("type") != "nn":
            raise ValueError("Only NN transfer models are supported")
        nn_model = _load_nn_from_transfer_models(transfer_models)
        nn_rna_log_normalized = _read_nn_rna_log_normalized(transfer_models, config)
        nn_peak_to_idx = transfer_models["data"]["peak_to_idx"]
        nn_gene_to_idx = transfer_models["data"]["gene_to_idx"]

    pert_cfg = config.get("perturbation", {})
    pert_mode = pert_cfg.get("perturbation_mode", "bin_forward")

    if pert_mode != "bin_forward":
        logger.warning(f"perturbation_mode='{pert_mode}' is not supported; using 'bin_forward' instead")

    return _propagate_bin_forward(
        target_genes, rna_adata, atac_adata, pseudotime_df,
        granger_edges, transfer_functions, tf_peak_weights,
        config, nn_model=nn_model,
        nn_peak_to_idx=nn_peak_to_idx, nn_gene_to_idx=nn_gene_to_idx,
        nn_rna_log_normalized=nn_rna_log_normalized,
        gene_types=gene_types,
        root_atac=root_atac,
    )

def _safe_index(names: list, key: str) -> Optional[int]:
    """Safely look up the index of a gene name in a list; returns None if not found."""
    try:
        return names.index(key)
    except ValueError:
        return None


def _center_of_mass(expr: np.ndarray, bins: np.ndarray) -> Optional[float]:
    """Compute the center of mass of expression along pseudotime bins (weighted mean bin position).

    Returning None means the gene has no expression in any bin.
    """
    total = expr.sum()
    if total < 1e-10:
        return None
    return float(np.sum(bins * expr) / total)


def _find_proxy_tfs(
    target_gene: str,
    rna_adata,
    tf_peak_weights: pd.DataFrame,
    rna_binned: np.ndarray = None,
    var_names: list = None,
    min_correlation: float = 0.1,
    max_proxy_tfs: int = 10,
    pseudotime_earliness_threshold: float = 0.5,
    rna_dense=None,
) -> Dict[str, float]:
    """
    Find co-expressed proxy TFs for a non-TF target gene.

    When the gene being KO'd is not itself a TF, use co-expression to find the
    TFs affected by it and distribute the KO effect onto those TFs weighted by
    correlation, completing the RNA→TF propagation chain.

    Parameters
    ----------
    target_gene : str
        Name of the gene being KO'd
    rna_adata : AnnData
        RNA expression data (used to compute co-expression)
    tf_peak_weights : DataFrame
        TF→peak weight table; columns must include "tf_gene"
    rna_binned : np.ndarray or None
        Pseudotime-binned expression matrix (n_bins, n_genes), used to judge
        direction. None = do not check pseudotime direction.
    var_names : list or None
        Gene name list corresponding to rna_binned
    min_correlation : float
        Minimum Pearson correlation coefficient threshold
    max_proxy_tfs : int
        Maximum number of proxy TFs to use
    pseudotime_earliness_threshold : float
        Tolerance (in bins) for the pseudotime direction check.
        When a proxy TF's expression center of mass is earlier than the target
        gene's by more than this threshold, the proxy TF may be an upstream
        regulator; KO'ing the target gene should not affect upstream factors
        → exclude it.
        0.0 = exclude on any earliness (default)

    Returns
    -------
    proxy_tfs : dict
        {tf_name: weight} — weights are unnormalized |pearson r|
        (absolute value of the co-expression correlation, no normalization;
        each proxy TF's knockdown magnitude = |r| × ko_strength, so strongly
        correlated TFs are not diluted by weakly correlated ones)
    """
    from scipy.stats import pearsonr

    if tf_peak_weights.empty or "tf_gene" not in tf_peak_weights.columns:
        return {}

    # Fetch all known TFs
    all_tfs = sorted(tf_peak_weights["tf_gene"].unique())

    if target_gene not in rna_adata.var_names:
        return {}

    # Extract the target gene's expression vector
    if issparse(rna_adata.X):
        target_expr = rna_adata[:, target_gene].X.toarray().flatten()
    else:
        target_expr = np.array(rna_adata[:, target_gene].X).flatten()

    if np.std(target_expr) < 1e-8:
        logger.warning(f"  {target_gene} has no expression variance; cannot compute co-expression")
        return {}

    # Densify up front (avoids per-TF sparse slicing overhead)
    if rna_dense is not None:
        pass  # already supplied by the caller; reuse it
    elif issparse(rna_adata.X):
        rna_dense = rna_adata.X.toarray()
    else:
        rna_dense = np.array(rna_adata.X)

    gene_to_idx = {g: i for i, g in enumerate(rna_adata.var_names)}
    correlations = []

    for tf in all_tfs:
        if tf == target_gene:
            continue
        idx = gene_to_idx.get(tf)
        if idx is None:
            continue
        tf_expr = rna_dense[:, idx]
        if np.std(tf_expr) < 1e-8:
            continue
        corr, _ = pearsonr(target_expr, tf_expr)
        if abs(corr) >= min_correlation:
            correlations.append((tf, corr))

    if not correlations:
        logger.warning(f"  {target_gene}: no co-expressed TF met the threshold")
        return {}

    # --- Upstream exclusion: based on pseudotime direction ---
    # A proxy TF's expression center of mass is earlier than the target gene's
    # → the proxy TF is expressed before the target gene in pseudotime
    # → it may be an upstream regulator, and KO'ing the target gene should not
    #   affect upstream factors → exclude it
    excluded_upstream = []
    if rna_binned is not None and var_names is not None:
        bins = np.arange(rna_binned.shape[0])
        # The target gene's pseudotime expression center of mass
        target_idx = _safe_index(var_names, target_gene)
        if target_idx is not None:
            target_expr_bin = rna_binned[:, target_idx]
            target_com = _center_of_mass(target_expr_bin, bins)
            if target_com is not None:
                filtered = []
                for tf, corr in correlations:
                    tf_idx = _safe_index(var_names, tf)
                    if tf_idx is None:
                        filtered.append((tf, corr))
                        continue
                    tf_expr_bin = rna_binned[:, tf_idx]
                    tf_com = _center_of_mass(tf_expr_bin, bins)
                    if tf_com is None:
                        filtered.append((tf, corr))
                        continue
                    earliness = target_com - tf_com  # >0 = TF is earlier than the target gene
                    if earliness > pseudotime_earliness_threshold:
                        excluded_upstream.append(
                            f"{tf}(Δbin={earliness:.1f})"
                        )
                        continue
                    filtered.append((tf, corr))
                if excluded_upstream:
                    logger.info(
                        f"  {target_gene}: excluded {len(excluded_upstream)} upstream proxy TFs — "
                        f"{', '.join(excluded_upstream)}"
                    )
                correlations = filtered

    if not correlations:
        logger.warning(f"  {target_gene}: all candidate proxy TFs were excluded as upstream")
        return {}

    # Sort by |r|, take the top-N
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    correlations = correlations[:max_proxy_tfs]

    # Use |r| directly as the weight, without normalization.
    # Each proxy TF's knockdown magnitude = |r| × ko_strength.
    # Strongly correlated TFs are not diluted by weakly correlated TFs.
    proxy_tfs = {tf: abs(corr) for tf, corr in correlations}

    logger.info(
        f"  {target_gene} is not a TF → found {len(proxy_tfs)} proxy TFs: "
        f"{', '.join(f'{t}({w:.2f})' for t, w in list(proxy_tfs.items())[:5])}"
        + ("..." if len(proxy_tfs) > 5 else "")
    )

    return proxy_tfs


def _active_proxy_seeds(proxy_tf_map: Dict[str, float], rna_pert: np.ndarray,
                        rna_binned_orig: np.ndarray, name_to_idx: dict) -> set:
    """Return proxies whose own perturbation changed at least one bin."""
    seeds = set()
    for tf in proxy_tf_map:
        idx = name_to_idx.get(tf)
        if idx is not None and np.any(rna_pert[:, idx] != rna_binned_orig[:, idx]):
            seeds.add(tf)
    return seeds


def _resolved_ko_seeds(applied_indices: set, rna_var_names: list) -> set:
    """Return names for KO indices that were actually applied."""
    return {rna_var_names[idx] for idx in applied_indices
            if 0 <= idx < len(rna_var_names)}


# ============================================================================
# Subnetwork extraction
# ============================================================================

def _extract_subnetwork(
    target_gene: str,
    granger_edges: pd.DataFrame,
    transfer_functions: pd.DataFrame,
    tf_peak_weights: pd.DataFrame,
    depth: int = 2,
) -> Dict:
    """
    Breadth-first search extracting the target gene's multi-layer regulatory neighborhood.

    Layer 0: the target gene itself
    Layer 1: directly causally linked peaks + the other genes those peaks regulate
    Layer 2+: iteratively expand to deeper peaks and genes
    """
    if granger_edges.empty:
        return {"genes": set(), "peaks": set(), "edges": pd.DataFrame()}

    all_genes = {target_gene}
    all_peaks = set()
    frontier_genes = {target_gene}

    for level in range(depth):
        # Peaks linked to the current layer's genes (two sources):
        # (a) Granger causality: peak→gene (the peak is an upstream cause of the gene)
        # (b) TF→peak weights: TF→peak (the peak is a downstream regulatory target of the TF)
        new_peaks = set(
            granger_edges[granger_edges["gene"].isin(frontier_genes)]["peak_id"]
        )
        if not tf_peak_weights.empty:
            new_peaks |= set(
                tf_peak_weights[tf_peak_weights["tf_gene"].isin(frontier_genes)]["peak_id"]
            )
        new_peaks -= all_peaks

        if not new_peaks:
            break

        all_peaks |= new_peaks

        # Genes regulated by the new peaks + TFs that regulate the new peaks
        new_genes = set()
        if not granger_edges.empty:
            new_genes |= set(
                granger_edges[granger_edges["peak_id"].isin(new_peaks)]["gene"]
            )
        if not tf_peak_weights.empty:
            new_genes |= set(
                tf_peak_weights[tf_peak_weights["peak_id"].isin(new_peaks)]["tf_gene"]
            )
        new_genes -= all_genes
        all_genes |= new_genes
        frontier_genes = new_genes

    sub_edges = granger_edges[
        (granger_edges["peak_id"].isin(all_peaks))
        | (granger_edges["gene"].isin(all_genes))
    ]

    return {"genes": all_genes, "peaks": all_peaks, "edges": sub_edges}


def _validate_subnetwork_depth(value):
    if value == "unlimited":
        return value
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(
            "perturbation.subnetwork_depth must be a positive integer or "
            'the exact string "unlimited"'
        )
    return int(value)


def _validate_unlimited_edge_budget(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive built-in integer")
    return value


def _extract_unlimited_subnetwork(
    seed_genes, transfer_functions, tf_peak_weights, rna_names, atac_names,
    min_r2_threshold=0.3, min_lag_per_peak_filter=True, max_edges=100000,
    max_work_edges=None,
):
    """Build a bounded executable closure with separate work/final budgets.

    Lag selection is deliberately postponed until after the directed closure
    has been found.  In particular, a long-lag edge may be the only route to
    a TF whose short-lag edge points back to an already discovered peak.
    """
    max_edges = _validate_unlimited_edge_budget(
        max_edges, "subnetwork_unlimited_max_edges"
    )
    if max_work_edges is None:
        # Conservative safety margin; this is not the final retained edge count.
        max_work_edges = max(max_edges * 10, max_edges + 100)
    max_work_edges = _validate_unlimited_edge_budget(
        max_work_edges, "subnetwork_unlimited_max_work_edges"
    )
    rna_names, atac_names = set(rna_names), set(atac_names)

    # Construct both indexes in one pass over validated rows.  TF rows are
    # reduced exactly as simulation does (one row per TF/peak when lagged),
    # while peak→gene keeps duplicates because simulation keeps those rows.
    tf_to_peaks, tf_lags, tf_edge_counts = {}, {}, {}
    tf = tf_peak_weights.copy() if tf_peak_weights is not None else pd.DataFrame()
    if not tf.empty:
        valid_tf = tf[tf["tf_gene"].isin(rna_names) & tf["peak_id"].isin(atac_names)]
        if "tf_lag" in valid_tf.columns:
            valid_tf = valid_tf.assign(_lag=pd.to_numeric(valid_tf["tf_lag"], errors="coerce"))
            valid_tf = valid_tf.dropna(subset=["_lag"])
            for (gene, peak), group in valid_tf.groupby(["tf_gene", "peak_id"], sort=False):
                if "_window_only" in group.columns:
                    main = group[group["_window_only"] != True]
                    if not main.empty:
                        group = main
                row = group.iloc[group["_lag"].values.argmin()]
                edge = (gene, peak)
                tf_to_peaks.setdefault(gene, []).append(peak)
                tf_lags[edge] = float(row["_lag"])
                tf_edge_counts[edge] = 1
        else:
            for row in valid_tf.itertuples(index=False):
                edge = (row.tf_gene, row.peak_id)
                if edge not in tf_edge_counts:
                    tf_to_peaks.setdefault(edge[0], []).append(edge[1])
                    tf_edge_counts[edge] = 0
                tf_edge_counts[edge] += 1

    peak_to_genes = {}
    pg = transfer_functions.copy() if transfer_functions is not None else pd.DataFrame()
    if not pg.empty:
        valid_pg = pg[(pg["r2_score"] >= min_r2_threshold)
                      & pg["peak_id"].isin(atac_names) & pg["gene"].isin(rna_names)]
        for row in valid_pg.itertuples(index=False):
            peak_to_genes.setdefault(row.peak_id, []).append(row.gene)

    all_genes = set(seed_genes) & rna_names
    all_peaks, selected_tf, selected_pg = set(), set(), set()
    processed_tfs, processed_peaks = set(), set()
    queue = deque((gene, 0) for gene in all_genes if gene in tf_to_peaks)
    queued_tfs = set(gene for gene, _ in queue)
    gene_depth = {gene: 0 for gene in queued_tfs}
    peak_depth = {}
    n_pg_rows = 0
    max_hop = 0
    work_edges = 0

    def charge_work():
        nonlocal work_edges
        work_edges += 1
        if work_edges > max_work_edges:
            raise ValueError(
                "unlimited subnetwork traversal/work budget exceeded; scanned "
                f"more than {max_work_edges} candidate rows. This conservative "
                "safety cap is separate from the final retained simulation-edge "
                "cap; increase subnetwork_unlimited_max_work_edges explicitly "
                f"(while subnetwork_unlimited_max_edges remains {max_edges}) "
                "or use finite subnetwork_depth."
            )

    while queue:
        gene, depth = queue.popleft()
        if gene in processed_tfs:
            continue
        processed_tfs.add(gene)
        for peak in tf_to_peaks.get(gene, ()):
            charge_work()
            edge = (gene, peak)
            selected_tf.add(edge)
            if peak not in all_peaks:
                all_peaks.add(peak)
                peak_depth[peak] = depth
                # Processing a peak immediately is safe: FIFO only affects
                # discovery order, not closure membership.
                if peak not in processed_peaks:
                    processed_peaks.add(peak)
                    for downstream in peak_to_genes.get(peak, ()):
                        charge_work()
                        selected_pg.add((peak, downstream))
                        n_pg_rows += 1
                        if downstream not in all_genes:
                            all_genes.add(downstream)
                        if downstream in tf_to_peaks and downstream not in queued_tfs:
                            queued_tfs.add(downstream)
                            gene_depth[downstream] = depth + 1
                            max_hop = max(max_hop, depth + 1)
                            queue.append((downstream, depth + 1))

    # Min-lag is an executable-edge choice, not a property of the complete
    # graph.  A shorter-lag TF which was never reached must not suppress the
    # edge that made this closure reachable (e.g. KO->p lag 2 vs UP->p lag 1).
    # The closure above establishes the reachable endpoint set first; only
    # then apply the same choice the simulator makes to those endpoints.
    reachable_tf_edges = {
        edge for edge in tf_edge_counts
        if edge[0] in all_genes and edge[1] in all_peaks
    }
    if min_lag_per_peak_filter and tf_lags:
        peak_min_lag = {}
        for edge in reachable_tf_edges:
            peak = edge[1]
            peak_min_lag[peak] = min(peak_min_lag.get(peak, tf_lags[edge]), tf_lags[edge])
        selected_tf = {
            edge for edge in reachable_tf_edges
            if tf_lags[edge] == peak_min_lag[edge[1]]
        }
    else:
        selected_tf = reachable_tf_edges
    # Count retained TF rows for every policy, including disabled/absent lag.
    n_tf_rows = sum(tf_edge_counts.get(edge, 1) for edge in selected_tf)
    n_edges = n_tf_rows + n_pg_rows
    if n_edges > max_edges:
        raise ValueError(
            "unlimited subnetwork final edge budget exceeded; selected more than "
            f"{max_edges} edges. Set subnetwork_unlimited_max_edges higher "
            "explicitly or use finite subnetwork_depth."
        )
    logger.info("Unlimited gene-KO subnetwork: %d genes, %d peaks, %d edges, %d work rows, max hop depth=%d",
                len(all_genes), len(all_peaks), n_edges, work_edges, max_hop)
    return {"genes": all_genes, "peaks": all_peaks, "edges": pd.DataFrame(),
            "max_hop_depth": max_hop, "n_edges": n_edges,
            "n_work_edges": work_edges}


# ============================================================================
# Utility functions
# ============================================================================

def _check_atac_mediation(
    affected_gene: str,
    delta_atac: pd.Series,
    granger_edges: pd.DataFrame,
) -> bool:
    """Determine whether a gene change is mediated through ATAC."""
    if delta_atac.abs().max() < 1e-6:
        return False
    gene_peaks = set(granger_edges[granger_edges["gene"] == affected_gene]["peak_id"])
    changed_peaks = set(delta_atac[delta_atac.abs() > 1e-6].index)
    return len(gene_peaks & changed_peaks) > 0


# ============================================================================
# Per-bin forward perturbation simulation (v1.0)
# ============================================================================

def _build_atac_changes_df(delta_atac_list, atac_adata,
                           atac_z_score=None, z_threshold: float = 2.0) -> pd.DataFrame:
    """Convert a list of per-peak delta_atac Series into a BED-ready DataFrame.

    Parameters
    ----------
    delta_atac_list : list of pd.Series or pd.Series
        A single or multiple per-peak change Series
    atac_adata : AnnData
        Used to obtain the chr/start/end coordinates of each peak
    atac_z_score : pd.Series, optional
        Per-peak WT null-distribution z-score (delta / WT std across bins)
    z_threshold : float
        |z| >= z_threshold is considered a significant change

    Returns
    -------
    pd.DataFrame with columns: chr, start, end, peak_id, delta_accessibility
    """
    if isinstance(delta_atac_list, pd.Series):
        delta_series = delta_atac_list
    elif len(delta_atac_list) == 0:
        return pd.DataFrame(columns=["chr", "start", "end", "peak_id", "delta_accessibility"])
    elif len(delta_atac_list) == 1:
        delta_series = delta_atac_list[0]
    else:
        # Multiple KO results: take the largest-magnitude change for each peak
        all_peaks = set()
        for ds in delta_atac_list:
            all_peaks.update(ds.index)
        delta_dict = {}
        for peak_id in all_peaks:
            values = [ds.get(peak_id, 0.0) for ds in delta_atac_list]
            delta_dict[peak_id] = max(values, key=abs)
        delta_series = pd.Series(delta_dict)

    # Filter out zero changes
    delta_series = delta_series[delta_series.abs() > 1e-10]
    if len(delta_series) == 0:
        return pd.DataFrame(columns=["chr", "start", "end", "peak_id", "delta_accessibility"])

    var_df = atac_adata.var[["chr", "start", "end"]]
    result = var_df.loc[var_df.index.isin(delta_series.index)].copy()
    result["peak_id"] = result.index
    result["delta_accessibility"] = result["peak_id"].map(delta_series)

    # Attach the WT null-distribution z-score (if available)
    if atac_z_score is not None:
        result["z_score"] = result["peak_id"].map(atac_z_score).fillna(0.0)
        result["is_significant"] = result["z_score"].abs() >= z_threshold
    return result.reset_index(drop=True)


def _propagate_bin_forward(
    target_genes: List[str],
    rna_adata,
    atac_adata,
    pseudotime_df,
    granger_edges: pd.DataFrame,
    transfer_functions: pd.DataFrame,
    tf_peak_weights: pd.DataFrame,
    config: dict,
    nn_model=None,
    nn_peak_to_idx: Optional[dict] = None,
    nn_gene_to_idx: Optional[dict] = None,
    gene_types: Optional[dict] = None,
    root_atac: Optional[np.ndarray] = None,
    nn_rna_log_normalized: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Batch perturbation simulation in bin_forward mode.

    Supports both single-gene and multi-gene joint knockout. When
    joint_knockout=true, all target_genes are knocked out simultaneously and
    their effects automatically cross within the bin loop.
    """
    if gene_types is None:
        gene_types = {}

    pert_cfg = config.get("perturbation", {})
    joint_ko = pert_cfg.get("joint_knockout", False)
    # Cell-level counterfactuals are also the canonical source for merged L0
    # aggregation.  They must therefore be produced even when the optional
    # projection report is disabled.
    projection_enabled = True

    if joint_ko:
        # Multi-gene joint knockout: simulate all genes in one pass
        logger.info(
            f"Bin-forward joint KO: {len(target_genes)} genes "
            f"({', '.join(target_genes[:5])}{'...' if len(target_genes) > 5 else ''})"
        )
        result = _simulate_knockout_bin_forward(
            target_genes, rna_adata, atac_adata, pseudotime_df,
            granger_edges, transfer_functions, tf_peak_weights,
            config, nn_model=nn_model,
            nn_peak_to_idx=nn_peak_to_idx, nn_gene_to_idx=nn_gene_to_idx,
            nn_rna_log_normalized=nn_rna_log_normalized,
            gene_types=gene_types,
            root_atac=root_atac,
        )

        if "error" in result:
            logger.warning(f"Bin-forward joint KO failed: {result.get('error')}")
            return {
                "perturbation_results": pd.DataFrame(),
                "pathway_edges": pd.DataFrame(),
            }

        delta_rna = result["delta_rna"]
        delta_atac = result["delta_atac"]

        min_delta = pert_cfg.get("min_delta_threshold", 0.0)
        all_results = []
        for affected_gene, delta in delta_rna.items():
            if not _reportable_effect(delta, min_delta):
                continue
            if affected_gene in target_genes:
                continue

            mediated_by_atac = _check_atac_mediation(
                affected_gene, delta_atac, granger_edges
            )
            all_results.append({
                "target_gene": ",".join(result.get("ko_genes_applied", target_genes)),
                "affected_gene": affected_gene,
                "delta_rna": delta,
                "mediated_by_atac": mediated_by_atac,
                "mechanism": "ATAC-mediated" if mediated_by_atac else "direct-interaction",
                # Graph cascade depth is independent of the pseudotime lag.
                "propagation_depth": result.get("gene_cascade_depth", {}).get(
                    affected_gene, np.nan
                ),
                "converged": True,
                "is_fallback": False,
                "perturbation_mode": "bin_forward",
            })

        logger.info(f"Bin-forward joint KO finished: {len(all_results)} effect relations"
                    + (f" (|delta| >= {min_delta:.1f})" if min_delta > 0 else ""))

        output = {
            "perturbation_results": pd.DataFrame(all_results),
            "pathway_edges": result.get("pathway_edges",
                                         pd.DataFrame(columns=["target_gene", "tf_name",
                                                               "peak_id", "affected_gene",
                                                               "delta_r_raw", "delta_r_total",
                                                               "delta_rna_log2fc",
                                                               "mechanism", "r2_score", "tf_lag", "bin"])),
            "aggregated_tf_peak_edges": result.get("aggregated_tf_peak_edges",
                                                    pd.DataFrame()),
            "atac_changes": _build_atac_changes_df(
                delta_atac, atac_adata,
                atac_z_score=result.get("atac_z_score"),
                z_threshold=pert_cfg.get("atac_significance_zscore", 2.0),
            ),
        }
        if result.get("projection_inputs") is not None:
            output["projection_inputs"] = [result["projection_inputs"]]
        if result.get("projection_skip_status"):
            output["projection_skip_status"] = result["projection_skip_status"]
            output["projection_skip_diagnostic"] = result.get("projection_skip_diagnostic", "")
        return output

    # Per-gene independent simulation
    all_results = []
    all_pathways = []
    all_delta_atac = []  # collected per gene, used to build the ATAC changes BED
    all_z_scores = []    # collected per gene for significance annotation
    all_aggregated = []  # collected per gene, aggregated TF→peak edges
    projection_inputs = []
    projection_skips = []
    for i, gene in enumerate(target_genes):
        gtype = gene_types.get(gene, "TF")
        logger.info(
            f"[{i+1}/{len(target_genes)}] Bin-forward KO: {gene} (type={gtype})"
        )

        result = _simulate_knockout_bin_forward(
            [gene], rna_adata, atac_adata, pseudotime_df,
            granger_edges, transfer_functions, tf_peak_weights,
            config, nn_model=nn_model,
            nn_peak_to_idx=nn_peak_to_idx, nn_gene_to_idx=nn_gene_to_idx,
            nn_rna_log_normalized=nn_rna_log_normalized,
            gene_types={gene: gtype},
            root_atac=root_atac,
        )

        if "error" in result:
            continue
        if result.get("projection_inputs") is not None:
            projection_inputs.append(result["projection_inputs"])
        if result.get("projection_skip_status"):
            projection_skips.append({"status": result["projection_skip_status"],
                                     "diagnostic": result.get("projection_skip_diagnostic", "")})

        agg = result.get("aggregated_tf_peak_edges")
        if agg is not None and not agg.empty:
            all_aggregated.append(agg)

        delta_rna = result["delta_rna"]
        delta_atac = result["delta_atac"]

        min_delta = pert_cfg.get("min_delta_threshold", 0.0)
        for affected_gene, delta in delta_rna.items():
            if not _reportable_effect(delta, min_delta) or affected_gene == gene:
                continue

            mediated_by_atac = _check_atac_mediation(
                affected_gene, delta_atac, granger_edges
            )
            all_results.append({
                "target_gene": gene,
                "affected_gene": affected_gene,
                "delta_rna": delta,
                "mediated_by_atac": mediated_by_atac,
                "mechanism": "ATAC-mediated" if mediated_by_atac else "direct-interaction",
                # Graph cascade depth is independent of the pseudotime lag.
                "propagation_depth": result.get("gene_cascade_depth", {}).get(
                    affected_gene, np.nan
                ),
                "converged": True,
                "is_fallback": False,
                "perturbation_mode": "bin_forward",
            })

        # Collect pathway-level edge records
        pathway_edges = result.get("pathway_edges")
        if pathway_edges is not None and not pathway_edges.empty:
            all_pathways.append(pathway_edges)

        # Collect delta_atac for BED export
        all_delta_atac.append(delta_atac)
        zs = result.get("atac_z_score")
        if zs is not None:
            all_z_scores.append(zs)

    logger.info(f"Bin-forward perturbation simulation finished: {len(all_results)} target→affected relations"
                + (f" (|delta| >= {pert_cfg.get('min_delta_threshold', 0):.1f})" if pert_cfg.get("min_delta_threshold", 0) > 0 else ""))
    merged_pathways = pd.concat(all_pathways, ignore_index=True) if all_pathways else pd.DataFrame(
        columns=["target_gene", "tf_name", "peak_id", "affected_gene",
                  "delta_r_raw", "delta_r_total", "delta_rna_log2fc",
                  "mechanism", "r2_score", "tf_lag", "bin"])

    # --- Build the ATAC changes BED-ready DataFrame ---
    # Per-gene mode: merge peak changes across all genes (taking the largest
    # absolute per-peak value)
    # z-scores are likewise collected per gene from the result dicts: take the
    # largest |z| per peak
    _merged_z = None
    if all_z_scores:
        _z_df = pd.DataFrame(all_z_scores).T
        _merged_z = _z_df.abs().idxmax(axis=1) if _z_df.shape[1] > 1 else _z_df.iloc[:, 0]
        _merged_z = _merged_z.apply(
            lambda peak: float(_z_df.loc[peak, _z_df.loc[peak].abs().idxmax()])
        )
    atac_changes = _build_atac_changes_df(
        all_delta_atac, atac_adata,
        atac_z_score=_merged_z,
        z_threshold=pert_cfg.get("atac_significance_zscore", 2.0),
    )

    merged_aggregated = (
        pd.concat(all_aggregated, ignore_index=True).drop_duplicates(
            subset=["tf_gene", "peak_id"]
        ) if all_aggregated else pd.DataFrame()
    )
    output = {
        "perturbation_results": pd.DataFrame(all_results),
        "pathway_edges": merged_pathways,
        "aggregated_tf_peak_edges": merged_aggregated,
        "atac_changes": atac_changes,
    }
    output["projection_inputs"] = projection_inputs
    if projection_skips:
        output["projection_skips"] = projection_skips
    return output


def _record_gene_cascade_depth(
    gene_cascade_depth: Dict[str, int],
    source_depth: Dict[int, int],
    gene_idx: int,
    gene_name: str,
    tf_contribs: Dict[int, float],
    delta_r: float,
) -> None:
    """Record the shortest known TF→peak→gene graph path.

    ``tf_contribs`` is the attribution for the peak value that fired the
    edge.  Contributors without a known source path are intentionally ignored
    so an effect cannot acquire a fabricated depth.
    """
    if not np.isfinite(delta_r) or abs(delta_r) <= 1e-10:
        return

    known_depths = [
        source_depth[tf_idx]
        for tf_idx, tf_contrib in tf_contribs.items()
        if tf_idx in source_depth
        and np.isfinite(tf_contrib)
        and abs(tf_contrib) > 1e-10
    ]
    if not known_depths:
        return

    depth = min(known_depths) + 1
    previous_depth = gene_cascade_depth.get(gene_name)
    if previous_depth is None or depth < previous_depth:
        gene_cascade_depth[gene_name] = depth
    if gene_idx not in source_depth or depth < source_depth[gene_idx]:
        source_depth[gene_idx] = depth


def _accumulate_tf_l1_output_event(
    output_delta_l1: np.ndarray,
    gene_idx: int,
    l1_orig_at_t: float,
    delta_r_l2: float,
    t: int,
    n_bins: int,
) -> None:
    """Accumulate a first-fired executable-TF event for single-round output.

    The simulation remains in its existing state/update units.  This is only
    an output accumulator: the event is decoded from L2 to L1 at its firing
    bin and weighted by the bins for which it is observable thereafter.
    """
    event_delta_l1 = (
        np.expm1(np.log1p(l1_orig_at_t) + delta_r_l2) - l1_orig_at_t
    )
    output_delta_l1[gene_idx] += event_delta_l1 * (n_bins - t) / n_bins


def _single_round_tf_output_means(
    rna_mean_orig: np.ndarray,
    initialization_l1_offset: np.ndarray,
    weighted_event_l1: np.ndarray,
    is_tf: np.ndarray,
    is_ko_target: np.ndarray,
) -> np.ndarray:
    """Build single-round L1 means for non-KO executable TFs only."""
    output = rna_mean_orig.copy()
    eligible = is_tf & ~is_ko_target
    proposed = (
        rna_mean_orig[eligible]
        + initialization_l1_offset[eligible]
        + weighted_event_l1[eligible]
    )
    output[eligible] = np.where(proposed > 0, proposed, np.nan)
    return output


def _projection_operators(pseudotime_df, n_bins, smooth, min_cells):
    """Return normalized bin aggregation and cell membership matrices.

    Both matrices derive from the same hard/Gaussian kernel.  The first is
    row-normalized over cells (the simulation's bin means); the second is
    row-normalized over bins (per-cell reconstruction weights).
    """
    pt = pseudotime_df["pseudotime"].to_numpy(dtype=float)
    bins = pseudotime_df["bin"].to_numpy(dtype=int)
    kernel = np.zeros((n_bins, len(pt)), dtype=float)
    if smooth != "gaussian":
        kernel[bins, np.arange(len(pt))] = 1.0
    else:
        pseudo_min, pseudo_max = pt.min(), pt.max()
        span = pseudo_max - pseudo_min
        if span <= 0:
            span = 1.0
        sigma = span / n_bins
        avg_cells = len(pt) / max(n_bins, 1)
        if avg_cells < min_cells:
            sigma *= min_cells / max(avg_cells, 1.0)
        width = span / n_bins
        centers = np.linspace(pseudo_min + width / 2, pseudo_max - width / 2, n_bins)
        diff = centers[:, None] - pt[None, :]
        kernel = np.exp(-0.5 * (diff / sigma) ** 2)
    bin_operator = kernel / (kernel.sum(axis=1, keepdims=True) + 1e-10)
    membership = kernel.T / (kernel.sum(axis=0, keepdims=True).T + 1e-10)
    return bin_operator, membership


def _projection_membership_weights(pseudotime_df, n_bins, smooth, min_cells):
    return _projection_operators(pseudotime_df, n_bins, smooth, min_cells)[1]


def _simulate_knockout_bin_forward(
    target_genes: List[str],
    rna_adata,
    atac_adata,
    pseudotime_df,
    granger_edges: pd.DataFrame,
    transfer_functions: pd.DataFrame,
    tf_peak_weights: pd.DataFrame,
    config: dict,
    nn_model=None,
    nn_peak_to_idx: Optional[dict] = None,
    nn_gene_to_idx: Optional[dict] = None,
    gene_types: Optional[dict] = None,
    root_atac: Optional[np.ndarray] = None,
    nn_rna_log_normalized: Optional[bool] = None,
) -> Dict:
    """
    Per-bin forward perturbation simulation.

    Traverse each bin forward along the pseudotime axis:
      t = max_lag → n_bins-1:
        [RNA→ATAC] TF(t - lag[tf]) changes → peak(t) accessibility changes (Pearson weight: r·sd(Y)/sd(X))
        [ATAC→RNA] peak(t) changes → gene(t) expression changes (NN or linear gain)

    Properties of the current implementation (bin_forward):
    - Time lags are governed by each edge's TF-specific lag (the first significant lag)
    - The forward direction + lag naturally guarantee causal ordering, no time-domain check needed
    - Feedback is naturally preserved: TF changes affect downstream TFs through ATAC→gene, which then propagate in later bins
    - Multi-gene KO only requires applying all KOs simultaneously at initialization

    Parameters
    ----------
    target_genes : List[str]
        List of knockout target genes (single-gene and multi-gene supported)
    root_atac : np.ndarray, optional
        shape (n_peaks,), per-peak baseline from root cells
    """
    cfg = config["perturbation"]
    ko_strength = cfg.get("ko_strength", 1.0)
    propagation_rounds = cfg.get("propagation_rounds", 1)

    # --- 1. Prepare binned data ---
    pcfg = config.get("pseudotime", {})
    smooth_kernel = pcfg.get("smooth_kernel", "hard")
    # Adaptive binning by cluster size
    _overlap = pcfg.get("smooth_overlap_factor") if smooth_kernel == "gaussian" else None
    bin_indices, n_bins = _adaptive_rebin(
        pseudotime_df, rna_adata.n_obs,
        max_bins=pcfg["n_bins"],
        min_cells_per_bin=pcfg.get("min_cells_per_bin", 5),
        target_bins=pcfg["n_bins"] if smooth_kernel == "gaussian" else None,
        overlap_factor=_overlap,
    )
    pseudotime_df["bin"] = bin_indices

    min_cpb = pcfg.get("min_cells_per_bin", 5)
    rna_binned_orig = _bin_expression(rna_adata, pseudotime_df, n_bins,
                                      smooth=smooth_kernel, min_cells_per_bin=min_cpb)
    atac_binned_orig = _bin_expression(atac_adata, pseudotime_df, n_bins,
                                       smooth=smooth_kernel, min_cells_per_bin=min_cpb)

    rna_var_names = list(rna_adata.var_names)
    atac_var_names = list(atac_adata.var_names)

    # --- [-1,1] scaling: root-cell ATAC as the baseline ---
    cfg_atac = config.get("atac_to_rna", {})
    if root_atac is None:
        root_atac = _compute_root_atac(
            atac_adata, pseudotime_df,
            quantile=cfg_atac.get("root_cell_quantile", 0.05),
            min_count=cfg_atac.get("root_cell_min_count", 5),
            floor=cfg_atac.get("root_atac_floor", 0.05),
        )
    atac_binned_scaled = _scale_atac(atac_binned_orig, root_atac)
    logger.info(
        f"Perturbation [-1,1] scaling: scaled ∈ [{atac_binned_scaled.min():.3f}, "
        f"{atac_binned_scaled.max():.3f}]"
    )

    # Pre-build name→idx mappings to avoid O(n) list.index() scans in the hot loop
    rna_name_to_idx = {name: i for i, name in enumerate(rna_var_names)}
    atac_name_to_idx = {name: i for i, name in enumerate(atac_var_names)}

    rna_pert = rna_binned_orig.copy()
    atac_pert = atac_binned_scaled.copy()

    # --- 2. Apply KO ---
    # Triple screen (statistical credibility check of the complete pathway):
    #   1. The TF is expressed (> 0) in bin t (there is protein available to knock out)
    #   2. At least one TF→peak edge exists whose lag satisfies t + lag < n_bins
    #      (the effect can reach a downstream bin)
    #   3. The target peak of that TF→peak edge has at least one downstream gene
    #      connection (peak→gene edge with R² ≥ threshold)
    #      (a TF→peak edge whose peak has no downstream gene = incomplete pathway, not counted)
    gtype_map = gene_types or {}
    ko_genes_applied = []
    _applied_ko_indices: set = set()

    # Pre-build the peak downstream-connectivity set: only peaks with downstream gene connections count toward the pathway
    min_r2 = cfg.get("min_r2_threshold", 0.3)

    _peaks_with_downstream: set = set()
    if transfer_functions is not None and not transfer_functions.empty:
        for _, row in transfer_functions.iterrows():
            if row.get("r2_score", 0) >= min_r2:
                _peaks_with_downstream.add(row["peak_id"])

    # Pre-extract TF→peak lag info, keeping only edges whose target peak has a downstream gene connection
    _tf_lags: Dict[int, set] = {}  # tf_idx → {lag values}
    _tf_lags_all: Dict[int, set] = {}  # unfiltered lags (kept for log comparison)
    if not tf_peak_weights.empty and "tf_lag" in tf_peak_weights.columns:
        has_window_flag = "_window_only" in tf_peak_weights.columns
        for _, row in tf_peak_weights.iterrows():
            tf_idx = rna_name_to_idx.get(row["tf_gene"])
            if tf_idx is None:
                continue
            # Extension lags (for window validation only) do not participate in KO application or lag statistics
            if has_window_flag and row.get("_window_only", False):
                continue
            lag = int(row["tf_lag"])
            _tf_lags_all.setdefault(tf_idx, set()).add(lag)
            if _peaks_with_downstream and row["peak_id"] not in _peaks_with_downstream:
                continue
            _tf_lags.setdefault(tf_idx, set()).add(lag)

    # Pre-build a case-insensitive mapping for gene alias resolution
    _upper_to_name = {n.upper(): n for n in rna_var_names}
    _seen_ko_indices: set = set()  # prevent the same gene from being KO'd twice (e.g. TAL1/SCL are the same gene)

    for target_gene in target_genes:
        ko_idx = rna_name_to_idx.get(target_gene)
        if ko_idx is None:
            # Case-insensitive lookup (handles aliases/case variants, e.g. SCL=TAL1)
            case_match = _upper_to_name.get(target_gene.upper())
            if case_match:
                logger.warning(
                    f"  {target_gene} is not in the RNA data, but {case_match} was found; "
                    f"substituting {case_match}"
                )
                ko_idx = rna_name_to_idx[case_match]
            else:
                logger.warning(f"  {target_gene} is not in the RNA data; skipping")
                continue

        # Deduplication: the same gene index cannot be KO'd twice
        if ko_idx in _seen_ko_indices:
            logger.warning(
                f"  {target_gene} duplicates an already-processed gene (idx={ko_idx}); skipping"
            )
            continue
        _seen_ko_indices.add(ko_idx)

        ko_expr = rna_binned_orig[:, ko_idx]
        active_mask = np.zeros(n_bins, dtype=bool)
        lags = _tf_lags.get(ko_idx, set())
        lags_all = _tf_lags_all.get(ko_idx, set())
        for t in range(n_bins):
            if ko_expr[t] <= 0:
                continue
            if lags:
                if not any(t + lag < n_bins for lag in lags):
                    continue
            active_mask[t] = True

        n_active = active_mask.sum()
        if n_active == 0:
            tf_name = target_gene
            logger.warning(f"  {tf_name}: no bins satisfy the conditions (expression>0 and t+lag<n_bins); skipping")
            continue
        rna_pert[active_mask, ko_idx] *= (1.0 - ko_strength)
        ko_genes_applied.append(target_gene)
        _applied_ko_indices.add(ko_idx)
        n_lags_all = len(lags_all)
        n_lags_credible = len(lags)
        logger.info(
            f"  {target_gene}: applied KO in {n_active}/{n_bins} bins"
            + (f" (TF→peak lag={sorted(lags)})" if lags else " (no lag restriction)")
        )
        if n_lags_all > n_lags_credible:
            logger.info(
                f"    → pathway-completeness filter: {n_lags_all} TF→peak lags → {n_lags_credible}"
                f" ({n_lags_all - n_lags_credible} lags whose peaks have no downstream gene connection)"
            )

    if not ko_genes_applied:
        return {
            "target_genes": target_genes,
            "ko_genes_applied": [],
            "gene_cascade_depth": {},
            "error": "no genes found in RNA data",
        }

    target_gene_label = ",".join(ko_genes_applied)  # used for pathway records

    # --- 3. Proxy TF mapping (non-TF targets) ---
    # Densify the RNA matrix up front to avoid repeating toarray() for every non-TF target gene
    _rna_dense = None
    all_proxy_tfs: set = set()  # complete proxy TF set, for finite-depth subnetwork expansion
    active_proxy_tfs: set = set()  # only for unlimited closure seed selection
    for target_gene in target_genes:
        gtype = gtype_map.get(target_gene, "TF")
        target_is_tf = (gtype == "TF")

        tf_has_weights = False
        if target_is_tf and not tf_peak_weights.empty:
            tf_has_weights = (
                tf_peak_weights["tf_gene"].str.upper() == target_gene.upper()
            ).any()

        if target_is_tf and not tf_has_weights:
            logger.info(f"  {target_gene} is marked as a TF but has no TF→peak weights; skipping")
            continue

        if not target_is_tf:
            logger.info(f"  {target_gene} is a non-TF; searching for co-expressed proxy TFs...")

            if _rna_dense is None:
                from scipy.sparse import issparse as _issparse
                _rna_dense = rna_adata.X.toarray() if _issparse(rna_adata.X) else np.array(rna_adata.X)

            proxy_tf_map = _find_proxy_tfs(
                target_gene, rna_adata, tf_peak_weights,
                rna_binned=rna_binned_orig,
                var_names=rna_var_names,
                min_correlation=cfg.get("proxy_tf_min_correlation", 0.1),
                max_proxy_tfs=cfg.get("proxy_tf_max_tfs", 10),
                pseudotime_earliness_threshold=cfg.get("proxy_pseudotime_earliness_threshold", 0.5),
                rna_dense=_rna_dense,
            )
            if proxy_tf_map:
                # Keep the complete legacy proxy set for finite expansion.
                all_proxy_tfs |= set(proxy_tf_map.keys())
                # Proxy TFs are co-expressed with the target gene, so their KO-active bins
                # should match the target gene's
                ko_active = rna_pert[:, rna_name_to_idx[target_gene]] != rna_binned_orig[:, rna_name_to_idx[target_gene]]
                for tf, weight in proxy_tf_map.items():
                    tf_idx = rna_name_to_idx.get(tf)
                    if tf_idx is not None:
                        rna_pert[ko_active, tf_idx] *= (1.0 - abs(weight) * ko_strength)
                        # A proxy is a closure seed only when this proxy itself
                        # was changed.  The target's KO activity is not a
                        # substitute for individual proxy activity (notably for
                        # zero-expression proxies).
                        active_proxy_tfs |= _active_proxy_seeds(
                            {tf: weight}, rna_pert, rna_binned_orig, rna_name_to_idx
                        )
                logger.info(
                    f"  {target_gene} → {len(proxy_tf_map)} proxy TFs"
                )

    # --- 4. Subnetwork extraction ---
    # Extract the subnetwork only for genes to which a KO was actually applied
    # (skipped genes must not expand the propagation scope)
    subnet_depth = _validate_subnetwork_depth(cfg.get("subnetwork_depth", 3))
    all_subnet_genes = set()
    all_subnet_peaks = set()
    if subnet_depth == "unlimited":
        resolved_seeds = _resolved_ko_seeds(
            _applied_ko_indices, rna_var_names
        ) | (set(active_proxy_tfs) & set(rna_var_names))
        subnet = _extract_unlimited_subnetwork(
            resolved_seeds, transfer_functions, tf_peak_weights,
            rna_var_names, atac_var_names,
            min_r2_threshold=min_r2,
            min_lag_per_peak_filter=cfg.get("min_lag_per_peak_filter", True),
            max_edges=cfg.get("subnetwork_unlimited_max_edges", 100000),
            max_work_edges=cfg.get("subnetwork_unlimited_max_work_edges"),
        )
        all_subnet_genes |= subnet["genes"]
        all_subnet_peaks |= subnet["peaks"]
    else:
        for target_gene in ko_genes_applied:
            subnet = _extract_subnetwork(
                target_gene, granger_edges, transfer_functions, tf_peak_weights,
                depth=subnet_depth,
            )
            all_subnet_genes |= subnet["genes"]
            all_subnet_peaks |= subnet["peaks"]

    # --- 4.1 Proxy TF subnetwork expansion ---
    # Proxy TFs are selected by co-expression and are not necessarily inside the
    # causal-topology subnetwork. The proxy TFs, the peaks they regulate, and the
    # downstream target genes of those peaks must all be added to the subnetwork;
    # otherwise the TF→peak and peak→gene edges would be dropped by later filtering
    # and the KO effect could not propagate.
    if subnet_depth != "unlimited" and all_proxy_tfs and not tf_peak_weights.empty:
        n_added_genes = 0
        n_added_peaks = 0
        n_added_downstream = 0
        for tf in all_proxy_tfs:
            if tf not in all_subnet_genes:
                all_subnet_genes.add(tf)
                n_added_genes += 1
            tf_peaks = set(tf_peak_weights[tf_peak_weights["tf_gene"] == tf]["peak_id"])
            new_peaks = tf_peaks - all_subnet_peaks
            all_subnet_peaks |= new_peaks
            n_added_peaks += len(new_peaks)
            # Add the downstream target genes of the proxy TF's regulated peaks to the subnetwork too
            if transfer_functions is not None and not transfer_functions.empty:
                downstream = set(
                    transfer_functions[transfer_functions["peak_id"].isin(tf_peaks)]["gene"]
                )
                new_downstream = downstream - all_subnet_genes
                all_subnet_genes |= new_downstream
                n_added_downstream += len(new_downstream)
        if n_added_genes > 0:
            logger.info(
                f"  Subnetwork expansion: added {n_added_genes} proxy TFs, "
                f"{n_added_peaks} peaks, "
                f"{n_added_downstream} downstream genes "
                f"(total {len(all_subnet_genes)} genes, {len(all_subnet_peaks)} peaks)"
            )

    if not all_subnet_genes and not all_subnet_peaks:
        logger.warning("Subnetwork is empty; no propagation path")
        return {
            "target_genes": target_genes,
            "ko_genes_applied": ko_genes_applied,
            "delta_rna": pd.Series(0.0, index=rna_var_names),
            "delta_atac": pd.Series(0.0, index=atac_var_names),
            "gene_cascade_depth": {
                rna_var_names[idx]: 0 for idx in _seen_ko_indices
                if idx < len(rna_var_names)
            },
            "max_lag": 1,
            "n_bins": n_bins,
        }

    # --- 5. Build fast indexes (TF→peak, peak→gene) ---
    # Pre-resolve to integer indexes; zero string lookups in the hot loop
    # TF→peak: {tf_idx: [(peak_idx, weight, lag), ...]}

    tf_to_peaks: Dict[int, list] = {}
    _aggregated_edges = []  # final aggregated edges (including flip information)

    if not tf_peak_weights.empty:
        has_tf_lag = "tf_lag" in tf_peak_weights.columns

        if has_tf_lag:
            _aggregated_edges = []      # aggregated (TF, peak) edges, including flip information
            for (tf_gene, peak_id), group in tf_peak_weights.groupby(["tf_gene", "peak_id"]):
                if peak_id not in all_subnet_peaks:
                    continue
                if tf_gene not in all_subnet_genes:
                    continue
                tf_idx = rna_name_to_idx.get(tf_gene)
                peak_idx = atac_name_to_idx.get(peak_id)
                if tf_idx is None or peak_idx is None:
                    continue

                # Multiple lags: pick the best via argmin lag (the shortest lag is the most
                # credible; avoids weight inversion caused by penalty compensation)
                if "_window_only" in group.columns:
                    main_mask = group["_window_only"] != True
                    main_group = group[main_mask] if main_mask.any() else group
                else:
                    main_group = group
                best_idx = main_group["tf_lag"].values.argmin()
                row = main_group.iloc[best_idx]
                net_weight = float(row["weight"])
                best_lag = int(row["tf_lag"])

                # Record the final aggregated edge.
                # v2.0: window validation removed — auto max_lag + early-stop of
                # per-lag independent regressions eliminates boundary effects
                _aggregated_edges.append({
                    "tf_gene": tf_gene,
                    "peak_id": peak_id,
                    "weight": net_weight,
                    "lag": int(best_lag),
                })

                tf_to_peaks.setdefault(tf_idx, []).append(
                    (peak_idx, net_weight, int(best_lag))
                )
            _n_agg_before = len(tf_peak_weights)
            _n_agg_after = sum(len(v) for v in tf_to_peaks.values())
            _agg_method = "argmin_lag"

            # --- Min-lag filter: keep only the TF(s) with the smallest lag per peak ---
            # Biological rationale: at a given pseudotime t, a peak's accessibility is
            # directly regulated by the most recent TF activity; TF→peak relationships
            # with longer lags are more likely indirect effects or statistical noise.
            # If multiple TFs share the same minimal lag, keep them all (co-regulation).
            _n_minlag_filtered = 0
            _min_lag_filter = cfg.get("min_lag_per_peak_filter", True)
            if _min_lag_filter and _aggregated_edges:
                _peak_min_lag: dict = {}
                for e in _aggregated_edges:
                    pid = e["peak_id"]
                    cur = _peak_min_lag.get(pid)
                    if cur is None or e["lag"] < cur:
                        _peak_min_lag[pid] = e["lag"]

                _n_before_minlag = len(_aggregated_edges)
                _aggregated_edges = [
                    e for e in _aggregated_edges
                    if e["lag"] == _peak_min_lag[e["peak_id"]]
                ]
                _n_minlag_filtered = _n_before_minlag - len(_aggregated_edges)

                # Rebuild tf_to_peaks (long-lag edges already filtered out)
                if _n_minlag_filtered > 0:
                    tf_to_peaks.clear()
                    for e in _aggregated_edges:
                        _ti = rna_name_to_idx.get(e["tf_gene"])
                        _pi = atac_name_to_idx.get(e["peak_id"])
                        if _ti is not None and _pi is not None:
                            tf_to_peaks.setdefault(_ti, []).append(
                                (_pi, e["weight"], e["lag"])
                            )
                    _n_agg_after = sum(len(v) for v in tf_to_peaks.values())

            logger.info(
                f"  Multi-lag edge aggregation: {_n_agg_before} (TF,peak,lag) edges"
                f" → {_n_agg_after} (TF,peak) edges ({_agg_method}"
                f"{', min-lag filter removed ' + str(_n_minlag_filtered) + ' edges' if _min_lag_filter and _n_minlag_filtered > 0 else ''}"
                f")"
            )
        else:
            for _, row in tf_peak_weights.iterrows():
                peak_id = row["peak_id"]
                tf_gene = row["tf_gene"]
                if peak_id not in all_subnet_peaks:
                    continue
                if tf_gene not in all_subnet_genes:
                    continue
                tf_idx = rna_name_to_idx.get(tf_gene)
                peak_idx = atac_name_to_idx.get(peak_id)
                if tf_idx is None or peak_idx is None:
                    continue
                tf_to_peaks.setdefault(tf_idx, []).append(
                    (peak_idx, row["weight"], 1)
                )

    # TF→peak lag reverse lookup: {(peak_idx, tf_idx): lag}
    _peak_tf_lag: Dict[Tuple[int, int], int] = {}
    for tf_idx, peaks in tf_to_peaks.items():
        for peak_idx, _weight, lag in peaks:
            _peak_tf_lag[(peak_idx, tf_idx)] = lag

    # peak→gene: {peak_idx: [{gene, gene_idx, peak_id, gain, r2_score, ...}, ...]}
    peak_to_genes: Dict[int, list] = {}
    if transfer_functions is not None and not transfer_functions.empty:
        for _, row in transfer_functions.iterrows():
            peak_id = row["peak_id"]
            gene = row["gene"]
            if peak_id not in all_subnet_peaks:
                continue
            # Finite-depth mode historically retained all validated
            # peak→gene rows for an extracted peak.  The endpoint-membership
            # restriction is needed only for unlimited closure alignment.
            if subnet_depth == "unlimited" and gene not in all_subnet_genes:
                continue
            peak_idx = atac_name_to_idx.get(peak_id)
            gene_idx = rna_name_to_idx.get(gene)
            if peak_idx is None or gene_idx is None:
                continue
            peak_to_genes.setdefault(peak_idx, []).append({
                "gene": gene,
                "gene_idx": gene_idx,
                "peak_id": peak_id,
                "gain": row.get("steady_state_gain", 0.0),
                "r2_score": row.get("r2_score", 0.0),
            })

    # R² hard filter: directly exclude low-confidence peak→gene edges
    min_r2 = cfg.get("min_r2_threshold", 0.3)

    before_filter = sum(len(v) for v in peak_to_genes.values())
    peak_to_genes = {
        p_idx: [gi for gi in gi_list if gi.get("r2_score", 0) >= min_r2]
        for p_idx, gi_list in peak_to_genes.items()
    }
    peak_to_genes = {
        p_idx: gi_list for p_idx, gi_list in peak_to_genes.items() if gi_list
    }
    after_filter = sum(len(v) for v in peak_to_genes.values())
    logger.info(
        f"  R² hard filter (≥{min_r2}): {before_filter} → {after_filter} peak→gene edges "
        f"({before_filter - after_filter} filtered out)"
    )

    # --- 6. Determine the maximum lag ---
    max_lag = 1
    for peaks in tf_to_peaks.values():
        for _, _, lag in peaks:
            max_lag = max(max_lag, lag)
    max_lag = max(max_lag, 1)

    if max_lag >= n_bins:
        logger.warning(f"max_lag ({max_lag}) >= n_bins ({n_bins}); cannot simulate")
        return {
            "target_genes": target_genes,
            "ko_genes_applied": ko_genes_applied,
            "delta_rna": pd.Series(0.0, index=rna_var_names),
            "delta_atac": pd.Series(0.0, index=atac_var_names),
            "gene_cascade_depth": {
                rna_var_names[idx]: 0 for idx in _seen_ko_indices
                if idx < len(rna_var_names)
            },
            "max_lag": max_lag,
            "n_bins": n_bins,
        }

    # --- 7. NN model and per-edge linear fallback ---
    use_nn = nn_model is not None
    _rna_log_normalized = (
        bool(config.get("atac_to_rna", {}).get("log_normalize_rna", True))
        if nn_rna_log_normalized is None else bool(nn_rna_log_normalized)
    )
    if use_nn:
        nn_model.cpu().eval()
    # =========================================================================
    # Accumulation space choice (v1.36): accumulate directly in log1p-log1p
    # (the NN training space) by default
    # =========================================================================
    # When _rna_log_normalized=True, the NN's delta_r is output in
    # (log1p ∘ log1p)(RNA) units. The original logic converted each edge back to
    # raw via _delta_r_log_to_raw before accumulating into delta_cum; because
    # expm1 is not invertible plus raw-space one-sided saturation (downward
    # excursions are pinned at -1), accumulation produced a systematic positive
    # bias. Now the NN-native log delta is accumulated directly (+δ and −δ
    # cancel symmetrically), and at the end /log(2) converts to the log2 scale
    # and outputs delta_rna directly. This is equivalent to the multiplicative
    # model's log2 ratio in (L1+1) space, and also removes the asymmetric
    # residuals of the original neginf=-5 / posinf=+5 clipping.
    # =========================================================================
    _log_accumulation = bool(
        _rna_log_normalized
        and cfg.get("log_accumulation", True)
    )
    if _log_accumulation:
        logger.info(
            "  Accumulation space: log1p-log1p (NN-native, v1.36 default), "
            "delta_rna = delta_cum / log(2)"
        )
    elif _rna_log_normalized:
        logger.warning(
            "  Accumulation space: legacy raw (log_accumulation=False), "
            "keeps the original exponential positive bias; for regression comparison only"
        )

    # --- 7.5 Pathway-level edge records (TF→peak→gene, for cancellation analysis and regulatory-chain tracing) ---
    pathway_records: list = []  # independent contribution of each (TF, peak, gene, bin) tuple

    # --- 8. Per-bin forward simulation (integer indices, zero string lookups, state inheritance) ---
    # Core logic:
    #   1. Inherit upstream: genes/peaks modified in the previous bin carry their fold change into the current bin
    #   2. Propagate within this bin: TF(t-lag)→peak(t)→gene(t)
    #   3. Record changes: genes/peaks newly modified in this bin update their fold change for later bins to inherit
    n_affected_edges = 0
    n_rna = rna_binned_orig.shape[1]
    n_atac = atac_binned_orig.shape[1]

    # Additive accumulation: persist on first firing (the same (peak,gene) edge is not accumulated twice)
    delta_cum = np.zeros(n_rna)
    # Single-round log-accumulation output correction for executable TFs.
    # This does not participate in propagation or alter delta_cum.
    tf_l1_output_delta = np.zeros(n_rna)
    fired_edges: set = set()  # (t, peak_idx, gene_idx) — per-bin first-fire deduplication
    delta_cum_history: list = []  # per-bin snapshots of delta_cum, used for cell-fate shift projection
    rna_state_history = np.full((n_bins, n_rna), np.nan, dtype=float)
    local_native = np.zeros((n_bins, n_rna), dtype=float)
    local_events = []
    local_event_keys = set()

    # TF mask: only TF genes need rna_pert modifications for cascade propagation
    is_tf = np.zeros(n_rna, dtype=bool)
    for tf_idx in tf_to_peaks:
        is_tf[tf_idx] = True
    # KO target genes: rna_pert has already been modified directly by the KO
    is_ko_target = np.zeros(n_rna, dtype=bool)
    for idx in _seen_ko_indices:
        if idx < n_rna:
            is_ko_target[idx] = True
    rna_pert_modified = is_tf | is_ko_target  # genes whose output should be read from rna_pert

    # fold change tracking: initialized to 1.0, updated after modification
    rna_fold = np.ones(n_rna)
    atac_fold = np.ones(n_atac)

    # Record the fold changes produced during the initialization phase (KO + proxy TF)
    rna_init_modified = np.abs(rna_pert - rna_binned_orig).max(axis=0) > 1e-10
    for g_idx in np.where(rna_init_modified)[0]:
        valid = rna_binned_orig[:, g_idx] > 1e-10
        if valid.any():
            rna_fold[g_idx] = rna_pert[valid, g_idx].mean() / rna_binned_orig[valid, g_idx].mean()
    rna_fold_ko = rna_fold.copy()   # save the KO fold for reuse across branches
    atac_fold_ko = atac_fold.copy()
    # Snapshot initialization-only offset before the forward loop adds state.
    rna_initialization_l1_offset = (
        rna_pert.mean(axis=0) - rna_binned_orig.mean(axis=0)
    )
    # Immutable initialization-only L1 base.  All projection states are
    # decoded from this base plus the cumulative simulated effects exactly
    # once; rna_pert is an execution buffer and must not be used as a base.
    rna_projection_base = rna_pert.copy()
    # Pre-lag bins are real state boundaries too: they contain direct
    # KO/proxy effects but no downstream lagged propagation yet.
    for _b in range(min(max_lag, n_bins)):
        rna_state_history[_b] = rna_projection_base[_b]

    # Starting bin of each branch: reset propagation state across branch boundaries (keep the KO fold)
    branch_boundaries = pseudotime_df.attrs.get("branch_boundaries") if hasattr(pseudotime_df, "attrs") else None
    branch_starts = set()
    if branch_boundaries is not None and len(branch_boundaries) > 1:
        for br_start, _ in branch_boundaries:
            if br_start > 0:
                branch_starts.add(br_start)
    if branch_starts:
        logger.info(f"Branch-aware forward propagation: {len(branch_boundaries)} branches, "
                    f"starting bins={sorted(branch_starts)}")

    # NN: move to CPU for inference
    if use_nn:
        nn_model.cpu().eval()

    # Multi-round propagation: collect deltas per round for cascade-amplification diagnostics
    per_round_delta = []

    # TF→peak contribution tracking: record each peak's per-TF contributions from the previous round, for the ATAC→RNA step attribution
    prev_peak_tf_contrib: Dict[int, Dict[int, float]] = {}  # {peak_idx: {tf_idx: contrib}}

    # True graph distance from an applied gene KO.  This deliberately does
    # not use tf_lag/max_lag: those describe pseudotime scheduling, not hops.
    gene_cascade_depth: Dict[str, int] = {
        rna_var_names[idx]: 0 for idx in _seen_ko_indices
        if idx < len(rna_var_names)
    }
    source_depth: Dict[int, int] = {
        idx: 0 for idx in _seen_ko_indices if idx < n_rna
    }

    for r in range(propagation_rounds):
        if propagation_rounds > 1 and r > 0:
            # Round 2+: reset the perturbation matrices and re-propagate with the previous round's fold as the initial state
            rna_pert = rna_binned_orig.copy()
            atac_pert = atac_binned_scaled.copy()
            # Re-apply the KO
            for target_gene in ko_genes_applied:
                ko_idx = rna_name_to_idx.get(target_gene)
                if ko_idx is not None:
                    ko_expr = rna_binned_orig[:, ko_idx]
                    active_mask = np.zeros(n_bins, dtype=bool)
                    lags = _tf_lags.get(ko_idx, set())
                    for t_ko in range(n_bins):
                        if ko_expr[t_ko] <= 0:
                            continue
                        if lags and not any(t_ko + lag < n_bins for lag in lags):
                            continue
                        active_mask[t_ko] = True
                    rna_pert[active_mask, ko_idx] *= (1.0 - ko_strength)
            rna_fold = rna_fold_ko.copy()
            atac_fold = atac_fold_ko.copy()

        for t in range(max_lag, n_bins):
            if propagation_rounds > 1:
                # ---- 8.0 Reset propagation state at branch boundaries ----
                if t in branch_starts:
                    rna_fold = rna_fold_ko.copy()
                    atac_fold = atac_fold_ko.copy()

                # ---- 8.1 Inherit upstream state ----
                rna_pert[t, :] = rna_binned_orig[t, :] * rna_fold
                atac_pert[t, :] = atac_binned_scaled[t, :] * atac_fold
            else:
                # ---- 8.0 Only TFs inherit delta for cascade propagation; non-TFs do not inherit across bins ----
                active = (np.abs(delta_cum) > 1e-10) & is_tf
                if np.any(active):
                    # L2 delta_cum is added directly to L1 → L1_pert:
                    #   log_accum: delta_cum is in L2 space, delta_tf = (L1+L2) − L1 = the L2 delta;
                    #              during Pearson training X=log1p(RNA)=L1, the weight is on the L1→scaled_ATAC scale,
                    #              so the L2 delta used as the delta_tf input is approximately correct in magnitude (for small deltas).
                    #   legacy:    delta_cum is in L1 space, plain addition.
                    rna_pert[t, active] = rna_binned_orig[t, active] + delta_cum[active]

            # ---- 8a. RNA→ATAC: TF(t-lag) → peak(t) ----
            curr_peak_tf_contrib: Dict[int, Dict[int, float]] = {}
            for tf_idx, peaks in tf_to_peaks.items():
                for peak_idx, weight, lag in peaks:
                    delta_tf = rna_pert[t - lag, tf_idx] - rna_binned_orig[t - lag, tf_idx]
                    if abs(delta_tf) < 1e-10:
                        continue
                    contrib = weight * delta_tf
                    atac_pert[t, peak_idx] += contrib
                    curr_peak_tf_contrib.setdefault(peak_idx, {})\
                        .setdefault(tf_idx, 0.0)
                    curr_peak_tf_contrib[peak_idx][tf_idx] += contrib
                    n_affected_edges += 1

            # Clip atac_pert[t] to valid range after all TF contributions
            np.clip(atac_pert[t, :], -1.0, 1.0, out=atac_pert[t, :])

            # ---- 8b. ATAC→RNA: peak(t-1) → gene(t) ----
            for peak_idx, gene_infos in peak_to_genes.items():
                delta_a = atac_pert[t - 1, peak_idx] - atac_binned_scaled[t - 1, peak_idx]
                if abs(delta_a) < 1e-10:
                    continue

                a_orig = atac_binned_scaled[t - 1, peak_idx]
                a_new = atac_pert[t - 1, peak_idx]

                # TF attribution: distribute proportionally from the previous bin's TF→peak contributions
                tf_contribs = prev_peak_tf_contrib.get(peak_idx, {})
                total_contrib = sum(tf_contribs.values())
                has_tf_attribution = abs(total_contrib) > 1e-10

                for gi in gene_infos:
                    gene_idx = gi["gene_idx"]

                    if use_nn:
                        delta_r, mechanism = _nn_delta_or_linear(
                            _nn_predict_delta_r,
                            (nn_model, a_orig, a_new, gi["peak_id"], gi["gene"],
                             nn_peak_to_idx, nn_gene_to_idx),
                            delta_a, gi.get("gain", 0.0), gi.get("gain", 0.0),
                        )
                    else:
                        gain = gi.get("gain", 0.0)
                        if abs(gain) < 1e-10:
                            continue
                        delta_r, mechanism = gain * delta_a, "linear"

                    local_key = (t, peak_idx, gene_idx)
                    if local_key not in local_event_keys and np.isfinite(delta_r):
                        local_event_keys.add(local_key)
                        local_native[t, gene_idx] += delta_r
                        _sources = list(tf_contribs) or [None]
                        for _src in _sources:
                            _lag = _peak_tf_lag.get((peak_idx, _src), 1) if _src is not None else None
                            local_events.append({
                                "bin": int(t), "peak": atac_var_names[peak_idx],
                                "gene": rna_var_names[gene_idx],
                                "native_delta": float(delta_r),
                                "causal_peak_bin": int(max(t - 1, 0)),
                                "source_tf": rna_var_names[_src] if _src is not None else None,
                                "source_tf_bin": int(t - 1 - _lag) if _lag is not None else None,
                                "source_tf_lag": _lag,
                                "mechanism": mechanism,
                                "r2_score": float(gi.get("r2_score", 0.0)),
                                "native_effect_unit": "delta_log1p_rna_log1p_cp10k",
                                "effect_schema_version": "bin_local_effect_v1",
                            })

                    # Global first-fire deduplication: each (peak,gene) edge accumulates only on first firing
                    edge_key = (peak_idx, gene_idx)
                    if edge_key not in fired_edges:
                        # _log_accumulation=True (v1.36 default): directly accumulate the NN's delta_r in L2
                        # (log1p∘log1p RNA) units; +δ and −δ cancel symmetrically in the linear
                        # accumulation region, eliminating at the source the exponential amplification positive bias of raw accumulation.
                        # _log_accumulation=False (legacy, unit tests/regression): still go through the original
                        # _delta_r_log_to_raw conversion back to L1 units and then raw accumulation
                        if (not _log_accumulation) and _rna_log_normalized:
                            delta_r = _delta_r_log_to_raw(
                                delta_r, rna_binned_orig[t, gene_idx]
                            )
                        delta_cum[gene_idx] += delta_r
                        fired_edges.add(edge_key)

                        if (
                            propagation_rounds == 1
                            and _log_accumulation
                            and is_tf[gene_idx]
                        ):
                            _accumulate_tf_l1_output_event(
                                tf_l1_output_delta,
                                gene_idx,
                                rna_binned_orig[t, gene_idx],
                                delta_r,
                                t,
                                n_bins,
                            )

                        _record_gene_cascade_depth(
                            gene_cascade_depth, source_depth, gene_idx,
                            gi["gene"], tf_contribs, delta_r,
                        )

                        # --- Record pathway-level edge contributions (TF proportional attribution), written only on first firing ---
                        peak_id_str = atac_var_names[peak_idx]
                        gene_name = gi["gene"]
                        if has_tf_attribution:
                            for tf_idx, tf_contrib in tf_contribs.items():
                                tf_name = rna_var_names[tf_idx]
                                frac = tf_contrib / total_contrib
                                pathway_records.append({
                                    "target_gene": target_gene_label,
                                    "tf_name": tf_name,
                                    "peak_id": peak_id_str,
                                    "affected_gene": gene_name,
                                    "delta_r_raw": float(delta_r * frac),
                                    "delta_r_total": float(delta_r),
                                    "mechanism": mechanism,
                                    "bin": t,
                                    "tf_lag": _peak_tf_lag.get((peak_idx, tf_idx), 1),
                                    "r2_score": gi.get("r2_score", 0.0),
                                })
                        else:
                            pathway_records.append({
                                "target_gene": target_gene_label,
                                "tf_name": "unknown",
                                "peak_id": peak_id_str,
                                "affected_gene": gene_name,
                                "delta_r_raw": float(delta_r),
                                "delta_r_total": float(delta_r),
                                "mechanism": mechanism,
                                "bin": t,
                                "tf_lag": 0,
                                "r2_score": gi.get("r2_score", 0.0),
                            })
                    # Only TF genes update rna_pert for cascade propagation
                    if is_tf[gene_idx]:
                        # Same as L1251: the L2/L1 delta_cum is added directly to L1, keeping units consistent
                        rna_pert[t, gene_idx] = rna_binned_orig[t, gene_idx] + delta_cum[gene_idx]

            # Pass the current bin's TF→peak contributions to the next bin's ATAC→RNA attribution
            prev_peak_tf_contrib = curr_peak_tf_contrib

            # Complete canonical RNA boundary state.  In log accumulation
            # delta_cum is L2 (log1p-log1p); decode exactly at the state
            # boundary rather than projecting delta_cum itself.
            _base_state = np.asarray(rna_projection_base[t], dtype=float).copy()
            if _log_accumulation:
                with np.errstate(invalid="ignore", over="ignore"):
                    _canonical_state = np.expm1(np.log1p(_base_state) + delta_cum)
            else:
                _canonical_state = _base_state + delta_cum
            rna_state_history[t] = _canonical_state

            # Per-bin snapshot of delta_cum, used for cell-fate shift projection
            delta_cum_history.append(delta_cum.copy())

            if propagation_rounds > 1:
                # ---- 8c. Update fold changes ----
                rna_modified = np.abs(rna_pert[t, :] - rna_binned_orig[t, :]) > 1e-10
                rna_valid = rna_modified & (rna_binned_orig[t, :] > 1e-10)
                rna_fold[rna_valid] = rna_pert[t, rna_valid] / rna_binned_orig[t, rna_valid]

                atac_modified = np.abs(atac_pert[t, :] - atac_binned_scaled[t, :]) > 1e-10
                atac_valid = atac_modified & (atac_binned_scaled[t, :] > 1e-10)
                atac_fold[atac_valid] = atac_pert[t, atac_valid] / atac_binned_scaled[t, atac_valid]

            # Multi-round mode: collect this round's delta (log2FC)
            if propagation_rounds > 1:
                rna_mp = rna_pert.mean(axis=0)
                rna_mo = rna_binned_orig.mean(axis=0)
                rna_mo_s = np.where(rna_mo > 0, rna_mo, np.nan)
                rna_mp_s = np.where(rna_mp > 0, rna_mp, np.nan)
                with np.errstate(divide="ignore", invalid="ignore"):
                    r_l2fc = np.log2(rna_mp_s / rna_mo_s)
                round_delta = pd.Series(r_l2fc, index=rna_var_names)
                per_round_delta.append(round_delta)

    # --- 9. Aggregate deltas (log2 fold change) ---
    # Single-round mode: TF/KO genes are read from rna_pert, non-TF genes get an equivalent perturbed mean from delta_cum
    # Multi-round mode: all genes modify rna_pert via folds and are read uniformly from rna_pert
    rna_mean_pert_direct = rna_pert.mean(axis=0)
    rna_mean_orig = rna_binned_orig.mean(axis=0)

    if _log_accumulation:
        # v1.36: delta_cum is already in L2 (log1p∘log1p RNA) units; the accumulation layer keeps the multiplicative
        #       cancellation correctness (+δ and −δ symmetric).  At the end, back-convert to L1 and compute
        #       log2(L1_pert/L1_orig), restoring the legacy output scale and the ±5 clipping,
        #       avoiding magnitude compression by delta_cum/log(2) degrading the benchmark directional agreement.
        # rna_mean_pert_equiv (L1 scale) is used only by the propagation_rounds>1 multi-round path
        rna_mean_pert_equiv = np.expm1(np.log1p(rna_mean_orig) + delta_cum)
        if propagation_rounds > 1:
            rna_mean_pert_final = rna_mean_pert_direct
            _orig_safe = np.where(rna_mean_orig > 0, rna_mean_orig, np.nan)
            _pert_safe = np.where(rna_mean_pert_final > 0, rna_mean_pert_final, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                log2fc = np.log2(_pert_safe / _orig_safe)
        else:
            # Single-round default path: back-convert the L2 delta → L1 pert, then compute the standard log2FC
            # Executable TFs use the event-time-aware L1 output accumulator;
            # non-TFs retain the existing delta_cum endpoint reconstruction.
            rna_mean_pert_final = rna_mean_pert_equiv.copy()
            _tf_output_means = _single_round_tf_output_means(
                rna_mean_orig,
                rna_initialization_l1_offset,
                tf_l1_output_delta,
                is_tf,
                is_ko_target,
            )
            _non_ko_tf = is_tf & ~is_ko_target
            rna_mean_pert_final[_non_ko_tf] = _tf_output_means[_non_ko_tf]
            # KO targets retain their existing direct output state (they are
            # excluded from perturbation result rows).
            rna_mean_pert_final[is_ko_target] = rna_mean_pert_direct[is_ko_target]
            _orig_safe = np.where(rna_mean_orig > 0, rna_mean_orig, np.nan)
            _pert_safe = np.where(rna_mean_pert_final > 0, rna_mean_pert_final, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                log2fc = np.log2(_pert_safe / _orig_safe)
    else:
        # Legacy raw-accumulation path (log_accumulation=False)
        rna_mean_pert_equiv = rna_mean_orig + delta_cum
        if propagation_rounds > 1:
            rna_mean_pert_final = rna_mean_pert_direct
        else:
            rna_mean_pert_final = np.where(rna_pert_modified, rna_mean_pert_direct, rna_mean_pert_equiv)

        rna_mean_orig_safe = np.where(rna_mean_orig > 0, rna_mean_orig, np.nan)
        rna_mean_pert_safe = np.where(rna_mean_pert_final > 0, rna_mean_pert_final, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            log2fc = np.log2(rna_mean_pert_safe / rna_mean_orig_safe)
    delta_rna = pd.Series(log2fc, index=rna_var_names)

    # Unscale ATAC back to original [0,1] space for delta_atac output
    atac_pert_unscaled = _unscale_atac(atac_pert, root_atac)
    delta_atac = pd.Series(
        atac_pert_unscaled.mean(axis=0) - atac_binned_orig.mean(axis=0),
        index=atac_var_names,
    )

    # WT null-distribution z-score: per-peak standard deviation across pseudotime bins = natural WT fluctuation
    # z = model change / WT fluctuation — the significance threshold is controlled by config.perturbation.atac_significance_zscore
    _atac_wt_std = np.std(atac_binned_orig, axis=0)
    _atac_z = np.divide(
        delta_atac.values,
        _atac_wt_std,
        out=np.zeros(len(delta_atac)),
        where=_atac_wt_std > 1e-10,
    )
    atac_z_score = pd.Series(_atac_z, index=atac_var_names)

    # --- 9.5 Aggregate pathway-level edge records ---
    pathway_df = pd.DataFrame(pathway_records) if pathway_records else pd.DataFrame(
        columns=["target_gene", "tf_name", "peak_id", "affected_gene",
                  "delta_r_raw", "delta_r_total", "delta_rna_log2fc",
                  "mechanism", "r2_score", "tf_lag", "bin"]
    )
    if not pathway_df.empty:
        # Aggregate contributions across bins by (target_gene, tf_name, peak_id, affected_gene)
        # tf_lag is identical for the same (tf, peak) pair, take first; bin takes a range
        group_cols = ["target_gene", "tf_name", "peak_id", "affected_gene", "mechanism"]
        agg_dict = {
            "delta_r_raw": "sum", "delta_r_total": "sum",
            "r2_score": "first", "tf_lag": "first",
            "bin": ["min", "max"],
        }
        pathway_df = pathway_df.groupby(group_cols, as_index=False).agg(agg_dict)
        # Flatten multi-level column names
        pathway_df.columns = [
            "_".join(c).rstrip("_") if isinstance(c, tuple) else c
            for c in pathway_df.columns
        ]

        # delta_r_total should be the true total effect of that (peak, gene) edge (= the sum of all TFs' delta_r_raw)
        edge_total = pathway_df.groupby(
            ["target_gene", "peak_id", "affected_gene", "mechanism"]
        )["delta_r_raw_sum"].sum().reset_index()
        edge_total.rename(columns={"delta_r_raw_sum": "delta_r_edge_total"}, inplace=True)
        pathway_df = pathway_df.merge(
            edge_total,
            on=["target_gene", "peak_id", "affected_gene", "mechanism"],
            how="left",
        )
        pathway_df.drop(columns=["delta_r_total_sum"], inplace=True)
        pathway_df.rename(columns={"delta_r_edge_total": "delta_r_total"}, inplace=True)

        # Add per-gene log2FC (aligned with perturbation_results)
        # delta_rna may have duplicate indices, use a dict mapping to avoid InvalidIndexError
        log2fc_map = dict(zip(rna_var_names, delta_rna.values))
        pathway_df["delta_rna_log2fc"] = pathway_df["affected_gene"].map(log2fc_map)

    logger.info(
        f"  Bin-forward finished: {n_affected_edges} edge propagations, "
        f"{(delta_rna.abs() > 1e-6).sum()} affected genes, "
        f"{(delta_atac.abs() > 1e-6).sum()} affected peaks, "
        f"{len(pathway_df)} TF→peak→gene pathways"
    )

    result = {
        "target_genes": target_genes,
        "ko_genes_applied": ko_genes_applied,
        "delta_rna": delta_rna,
        "delta_atac": delta_atac,
        "gene_cascade_depth": gene_cascade_depth,
        "atac_z_score": atac_z_score,
        "delta_cum": delta_cum,
        "delta_cum_history": delta_cum_history,
        "max_lag": max_lag,
        "n_bins": n_bins,
        "n_affected_edges": n_affected_edges,
        "prediction_mode": "bin_forward_nn" if use_nn else "bin_forward_linear",
        "pathway_edges": pathway_df,
        "aggregated_tf_peak_edges": pd.DataFrame(_aggregated_edges),
    }
    # This reconstruction is required for merged cell-level L0 aggregation,
    # not just for the optional visualization.  Do not silently use the old
    # dense mean-L1 fallback for unsupported semantics.
    if propagation_rounds != 1:
        raise ValueError(
            "merged cell-L0 aggregation requires propagation_rounds=1; "
            f"got {propagation_rounds}"
        )
    if not _log_accumulation:
        raise ValueError(
            "merged cell-L0 aggregation requires log_accumulation=True"
        )
    else:
            # Projection consumes explicit log1p RNA states, never delta_cum.
            # rna_adata is already canonical log1p RNA after preprocessing.
            _cell_x = rna_adata.X.toarray() if hasattr(rna_adata.X, "toarray") else np.asarray(rna_adata.X)
            _bin_operator, _weights = _projection_operators(
                pseudotime_df, n_bins, smooth_kernel, min_cpb
            )
            _projection_base = _bin_operator @ np.asarray(_cell_x, dtype=float)
            _direct_multiplier = np.divide(
                np.asarray(rna_projection_base, dtype=float),
                np.asarray(rna_binned_orig, dtype=float),
                out=np.ones_like(rna_projection_base, dtype=float),
                where=np.asarray(rna_binned_orig, dtype=float) > 1e-12,
            )
            _direct_state = _projection_base * _direct_multiplier
            _projection_pert = np.expm1(np.log1p(_direct_state) + local_native)
            _invalid = np.argwhere(~np.isfinite(_projection_pert))
            _projection_pert, _projection_floor_count = _floor_rna_l1(_projection_pert)
            _branch = pseudotime_df["branch"].to_numpy(dtype=object) if "branch" in pseudotime_df else np.zeros(rna_adata.n_obs, dtype=int)
            # Lightweight simulation/test AnnData-compatible objects may not
            # expose obs_names.  Cell IDs are metadata only; reconstruction
            # and L0 aggregation use row alignment, so deterministic integer
            # IDs preserve the real AnnData path without weakening it.
            _cell_ids = getattr(rna_adata, "obs_names", None)
            if _cell_ids is None:
                _cell_ids = np.arange(rna_adata.n_obs, dtype=int)
            result["projection_inputs"] = {
                "target_genes": tuple(target_genes),
                "control_cell_states": np.asarray(_cell_x, dtype=float),
                "control_bin_states": _projection_base,
                "perturbed_bin_states": _projection_pert,
                "cell_bin_weights": _weights,
                "direct_bin_multiplier": _direct_multiplier,
                "native_bin_effect": local_native,
                "local_events": local_events,
                "cell_ids": np.asarray(_cell_ids, dtype=object),
                "bins": np.asarray(bin_indices, dtype=int),
                "pseudotime": pseudotime_df["pseudotime"].to_numpy(dtype=float),
                "branch": _branch,
                "features": tuple(str(v) for v in rna_adata.var_names),
                "state_unit": config.get("perturbation", {}).get("cell_projection", {}).get(
                    "state_unit", "rna_log1p_cp10k"
                ),
                "bin_state_mode": "gaussian_shared_operator" if smooth_kernel == "gaussian" else "hard_shared_operator",
                "invalid_state": bool(len(_invalid)),
                "invalid_diagnostic": (f"canonical bin state has {len(_invalid)} invalid values; "
                                        f"first={_invalid[0].tolist()}" if len(_invalid) else None),
                "effect_schema_version": "bin_local_effect_v1",
                "l1_floor_applied_count": int(_projection_floor_count),
                "l1_floor_applied_fraction": float(_projection_floor_count / max(_projection_pert.size, 1)),
                "l1_floor_status": "clipped" if _projection_floor_count else "none",
                "native_effect_unit": "delta_log1p_rna_log1p_cp10k",
                "log_accumulation": True,
            }
    if propagation_rounds > 1:
        result["per_round_delta"] = per_round_delta
    return result


# ============================================================================
# Peak perturbation (peak KO): user-specified peak + Δ accessibility, propagated
# downstream along the existing TF→Peak→Gene network
#
# Design principles (aligned with the "Peak perturbation feature development notes"):
#   - Not a new model, just a second intervention entry point of the existing bin_forward engine
#   - Downstream propagation = perturbation simulation; Upstream attribution = attribution/interpretation
#   - Upstream TFs only do attribution and do not participate in reverse causal propagation
#     (engine step 8a only reads TF RNA delta, which naturally guarantees this boundary)
#   - direct mode (depth=1): only outputs the direct target genes of that peak, no 8a cascade
#   - ΔR_g = f(A'_p) - f(A_p) reuses the trained NN transfer function, no new regression
# ============================================================================


def _match_peak(requested_id: str, atac_adata, mode: str = "exact"):
    """Peak matching: exact → overlap → (nearest) → failure.

    Parameters
    ----------
    requested_id : str
        User input, in the form "chr3:123400-123900"
    atac_adata : AnnData
        var_names = peak_id, .var must contain chr/start/end coordinates
    mode : str
        "exact" (default) | "overlap" | "nearest" — whether stepwise fallback is allowed

    Returns
    -------
    (matched_peak_id | None, info_dict)
    info_dict: {"method": "exact"/"overlap"/"nearest"/None, "distance": int}
    """
    import re
    # Input normalization: strip thousands separators + normalize unicode hyphens (–/—/‐/‑/‒/―) to ASCII '-'
    # Coordinates copied from papers/UCSC often carry these formats (e.g. "chr3:34,691,323–34,693,322")
    requested_id = requested_id.replace(",", "").strip()
    for _dash in ("–", "—", "‐", "‑", "‒", "―"):
        requested_id = requested_id.replace(_dash, "-")

    var_names = list(atac_adata.var_names)

    # 1. exact coordinate match
    if requested_id in var_names:
        return requested_id, {"method": "exact", "distance": 0}

    m = re.match(r"^(\w+):(\d+)-(\d+)$", requested_id)
    if m is None:
        return None, {"method": None, "distance": None}

    chrom, start, end = m.group(1), int(m.group(2)), int(m.group(3))
    var_df = atac_adata.var
    has_coords = all(c in var_df.columns for c in ("chr", "start", "end"))

    # 2. overlap interval intersection (take the one with the largest overlap length)
    if has_coords:
        same_chr = var_df[var_df["chr"] == chrom]
        if not same_chr.empty:
            overlap = same_chr[(same_chr["start"] < end) & (same_chr["end"] > start)]
            if not overlap.empty:
                o_len = overlap["end"].clip(upper=end) - overlap["start"].clip(lower=start)
                best = o_len.idxmax()
                return best, {"method": "overlap", "distance": 0}

    # 3. nearest closest peak (only when explicitly enabled; distance must be reported, no silent replacement)
    if mode == "nearest" and has_coords:
        same_chr = var_df[var_df["chr"] == chrom]
        if not same_chr.empty:
            center = (start + end) / 2
            cand_center = (same_chr["start"] + same_chr["end"]) / 2
            dist = (cand_center - center).abs()
            best = dist.idxmin()
            return best, {"method": "nearest", "distance": int(dist.loc[best])}

    # 4. not found
    return None, {"method": None, "distance": None}


def _extract_peak_subnetwork(
    peak_id: str,
    transfer_functions: pd.DataFrame,
    tf_peak_weights: pd.DataFrame,
    depth: int = 1,
) -> Dict:
    """Peak-rooted BFS subnetwork extraction (the gene-rooted counterpart aligns with _extract_subnetwork).

    Layer 1: downstream genes regulated by this peak (the raw set before R² filtering in transfer_functions)
    Layer 2+: if a downstream gene is a TF, keep expanding the peaks it regulates (used in propagation mode)
    """
    all_peaks = {peak_id}
    all_genes = set()
    frontier_peaks = {peak_id}

    for _level in range(max(depth, 1)):
        # Downstream genes of the peaks in the current layer
        new_genes = set()
        if transfer_functions is not None and not transfer_functions.empty:
            new_genes |= set(
                transfer_functions[transfer_functions["peak_id"].isin(frontier_peaks)]["gene"]
            )
        new_genes -= all_genes
        all_genes |= new_genes

        # If any downstream gene is a TF, expand the peaks it regulates
        new_peaks = set()
        if new_genes and tf_peak_weights is not None and not tf_peak_weights.empty:
            new_peaks |= set(
                tf_peak_weights[tf_peak_weights["tf_gene"].isin(new_genes)]["peak_id"]
            )
        new_peaks -= all_peaks
        all_peaks |= new_peaks
        frontier_peaks = new_peaks if new_peaks else frontier_peaks

    return {"genes": all_genes, "peaks": all_peaks}


def _simulate_peak_direct_forward(
    peak_id: str,
    rna_adata,
    atac_adata,
    pseudotime_df,
    transfer_functions: pd.DataFrame,
    tf_peak_weights: pd.DataFrame,
    config: dict,
    transfer_models: Optional[dict] = None,
    root_atac: Optional[np.ndarray] = None,
) -> Dict:
    """Peak perturbation direct mode core: single cluster, propagates only the direct target genes of this peak.

    Shares with _simulate_knockout_bin_forward: binning, [-1,1] scaling, the NN
    transfer function, L2 accumulation (v1.36), and the log2FC output convention. Differences:
      - The perturbation is injected into atac_pert[:, p_idx] instead of a rna_pert KO
      - No 8a (RNA→ATAC) step and no TF cascade — the boundary naturally does not propagate backwards
      - Each (peak, gene) edge still follows the engine's "fire once, accumulate" semantics (fired_edges)
    """
    cfg = config["perturbation"]
    peak_cfg = cfg.get("peak_ko", {})
    strength = float(peak_cfg.get("strength", -1.0))
    peak_mode = peak_cfg.get("mode", "relative")
    min_r2 = cfg.get("min_r2_threshold", 0.3)
    nn_model = nn_peak_to_idx = nn_gene_to_idx = None
    nn_rna_log_normalized = bool(config.get("atac_to_rna", {}).get("log_normalize_rna", True))
    if transfer_models is not None:
        if transfer_models.get("type") != "nn":
            raise ValueError("Only NN transfer models are supported")
        nn_model = _load_nn_from_transfer_models(transfer_models)
        nn_rna_log_normalized = _read_nn_rna_log_normalized(transfer_models, config)
        nn_peak_to_idx = transfer_models["data"]["peak_to_idx"]
        nn_gene_to_idx = transfer_models["data"]["gene_to_idx"]

    # --- 1. Binning (reuses the engine logic) ---
    pcfg = config.get("pseudotime", {})
    smooth_kernel = pcfg.get("smooth_kernel", "hard")
    _overlap = pcfg.get("smooth_overlap_factor") if smooth_kernel == "gaussian" else None
    bin_indices, n_bins = _adaptive_rebin(
        pseudotime_df, rna_adata.n_obs,
        max_bins=pcfg["n_bins"],
        min_cells_per_bin=pcfg.get("min_cells_per_bin", 5),
        target_bins=pcfg["n_bins"] if smooth_kernel == "gaussian" else None,
        overlap_factor=_overlap,
    )
    pseudotime_df["bin"] = bin_indices
    min_cpb = pcfg.get("min_cells_per_bin", 5)
    rna_binned_orig = _bin_expression(rna_adata, pseudotime_df, n_bins,
                                      smooth=smooth_kernel, min_cells_per_bin=min_cpb)
    atac_binned_orig = _bin_expression(atac_adata, pseudotime_df, n_bins,
                                       smooth=smooth_kernel, min_cells_per_bin=min_cpb)
    rna_var_names = list(rna_adata.var_names)
    atac_var_names = list(atac_adata.var_names)
    if peak_id not in atac_var_names:
        return {"error": f"peak {peak_id} not in atac_adata var_names"}
    p_idx = atac_var_names.index(peak_id)

    # --- 2. [-1,1] scaling (root_atac baseline, consistent with the engine) ---
    cfg_atac = config.get("atac_to_rna", {})
    if root_atac is None:
        root_atac = _compute_root_atac(
            atac_adata, pseudotime_df,
            quantile=cfg_atac.get("root_cell_quantile", 0.05),
            min_count=cfg_atac.get("root_cell_min_count", 5),
            floor=cfg_atac.get("root_atac_floor", 0.05),
        )
    atac_binned_scaled = _scale_atac(atac_binned_orig, root_atac)

    # --- 3. Perturbation injection: defined in raw space (A = binarized mean ∈ [0,1]), then rescaled into [-1,1] ---
    # relative: A'_raw = clip(A_raw*(1+s), 0, 1); s=-1 → fully off (raw 0 = truly off)
    # absolute: A'_raw = clip(s, 0, 1)
    a_raw = atac_binned_orig[:, p_idx].copy()
    if peak_mode == "absolute":
        a_pert_raw = np.full_like(a_raw, float(np.clip(strength, 0.0, 1.0)))
    else:
        a_pert_raw = np.clip(a_raw * (1.0 + strength), 0.0, 1.0)
    a_pert_scaled = _scale_atac(
        a_pert_raw.reshape(-1, 1), root_atac[p_idx:p_idx + 1]
    ).ravel()

    atac_pert = atac_binned_scaled.copy()
    atac_pert[:, p_idx] = a_pert_scaled

    # Extrapolation warning: the perturbed scaled value falls outside this peak's training (WT bin) range
    _a_train_min = float(atac_binned_scaled[:, p_idx].min())
    _a_train_max = float(atac_binned_scaled[:, p_idx].max())
    _extrap = (a_pert_scaled.min() < _a_train_min - 1e-9) or (a_pert_scaled.max() > _a_train_max + 1e-9)
    if _extrap:
        logger.warning(
            f"  Peak perturbation falls outside this peak's training-visible range "
            f"[{_a_train_min:.3f}, {_a_train_max:.3f}] "
            f"(perturbed ∈ [{a_pert_scaled.min():.3f}, {a_pert_scaled.max():.3f}]); "
            f"the prediction involves extrapolation."
        )

    # --- 4. Direct downstream genes (hard R² filter, consistent with the engine) ---
    gene_infos: list = []
    if transfer_functions is not None and not transfer_functions.empty:
        tf_sub = transfer_functions[transfer_functions["peak_id"] == peak_id]
        for _, row in tf_sub.iterrows():
            r2 = float(row.get("r2_score", 0.0))
            if r2 >= min_r2:
                gene_infos.append({
                    "gene": str(row["gene"]),
                    "r2_score": r2,
                    "gain": float(row.get("steady_state_gain", 0.0)),
                })

    use_nn = nn_model is not None
    _rna_log_normalized = nn_rna_log_normalized
    _log_accumulation = bool(_rna_log_normalized and cfg.get("log_accumulation", True))

    rna_name_to_idx = {n: i for i, n in enumerate(rna_var_names)}
    n_rna = len(rna_var_names)
    delta_cum = np.zeros(n_rna)
    fired_edges: set = set()
    pathway_records: list = []

    # --- 6. Bin loop: ATAC→RNA only (direct, no 8a cascade) ---
    # Read t-1 starting from t=1 (covers the perturbation injected into bin 0); each (peak,gene) edge accumulates only on its first fire
    n_affected_edges = 0
    for t in range(1, n_bins):
        a_orig = float(atac_binned_scaled[t - 1, p_idx])
        a_new = float(atac_pert[t - 1, p_idx])
        delta_a = a_new - a_orig
        if abs(delta_a) < 1e-10:
            continue

        for gi in gene_infos:
            gene_idx = rna_name_to_idx.get(gi["gene"])
            if gene_idx is None:
                continue
            edge_key = (p_idx, gene_idx)
            if edge_key in fired_edges:
                continue

            delta_r = 0.0
            mechanism = "linear"
            _delta_computed = False
            if use_nn:
                try:
                    delta_r = _nn_predict_delta_r(
                        nn_model, a_orig, a_new, peak_id, gi["gene"],
                        nn_peak_to_idx, nn_gene_to_idx,
                    )
                    corrected_gain = gi["gain"]
                    if corrected_gain != 0 and delta_r * corrected_gain * delta_a < 0:
                        delta_r = -delta_r
                    mechanism = "nn"
                    _delta_computed = True
                except Exception:
                    pass

            if not _delta_computed:
                delta_r = gi["gain"] * delta_a
                mechanism = "linear"

            if (not _log_accumulation) and _rna_log_normalized:
                delta_r = _delta_r_log_to_raw(delta_r, rna_binned_orig[t, gene_idx])
            delta_cum[gene_idx] += delta_r
            fired_edges.add(edge_key)
            n_affected_edges += 1

            pathway_records.append({
                "target_gene": f"PEAK:{peak_id}",
                "tf_name": "peak",
                "peak_id": peak_id,
                "affected_gene": gi["gene"],
                "delta_r_raw": float(delta_r),
                "delta_r_total": float(delta_r),
                "mechanism": mechanism,
                "bin": t,
                "tf_lag": 0,
                "r2_score": gi["r2_score"],
            })

    # --- 7. Output delta_rna (log2FC, same convention as engine L1510-1533) ---
    rna_mean_orig = rna_binned_orig.mean(axis=0)
    if _log_accumulation:
        rna_mean_pert_equiv = np.expm1(np.log1p(rna_mean_orig) + delta_cum)
    else:
        rna_mean_pert_equiv = rna_mean_orig + delta_cum
    _orig_safe = np.where(rna_mean_orig > 0, rna_mean_orig, np.nan)
    _pert_safe = np.where(rna_mean_pert_equiv > 0, rna_mean_pert_equiv, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        log2fc = np.log2(_pert_safe / _orig_safe)
    delta_rna = pd.Series(log2fc, index=rna_var_names)

    # --- 8. delta_atac + WT null-distribution z-score (following the v1.36.1 convention) ---
    atac_pert_unscaled = _unscale_atac(atac_pert, root_atac)
    delta_atac = pd.Series(
        atac_pert_unscaled.mean(axis=0) - atac_binned_orig.mean(axis=0),
        index=atac_var_names,
    )
    _atac_wt_std = np.std(atac_binned_orig, axis=0)
    _atac_z = np.divide(
        delta_atac.values, _atac_wt_std,
        out=np.zeros(len(delta_atac)),
        where=_atac_wt_std > 1e-10,
    )
    atac_z_score = pd.Series(_atac_z, index=atac_var_names)

    # --- 9. Upstream TF attribution (read-only query, no back-propagation) ---
    upstream = pd.DataFrame()
    if tf_peak_weights is not None and not tf_peak_weights.empty:
        upstream = tf_peak_weights[tf_peak_weights["peak_id"] == peak_id].copy()
        if not upstream.empty:
            upstream["motif_supported"] = True
            if "weight" in upstream.columns:
                upstream = upstream.sort_values(
                    "weight", key=lambda s: s.abs(), ascending=False
                )

    logger.info(
        f"  Peak direct finished: {n_affected_edges} peak→gene edges fired, "
        f"{(delta_rna.abs() > 1e-6).sum()} affected genes, "
        f"{len(upstream)} upstream candidate TFs"
    )

    return {
        "peak_id": peak_id,
        "delta_rna": delta_rna,
        "delta_atac": delta_atac,
        "atac_z_score": atac_z_score,
        "upstream_tfs": upstream,
        "pathway_edges": pd.DataFrame(pathway_records),
        "n_bins": n_bins,
        "n_affected_edges": n_affected_edges,
        "prediction_mode": "peak_direct",
        "baseline_accessibility": float(atac_binned_orig[:, p_idx].mean()),
        "perturbed_accessibility": float(atac_pert_unscaled[:, p_idx].mean()),
        "delta_accessibility": float(delta_atac.iloc[p_idx]),
    }


def perturb_peak(
    peak_id: str,
    rna_adata,
    atac_adata,
    pseudotime_df,
    granger_edges: pd.DataFrame,
    transfer_functions: pd.DataFrame,
    tf_peak_weights: pd.DataFrame,
    config: dict,
    transfer_models: Optional[dict] = None,
    root_atac: Optional[np.ndarray] = None,
    cluster_label: str = "",
) -> Dict:
    """Public peak-KO entry point: match → direct propagation → package results
    in the same shape as gene KO.

    Returns
    -------
    dict
        "perturbation_results": DataFrame (affected_gene × delta_rna)
        "atac_changes": BED-ready DataFrame (perturbed peak only)
        "upstream_tfs": DataFrame
        "pathway_edges": DataFrame
    """
    peak_cfg = config.get("perturbation", {}).get("peak_ko", {})
    match_mode = peak_cfg.get("match", "exact")
    matched, match_info = _match_peak(peak_id, atac_adata, match_mode)
    if matched is None:
        logger.warning(
            f"  Peak {peak_id} is not in the peak set represented by the network "
            f"(match={match_mode}, no match found)"
        )
        return {"error": "peak_not_found", "requested_peak": peak_id}

    if matched != peak_id:
        logger.info(
            f"  Peak match: requested {peak_id} → network peak {matched} "
            f"(method={match_info['method']}, distance={match_info['distance']} bp)"
        )
        peak_id = matched

    depth = int(peak_cfg.get("depth", 1))
    if depth >= 2:
        result = _simulate_peak_propagation_forward(
            peak_id, rna_adata, atac_adata, pseudotime_df,
            granger_edges, transfer_functions, tf_peak_weights, config,
            transfer_models=transfer_models, root_atac=root_atac,
            subnetwork_depth=depth,
        )
    else:
        result = _simulate_peak_direct_forward(
            peak_id, rna_adata, atac_adata, pseudotime_df,
            transfer_functions, tf_peak_weights, config,
            transfer_models=transfer_models, root_atac=root_atac,
        )
    if "error" in result:
        return result

    # Package perturbation_results in the same shape as gene KO so downstream
    # aggregation and io can reuse it.
    min_delta = config.get("perturbation", {}).get("min_delta_threshold", 0.0)
    rows = []
    for gene, delta in result["delta_rna"].items():
        if not _reportable_effect(delta, min_delta):
            continue
        rows.append({
            "target_gene": f"PEAK:{peak_id}",
            "affected_gene": gene,
            "delta_rna": float(delta),
            "mediated_by_atac": True,
            "mechanism": "ATAC-mediated",
            "propagation_depth": depth,
            "converged": True,
            "is_fallback": False,
            "perturbation_mode": result.get("prediction_mode", "peak_direct"),
            "cluster": cluster_label,
        })
    result["perturbation_results"] = pd.DataFrame(rows)
    result["target_gene"] = f"PEAK:{peak_id}"
    result["atac_changes"] = _build_atac_changes_df(
        result["delta_atac"], atac_adata,
        atac_z_score=result.get("atac_z_score"),
        z_threshold=config.get("perturbation", {}).get("atac_significance_zscore", 2.0),
    )
    return result


def _simulate_peak_propagation_forward(
    peak_id: str,
    rna_adata,
    atac_adata,
    pseudotime_df,
    granger_edges: pd.DataFrame,
    transfer_functions: pd.DataFrame,
    tf_peak_weights: pd.DataFrame,
    config: dict,
    transfer_models: Optional[dict] = None,
    root_atac: Optional[np.ndarray] = None,
    subnetwork_depth: int = 2,
) -> Dict:
    """Core of peak-perturbation propagation mode: single cluster, full 8a+8b
    loop, TF cascade.

    The bin loop is exactly isomorphic to _simulate_knockout_bin_forward
    (first-trigger dedup via fired_edges / L2 accumulation / TF cascade /
    pathway attribution); it differs only in:
      - initialization: perturb atac_pert[:, p_idx] instead of KO of rna_pert
      - subnetwork: peak-rooted BFS (subnetwork_depth controls propagation levels)
      - no KO application / no proxy TF
    If a downstream gene is a TF, its RNA change triggers secondary peaks at
    step 8a → propagation continues (cascade).
    """
    cfg = config["perturbation"]
    peak_cfg = cfg.get("peak_ko", {})
    strength = float(peak_cfg.get("strength", -1.0))
    peak_mode = peak_cfg.get("mode", "relative")
    min_r2 = cfg.get("min_r2_threshold", 0.3)
    nn_model = nn_peak_to_idx = nn_gene_to_idx = None
    nn_rna_log_normalized = bool(config.get("atac_to_rna", {}).get("log_normalize_rna", True))
    if transfer_models is not None:
        if transfer_models.get("type") != "nn":
            raise ValueError("Only NN transfer models are supported")
        nn_model = _load_nn_from_transfer_models(transfer_models)
        nn_rna_log_normalized = _read_nn_rna_log_normalized(transfer_models, config)
        nn_peak_to_idx = transfer_models["data"]["peak_to_idx"]
        nn_gene_to_idx = transfer_models["data"]["gene_to_idx"]

    # --- 1. Binning + scaling (shared with direct mode) ---
    pcfg = config.get("pseudotime", {})
    smooth_kernel = pcfg.get("smooth_kernel", "hard")
    _overlap = pcfg.get("smooth_overlap_factor") if smooth_kernel == "gaussian" else None
    bin_indices, n_bins = _adaptive_rebin(
        pseudotime_df, rna_adata.n_obs,
        max_bins=pcfg["n_bins"],
        min_cells_per_bin=pcfg.get("min_cells_per_bin", 5),
        target_bins=pcfg["n_bins"] if smooth_kernel == "gaussian" else None,
        overlap_factor=_overlap,
    )
    pseudotime_df["bin"] = bin_indices
    min_cpb = pcfg.get("min_cells_per_bin", 5)
    rna_binned_orig = _bin_expression(rna_adata, pseudotime_df, n_bins,
                                      smooth=smooth_kernel, min_cells_per_bin=min_cpb)
    atac_binned_orig = _bin_expression(atac_adata, pseudotime_df, n_bins,
                                       smooth=smooth_kernel, min_cells_per_bin=min_cpb)
    rna_var_names = list(rna_adata.var_names)
    atac_var_names = list(atac_adata.var_names)
    if peak_id not in atac_var_names:
        return {"error": f"peak {peak_id} not in atac_adata var_names"}
    p_idx = atac_var_names.index(peak_id)

    cfg_atac = config.get("atac_to_rna", {})
    if root_atac is None:
        root_atac = _compute_root_atac(
            atac_adata, pseudotime_df,
            quantile=cfg_atac.get("root_cell_quantile", 0.05),
            min_count=cfg_atac.get("root_cell_min_count", 5),
            floor=cfg_atac.get("root_atac_floor", 0.05),
        )
    atac_binned_scaled = _scale_atac(atac_binned_orig, root_atac)

    # --- 2. Peak perturbation injection (same as direct mode) ---
    a_raw = atac_binned_orig[:, p_idx].copy()
    if peak_mode == "absolute":
        a_pert_raw = np.full_like(a_raw, float(np.clip(strength, 0.0, 1.0)))
    else:
        a_pert_raw = np.clip(a_raw * (1.0 + strength), 0.0, 1.0)
    a_pert_scaled = _scale_atac(
        a_pert_raw.reshape(-1, 1), root_atac[p_idx:p_idx + 1]
    ).ravel()

    rna_pert = rna_binned_orig.copy()
    atac_pert = atac_binned_scaled.copy()
    atac_pert[:, p_idx] = a_pert_scaled

    _a_train_min = float(atac_binned_scaled[:, p_idx].min())
    _a_train_max = float(atac_binned_scaled[:, p_idx].max())
    if (a_pert_scaled.min() < _a_train_min - 1e-9) or (a_pert_scaled.max() > _a_train_max + 1e-9):
        logger.warning(
            f"  Peak perturbation is outside this peak's training-visible range "
            f"[{_a_train_min:.3f}, {_a_train_max:.3f}]; the prediction involves extrapolation."
        )

    # --- 3. Peak-rooted subnetwork + edge indices (BFS depth = propagation levels) ---
    subnet = _extract_peak_subnetwork(
        peak_id, transfer_functions, tf_peak_weights, depth=subnetwork_depth
    )
    all_genes = subnet["genes"]
    all_peaks = subnet["peaks"]
    logger.info(
        f"  Peak subnetwork (depth={subnetwork_depth}): {len(all_genes)} genes, {len(all_peaks)} peaks"
    )

    rna_name_to_idx = {n: i for i, n in enumerate(rna_var_names)}
    atac_name_to_idx = {n: i for i, n in enumerate(atac_var_names)}

    tf_to_peaks: Dict[int, list] = {}
    if tf_peak_weights is not None and not tf_peak_weights.empty:
        has_tf_lag = "tf_lag" in tf_peak_weights.columns
        for _, row in tf_peak_weights.iterrows():
            tf_g = row["tf_gene"]
            pk = row["peak_id"]
            if tf_g not in all_genes or pk not in all_peaks:
                continue
            tf_idx = rna_name_to_idx.get(tf_g)
            pk_idx = atac_name_to_idx.get(pk)
            if tf_idx is None or pk_idx is None:
                continue
            lag = int(row["tf_lag"]) if has_tf_lag else 1
            tf_to_peaks.setdefault(tf_idx, []).append((pk_idx, float(row["weight"]), lag))

    peak_to_genes: Dict[int, list] = {}
    if transfer_functions is not None and not transfer_functions.empty:
        for _, row in transfer_functions.iterrows():
            pk = row["peak_id"]
            gene = row["gene"]
            r2 = float(row.get("r2_score", 0.0))
            if r2 < min_r2:
                continue
            if pk not in all_peaks or gene not in all_genes:
                continue
            pk_idx = atac_name_to_idx.get(pk)
            gene_idx = rna_name_to_idx.get(gene)
            if pk_idx is None or gene_idx is None:
                continue
            peak_to_genes.setdefault(pk_idx, []).append({
                "gene": gene,
                "gene_idx": gene_idx,
                "peak_id": pk,
                "gain": float(row.get("steady_state_gain", 0.0)),
                "r2_score": r2,
            })

    max_lag = 1
    for peaks_list in tf_to_peaks.values():
        for _, _, lag in peaks_list:
            max_lag = max(max_lag, lag)
    if max_lag >= n_bins:
        logger.warning(f"max_lag ({max_lag}) >= n_bins ({n_bins}); cannot simulate")
        return {"error": "max_lag_exceeds_bins"}

    use_nn = nn_model is not None
    _rna_log_normalized = nn_rna_log_normalized
    _log_accumulation = bool(_rna_log_normalized and cfg.get("log_accumulation", True))

    n_rna = len(rna_var_names)
    delta_cum = np.zeros(n_rna)
    fired_edges: set = set()
    pathway_records: list = []

    is_tf = np.zeros(n_rna, dtype=bool)
    for tf_idx in tf_to_peaks:
        is_tf[tf_idx] = True

    _peak_tf_lag: Dict[Tuple[int, int], int] = {}
    for tf_idx, peaks_list in tf_to_peaks.items():
        for peak_idx, _w, lag in peaks_list:
            _peak_tf_lag[(peak_idx, tf_idx)] = lag

    # --- 5. Bin loop (isomorphic to _simulate_knockout_bin_forward L1292-1466) ---
    n_affected_edges = 0
    prev_peak_tf_contrib: Dict[int, Dict[int, float]] = {}
    for t in range(max(max_lag, 1), n_bins):
        active = (np.abs(delta_cum) > 1e-10) & is_tf
        if np.any(active):
            rna_pert[t, active] = rna_binned_orig[t, active] + delta_cum[active]

        curr_peak_tf_contrib: Dict[int, Dict[int, float]] = {}
        for tf_idx, peaks_list in tf_to_peaks.items():
            for peak_idx, weight, lag in peaks_list:
                if t - lag < 0:
                    continue
                delta_tf = rna_pert[t - lag, tf_idx] - rna_binned_orig[t - lag, tf_idx]
                if abs(delta_tf) < 1e-10:
                    continue
                contrib = weight * delta_tf
                atac_pert[t, peak_idx] += contrib
                curr_peak_tf_contrib.setdefault(peak_idx, {}).setdefault(tf_idx, 0.0)
                curr_peak_tf_contrib[peak_idx][tf_idx] += contrib
                n_affected_edges += 1
        np.clip(atac_pert[t, :], -1.0, 1.0, out=atac_pert[t, :])

        for peak_idx, gene_infos in peak_to_genes.items():
            delta_a = atac_pert[t - 1, peak_idx] - atac_binned_scaled[t - 1, peak_idx]
            if abs(delta_a) < 1e-10:
                continue
            a_orig = float(atac_binned_scaled[t - 1, peak_idx])
            a_new = float(atac_pert[t - 1, peak_idx])

            tf_contribs = prev_peak_tf_contrib.get(peak_idx, {})
            total_contrib = sum(tf_contribs.values())
            has_tf_attribution = abs(total_contrib) > 1e-10

            for gi in gene_infos:
                gene_idx = gi["gene_idx"]
                edge_key = (peak_idx, gene_idx)
                if edge_key in fired_edges:
                    continue

                delta_r = 0.0
                mechanism = "linear"
                _delta_computed = False
                if use_nn:
                    try:
                        delta_r = _nn_predict_delta_r(
                            nn_model, a_orig, a_new, gi["peak_id"], gi["gene"],
                            nn_peak_to_idx, nn_gene_to_idx,
                        )
                        corrected_gain = gi["gain"]
                        if corrected_gain != 0 and delta_r * corrected_gain * delta_a < 0:
                            delta_r = -delta_r
                        mechanism = "nn"
                        _delta_computed = True
                    except Exception:
                        pass

                if not _delta_computed:
                    delta_r = gi["gain"] * delta_a
                    mechanism = "linear"

                if (not _log_accumulation) and _rna_log_normalized:
                    delta_r = _delta_r_log_to_raw(delta_r, rna_binned_orig[t, gene_idx])
                delta_cum[gene_idx] += delta_r
                fired_edges.add(edge_key)
                n_affected_edges += 1

                if has_tf_attribution:
                    for tf_idx, tf_contrib in tf_contribs.items():
                        tf_name = rna_var_names[tf_idx]
                        frac = tf_contrib / total_contrib
                        pathway_records.append({
                            "target_gene": f"PEAK:{peak_id}",
                            "tf_name": tf_name,
                            "peak_id": gi["peak_id"],
                            "affected_gene": gi["gene"],
                            "delta_r_raw": float(delta_r * frac),
                            "delta_r_total": float(delta_r),
                            "mechanism": mechanism,
                            "bin": t,
                            "tf_lag": _peak_tf_lag.get((peak_idx, tf_idx), 1),
                            "r2_score": gi["r2_score"],
                        })
                else:
                    pathway_records.append({
                        "target_gene": f"PEAK:{peak_id}",
                        "tf_name": "peak",
                        "peak_id": gi["peak_id"],
                        "affected_gene": gi["gene"],
                        "delta_r_raw": float(delta_r),
                        "delta_r_total": float(delta_r),
                        "mechanism": mechanism,
                        "bin": t,
                        "tf_lag": 0,
                        "r2_score": gi["r2_score"],
                    })

                if is_tf[gene_idx]:
                    rna_pert[t, gene_idx] = rna_binned_orig[t, gene_idx] + delta_cum[gene_idx]

        prev_peak_tf_contrib = curr_peak_tf_contrib

    # --- 6. Output (same conventions as direct mode) ---
    rna_mean_orig = rna_binned_orig.mean(axis=0)
    if _log_accumulation:
        rna_mean_pert_equiv = np.expm1(np.log1p(rna_mean_orig) + delta_cum)
    else:
        rna_mean_pert_equiv = rna_mean_orig + delta_cum
    _orig_safe = np.where(rna_mean_orig > 0, rna_mean_orig, np.nan)
    _pert_safe = np.where(rna_mean_pert_equiv > 0, rna_mean_pert_equiv, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        log2fc = np.log2(_pert_safe / _orig_safe)
    delta_rna = pd.Series(log2fc, index=rna_var_names)

    atac_pert_unscaled = _unscale_atac(atac_pert, root_atac)
    delta_atac = pd.Series(
        atac_pert_unscaled.mean(axis=0) - atac_binned_orig.mean(axis=0),
        index=atac_var_names,
    )
    _atac_wt_std = np.std(atac_binned_orig, axis=0)
    _atac_z = np.divide(
        delta_atac.values, _atac_wt_std,
        out=np.zeros(len(delta_atac)),
        where=_atac_wt_std > 1e-10,
    )
    atac_z_score = pd.Series(_atac_z, index=atac_var_names)

    upstream = pd.DataFrame()
    if tf_peak_weights is not None and not tf_peak_weights.empty:
        upstream = tf_peak_weights[tf_peak_weights["peak_id"] == peak_id].copy()
        if not upstream.empty:
            upstream["motif_supported"] = True
            if "weight" in upstream.columns:
                upstream = upstream.sort_values(
                    "weight", key=lambda s: s.abs(), ascending=False
                )

    logger.info(
        f"  Peak propagation complete: {n_affected_edges} edge propagations, "
        f"{(delta_rna.abs() > 1e-6).sum()} affected genes, "
        f"{len(pathway_records)} pathways, {len(upstream)} upstream candidate TFs"
    )

    return {
        "peak_id": peak_id,
        "delta_rna": delta_rna,
        "delta_atac": delta_atac,
        "atac_z_score": atac_z_score,
        "upstream_tfs": upstream,
        "pathway_edges": pd.DataFrame(pathway_records),
        "n_bins": n_bins,
        "n_affected_edges": n_affected_edges,
        "prediction_mode": "peak_propagation",
        "baseline_accessibility": float(atac_binned_orig[:, p_idx].mean()),
        "perturbed_accessibility": float(atac_pert_unscaled[:, p_idx].mean()),
        "delta_accessibility": float(delta_atac.iloc[p_idx]),
    }
