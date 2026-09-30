"""
CausalBridge Granger causal inference module

Core principle (see GrID-Net, Wu et al., Nature Communications, 2025):
  Infer causal regulatory relationships from time-lagged effects between ATAC and RNA.
  Chromatin accessibility changes occur before transcriptional changes—in the same cell,
  the ATAC signal captures the state at an "earlier time point", while the RNA signal
  captures the state at a "later time point".
  This "cell-state parallax" makes Granger causality testing possible.

Granger causality testing logic:
  - Reduced model: predict current gene expression from its past expression  Y(t) ~ Y(t-1)
  - Full model:    add past accessibility of the candidate peak    Y(t) ~ Y(t-1) + X(t-1)
  - F test: determine whether the peak signal significantly improves prediction → evidence of causality
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.sparse import issparse, csr_matrix
from statsmodels.stats.multitest import multipletests
import logging
from typing import List, Optional

from .kinetics import _adaptive_rebin, _branch_aware_lag_pairs

logger = logging.getLogger(__name__)


# ============================================================================
# Pseudotime inference
# ============================================================================

def infer_pseudotime(
    rna_adata,
    atac_adata,
    config: dict,
) -> pd.DataFrame:
    """
    Infer developmental pseudotime from multi-omics data.

    Method (controlled by config["pseudotime"]["method"]):

    - "dpt" (default): Diffusion Pseudotime via scanpy
      Based on diffusion distance, robust to noise, with no additional dependencies
    - "palantir": Palantir pseudotime
      Based on a diffusion map + entropy, naturally supports multi-lineage branching; requires pip install palantir
    When the selected method is unavailable, automatically fall back to the next method (Palantir → DPT → fallback).

    Returns
    -------
    pseudotime_df : DataFrame
        Columns: cell_barcode, pseudotime, branch (if branching is present)
    """
    cfg = config["pseudotime"]
    method = cfg.get("method", "dpt")

    # --- Step 1: construct a joint low-dimensional representation ---
    rna_adata_copy = rna_adata.copy()
    joint_repr, _ = _build_joint_representation(rna_adata_copy, atac_adata)
    rna_adata_copy.obsm["X_joint"] = joint_repr

    # --- Step 2: infer pseudotime using the configured method (with automatic fallback) ---
    pseudotime, branch_labels = _dispatch_pseudotime_method(
        method, rna_adata_copy, joint_repr, cfg, config
    )

    # --- Step 3: bin pseudotime (single entry point; downstream code only reads) ---
    # Merge branches that are too small (cell count < min_cells_per_branch)
    min_cells_for_branch = cfg.get("min_cells_per_branch", 10)
    if branch_labels is not None and len(np.unique(branch_labels)) > 1:
        unique_branches = np.unique(branch_labels)
        valid_branches = []
        for br in unique_branches:
            n_cells_br = (branch_labels == br).sum()
            if n_cells_br >= min_cells_for_branch:
                valid_branches.append(br)
        if len(valid_branches) < len(unique_branches):
            for br in unique_branches:
                if br in valid_branches:
                    continue
                nearest = min(valid_branches, key=lambda vb:
                    abs(np.median(pseudotime[branch_labels == vb]) -
                        np.median(pseudotime[branch_labels == br])))
                branch_labels[branch_labels == br] = nearest
                logger.warning(
                    f"Branch {br}: only {(branch_labels == br).sum()} cells "
                    f"(<{min_cells_for_branch}); merging into branch {nearest}"
                )

    # Construct a temporary DataFrame for _adaptive_rebin
    temp_df = pd.DataFrame({"pseudotime": pseudotime})
    if branch_labels is not None:
        temp_df["branch"] = branch_labels

    _smooth_kernel = cfg.get("smooth_kernel", "hard")
    _overlap = cfg.get("smooth_overlap_factor") if _smooth_kernel == "gaussian" else None
    bin_indices, total_bins = _adaptive_rebin(
        temp_df, rna_adata_copy.n_obs,
        max_bins=cfg["n_bins"],
        min_cells_per_bin=cfg.get("min_cells_per_bin", 5),
        target_bins=cfg["n_bins"] if _smooth_kernel == "gaussian" else None,
        overlap_factor=_overlap,
    )

    results = pd.DataFrame({
        "pseudotime": pseudotime,
        "bin": bin_indices,
    }, index=rna_adata_copy.obs_names)
    results.index.name = "cell_barcode"
    if branch_labels is not None:
        results["branch"] = branch_labels
    # _adaptive_rebin writes branch_boundaries to temp_df.attrs; transfer them to results
    results.attrs["branch_boundaries"] = temp_df.attrs.get("branch_boundaries")

    bin_counts = results["bin"].value_counts().sort_index()
    logger.info(
        f"Pseudotime inference complete ({method}): "
        f"pseudotime range [{pseudotime.min():.3f}, {pseudotime.max():.3f}], "
        f"{total_bins} bins, "
        f"cells per bin: min={bin_counts.min()}, median={bin_counts.median():.0f}, "
        f"max={bin_counts.max()}"
    )
    return results


def _dispatch_pseudotime_method(method, rna_adata, joint_repr, cfg, config):
    """Dispatch pseudotime inference according to method, falling back on failure."""
    fallback_chain = {
        "palantir": ["dpt", "fallback"],
        "dpt":      ["fallback"],
    }
    methods = [method] + fallback_chain.get(method, ["dpt", "fallback"])

    for m in methods:
        if m == "palantir":
            if _palantir_available():
                result = _pseudotime_via_palantir(rna_adata, joint_repr, cfg, config)
                if result is not None:
                    return result
            logger.warning("Palantir unavailable; falling back to the next method")
        elif m == "dpt":
            if _scanpy_available():
                return _pseudotime_via_dpt(rna_adata, joint_repr, cfg, config)
            logger.warning("scanpy unavailable; falling back to the next method")
        elif m == "fallback":
            logger.warning("Using the custom diffusion pseudotime implementation (fallback)")
            return _diffusion_pseudotime_fallback(joint_repr, rna_adata, cfg, config)

    return _diffusion_pseudotime_fallback(joint_repr, rna_adata, cfg, config)


def _scanpy_available() -> bool:
    """Check whether scanpy is available."""
    try:
        import scanpy  # noqa
        return True
    except ImportError:
        return False


def _palantir_available() -> bool:
    """Check whether Palantir is available."""
    try:
        import palantir  # noqa
        return True
    except ImportError:
        return False


def _build_joint_representation(rna_adata, atac_adata):
    """
    Build a joint low-dimensional representation of ATAC+RNA.

    Uses a WNN-style strategy: reduce dimensionality independently for each
    modality, then concatenate.
    Improvements over naively concatenating PCA embeddings:
    - Uses scanpy.pp.pca instead of sklearn PCA (better support for sparse matrices)
    - Optionally weights the two modalities (normalized via explained_variance_ratio)
    """
    rna_mat = rna_adata.X.toarray() if issparse(rna_adata.X) else np.array(rna_adata.X)
    atac_mat = (
        atac_adata.X.toarray() if issparse(atac_adata.X) else np.array(atac_adata.X)
    )

    # scanpy PCA (when available, use the more efficient and canonical implementation)
    if _scanpy_available():
        import scanpy as sc
        rna_temp = rna_adata.copy()
        atac_temp = atac_adata.copy()
        rna_temp.X = rna_mat
        atac_temp.X = atac_mat
        sc.pp.pca(rna_temp, n_comps=15, zero_center=True)
        sc.pp.pca(atac_temp, n_comps=15, zero_center=True)
        pca_rna = rna_temp.obsm["X_pca"]
        pca_atac = atac_temp.obsm["X_pca"]
        # Weight by explained variance so that the two modalities contribute in a balanced way
        rna_weight = np.sqrt(rna_temp.uns["pca"]["variance_ratio"].sum())
        atac_weight = np.sqrt(atac_temp.uns["pca"]["variance_ratio"].sum())
        total_weight = rna_weight + atac_weight + 1e-10
        pca_rna = pca_rna * (atac_weight / total_weight)
        pca_atac = pca_atac * (rna_weight / total_weight)
    else:
        from sklearn.decomposition import PCA
        pca_rna = PCA(n_components=15).fit_transform(rna_mat)
        pca_atac = PCA(n_components=15).fit_transform(atac_mat)

    joint_repr = np.hstack([pca_rna, pca_atac])
    logger.info(f"Joint low-dimensional representation: {joint_repr.shape}")

    return joint_repr, rna_adata


def _pseudotime_via_dpt(rna_adata_with_repr, joint_repr, cfg, config):
    """
    Diffusion Pseudotime (DPT) via scanpy.

    DPT principle (Haghverdi et al., Nature Methods, 2016):
    1. Define a diffusion process (random walk) on the KNN graph
    2. Compute the diffusion distance between any two points — accounting for
       all possible paths, not just the shortest path
    3. The diffusion distance from the root cell = pseudotime
    """
    import scanpy as sp

    adata = rna_adata_with_repr.copy()
    adata.obsm["X_pca"] = joint_repr
    adata.X = adata.X.toarray() if issparse(adata.X) else adata.X

    sp.pp.neighbors(
        adata,
        n_neighbors=cfg["n_neighbors"],
        use_rep="X_pca",
        random_state=42,
    )

    root_idx = _find_root_cell(
        rna_adata_with_repr, cfg.get("root_cell_marker"),
        method=cfg.get("root_cell_method"),
        n_top_genes=cfg.get("cytotrace_n_top_genes", 200),
    )
    adata.uns["iroot"] = root_idx

    sp.tl.dpt(adata, n_branchings=0)

    pseudotime = adata.obs["dpt_pseudotime"].values.astype(np.float64)
    pseudotime[np.isinf(pseudotime)] = np.nanmax(
        pseudotime[~np.isinf(pseudotime)]
    ) * 1.1

    logger.info(f"DPT pseudotime inference complete (scanpy), root cell #{root_idx}")
    return pseudotime, None


def _pseudotime_via_palantir(rna_adata_with_repr, joint_repr, cfg, config):
    """
    Palantir pseudotime inference.

    Palantir principle (Setty et al., Nature Biotechnology, 2019):
    1. Diffusion map embedding → captures the global structure of the developmental trajectory
    2. Shortest paths from the root cell → initial pseudotime
    3. Entropy computation → differentiation potential (cells with low entropy sit at the end of differentiation)
    4. Supports multi-lineage branching detection

    Requires pip install palantir
    """
    import scanpy as sp
    import palantir

    adata = rna_adata_with_repr.copy()
    adata.obsm["X_pca"] = joint_repr
    adata.X = adata.X.toarray() if issparse(adata.X) else adata.X

    sp.pp.neighbors(
        adata,
        n_neighbors=cfg["n_neighbors"],
        use_rep="X_pca",
        random_state=42,
    )

    palantir.utils.run_diffusion_maps(adata)
    palantir.utils.determine_multiscale_space(adata)

    root_idx = _find_root_cell(
        rna_adata_with_repr, cfg.get("root_cell_marker"),
        method=cfg.get("root_cell_method"),
        n_top_genes=cfg.get("cytotrace_n_top_genes", 200),
    )
    root_cell = adata.obs_names[root_idx]

    pr_res = palantir.utils.run_palantir(
        adata, root_cell, num_waypoints=min(500, adata.n_obs)
    )

    pseudotime = pr_res.pseudotime.values.astype(np.float64)
    pseudotime[np.isinf(pseudotime)] = np.nanmax(
        pseudotime[~np.isinf(pseudotime)]
    ) * 1.1
    pseudotime = np.nan_to_num(pseudotime, nan=np.nanmedian(pseudotime))

    branch_labels = None
    if pr_res.branch_probs is not None and not pr_res.branch_probs.empty:
        branch_labels = np.asarray(
            pr_res.branch_probs.idxmax(axis=1)
        )
        logger.info(
            f"Palantir detected {pr_res.branch_probs.shape[1]} branches, "
            f"branch_probs shape={pr_res.branch_probs.shape}"
        )
    else:
        logger.info(
            "Palantir detected no branches (branch_probs is empty/None); "
            "global pseudotime binning will be used"
        )

    logger.info(f"Palantir pseudotime inference complete, root cell #{root_idx}")
    return pseudotime, branch_labels


def _diffusion_pseudotime_fallback(joint_repr, rna_adata, cfg, config):
    """
    Self-implemented diffusion pseudotime (fallback when scanpy is unavailable).

    Simplified diffusion pseudotime:
    1. Build a mutual KNN graph (mutual nearest neighbors) to reduce noisy edges
    2. Use a diffusion process: T = D^{-1/2} · A_sym · D^{-1/2}
    3. After spectral decomposition, take the second eigenvector (the first
       non-trivial component of the diffusion map)
    4. Anchor the direction using the root cell

    This is a fundamental improvement over the initial Dijkstra shortest-path version:
    - Shortest paths are extremely sensitive to the weight of a single edge
    - Diffusion mapping accounts for all paths and is naturally smooth
    """
    from sklearn.neighbors import NearestNeighbors

    n_cells = joint_repr.shape[0]
    n_neighbors = min(cfg["n_neighbors"], n_cells - 1)

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(joint_repr)
    distances, indices = nn.kneighbors(joint_repr)

    # Build a mutual KNN graph (mutual nearest neighbors) — an edge exists only
    # when two cells are neighbors in both directions
    # This is sparser and more reliable than the full KNN graph
    rows, cols, data = [], [], []
    for i in range(n_cells):
        for j_idx, d in zip(indices[i], distances[i]):
            if i in indices[j_idx]:  # mutual-neighbor check
                rows.append(i)
                cols.append(j_idx)
                weight = np.exp(-d ** 2 / (np.median(distances) ** 2 + 1e-10))
                data.append(weight)

    if len(data) == 0:
        # If the mutual KNN is too strict and yields no edges, fall back to the full KNN
        logger.warning("Mutual KNN graph is empty; falling back to the full KNN graph")
        for i in range(n_cells):
            for j_idx, d in zip(indices[i], distances[i]):
                rows.append(i)
                cols.append(j_idx)
                data.append(np.exp(-d / (np.median(distances) + 1e-10)))

    adj = csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
    adj = (adj + adj.T) / 2  # symmetrize

    # Diffusion map: take the second eigenvector (the first non-trivial component)
    from scipy.sparse.linalg import eigsh

    degree = np.array(adj.sum(axis=1)).flatten()
    degree_sqrt_inv = np.diag(1.0 / np.sqrt(degree + 1e-10))
    laplacian = degree_sqrt_inv @ adj @ degree_sqrt_inv

    try:
        _, eigenvectors = eigsh(laplacian, k=3, which="LM")
        diffusion_component = eigenvectors[:, 1]  # second eigenvector
    except Exception:
        # fallback: simple spectral decomposition
        from scipy.sparse.csgraph import dijkstra
        root_idx = _find_root_cell(
        rna_adata, cfg.get("root_cell_marker"),
        method=cfg.get("root_cell_method"),
        n_top_genes=cfg.get("cytotrace_n_top_genes", 200),
    )
        diffusion_component = dijkstra(adj, directed=False, indices=root_idx)

    # Determine direction: make the root-cell end equal 0
    root_idx = _find_root_cell(
        rna_adata, cfg.get("root_cell_marker"),
        method=cfg.get("root_cell_method"),
        n_top_genes=cfg.get("cytotrace_n_top_genes", 200),
    )
    if diffusion_component[root_idx] > np.median(diffusion_component):
        diffusion_component = -diffusion_component
    pseudotime = diffusion_component - diffusion_component[root_idx]
    pseudotime = np.maximum(pseudotime, 0)  # non-negativity constraint

    branch_labels = None
    logger.info(f"Self-implemented diffusion pseudotime inference complete, root cell #{root_idx}")
    return pseudotime, branch_labels


def _compute_cytotrace_scores(
    rna_adata,
    n_top_genes: int = 200,
) -> np.ndarray:
    """CytoTRACE 2020 (Gulati et al., Science): per-cell differentiation potential.

    Higher score = more stem-like / undifferentiated (pluripotent).

    Algorithm:
      1. Gene Count (gc): the number of genes with expression > 0. Fewer → more stem-like.
      2. Mean Pairwise Correlation (mpc): the mean pairwise Pearson correlation among
         each cell's own top-N highly expressed genes (correlations are computed across
         the full dataset). Higher → more stem-like
         (high transcriptional coordination indicates the cell is in a more upstream progenitor state).
      3. CytoTRACE score = (rev_rank(gc) + rank(mpc)) / 2, both normalized to (0, 1].

    Notes:
    - Prefer ``rna_adata.layers["raw"]`` (raw counts) for computing gc;
      when no raw layer exists, fall back to ``.X`` (if already log-normalized, gc values
      are more continuous but still comparable).
    - The gene universe is the current gene set of rna_adata (typically already filtered
      by preprocess_rna to HVG + target genes, ~3K-3.5K), so the gene-gene correlation
      matrix stays manageable in size.
    - Computational complexity is O(G²·n + n·G·log_G), which runs in seconds at HVG scale.
    """
    from scipy.stats import rankdata

    X = rna_adata.layers.get("raw")
    if X is None:
        X = rna_adata.X
    if issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    n_cells, n_genes = X.shape

    if n_cells == 0 or n_genes == 0:
        logger.warning("CytoTRACE: empty matrix, returning zeros")
        return np.zeros(n_cells)

    # --- 1. Gene count ---
    gc = (X > 0).sum(axis=1).astype(float)

    n_top = int(min(n_top_genes, n_genes))
    if n_top < 2 or n_cells < 3:
        # Not enough data to compute correlations → use gene count alone as a single signal (revrank)
        cyto = (n_cells - rankdata(gc)) / max(n_cells, 1)
        return cyto

    # --- 2. Build the G x G gene-gene Pearson correlation matrix (across all cells) ---
    X_centered = X - X.mean(axis=0, keepdims=True)
    X_std = X_centered.std(axis=0, ddof=0)
    valid_genes = X_std > 1e-10
    X_normed = np.zeros_like(X_centered)
    X_normed[:, valid_genes] = X_centered[:, valid_genes] / X_std[valid_genes]
    gene_corr_mat = (X_normed.T @ X_normed) / max(n_cells, 1)
    np.fill_diagonal(gene_corr_mat, 0.0)

    # --- 3. For each cell, take the mean of the sub-correlation matrix over its own top-N genes ---
    triu_i, triu_j = np.triu_indices(n_top, k=1)
    mpc = np.zeros(n_cells)
    for c in range(n_cells):
        expr = X[c]
        if n_top < n_genes:
            top_idx = np.argpartition(expr, -n_top)[-n_top:]
        else:
            top_idx = np.arange(n_genes)
        sub_corr = gene_corr_mat[np.ix_(top_idx, top_idx)]
        vals = sub_corr[triu_i, triu_j]
        mpc[c] = float(vals.mean()) if vals.size else 0.0

    # --- 4. Composite score (rank-normalized into (0, 1]) ---
    rev_gc_rank = (n_cells - rankdata(gc)) / max(n_cells, 1)
    mpc_rank = rankdata(mpc) / max(n_cells, 1)
    cyto = (rev_gc_rank + mpc_rank) / 2.0
    return cyto


def _find_root_cell(
    rna_adata,
    marker_gene: Optional[str] = None,
    method: Optional[str] = None,
    n_top_genes: int = 200,
) -> int:
    """
    Determine the root cell (starting point) of the developmental trajectory.
    The root-selection strategy is controlled by ``method``.

    method (None / "auto" / "marker" / "cytotrace" / "min_umi"):
      - "marker":    force use of marker_gene; fall back to min_umi if the marker is missing
      - "cytotrace": use CytoTRACE 2020 stemness scores and pick the cell with the
                     highest stemness within the cluster (read the global cache in
                     obs["cytotrace_score"] first; if absent, compute on the fly within this cluster)
      - "min_umi":   the cell with the lowest transcriptional diversity (standard approach in the DPT paper)
      - None / "auto": marker_gene given → marker; otherwise default to cytotrace (new default)
    """
    valid = {"marker", "cytotrace", "min_umi", "auto", None}
    if method not in valid:
        logger.warning(f"Unknown root_cell_method='{method}'; falling back to auto")
        method = None

    use = method
    if use in (None, "auto"):
        use = "marker" if marker_gene else "cytotrace"

    # --- (1) marker ---
    if use == "marker":
        if marker_gene and marker_gene in rna_adata.var_names:
            expr = (
                rna_adata[:, marker_gene].X.toarray().flatten()
                if issparse(rna_adata[:, marker_gene].X)
                else rna_adata[:, marker_gene].X.flatten()
            )
            root = int(np.argmax(expr))
            logger.info(f"Root cell selected by marker gene: {marker_gene} → cell #{root}")
            return root
        logger.info(f"marker method requested but marker='{marker_gene}' is missing; falling back to min_umi")
        use = "min_umi"

    # --- (2) cytotrace ---
    if use == "cytotrace":
        if "cytotrace_score" in rna_adata.obs.columns:
            root = int(np.argmax(rna_adata.obs["cytotrace_score"].values))
            logger.info(
                f"Root cell selected by CytoTRACE: cell #{root} "
                f"(stemness={rna_adata.obs['cytotrace_score'].iloc[root]:.4f}, "
                f"from the global obs cache)"
            )
            return root
        # Compute on the fly within the sub-cluster (scheme A fallback; triggered when
        # the pipeline has not injected global scores)
        scores = _compute_cytotrace_scores(rna_adata, n_top_genes=n_top_genes)
        root = int(np.argmax(scores))
        logger.info(
            f"Root cell computed on the fly by CytoTRACE: cell #{root} "
            f"(stemness={scores[root]:.4f})"
        )
        return root

    # --- (3) min_umi ---
    raw_counts = rna_adata.layers.get("raw", rna_adata.X)
    total_umi = np.array(
        raw_counts.sum(axis=1) if issparse(raw_counts)
        else raw_counts.sum(axis=1)
    ).flatten()
    if total_umi.min() > 0:
        root = int(np.argmin(total_umi))
        logger.info(f"Root cell selected by transcriptional activity: cell #{root}")
        return root
    return 0


# ============================================================================
# Granger causality testing
# ============================================================================

def granger_test(
    rna_adata,
    atac_adata,
    pseudotime_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Run Granger causality testing for every candidate peak-gene pair.

    Candidate pair generation rules:
    - The genomic distance between the peak and the gene's TSS is < distance_thresh (default 1Mb)
    - Both signals vary sufficiently along pseudotime (constant signals are excluded)

    For each candidate pair:
    1. Build time-lagged features on binned pseudotime:
       - Y(t) = mean expression of the gene in bin t
       - Y(t-1) = mean expression of the gene in bin t-1
       - X(t-1) = mean accessibility of the peak in bin t-1
    2. Fit two linear models:
       - Reduced:  Y(t) ~ Y(t-1)
       - Full:     Y(t) ~ Y(t-1) + X(t-1)
    3. F test: does the lagged ATAC term significantly improve prediction (F(df=1, n-3))
    4. Multiple-testing correction (FDR)

    Returns
    -------
    results : DataFrame
        Columns: peak_id, gene, peak_chr, peak_start, peak_end, gene_chr, gene_tss,
             F_statistic, p_value, p_adj, delta_r2, is_causal
    """
    cfg = config["granger"]
    pcfg = config["pseudotime"]
    ablation_mode = cfg.get("ablation_lag0", False)

    if ablation_mode:
        logger.info("=== Granger ablation mode: drop the time lag and test the pure correlation Y(t)~X(t) ===")

    # Adaptive binning based on cluster size
    _smooth_kernel = pcfg.get("smooth_kernel", "hard")
    _overlap = pcfg.get("smooth_overlap_factor") if _smooth_kernel == "gaussian" else None

    # Debug logging: check whether pseudotime has been shuffled
    if hasattr(pseudotime_df, 'attrs') and pseudotime_df.attrs.get("_need_rebin"):
        logger.info(f"  [DEBUG] granger_test: detected the _need_rebin flag; pseudotime has been shuffled")
        logger.info(f"  [DEBUG] pseudotime range: [{pseudotime_df['pseudotime'].min():.4f}, {pseudotime_df['pseudotime'].max():.4f}]")

    bin_indices, n_bins = _adaptive_rebin(
        pseudotime_df, rna_adata.n_obs,
        max_bins=pcfg["n_bins"],
        min_cells_per_bin=pcfg.get("min_cells_per_bin", 5),
        target_bins=pcfg["n_bins"] if _smooth_kernel == "gaussian" else None,
        overlap_factor=_overlap,
    )
    pseudotime_df["bin"] = bin_indices
    mode_label = "ablation" if ablation_mode else "Granger"
    logger.info(f"{mode_label} binning: {rna_adata.n_obs} cells → {n_bins} bins "
                f"(smooth={_smooth_kernel})")

    # Debug logging: check the recomputed bin distribution
    if hasattr(pseudotime_df, 'attrs') and pseudotime_df.attrs.get("_need_rebin"):
        bin_counts = np.bincount(bin_indices.astype(int))
        logger.info(f"  [DEBUG] recomputed bin distribution: {bin_counts[:5]}...")

    # --- Step 1: aggregate ATAC and RNA signals over binned pseudotime ---
    # Save the pre-shuffle binning result for comparison
    _need_rebin = hasattr(pseudotime_df, 'attrs') and pseudotime_df.attrs.get("_need_rebin")
    if _need_rebin:
        rna_binned_before = _bin_expression(rna_adata, pseudotime_df, n_bins,
                                           smooth="hard",
                                           min_cells_per_bin=pcfg.get("min_cells_per_bin", 5))
    
    rna_binned = _bin_expression(rna_adata, pseudotime_df, n_bins,
                                 smooth=pcfg.get("smooth_kernel", "hard"),
                                 min_cells_per_bin=pcfg.get("min_cells_per_bin", 5))
    atac_binned = _bin_expression(atac_adata, pseudotime_df, n_bins,
                                  smooth=pcfg.get("smooth_kernel", "hard"),
                                  min_cells_per_bin=pcfg.get("min_cells_per_bin", 5))

    logger.info(f"Binned data: RNA {rna_binned.shape}, ATAC {atac_binned.shape}")

    # Debug logging: check whether the binning result actually changed
    if _need_rebin:
        rna_diff = np.abs(rna_binned - rna_binned_before).max()
        logger.info(f"  [DEBUG] max difference in RNA binning before vs. after shuffling: {rna_diff:.6f}")
        if rna_diff < 1e-6:
            logger.warning("  ⚠️ Warning: RNA binning results before and after shuffling are almost identical!")
        else:
            logger.info(f"  ✓ RNA binning results before and after shuffling differ; the difference is significant")

    # --- Lag pre-slicing ---
    branch_boundaries = pseudotime_df.attrs.get("branch_boundaries") if hasattr(pseudotime_df, "attrs") else None

    if ablation_mode:
        # Ablation mode: use same-time-point data, no lag processing
        # Tests the pure correlation Y(t) ~ X(t)
        _rna_t = rna_binned
        _rna_t1 = None  # ablation mode does not need Y(t-1)
        _atac_lag_mats = None
        _atac_current = atac_binned  # X(t): same-time-point ATAC
        n_valid_pairs = rna_binned.shape[0]
        logger.info(f"Ablation mode valid pairs: {n_valid_pairs} (no lag, contemporaneous)")
    else:
        # Standard Granger mode: use max_lag from the configuration
        granger_max_lag = cfg.get("max_lag", 1)
        logger.info(f"Granger mode: lag={granger_max_lag}")

        # Tests Y(t) ~ Y(t-lag) + X(t-lag)
        rna_lag0, rna_lag1 = _branch_aware_lag_pairs(
            rna_binned, rna_binned, lag=granger_max_lag,
            branch_boundaries=branch_boundaries,
        )
        atac_lag0, atac_lag1 = _branch_aware_lag_pairs(
            atac_binned, atac_binned, lag=granger_max_lag,
            branch_boundaries=branch_boundaries,
        )
        _rna_t = rna_lag1      # Y(t): RNA at time t
        _rna_t1 = rna_lag0     # Y(t-lag): RNA at time t-lag
        _atac_lag_mats = [atac_lag0]  # X(t-lag): ATAC at time t-lag
        _atac_current = atac_lag1     # X(t): ATAC at time t (spare)
        n_valid_pairs = _rna_t.shape[0]
        logger.info(f"Granger lag slicing: {n_valid_pairs} valid pairs (lag={granger_max_lag})")

    if n_valid_pairs < 5:
        logger.warning(f"Lag slicing produced only {n_valid_pairs} valid pairs; "
                       f"the Granger test cannot be performed")
        return pd.DataFrame()

    # --- Step 2: generate candidate peak-gene pairs ---
    candidates = _generate_candidate_pairs(
        rna_adata, atac_adata, cfg["distance_thresh"]
    )
    logger.info(f"Candidate peak-gene pairs: {len(candidates)}")

    if len(candidates) == 0:
        logger.warning("No candidate peak-gene pairs were found; please check the genomic coordinate annotations")
        return pd.DataFrame()

    # --- Step 3: Granger test pair by pair ---
    # Pre-build index maps to avoid an O(n) lookup every time
    atac_name_to_idx = {name: i for i, name in enumerate(atac_adata.var_names)}
    rna_name_to_idx = {name: i for i, name in enumerate(rna_adata.var_names)}
    # Pre-extract genomic position info into numpy arrays to avoid repeated loc indexing
    atac_chr_arr = atac_adata.var["chr"].values
    atac_start_arr = atac_adata.var["start"].values.astype(int)
    atac_end_arr = atac_adata.var["end"].values.astype(int)
    rna_chr_arr = rna_adata.var["chr"].values
    rna_start_arr = rna_adata.var["start"].values.astype(int)

    # Pre-compute each gene's autoregressive SSR (each gene is used only once)
    ones = np.ones(n_valid_pairs)
    gene_ssr_cache = {}

    results = []
    total = len(candidates)
    log_interval = max(total // 20, 1000)  # log progress every 5%

    for i, (peak_id, gene) in enumerate(candidates):
        peak_idx = atac_name_to_idx.get(peak_id)
        gene_idx = rna_name_to_idx.get(gene)
        if peak_idx is None or gene_idx is None:
            continue

        Y_t = _rna_t[:, gene_idx]
        Y_lag = _rna_t1[:, gene_idx] if _rna_t1 is not None else None

        if np.var(Y_t) < 1e-6:
            continue

        # Reduced model: Y(t) ~ Y(t-1); cache the result so each gene is not computed repeatedly
        if ablation_mode:
            # Ablation mode: Reduced = Y(t) ~ 1 (intercept only, no autoregression)
            ssr_reduced = float(np.sum((Y_t - np.mean(Y_t)) ** 2))
            if ssr_reduced < 1e-10:
                continue
        else:
            if gene_idx in gene_ssr_cache:
                ssr_reduced = gene_ssr_cache[gene_idx]
            else:
                Y_lag_const = np.column_stack([ones, Y_lag])
                _, residuals_reduced, _, _ = np.linalg.lstsq(
                    Y_lag_const, Y_t, rcond=None
                )
                ssr_reduced = np.sum(residuals_reduced ** 2)
                if ssr_reduced < 1e-10:
                    ssr_reduced = 0.0
                gene_ssr_cache[gene_idx] = ssr_reduced

        if ssr_reduced < 1e-10:
            continue

        # Full model
        if ablation_mode:
            # Ablation mode: Y(t) ~ X(t) (same-time-point peak, no lag)
            x_col = _atac_current[:, peak_idx]
            if np.var(x_col) < 1e-6:
                continue
            X_const = np.column_stack([ones, x_col])
            coeffs, residuals_full, _, _ = np.linalg.lstsq(
                X_const, Y_t, rcond=None
            )
            atac_coef = float(coeffs[1])  # β₁: ATAC_t → RNA_t
            ssr_full = np.sum(residuals_full ** 2)
            df_full = n_valid_pairs - 2
        else:
            # Standard Granger: Y(t) ~ Y(t-1) + X(t-1)
            x_col = _atac_lag_mats[0][:, peak_idx]
            if np.var(x_col) < 1e-6:
                continue
            YX_lag_const = np.column_stack([ones, Y_lag, x_col])
            coeffs, residuals_full, _, _ = np.linalg.lstsq(
                YX_lag_const, Y_t, rcond=None
            )
            atac_coef = float(coeffs[2])  # β₂: ATAC_{t-1} → RNA_t
            ssr_full = np.sum(residuals_full ** 2)
            df_full = n_valid_pairs - 3

        if ssr_full < 1e-10 or df_full <= 0:
            continue

        f_num = ssr_reduced - ssr_full
        f_den = ssr_full / df_full
        if f_den <= 0:
            continue

        f_stat = f_num / f_den
        p_value = 1 - stats.f.cdf(f_stat, 1, df_full)
        delta_r2 = 1 - ssr_full / ssr_reduced

        # Pearson r: X vs Y — used for direction judgment (standard mode=lagged, ablation mode=contemporaneous)
        pearson_r = float(np.corrcoef(x_col, Y_t)[0, 1]) if np.std(x_col) > 1e-6 else 0.0

        results.append({
            "peak_id": peak_id,
            "gene": gene,
            "peak_chr": atac_chr_arr[peak_idx],
            "peak_start": int(atac_start_arr[peak_idx]),
            "peak_end": int(atac_end_arr[peak_idx]),
            "gene_chr": rna_chr_arr[gene_idx],
            "gene_tss": int(rna_start_arr[gene_idx]),
            "F_statistic": f_stat,
            "p_value": p_value,
            "delta_r2": delta_r2,
            "atac_coef": atac_coef,
            "pearson_r": pearson_r,
        })

        if (i + 1) % log_interval == 0:
            pct = (i + 1) / total * 100
            mode_label = "ablation" if ablation_mode else "Granger"
            logger.info(f"  {mode_label} test progress: {i+1}/{total} ({pct:.0f}%), "
                        f"{sum(1 for r in results if r['p_value'] < 0.05)} significant pairs found so far")

    if len(results) == 0:
        mode_label = "ablation" if ablation_mode else "Granger"
        logger.warning(f"The {mode_label} test produced no valid results")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # --- Step 4: multiple-testing correction (FDR) ---
    _, p_adj, _, _ = multipletests(
        results_df["p_value"].values,
        method=cfg["correction_method"],
    )
    results_df["p_adj"] = p_adj

    # --- Step 5: causality determination ---
    sig_threshold = cfg["significance_threshold"]
    min_effect = cfg["min_effect_size"]

    # Degrees-of-freedom correction: few bins → few observations → inflated ΔR² (~6x @23bin)
    # correction = (df_ref / df_actual)², a quadratic penalty that is stricter for clusters with few bins
    n_ref_bins = cfg.get("reference_bins", 50)
    df_ref = n_ref_bins - 3
    df_actual = max(n_valid_pairs - 3, 1)
    correction = (df_ref / df_actual) ** 2
    corrected_sig = sig_threshold / correction
    corrected_effect = min_effect * correction
    logger.info(
        f"Granger degrees-of-freedom correction: n_bins={n_bins}, "
        f"df_actual={df_actual}, "
        f"correction={correction:.3f}, p_threshold={sig_threshold:.4f}→{corrected_sig:.4f}, "
        f"effect_threshold={min_effect:.3f}→{corrected_effect:.3f}"
    )

    # Composite score (always computed, exported to the results table)
    # p_score: the smaller p_adj is → the closer to 1
    p_score = np.clip(1.0 - results_df["p_adj"].values / corrected_sig, 0, 1)
    # effect_score: the larger delta_r2 is → the closer to 1
    effect_score = np.clip(
        results_df["delta_r2"].values / max(corrected_effect, 1e-10), 0, 2
    ) / 2.0
    w = cfg.get("composite_weights", [0.4, 0.6])
    results_df["composite_score"] = p_score * w[0] + effect_score * w[1]

    if cfg.get("soft_threshold", False):
        min_score = cfg.get("min_composite_score", 0.3)
        results_df["is_causal"] = results_df["composite_score"] >= min_score
        logger.info(
            f"Using soft-threshold filtering (min_composite_score={min_score})"
        )
    else:
        results_df["is_causal"] = (
            (results_df["p_adj"] < corrected_sig)
            & (results_df["delta_r2"] > corrected_effect)
        )

    n_causal = results_df["is_causal"].sum()
    logger.info(
        f"Granger test complete: {n_causal}/{len(results_df)} pairs passed the causality filter "
        f"(p_adj < {corrected_sig:.2e}, "
        f"Delta R2 > {corrected_effect:.3f}, "
        f"df_correction={correction:.2f})"
    )

    return results_df


def _bin_expression(adata, pseudotime_df, n_bins: int,
                    smooth: str = "hard", min_cells_per_bin: int = 15) -> np.ndarray:
    """Aggregate each gene/peak's expression/accessibility over pseudotime bins.

    smooth="hard": hard binning; take the mean over cells within each bin (original behavior)
    smooth="gaussian": Gaussian-kernel weighted smoothing; each bin shares information
                       from neighboring bins, reducing the noise of small-cluster bin means;
                       min_cells_per_bin controls sigma
    """
    mat = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)

    if smooth == "hard":
        binned = np.zeros((n_bins, mat.shape[1]))
        for b in range(n_bins):
            mask = pseudotime_df["bin"].values == b
            if mask.sum() > 0:
                binned[b] = mat[mask].mean(axis=0)
        return binned

    # --- Gaussian kernel smoothing ---
    pseudo_vals = pseudotime_df["pseudotime"].values.astype(float)
    pseudo_min, pseudo_max = pseudo_vals.min(), pseudo_vals.max()
    pseudo_span = pseudo_max - pseudo_min
    if pseudo_span <= 0:
        pseudo_span = 1.0

    n_cells = len(pseudo_vals)
    avg_cells_per_bin = n_cells / max(n_bins, 1)
    base_sigma = pseudo_span / n_bins

    # Adaptive sigma: increase sigma for small clusters so they can borrow information from neighboring bins
    if avg_cells_per_bin >= min_cells_per_bin:
        sigma = base_sigma
    else:
        sigma = base_sigma * (min_cells_per_bin / max(avg_cells_per_bin, 1))

    bin_width = pseudo_span / n_bins
    bin_centers = np.linspace(pseudo_min + bin_width / 2, pseudo_max - bin_width / 2, n_bins)

    diff = bin_centers[:, None] - pseudo_vals[None, :]  # (n_bins, n_cells)
    W = np.exp(-0.5 * (diff / sigma) ** 2)
    W /= W.sum(axis=1, keepdims=True) + 1e-10

    return W @ mat


def _generate_candidate_pairs(
    rna_adata, atac_adata, distance_thresh: int
) -> List[tuple]:
    """
    Generate candidate peak-gene pairs based on genomic distance.

    Optimization: use np.searchsorted over the sorted gene TSSs to do an
    interval search, avoiding building a full (n_peaks × n_genes) distance matrix.
    """
    pairs = []
    atac_chr = atac_adata.var["chr"].values
    atac_mid = ((atac_adata.var["start"] + atac_adata.var["end"]) / 2).values
    rna_chr = rna_adata.var["chr"].values
    rna_tss = rna_adata.var["start"].values.astype(float)

    for chrom in np.unique(atac_chr):
        atac_mask = atac_chr == chrom
        rna_mask = rna_chr == chrom
        if not atac_mask.any() or not rna_mask.any():
            continue

        atac_mids = atac_mid[atac_mask]
        atac_names = atac_adata.var_names[atac_mask]
        rna_tsss = rna_tss[rna_mask]
        rna_names = rna_adata.var_names[rna_mask]

        # Sort by TSS to enable interval search
        sort_idx = np.argsort(rna_tsss)
        sorted_tss = rna_tsss[sort_idx]
        sorted_names = rna_names[sort_idx]

        for pi, peak_mid in enumerate(atac_mids):
            # Use searchsorted to find genes within the interval [mid - thresh, mid + thresh]
            left = np.searchsorted(sorted_tss, peak_mid - distance_thresh)
            right = np.searchsorted(sorted_tss, peak_mid + distance_thresh)
            for gi in range(left, right):
                pairs.append((atac_names[pi], sorted_names[gi]))

    return pairs


# ============================================================================
# Causal GRN construction
# ============================================================================

def build_causal_grn(granger_results: pd.DataFrame) -> pd.DataFrame:
    """
    Extract causal peak-gene edges from the Granger test results as the base layer of the GRN.

    The returned causal edges will be expanded by the kinetics module into the
    full TF→peak→gene network.
    """
    causal_edges = granger_results[granger_results["is_causal"]].copy()
    logger.info(f"Building the GRN base layer from {len(causal_edges)} causal peak-gene edges")
    return causal_edges
