"""
CausalBridge Kinetics Modeling Module

Contains the two modeling directions at the core of the scheme:

1. ATAC→RNA transfer function fitting:
   For each causal peak-gene pair, a shared neural network (NN) fits a nonlinear
   transfer function:
     R_g(t) = f(A_peak(t-1))
   Pair specificity is captured through a learned embedding; the basis function
   is monotonically increasing in ATAC. When NN inference fails, the
   perturbation engine falls back linearly (gain × ΔA).

2. RNA→ATAC regulatory relationship learning:
   For each peak, learn how its accessibility is regulated by TF expression:
     A_peak(t) = g(TF₁(t), TF₂(t), ..., TF_k(t))
   The TF candidate set is determined by real JASPAR motif scans (not a
   hard-coded list). g() performs a single-feature Pearson test per TF and per
   lag on the candidate TFs selected by motif scanning; the weight is the
   regulatory strength.
"""

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import pearsonr
import logging
import pickle
import warnings
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from sysconfig import get_paths

logger = logging.getLogger(__name__)


def _get_cache_dir(config: Optional[dict] = None) -> Path:
    """Get the cache directory, reading from config first, otherwise using the default path."""
    if config:
        cache_dir = config.get("output", {}).get("cache_dir")
        if cache_dir:
            return Path(cache_dir)
    return Path.home() / ".CausalBridge"


# ============================================================================
# [-1,1] piecewise scaling: use root-cell ATAC as baseline to symmetrize the
# perturbation space
# ============================================================================

def _compute_root_atac(atac_adata, pseudotime_df,
                       quantile: float = 0.05,
                       min_count: int = 5,
                       floor: float = 0.05) -> np.ndarray:
    """Compute the baseline ATAC value of each peak at the start of pseudotime.

    Uses the mean ATAC of the earliest-quantile cells in pseudotime as the
    "ground state" reference point. The floor prevents noise amplification for
    low-baseline peaks (baselines below floor are raised to floor).

    Returns
    -------
    root_atac : np.ndarray, shape (n_peaks,), aligned with atac_adata.var_names
    """
    from scipy.sparse import issparse

    pseudo_vals = pseudotime_df["pseudotime"].values.astype(float)
    quantile_pct = quantile * 100.0
    root_thresh = np.percentile(pseudo_vals, quantile_pct)
    root_mask = pseudo_vals <= root_thresh

    if root_mask.sum() < min_count:
        fallback_pct = min(quantile_pct * 2.0, 50.0)
        root_thresh = np.percentile(pseudo_vals, fallback_pct)
        root_mask = pseudo_vals <= root_thresh

    X = atac_adata.X.toarray() if issparse(atac_adata.X) else np.array(atac_adata.X)
    # Binarize ATAC (consistent with downstream binned aggregation)
    X_bin = (X > 0).astype(np.float64)
    root_atac = X_bin[root_mask].mean(axis=0)
    root_atac = np.maximum(root_atac, floor)

    logger.info(
        f"Root-cell ATAC baseline: {root_mask.sum()} cells "
        f"(pseudotime <= {root_thresh:.4f}), "
        f"quantile={quantile}, min_count={min_count}, floor={floor}, "
        f"baseline ∈ [{root_atac.min():.4f}, {root_atac.max():.4f}]"
    )
    return root_atac


def _scale_atac(atac: np.ndarray, baseline: np.ndarray,
                floor: float = 0.05) -> np.ndarray:
    """Piecewise-scale ATAC values into the [-1, 1] range.

    baseline is the reference point for each peak (root-cell mean), floored so
    it never drops below floor.
    - atac > baseline: scaled upward into (0, 1]
    - atac <= baseline: scaled downward into [-1, 0]

    Parameters
    ----------
    atac : (n_bins, n_peaks) or (n_peaks,)
    baseline : (n_peaks,)
    """
    baseline_safe = np.maximum(np.atleast_1d(baseline.copy()), floor)
    if atac.ndim == 2 and baseline_safe.ndim == 1:
        baseline_safe = np.broadcast_to(baseline_safe.reshape(1, -1), atac.shape)

    result = np.zeros_like(atac, dtype=np.float64)
    up_mask = atac > baseline_safe

    # Upward: (atac - baseline) / (1 - baseline)
    denom_up = np.maximum(1.0 - baseline_safe, 1e-6)
    result[up_mask] = (atac[up_mask] - baseline_safe[up_mask]) / denom_up[up_mask]

    # Downward: (atac - baseline) / baseline
    result[~up_mask] = (atac[~up_mask] - baseline_safe[~up_mask]) / baseline_safe[~up_mask]

    return result


def _unscale_atac(atac_scaled: np.ndarray, baseline: np.ndarray,
                  floor: float = 0.05) -> np.ndarray:
    """Inverse scaling: [-1, 1] → [0, 1] original ATAC space.

    Parameters
    ----------
    atac_scaled : (n_bins, n_peaks) or (n_peaks,)
    baseline : (n_peaks,)
    """
    baseline_safe = np.maximum(np.atleast_1d(baseline.copy()), floor)
    if atac_scaled.ndim == 2 and baseline_safe.ndim == 1:
        baseline_safe = np.broadcast_to(baseline_safe.reshape(1, -1), atac_scaled.shape)

    result = np.zeros_like(atac_scaled, dtype=np.float64)
    up_mask = atac_scaled > 0

    # Forward recovery: scaled * (1 - baseline) + baseline
    result[up_mask] = atac_scaled[up_mask] * (1.0 - baseline_safe[up_mask]) + baseline_safe[up_mask]

    # Negative recovery: scaled * baseline + baseline
    result[~up_mask] = atac_scaled[~up_mask] * baseline_safe[~up_mask] + baseline_safe[~up_mask]

    return result


# ============================================================================
# Step 1: ATAC→RNA transfer function fitting
# ============================================================================

def fit_atac_to_rna(
    rna_adata,
    atac_adata,
    pseudotime_df,
    causal_edges: pd.DataFrame,
    config: dict,
    granger_results: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, dict, np.ndarray]:
    """
    For each causal peak-gene pair, fit an ATAC→RNA transfer function.

    Core equation:
      R_g(t) = f(A_peak(t))

    At steady state (dR_g/dt ≈ 0):
      R_g(t) = α(A_peak(t)) / β_g

    Method: shared neural network (NN) fitting of the nonlinear transfer
    function.

    Returns
    -------
    transfer_functions : DataFrame
    transfer_models : dict
        NN model bundle
    root_atac : np.ndarray
        shape (n_peaks,), per-peak baseline from root cells
    """
    cfg = config["atac_to_rna"]
    pcfg = config["pseudotime"]

    if cfg.get("method", "nn") != "nn":
        raise ValueError("Only NN ATAC-to-RNA transfer fitting is supported")

    # Adaptive binning based on cluster scale
    _smooth_kernel = pcfg.get("smooth_kernel", "hard")
    _overlap = pcfg.get("smooth_overlap_factor") if _smooth_kernel == "gaussian" else None
    bin_indices, n_bins = _adaptive_rebin(
        pseudotime_df, rna_adata.n_obs,
        max_bins=pcfg["n_bins"],
        min_cells_per_bin=pcfg.get("min_cells_per_bin", 5),
        target_bins=pcfg["n_bins"] if _smooth_kernel == "gaussian" else None,
        overlap_factor=_overlap,
    )
    pseudotime_df["bin"] = bin_indices
    logger.info(f"ATAC→RNA adaptive binning: {rna_adata.n_obs} cells → {n_bins} bins "
                f"(smooth={_smooth_kernel})")

    rna_binned = _bin_matrix(rna_adata.X, pseudotime_df, n_bins,
                             smooth=pcfg.get("smooth_kernel", "hard"),
                             min_cells_per_bin=pcfg.get("min_cells_per_bin", 5))
    atac_binned = _bin_matrix(atac_adata.X, pseudotime_df, n_bins,
                              smooth=pcfg.get("smooth_kernel", "hard"),
                              min_cells_per_bin=pcfg.get("min_cells_per_bin", 5))

    # --- RNA log-normalization: after the log1p transform, gain becomes log-fold-change ---
    if cfg.get("log_normalize_rna", True):
        rna_binned = np.log1p(rna_binned)
        logger.info(
            f"RNA log1p normalization: binned range "
            f"[{rna_binned.min():.4f}, {rna_binned.max():.4f}]"
        )

    # --- [-1,1] scaling: root-cell ATAC as baseline ---
    root_atac = _compute_root_atac(
        atac_adata, pseudotime_df,
        quantile=cfg.get("root_cell_quantile", 0.05),
        min_count=cfg.get("root_cell_min_count", 5),
        floor=cfg.get("root_atac_floor", 0.05),
    )
    atac_binned_scaled = _scale_atac(atac_binned, root_atac)
    logger.info(
        f"ATAC [-1,1] scaling: baseline ∈ [{root_atac.min():.4f}, {root_atac.max():.4f}], "
        f"scaled ∈ [{atac_binned_scaled.min():.3f}, {atac_binned_scaled.max():.3f}]"
    )

    # Get branch boundary information
    from .nn_transfer import _fit_atac_to_rna_nn
    result = _fit_atac_to_rna_nn(
        rna_adata, atac_adata, pseudotime_df, causal_edges, config,
        root_atac, atac_binned_scaled, rna_binned,
        granger_results=granger_results,
    )
    if result is None:
        return pd.DataFrame(), {"type": "nn", "data": {}}, root_atac
    transfer_functions, transfer_models = result
    return transfer_functions, transfer_models, root_atac


# ============================================================================
# Step 2: RNA→ATAC regulatory relationship learning (real motif scanning)
# ============================================================================

def fit_rna_to_atac(
    rna_adata,
    atac_adata,
    pseudotime_df,
    causal_edges: pd.DataFrame,
    config: dict,
    target_gene_types: Optional[dict] = None,
    root_atac: Optional[np.ndarray] = None,
    rna_raw = None,
) -> pd.DataFrame:
    """
    For each causal peak, learn how its accessibility is regulated by TF
    expression.

    Two-stage strategy:
    Stage 1: motif-scan the peak DNA sequences to determine the candidate TF
    set.
    Stage 2: single-feature Pearson test per TF and per lag —
    TF(t-lag) → A_peak(t); the first significant lag wins.

    Stage 1.5 (when target_gene_types is provided):
      For target genes annotated as TFs, check whether motif matches are
      sufficient. If not, supplement the matches via the best motif proxy from
      the same family.

    Parameters
    ----------
    target_gene_types : dict, optional
        {gene_name: "TF" | "target"}; gene type annotations from the IO module.
    root_atac : np.ndarray, optional
        shape (n_peaks,), per-peak baseline from root cells.
        If None, computed automatically from atac_adata.

    Returns
    -------
    tf_weights : DataFrame
        Columns: peak_id, tf_gene, weight, p_value, abs_contribution

    """
    cfg = config["rna_to_atac"]
    pcfg = config["pseudotime"]

    # Adaptive binning based on cluster scale
    _smooth_kernel = pcfg.get("smooth_kernel", "hard")
    _overlap = pcfg.get("smooth_overlap_factor") if _smooth_kernel == "gaussian" else None
    bin_indices, n_bins = _adaptive_rebin(
        pseudotime_df, rna_adata.n_obs,
        max_bins=pcfg["n_bins"],
        min_cells_per_bin=pcfg.get("min_cells_per_bin", 5),
        target_bins=pcfg["n_bins"] if _smooth_kernel == "gaussian" else None,
        overlap_factor=_overlap,
    )
    pseudotime_df["bin"] = bin_indices
    logger.info(f"RNA→ATAC adaptive binning: {rna_adata.n_obs} cells → {n_bins} bins "
                f"(smooth={_smooth_kernel})")

    # Pseudotime-binned aggregation
    rna_binned = _bin_matrix(rna_adata.X, pseudotime_df, n_bins,
                             smooth=pcfg.get("smooth_kernel", "hard"),
                             min_cells_per_bin=pcfg.get("min_cells_per_bin", 5))
    atac_binned = _bin_matrix(atac_adata.X, pseudotime_df, n_bins,
                              smooth=pcfg.get("smooth_kernel", "hard"),
                              min_cells_per_bin=pcfg.get("min_cells_per_bin", 5))

    # --- [-1,1] scaling: target variable Y = scaled ATAC ---
    if root_atac is None:
        root_atac = _compute_root_atac(
        atac_adata, pseudotime_df,
        quantile=cfg.get("root_cell_quantile", 0.05),
        min_count=cfg.get("root_cell_min_count", 5),
        floor=cfg.get("root_atac_floor", 0.05),
    )
    atac_binned_scaled = _scale_atac(atac_binned, root_atac)
    logger.info(
        f"RNA→ATAC [-1,1] scaling: Y ∈ [{atac_binned_scaled.min():.3f}, "
        f"{atac_binned_scaled.max():.3f}]"
    )

    # --- Stage 1: motif scanning ---
    logger.info("RNA→ATAC stage 1: motif-scanning peak sequences")
    cache_dir = _get_cache_dir(config)
    motif_db = _load_motif_database(cfg.get("motif_db", "jaspar2024"), cache_dir)

    if motif_db is not None:
        logger.info(f"  Loaded {len(motif_db)} motifs")
        peak_to_tfs = _scan_peaks_with_jaspar(
            atac_adata, rna_adata, causal_edges, motif_db, cfg, config
        )
    else:
        logger.warning(
            "  Motif database unavailable; falling back to TF-database-based matching."
        )
        peak_to_tfs = _assign_tfs_fallback(
            atac_adata, rna_adata, causal_edges, cache_dir,
            max_tfs_per_peak=cfg.get("max_tfs_per_peak", 5),
        )

    # --- Stage 1.5: family proxy motif supplementation ---
    if target_gene_types and motif_db is not None:
        _supplement_family_proxies(
            peak_to_tfs, motif_db, target_gene_types, cfg, config
        )

    # --- Stage 1.6: TF candidate validation ---
    #   Exclude TFs not present in the RNA gene set (fall back to the full gene
    #   set when a gene is missing from HVGs)
    if cfg.get("tf_expression_filter", True):
        rna_var_names = list(rna_adata.var_names)
        rna_var_names_full = list(rna_raw.var_names) if rna_raw is not None else None
        peak_to_tfs = _filter_tfs_by_expression(
            peak_to_tfs, rna_binned, rna_var_names,
            rna_var_names_full=rna_var_names_full,
        )

    # --- Stage 2: TF→peak regulatory learning ---
    max_lag_cfg = cfg.get("max_lag", 5)

    # Auto max_lag: "auto" / 0 / negative → n_bins // 4
    if isinstance(max_lag_cfg, str) and max_lag_cfg.lower() == "auto":
        max_lag_cfg = max(1, n_bins // 4)
        logger.info(f"max_lag auto-computed: {max_lag_cfg} (n_bins={n_bins}, n_bins//4)")
    elif isinstance(max_lag_cfg, (int, float)) and max_lag_cfg <= 0:
        max_lag_cfg = max(1, n_bins // 4)
        logger.info(f"max_lag auto-computed: {max_lag_cfg} (n_bins={n_bins}, n_bins//4)")

    time_lag = cfg.get("time_lag", True)
    branch_boundaries = pseudotime_df.attrs.get("branch_boundaries") if hasattr(pseudotime_df, "attrs") else None

    if time_lag:
        logger.info(
            f"RNA→ATAC stage 2: single-feature Pearson test per TF and per lag "
            f"on motif candidate TFs, TF(t-lag)→peak(t), lag=0..{max_lag_cfg}, "
            f"independent search per edge, first significant lag wins"
        )
    else:
        logger.info(
            f"RNA→ATAC stage 2: single-feature Pearson test for TF→peak weights "
            f"(no time lag, TF(t)→peak(t), max_lag=0)"
        )
    results = _fit_tf_peak_regression(
        peak_to_tfs, rna_adata, atac_adata, rna_binned, atac_binned_scaled,
        max_lag=max_lag_cfg if time_lag else 0,
        pvalue_threshold=cfg.get("tf_peak_pvalue_threshold", 0.05),
        branch_boundaries=branch_boundaries,
        min_abs_contribution=cfg.get("min_abs_contribution", 0.0),
    )

    logger.info(f"RNA→ATAC fitting complete: {len(results)} TF→peak regulatory relationships")
    return pd.DataFrame(results)


# ---- JASPAR motif scanning implementation ----

# ---- Motif database loading ----

def _load_motif_database(db_name: str = "jaspar2024", cache_dir: Optional[Path] = None) -> Optional[dict]:
    """
    Load the motif database. Prefer gimmemotifs (gimme.vertebrate + CIS-BP);
    fall back to JASPAR when unavailable.

    Returns
    -------
    motifs : dict or None
        {motif_id: {"name": str, "tf_name": str, "pwm": ndarray}}
    """
    # Strategy A: gimmemotifs data files
    motifs = _load_gimmemotifs_database(cache_dir)
    if motifs and len(motifs) > 100:
        return motifs

    # Strategy B: JASPAR online/cached
    return _load_jaspar_motifs(db_name, cache_dir)


def _load_gimmemotifs_database(cache_dir: Optional[Path] = None) -> Optional[dict]:
    """
    Load the motif database from the gimmemotifs installation directory.

    Loads gimme.vertebrate.v5.0 (curated vertebrate motifs) and supplements
    non-duplicate TFs from CIS-BP (large-scale computationally inferred motifs).

    Maps motif cluster IDs to specific TF names via the motif2factors mapping,
    keeping only one best motif per TF (curated evidence preferred).

    Returns
    -------
    motifs : dict or None
        Same format as _load_jaspar_motifs
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".CausalBridge"
    # Load from cache first
    cache_path = cache_dir / "gimmemotifs_cache.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                motifs = pickle.load(f)
            logger.info(f"Loaded {len(motifs)} motifs from the gimmemotifs cache")
            return motifs
        except Exception:
            pass

    # Use the active interpreter's site-packages directory rather than a
    # machine- or environment-name-specific Conda path.
    possible_dirs = [
        Path(get_paths()["purelib"]) / "data" / "motif_databases",
    ]

    data_dir = None
    for d in possible_dirs:
        if d.exists():
            data_dir = d
            break

    if data_dir is None:
        logger.debug("gimmemotifs data directory not found")
        return None

    try:
        motifs = {}

        # Priority: gimme.vertebrate.v5.0 (curated) > CIS-BP (broad)
        sources = [
            ("gimme.vertebrate.v5.0", "gimme.vertebrate.v5.0.pfm",
             "gimme.vertebrate.v5.0.motif2factors.txt"),
        ]

        for source_name, pfm_file, mapping_file in sources:
            pfm_path = data_dir / pfm_file
            mapping_path = data_dir / mapping_file
            if not pfm_path.exists() or not mapping_path.exists():
                continue

            n_loaded = _parse_gimmemotifs_pfm(pfm_path, mapping_path, motifs)
            logger.info(f"  Loaded {n_loaded} motifs from {source_name}")

        # Supplement with CIS-BP
        cisbp_pfm = data_dir / "CIS-BP.pfm"
        cisbp_map = data_dir / "CIS-BP.motif2factors.txt"
        if cisbp_pfm.exists() and cisbp_map.exists():
            n_cisbp = _parse_cisbp_pfm(cisbp_pfm, cisbp_map, motifs)
            logger.info(f"  Supplemented {n_cisbp} motifs from CIS-BP")

        if not motifs:
            return None

        logger.info(f"  gimmemotifs raw: {len(motifs)} motifs")

        # TF-centric deduplication: merge all_factors across motifs of the same
        # TF and keep only the best PWM
        motifs = _dedup_motifs_by_tf(motifs)
        logger.info(f"  gimmemotifs after dedup: {len(motifs)} motifs")

        # Save cache
        cache_path = cache_dir / "gimmemotifs_cache.pkl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(motifs, f)
        logger.info(f"  gimmemotifs motif dataset cached at {cache_path}")

        return motifs

    except Exception as e:
        logger.warning(f"Failed to load gimmemotifs database: {e}")
        return None


def _parse_gimmemotifs_pfm(
    pfm_path: Path, mapping_path: Path, existing: dict
) -> int:
    """Parse the gimme.vertebrate PFM file, merging into the existing dict."""
    from collections import defaultdict

    # Parse the motif2factors mapping: motif_id → [(factor, curated), ...]
    motif_to_factors = defaultdict(list)
    with open(mapping_path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                motif_id, factor, evidence, curated = parts[0], parts[1], parts[2], parts[3]
                motif_to_factors[motif_id].append({
                    "factor": factor.upper(),
                    "curated": curated == "Y",
                    "evidence": evidence,
                })

    # Parse the PFM file
    motifs = {}
    current_id = None
    current_pfm = None
    current_comment = None

    with open(pfm_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                current_comment = line[1:]  # strip #
                continue
            if line.startswith(">"):
                # Save the previous motif
                if current_id is not None and current_pfm is not None:
                    motifs[current_id] = {
                        "pfm": np.array(current_pfm, dtype=float),
                        "factors": motif_to_factors.get(current_id, []),
                        "comment": current_comment,
                    }
                current_id = line[1:]
                current_pfm = []
                current_comment = None
                continue
            # PFM rows: A C G T
            values = [float(x) for x in line.split("\t")]
            if len(values) == 4:
                if current_pfm is None:
                    current_pfm = []
                current_pfm.append(values)

    # Save the last one
    if current_id is not None and current_pfm is not None:
        motifs[current_id] = {
            "pfm": np.array(current_pfm, dtype=float),
            "factors": motif_to_factors.get(current_id, []),
            "comment": current_comment,
        }

    # Build motif-centric entries: keep one PWM per motif plus the full factor
    # list (including cofactors — factors without their own PWM that link to
    # the peak through this motif)
    n_added = 0
    for motif_id, info in motifs.items():
        pfm = info["pfm"]
        if pfm.shape[0] < 5:
            continue
        all_factors = info.get("factors", [])
        if not all_factors:
            continue

        pwm = _pfm_to_pwm(pfm.T, pseudocount=0.01)

        # Select the primary factor (curated preferred, otherwise the first)
        primary = all_factors[0]["factor"]
        for fi in all_factors:
            if fi.get("curated"):
                primary = fi["factor"]
                break

        key = f"GM_{motif_id}"
        if key in existing:
            # Merge factors (same-named motifs coming from multiple sources)
            seen = {f["factor"] for f in existing[key].get("all_factors", [])}
            for fi in all_factors:
                if fi["factor"] not in seen:
                    existing[key]["all_factors"].append(fi)
                    seen.add(fi["factor"])
        else:
            existing[key] = {
                "tf_name": primary,
                "pfm": pfm,
                "pwm": pwm,
                "source": "gimmemotifs",
                "all_factors": all_factors,
            }
            n_added += 1

    return n_added


def _parse_cisbp_pfm(pfm_path: Path, mapping_path: Path, existing: dict) -> int:
    """Parse the CIS-BP PFM file, supplementing TFs missing from the existing dict."""
    from collections import defaultdict

    # Parse motif2factors
    motif_to_factors = defaultdict(set)
    with open(mapping_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                motif_to_factors[parts[0]].add(parts[1].upper())

    # Parse the PFM
    current_id = None
    current_pfm = None
    cisbp_motifs = {}

    with open(pfm_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if current_id is not None and current_pfm is not None:
                    cisbp_motifs[current_id] = {
                        "pfm": np.array(current_pfm, dtype=float),
                        "factors": motif_to_factors.get(current_id, set()),
                    }
                current_id = line[1:]
                current_pfm = []
                continue
            values = [float(x) for x in line.split("\t")]
            if len(values) == 4:
                current_pfm.append(values)

    if current_id is not None and current_pfm is not None:
        cisbp_motifs[current_id] = {
            "pfm": np.array(current_pfm, dtype=float),
            "factors": motif_to_factors.get(current_id, set()),
        }

    # Build motif-centric entries: keep the full factor list
    n_added = 0
    for motif_id, info in cisbp_motifs.items():
        pfm = info["pfm"]
        if pfm.shape[0] < 5:
            continue
        all_factors_raw = info.get("factors", set())
        if not all_factors_raw:
            continue

        # Convert to the unified format
        all_factors = [
            {"factor": f.upper(), "curated": False, "evidence": "Direct"}
            for f in all_factors_raw
        ]
        pwm = _pfm_to_pwm(pfm.T, pseudocount=0.01)
        primary = all_factors[0]["factor"]

        key = f"CISBP_{motif_id}"
        if key in existing:
            seen = {f["factor"] for f in existing[key].get("all_factors", [])}
            for fi in all_factors:
                if fi["factor"] not in seen:
                    existing[key]["all_factors"].append(fi)
                    seen.add(fi["factor"])
        else:
            existing[key] = {
                "tf_name": primary,
                "pfm": pfm,
                "pwm": pwm,
                "source": "CIS-BP",
                "all_factors": all_factors,
            }
            n_added += 1

    return n_added


def _dedup_motifs_by_tf(motifs: dict) -> dict:
    """
    TF-centric deduplication: each TF independently picks its best motif
    (IC + curated bonus), while merging that TF's all_factors across all
    motifs.

    Follows the legacy tf_best logic:
    - Each TF (including cofactors) picks the best PWM among all associated
      motifs
    - Score = _compute_ic(PFM) + 10.0 (curated bonus)
    - Merge all_factors: collect every factor association of that TF across all
      motifs
    - Avoid key collisions: when two TFs share the same best motif,
      disambiguate with a suffix
    """
    from collections import defaultdict

    # Per TF: pick the best motif + collect all_factors
    tf_best = {}           # tf_name → (score, motif_id, info)
    tf_all_factors = defaultdict(dict)  # tf_name → {factor_name: factor_info}

    for motif_id, info in motifs.items():
        pfm = info.get("pfm")
        if pfm is None or pfm.shape[0] < 5:
            continue

        ic = _compute_ic(pfm)

        all_factors = info.get("all_factors", [])
        if not all_factors:
            continue

        for fi in all_factors:
            tf = fi["factor"]
            score = ic + (10.0 if fi.get("curated") else 0.0)

            # Pick the best motif
            if tf not in tf_best or score > tf_best[tf][0]:
                tf_best[tf] = (score, motif_id, info)

            # Collect every factor of this motif under this TF (including
            # cofactor associations)
            for fj in all_factors:
                fname = fj["factor"]
                if fname not in tf_all_factors[tf]:
                    tf_all_factors[tf][fname] = dict(fj)
                elif fj.get("curated") and not tf_all_factors[tf][fname].get("curated"):
                    tf_all_factors[tf][fname] = dict(fj)

    # Build the deduplicated dict
    deduped = {}
    for tf, (score, motif_id, info) in tf_best.items():
        merged_factors = list(tf_all_factors[tf].values())
        # Sort: the TF itself first, then curated, then the rest
        merged_factors.sort(key=lambda f: (
            f["factor"] != tf,
            not f.get("curated", False),
        ))

        new_info = {
            "tf_name": tf,
            "pfm": info.get("pfm"),
            "pwm": info["pwm"],
            "source": info.get("source", "unknown"),
            "all_factors": merged_factors,
        }

        # Avoid key collisions: when two TFs share the same best motif,
        # disambiguate with a suffix
        if motif_id in deduped:
            motif_id = f"{motif_id}__{tf}"
        deduped[motif_id] = new_info

    return deduped


def _compute_ic(pfm: np.ndarray) -> float:
    """Compute the Information Content (IC) of a PFM, used for motif quality ranking."""
    n = pfm.sum(axis=0, keepdims=True)
    ppm = (pfm + 0.01) / (n + 0.04)
    ic_per_pos = 2.0 + np.sum(ppm * np.log2(ppm + 1e-10), axis=0)
    return float(np.sum(ic_per_pos))


def _load_jaspar_motifs(db_name: str = "jaspar2024", cache_dir: Optional[Path] = None) -> Optional[dict]:
    """
    Load the JASPAR motif database.

    Prefer fetching online via biopython's JASPAR API;
    if unavailable, try loading from a local cache file;
    if still unavailable, return None (triggering the fallback strategy).

    Returns
    -------
    motifs : dict or None
        {motif_id: {"name": str, "tf_name": str, "pwm": ndarray}}
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".CausalBridge"

    # Strategy A: biopython online API
    try:
        from Bio.motifs import parse as parse_motifs
        import io, urllib.request

        # JASPAR 2024 vertebrate core set URL
        url = "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt"
        response = urllib.request.urlopen(url, timeout=30)
        content = response.read().decode("utf-8")

        motifs = {}
        for record in parse_motifs(io.StringIO(content), "jaspar"):
            motif_id = record.matrix_id
            pfm = np.array([record.counts[n] for n in "ACGT"], dtype=float)
            # PFM → PWM (Position Weight Matrix): log-odds after adding a pseudocount
            pwm = _pfm_to_pwm(pfm, pseudocount=0.01, background=[0.25, 0.25, 0.25, 0.25])
            motifs[motif_id] = {
                "name": record.name,
                "tf_name": record.name.split("(")[0].strip() if hasattr(record, "name") else motif_id,
                "pwm": pwm,
            }
        logger.info(f"Loaded {len(motifs)} motifs from JASPAR online")

        # Save local cache
        cache_path = cache_dir / "jaspar_motifs.pkl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(motifs, f)
        logger.info(f"JASPAR motifs cached at {cache_path}")

        return motifs

    except Exception as e:
        logger.debug(f"JASPAR online load failed: {e}")

    # Strategy B: local cache
    cache_path = cache_dir / "jaspar_motifs.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                motifs = pickle.load(f)
            logger.info(f"Loaded {len(motifs)} motifs from the local cache")
            return motifs
        except Exception:
            pass

    # Unavailable
    return None


def _pfm_to_pwm(
    pfm: np.ndarray,
    pseudocount: float = 0.01,
    background: list = None,
) -> np.ndarray:
    """
    Position Frequency Matrix → Position Weight Matrix.

    PWM[i, j] = log2( (PFM[i,j] + pc) / (N + 4*pc) / bg[i] )
    where N is the column sum, pc is the pseudocount, and bg[i] is the
    background frequency of base i.
    """
    if background is None:
        background = [0.25, 0.25, 0.25, 0.25]
    N = pfm.sum(axis=0, keepdims=True)
    ppm = (pfm + pseudocount) / (N + 4 * pseudocount)
    pwm = np.log2(ppm / np.array(background)[:, None])
    return pwm


def _scan_peaks_with_jaspar(
    atac_adata,
    rna_adata,
    causal_edges: pd.DataFrame,
    motif_db: dict,
    cfg: dict,
    config: dict,
) -> Dict[str, List[str]]:
    """
    Scan JASPAR motifs against the DNA sequence of each causal peak
    (vectorized implementation).

    Workflow:
    1. Check the cache; return directly if present
    2. Extract peak sequences from the genome FASTA
    3. For each motif, score all positions at once with sliding_window_view
    4. Matches above the threshold → record the corresponding TF
    5. Write the cache

    If the genome FASTA is unavailable, fall back to the fallback strategy.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    max_tfs = cfg.get("max_tfs_per_peak", 5)
    threshold = cfg.get("motif_pval_threshold", 1e-4)

    # --- Check cache ---
    cache_dir = _get_cache_dir(config)
    cache_path = cache_dir / "peak_tf_cache_v2.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached_raw = pickle.load(f)
            if isinstance(cached_raw, dict) and "threshold" in cached_raw:
                if cached_raw["threshold"] == threshold:
                    cached = cached_raw["data"]
                    logger.info(f"  Motif scan results loaded from cache ({len(cached)} peaks, "
                                f"threshold={threshold})")
                    return cached
                else:
                    logger.info(f"  Motif cache threshold mismatch (cached={cached_raw['threshold']}, "
                                f"current={threshold}); rescanning")
            else:
                logger.info(f"  Motif cache is stale (no threshold metadata); rescanning")
        except Exception:
            pass

    # --- Extract peak sequences ---
    peak_sequences = _extract_peak_sequences(atac_adata, config)
    if peak_sequences is None:
        logger.warning("  Cannot extract peak sequences (genome FASTA missing); falling back to the fallback strategy")
        return _assign_tfs_fallback(atac_adata, rna_adata, causal_edges, cache_dir,
                                    max_tfs_per_peak=cfg.get("max_tfs_per_peak", 5))

    # --- Precompute motif metadata ---
    nucleotide_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    motif_list = []  # [(motif_id, tf_name, pwm, motif_len, all_factors), ...]
    for motif_id, info in motif_db.items():
        pwm = info["pwm"]
        mlen = pwm.shape[1]
        if mlen < 5:
            continue
        motif_list.append(
            (motif_id, info["tf_name"], pwm, mlen,
             info.get("all_factors", []))
        )
    logger.info(f"  Preparing to scan {len(motif_list)} motifs (min_len=5)")

    # --- Scan peak by peak ---
    peak_to_tfs = {}
    unique_peaks = [p for p in causal_edges["peak_id"].unique() if p in peak_sequences]
    total = len(unique_peaks)

    for i, peak_id in enumerate(unique_peaks):
        seq = peak_sequences[peak_id].upper()
        # Pre-convert bases to an integer array (A=0, C=1, G=2, T=3, other=-1)
        seq_ints = np.array([nucleotide_to_idx.get(nt, -1) for nt in seq], dtype=np.int8)
        seq_len = len(seq_ints)
        matched = []

        for motif_id, tf_name, pwm, mlen, all_factors in motif_list:
            if mlen > seq_len:
                continue

            # Vectorized sliding window: score all positions at once
            windows = sliding_window_view(seq_ints, mlen)  # (n_pos, mlen), zero-copy view
            # Filter out windows containing invalid bases (N, etc.)
            valid_mask = (windows >= 0).all(axis=1)
            if not valid_mask.any():
                continue

            # Score only valid windows: pwm[windows, arange] → (n_valid, mlen) → sum
            valid_windows = windows[valid_mask]  # (n_valid, mlen)
            col_idx = np.arange(mlen)
            scores = np.sum(pwm[valid_windows[:, :], col_idx], axis=1)
            max_score = float(scores.max())

            # Empirical null distribution + Gumbel extreme-value approximation
            # For the max score over sliding windows, Gumbel is more accurate
            # than the normal distribution
            null_mean = float(np.mean(scores))
            null_std = float(np.std(scores))
            if null_std < 1e-10:
                continue
            gumbel_beta = null_std * np.sqrt(6.0) / np.pi
            gumbel_mu = null_mean - np.euler_gamma * gumbel_beta
            p_val = 1.0 - np.exp(-np.exp(-(max_score - gumbel_mu) / gumbel_beta))
            p_val = float(np.clip(p_val, 0.0, 1.0))
            if p_val < threshold:
                # Add all factors associated with this motif (including cofactors)
                # all_factors may contain tf_name itself; the seen set deduplicates
                for fi in all_factors:
                    factor_name = fi["factor"].strip().upper()
                    if factor_name:
                        matched.append((factor_name, p_val, max_score))
                # Add the primary TF (tf_name or split dimers)
                for single_tf in tf_name.split("::"):
                    tf_upper = single_tf.strip().upper()
                    if tf_upper:
                        matched.append((tf_upper, p_val, max_score))

        if matched:
            matched.sort(key=lambda x: x[1])
            seen = set()
            top_tfs = []
            for tf, pv, sc in matched:
                if tf not in seen:
                    seen.add(tf)
                    top_tfs.append(tf)
                    if len(top_tfs) >= max_tfs:
                        break
            peak_to_tfs[peak_id] = top_tfs

        if (i + 1) % 2000 == 0:
            logger.info(f"  Motif scan progress: {i+1}/{total} peaks")

    # --- Write cache (with threshold metadata) ---
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"threshold": threshold, "data": peak_to_tfs}, f)
    logger.info(f"  Motif scan results cached at {cache_path} (threshold={threshold})")

    n_with_tfs = sum(1 for tfs in peak_to_tfs.values() if tfs)
    logger.info(f"  Motif scan: {n_with_tfs}/{len(peak_to_tfs)} peaks matched at least one TF")
    return peak_to_tfs


# ============================================================================
# Family proxy motif mechanism
# ============================================================================


def _build_family_groups(motif_db: dict) -> Tuple[Dict[str, set], Dict[str, str]]:
    """
    Build TF family groupings from the motif database.

    Two-tier strategy:
    1. GM motif ID format: GM_GM.5.0.{FAMILY}.{number} → extract the family
       name directly
    2. Prefix inference: for TFs not covered by GM grouping, match to a known
       family or other same-prefix TFs via the gene-name prefix (root name with
       trailing digits removed)

    Returns
    -------
    family_groups : dict {family_name: {tf1, tf2, ...}}
    tf_to_family : dict {tf_name: family_name}
    """
    from collections import defaultdict
    import re

    family_groups = defaultdict(set)
    tf_to_family = {}

    # --- Tier 1: extract families from GM motif IDs ---
    for motif_id, info in motif_db.items():
        if motif_id.startswith("GM_GM.5.0."):
            parts = motif_id.split(".")
            if len(parts) >= 5:
                family = parts[3]
                # Add all associated factors (including cofactors)
                for fi in info.get("all_factors", []):
                    tf = fi["factor"]
                    if tf:
                        family_groups[family].add(tf)
                        if tf not in tf_to_family:
                            tf_to_family[tf] = family
                # Backward compatibility: legacy format without all_factors
                tf = info.get("tf_name", "")
                if tf:
                    family_groups[family].add(tf)
                    if tf not in tf_to_family:
                        tf_to_family[tf] = family

    # --- Tier 2: prefix inference, covering CIS-BP and TFs from other sources ---
    # Extract gene-name roots (strip trailing digits): RFX8 → RFX, FOXA1 → FOXA
    def _gene_root(name: str) -> str:
        m = re.match(r"^([A-Za-z]+?)\d*$", name)
        return m.group(1).upper() if m else name.upper()

    # Collect all known TFs and their roots
    all_tfs = set()
    for info in motif_db.values():
        tf = info.get("tf_name", "")
        if tf:
            all_tfs.add(tf)
        for fi in info.get("all_factors", []):
            fname = fi["factor"]
            if fname:
                all_tfs.add(fname)

    # Build the root → {tf1, tf2, ...} mapping
    root_groups = defaultdict(set)
    for tf in all_tfs:
        root_groups[_gene_root(tf)].add(tf)

    # For TFs not yet assigned to a family, try matching via the root
    #   Priority: match a known GM family name → match a meaningful root group
    for tf in all_tfs:
        if tf in tf_to_family:
            continue
        root = _gene_root(tf)

        # If the root happens to be a known GM family name (e.g. RFX, SOX, FOX, GATA)
        if root in family_groups:
            tf_to_family[tf] = root
            family_groups[root].add(tf)
            continue

        # If a member of the root group already belongs to a family, join the
        # same family
        root_members = root_groups.get(root, set())
        for member in root_members:
            if member in tf_to_family:
                tf_to_family[tf] = tf_to_family[member]
                family_groups[tf_to_family[member]].add(tf)
                break
        else:
            # If the root group has at least 2 members and none of them belong
            # to any family, create a new family named after the root
            unassigned = [m for m in root_members if m not in tf_to_family]
            if len(unassigned) >= 2:
                family_groups[root].update(unassigned)
                for m in unassigned:
                    tf_to_family[m] = root
            elif len(unassigned) == 1:
                # Only one member: assign it to the root family for later expansion
                family_groups[root].add(unassigned[0])
                tf_to_family[unassigned[0]] = root

    logger.debug(f"Family grouping: {len(family_groups)} families, {len(tf_to_family)} TFs")
    return dict(family_groups), tf_to_family


def _supplement_family_proxies(
    peak_to_tfs: dict,
    motif_db: dict,
    target_gene_types: dict,
    cfg: dict,
    config: dict,
) -> dict:
    """
    Target-TF family motif expansion: a two-stage strategy.

    **Stage A — passive borrowing (fast path)**:
    For each peak, if its top-5 motif matches already include a family member
    of a target TF, add that target TF directly to the peak's candidate list.
    No DNA rescanning is needed; it relies entirely on the existing cache.

    **Stage B — whole-family PWM scanning (forced)**:
    For every target TF with family information, scan each causal peak with
    the PWMs of **all** its family members. Any family-member match
    (p < threshold) → the target TF is added to that peak. PWMs of different
    family members differ subtly, so a full scan covers more ground than a
    single best PWM.

    **Core principle (v1.29)**:
    Only expand the target TF itself; never write all family members into all
    peaks. This avoids association explosion for large families such as
    C2H2_ZF (300+ members).

    Example:
      Target TF: RUNX1, family: {RUNX,RUNX1,RUNX2,RUNX3}
      Stage A: a peak has RUNX2 → add RUNX1 (passive, zero cost)
      Stage B: scan all peaks with RUNX3's PWM →
               peaks with p<0.05 also get RUNX1 (active scanning)

    Returns
    -------
    supplemented : dict {gene: {before, after, delta, proxy_tf, proxy_motif_id}}
        Change in match counts for the target-gene TFs that benefited
    """
    from numpy.lib.stride_tricks import sliding_window_view

    # --- Collect the target TF set ---
    target_tfs = set()
    for gene, gtype in target_gene_types.items():
        if gtype == "TF":
            target_tfs.add(gene.upper())

    if not target_tfs:
        logger.debug("  No target TFs; skipping family expansion")
        return {}

    # --- Build family groupings ---
    family_groups, tf_to_family = _build_family_groups(motif_db)
    if not family_groups:
        logger.debug("  No family grouping information; skipping family expansion")
        return {}

    # --- Find the family each target TF belongs to ---
    target_family_map = {}
    for tf in target_tfs:
        family = tf_to_family.get(tf)
        if family:
            target_family_map[tf] = family

    if not target_family_map:
        logger.debug("  No target TF has family information; skipping family expansion")
        return {}

    # --- Stage A: passive borrowing (based on the existing cache) ---
    tf_match_before = {}
    for tf_list in peak_to_tfs.values():
        for tf in tf_list:
            tf_upper = tf.upper()
            if tf_upper in target_tfs:
                tf_match_before[tf_upper] = tf_match_before.get(tf_upper, 0) + 1

    n_passive_expanded = 0
    n_passive_added = 0
    for peak_id, tf_list in peak_to_tfs.items():
        existing_upper = {tf.upper() for tf in tf_list}
        tfs_to_add = []
        for target_tf, family in target_family_map.items():
            if target_tf in existing_upper:
                continue
            family_members = family_groups.get(family, set())
            if any(tf.upper() in family_members for tf in tf_list):
                tfs_to_add.append(target_tf)

        if tfs_to_add:
            n_passive_expanded += 1
            new_tfs = list(tf_list)
            for tf in tfs_to_add:
                new_tfs.append(tf)
                n_passive_added += 1
            peak_to_tfs[peak_id] = new_tfs

    # Match counts after Stage A
    tf_match_after_passive = {}
    for tf_list in peak_to_tfs.values():
        for tf in tf_list:
            tf_upper = tf.upper()
            if tf_upper in target_tfs:
                tf_match_after_passive[tf_upper] = tf_match_after_passive.get(tf_upper, 0) + 1

    if n_passive_expanded > 0:
        logger.info(
            f"  Stage A (passive borrowing): {n_passive_expanded}/{len(peak_to_tfs)} peaks expanded, "
            f"{n_passive_added} TF-peak associations added in total"
        )

    # --- Stage B: active PWM scanning (whole-family motifs, forced for all target TFs) ---
    threshold = cfg.get("motif_pval_threshold", 1e-4)
    nucleotide_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}

    # Every target TF with family information participates in active scanning
    scan_targets = {}
    for tf_upper in sorted(target_tfs):
        if tf_upper in tf_to_family:
            current = tf_match_after_passive.get(tf_upper, 0)
            scan_targets[tf_upper] = current

    if scan_targets:
        logger.info(
            f"  Stage B (whole-family PWM scanning): launching scans for {len(scan_targets)} target TFs..."
        )

        # Extract peak sequences (extract once, shared by all target TFs)
        peak_sequences = _extract_peak_sequences_from_causal(
            peak_to_tfs.keys(), config
        )

        if peak_sequences:
            for tf_upper, current_matches in scan_targets.items():
                family = tf_to_family[tf_upper]
                family_members = family_groups.get(family, set())

                # Collect valid whole-family PWMs (excluding the target TF itself)
                family_pwms = []
                for mid, info in motif_db.items():
                    member_tf = info.get("tf_name", "")
                    if member_tf == tf_upper:
                        continue  # skip itself
                    if member_tf.upper() not in family_members and member_tf not in family_members:
                        continue
                    pwm = np.array(info.get("pwm", []))
                    if pwm.ndim != 2 or pwm.shape[1] < 5:
                        continue
                    family_pwms.append((mid, member_tf, pwm))

                if not family_pwms:
                    logger.debug(
                        f"    {tf_upper}: no usable family-member PWM in family {family}"
                    )
                    continue

                logger.info(
                    f"    {tf_upper} (currently {current_matches}): "
                    f"← scanning with {len(family_pwms)} PWMs from family {family}"
                )

                # For each peak, check whether it is matched by any family
                # member's PWM
                n_new = 0
                for peak_id, seq in peak_sequences.items():
                    seq_upper = seq.upper()
                    seq_ints = np.array(
                        [nucleotide_to_idx.get(nt, -1) for nt in seq_upper],
                        dtype=np.int8,
                    )

                    # Check whether the target TF is already present
                    existing_tfs = peak_to_tfs.get(peak_id, [])
                    existing_upper = {t.upper() for t in existing_tfs}
                    if tf_upper in existing_upper:
                        continue

                    # Scan this peak with the whole-family PWMs; any match adds it
                    matched = False
                    for mid, member_tf, pwm in family_pwms:
                        mlen = pwm.shape[1]
                        if len(seq_ints) < mlen:
                            continue

                        windows = sliding_window_view(seq_ints, mlen)
                        valid_mask = (windows >= 0).all(axis=1)
                        if not valid_mask.any():
                            continue

                        valid_windows = windows[valid_mask]
                        col_idx = np.arange(mlen)
                        scores = np.sum(pwm[valid_windows, col_idx], axis=1)
                        max_score = float(scores.max())

                        # Gumbel p-value
                        null_mean = float(np.mean(scores))
                        null_std = float(np.std(scores))
                        if null_std < 1e-10:
                            continue
                        gumbel_beta = null_std * np.sqrt(6.0) / np.pi
                        gumbel_mu = null_mean - np.euler_gamma * gumbel_beta
                        p_val = 1.0 - np.exp(-np.exp(
                            -(max_score - gumbel_mu) / gumbel_beta
                        ))
                        p_val = float(np.clip(p_val, 0.0, 1.0))

                        if p_val < threshold:
                            matched = True
                            break  # stop on any family-member match

                    if matched:
                        peak_to_tfs[peak_id] = existing_tfs + [tf_upper]
                        n_new += 1

                logger.info(
                    f"    {tf_upper}: whole-family scan added {n_new} new peak matches "
                    f"({current_matches} → {current_matches + n_new})"
                )
        else:
            logger.warning("  Cannot extract peak sequences; Stage B skipped")
    else:
        logger.info("  Stage B (whole-family PWM scanning): no target TFs to scan (no family information)")

    # --- Tally final match counts ---
    tf_match_final = {}
    for tf_list in peak_to_tfs.values():
        for tf in tf_list:
            tf_upper = tf.upper()
            if tf_upper in target_tfs:
                tf_match_final[tf_upper] = tf_match_final.get(tf_upper, 0) + 1

    # --- Report ---
    supplemented = {}
    for tf_upper in sorted(target_tfs):
        before = tf_match_before.get(tf_upper, 0)
        after = tf_match_final.get(tf_upper, 0)
        if after > before:
            supplemented[tf_upper] = {
                "before": before,
                "after": after,
                "delta": after - before,
            }
            logger.info(
                f"  {tf_upper}: family expansion {before} → {after} peak matches "
                f"(+{after - before})"
            )
        else:
            logger.debug(
                f"  {tf_upper}: family expansion unchanged ({before} peak matches)"
            )

    return supplemented


def _extract_peak_sequences_from_causal(
    peak_ids,
    config: dict,
) -> Optional[Dict[str, str]]:
    """
    Extract DNA sequences for causal peaks only (used for family proxy motif
    rescanning).

    Unlike _extract_peak_sequences, this function processes only the given set
    of peak IDs.
    """
    from Bio import SeqIO

    genome_path = config.get("input", {}).get("genome_fasta")
    if not genome_path or not Path(genome_path).exists():
        return None

    try:
        genome = SeqIO.to_dict(SeqIO.parse(genome_path, "fasta"))
    except Exception as e:
        logger.warning(f"Failed to load genome FASTA: {e}")
        return None

    sequences = {}
    peak_ids_list = list(peak_ids) if not isinstance(peak_ids, list) else peak_ids

    for peak_id in peak_ids_list:
        try:
            if ":" in peak_id:
                chrom, coords = peak_id.split(":")
                start, end = map(int, coords.split("-"))
            else:
                # BED format: chr1-100000-200000
                parts = peak_id.split("-")
                chrom = parts[0]
                start, end = int(parts[-2]), int(parts[-1])
            if chrom not in genome:
                continue
            seq = str(genome[chrom].seq[start:end])
            sequences[peak_id] = seq
        except (ValueError, KeyError, IndexError):
            continue

    return sequences


def _extract_peak_sequences(
    atac_adata, config: dict
) -> Optional[Dict[str, str]]:
    """
    Extract peak DNA sequences from the genome FASTA.

    Requires the user to provide a reference genome FASTA file path
    (config["input"]["genome_fasta"]).

    Returns
    -------
    sequences : dict {peak_id: dna_sequence} or None
    """
    genome_path = config.get("input", {}).get("genome_fasta")
    if not genome_path or not Path(genome_path).exists():
        return None

    try:
        from Bio import SeqIO

        # Build a chromosome → sequence index (load the needed chromosomes only)
        logger.info(f"  Loading genome: {genome_path}")
        genome = SeqIO.to_dict(SeqIO.parse(genome_path, "fasta"))

        sequences = {}
        for peak_id in atac_adata.var_names:
            info = atac_adata.var.loc[peak_id]
            chrom = info["chr"]
            # Try several chromosome naming formats
            for chrom_fmt in [chrom, f"chr{chrom}", chrom.replace("chr", "")]:
                if chrom_fmt in genome:
                    try:
                        seq = str(genome[chrom_fmt].seq[
                            int(info["start"]) : int(info["end"])
                        ])
                        sequences[peak_id] = seq
                        break
                    except Exception:
                        continue

        if len(sequences) == 0:
            logger.warning(
                "  Cannot match chromosome names. Please ensure the chromosome names "
                "in the genome FASTA match those in AnnData .var['chr']"
            )
            return None

        logger.info(f"  Successfully extracted sequences for {len(sequences)} peaks")
        return sequences

    except ImportError:
        logger.warning("  biopython unavailable; cannot extract sequences. pip install biopython")
        return None
    except Exception as e:
        logger.warning(f"  Sequence extraction failed: {e}")
        return None


def _assign_tfs_fallback(
    atac_adata, rna_adata, causal_edges: pd.DataFrame,
    cache_dir: Optional[Path] = None,
    max_tfs_per_peak: int = 5,
) -> Dict[str, List[str]]:
    """
    Fallback strategy: when motif scanning is unavailable, match via a TF
    database.

    Finds the genes linked to each peak from causal_edges and keeps the ones
    that are TFs. This relies on a sizable TF database file rather than a
    hard-coded list.

    Improvements (compared to the hard-coded list of the initial version):
    - Load the TF database from an external file (e.g. AnimalTFDB)
    - If the file is absent, automatically infer a candidate TF set from genes
      in the RNA data that have motif structural domains
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".CausalBridge"
    tf_db_path = cache_dir / "tf_database.txt"

    if tf_db_path.exists():
        with open(tf_db_path) as f:
            tf_set = {line.strip().split("\t")[0] for line in f if line.strip()}
    else:
        # Auto-build: use the pre-compiled TF list (far larger than the initial 40)
        tf_set = _get_extended_tf_list()
        # Write cache
        tf_db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tf_db_path, "w") as f:
            for tf in sorted(tf_set):
                f.write(f"{tf}\n")
        logger.info(f"  TF database cached: {tf_db_path} ({len(tf_set)} TFs)")

    peak_to_tfs = {}
    for peak_id in causal_edges["peak_id"].unique():
        linked_genes = set(causal_edges[causal_edges["peak_id"] == peak_id]["gene"])
        tfs = [g for g in linked_genes if g in tf_set]
        if tfs:
            # Truncate to max_tfs_per_peak, keeping the most likely relevant TFs
            peak_to_tfs[peak_id] = tfs[:max_tfs_per_peak]

    return peak_to_tfs


def _get_extended_tf_list() -> set:
    """
    Extended TF database (~800 human/mouse TFs).

    Source: subset of the AnimalTFDB v4.0 human TF list plus manually added key
    factors. The full list should be loaded from a file; this function serves as
    the built-in fallback.
    """
    return {
        # Hematopoietic TF families
        "GATA1", "GATA2", "GATA3", "GATA4", "GATA5", "GATA6",
        "KLF1", "KLF2", "KLF3", "KLF4", "KLF5", "KLF6", "KLF7", "KLF8",
        "KLF9", "KLF10", "KLF11", "KLF12", "KLF13", "KLF14", "KLF15",
        "RUNX1", "RUNX2", "RUNX3",
        "SPI1", "SPIB", "SPIC",
        "CEBPA", "CEBPB", "CEBPD", "CEBPE", "CEBPG", "CEBPZ",
        "TCF3", "TCF4", "TCF7", "TCF7L1", "TCF7L2", "LEF1",
        "MYC", "MYCN", "MYCL", "MYB", "MYBL1", "MYBL2",
        "FOS", "FOSB", "FOSL1", "FOSL2",
        "JUN", "JUNB", "JUND",
        "STAT1", "STAT2", "STAT3", "STAT4", "STAT5A", "STAT5B", "STAT6",
        "NFKB1", "NFKB2", "RELA", "RELB", "REL",
        "IRF1", "IRF2", "IRF3", "IRF4", "IRF5", "IRF6", "IRF7", "IRF8", "IRF9",
        "PAX5", "PAX2", "PAX6", "PAX8",
        "EBF1", "EBF2", "EBF3", "EBF4",
        "BCL11A", "BCL11B",
        "GFI1", "GFI1B",
        "TAL1", "TAL2", "LYL1",
        "LMO2", "LMO1",
        "ERG", "FLI1", "ETV2", "ETV6", "ETS1", "ETS2", "ELK1", "ELK4",
        "FOXP3", "FOXP1", "FOXP2", "FOXP4",
        "TBX21", "EOMES", "TBX1", "TBX2", "TBX5",
        "SOX2", "SOX4", "SOX6", "SOX9", "SOX10", "SOX17",
        "POU5F1", "POU2F1", "POU3F2",
        "NANOG",
        "ESRRB", "ESRRA", "ESRRG",
        "ZFP42",
        "PRDM14", "PRDM1", "PRDM2", "PRDM4",
        "TFAP2A", "TFAP2C",
        # Developmental TF families
        "HOXA1", "HOXA5", "HOXA9", "HOXA10", "HOXA11", "HOXA13",
        "HOXB4", "HOXB5", "HOXB7", "HOXC4", "HOXD4", "HOXD13",
        "MEIS1", "MEIS2", "PBX1", "PBX2", "PBX3",
        "CDX1", "CDX2", "CDX4",
        "FOXA1", "FOXA2", "FOXA3",
        "HNF1A", "HNF1B", "HNF4A", "HNF4G",
        "LHX1", "LHX2", "LHX3", "ISL1",
        "HAND1", "HAND2", "TWIST1", "TWIST2",
        "SNAI1", "SNAI2", "ZEB1", "ZEB2",
        "MITF", "TFEB", "TFE3", "USF1", "USF2",
        "MAX", "MNT", "MXD1", "MXI1",
        # Nuclear receptor family
        "AR", "ESR1", "ESR2", "PGR", "NR3C1", "NR3C2",
        "PPARA", "PPARD", "PPARG", "RARA", "RARB", "RARG",
        "VDR", "THRA", "THRB", "NR1H3", "NR1H4",
        # Chromatin remodelers & transcriptional cofactors
        "CTCF", "CTCFL", "YY1", "YAP1", "TAZ",
        "RBPJ", "CSL",
        "REST", "RCOR1", "RCOR2",
        "SUZ12", "EZH2", "EED", "RING1", "BMI1",
        "BRD2", "BRD3", "BRD4", "BRDT",
        "EP300", "CREBBP",
        "MED1", "MED12", "MED15",
        # p53 and related
        "TP53", "TP63", "TP73",
        "RB1", "RBL1", "RBL2",
        "E2F1", "E2F2", "E2F3", "E2F4", "E2F5", "E2F6",
        # Other important TFs
        "SP1", "SP2", "SP3", "SP4",
        "NFYA", "NFYB", "NFYC",
        "ATF1", "ATF2", "ATF3", "ATF4", "ATF6",
        "CREB1", "CREB3", "CREB5",
        "XBP1",
        "NRF1", "NRF2",  # NFE2L1, NFE2L2
        "BACH1", "BACH2",
        "MAF", "MAFA", "MAFB", "MAFF", "MAFG", "MAFK", "NFE2",
        "SMAD1", "SMAD2", "SMAD3", "SMAD4", "SMAD5", "SMAD7",
        "NOTCH1", "NOTCH2", "NOTCH3",
        "HEY1", "HEY2", "HES1", "HES5", "HES6",
        "GLI1", "GLI2", "GLI3", "ZIC1", "ZIC2",
    }


# ---- TF expression trend filter ----

def _filter_tfs_by_expression(
    peak_to_tfs: Dict[str, List[str]],
    rna_binned: np.ndarray,
    rna_var_names: List[str],
    rna_var_names_full: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Filter out TF candidates not present in the RNA gene set.

    Gene-name validation only: when a gene is not found among HVGs, fall back
    to the full gene set; if it is absent there too, it is excluded.

    (Median=0 filtering and Spearman decreasing-trend filtering were deprecated
    and removed.)

    Parameters
    ----------
    peak_to_tfs : {peak_id: [tf_names]}
    rna_binned : (n_bins, n_genes) binned gene expression (kept for caller
        signature compatibility)
    rna_var_names : gene name list
    rna_var_names_full : full gene-name list, used for fallback lookup


    Returns
    -------
    filtered peak_to_tfs (modified in place + reference returned)
    """
    rna_name_to_idx = {n: i for i, n in enumerate(rna_var_names)}
    # Full gene-set index (fallback, for finding motif TFs not in the HVGs)
    rna_name_to_idx_full = {n.upper(): i for i, n in enumerate(rna_var_names_full)} if rna_var_names_full else {}

    # Collect all candidate TFs
    all_tfs = set()
    for tfs in peak_to_tfs.values():
        all_tfs.update(tfs)

    excluded_not_in_rna = []       # gene name not in the RNA data
    kept = []

    for tf_name in sorted(all_tfs):
        tf_idx = rna_name_to_idx.get(tf_name)
        if tf_idx is None:
            # Not found in HVGs → try the full gene set
            if rna_name_to_idx_full:
                if tf_name.upper() not in rna_name_to_idx_full:
                    excluded_not_in_rna.append(tf_name)
                    continue
                # Present in the full gene set; keep it
                kept.append(tf_name)
                continue
            else:
                excluded_not_in_rna.append(tf_name)
                continue

        kept.append(tf_name)

    # Remove TFs not in the gene set from peak_to_tfs
    excluded_set = set(excluded_not_in_rna)
    n_removed_total = 0
    for peak_id in peak_to_tfs:
        old = peak_to_tfs[peak_id]
        new = [t for t in old if t not in excluded_set]
        n_removed_total += len(old) - len(new)
        peak_to_tfs[peak_id] = new

    logger.info(
        f"TF candidate filtering: {len(kept)} kept, "
        f"{len(excluded_not_in_rna)} (not in the gene set), "
        f"removed {n_removed_total} peak-TF candidate pairs in total"
    )

    return peak_to_tfs


# ---- Stage 2: per-lag TF→peak Pearson test ----

def _fit_tf_peak_regression(
    peak_to_tfs: Dict[str, List[str]],
    rna_adata,
    atac_adata,
    rna_binned: np.ndarray,
    atac_binned: np.ndarray,
    max_lag: int = 5,
    pvalue_threshold: float = 0.05,
    branch_boundaries=None,
    min_abs_contribution: float = 0.0,
) -> List[dict]:
    """
    For the candidate TFs selected by motif scanning, run a single-feature
    Pearson test per TF and per lag: TF(t-lag) → A_peak(t).

    Fit each lag 0..max_lag independently (each lag has only one column of X);
    the first significant lag wins → output that (TF, peak, lag).
    No collinearity between lags; TFs with high autocorrelation naturally take
    lag=0, while genuinely delayed-regulating TFs get their true lag.
    With max_lag=0 this is equivalent to TF(t)→peak(t).
    """
    rna_var_names = list(rna_adata.var_names)
    atac_var_names = list(atac_adata.var_names)
    n_bins = len(rna_binned)

    results = []
    rna_upper_map = {g.upper(): g for g in rna_var_names}

    for peak_id, candidate_tfs in peak_to_tfs.items():
        if not candidate_tfs:
            continue

        candidate_tfs = list(dict.fromkeys(candidate_tfs))

        try:
            peak_idx = atac_var_names.index(peak_id)
        except ValueError:
            continue

        # Case-insensitive matching
        valid_tfs = []
        tf_indices = []
        for tf in candidate_tfs:
            actual_name = rna_upper_map.get(tf.upper())
            if actual_name is not None:
                valid_tfs.append(actual_name)
                tf_indices.append(rna_var_names.index(actual_name))
        if len(valid_tfs) == 0:
            continue

        # ================================================================
        # Build single-feature input per TF independently + independent
        # Pearson test per lag
        #
        # Fit each candidate TF independently (one TF, one model),
        # test each lag 0..max_lag independently, and take the first
        # significant lag.
        #   Each lag's regression has only 1 column of X → no collinearity
        #   between lags → TFs with high autocorrelation naturally take
        #   lag=0, while delayed-regulating TFs take their true lag.
        #   ("reverse time direction": starting from peak(t), find the first
        #   significant TF(t-lag) encountered)
        #
        # per_lag mode: Y is aligned per lag inside the lag loop (all lags
        # share the same Y = Peak[max_lag:], same row counts, comparable
        # statistical power).
        # ================================================================
        # Verify lag=0 is at least feasible (maximum row count)
        if n_bins < 3:
            continue

        for tf_name, tf_idx_global in zip(valid_tfs, tf_indices):
            _found_sig = False  # early-stop flag
            _branch_Y_cache = None  # reuse the shared Y

            for k in range(0, max_lag + 1):
                if _found_sig:
                    break

                # ---- Single-lag independent X construction (all lags share Y, same row count) ----
                _shared_rows = n_bins - max_lag  # unified row count across lags
                if _shared_rows < 3:
                    continue

                if branch_boundaries is not None and len(branch_boundaries) > 1:
                    # Precompute the shared Y (identical for all lags)
                    if k == 0:
                        _Y_shared_parts = []
                        _X_info_parts = []  # [(start, br_usable_shared), ...]
                        for start, end in branch_boundaries:
                            br_len = end - start
                            br_usable_shared = br_len - max_lag
                            if br_usable_shared < 3:
                                continue
                            _Y_shared_parts.append(
                                atac_binned[start + max_lag : end, peak_idx]
                            )
                            _X_info_parts.append((start, br_usable_shared))
                        if not _Y_shared_parts:
                            continue
                        _Y = np.concatenate(_Y_shared_parts)
                        # Save for reuse by subsequent lags
                        _branch_Y_cache = _Y
                        _branch_X_info_cache = _X_info_parts
                    else:
                        if _branch_Y_cache is None:
                            continue
                        _Y = _branch_Y_cache
                        _X_info_parts = _branch_X_info_cache

                    X_parts = []
                    for start, br_usable_shared in _X_info_parts:
                        # TF[t-k] for t in [start+max_lag, end)
                        # = TF[start + max_lag - k : end - k]
                        X_parts.append(
                            rna_binned[start + max_lag - k : end - k,
                                       tf_idx_global]
                        )
                    col = np.concatenate(X_parts)
                else:
                    # Non-branch: shared Y = Peak[max_lag:], X = TF offset by lag
                    if k == 0:
                        _Y = atac_binned[max_lag:, peak_idx]
                        _branch_Y_cache = _Y  # reuse for subsequent lags
                    else:
                        _Y = _branch_Y_cache
                    # TF[t-k] for t in [max_lag, n_bins)
                    col = rna_binned[max_lag - k : n_bins - k, tf_idx_global]

                if np.std(col) < 1e-6:
                    continue
                X = col.reshape(-1, 1)

                # Ensure X and Y have the same number of rows
                if X.shape[0] != len(_Y):
                    min_rows = min(X.shape[0], len(_Y))
                    if min_rows < 3:
                        continue
                    X = X[:min_rows]
                    _Y = _Y[:min_rows]

                # ============================================================
                # Single-feature Pearson test per lag
                # ============================================================
                # Each lag's X has only 1 column, so pearsonr + t test directly
                # yields r / p / weight / contribution: for a single feature
                # w = r·σ(Y)/σ(X), and weight = r·σ(Y)/σ(X), so p values of
                # different lags are directly comparable.
                _px = X[:, 0]
                _py = np.asarray(_Y).ravel()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    _r, _p = pearsonr(_px, _py)
                if not np.isnan(_r) and _p <= pvalue_threshold:
                    _x_std = float(np.std(_px))
                    _y_std = float(np.std(_py))
                    if _x_std > 1e-10:
                        _w = float(_r * _y_std / _x_std)
                        _contrib = float(np.abs(_w) * _x_std)
                        if _contrib >= min_abs_contribution:
                            result = {
                                "peak_id": peak_id,
                                "tf_gene": tf_name,
                                "weight": _w,
                                "abs_contribution": _contrib,
                                "p_value": float(_p),
                                "tf_lag": k,
                            }
                            results.append(result)
                            _found_sig = True

    return results


# ============================================================================
# Utility functions
# ============================================================================

def _adaptive_rebin(pseudotime_df, n_cells: int, max_bins: int = 100,
                    min_cells_per_bin: int = 5, target_bins: int = None,
                    overlap_factor: float = None):
    """
    Adaptively assign pseudotime bins based on cell counts.

    When pseudotime_df.attrs["branch_boundaries"] exists and there are multiple
    branches, qcut is applied independently within each branch, assigning
    globally unique bin IDs.

    No target_bins: adapt via n_cells // min_cells_per_bin, capped at max_bins
    and floored at 5.
    target_bins with overlap_factor: Gaussian smoothing mode.
      n_bins = min(target_bins, max(5, floor(k * n_cells / min_cells_per_bin)))
      where k = overlap_factor, constraining sigma ≤ k * bin_width so that
      adjacent bins do not overlap excessively.
    target_bins without overlap_factor: use target_bins directly (legacy
    behavior).
    """
    if target_bins is not None:
        if overlap_factor is not None:
            cells_per_bin_loose = n_cells / max(min_cells_per_bin, 1)
            n_bins = max(5, min(target_bins, int(overlap_factor * cells_per_bin_loose)))
        else:
            n_bins = max(5, target_bins)
    else:
        n_bins = max(5, min(int(n_cells // min_cells_per_bin), max_bins))

    # --- branch-aware: per-branch qcut ---
    # Prefer detecting the branch column (produced by infer_pseudotime), falling
    # back to attrs
    branch_labels = pseudotime_df.get("branch") if hasattr(pseudotime_df, "get") else None
    if branch_labels is None:
        branch_boundaries_attr = pseudotime_df.attrs.get("branch_boundaries") if hasattr(pseudotime_df, "attrs") else None
        if branch_boundaries_attr is not None and len(branch_boundaries_attr) > 1:
            branch_labels = _infer_branch_from_bins(pseudotime_df, branch_boundaries_attr)

    if branch_labels is not None and branch_labels.nunique() > 1:

        # Allocate bins proportionally to cell counts
        unique_branches = np.unique(branch_labels)
        branch_cell_counts = np.array([
            (branch_labels == br).sum() for br in unique_branches
        ])
        raw_allocation = np.maximum(
            (branch_cell_counts / branch_cell_counts.sum() * n_bins).astype(int), 3
        )
        delta = n_bins - raw_allocation.sum()
        while delta > 0:
            raw_allocation[np.argmax(branch_cell_counts - raw_allocation * 3)] += 1
            delta -= 1
        while delta < 0:
            idx = np.argmax(raw_allocation - 3)
            if raw_allocation[idx] <= 3:
                break
            raw_allocation[idx] -= 1
            delta += 1

        # Independent qcut per branch
        global_offset = 0
        new_boundaries = []
        bins = np.full(len(pseudotime_df), -1, dtype=int)

        for br, n_br_bins in zip(unique_branches, raw_allocation):
            br_mask = (branch_labels == br).values
            br_pt = pseudotime_df["pseudotime"].values[br_mask].astype(float)
            try:
                br_bins = pd.qcut(br_pt, n_br_bins, labels=False, duplicates="drop")
            except (ValueError, TypeError):
                br_bins = np.zeros(br_mask.sum(), dtype=int)
            n_br_actual = int(br_bins.max()) + 1
            bins[br_mask] = br_bins + global_offset
            new_boundaries.append((global_offset, global_offset + n_br_actual))
            global_offset += n_br_actual

        pseudotime_df.attrs["branch_boundaries"] = new_boundaries
        n_actual = global_offset
        return bins, n_actual

    # --- global qcut (existing behavior) ---
    pseudo_vals = pseudotime_df["pseudotime"].values.astype(float)
    try:
        bins = pd.qcut(pseudo_vals, n_bins, labels=False, duplicates="drop")
    except (ValueError, TypeError):
        bins = np.zeros(len(pseudo_vals), dtype=int)
    n_actual = int(bins.max()) + 1
    return bins, n_actual


def _infer_branch_from_bins(pseudotime_df, branch_boundaries):
    """Recover each cell's branch label from branch_boundaries."""
    bin_col = pseudotime_df["bin"].values
    branch_labels = np.full(len(pseudotime_df), -1, dtype=int)
    for br_idx, (start, end) in enumerate(branch_boundaries):
        mask = (bin_col >= start) & (bin_col < end)
        branch_labels[mask] = br_idx
    return pd.Series(branch_labels, index=pseudotime_df.index)


def _branch_aware_lag_pairs(mat_before, mat_after, lag,
                            branch_boundaries=None):
    """Perform lag slicing independently within each branch, then concatenate
    the results.

    Without branch information this is equivalent to mat_before[:-lag],
    mat_after[lag:]. With branches, branches whose bin count is <= lag are
    skipped. When lag=0 the original matrices are returned directly (no
    slicing).
    """
    if lag == 0:
        return mat_before, mat_after

    if branch_boundaries is None or len(branch_boundaries) <= 1:
        return mat_before[:-lag, :], mat_after[lag:, :]

    parts_b, parts_a = [], []
    for start, end in branch_boundaries:
        if end - start > lag:
            parts_b.append(mat_before[start:end - lag, :])
            parts_a.append(mat_after[start + lag:end, :])

    if not parts_b:
        n_feat = mat_before.shape[1]
        return np.zeros((0, n_feat)), np.zeros((0, n_feat))

    return np.vstack(parts_b), np.vstack(parts_a)


def _bin_matrix(mat, pseudotime_df, n_bins: int,
                smooth: str = "hard", min_cells_per_bin: int = 15) -> np.ndarray:
    """Aggregate a sparse or dense matrix over pseudotime bins.

    smooth="hard": hard binning (original behavior)
    smooth="gaussian": Gaussian-kernel weighted smoothing; min_cells_per_bin
    controls the adaptive sigma
    """
    X = mat.toarray() if issparse(mat) else np.array(mat)

    if smooth == "hard":
        binned = np.zeros((n_bins, X.shape[1]))
        for b in range(n_bins):
            mask = pseudotime_df["bin"].values == b
            if mask.sum() > 0:
                binned[b] = X[mask].mean(axis=0)
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

    if avg_cells_per_bin >= min_cells_per_bin:
        sigma = base_sigma
    else:
        sigma = base_sigma * (min_cells_per_bin / max(avg_cells_per_bin, 1))

    bin_width = pseudo_span / n_bins
    bin_centers = np.linspace(pseudo_min + bin_width / 2, pseudo_max - bin_width / 2, n_bins)

    diff = bin_centers[:, None] - pseudo_vals[None, :]
    W = np.exp(-0.5 * (diff / sigma) ** 2)
    W /= W.sum(axis=1, keepdims=True) + 1e-10

    return W @ X
