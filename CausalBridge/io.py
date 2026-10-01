"""
CausalBridge I/O module: configuration parsing, data loading, and result saving

Unified entry point for all input/output operations, ensuring that the entire pipeline uses consistent file formats.
"""

import yaml
import logging
import pandas as pd
import numpy as np
import anndata as ad
from typing import Tuple, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


def normalize_anndata_string_metadata(adata: ad.AnnData) -> ad.AnnData:
    """Make nullable pandas strings writable by older AnnData backends.

    Recent pandas versions use ``StringArray`` for ``dtype="string"``.  Some
    AnnData versions refuse to write that dtype when nullable-string writing is
    disabled (most visibly for ``obs/_index``).  Convert only those values,
    leaving the rest of the AnnData object and its analysis metadata intact.
    Missing values become empty strings, which are supported by the HDF5
    variable-length string writer used by older AnnData versions.
    """
    for axis_name in ("obs", "var"):
        frame = getattr(adata, axis_name)
        index = frame.index
        if isinstance(index.dtype, pd.StringDtype):
            values = index.to_numpy(dtype=object, na_value=None)
            values = ["" if value is None else str(value) for value in values]
            frame.index = pd.Index(values, dtype=object, name=index.name)

        for column in frame.columns:
            if isinstance(frame[column].dtype, pd.StringDtype):
                values = frame[column].array.to_numpy(dtype=object, na_value=None)
                values = ["" if value is None else str(value) for value in values]
                frame[column] = pd.Series(values, index=frame.index, dtype=object)

    return adata


# ============================================================================
# Configuration loading
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and supplement it with default values.

    The user only needs to provide key parameters (such as input paths); the remaining
    parameters use built-in defaults (kept in sync with config.yaml). This function
    first loads the user configuration and then merges it with the defaults (user values take precedence).

    Parameters
    ----------
    config_path : str
        Path to the user's YAML configuration file

    Returns
    -------
    config : dict
        Complete configuration dictionary containing all parameters
    """
    # Read the user configuration
    with open(config_path, "r") as f:
        user_config = yaml.safe_load(f)

    # Defaults (ensuring that all required fields are present)
    defaults = {
        "preprocess": {
            "rna": {"min_cells_per_gene": 50, "min_umi_per_cell": 500,
                    "n_highly_variable_genes": 3000},
            "atac": {"min_cells_per_peak": 10, "min_peaks_per_cell": 1000,
                     "binarize": True},
        },
        "pseudotime": {
            "method": "dpt", "n_neighbors": 30,
            "root_cell_marker": None, "n_bins": 100,
            "min_cells_per_bin": 10,
            "ablation_shuffle_pseudotime": False,
            "root_cell_method": "cytotrace",
            "cytotrace_n_top_genes": 200,
        },
        "granger": {
            "distance_thresh": 1_000_000,
            "significance_threshold": 0.10, "correction_method": "fdr_bh",
            "min_effect_size": 0.02,
            "soft_threshold": True, "min_composite_score": 0.3,
            "composite_weights": [0.4, 0.6],
            "ablation_lag0": False,
        },
        "atac_to_rna": {
            "min_observations": 5,
            "log_normalize_rna": True,
        },
        "rna_to_atac": {
            "motif_db": "jaspar2024", "motif_pval_threshold": 1e-4,
            "max_tfs_per_peak": 5,
            "tf_peak_pvalue_threshold": 0.05,  # TF→peak edge significance filter
            "time_lag": True,  # Adaptive lag: dynamically determine the number of steps from the cluster's transcriptional change rate
            "tf_expression_filter": True,  # TF expression filter: verify that gene names are present in the RNA matrix
            "max_lag": 0,  # 0=automatic: max(3, min(n_bins//4, 12))
        },
        "perturbation": {
            "ko_strength": 1.0,
            "subnetwork_depth": "unlimited",
            "min_lag_per_peak_filter": True,
            # Conservative traversal safety cap; when unset it is derived
            # from subnetwork_unlimited_max_edges by the unlimited extractor.
            "subnetwork_unlimited_max_work_edges": None,
            "cell_projection": {
                "enabled": False,
                "n_components": None,
                "n_neighbors": 10,
                "history_delta_threshold": 1e-8,
                "state_unit": "rna_log1p_cp10k",
            },
        },
        "clustering": {
            "enabled": True, "cluster_key": "leiden", "n_neighbors": 20,
            "n_pcs": 20, "resolution": 0.8,
            "random_state": 42, "min_cells_per_cluster": 50,
            "fallback_pseudotime_method": "palantir",
            "paga": {
                "enabled": False,
                "connectivity_threshold": 0.1,
                "min_cells_per_lineage": 50,
            },
        },
        "output": {
            "dir": "results/", "cache_dir": "cache/",
            "checkpoint_dir": None,
            "save_intermediate": True,
        },
    }

    # Recursive merge: user values override defaults
    config = _deep_merge(defaults, user_config)
    _validate_config(config)
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two dictionaries, with values in override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _validate_config(config: dict):
    """Check that required fields are present in the configuration and are reasonable."""
    required_inputs = ["rna_h5ad", "atac_h5ad"]
    for key in required_inputs:
        if key not in config.get("input", {}):
            raise ValueError(f"Configuration is missing required field: input.{key}")

    rna_path = Path(config["input"]["rna_h5ad"])
    atac_path = Path(config["input"]["atac_h5ad"])
    if not rna_path.exists():
        raise FileNotFoundError(f"RNA file does not exist: {rna_path}")
    if not atac_path.exists():
        raise FileNotFoundError(f"ATAC file does not exist: {atac_path}")

    # Check that parameter ranges are reasonable
    gr = config["granger"]
    if gr["significance_threshold"] <= 0 or gr["significance_threshold"] >= 1:
        raise ValueError("significance_threshold must lie in (0, 1)")
    if gr["distance_thresh"] < 0:
        raise ValueError("distance_thresh cannot be negative")

    logger.info("Configuration validation passed")


# ============================================================================
# Data loading
# ============================================================================

def load_data(config: dict) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Load the RNA and ATAC AnnData files and perform basic validation.

    Validation includes:
    1. Whether both files can be read successfully
    2. Whether cell barcodes match one-to-one
    3. Whether .var contains genomic coordinate information (chr, start, end)

    Returns
    -------
    rna_adata, atac_adata : AnnData
    """
    rna_path = config["input"]["rna_h5ad"]
    atac_path = config["input"]["atac_h5ad"]

    logger.info(f"Loading RNA data: {rna_path}")
    rna_adata = ad.read_h5ad(rna_path)

    logger.info(f"Loading ATAC data: {atac_path}")
    atac_adata = ad.read_h5ad(atac_path)

    # --- Validate genomic coordinate columns ---
    for label, adata in [("RNA", rna_adata), ("ATAC", atac_adata)]:
        for col in ["chr", "start", "end"]:
            if col not in adata.var.columns:
                raise KeyError(
                    f"{label} AnnData .var is missing the '{col}' column."
                    f"Please add genomic coordinate information. Current columns: {list(adata.var.columns)}"
                )

    # --- Validate cell matching ---
    rna_cells = set(rna_adata.obs_names)
    atac_cells = set(atac_adata.obs_names)
    common = rna_cells & atac_cells

    if len(common) == 0:
        raise ValueError(
            "RNA and ATAC data have no shared cell barcodes, "
            "please confirm that both files come from the same set of cells in a multi-omics assay."
        )

    if len(common) < len(rna_cells) or len(common) < len(atac_cells):
        logger.warning(
            f"Cell sets do not fully match: RNA={len(rna_cells)}, ATAC={len(atac_cells)}, "
            f"shared={len(common)}. Only shared cells will be retained."
        )
        rna_adata = rna_adata[list(common)].copy()
        atac_adata = atac_adata[list(common)].copy()

    # --- Ensure consistent cell order ---
    if not np.array_equal(rna_adata.obs_names, atac_adata.obs_names):
        atac_adata = atac_adata[rna_adata.obs_names].copy()

    logger.info(f"Data loading complete: {rna_adata.n_obs} cells, "
                f"{rna_adata.n_vars} genes, {atac_adata.n_vars} peaks")
    return rna_adata, atac_adata


def load_target_genes(path: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Load the knockout target gene list and gene-type annotations.

    Three formats are supported:
    1. Two-column format: each line is "gene_name  gene_type" (gene_type = TF or target)
    2. Single-column format (backward compatible): one gene symbol per line, with gene_type = "TF" by default
    3. Whole-genome mode: the file content is "ALL" or "ALL_EXPRESSED"

    Returns
    -------
    genes : list of str
    gene_types : dict {gene_name: gene_type}
        gene_type is "TF" or "target"
        Defaults to "TF" when a gene has no annotation
    """
    with open(path, "r") as f:
        raw = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

    if len(raw) == 0:
        raise ValueError(f"Target gene file is empty: {path}")

    if raw[0].upper().startswith("ALL"):
        logger.info("Using whole-genome mode (all expressed genes can serve as KO targets)")
        return ["__ALL__"], {}

    genes = []
    gene_types = {}

    for line in raw:
        parts = line.split()
        gene = parts[0]
        genes.append(gene)

        if len(parts) >= 2:
            gtype = parts[1].upper()
            if gtype in ("TF", "TARGET"):
                gene_types[gene] = gtype
            else:
                logger.warning(
                    f"Unknown gene_type '{parts[1]}' (expected TF or target); "
                    f"gene {gene} will fall back to 'TF'"
                )
                gene_types[gene] = "TF"
        else:
            gene_types[gene] = "TF"  # Default

    n_tf = sum(1 for t in gene_types.values() if t == "TF")
    n_target = sum(1 for t in gene_types.values() if t == "TARGET")
    logger.info(
        f"Loaded {len(genes)} target genes "
        f"(TF: {n_tf}, target: {n_target})"
    )
    return genes, gene_types


# ============================================================================
# Result saving
# ============================================================================

def save_results(results: dict, config: dict):
    """
    Save all output results to the specified directory.

    Results include:
    - perturbation_results.csv: core results table (transcriptomic + chromatin shifts for each KO, net effect per gene)
    - perturbation_pathways.csv: pathway-level edge records (independent TF→peak→gene contributions, including TF proportion attribution)
    - causal_peak_gene_edges.csv: causal peak→gene edges
    - tf_peak_weights.csv: TF→peak regulatory weights
    - Detailed results for each KO (such as network propagation paths)

    Parameters
    ----------
    results : dict
        Results dictionary produced by the pipeline modules
    config : dict
        Configuration dictionary (used to obtain output paths)
    """
    out_dir = Path(config["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Opt-in, bounded cell-level projection outputs.  The projection core's
    # full cell-by-gene states are intentionally never persisted here.
    if config.get("perturbation", {}).get("cell_projection", {}).get("enabled", False):
        for key, filename in (
            ("cell_projection", "perturbation_cell_projection.csv"),
            ("projection_summary", "perturbation_projection_summary.csv"),
            ("projection_branch_probabilities", "perturbation_projection_branch_probabilities.csv"),
            ("projection_history", "perturbation_projection_bin_history.csv"),
            ("projection_bin_metadata", "perturbation_projection_bin_metadata.csv"),
            ("projection_events", "perturbation_projection_events.csv"),
        ):
            frame = results.get(key)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frame.to_csv(out_dir / filename, index=False)

    # --- Causal regulatory edges ---
    if "causal_edges" in results:
        results["causal_edges"].to_csv(
            out_dir / "causal_peak_gene_edges.csv", index=False
        )
        logger.info(f"Causal edges saved: {len(results['causal_edges'])}")

    # --- TF→peak weights ---
    if "tf_peak_weights" in results:
        results["tf_peak_weights"].to_csv(
            out_dir / "tf_peak_weights.csv", index=False
        )

    # --- Perturbation results ---
    if "perturbation_results" in results:
        pert = results["perturbation_results"]
        if not pert.empty and "delta_rna" in pert.columns:
            pert = pert.sort_values(
                ["cluster", "delta_rna"],
                ascending=[True, False],
            )
        pert.to_csv(out_dir / "perturbation_results.csv", index=False)

    # --- Weighted-average perturbation results (merged across clusters) ---
    if "perturbation_merged" in results:
        pm = results["perturbation_merged"]
        if not pm.empty and "delta_rna" in pm.columns:
            pm = pm.sort_values("delta_rna", ascending=False)
        pm.to_csv(out_dir / "perturbation_merged.csv", index=False)
        logger.info(f"Weighted-average perturbation results saved: {len(pm)}")

    # --- Pathway-level edge records (TF→peak→gene, for cancellation analysis and regulatory-chain tracing) ---
    if "pathway_edges" in results:
        pw = results["pathway_edges"]
        if not pw.empty:
            pw.to_csv(out_dir / "perturbation_pathways.csv", index=False)
            logger.info(f"Pathway edges saved: {len(pw)}")

    # --- Aggregated TF→peak edges (including window-validation flip information) ---
    if "aggregated_tf_peak_edges" in results:
        ae = results["aggregated_tf_peak_edges"]
        if not ae.empty:
            ae.to_csv(out_dir / "aggregated_tf_peak_edges.csv", index=False)
            n_flipped = ae["flipped"].sum() if "flipped" in ae.columns else 0
            logger.info(f"Aggregated TF→peak edges saved: {len(ae)} (including {n_flipped} flipped)")


    # --- ATAC changes (BED format, viewable in IGV) ---
    if "atac_changes" in results:
        atac_df = results["atac_changes"]
        if not atac_df.empty:
            _save_atac_bed(atac_df, out_dir / "atac_changes.bed")
            # Significant ATAC changes: |z_score| >= threshold (z-score computed from the WT null distribution)
            if "is_significant" in atac_df.columns:
                sig_df = atac_df[atac_df["is_significant"]].copy()
                if not sig_df.empty:
                    _save_atac_bed(sig_df, out_dir / "atac_changes_significant.bed")
                    logger.info(
                        f"Significant ATAC changes (|z| threshold): {len(sig_df)} peaks → "
                        f"atac_changes_significant.bed"
                    )

    # --- ATAC changes (weighted average across clusters, aligned with perturbation_merged logic) ---
    if "atac_changes_merged" in results:
        acm = results["atac_changes_merged"]
        if acm is not None and not acm.empty:
            acm.to_csv(out_dir / "atac_changes_merged.csv", index=False)
            _save_atac_bed(acm, out_dir / "atac_changes_merged.bed")
            if "is_significant" in acm.columns:
                sig_m = acm[acm["is_significant"]].copy()
                if not sig_m.empty:
                    _save_atac_bed(sig_m, out_dir / "atac_changes_significant_merged.bed")
                    logger.info(
                        f"Weighted-average significant ATAC changes across clusters: {len(sig_m)} peaks → "
                        f"atac_changes_significant_merged.bed"
                    )
            logger.info(f"Weighted-average ATAC changes across clusters saved: {len(acm)} peaks")

    # --- Peak perturbation output (knockout_type=peak mode) ---
    if "perturbation_results" in results:
        pert_all = results["perturbation_results"]
        if not pert_all.empty and "target_gene" in pert_all.columns:
            peak_res = pert_all[
                pert_all["target_gene"].astype(str).str.startswith("PEAK:")
            ].copy()
            if not peak_res.empty:
                peak_res.to_csv(out_dir / "peak_perturbation_results.csv", index=False)
                logger.info(f"Peak perturbation results saved: {len(peak_res)} → peak_perturbation_results.csv")

    if "upstream_tfs" in results and results["upstream_tfs"] is not None \
            and not results["upstream_tfs"].empty:
        results["upstream_tfs"].to_csv(out_dir / "peak_upstream_tfs.csv", index=False)
        logger.info(
            f"Upstream TF attribution saved: {len(results['upstream_tfs'])} → peak_upstream_tfs.csv"
        )

    if "upstream_tfs_merged" in results and results["upstream_tfs_merged"] is not None \
            and not results["upstream_tfs_merged"].empty:
        results["upstream_tfs_merged"].to_csv(
            out_dir / "peak_upstream_tfs_merged.csv", index=False
        )
        logger.info(
            f"Upstream TF weighted merge across clusters saved: {len(results['upstream_tfs_merged'])} "
            f"→ peak_upstream_tfs_merged.csv"
        )

    if results.get("peak_summary") is not None and not results["peak_summary"].empty:
        _write_peak_report(results["peak_summary"], out_dir / "peak_perturbation_report.txt")

    # --- Intermediate files ---
    if config["output"]["save_intermediate"]:
        inter_dir = out_dir / "intermediate"
        inter_dir.mkdir(exist_ok=True)

        for key, df in results.items():
            if isinstance(df, pd.DataFrame) and not key.endswith("_saved"):
                df.to_csv(inter_dir / f"{key}.csv", index=False)

        # Save perturbation results for each cluster
        if "cluster_results" in results:
            for i, cl_res in enumerate(results["cluster_results"]):
                pert = cl_res.get("perturbation_results")
                if isinstance(pert, pd.DataFrame) and not pert.empty:
                    cluster_label = pert["cluster"].iloc[0] if "cluster" in pert.columns else f"cl_{i}"
                    pert.to_csv(inter_dir / f"perturbation_{cluster_label}.csv", index=False)

    logger.info(f"All results saved to: {out_dir.resolve()}")


def _save_atac_bed(atac_changes: pd.DataFrame, path: Path):
    """Output predicted ATAC changes in standard BED format, viewable in IGV.

    Parameters
    ----------
    atac_changes : DataFrame
        Columns must include chr, start, end, peak_id, delta_accessibility
    path : Path
        Output .bed file path
    """
    bed_cols = ["chr", "start", "end", "peak_id", "delta_accessibility"]
    if not all(c in atac_changes.columns for c in bed_cols[:3]):
        logger.warning("ATAC change data is missing BED columns (chr/start/end); skipping BED export")
        return

    with open(path, "w") as f:
        f.write('# track name="CausalBridge predicted ATAC changes"\n')
        f.write('# itemRgb="On"\n')
        for _, row in atac_changes.iterrows():
            score = int(np.clip(abs(row.get("delta_accessibility", 0)) * 1000, 0, 1000))
            f.write(f"{row['chr']}\t{int(row['start'])}\t{int(row['end'])}\t"
                    f"{row.get('peak_id', '.')}\t{score}\n")
    logger.info(f"ATAC BED track saved: {path}")


def _write_peak_report(peak_summary: pd.DataFrame, path: Path):
    """Generate a four-section human-readable peak perturbation report (A perturbed peak / B upstream TF / C direct effect / D propagation summary)."""

    def _i(v):
        return int(v) if v is not None else 0

    lines = [
        "=" * 64,
        "Peak Perturbation Report (CausalBridge v1.39, peak KO mode)",
        "=" * 64,
    ]
    for _, row in peak_summary.iterrows():
        lines.append("")
        lines.append(f"--- Perturbed peak: {row.get('peak_id', '')} "
                     f"(cluster {row.get('cluster', '')}) ---")
        lines.append(f"  A. baseline accessibility : {row.get('baseline_accessibility', float('nan')):.4f}")
        lines.append(f"     perturbed accessibility: {row.get('perturbed_accessibility', float('nan')):.4f}")
        lines.append(f"     delta accessibility    : {row.get('delta_accessibility', float('nan')):+.4f}")
        lines.append(f"  C. affected genes (direct): {_i(row.get('n_affected_genes'))}")
        lines.append(f"     peak→gene edges fired : {_i(row.get('n_affected_edges'))}")
        if _i(row.get('n_upstream_tfs')) > 0:
            lines.append(f"  B. upstream TF candidates : {_i(row.get('n_upstream_tfs'))} "
                         f"(see peak_upstream_tfs.csv)")
        else:
            lines.append(f"  B. upstream TF candidates : no confident upstream TF identified "
                         f"(an incomplete motif database, low TF expression, or failure to pass the Pearson test may each be responsible)")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Peak perturbation report saved: {path}")
