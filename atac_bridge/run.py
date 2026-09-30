"""
CausalBridge main pipeline runner

Chains all modules together into a complete analysis workflow. This is the
entry point users invoke for the "one-click" analysis.

Pipeline overview:
  Step 1: Load configuration + data
  Step 2: Preprocessing (QC, normalization, highly variable genes)
  Steps 3-8: Model per cell cluster + perturbation simulation (pseudotime is
  inferred independently within each cluster)
  Step 9: Aggregate results + generate report

Usage:
  from atac_bridge.run import run_pipeline
  results = run_pipeline("config.yaml")
"""

import logging
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from .io import (
    load_config, load_data, load_target_genes, save_results,
    normalize_anndata_string_metadata,
)
from .preprocess import (
    preprocess_rna, preprocess_atac, match_cells, cluster_cells,
    save_cluster_top_genes, build_paga_lineages,
)
from .granger import infer_pseudotime, granger_test, build_causal_grn
from .kinetics import fit_atac_to_rna, fit_rna_to_atac
from .perturbation import propagate_perturbation, perturb_peak
from .reference_projection import (
    fit_reference_projection, project_reference, reconstruct_counterfactual,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("atac_bridge")


def _aggregate_context_l0(context_registry, modeled_context_l0, gene_names,
                          sparse_results=None, min_delta_threshold=0.0,
                          values_are_log_l0=False):
    """Aggregate true per-cell L0 means across every post-QC context.

    ``context_registry`` contains every context, including skipped/failed
    contexts.  ``modeled_context_l0`` only contains successful reconstructions;
    missing contexts deliberately use their original means for both sides.
    """
    if not context_registry:
        return pd.DataFrame()
    genes = list(gene_names)
    original = {}
    for c, v in context_registry.items():
        if "original_log_l0_mean" in v:
            original[c] = np.asarray(v["original_log_l0_mean"], dtype=float)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                original[c] = np.log(np.asarray(v["original_l0_mean"], dtype=float))
    modeled = modeled_context_l0
    sparse_meta = (sparse_results.drop_duplicates(["target_gene", "affected_gene"])
                   if sparse_results is not None and not sparse_results.empty else pd.DataFrame())
    rows = []
    threshold = max(float(min_delta_threshold), 1e-6)
    total_contexts = len(context_registry)
    total_cells = sum(int(v["n_cells"]) for v in context_registry.values())
    for target, target_contexts in modeled.items():
        numerator = np.full(len(genes), -np.inf, dtype=float)
        modeled_contexts = 0
        modeled_cells = 0
        for context, info in context_registry.items():
            n = int(info["n_cells"])
            base = original[context]
            pert = target_contexts.get(context)
            if pert is None:
                numerator = np.logaddexp(numerator, base + np.log(n))
            else:
                if values_are_log_l0:
                    pert_log = np.asarray(pert, dtype=float)
                else:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        pert_log = np.log(np.asarray(pert, dtype=float))
                numerator = np.logaddexp(numerator, pert_log + np.log(n))
                modeled_contexts += 1
                modeled_cells += n
        # The loop starts from -inf so context contributions are pooled in log
        # space; denominator is pooled identically from original means.
        denominator = np.full(len(genes), -np.inf, dtype=float)
        for context, info in context_registry.items():
            denominator = np.logaddexp(
                denominator, original[context] + np.log(int(info["n_cells"]))
            )
        unchanged_contexts = total_contexts - modeled_contexts
        unchanged_cells = total_cells - modeled_cells
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = (numerator - denominator) / np.log(2.0)
        raw[np.isneginf(numerator) & np.isneginf(denominator)] = 0.0
        display = np.clip(np.nan_to_num(raw, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
        for i, gene in enumerate(genes):
            if gene in str(target).split(",") or abs(display[i]) < threshold:
                continue
            meta = (sparse_meta[(sparse_meta.target_gene == target) &
                                (sparse_meta.affected_gene == gene)]
                    if not sparse_meta.empty else pd.DataFrame())
            clipped = bool(not np.isfinite(raw[i]) or raw[i] != display[i])
            row = {"target_gene": target, "affected_gene": gene,
                   "delta_rna": abs(float(display[i])),
                   "delta_rna_signed": float(display[i]),
                   "n_clusters_observed": modeled_contexts,
                   "aggregation_method": "pooled_cell_l0_log2fc",
                   "aggregation_total_contexts": total_contexts,
                   "aggregation_modeled_contexts": modeled_contexts,
                   "aggregation_unchanged_contexts": unchanged_contexts,
                   "aggregation_total_cells": total_cells,
                   "aggregation_modeled_cells": modeled_cells,
                   "aggregation_unchanged_cells": unchanged_cells,
                   "l0_log2fc_unclipped": float(raw[i]),
                   "display_clipped": clipped}
            # Retain legacy columns, but do not pretend cell-L0 aggregation
            # has an L1 censoring count.
            for col in ["mediated_by_atac", "mechanism", "propagation_depth",
                        "converged", "is_fallback", "perturbation_mode"]:
                row[col] = (meta.iloc[0][col] if not meta.empty and col in meta
                            else False if col in ("mediated_by_atac", "is_fallback")
                            else True if col == "converged"
                            else "bin_forward" if col == "perturbation_mode" else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def _validate_step5_checkpoint(ckpt: dict) -> None:
    """Reject checkpoints produced by the removed non-NN Step-5 path."""
    transfer_models = ckpt.get("transfer_models")
    if transfer_models is None or transfer_models.get("type") != "nn":
        raise RuntimeError(
            "step5 checkpoint is not NN-only compatible; delete it and rerun step 5."
        )


def _project_bin_forward_inputs(inputs: dict, config: dict, context: str, target: str) -> tuple:
    """Project one modeled context without sharing references across contexts."""
    try:
        if inputs.get("invalid_state", False):
            raise ValueError(f"invalid canonical bin state: {inputs.get('invalid_diagnostic')}")
        if inputs.get("effect_schema_version") == "bin_local_effect_v1":
            if inputs.get("state_unit") != "rna_log1p_cp10k":
                raise ValueError("unsupported canonical projection state_unit")
            if inputs.get("native_effect_unit") != "delta_log1p_rna_log1p_cp10k":
                raise ValueError("unsupported native projection effect unit")
        features = inputs["features"]
        source = pd.DataFrame(inputs["control_cell_states"], columns=features)
        n_cells = len(source)
        for key in ("cell_ids", "branch", "pseudotime", "bins"):
            if len(inputs[key]) != n_cells:
                raise ValueError(f"unaligned projection metadata: {key} length != cell states")
        weights = np.asarray(inputs["cell_bin_weights"], dtype=float)
        if weights.shape[0] != n_cells or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("invalid cell/bin membership weights")
        model = fit_reference_projection(
            source, inputs["branch"], inputs["pseudotime"],
            feature_names=features, cell_ids=inputs["cell_ids"],
            n_components=config.get("perturbation", {}).get("cell_projection", {}).get("n_components"),
            n_neighbors=config.get("perturbation", {}).get("cell_projection", {}).get("n_neighbors", 10),
            state_unit=inputs["state_unit"],
        )
        if inputs.get("effect_schema_version") == "bin_local_effect_v1":
            direct = np.asarray(inputs["direct_bin_multiplier"], dtype=float)
            native = np.asarray(inputs["native_bin_effect"], dtype=float)
            if direct.shape != native.shape or weights.shape[1] != direct.shape[0]:
                raise ValueError("unaligned bin-local projection effects")
            cell_direct = weights @ direct
            cell_native = weights @ native
            if inputs.get("log_accumulation", True):
                reconstructed = pd.DataFrame(
                    np.expm1(np.log1p(source.to_numpy() * cell_direct) + cell_native),
                    columns=features,
                )
            else:
                reconstructed = pd.DataFrame(
                    source.to_numpy() * cell_direct + cell_native, columns=features
                )
            reconstructed.attrs["state_unit"] = inputs["state_unit"]
        else:
            reconstructed = reconstruct_counterfactual(
                source, pd.DataFrame(inputs["control_bin_states"], columns=features),
                pd.DataFrame(inputs["perturbed_bin_states"], columns=features),
                inputs["cell_bin_weights"], feature_names=features,
                state_unit=inputs["state_unit"],
            )
        reconstructed_values = reconstructed.to_numpy(dtype=float)
        floor_mask = np.isfinite(reconstructed_values) & (reconstructed_values < 0)
        floor_count = int(floor_mask.sum())
        if floor_count:
            reconstructed_values[floor_mask] = 0.0
            reconstructed = pd.DataFrame(reconstructed_values, columns=features)
        floor_fraction = floor_count / max(reconstructed_values.size, 1)
        cell_floor_count = floor_mask.sum(axis=1).astype(int)
        cell_floor_fraction = cell_floor_count / max(reconstructed_values.shape[1], 1)
        table = project_reference(
            model, source, reconstructed, feature_names=features,
            cell_ids=inputs["cell_ids"], source_control_ids=inputs["cell_ids"],
            state_unit=inputs["state_unit"],
        )
        table["source_context"] = context
        table["target_gene"] = target
        table["source_bin"] = inputs["bins"]
        table["source_pseudotime"] = inputs["pseudotime"]
        table["source_branch"] = inputs["branch"]
        table["state_unit"] = inputs["state_unit"]
        table["native_effect_unit"] = inputs.get("native_effect_unit", inputs["state_unit"])
        # Keep aggregate values, but label them as context-level diagnostics;
        # the cell-level columns below are the values suitable for row-wise use.
        table["context_l1_floor_applied_count"] = floor_count
        table["context_l1_floor_applied_fraction"] = floor_fraction
        table["context_l1_floor_status"] = "clipped" if floor_count else "none"
        table["cell_l1_floor_applied_count"] = cell_floor_count
        table["cell_l1_floor_applied_fraction"] = cell_floor_fraction
        table["cell_l1_floor_status"] = np.where(cell_floor_count > 0, "clipped", "none")
        threshold = config.get("perturbation", {}).get("cell_projection", {}).get(
            "history_delta_threshold", 1e-8
        )
        base = np.asarray(inputs["control_bin_states"], dtype=float)
        if inputs.get("effect_schema_version") == "bin_local_effect_v1":
            direct_bin = base * np.asarray(inputs["direct_bin_multiplier"], dtype=float)
            native_bin = np.asarray(inputs["native_bin_effect"], dtype=float)
            history = (np.expm1(np.log1p(direct_bin) + native_bin)
                       if inputs.get("log_accumulation", True)
                       else direct_bin + native_bin)
        else:
            history = np.asarray(inputs["state_history"], dtype=float)
        branches = np.asarray(inputs["branch"], dtype=object)
        bins = np.asarray(inputs["bins"], dtype=int)
        pt = np.asarray(inputs["pseudotime"], dtype=float)
        history_rows = []
        metadata_rows = []
        local_lookup = {}
        for branch in pd.unique(branches):
            for local, global_bin in enumerate(sorted(set(bins[branches == branch]))):
                local_lookup[(branch, global_bin)] = local
        for b in range(history.shape[0]):
            members = bins == b
            branch = pd.Series(branches[members]).mode().iloc[0] if members.any() else "unknown"
            coordinate = float(pt[members].mean()) if members.any() else np.nan
            metadata_rows.append({
                "source_context": context, "target_gene": target,
                "branch": branch, "global_bin": int(b),
                "local_bin": int(local_lookup.get((branch, b), -1)),
                "pseudotime": coordinate, "state_unit": inputs["state_unit"],
                "native_effect_unit": inputs.get("native_effect_unit", inputs["state_unit"]),
                "metadata_fingerprint": model.fingerprint,
                "bin_state_mode": inputs.get("bin_state_mode", "shared_operator"),
                "effect_schema_version": "bin_local_effect_v1",
            })
            for g, delta in enumerate(history[b] - base[b]):
                if np.isfinite(delta) and abs(delta) >= threshold:
                    direct_value = float(base[b, g] * np.asarray(inputs.get("direct_bin_multiplier", np.ones_like(base)), dtype=float)[b, g])
                    native_value = float(np.asarray(inputs.get("native_bin_effect", np.zeros_like(base)), dtype=float)[b, g])
                    history_rows.append({
                        "source_context": context, "target_gene": target,
                        "branch": branch, "bin": int(b), "global_bin": int(b),
                        "local_bin": int(local_lookup.get((branch, b), -1)),
                        "pseudotime": coordinate, "gene": features[g],
                        "state_unit": inputs["state_unit"], "state_value": float(history[b, g]),
                        "delta_state": float(delta), "metadata_fingerprint": model.fingerprint,
                        "direct_multiplier": float(np.asarray(inputs.get("direct_bin_multiplier", np.ones_like(base)), dtype=float)[b, g]),
                        "aggregated_native_delta": native_value,
                        "downstream_canonical_delta": float(history[b, g] - direct_value),
                        "total_canonical_delta": float(history[b, g] - base[b, g]),
                        "history_format": "sparse_nonzero_delta_v1",
                        "effect_schema_version": inputs.get("effect_schema_version", "legacy"),
                        "native_effect_unit": inputs.get("native_effect_unit", inputs["state_unit"]),
                    })
        events = pd.DataFrame(inputs.get("local_events", []))
        if not events.empty:
            events["event_key"] = events.apply(
                lambda r: f"{context}|{target}|b{int(r['bin'])}|{r['peak']}|{r['gene']}", axis=1
            )
            events["native_effect_unit"] = inputs.get("native_effect_unit", inputs["state_unit"])
            events["global_bin"] = events["bin"].astype(int)
            events["branch"] = events["bin"].map(
                {int(r["global_bin"]): r["branch"] for _, r in pd.DataFrame(metadata_rows).iterrows()}
            )
            events["local_bin"] = events["bin"].map(
                {int(r["global_bin"]): r["local_bin"] for _, r in pd.DataFrame(metadata_rows).iterrows()}
            )
            events["source_context"] = context
            events["target_gene"] = target
            events["metadata_fingerprint"] = model.fingerprint
        return table, model.fingerprint, pd.DataFrame(history_rows), pd.DataFrame(metadata_rows), events
    except (ValueError, KeyError) as exc:
        logger.warning("Cell projection skipped for %s/%s: %s", context, target, exc)
        return pd.DataFrame(), None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def _validate_step6_checkpoint(tf_weights) -> None:
    """Reject Step-6 checkpoints containing removed bagging statistics."""
    legacy_columns = {"weight_std", "bagging_stability"}.intersection(tf_weights.columns)
    if legacy_columns:
        raise RuntimeError(
            "step6 checkpoint contains legacy bagging-derived columns "
            f"({', '.join(sorted(legacy_columns))}); delete it and rerun Step 6."
        )


def _log_paths(config: dict):
    """Print all key paths at the start of the run to avoid cache/checkpoint confusion."""
    inp = config.get("input", {})
    out = config.get("output", {})
    ckpt_dir = _checkpoint_dir(config)

    lines = [
        ("=" * 60),
        ("Path overview"),
        ("=" * 60),
        ("[Input]"),
        (f"  rna_h5ad:         {inp.get('rna_h5ad', '?')}"),
        (f"  atac_h5ad:        {inp.get('atac_h5ad', '?')}"),
        (f"  genome_fasta:     {inp.get('genome_fasta', '?')}"),
        (f"  target_genes:     {inp.get('target_genes', '?')}"),
        ("[Output]"),
        (f"  output_dir:       {out.get('dir', '?')}"),
        (f"  cache_dir:        {out.get('cache_dir', '?')}"),
        (f"  checkpoint_dir:   {ckpt_dir}"),
    ]
    for line in lines:
        logger.info(line)


def _checkpoint_dir(config: dict, suffix: str = None) -> Path:
    """Checkpoint save directory. ``suffix`` is used to store per-cluster subdirectories.

    Prefers output.checkpoint_dir; falls back to output.dir/checkpoints when unset.
    """
    custom = config.get("output", {}).get("checkpoint_dir")
    if custom:
        base = Path(custom)
    else:
        base = Path(config["output"]["dir"]) / "checkpoints"
    return base / suffix if suffix else base


def _load_checkpoint(config: dict, name: str, suffix: str = None):
    """Load a checkpoint; return None if it does not exist."""
    path = _checkpoint_dir(config, suffix) / f"{name}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            logger.info(f"Resumed from checkpoint: {name} ({path})")
            return data
        except Exception as e:
            logger.warning(f"Failed to load checkpoint ({name}): {e}")
    return None


def _save_checkpoint(config: dict, name: str, data: dict, suffix: str = None):
    """Save a checkpoint. ``suffix`` stores per-cluster subdirectories."""
    ckpt_dir = _checkpoint_dir(config, suffix)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Checkpoint saved: {name} ({path})")


# ============================================================================
# Modeling sub-pipeline (Steps 3-8), reusable for global or per-cluster mode
# ============================================================================

def _run_modeling_for_subset(
    rna_adata,
    atac_adata,
    target_genes: list,
    gene_types: dict,
    config: dict,
    suffix: str = None,
    rna_raw = None,
) -> dict:
    """
    Run the modeling steps (Steps 3-8) on a subset of cells.

    Design intent: cluster first, then infer pseudotime independently per
    cluster — each lineage has its own developmental trajectory, which avoids
    the global DPT crossing different lineages and distorting the pseudotime
    ordering.

    Works in global mode or per-cluster mode. ``suffix`` controls the
    checkpoint subdirectory.
    """
    _log = logger.getChild(suffix or "global")

    # ====================================================================
    # Adaptive parameters: relax thresholds for small clusters to reduce
    # false negatives caused by insufficient statistical power
    # ====================================================================
    n_cells = rna_adata.n_obs
    import copy
    _cfg = copy.deepcopy(config)

    # 1. DPT n_neighbors: use fewer neighbors for small clusters to keep the
    #    KNN graph from becoming overly dense
    default_n_neighbors = _cfg["pseudotime"]["n_neighbors"]
    adaptive_n_neighbors = min(default_n_neighbors, max(5, n_cells // 3))
    _cfg["pseudotime"]["n_neighbors"] = adaptive_n_neighbors

    # 2. Granger reference_bins: must not exceed the actual bin count of the
    #    cluster, otherwise correction = (df_ref/df_actual)² over-penalizes
    #    small clusters
    min_cells_per_bin = _cfg["pseudotime"].get("min_cells_per_bin", 5)
    max_bins = _cfg["pseudotime"]["n_bins"]
    est_bins = max(5, min(n_cells // min_cells_per_bin, max_bins))
    adaptive_ref_bins = min(
        _cfg["granger"].get("reference_bins", 50),
        max(est_bins, 10),
    )
    _cfg["granger"]["reference_bins"] = adaptive_ref_bins

    # 3. Granger composite score threshold: 50 cells → 50% baseline,
    #    200+ cells → 100% baseline
    base_composite = _cfg["granger"].get("min_composite_score", 0.3)
    scale = min(1.0, n_cells / 200.0)
    _cfg["granger"]["min_composite_score"] = round(base_composite * (0.5 + 0.5 * scale), 3)

    if n_cells < 200:
        _log.info(
            f"Adaptive parameters (n_cells={n_cells}, est_bins={est_bins}): "
            f"n_neighbors={adaptive_n_neighbors}, "
            f"reference_bins={adaptive_ref_bins}, "
            f"min_composite_score={_cfg['granger']['min_composite_score']}"
        )

    # ====================================================================
    # Step 3: Pseudotime inference (independent within each cluster —
    # cluster first, then pseudotime, not pseudotime first then clustering)
    # ====================================================================
    _log.info("--- Step 3: pseudotime inference (independent per cluster) ---")
    pseudotime_df = infer_pseudotime(rna_adata, atac_adata, _cfg)

    # Get the actual bin count to further calibrate reference_bins
    actual_bins = pseudotime_df["bin"].nunique()
    if actual_bins < est_bins * 0.8:
        _cfg["granger"]["reference_bins"] = max(actual_bins, 10)
        _log.info(
            f"reference_bins recalibrated: {adaptive_ref_bins} → "
            f"{_cfg['granger']['reference_bins']} (actual bins={actual_bins})"
        )

    # ====================================================================
    # Ablation experiment: randomly shuffle pseudotime values
    # Shuffle the pseudotime values themselves so that the pd.qcut in the
    # downstream _adaptive_rebin cuts quantiles on out-of-order values and
    # each bin aggregates random cells. This destroys the temporal-axis
    # ordering at its source and verifies whether pseudotime is the core
    # information source for causal inference.
    # ====================================================================
    _ablation_shuffle = _cfg["pseudotime"].get("ablation_shuffle_pseudotime", False)
    if _ablation_shuffle:
        import numpy as np
        from .kinetics import _adaptive_rebin
        from .granger import _bin_expression

        _old_pt = pseudotime_df["pseudotime"].values.copy()
        _old_bin = pseudotime_df["bin"].values.copy() if "bin" in pseudotime_df.columns else None

        # Save the pre-shuffle binning results for comparison
        pcfg = _cfg["pseudotime"]
        _smooth_kernel = pcfg.get("smooth_kernel", "hard")
        _overlap = pcfg.get("smooth_overlap_factor") if _smooth_kernel == "gaussian" else None
        old_bin_indices, old_n_bins = _adaptive_rebin(
            pseudotime_df, rna_adata.n_obs,
            max_bins=pcfg["n_bins"],
            min_cells_per_bin=pcfg.get("min_cells_per_bin", 5),
            target_bins=pcfg["n_bins"] if _smooth_kernel == "gaussian" else None,
            overlap_factor=_overlap,
        )
        pseudotime_df["bin"] = old_bin_indices
        rna_binned_before = _bin_expression(rna_adata, pseudotime_df, old_n_bins,
                                           smooth="hard",
                                           min_cells_per_bin=pcfg.get("min_cells_per_bin", 5))

        # More aggressive shuffling: assign cells to bins completely at random
        # instead of shuffling pseudotime values (because qcut cuts on
        # quantiles, shuffling values may have limited effect)
        rng = np.random.default_rng(42)
        n_cells = len(pseudotime_df)
        n_bins = old_n_bins

        # Generate a random bin assignment (each cell is assigned to a random bin)
        random_bins = rng.integers(0, n_bins, size=n_cells)
        pseudotime_df["bin"] = random_bins

        # Shuffle the pseudotime values (keeping consistency)
        rng.shuffle(_old_pt)
        pseudotime_df["pseudotime"] = _old_pt

        # After shuffling, branch_boundaries are meaningless; clear them
        if hasattr(pseudotime_df, "attrs"):
            pseudotime_df.attrs["branch_boundaries"] = None

        # Force hard mode to avoid gaussian smoothing washing out the shuffle effect
        _original_smooth = _cfg["pseudotime"].get("smooth_kernel", "hard")
        if _original_smooth == "gaussian":
            _cfg["pseudotime"]["smooth_kernel"] = "hard"
            _log.info(f"  Ablation: gaussian mode would smooth away the shuffle effect; forcing hard mode")

        # Compute the post-shuffle binning results
        rna_binned_after = _bin_expression(rna_adata, pseudotime_df, n_bins,
                                          smooth="hard",
                                          min_cells_per_bin=pcfg.get("min_cells_per_bin", 5))

        _log.info(
            f"=== Ablation: cells randomly assigned to bins === "
            f"(seed=42, smooth_kernel=hard, fully random assignment, independent of pseudotime)"
        )
        _log.info(f"  Pseudotime range before shuffling: [{_old_pt.min():.4f}, {_old_pt.max():.4f}]")
        _log.info(f"  Pseudotime range after shuffling: [{pseudotime_df['pseudotime'].min():.4f}, {pseudotime_df['pseudotime'].max():.4f}]")

        _log.info(f"  Bin distribution before shuffling: {np.bincount(old_bin_indices.astype(int))[:5]}...")
        _log.info(f"  Bin distribution after shuffling: {np.bincount(random_bins.astype(int))[:5]}...")

        # Check whether the bins actually changed
        bin_changed = not np.array_equal(old_bin_indices, random_bins)
        _log.info(f"  Did the bins change: {bin_changed}")

        # Debug: check whether pseudotime_df["bin"] was really updated
        _log.info(f"  [DEBUG] first 5 values of pseudotime_df['bin']: {pseudotime_df['bin'].values[:5]}")
        _log.info(f"  [DEBUG] first 5 values of random_bins: {random_bins[:5]}")
        _log.info(f"  [DEBUG] does pseudotime_df['bin'] equal random_bins: {np.array_equal(pseudotime_df['bin'].values, random_bins)}")

        # Debug: check whether rna_binned_before and rna_binned_after really differ
        _log.info(f"  [DEBUG] rna_binned_before shape: {rna_binned_before.shape}")
        _log.info(f"  [DEBUG] rna_binned_after shape: {rna_binned_after.shape}")
        _log.info(f"  [DEBUG] first 5 values of rna_binned_before: {rna_binned_before[0, :5]}")
        _log.info(f"  [DEBUG] first 5 values of rna_binned_after: {rna_binned_after[0, :5]}")

        # Check whether the binning results really changed
        rna_diff = np.abs(rna_binned_after - rna_binned_before).max()
        _log.info(f"  Max RNA binning difference before vs after shuffling: {rna_diff:.6f}")
        if rna_diff < 1e-6:
            _log.warning("  ⚠️ Warning: RNA binning results are nearly identical before and after shuffling!")
        else:
            _log.info(f"  ✓ RNA binning results differ before vs after shuffling; the difference is significant")

        # Flag downstream modules that the bins need to be recomputed
        pseudotime_df.attrs["_need_rebin"] = True

    # ====================================================================
    # Step 4: Granger causality test
    # ====================================================================
    _log.info("--- Step 4: Granger causality test (peak→gene) ---")
    _log.info(
        "Core principle: exploit the ATAC-RNA time-lag effect — "
        "chromatin opens before transcriptional activation"
    )

    ckpt = _load_checkpoint(config, "step4_granger", suffix)
    if ckpt is not None:
        granger_results = ckpt["granger_results"]
    else:
        granger_results = granger_test(
            rna_adata, atac_adata, pseudotime_df, _cfg
        )
        _save_checkpoint(config, "step4_granger",
                         {"granger_results": granger_results}, suffix)

    if granger_results.empty:
        _log.error("Granger test found no causal pairs")
        return {"error": "no_causal_pairs"}

    # ====================================================================
    # Step 5: ATAC→RNA transfer function fitting
    # ====================================================================
    _log.info("--- Step 5: ATAC→RNA transfer function fitting ---")
    _log.info("Method: shared neural network learning the A_peak → R_g mapping")

    ckpt = _load_checkpoint(config, "step5_atac_to_rna", suffix)
    if ckpt is not None:
        causal_edges = ckpt["causal_edges"]
        transfer_functions = ckpt["transfer_functions"]
        transfer_models = ckpt.get("transfer_models")
        _validate_step5_checkpoint(ckpt)
        root_atac = ckpt.get("root_atac")
    else:
        causal_edges = build_causal_grn(granger_results)
        transfer_functions, transfer_models, root_atac = fit_atac_to_rna(
            rna_adata, atac_adata, pseudotime_df, causal_edges, _cfg,
            granger_results=granger_results,
        )
        _save_checkpoint(config, "step5_atac_to_rna", {
            "causal_edges": causal_edges,
            "transfer_functions": transfer_functions,
            "transfer_models": transfer_models,
            "root_atac": root_atac,
        }, suffix)

    # ====================================================================
    # Step 6: RNA→ATAC TF regulatory weight learning
    # ====================================================================
    _log.info("--- Step 6: RNA→ATAC TF regulatory weight learning ---")
    _log.info(
        "Method: motif scanning to identify candidate TFs + per-TF/per-lag "
        "single-feature Pearson to determine regulatory weights"
    )

    ckpt = _load_checkpoint(config, "step6_rna_to_atac", suffix)
    if ckpt is not None:
        tf_weights = ckpt["tf_weights"]
        _validate_step6_checkpoint(tf_weights)
    else:
        tf_weights = fit_rna_to_atac(
            rna_adata, atac_adata, pseudotime_df, causal_edges, _cfg,
            target_gene_types=gene_types,
            root_atac=root_atac,
            rna_raw=rna_raw,
        )
        _save_checkpoint(config, "step6_rna_to_atac",
                         {"tf_weights": tf_weights}, suffix)

    # ====================================================================
    # Step 7: Causal GRN integration
    # ====================================================================
    _log.info("--- Step 7: causal GRN integration ---")
    n_tf_edges = len(tf_weights)
    n_peak_gene_edges = len(causal_edges)
    n_atac_rna_edges = len(transfer_functions)
    _log.info(
        f"Network statistics: causal peak→gene edges={n_peak_gene_edges}, "
        f"TF→peak relations={n_tf_edges}, "
        f"ATAC→RNA transfer functions={n_atac_rna_edges}, "
        f"gene count={len(causal_edges['gene'].unique())}"
    )

    # ====================================================================
    # Step 8: Perturbation simulation (knockout_type switch: "gene" = TF/gene
    # KO, "peak" = peak KO)
    # ====================================================================
    ko_type = _cfg.get("perturbation", {}).get("knockout_type", "gene")
    if ko_type == "peak":
        _log.info("--- Step 8: peak perturbation simulation (direct mode) ---")
        _log.info("Method: Δ accessibility of the specified peak → propagated along the existing peak→gene transfer functions")
        peak_cfg = _cfg.get("perturbation", {}).get("peak_ko", {})
        if not peak_cfg.get("peak_id"):
            _log.error("  knockout_type=peak but perturbation.peak_ko.peak_id is not configured")
            peak_cfg["peak_id"] = ""
        pert_output = perturb_peak(
            peak_cfg["peak_id"],
            rna_adata,
            atac_adata,
            pseudotime_df,
            causal_edges,
            transfer_functions,
            tf_weights,
            _cfg,
            transfer_models=transfer_models,
            root_atac=root_atac,
        )
        aggregated_tf_peak = pd.DataFrame()
    else:
        _log.info("--- Step 8: perturbation simulation (bin_forward per-bin forward propagation) ---")
        _log.info(
            "Method: [RNA→ATAC] changed TF → recompute accessibility of affected peaks; "
            "[ATAC→RNA] changed peaks → recompute expression of affected genes"
        )
        pert_output = propagate_perturbation(
            target_genes,
            rna_adata,
            atac_adata,
            pseudotime_df,
            causal_edges,
            transfer_functions,
            tf_weights,
            _cfg,
            transfer_models=transfer_models,
            gene_types=gene_types,
            root_atac=root_atac,
        )
        aggregated_tf_peak = pert_output.get("aggregated_tf_peak_edges", pd.DataFrame())

    if "error" in pert_output:
        _log.warning(f"  Perturbation failed: {pert_output.get('error')}")
        perturbation_results = pd.DataFrame()
        pathway_edges = pd.DataFrame()
        upstream_tfs = pd.DataFrame()
    else:
        perturbation_results = pert_output["perturbation_results"]
        pathway_edges = pert_output.get("pathway_edges", pd.DataFrame())
        upstream_tfs = pert_output.get("upstream_tfs", pd.DataFrame())
    projection_table = pd.DataFrame()
    if _cfg.get("perturbation", {}).get("cell_projection", {}).get("enabled", False) and ko_type != "peak":
        projection_frames = []
        history_frames = []
        metadata_frames = []
        event_frames = []
        for projection_input in pert_output.get("projection_inputs", []):
            projected, _, history, metadata, events = _project_bin_forward_inputs(
                projection_input, _cfg, suffix or "global",
                ",".join(projection_input.get("target_genes", target_genes)),
            )
            if not projected.empty:
                projection_frames.append(projected)
            if not history.empty:
                history_frames.append(history)
            if not metadata.empty:
                metadata_frames.append(metadata)
            if not events.empty:
                event_frames.append(events)
        if projection_frames:
            projection_table = pd.concat(projection_frames, ignore_index=True)
        projection_history = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
        projection_metadata = pd.concat(metadata_frames, ignore_index=True) if metadata_frames else pd.DataFrame()
        projection_events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    else:
        projection_history = pd.DataFrame()
        projection_metadata = pd.DataFrame()
        projection_events = pd.DataFrame()
    output = {
        "causal_edges": causal_edges,
        "transfer_functions": transfer_functions,
        "tf_peak_weights": tf_weights,
        "perturbation_results": perturbation_results,
        "pathway_edges": pathway_edges,
        "aggregated_tf_peak_edges": aggregated_tf_peak,
        "atac_changes": pert_output.get("atac_changes", pd.DataFrame()),
        "upstream_tfs": upstream_tfs,
        "peak_id": pert_output.get("peak_id", ""),
        "prediction_mode": pert_output.get("prediction_mode", ""),
        "baseline_accessibility": pert_output.get("baseline_accessibility"),
        "perturbed_accessibility": pert_output.get("perturbed_accessibility"),
        "delta_accessibility": pert_output.get("delta_accessibility"),
        "n_affected_edges": pert_output.get("n_affected_edges", 0),
        "granger_results": granger_results,
        "pseudotime": pseudotime_df,
        "n_cells": rna_adata.n_obs,
        # Required by merged cell-L0 aggregation even when the optional
        # projection report is disabled.
        "projection_inputs": pert_output.get("projection_inputs", []),
    }
    if _cfg.get("perturbation", {}).get("cell_projection", {}).get("enabled", False):
        output["cell_projection"] = projection_table
        output["projection_history"] = projection_history
        output["projection_bin_metadata"] = projection_metadata
        output["projection_events"] = projection_events
    return output


# ============================================================================
# Main pipeline
# ============================================================================

def run_pipeline(config_path: str) -> dict:
    """
    Full CausalBridge analysis workflow.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file

    Returns
    -------
    results : dict
        Dictionary containing all module outputs and the final results
    """
    # ========================================================================
    # Step 1: Load configuration + data
    # ========================================================================
    logger.info("=" * 60)
    logger.info("Step 1/8: load configuration and data")
    logger.info("=" * 60)

    config = load_config(config_path)

    # Print all paths to avoid cache/checkpoint/data mix-ups
    _log_paths(config)
    rna_raw, atac_raw = load_data(config)
    target_genes, gene_types = load_target_genes(config["input"]["target_genes"])

    # If the user requested whole-genome mode, expand to all expressed genes
    if target_genes == ["__ALL__"]:
        target_genes = list(rna_raw.var_names)
        gene_types = {g: "TF" for g in target_genes}  # whole-genome defaults to TF
        logger.info(f"Whole-genome mode: {len(target_genes)} genes")

    # ========================================================================
    # Step 2: Preprocessing
    # ========================================================================
    logger.info("=" * 60)
    logger.info("Step 2/8: data preprocessing")
    logger.info("=" * 60)

    rna_adata = preprocess_rna(
        rna_raw,
        min_cells_per_gene=config["preprocess"]["rna"]["min_cells_per_gene"],
        min_umi_per_cell=config["preprocess"]["rna"]["min_umi_per_cell"],
        n_highly_variable_genes=config["preprocess"]["rna"]["n_highly_variable_genes"],
        keep_genes=target_genes,
    )
    atac_adata = preprocess_atac(
        atac_raw,
        min_cells_per_peak=config["preprocess"]["atac"]["min_cells_per_peak"],
        min_peaks_per_cell=config["preprocess"]["atac"]["min_peaks_per_cell"],
        binarize=config["preprocess"]["atac"]["binarize"],
    )

    # Ensure the RNA and ATAC datasets contain exactly matching cells
    rna_adata, atac_adata = match_cells(rna_adata, atac_adata)

    # Clustering: PCA + KNN + Leiden, providing labels for per-cluster modeling
    cluster_cfg = config.get("clustering", {})
    cluster_key = cluster_cfg.get("cluster_key", "leiden")
    clustering_enabled = cluster_cfg.get("enabled", True)

    if clustering_enabled:
        if cluster_key in rna_adata.obs.columns:
            logger.info(
                f"Found existing cluster labels '{cluster_key}' "
                f"({rna_adata.obs[cluster_key].nunique()} clusters); skipping reclustering"
            )
        else:
            logger.info(f"No '{cluster_key}' labels found; running PCA + KNN + Leiden clustering")
            rna_adata = cluster_cells(
                rna_adata,
                n_neighbors=cluster_cfg.get("n_neighbors", 20),
                n_pcs=cluster_cfg.get("n_pcs", 20),
                resolution=cluster_cfg.get("resolution", 0.8),
                key_added=cluster_key,
                random_state=cluster_cfg.get("random_state", 42),
            )

        # Export the top 50 highly expressed genes per cluster (excluding
        # mitochondrial genes) to verify the cluster mapping
        save_cluster_top_genes(
            rna_adata,
            cluster_key=cluster_key,
            n_top=50,
            output_path=str(
                Path(config["output"]["dir"]) / f"cluster_top50_genes.csv"
            ),
        )

        # Export the clustered RNA AnnData (with PCA, KNN graph, Leiden labels)
        clustered_h5ad_path = str(
            Path(config["output"]["dir"]) / "rna_clustered.h5ad"
        )
        normalize_anndata_string_metadata(rna_adata)
        rna_adata.write_h5ad(clustered_h5ad_path)
        logger.info(f"Clustered RNA data saved to: {clustered_h5ad_path}")

        # ====================================================================
        # PAGA lineage construction: merge connected clusters into continuous
        # lineages to avoid fragmenting them
        # ====================================================================
        paga_cfg = cluster_cfg.get("paga", {})
        if paga_cfg.get("enabled", True):
            build_paga_lineages(
                rna_adata,
                cluster_key=cluster_key,
                connectivity_threshold=paga_cfg.get("connectivity_threshold", 0.1),
                min_cells_per_lineage=paga_cfg.get("min_cells_per_lineage", 50),
            )
            use_lineages = True
        else:
            logger.info("PAGA lineage merging disabled (clustering.paga.enabled=false)")
            use_lineages = False
    else:
        logger.info("Clustering disabled (clustering.enabled=false); using global mode")
        use_lineages = False

    # Update the target gene list (drop genes lost during QC)
    target_genes = [g for g in target_genes if g in rna_adata.var_names]
    gene_types = {g: gene_types.get(g, "TF") for g in target_genes}
    if len(target_genes) == 0:
        if config.get("perturbation", {}).get("knockout_type", "gene") == "peak":
            logger.info(
                "peak KO mode: target_genes is invalid or empty; ignoring it "
                "(the perturbation target is perturbation.peak_ko.peak_id)"
            )
        else:
            raise ValueError("All target genes were filtered out by QC; please check the gene names")
    logger.info(f"Valid target genes: {len(target_genes)}")

    # ========================================================================
    # Step 2.5: Compute CytoTRACE differentiation-potential scores
    # (Plan B: global computation → per-cluster mapping)
    # One global computation, stored in rna_adata.obs["cytotrace_score"];
    # when slicing per cluster later, the .obs column is automatically kept
    # with the slice, and _find_root_cell picks the root by argmax on that
    # column within each cluster (Plan B).
    # ========================================================================
    _root_method = config["pseudotime"].get("root_cell_method", "cytotrace")
    if _root_method == "cytotrace":
        from .granger import _compute_cytotrace_scores
        _n_top_genes = config["pseudotime"].get("cytotrace_n_top_genes", 200)
        cyto_scores = _compute_cytotrace_scores(rna_adata, n_top_genes=_n_top_genes)
        rna_adata.obs["cytotrace_score"] = cyto_scores
        logger.info(
            f"CytoTRACE global stemness scores: {len(cyto_scores)} cells, "
            f"stemness ∈ [{cyto_scores.min():.3f}, {cyto_scores.max():.3f}], "
            f"highest-stemness cell #{int(np.argmax(cyto_scores))} "
            f"→ obs['cytotrace_score']"
        )

    # ========================================================================
    # Steps 3-8: loop over lineages or clusters to run modeling + perturbation
    # simulation. After PAGA merges connected clusters, each lineage shares
    # one continuous pseudotime trajectory. If PAGA is disabled, fall back to
    # the original per-cluster mode.
    # ========================================================================
    logger.info("=" * 60)
    if use_lineages:
        logger.info("Step 3-8/8: modeling and perturbation simulation per PAGA lineage")
    else:
        logger.info("Step 3-8/8: modeling and perturbation simulation per cell cluster")
    logger.info("=" * 60)

    cluster_cfg = config.get("clustering", {})
    cluster_key = cluster_cfg.get("cluster_key", "leiden")
    clustering_enabled = cluster_cfg.get("enabled", True)
    min_cells = cluster_cfg.get("min_cells_per_cluster", 50)

    # --- Build group → cell mapping ---
    if use_lineages:
        group_key = "lineage"
        try:
            group_labels = sorted(rna_adata.obs[group_key].unique())
        except (ValueError, TypeError):
            group_labels = sorted(rna_adata.obs[group_key].unique(), key=str)
        cells_by_group = {
            g: rna_adata.obs_names[rna_adata.obs[group_key] == g]
            for g in group_labels
        }
        # Log the original clusters contained in each lineage
        logger.info(f"PAGA merging: {len(group_labels)} lineages")
        for lin in group_labels:
            lin_mask = rna_adata.obs[group_key] == lin
            lin_clusters = sorted(
                set(rna_adata.obs.loc[lin_mask, cluster_key]),
                key=lambda x: (str(x).isdigit(), str(x)),
            )
            n_lin_cells = lin_mask.sum()
            logger.info(
                f"  {lin}: {n_lin_cells} cells, "
                f"original clusters = [{', '.join(str(c) for c in lin_clusters)}]"
            )

    elif clustering_enabled and cluster_key in rna_adata.obs.columns:
        group_key = cluster_key
        try:
            group_labels = sorted(rna_adata.obs[cluster_key].unique(), key=int)
        except ValueError:
            group_labels = sorted(rna_adata.obs[cluster_key].unique(), key=str)
        cells_by_group = {
            cl: rna_adata.obs_names[rna_adata.obs[cluster_key] == cl]
            for cl in group_labels
        }
        logger.info(f"Found {len(group_labels)} clusters (key='{cluster_key}')")

    elif clustering_enabled:
        logger.warning(
            f"Column '{cluster_key}' not found in AnnData.obs; falling back to global mode. "
            f"Run sc.pp.neighbors + sc.tl.leiden first to add cluster labels."
        )
        cells_by_group = {"global": rna_adata.obs_names}

    else:
        logger.info("Clustering disabled; using global Palantir pseudotime mode")
        cells_by_group = {"global": rna_adata.obs_names}

    # --- Check groups with too few cells ---
    small_groups = [
        g for g, cells in cells_by_group.items()
        if len(cells) < min_cells
    ]
    if small_groups:
        n_small = sum(len(cells_by_group[g]) for g in small_groups)
        if len(small_groups) == len(cells_by_group):
            # All groups are too small → fall back to global
            logger.warning(
                f"All groups {small_groups} have too few cells (<{min_cells}), "
                f"{n_small} cells in total. Falling back to global mode."
            )
            cells_by_group = {"global": rna_adata.obs_names}
            use_lineages = False
            fallback_method = cluster_cfg.get("fallback_pseudotime_method", "palantir")
            saved_method = config["pseudotime"]["method"]
            config["pseudotime"]["method"] = fallback_method
            logger.info(
                f"Pseudotime method switched: {saved_method} → {fallback_method}"
            )
        else:
            large_groups = [
                g for g in cells_by_group if g not in small_groups
            ]
            logger.info(
                f"Groups {small_groups} have too few cells (<{min_cells}); "
                f"{n_small} cells in total will be skipped. "
                f"Continuing with {len(large_groups)} larger groups."
            )

    # --- Per-group loop ---
    all_cluster_results = []
    cluster_cell_counts = {}
    n_skipped = 0

    for grp, cells in cells_by_group.items():
        n_cells = len(cells)
        if n_cells < min_cells:
            logger.warning(
                f"  Group '{grp}': {n_cells} cells < {min_cells}; skipping"
            )
            n_skipped += 1
            continue

        # peak KO mode + a state that pins a single cluster: run only the
        # targeted cluster
        _ko_type_loop = config.get("perturbation", {}).get("knockout_type", "gene")
        _peak_state = config.get("perturbation", {}).get("peak_ko", {}).get("state", "all")
        if _ko_type_loop == "peak" and str(_peak_state) != "all" and str(grp) != str(_peak_state):
            logger.info(f"  peak_ko.state={_peak_state}: skipping cluster '{grp}'")
            n_skipped += 1
            continue

        if use_lineages:
            suffix = f"lineage_{grp}"
            label = f"lineage '{grp}'"
        elif grp != "global":
            suffix = f"cluster_{grp}"
            label = f"cluster '{grp}'"
        else:
            suffix = None
            label = "global"
        logger.info(f"--- {label}: {n_cells} cells ---")

        result = _run_modeling_for_subset(
            rna_adata[cells],
            atac_adata[cells],
            target_genes,
            gene_types,
            config,
            suffix=suffix,
            rna_raw=rna_raw,
        )

        if "error" in result:
            logger.warning(f"  {label}: {result['error']}")
            continue

        cluster_cell_counts[str(grp)] = n_cells

        # Annotate the group of origin
        pert = result["perturbation_results"]
        if not pert.empty:
            pert = pert.copy()
            pert["cluster"] = str(grp)
            # When lineages are used, keep the original cluster labels too
            if use_lineages and cluster_key in rna_adata[cells].obs.columns:
                # Keep each cell's original cluster label
                cell_to_cluster = rna_adata[cells].obs[cluster_key]
                pert["_orig_cluster"] = pert.index.map(
                    lambda b: str(cell_to_cluster.get(b, ""))
                )
            result["perturbation_results"] = pert

        pw = result.get("pathway_edges")
        if pw is not None and not pw.empty:
            pw = pw.copy()
            pw["cluster"] = str(grp)
            result["pathway_edges"] = pw

        up = result.get("upstream_tfs")
        if up is not None and not up.empty:
            up = up.copy()
            up["cluster"] = str(grp)
            result["upstream_tfs"] = up

        result["cluster"] = str(grp)
        all_cluster_results.append(result)

    if n_skipped > 0:
        logger.info(f"Skipped {n_skipped} clusters with insufficient sample size")

    if not all_cluster_results:
        logger.error("No cluster produced valid results; aborting the pipeline")
        return {"error": "all_clusters_failed", "config": config}

    # ========================================================================
    # Step 9: Aggregate results and save
    # ========================================================================
    logger.info("=" * 60)
    logger.info("Aggregating results and saving")
    logger.info("=" * 60)

    # Merge the perturbation results from all clusters
    perturbation_dfs = [
        r["perturbation_results"] for r in all_cluster_results
        if not r["perturbation_results"].empty
    ]
    if perturbation_dfs:
        all_perturbation = pd.concat(perturbation_dfs, ignore_index=True)
    else:
        all_perturbation = pd.DataFrame()

    # Compute the whole-tissue weighted average delta_rna (weighted by the
    # per-cluster cell counts)
    total_cells = sum(cluster_cell_counts.values())
    if not all_perturbation.empty and total_cells > 0 and len(cluster_cell_counts) > 1:
        pert_w = all_perturbation.copy()
        pert_w["_weight"] = pert_w["cluster"].map(cluster_cell_counts)
        grouped = pert_w.groupby(["target_gene", "affected_gene"])
        merged_rows = []
        for (tg, ag), grp in grouped:
            wsum = grp["_weight"].sum()
            if wsum == 0:
                continue
            signed_delta = (grp["delta_rna"] * grp["_weight"]).sum() / total_cells
            abs_delta = (grp["delta_rna"].abs() * grp["_weight"]).sum() / total_cells
            merged_rows.append({
                "target_gene": tg, "affected_gene": ag,
                "delta_rna": abs_delta, "delta_rna_signed": signed_delta,
                "mediated_by_atac": grp["mediated_by_atac"].iloc[0],
                "mechanism": grp["mechanism"].iloc[0],
                "propagation_depth": grp["propagation_depth"].iloc[0],
                "converged": grp["converged"].iloc[0],
                "is_fallback": grp["is_fallback"].iloc[0],
                "perturbation_mode": grp["perturbation_mode"].iloc[0],
                "n_clusters_observed": len(grp),
            })
        merged_perturbation = pd.DataFrame(merged_rows)
    else:
        merged_perturbation = all_perturbation.copy() if not all_perturbation.empty else pd.DataFrame()

    # Merge the pathway-level edge records from all clusters
    pathway_dfs = [
        r["pathway_edges"] for r in all_cluster_results
        if r.get("pathway_edges") is not None and not r["pathway_edges"].empty
    ]
    if pathway_dfs:
        all_pathway_edges = pd.concat(pathway_dfs, ignore_index=True)
    else:
        all_pathway_edges = pd.DataFrame(columns=[
            "target_gene", "tf_name", "peak_id", "affected_gene",
            "delta_r_raw", "delta_r_total", "mechanism", "r2_score", "tf_lag", "bin_min", "bin_max", "cluster"])

    # Merge the aggregated TF→peak edges from all clusters (including
    # window-validated flip information)
    agg_edge_dfs = [
        r["aggregated_tf_peak_edges"] for r in all_cluster_results
        if r.get("aggregated_tf_peak_edges") is not None
        and not r["aggregated_tf_peak_edges"].empty
    ]
    if agg_edge_dfs:
        all_agg_edges = pd.concat(agg_edge_dfs, ignore_index=True)
    else:
        all_agg_edges = pd.DataFrame(columns=[
            "tf_gene", "peak_id", "weight", "lag", "flipped", "original_lag"])

    # Merge the graph structures from all clusters, annotating the cluster of origin
    causal_edges_list = []
    transfer_functions_list = []
    tf_weights_list = []
    for i, r in enumerate(all_cluster_results):
        cl_label = str(r.get("cluster", i))
        ce = r["causal_edges"].copy()
        ce["cluster"] = cl_label
        causal_edges_list.append(ce)
        tfw = r["tf_peak_weights"].copy()
        tfw["cluster"] = cl_label
        tf_weights_list.append(tfw)
        tf = r["transfer_functions"].copy()
        tf["cluster"] = cl_label
        transfer_functions_list.append(tf)

    merged_causal_edges = pd.concat(causal_edges_list, ignore_index=True)
    merged_tf_weights = pd.concat(tf_weights_list, ignore_index=True)
    merged_transfer_functions = pd.concat(transfer_functions_list, ignore_index=True)

    # Merge the pseudotime across clusters (inferred independently per
    # cluster; annotated with the cluster of origin)
    pseudotime_dfs = []
    for i, r in enumerate(all_cluster_results):
        cl_label = str(r.get("cluster", i))
        pt_df = r.get("pseudotime")
        if pt_df is not None and not pt_df.empty:
            pt_df = pt_df.copy()
            pt_df["cluster"] = cl_label
            pseudotime_dfs.append(pt_df)
    combined_pseudotime = pd.concat(pseudotime_dfs) if pseudotime_dfs else pd.DataFrame()

    # Merge the ATAC changes across clusters (for IGV visualization)
    atac_changes_dfs = [
        r["atac_changes"] for r in all_cluster_results
        if r.get("atac_changes") is not None and not r["atac_changes"].empty
    ]
    combined_atac_changes = pd.concat(atac_changes_dfs, ignore_index=True) if atac_changes_dfs else pd.DataFrame()

    # Cross-cluster weighted average of ATAC changes (aligned with the
    # perturbation_merged logic: weighted by per-cluster cell counts).
    # Each peak gets both signed and absolute delta columns to prevent
    # cross-cluster sign cancellation.
    # Unlike merged cell-L0 aggregation, these tables only contain successful
    # modeled contexts, so skipped/failed contexts must not enter this
    # denominator.  Keep this count local to preserve the prior ATAC behavior.
    modeled_total_cells = sum(cluster_cell_counts.values())
    if atac_changes_dfs and modeled_total_cells > 0 and len(cluster_cell_counts) > 1:
        ac_tagged = []
        for r in all_cluster_results:
            ac = r.get("atac_changes")
            if ac is None or ac.empty:
                continue
            ac = ac.copy()
            cl = str(r.get("cluster", ""))
            ac["cluster"] = cl
            ac["_weight"] = cluster_cell_counts.get(cl, 0)
            ac_tagged.append(ac)
        ac_full = pd.concat(ac_tagged, ignore_index=True)
        merged_atac_rows = []
        for peak_id, grp in ac_full.groupby("peak_id"):
            wsum = grp["_weight"].sum()
            if wsum == 0:
                continue
            row = {
                "peak_id": peak_id,
                "chr": grp["chr"].iloc[0],
                "start": int(grp["start"].iloc[0]),
                "end": int(grp["end"].iloc[0]),
                "delta_accessibility_signed": float(
                    (grp["delta_accessibility"] * grp["_weight"]).sum() / modeled_total_cells
                ),
                "delta_accessibility": float(
                    (grp["delta_accessibility"].abs() * grp["_weight"]).sum() / modeled_total_cells
                ),
                "n_clusters_observed": len(grp),
            }
            if "z_score" in grp.columns:
                row["z_score"] = float(
                    (grp["z_score"] * grp["_weight"]).sum() / modeled_total_cells
                )
                row["is_significant"] = abs(row["z_score"]) >= config.get("perturbation", {}).get(
                    "atac_significance_zscore", 2.0
                )
            merged_atac_rows.append(row)
        merged_atac_changes = pd.DataFrame(merged_atac_rows)
    else:
        merged_atac_changes = combined_atac_changes.copy() if not combined_atac_changes.empty else pd.DataFrame()

    # Merge the upstream TF attribution across clusters (peak KO mode; empty
    # in gene mode)
    upstream_dfs = [
        r["upstream_tfs"] for r in all_cluster_results
        if r.get("upstream_tfs") is not None and not r["upstream_tfs"].empty
    ]
    all_upstream_tfs = pd.concat(upstream_dfs, ignore_index=True) if upstream_dfs else pd.DataFrame()

    # Cross-cluster weighted merged view of upstream TFs (weighted by
    # per-cluster cell counts, following the perturbation_merged idea)
    merged_upstream_tfs = pd.DataFrame()
    if not all_upstream_tfs.empty and modeled_total_cells > 0 and len(cluster_cell_counts) > 1:
        _up = all_upstream_tfs.copy()
        _up["_weight"] = _up["cluster"].map(cluster_cell_counts).fillna(0)
        _up_rows = []
        for (pk, tf), grp in _up.groupby(["peak_id", "tf_gene"]):
            wsum = grp["_weight"].sum()
            if wsum == 0:
                continue
            _up_rows.append({
                "peak_id": pk,
                "tf_gene": tf,
                "weight_signed_merged": float((grp["weight"] * grp["_weight"]).sum() / modeled_total_cells),
                "weight_abs_merged": float((grp["weight"].abs() * grp["_weight"]).sum() / modeled_total_cells),
                "n_clusters_observed": len(grp),
            })
        if _up_rows:
            merged_upstream_tfs = pd.DataFrame(_up_rows).sort_values(
                "weight_abs_merged", ascending=False
            )

    # Peak mode: summarize the baseline/perturbed/Δ accessibility of the
    # perturbed peak per cluster (for four-part report generation)
    peak_summary = pd.DataFrame()
    if any(r.get("prediction_mode", "").startswith("peak_") for r in all_cluster_results):
        _ps_rows = []
        for r in all_cluster_results:
            if not r.get("prediction_mode", "").startswith("peak_"):
                continue
            _ps_rows.append({
                "peak_id": r.get("peak_id", ""),
                "cluster": str(r.get("cluster", "")),
                "baseline_accessibility": r.get("baseline_accessibility"),
                "perturbed_accessibility": r.get("perturbed_accessibility"),
                "delta_accessibility": r.get("delta_accessibility"),
                "n_affected_genes": len(r.get("perturbation_results", pd.DataFrame())),
                "n_upstream_tfs": len(r.get("upstream_tfs", pd.DataFrame())),
                "n_affected_edges": r.get("n_affected_edges", 0),
            })
        peak_summary = pd.DataFrame(_ps_rows)

    projection_tables = [r.get("cell_projection") for r in all_cluster_results
                         if r.get("cell_projection") is not None and not r.get("cell_projection").empty]
    projection_table = pd.concat(projection_tables, ignore_index=True) if projection_tables else pd.DataFrame()
    projection_histories = [r.get("projection_history") for r in all_cluster_results
                            if r.get("projection_history") is not None and not r.get("projection_history").empty]
    projection_history = pd.concat(projection_histories, ignore_index=True) if projection_histories else pd.DataFrame()
    projection_metadata_frames = [r.get("projection_bin_metadata") for r in all_cluster_results
                                  if r.get("projection_bin_metadata") is not None and not r.get("projection_bin_metadata").empty]
    projection_metadata = pd.concat(projection_metadata_frames, ignore_index=True) if projection_metadata_frames else pd.DataFrame()
    projection_event_frames = [r.get("projection_events") for r in all_cluster_results
                               if r.get("projection_events") is not None and not r.get("projection_events").empty]
    projection_events = pd.concat(projection_event_frames, ignore_index=True) if projection_event_frames else pd.DataFrame()
    projection_summary = pd.DataFrame()
    projection_branch_probabilities = pd.DataFrame()
    if not projection_table.empty:
        projection_summary = (
            projection_table.groupby(["source_context", "target_gene", "projection_status"], dropna=False)
            .size().reset_index(name="n_cells")
        )
        probability_cols = [c for c in projection_table.columns if c.startswith("branch_probability_")]
        if probability_cols:
            projection_branch_probabilities = projection_table[
                ["cell_id", "source_context", "target_gene"] + probability_cols
            ].melt(id_vars=["cell_id", "source_context", "target_gene"],
                   var_name="branch", value_name="branch_probability")
            projection_branch_probabilities["branch"] = projection_branch_probabilities["branch"].str.replace(
                "branch_probability_", "", regex=False
            )

    all_results = {
        "causal_edges": merged_causal_edges,
        "transfer_functions": merged_transfer_functions,
        "tf_peak_weights": merged_tf_weights,
        "perturbation_results": all_perturbation,
        "perturbation_merged": merged_perturbation,
        "pathway_edges": all_pathway_edges,
        "aggregated_tf_peak_edges": all_agg_edges,
        "atac_changes": combined_atac_changes,
        "atac_changes_merged": merged_atac_changes,
        "upstream_tfs": all_upstream_tfs,
        "upstream_tfs_merged": merged_upstream_tfs,
        "peak_summary": peak_summary,
        "granger_results": all_cluster_results[0]["granger_results"],
        "pseudotime": combined_pseudotime,
        "cluster_results": all_cluster_results,
    }
    if config.get("perturbation", {}).get("cell_projection", {}).get("enabled", False):
        all_results["cell_projection"] = projection_table
        all_results["projection_history"] = projection_history
        all_results["projection_bin_metadata"] = projection_metadata
        all_results["projection_events"] = projection_events
        all_results["projection_summary"] = projection_summary
        all_results["projection_branch_probabilities"] = projection_branch_probabilities
    save_results(all_results, config)

    # Summary statistics
    if not all_perturbation.empty and "target_gene" in all_perturbation.columns:
        n_clusters = all_perturbation["cluster"].nunique() if "cluster" in all_perturbation.columns else 1
        n_unique_targets = all_perturbation["target_gene"].nunique()
        n_unique_affected = all_perturbation["affected_gene"].nunique()
        atac_mediated_pct = (
            all_perturbation["mediated_by_atac"].mean() * 100
            if "mediated_by_atac" in all_perturbation.columns
            else 0
        )
        logger.info("=" * 60)
        logger.info("CausalBridge analysis complete!")
        logger.info(f"  Number of cell clusters: {n_clusters}")
        logger.info(f"  Number of target genes: {n_unique_targets}")
        logger.info(f"  Number of affected genes: {n_unique_affected}")
        logger.info(f"  Predicted target→affected relations: {len(all_perturbation)}")
        logger.info(f"  Fraction mediated via ATAC: {atac_mediated_pct:.1f}%")
        logger.info(f"  Results directory: {config['output']['dir']}")
        logger.info("=" * 60)
    else:
        logger.warning("=" * 60)
        logger.warning("CausalBridge analysis finished, but perturbation simulation produced no results")
        logger.warning("  Possible causes: the target genes are absent from the causal network, or TF→peak edges are missing")
        logger.warning(f"  Results directory: {config['output']['dir']}")
        logger.warning("=" * 60)

    return all_results
