"""
CausalBridge I/O 模块：配置文件解析、数据加载、结果保存

所有输入/输出操作的统一入口，确保整个 pipeline 使用一致的文件格式。
"""

import yaml
import json
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
# 配置加载
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件，补充默认值。

    用户只需提供关键参数（如输入路径），其余参数使用内置默认值
    （与 config.yaml 同步）。此函数先加载用户配置，再与默认值合并（用户值优先）。

    Parameters
    ----------
    config_path : str
        用户 YAML 配置文件的路径

    Returns
    -------
    config : dict
        包含所有参数的完整配置字典
    """
    # 读取用户配置
    with open(config_path, "r") as f:
        user_config = yaml.safe_load(f)

    from .saturating_hybrid import HybridConfig
    from .spline_hybrid import SplineConfig
    # 默认值（确保所有必需字段存在）
    defaults = {
        "input": {"gene_identity_mode": "legacy", "annotation_gtf": None,
                  "rna_gene_id_column": "gene_ids"},
        "preprocess": {
            "atac_depth_normalization": {"enabled": False, "target_sum": 10000., "output_cap_expansion": .25},
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
            "adaptive_binning": {"enabled": False, "state_modality": "joint", "embedding": {
                "n_components": 5, "max_features": 512,
                "max_dense_elements": 2_000_000, "random_seed": 42}},
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
            "model_mode": "saturating_hybrid",
            "response_model": "saturation",
            "nn_response": {"hidden_width": 4, "epochs": 300, "lr": .01, "l2": .01, "seed": 42, "restarts": 1},
            "spline_response": SplineConfig().to_dict(),
            "saturating_hybrid": HybridConfig().to_dict(),
            "log_normalize_rna": True,
        },
        "rna_to_atac": {
            "response_model": "saturation",
            "nn_response": {"hidden_width": 4, "epochs": 300, "lr": .01, "l2": .01, "seed": 42, "restarts": 1},
            "spline_response": SplineConfig().to_dict(),
            "motif_db": "jaspar2024", "motif_pval_threshold": 1e-4,
            "motif_scan_workers": 1,
            "max_tfs_per_peak": 5,
            "tf_peak_pvalue_threshold": 0.05,  # TF→peak 边显著性过滤
            "min_abs_contribution": 0.01,
            "time_lag": True,  # 每个目标bin独立选择首个受附近WT数据支持的lag
            "tf_expression_filter": True,  # TF 表达过滤: 验证基因名是否在 RNA 矩阵中
            "max_lag": 0,  # 自动=max(1,n_bins//4)，只限制回看范围，不丢弃早期bin
        },
        "perturbation": {
            "aggregation_policy": "strict_complete",  # Output aggregation only; preserve library default.
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
            "export_details": False,
            "save_intermediate": True,
        },
    }

    # 递归合并：用户值覆盖默认值
    config = _deep_merge(defaults, user_config)
    _validate_config(config)
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并两个字典，override 中的值优先。"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _validate_config(config: dict):
    """检查配置中必需字段是否存在且合理。"""
    if not isinstance(config.get("output", {}).get("export_details", False), bool):
        raise ValueError("output.export_details must be a boolean")
    policy = config.get("perturbation", {}).get("aggregation_policy", "strict_complete")
    if policy not in ("strict_complete", "valid_cell_means"):
        raise ValueError("perturbation.aggregation_policy must be 'strict_complete' or "
                         f"'valid_cell_means'; got {policy!r}")
    from .atac_normalization import validate_atac_depth_config
    validate_atac_depth_config(config)
    _gene_identity_enabled(config)
    from .kinetics import _validate_motif_scan_workers
    from ._wt_aggregation import validate_model_mode
    validate_model_mode(config)
    if config.get("atac_to_rna", {}).get("model_mode") == "saturating_hybrid":
        from .saturating_transfer import validate_saturating_config
        validate_saturating_config(config)
        if config.get("pseudotime", {}).get("ablation_shuffle_pseudotime", False):
            raise ValueError("saturating_hybrid does not support shuffled pseudotime")
    _validate_motif_scan_workers(config.get("rna_to_atac", {}).get("motif_scan_workers", 1))
    from .screening_parallel import screening_workers
    for key in ("granger_workers", "local_lag_workers"):
        screening_workers(config, key)
    from .binning_adapter import _selector, _embedding_settings
    from .adaptive_binning import BinningConfig
    settings = _selector(config)
    _embedding_settings(settings.get("embedding", {}))
    BinningConfig(**{k: v for k, v in settings.items() if k not in {"enabled", "embedding", "state_modality"}})
    if settings.get("enabled", False) and config["pseudotime"].get("ablation_shuffle_pseudotime", False):
        raise ValueError("adaptive binning does not support shuffled pseudotime")
    required_inputs = ["rna_h5ad", "atac_h5ad"]
    for key in required_inputs:
        if key not in config.get("input", {}):
            raise ValueError(f"配置缺少必需字段: input.{key}")

    rna_path = Path(config["input"]["rna_h5ad"])
    atac_path = Path(config["input"]["atac_h5ad"])
    if not rna_path.exists():
        raise FileNotFoundError(f"RNA 文件不存在: {rna_path}")
    if not atac_path.exists():
        raise FileNotFoundError(f"ATAC 文件不存在: {atac_path}")

    # 检查参数范围合理性
    gr = config["granger"]
    if gr["significance_threshold"] <= 0 or gr["significance_threshold"] >= 1:
        raise ValueError("significance_threshold 应在 (0, 1) 之间")
    if gr["distance_thresh"] < 0:
        raise ValueError("distance_thresh 不能为负数")

    logger.info("配置验证通过")


def _gene_identity_enabled(config):
    settings = config.get("input", {})
    mode = settings.get("gene_identity_mode", "legacy")
    if mode not in ("legacy", "gtf_gene_id"):
        raise ValueError("input.gene_identity_mode must be legacy or gtf_gene_id")
    if mode == "gtf_gene_id":
        if not settings.get("annotation_gtf") or not Path(settings["annotation_gtf"]).is_file():
            raise ValueError("gtf_gene_id requires an existing input.annotation_gtf")
        if not isinstance(settings.get("rna_gene_id_column", "gene_ids"), str) or not settings.get("rna_gene_id_column", "gene_ids"):
            raise ValueError("input.rna_gene_id_column must be a nonempty column name")
    return mode == "gtf_gene_id"


def save_hvg_genes(rna, config):
    """Export actual selected model HVG flags; missing flags are not all-genes HVG."""
    if "highly_variable" not in rna.var:
        logger.warning("RNA lacks highly_variable flags; hvg_genes.csv was not exported")
        return None
    flags = rna.var["highly_variable"]
    if not pd.api.types.is_bool_dtype(flags.dtype) or flags.isna().any():
        raise ValueError("highly_variable must contain nonmissing boolean flags for HVG export")
    selected = rna.var.loc[flags].copy()
    identity = _gene_identity_enabled(config)
    frame = pd.DataFrame({
        "gene_id": selected.index.astype(str),
        "gene_name": (selected["gene_name"].astype(str).to_numpy()
                      if "gene_name" in selected else selected.index.astype(str)),
    })
    if not identity:
        # Legacy names are the modeling axis; prefer input IDs when available.
        column = config.get("input", {}).get("rna_gene_id_column", "gene_ids")
        if column in selected:
            frame["gene_id"] = selected[column].to_numpy()
    if "original_var_name" in selected:
        frame["original_var_name"] = selected.original_var_name.to_numpy()
    directory = Path(config["output"]["dir"])
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "hvg_genes.csv", index=False)
    logger.info("模型 HVG 列表已保存: %s (%d genes)", directory / "hvg_genes.csv", len(frame))
    return frame


def _gene_identity_runtime_config(config):
    """Validate identity settings on a copy, preserving configured output roots."""
    import copy
    result = copy.deepcopy(config)
    _gene_identity_enabled(result)
    return result


# ============================================================================
# 数据加载
# ============================================================================

def load_data(config: dict) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    加载 RNA 和 ATAC 两个 AnnData 文件，做基础校验。

    校验内容：
    1. 两个文件是否可正常读取
    2. 细胞 barcode 是否一一匹配
    3. .var 是否包含基因组坐标信息（chr, start, end）

    Returns
    -------
    rna_adata, atac_adata : AnnData
    """
    rna_path = config["input"]["rna_h5ad"]
    atac_path = config["input"]["atac_h5ad"]

    logger.info(f"加载 RNA 数据: {rna_path}")
    rna_adata = ad.read_h5ad(rna_path)

    logger.info(f"加载 ATAC 数据: {atac_path}")
    atac_adata = ad.read_h5ad(atac_path)

    # --- 校验基因组坐标列 ---
    for label, adata in [("RNA", rna_adata), ("ATAC", atac_adata)]:
        if label == "RNA" and _gene_identity_enabled(config):
            # The GTF adapter supplies authoritative RNA spans before QC.
            continue
        for col in ["chr", "start", "end"]:
            if col not in adata.var.columns:
                raise KeyError(
                    f"{label} AnnData 的 .var 缺少 '{col}' 列。"
                    f"请添加基因组坐标信息。当前列: {list(adata.var.columns)}"
                )

    # --- 校验细胞匹配 ---
    rna_cells = set(rna_adata.obs_names)
    atac_cells = set(atac_adata.obs_names)
    common = rna_cells & atac_cells

    if len(common) == 0:
        raise ValueError(
            "RNA 和 ATAC 数据没有共享的细胞 barcode，"
            "请确认两个文件来自同一组细胞的多组学测序。"
        )

    if len(common) < len(rna_cells) or len(common) < len(atac_cells):
        logger.warning(
            f"细胞不完全匹配: RNA={len(rna_cells)}, ATAC={len(atac_cells)}, "
            f"共有={len(common)}。将只保留共有细胞。"
        )
        rna_adata = rna_adata[list(common)].copy()
        atac_adata = atac_adata[list(common)].copy()

    # --- 确保细胞顺序一致 ---
    if not np.array_equal(rna_adata.obs_names, atac_adata.obs_names):
        atac_adata = atac_adata[rna_adata.obs_names].copy()

    logger.info(f"数据加载完成: {rna_adata.n_obs} 个细胞, "
                f"{rna_adata.n_vars} 个基因, {atac_adata.n_vars} 个 peak")
    return rna_adata, atac_adata


def load_target_genes(path: str) -> Tuple[List[str], Dict[str, str]]:
    """
    加载敲除目标基因列表及基因类型标注。

    支持三种格式：
    1. 双列格式：每行 "gene_name  gene_type"（gene_type = TF 或 target）
    2. 单列格式（向后兼容）：每行一个基因符号，默认 gene_type = "TF"
    3. 全基因组模式：文件内容为 "ALL" 或 "ALL_EXPRESSED"

    Returns
    -------
    genes : list of str
    gene_types : dict {gene_name: gene_type}
        gene_type 为 "TF" 或 "target"
        当基因不在标注中时，默认 "TF"
    """
    with open(path, "r") as f:
        raw = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

    if len(raw) == 0:
        raise ValueError(f"目标基因文件为空: {path}")

    if raw[0].upper().startswith("ALL"):
        logger.info("将使用全基因组模式（所有表达基因均可作为 KO 目标）")
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
                    f"未知 gene_type '{parts[1]}' (应为 TF 或 target)，"
                    f"基因 {gene} 回退为 'TF'"
                )
                gene_types[gene] = "TF"
        else:
            gene_types[gene] = "TF"  # 默认

    n_tf = sum(1 for t in gene_types.values() if t == "TF")
    n_target = sum(1 for t in gene_types.values() if t == "TARGET")
    logger.info(
        f"加载了 {len(genes)} 个目标基因 "
        f"(TF: {n_tf}, target: {n_target})"
    )
    return genes, gene_types


# ============================================================================
# 结果保存
# ============================================================================

def _present_gene_frame(frame, gene_identity):
    """Copy explicit output roles only; never alter model feature arrays."""
    from .gene_identity import annotate_gene_columns
    result = frame.copy(deep=True)
    for column in ("gene", "affected_gene", "target_gene", "tf_gene", "tf_name", "source_tf", "intermediate_tf"):
        if column not in result:
            continue
        fields = [key for key in result if key == column or key.startswith(column + "_")]
        values = result[fields].reset_index(drop=True).copy()
        sentinel = values[column].map(lambda value: isinstance(value, str) and (
            value.startswith("PEAK:") or value in ("unknown", "joint_vector") or (column == "tf_name" and value == "peak")))
        if column + "_id" in values:
            from .gene_identity import normalize_gene_id
            biological_id = values[column + "_id"].map(lambda value: isinstance(value, str)
                and normalize_gene_id(value) in gene_identity.by_id)
            sentinel &= ~biological_id
        chunks = []
        for mask, role in ((~sentinel, "genes" if column == "target_gene" else "gene"),
                           (sentinel, "gene_or_sentinel")):
            if mask.any():
                chunks.append(annotate_gene_columns(values.loc[mask], gene_identity, columns={column: role}))
        if chunks:
            annotated = pd.concat(chunks).sort_index()
            for key in annotated:
                result[key] = annotated[key].to_numpy()
    if "unavailable_origins" in result:
        labels, bases = [], []
        for value in result["unavailable_origins"]:
            if pd.isna(value):
                labels.append(value)
                bases.append(value)
                continue
            origins = json.loads(value)
            display_origins, base_origins = [], []
            for origin in origins:
                gene_id, separator, status = origin.partition(":")
                if gene_id not in gene_identity.by_id:
                    raise ValueError(f"unknown gene ID in unavailable origin: {gene_id}")
                display_origins.append(gene_identity.display_labels[gene_id] + separator + status)
                base_origins.append(gene_identity.by_id[gene_id].gene_name + separator + status)
            labels.append(json.dumps(display_origins))
            bases.append(json.dumps(base_origins))
        result["unavailable_origins_display_json"] = labels
        result["unavailable_origins_names_json"] = bases
    return result


def _present_gene_results(results, gene_identity):
    result = dict(results)
    for key, value in results.items():
        if isinstance(value, pd.DataFrame) and key not in ("gene_identity", "gene_filter_report", "gene_target_resolution"):
            result[key] = _present_gene_frame(value, gene_identity)
    if "cluster_results" in results:
        result["cluster_results"] = [_present_gene_results(v, gene_identity) for v in results["cluster_results"]]
    return result


# Canonical CSV names. Managed keys never fall back to intermediate/, including
# aliases and optional tables disabled by configuration.
RESULT_CSV_EXPORTS = {
    **{key: f"{key}.csv" for key in (
        "saturating_gene_metrics", "saturating_tf_metrics", "saturating_response_weights",
        "saturating_support_diagnostics", "saturating_gates", "saturating_response_curves",
        "saturating_endpoints", "saturating_aggregation_diagnostics",
        "saturating_peak_attributions", "saturating_tf_peak_contributions", "l1_floor_diagnostics",
        "atac_changes", "gene_identity", "gene_filter_report", "gene_target_resolution",
        "tf_peak_weights", "tf_peak_lag_routes", "perturbation_results", "perturbation_merged",
        "aggregated_tf_peak_edges", "atac_changes_merged",
    )},
    "saturating_pathways": "perturbation_pathways.csv",
    "pathway_edges": "perturbation_pathways.csv",
    "causal_edges": "causal_peak_gene_edges.csv",
    "upstream_tfs": "peak_upstream_tfs.csv",
    "upstream_tfs_merged": "peak_upstream_tfs_merged.csv",
    "cell_projection": "perturbation_cell_projection.csv",
    "projection_summary": "perturbation_projection_summary.csv",
    "projection_branch_probabilities": "perturbation_projection_branch_probabilities.csv",
    "projection_history": "perturbation_projection_bin_history.csv",
    "projection_bin_metadata": "perturbation_projection_bin_metadata.csv",
    "projection_events": "perturbation_projection_events.csv",
}
DETAIL_EXPORT_KEYS = frozenset({
    "tf_peak_lag_routes",
    "saturating_response_weights", "saturating_support_diagnostics", "saturating_gates",
    "saturating_response_curves", "saturating_peak_attributions",
    "saturating_tf_peak_contributions", "l1_floor_diagnostics",
    "projection_history", "projection_events",
})


def save_results(results: dict, config: dict, *, gene_identity=None):
    """
    保存汇总结果；output.export_details=true 时另存明细，每张表仅存一份。

    结果包括：
    - perturbation_results.csv：核心结果表（每个 KO 的转录组+染色质偏移，net effect per gene）
    - perturbation_pathways.csv：路径记录（saturating 模式为时间事件链接，无独立 TF 归因）
    - causal_peak_gene_edges.csv：因果 peak→gene 边
    - tf_peak_weights.csv：TF→peak 调控权重
    - 各 KO 的详细结果（如网络传播路径）

    Parameters
    ----------
    results : dict
        pipeline 各模块产出的结果字典
    config : dict
        配置字典（用于获取输出路径）
    """
    config = _gene_identity_runtime_config(config)
    export_details = config["output"].get("export_details", False)
    if not isinstance(export_details, bool):
        raise ValueError("output.export_details must be a boolean")
    skipped_details = sorted(DETAIL_EXPORT_KEYS.intersection(results)) if not export_details else []
    # Filter before identity presentation to avoid copying large unused tables.
    # This new dictionary leaves the caller's in-memory results untouched.
    from .result_store import ExportResults
    results = ExportResults(results, excluded=skipped_details)
    if _gene_identity_enabled(config):
        if gene_identity is None:
            raise ValueError("gtf_gene_id output requires the runtime gene_identity index")
        from .kinetics import _resource_file_identity
        if _resource_file_identity(config["input"]["annotation_gtf"])["sha256"] != gene_identity.metadata["reference_sha256"]:
            raise ValueError("output gene identity differs from configured GTF content")
        expected = {"schema": gene_identity.metadata["schema"], "fingerprint": gene_identity.fingerprint,
                    "reference_sha256": gene_identity.metadata["reference_sha256"]}
        metadata = results.get("gene_identity_metadata")
        if metadata is not None and any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError("output results have mismatched gene identity provenance")
        # Per-context frames are already represented by the merged root tables.
        # Keep only their metadata here rather than copying all detail frames a
        # second time just to decorate names that will never be exported.
        def present(key, value):
            if isinstance(value, pd.DataFrame) and key not in ("gene_identity", "gene_filter_report", "gene_target_resolution"):
                return _present_gene_frame(value, gene_identity)
            return value
        results = ExportResults(results, presenter=present)
        # Validate presentation before creating any public output, preserving
        # the old fail-before-write contract. Discard each decorated table
        # immediately instead of retaining a second full set of frames.
        for key in results:
            if key not in ("cluster_results", "projection_inputs"):
                value = results[key]
                del value
    out_dir = Path(config["output"]["dir"])
    from ._wt_aggregation import validate_model_mode
    validate_model_mode(config)
    saturation = config.get("atac_to_rna", {}).get("model_mode") == "saturating_hybrid"
    from .atac_normalization import validate_atac_depth_config
    validate_atac_depth_config(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    if saturation:
        for key in ("saturating_gene_metrics", "saturating_tf_metrics", "saturating_response_weights",
                    "saturating_support_diagnostics", "saturating_gates", "saturating_response_curves",
                    "saturating_endpoints", "saturating_aggregation_diagnostics",
                    "saturating_peak_attributions", "saturating_tf_peak_contributions", "l1_floor_diagnostics"):
            frame = results.get(key)
            if isinstance(frame, pd.DataFrame):
                frame.to_csv(out_dir / f"{key}.csv", index=False)
        evidence = results.get("saturating_context_metadata", [])
        if isinstance(evidence, dict):
            evidence = [evidence]
        (out_dir / "saturating_contexts.json").write_text(
            json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
    for key in ("gene_identity", "gene_filter_report", "gene_target_resolution"):
        if isinstance(results.get(key), pd.DataFrame):
            results[key].to_csv(out_dir / f"{key}.csv", index=False)
    if "gene_identity_metadata" in results:
        (out_dir / "gene_identity_metadata.json").write_text(
            json.dumps(results["gene_identity_metadata"], indent=2, allow_nan=False), encoding="utf-8")
    # Primitive per-context references only: never pickle plans or membership.
    contexts = results.get("cluster_results", [results])
    metadata = [entry["binning_metadata"] for entry in contexts if "binning_metadata" in entry]
    if metadata:
        (out_dir / "binning_contexts.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")

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

    # --- 因果调控边 ---
    if "causal_edges" in results:
        results["causal_edges"].to_csv(
            out_dir / "causal_peak_gene_edges.csv", index=False
        )
        logger.info(f"因果边已保存: {len(results['causal_edges'])} 条")

    # --- TF→peak 权重 ---
    if export_details and isinstance(results.get("tf_peak_lag_routes"), pd.DataFrame):
        results["tf_peak_lag_routes"].to_csv(out_dir / "tf_peak_lag_routes.csv", index=False)
    if "tf_peak_weights" in results:
        weights = results["tf_peak_weights"]
        if export_details and "local_routes" in weights:
            route_rows = []
            for edge in weights.to_dict("records"):
                identity = {k:edge[k] for k in ("cluster", "tf_gene", "tf_gene_id", "tf_gene_name", "peak_id") if k in edge}
                for route in edge.get("local_routes", ()):
                    route_rows.append(dict(identity, **route))
            pd.DataFrame(route_rows).to_csv(out_dir / "tf_peak_lag_routes.csv", index=False)
        weights.drop(columns=["local_routes"], errors="ignore").to_csv(
            out_dir / "tf_peak_weights.csv", index=False
        )

    # --- 扰结果 ---
    if "perturbation_results" in results:
        pert = results["perturbation_results"]
        if not pert.empty and "delta_rna" in pert.columns:
            columns = (["cluster"] if "cluster" in pert else []) + ["delta_rna"]
            pert = pert.sort_values(columns, ascending=([True] if "cluster" in pert else []) + [False])
        pert.to_csv(out_dir / "perturbation_results.csv", index=False)

    # --- 加权平均扰动结果 (跨簇合并) ---
    if "perturbation_merged" in results:
        pm = results["perturbation_merged"]
        if not pm.empty and "delta_rna" in pm.columns:
            pm = pm.sort_values("delta_rna", ascending=False)
        pm.to_csv(out_dir / "perturbation_merged.csv", index=False)
        logger.info(f"加权平均扰动结果已保存: {len(pm)} 条")

    # --- 路径级边记录 (TF→peak→gene, 用于抵消分析与调控链追踪) ---
    if "pathway_edges" in results or (saturation and "saturating_pathways" in results):
        pw = results["saturating_pathways"] if saturation and "saturating_pathways" in results else results["pathway_edges"]
        if not pw.empty or (saturation and "saturating_pathways" in results):
            pw.to_csv(out_dir / "perturbation_pathways.csv", index=False)
            logger.info(f"路径边已保存: {len(pw)} 条")

    # --- 聚合后的 TF→peak 边 (含窗口验证翻转信息) ---
    if "aggregated_tf_peak_edges" in results:
        ae = results["aggregated_tf_peak_edges"]
        if not ae.empty:
            ae.to_csv(out_dir / "aggregated_tf_peak_edges.csv", index=False)
            n_flipped = ae["flipped"].sum() if "flipped" in ae.columns else 0
            logger.info(f"聚合 TF→peak 边已保存: {len(ae)} 条 (其中翻转 {n_flipped} 条)")


    # --- ATAC 变化 (BED 格式，可用 IGV 可视化) ---
    if "atac_changes" in results:
        atac_df = results["atac_changes"]
        atac_df.to_csv(out_dir / RESULT_CSV_EXPORTS["atac_changes"], index=False)
        if not atac_df.empty:
            _save_atac_bed(atac_df, out_dir / "atac_changes.bed")
            # 显著 ATAC 变化: |z_score| >= 阈值 (z-score 由 WT 零分布计算)
            if "is_significant" in atac_df.columns:
                sig_df = atac_df[_significant_mask(atac_df["is_significant"])].copy()
                if not sig_df.empty:
                    _save_atac_bed(sig_df, out_dir / "atac_changes_significant.bed")
                    logger.info(
                        f"显著 ATAC 变化 (|z| 阈值): {len(sig_df)} peaks → "
                        f"atac_changes_significant.bed"
                    )

    # --- ATAC 变化 (跨簇加权平均, 对齐 perturbation_merged 逻辑) ---
    if "atac_changes_merged" in results:
        acm = results["atac_changes_merged"]
        if acm is not None and not acm.empty:
            acm.to_csv(out_dir / "atac_changes_merged.csv", index=False)
            _save_atac_bed(acm, out_dir / "atac_changes_merged.bed")
            if "is_significant" in acm.columns:
                sig_m = acm[_significant_mask(acm["is_significant"])].copy()
                if not sig_m.empty:
                    _save_atac_bed(sig_m, out_dir / "atac_changes_significant_merged.bed")
                    logger.info(
                        f"跨簇加权平均 ATAC 显著变化: {len(sig_m)} peaks → "
                        f"atac_changes_significant_merged.bed"
                    )
            logger.info(f"跨簇加权平均 ATAC 变化已保存: {len(acm)} peaks")

    # --- Peak 扰动输出 (knockout_type=peak 模式) ---
    if "perturbation_results" in results:
        pert_all = results["perturbation_results"]
        if not pert_all.empty and "target_gene" in pert_all.columns:
            peak_res = pert_all[
                pert_all["target_gene"].astype(str).str.startswith("PEAK:")
            ].copy()
            if not peak_res.empty:
                peak_res.to_csv(out_dir / "peak_perturbation_results.csv", index=False)
                logger.info(f"Peak 扰动结果已保存: {len(peak_res)} 条 → peak_perturbation_results.csv")

    if "upstream_tfs" in results and results["upstream_tfs"] is not None \
            and not results["upstream_tfs"].empty:
        results["upstream_tfs"].to_csv(out_dir / "peak_upstream_tfs.csv", index=False)
        logger.info(
            f"上游 TF 归因已保存: {len(results['upstream_tfs'])} 条 → peak_upstream_tfs.csv"
        )

    if "upstream_tfs_merged" in results and results["upstream_tfs_merged"] is not None \
            and not results["upstream_tfs_merged"].empty:
        results["upstream_tfs_merged"].to_csv(
            out_dir / "peak_upstream_tfs_merged.csv", index=False
        )
        logger.info(
            f"上游 TF 跨簇加权合并已保存: {len(results['upstream_tfs_merged'])} 条 "
            f"→ peak_upstream_tfs_merged.csv"
        )

    if results.get("peak_summary") is not None and not results["peak_summary"].empty:
        _write_peak_report(results["peak_summary"], out_dir / "peak_perturbation_report.txt")

    # --- 中间文件 ---
    if config["output"].get("save_intermediate", True):
        inter_dir = out_dir / "intermediate"

        for key, df in results.items():
            if (isinstance(df, pd.DataFrame) and key not in RESULT_CSV_EXPORTS
                    and key not in {"peak_summary", "cell_states", "projection_cell_states"}
                    and not key.endswith("_saved")):
                inter_dir.mkdir(exist_ok=True)
                df.to_csv(inter_dir / f"{key}.csv", index=False)

    if skipped_details:
        logger.info("汇总导出：跳过 %d 类明细 (%s)；需要时设置 output.export_details: true。"
                    "历史／事件／响应曲线诊断图需要这些明细。",
                    len(skipped_details), ", ".join(skipped_details))

    logger.info(f"所有结果已保存至: {out_dir.resolve()}")


def _save_atac_bed(atac_changes: pd.DataFrame, path: Path):
    """将预测的 ATAC 变化输出为标准 BED 格式，可用 IGV 可视化。

    Parameters
    ----------
    atac_changes : DataFrame
        列需包含 chr, start, end, peak_id, delta_accessibility
    path : Path
        输出 .bed 文件路径
    """
    bed_cols = ["chr", "start", "end", "peak_id", "delta_accessibility"]
    if not all(c in atac_changes.columns for c in bed_cols[:3]):
        logger.warning("ATAC 变化数据缺少 BED 列 (chr/start/end)，跳过 BED 导出")
        return

    with open(path, "w") as f:
        f.write('# track name="CausalBridge predicted ATAC changes"\n')
        f.write('# itemRgb="On"\n')
        for _, row in atac_changes.iterrows():
            if not np.isfinite(row.get("delta_accessibility", 0)):
                continue
            score = int(np.clip(abs(row.get("delta_accessibility", 0)) * 1000, 0, 1000))
            f.write(f"{row['chr']}\t{int(row['start'])}\t{int(row['end'])}\t"
                    f"{row.get('peak_id', '.')}\t{score}\n")
    logger.info(f"ATAC BED track 已保存: {path}")


def _significant_mask(values):
    """CSV-like strings must not become truthy boolean significance flags."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False)
    normalized = values.map(lambda v: str(v).strip().lower() if pd.notna(v) else "false")
    if not normalized.isin(["true", "false", "1", "0"]).all():
        raise ValueError("is_significant must contain boolean values")
    return normalized.isin(["true", "1"])


def _write_peak_report(peak_summary: pd.DataFrame, path: Path):
    """生成 peak 扰动四段式人读报告 (A 扰动peak / B 上游TF / C 直接效应 / D 传播摘要)."""

    def _i(v):
        return int(v) if v is not None else 0

    lines = [
        "=" * 64,
        "Peak Perturbation Report (CausalBridge v1.40, peak KO mode)",
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
                         f"(详见 peak_upstream_tfs.csv)")
        else:
            lines.append(f"  B. upstream TF candidates : no confident upstream TF identified "
                         f"(motif 数据库不完整 / TF 低表达 / Pearson 检验未通过均可能导致)")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Peak 扰动报告已保存: {path}")
