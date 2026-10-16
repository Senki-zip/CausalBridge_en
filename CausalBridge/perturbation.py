"""Public intervention entry points and current output helpers."""
import pandas as pd
from typing import Dict, List, Optional


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
    bin_plan=None,
    *,
    tf_model_registry=None,
    cluster_label="",
) -> dict:
    """
    批量模拟多个基因敲除，返回汇总结果表。

    扰动模式: bin_forward — 沿伪时间轴逐bin前向模拟，尊重 TF 特异性时间滞后

    Returns
    -------
    dict
        "perturbation_results": DataFrame (net effect per gene)
        "pathway_edges": DataFrame (per-edge TF→peak→gene contributions)
    """
    from .atac_normalization import validate_atac_depth_config
    validate_atac_depth_config(config)
    from ._wt_aggregation import validate_model_mode
    validate_model_mode(config)
    from .saturating_perturbation import propagate_saturating
    from .saturating_transfer import validate_registry
    from .atac_normalization import expected_registry_schema
    validate_registry(transfer_models, expected_type="saturating_hybrid_peak_gene",
                      expected_schema=expected_registry_schema("saturating_hybrid_peak_gene",config))
    validate_registry(tf_model_registry, expected_type="saturating_hybrid_tf_peak",
                      expected_schema=expected_registry_schema("saturating_hybrid_tf_peak",config))
    return propagate_saturating(target_genes, rna_adata, atac_adata, pseudotime_df,
        granger_edges, transfer_functions, tf_peak_weights, config,
        transfer_models=transfer_models, tf_model_registry=tf_model_registry,
        gene_types=gene_types, bin_plan=bin_plan,
        cluster_label=cluster_label)


def _build_atac_changes_df(delta_atac_list, atac_adata,
                           atac_z_score=None, z_threshold: float = 2.0) -> pd.DataFrame:
    """将 per-peak delta_atac Series 列表转换为 BED 就绪的 DataFrame。

    Parameters
    ----------
    delta_atac_list : list of pd.Series or pd.Series
        单个或多个 per-peak 变化 Series
    atac_adata : AnnData
        用于获取 peak 的 chr/start/end 坐标
    atac_z_score : pd.Series, optional
        per-peak WT 零分布 z-score (delta / WT bin 间 std)
    z_threshold : float
        |z| >= z_threshold 判定为显著变化

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
        # 多个 KO 结果: 取每个 peak 绝对值最大的变化
        all_peaks = set()
        for ds in delta_atac_list:
            all_peaks.update(ds.index)
        delta_dict = {}
        for peak_id in all_peaks:
            values = [ds.get(peak_id, 0.0) for ds in delta_atac_list]
            delta_dict[peak_id] = max(values, key=abs)
        delta_series = pd.Series(delta_dict)

    # 过滤掉零变化
    delta_series = delta_series[delta_series.abs() > 1e-10]
    if len(delta_series) == 0:
        return pd.DataFrame(columns=["chr", "start", "end", "peak_id", "delta_accessibility"])

    var_df = atac_adata.var[["chr", "start", "end"]]
    result = var_df.loc[var_df.index.isin(delta_series.index)].copy()
    result["peak_id"] = result.index
    result["delta_accessibility"] = result["peak_id"].map(delta_series)

    # 附上 WT 零分布 z-score (若有)
    if atac_z_score is not None:
        result["z_score"] = result["peak_id"].map(atac_z_score).fillna(0.0)
        result["is_significant"] = result["z_score"].abs() >= z_threshold
    return result.reset_index(drop=True)








# ============================================================================
# Peak 扰动 (peak KO): 用户指定 peak + Δ开放度, 沿现有 TF→Peak→Gene 网络下游传播
#
# 设计原则 (对齐《Peak perturbation 功能开发须知》):
#   - 不是新模型, 只是现有 bin_forward 引擎的第二个干预入口
#   - Downstream propagation = 扰动模拟; Upstream attribution = 归因解释
#   - 上游 TF 只做 attribution, 不参与反向因果传播 (引擎 8a 步只读 TF RNA
#     delta, 天然保证此边界)
#   - direct 模式 (depth=1): 只输出该 peak 的直接靶基因, 无 8a 级联
#   - ΔR_g = f(A'_p) - f(A_p) 复用训练的 NN 传递函数, 不新增回归
# ============================================================================


def _match_peak(requested_id: str, atac_adata, mode: str = "exact"):
    """peak 匹配: exact → overlap → (nearest) → 失败.

    Parameters
    ----------
    requested_id : str
        用户输入, 形如 "chr3:123400-123900"
    atac_adata : AnnData
        var_names = peak_id, .var 需含 chr/start/end 坐标
    mode : str
        "exact" (默认) | "overlap" | "nearest" — 是否允许逐级回退

    Returns
    -------
    (matched_peak_id | None, info_dict)
    info_dict: {"method": "exact"/"overlap"/"nearest"/None, "distance": int}
    """
    import re
    # 输入规范化: 去千分位逗号 + 统一 unicode 连字符 (–/—/‐/‑/‒/―) 为 ASCII '-'
    # 论文/UCSC 复制的坐标常带这些格式 (如 "chr3:34,691,323–34,693,322")
    requested_id = requested_id.replace(",", "").strip()
    for _dash in ("–", "—", "‐", "‑", "‒", "―"):
        requested_id = requested_id.replace(_dash, "-")

    var_names = list(atac_adata.var_names)

    # 1. exact 坐标匹配
    if requested_id in var_names:
        return requested_id, {"method": "exact", "distance": 0}

    m = re.match(r"^(\w+):(\d+)-(\d+)$", requested_id)
    if m is None:
        return None, {"method": None, "distance": None}

    chrom, start, end = m.group(1), int(m.group(2)), int(m.group(3))
    var_df = atac_adata.var
    has_coords = all(c in var_df.columns for c in ("chr", "start", "end"))

    # 2. overlap 区间相交 (取重叠长度最大者)
    if has_coords:
        same_chr = var_df[var_df["chr"] == chrom]
        if not same_chr.empty:
            overlap = same_chr[(same_chr["start"] < end) & (same_chr["end"] > start)]
            if not overlap.empty:
                o_len = overlap["end"].clip(upper=end) - overlap["start"].clip(lower=start)
                best = o_len.idxmax()
                return best, {"method": "overlap", "distance": 0}

    # 3. nearest 最近 peak (仅显式启用时; 必须报告距离, 不静默替换)
    if mode == "nearest" and has_coords:
        same_chr = var_df[var_df["chr"] == chrom]
        if not same_chr.empty:
            center = (start + end) / 2
            cand_center = (same_chr["start"] + same_chr["end"]) / 2
            dist = (cand_center - center).abs()
            best = dist.idxmin()
            return best, {"method": "nearest", "distance": int(dist.loc[best])}

    # 4. 未找到
    return None, {"method": None, "distance": None}




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
    cluster_label: str = "",
    bin_plan=None,
    *,
    tf_model_registry=None,
) -> Dict:
    """peak KO 公共入口: 匹配 → direct 传播 → 打包为与 gene KO 同构的结果.

    Returns
    -------
    dict
        "perturbation_results": DataFrame (affected_gene × delta_rna)
        "atac_changes": BED-ready DataFrame (仅扰动 peak)
        "upstream_tfs": DataFrame
        "pathway_edges": DataFrame
    """
    from .atac_normalization import validate_atac_depth_config
    validate_atac_depth_config(config)
    from ._wt_aggregation import validate_model_mode
    validate_model_mode(config)
    from .saturating_perturbation import perturb_peak_saturating
    from .saturating_transfer import validate_registry
    from .atac_normalization import expected_registry_schema
    validate_registry(transfer_models, expected_type="saturating_hybrid_peak_gene",
                      expected_schema=expected_registry_schema("saturating_hybrid_peak_gene",config))
    validate_registry(tf_model_registry, expected_type="saturating_hybrid_tf_peak",
                      expected_schema=expected_registry_schema("saturating_hybrid_tf_peak",config))
    return perturb_peak_saturating(peak_id, rna_adata, atac_adata, pseudotime_df,
        granger_edges, transfer_functions, tf_peak_weights, config,
        transfer_models=transfer_models, tf_model_registry=tf_model_registry,
        bin_plan=bin_plan, cluster_label=cluster_label)
