"""
CausalBridge Granger 因果推断模块

核心原理（参考 GrID-Net, Wu et al., Nature Communications, 2025）：
  利用 ATAC 和 RNA 之间的时间滞后效应推断因果调控关系。
  染色质可及性变化发生在转录变化之前——同一细胞中 ATAC 信号捕获的是
  "较早时间点"的状态，而 RNA 信号捕获的是"较晚时间点"的状态。
  这种"细胞状态视差"(cell-state parallax) 使得 Granger 因果检验成为可能。

Granger 因果检验逻辑：
  - 简约模型 (reduced): 用基因过去的表达预测当前表达  Y(t) ~ Y(t-1)
  - 完整模型 (full):    加入候选 peak 的过去可及性    Y(t) ~ Y(t-1) + X(t-1)
  - F 检验判断: peak 信号是否显著改善预测 → 因果性证据
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
# 伪时间推断
# ============================================================================

def infer_pseudotime(
    rna_adata,
    atac_adata,
    config: dict,
    *,
    defer_binning: bool = False,
    joint_repr=None,
) -> pd.DataFrame:
    """
    从多组学数据推断发育伪时间。

    方法（由 config["pseudotime"]["method"] 控制）：

    - "dpt" (默认): Diffusion Pseudotime via scanpy
      基于扩散距离，对噪声鲁棒，无需额外依赖
    - "palantir": Palantir 伪时间
      基于扩散图 + 熵，天然支持多谱系分岔，需 pip install palantir
    当所选方法不可用时，自动回退到下一级方法（Palantir → DPT → fallback）。

    Returns
    -------
    pseudotime_df : DataFrame
        列: cell_barcode, pseudotime, branch (如有分岔)
    """
    cfg = config["pseudotime"]
    if ("gene_identity" in rna_adata.uns
            or config.get("input", {}).get("gene_identity_mode") == "gtf_gene_id"):
        marker = cfg.get("root_cell_marker")
        if marker and marker not in rna_adata.var_names:
            raise ValueError(f"root_cell_marker {marker!r} must resolve to a retained RNA gene ID")
    method = cfg.get("method", "dpt")

    # --- 步骤1: 构造联合低维表示 ---
    rna_adata_copy = rna_adata.copy()
    if joint_repr is None:
        joint_repr, _ = _build_joint_representation(rna_adata_copy, atac_adata)
    else:
        if (not rna_adata.obs_names.is_unique or not atac_adata.obs_names.is_unique
                or rna_adata.obs_names.hasnans or atac_adata.obs_names.hasnans
                or not rna_adata.obs_names.equals(atac_adata.obs_names)):
            raise ValueError("joint_repr requires unique RNA/ATAC cell IDs in exact order")
        if isinstance(joint_repr, pd.DataFrame) and not joint_repr.index.equals(rna_adata.obs_names):
            raise ValueError("joint_repr cell IDs must match RNA/ATAC exactly in order")
        joint_repr = np.asarray(joint_repr, dtype=float)
        if (joint_repr.ndim != 2 or joint_repr.shape[0] != rna_adata.n_obs
                or joint_repr.shape[0] != atac_adata.n_obs or joint_repr.shape[1] == 0
                or not np.isfinite(joint_repr).all()):
            raise ValueError("joint_repr must be finite with one row per RNA/ATAC cell")
    rna_adata_copy.obsm["X_joint"] = joint_repr

    # --- 步骤2: 按配置方法推断伪时间（带自动回退） ---
    pseudotime, branch_labels = _dispatch_pseudotime_method(
        method, rna_adata_copy, joint_repr, cfg, config
    )

    # --- 步骤3: 伪时间分箱（统一入口，下游只读不写） ---
    # 合并过小的 branch（细胞数 < min_cells_per_branch）
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
                    f"Branch {br}: 仅 {(branch_labels == br).sum()} 细胞 "
                    f"(<{min_cells_for_branch})，合并到 branch {nearest}"
                )

    if defer_binning:
        results = pd.DataFrame({"pseudotime": pseudotime}, index=rna_adata_copy.obs_names)
        results.index.name = "cell_barcode"
        if branch_labels is not None:
            results["branch"] = branch_labels
        results.attrs["branch_boundaries"] = None
        return results

    # 构造临时 DataFrame 供 _adaptive_rebin 使用
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
    # branch_boundaries 由 _adaptive_rebin 写入 temp_df.attrs，转移到 results
    results.attrs["branch_boundaries"] = temp_df.attrs.get("branch_boundaries")

    bin_counts = results["bin"].value_counts().sort_index()
    logger.info(
        f"伪时间推断完成 ({method}): "
        f"伪时间范围 [{pseudotime.min():.3f}, {pseudotime.max():.3f}], "
        f"{total_bins} bins, "
        f"每 bin 细胞数: min={bin_counts.min()}, median={bin_counts.median():.0f}, "
        f"max={bin_counts.max()}"
    )
    return results


def _dispatch_pseudotime_method(method, rna_adata, joint_repr, cfg, config):
    """按 method 配置调度伪时间推断，失败时自动回退到下一级。"""
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
            logger.warning("Palantir 不可用，回退到下一级方法")
        elif m == "dpt":
            if _scanpy_available():
                return _pseudotime_via_dpt(rna_adata, joint_repr, cfg, config)
            logger.warning("scanpy 不可用，回退到下一级方法")
        elif m == "fallback":
            logger.warning("使用自实现扩散伪时间（fallback）")
            return _diffusion_pseudotime_fallback(joint_repr, rna_adata, cfg, config)

    return _diffusion_pseudotime_fallback(joint_repr, rna_adata, cfg, config)


def _scanpy_available() -> bool:
    """检测 scanpy 是否可用。"""
    try:
        import scanpy  # noqa
        return True
    except ImportError:
        return False


def _palantir_available() -> bool:
    """检测 Palantir 是否可用。"""
    try:
        import palantir  # noqa
        return True
    except ImportError:
        return False


def _build_joint_representation(rna_adata, atac_adata):
    """
    构造 ATAC+RNA 联合低维表示。

    使用 WNN 风格策略: 对每个模态独立降维后拼接。
    相比简单拼接 PCA 的改进:
    - 使用 scanpy.pp.pca 而非 sklearn PCA（更好的稀疏矩阵支持）
    - 可选地加权两个模态（通过 explained_variance_ratio 归一化）
    """
    rna_mat = rna_adata.X.toarray() if issparse(rna_adata.X) else np.array(rna_adata.X)
    atac_mat = (
        atac_adata.X.toarray() if issparse(atac_adata.X) else np.array(atac_adata.X)
    )

    # scanpy PCA（当可用时使用更高效和规范的实现）
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
        # 按 explained variance 加权，使两种模态的贡献均衡
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
    logger.info(f"联合低维表示: {joint_repr.shape}")

    return joint_repr, rna_adata


def _pseudotime_via_dpt(rna_adata_with_repr, joint_repr, cfg, config):
    """
    Diffusion Pseudotime (DPT) via scanpy。

    DPT 原理（Haghverdi et al., Nature Methods, 2016）：
    1. 在 KNN 图上定义扩散过程（随机游走）
    2. 计算任意两点间的扩散距离——考虑所有可能路径，不是仅最短路径
    3. 从根细胞出发的扩散距离 = 伪时间
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

    logger.info(f"DPT 伪时间推断完成 (scanpy), 根细胞 #{root_idx}")
    return pseudotime, None


def _pseudotime_via_palantir(rna_adata_with_repr, joint_repr, cfg, config):
    """
    Palantir 伪时间推断。

    Palantir 原理（Setty et al., Nature Biotechnology, 2019）：
    1. 扩散图降维 → 捕捉发育轨迹的全局结构
    2. 从根细胞出发的最短路径 → 初始伪时间
    3. 熵计算 → 分化潜能（entropy 低的细胞处于分化末端）
    4. 支持多谱系分岔检测

    需要 pip install palantir
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
            f"Palantir 检测到 {pr_res.branch_probs.shape[1]} 个分支, "
            f"branch_probs shape={pr_res.branch_probs.shape}"
        )
    else:
        logger.info(
            "Palantir 未检测到分支 (branch_probs 为空/None)，"
            "将使用全局伪时间分箱"
        )

    logger.info(f"Palantir 伪时间推断完成, 根细胞 #{root_idx}")
    return pseudotime, branch_labels


def _diffusion_pseudotime_fallback(joint_repr, rna_adata, cfg, config):
    """
    自实现扩散伪时间（scanpy 不可用时的 fallback）。

    简化版扩散伪时间:
    1. 构造相互 KNN 图（mutual nearest neighbors）以减少噪声边
    2. 使用扩散过程: T = D^{-1/2} · A_sym · D^{-1/2}
    3. 谱分解后取第二特征向量（扩散图的第一非平凡分量）
    4. 以根细胞锚定方向

    这比初版的 Dijkstra 最短路径有本质改进:
    - 最短路径对单一边的权重极度敏感
    - 扩散映射考虑了所有路径，自然平滑
    """
    from sklearn.neighbors import NearestNeighbors

    n_cells = joint_repr.shape[0]
    n_neighbors = min(cfg["n_neighbors"], n_cells - 1)

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(joint_repr)
    distances, indices = nn.kneighbors(joint_repr)

    # 构造互 KNN 图（mutual nearest neighbors）— 边仅在双向为近邻时存在
    # 这比全 KNN 图更稀疏、更可靠
    rows, cols, data = [], [], []
    for i in range(n_cells):
        for j_idx, d in zip(indices[i], distances[i]):
            if i in indices[j_idx]:  # 互近邻检查
                rows.append(i)
                cols.append(j_idx)
                weight = np.exp(-d ** 2 / (np.median(distances) ** 2 + 1e-10))
                data.append(weight)

    if len(data) == 0:
        # 如果互 KNN 太严格导致没有边，退化为全 KNN
        logger.warning("互 KNN 图为空，退化为全 KNN 图")
        for i in range(n_cells):
            for j_idx, d in zip(indices[i], distances[i]):
                rows.append(i)
                cols.append(j_idx)
                data.append(np.exp(-d / (np.median(distances) + 1e-10)))

    adj = csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
    adj = (adj + adj.T) / 2  # 对称化

    # 扩散映射: 取第二特征向量（第一非平凡分量）
    from scipy.sparse.linalg import eigsh

    degree = np.array(adj.sum(axis=1)).flatten()
    degree_sqrt_inv = np.diag(1.0 / np.sqrt(degree + 1e-10))
    laplacian = degree_sqrt_inv @ adj @ degree_sqrt_inv

    try:
        _, eigenvectors = eigsh(laplacian, k=3, which="LM")
        diffusion_component = eigenvectors[:, 1]  # 第二特征向量
    except Exception:
        # fallback: 简单谱分解
        from scipy.sparse.csgraph import dijkstra
        root_idx = _find_root_cell(
        rna_adata, cfg.get("root_cell_marker"),
        method=cfg.get("root_cell_method"),
        n_top_genes=cfg.get("cytotrace_n_top_genes", 200),
    )
        diffusion_component = dijkstra(adj, directed=False, indices=root_idx)

    # 确定方向: 使根细胞端为 0
    root_idx = _find_root_cell(
        rna_adata, cfg.get("root_cell_marker"),
        method=cfg.get("root_cell_method"),
        n_top_genes=cfg.get("cytotrace_n_top_genes", 200),
    )
    if diffusion_component[root_idx] > np.median(diffusion_component):
        diffusion_component = -diffusion_component
    pseudotime = diffusion_component - diffusion_component[root_idx]
    pseudotime = np.maximum(pseudotime, 0)  # 非负约束

    branch_labels = None
    logger.info(f"自实现扩散伪时间推断完成, 根细胞 #{root_idx}")
    return pseudotime, branch_labels


def _compute_cytotrace_scores(
    rna_adata,
    n_top_genes: int = 200,
) -> np.ndarray:
    """Read verified official per-cohort scores; n_top_genes is legacy API only."""
    from .cytotrace import verified_scores
    return verified_scores(rna_adata)


def _find_root_cell(
    rna_adata,
    marker_gene: Optional[str] = None,
    method: Optional[str] = None,
    n_top_genes: int = 200,
    *, gene_identity=None,
) -> int:
    """
    确定发育轨迹的根细胞（起点）。由 ``method`` 控制根选取策略。

    method (None / "auto" / "marker" / "cytotrace" / "min_umi"):
      - "marker":    强制使用 marker_gene；marker 缺失则回退 min_umi
      - "cytotrace": 使用 CytoTRACE 2020 干性分数, 选簇内 stemness 最高的细胞
                      (仅接受官方算法按本模型细胞队列计算的完整原始计数分数)
      - "min_umi":   转录多样性最低的细胞 (DPT 论文标准做法)
      - None / "auto": marker_gene 给定 → marker; 否则默认 cytotrace (新默认)
    """
    if marker_gene and (gene_identity is not None or "gene_identity" in rna_adata.uns):
        if gene_identity is not None:
            from .gene_identity import resolve_gene_token
            marker_gene = resolve_gene_token(marker_gene, gene_identity,
                                            allowed_ids=rna_adata.var_names, purpose="root marker")
        elif marker_gene not in rna_adata.var_names:
            raise ValueError(f"root marker {marker_gene!r} must resolve to a retained RNA gene ID")
    valid = {"marker", "cytotrace", "min_umi", "auto", None}
    if method not in valid:
        logger.warning(f"未知的 root_cell_method='{method}'，回退 auto")
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
            logger.info(f"根细胞由标记基因选定: {marker_gene} → 细胞 #{root}")
            return root
        logger.info(f"marker 方法指定但 marker='{marker_gene}' 缺失, 回退 min_umi")
        use = "min_umi"

    # --- (2) cytotrace ---
    if use == "cytotrace":
        scores = _compute_cytotrace_scores(rna_adata, n_top_genes=n_top_genes)
        root = int(np.argmax(scores))
        logger.info(
            f"根细胞由 CytoTRACE on-the-fly 计算: 细胞 #{root} "
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
        logger.info(f"根细胞由转录活性选定: 细胞 #{root}")
        return root
    return 0


# ============================================================================
# Granger 因果检验
# ============================================================================

def granger_test(
    rna_adata,
    atac_adata,
    pseudotime_df: pd.DataFrame,
    config: dict,
    *,
    bin_plan=None,
) -> pd.DataFrame:
    """
    对每个候选 peak-gene 对进行 Granger 因果检验。

    候选对生成规则：
    - peak 和 gene 的 TSS 在基因组上的距离 < distance_thresh（默认 1Mb）
    - 两者在伪时间上有足够变化（排除恒定信号）

    对每个候选对：
    1. 在 binned 伪时间上构造 time-lagged 特征:
       - Y(t) = 基因在 bin t 的平均表达
       - Y(t-1) = 基因在 bin t-1 的平均表达
       - X(t-1) = peak 在 bin t-1 的平均可及性
    2. 拟合两个线性模型:
       - Reduced:  Y(t) ~ Y(t-1)
       - Full:     Y(t) ~ Y(t-1) + X(t-1)
    3. F 检验: ATAC 滞后项是否显著改善预测 (F(df=1, n-3))
    4. 多重检验校正 (FDR)

    Returns
    -------
    results : DataFrame
        列: peak_id, gene, peak_chr, peak_start, peak_end, gene_chr, gene_tss,
             F_statistic, p_value, p_adj, delta_r2, is_causal
    """
    cfg = config["granger"]
    pcfg = config["pseudotime"]
    from .binning_adapter import require_bin_plan, validate_bin_plan, execution_pseudotime
    require_bin_plan(config, bin_plan)
    if bin_plan is not None:
        validate_bin_plan(bin_plan, pseudotime_df, rna_adata.obs_names)
        validate_bin_plan(bin_plan, pseudotime_df, atac_adata.obs_names)
        pseudotime_df = execution_pseudotime(bin_plan, pseudotime_df, rna_adata.obs_names)
    ablation_mode = cfg.get("ablation_lag0", False)

    if ablation_mode:
        logger.info("=== Granger 消融模式: 去掉时间滞后, Y(t)~X(t) 纯相关检验 ===")

    # 按簇规模自适应分箱
    _smooth_kernel = pcfg.get("smooth_kernel", "hard")
    _overlap = pcfg.get("smooth_overlap_factor") if _smooth_kernel == "gaussian" else None
    
    # 调试日志：检查伪时间是否被打乱
    if hasattr(pseudotime_df, 'attrs') and pseudotime_df.attrs.get("_need_rebin"):
        logger.info(f"  [DEBUG] granger_test: 检测到 _need_rebin 标记，伪时间已被打乱")
        logger.info(f"  [DEBUG] 伪时间范围: [{pseudotime_df['pseudotime'].min():.4f}, {pseudotime_df['pseudotime'].max():.4f}]")
    
    if bin_plan is None:
        bin_indices, n_bins = _adaptive_rebin(
            pseudotime_df, rna_adata.n_obs,
            max_bins=pcfg["n_bins"],
            min_cells_per_bin=pcfg.get("min_cells_per_bin", 5),
            target_bins=pcfg["n_bins"] if _smooth_kernel == "gaussian" else None,
            overlap_factor=_overlap,
        )
        pseudotime_df["bin"] = bin_indices
    else:
        bin_indices, n_bins = bin_plan.labels, bin_plan.n_bins
    mode_label = "消融" if ablation_mode else "Granger"
    logger.info(f"{mode_label} 分箱: {rna_adata.n_obs} 细胞 → {n_bins} bins "
                f"(smooth={_smooth_kernel})")
    
    # 调试日志：检查重新计算的 bin 分布
    if hasattr(pseudotime_df, 'attrs') and pseudotime_df.attrs.get("_need_rebin"):
        bin_counts = np.bincount(bin_indices.astype(int))
        logger.info(f"  [DEBUG] 重新计算的 bin 分布: {bin_counts[:5]}...")

    # --- 步骤1: 在 binned 伪时间上聚合 ATAC 和 RNA 信号 ---
    # 保存打乱前的分箱结果用于比较
    _need_rebin = bin_plan is None and hasattr(pseudotime_df, 'attrs') and pseudotime_df.attrs.get("_need_rebin")
    if _need_rebin:
        rna_binned_before = _bin_expression(rna_adata, pseudotime_df, n_bins,
                                           smooth="hard",
                                           min_cells_per_bin=pcfg.get("min_cells_per_bin", 5))
    
    rna_binned = _bin_expression(rna_adata, pseudotime_df, n_bins,
                                 **({"bin_plan": bin_plan} if bin_plan is not None else {}),
                                 smooth=pcfg.get("smooth_kernel", "hard"),
                                 min_cells_per_bin=pcfg.get("min_cells_per_bin", 5))
    atac_binned = _bin_expression(atac_adata, pseudotime_df, n_bins,
                                  **({"bin_plan": bin_plan} if bin_plan is not None else {}),
                                  smooth=pcfg.get("smooth_kernel", "hard"),
                                  min_cells_per_bin=pcfg.get("min_cells_per_bin", 5))

    logger.info(f"Binned 数据: RNA {rna_binned.shape}, ATAC {atac_binned.shape}")
    
    # 调试日志：检查分箱结果是否真的变化了
    if _need_rebin:
        rna_diff = np.abs(rna_binned - rna_binned_before).max()
        logger.info(f"  [DEBUG] 打乱前后 RNA 分箱最大差异: {rna_diff:.6f}")
        if rna_diff < 1e-6:
            logger.warning("  ⚠️ 警告: 打乱前后 RNA 分箱结果几乎相同！")
        else:
            logger.info(f"  ✓ 打乱前后 RNA 分箱结果不同，差异显著")

    # --- 滞后预切片 ---
    branch_boundaries = pseudotime_df.attrs.get("branch_boundaries") if hasattr(pseudotime_df, "attrs") else None

    if ablation_mode:
        # 消融模式：使用同时刻数据，不做任何 lag 处理
        # 检验 Y(t) ~ X(t) 纯相关性
        _rna_t = rna_binned
        _rna_t1 = None  # 消融模式不需要 Y(t-1)
        _atac_lag_mats = None
        _atac_current = atac_binned  # X(t): 同时刻 ATAC
        n_valid_pairs = rna_binned.shape[0]
        logger.info(f"消融模式 有效 pairs: {n_valid_pairs} (无滞后, contemporaneous)")
    else:
        # 标准 Granger 模式：使用配置中的 max_lag
        granger_max_lag = cfg.get("max_lag", 1)
        logger.info(f"Granger 模式: lag={granger_max_lag}")
        
        # 检验 Y(t) ~ Y(t-lag) + X(t-lag)
        rna_lag0, rna_lag1 = _branch_aware_lag_pairs(
            rna_binned, rna_binned, lag=granger_max_lag,
            branch_boundaries=branch_boundaries,
        )
        atac_lag0, atac_lag1 = _branch_aware_lag_pairs(
            atac_binned, atac_binned, lag=granger_max_lag,
            branch_boundaries=branch_boundaries,
        )
        _rna_t = rna_lag1      # Y(t): RNA 在时间 t
        _rna_t1 = rna_lag0     # Y(t-lag): RNA 在时间 t-lag
        _atac_lag_mats = [atac_lag0]  # X(t-lag): ATAC 在时间 t-lag
        _atac_current = atac_lag1     # X(t): ATAC 在时间 t (备用)
        n_valid_pairs = _rna_t.shape[0]
        logger.info(f"Granger 滞后切片: {n_valid_pairs} 有效 pairs (lag={granger_max_lag})")

    if n_valid_pairs < 5:
        logger.warning(f"滞后切片仅产生 {n_valid_pairs} 个有效 pair，"
                       f"Granger 检验无法执行")
        return pd.DataFrame()

    # --- 步骤2: 生成候选 peak-gene 对 ---
    candidates = _generate_candidate_pairs(
        rna_adata, atac_adata, cfg["distance_thresh"]
    )
    logger.info(f"候选 peak-gene 对: {len(candidates)}")

    if len(candidates) == 0:
        logger.warning("没有找到任何候选 peak-gene 对，请检查基因组坐标注释")
        return pd.DataFrame()

    # --- 步骤3: 逐对 Granger 检验 ---
    # 预建索引映射，避免每次 O(n) 查找
    atac_name_to_idx = {name: i for i, name in enumerate(atac_adata.var_names)}
    rna_name_to_idx = {name: i for i, name in enumerate(rna_adata.var_names)}
    # 预提基因组位置信息为 numpy 数组，避免重复 loc 索引
    atac_chr_arr = atac_adata.var["chr"].values
    atac_start_arr = atac_adata.var["start"].values.astype(int)
    atac_end_arr = atac_adata.var["end"].values.astype(int)
    rna_chr_arr = rna_adata.var["chr"].values
    from .gene_identity import gene_tss_coordinates
    rna_tss_arr = gene_tss_coordinates(rna_adata)

    # 候选对分块检验；块内复用同一基因的自回归 SSR。
    from .screening_parallel import granger_candidates, screening_workers
    results = granger_candidates(
        candidates, dict(
            atac_name_to_idx=atac_name_to_idx, rna_name_to_idx=rna_name_to_idx,
            atac_chr_arr=atac_chr_arr, atac_start_arr=atac_start_arr,
            atac_end_arr=atac_end_arr, rna_chr_arr=rna_chr_arr,
            rna_tss_arr=rna_tss_arr, _rna_t=_rna_t, _rna_t1=_rna_t1,
            _atac_lag_mats=_atac_lag_mats, _atac_current=_atac_current,
            n_valid_pairs=n_valid_pairs, ablation_mode=ablation_mode),
        workers=screening_workers(config, "granger_workers"))

    if len(results) == 0:
        mode_label = "消融" if ablation_mode else "Granger"
        logger.warning(f"{mode_label} 检验未产生任何有效结果")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # --- 步骤4: 多重检验校正 (FDR) ---
    _, p_adj, _, _ = multipletests(
        results_df["p_value"].values,
        method=cfg["correction_method"],
    )
    results_df["p_adj"] = p_adj

    # --- 步骤5: 因果性判定 ---
    sig_threshold = cfg["significance_threshold"]
    min_effect = cfg["min_effect_size"]

    # 自由度校正: 少 bin → 少观测 → ΔR² 虚高 (~6x @23bin)
    # correction = (df_ref / df_actual)², 二次惩罚对少 bin 簇更严格
    n_ref_bins = cfg.get("reference_bins", 50)
    df_ref = n_ref_bins - 3
    df_actual = max(n_valid_pairs - 3, 1)
    correction = (df_ref / df_actual) ** 2
    corrected_sig = sig_threshold / correction
    corrected_effect = min_effect * correction
    logger.info(
        f"Granger 自由度校正: n_bins={n_bins}, "
        f"df_actual={df_actual}, "
        f"correction={correction:.3f}, p_threshold={sig_threshold:.4f}→{corrected_sig:.4f}, "
        f"effect_threshold={min_effect:.3f}→{corrected_effect:.3f}"
    )

    # 复合分数 (始终计算，输出到结果表)
    # p_score: p_adj 越小 → 越接近 1
    p_score = np.clip(1.0 - results_df["p_adj"].values / corrected_sig, 0, 1)
    # effect_score: delta_r2 越大 → 越接近 1
    effect_score = np.clip(
        results_df["delta_r2"].values / max(corrected_effect, 1e-10), 0, 2
    ) / 2.0
    w = cfg.get("composite_weights", [0.4, 0.6])
    results_df["composite_score"] = p_score * w[0] + effect_score * w[1]

    if cfg.get("soft_threshold", False):
        min_score = cfg.get("min_composite_score", 0.3)
        results_df["is_causal"] = results_df["composite_score"] >= min_score
        logger.info(
            f"使用软阈值过滤 (min_composite_score={min_score})"
        )
    else:
        results_df["is_causal"] = (
            (results_df["p_adj"] < corrected_sig)
            & (results_df["delta_r2"] > corrected_effect)
        )

    n_causal = results_df["is_causal"].sum()
    logger.info(
        f"Granger 检验完成: {n_causal}/{len(results_df)} 对通过因果筛选 "
        f"(p_adj < {corrected_sig:.2e}, "
        f"Delta R2 > {corrected_effect:.3f}, "
        f"df_correction={correction:.2f})"
    )

    return results_df


def _bin_expression(adata, pseudotime_df, n_bins: int,
                    smooth: str = "hard", min_cells_per_bin: int = 15,
                    *, bin_plan=None) -> np.ndarray:
    """将每个基因/peak 的表达/可及性在伪时间 bin 上聚合。

    smooth="hard": 硬分箱，bin 内细胞取均值（原始行为）
    smooth="gaussian": 高斯核加权平滑，每个 bin 共享邻 bin 信息，
                       降低小簇 bin 均值噪声，min_cells_per_bin 控制 sigma
    """
    if bin_plan is not None:
        from .binning_adapter import aggregate_with_plan, validate_bin_plan
        if n_bins != bin_plan.n_bins:
            raise ValueError("n_bins must equal the supplied bin_plan.n_bins")
        validate_bin_plan(bin_plan, pseudotime_df, adata.obs_names)
        result = aggregate_with_plan(bin_plan, adata.X, cell_ids=adata.obs_names)
        return result.toarray() if issparse(result) else np.asarray(result)
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

    # 自适应 sigma: 小簇增大 sigma 以从邻 bin 借信息
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
    基于基因组距离生成候选 peak-gene 对。

    优化: 使用 np.searchsorted 在排序后的基因 TSS 上做区间搜索，
    避免构造全量 (n_peaks × n_genes) 距离矩阵。
    """
    pairs = []
    atac_chr = atac_adata.var["chr"].values
    atac_mid = ((atac_adata.var["start"] + atac_adata.var["end"]) / 2).values
    rna_chr = rna_adata.var["chr"].values
    from .gene_identity import gene_tss_coordinates
    rna_tss = gene_tss_coordinates(rna_adata)

    for chrom in np.unique(atac_chr):
        atac_mask = atac_chr == chrom
        rna_mask = rna_chr == chrom
        if not atac_mask.any() or not rna_mask.any():
            continue

        atac_mids = atac_mid[atac_mask]
        atac_names = atac_adata.var_names[atac_mask]
        rna_tsss = rna_tss[rna_mask]
        rna_names = rna_adata.var_names[rna_mask]

        # 按 TSS 排序便于区间搜索
        sort_idx = np.argsort(rna_tsss)
        sorted_tss = rna_tsss[sort_idx]
        sorted_names = rna_names[sort_idx]

        for pi, peak_mid in enumerate(atac_mids):
            # 用 searchsorted 找到 [mid - thresh, mid + thresh] 区间内的基因
            left = np.searchsorted(sorted_tss, peak_mid - distance_thresh)
            right = np.searchsorted(sorted_tss, peak_mid + distance_thresh)
            for gi in range(left, right):
                pairs.append((atac_names[pi], sorted_names[gi]))

    return pairs


# ============================================================================
# 因果 GRN 构建
# ============================================================================

def build_causal_grn(granger_results: pd.DataFrame) -> pd.DataFrame:
    """
    从 Granger 检验结果提取因果 peak-gene 边，作为 GRN 的基础层。

    返回的因果边将在 kinetics 模块中被扩展为完整的 TF→peak→gene 网络。
    """
    causal_edges = granger_results[granger_results["is_causal"]].copy()
    logger.info(f"基于 {len(causal_edges)} 条因果 peak-gene 边构建 GRN 基础层")
    return causal_edges
