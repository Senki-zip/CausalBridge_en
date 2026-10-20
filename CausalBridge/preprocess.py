"""
CausalBridge 预处理模块

对 RNA 和 ATAC AnnData 进行质量控制、归一化、高变基因筛选。
所有过滤步骤保留原始计数在 .layers["raw"] 中，供下游动力学建模使用。
"""

import numpy as np
from scipy import sparse
import scanpy as sc
import anndata as ad
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def preprocess_rna(
    adata: ad.AnnData,
    min_cells_per_gene: int = 50,
    min_umi_per_cell: int = 500,
    n_highly_variable_genes: int = 3000,
    keep_genes: Optional[list] = None,
    *,
    filter_cells: bool = True,
    retain_low_count_targets: bool = False,
) -> ad.AnnData:
    """
    RNA 数据质量控制和归一化。

    处理流程：
    1. 保留原始计数到 .layers["raw"] — 后续动力学建模需要原始值
    2. 过滤低表达基因和低质量细胞
    3. 文库大小归一化 → log1p 变换
    4. 筛选高变基因 — 减少维度，加速下游 Granger 检验和回归
    5. 保留 keep_genes 中的基因（如 KO 目标基因），即使它们不在高变基因中

    Parameters
    ----------
    adata : AnnData
        原始 RNA 计数数据，.X 为稀疏矩阵（细胞×基因）
    min_cells_per_gene : int
        基因至少在 N 个细胞中检出才保留
    min_umi_per_cell : int
        细胞至少需 N 个 UMI 才保留
    n_highly_variable_genes : int
        高变基因保留数量
    keep_genes : list or None
        必须保留的基因列表（如敲除目标基因），即使不在高变基因筛选中
    filter_cells : bool
        是否执行细胞 QC；模型视图使用参考聚类的细胞集合时设为 False。
    retain_low_count_targets : bool
        模型视图保留有非零计数但未通过基因 QC 的指定靶点；
        不改变其他基因的归一化分母和 HVG 选择，参考视图默认关闭。

    Returns
    -------
    adata : AnnData
        预处理后的数据，.layers["raw"] 为原始计数
    """
    adata = adata.copy()

    # --- 步骤1: 保留原始计数 ---
    # 后续动力学参数拟合需要未归一化的原始值
    adata.layers["raw"] = adata.X.copy()

    # --- 步骤2: 基因/细胞质控 ---
    logger.info(f"QC 前: {adata.n_obs} 细胞, {adata.n_vars} 基因")
    rescued = []
    if retain_low_count_targets and keep_genes:
        detected = np.asarray((adata.X > 0).sum(axis=0)).ravel()
        requested = adata.var_names.isin(keep_genes)
        rescued_mask = requested & (detected > 0) & (detected < min_cells_per_gene)
        rescued = adata.var_names[rescued_mask].tolist()
        for gene in rescued:
            logger.warning("强制保留低检出靶基因 %s: %d/%d 细胞检出，min_cells_per_gene=%d；不参与 HVG 排名",
                           gene, detected[adata.var_names.get_loc(gene)], adata.n_obs, min_cells_per_gene)
        adata.var['n_cells'] = detected
        adata = adata[:, (detected >= min_cells_per_gene) | rescued_mask].copy()
    else:
        sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    if filter_cells:
        sc.pp.filter_cells(adata, min_counts=min_umi_per_cell)
    logger.info(f"QC 后: {adata.n_obs} 细胞, {adata.n_vars} 基因")

    if adata.n_obs < 100 or adata.n_vars < 500:
        raise ValueError(
            f"QC 后数据量过小 ({adata.n_obs} 细胞, {adata.n_vars} 基因)，"
            f"请放宽过滤阈值或检查输入数据质量。"
        )

    # --- 步骤3: 归一化 + log 变换（用于聚类和可视化）---
    if rescued:
        # Preserve the original QC-passing library denominator. A rare target
        # must not change normalization or HVG ranking of the other genes.
        qc_columns = ~adata.var_names.isin(rescued)
        totals = np.asarray(adata.X[:, qc_columns].sum(axis=1)).ravel()
        factors = np.divide(1e4, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
        if (totals <= 0).any():
            raise ValueError("cannot normalize rescued targets: cells have zero QC-passing RNA depth")
        adata.X = (adata.X.multiply(factors[:, None]).tocsr() if sparse.issparse(adata.X)
                   else adata.X * factors[:, None])
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # --- 步骤4: 高变基因筛选 ---
    # 高变基因捕获了数据中最主要的生物学变异信号，同时显著减少计算量
    # seurat_v3 需要原始计数，因此临时将 X 替换为 raw
    normalized_X = adata.X
    adata.X = adata.layers["raw"]
    if rescued:
        hvg_view = adata[:, ~adata.var_names.isin(rescued)].copy()
        sc.pp.highly_variable_genes(hvg_view, n_top_genes=n_highly_variable_genes, flavor="seurat_v3")
        for column in hvg_view.var.columns.difference(adata.var.columns).union(
                ["highly_variable", "highly_variable_rank", "means", "variances", "variances_norm"]):
            if column in hvg_view.var:
                adata.var[column] = hvg_view.var[column].reindex(adata.var_names)
        adata.var["highly_variable"] = adata.var["highly_variable"].fillna(False).astype(bool)
        adata.uns["hvg"] = hvg_view.uns["hvg"].copy()
    else:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_highly_variable_genes, flavor="seurat_v3"
        )
    adata.X = normalized_X

    # --- 步骤5: 确定下游分析保留的基因 ---
    # 下游分析需要 HVG + 用户指定的靶基因。
    # 关键：绝不修改 highly_variable 列本身 —— PCA/聚类依赖它选择特征，
    # 若将靶基因写入 highly_variable=True，更换靶基因会改变 PCA 输入，
    # 导致聚类结果不一致。
    genes_to_keep = adata.var["highly_variable"].copy()
    if keep_genes:
        keep_set = {g for g in keep_genes if g in adata.var_names}
        n_forced = 0
        for g in keep_set:
            if not genes_to_keep[g]:
                genes_to_keep[g] = True
                n_forced += 1
        if n_forced > 0:
            logger.info(
                f"额外保留 {n_forced} 个靶基因（不在高变基因中）用于下游分析"
            )

    adata_out = adata[:, genes_to_keep].copy()
    logger.info(
        f"保留 {adata_out.n_vars} 个基因用于下游分析 "
        f"(HVG={adata_out.var['highly_variable'].sum()}, "
        f"含靶基因补充)"
    )

    return adata_out


def preprocess_atac(
    adata: ad.AnnData,
    min_cells_per_peak: int = 10,
    min_peaks_per_cell: int = 1000,
    binarize: bool = True,
) -> ad.AnnData:
    """
    ATAC 数据质量控制和预处理。

    ATAC 数据极度稀疏（单细胞中 <1% peaks 非零）。处理步骤：
    1. 过滤低覆盖 peak 和低质量细胞
    2. 可选二值化（0/1）— ATAC 信号本质是 binary 的（开放/不开放），
       二值化可以减少 PCR 扩增偏差和测序深度差异的影响
    3. TF-IDF 归一化 — ATAC-seq 数据的标准归一化方法，放大稀有 peak 的权重

    Parameters
    ----------
    adata : AnnData
        原始 ATAC 计数数据
    min_cells_per_peak : int
        peak 至少在 N 个细胞中有信号
    min_peaks_per_cell : int
        细胞至少 N 个 peak
    binarize : bool
        是否将计数矩阵二值化

    Returns
    -------
    adata : AnnData
        预处理后的 ATAC 数据
    """
    from .atac_normalization import META, SOURCE_SCHEMA, validate_full_atac_source, finalize_atac_qc
    prepared_source = adata if adata.uns.get(META, {}).get("schema") == SOURCE_SCHEMA else None
    if prepared_source is not None:
        validate_full_atac_source(prepared_source)
    adata = adata.copy()

    logger.info(f"ATAC QC 前: {adata.n_obs} 细胞, {adata.n_vars} peaks")

    # --- 步骤1: 可选二值化 ---
    if binarize:
        adata.X = (adata.X > 0).astype(np.float32)
        logger.info("ATAC 矩阵已二值化")

    # --- 步骤2: 过滤 ---
    sc.pp.filter_genes(adata, min_cells=min_cells_per_peak)
    sc.pp.filter_cells(adata, min_genes=min_peaks_per_cell)
    logger.info(f"ATAC QC 后: {adata.n_obs} 细胞, {adata.n_vars} peaks")

    if adata.n_vars < 1000:
        logger.warning(
            f"ATAC peak 数量较少 ({adata.n_vars})，可能影响因果检验的统计效力"
        )

    return finalize_atac_qc(adata, prepared_source) if prepared_source is not None else adata


def cluster_cells(
    adata: ad.AnnData,
    n_neighbors: int = 20,
    n_pcs: int = 20,
    resolution: float = 0.8,
    key_added: str = "leiden",
    random_state: int = 42,
) -> ad.AnnData:
    """
    对 RNA 数据执行 PCA + KNN 图构建 + Leiden 聚类。

    聚类标签写入 adata.obs[key_added]，供 pipeline 按簇分别建模。

    Parameters
    ----------
    adata : AnnData
        log1p 归一化后的 RNA 数据
    n_neighbors : int
        KNN 图邻居数
    n_pcs : int
        使用的 PCA 主成分数
    resolution : float
        Leiden 聚类分辨率（越大簇越多）
    key_added : str
        obs 中聚类标签的列名
    random_state : int
        随机种子

    Returns
    -------
    adata : AnnData
        带有聚类标签和 UMAP 坐标的数据
    """
    logger.info(f"PCA → KNN(n={n_neighbors}, pcs={n_pcs}) → Leiden(res={resolution}) → UMAP")

    # 计算 PCA（scanpy 会检查是否已有 PCA，有则跳过）
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=max(50, n_pcs), random_state=random_state)
        logger.info(f"  已计算 PCA: {adata.obsm['X_pca'].shape}")

    # 构建邻域图
    sc.pp.neighbors(
        adata, n_neighbors=n_neighbors, n_pcs=n_pcs,
        random_state=random_state,
    )

    # Leiden 聚类
    sc.tl.leiden(adata, resolution=resolution, key_added=key_added,
                 random_state=random_state)

    # UMAP 降维（复用 neighbors 图）
    sc.tl.umap(adata, random_state=random_state)
    logger.info(f"  已计算 UMAP: {adata.obsm['X_umap'].shape}")

    n_clusters = adata.obs[key_added].nunique()
    cluster_sizes = adata.obs[key_added].value_counts()
    try:
        cluster_sizes = cluster_sizes.reindex(
            sorted(cluster_sizes.index, key=int)
        )
    except ValueError:
        cluster_sizes = cluster_sizes.reindex(
            sorted(cluster_sizes.index, key=str)
        )
    logger.info(f"  聚类完成: {n_clusters} 个簇, "
                f"大小 = {dict(cluster_sizes)}")

    return adata


def build_paga_lineages(
    adata: ad.AnnData,
    cluster_key: str = "leiden",
    connectivity_threshold: float = 0.1,
    min_cells_per_lineage: int = 50,
) -> dict:
    """
    在 Leiden 聚类基础上运行 PAGA，将连通的簇合并为连续谱系 (lineage)。

    目的：Leiden 聚类可能将连续发育谱系切碎成多个小簇。PAGA 的图抽象
    能识别这些簇之间的拓扑连通性，将实际属于同一谱系的簇合并，
    使得伪时间推断在合并后的连续谱系上运行，避免轨迹断裂。

    Parameters
    ----------
    adata : AnnData
        已完成 PCA + neighbors + Leiden 聚类的 RNA 数据。
        需包含 .obsm["X_pca"] 和 .uns["neighbors"]。
    cluster_key : str
        .obs 中 Leiden 聚类标签的列名
    connectivity_threshold : float
        PAGA 连通性阈值。connectivities[i,j] >= threshold 的簇对视为连通。
    min_cells_per_lineage : int
        合并后最少细胞数。低于此值的 lineage 合并到最近的 lineage。

    Returns
    -------
    lineages : dict
        {cluster_label: lineage_id}，lineage_id 为 "L0", "L1", ...
    """
    from scipy.sparse import issparse
    from scipy.sparse.csgraph import connected_components

    logger.info(
        f"PAGA lineage 构建: groupby='{cluster_key}', "
        f"threshold={connectivity_threshold}"
    )

    # 确保 neighbors 图存在
    if "neighbors" not in adata.uns:
        raise RuntimeError(
            "PAGA 需要预先计算 KNN 图，请先运行 sc.pp.neighbors()"
        )

    # 运行 PAGA（model='v1.0' 不依赖 igraph cluster_graph，兼容 igraph>=0.11）
    sc.tl.paga(adata, groups=cluster_key, model="v1.0")

    # 提取连通性矩阵
    paga_conn = adata.uns["paga"]["connectivities"]
    if issparse(paga_conn):
        paga_conn = paga_conn.toarray()
    paga_conn = np.asarray(paga_conn)

    # 簇标签（保持与 cluster_cells 一致的排序）
    cluster_labels = adata.obs[cluster_key].unique()
    try:
        cluster_labels = sorted(cluster_labels, key=int)
    except (ValueError, TypeError):
        cluster_labels = sorted(cluster_labels, key=str)

    # 构建簇级连通图：connectivities >= threshold 则连边
    adj = paga_conn >= connectivity_threshold
    # 确保无向（PAGA connectivities 理论上是对称的，但数值可能有差异）
    adj = adj | adj.T

    # 连通分量分析
    n_components, component_ids = connected_components(adj, directed=False)

    # 构建 cluster -> lineage 映射
    lineage_map = {}
    for i, cl in enumerate(cluster_labels):
        comp_id = int(component_ids[i])
        lineage_map[cl] = f"L{comp_id}"

    # 统计每个 lineage 的细胞数
    lineage_cells = {}
    for cl, lin in lineage_map.items():
        n = (adata.obs[cluster_key] == cl).sum()
        lineage_cells[lin] = lineage_cells.get(lin, 0) + n

    # 处理过小的 lineage：合并到最近的 lineage
    small_lineages = {
        lin for lin, n in lineage_cells.items()
        if n < min_cells_per_lineage
    }
    if small_lineages and len(lineage_cells) > 1:
        for small_lin in small_lineages:
            # 找 paga connectivities 最高的相邻 lineage
            small_clusters = [
                cl for cl, lin in lineage_map.items() if lin == small_lin
            ]
            best_conn = -1.0
            best_target = None
            for sc_cl in small_clusters:
                sc_idx = list(cluster_labels).index(sc_cl)
                for other_cl, other_lin in lineage_map.items():
                    if other_lin == small_lin:
                        continue
                    other_idx = list(cluster_labels).index(other_cl)
                    conn_val = float(paga_conn[sc_idx, other_idx])
                    if conn_val > best_conn:
                        best_conn = conn_val
                        best_target = other_lin
            if best_target is not None:
                for cl in small_clusters:
                    lineage_map[cl] = best_target
                logger.info(
                    f"Lineage {small_lin} ({lineage_cells[small_lin]} 细胞) "
                    f"合并到 {best_target} "
                    f"(PAGA connectivity={best_conn:.3f})"
                )

    # 重新编号 lineage（合并后可能有空缺）
    unique_lineages = sorted(set(lineage_map.values()))
    if unique_lineages:
        new_ids = {old: f"L{i}" for i, old in enumerate(unique_lineages)}
        lineage_map = {cl: new_ids[lin] for cl, lin in lineage_map.items()}

    # 输出摘要
    final_counts = {}
    for cl, lin in lineage_map.items():
        n = (adata.obs[cluster_key] == cl).sum()
        final_counts[lin] = final_counts.get(lin, 0) + n

    for lin in sorted(final_counts.keys()):
        member_clusters = sorted(
            [str(c) for c, l in lineage_map.items() if l == lin],
            key=lambda x: (x.isdigit(), x),
        )
        logger.info(
            f"  {lin}: {final_counts[lin]} 细胞, "
            f"簇 = [{', '.join(member_clusters)}]"
        )

    # 将 lineage 标签写入 adata.obs
    adata.obs["lineage"] = adata.obs[cluster_key].map(lineage_map)
    logger.info(
        f"PAGA lineage 构建完成: {len(unique_lineages)} 个 lineage, "
        f"覆盖 {sum(final_counts.values())} 个细胞"
    )

    return lineage_map


def save_cluster_top_genes(
    adata: ad.AnnData,
    cluster_key: str = "leiden",
    n_top: int = 50,
    output_path: str = None,
    *, gene_identity=None,
):
    """
    对每个簇做差异表达分析，输出 top N 标记基因及统计量。

    使用 scanpy rank_genes_groups (Wilcoxon) 计算：
      - logFC: 该簇 vs 其他簇的对数倍变化
      - p_value / p_adj: Wilcoxon 秩和检验 p 值及 FDR 校正
      - pct_in: 簇内表达该基因的细胞比例
      - pct_out: 簇外表达该基因的细胞比例

    Parameters
    ----------
    adata : AnnData
        预处理后的 RNA 数据（.layers["raw"] 需包含原始计数）
    cluster_key : str
        聚类标签列名
    n_top : int
        每个簇输出的 top 基因数
    output_path : str or None
        输出 CSV 路径
    """
    import pandas as pd
    from .kinetics import _gene_base_map
    base_map = _gene_base_map(adata, gene_identity)

    if cluster_key not in adata.obs.columns:
        logger.warning(f"未找到 '{cluster_key}' 列，跳过簇标记基因输出")
        return

    # 过滤掉细胞数过少的簇，避免 rank_genes_groups 崩溃
    cluster_counts = adata.obs[cluster_key].value_counts()
    valid_clusters = cluster_counts[cluster_counts >= 5].index.tolist()
    if len(valid_clusters) < len(cluster_counts):
        skipped = set(cluster_counts.index) - set(valid_clusters)
        logger.warning(
            f"跳过 {len(skipped)} 个细胞数不足的簇: {sorted(skipped)}"
        )
        if len(valid_clusters) < 2:
            logger.warning("有效簇不足 2 个，跳过簇标记基因输出")
            return
        adata = adata[adata.obs[cluster_key].isin(valid_clusters)].copy()

    # 排除线粒体基因
    # ``var_names`` may use pandas' nullable string dtype, which makes this
    # expression a nullable BooleanArray.  AnnData requires a plain NumPy
    # boolean array for boolean variable indexing; ``na=False`` also keeps
    # missing/non-string names as non-mitochondrial.
    is_mito = np.asarray(
        pd.Index(list(base_map.values()) if base_map is not None else adata.var_names)
        .str.upper().str.startswith("MT-", na=False),
        dtype=bool,
    )
    adata_no_mt = adata[:, ~is_mito].copy()
    logger.info(
        f"排除 {is_mito.sum()} 个线粒体基因，"
        f"剩余 {adata_no_mt.n_vars} 个基因用于差异分析"
    )

    # Wilcoxon 差异表达（需要原始计数）
    # 临时将 X 替换为 raw，用完恢复
    saved_X = adata_no_mt.X
    if "raw" in adata_no_mt.layers:
        adata_no_mt.X = adata_no_mt.layers["raw"]

    sc.tl.rank_genes_groups(
        adata_no_mt, groupby=cluster_key,
        method="wilcoxon", n_genes=n_top,
        tie_correct=True,
        **({"use_raw": False} if base_map is not None else {}),
    )

    adata_no_mt.X = saved_X

    result = adata_no_mt.uns["rank_genes_groups"]
    # 兼容整数簇号（Leiden）和字符串簇名（作者标注）
    try:
        clusters = sorted(adata_no_mt.obs[cluster_key].unique(), key=int)
    except ValueError:
        clusters = sorted(adata_no_mt.obs[cluster_key].unique(), key=str)
    rows = []

    for cl in clusters:
        cl_str = str(cl)
        names = result["names"][cl_str]
        logfcs = result["logfoldchanges"][cl_str]
        pvals = result["pvals"][cl_str]
        padjs = result["pvals_adj"][cl_str]

        mask_in = adata_no_mt.obs[cluster_key] == cl
        mask_out = ~mask_in

        for rank in range(min(n_top, len(names))):
            gene = names[rank]
            if not gene:
                continue

            gene_idx = adata_no_mt.var_names.get_loc(gene)
            expr_in = adata_no_mt[mask_in, gene_idx].X
            expr_out = adata_no_mt[mask_out, gene_idx].X
            pct_in = (expr_in > 0).mean()
            pct_out = (expr_out > 0).mean()

            rows.append({
                "cluster": cl,
                "rank": rank + 1,
                "gene": gene,
                "logFC": round(float(logfcs[rank]), 4),
                "p_value": _safe_float(pvals[rank]),
                "p_adj": _safe_float(padjs[rank]),
                "pct_in": round(float(pct_in), 4),
                "pct_out": round(float(pct_out), 4),
            })

    top_df = pd.DataFrame(rows)
    if base_map is not None and "gene" in top_df:
        top_df["gene_id"] = top_df["gene"]
        top_df["gene_name"] = top_df["gene"].map(base_map)
        top_df["gene"] = top_df["gene"].map(adata.var["gene_display_name"])
        top_df["gene_display_name"] = top_df["gene"]

    if output_path is None:
        output_path = f"cluster_top{n_top}_genes.csv"
    top_df.to_csv(output_path, index=False)

    for cl in clusters:
        cl_genes = top_df[top_df["cluster"] == cl]["gene"].head(10).tolist()
        logger.info(f"  簇 {cl} top10: {', '.join(cl_genes)}")

    logger.info(f"簇标记基因已保存至: {output_path}")


def _export_cluster_markers(adata, cluster_key="leiden", n_top=50,
                            output_path=None, *, gene_identity=None):
    """Identity-aware marker export entry point for pipeline callers."""
    return save_cluster_top_genes(adata, cluster_key, n_top, output_path,
                                 gene_identity=gene_identity)


def _safe_float(val) -> float:
    """安全转换为 float，NaN/Inf 写为 1.0（最不显著的值）。"""
    v = float(val)
    if np.isnan(v) or np.isinf(v):
        return 1.0
    return v


def match_cells(
    rna_adata: ad.AnnData,
    atac_adata: ad.AnnData,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    确保 RNA 和 ATAC 两个 AnnData 对象的细胞完全匹配、顺序一致。

    这是多组学分析中最容易出错的步骤——即使两个文件来自同一个 10x Multiome
    实验，经过不同的 QC 过滤后，保留的细胞可能不同。此函数：
    1. 取两者细胞交集
    2. 将细胞按统一顺序排列
    3. 验证只有共有细胞被保留

    Returns
    -------
    rna_adata, atac_adata : 细胞匹配且顺序一致的两个 AnnData
    """
    common_cells = list(
        set(rna_adata.obs_names) & set(atac_adata.obs_names)
    )
    if len(common_cells) == 0:
        raise ValueError("RNA 和 ATAC 没有共享细胞")

    rna_adata = rna_adata[common_cells].copy()
    atac_adata = atac_adata[common_cells].copy()

    # 统一排序，确保索引一致
    sorted_barcodes = sorted(common_cells)
    rna_adata = rna_adata[sorted_barcodes]
    atac_adata = atac_adata[sorted_barcodes]

    logger.info(f"细胞匹配完成: {len(common_cells)} 个共有细胞")
    return rna_adata, atac_adata
