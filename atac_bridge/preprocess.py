"""
 CausalBridge preprocessing module

 Perform quality control, normalization, and highly variable gene selection on RNA and ATAC AnnData.
 All filtering steps preserve the original counts in .layers["raw"] for downstream kinetic modeling.
"""

import numpy as np
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
) -> ad.AnnData:
    """
    Quality control and normalization of RNA data.

    Processing steps:
    1. Preserve the original counts in .layers["raw"] — downstream kinetic modeling requires the original values
    2. Filter low-expression genes and low-quality cells
    3. Normalize by library size → apply a log1p transformation
    4. Select highly variable genes — reduce dimensionality and accelerate downstream Granger tests and regression
    5. Retain genes in keep_genes (such as KO target genes), even if they are not highly variable

    Parameters
    ----------
    adata : AnnData
        Raw RNA count data; .X is a sparse matrix (cells×genes)
    min_cells_per_gene : int
        Retain genes detected in at least N cells
    min_umi_per_cell : int
        Retain cells with at least N UMIs
    n_highly_variable_genes : int
        Number of highly variable genes to retain
    keep_genes : list or None
        List of genes that must be retained (such as knockout target genes), even if they are not selected as highly variable

    Returns
    -------
    adata : AnnData
        Preprocessed data; .layers["raw"] contains the original counts
    """
    adata = adata.copy()

    # --- Step 1: Preserve the original counts ---
    # Downstream kinetic parameter fitting requires the unnormalized original values
    adata.layers["raw"] = adata.X.copy()

    # --- Step 2: Gene/cell quality control ---
    logger.info(f"Before QC: {adata.n_obs} cells, {adata.n_vars} genes")
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    sc.pp.filter_cells(adata, min_counts=min_umi_per_cell)
    logger.info(f"After QC: {adata.n_obs} cells, {adata.n_vars} genes")

    if adata.n_obs < 100 or adata.n_vars < 500:
        raise ValueError(
            f"Dataset is too small after QC ({adata.n_obs} cells, {adata.n_vars} genes); "
            f"relax the filtering thresholds or check the quality of the input data."
        )

    # --- Step 3: Normalization + log transformation (for clustering and visualization) ---
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # --- Step 4: Highly variable gene selection ---
    # Highly variable genes capture the main biological variation in the data while substantially reducing computation
    # seurat_v3 requires raw counts, so temporarily replace X with raw
    normalized_X = adata.X
    adata.X = adata.layers["raw"]
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_highly_variable_genes, flavor="seurat_v3"
    )
    adata.X = normalized_X

    # --- Step 5: Determine genes to retain for downstream analysis ---
    # Downstream analysis requires HVGs + user-specified target genes.
    # Important: never modify the highly_variable column itself — PCA/clustering relies on it to select features;
    # setting target genes to highly_variable=True would change the PCA input when the target genes change,
    # resulting in inconsistent clustering results.
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
                f"Retaining {n_forced} additional target genes (not highly variable) for downstream analysis"
            )

    adata_out = adata[:, genes_to_keep].copy()
    logger.info(
        f"Retaining {adata_out.n_vars} genes for downstream analysis "
        f"(HVG={adata_out.var['highly_variable'].sum()}, "
        f"including target gene additions)"
    )

    return adata_out


def preprocess_atac(
    adata: ad.AnnData,
    min_cells_per_peak: int = 10,
    min_peaks_per_cell: int = 1000,
    binarize: bool = True,
) -> ad.AnnData:
    """
    Quality control and preprocessing of ATAC data.

    ATAC data is extremely sparse (<1% of peaks are nonzero in a single cell). Processing steps:
    1. Filter low-coverage peaks and low-quality cells
    2. Optional binarization (0/1) — ATAC signal is inherently binary (open/closed);
       binarization can reduce the effects of PCR amplification bias and sequencing-depth differences
    3. TF-IDF normalization — the standard normalization method for ATAC-seq data, increasing the weight of rare peaks

    Parameters
    ----------
    adata : AnnData
        Raw ATAC count data
    min_cells_per_peak : int
        Peaks with signal in at least N cells
    min_peaks_per_cell : int
        Cells with at least N peaks
    binarize : bool
        Whether to binarize the count matrix

    Returns
    -------
    adata : AnnData
        Preprocessed ATAC data
    """
    adata = adata.copy()

    logger.info(f"Before ATAC QC: {adata.n_obs} cells, {adata.n_vars} peaks")

    # --- Step 1: Optional binarization ---
    if binarize:
        adata.X = (adata.X > 0).astype(np.float32)
        logger.info("ATAC matrix binarized")

    # --- Step 2: Filtering ---
    sc.pp.filter_genes(adata, min_cells=min_cells_per_peak)
    sc.pp.filter_cells(adata, min_genes=min_peaks_per_cell)
    logger.info(f"After ATAC QC: {adata.n_obs} cells, {adata.n_vars} peaks")

    if adata.n_vars < 1000:
        logger.warning(
            f"ATAC peak count is low ({adata.n_vars}), which may affect the statistical power of causal tests"
        )

    return adata


def cluster_cells(
    adata: ad.AnnData,
    n_neighbors: int = 20,
    n_pcs: int = 20,
    resolution: float = 0.8,
    key_added: str = "leiden",
    random_state: int = 42,
) -> ad.AnnData:
    """
    Perform PCA + KNN graph construction + Leiden clustering on RNA data.

    Write clustering labels to adata.obs[key_added] for per-cluster modeling by the pipeline.

    Parameters
    ----------
    adata : AnnData
        log1p-normalized RNA data
    n_neighbors : int
        Number of neighbors in the KNN graph
    n_pcs : int
        Number of PCA components to use
    resolution : float
        Leiden clustering resolution (higher values produce more clusters)
    key_added : str
        Column name for clustering labels in obs
    random_state : int
        Random seed

    Returns
    -------
    adata : AnnData
        Data with clustering labels and UMAP coordinates
    """
    logger.info(f"PCA → KNN(n={n_neighbors}, pcs={n_pcs}) → Leiden(res={resolution}) → UMAP")

    # Compute PCA (scanpy checks whether PCA already exists and skips it if so)
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=max(50, n_pcs), random_state=random_state)
        logger.info(f"  PCA computed: {adata.obsm['X_pca'].shape}")

    # Build the neighborhood graph
    sc.pp.neighbors(
        adata, n_neighbors=n_neighbors, n_pcs=n_pcs,
        random_state=random_state,
    )

    # Leiden clustering
    sc.tl.leiden(adata, resolution=resolution, key_added=key_added,
                 random_state=random_state)

    # UMAP dimensionality reduction (reuse the neighbors graph)
    sc.tl.umap(adata, random_state=random_state)
    logger.info(f"  UMAP computed: {adata.obsm['X_umap'].shape}")

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
    logger.info(f"  Clustering complete: {n_clusters} clusters, "
                f"sizes = {dict(cluster_sizes)}")

    return adata


def build_paga_lineages(
    adata: ad.AnnData,
    cluster_key: str = "leiden",
    connectivity_threshold: float = 0.1,
    min_cells_per_lineage: int = 50,
) -> dict:
    """
    Run PAGA on the Leiden clustering and merge connected clusters into continuous lineages.

    Purpose: Leiden clustering may fragment continuous developmental lineages into multiple small clusters. PAGA's graph abstraction
    can identify the topological connectivity between these clusters and merge clusters that belong to the same lineage,
    allowing pseudotime inference to run on the merged continuous lineages and avoiding broken trajectories.

    Parameters
    ----------
    adata : AnnData
        RNA data with PCA + neighbors + Leiden clustering completed.
        Must contain .obsm["X_pca"] and .uns["neighbors"].
    cluster_key : str
        Column name for Leiden clustering labels in .obs
    connectivity_threshold : float
        PAGA connectivity threshold. Cluster pairs with connectivities[i,j] >= threshold are considered connected.
    min_cells_per_lineage : int
        Minimum number of cells after merging. Lineages below this value are merged into the nearest lineage.

    Returns
    -------
    lineages : dict
        {cluster_label: lineage_id}, where lineage_id is "L0", "L1", ...
    """
    from scipy.sparse import issparse
    from scipy.sparse.csgraph import connected_components

    logger.info(
        f"Building PAGA lineages: groupby='{cluster_key}', "
        f"threshold={connectivity_threshold}"
    )

    # Ensure that the neighbors graph exists
    if "neighbors" not in adata.uns:
        raise RuntimeError(
            "PAGA requires a precomputed KNN graph; run sc.pp.neighbors() first"
        )

    # Run PAGA (model='v1.0' does not depend on igraph cluster_graph and is compatible with igraph>=0.11)
    sc.tl.paga(adata, groups=cluster_key, model="v1.0")

    # Extract the connectivity matrix
    paga_conn = adata.uns["paga"]["connectivities"]
    if issparse(paga_conn):
        paga_conn = paga_conn.toarray()
    paga_conn = np.asarray(paga_conn)

    # Cluster labels (keep the ordering consistent with cluster_cells)
    cluster_labels = adata.obs[cluster_key].unique()
    try:
        cluster_labels = sorted(cluster_labels, key=int)
    except (ValueError, TypeError):
        cluster_labels = sorted(cluster_labels, key=str)

    # Build the cluster-level connectivity graph: add an edge when connectivities >= threshold
    adj = paga_conn >= connectivity_threshold
    # Ensure it is undirected (PAGA connectivities should theoretically be symmetric, but values may differ)
    adj = adj | adj.T

    # Analyze connected components
    n_components, component_ids = connected_components(adj, directed=False)

    # Build the cluster -> lineage mapping
    lineage_map = {}
    for i, cl in enumerate(cluster_labels):
        comp_id = int(component_ids[i])
        lineage_map[cl] = f"L{comp_id}"

    # Count cells in each lineage
    lineage_cells = {}
    for cl, lin in lineage_map.items():
        n = (adata.obs[cluster_key] == cl).sum()
        lineage_cells[lin] = lineage_cells.get(lin, 0) + n

    # Handle small lineages: merge them into the nearest lineage
    small_lineages = {
        lin for lin, n in lineage_cells.items()
        if n < min_cells_per_lineage
    }
    if small_lineages and len(lineage_cells) > 1:
        for small_lin in small_lineages:
            # Find the adjacent lineage with the highest PAGA connectivity
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
                    f"Lineage {small_lin} ({lineage_cells[small_lin]} cells) "
                    f"merged into {best_target} "
                    f"(PAGA connectivity={best_conn:.3f})"
                )

    # Renumber lineages (merging may leave gaps)
    unique_lineages = sorted(set(lineage_map.values()))
    if unique_lineages:
        new_ids = {old: f"L{i}" for i, old in enumerate(unique_lineages)}
        lineage_map = {cl: new_ids[lin] for cl, lin in lineage_map.items()}

    # Output summary
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
            f"  {lin}: {final_counts[lin]} cells, "
            f"clusters = [{', '.join(member_clusters)}]"
        )

    # Write lineage labels to adata.obs
    adata.obs["lineage"] = adata.obs[cluster_key].map(lineage_map)
    logger.info(
        f"PAGA lineage construction complete: {len(unique_lineages)} lineages, "
        f"covering {sum(final_counts.values())} cells"
    )

    return lineage_map


def save_cluster_top_genes(
    adata: ad.AnnData,
    cluster_key: str = "leiden",
    n_top: int = 50,
    output_path: str = None,
):
    """
    Perform differential expression analysis for each cluster and output top N marker genes and statistics.

    Use scanpy rank_genes_groups (Wilcoxon) to calculate:
      - logFC: log fold change for this cluster vs. other clusters
      - p_value / p_adj: Wilcoxon rank-sum test p-value and FDR correction
      - pct_in: proportion of cells expressing this gene within the cluster
      - pct_out: proportion of cells expressing this gene outside the cluster

    Parameters
    ----------
    adata : AnnData
        Preprocessed RNA data (.layers["raw"] must contain the original counts)
    cluster_key : str
        Column name for clustering labels
    n_top : int
        Number of top genes to output for each cluster
    output_path : str or None
        Output CSV path
    """
    import pandas as pd

    if cluster_key not in adata.obs.columns:
        logger.warning(f"Column '{cluster_key}' not found; skipping cluster marker gene output")
        return

    # Filter out clusters with too few cells to prevent rank_genes_groups from failing
    cluster_counts = adata.obs[cluster_key].value_counts()
    valid_clusters = cluster_counts[cluster_counts >= 5].index.tolist()
    if len(valid_clusters) < len(cluster_counts):
        skipped = set(cluster_counts.index) - set(valid_clusters)
        logger.warning(
            f"Skipping {len(skipped)} clusters with too few cells: {sorted(skipped)}"
        )
        if len(valid_clusters) < 2:
            logger.warning("Fewer than 2 valid clusters; skipping cluster marker gene output")
            return
        adata = adata[adata.obs[cluster_key].isin(valid_clusters)].copy()

    # Exclude mitochondrial genes
    # ``var_names`` may use pandas' nullable string dtype, which makes this
    # expression a nullable BooleanArray.  AnnData requires a plain NumPy
    # boolean array for boolean variable indexing; ``na=False`` also keeps
    # missing/non-string names as non-mitochondrial.
    is_mito = np.asarray(
        adata.var_names.str.upper().str.startswith("MT-", na=False),
        dtype=bool,
    )
    adata_no_mt = adata[:, ~is_mito].copy()
    logger.info(
        f"Excluded {is_mito.sum()} mitochondrial genes, "
        f"leaving {adata_no_mt.n_vars} genes for differential expression analysis"
    )

    # Wilcoxon differential expression (requires raw counts)
    # Temporarily replace X with raw, then restore it
    saved_X = adata_no_mt.X
    if "raw" in adata_no_mt.layers:
        adata_no_mt.X = adata_no_mt.layers["raw"]

    sc.tl.rank_genes_groups(
        adata_no_mt, groupby=cluster_key,
        method="wilcoxon", n_genes=n_top,
        tie_correct=True,
    )

    adata_no_mt.X = saved_X

    result = adata_no_mt.uns["rank_genes_groups"]
    # Support both integer cluster IDs (Leiden) and string cluster names (author annotations)
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

    if output_path is None:
        output_path = f"cluster_top{n_top}_genes.csv"
    top_df.to_csv(output_path, index=False)

    for cl in clusters:
        cl_genes = top_df[top_df["cluster"] == cl]["gene"].head(10).tolist()
        logger.info(f"  Cluster {cl} top10: {', '.join(cl_genes)}")

    logger.info(f"Cluster marker genes saved to: {output_path}")


def _safe_float(val) -> float:
    """Safely convert to float; write NaN/Inf as 1.0 (the least significant value)."""
    v = float(val)
    if np.isnan(v) or np.isinf(v):
        return 1.0
    return v


def match_cells(
    rna_adata: ad.AnnData,
    atac_adata: ad.AnnData,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Ensure that the cells in the RNA and ATAC AnnData objects match exactly and have the same order.

    This is one of the most error-prone steps in multi-omics analysis—even if the two files come from the same 10x Multiome
    experiment, the retained cells may differ after different QC filters. This function:
    1. Takes the intersection of the cells in both objects
    2. Orders the cells consistently
    3. Verifies that only shared cells are retained

    Returns
    -------
     rna_adata, atac_adata : Two AnnData objects with matching cells in the same order
    """
    common_cells = list(
        set(rna_adata.obs_names) & set(atac_adata.obs_names)
    )
    if len(common_cells) == 0:
        raise ValueError("RNA and ATAC have no shared cells")

    rna_adata = rna_adata[common_cells].copy()
    atac_adata = atac_adata[common_cells].copy()

    # Sort consistently to ensure matching indices
    sorted_barcodes = sorted(common_cells)
    rna_adata = rna_adata[sorted_barcodes]
    atac_adata = atac_adata[sorted_barcodes]

    logger.info(f"Cell matching complete: {len(common_cells)} shared cells")
    return rna_adata, atac_adata
