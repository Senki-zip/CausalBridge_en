"""Legacy WT clustering on original RNA, independent of model gene annotation.

No persistent cache: each run builds its reference using the existing preprocessing
and clustering implementations. ATAC QC is performed once and returned for modeling.
"""
import copy
import hashlib
import json
import logging
import re
from importlib.metadata import version, PackageNotFoundError

import numpy as np
from scipy import sparse

from .preprocess import preprocess_rna, preprocess_atac, match_cells, cluster_cells

logger = logging.getLogger(__name__)
SCHEMA = "legacy-wt-clustering-reference-v1"


def _unique_cells(adata):
    if not adata.obs_names.is_unique:
        raise ValueError("reference clustering requires unique cell barcodes")


def raw_target_symbols(rna, targets, gene_id_column="gene_ids"):
    """Resolve only original names/IDs; never consult GTF or model aliases."""
    if list(targets) == ["__ALL__"]:
        return list(rna.var_names)
    base = lambda value: re.sub(r"\.\d+$", "", str(value))
    ids = rna.var.get(gene_id_column)
    result = []
    for token in targets:
        if token in rna.var_names:
            symbol = token
        elif ids is not None:
            hits = np.flatnonzero(np.asarray([base(v) == base(token) for v in ids]))
            if len(hits) > 1:
                raise ValueError(f"ambiguous raw RNA target ID: {token}")
            symbol = str(rna.var_names[hits[0]]) if len(hits) else token
        else:
            symbol = token
        if symbol not in rna.var_names:
            logger.warning("Reference target %s absent from original RNA; legacy HVG retention unchanged", token)
        if symbol not in result:
            result.append(symbol)
    return result


def _input_fingerprint(adata):
    digest = hashlib.sha256()
    for axis in (adata.obs_names, adata.var_names):
        digest.update(json.dumps(list(map(str, axis)), ensure_ascii=False).encode())
    matrix = adata.X
    digest.update(str(matrix.shape).encode())
    if sparse.issparse(matrix):
        matrix = matrix.tocsr(copy=True)
        matrix.sort_indices()
        arrays = (matrix.indptr, matrix.indices, matrix.data)
    else:
        arrays = (np.asarray(matrix),)
    for array in arrays:
        digest.update(str(array.dtype).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def build_reference_clustering(rna_raw, atac_raw, targets, config):
    """Legacy order: RNA QC/HVG, ATAC QC, barcode intersection, RNA clustering."""
    _unique_cells(rna_raw)
    _unique_cells(atac_raw)
    from .atac_normalization import validate_atac_depth_config, prepare_atac_counts, META, SOURCE_SCHEMA
    if validate_atac_depth_config(config)["enabled"]:
        if META not in atac_raw.uns:
            atac_raw = prepare_atac_counts(atac_raw)
        elif atac_raw.uns[META].get("schema") != SOURCE_SCHEMA:
            raise ValueError("reference ATAC QC requires original counts, not a continuous model view")
    targets = raw_target_symbols(rna_raw, targets,
        config.get("input", {}).get("rna_gene_id_column", "gene_ids"))
    cfg = config.get("clustering", {})
    clustering = dict(enabled=cfg.get("enabled", True),
        cluster_key=cfg.get("cluster_key", "leiden"), n_neighbors=cfg.get("n_neighbors", 20),
        n_pcs=cfg.get("n_pcs", 20), resolution=cfg.get("resolution", 0.8),
        random_state=cfg.get("random_state", 42))
    versions = {}
    for package in ("numpy", "scipy", "pandas", "anndata", "scanpy", "scikit-misc", "scikit-learn", "igraph", "leidenalg", "umap-learn"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    reference_preprocess = {k: copy.deepcopy(v) for k,v in config["preprocess"].items()
                            if k != "atac_depth_normalization"}
    provenance: dict = dict(schema=SCHEMA, rna_input=_input_fingerprint(rna_raw),
        atac_input=_input_fingerprint(atac_raw), preprocess=reference_preprocess,
        clustering=clustering, targets=targets, versions=versions)
    provenance["fingerprint"] = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    rna = preprocess_rna(rna_raw.copy(), **config["preprocess"]["rna"], keep_genes=targets)
    atac = preprocess_atac(atac_raw, **config["preprocess"]["atac"])
    rna, atac = match_cells(rna, atac)
    rna = rna.copy()
    key = clustering["cluster_key"]
    if clustering["enabled"] and key not in rna.obs:
        rna = cluster_cells(rna, n_neighbors=clustering["n_neighbors"],
            n_pcs=clustering["n_pcs"], resolution=clustering["resolution"],
            random_state=clustering["random_state"], key_added=key)
    provenance.update(n_cells=rna.n_obs, n_genes=rna.n_vars,
        cell_ids=list(rna.obs_names), gene_names=list(rna.var_names),
        feature_names_unique=bool(rna.var_names.is_unique),
        representation_purpose="original-RNA legacy clustering, not model RNA PCA")
    if key in rna.obs:
        provenance["cluster_labels"] = list(rna.obs[key].astype(str))
    # Labels inherited from input are also checkpoint evidence, not merely counts.
    provenance["fingerprint"] = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    rna.uns["clustering_reference"] = provenance
    return rna, atac


def align_model_cells(model, reference):
    """Reference QC is authoritative; absent/duplicate barcodes are errors."""
    _unique_cells(model)
    _unique_cells(reference)
    missing = reference.obs_names.difference(model.obs_names)
    if len(missing):
        raise ValueError(f"model is missing reference cell barcodes: {list(missing[:10])}")
    return model[reference.obs_names].copy()


def attach_reference_clustering(model, reference, cluster_key="leiden"):
    """Copy labels/graph by barcode, without copying feature-dependent loadings."""
    _unique_cells(model)
    _unique_cells(reference)
    if set(model.obs_names) != set(reference.obs_names):
        raise ValueError("model/reference cell barcodes must match exactly")
    ordered = reference[model.obs_names]
    if cluster_key in reference.obs:
        if reference.obs[cluster_key].isna().any():
            raise ValueError("reference contains missing cluster labels")
        model.obs[cluster_key] = ordered.obs[cluster_key].astype("category").copy()
    for key in ("X_pca", "X_umap"):
        if key in ordered.obsm:
            model.obsm[key] = ordered.obsm[key].copy()
    if "neighbors" in reference.uns:
        model.uns["neighbors"] = copy.deepcopy(reference.uns["neighbors"])
        model.uns["neighbors"]["reference_purpose"] = "legacy WT clustering"
        for field in ("connectivities_key", "distances_key"):
            key = reference.uns["neighbors"].get(field)
            if key in ordered.obsp:
                model.obsp[key] = ordered.obsp[key].copy()
    if cluster_key in reference.uns:
        model.uns[cluster_key] = copy.deepcopy(reference.uns[cluster_key])
    model.uns["clustering_reference"] = copy.deepcopy(reference.uns["clustering_reference"])
    return model
