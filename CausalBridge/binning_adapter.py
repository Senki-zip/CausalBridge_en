"""Bounded WT embedding and strict integration contract for frozen binning.

No pipeline/config objects are serialized. Fitting only updates primitive
pseudotime labels/attrs; diagnostics are written exclusively on explicit request.
"""
from dataclasses import dataclass, fields, asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Optional
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

from .adaptive_binning import BinPlan, BinningConfig, FittedBinningPolicy, aggregate, fit_policy

SCHEMA_VERSION = "binning-adapter-1"


def _primitive(value):
    if isinstance(value, np.ndarray):
        return _primitive(value.tolist())
    if isinstance(value, np.generic):
        return _primitive(value.item())
    if isinstance(value, (tuple, list)):
        return [_primitive(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _primitive(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"nonprimitive metadata/config value: {type(value).__name__}")


def _selector(config):
    settings = config.get("pseudotime", {}).get("adaptive_binning", {})
    if not isinstance(settings, dict):
        raise ValueError("pseudotime.adaptive_binning must be a dictionary")
    allowed = {f.name for f in fields(BinningConfig)} | {"enabled", "embedding", "state_modality"}
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(f"unknown adaptive_binning settings: {sorted(unknown)}")
    if not isinstance(settings.get("enabled", False), bool):
        raise ValueError("adaptive_binning.enabled must be boolean")
    _validate_state_modality(settings.get("state_modality", "joint"))
    return settings


def _validate_state_modality(value):
    if not isinstance(value, str) or value not in {"joint", "rna"}:
        raise ValueError("adaptive_binning.state_modality must be 'joint' or 'rna'")
    return value


def select_state_embedding(joint_embedding, metadata, state_modality="joint"):
    """Reuse prepared PCs for state decisions, never refit or change trajectory.

    RNA mode requires contiguous half-open RNA/ATAC column slices covering the
    joint representation exactly. Legacy arbitrary embeddings are accepted only
    in joint mode; component counts must never be guessed.
    """
    mode = _validate_state_modality(state_modality)
    if mode == "joint":
        return joint_embedding
    try:
        modalities = metadata["modalities"]
        edge = 0
        for name in ("rna", "atac"):
            block = modalities[name]
            start, stop = block["column_slice"]
            count = block["n_components"]
            if any(isinstance(v, bool) or not isinstance(v, (int, np.integer))
                   for v in (start, stop, count)):
                raise ValueError
            if start != edge or stop <= start or stop - start != count:
                raise ValueError
            edge = stop
        if edge != joint_embedding.shape[1]:
            raise ValueError
        start, stop = modalities["rna"]["column_slice"]
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("RNA state requires structured RNA/ATAC column_slice metadata matching the joint embedding") from exc
    return (joint_embedding.iloc[:, start:stop] if isinstance(joint_embedding, pd.DataFrame)
            else np.asarray(joint_embedding)[:, start:stop])


def require_bin_plan(config, bin_plan):
    """Legacy callers may omit a plan; enabled callers may never silently rebin."""
    if _selector(config).get("enabled", False) and bin_plan is None:
        raise ValueError("adaptive binning enabled: a fitted bin_plan is required")


def _ids(ids, name):
    index = pd.Index(ids)
    if not index.is_unique or index.hasnans or not len(index):
        raise ValueError(f"{name} cell IDs must be nonempty, unique and nonmissing")
    return index


def _aligned(rna, atac):
    ids = _ids(rna.obs_names, "RNA")
    other = _ids(atac.obs_names, "ATAC")
    if not ids.equals(other):
        raise ValueError("RNA/ATAC cell IDs must match exactly, including order")
    return ids


def _usable(plan):
    if not isinstance(plan, BinPlan):
        raise TypeError("plan must be BinPlan")
    if not plan.valid.all():
        raise ValueError("bin_plan contains invalid bins; refusing NaN/zero imputation")
    if not plan.cell_valid.all():
        raise ValueError("bin_plan has uncovered training cells")
    for name in ("raw_kernel", "aggregate_weights", "membership"):
        matrix = getattr(plan, name)
        values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
        if not np.isfinite(values).all():
            raise ValueError(f"bin_plan {name} is not finite")


def plan_branch_boundaries(plan):
    """Derive ordered half-open bin-row ranges from the plan, never pt attrs.

    Empty qcut labels inside a branch range are allowed (Gaussian estimates can
    remain supported), but overlapping, interleaved or unassigned row ranges
    cannot define a safe execution partition.
    """
    _usable(plan)
    boundaries = []
    for branch in np.unique(plan.branches):
        labels = np.asarray(plan.labels[plan.branches == branch])
        if (not len(labels) or not np.isfinite(labels).all()
                or not np.equal(labels, labels.astype(int)).all()
                or labels.min() < 0 or labels.max() >= plan.n_bins):
            raise ValueError("bin_plan branch labels do not define valid bin rows")
        boundaries.append((int(labels.min()), int(labels.max()) + 1))
    boundaries.sort()
    edge = 0
    for start, end in boundaries:
        if start != edge:
            raise ValueError("bin_plan branch row ranges overlap or leave uncovered bins")
        edge = end
    if edge != plan.n_bins:
        raise ValueError("bin_plan branch row ranges leave uncovered bins")
    return tuple(boundaries)


def execution_pseudotime(plan, pt, cell_ids):
    """Validate caller metadata, then supply plan-derived attrs on a private copy."""
    validate_bin_plan(plan, pt, cell_ids)
    execution = pt.copy(deep=True)
    execution["bin"] = plan.labels
    execution.attrs["branch_boundaries"] = plan_branch_boundaries(plan)
    execution.attrs.update(bin_plan_fingerprint=plan.fingerprint,
                           bin_policy_fingerprint=plan.policy_fingerprint,
                           bin_plan_centers=plan.centers.tolist())
    return execution


def validate_bin_plan(plan, pt, cell_ids):
    """Check ordered IDs, times, explicit branch mapping and hard qcut labels."""
    _usable(plan)
    ids = _ids(cell_ids, "matrix")
    pt_ids = _ids(pt.index, "pseudotime")
    if not ids.equals(pd.Index(plan.cell_ids)) or not pt_ids.equals(ids):
        raise ValueError("plan, pseudotime and matrix cell IDs must match exactly in order")
    for column, expected in (("pseudotime", plan.times), ("bin", plan.labels)):
        if column not in pt or not np.array_equal(pt[column].to_numpy(), expected):
            raise ValueError(f"pseudotime {column} differs from frozen bin_plan")
    branches = pt["branch"].to_numpy() if "branch" in pt else np.full(len(pt), "__global__")
    if not np.array_equal(branches, plan.branches):
        raise ValueError("pseudotime branch mapping differs from frozen bin_plan")
    for key, expected in (("bin_plan_fingerprint", plan.fingerprint),
                          ("bin_policy_fingerprint", plan.policy_fingerprint),
                          ("bin_plan_centers", plan.centers.tolist())):
        if key in pt.attrs and _primitive(pt.attrs[key]) != _primitive(expected):
            raise ValueError(f"pseudotime attr {key} differs from frozen bin_plan")
    if "branch_boundaries" in pt.attrs:
        boundaries = plan_branch_boundaries(plan)
        if _primitive(pt.attrs["branch_boundaries"]) != _primitive(boundaries):
            raise ValueError("pseudotime branch_boundaries differ from frozen bin_plan")


def aggregate_with_plan(plan, matrix, *, cell_ids):
    _usable(plan)
    result = aggregate(plan, matrix, cell_ids=cell_ids)
    if not np.isfinite(result).all():
        raise ValueError("bin_plan aggregation produced nonfinite values")
    return result


def _embedding_settings(settings):
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise ValueError("embedding settings must be a dictionary")
    defaults = dict(max_features=512, n_components=5, max_dense_elements=2_000_000, random_seed=42)
    if set(settings) - set(defaults):
        raise ValueError(f"unknown embedding settings: {sorted(set(settings) - set(defaults))}")
    cfg = dict(defaults, **settings)
    for key, cap in defaults.items():
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1 <= value <= cap:
            raise ValueError(f"embedding {key} must be a positive integer <= {cap}")
    if cfg["random_seed"] != 42:
        raise ValueError("embedding random_seed is fixed at 42")
    return cfg


def prepare_wt_embedding(rna, atac, settings):
    """Variance-capped centered PCA, equal block RMS, without PC whitening.

    Settings: max_features (<=512/modality), n_components (<=5/modality),
    max_dense_elements (<=2,000,000 total selected inputs), random_seed (=42).
    Sparse variance is computed before any selected input is densified.
    Metadata records effective n_components and half-open column_slice boundaries
    for each modality in the joint embedding, including rank-reduced blocks.
    """
    ids = _aligned(rna, atac)
    cfg = _embedding_settings(settings)
    selected = []
    for name, adata in (("rna", rna), ("atac", atac)):
        x = adata.X
        if x.ndim != 2 or x.shape[0] != len(ids) or len(ids) < 2:
            raise ValueError(f"{name} matrix must have >=2 aligned cells")
        values = x.data if sparse.issparse(x) else np.asarray(x)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} matrix must be finite")
        if sparse.issparse(x):
            # Floating point moments avoid integer overflow and never densify X.
            x = x.astype(np.float64)
            mean = np.asarray(x.mean(axis=0)).ravel()
            variance = np.asarray(x.multiply(x).mean(axis=0)).ravel() - mean**2
        else:
            variance = np.var(np.asarray(x, dtype=float), axis=0)
        order = np.argsort(-variance, kind="stable")
        indices = order[variance[order] > 0][:cfg["max_features"]]
        if not len(indices):
            raise ValueError(f"{name} modality is constant")
        selected.append((name, adata, x, indices))
    elements = sum(len(ids) * len(indices) for _, _, _, indices in selected)
    if elements > cfg["max_dense_elements"]:
        raise MemoryError(f"selected embedding inputs {elements} exceed max_dense_elements={cfg['max_dense_elements']}")
    blocks = []
    metadata: dict[str, Any] = dict(schema_version=SCHEMA_VERSION, settings=cfg, dense_elements=elements,
                    normalization="equal modality total RMS; no PC whitening", modalities={})
    for name, adata, x, indices in selected:
        capped = x[:, indices]
        dense = capped.toarray() if sparse.issparse(capped) else np.asarray(capped, dtype=float)
        if np.all(dense == dense[0]):
            raise ValueError(f"{name} modality is constant (zero effective rank)")
        n = min(cfg["n_components"], len(ids)-1, len(indices))
        pca = PCA(n_components=n, svd_solver="full", whiten=False, random_state=42)
        scores = pca.fit_transform(dense)
        tolerance = np.finfo(float).eps * max(dense.shape) * pca.singular_values_[0]
        rank = int(np.count_nonzero(pca.singular_values_ > tolerance))
        if not rank:
            raise ValueError(f"{name} modality has zero effective rank")
        scores = scores[:, :rank]
        rms = float(np.sqrt(np.mean(np.sum(scores**2, axis=1))))
        start = sum(block.shape[1] for block in blocks)
        blocks.append(scores / rms)
        metadata["modalities"][name] = dict(feature_indices=indices.tolist(),
            features=_primitive(np.asarray(adata.var_names)[indices]), n_components=rank, total_rms=rms,
            column_slice=[start, start + rank])
    embedding = pd.DataFrame(np.hstack(blocks), index=ids)
    if not np.isfinite(embedding.to_numpy()).all():
        raise ValueError("embedding is not finite")
    return embedding, metadata


def _hash_array(digest, array):
    a = np.asarray(array)
    digest.update(str((a.shape, a.dtype)).encode())
    if a.dtype.kind in "OUS":
        digest.update(json.dumps(_primitive(a), ensure_ascii=False, sort_keys=True).encode())
    else:
        digest.update(np.ascontiguousarray(a).tobytes())


def _hash_matrix(digest, matrix):
    digest.update(str(matrix.shape).encode())
    if sparse.issparse(matrix):
        matrix = matrix.tocsr(copy=True)
        matrix.sort_indices()
        digest.update(b"csr")
        for buffer in (matrix.data, matrix.indices, matrix.indptr):
            _hash_array(digest, buffer)
    else:
        _hash_array(digest, matrix)


def _data_fingerprint(rna, atac):
    digest = hashlib.sha256(SCHEMA_VERSION.encode())
    for name, adata in (("rna", rna), ("atac", atac)):
        digest.update(name.encode())
        _hash_array(digest, adata.obs_names)
        _hash_array(digest, adata.var_names)
        _hash_matrix(digest, adata.X)
        for column in sorted(adata.var.columns):
            digest.update(str(column).encode())
            _hash_array(digest, adata.var[column].to_numpy())
        if name == "rna":
            raw = getattr(adata, "layers", {}).get("raw")
            if raw is not None:
                digest.update(b"raw-layer")
                _hash_matrix(digest, raw)
            raw = getattr(adata, "raw", None)
            if raw is not None:
                digest.update(b"raw-adata")
                _hash_array(digest, raw.var_names)
                _hash_matrix(digest, raw.X)
    from .atac_normalization import atac_input_contract
    representation = atac_input_contract(atac)
    if representation is not None:
        digest.update(json.dumps(representation,sort_keys=True,allow_nan=False).encode())
    return digest.hexdigest()


def _model_config(config) -> Any:
    # Exclude output locations and runtime concurrency, retain scientific inputs.
    if isinstance(config, dict):
        result = {k: _model_config(v) for k, v in config.items()
                  if k not in {"output", "output_dir", "output_path", "outdir"}}
        if "preprocess" in result:
            from .atac_normalization import validate_atac_depth_config
            depth = validate_atac_depth_config(config)
            if depth["enabled"]:
                result["preprocess"]["atac_depth_normalization"] = depth
            else:
                result["preprocess"].pop("atac_depth_normalization", None)
        if isinstance(result.get("perturbation"), dict):
            result["perturbation"].pop("joint_workers", None)
            # Output-only merge policy must not invalidate fitted bin/model caches.
            result["perturbation"].pop("aggregation_policy", None)
        section = result.get("atac_to_rna", {})
        if "atac_to_rna" in result:
            from .saturating_transfer import saturating_model_settings, response_contract, GENE_TYPE, TF_TYPE
            result["atac_to_rna"] = {"model_mode": "saturating_hybrid",
                # Keep the historical workers=1 field, independent of concurrency.
                "saturating_hybrid": saturating_model_settings(config), **response_contract(config, GENE_TYPE)}
            tf_section = result.get("rna_to_atac", {})
            tf_section.pop("response_model", None)
            tf_section.pop("nn_response", None)
            tf_section.pop("spline_response", None)
            tf_section.update(response_contract(config, TF_TYPE))
        from ._wt_aggregation import validate_model_mode
        validate_model_mode(config)
        return result
    return _primitive(config)


@dataclass(frozen=True)
class BinningContext:
    policy: FittedBinningPolicy
    plan: BinPlan
    metadata: dict
    data_fingerprint: str
    model_contract_fingerprint: str
    attrs: Optional[dict] = None


def fit_context_binning(rna, atac, pt, config, *, embedding, embedding_metadata):
    ids = _aligned(rna, atac)
    if not _ids(pt.index, "pseudotime").equals(ids):
        raise ValueError("pseudotime IDs must match RNA/ATAC exactly in order")
    settings = _selector(config)
    _embedding_settings(settings.get("embedding", {}))
    if not settings.get("enabled", False):
        raise ValueError("fit_context_binning requires adaptive_binning.enabled=true")
    pcfg = config.get("pseudotime", {})
    if pcfg.get("ablation_shuffle_pseudotime", False) or pt.attrs.get("_need_rebin", False):
        raise ValueError("adaptive binning does not support shuffled pseudotime")
    options = dict(max_bins=pcfg.get("n_bins", 100), min_cells_per_bin=pcfg.get("min_cells_per_bin", 5))
    if pcfg.get("smooth_kernel", "hard") == "gaussian":
        options.update(target_bins=pcfg.get("n_bins", 100), overlap_factor=pcfg.get("smooth_overlap_factor"))
    options.update({k: v for k, v in settings.items() if k not in {"enabled", "embedding", "state_modality"}})
    cfg = BinningConfig(**options)
    # Validate metadata/config before fitting or mutating pt.
    emb_metadata = _primitive(embedding_metadata)
    mode = settings.get("state_modality", "joint")
    state_embedding = select_state_embedding(embedding, emb_metadata, mode)
    def evidence(matrix):
        digest = hashlib.sha256()
        _hash_array(digest, np.asarray(matrix))
        return dict(shape=list(matrix.shape), fingerprint=digest.hexdigest())
    state_metadata = dict(state_modality=mode, joint=evidence(embedding), state=evidence(state_embedding))
    model_config = _model_config(config)
    target_path = config.get("input", {}).get("target_genes")
    if isinstance(target_path, (str, Path)):
        # Target content is model input, not just an output/cache path.
        model_config["target_genes_content"] = Path(target_path).read_text(encoding="utf-8")
    policy = fit_policy(pt["pseudotime"].to_numpy(), ids.to_numpy(),
                        pt["branch"].to_numpy() if "branch" in pt else None,
                        embedding=state_embedding, config=cfg)
    plan = policy.initial_plan
    _usable(plan)
    if not plan.reliable.all():
        warnings.warn("adaptive binning has low-reliability bins; estimates remain valid (no imputation)", RuntimeWarning, stacklevel=2)
    data_fp = _data_fingerprint(rna, atac)
    contract = dict(schema_version=SCHEMA_VERSION, config=model_config, binning_config=asdict(cfg),
                    data_fingerprint=data_fp, policy_fingerprint=policy.fingerprint,
                    plan_fingerprint=plan.fingerprint, embedding_metadata=emb_metadata)
    contract["state_embedding"] = state_metadata
    model_fp = hashlib.sha256(json.dumps(_primitive(contract), sort_keys=True, allow_nan=False).encode()).hexdigest()
    attrs = dict(branch_boundaries=policy.branch_boundaries, bin_plan_centers=plan.centers.tolist(),
                 bin_plan_fingerprint=plan.fingerprint, bin_policy_fingerprint=policy.fingerprint,
                 binning_schema_version=SCHEMA_VERSION, binning_data_fingerprint=data_fp,
                 binning_model_contract_fingerprint=model_fp)
    metadata = dict(schema_version=SCHEMA_VERSION, policy_fingerprint=policy.fingerprint,
                    plan_fingerprint=plan.fingerprint, data_fingerprint=data_fp,
                    model_contract_fingerprint=model_fp, embedding=emb_metadata,
                    diagnostics_refs={"metadata": "binning_metadata.json", "bins": "binning_bins.csv"})
    metadata["state_embedding"] = state_metadata
    pt["bin"] = plan.labels
    pt.attrs.update(attrs)
    validate_bin_plan(plan, pt, ids)
    return BinningContext(policy, plan, metadata, data_fp, model_fp, attrs)


def persist_binning_diagnostics(context, path):
    """Write primitive JSON and small per-bin CSV; never cell-by-bin CSV."""
    if not isinstance(context, BinningContext):
        raise TypeError("context must be BinningContext")
    plan = context.plan
    records = dict(bin=np.arange(plan.n_bins), center=plan.centers, h=plan.bandwidths,
                   ess=plan.diagnostics["ess"], support_cells=plan.diagnostics["support_cells"],
                   valid=plan.valid, reliable=plan.reliable)
    for key in ("state_confident", "chosen_score_evidenced", "supported_pilot_count", "reason"):
        if key in plan.bandwidth_diagnostics:
            records[key] = plan.bandwidth_diagnostics[key]
    if "geometric_coverage" in plan.bandwidth_diagnostics:
        diag = plan.bandwidth_diagnostics
        records["state_coverage"] = diag["geometric_coverage"][np.arange(plan.n_bins), diag["chosen_candidate_index"]]
    payload = _primitive(dict(context.metadata, config=asdict(context.policy.config), attrs=context.attrs,
                              diagnostics=plan.diagnostics, bandwidth_diagnostics=plan.bandwidth_diagnostics,
                              boundary_diagnostics=context.policy.boundary_diagnostics))
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "binning_metadata.json").write_text(text, encoding="utf-8")
    pd.DataFrame(records).to_csv(directory / "binning_bins.csv", index=False)
    return dict(context.metadata["diagnostics_refs"])
