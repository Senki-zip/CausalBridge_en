"""Owned continuous ATAC model views with a frozen full-input peak-count denominator."""
import copy
import hashlib
import json
import math

import numpy as np
from scipy import sparse

UNIT = "normalized_atac_log1p"
SOURCE_SCHEMA = "atac_full_input_count_depth_v2"
QC_SCHEMA = "atac_checked_retained_counts_v2"
VIEW_SCHEMA = "atac_depth_log1p_view_v2"
RAW_LAYER = "_atac_input_counts"
DEPTH = "_atac_full_input_depth"
ROW_PROOF = "_atac_full_input_depth_proof"
COUNT_PROOF = "_atac_retained_count_proof"
META = "atac_depth_normalization"
SUFFIX = "atac_depth_log1p"
DENOMINATOR_KIND = "full_input_peak_count_sum_proxy_not_total_fragments"


def validate_atac_depth_config(config):
    section = config.get("preprocess", {}).get("atac_depth_normalization", {})
    if not isinstance(section, dict) or set(section)-{"enabled", "target_sum", "output_cap_expansion"}:
        raise ValueError("unknown preprocess.atac_depth_normalization settings")
    settings = dict(enabled=False, target_sum=10000., output_cap_expansion=.25)
    settings.update(section)
    if type(settings["enabled"]) is not bool:
        raise ValueError("atac_depth_normalization.enabled must be boolean")
    for key in ("target_sum", "output_cap_expansion"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"atac_depth_normalization.{key} must be finite numeric")
        settings[key] = float(value)
    if settings["target_sum"] <= 0 or settings["output_cap_expansion"] < 0:
        raise ValueError("ATAC target_sum must be positive and cap expansion nonnegative")
    if settings["enabled"]:
        if config.get("atac_to_rna", {}).get("model_mode", "saturating_hybrid") != "saturating_hybrid":
            raise ValueError("ATAC depth normalization requires saturating_hybrid")
        if config.get("preprocess", {}).get("atac", {}).get("binarize", True) is not True:
            raise ValueError("ATAC depth normalization requires binary reference QC (binarize=true)")
    return settings


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _axis_hash(axis):
    return _digest(list(map(str, axis)))


def _counts(matrix):
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.rint(values)).any():
        raise ValueError("continuous ATAC requires finite nonnegative integer input counts, not transformed values")
    result = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    result.sum_duplicates(); result.sort_indices(); result.eliminate_zeros()
    if not np.isfinite(result.data).all():
        raise ValueError("ATAC count canonicalization overflow")
    return result


def _matrix_hash(matrix):
    m = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    m.sum_duplicates(); m.sort_indices(); m.eliminate_zeros()
    digest = hashlib.sha256(str(m.shape).encode())
    for values in (m.indptr.astype(np.int64), m.indices.astype(np.int64), m.data):
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _row_proof(cell, depth, source_hash):
    return _digest([str(cell), float(depth), source_hash])


def _count_row_proofs(counts, cells, peak_hash, source_hash):
    """One digest per cell; peak axis and original source are part of its domain."""
    proofs = []
    for row, cell in enumerate(cells):
        start, end = counts.indptr[row:row+2]
        digest = hashlib.sha256(_digest([str(cell), peak_hash, source_hash]).encode())
        digest.update(np.ascontiguousarray(counts.indices[start:end], dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(counts.data[start:end], dtype=np.float64).tobytes())
        proofs.append(digest.hexdigest())
    return proofs


def _source_evidence(meta):
    """Check the small complete depth vector, including cells removed by QC."""
    try:
        cells = list(map(str, meta["source_full_cell_ids"]))
        depth = np.asarray(meta["source_full_depth"], dtype=np.float64)
        shape = list(meta["source_shape"])
        if (len(shape) != 2 or len(cells) != shape[0] or len(set(cells)) != len(cells)
                or depth.shape != (len(cells),) or not np.isfinite(depth).all() or (depth < 0).any()
                or meta["source_full_cell_axis_sha256"] != _axis_hash(cells)
                or meta["source_depth_sha256"] != _digest(depth.tolist())):
            raise ValueError("full-input ATAC depth source evidence mismatch")
        return cells, depth
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("missing or malformed full-input ATAC depth source evidence") from exc


def _aligned_source_depth(meta, cells):
    source_cells, source_depth = _source_evidence(meta)
    index = {cell: row for row, cell in enumerate(source_cells)}
    try:
        return source_depth[[index[str(cell)] for cell in cells]]
    except KeyError as exc:
        raise ValueError("retained ATAC cell absent from full-input depth source") from exc


def validate_full_atac_source(prepared):
    """Verify original counts, both full axes, and exact row sums before any QC."""
    meta = prepared.uns.get(META, {})
    if meta.get("schema") != SOURCE_SCHEMA:
        raise ValueError("ATAC QC source must be the complete prepared original input")
    source_cells, source_depth = _source_evidence(meta)
    if (not prepared.obs_names.is_unique or not prepared.var_names.is_unique
            or list(map(str, prepared.obs_names)) != source_cells
            or list(prepared.shape) != list(meta["source_shape"])
            or _axis_hash(prepared.var_names) != meta["source_full_peak_axis_sha256"]):
        raise ValueError("complete full-input ATAC source axes are unavailable; finalize QC with the original source")
    if RAW_LAYER not in prepared.layers or DEPTH not in prepared.obs:
        raise ValueError("missing full-input ATAC count/depth source")
    counts = _counts(prepared.layers[RAW_LAYER])
    if (_matrix_hash(counts) != meta["source_counts_sha256"]
            or _matrix_hash(_counts(prepared.X)) != meta["source_counts_sha256"]):
        raise ValueError("full-input ATAC source counts were modified after preparation")
    if (not np.array_equal(np.asarray(counts.sum(axis=1)).ravel(), source_depth)
            or not np.array_equal(np.asarray(prepared.obs[DEPTH], dtype=float), source_depth)):
        raise ValueError("full-input ATAC source depths differ from original counts")
    expected = [_row_proof(c, d, meta["source_counts_sha256"]) for c, d in zip(source_cells, source_depth)]
    if ROW_PROOF not in prepared.obs or list(prepared.obs[ROW_PROOF]) != expected:
        raise ValueError("full-input ATAC source depth proofs were modified")
    return counts


def finalize_atac_qc(qc_adata, prepared_source):
    """Bind a legitimate retained subset to the verified complete pre-QC source.

    Cell slicing then preserves row proofs. Peak slicing requires this explicit
    handoff again while the original prepared source is still available.
    """
    source_counts = validate_full_atac_source(prepared_source)
    source_meta = prepared_source.uns[META]
    meta = qc_adata.uns.get(META, {})
    if meta.get("schema") not in (SOURCE_SCHEMA, QC_SCHEMA):
        raise ValueError("ATAC QC finalization requires a count view, not normalized model states")
    for key in ("source_shape", "source_full_cell_ids", "source_full_depth",
            "source_counts_sha256", "source_depth_sha256", "source_full_cell_axis_sha256",
            "source_full_peak_axis_sha256", "denominator_kind"):
        if key not in meta or _digest(np.asarray(meta[key]).tolist()) != _digest(np.asarray(source_meta[key]).tolist()):
            raise ValueError("retained ATAC provenance differs from the verified full-input source")
    if not qc_adata.obs_names.is_unique or not qc_adata.var_names.is_unique:
        raise ValueError("retained ATAC axes must be unique")
    rows = prepared_source.obs_names.get_indexer(qc_adata.obs_names)
    columns = prepared_source.var_names.get_indexer(qc_adata.var_names)
    if (rows < 0).any() or (columns < 0).any() or RAW_LAYER not in qc_adata.layers:
        raise ValueError("retained ATAC axes/count layer absent from the verified source")
    counts = _counts(qc_adata.layers[RAW_LAYER])
    expected_counts = source_counts[rows][:, columns].tocsr()
    if _matrix_hash(counts) != _matrix_hash(expected_counts):
        raise ValueError("retained ATAC counts differ from the verified full-input source subset")
    depth = _aligned_source_depth(source_meta, qc_adata.obs_names)
    if DEPTH not in qc_adata.obs or not np.array_equal(np.asarray(qc_adata.obs[DEPTH], dtype=float), depth):
        raise ValueError("retained frozen ATAC depth differs from full-input source evidence")
    expected_depth_proof = [_row_proof(c, d, source_meta["source_counts_sha256"]) for c, d in zip(qc_adata.obs_names, depth)]
    if ROW_PROOF not in qc_adata.obs or list(qc_adata.obs[ROW_PROOF]) != expected_depth_proof:
        raise ValueError("retained frozen ATAC depth proofs were modified")
    out = qc_adata.copy()
    peak_hash = _axis_hash(out.var_names)
    out.obs[COUNT_PROOF] = _count_row_proofs(counts, out.obs_names, peak_hash, source_meta["source_counts_sha256"])
    out.uns[META] = copy.deepcopy(source_meta)
    out.uns[META].update(schema=QC_SCHEMA, checked_retained_peak_axis_sha256=peak_hash,
        retained_count_proof_column=COUNT_PROOF)
    return out


def prepare_atac_counts(raw):
    """Copy original X counts before QC; never infer depth from existing QC fields."""
    if not raw.obs_names.is_unique or not raw.var_names.is_unique:
        raise ValueError("ATAC depth source requires unique cell and peak axes")
    if META in raw.uns or RAW_LAYER in raw.layers or DEPTH in raw.obs or ROW_PROOF in raw.obs or COUNT_PROOF in raw.obs:
        raise ValueError("ATAC full-input depth already prepared; do not recompute after QC")
    counts = _counts(raw.X)
    if "raw" in raw.layers and _matrix_hash(_counts(raw.layers["raw"])) != _matrix_hash(counts):
        raise ValueError("existing ATAC raw layer conflicts with authoritative input X counts")
    out = raw.copy()
    out.layers[RAW_LAYER] = counts.copy()
    depth = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    if not np.isfinite(depth).all():
        raise ValueError("ATAC full-input depth sum overflow")
    source_hash = _matrix_hash(counts)
    out.obs[DEPTH] = depth
    out.obs[ROW_PROOF] = [_row_proof(cell, d, source_hash) for cell, d in zip(out.obs_names, depth)]
    out.obs[DEPTH+"_status"] = np.where(depth > 0, "ok", "zero_depth")
    diagnostics = dict(nCount_atac_present="nCount_atac" in raw.obs)
    if "nCount_atac" in raw.obs:
        try:
            existing = np.asarray(raw.obs["nCount_atac"], dtype=float)
            diagnostics.update(nCount_atac_nonfinite=int((~np.isfinite(existing)).sum()),
                nCount_atac_mismatches=int((~np.isclose(existing, depth, rtol=1e-8, atol=1e-8)).sum()))
        except (TypeError, ValueError):
            diagnostics["nCount_atac_not_numeric"] = True
    out.uns[META] = dict(schema=SOURCE_SCHEMA, source_shape=list(raw.shape),
        source_full_peak_axis_sha256=_axis_hash(raw.var_names),
        source_full_cell_axis_sha256=_axis_hash(raw.obs_names), source_counts_sha256=source_hash,
        source_full_cell_ids=list(map(str, raw.obs_names)), source_full_depth=depth.tolist(),
        source_depth_sha256=_digest(depth.tolist()), depth_column=DEPTH, raw_layer=RAW_LAYER,
        denominator_kind=DENOMINATOR_KIND, diagnostics=diagnostics)
    return out


def _validate_depth(adata, reject_zero=False):
    meta = adata.uns.get(META)
    if not isinstance(meta, dict) or meta.get("schema") not in (QC_SCHEMA, VIEW_SCHEMA):
        raise ValueError("missing prepared full-input ATAC depth provenance")
    if RAW_LAYER not in adata.layers or DEPTH not in adata.obs or ROW_PROOF not in adata.obs:
        raise ValueError("missing aligned ATAC count layer/frozen depth evidence")
    depth = np.asarray(adata.obs[DEPTH], dtype=np.float64)
    if not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError("invalid frozen full-input ATAC depth")
    if not np.array_equal(depth, _aligned_source_depth(meta, adata.obs_names)):
        raise ValueError("frozen ATAC depth was modified relative to full-input source evidence")
    expected = [_row_proof(cell, d, meta["source_counts_sha256"]) for cell, d in zip(adata.obs_names, depth)]
    if list(adata.obs[ROW_PROOF]) != expected:
        raise ValueError("frozen ATAC depth/cell axis was modified after full-input preparation")
    counts = _counts(adata.layers[RAW_LAYER])
    peak_hash = _axis_hash(adata.var_names)
    if (meta.get("checked_retained_peak_axis_sha256") != peak_hash
            or meta.get("retained_count_proof_column") != COUNT_PROOF):
        raise ValueError("ATAC peak axis changed; finalize QC with the verified original source")
    if COUNT_PROOF not in adata.obs or list(adata.obs[COUNT_PROOF]) != _count_row_proofs(
            counts, adata.obs_names, peak_hash, meta["source_counts_sha256"]):
        raise ValueError("retained ATAC count layer was modified after checked QC")
    if (np.asarray(counts.sum(axis=1)).ravel() > depth+1e-8).any():
        raise ValueError("retained ATAC counts exceed the frozen full-input denominator")
    if reject_zero and (depth == 0).any():
        raise ValueError("continuous ATAC fitting rejects retained zero-depth cells; review binary reference QC")
    return meta, depth, counts


def depth_log1p_model_view(qc_adata, settings):
    """Transform counts on retained axes once; D is never recomputed after subsetting."""
    if qc_adata.uns.get(META,{}).get("schema")==VIEW_SCHEMA:
        raise ValueError("ATAC model view already normalized; do not normalize twice")
    if qc_adata.uns.get(META, {}).get("schema") == SOURCE_SCHEMA:
        # Only the intact complete source can be finalized without an explicit
        # source argument. Never derive a new proof from an unchecked subset.
        qc_adata = finalize_atac_qc(qc_adata, qc_adata)
    meta, depth, counts = _validate_depth(qc_adata)
    out = qc_adata.copy()
    scale = np.divide(settings["target_sum"], depth, out=np.zeros(len(depth)), where=depth > 0)
    model = counts.multiply(scale[:, None]).tocsr()
    model.data = np.log1p(model.data)
    model.eliminate_zeros()
    out.X = model
    representation = copy.deepcopy(meta)
    representation.update(schema=VIEW_SCHEMA, unit=UNIT, target_sum=float(settings["target_sum"]),
        output_cap_expansion=float(settings["output_cap_expansion"]),
        transformation="log1p(target_sum * retained_input_count / frozen_full_input_depth)")
    out.uns[META] = representation
    return out


def atac_representation(adata):
    meta = getattr(adata,"uns",{}).get(META, {})
    if meta.get("schema") != VIEW_SCHEMA:
        return dict(unit="raw_atac", representation="untagged_legacy_input")
    meta, depth, counts = _validate_depth(adata, reject_zero=True)
    settings = dict(target_sum=float(meta["target_sum"]), output_cap_expansion=float(meta["output_cap_expansion"]))
    if not math.isfinite(settings["target_sum"]) or settings["target_sum"]<=0 or not math.isfinite(settings["output_cap_expansion"]) or settings["output_cap_expansion"]<0:
        raise ValueError("invalid continuous ATAC representation settings")
    expected = counts.multiply((settings["target_sum"]/depth)[:,None]).tocsr()
    expected.data = np.log1p(expected.data)
    actual = sparse.csr_matrix(adata.X, dtype=np.float64)
    difference = actual-expected
    if difference.nnz and not np.allclose(difference.data,0.,rtol=0.,atol=1e-7):
        raise ValueError("ATAC model X differs from declared frozen-depth log1p representation")
    values = actual.data
    if not np.isfinite(values).all() or (values<0).any() or (values>np.log1p(settings["target_sum"])+1e-7).any():
        raise ValueError("continuous ATAC model states outside declared physical ceiling")
    result = dict(unit=UNIT, schema=VIEW_SCHEMA, **settings,
        denominator_kind=meta["denominator_kind"], source_shape=np.asarray(meta["source_shape"]).tolist(),
        source_full_peak_axis_sha256=meta["source_full_peak_axis_sha256"],
        source_full_cell_axis_sha256=meta["source_full_cell_axis_sha256"],
        source_counts_sha256=meta["source_counts_sha256"], source_depth_sha256=meta["source_depth_sha256"],
        source_full_cell_ids=list(map(str, meta["source_full_cell_ids"])),
        source_full_depth=np.asarray(meta["source_full_depth"], dtype=float).tolist(),
        retained_counts_sha256=_matrix_hash(counts),
        retained_cell_depth_sha256=_digest([[str(c),float(d)] for c,d in zip(adata.obs_names,depth)]),
        retained_peak_axis_sha256=_axis_hash(adata.var_names))
    validate_representation_metadata(result, adata.obs_names, adata.var_names)
    return result


def validate_representation_metadata(representation, cell_ids, peak_ids):
    """Validate v2 provenance structurally, even when a registry was re-signed."""
    hashes = {"source_full_peak_axis_sha256", "source_full_cell_axis_sha256",
        "source_counts_sha256", "source_depth_sha256", "retained_cell_depth_sha256",
        "retained_peak_axis_sha256", "retained_counts_sha256"}
    fields = hashes | {"unit", "schema", "target_sum", "output_cap_expansion",
        "denominator_kind", "source_shape", "source_full_cell_ids", "source_full_depth"}
    if not isinstance(representation, dict) or set(representation) != fields:
        raise ValueError("continuous ATAC representation fields mismatch")
    if representation["unit"] != UNIT or representation["schema"] != VIEW_SCHEMA:
        raise ValueError("continuous ATAC representation schema/unit mismatch")
    validate_atac_depth_config({"atac_to_rna": {"model_mode": "saturating_hybrid"},
        "preprocess": {"atac_depth_normalization": {"enabled": True,
            "target_sum": representation["target_sum"],
            "output_cap_expansion": representation["output_cap_expansion"]}}})
    if representation["denominator_kind"] != DENOMINATOR_KIND:
        raise ValueError("continuous ATAC denominator source mismatch")
    for key in hashes:
        value = representation[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid continuous ATAC representation provenance hash")
    shape = representation["source_shape"]
    if not isinstance(shape, (list, tuple, np.ndarray)) or len(shape) != 2 or any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v <= 0 for v in shape):
        raise ValueError("invalid continuous ATAC full-input source shape")
    if len(cell_ids) > shape[0] or len(peak_ids) > shape[1]:
        raise ValueError("continuous ATAC retained axes exceed full-input source shape")
    if representation["retained_peak_axis_sha256"] != _axis_hash(peak_ids):
        raise ValueError("continuous ATAC retained peak axis provenance mismatch")
    depth = _aligned_source_depth(representation, cell_ids)
    if representation["retained_cell_depth_sha256"] != _digest([[str(c), float(d)] for c, d in zip(cell_ids, depth)]):
        raise ValueError("continuous ATAC retained depth provenance differs from full-input evidence")


def representation_for_config(adata, config):
    settings = validate_atac_depth_config(config)
    representation = atac_representation(adata)
    continuous = representation["unit"] == UNIT
    if continuous != settings["enabled"]:
        raise ValueError("ATAC depth-normalization config and declared model representation disagree")
    if continuous and any(representation[key] != settings[key] for key in ("target_sum", "output_cap_expansion")):
        raise ValueError("ATAC depth-normalization settings differ from the constructed model view")
    return representation


def atac_output_bounds(y, representation):
    if representation["unit"] != UNIT:
        return (0.,1.)
    y = np.asarray(y,dtype=float)
    ceiling = float(np.log1p(representation["target_sum"]))
    if not len(y) or not np.isfinite(y).all() or (y<0).any() or (y>ceiling+1e-7).any():
        raise ValueError("invalid WT continuous ATAC target states")
    return (0.,float(min(ceiling,y.max()+representation["output_cap_expansion"]*np.ptp(y))))


def expected_registry_schema(kind, config):
    layer = "atac_to_rna" if kind == "saturating_hybrid_peak_gene" else "rna_to_atac"
    if config.get("atac_to_rna", {}).get("model_mode") == "saturating_hybrid" and config.get(layer, {}).get("response_model", "saturation") == "nn":
        validate_atac_depth_config(config)
        return f"hybrid_nn_{'peak_gene' if layer == 'atac_to_rna' else 'tf_peak'}_registry_v1"
    if config.get("atac_to_rna", {}).get("model_mode") == "saturating_hybrid" and config.get(layer, {}).get("response_model", "saturation") == "spline":
        validate_atac_depth_config(config)
        return f"hybrid_spline_{'peak_gene' if layer == 'atac_to_rna' else 'tf_peak'}_registry_v2"
    version = 2 if validate_atac_depth_config(config)["enabled"] else 1
    return f"saturating_{'peak_gene' if kind=='saturating_hybrid_peak_gene' else 'tf_peak'}_registry_v{version}"


def atac_input_contract(adata):
    """Consumed model-input provenance; output-cap knobs belong to fit contracts only."""
    if getattr(adata,"uns",{}).get(META,{}).get("schema") != VIEW_SCHEMA:
        return None
    result = atac_representation(adata)
    return {k:v for k,v in result.items() if k!="output_cap_expansion"}
