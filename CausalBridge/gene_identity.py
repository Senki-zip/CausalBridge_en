"""Stable gene identity and GTF-backed annotation boundary.

This module deliberately does not alter pipeline configuration or model code.
Stable IDs are the only internal key; symbols and numbered labels are display
metadata.  The parser is streaming and the index fingerprint excludes paths.
"""
from dataclasses import dataclass
from functools import cached_property
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd


SCHEMA = "gene_identity_v2_strand_tss"
_ATTR_RE = re.compile(r"([A-Za-z][A-Za-z0-9_.-]*)\s+((?:\"(?:[^\"\\]|\\.)*\")|[^;\"\s]+)")
_VERSION_RE = re.compile(r"^(.*)\.[0-9]+$")


def normalize_gene_id(value: str) -> str:
    """Remove only a terminal numeric version suffix (``ID.12`` -> ``ID``)."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("gene ID must be a non-empty string")
    value = value.strip()
    match = _VERSION_RE.match(value)
    if match and not match.group(1):
        raise ValueError("gene ID cannot consist only of a version suffix")
    return match.group(1) if match else value


def _attrs(text: str) -> dict:
    """Singular values are strings; tag/ontology_term are sorted unique tuples.

    Only these declared GTF attributes may repeat with different values.
    Their order and duplicate multiplicity carry no identity semantics.
    """
    repeatable = frozenset(("tag", "ontology_term"))
    result = {}
    cursor = len(text) - len(text.lstrip(" \t"))
    while cursor < len(text):
        if text[cursor:].strip(" ;\t") == "":
            break
        match = _ATTR_RE.match(text, cursor)
        if match is None:
            raise ValueError(f"malformed GTF attributes near {text[cursor:]!r}")
        value = match.group(2)
        if value.startswith('"'):
            value = value[1:-1].replace(r'\"', '"').replace(r'\\', '\\')
        key = match.group(1)
        if key in repeatable:
            result.setdefault(key, set()).add(value)
        elif key in result and result[key] != value:
            raise ValueError(f"conflicting GTF attribute {key}")
        else:
            result[key] = value
        cursor = match.end()
        while cursor < len(text) and text[cursor] in " \t":
            cursor += 1
        if cursor < len(text) and text[cursor] != ";":
            raise ValueError(f"malformed GTF attributes near {text[cursor:]!r}")
        if cursor < len(text) and text[cursor] == ";":
            cursor += 1
            while cursor < len(text) and text[cursor] in " \t":
                cursor += 1
    return {key: tuple(sorted(value)) if key in repeatable else value
            for key, value in result.items()}


def _open_bytes(path: Path):
    raw = open(path, "rb")
    prefix = raw.read(2)
    raw.seek(0)
    if prefix == b"\x1f\x8b":
        raw.close()
        return gzip.open(path, "rb")
    return raw


def _hash_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GeneRecord:
    """Reference gene; attributes contain scalar strings or repeatable tuples.

    JSON serialization of tuple-valued tag/ontology_term attributes yields
    arrays. Canonical ID/name/coordinate mappings remain scalar.
    """
    gene_id: str
    reference_id: str
    gene_name: str
    chromosome: str
    start: int
    end: int
    source: str
    strand: str
    attributes: Mapping

    @property
    def gene_tss(self):
        return self.start if self.strand == "+" else self.end - 1


def gene_tss_coordinates(adata):
    """Zero-based TSS; legacy inputs without a TSS use their documented left anchor.

    GTF identity requires explicit strand/TSS. Unknown legacy strand is never
    inferred from the gene span. Genomic start/end always remain full spans.
    """
    import numpy as np
    strict = "gene_identity" in adata.uns
    if strict and ("gene_tss" not in adata.var or "strand" not in adata.var):
        raise ValueError("GTF identity requires explicit strand and gene_tss")
    if "gene_tss" in adata.var:
        values = adata.var["gene_tss"].to_numpy(dtype=float)
    elif "strand" in adata.var and adata.var["strand"].isin(["+", "-"]).all():
        values = np.where(adata.var["strand"] == "+", adata.var["start"], adata.var["end"] - 1).astype(float)
    else:
        values = adata.var["start"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("gene_tss must contain finite nonnegative integer coordinates")
    if strict:
        if not adata.var["strand"].isin(["+", "-"]).all():
            raise ValueError("GTF identity requires + or - strand")
        expected = np.where(adata.var["strand"] == "+", adata.var["start"], adata.var["end"] - 1)
        if not np.array_equal(values, expected):
            raise ValueError("gene_tss conflicts with strand and gene span")
    return values


@dataclass(frozen=True)
class GeneIdentityIndex:
    records: tuple
    symbol_to_ids: Mapping
    display_labels: Mapping
    metadata: Mapping
    fingerprint: str

    @property
    def ids(self):
        return tuple(r.gene_id for r in self.records)

    @cached_property
    def by_id(self):
        return {r.gene_id: r for r in self.records}

    @cached_property
    def aliases(self):
        return {label: gene_id for gene_id, label in self.display_labels.items()
                if label != self.by_id[gene_id].gene_name}

    @cached_property
    def casefold_symbols(self):
        symbols = {}
        for name, ids in self.symbol_to_ids.items():
            symbols.setdefault(name.casefold(), []).extend(ids)
        return {name: tuple(sorted(ids)) for name, ids in symbols.items()}


@dataclass(frozen=True)
class GeneIdentityResult:
    adata: object
    filter_report: pd.DataFrame
    identity: GeneIdentityIndex
    metadata: Mapping


@dataclass(frozen=True)
class TFMappingResult:
    """Ordered IDs with aligned input-token origins and biological base symbols."""
    candidate_ids: tuple
    origins: tuple
    base_symbols: tuple
    mapping_counts: Mapping


def _display_labels(records: Sequence[GeneRecord]) -> dict:
    groups = {}
    for record in records:
        groups.setdefault(record.gene_name, []).append(record.gene_id)
    labels = {}
    for name, ids in groups.items():
        ordered = sorted(set(ids))
        if len(ordered) == 1:
            labels[ordered[0]] = name
        else:
            width = max(2, len(str(len(ordered))))
            for pos, gene_id in enumerate(ordered, 1):
                labels[gene_id] = f"{name}_{pos:0{width}d}"
    literal_names = set(groups)
    by_id = {r.gene_id: r for r in records}
    reserved = set(labels.values()) | literal_names
    for gene_id in sorted(labels):
        record = by_id[gene_id]
        if len(groups[record.gene_name]) > 1 and labels[gene_id] in literal_names:
            candidate = labels[gene_id] + "__" + gene_id
            while candidate in reserved:
                candidate += "_"
            labels[gene_id] = candidate
            reserved.add(candidate)
    return labels


def parse_gene_gtf(path) -> GeneIdentityIndex:
    """Parse gene features from a plain or gzip GTF with stable provenance."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    seen = {}
    with _open_bytes(path) as handle:
        for line_number, raw in enumerate(handle, 1):
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"GTF is not UTF-8 at line {line_number}") from exc
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n\r").split("\t")
            if len(fields) != 9:
                raise ValueError(f"GTF line {line_number} must have 9 tab-separated fields")
            chromosome, source, feature, start, end, _, strand, _, attribute_text = fields
            if feature != "gene":
                continue
            if not chromosome:
                raise ValueError(f"GTF line {line_number} has empty chromosome")
            if not re.fullmatch(r"[0-9]+", start) or not re.fullmatch(r"[0-9]+", end):
                raise ValueError(f"GTF line {line_number} has invalid coordinate syntax")
            try:
                one_start, one_end = int(start), int(end)
            except ValueError as exc:
                raise ValueError(f"GTF line {line_number} has invalid coordinates") from exc
            if one_start < 1 or one_end < one_start:
                raise ValueError(f"GTF line {line_number} has invalid coordinates")
            if strand not in ("+", "-"):
                raise ValueError(f"GTF line {line_number} requires + or - gene strand")
            attrs = _attrs(attribute_text)
            raw_id = attrs.get("gene_id")
            name = attrs.get("gene_name")
            if not raw_id or not name:
                raise ValueError(f"GTF line {line_number} requires gene_id and gene_name")
            reference_id = raw_id.strip()
            gene_id = normalize_gene_id(reference_id)
            if not name.strip():
                raise ValueError(f"GTF line {line_number} requires a nonempty gene_name")
            record = GeneRecord(gene_id, reference_id, name.strip(), chromosome,
                                one_start - 1, one_end, source, strand, dict(attrs))
            prior = seen.get(gene_id)
            if prior is not None:
                prior_attrs = {k: v for k, v in prior.attributes.items() if k not in ("gene_id", "gene_version")}
                current_attrs = {k: v for k, v in attrs.items() if k not in ("gene_id", "gene_version")}
                comparable = (prior.gene_name, prior.chromosome, prior.start,
                              prior.end, prior.strand, prior.source, prior_attrs)
                current = (record.gene_name, record.chromosome, record.start,
                           record.end, record.strand, record.source, current_attrs)
                if comparable != current:
                    raise ValueError(f"conflicting records for normalized gene ID {gene_id}")
                continue
            records.append(record)
            seen[gene_id] = record
    if not records:
        raise ValueError("GTF contains no gene records")
    records = tuple(sorted(records, key=lambda r: r.gene_id))
    symbols = {}
    for record in records:
        symbols.setdefault(record.gene_name, []).append(record.gene_id)
    symbols = {k: tuple(sorted(v)) for k, v in sorted(symbols.items())}
    labels = _display_labels(records)
    mapping = [{"gene_id": r.gene_id, "gene_name": r.gene_name,
                "display_name": labels[r.gene_id], "chromosome": r.chromosome,
                "start": r.start, "end": r.end, "strand": r.strand,
                "gene_tss": r.gene_tss, "tss_source": "gtf_strand_zero_based"} for r in records]
    content = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    file_sha = _hash_bytes(path)
    fingerprint = hashlib.sha256(content.encode()).hexdigest()
    metadata = {"schema": SCHEMA, "reference_sha256": file_sha,
                "mapping_sha256": fingerprint,
                "fingerprint": fingerprint,
                "reference_path": str(path), "record_count": len(records),
                "canonical_map": mapping}
    return GeneIdentityIndex(records, symbols, labels, metadata, fingerprint)


def _allowed(index: GeneIdentityIndex, allowed_ids: Optional[Iterable[str]]):
    ids = set(index.ids if allowed_ids is None else (normalize_gene_id(v) for v in allowed_ids))
    unknown = ids - set(index.ids)
    if unknown:
        raise ValueError(f"allowed_ids contain unknown reference IDs: {sorted(unknown)}")
    return ids


def resolve_gene_token(token: str, index: GeneIdentityIndex, *,
                       allowed_ids=None, purpose="gene") -> str:
    """Resolve ID/version, numbered alias, then unique allowed base symbol."""
    if not isinstance(token, str) or not token.strip():
        raise ValueError(f"{purpose} token must be non-empty")
    token = token.strip()
    allowed = _allowed(index, allowed_ids)
    full = [token] if token in index.by_id else []
    if full and full[0] in allowed:
        return full[0]
    normalized = normalize_gene_id(token)
    if normalized in allowed:
        return normalized
    if normalized in index.by_id:
        raise ValueError(f"{purpose} gene ID {token!r} is not in allowed_ids")
    alias = [index.aliases[token]] if token in index.aliases and index.aliases[token] in allowed else []
    if len(alias) == 1:
        return alias[0]
    symbols = [gene_id for gene_id in index.casefold_symbols.get(token.casefold(), ()) if gene_id in allowed]
    if len(symbols) == 1:
        return symbols[0]
    if len(symbols) > 1:
        raise ValueError(f"ambiguous {purpose} symbol {token!r}; use a stable ID or numbered alias")
    raise ValueError(f"unresolved {purpose} token {token!r}")


def map_tf_symbols(symbols: Iterable[str], index: GeneIdentityIndex, *, allowed_ids=None) -> TFMappingResult:
    """Expand base symbols only; origins align with IDs, counts key input tokens.

    Repeated input symbols retain their ordered expansion. Numbered display
    aliases are never interpreted as motif symbols unless literally a base.
    """
    allowed = _allowed(index, allowed_ids)
    candidates, origins, bases, counts = [], [], [], {}
    for symbol in symbols:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("TF symbols must be non-empty strings")
        base = symbol.strip()
        ids = tuple(g for g in index.casefold_symbols.get(base.casefold(), ()) if g in allowed)
        if not ids:
            raise ValueError(f"unresolved TF symbol {symbol!r}")
        candidates.extend(ids)
        origins.extend([base] * len(ids))
        bases.extend(index.by_id[g].gene_name for g in ids)
        counts[base] = len(ids)
    return TFMappingResult(tuple(candidates), tuple(origins), tuple(bases), counts)


def gene_labels(ids: Iterable[str], index: GeneIdentityIndex, display=True):
    result = []
    for value in ids:
        gene_id = normalize_gene_id(value)
        if gene_id not in index.by_id:
            raise ValueError(f"unknown reference gene ID {value!r}")
        result.append(index.display_labels[gene_id] if display else index.by_id[gene_id].gene_name)
    return result


def _annotation_equal(left, right):
    if pd.isna(left) or pd.isna(right):
        return bool(pd.isna(left) and pd.isna(right))
    return left == right


def _validate_annotation_fields(frame, column, values):
    """Existing audit fields are evidence, never an overwrite destination."""
    for key, expected in values.items():
        if key not in frame:
            continue
        for actual, wanted in zip(frame[key], expected):
            if _annotation_equal(actual, wanted):
                continue
            if key == column + "_id" and isinstance(actual, str) and isinstance(wanted, str):
                if ",".join(normalize_gene_id(v) for v in actual.split(",")) == wanted:
                    continue
            if key.endswith("_json") and isinstance(actual, str) and isinstance(wanted, str):
                try:
                    if json.loads(actual) == json.loads(wanted):
                        continue
                except ValueError:
                    pass
            raise ValueError(f"conflicting existing identity field {key!r}")


def annotate_gene_columns(frame: pd.DataFrame, index: GeneIdentityIndex, *, columns) -> pd.DataFrame:
    """Present explicitly declared columns without grouping or altering rows.

    ``columns`` is a sequence (single IDs) or mapping column -> role:
    ``gene`` (single ID), ``genes`` (comma-separated IDs), or ``gene_or_sentinel``
    (single ID, PEAK:..., peak, joint_vector, or unknown). The exact lowercase
    peak is a non-gene legacy pathway sentinel only in this explicit role.
    Sentinel *_id/name fields retain that sentinel, never a biological ID.
    Original columns become
    display labels; *_id retains canonical IDs; *_name is the unsuffixed base.
    Multi-gene roles additionally emit *_ids_json, *_names_json,
    *_display_names_json and *_separator (',') in original token order.
    Nulls and explicitly permitted sentinels remain unchanged.
    """
    result = frame.copy(deep=True)
    if isinstance(columns, str):
        columns = [columns]
    roles = columns if isinstance(columns, Mapping) else {v: "gene" for v in columns}
    for column, role in roles.items():
        if role not in ("gene", "genes", "gene_or_sentinel"):
            raise ValueError(f"unsupported gene column role {role!r}")
        if column not in result:
            raise ValueError(f"missing gene ID column {column!r}")
        ids, names, displays, id_json, name_json, display_json = [], [], [], [], [], []
        for position, value in enumerate(result[column]):
            existing_id = result[column + "_id"].iloc[position] if column + "_id" in result else value
            if pd.isna(value) or (role == "gene_or_sentinel" and isinstance(value, str)
                                 and (value.startswith("PEAK:") or value in ("peak", "joint_vector", "unknown"))):
                ids.append(value)
                names.append(value)
                displays.append(value)
                id_json.append(None)
                name_json.append(None)
                display_json.append(None)
                continue
            tokens = existing_id.split(",") if role == "genes" and isinstance(existing_id, str) else [existing_id]
            canonical = [normalize_gene_id(v) for v in tokens]
            if any(g not in index.by_id for g in canonical):
                raise ValueError(f"unknown gene ID {value!r} in {column}")
            base = [index.by_id[g].gene_name for g in canonical]
            labels = [index.display_labels[g] for g in canonical]
            if column + "_id" in result:
                visible = value.split(",") if role == "genes" and isinstance(value, str) else [value]
                if len(visible) != len(canonical):
                    raise ValueError(f"conflicting existing identity field {column!r}")
                for token, gene_id, name, label in zip(visible, canonical, base, labels):
                    is_id = isinstance(token, str) and normalize_gene_id(token) == gene_id
                    unique_base = token == name and index.symbol_to_ids.get(name) == (gene_id,)
                    if not (is_id or token == label or unique_base):
                        raise ValueError(f"conflicting existing identity field {column!r} versus {column + '_id'!r}")
            ids.append(",".join(canonical))
            names.append(",".join(base))
            displays.append(",".join(labels))
            id_json.append(json.dumps(canonical))
            name_json.append(json.dumps(base))
            display_json.append(json.dumps(labels))
        fields = {f"{column}_id": ids, f"{column}_name": names}
        if role == "genes":
            fields.update({f"{column}_ids_json": id_json, f"{column}_names_json": name_json,
                           f"{column}_display_names_json": display_json,
                           f"{column}_separator": [","] * len(result)})
        _validate_annotation_fields(result, column, fields)
        result[column] = displays
        for key, values in fields.items():
            result[key] = values
    return result


def apply_gene_identity(rna, index: GeneIdentityIndex, *, gene_id_column="gene_ids") -> GeneIdentityResult:
    """Filter and annotate an AnnData copy, preserving sparse matrices/layers/raw."""
    if gene_id_column not in rna.var:
        raise ValueError(f"RNA var is missing required {gene_id_column!r}")
    original_ids = list(rna.var[gene_id_column])
    normalized = [normalize_gene_id(value) for value in original_ids]
    if "gene_id" in rna.var and [normalize_gene_id(v) for v in rna.var["gene_id"]] != normalized:
        raise ValueError("conflicting existing gene_id versus input gene ID column")
    if len(set(normalized)) != len(normalized):
        raise ValueError("RNA gene IDs collide after version normalization")
    reference = index.by_id
    prior_identity = rna.uns.get("gene_identity")
    if prior_identity is not None:
        expected = {"schema": SCHEMA, "fingerprint": index.fingerprint,
                    "reference_sha256": index.metadata["reference_sha256"]}
        if not isinstance(prior_identity, Mapping) or any(prior_identity.get(k) != v for k, v in expected.items()):
            raise ValueError("existing gene identity provenance conflicts with reference")
        required = ("original_gene_id", "original_gene_name", "original_var_name", "gene_id",
                    "gene_name", "gene_display_name", "chr", "start", "end", "strand", "gene_tss")
        if any(k not in rna.var for k in required) or list(rna.var_names) != normalized:
            raise ValueError("existing gene identity annotation is incomplete or inconsistent")
        for pos, gene_id in enumerate(normalized):
            record = reference.get(gene_id)
            row = rna.var.iloc[pos]
            if record is None or normalize_gene_id(row["original_gene_id"]) != gene_id or (
                row["gene_name"], row["gene_display_name"], row["chr"], row["start"], row["end"], row["strand"], row["gene_tss"]) != (
                    record.gene_name, index.display_labels[gene_id], record.chromosome, record.start, record.end, record.strand, record.gene_tss):
                raise ValueError(f"existing gene identity annotation conflicts for {gene_id}")
    rows = []
    keep = []
    for pos, (original, gene_id) in enumerate(zip(original_ids, normalized)):
        if gene_id not in reference:
            rows.append({"original_gene_id": original, "gene_id": gene_id,
                         "gene_name": None, "display_name": None,
                         "included": False, "reason": "not_in_reference", "original_index": pos})
            continue
        record = reference[gene_id]
        keep.append(pos)
        rows.append({"original_gene_id": original, "gene_id": gene_id,
                     "gene_name": record.gene_name, "display_name": index.display_labels[gene_id],
                     "included": True, "reason": "reference_match", "original_index": pos})
    if not keep:
        raise ValueError("RNA contains no genes present in reference")
    result = rna.copy()
    result = result[:, keep].copy()
    old_var = rna.var.iloc[keep].copy()
    result.var = old_var
    records = [reference[normalized[pos]] for pos in keep]
    for column in ("gene_name", "chr", "start", "end"):
        if column in old_var:
            audit = "original_" + ("gene_name" if column == "gene_name" else column)
            if audit in old_var:
                if prior_identity is None:
                    raise ValueError(f"existing audit column {audit!r} would be overwritten")
            elif prior_identity is None:
                result.var[audit] = old_var[column].copy()
    if "original_gene_id" in old_var and prior_identity is None:
        raise ValueError("existing original_gene_id audit column would be overwritten")
    result.var[gene_id_column] = [r.gene_id for r in records]
    result.var["gene_id"] = [r.gene_id for r in records]
    result.var["gene_name"] = [r.gene_name for r in records]
    result.var["gene_display_name"] = [index.display_labels[r.gene_id] for r in records]
    if prior_identity is None:
        result.var["original_gene_id"] = [original_ids[pos] for pos in keep]
    if "original_var_name" in old_var and prior_identity is None:
        raise ValueError("existing original_var_name audit column would be overwritten")
    if prior_identity is None:
        result.var["original_var_name"] = list(rna.var_names[keep])
    if "original_gene_name" not in result.var:
        result.var["original_gene_name"] = list(map(str, rna.var_names[keep]))
    result.var["chr"] = [r.chromosome for r in records]
    result.var["start"] = [r.start for r in records]
    result.var["end"] = [r.end for r in records]
    result.var["strand"] = [r.strand for r in records]
    result.var["gene_tss"] = [r.gene_tss for r in records]
    result.var["tss_source"] = "gtf_strand_zero_based"
    result.var["reference_chr"] = [r.chromosome for r in records]
    result.var["reference_start"] = [r.start for r in records]
    result.var["reference_end"] = [r.end for r in records]
    result.var_names = pd.Index([r.gene_id for r in records])
    result.uns = dict(result.uns)
    result.uns["gene_identity"] = {"schema": SCHEMA, "fingerprint": index.fingerprint,
                                    "reference_sha256": index.metadata["reference_sha256"],
                                    "canonical_map": json.dumps(index.metadata["canonical_map"],
                                                                  sort_keys=True, separators=(",", ":"))}
    if rna.raw is not None:
        raw = rna.raw.to_adata()
        if gene_id_column not in raw.var:
            if not all(normalize_gene_id(v) in reference for v in raw.var_names):
                raise ValueError("raw RNA requires a gene ID column or reference-ID var_names")
            raw.var[gene_id_column] = list(raw.var_names)
        try:
            result.raw = apply_gene_identity(raw, index, gene_id_column=gene_id_column).adata
        except ValueError as exc:
            raise ValueError(f"raw {exc}") from exc
    report = pd.DataFrame(rows, columns=["original_gene_id", "gene_id", "gene_name", "display_name",
                                         "included", "reason", "original_index"])
    metadata = {"schema": SCHEMA, "fingerprint": index.fingerprint,
                "reference_sha256": index.metadata["reference_sha256"],
                "input_gene_count": len(original_ids), "retained_gene_count": len(keep),
                "excluded_gene_count": len(original_ids) - len(keep)}
    return GeneIdentityResult(result, report, index, metadata)
