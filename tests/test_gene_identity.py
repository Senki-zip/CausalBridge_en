import gzip
import hashlib
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from CausalBridge.gene_identity import (
    SCHEMA, annotate_gene_columns, apply_gene_identity, gene_labels,
    map_tf_symbols, normalize_gene_id, parse_gene_gtf, resolve_gene_token,
)


GTF = """## test reference
1\tref\tgene\t101\t200\t.\t+\t.\tgene_id \"ENSG00000000001.1\"; gene_name \"A\"; description \"alpha gene\";
1\tref\tgene\t301\t450\t.\t-\t.\tgene_id \"ENSG00000000002\"; gene_name \"A\"; description \"second alpha\";
2\tref\tgene\t501\t550\t.\t+\t.\tgene_id \"ENSG00000000003.9\"; gene_name \"B\";
2\tref\tgene\t601\t650\t.\t+\t.\tgene_id \"ENSG00000000004\"; gene_name \"C\";
"""


def write_gtf(tmp_path, text=GTF, name="genes.gtf"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_parser_coordinates_collision_labels_and_content_fingerprint(tmp_path):
    index = parse_gene_gtf(write_gtf(tmp_path))
    assert index.ids == ("ENSG00000000001", "ENSG00000000002", "ENSG00000000003", "ENSG00000000004")
    assert index.by_id["ENSG00000000001"].start == 100
    assert index.by_id["ENSG00000000001"].end == 200
    assert index.symbol_to_ids["A"] == ("ENSG00000000001", "ENSG00000000002")
    assert index.display_labels["ENSG00000000001"] == "A_01"
    assert index.display_labels["ENSG00000000002"] == "A_02"
    assert index.metadata["schema"] == SCHEMA
    assert str(tmp_path) in index.metadata["reference_path"]
    assert hashlib.sha256(write_gtf(tmp_path).read_bytes()).hexdigest() == index.metadata["reference_sha256"]


def test_gzip_same_content_and_path_not_identity(tmp_path):
    first = write_gtf(tmp_path, name="first.gtf")
    second = tmp_path / "second.gtf.gz"
    with gzip.open(second, "wb") as handle:
        handle.write(GTF.encode())
    a, b = parse_gene_gtf(first), parse_gene_gtf(second)
    assert a.fingerprint == b.fingerprint
    assert a.metadata["reference_sha256"] != b.metadata["reference_sha256"]
    assert a.metadata["mapping_sha256"] == b.metadata["mapping_sha256"]
    copied = tmp_path / "copied.gtf"
    copied.write_bytes(first.read_bytes())
    assert parse_gene_gtf(copied).fingerprint == a.fingerprint


def test_duplicate_normalized_same_record_is_consumed_and_conflict_rejected(tmp_path):
    duplicate = GTF + "1\tref\tgene\t101\t200\t.\t+\t.\tgene_id \"ENSG00000000001.7\"; gene_name \"A\"; description \"alpha gene\";\n"
    assert len(parse_gene_gtf(write_gtf(tmp_path, duplicate)).records) == 4
    conflict = GTF + "1\tref\tgene\t102\t200\t.\t+\t.\tgene_id \"ENSG00000000001.7\"; gene_name \"A\";\n"
    with pytest.raises(ValueError, match="conflicting"):
        parse_gene_gtf(write_gtf(tmp_path, conflict))


@pytest.mark.parametrize("text,error", [
    ("1\tref\tgene\t0\t2\t.\t+\t.\tgene_id \"x\"; gene_name \"X\";\n", "coordinates"),
    ("1\tref\tgene\t1\t2\t.\t+\t.\tgene_id \"x\";\n", "requires"),
    ("1 ref gene 1 2 . + . gene_id \"x\"; gene_name \"X\";\n", "9"),
    ("1\tref\tgene\t1\t2\t.\t+\t.\tgene_id \"x\" gene_name \"X\";\n", "malformed"),
])
def test_parser_rejects_bad_coordinates_attributes_and_shape(tmp_path, text, error):
    with pytest.raises(ValueError, match=error):
        parse_gene_gtf(write_gtf(tmp_path, text))


def test_terminal_normalization_only_numeric_suffix():
    assert normalize_gene_id("ENSG.12") == "ENSG"
    assert normalize_gene_id("A_01") == "A_01"
    with pytest.raises(ValueError):
        normalize_gene_id(3)


def test_resolution_precedence_aliases_and_ambiguity(tmp_path):
    index = parse_gene_gtf(write_gtf(tmp_path))
    assert resolve_gene_token("ENSG00000000003.2", index) == "ENSG00000000003"
    assert resolve_gene_token("A_02", index) == "ENSG00000000002"
    assert resolve_gene_token("B", index) == "ENSG00000000003"
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_gene_token("A", index)
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_gene_token("A", index, allowed_ids=["ENSG00000000001", "ENSG00000000002"])
    assert resolve_gene_token("A", index, allowed_ids=["ENSG00000000001"]) == "ENSG00000000001"
    with pytest.raises(ValueError, match="unresolved"):
        resolve_gene_token("missing", index, purpose="target")


def test_tf_mapping_expands_all_allowed_ids_and_preserves_origins(tmp_path):
    index = parse_gene_gtf(write_gtf(tmp_path))
    result = map_tf_symbols(["a", "B"], index, allowed_ids=index.ids)
    assert result.candidate_ids == ("ENSG00000000001", "ENSG00000000002", "ENSG00000000003")
    assert result.origins == ("a", "a", "B")
    assert result.base_symbols == ("A", "A", "B")
    assert result.mapping_counts == {"a": 2, "B": 1}
    with pytest.raises(ValueError, match="unresolved"):
        map_tf_symbols(["not_a_gene"], index)


def test_labels_are_stable_for_subsets_and_row_order(tmp_path):
    index = parse_gene_gtf(write_gtf(tmp_path))
    assert gene_labels(["ENSG00000000002", "ENSG00000000001"], index) == ["A_02", "A_01"]
    assert gene_labels(["ENSG00000000002"], index) == ["A_02"]
    assert gene_labels(["ENSG00000000001"], index, display=False) == ["A"]


def make_rna(index, raw=True):
    var = pd.DataFrame({"gene_ids": ["ENSG00000000002.4", "ENSG00000000001", "UNANNOTATED"],
                        "gene_name": ["wrong", "A", "C"]}, index=["old2", "old1", "oldc"])
    rna = ad.AnnData(sparse.csr_matrix([[1, 2, 3], [4, 5, 6]]),
                     obs=pd.DataFrame(index=["c1", "c2"]), var=var)
    rna.layers["counts"] = sparse.csr_matrix([[10, 20, 30], [40, 50, 60]])
    rna.obsm["embedding"] = np.ones((2, 2))
    if raw:
        raw_var = pd.DataFrame({"gene_ids": ["ENSG00000000001", "ENSG00000000002.2", "OLD"]},
                               index=["r1", "r2", "r3"])
        raw = ad.AnnData(sparse.csr_matrix([[7, 8, 9], [10, 11, 12]]),
                         obs=pd.DataFrame(index=["c1", "c2"]), var=raw_var)
        rna.raw = raw
    return rna


def test_apply_filters_all_ids_sparse_layers_raw_and_authoritative_coordinates(tmp_path):
    index = parse_gene_gtf(write_gtf(tmp_path))
    rna = make_rna(index)
    before = rna.X.copy()
    result = apply_gene_identity(rna, index)
    out = result.adata
    assert list(out.var_names) == ["ENSG00000000002", "ENSG00000000001"]
    assert list(out.var["gene_display_name"]) == ["A_02", "A_01"]
    assert list(out.var["reference_start"]) == [300, 100]
    assert sparse.issparse(out.X) and sparse.issparse(out.layers["counts"])
    np.testing.assert_array_equal(out.layers["counts"].toarray(), [[10, 20], [40, 50]])
    assert out.raw is not None and list(out.raw.var_names) == ["ENSG00000000001", "ENSG00000000002"]
    assert result.filter_report.shape[0] == 3
    assert result.filter_report.query("included == False").iloc[0].reason == "not_in_reference"
    np.testing.assert_array_equal(rna.X.toarray(), before.toarray())
    assert "gene_identity" in out.uns and out.uns["gene_identity"]["schema"] == SCHEMA


def test_apply_duplicate_normalized_ids_and_raw_no_match_fail(tmp_path):
    index = parse_gene_gtf(write_gtf(tmp_path))
    rna = make_rna(index, raw=False)
    rna.var.loc["old1", "gene_ids"] = "ENSG00000000002.99"
    with pytest.raises(ValueError, match="collide"):
        apply_gene_identity(rna, index)
    rna = make_rna(index, raw=True)
    rna.raw = ad.AnnData(sparse.csr_matrix(np.ones((2, 1))), obs=rna.obs.copy(),
                         var=pd.DataFrame({"gene_ids": ["UNKNOWN"]}, index=["x"]))
    with pytest.raises(ValueError, match="raw RNA contains no"):
        apply_gene_identity(rna, index)


def test_annotation_is_copy_and_keeps_distinct_rows(tmp_path):
    index = parse_gene_gtf(write_gtf(tmp_path))
    frame = pd.DataFrame({"source_gene": ["ENSG00000000001", "ENSG00000000002"], "value": [1, 2]})
    annotated = annotate_gene_columns(frame, index, columns=["source_gene"])
    assert len(annotated) == 2
    assert list(annotated.source_gene_name) == ["A", "A"]
    assert list(annotated.source_gene) == ["A_01", "A_02"]
    assert list(annotated.source_gene_id) == list(frame.source_gene)
    assert "source_gene_name" not in frame
    with pytest.raises(ValueError, match="missing"):
        annotate_gene_columns(frame, index, columns=["gene"])


def test_identity_metadata_survives_h5ad_roundtrip(tmp_path):
    index = parse_gene_gtf(write_gtf(tmp_path))
    result = apply_gene_identity(make_rna(index), index)
    output = tmp_path / "identity.h5ad"
    result.adata.write_h5ad(output)
    loaded = ad.read_h5ad(output)
    assert loaded.uns["gene_identity"]["fingerprint"] == index.fingerprint
    assert loaded.uns["gene_identity"]["reference_sha256"] == index.metadata["reference_sha256"]
    assert json.loads(loaded.uns["gene_identity"]["canonical_map"]) == index.metadata["canonical_map"]


def test_literal_label_and_generated_labels_remain_distinct(tmp_path):
    text = GTF + "3\tref\tgene\t1\t2\t.\t+\t.\tgene_id \"ENSG00000000005\"; gene_name \"A_01\";\n"
    index = parse_gene_gtf(write_gtf(tmp_path, text))
    assert index.display_labels["ENSG00000000005"] == "A_01"
    assert index.display_labels["ENSG00000000001"] == "A_01__ENSG00000000001"
    assert index.display_labels["ENSG00000000002"] == "A_02"
    assert len(set(index.display_labels.values())) == 5
    assert resolve_gene_token("A_01", index) == "ENSG00000000005"
    assert resolve_gene_token("A_01__ENSG00000000001", index) == "ENSG00000000001"


@pytest.mark.parametrize("value,expected", [("ID.4", "ID"), ("ID.abc", "ID.abc"), ("ID.2.x", "ID.2.x"), ("ID.2.3", "ID.2"), ("ID_01", "ID_01")])
def test_id_normalization_is_terminal_only(value, expected):
    assert normalize_gene_id(value) == expected


def test_quoted_attributes_support_semicolon_and_escaped_quotes(tmp_path):
    text = '1\tref\tgene\t1\t10\t.\t+\t.\tgene_id "X"; gene_name "Alpha name"; note "has; semicolon and \\"quote\\"";\n'
    record = parse_gene_gtf(write_gtf(tmp_path, text)).records[0]
    assert record.gene_name == "Alpha name"
    assert record.attributes["note"] == 'has; semicolon and "quote"'


@pytest.mark.parametrize("text", ["", "# only header\n", '1\tr\texon\t1\t4\t.\t+\t.\tgene_id "X";\n'])
def test_empty_reference_fails(tmp_path, text):
    with pytest.raises(ValueError, match="no gene"):
        parse_gene_gtf(write_gtf(tmp_path, text))


def test_full_reference_labels_survive_reference_and_input_shuffle(tmp_path):
    index = parse_gene_gtf(write_gtf(tmp_path))
    shuffled = parse_gene_gtf(write_gtf(tmp_path, "\n".join(reversed(GTF.splitlines())) + "\n", "shuffle.gtf"))
    assert index.display_labels == shuffled.display_labels
    assert index.fingerprint == shuffled.fingerprint
    rna = make_rna(index, raw=False)[:, [1, 0, 2]].copy()
    out = apply_gene_identity(rna, index).adata
    assert list(out.var.gene_display_name) == ["A_01", "A_02"]
    assert apply_gene_identity(rna[:, [1]].copy(), index).adata.var.gene_display_name.iloc[0] == "A_02"


def test_coordinates_audit_and_var_aligned_matrices(tmp_path):
    index = parse_gene_gtf(write_gtf(tmp_path))
    rna = make_rna(index, raw=False)
    rna.var["chr"] = ["old"] * 3
    rna.var["start"] = [1, 2, 3]
    rna.var["end"] = [10, 20, 30]
    rna.varm["features"] = np.arange(6).reshape(3, 2)
    rna.varp["graph"] = sparse.csr_matrix(np.arange(9).reshape(3, 3))
    out = apply_gene_identity(rna, index).adata
    assert list(out.var.chr) == ["1", "1"]
    assert list(out.var.start) == [300, 100]
    assert list(out.var.original_start) == [1, 2]
    assert list(out.var.original_gene_name) == ["wrong", "A"]
    np.testing.assert_array_equal(out.varm["features"], rna.varm["features"][:2])
    np.testing.assert_array_equal(out.varp["graph"].toarray(), rna.varp["graph"][:2, :2].toarray())
    assert list(rna.var.start) == [1, 2, 3]


def test_raw_counts_and_annotations_follow_independent_axis(tmp_path):
    index = parse_gene_gtf(write_gtf(tmp_path))
    rna = make_rna(index)
    out = apply_gene_identity(rna, index).adata
    np.testing.assert_array_equal(out.raw.X.toarray(), [[7, 8], [10, 11]])
    assert list(out.raw.var.gene_ids) == ["ENSG00000000001", "ENSG00000000002"]
    assert list(out.raw.var.start) == [100, 300]
    assert list(out.raw.obs_names) == list(rna.obs_names)
    assert list(rna.raw.var_names) == ["r1", "r2", "r3"]


def test_sparse_adapter_never_densifies(tmp_path, monkeypatch):
    index = parse_gene_gtf(write_gtf(tmp_path))
    rna = make_rna(index)
    def forbidden(*args, **kwargs):
        raise AssertionError("densification")
    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbidden)
    out = apply_gene_identity(rna, index).adata
    assert out.X.sum() == 12
    assert out.layers["counts"].sum() == 120
    assert out.raw.X.sum() == 36


def test_tf_casefold_expansion_does_not_overwrite_distinct_bases(tmp_path):
    text = GTF + '3\tr\tgene\t1\t2\t.\t+\t.\tgene_id "E5"; gene_name "a";\n'
    index = parse_gene_gtf(write_gtf(tmp_path, text))
    assert set(map_tf_symbols(["a"], index).candidate_ids) == {"E5", "ENSG00000000001", "ENSG00000000002"}
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_gene_token("a", index)
    with pytest.raises(ValueError, match="unresolved"):
        map_tf_symbols(["A_01"], index)


def test_explicit_output_roles_multiko_sentinels_and_unknown_ids(tmp_path):
    index = parse_gene_gtf(write_gtf(tmp_path))
    frame = pd.DataFrame({"target_gene": ["ENSG00000000002,ENSG00000000001"],
                          "affected_gene": ["ENSG00000000003"], "arbitrary": ["not_a_gene"]})
    out = annotate_gene_columns(frame, index, columns={"target_gene": "genes", "affected_gene": "gene"})
    assert out.target_gene.iloc[0] == "A_02,A_01"
    assert json.loads(out.target_gene_ids_json.iloc[0]) == ["ENSG00000000002", "ENSG00000000001"]
    assert json.loads(out.target_gene_names_json.iloc[0]) == ["A", "A"]
    assert json.loads(out.target_gene_display_names_json.iloc[0]) == ["A_02", "A_01"]
    assert out.target_gene_separator.iloc[0] == ","
    assert out.arbitrary.iloc[0] == "not_a_gene"
    sentinels = pd.DataFrame({"gene": ["PEAK:p", "joint_vector", "unknown", None]})
    assert annotate_gene_columns(sentinels, index, columns={"gene": "gene_or_sentinel"}).gene.tolist() == sentinels.gene.tolist()
    with pytest.raises(ValueError, match="unknown gene ID"):
        annotate_gene_columns(pd.DataFrame({"gene": ["A"]}), index, columns=["gene"])


def test_missing_ids_invalid_raw_and_no_match_fail_without_mutation(tmp_path):
    index = parse_gene_gtf(write_gtf(tmp_path))
    rna = make_rna(index, raw=False)
    with pytest.raises(ValueError, match="missing required"):
        apply_gene_identity(rna, index, gene_id_column="absent")
    with pytest.raises(ValueError, match="no genes"):
        apply_gene_identity(rna[:, [2]].copy(), index)
    rna.raw = ad.AnnData(sparse.csr_matrix([[1], [2]]), obs=rna.obs.copy(), var=pd.DataFrame(index=["A"]))
    with pytest.raises(ValueError, match="raw RNA requires"):
        apply_gene_identity(rna, index)
    assert list(rna.var_names) == ["old2", "old1", "oldc"]


def test_conflicting_duplicate_biological_attributes_rejected(tmp_path):
    text = GTF + '1\tref\tgene\t101\t200\t.\t+\t.\tgene_id "ENSG00000000001.2"; gene_name "A"; description "different gene";\n'
    with pytest.raises(ValueError, match="conflicting"):
        parse_gene_gtf(write_gtf(tmp_path, text))


def test_filter_report_all_input_features_and_original_names(tmp_path):
    index = parse_gene_gtf(write_gtf(tmp_path))
    result = apply_gene_identity(make_rna(index, raw=False), index)
    assert result.filter_report.original_index.tolist() == [0, 1, 2]
    assert result.filter_report.included.tolist() == [True, True, False]
    assert result.metadata["input_gene_count"] == 3
    assert result.metadata["retained_gene_count"] == 2
    assert result.metadata["excluded_gene_count"] == 1
    assert result.adata.var.original_var_name.tolist() == ["old2", "old1"]
    assert result.adata.var.original_gene_id.tolist() == ["ENSG00000000002.4", "ENSG00000000001"]


@pytest.mark.parametrize("start,end", [("1_0", "20"), ("1.0", "2"), ("2", "1"), ("-1", "4")])
def test_invalid_coordinate_syntax_and_order(tmp_path, start, end):
    text = f'1\tr\tgene\t{start}\t{end}\t.\t+\t.\tgene_id "X"; gene_name "X";\n'
    with pytest.raises(ValueError, match="coordinate"):
        parse_gene_gtf(write_gtf(tmp_path, text))


def test_unterminated_attribute_quote_rejected(tmp_path):
    text = '1\tr\tgene\t1\t2\t.\t+\t.\tgene_id "X; gene_name "X";\n'
    with pytest.raises(ValueError, match="malformed"):
        parse_gene_gtf(write_gtf(tmp_path, text))


def test_explicit_excluded_id_is_not_resolved_as_a_symbol(tmp_path):
    index = parse_gene_gtf(write_gtf(tmp_path))
    with pytest.raises(ValueError, match="not in allowed_ids"):
        resolve_gene_token("ENSG00000000001.2", index, allowed_ids=["ENSG00000000002"])


@pytest.mark.parametrize("field,value", [("gene_id", "ENSG00000000002"), ("gene_name", "wrong")])
def test_output_existing_identity_conflicts_rejected(tmp_path, field, value):
    index = parse_gene_gtf(write_gtf(tmp_path))
    frame = pd.DataFrame({"gene": ["ENSG00000000001"], field: [value]})
    before = frame.copy(deep=True)
    with pytest.raises(ValueError, match="conflicting"):
        annotate_gene_columns(frame, index, columns=["gene"])
    pd.testing.assert_frame_equal(frame, before)


def test_annotation_idempotence_and_correct_existing_fields(tmp_path):
    index = parse_gene_gtf(write_gtf(tmp_path))
    frame = pd.DataFrame({"gene": ["ENSG00000000001.8"], "gene_id": ["ENSG00000000001.9"], "gene_name": ["A"]})
    first = annotate_gene_columns(frame, index, columns=["gene"])
    second = annotate_gene_columns(first, index, columns=["gene"])
    pd.testing.assert_frame_equal(first, second)
    assert second.gene.iloc[0] == "A_01" and second.gene_id.iloc[0] == "ENSG00000000001"
    ambiguous = pd.DataFrame({"gene": ["A"], "gene_id": ["ENSG00000000001"]})
    with pytest.raises(ValueError, match="conflicting"):
        annotate_gene_columns(ambiguous, index, columns=["gene"])


def test_multigene_annotation_repeated_and_structured_fields_checked(tmp_path):
    index = parse_gene_gtf(write_gtf(tmp_path))
    first = annotate_gene_columns(pd.DataFrame({"target": ["ENSG00000000002,ENSG00000000001"]}), index, columns={"target": "genes"})
    pd.testing.assert_frame_equal(first, annotate_gene_columns(first, index, columns={"target": "genes"}))
    first["target_ids_json"] = '["ENSG00000000001", "ENSG00000000002"]'
    with pytest.raises(ValueError, match="conflicting existing identity field"):
        annotate_gene_columns(first, index, columns={"target": "genes"})


@pytest.mark.parametrize("token", ["peak", "Peak", "PEAK"])
def test_peak_sentinel_is_exact_and_explicit(tmp_path, token):
    index = parse_gene_gtf(write_gtf(tmp_path))
    frame = pd.DataFrame({"tf_name": [token]})
    if token == "peak":
        out = annotate_gene_columns(frame, index, columns={"tf_name": "gene_or_sentinel"})
        assert out.tf_name.iloc[0] == "peak" and out.tf_name_id.iloc[0] == "peak"
        pd.testing.assert_frame_equal(out, annotate_gene_columns(out, index, columns={"tf_name": "gene_or_sentinel"}))
    else:
        with pytest.raises(ValueError, match="unknown gene ID"):
            annotate_gene_columns(frame, index, columns={"tf_name": "gene_or_sentinel"})
    with pytest.raises(ValueError, match="unknown gene ID"):
        annotate_gene_columns(frame, index, columns={"tf_name": "gene"})


def test_input_gene_id_conflict_before_filter_and_matching_version_allowed(tmp_path):
    index = parse_gene_gtf(write_gtf(tmp_path))
    rna = make_rna(index, raw=False)
    rna.var["gene_id"] = ["ENSG00000000002.7", "ENSG00000000001", "OTHER"]
    before = rna.var.copy(deep=True)
    with pytest.raises(ValueError, match="conflicting existing gene_id"):
        apply_gene_identity(rna, index)
    pd.testing.assert_frame_equal(rna.var, before)
    rna.var.loc["oldc", "gene_id"] = "UNANNOTATED"
    result = apply_gene_identity(rna, index)
    assert list(result.adata.var.gene_id) == ["ENSG00000000002", "ENSG00000000001"]


def test_reapply_identity_preserves_all_original_audits_and_raw(tmp_path):
    index = parse_gene_gtf(write_gtf(tmp_path))
    rna = make_rna(index)
    rna.var["chr"], rna.var["start"], rna.var["end"] = "old", [1, 2, 3], [4, 5, 6]
    first = apply_gene_identity(rna, index).adata
    second = apply_gene_identity(first, index).adata
    pd.testing.assert_frame_equal(first.var, second.var)
    pd.testing.assert_frame_equal(first.raw.var, second.raw.var)
    np.testing.assert_array_equal(first.X.toarray(), second.X.toarray())
    np.testing.assert_array_equal(first.raw.X.toarray(), second.raw.X.toarray())
    assert second.var.original_gene_id.tolist() == ["ENSG00000000002.4", "ENSG00000000001"]
    assert second.var.original_var_name.tolist() == ["old2", "old1"]
    first.var.loc["ENSG00000000001", "original_gene_id"] = "ENSG00000000002"
    with pytest.raises(ValueError, match="annotation conflicts"):
        apply_gene_identity(first, index)


@pytest.mark.parametrize("compressed", [False, True])
def test_gencode_repeatable_tags_line121_plain_and_gzip(tmp_path, compressed):
    line = ('chr1\tHAVANA\tgene\t365389\t522928\t.\t-\t.\t'
            'gene_id "ENSG00000290385"; gene_version "1"; gene_type "lncRNA"; '
            'gene_name "ENSG00000290385"; tag "ncRNA_host"; tag "overlaps_pseudogene"; '
            'tag "ncRNA_host"; ontology_term "GO:0002"; ontology_term "GO:0001";\n')
    path = tmp_path / ('realistic.gtf.gz' if compressed else 'realistic.gtf')
    if compressed:
        with gzip.open(path, 'wt') as handle:
            handle.write(line)
    else:
        path.write_text(line)
    index = parse_gene_gtf(path)
    record = index.records[0]
    assert record.gene_id == 'ENSG00000290385'
    assert record.start == 365388 and record.end == 522928
    assert record.attributes['tag'] == ('ncRNA_host', 'overlaps_pseudogene')
    assert record.attributes['ontology_term'] == ('GO:0001', 'GO:0002')
    assert record.attributes['gene_name'] == 'ENSG00000290385'
    assert json.loads(json.dumps(record.attributes))['tag'] == ['ncRNA_host', 'overlaps_pseudogene']
    assert isinstance(index.metadata['canonical_map'][0]['gene_name'], str)


def test_repeatable_attribute_order_and_multiplicity_are_semantically_equal(tmp_path):
    prefix = '1\tr\tgene\t1\t10\t.\t+\t.\tgene_id "X"; gene_name "X"; '
    first_line = prefix + 'tag "b"; tag "a"; tag "a"; ontology_term "GO:2"; ontology_term "GO:1";\n'
    second_line = prefix + 'ontology_term "GO:1"; ontology_term "GO:2"; tag "a"; tag "b";\n'
    first = parse_gene_gtf(write_gtf(tmp_path, first_line, 'first.gtf'))
    second = parse_gene_gtf(write_gtf(tmp_path, second_line, 'second.gtf'))
    assert first.records[0].attributes == second.records[0].attributes
    assert first.fingerprint == second.fingerprint
    assert first.metadata['reference_sha256'] != second.metadata['reference_sha256']
    duplicate = parse_gene_gtf(write_gtf(tmp_path, first_line + second_line, 'duplicate.gtf'))
    assert len(duplicate.records) == 1 and duplicate.fingerprint == first.fingerprint


@pytest.mark.parametrize("key", ['gene_id', 'gene_name', 'gene_version', 'gene_type', 'gene_biotype', 'havana_gene', 'unknown_metadata'])
def test_conflicting_singular_gtf_attributes_still_rejected(tmp_path, key):
    line = ('1\tr\tgene\t1\t2\t.\t+\t.\tgene_id "X"; gene_name "X"; '
            + f'{key} "different"; {key} "another";\n')
    with pytest.raises(ValueError, match=f'conflicting GTF attribute {key}'):
        parse_gene_gtf(write_gtf(tmp_path, line))


def test_repeatable_attributes_preserve_escaped_quotes_and_semicolons(tmp_path):
    line = '1\tr\tgene\t1\t2\t.\t+\t.\tgene_id "X"; gene_name "X"; tag "has; semicolon"; tag "a \\"quote\\"";\n'
    record = parse_gene_gtf(write_gtf(tmp_path, line)).records[0]
    assert record.attributes['tag'] == ('a "quote"', 'has; semicolon')
