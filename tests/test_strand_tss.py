import anndata as ad
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from CausalBridge.gene_identity import parse_gene_gtf, apply_gene_identity, gene_tss_coordinates
from CausalBridge.granger import _generate_candidate_pairs


def test_gtf_strand_tss_span_and_candidate_distance(tmp_path):
    # PTPRD is deliberately the largest span and negative-strand gene. No local
    # reference path is needed to catch the former left-boundary error.
    path = tmp_path / "genes.gtf"
    path.write_text((Path(__file__).parent / "fixtures" / "ptprd_strand.gtf").read_text())
    index = parse_gene_gtf(path)
    rna = ad.AnnData(np.ones((3, 3)), var=pd.DataFrame({"gene_ids": ["ENSG00000153707", "PLUS", "ONE"]}))
    rna = apply_gene_identity(rna, index).adata
    np.testing.assert_array_equal(gene_tss_coordinates(rna), [10613001, 8314245, 41])
    assert rna.var.loc["ENSG00000153707", "start"] == 8314245
    assert rna.var.loc["ENSG00000153707", "end"] == 10613002
    assert max(index.records, key=lambda r: r.end-r.start).gene_name == "PTPRD"
    peaks = ad.AnnData(np.ones((3, 2)), var=pd.DataFrame(
        {"chr": ["chr9", "chr9"], "start": [10613000, 8314244], "end": [10613002, 8314246]}, index=["right", "left"]))
    assert set(_generate_candidate_pairs(rna, peaks, 2)) == {("right", "ENSG00000153707"), ("left", "PLUS")}
    canonical = {r["gene_id"]: r for r in index.metadata["canonical_map"]}
    assert canonical["ENSG00000153707"]["strand"] == "-"
    assert canonical["ENSG00000153707"]["gene_tss"] == 10613001
    changed = tmp_path / "changed.gtf"
    changed.write_text(path.read_text().replace('10613002\t.\t-', '10613002\t.\t+'))
    assert parse_gene_gtf(changed).fingerprint != index.fingerprint


def test_unknown_gtf_strand_rejected_legacy_left_anchor_supported(tmp_path):
    path = tmp_path / "bad.gtf"
    path.write_text('chr1\tx\tgene\t1\t2\t.\t.\t.\tgene_id "ID"; gene_name "A";\n')
    with pytest.raises(ValueError, match="strand"):
        parse_gene_gtf(path)
    rna = ad.AnnData(np.ones((2, 1)), var=pd.DataFrame({"start": [5], "end": [100]}))
    assert gene_tss_coordinates(rna).tolist() == [5]
    rna.var["gene_tss"] = 99
    assert gene_tss_coordinates(rna).tolist() == [99]
    rna.uns["gene_identity"] = {}
    with pytest.raises(ValueError, match="strand"):
        gene_tss_coordinates(rna)
