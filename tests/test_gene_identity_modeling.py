"""Biological symbols cross into models only as stable expression-column IDs."""
from copy import deepcopy
import io
import pickle

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from CausalBridge import granger, kinetics, preprocess
from CausalBridge.gene_identity import apply_gene_identity, parse_gene_gtf


@pytest.fixture(autouse=True)
def current_registry_adapter(monkeypatch):
    from CausalBridge import saturating_transfer as t
    monkeypatch.setattr(t, "_snapshot", lambda r, a, *args: {"R": r.X, "A": a.X})
    monkeypatch.setattr(t, "fit_tf_peak", lambda *args, **kwargs: {"type": t.TF_TYPE})
    monkeypatch.setattr(t, "validate_registry", lambda registry, **kwargs: registry)


@pytest.fixture
def inputs(tmp_path):
    path = tmp_path / "genes.gtf"
    symbols = ["A", "A", "RUNX1", "MT-CO1", "NONHVG"]
    path.write_text("".join(
        f'1\tref\tgene\t{i * 100 + 1}\t{i * 100 + 80}\t.\t+\t.\tgene_id "ID{i}"; gene_name "{symbol}";\n'
        for i, symbol in enumerate(symbols)))
    index = parse_gene_gtf(path)
    t = np.linspace(1, 10, 20)
    x = np.column_stack([t, t ** 2, t * 3, np.ones(20), t * 4])
    obs = pd.DataFrame(index=[f"c{i}" for i in range(20)])
    rna = ad.AnnData(x, obs=obs, var=pd.DataFrame({"gene_ids": index.ids}, index=symbols))
    rna = apply_gene_identity(rna, index).adata
    rna.layers["raw"] = rna.X.copy()
    atac = ad.AnnData(t[:, None], obs=obs, var=pd.DataFrame(
        {"chr": ["1"], "start": [10], "end": [30]}, index=["p"]))
    pt = pd.DataFrame({"pseudotime": np.linspace(0, 1, 20)}, index=obs.index)
    cfg = {"input": {"gene_identity_mode": "gtf_gene_id"},
           "pseudotime": {"n_bins": 10, "min_cells_per_bin": 1},
           "rna_to_atac": {"max_tfs_per_peak": 2, "max_lag": 0,
                           "time_lag": False, "tf_expression_filter": True},
           "output": {"cache_dir": str(tmp_path)}}
    edges = pd.DataFrame({"gene": ["ID0", "ID1", "ID2"], "peak_id": ["p"] * 3})
    resources = {"resource_cache_dir": str(tmp_path), "motif_db": {"m": {}},
                 "tf_database": frozenset({"A", "RUNX1", "NONHVG"})}
    return rna, atac, pt, cfg, edges, index, resources


@pytest.mark.parametrize("runtime", [False, True])
def test_symbol_expansion_actual_regression_and_immutable_candidates(inputs, monkeypatch, runtime):
    rna, atac, pt, cfg, edges, index, resources = inputs
    raw_candidates = {"p": ["A", "RUNX1"]}
    before = pickle.dumps(raw_candidates)
    original_x = rna.X.copy()
    original_var = rna.var.copy()
    monkeypatch.setattr(kinetics, "_scan_peaks_with_jaspar", lambda *a: raw_candidates)
    result = kinetics.fit_rna_to_atac(rna, atac, pt, edges, cfg,
         tf_model_registry={}, motif_resources=resources,
        gene_identity=index if runtime else None)
    assert set(result.tf_gene) == {"ID0", "ID1", "ID2"}
    assert result.set_index("tf_gene").tf_gene_name.to_dict() == {
        "ID0": "A", "ID1": "A", "ID2": "RUNX1"}
    assert result.set_index("tf_gene").tf_mapping_count.to_dict() == {
        "ID0": 2, "ID1": 2, "ID2": 1}
    assert result.set_index("tf_gene").alias_candidate.to_dict() == {
        "ID0": True, "ID1": True, "ID2": False}
    assert len(result) > cfg["rna_to_atac"]["max_tfs_per_peak"]
    assert pickle.dumps(raw_candidates) == before
    np.testing.assert_array_equal(rna.X, original_x)
    pd.testing.assert_frame_equal(rna.var, original_var)


def test_family_receives_base_targets(inputs, monkeypatch):
    rna, atac, pt, cfg, edges, index, resources = inputs
    candidates = {"p": ["RUNX1"]}
    monkeypatch.setattr(kinetics, "_scan_peaks_with_jaspar", lambda *a: candidates)
    seen = []
    def supplement(peak_tfs, db, types, *args, **kwargs):
        seen.append(types)
        peak_tfs["p"].append("A")
    monkeypatch.setattr(kinetics, "_supplement_family_proxies", supplement)
    result = kinetics.fit_rna_to_atac(rna, atac, pt, edges, cfg,
        {"ID0": "TF", "ID1": "TF", "ID2": "TF"},
         tf_model_registry={}, motif_resources=resources, gene_identity=index)
    assert seen == [{"A": "TF", "RUNX1": "TF"}]
    assert set(result.tf_gene) == {"ID0", "ID1", "ID2"}
    assert candidates == {"p": ["RUNX1"]}


def test_fallback_symbol_database_then_id_expansion(inputs):
    rna, atac, pt, cfg, edges, index, resources = inputs
    original = edges.copy()
    assigned = kinetics._assign_tfs_fallback(atac, rna, edges,
        tf_database=resources["tf_database"], gene_identity=index)
    assert set(assigned["p"]) == {"A", "RUNX1"}
    resources = dict(resources, motif_db=None)
    result = kinetics.fit_rna_to_atac(rna, atac, pt, edges, cfg,
         tf_model_registry={}, motif_resources=resources, gene_identity=index)
    assert set(result.tf_gene) == {"ID0", "ID1", "ID2"}
    pd.testing.assert_frame_equal(edges, original)
    assert resources["tf_database"] == frozenset({"A", "RUNX1", "NONHVG"})


def test_full_rna_presence_is_not_claimed_as_fitted(inputs, monkeypatch, caplog):
    rna, atac, pt, cfg, edges, index, resources = inputs
    retained = rna[:, :3].copy()
    monkeypatch.setattr(kinetics, "_scan_peaks_with_jaspar", lambda *a: {"p": ["NONHVG", "A"]})
    with caplog.at_level("INFO"):
        result = kinetics.fit_rna_to_atac(retained, atac, pt, edges, cfg,
            rna_raw=rna,  tf_model_registry={}, motif_resources=resources, gene_identity=index)
    assert set(result.tf_gene) == {"ID0", "ID1"}
    assert result.attrs["tf_mapping_diagnostics"] == [
        {"peak_id": "p", "tf_symbol": "NONHVG", "status": "not_in_model"}]
    assert "not in retained RNA model" not in caplog.text
    assert "保留ID候选: 2, 跳过symbol/peak: 1" in caplog.text


@pytest.mark.parametrize("tty", [False, True])
def test_tf_mapping_final_summary(inputs, monkeypatch, caplog, tty):
    rna, atac, pt, cfg, edges, index, resources = inputs
    cfg = deepcopy(cfg)
    cfg["rna_to_atac"]["tf_expression_filter"] = False
    candidates = {f"p{i}": ["A", "A", "RUNX1", "unknown", "unknown"] for i in range(4)}
    before = pickle.dumps(candidates)
    monkeypatch.setattr(kinetics, "_scan_peaks_with_jaspar", lambda *a: candidates)
    class Stream(io.StringIO):
        def isatty(self):
            return tty
    stream = Stream()
    updates = []
    actual_progress = kinetics.TerminalProgress
    class MappingProgress(actual_progress):
        def __init__(self, total, desc, **kwargs):
            assert total == 4
            assert desc == "Step 6 TF映射"
            super().__init__(total, desc, stream=stream, **kwargs)

        def update(self, completed):
            super().update(completed)
            updates.append(completed)
            if self.bar is not None:
                self.bar.refresh()
    monkeypatch.setattr(kinetics, "TerminalProgress", MappingProgress)
    seen = []
    def regression(mapped, *args, **kwargs):
        seen.append(deepcopy(mapped))
        return [{"peak_id": peak, "tf_gene": gene_id} for peak, ids in mapped.items() for gene_id in ids]
    monkeypatch.setattr(kinetics, "_fit_tf_peak_regression", regression)
    with caplog.at_level("INFO", logger=kinetics.__name__):
        result = kinetics.fit_rna_to_atac(rna, atac, pt, edges, cfg,
             tf_model_registry={}, motif_resources=resources, gene_identity=index)
    assert updates == [1, 2, 3, 4]
    if tty:
        frames = [line.rstrip() for line in stream.getvalue().split("\r") if line.strip()]
        assert frames[0].endswith("0/4")
        assert any(line.endswith("1/4") for line in frames)
        assert frames[-1].endswith("4/4")
        assert stream.getvalue().endswith("\n")
    else:
        assert stream.getvalue() == "Step 6 TF映射: 0/4\nStep 6 TF映射: 4/4\n"
    progress = [r for r in caplog.records if r.msg.startswith("TF候选映射完成")]
    assert len(progress) == 1
    assert progress[-1].getMessage() == "TF候选映射完成: 保留ID候选: 12, 跳过symbol/peak: 4"
    assert "TF candidate" not in caplog.text
    assert "not in retained RNA model" not in caplog.text
    assert seen == [{f"p{i}": ["ID0", "ID1", "ID2"] for i in range(4)}]
    assert result.attrs["tf_mapping_diagnostics"] == [
        {"peak_id": f"p{i}", "tf_symbol": "unknown", "status": "not_in_model"} for i in range(4)]
    assert result.tf_mapping_count.tolist() == [2, 2, 1] * 4
    assert result.alias_candidate.tolist() == [True, True, False] * 4
    assert pickle.dumps(candidates) == before


def test_tf_mapping_progress_empty_candidates(inputs, monkeypatch, caplog, capsys):
    rna, atac, pt, cfg, edges, index, resources = inputs
    monkeypatch.setattr(kinetics, "_scan_peaks_with_jaspar", lambda *a: {})
    with caplog.at_level("INFO", logger=kinetics.__name__):
        result = kinetics.fit_rna_to_atac(rna, atac, pt, edges, cfg,
             tf_model_registry={}, motif_resources=resources, gene_identity=index)
    assert capsys.readouterr().err == "Step 6 TF映射: 0/0\nStep 6 TF映射: 0/0\n"
    progress = [r.getMessage() for r in caplog.records if r.msg.startswith("TF候选映射完成")]
    assert len(progress) == 1
    assert progress[0] == "TF候选映射完成: 保留ID候选: 0, 跳过symbol/peak: 0"
    assert result.empty
    assert result.attrs["tf_mapping_diagnostics"] == []


@pytest.mark.parametrize("runtime", [False, True])
def test_tf_mapping_exact_token_cache_is_local_and_preserves_audits(inputs, monkeypatch, runtime):
    from CausalBridge import gene_identity
    rna, atac, pt, cfg, edges, index, resources = inputs
    cfg = deepcopy(cfg)
    cfg["rna_to_atac"]["tf_expression_filter"] = False
    tokens = ["A", "A", "a", "RUNX1", "unknown", "unknown", "A_01", " A "]
    candidates = {f"p{i}": list(tokens) for i in range(40)}
    before = pickle.dumps(candidates)
    monkeypatch.setattr(kinetics, "_scan_peaks_with_jaspar", lambda *a: candidates)
    calls = []
    actual = gene_identity.map_tf_symbols
    def counted(symbols, identity, **kwargs):
        calls.append((tuple(symbols), tuple(kwargs["allowed_ids"])))
        return actual(symbols, identity, **kwargs)
    monkeypatch.setattr(gene_identity, "map_tf_symbols", counted)
    seen = []
    def regression(mapped, *args, **kwargs):
        seen.append(deepcopy(mapped))
        return [{"peak_id": peak, "tf_gene": gene_id} for peak, ids in mapped.items() for gene_id in ids]
    monkeypatch.setattr(kinetics, "_fit_tf_peak_regression", regression)
    # Each call has different retained axes; no cache entry may cross calls.
    for ids in (["ID0", "ID1", "ID2"], ["ID1", "ID2"]):
        retained = rna[:, ids].copy()
        result = kinetics.fit_rna_to_atac(retained, atac, pt, edges[edges.gene.isin(ids)], cfg,
             tf_model_registry={}, motif_resources=resources,
            gene_identity=index if runtime else None)
        a_ids = [gene_id for gene_id in ids if gene_id != "ID2"]
        assert seen[-1] == {peak: a_ids + a_ids + ["ID2"] for peak in candidates}
        assert result.tf_mapping_count.tolist() == ([len(a_ids)] * (2 * len(a_ids)) + [1]) * 40
        assert result.alias_candidate.tolist() == ([len(a_ids) > 1] * (2 * len(a_ids)) + [False]) * 40
        assert result.attrs["tf_mapping_diagnostics"] == [
            {"peak_id": peak, "tf_symbol": symbol, "status": "not_in_model"}
            for peak in candidates for symbol in ("unknown", "A_01", " A ")]
    assert [tokens for tokens, _ in calls] == ([("A",), ("a",), ("RUNX1",)] * 2 if runtime else [])
    if runtime:
        assert calls[0][1] != calls[3][1]
    assert pickle.dumps(candidates) == before


def test_tf_mapping_resolver_exceptions_are_preserved(inputs, monkeypatch, capsys):
    from CausalBridge import gene_identity
    rna, atac, pt, cfg, edges, index, resources = inputs
    monkeypatch.setattr(kinetics, "_scan_peaks_with_jaspar", lambda *a: {"p": ["A"]})
    def fail(*args, **kwargs):
        raise ValueError("synthetic allowed-axis validation failure")
    monkeypatch.setattr(gene_identity, "map_tf_symbols", fail)
    with pytest.raises(ValueError, match="allowed-axis validation failure"):
        kinetics.fit_rna_to_atac(rna, atac, pt, edges, cfg,
             tf_model_registry={}, motif_resources=resources, gene_identity=index)
    assert capsys.readouterr().err == "Step 6 TF映射: 0/1\nStep 6 TF映射（中断）: 0/1\n"


def test_exact_ids_do_not_case_overwrite_regression():
    t = np.linspace(1, 10, 20)
    rna = ad.AnnData(np.column_stack([t, t ** 2]), var=pd.DataFrame(index=["id", "ID"]))
    atac = ad.AnnData(t[:, None], var=pd.DataFrame(index=["p"]))
    result = kinetics._fit_tf_peak_regression({"p": ["id", "ID"]}, rna, atac,
        rna.X, atac.X, max_lag=0)
    assert {row["tf_gene"] for row in result} == {"id", "ID"}
    weights = {row["tf_gene"]: row["weight"] for row in result}
    assert weights["id"] != weights["ID"]


def test_markers_exclude_mt_by_base_and_preserve_ids(inputs, tmp_path, monkeypatch):
    rna, _, _, _, _, index, _ = inputs
    rna.obs["leiden"] = pd.Categorical(["0"] * 10 + ["1"] * 10)
    rna.raw = rna.copy()
    before = rna.X.copy()
    seen = []
    def rank(adata, **kwargs):
        seen.append(list(adata.var_names))
        assert kwargs["use_raw"] is False
        assert "ID3" not in adata.var_names
        names = np.rec.fromarrays([["ID0", "ID1"], ["ID1", "ID0"]], names=["0", "1"])
        values = np.rec.fromarrays([[1., 2.], [2., 1.]], names=["0", "1"])
        adata.uns["rank_genes_groups"] = {"names": names, "logfoldchanges": values,
                                          "pvals": values, "pvals_adj": values}
    monkeypatch.setattr(preprocess.sc.tl, "rank_genes_groups", rank)
    path = tmp_path / "markers.csv"
    preprocess._export_cluster_markers(rna, n_top=2, output_path=path, gene_identity=index)
    result = pd.read_csv(path)
    assert set(result.gene_id) == {"ID0", "ID1"}
    assert set(result.gene) == {"A_01", "A_02"}
    assert set(result.gene_name) == {"A"}
    assert len(seen) == 1
    np.testing.assert_array_equal(rna.X, before)
    assert rna.shape == (20, 5)


def test_explicit_root_marker_and_runtime_resolution(inputs):
    rna, _, _, _, _, index, _ = inputs
    with pytest.raises(ValueError, match="retained RNA gene ID"):
        granger._find_root_cell(rna, "A")
    with pytest.raises(ValueError, match="ambiguous"):
        granger._find_root_cell(rna, "A", gene_identity=index)
    assert granger._find_root_cell(rna, "ID0") == 19
    assert granger._find_root_cell(rna, "A_01", gene_identity=index) == 19
    assert granger._find_root_cell(rna, "RUNX1", gene_identity=index) == 19


def test_root_config_precheck_before_pseudotime_work(inputs):
    rna, atac, _, cfg, _, _, _ = inputs
    cfg = deepcopy(cfg)
    cfg["pseudotime"]["root_cell_marker"] = "missing"
    with pytest.raises(ValueError, match="root_cell_marker"):
        granger.infer_pseudotime(rna, atac, cfg)


def test_runtime_index_validates_var_mapping(inputs):
    rna, _, _, _, _, index, _ = inputs
    rna.var.loc["ID0", "gene_name"] = "A_01"
    with pytest.raises(ValueError, match="mapping differs"):
        kinetics._gene_base_map(rna, index)


def test_legacy_symbol_filter_and_regression():
    t = np.linspace(1, 10, 20)
    rna = ad.AnnData(t[:, None], var=pd.DataFrame(index=["Runx1"]))
    atac = ad.AnnData(t[:, None], var=pd.DataFrame(index=["p"]))
    candidates = kinetics._filter_tfs_by_expression({"p": ["RUNX1", "absent"]}, rna.X,
                                                   list(rna.var_names))
    assert candidates == {"p": ["RUNX1"]}
    result = kinetics._fit_tf_peak_regression(candidates, rna, atac, rna.X, atac.X, max_lag=0)
    assert result[0]["tf_gene"] == "Runx1"
    assert kinetics._gene_base_map(rna) is None
