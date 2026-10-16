"""Reference-only tests: no fitting, motif scans or user output writes."""
import copy

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from CausalBridge import reference_clustering as ref
from CausalBridge.preprocess import preprocess_rna, preprocess_atac, match_cells, cluster_cells
from CausalBridge.gene_identity import apply_gene_identity, parse_gene_gtf
from CausalBridge import run


def inputs():
    rng = np.random.default_rng(14)
    counts = rng.poisson(rng.uniform(1, 5, (120, 600))).astype(float)
    counts[:, 0] = 5000  # Annotation removal makes independent model cell QC fail.
    rna = ad.AnnData(sparse.csr_matrix(counts),
        obs=pd.DataFrame(index=[f"c{i:03}" for i in range(120)]),
        var=pd.DataFrame({"gene_ids": [f"E{i}.1" for i in range(600)]},
                         index=[f"G{i}" for i in range(600)]))
    atac = ad.AnnData(sparse.csr_matrix(rng.binomial(1, .5, (120, 100))),
        obs=rna.obs.copy())
    cfg = {"input": {}, "preprocess": {
        "rna": dict(min_cells_per_gene=1, min_umi_per_cell=5000, n_highly_variable_genes=70),
        "atac": dict(min_cells_per_peak=1, min_peaks_per_cell=1, binarize=True)},
        "clustering": dict(enabled=True, n_neighbors=6, n_pcs=5, resolution=.8, random_state=42)}
    return rna, atac, cfg


def test_real_legacy_parity_and_gtf_model_count_axes(tmp_path):
    raw, atac, cfg = inputs()
    original = raw.X.copy()
    reference, processed_atac = ref.build_reference_clustering(raw, atac, ["G1"], cfg)
    legacy = preprocess_rna(raw, **cfg["preprocess"]["rna"], keep_genes=["G1"])
    legacy, _ = match_cells(legacy, preprocess_atac(atac, **cfg["preprocess"]["atac"]))
    legacy = cluster_cells(legacy, n_neighbors=6, n_pcs=5, resolution=.8, random_state=42)
    np.testing.assert_array_equal(reference.obs.leiden, legacy.obs.leiden)
    np.testing.assert_array_equal(reference.obsp["connectivities"].toarray(), legacy.obsp["connectivities"].toarray())
    np.testing.assert_array_equal(reference.var_names, legacy.var_names)
    gtf = tmp_path / "genes.gtf"
    gtf.write_text("\n".join(f'1\tr\tgene\t{i*100+1}\t{i*100+90}\t.\t+\t.\tgene_id "E{i}"; gene_name "G{i}";' for i in range(1, 600)) + "\n")
    cfg["input"].update(gene_identity_mode="gtf_gene_id", annotation_gtf=str(gtf))
    cfg["pseudotime"] = {}
    adapted, targets, _, runtime, identity, tables = run._prepare_gene_identity_inputs(raw, ["G1"], {}, cfg)
    model = run._prepare_identity_model_rna(raw, reference, adapted, targets, runtime, identity)
    expected = preprocess_rna(adapted, **cfg["preprocess"]["rna"],
        keep_genes=targets, filter_cells=False)
    np.testing.assert_array_equal(model.var_names, expected.var_names)
    np.testing.assert_array_equal(model.X.toarray(), expected.X.toarray())
    np.testing.assert_array_equal(model.var.highly_variable, expected.var.highly_variable)
    assert len(tables["gene_filter_report"]) == raw.n_vars
    # Explicitly reorder to detect accidental positional label/graph assignment.
    model = model[::-1].copy()
    ref.attach_reference_clustering(model, reference)
    pd.testing.assert_series_equal(model.obs.leiden, reference.obs.loc[model.obs_names, "leiden"])
    assert model.n_obs == processed_atac.n_obs == 120
    assert "E0" not in model.var_names and model.var_names.is_unique
    cols = [int(g[1:]) for g in model.var_names]
    rows = raw.obs_names.get_indexer(model.obs_names)
    np.testing.assert_array_equal(model.layers["raw"].toarray(), original[rows][:, cols].toarray())
    np.testing.assert_array_equal(raw.X.toarray(), original.toarray())
    model.write_h5ad(tmp_path / "model.h5ad")
    saved = ad.read_h5ad(tmp_path / "model.h5ad")
    pd.testing.assert_series_equal(saved.obs.leiden, model.obs.leiden)
    assert saved.uns["clustering_reference"]["n_genes"] == reference.n_vars


def test_raw_ids_symbols_and_model_settings_do_not_change_reference(monkeypatch):
    raw, atac, cfg = inputs()
    calls = []
    def fake_cluster(a, **kwargs):
        calls.append(kwargs)
        a.obs["leiden"] = pd.Categorical(np.arange(a.n_obs) % 2)
        return a
    monkeypatch.setattr(ref, "cluster_cells", fake_cluster)
    first, _ = ref.build_reference_clustering(raw, atac, ["G1"], cfg)
    other = copy.deepcopy(cfg)
    other.update(atac_to_rna={"model_mode": "saturating_hybrid", "saturating_hybrid": {"workers": 24}},
                 pseudotime={"root_cell_marker": "G599"}, perturbation={"ko_strength": .2})
    other["input"].update(gene_identity_mode="gtf_gene_id", annotation_gtf="unused")
    second, _ = ref.build_reference_clustering(raw, atac, ["E1.9"], other)
    assert first.uns["clustering_reference"] == second.uns["clustering_reference"]
    assert calls[0] == calls[1]
    assert ref.raw_target_symbols(raw, ["missing"]) == ["missing"]
    raw.var.iloc[2, 0] = "E1.2"
    with pytest.raises(ValueError, match="ambiguous raw"):
        ref.raw_target_symbols(raw, ["E1"])


def test_atac_preprocessed_once_and_barcode_protection(monkeypatch):
    raw, atac, cfg = inputs()
    cfg["clustering"]["enabled"] = False
    calls = []
    def once(a, **kwargs):
        calls.append(1)
        return preprocess_atac(a, **kwargs)
    monkeypatch.setattr(ref, "preprocess_atac", once)
    reference, processed = ref.build_reference_clustering(raw, atac, ["G1"], cfg)
    model = ref.align_model_cells(raw, reference)
    ref.attach_reference_clustering(model, reference)
    assert calls == [1] and processed.n_obs == model.n_obs
    with pytest.raises(ValueError, match="missing reference"):
        ref.align_model_cells(raw[1:], reference)
    with pytest.raises(ValueError, match="match exactly"):
        ref.attach_reference_clustering(model[1:].copy(), reference)
    duplicate = raw.copy()
    duplicate.obs_names = ["dup"] * duplicate.n_obs
    with pytest.raises(ValueError, match="unique cell"):
        ref.build_reference_clustering(duplicate, atac, ["G1"], cfg)


def test_skip_cell_qc_is_keyword_only_and_default_unchanged():
    import inspect
    param = inspect.signature(preprocess_rna).parameters["filter_cells"]
    assert param.default is True and param.kind is inspect.Parameter.KEYWORD_ONLY
