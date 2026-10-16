"""Model HVG/normalization use eligible GTF counts; reference stays original."""
import copy

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from CausalBridge import io, run, reference_clustering as ref
from CausalBridge.preprocess import preprocess_rna
from test_gene_identity_pipeline import fixture, execute_identity_subset


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def test_gtf_first_model_hvg_normalization_export_and_independent_reference(tmp_path):
    rng = np.random.default_rng(104)
    counts = rng.poisson(rng.uniform(.5, 8, (120, 640))).astype(float)
    counts[:, 0] = rng.integers(3000, 9000, 120)
    rna = ad.AnnData(sparse.csr_matrix(counts),
        obs=pd.DataFrame(index=[f"c{i:03}" for i in range(120)]),
        var=pd.DataFrame({"gene_ids": [f"E{i}.7" for i in range(640)]},
                         index=[f"input_feature_{i}" for i in range(640)]))
    rna.raw = rna.copy()
    rna.layers["evidence"] = rna.X.copy()
    atac = ad.AnnData(sparse.csr_matrix(rng.binomial(1, .5, (120, 70))), obs=rna.obs.copy())
    cfg = {"input": {}, "preprocess": {
        "rna": dict(min_cells_per_gene=1, min_umi_per_cell=5000, n_highly_variable_genes=70),
        "atac": dict(min_cells_per_peak=1, min_peaks_per_cell=1, binarize=True)},
        "clustering": dict(enabled=True, n_neighbors=6, n_pcs=5, resolution=.8, random_state=42),
        "pseudotime": {}, "output": {"dir": str(tmp_path / "explicit results"), "save_intermediate": False}}
    gtf = tmp_path / "genes.gtf"
    # E0 dominates input depth but is ineligible; duplicate symbols must not merge.
    gtf.write_text("\n".join(
        f'1\tr\tgene\t{i*100+1}\t{i*100+90}\t.\t+\t.\tgene_id "E{i}"; gene_name "DUP";'
        for i in range(1, 640)) + "\n")
    cfg["input"].update(gene_identity_mode="gtf_gene_id", annotation_gtf=str(gtf))
    full, _, _, _, identity, tables = run._prepare_gene_identity_inputs(rna, ["E1"], {}, cfg)
    selected = preprocess_rna(full, **cfg["preprocess"]["rna"], filter_cells=False)
    non_hvg = [g for g in full.var_names if g not in selected.var_names]
    target, marker = non_hvg[:2]
    cfg["pseudotime"]["root_cell_marker"] = marker
    full, targets, _, runtime, identity, tables = run._prepare_gene_identity_inputs(rna, [target], {}, cfg)
    reference, _ = ref.build_reference_clustering(rna, atac, [target], cfg)
    model = run._prepare_identity_model_rna(rna, reference, full, targets, runtime, identity)
    expected = preprocess_rna(full, **cfg["preprocess"]["rna"],
        keep_genes=[target, marker], filter_cells=False)
    assert len(tables["gene_filter_report"]) == 640
    assert "E0" not in full.var_names and full.n_vars == 639
    np.testing.assert_array_equal(model.var_names, expected.var_names)
    np.testing.assert_array_equal(dense(model.X), dense(expected.X))
    np.testing.assert_array_equal(model.var.highly_variable, expected.var.highly_variable)
    assert model.var.highly_variable.sum() == 70
    assert not model.var.loc[[target, marker], "highly_variable"].any()
    assert model.var.loc[target, "gene_name"] == model.var.loc[marker, "gene_name"] == "DUP"
    positions = [int(x[1:]) for x in model.var_names]
    normalized = np.log1p(counts / counts[:, 1:].sum(axis=1)[:, None] * 1e4)
    np.testing.assert_allclose(dense(model.X), normalized[:, positions], rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(dense(model.layers["raw"]), counts[:, positions])
    np.testing.assert_array_equal(dense(model.layers["evidence"]), counts[:, positions])
    np.testing.assert_array_equal(dense(model.raw.X), dense(expected.raw.X))
    np.testing.assert_array_equal(model.raw.var_names, expected.raw.var_names)
    assert set(full.var_names) - set(model.var_names)  # Full Step 6 presence axis.
    np.testing.assert_array_equal(dense(rna.X), counts)
    for key in ("X_pca", "X_umap"):
        np.testing.assert_array_equal(model.obsm[key], reference.obsm[key])
    for key in ("connectivities", "distances"):
        np.testing.assert_array_equal(dense(model.obsp[key]), dense(reference.obsp[key]))
    pd.testing.assert_series_equal(model.obs.leiden, reference.obs.leiden)
    assert "PCs" not in model.varm
    exported = io.save_hvg_genes(model, runtime)
    saved = pd.read_csv(tmp_path / "explicit results/hvg_genes.csv")
    pd.testing.assert_frame_equal(saved, exported)
    assert saved.gene_id.tolist() == model.var_names[model.var.highly_variable].tolist()
    assert len(saved) == 70 and not {"E0", target, marker} & set(saved.gene_id)
    assert saved.gene_name.eq("DUP").all() and saved.gene_id.is_unique
    assert saved.original_var_name.tolist() == model.var.loc[model.var.highly_variable, "original_var_name"].tolist()
    # GTF/model changes do not affect original-RNA reference geometry/cohort.
    for mode in ("saturating_hybrid",):
        other = copy.deepcopy(cfg)
        other["atac_to_rna"] = {"model_mode": mode}
        other_gtf = tmp_path / f"{mode}.gtf"
        other_gtf.write_text(gtf.read_text().split("\n", 1)[1])
        other["input"]["annotation_gtf"] = str(other_gtf)
        other_ref, _ = ref.build_reference_clustering(rna, atac, [target], other)
        assert other_ref.uns["clustering_reference"] == reference.uns["clustering_reference"]
        for key in ("X_pca", "X_umap"):
            np.testing.assert_array_equal(other_ref.obsm[key], reference.obsm[key])
        np.testing.assert_array_equal(dense(other_ref.obsp["connectivities"]), dense(reference.obsp["connectivities"]))
        pd.testing.assert_series_equal(other_ref.obs.leiden, reference.obs.leiden)


def changed_model_policy(*args, **kwargs):
    return run.original_model_policy(*args, **kwargs)


def test_model_policy_helper_hashed_and_old_checkpoint_rejected(tmp_path, monkeypatch):
    f = fixture(tmp_path, cells=200)
    execute_identity_subset(f)
    root = run._checkpoint_dir(f[2])
    before = {p.name: p.read_bytes() for p in root.glob("*.pkl")}
    stages = ("step4_granger", "step5_atac_to_rna", "step6_rna_to_atac")
    baseline = {s: run._stage_numerical_signatures(s, gene_identity_mode="gtf_gene_id") for s in stages}
    legacy = {s: run._stage_numerical_signatures(s) for s in stages}
    monkeypatch.setattr(run, "original_model_policy", run._prepare_identity_model_rna, raising=False)
    monkeypatch.setattr(run, "_prepare_identity_model_rna", changed_model_policy)
    for stage in stages:
        current = run._stage_numerical_signatures(stage, gene_identity_mode="gtf_gene_id")
        assert current != baseline[stage]
        assert f"CausalBridge.run.{changed_model_policy.__name__}" in current["functions"]
        assert run._stage_numerical_signatures(stage) == legacy[stage]
    with pytest.raises(RuntimeError, match="step4_granger.*incompatible.*使用新的checkpoint_dir"):
        execute_identity_subset(f)
    assert {p.name: p.read_bytes() for p in root.glob("*.pkl")} == before


def test_hvg_export_missing_empty_legacy_and_invalid_flags(tmp_path, caplog):
    rna = ad.AnnData(sparse.csr_matrix(np.ones((2, 3))),
        var=pd.DataFrame({"gene_ids": ["ID1", "ID2", "ID3"]}, index=["A", "A", "B"]))
    cfg = {"output": {"dir": str(tmp_path / "nested/gene_ids")}}
    assert io.save_hvg_genes(rna, cfg) is None
    assert "not exported" in caplog.text
    assert not (tmp_path / "nested").exists()
    rna.var["highly_variable"] = [False, True, False]
    frame = io.save_hvg_genes(rna, cfg)
    assert frame.to_dict("records") == [{"gene_id": "ID2", "gene_name": "A"}]
    rna.var["highly_variable"] = False
    io.save_hvg_genes(rna, cfg)
    assert pd.read_csv(tmp_path / "nested/gene_ids/hvg_genes.csv").empty
    rna.var["highly_variable"] = ["False", "True", "False"]
    with pytest.raises(ValueError, match="boolean flags"):
        io.save_hvg_genes(rna, cfg)


def test_pipeline_exports_global_hvg_before_modeling_even_without_intermediates(tmp_path, monkeypatch):
    rna, atac, cfg = fixture(tmp_path)
    cfg["input"]["target_genes"] = "unused"
    cfg["output"]["save_intermediate"] = False
    monkeypatch.setattr(run, "load_config", lambda _: cfg)
    monkeypatch.setattr(run, "load_data", lambda _: (rna, atac))
    monkeypatch.setattr(run, "load_target_genes", lambda _: (["RUNX1"], {"RUNX1": "TF"}))
    monkeypatch.setattr(ref, "preprocess_rna", lambda a, **k: a.copy())
    monkeypatch.setattr(ref, "preprocess_atac", lambda a, **k: a.copy())
    def model_preprocess(a, **kwargs):
        assert list(a.var_names) == ["E2", "E1", "E3", "E4"]
        assert kwargs["filter_cells"] is False
        a = a.copy()
        a.var["highly_variable"] = [True, True, False, False]
        return a
    monkeypatch.setattr(run, "preprocess_rna", model_preprocess)
    class FinishedPreprocessing(Exception):
        pass
    def modeling(*args, **kwargs):
        frame = pd.read_csv(tmp_path / "results/hvg_genes.csv")
        assert frame.gene_id.tolist() == ["E2", "E1"]
        assert frame.gene_name.tolist() == ["A", "A"]
        assert frame.original_var_name.tolist() == ["old2", "old1"]
        assert kwargs["rna_raw"].n_vars == 4
        raise FinishedPreprocessing
    monkeypatch.setattr(run, "_run_modeling_for_subset", modeling)
    with pytest.raises(FinishedPreprocessing):
        run.run_pipeline("unused")
