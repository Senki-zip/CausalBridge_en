"""Synthetic pipeline identity boundaries; no real-data modeling or motif scans."""
import copy
import json
import pickle
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import sparse

from CausalBridge import run, io
from CausalBridge import saturating_transfer
from CausalBridge import gene_identity as identity_module, kinetics, perturbation
from CausalBridge.gene_identity import parse_gene_gtf, apply_gene_identity
from CausalBridge import reference_clustering


@pytest.fixture(autouse=True)
def tiny_reference_preprocessing(monkeypatch):
    # These fixtures exercise modeling boundaries, not Scanpy's >=500-gene QC.
    monkeypatch.setattr(reference_clustering, "preprocess_rna", lambda a, **k: a.copy())
    monkeypatch.setattr(reference_clustering, "preprocess_atac", lambda a, **k: a.copy())


def fixture(tmp_path, cells=24):
    gtf = tmp_path / "genes.gtf"
    gtf.write_text('\n'.join([
        '1\tr\tgene\t101\t200\t.\t+\t.\tgene_id "E1.1"; gene_name "A";',
        '1\tr\tgene\t301\t400\t.\t+\t.\tgene_id "E2"; gene_name "A";',
        '1\tr\tgene\t501\t600\t.\t+\t.\tgene_id "E3"; gene_name "RUNX1";',
        '1\tr\tgene\t701\t800\t.\t+\t.\tgene_id "E4"; gene_name "MARKER";',
    ]) + '\n')
    ids = pd.Index([f"c{i}" for i in range(cells)])
    rng = np.random.default_rng(71)
    rna = ad.AnnData(sparse.csr_matrix(rng.uniform(1., 3., (cells, 5))),
        obs=pd.DataFrame(index=ids), var=pd.DataFrame({"gene_ids": ["E2.2", "E1", "E3", "E4", "EXCLUDED"],
            "gene_name": ["A", "A", "RUNX1", "MARKER", "C"]}, index=["old2", "old1", "RUNX1", "MARKER", "C"]))
    atac = ad.AnnData(sparse.csr_matrix(rng.binomial(1, .6, (cells, 2))), obs=pd.DataFrame(index=ids),
        var=pd.DataFrame({"chr": ["1", "1"], "start": [110, 310], "end": [120, 320]}, index=["p", "q"]))
    cfg = {"input": {"gene_identity_mode": "gtf_gene_id", "annotation_gtf": str(gtf),
                     "rna_gene_id_column": "gene_ids"},
           "output": {"dir": str(tmp_path / "results"), "checkpoint_dir": str(tmp_path / "checkpoints"),
                      "cache_dir": str(tmp_path / "cache"), "save_intermediate": True},
           "atac_to_rna": {"model_mode": "saturating_hybrid", "response_model":"spline", "log_normalize_rna": True},
           "pseudotime": {"root_cell_marker": "MARKER", "root_cell_method": "marker", "n_neighbors": 5,
                          "n_bins": 50, "min_cells_per_bin": 1, "adaptive_binning": {"enabled": False}},
           "granger": {"reference_bins": 50, "min_composite_score": .3, "significance_threshold": .1, "distance_thresh": 1000},
           "rna_to_atac": {"tf_expression_filter": False},
           "perturbation": {"ko_strength": .1, "joint_knockout": True, "subnetwork_depth": 2,
                            "min_r2_threshold": -1e9, "min_lag_per_peak_filter": False,
                            "cell_projection": {"enabled": False}},
           "clustering": {"enabled": False, "min_cells_per_cluster": 1},
           "preprocess": {"rna": {"min_cells_per_gene": 1, "min_umi_per_cell": 1, "n_highly_variable_genes": 1},
                          "atac": {"min_cells_per_peak": 1, "min_peaks_per_cell": 1, "binarize": True}}}
    return rna, atac, cfg


def prepare(f, tokens=None):
    rna, _, cfg = f
    tokens = tokens or ["RUNX1"]
    return run._prepare_gene_identity_inputs(rna, tokens, {v: "TF" for v in tokens}, cfg)


def test_preparation_full_membership_ids_coordinates_and_caller_immutability(tmp_path):
    f = fixture(tmp_path)
    before = copy.deepcopy(f[2])
    rna, targets, types, cfg, index, tables = prepare(f)
    assert list(rna.var_names) == ["E2", "E1", "E3", "E4"]
    assert targets == ["E3"] and types == {"E3": "TF"}
    assert cfg["pseudotime"]["root_cell_marker"] == "E4"
    assert f[2] == before
    assert list(f[0].var_names) == ["old2", "old1", "RUNX1", "MARKER", "C"]
    np.testing.assert_array_equal(rna.X.toarray(), f[0].X[:, :4].toarray())
    assert list(rna.var.gene_display_name[:2]) == ["A_02", "A_01"]
    assert list(rna.var.start[:2]) == [300, 100]
    assert len(tables["gene_filter_report"]) == 5
    assert tables["gene_filter_report"].reason.iloc[-1] == "not_in_reference"
    assert tables["gene_identity_metadata"]["requested_targets"] == ["RUNX1"]
    assert not any(value is index for value in tables.values())


@pytest.mark.parametrize("token,expected", [("E1.9", "E1"), ("A_02", "E2"), ("RUNX1", "E3"), ("runx1", "E3")])
def test_target_tokens_resolve_before_qc(tmp_path, token, expected):
    assert prepare(fixture(tmp_path), [token])[1] == [expected]


@pytest.mark.parametrize("token", ["A", "C", "EXCLUDED", "unknown"])
def test_bad_targets_fail_before_preprocessing(tmp_path, token, monkeypatch):
    f = fixture(tmp_path)
    monkeypatch.setattr(run, "load_config", lambda _: f[2])
    monkeypatch.setattr(run, "load_data", lambda _: f[:2])
    f[2]["input"]["target_genes"] = "unused"
    monkeypatch.setattr(run, "load_target_genes", lambda _: ([token], {token: "TF"}))
    monkeypatch.setattr(run, "preprocess_rna", lambda *a, **k: pytest.fail("QC executed before target validation"))
    with pytest.raises(ValueError, match="ambiguous|unresolved|allowed_ids"):
        run.run_pipeline("unused")


def test_all_expands_after_reference_filter_and_alias_numbering_is_fixed(tmp_path):
    f = fixture(tmp_path)
    rna, targets, types, _, identity, _ = prepare(f, ["__ALL__"])
    assert targets == list(rna.var_names) and "EXCLUDED" not in types
    out = io._present_gene_frame(pd.DataFrame({"gene": ["E2"]}), identity)
    assert out.gene.iloc[0] == "A_02"


class StopAfterPreprocessing(Exception):
    pass


def test_reference_precedes_gtf_and_pipeline_reuses_atac_qc(tmp_path, monkeypatch):
    rna, atac, cfg = fixture(tmp_path)
    cfg["input"]["target_genes"] = "unused"
    cfg["clustering"].update(enabled=True, paga={"enabled": False})
    events = []
    monkeypatch.setattr(run, "load_config", lambda _: cfg)
    monkeypatch.setattr(run, "load_data", lambda _: (rna, atac))
    monkeypatch.setattr(run, "load_target_genes", lambda _: (["RUNX1"], {"RUNX1": "TF"}))
    def reference_qc(a, **kwargs):
        assert list(a.var_names) == list(rna.var_names)
        assert kwargs["keep_genes"] == ["RUNX1"]  # no root-marker injection
        events.append("reference_rna")
        return a[1:, a.var_names != "MARKER"].copy()
    def atac_qc(a, **kwargs):
        events.append("atac")
        return a.copy()
    def cluster(a, **kwargs):
        events.append("reference_cluster")
        a.obs["leiden"] = pd.Categorical(["0"] * a.n_obs)
        return a
    monkeypatch.setattr(reference_clustering, "preprocess_rna", reference_qc)
    monkeypatch.setattr(reference_clustering, "preprocess_atac", atac_qc)
    monkeypatch.setattr(reference_clustering, "cluster_cells", cluster)
    original_prepare = run._prepare_gene_identity_inputs
    def annotation(a, *args, **kwargs):
        assert events == ["reference_rna", "atac", "reference_cluster"]
        assert a.n_obs == 24 and a.n_vars == 5  # audit includes all original RNA
        events.append("annotation")
        return original_prepare(a, *args, **kwargs)
    monkeypatch.setattr(run, "_prepare_gene_identity_inputs", annotation)
    def model_qc(a, **kwargs):
        assert kwargs["filter_cells"] is False
        assert kwargs["keep_genes"] == ["E3", "E4"]
        np.testing.assert_array_equal(a.X.toarray(), rna.X[:, :4].toarray())
        events.append("model_rna")
        return a.copy()
    monkeypatch.setattr(run, "preprocess_rna", model_qc)
    monkeypatch.setattr(run, "preprocess_atac", lambda *a, **k: pytest.fail("double ATAC QC"))
    monkeypatch.setattr(run, "cluster_cells", lambda *a, **k: pytest.fail("model reclustering"))
    def verify(a, **kwargs):
        assert a.n_obs == 23 and a.n_vars == 4
        assert list(a.obs_names) == sorted(rna.obs_names[1:])
        assert list(a.obs.leiden) == ["0"] * 23
        assert a.uns["clustering_reference"]["n_genes"] == 4
        assert events == ["reference_rna", "atac", "reference_cluster", "annotation", "model_rna"]
        raise StopAfterPreprocessing
    monkeypatch.setattr(run, "save_cluster_top_genes", verify)
    with pytest.raises(StopAfterPreprocessing):
        run.run_pipeline("unused")


@pytest.mark.parametrize("drop", [None, "E3", "E4"])
def test_pipeline_passes_id_targets_and_marker_hvg_retention_and_qc_failures(tmp_path, monkeypatch, drop):
    f = fixture(tmp_path)
    f[2]["input"]["target_genes"] = "unused"
    monkeypatch.setattr(run, "load_config", lambda _: f[2])
    monkeypatch.setattr(run, "load_data", lambda _: f[:2])
    monkeypatch.setattr(run, "load_target_genes", lambda _: (["RUNX1"], {"RUNX1": "TF"}))
    # Model preprocessing receives full canonical GTF-eligible counts, while
    # reference preprocessing remains on the independent original gene universe.
    monkeypatch.setattr(reference_clustering, "preprocess_rna",
        lambda a, **k: a[:, a.var_names != "MARKER"].copy())
    def preprocessing(rna, **kwargs):
        assert kwargs["filter_cells"] is False
        assert kwargs["keep_genes"] == ["E3", "E4"]
        assert list(rna.var_names) == ["E2", "E1", "E3", "E4"]
        return rna[:, [str(v).split(".")[0] != drop for v in rna.var.gene_ids]].copy()
    monkeypatch.setattr(run, "preprocess_rna", preprocessing)
    monkeypatch.setattr(run, "preprocess_atac", lambda a, **k: a.copy())
    def modeling(rna, atac, targets, types, config, **kwargs):
        assert targets == ["E3"] and types == {"E3": "TF"}
        assert config["pseudotime"]["root_cell_marker"] == "E4"
        assert kwargs["gene_identity"] is not None
        assert list(kwargs["rna_raw"].var_names) == ["E2", "E1", "E3", "E4"]
        assert config["output"]["dir"] == f[2]["output"]["dir"]
        raise StopAfterPreprocessing
    monkeypatch.setattr(run, "_run_modeling_for_subset", modeling)
    if drop:
        with pytest.raises(ValueError, match="removed by QC"):
            run.run_pipeline("unused")
    else:
        with pytest.raises(StopAfterPreprocessing):
            run.run_pipeline("unused")


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("custom_checkpoint", [False, True])
def test_runtime_namespace_composition_and_cache_unchanged(tmp_path, joint, custom_checkpoint):
    cfg = fixture(tmp_path)[2]
    cfg["atac_to_rna"]["model_mode"] = "saturating_hybrid"
    if not custom_checkpoint:
        cfg["output"]["checkpoint_dir"] = None
    before = copy.deepcopy(cfg)
    first = run._joint_runtime_config(cfg)
    assert cfg == before
    assert first["output"]["cache_dir"] == cfg["output"]["cache_dir"]
    expected_output = tmp_path / "results"
    assert first["output"]["dir"] == str(expected_output)
    assert run._joint_runtime_config(first) == first
    assert run._gene_identity_runtime_config(first) == first
    expected = Path(cfg["output"].get("checkpoint_dir") or (tmp_path / "results" / "checkpoints"))
    assert run._checkpoint_dir(first) == expected
    # Explicitly nested paths belong to the caller and must never be peeled.
    old_joint = copy.deepcopy(cfg)
    if joint:
        old_joint["output"]["dir"] = str(tmp_path / "results" / "joint_gene")
        old_joint["output"]["checkpoint_dir"] = str(tmp_path / "checkpoints") if custom_checkpoint else str(tmp_path / "results" / "checkpoints")
    assert run._joint_runtime_config(old_joint)["output"]["dir"] == old_joint["output"]["dir"]
    assert run._joint_runtime_config(old_joint)["output"]["checkpoint_dir"] == old_joint["output"]["checkpoint_dir"]


def test_output_names_ids_multi_targets_and_no_name_grouping(tmp_path):
    f = fixture(tmp_path)
    rna, _, _, cfg, identity, tables = prepare(f)
    frame = pd.DataFrame({"target_gene": ["E3", "E3"], "affected_gene": ["E1", "E2"], "delta_rna": [.1, .2]})
    results = dict(tables, perturbation_results=frame, perturbation_merged=frame.copy(),
        causal_edges=pd.DataFrame({"gene": ["E1", "E2"], "peak_id": ["p", "q"]}),
        tf_peak_weights=pd.DataFrame({"tf_gene": ["E3"], "peak_id": ["p"]}),
        saturating_support_diagnostics=pd.DataFrame({"gene": ["E1", "E2"], "target_gene": ["E3,E1", "PEAK:p"]}),
        cluster_results=[{"perturbation_results": frame.copy()}])
    before = frame.copy(deep=True)
    io.save_results(results, cfg, gene_identity=identity)
    out = tmp_path / "results"
    saved = pd.read_csv(out / "perturbation_results.csv")
    assert set(saved.affected_gene) == {"A_01", "A_02"}
    assert set(saved.affected_gene_id) == {"E1", "E2"}
    assert list(saved.affected_gene_name) == ["A", "A"]
    assert len(pd.read_csv(out / "perturbation_merged.csv")) == 2
    assert len(pd.read_csv(out / "intermediate" / "causal_edges.csv")) == 2
    diagnostics = pd.read_csv(out / "saturating_support_diagnostics.csv")
    assert diagnostics.target_gene.iloc[0] == "RUNX1,A_01"
    assert json.loads(diagnostics.target_gene_ids_json.iloc[0]) == ["E3", "E1"]
    assert diagnostics.target_gene.iloc[1] == "PEAK:p"
    pd.testing.assert_frame_equal(results["perturbation_results"], before)
    assert list(rna.var_names) == ["E2", "E1", "E3", "E4"]
    assert (out / "gene_identity.csv").exists() and (out / "gene_filter_report.csv").exists()
    assert not (out / "gene_ids").exists()


def test_presentation_idempotence_and_only_explicit_columns(tmp_path):
    identity = prepare(fixture(tmp_path))[4]
    frame = pd.DataFrame({"gene": ["E1"], "peak": ["joint_vector"], "features": ['["E1","E2"]']})
    first = io._present_gene_frame(frame, identity)
    second = io._present_gene_frame(first, identity)
    pd.testing.assert_frame_equal(first, second)
    assert second.features.iloc[0] == frame.features.iloc[0]
    assert second.peak.iloc[0] == "joint_vector"


def test_explicit_identity_output_preserves_custom_checkpoints(tmp_path):
    cfg = fixture(tmp_path)[2]
    cfg["output"]["dir"] = str(tmp_path / "results" / "gene_ids")
    effective = run._joint_runtime_config(cfg)
    assert effective["output"]["dir"] == cfg["output"]["dir"]
    assert run._checkpoint_dir(effective) == tmp_path / "checkpoints"
    assert run._checkpoint_dir(cfg) == tmp_path / "checkpoints"


def test_unavailable_origin_ids_preserved_with_parallel_labels(tmp_path):
    identity = prepare(fixture(tmp_path))[4]
    frame = pd.DataFrame({"gene": ["E2"], "unavailable_origins": ['["E1:gene_filtered"]']})
    out = io._present_gene_frame(frame, identity)
    assert out.unavailable_origins.iloc[0] == frame.unavailable_origins.iloc[0]
    assert json.loads(out.unavailable_origins_display_json.iloc[0]) == ["A_01:gene_filtered"]
    assert json.loads(out.unavailable_origins_names_json.iloc[0]) == ["A:gene_filtered"]


def test_config_defaults_legacy_and_gtf_mode_allows_missing_input_rna_coordinates(tmp_path):
    f = fixture(tmp_path)
    rna_path, atac_path = tmp_path / "rna.h5ad", tmp_path / "atac.h5ad"
    f[0].write_h5ad(rna_path)
    f[1].write_h5ad(atac_path)
    cfg = f[2]
    cfg["input"].update(rna_h5ad=str(rna_path), atac_h5ad=str(atac_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    loaded = io.load_config(config_path)
    rna, _ = io.load_data(loaded)
    assert "start" not in rna.var
    adapted = prepare((rna, f[1], loaded))[0]
    assert list(adapted.var.start) == [300, 100, 500, 700]
    config_path.write_text(yaml.safe_dump({"input": {"rna_h5ad": str(rna_path), "atac_h5ad": str(atac_path)}}))
    defaults = io.load_config(config_path)
    assert defaults["input"]["gene_identity_mode"] == "legacy"
    assert defaults["input"]["annotation_gtf"] is None
    with pytest.raises(KeyError, match="RNA"):
        io.load_data(defaults)


def execute_identity_subset(f):
    rna, atac, config = f
    prepared, targets, types, config, identity, _ = prepare(f)
    n = rna.n_obs
    pt = pd.DataFrame({"pseudotime": np.linspace(0., 1., n), "bin": np.repeat(np.arange(50), n // 50)}, index=rna.obs_names)
    edges = pd.DataFrame({"gene": ["E1", "E2"], "peak_id": ["p", "q"], "is_causal": [True, True]})
    weights = pd.DataFrame({"tf_gene": ["E3", "E3"], "peak_id": ["p", "q"], "weight": [.1, .1], "tf_lag": [0, 0]})
    def tf_fit(*args, **kwargs):
        assert kwargs["gene_identity"] is identity
        assert kwargs["target_gene_types"] == {"E3": "TF"}
        registry=saturating_transfer.fit_tf_peak(args[0],args[1],args[2],weights,args[4])
        kwargs['tf_model_registry'].update(registry)
        return weights.copy()
    with patch.object(run, "infer_pseudotime", return_value=pt), \
         patch.object(run, "granger_test", return_value=edges), \
         patch.object(run, "build_causal_grn", side_effect=lambda x: x.copy()), \
         patch.object(run, "fit_rna_to_atac", side_effect=tf_fit), \
         patch.object(run, "_prepare_adaptive_motif_resources", return_value={"motif_db": None, "tf_database": frozenset(), "resource_cache_dir": "/tmp/opencode"}):
        return run._run_modeling_for_subset(prepared, atac, targets, types, config,
                    rna_raw=prepared, gene_identity=identity), identity


def test_actual_hybrid_registry_engine_and_warm_checkpoints_keep_ids(tmp_path):
    f = fixture(tmp_path, cells=200)
    first, identity = execute_identity_subset(f)
    assert set(first["saturating_gene_metrics"].gene) == {"E1", "E2"}
    assert set(first["causal_edges"].gene) == {"E1", "E2"}
    assert set(first["tf_peak_weights"].tf_gene) == {"E3"}
    assert tuple(first["projection_inputs"][0]["features"]) == ("E2", "E1", "E3", "E4")
    ckpt_root = run._checkpoint_dir(run._joint_runtime_config(f[2]))
    payload = pickle.loads((ckpt_root / "step5_atac_to_rna.pkl").read_bytes())
    registry = payload["transfer_models"]
    assert set(registry["targets"]) == {"E1", "E2"}
    assert registry["context"]["gene_identity"]["fingerprint"] == identity.fingerprint
    with patch.object(run, "fit_atac_to_rna", side_effect=AssertionError("refit")):
        warm, _ = execute_identity_subset(f)
    pd.testing.assert_frame_equal(first["saturating_endpoints"], warm["saturating_endpoints"])


@pytest.mark.parametrize("change", ["reference_bytes", "mapping", "coordinates"])
def test_stage4_rejects_reference_changes_without_overwriting(tmp_path, change):
    f = fixture(tmp_path, cells=200)
    execute_identity_subset(f)
    root = run._checkpoint_dir(run._joint_runtime_config(f[2]))
    original = {p.name: p.read_bytes() for p in root.glob("*.pkl")}
    path = Path(f[2]["input"]["annotation_gtf"])
    text = path.read_text()
    if change == "reference_bytes":
        text += "# audit-only different reference bytes\n"
    elif change == "mapping":
        text = text.replace('gene_name "A"', 'gene_name "Renamed"')
    else:
        text = text.replace("\t101\t200\t", "\t102\t200\t")
    path.write_text(text)
    with pytest.raises(RuntimeError, match="step4_granger.*incompatible"):
        execute_identity_subset(f)
    assert {p.name: p.read_bytes() for p in root.glob("*.pkl")} == original


def test_wrong_identity_context_rejected_before_pseudotime(tmp_path, monkeypatch):
    f = fixture(tmp_path)
    rna, targets, types, cfg, identity, _ = prepare(f)
    rna.var.loc["E1", "gene_display_name"] = "wrong"
    monkeypatch.setattr(run, "infer_pseudotime", lambda *a, **k: pytest.fail("trajectory before identity validation"))
    with pytest.raises(ValueError, match="annotation differs"):
        run._run_modeling_for_subset(rna, f[1], targets, types, cfg, gene_identity=identity)


def test_provided_index_must_match_configured_reference_bytes(tmp_path):
    f = fixture(tmp_path)
    identity = parse_gene_gtf(f[2]["input"]["annotation_gtf"])
    path = Path(f[2]["input"]["annotation_gtf"])
    path.write_text(path.read_text() + "# changed\n")
    with pytest.raises(ValueError, match="configured GTF"):
        run._prepare_gene_identity_inputs(f[0], ["RUNX1"], {"RUNX1": "TF"}, f[2], gene_identity=identity)


@pytest.mark.parametrize("bad", ["mode", "gtf", "column", "marker"])
def test_invalid_identity_settings_and_marker_rejected(tmp_path, bad):
    f = fixture(tmp_path)
    if bad == "mode":
        f[2]["input"]["gene_identity_mode"] = "unknown"
    elif bad == "gtf":
        f[2]["input"]["annotation_gtf"] = None
    elif bad == "column":
        f[2]["input"]["rna_gene_id_column"] = "missing"
    else:
        f[2]["pseudotime"]["root_cell_marker"] = "A"
    with pytest.raises(ValueError):
        prepare(f)


def test_hybrid_public_gate_binds_identity_even_on_identical_numeric_axes(tmp_path):
    f = fixture(tmp_path, cells=200)
    output, identity = execute_identity_subset(f)
    root = run._checkpoint_dir(run._joint_runtime_config(f[2]))
    registry = pickle.loads((root / "step5_atac_to_rna.pkl").read_bytes())["transfer_models"]
    rna = apply_gene_identity(f[0], identity).adata
    tf_registry=pickle.loads((root/'step6_rna_to_atac.pkl').read_bytes())['tf_model_registry']
    kwargs=dict(transfer_models=registry,tf_model_registry=tf_registry)
    perturbation.propagate_perturbation(['E3'],rna,f[1],output['pseudotime'],output['causal_edges'],output['transfer_functions'],output['tf_peak_weights'],f[2],**kwargs)
    rna.uns["gene_identity"]["reference_sha256"] = "wrong"
    with pytest.raises(ValueError, match="identity|fingerprint|context"):
        perturbation.propagate_perturbation(['E3'],rna,f[1],output['pseudotime'],output['causal_edges'],output['transfer_functions'],output['tf_peak_weights'],f[2],**kwargs)


def test_identity_worker_knob_does_not_invalidate_stage4_or_refit(tmp_path):
    f = fixture(tmp_path, cells=200)
    execute_identity_subset(f)
    f[2]["rna_to_atac"]["motif_scan_workers"] = 2
    root = run._checkpoint_dir(run._joint_runtime_config(f[2]))
    before = {p.name: p.read_bytes() for p in root.glob("*.pkl")}
    with patch.object(run, "fit_atac_to_rna", side_effect=AssertionError("refit")):
        execute_identity_subset(f)
    assert {p.name: p.read_bytes() for p in root.glob("*.pkl")} == before


def test_present_duplicate_dataframe_indices_without_merging(tmp_path):
    identity = prepare(fixture(tmp_path))[4]
    frame = pd.DataFrame({"gene": ["E2", "E1"], "target_gene": ["PEAK:p", "E3,E1"]}, index=[7, 7])
    out = io._present_gene_frame(frame, identity)
    assert out.index.tolist() == [7, 7]
    assert out.gene.tolist() == ["A_02", "A_01"]
    assert out.target_gene.tolist() == ["PEAK:p", "RUNX1,A_01"]


def test_output_rejects_mismatched_reference_before_any_file_write(tmp_path):
    f = fixture(tmp_path)
    _, _, _, cfg, identity, tables = prepare(f)
    tables["gene_identity_metadata"]["reference_sha256"] = "wrong"
    with pytest.raises(ValueError, match="mismatched gene identity"):
        io.save_results(tables, cfg, gene_identity=identity)
    assert not (tmp_path / "results").exists()


def test_save_peak_pathway_preserves_exact_peak_sentinel(tmp_path):
    f = fixture(tmp_path)
    rna, _, _, cfg, identity, tables = prepare(f)
    result = {"pathway_edges": pd.DataFrame({"target_gene": ["PEAK:p"],
        "tf_name": ["peak"], "affected_gene": ["E1"], "peak_id": ["p"],
        "delta_rna": [.1]})}
    pathway = result["pathway_edges"]
    assert not pathway.empty and pathway.tf_name.eq("peak").all()
    before = pathway.copy(deep=True)
    io.save_results(dict(tables, **result), cfg, gene_identity=identity)
    saved = pd.read_csv(tmp_path / "results" / "perturbation_pathways.csv")
    assert saved.tf_name.eq("peak").all() and saved.tf_name_id.eq("peak").all()
    assert saved.target_gene.eq("PEAK:p").all()
    assert saved.affected_gene.eq("A_01").all() and saved.affected_gene_id.eq("E1").all()
    pd.testing.assert_frame_equal(pathway, before)


@pytest.mark.parametrize("column,value", [("gene_id", "E2"), ("gene_name", "wrong")])
def test_csv_presentation_rejects_existing_identity_conflicts_before_write(tmp_path, column, value):
    _, _, _, cfg, identity, tables = prepare(fixture(tmp_path))
    frame = pd.DataFrame({"gene": ["E1"], column: [value]})
    with pytest.raises(ValueError, match="conflicting"):
        io.save_results(dict(tables, causal_edges=frame), cfg, gene_identity=identity)
    assert not (tmp_path / "results").exists()


def test_kinetics_base_column_and_presentation_repeated_are_consistent(tmp_path):
    identity = prepare(fixture(tmp_path))[4]
    frame = pd.DataFrame({"tf_gene": ["E1", "E3"], "tf_gene_name": ["A", "RUNX1"]})
    first = io._present_gene_frame(frame, identity)
    pd.testing.assert_frame_equal(first, io._present_gene_frame(first, identity))
    assert first.tf_gene.tolist() == ["A_01", "RUNX1"]


def changed_normalizer(value):
    return identity_module.normalize_gene_id_original(value)


def changed_base_map(rna, gene_identity=None, config=None):
    return kinetics._gene_base_map_original(rna, gene_identity, config)


def test_identity_helper_bodies_and_schema_are_stage_scoped(monkeypatch):
    stages = ("step4_granger", "step5_atac_to_rna", "step6_rna_to_atac")
    baseline = {s: run._stage_numerical_signatures(s, gene_identity_mode="gtf_gene_id") for s in stages}
    legacy = {s: run._stage_numerical_signatures(s) for s in stages}
    monkeypatch.setattr(identity_module, "normalize_gene_id_original", identity_module.normalize_gene_id, raising=False)
    monkeypatch.setattr(identity_module, "normalize_gene_id", changed_normalizer)
    for stage in stages:
        current = run._stage_numerical_signatures(stage, gene_identity_mode="gtf_gene_id")
        assert current["functions"]["CausalBridge.gene_identity.normalize_gene_id"] != baseline[stage]["functions"]["CausalBridge.gene_identity.normalize_gene_id"]
        assert run._stage_numerical_signatures(stage) == legacy[stage]
        assert "CausalBridge.gene_identity.annotate_gene_columns" not in current["functions"]
    monkeypatch.setattr(identity_module, "SCHEMA", "gene_identity_future")
    assert run._stage_numerical_signatures(stages[0], gene_identity_mode="gtf_gene_id")["constants"]["gene_identity.SCHEMA"] == "gene_identity_future"


def test_changed_identity_helper_rejects_real_checkpoint_ancestor(monkeypatch, tmp_path):
    f = fixture(tmp_path, cells=200)
    execute_identity_subset(f)
    root = run._checkpoint_dir(run._joint_runtime_config(f[2]))
    before = {p.name: p.read_bytes() for p in root.glob("*.pkl")}
    monkeypatch.setattr(identity_module, "normalize_gene_id_original", identity_module.normalize_gene_id, raising=False)
    monkeypatch.setattr(identity_module, "normalize_gene_id", changed_normalizer)
    with pytest.raises(RuntimeError, match="step4_granger.*incompatible"):
        execute_identity_subset(f)
    assert {p.name: p.read_bytes() for p in root.glob("*.pkl")} == before


def test_tf_base_helper_change_invalidates_only_step6(monkeypatch, tmp_path):
    f = fixture(tmp_path, cells=200)
    execute_identity_subset(f)
    root = run._checkpoint_dir(run._joint_runtime_config(f[2]))
    before = {p.name: p.read_bytes() for p in root.glob("*.pkl")}
    baseline4 = run._stage_numerical_signatures("step4_granger", gene_identity_mode="gtf_gene_id")
    monkeypatch.setattr(kinetics, "_gene_base_map_original", kinetics._gene_base_map, raising=False)
    monkeypatch.setattr(kinetics, "_gene_base_map", changed_base_map)
    assert run._stage_numerical_signatures("step4_granger", gene_identity_mode="gtf_gene_id") == baseline4
    with pytest.raises(RuntimeError, match="step6_rna_to_atac.*incompatible"):
        execute_identity_subset(f)
    assert {p.name: p.read_bytes() for p in root.glob("*.pkl")} == before
