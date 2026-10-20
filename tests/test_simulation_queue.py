"""Real fit/simulation parity across the disk-backed spawn boundary."""
import json
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from CausalBridge import run, kinetics, simulation_queue as queue
from test_saturating_pipeline import public_fixture


@pytest.mark.parametrize("options", [
    {"defer_cluster_simulation": 1}, {"simulation_workers": True},
    {"simulation_workers": 0}, {"simulation_workers": -1},
    {"simulation_workers": 1.5}, {"simulation_workers": "4"}])
def test_execution_validation(options):
    with pytest.raises(ValueError):
        queue.execution_options({"execution": options})


def test_default_and_thread_environment(monkeypatch):
    assert queue.execution_options({}) == (False, 2)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    import os
    with queue._worker_environment():
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "7"


@pytest.mark.parametrize("workers", [1, 2, 3, 4, 32])
def test_positive_simulation_concurrency(workers):
    assert queue.execution_options({"execution": {"defer_cluster_simulation": True,
        "simulation_workers": workers}}) == (True, workers)


def test_artifact_write_never_overwrites(tmp_path):
    path = tmp_path / "snapshot.pkl"
    queue._write_pickle(path, {"original": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        queue._write_pickle(path, {"original": False})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def assert_same(actual, expected):
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)
        assert_same(actual.attrs, expected.attrs)
    elif isinstance(expected, np.ndarray):
        if expected.dtype.kind in "biufc":
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        else:
            np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_same(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("workers,ko_type,smooth", [(1, "gene", "hard"), (2, "gene", "gaussian"), (2, "peak", "hard"), (4, "gene", "hard")])
def test_real_deferred_spawn_parity(tmp_path, monkeypatch, workers, ko_type, smooth):
    rna, atac, pt, edges, cfg = public_fixture(smooth)
    cfg["output"].update(dir=str(tmp_path), checkpoint_dir=str(tmp_path / "checkpoints"), cache_dir=str(tmp_path / "cache"))
    cfg["pseudotime"].update(n_neighbors=10, method="dpt")
    cfg["granger"] = {"reference_bins": 24, "min_composite_score": .3}
    cfg["perturbation"].update(knockout_type=ko_type)
    cfg["perturbation"]["peak_ko"]["peak_id"] = "chr1:10-20"
    resources = {"motif_db": {"synthetic": {"tf_name": "TF", "pwm": np.full((4, 6), .25)}},
                 "tf_database": frozenset({"TF"}), "resource_cache_dir": str(tmp_path / "cache")}
    monkeypatch.setattr(run, "infer_pseudotime", lambda *a, **k: pt.copy(deep=True))
    monkeypatch.setattr(run, "granger_test", lambda *a, **k: edges.assign(is_causal=True))
    monkeypatch.setattr(run, "_prepare_adaptive_motif_resources", lambda *a, **k: resources)
    monkeypatch.setattr(kinetics, "_scan_peaks_with_jaspar", lambda *a, **k: {"chr1:10-20": ["TF"]})
    counts = {"gene": 0, "tf": 0}
    original_gene, original_tf = run.fit_atac_to_rna, run.fit_rna_to_atac
    def gene_fit(*args, **kwargs):
        counts["gene"] += 1
        return original_gene(*args, **kwargs)
    def tf_fit(*args, **kwargs):
        counts["tf"] += 1
        return original_tf(*args, **kwargs)
    monkeypatch.setattr(run, "fit_atac_to_rna", gene_fit)
    monkeypatch.setattr(run, "fit_rna_to_atac", tf_fit)
    manifests, expected = [], []
    for suffix in ("cluster_first", "cluster_second"):
        expected.append(run._run_modeling_for_subset(rna, atac, ["TF"], {"TF": "TF"}, cfg, suffix=suffix))
        manifests.append(run._run_modeling_for_subset(rna, atac, ["TF"], {"TF": "TF"}, cfg,
            suffix=suffix, defer_simulation=True)["simulation_manifest"])
    assert counts == {"gene": 2, "tf": 2}
    snapshots = {p: p.read_bytes() for p in (tmp_path / "checkpoints").rglob("*.pkl")}
    # These parent hooks cannot be inherited by spawned children. The child
    # entry point's source contract below separately excludes the fit runner.
    monkeypatch.setattr(run, "_run_modeling_for_subset", lambda *a, **k: pytest.fail("worker trained"))
    results = list(queue.run_simulation_jobs(manifests, workers=workers))
    from CausalBridge.result_store import DiskResults
    assert all(isinstance(result, DiskResults) for result in results)
    # Parent receives field references, not serialized full cohort payloads.
    import pickle
    assert all(len(pickle.dumps(result)) < 2048 for result in results)
    for actual, before in zip(results, expected):
        assert_same(actual, before)
    assert all(p.read_bytes() == payload for p, payload in snapshots.items())
    assert counts == {"gene": 2, "tf": 2}
    if workers == 1:
        _check_pipeline_barrier(tmp_path, monkeypatch, rna, atac, cfg, manifests, expected, results)
    restored = queue.load_simulation_inputs(manifests[0])
    assert restored["rna_adata"].X.dtype == rna.X.dtype
    np.testing.assert_array_equal(restored["rna_adata"].X.toarray(), rna.X.toarray())
    pd.testing.assert_frame_equal(restored["pseudotime_df"], pt)
    # Tampering or another run replacing a checkpoint must fail, never refit.
    manifest = json.loads(Path(manifests[0]).read_text())
    checkpoint = Path(manifest["checkpoints"]["step6_rna_to_atac"]["path"])
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        queue.load_simulation_inputs(manifests[0])


def _check_pipeline_barrier(tmp_path, monkeypatch, rna, atac, cfg, manifests, expected, results):
    """Exercise the real outer runner/merges while reusing verified cohort outputs."""
    import anndata as ad
    import yaml
    from CausalBridge import io
    combined_rna = ad.concat({"first": rna, "second": rna}, label="leiden", index_unique="_")
    combined_atac = ad.concat({"first": atac, "second": atac}, index_unique="_")
    dummy = tmp_path / "input.h5ad"
    dummy.touch()
    settings = copy.deepcopy(cfg)
    settings["input"] = {"rna_h5ad": str(dummy), "atac_h5ad": str(dummy), "target_genes": "unused"}
    settings["clustering"] = {"enabled": True, "paga": {"enabled": False}, "min_cells_per_cluster": 50}
    settings["pseudotime"].update(root_cell_method="marker", root_cell_marker="TF")
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(settings))
    config = io.load_config(path)
    monkeypatch.setattr(run, "load_config", lambda *a: copy.deepcopy(config))
    monkeypatch.setattr(run, "load_data", lambda *a: (combined_rna.copy(), combined_atac.copy()))
    monkeypatch.setattr(run, "load_target_genes", lambda *a: (["TF"], {"TF": "TF"}))
    monkeypatch.setattr(run, "preprocess_rna", lambda data, **k: data)
    monkeypatch.setattr(run, "preprocess_atac", lambda data, **k: data)
    monkeypatch.setattr(run, "save_hvg_genes", lambda *a: None)
    monkeypatch.setattr(run, "save_cluster_top_genes", lambda *a, **k: None)
    monkeypatch.setattr(run, "save_results", lambda *a, **k: None)
    events = []
    def prepared(*a, suffix=None, defer_simulation=False, **k):
        index = 0 if suffix == "cluster_first" else 1
        events.append(("train", suffix))
        if defer_simulation:
            return {"simulation_manifest": manifests[index]}
        return copy.deepcopy(expected[index])
    def simulate(paths, workers):
        assert events == [("train", "cluster_first"), ("train", "cluster_second")]
        assert paths == manifests
        events.append(("simulate", workers))
        yield from copy.deepcopy(results)
    monkeypatch.setattr(run, "_run_modeling_for_subset", prepared)
    monkeypatch.setattr(queue, "run_simulation_jobs", simulate)
    sequential = run.run_pipeline(str(path))
    events.clear()
    config["execution"] = {"defer_cluster_simulation": True, "simulation_workers": 2}
    deferred = run.run_pipeline(str(path))
    assert events[-1] == ("simulate", 2)
    assert_same(deferred, sequential)


def test_worker_has_no_training_entry():
    import inspect
    for function in (queue.simulate_job, queue.load_simulation_inputs, run._finish_modeling_for_subset):
        source = inspect.getsource(function)
        assert "_run_modeling_for_subset(" not in source
        assert "fit_atac_to_rna(" not in source
        assert "fit_rna_to_atac(" not in source


def test_adaptive_snapshot_preserves_frozen_plan_and_anndata(tmp_path, monkeypatch):
    from test_pipeline_bin_plan import inputs, install_stubs
    rna, atac, cfg = inputs(tmp_path)
    rna.layers["preserved"] = rna.X.astype(np.float32)
    rna.uns["audit"] = {"root": "cell17", "version": "test"}
    seen = install_stubs(monkeypatch)
    manifest = run._run_modeling_for_subset(rna, atac[:], ["KO"], {"KO": "TF"}, cfg,
        suffix="adaptive", defer_simulation=True)["simulation_manifest"]
    assert [stage for stage, _, _ in seen] == [4, 5, 6]
    original_plan = seen[0][1]
    monkeypatch.setattr(run, "fit_context_binning", lambda *a, **k: pytest.fail("rebin"))
    restored = queue.load_simulation_inputs(manifest)
    assert not restored["atac_adata"].is_view
    plan = restored["plan_kwargs"]["bin_plan"]
    assert plan.fingerprint == original_plan.fingerprint
    for name in ("cell_ids", "centers", "times", "branches", "weights", "operator"):
        if hasattr(plan, name):
            left, right = getattr(plan, name), getattr(original_plan, name)
            if hasattr(left, "toarray"):
                left, right = left.toarray(), right.toarray()
            assert_same(left, right)
    assert restored["rna_adata"].uns["audit"] == rna.uns["audit"]
    layer = restored["rna_adata"].layers["preserved"]
    assert layer.dtype == np.float32
    np.testing.assert_array_equal(layer.toarray(), rna.layers["preserved"].toarray())
