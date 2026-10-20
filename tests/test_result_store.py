"""Disk-backed results keep table semantics without keeping decoded objects."""
import copy
import gc
import pickle
import weakref
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from CausalBridge import io
from CausalBridge.result_store import DiskResults, ExportResults


def test_round_trip_no_ram_cache_and_explicit_updates(tmp_path):
    frame = pd.DataFrame({"value": np.arange(10000, dtype=float)})
    frame.attrs = {"provenance": ["unchanged"]}
    store = DiskResults.create(tmp_path, {"table": frame, "meta": {"cluster": "0"}})
    original = weakref.ref(frame)
    del frame
    gc.collect()
    assert original() is None
    loaded = store["table"]
    reference = weakref.ref(loaded)
    assert loaded.attrs == {"provenance": ["unchanged"]}
    loaded.iloc[0, 0] = -1
    assert store["table"].iloc[0, 0] == 0
    store["table"] = loaded
    del loaded
    gc.collect()
    assert reference() is None
    restored = pickle.loads(pickle.dumps(store))
    assert len(pickle.dumps(store)) < 1024
    assert restored["table"].iloc[0, 0] == -1
    assert list(restored) == ["table", "meta"]
    assert copy.deepcopy(restored)["meta"] == {"cluster": "0"}
    del store["meta"]
    assert "meta" not in store
    with pytest.raises(KeyError):
        store["missing"]


def test_failed_field_write_preserves_previous_value(tmp_path):
    store = DiskResults.create(tmp_path, {"original": {"keep": True}})
    before = {p.name: p.read_bytes() for p in Path(store.directory).iterdir()}
    with pytest.raises((AttributeError, pickle.PicklingError)):
        store["original"] = lambda: None
    assert store["original"] == {"keep": True}
    after = {p.name: p.read_bytes() for p in Path(store.directory).iterdir()}
    assert after == before


def test_lazy_export_skips_unrequested_payload(tmp_path, monkeypatch):
    store = DiskResults.create(tmp_path / "store", {
        "perturbation_results": pd.DataFrame({"delta_rna": [.3]}),
        "saturating_response_curves": pd.DataFrame({"huge": range(100)}),
        "projection_inputs": np.zeros((100, 100)),
    })
    original = DiskResults.__getitem__
    reads = []
    def read(self, key):
        reads.append(key)
        assert key != "saturating_response_curves", "disabled detail was deserialized"
        return original(self, key)
    monkeypatch.setattr(DiskResults, "__getitem__", read)
    view = ExportResults(store, excluded={"saturating_response_curves"})
    assert list(view) == ["perturbation_results", "projection_inputs"]
    assert not reads
    io.save_results(store, {"output": {"dir": str(tmp_path / "csv"), "export_details": False,
        "save_intermediate": False}, "atac_to_rna": {"model_mode": "saturating_hybrid"}})
    assert (tmp_path / "csv/perturbation_results.csv").exists()
    assert "saturating_response_curves" not in reads


@pytest.mark.parametrize("details", [False, True])
def test_disk_and_memory_export_files_identical(tmp_path, details):
    frame = pd.DataFrame({"value": [1., np.nan, 3.]})
    data = {key: frame.copy() for key in io.RESULT_CSV_EXPORTS}
    data["perturbation_results"] = pd.DataFrame({"cluster": ["b", "a"], "delta_rna": [-.3, .2]})
    data["cluster_results"] = [{"binning_metadata": {"context": "a"}}]
    data["saturating_context_metadata"] = [{"context": "a"}]
    data["extra_intermediate"] = frame.copy()
    store = DiskResults.create(tmp_path / "fields", data)
    for name, results in [("ram", data), ("disk", store)]:
        io.save_results(results, {"output": {"dir": str(tmp_path / name),
            "export_details": details, "save_intermediate": True},
            "atac_to_rna": {"model_mode": "saturating_hybrid"},
            "perturbation": {"cell_projection": {"enabled": True}}})
    files = lambda name: {p.relative_to(tmp_path / name): p.read_bytes()
                          for p in (tmp_path / name).rglob('*') if p.is_file()}
    assert files("ram") == files("disk")


def test_gene_identity_presentation_is_equal_for_disk_results(tmp_path):
    from test_gene_identity_pipeline import fixture, prepare
    _, _, _, config, identity, tables = prepare(fixture(tmp_path))
    data = dict(tables, causal_edges=pd.DataFrame({"gene": ["E1", "E2"], "peak_id": ["p", "q"]}),
                perturbation_results=pd.DataFrame({"target_gene": ["E3", "E3"],
                    "affected_gene": ["E1", "E2"], "delta_rna": [.1, -.2]}))
    store = DiskResults.create(tmp_path / "fields", data)
    for name, results in [("ram", data), ("disk", store)]:
        settings = copy.deepcopy(config)
        settings["output"]["dir"] = str(tmp_path / name)
        io.save_results(results, settings, gene_identity=identity)
    for filename in ["causal_peak_gene_edges.csv", "perturbation_results.csv", "gene_identity.csv"]:
        assert (tmp_path / "ram" / filename).read_bytes() == (tmp_path / "disk" / filename).read_bytes()
    assert store["causal_edges"].gene.tolist() == ["E1", "E2"]
