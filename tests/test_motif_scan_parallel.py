"""主 motif 扫描的真实 spawn、一致性及缓存测试。"""
import os
import pickle

import numpy as np
import pytest

from CausalBridge import kinetics as k
from CausalBridge.io import load_config
from test_shared_motif_cache import fixture, scan, cache_path


def motifs():
    rng = np.random.default_rng(91)
    return {str(i): {"pwm": k._pfm_to_pwm(rng.uniform(0, 20, (4, 5 + i))),
                     "tf_name": str(i)} for i in range(4)}


@pytest.mark.parametrize("workers", [2, 4])
def test_spawn_exact_order_and_environment(workers):
    db = motifs()
    sequences = ["ACGTGCAT" * 80, "NNNNNN", "ACG", "acgt" * 150,
                 "AAAAACGT" * 80, "ACGTNNACGT" * 70] * 2
    tasks = list(enumerate(sequences))
    before = dict(os.environ)
    expected = k._scan_sequence_records(tasks, db, 1., 1)
    actual = k._scan_sequence_records(tasks, db, 1., workers)
    assert actual == expected and list(actual) == list(expected)
    assert actual[1] == actual[2] == [] and len(actual[0]) > 1
    assert dict(os.environ) == before
    assert all(info["pwm"].flags.writeable for info in db.values())
    assert k._scan_sequence_records(tasks, db, 0., workers) == {key: [] for key, _ in tasks}


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, 2., "24", None])
def test_invalid_config_and_direct_scan(tmp_path, value):
    config = tmp_path / "config.yaml"
    config.write_text("rna_to_atac:\n  motif_scan_workers: " + repr(value).replace("None", "null") + "\n")
    with pytest.raises(ValueError, match="motif_scan_workers"):
        load_config(str(config))
    f = fixture(tmp_path)
    f[3]["rna_to_atac"]["motif_scan_workers"] = value
    with pytest.raises(ValueError, match="motif_scan_workers"):
        scan(f, ["A"])


def test_default_config(tmp_path):
    data = tmp_path / "data"
    data.touch()
    config = tmp_path / "config.yaml"
    config.write_text(f"input:\n  rna_h5ad: {data}\n  atac_h5ad: {data}\n")
    assert load_config(str(config))["rna_to_atac"]["motif_scan_workers"] == 1


def test_single_cpu_cap_no_pool(monkeypatch, caplog):
    caplog.set_level("INFO")
    monkeypatch.setattr(k.os, "sched_getaffinity", lambda _: {0})
    monkeypatch.setattr(k, "ProcessPoolExecutor", lambda **kw: pytest.fail("串行创建了池"))
    tasks = [(i, "ACGT" * 20) for i in range(3)]
    assert k._scan_sequence_records(tasks, motifs(), 1., 24)
    assert "requested_workers=24 effective_workers=1" in caplog.text
    assert k._scan_sequence_records([], motifs(), 1., 24) == {}


def test_cache_identity_warm_pool_skipped_parent_writer(tmp_path, monkeypatch):
    f = fixture(tmp_path)
    cfg = f[3]["rna_to_atac"]
    first = scan(f, ["A", "ZERO"])
    path = cache_path(f)
    before = path.read_bytes()
    cfg["motif_scan_workers"] = 4
    assert cache_path(f) == path
    original_pool = k.ProcessPoolExecutor
    monkeypatch.setattr(k, "ProcessPoolExecutor", lambda **kw: pytest.fail("暖缓存创建了池"))
    assert scan(f, ["A", "ZERO"]) == first
    assert path.read_bytes() == before
    monkeypatch.setattr(k, "ProcessPoolExecutor", original_pool)
    parent = os.getpid()
    merge = k._merge_raw_motif_cache
    writes = []
    def guarded_merge(*args):
        assert os.getpid() == parent
        writes.append(parent)
        return merge(*args)
    monkeypatch.setattr(k, "_merge_raw_motif_cache", guarded_merge)
    scan(f, ["A", "B", "ZERO"])
    assert writes == [parent]
    with path.open("rb") as handle:
        assert len(pickle.load(handle)["records"]) == 3


def test_worker_exception_leaves_cache_intact(tmp_path, monkeypatch):
    f = fixture(tmp_path)
    scan(f, ["A"])
    path = cache_path(f)
    before = path.read_bytes()
    f[3]["rna_to_atac"]["motif_scan_workers"] = 2
    monkeypatch.setattr(k, "_extract_peak_sequences", lambda *a, **kw: {"B": 123, "ZERO": "ACGT" * 20})
    env = dict(os.environ)
    with pytest.raises(AttributeError):
        scan(f, ["A", "B", "ZERO"])
    assert path.read_bytes() == before and dict(os.environ) == env


def test_worker_initialization_readonly(monkeypatch):
    db = motifs()
    monkeypatch.setattr(k.Path, "open", lambda *a, **kw: pytest.fail("worker 打开文件"))
    previous = (k._MOTIF_WORKER_DB, k._MOTIF_WORKER_THRESHOLD)
    try:
        k._init_motif_scan_worker(tuple((key, info["pwm"].copy()) for key, info in db.items()), 1.)
        with pytest.raises(TypeError):
            k._MOTIF_WORKER_DB["new"] = {}
        with pytest.raises(ValueError):
            k._MOTIF_WORKER_DB["0"]["pwm"][0, 0] = 0
    finally:
        if k._MOTIF_THREAD_LIMIT is not None:
            k._MOTIF_THREAD_LIMIT.restore_original_limits()
        k._MOTIF_WORKER_DB, k._MOTIF_WORKER_THRESHOLD = previous
