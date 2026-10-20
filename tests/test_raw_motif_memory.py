"""Legacy pickle compatibility, bounded record retention, and streamed unions."""
import pickle
import subprocess
import sys

import pytest

from CausalBridge import kinetics as k
from CausalBridge._raw_motif_io import DiskRecords, read_records, write_records


def resources():
    motifs = {f"m{i}": {"tf_name": f"TF{i}", "all_factors": []} for i in range(12)}
    descriptor = {"schema": k.RAW_MOTIF_SCHEMA, "scan_key": "test",
                  "identity": {"threshold": 1.0}}
    return motifs, descriptor


def record(i):
    return ((f"chr1:{i}-{i+10}", "chr1", i, i + 10),
            [{"motif_id": f"m{j}", "p_value": float((j % 3 + 1) / 10),
              "max_score": float(i + j)} for j in range(12)])


def test_tens_of_thousands_legacy_records_selective_topn_and_union(tmp_path):
    motifs, metadata = resources()
    path = tmp_path / "raw.pkl"
    original = dict(record(i) for i in range(20000))
    original[record(7)[0]] = []  # verified negative is distinct from missing
    payload = dict(metadata, records=original)
    path.write_bytes(pickle.dumps(payload, protocol=4))
    keys = [record(i)[0] for i in (7, 999, 19999)]
    selected = k._read_raw_motif_cache(path, metadata, motifs,
                                     requested_keys=keys, max_tfs=5)
    assert selected.valid
    assert list(selected) == keys
    assert selected == {key: k._materialize_raw_motif_hits(original[key], motifs, 5) for key in keys}
    additions = dict([record(20001), record(999)])
    legacy_before = path.read_bytes()
    k._merge_raw_motif_cache(path, metadata, motifs, additions)
    expected = dict(payload, records=original.copy())
    expected["records"].update(additions)
    actual = dict(metadata, records=dict(k._read_raw_motif_cache(path, metadata, motifs)))
    assert actual == expected
    assert list(actual) == list(expected)
    assert list(actual["records"]) == list(expected["records"])
    assert type(actual) is dict and type(actual["records"]) is dict
    assert type(actual["records"][record(999)[0]][0]["max_score"]) is float
    # A second update must reference the original memo, even after a previous
    # incremental stream inserted no-memo expressions into the pickle.
    k._merge_raw_motif_cache(path, metadata, motifs, {record(20002)[0]: []})
    assert len(k._read_raw_motif_cache(path, metadata, motifs)) == 20002
    assert path.read_bytes() == legacy_before


def test_legacy_read_and_incremental_write_under_128_mib_address_limit(tmp_path):
    _, metadata = resources()
    path = tmp_path / "large.pkl"
    path.write_bytes(pickle.dumps(dict(metadata, records=dict(record(i) for i in range(20000))), protocol=4))
    # Import only the primitive IO helper: CUDA/AnnData libraries are unrelated
    # to this regression and reserve large virtual address ranges at startup.
    script = r'''
import importlib.util, pathlib, resource, sys
resource.setrlimit(resource.RLIMIT_AS, (128*1024**2, 128*1024**2))
spec = importlib.util.spec_from_file_location("raw_io", sys.argv[1])
io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)
count = 0
def consume(key, hits):
    global count
    assert len(hits) == 12
    count += 1
path = pathlib.Path(sys.argv[2])
with path.open("rb") as handle:
    payload, memo, stop = io.read_records(handle, consume)
assert count == 20000
payload.pop("records")
with path.with_suffix(".merged").open("wb") as handle:
    io.write_records(handle, payload, {("extra", "chr1", 0, 10): []},
                     previous=path, records_memo=memo, stop_offset=stop)
print("20000 peaks read and merged under 128 MiB RLIMIT_AS")
'''
    from CausalBridge import _raw_motif_io
    result = subprocess.run([sys.executable, "-c", script, _raw_motif_io.__file__, str(path)],
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "under 128 MiB" in result.stdout
    with path.with_suffix(".merged").open("rb") as handle:
        assert len(pickle.load(handle)["records"]) == 20001


def test_stream_reader_drops_unrequested_hit_graph_and_keeps_aliases(tmp_path):
    _, metadata = resources()
    first, hits = record(0)
    second = record(1)[0]
    third = record(2)[0]
    # Genuine cross-record aliases in a legacy cache remain valid.
    payload = dict(metadata, records={first: hits, second: hits, third: []})
    path = tmp_path / "alias.pkl"
    path.write_bytes(pickle.dumps(payload, protocol=4))
    seen = {}
    with path.open("rb") as handle:
        header, _, _ = read_records(handle, lambda key, value: seen.setdefault(key, value))
    assert seen == payload["records"] and seen[first] is seen[second]
    assert header["records"] == {}  # the sink never retains complete peaks


def test_no_memo_growth_for_new_cache_and_disk_scan_results(tmp_path):
    motifs, metadata = resources()
    path = tmp_path / "new.pkl"
    with DiskRecords() as store:
        store.update([record(i) for i in range(20)])
        directory = k.Path(store.directory.name)
        with path.open("wb") as handle:
            write_records(handle, metadata, store)
        assert list(store.items()) == [record(i) for i in range(20)]
    assert not directory.exists()
    import pickletools
    memo_ops = [op.name for op, _, _ in pickletools.genops(path.read_bytes())
                if op.name in {"BINPUT", "LONG_BINPUT", "MEMOIZE"}]
    assert memo_ops == ["BINPUT", "BINPUT"]  # root and records only
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    assert payload == dict(metadata, records=dict(record(i) for i in range(20)))
    loaded = k._read_raw_motif_cache(path, metadata, motifs,
                                    requested_keys=[record(19)[0]], max_tfs=5)
    assert loaded.valid and loaded[record(19)[0]] == ["TF0", "TF3", "TF6", "TF9", "TF1"]


def test_streaming_validation_does_not_skip_unrequested_corruption(tmp_path):
    motifs, metadata = resources()
    payload = dict(metadata, records=dict([record(0), record(1)]))
    payload["records"][record(1)[0]][0]["p_value"] = 2.0
    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps(payload, protocol=4))
    with pytest.raises(ValueError, match="invalid motif hit"):
        k._read_raw_motif_cache(path, metadata, motifs,
                                requested_keys=[record(0)[0]], max_tfs=5)
    assert not path.with_suffix(".sqlite").exists()


def test_disk_results_cleanup_on_worker_failure(tmp_path):
    with pytest.raises(RuntimeError):
        with DiskRecords() as store:
            directory = k.Path(store.directory.name)
            store.update([record(0)])
            raise RuntimeError("worker failed")
    assert not directory.exists()
