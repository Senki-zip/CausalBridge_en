"""Real tiny FASTA/PWM cache tests; no network or biological training jobs."""
from concurrent.futures import ThreadPoolExecutor
import pickle
import threading

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from CausalBridge import kinetics as k, run


def fixture(tmp_path):
    genome = tmp_path / "tiny.fa"
    sequence = "ACGTGCAATG" * 20 + "N" * 50
    genome.write_text(">chr1\n" + sequence + "\n")
    motifs = {"m1": {"pwm": np.tile(np.array([2., -1., 0., 1.])[:, None], (1, 5)),
                      "tf_name": "TF::PART", "all_factors": [{"factor": "CO"}]},
              "m2": {"pwm": np.tile(np.array([-1., 2., 1., 0.])[:, None], (1, 5)),
                      "tf_name": "ALT", "all_factors": []}}
    atac = ad.AnnData(np.ones((3, 4)), var=pd.DataFrame({"chr": ["chr1"]*4,
        "start": [0, 80, 200, 0], "end": [80, 160, 250, 80]}, index=["A", "B", "ZERO", "MISSING"]))
    atac.var.loc["MISSING", "chr"] = "absent"
    rna = ad.AnnData(np.ones((3, 4)), var=pd.DataFrame(index=["CO", "TF", "PART", "ALT"]))
    cfg = {"input": {"genome_fasta": str(genome)}, "output": {"cache_dir": str(tmp_path / "base")},
           "rna_to_atac": {"motif_pval_threshold": 1., "max_tfs_per_peak": 1}}
    return rna, atac, motifs, cfg


def scan(f, peaks, *, descriptor=None, frozen=frozenset()):
    rna, atac, motifs, config = f
    edges = pd.DataFrame({"peak_id": peaks, "gene": ["TF"]*len(peaks)})
    return k._scan_peaks_with_jaspar(atac, rna, edges, motifs, config["rna_to_atac"], config, frozen, descriptor)


def spy_scan(monkeypatch):
    original = k._scan_one_sequence
    calls = []
    def scanning(*args):
        calls.append(args[0])
        return original(*args)
    monkeypatch.setattr(k, "_scan_one_sequence", scanning)
    return calls


def cache_path(f):
    descriptor = k._raw_scan_cache_descriptor(f[3], f[2])
    return k.Path(descriptor["base_cache_dir"]) / "peak_motif_raw_v1" / (descriptor["scan_key"] + ".pkl")


def test_repeat_changed_model_rna_bins_topn_no_extract_or_scan(tmp_path, monkeypatch):
    f = fixture(tmp_path)
    calls = spy_scan(monkeypatch)
    first = scan(f, ["A", "B"])
    assert len(calls) == 2 and all(len(names) == 1 for names in first.values())
    path = cache_path(f)
    before = path.read_bytes()
    f[3].update(pseudotime={"n_bins": 99}, granger={"max_lag": 4}, targets=["OTHER"], cluster="different")
    f[0].X *= 3
    f[3]["rna_to_atac"]["max_tfs_per_peak"] = 4
    monkeypatch.setattr(k, "_extract_peak_sequences", lambda *args, **kwargs: pytest.fail("covered request extracted FASTA"))
    original_open = k.Path.open
    genome = k.Path(f[3]["input"]["genome_fasta"])
    def guarded_open(path, *args, **kwargs):
        assert path.resolve() != genome.resolve(), "covered request reopened FASTA for hashing"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(k.Path, "open", guarded_open)
    second = scan(f, ["A", "B"])
    assert len(calls) == 2  # two requests: scan counts 2, then 0
    assert all(len(names) == 4 for names in second.values())
    assert path == cache_path(f) and path.read_bytes() == before
    second["A"].clear()
    assert scan(f, ["A"])["A"]  # materialized lists never alias cached records


def test_subset_union_and_negative_hits(tmp_path, monkeypatch):
    f = fixture(tmp_path)
    calls = spy_scan(monkeypatch)
    original = k._extract_peak_sequences
    requested = []
    def extraction(*args, **kwargs):
        requested.append(kwargs["requested_peaks"])
        return original(*args, **kwargs)
    monkeypatch.setattr(k, "_extract_peak_sequences", extraction)
    scan(f, ["A"])
    scan(f, ["B"])
    scan(f, ["A", "B"])
    assert len(calls) == 2 and requested == [["A"], ["B"]]
    assert scan(f, ["ZERO"]) == {}
    assert len(calls) == 3
    monkeypatch.setattr(k, "_extract_peak_sequences", lambda *args, **kwargs: pytest.fail("cached zero extracted"))
    assert scan(f, ["ZERO"]) == {} and len(calls) == 3
    with cache_path(f).open("rb") as handle:
        records = pickle.load(handle)["records"]
    assert len(records) == 3 and records[("ZERO", "chr1", 200, 250)] == []


@pytest.mark.parametrize("change", ["genome", "pwm", "mapping", "threshold", "version", "motif_order"])
def test_identity_invalidation(tmp_path, monkeypatch, change):
    f = fixture(tmp_path)
    calls = spy_scan(monkeypatch)
    scan(f, ["A"])
    before = cache_path(f)
    if change == "genome":
        k.Path(f[3]["input"]["genome_fasta"]).write_text(">chr1\n" + "TGCA"*60 + "\n")
    elif change == "pwm":
        f[2]["m1"]["pwm"][0, 0] += .1
    elif change == "mapping":
        f[2]["m1"]["all_factors"][0]["factor"] = "NEW"
    elif change == "threshold":
        f[3]["rna_to_atac"]["motif_pval_threshold"] = .9
    elif change == "version":
        monkeypatch.setattr(k, "RAW_MOTIF_SCANNER_VERSION", "changed")
    else:
        f = (f[0], f[1], dict(reversed(list(f[2].items()))), f[3])
    scan(f, ["A"])
    assert len(calls) == 2 and cache_path(f) != before


def test_coordinate_change_same_id_and_partial_failure_not_negative(tmp_path, monkeypatch):
    f = fixture(tmp_path)
    calls = spy_scan(monkeypatch)
    assert set(scan(f, ["A", "MISSING"])) == {"A"}
    assert len(calls) == 1
    f[1].var.loc["A", ["start", "end"]] = [10, 90]
    scan(f, ["A"])
    assert len(calls) == 2
    with cache_path(f).open("rb") as handle:
        records = pickle.load(handle)["records"]
    assert len(records) == 2 and not any(key[0] == "MISSING" for key in records)


@pytest.mark.parametrize("mode", ["missing", "unusable"])
def test_failed_genome_frozen_fallback_never_cached(tmp_path, monkeypatch, mode):
    f = fixture(tmp_path)
    genome = k.Path(f[3]["input"]["genome_fasta"])
    if mode == "missing": genome.unlink()
    else: genome.write_text(">absent\nAAAAAA\n")
    monkeypatch.setattr(k, "_load_tf_database", lambda *args: pytest.fail("frozen TF reloaded"))
    assert scan(f, ["A"], frozen=frozenset({"TF"})) == {"A": ["TF"]}
    assert scan(f, ["A"], frozen=frozenset()) == {}
    assert not cache_path(f).exists()


def test_descriptor_base_not_model_namespace_legacy_not_promoted(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO")
    f = fixture(tmp_path)
    calls = spy_scan(monkeypatch)
    base = k.Path(f[3]["output"]["cache_dir"])
    base.mkdir()
    old = base / "peak_tf_cache_v2.pkl"
    old.write_bytes(pickle.dumps({"threshold": 1., "data": {"A": ["WRONG_LEGACY"]}}))
    before = old.read_bytes()
    descriptor = k._raw_scan_cache_descriptor(f[3], f[2])
    f[3]["output"]["cache_dir"] = str(tmp_path / "model_namespace")
    result = scan(f, ["A"], descriptor=descriptor)
    assert result["A"] == ["WRONG_LEGACY"] and len(calls) == 0
    assert old.read_bytes() == before
    assert "Reusing legacy peak-TF candidates" in caplog.text
    assert not list((base / "peak_motif_raw_v1").glob("*.pkl"))
    assert not (tmp_path / "model_namespace" / "peak_motif_raw_v1").exists()


def test_legacy_two_requests_zero_scans_no_promotion(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO")
    f = fixture(tmp_path)
    base = k.Path(f[3]["output"]["cache_dir"])
    base.mkdir()
    old = base / "peak_tf_cache_v2.pkl"
    old.write_bytes(pickle.dumps({"threshold": 1., "data": {"A": ["FIRST", "SECOND"], "B": ["BTF"]}}))
    before = old.read_bytes()
    calls = spy_scan(monkeypatch)
    monkeypatch.setattr(k, "_extract_peak_sequences", lambda *args, **kwargs: pytest.fail("legacy-covered extraction"))
    assert scan(f, ["A", "B"]) == {"A": ["FIRST"], "B": ["BTF"]}
    f[3]["pseudotime"] = {"n_bins": 999}
    f[3]["rna_to_atac"]["max_tfs_per_peak"] = 99
    assert scan(f, ["A", "B"]) == {"A": ["FIRST", "SECOND"], "B": ["BTF"]}
    assert calls == [] and old.read_bytes() == before
    assert not cache_path(f).exists()
    assert "reused_legacy=2 scanned=0" in caplog.text
    assert "Not promoting" not in caplog.text


def test_mixed_raw_legacy_new_subset_and_missing_genome(tmp_path, monkeypatch):
    f = fixture(tmp_path)
    calls = spy_scan(monkeypatch)
    scan(f, ["ZERO"])  # authoritative verified negative
    base = k.Path(f[3]["output"]["cache_dir"])
    old = base / "peak_tf_cache_v2.pkl"
    old.write_bytes(pickle.dumps({"threshold": 1., "data": {"A": ["LEGACY"], "ZERO": ["WRONG"]}}))
    calls.clear()
    result = scan(f, ["A", "B", "ZERO", "OUTSIDE"])
    assert result["A"] == ["LEGACY"] and "B" in result and "ZERO" not in result and "OUTSIDE" not in result
    assert len(calls) == 1
    with cache_path(f).open("rb") as handle:
        assert {key[0] for key in pickle.load(handle)["records"]} == {"B", "ZERO"}
    k.Path(f[3]["input"]["genome_fasta"]).unlink()
    result = scan(f, ["A", "B"], frozen=frozenset({"TF"}))
    assert result == {"A": ["LEGACY"], "B": ["TF"]}
    assert len(calls) == 1


@pytest.mark.parametrize("damage", ["threshold", "pickle", "empty", "malformed"])
def test_legacy_bad_or_empty_only_uncovered_scanned(tmp_path, monkeypatch, damage):
    f = fixture(tmp_path)
    base = k.Path(f[3]["output"]["cache_dir"])
    base.mkdir()
    path = base / "peak_tf_cache_v2.pkl"
    payload = {"threshold": .5 if damage == "threshold" else 1.,
               "data": {"A": ["LEGACY"], "B": [] if damage == "empty" else [None]}}
    path.write_bytes(b"broken" if damage == "pickle" else pickle.dumps(payload))
    before = path.read_bytes()
    calls = spy_scan(monkeypatch)
    result = scan(f, ["A", "B"])
    assert len(calls) == (2 if damage in {"threshold", "pickle"} else 1)
    if damage in {"empty", "malformed"}: assert result["A"] == ["LEGACY"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("damage", ["schema", "identity", "record", "pickle"])
def test_malformed_cache_rescanned(tmp_path, monkeypatch, damage, caplog):
    f = fixture(tmp_path)
    calls = spy_scan(monkeypatch)
    scan(f, ["A"])
    path = cache_path(f)
    with path.open("rb") as handle: payload = pickle.load(handle)
    if damage == "schema": payload["schema"] = "unknown"
    elif damage == "identity": payload["identity"] = {}
    elif damage == "record": payload["records"][("A", "chr1", 0, 80)] = [{"bad": 1}]
    path.write_bytes(b"broken" if damage == "pickle" else pickle.dumps(payload))
    scan(f, ["A"])
    assert len(calls) == 2 and "Ignoring invalid shared" in caplog.text


def test_concurrent_disjoint_misses_merge_union(tmp_path, monkeypatch):
    f = fixture(tmp_path)
    barrier = threading.Barrier(2)
    original = k._scan_one_sequence
    def concurrent(sequence, motifs, threshold):
        hits = original(sequence, motifs, threshold)
        barrier.wait(timeout=10)  # both miss/scan before either merge obtains lock
        return hits
    monkeypatch.setattr(k, "_scan_one_sequence", concurrent)
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(scan, f, [peak]) for peak in ("A", "B")]
        assert all(future.result(timeout=15) for future in futures)
    with cache_path(f).open("rb") as handle:
        assert len(pickle.load(handle)["records"]) == 2
    monkeypatch.setattr(k, "_extract_peak_sequences", lambda *args, **kwargs: pytest.fail("union lost coverage"))
    assert set(scan(f, ["A", "B"])) == {"A", "B"}


def test_stream_hash_shared_runner_memo_and_numeric_oracle(tmp_path):
    f = fixture(tmp_path)
    assert run._RESOURCE_DIGEST_MEMO is k._RESOURCE_DIGEST_MEMO
    assert run._resource_file_identity(f[3]["input"]["genome_fasta"]) == k._resource_file_identity(f[3]["input"]["genome_fasta"])
    sequence = "ACGTGCAATG"*8
    hits = k._scan_one_sequence(sequence, f[2], 1.)
    from numpy.lib.stride_tricks import sliding_window_view
    encoded = np.array(["ACGT".index(base) for base in sequence])
    for hit in hits:
        pwm = f[2][hit["motif_id"]]["pwm"]
        scores = pwm[sliding_window_view(encoded, 5), np.arange(5)].sum(axis=1)
        beta = scores.std()*np.sqrt(6)/np.pi
        mu = scores.mean()-np.euler_gamma*beta
        expected = float(np.clip(1-np.exp(-np.exp(-(scores.max()-mu)/beta)), 0, 1))
        assert hit["p_value"] == expected and hit["max_score"] == float(scores.max())


def test_atomic_failure_cleans_temp_and_preserves_previous_cache(tmp_path, monkeypatch):
    f = fixture(tmp_path)
    scan(f, ["A"])
    path = cache_path(f)
    before = path.read_bytes()
    descriptor = k._raw_scan_cache_descriptor(f[3], f[2])
    def failed_replace(*args):
        raise OSError("simulated publish failure")
    monkeypatch.setattr(k.os, "replace", failed_replace)
    with pytest.raises(OSError, match="publish failure"):
        k._merge_raw_motif_cache(path, descriptor, f[2], {("B", "chr1", 80, 160): []})
    assert path.read_bytes() == before
    assert not list(path.parent.glob(path.name + ".*"))


def test_empty_or_out_of_range_extraction_never_cached_as_zero(tmp_path):
    f = fixture(tmp_path)
    f[1].var.loc["A", ["start", "end"]] = [240, 999]
    assert scan(f, ["A"]) == {}
    assert not cache_path(f).exists()
