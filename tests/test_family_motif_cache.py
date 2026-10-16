"""Stage-B family evidence cache tests with tiny in-memory sequences."""
import pickle
import json
import os
from pathlib import Path
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from CausalBridge import kinetics as k


def fixture(tmp_path):
    atac = ad.AnnData(np.ones((2, 1)), var=pd.DataFrame(
        {"chr": ["chr1"], "start": [0], "end": [80]}, index=["peak"]))
    motifs = {"fam": {"tf_name": "MEMBER", "all_factors": [],
                       "pwm": np.array([[2, -1, 0, 1, 2], [-1, 2, 1, 0, -1],
                                        [0, 1, 2, -1, 0], [1, 0, -1, 2, 1]], dtype=float)}}
    config = {"input": {"genome_fasta": str(tmp_path / "genome.fa")},
              "output": {"cache_dir": str(tmp_path / "base")}}
    (tmp_path / "genome.fa").write_text(">chr1\n" + "ACGTG" * 20 + "\n")
    return atac, motifs, config


def run_family(atac, motifs, config, monkeypatch, sequence_calls):
    monkeypatch.setattr(k, "_build_family_groups", lambda db: ({"FAM": {"MEMBER"}}, {"TARGET": "FAM"}))
    def extract(ids, cfg):
        sequence_calls.append(tuple(ids))
        return {peak: "ACGTG" * 16 for peak in ids}
    monkeypatch.setattr(k, "_extract_peak_sequences_from_causal", extract)
    candidates = {"peak": ["MEMBER"]}
    result = k._supplement_family_proxies(
        candidates, motifs, {"TARGET": "TF"},
        {"motif_pval_threshold": 1.0}, config,
        atac_adata=atac)
    return result, candidates


def test_family_cache_cold_hit_and_no_extraction(tmp_path, monkeypatch):
    atac, motifs, config = fixture(tmp_path)
    calls = []
    first, candidates = run_family(atac, motifs, config, monkeypatch, calls)
    assert calls == [("peak",)]
    assert candidates["peak"] == ["MEMBER", "TARGET"]
    cache_files = list((tmp_path / "base" / "peak_family_motif_v2").glob("*.pkl"))
    assert len(cache_files) == 1
    second_calls = []
    second, candidates2 = run_family(atac, motifs, config, monkeypatch, second_calls)
    assert second_calls == []
    assert candidates2 == {"peak": ["MEMBER", "TARGET"]}
    assert second == first


def test_family_cache_records_negative_and_new_peak_only(tmp_path, monkeypatch):
    atac, motifs, config = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(k, "_build_family_groups", lambda db: ({"FAM": {"MEMBER"}}, {"TARGET": "FAM"}))
    monkeypatch.setattr(k, "_extract_peak_sequences_from_causal",
                        lambda ids, cfg: (calls.append(tuple(ids)) or {p: "NNNNN" * 16 for p in ids}))
    candidates = {"peak": []}
    k._supplement_family_proxies(candidates, motifs, {"TARGET": "TF"},
                                 {"motif_pval_threshold": 1.0}, config, atac_adata=atac)
    assert calls == [("peak",)]
    # Add a new current peak; only that key needs extraction.
    atac2 = ad.AnnData(np.ones((2, 2)), var=pd.DataFrame(
        {"chr": ["chr1", "chr1"], "start": [0, 5], "end": [80, 85]}, index=["peak", "new"]))
    calls.clear()
    candidates2 = {"peak": [], "new": []}
    k._supplement_family_proxies(candidates2, motifs, {"TARGET": "TF"},
                                 {"motif_pval_threshold": 1.0}, config, atac_adata=atac2)
    assert calls == [("new",)]
    assert candidates2["peak"] == [] and candidates2["new"] == []


@pytest.mark.parametrize("change", ["threshold", "motif", "target", "genome"])
def test_family_identity_changes_invalidate(tmp_path, monkeypatch, change):
    atac, motifs, config = fixture(tmp_path)
    calls = []
    run_family(atac, motifs, config, monkeypatch, calls)
    if change == "threshold":
        threshold = .5
    else:
        threshold = 1.0
    if change == "motif":
        motifs = {"fam": {**motifs["fam"], "pwm": motifs["fam"]["pwm"] + .1}}
    if change == "target":
        monkeypatch.setattr(k, "_build_family_groups", lambda db: ({"FAM": {"MEMBER"}}, {"OTHER": "FAM"}))
    if change == "genome":
        (tmp_path / "genome.fa").write_text(">chr1\n" + "TGCA" * 20 + "\n")
    calls.clear()
    monkeypatch.setattr(k, "_extract_peak_sequences_from_causal",
                        lambda ids, cfg: (calls.append(tuple(ids)) or {p: "ACGTG" * 16 for p in ids}))
    candidates = {"peak": []}
    k._supplement_family_proxies(candidates, motifs,
        {"OTHER" if change == "target" else "TARGET": "TF"},
        {"motif_pval_threshold": threshold}, config, atac_adata=atac)
    assert calls == [("peak",)]


def test_family_progress_and_cache_is_separate_from_raw(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO")
    atac, motifs, config = fixture(tmp_path)
    calls = []
    run_family(atac, motifs, config, monkeypatch, calls)
    assert "阶段B (全家族 PWM 扫描)" in caplog.text
    assert list((tmp_path / "base" / "peak_family_motif_v2").glob("*.pkl"))
    assert not (tmp_path / "base" / "peak_motif_raw_v1").exists()


def test_family_membership_and_identity_across_fresh_hash_seeds(tmp_path):
    """Ambiguous C2H2 roots and CIS-BP sets must not change cache identity."""
    pfm = tmp_path / "tiny.pfm"
    pfm.write_text(">tiny\n" + "9\t1\t1\t1\n" * 5)
    mapping = tmp_path / "tiny.mapping"
    mapping.write_text("tiny\tZIC3\ntiny\tZIC1\ntiny\tZIC2\n")
    program = r'''
import hashlib
import json
import sys
import numpy as np
from pathlib import Path
from CausalBridge import kinetics as k

def motif(tf, factors=()):
    return {"tf_name": tf, "all_factors": [{"factor": f} for f in factors],
            "pwm": np.arange(20, dtype=float).reshape(4, 5)}

# Preserve GM source order and ownership even when a cofactor is shared.
db = {"GM_GM.5.0.C2H2ZF.1": motif("ZIC1", ("ZIC1", "SHARED")),
      "GM_GM.5.0.OTHER.1": motif("ZIC4", ("ZIC4", "SHARED")),
      "unassigned": motif("ZIC2"), "new1": motif("NEW1"),
      "new2": motif("NEW2"), "exact": motif("OTHER2")}
groups, owners = k._build_family_groups(db)
assert owners["ZIC2"] == "C2H2ZF"
assert owners["ZIC4"] == "OTHER"
assert owners["SHARED"] == "C2H2ZF"
assert owners["OTHER2"] == "OTHER"
assert groups["NEW"] == {"NEW1", "NEW2"}
config = {"output": {"cache_dir": sys.argv[3]}}
descriptor = k._family_cache_descriptor(config, db, {"ZIC2": owners["ZIC2"]}, groups, 1e-4)

cisbp = {}
k._parse_cisbp_pfm(Path(sys.argv[1]), Path(sys.argv[2]), cisbp)
assert cisbp["CISBP_tiny"]["tf_name"] == "ZIC1"
assert [f["factor"] for f in cisbp["CISBP_tiny"]["all_factors"]] == ["ZIC1", "ZIC2", "ZIC3"]
# Equal-score motif ties still keep the first source motif, including its PWM.
cisbp["later"] = {**cisbp["CISBP_tiny"], "pwm": cisbp["CISBP_tiny"]["pwm"] + 1}
deduped = k._dedup_motifs_by_tf(cisbp)
assert list(deduped) == ["CISBP_tiny", "CISBP_tiny__ZIC2", "CISBP_tiny__ZIC3"]
assert all(np.array_equal(info["pwm"], cisbp["CISBP_tiny"]["pwm"]) for info in deduped.values())
cis_groups, cis_owners = k._build_family_groups(deduped)
cis_descriptor = k._family_cache_descriptor(config, deduped, {"ZIC2": cis_owners["ZIC2"]}, cis_groups, 1e-4)
print(json.dumps({"groups": {name: sorted(members) for name, members in groups.items()},
                  "owners": list(owners.items()), "descriptor": descriptor,
                  "cis_descriptor": cis_descriptor,
                  "deduped": [(mid, info["tf_name"], info["all_factors"],
                               hashlib.sha256(info["pwm"].tobytes()).hexdigest())
                              for mid, info in deduped.items()]}, sort_keys=True))
'''
    results = []
    root = Path(__file__).resolve().parents[1]
    for seed in ("1", "2", "3", "random"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root), env.get("PYTHONPATH")]))
        result = subprocess.run(
            [sys.executable, "-c", program, str(pfm), str(mapping), str(tmp_path)],
            cwd=root, env=env, capture_output=True, text=True, check=True,
        )
        results.append(json.loads(result.stdout))
    assert all(result == results[0] for result in results[1:])


@pytest.mark.parametrize("sequence,expected", [("ACGTG" * 16, ["RFX2", "RFX3"]),
                                              ("N" * 80, [])])
def test_family_evidence_independent_of_primary_and_passive(tmp_path, monkeypatch, sequence, expected):
    atac, motifs, config = fixture(tmp_path)
    motifs = {"fam": {**motifs["fam"], "tf_name": "RFX1"}}
    peak = "chr1:0-80"
    atac.var_names = [peak]
    (tmp_path / "genome.fa").write_text(">chr1\n" + sequence + "\n")
    targets = {"RFX2": "TF", "RFX3": "TF"}
    # Real family grouping and numerical scanning, including a second target
    # passively supplied on the first call but absent from subsequent contexts.
    motifs["self"] = {"tf_name": "RFX2", "pwm": []}
    motifs["other"] = {"tf_name": "RFX3", "pwm": []}
    first = {peak: ["RFX2", "RFX1"]}
    k._supplement_family_proxies(first, motifs, targets,
                                {"motif_pval_threshold": 1.0}, config, atac_adata=atac)
    assert first[peak] == ["RFX2", "RFX1", "RFX3"]
    cache = next((tmp_path / "base" / "peak_family_motif_v2").glob("*.pkl"))
    payload = pickle.loads(cache.read_bytes())
    assert payload["records"][(peak, "chr1", 0, 80)] == expected
    cold = {peak: []}
    cold_config = {**config, "output": {"cache_dir": str(tmp_path / "cold")}}
    k._supplement_family_proxies(cold, motifs, targets,
                                {"motif_pval_threshold": 1.0}, cold_config, atac_adata=atac)
    def forbidden(*args):
        pytest.fail("complete family evidence must not extract on warm reuse")
    monkeypatch.setattr(k, "_extract_peak_sequences_from_causal", forbidden)
    warm = {peak: []}
    k._supplement_family_proxies(warm, motifs, targets,
                                {"motif_pval_threshold": 1.0}, config, atac_adata=atac)
    assert warm == cold == {peak: expected}


@pytest.mark.parametrize("failure", ["missing", "malformed", "partial"])
def test_family_failed_extraction_is_uncovered(tmp_path, caplog, failure):
    caplog.set_level("INFO")
    atac, motifs, config = fixture(tmp_path)
    motifs = {"member": {**motifs["fam"], "tf_name": "RFX1"},
              "target": {"tf_name": "RFX2", "pwm": []}}
    peaks = ["chr1:0-80", "chrMissing:0-80", "bad-coordinates", "chr1:200-280"]
    atac = ad.AnnData(np.ones((2, 4)), var=pd.DataFrame(
        {"chr": ["chr1", "chrMissing", "chr1", "chr1"],
         "start": [0, 0, 0, 200], "end": [80, 80, 80, 280]}, index=peaks))
    if failure == "missing":
        config["input"]["genome_fasta"] = str(tmp_path / "absent.fa")
    elif failure == "malformed":
        # Duplicate FASTA IDs make SeqIO.to_dict fail rather than silently parse.
        (tmp_path / "genome.fa").write_text(">chr1\nACGT\n>chr1\nACGT\n")
    candidates = {p: ["RFX1"] for p in peaks}
    k._supplement_family_proxies(candidates, motifs, {"RFX2": "TF"},
                                {"motif_pval_threshold": 1.0}, config, atac_adata=atac)
    assert all(names == ["RFX1", "RFX2"] for names in candidates.values())
    files = list((tmp_path / "base" / "peak_family_motif_v2").glob("*.pkl"))
    if failure == "partial":
        records = pickle.loads(files[0].read_bytes())["records"]
        assert list(records) == [(peaks[0], "chr1", 0, 80)]
        assert "scanned_peaks=1 missing_failed=3" in caplog.text
    else:
        assert files == []
        assert "scanned_peaks=0 missing_failed=4" in caplog.text
        assert "阶段B 跳过" in caplog.text


def test_family_without_usable_pwm_is_complete_empty_evidence(tmp_path):
    atac, _, config = fixture(tmp_path)
    atac.var_names = ["chr1:0-80"]
    motifs = {"a": {"tf_name": "RFX1", "pwm": []},
              "b": {"tf_name": "RFX2", "pwm": []}}
    candidates = {"chr1:0-80": []}
    k._supplement_family_proxies(candidates, motifs, {"RFX2": "TF"}, {}, config,
                                atac_adata=atac)
    assert candidates == {"chr1:0-80": []}
    cache = next((tmp_path / "base" / "peak_family_motif_v2").glob("*.pkl"))
    assert list(pickle.loads(cache.read_bytes())["records"].values()) == [[]]
