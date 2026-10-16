"""Streaming/memo and frozen local resource identities, without network calls."""
import hashlib
from pathlib import Path
import pickle

import pytest
from CausalBridge import run, kinetics


def test_streamed_digest_memo_and_same_size_same_mtime_replacement(tmp_path, monkeypatch):
    path = tmp_path / "genome.fa"
    content = b"ACGT" * 100
    path.write_bytes(content)
    run._RESOURCE_DIGEST_MEMO.clear()
    monkeypatch.setattr(run, "_RESOURCE_CHUNK_BYTES", 17)
    original = Path.open
    reads = []
    class Reader:
        def __init__(self, handle): self.handle = handle
        def __enter__(self): return self
        def __exit__(self, *args): self.handle.close()
        def read(self, size):
            assert size == 17  # never read the whole genome
            reads.append(size)
            return self.handle.read(size)
    def opening(self, *args, **kwargs):
        handle = original(self, *args, **kwargs)
        return Reader(handle) if self == path.resolve() and args == ("rb",) else handle
    monkeypatch.setattr(Path, "open", opening)
    first = run._resource_file_identity(path)
    assert first["sha256"] == hashlib.sha256(content).hexdigest()
    count = len(reads)
    assert run._resource_file_identity(path) == first and len(reads) == count
    old_stat = path.stat()
    replacement = tmp_path / "replacement.fa"
    replacement.write_bytes(b"TGCA" * 100)
    import os
    os.utime(replacement, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    replacement.replace(path)
    assert run._resource_file_identity(path)["sha256"] != first["sha256"]
    assert len(reads) > count


def test_local_first_acquisition_frozen_before_candidate_namespacing(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    motifs = {"local": {"tf_name": "TF"}}
    with (cache / "jaspar_motifs.pkl").open("wb") as handle:
        pickle.dump(motifs, handle)
    monkeypatch.setattr(kinetics, "_load_gimmemotifs_database", lambda *args: None)
    monkeypatch.setattr(kinetics, "_load_jaspar_motifs", lambda *args: pytest.fail("network with materialized local resource"))
    config = {"output": {"cache_dir": str(cache)}, "rna_to_atac": {"motif_db": "jaspar2024"}}
    frozen = kinetics._prepare_adaptive_motif_resources(config)
    assert frozen["motif_db"] == motifs
    identity = run._resource_identity(config, frozen)
    assert identity["configured:jaspar_motifs.pkl"]["availability"] == "available"
    assert identity["online_request"]["remote_freshness_verified"] is False
    assert all("peak_tf_cache" not in key for key in identity)


def test_newly_materialized_cache_included_in_first_provenance(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    config = {"output": {"cache_dir": str(cache)}}
    def acquiring(*args):
        cache.mkdir()
        motifs = {"new": {"tf_name": "TF"}}
        with (cache / "gimmemotifs_cache.pkl").open("wb") as handle:
            pickle.dump(motifs, handle)
        return motifs
    monkeypatch.setattr(kinetics, "_load_motif_database", acquiring)
    frozen = kinetics._prepare_adaptive_motif_resources(config)
    identity = run._resource_identity(config, frozen)
    assert identity["configured:gimmemotifs_cache.pkl"]["availability"] == "available"
    assert identity["materialized_motifs"] == run._provenance_fingerprint(frozen["motif_db"])


@pytest.mark.parametrize("genome_mode", ["missing", "unusable"])
@pytest.mark.parametrize("empty_tf", [False, True])
def test_actual_scanner_fasta_fallback_uses_precaptured_base_tf_set(tmp_path, monkeypatch, genome_mode, empty_tf):
    import anndata as ad
    import pandas as pd
    import numpy as np
    base = tmp_path / "base"
    base.mkdir()
    tf_path = base / "tf_database.txt"
    tf_path.write_text("" if empty_tf else "BASE_TF\n")
    genome = tmp_path / "genome.fa"
    if genome_mode == "unusable":
        # Valid FASTA, but no matching chromosome: actual extraction returns None.
        genome.write_text(">unmatched_chromosome\nAAAAAA\n")
    config = {"output": {"cache_dir": str(base)}, "input": {"genome_fasta": str(genome)},
              "rna_to_atac": {"max_tfs_per_peak": 5}}
    motifs = {"present": {"tf_name": "BASE_TF", "pwm": np.ones((4, 5))}}
    monkeypatch.setattr(kinetics, "_load_motif_database", lambda *args: motifs)
    original_load = kinetics._load_tf_database
    loads = []
    def load(directory=None):
        assert Path(directory) == base, "candidate namespace TF resource read"
        loads.append(Path(directory))
        return original_load(directory)
    monkeypatch.setattr(kinetics, "_load_tf_database", load)
    frozen = kinetics._prepare_adaptive_motif_resources(config)
    assert loads == [base]
    assert frozen["tf_database"] == (frozenset() if empty_tf else frozenset({"BASE_TF"}))
    identity = run._resource_identity(config, frozen)
    assert identity["configured:tf_database.txt"]["sha256"] == hashlib.sha256(tf_path.read_bytes()).hexdigest()
    assert identity["materialized_tf_database"] == run._provenance_fingerprint(sorted(frozen["tf_database"]))
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "tf_database.txt").write_text("CANDIDATE_TF\n")
    config["output"]["cache_dir"] = str(candidate)
    # No resource load after capture is permitted. The empty frozen set must
    # not be mistaken for an absent argument that reloads candidate resources.
    monkeypatch.setattr(kinetics, "_load_tf_database", lambda *args: pytest.fail("TF database reloaded after capture"))
    rna = ad.AnnData(np.ones((4, 2)), var=pd.DataFrame(index=["BASE_TF", "CANDIDATE_TF"]))
    atac = ad.AnnData(np.ones((4, 1)), var=pd.DataFrame(
        {"chr": ["chr1"], "start": [0], "end": [5]}, index=["peak"]))
    edges = pd.DataFrame({"peak_id": ["peak", "peak"], "gene": ["BASE_TF", "CANDIDATE_TF"]})
    assert kinetics._extract_peak_sequences(atac, config) is None
    result = kinetics._scan_peaks_with_jaspar(atac, rna, edges, motifs,
        config["rna_to_atac"], config, frozen["tf_database"])
    assert result == ({} if empty_tf else {"peak": ["BASE_TF"]})
    assert loads == [base]


def test_actual_scanner_legacy_fallback_still_loads_legacy_cache(tmp_path, monkeypatch):
    import anndata as ad
    import pandas as pd
    import numpy as np
    (tmp_path / "tf_database.txt").write_text("LEGACY_TF\n")
    config = {"output": {"cache_dir": str(tmp_path)}}
    rna = ad.AnnData(np.ones((4, 1)), var=pd.DataFrame(index=["LEGACY_TF"]))
    atac = ad.AnnData(np.ones((4, 1)), var=pd.DataFrame(index=["peak"]))
    edges = pd.DataFrame({"peak_id": ["peak"], "gene": ["LEGACY_TF"]})
    motifs = {"present": {"pwm": np.ones((4, 5)), "tf_name": "LEGACY_TF"}}
    atac.var["chr"], atac.var["start"], atac.var["end"] = "chr1", 0, 5
    assert kinetics._scan_peaks_with_jaspar(atac, rna, edges, motifs, {}, config) == {"peak": ["LEGACY_TF"]}
    with pytest.raises(ValueError, match="not None"):
        kinetics._scan_peaks_with_jaspar(atac, rna, edges, {"present": {}}, {}, config, None)
