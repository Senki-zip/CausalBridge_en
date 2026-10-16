"""Schema-3 scientific dependency boundaries and checkpoint preservation."""
from copy import deepcopy
import ast
import hashlib
import inspect
import pickle
import textwrap

import pytest

from CausalBridge import run, kinetics, granger, adaptive_binning, binning_adapter
from test_pipeline_bin_plan import inputs, install_stubs


STAGES = ("step4_granger", "step5_atac_to_rna", "step6_rna_to_atac")


def saved_contracts(cfg):
    return {stage: run._load_checkpoint(cfg, stage)["adaptive_checkpoint"] for stage in STAGES}


@pytest.mark.parametrize("change,affected", [
    ("workers", ()), ("perturbation", ()), ("output", ()), ("raw_values", ()),
    ("granger", STAGES), ("nn", STAGES[1:]),
    ("data", STAGES), ("coords", STAGES), ("plan", STAGES), ("state_modality", STAGES),
    ("bandwidth_selection", STAGES), ("boundary", STAGES),
    ("resource", STAGES[2:]), ("targets", STAGES[2:]), ("gene_types", STAGES[2:]),
    ("raw_gene_axis", STAGES[2:]), ("family_source", STAGES[2:]),
    ("scanner_source", STAGES[2:]), ("nn_source", STAGES[1:]),
    ("granger_source", STAGES), ("legacy_sse_source", STAGES),
    ("aggregation_source", STAGES), ("policy_source", STAGES),
    ("bin_expression_actual", STAGES),
])
def test_stage_dependency_matrix_and_inner_nn_contract(tmp_path, monkeypatch, change, affected):
    rna, atac, cfg = inputs(tmp_path)
    seen = install_stubs(monkeypatch)
    cfg["rna_to_atac"]["tf_expression_filter"] = True
    genome = tmp_path / "genome.fa"
    genome.write_text(">tiny\nACGTACGT\n")
    cfg["input"] = {"genome_fasta": str(genome)}
    raw = rna.copy()
    targets, types = ["KO"], {"KO": "TF"}
    run._run_modeling_for_subset(rna, atac, targets, types, cfg, rna_raw=raw)
    before = deepcopy(saved_contracts(cfg))
    checkpoint_root = run._checkpoint_dir(cfg)
    original_files = {stage: (checkpoint_root / (stage + ".pkl")).read_bytes() for stage in STAGES}
    seen.clear()

    if change == "workers":
        cfg["rna_to_atac"]["motif_scan_workers"] = 3
    elif change == "perturbation":
        cfg["perturbation"].update(ko_strength=.7, subnetwork_depth=4)
    elif change == "output":
        cfg["output"]["checkpoint_dir"] = str(checkpoint_root)
        cfg["output"]["dir"] = str(tmp_path / "other_results")
        cfg["output"]["cache_dir"] = str(tmp_path / "other_cache")
    elif change == "raw_values":
        raw.X.data[0] += .25  # Step 6 consumes raw gene names, not raw counts.
    elif change == "granger":
        cfg["granger"]["max_lag"] = 2
    elif change == "nn":
        cfg["atac_to_rna"]["saturating_hybrid"] = {"ridge": .123}
    elif change == "root":
        cfg["atac_to_rna"]["root_cell_quantile"] = .2
    elif change == "data":
        rna.X.data[0] += .25
    elif change == "coords":
        atac.var.loc["p", "end"] += 1
    elif change == "plan":
        cfg["pseudotime"]["n_bins"] = 11
    elif change == "state_modality":
        cfg["pseudotime"]["adaptive_binning"]["state_modality"] = "rna"
    elif change == "bandwidth_selection":
        cfg["pseudotime"]["adaptive_binning"]["bandwidth_selection"] = "ess"
    elif change == "boundary":
        cfg["pseudotime"]["adaptive_binning"]["boundary"] = "none"
    elif change == "resource":
        genome.write_text(">tiny\nTGCATGCA\n")
    elif change == "targets":
        targets = ["TF"]
    elif change == "gene_types":
        types = {}
    elif change == "raw_gene_axis":
        raw.var_names = ["KO", "TF", "OTHER"]
    elif change == "bin_expression_actual":
        def zero_expression(*args, **kwargs):
            return 0
        monkeypatch.setattr(granger, "_bin_expression", zero_expression)
    elif change == "legacy_sse_source":
        # Reconstruct the pre-fix numerical source signature: all three SSE
        # sites squared NumPy's SSE and p used 1-cdf. Each downstream contract
        # must differ even when the mocked scientific outputs are identical.
        source = textwrap.dedent(inspect.getsource(granger.granger_test))
        for name in ("reduced", "full"):
            source = source.replace(f"float(np.sum(residuals_{name}))",
                                    f"np.sum(residuals_{name} ** 2)")
        source = source.replace("stats.f.sf(f_stat, 1, df_full)",
                                "1 - stats.f.cdf(f_stat, 1, df_full)")
        tree = ast.parse(source)
        tree.body[0].body.pop(0)  # Match the production hash's docstring exclusion.
        legacy_hash = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
        original_hash = run._numerical_source_hash
        assert legacy_hash != original_hash(granger.granger_test)
        monkeypatch.setattr(run, "_numerical_source_hash",
            lambda candidate: legacy_hash if candidate is granger.granger_test else original_hash(candidate))
    else:
        original_hash = run._numerical_source_hash
        function = {"family_source": kinetics._build_family_groups,
                    "scanner_source": kinetics._scan_one_sequence,
                    "nn_source": kinetics.fit_atac_to_rna,
                    "granger_source": granger.granger_test,
                    "aggregation_source": binning_adapter.aggregate_with_plan,
                    "policy_source": adaptive_binning.fit_policy}[change]
        monkeypatch.setattr(run, "_numerical_source_hash",
            lambda candidate: "changed-numerical-source" if candidate is function else original_hash(candidate))

    if affected:
        with pytest.raises(RuntimeError, match=affected[0] + ".*使用新的checkpoint_dir"):
            run._run_modeling_for_subset(rna, atac, targets, types, cfg, rna_raw=raw)
        assert not seen
    else:
        run._run_modeling_for_subset(rna, atac, targets, types, cfg, rna_raw=raw)
        assert [stage for stage, _, _ in seen] == [8]
    assert all((checkpoint_root / (stage + ".pkl")).read_bytes() == content
               for stage, content in original_files.items())

    # A new synthetic checkpoint root exposes each stage's complete contract;
    # even identical mocked upstream outputs must propagate changed signatures.
    cfg["output"].update(dir=str(tmp_path / "fresh"), checkpoint_dir=str(tmp_path / "fresh_checkpoints"))
    run._run_modeling_for_subset(rna, atac, targets, types, cfg, rna_raw=raw)
    after = saved_contracts(cfg)
    assert tuple(stage for stage in STAGES if before[stage] != after[stage]) == tuple(affected)


def test_signatures_are_explicit_functions_and_stage_versions(monkeypatch):
    observed = []
    original = run._numerical_source_hash
    def recording(function):
        assert inspect.isfunction(function) or inspect.isclass(function)
        observed.append(function.__name__)
        return original(function)
    monkeypatch.setattr(run, "_numerical_source_hash", recording)
    four = run._stage_numerical_signatures(STAGES[0])
    assert "granger_test" in observed and "aggregate" in observed and "fit_policy" in observed
    assert "_bin_expression" in observed
    assert "_build_family_groups" not in observed and "fit_atac_to_rna" not in observed
    assert "torch" not in four["versions"] and "biopython" not in four["versions"]
    observed.clear()
    five = run._stage_numerical_signatures(STAGES[1])
    assert "fit_atac_to_rna" in observed and "_compute_root_atac" not in observed
    assert "_build_family_groups" not in observed and "_scan_one_sequence" not in observed
    assert "torch" not in five["versions"] and "biopython" not in five["versions"]
    observed.clear()
    run._stage_numerical_signatures(STAGES[2])
    assert "_build_family_groups" in observed and "_scan_one_sequence" in observed
    assert "_train_nn" not in observed


def test_schema2_broad_hash_is_rejected_and_preserved(tmp_path, monkeypatch):
    rna, atac, cfg = inputs(tmp_path)
    seen = install_stubs(monkeypatch)
    run._save_checkpoint(cfg, STAGES[0], {"granger_results": "old",
        "adaptive_checkpoint": {"schema_version": "adaptive-pipeline-checkpoint-2",
                                "implementation_fingerprint": "a30"}})
    path = run._checkpoint_dir(cfg) / (STAGES[0] + ".pkl")
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="schema mismatch.*cannot be safely auto-migrated.*使用新的checkpoint_dir"):
        run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg)
    assert path.read_bytes() == original
    assert not seen
    assert not (path.parent / (STAGES[1] + ".pkl")).exists()


def test_field_diff_is_nested_bounded_and_missing_fields_explicit(tmp_path, monkeypatch):
    rna, atac, cfg = inputs(tmp_path)
    install_stubs(monkeypatch)
    run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg)
    path = run._checkpoint_dir(cfg) / (STAGES[0] + ".pkl")
    checkpoint = pickle.loads(path.read_bytes())
    checkpoint["adaptive_checkpoint"]["contract"]["granger"]["reference_bins"] = 99
    with path.open("wb") as handle:
        pickle.dump(checkpoint, handle)
    with pytest.raises(RuntimeError, match=r"contract.granger.reference_bins: saved=99 current=10"):
        run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg)
    differences = run._checkpoint_field_differences({str(i): "a" * 64 for i in range(20)},
                                                  {str(i): "b" * 64 for i in range(20)})
    assert len(differences) == 10 and len("; ".join(differences)) < 2000
    assert run._checkpoint_field_differences({}, {"contract": {}}) == ["contract: missing saved field"]


def test_changed_upstream_output_rejects_downstream_even_with_same_contract(tmp_path, monkeypatch):
    rna, atac, cfg = inputs(tmp_path)
    seen = install_stubs(monkeypatch)
    run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg)
    path = run._checkpoint_dir(cfg) / (STAGES[0] + ".pkl")
    checkpoint = pickle.loads(path.read_bytes())
    checkpoint["granger_results"]["scientific_output"] = .123
    with path.open("wb") as handle:
        pickle.dump(checkpoint, handle)
    seen.clear()
    with pytest.raises(RuntimeError, match="step5_atac_to_rna.*granger_output_fingerprint"):
        run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg)
    assert not seen


def test_checkpoint_audit_metadata_is_not_a_validation_domain(tmp_path, monkeypatch):
    rna, atac, cfg = inputs(tmp_path)
    seen = install_stubs(monkeypatch)
    run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg)
    for stage in STAGES:
        path = run._checkpoint_dir(cfg) / (stage + ".pkl")
        checkpoint = pickle.loads(path.read_bytes())
        checkpoint["adaptive_checkpoint"]["audit"] = {"full_config_hash": "irrelevant"}
        with path.open("wb") as handle:
            pickle.dump(checkpoint, handle)
    seen.clear()
    run._run_modeling_for_subset(rna, atac, ["KO"], {}, cfg)
    assert [stage for stage, _, _ in seen] == [8]


def test_state_change_preserves_raw_and_family_evidence_cache(tmp_path, monkeypatch):
    from test_shared_motif_cache import fixture, scan, spy_scan, cache_path
    from test_family_motif_cache import fixture as family_fixture, run_family
    raw = fixture(tmp_path)
    raw[3]["pseudotime"] = {"adaptive_binning": {"enabled": True, "state_modality": "joint"}}
    calls = spy_scan(monkeypatch)
    first = scan(raw, ["A", "B"])
    path = cache_path(raw)
    before = path.read_bytes()
    raw[3]["pseudotime"]["adaptive_binning"]["state_modality"] = "rna"
    assert scan(raw, ["A", "B"]) == first
    assert len(calls) == 2 and cache_path(raw) == path and path.read_bytes() == before

    atac, motifs, config = family_fixture(tmp_path)
    config["pseudotime"] = {"adaptive_binning": {"enabled": True, "state_modality": "joint"}}
    sequences = []
    first, candidates = run_family(atac, motifs, config, monkeypatch, sequences)
    family_files = {p: p.read_bytes() for p in (tmp_path / "base" / "peak_family_motif_v2").glob("*.pkl")}
    assert family_files and sequences == [("peak",)]
    config["pseudotime"]["adaptive_binning"]["state_modality"] = "rna"
    sequences.clear()
    second, second_candidates = run_family(atac, motifs, config, monkeypatch, sequences)
    assert second == first and second_candidates == candidates and not sequences
    assert {p: p.read_bytes() for p in (tmp_path / "base" / "peak_family_motif_v2").glob("*.pkl")} == family_files
