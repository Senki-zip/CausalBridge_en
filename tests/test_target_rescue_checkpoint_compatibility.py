"""Target rescue alone must not invalidate otherwise identical old checkpoints."""
import ast
import hashlib
import inspect
import textwrap
from copy import deepcopy

import pytest

from CausalBridge import preprocess, run


def raw_hash(source):
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


@pytest.mark.parametrize("stage", ["step4_granger", "step5_atac_to_rna", "step6_rna_to_atac"])
def test_rescue_uses_verified_pre_rescue_source_contract(stage):
    signatures = run._stage_numerical_signatures(stage, gene_identity_mode="gtf_gene_id")
    for name, (reviewed, historical) in run._TARGET_RESCUE_SOURCE_COMPATIBILITY.items():
        if name in signatures["functions"]:
            assert signatures["functions"][name] == historical
    for function in (preprocess.preprocess_rna, run._prepare_identity_model_rna):
        key = function.__module__ + "." + function.__qualname__
        assert raw_hash(inspect.getsource(function)) == run._TARGET_RESCUE_SOURCE_COMPATIBILITY[key][0]


@pytest.mark.parametrize("function", [preprocess.preprocess_rna, run._prepare_identity_model_rna])
def test_alias_does_not_hide_future_source_edits(monkeypatch, function):
    original = inspect.getsource
    modified = textwrap.dedent(original(function)) + "\n"
    tree = ast.parse(modified)
    tree.body[0].body.append(ast.parse("raise RuntimeError('changed numerical implementation')").body[0])
    modified = ast.unparse(ast.fix_missing_locations(tree))
    monkeypatch.setattr(inspect, "getsource", lambda obj: modified if obj is function else original(obj))
    assert run._numerical_source_hash(function) == raw_hash(modified)
    key = function.__module__ + "." + function.__qualname__
    assert run._numerical_source_hash(function) != run._TARGET_RESCUE_SOURCE_COMPATIBILITY[key][1]


def test_live_source_line_shift_is_rejected_before_hashing(monkeypatch):
    original = inspect.getsource
    monkeypatch.setattr(inspect, "getsource", lambda obj:
        original(preprocess.preprocess_rna) if obj is preprocess.preprocess_atac else original(obj))
    with pytest.raises(RuntimeError, match="source moved while process was running"):
        run._numerical_source_hash(preprocess.preprocess_atac)


def test_lambda_source_remains_hashable():
    fixture_callback = lambda value: value + 1
    assert run._numerical_source_hash(fixture_callback) == raw_hash(inspect.getsource(fixture_callback))


@pytest.fixture(scope="module")
def shifted_contracts():
    stages = {}
    for stage in ("step4_granger", "step5_atac_to_rna", "step6_rna_to_atac"):
        stages[stage] = dict(data_fingerprint="same matrices and axes",
            numerical=run._stage_numerical_signatures(stage, gene_identity_mode="gtf_gene_id"))
    stages["step5_atac_to_rna"]["step4_contract"] = run._provenance_fingerprint(stages["step4_granger"])
    stages["step5_atac_to_rna"]["granger_output_fingerprint"] = "actual output"
    stages["step6_rna_to_atac"]["step5_contract"] = run._provenance_fingerprint(stages["step5_atac_to_rna"])
    current = dict(stages=stages, adapter_contract={}, atac_n_vars=2)
    # Construct the historical fixture independently, including nested digests.
    old = deepcopy(current)
    for contract in old["stages"].values():
        for key, (_, shifted) in run._SHIFTED_RESCUE_SOURCE_HASHES.items():
            if key in contract["numerical"]["functions"]:
                contract["numerical"]["functions"][key] = shifted
    old["stages"]["step5_atac_to_rna"]["step4_contract"] = run._provenance_fingerprint(old["stages"]["step4_granger"])
    old["stages"]["step6_rna_to_atac"]["step5_contract"] = run._provenance_fingerprint(old["stages"]["step5_atac_to_rna"])
    return current, old


@pytest.mark.parametrize("stage", ["step4_granger", "step5_atac_to_rna", "step6_rna_to_atac"])
@pytest.mark.parametrize("damage", [None, "data", "other_source", "upstream", "chain", "partial_signature"])
def test_exact_shifted_contract_recovers_and_other_changes_reject(shifted_contracts, monkeypatch, stage, damage):
    current, old = deepcopy(shifted_contracts)
    metadata = run._stage_checkpoint_metadata(old, stage, "upstream output")
    if damage == "data":
        metadata["contract"]["data_fingerprint"] = "changed data"
    elif damage == "other_source":
        metadata["contract"]["numerical"]["functions"]["CausalBridge.gene_identity.normalize_gene_id"] = "different implementation"
    elif damage == "upstream":
        metadata["upstream_fingerprint"] = "changed upstream"
    elif damage == "chain":
        if stage == "step4_granger":
            metadata["contract"]["unexpected"] = True
        else:
            metadata["contract"]["step4_contract" if stage == "step5_atac_to_rna" else "step5_contract"] = "wrong chain digest"
    elif damage == "partial_signature":
        key = "CausalBridge.run._validate_identity_required_genes"
        metadata["contract"]["numerical"]["functions"][key] = run._SHIFTED_RESCUE_SOURCE_HASHES[key][0]
    checkpoint = dict(adaptive_checkpoint=metadata, granger_results={}, causal_edges={},
                      transfer_functions={}, transfer_models={}, tf_weights={})
    before = deepcopy(checkpoint)
    checked = []
    monkeypatch.setattr(run, "_validate_step5_binning_bundle", lambda *args: checked.append(True))
    if damage:
        with pytest.raises(RuntimeError, match="incompatible adaptive checkpoint"):
            run._validate_adaptive_checkpoint(checkpoint, current, stage, "upstream output")
    else:
        run._validate_adaptive_checkpoint(checkpoint, current, stage, "upstream output")
        if stage == "step5_atac_to_rna":
            assert checked == [True]
    assert checkpoint == before
