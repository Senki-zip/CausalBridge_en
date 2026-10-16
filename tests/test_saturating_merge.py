import json

import numpy as np
import pandas as pd
import pytest

from CausalBridge import run


def endpoint(context, wt, pert, **kwargs):
    return dict(source_context=context, target_gene="KO", affected_gene="G",
                control_mean_cp10k=wt, perturbed_mean_cp10k=pert,
                status="ok", selected=True, direct_target=False, **kwargs)


def merge(rows, counts, threshold=0):
    return run._aggregate_saturating_cell_means(pd.DataFrame(rows),
        {c: {"n_cells": n} for c, n in counts.items()}, threshold)


def test_unequal_cell_counts_pool_means_before_ratio():
    rows = [endpoint("a", 2., 8.), endpoint("b", 10., 5.)]
    result, diagnostic = merge(rows, {"a": 2, "b": 6})
    row = result.iloc[0]
    assert row.control_mean_cp10k == 8.
    assert row.perturbed_mean_cp10k == 5.75
    assert row.delta_rna_signed == pytest.approx(np.log2(5.75 / 8.))
    assert row.delta_rna_signed != pytest.approx((2 * 2 + 6 * -1) / 8)
    assert row.aggregation_scope == "full"
    assert row.aggregation_cell_fraction == 1.
    assert diagnostic.empty


def test_invalid_subset_uses_identical_wt_and_pert_contexts_and_is_invariant():
    bad = endpoint("b", 1000., 9000.); bad["status"] = "quality_filtered"
    rows = [endpoint("a", 2., 8.), bad]
    result, diagnostic = merge(rows, {"a": 2, "b": 6, "skipped": 3})
    row = result.iloc[0]
    assert row.delta_rna_signed == pytest.approx(2.)
    assert row.control_mean_cp10k == 2.
    assert row.aggregation_used_cells == 2
    assert row.aggregation_total_cells == 11
    assert row.aggregation_cell_fraction == pytest.approx(2 / 11)
    assert row.aggregation_scope == "partial"
    assert json.loads(row.aggregation_context_exclusions) == {
        "b": "quality_filtered", "skipped": "endpoint_unavailable"}
    assert len(diagnostic) == 2
    bad.update(control_mean_cp10k=1e100, perturbed_mean_cp10k=np.nan)
    changed, _ = merge(rows, {"a": 2, "b": 600, "skipped": 300})
    assert changed.iloc[0].delta_rna_signed == row.delta_rna_signed


def test_unselected_baseline_requires_explicit_valid_endpoint():
    baseline = endpoint("b", 10., 10.); baseline["selected"] = False
    result, _ = merge([endpoint("a", 2., 8.), baseline], {"a": 2, "b": 6})
    row = result.iloc[0]
    assert row.delta_rna_signed == pytest.approx(np.log2(9.5 / 8.))
    assert row.aggregation_baseline_assumption == "unchanged_within_quality_pruned_network"
    assert json.loads(row.aggregation_baseline_contexts) == ["b"]
    baseline["status"] = "quality_filtered"
    result, _ = merge([endpoint("a", 2., 8.), baseline], {"a": 2, "b": 6})
    assert result.iloc[0].delta_rna_signed == pytest.approx(2.)
    assert result.iloc[0].aggregation_baseline_assumption == "none"


@pytest.mark.parametrize("wt,pert,status,n", [
    (0., 0., "ok", 3), (0., 2., "ok", 3), (2., 0., "ok", 3),
    (np.nan, 2., "ok", 3), (2., 4., "unavailable", 3), (2., 4., "ok", 0)])
def test_no_valid_positive_conditions_stay_undefined(wt, pert, status, n):
    item = endpoint("a", wt, pert); item["status"] = status
    result, _ = merge([item], {"a": n})
    assert np.isnan(result.iloc[0].delta_rna_signed)
    assert result.iloc[0].status == "no_valid_positive_cell_means"
    assert result.iloc[0].aggregation_used_cells == 0
    assert result.iloc[0].aggregation_scope == "unavailable"


def test_gene_specific_validity_threshold_and_direct_ko_exclusion():
    rows = [endpoint("a", 2., 8.), endpoint("b", 0., 0.)]
    other = endpoint("b", 4., 2.); other["affected_gene"] = "H"
    direct = endpoint("a", 2., 1.); direct["affected_gene"] = "KO"
    rows += [other, direct]
    result, _ = merge(rows, {"a": 2, "b": 6})
    assert set(result.affected_gene) == {"G", "H"}
    assert result.set_index("affected_gene").loc["G", "aggregation_used_cells"] == 2
    assert result.set_index("affected_gene").loc["H", "aggregation_used_cells"] == 6
    result, _ = merge(rows, {"a": 2, "b": 6}, threshold=1.5)
    assert list(result.affected_gene) == ["G"]


def test_policy_dispatch_default_compatibility_and_training_signatures():
    from CausalBridge.saturating_hybrid import HybridConfig
    cfg = {"atac_to_rna": {"model_mode": "saturating_hybrid",
            "saturating_hybrid": HybridConfig().to_dict()}, "perturbation": {}}
    signatures = {s: run._stage_numerical_signatures(s, "saturating_hybrid")
                  for s in ("step5_atac_to_rna", "step6_rna_to_atac")}
    fit = run._step5_model_config(cfg)
    cfg["perturbation"]["aggregation_policy"] = "valid_cell_means"
    assert run._step5_model_config(cfg) == fit
    assert signatures == {s: run._stage_numerical_signatures(s, "saturating_hybrid") for s in signatures}
    context = {"cluster": "a", "saturating_endpoints": pd.DataFrame([endpoint("a", 2., 8.)])}
    registry = {"a": {"n_cells": 2, "original_l0_mean": np.array([2.])}}
    default, _ = run._saturating_merge_contexts([context], registry, ["G"])
    explicit, _ = run._saturating_merge_contexts([context], registry, ["G"], policy="strict_complete")
    pd.testing.assert_frame_equal(default, explicit)
    result, _ = run._saturating_merge_contexts([context], registry, ["G"], policy="valid_cell_means")
    assert result.iloc[0].delta_rna_signed == pytest.approx(2.)
    with pytest.raises(ValueError, match="unknown"):
        run._saturating_merge_contexts([], {}, [], policy="typo")


def test_duplicate_endpoint_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        merge([endpoint("a", 2., 8.)] * 2, {"a": 2})


def test_existing_aggregation_and_strict_default_numbers_are_preserved(monkeypatch):
    registry = {
        "a": {"n_cells": 2, "original_l0_mean": np.array([2.])},
        "skipped": {"n_cells": 6, "original_l0_mean": np.array([10.])}}
    # Existing joint/common helper retains its original missing-context baseline.
    old = run._aggregate_context_l0(registry, {"KO": {"a": np.array([8.])}}, ["G"])
    assert old.iloc[0].delta_rna_signed == pytest.approx(np.log2(76 / 64))
    context = {"cluster": "a", "projection_inputs": [{"target_genes": ["KO"],
        "cell_ids": ["a0", "a1"]}],
        "saturating_endpoints": pd.DataFrame([endpoint("a", 2., 8.)])}
    monkeypatch.setattr(run, "_cell_l0_from_saturating_payload",
        lambda p: (["a0", "a1"], ["G"], np.array([[8.], [8.]])))
    strict, _ = run._saturating_merge_contexts([context], registry, ["G"])
    assert np.isnan(strict.iloc[0].delta_rna_signed)
    assert strict.iloc[0].status == "incomplete_context_coverage"
    complete, _ = run._saturating_merge_contexts([context], {"a": registry["a"]}, ["G"])
    assert complete.iloc[0].delta_rna_signed == pytest.approx(2.)
    partial, _ = run._saturating_merge_contexts([context], registry, ["G"], policy="valid_cell_means")
    assert partial.iloc[0].delta_rna_signed == pytest.approx(2.)


@pytest.mark.parametrize("policy", ["valid_cell_means", "strict_complete"])
def test_merge_collects_only_endpoints_and_reuses_supplied_table(monkeypatch, policy):
    class UnusedReport(pd.DataFrame):
        def copy(self, *args, **kwargs):
            raise AssertionError("merge copied an unrelated report")

    contexts = [{"cluster": "a", "projection_inputs": [{"target_genes": ["KO"],
        "cell_ids": ["a0", "a1"]}],
        "saturating_endpoints": pd.DataFrame([endpoint("stale_label", 2., 8.)]),
        "saturating_response_curves": UnusedReport({"large_report": [1]})}]
    registry = {"a": {"n_cells": 2, "original_l0_mean": np.array([2.])}}
    monkeypatch.setattr(run, "_cell_l0_from_saturating_payload",
        lambda p: (["a0", "a1"], ["G"], np.array([[8.], [8.]])))
    endpoints = run._collect_saturating_table(contexts, "saturating_endpoints")
    assert list(endpoints.source_context) == ["a"]
    assert list(contexts[0]["saturating_endpoints"].source_context) == ["stale_label"]
    original = endpoints.copy(deep=True)
    expected = run._saturating_merge_contexts(contexts, registry, ["G"], policy=policy)

    def unexpected_collection(*args, **kwargs):
        raise AssertionError("already collected endpoint table was collected again")

    monkeypatch.setattr(run, "_collect_saturating_table", unexpected_collection)
    actual = run._saturating_merge_contexts(contexts, registry, ["G"],
        policy=policy, endpoints=endpoints)
    for left, right in zip(expected, actual):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    pd.testing.assert_frame_equal(endpoints, original, check_exact=True)
    assert actual[0].iloc[0].delta_rna_signed == pytest.approx(2.)


def test_supplied_empty_endpoint_table_is_not_recollected(monkeypatch):
    def unexpected_collection(*args, **kwargs):
        raise AssertionError("empty supplied table must remain authoritative")

    monkeypatch.setattr(run, "_collect_saturating_table", unexpected_collection)
    result, diagnostics = run._saturating_merge_contexts([], {}, [],
        policy="valid_cell_means", endpoints=pd.DataFrame())
    assert result.empty and diagnostics.empty
