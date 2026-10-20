"""Temporal export semantics, state filtering, identity and engine wiring."""
import copy

import numpy as np
import pandas as pd
import pytest

from CausalBridge import io, run, saturating_perturbation as engine
from CausalBridge.saturating_pathways import KEY, COLUMNS, build_pathways, empty_pathways
from test_saturating_transfer import fitted


def tables():
    common = dict(target_gene="KO", source_context="c")
    tf = pd.DataFrame([dict(common, bin=b, source_bin=b, tf_gene=name, peak_id="p",
        delta_atac_raw=value, method="frozen_alpha_single_response")
        for b in (0, 1) for name, value in (("TF1", .2), ("TF2", -.1))])
    peak = pd.DataFrame([dict(common, bin=b, gene="G", peak_id="p", delta_l2=.3,
        method="ordered_telescoping_not_causal") for b in (1, 2)])
    endpoint = pd.DataFrame([dict(common, affected_gene="G", delta_rna=-.5, status="ok",
        selected=True, direct_target=False, train_r2=.8, quality_scope="target_full_training")])
    return tf, peak, endpoint


def test_multiple_tfs_shared_peak_not_an_allocation_and_batch_parity():
    tf, peak, endpoint = tables()
    result = build_pathways(tf, peak, endpoint, branch_boundaries=[(0, 3)])
    assert len(result) == 2 and result.event_count.tolist() == [2, 2]
    assert result.n_bins.tolist() == [2, 2]
    assert result.tf_input_contribution_sum.tolist() == pytest.approx([.4, -.2])
    assert result.peak_input_contribution_sum.tolist() == pytest.approx([.6, .6])
    assert result.delta_rna_log2fc.tolist() == [-.5, -.5]
    assert result.shared_peak_contribution.all() and not result.causal_claim.any()
    assert result.delta_r_raw_sum.isna().all() and result.delta_r_total.isna().all()
    assert set(result.pathway_allocation_status) == {"unavailable"}
    assert set(result.pathway_type) == {"temporal_event_link"}
    assert result.bin_min.tolist() == [1, 1] and result.peak_bin_start.tolist() == [0, 0]
    batched = build_pathways([tf.iloc[:1], tf.iloc[1:]], [peak.iloc[:1], peak.iloc[1:]],
        endpoint, branch_boundaries=[(0, 3)])
    pd.testing.assert_frame_equal(result, batched)


@pytest.mark.parametrize("field,value", [("source_context", "other"), ("target_gene", "other"),
    ("peak_id", "other"), ("bin", 4), ("delta_l2", 0), ("delta_l2", np.nan),
    ("delta_l2", np.inf)])
def test_only_same_context_intervention_peak_next_bin_finite_nonzero(field, value):
    tf, peak, endpoint = tables()
    peak[field] = value
    assert build_pathways(tf, peak, endpoint).empty


def test_branch_warm_boundary_and_tf_source_lag():
    tf, peak, endpoint = tables()
    result = build_pathways(tf, peak, endpoint, branch_boundaries=[(0, 2), (2, 3)])
    assert result.event_count.tolist() == [1, 1]  # peak 1→gene 2 crosses the boundary.
    tf["bin"] = 2; tf["source_bin"] = 1
    peak["bin"] = 3
    assert build_pathways(tf, peak, endpoint, branch_boundaries=[(0, 2), (2, 4)]).empty
    tf["source_bin"] = 2
    result = build_pathways(tf, peak, endpoint, branch_boundaries=[(0, 2), (2, 4)])
    assert set(result.tf_lag) == {0} and set(result.branch) == {"1"}


@pytest.mark.parametrize("value", [0., np.nan, np.inf])
def test_zero_and_nonfinite_incoming_contributions_excluded(value):
    tf, peak, endpoint = tables(); tf["delta_atac_raw"] = value
    assert build_pathways(tf, peak, endpoint).empty


@pytest.mark.parametrize("state", ["same", "unavailable", "nonfinite", "changed"])
def test_final_peak_state_excludes_canceled_clipped_to_wt_or_bad(state):
    tf, peak, endpoint = tables()
    wt = np.ones((3, 1)) * .5
    final = wt.copy(); bad = np.zeros_like(wt, dtype=bool)
    if state != "same":
        final[:2] += .1
    if state == "unavailable":
        bad[:2] = True
    if state == "nonfinite":
        final[:2] = np.nan
    result = build_pathways(tf, peak, endpoint, control_atac_states=wt,
        atac_state_history=final, peak_ids=["p"], atac_unavailable_mask=bad)
    assert result.empty == (state != "changed")
    if not result.empty:
        assert set(result.peak_state_check) == {"changed_available"}


def test_duplicate_event_cancellation_and_native_units_clipping():
    tf, peak, endpoint = tables()
    opposite = tf.copy(); opposite.delta_atac_raw *= -1
    assert build_pathways([tf, opposite], peak, endpoint).empty
    tf["atac_state_unit"] = "normalized_atac_log1p"
    events = tf.rename(columns={"tf_gene": "unused"}).copy()
    events["status"] = "ATAC_physical_clip"
    clips = peak.copy(); clips["status"] = "upper_cap"
    result = build_pathways(tf, peak, endpoint, local_events=events, rna_clips=clips)
    assert set(result.tf_input_unit) == {"delta_normalized_atac_log1p"}
    assert set(result.peak_clipping_status) == {"ATAC_physical_clip"}
    assert set(result.rna_clipping_status) == {"upper_cap"}


@pytest.mark.parametrize("status", ["quality_filtered", "unavailable", "zero_control"])
def test_incomplete_endpoints_never_reported(status):
    tf, peak, endpoint = tables(); endpoint["status"] = status
    assert build_pathways(tf, peak, endpoint).empty


@pytest.mark.parametrize("empty", [False, True])
def test_context_collection_and_canonical_csv_even_empty(tmp_path, empty):
    paths = empty_pathways() if empty else build_pathways(*tables())
    contexts = [{"cluster": "c", KEY: paths}, {"cluster": "d", KEY: paths.iloc[:0]}]
    result = run._collect_saturating_tables(contexts)
    pd.testing.assert_frame_equal(result[KEY], result["pathway_edges"])
    assert list(result[KEY].columns) == COLUMNS
    io.save_results(result, {"atac_to_rna": {"model_mode": "saturating_hybrid"},
        "output": {"dir": str(tmp_path), "save_intermediate": False}})
    output = tmp_path
    standard = pd.read_csv(output / "perturbation_pathways.csv")
    assert not (output / "saturating_pathways.csv").exists()
    assert len(standard) == len(paths)
    assert standard.delta_r_raw_sum.isna().all()


def test_legacy_empty_export_behavior_unchanged(tmp_path):
    io.save_results({"pathway_edges": pd.DataFrame()},
        {"output": {"dir": str(tmp_path), "save_intermediate": False}})
    assert not (tmp_path / "perturbation_pathways.csv").exists()


def test_new_key_alone_writes_standard_file(tmp_path):
    io.save_results({KEY: empty_pathways()}, {"atac_to_rna": {"model_mode": "saturating_hybrid"},
        "output": {"dir": str(tmp_path), "save_intermediate": False}})
    assert list(pd.read_csv(tmp_path / "perturbation_pathways.csv").columns) == COLUMNS


def test_output_builder_is_not_a_stage5_or_stage6_source_dependency(monkeypatch):
    from CausalBridge import saturating_pathways, saturating_transfer
    stages = ("step5_atac_to_rna", "step6_rna_to_atac")
    before = [run._stage_numerical_signatures(stage, "saturating_hybrid") for stage in stages]
    registry_before = [saturating_transfer._source_signature(kind) for kind in
        (saturating_transfer.GENE_TYPE, saturating_transfer.TF_TYPE)]
    monkeypatch.setattr(saturating_pathways, "build_pathways", lambda *args, **kwargs: empty_pathways())
    assert before == [run._stage_numerical_signatures(stage, "saturating_hybrid") for stage in stages]
    assert registry_before == [saturating_transfer._source_signature(kind) for kind in
        (saturating_transfer.GENE_TYPE, saturating_transfer.TF_TYPE)]


def test_actual_io_stable_gene_ids_and_display_names(tmp_path):
    from test_gene_identity_pipeline import fixture, prepare
    _, _, _, cfg, identity, metadata = prepare(fixture(tmp_path))
    tf, peak, endpoint = tables()
    tf["target_gene"] = peak["target_gene"] = endpoint["target_gene"] = "E3"
    tf["tf_gene"] = ["E1", "E2", "E1", "E2"]
    peak["gene"] = endpoint["affected_gene"] = "E4"
    paths = build_pathways(tf, peak, endpoint)
    before = paths.copy(deep=True)
    cfg["atac_to_rna"]["model_mode"] = "saturating_hybrid"
    io.save_results(dict(metadata, **{KEY: paths, "pathway_edges": paths}), cfg, gene_identity=identity)
    saved = pd.read_csv(tmp_path / "results" / "perturbation_pathways.csv")
    assert set(saved.tf_name_id) == {"E1", "E2"}
    assert set(saved.tf_name) == {"A_01", "A_02"}
    assert set(saved.affected_gene_id) == {"E4"} and set(saved.target_gene_id) == {"E3"}
    pd.testing.assert_frame_equal(paths, before)


def test_real_engine_multiple_interventions_and_zero_ko(fitted):
    rna, atac, pt, edges, screen, cfg, transfers, _, genes, tfs = fitted
    cfg = copy.deepcopy(cfg); cfg["perturbation"]["joint_knockout"] = False
    output = engine.propagate_saturating(["A", "B"], rna, atac, pt, edges, transfers, screen, cfg,
        transfer_models=genes, tf_model_registry=tfs, cluster_label="c")
    pd.testing.assert_frame_equal(output[KEY], output["pathway_edges"])
    expected = pd.concat([r[KEY] for r in output["saturating_interventions"]], ignore_index=True)
    pd.testing.assert_frame_equal(output[KEY], expected)
    assert not output[KEY].empty
    assert set(output[KEY].source_context) == {"c"} and not output[KEY].causal_claim.any()
    cfg["perturbation"]["ko_strength"] = 0
    zero = engine.propagate_saturating(["A"], rna, atac, pt, edges, transfers, screen, cfg,
        transfer_models=genes, tf_model_registry=tfs)
    assert zero[KEY].empty and list(zero["pathway_edges"].columns) == COLUMNS
