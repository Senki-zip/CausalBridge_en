"""Current cell projection output, frozen WT operators, and physical floors."""
import numpy as np
import pandas as pd
import pytest
from CausalBridge.io import save_results
from CausalBridge.run import _project_bin_forward_inputs
from CausalBridge._wt_aggregation import _weights, _full_wt_aggregates


def payload():
    cells = np.array([[0., 1.], [2., 2.], [3., 3.], [4., 4.]])
    operator = np.array([[.5, .5, 0., 0.], [0., 0., .5, .5]])
    base = operator @ cells
    return dict(effect_schema_version="bin_saturating_effect_v1",
        intervention_unit="cp10k_before_log1p_and_aggregation",
        state_unit="rna_log1p_cp10k", native_effect_unit="delta_log1p_rna_log1p_cp10k",
        control_cell_states=cells, direct_cell_states=cells.copy(),
        direct_cell_cp10k_factors=np.ones_like(cells),
        control_bin_states=base, direct_bin_states=base.copy(), perturbed_bin_states=base.copy(),
        native_bin_effect=np.zeros_like(base), bin_cell_operator=operator,
        cell_bin_weights=np.repeat(np.eye(2), 2, axis=0),
        features=("g1", "g2"), cell_ids=["c0", "c1", "c2", "c3"], target_genes=(),
        branch=["a"] * 4, bins=[0, 0, 1, 1], pseudotime=[0., .2, .8, 1.],
        bin_branch_ids=["a", "a"], bin_plan_centers=[.1, .9])


def test_projection_outputs_are_opt_in_and_bounded(tmp_path):
    frame = pd.DataFrame({"cell_id": ["c1"], "projected_latent_1": [0.]})
    cfg = {"output": {"dir": str(tmp_path), "save_intermediate": False},
           "perturbation": {"cell_projection": {"enabled": False}}}
    save_results({"cell_projection": frame}, cfg)
    assert not (tmp_path / "perturbation_cell_projection.csv").exists()
    cfg["perturbation"]["cell_projection"]["enabled"] = True
    save_results({"cell_projection": frame, "projection_summary": pd.DataFrame({"n_cells": [1]})}, cfg)
    assert (tmp_path / "perturbation_cell_projection.csv").exists()
    assert (tmp_path / "perturbation_projection_summary.csv").exists()


def test_current_projection_noop_preserves_cells_and_complete_history():
    p = payload()
    table, states, history, metadata, _ = _project_bin_forward_inputs(p, {"perturbation": {}}, "ctx", "G")
    assert len(table) == 4 and len(history) == 4 and len(metadata) == 2
    np.testing.assert_allclose(states[["g1", "g2"]], p["control_cell_states"])
    assert (table["cell_l1_floor_applied_count"] == 0).all()


def test_current_projection_floor_diagnostics_are_per_cell():
    p = payload()
    p["native_bin_effect"][0, 0] = -.1
    p["perturbed_bin_states"] = np.expm1(np.log1p(p["direct_bin_states"]) + p["native_bin_effect"])
    table, states, _, _, _ = _project_bin_forward_inputs(p, {"perturbation": {}}, "ctx", "G")
    assert table.cell_l1_floor_applied_count.tolist() == [1, 0, 0, 0]
    assert states.g1.iloc[0] == 0.
    assert (states[["g1", "g2"]].to_numpy() >= 0).all()


@pytest.mark.parametrize("mode", ["hard", "gaussian"])
def test_current_shared_kernel_preserves_mean_and_membership_normalization(mode):
    pt = pd.DataFrame({"pseudotime": [0., .1, .9], "bin": [0, 0, 1]})
    kernel, _ = _weights(pt, 2, {"pseudotime": {"smooth_kernel": mode, "min_cells_per_bin": 1}}, None)
    values = np.array([[1.], [3.], [10.]])
    a, r, valid = _full_wt_aggregates(kernel, values, values, None, smooth=mode)
    assert valid.all()
    np.testing.assert_allclose(a, r)
    membership = kernel.toarray().T / np.asarray(kernel.sum(axis=0)).ravel()[:, None]
    np.testing.assert_allclose(membership.sum(axis=1), 1.)
    if mode == "hard":
        np.testing.assert_allclose(a, [[2.], [10.]])


def test_obsolete_projection_payload_is_explicitly_rejected():
    with pytest.raises(ValueError, match="obsolete projection"):
        _project_bin_forward_inputs({"effect_schema_version": "bin_local_effect_v1"}, {}, "ctx", "G")
