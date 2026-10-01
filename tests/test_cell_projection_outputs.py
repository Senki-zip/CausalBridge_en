import pandas as pd
import numpy as np

from CausalBridge.io import save_results
from CausalBridge.run import _project_bin_forward_inputs
from CausalBridge.perturbation import (
    _floor_rna_l1,
    _projection_membership_weights,
    _projection_operators,
)


def test_floor_rna_l1_clips_negative_values_and_counts_them():
    values, count = _floor_rna_l1(np.array([-2.0, 0.0, 1.5, -0.25]))
    assert count == 2
    assert np.array_equal(values, [0.0, 0.0, 1.5, 0.0])


def test_floor_rna_l1_leaves_nonnegative_values_unchanged():
    original = np.array([0.0, 0.5, 3.0])
    values, count = _floor_rna_l1(original)
    assert count == 0
    assert np.array_equal(values, original)


def test_projection_outputs_are_opt_in_and_bounded(tmp_path):
    frame = pd.DataFrame({"cell_id": ["c1"], "projected_latent_1": [0.0]})
    config = {"output": {"dir": str(tmp_path), "save_intermediate": False},
              "perturbation": {"cell_projection": {"enabled": False}}}
    save_results({"cell_projection": frame}, config)
    assert not (tmp_path / "perturbation_cell_projection.csv").exists()

    config["perturbation"]["cell_projection"]["enabled"] = True
    save_results({"cell_projection": frame,
                  "projection_summary": pd.DataFrame({"n_cells": [1]}),
                  "projection_branch_probabilities": pd.DataFrame({"branch_probability": [.9]})}, config)
    assert (tmp_path / "perturbation_cell_projection.csv").exists()
    assert (tmp_path / "perturbation_projection_summary.csv").exists()
    assert (tmp_path / "perturbation_projection_branch_probabilities.csv").exists()
    save_results({"projection_history": pd.DataFrame({"bin": [0], "delta_state": [.2]})}, config)
    assert (tmp_path / "perturbation_projection_bin_history.csv").exists()
    save_results({"projection_bin_metadata": pd.DataFrame({"global_bin": [0], "local_bin": [0]})}, config)
    assert (tmp_path / "perturbation_projection_bin_metadata.csv").exists()


def test_opt_in_projection_seam_noop():
    payload = {
        "control_cell_states": [[0., 0.], [1., 1.], [2., 2.]],
        "control_bin_states": [[0., 0.], [2., 2.]],
        "perturbed_bin_states": [[0., 0.], [2., 2.]],
        "cell_bin_weights": [[1., 0.], [0., 1.], [0., 1.]],
        "cell_ids": ["c0", "c1", "c2"], "bins": [0, 1, 1],
        "pseudotime": [0., .5, 1.], "branch": ["a", "a", "a"],
        "features": ("g1", "g2"), "state_unit": "rna_log1p_cp10k",
        "target_genes": ("G",),
    }
    payload["state_history"] = payload["control_bin_states"]
    table, fingerprint, history, _, _ = _project_bin_forward_inputs(payload, {"perturbation": {}}, "ctx", "G")
    assert fingerprint
    assert (table["projection_status"] == "ok").all()
    assert (table["projected_distance"] == 0).all()
    assert history.empty


def test_projection_floor_diagnostics_are_per_cell():
    payload = {
        "control_cell_states": [[0., 0.], [1., 1.]],
        "control_bin_states": [[0., 0.], [0., 0.]],
        "perturbed_bin_states": [[0., 0.], [0., 0.]],
        "direct_bin_multiplier": [[1., 1.], [1., 1.]],
        "native_bin_effect": [[-2., 0.], [0., 0.]],
        "cell_bin_weights": [[1., 0.], [0., 1.]], "cell_ids": ["c0", "c1"],
        "bins": [0, 1], "pseudotime": [0., 1.], "branch": ["a", "a"],
        "features": ("g1", "g2"), "state_unit": "rna_log1p_cp10k",
        "target_genes": ("G",), "state_history": [[-1., 0.], [0., 0.]],
        "effect_schema_version": "bin_local_effect_v1",
        "native_effect_unit": "delta_log1p_rna_log1p_cp10k",
        "log_accumulation": True,
    }
    table, _, _, _, _ = _project_bin_forward_inputs(payload, {"perturbation": {}}, "ctx", "G")
    assert "context_l1_floor_applied_count" in table
    assert "cell_l1_floor_applied_count" in table
    assert table["cell_l1_floor_applied_count"].tolist() == [1, 0]


def test_gaussian_branch_membership_and_downstream_history():
    pt = pd.DataFrame({"pseudotime": [0., .2, .8, 1.], "bin": [0, 0, 1, 1],
                       "branch": ["left", "left", "right", "right"]})
    weights = _projection_membership_weights(pt, 2, "gaussian", 1)
    span = 1.0
    sigma = span / 2
    centers = np.array([.25, .75])
    expected = np.exp(-0.5 * ((centers[:, None] - np.array([0., .2, .8, 1.])) / sigma) ** 2)
    expected /= expected.sum(axis=1, keepdims=True) + 1e-10
    assert np.allclose(weights.sum(axis=1), 1.0)
    # The same raw kernel yields the simulation bin operator after row
    # normalization and the cell membership operator after column transpose.
    raw = np.exp(-0.5 * ((centers[:, None] - np.array([0., .2, .8, 1.])) / sigma) ** 2)
    expected_membership = raw.T / raw.sum(axis=0, keepdims=True).T
    assert np.allclose(weights, expected_membership)
    payload = {
        "control_cell_states": [[0., 0.], [1., 1.]],
        "control_bin_states": [[0., 0.], [1., 1.]],
        "perturbed_bin_states": [[0., 0.], [1., 1.5]],
        "cell_bin_weights": [[1., 0.], [0., 1.]], "cell_ids": ["c0", "c1"],
        "bins": [0, 1], "pseudotime": [0., 1.], "branch": ["a", "a"],
        "features": ("g1", "g2"), "state_unit": "log1p", "target_genes": ("G",),
        "state_history": [[0., 0.], [1., 1.5]],
    }
    _, _, history, metadata, _ = _project_bin_forward_inputs(payload, {"perturbation": {}}, "ctx", "G")
    assert history.loc[0, "gene"] == "g2"
    assert history.loc[0, "delta_state"] == .5
    assert len(metadata) == 2


def test_hard_operator_is_mean_not_sum():
    pt = pd.DataFrame({"pseudotime": [0., .1, .9], "bin": [0, 0, 1]})
    bin_operator, membership = _projection_operators(pt, 2, "hard", 1)
    values = np.array([[1.], [3.], [10.]])
    assert np.allclose(bin_operator @ values, [[2.], [10.]])
    assert np.allclose(membership.sum(axis=1), 1.0)
