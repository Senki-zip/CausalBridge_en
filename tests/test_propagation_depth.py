import numpy as np
import pandas as pd

from CausalBridge.perturbation import (
    _accumulate_tf_l1_output_event,
    _active_proxy_seeds,
    _record_gene_cascade_depth,
    _reportable_effect,
    _resolved_ko_seeds,
    _simulate_knockout_bin_forward,
    _single_round_tf_output_means,
)


def test_closure_seeds_only_active_individual_proxies_and_resolved_kos():
    original = np.ones((2, 2))
    perturbed = original.copy()
    perturbed[0, 0] = 0.5
    assert _active_proxy_seeds(
        {"active": 1.0, "inactive": 1.0}, perturbed, original,
        {"active": 0, "inactive": 1},
    ) == {"active"}
    assert _resolved_ko_seeds({1, 99}, ["not_applied", "applied"]) == {"applied"}


def test_finite_simulation_keeps_valid_peak_gene_row_outside_gene_index():
    class _Adata:
        def __init__(self, names, values):
            self.var_names = names
            self.X = np.asarray(values, dtype=float)
            self.n_obs = self.X.shape[0]
            self.var = pd.DataFrame({"chr": ["1"] * len(names),
                                     "start": range(len(names)),
                                     "end": range(1, len(names) + 1)}, index=names)

    rna = _Adata(["KO", "B"], [[1, 1]] * 6)
    atac = _Adata(["p"], [[1]] * 6)
    pseudo = pd.DataFrame({"pseudotime": np.arange(6, dtype=float)})
    tf = pd.DataFrame([{"tf_gene": "KO", "peak_id": "p",
                        "weight": 1.0, "tf_lag": 1}])
    transfer = pd.DataFrame([{"peak_id": "p", "gene": "B",
                              "r2_score": .9, "steady_state_gain": 1.0}])
    # The finite graph index sees p through a granger p→KO row, but does not
    # itself list B.  Production simulation must still retain p→B.
    granger = pd.DataFrame([{"peak_id": "p", "gene": "KO"}])
    result = _simulate_knockout_bin_forward(
        ["KO"], rna, atac, pseudo, granger, transfer, tf,
        {"perturbation": {"subnetwork_depth": 1, "ko_strength": 1.0,
                           "min_r2_threshold": .3, "propagation_rounds": 1},
         "pseudotime": {"n_bins": 3, "min_cells_per_bin": 1},
         "atac_to_rna": {}},
        root_atac=np.array([1.0]), gene_types={"KO": "TF"},
    )
    assert result["gene_cascade_depth"]["B"] == 1


def test_direct_and_multihop_depths_are_graph_hops():
    depths = {"KO": 0}
    sources = {0: 0}

    _record_gene_cascade_depth(depths, sources, 1, "downstream_tf", {0: -2.0}, -1.0)
    _record_gene_cascade_depth(depths, sources, 2, "second_hop", {1: 1.0}, -1.0)

    assert depths == {"KO": 0, "downstream_tf": 1, "second_hop": 2}


def test_joint_ko_uses_minimum_known_source_depth():
    depths = {"KO_A": 0, "KO_B": 0}
    sources = {10: 0, 11: 0}

    _record_gene_cascade_depth(
        depths, sources, 12, "shared_target", {10: 0.2, 11: -0.8}, -1.0
    )

    assert depths["shared_target"] == 1


def test_unknown_attribution_and_high_lag_do_not_invent_depth():
    depths = {"KO": 0}
    sources = {0: 0}

    # The helper receives only graph attribution; a large pseudotime lag is
    # deliberately irrelevant to the resulting depth.
    _record_gene_cascade_depth(depths, sources, 1, "lagged", {0: 1.0}, -1.0)
    _record_gene_cascade_depth(depths, sources, 2, "unknown", {99: 1.0}, -1.0)
    _record_gene_cascade_depth(depths, sources, 3, "nan_source", {0: np.nan}, -1.0)
    _record_gene_cascade_depth(depths, sources, 4, "zero_effect", {0: 1.0}, 0.0)

    assert depths["lagged"] == 1
    assert "unknown" not in depths
    assert "nan_source" not in depths
    assert "zero_effect" not in depths


def test_tf_output_event_decodes_l2_and_weights_remaining_bins():
    output = np.zeros(1)
    baseline = 2.0
    l2_delta = np.log1p(1.5) - np.log1p(baseline)
    expected_l1_delta = np.expm1(np.log1p(baseline) + l2_delta) - baseline

    _accumulate_tf_l1_output_event(output, 0, baseline, l2_delta, 5, 50)

    assert np.isclose(output[0], expected_l1_delta * 45 / 50)


def test_tf_output_events_add_without_mutating_simulation_state():
    output = np.zeros(1)
    original = output.copy()
    _accumulate_tf_l1_output_event(output, 0, 1.0, 0.2, 2, 10)
    first = output[0]
    _accumulate_tf_l1_output_event(output, 0, 1.0, -0.1, 7, 10)

    assert np.array_equal(original, np.zeros(1))
    assert np.isclose(
        output[0],
        first + (np.expm1(np.log1p(1.0) - 0.1) - 1.0) * 3 / 10,
    )


def test_tf_output_keeps_initialization_offset_and_respects_resolved_ko_mask():
    means = _single_round_tf_output_means(
        np.array([10.0, 10.0]), np.array([2.0, 3.0]),
        np.array([1.0, 4.0]), np.array([True, True]),
        np.array([False, True]),
    )
    assert np.array_equal(means, np.array([13.0, 10.0]))


def test_only_non_ko_tf_nonpositive_output_is_nan():
    means = _single_round_tf_output_means(
        np.array([1.0, 1.0, 1.0]), np.array([-2.0, -2.0, -2.0]),
        np.zeros(3), np.array([True, True, False]),
        np.array([False, True, False]),
    )
    assert np.isnan(means[0])
    assert means[1] == 1.0  # resolved KO retains its direct state
    assert means[2] == 1.0  # non-TF path is untouched


def test_invalid_endpoint_is_not_a_reportable_effect():
    # NaN means "cannot evaluate the counterfactual", never a zero effect.
    # abs(nan) < threshold is False, so a bare magnitude check would let it
    # through and silently inflate the result tables.
    assert _reportable_effect(np.nan, 0.0) is False
    assert _reportable_effect(np.nan, 1.0) is False
    assert _reportable_effect(np.inf, 0.0) is False
    assert _reportable_effect(-np.inf, 0.0) is False


def test_reportable_effect_keeps_old_magnitude_semantics():
    assert _reportable_effect(0.5, 0.0) is True
    assert _reportable_effect(-0.5, 0.0) is True
    assert _reportable_effect(0.0, 0.0) is False
    assert _reportable_effect(0.4, 0.5) is False
    assert _reportable_effect(0.6, 0.5) is True


if __name__ == "__main__":
    tests = [value for name, value in globals().items()
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"passed {len(tests)} focused propagation tests")
