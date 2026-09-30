import pandas as pd
import numpy as np
from typing import Any

from atac_bridge.perturbation import (
    _active_proxy_seeds,
    _extract_subnetwork,
    _extract_unlimited_subnetwork,
    _validate_subnetwork_depth,
)


RNA = ["KO", "PROXY", "B", "C", "UP", "SIDE"]
PEAKS = ["p0", "p1", "p2", "pup", "pside"]


def _graphs():
    tf = pd.DataFrame([
        {"tf_gene": "KO", "peak_id": "p0", "tf_lag": 1},
        {"tf_gene": "PROXY", "peak_id": "p0", "tf_lag": 1},
        {"tf_gene": "B", "peak_id": "p1", "tf_lag": 1},
        {"tf_gene": "C", "peak_id": "p2", "tf_lag": 1},
        {"tf_gene": "UP", "peak_id": "pup", "tf_lag": 1},
        {"tf_gene": "SIDE", "peak_id": "pside", "tf_lag": 1},
        {"tf_gene": "B", "peak_id": "p0", "tf_lag": 9},
    ])
    pg = pd.DataFrame([
        {"peak_id": "p0", "gene": "B", "r2_score": .9},
        {"peak_id": "p1", "gene": "C", "r2_score": .9},
        {"peak_id": "p2", "gene": "KO", "r2_score": .9},  # cycle
        {"peak_id": "pup", "gene": "KO", "r2_score": .9},
        {"peak_id": "pside", "gene": "SIDE", "r2_score": .9},
        {"peak_id": "p0", "gene": "UP", "r2_score": .1},  # low R2
    ])
    return tf, pg


def test_finite_depth_uses_existing_mixed_bfs():
    tf, pg = _graphs()
    granger = pg.copy()
    finite = _extract_subnetwork("KO", granger, pg, tf, depth=1)
    assert "p0" in finite["peaks"]
    assert "B" in finite["genes"]


def test_unlimited_is_downstream_and_follows_proxy_seed_and_cycle():
    tf, pg = _graphs()
    result = _extract_unlimited_subnetwork(
        {"KO", "PROXY"}, pg, tf, RNA, PEAKS, max_edges=100
    )
    assert {"KO", "PROXY", "B", "C"} <= result["genes"]
    assert {"p0", "p1", "p2"} <= result["peaks"]
    assert "UP" not in result["genes"]  # low-R2 / reverse direction excluded
    assert "pup" not in result["peaks"]
    assert "SIDE" not in result["genes"]
    assert result["max_hop_depth"] == 2


def test_unlimited_honors_min_lag_ties_and_endpoint_names():
    tf, pg = _graphs()
    tf = pd.concat([tf, pd.DataFrame([{
        "tf_gene": "KO", "peak_id": "p0", "tf_lag": 1,
    }, {"tf_gene": "MISSING", "peak_id": "missing", "tf_lag": 1}])], ignore_index=True)
    result = _extract_unlimited_subnetwork(
        {"KO"}, pg, tf, RNA, PEAKS, min_lag_per_peak_filter=True
    )
    assert result["n_edges"] == 6
    assert "missing" not in result["peaks"]


def test_unlimited_min_lag_does_not_drop_seed_edge_for_unreachable_tf():
    tf = pd.DataFrame([
        {"tf_gene": "KO", "peak_id": "p", "tf_lag": 2},
        {"tf_gene": "UP", "peak_id": "p", "tf_lag": 1},
        {"tf_gene": "DOWN", "peak_id": "q", "tf_lag": 1},
    ])
    pg = pd.DataFrame([
        {"peak_id": "p", "gene": "DOWN", "r2_score": .9},
        {"peak_id": "q", "gene": "gene", "r2_score": .9},
    ])
    result = _extract_unlimited_subnetwork(
        {"KO"}, pg, tf, ["KO", "UP", "DOWN", "gene"], ["p", "q"],
        min_lag_per_peak_filter=True,
    )
    assert {"p", "q"} <= result["peaks"]
    assert {"DOWN", "gene"} <= result["genes"]


def test_unlimited_closure_keeps_lagged_route_before_minlag_filter():
    """A later short-lag edge must not erase the edge used by the closure."""
    tf = pd.DataFrame([
        {"tf_gene": "KO", "peak_id": "P", "tf_lag": 2},
        {"tf_gene": "TF_B", "peak_id": "P", "tf_lag": 1},
    ])
    pg = pd.DataFrame([{"peak_id": "P", "gene": "TF_B", "r2_score": .9}])
    result = _extract_unlimited_subnetwork(
        {"KO"}, pg, tf, ["KO", "TF_B"], ["P"],
        min_lag_per_peak_filter=True,
    )
    assert {"KO", "TF_B"} <= result["genes"]
    assert "P" in result["peaks"]
    assert result["n_work_edges"] == 3  # both TF rows and the one p→gene row


def test_unlimited_long_chain_processes_each_adjacency_once():
    """Work accounting exposes one pass over each indexed adjacency row."""
    length = 120
    tf = pd.DataFrame([
        {"tf_gene": f"TF{i}", "peak_id": f"P{i}", "tf_lag": 1}
        for i in range(length)
    ])
    pg = pd.DataFrame([
        {"peak_id": f"P{i}", "gene": f"TF{i + 1}", "r2_score": .9}
        for i in range(length - 1)
    ] + [{"peak_id": f"P{length - 1}", "gene": "END", "r2_score": .9}])
    names = ["END"] + [f"TF{i}" for i in range(length)]
    result = _extract_unlimited_subnetwork(
        {"TF0"}, pg, tf, names, [f"P{i}" for i in range(length)],
        max_edges=length * 3,
    )
    assert len(result["peaks"]) == length
    assert "END" in result["genes"]
    assert result["n_work_edges"] == 2 * length


def test_unlimited_validates_value_and_budget():
    assert _validate_subnetwork_depth(2) == 2
    assert _validate_subnetwork_depth("unlimited") == "unlimited"
    for value in (True, False, 0, -1, "all", 1.5):
        try:
            _validate_subnetwork_depth(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid depth {value!r}")
    tf, pg = _graphs()
    try:
        _extract_unlimited_subnetwork({"KO"}, pg, tf, RNA, PEAKS, max_edges=1)
    except ValueError as exc:
        assert "subnetwork_unlimited_max_edges" in str(exc)
    else:
        raise AssertionError("edge budget was not enforced")


def test_unlimited_budget_counts_duplicate_simulation_rows():
    tf = pd.DataFrame([{"tf_gene": "KO", "peak_id": "p0", "tf_lag": 1}])
    pg = pd.DataFrame([
        {"peak_id": "p0", "gene": "B", "r2_score": .9},
        {"peak_id": "p0", "gene": "B", "r2_score": .9},
        # This row is outside the extracted peak closure and must not count.
        {"peak_id": "pup", "gene": "B", "r2_score": .9},
    ])
    try:
        _extract_unlimited_subnetwork({"KO"}, pg, tf, RNA, PEAKS, max_edges=2)
    except ValueError as exc:
        assert "subnetwork_unlimited_max_edges" in str(exc)
    else:
        raise AssertionError("duplicate peak→gene rows bypassed the edge budget")


def test_unlimited_zero_expression_proxy_is_not_a_seed():
    """A proxy with no activity must not expand the executable closure."""
    tf = pd.DataFrame([
        {"tf_gene": "KO", "peak_id": "p0", "tf_lag": 1},
        {"tf_gene": "PROXY", "peak_id": "pup", "tf_lag": 1},
    ])
    pg = pd.DataFrame([
        {"peak_id": "p0", "gene": "B", "r2_score": .9},
        {"peak_id": "pup", "gene": "UP", "r2_score": .9},
    ])
    result = _extract_unlimited_subnetwork({"KO"}, pg, tf, RNA, PEAKS)
    assert "PROXY" not in result["genes"]
    assert "pup" not in result["peaks"]


def test_unlimited_final_edge_cap_checked_after_closure_selection():
    """Final edge cap is checked after closure selection; work budget stops early."""
    tf = pd.DataFrame([
        {"tf_gene": "KO", "peak_id": "p0", "tf_lag": 1},
        {"tf_gene": "KO", "peak_id": "p1", "tf_lag": 1},
    ])
    pg = pd.DataFrame([
        {"peak_id": "p0", "gene": "B", "r2_score": .9},
        {"peak_id": "p1", "gene": "C", "r2_score": .9},
    ])
    try:
        _extract_unlimited_subnetwork({"KO"}, pg, tf, RNA, PEAKS, max_edges=1)
    except ValueError as exc:
        assert "subnetwork_unlimited_max_edges" in str(exc)
    else:
        raise AssertionError("final edge cap was not checked after closure selection")


def test_unlimited_rejects_invalid_edge_budget_config():
    tf, pg = _graphs()
    invalid_values: tuple[Any, ...] = (True, False, 0, -1, 1.5, "100")
    for value in invalid_values:
        try:
            _extract_unlimited_subnetwork(
                {"KO"}, pg, tf, RNA, PEAKS, max_edges=value
            )
        except ValueError as exc:
            assert "subnetwork_unlimited_max_edges" in str(exc)
        else:
            raise AssertionError(f"accepted invalid max_edges {value!r}")

    for value in invalid_values:
        try:
            _extract_unlimited_subnetwork(
                {"KO"}, pg, tf, RNA, PEAKS, max_work_edges=value
            )
        except ValueError as exc:
            assert "subnetwork_unlimited_max_work_edges" in str(exc)
        else:
            raise AssertionError(f"accepted invalid max_work_edges {value!r}")


def test_unlimited_budget_counts_only_retained_min_lag_tf_edges():
    tf = pd.DataFrame([
        {"tf_gene": "KO", "peak_id": "p", "tf_lag": 2},
        {"tf_gene": "UP", "peak_id": "p", "tf_lag": 1},
    ])
    pg = pd.DataFrame([{"peak_id": "p", "gene": "B", "r2_score": .9}])
    result = _extract_unlimited_subnetwork(
        {"KO", "UP"}, pg, tf, ["KO", "UP", "B"], ["p"],
        min_lag_per_peak_filter=True, max_edges=2,
    )
    assert result["n_edges"] == 2  # retained UP→p plus p→B


def test_unlimited_final_budget_counts_tf_row_without_lag_policy():
    """A one-edge TF→peak plus one peak→gene pair is two final rows."""
    tf = pd.DataFrame([{"tf_gene": "KO", "peak_id": "p", "tf_lag": 1}])
    pg = pd.DataFrame([{"peak_id": "p", "gene": "B", "r2_score": .9}])
    try:
        _extract_unlimited_subnetwork(
            {"KO"}, pg, tf, ["KO", "B"], ["p"],
            min_lag_per_peak_filter=False, max_edges=1,
        )
    except ValueError as exc:
        assert "subnetwork_unlimited_max_edges" in str(exc)
    else:
        raise AssertionError("final budget omitted the retained TF→peak row")


def test_unlimited_work_budget_fails_before_more_candidate_scans():
    """The separate conservative work cap stops expansion before finalization."""
    tf = pd.DataFrame([
        {"tf_gene": "KO", "peak_id": "p0", "tf_lag": 1},
        {"tf_gene": "KO", "peak_id": "p1", "tf_lag": 1},
    ])
    pg = pd.DataFrame([
        {"peak_id": "p0", "gene": "B", "r2_score": .9},
        {"peak_id": "p1", "gene": "C", "r2_score": .9},
    ])
    try:
        _extract_unlimited_subnetwork(
            {"KO"}, pg, tf, RNA, PEAKS,
            max_edges=100, max_work_edges=1,
        )
    except ValueError as exc:
        message = str(exc)
        assert "subnetwork_unlimited_max_work_edges" in message
        assert "conservative" in message
    else:
        raise AssertionError("work budget did not stop candidate traversal")


def test_inactive_proxy_is_unlimited_only_seed_filter():
    original = {"ACTIVE": .8, "INACTIVE": .7}
    perturbed = np.array([[1.0, 0.0]])
    baseline = np.array([[1.0, 0.0]])
    active = _active_proxy_seeds(
        original, perturbed, baseline, {"ACTIVE": 0, "INACTIVE": 1}
    )
    assert set(original) == {"ACTIVE", "INACTIVE"}  # finite expansion set
    assert active == set()  # unlimited closure seeds


def test_unlimited_keeps_all_window_only_tf_group():
    tf = pd.DataFrame([{
        "tf_gene": "KO", "peak_id": "p0", "tf_lag": 1, "_window_only": True,
    }])
    pg = pd.DataFrame([{"peak_id": "p0", "gene": "B", "r2_score": .9}])
    result = _extract_unlimited_subnetwork({"KO"}, pg, tf, RNA, PEAKS)
    assert result["genes"] >= {"KO", "B"}
    assert result["peaks"] >= {"p0"}


if __name__ == "__main__":
    tests = [v for n, v in globals().items() if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"passed {len(tests)} unlimited-subnetwork tests")
