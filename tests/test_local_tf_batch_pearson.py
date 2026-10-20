"""Batch Pearson must retain the complete scalar routing contract."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import pearsonr

from CausalBridge import kinetics


def fixture(seed=0, n=40, pairs=4):
    rng = np.random.default_rng(seed)
    x, y = rng.normal(size=(2, n, pairs))
    y[:, 0] = x[:, 0] + rng.normal(scale=.05, size=n)
    y[3:, 1] = x[:-3, 1] + rng.normal(scale=.05, size=n-3)
    genes = [f"TF{i}" for i in range(pairs)]
    peaks = [f"P{i}" for i in range(pairs)]
    return [dict(zip(peaks, [[g] for g in genes])),
            SimpleNamespace(var_names=genes), SimpleNamespace(var_names=peaks),
            x, y, 10, .05, None, .01, None, 12]


def assert_exact(left, right):
    assert type(left) is type(right)
    if isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            assert_exact(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_exact(a, b)
    elif isinstance(left, (float, np.floating)):
        assert np.asarray(left).tobytes() == np.asarray(right).tobytes()
    else:
        assert left == right


def compare(args):
    assert_exact(kinetics._local_tf_peak_routes_scalar(*args),
                 kinetics._local_tf_peak_routes(*args))


@pytest.mark.parametrize("seed", range(24))
def test_random_routes_are_bitwise_equal(seed):
    compare(fixture(seed))


@pytest.mark.parametrize("mode", ["nan", "inf", "constant_x", "constant_y",
                                  "tiny", "std_gate", "float32", "branches",
                                  "aliases", "window3", "window50", "lag0"])
def test_edge_cases(mode):
    args = fixture()
    if mode in ("nan", "inf"):
        args[3][7, 0] = getattr(np, mode)
        args[4][12, 1] = getattr(np, mode)
    elif mode == "constant_x":
        args[3][:] = 1
    elif mode == "constant_y":
        args[4][:] = 0
    elif mode == "tiny":
        args[3] *= 1e-7
    elif mode == "std_gate":
        args[3][:, 0] = np.tile([-1e-6, 1e-6], 20)
    elif mode == "float32":
        args[3:5] = [a.astype(np.float32) for a in args[3:5]]
    elif mode == "branches":
        args[7] = [(0, 2), (2, 7), (7, 40)]
    elif mode == "aliases":
        args[0] = {"missing": ["TF0"], "P1": ["tf1", "TF1", "unknown"],
                   "P0": ["TF0", "TF0"]}
    elif mode.startswith("window"):
        args[-1] = int(mode[6:])
    elif mode == "lag0":
        args[5] = 0
    compare(args)


@pytest.mark.parametrize("gate", ["p", "contribution"])
@pytest.mark.parametrize("direction", [-np.inf, None, np.inf])
def test_scalar_boundary_decisions(gate, direction, monkeypatch):
    args = fixture(n=12)
    args[5] = 0
    x, y = args[3][:, 0], args[4][:, 0]
    r, p = pearsonr(x, y)
    weight = float(r*float(np.std(y))/float(np.std(x)))
    boundary = p if gate == "p" else abs(weight)*float(np.std(x))
    threshold = boundary if direction is None else np.nextafter(boundary, direction)
    args[6 if gate == "p" else 8] = threshold
    calls = []
    def counted(x, y, *, axis=0):
        calls.append(np.ndim(x))
        return pearsonr(x, y, axis=axis)
    monkeypatch.setattr(kinetics, "pearsonr", counted)
    compare(args)
    assert 2 in calls and 1 in calls


@pytest.mark.parametrize("index,value", [(5, True), (5, -1), (5, 1.5),
                                         (10, True), (10, 2), (10, 3.5)])
def test_invalid_options_preserve_errors(index, value):
    args = fixture()
    args[index] = value
    with pytest.raises(ValueError) as original:
        kinetics._local_tf_peak_routes_scalar(*args)
    with pytest.raises(ValueError, match=str(original.value)):
        kinetics._local_tf_peak_routes(*args)


def test_batch_reuses_contexts_and_preserves_logical_counts(monkeypatch):
    args = fixture()
    expected = kinetics._local_tf_peak_routes_scalar(*args)
    calls = []
    def counted(x, y, *, axis=0):
        calls.append(np.shape(x))
        return pearsonr(x, y, axis=axis)
    monkeypatch.setattr(kinetics, "pearsonr", counted)
    actual = kinetics._local_tf_peak_routes(*args)
    assert_exact(expected, actual)
    logical = sum(r["n_stat_tests"] for e in actual for r in e["local_routes"])
    assert len(calls) < logical / 10
    assert any(len(shape) == 2 and shape[0] > 1 for shape in calls)


def test_invalid_bin_mask_and_branch_layout(monkeypatch):
    from CausalBridge import binning_adapter
    args = fixture()
    valid = np.ones(40, dtype=bool)
    valid[[0, 3, 18, 19, 20, 39]] = False
    args[9] = SimpleNamespace(n_bins=40, valid=valid)
    monkeypatch.setattr(binning_adapter, "plan_branch_boundaries",
                        lambda plan: [(0, 19), (19, 40)])
    compare(args)
    routes = kinetics._local_tf_peak_routes(*args)[0]["local_routes"]
    assert all(routes[b]["source_bin"] == -1 for b in np.flatnonzero(~valid))


def test_near_constant_input_uses_scalar_statistics(monkeypatch):
    args = fixture()
    args[3] += 1e14
    args[4] += 1e14
    calls = []
    def counted(x, y, *, axis=0):
        calls.append(np.ndim(x))
        return pearsonr(x, y, axis=axis)
    monkeypatch.setattr(kinetics, "pearsonr", counted)
    compare(args)
    assert 2 in calls
    calls.clear()
    kinetics._local_tf_peak_routes(*args)
    assert calls.count(1) > calls.count(2)


@pytest.mark.parametrize("threshold", [np.nextafter(.05, 0), .05, np.nextafter(.05, 1)])
def test_default_p_gate_neighborhood(threshold):
    from scipy.stats import t
    args = fixture(n=12)
    x = np.arange(12.)
    x = (x-x.mean())/x.std()
    z = np.sin(np.arange(12.))
    z -= z.mean()
    z -= np.dot(z, x)/np.dot(x, x)*x
    z /= z.std()
    critical = t.isf(.05/2, 10)
    r = critical/np.sqrt(critical**2+10)
    args[3][:, 0] = x
    args[4][:, 0] = r*x+np.sqrt(1-r*r)*z
    args[5], args[6] = 0, threshold
    compare(args)


@pytest.mark.parametrize("threshold", [np.nextafter(.01, 0), .01, np.nextafter(.01, 1)])
def test_default_contribution_gate_neighborhood(threshold):
    args = fixture(n=12)
    x = np.arange(12.)
    args[3][:, 0] = x
    args[4][:, 0] = x*(.01/float(np.std(x)))
    args[5], args[8] = 0, threshold
    compare(args)
