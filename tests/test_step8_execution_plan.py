"""Private Step 8 execution agrees exactly with the general query predictor."""
import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from CausalBridge import saturating_hybrid as core
from CausalBridge import saturating_perturbation as engine
from CausalBridge import saturating_transfer as transfer
from test_saturating_transfer import fitted
from test_local_tf_lag import local_data
from test_saturating_perturbation import grouped_fit


class ReferencePlan:
    def __init__(self, model, old, supported, declared, *, wt_cache=None):
        self.model, self.old, self.supported, self.declared = model, old, supported, declared

    def predict(self, position, new):
        m = self.model
        return core.predict_delta(m, self.old[position:position+1], new,
            changed_inputs=self.declared, branch_ids=m.branch_ids[position:position+1],
            input_support=self.supported[position:position+1],
            target_rows=m.target_rows[position:position+1] if any(k[1] == "local" for k in m.input_keys) else None)


def assert_same(left, right):
    if isinstance(left, pd.DataFrame):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    elif isinstance(left, pd.Series):
        pd.testing.assert_series_equal(left, right, check_exact=True)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    elif isinstance(left, (float, np.floating)) and np.isnan(left):
        assert np.isnan(right)
    else:
        assert left == right


@pytest.mark.parametrize("strength", [0., .35, 1.])
@pytest.mark.parametrize("mode", ["joint", "independent", "peak"])
def test_complete_intervention_outputs_match_general_predictor(fitted, monkeypatch, strength, mode):
    r, a, pt, edges, tf, cfg, table, _, genes, tfs = fitted
    cfg = copy.deepcopy(cfg)
    cfg["perturbation"]["ko_strength"] = strength
    cfg["perturbation"]["joint_knockout"] = mode != "independent"
    cfg["perturbation"].setdefault("peak_ko", {}).update(strength=-strength, depth=3)
    def invoke():
        public = engine.perturb_peak_saturating if mode == "peak" else engine.propagate_saturating
        return public("p0" if mode == "peak" else ["A", "B"], r, a, pt, edges, table, tf, cfg,
            transfer_models=genes, tf_model_registry=tfs)
    actual = invoke()
    monkeypatch.setattr(engine, "_SinglesPlan", ReferencePlan)
    assert_same(actual, invoke())


@pytest.mark.parametrize("family", ["saturation", "spline", "nn"])
@pytest.mark.parametrize("minimum", [False, True])
def test_local_plan_matches_values_routes_and_all_diagnostics(family, minimum):
    data = local_data()
    if family == "spline":
        from CausalBridge.spline_hybrid import SplineConfig
        data = replace(data, provenance=dict(data.provenance, response_family=family,
            spline_response_config=SplineConfig(max_nfev=300, restarts=1).to_dict()))
    elif family == "nn":
        data = replace(data, provenance=dict(data.provenance, response_family=family,
            nn_response_config=dict(epochs=3)))
    model = core.fit_target(data, (0., 1.), core.HybridConfig(max_nfev=1000, min_group_rows=100))
    engine._bind_rows(model, dict(bin_to_support=list(range(len(data.X)))))
    supported = engine._input_support(model, model._query_global_rows, minimum)
    plan = engine._SinglesPlan(model, data.X, supported, model.input_keys)
    reference = ReferencePlan(model, data.X, supported, model.input_keys)
    for factor in (1., 0., .5, 5.):
        for row in range(len(data.X)):
            new = data.X[row:row+1]*factor
            # Unsupported placeholders can change without becoming active.
            new[~supported[row:row+1]] = 99.
            actual, expected = plan.predict(row, new), reference.predict(row, new)
            np.testing.assert_array_equal(actual.values, expected.values)
            assert actual.route == expected.route
            assert actual.group_indices == expected.group_indices
            assert core._plain(actual.diagnostics) == core._plain(expected.diagnostics)
            if actual.single_predictions is not None:
                old_predictions, new_predictions = actual.single_predictions
                contributions = [(i, float(weight*(new_predictions[0,i]-old_predictions[0,i])))
                    for i, weight in enumerate(model.weights) if supported[row,i]]
                expected_contributions, _ = engine._contributions(model, data.X[row:row+1], new,
                    expected.route, expected.group_indices, supported[row:row+1])
                assert contributions == expected_contributions
    with pytest.raises(ValueError, match="finite"):
        plan.predict(0, np.full(data.X[:1].shape, np.nan))
    with pytest.raises(ValueError, match="physical"):
        plan.predict(0, np.full(data.X[:1].shape, -1.))
    undeclared = engine._SinglesPlan(model, data.X, supported, ())
    with pytest.raises(ValueError, match="declared changed_inputs"):
        undeclared.predict(2, data.X[2:3]*.5)


@pytest.mark.parametrize("family", ["saturation", "spline", "nn"])
def test_gaussian_branch_local_complete_outputs_match(family, monkeypatch):
    from CausalBridge import kinetics
    from test_saturating_transfer import fixture_data
    r, a, pt, edges, _, cfg = fixture_data(32, 2)
    cfg["pseudotime"]["smooth_kernel"] = "gaussian"
    cfg["perturbation"]["min_lag_per_peak_filter"] = True
    for layer in ("atac_to_rna", "rna_to_atac"):
        cfg.setdefault(layer, {}).update(response_model=family,
            nn_response=dict(epochs=3), spline_response=dict(max_nfev=300, restarts=1))
    snapshot = transfer._snapshot(r, a, pt, cfg)
    records = kinetics._fit_tf_peak_regression({"p0": ["A", "C"], "p1": ["B"]},
        r, a, snapshot["R"], snapshot["A"], max_lag=8,
        branch_boundaries=snapshot["boundaries"], min_abs_contribution=.01)
    tf = pd.DataFrame(records)
    table, _, genes = transfer.fit_peak_gene(r, a, pt, edges, cfg)
    tfs = transfer.fit_tf_peak(r, a, pt, tf, cfg)
    def invoke():
        return engine.propagate_saturating(["A"], r, a, pt, edges, table, tf, cfg,
            transfer_models=genes, tf_model_registry=tfs)
    actual = invoke()
    monkeypatch.setattr(engine, "_SinglesPlan", ReferencePlan)
    assert_same(actual, invoke())


def test_no_information_preview_matches_general_predictor():
    data = local_data()
    data = replace(data, y=np.full(len(data.y), .5))
    model = core.fit_target(data, (0., 1.), core.HybridConfig(min_group_rows=100))
    assert model.status == "no_information"
    plan = engine._SinglesPlan(model, data.X, data.route_supported, model.input_keys)
    reference = ReferencePlan(model, data.X, data.route_supported, model.input_keys)
    for row in range(len(data.X)):
        new = data.X[row:row+1]*.5
        actual, expected = plan.predict(row, new), reference.predict(row, new)
        np.testing.assert_array_equal(actual.values, expected.values)
        assert actual.route == expected.route == "no_information"
        assert core._plain(actual.diagnostics) == core._plain(expected.diagnostics)


@pytest.mark.parametrize("peak", [False, True])
def test_normalized_atac_complete_outputs_match(fitted, monkeypatch, peak):
    from test_atac_normalization import continuous_refit
    r, a, pt, edges, tf, cfg, table, _, genes, tfs = continuous_refit(fitted)
    def invoke():
        public = engine.perturb_peak_saturating if peak else engine.propagate_saturating
        return public("p0" if peak else ["A"], r, a, pt, edges, table, tf, cfg,
            transfer_models=genes, tf_model_registry=tfs)
    actual = invoke()
    monkeypatch.setattr(engine, "_SinglesPlan", ReferencePlan)
    assert_same(actual, invoke())


def test_locked_group_complete_outputs_match(grouped_fit, monkeypatch):
    from test_saturating_perturbation import execute
    actual = execute(grouped_fit, targets=("A", "D"))
    assert actual["saturating_context_metadata"]["route_passes"] == 2
    assert actual["saturating_context_metadata"]["locked_groups"]
    monkeypatch.setattr(engine, "_SinglesPlan", ReferencePlan)
    assert_same(actual, execute(grouped_fit, targets=("A", "D")))


@pytest.mark.parametrize("restricted", [False, True])
def test_preview_group_fallback_diagnostics_match(grouped_fit, restricted):
    r, a, pt, _, _, cfg, _, _, _, tfs = grouped_fit
    model = core.TargetModel.from_bundle(tfs["targets"]["p0"])
    snapshot = transfer._snapshot(r, a, pt, cfg)
    engine._bind_rows(model, snapshot["context"])
    index = {name: i for i, name in enumerate(snapshot["context"]["gene_ids"])}
    old = engine._inputs(model, model._query_global_rows, snapshot["R"], index)
    supported = np.ones(old.shape, dtype=bool)
    if restricted:
        supported[:, 0] = False
    plan = engine._SinglesPlan(model, old, supported, model.input_keys)
    reference = ReferencePlan(model, old, supported, model.input_keys)
    actual, expected = plan.predict(0, old[:1]*.5), reference.predict(0, old[:1]*.5)
    np.testing.assert_array_equal(actual.values, expected.values)
    assert actual.route == expected.route == "singles"
    assert actual.diagnostics["group_fallbacks"]
    assert core._plain(actual.diagnostics) == core._plain(expected.diagnostics)


def test_wt_cache_is_shared_and_only_changed_new_responses_are_evaluated(monkeypatch):
    data = local_data()
    model = core.fit_target(data, (0., 1.), core.HybridConfig(max_nfev=1000, min_group_rows=100))
    position = 9
    assert data.route_supported[position].all()
    new = data.X[position:position+1].copy()
    new[0,0] *= .5
    expected = ReferencePlan(model, data.X, data.route_supported, model.input_keys).predict(position, new)
    restricted = data.route_supported.copy()
    restricted[position,1] = False
    restricted_expected = ReferencePlan(model, data.X, restricted, model.input_keys).predict(position, new)
    predict = core.SingleModel.predict
    calls = [0, 0]
    def counted(single, values):
        for i, fitted in enumerate(model.singles):
            if single is fitted:
                calls[i] += 1
        return predict(single, values)
    monkeypatch.setattr(core.SingleModel, "predict", counted)
    cache = engine._WTResponseCache(model, data.X)
    plan = engine._SinglesPlan(model, data.X, data.route_supported, model.input_keys, wt_cache=cache)
    actual = plan.predict(position, new)
    np.testing.assert_array_equal(actual.values, expected.values)
    assert calls == [2, 1]  # WT once per supported input; only changed new input.
    plan.predict(position, new)
    assert calls == [3, 1]
    second = engine._SinglesPlan(model, data.X, data.route_supported, model.input_keys, wt_cache=cache)
    assert second.old_values is plan.old_values
    second.predict(position, new)
    assert calls == [4, 1]  # Another intervention reuses the checked WT responses.
    second.predict(position, data.X[position:position+1])
    unsupported = data.X[3:4].copy()
    unsupported[0,0] = 99.
    second.predict(3, unsupported)
    assert calls == [4, 1]
    filtered = engine._SinglesPlan(model, data.X, restricted, model.input_keys, wt_cache=cache)
    assert filtered.old_values is not plan.old_values
    np.testing.assert_array_equal(filtered.predict(position, new).values, restricted_expected.values)
    assert calls == [5, 1]
    with pytest.raises(ValueError, match="WT response cache"):
        engine._SinglesPlan(model, data.X.copy(), data.route_supported, model.input_keys, wt_cache=cache)
