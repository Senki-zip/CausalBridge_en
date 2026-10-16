"""Synthetic regression coverage for pruning failed WT single relations."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from CausalBridge import saturating_hybrid as core
from CausalBridge import saturating_transfer as transfer
from CausalBridge.saturating_perturbation import _validate_screening_topology
from CausalBridge.run import _saturating_registry_reports


def fixture_data(p=3):
    x = np.linspace(.1, .9, 24)
    X = np.column_stack([x ** (i + 1) for i in range(p)])
    rows = np.arange(24)
    return core.TargetData("g", tuple((f"p{i}", 0) for i in range(p)), X,
        .5 + .3 * (x - .5) / (.2 + abs(x - .5)),
        np.tile(rows[:, None], (1, p)), rows, ("b",) * 24,
        "raw_atac", "raw_atac", {}, cell_weights=np.eye(24))


def injected_fit(monkeypatch, failed=(0,), numeric=False, constant=False):
    calls = []
    def fit(x, y, bounds, config):
        i = len(calls)
        calls.append(i)
        if numeric and i in failed:
            raise FloatingPointError("synthetic solver failure")
        status = "budget_exhausted" if i in failed else "constant_input" if constant else "ok"
        a = 0. if constant else .3
        m = core.SingleModel(.5, a, .5, .2, bounds, (float(x.min()), float(x.max())),
            dict(core._metrics(y, core._single_values(.5, a, .5, .2, x, bounds)),
                 status=status, termination_status=0 if i in failed else "not_needed" if constant else 1,
                 nfev=7))
        return m
    monkeypatch.setattr(core, "fit_single", fit)


@pytest.mark.parametrize("failed", [(0,), (0, 1)])
def test_pruned_scores_columns_quality_and_frozen_weights(monkeypatch, failed):
    d = fixture_data()
    injected_fit(monkeypatch, failed)
    m = core.fit_target(d, (0., 1.))
    keep = [i for i in range(3) if i not in failed]
    assert m.input_keys == tuple(d.input_keys[i] for i in keep)
    np.testing.assert_array_equal(m.source_bins, d.source_bins[:, keep])
    np.testing.assert_array_equal(m.target_rows, d.target_rows)
    predictions = np.column_stack([s.predict(d.X[:, i]) for s, i in zip(m.singles, keep)])
    q = np.mean(abs(np.diff(predictions, axis=0)), axis=0)
    np.testing.assert_array_equal(m.scores, q)
    np.testing.assert_allclose(m.weights, q / q.sum())
    assert len(m.target_rows) == len(d.y)
    assert transfer._net_metrics(m, d.X[:, keep], d.y)["status"] == "ok"
    assert all(max(g.indices) < len(keep) for g in m.groups)
    bundle = m.to_bundle()
    assert core.TargetModel.from_bundle(bundle).fingerprint == m.fingerprint
    audit = bundle["provenance"]["excluded_single_relations"]
    assert audit["original_input_count"] == 3
    assert [r["input_key"] for r in audit["records"]] == [list(d.input_keys[i]) for i in failed]
    assert all(r["nfev"] == 7 for r in audit["records"])
    before = m.weights
    core.predict_delta(m, d.X[:, keep], np.zeros_like(d.X[:, keep]), changed_inputs=m.input_keys,
                       branch_ids=m.branch_ids)
    assert m.weights == before


def test_all_failed_and_numeric_audit(monkeypatch):
    d = fixture_data()
    injected_fit(monkeypatch, (0, 1, 2), numeric=True)
    with pytest.raises(core.TargetFitFailure) as exc:
        core.fit_target(d, (0., 1.))
    audit = exc.value.diagnostics
    assert audit["status"] == "no_information" and audit["train_r2"] is None
    assert len(audit["excluded_single_relations"]["records"]) == 3
    assert all(r["training_rmse"] is None for r in audit["excluded_single_relations"]["records"])


def test_constant_retained_zero_and_value_error_not_swallowed(monkeypatch):
    d = fixture_data()
    injected_fit(monkeypatch, (0,), constant=True)
    m = core.fit_target(d, (0., 1.))
    assert m.status == "no_information" and m.weights == (0., 0.)
    def invalid(*args):
        raise ValueError("invalid protocol")
    monkeypatch.setattr(core, "fit_single", invalid)
    with pytest.raises(ValueError, match="invalid protocol"):
        core.fit_target(d, (0., 1.))
    with pytest.raises(ValueError, match="global WT"):
        core.fit_target(replace(d, output_unit="log1p_rna_log1p_cp10k"), (0., .1))


def test_topology_export_and_audit_validation(monkeypatch):
    d = fixture_data()
    injected_fit(monkeypatch)
    m = core.fit_target(d, (0., 1.))
    registry = dict(type=transfer.GENE_TYPE, targets={"g": m.to_bundle()}, failures={},
        target_metrics={}, input_unit=m.input_unit, output_unit=m.output_unit)
    empty = dict(type=transfer.TF_TYPE, targets={}, failures={})
    edges = pd.DataFrame(dict(gene=["g"] * 3, peak_id=["p0", "p1", "p2"], lag=[0] * 3))
    _validate_screening_topology(edges, pd.DataFrame(), registry, empty)
    foreign = edges.copy(); foreign.loc[0, "peak_id"] = "foreign"
    with pytest.raises(ValueError, match="input keys"):
        _validate_screening_topology(foreign, pd.DataFrame(), registry, empty)
    _, weights, _ = _saturating_registry_reports(registry, "peak_gene")
    excluded = weights.loc[~weights.included].iloc[0]
    assert excluded.input_id == "p0" and excluded.alpha == 0
    assert excluded.fit_status == "budget_exhausted" and excluded.nfev == 7
    provenance = core._plain(m.provenance)
    provenance["excluded_single_relations"]["records"][0]["input_key"] = list(m.input_keys[0])
    with pytest.raises(ValueError, match="identity"):
        replace(m, provenance=provenance)


def test_real_budget_failure():
    d = fixture_data(1)
    with pytest.raises(core.TargetFitFailure) as exc:
        core.fit_target(d, (0., 1.), core.HybridConfig(max_nfev=1))
    record = exc.value.diagnostics["excluded_single_relations"]["records"][0]
    assert record["status"] == "budget_exhausted" and record["nfev"] == 1


@pytest.mark.parametrize("policy", ["exclude", "best_finite"])
@pytest.mark.parametrize("kind", [transfer.GENE_TYPE, transfer.TF_TYPE])
@pytest.mark.parametrize("continuous", [False, True])
def test_real_mixed_budget_spawn_matches_serial(kind, continuous, policy):
    x = np.linspace(.1, .9, 24)
    X = np.column_stack([x, x**2])
    y = core._single_values(.5, .4, float(np.median(x)), float(np.ptp(x)), x, (0., 1.))
    atac_unit = "normalized_atac_log1p" if continuous else "raw_atac"
    units = (atac_unit, "log1p_rna_log1p_cp10k") if kind == transfer.GENE_TYPE else ("rna_log1p_cp10k", atac_unit)
    rows = np.arange(24)
    jobs = []
    for target in ("t0", "t1"):
        payload = dict(target_id=target, input_keys=(("good", 0), ("bad", 0)), X=X, y=y,
            source_bins=np.tile(rows[:, None], (1, 2)), target_rows=rows, branch_ids=("b",) * 24,
            input_unit=units[0], output_unit=units[1], provenance={"budget_exhausted_policy": policy}, support_rows=rows)
        jobs.append((target, 2, dict(data=payload, bounds=(0., 1.)), None))
    support = core.CellSupport(np.eye(24))
    config = core.HybridConfig(max_nfev=1)
    serial = transfer._execute_target_jobs(iter(jobs), support, config, 1, 2, 4, kind)
    parallel = transfer._execute_target_jobs(iter(jobs), support, config, 2, 2, 4, kind)
    assert parallel == serial
    for bundle, metric in parallel.values():
        if policy == "best_finite":
            assert bundle["input_keys"] == [["good", 0], ["bad", 0]]
            assert bundle["weights"][1] > 0
            assert bundle["singles"][1]["metrics"]["status"] == "best_finite"
            assert bundle["singles"][1]["metrics"]["solver_converged"] is False
            assert metric["status"] == "ok"
            continue
        assert bundle["input_keys"] == [["good", 0]]
        assert bundle["weights"] == [1.]
        assert metric["status"] == "ok" and metric["train_r2"] == pytest.approx(1.)
        assert bundle["provenance"]["excluded_single_relations"]["records"][0]["status"] == "budget_exhausted"


def test_best_finite_retained_scores_and_signed_codec(monkeypatch):
    d = fixture_data()
    injected_fit(monkeypatch)
    old = core.fit_target(d, (0., 1.))
    injected_fit(monkeypatch)
    accepted = core.fit_target(replace(d, provenance={"budget_exhausted_policy": "best_finite"}), (0., 1.))
    assert accepted.input_keys == d.input_keys
    predictions = np.column_stack([s.predict(d.X[:, i]) for i, s in enumerate(accepted.singles)])
    q = np.mean(abs(np.diff(predictions, axis=0)), axis=0)
    np.testing.assert_array_equal(accepted.scores, q)
    np.testing.assert_allclose(accepted.weights, q / q.sum())
    assert accepted.weights[1] != old.weights[0]
    metrics = accepted.singles[0].metrics
    assert metrics["status"] == "best_finite" and metrics["original_status"] == "budget_exhausted"
    assert metrics["termination_status"] == 0 and metrics["solver_converged"] is False
    bundle = accepted.to_bundle()
    assert core.TargetModel.from_bundle(bundle).to_bundle() == bundle
    bundle["singles"][0]["metrics"]["solver_converged"] = True
    bundle["fingerprint"] = core._digest({k: v for k, v in bundle.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="best-finite"):
        core.TargetModel.from_bundle(bundle)


def test_best_finite_numeric_failure_and_zero_information(monkeypatch):
    d = replace(fixture_data(), provenance={"budget_exhausted_policy": "best_finite"})
    injected_fit(monkeypatch, (0, 1, 2), numeric=True)
    with pytest.raises(core.TargetFitFailure) as exc:
        core.fit_target(d, (0., 1.))
    assert exc.value.diagnostics["status"] == "no_information"
    assert all(r["status"] == "failed" for r in exc.value.diagnostics["excluded_single_relations"]["records"])
    injected_fit(monkeypatch, (0,), numeric=True, constant=True)
    model = core.fit_target(d, (0., 1.))
    assert model.input_keys == d.input_keys[1:]
    assert model.status == "no_information" and model.scores == model.weights == (0., 0.)
