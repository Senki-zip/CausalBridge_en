"""Decisive reference compatibility and independent frozen-policy checks."""
import numpy as np
import pandas as pd
import pytest
from dataclasses import replace
from scipy import sparse
import anndata as ad

from CausalBridge.adaptive_binning import (
    BinningConfig, aggregate, build_plan, fit_policy, query_membership,
)
from CausalBridge import granger, kinetics, perturbation


def dense(x):
    return x.toarray() if sparse.issparse(x) else x


def baseline_config(**kwargs):
    """Explicit pre-migration legacy/ESS settings; keep baseline assertions intact."""
    settings = dict(mode="hard", branch_policy="legacy", bandwidth_selection="ess",
                    boundary="none", max_bandwidth_multiplier=.5)
    settings.update(kwargs)
    return BinningConfig(**settings)


@pytest.mark.parametrize("mode", ["hard", "gaussian"])
@pytest.mark.parametrize("csr", [False, True])
@pytest.mark.parametrize("case", ["irregular", "repeated", "small", "target", "overlap", "branches"])
def test_real_reference(mode, csr, case):
    rng = np.random.default_rng(7)
    times = np.sort(rng.uniform(0, 1, 53))
    branches = None
    kwargs = dict(max_bins=11, min_cells_per_bin=5)
    if case == "repeated":
        times = np.repeat([0., .25, .5, .75, 1.], 9)
    if case == "small":
        times = np.array([0., .3, .6, 1.])
        kwargs["min_cells_per_bin"] = 9
    if case == "target":
        kwargs.update(target_bins=13)
    if case == "overlap":
        kwargs.update(target_bins=17, overlap_factor=.7)
    if case == "branches":
        branches = np.array(["a"]*19 + ["b"]*34)
        times[:19] *= .4
        times[19:] = .6 + times[19:]*.4
    ids = np.array([f"c{i}" for i in range(len(times))])
    pt = pd.DataFrame({"pseudotime": times}, index=ids)
    if branches is not None:
        pt["branch"] = branches
    reference_pt = pt.copy()
    labels, n = kinetics._adaptive_rebin(reference_pt, len(times), **kwargs)
    reference_pt["bin"] = labels
    cfg = baseline_config(mode=mode, **kwargs)
    policy = fit_policy(times, ids, branches, config=cfg)
    plan = policy.initial_plan
    np.testing.assert_array_equal(plan.labels, labels)
    assert plan.n_bins == n
    if branches is not None:
        assert list(policy.branch_boundaries) == reference_pt.attrs["branch_boundaries"]
        assert all(policy.row_branches[a] == policy.row_branches[b] for a, b in policy.neighbors)
    matrix = rng.uniform(.2, 3, (len(times), 3))
    source = sparse.csr_matrix(matrix) if csr else matrix
    expected = kinetics._bin_matrix(source, reference_pt, n, smooth=mode, min_cells_per_bin=kwargs["min_cells_per_bin"])
    np.testing.assert_allclose(aggregate(plan, source, cell_ids=ids), expected, atol=1e-12, rtol=1e-12)
    rna = ad.AnnData(source, obs=pd.DataFrame(index=ids))
    np.testing.assert_allclose(granger._bin_expression(rna, reference_pt, n, smooth=mode, min_cells_per_bin=kwargs["min_cells_per_bin"]), expected, atol=1e-12)
    from CausalBridge._wt_aggregation import _weights
    kernel, _ = _weights(reference_pt, n, {"pseudotime": {
        "smooth_kernel": mode, "min_cells_per_bin": kwargs["min_cells_per_bin"]}}, None)
    kernel = kernel.toarray()
    projector = kernel / (kernel.sum(axis=1, keepdims=True) + 1e-10)
    membership = kernel.T / (kernel.sum(axis=0, keepdims=True).T + 1e-10)
    np.testing.assert_allclose(dense(plan.projection_weights), projector, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(dense(plan.membership), membership, atol=1e-12, rtol=1e-12)
    if mode == "hard":
        assert sparse.isspmatrix_csr(plan.raw_kernel)
        counts = dense(plan.raw_kernel).sum(1)
        good = counts > 0
        np.testing.assert_allclose(dense(plan.aggregate_weights)[good], projector[good] * ((counts[good]+1e-10)/counts[good])[:, None], atol=1e-14)
    reapplied = build_plan(policy, times, ids, branches)
    np.testing.assert_array_equal(reapplied.labels, labels)
    np.testing.assert_allclose(dense(reapplied.raw_kernel), dense(plan.raw_kernel), atol=1e-14)


def test_branch_masks_and_shared_plan():
    times = np.tile(np.linspace(0, 1, 60), 2)
    ids = np.arange(120)
    branches = np.repeat(["a", "b"], 60)
    matrix = np.column_stack([times, branches == "b"])
    legacy = fit_policy(times, ids, branches, config=baseline_config(mode="gaussian", max_bins=12))
    assert legacy.initial_plan.diagnostics["legacy_cross_branch_mixing"]
    separate = fit_policy(times, ids, branches, config=baseline_config(mode="gaussian", branch_policy="separate", max_bins=12))
    plan = separate.initial_plan
    assert not plan.diagnostics["legacy_cross_branch_mixing"]
    for i, name in enumerate(separate.row_branches):
        assert np.all(plan.raw_kernel[i, branches != name] == 0)
    np.testing.assert_allclose(aggregate(plan, matrix, cell_ids=ids), aggregate(plan, sparse.csr_matrix(matrix), cell_ids=ids))
    # A single shared plan is applicable to arbitrary RNA/ATAC column counts.
    assert aggregate(plan, matrix[:, :1], cell_ids=ids).shape == (12, 1)
    assert aggregate(plan, matrix, cell_ids=ids).shape == (12, 2)


def test_frozen_queries_and_unseen_ranges():
    times = np.linspace(0, 1, 200); ids = np.arange(200)
    policy = fit_policy(times, ids, config=baseline_config(mode="adaptive_gaussian", branch_policy="separate", max_bins=10))
    centers = policy.centers.copy(); h = policy.bandwidths.copy(); fingerprint = policy.fingerprint
    q = np.array([.1, .7, -1., 2., .5]); qbr = np.array(["__global__"]*4 + ["unseen"])
    membership, valid = query_membership(policy, q, np.arange(5), qbr)
    np.testing.assert_array_equal(valid, [True, True, False, False, False])
    np.testing.assert_allclose(dense(membership)[~valid], 0)
    np.testing.assert_allclose(dense(membership)[valid].sum(1), 1, atol=1e-8)
    other = build_plan(policy, q[:2], ["new0", "new1"])
    np.testing.assert_array_equal(other.centers, centers)
    np.testing.assert_array_equal(other.bandwidths, h)
    assert policy.fingerprint == fingerprint
    assert not other.valid.any()  # two query cells cannot support an estimate
    assert np.isnan(aggregate(other, np.ones((2, 1)), cell_ids=["new0", "new1"])).all()


def test_boundary_mask_and_reproducible_noise_splits():
    times = np.linspace(0, 1, 240); ids = np.arange(240)
    embedding = pd.DataFrame({"external": np.where(times < .5, 0., 10.)}, index=ids)
    cfg = baseline_config(mode="adaptive_gaussian", branch_policy="separate", max_bins=12, boundary="split_noise", max_bandwidth_multiplier=2.)
    first = fit_policy(times, ids, embedding=embedding, config=cfg)
    second = fit_policy(times, ids, embedding=embedding, config=cfg)
    assert first.fingerprint == second.fingerprint
    assert first.boundary_diagnostics == second.boundary_diagnostics
    assert any(d["status"] == "stable" for d in first.boundary_diagnostics)
    edge = next(edge for _, edge in first.stable_boundaries)
    raw = first.initial_plan.raw_kernel
    for i, center in enumerate(first.centers):
        assert np.all(raw[i, (times < edge) != (center < edge)] == 0)
    membership, valid = query_membership(first, [edge], ["at_boundary"])
    assert valid[0]
    assert np.all(dense(membership)[0, first.centers < edge] == 0)
    uncertain = fit_policy(times, ids, embedding=np.zeros((240, 1)), config=cfg)
    assert not uncertain.stable_boundaries
    assert all(d["status"] == "uncertain_noise" for d in uncertain.boundary_diagnostics)


def test_bandwidth_candidates_and_reliability():
    times = np.linspace(0, 1, 400); ids = np.arange(400)
    cfg = baseline_config(mode="adaptive_gaussian", branch_policy="separate", max_bins=10)
    p = fit_policy(times, ids, config=cfg)
    spacing = (times.max()-times.min())/10
    assert np.isin(p.bandwidths/spacing, [.25, .5]).all()
    for row in range(p.initial_plan.n_bins):
        small = np.exp(-.5*((times-p.centers[row])/(spacing*.25))**2)
        ess = small.sum()**2/(small*small).sum()
        assert p.bandwidths[row] == spacing*(.25 if ess >= 40 else .5)
    unmet = fit_policy(times, ids, config=baseline_config(mode="adaptive_gaussian", branch_policy="separate", max_bins=10, target_ess=10000))
    assert unmet.initial_plan.valid.all() and not unmet.initial_plan.reliable.any()
    assert np.isfinite(aggregate(unmet.initial_plan, times[:, None], cell_ids=ids)).all()


def test_empty_bins_and_minimum_support():
    times = np.repeat([0., 1.], 10); ids = np.arange(20)
    p = fit_policy(times, ids, config=baseline_config(target_bins=100, min_cells_per_bin=1))
    empty = p.initial_plan.diagnostics["support_cells"] == 0
    assert empty.any()
    assert not p.initial_plan.valid[empty].any()
    assert np.all(aggregate(p.initial_plan, np.ones((20, 1)), cell_ids=ids)[empty] == 0)
    separate = fit_policy(times, ids, config=baseline_config(branch_policy="separate", target_bins=100, min_cells_per_bin=1))
    assert np.isnan(aggregate(separate.initial_plan, np.ones((20, 1)), cell_ids=ids)[empty]).all()
    small = fit_policy(np.linspace(0, 1, 4), np.arange(4), config=baseline_config(mode="gaussian", branch_policy="separate"))
    assert not small.initial_plan.valid.any()


def test_fingerprints_and_owned_readonly_inputs():
    times = np.linspace(0, 1, 60); ids = np.arange(60); emb = np.zeros((60, 2))
    cfg = baseline_config()
    p = fit_policy(times, ids, embedding=emb, config=cfg)
    assert p.fingerprint == fit_policy(times, ids, embedding=emb, config=cfg).fingerprint
    changed = emb.copy(); changed[0, 0] = 1
    assert p.fingerprint != fit_policy(times, ids, embedding=changed, config=cfg).fingerprint
    assert p.fingerprint != fit_policy(times**2, ids, embedding=emb, config=cfg).fingerprint
    assert p.fingerprint != fit_policy(times, ids+100, embedding=emb, config=cfg).fingerprint
    assert p.fingerprint != fit_policy(times, ids, embedding=emb, config=baseline_config(max_bins=8)).fingerprint
    times[:] = 50; emb[:] = 99; ids[:] = 999
    assert p.train_times.max() == 1 and p.train_embedding.max() == 0
    with pytest.raises(ValueError):
        p.centers[0] = 99
    with pytest.raises(ValueError):
        p.initial_plan.raw_kernel.data[0] = 2


@pytest.mark.parametrize("times,ids", [([], []), ([0, np.nan], [1, 2]), ([0, 1], [1, 1]), ([0, 1], [1]), ([0, 0], [1, 2])])
def test_reject_invalid_inputs(times, ids):
    with pytest.raises(ValueError):
        fit_policy(times, ids, config=baseline_config())


def test_reject_alignment_embedding_config_budget():
    times = np.linspace(0, 1, 40); ids = np.arange(40)
    with pytest.raises(ValueError):
        baseline_config(mode="adaptive_gaussian")
    with pytest.raises(ValueError):
        baseline_config(boundary="split_noise")
    with pytest.raises(ValueError):
        fit_policy(times, ids, embedding=np.full((40, 1), np.nan))
    with pytest.raises(ValueError):
        fit_policy(times, ids, embedding=pd.DataFrame(np.ones((40, 1)), index=ids[::-1]))
    with pytest.raises(ValueError):
        fit_policy(times, ids, config=baseline_config(mode="adaptive_gaussian", branch_policy="separate", boundary="split_noise"))
    with pytest.raises(MemoryError):
        fit_policy(times, ids, config=baseline_config(mode="gaussian", max_kernel_elements=10))
    p = fit_policy(times, ids, config=baseline_config())
    with pytest.raises(ValueError):
        aggregate(p.initial_plan, np.ones((40, 1)), cell_ids=ids[::-1])
    with pytest.raises(TypeError):
        aggregate(p.initial_plan, np.ones((40, 1)))
    with pytest.raises(ValueError):
        aggregate(p.initial_plan, np.full((40, 1), np.nan), cell_ids=ids)
    with pytest.raises(ValueError):
        build_plan(p, [], [])


def test_no_business_imports_and_budget_before_dense_allocation(monkeypatch):
    import CausalBridge.adaptive_binning as standalone
    original = np.exp
    calls = []
    def exp(x):
        calls.append(x.shape)
        return original(x)
    monkeypatch.setattr(standalone.np, "exp", exp)
    with pytest.raises(MemoryError):
        fit_policy(np.linspace(0, 1, 100), np.arange(100), config=baseline_config(mode="gaussian", max_kernel_elements=20))
    assert not calls


@pytest.mark.parametrize("mode", ["hard", "gaussian"])
def test_legacy_branch_minimum_exceeds_requested_total(mode):
    times = np.tile(np.linspace(0, 1, 10), 3)
    branches = np.repeat(["a", "b", "c"], 10)
    ids = np.arange(30)
    pt = pd.DataFrame({"pseudotime": times, "branch": branches}, index=ids)
    labels, total = kinetics._adaptive_rebin(pt, 30, max_bins=5, min_cells_per_bin=5)
    policy = fit_policy(times, ids, branches, config=baseline_config(mode=mode, max_bins=5))
    assert total == policy.initial_plan.n_bins == 9
    np.testing.assert_array_equal(policy.initial_plan.labels, labels)


def test_plan_fingerprint_identifies_query_and_order():
    p = fit_policy(np.linspace(0, 1, 60), np.arange(60), config=baseline_config())
    initial = p.initial_plan
    assert initial.policy_fingerprint == p.fingerprint
    assert initial.fingerprint == build_plan(p, p.train_times, p.train_ids).fingerprint
    q = build_plan(p, [.1, .8], ["x", "y"])
    assert q.policy_fingerprint == p.fingerprint
    assert q.fingerprint == build_plan(p, [.1, .8], ["x", "y"]).fingerprint
    alternatives = [build_plan(p, [.8, .1], ["y", "x"]),
                    build_plan(p, [.2, .8], ["x", "y"]),
                    build_plan(p, [.1, .8], ["z", "y"]),
                    build_plan(p, [.1, .8], ["x", "y"], ["unseen", "__global__"])]
    assert len({initial.fingerprint, q.fingerprint, *(x.fingerprint for x in alternatives)}) == 6
    assert all(x.policy_fingerprint == p.fingerprint for x in alternatives)


@pytest.mark.parametrize("mode", ["hard", "gaussian", "adaptive_gaussian"])
def test_huge_requested_qcut_budget_preflight(monkeypatch, mode):
    def forbidden(*args, **kwargs):
        pytest.fail("pd.qcut called before resource preflight")
    monkeypatch.setattr(pd, "qcut", forbidden)
    cfg = baseline_config(mode=mode, branch_policy="separate", target_bins=10**12)
    with pytest.raises(MemoryError, match="max_kernel_elements"):
        fit_policy(np.linspace(0, 1, 30), np.arange(30), config=cfg)
    # Account for legacy branch-minimum overflow, not just requested=5.
    with pytest.raises(MemoryError, match="9x30"):
        fit_policy(np.tile(np.linspace(0, 1, 10), 3), np.arange(30),
                   np.repeat(["a", "b", "c"], 10),
                   config=baseline_config(mode=mode, branch_policy="separate", max_bins=5, max_kernel_elements=200))


def test_boundary_fixed_time_pilots_under_uneven_density():
    times = np.r_[np.linspace(0, .49, 240), np.linspace(.51, 1, 60)]
    cfg = baseline_config(mode="adaptive_gaussian", branch_policy="separate", max_bins=10,
                        boundary="split_noise", boundary_min_side=5)
    p = fit_policy(times, np.arange(len(times)), embedding=np.where(times < .5, 0., 10.)[:, None], config=cfg)
    assert len(p.stable_boundaries) == 1
    assert p.stable_boundaries[0][0] == "__global__"
    assert p.stable_boundaries[0][1] == pytest.approx(.5)
    for d in p.boundary_diagnostics:
        i = d["left_bin"]
        edges = (p.centers[:-1] + p.centers[1:]) / 2
        pilots = np.searchsorted(edges, times, side="right")
        assert d["left_count"] == (pilots == i).sum()
        assert d["right_count"] == (pilots == i+1).sum()
    raw = p.initial_plan.raw_kernel
    assert np.all(raw[p.centers < .5][:, times >= .5] == 0)
    assert np.all(raw[p.centers >= .5][:, times < .5] == 0)


def test_boundary_equal_counts_genuine_resampling_and_noisy_null():
    times = (np.arange(400) + .5) / 400
    cfg = baseline_config(mode="adaptive_gaussian", branch_policy="separate", max_bins=10,
                        boundary="split_noise", boundary_splits=40)
    embedding = np.random.default_rng(321).normal(size=(400, 2))
    p = fit_policy(times, np.arange(400), embedding=embedding, config=cfg)
    repeat = fit_policy(times, np.arange(400), embedding=embedding, config=cfg)
    assert p.boundary_diagnostics == repeat.boundary_diagnostics
    assert len(p.stable_boundaries) < len(p.boundary_diagnostics) / 2
    for d in p.boundary_diagnostics:
        assert d["left_count"] == d["right_count"] == 40
        assert d["resample_size"] == 20
        assert np.ptp(d["observed_samples"]) > 0
        assert np.ptp(d["noise_samples"]) > 0


@pytest.mark.parametrize("mode", ["hard", "gaussian"])
def test_forward_reconstruction_selects_valid_training_bins(mode):
    times = np.r_[np.linspace(0, 1, 50), [0., 1.]]
    branches = np.array(["a"]*50 + ["b"]*2)
    ids = np.arange(len(times))
    p = fit_policy(times, ids, branches, config=baseline_config(mode=mode, branch_policy="separate", max_bins=8))
    y = aggregate(p.initial_plan, np.ones((len(times), 1)), cell_ids=ids)
    assert np.isnan(y).any() and p.initial_plan.valid.any()
    membership, valid = query_membership(p, [.5, .5, 2.], ["q1", "q2", "q3"], ["a", "b", "a"])
    good = p.initial_plan.valid
    reconstructed = np.asarray(membership[:, good] @ y[good])
    reconstructed[~valid] = np.nan
    np.testing.assert_array_equal(valid, [True, False, False])
    np.testing.assert_allclose(reconstructed[valid], 1, atol=1e-8)
    assert np.isnan(reconstructed[~valid]).all()


def state_config(**kwargs):
    return replace(baseline_config(mode="adaptive_gaussian", branch_policy="separate",
                                 bandwidth_selection="state", max_bins=10, max_bandwidth_multiplier=2.), **kwargs)


def state_fixture():
    times = (np.arange(1600)+.5)/1600
    return times, np.arange(len(times)), np.random.default_rng(17).normal(size=(len(times), 1))


@pytest.mark.parametrize("setting", [{"bandwidth_selection": "bad"}, {"state_budget": -1},
                                     {"state_floor_fraction": np.nan}, {"state_uncertainty_multiplier": np.inf},
                                     {"state_pilot_subdivisions": 0}, {"state_min_pilot_cells": 2},
                                     {"state_bootstrap_repeats": 1}, {"random_seed": -1},
                                     {"state_floor_fraction": 1.1}, {"state_bootstrap_repeats": np.inf},
                                     {"random_seed": np.inf}])
def test_state_invalid_config(setting):
    with pytest.raises(ValueError):
        state_config(**setting)


def test_state_requires_external_training_embedding_and_adaptive_separate():
    with pytest.raises(ValueError):
        baseline_config(bandwidth_selection="state")
    times, ids, _ = state_fixture()
    with pytest.raises(ValueError, match="external aligned training WT embedding"):
        fit_policy(times, ids, config=state_config())
    with pytest.raises(ValueError):
        fit_policy(times, ids, embedding=np.full((len(times), 1), np.nan), config=state_config())
    with pytest.raises(ValueError):
        fit_policy(times, ids, embedding=pd.DataFrame(np.ones((len(times), 1)), index=ids[::-1]), config=state_config())


@pytest.mark.parametrize("sigma", [0., .1, .5])
def test_state_flat_selects_widest_finite_and_reproducible(sigma):
    times, ids, noise = state_fixture()
    p = fit_policy(times, ids, embedding=sigma*noise, config=state_config())
    repeat = fit_policy(times, ids, embedding=sigma*noise, config=state_config())
    d = p.bandwidth_diagnostics
    np.testing.assert_array_equal(d["chosen_multiplier"], 2)
    assert np.isfinite(d["score"]).all()
    np.testing.assert_array_equal(d["score"], repeat.bandwidth_diagnostics["score"])
    assert p.fingerprint == repeat.fingerprint
    if sigma == 0:
        assert d["floor_squared"] == d["global_scale_squared"] == 0
        np.testing.assert_array_equal(d["score"], 0)
    with pytest.raises(ValueError):
        d["score"][0, 0] = 99


def test_state_noiseless_drift_floor_middle_and_no_budget_acceptance():
    times, ids, _ = state_fixture()
    p = fit_policy(times, ids, embedding=times[:, None], config=state_config())
    d = p.bandwidth_diagnostics
    assert d["floor_squared"] > 0
    np.testing.assert_array_equal(d["chosen_multiplier"], .5)
    assert np.isfinite(d["score"]).all()
    # Local-linear fit removes gradual drift from per-cell residual noise.
    assert d["pilot_noise"].max() < 1e-25
    no = fit_policy(times, ids, embedding=times[:, None], config=state_config(state_budget=0.))
    np.testing.assert_array_equal(no.bandwidth_diagnostics["chosen_multiplier"], .25)
    assert np.all(no.bandwidth_diagnostics["reason"] == "state_budget_unmet_at_min")
    assert not no.bandwidth_diagnostics["state_confident"].any()


@pytest.mark.parametrize("offset", [.45, .5, .517, .63])
def test_state_shifted_steps_narrow_near_wider_away(offset):
    times, ids, noise = state_fixture()
    p = fit_policy(times, ids, embedding=4*(times[:, None]>=offset)+.1*noise, config=state_config())
    d = p.bandwidth_diagnostics
    near = np.abs(p.centers-offset) < .1
    away = np.abs(p.centers-offset) > .3
    assert np.max(d["chosen_multiplier"][near]) <= .5
    assert np.median(d["chosen_multiplier"][away]) > np.median(d["chosen_multiplier"][near])
    # Every candidate was evaluated, and selection uses the largest acceptance.
    for i in range(len(p.centers)):
        accepted = d["candidates"][d["accepted"][i]]
        if len(accepted):
            assert d["chosen_multiplier"][i] == accepted.max()


def test_state_paired_higher_noise_widens_not_ess():
    times, ids, noise = state_fixture()
    low = fit_policy(times, ids, embedding=times[:, None]+.1*noise, config=state_config())
    high = fit_policy(times, ids, embedding=times[:, None]+.5*noise, config=state_config())
    assert high.bandwidths.mean() > low.bandwidths.mean()
    unmet = fit_policy(times, ids, embedding=times[:, None]+.1*noise, config=state_config(target_ess=1e9))
    np.testing.assert_array_equal(low.bandwidths, unmet.bandwidths)
    np.testing.assert_array_equal(low.bandwidth_diagnostics["score"], unmet.bandwidth_diagnostics["score"])
    assert low.initial_plan.reliable.all() and not unmet.initial_plan.reliable.any()


def test_state_missing_pilots_choose_max_flag_and_query_freeze():
    times = (np.arange(80)+.5)/80
    p = fit_policy(times, np.arange(len(times)), embedding=times[:, None], config=state_config())
    d = p.bandwidth_diagnostics
    assert np.all(d["reason"] == "insufficient_state_evidence")
    np.testing.assert_array_equal(d["chosen_multiplier"], 2)
    assert not d["state_confident"].any()
    for query in ([.1, .9], np.linspace(.2, .4, 100)):
        plan = build_plan(p, query, np.arange(len(query))+1000)
        np.testing.assert_array_equal(plan.bandwidths, p.bandwidths)
        assert plan.bandwidth_diagnostics is p.bandwidth_diagnostics
        assert plan.bandwidth_diagnostics["source"] == "frozen_training"


def test_state_local_linear_oracle_and_geometric_dispersion():
    times, ids, noise = state_fixture()
    p = fit_policy(times, ids, embedding=times[:, None]+.1*noise, config=state_config())
    d = p.bandwidth_diagnostics
    s = d["pilot_centers"][0]
    mask = times < .025
    x = times[mask]; z = p.train_embedding[mask]
    design = np.column_stack([np.ones(len(x)), x-s])
    beta = np.linalg.lstsq(design, z, rcond=None)[0]
    residual = z-design @ beta
    expected_noise = (residual**2).sum()/(len(x)-2)
    expected_v = expected_noise*(1/len(x)+(s-x.mean())**2/np.sum((x-x.mean())**2))
    np.testing.assert_allclose(d["pilot_means"][0], beta[0], atol=1e-14)
    np.testing.assert_allclose(d["pilot_noise"][0], expected_noise, rtol=1e-12)
    np.testing.assert_allclose(d["pilot_mean_variance"][0], expected_v, rtol=1e-12)
    spacing = (times.max()-times.min())/len(p.centers)
    weights = np.exp(-.5*((d["pilot_centers"]-p.centers[3])/spacing)**2)
    weights /= weights.sum()
    mu = d["pilot_means"]
    expected_D = max(0., weights @ np.sum((mu-weights @ mu)**2, axis=1)
                     - (weights*(1-weights)) @ d["pilot_mean_variance"])
    np.testing.assert_allclose(d["D"][3, 2], expected_D)
    np.testing.assert_allclose(d["N"][3, 2], weights @ d["pilot_noise"])
    expected_score = np.sqrt(np.maximum(0, d["D"]-2*d["U"])/(d["N"]+d["floor_squared"]))
    np.testing.assert_allclose(d["score"], expected_score)
    assert np.all(d["geometric_coverage"] == 1)


def test_state_fingerprint_parameters_seed_and_storage_preflight():
    times, ids, noise = state_fixture()
    p = fit_policy(times, ids, embedding=noise, config=state_config())
    for change in ({"random_seed": 43}, {"state_budget": .8}, {"state_floor_fraction": .3}):
        other = fit_policy(times, ids, embedding=noise, config=state_config(**change))
        assert other.fingerprint != p.fingerprint
    # qcut/kernel fit fits this budget; state bootstrap storage does not.
    with pytest.raises(MemoryError, match="state working storage"):
        fit_policy(times, ids, embedding=np.tile(noise, (1, 8)), config=state_config(max_kernel_elements=20000))


def test_state_irregular_density_geometric_not_cell_count_weighted():
    times = np.r_[(np.arange(1200)+.5)/2400, .5+(np.arange(400)+.5)/800]
    embedding = times[:, None]+.05*np.random.default_rng(71).normal(size=(len(times), 1))
    p = fit_policy(times, np.arange(len(times)), embedding=embedding, config=state_config())
    d = p.bandwidth_diagnostics
    assert d["pilot_cell_counts"].max() > 2*d["pilot_cell_counts"].min()
    spacing = (times.max()-times.min())/len(p.centers)
    weights = np.exp(-.5*((d["pilot_centers"]-p.centers[4])/(2*spacing))**2)
    weights /= weights.sum()
    np.testing.assert_allclose(d["N"][4, 3], weights @ d["pilot_noise"])
    scale = np.mean(np.sum((d["pilot_means"]-d["pilot_means"].mean(axis=0))**2, axis=1))
    scale -= (1-1/len(weights))*d["pilot_mean_variance"].mean()
    np.testing.assert_allclose(d["global_scale_squared"], max(0, scale))


def test_state_identical_pilot_times_intercept_only():
    times = np.repeat(np.linspace(0, 1, 40), 10)
    p = fit_policy(times, np.arange(len(times)), embedding=times[:, None], config=state_config())
    d = p.bandwidth_diagnostics
    np.testing.assert_array_equal(d["pilot_rank"], 1)
    assert d["pilot_supported"].all()
    assert np.isfinite(d["score"]).all()


def test_state_boundary_masks_and_separate_rng_stream():
    times, ids, _ = state_fixture()
    cfg = state_config(boundary="split_noise", boundary_min_side=5)
    embedding = 4*(times[:, None]>=.5)
    p = fit_policy(times, ids, embedding=embedding, config=cfg)
    assert len(p.stable_boundaries) == 1
    repeat = fit_policy(times, ids, embedding=embedding, config=replace(cfg, boundary_splits=40))
    assert repeat.stable_boundaries == p.stable_boundaries
    np.testing.assert_array_equal(p.bandwidth_diagnostics["score"], repeat.bandwidth_diagnostics["score"])
    np.testing.assert_array_equal(p.bandwidth_diagnostics["chosen_multiplier"], 2)
    edge = p.stable_boundaries[0][1]
    for i, center in enumerate(p.centers):
        assert np.all(p.initial_plan.raw_kernel[i, (times<edge) != (center<edge)] == 0)


def test_state_bootstrap_uncertainty_independent_oracle():
    times, ids, noise = state_fixture()
    p = fit_policy(times, ids, embedding=times[:, None]+.1*noise, config=state_config())
    d = p.bandwidth_diagnostics
    edges = np.linspace(times.min(), times.max(), 41)
    labels = np.searchsorted(edges[1:-1], times, side="right")
    rng = np.random.default_rng(np.random.SeedSequence([42, 0x53544154]))
    boot = []
    for j, s in enumerate(d["pilot_centers"]):
        x = times[labels == j]; z = p.train_embedding[labels == j]
        X = np.column_stack([np.ones(len(x)), x-s])
        inverse = np.linalg.pinv(X)
        residual = z-X @ (inverse @ z)
        residual -= residual.mean(axis=0)
        residual *= np.sqrt(len(x)/(len(x)-2))
        boot.append([inverse[0] @ residual[rng.integers(0, len(x), len(x))] for _ in range(32)])
    boot = np.asarray(boot)
    spacing = (times.max()-times.min())/len(p.centers)
    weights = np.exp(-.5*((d["pilot_centers"]-p.centers[4])/spacing)**2)
    weights /= weights.sum()
    correction = (weights*(1-weights)) @ d["pilot_mean_variance"]
    distances = []
    for r in range(32):
        mu = d["pilot_means"]+boot[:, r]
        distances.append(max(0, weights @ np.sum((mu-weights @ mu)**2, axis=1)-correction))
    np.testing.assert_allclose(d["U"][4, 2], np.std(distances, ddof=1), rtol=1e-10)


def test_state_branch_scale_pooling_equal_not_density_weighted():
    times = np.r_[(np.arange(1200)+.5)/1200, (np.arange(400)+.5)/400]
    branches = np.repeat(["a", "b"], [1200, 400])
    mean = times*np.where(branches == "a", 1., 2.)+100*(branches == "b")
    embedding = mean[:, None]+.05*np.random.default_rng(19).normal(size=(len(times), 1))
    p = fit_policy(times, np.arange(len(times)), branches, embedding=embedding, config=state_config())
    d = p.bandwidth_diagnostics
    scales = []
    for branch in ("a", "b"):
        good = (d["pilot_branches"] == branch) & d["pilot_supported"]
        mu = d["pilot_means"][good]
        scales.append(max(0, np.mean(np.sum((mu-mu.mean(axis=0))**2, axis=1))
                          -(1-1/len(mu))*d["pilot_mean_variance"][good].mean()))
    np.testing.assert_allclose(d["global_scale_squared"], np.mean(scales))
    for i, branch in enumerate(p.row_branches):
        assert np.all(p.initial_plan.raw_kernel[i, branches != branch] == 0)


@pytest.mark.parametrize("control", ["accepted", "none_accepted", "all_absent"])
def test_state_sparse_gap_partial_evidence_never_overrides_known_rejection(control):
    times = np.r_[np.linspace(0, .1, 400), np.linspace(.9, 1, 400)]
    ids = np.arange(len(times))
    embedding = np.r_[np.zeros(400), np.full(400, 4.)][:, None]
    cfg = state_config(max_bins=30)
    if control == "none_accepted":
        cfg = replace(cfg, state_budget=0.)
    elif control == "all_absent":
        cfg = replace(cfg, state_min_pilot_cells=100)
    p = fit_policy(times, ids, embedding=embedding, config=cfg)
    d = p.bandwidth_diagnostics
    for i in (14, 15):
        if control == "all_absent":
            assert not d["evidence"][i].any()
            assert d["chosen_multiplier"][i] == 2
            assert d["reason"][i] == "insufficient_state_evidence"
            assert not d["chosen_score_evidenced"][i]
        else:
            np.testing.assert_array_equal(d["evidence"][i], [False, True, True, True])
            assert np.isnan(d["score"][i, 0])
            assert d["score"][i, 3] == pytest.approx(2.047, abs=.002)
            assert not d["accepted"][i, 3]
            assert d["partial_candidate_evidence"][i]
            assert d["chosen_score_evidenced"][i]
            if control == "accepted":
                assert d["chosen_multiplier"][i] == 1
                assert d["accepted"][i, 2]
                assert d["reason"][i] == "largest_accepted"
            else:
                assert d["chosen_multiplier"][i] == .5
                assert not d["accepted"][i].any()
                assert d["reason"][i] == "state_budget_unmet_at_min_evidenced"
                assert d["state_budget_unmet_at_min"][i]
        assert not d["state_confident"][i]
    repeat = fit_policy(times, ids, embedding=embedding, config=cfg)
    assert p.fingerprint == repeat.fingerprint
    np.testing.assert_array_equal(d["score"], repeat.bandwidth_diagnostics["score"])
    reapplied = build_plan(p, times, ids)
    assert reapplied.fingerprint == p.initial_plan.fingerprint
    query = build_plan(p, [.48, .52], ["q1", "q2"])
    assert query.policy_fingerprint == p.fingerprint
    assert query.bandwidth_diagnostics is d
    np.testing.assert_array_equal(query.bandwidths, p.bandwidths)


def test_default_config_is_standalone_state_boundary_combo():
    cfg = BinningConfig()
    assert (cfg.mode, cfg.branch_policy, cfg.bandwidth_selection, cfg.boundary) == (
        "adaptive_gaussian", "separate", "state", "split_noise")
    assert cfg.max_bandwidth_multiplier == 2.
    assert cfg.bandwidth_multipliers == (.25, .5, 1., 2.)
    assert cfg.max_bins == 100  # independent module default; not benchmark's 30
    assert cfg.state_budget == 1. and cfg.state_floor_fraction == .2
    assert cfg.state_uncertainty_multiplier == 2.
    assert (cfg.state_pilot_subdivisions, cfg.state_min_pilot_cells, cfg.state_bootstrap_repeats) == (4, 4, 32)
    assert (cfg.boundary_splits, cfg.boundary_min_side, cfg.boundary_stability) == (20, 10, .8)


@pytest.mark.parametrize("config", [None, BinningConfig()])
def test_default_combo_requires_training_embedding_no_ess_fallback(config):
    with pytest.raises(ValueError, match="external aligned training WT embedding"):
        fit_policy(np.linspace(0, 1, 60), np.arange(60), config=config)


def test_default_combo_training_step_boundary_isolation_and_frozen_queries():
    times = np.linspace(0, 1, 2400)
    ids = np.array([f"wt_{i}" for i in range(len(times))])
    embedding = pd.DataFrame({"state": 4*(times >= .5)}, index=ids)
    policy = fit_policy(times, ids, embedding=embedding)
    explicit = fit_policy(times, ids, embedding=embedding, config=BinningConfig())
    assert policy.fingerprint == explicit.fingerprint
    assert policy.config == BinningConfig()
    assert policy.bandwidth_diagnostics["selection"] == "state"
    assert len(policy.stable_boundaries) == 1
    edge = policy.stable_boundaries[0][1]
    assert edge == pytest.approx(.5)
    np.testing.assert_array_equal(policy.bandwidth_diagnostics["chosen_multiplier"], 2.)
    plan = policy.initial_plan
    for i, center in enumerate(plan.centers):
        assert np.all(plan.raw_kernel[i, (times < edge) != (center < edge)] == 0)
    output = aggregate(plan, embedding.to_numpy(), cell_ids=ids)
    np.testing.assert_allclose(output[policy.centers < edge], 0, atol=1e-12)
    np.testing.assert_allclose(output[policy.centers >= edge], 4, atol=1e-8)
    query_times = [.25, edge, .75, 2.]
    query = build_plan(policy, query_times, ["q0", "q1", "q2", "q3"])
    np.testing.assert_array_equal(query.bandwidths, policy.bandwidths)
    assert query.bandwidth_diagnostics is policy.bandwidth_diagnostics
    assert query.policy_fingerprint == policy.fingerprint
    membership, valid = query_membership(policy, query_times, ["q0", "q1", "q2", "q3"])
    np.testing.assert_array_equal(valid, [True, True, True, False])
    assert np.all(membership[0, policy.centers >= edge] == 0)
    assert np.all(membership[1:3, policy.centers < edge] == 0)
    assert np.all(membership[3] == 0)
