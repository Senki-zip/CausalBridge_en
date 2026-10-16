"""Real spawn parity, runtime contracts, bounded submission and failure cleanup."""
import copy
import multiprocessing
import pickle

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from CausalBridge import atac_normalization as norm, binning_adapter, run
from CausalBridge import saturating_hybrid as core, saturating_transfer as transfer


def parallel_inputs(continuous=False):
    from test_saturating_transfer import fixture_data
    rna, atac, pt, _, _, cfg = fixture_data(64, 2)
    # Four targets in each layer, distinct lagged keys, two WT branches.
    rna = ad.AnnData(sparse.hstack([rna.X, rna.X[:, :1]*.8], format="csr"),
        obs=rna.obs.copy(), var=pd.DataFrame(index=["A", "B", "C", "D"]))
    atac = ad.AnnData(sparse.hstack([atac.X, atac.X*.7], format="csr"),
        obs=atac.obs.copy(), var=pd.DataFrame(index=["p0", "p1", "p2", "p3"]))
    if continuous:
        values = np.rint(atac.X.toarray()*10)
        raw = ad.AnnData(sparse.csr_matrix(np.column_stack([values, np.full(atac.n_obs, 100)])),
            obs=atac.obs.copy(), var=pd.DataFrame(index=[*atac.var_names, "background"]))
        prepared = norm.prepare_atac_counts(raw)
        qc = norm.finalize_atac_qc(prepared[:, :4].copy(), prepared)
        cfg["preprocess"] = {"atac": {"binarize": True}, "atac_depth_normalization": {
            "enabled": True, "target_sum": 10000., "output_cap_expansion": .25}}
        atac = norm.depth_log1p_model_view(qc, norm.validate_atac_depth_config(cfg))
    edges = pd.DataFrame([(gene, peak) for gene in rna.var_names for peak in atac.var_names[:2]],
        columns=["gene", "peak_id"])
    screen = pd.DataFrame([(gene, peak, lag) for peak in atac.var_names
        for gene, lag in (("A", 0), ("B", 1))], columns=["tf_gene", "peak_id", "tf_lag"])
    return rna, atac, pt, edges, screen, cfg


@pytest.mark.parametrize("policy", ["exclude", "best_finite"])
@pytest.mark.parametrize("continuous", [False, True])
def test_real_spawn_both_layers_exact_registry_parity_and_bounded_jobs(continuous, policy, monkeypatch, caplog):
    rna, atac, pt, edges, screen, cfg = parallel_inputs(continuous)
    cfg["atac_to_rna"]["saturating_hybrid"] = {"budget_exhausted_policy": policy, "max_nfev": 1}
    serial_table, serial_metrics, serial = transfer.fit_peak_gene(rna, atac, pt, edges, cfg)
    serial_tf = transfer.fit_tf_peak(rna, atac, pt, screen, cfg)
    changed = copy.deepcopy(cfg); changed["atac_to_rna"]["saturating_hybrid"]["workers"] = 4
    payloads = []
    windows = []
    actual_pool = transfer.ProcessPoolExecutor
    actual_wait = transfer.wait
    class RecordingPool(actual_pool):
        def submit(self, function, job):
            payloads.append(pickle.loads(pickle.dumps(job)))
            return super().submit(function, job)
    def record_wait(futures, **kwargs):
        windows.append(len(futures))
        return actual_wait(futures, **kwargs)
    monkeypatch.setattr(transfer, "ProcessPoolExecutor", RecordingPool)
    monkeypatch.setattr(transfer, "wait", record_wait)
    with caplog.at_level("INFO", logger=transfer.__name__):
        table, metrics, parallel = transfer.fit_peak_gene(rna, atac, pt, edges, changed)
        parallel_tf = transfer.fit_tf_peak(rna, atac, pt, screen, changed)
    pd.testing.assert_frame_equal(table, serial_table, check_exact=True)
    pd.testing.assert_frame_equal(metrics, serial_metrics, check_exact=True)
    assert parallel == serial and parallel_tf == serial_tf
    assert parallel["config"]["workers"] == 1
    assert list(parallel["targets"]) == sorted(parallel["targets"])
    assert list(parallel_tf["targets"]) == sorted(parallel_tf["targets"])
    assert windows and max(windows) <= 8 and len(payloads) == 8
    assert all(set(job) == {"data", "bounds"} for job in payloads)
    assert all("cell_weights" not in job["data"] and "snapshot" not in job for job in payloads)
    assert all(job["data"]["X"].shape[1] == 2 for job in payloads)
    assert "workers=4" in caplog.text and "completed=4/4" in caplog.text
    assert "registry validated" in caplog.text
    for registry in (parallel, parallel_tf):
        transfer.validate_registry(registry)
        assert registry.get("budget_exhausted_policy", "exclude") == policy
        assert all(b["provenance"].get("budget_exhausted_policy", "exclude") == policy
                   for b in registry["targets"].values())
    assert not multiprocessing.active_children()


def test_core_direct_runtime_worker_count_is_canonical_and_legacy_codec_unchanged():
    from test_saturating_hybrid import data
    d = data()
    one = core.fit_target(d, (0., 1.), core.HybridConfig())
    four = core.fit_target(d, (0., 1.), core.HybridConfig(workers=4))
    assert four.to_bundle() == one.to_bundle() and four.config.workers == 1
    from test_atac_normalization import test_frozen_pre_continuous_v1_bundle_exact_roundtrip
    test_frozen_pre_continuous_v1_bundle_exact_roundtrip()


def test_accepted_group_real_spawn_roundtrip():
    from test_saturating_perturbation import grouped_fit
    rna, atac, pt, edges, screen, cfg, *_ = grouped_fit.__wrapped__()
    # Two distinct peak targets ensure an actual parallel pool.
    atac = ad.AnnData(sparse.hstack([atac.X, atac.X], format="csr"), obs=atac.obs.copy(),
        var=pd.DataFrame(index=["p0", "p1"]))
    screen = pd.concat([screen, screen.assign(peak_id="p1")], ignore_index=True)
    serial = transfer.fit_tf_peak(rna, atac, pt, screen, cfg)
    parallel_cfg = copy.deepcopy(cfg); parallel_cfg["atac_to_rna"]["saturating_hybrid"]["workers"] = 4
    parallel = transfer.fit_tf_peak(rna, atac, pt, screen, parallel_cfg)
    assert parallel == serial
    assert all(bundle["groups"] for bundle in parallel["targets"].values())
    transfer.validate_registry(parallel)


def test_worker_failure_cleans_up_without_returning_partial_registry():
    rna, atac, pt, edges, _, cfg = parallel_inputs()
    numeric = transfer.validate_saturating_config(cfg)
    snapshot = transfer._snapshot(rna, atac, pt, cfg)
    gi = {v: i for i, v in enumerate(rna.var_names)}
    ai = {v: i for i, v in enumerate(atac.var_names)}
    inputs = {g: {(p, 1) for p in atac.var_names[:2]} for g in rna.var_names}
    jobs = list(transfer._target_jobs(transfer.GENE_TYPE, snapshot, inputs, gi, ai, numeric))
    jobs[1][2]["data"]["y"] = np.full_like(jobs[1][2]["data"]["y"], np.nan)
    before = {p.pid for p in multiprocessing.active_children()}
    with pytest.raises(RuntimeError, match="worker failed for B; no registry produced.*finite"):
        transfer._execute_target_jobs(iter(jobs), snapshot["support"], numeric, 2, 4, 8, transfer.GENE_TYPE)
    assert {p.pid for p in multiprocessing.active_children()} == before
    # A fresh invocation succeeds after failure: no leaked worker state/pool.
    changed = copy.deepcopy(cfg); changed["atac_to_rna"]["saturating_hybrid"] = {"workers": 2}
    assert len(transfer.fit_peak_gene(rna, atac, pt, edges, changed)[2]["targets"]) == 4


@pytest.mark.parametrize("continuous", [False, True])
def test_worker_count_does_not_change_pipeline_or_binning_contracts(continuous):
    *_, cfg = parallel_inputs(continuous)
    cfg["atac_to_rna"]["saturating_hybrid"] = {"workers": 1}
    signature = run._step5_model_config(cfg)
    contract = binning_adapter._model_config(cfg)
    source = run._stage_numerical_signatures("step5_atac_to_rna", "saturating_hybrid")
    cfg["atac_to_rna"]["saturating_hybrid"]["workers"] = 4
    assert run._step5_model_config(cfg) == signature
    assert binning_adapter._model_config(cfg) == contract
    assert run._stage_numerical_signatures("step5_atac_to_rna", "saturating_hybrid") == source
    assert transfer.validate_saturating_config(cfg).workers == 1


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "4"])
def test_invalid_runtime_workers_rejected(workers):
    with pytest.raises(ValueError, match="workers"):
        core.HybridConfig(workers=workers)


def test_budget_policy_semantic_contract_and_early_rejection(monkeypatch):
    from types import SimpleNamespace
    rna, atac, pt, edges, _, cfg = parallel_inputs()
    context = SimpleNamespace(
        plan=SimpleNamespace(fingerprint="plan", cell_ids=np.asarray(rna.obs_names),
            times=np.linspace(0., 1., rna.n_obs), branches=np.array(["b"] * rna.n_obs),
            centers=np.array([.5])),
        policy=SimpleNamespace(fingerprint="policy", branch_boundaries=[]),
        metadata={}, model_contract_fingerprint="adapter")
    monkeypatch.setattr(run, "_resource_identity", lambda *args: {})
    def stages():
        return run._adaptive_checkpoint_metadata(context, cfg, ["A"], {"A": "TF"},
            rna=rna, atac=atac, n_peaks=atac.n_vars)["stages"]
    old_stages = stages()
    old = run._step5_model_config(cfg)
    old_binning = binning_adapter._model_config(cfg)
    cfg["atac_to_rna"]["saturating_hybrid"] = {"budget_exhausted_policy": "best_finite"}
    current = run._step5_model_config(cfg)
    new_stages = stages()
    assert new_stages["step4_granger"] == old_stages["step4_granger"]
    for stage in ("step5_atac_to_rna", "step6_rna_to_atac"):
        assert new_stages[stage] != old_stages[stage]
    assert current != old and binning_adapter._model_config(cfg) != old_binning
    assert "budget_exhausted_policy" not in transfer.validate_saturating_config(cfg).to_dict()
    cfg["atac_to_rna"]["saturating_hybrid"]["workers"] = 4
    assert run._step5_model_config(cfg) == current
    assert stages() == new_stages
    for budget in (500, 1500, 3000):
        cfg["atac_to_rna"]["saturating_hybrid"]["max_nfev"] = budget
        assert transfer.validate_saturating_config(cfg).max_nfev == budget
    def forbidden(*args, **kwargs):
        pytest.fail("invalid policy must fail before snapshot/data loading")
    monkeypatch.setattr(transfer, "_snapshot", forbidden)
    cfg["atac_to_rna"]["saturating_hybrid"]["budget_exhausted_policy"] = "unknown"
    with pytest.raises(ValueError, match="budget_exhausted_policy"):
        transfer.fit_peak_gene(rna, atac, pt, edges, cfg)


def test_empty_and_single_target_do_not_start_pool(monkeypatch):
    rna, atac, pt, edges, screen, cfg = parallel_inputs()
    cfg["atac_to_rna"]["saturating_hybrid"] = {"workers": 4}
    def forbidden(*args, **kwargs):
        pytest.fail("empty/single-target fit must not start a process pool")
    monkeypatch.setattr(transfer, "ProcessPoolExecutor", forbidden)
    assert not transfer.fit_peak_gene(rna, atac, pt, edges.iloc[:0], cfg)[2]["targets"]
    assert len(transfer.fit_tf_peak(rna, atac, pt, screen.query("peak_id == 'p0'"), cfg)["targets"]) == 1


def test_real_pool_rolling_refill_and_initializer_failure(monkeypatch):
    from types import SimpleNamespace
    rna, atac, pt, _, _, cfg = parallel_inputs()
    numeric = transfer.validate_saturating_config(cfg)
    snapshot = transfer._snapshot(rna, atac, pt, cfg)
    jobs = []
    for i in range(20):
        rows = np.arange(20, dtype=np.int64)
        job = dict(data=dict(target_id=f"t{i:02}", input_keys=(("p0", 0),),
            X=np.ones((20, 1)), y=np.ones(20)*.3, source_bins=rows[:, None],
            target_rows=rows, branch_ids=("0",)*20, input_unit="raw_atac", output_unit="raw_atac",
            provenance={}, support_rows=rows), bounds=(0., 1.))
        jobs.append((f"t{i:02}", 1, job, None))
    windows = []
    actual_wait = transfer.wait
    def tracked(futures, **kwargs):
        windows.append(len(futures))
        return actual_wait(futures, **kwargs)
    monkeypatch.setattr(transfer, "wait", tracked)
    result = transfer._execute_target_jobs(iter(jobs), snapshot["support"], numeric, 2, 20, 20, transfer.TF_TYPE)
    assert len(result) == 20 and list(result) == sorted(result)
    assert len(windows) > 1 and max(windows) == 4
    bad_support = SimpleNamespace(weights=snapshot["support"].weights, fingerprint="0"*64)
    with pytest.raises(RuntimeError, match="worker failed.*no registry produced"):
        transfer._execute_target_jobs(iter(jobs), bad_support, numeric, 2, 20, 20, transfer.TF_TYPE)
    assert not multiprocessing.active_children()
