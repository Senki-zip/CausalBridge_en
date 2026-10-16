"""Independent bounded numerical validation of standalone adaptive binning.

Run with the CausalBridge interpreter, from any directory::

    python -B scripts/validate_adaptive_binning.py \
        --output-dir /tmp/opencode/CausalBridge_default_combo_validation

This script never edits configuration or fits on heldout measurements. Results
describe synthetic noise/fidelity tradeoffs, not biological superiority.
"""
import os
os.environ.update({k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
import sys
sys.dont_write_bytecode = True
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CausalBridge.adaptive_binning import BinningConfig, aggregate, fit_policy, build_plan, query_membership

STEP_OFFSETS = {"step_045": .45, "step_0517": .517, "step_063": .63}
FEATURES = ["smooth", "pulse", *STEP_OFFSETS]
DESIGNS = ("uniform", "sparse_region")


def signals(times):
    return np.column_stack([2+.6*np.sin(2*np.pi*times),
                            2+.8*((times >= .45) & (times < .55)),
                            *[2+.6*(times >= offset) for offset in STEP_OFFSETS.values()]])


def timing(x, y, threshold, pulse):
    high = y >= threshold
    starts = np.where(high & np.r_[True, ~high[:-1]])[0]
    ends = np.where(high & np.r_[~high[1:], True])[0]
    if not len(starts):
        return "missing", None, None
    if len(starts) != 1:
        return "ambiguous", None, None
    a, b = int(starts[0]), int(ends[0])
    if a == 0 or (pulse and b == len(x)-1) or (not pulse and b != len(x)-1):
        return "ambiguous_boundary_or_nonpersistent", None, None
    onset = float(x[a-1] + (x[a]-x[a-1])*(threshold-y[a-1])/(y[a]-y[a-1]))
    end = float(x[b] + (x[b+1]-x[b])*(threshold-y[b])/(y[b+1]-y[b])) if pulse else None
    return "defined", onset, end


def repair_evidence():
    """Bounded metadata/embedding checks, independent of expression arms."""
    legacy = fit_policy(np.tile(np.linspace(0, 1, 10), 3), np.arange(30),
                        np.repeat(["a", "b", "c"], 10),
                        config=BinningConfig(mode="hard", branch_policy="legacy", bandwidth_selection="ess",
                                             boundary="none", max_bandwidth_multiplier=.5, max_bins=5))
    assert legacy.initial_plan.n_bins == 9
    q = build_plan(legacy, [.2, .8], ["x", "y"], ["a", "a"])
    reversed_q = build_plan(legacy, [.8, .2], ["y", "x"], ["a", "a"])
    assert q.policy_fingerprint == reversed_q.policy_fingerprint == legacy.fingerprint
    assert q.fingerprint != reversed_q.fingerprint
    cfg = BinningConfig(mode="adaptive_gaussian", branch_policy="separate", bandwidth_selection="ess",
                        max_bandwidth_multiplier=.5, max_bins=10,
                        boundary="split_noise", boundary_min_side=5, boundary_splits=40)
    t = np.r_[np.linspace(0, .49, 240), np.linspace(.51, 1, 60)]
    step = fit_policy(t, np.arange(len(t)), embedding=np.where(t < .5, 0., 10.)[:, None], config=cfg)
    assert len(step.stable_boundaries) == 1
    edge = step.stable_boundaries[0][1]
    assert abs(edge - .5) < 1e-12
    t = (np.arange(400) + .5) / 400
    emb = np.random.default_rng(321).normal(size=(400, 2))
    null = fit_policy(t, np.arange(400), embedding=emb, config=cfg)
    repeat = fit_policy(t, np.arange(400), embedding=emb, config=cfg)
    assert null.boundary_diagnostics == repeat.boundary_diagnostics
    varying = sum(bool(np.ptp(d["observed_samples"]) > 0) for d in null.boundary_diagnostics)
    assert varying == len(null.boundary_diagnostics)
    assert len(null.stable_boundaries) < len(null.boundary_diagnostics) / 2
    gap_times = np.r_[np.linspace(0, .1, 400), np.linspace(.9, 1, 400)]
    gap_ids = np.arange(len(gap_times))
    gap_embedding = np.r_[np.zeros(400), np.full(400, 4.)][:, None]
    gap = fit_policy(gap_times, gap_ids, embedding=gap_embedding,
                     config=BinningConfig(mode="adaptive_gaussian", branch_policy="separate",
                                          bandwidth_selection="state", boundary="none", max_bins=30, max_bandwidth_multiplier=2.))
    d = gap.bandwidth_diagnostics
    gap_records = []
    for i in (14, 15):
        assert np.array_equal(d["evidence"][i], [False, True, True, True])
        assert d["chosen_multiplier"][i] == 1 and d["score"][i, 3] > 1
        assert not d["state_confident"][i]
        gap_records.append(dict(center=float(gap.centers[i]), chosen_multiplier=float(d["chosen_multiplier"][i]),
                                evidence=d["evidence"][i].tolist(), widest_score=float(d["score"][i, 3]),
                                chosen_score=float(d["score"][i, 2]), reason=str(d["reason"][i]),
                                chosen_coverage=float(d["geometric_coverage"][i, 2]), state_confident=False))
    assert build_plan(gap, gap_times, gap_ids).fingerprint == gap.initial_plan.fingerprint
    default_times = np.linspace(0, 1, 2400)
    default_ids = np.arange(len(default_times))
    default_policy = fit_policy(default_times, default_ids, embedding=4*(default_times[:, None] >= .5))
    assert default_policy.config == BinningConfig()
    assert len(default_policy.stable_boundaries) == 1
    assert abs(default_policy.stable_boundaries[0][1]-.5) < 1e-12
    raw = default_policy.initial_plan.raw_kernel
    edge = default_policy.stable_boundaries[0][1]
    cross_mask = (default_policy.centers[:, None] < edge) != (default_times[None, :] < edge)
    assert np.all(raw[cross_mask] == 0)
    assert default_policy.bandwidth_diagnostics["selection"] == "state"
    assert build_plan(default_policy, default_times, default_ids).fingerprint == default_policy.initial_plan.fingerprint
    try:
        fit_policy(default_times, default_ids)
    except ValueError as exc:
        assert "external aligned training WT embedding" in str(exc)
    else:
        raise AssertionError("default silently accepted missing training embedding")
    return dict(default_combo=dict(config=asdict(default_policy.config), n_bins=default_policy.initial_plan.n_bins,
                                   detected_step_edge=edge, cross_boundary_nonzeros=int(np.count_nonzero(raw[cross_mask])),
                                   frozen=True, missing_embedding_rejected=True),
                legacy_requested_bins=5, legacy_actual_bins=legacy.initial_plan.n_bins,
                query_order_changes_plan_hash=True, shared_policy_hash=True,
                uneven_density_step_edge=edge, step_edge_absolute_error=abs(edge-.5),
                null_stable_boundaries=len(null.stable_boundaries), null_tested_boundaries=len(null.boundary_diagnostics),
                equal_count_varying_resamples=varying, resampling_reproducible=True,
                reviewed_sparse_gap=gap_records)


def candidate_records(policy, design, arm):
    d = policy.bandwidth_diagnostics
    if d["selection"] != "state":
        return []
    records = []
    for i, center in enumerate(policy.centers):
        for j, multiplier in enumerate(d["candidates"]):
            records.append(dict(design=design, arm=arm, bin=i, center=float(center), multiplier=float(multiplier),
                                **{key: float(d[key][i, j]) for key in ("D", "N", "U", "score", "geometric_coverage")},
                                accepted=bool(d["accepted"][i, j]), evidence=bool(d["evidence"][i, j]),
                                chosen_multiplier=float(d["chosen_multiplier"][i]), reason=str(d["reason"][i]),
                                state_confident=bool(d["state_confident"][i]), supported_pilots=int(d["supported_pilot_count"][i]),
                                ESS=float(policy.initial_plan.diagnostics["ess"][i]),
                                global_scale_squared=d["global_scale_squared"], floor_squared=d["floor_squared"]))
    return records


def mechanism_validation(out, seeds):
    """Predeclared fixed-density mechanisms; same standardized WT noise paired across levels."""
    times = (np.arange(1600)+.5)/1600
    ids = np.arange(len(times))
    noise = np.random.default_rng(17).normal(size=(len(times), 1))
    cfg = BinningConfig(mode="adaptive_gaussian", branch_policy="separate", bandwidth_selection="state",
                        boundary="none", max_bins=10, max_bandwidth_multiplier=2.)
    functions = {"flat": lambda t: np.zeros(len(t)), "drift": lambda t: t,
                 **{name: (lambda t, offset=offset: 4*(t >= offset)) for name, offset in STEP_OFFSETS.items()}}
    summaries, candidates = [], []
    for name, function in functions.items():
        for sigma in (0., .1, .5):
            mu = function(times)[:, None]
            # All policies fit before any heldout measurements are drawn.
            p = fit_policy(times, ids, embedding=mu+sigma*noise, config=cfg)
            d = p.bandwidth_diagnostics
            candidates.extend(candidate_records(p, f"mechanism_{name}_sd{sigma}", "adaptive_state_cap2"))
            latent = aggregate(p.initial_plan, mu, cell_ids=ids)
            center_truth = function(p.centers)[:, None]
            noise_errors, fidelity_errors = [], []
            for seed in seeds:
                measured = mu + sigma*np.random.default_rng(seed).normal(size=mu.shape)
                y = aggregate(p.initial_plan, measured, cell_ids=ids)
                noise_errors.append(float(np.sqrt(np.mean((y-latent)**2))))
                fidelity_errors.append(float(np.sqrt(np.mean((y-center_truth)**2))))
            summaries.append(dict(mechanism=name, sigma=sigma, chosen_multipliers=d["chosen_multiplier"].tolist(),
                                  multiplier_mean=float(d["chosen_multiplier"].mean()),
                                  multiplier_median=float(np.median(d["chosen_multiplier"])),
                                  ESS_median=float(np.median(p.initial_plan.diagnostics["ess"])),
                                  minimum_coverage=float(d["geometric_coverage"].min()),
                                  reasons={str(k): int(v) for k, v in zip(*np.unique(d["reason"], return_counts=True))},
                                  noise_RMSE_mean=float(np.mean(noise_errors)), measured_temporal_RMSE_mean=float(np.mean(fidelity_errors)),
                                  latent_temporal_RMSE=float(np.sqrt(np.mean((latent-center_truth)**2)))))
    (out / "mechanisms.json").write_text(json.dumps(summaries, indent=2))
    return summaries, candidates


def validate(out):
    started = time.monotonic()
    seeds = [11, 22, 33]
    configs = {
        "hard": BinningConfig(mode="hard", branch_policy="separate", bandwidth_selection="ess", boundary="none",
                              max_bandwidth_multiplier=.5, max_bins=30, min_cells_per_bin=30),
        "fixed_gaussian": BinningConfig(mode="gaussian", branch_policy="separate", bandwidth_selection="ess", boundary="none",
                                        max_bandwidth_multiplier=.5, max_bins=30, min_cells_per_bin=30),
        "adaptive_ess_cap05": BinningConfig(mode="adaptive_gaussian", branch_policy="separate", bandwidth_selection="ess", boundary="none",
                                            max_bandwidth_multiplier=.5, max_bins=30, min_cells_per_bin=30),
        # Deliberate support sensitivity, not a matched-ESS claim: target40 with
        # allowed1/2 may use larger h than conservative default. Fixed Gaussian
        # is exactly the h=1 center-spacing candidate when no sigma inflation.
        "adaptive_ess_cap2": BinningConfig(mode="adaptive_gaussian", branch_policy="separate", bandwidth_selection="ess", boundary="none",
                                           max_bins=30, min_cells_per_bin=30, max_bandwidth_multiplier=2.),
        "adaptive_state_cap2": BinningConfig(mode="adaptive_gaussian", branch_policy="separate", max_bins=30, min_cells_per_bin=30,
                                             bandwidth_selection="state", boundary="none", max_bandwidth_multiplier=2.),
        "adaptive_state_boundary_default_cap2": BinningConfig(max_bins=30, min_cells_per_bin=30),
    }
    source_paths = [ROOT / "CausalBridge" / f"{name}.py" for name in ("adaptive_binning", "kinetics", "granger", "perturbation")]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    manifest = dict(seeds=seeds, configs={name: asdict(c) for name, c in configs.items()},
                    features=FEATURES, step_offsets=STEP_OFFSETS, sigma=.3,
                    measurement="independent bounded uniform noise sd.3, no dropout or clipping",
                    designs={"uniform": "900 cells, (i+.5)/900", "sparse_region": "400 cells before.43, 30 between.43-.57, 400 after.57; gap region sparse"},
                    primary_default_arm="adaptive_state_boundary_default_cap2; module defaults except benchmark max_bins30/min_cells30; standalone max_bins remains100",
                    policy_training="State: independent TRAIN WT realization of represented features + uniform noise sd.3, seed991. Others metadata only. All policies frozen before heldout draw. Boundaries enabled only for explicitly labelled default combo arm; no PCA or heldout tuning.",
                    mechanism_training="1600 fixed-time cells; flat, drift, steps at .45/.517/.63; paired standardized train noise seed17 at sd0/.1/.5; default state budget1/floor.2/uncertainty2, candidates .25/.5/1/2. Heldout seeds11/22/33 independent.",
                    caveats="State-cap2 is experimental and loses pulse fidelity in this experiment. State scores are no demonstrated excess distortion heuristic, not confidence guarantees. Calculable evidence does not assure local pilot coverage. Embedding represents all synthetic features, not necessarily biological expression. Full kernel tails and finite pilots limit resolution. Density affects pilot availability, estimation uncertainty, and reliability, not ESS-driven widening.",
                    sensitivity="Explicit ESS cap.5 and cap2 plus state-only cap2 are preserved baselines; standalone default is combined state+boundary cap2. Support is NOT matched to fixedGaussian; observedESS and nearestfixed ESS gaps reported. No best-arm selection.",
                    output_dir=str(out), source_sha256=hashes, scope="noise vs latent/temporal fidelity; no default adaptive beats-all or significance claims")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    metrics, predictions, checks, errors, candidate_rows, boundary_rows = [], [], [], [], [], []
    mechanisms = []
    repairs = None
    try:
        repairs = repair_evidence()
        mechanisms, candidate_rows = mechanism_validation(out, seeds)
        for design in DESIGNS:
            if design == "uniform":
                times = (np.arange(900)+.5)/900
            else:
                times = np.r_[np.linspace(.0005, .4295, 400), np.linspace(.431, .569, 30), np.linspace(.5705, .9995, 400)]
            ids = np.array([f"{design}_{i}" for i in range(len(times))])
            mu = signals(times)
            train_embedding = mu + np.random.default_rng(991).uniform(-np.sqrt(3)*.3, np.sqrt(3)*.3, mu.shape)
            policies = {name: fit_policy(times, ids, embedding=train_embedding if c.bandwidth_selection == "state" else None, config=c)
                        for name, c in configs.items()}
            for arm, policy in policies.items():
                candidate_rows.extend(candidate_records(policy, design, arm))
                boundary_rows.extend(dict(design=design, arm=arm, **item) for item in policy.boundary_diagnostics)
            fixedess = policies["fixed_gaussian"].initial_plan.diagnostics["ess"]
            for seed in seeds:
                rng = np.random.default_rng(seed)
                measured = mu + rng.uniform(-np.sqrt(3)*.3, np.sqrt(3)*.3, mu.shape)
                assert (measured > 0).all() and np.isfinite(measured).all()
                for arm, policy in policies.items():
                    plan = policy.initial_plan
                    y = aggregate(plan, measured, cell_ids=ids)
                    latent = aggregate(plan, mu, cell_ids=ids)
                    reference = signals(plan.centers)
                    frozen = build_plan(policy, times, ids)
                    assert np.array_equal(frozen.centers, policy.centers)
                    assert np.array_equal(frozen.bandwidths, policy.bandwidths)
                    assert np.allclose(aggregate(frozen, measured, cell_ids=ids), y, equal_nan=True)
                    membership, valid = query_membership(policy, [-1, .5, 2], ["left", "inside", "right"])
                    assert not valid[0] and not valid[-1]
                    assert np.isfinite(y[plan.valid]).all()
                    assert (plan.diagnostics["cross_branch_mass"] == 0).all()
                    checks.append(dict(seed=seed, design=design, arm=arm, passed=True))
                    for j, feature in enumerate(FEATURES):
                        good = plan.valid
                        noise_mse = float(np.mean((y[good, j]-latent[good, j])**2)) if good.any() else None
                        raw_mse = float(np.mean((measured[:, j]-mu[:, j])**2))
                        record = dict(seed=seed, design=design, arm=arm, feature=feature,
                                      noise_RMSE=float(np.sqrt(noise_mse)) if noise_mse is not None else None,
                                      noise_MSE_reduction=1-noise_mse/raw_mse if noise_mse is not None else None,
                                      latent_temporal_RMSE=float(np.sqrt(np.mean((latent[good, j]-reference[good, j])**2))) if good.any() else None,
                                      measured_temporal_RMSE=float(np.sqrt(np.mean((y[good, j]-reference[good, j])**2))) if good.any() else None,
                                      ESS_median=float(np.median(plan.diagnostics["ess"])), ESS_min=float(np.min(plan.diagnostics["ess"])),
                                      positive_support_min=int(np.min(plan.diagnostics["support_cells"])), valid_bins=int(good.sum()), reliable_bins=int(plan.reliable.sum()),
                                      adjacent_overlap_mean=float(np.mean(plan.diagnostics["neighboring_overlap"])),
                                      ESS_gap_vs_fixed_median=float(np.median(plan.diagnostics["ess"]-fixedess)),
                                      cap_sensitivity_not_matched_ESS=True)
                        if feature == "pulse" or feature in STEP_OFFSETS:
                            amp = .8 if feature == "pulse" else .6
                            for name, values in (("latent", latent[:, j]), ("measured", y[:, j])):
                                if not good.all():
                                    status, onset, end = "invalid_bins", None, None
                                else:
                                    status, onset, end = timing(plan.centers, values, 2+amp/2, feature == "pulse")
                                record[name+"_timing_status"] = status
                                record[name+"_onset_error"] = abs(onset-(.45 if feature == "pulse" else STEP_OFFSETS[feature])) if onset is not None else None
                                record[name+"_end_error"] = abs(end-.55) if end is not None else None
                            if feature == "pulse":
                                record["latent_pulse_retention"] = float((np.max(latent[good, j])-2)/.8) if good.any() else None
                                record["measured_pulse_retention"] = float((np.max(y[good, j])-2)/.8) if good.any() else None
                        metrics.append(record)
                        for i in range(plan.n_bins):
                            predictions.append(dict(seed=seed, design=design, arm=arm, feature=feature, bin=i, center=float(plan.centers[i]),
                                                    measured=float(y[i, j]), latentaggregate=float(latent[i, j]), temporal_truth=float(reference[i, j]),
                                                    ESS=float(plan.diagnostics["ess"][i]), h=float(plan.bandwidths[i]), support=int(plan.diagnostics["support_cells"][i]), valid=bool(plan.valid[i]), reliable=bool(plan.reliable[i])))
            for arm, policy in policies.items():
                np.savez(out / f"{design}_{arm}_weights.npz", centers=policy.centers, h=policy.bandwidths,
                         ess=policy.initial_plan.diagnostics["ess"], positive_support=policy.initial_plan.diagnostics["support_cells"],
                         valid=policy.initial_plan.valid, reliable=policy.initial_plan.reliable)
                if policy.config.bandwidth_selection == "state":
                    d = policy.bandwidth_diagnostics
                    np.savez(out / f"{design}_{arm}_training_diagnostics.npz",
                             **{key: value for key, value in d.items() if isinstance(value, np.ndarray)})
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)] for p in source_paths)
        assert len(metrics) == len(DESIGNS)*len(configs)*len(seeds)*len(FEATURES)
        assert time.monotonic()-started < 120
    except Exception:
        errors.append(traceback.format_exc())
    df = pd.DataFrame(metrics) if metrics else pd.DataFrame(columns=["design", "arm", "feature", "seed",
                                                                    "measured_timing_status", "latent_pulse_retention"])
    df.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "perbin_predictions.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(out / "state_candidate_diagnostics.csv", index=False)
    pd.DataFrame(boundary_rows).to_csv(out / "boundary_diagnostics.csv", index=False)
    elapsed = time.monotonic()-started
    state_pulse = df[(df.arm == "adaptive_state_cap2") & (df.feature == "pulse")]
    fidelity_failures = state_pulse[state_pulse.measured_timing_status != "defined"][
        ["design", "seed", "arm", "feature", "measured_timing_status", "latent_pulse_retention"]].to_dict(orient="records")
    pulse_detection_passed = len(state_pulse) == len(DESIGNS)*len(seeds) and not fidelity_failures
    combo_pulse = df[(df.arm == "adaptive_state_boundary_default_cap2") & (df.feature == "pulse")]
    combo_failures = combo_pulse[combo_pulse.measured_timing_status != "defined"][
        ["design", "seed", "arm", "feature", "measured_timing_status", "latent_pulse_retention"]].to_dict(orient="records")
    combo_detection_passed = len(combo_pulse) == len(DESIGNS)*len(seeds) and not combo_failures
    (out / "validation.json").write_text(json.dumps(dict(passed=not errors, invariants_passed=not errors,
                                                       passed_meaning="implementation invariants only; not temporal fidelity or superiority",
                                                       state_pulse_detection_passed=pulse_detection_passed,
                                                       default_combo_pulse_detection_passed=combo_detection_passed,
                                                       default_combo_fidelity_failures=combo_failures,
                                                       fidelity_failures=fidelity_failures, elapsed_seconds=elapsed, checks=checks,
                                                       repair_evidence=repairs, errors=errors, metric_rows=len(df)), indent=2))
    def fmt(series):
        s = series.dropna()
        return "undefined" if not len(s) else f"{s.median():.5g} [{s.min():.5g},{s.max():.5g}]"
    lines = ["# Standalone default state + boundary: independent synthetic validation",
             f"Elapsed {elapsed:.3f}s, implementation invariants passed {not errors}. Three paired seeds; no significance claims.",
             f"Explicit state-only cap2 pulse detection passed {pulse_detection_passed}: {len(fidelity_failures)} missing/ambiguous heldout pulse detections. Invariants passing does NOT mean fidelity passing.",
             f"Primary combined-default arm pulse detection passed {combo_detection_passed}: {len(combo_failures)} missing/ambiguous detections. This limited detection check is not a general fidelity guarantee.",
             "State policies fit independent TRAIN WT external embedding; other policies fit metadata only. All frozen before heldout measurements; no PCA or heldout tuning. Boundaries enabled only for the labelled combined-default arm. Full kernels except explicit masks; no tail truncation.",
             manifest["caveats"],
             "Fixed Gaussian and adaptive are NOT ESS-matched: cap sensitivity is explicitly exploratory; ESS/overlap/support and ESS gaps are reported. The explicit conservative ESS baseline can fail targetESS without invalidating an otherwise supported estimate. Combined state+boundary defaults are standalone only, not pipeline integration. This experiment does not claim adaptive beats all.",
             "|Design|Arm|Feature|NoiseRMSE|Latent temporalRMSE|Measured temporalRMSE|ESS median|Reliable bins|",
             "|---|---|---|---|---|---|---|---|"]
    for key, group in df.groupby(["design", "arm", "feature"]):
        lines.append("|"+"|".join(list(key)+[fmt(group[n]) for n in ("noise_RMSE", "latent_temporal_RMSE", "measured_temporal_RMSE", "ESS_median", "reliable_bins")])+"|")
    lines += ["## Pulse fidelity and timing", "|Design|Arm|Latent retention|Measured retention|Latent onset error|Measured timing statuses|", "|---|---|---|---|---|---|"]
    for key, group in df[df.feature == "pulse"].groupby(["design", "arm"]):
        lines.append("|"+"|".join(list(key)+[fmt(group[n]) for n in ("latent_pulse_retention", "measured_pulse_retention", "latent_onset_error")]+[str(group.measured_timing_status.value_counts().to_dict())])+"|")
    lines += ["Raw per-bin positive-support/ESS/h/valid/reliable and predicted trajectories are saved. Timing missing/ambiguous/invalid statuses are not zero-imputed. Shifted-step diagnostics are in metrics.csv. Combined-default boundary diagnostics are in boundary_diagnostics.csv; parameters were not selected on heldout data."]
    lines += ["## Predeclared fixed-density mechanism checks", "|Mechanism|Noise sd|Mean multiplier|Noise RMSE|Latent temporal RMSE|", "|---|---|---|---|---|"]
    for row in mechanisms:
        lines.append(f"|{row['mechanism']}|{row['sigma']}|{row['multiplier_mean']:.4g}|{row['noise_RMSE_mean']:.4g}|{row['latent_temporal_RMSE']:.4g}|")
    lines += ["## Counterexamples (no heldout budget tuning)",
              "The represented multifeature embedding can accept wide state bandwidths even for a short pulse. Read the pulse retention/timing table above: noise reduction is not temporal fidelity, and missing pulses are failures, not zero timing error.",
              "Fixed-density drift widens under larger per-cell noise but incurs more edge bias. A step lying near an output center can remain badly mixed even at the minimum candidate; state_budget_unmet_at_min is explicitly recorded in mechanisms.json and candidate diagnostics.",
              "Global S² pools branch-centered pilot variation, so embedding feature scaling/representation influences the floor. Gaussian tails are never truncated; pilot regression cannot resolve sub-pilot transitions universally."]
    lines += errors or ["Errors: none. Source hashes unchanged."]
    (out / "report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(passed=not errors, invariants_passed=not errors, state_pulse_detection_passed=pulse_detection_passed,
                         default_combo_pulse_detection_passed=combo_detection_passed,
                         default_combo_pulse_detection_failures=len(combo_failures),
                         pulse_detection_failures=len(fidelity_failures), elapsed_seconds=elapsed, metrics=len(df),
                         repair_evidence=repairs, output=str(out), errors=errors)))
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    expected = Path("/tmp/opencode/CausalBridge_default_combo_validation")
    if args.output_dir.absolute() != expected:
        parser.error(f"this bounded validator writes only to {expected}")
    if not expected.parent.is_dir():
        parser.error("approved output parent does not exist")
    if expected.exists():
        parser.error("refusing to overwrite existing default-combo validation artifacts")
    expected.mkdir()
    return validate(expected)


if __name__ == "__main__":
    raise SystemExit(main())
