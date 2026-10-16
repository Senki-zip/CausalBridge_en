"""Bounded synthetic strategy comparison; no real data or pipeline checkpoints."""
import os
os.environ.update({key: "1" for key in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from CausalBridge.adaptive_binning import BinningConfig, fit_policy, query_membership, aggregate


def compare():
    rng = np.random.default_rng(42)
    times = (np.arange(600) + .5) / 600
    ids = np.arange(len(times))
    query_times = (np.arange(600) + .25) / 600
    latent = (query_times >= .5).astype(float)
    # State and expression noise are independent; heldout truth never fits a policy.
    embedding = (4 * (times >= .5) + rng.normal(0, .2, len(times)))[:, None]
    measured = (times >= .5).astype(float) + rng.normal(0, .3, len(times))
    records = {}
    for name, selection, boundary in (
        ("ess_boundary", "ess", "split_noise"),
        ("ess_only", "ess", "none"),
        ("state_boundary", "state", "split_noise"),
    ):
        cfg = BinningConfig(max_bins=20, min_cells_per_bin=10,
                            bandwidth_selection=selection, boundary=boundary)
        policy = fit_policy(times, ids, embedding=embedding if boundary != "none" else None,
                            config=cfg)
        plan = policy.initial_plan
        estimates = aggregate(plan, measured[:, None], cell_ids=ids)[:, 0]
        membership, query_valid = query_membership(policy, query_times, ids + 1000)
        supported = query_valid & (membership[:, plan.valid].sum(axis=1) > 0)
        reconstructed = membership[supported][:, plan.valid] @ estimates[plan.valid]
        cross_mask = (plan.centers[:, None] < .5) != (times[None, :] < .5)
        raw = plan.raw_kernel
        records[name] = dict(
            config=asdict(cfg), n_bins=plan.n_bins,
            stable_boundaries=list(policy.stable_boundaries),
            valid_bins=int(plan.valid.sum()), reliable_bins=int(plan.reliable.sum()),
            supported_queries=int(supported.sum()),
            heldout_rmse=float(np.sqrt(np.mean((reconstructed - latent[supported]) ** 2))),
            median_ess=float(np.median(plan.diagnostics["ess"])),
            median_bandwidth=float(np.median(plan.bandwidths)),
            cross_step_kernel_fraction=float(raw[cross_mask].sum() / raw.sum()),
        )
    assert records["ess_boundary"]["stable_boundaries"]
    assert records["ess_boundary"]["cross_step_kernel_fraction"] == 0
    assert records["ess_only"]["cross_step_kernel_fraction"] > 0
    return dict(seed=42, train_cells=600, heldout_cells=600,
                caveat="Single synthetic step; unequal ESS, no biological superiority claim.",
                strategies=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "comparison.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f"Saved {path}")
