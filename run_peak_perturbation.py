#!/usr/bin/env python3
"""CausalBridge peak KO quick-run CLI: reuse an existing checkpoint and change peak/strength/depth in seconds.

Usage:
    python run_peak_perturbation.py --config config.yaml \
        --peak chr3:123400-123900 --strength -1.0 --depth 1 \
        [--state all] [--match exact] [--out-dir results/test/MY_PEAK]

Principle: derive a peak-mode config from the base config (knockout_type=peak,
the peak_ko section, an isolated output.dir, and checkpoint reuse from the
original directory), then pass it to run_pipeline — Steps 4-6 load from the
checkpoint, while only Steps 1-3 (data loading/preprocessing/clustering/
pseudotime) and Step 8 (peak perturbation) are recomputed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(description="CausalBridge peak KO quick run (reuse checkpoint)")
    ap.add_argument("--config", required=True, help="Path to the base config.yaml")
    ap.add_argument("--peak", required=True, help="Target peak, e.g. chr3:123400-123900")
    ap.add_argument("--strength", type=float, default=-1.0,
                     help="Perturbation strength s∈[-1,1]: -1=fully closed, -0.5=50%% reduction, +0.5=enhanced")
    ap.add_argument("--mode", choices=["relative", "absolute"], default="relative",
                     help="relative: A'=A(1+s) | absolute: A'=s (raw [0,1] target value)")
    ap.add_argument("--depth", type=int, default=1,
                     help="Propagation depth: 1=direct (default) | 2/3=cascade propagation")
    ap.add_argument("--state", default="all", help="Leiden cluster label or all (default)")
    ap.add_argument("--match", choices=["exact", "overlap", "nearest"], default=None,
                     help="Peak matching strategy (default: use peak_ko.match from config.yaml)")
    ap.add_argument("--out-dir", default=None,
                     help="Output directory (default: <base output.dir>/PEAK_<short_peak>_d<depth>)")
    args = ap.parse_args()

    import yaml
    config = yaml.safe_load(open(args.config))

    config["perturbation"]["knockout_type"] = "peak"
    pk = config["perturbation"].setdefault("peak_ko", {})
    pk["peak_id"] = args.peak
    pk["strength"] = args.strength
    pk["mode"] = args.mode
    pk["depth"] = args.depth
    pk["state"] = args.state
    if args.match is not None:
        pk["match"] = args.match

    orig_out = config["output"]["dir"]
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        short = args.peak.replace(":", "_").replace("-", "_")
        out_dir = Path(orig_out) / f"PEAK_{short}_d{args.depth}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config["output"]["dir"] = str(out_dir) + "/"
    if not config["output"].get("checkpoint_dir"):
        config["output"]["checkpoint_dir"] = str(Path(orig_out) / "checkpoints")

    derived = out_dir / "config_peak.yaml"
    with open(derived, "w") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    print(f"[peak-ko] Derived config → {derived}")
    print(f"[peak-ko] peak={args.peak} strength={args.strength} "
          f"depth={args.depth} state={args.state}")
    print(f"[peak-ko] output.dir={config['output']['dir']}")
    print(f"[peak-ko] checkpoint_dir={config['output']['checkpoint_dir']} (reused)")

    from atac_bridge.run import run_pipeline
    res = run_pipeline(str(derived))
    if "error" in res:
        print(f"[peak-ko] Failed: {res['error']}")
        sys.exit(1)
    pr = res.get("perturbation_results")
    n = 0 if pr is None or pr.empty else len(pr)
    up = res.get("upstream_tfs")
    n_up = 0 if up is None or up.empty else len(up)
    print(f"[peak-ko] Completed: {n} target-gene effects, {n_up} upstream TFs, output directory: {out_dir}")


if __name__ == "__main__":
    main()
