#!/usr/bin/env python3
"""CausalBridge peak KO 快速试跑 CLI: 复用已有 checkpoint, 秒级换 peak/强度/深度.

用法:
    python run_peak_perturbation.py --config config.yaml \
        --peak chr3:123400-123900 --strength -1.0 --depth 1 \
        [--state all] [--match exact] [--out-dir results/test/MY_PEAK]

原理: 在基础 config 上派生一份 peak 模式 config (knockout_type=peak +
peak_ko 段 + 隔离 output.dir + checkpoint 指向原目录复用), 交给
run_pipeline 执行 — Step 4-6 从 checkpoint 加载, 只重算 Step 1-3
(数据加载/预处理/聚类/伪时间) 与 Step 8 (peak 扰动)。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(description="CausalBridge peak KO 快速试跑 (复用 checkpoint)")
    ap.add_argument("--config", required=True, help="基础 config.yaml 路径")
    ap.add_argument("--peak", required=True, help="目标 peak, 如 chr3:123400-123900")
    ap.add_argument("--strength", type=float, default=-1.0,
                    help="扰动强度 s∈[-1,1]: -1=完全关闭, -0.5=降50%%, +0.5=增强")
    ap.add_argument("--mode", choices=["relative", "absolute"], default="relative",
                    help="relative: A'=A(1+s) | absolute: A'=s (raw [0,1] 目标值)")
    ap.add_argument("--depth", type=int, default=1,
                    help="传播深度: 1=direct (默认) | 2/3=级联传播")
    ap.add_argument("--state", default="all", help="Leiden 簇标签 或 all (默认)")
    ap.add_argument("--match", choices=["exact", "overlap", "nearest"], default=None,
                    help="peak 匹配策略 (默认: 沿用 config.yaml 的 peak_ko.match)")
    ap.add_argument("--out-dir", default=None,
                    help="输出目录 (默认: 沿用配置的 output.dir)")
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

    if args.out_dir:
        config["output"]["dir"] = args.out_dir
    out_dir = Path(config["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    derived = out_dir / "config_peak.yaml"
    with open(derived, "w") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    print(f"[peak-ko] 派生 config → {derived}")
    print(f"[peak-ko] peak={args.peak} strength={args.strength} "
          f"depth={args.depth} state={args.state}")
    print(f"[peak-ko] output.dir={config['output']['dir']}")
    checkpoint_dir = config["output"].get("checkpoint_dir") or out_dir / "checkpoints"
    print(f"[peak-ko] checkpoint_dir={checkpoint_dir}")

    from CausalBridge.run import run_pipeline
    res = run_pipeline(str(derived))
    if "error" in res:
        print(f"[peak-ko] 失败: {res['error']}")
        sys.exit(1)
    pr = res.get("perturbation_results")
    n = 0 if pr is None or pr.empty else len(pr)
    up = res.get("upstream_tfs")
    n_up = 0 if up is None or up.empty else len(up)
    print(f"[peak-ko] 完成: {n} 条靶基因效应, {n_up} 条上游 TF, 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
