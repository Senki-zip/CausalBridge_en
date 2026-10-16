#!/usr/bin/env python3
"""
CausalBridge 使用示例

用法:
    python run_example.py config.yaml

最小化快速启动:
    1. 准备好两个 AnnData 文件 (rna_adata.h5ad, atac_adata.h5ad)
    2. 创建 target_genes.txt (每行一个基因符号)
    3. 编辑 config.yaml 中的 input 路径
    4. python run_example.py config.yaml
"""

import sys
import argparse
from pathlib import Path

# 将 CausalBridge 包所在目录加入路径
sys.path.insert(0, str(Path(__file__).parent))

from CausalBridge.run import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="CausalBridge: 基于 ATAC 桥接的因果扰动预测框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python run_example.py config.yaml
    python run_example.py config.yaml --verbose
        """,
    )
    parser.add_argument(
        "config",
        type=str,
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="输出详细日志",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅验证配置和数据，不执行完整分析",
    )

    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.getLogger("CausalBridge").setLevel(logging.DEBUG)

    if args.dry_run:
        print("Dry-run 模式: 验证配置和数据格式...")
        from CausalBridge.io import load_config, load_data
        config = load_config(args.config)
        rna, atac = load_data(config)
        print(f"  RNA: {rna.n_obs} 细胞 × {rna.n_vars} 基因")
        print(f"  ATAC: {atac.n_obs} 细胞 × {atac.n_vars} peaks")
        print("  格式验证通过！")
        return

    # 执行完整 pipeline
    results = run_pipeline(args.config)

    if "error" in results:
        print(f"错误: {results['error']}")
        sys.exit(1)

    print("\n分析成功完成！结果已保存至配置中指定的输出目录。")


if __name__ == "__main__":
    main()
