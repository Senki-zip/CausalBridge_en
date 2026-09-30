#!/usr/bin/env python3
"""
CausalBridge usage example

Usage:
    python run_example.py config.yaml

Minimal quick start:
    1. Prepare two AnnData files (rna_adata.h5ad, atac_adata.h5ad)
    2. Create target_genes.txt (one gene symbol per line)
    3. Edit the input paths in config.yaml
    4. python run_example.py config.yaml
"""

import sys
import argparse
from pathlib import Path

# Add the directory containing the atac_bridge package to the path
sys.path.insert(0, str(Path(__file__).parent))

from atac_bridge.run import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="CausalBridge: causal perturbation prediction framework bridged by ATAC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_example.py config.yaml
    python run_example.py config.yaml --verbose
        """,
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Output verbose logs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and data only; do not run the full analysis",
    )

    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.getLogger("atac_bridge").setLevel(logging.DEBUG)

    if args.dry_run:
        print("Dry-run mode: validating configuration and data format...")
        from atac_bridge.io import load_config, load_data
        config = load_config(args.config)
        rna, atac = load_data(config)
        print(f"  RNA: {rna.n_obs} cells × {rna.n_vars} genes")
        print(f"  ATAC: {atac.n_obs} cells × {atac.n_vars} peaks")
        print("  Format validation passed!")
        return

    # Run the full pipeline
    results = run_pipeline(args.config)

    if "error" in results:
        print(f"Error: {results['error']}")
        sys.exit(1)

    print("\nAnalysis completed successfully! Results were saved to the output directory specified in the configuration.")


if __name__ == "__main__":
    main()
