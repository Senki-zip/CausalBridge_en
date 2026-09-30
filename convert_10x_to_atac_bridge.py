#!/usr/bin/env python3
"""
10x output → CausalBridge input format conversion tool

Supports two input modes:

  Mode A — combined Multiome file (h5 or mtx directory) ★ Recommended
    Cell Ranger ARC outputs ATAC+RNA to one h5 file or mtx directory;
    features are automatically split by type label (Gene Expression / Peaks).

    python convert_10x_to_atac_bridge.py \
        -i /home/huangtao/Desktop/huangchengqi/GSE282390_WT_KO/WT/ \
        -g /home/huangtao/Desktop/huangchengqi/reference_documents/refdata-cellranger-arc-mm10-2020-A-2.0.0/genes/genes.gtf \
        -o /home/huangtao/Desktop/huangchengqi/test_1

    python convert_10x_to_atac_bridge.py \
        -i /home/huangtao/Desktop/single_cell/analysis_results/CTRL_ARC_Output/outs/raw_feature_bc_matrix/ \
        -g genes.gtf \
        -o /home/huangtao/Desktop/huangchengqi/atac_bridge/result/

  Mode B — separate RNA and ATAC paths
    Use when RNA and ATAC come from different 10x output directories.

    python convert_10x_to_atac_bridge.py \
        --rna  RNA/ \
        --atac scATAC-seq/Matrix_Export/ \
        --atac-peaks peaks.bed \
        -g genes.gtf \
        -o ./output

Dependencies: scanpy, anndata, pandas, numpy
"""

import argparse
import sys
import json
import gzip
import urllib.request
from pathlib import Path

import numpy as np
from atac_bridge.io import normalize_anndata_string_metadata
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import issparse


def main():
    parser = argparse.ArgumentParser(
        description="10x output → CausalBridge AnnData conversion"
    )

    # --- Input paths ---
    parser.add_argument(
        "--input", "-i", default=None,
        help="Path to a combined Multiome file (.h5) or mtx directory. "
             "Use this instead of --rna/--atac.",
    )
    parser.add_argument(
        "--rna", default=None,
        help="Path to the RNA 10x mtx directory (with matrix.mtx.gz + barcodes.tsv.gz + features.tsv.gz)",
    )
    parser.add_argument(
        "--atac", default=None,
        help="Path to the ATAC 10x mtx directory (with matrix.mtx + barcodes.tsv)",
    )
    parser.add_argument(
        "--atac-peaks", default=None,
        help="ATAC peak annotation file (peaks.bed), required when the ATAC mtx directory lacks features. "
             "Format: chr start end",
    )

    # --- Gene coordinates ---
    parser.add_argument(
        "--gtf", "-g", default=None,
        help="Gene annotation GTF/GFF file (supports .gz), used to extract gene TSS coordinates",
    )
    parser.add_argument(
        "--genome", default=None,
        help="Genome version (hg38/hg19/mm10/mm39), used for online MyGene API queries",
    )

    # --- Output and filtering ---
    parser.add_argument(
        "--output", "-o", default="./atac_bridge_input",
        help="Output directory (default: ./atac_bridge_input)",
    )
    parser.add_argument(
        "--min-genes", type=int, default=200,
        help="Minimum genes detected per cell (default: 200)",
    )
    parser.add_argument(
        "--min-peaks", type=int, default=500,
        help="Minimum peaks detected per cell (default: 500)",
    )

    args = parser.parse_args()

    # --- Validate input combinations ---
    if args.input and (args.rna or args.atac):
        print("Error: --input cannot be used together with --rna/--atac; choose one")
        sys.exit(1)
    if not args.input and not (args.rna and args.atac):
        print("Error: provide --input or both --rna and --atac")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Step 1: Read data
    # ========================================================================
    print("[1/5] Reading input data...")

    if args.input:
        # Mode A: combined multi-omics data
        rna_adata, atac_adata = _read_combined_input(args.input)
    else:
        # Mode B: separate RNA and ATAC data
        rna_adata = _read_10x_mtx_dir(args.rna, modality="rna")
        atac_adata = _read_atac_mtx_dir(args.atac, args.atac_peaks)

        # Warning for Mode B: cells may not match (not paired multi-omics data)
        print("      Note: RNA and ATAC come from different experiments; cells may not match.")
        print("      CausalBridge requires paired multi-omics data (ATAC+RNA from the same cells).")
        print("      If cell counts differ, only shared barcodes will be retained.")

    # ========================================================================
    # Step 2: Basic QC
    # ========================================================================
    print("[2/5] Quality control...")

    rna_adata, atac_adata = _filter_cells(
        rna_adata, atac_adata, args.min_genes, args.min_peaks
    )

    # ========================================================================
    # Step 3: Add genomic coordinates
    # ========================================================================
    print("[3/5] Adding genomic coordinates...")

    # ATAC: parse from peak names or peaks.bed
    atac_adata = _annotate_peak_coords(atac_adata)

    # RNA: obtain gene TSS from GTF or MyGene API
    rna_adata = _annotate_gene_coords(
        rna_adata, args.gtf, args.genome
    )

    # ========================================================================
    # Step 4: Align cell order
    # ========================================================================
    print("[4/5] Aligning cell order...")

    common = sorted(set(rna_adata.obs_names) & set(atac_adata.obs_names))
    if len(common) == 0:
        print("Error: RNA and ATAC have no shared cell barcodes")
        sys.exit(1)

    rna_adata = rna_adata[common].copy()
    atac_adata = atac_adata[common].copy()

    print(f"      Final cell count: {len(common)}")

    # ========================================================================
    # Step 5: Save
    # ========================================================================
    print("[5/5] Saving...")

    rna_path = output_dir / "rna_adata.h5ad"
    atac_path = output_dir / "atac_adata.h5ad"

    normalize_anndata_string_metadata(rna_adata)
    normalize_anndata_string_metadata(atac_adata)
    rna_adata.write(rna_path)
    atac_adata.write(atac_path)

    print(f"      RNA  → {rna_path}")
    print(f"             {rna_adata.n_obs} cells × {rna_adata.n_vars} genes")
    print(f"      ATAC → {atac_path}")
    print(f"             {atac_adata.n_obs} cells × {atac_adata.n_vars} peaks")
    print()
    print("Add to config.yaml:")
    print(f"  input.rna_h5ad: {rna_path.resolve()}")
    print(f"  input.atac_h5ad: {atac_path.resolve()}")

    # Check coordinate completeness
    for label, a in [("RNA", rna_adata), ("ATAC", atac_adata)]:
        missing = [c for c in ["chr", "start", "end"] if c not in a.var.columns]
        if missing:
            print(f"  ⚠ {label} is missing columns: {missing}; check the input data")


# ============================================================================
# Data reading
# ============================================================================

def _read_combined_input(path_str: str):
    """Mode A: read combined multi-omics data and automatically split RNA and ATAC."""
    path = Path(path_str)

    if path.is_dir():
        adata = _read_10x_mtx_dir(path_str, modality="multiome")
    elif path.suffix in (".h5", ".h5ad"):
        # gex_only=False must be explicit: the default True silently discards all Peaks features
        adata = sc.read_10x_h5(str(path), gex_only=False)
    else:
        print(f"Error: unrecognized input format: {path}")
        sys.exit(1)

    # Determine the feature-type column
    if "feature_types" not in adata.var.columns:
        adata.var["feature_types"] = _infer_feature_types(adata)

    rna_mask = adata.var["feature_types"] == "Gene Expression"
    atac_mask = adata.var["feature_types"] == "Peaks"

    if rna_mask.sum() == 0:
        print("Error: no Gene Expression features found")
        sys.exit(1)
    if atac_mask.sum() == 0:
        print("Error: no Peaks features found; is this Multiome data?")
        sys.exit(1)

    rna = adata[:, rna_mask].copy()
    atac = adata[:, atac_mask].copy()

    print(f"      RNA:  {rna.n_obs} cells × {rna.n_vars} genes")
    print(f"      ATAC: {atac.n_obs} cells × {atac.n_vars} peaks")
    return rna, atac


def _read_10x_mtx_dir(dir_path: str, modality: str = "rna"):
    """
    Read a 10x mtx-format directory.

    A standard 10x directory contains:
      matrix.mtx.gz (or .mtx)
      barcodes.tsv.gz (or .tsv)
      features.tsv.gz (or .tsv)   — three columns: feature_id, feature_name, feature_type
    """
    path = Path(dir_path)

    # Find files (supports .gz and uncompressed files)
    def _find(pattern):
        # Try the .gz suffix and the uncompressed version
        candidates = list(path.glob(pattern)) + list(path.glob(pattern.replace(".gz", "")))
        for c in candidates:
            if c.is_file():
                return str(c)
        for c in list(path.glob(pattern + ".*")):
            if c.is_file():
                return str(c)
        return None

    mtx_file = _find("matrix.mtx*")
    barcode_file = _find("barcodes.tsv*")
    feature_file = _find("features.tsv*") or _find("genes.tsv*")

    if not mtx_file:
        print(f"Error: matrix.mtx(.gz) not found in {dir_path}")
        sys.exit(1)

    adata = sc.read_10x_mtx(
        path,
        var_names="gene_symbols",
        gex_only=False,
    )

    print(f"      mtx read: {adata.n_obs} cells × {adata.n_vars} features")
    return adata


def _read_atac_mtx_dir(dir_path: str, peaks_bed: str = None):
    """
    Read an ATAC mtx matrix.

    ATAC directories usually do not contain features.tsv (peak information is in a separate peaks.bed).
    If peaks.bed is provided, use it for var annotations; otherwise infer from the barcode filename.

    Also try reading directly with scanpy's read_10x_mtx (if the directory has a features file).
    """
    path = Path(dir_path)

    # First try standard 10x mtx reading
    mtx_file = list(path.glob("matrix.mtx*"))
    if mtx_file:
        try:
            adata = sc.read_10x_mtx(str(path), gex_only=False)
            print(f"      ATAC mtx: {adata.n_obs} cells × {adata.n_vars} features")
            return adata
        except Exception as e:
            print(f"      Standard mtx read failed ({e}); trying manual reading...")

    # Manual reading: matrix.mtx + barcodes.tsv
    import scipy.io
    from scipy.sparse import csr_matrix

    # Read matrix.mtx
    mtx_candidates = list(path.glob("matrix.mtx*"))
    if not mtx_candidates:
        print(f"Error: matrix.mtx not found in {dir_path}")
        sys.exit(1)

    mat = scipy.io.mmread(str(mtx_candidates[0]))
    if not issparse(mat):
        mat = csr_matrix(mat)

    # Read barcodes
    bc_candidates = list(path.glob("barcodes.tsv*"))
    barcodes = []
    if bc_candidates:
        bc_file = str(bc_candidates[0])
        opener = gzip.open if bc_file.endswith(".gz") else open
        with opener(bc_file, "rt") as f:
            barcodes = [l.strip() for l in f if l.strip()]
    else:
        barcodes = [f"cell_{i}" for i in range(mat.shape[0])]

    # Read or generate peak names
    features_candidates = list(path.glob("features.tsv*"))
    if features_candidates:
        feat_file = str(features_candidates[0])
        opener = gzip.open if feat_file.endswith(".gz") else open
        peak_names = []
        with opener(feat_file, "rt") as f:
            for line in f:
                parts = line.strip().split("\t")
                # Use the second column as the name (10x format: id name type)
                peak_names.append(parts[1] if len(parts) > 1 else parts[0])
    elif peaks_bed:
        with open(peaks_bed) as f:
            peak_names = []
            for line in f:
                if line.strip():
                    c, s, e = line.strip().split("\t")[:3]
                    peak_names.append(f"{c}:{s}-{e}")
    else:
        # Fallback: check whether the directory contains peaks.bed
        bed_candidates = list(path.parent.glob("peaks.bed")) + list(path.parent.glob("*.bed"))
        if bed_candidates:
            with open(bed_candidates[0]) as f:
                peak_names = []
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        c, s, e = line.strip().split("\t")[:3]
                        peak_names.append(f"{c}:{s}-{e}")
        else:
            peak_names = [f"peak_{i}" for i in range(mat.shape[1])]
            print("      Warning: no peak annotation file found; using placeholder names")

    # Construct AnnData (transpose to cells × peaks)
    if mat.shape[0] == len(barcodes) and mat.shape[1] == len(peak_names):
        adata_atac = ad.AnnData(
            X=mat.tocsr(),
            obs=pd.DataFrame(index=barcodes),
            var=pd.DataFrame(index=peak_names),
        )
    elif mat.shape[1] == len(barcodes) and mat.shape[0] == len(peak_names):
        adata_atac = ad.AnnData(
            X=mat.tocsr().T,
            obs=pd.DataFrame(index=barcodes),
            var=pd.DataFrame(index=peak_names),
        )
    else:
        # Try matching the larger dimension
        if mat.shape[0] > mat.shape[1]:
            adata_atac = ad.AnnData(
                X=mat.tocsr(),
                obs=pd.DataFrame(index=barcodes[:mat.shape[0]]),
                var=pd.DataFrame(index=peak_names[:mat.shape[1]]),
            )
        else:
            adata_atac = ad.AnnData(
                X=mat.tocsr().T,
                obs=pd.DataFrame(index=barcodes[:mat.shape[1]]),
                var=pd.DataFrame(index=peak_names[:mat.shape[0]]),
            )

    print(f"      ATAC mtx (manual): {adata_atac.n_obs} cells × {adata_atac.n_vars} peaks")
    return adata_atac


# ============================================================================
# Feature type inference
# ============================================================================

def _infer_feature_types(adata: ad.AnnData) -> pd.Series:
    """Infer feature type from the name. Peak: chr1:12345-67890; gene: anything else."""
    types = []
    for name in adata.var_names:
        if _is_peak_name(name):
            types.append("Peaks")
        else:
            types.append("Gene Expression")
    series = pd.Series(types, index=adata.var_names)
    n_peaks = (series == "Peaks").sum()
    print(f"      Inferred: {len(series) - n_peaks} genes, {n_peaks} peaks")
    return series


def _is_peak_name(name: str) -> bool:
    """Return whether the name matches peak format: chr1:12345-67890 or chr1-12345-67890."""
    try:
        return _parse_peak_name(name) is not None
    except (AttributeError, TypeError):
        return False


def _parse_peak_name(name: str):
    """Parse the two peak-name formats chr:start-end and chr-start-end."""
    if ":" in name:
        parts = name.split(":")
        if len(parts) != 2:
            return None
        chromosome, coordinates = parts
        coords = coordinates.split("-")
    else:
        parts = name.split("-")
        if len(parts) != 3:
            return None
        chromosome, coords = parts[0], parts[1:]

    if len(coords) != 2:
        return None
    try:
        return chromosome, int(coords[0]), int(coords[1])
    except ValueError:
        return None


# ============================================================================
# QC filtering
# ============================================================================

def _filter_cells(rna_adata, atac_adata, min_genes: int, min_peaks: int):
    """Basic cell filtering; retain cells that pass both modalities."""
    # RNA
    rna_n = np.array((rna_adata.X > 0).sum(axis=1)).flatten() if issparse(rna_adata.X) else (rna_adata.X > 0).sum(axis=1)
    rna_keep = rna_n >= min_genes

    # ATAC
    atac_n = np.array((atac_adata.X > 0).sum(axis=1)).flatten() if issparse(atac_adata.X) else (atac_adata.X > 0).sum(axis=1)
    atac_keep = atac_n >= min_peaks

    keep = rna_adata.obs_names[rna_keep & atac_keep]
    print(f"      Before QC: {rna_adata.n_obs} cells")
    print(f"      After QC: {len(keep)} cells")

    return rna_adata[keep].copy(), atac_adata[keep].copy()


# ============================================================================
# Peak coordinate parsing
# ============================================================================

def _annotate_peak_coords(atac_adata: ad.AnnData) -> ad.AnnData:
    """Parse chr/start/end from peak names. Supports colon- and hyphen-separated formats."""
    chrs, starts, ends = [], [], []
    failed = 0

    for name in atac_adata.var_names:
        parsed = _parse_peak_name(name)
        if parsed is not None:
            c, s, e = parsed
            chrs.append(c)
            starts.append(s)
            ends.append(e)
        else:
            chrs.append("unknown")
            starts.append(-1)
            ends.append(-1)
            failed += 1

    atac_adata.var["chr"] = chrs
    atac_adata.var["start"] = starts
    atac_adata.var["end"] = ends

    n = len(atac_adata.var_names)
    if failed > 0:
        print(f"      Warning: could not parse coordinates for {failed}/{n} peaks")
    else:
        print(f"      ATAC: successfully parsed coordinates for {n} peaks")
    return atac_adata


# ============================================================================
# Gene coordinate annotation
# ============================================================================

def _annotate_gene_coords(rna_adata: ad.AnnData, gtf_path: str = None, genome: str = None) -> ad.AnnData:
    """
    Add chr/start/end columns to RNA .var.

    Priority: GTF file > MyGene.info API > local cache > placeholder values
    """
    gene_info = None

    # Strategy 1: GTF
    if gtf_path and Path(gtf_path).exists():
        print(f"      Parsing GTF: {gtf_path}")
        gene_info = _parse_gtf_genes(gtf_path, set(rna_adata.var_names))
        if gene_info:
            print(f"      GTF matches: {len(gene_info)}/{rna_adata.n_vars} genes")

    # Strategy 2: MyGene API
    if gene_info is None or len(gene_info) < rna_adata.n_vars * 0.3:
        print("      Trying online MyGene.info query...")
        gene_info = _query_mygene(list(rna_adata.var_names), genome)

    # Strategy 3: local cache
    if gene_info is None:
        cache = Path.home() / ".atac_bridge" / "gene_coords.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
            gene_info = {
                r["gene_name"]: {"chr": r["chr"], "start": r["start"], "end": r["end"]}
                for _, r in df.iterrows()
            }
            n = sum(1 for g in rna_adata.var_names if g in gene_info)
            print(f"      Cache matches: {n}/{rna_adata.n_vars} genes")

    # Fill annotations
    if gene_info is None:
        print("      Warning: unable to obtain gene coordinates; using placeholder values")
        print("      (CausalBridge can still run, but genomic-distance filtering will be skipped)")
        rna_adata.var["chr"] = "unknown"
        rna_adata.var["start"] = -1
        rna_adata.var["end"] = -1
        return rna_adata

    for col in ["chr", "start", "end"]:
        rna_adata.var[col] = [
            gene_info.get(g, {}).get(col, "unknown" if col == "chr" else -1)
            for g in rna_adata.var_names
        ]
    n = (rna_adata.var["chr"] != "unknown").sum()
    print(f"      Gene coordinates: {n}/{rna_adata.n_vars} annotated")
    return rna_adata


def _parse_gtf_genes(gtf_path: str, gene_set: set) -> dict:
    """Extract chr and TSS for each gene from GTF/GFF, accounting for strand direction."""
    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    info = {}

    try:
        with opener(gtf_path, "rt") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                fields = line.strip().split("\t")
                if len(fields) < 9 or fields[2] != "gene":
                    continue

                chrom, start, end, strand = fields[0], int(fields[3]), int(fields[4]), fields[6]
                tss = start if strand == "+" else end

                attrs = {}
                for a in fields[8].strip().rstrip(";").split(";"):
                    a = a.strip()
                    if not a:
                        continue
                    kv = a.split(" ", 1) if " " in a else a.split("=", 1)
                    if len(kv) == 2:
                        attrs[kv[0].strip()] = kv[1].strip().strip('"')

                for key in ["gene_name", "gene_id"]:
                    val = attrs.get(key, "")
                    if val and val in gene_set:
                        info[val] = {"chr": chrom, "start": tss, "end": tss + 1}
                        break
    except Exception as e:
        print(f"      GTF parsing failed: {e}")
        return None

    return info if info else None


def _query_mygene(gene_names, genome: str = None) -> dict:
    """Fetch gene coordinates in batches through the MyGene.info API and cache them."""
    species = {"hg38": "human", "hg19": "human", "mm10": "mouse", "mm39": "mouse"}
    sp = species.get(genome, "human")

    try:
        req = urllib.request.Request(
            "https://mygene.info/v3/query",
            data=json.dumps({
                "q": list(gene_names)[:1000],
                "scopes": "symbol",
                "fields": "genomic_pos",
                "species": sp,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            results = json.loads(resp.read())

        info = {}
        for item in results:
            gene = item.get("query")
            pos = item.get("genomic_pos")
            if gene and pos and "chr" in pos:
                info[gene] = {
                    "chr": pos["chr"],
                    "start": pos.get("start", 0),
                    "end": pos.get("end", 0),
                }

        if info:
            cache = Path.home() / ".atac_bridge" / "gene_coords.parquet"
            cache.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([
                {"gene_name": g, "chr": i["chr"], "start": i["start"], "end": i["end"]}
                for g, i in info.items()
            ]).to_parquet(cache)

        print(f"      MyGene returned: {len(info)} genes")
        return info if info else None

    except Exception as e:
        print(f"      MyGene query failed: {e}")
        return None


if __name__ == "__main__":
    main()
