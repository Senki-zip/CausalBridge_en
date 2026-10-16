#!/usr/bin/env python3
"""
10x 输出 → CausalBridge 输入格式转换工具

支持两种输入模式：

  模式A — 合并的 Multiome 文件（h5 或 mtx 目录） ★ 推荐
    Cell Ranger ARC 同时输出 ATAC+RNA 到一个 h5 文件或 mtx 目录，
    feature 按类型标签（Gene Expression / Peaks），自动拆分。

    python convert_10x_to_CausalBridge.py \
        -i /home/huangtao/Desktop/huangchengqi/GSE282390_WT_KO/WT/ \
        -g /home/huangtao/Desktop/huangchengqi/reference_documents/refdata-cellranger-arc-mm10-2020-A-2.0.0/genes/genes.gtf \
        -o /home/huangtao/Desktop/huangchengqi/test_1

    python convert_10x_to_CausalBridge.py \
        -i /home/huangtao/Desktop/single_cell/analysis_results/CTRL_ARC_Output/outs/raw_feature_bc_matrix/ \
        -g genes.gtf \
        -o /home/huangtao/Desktop/huangchengqi/CausalBridge/result/

  模式B — 分离的 RNA 和 ATAC 路径
    当 RNA 和 ATAC 来自不同的 10x 输出目录时使用。

    python convert_10x_to_CausalBridge.py \
        --rna  RNA/ \
        --atac scATAC-seq/Matrix_Export/ \
        --atac-peaks peaks.bed \
        -g genes.gtf \
        -o ./output

依赖：scanpy, anndata, pandas, numpy
"""

import argparse
import sys
import json
import gzip
import urllib.request
from pathlib import Path

import numpy as np
from CausalBridge.io import normalize_anndata_string_metadata
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import issparse


def main():
    parser = argparse.ArgumentParser(
        description="10x 输出 → CausalBridge AnnData 转换"
    )

    # --- 输入路径 ---
    parser.add_argument(
        "--input", "-i", default=None,
        help="合并的 Multiome 文件（.h5）或 mtx 目录路径。"
             "与 --rna/--atac 二选一。",
    )
    parser.add_argument(
        "--rna", default=None,
        help="RNA 数据的 10x mtx 目录路径（含 matrix.mtx.gz + barcodes.tsv.gz + features.tsv.gz）",
    )
    parser.add_argument(
        "--atac", default=None,
        help="ATAC 数据的 10x mtx 目录路径（含 matrix.mtx + barcodes.tsv）",
    )
    parser.add_argument(
        "--atac-peaks", default=None,
        help="ATAC peak 注释文件（peaks.bed），ATAC mtx 目录缺少 features 时需要。"
             "格式: chr start end",
    )

    # --- 基因坐标 ---
    parser.add_argument(
        "--gtf", "-g", default=None,
        help="基因注释 GTF/GFF 文件（支持 .gz），用于提取基因 TSS 坐标",
    )
    parser.add_argument(
        "--genome", default=None,
        help="基因组版本（hg38/hg19/mm10/mm39），用于 MyGene API 在线查询",
    )

    # --- 输出与过滤 ---
    parser.add_argument(
        "--output", "-o", default="./CausalBridge_input",
        help="输出目录（默认 ./CausalBridge_input）",
    )
    parser.add_argument(
        "--min-genes", type=int, default=200,
        help="细胞至少检测到的基因数（默认 200）",
    )
    parser.add_argument(
        "--min-peaks", type=int, default=500,
        help="细胞至少检测到的 peak 数（默认 500）",
    )

    args = parser.parse_args()

    # --- 验证输入组合 ---
    if args.input and (args.rna or args.atac):
        print("错误: --input 与 --rna/--atac 不能同时使用，请二选一")
        sys.exit(1)
    if not args.input and not (args.rna and args.atac):
        print("错误: 必须提供 --input 或同时提供 --rna 和 --atac")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Step 1: 读取数据
    # ========================================================================
    print("[1/5] 读取输入数据...")

    if args.input:
        # 模式A: 合并的多组学数据
        rna_adata, atac_adata = _read_combined_input(args.input)
    else:
        # 模式B: 分离的 RNA 和 ATAC 数据
        rna_adata = _read_10x_mtx_dir(args.rna, modality="rna")
        atac_adata = _read_atac_mtx_dir(args.atac, args.atac_peaks)

        # 模式B 的警告: 细胞可能不匹配（非多组学配对数据）
        print("      注意: RNA 和 ATAC 来自不同实验，细胞可能不匹配。")
        print("      CausalBridge 需要配对的多组学数据（同一细胞的 ATAC+RNA）。")
        print("      如果细胞数不同，将仅保留共有 barcode。")

    # ========================================================================
    # Step 2: 基础 QC
    # ========================================================================
    print("[2/5] 质量控制...")

    rna_adata, atac_adata = _filter_cells(
        rna_adata, atac_adata, args.min_genes, args.min_peaks
    )

    # ========================================================================
    # Step 3: 添加基因组坐标
    # ========================================================================
    print("[3/5] 添加基因组坐标...")

    # ATAC: 从 peak 名称或 peaks.bed 解析
    atac_adata = _annotate_peak_coords(atac_adata)

    # RNA: 从 GTF 或 MyGene API 获取基因 TSS
    rna_adata = _annotate_gene_coords(
        rna_adata, args.gtf, args.genome
    )

    # ========================================================================
    # Step 4: 统一细胞顺序
    # ========================================================================
    print("[4/5] 统一细胞顺序...")

    common = sorted(set(rna_adata.obs_names) & set(atac_adata.obs_names))
    if len(common) == 0:
        print("错误: RNA 和 ATAC 无共享细胞 barcode")
        sys.exit(1)

    rna_adata = rna_adata[common].copy()
    atac_adata = atac_adata[common].copy()

    print(f"      最终细胞数: {len(common)}")

    # ========================================================================
    # Step 5: 保存
    # ========================================================================
    print("[5/5] 保存...")

    rna_path = output_dir / "rna_adata.h5ad"
    atac_path = output_dir / "atac_adata.h5ad"

    normalize_anndata_string_metadata(rna_adata)
    normalize_anndata_string_metadata(atac_adata)
    rna_adata.write(rna_path)
    atac_adata.write(atac_path)

    print(f"      RNA  → {rna_path}")
    print(f"             {rna_adata.n_obs} 细胞 × {rna_adata.n_vars} 基因")
    print(f"      ATAC → {atac_path}")
    print(f"             {atac_adata.n_obs} 细胞 × {atac_adata.n_vars} peaks")
    print()
    print("config.yaml 中填入:")
    print(f"  input.rna_h5ad: {rna_path.resolve()}")
    print(f"  input.atac_h5ad: {atac_path.resolve()}")

    # 检查坐标完整性
    for label, a in [("RNA", rna_adata), ("ATAC", atac_adata)]:
        missing = [c for c in ["chr", "start", "end"] if c not in a.var.columns]
        if missing:
            print(f"  ⚠ {label} 缺少列: {missing}，请检查输入数据")


# ============================================================================
# 数据读取
# ============================================================================

def _read_combined_input(path_str: str):
    """模式A: 读取合并的多组学数据，自动拆分为 RNA 和 ATAC。"""
    path = Path(path_str)

    if path.is_dir():
        adata = _read_10x_mtx_dir(path_str, modality="multiome")
    elif path.suffix in (".h5", ".h5ad"):
        # gex_only=False 必须显式传: 默认 True 会静默丢弃全部 Peaks features
        adata = sc.read_10x_h5(str(path), gex_only=False)
    else:
        print(f"错误: 无法识别输入格式: {path}")
        sys.exit(1)

    # 确定 feature 类型列
    if "feature_types" not in adata.var.columns:
        adata.var["feature_types"] = _infer_feature_types(adata)

    rna_mask = adata.var["feature_types"] == "Gene Expression"
    atac_mask = adata.var["feature_types"] == "Peaks"

    if rna_mask.sum() == 0:
        print("错误: 未找到 Gene Expression features")
        sys.exit(1)
    if atac_mask.sum() == 0:
        print("错误: 未找到 Peaks features，是否为 Multiome 数据？")
        sys.exit(1)

    rna = adata[:, rna_mask].copy()
    atac = adata[:, atac_mask].copy()

    print(f"      RNA:  {rna.n_obs} 细胞 × {rna.n_vars} 基因")
    print(f"      ATAC: {atac.n_obs} 细胞 × {atac.n_vars} peaks")
    return rna, atac


def _read_10x_mtx_dir(dir_path: str, modality: str = "rna"):
    """
    读取 10x mtx 格式目录。

    标准 10x 目录包含:
      matrix.mtx.gz (或 .mtx)
      barcodes.tsv.gz (或 .tsv)
      features.tsv.gz (或 .tsv)   — 三列: feature_id, feature_name, feature_type
    """
    path = Path(dir_path)

    # 查找文件（支持 .gz 和未压缩）
    def _find(pattern):
        # 尝试 .gz 后缀和未压缩版本
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
        print(f"错误: 在 {dir_path} 中未找到 matrix.mtx(.gz)")
        sys.exit(1)

    adata = sc.read_10x_mtx(
        path,
        var_names="gene_symbols",
        gex_only=False,
    )

    print(f"      mtx 读取: {adata.n_obs} 细胞 × {adata.n_vars} features")
    return adata


def _read_atac_mtx_dir(dir_path: str, peaks_bed: str = None):
    """
    读取 ATAC mtx 矩阵。

    ATAC 目录通常不包含 features.tsv（peak 信息在单独的 peaks.bed 中）。
    如果提供了 peaks.bed，用其作为 var 注释；否则从 barcode 文件名推断。

    同时尝试用 scanpy 的 read_10x_mtx 直接读取（如果目录有 features 文件）。
    """
    path = Path(dir_path)

    # 先尝试标准 10x mtx 读取
    mtx_file = list(path.glob("matrix.mtx*"))
    if mtx_file:
        try:
            adata = sc.read_10x_mtx(str(path), gex_only=False)
            print(f"      ATAC mtx: {adata.n_obs} 细胞 × {adata.n_vars} features")
            return adata
        except Exception as e:
            print(f"      标准 mtx 读取失败 ({e})，尝试手动读取...")

    # 手动读取: matrix.mtx + barcodes.tsv
    import scipy.io
    from scipy.sparse import csr_matrix

    # 读取 matrix.mtx
    mtx_candidates = list(path.glob("matrix.mtx*"))
    if not mtx_candidates:
        print(f"错误: 在 {dir_path} 中未找到 matrix.mtx")
        sys.exit(1)

    mat = scipy.io.mmread(str(mtx_candidates[0]))
    if not issparse(mat):
        mat = csr_matrix(mat)

    # 读取 barcodes
    bc_candidates = list(path.glob("barcodes.tsv*"))
    barcodes = []
    if bc_candidates:
        bc_file = str(bc_candidates[0])
        opener = gzip.open if bc_file.endswith(".gz") else open
        with opener(bc_file, "rt") as f:
            barcodes = [l.strip() for l in f if l.strip()]
    else:
        barcodes = [f"cell_{i}" for i in range(mat.shape[0])]

    # 读取或生成 peak 名称
    features_candidates = list(path.glob("features.tsv*"))
    if features_candidates:
        feat_file = str(features_candidates[0])
        opener = gzip.open if feat_file.endswith(".gz") else open
        peak_names = []
        with opener(feat_file, "rt") as f:
            for line in f:
                parts = line.strip().split("\t")
                # 取第二列作为名称（10x 格式: id name type）
                peak_names.append(parts[1] if len(parts) > 1 else parts[0])
    elif peaks_bed:
        with open(peaks_bed) as f:
            peak_names = []
            for line in f:
                if line.strip():
                    c, s, e = line.strip().split("\t")[:3]
                    peak_names.append(f"{c}:{s}-{e}")
    else:
        # 兜底: 检查目录中是否有 peaks.bed
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
            print("      警告: 未找到 peak 注释文件，将使用占位名称")

    # 构造 AnnData（转置为 cells × peaks）
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
        # 尝试匹配较大维度
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

    print(f"      ATAC mtx (手动): {adata_atac.n_obs} 细胞 × {adata_atac.n_vars} peaks")
    return adata_atac


# ============================================================================
# Feature 类型推断
# ============================================================================

def _infer_feature_types(adata: ad.AnnData) -> pd.Series:
    """从名称推断 feature 类型。Peak: chr1:12345-67890, 基因: 其他。"""
    types = []
    for name in adata.var_names:
        if _is_peak_name(name):
            types.append("Peaks")
        else:
            types.append("Gene Expression")
    series = pd.Series(types, index=adata.var_names)
    n_peaks = (series == "Peaks").sum()
    print(f"      推断: {len(series) - n_peaks} 基因, {n_peaks} peaks")
    return series


def _is_peak_name(name: str) -> bool:
    """判断名称是否符合 peak 格式: chr1:12345-67890 或 chr1-12345-67890。"""
    try:
        return _parse_peak_name(name) is not None
    except (AttributeError, TypeError):
        return False


def _parse_peak_name(name: str):
    """解析 chr:start-end 和 chr-start-end 两种 peak 名称格式。"""
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
# QC 过滤
# ============================================================================

def _filter_cells(rna_adata, atac_adata, min_genes: int, min_peaks: int):
    """基础细胞过滤，保留两个模态均通过的细胞。"""
    # RNA
    rna_n = np.array((rna_adata.X > 0).sum(axis=1)).flatten() if issparse(rna_adata.X) else (rna_adata.X > 0).sum(axis=1)
    rna_keep = rna_n >= min_genes

    # ATAC
    atac_n = np.array((atac_adata.X > 0).sum(axis=1)).flatten() if issparse(atac_adata.X) else (atac_adata.X > 0).sum(axis=1)
    atac_keep = atac_n >= min_peaks

    keep = rna_adata.obs_names[rna_keep & atac_keep]
    print(f"      QC 前: {rna_adata.n_obs} 细胞")
    print(f"      QC 后: {len(keep)} 细胞")

    return rna_adata[keep].copy(), atac_adata[keep].copy()


# ============================================================================
# Peak 坐标解析
# ============================================================================

def _annotate_peak_coords(atac_adata: ad.AnnData) -> ad.AnnData:
    """从 peak 名称解析 chr/start/end。支持冒号和连字符分隔格式。"""
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
        print(f"      警告: {failed}/{n} 个 peak 无法解析坐标")
    else:
        print(f"      ATAC: {n} 个 peak 坐标解析成功")
    return atac_adata


# ============================================================================
# 基因坐标注释
# ============================================================================

def _annotate_gene_coords(rna_adata: ad.AnnData, gtf_path: str = None, genome: str = None) -> ad.AnnData:
    """
    为 RNA .var 添加 chr/start/end 列。

    优先级: GTF 文件 > MyGene.info API > 本地缓存 > 占位值
    """
    gene_info = None

    # 策略1: GTF
    if gtf_path and Path(gtf_path).exists():
        print(f"      解析 GTF: {gtf_path}")
        gene_info = _parse_gtf_genes(gtf_path, set(rna_adata.var_names))
        if gene_info:
            print(f"      GTF 匹配: {len(gene_info)}/{rna_adata.n_vars} 基因")

    # 策略2: MyGene API
    if gene_info is None or len(gene_info) < rna_adata.n_vars * 0.3:
        print("      尝试 MyGene.info 在线查询...")
        gene_info = _query_mygene(list(rna_adata.var_names), genome)

    # 策略3: 本地缓存
    if gene_info is None:
        cache = Path.home() / ".CausalBridge" / "gene_coords.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
            gene_info = {
                r["gene_name"]: {"chr": r["chr"], "start": r["start"], "end": r["end"]}
                for _, r in df.iterrows()
            }
            n = sum(1 for g in rna_adata.var_names if g in gene_info)
            print(f"      缓存匹配: {n}/{rna_adata.n_vars} 基因")

    # 填充
    if gene_info is None:
        print("      警告: 无法获取基因坐标，将使用占位值")
        print("      (CausalBridge 仍可运行，但会跳过基因组距离过滤)")
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
    print(f"      基因坐标: {n}/{rna_adata.n_vars} 已注释")
    return rna_adata


def _parse_gtf_genes(gtf_path: str, gene_set: set) -> dict:
    """从 GTF/GFF 提取每个 gene 的 chr 和 TSS（考虑链方向）。"""
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
        print(f"      GTF 解析失败: {e}")
        return None

    return info if info else None


def _query_mygene(gene_names, genome: str = None) -> dict:
    """通过 MyGene.info API 批量获取基因坐标并缓存。"""
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
            cache = Path.home() / ".CausalBridge" / "gene_coords.parquet"
            cache.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([
                {"gene_name": g, "chr": i["chr"], "start": i["start"], "end": i["end"]}
                for g, i in info.items()
            ]).to_parquet(cache)

        print(f"      MyGene 返回: {len(info)} 基因")
        return info if info else None

    except Exception as e:
        print(f"      MyGene 查询失败: {e}")
        return None


if __name__ == "__main__":
    main()
