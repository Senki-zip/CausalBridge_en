"""Strict sparse transport to the unmodified official CytoTRACE 0.3.1 R source."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

OFFICIAL_COMMIT = "f4ff6c0ab93cc704eef44e17adcd5c6baf09703a"
OFFICIAL_SOURCE_SHA256 = "7dafcfa42c792eaffe1048f4716cb54105b31f2ad01c4f0c26868c10593b155d"
SCHEMA = "official_cytotrace_0.3.1_cluster_full_counts_v1"


def axis_hash(values):
    return hashlib.sha256(json.dumps(list(map(str, values)), separators=(",", ":")).encode()).hexdigest()


def runtime_settings(settings=None):
    settings = dict(settings or {})
    base = Path(__file__).resolve().parents[1]
    def resolve(value):
        path = Path(value).expanduser()
        return (base / path).resolve() if not path.is_absolute() else path
    source = resolve(settings.get("official_source_dir", "../CytoTRACE_official"))
    prefix = Path(sys.prefix)
    executable = settings.get("rscript_binary", str(prefix / "bin" / "Rscript")
                              if (prefix / "conda-meta").is_dir() else "Rscript")
    if os.path.dirname(executable):
        executable = str(resolve(executable))
    rscript = shutil.which(executable)
    if rscript is None:
        raise RuntimeError("Official CytoTRACE requires Rscript; configure cytotrace.rscript_binary")
    prefix = Path(rscript).absolute().parent.parent
    library = resolve(settings.get("r_library_dir", str(prefix / "lib/R/library")))
    native = resolve(settings.get("native_library_dir", str(prefix / "lib")))
    if library != prefix / "lib/R/library" or native != prefix / "lib":
        raise RuntimeError("CytoTRACE Rscript, R library and native library must belong to the same environment")
    rsource = source / "R" / "CytoTRACE.R"
    if not rsource.is_file() or not (source / "DESCRIPTION").is_file():
        raise RuntimeError("Official CytoTRACE source missing; configure cytotrace.official_source_dir")
    sha = hashlib.sha256(rsource.read_bytes()).hexdigest()
    if sha != OFFICIAL_SOURCE_SHA256 or "Version: 0.3.1" not in (source / "DESCRIPTION").read_text():
        raise RuntimeError("Expected verified unmodified official CytoTRACE 0.3.1 source")
    return dict(source=str(rsource), source_sha256=sha, official_commit=OFFICIAL_COMMIT,
                rscript=rscript, library=str(library), native_library=str(native),
                seed=int(settings.get("seed", 42)), ncores=1, enableFast=False,
                subsamplesize=1000, batch=None, schema=SCHEMA)


def compute_scores(counts, cell_ids, gene_ids, settings=None):
    """Accept cells × FULL original genes, preserving both ordered axes exactly."""
    cells, genes = list(map(str, cell_ids)), list(map(str, gene_ids))
    matrix = sparse.csr_matrix(counts, dtype=np.float64)
    if matrix.shape != (len(cells), len(genes)) or not cells or not genes:
        raise ValueError("CytoTRACE counts/axis dimensions are empty or inconsistent")
    if len(set(cells)) != len(cells) or len(set(genes)) != len(genes):
        raise ValueError("CytoTRACE requires unique stable cell and gene IDs")
    if any("\n" in value or "\r" in value for value in cells + genes):
        raise ValueError("CytoTRACE axis identifiers cannot contain newlines")
    values = matrix.data
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Official CytoTRACE requires finite nonnegative integer raw counts")
    cell_sums = np.asarray(matrix.sum(axis=1)).ravel()
    if (cell_sums <= 0).any():
        raise ValueError("Official CytoTRACE cannot score cells with zero raw counts")
    runtime = runtime_settings(settings)
    digest = hashlib.sha256()
    for array in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(np.ascontiguousarray(array).tobytes())
    provenance = dict(runtime, cell_hash=axis_hash(cells), gene_hash=axis_hash(genes),
                      counts_sha256=digest.hexdigest(), n_cells=len(cells), n_genes=len(genes),
                      input_dtype=str(matrix.dtype), transport="MatrixMarket integer gene_by_cell",
                      scope="model_cohort_full_original_raw_RNA")
    provenance["runner_sha256"] = hashlib.sha256(Path(__file__).with_name("cytotrace_runner.R").read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="causalbridge_cytotrace_") as tmp:
        folder = Path(tmp)
        mmwrite(folder / "counts.mtx", matrix.T.astype(np.int64), field="integer")
        (folder / "cells.txt").write_text("\n".join(cells) + "\n")
        (folder / "genes.txt").write_text("\n".join(genes) + "\n")
        np.savetxt(folder / "cell_sums.txt", cell_sums, fmt="%.17g")
        np.savetxt(folder / "gene_sums.txt", np.asarray(matrix.sum(axis=0)).ravel(), fmt="%.17g")
        env = os.environ.copy()
        # The selected Rscript owns its R installation, even inside another
        # activated environment. Inherited R_HOME/R_ARCH can redirect it.
        env.pop("R_HOME", None)
        env.pop("R_ARCH", None)
        env.pop("R_LIBS", None)
        env["R_LIBS_SITE"] = runtime["library"]
        env["R_LIBS_USER"] = runtime["library"]
        env["LD_LIBRARY_PATH"] = runtime["native_library"]
        command = [runtime["rscript"], "--vanilla", str(Path(__file__).with_name("cytotrace_runner.R")),
                   tmp, runtime["source"], runtime["library"], str(runtime["seed"])]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("Official CytoTRACE failed; check cytotrace R source/library/native runtime.\n"
                               + "Rscript: " + runtime["rscript"] + "\n"
                               + result.stdout[-6000:] + result.stderr[-6000:])
        frame = pd.read_csv(folder / "scores.csv", dtype={"cell_id": str})
        if list(frame.cell_id) != cells or not np.isfinite(frame.score).all():
            raise RuntimeError("Official CytoTRACE returned missing, reordered, or nonfinite cell scores")
        provenance["r_runtime"] = (folder / "runtime.txt").read_text()
        return frame.score.to_numpy(dtype=float), provenance


def attach_scores(model_rna, full_raw_rna, settings=None):
    """Compute once per selected cohort; never trust input obs score caches."""
    if not model_rna.obs_names.is_unique or not full_raw_rna.obs_names.is_unique:
        raise ValueError("CytoTRACE cell IDs must be unique")
    positions = full_raw_rna.obs_names.get_indexer(model_rna.obs_names)
    if (positions < 0).any():
        raise ValueError("CytoTRACE cohort contains cells absent from original raw RNA")
    gene_column = "gene_ids" if "gene_ids" in full_raw_rna.var else None
    genes = full_raw_rna.var[gene_column] if gene_column else full_raw_rna.var_names
    scores, provenance = compute_scores(full_raw_rna.X[positions], model_rna.obs_names, genes, settings)
    model_rna.obs["cytotrace_score"] = scores
    model_rna.uns["official_cytotrace"] = provenance
    return provenance


def verified_scores(rna):
    provenance = rna.uns.get("official_cytotrace", {})
    if (provenance.get("schema") != SCHEMA or provenance.get("cell_hash") != axis_hash(rna.obs_names)
            or "cytotrace_score" not in rna.obs):
        raise RuntimeError("Official per-cohort CytoTRACE scores required: supply full original raw RNA via attach_scores")
    scores = rna.obs["cytotrace_score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Official CytoTRACE scores are nonfinite")
    return scores
