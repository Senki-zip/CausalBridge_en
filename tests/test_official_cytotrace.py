from pathlib import Path
import subprocess
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmread, mmwrite

from CausalBridge import cytotrace as ct
from CausalBridge.granger import _find_root_cell


def test_sparse_adapter_axes_counts_and_failure():
    counts = sparse.csr_matrix([[4, 0, 2], [0, 7, 1]])
    def fake(command, **kwargs):
        folder = Path(command[3])
        np.testing.assert_array_equal(mmread(folder / "counts.mtx").toarray(), counts.T.toarray())
        assert (folder / "cells.txt").read_text().splitlines() == ["b", "a"]
        (folder / "scores.csv").write_text("cell_id,score\nb,0\na,1\n")
        (folder / "runtime.txt").write_text("mock R dependencies")
        return subprocess.CompletedProcess(command, 0, "", "")
    with patch.object(ct.subprocess, "run", side_effect=fake):
        scores, provenance = ct.compute_scores(counts, ["b", "a"], ["g1", "g2", "g3"])
    np.testing.assert_array_equal(scores, [0, 1])
    assert provenance["n_genes"] == 3
    with patch.object(ct.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "missing nnls")):
        with pytest.raises(RuntimeError, match="missing nnls"):
            ct.compute_scores(counts, ["b", "a"], ["g1", "g2", "g3"])
    with pytest.raises(ValueError, match="integer raw counts"):
            ct.compute_scores(counts * .1, ["b", "a"], ["g1", "g2", "g3"])


def test_configured_rscript_ignores_path_and_inherited_r_home(monkeypatch):
    import yaml
    config = yaml.safe_load((Path(ct.__file__).parents[1] / "config.yaml").read_text())
    selected = config["cytotrace"]["rscript_binary"]
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("R_HOME", "/invalid/inherited/R")
    monkeypatch.setenv("R_ARCH", "/invalid-arch")
    monkeypatch.setenv("R_LIBS", "/invalid/external/library")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/invalid/external/native")
    assert ct.runtime_settings(config["cytotrace"])["rscript"] == selected

    def fake(command, **kwargs):
        assert command[0] == selected
        assert "R_HOME" not in kwargs["env"] and "R_ARCH" not in kwargs["env"]
        assert "R_LIBS" not in kwargs["env"]
        assert kwargs["env"]["LD_LIBRARY_PATH"] == ct.runtime_settings()["native_library"]
        assert kwargs["env"]["R_LIBS_SITE"] == ct.runtime_settings()["library"]
        assert kwargs["env"]["R_LIBS_USER"] == ct.runtime_settings()["library"]
        return subprocess.CompletedProcess(command, 1, "", "underlying dependency failure")

    with patch.object(ct.subprocess, "run", side_effect=fake):
        with pytest.raises(RuntimeError, match="underlying dependency failure"):
            ct.compute_scores([[1, 2], [3, 4]], ["a", "b"], ["x", "y"], config["cytotrace"])


def test_runtime_defaults_follow_python_conda_environment(monkeypatch, tmp_path):
    (tmp_path / "conda-meta").mkdir()
    monkeypatch.setattr(ct.sys, "prefix", str(tmp_path))
    with pytest.raises(RuntimeError, match="requires Rscript"):
        ct.runtime_settings()
    (tmp_path / "bin").mkdir()
    executable = tmp_path / "bin/Rscript"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    runtime = ct.runtime_settings()
    assert runtime["rscript"] == str(executable)
    assert runtime["library"] == str(tmp_path / "lib/R/library")
    assert runtime["native_library"] == str(tmp_path / "lib")


def test_runtime_rejects_external_library():
    with pytest.raises(RuntimeError, match="same environment"):
        ct.runtime_settings({"r_library_dir": "/tmp/external-R-library"})


def test_runner_reports_underlying_namespace_error(tmp_path):
    runtime = ct.runtime_settings()
    # Real R with a controlled load failure: retain the transitive dependency
    # detail even when the top-level dependency itself is installed.
    script = tmp_path / "failure.R"
    script.write_text('loadNamespace <- function(...) stop("transitive Rcpp load failure")\n'
                      'source(commandArgs(TRUE)[5])\n')
    result = subprocess.run([runtime["rscript"], "--vanilla", str(script),
                             str(tmp_path), runtime["source"], runtime["library"], "42",
                             str(Path(ct.__file__).with_name("cytotrace_runner.R"))],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "transitive Rcpp load failure" in result.stderr
    assert "Cannot load R dependency HiClimR" in result.stderr
    assert ".libPaths():" in result.stderr and "R version" in result.stderr


def test_untrusted_score_cache_is_rejected_and_scope_is_checked():
    rna = ad.AnnData(np.ones((3, 2)))
    rna.obs["cytotrace_score"] = [0, 1, .5]
    with pytest.raises(RuntimeError, match="per-cohort"):
        _find_root_cell(rna, method="cytotrace")
    rna.uns["official_cytotrace"] = {"schema": ct.SCHEMA, "cell_hash": ct.axis_hash(rna.obs_names)}
    assert _find_root_cell(rna, method="cytotrace") == 1
    with pytest.raises(RuntimeError, match="per-cohort"):
        _find_root_cell(rna[:2], method="cytotrace")


def test_full_raw_gene_axis_and_exact_cohort_order_are_preserved():
    full = ad.AnnData(sparse.csr_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                      obs=pd.DataFrame(index=["a", "b", "c"]),
                      var=pd.DataFrame({"gene_ids": ["ID1", "ID2", "ID3"]}, index=["A", "B", "C"]))
    model = full[["c", "a"], ["A"]].copy()
    model.obs["cytotrace_score"] = [1000., -1000.]
    before = model.X.copy()
    def compute(counts, cells, genes, settings):
        assert list(cells) == ["c", "a"] and list(genes) == ["ID1", "ID2", "ID3"]
        np.testing.assert_array_equal(counts.toarray(), [[7, 8, 9], [1, 2, 3]])
        return np.array([0., 1.]), {"schema": ct.SCHEMA, "cell_hash": ct.axis_hash(cells)}
    with patch.object(ct, "compute_scores", side_effect=compute) as call:
        ct.attach_scores(model, full)
    assert call.call_count == 1
    assert model.n_vars == 1
    np.testing.assert_array_equal(model.X.toarray(), before.toarray())
    assert _find_root_cell(model, method="cytotrace") == 1


def test_real_official_adapter_matches_independent_direct_source(tmp_path):
    # >50 cells, heterogeneous library sizes and gene expression; actual GCS,
    # NNLS and diffusion run in both paths, never a surrogate implementation.
    rng = np.random.default_rng(2020)
    counts = rng.poisson(rng.lognormal(0, 1, (60, 1)) * rng.lognormal(0, 1, (1, 450)))
    cells = [f"cell{i}" for i in range(60)]
    genes = [f"gene{i}" for i in range(450)]
    scores, provenance = ct.compute_scores(sparse.csr_matrix(counts), cells, genes)
    runtime = ct.runtime_settings()
    mmwrite(tmp_path / "direct.mtx", sparse.csr_matrix(counts.T), field="integer")
    script = tmp_path / "direct.R"
    script.write_text('args <- commandArgs(TRUE)\n.libPaths(c(args[1],.libPaths()))\nsource(args[2])\n'
                      'x <- as.matrix(Matrix::readMM(args[3]))\n'
                      'rownames(x) <- paste0("gene",0:(nrow(x)-1)); colnames(x) <- paste0("cell",0:(ncol(x)-1))\n'
                      'set.seed(42); y <- CytoTRACE(x,batch=NULL,enableFast=FALSE,ncores=1,subsamplesize=1000)\n'
                      'write.csv(data.frame(cell_id=names(y$CytoTRACE),score=y$CytoTRACE),args[4],row.names=FALSE)\n')
    import os
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = runtime["native_library"] + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    subprocess.run([runtime["rscript"], "--vanilla", str(script), runtime["library"], runtime["source"],
                    str(tmp_path / "direct.mtx"), str(tmp_path / "direct.csv")], env=env, check=True, capture_output=True)
    direct = pd.read_csv(tmp_path / "direct.csv")
    assert direct.cell_id.tolist() == cells
    np.testing.assert_allclose(scores, direct.score, rtol=0, atol=1e-14)
    assert provenance["source_sha256"] == ct.OFFICIAL_SOURCE_SHA256
    assert "nnls" in provenance["r_runtime"]
