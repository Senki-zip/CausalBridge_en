"""Low-count target rescue preserves the QC-passing model universe."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from CausalBridge.preprocess import preprocess_rna


@pytest.mark.parametrize("as_sparse", [False, True])
def test_rare_target_rescue_preserves_other_values_and_hvg(as_sparse):
    rng = np.random.default_rng(42)
    counts = rng.poisson(rng.uniform(1, 8, (120, 600))).astype(float)
    rare = np.zeros((120, 2))
    rare[:17, 0] = 1
    matrix = np.column_stack([counts, rare])
    names = [f"g{i}" for i in range(600)] + ["Zic1", "ZERO"]
    source = ad.AnnData(sparse.csr_matrix(matrix) if as_sparse else matrix,
                        var=pd.DataFrame(index=names))
    kwargs = dict(min_cells_per_gene=50, n_highly_variable_genes=80, filter_cells=False)
    ordinary = preprocess_rna(source, **kwargs)
    rescued = preprocess_rna(source, keep_genes=["Zic1", "ZERO"],
                             retain_low_count_targets=True, **kwargs)
    assert "Zic1" in rescued.var_names and "ZERO" not in rescued.var_names
    assert not rescued.var.loc["Zic1", "highly_variable"]
    assert rescued.var.loc["Zic1", "n_cells"] == 17
    assert rescued.var_names.drop("Zic1").equals(ordinary.var_names)
    to_dense = lambda x: x.toarray() if sparse.issparse(x) else x
    np.testing.assert_allclose(to_dense(rescued[:, ordinary.var_names].X), to_dense(ordinary.X), rtol=1e-6)
    pd.testing.assert_series_equal(rescued.var.loc[ordinary.var_names, "highly_variable"], ordinary.var.highly_variable)
    expected = np.log1p(rare[:, 0] * 1e4 / counts.sum(axis=1))
    np.testing.assert_allclose(to_dense(rescued[:, ["Zic1"]].X).ravel(), expected)
    np.testing.assert_array_equal(to_dense(rescued[:, ["Zic1"]].layers["raw"]).ravel(), rare[:, 0])
    np.testing.assert_array_equal(to_dense(source.X), matrix)
    reference = preprocess_rna(source, keep_genes=["Zic1"], **kwargs)
    assert "Zic1" not in reference.var_names
