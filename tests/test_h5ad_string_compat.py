import numpy as np
import pandas as pd
import anndata as ad

from CausalBridge.io import normalize_anndata_string_metadata


def test_nullable_string_metadata_is_writable(tmp_path):
    obs_index = pd.Index(pd.array(["cell-1", "cell-2"], dtype="string"))
    var_index = pd.Index(pd.array(["gene-1"], dtype="string"))
    obs = pd.DataFrame(
        {"label": pd.Series(["treated", pd.NA], index=obs_index, dtype="string")},
        index=obs_index,
    )
    var = pd.DataFrame(index=var_index)
    adata = ad.AnnData(np.ones((2, 1)), obs=obs, var=var)

    normalize_anndata_string_metadata(adata)
    output = tmp_path / "nullable_strings.h5ad"
    adata.write_h5ad(output)

    assert adata.obs.index.dtype == object
    assert adata.var.index.dtype == object
    assert adata.obs["label"].dtype == object
    assert adata.obs["label"].tolist() == ["treated", ""]
    assert ad.read_h5ad(output).obs_names.tolist() == ["cell-1", "cell-2"]
