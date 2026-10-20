#!/usr/bin/env python3
"""Export saved embedding and obs without initializing R's Python bridge."""
import sys
import anndata as ad
import numpy as np

data = ad.read_h5ad(sys.argv[1], backed="r")
key = next((k for k in ("X_umap", "umap", "X_UMAP") if k in data.obsm), None)
if key is None:
    raise ValueError("Saved h5ad has no UMAP embedding")
obs = data.obs.copy()
obs.insert(0, "cell_barcode", data.obs_names.astype(str))
coords = np.asarray(data.obsm[key])
obs["UMAP_1"] = coords[:, 0]
obs["UMAP_2"] = coords[:, 1]
obs.to_csv(sys.argv[2], index=False)
data.file.close()
