"""Unrelated shared TF inputs must remain exact WT under direct RNA clamps."""
import copy
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from CausalBridge import atac_normalization as norm, perturbation
from CausalBridge import saturating_hybrid as core, saturating_transfer as transfer
from CausalBridge import saturating_perturbation as simulation


def fitted(continuous, smooth, branches=1):
    from test_saturating_transfer import fixture_data
    rna, atac, pt, edges, _, cfg = fixture_data(64 if branches == 2 else 32, branches)
    matrix=rna.X.toarray()
    matrix[:,2]=np.linspace(.1,3.,rna.n_obs)
    rna.X=sparse.csr_matrix(matrix)
    cfg["pseudotime"]["smooth_kernel"]=smooth
    cfg["perturbation"]["min_r2_threshold"]=1.1
    cfg["perturbation"]["min_lag_per_peak_filter"]=True
    screen=pd.DataFrame({"tf_gene":["A","C","C"],"peak_id":["p0"]*3,
        "tf_lag":[0,0,1],"weight":[1.,1.,1.]})
    if continuous:
        raw=ad.AnnData(sparse.hstack([atac.X,sparse.csr_matrix(np.full((atac.n_obs,1),100.))],format="csr"),
            obs=atac.obs.copy(),var=pd.concat([atac.var,pd.DataFrame(index=["background"])]))
        prepared=norm.prepare_atac_counts(raw)
        qc=norm.finalize_atac_qc(prepared[:,:2].copy(),prepared)
        cfg["preprocess"]={"atac":{"binarize":True},"atac_depth_normalization":{
            "enabled":True,"target_sum":10000.,"output_cap_expansion":.25}}
        atac=norm.depth_log1p_model_view(qc,norm.validate_atac_depth_config(cfg))
    table,_,genes=transfer.fit_peak_gene(rna,atac,pt,edges,cfg)
    tfs=transfer.fit_tf_peak(rna,atac,pt,screen,cfg)
    return rna,atac,pt,edges,screen,cfg,table,genes,tfs


@pytest.mark.parametrize("continuous",[False,True])
@pytest.mark.parametrize("smooth",["hard","gaussian"])
@pytest.mark.parametrize("branches",[1,2])
def test_public_ko_preserves_unselected_shared_tf_inputs_exactly(continuous,smooth,branches,monkeypatch):
    rna,atac,pt,edges,screen,cfg,table,genes,tfs=fitted(continuous,smooth,branches)
    frozen=copy.deepcopy((genes,tfs))
    cells=rna.X.toarray()
    # Show that the previous all-column transform introduces actual changes.
    assert np.any(np.log1p(np.expm1(cells[:,2])) != cells[:,2])
    snapshot=transfer._snapshot(rna,atac,pt,cfg)
    calls=[]
    proxy=SimpleNamespace(**core.__dict__)
    def checked(model,old,new,**kwargs):
        if model.target_id=="p0":
            assert kwargs["changed_inputs"]==(("A",0),)
            for index,key in enumerate(model.input_keys):
                if key[0]=="C":
                    np.testing.assert_array_equal(new[:,index],old[:,index])
            calls.append(True)
        return core.predict_delta(model,old,new,**kwargs)
    proxy.predict_delta=checked
    monkeypatch.setattr(simulation,"core",proxy)
    result=perturbation.propagate_perturbation(["A"],rna,atac,pt,edges,table,screen,cfg,
        transfer_models=genes,tf_model_registry=tfs)
    assert calls and not result["atac_unavailable_mask"].any()
    payload=result["projection_inputs"][0]
    np.testing.assert_array_equal(payload["direct_cell_states"][:,1:],cells[:,1:])
    np.testing.assert_array_equal(payload["direct_bin_states"][:,1:],snapshot["R"][:,1:])
    np.testing.assert_array_equal(result["rna_state_history"][:,1:],snapshot["R"][:,1:])
    expected=np.log1p(np.expm1(cells[:,0])*.5)
    np.testing.assert_array_equal(payload["direct_cell_states"][:,0],expected)
    np.testing.assert_allclose(payload["direct_bin_states"][:,0],snapshot["operator"]@expected,rtol=0,atol=1e-15)
    assert (genes,tfs)==frozen
    cfg=copy.deepcopy(cfg);cfg["perturbation"]["ko_strength"]=0.
    zero=perturbation.propagate_perturbation(["A"],rna,atac,pt,edges,table,screen,cfg,
        transfer_models=genes,tf_model_registry=tfs)
    np.testing.assert_array_equal(zero["rna_state_history"],snapshot["R"])
    np.testing.assert_array_equal(zero["atac_state_history"],snapshot["A"])


def test_strict_core_still_rejects_real_undeclared_shared_input_change():
    rna,atac,pt,_,_,cfg,_,_,tfs=fitted(True,"hard")
    model=core.TargetModel.from_bundle(tfs["targets"]["p0"])
    snapshot=transfer._snapshot(rna,atac,pt,cfg)
    rows=simulation._rows(model,snapshot["context"])
    index={g:i for i,g in enumerate(rna.var_names)}
    old=simulation._inputs(model,rows,snapshot["R"],index)
    new=old.copy();new[:,1]*=.9
    with pytest.raises(ValueError,match="cover all actual changes"):
        core.predict_delta(model,old,new,changed_inputs=(("A",0),),branch_ids=model.branch_ids)
