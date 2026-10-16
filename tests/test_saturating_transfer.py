"""Adapter fixtures fit synthetic WT only, with no pipeline or file output."""
import copy
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from CausalBridge import saturating_hybrid as core
from CausalBridge import saturating_transfer as transfer


def fixture_data(bins=20,branches=1):
    per=4;n=bins*per
    a=np.tile(np.linspace(.5,2.,bins//branches),branches)
    open0=np.clip(np.rint(per*(.15+.8*(a-.5)/1.5)),0,per).astype(int)
    open1=np.roll(open0,1)
    ATAC=np.zeros((n,2))
    for b in range(bins):
        ATAC[b*per:b*per+open0[b],0]=1
        ATAC[b*per:b*per+open1[b],1]=1
    peak_mean=ATAC.reshape(bins,per,2).mean(axis=1)
    x=np.r_[peak_mean[0,0],peak_mean[:-1,0]]
    y2=.5+.15*(x-.5)/(.2+abs(x-.5))
    B=np.expm1(y2)
    RNA=np.column_stack([np.repeat(a,per),np.repeat(B,per),np.repeat(B*.8,per)])
    # Within-bin heterogeneity makes CP KO distinct from multiplying L1.
    RNA[:,0]=np.log1p(np.expm1(RNA[:,0])*np.tile([.5,.8,1.2,1.5],bins))
    ids=[f"c{i}" for i in range(n)]
    rna=ad.AnnData(sparse.csr_matrix(RNA),obs=pd.DataFrame(index=ids),
        var=pd.DataFrame(index=["A","B","C"]))
    atac=ad.AnnData(sparse.csr_matrix(ATAC),obs=rna.obs.copy(),
        var=pd.DataFrame(dict(chr=["1","1"],start=[10,30],end=[20,40]),index=["p0","p1"]))
    labels=np.repeat(np.arange(bins),per)
    pt=pd.DataFrame(dict(pseudotime=np.repeat(np.tile(np.linspace(0,1,bins//branches),branches),per),
        bin=labels,branch=np.repeat(np.arange(branches),n//branches).astype(str)),index=ids)
    pt.attrs["branch_boundaries"]=[(i*bins//branches,(i+1)*bins//branches) for i in range(branches)]
    edges=pd.DataFrame(dict(gene=["B","C"],peak_id=["p0","p1"]))
    tf=pd.DataFrame(dict(tf_gene=["A","B"],peak_id=["p0","p1"],tf_lag=[0,1],weight=[1e6,-1e6]))
    cfg=dict(atac_to_rna=dict(model_mode="saturating_hybrid",log_normalize_rna=True,saturating_hybrid={}),
        pseudotime=dict(smooth_kernel="hard",min_cells_per_bin=4),
        perturbation=dict(perturbation_mode="bin_forward",propagation_rounds=1,ko_strength=.5,
            joint_knockout=True,min_r2_threshold=-1e9,subnetwork_depth="unlimited",
            cell_projection=dict(n_components=2,n_neighbors=5)))
    return rna,atac,pt,edges,tf,cfg


@pytest.fixture(scope="module")
def fitted():
    r,a,pt,e,tf,cfg=fixture_data()
    table,metrics,genes=transfer.fit_peak_gene(r,a,pt,e,cfg)
    tfs=transfer.fit_tf_peak(r,a,pt,tf,cfg)
    return r,a,pt,e,tf,cfg,table,metrics,genes,tfs


def test_fit_synthetic_explicit_units_frozen_support_and_all_rows(fitted):
    r,a,pt,e,tf,cfg,table,metrics,genes,tfs=fitted
    transfer.validate_registry(genes,expected_type=transfer.GENE_TYPE)
    transfer.validate_registry(tfs,expected_type=transfer.TF_TYPE)
    assert genes["model_contract_version"] == 2
    assert "root_scaling_applied" not in genes["context"]
    assert genes["input_unit"]=="raw_atac" and genes["output_unit"]=="log1p_rna_log1p_cp10k"
    assert tfs["input_unit"]=="rna_log1p_cp10k" and tfs["output_unit"]=="raw_atac"
    assert not table.attrs and not metrics.attrs and len(genes["targets"])==2
    assert set(table.alpha)=={1.}
    for model in genes["targets"].values():
        loaded=core.TargetModel.from_bundle(model)
        assert len(loaded.target_rows)==19 and loaded.target_rows[0]==1
        assert len(loaded.singles)==1
    assert len(tfs["targets"]["p0"]["target_rows"])==20
    assert len(tfs["targets"]["p1"]["target_rows"])==19
    assert (metrics.quality_scope=="target_full_training").all()


def test_branch_local_lags_do_not_cross_boundaries():
    r,a,pt,e,tf,cfg=fixture_data(32,2)
    _,_,reg=transfer.fit_peak_gene(r,a,pt,e,cfg)
    m=core.TargetModel.from_bundle(reg["targets"]["B"])
    assert len(m.target_rows)==30 and tuple(m.target_rows)==tuple(range(1,16))*2
    assert tuple(m.source_bins[:,0])==tuple(range(15))*2
    assert set(m.branch_ids)=={"0","1"}
    assert m.diagnostics["n_adjacent_pairs"]==28


def test_tf_regression_weight_is_not_used_for_fitting(fitted):
    r,a,pt,e,tf,cfg,_,_,_,original=fitted
    changed=tf.copy();changed.weight=0.
    actual=transfer.fit_tf_peak(r,a,pt,changed,cfg)
    assert original==actual


def test_missing_policy_registry_load_preserves_identity_and_rejects_relabel(fitted):
    registry = fitted[-2]
    before = copy.deepcopy(registry)
    assert "budget_exhausted_policy" not in registry
    transfer.validate_registry(registry)
    assert registry == before
    changed = copy.deepcopy(registry)
    changed["budget_exhausted_policy"] = "best_finite"
    changed["fingerprint"] = core._digest({k: v for k, v in changed.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="budget policy mismatch"):
        transfer.validate_registry(changed)


@pytest.mark.parametrize("patch",[
    {"atac_to_rna":{"log_normalize_rna":False}},
    {"perturbation":{"propagation_rounds":2}},
    {"perturbation":{"propagation_rounds":True}},
    {"perturbation":{"perturbation_mode":"iterative"}},
    {"perturbation":{"log_accumulation":False}},
    {"perturbation":{"min_lag_per_peak_filter":"false"}},
    {"gene_types":{"A":"TARGET"}},
])
def test_config_rejects_incompatible(patch):
    cfg=fixture_data()[-1]
    for key,values in patch.items():cfg.setdefault(key,{}).update(values)
    with pytest.raises(ValueError):transfer.validate_saturating_config(cfg)


def test_workers_and_unknown_core_options_strict_selector_unchanged():
    cfg=fixture_data()[-1]
    cfg["atac_to_rna"]["saturating_hybrid"] = {"workers":4}
    assert transfer.validate_saturating_config(cfg).workers == 1
    for options in ({"workers":0},{"workers":True},{"workers":1.5},{"unknown":1}):
        cfg["atac_to_rna"]["saturating_hybrid"]=options
        with pytest.raises(ValueError):transfer.validate_saturating_config(cfg)
    for mode in ("legacy_pair_nn","joint_gene",None):
        cfg["atac_to_rna"]["model_mode"]=mode
        with pytest.raises(ValueError):
            transfer.is_saturating_mode(cfg)


def test_axis_mismatch_and_missing_screen_ids_rejected():
    r,a,pt,e,tf,cfg=fixture_data()
    with pytest.raises(ValueError):transfer.fit_peak_gene(r,a[::-1],pt,e,cfg)
    e.gene="absent"
    with pytest.raises(ValueError):transfer.fit_peak_gene(r,a,pt,e,cfg)
    tf.tf_lag=-1
    with pytest.raises(ValueError):transfer.fit_tf_peak(r,a,pt,tf,cfg)




def resign(reg):
    reg.pop("fingerprint",None);reg["fingerprint"]=core._digest(reg)
    return reg


def test_corrupt_and_mixed_registry_rejected(fitted):
    reg=fitted[-2]
    bad=copy.deepcopy(reg);bad["type"]="joint_gene"
    with pytest.raises(ValueError):transfer.validate_registry(resign(bad))
    with pytest.raises(ValueError):transfer.validate_registry(reg,expected_type=transfer.TF_TYPE)
    bad=copy.deepcopy(reg);bad["targets"]["B"]["weights"]=[0.]
    with pytest.raises(ValueError):transfer.validate_registry(resign(bad))
    bad=copy.deepcopy(reg);bad["cell_support"]["weights"][0][0]=.5
    with pytest.raises(ValueError):transfer.validate_registry(resign(bad))


def test_high_lag_insufficient_rows_is_a_failure_record():
    r,a,pt,e,tf,cfg=fixture_data()
    tf.tf_lag=19
    reg=transfer.fit_tf_peak(r,a,pt,tf,cfg)
    assert not reg["targets"] and len(reg["failures"])==2
    assert all(v["status"]=="insufficient_fit_rows" for v in reg["failures"].values())
