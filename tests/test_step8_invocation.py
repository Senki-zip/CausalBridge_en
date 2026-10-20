"""Integrity and reuse boundaries of a single Step 8 public invocation."""
import copy
import cProfile

import numpy as np
import pandas as pd
import pytest

from CausalBridge import perturbation
from CausalBridge import saturating_perturbation as pert
from CausalBridge import saturating_transfer as transfer
from test_saturating_transfer import fitted


def invoke(fixture, targets=("A", "B")):
    r,a,pt,e,tf,cfg,table,_,genes,tfs=fixture
    return perturbation.propagate_perturbation(targets,r,a,pt,e,table,tf,cfg,
        transfer_models=genes,tf_model_registry=tfs)


def test_independent_interventions_share_checks_but_match_separate_invocations(fitted):
    fixture=list(fitted)
    fixture[5]=copy.deepcopy(fixture[5])
    fixture[5]["perturbation"]["joint_knockout"]=False
    # Profiling leaves the validator's consumed source signature intact.
    profile=cProfile.Profile()
    together=profile.runcall(invoke,fixture)
    calls=sum(entry.callcount for entry in profile.getstats()
        if entry.code is transfer.validate_registry.__code__)
    assert calls==2
    for target,result in zip(("A","B"),together["saturating_interventions"]):
        separate=invoke(fixture,(target,))["saturating_interventions"][0]
        for name in ("rna_state_history","atac_state_history","control_rna_states","control_atac_states"):
            np.testing.assert_array_equal(result[name],separate[name])
        for name in ("saturating_endpoints","saturating_support_diagnostics","saturating_pathways"):
            pd.testing.assert_frame_equal(result[name],separate[name],check_exact=True)


def test_tampering_after_successful_call_is_validated_again(fitted):
    fixture=list(fitted)
    fixture[-2]=copy.deepcopy(fixture[-2])
    invoke(fixture)
    fixture[-2]["targets"]["B"]["weights"][0]=.125
    with pytest.raises(ValueError):
        invoke(fixture)


def test_caller_registry_mutation_cannot_change_owned_invocation(fitted,monkeypatch):
    fixture=list(fitted)
    fixture[-2]=copy.deepcopy(fixture[-2])
    expected=invoke(fixture)
    verify=pert._verify_wt_metrics
    def mutate_caller(registry,models,snapshot):
        fixture[-2]["targets"]["B"]["weights"][0]=.125
        return verify(registry,models,snapshot)
    monkeypatch.setattr(pert,"_verify_wt_metrics",mutate_caller)
    actual=invoke(fixture)
    np.testing.assert_array_equal(actual["rna_state_history"],expected["rna_state_history"])
    pd.testing.assert_frame_equal(actual["saturating_endpoints"],expected["saturating_endpoints"],check_exact=True)
    with pytest.raises(ValueError):
        invoke(fixture)


def test_caller_cannot_supply_trusted_models(fitted):
    r,a,pt,e,tf,cfg,table,_,genes,tfs=fitted
    with pytest.raises(TypeError):
        pert.propagate_saturating(["A"],r,a,pt,e,table,tf,cfg,
            transfer_models=genes,tf_model_registry=tfs,_prepared={})


@pytest.mark.parametrize("peak",[False,True])
@pytest.mark.parametrize("index",[-2,-1])
def test_public_entry_rejects_forged_registry(fitted,peak,index):
    fixture=list(fitted)
    fixture[index]=copy.deepcopy(fixture[index])
    fixture[index]["schema"]="forged"
    r,a,pt,e,tf,cfg,table,_,genes,tfs=fixture
    public=perturbation.perturb_peak if peak else perturbation.propagate_perturbation
    with pytest.raises(ValueError):
        public("p0" if peak else ["A"],r,a,pt,e,table,tf,cfg,
            transfer_models=genes,tf_model_registry=tfs)


def test_missing_peak_checks_libraries_without_wt_reconstruction(fitted,monkeypatch):
    r,a,pt,e,tf,cfg,table,_,genes,tfs=fitted
    monkeypatch.setattr(pert,"_snapshot",lambda *args,**kwargs:pytest.fail("missing peak has no WT simulation"))
    result=perturbation.perturb_peak("missing",r,a,pt,e,table,tf,cfg,
        transfer_models=genes,tf_model_registry=tfs)
    assert result["error"]=="peak_not_found"
    bad=copy.deepcopy(genes)
    bad["schema"]="forged"
    with pytest.raises(ValueError):
        perturbation.perturb_peak("missing",r,a,pt,e,table,tf,cfg,
            transfer_models=bad,tf_model_registry=tfs)
