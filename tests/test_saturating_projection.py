import copy
import numpy as np
import pytest

from CausalBridge.saturating_projection import project_saturating_inputs,cell_l0_from_saturating_payload,SCHEMA
from test_saturating_transfer import fitted
from test_saturating_perturbation import execute


def test_projection_new_schema_outputs_complete_control_history(fitted):
    out=execute(fitted);payload=out["projection_inputs"][0]
    projected,cells,history,metadata,events=project_saturating_inputs(payload,fitted[5],"cluster_test","A")
    assert len(projected)==80 and len(cells)==80 and len(history)==60 and len(metadata)==20
    assert set(projected.effect_schema_version)=={SCHEMA}
    assert set(history.history_format)=={"complete_WT_and_perturbed_v1"}
    assert {"control_state","direct_state","native_delta_L2"}.issubset(history)
    np.testing.assert_allclose(np.expm1(cells.A),np.expm1(fitted[0].X.toarray()[:,0])*.5,rtol=1e-12)
    assert not cells.attrs


@pytest.mark.parametrize("field,value",[
    ("effect_schema_version","bin_joint_effect_v1"),
    ("intervention_unit","L1_multiplier"),("state_unit","raw_counts"),
    ("native_effect_unit","delta_CP10k"),("invalid_state",True),
])
def test_old_decoder_units_or_invalid_payload_explicitly_rejected(fitted,field,value):
    payload=copy.deepcopy(execute(fitted)["projection_inputs"][0]);payload[field]=value
    with pytest.raises(ValueError):cell_l0_from_saturating_payload(payload)


def test_projection_rejects_silent_l1_half_instead_of_raw_half(fitted):
    payload=copy.deepcopy(execute(fitted)["projection_inputs"][0])
    payload["direct_cell_states"][:,0]=payload["control_cell_states"][:,0]*.5
    with pytest.raises(ValueError,match="raw-CP10k"):cell_l0_from_saturating_payload(payload)


def test_projection_full_ko_and_zero_ko_exact(fitted):
    for strength in (0.,1.):
        cfg=copy.deepcopy(fitted[5]);cfg["perturbation"]["ko_strength"]=strength
        payload=execute(fitted,cfg)["projection_inputs"][0]
        _,_,cp=cell_l0_from_saturating_payload(payload)
        if strength==1:assert np.array_equal(cp[:,0],np.zeros(len(cp)))
        else:np.testing.assert_allclose(cp,np.expm1(fitted[0].X.toarray()),rtol=1e-12)
