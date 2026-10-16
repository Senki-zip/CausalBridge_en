"""Current intervention boundaries use the frozen WT bin plan."""
import copy
import numpy as np
import pytest
from CausalBridge import perturbation, saturating_transfer as transfer
from test_saturating_pipeline import public_fixture


@pytest.mark.parametrize("smooth", ["hard", "gaussian"])
def test_current_public_wrappers_require_complete_frozen_registries(smooth):
    r, a, pt, edges, cfg = public_fixture(smooth)
    cfg["atac_to_rna"]["model_mode"] = "saturating_hybrid"
    for entry, target in ((perturbation.propagate_perturbation, ["TF"]),
                          (perturbation.perturb_peak, "p0")):
        with pytest.raises(ValueError, match="registry"):
            entry(target, r, a, pt, edges, None, None, cfg)


def test_current_public_wrapper_rejects_removed_mode_before_simulation():
    for mode in ("legacy_pair_nn", "joint_gene"):
        cfg = {"atac_to_rna": {"model_mode": mode}}
        with pytest.raises(ValueError, match="removed"):
            perturbation.propagate_perturbation([], None, None, None, None, None, None, cfg)
