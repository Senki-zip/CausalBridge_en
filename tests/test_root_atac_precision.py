"""Root ATAC audits have been removed from current model contracts."""
import inspect
import pytest
from CausalBridge import kinetics, perturbation, saturating_transfer as transfer
from test_saturating_transfer import fixture_data


def test_current_model_has_no_root_scaling_or_public_root_parameter():
    for name in ("_compute_root_atac", "_scale_atac", "_unscale_atac"):
        assert not hasattr(kinetics, name)
    for function in (kinetics.fit_rna_to_atac, transfer.fit_peak_gene,
                     perturbation.propagate_perturbation, perturbation.perturb_peak):
        assert "root_atac" not in inspect.signature(function).parameters


def test_new_registry_rejects_old_audit_contract_without_modifying_inputs():
    r, a, pt, edges, _, cfg = fixture_data()
    _, _, registry = transfer.fit_peak_gene(r, a, pt, edges, cfg)
    assert registry["model_contract_version"] == 2
    assert not any("root" in key for key in registry)
    assert not any("root" in key for key in registry["context"])
    registry["root_atac_reference"] = [0., 0.]
    with pytest.raises(ValueError, match="fields mismatch"):
        transfer.validate_registry(registry)
