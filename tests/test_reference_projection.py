import numpy as np
import pandas as pd
import pytest

from atac_bridge.reference_projection import fit_reference_projection, project_reference, reconstruct_counterfactual


def _model():
    x = pd.DataFrame([[0, 0, 0], [.1, 0, 0], [5, 5, 0], [5.1, 5, 0]], columns=["g1", "g2", "g3"])
    return fit_reference_projection(x, ["a", "a", "b", "b"], [0, .1, .9, 1],
                                    cell_ids=["a0", "a1", "b0", "b1"], n_neighbors=3,
                                    state_unit="log1p_cpm"), x


def test_exact_source_metadata_and_noop_semantics():
    model, x = _model()
    out = project_reference(model, x, x, source_control_ids=["a0", "a1", "b0", "b1"], state_unit="log1p_cpm")
    assert list(out.source_control_id) == ["a0", "a1", "b0", "b1"]
    assert list(out.source_control_branch) == ["a", "a", "b", "b"]
    assert np.allclose(out.source_control_pseudotime, [0, .1, .9, 1])
    assert np.allclose(out.same_branch_displacement, 0)
    assert list(out.projected_branch) == ["a", "a", "b", "b"]


def test_branch_local_pseudotime_and_gap_abstention():
    model, _ = _model()
    source = pd.DataFrame([[0, 0, 0]], columns=["g1", "g2", "g3"])
    shifted = pd.DataFrame([[.2, 0, 0]], columns=["g1", "g2", "g3"])
    out = project_reference(model, source, shifted, source_control_ids=["a0"], state_unit="log1p_cpm")
    assert out.loc[0, "projected_branch"] == "a"
    assert out.loc[0, "same_branch_displacement"] > 0
    gap = pd.DataFrame([[2.5, 2.5, 0]], columns=shifted.columns)
    gap_out = project_reference(model, gap, state_unit="log1p_cpm")
    assert gap_out.loc[0, "ambiguity_status"] == "ambiguous"
    assert pd.isna(gap_out.loc[0, "projected_pseudotime"])


def test_orthogonal_shift_is_ood():
    model, _ = _model()
    shifted = pd.DataFrame([[0, 0, 100]], columns=["g1", "g2", "g3"])
    out = project_reference(model, shifted, state_unit="log1p_cpm")
    assert out.loc[0, "ood_status"] == "ood"
    assert pd.isna(out.loc[0, "projected_branch"])
    assert out.loc[0, "reconstruction_residual"] > 0


def test_full_rank_pca_ignores_degenerate_residual_null_but_keeps_latent_ood():
    x = pd.DataFrame([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
    ], columns=["g1", "g2", "g3"])
    model = fit_reference_projection(
        x, ["a", "a", "b", "b", "a"], [0, .1, .8, .9, .5],
        n_components=None, n_neighbors=2, state_unit="log1p_cpm",
    )
    assert model.residual_ood_enabled is False
    out = project_reference(model, pd.DataFrame([[100, 100, 100]], columns=x.columns),
                            state_unit="log1p_cpm")
    assert out.loc[0, "ood_status"] == "ood"  # latent-distance OOD remains active


def test_non_degenerate_residual_null_detects_ood():
    x = pd.DataFrame([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
                     columns=["g1", "g2", "g3"])
    model = fit_reference_projection(
        x, ["a", "a", "b", "b"], [0, .1, .8, .9],
        n_components=2, n_neighbors=2, state_unit="log1p_cpm",
    )
    assert model.residual_ood_enabled is True
    out = project_reference(model, pd.DataFrame([[0, 0, 100]], columns=x.columns),
                            state_unit="log1p_cpm")
    assert out.loc[0, "ood_status"] == "ood"


def test_feature_units_and_negative_state_diagnostics():
    model, x = _model()
    with pytest.raises(ValueError, match="state_unit"):
        project_reference(model, x, state_unit="counts")
    with pytest.raises(ValueError, match="nonnegative"):
        project_reference(model, pd.DataFrame([[-1, 0, 0]], columns=x.columns), state_unit="log1p_cpm")
    with pytest.raises(ValueError, match="negative.*diagnostic"):
        reconstruct_counterfactual([[0, 0]], [[1, 0]], [[0, 0]], [[1]],
                                    feature_names=["g1", "g2"], state_unit="log1p_cpm")


def test_feature_order_is_checked():
    model, _ = _model()
    with pytest.raises(ValueError, match="feature order"):
        project_reference(model, pd.DataFrame([[0, 0, 0]], columns=["g2", "g1", "g3"]), state_unit="log1p_cpm")
