"""Export policy is independent of prediction and the intermediate toggle."""
import copy

import pandas as pd
import pytest

from CausalBridge import io


@pytest.mark.parametrize("details", [None, False, True])
@pytest.mark.parametrize("intermediate", [False, True])
def test_canonical_exports_and_optional_details(tmp_path, details, intermediate):
    frame = pd.DataFrame({"value": [1., 2.]})
    results = {key: frame.copy() for key in io.RESULT_CSV_EXPORTS}
    results["perturbation_results"] = pd.DataFrame({"cluster": ["a"], "delta_rna": [.2]})
    results["cluster_results"] = [{"perturbation_results": results["perturbation_results"].copy()}]
    results["actual_intermediate"] = frame.copy()
    results["cell_states"] = frame.copy()
    before = copy.deepcopy(results)
    config = {"output": {"dir": str(tmp_path), "save_intermediate": intermediate},
              "atac_to_rna": {"model_mode": "saturating_hybrid"},
              "perturbation": {"cell_projection": {"enabled": True}}}
    if details is not None:
        config["output"]["export_details"] = details
    io.save_results(results, config)
    for key, filename in io.RESULT_CSV_EXPORTS.items():
        assert (tmp_path / filename).exists() == (details is True or key not in io.DETAIL_EXPORT_KEYS), key
    assert not (tmp_path / "saturating_pathways.csv").exists()
    actual_intermediates = list((tmp_path / "intermediate").glob("*.csv"))
    assert {p.name for p in actual_intermediates} == ({"actual_intermediate.csv"} if intermediate else set())
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(value, before[key])


@pytest.mark.parametrize("invalid", ["false", "true", 0, 1, None])
def test_export_details_requires_boolean_before_writing(tmp_path, invalid):
    directory = tmp_path / "not_created"
    config = {"output": {"dir": str(directory), "export_details": invalid}}
    with pytest.raises(ValueError, match="output.export_details must be a boolean"):
        io.save_results({}, config)
    with pytest.raises(ValueError, match="output.export_details must be a boolean"):
        io._validate_config(config)
    assert not directory.exists()


def test_summary_never_removes_existing_detail_or_duplicate_files(tmp_path):
    names = ["saturating_response_curves.csv", "saturating_pathways.csv",
             "intermediate/projection_events.csv"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("previous results\n")
    before = {name: ((tmp_path / name).stat().st_mtime_ns, (tmp_path / name).read_bytes()) for name in names}
    io.save_results({"saturating_response_curves": pd.DataFrame({"value": [1]})},
                    {"output": {"dir": str(tmp_path), "save_intermediate": True}})
    assert before == {name: ((tmp_path / name).stat().st_mtime_ns, (tmp_path / name).read_bytes()) for name in names}


def test_atac_table_keeps_full_columns_in_canonical_csv(tmp_path):
    frame = pd.DataFrame({"peak_id": ["p1"], "delta_accessibility": [.5], "z_score": [2.7]})
    io.save_results({"atac_changes": frame},
                    {"output": {"dir": str(tmp_path), "save_intermediate": True},
                     "atac_to_rna": {"model_mode": "saturating_hybrid"}})
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "atac_changes.csv"), frame)
    assert not (tmp_path / "intermediate/atac_changes.csv").exists()


@pytest.mark.parametrize("details", [False,True])
def test_local_lag_routes_are_separate_optional_details(tmp_path, details):
    frame = pd.DataFrame([dict(tf_gene="TF",peak_id="P",tf_lag=None,lag_mode="local",
        lag_min=0,lag_max=0,local_routes=[dict(target_bin=0,source_bin=0,selected_lag=0,status="supported"),
            dict(target_bin=1,source_bin=-1,selected_lag=-1,status="no_supported_lag")])])
    before = copy.deepcopy(frame)
    io.save_results({"tf_peak_weights":frame}, {"output":{"dir":str(tmp_path),"export_details":details}})
    summary = pd.read_csv(tmp_path / "tf_peak_weights.csv")
    assert "local_routes" not in summary and summary.tf_lag.isna().all()
    assert (tmp_path / "tf_peak_lag_routes.csv").exists() == details
    if details:
        routes = pd.read_csv(tmp_path / "tf_peak_lag_routes.csv")
        assert routes.source_bin.tolist() == [0,-1]
    pd.testing.assert_frame_equal(frame,before)
