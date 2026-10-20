"""Inference-only plot contracts: no fitting and no writes to result directories."""
import copy
import importlib.util
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("model_visualizations", ROOT / "scripts/plot_model_diagnostics.py")
vis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vis)


def test_layout_reserves_footer_and_external_legend_without_changing_curves(tmp_path):
    fig, ax = vis.plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 1, 20)
    line, = ax.plot(x, x ** 2, label="chr19:20038957-20039851  lag=1  train R²=0.98")
    ax.set(title="ENSG00000159216 → peaks · Cluster 4\nHigh training-fit rank; no validation confidence probability",
           xlabel="ENSG00000159216 input (rna_log1p_cp10k)",
           ylabel="Fitted ENSG00000256771 response (rna_log1p_cp10k)")
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)
    vis.figure_footer(fig, "Saved WT support; extrapolation is not validation.\n" * 2)
    vis.layout_figure(fig)
    renderer = fig.canvas.get_renderer()
    footer = fig._diagnostic_footer.get_window_extent(renderer)
    assert footer.y1 < ax.get_tightbbox(renderer).y0
    assert legend.get_window_extent(renderer).x0 > ax.get_window_extent(renderer).x1
    for artist in (ax.title, ax.xaxis.label, ax.yaxis.label, fig._diagnostic_footer):
        box = artist.get_window_extent(renderer)
        assert fig.bbox.contains(box.x0, box.y0)
        assert fig.bbox.contains(box.x1, box.y1)
    assert "\n(rna_log1p_cp10k)" in ax.get_ylabel()
    np.testing.assert_array_equal(line.get_xdata(), x)
    np.testing.assert_array_equal(line.get_ydata(), x ** 2)
    vis.save_figure(fig, tmp_path, "long_response")


@pytest.mark.parametrize("gene_label", ["ENSG00000256771", "LONG_GENE_SYMBOL_WITH_STABLE_ID_ENSG00000256771"])
def test_layout_long_trajectory_labels_do_not_intersect_neighboring_panels(gene_label):
    fig, axes = vis.plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, label in zip(axes, ("TF: " + gene_label, "Peak: chr19:20038957-20039851", "Gene: " + gene_label)):
        ax.plot([0, 1], [0, 1])
        ax.set_ylabel(label + "\nnormalized_atac_log1p")
    axes[0].set_title("ENSG00000159216 → chr19:20038957-20039851 → ENSG00000256771 · Cluster 4\n"
                      "Saved temporal path; TF lag 0 bin(s); no allocated causal claim")
    axes[-1].set_xlabel("Native pseudotime (lag retained; branches shown separately)")
    vis.layout_figure(fig)
    renderer = fig.canvas.get_renderer()
    assert "chr19:20038957-20039851" in axes[1].get_ylabel().splitlines()
    for upper, lower in zip(axes, axes[1:]):
        assert lower.get_tightbbox(renderer).y1 < upper.get_tightbbox(renderer).y0
    for ax in axes:
        for artist in (ax.title, ax.xaxis.label, ax.yaxis.label):
            if artist.get_text():
                box = artist.get_window_extent(renderer)
                assert fig.bbox.contains(box.x0, box.y0)
                assert fig.bbox.contains(box.x1, box.y1)
    vis.plt.close(fig)


def test_response_pathway_union_aliases_and_legacy_fallback(tmp_path):
    import pandas as pd
    pd.DataFrame({"gene_id": ["G1", "G2"], "gene_name": ["SAME", "SAME"],
                  "display_name": ["SAME_1", "SAME_2"]}).to_csv(tmp_path / "gene_identity.csv", index=False)
    rows = [dict(affected_gene_id="G1", affected_gene="SAME_1", tf_name_id="T1",
                 tf_name="TF", peak_id=f"p{i}", source_context="cluster_2", tf_lag=i)
            for i in range(5)]
    rows.append(dict(rows[0], affected_gene_id="G2", affected_gene="SAME_2", peak_id="other"))
    pd.DataFrame(rows).to_csv(tmp_path / "saturating_pathways.csv", index=False)
    paths, aliases = vis.load_pathways(tmp_path, {})
    selected = vis.requested_pathways(paths, ["SAME_1", "G2"], aliases)
    edges = vis.pathway_edges(selected)
    assert len(edges) == 6
    assert ("2", "T1", "p4", 4) in edges
    with pytest.raises(ValueError, match="ambiguous downstream gene"):
        vis.requested_pathways(paths, ["SAME"], aliases)
    pd.DataFrame(columns=pd.DataFrame(rows).columns).to_csv(tmp_path / "perturbation_pathways.csv", index=False)
    assert vis.load_pathways(tmp_path, {})[0].empty  # Empty canonical export wins.


def test_response_path_filter_keeps_exact_lag_before_ranking(tmp_path, monkeypatch):
    import pandas as pd
    from types import SimpleNamespace

    def curve(peak, lag, r2, tf="T1"):
        model = SimpleNamespace(metrics={"status": "ok", "training_r2": r2},
                                x_range=(0., 1.), predict=lambda x: np.asarray(x))
        return dict(context="2", tf=tf, peak=peak, lag=lag, alpha=1., model=model,
                    bundle={}, input_unit="RNA", output_unit="ATAC")

    curves = [curve("p1", 0, .99), curve("p1", 2, .7), curve("p2", 0, .95),
              curve("p3", 0, .8), curve("p4", 0, .4), curve("p5", 0, .98, "indirect")]
    weights = pd.DataFrame({"tf_gene": [c["tf"] for c in curves],
                            "peak_id": [c["peak"] for c in curves], "tf_lag": [c["lag"] for c in curves]})
    archives = {"2": ({}, curves, weights)}
    def check_response(fig, *_):
        ax = fig.axes[0]
        assert not any("mean" in line.get_label().lower() for line in ax.lines)
        assert len(ax.lines) == len(ax.collections) + 1
        vis.plt.close(fig)
    monkeypatch.setattr(vis, "save_figure", check_response)
    edges = {("2", "T1", "p1", 2), ("2", "T1", "p3", 0),
             ("2", "T1", "p4", 0), ("2", "indirect", "p5", 0)}
    vis.response_plots(archives, {"T1"}, {}, tmp_path, 10, .5, edges)
    selected = pd.read_csv(tmp_path / "tf_response_selected.csv")
    assert set(zip(selected.peak, selected.lag)) == {("p1", 2), ("p3", 0)}
    vis.response_plots(archives, {"T1"}, {}, tmp_path, 1, .5)
    selected = pd.read_csv(tmp_path / "tf_response_selected.csv")
    assert list(zip(selected.peak, selected.lag)) == [("p1", 0)]


def gene_archive_fixture():
    item = dict(response_type="nn", W=[[1.]], bias=[0.], v=[1.], offset=0.,
                mean=[0.], scale=[1.], bounds=[0., 2.], x_range=[.2, 2.],
                input_unit="normalized_atac_log1p", output_unit="log1p_rna_log1p_cp10k",
                metrics=dict(n_rows=12, training_rmse=.1, training_r2=.8, status="ok",
                    solver_converged=False, training_status="fixed_epochs_completed",
                    epochs=300, restarts=1, best_objective=.1, identifiable=False))
    bundle = dict(target_id="G1", schema="saturating_hybrid_nn_target_v1",
                  input_unit=item["input_unit"], output_unit=item["output_unit"],
                  provenance={"response_family": "nn"}, input_keys=[["P1", 1], ["P2", 1]],
                  singles=[item, copy.deepcopy(item)], weights=[.5, .5])
    bundle["fingerprint"] = vis.core._digest(bundle)
    registry = dict(type="saturating_hybrid_peak_gene", schema="hybrid_nn_peak_gene_registry_v1",
                    schema_version="hybrid_nn_peak_gene_registry_v1", model_mode="saturating_hybrid",
                    model_contract_version=2, numerical_signature="historical-source-sha",
                    input_unit=item["input_unit"], output_unit=item["output_unit"],
                    context={"response_family": "nn"}, targets={"G1": bundle})
    registry["fingerprint"] = vis.core._digest(registry)
    return registry


def test_gene_archive_selects_only_requested_saved_components_and_rejects_tampering():
    registry = gene_archive_fixture()
    curves = vis.gene_curve_archive(registry, {("G1", "P1", 1)})
    assert [(r["gene"], r["peak"], r["lag"]) for r in curves] == [("G1", "P1", 1)]
    x = np.array([0., .2, 2.])
    np.testing.assert_allclose(curves[0]["model"].predict(x), 2 / (1 + np.exp(-np.tanh(x))))
    assert vis.gene_curve_archive(registry, {("OTHER", "P1", 1)}) == []
    registry["targets"]["G1"]["weights"][0] = .9
    registry["fingerprint"] = vis.core._digest({k: v for k, v in registry.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        vis.gene_curve_archive(registry, {("G1", "P1", 1)})


def test_peak_gene_plots_without_raw_atac_or_step6_and_never_average(tmp_path, monkeypatch):
    import pickle
    import pandas as pd
    checkpoint = tmp_path / "checkpoints" / "cluster_2"
    checkpoint.mkdir(parents=True)
    with (checkpoint / "step5_atac_to_rna.pkl").open("wb") as stream:
        pickle.dump({"transfer_models": gene_archive_fixture()}, stream)
    paths = pd.DataFrame(dict(context=["2", "2"], tf_id=["T1", "indirect"],
                              gene_id=["G1", "G1"], peak_id=["P1", "P2"]))
    captured = []
    def capture(fig, *_):
        ax = fig.axes[0]
        captured.append(ax)
        assert len(ax.lines) == 3  # One solid, one dashed, one zero-input guide.
        assert len(ax.collections) == 2  # Endpoints and input-zero marker.
        assert not any("mean" in line.get_label().lower() for line in ax.lines)
        assert "log1p_rna_log1p_cp10k" in ax.get_ylabel()
        vis.plt.close(fig)
    monkeypatch.setattr(vis, "save_figure", capture)
    vis.peak_gene_response_plots(tmp_path, paths, {"T1"}, {}, tmp_path, 10, .5)
    selected = pd.read_csv(tmp_path / "peak_gene_response_selected.csv")
    assert selected.peak.tolist() == ["P1"]
    assert len(captured) == 1


@pytest.mark.parametrize("family", ["spline", "saturation"])
def test_gene_archive_other_families_preserve_component_math(family):
    registry = gene_archive_fixture()
    bundle = registry["targets"]["G1"]
    if family == "spline":
        item = constant_spline()
        item.update(input_unit=registry["input_unit"], output_unit=registry["output_unit"])
        schema = "hybrid_spline_peak_gene_registry_v2"
        bundle["schema"] = "saturating_hybrid_spline_target_v2"
    else:
        item = dict(b=.5, a=.2, c=1., s=.5, bounds=[0., 2.], x_range=[0., 2.],
                    metrics=dict(n_rows=12, training_rmse=.1, training_r2=.8, status="ok", termination_status=1),
                    schema=vis.core.CONTINUOUS_SINGLE_SCHEMA,
                    input_unit=registry["input_unit"], output_unit=registry["output_unit"])
        schema = "saturating_peak_gene_registry_v2"
        bundle["schema"] = vis.core.CONTINUOUS_SCHEMA
    registry["schema"] = registry["schema_version"] = schema
    registry["context"]["response_family"] = family
    bundle["provenance"]["response_family"] = family
    bundle["singles"] = [item, copy.deepcopy(item)]
    bundle["fingerprint"] = vis.core._digest({k: v for k, v in bundle.items() if k != "fingerprint"})
    registry["fingerprint"] = vis.core._digest({k: v for k, v in registry.items() if k != "fingerprint"})
    curve = vis.gene_curve_archive(registry, {("G1", "P1", 1)})[0]
    x = np.array([-1., 0., 2., 4.])
    expected = vis.decode_single(item, family, registry["input_unit"], registry["output_unit"]).predict(x)
    np.testing.assert_allclose(curve["model"].predict(x), expected)


def test_umap_export_ignores_reticulate_and_preserves_cell_order(tmp_path):
    import anndata as ad
    import pandas as pd
    import sys
    data = ad.AnnData(np.zeros((3, 2)), obs=pd.DataFrame(
        {"leiden": ["2", "0", "2"]}, index=["cell_z", "cell_a", "cell_b"]))
    data.obsm["X_umap"] = np.array([[4., 3.], [2., 1.], [6., 5.]])
    source = tmp_path / "embedding.h5ad"
    exported = tmp_path / "embedding.csv"
    data.write_h5ad(source)
    subprocess.run([sys.executable, str(ROOT / "scripts/export_umap.py"),
                    str(source), str(exported)], check=True, cwd=tmp_path,
                   env=dict(os.environ, RETICULATE_PYTHON="/nonexistent/uv/python"))
    frame = pd.read_csv(exported)
    assert frame.cell_barcode.tolist() == ["cell_z", "cell_a", "cell_b"]
    assert frame.leiden.tolist() == [2, 0, 2]
    np.testing.assert_array_equal(frame[["UMAP_1", "UMAP_2"]], data.obsm["X_umap"])


def test_peak_baseline_rejects_wrong_original_counts(tmp_path):
    import anndata as ad
    from CausalBridge import atac_normalization as norm
    from scipy import sparse
    data = ad.AnnData(sparse.csr_matrix([[1., 2.], [3., 4.]]))
    prepared = norm.prepare_atac_counts(data)
    metadata = dict(prepared.uns[norm.META], unit="normalized_atac_log1p")
    registry = {"context": {"atac_representation": metadata}}
    source = tmp_path / "wrong.h5ad"
    data.X[0, 0] = 2.
    data.write_h5ad(source)
    with pytest.raises(ValueError, match="differs from saved full-input"):
        vis.validated_peak_baseline(source, registry, [data.var_names[0]])


def test_r_source_resolves_script_directory_from_other_working_directory(tmp_path):
    env = dict(os.environ, CAUSALBRIDGE_VIS_ADAPTERS_ONLY="1")
    script = ROOT / "visualize_perturbation.R"
    caller = tmp_path / "caller.R"
    caller.write_text(f'source("{script}")\ncat(PROJECT_DIR)\n')
    result = subprocess.run(["Rscript", str(caller)], cwd=tmp_path, env=env,
                            text=True, capture_output=True, check=True)
    assert result.stdout.strip() == str(ROOT)


def test_r_python_failure_reports_command_exit_status_and_stderr(tmp_path):
    # Exercise the actual R subprocess boundary without reading or writing results.
    executable = tmp_path / "failing python"
    executable.write_text('#!/bin/sh\necho "diagnostic subprocess failed" >&2\nexit 23\n')
    executable.chmod(0o755)
    output = tmp_path / "output with spaces"
    env = dict(os.environ, CAUSALBRIDGE_VIS_PYTHON=str(executable),
               CAUSALBRIDGE_OUTPUT_DIR=str(output))
    result = subprocess.run(["Rscript", str(ROOT / "visualize_perturbation.R"),
                             "--new-visualizations-only"], cwd=tmp_path, env=env,
                            text=True, capture_output=True)
    assert result.returncode == 23
    log = (output / "model_diagnostics.log").read_text()
    assert "Exit status: 23" in log
    assert "diagnostic subprocess failed" in log
    assert str(ROOT / "scripts/plot_model_diagnostics.py") in log
    assert "diagnostic subprocess failed" in result.stderr
    assert "Python dependency preflight failed" in log


def test_r_python_resolution_ignores_reticulate_and_cleans_python_environment(tmp_path):
    import sys
    rscript = Path(sys.executable).parent / "Rscript"
    if not rscript.exists():
        pytest.skip("Requires R in the test Python's conda environment")
    fake = tmp_path / "python"
    fake.write_text('#!/bin/sh\necho "reticulate Python must not run" >&2\nexit 91\n')
    fake.chmod(0o755)
    env = dict(os.environ, CAUSALBRIDGE_VIS_ADAPTERS_ONLY="1",
               RETICULATE_PYTHON=str(fake), PYTHONHOME=str(tmp_path / "invalid"),
               PYTHONPATH=str(tmp_path / "invalid"),
               PATH=str(tmp_path) + os.pathsep + os.environ["PATH"])
    env.pop("CAUSALBRIDGE_VIS_PYTHON", None)
    code = (f'source("{ROOT / "visualize_perturbation.R"}"); '
            'x <- vis_resolve_python(); stopifnot(x$status == 0L); '
            'cat(x$python, "\\n"); '
            'stopifnot(nzchar(Sys.getenv("PYTHONHOME")))')
    result = subprocess.run([str(rscript), "-e", code], env=env, cwd=tmp_path,
                            text=True, capture_output=True, check=True)
    assert Path(result.stdout.strip()).resolve() == Path(sys.executable).resolve()


def test_r_python_manual_and_environment_override_priority(tmp_path):
    first = tmp_path / "manual python"
    second = tmp_path / "environment python"
    for executable, label in ((first, "manual"), (second, "environment")):
        executable.write_text(f'#!/bin/sh\necho "{label}: ModuleNotFoundError: No module named matplotlib" >&2\nexit 29\n')
        executable.chmod(0o755)
    env = dict(os.environ, CAUSALBRIDGE_VIS_ADAPTERS_ONLY="1",
               CAUSALBRIDGE_VIS_PYTHON=str(second))
    code = (f'source("{ROOT / "visualize_perturbation.R"}"); '
            f'x <- vis_resolve_python("{first}"); '
            'stopifnot(x$status == 29L, x$source == "NEW_VIS_PYTHON"); '
            'cat(x$output, "\\n"); '
            'x <- vis_resolve_python(); '
            'stopifnot(x$status == 29L, x$source == "CAUSALBRIDGE_VIS_PYTHON"); '
            'cat(x$output, "\\n")')
    result = subprocess.run(["Rscript", "-e", code], env=env, cwd=tmp_path,
                            text=True, capture_output=True, check=True)
    assert "manual: ModuleNotFoundError: No module named matplotlib" in result.stdout
    assert "environment: ModuleNotFoundError: No module named matplotlib" in result.stdout


def test_r_python_discovers_named_conda_before_path(tmp_path):
    import sys
    prefix = Path(sys.executable).parent.parent
    if prefix.name.lower() != "causalbridge" or not (prefix / "conda-meta").exists():
        pytest.skip("Requires the CausalBridge conda environment")
    conda = tmp_path / "conda"
    import json
    conda.write_text("#!/bin/sh\ncat <<'JSON'\n" +
                     json.dumps({"envs": [str(prefix)]}) + "\nJSON\n")
    conda.chmod(0o755)
    fake = tmp_path / "python"
    fake.write_text('#!/bin/sh\necho "PATH Python must not run" >&2\nexit 91\n')
    fake.chmod(0o755)
    env = dict(os.environ, CAUSALBRIDGE_VIS_ADAPTERS_ONLY="1",
               CONDA_EXE=str(conda), CONDA_PREFIX=str(tmp_path / "invalid"),
               PATH=str(tmp_path) + os.pathsep + os.environ["PATH"])
    env.pop("CAUSALBRIDGE_VIS_PYTHON", None)
    code = (f'source("{ROOT / "visualize_perturbation.R"}"); '
            f'x <- vis_resolve_python(r_home="{tmp_path}"); '
            'stopifnot(x$status == 0L, x$source == "discovered CausalBridge conda environment"); '
            'cat(x$python)')
    result = subprocess.run([str(prefix / "bin/Rscript"), "-e", code], env=env,
                            text=True, capture_output=True, check=True)
    assert Path(result.stdout.strip()).resolve() == Path(sys.executable).resolve()


def test_failed_plot_does_not_abort_other_plot_types(tmp_path, monkeypatch, capsys):
    import pandas as pd
    called = []
    monkeypatch.setattr(vis, "labels_and_targets", lambda _: ({}, {"T1"}))
    monkeypatch.setattr(vis, "load_archives", lambda *_, **kw: {"0": ({}, [], None)})
    monkeypatch.setattr(vis, "load_history", lambda *_: pd.DataFrame())
    monkeypatch.setattr(vis, "validate_history_axes", lambda history, _: history)

    def fail_response(*_):
        raise ValueError("invalid saved response")

    monkeypatch.setattr(vis, "response_plots", fail_response)
    monkeypatch.setattr(vis, "cluster_top_responses", lambda *args, **kwargs: fail_response())
    monkeypatch.setattr(vis, "diagnostic_scatter", lambda *_: called.append("scatter"))
    monkeypatch.setattr(vis, "trajectory_plots", lambda *_: called.append("trajectory"))
    status = vis.main(["--result-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")])
    assert status == 1
    assert called == ["scatter", "trajectory"]
    captured = capsys.readouterr()
    assert "cluster top 5 responses" in captured.out
    assert "invalid saved response" in captured.err


def constant_spline():
    return dict(response_type="spline", knots=[0.] * 4 + [1.] * 4,
                coefficients=[0.] * 4, x_range=[0., 2.], bounds=[0., 1.],
                tail_scale=.5, config={}, schema="bounded_spline_single_v1",
                input_unit="rna_log1p_cp10k", output_unit="normalized_atac_log1p",
                metrics=dict(status="constant_output", constant_value=.3,
                             training_rmse=0., training_r2=None, n_rows=12))


def test_saved_spline_predicts_without_fit_and_rejects_wrong_units():
    item = constant_spline()
    model = vis.decode_single(item, "spline", "rna_log1p_cp10k", "normalized_atac_log1p")
    np.testing.assert_allclose(model.predict(np.array([-100., 0., 1., 100.])), .3)
    assert "response_type" in item  # Decoder does not mutate the saved bundle.
    with pytest.raises(ValueError, match="unit"):
        vis.decode_single(item, "spline", "wrong", "normalized_atac_log1p")


def test_saved_nn_and_saturation_match_saved_math():
    nn = dict(response_type="nn", W=[[1.]], bias=[0.], v=[1.], offset=0.,
              mean=[0.], scale=[1.], bounds=[0., 1.], x_range=[0., 2.],
              input_unit="rna_log1p_cp10k", output_unit="normalized_atac_log1p",
              metrics=dict(n_rows=12, training_rmse=.1, training_r2=.8, status="ok",
                           solver_converged=False, training_status="fixed_epochs_completed",
                           epochs=300, restarts=1, best_objective=.1, identifiable=False))
    x = np.array([-2., 0., 2.])
    model = vis.decode_single(nn, "nn", "rna_log1p_cp10k", "normalized_atac_log1p")
    np.testing.assert_allclose(model.predict(x), 1 / (1 + np.exp(-np.tanh(x))))
    sat = dict(b=.5, a=.2, c=1., s=.5, bounds=[0., 1.], x_range=[0., 2.],
               metrics=dict(n_rows=12, training_rmse=.1, training_r2=.8, status="ok", termination_status=1))
    expected = vis.core.SingleModel(**sat).predict(x)
    sat.update(schema=vis.core.CONTINUOUS_SINGLE_SCHEMA,
               input_unit="rna_log1p_cp10k", output_unit="normalized_atac_log1p")
    actual = vis.decode_single(sat, "saturation", "rna_log1p_cp10k", "normalized_atac_log1p")
    np.testing.assert_allclose(actual.predict(x), expected)


def test_authentication_rejects_changed_archive():
    payload = dict(schema="example", data=[1, 2])
    payload["fingerprint"] = vis.core._digest(payload)
    vis.authenticated(payload, "fixture")
    changed = copy.deepcopy(payload)
    changed["data"][0] = 7
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        vis.authenticated(changed, "fixture")


def test_input_support_uses_real_wt_span_and_keeps_constant_undefined():
    np.testing.assert_allclose(vis.support_distance(np.array([-1., 1., 3., 5.]), (1., 3.)), [1., 0., 0., 1.])
    actual = vis.support_distance(np.array([0., 1., 2.]), (1., 1.))
    assert np.isnan(actual[0]) and actual[1] == 0 and np.isnan(actual[2])


def archive_fixture():
    bundle = dict(target_id="P1", schema="saturating_hybrid_spline_target_v2",
                  input_unit="rna_log1p_cp10k", output_unit="normalized_atac_log1p",
                  provenance={"response_family": "spline"}, input_keys=[["T1", 1]],
                  singles=[constant_spline()], weights=[1.])
    bundle["fingerprint"] = vis.core._digest(bundle)
    registry = dict(type="saturating_hybrid_tf_peak", schema="hybrid_spline_tf_peak_registry_v2",
                    schema_version="hybrid_spline_tf_peak_registry_v2", model_mode="saturating_hybrid",
                    input_unit="rna_log1p_cp10k", output_unit="normalized_atac_log1p",
                    context={"response_family": "spline"}, targets={"P1": bundle})
    registry["fingerprint"] = vis.core._digest(registry)
    return registry


def test_legacy_archive_is_single_only_and_target_tampering_is_rejected():
    archive = archive_fixture()
    curves = vis.curve_archive(archive, {"T1"})
    assert len(curves) == 1 and curves[0]["archive_mode"] == "legacy_saved_singles_only"
    assert vis.curve_archive(archive, {"OTHER"}) == []
    archive["targets"]["P1"]["weights"][0] = .5
    archive["fingerprint"] = vis.core._digest({k: v for k, v in archive.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="target P1: saved fingerprint mismatch"):
        vis.curve_archive(archive)


def test_current_archive_must_pass_production_validator(monkeypatch):
    from CausalBridge import saturating_transfer
    archive = archive_fixture()
    archive["model_contract_version"] = 2
    archive["fingerprint"] = vis.core._digest({k: v for k, v in archive.items() if k != "fingerprint"})
    called = []
    def reject(registry):
        called.append(registry)
        raise ValueError("production contract rejected")
    monkeypatch.setattr(saturating_transfer, "validate_registry", reject)
    with pytest.raises(ValueError, match="production contract rejected"):
        vis.curve_archive(archive)
    assert called == [archive]


def test_repeated_pathway_aliases_preserve_all_ambiguous_identities(tmp_path):
    import pandas as pd
    frame = pd.DataFrame([
        dict(affected_gene_id='G1', affected_gene='SHARED', affected_gene_name='Label1',
             tf_name_id='T1', tf_name='TF', peak_id='p', cluster='cluster_0'),
        dict(affected_gene_id='G2', affected_gene='SHARED', affected_gene_name='Label2',
             tf_name_id='T1', tf_name='TF', peak_id='p', cluster='cluster_0'),
    ] * 20)
    frame.to_csv(tmp_path / 'perturbation_pathways.csv', index=False)
    paths, aliases = vis.load_pathways(tmp_path, {})
    assert len(paths) == 40
    assert aliases['SHARED'] == {'G1', 'G2'}
    assert aliases['Label1'] == {'G1'} and aliases['Label2'] == {'G2'}
    assert paths.context.tolist() == ['0'] * 40
    with pytest.raises(ValueError, match='ambiguous downstream gene'):
        vis.requested_pathways(paths, ['SHARED'], aliases)


def test_trajectory_reuses_selected_pathways_without_reading_again(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(vis, 'load_pathways', lambda *_:
        pytest.fail('already selected pathway table must not be reloaded'))
    vis.trajectory_plots(tmp_path, {}, pd.DataFrame(), ['G1'], {'T1'}, {},
                         tmp_path, 3, None, selected_pathways=pd.DataFrame())


def test_real_spawn_cluster_archive_and_gene_plot_parity(tmp_path):
    import pickle
    import pandas as pd
    import sys
    for ctx in ('0', '1'):
        directory = tmp_path / 'checkpoints' / f'cluster_{ctx}'
        directory.mkdir(parents=True)
        with (directory / 'step6_rna_to_atac.pkl').open('wb') as stream:
            pickle.dump(dict(tf_model_registry=archive_fixture(),
                tf_weights=pd.DataFrame(dict(tf_gene=['T1'],peak_id=['P1'],tf_lag=[1],weight=[1.]))), stream)
        with (directory / 'step5_atac_to_rna.pkl').open('wb') as stream:
            pickle.dump(dict(transfer_models=gene_archive_fixture()), stream)
    script = tmp_path / 'check_parallel.py'
    script.write_text('''
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scripts import plot_model_diagnostics as v
if __name__ == '__main__':
 root=Path(sys.argv[1])
 serial=v.load_archives(root, {'T1'})
 parallel=v.load_archives(root, {'T1'}, workers=2)
 assert list(serial)==list(parallel)==['0','1']
 for ctx in serial:
  a,b=serial[ctx],parallel[ctx]
  pd.testing.assert_frame_equal(a[2],b[2])
  assert a[0]['context']==b[0]['context']
  for x,y in zip(a[1],b[1]):
   assert {k:z for k,z in x.items() if k!='model'}=={k:z for k,z in y.items() if k!='model'}
   np.testing.assert_array_equal(x['model'].predict(np.arange(4.)),y['model'].predict(np.arange(4.)))
 paths=pd.DataFrame([dict(context=c, tf_id='T1',gene_id='G1',peak_id='P1',peak_lag=1) for c in ['0','1']])
 temporary_directory=v.tempfile.TemporaryDirectory
 def checked_temporary_directory(*args, **kwargs):
  if 'gene-plots' in kwargs.get('prefix',''):
   assert Path(kwargs['dir'])==root/'2'
  return temporary_directory(*args, **kwargs)
 v.tempfile.TemporaryDirectory=checked_temporary_directory
 for count in [1,2]:
  out=root/str(count);out.mkdir()
  v.peak_gene_response_plots(root,paths,{'T1'},{},out,2,.5,workers=count)
 pd.testing.assert_frame_equal(pd.read_csv(root/'1/peak_gene_response_selected.csv'),pd.read_csv(root/'2/peak_gene_response_selected.csv'))
 for image in (root/'1').glob('*.png'):
  assert image.read_bytes()==(root/'2'/image.name).read_bytes()
 top=root/'top';top.mkdir()
 v.cluster_top_responses(root, {}, {}, top, workers=2)
 ranked=pd.read_csv(top/'top5_peak_gene_selected.csv')
 assert len(ranked)==4 and set(ranked.context.astype(str))=={'0','1'}
 assert len(list(top.glob('top5_peak_gene_*.png')))==4
''')
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run([sys.executable, str(script), str(tmp_path)], cwd=ROOT, env=env,
                   check=True, capture_output=True, text=True, timeout=90)


def test_cluster_ranking_uses_finite_r2_without_gallery_threshold():
    from types import SimpleNamespace
    values = [.2, .9, .7, -.1, .4, .3, None, float('nan')]
    curves = [dict(tf='T1', peak=f'P{i}', lag=1, alpha=1.,
                   model=SimpleNamespace(metrics={'training_r2': r2})) for i,r2 in enumerate(values)]
    curves.append(dict(curves[0], peak='zero_alpha', alpha=0.))
    chosen = vis.ranked_response_curves(curves)
    assert [c['peak'] for c in chosen] == ['P1','P2','P4','P5','P0']


def test_requested_peaks_exact_intersection_and_missing_errors():
    import pandas as pd
    paths = pd.DataFrame(dict(gene_id=['G1','G1','G2'], peak_id=['chr1:10-20','chr1:30-40','chr1:10-20']))
    selected = vis.requested_peak_pathways(paths,['chr1:10-20'])
    assert selected.gene_id.tolist()==['G1','G2']
    gene = vis.requested_pathways(paths,['G1'],{'G1':{'G1'},'G2':{'G2'}})
    assert vis.requested_peak_pathways(gene,['chr1:10-20']).gene_id.tolist()==['G1']
    with pytest.raises(ValueError,match='No saved pathway'):
        vis.requested_peak_pathways(paths,['chr1:11-21'])
    assert vis.requested_peak_pathways(paths,[]) is paths


def test_peak_only_entry_infers_genes_and_passes_exact_paths(tmp_path,monkeypatch):
    import pandas as pd
    paths=pd.DataFrame(dict(gene_id=['G1','G2','G3'],tf_id=['T1','T1','OTHER'],
                            peak_id=['P1','P2','P1'],context=['0']*3))
    monkeypatch.setattr(vis,'labels_and_targets',lambda _:({}, {'T1'}))
    monkeypatch.setattr(vis,'load_pathways',lambda *_:(paths,{}))
    monkeypatch.setattr(vis,'load_archives',lambda *a,**k:{})
    monkeypatch.setattr(vis,'load_history',lambda *a:pd.DataFrame())
    monkeypatch.setattr(vis,'validate_history_axes',lambda h,a:h)
    monkeypatch.setattr(vis,'response_plots',lambda *a:None)
    monkeypatch.setattr(vis,'peak_gene_response_plots',lambda *a,**k:None)
    seen=[]
    monkeypatch.setattr(vis,'trajectory_plots',lambda *a,**k:seen.append((a[3],k['selected_pathways'])))
    assert vis.main(['--result-dir',str(tmp_path),'--output-dir',str(tmp_path/'out'),
                     '--trajectory-peaks','P1','--model-diagnostic-plots','false'])==0
    assert seen[0][0]==['G1']
    assert seen[0][1].peak_id.tolist()==['P1']


def test_trajectory_companions_use_exact_path_edges_even_below_gallery_r2(tmp_path, monkeypatch):
    import pickle
    import pandas as pd
    directory = tmp_path/'checkpoints/cluster_2'
    directory.mkdir(parents=True)
    with (directory/'step5_atac_to_rna.pkl').open('wb') as stream:
        pickle.dump({'transfer_models': gene_archive_fixture()}, stream)
    curve = vis.curve_archive(archive_fixture(), {'T1'})[0]
    path = pd.Series(dict(gene_id='G1', peak_id='P1', tf_id='T1', peak_lag=1))
    selected = []
    monkeypatch.setattr(vis, 'individual_response', lambda c, ctx, labels, output, name, layer:
                        selected.append((c,ctx,name,layer)))
    vis.trajectory_companion_responses(tmp_path,path,curve,'2',{},tmp_path,'trajectory_example')
    assert len(selected)==2
    assert selected[0][0]['tf']=='T1' and selected[0][0]['peak']=='P1'
    assert selected[1][0]['gene']=='G1' and selected[1][0]['peak']=='P1'
    assert selected[0][2]=='trajectory_example_tf_response'
    assert selected[1][2]=='trajectory_example_peak_gene_response'


def test_no_gene_top5_includes_both_layers_per_cluster(tmp_path):
    import pickle
    import pandas as pd
    directory = tmp_path/'checkpoints/cluster_2'
    directory.mkdir(parents=True)
    with (directory/'step5_atac_to_rna.pkl').open('wb') as stream:
        pickle.dump({'transfer_models': gene_archive_fixture()}, stream)
    registry = archive_fixture()
    curves = vis.curve_archive(registry, {'T1'})
    # This constant fixture has undefined R² and must not get a fake rank.
    assert vis.ranked_response_curves(curves)==[]
    vis.cluster_top_responses(tmp_path, {'2':(registry,curves,None)}, {}, tmp_path)
    selected = pd.read_csv(tmp_path/'top5_peak_gene_selected.csv')
    assert len(selected)==2
    assert set(selected.gene)=={'G1'}
    assert set(selected.peak)=={'P1','P2'}
    assert len(list(tmp_path.glob('top5_peak_gene_context_2_*.png')))==2


def test_history_axes_cannot_silently_mix_checkpoints():
    import pandas as pd
    registry = dict(input_unit="rna_log1p_cp10k", context=dict(
        bin_centers=[.1, .4], bin_branch_ids=["A", "B"]))
    history = pd.DataFrame(dict(context=["0", "0"], bin=[0, 1], pseudotime=[.1, .4],
                                branch=["A", "B"], state_unit=["rna_log1p_cp10k"] * 2))
    assert len(vis.validate_history_axes(history, {"0": (registry, [], None)})) == 2
    history.loc[1, "branch"] = "A"
    assert vis.validate_history_axes(history, {"0": (registry, [], None)}).empty


def test_triple_trace_uses_saved_peak_delta_bounds_and_missing_gaps(tmp_path, monkeypatch):
    import pandas as pd
    source = tmp_path / "input"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    pd.DataFrame(dict(target_gene=["T1"], affected_gene=["G1"], tf_name=["T1"],
        peak_id=["P1"], cluster=["cluster_0"], source_context=["0"], tf_lag=[1],
        peak_input_contribution_mean=[.1])).to_csv(source / "perturbation_pathways.csv", index=False)
    pd.DataFrame(dict(source_context=["cluster_0", "cluster_0"], bin=[0, 2], peak_id=["P1", "P1"],
        relation_type=["saturating_hybrid_tf_peak"] * 2, status=["ok", "ATAC_physical_clip"],
        delta_native=[-2., 5.])).to_csv(source / "perturbation_projection_events.csv", index=False)
    history = pd.DataFrame([dict(context="0", branch=b, bin=i, pseudotime=t, gene_id=g,
        control_state=1., state_value=.5, state_unit="rna_log1p_cp10k")
        for g in ("T1", "G1") for i, (b, t) in enumerate(zip(["a", "a", "b"], [.1, .4, .8]))])
    registry = dict(output_unit="normalized_atac_log1p", context=dict(
        bin_branch_ids=["a", "a", "b"], bin_centers=[.1, .4, .8]))
    curve = dict(tf="T1", peak="P1", lag=1, bundle=dict(bounds=[0., 4.], target_rows=[0, 1, 2]))
    monkeypatch.setattr(vis, "validated_peak_baseline", lambda *args: np.array([[1.], [2.], [3.]]))
    monkeypatch.setattr(vis, 'trajectory_companion_responses', lambda *args: None)
    vis.trajectory_plots(source, {"0": (registry, [curve], None)}, history, ["G1"], {"T1"}, {}, output, 1, "raw.h5ad")
    traces = pd.read_csv(output / "trajectory_trace_data.csv")
    peak = traces[traces.role == "Peak"]
    np.testing.assert_allclose(peak.control_state, [1., 2., 3.])
    assert peak.state_value.iloc[0] == 0 and np.isnan(peak.state_value.iloc[1]) and peak.state_value.iloc[2] == 4
    assert len(list(output.glob("trajectory_*.png"))) == 1


def test_new_r_controls_do_not_change_existing_pathway_selection():
    import shutil
    if not shutil.which("Rscript"):
        pytest.skip("Rscript unavailable")
    code = '''
      Sys.setenv(CAUSALBRIDGE_VIS_ADAPTERS_ONLY="1")
      source("visualize_perturbation.R")
      args <- c("--trajectory-genes", "NOLC1,CDH1", "--tf-response-top-n", "7",
                "--trajectory-peaks", "chr1:10-20,chr2:30-40",
                "--model-diagnostic-plots", "false", "--result-dir", "/tmp/input",
                "--output-dir", "/tmp/output", "--genes", "OLD", "--new-visualizations-only")
      old <- vis_config(args, result_dir="old", pathway_genes=c("EXISTING"))
      new <- vis_new_config(args)
      stopifnot(old$result_dir == "/tmp/input", old$output_dir == "/tmp/output",
                identical(old$pathway_genes, "OLD"), identical(new$genes, c("NOLC1", "CDH1")),
                new$top_n == 7, !new$diagnostics, new$only, is.null(PATHWAY_GENES))
      stopifnot(identical(new$peaks,c("chr1:10-20","chr2:30-40")),
                is.null(vis_new_config(c("--trajectory-peaks","NULL"))$peaks))
      stopifnot(inherits(try(vis_new_config(c("--tf-response-top-n", "-1")), silent=TRUE), "try-error"))
    '''
    subprocess.run(["Rscript", "-e", code], cwd=ROOT, env=os.environ.copy(), check=True,
                   capture_output=True, text=True)
