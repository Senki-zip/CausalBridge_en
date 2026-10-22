"""SYNTHETIC schema/selection fixtures, never joint biological predictions."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("joint_diagnostics", ROOT / "scripts/plot_model_diagnostics.py")
vis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vis)


def identity_rows():
    return dict(target_gene="display_coalition", target_gene_id="joint:coalition",
                target_gene_ids_json=json.dumps(["T_1", "T-2"]),
                target_gene_names_json=json.dumps(['DUP,+_"', 'DUP,+_"']))


def test_joint_json_identity_and_mixed_intervention_rejection(tmp_path):
    rows = [dict(identity_rows(), affected_gene="G", affected_gene_id="G")]
    pd.DataFrame(rows).to_csv(tmp_path / "perturbation_results.csv", index=False)
    labels, targets = vis.labels_and_targets(tmp_path)
    assert targets == {"T_1", "T-2"}
    assert labels["T_1"] == labels["T-2"] == 'DUP,+_"'
    rows.append(dict(rows[0], target_gene_ids_json='["OTHER"]', target_gene_names_json='["OTHER"]'))
    pd.DataFrame(rows).to_csv(tmp_path / "perturbation_results.csv", index=False)
    with pytest.raises(ValueError, match="Multiple interventions"):
        vis.labels_and_targets(tmp_path)


def test_joint_response_top_n_applies_per_tf_and_context(tmp_path, monkeypatch):
    records = []
    archives = {}
    for ctx in ("0", "1"):
        curves = []
        for tf, r2 in (("T_1", .99), ("T-2", .6)):
            for i in range(2):
                model = SimpleNamespace(metrics={"status": "ok", "training_r2": r2 - .01 * i},
                                        x_range=(0., 1.), predict=lambda x: np.asarray(x))
                curves.append(dict(context=ctx, tf=tf, peak=f"P{i}", lag=i, alpha=1.,
                    model=model, bundle={}, input_unit="RNA", output_unit="ATAC"))
        archives[ctx] = ({}, curves, pd.DataFrame({"tf_gene": [r["tf"] for r in curves],
            "peak_id": [r["peak"] for r in curves], "tf_lag": [r["lag"] for r in curves]}))
    def save(fig, *_):
        records.append(fig.axes[0].get_title())
        vis.plt.close(fig)
    monkeypatch.setattr(vis, "save_figure", save)
    old_plot = tmp_path / "tf_response_FIRST_T_1_context_0.png"
    old_plot.write_bytes(b"old split plot")
    vis.response_plots(archives, {"T_1", "T-2"}, {"T_1": "FIRST", "T-2": "SECOND"}, tmp_path, 1, .5)
    selected = pd.read_csv(tmp_path / "tf_response_selected.csv")
    assert not old_plot.exists()
    assert (tmp_path / "legacy_split_joint_plots" / old_plot.name).exists()
    assert len(records) == 2
    assert len(selected) == 4
    assert set(selected.tf) == {"T_1", "T-2"}
    assert all("Joint KO:" in title and "FIRST" in title and "SECOND" in title for title in records)


@pytest.mark.parametrize("shared_peak", [False, True])
def test_joint_trajectory_keeps_both_tf_paths_and_actual_lag(tmp_path, monkeypatch, shared_peak):
    source, output = tmp_path / "source", tmp_path / "figures"
    source.mkdir()
    output.mkdir()
    second_peak = "P1" if shared_peak else "P2"
    pd.DataFrame([dict(identity_rows(), affected_gene="G", tf_name=tf, peak_id=peak,
        cluster="0", tf_lag=lag, peak_input_contribution_mean=weight)
        for tf, peak, lag, weight in (("T_1", "P1", 2, 10.), ("T-2", second_peak, 1, .01))]
    ).to_csv(source / "perturbation_pathways.csv", index=False)
    pd.DataFrame([dict(source_context="0", bin=0, peak_id=peak,
        relation_type="saturating_hybrid_tf_peak", status="ok", delta_native=-.5)
        for peak in sorted({"P1", second_peak})]).to_csv(source / "perturbation_projection_events.csv", index=False)
    history = pd.DataFrame([dict(context="0", branch="a", bin=0, pseudotime=.2,
        gene_id=g, control_state=1., state_value=0. if g != "G" else .5,
        state_unit="rna_log1p_cp10k") for g in ("T_1", "T-2", "G")])
    registry = dict(output_unit="ATAC", context=dict(bin_branch_ids=["a"], bin_centers=[.2]))
    curves = [dict(tf=tf, peak=p, lag=lag, bundle=dict(bounds=[0., 4.], target_rows=[0]))
              for tf, p, lag in (("T_1", "P1", 2), ("T-2", second_peak, 1))]
    monkeypatch.setattr(vis, "validated_peak_baseline", lambda *args: np.array([[1.]]))
    companions = []
    monkeypatch.setattr(vis, 'trajectory_companion_responses', lambda root,path,curve,ctx,labels,out,name:
                        companions.extend((c['tf'], c['peak'], c['lag'], name) for c in curve))
    titles = []
    def save(fig, *_):
        titles.append(fig.axes[0].get_title())
        assert {line.get_color() for line in fig.axes[0].lines} == {"#346ca0", "#ce5145"}
        vis.plt.close(fig)
    monkeypatch.setattr(vis, "save_figure", save)
    vis.trajectory_plots(source, {"0": (registry, curves, None)}, history, ["G"],
        {"T_1", "T-2"}, {}, output, 2, "SYNTHETIC_NOT_BIOLOGICAL.h5ad")
    trace = pd.read_csv(output / "trajectory_trace_data.csv")
    assert set(zip(trace.tf, trace.tf_lag)) == {("T_1", 2), ("T-2", 1)}
    assert (trace.loc[trace.role.eq("TF"), "state_value"] == 0).all()
    assert len(titles) == (1 if shared_peak else 2) and all("Joint KO:" in title for title in titles)
    assert trace.loc[trace.role.eq("TF"), "feature_id"].isin(["T_1", "T-2"]).all()
    assert all("joint_" in name for *_, name in companions)
    assert {(tf,peak,lag) for tf,peak,lag,_ in companions} == {('T_1','P1',2),('T-2',second_peak,1)}


def test_history_rejects_other_intervention_before_gene_filter(tmp_path):
    rows = [dict(identity_rows(), target_gene_ids_json='["OTHER"]',
        target_gene_names_json='["OTHER"]', gene="unused", control_state=1., state_value=0.,
        bin=0, branch="a", pseudotime=.1, source_context="0")]
    pd.DataFrame(rows).to_csv(tmp_path / "perturbation_projection_bin_history.csv", index=False)
    with pytest.raises(ValueError, match="differs from endpoint"):
        vis.load_history(tmp_path, {"G"}, {"T_1", "T-2"})


def test_r_joint_excludes_both_direct_targets_and_checks_all_rows(tmp_path):
    rscript = shutil.which("Rscript")
    if not rscript:
        pytest.skip("Rscript unavailable")
    script = tmp_path / "joint.R"
    script.write_text('''source("visualize_perturbation.R")
library(dplyr)
x <- data.frame(target_gene="joint:display", target_gene_ids_json='["T_1","T-2"]',
 target_gene_names_json='["DUP","DUP"]')
t <- vis_targets(x)
stopifnot(identical(t$ids, c("T_1", "T-2")), identical(vis_label(t$ids), c("DUP","DUP")))
bad <- rbind(x, transform(x, target_gene_ids_json='["OTHER"]', target_gene_names_json='["OTHER"]'))
stopifnot(inherits(try(vis_targets(bad), silent=TRUE), "try-error"))
h <- data.frame(source_context="0", branch="a", bin=0, gene=c("T_1","T-2","G"),
 control_state=c(10,20,1), state_value=c(0,0,.5), state_unit="rna_log1p_cp10k",
 native_effect_unit="delta_log1p_rna_log1p_cp10k", status="ok")
r <- vis_history_bin_effects(h, data.frame(cluster="0", bin=0, branch="a"), t$ids)
stopifnot(r$downstream_abs_sum == .5, r$n_affected_genes == 1,
 r$mean_abs_downstream_effect == .5)
''')
    result = subprocess.run([rscript, str(script)], cwd=ROOT, text=True, capture_output=True,
        env={**os.environ, "CAUSALBRIDGE_VIS_ADAPTERS_ONLY": "1"}, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_joint_companion_keeps_three_tf_models_in_one_figure(tmp_path, monkeypatch):
    curves = [dict(tf=tf, peak="P", lag=i, alpha=.2,
        input_unit="RNA",output_unit="ATAC",model=SimpleNamespace(
            x_range=(0.,1.),predict=lambda x:np.asarray(x),metrics={"training_r2":.8}))
        for i,tf in enumerate(["T1","T2","T3"])]
    seen = []
    def save(fig, output, name):
        seen.append((len(fig.axes),fig._suptitle.get_text(),name))
        vis.plt.close(fig)
    monkeypatch.setattr(vis,"save_figure",save)
    vis.joint_companion_response(curves,"0",{},tmp_path,"joint_response",{"T1","T2","T3"})
    assert seen == [(3,"Joint KO: T1 + T2 + T3","joint_response")]


def test_shared_joint_peak_retains_each_tf_local_lag(tmp_path, monkeypatch):
    source, output = tmp_path/'source', tmp_path/'figures'
    source.mkdir(); output.mkdir()
    pd.DataFrame([dict(identity_rows(),affected_gene='G',tf_name=t,peak_id='P',cluster='0',
        lag_mode='local',tf_lag=np.nan,peak_input_contribution_mean=.3)
        for t in ('T_1','T-2')]).to_csv(source/'perturbation_pathways.csv',index=False)
    pd.DataFrame([dict(source_context='0',bin=b,peak_id='P',relation_type='saturating_hybrid_tf_peak',
        status='ok',delta_native=-.2) for b in range(3)]).to_csv(source/'perturbation_projection_events.csv',index=False)
    history=pd.DataFrame([dict(context='0',branch='a',bin=b,pseudotime=b/2,gene_id=g,
        control_state=1.,state_value=.5,state_unit='rna_log1p_cp10k')
        for g in ('T_1','T-2','G') for b in range(3)])
    bundle=dict(bounds=[0.,4.],support_rows=[0,1,2],input_keys=[['T_1','local'],['T-2','local']],
        selected_lag=[[0,1],[1,0],[2,0]])
    curves=[dict(tf=t,peak='P',lag='local',bundle=bundle) for t in ('T_1','T-2')]
    registry=dict(output_unit='ATAC',context=dict(bin_branch_ids=['a']*3,
        bin_centers=[0.,.5,1.],bin_to_support=[0,1,2]))
    monkeypatch.setattr(vis,'validated_peak_baseline',lambda *a:np.ones((3,1)))
    monkeypatch.setattr(vis,'trajectory_companion_responses',lambda *a:None)
    seen=[]
    def save(fig,*args):
        seen.append(args[-1]);vis.plt.close(fig)
    monkeypatch.setattr(vis,'save_figure',save)
    vis.trajectory_plots(source,{'0':(registry,curves,None)},history,['G'],{'T_1','T-2'}, {},output,1,'raw.h5ad')
    frame=pd.read_csv(output/'trajectory_trace_data.csv')
    assert len(seen)==1 and frame.trajectory_id.nunique()==1
    assert frame.loc[frame.tf.eq('T_1')].groupby('bin').tf_lag.first().tolist()==[0,1,2]
    assert frame.loc[frame.tf.eq('T-2')].groupby('bin').tf_lag.first().tolist()==[1,0,0]
