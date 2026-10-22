"""Prefixed single-KO plots must never consume another intervention's tables."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prefixed_diagnostics', ROOT/'scripts/plot_model_diagnostics.py')
vis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vis)


def fixture(root):
    mapping = {'T1': 'RUNX1', 'T2': 'KLF1'}
    (root/'intervention_outputs.json').write_text(json.dumps({'filename_prefixes': mapping}))
    for target, prefix in mapping.items():
        row = dict(target_gene=prefix, target_gene_id=target,
                   target_gene_ids_json=json.dumps([target]),
                   target_gene_names_json=json.dumps([prefix]), affected_gene='G')
        pd.DataFrame([row]).to_csv(root/f'{prefix}_perturbation_results.csv', index=False)
        pd.DataFrame([dict(row, tf_name=target, peak_id='P', source_context='cluster_0')]).to_csv(
            root/f'{prefix}_perturbation_pathways.csv', index=False)
        pd.DataFrame([dict(row, gene='G', control_state=1., state_value=2. if target=='T1' else 3.,
            bin=0, branch='a', pseudotime=.2, source_context='cluster_0')]).to_csv(
            root/f'{prefix}_perturbation_projection_bin_history.csv', index=False)
    # Stale legacy files must not win over the selected prefixed export.
    pd.DataFrame([dict(target_gene='WRONG', affected_gene='G')]).to_csv(root/'perturbation_results.csv', index=False)
    return mapping


def test_manifest_selection_identity_and_history_isolation(tmp_path):
    mapping = fixture(tmp_path)
    assert vis.select_interventions(tmp_path) == mapping
    assert vis.select_interventions(tmp_path, 'RUNX1') == {'T1':'RUNX1'}
    assert vis.select_interventions(tmp_path, 'T2') == {'T2':'KLF1'}
    with pytest.raises(ValueError, match='Unknown'):
        vis.select_interventions(tmp_path, 'missing')
    for target, prefix in mapping.items():
        labels, targets = vis.labels_and_targets(tmp_path, prefix=prefix)
        assert targets == {target} and labels[target] == prefix
        paths, _ = vis.load_pathways(tmp_path, labels, prefix=prefix)
        assert paths.tf_id.tolist() == [target]
        history = vis.load_history(tmp_path, {'G'}, targets, prefix=prefix)
        assert history.state_value.tolist() == [2. if target=='T1' else 3.]
    (tmp_path/'RUNX1_perturbation_results.csv').unlink()
    assert vis.read_table(tmp_path, 'perturbation_results.csv', prefix='RUNX1').empty


def test_auto_dispatch_keeps_figures_and_targets_separate(tmp_path, monkeypatch):
    fixture(tmp_path)
    seen = []
    monkeypatch.setattr(vis, 'load_archives', lambda *a, **k: {})
    monkeypatch.setattr(vis, 'cluster_top_responses', lambda root, archives, labels, output, **k:
                        seen.append(('fit', output.name)))
    monkeypatch.setattr(vis, 'trajectory_plots', lambda root, archives, history, requested, targets,
        labels, output, *a, **k: seen.append((tuple(targets), output.name, k.get('prefix'))))
    status = vis.main(['--result-dir', str(tmp_path), '--output-dir', str(tmp_path/'figures'),
                       '--model-diagnostic-plots', 'false'])
    assert status == 0
    assert seen == [('fit','RUNX1'), (('T1',),'RUNX1','RUNX1'),
                    ('fit','KLF1'), (('T2',),'KLF1','KLF1')]


def test_unsafe_manifest_rejected(tmp_path):
    (tmp_path/'intervention_outputs.json').write_text(json.dumps({'filename_prefixes':{'T':'../escape'}}))
    with pytest.raises(ValueError, match='Invalid'):
        vis.select_interventions(tmp_path)


def test_r_config_and_input_resolution(tmp_path):
    rscript = shutil.which('Rscript')
    if not rscript:
        pytest.skip('Rscript unavailable')
    fixture(tmp_path)
    code = '''
    source(Sys.getenv("VIS_SOURCE"))
    root <- Sys.getenv("VIS_FIXTURE")
    all <- vis_config(c("--result-dir", root), result_dir="unused", output_dir=NULL)
    stopifnot(nrow(all$interventions)==2, all$output_dir==file.path(root,"figures_R"))
    one <- vis_config(c("--result-dir",root,"--intervention","T2", "--workers","2",
                        "--trajectory-peaks","P"), result_dir="unused", output_dir=NULL)
    stopifnot(one$interventions$prefix=="KLF1",one$output_dir==file.path(root,"figures_R","KLF1"))
    stopifnot(vis_result_file("perturbation_results.csv",root,"RUNX1")==file.path(root,"RUNX1_perturbation_results.csv"))
    stopifnot(vis_result_file("tf_peak_weights.csv",root,"RUNX1")==file.path(root,"tf_peak_weights.csv"))
    stopifnot(inherits(try(vis_interventions(root,"missing"),silent=TRUE),"try-error"))
    '''
    env = dict(os.environ, CAUSALBRIDGE_VIS_ADAPTERS_ONLY='1', VIS_SOURCE=str(ROOT/'visualize_perturbation.R'),
               VIS_FIXTURE=str(tmp_path))
    for key in ['CAUSALBRIDGE_RESULT_DIR','CAUSALBRIDGE_OUTPUT_DIR','CAUSALBRIDGE_VIS_INTERVENTION']:
        env.pop(key, None)
    result = subprocess.run([rscript,'-e',code], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_r_auto_dispatch_passes_each_intervention_to_python(tmp_path):
    rscript = shutil.which('Rscript')
    if not rscript:
        pytest.skip('Rscript unavailable')
    fixture(tmp_path)
    executable = tmp_path/'fake python'
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\nexit 0\n')
    executable.chmod(0o755)
    output = tmp_path/'figures with spaces'
    env = dict(os.environ, CAUSALBRIDGE_VIS_PYTHON=str(executable))
    for key in ['CAUSALBRIDGE_VIS_ADAPTERS_ONLY','CAUSALBRIDGE_OUTPUT_DIR','CAUSALBRIDGE_VIS_INTERVENTION']:
        env.pop(key, None)
    result = subprocess.run([rscript,str(ROOT/'visualize_perturbation.R'),
        '--result-dir',str(tmp_path),'--output-dir',str(output),'--new-visualizations-only'],
        env=env,text=True,capture_output=True,timeout=60)
    assert result.returncode == 0, result.stderr
    for target,prefix in [('T1','RUNX1'),('T2','KLF1')]:
        log = (output/prefix/'model_diagnostics.log').read_text()
        assert f'--intervention\n{target}\n' in log
        assert 'Exit status: 0' in log
    assert not (output/'model_diagnostics.log').exists()
