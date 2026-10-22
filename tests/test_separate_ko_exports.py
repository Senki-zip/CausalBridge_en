"""Separate interventions must never share response files or ATAC averages."""
import copy
import json

import pandas as pd
import pytest

from CausalBridge.io import save_results
from CausalBridge.run import _merge_atac_changes


def configuration(path, details=True, joint=False):
    return {'output': {'dir': str(path), 'export_details': details, 'save_intermediate': True},
            'atac_to_rna': {'model_mode': 'saturating_hybrid'},
            'perturbation': {'joint_knockout': joint, 'cell_projection': {'enabled': True}}}


def results():
    endpoints = pd.DataFrame({'target_gene': ['id_A', 'id_B'],
        'target_gene_id': ['id_A', 'id_B'], 'target_gene_name': ['RUNX1', 'KLF1'],
        'target_gene_ids_json': ['["id_A"]', '["id_B"]'],
        'delta_rna': [1., -2.], 'cluster': ['0', '0']})
    atac = endpoints.copy()
    atac['peak_id'] = 'chr1:10-20'
    atac['chr'], atac['start'], atac['end'] = 'chr1', 10, 20
    atac['delta_accessibility'] = [.5, -.25]
    atac['is_significant'] = [True, False]
    return {'perturbation_results': endpoints.copy(), 'perturbation_merged': endpoints.copy(),
        'saturating_endpoints': endpoints.copy(), 'saturating_peak_attributions': endpoints.copy(),
        'saturating_tf_peak_contributions': endpoints.copy(), 'projection_history': endpoints.copy(),
        'projection_events': endpoints.copy(), 'projection_bin_metadata': endpoints.copy(),
        'projection_summary': endpoints.copy(), 'cell_projection': endpoints.copy(),
        'saturating_pathways': endpoints.copy(), 'atac_changes': atac.copy(),
        'atac_changes_merged': atac.copy(),
        'saturating_gene_metrics': pd.DataFrame({'gene': ['shared'], 'train_r2': [.7]}),
        'custom_intermediate': endpoints.copy()}


@pytest.mark.parametrize('details', [True, False])
def test_separate_tf_filenames_and_contents(tmp_path, details):
    data = results(); before = copy.deepcopy(data)
    save_results(data, configuration(tmp_path, details))
    for prefix, target, effect in [('RUNX1', 'id_A', 1.), ('KLF1', 'id_B', -2.)]:
        for name in ['perturbation_results', 'perturbation_merged', 'saturating_endpoints',
                     'perturbation_pathways', 'atac_changes', 'atac_changes_merged',
                     'perturbation_projection_summary', 'perturbation_cell_projection']:
            f = pd.read_csv(tmp_path / f'{prefix}_{name}.csv')
            assert f.target_gene_id.tolist() == [target]
            assert f.delta_rna.tolist() == [effect]
            assert not (tmp_path / f'{name}.csv').exists()
        assert (tmp_path / f'{prefix}_atac_changes.bed').exists()
        for name in ['saturating_peak_attributions', 'saturating_tf_peak_contributions',
                     'perturbation_projection_bin_history', 'perturbation_projection_events']:
            assert (tmp_path / f'{prefix}_{name}.csv').exists() == details
        assert (tmp_path / 'intermediate' / f'{prefix}_custom_intermediate.csv').exists()
    assert (tmp_path / 'saturating_gene_metrics.csv').exists()
    manifest = json.loads((tmp_path / 'intervention_outputs.json').read_text())
    assert manifest['filename_prefixes'] == {'id_A': 'RUNX1', 'id_B': 'KLF1'}
    for key in data:
        pd.testing.assert_frame_equal(data[key], before[key])


def test_joint_export_retains_existing_names(tmp_path):
    data = results()
    save_results(data, configuration(tmp_path, joint=True))
    assert len(pd.read_csv(tmp_path / 'perturbation_results.csv')) == 2
    assert not list(tmp_path.glob('RUNX1_*.csv'))
    assert not (tmp_path / 'intervention_outputs.json').exists()


def test_atac_aggregation_never_mixes_interventions():
    def cluster(label, a, b):
        return {'cluster': label, 'atac_changes': pd.DataFrame({
            'target_gene': ['A', 'B'], 'peak_id': ['p', 'p'], 'chr': ['chr1']*2,
            'start': [10]*2, 'end': [20]*2, 'delta_accessibility': [a, b]})}
    merged = _merge_atac_changes([cluster('0', 2., -4.), cluster('1', 6., 8.)],
                                {'0': 10, '1': 30}, {})
    a = merged.set_index('target_gene')
    assert a.loc['A', 'delta_accessibility_signed'] == 5.
    assert a.loc['B', 'delta_accessibility_signed'] == 5.
    assert a.loc['B', 'delta_accessibility'] == 7.
    assert a.n_clusters_observed.tolist() == [2, 2]


def test_filename_collisions_are_disambiguated_and_cannot_escape_directory(tmp_path):
    data = results()
    for key, frame in data.items():
        if 'target_gene_name' in frame:
            frame['target_gene_name'] = ['../A/B', '../A B']
    save_results(data, configuration(tmp_path))
    names = json.loads((tmp_path / 'intervention_outputs.json').read_text())['filename_prefixes']
    assert len(set(names.values())) == 2
    assert all('/' not in name and not name.startswith('.') for name in names.values())
    assert all((tmp_path / (prefix + '_perturbation_results.csv')).exists() for prefix in names.values())


def test_separate_simulations_share_one_prepared_model_and_keep_atac_labels(monkeypatch):
    from CausalBridge import saturating_perturbation as sp
    prepared_calls = []
    registry = object()
    table_keys = ('perturbation_results', 'atac_changes', 'saturating_endpoints',
        'saturating_peak_attributions', 'saturating_tf_peak_contributions',
        'saturating_support_diagnostics', 'l1_floor_diagnostics', 'saturating_gene_metrics',
        'saturating_pathways', 'pathway_edges')
    def prepare(*args, **kwargs):
        assert args[4] is registry
        prepared_calls.append(True)
        def simulate(*, targets):
            target = targets[0]
            calls.append(tuple(targets))
            result = {key: pd.DataFrame({'value': [1.]}) for key in table_keys}
            result.update(target_gene=target, projection_inputs={}, saturating_context_metadata={})
            return result
        return simulate
    calls = []
    monkeypatch.setattr(sp, 'validate_saturating_config', lambda config: None)
    monkeypatch.setattr(sp, '_prepare_simulation', prepare)
    result = sp.propagate_saturating(['RUNX1', 'KLF1'], None, None, None, None, None, None,
        {'perturbation': {'joint_knockout': False}}, transfer_models=registry,
        tf_model_registry=object(), cluster_label='3')
    assert len(prepared_calls) == 1
    assert calls == [('RUNX1',), ('KLF1',)]
    assert result['atac_changes'].target_gene.tolist() == ['RUNX1', 'KLF1']
    assert result['atac_changes'].cluster.tolist() == ['3', '3']


def test_stable_id_prefixes_survive_gene_identity_presentation(tmp_path):
    from test_gene_identity_pipeline import fixture, prepare
    f = fixture(tmp_path)
    _, _, _, config, identity, _ = prepare(f, ['E1', 'E2'])
    # Pick valid stable identities from the fixture, independent of display symbols.
    ids = list(identity.by_id)[:2]
    config.setdefault('perturbation', {})['joint_knockout'] = False
    config['output'].update(dir=str(tmp_path / 'output'), export_details=True)
    frame = pd.DataFrame({'target_gene': ids, 'affected_gene': ids,
                          'target_gene_ids_json': [json.dumps([g]) for g in ids],
                          'delta_rna': [1., -1.]})
    save_results({'saturating_endpoints': frame, 'perturbation_results': frame},
                 config, gene_identity=identity)
    for g in ids:
        prefix = identity.display_labels[g]
        path = tmp_path / 'output' / f'{prefix}_perturbation_results.csv'
        assert path.exists()
        assert pd.read_csv(path).target_gene_id.tolist() == [g]
