"""Loaded-code identities survive file edits without weakening new-run checks."""
import importlib.util
import linecache
import sys

import pytest

from CausalBridge import run


def test_loaded_function_snapshot_survives_truncated_source(tmp_path):
    path = tmp_path / 'numeric_fixture.py'
    path.write_text('\n' * 100 + 'def calculate(value):\n    return value + 1\n')
    spec = importlib.util.spec_from_file_location('numeric_snapshot_fixture', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        expected = run._numerical_source_hash(module.calculate)

        @run._with_pipeline_source_snapshot
        def calculation():
            assert run._numerical_source_hash(module.calculate) == expected
            path.write_text('# source was replaced during calculation\n')
            linecache.clearcache()
            assert module.calculate(2) == 3
            assert run._numerical_source_hash(module.calculate) == expected

        calculation()
        assert run._PIPELINE_SOURCE_HASHES.get() is None
        # A later invocation must not silently inherit the prior signature.
        with pytest.raises(RuntimeError, match='请重启进程'):
            calculation()
    finally:
        sys.modules.pop(spec.name, None)
        linecache.clearcache()


def test_snapshot_is_reset_after_failure():
    @run._with_pipeline_source_snapshot
    def failing():
        assert run._PIPELINE_SOURCE_HASHES.get() == {}
        raise ValueError('failed run')

    with pytest.raises(ValueError, match='failed run'):
        failing()
    assert run._PIPELINE_SOURCE_HASHES.get() is None


def test_each_invocation_rechecks_changed_source(monkeypatch):
    def numerical(value):
        return value + 1

    @run._with_pipeline_source_snapshot
    def signature():
        return run._numerical_source_hash(numerical)

    before = signature()
    original = run.inspect.getsource
    monkeypatch.setattr(run.inspect, 'getsource', lambda obj:
        'def numerical(value):\n    return value + 2\n' if obj is numerical else original(obj))
    assert signature() != before


def test_pipeline_snapshots_all_stages_before_loading_data(monkeypatch):
    cfg = {'atac_to_rna': {'model_mode': 'saturating_hybrid'},
           'input': {'gene_identity_mode': 'gtf_gene_id'}}
    stages = []
    monkeypatch.setattr(run, 'load_config', lambda _: cfg)
    monkeypatch.setattr(run, '_joint_runtime_config', lambda config: config)
    monkeypatch.setattr(run, '_log_paths', lambda _: None)

    def capture(stage, mode, *, gene_identity_mode):
        assert run._PIPELINE_SOURCE_HASHES.get() is not None
        assert mode == 'saturating_hybrid'
        assert gene_identity_mode == 'gtf_gene_id'
        stages.append(stage)

    def load(_):
        assert stages == ['step4_granger', 'step5_atac_to_rna', 'step6_rna_to_atac']
        raise ValueError('stop before loading production inputs')

    monkeypatch.setattr(run, '_stage_numerical_signatures', capture)
    monkeypatch.setattr(run, 'load_data', load)
    with pytest.raises(ValueError, match='stop before loading'):
        run.run_pipeline('unused')
    assert run._PIPELINE_SOURCE_HASHES.get() is None
