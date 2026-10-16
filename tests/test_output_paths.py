"""Configured roots are literal user choices across all model/identity modes."""
import copy
import itertools
import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from CausalBridge import io, run
from test_gene_identity_pipeline import fixture, prepare


MODES = [("saturating_hybrid", False), ("saturating_hybrid", True)]
FAMILIES = list(itertools.product(["saturation", "nn"], repeat=2))
PATHS = ["relative results/", "with spaces/gene_ids/joint_gene/",
         "gene_ids/saturating_hybrid/atac_depth_log1p/layers_tf_saturation_gene_nn/atac_depth_log1p/"]


def config(tmp_path, mode, continuous, gene_family, tf_family, identity):
    rna, atac, cfg = fixture(tmp_path)
    cfg["input"]["gene_identity_mode"] = identity
    cfg["atac_to_rna"].update(model_mode=mode, response_model=gene_family)
    cfg["rna_to_atac"]["response_model"] = tf_family
    cfg["preprocess"]["atac_depth_normalization"] = {"enabled": continuous}
    for name in ("rna_h5ad", "atac_h5ad"):
        path = tmp_path / name
        path.touch()  # load_config checks existence; never loads biology here.
        cfg["input"][name] = str(path)
    return rna, atac, cfg


@pytest.mark.parametrize("mode,continuous", MODES)
@pytest.mark.parametrize("gene_family,tf_family", FAMILIES)
@pytest.mark.parametrize("identity", ["legacy", "gtf_gene_id"])
@pytest.mark.parametrize("output", PATHS)
@pytest.mark.parametrize("custom", [False, True])
def test_load_and_runtime_keep_exact_paths(tmp_path, monkeypatch, mode, continuous,
                                         gene_family, tf_family, identity, output, custom):
    monkeypatch.chdir(tmp_path)
    cfg = config(tmp_path, mode, continuous, gene_family, tf_family, identity)[2]
    checkpoint = "custom checkpoints/gene_ids/saturating_hybrid/atac_depth_log1p/layers_tf_nn_gene_nn/" if custom else None
    cfg["output"].update(dir=output, checkpoint_dir=checkpoint)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    loaded = io.load_config(path)
    before = copy.deepcopy(loaded)
    assert loaded["output"]["dir"] == output
    assert loaded["output"]["checkpoint_dir"] == checkpoint
    for first, second in [(run._joint_runtime_config, io._gene_identity_runtime_config),
                          (io._gene_identity_runtime_config, run._joint_runtime_config)]:
        effective = first(loaded)
        assert first(effective) == effective
        composed = second(effective)
        assert second(first(composed)) == composed
        assert effective["output"]["dir"] == output
        assert effective["output"]["checkpoint_dir"] == checkpoint
        assert effective["output"]["cache_dir"] == loaded["output"]["cache_dir"]
        assert effective["atac_to_rna"] == loaded["atac_to_rna"]
        assert effective["rna_to_atac"] == loaded["rna_to_atac"]
        expected = Path(checkpoint) if custom else Path(output) / "checkpoints"
        assert run._checkpoint_dir(effective) == expected
        assert run._checkpoint_dir(effective, "cluster_2") == expected / "cluster_2"
        assert run._checkpoint_dir(loaded) == expected
    assert loaded == before
    assert not Path(output).exists()
    assert path.read_text() == yaml.safe_dump(cfg)


@pytest.mark.parametrize("mode,continuous", MODES)
@pytest.mark.parametrize("gene_family,tf_family", FAMILIES)
@pytest.mark.parametrize("identity", ["legacy", "gtf_gene_id"])
def test_save_results_writes_only_configured_root(tmp_path, monkeypatch, mode, continuous,
                                                gene_family, tf_family, identity):
    monkeypatch.chdir(tmp_path)
    rna, atac, cfg = config(tmp_path, mode, continuous, gene_family, tf_family, identity)
    cfg["output"]["dir"] = PATHS[-1]
    gene_identity = prepare((rna, atac, cfg))[4] if identity == "gtf_gene_id" else None
    frame = pd.DataFrame({"target_gene": ["E3"], "affected_gene": ["E1"], "delta_rna": [.125]})
    results = {"perturbation_results": frame, "perturbation_merged": frame.copy()}
    before = copy.deepcopy(cfg)
    effective = run._joint_runtime_config(cfg)
    io.save_results(results, effective, gene_identity=gene_identity)
    # A second save and the original config must resolve to the same destination.
    io.save_results(results, cfg, gene_identity=gene_identity)
    saved = pd.read_csv(Path(cfg["output"]["dir"]) / "perturbation_results.csv")
    assert saved.delta_rna.tolist() == [.125]
    assert cfg == before
    output = tmp_path / cfg["output"]["dir"]
    assert set(tmp_path.rglob("perturbation_results.csv")) == {
        output / "perturbation_results.csv", output / "intermediate/perturbation_results.csv"}


@pytest.mark.parametrize("mode", ["legacy_pair_nn"])
def test_continuous_validation_retained_for_unsupported_modes(tmp_path, mode):
    cfg = config(tmp_path, mode, True, "nn", "saturation", "legacy")[2]
    with pytest.raises(ValueError, match="requires saturating_hybrid"):
        io._validate_config(cfg)


def test_old_nested_checkpoints_are_never_searched(tmp_path):
    cfg = config(tmp_path, "saturating_hybrid", True, "nn", "saturation", "gtf_gene_id")[2]
    cfg["output"].update(dir=str(tmp_path / "results"), checkpoint_dir=None)
    old = tmp_path / "results/checkpoints/gene_ids/saturating_hybrid/atac_depth_log1p/layers_tf_saturation_gene_nn/cluster_2/step5_atac_to_rna.pkl"
    old.parent.mkdir(parents=True)
    payload = pickle.dumps({"transfer_models": {"type": "nn"}})
    old.write_bytes(payload)
    assert run._load_checkpoint(cfg, "step5_atac_to_rna", "cluster_2") is None
    assert old.read_bytes() == payload
    with pytest.raises(RuntimeError, match="使用新的checkpoint_dir或备份后移走Step5/6不兼容文件"):
        run._validate_step5_checkpoint(pickle.loads(payload), "saturating_hybrid", config=cfg)


@pytest.mark.parametrize("override", [None, "explicit output/gene_ids/"])
@pytest.mark.parametrize("custom_checkpoint", [None, "explicit checkpoint/joint_gene/"])
def test_peak_cli_uses_configured_roots(tmp_path, monkeypatch, override, custom_checkpoint):
    import run_peak_perturbation as cli
    monkeypatch.chdir(tmp_path)
    cfg = {"output": {"dir": "configured output/", "checkpoint_dir": custom_checkpoint},
           "perturbation": {}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    args = ["peak-ko", "--config", str(path), "--peak", "chr1:1-2"]
    if override:
        args += ["--out-dir", override]
    monkeypatch.setattr(sys, "argv", args)
    captured = []
    def fake_pipeline(path):
        captured.append(yaml.safe_load(Path(path).read_text()))
        return {}
    monkeypatch.setattr(run, "run_pipeline", fake_pipeline)
    cli.main()
    assert captured[0]["output"]["dir"] == (override or cfg["output"]["dir"])
    assert captured[0]["output"]["checkpoint_dir"] == custom_checkpoint
    assert path.read_text() == yaml.safe_dump(cfg)
    assert not list(tmp_path.glob("**/PEAK_*"))
