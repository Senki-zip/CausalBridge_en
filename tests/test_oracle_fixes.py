"""Small direct-script regression checks for the oracle fixes."""

import tempfile
from pathlib import Path

import pandas as pd

from CausalBridge.io import load_config
from CausalBridge.run import _validate_step5_checkpoint, _validate_step6_checkpoint








def test_legacy_step5_step6_checkpoints_are_rejected():
    try:
        _validate_step5_checkpoint({"transfer_models": {"type": "linear"}})
    except RuntimeError:
        pass
    else:
        raise AssertionError("legacy Step-5 checkpoint was accepted")

    try:
        _validate_step6_checkpoint(pd.DataFrame({"weight": [1.0], "weight_std": [0.1]}))
    except RuntimeError:
        pass
    else:
        raise AssertionError("legacy Step-6 checkpoint was accepted")


def test_io_default_normalization_is_true():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "rna.h5ad").touch()
        (root / "atac.h5ad").touch()
        config_path = root / "config.yaml"
        config_path.write_text(
            "input:\n  rna_h5ad: %s\n  atac_h5ad: %s\n"
            % (root / "rna.h5ad", root / "atac.h5ad")
        )
        assert load_config(str(config_path))["atac_to_rna"]["log_normalize_rna"] is True


if __name__ == "__main__":
    tests = [value for name, value in globals().items()
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"passed {len(tests)} oracle-fix tests")
