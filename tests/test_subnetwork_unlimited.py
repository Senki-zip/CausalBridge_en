"""Current unlimited closure follows downstream relations and counts work."""
from types import SimpleNamespace
import pytest
from CausalBridge.saturating_perturbation import _closure


def test_current_unlimited_closure_handles_cycle_and_disconnected_component():
    model = lambda keys: SimpleNamespace(input_keys=keys)
    genes = {"B": model((("p", 1),)), "KO": model((("q", 1),)),
             "SIDE": model((("side", 1),))}
    tfs = {"p": model((("KO", 2),)), "q": model((("B", 0),)),
           "side": model((("SIDE", 0),))}
    assert _closure(("KO",), None, genes, tfs, "unlimited", work_budget=4) == ({"KO", "B"}, {"p", "q"})
    with pytest.raises(ValueError, match="work-edge budget exceeded"):
        _closure(("KO",), None, genes, tfs, "unlimited", work_budget=3)
