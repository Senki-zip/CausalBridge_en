"""Current finite and joint KO closure preserve alternating graph depth."""
from types import SimpleNamespace
from CausalBridge.saturating_perturbation import _closure


def test_joint_knockouts_seed_both_routes_and_respect_finite_depth():
    m = lambda *keys: SimpleNamespace(input_keys=keys)
    genes = {"B": m(("p", 1)), "C": m(("q", 1)), "D": m(("next", 1))}
    tfs = {"p": m(("KO1", 4)), "q": m(("KO2", 0)), "next": m(("B", 1))}
    assert _closure(("KO1", "KO2"), None, genes, tfs, 1) == ({"KO1", "KO2", "B", "C"}, {"p", "q"})
    assert _closure(("KO1", "KO2"), None, genes, tfs, 2) == ({"KO1", "KO2", "B", "C", "D"}, {"p", "q", "next"})
