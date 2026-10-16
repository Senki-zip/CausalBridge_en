"""Frontier closure preserves the original alternating-round node sets."""
from types import SimpleNamespace
import numpy as np
import pytest
from CausalBridge.saturating_perturbation import _closure


def reference(targets, seed_peak, genes, tfs, depth, tf_keys=None):
    selected_genes=set(targets)
    selected_peaks=set(() if seed_peak is None else (seed_peak,))
    rounds=len(genes)+len(tfs)+1 if depth in (None,"unlimited") else depth
    keys={p:m.input_keys for p,m in tfs.items()} if tf_keys is None else tf_keys
    for _ in range(rounds):
        new_peaks={p for p,inputs in keys.items() if any(g in selected_genes for g,_ in inputs)}
        new_genes={g for g,m in genes.items() if any(p in selected_peaks|new_peaks for p,_ in m.input_keys)}
        updated=(selected_genes|new_genes,selected_peaks|new_peaks)
        if updated==(selected_genes,selected_peaks):break
        selected_genes,selected_peaks=updated
    return selected_genes,selected_peaks


def model(keys):
    return SimpleNamespace(input_keys=tuple(keys))


@pytest.mark.parametrize("depth", [0,1,2,3,None,"unlimited"])
@pytest.mark.parametrize("seed_kind", ["gene","peak","both"])
def test_randomized_reference_equivalence(depth,seed_kind):
    rng=np.random.default_rng(193)
    for _ in range(60):
        # The final component is disconnected; input rows include lag duplicates.
        genes={f"g{i}":model([(f"p{int(rng.integers(6))}",int(rng.integers(3)))
            for _ in range(int(rng.integers(9)))]) for i in range(6)}
        tfs={f"p{i}":model([(f"g{int(rng.integers(6))}",int(rng.integers(3)))
            for _ in range(int(rng.integers(9)))]) for i in range(6)}
        genes["disconnected_g"]=model([("disconnected_p",1)])
        tfs["disconnected_p"]=model([("disconnected_g",0)])
        targets=("g0",) if seed_kind in ("gene","both") else ()
        peak="p0" if seed_kind in ("peak","both") else None
        for keys in (None,{p:tuple(k for k in m.input_keys if k[1]==0) for p,m in tfs.items()}):
            row_count=sum(len(m.input_keys) for m in genes.values())+sum(
                len(k) for k in (keys or {p:m.input_keys for p,m in tfs.items()}).values())
            assert _closure(targets,peak,genes,tfs,depth,tf_keys=keys,work_budget=row_count)==reference(
                targets,peak,genes,tfs,depth,keys)


def test_chain_rounds_cycles_and_duplicate_budget():
    genes={"g1":model([("p0",1),("p0",2)]),"g2":model([("p1",1)])}
    tfs={"p0":model([("g0",0),("g2",1)]),"p1":model([("g1",0)])}
    assert _closure(("g0",),None,genes,tfs,1)==({"g0","g1"},{"p0"})
    assert _closure((),"p0",genes,tfs,1)==({"g1"},{"p0"})
    assert _closure((),"p0",genes,tfs,2)==({"g1","g2"},{"p0","p1"})
    assert _closure(("g0",),None,genes,tfs,"unlimited",work_budget=6)==({"g0","g1","g2"},{"p0","p1"})
    with pytest.raises(ValueError,match="budget=5, scanned=6 reachable relation rows"):
        _closure(("g0",),None,genes,tfs,"unlimited",work_budget=5)
    assert _closure(("absent",),None,genes,tfs,"unlimited",work_budget=0)==({"absent"},set())


def test_large_disconnected_graph_does_not_charge_full_graph_each_round():
    genes={"g1":model([("p0",1)])}
    tfs={"p0":model([("g0",0)])}
    # Repeated relation rows still count individually when reachable.
    genes["unreachable"]=model([("unreachable_peak",1)]*130000)
    rows=sum(len(m.input_keys) for m in genes.values())+sum(len(m.input_keys) for m in tfs.values())
    assert rows==130002 and 2*rows>250000
    expected=reference(("g0",),None,genes,tfs,"unlimited")
    assert _closure(("g0",),None,genes,tfs,"unlimited",work_budget=2)==expected
    with pytest.raises(ValueError,match="budget=1, scanned=2"):
        _closure(("g0",),None,genes,tfs,"unlimited",work_budget=1)
