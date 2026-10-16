"""Explicit raw-CP10k intervention decoder; never use the legacy L1 multiplier."""
import numpy as np
import pandas as pd
from scipy import sparse

from .reference_projection import fit_reference_projection, project_reference

SCHEMA = "bin_saturating_effect_v1"


def _matrix(value,name):
    array = value.toarray() if sparse.issparse(value) else np.asarray(value,dtype=float)
    if array.ndim!=2 or not np.isfinite(array).all():
        raise ValueError(f"invalid saturating projection {name}")
    return array


def _decode(payload):
    if payload.get("effect_schema_version")!=SCHEMA or payload.get("intervention_unit")!="cp10k_before_log1p_and_aggregation":
        raise ValueError("explicit saturating/raw-CP10k projection schema required")
    if payload.get("state_unit")!="rna_log1p_cp10k" or payload.get("native_effect_unit")!="delta_log1p_rna_log1p_cp10k":
        raise ValueError("saturating projection unit mismatch")
    if payload.get("invalid_state",False):
        raise ValueError("saturating canonical state is unavailable")
    cells = _matrix(payload["control_cell_states"],"control cells")
    direct = _matrix(payload["direct_cell_states"],"direct cells")
    factors = _matrix(payload["direct_cell_cp10k_factors"],"raw intervention factors")
    base = _matrix(payload["control_bin_states"],"control bins")
    direct_bins = _matrix(payload["direct_bin_states"],"direct bins")
    state = _matrix(payload["perturbed_bin_states"],"perturbed bins")
    native = _matrix(payload["native_bin_effect"],"native bin effects")
    weights = _matrix(payload["cell_bin_weights"],"cell memberships")
    operator = _matrix(payload["bin_cell_operator"],"bin operator")
    features = tuple(payload["features"]);ids = tuple(payload["cell_ids"])
    if len(set(features))!=len(features) or len(set(ids))!=len(ids) or len(ids)!=len(cells) or len(features)!=cells.shape[1]:
        raise ValueError("saturating projection axes must be unique and aligned")
    if cells.shape!=direct.shape or cells.shape!=factors.shape or base.shape!=state.shape or base.shape!=direct_bins.shape or native.shape!=base.shape:
        raise ValueError("saturating projection state dimensions mismatch")
    if weights.shape!=(len(cells),len(base)) or operator.shape!=(len(base),len(cells)):
        raise ValueError("saturating projection operator dimensions mismatch")
    if min(cells.min(),direct.min(),base.min(),direct_bins.min(),state.min(),weights.min(),operator.min(),factors.min())<0 or factors.max()>1:
        raise ValueError("saturating projection violates physical state/factor bounds")
    if not np.allclose(weights.sum(axis=1),1.,atol=1e-10):
        raise ValueError("cell memberships must be normalized")
    expected_direct=np.log1p(np.expm1(cells)*factors)
    if not np.allclose(direct,expected_direct,rtol=1e-10,atol=1e-12):
        raise ValueError("direct RNA does not match raw-CP10k intervention")
    if not np.allclose(operator@cells,base,rtol=1e-10,atol=1e-12) or not np.allclose(operator@direct,direct_bins,rtol=1e-10,atol=1e-12):
        raise ValueError("saturating projection WT/direct aggregate mismatch")
    if not np.allclose(np.log1p(state)-np.log1p(direct_bins),native,rtol=1e-10,atol=1e-12):
        raise ValueError("saturating canonical/native state mismatch")
    for key in ("branch","bins","pseudotime"):
        if len(payload[key])!=len(cells):
            raise ValueError("saturating projection metadata mismatch")
    with np.errstate(over="raise",invalid="raise"):
        altered=direct.copy() if not np.any(native) else np.expm1(np.log1p(direct)+weights@native)
    floor=altered<0
    altered=np.maximum(altered,0.)
    # Direct cell-level clamp is authoritative, including partial KO and zeros.
    for target in payload.get("target_genes",()):
        if target not in features:
            raise ValueError("direct target absent from projection features")
        column=features.index(target);altered[:,column]=direct[:,column]
    if not np.isfinite(altered).all():
        raise ValueError("nonfinite reconstructed saturating cell state")
    return cells,altered,floor


def cell_l0_from_saturating_payload(payload,config=None):
    """Return (cell_ids, gene_ids, normalized CP10k states), not raw UMI counts."""
    _,altered,_=_decode(payload)
    with np.errstate(over="raise",invalid="raise"):
        cp10k=np.expm1(altered)
    if not np.isfinite(cp10k).all():
        raise ValueError("nonfinite reconstructed CP10k state")
    return np.asarray(payload["cell_ids"],dtype=object),tuple(payload["features"]),cp10k


def project_saturating_inputs(inputs,config,context,target):
    cells,altered,floor=_decode(inputs)
    features=tuple(inputs["features"])
    cfg=config.get("perturbation",{}).get("cell_projection",{})
    model=fit_reference_projection(cells,inputs["branch"],inputs["pseudotime"],
        feature_names=features,cell_ids=inputs["cell_ids"],n_components=cfg.get("n_components"),
        n_neighbors=cfg.get("n_neighbors",10),state_unit="rna_log1p_cp10k")
    projected=project_reference(model,cells,altered,feature_names=features,
        cell_ids=inputs["cell_ids"],source_control_ids=inputs["cell_ids"],state_unit="rna_log1p_cp10k")
    projected["source_context"]=context;projected["target_gene"]=target
    projected["effect_schema_version"]=SCHEMA
    projected["source_bin"]=inputs["bins"];projected["source_branch"]=inputs["branch"]
    projected["source_pseudotime"]=inputs["pseudotime"]
    projected["cell_l1_floor_applied_count"]=floor.sum(axis=1)
    projected["context_l1_floor_applied_count"]=int(floor.sum())
    cell_states=pd.DataFrame(altered,columns=features)
    cell_states.insert(0,"cell_id",inputs["cell_ids"])
    cell_states["source_context"]=context;cell_states["target_gene"]=target
    cell_states["state_unit"]="rna_log1p_cp10k"
    base=np.asarray(inputs["control_bin_states"]);state=np.asarray(inputs["perturbed_bin_states"])
    direct=np.asarray(inputs["direct_bin_states"]);native=np.asarray(inputs["native_bin_effect"])
    branches=inputs["bin_branch_ids"];centers=inputs["bin_plan_centers"]
    metadata=[]
    for b in range(len(base)):
        common=dict(source_context=context,target_gene=target,branch=branches[b],bin=b,global_bin=b,
            pseudotime=float(centers[b]),state_unit="rna_log1p_cp10k",
            native_effect_unit="delta_log1p_rna_log1p_cp10k",effect_schema_version=SCHEMA,
            metadata_fingerprint=model.fingerprint,intervention_unit=inputs["intervention_unit"])
        metadata.append(common)
    metadata=pd.DataFrame(metadata)
    # Preserve bin-major/gene-minor order and the complete history schema without
    # constructing a Python dictionary for every bin/gene pair.
    history=metadata.iloc[np.repeat(np.arange(len(base)),len(features))].reset_index(drop=True)
    history["gene"]=np.tile(pd.Series(features).to_numpy(),len(base));history["gene_id"]=history["gene"]
    history["state_value"]=state.ravel();history["control_state"]=base.ravel()
    history["direct_state"]=direct.ravel();history["delta_state"]=(state-base).ravel()
    history["native_delta_L2"]=native.ravel()
    history["history_format"]="complete_WT_and_perturbed_v1"
    events=pd.DataFrame(inputs.get("local_events",[]))
    if not events.empty:
        events["source_context"]=context;events["target_gene"]=target
    return projected,cell_states,history,metadata,events
