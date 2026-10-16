"""Bounded target-parallel saturating adapters on frozen WT bins; no file IO.

Separate gene and TF registries. Old regression weights are screening evidence
only. Core bundles never live in DataFrame.attrs.
"""
import ast
import hashlib
import inspect
import textwrap
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
from importlib.metadata import version

import numpy as np
import pandas as pd
from scipy import sparse

from . import saturating_hybrid as core
from ._wt_aggregation import _weights, _full_wt_aggregates
from .binning_adapter import _hash_array, _hash_matrix
from .terminal_progress import TerminalProgress

GENE_TYPE = "saturating_hybrid_peak_gene"
TF_TYPE = "saturating_hybrid_tf_peak"
GENE_SCHEMA = "saturating_peak_gene_registry_v1"
TF_SCHEMA = "saturating_tf_peak_registry_v1"
GENE_SCHEMA_V2 = "saturating_peak_gene_registry_v2"
TF_SCHEMA_V2 = "saturating_tf_peak_registry_v2"
logger = logging.getLogger(__name__)
_WORKER_SUPPORT = None
_WORKER_CONFIG = None
_WORKER_THREAD_LIMIT = None


def registry_schema(kind, config):
    from .atac_normalization import validate_atac_depth_config
    continuous = validate_atac_depth_config(config)["enabled"]
    if response_settings(config, kind)[0] == "nn":
        return "hybrid_nn_peak_gene_registry_v1" if kind == GENE_TYPE else "hybrid_nn_tf_peak_registry_v1"
    if response_settings(config, kind)[0] == "spline":
        return "hybrid_spline_peak_gene_registry_v2" if kind == GENE_TYPE else "hybrid_spline_tf_peak_registry_v2"
    return (GENE_SCHEMA_V2 if continuous else GENE_SCHEMA) if kind==GENE_TYPE else (TF_SCHEMA_V2 if continuous else TF_SCHEMA)


def is_saturating_mode(config):
    from ._wt_aggregation import validate_model_mode
    return validate_model_mode(config) == "saturating_hybrid"


def response_settings(config, kind):
    is_saturating_mode(config)
    from .nn_hybrid import NNConfig
    layer = "atac_to_rna" if kind == GENE_TYPE else "rna_to_atac"
    options = config.get(layer, {})
    family = options.get("response_model", "saturation")
    if family not in ("nn", "saturation", "spline"):
        raise ValueError(f"{layer}.response_model must be nn, saturation or spline")
    if family == "spline":
        from .spline_hybrid import SplineConfig
        return family, SplineConfig.from_dict(options.get("spline_response", {})).to_dict()
    return family, NNConfig.from_dict(options.get("nn_response", {})).to_dict() if family == "nn" else None


def response_contract(config, kind):
    family, settings = response_settings(config, kind)
    return {"response_model":family, family+"_response":settings} if family in ("nn", "spline") else {}


def response_context(context, config, kind):
    result = dict(context)
    family, settings = response_settings(config, kind)
    if family in ("nn", "spline"):
        result.update(response_family=family, **{family+"_response_config":settings})
        result["fingerprint"] = core._digest({k:v for k,v in result.items() if k != "fingerprint"})
    return result


def budget_exhausted_policy(config):
    options = config.get("atac_to_rna", {}).get("saturating_hybrid", {})
    return core.validate_budget_exhausted_policy(options.get("budget_exhausted_policy", "exclude"))


def saturating_model_settings(config):
    """Semantic adapter settings; the core codec remains unchanged."""
    settings = validate_saturating_config(config).to_dict()
    policy = budget_exhausted_policy(config)
    if policy != "exclude":
        settings["budget_exhausted_policy"] = policy
    return settings


def validate_saturating_config(config):
    from .atac_normalization import validate_atac_depth_config
    validate_atac_depth_config(config)
    options = config.get("atac_to_rna", {})
    settings = dict(options.get("saturating_hybrid", {}))
    core.validate_budget_exhausted_policy(settings.pop("budget_exhausted_policy", "exclude"))
    cfg = core.HybridConfig.from_dict(settings)
    for name in ("atac_to_rna", "rna_to_atac"):
        response_settings(config, GENE_TYPE if name == "atac_to_rna" else TF_TYPE)
    if options.get("log_normalize_rna", True) is not True:
        raise ValueError("saturating_hybrid requires log_normalize_rna=true (L2 output)")
    p = config.get("perturbation", {})
    if p.get("perturbation_mode", "bin_forward") != "bin_forward":
        raise ValueError("saturating_hybrid requires bin_forward")
    if core._integer(p.get("propagation_rounds", 1), "propagation_rounds", 1) != 1:
        raise ValueError("saturating_hybrid requires propagation_rounds=1")
    if p.get("log_accumulation", True) is not True:
        raise ValueError("saturating_hybrid requires log_accumulation=true (native L2 response)")
    types = config.get("gene_types", config.get("input", {}).get("gene_types", {}))
    for name in ("min_lag_per_peak_filter", "joint_knockout"):
        if name in p and not isinstance(p[name], bool):
            raise ValueError(f"saturating_hybrid {name} must be boolean")
    if any(str(kind).upper() != "TF" for kind in types.values()):
        raise ValueError("saturating_hybrid does not support TARGET proxy interventions")
    return replace(cfg, workers=1)


def _initialize_target_worker(weights, fingerprint, settings):
    """Send small support once per spawned worker; full WT matrices stay in parent."""
    global _WORKER_SUPPORT, _WORKER_CONFIG, _WORKER_THREAD_LIMIT
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        _WORKER_THREAD_LIMIT = None
    else:
        _WORKER_THREAD_LIMIT = threadpool_limits(limits=1)
    _WORKER_SUPPORT = core.CellSupport(weights, fingerprint)
    _WORKER_CONFIG = core.HybridConfig.from_dict(settings)


def _fit_target_job(job, support, cfg):
    data = core.TargetData(**job["data"], cell_weights=support.weights,
        cell_support_ref=support.fingerprint)
    try:
        model = core.fit_target(data, job["bounds"], cfg)
    except core.TargetFitFailure as exc:
        return None, exc.diagnostics
    columns = [data.input_keys.index(key) for key in model.input_keys]
    return model.to_bundle(), _net_metrics(model, data.X[:,columns], data.y)


def _worker_fit_target(job):
    if _WORKER_SUPPORT is None or _WORKER_CONFIG is None:
        raise RuntimeError("saturating target worker was not initialized")
    if job["data"]["provenance"].get("response_family") == "nn":
        from .nn_hybrid import initialize_worker_torch
        initialize_worker_torch()
    return _fit_target_job(job, _WORKER_SUPPORT, _WORKER_CONFIG)


def _target_jobs(kind, snapshot, inputs, gi, ai, cfg, policy="exclude"):
    """Gather only this target's bin states, not the full WT snapshot."""
    for target in sorted(inputs):
        keys = tuple(sorted(inputs[target]))
        rows, local, branches, source = _aligned_rows(snapshot, keys)
        if len(rows) < cfg.min_fit_rows:
            yield target, len(keys), None, dict(status="insufficient_fit_rows", n_rows=len(rows))
            continue
        absolute_source = np.column_stack([rows-lag for _, lag in keys])
        matrix, index = (snapshot["A"], ai) if kind == GENE_TYPE else (snapshot["R"], gi)
        X = np.column_stack([matrix[absolute_source[:, i], index[key]] for i, (key, _) in enumerate(keys)])
        y = np.log1p(snapshot["R"][rows, gi[target]]) if kind == GENE_TYPE else snapshot["A"][rows, ai[target]]
        rep = snapshot["context"].get("atac_representation", {"unit": "raw_atac"})
        from .atac_normalization import atac_output_bounds
        bounds = core.rna_output_bounds(y, cfg) if kind == GENE_TYPE else atac_output_bounds(y, rep)
        data = dict(target_id=target, input_keys=keys, X=X, y=y, source_bins=source,
            target_rows=local, branch_ids=branches,
            input_unit=rep["unit"] if kind == GENE_TYPE else "rna_log1p_cp10k",
            output_unit="log1p_rna_log1p_cp10k" if kind == GENE_TYPE else rep["unit"],
            provenance=dict(context_fingerprint=snapshot["context"]["fingerprint"], relation_type=kind),
            support_rows=snapshot["bin_to_support"][rows])
        if policy != "exclude" and snapshot["context"].get("response_family", "saturation") == "saturation":
            data["provenance"]["budget_exhausted_policy"] = policy
        if snapshot["context"].get("response_family") == "nn":
            data["provenance"].update(response_family="nn", nn_response_config=snapshot["context"]["nn_response_config"])
        if snapshot["context"].get("response_family") == "spline":
            data["provenance"].update(response_family="spline", spline_response_config=snapshot["context"]["spline_response_config"])
        yield target, len(keys), dict(data=data, bounds=bounds), None


def _execute_target_jobs(jobs, support, cfg, workers, total, relation_count, kind):
    """Bound pending tasks to 2*workers, merge deterministically, fail transactionally."""
    started = time.monotonic()
    completed = completed_edges = 0
    results = {}
    label = "Step 5 genes" if kind == GENE_TYPE else "Step 6 peaks"
    logger.info("Saturating %s: targets=%d screened_relations=%d workers=%d; preparing first target",
        label, total, relation_count, workers)
    desc = "Step 5 拟合" if kind == GENE_TYPE else "Step 6 拟合"
    with TerminalProgress(total, desc, logger=logger) as progress:
        if workers == 1 or total == 0:
            for target, edges, job, failure in jobs:
                results[target] = (None, failure) if job is None else _fit_target_job(job, support, cfg)
                completed += 1
                completed_edges += edges
                progress.update(completed)
        else:
            if multiprocessing.current_process().daemon:
                raise RuntimeError("saturating target multiprocessing cannot run inside a daemon; use workers=1 there")
            try:
                pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                    initializer=_initialize_target_worker,
                    initargs=(support.weights, support.fingerprint, cfg.to_dict()))
            except Exception as exc:
                raise RuntimeError(f"saturating {label} pool startup failed; no registry produced: {exc}") from exc
            pending = {}
            iterator = iter(jobs)
            exhausted = False
            try:
                while pending or not exhausted:
                    while not exhausted and len(pending) < 2*workers:
                        try:
                            target, edges, job, failure = next(iterator)
                        except StopIteration:
                            exhausted = True
                            break
                        if job is None:
                            results[target] = (None, failure)
                            completed += 1
                            completed_edges += edges
                            progress.update(completed)
                        else:
                            pending[pool.submit(_worker_fit_target, job)] = (target, edges)
                    if pending:
                        done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                        for future in done:
                            target, edges = pending.pop(future)
                            try:
                                results[target] = future.result()
                            except Exception as exc:
                                raise RuntimeError(f"saturating {label} worker failed for {target}; no registry produced: {exc}") from exc
                            completed += 1
                            completed_edges += edges
                            progress.update(completed)
            except BaseException:
                for future in pending:
                    future.cancel()
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)
    logger.info("Saturating %s complete: completed=%d/%d relations=%d/%d elapsed=%.1fs",
        label, completed, total, completed_edges, relation_count, time.monotonic()-started)
    return {target: results[target] for target in sorted(results)}


def _dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix, dtype=float)


def _source_signature(kind, continuous=False, response_family="saturation"):
    consumed = [core.fit_single, core.fit_target, core.TargetFitFailure, core.validate_excluded_relations, core.validate_budget_exhausted_policy, core._fit_group, core._features,
                core._single_values, core._group_values, core._effective_rows,
                core._cell_ess,core._softsign,core.rna_output_bounds,core.HybridConfig,
                core.predict_target, core.predict_delta, core._responses,
                core._query_support, core.validate_cell_support,core.TargetModel.__post_init__,
                core.SingleModel.__post_init__,core.GroupModel.__post_init__,
                _weights, _full_wt_aggregates, _snapshot, _aligned_rows,_fit,_registry,_net_metrics,
                _metadata, _data_fingerprint, _lag, _dense, validate_saturating_config, budget_exhausted_policy, saturating_model_settings, validate_registry,
                fit_peak_gene if kind == GENE_TYPE else fit_tf_peak,
                _initialize_target_worker, _fit_target_job, _worker_fit_target, _target_jobs, _execute_target_jobs]
    if response_family == "nn":
        from . import nn_hybrid as nn
        consumed += [nn.NNConfig, nn.NNSingleModel, nn.NNGroupModel, nn._network, nn._validate_network, nn.initialize_worker_torch,
                     nn._seed, nn._train, nn.fit_singles, nn.fit_group, response_settings, response_contract,
                     response_context, core.TargetModel.to_bundle, core.TargetModel.from_bundle]
    if response_family == "spline":
        from . import spline_hybrid as spline
        consumed += [obj for obj in vars(spline).values() if
                     (inspect.isfunction(obj) or inspect.isclass(obj)) and getattr(obj,"__module__",None)==spline.__name__]
        consumed += [response_settings, response_contract, response_context, core.TargetModel.to_bundle, core.TargetModel.from_bundle]
    if continuous:
        from . import atac_normalization as normalization
        consumed += [core.TargetModel.to_bundle, core.TargetModel.from_bundle]
        consumed += [obj for obj in vars(normalization).values() if inspect.isfunction(obj)
                     and getattr(obj,"__module__",None)==normalization.__name__]
    hashes = {}
    for function in consumed:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
        hashes[function.__module__ + "." + function.__qualname__] = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    return dict(functions=hashes, versions={p: version(p) for p in (("numpy", "scipy", "torch") if response_family == "nn" else ("numpy", "scipy"))},
                core_schema=nn.SCHEMA if response_family == "nn" else spline.SCHEMA if response_family == "spline" else core.SCHEMA,registry_type=kind,
                **(dict(spline_single_schema=spline.SINGLE_SCHEMA, spline_degree=spline.DEGREE,
                        spline_group_schema=spline.GROUP_SCHEMA,
                        spline_quadrature_order=spline.QUADRATURE_ORDER) if response_family == "spline" else {}),
                **(dict(continuous_target_schema=core.CONTINUOUS_SCHEMA,
                    continuous_single_schema=core.CONTINUOUS_SINGLE_SCHEMA,
                    continuous_group_schema=core.CONTINUOUS_GROUP_SCHEMA,
                    representation_schema=normalization.VIEW_SCHEMA,representation_unit=normalization.UNIT,
                    source_schema=normalization.SOURCE_SCHEMA,qc_schema=normalization.QC_SCHEMA,
                    count_proof_column=normalization.COUNT_PROOF,
                    denominator_kind=normalization.DENOMINATOR_KIND,
                    raw_layer=normalization.RAW_LAYER,depth_column=normalization.DEPTH,
                    registry_schema=GENE_SCHEMA_V2 if kind==GENE_TYPE else TF_SCHEMA_V2) if continuous else {}))


def _metadata(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return _metadata(value.tolist())
    if isinstance(value,dict):return {k:_metadata(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_metadata(v) for v in value]
    return core._plain(value)


def _data_fingerprint(rna, atac):
    digest = hashlib.sha256(b"saturating-WT-data-v1\0")
    for adata in (rna, atac):
        _hash_array(digest, adata.obs_names.to_numpy())
        _hash_array(digest, adata.var_names.to_numpy())
        _hash_matrix(digest, adata.X)
        for field in ("chr", "start", "end"):
            _hash_array(digest, adata.var[field].to_numpy() if field in adata.var else np.array([]))
    from .atac_normalization import atac_input_contract
    representation = atac_input_contract(atac)
    if representation is not None:
        digest.update(core._digest(representation).encode())
    return digest.hexdigest()


def _snapshot(rna, atac, pt, config, bin_plan=None):
    """Use existing legacy/frozen-plan aggregation, masking branch crossings."""
    cells = tuple(map(str, rna.obs_names))
    genes, peaks = tuple(map(str, rna.var_names)), tuple(map(str, atac.var_names))
    if cells != tuple(map(str, atac.obs_names)) or cells != tuple(map(str, pt.index)) or len(set(cells)) != len(cells):
        raise ValueError("saturating adapters require identical unique ordered cell axes")
    if len(set(genes)) != len(genes) or len(set(peaks)) != len(peaks):
        raise ValueError("saturating adapters require unique feature axes")
    from .atac_normalization import representation_for_config, UNIT
    representation = representation_for_config(atac,config)
    for matrix, unit in ((rna.X, "rna_log1p_cp10k"), (atac.X, representation["unit"])):
        values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
        if not np.isfinite(values).all():
            raise ValueError("WT matrices must be finite")
        core._unit_domain(values, unit)
    labels = core._array(pt["bin"].to_numpy(), "WT bin labels", 1, True)
    times = core._array(pt["pseudotime"].to_numpy(), "WT pseudotime", 1)
    if len(labels) != len(cells) or not len(labels) or (labels < 0).any():
        raise ValueError("invalid WT bin labels")
    if bin_plan is not None:
        from .binning_adapter import validate_bin_plan, plan_branch_boundaries
        validate_bin_plan(bin_plan, pt, rna.obs_names)
        bins = bin_plan.n_bins
        boundaries = plan_branch_boundaries(bin_plan)
    else:
        bins = int(labels.max()) + 1
        boundaries = pt.attrs.get("branch_boundaries") or [(0, bins)]
    boundaries = tuple((core._integer(a,"branch start"),core._integer(b,"branch end",1)) for a,b in boundaries)
    if not boundaries or boundaries[0][0] != 0 or boundaries[-1][1] != bins or any(b <= a or (i and a != boundaries[i-1][1]) for i,(a,b) in enumerate(boundaries)):
        raise ValueError("invalid contiguous branch bin boundaries")
    kernel, centers = _weights(pt, bins, config, bin_plan)
    bin_branches = []
    for bid,(start,end) in enumerate(boundaries):
        mask = (labels >= start) & (labels < end)
        if not mask.any():
            raise ValueError("empty WT branch")
        branch = str(bid)
        if "branch" in pt:
            names = tuple(pd.unique(pt.loc[mask,"branch"].astype(str)))
            if len(names) != 1:
                raise ValueError("WT branch labels disagree with bin boundaries")
            branch = names[0]
        bin_branches.extend([branch]*(end-start))
        kernel[start:end] = kernel[start:end].multiply(mask.astype(float)[None,:])
    smooth = config.get("pseudotime",{}).get("smooth_kernel","hard")
    A,R,valid = _full_wt_aggregates(kernel,atac.X,rna.X,bin_plan,smooth=smooth)
    if not valid.any():
        raise ValueError("no covered WT bins")
    if bin_plan is not None:
        operator = sparse.csr_matrix(bin_plan.aggregate_weights,copy=True)
        # The plan must already enforce the same branch mask.
        for start,end in boundaries:
            mask = (labels >= start)&(labels < end)
            if operator[start:end][:,~mask].nnz:
                raise ValueError("frozen operator crosses WT branches")
    else:
        mass = np.asarray(kernel.sum(axis=1)).ravel()
        operator = kernel.copy().tocsr()
        denominator = mass+(1e-10 if smooth=="gaussian" else 0.)
        entry_mass = np.repeat(denominator,np.diff(operator.indptr))
        operator.data = np.divide(operator.data,entry_mass,out=np.zeros_like(operator.data),where=entry_mass>0)
    W = _dense(operator)
    if representation["unit"]==UNIT:
        A=_dense(operator@atac.X)
    normalized = W[valid]/W[valid].sum(axis=1,keepdims=True)
    support = core.CellSupport(normalized)
    bin_to_support = np.full(bins,-1,dtype=int);bin_to_support[valid] = np.arange(valid.sum())
    cell_weights = kernel.T.tocsr()
    cell_mass = np.asarray(cell_weights.sum(axis=1)).ravel()
    if (cell_mass <= 0).any():
        raise ValueError("cell reconstruction has uncovered WT cells")
    cell_weights.data /= np.repeat(cell_mass,np.diff(cell_weights.indptr))
    context = dict(cell_ids=list(cells),gene_ids=list(genes),peak_ids=list(peaks),
        n_bins=bins,bin_labels=labels.tolist(),cell_times=times.tolist(),
        cell_branch_ids=(pt.branch.astype(str).tolist() if "branch" in pt else [bin_branches[b] for b in labels]),
        bin_branch_ids=bin_branches,branch_boundaries=[list(v) for v in boundaries],
        bin_centers=np.asarray(centers).tolist(),operator=W.tolist(),
        bin_to_support=bin_to_support.tolist(),valid_bins=valid.tolist(),
        data_fingerprint=_data_fingerprint(rna,atac),
        gene_identity=_metadata(rna.uns.get("gene_identity",{})),
        clustering_reference=_metadata(rna.uns.get("clustering_reference",{})),
        bin_plan_fingerprint=getattr(bin_plan,"fingerprint",None),
        aggregate_unit="RNA_L1/raw_ATAC")
    context["fingerprint"] = core._digest(context)
    if representation["unit"]==UNIT:
        context["atac_representation"] = representation
        context["aggregate_unit"] = "RNA_L1/normalized_atac_log1p"
        context["fingerprint"] = core._digest({k:v for k,v in context.items() if k!="fingerprint"})
    return dict(context=context,R=R,A=A,operator=operator,cell_bin_weights=cell_weights,
                support=support,valid=valid,boundaries=boundaries,bin_to_support=bin_to_support)


def _aligned_rows(snapshot, keys):
    rows,local,branches,sources = [],[],[],[]
    maxlag = max(lag for _,lag in keys)
    for start,end in snapshot["boundaries"]:
        for target in range(start+maxlag,end):
            source = [target-lag for _,lag in keys]
            if snapshot["valid"][target] and all(snapshot["valid"][s] for s in source):
                rows.append(target);local.append(target-start)
                sources.append([s-start for s in source])
                branches.append(snapshot["context"]["bin_branch_ids"][target])
    return np.array(rows,dtype=int),np.array(local,dtype=int),tuple(branches),np.array(sources,dtype=int).reshape(-1,len(keys))


def _lag(value):
    number = core._number(value,"screened TF lag")
    if number < 0 or number != int(number):
        raise ValueError("screened TF lag must be a nonnegative integer")
    return int(number)


def _net_metrics(model, X, y):
    prediction = core.predict_target(model,X,branch_ids=model.branch_ids).values
    values = core._metrics(y,prediction)
    statuses = [s.metrics["status"] for s in model.singles]
    status = "budget_exhausted" if "budget_exhausted" in statuses else model.status
    return dict(train_r2=values["training_r2"],training_rmse=values["training_rmse"],
        n_rows=len(y),status=status,quality_scope="target_full_training",validation_performed=False,
        single_fit_statuses=statuses)


def _registry(kind,snapshot,cfg,targets,metrics,failures,policy="exclude"):
    context = snapshot["context"]
    representation=context.get("atac_representation",{})
    continuous=representation.get("unit")=="normalized_atac_log1p"
    schema=(GENE_SCHEMA_V2 if kind==GENE_TYPE else TF_SCHEMA_V2) if continuous else (GENE_SCHEMA if kind==GENE_TYPE else TF_SCHEMA)
    family = context.get("response_family", "saturation")
    if family == "nn":
        schema = "hybrid_nn_peak_gene_registry_v1" if kind == GENE_TYPE else "hybrid_nn_tf_peak_registry_v1"
    if family == "spline":
        schema = "hybrid_spline_peak_gene_registry_v2" if kind == GENE_TYPE else "hybrid_spline_tf_peak_registry_v2"
    atac_unit="normalized_atac_log1p" if continuous else "raw_atac"
    registry = dict(type=kind,schema=schema,
        schema_version=schema,model_mode="saturating_hybrid",
        targets=targets,target_metrics=metrics,failures=failures,config=cfg.to_dict(),context=context,
        cell_support=dict(weights=snapshot["support"].weights.tolist(),fingerprint=snapshot["support"].fingerprint),
        input_unit=atac_unit if kind==GENE_TYPE else "rna_log1p_cp10k",
        output_unit="log1p_rna_log1p_cp10k" if kind==GENE_TYPE else atac_unit,
        numerical_signature=_source_signature(kind,continuous,family),
        model_contract_version=2,
        binning_metadata=dict(n_bins=context["n_bins"],bin_plan_centers=context["bin_centers"],
            branch_boundaries=context["branch_boundaries"],bin_plan_fingerprint=context["bin_plan_fingerprint"],
            binning_data_fingerprint=context["data_fingerprint"],binning_model_contract_domain=kind))
    if policy != "exclude" and family == "saturation":
        registry["budget_exhausted_policy"] = policy
    registry["fingerprint"] = core._digest(registry)
    validate_registry(registry)
    return registry


def _fit(kind,rna,atac,pt,relations,config,bin_plan=None):
    cfg = validate_saturating_config(config)
    settings = dict(config.get("atac_to_rna", {}).get("saturating_hybrid", {}))
    policy = core.validate_budget_exhausted_policy(settings.pop("budget_exhausted_policy", "exclude"))
    requested_workers = core.HybridConfig.from_dict(settings).workers
    snapshot = _snapshot(rna,atac,pt,config,bin_plan)
    snapshot["context"] = response_context(snapshot["context"], config, kind)
    genes,peaks = snapshot["context"]["gene_ids"],snapshot["context"]["peak_ids"]
    gi,ai = {v:i for i,v in enumerate(genes)},{v:i for i,v in enumerate(peaks)}
    inputs = {}
    if kind==GENE_TYPE:
        if not relations.empty and not {"gene","peak_id"}.issubset(relations):
            raise ValueError("peak-gene screen requires gene/peak_id")
        for row in relations.itertuples(index=False):
            gene,peak = str(getattr(row,"gene_id",row.gene)),str(row.peak_id)
            if gene not in gi or peak not in ai:
                raise ValueError("screened peak-gene axis absent from WT inputs")
            if hasattr(row,"lag") and _lag(row.lag)!=1:
                raise ValueError("saturating peak-gene fitting requires lag=1")
            inputs.setdefault(gene,set()).add((peak,1))
    else:
        if not relations.empty and not {"tf_gene","peak_id","tf_lag"}.issubset(relations):
            raise ValueError("TF screen requires tf_gene/peak_id/tf_lag")
        for row in relations.itertuples(index=False):
            tf,peak = str(getattr(row,"tf_gene_id",row.tf_gene)),str(row.peak_id)
            if tf not in gi or peak not in ai:
                raise ValueError("screened TF-peak axis absent from WT inputs")
            inputs.setdefault(peak,set()).add((tf,_lag(row.tf_lag)))
    bundles,metrics,failures,table = {},{},{},[]
    workers = min(requested_workers, max(1, len(inputs)))
    jobs = _target_jobs(kind, snapshot, inputs, gi, ai, cfg, policy)
    results = _execute_target_jobs(jobs, snapshot["support"], cfg, workers, len(inputs),
        sum(len(keys) for keys in inputs.values()), kind)
    pack_started = time.monotonic()
    for target, (bundle, metric) in results.items():
        if bundle is None:
            failures[target] = metric
            continue
        bundles[target] = bundle; metrics[target] = metric
        keys = bundle["input_keys"]
        for i,(key,lag) in enumerate(keys):
            table.append(dict(gene=target if kind==GENE_TYPE else key,peak_id=key if kind==GENE_TYPE else target,
                tf_gene=None if kind==GENE_TYPE else key,lag=lag,tf_lag=lag,
                alpha=bundle["weights"][i],wt_response_score=bundle["scores"][i],
                r2_score=metrics[target]["train_r2"],fit_status=metrics[target]["status"],
                quality_scope="target_full_training",model_mode="saturating_hybrid"))
    logger.info("Saturating %s: packing complete fitted=%d skipped=%d elapsed=%.1fs; validating registry",
        "Step 5" if kind == GENE_TYPE else "Step 6", len(bundles), len(failures), time.monotonic()-pack_started)
    validate_started = time.monotonic()
    registry = _registry(kind,snapshot,cfg,bundles,metrics,failures,policy)
    logger.info("Saturating %s: registry validated in %.1fs",
        "Step 5" if kind == GENE_TYPE else "Step 6", time.monotonic()-validate_started)
    metric_table = pd.DataFrame([dict(target_id=target,gene=target if kind==GENE_TYPE else None,
        peak_id=target if kind==TF_TYPE else None,**values) for target,values in metrics.items()],
        columns=["target_id","gene","peak_id","train_r2","training_rmse","n_rows","status",
                 "quality_scope","validation_performed","single_fit_statuses"])
    return pd.DataFrame(table,columns=["gene","peak_id","tf_gene","lag","tf_lag","alpha","wt_response_score","r2_score","fit_status","quality_scope","model_mode"]),metric_table,registry


def fit_peak_gene(rna_adata,atac_adata,pseudotime_df,causal_edges,config,*,bin_plan=None):
    return _fit(GENE_TYPE,rna_adata,atac_adata,pseudotime_df,causal_edges,config,bin_plan)


def fit_tf_peak(rna_adata,atac_adata,pseudotime_df,screened_tf_weights,config,*,bin_plan=None):
    return _fit(TF_TYPE,rna_adata,atac_adata,pseudotime_df,screened_tf_weights,config,bin_plan)[2]


def validate_registry(registry,*,expected_type=None,expected_schema=None):
    if not isinstance(registry,dict) or registry.get("type") not in (GENE_TYPE,TF_TYPE):
        raise ValueError("not a saturating registry")
    if registry.get("model_contract_version") != 2:
        raise ValueError("obsolete model contract; refit Step 5 and Step 6 with current code in a new checkpoint_dir")
    required={"type","schema","schema_version","model_mode","targets","target_metrics","failures",
        "config","context","cell_support","input_unit","output_unit","numerical_signature",
        "model_contract_version","binning_metadata","fingerprint"}
    if set(registry) not in (required, required | {"budget_exhausted_policy"}):
        raise ValueError("saturating registry fields mismatch")
    policy = core.validate_budget_exhausted_policy(registry.get("budget_exhausted_policy", "exclude"))
    kind = registry["type"]
    continuous=registry["context"].get("atac_representation",{}).get("unit")=="normalized_atac_log1p"
    schema=(GENE_SCHEMA_V2 if kind==GENE_TYPE else TF_SCHEMA_V2) if continuous else (GENE_SCHEMA if kind==GENE_TYPE else TF_SCHEMA)
    family = registry["context"].get("response_family", "saturation")
    if family not in ("nn", "saturation", "spline"):
        raise ValueError("unknown registry response family")
    if family == "nn":
        from .nn_hybrid import NNConfig
        NNConfig.from_dict(registry["context"].get("nn_response_config", {}))
        schema = "hybrid_nn_peak_gene_registry_v1" if kind == GENE_TYPE else "hybrid_nn_tf_peak_registry_v1"
    if family == "spline":
        from .spline_hybrid import SplineConfig
        if "spline_response_config" not in registry["context"]:
            raise ValueError("spline registry configuration required")
        SplineConfig.from_dict(registry["context"]["spline_response_config"])
        schema = "hybrid_spline_peak_gene_registry_v2" if kind == GENE_TYPE else "hybrid_spline_tf_peak_registry_v2"
    if registry.get("schema")!=schema or registry.get("schema_version")!=schema or registry.get("model_mode")!="saturating_hybrid":
        raise ValueError("saturating registry schema mismatch")
    if expected_type is not None and kind!=expected_type or expected_schema is not None and schema!=expected_schema:
        raise ValueError("unexpected saturating registry type/schema")
    payload = {k:v for k,v in registry.items() if k!="fingerprint"}
    if registry.get("fingerprint")!=core._digest(payload):
        raise ValueError("saturating registry fingerprint mismatch")
    cfg = core.HybridConfig.from_dict(registry["config"])
    if cfg.workers != 1:
        raise ValueError("registry fitted config must use canonical runtime workers=1")
    if registry["numerical_signature"]!=_source_signature(kind,continuous,family):
        raise ValueError("saturating consumed numerical source/version mismatch")
    context = registry["context"]
    if context["fingerprint"]!=core._digest({k:v for k,v in context.items() if k!="fingerprint"}):
        raise ValueError("WT context fingerprint mismatch")
    for axis in ("cell_ids","gene_ids","peak_ids"):
        if not context[axis] or len(context[axis])!=len(set(context[axis])):
            raise ValueError("invalid registry ordered axes")
    n,c = context["n_bins"],len(context["cell_ids"])
    operator = core._array(context["operator"],"registry operator",2)
    if operator.shape!=(n,c) or (operator<0).any() or len(context["bin_branch_ids"])!=n:
        raise ValueError("invalid registry operator")
    support = core.CellSupport(**registry["cell_support"])
    mapping = core._array(context["bin_to_support"],"bin support mapping",1,True)
    valid = operator.sum(axis=1)>0
    if len(mapping)!=n or not np.array_equal(mapping[valid],np.arange(valid.sum())) or (mapping[~valid]!=-1).any():
        raise ValueError("invalid shared support bin mapping")
    if support.weights.shape!=(int(valid.sum()),c) or not np.allclose(support.weights,operator[valid]/operator[valid].sum(axis=1,keepdims=True),rtol=1e-10,atol=1e-12):
        raise ValueError("shared support weights differ from frozen operator")
    atac_unit="normalized_atac_log1p" if continuous else "raw_atac"
    input_unit,output_unit = (atac_unit,"log1p_rna_log1p_cp10k") if kind==GENE_TYPE else ("rna_log1p_cp10k",atac_unit)
    if continuous:
        rep=context["atac_representation"]
        from .atac_normalization import validate_representation_metadata
        validate_representation_metadata(rep,context["cell_ids"],context["peak_ids"])
        if context["aggregate_unit"] != "RNA_L1/normalized_atac_log1p":
            raise ValueError("continuous ATAC aggregation coordinate mismatch")
    if registry["input_unit"]!=input_unit or registry["output_unit"]!=output_unit or set(registry["target_metrics"])!=set(registry["targets"]):
        raise ValueError("registry target units/metrics mismatch")
    axis = set(context["gene_ids"] if kind==GENE_TYPE else context["peak_ids"])
    input_axis = set(context["peak_ids"] if kind==GENE_TYPE else context["gene_ids"])
    if set(registry["failures"]) & set(registry["targets"]):
        raise ValueError("registry failures overlap fitted targets")
    for target, failure in registry["failures"].items():
        if target not in axis or not isinstance(failure, dict):
            raise ValueError("invalid registry target failure")
        if "excluded_single_relations" in failure:
            core.validate_excluded_relations(failure,target,())
            if failure.get("status") != "no_information" or failure.get("train_r2") is not None:
                raise ValueError("invalid all-failed target status/quality")
            for record in failure["excluded_single_relations"]["records"]:
                key,lag=record["input_key"]
                if key not in input_axis or kind==GENE_TYPE and lag!=1:
                    raise ValueError("registry failed input key/lag mismatch")
    inverse={int(s):b for b,s in enumerate(mapping) if s>=0}
    for key,bundle in registry["targets"].items():
        model = core.TargetModel.from_bundle(bundle,cell_support=support)
        if model.provenance.get("response_family", "saturation") != family or family == "nn" and core._plain(model.provenance.get("nn_response_config")) != registry["context"]["nn_response_config"]:
            raise ValueError("registry target response family/config mismatch")
        if family == "spline" and core._plain(model.provenance.get("spline_response_config")) != registry["context"]["spline_response_config"]:
            raise ValueError("registry spline target configuration mismatch")
        if model.provenance.get("budget_exhausted_policy", "exclude") != policy:
            raise ValueError("registry target budget policy mismatch")
        if model.target_id!=key or key not in axis or model.input_unit!=input_unit or model.output_unit!=output_unit or model.config!=cfg:
            raise ValueError("registry target contract mismatch")
        if continuous and kind==TF_TYPE and model.bounds[1]>np.log1p(context["atac_representation"]["target_sum"])+1e-10:
            raise ValueError("continuous TF target cap exceeds physical ceiling")
        excluded = model.provenance.get("excluded_single_relations", {}).get("records", ())
        all_keys = model.input_keys + tuple(tuple(record["input_key"]) for record in excluded)
        if any(key not in input_axis or kind==GENE_TYPE and lag!=1 for key,lag in all_keys):
            raise ValueError("registry input key/lag mismatch")
        if model.provenance.get("context_fingerprint")!=context["fingerprint"] or model.provenance.get("relation_type")!=kind:
            raise ValueError("registry model context mismatch")
        for local,branch,support_row in zip(model.target_rows,model.branch_ids,model.support_rows):
            if int(support_row) not in inverse:
                raise ValueError("target support row absent from registry")
            global_bin=inverse[int(support_row)]
            start=next((a for a,b in context["branch_boundaries"] if a<=global_bin<b),None)
            if start is None or global_bin-start!=local or context["bin_branch_ids"][global_bin]!=branch:
                raise ValueError("target branch-local/global support mapping mismatch")
        metric = registry["target_metrics"][key]
        if metric["n_rows"]!=len(model.target_rows) or metric["status"] not in ("ok","no_information","budget_exhausted"):
            raise ValueError("registry target metrics mismatch")
        if "excluded_single_relations" in model.provenance and (metric["status"] != model.status or
                metric.get("single_fit_statuses") != [s.metrics["status"] for s in model.singles]):
            raise ValueError("pruned target metrics mismatch")
        r2 = metric["train_r2"]
        if r2 is not None and (core._number(r2,"target train R2")>1):
            raise ValueError("invalid registry train R2")
    return registry


def load_models(registry,expected_type=None):
    validate_registry(registry,expected_type=expected_type)
    support = core.CellSupport(**registry["cell_support"])
    return {key:core.TargetModel.from_bundle(value,cell_support=support) for key,value in registry["targets"].items()},support
