"""
CausalBridge 主 Pipeline Runner

将所有模块串联为完整的分析流程。这是用户执行"一键分析"的入口。

流程概要：
  Step 1: 加载配置 + 数据
  Step 2: 预处理（QC、归一化、高变基因）
  Step 3-8: 按细胞簇建模 + 扰动模拟（伪时间在簇内独立推断）
  Step 9: 结果汇总 + 报告生成

使用方式：
  from CausalBridge.run import run_pipeline
  results = run_pipeline("config.yaml")
"""

import logging
import sys
import pickle
import hashlib
import json
import ast
import inspect
import textwrap
import copy
from contextvars import ContextVar
from functools import wraps
from types import SimpleNamespace
from importlib.metadata import version, PackageNotFoundError
from scipy import sparse
from pathlib import Path

import numpy as np
import pandas as pd
from .io import (
    load_config, load_data, load_target_genes, save_results,
    normalize_anndata_string_metadata, save_hvg_genes,
    _gene_identity_enabled, _gene_identity_runtime_config,
)
from .preprocess import (
    preprocess_rna, preprocess_atac, match_cells, cluster_cells,
    save_cluster_top_genes, build_paga_lineages,
)
from .granger import infer_pseudotime, granger_test, build_causal_grn
from .kinetics import (fit_atac_to_rna, fit_rna_to_atac, _local_motif_resource_paths,
                       _prepare_adaptive_motif_resources, JASPAR_RESOURCE_URL,
                       _raw_scan_cache_descriptor, _RESOURCE_DIGEST_MEMO,
                       _resource_file_identity as _kinetics_resource_file_identity)
from .perturbation import propagate_perturbation, perturb_peak
from .binning_adapter import (
    prepare_wt_embedding, fit_context_binning, persist_binning_diagnostics,
    validate_bin_plan, _selector, _hash_array, _hash_matrix,
)
from ._wt_aggregation import validate_model_mode
from .gene_identity import parse_gene_gtf, apply_gene_identity, resolve_gene_token
from .reference_clustering import build_reference_clustering, align_model_cells, attach_reference_clustering

ADAPTIVE_CHECKPOINT_SCHEMA = "adaptive-pipeline-checkpoint-4"
_RESOURCE_CHUNK_BYTES = 1024 * 1024


def _resource_file_identity(path):
    """Compatibility wrapper over the single shared streaming/memo implementation."""
    return _kinetics_resource_file_identity(path, chunk_bytes=_RESOURCE_CHUNK_BYTES)


def _resource_identity(config, motif_resources=None):
    identity = {name: _resource_file_identity(path)
                for name, path in _local_motif_resource_paths(config).items()}
    identity["online_request"] = {
        "loader_api": "local-first-materialized-motifs-v1", "url": JASPAR_RESOURCE_URL,
        "requested_db": config.get("rna_to_atac", {}).get("motif_db", "jaspar2024"),
        "remote_freshness_verified": False}
    if motif_resources is not None:
        identity["materialized_motifs"] = _provenance_fingerprint(motif_resources["motif_db"])
        tf_database = motif_resources.get("tf_database")
        identity["materialized_tf_database"] = _provenance_fingerprint(
            sorted(tf_database) if tf_database is not None else None)
    return identity


def _provenance_fingerprint(value):
    """Content hash for upstream tables/NN buffers, without pickling runtime objects."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, pd.DataFrame):
            visit(item.columns.tolist())
            visit(item.index.to_numpy())
            for column in item:
                visit(item[column].to_numpy())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(str(key)); visit(item[key])
        elif isinstance(item, (set, frozenset)):
            visit(sorted(item, key=str))
        elif isinstance(item, (tuple, list)):
            digest.update(str(len(item)).encode())
            for entry in item:
                visit(entry)
        elif isinstance(item, np.ndarray):
            _hash_array(digest, item)
        elif hasattr(item, "detach"):
            visit(item.detach().cpu().numpy())
        else:
            digest.update(json.dumps(item, sort_keys=True, default=str).encode())
        digest.update(b"\x00")
    visit(value)
    return digest.hexdigest()


# Exact reviewed source aliases for the low-count target rescue change only.
# Actual matrices/axes/configuration remain covered by checkpoint fingerprints.
# Any further numerical source edit produces a different hash and is NOT aliased.
_TARGET_RESCUE_SOURCE_COMPATIBILITY = {
    "CausalBridge.preprocess.preprocess_rna": (
        "6203d4322cee79e7dca0003d88d65a76f85b801ecfcd82779928d10756cab59b",
        "49e113765769bae262963adc064870e0178e16d4dd7010e127dbf63def137de2",
    ),
    "CausalBridge.run._prepare_identity_model_rna": (
        "61ae6a5ad3d3d84002dbb61eacd6f550f03a69ac13d34ed61d34a90020572906",
        "eab14e09f08982515ac6b75e449930b249040748119dd6e9a30f7b4ec154aacc",
    ),
}


# Reviewed execution-only partitioning of candidate pairs. An alias applies
# only while BOTH the dispatched function and complete worker implementation
# match the verified version. Later arithmetic/kernel edits invalidate it.
_SCREENING_SOURCE_COMPATIBILITY = {
    "CausalBridge.granger.granger_test": (
        "1cc978025c3930ac6770fa155ec8e871719e9f46d8425a2eafe2ab95aa419229",
        "1eead3e505217f61a0e1cfe39fb421a4439d1c6b0b5850c482123ece8e27bf1d",
    ),
    "CausalBridge.kinetics.fit_rna_to_atac": (
        "d37569c14093ca94fce6586571b7fb115dd4f0f80ed698eff476dd7810295f47",
        "f3e61ac19679962463eea27d7230b248510f30f880468d29d655f9ba4d57e6bb",
    ),
}
_SCREENING_EXECUTOR_SHA256 = '8cd9adf720d5e5dce13c6dc094b5d229a34e0f9be91aae0fe9fcf9f0911d257f'

# Only repairs reference means that previously failed strict model validation.
# Already valid model bundles and all fitted responses retain their contract.
_WT_REFERENCE_SOURCE_COMPATIBILITY = {
    "CausalBridge.saturating_hybrid.fit_target": (
        "39a242bacfcc0db144005ba25589d68985faf4945ee9530a620cc0585e29fcd6",
        "fc0ba8371457ec2fa04fcb064fbed600ba5f31d20cf9937c1cc486d96f5f36ba",
    ),
    "CausalBridge.saturating_transfer._source_signature": (
        "cacd564ca7dcf18c5d2ed3abd53ef63a1aa8bc5709366bc56c65d6d13751ab39",
        "1b1f56d111c02857f765e45e1aabf0dbbc7357274ea80eab7c8d8463f40bc802",
    ),
}

_PIPELINE_SOURCE_HASHES = ContextVar('pipeline_source_hashes', default=None)


def _with_pipeline_source_snapshot(function):
    """Scope source identities to one invocation, without leaking across runs."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _PIPELINE_SOURCE_HASHES.set({})
        try:
            return function(*args, **kwargs)
        finally:
            _PIPELINE_SOURCE_HASHES.reset(token)
    return wrapped


def _numerical_source_hash(function):
    snapshot = _PIPELINE_SOURCE_HASHES.get()
    if snapshot is not None and function in snapshot:
        return snapshot[function]
    digest = _read_numerical_source_hash(function)
    if snapshot is not None:
        snapshot[function] = digest
    return digest


def _read_numerical_source_hash(function):
    """Hash an explicit consumed function/class, excluding comments/docstrings."""
    try:
        source = inspect.getsource(function)
    except (OSError, IndexError) as exc:
        raise RuntimeError(
            f"cannot read loaded numerical source: {function.__module__}.{function.__qualname__}; "
            "source may have moved while process was running; "
            "源码与已加载代码不一致，请重启进程后从现有 checkpoint 继续，无需删除断点") from exc
    tree = ast.parse(textwrap.dedent(source))
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    lambda_source = function.__name__ == "<lambda>" and any(isinstance(node, ast.Lambda) for node in ast.walk(tree))
    if not lambda_source and (len(definitions) != 1 or definitions[0].name != function.__name__):
        raise RuntimeError(
            f"source moved while process was running: {function.__module__}.{function.__qualname__}; "
            "restart the process before creating checkpoints")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
    digest = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    key = function.__module__ + "." + function.__qualname__
    compatible = _TARGET_RESCUE_SOURCE_COMPATIBILITY.get(key)
    if compatible and digest == compatible[0]:
        return compatible[1]
    compatible = _WT_REFERENCE_SOURCE_COMPATIBILITY.get(key)
    if compatible and digest == compatible[0]:
        return compatible[1]
    parallel = _SCREENING_SOURCE_COMPATIBILITY.get(key)
    if parallel and digest == parallel[0]:
        from . import screening_parallel
        worker_digest = hashlib.sha256(Path(screening_parallel.__file__).read_bytes()).hexdigest()
        if worker_digest == _SCREENING_EXECUTOR_SHA256:
            return parallel[1]
        # Include helper changes even if the dispatch wrapper did not change.
        return hashlib.sha256((digest + worker_digest).encode()).hexdigest()
    return digest


def _stage_numerical_signatures(stage, model_mode="saturating_hybrid", *, gene_identity_mode="legacy"):
    """Explicit dependency lists: no runner/module-file or perturbation hashes."""
    validate_model_mode({'atac_to_rna':{'model_mode':model_mode}})
    from . import adaptive_binning as ab, binning_adapter as adapter, granger, kinetics
    dependencies = {
        adapter: ("require_bin_plan", "_ids", "_usable", "plan_branch_boundaries",
                  "validate_bin_plan", "execution_pseudotime", "aggregate_with_plan"),
        ab: ("BinningConfig", "BinPlan", "FittedBinningPolicy", "fit_policy", "_inputs", "_requested_bins", "_qcut_layout",
             "_budget", "_center_layout", "_detect_boundaries", "_state_bandwidths",
             "_cell_valid", "_kernel", "_fingerprint", "_plan", "aggregate",
             "_owned", "_readonly_matrix"),
    }
    if stage == "step4_granger":
        dependencies[granger] = ("granger_test", "_generate_candidate_pairs", "_bin_expression",
                                 "_compute_cytotrace_scores", "_find_root_cell")
        from . import cytotrace, gene_identity
        dependencies[gene_identity] = ("gene_tss_coordinates",)
        dependencies[cytotrace] = ("compute_scores", "attach_scores", "verified_scores", "runtime_settings", "axis_hash")
        dependencies[kinetics] = ("_branch_aware_lag_pairs",)
        packages = ("numpy", "pandas", "scipy", "statsmodels")
    elif stage == "step5_atac_to_rna":
        dependencies[granger] = ("build_causal_grn",)
        dependencies[kinetics] = ("fit_atac_to_rna", "_branch_aware_lag_pairs")
        packages = ("numpy", "pandas", "scipy", "scikit-learn")
    elif stage == "step6_rna_to_atac":
        dependencies[kinetics] = ("fit_rna_to_atac", "_scan_one_sequence",
            "_materialize_raw_motif_hits", "_scan_peaks_with_jaspar", "_build_family_groups",
            "_supplement_family_proxies", "_extract_peak_sequences", "_extract_peak_sequences_from_causal",
            "_assign_tfs_fallback", "_filter_tfs_by_expression", "_fit_tf_peak_regression", "_local_tf_peak_routes", "_local_tf_peak_routes_scalar",
            "_branch_aware_lag_pairs", "_parse_gimmemotifs_pfm", "_parse_cisbp_pfm",
            "_dedup_motifs_by_tf", "_compute_ic", "_pfm_to_pwm")
        packages = ("numpy", "pandas", "scipy", "biopython", "gimmemotifs")
    else:
        raise ValueError(f"unknown checkpoint stage: {stage}")
    if model_mode == "saturating_hybrid" and stage in ("step5_atac_to_rna", "step6_rna_to_atac"):
        from . import saturating_transfer, saturating_hybrid, _wt_aggregation
        dependencies[_wt_aggregation] = ("_weights", "_full_wt_aggregates")
        if stage == "step5_atac_to_rna":
            dependencies[kinetics] = ("fit_atac_to_rna",)
        for module in (saturating_transfer, saturating_hybrid):
            dependencies[module] = tuple(name for name, obj in vars(module).items()
                if (inspect.isfunction(obj) or inspect.isclass(obj))
                and getattr(obj, "__module__", None) == module.__name__)
        from . import atac_normalization
        dependencies[atac_normalization] = tuple(name for name,obj in vars(atac_normalization).items()
            if inspect.isfunction(obj) and getattr(obj,"__module__",None)==atac_normalization.__name__)
    if gene_identity_mode == "gtf_gene_id":
        from . import gene_identity
        dependencies[gene_identity] = ("normalize_gene_id", "parse_gene_gtf", "_attrs", "_display_labels",
            "_open_bytes", "_hash_bytes", "apply_gene_identity", "resolve_gene_token", "_allowed", "GeneIdentityIndex", "GeneRecord", "gene_tss_coordinates")
        if stage == "step4_granger":
            from . import reference_clustering, preprocess
            dependencies[reference_clustering] = ("build_reference_clustering", "raw_target_symbols",
                "_input_fingerprint", "_unique_cells", "align_model_cells", "attach_reference_clustering")
            dependencies[preprocess] = ("preprocess_rna", "preprocess_atac", "match_cells", "cluster_cells")
        if stage == "step6_rna_to_atac":
            dependencies[gene_identity] += ("map_tf_symbols",)
            dependencies[kinetics] += ("_gene_base_map",)
    sources = {f"{module.__name__}.{name}": _numerical_source_hash(getattr(module, name))
               for module, names in dependencies.items() for name in names}
    if gene_identity_mode == "gtf_gene_id":
        for function in (_prepare_gene_identity_inputs, _validate_identity_axis,
                         _validate_runtime_identity, _validate_identity_required_genes,
                         _prepare_identity_model_rna):
            sources[f"{__name__}.{function.__name__}"] = _numerical_source_hash(function)
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    constants = {"adaptive_binning.VERSION": ab.VERSION, "adaptive_binning.EPSILON": ab.EPSILON,
                 "binning_adapter.SCHEMA_VERSION": adapter.SCHEMA_VERSION}
    if gene_identity_mode == "gtf_gene_id":
        constants["gene_identity.SCHEMA"] = gene_identity.SCHEMA
        constants["gene_identity.ATTR_PATTERN"] = gene_identity._ATTR_RE.pattern
        constants["gene_identity.VERSION_PATTERN"] = gene_identity._VERSION_RE.pattern
    if stage == "step6_rna_to_atac":
        constants["kinetics.RAW_MOTIF_SCANNER_VERSION"] = kinetics.RAW_MOTIF_SCANNER_VERSION
    if model_mode == "saturating_hybrid" and stage != "step4_granger":
        constants.update(saturating_target_schema="saturating_hybrid_target_v1",
            saturating_gene_registry_schema="saturating_peak_gene_registry_v1",
            saturating_tf_registry_schema="saturating_tf_peak_registry_v1",
            saturating_domain="saturating-hybrid-v1")
    return {"functions": sources, "versions": versions, "constants": constants}


def _ordered_model_data_fingerprint(rna, atac):
    """Only fitted matrices, ordered axes and consumed genomic coordinates."""
    digest = hashlib.sha256()
    for adata in (rna, atac):
        _hash_array(digest, adata.obs_names)
        _hash_array(digest, adata.var_names)
        _hash_matrix(digest, adata.X)
        for column in ("chr", "start", "end", "strand", "gene_tss", "tss_source"):
            digest.update(column.encode())
            _hash_array(digest, adata.var[column].to_numpy() if column in adata.var else np.array([]))
    from .atac_normalization import atac_input_contract
    digest.update(json.dumps(rna.uns.get("official_cytotrace", {}), sort_keys=True).encode())
    representation = atac_input_contract(atac)
    if representation is not None:
        digest.update(json.dumps(representation,sort_keys=True,allow_nan=False).encode())
    return digest.hexdigest()


def _resource_contract(identity):
    # Locations are audit information; availability and content remain strict.
    if isinstance(identity, dict):
        return {key: _resource_contract(value) for key, value in identity.items() if key != "path"}
    return identity


def _adaptive_checkpoint_metadata(context, config, target_genes, gene_types, rna_raw=None,
                                  *, rna, atac, n_peaks=None, motif_resources=None):
    common = {
        "data_fingerprint": _ordered_model_data_fingerprint(rna, atac),
        "policy_fingerprint": context.policy.fingerprint, "plan_fingerprint": context.plan.fingerprint,
        "pseudotime_fingerprint": _provenance_fingerprint([context.plan.cell_ids, context.plan.times, context.plan.branches]),
        "bin_plan_centers": context.plan.centers.tolist(),
        "branch_boundaries": [list(pair) for pair in context.policy.branch_boundaries],
        "state_modality": context.metadata.get("state_embedding", {}).get("state_modality", "joint"),
    }
    if _gene_identity_enabled(config):
        common["gene_identity"] = _identity_contract(rna)
        common["rna_feature_policy"] = "gtf_eligible_universe_normalization_hvg_v1"
        common["clustering_reference"] = rna.uns.get("clustering_reference", {})
        common["resolved_targets"] = list(target_genes)
        common["resolved_gene_types"] = dict(gene_types)
        common["resolved_root_marker"] = config.get("pseudotime", {}).get("root_cell_marker")
    stages = {}
    identity_mode = config.get("input", {}).get("gene_identity_mode", "legacy")
    stages["step4_granger"] = dict(common, granger=config.get("granger", {}),
        numerical=_stage_numerical_signatures("step4_granger", gene_identity_mode=identity_mode))
    mode = config.get("atac_to_rna", {}).get("model_mode", "saturating_hybrid")
    stages["step5_atac_to_rna"] = dict(common, atac_to_rna=_step5_model_config(config),
        step4_contract=_provenance_fingerprint(stages["step4_granger"]),
        numerical=_stage_numerical_signatures("step5_atac_to_rna", mode, gene_identity_mode=identity_mode))
    resources = _resource_identity(config, motif_resources)
    stages["step6_rna_to_atac"] = dict(common,
        rna_to_atac=_step6_model_config(config),
        targets=list(target_genes), gene_types=gene_types,
        # Step 6 uses only the full gene axis for its expression-presence filter.
        raw_rna_gene_axis=(list(rna_raw.var_names) if rna_raw is not None
                          and config.get("rna_to_atac", {}).get("tf_expression_filter", True) else None),
        resources=_resource_contract(resources),
        motif_order=list((motif_resources or {}).get("motif_db") or {}),
        step5_contract=_provenance_fingerprint(stages["step5_atac_to_rna"]),
        numerical=_stage_numerical_signatures("step6_rna_to_atac", mode, gene_identity_mode=identity_mode))
    saturation_data = {}
    if mode == "saturating_hybrid":
        from .saturating_transfer import _data_fingerprint, _source_signature, response_settings, GENE_TYPE, TF_TYPE
        saturation_data["saturating_data_fingerprint"] = _data_fingerprint(rna, atac)
        for stage, kind in (("step5_atac_to_rna", GENE_TYPE), ("step6_rna_to_atac", TF_TYPE)):
            family = response_settings(config, kind)[0]
            if family in ("nn", "spline"):
                from .atac_normalization import validate_atac_depth_config
                stages[stage]["numerical"][family+"_response"] = _source_signature(
                    kind, validate_atac_depth_config(config)["enabled"], family)
    return {"schema_version": ADAPTIVE_CHECKPOINT_SCHEMA, "stages": stages, **saturation_data,
            "resource_identity": resources, "atac_n_vars": n_peaks,
            "atac_peak_ids": list(atac.var_names), "rna_gene_ids": list(rna.var_names),
            # Used only to namespace candidate caches, never checkpoint validation.
            "model_contract_fingerprint": _provenance_fingerprint(stages),
            "audit": {"adapter_model_contract_fingerprint": context.model_contract_fingerprint}}


def _stage_checkpoint_metadata(metadata, stage, upstream):
    return {"schema_version": ADAPTIVE_CHECKPOINT_SCHEMA, "stage": stage,
            "contract": metadata["stages"][stage], "upstream_fingerprint": upstream}


def _bind_step5_contract(metadata, context, pseudotime_df, granger_results):
    """Use one Step-5 domain for training, recovery and perturbation NN gates."""
    contract = metadata["stages"]["step5_atac_to_rna"]
    contract["granger_output_fingerprint"] = _provenance_fingerprint(granger_results)
    fingerprint = _provenance_fingerprint(contract)
    metadata["stages"]["step6_rna_to_atac"]["step5_contract"] = fingerprint
    adapter_contract = {
        "bin_plan_fingerprint": context.plan.fingerprint,
        "bin_policy_fingerprint": context.policy.fingerprint,
        "binning_data_fingerprint": contract["data_fingerprint"],
        "binning_model_contract_fingerprint": fingerprint,
        "binning_model_contract_domain": "saturating-hybrid-v1",
        "n_bins": context.plan.n_bins, "bin_plan_centers": context.plan.centers.tolist(),
        "branch_boundaries": [list(pair) for pair in context.policy.branch_boundaries],
    }
    metadata["adapter_contract"] = adapter_contract
    pseudotime_df.attrs.update(adapter_contract)


def _checkpoint_field_differences(saved, expected, prefix="", limit=10):
    differences = []
    def brief(value):
        if isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value):
            return value[:12] + "..."
        text = json.dumps(value, sort_keys=True, default=str)
        return text if len(text) <= 96 else text[:93] + "..."
    def visit(old, new, path):
        if len(differences) >= limit:
            return
        if isinstance(old, dict) and isinstance(new, dict):
            for key in sorted(set(old) | set(new)):
                child = f"{path}.{key}" if path else str(key)
                if key not in old or key not in new:
                    differences.append(f"{child}: {'missing saved field' if key not in old else 'unexpected saved field'}")
                else:
                    visit(old[key], new[key], child)
                if len(differences) >= limit:
                    break
        elif old != new:
            differences.append(f"{path}: saved={brief(old)} current={brief(new)}")
    visit(saved, expected, prefix)
    return differences


CHECKPOINT_RECOVERY_GUIDANCE = (
    "使用新的checkpoint_dir或备份后移走Step5/6不兼容文件；"
    "其他阶段若不兼容，也须备份后移走对应文件再重跑。不会自动搜索旧子目录或迁移断点。"
)


# Reproduced inspect.getsource line-offset artifacts from the running pre-rescue
# process after the target-rescue source edit. Never apply these as global hash
# aliases: only the complete historical stage contract below can be accepted.
_SHIFTED_RESCUE_SOURCE_HASHES = {
    "CausalBridge.preprocess.preprocess_rna": (
        "49e113765769bae262963adc064870e0178e16d4dd7010e127dbf63def137de2",
        "6079921b46c0ae4eff19930f3e3b8d33643b5ae283231e40a5874a82a759df1e"),
    "CausalBridge.preprocess.preprocess_atac": (
        "864b262d8cf533967c54bc6575badf5d46ac688e835d3d5ca8aecee470f8d1fb",
        "6203d4322cee79e7dca0003d88d65a76f85b801ecfcd82779928d10756cab59b"),
    "CausalBridge.preprocess.match_cells": (
        "2da8babdbcc522458677918ae8c3679d751e5f92f0c8d2a32d61fe4a7c54b22c",
        "d65775c7e795c10737653fc5a33fd84e034d32abe773ad04cc51a98d82a542ae"),
    "CausalBridge.preprocess.cluster_cells": (
        "402faec1a4d69a54e202ed5008c0fb08a0eb85bd1f8f3cf7470f71acd79ad7b4",
        "864b262d8cf533967c54bc6575badf5d46ac688e835d3d5ca8aecee470f8d1fb"),
    "CausalBridge.run._validate_identity_required_genes": (
        "914c69a16f61a7c5067365e1a2c0e35ceded9680b7cb27ca570fba4499837230",
        "61ae6a5ad3d3d84002dbb61eacd6f550f03a69ac13d34ed61d34a90020572906"),
    "CausalBridge.run._prepare_identity_model_rna": (
        "eab14e09f08982515ac6b75e449930b249040748119dd6e9a30f7b4ec154aacc",
        "61ae6a5ad3d3d84002dbb61eacd6f550f03a69ac13d34ed61d34a90020572906"),
}


def _shifted_rescue_checkpoint_contract(expected, stage, upstream):
    """Reconstruct one exact historical signature error, keeping all data checks.

    Contracts are never edited on disk. Rebuild chained digests from the full
    expected contracts, rather than copying unverified saved digest values.
    Any future edit of an affected source disables this compatibility path.
    """
    historical = copy.deepcopy(expected)
    try:
        stages = historical["stages"]
        functions = stages["step4_granger"]["numerical"]["functions"]
        if any(functions.get(key) != good for key, (good, _) in _SHIFTED_RESCUE_SOURCE_HASHES.items()):
            return None
        for name in ("step4_granger", "step5_atac_to_rna", "step6_rna_to_atac"):
            functions = stages[name]["numerical"]["functions"]
            for key, (good, shifted) in _SHIFTED_RESCUE_SOURCE_HASHES.items():
                if key in functions:
                    if functions[key] != good:
                        return None
                    functions[key] = shifted
        stages["step5_atac_to_rna"]["step4_contract"] = _provenance_fingerprint(stages["step4_granger"])
        stages["step6_rna_to_atac"]["step5_contract"] = _provenance_fingerprint(stages["step5_atac_to_rna"])
        return _stage_checkpoint_metadata(historical, stage, upstream)
    except KeyError:
        return None


def _validate_adaptive_checkpoint(checkpoint, expected, stage, upstream):
    metadata = checkpoint.get("adaptive_checkpoint") if isinstance(checkpoint, dict) else None
    guidance = CHECKPOINT_RECOVERY_GUIDANCE
    if not isinstance(metadata, dict) or metadata.get("schema_version") != ADAPTIVE_CHECKPOINT_SCHEMA:
        raise RuntimeError(
            f"{stage}: missing or mismatched adaptive checkpoint metadata; schema mismatch "
            f"(saved={metadata.get('schema_version') if isinstance(metadata, dict) else 'missing'}, "
            f"current={ADAPTIVE_CHECKPOINT_SCHEMA}). Schema 2 broad hashes cannot reconstruct historical "
            f"stage signatures and cannot be safely auto-migrated. {guidance}")
    required = _stage_checkpoint_metadata(expected, stage, upstream)
    # Optional audit fields cannot affect scientific compatibility.
    consumed = {key: metadata[key] for key in required if key in metadata}
    differences = _checkpoint_field_differences(consumed, required)
    if differences:
        historical = _shifted_rescue_checkpoint_contract(expected, stage, upstream)
        if historical is not None and not _checkpoint_field_differences(consumed, historical):
            logger.warning("%s: 兼容已复现的靶点捞回期间源码行号错位签名；数据、参数、上游结果及模型校验保持严格", stage)
            differences = []
    if differences:
        raise RuntimeError(f"{stage}: incompatible adaptive checkpoint fields (up to 10): "
                           + "; ".join(differences) + ". " + guidance)
    required_payload = {
        "step4_granger": ("granger_results",),
        "step5_atac_to_rna": ("causal_edges", "transfer_functions", "transfer_models"),
        "step6_rna_to_atac": ("tf_weights",),
    }.get(stage, ())
    if any(checkpoint.get(key) is None for key in required_payload):
        raise RuntimeError(f"{stage}: incomplete adaptive checkpoint; {guidance}")
    if "saturating_data_fingerprint" in expected and stage in ("step5_atac_to_rna", "step6_rna_to_atac"):
        from .saturating_transfer import validate_registry, _source_signature
        kind = "saturating_hybrid_peak_gene" if stage == "step5_atac_to_rna" else "saturating_hybrid_tf_peak"
        key = "transfer_models" if stage == "step5_atac_to_rna" else "tf_model_registry"
        try:
            registry = validate_registry(checkpoint.get(key), expected_type=kind)
            family = registry["context"].get("response_family", "saturation")
            layer = "atac_to_rna" if stage == "step5_atac_to_rna" else "rna_to_atac"
            layer_contract = expected["stages"][stage][layer]
            settings = dict(expected["stages"]["step5_atac_to_rna"]["atac_to_rna"]["saturating_hybrid"])
            registry_settings = dict(registry["config"])
            if family == "saturation":
                if registry.get("budget_exhausted_policy", "exclude") != "exclude":
                    registry_settings["budget_exhausted_policy"] = registry["budget_exhausted_policy"]
            else:
                settings.pop("budget_exhausted_policy", None)
            if (registry["context"]["data_fingerprint"] != expected["saturating_data_fingerprint"]
                    or registry["context"]["gene_ids"] != expected["rna_gene_ids"]
                    or registry["context"]["peak_ids"] != expected["atac_peak_ids"]
                    or family != layer_contract.get("response_model", "saturation")
                    or (family == "nn" and registry["context"].get("nn_response_config") != layer_contract.get("nn_response"))
                    or (family == "spline" and registry["context"].get("spline_response_config") != layer_contract.get("spline_response"))
                    or registry_settings != settings
                    or registry["numerical_signature"] != _source_signature(kind, bool(registry["context"].get("atac_representation")), registry["context"].get("response_family", "saturation"))):
                raise ValueError("saturation registry data/config/source contract mismatch")
        except (ValueError, KeyError) as exc:
            raise RuntimeError(f"{stage}: incompatible saturation registry: {exc}; {guidance}") from exc
    if stage == "step5_atac_to_rna":
        _validate_step5_binning_bundle(checkpoint["transfer_models"], expected["adapter_contract"],
                                       expected["atac_n_vars"], expected.get("atac_peak_ids"))


def _validate_step5_binning_bundle(bundle, adapter_contract, n_peaks, peak_ids=None):
    """Validate frozen current registry axes and bin metadata."""
    if adapter_contract.get("binning_model_contract_domain") == "saturating-hybrid-v1":
        from .saturating_transfer import validate_registry
        # The codec checks the schema against the actual response family and
        # ATAC representation, including NN and continuous saturation schemas.
        validate_registry(bundle, expected_type="saturating_hybrid_peak_gene")
        context = bundle["context"]
        if (context["n_bins"] != adapter_contract["n_bins"]
                or context["bin_centers"] != adapter_contract["bin_plan_centers"]
                or context["branch_boundaries"] != adapter_contract["branch_boundaries"]
                or (peak_ids is not None and context["peak_ids"] != list(peak_ids))):
            raise RuntimeError(f"step5: saturation context mismatch; {CHECKPOINT_RECOVERY_GUIDANCE}")
        return
    raise RuntimeError(f"step5: obsolete model contract; {CHECKPOINT_RECOVERY_GUIDANCE}")


def _step6_model_config(config):
    validate_model_mode(config)
    settings = copy.deepcopy(config.get("rna_to_atac", {}))
    for key in ("motif_scan_workers", "response_model", "nn_response", "spline_response"):
        settings.pop(key, None)
    from .saturating_transfer import response_contract, TF_TYPE
    settings.update(response_contract(config, TF_TYPE))
    return settings


def _step5_model_config(config):
    validate_model_mode(config)
    from .saturating_transfer import saturating_model_settings, response_contract, GENE_TYPE
    from .atac_normalization import validate_atac_depth_config
    depth = validate_atac_depth_config(config)
    return {"model_mode": "saturating_hybrid",
            # validate_saturating_config canonicalizes runtime workers to 1.
            "saturating_hybrid": saturating_model_settings(config),
            **response_contract(config, GENE_TYPE),
            **({"atac_depth_normalization": depth}
               if depth["enabled"] else {})}


def _joint_runtime_config(config):
    """Validate model settings without changing caller-selected output roots."""
    result = _gene_identity_runtime_config(config)
    validate_model_mode(result)
    result.setdefault("atac_to_rna", {}).setdefault("model_mode", "saturating_hybrid")
    if result.get("atac_to_rna", {}).get("model_mode") == "saturating_hybrid":
        from .saturating_transfer import validate_saturating_config
        validate_saturating_config(result)
        output = result.setdefault("output", {})
        from .atac_normalization import validate_atac_depth_config
        validate_atac_depth_config(result)
        output["model_mode"] = "saturating_hybrid"
        return result
    return result


def _identity_contract(rna):
    metadata = rna.uns.get("gene_identity")
    if not isinstance(metadata, dict):
        raise ValueError("gtf_gene_id RNA lacks gene_identity metadata")
    return {key: metadata[key] for key in ("schema", "fingerprint", "reference_sha256")}


def _validate_identity_axis(rna, identity):
    expected = {"schema": identity.metadata["schema"], "fingerprint": identity.fingerprint,
                "reference_sha256": identity.metadata["reference_sha256"]}
    if _identity_contract(rna) != expected:
        raise ValueError("RNA gene identity provenance differs from runtime reference")
    for gene_id in rna.var_names:
        if gene_id not in identity.by_id:
            raise ValueError(f"RNA model axis is not a reference gene ID: {gene_id}")
        record = identity.by_id[gene_id]
        row = rna.var.loc[gene_id]
        if row.get("strand") != record.strand or row.get("gene_tss") != record.gene_tss:
            raise ValueError(f"RNA strand/TSS differs from runtime gene identity: {gene_id}")
        if (row.get("gene_name"), row.get("gene_display_name"), row.get("chr"), row.get("start"), row.get("end")) != (
                record.gene_name, identity.display_labels[gene_id], record.chromosome, record.start, record.end):
            raise ValueError(f"RNA annotation differs from runtime gene identity: {gene_id}")


def _validate_runtime_identity(identity, config):
    if _resource_file_identity(config["input"]["annotation_gtf"])["sha256"] != identity.metadata["reference_sha256"]:
        raise ValueError("runtime gene identity index differs from configured GTF content")


def _prepare_gene_identity_inputs(rna, target_genes, gene_types, config, *, gene_identity=None):
    """Audit full-input membership and resolve tokens; not model preprocessing."""
    runtime = copy.deepcopy(config)
    if not _gene_identity_enabled(runtime):
        return rna, target_genes, gene_types, runtime, None, {}
    identity = gene_identity or parse_gene_gtf(runtime["input"]["annotation_gtf"])
    _validate_runtime_identity(identity, runtime)
    adapted = apply_gene_identity(rna, identity,
        gene_id_column=runtime["input"].get("rna_gene_id_column", "gene_ids"))
    allowed = list(adapted.adata.var_names)
    requested = list(target_genes)
    tokens = allowed if requested == ["__ALL__"] else requested
    resolved, types, records = [], {}, []
    for token in tokens:
        gene_id = resolve_gene_token(token, identity, allowed_ids=allowed, purpose="target")
        kind = gene_types.get(token, "TF")
        if gene_id in types and types[gene_id] != kind:
            raise ValueError(f"conflicting target types for gene ID {gene_id}")
        if gene_id not in resolved:
            resolved.append(gene_id)
        types[gene_id] = kind
        records.append({"requested_token": token, "gene_id": gene_id,
                        "gene_name": identity.by_id[gene_id].gene_name,
                        "gene_display_name": identity.display_labels[gene_id], "gene_type": kind})
    marker = runtime.get("pseudotime", {}).get("root_cell_marker")
    if marker is not None:
        runtime.setdefault("pseudotime", {})["root_cell_marker"] = resolve_gene_token(
            marker, identity, allowed_ids=allowed, purpose="root marker")
    metadata = dict(adapted.metadata, reference_path=identity.metadata["reference_path"],
                    requested_targets=requested, resolved_targets=resolved,
                    requested_root_marker=marker,
                    resolved_root_marker=runtime.get("pseudotime", {}).get("root_cell_marker"))
    tables = {"gene_filter_report": adapted.filter_report,
              "gene_identity": adapted.adata.var.reset_index(drop=True).copy(),
              "gene_target_resolution": pd.DataFrame(records), "gene_identity_metadata": metadata}
    return adapted.adata, resolved, types, runtime, identity, tables


def _prepare_identity_model_rna(original, reference, canonical_full, targets, config, identity):
    """Select model HVG on GTF-eligible RNA; reference geometry/cohort stay independent."""
    required = list(targets)
    marker = config.get("pseudotime", {}).get("root_cell_marker")
    if marker is not None:
        required.append(marker)
    # Full-input preparation already canonicalized IDs and recorded exclusions.
    # Do not apply GTF twice or use duplicate display symbols for retention.
    _validate_identity_axis(canonical_full, identity)
    model = preprocess_rna(canonical_full, **config["preprocess"]["rna"],
        keep_genes=list(dict.fromkeys(required)), filter_cells=False,
        retain_low_count_targets=True)
    # Reference PCA loadings are not a fit on the GTF-restricted feature axis.
    model.varm.pop("PCs", None)
    model.uns.pop("pca", None)
    _validate_identity_required_genes(model, targets, config)
    model = align_model_cells(model, reference)
    return attach_reference_clustering(model, reference,
        config.get("clustering", {}).get("cluster_key", "leiden"))


def _validate_identity_required_genes(rna, targets, config):
    if not _gene_identity_enabled(config):
        return
    required = list(targets)
    marker = config.get("pseudotime", {}).get("root_cell_marker")
    if marker is not None:
        required.append(marker)
    missing = [g for g in required if g not in rna.var_names]
    if missing:
        raise ValueError(f"required gene IDs removed by QC/HVG or absent in model context: {missing}")


def _fixed_grid_context(rna, atac, pt, config):
    """Checkpoint evidence for the current fixed grid, without fitting a plan."""
    from .kinetics import _adaptive_rebin
    from ._wt_aggregation import _weights
    pcfg = config["pseudotime"]
    smooth = pcfg.get("smooth_kernel", "hard")
    labels, bins = _adaptive_rebin(pt, rna.n_obs, max_bins=pcfg["n_bins"],
        min_cells_per_bin=pcfg.get("min_cells_per_bin", 5),
        target_bins=pcfg["n_bins"] if smooth == "gaussian" else None,
        overlap_factor=pcfg.get("smooth_overlap_factor") if smooth == "gaussian" else None)
    pt["bin"] = labels
    kernel, centers = _weights(pt, bins, config, None)
    boundaries = pt.attrs.get("branch_boundaries") or [(0, bins)]
    evidence = dict(labels=labels, centers=centers, kernel=kernel.toarray(),
                    times=pt.pseudotime.to_numpy(), cells=rna.obs_names.to_numpy(), boundaries=boundaries)
    fingerprint = _provenance_fingerprint(evidence)
    plan = SimpleNamespace(cell_ids=rna.obs_names.to_numpy(), times=pt.pseudotime.to_numpy(),
        branches=pt.branch.to_numpy() if "branch" in pt else np.repeat("__global__", rna.n_obs),
        centers=centers, fingerprint=fingerprint, n_bins=bins)
    return SimpleNamespace(plan=plan, policy=SimpleNamespace(fingerprint=fingerprint, branch_boundaries=boundaries),
                           metadata={"state_embedding": {"state_modality": "fixed_grid"}}, model_contract_fingerprint=fingerprint)


def _adaptive_namespace(context, suffix):
    """Compatibility helper: provenance validation does not change context paths."""
    return suffix

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CausalBridge")


def _aggregate_context_l0(context_registry, modeled_context_l0, gene_names,
                          sparse_results=None, min_delta_threshold=0.0,
                          values_are_log_l0=False):
    """Aggregate true per-cell L0 means across every post-QC context.

    ``context_registry`` contains every context, including skipped/failed
    contexts.  ``modeled_context_l0`` only contains successful reconstructions;
    missing contexts deliberately use their original means for both sides.
    """
    if not context_registry:
        return pd.DataFrame()
    genes = list(gene_names)
    original = {}
    for c, v in context_registry.items():
        if "original_log_l0_mean" in v:
            original[c] = np.asarray(v["original_log_l0_mean"], dtype=float)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                original[c] = np.log(np.asarray(v["original_l0_mean"], dtype=float))
    modeled = modeled_context_l0
    sparse_meta = (sparse_results.drop_duplicates(["target_gene", "affected_gene"])
                   if sparse_results is not None and not sparse_results.empty else pd.DataFrame())
    rows = []
    threshold = max(float(min_delta_threshold), 1e-6)
    total_contexts = len(context_registry)
    total_cells = sum(int(v["n_cells"]) for v in context_registry.values())
    for target, target_contexts in modeled.items():
        numerator = np.full(len(genes), -np.inf, dtype=float)
        modeled_contexts = 0
        modeled_cells = 0
        for context, info in context_registry.items():
            n = int(info["n_cells"])
            base = original[context]
            pert = target_contexts.get(context)
            if pert is None:
                numerator = np.logaddexp(numerator, base + np.log(n))
            else:
                if values_are_log_l0:
                    pert_log = np.asarray(pert, dtype=float)
                else:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        pert_log = np.log(np.asarray(pert, dtype=float))
                numerator = np.logaddexp(numerator, pert_log + np.log(n))
                modeled_contexts += 1
                modeled_cells += n
        # The loop starts from -inf so context contributions are pooled in log
        # space; denominator is pooled identically from original means.
        denominator = np.full(len(genes), -np.inf, dtype=float)
        for context, info in context_registry.items():
            denominator = np.logaddexp(
                denominator, original[context] + np.log(int(info["n_cells"]))
            )
        unchanged_contexts = total_contexts - modeled_contexts
        unchanged_cells = total_cells - modeled_cells
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = (numerator - denominator) / np.log(2.0)
        raw[np.isneginf(numerator) & np.isneginf(denominator)] = 0.0
        display = np.clip(np.nan_to_num(raw, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
        for i, gene in enumerate(genes):
            if gene in str(target).split(",") or abs(display[i]) < threshold:
                continue
            meta = (sparse_meta[(sparse_meta.target_gene == target) &
                                (sparse_meta.affected_gene == gene)]
                    if not sparse_meta.empty else pd.DataFrame())
            clipped = bool(not np.isfinite(raw[i]) or raw[i] != display[i])
            row = {"target_gene": target, "affected_gene": gene,
                   "delta_rna": abs(float(display[i])),
                   "delta_rna_signed": float(display[i]),
                   "n_clusters_observed": modeled_contexts,
                   "aggregation_method": "pooled_cell_l0_log2fc",
                   "aggregation_total_contexts": total_contexts,
                   "aggregation_modeled_contexts": modeled_contexts,
                   "aggregation_unchanged_contexts": unchanged_contexts,
                   "aggregation_total_cells": total_cells,
                   "aggregation_modeled_cells": modeled_cells,
                   "aggregation_unchanged_cells": unchanged_cells,
                   "l0_log2fc_unclipped": float(raw[i]),
                   "display_clipped": clipped}
            # Retain legacy columns, but do not pretend cell-L0 aggregation
            # has an L1 censoring count.
            for col in ["mediated_by_atac", "mechanism", "propagation_depth",
                        "converged", "is_fallback", "perturbation_mode"]:
                row[col] = (meta.iloc[0][col] if not meta.empty and col in meta
                            else False if col in ("mediated_by_atac", "is_fallback")
                            else True if col == "converged"
                            else "bin_forward" if col == "perturbation_mode" else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def _validate_step5_checkpoint(ckpt: dict, model_mode="saturating_hybrid", *, config=None) -> None:
    """Reject checkpoints produced by the removed non-NN Step-5 path."""
    transfer_models = ckpt.get("transfer_models")
    if model_mode == "saturating_hybrid":
        from .saturating_transfer import validate_registry
        from .atac_normalization import expected_registry_schema
        try:
            validate_registry(transfer_models, expected_type="saturating_hybrid_peak_gene",
                              expected_schema=expected_registry_schema("saturating_hybrid_peak_gene",config or {}))
        except ValueError as exc:
            raise RuntimeError(f"step5: incompatible saturation checkpoint: {exc}; {CHECKPOINT_RECOVERY_GUIDANCE}") from exc
        return
    raise RuntimeError(f"model mode {model_mode!r} has been removed; use saturating_hybrid; {CHECKPOINT_RECOVERY_GUIDANCE}")




def _project_bin_forward_inputs(inputs: dict, config: dict, context: str, target: str) -> tuple:
    """Project current CP10k intervention states; reject obsolete model payloads."""
    if inputs.get("effect_schema_version") != "bin_saturating_effect_v1":
        raise ValueError("obsolete projection payload; regenerate with saturating_hybrid")
    try:
        from .saturating_projection import project_saturating_inputs
        return project_saturating_inputs(inputs, config, context, target)
    except (ImportError, ValueError, KeyError) as exc:
        logger.warning("Saturating cell projection skipped for %s/%s: %s", context, target, exc)
        metadata = pd.DataFrame([dict(source_context=context, target_gene=target,
            effect_schema_version="bin_saturating_effect_v1", projection_status="unsupported_or_invalid",
            diagnostic=str(exc))])
        return pd.DataFrame(), None, pd.DataFrame(), metadata, pd.DataFrame()


def _cell_l0_from_saturating_payload(payload):
    """Never apply the L1 decoder to raw CP10k saturation states."""
    if payload.get("effect_schema_version") == "bin_saturating_effect_v1":
        from .saturating_projection import cell_l0_from_saturating_payload
        return cell_l0_from_saturating_payload(payload)
    raise ValueError("unsupported cell payload schema; joint_gene has been removed")


def _collect_saturating_table(contexts, key):
    """Collect one report without materializing unrelated context reports."""
    frames = []
    for context in contexts:
        frame = context.get(key)
        if isinstance(frame, pd.DataFrame):
            frame = frame.copy()
            frame.attrs = {}
            frame["source_context"] = str(context.get("cluster", "global"))
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _collect_saturating_tables(contexts, *, destination=None):
    result = {} if destination is None else destination
    result.update(model_mode="saturating_hybrid", effect_schema_version="bin_saturating_effect_v1")
    for key in ("saturating_gene_metrics", "saturating_tf_metrics", "saturating_response_weights",
                "saturating_support_diagnostics", "saturating_gates", "saturating_response_curves",
                "saturating_endpoints", "saturating_peak_attributions", "saturating_tf_peak_contributions",
                "l1_floor_diagnostics", "saturating_pathways"):
        result[key] = _collect_saturating_table(contexts, key)
    from .saturating_pathways import KEY, empty_pathways
    if result[KEY].empty:
        result[KEY] = empty_pathways()
    result["pathway_edges"] = result[KEY].copy()
    result["saturating_context_metadata"] = [c["saturating_context_metadata"] for c in contexts
                                             if "saturating_context_metadata" in c]
    return result


def _saturating_registry_reports(registry, layer):
    metrics, weights, curves = [], [], []
    for target, values in registry.get("target_metrics", {}).items():
        identity = {"gene": target} if layer == "peak_gene" else {"peak_id": target}
        metrics.append(dict(identity, layer=layer, target_id=target,
            input_unit=registry["input_unit"], output_unit=registry["output_unit"], **values))
    for target, values in registry.get("failures", {}).items():
        identity = {"gene": target} if layer == "peak_gene" else {"peak_id": target}
        values = {k:v for k,v in values.items() if k != "local_route_diagnostics"}
        metrics.append(dict(identity, layer=layer, target_id=target,
            input_unit=registry["input_unit"], output_unit=registry["output_unit"], **values))
    for target, bundle in registry.get("targets", {}).items():
        for index, (source, lag) in enumerate(bundle["input_keys"]):
            row = dict(layer=layer, target_id=target, input_id=source, lag=lag,
                lag_mode="local" if lag == "local" else "fixed",
                alpha=bundle["weights"][index], wt_response_score=bundle["scores"][index],
                included=True, fit_status=bundle["singles"][index]["metrics"]["status"],
                input_unit=registry["input_unit"], output_unit=registry["output_unit"])
            row.update(gene=target, peak_id=source) if layer == "peak_gene" else row.update(tf_gene=source, peak_id=target)
            weights.append(row)
            component = bundle["singles"][index]
            if component.get("response_type") == "spline":
                curves.append(dict(row, response_family="spline", b=np.nan, a=np.nan, c=np.nan, s=np.nan,
                    spline_schema=component["schema"], spline_knots_json=json.dumps(component["knots"]),
                    spline_coefficients_json=json.dumps(component["coefficients"]),
                    spline_response_config_json=json.dumps(component["config"]),
                    tail_scale=component["tail_scale"], bounds=component["bounds"], x_range=component["x_range"],
                    metrics=component["metrics"]))
            elif component.get("response_type") == "nn":
                curves.append(dict(row, response_family="nn", b=np.nan, a=np.nan, c=np.nan, s=np.nan,
                    nn_hidden_width=len(component["bias"]), nn_weights_json=json.dumps({k:component[k] for k in ("W", "bias", "v", "offset", "mean", "scale")}),
                    training_status=component["metrics"]["training_status"], epochs=component["metrics"].get("epochs", 0),
                    best_objective=component["metrics"].get("best_objective"), bounds=component["bounds"], x_range=component["x_range"]))
            else:
                curves.append(dict(row, **component))
    audits = [(target, bundle["provenance"].get("excluded_single_relations", {}))
              for target, bundle in registry.get("targets", {}).items()]
    audits += [(target, values.get("excluded_single_relations", {}))
               for target, values in registry.get("failures", {}).items()]
    for target, audit in audits:
        for record in audit.get("records", ()):
            source, lag = record["input_key"]
            row = dict(layer=layer, target_id=target, input_id=source, lag=lag,
                alpha=0., wt_response_score=None, included=False, fit_status=record["status"],
                training_rmse=record["training_rmse"], training_r2=record["training_r2"],
                nfev=record["nfev"], exclusion_reason=record["reason"],
                input_unit=registry["input_unit"], output_unit=registry["output_unit"])
            row.update(gene=target, peak_id=source) if layer == "peak_gene" else row.update(tf_gene=source, peak_id=target)
            weights.append(row)
    return pd.DataFrame(metrics), pd.DataFrame(weights), pd.DataFrame(curves)


def _aggregate_saturating_cell_means(endpoints, registry, min_delta=0):
    """Pool saved CP10k cell means over the same valid contexts per gene.

    Saturation endpoints are means of reconstructed cells, not bin endpoints.
    Unknown/quality-excluded contexts never supply an assumed zero effect.
    """
    required = {"source_context", "target_gene", "affected_gene", "status",
                "selected", "direct_target", "control_mean_cp10k", "perturbed_mean_cp10k"}
    if endpoints.empty:
        return pd.DataFrame(), pd.DataFrame()
    if not required.issubset(endpoints):
        raise ValueError("valid_cell_means requires saved per-cell CP10k endpoint means")
    table = endpoints.copy()
    table["source_context"] = table.source_context.astype(str)
    if table.duplicated(["source_context", "target_gene", "affected_gene"]).any():
        raise ValueError("duplicate saturation context/target/gene endpoints")
    counts = {str(c): int(info["n_cells"]) for c, info in registry.items()}
    if any(n < 0 for n in counts.values()):
        raise ValueError("negative context cell count")
    unknown = set(table.source_context) - set(counts)
    if unknown:
        raise ValueError(f"endpoint contexts absent from registry: {sorted(unknown)}")
    total = sum(counts.values())
    rows, diagnostics = [], []
    threshold = max(float(min_delta), 1e-6)
    for (target, gene), group in table.groupby(["target_gene", "affected_gene"], sort=True):
        if gene in str(target).split(",") or group.direct_target.any() or not group.selected.any():
            continue
        # Duplicate context keys were rejected above. Keep the registry iteration
        # and summation order, avoiding a mixed-dtype Series for every endpoint.
        indexed = dict(zip(group.source_context, zip(group.control_mean_cp10k,
            group.perturbed_mean_cp10k, group.status, group.selected)))
        used, excluded, assumptions = [], {}, []
        for context, n in counts.items():
            reason = None
            if n == 0:
                reason = "zero_cells"
            elif context not in indexed:
                reason = "endpoint_unavailable"
            else:
                wt, pert, status, selected = indexed[context]
                wt = float(wt)
                pert = float(pert)
                if status != "ok":
                    reason = str(status)
                elif not np.isfinite(wt) or not np.isfinite(pert) or wt <= 0 or pert <= 0:
                    reason = "invalid_or_nonpositive_cell_mean"
                elif not selected and not np.isclose(wt, pert, rtol=1e-10, atol=1e-12):
                    reason = "unselected_baseline_mismatch"
                else:
                    used.append((context, n, wt, pert))
                    if not selected:
                        assumptions.append(context)
            if reason is not None:
                excluded[context] = reason
                diagnostics.append(dict(source_context=context, target_gene=target,
                    affected_gene=gene, status=reason, aggregation_policy="valid_cell_means"))
        used_cells = sum(item[1] for item in used)
        wt_mean = pert_mean = effect = np.nan
        if used_cells:
            # Identical cell weights and denominator on both sides; no logFC averaging.
            wt_mean = sum((n / used_cells) * wt for _, n, wt, _ in used)
            pert_mean = sum((n / used_cells) * pert for _, n, _, pert in used)
            if np.isfinite(wt_mean) and np.isfinite(pert_mean) and wt_mean > 0 and pert_mean > 0:
                effect = float((np.log(pert_mean) - np.log(wt_mean)) / np.log(2.))
        if np.isfinite(effect) and abs(np.clip(effect, -10., 10.)) < threshold:
            continue
        display = float(np.clip(effect, -10., 10.)) if np.isfinite(effect) else np.nan
        rows.append(dict(target_gene=target, affected_gene=gene,
            delta_rna=abs(display), delta_rna_signed=display, l0_log2fc_unclipped=effect,
            display_clipped=bool(np.isfinite(effect) and effect != display),
            control_mean_cp10k=wt_mean, perturbed_mean_cp10k=pert_mean,
            status="ok" if np.isfinite(effect) else "no_valid_positive_cell_means",
            aggregation_policy="valid_cell_means", aggregation_method="pooled_cell_cp10k_log2fc",
            aggregation_scope="full" if len(used) == len(counts) else "partial" if used else "unavailable",
            aggregation_total_contexts=len(counts), aggregation_modeled_contexts=len(used),
            aggregation_used_contexts=len(used), aggregation_unavailable_contexts=len(excluded),
            aggregation_total_cells=total, aggregation_used_cells=used_cells,
            aggregation_excluded_cells=total-used_cells,
            aggregation_cell_fraction=used_cells / total if total else 0.,
            aggregation_context_exclusions=json.dumps(excluded, sort_keys=True),
            aggregation_used_context_labels=json.dumps([item[0] for item in used]),
            aggregation_baseline_assumption="unchanged_within_quality_pruned_network" if assumptions else "none",
            aggregation_baseline_contexts=json.dumps(assumptions), n_clusters_observed=len(used)))
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def _saturating_merge_contexts(contexts, registry, gene_names, min_delta=0, *, policy="strict_complete", endpoints=None):
    if policy not in ("strict_complete", "valid_cell_means"):
        raise ValueError(f"unknown saturation aggregation policy: {policy}")
    if endpoints is None:
        endpoints = _collect_saturating_table(contexts, "saturating_endpoints")
    if policy == "valid_cell_means":
        return _aggregate_saturating_cell_means(endpoints, registry, min_delta)
    modeled, diagnostics = {}, []
    for context in contexts:
        label = str(context.get("cluster", "global"))
        payloads = context.get("projection_inputs", [])
        if isinstance(payloads, dict):
            payloads = [payloads]
        for payload in payloads:
            context_endpoints = context.get("saturating_endpoints", pd.DataFrame())
            fallback_target = context_endpoints.target_gene.iloc[0] if not context_endpoints.empty else ""
            target = ",".join(payload.get("target_genes", ())) or str(payload.get("target_gene", fallback_target))
            modeled.setdefault(target, {})
            try:
                cell_ids, features, cells = _cell_l0_from_saturating_payload(payload)
                cells = np.asarray(cells, dtype=float)
                if cells.shape != (registry[label]["n_cells"], len(gene_names)) or not np.isfinite(cells).all() or (cells < 0).any():
                    raise ValueError("invalid saturation CP10k cell states")
                if list(features) != list(gene_names) or list(cell_ids) != list(payload["cell_ids"]):
                    raise ValueError("saturation cell gene axis differs from current model")
                modeled[target][label] = cells.mean(axis=0)
            except (ValueError, KeyError) as exc:
                diagnostics.append(dict(source_context=label, target_gene=target,
                    status="projection_unavailable", diagnostic=str(exc)))
    merged = _aggregate_context_l0(registry, modeled, gene_names, min_delta_threshold=min_delta)
    if merged.empty:
        if not any(modeled.values()) and not endpoints.empty:
            failed = endpoints.copy()
            if {"selected", "direct_target"}.issubset(failed):
                failed = failed[failed.selected & ~failed.direct_target].copy()
            failed["delta_rna"] = np.nan
            failed["status"] = failed.status.where(failed.status != "ok", "projection_unavailable")
            failed["aggregation_method"] = "pooled_cell_l0_log2fc"
            return failed, pd.DataFrame(diagnostics)
        return merged, pd.DataFrame(diagnostics)
    if not endpoints.empty and {"selected", "direct_target"}.issubset(endpoints):
        selected = endpoints[endpoints.selected & ~endpoints.direct_target]
        allowed = set(zip(selected.target_gene, selected.affected_gene))
        merged = merged[[pair in allowed for pair in zip(merged.target_gene, merged.affected_gene)]].copy()
    for index, row in merged.iterrows():
        bad = endpoints[(endpoints.target_gene == row.target_gene) &
                        (endpoints.affected_gene == row.affected_gene) & (endpoints.status != "ok")] if not endpoints.empty else pd.DataFrame()
        if merged.at[index, "aggregation_modeled_contexts"] != len(registry) or not bad.empty:
            for column in ("delta_rna", "delta_rna_signed", "l0_log2fc_unclipped"):
                merged.at[index, column] = np.nan
            merged.at[index, "status"] = ";".join(sorted(set(bad.status))) if not bad.empty else "incomplete_context_coverage"
    return merged, pd.DataFrame(diagnostics)


def _validate_step6_checkpoint(tf_weights) -> None:
    """Reject Step-6 checkpoints containing removed bagging statistics."""
    legacy_columns = {"weight_std", "bagging_stability"}.intersection(tf_weights.columns)
    if legacy_columns:
        raise RuntimeError(
            "step6 checkpoint contains legacy bagging-derived columns "
            f"({', '.join(sorted(legacy_columns))}); delete it and rerun Step 6."
        )


def _log_paths(config: dict):
    """运行开始时打印所有关键路径，避免缓存/断点混淆。"""
    inp = config.get("input", {})
    out = config.get("output", {})
    ckpt_dir = _checkpoint_dir(config)

    lines = [
        ("=" * 60),
        ("路径总览"),
        ("=" * 60),
        ("[输入]"),
        (f"  rna_h5ad:         {inp.get('rna_h5ad', '?')}"),
        (f"  atac_h5ad:        {inp.get('atac_h5ad', '?')}"),
        (f"  genome_fasta:     {inp.get('genome_fasta', '?')}"),
        (f"  target_genes:     {inp.get('target_genes', '?')}"),
        ("[输出]"),
        (f"  output_dir:       {out.get('dir', '?')}"),
        (f"  cache_dir:        {out.get('cache_dir', '?')}"),
        (f"  checkpoint_dir:   {ckpt_dir}"),
    ]
    for line in lines:
        logger.info(line)


def _checkpoint_dir(config: dict, suffix: str = None) -> Path:
    """断点保存目录。suffix 用于按簇分目录存储。

    优先使用 output.checkpoint_dir，未设置则回退到 output.dir/checkpoints。
    """
    from .atac_normalization import validate_atac_depth_config
    if validate_atac_depth_config(config)["enabled"]:
        config = _joint_runtime_config(config)
    config = _gene_identity_runtime_config(config)
    custom = config.get("output", {}).get("checkpoint_dir")
    if custom:
        base = Path(custom)
    else:
        base = Path(config["output"]["dir"]) / "checkpoints"
    return base / suffix if suffix else base


def _load_checkpoint(config: dict, name: str, suffix: str = None):
    """加载断点，不存在则返回 None。"""
    path = _checkpoint_dir(config, suffix) / f"{name}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            logger.info(f"从断点恢复: {name} ({path})")
            return data
        except Exception as e:
            logger.warning(f"断点加载失败 ({name}): {e}")
    return None


def _save_checkpoint(config: dict, name: str, data: dict, suffix: str = None):
    """保存断点。suffix 用于按簇分目录存储。"""
    ckpt_dir = _checkpoint_dir(config, suffix)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"断点已保存: {name} ({path})")


# ============================================================================
# 建模子流程 (Step 3-8)，可复用于全局或按簇模式
# ============================================================================

def _run_modeling_for_subset(
    rna_adata,
    atac_adata,
    target_genes: list,
    gene_types: dict,
    config: dict,
    suffix: str = None,
    rna_raw = None,
    gene_identity=None,
    defer_simulation: bool = False,
) -> dict:
    """
    对细胞子集执行建模步骤 (Step 3-8)。

    设计意图：先分簇，再对每簇独立推断伪时间——每个谱系有自己的发育轨迹，
    避免全局 DPT 跨越不同谱系导致伪时间排序失真。

    可用于全局模式或按簇模式。suffix 控制 checkpoint 子目录。
    """
    config = _joint_runtime_config(config)
    identity_mode = _gene_identity_enabled(config)
    if identity_mode:
        if gene_identity is None:
            gene_identity = parse_gene_gtf(config["input"]["annotation_gtf"])
        _validate_runtime_identity(gene_identity, config)
        _validate_identity_axis(rna_adata, gene_identity)
        if rna_raw is not None:
            _validate_identity_axis(rna_raw, gene_identity)
        _validate_identity_required_genes(rna_adata, target_genes, config)
        if any(g not in target_genes for g in gene_types):
            raise ValueError("gene_types must be keyed by resolved target IDs")
    if any(str(gene_types.get(g, "TF")).upper() != "TF" for g in target_genes):
        raise ValueError("saturating_hybrid does not support TARGET proxy interventions")
    _log = logger.getChild(suffix or "global")

    # ====================================================================
    # 自适应参数：小簇放宽检验标准，减少统计效力不足导致的假阴性
    # ====================================================================
    n_cells = rna_adata.n_obs
    import copy
    _cfg = copy.deepcopy(config)
    enabled = _selector(_cfg).get("enabled", False)
    if enabled and _cfg["pseudotime"].get("ablation_shuffle_pseudotime", False):
        raise ValueError("adaptive binning does not support shuffled pseudotime")
    binning_context = None
    plan_kwargs = {}

    # 1. DPT n_neighbors: 小簇降低邻居数，避免 KNN 图过于稠密
    default_n_neighbors = _cfg["pseudotime"]["n_neighbors"]
    adaptive_n_neighbors = min(default_n_neighbors, max(5, n_cells // 3))
    _cfg["pseudotime"]["n_neighbors"] = adaptive_n_neighbors

    # 2. Granger reference_bins: 不应超过该簇的实际 bin 数
    #    否则 correction = (df_ref/df_actual)² 会过度惩罚小簇
    min_cells_per_bin = _cfg["pseudotime"].get("min_cells_per_bin", 5)
    max_bins = _cfg["pseudotime"]["n_bins"]
    est_bins = max(5, min(n_cells // min_cells_per_bin, max_bins))
    adaptive_ref_bins = min(
        _cfg["granger"].get("reference_bins", 50),
        max(est_bins, 10),
    )
    _cfg["granger"]["reference_bins"] = adaptive_ref_bins

    # 3. Granger 复合分数阈值: 50 细胞 → 50% 基准, 200+ 细胞 → 100% 基准
    base_composite = _cfg["granger"].get("min_composite_score", 0.3)
    scale = min(1.0, n_cells / 200.0)
    _cfg["granger"]["min_composite_score"] = round(base_composite * (0.5 + 0.5 * scale), 3)

    if n_cells < 200:
        _log.info(
            f"自适应参数 (n_cells={n_cells}, est_bins={est_bins}): "
            f"n_neighbors={adaptive_n_neighbors}, "
            f"reference_bins={adaptive_ref_bins}, "
            f"min_composite_score={_cfg['granger']['min_composite_score']}"
        )

    # ====================================================================
    # Step 3: 伪时间推断（簇内独立——先分簇后伪时间，非先伪时间后分簇）
    # ====================================================================
    _log.info("--- Step 3: 伪时间推断（簇内独立） ---")
    if enabled:
        embedding, embedding_metadata = prepare_wt_embedding(
            rna_adata, atac_adata, _cfg["pseudotime"]["adaptive_binning"].get("embedding", {}))
        # State selection happens only inside fit_context_binning; trajectory
        # inference always receives this unchanged joint representation.
        pseudotime_df = infer_pseudotime(rna_adata, atac_adata, _cfg,
                                       defer_binning=True, joint_repr=embedding)
        binning_context = fit_context_binning(
            rna_adata, atac_adata, pseudotime_df, _cfg,
            embedding=embedding, embedding_metadata=embedding_metadata)
        validate_bin_plan(binning_context.plan, pseudotime_df, rna_adata.obs_names)
        validate_bin_plan(binning_context.plan, pseudotime_df, atac_adata.obs_names)
        plan_kwargs = {"bin_plan": binning_context.plan}
    else:
        pseudotime_df = infer_pseudotime(rna_adata, atac_adata, _cfg)
    checkpoint_context = binning_context
    if not enabled:
        checkpoint_context = _fixed_grid_context(rna_adata, atac_adata, pseudotime_df, _cfg)

    # 获取实际 bin 数，进一步校准 reference_bins
    actual_bins = checkpoint_context.plan.n_bins
    if actual_bins < est_bins * 0.8:
        _cfg["granger"]["reference_bins"] = max(actual_bins, 10)
        _log.info(
            f"reference_bins 二次校准: {adaptive_ref_bins} → "
            f"{_cfg['granger']['reference_bins']} (实际 bins={actual_bins})"
        )

    _cfg["granger"]["reference_bins"] = min(
        config["granger"].get("reference_bins", 50), max(actual_bins, 10))
    resource_config = copy.deepcopy(_cfg)
    motif_resources = _prepare_adaptive_motif_resources(resource_config)
    raw_scan_cache = (_raw_scan_cache_descriptor(resource_config, motif_resources["motif_db"])
                      if motif_resources["motif_db"] is not None else None)
    checkpoint_metadata = _adaptive_checkpoint_metadata(
        checkpoint_context, _cfg, target_genes, gene_types, rna_raw,
        rna=rna_adata, atac=atac_adata, n_peaks=atac_adata.n_vars, motif_resources=motif_resources)
    checkpoint_suffix = suffix
    upstream = {}
    def stage_load(name):
        checkpoint = _load_checkpoint(config, name, checkpoint_suffix)
        if identity_mode and checkpoint is None and (_checkpoint_dir(config, checkpoint_suffix) / f"{name}.pkl").exists():
            raise RuntimeError(f"{name}: unreadable gene identity checkpoint; {CHECKPOINT_RECOVERY_GUIDANCE}")
        if checkpoint is None and (_checkpoint_dir(config, checkpoint_suffix) / f"{name}.pkl").exists():
            raise RuntimeError(f"{name}: unreadable adaptive checkpoint; {CHECKPOINT_RECOVERY_GUIDANCE}")
        if checkpoint is not None:
            # Deliberately outside _load_checkpoint's broad deserialization catch.
            _validate_adaptive_checkpoint(checkpoint, checkpoint_metadata, name, upstream.get(name))
        return checkpoint
    def stage_save(name, payload):
        payload = dict(payload, adaptive_checkpoint=_stage_checkpoint_metadata(
            checkpoint_metadata, name, upstream.get(name)))
        _save_checkpoint(config, name, payload, checkpoint_suffix)


    # ====================================================================
    # 消融实验: 随机打乱伪时间值
    # 打乱 pseudotime 值本身, 使下游 _adaptive_rebin 的 pd.qcut
    # 对乱序值做分位数切割, 每个 bin 聚合的是随机细胞,
    # 从根源破坏时间轴排序, 验证伪时间是否为因果推断的核心信息源
    # ====================================================================
    _ablation_shuffle = _cfg["pseudotime"].get("ablation_shuffle_pseudotime", False)
    if _ablation_shuffle:
        import numpy as np
        from .kinetics import _adaptive_rebin
        from .granger import _bin_expression
        
        _old_pt = pseudotime_df["pseudotime"].values.copy()
        _old_bin = pseudotime_df["bin"].values.copy() if "bin" in pseudotime_df.columns else None
        
        # 保存打乱前的分箱结果用于比较
        pcfg = _cfg["pseudotime"]
        _smooth_kernel = pcfg.get("smooth_kernel", "hard")
        _overlap = pcfg.get("smooth_overlap_factor") if _smooth_kernel == "gaussian" else None
        old_bin_indices, old_n_bins = _adaptive_rebin(
            pseudotime_df, rna_adata.n_obs,
            max_bins=pcfg["n_bins"],
            min_cells_per_bin=pcfg.get("min_cells_per_bin", 5),
            target_bins=pcfg["n_bins"] if _smooth_kernel == "gaussian" else None,
            overlap_factor=_overlap,
        )
        pseudotime_df["bin"] = old_bin_indices
        rna_binned_before = _bin_expression(rna_adata, pseudotime_df, old_n_bins,
                                           smooth="hard",
                                           min_cells_per_bin=pcfg.get("min_cells_per_bin", 5))
        
        # 更激进的打乱方式：完全随机分配细胞到 bin
        # 而不是打乱伪时间值（因为 qcut 按分位数切割，打乱值可能效果有限）
        rng = np.random.default_rng(42)
        n_cells = len(pseudotime_df)
        n_bins = old_n_bins
        
        # 生成随机的 bin 分配（每个细胞随机分配到一个 bin）
        random_bins = rng.integers(0, n_bins, size=n_cells)
        pseudotime_df["bin"] = random_bins
        
        # 打乱伪时间值（保持一致性）
        rng.shuffle(_old_pt)
        pseudotime_df["pseudotime"] = _old_pt
        
        # 打乱后 branch_boundaries 无意义, 清除
        if hasattr(pseudotime_df, "attrs"):
            pseudotime_df.attrs["branch_boundaries"] = None
        
        # 强制使用 hard 模式，避免 gaussian 平滑掉打乱效果
        _original_smooth = _cfg["pseudotime"].get("smooth_kernel", "hard")
        if _original_smooth == "gaussian":
            _cfg["pseudotime"]["smooth_kernel"] = "hard"
            _log.info(f"  消融实验: gaussian 模式会平滑掉打乱效果，强制切换为 hard 模式")
        
        # 计算打乱后的分箱结果
        rna_binned_after = _bin_expression(rna_adata, pseudotime_df, n_bins,
                                          smooth="hard",
                                          min_cells_per_bin=pcfg.get("min_cells_per_bin", 5))
        
        _log.info(
            f"=== 消融: 细胞随机分配到 bin === "
            f"(seed=42, smooth_kernel=hard, 完全随机分配，不依赖伪时间)"
        )
        _log.info(f"  打乱前伪时间范围: [{_old_pt.min():.4f}, {_old_pt.max():.4f}]")
        _log.info(f"  打乱后伪时间范围: [{pseudotime_df['pseudotime'].min():.4f}, {pseudotime_df['pseudotime'].max():.4f}]")
        
        _log.info(f"  打乱前 bin 分布: {np.bincount(old_bin_indices.astype(int))[:5]}...")
        _log.info(f"  打乱后 bin 分布: {np.bincount(random_bins.astype(int))[:5]}...")
        
        # 检查 bin 是否真的变化了
        bin_changed = not np.array_equal(old_bin_indices, random_bins)
        _log.info(f"  bin 是否变化: {bin_changed}")
        
        # 调试：检查 pseudotime_df["bin"] 是否真的被更新了
        _log.info(f"  [DEBUG] pseudotime_df['bin'] 前5个值: {pseudotime_df['bin'].values[:5]}")
        _log.info(f"  [DEBUG] random_bins 前5个值: {random_bins[:5]}")
        _log.info(f"  [DEBUG] pseudotime_df['bin'] 是否等于 random_bins: {np.array_equal(pseudotime_df['bin'].values, random_bins)}")
        
        # 调试：检查 rna_binned_before 和 rna_binned_after 是否真的不同
        _log.info(f"  [DEBUG] rna_binned_before 形状: {rna_binned_before.shape}")
        _log.info(f"  [DEBUG] rna_binned_after 形状: {rna_binned_after.shape}")
        _log.info(f"  [DEBUG] rna_binned_before 前5个值: {rna_binned_before[0, :5]}")
        _log.info(f"  [DEBUG] rna_binned_after 前5个值: {rna_binned_after[0, :5]}")
        
        # 检查分箱结果是否真的变化了
        rna_diff = np.abs(rna_binned_after - rna_binned_before).max()
        _log.info(f"  打乱前后 RNA 分箱最大差异: {rna_diff:.6f}")
        if rna_diff < 1e-6:
            _log.warning("  ⚠️ 警告: 打乱前后 RNA 分箱结果几乎相同！")
        else:
            _log.info(f"  ✓ 打乱前后 RNA 分箱结果不同，差异显著")
        
        # 标记下游模块需要重新计算 bin
        pseudotime_df.attrs["_need_rebin"] = True

    # ====================================================================
    # Step 4: Granger 因果检验
    # ====================================================================
    _log.info("--- Step 4: Granger 因果检验 (peak→gene) ---")
    _log.info(
        "核心原理: 利用 ATAC-RNA 时间滞后效应 —— "
        "染色质开放先于转录激活"
    )

    ckpt = stage_load("step4_granger")
    if enabled:
        diagnostic_key = _provenance_fingerprint({"context": suffix or "global", "ids": binning_context.plan.cell_ids})[:20]
        diagnostic_dir = Path(config["output"]["dir"]) / "binning" / diagnostic_key
        refs = persist_binning_diagnostics(binning_context, diagnostic_dir)
        binning_output_metadata = dict(binning_context.metadata,
            checkpoint_metadata=checkpoint_metadata, context=suffix or "global",
            diagnostic_directory=str(diagnostic_dir), diagnostics_refs=refs,
            bin_plan_centers=binning_context.plan.centers.tolist(), n_bins=actual_bins,
            raw_motif_cache=raw_scan_cache)
    if ckpt is not None:
        granger_results = ckpt["granger_results"]
    else:
        granger_results = granger_test(
            rna_adata, atac_adata, pseudotime_df, _cfg, **plan_kwargs
        )
        stage_save("step4_granger", {"granger_results": granger_results})

    if granger_results.empty:
        _log.error("Granger 检验未发现任何因果对")
        return dict(error="no_causal_pairs", **({"binning_metadata": binning_output_metadata} if enabled else {}))

    # ====================================================================
    # Step 5: ATAC→RNA 传递函数拟合
    # ====================================================================
    _log.info("--- Step 5: ATAC→RNA 传递函数拟合 ---")
    model_mode = validate_model_mode(config)
    _log.info("方法: saturating_hybrid，peak→gene 响应=%s", _cfg.get("atac_to_rna", {}).get("response_model", "saturation"))

    upstream["step5_atac_to_rna"] = _provenance_fingerprint(granger_results)
    _bind_step5_contract(checkpoint_metadata, checkpoint_context, pseudotime_df, granger_results)
    ckpt = stage_load("step5_atac_to_rna")
    if ckpt is not None:
        causal_edges = ckpt["causal_edges"]
        transfer_functions = ckpt["transfer_functions"]
        transfer_models = ckpt.get("transfer_models")
        _validate_step5_checkpoint(ckpt, model_mode,config=_cfg)
    else:
        causal_edges = build_causal_grn(granger_results)
        transfer_functions, transfer_models = fit_atac_to_rna(
            rna_adata, atac_adata, pseudotime_df, causal_edges, _cfg,
            granger_results=granger_results,
            **plan_kwargs,
        )
        _validate_step5_binning_bundle(transfer_models, checkpoint_metadata["adapter_contract"],
                                       atac_adata.n_vars, list(atac_adata.var_names))
        stage_save("step5_atac_to_rna", {
            "causal_edges": causal_edges,
            "transfer_functions": transfer_functions,
            "transfer_models": transfer_models,
        })

    # ====================================================================
    # Step 6: RNA→ATAC TF 调控权重学习
    # ====================================================================
    _log.info("--- Step 6: RNA→ATAC TF 调控权重学习 ---")
    _log.info(
        "方法: motif 扫描确定候选 TF + 逐 TF/逐 lag 单特征 Pearson 确定调控权重"
    )

    upstream["step6_rna_to_atac"] = _provenance_fingerprint(
        [granger_results, causal_edges, transfer_functions, transfer_models])
    if _resource_identity(resource_config, motif_resources) != checkpoint_metadata["resource_identity"]:
        raise RuntimeError(f"Step6 resources changed after provenance capture; {CHECKPOINT_RECOVERY_GUIDANCE}")
    tf_model_registry = {}
    ckpt = stage_load("step6_rna_to_atac")
    if ckpt is not None:
        tf_weights = ckpt["tf_weights"]
        _validate_step6_checkpoint(tf_weights)
        from .saturating_transfer import validate_registry
        from .atac_normalization import expected_registry_schema
        tf_model_registry = ckpt.get("tf_model_registry")
        validate_registry(tf_model_registry, expected_type="saturating_hybrid_tf_peak",
                          expected_schema=expected_registry_schema("saturating_hybrid_tf_peak",_cfg))
    else:
        tf_weights = fit_rna_to_atac(
            rna_adata, atac_adata, pseudotime_df, causal_edges, _cfg,
            target_gene_types=gene_types,
            rna_raw=rna_raw,
            **({"tf_model_registry": tf_model_registry}),
            **plan_kwargs,
            **({"motif_resources": motif_resources}),
            **({"raw_scan_cache": raw_scan_cache}),
            **({"gene_identity": gene_identity} if identity_mode else {}),
        )
        stage_save("step6_rna_to_atac", {"tf_weights": tf_weights,
            **({"tf_model_registry": tf_model_registry})})

    # ====================================================================
    # Step 7: 因果 GRN 整合
    # ====================================================================
    _log.info("--- Step 7: 因果 GRN 整合 ---")
    n_tf_edges = len(tf_weights)
    n_peak_gene_edges = len(causal_edges)
    n_atac_rna_edges = len(transfer_functions)
    _log.info(
        f"网络统计: 因果 peak→gene 边={n_peak_gene_edges}, "
        f"TF→peak 关系={n_tf_edges}, "
        f"ATAC→RNA 传递函数={n_atac_rna_edges}, "
        f"基因数={len(causal_edges['gene'].unique())}"
    )

    # ====================================================================
    # Freeze the exact post-training context; workers never revisit Steps 3-7.
    simulation_inputs = dict(
        rna_adata=rna_adata, atac_adata=atac_adata, target_genes=target_genes,
        gene_types=gene_types, config=config, _cfg=_cfg, suffix=suffix,
        pseudotime_df=pseudotime_df, plan_kwargs=plan_kwargs,
        enabled=enabled, identity_mode=identity_mode,
        binning_output_metadata=binning_output_metadata if enabled else None)
    if defer_simulation:
        from .simulation_queue import save_simulation_job
        stages = ("step4_granger", "step5_atac_to_rna", "step6_rna_to_atac")
        manifest = save_simulation_job(
            simulation_inputs,
            {stage: _checkpoint_dir(config, checkpoint_suffix) / f"{stage}.pkl" for stage in stages},
            checkpoint_metadata, upstream)
        return {"simulation_manifest": manifest}
    return _finish_modeling_for_subset(
        **simulation_inputs, granger_results=granger_results,
        causal_edges=causal_edges, transfer_functions=transfer_functions,
        transfer_models=transfer_models, tf_weights=tf_weights,
        tf_model_registry=tf_model_registry)



def _merge_atac_changes(cluster_results, cluster_cell_counts, config):
    """Keep separate KO conditions apart during cell-weighted ATAC merging."""
    frames = []
    for result in cluster_results:
        frame = result.get("atac_changes")
        if frame is None or frame.empty:
            continue
        frame = frame.copy()
        frame["cluster"] = str(result.get("cluster", ""))
        frame["_weight"] = cluster_cell_counts.get(str(result.get("cluster", "")), 0)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    all_changes = pd.concat(frames, ignore_index=True)
    total_cells = sum(cluster_cell_counts.values())
    if total_cells <= 0 or len(cluster_cell_counts) <= 1:
        return pd.concat([r["atac_changes"] for r in cluster_results
                          if r.get("atac_changes") is not None and not r["atac_changes"].empty],
                         ignore_index=True)
    keys = (["target_gene"] if "target_gene" in all_changes else []) + ["peak_id"]
    rows = []
    for identity, group in all_changes.groupby(keys, sort=True, dropna=False):
        if group["_weight"].sum() == 0:
            continue
        if not isinstance(identity, tuple):
            identity = (identity,)
        row = dict(zip(keys, identity))
        row.update(chr=group["chr"].iloc[0], start=int(group["start"].iloc[0]),
                   end=int(group["end"].iloc[0]),
                   delta_accessibility_signed=float((group.delta_accessibility * group._weight).sum() / total_cells),
                   delta_accessibility=float((group.delta_accessibility.abs() * group._weight).sum() / total_cells),
                   n_clusters_observed=group.cluster.nunique())
        for unit in ("atac_state_unit", "atac_intervention_unit"):
            if unit in group:
                row[unit] = group[unit].iloc[0]
        if "z_score" in group:
            row["z_score"] = float((group.z_score * group._weight).sum() / total_cells)
            row["is_significant"] = abs(row["z_score"]) >= config.get("perturbation", {}).get("atac_significance_zscore", 2.)
        if "status" in group and group.status.eq("unavailable").any():
            row.update(status="unavailable", delta_accessibility=np.nan,
                       delta_accessibility_signed=np.nan, z_score=np.nan, is_significant=False)
        rows.append(row)
    return pd.DataFrame(rows)


def _finish_modeling_for_subset(*, rna_adata, atac_adata, target_genes,
        gene_types, config, _cfg, suffix, pseudotime_df, plan_kwargs, enabled,
        identity_mode, binning_output_metadata, granger_results, causal_edges,
        transfer_functions, transfer_models, tf_weights, tf_model_registry):
    """Shared simulation, cell projection and reporting for a fitted cohort."""
    _log = logger.getChild(suffix or "global")
    checkpoint_suffix = suffix
    # Step 8: 扰动模拟 (knockout_type 开关: "gene"=TF/基因 KO, "peak"=peak KO)
    # ====================================================================
    ko_type = _cfg.get("perturbation", {}).get("knockout_type", "gene")
    if ko_type == "peak":
        _log.info("--- Step 8: Peak 扰动模拟 (direct 模式) ---")
        _log.info("方法: 指定 peak 的 Δ可及性 → 沿现有 peak→gene 传递函数传播")
        peak_cfg = _cfg.get("perturbation", {}).get("peak_ko", {})
        if not peak_cfg.get("peak_id"):
            _log.error("  knockout_type=peak 但未配置 perturbation.peak_ko.peak_id")
            peak_cfg["peak_id"] = ""
        pert_output = perturb_peak(
            peak_cfg["peak_id"],
            rna_adata,
            atac_adata,
            pseudotime_df,
            causal_edges,
            transfer_functions,
            tf_weights,
            _cfg,
            transfer_models=transfer_models,
            **({"tf_model_registry": tf_model_registry, "cluster_label": suffix or "global"}),
            **plan_kwargs,
        )
        aggregated_tf_peak = pd.DataFrame()
    else:
        _log.info("--- Step 8: 扰动模拟 (bin_forward 逐 bin 前向传播) ---")
        _log.info(
            "方法: [RNA→ATAC] 变化的 TF → 受影响 peak 重算可及性; "
            "[ATAC→RNA] 变化的 peak → 受影响基因重算表达"
        )
        pert_output = propagate_perturbation(
            target_genes,
            rna_adata,
            atac_adata,
            pseudotime_df,
            causal_edges,
            transfer_functions,
            tf_weights,
            _cfg,
            transfer_models=transfer_models,
            gene_types=gene_types,
            **({"tf_model_registry": tf_model_registry, "cluster_label": suffix or "global"}),
            **plan_kwargs,
        )
        aggregated_tf_peak = pert_output.get("aggregated_tf_peak_edges", pd.DataFrame())

    if "error" in pert_output:
        _log.warning(f"  扰动失败: {pert_output.get('error')}")
        perturbation_results = pd.DataFrame()
        pathway_edges = pd.DataFrame()
        upstream_tfs = pd.DataFrame()
    else:
        perturbation_results = pert_output["perturbation_results"]
        pathway_edges = pert_output.get("pathway_edges", pd.DataFrame())
        upstream_tfs = pert_output.get("upstream_tfs", pd.DataFrame())
    projection_table = pd.DataFrame()
    projection_inputs = pert_output.get("projection_inputs", [])
    if isinstance(projection_inputs, dict):
        projection_inputs = [projection_inputs]
    pert_output["projection_inputs"] = projection_inputs
    if _cfg.get("perturbation", {}).get("cell_projection", {}).get("enabled", False):
        projection_frames = []
        history_frames = []
        metadata_frames = []
        event_frames = []
        for projection_input in pert_output.get("projection_inputs", []):
            projected, _, history, metadata, events = _project_bin_forward_inputs(
                projection_input, _cfg, suffix or "global",
                (",".join(projection_input.get("target_genes", target_genes)) or
                 (f"PEAK:{pert_output.get('peak_id', '')}" if ko_type == "peak" else ",".join(target_genes))),
            )
            if not projected.empty:
                projection_frames.append(projected)
            if not history.empty:
                history_frames.append(history)
            if not metadata.empty:
                metadata_frames.append(metadata)
            if not events.empty:
                event_frames.append(events)
        if projection_frames:
            projection_table = pd.concat(projection_frames, ignore_index=True)
        projection_history = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
        projection_metadata = pd.concat(metadata_frames, ignore_index=True) if metadata_frames else pd.DataFrame()
        projection_events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    else:
        projection_history = pd.DataFrame()
        projection_metadata = pd.DataFrame()
        projection_events = pd.DataFrame()
    output = {
        "causal_edges": causal_edges,
        "transfer_functions": transfer_functions,
        "tf_peak_weights": tf_weights,
        "perturbation_results": perturbation_results,
        "pathway_edges": pathway_edges,
        "aggregated_tf_peak_edges": aggregated_tf_peak,
        "atac_changes": pert_output.get("atac_changes", pd.DataFrame()),
        "upstream_tfs": upstream_tfs,
        "peak_id": pert_output.get("peak_id", ""),
        "prediction_mode": pert_output.get("prediction_mode", ""),
        "baseline_accessibility": pert_output.get("baseline_accessibility"),
        "perturbed_accessibility": pert_output.get("perturbed_accessibility"),
        "delta_accessibility": pert_output.get("delta_accessibility"),
        "n_affected_edges": pert_output.get("n_affected_edges", 0),
        "granger_results": granger_results,
        "pseudotime": pseudotime_df,
        "n_cells": rna_adata.n_obs,
        # Required by merged cell-L0 aggregation even when the optional
        # projection report is disabled.
        "projection_inputs": pert_output.get("projection_inputs", []),
    }
    if _cfg.get("perturbation", {}).get("cell_projection", {}).get("enabled", False):
        output["cell_projection"] = projection_table
        output["projection_history"] = projection_history
        output["projection_bin_metadata"] = projection_metadata
        output["projection_events"] = projection_events
    if enabled:
        output["binning_metadata"] = binning_output_metadata
    output.update(model_mode="saturating_hybrid", effect_schema_version="bin_saturating_effect_v1")
    gm, gw, gc = _saturating_registry_reports(transfer_models, "peak_gene")
    tm, tw, tc = _saturating_registry_reports(tf_model_registry, "tf_peak")
    output["saturating_gene_metrics"] = gm
    output["saturating_tf_metrics"] = tm
    for key in ("saturating_response_weights", "saturating_support_diagnostics",
                "saturating_gates", "saturating_response_curves", "saturating_endpoints",
                "saturating_peak_attributions", "saturating_tf_peak_contributions", "l1_floor_diagnostics",
                "saturating_pathways"):
        output[key] = pert_output.get(key, pd.DataFrame())
    output["saturating_response_weights"] = pd.concat([gw, tw], ignore_index=True)
    output["saturating_response_curves"] = pd.concat([gc, tc], ignore_index=True)
    output["saturating_gates"] = pd.DataFrame([
        dict(layer=layer, target_id=target, gene=target if layer == "peak_gene" else None,
             peak_id=target if layer == "tf_peak" else None, status=bundle["status"],
             output_bounds=json.dumps(bundle["bounds"]),
             diagnostics_json=json.dumps(bundle["diagnostics"], allow_nan=False))
        for layer, registry in (("peak_gene", transfer_models), ("tf_peak", tf_model_registry))
        for target, bundle in registry["targets"].items()])
    output["saturating_context_metadata"] = dict(
        context=suffix or "global", model_mode="saturating_hybrid",
        effect_schema_version="bin_saturating_effect_v1",
        effective_config=_step5_model_config(_cfg)["saturating_hybrid"],
        gene_response_model=transfer_models["context"].get("response_family", "saturation"),
        tf_response_model=tf_model_registry["context"].get("response_family", "saturation"),
        response_settings={"peak_gene":_step5_model_config(_cfg), "tf_peak":_step6_model_config(_cfg)},
        quality_policy={k: v for k, v in _cfg.get("perturbation", {}).items()
                        if k in ("min_r2_threshold", "min_delta_threshold", "ko_strength")},
        consumed_quality_policy=pert_output.get("quality_policy"),
        excluded_gene_ids=pert_output.get("excluded_gene_ids", []),
        input_unit=transfer_models["input_unit"], output_unit=transfer_models["output_unit"],
        effective_output_dir=config["output"]["dir"],
        effective_checkpoint_dir=str(_checkpoint_dir(config, checkpoint_suffix)),
        diagnostics=pert_output.get("saturating_context_metadata", {}))
    if transfer_models["context"].get("atac_representation"):
        output["saturating_context_metadata"]["atac_representation"]=transfer_models["context"]["atac_representation"]
    if identity_mode:
        output["gene_identity_metadata"] = dict(_identity_contract(rna_adata),
            effective_output_dir=config["output"]["dir"],
            effective_checkpoint_dir=str(_checkpoint_dir(config, checkpoint_suffix)))
    return output


# ============================================================================
# 主 Pipeline
# ============================================================================

@_with_pipeline_source_snapshot
def run_pipeline(config_path: str) -> dict:
    """
    CausalBridge 完整分析流程。

    Parameters
    ----------
    config_path : str
        YAML 配置文件的路径

    Returns
    -------
    results : dict
        包含所有模块输出和最终结果的字典
    """
    # ========================================================================
    # Step 1: 加载配置 + 数据
    # ========================================================================
    logger.info("=" * 60)
    logger.info("Step 1/8: 加载配置和数据")
    logger.info("=" * 60)

    config = _joint_runtime_config(load_config(config_path))
    # Resolve every consumed numerical function before loading data or fitting
    # any cluster. Later checkpoint operations use these loaded-code identities,
    # not inspect line offsets into files edited during a long run.
    mode = validate_model_mode(config)
    identity_mode = config.get("input", {}).get("gene_identity_mode", "legacy")
    for stage in ("step4_granger", "step5_atac_to_rna", "step6_rna_to_atac"):
        _stage_numerical_signatures(stage, mode, gene_identity_mode=identity_mode)
    from .simulation_queue import execution_options, run_simulation_jobs
    defer_simulation, simulation_workers = execution_options(config)

    # 打印所有路径，避免缓存/断点/数据混淆
    _log_paths(config)
    rna_raw, atac_raw = load_data(config)
    # Preserve the full original RNA gene universe before QC/GTF/HVG filtering.
    original_full_raw_rna = rna_raw.copy()
    from .atac_normalization import validate_atac_depth_config, prepare_atac_counts, depth_log1p_model_view
    atac_depth_settings = validate_atac_depth_config(config)
    if atac_depth_settings["enabled"]:
        atac_raw = prepare_atac_counts(atac_raw)
    target_genes, gene_types = load_target_genes(config["input"]["target_genes"])
    # Clone original counts before any GTF adaptation. Reference QC/labels never
    # consume model feature membership, root-marker aliases or model mode.
    reference_rna = None
    if _gene_identity_enabled(config):
        model_original = rna_raw.copy()
        reference_rna, reference_atac = build_reference_clustering(
            rna_raw, atac_raw, target_genes, config)
        rna_raw = model_original
    rna_raw, target_genes, gene_types, config, gene_identity, identity_tables = _prepare_gene_identity_inputs(
        rna_raw, target_genes, gene_types, config)
    if any(str(gene_types.get(g, "TF")).upper() != "TF" for g in target_genes):
        raise ValueError("saturating_hybrid does not support TARGET proxy interventions")
    Path(config["output"]["dir"]).mkdir(parents=True, exist_ok=True)

    # 如果用户指定了全基因组模式，展开为所有表达基因
    if target_genes == ["__ALL__"]:
        target_genes = list(rna_raw.var_names)
        gene_types = {g: "TF" for g in target_genes}  # 全基因组默认 TF
        logger.info(f"全基因组模式: {len(target_genes)} 个基因")

    # ========================================================================
    # Step 2: 预处理
    # ========================================================================
    logger.info("=" * 60)
    logger.info("Step 2/8: 数据预处理")
    logger.info("=" * 60)

    if reference_rna is not None:
        rna_adata = _prepare_identity_model_rna(model_original, reference_rna, rna_raw,
            target_genes, config, gene_identity)
        atac_adata = reference_atac
    else:
        rna_adata = preprocess_rna(
            rna_raw,
            min_cells_per_gene=config["preprocess"]["rna"]["min_cells_per_gene"],
            min_umi_per_cell=config["preprocess"]["rna"]["min_umi_per_cell"],
            n_highly_variable_genes=config["preprocess"]["rna"]["n_highly_variable_genes"],
            keep_genes=target_genes,
        )
        _validate_identity_required_genes(rna_adata, target_genes, config)
        atac_adata = preprocess_atac(
            atac_raw,
            min_cells_per_peak=config["preprocess"]["atac"]["min_cells_per_peak"],
            min_peaks_per_cell=config["preprocess"]["atac"]["min_peaks_per_cell"],
            binarize=config["preprocess"]["atac"]["binarize"],
        )

    # 确保 RNA 和 ATAC 数据集的细胞完全匹配
    if reference_rna is None:
        rna_adata, atac_adata = match_cells(rna_adata, atac_adata)
    if atac_depth_settings["enabled"]:
        atac_adata = depth_log1p_model_view(atac_adata, atac_depth_settings)

    # Export global model HVG once, before per-cluster modeling; never overwrite
    # it with a subset's axis or include forced non-HVG targets/markers.
    save_hvg_genes(rna_adata, config)

    # 聚类: PCA + KNN + Leiden，为后续按簇建模提供标签
    cluster_cfg = config.get("clustering", {})
    cluster_key = cluster_cfg.get("cluster_key", "leiden")
    clustering_enabled = cluster_cfg.get("enabled", True)

    if clustering_enabled:
        if cluster_key in rna_adata.obs.columns:
            logger.info(
                f"检测到已有聚类标签 '{cluster_key}' "
                f"({rna_adata.obs[cluster_key].nunique()} 个簇)，跳过重新聚类"
            )
        else:
            logger.info(f"未找到 '{cluster_key}' 标签，执行 PCA + KNN + Leiden 聚类")
            rna_adata = cluster_cells(
                rna_adata,
                n_neighbors=cluster_cfg.get("n_neighbors", 20),
                n_pcs=cluster_cfg.get("n_pcs", 20),
                resolution=cluster_cfg.get("resolution", 0.8),
                key_added=cluster_key,
                random_state=cluster_cfg.get("random_state", 42),
            )

        # 输出每个簇的 top50 高表达基因（排除线粒体），供验证聚类映射
        save_cluster_top_genes(
            rna_adata,
            cluster_key=cluster_key,
            n_top=50,
            output_path=str(
                Path(config["output"]["dir"]) / f"cluster_top50_genes.csv"
            ),
        )

        # 导出聚类后的 RNA AnnData（含 PCA、KNN 图、Leiden 标签）
        clustered_h5ad_path = str(
            Path(config["output"]["dir"]) / "rna_clustered.h5ad"
        )
        normalize_anndata_string_metadata(rna_adata)
        rna_adata.write_h5ad(clustered_h5ad_path)
        logger.info(f"聚类后 RNA 数据已保存至: {clustered_h5ad_path}")

        # ====================================================================
        # PAGA lineage 构建: 将连通的簇合并为连续谱系，避免被切碎
        # ====================================================================
        paga_cfg = cluster_cfg.get("paga", {})
        if paga_cfg.get("enabled", True):
            build_paga_lineages(
                rna_adata,
                cluster_key=cluster_key,
                connectivity_threshold=paga_cfg.get("connectivity_threshold", 0.1),
                min_cells_per_lineage=paga_cfg.get("min_cells_per_lineage", 50),
            )
            use_lineages = True
        else:
            logger.info("PAGA lineage 合并已禁用 (clustering.paga.enabled=false)")
            use_lineages = False
    else:
        logger.info("聚类已禁用 (clustering.enabled=false)，使用全局模式")
        use_lineages = False

    # 更新目标基因列表（过滤掉在 QC 中丢失的基因）
    target_genes = [g for g in target_genes if g in rna_adata.var_names]
    gene_types = {g: gene_types.get(g, "TF") for g in target_genes}
    if len(target_genes) == 0:
        if config.get("perturbation", {}).get("knockout_type", "gene") == "peak":
            logger.info(
                "peak KO 模式: target_genes 无效或为空, 忽略 "
                "(扰动目标为 perturbation.peak_ko.peak_id)"
            )
        else:
            raise ValueError("所有目标基因均在 QC 后被过滤，请检查基因名称是否正确")
    logger.info(f"有效目标基因: {len(target_genes)}")

    # ========================================================================
    # Step 3-8: 按谱系 (lineage) 或簇循环执行建模 + 扰动模拟
    #   PAGA 将连通的簇合并后，每个 lineage 共享一条连续伪时间轨迹。
    #   若 PAGA 禁用则回退到原始逐簇模式。
    # ========================================================================
    logger.info("=" * 60)
    if use_lineages:
        logger.info("Step 3-8/8: 按 PAGA lineage 建模与扰动模拟")
    else:
        logger.info("Step 3-8/8: 按细胞簇建模与扰动模拟")
    logger.info("=" * 60)

    cluster_cfg = config.get("clustering", {})
    cluster_key = cluster_cfg.get("cluster_key", "leiden")
    clustering_enabled = cluster_cfg.get("enabled", True)
    min_cells = cluster_cfg.get("min_cells_per_cluster", 50)

    # --- 构建分组 → 细胞映射 ---
    if use_lineages:
        group_key = "lineage"
        try:
            group_labels = sorted(rna_adata.obs[group_key].unique())
        except (ValueError, TypeError):
            group_labels = sorted(rna_adata.obs[group_key].unique(), key=str)
        cells_by_group = {
            g: rna_adata.obs_names[rna_adata.obs[group_key] == g]
            for g in group_labels
        }
        # 输出每个 lineage 包含的原始簇
        logger.info(f"PAGA 合并: {len(group_labels)} 个 lineage")
        for lin in group_labels:
            lin_mask = rna_adata.obs[group_key] == lin
            lin_clusters = sorted(
                set(rna_adata.obs.loc[lin_mask, cluster_key]),
                key=lambda x: (str(x).isdigit(), str(x)),
            )
            n_lin_cells = lin_mask.sum()
            logger.info(
                f"  {lin}: {n_lin_cells} 细胞, "
                f"原始簇 = [{', '.join(str(c) for c in lin_clusters)}]"
            )

    elif clustering_enabled and cluster_key in rna_adata.obs.columns:
        group_key = cluster_key
        try:
            group_labels = sorted(rna_adata.obs[cluster_key].unique(), key=int)
        except ValueError:
            group_labels = sorted(rna_adata.obs[cluster_key].unique(), key=str)
        cells_by_group = {
            cl: rna_adata.obs_names[rna_adata.obs[cluster_key] == cl]
            for cl in group_labels
        }
        logger.info(f"发现 {len(group_labels)} 个簇 (key='{cluster_key}')")

    elif clustering_enabled:
        logger.warning(
            f"AnnData.obs 中未找到 '{cluster_key}' 列，回退到全局模式。"
            f"请先运行 sc.pp.neighbors + sc.tl.leiden 添加聚类标签。"
        )
        cells_by_group = {"global": rna_adata.obs_names}

    else:
        logger.info("聚类已禁用，使用全局 Palantir 伪时间模式")
        cells_by_group = {"global": rna_adata.obs_names}

    # --- 检查细胞数不足的分组 ---
    small_groups = [
        g for g, cells in cells_by_group.items()
        if len(cells) < min_cells
    ]
    if small_groups:
        n_small = sum(len(cells_by_group[g]) for g in small_groups)
        if len(small_groups) == len(cells_by_group):
            # 全部分组都不足 → 回退全局
            logger.warning(
                f"所有分组 {small_groups} 细胞数均不足 (<{min_cells})，"
                f"共 {n_small} 个细胞。回退到全局模式。"
            )
            cells_by_group = {"global": rna_adata.obs_names}
            use_lineages = False
            fallback_method = cluster_cfg.get("fallback_pseudotime_method", "palantir")
            saved_method = config["pseudotime"]["method"]
            config["pseudotime"]["method"] = fallback_method
            logger.info(
                f"伪时间方法切换: {saved_method} → {fallback_method}"
            )
        else:
            large_groups = [
                g for g in cells_by_group if g not in small_groups
            ]
            logger.info(
                f"分组 {small_groups} 细胞数不足 (<{min_cells})，"
                f"共 {n_small} 个细胞将被跳过。"
                f"保留 {len(large_groups)} 个大分组继续建模。"
            )

    # --- 逐分组循环 ---
    prepared_cohorts = []
    all_cluster_results = []
    cluster_cell_counts = {}
    n_skipped = 0
    joint_context_registry = {}
    for grp, cells in cells_by_group.items():
        values = rna_adata[cells].X
        if sparse.issparse(values):
            l0 = values.copy().astype(float)
            l0.data = np.expm1(l0.data)
            original = np.asarray(l0.mean(axis=0)).ravel()
        else:
            original = np.expm1(np.asarray(values)).mean(axis=0)
        with np.errstate(divide="ignore"):
            joint_context_registry[str(grp)] = dict(n_cells=len(cells), original_log_l0_mean=np.log(original))

    for grp, cells in cells_by_group.items():
        n_cells = len(cells)
        if n_cells < min_cells:
            logger.warning(
                f"  分组 '{grp}': {n_cells} 个细胞 < {min_cells}，跳过"
            )
            n_skipped += 1
            continue

        # peak KO 模式 + state 指定单簇: 只跑目标簇
        _ko_type_loop = config.get("perturbation", {}).get("knockout_type", "gene")
        _peak_state = config.get("perturbation", {}).get("peak_ko", {}).get("state", "all")
        if _ko_type_loop == "peak" and str(_peak_state) != "all" and str(grp) != str(_peak_state):
            logger.info(f"  peak_ko.state={_peak_state}: 跳过簇 '{grp}'")
            n_skipped += 1
            continue

        if use_lineages:
            suffix = f"lineage_{grp}"
            label = f"lineage '{grp}'"
        elif grp != "global":
            suffix = f"cluster_{grp}"
            label = f"簇 '{grp}'"
        else:
            suffix = None
            label = "全局"
        logger.info(f"--- {label}: {n_cells} 个细胞 ---")

        cohort_rna = rna_adata[cells].copy()
        root_method = config["pseudotime"].get("root_cell_method", "auto")
        if root_method == "cytotrace" or (root_method in (None, "auto") and not config["pseudotime"].get("root_cell_marker")):
            from .cytotrace import attach_scores
            provenance = attach_scores(cohort_rna, original_full_raw_rna, config.get("cytotrace"))
            root_index = int(np.argmax(cohort_rna.obs["cytotrace_score"].to_numpy()))
            provenance["root_index"] = root_index
            provenance["root_cell_id"] = str(cohort_rna.obs_names[root_index])
            output_dir = Path(config["output"]["dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            scope = suffix or "global"
            cohort_rna.obs[["cytotrace_score"]].to_csv(output_dir / f"cytotrace_scores_{scope}.csv", index_label="cell_id")
            (output_dir / f"cytotrace_provenance_{scope}.json").write_text(json.dumps(provenance, indent=2))
            logger.info(f"Official CytoTRACE {scope}: {provenance['n_cells']} cells × {provenance['n_genes']} raw genes; root #{root_index} ({provenance['root_cell_id']})")
        result = _run_modeling_for_subset(
            cohort_rna,
            atac_adata[cells],
            target_genes,
            gene_types,
            config,
            suffix=suffix,
            rna_raw=rna_raw,
            **({"defer_simulation": True} if defer_simulation else {}),
            **({"gene_identity": gene_identity} if gene_identity is not None else {}),
        )

        if "error" in result:
            logger.warning(f"  {label}: {result['error']}")
            continue

        prepared_cohorts.append((grp, cells, n_cells, result))
        del cohort_rna, result

    if defer_simulation:
        simulation_results = run_simulation_jobs(
            [item[3]["simulation_manifest"] for item in prepared_cohorts],
            workers=simulation_workers)
    else:
        simulation_results = (item[3] for item in prepared_cohorts)

    for (grp, cells, n_cells, _), result in zip(prepared_cohorts, simulation_results):
        cluster_cell_counts[str(grp)] = n_cells

        # 标注分组来源
        pert = result["perturbation_results"]
        if not pert.empty:
            pert = pert.copy()
            pert["cluster"] = str(grp)
            # 若使用 lineage，同时保留原始簇标签
            if use_lineages and cluster_key in rna_adata[cells].obs.columns:
                # 每个细胞保留其原始簇标签
                cell_to_cluster = rna_adata[cells].obs[cluster_key]
                pert["_orig_cluster"] = pert.index.map(
                    lambda b: str(cell_to_cluster.get(b, ""))
                )
            result["perturbation_results"] = pert

        pw = result.get("pathway_edges")
        if pw is not None and not pw.empty:
            pw = pw.copy()
            pw["cluster"] = str(grp)
            result["pathway_edges"] = pw

        up = result.get("upstream_tfs")
        if up is not None and not up.empty:
            up = up.copy()
            up["cluster"] = str(grp)
            result["upstream_tfs"] = up

        result["cluster"] = str(grp)
        all_cluster_results.append(result)
        if defer_simulation:
            del result, pert, pw, up

    if n_skipped > 0:
        logger.info(f"跳过了 {n_skipped} 个样本量不足的簇")

    if not all_cluster_results:
        logger.error("所有簇均无有效结果，流程终止")
        return {"error": "all_clusters_failed", "config": config}

    # ========================================================================
    # Step 9: 结果汇总与保存
    # ========================================================================
    logger.info("=" * 60)
    logger.info("结果汇总与保存")
    logger.info("=" * 60)

    from .result_store import DiskResults
    all_results = DiskResults.create(Path(config["output"]["dir"]) / "simulation_results" / "merged") if defer_simulation else {}

    # 合并各簇的扰动结果
    perturbation_dfs = [
        r["perturbation_results"] for r in all_cluster_results
        if not r["perturbation_results"].empty
    ]
    if perturbation_dfs:
        all_results['perturbation_results'] = pd.concat(perturbation_dfs, ignore_index=True)
    else:
        all_results['perturbation_results'] = pd.DataFrame()

    del perturbation_dfs

    # 计算全组织加权平均 delta_rna (按簇细胞数加权)
    total_cells = sum(cluster_cell_counts.values())
    if not all_results['perturbation_results'].empty and total_cells > 0 and len(cluster_cell_counts) > 1:
        pert_w = all_results['perturbation_results'].copy()
        pert_w["_weight"] = pert_w["cluster"].map(cluster_cell_counts)
        grouped = pert_w.groupby(["target_gene", "affected_gene"])
        merged_rows = []
        for (tg, ag), grp in grouped:
            wsum = grp["_weight"].sum()
            if wsum == 0:
                continue
            signed_delta = (grp["delta_rna"] * grp["_weight"]).sum() / total_cells
            abs_delta = (grp["delta_rna"].abs() * grp["_weight"]).sum() / total_cells
            merged_rows.append({
                "target_gene": tg, "affected_gene": ag,
                "delta_rna": abs_delta, "delta_rna_signed": signed_delta,
                "mediated_by_atac": grp["mediated_by_atac"].iloc[0],
                "mechanism": grp["mechanism"].iloc[0],
                "propagation_depth": grp["propagation_depth"].iloc[0],
                "converged": grp["converged"].iloc[0],
                "is_fallback": grp["is_fallback"].iloc[0],
                "perturbation_mode": grp["perturbation_mode"].iloc[0],
                "n_clusters_observed": len(grp),
            })
        all_results['perturbation_merged'] = pd.DataFrame(merged_rows)
    else:
        all_results['perturbation_merged'] = all_results['perturbation_results'].copy() if not all_results['perturbation_results'].empty else pd.DataFrame()

    if "pert_w" in locals():
        del pert_w, grouped, grp

    # 合并各簇的路径级边记录
    pathway_dfs = [
        r["pathway_edges"] for r in all_cluster_results
        if r.get("pathway_edges") is not None and not r["pathway_edges"].empty
    ]
    if pathway_dfs:
        all_results['pathway_edges'] = pd.concat(pathway_dfs, ignore_index=True)
    else:
        all_results['pathway_edges'] = pd.DataFrame(columns=[
            "target_gene", "tf_name", "peak_id", "affected_gene",
            "delta_r_raw", "delta_r_total", "mechanism", "r2_score", "tf_lag", "bin_min", "bin_max", "cluster"])

    del pathway_dfs

    # 合并各簇的聚合 TF→peak 边 (含窗口验证翻转信息)
    agg_edge_dfs = [
        r["aggregated_tf_peak_edges"] for r in all_cluster_results
        if r.get("aggregated_tf_peak_edges") is not None
        and not r["aggregated_tf_peak_edges"].empty
    ]
    if agg_edge_dfs:
        all_results['aggregated_tf_peak_edges'] = pd.concat(agg_edge_dfs, ignore_index=True)
    else:
        all_results['aggregated_tf_peak_edges'] = pd.DataFrame(columns=[
            "tf_gene", "peak_id", "weight", "lag", "flipped", "original_lag"])

    del agg_edge_dfs

    # Merge one report family at a time; fitted weights, curves and edges
    # must not all be duplicated in RAM during final aggregation.
    for key in ("causal_edges", "tf_peak_weights", "transfer_functions"):
        frames = []
        for i, context in enumerate(all_cluster_results):
            frame = context[key].copy()
            frame["cluster"] = str(context.get("cluster", i))
            if key == "transfer_functions":
                frame.attrs = {}
            frames.append(frame)
        all_results[key] = pd.concat(frames, ignore_index=True)
        del frames, frame

    # 合并各簇的伪时间（每簇独立推断，标注簇来源）
    pseudotime_dfs = []
    for i, r in enumerate(all_cluster_results):
        cl_label = str(r.get("cluster", i))
        pt_df = r.get("pseudotime")
        if pt_df is not None and not pt_df.empty:
            pt_df = pt_df.copy()
            pt_df["cluster"] = cl_label
            pseudotime_dfs.append(pt_df)
    all_results['pseudotime'] = pd.concat(pseudotime_dfs) if pseudotime_dfs else pd.DataFrame()

    del pseudotime_dfs, r

    # 合并各簇的 ATAC 变化 (用于 IGV 可视化)
    atac_changes_dfs = [
        r["atac_changes"] for r in all_cluster_results
        if r.get("atac_changes") is not None and not r["atac_changes"].empty
    ]
    all_results['atac_changes'] = pd.concat(atac_changes_dfs, ignore_index=True) if atac_changes_dfs else pd.DataFrame()

    # 跨簇加权平均 ATAC 变化 (对齐 perturbation_merged 逻辑: 按簇细胞数加权)
    # 每个 peak 输出 signed + abs 双列 delta, 避免跨簇方向抵消
    # Unlike merged cell-L0 aggregation, these tables only contain successful
    # modeled contexts, so skipped/failed contexts must not enter this
    # denominator.  Keep this count local to preserve the prior ATAC behavior.
    modeled_total_cells = sum(cluster_cell_counts.values())
    all_results['atac_changes_merged'] = _merge_atac_changes(
        all_cluster_results, cluster_cell_counts, config)

    del atac_changes_dfs

    # 合并各簇的上游 TF 归因 (peak KO 模式; gene 模式下为空)
    upstream_dfs = [
        r["upstream_tfs"] for r in all_cluster_results
        if r.get("upstream_tfs") is not None and not r["upstream_tfs"].empty
    ]
    all_results['upstream_tfs'] = pd.concat(upstream_dfs, ignore_index=True) if upstream_dfs else pd.DataFrame()

    del upstream_dfs

    # 上游 TF 跨簇加权合并视图 (按簇细胞数加权, 对齐 perturbation_merged 思路)
    all_results['upstream_tfs_merged'] = pd.DataFrame()
    if not all_results['upstream_tfs'].empty and modeled_total_cells > 0 and len(cluster_cell_counts) > 1:
        _up = all_results['upstream_tfs'].copy()
        _up["_weight"] = _up["cluster"].map(cluster_cell_counts).fillna(0)
        _up_rows = []
        for (pk, tf), grp in _up.groupby(["peak_id", "tf_gene"]):
            wsum = grp["_weight"].sum()
            if wsum == 0:
                continue
            _up_rows.append({
                "peak_id": pk,
                "tf_gene": tf,
                "weight_signed_merged": float((grp["weight"] * grp["_weight"]).sum() / modeled_total_cells),
                "weight_abs_merged": float((grp["weight"].abs() * grp["_weight"]).sum() / modeled_total_cells),
                "n_clusters_observed": len(grp),
            })
        if _up_rows:
            all_results['upstream_tfs_merged'] = pd.DataFrame(_up_rows).sort_values(
                "weight_abs_merged", ascending=False
            )

    if "_up" in locals():
        del _up, grp

    # peak 模式: 汇总各簇扰动 peak 的基线/扰后/Δ (供四段式 report 生成)
    all_results['peak_summary'] = pd.DataFrame()
    if any(r.get("prediction_mode", "").startswith("peak_") for r in all_cluster_results):
        _ps_rows = []
        for r in all_cluster_results:
            if not r.get("prediction_mode", "").startswith("peak_"):
                continue
            _ps_rows.append({
                "peak_id": r.get("peak_id", ""),
                "cluster": str(r.get("cluster", "")),
                "baseline_accessibility": r.get("baseline_accessibility"),
                "perturbed_accessibility": r.get("perturbed_accessibility"),
                "delta_accessibility": r.get("delta_accessibility"),
                "n_affected_genes": len(r.get("perturbation_results", pd.DataFrame())),
                "n_upstream_tfs": len(r.get("upstream_tfs", pd.DataFrame())),
                "n_affected_edges": r.get("n_affected_edges", 0),
            })
        all_results['peak_summary'] = pd.DataFrame(_ps_rows)

    projection_tables = [r.get("cell_projection") for r in all_cluster_results
                         if r.get("cell_projection") is not None and not r.get("cell_projection").empty]
    all_results['cell_projection'] = pd.concat(projection_tables, ignore_index=True) if projection_tables else pd.DataFrame()
    projection_tables.clear()
    projection_histories = [r.get("projection_history") for r in all_cluster_results
                            if r.get("projection_history") is not None and not r.get("projection_history").empty]
    all_results['projection_history'] = pd.concat(projection_histories, ignore_index=True) if projection_histories else pd.DataFrame()
    projection_histories.clear()
    projection_metadata_frames = [r.get("projection_bin_metadata") for r in all_cluster_results
                                  if r.get("projection_bin_metadata") is not None and not r.get("projection_bin_metadata").empty]
    all_results['projection_bin_metadata'] = pd.concat(projection_metadata_frames, ignore_index=True) if projection_metadata_frames else pd.DataFrame()
    projection_metadata_frames.clear()
    projection_event_frames = [r.get("projection_events") for r in all_cluster_results
                               if r.get("projection_events") is not None and not r.get("projection_events").empty]
    all_results['projection_events'] = pd.concat(projection_event_frames, ignore_index=True) if projection_event_frames else pd.DataFrame()
    all_results['projection_summary'] = pd.DataFrame()
    all_results['projection_branch_probabilities'] = pd.DataFrame()
    if not all_results['cell_projection'].empty:
        all_results['projection_summary'] = (
            all_results['cell_projection'].groupby(["source_context", "target_gene", "projection_status"], dropna=False)
            .size().reset_index(name="n_cells")
        )
        probability_cols = [c for c in all_results['cell_projection'].columns if c.startswith("branch_probability_")]
        if probability_cols:
            all_results['projection_branch_probabilities'] = all_results['cell_projection'][
                ["cell_id", "source_context", "target_gene"] + probability_cols
            ].melt(id_vars=["cell_id", "source_context", "target_gene"],
                   var_name="branch", value_name="branch_probability")
            probabilities = all_results['projection_branch_probabilities']
            probabilities["branch"] = probabilities["branch"].str.replace("branch_probability_", "", regex=False)
            all_results['projection_branch_probabilities'] = probabilities
            del probabilities

    del projection_tables, projection_histories, projection_metadata_frames, projection_event_frames

    all_results.update(granger_results=all_cluster_results[0]["granger_results"],
                       cluster_results=all_cluster_results)
    _collect_saturating_tables(all_cluster_results, destination=all_results)
    merged, diagnostics = _saturating_merge_contexts(all_cluster_results, joint_context_registry,
        list(rna_adata.var_names), config.get("perturbation", {}).get("min_delta_threshold", 0),
        policy=config.get("perturbation", {}).get("aggregation_policy", "strict_complete"),
        endpoints=all_results["saturating_endpoints"])
    all_results["perturbation_merged"] = merged
    all_results["saturating_aggregation_diagnostics"] = diagnostics
    if not config.get("perturbation", {}).get("cell_projection", {}).get("enabled", False):
        for key in ("cell_projection", "projection_history", "projection_bin_metadata",
                    "projection_events", "projection_summary", "projection_branch_probabilities"):
            del all_results[key]
    del merged, diagnostics
    if gene_identity is not None:
        all_results.update(identity_tables)
        metadata = all_results["gene_identity_metadata"]
        metadata.update(effective_output_dir=config["output"]["dir"],
            effective_checkpoint_dir=str(_checkpoint_dir(config)), cache_dir=config["output"].get("cache_dir"))
        all_results["gene_identity_metadata"] = metadata
    save_results(all_results, config, **({"gene_identity": gene_identity} if gene_identity is not None else {}))

    # 汇总统计
    if not all_results['perturbation_results'].empty and "target_gene" in all_results['perturbation_results'].columns:
        n_clusters = all_results['perturbation_results']["cluster"].nunique() if "cluster" in all_results['perturbation_results'].columns else 1
        n_unique_targets = all_results['perturbation_results']["target_gene"].nunique()
        n_unique_affected = all_results['perturbation_results']["affected_gene"].nunique()
        atac_mediated_pct = (
            all_results['perturbation_results']["mediated_by_atac"].mean() * 100
            if "mediated_by_atac" in all_results['perturbation_results'].columns
            else 0
        )
        logger.info("=" * 60)
        logger.info("CausalBridge 分析完成！")
        logger.info(f"  细胞簇数: {n_clusters}")
        logger.info(f"  目标基因数: {n_unique_targets}")
        logger.info(f"  受影响基因数: {n_unique_affected}")
        logger.info(f"  预测的 target→affected 关系: {len(all_results['perturbation_results'])}")
        logger.info(f"  通过 ATAC 介导的比例: {atac_mediated_pct:.1f}%")
        logger.info(f"  结果目录: {config['output']['dir']}")
        logger.info("=" * 60)
    else:
        logger.warning("=" * 60)
        logger.warning("CausalBridge 分析完成，但扰动模拟未产生结果")
        logger.warning("  可能原因: 目标基因不在因果网络中，或 TF→peak 边缺失")
        logger.warning(f"  结果目录: {config['output']['dir']}")
        logger.warning("=" * 60)

    return all_results
