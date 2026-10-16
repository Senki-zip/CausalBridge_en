"""Current response fitting, shared motif resources, and pseudotime binning."""

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import pearsonr
import logging
import pickle
import warnings
import hashlib
import json
import inspect
import os
import tempfile
import fcntl
import time
import multiprocessing
import threading
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from types import MappingProxyType
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from sysconfig import get_paths

logger = logging.getLogger(__name__)
from .terminal_progress import TerminalProgress


def _get_cache_dir(config: Optional[dict] = None) -> Path:
    """获取缓存目录，优先从 config 读取，否则使用默认路径。"""
    if config:
        cache_dir = config.get("output", {}).get("cache_dir")
        if cache_dir:
            return Path(cache_dir)
    return Path.home() / ".CausalBridge"


JASPAR_RESOURCE_URL = "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt"
_UNSET_TF_DATABASE = object()
RAW_MOTIF_SCHEMA = "peak-motif-raw-v1"
RAW_MOTIF_SCANNER_VERSION = "gumbel-window-scanner-1"
_RESOURCE_DIGEST_MEMO = {}
_MOTIF_SPAWN_LOCK = threading.Lock()
_MOTIF_WORKER_DB = None
_MOTIF_WORKER_THRESHOLD = None
_MOTIF_THREAD_LIMIT = None


def _validate_motif_scan_workers(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("rna_to_atac.motif_scan_workers 必须是正整数（进程数）")
    return value


def _init_motif_scan_worker(motifs, threshold):
    global _MOTIF_WORKER_DB, _MOTIF_WORKER_THRESHOLD, _MOTIF_THREAD_LIMIT
    prepared = {}
    for motif_id, pwm in motifs:
        pwm.flags.writeable = False
        prepared[motif_id] = MappingProxyType({"pwm": pwm})
    _MOTIF_WORKER_DB = MappingProxyType(prepared)
    _MOTIF_WORKER_THRESHOLD = threshold
    try:
        from threadpoolctl import threadpool_limits
        _MOTIF_THREAD_LIMIT = threadpool_limits(limits=1)
    except ImportError:
        pass  # 子进程导入前已有 BLAS/OMP 环境限制。


def _scan_motif_batch(batch):
    return [(key, _scan_one_sequence(sequence, _MOTIF_WORKER_DB,
                                     _MOTIF_WORKER_THRESHOLD)) for key, sequence in batch]


def _scan_sequence_records(tasks, motif_db, threshold, workers=1):
    """按输入顺序扫描，最多保留两倍 worker 数量的在途批次。"""
    requested = _validate_motif_scan_workers(workers)
    if not tasks:
        logger.info("Primary motif scan: requested_workers=%d effective_workers=0 peaks=0", requested)
        return {}
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    effective = min(requested, available, len(tasks))
    logger.info("Primary motif scan: requested_workers=%d effective_workers=%d peaks=%d",
                requested, effective, len(tasks))
    records = {}
    def collect(batch):
        records.update(batch)
        progress.update(len(records))

    with TerminalProgress(len(tasks), "Step 6 扫描", logger=logger) as progress:
        if effective == 1:
            for key, sequence in tasks:
                collect([(key, _scan_one_sequence(sequence, motif_db, threshold))])
        else:
            batch_size = min(16, max(1, len(tasks) // (effective * 4)))
            batches = iter(tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size))
            motifs = tuple((key, np.asarray(info["pwm"])) for key, info in motif_db.items())
            variables = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                         "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            pending = deque()
            # spawn 在 submit 时启动所有子进程；恢复环境后再等待和补充任务。
            # 锁防止本模块的并发建池互相覆盖环境，父进程原有环境始终恢复。
            with ProcessPoolExecutor(max_workers=effective,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init_motif_scan_worker, initargs=(motifs, threshold)) as pool:
                with _MOTIF_SPAWN_LOCK:
                    previous = {name: os.environ.get(name) for name in variables}
                    try:
                        for name in variables:
                            os.environ[name] = "1"
                        for batch in batches:
                            pending.append(pool.submit(_scan_motif_batch, batch))
                            if len(pending) >= effective * 2:
                                break
                    finally:
                        for name, value in previous.items():
                            if value is None:
                                os.environ.pop(name, None)
                            else:
                                os.environ[name] = value
                while pending:
                    collect(pending.popleft().result())
                    batch = next(batches, None)
                    if batch is not None:
                        pending.append(pool.submit(_scan_motif_batch, batch))
    logger.info("Primary motif scan complete: peaks=%d/%d", len(records), len(tasks))
    return records


def _resource_file_identity(path, *, chunk_bytes=1024 * 1024):
    """Single streaming SHA256 implementation shared by runner and raw cache."""
    if path is None:
        return {"path": None, "availability": "unconfigured", "sha256": None}
    path = Path(path).resolve()
    try:
        stat = path.stat()
        if not path.is_file():
            return {"path": str(path), "availability": "not_a_file", "sha256": None}
        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = _RESOURCE_DIGEST_MEMO.get(str(path))
        if cached is not None and cached[0] == stamp:
            fingerprint = cached[1]
        else:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(chunk_bytes)
                    if not chunk:
                        break
                    digest.update(chunk)
            after = path.stat()
            if stamp != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise RuntimeError(f"resource changed while hashing: {path}; rerun with stable resources")
            fingerprint = digest.hexdigest()
            _RESOURCE_DIGEST_MEMO[str(path)] = (stamp, fingerprint)
        return {"path": str(path), "availability": "available", "size": stat.st_size, "sha256": fingerprint}
    except FileNotFoundError:
        _RESOURCE_DIGEST_MEMO.pop(str(path), None)
        return {"path": str(path), "availability": "missing", "sha256": None}
    except PermissionError:
        return {"path": str(path), "availability": "unreadable", "sha256": None}


def _raw_scan_cache_descriptor(config, motif_db, cfg=None):
    """Primitive runtime descriptor built against the original base cache."""
    cfg = config.get("rna_to_atac", {}) if cfg is None else cfg
    digest = hashlib.sha256()
    for motif_id, info in motif_db.items():
        pwm = np.asarray(info["pwm"])
        if (pwm.ndim != 2 or pwm.shape[0] != 4 or pwm.shape[1] < 1
                or not np.isfinite(pwm).all() or not isinstance(info["tf_name"], str)):
            raise ValueError("motif resource requires a finite 4-by-length PWM and TF name")
        mapping = {"motif_id": motif_id, "tf_name": info["tf_name"],
                   "factors": [factor["factor"] for factor in info.get("all_factors", [])],
                   "shape": pwm.shape, "dtype": str(pwm.dtype)}
        digest.update(json.dumps(mapping, sort_keys=True).encode())
        digest.update(np.ascontiguousarray(pwm).tobytes())
    genome = _resource_file_identity(config.get("input", {}).get("genome_fasta"))
    identity = {"schema": RAW_MOTIF_SCHEMA, "scanner_version": RAW_MOTIF_SCANNER_VERSION,
        "scanner_source": hashlib.sha256(inspect.getsource(_scan_one_sequence).encode()).hexdigest(),
        "genome": {k: v for k, v in genome.items() if k != "path"},
        "motifs": digest.hexdigest(), "threshold": float(cfg.get("motif_pval_threshold", 1e-4)),
        "parameters": {"min_motif_length": 5, "min_score_std": 1e-10, "null": "per-sequence-gumbel",
                       "numpy_version": np.__version__}}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return {"base_cache_dir": str(_get_cache_dir(config)), "scan_key": key, "identity": identity}


def _scan_one_sequence(sequence, motif_db, threshold):
    """Pure existing PWM/Gumbel numerics; retain all passing motif records."""
    from numpy.lib.stride_tricks import sliding_window_view
    nucleotide_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq_ints = np.array([nucleotide_to_idx.get(nt, -1) for nt in sequence.upper()], dtype=np.int8)
    hits = []
    for motif_id, info in motif_db.items():
        pwm = np.asarray(info["pwm"])
        mlen = pwm.shape[1]
        if mlen < 5 or mlen > len(seq_ints):
            continue
        windows = sliding_window_view(seq_ints, mlen)
        valid_mask = (windows >= 0).all(axis=1)
        if not valid_mask.any():
            continue
        valid_windows = windows[valid_mask]
        scores = np.sum(pwm[valid_windows[:, :], np.arange(mlen)], axis=1)
        max_score = float(scores.max())
        null_mean = float(np.mean(scores))
        null_std = float(np.std(scores))
        if null_std < 1e-10:
            continue
        gumbel_beta = null_std * np.sqrt(6.0) / np.pi
        gumbel_mu = null_mean - np.euler_gamma * gumbel_beta
        p_val = 1.0 - np.exp(-np.exp(-(max_score - gumbel_mu) / gumbel_beta))
        p_val = float(np.clip(p_val, 0.0, 1.0))
        if p_val < threshold:
            hits.append({"motif_id": motif_id, "p_value": p_val, "max_score": max_score})
    return hits


def _read_raw_motif_cache(path, descriptor, motif_db):
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if (not isinstance(payload, dict) or payload.get("schema") != RAW_MOTIF_SCHEMA
                or payload.get("identity") != descriptor["identity"]
                or payload.get("scan_key") != descriptor["scan_key"]
                or not isinstance(payload.get("records"), dict)):
            raise ValueError("schema/identity mismatch")
        order = {motif: i for i, motif in enumerate(motif_db)}
        for key, hits in payload["records"].items():
            if (not isinstance(key, tuple) or len(key) != 4
                    or not all(isinstance(x, str) for x in key[:2])
                    or not all(type(x) is int for x in key[2:])
                    or key[2] < 0 or key[3] <= key[2] or not isinstance(hits, list)):
                raise ValueError("invalid peak record")
            previous = -1
            for hit in hits:
                if (not isinstance(hit, dict) or set(hit) != {"motif_id", "p_value", "max_score"}
                        or hit["motif_id"] not in order or order[hit["motif_id"]] <= previous
                        or not isinstance(hit["p_value"], (float, int))
                        or not 0 <= hit["p_value"] < descriptor["identity"]["threshold"]
                        or not isinstance(hit["max_score"], (float, int))
                        or not np.isfinite(hit["max_score"])):
                    raise ValueError("invalid motif hit")
                previous = order[hit["motif_id"]]
        return payload["records"]
    except Exception as exc:
        logger.warning("Ignoring invalid shared raw motif cache %s: %s; rescanning requested peaks", path, exc)
        return {}


def _merge_raw_motif_cache(path, descriptor, motif_db, additions):
    """Scan outside the short lock; reload/merge latest union and replace atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    with path.with_suffix(".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            records = _read_raw_motif_cache(path, descriptor, motif_db)
            records.update(additions)
            payload = {"schema": RAW_MOTIF_SCHEMA, "scan_key": descriptor["scan_key"],
                       "identity": descriptor["identity"], "records": records}
            with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=path.name + ".", delete=False) as handle:
                temporary = Path(handle.name)
                pickle.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            fcntl.flock(lock, fcntl.LOCK_UN)


def _materialize_raw_motif_hits(hits, motif_db, max_tfs):
    matched = []
    for hit in hits:
        info = motif_db[hit["motif_id"]]
        names = [fi["factor"].strip().upper() for fi in info.get("all_factors", [])]
        names.extend(tf.strip().upper() for tf in info["tf_name"].split("::"))
        matched.extend((name, hit["p_value"]) for name in names if name)
    matched.sort(key=lambda item: item[1])
    result = []
    for name, _ in matched:
        if name not in result:
            result.append(name)
            if len(result) >= max_tfs:
                break
    return result


def _family_cache_descriptor(config, motif_db, target_family_map, family_groups, threshold,
                             raw_scan_cache=None):
    digest = hashlib.sha256()
    for motif_id, info in motif_db.items():
        if info.get("tf_name", "").upper() not in {
            member.upper() for family in target_family_map.values()
            for member in family_groups.get(family, set())
        }:
            continue
        pwm = np.asarray(info.get("pwm"))
        digest.update(json.dumps({"id": motif_id, "tf": info.get("tf_name"),
                                  "factors": info.get("all_factors", []),
                                  "shape": pwm.shape, "dtype": str(pwm.dtype)},
                                 sort_keys=True, default=str).encode())
        digest.update(np.ascontiguousarray(pwm).tobytes())
    identity = {"schema": "peak-family-motif-v2", "scanner": "family-gumbel-1",
                "source": hashlib.sha256(inspect.getsource(_supplement_family_proxies).encode()).hexdigest(),
                "genome": _resource_file_identity(config.get("input", {}).get("genome_fasta")),
                "targets": sorted(target_family_map.items()),
                "families": {k: sorted(v) for k, v in sorted(family_groups.items())
                             if k in set(target_family_map.values())},
                "threshold": float(threshold), "motifs": digest.hexdigest()}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    base = Path(raw_scan_cache["base_cache_dir"]) if raw_scan_cache else _get_cache_dir(config)
    return {"base_cache_dir": str(base), "scan_key": key, "identity": identity}


def _family_cache_read(path, descriptor):
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if (payload.get("schema") != "peak-family-motif-v2"
                or payload.get("scan_key") != descriptor["scan_key"]
                or payload.get("identity") != descriptor["identity"]
                or not isinstance(payload.get("records"), dict)):
            return {}
        return payload["records"]
    except Exception as exc:
        logger.warning("Ignoring invalid family motif cache %s: %s", path, exc)
        return {}


def _family_cache_merge(path, descriptor, additions):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            records = _family_cache_read(path, descriptor)
            records.update(additions)
            payload = {"schema": "peak-family-motif-v2", "scan_key": descriptor["scan_key"],
                       "identity": descriptor["identity"], "records": records}
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                                 prefix=path.name + ".", delete=False) as handle:
                    temporary = Path(handle.name)
                    pickle.dump(payload, handle)
                    handle.flush(); os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _read_legacy_peak_tf_candidates(path, threshold, requested, max_tfs):
    """Read the historical threshold-only candidate cache without promoting it.

    The caller explicitly accepts the historical resource assumptions. Lists
    remain ordered and are only truncated for a smaller current top-N; empty
    lists are not treated as verified negative coverage.
    """
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict) or payload.get("threshold") != threshold:
            logger.warning("Legacy peak-TF cache threshold mismatch at %s; scanning uncovered peaks", path)
            return {}
        data = payload.get("data")
        if not isinstance(data, dict):
            logger.warning("Malformed legacy peak-TF data at %s; scanning uncovered peaks", path)
            return {}
        result = {}
        for peak in requested:
            value = data.get(peak)
            if not isinstance(value, list) or not value:
                continue
            if not all(isinstance(tf, str) and tf.strip() for tf in value):
                continue
            result[peak] = list(value[:max_tfs])
        if result:
            logger.info("Reusing legacy peak-TF candidates from %s: threshold matched; provenance not recorded; assumed unchanged resources per user", path)
        return result
    except Exception as exc:
        logger.warning("Malformed legacy peak-TF cache %s; scanning uncovered peaks: %s", path, exc)
        return {}


def _local_motif_resource_paths(config):
    """Concrete loader resource paths; excludes model-derived candidate caches."""
    installation = Path(get_paths()["purelib"]) / "data" / "motif_databases"
    paths = {f"installed:{name}": installation / name for name in (
        "gimme.vertebrate.v5.0.pfm", "gimme.vertebrate.v5.0.motif2factors.txt",
        "CIS-BP.pfm", "CIS-BP.motif2factors.txt")}
    for label, directory in (("configured", _get_cache_dir(config)),
                             ("home", Path.home() / ".CausalBridge")):
        for name in ("gimmemotifs_cache.pkl", "jaspar_motifs.pkl", "tf_database.txt"):
            paths[f"{label}:{name}"] = directory / name
    genome = config.get("input", {}).get("genome_fasta")
    paths["genome_fasta"] = Path(genome) if genome else None
    return paths


def _prepare_adaptive_motif_resources(config):
    """Resolve once before recovery, freeze inputs outside config/checkpoints.

    Enabled mode prefers materialized local resources. A required online fetch
    is performed before provenance capture; only its materialized motif content
    and explicit endpoint/version parameters are claimed, not remote freshness.
    """
    directory = _get_cache_dir(config)
    motifs = _load_motif_database(config.get("rna_to_atac", {}).get("motif_db", "jaspar2024"),
                                  directory, True)
    # Motifs do not guarantee usable FASTA/sequence extraction. Freeze fallback
    # TFs before provenance capture as well, never from a candidate namespace.
    tf_database = frozenset(_load_tf_database(directory))
    return {"motif_db": motifs, "tf_database": tf_database, "resource_cache_dir": str(directory)}


# ============================================================================
# 步骤1: ATAC→RNA 传递函数拟合
# ============================================================================

def fit_atac_to_rna(
    rna_adata, atac_adata, pseudotime_df, causal_edges: pd.DataFrame,
    config: dict, granger_results: pd.DataFrame = None, *, bin_plan=None,
) -> Tuple[pd.DataFrame, dict]:
    """Fit current independent peak→gene responses; return table and registry."""
    from ._wt_aggregation import validate_model_mode
    from .saturating_transfer import fit_peak_gene, validate_saturating_config
    from .binning_adapter import require_bin_plan, execution_pseudotime
    validate_model_mode(config)
    validate_saturating_config(config)
    require_bin_plan(config, bin_plan)
    if bin_plan is not None:
        pseudotime_df = execution_pseudotime(bin_plan, pseudotime_df, rna_adata.obs_names)
    elif "bin" not in pseudotime_df:
        pcfg = config["pseudotime"]
        labels, _ = _adaptive_rebin(pseudotime_df, rna_adata.n_obs,
            max_bins=pcfg["n_bins"], min_cells_per_bin=pcfg.get("min_cells_per_bin", 5),
            target_bins=pcfg["n_bins"] if pcfg.get("smooth_kernel") == "gaussian" else None,
            overlap_factor=pcfg.get("smooth_overlap_factor"))
        pseudotime_df["bin"] = labels
    transfers, metrics, registry = fit_peak_gene(
        rna_adata, atac_adata, pseudotime_df, causal_edges, config, bin_plan=bin_plan)
    return transfers, registry


def _gene_base_map(rna_adata, gene_identity=None, config=None):
    """Validate the identity boundary without changing expression columns."""
    enabled = (gene_identity is not None or "gene_identity" in rna_adata.uns
               or (config or {}).get("input", {}).get("gene_identity_mode") == "gtf_gene_id")
    if not enabled:
        return None
    if not rna_adata.var_names.is_unique:
        raise ValueError("gene identity requires unique RNA gene IDs")
    for column in ("gene_name", "gene_display_name"):
        if column not in rna_adata.var:
            raise ValueError(f"gene identity requires RNA var[{column!r}]")
        if any(not isinstance(v, str) or not v.strip() for v in rna_adata.var[column]):
            raise ValueError(f"gene identity requires nonempty {column}")
    bases = dict(zip(rna_adata.var_names, rna_adata.var["gene_name"]))
    if gene_identity is not None:
        fingerprint = rna_adata.uns.get("gene_identity", {}).get("fingerprint")
        if fingerprint is not None and fingerprint != gene_identity.fingerprint:
            raise ValueError("RNA gene identity fingerprint differs from runtime index")
        for gene_id, base in bases.items():
            record = gene_identity.by_id.get(gene_id)
            if (record is None or record.gene_name != base
                    or gene_identity.display_labels[gene_id] != rna_adata.var.loc[gene_id, "gene_display_name"]):
                raise ValueError(f"RNA gene identity mapping differs for {gene_id!r}")
    return bases


def fit_rna_to_atac(
    rna_adata,
    atac_adata,
    pseudotime_df,
    causal_edges: pd.DataFrame,
    config: dict,
    target_gene_types: Optional[dict] = None,
    rna_raw = None,
    *,
    bin_plan=None,
    gene_identity=None,
    motif_resources=None,
    raw_scan_cache=None,
    tf_model_registry=None,
) -> pd.DataFrame:
    """
    对每个因果 peak，学习其可及性如何受 TF 表达调控。

    两阶段策略：
    阶段1: 对 peak 的 DNA 序列做 motif 扫描，确定候选 TF 集合。
    阶段2: 逐 TF、逐 lag 单特征 Pearson 检验 —— TF(t-lag) → A_peak(t)，第一个显著 lag 胜出。

    阶段1.5 (当 target_gene_types 提供时):
      对标记为 TF 的靶基因检查 motif 匹配是否充足。
      若不足，通过同家族最优 motif 代理补充匹配。

    Parameters
    ----------
    target_gene_types : dict, optional
        {gene_name: "TF" | "target"}，来自 IO 模块的基因类型标注。

    Returns
    -------
    tf_weights : DataFrame
        列: peak_id, tf_gene, weight, p_value, abs_contribution

    """
    from ._wt_aggregation import validate_model_mode
    validate_model_mode(config)
    from .atac_normalization import validate_atac_depth_config, expected_registry_schema
    continuous = validate_atac_depth_config(config)["enabled"]
    from .saturating_transfer import validate_saturating_config, fit_tf_peak, validate_registry
    validate_saturating_config(config)
    if not isinstance(tf_model_registry, dict):
        raise ValueError("saturating_hybrid requires a caller-owned tf_model_registry dict; rerun Step 6")
    if any(str(kind).upper() == "TARGET" for kind in (target_gene_types or {}).values()):
        raise ValueError("saturating_hybrid does not support TARGET proxy interventions")
    cfg = config["rna_to_atac"]
    pcfg = config["pseudotime"]
    base_map = _gene_base_map(rna_adata, gene_identity, config)
    full_base_map = (_gene_base_map(rna_raw, gene_identity, config)
                     if base_map is not None and rna_raw is not None else None)
    symbol_edges = causal_edges
    if base_map is not None:
        target_bases = dict(full_base_map or {}, **base_map)
        if gene_identity is not None:
            target_bases.update({g: r.gene_name for g, r in gene_identity.by_id.items()})
        symbol_edges = causal_edges.copy()
        symbol_edges["gene"] = [target_bases[g] for g in causal_edges["gene"]]
        if target_gene_types:
            symbol_types = {}
            for g, kind in target_gene_types.items():
                base = target_bases[g]
                if kind == "TF" or base not in symbol_types:
                    symbol_types[base] = kind
            target_gene_types = symbol_types
    from .binning_adapter import require_bin_plan, validate_bin_plan, aggregate_with_plan, execution_pseudotime
    require_bin_plan(config, bin_plan)
    if bin_plan is not None:
        validate_bin_plan(bin_plan, pseudotime_df, rna_adata.obs_names)
        validate_bin_plan(bin_plan, pseudotime_df, atac_adata.obs_names)
        pseudotime_df = execution_pseudotime(bin_plan, pseudotime_df, rna_adata.obs_names)

    # 按簇规模自适应分箱
    _smooth_kernel = pcfg.get("smooth_kernel", "hard")
    _overlap = pcfg.get("smooth_overlap_factor") if _smooth_kernel == "gaussian" else None
    if bin_plan is None and "bin" in pseudotime_df:
        n_bins = int(pseudotime_df["bin"].max()) + 1
    elif bin_plan is None:
        bin_indices, n_bins = _adaptive_rebin(
            pseudotime_df, rna_adata.n_obs,
            max_bins=pcfg["n_bins"],
            min_cells_per_bin=pcfg.get("min_cells_per_bin", 5),
            target_bins=pcfg["n_bins"] if _smooth_kernel == "gaussian" else None,
            overlap_factor=_overlap,
        )
        pseudotime_df["bin"] = bin_indices
    else:
        n_bins = bin_plan.n_bins
    logger.info(f"RNA→ATAC 自适应分箱: {rna_adata.n_obs} 细胞 → {n_bins} bins "
                f"(smooth={_smooth_kernel})")

    # 伪时间分箱聚合
    from .saturating_transfer import _snapshot
    snapshot = _snapshot(rna_adata, atac_adata, pseudotime_df, config, bin_plan)
    rna_binned, atac_binned = snapshot["R"], snapshot["A"]

    # --- 阶段1: motif 扫描 ---
    logger.info("RNA→ATAC 阶段1: motif 扫描 peak 序列")
    cache_dir = Path(motif_resources["resource_cache_dir"]) if motif_resources is not None else _get_cache_dir(config)
    motif_db = motif_resources["motif_db"] if motif_resources is not None else _load_motif_database(cfg.get("motif_db", "jaspar2024"), cache_dir)

    if motif_db is not None:
        logger.info(f"  已加载 {len(motif_db)} 个 motif")
        if motif_resources is None:
            peak_to_tfs = _scan_peaks_with_jaspar(
                atac_adata, rna_adata, symbol_edges, motif_db, cfg, config,
                _UNSET_TF_DATABASE, raw_scan_cache)
        else:
            peak_to_tfs = _scan_peaks_with_jaspar(
                atac_adata, rna_adata, symbol_edges, motif_db, cfg, config,
                motif_resources["tf_database"], raw_scan_cache)
    else:
        logger.warning(
            "  motif 数据库不可用，回退到基于 TF 数据库的匹配策略。"
        )
        peak_to_tfs = _assign_tfs_fallback(
            atac_adata, rna_adata, symbol_edges, cache_dir,
            max_tfs_per_peak=cfg.get("max_tfs_per_peak", 5),
            **({"tf_database": motif_resources["tf_database"]} if motif_resources is not None else {}),
        )

    # Candidate materialization must not mutate shared/cache-owned symbol lists.
    peak_to_tfs = {peak: list(tfs) for peak, tfs in peak_to_tfs.items()}
    # --- 阶段1.5: 家族代理 motif 补充 ---
    if target_gene_types and motif_db is not None:
        _supplement_family_proxies(
            peak_to_tfs, motif_db, target_gene_types, cfg, config,
            atac_adata=atac_adata, raw_scan_cache=raw_scan_cache
        )

    # --- 阶段1.6: TF 候选验证 ---
    #   排除不在 RNA 基因集中的 TF（HVG 找不到时回退全量基因集）
    if cfg.get("tf_expression_filter", True):
        rna_var_names = list(base_map.values()) if base_map is not None else list(rna_adata.var_names)
        rna_var_names_full = (list(full_base_map.values()) if full_base_map is not None
                              else list(rna_raw.var_names) if rna_raw is not None else None)
        peak_to_tfs = _filter_tfs_by_expression(
            peak_to_tfs, rna_binned, rna_var_names,
            rna_var_names_full=rna_var_names_full,
        )

    mapping_metadata = {}
    mapping_diagnostics = []
    if base_map is not None:
        from .gene_identity import map_tf_symbols
        by_symbol = {}
        for gene_id, base in base_map.items():
            by_symbol.setdefault(base.casefold(), []).append(gene_id)
        mapped = {}
        total_peaks = len(peak_to_tfs)
        retained_count = 0
        # Exact tokens preserve case/whitespace semantics and per-peak audits.
        # Keep this cache local: retained axes/index can differ on the next call.
        symbol_ids = {}
        with TerminalProgress(total_peaks, "Step 6 TF映射", logger=logger) as progress:
            for done, (peak, symbols) in enumerate(peak_to_tfs.items(), 1):
                mapped[peak] = []
                for symbol in dict.fromkeys(symbols):
                    if symbol not in symbol_ids:
                        ids = tuple(sorted(by_symbol.get(symbol.casefold(), ())))
                        if gene_identity is not None and ids:
                            ids = tuple(map_tf_symbols([symbol], gene_identity,
                                                       allowed_ids=base_map).candidate_ids)
                        symbol_ids[symbol] = ids
                    ids = symbol_ids[symbol]
                    if not ids:
                        mapping_diagnostics.append({"peak_id": peak, "tf_symbol": symbol,
                                                    "status": "not_in_model"})
                    retained_count += len(ids)
                    for gene_id in ids:
                        mapped[peak].append(gene_id)
                        mapping_metadata[(peak, gene_id)] = {
                            "tf_gene_name": base_map[gene_id], "tf_mapping_count": len(ids),
                            "alias_candidate": len(ids) > 1}
                progress.update(done)
        peak_to_tfs = mapped
        logger.info("TF候选映射完成: 保留ID候选: %d, 跳过symbol/peak: %d",
                    retained_count, len(mapping_diagnostics))

    # --- 阶段2: TF→peak 调控学习 ---
    max_lag_cfg = cfg.get("max_lag", 5)

    # Auto max_lag: "auto" / 0 / 负数 → n_bins // 4
    if isinstance(max_lag_cfg, str) and max_lag_cfg.lower() == "auto":
        max_lag_cfg = max(1, n_bins // 4)
        logger.info(f"max_lag 自动计算: {max_lag_cfg} (n_bins={n_bins}, n_bins//4)")
    elif isinstance(max_lag_cfg, (int, float)) and max_lag_cfg <= 0:
        max_lag_cfg = max(1, n_bins // 4)
        logger.info(f"max_lag 自动计算: {max_lag_cfg} (n_bins={n_bins}, n_bins//4)")

    time_lag = cfg.get("time_lag", True)
    branch_boundaries = pseudotime_df.attrs.get("branch_boundaries") if hasattr(pseudotime_df, "attrs") else None

    if time_lag:
        logger.info(
            f"RNA→ATAC 阶段2: motif 候选 TF 逐 TF、逐 lag 单特征 Pearson 检验 "
            f"TF(t-lag)→peak(t), lag=0..{max_lag_cfg}, "
            f"每边独立搜索, 第一个显著 lag 胜出"
        )
    else:
        logger.info(
            f"RNA→ATAC 阶段2: 单特征 Pearson 检验 TF→peak 权重 "
            f"(无时间滞后, TF(t)→peak(t), max_lag=0)"
        )
    results = _fit_tf_peak_regression(
        peak_to_tfs, rna_adata, atac_adata, rna_binned, atac_binned,
        max_lag=max_lag_cfg if time_lag else 0,
        pvalue_threshold=cfg.get("tf_peak_pvalue_threshold", 0.05),
        branch_boundaries=branch_boundaries,
        min_abs_contribution=cfg.get("min_abs_contribution", 0.0),
        **({"bin_plan": bin_plan} if bin_plan is not None else {}),
    )

    logger.info(f"RNA→ATAC 拟合完成: {len(results)} 条 TF→peak 调控关系")
    frame = pd.DataFrame(results)
    if base_map is not None:
        for column in ("tf_gene_name", "tf_mapping_count", "alias_candidate"):
            frame[column] = [mapping_metadata[(row["peak_id"], row["tf_gene"])][column]
                             for row in results]
        frame.attrs["tf_mapping_diagnostics"] = mapping_diagnostics
    frame["screening_status"] = "linear_pearson_screen"
    if continuous:
        frame["atac_state_unit"] = "normalized_atac_log1p"
    registry = fit_tf_peak(rna_adata, atac_adata, pseudotime_df, frame, config,
                           bin_plan=bin_plan)
    validate_registry(registry, expected_type="saturating_hybrid_tf_peak",
                      expected_schema=expected_registry_schema("saturating_hybrid_tf_peak",config))
    tf_model_registry.clear()
    tf_model_registry.update(registry)
    return frame


# ---- JASPAR motif 扫描实现 ----

# ---- motif 数据库加载 ----

def _load_motif_database(db_name: str = "jaspar2024", cache_dir: Optional[Path] = None,
                         local_first=False) -> Optional[dict]:
    """
    加载 motif 数据库。优先使用 gimmemotifs (gimme.vertebrate + CIS-BP)，
    不可用时回退到 JASPAR。

    Returns
    -------
    motifs : dict or None
        {motif_id: {"name": str, "tf_name": str, "pwm": ndarray}}
    """
    # 策略A: gimmemotifs 数据文件
    motifs = _load_gimmemotifs_database(cache_dir)
    if motifs and len(motifs) > 100:
        return motifs

    # 策略B: JASPAR 在线/缓存
    if local_first:
        directory = cache_dir if cache_dir is not None else Path.home() / ".CausalBridge"
        path = directory / "jaspar_motifs.pkl"
        if path.exists():
            try:
                with path.open("rb") as handle:
                    return pickle.load(handle)
            except Exception:
                pass
    return _load_jaspar_motifs(db_name, cache_dir)


def _load_gimmemotifs_database(cache_dir: Optional[Path] = None) -> Optional[dict]:
    """
    从 gimmemotifs 安装目录加载 motif 数据库。

    加载 gimme.vertebrate.v5.0 (curated vertebrate motifs) 并补充
    CIS-BP (大规模计算推断 motif) 中不重复的 TF。

    通过 motif2factors 映射将 motif cluster ID 关联到具体 TF 名称，
    每个 TF 只保留一个最优 motif（优先 curated 证据）。

    Returns
    -------
    motifs : dict or None
        与 _load_jaspar_motifs 相同格式
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".CausalBridge"
    # 优先从缓存加载
    cache_path = cache_dir / "gimmemotifs_cache.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                motifs = pickle.load(f)
            logger.info(f"从 gimmemotifs 缓存加载了 {len(motifs)} 个 motif")
            return motifs
        except Exception:
            pass

    # Use the active interpreter's site-packages directory rather than a
    # machine- or environment-name-specific Conda path.
    possible_dirs = [
        Path(get_paths()["purelib"]) / "data" / "motif_databases",
    ]

    data_dir = None
    for d in possible_dirs:
        if d.exists():
            data_dir = d
            break

    if data_dir is None:
        logger.debug("未找到 gimmemotifs 数据目录")
        return None

    try:
        motifs = {}

        # 优先级: gimme.vertebrate.v5.0 (curated) > CIS-BP (broad)
        sources = [
            ("gimme.vertebrate.v5.0", "gimme.vertebrate.v5.0.pfm",
             "gimme.vertebrate.v5.0.motif2factors.txt"),
        ]

        for source_name, pfm_file, mapping_file in sources:
            pfm_path = data_dir / pfm_file
            mapping_path = data_dir / mapping_file
            if not pfm_path.exists() or not mapping_path.exists():
                continue

            n_loaded = _parse_gimmemotifs_pfm(pfm_path, mapping_path, motifs)
            logger.info(f"  从 {source_name} 加载了 {n_loaded} 个 motif")

        # 补充 CIS-BP
        cisbp_pfm = data_dir / "CIS-BP.pfm"
        cisbp_map = data_dir / "CIS-BP.motif2factors.txt"
        if cisbp_pfm.exists() and cisbp_map.exists():
            n_cisbp = _parse_cisbp_pfm(cisbp_pfm, cisbp_map, motifs)
            logger.info(f"  从 CIS-BP 补充了 {n_cisbp} 个 motif")

        if not motifs:
            return None

        logger.info(f"  gimmemotifs 原始: {len(motifs)} 个 motif")

        # TF 中心去重: 同一 TF 的多个 motif 合并 all_factors, 仅保留最优 PWM
        motifs = _dedup_motifs_by_tf(motifs)
        logger.info(f"  gimmemotifs 去重后: {len(motifs)} 个 motif")

        # 保存缓存
        cache_path = cache_dir / "gimmemotifs_cache.pkl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(motifs, f)
        logger.info(f"  gimmemotifs motif 数据集已缓存至 {cache_path}")

        return motifs

    except Exception as e:
        logger.warning(f"gimmemotifs 数据库加载失败: {e}")
        return None


def _parse_gimmemotifs_pfm(
    pfm_path: Path, mapping_path: Path, existing: dict
) -> int:
    """解析 gimme.vertebrate PFM 文件，合并到 existing dict 中。"""
    from collections import defaultdict

    # 解析 motif2factors 映射: motif_id → [(factor, curated), ...]
    motif_to_factors = defaultdict(list)
    with open(mapping_path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                motif_id, factor, evidence, curated = parts[0], parts[1], parts[2], parts[3]
                motif_to_factors[motif_id].append({
                    "factor": factor.upper(),
                    "curated": curated == "Y",
                    "evidence": evidence,
                })

    # 解析 PFM 文件
    motifs = {}
    current_id = None
    current_pfm = None
    current_comment = None

    with open(pfm_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                current_comment = line[1:]  # 去掉 #
                continue
            if line.startswith(">"):
                # 保存上一个 motif
                if current_id is not None and current_pfm is not None:
                    motifs[current_id] = {
                        "pfm": np.array(current_pfm, dtype=float),
                        "factors": motif_to_factors.get(current_id, []),
                        "comment": current_comment,
                    }
                current_id = line[1:]
                current_pfm = []
                current_comment = None
                continue
            # PFM 行: A C G T
            values = [float(x) for x in line.split("\t")]
            if len(values) == 4:
                if current_pfm is None:
                    current_pfm = []
                current_pfm.append(values)

    # 保存最后一个
    if current_id is not None and current_pfm is not None:
        motifs[current_id] = {
            "pfm": np.array(current_pfm, dtype=float),
            "factors": motif_to_factors.get(current_id, []),
            "comment": current_comment,
        }

    # 构建 motif 中心条目: 每个 motif 保留一份 PWM 和完整的 factor 列表
    # (含 cofactor — 无自身 PWM 但通过 motif 关联到 peak)
    n_added = 0
    for motif_id, info in motifs.items():
        pfm = info["pfm"]
        if pfm.shape[0] < 5:
            continue
        all_factors = info.get("factors", [])
        if not all_factors:
            continue

        pwm = _pfm_to_pwm(pfm.T, pseudocount=0.01)

        # 选主因子 (优先 curated, 否则取第一个)
        primary = all_factors[0]["factor"]
        for fi in all_factors:
            if fi.get("curated"):
                primary = fi["factor"]
                break

        key = f"GM_{motif_id}"
        if key in existing:
            # 合并因子 (来自多个 source 的同名 motif)
            seen = {f["factor"] for f in existing[key].get("all_factors", [])}
            for fi in all_factors:
                if fi["factor"] not in seen:
                    existing[key]["all_factors"].append(fi)
                    seen.add(fi["factor"])
        else:
            existing[key] = {
                "tf_name": primary,
                "pfm": pfm,
                "pwm": pwm,
                "source": "gimmemotifs",
                "all_factors": all_factors,
            }
            n_added += 1

    return n_added


def _parse_cisbp_pfm(pfm_path: Path, mapping_path: Path, existing: dict) -> int:
    """解析 CIS-BP PFM 文件，补充 existing dict 中缺失的 TF。"""
    from collections import defaultdict

    # 解析 motif2factors
    motif_to_factors = defaultdict(set)
    with open(mapping_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                motif_to_factors[parts[0]].add(parts[1].upper())

    # 解析 PFM
    current_id = None
    current_pfm = None
    cisbp_motifs = {}

    with open(pfm_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if current_id is not None and current_pfm is not None:
                    cisbp_motifs[current_id] = {
                        "pfm": np.array(current_pfm, dtype=float),
                        "factors": motif_to_factors.get(current_id, set()),
                    }
                current_id = line[1:]
                current_pfm = []
                continue
            values = [float(x) for x in line.split("\t")]
            if len(values) == 4:
                current_pfm.append(values)

    if current_id is not None and current_pfm is not None:
        cisbp_motifs[current_id] = {
            "pfm": np.array(current_pfm, dtype=float),
            "factors": motif_to_factors.get(current_id, set()),
        }

    # 构建 motif 中心条目: 保留完整 factor 列表
    n_added = 0
    for motif_id, info in cisbp_motifs.items():
        pfm = info["pfm"]
        if pfm.shape[0] < 5:
            continue
        all_factors_raw = info.get("factors", set())
        if not all_factors_raw:
            continue

        # The producer stores factors in a set; lexical order gives a stable
        # primary factor and stable downstream TF/motif insertion order.
        all_factors = [
            {"factor": f.upper(), "curated": False, "evidence": "Direct"}
            for f in sorted(all_factors_raw)
        ]
        pwm = _pfm_to_pwm(pfm.T, pseudocount=0.01)
        primary = all_factors[0]["factor"]

        key = f"CISBP_{motif_id}"
        if key in existing:
            seen = {f["factor"] for f in existing[key].get("all_factors", [])}
            for fi in all_factors:
                if fi["factor"] not in seen:
                    existing[key]["all_factors"].append(fi)
                    seen.add(fi["factor"])
        else:
            existing[key] = {
                "tf_name": primary,
                "pfm": pfm,
                "pwm": pwm,
                "source": "CIS-BP",
                "all_factors": all_factors,
            }
            n_added += 1

    return n_added


def _dedup_motifs_by_tf(motifs: dict) -> dict:
    """
    TF 中心去重: 每个 TF 独立选择最优 motif (IC + curated bonus),
    同时合并该 TF 在所有 motif 中的 all_factors。

    沿用旧 tf_best 逻辑:
    - 每个 TF (含 cofactor) 从所有关联 motif 中选出最优 PWM
    - Score = _compute_ic(PFM) + 10.0 (curated bonus)
    - 合并 all_factors: 收集该 TF 出现在所有 motif 中的全部 factor 关联
    - 避免 key 冲突: 两 TF 共享同一最优 motif 时用后缀区分
    """
    from collections import defaultdict

    # 每个 TF 独立: 选最优 motif + 收集 all_factors
    tf_best = {}           # tf_name → (score, motif_id, info)
    tf_all_factors = defaultdict(dict)  # tf_name → {factor_name: factor_info}

    for motif_id, info in motifs.items():
        pfm = info.get("pfm")
        if pfm is None or pfm.shape[0] < 5:
            continue

        ic = _compute_ic(pfm)

        all_factors = info.get("all_factors", [])
        if not all_factors:
            continue

        for fi in all_factors:
            tf = fi["factor"]
            score = ic + (10.0 if fi.get("curated") else 0.0)

            # Preserve the existing first-motif priority on equal scores.
            # Factor lists from set-backed producers must already be stable.
            if tf not in tf_best or score > tf_best[tf][0]:
                tf_best[tf] = (score, motif_id, info)

            # 收集该 motif 的全部 factor 到该 TF (含 cofactor 关联)
            for fj in all_factors:
                fname = fj["factor"]
                if fname not in tf_all_factors[tf]:
                    tf_all_factors[tf][fname] = dict(fj)
                elif fj.get("curated") and not tf_all_factors[tf][fname].get("curated"):
                    tf_all_factors[tf][fname] = dict(fj)

    # 构建去重后的 dict
    deduped = {}
    for tf, (score, motif_id, info) in tf_best.items():
        merged_factors = list(tf_all_factors[tf].values())
        # 排序: tf 自身第一, 然后 curated, 然后其他
        merged_factors.sort(key=lambda f: (
            f["factor"] != tf,
            not f.get("curated", False),
        ))

        new_info = {
            "tf_name": tf,
            "pfm": info.get("pfm"),
            "pwm": info["pwm"],
            "source": info.get("source", "unknown"),
            "all_factors": merged_factors,
        }

        # 避免 key 冲突: 两个 TF 共享同一最优 motif 时用后缀区分
        if motif_id in deduped:
            motif_id = f"{motif_id}__{tf}"
        deduped[motif_id] = new_info

    return deduped


def _compute_ic(pfm: np.ndarray) -> float:
    """计算 PFM 的信息量 (Information Content)，用于 motif 质量排序。"""
    n = pfm.sum(axis=0, keepdims=True)
    ppm = (pfm + 0.01) / (n + 0.04)
    ic_per_pos = 2.0 + np.sum(ppm * np.log2(ppm + 1e-10), axis=0)
    return float(np.sum(ic_per_pos))


def _load_jaspar_motifs(db_name: str = "jaspar2024", cache_dir: Optional[Path] = None) -> Optional[dict]:
    """
    加载 JASPAR motif 数据库。

    优先使用 biopython 的 JASPAR API 在线获取；
    如不可用，尝试从本地缓存文件加载；
    如仍不可用，返回 None（触发 fallback 策略）。

    Returns
    -------
    motifs : dict or None
        {motif_id: {"name": str, "tf_name": str, "pwm": ndarray}}
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".CausalBridge"

    # 策略A: biopython 在线 API
    try:
        from Bio.motifs import parse as parse_motifs
        import io, urllib.request

        # JASPAR 2024 脊椎动物核心集 URL
        url = JASPAR_RESOURCE_URL
        response = urllib.request.urlopen(url, timeout=30)
        content = response.read().decode("utf-8")

        motifs = {}
        for record in parse_motifs(io.StringIO(content), "jaspar"):
            motif_id = record.matrix_id
            pfm = np.array([record.counts[n] for n in "ACGT"], dtype=float)
            # PFM → PWM (Position Weight Matrix): 加伪计数后 log-odds
            pwm = _pfm_to_pwm(pfm, pseudocount=0.01, background=[0.25, 0.25, 0.25, 0.25])
            motifs[motif_id] = {
                "name": record.name,
                "tf_name": record.name.split("(")[0].strip() if hasattr(record, "name") else motif_id,
                "pwm": pwm,
            }
        logger.info(f"从 JASPAR 在线加载了 {len(motifs)} 个 motif")

        # 保存本地缓存
        cache_path = cache_dir / "jaspar_motifs.pkl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(motifs, f)
        logger.info(f"JASPAR motif 已缓存至 {cache_path}")

        return motifs

    except Exception as e:
        logger.debug(f"JASPAR 在线加载失败: {e}")

    # 策略B: 本地缓存
    cache_path = cache_dir / "jaspar_motifs.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                motifs = pickle.load(f)
            logger.info(f"从本地缓存加载了 {len(motifs)} 个 motif")
            return motifs
        except Exception:
            pass

    # 不可用
    return None


def _pfm_to_pwm(
    pfm: np.ndarray,
    pseudocount: float = 0.01,
    background: list = None,
) -> np.ndarray:
    """
    Position Frequency Matrix → Position Weight Matrix.

    PWM[i, j] = log2( (PFM[i,j] + pc) / (N + 4*pc) / bg[i] )
    其中 N 是列总和，pc 是伪计数，bg[i] 是碱基 i 的背景频率。
    """
    if background is None:
        background = [0.25, 0.25, 0.25, 0.25]
    N = pfm.sum(axis=0, keepdims=True)
    ppm = (pfm + pseudocount) / (N + 4 * pseudocount)
    pwm = np.log2(ppm / np.array(background)[:, None])
    return pwm


def _scan_peaks_with_jaspar(
    atac_adata,
    rna_adata,
    causal_edges: pd.DataFrame,
    motif_db: dict,
    cfg: dict,
    config: dict,
    frozen_tf_database=_UNSET_TF_DATABASE,
    raw_scan_cache=None,
) -> Dict[str, List[str]]:
    """Reuse complete raw hits by sequence resources, then materialize TFs anew.

    Only successful extracted sequences (including zero hits) establish cache
    coverage. Uncovered peaks use the existing, uncached frozen-TF fallback.
    """
    if frozen_tf_database is None:
        raise ValueError("frozen TF database must be a set (possibly empty), not None")

    max_tfs = cfg.get("max_tfs_per_peak", 5)
    threshold = cfg.get("motif_pval_threshold", 1e-4)
    workers = _validate_motif_scan_workers(cfg.get("motif_scan_workers", 1))

    descriptor_config = dict(config, output=dict(config.get("output", {})))
    if raw_scan_cache is not None:
        descriptor_config["output"]["cache_dir"] = raw_scan_cache["base_cache_dir"]
    descriptor = _raw_scan_cache_descriptor(descriptor_config, motif_db, cfg)
    if raw_scan_cache is not None and descriptor != raw_scan_cache:
        raise ValueError("raw motif scan descriptor differs from current resources/parameters")
    base = Path(descriptor["base_cache_dir"])
    path = base / "peak_motif_raw_v1" / (descriptor["scan_key"] + ".pkl")
    records = _read_raw_motif_cache(path, descriptor, motif_db)
    if not atac_adata.var_names.is_unique:
        raise ValueError("raw motif cache requires unique peak IDs")
    requested = [peak for peak in causal_edges["peak_id"].unique() if peak in atac_adata.var_names]
    keys = {}
    for peak in requested:
        row = atac_adata.var.loc[peak]
        key = (str(peak), str(row["chr"]), int(row["start"]), int(row["end"]))
        if key[2] < 0 or key[3] <= key[2]:
            raise ValueError("invalid peak coordinates")
        keys[peak] = key
    legacy_path = base / "peak_tf_cache_v2.pkl"
    raw_missing = [peak for peak in requested if keys[peak] not in records]
    legacy_result = _read_legacy_peak_tf_candidates(
        legacy_path, threshold, raw_missing, max_tfs) if raw_missing else {}
    missing = [peak for peak in requested
               if keys[peak] not in records and peak not in legacy_result]
    additions = {}
    if missing:
        sequences = _extract_peak_sequences(atac_adata, config, requested_peaks=missing) or {}
        tasks = [(keys[peak], sequences[peak]) for peak in missing if sequences.get(peak)]
        additions = _scan_sequence_records(tasks, motif_db, threshold, workers)
        if additions:
            _merge_raw_motif_cache(path, descriptor, motif_db, additions)
            records.update(additions)
    else:
        logger.info("Primary motif scan: requested_workers=%d effective_workers=0 peaks=0 (cache covered)", workers)

    result = dict(legacy_result)
    for peak in requested:
        if keys[peak] in records:
            names = _materialize_raw_motif_hits(records[keys[peak]], motif_db, max_tfs)
            if names:
                result[peak] = names
    uncovered = [peak for peak in missing if keys[peak] not in records]
    if uncovered:
        logger.warning("No sequence coverage for %d requested peaks; uncached TF fallback", len(uncovered))
        result.update(_assign_tfs_fallback(atac_adata, rna_adata,
            causal_edges[causal_edges["peak_id"].isin(uncovered)], _get_cache_dir(config),
            max_tfs_per_peak=max_tfs,
            **({"tf_database": frozen_tf_database} if frozen_tf_database is not _UNSET_TF_DATABASE else {})))
    logger.info("Shared motif candidates: requested=%d verified_raw=%d reused_legacy=%d scanned=%d missing_failed=%d",
                len(requested), sum(keys[p] in records for p in requested),
                len(legacy_result), len(additions), len(uncovered))
    return result


# ============================================================================
# 家族代理 motif 机制
# ============================================================================


def _build_family_groups(motif_db: dict) -> Tuple[Dict[str, set], Dict[str, str]]:
    """
    从 motif 数据库构建 TF 家族分组。

    两层策略:
    1. GM motif ID 格式: GM_GM.5.0.{FAMILY}.{number} → 直接提取家族名
    2. 前缀推断: 对 GM 分组未覆盖的 TF，通过基因名前缀
       (去掉尾部数字的根名) 匹配到已知家族或其他同前缀 TF

    Returns
    -------
    family_groups : dict {family_name: {tf1, tf2, ...}}
    tf_to_family : dict {tf_name: family_name}
    """
    from collections import defaultdict
    import re

    family_groups = defaultdict(set)
    tf_to_family = {}

    # --- 第1层: 从 GM motif ID 提取家族 ---
    for motif_id, info in motif_db.items():
        if motif_id.startswith("GM_GM.5.0."):
            parts = motif_id.split(".")
            if len(parts) >= 5:
                family = parts[3]
                # 添加所有关联因子 (含 cofactor)
                for fi in info.get("all_factors", []):
                    tf = fi["factor"]
                    if tf:
                        family_groups[family].add(tf)
                        if tf not in tf_to_family:
                            tf_to_family[tf] = family
                # 向后兼容: 无 all_factors 的旧格式
                tf = info.get("tf_name", "")
                if tf:
                    family_groups[family].add(tf)
                    if tf not in tf_to_family:
                        tf_to_family[tf] = family

    # --- 第2层: 前缀推断，覆盖 CIS-BP 及其他来源的 TF ---
    # 提取基因名根 (去掉尾部数字): RFX8 → RFX, FOXA1 → FOXA
    def _gene_root(name: str) -> str:
        m = re.match(r"^([A-Za-z]+?)\d*$", name)
        return m.group(1).upper() if m else name.upper()

    # 搜集所有已知 TF 及其 root
    all_tfs = set()
    for info in motif_db.values():
        tf = info.get("tf_name", "")
        if tf:
            all_tfs.add(tf)
        for fi in info.get("all_factors", []):
            fname = fi["factor"]
            if fname:
                all_tfs.add(fname)

    # 构建 root → {tf1, tf2, ...} 映射
    root_groups = defaultdict(set)
    for tf in sorted(all_tfs):
        root_groups[_gene_root(tf)].add(tf)

    # 对于尚未分配家族的 TF，尝试通过 root 匹配
    #   优先级: 匹配已知 GM 家族名 → 匹配有意义的 root 组
    # Keep GM assignments and the existing inference priorities. Lexical order
    # makes first-assigned ownership stable when a root spans several families.
    for tf in sorted(all_tfs):
        if tf in tf_to_family:
            continue
        root = _gene_root(tf)

        # 如果 root 恰好是一个已知的 GM 家族名 (如 RFX, SOX, FOX, GATA)
        if root in family_groups:
            tf_to_family[tf] = root
            family_groups[root].add(tf)
            continue

        # 如果 root 组内已有成员在某个家族中，归入同一家族
        root_members = root_groups.get(root, set())
        # Inherit from the lexical first already-assigned member of this root.
        for member in sorted(root_members):
            if member in tf_to_family:
                tf_to_family[tf] = tf_to_family[member]
                family_groups[tf_to_family[member]].add(tf)
                break
        else:
            # 如果 root 组至少有 2 个成员且都不在任何家族中，
            # 创建一个以 root 命名的新家族
            unassigned = [m for m in sorted(root_members) if m not in tf_to_family]
            if len(unassigned) >= 2:
                family_groups[root].update(unassigned)
                for m in unassigned:
                    tf_to_family[m] = root
            elif len(unassigned) == 1:
                # 只有一个成员，归属到 root 家族便于后续扩展
                family_groups[root].add(unassigned[0])
                tf_to_family[unassigned[0]] = root

    logger.debug(f"家族分组: {len(family_groups)} 个家族, {len(tf_to_family)} 个 TF")
    return dict(family_groups), tf_to_family


def _supplement_family_proxies(
    peak_to_tfs: dict,
    motif_db: dict,
    target_gene_types: dict,
    cfg: dict,
    config: dict,
    *,
    atac_adata=None,
    raw_scan_cache=None,
) -> dict:
    """
    靶 TF 家族 motif 扩展：两阶段策略。

    **阶段 A — 被动借用（快速路径）**：
    对每个 peak，如果其 top-5 motif 匹配中已有靶 TF 的家族成员，
    直接将靶 TF 也加入该 peak 的候选列表。
    无需重新扫描 DNA，完全基于现有缓存。

    **阶段 B — 全家族 PWM 扫描（强制）**：
    所有有家族信息的靶 TF，用其**全部**家族成员的 PWM 扫描每个
    causal peak。任意家族成员匹配（p < threshold）→ 靶 TF 加入该 peak。
    不同家族成员的 PWM 有细微差异，全扫描比单最优 PWM 覆盖更全面。

    **核心原则（v1.29）**：
    只扩展靶 TF 自身，绝不将所有家族成员写入所有 peak。
    这避免了 C2H2_ZF 等大家族（300+ 成员）的关联爆炸。

    示例：
      靶 TF: RUNX1, 家族: {RUNX,RUNX1,RUNX2,RUNX3}
      阶段A: peak 有 RUNX2 → 加入 RUNX1（被动，0 成本）
      阶段B: 用 RUNX3 的 PWM 扫描所有 peak →
              p<0.05 的 peak 也加入 RUNX1（主动扫描）

    Returns
    -------
    supplemented : dict {gene: {before, after, delta, proxy_tf, proxy_motif_id}}
        受益的靶基因 TF 的匹配数变化
    """
    from numpy.lib.stride_tricks import sliding_window_view

    # --- 收集靶 TF 集合 ---
    target_tfs = set()
    for gene, gtype in target_gene_types.items():
        if gtype == "TF":
            target_tfs.add(gene.upper())

    if not target_tfs:
        logger.debug("  无靶 TF，跳过家族扩展")
        return {}

    # --- 构建家族分组 ---
    family_groups, tf_to_family = _build_family_groups(motif_db)
    if not family_groups:
        logger.debug("  无家族分组信息，跳过家族扩展")
        return {}

    # --- 找到每个靶 TF 所属家族 ---
    target_family_map = {}
    for tf in target_tfs:
        family = tf_to_family.get(tf)
        if family:
            target_family_map[tf] = family

    if not target_family_map:
        logger.debug("  靶 TF 均无家族信息，跳过家族扩展")
        return {}

    # --- 阶段 A: 被动借用（基于现有缓存） ---
    tf_match_before = {}
    for tf_list in peak_to_tfs.values():
        for tf in tf_list:
            tf_upper = tf.upper()
            if tf_upper in target_tfs:
                tf_match_before[tf_upper] = tf_match_before.get(tf_upper, 0) + 1

    n_passive_expanded = 0
    n_passive_added = 0
    for peak_id, tf_list in peak_to_tfs.items():
        existing_upper = {tf.upper() for tf in tf_list}
        tfs_to_add = []
        for target_tf, family in target_family_map.items():
            if target_tf in existing_upper:
                continue
            family_members = family_groups.get(family, set())
            if any(tf.upper() in family_members for tf in tf_list):
                tfs_to_add.append(target_tf)

        if tfs_to_add:
            n_passive_expanded += 1
            new_tfs = list(tf_list)
            for tf in tfs_to_add:
                new_tfs.append(tf)
                n_passive_added += 1
            peak_to_tfs[peak_id] = new_tfs

    # 阶段 A 后的匹配数
    tf_match_after_passive = {}
    for tf_list in peak_to_tfs.values():
        for tf in tf_list:
            tf_upper = tf.upper()
            if tf_upper in target_tfs:
                tf_match_after_passive[tf_upper] = tf_match_after_passive.get(tf_upper, 0) + 1

    if n_passive_expanded > 0:
        logger.info(
            f"  阶段A (被动借用): {n_passive_expanded}/{len(peak_to_tfs)} 个 peak 被扩展, "
            f"共添加 {n_passive_added} 个 TF-peak 关联"
        )

    # --- 阶段 B: 主动 PWM 扫描（全家族 motif，所有靶 TF 强制） ---
    threshold = cfg.get("motif_pval_threshold", 1e-4)
    nucleotide_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}

    # 所有有家族信息的靶 TF 都参与主动扫描
    scan_targets = {}
    for tf_upper in sorted(target_tfs):
        if tf_upper in tf_to_family:
            current = tf_match_after_passive.get(tf_upper, 0)
            scan_targets[tf_upper] = current

    if scan_targets:
        family_descriptor = None
        family_records = {}
        family_path = None
        family_missing = list(peak_to_tfs.keys())
        if atac_adata is not None:
            family_descriptor = _family_cache_descriptor(
                config, motif_db, target_family_map, family_groups, threshold, raw_scan_cache)
            family_path = (Path(family_descriptor["base_cache_dir"]) /
                           "peak_family_motif_v2" /
                           (family_descriptor["scan_key"] + ".pkl"))
            family_records = _family_cache_read(family_path, family_descriptor)
            family_missing = []
            for peak_id in peak_to_tfs:
                if peak_id not in atac_adata.var_names:
                    family_missing.append(peak_id)
                    continue
                row = atac_adata.var.loc[peak_id]
                key = (str(peak_id), str(row["chr"]), int(row["start"]), int(row["end"]))
                additions = family_records.get(key)
                if isinstance(additions, list) and all(tf in scan_targets for tf in additions):
                    existing = peak_to_tfs.get(peak_id, [])
                    for tf in additions:
                        if tf not in {name.upper() for name in existing}:
                            peak_to_tfs[peak_id] = list(existing) + [tf]
                            existing = peak_to_tfs[peak_id]
                else:
                    family_missing.append(peak_id)
        logger.info(
            f"  阶段B (全家族 PWM 扫描): targets={len(scan_targets)}, "
            f"family_pwms≈{sum(1 for info in motif_db.values() if info.get('tf_name'))}, "
            f"requested_peaks={len(peak_to_tfs)}, missing_peaks={len(family_missing)}"
        )

        # 提取 peak 序列（只提取一次，所有靶 TF 共用）
        peak_sequences = _extract_peak_sequences_from_causal(
            family_missing, config
        ) if family_missing else {}
        # Missing extraction is uncovered, not successful negative evidence.
        peak_sequences = peak_sequences or {}
        family_new_records = {}
        if atac_adata is not None:
            for peak_id in peak_sequences:
                if peak_id not in atac_adata.var_names:
                    continue
                row = atac_adata.var.loc[peak_id]
                family_new_records[(str(peak_id), str(row["chr"]),
                                    int(row["start"]), int(row["end"]))] = []

        if peak_sequences:
            for tf_upper, current_matches in scan_targets.items():
                family = tf_to_family[tf_upper]
                family_members = family_groups.get(family, set())

                # 收集全家族的有效 PWM（排除靶 TF 自身）
                family_pwms = []
                for mid, info in motif_db.items():
                    member_tf = info.get("tf_name", "")
                    if member_tf == tf_upper:
                        continue  # 跳过自身
                    if member_tf.upper() not in family_members and member_tf not in family_members:
                        continue
                    pwm = np.array(info.get("pwm", []))
                    if pwm.ndim != 2 or pwm.shape[1] < 5:
                        continue
                    family_pwms.append((mid, member_tf, pwm))

                if not family_pwms:
                    logger.debug(
                        f"    {tf_upper}: 家族 {family} 中无可用家族成员 PWM"
                    )
                    continue

                logger.debug(
                    f"    {tf_upper} (当前 {current_matches}): "
                    f"← 使用家族 {family} 的 {len(family_pwms)} 个 PWM 扫描"
                )

                # 对每个 peak，检查是否被任意家族成员的 PWM 匹配
                n_new = 0
                for peak_id, seq in peak_sequences.items():
                    seq_upper = seq.upper()
                    seq_ints = np.array(
                        [nucleotide_to_idx.get(nt, -1) for nt in seq_upper],
                        dtype=np.int8,
                    )

                    # Primary/passive candidates are not independent PWM evidence.
                    existing_tfs = peak_to_tfs.get(peak_id, [])
                    existing_upper = {t.upper() for t in existing_tfs}

                    # 用全家族 PWM 扫描此 peak，任意匹配即加入
                    matched = False
                    for mid, member_tf, pwm in family_pwms:
                        mlen = pwm.shape[1]
                        if len(seq_ints) < mlen:
                            continue

                        windows = sliding_window_view(seq_ints, mlen)
                        valid_mask = (windows >= 0).all(axis=1)
                        if not valid_mask.any():
                            continue

                        valid_windows = windows[valid_mask]
                        col_idx = np.arange(mlen)
                        scores = np.sum(pwm[valid_windows, col_idx], axis=1)
                        max_score = float(scores.max())

                        # Gumbel p-value
                        null_mean = float(np.mean(scores))
                        null_std = float(np.std(scores))
                        if null_std < 1e-10:
                            continue
                        gumbel_beta = null_std * np.sqrt(6.0) / np.pi
                        gumbel_mu = null_mean - np.euler_gamma * gumbel_beta
                        p_val = 1.0 - np.exp(-np.exp(
                            -(max_score - gumbel_mu) / gumbel_beta
                        ))
                        p_val = float(np.clip(p_val, 0.0, 1.0))

                        if p_val < threshold:
                            matched = True
                            break  # 任意家族成员匹配即停止

                    if matched:
                        if tf_upper not in existing_upper:
                            peak_to_tfs[peak_id] = existing_tfs + [tf_upper]
                            n_new += 1
                        if atac_adata is not None and peak_id in atac_adata.var_names:
                            row = atac_adata.var.loc[peak_id]
                            key = (str(peak_id), str(row["chr"]), int(row["start"]), int(row["end"]))
                            family_new_records.setdefault(key, []).append(tf_upper)

                logger.debug(
                    f"    {tf_upper}: 全家族扫描新增 {n_new} 个 peak 匹配 "
                    f"({current_matches} → {current_matches + n_new})"
                )
        if family_path is not None and family_new_records:
            _family_cache_merge(family_path, family_descriptor, family_new_records)
        logger.info("  阶段B family cache: path=%s reused_peaks=%d scanned_peaks=%d missing_failed=%d new_records=%d",
                    family_path, len(peak_to_tfs) - len(family_missing),
                    len(peak_sequences), len(set(family_missing) - set(peak_sequences)),
                    len(family_new_records))
        if family_missing and not peak_sequences:
            logger.warning("  无法提取 peak 序列，阶段B 跳过")
    else:
        logger.info("  阶段B (全家族 PWM 扫描): 无靶 TF 需要扫描（无家族信息）")

    # --- 统计最终匹配数 ---
    tf_match_final = {}
    for tf_list in peak_to_tfs.values():
        for tf in tf_list:
            tf_upper = tf.upper()
            if tf_upper in target_tfs:
                tf_match_final[tf_upper] = tf_match_final.get(tf_upper, 0) + 1

    # --- 报告 ---
    supplemented = {}
    for tf_upper in sorted(target_tfs):
        before = tf_match_before.get(tf_upper, 0)
        after = tf_match_final.get(tf_upper, 0)
        if after > before:
            supplemented[tf_upper] = {
                "before": before,
                "after": after,
                "delta": after - before,
            }
            logger.debug(
                f"  {tf_upper}: 家族扩展 {before} → {after} 个 peak 匹配 "
                f"(+{after - before})"
            )
        else:
            logger.debug(
                f"  {tf_upper}: 家族扩展无变化 ({before} 个 peak 匹配)"
            )

    return supplemented


def _extract_peak_sequences_from_causal(
    peak_ids,
    config: dict,
) -> Optional[Dict[str, str]]:
    """
    仅提取因果 peak 的 DNA 序列（用于家族代理 motif 重扫描）。

    与 _extract_peak_sequences 不同，此函数只处理指定的 peak ID 集合。
    """
    from Bio import SeqIO

    genome_path = config.get("input", {}).get("genome_fasta")
    if not genome_path or not Path(genome_path).exists():
        return None

    try:
        genome = SeqIO.to_dict(SeqIO.parse(genome_path, "fasta"))
    except Exception as e:
        logger.warning(f"基因组 FASTA 加载失败: {e}")
        return None

    sequences = {}
    peak_ids_list = list(peak_ids) if not isinstance(peak_ids, list) else peak_ids

    for peak_id in peak_ids_list:
        try:
            if ":" in peak_id:
                chrom, coords = peak_id.split(":")
                start, end = map(int, coords.split("-"))
            else:
                # BED 格式: chr1-100000-200000
                parts = peak_id.split("-")
                chrom = parts[0]
                start, end = int(parts[-2]), int(parts[-1])
            if chrom not in genome:
                continue
            if start < 0 or end <= start or end > len(genome[chrom].seq):
                continue
            seq = str(genome[chrom].seq[start:end])
            sequences[peak_id] = seq
        except (ValueError, KeyError, IndexError):
            continue

    return sequences


def _extract_peak_sequences(
    atac_adata, config: dict, requested_peaks=None
) -> Optional[Dict[str, str]]:
    """
    从基因组 FASTA 提取 peak 的 DNA 序列。

    需要用户提供参考基因组 FASTA 文件路径（config["input"]["genome_fasta"]）。

    Returns
    -------
    sequences : dict {peak_id: dna_sequence} or None
    """
    genome_path = config.get("input", {}).get("genome_fasta")
    if not genome_path or not Path(genome_path).exists():
        return None

    try:
        from Bio import SeqIO

        # 构建染色体 → 序列的索引（仅加载所需染色体）
        logger.info(f"  加载基因组: {genome_path}")
        genome = SeqIO.to_dict(SeqIO.parse(genome_path, "fasta"))

        sequences = {}
        for peak_id in (atac_adata.var_names if requested_peaks is None else requested_peaks):
            info = atac_adata.var.loc[peak_id]
            chrom = info["chr"]
            # 尝试多种染色体命名格式
            for chrom_fmt in [chrom, f"chr{chrom}", chrom.replace("chr", "")]:
                if chrom_fmt in genome:
                    try:
                        start, end = int(info["start"]), int(info["end"])
                        if requested_peaks is not None and not 0 <= start < end <= len(genome[chrom_fmt].seq):
                            continue
                        seq = str(genome[chrom_fmt].seq[
                            start : end
                        ])
                        sequences[peak_id] = seq
                        break
                    except Exception:
                        continue

        if len(sequences) == 0:
            logger.warning(
                "  无法匹配染色体名。请确保基因组 FASTA 中的染色体名 "
                "与 AnnData .var['chr'] 中的一致"
            )
            return None

        logger.info(f"  成功提取 {len(sequences)} 个 peak 的序列")
        return sequences

    except ImportError:
        logger.warning("  biopython 不可用，无法提取序列。pip install biopython")
        return None
    except Exception as e:
        logger.warning(f"  序列提取失败: {e}")
        return None


def _assign_tfs_fallback(
    atac_adata, rna_adata, causal_edges: pd.DataFrame,
    cache_dir: Optional[Path] = None,
    max_tfs_per_peak: int = 5,
    *, tf_database=None, gene_identity=None,
) -> Dict[str, List[str]]:
    """
    Fallback 策略: 当 motif 扫描不可用时，基于 TF 数据库匹配。

    从 causal_edges 中找出与每个 peak 关联的基因，筛选其中是 TF 的基因。
    这依赖于一个较大的 TF 数据库文件而非硬编码列表。

    改进（相比初版的硬编码列表）：
    - 从外部文件加载 TF 数据库（如 AnimalTFDB）
    - 如文件不存在，从 RNA 数据中有 motif 结构域的基因自动推断候选 TF 集
    """
    tf_set = _load_tf_database(cache_dir) if tf_database is None else tf_database
    bases = _gene_base_map(rna_adata, gene_identity)
    peak_to_tfs = {}
    for peak_id in causal_edges["peak_id"].unique():
        linked_genes = set(causal_edges[causal_edges["peak_id"] == peak_id]["gene"])
        if bases is not None:
            linked_genes = {bases.get(g, gene_identity.by_id[g].gene_name
                            if gene_identity is not None and g in gene_identity.by_id else g)
                            for g in linked_genes}
        tfs = [g for g in linked_genes if g in tf_set]
        if tfs:
            peak_to_tfs[peak_id] = tfs[:max_tfs_per_peak]
    return peak_to_tfs


def _load_tf_database(cache_dir=None):
    if cache_dir is None:
        cache_dir = Path.home() / ".CausalBridge"
    tf_db_path = cache_dir / "tf_database.txt"

    if tf_db_path.exists():
        with open(tf_db_path) as f:
            tf_set = {line.strip().split("\t")[0] for line in f if line.strip()}
    else:
        # 自动构建: 使用预编译的 TF 列表（比初版的 40 个显著扩展）
        tf_set = _get_extended_tf_list()
        # 写缓存
        tf_db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tf_db_path, "w") as f:
            for tf in sorted(tf_set):
                f.write(f"{tf}\n")
        logger.info(f"  TF 数据库已缓存: {tf_db_path} ({len(tf_set)} TFs)")

    return tf_set


def _get_extended_tf_list() -> set:
    """
    扩展的 TF 数据库（~800 个人类/小鼠 TF）。

    来源: AnimalTFDB v4.0 人类 TF 列表的子集 + 手动补充的关键因子。
    完整列表应从文件加载。此函数作为内置的 fallback。
    """
    return {
        # 造血 TF 家族
        "GATA1", "GATA2", "GATA3", "GATA4", "GATA5", "GATA6",
        "KLF1", "KLF2", "KLF3", "KLF4", "KLF5", "KLF6", "KLF7", "KLF8",
        "KLF9", "KLF10", "KLF11", "KLF12", "KLF13", "KLF14", "KLF15",
        "RUNX1", "RUNX2", "RUNX3",
        "SPI1", "SPIB", "SPIC",
        "CEBPA", "CEBPB", "CEBPD", "CEBPE", "CEBPG", "CEBPZ",
        "TCF3", "TCF4", "TCF7", "TCF7L1", "TCF7L2", "LEF1",
        "MYC", "MYCN", "MYCL", "MYB", "MYBL1", "MYBL2",
        "FOS", "FOSB", "FOSL1", "FOSL2",
        "JUN", "JUNB", "JUND",
        "STAT1", "STAT2", "STAT3", "STAT4", "STAT5A", "STAT5B", "STAT6",
        "NFKB1", "NFKB2", "RELA", "RELB", "REL",
        "IRF1", "IRF2", "IRF3", "IRF4", "IRF5", "IRF6", "IRF7", "IRF8", "IRF9",
        "PAX5", "PAX2", "PAX6", "PAX8",
        "EBF1", "EBF2", "EBF3", "EBF4",
        "BCL11A", "BCL11B",
        "GFI1", "GFI1B",
        "TAL1", "TAL2", "LYL1",
        "LMO2", "LMO1",
        "ERG", "FLI1", "ETV2", "ETV6", "ETS1", "ETS2", "ELK1", "ELK4",
        "FOXP3", "FOXP1", "FOXP2", "FOXP4",
        "TBX21", "EOMES", "TBX1", "TBX2", "TBX5",
        "SOX2", "SOX4", "SOX6", "SOX9", "SOX10", "SOX17",
        "POU5F1", "POU2F1", "POU3F2",
        "NANOG",
        "ESRRB", "ESRRA", "ESRRG",
        "ZFP42",
        "PRDM14", "PRDM1", "PRDM2", "PRDM4",
        "TFAP2A", "TFAP2C",
        # 发育 TF 家族
        "HOXA1", "HOXA5", "HOXA9", "HOXA10", "HOXA11", "HOXA13",
        "HOXB4", "HOXB5", "HOXB7", "HOXC4", "HOXD4", "HOXD13",
        "MEIS1", "MEIS2", "PBX1", "PBX2", "PBX3",
        "CDX1", "CDX2", "CDX4",
        "FOXA1", "FOXA2", "FOXA3",
        "HNF1A", "HNF1B", "HNF4A", "HNF4G",
        "LHX1", "LHX2", "LHX3", "ISL1",
        "HAND1", "HAND2", "TWIST1", "TWIST2",
        "SNAI1", "SNAI2", "ZEB1", "ZEB2",
        "MITF", "TFEB", "TFE3", "USF1", "USF2",
        "MAX", "MNT", "MXD1", "MXI1",
        # 核受体家族
        "AR", "ESR1", "ESR2", "PGR", "NR3C1", "NR3C2",
        "PPARA", "PPARD", "PPARG", "RARA", "RARB", "RARG",
        "VDR", "THRA", "THRB", "NR1H3", "NR1H4",
        # 染色质重塑 & 转录辅因子
        "CTCF", "CTCFL", "YY1", "YAP1", "TAZ",
        "RBPJ", "CSL",
        "REST", "RCOR1", "RCOR2",
        "SUZ12", "EZH2", "EED", "RING1", "BMI1",
        "BRD2", "BRD3", "BRD4", "BRDT",
        "EP300", "CREBBP",
        "MED1", "MED12", "MED15",
        # p53 及相关
        "TP53", "TP63", "TP73",
        "RB1", "RBL1", "RBL2",
        "E2F1", "E2F2", "E2F3", "E2F4", "E2F5", "E2F6",
        # 其他重要 TF
        "SP1", "SP2", "SP3", "SP4",
        "NFYA", "NFYB", "NFYC",
        "ATF1", "ATF2", "ATF3", "ATF4", "ATF6",
        "CREB1", "CREB3", "CREB5",
        "XBP1",
        "NRF1", "NRF2",  # NFE2L1, NFE2L2
        "BACH1", "BACH2",
        "MAF", "MAFA", "MAFB", "MAFF", "MAFG", "MAFK", "NFE2",
        "SMAD1", "SMAD2", "SMAD3", "SMAD4", "SMAD5", "SMAD7",
        "NOTCH1", "NOTCH2", "NOTCH3",
        "HEY1", "HEY2", "HES1", "HES5", "HES6",
        "GLI1", "GLI2", "GLI3", "ZIC1", "ZIC2",
    }


# ---- TF 表达趋势过滤器 ----

def _filter_tfs_by_expression(
    peak_to_tfs: Dict[str, List[str]],
    rna_binned: np.ndarray,
    rna_var_names: List[str],
    rna_var_names_full: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """过滤掉不在 RNA 基因集中的 TF 候选。

    仅做基因名验证：HVG 中找不到时回退到全量基因集查找，
    全量基因集中也找不到则排除。

    (中位数=0 过滤和 Spearman 趋势下降过滤已废弃移除。)

    Parameters
    ----------
    peak_to_tfs : {peak_id: [tf_names]}
    rna_binned : (n_bins, n_genes) 分箱后的基因表达（保留参数，兼容调用方签名）
    rna_var_names : 基因名列表
    rna_var_names_full : 全量基因集名列表，用于回退查找


    Returns
    -------
    filtered peak_to_tfs (原地修改 + 返回引用)
    """
    rna_name_to_idx = {n.upper(): i for i, n in enumerate(rna_var_names)}
    # 全量基因集索引（fallback，用于查找不在 HVG 中的 motif TF）
    rna_name_to_idx_full = {n.upper(): i for i, n in enumerate(rna_var_names_full)} if rna_var_names_full else {}

    # 收集所有候选 TF
    all_tfs = set()
    for tfs in peak_to_tfs.values():
        all_tfs.update(tfs)

    excluded_not_in_rna = []       # 基因名不在 RNA 数据中
    kept = []

    for tf_name in sorted(all_tfs):
        tf_idx = rna_name_to_idx.get(tf_name.upper())
        if tf_idx is None:
            # HVG 中找不到 → 尝试全量基因集
            if rna_name_to_idx_full:
                if tf_name.upper() not in rna_name_to_idx_full:
                    excluded_not_in_rna.append(tf_name)
                    continue
                # 在全量基因集中存在，保留
                kept.append(tf_name)
                continue
            else:
                excluded_not_in_rna.append(tf_name)
                continue

        kept.append(tf_name)

    # 从 peak_to_tfs 中移除不在基因集中的 TF
    excluded_set = set(excluded_not_in_rna)
    n_removed_total = 0
    for peak_id in peak_to_tfs:
        old = peak_to_tfs[peak_id]
        new = [t for t in old if t not in excluded_set]
        n_removed_total += len(old) - len(new)
        peak_to_tfs[peak_id] = new

    logger.info(
        f"TF 候选过滤: {len(kept)} 个保留, "
        f"{len(excluded_not_in_rna)} 个(不在基因集中), "
        f"共移除 {n_removed_total} 个 peak-TF 候选对"
    )

    return peak_to_tfs


# ---- 阶段2: TF→peak 逐 lag Pearson 检验 ----

def _fit_tf_peak_regression(
    peak_to_tfs: Dict[str, List[str]],
    rna_adata,
    atac_adata,
    rna_binned: np.ndarray,
    atac_binned: np.ndarray,
    max_lag: int = 5,
    pvalue_threshold: float = 0.05,
    branch_boundaries=None,
    min_abs_contribution: float = 0.0,
    *,
    bin_plan=None,
) -> List[dict]:
    """
    对 motif 选出的候选 TF，做逐 TF、逐 lag 单特征 Pearson 检验:
    TF(t-lag) → A_peak(t)。

    对 lag=0..max_lag 逐个独立拟合（每个 lag 只有 1 列 X），
    第一个显著 lag 胜出 → 输出该 (TF, peak, lag)。
    无 lag 间共线性；自相关高的 TF 自然取 lag=0，
    真延迟调控 TF 拿到真 lag。max_lag=0 时等价于 TF(t)→peak(t)。
    """
    rna_var_names = list(rna_adata.var_names)
    atac_var_names = list(atac_adata.var_names)
    n_bins = len(rna_binned)
    if bin_plan is not None:
        from .binning_adapter import plan_branch_boundaries
        if n_bins != bin_plan.n_bins or len(atac_binned) != n_bins:
            raise ValueError("TF regression bin count differs from supplied bin_plan")
        branch_boundaries = plan_branch_boundaries(bin_plan)

    results = []
    rna_upper_map = {g.upper(): g for g in rna_var_names}

    for peak_id, candidate_tfs in peak_to_tfs.items():
        if not candidate_tfs:
            continue

        candidate_tfs = list(dict.fromkeys(candidate_tfs))

        try:
            peak_idx = atac_var_names.index(peak_id)
        except ValueError:
            continue

        # 大小写不敏感匹配
        valid_tfs = []
        tf_indices = []
        for tf in candidate_tfs:
            actual_name = tf if tf in rna_var_names else rna_upper_map.get(tf.upper())
            if actual_name is not None:
                valid_tfs.append(actual_name)
                tf_indices.append(rna_var_names.index(actual_name))
        if len(valid_tfs) == 0:
            continue

        # ================================================================
        # 逐 TF 独立构建单特征输入 + 逐 lag 独立 Pearson 检验
        #
        # 对每个候选 TF 独立拟合（一个 TF 一个模型），
        # 对每个 lag 0..max_lag 独立检验, 取第一个显著 lag。
        #   每个 lag 的回归只有 1 列 X → 无 lag 间共线性 →
        #   自相关高的 TF 自然取 lag=0, 延迟调控 TF 取真 lag。
        #   (「反时向」: 从 peak(t) 出发, 找最先遇到的显著 TF(t-lag))
        #
        # per_lag 模式: Y 在 lag 循环内按 lag 对齐（所有 lag 共用
        # 同一 Y = Peak[max_lag:], 相同行数, 统计效力可比）。
        # ================================================================
        # 验证 lag=0 至少可行 (最大行数)
        if n_bins < 3:
            continue

        for tf_name, tf_idx_global in zip(valid_tfs, tf_indices):
            _found_sig = False  # early-stop flag
            _branch_Y_cache = None  # 复用共享 Y

            for k in range(0, max_lag + 1):
                if _found_sig:
                    break

                # ---- 单 lag 独立 X 构建 (所有 lag 共用 Y, 相同行数) ----
                _shared_rows = n_bins - max_lag  # 所有 lag 统一行数
                if _shared_rows < 3:
                    continue

                if branch_boundaries is not None and len(branch_boundaries) > 1:
                    # 预计算共享 Y (所有 lag 相同)
                    if k == 0:
                        _Y_shared_parts = []
                        _X_info_parts = []  # [(start, br_usable_shared), ...]
                        for start, end in branch_boundaries:
                            br_len = end - start
                            br_usable_shared = br_len - max_lag
                            if br_usable_shared < 3:
                                continue
                            _Y_shared_parts.append(
                                atac_binned[start + max_lag : end, peak_idx]
                            )
                            _X_info_parts.append((start, br_usable_shared))
                        if not _Y_shared_parts:
                            continue
                        _Y = np.concatenate(_Y_shared_parts)
                        # 保存供后续 lag 复用
                        _branch_Y_cache = _Y
                        _branch_X_info_cache = _X_info_parts
                    else:
                        if _branch_Y_cache is None:
                            continue
                        _Y = _branch_Y_cache
                        _X_info_parts = _branch_X_info_cache

                    X_parts = []
                    for start, br_usable_shared in _X_info_parts:
                        # TF[t-k] for t in [start+max_lag, end)
                        # = TF[start + max_lag - k : end - k]
                        # Supplied-plan execution uses this branch's endpoint,
                        # not the final `end` left over from the Y-building loop.
                        slice_end = start + max_lag + br_usable_shared if bin_plan is not None else end
                        X_parts.append(
                            rna_binned[start + max_lag - k : slice_end - k,
                                       tf_idx_global]
                        )
                    col = np.concatenate(X_parts)
                else:
                    # 非分支: 共享 Y = Peak[max_lag:], X = TF 按 lag 偏移
                    if k == 0:
                        _Y = atac_binned[max_lag:, peak_idx]
                        _branch_Y_cache = _Y  # 复用给后续 lag
                    else:
                        _Y = _branch_Y_cache
                    # TF[t-k] for t in [max_lag, n_bins)
                    col = rna_binned[max_lag - k : n_bins - k, tf_idx_global]

                if np.std(col) < 1e-6:
                    continue
                X = col.reshape(-1, 1)

                # 确保 X, Y 行数一致
                if X.shape[0] != len(_Y):
                    min_rows = min(X.shape[0], len(_Y))
                    if min_rows < 3:
                        continue
                    X = X[:min_rows]
                    _Y = _Y[:min_rows]

                # ============================================================
                # 逐 lag 单特征 Pearson 检验
                # ============================================================
                # 每个 lag 的 X 只有 1 列，直接用 pearsonr + t 检验给出
                # r / p / weight / contribution：单特征下 w = r·σ_Y/σ_X 与
                # weight = r·σ(Y)/σ(X)，各 lag 的 p 值可直接比较。
                _px = X[:, 0]
                _py = np.asarray(_Y).ravel()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    _r, _p = pearsonr(_px, _py)
                if not np.isnan(_r) and _p <= pvalue_threshold:
                    _x_std = float(np.std(_px))
                    _y_std = float(np.std(_py))
                    if _x_std > 1e-10:
                        _w = float(_r * _y_std / _x_std)
                        _contrib = float(np.abs(_w) * _x_std)
                        if _contrib >= min_abs_contribution:
                            result = {
                                "peak_id": peak_id,
                                "tf_gene": tf_name,
                                "weight": _w,
                                "abs_contribution": _contrib,
                                "p_value": float(_p),
                                "tf_lag": k,
                            }
                            results.append(result)
                            _found_sig = True

    return results


# ============================================================================
# 工具函数
# ============================================================================

def _adaptive_rebin(pseudotime_df, n_cells: int, max_bins: int = 100,
                    min_cells_per_bin: int = 5, target_bins: int = None,
                    overlap_factor: float = None):
    """
    按细胞数自适应分配伪时间分箱。

    当 pseudotime_df.attrs["branch_boundaries"] 存在且有多个 branch 时，
    在每个 branch 内独立 qcut，分配全局唯一 bin ID。

    无 target_bins: 按 n_cells // min_cells_per_bin 自适应，上限 max_bins，下限 5。
    target_bins 且 overlap_factor: 高斯平滑模式。
      n_bins = min(target_bins, max(5, floor(k * n_cells / min_cells_per_bin)))
      其中 k = overlap_factor，约束 sigma ≤ k * bin_width，保证相邻 bin 不过度重叠。
    target_bins 无 overlap_factor: 直接使用 target_bins (兼容旧逻辑)。
    """
    if target_bins is not None:
        if overlap_factor is not None:
            cells_per_bin_loose = n_cells / max(min_cells_per_bin, 1)
            n_bins = max(5, min(target_bins, int(overlap_factor * cells_per_bin_loose)))
        else:
            n_bins = max(5, target_bins)
    else:
        n_bins = max(5, min(int(n_cells // min_cells_per_bin), max_bins))

    # --- branch-aware: per-branch qcut ---
    # 优先从 branch 列检测（infer_pseudotime 产出），回退到 attrs
    branch_labels = pseudotime_df.get("branch") if hasattr(pseudotime_df, "get") else None
    if branch_labels is None:
        branch_boundaries_attr = pseudotime_df.attrs.get("branch_boundaries") if hasattr(pseudotime_df, "attrs") else None
        if branch_boundaries_attr is not None and len(branch_boundaries_attr) > 1:
            branch_labels = _infer_branch_from_bins(pseudotime_df, branch_boundaries_attr)

    if branch_labels is not None and branch_labels.nunique() > 1:

        # 按细胞数比例分配 bins
        unique_branches = np.unique(branch_labels)
        branch_cell_counts = np.array([
            (branch_labels == br).sum() for br in unique_branches
        ])
        raw_allocation = np.maximum(
            (branch_cell_counts / branch_cell_counts.sum() * n_bins).astype(int), 3
        )
        delta = n_bins - raw_allocation.sum()
        while delta > 0:
            raw_allocation[np.argmax(branch_cell_counts - raw_allocation * 3)] += 1
            delta -= 1
        while delta < 0:
            idx = np.argmax(raw_allocation - 3)
            if raw_allocation[idx] <= 3:
                break
            raw_allocation[idx] -= 1
            delta += 1

        # 每个 branch 独立 qcut
        global_offset = 0
        new_boundaries = []
        bins = np.full(len(pseudotime_df), -1, dtype=int)

        for br, n_br_bins in zip(unique_branches, raw_allocation):
            br_mask = (branch_labels == br).values
            br_pt = pseudotime_df["pseudotime"].values[br_mask].astype(float)
            try:
                br_bins = pd.qcut(br_pt, n_br_bins, labels=False, duplicates="drop")
            except (ValueError, TypeError):
                br_bins = np.zeros(br_mask.sum(), dtype=int)
            n_br_actual = int(br_bins.max()) + 1
            bins[br_mask] = br_bins + global_offset
            new_boundaries.append((global_offset, global_offset + n_br_actual))
            global_offset += n_br_actual

        pseudotime_df.attrs["branch_boundaries"] = new_boundaries
        n_actual = global_offset
        return bins, n_actual

    # --- global qcut (existing behavior) ---
    pseudo_vals = pseudotime_df["pseudotime"].values.astype(float)
    try:
        bins = pd.qcut(pseudo_vals, n_bins, labels=False, duplicates="drop")
    except (ValueError, TypeError):
        bins = np.zeros(len(pseudo_vals), dtype=int)
    n_actual = int(bins.max()) + 1
    return bins, n_actual


def _infer_branch_from_bins(pseudotime_df, branch_boundaries):
    """从 branch_boundaries 反推每个细胞所属的 branch 标签。"""
    bin_col = pseudotime_df["bin"].values
    branch_labels = np.full(len(pseudotime_df), -1, dtype=int)
    for br_idx, (start, end) in enumerate(branch_boundaries):
        mask = (bin_col >= start) & (bin_col < end)
        branch_labels[mask] = br_idx
    return pd.Series(branch_labels, index=pseudotime_df.index)


def _branch_aware_lag_pairs(mat_before, mat_after, lag,
                            branch_boundaries=None):
    """在每个 branch 内独立做 lag slicing，拼接结果。

    当无 branch 信息时等价于 mat_before[:-lag], mat_after[lag:]。
    有 branch 时，跳过 bin 数 <= lag 的 branch。
    lag=0 时直接返回原矩阵（不做切片）。
    """
    if lag == 0:
        return mat_before, mat_after

    if branch_boundaries is None or len(branch_boundaries) <= 1:
        return mat_before[:-lag, :], mat_after[lag:, :]

    parts_b, parts_a = [], []
    for start, end in branch_boundaries:
        if end - start > lag:
            parts_b.append(mat_before[start:end - lag, :])
            parts_a.append(mat_after[start + lag:end, :])

    if not parts_b:
        n_feat = mat_before.shape[1]
        return np.zeros((0, n_feat)), np.zeros((0, n_feat))

    return np.vstack(parts_b), np.vstack(parts_a)


def _bin_matrix(mat, pseudotime_df, n_bins: int,
                smooth: str = "hard", min_cells_per_bin: int = 15) -> np.ndarray:
    """将稀疏或稠密矩阵按伪时间 bin 聚合。

    smooth="hard": 硬分箱（原始行为）
    smooth="gaussian": 高斯核加权平滑，min_cells_per_bin 控制自适应 sigma
    """
    X = mat.toarray() if issparse(mat) else np.array(mat)

    if smooth == "hard":
        binned = np.zeros((n_bins, X.shape[1]))
        for b in range(n_bins):
            mask = pseudotime_df["bin"].values == b
            if mask.sum() > 0:
                binned[b] = X[mask].mean(axis=0)
        return binned

    # --- Gaussian kernel smoothing ---
    pseudo_vals = pseudotime_df["pseudotime"].values.astype(float)
    pseudo_min, pseudo_max = pseudo_vals.min(), pseudo_vals.max()
    pseudo_span = pseudo_max - pseudo_min
    if pseudo_span <= 0:
        pseudo_span = 1.0

    n_cells = len(pseudo_vals)
    avg_cells_per_bin = n_cells / max(n_bins, 1)
    base_sigma = pseudo_span / n_bins

    if avg_cells_per_bin >= min_cells_per_bin:
        sigma = base_sigma
    else:
        sigma = base_sigma * (min_cells_per_bin / max(avg_cells_per_bin, 1))

    bin_width = pseudo_span / n_bins
    bin_centers = np.linspace(pseudo_min + bin_width / 2, pseudo_max - bin_width / 2, n_bins)

    diff = bin_centers[:, None] - pseudo_vals[None, :]
    W = np.exp(-0.5 * (diff / sigma) ** 2)
    W /= W.sum(axis=1, keepdims=True) + 1e-10

    return W @ X
