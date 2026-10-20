"""Bounded process execution over shared read-only binned screening inputs."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import logging
import multiprocessing
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import scipy.stats as stats
from threadpoolctl import threadpool_limits

from .terminal_progress import TerminalProgress

logger = logging.getLogger(__name__)
_STATE = None
_THREAD_LIMIT = None


def screening_workers(config, key):
    execution = config.get('execution', {})
    if not isinstance(execution, dict):
        raise ValueError('execution must be a mapping')
    value = execution.get(key, 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'execution.{key} must be a positive integer (process workers)')
    return value


def _initialize(state, paths):
    global _STATE, _THREAD_LIMIT
    _STATE = dict(state)
    for key, path in paths.items():
        _STATE[key] = np.load(path, mmap_mode='r', allow_pickle=False)
    if _STATE.get('kind') == 'granger':
        _STATE['_atac_lag_mats'] = ([_STATE['_atac_lag']] if '_atac_lag' in _STATE else None)
    _THREAD_LIMIT = threadpool_limits(limits=1)
    # Parent owns progress. Warnings/errors remain visible from kernels.
    logging.getLogger('CausalBridge.kinetics').setLevel(logging.WARNING)


def _execute(job):
    index, candidates = job
    function = _granger_batch if _STATE['kind'] == 'granger' else _local_batch
    return index, len(candidates), function(_STATE, candidates)


@contextmanager
def _worker_environment():
    keys = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update({key: '1' for key in keys})
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _screen(candidates, state, workers, desc, chunk_size):
    """At most 2*workers tasks in flight; compact output indexed in input order."""
    if not candidates:
        return []
    workers = min(workers, (len(candidates)+chunk_size-1)//chunk_size)
    chunks = [(i, candidates[start:start+chunk_size])
              for i, start in enumerate(range(0, len(candidates), chunk_size))]
    logger.info('%s: candidates=%d workers=%d', desc, len(candidates), workers)
    output = {}
    completed = 0
    with TerminalProgress(len(candidates), desc, logger=logger) as progress:
        if workers == 1:
            function = _granger_batch if state['kind'] == 'granger' else _local_batch
            for index, chunk in chunks:
                output[index] = function(state, chunk)
                completed += len(chunk)
                progress.update(completed)
        else:
            # Only numeric binned matrices are mapped; no AnnData/count matrix
            # travels with a task or is copied once per worker.
            with tempfile.TemporaryDirectory(prefix='causalbridge-screening-') as directory:
                paths, metadata = {}, dict(state)
                for key, value in state.items():
                    if isinstance(value, np.ndarray) and value.dtype.kind not in 'OU':
                        path = Path(directory) / (key+'.npy')
                        np.save(path, value, allow_pickle=False)
                        paths[key] = str(path)
                        metadata.pop(key)
                if state['kind'] == 'granger':
                    metadata.pop('_atac_lag_mats', None)
                with _worker_environment(), ProcessPoolExecutor(max_workers=workers,
                        mp_context=multiprocessing.get_context('spawn'),
                        initializer=_initialize, initargs=(metadata, paths)) as executor:
                    iterator = iter(chunks)
                    pending = {}
                    def submit():
                        job = next(iterator, None)
                        if job is not None:
                            pending[executor.submit(_execute, job)] = job[0]
                    for _ in range(2*workers):
                        submit()
                    try:
                        while pending:
                            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                            for future in ready:
                                pending.pop(future)
                                index, count, rows = future.result()
                                output[index] = rows
                                completed += count
                                progress.update(completed)
                                submit()
                    except BaseException:
                        for future in pending:
                            future.cancel()
                        raise
    logger.info('%s complete: %d/%d candidates', desc, completed, len(candidates))
    return [row for index in range(len(chunks)) for row in output[index]]


def granger_candidates(candidates, state, *, workers=1):
    state = dict(state, kind='granger')
    if state['_atac_lag_mats'] is not None:
        state['_atac_lag'] = state['_atac_lag_mats'][0]
    return _screen(candidates, state, workers, 'Step 4 Granger', 512)


def local_lag_candidates(peak_to_tfs, rna_adata, atac_adata, rna_binned, atac_binned,
        max_lag=5, pvalue_threshold=.05, branch_boundaries=None, min_abs_contribution=0.,
        *, bin_plan=None, local_lag_window_rows=12, workers=1):
    from . import kinetics
    # Canonical candidates determine work counts; serial execution retains the
    # original mapping boundary (including duplicates and audit-only entries).
    genes, peaks = list(rna_adata.var_names), list(atac_adata.var_names)
    gene_set, peak_set = set(genes), set(peaks)
    upper = {gene.upper(): gene for gene in genes}
    candidates = []
    for peak, names in peak_to_tfs.items():
        if peak in peak_set:
            canonical = dict.fromkeys(name if name in gene_set else upper.get(name.upper()) for name in names)
            candidates.extend((peak, gene) for gene in canonical if gene is not None)
    kwargs = dict(max_lag=max_lag, pvalue_threshold=pvalue_threshold,
                  branch_boundaries=branch_boundaries, min_abs_contribution=min_abs_contribution,
                  bin_plan=bin_plan, local_lag_window_rows=local_lag_window_rows)
    if not candidates:
        return kinetics._fit_tf_peak_regression(peak_to_tfs, rna_adata,
            atac_adata, rna_binned, atac_binned, **kwargs)
    quiet = logging.getLogger('CausalBridge.kinetics')
    if workers == 1:
        with TerminalProgress(len(candidates), 'Step 6 局部 lag', logger=quiet) as progress:
            class ProgressFilter(logging.Filter):
                def filter(self, record):
                    if str(record.msg).startswith('TF local lag '):
                        if 'processed_pairs=' in str(record.msg):
                            progress.update(min(int(record.args[0]), len(candidates)))
                        return False
                    return True
            progress_filter = ProgressFilter()
            level = quiet.level
            quiet.addFilter(progress_filter)
            quiet.setLevel(logging.INFO)
            try:
                rows = kinetics._fit_tf_peak_regression(peak_to_tfs, rna_adata,
                    atac_adata, rna_binned, atac_binned, **kwargs)
                progress.update(len(candidates))
                return rows
            finally:
                quiet.removeFilter(progress_filter)
                quiet.setLevel(level)
    # Validate even an empty parallel job, using the unchanged numerical core.
    level = quiet.level
    quiet.setLevel(logging.WARNING)
    try:
        kinetics._local_tf_peak_routes({}, rna_adata, atac_adata, rna_binned,
            atac_binned, max_lag, pvalue_threshold, branch_boundaries,
            min_abs_contribution, bin_plan, local_lag_window_rows)
    finally:
        quiet.setLevel(level)
    return _screen(candidates, dict(kind='local', genes=genes, peaks=peaks,
                   rna=rna_binned, atac=atac_binned, kwargs=kwargs), workers, 'Step 6 局部 lag', 128)


def _local_batch(state, candidates):
    from . import kinetics
    mapping = {}
    for peak, gene in candidates:
        mapping.setdefault(peak, []).append(gene)
    quiet = logging.getLogger('CausalBridge.kinetics')
    level = quiet.level
    quiet.setLevel(logging.WARNING)
    try:
        return kinetics._fit_tf_peak_regression(mapping, SimpleNamespace(var_names=state['genes']),
            SimpleNamespace(var_names=state['peaks']), state['rna'], state['atac'], **state['kwargs'])
    finally:
        quiet.setLevel(level)
def _granger_batch(state, candidates):
    """Original per-pair arithmetic; FDR remains once in the parent."""
    atac_name_to_idx = state['atac_name_to_idx']
    rna_name_to_idx = state['rna_name_to_idx']
    atac_chr_arr = state['atac_chr_arr']
    atac_start_arr = state['atac_start_arr']
    atac_end_arr = state['atac_end_arr']
    rna_chr_arr = state['rna_chr_arr']
    rna_tss_arr = state['rna_tss_arr']
    _rna_t = state['_rna_t']
    _rna_t1 = state['_rna_t1']
    _atac_lag_mats = state['_atac_lag_mats']
    _atac_current = state['_atac_current']
    n_valid_pairs = state['n_valid_pairs']
    ablation_mode = state['ablation_mode']
    ones = np.ones(n_valid_pairs)
    gene_ssr_cache = {}
    results = []
    for i, (peak_id, gene) in enumerate(candidates):
        peak_idx = atac_name_to_idx.get(peak_id)
        gene_idx = rna_name_to_idx.get(gene)
        if peak_idx is None or gene_idx is None:
            continue

        Y_t = _rna_t[:, gene_idx]
        Y_lag = _rna_t1[:, gene_idx] if _rna_t1 is not None else None

        if np.var(Y_t) < 1e-6:
            continue

        # Reduced model: Y(t) ~ Y(t-1), cache results once per gene
        if ablation_mode:
            # Ablation: Reduced = Y(t) ~ 1 (intercept only, no autoregression)
            ssr_reduced = float(np.sum((Y_t - np.mean(Y_t)) ** 2))
            if ssr_reduced < 1e-10:
                continue
        else:
            if gene_idx in gene_ssr_cache:
                ssr_reduced = gene_ssr_cache[gene_idx]
            else:
                Y_lag_const = np.column_stack([ones, Y_lag])
                _, residuals_reduced, _, _ = np.linalg.lstsq(
                    Y_lag_const, Y_t, rcond=None
                )
                # lstsq returns sums of squared residuals, not residual vectors.
                # Empty arrays (rank-deficient fits) retain the existing skip policy.
                ssr_reduced = float(np.sum(residuals_reduced))
                if ssr_reduced < 1e-10:
                    ssr_reduced = 0.0
                gene_ssr_cache[gene_idx] = ssr_reduced

        if ssr_reduced < 1e-10:
            continue

        # Full model
        if ablation_mode:
            # Ablation: Y(t) ~ X(t) (contemporaneous peak, no lag)
            x_col = _atac_current[:, peak_idx]
            if np.var(x_col) < 1e-6:
                continue
            X_const = np.column_stack([ones, x_col])
            coeffs, residuals_full, _, _ = np.linalg.lstsq(
                X_const, Y_t, rcond=None
            )
            atac_coef = float(coeffs[1])  # β₁: ATAC_t → RNA_t
            # NumPy has already squared and summed the residual vector.
            ssr_full = float(np.sum(residuals_full))
            df_full = n_valid_pairs - 2
        else:
            # Standard Granger: Y(t) ~ Y(t-1) + X(t-1)
            x_col = _atac_lag_mats[0][:, peak_idx]
            if np.var(x_col) < 1e-6:
                continue
            YX_lag_const = np.column_stack([ones, Y_lag, x_col])
            coeffs, residuals_full, _, _ = np.linalg.lstsq(
                YX_lag_const, Y_t, rcond=None
            )
            atac_coef = float(coeffs[2])  # β₂: ATAC_{t-1} → RNA_t
            # NumPy has already squared and summed the residual vector.
            ssr_full = float(np.sum(residuals_full))
            df_full = n_valid_pairs - 3

        if ssr_full < 1e-10 or df_full <= 0:
            continue

        f_num = ssr_reduced - ssr_full
        f_den = ssr_full / df_full
        if f_den <= 0:
            continue

        f_stat = f_num / f_den
        p_value = stats.f.sf(f_stat, 1, df_full)
        delta_r2 = 1 - ssr_full / ssr_reduced

        # Pearson r: X vs Y — direction (standard=lagged, ablation=contemporaneous)
        pearson_r = float(np.corrcoef(x_col, Y_t)[0, 1]) if np.std(x_col) > 1e-6 else 0.0

        results.append({
            "peak_id": peak_id,
            "gene": gene,
            "peak_chr": atac_chr_arr[peak_idx],
            "peak_start": int(atac_start_arr[peak_idx]),
            "peak_end": int(atac_end_arr[peak_idx]),
            "gene_chr": rna_chr_arr[gene_idx],
            "gene_tss": int(rna_tss_arr[gene_idx]),
            "F_statistic": f_stat,
            "p_value": p_value,
            "delta_r2": delta_r2,
            "atac_coef": atac_coef,
            "pearson_r": pearson_r,
        })

    return results
