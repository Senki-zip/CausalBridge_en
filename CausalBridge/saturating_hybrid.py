"""Opt-in numerical core only; no pipeline dispatch, IO, or real-data fitting.

All fits use all supplied WT rows. Support/plateau thresholds are experimental
assumptions, not validated biological limits. Shared cell weights are referenced,
never duplicated into a serialized target. Group restart checks are NOT validation.
"""
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any
import hashlib
import json
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist

SCHEMA = "saturating_hybrid_target_v1"
CONTINUOUS_SCHEMA = "saturating_hybrid_target_v2"
CONTINUOUS_SINGLE_SCHEMA = "saturating_hybrid_single_v2"
CONTINUOUS_GROUP_SCHEMA = "saturating_hybrid_group_v2"
INPUT_UNITS = ("rna_log1p_cp10k", "raw_atac", "normalized_atac_log1p")
OUTPUT_UNITS = ("log1p_rna_log1p_cp10k", "raw_atac", "normalized_atac_log1p")


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _plain(value) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise ValueError("metadata keys must be strings")
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _number(value, "metadata")
    raise ValueError(f"non-primitive metadata: {type(value).__name__}")


def _freeze(value):
    value = _plain(value)
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _array(value, name, ndim=None, integer=False) -> np.ndarray:
    original = np.asarray(value)
    if original.dtype.kind not in "iuf" or (integer and original.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a numeric {'integer ' if integer else ''}array")
    array = np.asarray(value, dtype=np.int64 if integer else np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    # Immutable byte buffer also prevents setflags(write=True) on exported arrays.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _bounds(value):
    if len(value) != 2:
        raise ValueError("bounds must contain L,U")
    lower, upper = (_number(v, "bound") for v in value)
    if lower < 0 or upper < lower:
        raise ValueError("physical bounds require 0 <= L <= U")
    return lower, upper


def _keys(value):
    result = []
    for key in value:
        if not isinstance(key, (tuple, list)) or len(key) != 2 or not isinstance(key[0], str) or not key[0]:
            raise ValueError("input keys must be (nonempty input_id, lag) pairs")
        result.append((key[0], "local" if key[1] == "local" else _integer(key[1], "lag")))
    if not result or len(set(result)) != len(result):
        raise ValueError("input keys must be nonempty and unique")
    return tuple(result)


def _routes(keys, rows, source, provenance, route_supported):
    """Validate causal branch-local routes, including explicit unsupported slots."""
    local = any(k[1] == "local" for k in keys)
    if not local:
        mask = np.ones(source.shape, dtype=bool)
        if route_supported is not None and not np.array_equal(route_supported, mask):
            raise ValueError("fixed routes must all be supported")
        if (source < 0).any() or not np.array_equal(source, rows[:,None]-np.array([k[1] for k in keys])):
            raise ValueError("source bins must match target rows minus input lags")
    else:
        if provenance.get("relation_type") != "saturating_hybrid_tf_peak" or any(k[1] != "local" for k in keys):
            raise ValueError("variable source keys are reserved for local TF routes")
        if len({k[0] for k in keys}) != len(keys):
            raise ValueError("local TF inputs must be unique")
        mask = np.asarray(route_supported)
        if mask.dtype != np.bool_ or mask.shape != source.shape:
            raise ValueError("local route_supported must be an aligned boolean matrix")
        maxlag = _integer(provenance.get("local_lag_max"), "local_lag_max")
        lag = rows[:,None]-source
        if (source[~mask] != -1).any() or (source[mask] < 0).any() or (lag[mask] < 0).any() or (lag[mask] > maxlag).any():
            raise ValueError("invalid supported local source bins")
    mask = np.frombuffer(mask.tobytes(),dtype=bool).reshape(mask.shape)
    return mask


def _wt_scores(predictions, rows, branches, source, supported, local):
    adjacent = (np.diff(rows)==1)&(np.array(branches[1:])==np.array(branches[:-1]))
    if not local:
        adjacent &= np.all(np.diff(source,axis=0)==1,axis=1)
    if not adjacent.any():
        return np.zeros(predictions.shape[1]), adjacent
    changes = np.abs(np.diff(predictions, axis=0))
    if local:
        changes = np.where(supported[1:] & supported[:-1], changes, 0.)
    return np.mean(changes[adjacent], axis=0), adjacent


def _single_predictions(singles, X, supported, reference_means):
    """Never evaluate a response at a masked placeholder input."""
    out = np.empty(X.shape, dtype=float)
    for i, single in enumerate(singles):
        out[:,i] = reference_means[i]
        mask = supported[:,i]
        if mask.any():
            out[mask,i] = single.predict(X[mask,i])
    return out


def _branches(value, n):
    result = tuple(value)
    if len(result) != n or not all(isinstance(v, str) and v for v in result):
        raise ValueError("branch_ids must be aligned nonempty strings")
    return result


def _unit_domain(X, unit, tolerance=0.):
    if (X < -tolerance).any() or (unit == "raw_atac" and (X > 1+tolerance).any()):
        raise ValueError(f"inputs outside physical {unit} domain")


@dataclass(frozen=True)
class HybridConfig:
    max_nfev: int = 500
    det_starts: int = 3
    rna_cap_lambda: float = .25
    min_fit_rows: int = 12
    group_sizes: tuple = (2, 3)
    max_groups: int = 2
    polynomial_degree: int = 2
    ridge: float = .1
    min_group_rows: int = 24
    min_group_cell_ess: float = 30.
    min_query_cell_ess: float = 10.
    min_group_effective_rows: float = 12.
    max_geometry_condition: float = 10000.
    support_radius_quantile: float = .95
    min_penalized_gain: float = .05
    restart_prediction_spread_max: float = .05
    query_neighbors: int = 10
    workers: int = 1

    def __post_init__(self):
        for name in ("max_nfev", "det_starts", "min_fit_rows", "max_groups", "polynomial_degree",
                     "min_group_rows", "query_neighbors", "workers"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        if self.max_nfev > 3000 or self.det_starts > 3 or self.max_groups > 2:
            raise ValueError("solver/candidate budgets exceed initial bounded implementation")
        if self.polynomial_degree != 2:
            raise ValueError("only degree-two groups are supported")
        sizes = tuple(_integer(v, "group size", 2) for v in self.group_sizes)
        if not sizes or len(set(sizes)) != len(sizes) or any(v not in (2, 3) for v in sizes):
            raise ValueError("group_sizes must be a unique subset of (2,3)")
        object.__setattr__(self, "group_sizes", sizes)
        for name in ("rna_cap_lambda", "ridge", "min_group_cell_ess", "min_query_cell_ess",
                     "min_group_effective_rows", "max_geometry_condition", "support_radius_quantile",
                     "min_penalized_gain", "restart_prediction_spread_max"):
            value = _number(getattr(self, name), name)
            object.__setattr__(self, name, value)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if min(self.min_group_cell_ess, self.min_query_cell_ess, self.min_group_effective_rows) < 1:
            raise ValueError("ESS/effective-row thresholds must be >=1")
        if self.max_geometry_condition < 1 or not 0 < self.support_radius_quantile <= 1:
            raise ValueError("invalid geometry/radius threshold")
        if self.min_penalized_gain >= 1 or self.restart_prediction_spread_max > 1:
            raise ValueError("gain/spread thresholds must be fractions")

    def to_dict(self):
        return _plain({f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or set(value) - {f.name for f in fields(cls)}:
            raise ValueError("unknown HybridConfig keys")
        return cls(**dict(value))


@dataclass(frozen=True)
class CellSupport:
    """Shared normalized WT row-by-cell weights; integer array indices are row IDs."""
    weights: np.ndarray
    fingerprint: str = ""

    def __post_init__(self):
        w = _array(self.weights, "cell weights", 2)
        if not all(w.shape) or (w < 0).any() or not np.allclose(w.sum(axis=1), 1., atol=1e-10, rtol=1e-10):
            raise ValueError("cell weights must be nonnegative normalized rows")
        digest = hashlib.sha256(b"hybrid-cell-support-v1\0" + str(w.shape).encode() + w.tobytes()).hexdigest()
        if self.fingerprint and self.fingerprint != digest:
            raise ValueError("cell support fingerprint mismatch")
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "fingerprint", digest)


@dataclass(frozen=True)
class TargetData:
    """Aligned WT samples; caller owns correct lag/branch assembly and state units.

    target_rows and source_bins are branch-local bin numbers (source=target-lag).
    support_rows addresses the shared CellSupport's global row axis, independently
    of branch-local numbering. X has already gathered each ordered input/lag; y is
    already in output_unit. No normalization, lag reconstruction, or QC occurs here.
    """
    target_id: str
    input_keys: tuple
    X: np.ndarray
    y: np.ndarray
    source_bins: np.ndarray
    target_rows: np.ndarray
    branch_ids: tuple
    input_unit: str
    output_unit: str
    provenance: Mapping
    cell_weights: np.ndarray | None = None
    cell_support_ref: str = ""
    support_rows: np.ndarray | None = None
    route_supported: np.ndarray | None = None
    selected_lag: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.target_id, str) or not self.target_id:
            raise ValueError("target_id must be a nonempty string")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be primitive mapping metadata")
        keys = _keys(self.input_keys)
        X, y = _array(self.X, "X", 2), _array(self.y, "y", 1)
        source = _array(self.source_bins, "source_bins", 2, True)
        rows = _array(self.target_rows, "target_rows", 1, True)
        if X.shape != (len(y), len(keys)) or source.shape != X.shape or len(rows) != len(y) or not len(y):
            raise ValueError("target arrays must be aligned")
        branches = _branches(self.branch_ids, len(y))
        if (rows < 0).any() or len(set(zip(branches, rows.tolist()))) != len(rows):
            raise ValueError("target/source bins must be nonnegative and target rows unique within branch")
        route_supported = _routes(keys, rows, source, self.provenance, self.route_supported)
        selected_lag = _array(np.where(route_supported, rows[:,None]-source, -1),"selected_lag",2,True)
        if self.selected_lag is not None and not np.array_equal(_array(self.selected_lag,"selected_lag",2,True), selected_lag):
            raise ValueError("selected_lag differs from source bins/support")
        if self.input_unit not in INPUT_UNITS or self.output_unit not in OUTPUT_UNITS:
            raise ValueError("explicit supported input/output units required")
        _unit_domain(X, self.input_unit)
        if (y < 0).any() or (self.output_unit == "raw_atac" and (y > 1).any()):
            raise ValueError("outputs outside physical domain")
        support_rows = _array(np.arange(len(y)) if self.support_rows is None else self.support_rows,
                              "support_rows", 1, True)
        if len(support_rows) != len(y) or (support_rows < 0).any() or len(set(support_rows.tolist())) != len(y):
            raise ValueError("support row mapping must be unique and aligned")
        ref = self.cell_support_ref
        weights = None
        if self.cell_weights is not None:
            support = CellSupport(self.cell_weights, ref)
            if support_rows.max() >= len(support.weights):
                raise ValueError("support row mapping outside shared weights")
            weights, ref = support.weights, support.fingerprint
        if not isinstance(ref, str):
            raise ValueError("cell_support_ref must be a string")
        for name, value in dict(input_keys=keys, X=X, y=y, source_bins=source, target_rows=rows,
                branch_ids=branches, provenance=_freeze(self.provenance), cell_weights=weights,
                cell_support_ref=ref, support_rows=support_rows, route_supported=route_supported, selected_lag=selected_lag).items():
            object.__setattr__(self, name, value)


def rna_output_bounds(y_L2, config=HybridConfig()):
    y = _array(y_L2, "y_L2", 1)
    if not len(y) or (y < 0).any():
        raise ValueError("RNA L2 observations must be nonempty/nonnegative")
    return 0., float(y.max() + config.rna_cap_lambda * np.ptp(y))


def _softsign(z):
    with np.errstate(over="ignore", invalid="ignore"):
        finite = np.isfinite(z)
        result = np.divide(z, 1. + np.abs(z), out=np.zeros_like(z), where=finite)
        return np.where(finite, result, np.sign(z))


def _single_values(b, a, c, s, x, bounds):
    with np.errstate(over="ignore", invalid="ignore"):
        z = (np.asarray(x, dtype=np.longdouble) - c) / s
    return np.asarray(np.clip(b + a * _softsign(z), *bounds), dtype=float)


@dataclass(frozen=True)
class SingleModel:
    b: float
    a: float
    c: float
    s: float
    bounds: tuple
    x_range: tuple
    metrics: Mapping

    def __post_init__(self):
        bounds = _bounds(self.bounds)
        if not isinstance(self.metrics,Mapping):
            raise ValueError("single metrics must be a mapping")
        _integer(self.metrics.get("n_rows"),"single n_rows",1)
        if _number(self.metrics.get("training_rmse"),"training RMSE")<0:
            raise ValueError("invalid training RMSE")
        r2=self.metrics.get("training_r2")
        if r2 is not None and _number(r2,"training R2")>1+1e-12:
            raise ValueError("invalid training R2")
        if self.metrics.get("status") not in ("ok","budget_exhausted","best_finite","constant_input","constant_output"):
            raise ValueError("unknown single fit status")
        status=self.metrics.get("termination_status")
        if self.metrics["status"]=="ok":
            if _integer(status,"single termination",1)>4:
                raise ValueError("invalid solver termination")
        elif self.metrics["status"] in ("budget_exhausted","best_finite"):
            if _integer(status,"single termination")!=0:
                raise ValueError("budget termination mismatch")
            if self.metrics["status"] == "best_finite" and (
                    self.metrics.get("solver_converged") is not False or
                    self.metrics.get("accepted_best_finite") is not True or
                    self.metrics.get("original_status") != "budget_exhausted"):
                raise ValueError("best-finite acceptance/termination mismatch")
        elif status!="not_needed":
            raise ValueError("constant-fit termination mismatch")
        for name in ("b", "a", "c", "s"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.s <= 0 or min(self.b-self.a, self.b+self.a) < bounds[0]-1e-12 or max(self.b-self.a, self.b+self.a) > bounds[1]+1e-12:
            raise ValueError("single endpoints/scale violate physical bounds")
        xr = tuple(_number(v, "x_range") for v in self.x_range)
        if len(xr) != 2 or xr[1] < xr[0]:
            raise ValueError("invalid single WT range")
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "x_range", xr)
        object.__setattr__(self, "metrics", _freeze(self.metrics))

    def predict(self, x):
        x = _array(x, "single query")
        return _single_values(self.b, self.a, self.c, self.s, x, self.bounds)

    def to_dict(self):
        return _plain({f.name: getattr(self, f.name) for f in fields(self)})


def _metrics(y, predicted):
    mse = float(np.mean((predicted-y)**2))
    variance = float(np.mean((y-y.mean())**2))
    return dict(training_rmse=math.sqrt(mse), training_r2=None if variance == 0 or np.ptp(y)==0 else 1-mse/variance,
                n_rows=len(y), evaluation="all_WT_training_rows_not_validation")


def fit_single(x, y, bounds, config=HybridConfig()):
    if not isinstance(config, HybridConfig):
        raise ValueError("HybridConfig required")
    x, y = _array(x, "x", 1), _array(y, "y", 1)
    bounds = _bounds(bounds)
    if len(x) != len(y) or len(y) < config.min_fit_rows:
        raise ValueError("insufficient/alignment error in single WT rows")
    L, U = bounds
    if (y < L).any() or (y > U).any():
        raise ValueError("WT outputs outside supplied bounds")
    xmin, xmax = float(x.min()), float(x.max())
    if xmin == xmax or np.ptp(y) == 0 or L == U:
        b = float(np.clip(y.mean(),L,U))
        metrics = dict(_metrics(y, np.full(len(y), b)), status="constant_input" if xmin == xmax else "constant_output",
                       identifiable=False, termination_status="not_needed", parameter_profile="not_identifiable")
        return SingleModel(b, 0., float(x.mean()), 1., bounds, (xmin, xmax), metrics)
    scale = max(xmax-xmin, 1e-8)
    if scale > 1e100 or max(abs(xmin), abs(xmax), U) > 1e100:
        raise ValueError("training numerical scale exceeds solver budget")
    low = np.array([L, L, xmin-scale, math.log(scale*1e-6)])
    high = np.array([U, U, xmax+scale, math.log(scale*100)])
    mean = float(y.mean())
    starts = [(L+.1*(U-L), U-.1*(U-L), np.median(x), math.log(scale)),
              (U-.1*(U-L), L+.1*(U-L), np.median(x), math.log(scale)),
              (mean, mean, np.median(x), math.log(scale*.1))]
    def predict(p):
        return _single_values((p[0]+p[1])/2, (p[1]-p[0])/2, p[2], math.exp(p[3]), x, bounds)
    results = [least_squares(lambda p: (predict(p)-y)/(U-L), np.asarray(start),
                bounds=(low, high), max_nfev=config.max_nfev) for start in starts[:config.det_starts]]
    valid = [r for r in results if r.status >= 0 and np.isfinite(r.x).all()
             and (r.x >= low).all() and (r.x <= high).all()
             and np.isfinite(predict(r.x)).all() and np.isfinite(np.mean((predict(r.x)-y)**2))]
    if not valid:
        raise FloatingPointError("no finite bounded solver candidate")
    best = min(valid, key=lambda r: (float(np.mean((predict(r.x)-y)**2)), tuple(r.x)))
    p = best.x
    metrics = dict(_metrics(y, predict(p)), status="ok" if best.success else "budget_exhausted",
        identifiable=False, termination_status=int(best.status), nfev=int(best.nfev),
        solver_converged=bool(best.success), sum_nfev=sum(int(r.nfev) for r in results),
        starts=[dict(termination_status=int(r.status),nfev=int(r.nfev),
            solver_converged=bool(r.success),valid_candidate=any(r is v for v in valid),
            training_mse=float(np.mean((predict(r.x)-y)**2)) if any(r is v for v in valid) else None)
            for r in results],
        parameter_profile="not_tested; bounded parameters may remain weakly identified")
    return SingleModel((p[0]+p[1])/2, (p[1]-p[0])/2, p[2], math.exp(p[3]), bounds, (xmin, xmax), metrics)


def _features(u):
    return np.column_stack([np.ones(len(u))] + [u[:, i] for i in range(u.shape[1])] +
        [u[:, i]**2 for i in range(u.shape[1])] +
        [u[:, i]*u[:, j] for i in range(u.shape[1]) for j in range(i+1, u.shape[1])])


def _group_values(coefficients, mean, scale, X, bounds):
    u = (np.asarray(X,dtype=np.longdouble)-mean)/scale
    with np.errstate(over="ignore", invalid="ignore"):
        index = _features(u) @ coefficients
    return np.asarray(np.clip((bounds[0]+bounds[1])/2 + (bounds[1]-bounds[0])/2*_softsign(index), *bounds),dtype=float)


@dataclass(frozen=True)
class GroupModel:
    indices: tuple
    coefficients: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    reference_u: np.ndarray
    branch_ids: tuple
    support_rows: np.ndarray
    radius_by_branch: Mapping
    bounds: tuple
    diagnostics: Mapping

    def __post_init__(self):
        if not isinstance(self.radius_by_branch,Mapping) or not isinstance(self.diagnostics,Mapping):
            raise ValueError("group metadata must be mappings")
        indices = tuple(_integer(v, "group index") for v in self.indices)
        width = len(indices)
        if width not in (2, 3) or len(set(indices)) != width:
            raise ValueError("groups require 2 or 3 unique input indices")
        coef = _array(self.coefficients, "coefficients", 1)
        mean, scale = _array(self.mean, "mean", 1), _array(self.scale, "scale", 1)
        ref = _array(self.reference_u, "reference_u", 2)
        rows = _array(self.support_rows, "group support_rows", 1, True)
        if len(coef) != 1+2*width+width*(width-1)//2 or mean.shape != (width,) or scale.shape != (width,) or ref.shape[1] != width:
            raise ValueError("invalid group feature dimensions")
        if (scale <= 0).any() or len(rows) != len(ref) or (rows < 0).any() or len(set(rows.tolist())) != len(rows):
            raise ValueError("invalid group scale/support rows")
        if (np.abs(coef)>100).any() or not np.allclose(ref.mean(axis=0),0.,atol=1e-10) or not np.allclose(ref.std(axis=0),1.,atol=1e-10):
            raise ValueError("invalid coefficient budget/standardized reference")
        branches = _branches(self.branch_ids, len(ref))
        radii = dict(self.radius_by_branch)
        if set(radii) != set(branches) or any(_number(v, "radius") <= 0 for v in radii.values()):
            raise ValueError("invalid branch-local support radius")
        for name, value in dict(indices=indices, coefficients=coef, mean=mean, scale=scale,
                reference_u=ref, branch_ids=branches, support_rows=rows,
                radius_by_branch=_freeze(radii), bounds=_bounds(self.bounds), diagnostics=_freeze(self.diagnostics)).items():
            object.__setattr__(self, name, value)

    def predict(self, X):
        X = _array(X,"group query",2)
        if X.shape[1]!=len(self.indices):
            raise ValueError("group query width mismatch")
        return _group_values(self.coefficients, self.mean, self.scale, X, self.bounds)

    def to_dict(self):
        return _plain({f.name: getattr(self, f.name) for f in fields(self)})


def _cell_ess(weights):
    pooled = weights.mean(axis=0)
    return float(1./np.dot(pooled, pooled))


def _effective_rows(weights):
    gram = weights @ weights.T
    return float(np.trace(gram)**2 / np.sum(gram**2))


def _fit_group(data, indices, baseline, bounds, config):
    diagnostic: dict = dict(indices=list(indices), accepted=False, stability_evidence="deterministic_restarts_only_not_validation")
    def fail(reason):
        diagnostic["reason"] = reason
        return None, diagnostic
    if len(data.y) < config.min_group_rows:
        return fail("insufficient_group_rows")
    if data.cell_weights is None:
        return fail("training_cell_support_missing")
    weights = data.cell_weights[data.support_rows]
    parameters = 1+2*len(indices)+len(indices)*(len(indices)-1)//2
    effective_required = max(config.min_group_effective_rows,2*parameters)
    diagnostic.update(pooled_unique_cell_ess=_cell_ess(weights), kernel_effective_rows=_effective_rows(weights),
                      effective_rows_required=effective_required)
    if diagnostic["pooled_unique_cell_ess"] < config.min_group_cell_ess:
        return fail("training_cell_ess")
    if diagnostic["kernel_effective_rows"] < effective_required:
        return fail("kernel_effective_rows")
    X = data.X[:, indices]
    mean, scale = X.mean(axis=0), X.std(axis=0)
    if (scale <= 0).any():
        return fail("constant_group_input")
    u = (X-mean)/scale
    condition = float(np.linalg.cond(u))
    if not math.isfinite(condition) or condition > config.max_geometry_condition or np.linalg.matrix_rank(u) != len(indices):
        return fail("rank_deficient_or_ill_conditioned_geometry")
    diagnostic["geometry_condition"] = condition
    A = _features(u)
    if np.linalg.matrix_rank(A) != A.shape[1]:
        return fail("rank_deficient_polynomial_design")
    radii = {}
    for branch in sorted(set(data.branch_ids)):
        ref = u[np.array(data.branch_ids)==branch]
        if len(ref) < len(indices)+2:
            return fail("insufficient_branch_geometry_rows")
        centered = ref-ref.mean(axis=0)
        branch_condition = float(np.linalg.cond(centered))
        if np.linalg.matrix_rank(centered)!=len(indices) or not math.isfinite(branch_condition) or branch_condition>config.max_geometry_condition:
            return fail("rank_deficient_branch_geometry")
        distances = cdist(ref, ref)
        np.fill_diagonal(distances, np.inf)
        radius = float(np.quantile(distances.min(axis=1), config.support_radius_quantile))
        if not radius > 0 or not math.isfinite(radius):
            return fail("degenerate_support_radius")
        radii[branch] = radius
    L, U = bounds
    if L == U:
        return fail("constant_output_bounds")
    half = (U-L)/2
    desired = np.clip((data.y-(L+U)/2)/half, -.95, .95)
    index = desired/(1-np.abs(desired))
    initial = np.linalg.lstsq(A.T@A + max(config.ridge, 1e-12)*np.eye(A.shape[1]), A.T@index, rcond=None)[0]
    initial = np.clip(initial, -99., 99.)
    starts = [initial, np.zeros(A.shape[1]), initial*.5][:config.det_starts]
    def predict(coef):
        return (L+U)/2 + half*_softsign(A@coef)
    def residual(coef):
        return np.r_[(predict(coef)-data.y)/half,
                     math.sqrt(config.ridge*len(data.y)/len(coef))*coef]
    fits = [least_squares(residual, start, bounds=(-100.,100.), max_nfev=config.max_nfev) for start in starts]
    predictions = np.array([predict(f.x) for f in fits])
    spread = float(np.ptp(predictions, axis=0).max()/(U-L))
    diagnostic.update(restart_prediction_spread=spread, solver_statuses=[int(f.status) for f in fits])
    if not all(f.success for f in fits):
        return fail("group_solver_not_converged")
    if spread > config.restart_prediction_spread_max:
        return fail("restart_prediction_spread")
    baseline_mse = float(np.mean((baseline-data.y)**2))
    def objective(fit):
        mse = float(np.mean((predict(fit.x)-data.y)**2))
        ridge = config.ridge*float(np.mean(fit.x**2))*half**2
        complexity = len(fit.x)/len(data.y)*baseline_mse
        return mse+ridge+complexity
    best = min(fits, key=lambda f: (objective(f), tuple(f.x)))
    penalized = objective(best)
    gain = 0. if baseline_mse == 0 else (baseline_mse-penalized)/baseline_mse
    diagnostic.update(_metrics(data.y,predict(best.x)), baseline_mse=baseline_mse,
        penalized_group_mse=penalized, penalized_gain=gain,
        penalty_definition="MSE + ridge*mean(coef^2)*half_output_range^2 + n_params/n_rows*baseline_MSE")
    if gain < config.min_penalized_gain:
        return fail("insufficient_penalized_gain")
    diagnostic.update(accepted=True,reason="accepted")
    return GroupModel(indices,best.x,mean,scale,u,data.branch_ids,data.support_rows,radii,bounds,diagnostic), diagnostic


@dataclass(frozen=True)
class TargetModel:
    target_id: str
    input_keys: tuple
    singles: tuple
    scores: tuple
    weights: tuple
    bounds: tuple
    wt_mean: float
    status: str
    input_unit: str
    output_unit: str
    config: HybridConfig
    target_rows: np.ndarray
    source_bins: np.ndarray
    branch_ids: tuple
    support_rows: np.ndarray
    cell_support_ref: str
    data_fingerprint: str
    groups: tuple
    diagnostics: Mapping
    provenance: Mapping
    route_supported: np.ndarray | None = None
    selected_lag: np.ndarray | None = None

    def __post_init__(self):
        keys, bounds = _keys(self.input_keys), _bounds(self.bounds)
        p = len(keys)
        if not isinstance(self.target_id,str) or not self.target_id or not isinstance(self.config,HybridConfig):
            raise ValueError("invalid target/config")
        if self.config.workers != 1:
            raise ValueError("fitted target config must use canonical runtime workers=1")
        if self.input_unit not in INPUT_UNITS or self.output_unit not in OUTPUT_UNITS:
            raise ValueError("invalid bundle units")
        if not isinstance(self.provenance,Mapping) or not isinstance(self.diagnostics,Mapping):
            raise ValueError("bundle metadata must be mappings")
        validate_excluded_relations(self.provenance,self.target_id,keys)
        policy = validate_budget_exhausted_policy(self.provenance.get("budget_exhausted_policy", "exclude"))
        if policy != "best_finite" and any(s.metrics["status"]=="best_finite" for s in self.singles):
            raise ValueError("best-finite singles require explicit policy provenance")
        if "excluded_single_relations" in self.provenance and any(s.metrics["status"]=="budget_exhausted" for s in self.singles):
            raise ValueError("pruned models cannot execute exhausted singles")
        if self.diagnostics.get("alpha_frozen") is not True or self.diagnostics.get("active_input_renormalization") is not False:
            raise ValueError("bundle frozen-weight contract mismatch")
        if self.output_unit == "raw_atac" and bounds != (0.,1.):
            raise ValueError("raw ATAC output bounds must be (0,1)")
        scores = tuple(_number(v,"score") for v in self.scores)
        weights = tuple(_number(v,"weight") for v in self.weights)
        if len(self.singles) != p or len(scores) != p or len(weights) != p or min(scores+weights)<0:
            raise ValueError("invalid scores/weights/single dimensions")
        family = self.provenance.get("response_family", "saturation")
        if family not in ("saturation", "nn", "spline"):
            raise ValueError("unknown response family")
        single_type, group_type = SingleModel, GroupModel
        if family == "nn":
            from .nn_hybrid import NNSingleModel, NNGroupModel, NNConfig
            single_type, group_type = NNSingleModel, NNGroupModel
            if "nn_response_config" not in self.provenance:
                raise ValueError("NN configuration provenance required")
            nn_cfg = NNConfig.from_dict(self.provenance["nn_response_config"])
        if family == "spline":
            from .spline_hybrid import SplineSingleModel, SplineGroupModel, SplineConfig
            single_type, group_type = SplineSingleModel, SplineGroupModel
            spline_cfg = SplineConfig.from_dict(_plain(self.provenance.get("spline_response_config")))
            if any(_plain(s.config) != spline_cfg.to_dict() for s in self.singles if isinstance(s, SplineSingleModel)) or any(_plain(g.config)!=spline_cfg.to_dict() for g in self.groups):
                raise ValueError("spline configuration mismatch")
        if any(not isinstance(s,single_type) or s.bounds != bounds for s in self.singles):
            raise ValueError("single bounds mismatch")
        threshold = self.provenance.get('single_min_r2')
        if threshold is not None:
            threshold = _number(threshold, 'single_min_r2')
            if self.provenance.get('relation_type') != 'saturating_hybrid_peak_gene' or not 0 <= threshold <= 1:
                raise ValueError('invalid peak-gene single_min_r2 policy')
            if any(s.metrics.get('training_r2') is None or s.metrics['training_r2'] < threshold for s in self.singles):
                raise ValueError('retained single model fails single_min_r2')
        if family == "nn" and any(len(s.bias) != nn_cfg.hidden_width or s.metrics["status"] == "ok" and
                (s.metrics["epochs"] != nn_cfg.epochs or s.metrics["restarts"] != nn_cfg.restarts) for s in self.singles):
            raise ValueError("NN configuration/evidence mismatch")
        for s in self.singles:
            _unit_domain(np.array(s.x_range),self.input_unit)
        total = sum(scores)
        if not math.isfinite(total):
            raise ValueError("WT score total must be finite")
        if total>0:
            weight_total = sum(weights)
            if not math.isfinite(weight_total) or not math.isclose(weight_total,1.,rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError("informative weights must sum to one")
        elif any(weight!=0 for weight in weights):
            raise ValueError("no-information weights must be zero")
        expected = np.array(scores)/total if total>0 else np.zeros(p)
        if not np.allclose(weights,expected,atol=1e-12,rtol=1e-12):
            raise ValueError("weights must equal frozen normalized WT scores")
        if self.status != ("ok" if total>0 else "no_information"):
            raise ValueError("status inconsistent with WT scores")
        if total==0 and self.groups:
            raise ValueError("no-information models cannot contain groups")
        mean = _number(self.wt_mean,"wt_mean")
        if not bounds[0] <= mean <= bounds[1]:
            raise ValueError("WT mean outside bounds")
        rows = _array(self.target_rows,"target_rows",1,True)
        source = _array(self.source_bins,"source_bins",2,True)
        support = _array(self.support_rows,"support_rows",1,True)
        branches = _branches(self.branch_ids,len(rows))
        if source.shape != (len(rows),p) or len(support)!=len(rows) or (support<0).any() or len(set(support.tolist()))!=len(rows):
            raise ValueError("invalid bundle row mappings")
        if (rows<0).any() or len(set(zip(branches,rows.tolist())))!=len(rows):
            raise ValueError("invalid causal bundle bins")
        route_supported = _routes(keys, rows, source, self.provenance, self.route_supported)
        selected_lag = _array(np.where(route_supported, rows[:,None]-source, -1),"selected_lag",2,True)
        if self.selected_lag is not None and not np.array_equal(_array(self.selected_lag,"selected_lag",2,True),selected_lag):
            raise ValueError("selected_lag differs from source bins/support")
        local = any(k[1] == "local" for k in keys)
        if len(rows)<self.config.min_fit_rows or any(s.metrics.get("n_rows")!=int(route_supported[:,i].sum()) for i,s in enumerate(self.singles)):
            raise ValueError("bundle WT row counts mismatch")
        if local:
            if (route_supported.sum(axis=0) < self.config.min_fit_rows).any():
                raise ValueError("local singles require enough supported WT rows")
            means = _array(self.diagnostics.get("single_WT_reference_means"), "single WT references", 1)
            if len(means) != p or (means < bounds[0]).any() or (means > bounds[1]).any():
                raise ValueError("invalid local single WT references")
        for digest in (self.data_fingerprint,self.cell_support_ref):
            if not isinstance(digest,str) or (digest and (len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest))):
                raise ValueError("invalid reference digest")
        if not self.data_fingerprint:
            raise ValueError("data fingerprint required")
        ranking = sorted(range(p),key=lambda i:(-scores[i],i))
        if len(self.groups)>self.config.max_groups:
            raise ValueError("too many groups")
        seen=set()
        for g in self.groups:
            if not isinstance(g,group_type) or g.bounds!=bounds or g.indices in seen or tuple(ranking[:len(g.indices)])!=g.indices:
                raise ValueError("invalid ranked group/bounds")
            if len(g.indices) not in self.config.group_sizes:
                raise ValueError("group size outside configured candidates")
            seen.add(g.indices)
            group_mask = np.all(route_supported[:,g.indices], axis=1) if local else np.ones(len(rows),dtype=bool)
            group_branches = tuple(np.asarray(branches)[group_mask])
            group_n = int(group_mask.sum())
            if not self.cell_support_ref or len(g.reference_u)!=group_n or g.branch_ids!=group_branches or not np.array_equal(g.support_rows,support[group_mask]):
                raise ValueError("group shared support references mismatch")
            d=g.diagnostics
            gain=_number(d.get("penalized_gain",-1),"penalized_gain")
            spread=_number(d.get("restart_prediction_spread",-1),"restart_prediction_spread")
            if d.get("accepted") is not True or not self.config.min_penalized_gain<=gain<=1 or not 0<=spread<=self.config.restart_prediction_spread_max:
                raise ValueError("group acceptance diagnostics mismatch")
            parameters = g.n_parameters if family in ("nn", "spline") else len(g.coefficients)
            required=max(self.config.min_group_effective_rows,2*parameters)
            row_floor = max(self.config.min_group_rows, 2*parameters) if family in ("nn", "spline") else self.config.min_group_rows
            if _number(d.get("pooled_unique_cell_ess",0),"cell ESS")<self.config.min_group_cell_ess or _number(d.get("kernel_effective_rows",0),"effective rows")<required or group_n<row_floor:
                raise ValueError("group training support gates mismatch")
            statuses=d.get("solver_statuses",())
            if family == "nn":
                if len(g.bias) != nn_cfg.hidden_width or d.get("training_evidence") != "fixed_epochs_all_WT_not_validation" or d.get("training_status") != "fixed_epochs_completed" or d.get("epochs") != nn_cfg.epochs or d.get("restarts") != nn_cfg.restarts or statuses:
                    raise ValueError("NN group training evidence mismatch")
            elif len(statuses)!=(spline_cfg.restarts if family=="spline" else self.config.det_starts) or any(_integer(v,"solver status",1)>4 for v in statuses):
                raise ValueError("group solver termination evidence mismatch")
            if d.get("n_rows")!=group_n or d.get("effective_rows_required")!=required:
                raise ValueError("group row/parameter support evidence mismatch")
            baseline_mse=_number(d.get("baseline_mse",-1),"baseline MSE")
            rmse=_number(d.get("training_rmse",-1),"training RMSE")
            if baseline_mse<=0 or rmse<0:
                raise ValueError("invalid group gain evidence")
            if family == "nn":
                square = sum(float(np.sum(v**2)) for v in (g.W, g.bias, g.v)) + g.offset**2
                penalty=rmse**2+nn_cfg.l2*square/parameters*(bounds[1]-bounds[0])**2+parameters/group_n*baseline_mse
            elif family == "spline":
                penalty=rmse**2+(bounds[1]-bounds[0])**2*(spline_cfg.smoothness*d['shape_penalty']+spline_cfg.group_interaction_penalty*d['interaction_square'])+parameters/group_n*baseline_mse
            else:
                penalty=rmse**2+self.config.ridge*float(np.mean(g.coefficients**2))*((bounds[1]-bounds[0])/2)**2+len(g.coefficients)/group_n*baseline_mse
            if not math.isclose(penalty,_number(d.get("penalized_group_mse",-1),"penalized MSE"),rel_tol=1e-10,abs_tol=1e-14) or not math.isclose(gain,(baseline_mse-penalty)/baseline_mse,rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError("group penalized-gain algebra mismatch")
            condition=float(np.linalg.cond(g.reference_u))
            if np.linalg.matrix_rank(g.reference_u)!=len(g.indices) or not math.isfinite(condition) or condition>self.config.max_geometry_condition:
                raise ValueError("group geometry gate mismatch")
            if family == "saturation" and np.linalg.matrix_rank(_features(g.reference_u))!=len(g.coefficients):
                raise ValueError("group polynomial rank mismatch")
            _unit_domain(g.reference_u*g.scale+g.mean,self.input_unit,tolerance=1e-12)
            for branch in sorted(set(group_branches)):
                reference=g.reference_u[np.array(group_branches)==branch]
                centered=reference-reference.mean(axis=0)
                if len(reference)<len(g.indices)+2 or np.linalg.matrix_rank(centered)!=len(g.indices) or np.linalg.cond(centered)>self.config.max_geometry_condition:
                    raise ValueError("group branch geometry mismatch")
                distances=cdist(reference,reference);np.fill_diagonal(distances,np.inf)
                radius=float(np.quantile(distances.min(axis=1),self.config.support_radius_quantile))
                if not math.isclose(g.radius_by_branch[branch],radius,rel_tol=1e-10,abs_tol=0):
                    raise ValueError("group radius calibration mismatch")
        for name,value in dict(input_keys=keys,singles=tuple(self.singles),scores=scores,weights=weights,
                bounds=bounds,wt_mean=mean,target_rows=rows,source_bins=source,branch_ids=branches,
                support_rows=support,groups=tuple(self.groups),diagnostics=_freeze(self.diagnostics),provenance=_freeze(self.provenance),route_supported=route_supported, selected_lag=selected_lag).items():
            object.__setattr__(self,name,value)

    def to_bundle(self):
        payload = {f.name:getattr(self,f.name) for f in fields(self)}
        if not any(lag == "local" for _,lag in self.input_keys):
            payload.pop("route_supported")
            payload.pop("selected_lag")
        payload.update(schema=CONTINUOUS_SCHEMA if "normalized_atac_log1p" in (self.input_unit,self.output_unit) else SCHEMA,config=self.config.to_dict(),
                       singles=[s.to_dict() for s in self.singles],groups=[g.to_dict() for g in self.groups])
        if self.provenance.get("response_family", "saturation") == "nn":
            from .nn_hybrid import SCHEMA as NN_SCHEMA
            payload["schema"] = NN_SCHEMA
            for component in payload["singles"] + payload["groups"]:
                component.update(input_unit=self.input_unit, output_unit=self.output_unit)
        if self.provenance.get("response_family") == "spline":
            from .spline_hybrid import SCHEMA as SPLINE_SCHEMA
            payload["schema"] = SPLINE_SCHEMA
            for component in payload["singles"] + payload["groups"]:
                component.update(input_unit=self.input_unit, output_unit=self.output_unit)
        if payload["schema"] == CONTINUOUS_SCHEMA:
            for key, schema in (("singles", CONTINUOUS_SINGLE_SCHEMA), ("groups", CONTINUOUS_GROUP_SCHEMA)):
                for component in payload[key]:
                    component.update(schema=schema, input_unit=self.input_unit, output_unit=self.output_unit)
        payload = _plain(payload)
        payload["fingerprint"] = _digest(payload)
        return payload

    @property
    def fingerprint(self):
        return self.to_bundle()["fingerprint"]

    @classmethod
    def from_bundle(cls,bundle, *, cell_support=None):
        bundle = _plain(bundle)
        expected={f.name for f in fields(cls)}|{"schema","fingerprint"}
        from .nn_hybrid import SCHEMA as NN_SCHEMA, NNSingleModel, NNGroupModel
        from .spline_hybrid import SCHEMA as SPLINE_SCHEMA, SplineSingleModel, SplineGroupModel
        optional = {"route_supported", "selected_lag"}
        if not isinstance(bundle,dict) or set(bundle) not in (expected, expected-optional) or bundle["schema"] not in (SCHEMA,CONTINUOUS_SCHEMA,NN_SCHEMA,SPLINE_SCHEMA):
            raise ValueError("unknown/malformed hybrid bundle schema")
        continuous = "normalized_atac_log1p" in (bundle["input_unit"],bundle["output_unit"])
        neural = bundle["schema"] == NN_SCHEMA
        spline = bundle["schema"] == SPLINE_SCHEMA
        if spline != (bundle["provenance"].get("response_family") == "spline"):
            raise ValueError("spline schema/family mismatch")
        if neural != (bundle["provenance"].get("response_family", "saturation") == "nn"):
            raise ValueError("hybrid schema/family mismatch")
        if any(lag == "local" for _,lag in bundle["input_keys"]) and not optional.issubset(bundle):
            raise ValueError("local bundle requires route support and selected lags")
        if not neural and not spline and (bundle["schema"]==CONTINUOUS_SCHEMA) != continuous:
            raise ValueError("hybrid target schema/unit mismatch")
        signature=bundle.pop("fingerprint")
        if signature!=_digest(bundle):
            raise ValueError("hybrid bundle fingerprint mismatch")
        bundle.pop("schema")
        try:
            if continuous and not neural and not spline:
                for key, schema in (("singles", CONTINUOUS_SINGLE_SCHEMA), ("groups", CONTINUOUS_GROUP_SCHEMA)):
                    for component in bundle[key]:
                        if (component.pop("schema", None) != schema or
                                component.pop("input_unit", None) != bundle["input_unit"] or
                                component.pop("output_unit", None) != bundle["output_unit"]):
                            raise ValueError("continuous hybrid component schema/unit mismatch")
            bundle["config"]=HybridConfig.from_dict(bundle["config"])
            if neural:
                for component in bundle["singles"] + bundle["groups"]:
                    if component.pop("response_type", None) != "nn":
                        raise ValueError("NN component family mismatch")
                    if (component.pop("input_unit", None) != bundle["input_unit"] or
                            component.pop("output_unit", None) != bundle["output_unit"]):
                        raise ValueError("NN component unit mismatch")
            if spline:
                for component in bundle["singles"] + bundle["groups"]:
                    if component.pop("response_type", None) != "spline":
                        raise ValueError("spline component family mismatch")
                    if (component.pop("input_unit", None) != bundle["input_unit"] or
                            component.pop("output_unit", None) != bundle["output_unit"]):
                        raise ValueError("spline component unit mismatch")
            bundle["singles"]=tuple((SplineSingleModel if spline else NNSingleModel if neural else SingleModel)(**v) for v in bundle["singles"])
            bundle["groups"]=tuple((SplineGroupModel if spline else NNGroupModel if neural else GroupModel)(**v) for v in bundle["groups"])
            model=cls(**bundle)
            if cell_support is not None:
                validate_cell_support(model,cell_support)
            return model
        except (TypeError,KeyError) as exc:
            raise ValueError("malformed hybrid bundle fields") from exc


def validate_cell_support(model, support):
    """Adapter load-time verification against the one shared WT support dataset.

    A bundle alone checks its reference/structure, not the external cell
    evidence. This check recomputes pooled ESS and kernel effective rows without
    modifying the immutable bundle. Query routing also requires this same digest.
    """
    if not isinstance(support,CellSupport) or support.fingerprint!=model.cell_support_ref:
        raise ValueError("shared cell support fingerprint mismatch")
    if model.support_rows.max()>=len(support.weights):
        raise ValueError("shared support row mapping outside dataset")
    w=support.weights[model.support_rows]
    ess,effective=_cell_ess(w),_effective_rows(w)
    for g in model.groups:
        d=g.diagnostics
        group_weights = support.weights[g.support_rows]
        if not math.isclose(_cell_ess(group_weights),d["pooled_unique_cell_ess"],rel_tol=1e-10) or not math.isclose(_effective_rows(group_weights),d["kernel_effective_rows"],rel_tol=1e-10):
            raise ValueError("serialized group cell evidence mismatch")
    return dict(fingerprint=support.fingerprint,pooled_unique_cell_ess=ess,kernel_effective_rows=effective)


class TargetFitFailure(RuntimeError):
    """No executable single relation remains; adapters retain the audit."""

    def __init__(self, diagnostics):
        super().__init__("no executable single relations remain")
        self.diagnostics = _plain(diagnostics)


def validate_excluded_relations(provenance, target_id, input_keys):
    audit = provenance.get("excluded_single_relations")
    if audit is None:
        return
    if not isinstance(audit, Mapping) or set(audit) != {"original_input_count", "records"}:
        raise ValueError("invalid excluded single audit")
    records = audit["records"]
    if not isinstance(records, (list, tuple)):
        raise ValueError("invalid excluded single records")
    count = _integer(audit["original_input_count"], "original input count", 1)
    if count != len(input_keys) + len(records):
        raise ValueError("excluded single count mismatch")
    seen = set(input_keys)
    for record in records:
        required = {"target_id", "input_key", "status", "training_rmse", "training_r2", "nfev", "reason"}
        if not isinstance(record, Mapping) or set(record) != required:
            raise ValueError("invalid excluded single record")
        key = _keys((record["input_key"],))[0]
        if key in seen or record["target_id"] != target_id:
            raise ValueError("excluded single identity mismatch")
        seen.add(key)
        if record["status"] not in ("budget_exhausted", "failed", "no_supported_lag", "insufficient_supported_rows", "low_single_r2") or not isinstance(record["reason"], str) or not record["reason"]:
            raise ValueError("invalid excluded single status/reason")
        for name in ("training_rmse", "training_r2"):
            value = record[name]
            if value is not None:
                value = _number(value, name)
                if name == "training_rmse" and value < 0 or name == "training_r2" and value > 1:
                    raise ValueError("invalid excluded single metric")
        if record["nfev"] is not None:
            _integer(record["nfev"], "excluded nfev", 0)


def validate_budget_exhausted_policy(value):
    if not isinstance(value, str) or value not in ("exclude", "best_finite"):
        raise ValueError("budget_exhausted_policy must be exclude or best_finite")
    return value


def fit_target(data, bounds, config=HybridConfig()):
    """Fit singles on all WT rows; freeze scores; attempt top-2/top-3 groups."""
    if not isinstance(data,TargetData) or not isinstance(config,HybridConfig):
        raise ValueError("TargetData/HybridConfig required")
    # Concurrency belongs to adapters, never numerical fits or their codecs.
    config = replace(config, workers=1)
    policy = validate_budget_exhausted_policy(data.provenance.get("budget_exhausted_policy", "exclude"))
    if data.support_rows is None:
        raise ValueError("validated target support-row mapping required")
    bounds=_bounds(bounds)
    single_threshold = data.provenance.get('single_min_r2')
    if single_threshold is not None:
        single_threshold = _number(single_threshold, 'single_min_r2')
        if data.provenance.get('relation_type') != 'saturating_hybrid_peak_gene' or not 0 <= single_threshold <= 1:
            raise ValueError('single_min_r2 requires peak-gene relation and threshold in [0,1]')
    if data.output_unit=="raw_atac" and bounds!=(0.,1.):
        raise ValueError("raw ATAC output bounds must be (0,1)")
    if len(data.y) < config.min_fit_rows or (data.y < bounds[0]).any() or (data.y > bounds[1]).any():
        raise ValueError("invalid global WT rows/output bounds")
    singles=[]; keep=[]; excluded=[]
    local = any(k[1] == "local" for k in data.input_keys)
    original_count=len(data.input_keys)
    family = data.provenance.get("response_family", "saturation")
    if family not in ("nn", "saturation", "spline"):
        raise ValueError("unknown response family")
    if family == "nn":
        from .nn_hybrid import NNConfig, fit_singles, fit_group
        nn_cfg = NNConfig.from_dict(data.provenance.get("nn_response_config", {}))
        neural_singles = None if local else fit_singles(data.X, data.y, data.input_keys, bounds, nn_cfg)
    if family == "spline":
        from .spline_hybrid import SplineConfig, fit_single as fit_spline_single, fit_group as fit_spline_group
        spline_cfg = SplineConfig.from_dict(_plain(data.provenance.get("spline_response_config")))
    for i,key in enumerate(data.input_keys):
        mask = data.route_supported[:,i]
        if local and int(mask.sum()) < config.min_fit_rows:
            status = "no_supported_lag" if not mask.any() else "insufficient_supported_rows"
            excluded.append(dict(target_id=data.target_id,input_key=key,status=status,
                training_rmse=None,training_r2=None,nfev=None,reason=status))
            continue
        x, y = data.X[mask,i], data.y[mask]
        try:
            single=((fit_singles(x[:,None], y, (key,), bounds, nn_cfg)[0] if local else neural_singles[i]) if family == "nn" else
                    fit_spline_single(x,y,bounds,spline_cfg) if family == "spline" else
                    fit_single(x,y,bounds,config))
            if single is None:
                raise FloatingPointError("nonfinite NN training")
        except FloatingPointError as exc:
            excluded.append(dict(target_id=data.target_id,input_key=key,status="failed",
                training_rmse=None,training_r2=None,nfev=getattr(exc,"nfev",None),reason=f"solver_numeric_failure: {exc}"))
            continue
        if single.metrics["status"] == "budget_exhausted":
            if policy == "exclude":
                excluded.append(dict(target_id=data.target_id,input_key=key,status="budget_exhausted",
                    training_rmse=single.metrics["training_rmse"],training_r2=single.metrics["training_r2"],
                    nfev=single.metrics.get("nfev"),reason="solver_budget_exhausted"))
                continue
            prediction = single.predict(x)
            if not np.isfinite(prediction).all() or (prediction < bounds[0]).any() or (prediction > bounds[1]).any():
                excluded.append(dict(target_id=data.target_id,input_key=key,status="failed",
                    training_rmse=None,training_r2=None,nfev=single.metrics.get("nfev"),
                    reason="solver_numeric_failure: invalid best-finite predictions"))
                continue
            single = replace(single, metrics=dict(_plain(single.metrics),
                status="best_finite",original_status="budget_exhausted",solver_converged=False,
                accepted_best_finite=True))
        r2 = single.metrics.get('training_r2')
        if single_threshold is not None and (r2 is None or not math.isfinite(r2) or r2 < single_threshold):
            excluded.append(dict(target_id=data.target_id,input_key=key,status='low_single_r2',
                training_rmse=single.metrics['training_rmse'],training_r2=r2,
                nfev=single.metrics.get('nfev'),reason='single_training_r2_below_threshold_or_undefined'))
            continue
        singles.append(single); keep.append(i)
    provenance=_plain(data.provenance)
    provenance["excluded_single_relations"]=dict(original_input_count=original_count,records=excluded)
    if not keep:
        raise TargetFitFailure(dict(status="no_information",train_r2=None,training_rmse=None,
            n_rows=len(data.y),excluded_single_relations=provenance["excluded_single_relations"],
            **(dict(local_route_diagnostics=provenance.get("local_route_diagnostics", {})) if local else {})))
    data=replace(data,input_keys=tuple(data.input_keys[i] for i in keep),X=data.X[:,keep],
        source_bins=data.source_bins[:,keep],route_supported=data.route_supported[:,keep],selected_lag=data.selected_lag[:,keep],provenance=provenance)
    singles=tuple(singles)
    # A mean of bounded predictions can round one ULP outside their bounds
    # (e.g. mean(full(37, .3)) > .3). Keep the stored baseline in its domain.
    references = [float(np.clip(s.predict(data.X[data.route_supported[:,i],i]).mean(), *bounds)) for i,s in enumerate(singles)] if local else None
    predictions=(_single_predictions(singles, data.X, data.route_supported, references) if local else
                 np.column_stack([s.predict(data.X[:,i]) for i,s in enumerate(singles)]))
    # Only neighboring supplied target rows with all corresponding source bins
    # consecutive in the same branch. Never sort away or bridge missing rows.
    scores, adjacent = _wt_scores(predictions, data.target_rows, data.branch_ids,
        data.source_bins, data.route_supported, local)
    with np.errstate(over="ignore",invalid="ignore"):
        total=float(scores.sum())
    if not math.isfinite(total):
        raise ValueError("WT score total must be finite")
    weights=scores/total if total>0 else np.zeros(len(scores))
    baseline=predictions@weights if total>0 else np.full(len(data.y),data.y.mean())
    groups=[]; attempts=[]
    ranked=sorted(range(len(scores)),key=lambda i:(-scores[i],i))
    if total>0:
        for size in config.group_sizes:
            if size>len(scores) or len(attempts)>=config.max_groups:
                continue
            indices = tuple(ranked[:size])
            group_data, group_baseline = data, baseline
            if local:
                common = np.all(data.route_supported[:,indices], axis=1)
                if not common.any():
                    attempts.append(dict(indices=list(indices),accepted=False,reason="no_common_supported_rows"))
                    continue
                group_data = replace(data, X=data.X[common], y=data.y[common],
                    source_bins=data.source_bins[common], route_supported=data.route_supported[common],
                    selected_lag=data.selected_lag[common],
                    target_rows=data.target_rows[common], branch_ids=tuple(np.asarray(data.branch_ids)[common]),
                    support_rows=data.support_rows[common])
                group_baseline = baseline[common]
            if family == "spline":
                group,diagnostic=fit_spline_group(group_data,indices,group_baseline,bounds,config,spline_cfg)
                attempts.append(diagnostic)
                if group is not None:groups.append(group)
                continue
            group,diagnostic=(fit_group(group_data,indices,group_baseline,bounds,config,nn_cfg)
                              if family == "nn" else _fit_group(group_data,indices,group_baseline,bounds,config))
            attempts.append(diagnostic)
            if group is not None:groups.append(group)
    fingerprint=_digest(dict(domain=CONTINUOUS_SCHEMA if "normalized_atac_log1p" in (data.input_unit,data.output_unit) else SCHEMA,target_id=data.target_id,input_keys=data.input_keys,X=data.X,y=data.y,
        target_rows=data.target_rows,source_bins=data.source_bins,branch_ids=data.branch_ids,
        input_unit=data.input_unit,output_unit=data.output_unit,provenance=data.provenance,
        cell_support_ref=data.cell_support_ref,support_rows=data.support_rows))
    diagnostics=dict(n_adjacent_pairs=int(adjacent.sum()),group_attempts=attempts,
        score_definition="mean_abs_fitted_single_change_on_common_consecutive_branch_local_WT_rows",
        all_WT_rows_used=True,alpha_frozen=True,active_input_renormalization=False,
        support_thresholds="experimental",parameter_identifiability="not_established",
        single_fit_terminations=[s.metrics["status"] for s in singles])
    if local:
        diagnostics.update(single_WT_reference_means=references,
            single_fit_n_rows=[int(data.route_supported[:,i].sum()) for i in range(len(singles))],
            prediction_n_rows=len(data.y), score_definition="supported_adjacent_target_changes_divided_by_common_target_pair_count",
            unsupported_absolute_baseline="single_WT_training_prediction_mean", unsupported_delta=0.)
    return TargetModel(data.target_id,data.input_keys,singles,tuple(scores),tuple(weights),bounds,float(np.clip(data.y.mean(),*bounds)),
        "ok" if total>0 else "no_information",data.input_unit,data.output_unit,config,data.target_rows,
        data.source_bins,data.branch_ids,data.support_rows,data.cell_support_ref,fingerprint,tuple(groups),diagnostics,data.provenance,data.route_supported)


@dataclass(frozen=True)
class PredictionResult:
    values: np.ndarray
    route: str
    group_indices: tuple
    diagnostics: Mapping
    output_unit: str

    def __post_init__(self):
        object.__setattr__(self,"values",_array(self.values,"prediction",1))
        object.__setattr__(self,"diagnostics",_freeze(self.diagnostics))


def _query_support(group, X, branches, support, config):
    with np.errstate(over="ignore",invalid="ignore"):
        u=(X[:,group.indices]-group.mean)/group.scale
    ess=[]
    if not np.isfinite(u).all():return False,"query_geometry_nonfinite",ess
    for row,branch in zip(u,branches):
        mask=np.array(group.branch_ids)==branch
        if not mask.any():return False,"query_branch_unsupported",ess
        with np.errstate(over="ignore",invalid="ignore"):
            distances=np.linalg.norm(group.reference_u[mask]-row,axis=1)
        radius=group.radius_by_branch[branch]
        if not np.isfinite(distances).all() or distances.min()>radius:
            return False,"query_geometry_out_of_support",ess
        nearest=np.argsort(distances,kind="stable")[:config.query_neighbors]
        local=np.exp(-.5*(distances[nearest]/radius)**2)
        local/=local.sum()
        row_ids=group.support_rows[mask][nearest]
        if row_ids.max()>=len(support.weights):return False,"query_support_rows_missing",ess
        pooled=local@support.weights[row_ids]
        value=float(1./np.dot(pooled,pooled));ess.append(value)
        if value<config.min_query_cell_ess:return False,"query_cell_ess",ess
    return True,"supported",ess


def _responses(model, X_wt, X_new, changed_inputs, branch_ids, cell_support, input_support=None, target_rows=None):
    if not isinstance(model,TargetModel):
        raise ValueError("TargetModel required")
    old,new=_array(X_wt,"WT query",2),_array(X_new,"new query",2)
    if old.shape!=new.shape or old.shape[1]!=len(model.input_keys) or not len(old):
        raise ValueError("query matrices must be aligned to input_keys")
    _unit_domain(old,model.input_unit);_unit_domain(new,model.input_unit)
    local = any(k[1] == "local" for k in model.input_keys)
    if input_support is None:
        if local and old.shape != model.route_supported.shape:
            raise ValueError("partial local queries require explicit input_support")
        supported = model.route_supported if local else np.ones(old.shape,dtype=bool)
    else:
        supported = np.asarray(input_support)
        if supported.dtype != np.bool_ or supported.shape != old.shape:
            raise ValueError("query input_support must be an aligned boolean matrix")
    actual={model.input_keys[i] for i in range(old.shape[1]) if np.any((old[:,i]!=new[:,i]) & supported[:,i])}
    declared=actual if changed_inputs is None else set(_keys(changed_inputs)) if changed_inputs else set()
    if not declared.issubset(set(model.input_keys)) or not actual.issubset(declared):
        raise ValueError("declared changed_inputs must cover all actual changes with known keys")
    if branch_ids is None:
        branches=model.branch_ids if len(old)==len(model.branch_ids) else None
        if branches is None:raise ValueError("branch_ids required for partial trajectory queries")
    else:branches=_branches(branch_ids,len(old))
    if local:
        if target_rows is None:
            if len(old) != len(model.target_rows) or branches != model.branch_ids:
                raise ValueError("partial local queries require target_rows")
            reference_support = model.route_supported
        else:
            query_rows = _array(target_rows,"query target_rows",1,True)
            if len(query_rows) != len(old):
                raise ValueError("query target rows must be aligned")
            lookup = {key:i for i,key in enumerate(zip(model.branch_ids,model.target_rows.tolist()))}
            try:
                positions = [lookup[(branch,int(row))] for branch,row in zip(branches,query_rows)]
            except KeyError as exc:
                raise ValueError("query target outside frozen local routes") from exc
            reference_support = model.route_supported[positions]
        if (supported & ~reference_support).any():
            raise ValueError("query input_support cannot enable an unsupported frozen route")
    if cell_support is not None and not isinstance(cell_support,CellSupport):
        raise ValueError("cell_support must be a shared CellSupport")
    if cell_support is not None and cell_support.fingerprint!=model.cell_support_ref:
        raise ValueError("query cell support fingerprint mismatch")
    if cell_support is not None and model.groups:
        validate_cell_support(model,cell_support)
    out_of_range=[model.input_keys[i] for i,s in enumerate(model.singles)
        if np.any((old[:,i]<s.x_range[0]) & supported[:,i]) or np.any((old[:,i]>s.x_range[1]) & supported[:,i]) or np.any((new[:,i]<s.x_range[0]) & supported[:,i]) or np.any((new[:,i]>s.x_range[1]) & supported[:,i])]
    diagnostic: dict[str,Any] = dict(changed_inputs=sorted(declared),single_WT_range_exceeded=out_of_range,
        saturation_out_of_WT_support=bool(out_of_range),alpha_frozen=True,active_input_renormalization=False,
        group_fallbacks=[],route_scope="whole_query_batch",group_corrections_added=False)
    if model.status=="no_information":
        mean=np.full(len(old),model.wt_mean)
        return mean,mean,"no_information",(),diagnostic
    references = model.diagnostics.get("single_WT_reference_means", [model.wt_mean]*len(model.singles))
    def single_sum(X):
        return np.clip(_single_predictions(model.singles,X,supported,references)@np.array(model.weights),*model.bounds)
    if not actual:
        predicted=single_sum(old)
        return predicted,predicted,"no_change",(),diagnostic
    changed_indices={model.input_keys.index(k) for k in declared}
    ordered=sorted(model.groups,key=lambda g:(-g.diagnostics["penalized_gain"],g.indices))
    for group in ordered:
        if not supported[:,group.indices].all():
            diagnostic["group_fallbacks"].append(dict(indices=group.indices,reason="query_local_route_unsupported"));continue
        if not changed_indices.issubset(set(group.indices)):
            diagnostic["group_fallbacks"].append(dict(indices=group.indices,reason="changed_inputs_not_covered"));continue
        if cell_support is None:
            diagnostic["group_fallbacks"].append(dict(indices=group.indices,reason="query_cell_support_missing"));continue
        ok,reason,ess_old=_query_support(group,old,branches,cell_support,model.config)
        ess_new=[]
        if ok:ok,reason,ess_new=_query_support(group,new,branches,cell_support,model.config)
        if not ok:
            diagnostic["group_fallbacks"].append(dict(indices=group.indices,reason=reason));continue
        diagnostic.update(query_cell_ess_WT=ess_old,query_cell_ess_new=ess_new)
        return group.predict(old[:,group.indices]),group.predict(new[:,group.indices]),"group",group.indices,diagnostic
    old_values=single_sum(old)
    new_values=single_sum(new)
    return old_values,new_values,"singles",(),diagnostic


def predict_delta(model, X_wt, X_new, *, changed_inputs=None, branch_ids=None, cell_support=None, input_support=None, target_rows=None):
    """One whole-batch route: supported group replacement OR frozen single sum."""
    old,new,route,indices,diagnostic=_responses(model,X_wt,X_new,changed_inputs,branch_ids,cell_support,input_support,target_rows)
    return PredictionResult(new-old,route,indices,diagnostic,model.output_unit)


def predict_target(model, X, *, wt_X=None, changed_inputs=None, branch_ids=None, cell_support=None, input_support=None, target_rows=None):
    """Absolute output; wt_X is required to select a changed-input group route.

    With no wt_X, this is the frozen single baseline (or no_information WT mean).
    Supply matching WT/new batches to use exactly the predict_delta routing.
    """
    old,new,route,indices,diagnostic=_responses(model,X if wt_X is None else wt_X,X,
        changed_inputs,branch_ids,cell_support,input_support,target_rows)
    return PredictionResult(new,route,indices,diagnostic,model.output_unit)
