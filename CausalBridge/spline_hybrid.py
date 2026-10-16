"""Experimental bounded univariate cubic splines with C1 finite exponential tails.

Equal-bin normalized WT MSE + curvature/endpoint-slope penalty; no ridge or CV.
Compact natural-cubic interaction groups replace the whole response when supported.
"""
from dataclasses import dataclass, asdict, fields
import json
import itertools

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline, CubicSpline
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares
from scipy.special import expit, logit

from . import saturating_hybrid as core

SCHEMA = "saturating_hybrid_spline_target_v2"
SINGLE_SCHEMA = "bounded_spline_single_v1"
GROUP_SCHEMA = "bounded_spline_interaction_group_v1"
DEGREE = 3
QUADRATURE_ORDER = 12


@dataclass(frozen=True)
class SplineConfig:
    smoothness: float = 1e-6
    tail_scale: float = .5
    internal_knots: int = 2
    max_nfev: int = 1000
    restarts: int = 3
    seed: int = 42
    group_interaction_penalty: float = .01

    def __post_init__(self):
        for key in ("smoothness", "tail_scale", "group_interaction_penalty"):
            value = core._number(getattr(self, key), key)
            if value < 0 or key == "tail_scale" and value == 0:
                raise ValueError(f"invalid spline {key}")
            object.__setattr__(self, key, value)
        for key, minimum in (("internal_knots", 0), ("max_nfev", 1), ("restarts", 1), ("seed", 0)):
            object.__setattr__(self, key, core._integer(getattr(self, key), key, minimum))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("unknown/malformed spline_response settings")
        return cls(**value)

    def to_dict(self):
        return asdict(self)


def _basis(knots):
    return BSpline(knots, np.eye(len(knots) - DEGREE - 1), DEGREE, extrapolate=False)


def _knots(x, internal_knots):
    a, b = float(x.min()), float(x.max())
    z = (x - a) / (b - a)
    interior = np.unique(np.quantile(z, np.arange(1, internal_knots + 1) / (internal_knots + 1))) if internal_knots else np.array([])
    interior = interior[(interior > 0) & (interior < 1)]
    return np.r_[np.zeros(DEGREE + 1), interior, np.ones(DEGREE + 1)]


def _design(x, x_range, knots, tail_scale):
    a, b = x_range
    z = (np.asarray(x, dtype=float).reshape(-1) - a) / (b - a)
    basis = _basis(knots)
    B = basis(np.clip(z, 0, 1))
    left, right = z < 0, z > 1
    with np.errstate(over="ignore", under="ignore"):
        B[left] = basis(0) + tail_scale * np.expm1(z[left, None] / tail_scale) * basis.derivative()(0)
        B[right] = basis(1) - tail_scale * np.expm1(-(z[right, None] - 1) / tail_scale) * basis.derivative()(1)
    return B


def _penalty(knots, tail_scale):
    basis = _basis(knots)
    q, w = leggauss(QUADRATURE_ORDER)
    blocks = []
    nodes = np.unique(knots)
    for a, b in zip(nodes[:-1], nodes[1:]):
        z = (a + b) / 2 + (b - a) / 2 * q
        blocks.append(basis.derivative(2)(z) * np.sqrt(w * (b - a) / 2)[:, None])
    return np.vstack(blocks + [basis.derivative()(0)[None, :] / np.sqrt(2 * tail_scale),
                              basis.derivative()(1)[None, :] / np.sqrt(2 * tail_scale)])


class SplineFitFailure(FloatingPointError):
    def __init__(self, starts):
        self.starts = starts
        self.nfev = sum(s["nfev"] for s in starts)
        super().__init__("spline: no successful finite candidate; exhausted/failed starts are not accepted: " + json.dumps(starts))


@dataclass(frozen=True)
class SplineSingleModel:
    knots: np.ndarray
    coefficients: np.ndarray
    x_range: tuple
    bounds: tuple
    tail_scale: float
    config: dict
    metrics: dict
    schema: str = SINGLE_SCHEMA

    def __post_init__(self):
        if self.schema != SINGLE_SCHEMA:
            raise ValueError("unknown spline component schema")
        cfg = SplineConfig.from_dict(core._plain(self.config))
        knots = core._array(self.knots, "spline knots", 1)
        theta = core._array(self.coefficients, "spline coefficients", 1)
        xr = tuple(core._number(x, "spline WT range") for x in self.x_range)
        bounds = core._bounds(self.bounds)
        if len(xr) != 2 or xr[0] > xr[1] or self.tail_scale != cfg.tail_scale:
            raise ValueError("spline range/tail configuration mismatch")
        if (len(knots) < 8 or len(theta) != len(knots) - 4 or
                not np.array_equal(knots[:4], np.zeros(4)) or not np.array_equal(knots[-4:], np.ones(4)) or
                np.any(np.diff(knots) < 0) or len(knots[4:-4]) > cfg.internal_knots or
                np.any(knots[4:-4] <= 0) or np.any(knots[4:-4] >= 1) or
                len(np.unique(knots[4:-4])) != len(knots[4:-4])):
            raise ValueError("invalid spline knot/coefficient dimensions")
        m = core._plain(self.metrics)
        if m.get("status") not in ("ok", "constant_input", "constant_output"):
            raise ValueError("invalid spline fit status")
        if m.get("status") == "ok" and (xr[0] == xr[1] or bounds[0] == bounds[1] or
                m.get("solver_converged") is not True or m.get("termination_status") not in (1, 2, 3, 4)):
            raise ValueError("spline requires successful solver evidence")
        if m.get("status") == "ok":
            core._integer(m.get("nfev"), "spline nfev", 1)
            core._number(m.get("best_objective"), "spline objective")
            starts = m.get("starts")
            if (m["nfev"] > cfg.max_nfev or not isinstance(starts, list) or len(starts) != cfg.restarts or
                    not any(s.get("success") is True and s.get("finite") is True and
                            s.get("status") == m["termination_status"] and s.get("nfev") == m["nfev"] and
                            s.get("objective") == m["best_objective"] for s in starts) or
                    m.get("smoothness") != cfg.smoothness or m.get("tail_scale") != cfg.tail_scale or m.get("ridge") != 0.):
                raise ValueError("spline configuration/solver evidence mismatch")
        core._number(m.get("training_rmse"), "spline training RMSE")
        if m.get("training_r2") is not None and core._number(m["training_r2"], "spline R2") > 1 + 1e-12:
            raise ValueError("invalid spline R2")
        core._integer(m.get("n_rows"), "spline rows", 1)
        if m.get("status") != "ok":
            value = core._number(m.get("constant_value"), "spline constant")
            if not bounds[0] <= value <= bounds[1] or np.any(theta != 0) or len(theta) != 4:
                raise ValueError("invalid spline constant representation")
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "coefficients", theta)
        object.__setattr__(self, "x_range", xr)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "config", core._freeze(cfg.to_dict()))
        object.__setattr__(self, "metrics", core._freeze(m))

    def predict(self, x):
        x = core._array(x, "spline query")
        if self.metrics["status"] != "ok":
            return np.full(x.shape, self.metrics["constant_value"], dtype=float)
        value = _design(x, self.x_range, self.knots, self.tail_scale) @ self.coefficients
        L, U = self.bounds
        return (L + (U - L) * expit(value)).reshape(x.shape)

    def to_dict(self):
        return dict(response_type="spline", knots=self.knots.tolist(), coefficients=self.coefficients.tolist(),
                    x_range=list(self.x_range), bounds=list(self.bounds), tail_scale=self.tail_scale,
                    config=core._plain(self.config), metrics=core._plain(self.metrics), schema=self.schema)


def fit_single(x, y, bounds, cfg=SplineConfig()):
    x, y = core._array(x, "spline X", 1), core._array(y, "spline y", 1)
    L, U = core._bounds(bounds)
    if len(x) != len(y) or not len(y) or (y < L).any() or (y > U).any():
        raise ValueError("invalid spline WT rows/bounds")
    xr = (float(x.min()), float(x.max()))
    if xr[0] == xr[1] or np.ptp(y) == 0 or L == U:
        value = float(np.clip(y.mean(), L, U))
        m = dict(core._metrics(y, np.full(len(y), value)),
                 status="constant_input" if xr[0] == xr[1] else "constant_output",
                 constant_value=value, solver_converged=False, identifiable=False,
                 termination_status="not_needed", nfev=0)
        return SplineSingleModel(np.r_[np.zeros(4), np.ones(4)], np.zeros(4), xr, (L, U), cfg.tail_scale, cfg.to_dict(), m)
    knots = _knots(x, cfg.internal_knots)
    B = _design(x, xr, knots, cfg.tail_scale)
    P = np.sqrt(cfg.smoothness) * _penalty(knots, cfg.tail_scale)
    yy = (y - L) / (U - L)
    initial = np.linalg.lstsq(B, logit(np.clip(yy, 1e-4, 1-1e-4)), rcond=None)[0]
    rng = np.random.default_rng(cfg.seed)
    starts = [initial] + [initial + rng.normal(0, .1, len(initial)) for _ in range(cfg.restarts - 1)]
    def residual(theta):
        return np.r_[(expit(B @ theta) - yy) / np.sqrt(len(y)), P @ theta]
    def jacobian(theta):
        p = expit(B @ theta)
        return np.vstack([p[:, None] * (1-p[:, None]) * B / np.sqrt(len(y)), P])
    candidates, evidence = [], []
    for start in starts:
        try:
            result = least_squares(residual, start, jac=jacobian, max_nfev=cfg.max_nfev)
        except (FloatingPointError, OverflowError):
            evidence.append(dict(success=False, finite=False, status="numeric_failure", nfev=0))
            continue
        finite = bool(np.isfinite(result.x).all() and np.isfinite(result.fun).all())
        evidence.append(dict(success=bool(result.success), finite=finite, status=int(result.status),
                             nfev=int(result.nfev), objective=float(np.sum(result.fun**2)) if finite else None))
        if result.success and finite:
            candidates.append(result)
    if not candidates:
        raise SplineFitFailure(evidence)
    best = min(candidates, key=lambda r: (float(np.sum(r.fun**2)), tuple(r.x)))
    prediction = L + (U-L) * expit(B @ best.x)
    m = dict(core._metrics(y, prediction), status="ok", solver_converged=True, identifiable=False,
             termination_status=int(best.status), nfev=int(best.nfev), sum_nfev=sum(s["nfev"] for s in evidence),
             starts=evidence, best_objective=float(np.sum(best.fun**2)),
             smoothness=cfg.smoothness, ridge=0., tail_scale=cfg.tail_scale,
             objective="equal_bin_normalized_MSE + smoothness*(integral_g_second_derivative_squared + endpoint_slopes_squared/(2*tau)); no_ridge_no_CV", 
             parameter_profile="experimental smoothness; no validation or identifiability claim")
    return SplineSingleModel(knots, best.x, xr, (L, U), cfg.tail_scale, cfg.to_dict(), m)


def _natural(z, median, tau, order=0):
    """Two nonconstant natural cardinal columns; exact C1 exponential tails."""
    z = np.asarray(z, dtype=float)
    cs = CubicSpline([0., median, 1.], np.eye(3)[:, 1:], bc_type="natural")
    t = np.clip(z, 0, 1)
    B = cs(t, nu=order)
    for mask, edge, sign in ((z < 0, 0., 1.), (z > 1, 1., -1.)):
        u = z[mask] if edge == 0 else 1-z[mask]
        e = np.exp(u/tau)
        if order == 0:
            B[mask] = cs(edge) + sign*tau*cs(edge, 1)*np.expm1(u[:, None]/tau)
        elif order == 1:
            B[mask] = e[:, None]*cs(edge, 1)
        else:
            B[mask] = sign/tau*e[:, None]*cs(edge, 1)
    return B


def _linear_tail(z, tau, order=0):
    z = np.asarray(z, dtype=float)
    if order == 0:
        return np.where(z < 0, tau*np.expm1(np.minimum(z, 0)/tau),
                        np.where(z > 1, 1-tau*np.expm1(-np.maximum(z-1, 0)/tau), z))
    e = np.exp(-np.maximum(np.maximum(-z, z-1), 0)/tau)
    return e if order == 1 else np.where(z < 0, e/tau, np.where(z > 1, -e/tau, 0.))


def _group_design(z, medians, centers, qcenters, tau, derivative=None):
    """Natural additive terms plus centered pair and triple linear interactions."""
    z = np.asarray(z, dtype=float)
    n, d = z.shape
    orders = np.zeros(d, dtype=int) if derivative is None else np.asarray(derivative)
    columns = [np.ones(n) if not orders.any() else np.zeros(n)]
    for j in range(d):
        value = _natural(z[:, j], medians[j], tau, int(orders[j]))
        if orders[j] == 0:
            value = value-centers[j]
        if np.any(np.delete(orders, j)):
            value = np.zeros_like(value)
        columns.extend(value.T)
    for size in (2, 3):
        for indices in itertools.combinations(range(d), size):
            value = np.ones(n)
            if any(orders[j] for j in range(d) if j not in indices):
                value *= 0
            for j in indices:
                q = _linear_tail(z[:, j], tau, int(orders[j]))
                if orders[j] == 0:
                    q = q-qcenters[j]
                value *= q
            columns.append(value)
    return np.column_stack(columns)


def _group_penalty(medians, centers, qcenters, tau):
    """Tensor quadrature of full Hessian Frobenius norm and face normal slopes."""
    d = len(medians)
    nodes, weights = [], []
    q, w = leggauss(4)
    for median in medians:
        nodes.append(np.r_[median/2*(q+1), median+(1-median)/2*(q+1)])
        weights.append(np.r_[median/2*w, (1-median)/2*w])
    def tensor(n, w):
        values = np.array(list(itertools.product(*n)))
        mass = np.prod(np.array(list(itertools.product(*w))), axis=1)
        return values, np.sqrt(mass)
    Z, sw = tensor(nodes, weights)
    blocks = []
    for j in range(d):
        for k in range(j, d):
            order = np.zeros(d, int); order[j] += 1; order[k] += 1
            blocks.append(_group_design(Z, medians, centers, qcenters, tau, order)*sw[:, None]*np.sqrt(1 if j == k else 2))
    for j in range(d):
        for edge in (0., 1.):
            n = list(nodes); w = list(weights); n[j] = np.array([edge]); w[j] = np.array([1.])
            Z, sw = tensor(n, w); order = np.zeros(d, int); order[j] = 1
            blocks.append(_group_design(Z, medians, centers, qcenters, tau, order)*sw[:, None]/np.sqrt(2*tau))
    return np.vstack(blocks)


class SplineGroupNumericalError(FloatingPointError, ValueError):
    """A valid candidate cannot be represented with reliable float64 geometry."""


def _standardization_tolerance(mean, scale, x_ranges, n_rows):
    # Subtracting a large origin before dividing by a narrow scale amplifies
    # coordinate roundoff. Include the reduction error of the original mean;
    # do not use a fixed tolerance in standardized coordinates.
    magnitude = np.maximum(np.abs(mean), np.max(np.abs(x_ranges), axis=1))
    eps = np.finfo(float).eps
    tolerance = 8 * eps * max(n_rows, 1) * (1 + magnitude / scale)
    if not np.isfinite(tolerance).all() or (tolerance > 1e-4).any():
        raise SplineGroupNumericalError("spline group geometry loses float64 precision")
    return tolerance


@dataclass(frozen=True)
class SplineGroupModel:
    indices: tuple
    coefficients: np.ndarray
    x_ranges: np.ndarray
    medians: np.ndarray
    centers: np.ndarray
    qcenters: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    reference_u: np.ndarray
    support_rows: np.ndarray
    branch_ids: tuple
    radius_by_branch: dict
    bounds: tuple
    config: dict
    diagnostics: dict
    schema: str = GROUP_SCHEMA

    @property
    def n_parameters(self):
        return 6 if len(self.indices) == 2 else 11

    def __post_init__(self):
        indices = tuple(core._integer(i, "spline group index", 0) for i in self.indices)
        d = len(indices)
        if self.schema != GROUP_SCHEMA or d not in (2, 3) or len(set(indices)) != d:
            raise ValueError("invalid spline group schema/indices")
        cfg = SplineConfig.from_dict(core._plain(self.config))
        arrays = {}
        for name, ndim in (("coefficients",1),("x_ranges",2),("medians",1),("centers",2),("qcenters",1),
                           ("mean",1),("scale",1),("reference_u",2),("support_rows",1)):
            arrays[name] = core._array(getattr(self,name), name, ndim, name == "support_rows")
        if (arrays['coefficients'].shape != (self.n_parameters,) or arrays['x_ranges'].shape != (d,2) or
                np.any(np.diff(arrays['x_ranges'],axis=1)<=0) or arrays['centers'].shape != (d,2) or
                any(arrays[k].shape != (d,) for k in ('medians','qcenters','mean','scale')) or
                (arrays['scale']<=0).any() or (arrays['medians']<=0).any() or (arrays['medians']>=1).any()):
            raise ValueError("invalid spline group parameter dimensions")
        ref = arrays['reference_u']; rows = arrays['support_rows']; branches = core._branches(self.branch_ids,len(ref))
        if ref.shape != (len(rows),d) or len(set(rows.tolist())) != len(rows) or (rows<0).any():
            raise ValueError("invalid spline group support")
        tolerance = _standardization_tolerance(arrays['mean'], arrays['scale'], arrays['x_ranges'], len(ref))
        if (np.any(np.abs(ref.mean(0)) > tolerance) or
                np.any(np.abs(ref.std(0)-1) > tolerance)):
            raise ValueError("spline group reference geometry not standardized")
        radii = {str(k):core._number(v,'radius') for k,v in self.radius_by_branch.items()}
        if set(radii) != set(branches) or any(v<=0 for v in radii.values()):
            raise ValueError("invalid spline group branch radii")
        X = ref*arrays['scale']+arrays['mean']; xr = arrays['x_ranges']; z=(X-xr[:,0])/(xr[:,1]-xr[:,0])
        if (not np.allclose(X.min(0),xr[:,0],atol=1e-12,rtol=1e-12) or
                not np.allclose(X.max(0),xr[:,1],atol=1e-12,rtol=1e-12) or
                not np.allclose(np.median(z,axis=0),arrays['medians'],atol=1e-12) or
                not np.allclose(z.mean(0),arrays['qcenters'],atol=1e-12) or
                not np.allclose(np.array([_natural(z[:,j],arrays['medians'][j],cfg.tail_scale).mean(0) for j in range(d)]),arrays['centers'],atol=1e-12)):
            raise ValueError("spline group natural basis evidence mismatch")
        A=_group_design(z,arrays['medians'],arrays['centers'],arrays['qcenters'],cfg.tail_scale)
        if np.linalg.matrix_rank(A)!=self.n_parameters:
            raise ValueError("rank deficient spline group design")
        P=_group_penalty(arrays['medians'],arrays['centers'],arrays['qcenters'],cfg.tail_scale)
        shape=float(np.sum((P@arrays['coefficients'])**2));interaction=float(np.sum(arrays['coefficients'][1+2*d:]**2))
        diag=core._plain(self.diagnostics)
        bounds=core._bounds(self.bounds)
        predictions=bounds[0]+(bounds[1]-bounds[0])*expit(A@arrays['coefficients'])
        if (diag.get('n_parameters')!=self.n_parameters or
                diag.get('training_prediction_fingerprint')!=core._digest(np.round(predictions,10)) or
                len(diag.get('nfev',[]))!=cfg.restarts or
                any(core._integer(v,'spline group nfev',1)>cfg.max_nfev for v in diag['nfev'])):
            raise ValueError("spline group parameter/training prediction evidence mismatch")
        if not np.isclose(diag.get('shape_penalty',-1),shape,rtol=1e-10,atol=1e-12) or not np.isclose(diag.get('interaction_square',-1),interaction,rtol=1e-10,atol=1e-12):
            raise ValueError("spline group penalty evidence mismatch")
        for k,v in arrays.items():object.__setattr__(self,k,v)
        object.__setattr__(self,'indices',indices);object.__setattr__(self,'branch_ids',branches)
        object.__setattr__(self,'bounds',core._bounds(self.bounds));object.__setattr__(self,'config',core._freeze(cfg.to_dict()))
        object.__setattr__(self,'diagnostics',core._freeze(diag));object.__setattr__(self,'radius_by_branch',core._freeze(radii))

    def predict(self,X):
        X=core._array(X,'spline group query',2)
        if X.shape[1]!=len(self.indices):raise ValueError('spline group query dimensions')
        z=(X-self.x_ranges[:,0])/(self.x_ranges[:,1]-self.x_ranges[:,0])
        A=_group_design(z,self.medians,self.centers,self.qcenters,self.config['tail_scale'])
        L,U=self.bounds
        return L+(U-L)*expit(A@self.coefficients)

    def to_dict(self):
        return dict(response_type='spline',**core._plain({f.name:getattr(self,f.name) for f in fields(self)}))


def fit_group(data,indices,baseline,bounds,config,cfg):
    # Numerical failures of optional groups must not discard executable singles.
    # Codec/contract ValueErrors and programming errors intentionally propagate.
    try:
        return _fit_group_impl(data,indices,baseline,bounds,config,cfg)
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        return None, dict(indices=list(indices),accepted=False,response_family='spline',
                          reason='group_numerical_failure',diagnostic=str(exc))


def _fit_group_impl(data,indices,baseline,bounds,config,cfg):
    d=len(indices);p=6 if d==2 else 11
    diagnostic=dict(indices=list(indices),accepted=False,response_family='spline',n_parameters=p,
                    training_evidence='all_WT_training_rows_not_validation')
    def fail(reason):diagnostic['reason']=reason;return None,diagnostic
    required=max(config.min_group_effective_rows,2*p)
    if len(data.y)<max(config.min_group_rows,2*p):return fail('insufficient_group_rows')
    if data.cell_weights is None:return fail('training_cell_support_missing')
    W=data.cell_weights[data.support_rows]
    diagnostic.update(pooled_unique_cell_ess=core._cell_ess(W),kernel_effective_rows=core._effective_rows(W),effective_rows_required=required)
    if diagnostic['pooled_unique_cell_ess']<config.min_group_cell_ess:return fail('training_cell_ess')
    if diagnostic['kernel_effective_rows']<required:return fail('kernel_effective_rows')
    X=data.X[:,indices];mean=X.mean(0);scale=X.std(0);xr=np.column_stack([X.min(0),X.max(0)])
    if (scale<=0).any():return fail('constant_group_input')
    _standardization_tolerance(mean,scale,xr,len(X))
    u=(X-mean)/scale;condition=float(np.linalg.cond(u));diagnostic['geometry_condition']=condition
    if not np.isfinite(condition) or condition>config.max_geometry_condition or np.linalg.matrix_rank(u)!=d:return fail('rank_deficient_or_ill_conditioned_geometry')
    z=(X-xr[:,0])/(xr[:,1]-xr[:,0]);medians=np.median(z,axis=0)
    if (medians<=0).any() or (medians>=1).any():return fail('degenerate_natural_knot')
    centers=np.array([_natural(z[:,j],medians[j],cfg.tail_scale).mean(0) for j in range(d)]);qc=z.mean(0)
    A=_group_design(z,medians,centers,qc,cfg.tail_scale)
    if np.linalg.matrix_rank(A)!=p:return fail('rank_deficient_spline_design')
    radii={}
    for branch in sorted(set(data.branch_ids)):
        ref=u[np.array(data.branch_ids)==branch];centered=ref-ref.mean(0)
        if len(ref)<d+2 or np.linalg.matrix_rank(centered)!=d or np.linalg.cond(centered)>config.max_geometry_condition:return fail('rank_deficient_branch_geometry')
        dist=cdist(ref,ref);np.fill_diagonal(dist,np.inf);radius=float(np.quantile(dist.min(1),config.support_radius_quantile))
        if not np.isfinite(radius) or radius<=0:return fail('degenerate_support_radius')
        radii[branch]=radius
    L,U=bounds
    if U==L:return fail('constant_output_bounds')
    P=_group_penalty(medians,centers,qc,cfg.tail_scale);I=np.eye(p)[1+2*d:]
    R=np.vstack([np.sqrt(cfg.smoothness)*P,np.sqrt(cfg.group_interaction_penalty)*I])
    yy=(data.y-L)/(U-L);initial=np.linalg.lstsq(A,logit(np.clip(yy,1e-4,1-1e-4)),rcond=None)[0]
    rng=np.random.default_rng(cfg.seed);starts=[initial]+[initial+rng.normal(0,.1,p) for _ in range(cfg.restarts-1)]
    def residual(t):return np.r_[(expit(A@t)-yy)/np.sqrt(len(yy)),R@t]
    def jac(t):
        v=expit(A@t);return np.vstack([v[:,None]*(1-v[:,None])*A/np.sqrt(len(yy)),R])
    fits=[least_squares(residual,t,jac=jac,max_nfev=cfg.max_nfev) for t in starts]
    diagnostic['solver_statuses']=[int(f.status) for f in fits]
    if not all(f.success and np.isfinite(f.x).all() and np.isfinite(f.fun).all() for f in fits):return fail('group_solver_not_converged')
    predictions=np.array([L+(U-L)*expit(A@f.x) for f in fits]);spread=float(np.ptp(predictions,axis=0).max()/(U-L));diagnostic['restart_prediction_spread']=spread
    if spread>config.restart_prediction_spread_max:return fail('restart_prediction_spread')
    best=min(range(len(fits)),key=lambda i:(float(np.sum(fits[i].fun**2)),tuple(fits[i].x)));coef=fits[best].x
    mse=float(np.mean((predictions[best]-data.y)**2));base=float(np.mean((baseline-data.y)**2));shape=float(np.sum((P@coef)**2));interaction=float(np.sum((I@coef)**2))
    J=mse+(U-L)**2*(cfg.smoothness*shape+cfg.group_interaction_penalty*interaction)+p*base/len(data.y)
    gain=(base-J)/base if base>0 else -1.
    diagnostic.update(n_rows=len(data.y),training_rmse=np.sqrt(mse),baseline_mse=base,shape_penalty=shape,interaction_square=interaction,
                      penalized_group_mse=J,penalized_gain=gain,nfev=[int(f.nfev) for f in fits],
                      training_prediction_fingerprint=core._digest(np.round(predictions[best],10)))
    if gain<config.min_penalized_gain:return fail('insufficient_penalized_gain')
    diagnostic['accepted']=True
    return SplineGroupModel(tuple(indices),coef,xr,medians,centers,qc,mean,scale,u,data.support_rows,data.branch_ids,radii,bounds,cfg.to_dict(),diagnostic),diagnostic
