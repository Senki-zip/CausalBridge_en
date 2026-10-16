"""Independent bounded WT response networks. Torch is imported only for fitting.

Inference and immutable model validation use NumPy. There is no holdout: best
snapshots minimize all-WT normalized MSE plus mean-square parameter penalty.
"""
from dataclasses import dataclass, fields
from collections.abc import Mapping
import multiprocessing
import numpy as np
from scipy.special import expit
from scipy.spatial.distance import cdist
from . import saturating_hybrid as core

SCHEMA = "saturating_hybrid_nn_target_v1"
_WORKER_TORCH_INITIALIZED = False


def initialize_worker_torch():
    """Configure only spawned NN fit workers, once; never alter parent interop."""
    global _WORKER_TORCH_INITIALIZED
    if not _WORKER_TORCH_INITIALIZED:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        _WORKER_TORCH_INITIALIZED = True


@dataclass(frozen=True)
class NNConfig:
    hidden_width: int = 4
    epochs: int = 300
    lr: float = .01
    l2: float = .01
    seed: int = 42
    restarts: int = 1

    def __post_init__(self):
        for name in ("hidden_width", "epochs", "restarts", "seed"):
            object.__setattr__(self, name, core._integer(getattr(self, name), name, 0 if name == "seed" else 1))
        for name in ("lr", "l2"):
            value = core._number(getattr(self, name), name)
            if value < 0 or name == "lr" and value == 0:
                raise ValueError(f"invalid NN {name}")
            object.__setattr__(self, name, value)
        if self.restarts > 3:
            raise ValueError("NN restarts must be 1, 2 or 3")

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or set(value) - {f.name for f in fields(cls)}:
            raise ValueError("unknown NNConfig keys")
        return cls(**dict(value))


def _network(X, mean, scale, W, bias, v, offset, bounds):
    u = (np.asarray(X, dtype=float) - mean) / scale
    with np.errstate(over="ignore"):
        hidden = np.tanh(u @ W.T + bias)
        return bounds[0] + (bounds[1] - bounds[0]) * expit(hidden @ v + offset)


def _validate_network(model, width):
    arrays = {name: core._array(getattr(model, name), name, ndim) for name, ndim in
              (("mean", 1), ("scale", 1), ("W", 2), ("bias", 1), ("v", 1))}
    h = len(arrays["bias"])
    if h < 1 or arrays["W"].shape != (h, width) or arrays["v"].shape != (h,) or arrays["mean"].shape != (width,) or arrays["scale"].shape != (width,) or (arrays["scale"] <= 0).any():
        raise ValueError("invalid NN dimensions/normalization")
    for name, value in arrays.items():
        object.__setattr__(model, name, value)
    object.__setattr__(model, "offset", core._number(model.offset, "offset"))
    object.__setattr__(model, "bounds", core._bounds(model.bounds))


@dataclass(frozen=True)
class NNSingleModel:
    W: np.ndarray
    bias: np.ndarray
    v: np.ndarray
    offset: float
    mean: np.ndarray
    scale: np.ndarray
    bounds: tuple
    x_range: tuple
    metrics: Mapping

    def __post_init__(self):
        _validate_network(self, 1)
        xr = tuple(core._number(v, "x_range") for v in self.x_range)
        if len(xr) != 2 or xr[0] > xr[1]:
            raise ValueError("invalid NN range")
        m = self.metrics
        if not isinstance(m, Mapping):
            raise ValueError("NN metrics must be a mapping")
        core._integer(m.get("n_rows"), "n_rows", 1)
        if core._number(m.get("training_rmse"), "RMSE") < 0:
            raise ValueError("invalid NN RMSE")
        if m.get("training_r2") is not None and core._number(m["training_r2"], "R2") > 1:
            raise ValueError("invalid NN R2")
        if m.get("status") not in ("ok", "constant_input", "constant_output") or m.get("solver_converged") is not False:
            raise ValueError("invalid NN training evidence")
        if m["status"] == "ok":
            if m.get("training_status") != "fixed_epochs_completed":
                raise ValueError("invalid NN epoch status")
            core._integer(m.get("epochs"), "epochs", 1)
            core._integer(m.get("restarts"), "restarts", 1)
            if core._number(m.get("best_objective"), "best_objective") < 0:
                raise ValueError("invalid NN objective")
        elif m.get("training_status") != "not_needed":
            raise ValueError("invalid constant NN training evidence")
        if m.get("identifiable") is not False:
            raise ValueError("invalid NN identifiability evidence")
        object.__setattr__(self, "x_range", xr)
        object.__setattr__(self, "metrics", core._freeze(m))

    def predict(self, x):
        x = core._array(x, "NN query")
        return _network(x.reshape(-1, 1), self.mean, self.scale, self.W, self.bias, self.v, self.offset, self.bounds).reshape(x.shape)

    def to_dict(self):
        return dict(response_type="nn", **core._plain({f.name: getattr(self, f.name) for f in fields(self)}))


@dataclass(frozen=True)
class NNGroupModel:
    indices: tuple
    W: np.ndarray
    bias: np.ndarray
    v: np.ndarray
    offset: float
    mean: np.ndarray
    scale: np.ndarray
    reference_u: np.ndarray
    branch_ids: tuple
    support_rows: np.ndarray
    radius_by_branch: Mapping
    bounds: tuple
    diagnostics: Mapping

    def __post_init__(self):
        indices = tuple(core._integer(i, "index") for i in self.indices)
        if len(indices) not in (2, 3) or len(set(indices)) != len(indices):
            raise ValueError("invalid NN group indices")
        _validate_network(self, len(indices))
        ref = core._array(self.reference_u, "reference_u", 2)
        rows = core._array(self.support_rows, "support_rows", 1, True)
        branches = core._branches(self.branch_ids, len(ref))
        if ref.shape[1] != len(indices) or len(rows) != len(ref) or (rows < 0).any() or len(set(rows.tolist())) != len(rows) or not np.allclose(ref.mean(0), 0, atol=1e-10) or not np.allclose(ref.std(0), 1, atol=1e-10):
            raise ValueError("invalid NN group reference")
        if set(self.radius_by_branch) != set(branches) or any(core._number(r, "radius") <= 0 for r in self.radius_by_branch.values()):
            raise ValueError("invalid NN support radii")
        if not isinstance(self.diagnostics, Mapping):
            raise ValueError("NN diagnostics must be a mapping")
        for name in ("epochs", "restarts", "n_rows"):
            core._integer(self.diagnostics.get(name), name, 1)
        for name in ("training_rmse", "baseline_mse", "penalized_group_mse", "effective_rows_required"):
            if core._number(self.diagnostics.get(name), name) < 0:
                raise ValueError("invalid NN group evidence")
        r2 = self.diagnostics.get("training_r2")
        if r2 is not None and core._number(r2, "training_r2") > 1:
            raise ValueError("invalid NN group R2")
        for name, value in dict(indices=indices, reference_u=ref, support_rows=rows, branch_ids=branches,
                                radius_by_branch=core._freeze(self.radius_by_branch), diagnostics=core._freeze(self.diagnostics)).items():
            object.__setattr__(self, name, value)

    @property
    def n_parameters(self):
        return self.W.size + self.bias.size + self.v.size + 1

    def predict(self, X):
        X = core._array(X, "NN group query", 2)
        if X.shape[1] != len(self.indices):
            raise ValueError("NN group query width mismatch")
        return _network(X, self.mean, self.scale, self.W, self.bias, self.v, self.offset, self.bounds)

    def to_dict(self):
        return dict(response_type="nn", **core._plain({f.name: getattr(self, f.name) for f in fields(self)}))


def _seed(key, seed, restart):
    text = core._plain([key, seed, restart])
    return int(core._digest(text)[:16], 16) % (2**63 - 1)


def _train(X, y, keys, bounds, cfg):
    """p x rows x inputs; every Adam coordinate and objective is independent."""
    import torch
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        p, n, d = X.shape
        mean, scale = X.mean(1), X.std(1)
        scale = np.where(scale > 0, scale, 1.)
        xx = torch.tensor((X - mean[:, None, :]) / scale[:, None, :], dtype=torch.float64)
        yy = torch.tensor((y - bounds[0]) / (bounds[1] - bounds[0]), dtype=torch.float64)
        best_loss = np.full(p, np.inf)
        best = [None] * p
        restart_predictions = []
        for restart in range(cfg.restarts):
            initial = []
            for key in keys:
                gen = torch.Generator(device="cpu").manual_seed(_seed(key, cfg.seed, restart))
                initial.append([torch.randn(shape, generator=gen, dtype=torch.float64) * .3 for shape in
                                ((cfg.hidden_width, d), (cfg.hidden_width,), (cfg.hidden_width,), ())])
            params = [torch.nn.Parameter(torch.stack([edge[j] for edge in initial])) for j in range(4)]
            optimizer = torch.optim.Adam(params, lr=cfg.lr)
            local_loss = np.full(p, np.inf)
            local = [None] * p
            failed = np.zeros(p, dtype=bool)
            def forward():
                W, b, v, c = params
                hidden = torch.tanh((xx[:, :, None, :] * W[:, None, :, :]).sum(-1) + b[:, None, :])
                return torch.sigmoid((hidden * v[:, None, :]).sum(-1) + c[:, None])
            for epoch in range(cfg.epochs + 1):
                pred = forward()
                penalty = sum(t.reshape(p, -1).square().sum(1) for t in params) / (cfg.hidden_width * (d + 2) + 1)
                loss = (pred - yy[None, :]).square().mean(1) + cfg.l2 * penalty
                values = loss.detach().numpy()
                failed |= ~np.isfinite(values)
                for i in np.flatnonzero((values < local_loss) & ~failed):
                    local_loss[i] = values[i]
                    local[i] = [t[i].detach().numpy().copy() for t in params]
                if epoch == cfg.epochs:
                    break
                optimizer.zero_grad()
                # Summation preserves each edge's gradient scale for any p.
                torch.where(torch.tensor(~failed), loss, torch.zeros_like(loss)).sum().backward()
                for t in params:
                    t.grad[torch.tensor(failed)] = 0
                optimizer.step()
            predictions = []
            for i in range(p):
                if failed[i]:
                    local[i] = None
                if local[i] is None:
                    predictions.append(None)
                    continue
                W, b, v, c = local[i]
                predictions.append(_network(X[i], mean[i], scale[i], W, b, v, float(c), bounds))
                if local_loss[i] < best_loss[i]:
                    best_loss[i], best[i] = local_loss[i], local[i]
            restart_predictions.append(predictions)
        return best, mean, scale, best_loss, restart_predictions
    finally:
        if multiprocessing.current_process().name == "MainProcess":
            torch.set_num_threads(old_threads)


def fit_singles(X, y, keys, bounds, cfg):
    X, y = core._array(X, "X", 2), core._array(y, "y", 1)
    bounds = core._bounds(bounds)
    if X.shape != (len(y), len(keys)) or (y < bounds[0]).any() or (y > bounds[1]).any():
        raise ValueError("invalid NN training arrays/bounds")
    result = [None] * len(keys)
    active = []
    for i in range(len(keys)):
        if np.ptp(X[:, i]) == 0 or np.ptp(y) == 0 or bounds[0] == bounds[1]:
            value = float(y.mean())
            ratio = (value - bounds[0]) / (bounds[1] - bounds[0]) if bounds[1] > bounds[0] else .5
            # Exact boundary constants are represented by a saturated sigmoid.
            offset = -1000. if ratio <= 0 else 1000. if ratio >= 1 else float(np.log(ratio / (1-ratio)))
            m = dict(core._metrics(y, np.full(len(y), value)), status="constant_input" if np.ptp(X[:, i]) == 0 else "constant_output",
                     training_status="not_needed", solver_converged=False, identifiable=False)
            result[i] = NNSingleModel(np.zeros((cfg.hidden_width, 1)), np.zeros(cfg.hidden_width), np.zeros(cfg.hidden_width), offset,
                np.array([X[:, i].mean()]), np.ones(1), bounds, (float(X[:, i].min()), float(X[:, i].max())), m)
        else:
            active.append(i)
    if active:
        trained, means, scales, losses, _ = _train(X[:, active].T[:, :, None], y, [keys[i] for i in active], bounds, cfg)
        for j, i in enumerate(active):
            if trained[j] is None:
                continue
            W, b, v, c = trained[j]
            pred = _network(X[:, i:i+1], means[j], scales[j], W, b, v, float(c), bounds)
            m = dict(core._metrics(y, pred), status="ok", training_status="fixed_epochs_completed", solver_converged=False,
                epochs=cfg.epochs, restarts=cfg.restarts, best_objective=float(losses[j]),
                objective="all_WT_normalized_MSE + l2*mean(all_parameters^2); no_holdout", identifiable=False)
            result[i] = NNSingleModel(W, b, v, float(c), means[j], scales[j], bounds,
                                     (float(X[:, i].min()), float(X[:, i].max())), m)
    return result


def fit_group(data, indices, baseline, bounds, config, cfg):
    d = dict(indices=list(indices), accepted=False, training_evidence="fixed_epochs_all_WT_not_validation")
    def fail(reason):
        d["reason"] = reason
        return None, d
    parameters = cfg.hidden_width * (len(indices) + 2) + 1
    required = max(config.min_group_effective_rows, 2 * parameters)
    if len(data.y) < max(config.min_group_rows, 2 * parameters):
        return fail("insufficient_group_rows")
    if data.cell_weights is None:
        return fail("training_cell_support_missing")
    weights = data.cell_weights[data.support_rows]
    d.update(pooled_unique_cell_ess=core._cell_ess(weights), kernel_effective_rows=core._effective_rows(weights), effective_rows_required=required)
    if d["pooled_unique_cell_ess"] < config.min_group_cell_ess:
        return fail("training_cell_ess")
    if d["kernel_effective_rows"] < required:
        return fail("kernel_effective_rows")
    X = data.X[:, indices]
    mean, scale = X.mean(0), X.std(0)
    if (scale <= 0).any():
        return fail("constant_group_input")
    u = (X - mean) / scale
    condition = float(np.linalg.cond(u))
    if not np.isfinite(condition) or condition > config.max_geometry_condition or np.linalg.matrix_rank(u) != len(indices):
        return fail("rank_deficient_or_ill_conditioned_geometry")
    radii = {}
    for branch in sorted(set(data.branch_ids)):
        ref = u[np.array(data.branch_ids) == branch]
        centered = ref - ref.mean(0)
        if len(ref) < len(indices)+2 or np.linalg.matrix_rank(centered) != len(indices) or np.linalg.cond(centered) > config.max_geometry_condition:
            return fail("rank_deficient_branch_geometry")
        distances = cdist(ref, ref); np.fill_diagonal(distances, np.inf)
        radius = float(np.quantile(distances.min(1), config.support_radius_quantile))
        if not np.isfinite(radius) or radius <= 0:
            return fail("degenerate_support_radius")
        radii[branch] = radius
    if bounds[0] == bounds[1]:
        return fail("constant_output_bounds")
    trained, _, _, losses, predictions = _train(X[None, :, :], data.y,
        [tuple(data.input_keys[i] for i in indices)], bounds, cfg)
    if trained[0] is None or any(p[0] is None for p in predictions):
        return fail("group_numeric_failure")
    spread = float(np.ptp(np.array([p[0] for p in predictions]), axis=0).max() / (bounds[1]-bounds[0]))
    d.update(restart_prediction_spread=spread, geometry_condition=condition,
             training_status="fixed_epochs_completed", restarts=cfg.restarts, epochs=cfg.epochs)
    if spread > config.restart_prediction_spread_max:
        return fail("restart_prediction_spread")
    W, b, v, c = trained[0]
    pred = _network(X, mean, scale, W, b, v, float(c), bounds)
    baseline_mse = float(np.mean((baseline-data.y)**2))
    penalized = float(losses[0])*(bounds[1]-bounds[0])**2 + parameters/len(data.y)*baseline_mse
    gain = 0. if baseline_mse == 0 else (baseline_mse-penalized)/baseline_mse
    d.update(core._metrics(data.y, pred), baseline_mse=baseline_mse, penalized_group_mse=penalized, penalized_gain=gain,
             penalty_definition="MSE + nn.l2*mean(all_parameters^2)*output_range^2 + n_params/n_rows*baseline_MSE")
    if gain < config.min_penalized_gain:
        return fail("insufficient_penalized_gain")
    d.update(accepted=True, reason="accepted")
    return NNGroupModel(indices, W, b, v, float(c), mean, scale, u, data.branch_ids,
                        data.support_rows, radii, bounds, d), d
