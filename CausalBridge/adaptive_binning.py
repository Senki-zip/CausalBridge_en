"""Standalone, WT-fitted binning; no integration with the production pipeline.

Usage::

    policy = fit_policy(times, ids, embedding=external_train_wt_embedding)
    rna = aggregate(policy.initial_plan, rna_matrix, cell_ids=ids)
    atac = aggregate(policy.initial_plan, atac_matrix, cell_ids=ids)
    heldout = build_plan(policy, new_times, new_ids)

The standalone default combines state-relative-to-noise bandwidths and
split-noise boundary masks (external aligned WT embedding, no internal PCA)::

    config = BinningConfig()  # adaptive_gaussian / separate / state / split_noise, cap 2
    policy = fit_policy(times, ids, embedding=external_train_wt_embedding, config=config)

Missing training embedding raises an error; there is no silent ESS fallback.
Legacy and ESS algorithms remain available by explicit selection, for example::

    legacy = BinningConfig(mode="hard", branch_policy="legacy",
                           bandwidth_selection="ess", boundary="none")
    ess = BinningConfig(bandwidth_selection="ess", boundary="none",
                        max_bandwidth_multiplier=.5)

State scores are a "no demonstrated excess distortion" heuristic, NOT a
confidence guarantee. Density affects pilot availability, estimation uncertainty,
and post-selection ESS reliability, never widening to an ESS target. Finite pilots and full Gaussian
tails cannot resolve every transition; results depend on embedding representation.

``hard/legacy`` and ``gaussian/legacy`` reproduce current qcut/aggregation
semantics. Legacy Gaussian deliberately mixes branches. Separate modes mask
branches and reject insufficient support. Policies never fit on query data.
Membership is a forward assignment, NOT an inverse of aggregation.

For forward reconstruction of bin estimates (which may contain NaN for invalid
training bins), select valid columns/rows BEFORE multiplying; dense zero * NaN
would otherwise contaminate supported queries::

    membership, query_valid = query_membership(policy, new_times, new_ids)
    train_valid = policy.initial_plan.valid
    reconstructed = membership[:, train_valid] @ rna[train_valid]
    reconstructed[~query_valid] = np.nan

Array fields are owned, read-only copies (including CSR buffers). Dataclasses
are frozen, but this is not a security/immutability guarantee: Python metadata
and scipy objects can still be deliberately mutated. No serialization/cache API.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

VERSION = "adaptive-binning-default-state-boundary-4"
EPSILON = 1e-10


def _owned(a, dtype=None):
    a = np.array(a, dtype=dtype, copy=True)
    a.setflags(write=False)
    return a


def _readonly_matrix(a):
    if sparse.issparse(a):
        a = a.tocsr(copy=True)
        a.sort_indices()
        for buf in (a.data, a.indices, a.indptr):
            buf.setflags(write=False)
        return a
    return _owned(a)


@dataclass(frozen=True)
class BinningConfig:
    mode: str = "adaptive_gaussian"
    branch_policy: str = "separate"
    max_bins: int = 100
    min_cells_per_bin: int = 5
    target_bins: Optional[int] = None
    overlap_factor: Optional[float] = None
    min_support_cells: int = 5
    target_ess: float = 40.0
    bandwidth_multipliers: Tuple[float, ...] = (.25, .5, 1., 2.)
    max_bandwidth_multiplier: float = 2.
    bandwidth_selection: str = "state"
    state_budget: float = 1.
    state_floor_fraction: float = .2
    state_uncertainty_multiplier: float = 2.
    state_pilot_subdivisions: int = 4
    state_min_pilot_cells: int = 4
    state_bootstrap_repeats: int = 32
    boundary: str = "split_noise"
    boundary_splits: int = 20
    boundary_min_side: int = 10
    boundary_stability: float = .8
    random_seed: int = 42
    max_kernel_elements: int = 2_000_000

    def __post_init__(self):
        if self.mode not in ("hard", "gaussian", "adaptive_gaussian"):
            raise ValueError("unknown binning mode")
        if self.branch_policy not in ("legacy", "separate"):
            raise ValueError("unknown branch policy")
        if self.mode == "adaptive_gaussian" and self.branch_policy != "separate":
            raise ValueError("adaptive_gaussian requires branch_policy='separate'")
        if self.bandwidth_selection not in ("ess", "state"):
            raise ValueError("unknown bandwidth selection")
        if self.bandwidth_selection == "state" and (self.mode != "adaptive_gaussian" or self.branch_policy != "separate"):
            raise ValueError("state selection requires adaptive_gaussian and separate branches")
        for name in ("state_budget", "state_floor_fraction", "state_uncertainty_multiplier"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.state_floor_fraction > 1:
            raise ValueError("state_floor_fraction must be in [0,1]")
        if self.boundary not in ("none", "split_noise"):
            raise ValueError("unknown boundary mode")
        if self.boundary != "none" and self.mode != "adaptive_gaussian":
            raise ValueError("boundary detection requires adaptive_gaussian")
        for name in ("max_bins", "min_cells_per_bin", "min_support_cells",
                     "max_kernel_elements", "boundary_splits", "boundary_min_side",
                     "state_pilot_subdivisions", "state_min_pilot_cells", "state_bootstrap_repeats"):
            v = getattr(self, name)
            if isinstance(v, bool) or not np.isfinite(v) or int(v) != v or v < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.state_min_pilot_cells < 3 or self.state_bootstrap_repeats < 2:
            raise ValueError("state pilots need >=3 cells and bootstrap needs >=2 repeats")
        if isinstance(self.random_seed, bool) or not np.isfinite(self.random_seed) or int(self.random_seed) != self.random_seed or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        for name in ("state_pilot_subdivisions", "state_min_pilot_cells", "state_bootstrap_repeats", "random_seed"):
            object.__setattr__(self, name, int(getattr(self, name)))
        if self.target_bins is not None and (int(self.target_bins) != self.target_bins or self.target_bins < 1):
            raise ValueError("target_bins must be positive")
        if self.overlap_factor is not None and (not np.isfinite(self.overlap_factor) or self.overlap_factor <= 0):
            raise ValueError("overlap_factor must be positive and finite")
        multipliers = tuple(float(x) for x in self.bandwidth_multipliers)
        object.__setattr__(self, "bandwidth_multipliers", multipliers)
        if (not multipliers or any(not np.isfinite(x) or x <= 0 for x in multipliers)
                or len(set(multipliers)) != len(multipliers)):
            raise ValueError("bandwidth candidates must be distinct, positive, finite")
        if not np.isfinite(self.max_bandwidth_multiplier) or self.max_bandwidth_multiplier <= 0:
            raise ValueError("maximum bandwidth multiplier must be positive")
        if not any(x <= self.max_bandwidth_multiplier for x in multipliers):
            raise ValueError("no bandwidth candidate is allowed")
        if not np.isfinite(self.target_ess) or self.target_ess <= 0:
            raise ValueError("target_ess must be positive and finite")
        if not 0 <= self.boundary_stability <= 1:
            raise ValueError("boundary_stability must be in [0,1]")


@dataclass(frozen=True)
class BinPlan:
    cell_ids: np.ndarray
    times: np.ndarray
    branches: np.ndarray
    labels: np.ndarray
    centers: np.ndarray
    bandwidths: np.ndarray
    raw_kernel: Any
    aggregate_weights: Any
    membership: Any
    valid: np.ndarray
    reliable: np.ndarray
    cell_valid: np.ndarray
    diagnostics: dict
    fingerprint: str
    policy_fingerprint: str
    mode: str
    branch_policy: str
    bandwidth_diagnostics: dict

    @property
    def n_bins(self):
        return len(self.centers)

    @property
    def projection_weights(self):
        """Explicit current projection row normalization, including hard epsilon.

        Hard aggregation divides by exact counts; the production projector
        instead divides by counts+1e-10. This distinction is intentional.
        """
        sums = np.asarray(self.raw_kernel.sum(axis=1)).ravel()
        if sparse.issparse(self.raw_kernel):
            return _readonly_matrix(sparse.diags(1 / (sums + EPSILON)) @ self.raw_kernel)
        return _owned(self.raw_kernel / (sums[:, None] + EPSILON))


@dataclass(frozen=True)
class FittedBinningPolicy:
    config: BinningConfig
    train_times: np.ndarray
    train_ids: np.ndarray
    train_branches: np.ndarray
    train_embedding: Optional[np.ndarray]
    centers: np.ndarray
    bandwidths: np.ndarray
    row_branches: np.ndarray
    branch_ranges: tuple
    branch_boundaries: tuple
    qcut_edges: tuple
    stable_boundaries: tuple
    boundary_diagnostics: tuple
    neighbors: tuple
    fingerprint: str
    bandwidth_diagnostics: dict
    initial_plan: BinPlan = field(repr=False)


def _inputs(times, cell_ids, branches):
    t = np.asarray(times, dtype=float)
    ids = np.asarray(cell_ids)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all():
        raise ValueError("times must be nonempty, finite, one-dimensional")
    if ids.ndim != 1 or len(ids) != len(t) or pd.isna(ids).any():
        raise ValueError("cell_ids must be nonmissing and match times")
    if not pd.Index(ids).is_unique:
        raise ValueError("duplicate cell IDs")
    if branches is None:
        br = np.full(len(t), "__global__", dtype=object)
    else:
        br = np.asarray(branches)
        if br.ndim != 1 or len(br) != len(t) or pd.isna(br).any():
            raise ValueError("branches must be nonmissing and match times")
    try:
        np.unique(br)
    except TypeError as exc:
        raise ValueError("branch labels must be consistently sortable") from exc
    return _owned(t), _owned(ids), _owned(br)


def _requested_bins(n, cfg):
    if cfg.target_bins is not None:
        if cfg.overlap_factor is not None:
            return max(5, min(cfg.target_bins, int(cfg.overlap_factor * n / cfg.min_cells_per_bin)))
        return max(5, cfg.target_bins)
    return max(5, min(n // cfg.min_cells_per_bin, cfg.max_bins))


def _qcut_layout(t, br, cfg):
    requested = _requested_bins(len(t), cfg)
    unique = np.unique(br)
    # Bound requested quantiles, not only the deduplicated output kernel. Even
    # hard qcut creates O(requested) quantile arrays. Branch minima may exceed
    # the requested total, as in the legacy allocator.
    _budget(max(requested, 3 * len(unique) if len(unique) > 1 else requested), len(t), cfg)
    counts = np.array([(br == x).sum() for x in unique])
    if len(unique) > 1:
        allocation = np.maximum((counts / counts.sum() * requested).astype(int), 3)
        delta = requested - allocation.sum()
        while delta > 0:
            allocation[np.argmax(counts - allocation * 3)] += 1
            delta -= 1
        while delta < 0:
            if np.max(allocation - 3) <= 0:
                break
            allocation[np.argmax(allocation - 3)] -= 1
            delta += 1
    else:
        allocation = np.array([requested])
    labels = np.full(len(t), -1, dtype=int)
    ranges, edges, boundaries, row_branches = [], [], [], []
    offset = 0
    for branch, k in zip(unique, allocation):
        mask = br == branch
        local, cuts = pd.qcut(t[mask], int(k), labels=False, retbins=True, duplicates="drop")
        if not np.isfinite(local).all() or not len(local) or len(cuts) < 2:
            raise ValueError(f"no valid qcut for branch {branch!r}")
        nlocal = int(np.max(local)) + 1
        labels[mask] = local.astype(int) + offset
        ranges.append((branch, float(t[mask].min()), float(t[mask].max())))
        edges.append(_owned(cuts))
        boundaries.append((offset, offset + nlocal))
        row_branches.extend([branch] * nlocal)
        offset += nlocal
    return _owned(labels), tuple(ranges), tuple(edges), tuple(boundaries), _owned(row_branches)


def _budget(n_bins, n_cells, cfg):
    if n_bins * n_cells > cfg.max_kernel_elements:
        raise MemoryError(f"kernel {n_bins}x{n_cells} exceeds max_kernel_elements={cfg.max_kernel_elements}")


def _center_layout(t, br, bounds, ranges, cfg):
    n_bins = bounds[-1][1]
    centers = np.empty(n_bins)
    h = np.empty(n_bins)
    spacing = np.empty(n_bins)
    blocks = [(0, n_bins, t)] if cfg.branch_policy == "legacy" else [
        (a, b, t[br == branch]) for (a, b), (branch, _, _) in zip(bounds, ranges)]
    for a, b, values in blocks:
        span = float(values.max() - values.min())
        width = (span if span > 0 else 1.) / (b - a)
        centers[a:b] = np.linspace(values.min() + width / 2, values.max() - width / 2, b - a)
        spacing[a:b] = width  # adaptive uses local center spacing, not inflated sigma
        sigma = width
        if len(values) / (b - a) < cfg.min_cells_per_bin:
            sigma *= cfg.min_cells_per_bin / max(len(values) / (b - a), 1)
        h[a:b] = sigma
    return centers, h, spacing


def _detect_boundaries(t, br, centers, ranges, bounds, embedding, cfg):
    if cfg.boundary == "none":
        return (), ()
    if embedding is None:
        raise ValueError("split_noise requires an external training embedding")
    rng = np.random.default_rng(cfg.random_seed)
    stable, diagnostics = [], []
    for (branch, _, _), (a, b) in zip(ranges, bounds):
        # Fixed-time pilots match the output mask exactly. Quantile groups can
        # describe a different location under irregular sampling density.
        pilot_edges = (centers[a:b-1] + centers[a+1:b]) / 2
        pilot_labels = np.searchsorted(pilot_edges, t, side="right") + a
        for i in range(a, b - 1):
            left = embedding[(br == branch) & (pilot_labels == i)]
            right = embedding[(br == branch) & (pilot_labels == i + 1)]
            edge = float((centers[i] + centers[i + 1]) / 2)
            record: dict = dict(branch=branch, edge=edge, left_bin=i, status="uncertain_insufficient_support",
                          left_count=len(left), right_count=len(right), stability=None, threshold=None)
            if min(len(left), len(right)) >= cfg.boundary_min_side:
                # Strict subsampling varies even equal-sized pilots. Use the
                # SAME sample size for observed and residual-noise contrasts.
                m = max(1, min(len(left), len(right)) // 2)
                # Calibrate within-pilot measurement noise, not a pool polluted
                # by the very between-pilot shift being tested. Pilot robust
                # centers are medians; residual splits preserve sample sizes.
                pool = np.vstack([left - np.median(left, axis=0), right - np.median(right, axis=0)])
                noise, observed = [], []
                for _ in range(cfg.boundary_splits):
                    order = rng.permutation(len(pool))[:2*m]
                    noise.append(np.linalg.norm(np.median(pool[order[:m]], axis=0) - np.median(pool[order[m:]], axis=0)))
                    l = left[rng.choice(len(left), m, replace=False)]
                    r = right[rng.choice(len(right), m, replace=False)]
                    observed.append(np.linalg.norm(np.median(l, axis=0) - np.median(r, axis=0)))
                median = float(np.median(noise))
                threshold = median + 3 * float(np.median(np.abs(np.asarray(noise) - median)))
                stability = float(np.mean(np.asarray(observed) > threshold))
                record.update(status="stable" if stability >= cfg.boundary_stability else "uncertain_noise",
                              stability=stability, threshold=threshold, observed_distance=float(np.median(observed)),
                              resample_size=m, observed_samples=tuple(observed), noise_samples=tuple(noise))
                if record["status"] == "stable":
                    stable.append((branch, edge))
            diagnostics.append(record)
    return tuple(stable), tuple(diagnostics)


def _state_bandwidths(t, br, embedding, centers, spacing, ranges, bounds, stable, cfg):
    """Local-linear, equal-time pilots; bounded residual bootstrap, no ESS input.

    Supported pilot means are equally weighted within branch for global S²;
    supported branches contribute equally. Candidate evidence requires >=2
    supported pilots in its branch/boundary region and positive geometric mass.
    Evidence means a score is calculable, not that local coverage is adequate.
    Coverage is reported, not silently imputed. The non-statistical confidence
    flag additionally requires acceptance and complete geometric pilot coverage.
    """
    candidates = np.array(sorted(x for x in cfg.bandwidth_multipliers if x <= cfg.max_bandwidth_multiplier))
    npilots = len(centers) * cfg.state_pilot_subdivisions
    dim = embedding.shape[1]
    # Upper bound persistent pilot/bootstrap data plus local fitting work and
    # candidate diagnostics. No cell-by-pilot-by-candidate tensor is constructed.
    elements = npilots * (dim * (cfg.state_bootstrap_repeats + 1) + 8) * 3 + len(t) * dim * 6 + len(centers) * len(candidates) * 20
    if elements > cfg.max_kernel_elements:
        raise MemoryError(f"state working storage {elements} exceeds max_kernel_elements={cfg.max_kernel_elements}")
    rng = np.random.default_rng(np.random.SeedSequence([cfg.random_seed, 0x53544154]))
    ps, pb, pr, counts, ranks, noises, variances, means, perturbations = [], [], [], [], [], [], [], [], []
    for (branch, lo, hi), (a, b) in zip(ranges, bounds):
        cuts = sorted(edge for name, edge in stable if name == branch)
        edges = np.linspace(lo, hi, (b-a)*cfg.state_pilot_subdivisions + 1)
        assigned = np.searchsorted(edges[1:-1], t, side="right")
        regions = np.searchsorted(cuts, t, side="right")
        for j, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            s = float((left + right) / 2)
            region = np.searchsorted(cuts, s, side="right")
            mask = (br == branch) & (assigned == j) & (regions == region)
            values = embedding[mask]
            times = t[mask]
            n = len(times)
            ps.append(s); pb.append(branch); pr.append(region); counts.append(n)
            mu = np.zeros(dim); noise = variance = 0.; rank = 0
            boot = np.zeros((cfg.state_bootstrap_repeats, dim))
            if n >= cfg.state_min_pilot_cells:
                dt = times - times.mean()
                ss = float(dt @ dt)
                rank = 2 if times.max() > times.min() else 1
                coefficient = np.full(n, 1/n)
                fitted = np.broadcast_to(values.mean(axis=0), values.shape).copy()
                if rank == 2:
                    slope = dt @ values / ss
                    fitted += dt[:, None] * slope
                    coefficient += (s-times.mean()) * dt / ss
                mu = coefficient @ values
                residual = values - fitted
                residual -= residual.mean(axis=0)
                noise = float(np.sum(residual**2) / (n-rank))
                variance = noise * float(coefficient @ coefficient)
                residual *= np.sqrt(n / (n-rank))
                for r in range(cfg.state_bootstrap_repeats):
                    boot[r] = coefficient @ residual[rng.integers(0, n, n)]
            ranks.append(rank); noises.append(noise); variances.append(variance); means.append(mu); perturbations.append(boot)
    ps, pb, pr = np.asarray(ps), np.asarray(pb), np.asarray(pr)
    counts, ranks = np.asarray(counts), np.asarray(ranks)
    noise, v, mu, boot = np.asarray(noises), np.asarray(variances), np.asarray(means), np.asarray(perturbations)
    supported = ranks > 0
    branch_scales = []
    for branch, _, _ in ranges:
        good = supported & (pb == branch)
        if good.sum() >= 2:
            branch_mu = mu[good]
            branch_scales.append(max(0., float(np.mean(np.sum((branch_mu-branch_mu.mean(axis=0))**2, axis=1))
                                              - (1-1/good.sum()) * v[good].mean())))
    scale = float(np.mean(branch_scales)) if branch_scales else 0.
    # Relative guard only: no absolute unit-dependent noise floor.
    guard = np.finfo(float).eps * max(scale, float(noise.max(initial=0)))
    floor = cfg.state_floor_fraction**2 * scale + guard
    shape = (len(centers), len(candidates))
    D, N, U, scores, coverage = [np.zeros(shape) for _ in range(5)]
    evidence = np.zeros(shape, bool); accepted = np.zeros(shape, bool)
    pilot_counts = np.zeros(len(centers), int)
    chosen = np.empty(len(centers)); chosen_indices = np.empty(len(centers), int); reasons = []
    confidence = np.zeros(len(centers), bool)
    for (branch, _, _), (a, b) in zip(ranges, bounds):
        cuts = sorted(edge for name, edge in stable if name == branch)
        for i in range(a, b):
            region = np.searchsorted(cuts, centers[i], side="right")
            allpilots = (pb == branch) & (pr == region)
            good = allpilots & supported
            pilot_counts[i] = good.sum()
            for j, multiplier in enumerate(candidates):
                geometric = np.exp(-.5 * ((ps[allpilots]-centers[i]) / (spacing[i]*multiplier))**2)
                total = geometric.sum()
                weights = np.exp(-.5 * ((ps[good]-centers[i]) / (spacing[i]*multiplier))**2)
                mass = weights.sum()
                coverage[i, j] = mass/total if total > 0 else 0.
                evidence[i, j] = good.sum() >= 2 and mass > 0
                if not evidence[i, j]:
                    scores[i, j] = np.nan
                    continue
                p = weights / mass
                m = p @ mu[good]
                correction = float((p*(1-p)) @ v[good])
                D[i, j] = max(0., float(p @ np.sum((mu[good]-m)**2, axis=1)) - correction)
                N[i, j] = p @ noise[good]
                # One replicate at a time; fixed v and fixed geometric weights.
                replicates = np.empty(cfg.state_bootstrap_repeats)
                for r in range(cfg.state_bootstrap_repeats):
                    perturbed = mu[good] + boot[good, r]
                    replicates[r] = max(0., float(p @ np.sum((perturbed-p @ perturbed)**2, axis=1)) - correction)
                U[i, j] = np.std(replicates, ddof=1)
                numerator = max(0., D[i, j] - cfg.state_uncertainty_multiplier*U[i, j])
                denominator = N[i, j] + floor
                scores[i, j] = np.sqrt(numerator / denominator) if denominator > 0 else (0. if numerator == 0 else np.inf)
                accepted[i, j] = scores[i, j] <= cfg.state_budget
            # All candidates are evaluated; acceptance need not be monotone.
            evidenced_accepted = evidence[i] & accepted[i]
            if evidenced_accepted.any():
                index = np.flatnonzero(evidenced_accepted)[-1]
                reason = "largest_accepted"
                confidence[i] = bool(np.isclose(coverage[i, index], 1., rtol=0, atol=1e-12))
            elif evidence[i].any():
                index = np.flatnonzero(evidence[i])[0]
                reason = "state_budget_unmet_at_min" if index == 0 else "state_budget_unmet_at_min_evidenced"
            else:
                index = len(candidates)-1
                reason = "insufficient_state_evidence"
            chosen[i] = candidates[index]
            chosen_indices[i] = index
            reasons.append(reason)
    diagnostics = dict(source="frozen_training", selection="state", candidates=_owned(candidates),
                       D=_owned(D), N=_owned(N), U=_owned(U), score=_owned(scores), accepted=_owned(accepted),
                       chosen_multiplier=_owned(chosen), reason=_owned(reasons), allowed_cap=cfg.max_bandwidth_multiplier,
                       geometric_coverage=_owned(coverage), evidence=_owned(evidence), state_confident=_owned(confidence),
                       chosen_candidate_index=_owned(chosen_indices),
                       chosen_score_evidenced=_owned(evidence[np.arange(len(centers)), chosen_indices]),
                       partial_candidate_evidence=_owned(evidence.any(axis=1) & ~evidence.all(axis=1)),
                       insufficient_state_evidence=_owned(np.asarray(reasons) == "insufficient_state_evidence"),
                       state_budget_unmet_at_min=_owned(np.isin(reasons, ["state_budget_unmet_at_min", "state_budget_unmet_at_min_evidenced"])),
                       incomplete_pilot_coverage=_owned(coverage < 1-1e-12),
                       supported_pilot_count=_owned(pilot_counts), pilot_centers=_owned(ps), pilot_branches=_owned(pb),
                       pilot_regions=_owned(pr), pilot_cell_counts=_owned(counts), pilot_rank=_owned(ranks),
                       pilot_residual_df=_owned(np.where(supported, counts-ranks, 0)),
                       pilot_status=_owned(np.where(~supported, "insufficient_pilot_cells",
                                                    np.where(ranks == 1, "supported_intercept_only", "supported_local_linear"))),
                       pilot_supported=_owned(supported), pilot_means=_owned(mu), pilot_noise=_owned(noise), pilot_mean_variance=_owned(v),
                       global_scale_squared=scale, floor_squared=floor, state_budget=cfg.state_budget,
                       floor_fraction=cfg.state_floor_fraction, uncertainty_multiplier=cfg.state_uncertainty_multiplier,
                       bootstrap_repeats=cfg.state_bootstrap_repeats, pilot_subdivisions=cfg.state_pilot_subdivisions,
                       min_pilot_cells=cfg.state_min_pilot_cells,
                       scale_definition="equal supported pilots within branch, corrected mean variance; equal supported branches",
                       evidence_definition="calculable score: >=2 supported pilots in branch/boundary region and positive geometric mass; not local coverage assurance",
                       confidence_definition="accepted chosen score and complete geometric pilot coverage; not a statistical confidence guarantee",
                       selection_definition="largest evidenced acceptance; else smallest evidenced candidate with budget-unmet flag; only all-absent evidence chooses widest",
                       score_interpretation="no demonstrated excess distortion heuristic; not a confidence guarantee")
    return spacing * chosen, diagnostics


def _cell_valid(t, br, ranges):
    valid = np.zeros(len(t), dtype=bool)
    for branch, lo, hi in ranges:
        valid |= (br == branch) & (t >= lo) & (t <= hi)
    return valid


def _kernel(t, br, labels, centers, h, row_branches, ranges, stable, cfg):
    valid = _cell_valid(t, br, ranges)
    n_bins = len(centers)
    if cfg.mode == "hard":
        good = valid & (labels >= 0)
        return sparse.csr_matrix((np.ones(good.sum()), (labels[good], np.where(good)[0])), shape=(n_bins, len(t)))
    _budget(n_bins, len(t), cfg)  # BEFORE dense allocation
    k = np.exp(-.5 * ((centers[:, None] - t[None, :]) / h[:, None]) ** 2)
    k[:, ~valid] = 0
    if cfg.branch_policy == "separate":
        k *= row_branches[:, None] == br[None, :]
    for branch in np.unique(row_branches):
        cuts = sorted(edge for name, edge in stable if name == branch)
        if cuts:
            rows = np.where(row_branches == branch)[0]
            # Exact edge belongs to the RIGHT side, for centers and cells alike.
            regions = np.searchsorted(cuts, t, side="right")
            rowregions = np.searchsorted(cuts, centers[rows], side="right")
            k[rows] *= rowregions[:, None] == regions[None, :]
    return k


def _fingerprint(cfg, arrays, metadata):
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(version=VERSION, config=asdict(cfg), metadata=metadata), sort_keys=True, default=str).encode())
    for a in arrays:
        if a is None:
            digest.update(b"None")
            continue
        a = np.asarray(a)
        digest.update(str(a.dtype).encode())
        digest.update(str(a.shape).encode())
        if a.dtype.kind in "OUS":
            digest.update(json.dumps(a.tolist(), default=str, ensure_ascii=False).encode())
        else:
            digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


def _plan(cfg, t, ids, br, labels, centers, h, row_branches, ranges, stable, neighbors, fingerprint, bandwidth_diag):
    k = _kernel(t, br, labels, centers, h, row_branches, ranges, stable, cfg)
    sums = np.asarray(k.sum(axis=1)).ravel()
    squares = np.asarray(k.multiply(k).sum(axis=1)).ravel() if sparse.issparse(k) else (k*k).sum(axis=1)
    support = np.diff(k.indptr) if sparse.issparse(k) else (k > 0).sum(axis=1)
    ess = np.divide(sums*sums, squares, out=np.zeros_like(sums), where=squares > 0)
    valid = sums > 0
    if cfg.branch_policy == "separate":
        valid &= support >= cfg.min_support_cells
    reliable = valid & (support >= cfg.min_support_cells)
    if cfg.mode == "adaptive_gaussian":
        reliable &= ess >= cfg.target_ess
    if cfg.mode == "hard":
        aw = sparse.diags(np.divide(1., sums, out=np.zeros_like(sums), where=sums > 0)) @ k
    else:
        aw = k / (sums[:, None] + EPSILON)
    # Exclude invalid bins from membership even when their estimates are present.
    if sparse.issparse(k):
        mk = sparse.diags(valid.astype(float)) @ k
        colsum = np.asarray(mk.sum(axis=0)).ravel()
        membership = sparse.diags(1 / (colsum + EPSILON)) @ mk.T
    else:
        mk = k * valid[:, None]
        colsum = mk.sum(axis=0)
        membership = mk.T / (colsum[:, None] + EPSILON)
    cv = _cell_valid(t, br, ranges) & (colsum > 0)
    exactweights = sparse.diags(np.divide(1., sums, out=np.zeros_like(sums), where=sums > 0)) @ k if sparse.issparse(k) else np.divide(k, sums[:, None], out=np.zeros_like(k), where=sums[:, None] > 0)
    mean = np.asarray(exactweights @ t).ravel()
    second = np.asarray(exactweights @ (t*t)).ravel()
    spread = np.sqrt(np.maximum(second - mean*mean, 0))
    overlap, mass = [], np.zeros(len(centers))
    for a, b in neighbors:
        if sparse.issparse(k):
            dot = float(k.getrow(a).multiply(k.getrow(b)).sum())
        else:
            dot = float(k[a] @ k[b])
        overlap.append(dot / np.sqrt(squares[a]*squares[b]) if squares[a]*squares[b] > 0 else 0.)
    for i, branch in enumerate(row_branches):
        if sparse.issparse(aw):
            mass[i] = aw.getrow(i)[:, br != branch].sum()
        else:
            mass[i] = aw[i, br != branch].sum()
    gram_rank = None
    if len(centers) <= 128 and len(centers)**2 <= cfg.max_kernel_elements:
        gram = aw @ aw.T
        gram = gram.toarray() if sparse.issparse(gram) else gram
        trace = float(np.trace(gram))
        denom = float(np.sum(gram*gram))
        gram_rank = trace*trace / denom if denom > 0 else 0.
    diag = dict(support_cells=_owned(support), ess=_owned(ess), time_mean=_owned(mean), time_spread=_owned(spread),
                neighboring_overlap=_owned(overlap), neighbors=neighbors, cross_branch_mass=_owned(mass),
                nnz=int(k.nnz if sparse.issparse(k) else np.count_nonzero(k)), uncovered_cells=_owned(np.where(~cv)[0]),
                aggregate_row_sums=_owned(np.asarray(aw.sum(axis=1)).ravel()),
                membership_row_sums=_owned(np.asarray(membership.sum(axis=1)).ravel()),
                gram_effective_rank=gram_rank, gram_rank_definition="participation ratio of weight Gram eigenvalues",
                legacy_cross_branch_mixing=bool(cfg.branch_policy == "legacy" and np.any(mass > 0)),
                support_definition="count of strictly positive raw kernel entries (including Gaussian tails)",
                insufficient_support=_owned(support < cfg.min_support_cells),
                ess_target_unmet=_owned(ess < cfg.target_ess) if cfg.mode == "adaptive_gaussian" else _owned(np.zeros(len(centers), bool)))
    return BinPlan(_owned(ids), _owned(t), _owned(br), _owned(labels), _owned(centers), _owned(h),
                   _readonly_matrix(k), _readonly_matrix(aw), _readonly_matrix(membership), _owned(valid), _owned(reliable),
                   _owned(cv), diag, _fingerprint(cfg, [ids, t, br, labels], dict(policy_fingerprint=fingerprint)),
                   fingerprint, cfg.mode, cfg.branch_policy, bandwidth_diag)


def fit_policy(times, cell_ids, branches=None, embedding=None, config=None):
    """Fit once on WT metadata/optional external embedding, never expression.

    Default state selection plus split-noise boundaries requires an external
    aligned, finite training WT embedding. State chooses the largest accepted
    distortion/noise candidate; ESS only marks reliability afterward. Explicit
    ESS selection uses the smallest allowed candidate meeting target ESS.
    Full Gaussian support is retained except explicit branch/boundary masks.
    """
    cfg = BinningConfig() if config is None else config
    if not isinstance(cfg, BinningConfig):
        raise TypeError("config must be BinningConfig")
    t, ids, br = _inputs(times, cell_ids, branches)
    emb = None
    if embedding is not None:
        if isinstance(embedding, pd.DataFrame):
            if not embedding.index.equals(pd.Index(ids)):
                raise ValueError("embedding IDs must match cell_ids exactly")
            embedding = embedding.to_numpy()
        emb = np.asarray(embedding, dtype=float)
        if emb.ndim != 2 or emb.shape[0] != len(t) or emb.shape[1] < 1 or not np.isfinite(emb).all():
            raise ValueError("embedding must be finite, two-dimensional, aligned to training cells")
        if cfg.bandwidth_selection == "state":
            _budget(len(t), emb.shape[1], cfg)  # before owned embedding copy
        emb = _owned(emb)
    if cfg.bandwidth_selection == "state" and emb is None:
        raise ValueError("state selection requires an external aligned training WT embedding")
    labels, ranges, edges, bounds, row_branches = _qcut_layout(t, br, cfg)
    if cfg.mode != "hard":
        _budget(bounds[-1][1], len(t), cfg)
    centers, h, spacing = _center_layout(t, br, bounds, ranges, cfg)
    stable, boundary_diag = _detect_boundaries(t, br, centers, ranges, bounds, emb, cfg)
    bandwidth_diag = dict(source="frozen_training", selection=cfg.bandwidth_selection)
    if cfg.bandwidth_selection == "state":
        h, bandwidth_diag = _state_bandwidths(t, br, emb, centers, spacing, ranges, bounds, stable, cfg)
    elif cfg.mode == "adaptive_gaussian":
        candidates = sorted(x for x in cfg.bandwidth_multipliers if x <= cfg.max_bandwidth_multiplier)
        h = spacing * candidates[-1]
        # Selection is based exclusively on training metadata and frozen masks.
        selected = np.zeros(len(centers), bool)
        for multiplier in candidates:
            candidate_h = spacing * multiplier
            k = _kernel(t, br, labels, centers, candidate_h, row_branches, ranges, stable, cfg)
            sums = k.sum(axis=1)
            squares = (k*k).sum(axis=1)
            ess = np.divide(sums*sums, squares, out=np.zeros_like(sums), where=squares > 0)
            choose = ~selected & (ess >= cfg.target_ess)
            h[choose] = candidate_h[choose]
            selected |= choose
        bandwidth_diag.update(candidates=_owned(candidates), chosen_multiplier=_owned(h / spacing),
                              reason=_owned(np.where(selected, "smallest_ess_accepted", "ess_target_unmet")),
                              allowed_cap=cfg.max_bandwidth_multiplier)
    neighbors = tuple((i, i+1) for a, b in bounds for i in range(a, b-1))
    fingerprint = _fingerprint(cfg, [t, ids, br, emb, centers, h, labels, *edges],
                               dict(ranges=ranges, boundaries=bounds, stable=stable, diagnostics=boundary_diag,
                                    bandwidth_diagnostics={key: value.tolist() if isinstance(value, np.ndarray) else value
                                                           for key, value in bandwidth_diag.items()}))
    initial = _plan(cfg, t, ids, br, labels, centers, h, row_branches, ranges, stable, neighbors, fingerprint, bandwidth_diag)
    return FittedBinningPolicy(cfg, t, ids, br, emb, _owned(centers), _owned(h), row_branches,
                               ranges, bounds, edges, stable, boundary_diag, neighbors, fingerprint, bandwidth_diag, initial)


def build_plan(policy, times, cell_ids, branches=None):
    """Apply frozen ranges/cuts/centers/bandwidths/masks. No refit or new embedding."""
    if not isinstance(policy, FittedBinningPolicy):
        raise TypeError("policy must be FittedBinningPolicy")
    t, ids, br = _inputs(times, cell_ids, branches)
    labels = np.full(len(t), -1, dtype=int)
    for (branch, _, _), cuts, (a, _) in zip(policy.branch_ranges, policy.qcut_edges, policy.branch_boundaries):
        mask = br == branch
        if mask.any():
            local = np.asarray(pd.cut(t[mask], cuts, labels=False, include_lowest=True, right=True), dtype=float)
            good = np.isfinite(local)
            positions = np.where(mask)[0][good]
            labels[positions] = local[good].astype(int) + a
    cfg = policy.config
    return _plan(cfg, t, ids, br, labels, policy.centers, policy.bandwidths, policy.row_branches,
                 policy.branch_ranges, policy.stable_boundaries, policy.neighbors, policy.fingerprint, policy.bandwidth_diagnostics)


def aggregate(plan, matrix, *, cell_ids):
    """Aggregate dense/CSR expression using one aligned plan for RNA and ATAC.

    IDs are mandatory and order-sensitive. Legacy hard empty bins are zero;
    invalid separate bins are NaN. Reliability is a diagnostic, not imputation.
    """
    ids = np.asarray(cell_ids)
    if ids.ndim != 1 or not np.array_equal(ids, plan.cell_ids):
        raise ValueError("matrix cell_ids must exactly match plan row order")
    if sparse.issparse(matrix):
        if matrix.ndim != 2 or matrix.shape[0] != len(ids) or not np.isfinite(matrix.data).all():
            raise ValueError("matrix must be finite, two-dimensional, cell-aligned")
    else:
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids) or not np.isfinite(matrix).all():
            raise ValueError("matrix must be finite, two-dimensional, cell-aligned")
    output = plan.aggregate_weights @ matrix
    output = output.toarray() if sparse.issparse(output) else np.asarray(output)
    output = np.array(output, dtype=float, copy=True)
    if plan.branch_policy == "separate":
        output[~plan.valid] = np.nan
    return output


def query_membership(policy, query_times, query_ids, query_branches=None):
    """Return (cell-by-bin membership, valid-cell mask), without extrapolation.

    Invalid/out-of-range/unseen-branch queries have zero rows. Membership only
    uses valid bins supported by the WT initial plan; it is not a reconstruction
    inverse. Current projection epsilon normalization is retained.
    """
    t, ids, br = _inputs(query_times, query_ids, query_branches)
    plan = build_plan(policy, t, ids, br)
    k = plan.raw_kernel
    validbins = policy.initial_plan.valid
    if sparse.issparse(k):
        k = sparse.diags(validbins.astype(float)) @ k
        sums = np.asarray(k.sum(axis=0)).ravel()
        membership = sparse.diags(1/(sums + EPSILON)) @ k.T
    else:
        k = k * validbins[:, None]
        sums = k.sum(axis=0)
        membership = k.T / (sums[:, None] + EPSILON)
    valid = _cell_valid(t, br, policy.branch_ranges) & (sums > 0)
    return _readonly_matrix(membership), _owned(valid)
