"""RNA-only reference projection primitives.

This module deliberately contains no pipeline or file I/O.  It provides a
deterministic PCA reference, nearest-neighbour projection, and a small
cell-anchored counterfactual reconstruction useful to later integrations.
Projection is an inference/visualisation aid; it makes no claim about real
time, cell fate, or biological potential.
"""

from dataclasses import dataclass
import hashlib
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


ArrayLike = Union[np.ndarray, pd.DataFrame]


def _matrix(x: ArrayLike, feature_names=None, name="RNA"):
    if isinstance(x, pd.DataFrame):
        names = [str(v) for v in x.columns]
        a = x.to_numpy(dtype=float)
        if feature_names is not None and names != [str(v) for v in feature_names]:
            raise ValueError(f"{name} feature order does not match the reference")
    else:
        a = np.asarray(x, dtype=float)
        if a.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional array")
        if feature_names is None:
            raise ValueError("feature_names is required for array inputs")
        names = [str(v) for v in feature_names]
    if a.ndim != 2 or len(names) != a.shape[1]:
        raise ValueError(f"{name} has invalid feature dimensions")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values")
    if (a < 0).any():
        raise ValueError(f"{name} must be nonnegative; negative RNA state values are invalid")
    return a, tuple(names)


def _fingerprint(features, n_components, random_state, state_unit):
    text = "|".join(features) + f"|{n_components}|{random_state}|{state_unit}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class ReferenceProjectionModel:
    """Fitted RNA reference and projection thresholds."""

    features: tuple
    pca: PCA
    neighbors: NearestNeighbors
    latent: np.ndarray
    branch: np.ndarray
    pseudotime: np.ndarray
    reference_ids: np.ndarray
    distance_threshold: float
    ambiguity_threshold: float
    n_neighbors: int
    fingerprint: str
    state_unit: str
    residual_threshold: float
    residual_scale: float
    residual_ood_enabled: bool = True


def fit_reference_projection(
    control_rna: ArrayLike,
    branch_labels: Sequence,
    pseudotime: Sequence,
    *,
    feature_names: Optional[Sequence] = None,
    cell_ids: Optional[Sequence] = None,
    n_components: Optional[int] = None,
    n_neighbors: int = 10,
    ood_quantile: float = 0.99,
    ambiguity_threshold: float = 0.60,
    random_state: int = 0,
    state_unit: str,
) -> ReferenceProjectionModel:
    """Fit a deterministic PCA + nearest-neighbour RNA reference.

    ``control_rna`` is cells by genes.  DataFrame columns define feature
    order; array inputs must provide ``feature_names`` explicitly.
    ``branch_labels`` and ``pseudotime`` describe source-control cells.
    """
    if not isinstance(state_unit, str) or not state_unit.strip():
        raise ValueError("state_unit is required and must be a non-empty canonical unit string")
    x, features = _matrix(control_rna, feature_names, "control_rna")
    n = x.shape[0]
    branch = np.asarray(branch_labels, dtype=object)
    time = np.asarray(pseudotime, dtype=float)
    if len(branch) != n or len(time) != n or n < 2:
        raise ValueError("labels, pseudotime, and control cells must have compatible lengths (>=2)")
    if not np.isfinite(time).all():
        raise ValueError("pseudotime contains non-finite values")
    if not (0 < ood_quantile <= 1) or not (0 < ambiguity_threshold <= 1):
        raise ValueError("ood_quantile and ambiguity_threshold must be in (0, 1]")
    k = max(1, min(int(n_neighbors), n))
    # Retain all estimable components by default.  The first two are only a
    # visualisation convention; OOD inference uses the full retained space.
    requested = min(x.shape[1], max(1, n - 1)) if n_components is None else int(n_components)
    components = max(1, min(requested, x.shape[1], n))
    pca = PCA(n_components=components, svd_solver="full", random_state=random_state)
    latent = pca.fit_transform(x)
    rank = int(np.sum(pca.singular_values_ > (np.max(pca.singular_values_) * 1e-12)))
    if rank < components:
        components = max(1, rank)
        pca = PCA(n_components=components, svd_solver="full", random_state=random_state)
        latent = pca.fit_transform(x)
    nn = NearestNeighbors(n_neighbors=k).fit(latent)
    d, _ = nn.kneighbors(latent)
    # Leave-one-out distance where possible gives a stable training null.
    null = d[:, 1] if n > 1 and k > 1 else d[:, 0]
    threshold = float(np.quantile(null, ood_quantile))
    residuals = np.linalg.norm(x - pca.inverse_transform(latent), axis=1)
    residual_threshold = float(np.quantile(residuals, ood_quantile))
    residual_scale = float(np.std(residuals) + np.mean(residuals) + 1e-12)
    # A full-rank PCA reconstructs its training reference (up to floating
    # point error), so its residual null cannot provide an OOD threshold.
    # Do not, however, disable the independent latent-distance diagnostic.
    full_rank_pca = components >= min(x.shape[1], n - 1) and rank == components
    residual_null_scale = max(1.0, float(np.linalg.norm(x, axis=1).max(initial=0.0)))
    residual_null_degenerate = bool(np.all(residuals <= 1e-10 * residual_null_scale))
    residual_ood_enabled = not (full_rank_pca and residual_null_degenerate)
    ids = np.asarray(cell_ids if cell_ids is not None else [str(i) for i in range(n)], dtype=object)
    if len(ids) != n:
        raise ValueError("cell_ids must match the number of control cells")
    return ReferenceProjectionModel(
        features, pca, nn, latent, branch, time, ids, float(threshold),
        float(ambiguity_threshold), k, _fingerprint(features, components, random_state, state_unit),
        state_unit, residual_threshold, residual_scale, residual_ood_enabled
    )


def project_reference(
    model: ReferenceProjectionModel,
    source_control_rna: ArrayLike,
    projected_rna: Optional[ArrayLike] = None,
    *,
    feature_names: Optional[Sequence] = None,
    cell_ids: Optional[Sequence] = None,
    source_control_ids: Optional[Sequence] = None,
    state_unit: Optional[str] = None,
) -> pd.DataFrame:
    """Project source-control and optionally altered RNA states.

    The returned table is visualization-ready.  ``projected_*`` fields are
    abstained (NA) for OOD or ambiguous cells; latent coordinates and the
    distance diagnostics remain available.
    """
    if state_unit != model.state_unit:
        raise ValueError("state_unit must exactly match the fitted canonical RNA state unit")
    source, features = _matrix(source_control_rna, feature_names, "source_control_rna")
    if features != model.features:
        raise ValueError("source_control_rna feature order does not match the fitted reference")
    if projected_rna is None:
        altered = source
    else:
        altered, altered_features = _matrix(projected_rna, feature_names, "projected_rna")
        if altered_features != model.features:
            raise ValueError("projected_rna feature order does not match the fitted reference")
    if altered.shape != source.shape:
        raise ValueError("source_control_rna and projected_rna must have the same shape")
    n = len(source)
    ids = np.asarray(cell_ids if cell_ids is not None else [str(i) for i in range(n)], dtype=object)
    if len(ids) != n:
        raise ValueError("cell_ids must match the number of projected cells")
    source_ids = None if source_control_ids is None else np.asarray(source_control_ids, dtype=object)
    if source_ids is not None and len(source_ids) != n:
        raise ValueError("source_control_ids must match the number of source cells")
    ref_lookup = {v: i for i, v in enumerate(model.reference_ids)}
    if source_ids is not None and any(v not in ref_lookup for v in source_ids):
        raise ValueError("source_control_ids must identify fitted reference cells")

    def infer(x):
        z = model.pca.transform(x)
        dist, ind = model.neighbors.kneighbors(z)
        dist = np.maximum(dist, 0)
        weights = 1.0 / (dist + 1e-12)
        probs = []
        for w, ix in zip(weights, ind):
            totals = {b: float(w[np.asarray(model.branch[ix]) == b].sum()) for b in set(model.branch[ix])}
            probs.append(totals)
        labels = sorted(set(model.branch.tolist()), key=str)
        probability = np.zeros((n, len(labels)))
        for row, p in enumerate(probs):
            total = sum(p.values()) or 1.0
            for j, label in enumerate(labels):
                probability[row, j] = p.get(label, 0.0) / total
        entropy = -np.sum(np.where(probability > 0, probability * np.log(probability), 0), axis=1)
        winner = probability.argmax(axis=1)
        winner = probability.argmax(axis=1)
        pseudo = np.array([
            np.average(model.pseudotime[ix[np.asarray(model.branch[ix]) == labels[win]]],
                       weights=w[np.asarray(model.branch[ix]) == labels[win]])
            for w, ix, win in zip(weights, ind, winner)
        ])
        score = dist[:, 0] / max(model.distance_threshold, 1e-12)
        residual = np.linalg.norm(x - model.pca.inverse_transform(z), axis=1)
        residual_score = residual / max(model.residual_scale, 1e-12)
        # A probability gap is more interpretable than raw entropy for two branches.
        ambiguous = ((probability.max(axis=1) < model.ambiguity_threshold) |
                     (entropy > 0.5 * np.log(max(len(labels), 2))))
        residual_ood = (
            model.residual_ood_enabled
            & (residual > model.residual_threshold + model.residual_scale)
        )
        ood = (score > 1.0) | residual_ood
        return z, ind, dist[:, 0], probability, entropy, labels, winner, pseudo, score, residual, residual_score, ambiguous, ood

    sz, si, sd, sp, se, labels, sw, st, ss, sr, srs, sa, so = infer(source)
    pz, pi, pdist, pp, pe, _, pw, pt, ps, pr, prs, pa, po = infer(altered)
    rows = []
    for i in range(n):
        # Ambiguity is reported before OOD: a point between well-supported
        # branches is useful diagnostic information rather than being hidden
        # by its (necessarily large) distance to either branch.
        status = "ood_and_ambiguous" if po[i] and pa[i] else ("ood" if po[i] else ("ambiguous" if pa[i] else "ok"))
        source_status = "ood" if so[i] else ("ambiguous" if sa[i] else "ok")
        source_ix = si[i, 0] if source_ids is None else ref_lookup[source_ids[i]]
        source_branch = model.branch[source_ix] if source_ids is not None else model.branch[si[i, 0]]
        source_pt = model.pseudotime[source_ix] if source_ids is not None else st[i]
        same_branch = status == "ok" and labels[pw[i]] == source_branch
        row = {"cell_id": ids[i], "source_control_id": model.reference_ids[source_ix],
               "source_control_branch": source_branch,
               "source_control_pseudotime": source_pt, "projected_branch": (labels[pw[i]] if status == "ok" else pd.NA),
               "projected_pseudotime": (pt[i] if status == "ok" else np.nan),
               "source_distance": sd[i], "projected_distance": pdist[i],
               "source_ood_score": ss[i], "ood_score": ps[i], "source_reconstruction_residual": sr[i],
               "reconstruction_residual": pr[i], "reconstruction_residual_score": prs[i],
               "source_ood_status": source_status, "ood_status": ("ood" if po[i] else "ok"),
               "ambiguity_status": ("ambiguous" if pa[i] else "ok"), "projection_status": status,
               "same_branch_displacement": (0.0 if same_branch and abs(pt[i] - source_pt) < 1e-6
                                             else (pt[i] - source_pt if same_branch else np.nan)),
               "branch_entropy": pe[i], "state_unit": model.state_unit,
               "metadata_fingerprint": model.fingerprint}
        for j, label in enumerate(labels):
            row[f"branch_probability_{label}"] = pp[i, j]
        for j in range(min(2, sz.shape[1])): row[f"source_latent_{j+1}"] = sz[i, j]
        for j in range(min(2, pz.shape[1])): row[f"projected_latent_{j+1}"] = pz[i, j]
        rows.append(row)
    return pd.DataFrame(rows)


def reconstruct_counterfactual(
    control_cell_states: ArrayLike,
    control_bin_states: ArrayLike,
    perturbed_bin_states: ArrayLike,
    cell_bin_weights: ArrayLike,
    *, feature_names: Optional[Sequence] = None,
    state_unit: str,
    normalize_weights: bool = True,
) -> pd.DataFrame:
    """Add each cell's weighted bin-level RNA change to its control state."""
    if not isinstance(state_unit, str) or not state_unit.strip():
        raise ValueError("state_unit is required and must be a non-empty canonical unit string")
    cell, f = _matrix(control_cell_states, feature_names, "control_cell_states")
    cb, f2 = _matrix(control_bin_states, feature_names, "control_bin_states")
    pb, f3 = _matrix(perturbed_bin_states, feature_names, "perturbed_bin_states")
    w = np.asarray(cell_bin_weights, dtype=float)
    if f != f2 or f != f3 or cb.shape != pb.shape or w.shape != (len(cell), len(cb)):
        raise ValueError("cell/bin state features or membership weights have incompatible shapes")
    if not np.isfinite(w).all() or (w < 0).any() or (w.sum(axis=1) <= 0).any():
        raise ValueError("cell_bin_weights must be finite, non-negative, and non-zero per cell")
    if normalize_weights:
        w = w / w.sum(axis=1, keepdims=True)
    reconstructed = cell + w @ (pb - cb)
    if (reconstructed < 0).any():
        minimum = float(reconstructed.min())
        raise ValueError(
            f"reconstructed RNA state contains negative values (minimum={minimum:.6g}); "
            "diagnostic: control state plus weighted bin delta is invalid"
        )
    result = pd.DataFrame(reconstructed, columns=list(f))
    result.attrs["state_unit"] = state_unit
    return result


# Explicit aliases make the small public API easy to discover.
fit_reference = fit_reference_projection
project_reference_projection = project_reference
cell_anchored_counterfactual = reconstruct_counterfactual
