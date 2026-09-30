"""
Shared neural network fitting of the peak→gene transfer function.

PairEmbeddingNN: f(ATAC_value, peak_id, gene_id) → RNA_expression
Captures pair-specificity through learned embeddings, with the MLP sharing underlying nonlinear patterns.
Defined over the entire real domain, so OOB issues are inherently avoided.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None


def _check_torch():
    """Return the torch module, raising a clear error if it is not installed."""
    if torch is None:
        raise ImportError(
            "NN mode requires PyTorch. Please run: pip install torch>=2.0"
        )
    return torch


def _train_validation_sizes(n: int) -> Tuple[int, int]:
    """Return a non-empty train split and a bounded validation split."""
    if n < 1:
        raise ValueError("NN training requires at least one observation")
    if n == 1:
        return 1, 0
    n_train = max(1, min(n - 1, int(n * 0.9)))
    return n_train, n - n_train


# ============================================================================
# Model definition
# ============================================================================


class PairEmbeddingNN(torch.nn.Module if torch is not None else object):
    """PyTorch module for f(ATAC, peak_id, gene_id) → RNA_expression.

    Monotonic constraint + dot-product pair modulation:
      - ATAC pathway (shared): ATAC → nonnegative-weight MLP → f(atac) ∈ R^{d_atac}
        All weights >= 0 → each basis function f_i(x) is monotonically increasing
        All pairs share the same set of nonlinear basis functions
      - Embedding pathway (per-pair): peak + gene → (weights ∈ R^{d_atac}, bias ∈ R)
        Each pair learns how to combine the ATAC basis functions
      - Output: RNA = sum(weights ⊙ f(atac)) + bias

    Key advantages:
      - Monotonically increasing basis functions → gain sign is uniquely determined by pair weights
      - weights sum > 0 → gain > 0 (activating), weights sum < 0 → gain < 0 (repressing)
      - Avoids the sign contradiction of assigning negative gains to boosting edges
      - The ATAC signal cannot be bypassed — pair embeddings provide weights only, not predictions
    """

    def __init__(self, n_peaks: int, n_genes: int, d_embed: int = 4,
                 d_atac: int = 16, atac_hidden_dims: list = None,
                 dropout: float = 0.1):
        if torch is None:
            _check_torch()  # Trigger ImportError
        super().__init__()
        self.d_embed = d_embed
        self.d_atac = d_atac
        self.dropout = dropout
        self.atac_hidden_dims = atac_hidden_dims or [32, 64, 128]

        # --- Shared ATAC pathway: 1D → configurable multilayer MLP → d_atac-dimensional feature vector ---
        # Shared by all pairs; learns a set of basis functions representing the ATAC→RNA response
        # All weights >= 0 (monotonic constraint) → each basis function is monotonically increasing
        layers = []
        prev_dim = 1
        for h_dim in self.atac_hidden_dims:
            layers.extend([
                torch.nn.Linear(prev_dim, h_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(torch.nn.Linear(prev_dim, d_atac))
        self.atac_net = torch.nn.Sequential(*layers)

        # --- Embedding pathway: pair identity → (weights, bias) ---
        self.peak_embed = torch.nn.Embedding(n_peaks, d_embed)
        self.gene_embed = torch.nn.Embedding(n_genes, d_embed)
        # 2*d_embed → d_atac + 1: d_atac weights + 1 bias
        self.pair_head = torch.nn.Linear(2 * d_embed, d_atac + 1)

    def clamp_atac_weights(self):
        """Clamp the weights of all atac_net Linear layers to >= 0.

        This ensures that each basis function f_i(x) is monotonically increasing with ATAC input.
        The pair_head weights are unconstrained — they determine whether each pair is activating (positive) or repressing (negative).

        Call this after each optimizer.step().
        """
        for module in self.atac_net:
            if isinstance(module, torch.nn.Linear):
                module.weight.data.clamp_(min=0.0)

    def forward(self, atac, peak_idx, gene_idx):
        """Forward pass: RNA = sum(weights ⊙ f(atac)) + bias."""
        if atac.dim() == 1:
            atac = atac.unsqueeze(-1)  # (N, 1)

        # Shared ATAC pathway: (N, 1) → (N, d_atac)
        f_atac = self.atac_net(atac)

        # Embedding pathway: (N, 2*d_embed) → (N, d_atac + 1)
        p_emb = self.peak_embed(peak_idx)   # (N, d_embed)
        g_emb = self.gene_embed(gene_idx)   # (N, d_embed)
        pair_params = self.pair_head(torch.cat([p_emb, g_emb], dim=-1))
        weights = pair_params[:, :self.d_atac]   # (N, d_atac)
        bias = pair_params[:, self.d_atac:]      # (N, 1)

        # Dot-product combination: RNA = Σ(weights_i * f_atac_i) + bias
        return (weights * f_atac).sum(dim=-1) + bias.squeeze(-1)


# ============================================================================
# Training data collection
# ============================================================================


def _collect_training_data(
    atac_binned_scaled: np.ndarray,
    rna_binned: np.ndarray,
    causal_edges: pd.DataFrame,
    atac_name_to_idx: dict,
    rna_name_to_idx: dict,
    branch_boundaries,
    min_observations: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    """Collect training samples for all pairs from the binned data.

    Branch-aware lag-1 extraction logic (consistent with the perturbation engine's ATAC→RNA reading).

    Returns
    -------
    X_atac : (N,) float32
    X_peak_idx : (N,) int64
    X_gene_idx : (N,) int64
    y_rna : (N,) float32
    peak_to_int : {peak_name: int}
    gene_to_int : {gene_name: int}
    """
    atac_list, peak_list, gene_list, rna_list = [], [], [], []

    seen_peaks: set = set()
    seen_genes: set = set()

    for _, row in causal_edges.iterrows():
        peak_id = row["peak_id"]
        gene = row["gene"]
        peak_idx = atac_name_to_idx.get(peak_id)
        gene_idx = rna_name_to_idx.get(gene)
        if peak_idx is None or gene_idx is None:
            continue

        if branch_boundaries is not None and len(branch_boundaries) > 1:
            X_parts, R_parts = [], []
            for start, end in branch_boundaries:
                if end - start > 1:
                    X_parts.append(atac_binned_scaled[start:end - 1, peak_idx])
                    R_parts.append(rna_binned[start + 1:end, gene_idx])
            if X_parts:
                X_seq = np.concatenate(X_parts)
                R_seq = np.concatenate(R_parts)
            else:
                continue
        else:
            X_seq = atac_binned_scaled[:-1, peak_idx]
            R_seq = rna_binned[1:, gene_idx]

        valid = ~np.isnan(X_seq) & ~np.isnan(R_seq)
        X_seq = X_seq[valid]
        R_seq = R_seq[valid]

        if len(X_seq) < min_observations:
            continue

        seen_peaks.add(peak_id)
        seen_genes.add(gene)

        for x, r in zip(X_seq, R_seq):
            atac_list.append(float(x))
            peak_list.append(peak_id)
            gene_list.append(gene)
            rna_list.append(float(r))

    peak_to_int = {p: i for i, p in enumerate(sorted(seen_peaks))}
    gene_to_int = {g: i for i, g in enumerate(sorted(seen_genes))}

    X_atac = np.array(atac_list, dtype=np.float32)
    X_peak_idx = np.array([peak_to_int[p] for p in peak_list], dtype=np.int64)
    X_gene_idx = np.array([gene_to_int[g] for g in gene_list], dtype=np.int64)
    y_rna = np.array(rna_list, dtype=np.float32)

    logger.info(
        f"NN training data: {len(X_atac)} samples, "
        f"{len(seen_peaks)} peaks, {len(seen_genes)} genes"
    )
    return X_atac, X_peak_idx, X_gene_idx, y_rna, peak_to_int, gene_to_int


# ============================================================================
# Training loop
# ============================================================================


def _train_nn(
    model: PairEmbeddingNN,
    X_atac: np.ndarray,
    X_peak: np.ndarray,
    X_gene: np.ndarray,
    y: np.ndarray,
    device: str = "cuda",
    batch_size: int = 4096,
    lr: float = 0.001,
    max_epochs: int = 200,
    patience: int = 25,
) -> list:
    """Train PairEmbeddingNN and return the per-epoch train/val loss history."""
    _check_torch()

    # Fix random seeds so identical inputs produce identical models, eliminating training-randomness fluctuations
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    n = len(X_atac)
    n_train, n_val = _train_validation_sizes(n)
    indices = np.random.permutation(n)
    val_idx = indices[n_train:]

    X_atac_t = torch.from_numpy(X_atac)
    X_peak_t = torch.from_numpy(X_peak)
    X_gene_t = torch.from_numpy(X_gene)
    y_t = torch.from_numpy(y)

    t_device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; falling back to CPU")
    logger.info(f"NN training device: {t_device}")

    model.to(t_device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(max_epochs):
        # shuffle
        perm = torch.randperm(n_train)
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            b_idx = perm[start:start + batch_size]
            b_atac = X_atac_t[b_idx].to(t_device)
            b_peak = X_peak_t[b_idx].to(t_device)
            b_gene = X_gene_t[b_idx].to(t_device)
            b_y = y_t[b_idx].to(t_device)

            optimizer.zero_grad()
            pred = model.forward(b_atac, b_peak, b_gene)
            loss = loss_fn(pred, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.clamp_atac_weights()  # Enforce monotonically increasing basis functions
            train_loss_sum += float(loss.item())
            n_batches += 1

        # validation
        model.eval()
        with torch.no_grad():
            if n_val:
                val_pred = model.forward(
                    X_atac_t[val_idx].to(t_device),
                    X_peak_t[val_idx].to(t_device),
                    X_gene_t[val_idx].to(t_device),
                )
                val_loss = float(loss_fn(val_pred, y_t[val_idx].to(t_device)).item())
            else:
                # A one-observation dataset has no held-out validation set.
                val_loss = avg_train = train_loss_sum / max(n_batches, 1)

        scheduler.step(val_loss)
        avg_train = train_loss_sum / max(n_batches, 1)
        history.append((avg_train, val_loss))

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch + 1}/{max_epochs}: "
                f"train_loss={avg_train:.6f}, val_loss={val_loss:.6f}"
            )

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ============================================================================
# Per-pair metric computation
# ============================================================================


def _compute_pair_metrics(
    model: PairEmbeddingNN,
    X_atac: np.ndarray,
    X_peak: np.ndarray,
    X_gene: np.ndarray,
    y: np.ndarray,
    peak_to_int: dict,
    gene_to_int: dict,
    causal_edges: pd.DataFrame,
    atac_name_to_idx: dict,
    rna_name_to_idx: dict,
    pair_means: dict = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Compute R² and steady_state_gain for each peak-gene pair.

    The model is trained on centered targets (mean 0 for each pair), so pair_mean must be added back
    to compute absolute R². steady_state_gain is computed through autograd and is unaffected by centering.
    """
    _check_torch()
    model.eval()
    model.to(torch.device(device))
    t_device = torch.device(device)

    int_to_peak = {v: k for k, v in peak_to_int.items()}
    int_to_gene = {v: k for k, v in gene_to_int.items()}

    # Group ground-truth indices by pair
    pair_indices: dict = {}
    for i in range(len(X_atac)):
        p_int = int(X_peak[i])
        g_int = int(X_gene[i])
        pair_indices.setdefault((p_int, g_int), []).append(i)

    results = []
    for (p_int, g_int), idx_list in pair_indices.items():
        peak_id = int_to_peak.get(p_int)
        gene = int_to_gene.get(g_int)
        if peak_id is None or gene is None:
            continue

        idx_arr = np.array(idx_list)
        y_true = y[idx_arr]

        # R² — add pair_mean back to the model's centered predictions to obtain absolute predictions
        pair_mean = pair_means.get((p_int, g_int), 0.0) if pair_means else 0.0
        with torch.no_grad():
            pred_centered = model.forward(
                torch.from_numpy(X_atac[idx_arr]).to(t_device),
                torch.from_numpy(X_peak[idx_arr]).to(t_device),
                torch.from_numpy(X_gene[idx_arr]).to(t_device),
            )
            y_pred = pred_centered.cpu().numpy() + pair_mean

        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

        # steady_state_gain via autograd on 50 test points
        # Gain is unaffected by centering: d(pred)/d(atac) = d(pred_centered)/d(atac)
        x_min = float(X_atac[idx_arr].min())
        x_max = float(X_atac[idx_arr].max())
        x_test = np.linspace(x_min, x_max, 50, dtype=np.float32)
        x_t = torch.from_numpy(x_test).to(t_device)
        x_t.requires_grad = True

        p_t = torch.full_like(x_t, p_int, dtype=torch.long).to(t_device)
        g_t = torch.full_like(x_t, g_int, dtype=torch.long).to(t_device)

        pred_t = model.forward(x_t, p_t, g_t)
        grad_outputs = torch.ones_like(pred_t)
        grads = torch.autograd.grad(pred_t, x_t, grad_outputs=grad_outputs,
                                     create_graph=False, retain_graph=False)[0]
        gain = float(torch.clamp(grads.mean(), -5.0, 5.0).cpu())

        response_type = _classify_response(pred_t.detach().cpu().numpy())

        results.append({
            "peak_id": peak_id,
            "gene": gene,
            "r2_score": r2,
            "steady_state_gain": gain,
            "response_type": response_type,
            "n_effective_bins": len(idx_list),
            "X_operating_min": x_min,
            "X_operating_max": x_max,
        })

    logger.info(f"NN metric computation complete: {len(results)} pairs")
    return pd.DataFrame(results)


def _verify_gain_sign(
    transfer_functions: pd.DataFrame,
    granger_results: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """Validate the NN gain sign using the Pearson correlation coefficient sign.

    pearson_r = corr(X(t-1), Y(t)) in the Granger results directly measures
    the correlation direction between ATAC accessibility and gene expression
    (without interference from competition with the Y(t-1) autoregressive term):
      pearson_r > 0 → ATAC accessibility accompanies gene upregulation → gain should be positive
      pearson_r < 0 → ATAC accessibility accompanies gene downregulation → gain should be negative

    When the NN gain sign disagrees with the pearson_r sign, flip the gain sign.

    Returns
    -------
    transfer_functions : pd.DataFrame
        Sign-corrected transfer-function table
    stats : dict
        {n_total, n_flipped, n_skipped}
    """
    if granger_results is None or granger_results.empty:
        logger.info("Gain sign validation: no Granger results; skipping")
        return transfer_functions, {"n_total": len(transfer_functions),
                                     "n_flipped": 0, "n_skipped": 0}

    if "pearson_r" not in granger_results.columns:
        logger.warning("Granger results lack the pearson_r column; skipping gain sign validation")
        return transfer_functions, {"n_total": len(transfer_functions),
                                     "n_flipped": 0, "n_skipped": 0}

    # Build the peak_gene → pearson_r lookup table
    coef_lookup = {}
    for _, row in granger_results.iterrows():
        key = (row["peak_id"], row["gene"])
        coef_lookup[key] = float(row["pearson_r"])

    n_flipped = 0
    n_skipped = 0
    n_total = len(transfer_functions)

    for idx, row in transfer_functions.iterrows():
        key = (row["peak_id"], row["gene"])
        pearson_r = coef_lookup.get(key)
        if pearson_r is None:
            n_skipped += 1
            continue

        gain = row["steady_state_gain"]
        if gain == 0:
            n_skipped += 1
            continue

        # Signs disagree → flip gain
        if np.sign(gain) != np.sign(pearson_r):
            transfer_functions.loc[idx, "steady_state_gain"] = -gain
            n_flipped += 1

    logger.info(
        f"Gain sign validation (Pearson r): {n_total} pairs, "
        f"flipped {n_flipped} pairs ({n_flipped / max(n_total, 1) * 100:.1f}%), "
        f"skipped {n_skipped} pairs"
    )
    return transfer_functions, {"n_total": n_total, "n_flipped": n_flipped,
                                 "n_skipped": n_skipped}


def _classify_response(y_pred: np.ndarray) -> str:
    """Classify response monotonicity using Spearman rank correlation."""
    from scipy.stats import spearmanr
    if len(y_pred) < 3:
        return "unknown"
    rho, p = spearmanr(np.arange(len(y_pred)), y_pred)
    if p > 0.05:
        return "complex"
    return "activating" if rho > 0 else "repressing"


# ============================================================================
# Top-level entry point: NN transfer-function fitting for fit_atac_to_rna
# ============================================================================


def _fit_atac_to_rna_nn(
    rna_adata,
    atac_adata,
    pseudotime_df,
    causal_edges: pd.DataFrame,
    config: dict,
    root_atac: np.ndarray,
    atac_binned_scaled: np.ndarray,
    rna_binned: np.ndarray,
    granger_results: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, dict]:
    """NN-only ATAC→RNA transfer-function fitting.

    Returns (transfer_functions, transfer_models).
    transfer_models = {"type": "nn", "data": {...}}
    """
    cfg = config["atac_to_rna"]

    atac_name_to_idx = {name: i for i, name in enumerate(atac_adata.var_names)}
    rna_name_to_idx = {name: i for i, name in enumerate(rna_adata.var_names)}

    branch_boundaries = (
        pseudotime_df.attrs.get("branch_boundaries")
        if hasattr(pseudotime_df, "attrs") else None
    )

    logger.info("NN: collecting training data...")
    X_atac, X_peak, X_gene, y_rna, peak_to_int, gene_to_int = _collect_training_data(
        atac_binned_scaled, rna_binned, causal_edges,
        atac_name_to_idx, rna_name_to_idx, branch_boundaries,
        min_observations=cfg.get("min_observations", 5),
    )

    if len(X_atac) == 0:
        raise RuntimeError(
            "NN transfer fitting has no usable peak-gene observations; "
            "check causal edges, pseudotime binning, and min_observations."
        )

    # --- Key: per-pair target centering ---
    # If the model has a bias term, the embedding can memorize each pair's mean RNA and completely ignore ATAC.
    # After centering, each pair's target mean is 0, so the model must use ATAC features to predict deviations.
    pair_means: dict = {}
    pair_indices: dict = {}
    for i in range(len(X_atac)):
        key = (int(X_peak[i]), int(X_gene[i]))
        pair_indices.setdefault(key, []).append(i)

    y_centered = y_rna.copy()
    for key, idx_list in pair_indices.items():
        mu = float(y_rna[idx_list].mean())
        pair_means[key] = mu
        y_centered[idx_list] -= mu

    logger.info(
        f"NN: per-pair centering complete, "
        f"y_raw ∈ [{y_rna.min():.4f}, {y_rna.max():.4f}], "
        f"y_centered ∈ [{y_centered.min():.4f}, {y_centered.max():.4f}]"
    )

    d_embed = cfg.get("nn_d_embed", 4)
    d_atac = cfg.get("nn_d_atac", 16)
    atac_hidden_dims = cfg.get("nn_atac_hidden_dims", [32, 64, 128])
    dropout = cfg.get("nn_dropout", 0.1)
    batch_size = cfg.get("nn_batch_size", 4096)
    lr = cfg.get("nn_learning_rate", 0.001)
    max_epochs = cfg.get("nn_max_epochs", 200)
    patience = cfg.get("nn_early_stopping_patience", 25)
    device = cfg.get("nn_device", "cuda")

    n_peaks = len(peak_to_int)
    n_genes = len(gene_to_int)
    logger.info(
        f"NN model: {n_peaks} peaks x {n_genes} genes, "
        f"d_embed={d_embed}, d_atac={d_atac}, "
        f"atac_hidden={atac_hidden_dims}, dropout={dropout}"
    )

    model = PairEmbeddingNN(n_peaks, n_genes, d_embed, d_atac,
                            atac_hidden_dims, dropout)

    logger.info("NN: starting training (centered targets)...")
    _train_nn(model, X_atac, X_peak, X_gene, y_centered,
              device=device, batch_size=batch_size, lr=lr,
              max_epochs=max_epochs, patience=patience)

    logger.info("NN: computing per-pair metrics...")
    transfer_functions = _compute_pair_metrics(
        model, X_atac, X_peak, X_gene, y_rna,
        peak_to_int, gene_to_int, causal_edges,
        atac_name_to_idx, rna_name_to_idx,
        pair_means=pair_means,
        device="cpu",
    )

    # --- Validate the NN gain sign using the Pearson r sign ---
    if granger_results is not None and not granger_results.empty:
        transfer_functions, _ = _verify_gain_sign(
            transfer_functions, granger_results,
        )

    # Build the transfer_models data structure
    model.cpu().eval()
    # Convert pair_means to a serializable format: {(p_int, g_int): mean}
    pair_means_serializable = {f"{k[0]}:{k[1]}": v for k, v in pair_means.items()}
    nn_data = {
        "model_state_dict": model.state_dict(),
        "peak_to_idx": peak_to_int,
        "gene_to_idx": gene_to_int,
        "d_embed": d_embed,
        "d_atac": d_atac,
        "atac_hidden_dims": atac_hidden_dims,
        "dropout": dropout,
        "pair_means": pair_means_serializable,
        "rna_log_normalized": cfg.get("log_normalize_rna", True),
    }
    transfer_models = {"type": "nn", "data": nn_data}

    logger.info(f"NN fitting complete: {len(transfer_functions)} pairs")
    return transfer_functions, transfer_models


# ============================================================================
# Inference functions
# ============================================================================


def _load_nn_from_transfer_models(transfer_models: dict) -> PairEmbeddingNN:
    """Restore PairEmbeddingNN from the transfer_models dict."""
    _check_torch()
    data = transfer_models["data"]
    n_peaks = len(data["peak_to_idx"])
    n_genes = len(data["gene_to_idx"])
    model = PairEmbeddingNN(n_peaks, n_genes,
                            data["d_embed"], data.get("d_atac", 16),
                            data.get("atac_hidden_dims", [32, 64, 128]),
                            data["dropout"])
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    return model


def _nn_predict_delta_r(
    model,
    a_orig: float,
    a_new: float,
    peak_id: str,
    gene: str,
    peak_to_idx: dict,
    gene_to_idx: dict,
    device: str = "cpu",
) -> float:
    """NN prediction of delta_r = f(a_new) - f(a_orig).

    Defined over the entire real domain, with no OOB check.
    """
    _check_torch()

    p_idx = peak_to_idx.get(peak_id)
    g_idx = gene_to_idx.get(gene)
    if p_idx is None or g_idx is None:
        raise KeyError(
            f"NN transfer mapping missing for peak={peak_id!r}, gene={gene!r}"
        )

    t_device = torch.device(device)
    model.to(t_device)
    model.eval()

    with torch.no_grad():
        atac = torch.tensor([[a_orig], [a_new]], dtype=torch.float32, device=t_device)
        p_t = torch.tensor([p_idx, p_idx], dtype=torch.long, device=t_device)
        g_t = torch.tensor([g_idx, g_idx], dtype=torch.long, device=t_device)
        preds = model.forward(atac.squeeze(-1), p_t, g_t)
        delta = float((preds[1] - preds[0]).cpu().item())
        if not np.isfinite(delta) or not bool(torch.isfinite(preds).all().item()):
            raise FloatingPointError(
                f"NN transfer prediction is non-finite for peak={peak_id!r}, "
                f"gene={gene!r}"
            )
        return delta
