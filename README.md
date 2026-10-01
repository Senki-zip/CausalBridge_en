# CausalBridge

**A Causal Perturbation Prediction Framework Based on ATAC Bridging**

Leveraging single-cell multi-omics data (scRNA-seq + scATAC-seq), CausalBridge uses chromatin accessibility as the "mechanistic bridge" to build a three-layer TF → Peak → Gene causal regulatory network, enabling prediction of genome-wide transcriptional + chromatin state shifts after in silico gene knockout.

---

## 1. Core Idea

Traditional methods (e.g., CellOracle) infer regulatory relationships from co-expression correlation. The key insight of CausalBridge is: **within the same cell, the ATAC signal captures the state of chromatin openness at an earlier time point, while the RNA signal captures the transcriptional state at a later time point** — this "cell-state parallax" makes Granger causality testing possible.

Perturbation propagation path:

```
RNA change → TF activity change → ATAC change (peak opening/closing) → target gene transcription change → RNA change
```

---

## 2. Model Architecture: Three-Layer Causal Network

### Layer 1 RNA→ATAC: TF → Peak (protein→DNA)

| Stage | Method | Description |
|------|------|------|
| 1. Motif scanning | JASPAR / gimmemotifs + CIS-BP | Scan the peak DNA sequences for TF binding motifs to establish candidate TF→peak relationships |
| 1.5 Family expansion | Two-stage: passive borrowing + full-family PWM scanning | Stage A: fast borrowing of family-member matches based on the cache; Stage B: actively scan all causal peaks with full-family PWMs, and matches with p<0.1 are credited to the target TF. Only target TFs are expanded; non-target families are not expanded |
| 2. Per-TF lag-wise independent Pearson | Single-feature Pearson r + t-test per lag | TF(t-lag) → Peak(t): each motif-selected candidate TF is tested independently, with only 1 column of X per lag; Pearson r + t-test + early-stop is performed independently per lag, and the first significant lag wins |

- **Per-TF independent fitting**: for each peak, a single-feature X is constructed separately for every motif-selected candidate TF, rather than pooling all TFs into a joint design matrix.
- **Lag-wise independent testing + early-stop** (v1.33): lag=0..max_lag is tested one by one independently, and the first significant lag wins (p < 0.05). All lags share the same Y (same number of rows after alignment), ensuring comparable statistical power; early-stop avoids spurious significance at long lags
- **Shared-Y alignment**: `Y = Peak[max_lag : n_bins]`, `X_k = TF[max_lag-k : n_bins-k]`. All lags of the same (TF, peak) pair are compared at the same sample size
- **Single-feature Pearson path**: each lag's X has only 1 column, so `pearsonr(X, Y)` + t-test directly gives r/p/weight/contribution amount; `weight = r × sd(Y) / sd(X)`.
- **Per-edge lag labels**: each (TF, peak, lag) edge is kept independently; during perturbation propagation each TF propagates backward along the pseudotime axis by its own lag

### Layer 2 ATAC→RNA: Peak → Gene (DNA→RNA)

| Stage | Method | Description |
|------|------|------|
| 1. Granger causality test | Time-lagged F-test + FDR correction + degrees-of-freedom correction | On pseudotime-binned data, tests whether peak(t-1) significantly improves the prediction of gene(t) |
| 2. Causality filtering | Soft-thresholded composite score + degrees-of-freedom correction | Computes composite_score by combining p_adj and ΔR²; p/ΔR² thresholds are scaled quadratically by df, automatically tightening for clusters with few bins |
| 3. Transfer function | NN shared neural network | Fits the nonlinear mapping R_gene(t) = f(A_peak(t-1)) (lag=1), capturing saturation/threshold effects |

**Core equation**: R_gene(t) = f(A_peak(t-1)), with lag=1 consistent with the Granger test

**NN mode** (the sole ATAC→RNA path since v1.39; the `atac_to_rna.method` switch was removed together with the GP path):
- Shared neural network PairEmbeddingNN; all pairs share the same nonlinear basis functions
- **Monotonicity constraint**: all weights on the ATAC pathway are >= 0 → each basis function f_i(A) is monotonically increasing in ATAC
- **Pair specificity**: peak + gene embedding → learns combination weights w_i and bias, free to be positive or negative
- The gain sign is determined uniquely by Σw_i: Σw_i > 0 → activation, Σw_i < 0 → repression
- **Sign verification**: the Pearson r = corr(peak(t-1), gene(t)) is used to verify the NN gain sign, and the sign is flipped when inconsistent. Pearson r is not subject to competition from the Y(t-1) autoregressive term, giving higher directional accuracy than the Granger β₂ (atac_coef) (55.8% vs 50%)

### Layer 3: TF co-expression network (protein→protein, supplementary link)

- **Proxy TF mapping**: when a non-TF gene is knocked out, RNA changes are assigned to downstream TFs according to co-expression weight
- **Family expansion (two-stage strategy)**: Stage A passive borrowing (cache-based, 0 cost); Stage B forced full-family PWM scanning (for each target TF, all causal peaks are scanned with the PWMs of all its family members; any match → credited to that target TF). Only target TFs in the KO list are expanded; non-target families are not expanded, avoiding a C2H2_ZF explosion

---

## 3. Perturbation Simulation Engine

### Per-bin forward simulation (bin_forward)

```
Initialization:
  rna_pert = rna_binned.copy(); atac_pert = atac_binned.copy()
  Bin-aware KO: applied only where target gene expression > 0 and a downstream pathway with t+lag<n_bins exists
    rna_pert[active_mask, ko_gene] *= (1 - ko_strength)
  Non-TF targets: repression is applied to proxy TFs only in the bins actually knocked out

Forward traversal along pseudotime (t = max_lag → n_bins-1):
  [Single-round default] Only already-affected TFs retain their state in rna_pert to drive subsequent TF→peak; non-TFs are not written back into later bins
  [Multi-round mode] rna_pert[t] = rna_orig[t] * rna_fold; atac_pert[t] = atac_orig[t] * atac_fold
  [RNA→ATAC] Traverse all (TF, peak, lag) edges: a change in TF(t - lag) → change in peak(t) accessibility (Pearson weight)
  [ATAC→RNA] Change in peak(t-1) → change in gene(t) expression (NN gain + R² hard filter)
  [Fold-change update] Modified genes/peaks have their fold-change vectors updated

Aggregation: non-TFs are output along the existing delta_cum endpoint path; non-KO TFs accumulated via single-round log are aggregated by the L2→L1 change of their first triggering event and their fraction of remaining bins
```

Core features:
- **Lag-wise independent Pearson + per-edge lag labels**: each (TF, peak, lag) is tested as an independent candidate edge, and the first significant lag wins. Supports lag=0 (instantaneous effect) through lag=L (delayed effect)
- **Bin-aware KO**: knockout is applied only in bins that have expression and a downstream pathway
- **TF-specific state inheritance**: in single-round mode only TF state can enter subsequent RNA→ATAC propagation; multi-round mode uses fold-change to inherit across bins
- **Multi-gene joint KO**: all target_genes are knocked out simultaneously, with effects automatically crossing
- **R² hard filter**: peak→gene edges with R² < min_r2_threshold are discarded outright

### Subnetwork extraction (BFS)

Starting from the target gene, a breadth-first search extracts a multi-layer regulatory neighborhood:

```
Layer 0: target_gene
Layer 1: direct causal peaks (from Granger + TF→peak weights) + the genes regulated by these peaks
Layer 2+: recursive expansion (depth controlled by subnetwork_depth; both the shipped config.yaml and the code fallback use "unlimited")
```

### Key mechanisms

| Mechanism | Description |
|------|------|
| **Lag-wise independent Pearson + per-edge lag** | Each (TF, peak, lag) is tested as an independent candidate edge, and early-stop selects the first significant lag; weight = r × sd(Y) / sd(X) |
| **Bin-aware KO** | Knockout is applied only in bins where target gene expression > 0 and a downstream pathway exists |
| **TF-specific state inheritance** | In single-round mode only TF state enters subsequent RNA→ATAC; in multi-round mode inheritance across bins is via fold-change |
| **R² hard filter** | peak→gene edges with R² < min_r2_threshold (0.3) are discarded outright |
| **Proxy TF mapping** | When a non-TF gene is knocked out, effects are assigned to downstream TFs by co-expression weight |
| **Multi-gene joint KO** | All target_genes are knocked out simultaneously, with crossing/overlapping effects handled automatically |

---

## 4. Module Structure

```
atac_bridge/
├── run.py           Main pipeline entry; chains the full workflow (including checkpoint resume)
├── io.py            Config file parsing, data loading, result saving
├── preprocess.py    RNA/ATAC QC and normalization, PCA + KNN + Leiden clustering + UMAP
├── granger.py       Pseudotime inference + Granger causality testing + soft-threshold filtering
├── kinetics.py      NN transfer-function fitting dispatch + motif scanning + lag-wise Pearson + family expansion
├── nn_transfer.py   NN transfer-function implementation (PairEmbeddingNN training/prediction/sign verification)
├── perturbation.py  Perturbation simulation engine (subnetwork extraction, two-step propagation, proxy TFs)
```

### Auxiliary files

```
run_example.py                      Command-line entry script
run_peak_perturbation.py            Standalone CLI: peak perturbation trial runs (checkpoint reuse; try peaks/strengths/depths in seconds)
convert_10x_to_atac_bridge.py       Convert 10x output to the atac_bridge input format
plot_causal_network.py              Python visualization: causal network plots
visualize_perturbation.R            R visualization: Volcano + regulatory pathway plots
target_genes.txt                    List of knockout target genes (supports TF/TARGET type annotations)
config.yaml                         User configuration file
FIGUERS_README.md                   Figure-generation instructions
README.md                           This document
```

---

## 5. Analysis Pipeline

```
Step 1:  Load config + data (RNA + ATAC AnnData)
Step 2:  Preprocessing (QC, normalization, highly variable genes, ATAC binarization) + PCA + KNN + Leiden clustering
Step 3:  Per-cluster pseudotime inference (Palantir / DPT; per-cluster inference since v1.29) + Gaussian soft binning
Step 4:  Per-cluster Granger causality testing
Step 5:  Per-cluster NN transfer-function fitting
Step 6:  Per-cluster learning of RNA→ATAC TF regulatory weights (motif scanning + per-TF/per-lag single-feature Pearson)
Step 7:  Causal GRN integration
Step 8:  Per-cluster perturbation simulation (bin_forward propagation)
Step 9:  Result aggregation and saving
```

Advantages of cluster-wise modeling:
- TF regulatory logic is cell-type specific
- Effectively filters out false-positive causal edges that cross cell types

---

## 6. Usage

### Quick start

```bash
# 1. Prepare input files
#    - rna_adata.h5ad (.var must contain chr, start, end)
#    - atac_adata.h5ad (.var must contain chr, start, end)
#    - target_genes.txt (one gene symbol per line)

# 2. Edit config.yaml to change the input paths

# 3. Run
python run_example.py config.yaml
```

### Python API

```python
from atac_bridge.run import run_pipeline
results = run_pipeline("config.yaml")
```

### Target gene file format

```text
# Single-column format (default gene_type=TF)
Sox2
Pou5f1

# Two-column format (types specified)
Nanog    TF
Rfx8     TF
Actb     TARGET
```

---

## 7. Output Files

```
results/
├── perturbation_results.csv           Core results: KO gene × affected gene × ΔRNA × mechanism (per cluster)
├── perturbation_merged.csv            Tissue-wide weighting: weighted sum of |ΔRNA| (avoiding cross-cluster sign cancellation), includes the delta_rna_signed column
├── perturbation_pathways.csv          Path-level edge records: independent TF→peak→gene contributions (with TF proportional attribution, see column notes below)
├── causal_peak_gene_edges.csv         Causal peak→gene edges (with composite_score)
├── tf_peak_weights.csv                TF→peak regulatory weights (lag-wise Pearson weights)
├── aggregated_tf_peak_edges.csv       TF→peak edges aggregated across clusters (deduplicated merge)
├── atac_changes.bed                   ATAC changes (IGV visualization)
├── atac_changes_significant.bed       Significant ATAC changes (|z| ≥ atac_significance_zscore, default 2.0)
├── atac_changes_merged.csv            ATAC changes merged across clusters
├── atac_changes_significant_merged.bed Significant ATAC changes merged across clusters
├── rna_clustered.h5ad                 Clustered RNA AnnData (with PCA / KNN / Leiden / UMAP)
├── cluster_top50_genes.csv            Per-cluster Wilcoxon differential-expression marker genes (rank_genes_groups, validates the cluster mapping)
├── peak_perturbation_results.csv      (peak KO mode) core peak perturbation results
├── peak_upstream_tfs.csv              (peak KO mode) upstream TF attribution (per cluster)
├── peak_upstream_tfs_merged.csv       (peak KO mode) upstream TF attribution (weighted merge by cluster cell counts)
├── peak_perturbation_report.txt       (peak KO mode) four-section report (A perturbed peak / B upstream TFs / C direct effects / D summary)
├── perturbation_cell_projection.csv   (optional) cell-level reference projection results
├── perturbation_projection_summary.csv (optional) reference projection summary
├── perturbation_projection_branch_probabilities.csv (optional) branch probabilities
├── perturbation_projection_bin_history.csv (optional) projection bin history
├── perturbation_projection_bin_metadata.csv (optional) projection bin metadata
├── perturbation_projection_events.csv (optional) projection events
└── intermediate/                      Per-cluster intermediate results (written to disk under the results dict key names)
    ├── granger_results.csv
    ├── causal_edges.csv
    ├── transfer_functions.csv
    ├── tf_peak_weights.csv
    ├── pathway_edges.csv
    ├── aggregated_tf_peak_edges.csv
    ├── atac_changes.csv
    ├── perturbation_results.csv
    └── perturbation_0.csv, perturbation_1.csv, ...  (perturbation results saved per cluster)
```

**Column notes for perturbation_pathways.csv:**

| Column | Description |
|------|------|
| target_gene | KO target gene |
| tf_name | Name of the regulating TF |
| peak_id | Genomic coordinates of the mediating peak |
| affected_gene | The affected target gene |
| mechanism | Transfer-function type (nn; linear fallback when NN fails) |
| delta_r_raw | The **exclusive effect** of this TF on affected_gene (attributed by the TF→peak contribution proportion) |
| r2_score | R² of the peak→gene NN fit |
| delta_r_total | The **total effect** of this (peak, gene) edge (the sum of delta_r_raw over all TFs) |
| delta_rna_log2fc | log2 fold change of affected_gene (aligned with perturbation_results) |
| tf_lag | Pseudotime lag steps of this TF→peak edge (per-edge lag) |
| bin_min / bin_max | Pseudotime bin range where this contribution occurred/was aggregated |
| cluster | Source cell cluster |

`mechanism=direct-interaction` is a legacy fallback label in the results files: it is used when the current check does not identify any qualifying ATAC-mediated paths. It does **not** prove that a direct RNA regulatory edge exists; the preferred interpretation is "non-ATAC-mediated/unresolved".

### Cell-level reference projection (optional)

`perturbation.cell_projection.enabled` controls RNA reference projection. It is an optional output feature, not a replacement for the default perturbation results; when enabled and successfully generated, it saves the `perturbation_*projection*.csv` files listed in the table above. The projection results can be combined with the UMAP coordinates in `rna_clustered.h5ad` for cell-level UMAP display; when no projection files are generated, an ordinary cluster/bin UMAP plot should not be interpreted as a cell-level projection result.

> **On delta_r_raw and delta_r_total having opposite signs:**
>
> When multiple TFs jointly regulate the same peak, `delta_r_raw = delta_r × (tf_contrib / total_contrib)`. If a TF is a repressive regulator (its `tf_contrib` direction is opposite to `total_contrib`), its `frac` is negative, so `delta_r_raw` takes the opposite sign of `delta_r_total`. This is correct causal-attribution behavior — a repressive TF pulls in the opposite direction in the "tug of war". About 5-6% of edges show this, reflecting the competition between activating and repressive TFs.

---

## 8. Key Improvements and Recent Changes

### v1.39 — Transfer functions become NN-only + removal of lag modes 2/3, linear decay, and dead code (2026-09-16)

**Background**: `atac_to_rna.method` has long defaulted to `nn`; GP fitting and Bagging stability assessment no longer participate in the regular workflow; time-lag modes 2/3 and linear decay were already marked as kept for compatibility. This release removes all inactive paths in one pass and adds safe semantics for checkpoints and inference failures.

**Changes**:

| Item | Description |
|------|------|
| **ATAC→RNA becomes NN-only** | Removed the `atac_to_rna.method` switch and the GP fitting implementation (GP training, triage, kernel construction, serialization, and type dispatch in `kinetics.py`; GP posterior gradients, path integration, point-prediction helpers, and the GP branches of the three propagation paths in `perturbation.py`) |
| **Removed GP-only configuration** | Deleted `gp_kernel`, `gp_quality_threshold`, `gp_n_restarts`, `delta_r_method`, `path_integral_steps` |
| **Removed Bagging** | Deleted `rna_to_atac.bagging_n_estimators`, `bagging_stability_threshold`, the Bootstrap stability assessment in `_fit_tf_peak_regression`, and the `weight_std`/`bagging_stability` output columns |
| **Removed lag modes 2/3** | Deleted `multi_lag`, `auxiliary_lags`, and the multi-column joint design-matrix path; with `time_lag: true` only lag-wise independent single-feature Pearson tests + early-stop remain |
| **Removed linear decay** | Deleted `lag_penalty`, `lag_penalty_mode` |
| **Removed dead code** | Deleted the unreachable RidgeCV branch in `_fit_tf_peak_regression` along with the now-unused `variance_rescale` parameter; also deleted `_select_optimal_lag` and `_select_by_vote_and_rank` |
| **Explicit failure on empty training data** | When the NN training set is empty, an actionable `RuntimeError` is raised (checking causal edges / binning / `min_observations`) instead of silently falling back to the removed GP |
| **Linear fallback on inference failure** | Missing peak/gene indices or non-finite NN predictions are marked as inference failures and take the existing `gain × delta_a` linear fallback, instead of treating `0.0` as a valid prediction and silently erasing the edge |
| **Normalization metadata validation** | All three propagation paths read `rna_log_normalized` from `transfer_models` and refuse when it disagrees with the current `atac_to_rna.log_normalize_rna`, avoiding use of the wrong accumulation space when reusing old checkpoints |
| **Old checkpoint rejection** | Step 5 checkpoints that are missing or non-NN require deletion and rerun; Step 6 checkpoints containing Bagging-derived columns such as `weight_std`/`bagging_stability` are likewise rejected |
| **Small-sample NN split fix** | The training/validation split is now a bounded ratio; the degenerate case of a 0-size training set no longer occurs when `n < 1000` |
| **Kept** | `_branch_aware_lag_pairs` (still used by Granger); `time_lag: false` still forces `max_lag=0`; `fit_atac_to_rna` still returns the `(transfer_functions, transfer_models, root_atac)` triple |
| **New tests** | `tests/test_oracle_fixes.py`: NN split, inference-failure fallback, normalization metadata mismatch, rejection of old Step 5/6 checkpoints |

**Impact**: existing Step 5 / Step 6 checkpoints must be deleted and rerun; the GP, Bagging, mode 2/3, and linear-decay configuration entries in `config.yaml` and `io.py` are no longer read. The documentation has been updated to the NN-only + lag-wise independent Pearson design.

### v1.38.2 — Single-round TF RNA output weighted by event time (2026-09-10)

**Problem**: in the default single-round `bin_forward`, the TF's final RNA output is computed by directly averaging the propagation matrix `rna_pert`. That matrix exists to maintain the propagation state for TF→peak, and its internal L2 deltas participate directly in its per-bin updates; therefore it is not a suitable readout of the TF's final L1 RNA response.

**Fix**: the propagation simulation itself is unchanged. For each first-triggered `peak→gene` edge whose target gene is an executable TF, the model L2 delta is first back-converted on top of the original L1 expression at the triggering bin:

```text
ΔL1_event = expm1(log1p(L1_orig[t]) + ΔL2_event) - L1_orig[t]
```

Only the output contribution is recorded; nothing is written back into later bins and the TF→peak input is unchanged. The final mean contribution of the event is weighted by the fraction of remaining bins:

```text
weighted_ΔL1 = ΔL1_event × (n_bins - t) / n_bins
TF_mean_KO = TF_mean_WT + initialization_offset + Σ weighted_ΔL1
log2FC = log2(TF_mean_KO / TF_mean_WT)
```

Here `initialization_offset` preserves the direct repression effect on proxy TFs from non-TF KOs. Non-TFs still use the existing `delta_cum` endpoint reconstruction path; `log_accumulation: false` keeps the legacy behavior. If the simulated L1 endpoint is invalid (`<=0`), the current result is kept as `NaN`; downstream result filtering may omit that row. This is a numerical validity state, not a biological zero effect.

**Also fixed: propagation depth**. `propagation_depth` now denotes the actual number of TF→peak→gene graph hops from the KO source to a gene, not the pseudotime lag; edges with zero value or non-finite `delta_r` are no longer wrongly counted as propagated.

### v1.38.1 — Peak perturbation wrap-up: absolute-mode validation + explicit message when no upstream TF + removal of top_n (2026-08-31)

**Changes**:

| Item | Description |
|------|------|
| absolute mode verified | `peak_ko.mode: absolute` (A′=s, a raw [0,1] target value) passed all 7/7 synthetic smoke tests: `strength=0.9` → perturbed=0.9000 exactly; the opening-enhancement direction is correct (positive-gain genes up-regulated / negative-gain down-regulated), serving as the counterpoint to relative's direction-closing |
| no-upstream-TF message | When the upstream list in section B of the report is empty, it outputs `no confident upstream TF identified` plus a reason hint (incomplete motif database / low TF expression / Ridge not passed), aligned with the development notes §13 |
| removal of `top_n` | The display-layer truncation requirement was dropped; the `peak_ko.top_n` parameter and its references in the README/plan archive were deleted — peak perturbation results are saved in full, with no truncation layer |

### v1.38 — Peak perturbation Phase 2: cascade propagation + state filtering + weighted merge + standalone CLI (2026-08-31)

**Feature**: builds full propagation capability on top of v1.37's peak direct mode, making peak perturbation a configurable end-to-end feature.

**Additions**:

| Capability | Description |
|------|------|
| Cascade propagation | `peak_ko.depth: 2/3` enables the full bin loop (8a RNA→ATAC + 8b ATAC→RNA + TF cascade): if a direct target gene is a TF, its RNA change triggers secondary peaks → secondary genes, with BFS depth controlling the number of layers. The loops of `_simulate_peak_propagation_forward` and `_simulate_knockout_bin_forward` are isomorphic (first-fire of fired_edges / L2 accumulation / pathway attribution all reuse the same semantics) |
| state single-cluster filter | `peak_ko.state: "0"` runs only the specified Leiden cluster; `"all"` (default) runs all clusters |
| Weighted merge of upstream attribution | `peak_upstream_tfs_merged.csv`: weighted summary by cluster cell counts (weight_signed_merged / weight_abs_merged / n_clusters_observed), aligned with the perturbation_merged approach |
| Standalone CLI | `run_peak_perturbation.py --config X --peak chr3:... --strength -1 --depth 2`: derives a peak config + isolated output directory + checkpoint reuse; try peaks/strengths/depths in seconds |

**Validation** (RUNX1 checkpoint reuse): propagation-mode smoke tests passed (synthetic-data depth-2 cascade: peak→G0→TF cascade triggers a secondary peak); state filtering and weighted merging were verified in real runs.

**Backward compatibility**: `depth` defaults to 1 (direct, behavior identical to v1.37); `state` defaults to all.

### v1.37 — Peak perturbation (peak KO): mode switch + direct propagation + upstream attribution (2026-08-31)

**Feature**: the user specifies a peak and manually sets its openness change; the model propagates along the existing TF→Peak→Gene network to predict affected genes, and also returns candidate upstream TFs for that peak with their support. **This is not a new model — it is just a second intervention entry point for the existing bin_forward engine** — upstream TFs are used only for attribution and do not participate in reverse causal propagation (engine step 8a only reads the TF RNA delta, which naturally guarantees this boundary).

**Core mechanisms**:

| Mechanism | Description |
|------|------|
| Mode switch | `perturbation.knockout_type`: `"gene"` (default, zero change to the existing workflow) / `"peak"` (new) |
| ΔR reuses the transfer function | `_nn_predict_delta_r` — f(A′)−f(A), no new regression added |
| direct mode | By default only the direct target genes of that peak are propagated, with no 8a cascade (the depth parameter is reserved for propagation extension) |
| Four-level matching | exact → overlap → nearest (with the distance output explicitly, no silent substitution) → `peak not represented in inferred regulatory network` |
| Extrapolation guard | When perturbed goes beyond the training-visible range of the peak, an extrapolation warning is emitted |
| Perturbation space | Raw [0,1] definition (ATAC = binarized openness proportion; raw 0 = truly closed), entering the engine's [-1,1] via Scheme B scaling |

**Change points**:

| File | Content |
|------|------|
| `perturbation.py` | Added `_match_peak` (four-level matching), `_extract_peak_subnetwork` (peak-rooted BFS), `_simulate_peak_direct_forward` (direct-propagation core, reusing the engine's binning/scaling/L2 accumulation/output conventions), `perturb_peak` (public entry point) |
| `run.py` | Step 8 dispatches on `knockout_type`; the aggregation chain collects `upstream_tfs` + `peak_summary` |
| `io.py` | Added `peak_perturbation_results.csv`, `peak_upstream_tfs.csv`, `peak_perturbation_report.txt` (four sections: A perturbed peak / B upstream TFs / C direct effects / D summary) |
| `config.yaml` | `perturbation.knockout_type` + `perturbation.peak_ko` (peak_id/mode/strength/state/depth/match) |

**Validation** (RUNX1 data, checkpoint reuse): synthetic-data smoke tests 15/15 passed; in a real run, closing `chr10:1008923-1009780` down-regulated GTPBP4 in 3 clusters (-0.526/-0.088/-0.124), with 2 upstream RUNX1 attributions (weight ±0.66/-0.75) and a complete four-section report across five clusters.

**Backward compatibility**: `knockout_type` defaults to `"gene"`; without configuration, behavior is identical to v1.36.2.

### v1.36.1 — ATAC change significance: WT null-distribution z-score (2026-07-15)

**Problem**: `atac_changes.bed` contained only the full set of changes (194 thousand rows) with no significance criterion, making meaningful comparison against ATAC-seq results from real KOs impossible.

**Solution: null-distribution z-score based on WT data**

| Change point | File | Description |
|---|---|---|
| **z-score computation** | `perturbation.py` near L1505 | For each peak, `z = delta_accessibility / std(atac_binned_orig)`. `atac_binned_orig` is the bin-mean matrix of the WT input after pseudotime binning; the std across bins = the natural fluctuation of that peak along the developmental trajectory |
| **atac_changes enhancement** | `perturbation.py:_build_atac_changes_df` | The DataFrame gains `z_score` + `is_significant` (|z| ≥ threshold) columns; in cross-gene mode the value with the largest \|z\| per peak is taken |
| **Significant BED output** | `io.py:save_results` | Added `atac_changes_significant.bed` (default |z| ≥ 2.0); the full `atac_changes.bed` is kept for backward compatibility |
| **Configuration** | `config.yaml` | `perturbation.atac_significance_zscore: 2.0`, adjustable |

**Statistical meaning**: the z-score measures how many multiples "the model-predicted KO change" is relative to "the natural fluctuation of the WT data itself along pseudotime". |z| ≥ 2 means the change magnitude of that peak exceeds 95% of the natural fluctuation in WT (under a normal approximation) — a conservative significance standard. The threshold can be tightened via config (e.g., 3.0 = stricter) or relaxed.

**Cross-cluster handling**: each peak is modeled independently in multiple clusters; when deduplicating, the instance with the largest \|z\| is taken (consistent with taking the maximum absolute delta_atac), avoiding cross-cluster sign cancellation.

**Usage for comparison with real KO data**:

1. Compute `log2FC(KO/WT)` for the real KO ATAC-seq over the same peak set
2. Match peak coordinates against `atac_changes_significant.bed`
3. Compute the Spearman correlation (model z sign vs real log2FC sign), or test whether significant peaks are enriched in the regions with the largest real changes

---

### v1.36 — Log-space delta accumulation: fixes the systematic up-regulation bias of perturbations (2026-07-13)

**Problem**: the perturbation results for RUNX1 KO (5 hematopoietic clusters) were systematically biased positive: the per-cluster mean delta_rna was positive in every cluster (+0.155 ~ +2.361), and after merging 80.5% of genes were up-regulated. Even where the positive/negative counts were nearly 1:1 (cluster 2: 52.7%/47.3%), the mean was still +0.155 — proving the bias lies in **magnitude, not just counts**.

**Root cause (mathematical layer)**:

The NN is trained in `log1p∘log1p(RNA)` space (denoted L2) and outputs symmetric ±δ log-deltas. The original code, on every firing of a (TF→peak→gene) edge, called `_delta_r_log_to_raw` to convert the L2 delta back to L1 (`log1p(raw)`) before accumulating into `delta_cum`:

```python
delta_r_L1 = expm1(log1p(L1_orig) + delta_r_L2) − L1_orig
delta_cum[gene] += delta_r_L1       # accumulated in L1, with expm1 applied per edge
```

The nonlinearity of `expm1` makes the positive direction grow exponentially while the negative direction is compressed toward −1, so a symmetric L2 input yields an asymmetric L1 accumulation:

| Input (L2) | Converted back to L1 (r_orig=0.693) | Accumulated contribution |
|---|---|---|
| +1.0 | expm1(0.693+1) − 0.693 = **+3.74** | +3.74 |
| −1.0 | expm1(0.693−1) − 0.693 = **−0.96** | −0.96 |
| **sum** | | **+2.78** (should theoretically be 0) |

**Fix (route 3: symmetric accumulation layer + legacy convention preserved at the output layer)**:

| Change point | Line numbers (perturbation.py) | Change |
|---|---|---|
| **Flag** | near L1130 | Added `_log_accumulation = bool(_rna_log_normalized and cfg.log_accumulation)` (default True) |
| **edge firing** | L1352-1357 | When `_log_accumulation=True`, **skip** `_delta_r_log_to_raw` and directly `delta_cum[gene] += delta_r` (the NN's native L2 delta) |
| **per-bin TF inheritance** | L1251 | Changed to `rna_pert[t,active] = expm1(log1p(rna_binned_orig) + delta_cum[active])`, maintaining the L1 scale for downstream Ridge delta_tf |
| **TF cascade update** | L1394 | Same back-conversion as above |
| **Final output** | L1475-1504 | The accumulation layer stays in L2 (multiplicative cancellation is correct); the final output back-converts to L1 and then computes `log2(L1_pert/L1_orig)`; invalid L1 endpoints are kept as `NaN`, not fabricated finite effect values |
| **config switch** | config.yaml `perturbation.log_accumulation` | Default `true`; `false` takes the legacy raw path for regression comparison |

**Effect** (validated on synthetic data):

| Input L2 delta | Legacy raw accumulation | v1.36 log accumulation |
|---|---|---|
| +1.0 | +2.91, −1.07, **sum=+1.84** | +1.0, −1.0, **sum=0.000** |
| single-edge +1 log2FC | +2.378 | **+2.378** (magnitude 100% consistent) |

**Design philosophy**: the accumulation layer uses L2 space to guarantee symmetric cancellation of +δ/−δ (eliminating the positive bias amplified exponentially by raw accumulation), while the output layer uses `log2(L1_pert/L1_orig)` to preserve the true numerical state; invalid endpoints are represented as `NaN` rather than converted into a zero effect or an artificial `-5`.

**Backward compatibility**: `perturbation.log_accumulation: false` switches back to the original raw path, for comparison or regression testing.

---

### v1.35 — CytoTRACE 2020 root-cell selection (Plan B: global computation → within-cluster mapping) (2026-07-10)

**Problem**: the old `_find_root_cell` fallback used `np.argmin(total_umi)` — the cell with the lowest total UMI count became the DPT root. That is the standard DPT-paper fallback, but it is sensitive to sequencing-depth differences and small-cluster noise, uses only the single scalar UMI, and discards the expression-coordination signal.

**Approach B: global CytoTRACE → in-cluster argmax mapping**

| Stage | File | Description |
|------|------|------|
| **Global computation** | `granger.py:_compute_cytotrace_scores`, `run.py` Step 2.5 | After QC/clustering and before the cluster loop, compute the CytoTRACE 2020 differentiation-potency score over the whole dataset (Gulati et al., Science 2020) and write it to `rna_adata.obs["cytotrace_score"]` |
| **In-cluster mapping** | `granger.py:_find_root_cell` | Within each cluster take `np.argmax(obs["cytotrace_score"])` over its cells as `iroot`; since `.obs` columns are naturally retained on slicing, no explicit argument is needed |
| **Degradation safety** | Same as above | If the global score is missing (granger.py used standalone), recompute it on the fly within the current cluster (equivalent to Approach A) |

**CytoTRACE 2020 algorithm**: each cell combines two signals → a stemness score ∈ (0, 1], where higher = more stem-like / less differentiated.

The CytoTRACE score here is a globally computed stemness/root prior. It is not pseudotime itself and not a general-purpose trajectory-inference replacement. In the DPT path, each cluster uses the argmax of its local CytoTRACE scores to pick the DPT root and then infers that cluster's pseudotime; so it should not be read as an all-purpose trajectory replacement.

| Signal | Computation | Stemness direction |
|------|------|---------|
| Gene Count (gc) | Number of genes with expression > 0 | Fewer → more stem-like |
| Mean Pairwise Correlation (mpc) | Mean pairwise Pearson correlation among the cell's own top-N highly expressed genes (correlations computed over the whole dataset) | Higher → more stem-like |

Final score = `(rev_rank(gc) + rank(mpc)) / 2`, both normalized to (0, 1].

**New configuration** (see the §9 table):

| Config | Default | Description |
|------|------|------|
| `pseudotime.root_cell_method` | `"cytotrace"` | `cytotrace` / `marker` / `min_umi` / `auto` (auto: fall back to cytotrace when marker is missing) |
| `pseudotime.cytotrace_n_top_genes` | `200` | CytoTRACE takes the top-N highly expressed genes per cell |

**Backward compatibility**: `root_cell_marker` behavior is unchanged. Marker is still used only when `root_cell_method: "marker"`, or when method is absent and a marker is given; every other case goes through CytoTRACE.

**Comparison with CellOracle/Palantir**: Palantir entropy is a pseudotime result rather than a prior, so it cannot be used to select the root; CytoTRACE computes stemness directly from the expression matrix and is a true prior signal, which is why it works well with DPT.

---

### v1.34 — Cofactor support + proxy-TF weight/threshold optimization (2026-06-08)

This version addresses regulatory modeling of non-DNA-binding cofactors (such as LMO2) and optimizes proxy-TF weight assignment and the upstream-exclusion logic.

---

**1. Internal cofactor recognition: from motif2factors to the regulatory network**

**Background**: cofactors such as LMO2 do not bind DNA directly (they have no PWM of their own), yet they participate in transcriptional regulation through protein interactions within TF complexes. Under the old logic these molecules were labeled TARGET and went through the co-expression proxy-TF path, losing their direct regulatory information.

**Approach**: use the `motif2factors` mapping that ships with the motif database to bring cofactors into the regulatory network automatically.

```
motif2factors.txt:
  GATA.0004 → [{factor: GATA1}, {factor: LMO2}, ...]
  bHLH.0031 → [{factor: LMO2}, {factor: NHLH2}, ...]

Old: TF-centric parsing (tf_best), pick the best PWM per TF
    → motif_db[GM_GATA.0004] = {tf_name: GATA1, pwm}
    → no all_factors, so cofactor associations such as LMO2 are lost

New: motif-centric parsing → TF-centric dedup (_dedup_motifs_by_tf)
    ① Parsing keeps the complete all_factors of every motif
    ② Dedup reuses the old tf_best logic (IC + curated bonus),
       selects the best PWM per TF independently and merges that TF's
       all_factors across all motifs
    → motif_db[GM_GM.5.0.bHLH.0031] = {tf_name: LMO2, pwm, all_factors: [LMO2, NHLH2, ...]}
    → motif_db[GM_GATA.0012] = {tf_name: GATA1, pwm, all_factors: [GATA1, GATA2, TAL1, ...]}
    ③ A motif hitting a peak → all factors in all_factors join the candidates
    ④ Ridge regression → cofactors and TFs compete on equal footing and weights are fitted
```

| File | Change |
|------|------|
| `kinetics.py` `_parse_gimmemotifs_pfm` | Motif-centric parsing, keeps the complete `all_factors`, stores `pfm` for IC computation |
| `kinetics.py` `_parse_cisbp_pfm` | Same as above |
| `kinetics.py` `_dedup_motifs_by_tf` | **New**: reproduces the old `tf_best` dedup (IC + curated bonus), merges all_factors, avoids key conflicts |
| `kinetics.py` `_scan_peaks_with_jaspar` | Adds every factor in `all_factors` to `peak_to_tfs` when a motif hits |
| `kinetics.py` `_build_family_groups` | Iterates `all_factors` to collect every factor into its family group |

**Effect**: cofactors are still labeled `TF` externally (no third type is needed in `target_genes.txt`); internally they are associated with peaks through `all_factors` and follow the TF propagation path to regulate downstream genes directly.

---

**2. Proxy-TF weights: normalization → direct |r|**

**Old logic**: `weight = |r| / Σ|r|`, so weights sum to 1. Weakly correlated TFs dilute strongly correlated TFs.

**New logic**: `weight = |r|`, using the correlation coefficient directly. The knockdown magnitude of a strongly correlated TF is no longer affected by weakly correlated TFs.

| Example (ko_strength=1.0) | \|r\| | Old knockdown | New knockdown |
|---|---|---|---|
| TF_A | 0.9 | 47% | 90% |
| TF_B | 0.6 | 32% | 60% |
| TF_C | 0.4 | 21% | 40% |

---

**3. Pseudotime upstream-exclusion threshold: 0.0 → 0.5 bin**

The old threshold `pseudotime_earliness_threshold=0.0` was too strict — a TF was excluded whenever its centroid was earlier than the target gene by even 0.001 bin. After changing it to 0.5 bin, TFs that are essentially co-expressed (Δbin ≤ 0.5) are retained.

---

**4. Silencing Pearson ConstantInputWarning**

The single-feature Pearson path now wraps constant inputs in `warnings.catch_warnings(RuntimeWarning)` to avoid log spam. `np.isnan(_r)` still acts as the final safeguard.

| Change | File | Location |
|--------|------|------|
| Internal cofactor recognition | `kinetics.py` | `_parse_gimmemotifs_pfm`, `_parse_cisbp_pfm`, `_scan_peaks_with_jaspar`, `_build_family_groups` |
| Proxy-TF weights | `perturbation.py` | `_find_proxy_tfs` L334-337 |
| Upstream-exclusion threshold | `perturbation.py` | `_find_proxy_tfs` L203, `_propagate_bin_forward` L873 |
| Pearson warning silencing | `kinetics.py` | Pearson shortcut L2167-2174 |

---

### v1.33 — Independent regression per lag + Pearson significance + single-lag Granger strategy: complete direction-consistency fix (2026-06-08) 🔧 Final version

This version focuses on restructuring the regression strategy of the RNA→ATAC layer (TF→peak) and resolves two problems: a bug in the lag-selection mechanism and a systematic bias in direction consistency. The following sections walk through the design and then each specific fix.

---

**0. Overview of the time-lag strategy across the three-layer causal network**

The three layers of CausalBridge handle "time" differently, each serving a different statistical goal:

| Layer | Method | Lag strategy | Design rationale |
|------|------|---------|---------|
| **RNA→ATAC** (TF→peak) | Per-TF independent regression per lag | Test lag=0..max_lag one by one; early-stop picks the best lag | Highly autocorrelated TFs naturally win at lag=0, while TFs with genuinely delayed regulation still recover their true lag |
| **ATAC→RNA** (peak→gene) | Granger F test + NN transfer function | lag=1 (single lag) | Strictest false-positive control. The lag=1 F(df=1, n-3) has the lowest numerator degrees of freedom and thus the highest significance bar. A joint multi-lag F test inflates false positives (reverted in v1.15→v1.16) |
| **Perturbation propagation** (bin_forward) | Bin-wise forward simulation | Each (TF, peak, lag) edge propagates by its own lag | A TF change at bin(t-lag) affects the peak at bin(t), which the causal time order guarantees |

Key distinction:
- **Granger uses a single lag (lag=1)**: the goal at the Granger stage is to decide whether a causal regulatory relationship exists for peak→gene (yes/no), not to find the optimal delay. The single-lag F test controls false positives most strictly. `granger.max_lag: 1`
- **RNA→ATAC uses a multi-lag search (lag=0..max_lag)**: the goal at the TF→peak stage is to find the optimal delay for each edge while keeping direction information. Independent testing per lag plus early-stop lets the first significant lag win. `rna_to_atac.max_lag: auto` (=n_bins//4)

---

**1. RNA→ATAC design: independent regression per lag**

The `rna_to_atac` layer uses independent regression per lag (the only design), controlled by `time_lag` + `max_lag`:

| Design | Config | X structure | Advantages | Disadvantages |
|------|------|--------|------|------|
| **Independent regression per lag** (only) | `time_lag: true` | A separate single-column test per lag | No collinearity between lags; highly autocorrelated TF→lag=0; true delay→accurate true lag | Runtime ≈ max_lag times |

**Why this design**:

- **Eliminates collinearity between lags**: TF expression is highly autocorrelated across adjacent time bins, so TF(t) and TF(t-1) are nearly collinear. In a joint regression Ridge splits coefficients arbitrarily among the collinear columns, making signs random. Independent regression per lag has only one X column per lag, so collinearity cannot arise
- **Highly autocorrelated TFs naturally win at lag=0**: for highly expressed TFs such as pioneer factors and master regulators, TF(t) ≈ TF(t-1) ≈ TF(t-2)..., so Cov(TF(t), peak(t)) at lag=0 is strongest and early-stop naturally selects lag=0
- **TFs with true delay retain their delay information**: some TFs do have delayed chromatin effects (for example when co-factor recruitment is required), and the per-lag search finds the genuinely optimal lag

**Early-stop plus shared-Y alignment**:

```
Y = Peak[max_lag : n_bins]          ← shared by all lags (same row count = same statistical power)
X_k = TF[max_lag - k : n_bins - k]  ← X for lag k, offset by the lag

for k = 0, 1, ..., max_lag:
    test TF(t-k) → Peak(t)
    if significant (p < 0.05): choose k, break
    else: continue with k+1
```

This guarantees that all lags of the same (TF, peak) pair are compared at the same sample size and therefore with identical statistical power.

---

**2. Problem 1 — per-lag indentation bug: every edge forced to lag=max_lag**

In `_fit_tf_peak_regression`, the roughly 240 lines that determine Y, fit RidgeCV, and save results were indented by 12 spaces, the same level as `for _try_lag in _lags_to_try:` (also 12 spaces), which in fact placed them **outside** the loop. Consequences:

```
for _try_lag in _lags_to_try:     # 12 spaces — loop starts
    # X construction ...          # 16 spaces — each iteration overwrites the previous lag's X
# Y determination + RidgeCV + results   # 12 spaces — outside the loop! always uses the last lag's X
```

All 48,279 ATF4 edges reported `tf_lag=9`, and perturbation output `TF→peak lag=[9]` (expected `[0,1,...,9]`).

**Fix**: indent the Y determination, RidgeCV fitting, and result saving (~240 lines) by 4 more spaces (12→16) so they fall inside the `for _try_lag` loop.

---

**3. Problem 2 — RidgeCV alpha lottery lowers direction consistency**

After fixing the indentation bug the lag distribution returned to normal (72% at lag=0), but direction consistency dropped from 71.8% to 59.3%.

**Root-cause chain**:

1. **Alpha oscillation**: with a single-feature X, the RidgeCV alpha search (0.01~1000) oscillates sharply between lags. For the same (TF, peak) pair, lag=0 may randomly hit alpha=1000 (coefficient→0→not significant→skipped) while lag=3 hits alpha=0.01 (large coefficient→significant→selected)

2. **Abnormal pile-up at lag=3**: 34,518 ATF4 edges piled up at lag=3, of which **93.6% had a positive sign** (lag=0 was only 40.6%). The sign flip happens because `Cov(TF[t-3], peak[t])` at lag=3 differs systematically from `Cov(TF[t], peak[t])` at lag=0

3. **Systematic upward bias**: the many positive weights at lag=3 → during perturbation propagation ATF4 KO → peaks↑ → genes↑ → the model predicts 74% UP (true 60%) → direction consistency falls

**Fix — replace RidgeCV with the Pearson correlation coefficient**:

For a single feature, Ridge's `variance_rescale` restores the coefficient exactly to `w = sign(r) × σ_Y / σ_X`, independently of alpha — alpha only affects the p-value threshold, not the final weight. So the only role RidgeCV plays for a single feature is to supply a p-value, and alpha oscillation makes that p-value incomparable across lags.

The fix uses the Pearson correlation coefficient directly: `r = Pearson(X, Y)`, `t = r × √(n-2) / √(1-r²)`. The Pearson t test is the gold standard for single-feature significance and is immune to the alpha lottery.

```python
# Old: the RidgeCV alpha lottery decides which lag passes significance
model = RidgeCV(alphas=np.logspace(-2, 3, 20))  # alpha ∈ [0.01, 1000]
model.fit(X, _Y)  # alpha oscillates sharply with lag
# → p-value depends on alpha → lag=3 accidentally accumulates 34,518 edges

# New: Pearson correlation coefficient — fair comparison across all lags
r, p = pearsonr(X[:, 0], _Y)  # t = r × √(n-2) / √(1-r²)
w = r × σ_Y / σ_X              # equivalent to variance-rescaled Ridge
# → all lags use the same significance standard → lag=0 wins with 74.8%
```

---

**4. Summary of changes**

| Change | Location | Description |
|------|------|------|
| **Per-lag indentation fix** | `kinetics.py:_fit_tf_peak_regression` | Y determination + RidgeCV fitting + result saving (~240 lines) indented 4 more spaces (12→16) so they are inside the for loop |
| **Pearson shortcut** | `kinetics.py:_fit_tf_peak_regression` | In per-lag single-feature mode, take the Pearson path before reaching RidgeCV (~25 new lines) to compute r/p/weight/contribution directly |
| **RidgeCV path retained** | `kinetics.py:_fit_tf_peak_regression` | Cases that are not single-feature still use the original RidgeCV path, unaffected |
| **Granger keeps a single lag** | `config.yaml` | `granger.max_lag: 1` — the Granger stage keeps using the single-lag F test to control false positives |
| **RNA→ATAC independent regression per lag** | `config.yaml` | `time_lag: true` — independent regression per lag + early-stop (the only design) |

---

**5. Effect validation** (ATF4, 200-peak validation sample)

| Metric | Before fix | After fix |
|------|--------|--------|
| lag=0 share | — | **74.8%** |
| lag=3 share | 11.5% (abnormal) | 1.3% |
| lag=3 positive-sign rate | 93.6% (abnormal) | 50% (balanced) |
| Perturbation lag range | `[9]` (indentation bug) | `[0,1,...,9]` (multiple lags) |
| Direction consistency (Top 25%) | 59.3% | pending full run |

---

**6. Summary of design principles**

1. **No regularization is needed for a single feature**: with one feature, Ridge's L2 penalty only shrinks the magnitude and `variance_rescale` fully restores it, which amounts to no-op work. Pearson gives the sign and significance directly with no redundant computation
2. **Fair comparison**: all lags use the same significance test (Pearson t-test), ensuring that early-stop picks the genuinely optimal lag rather than the lag most favorable to alpha
3. **Clear division of labor between Granger and TF→peak**: Granger decides "is there a causal relationship" (single lag, strict false-positive control); TF→peak decides "what is the optimal delay" (multi-lag search, retaining direction information)
4. **Backward compatible**: multi-feature mode and Ridge mode still use RidgeCV and are unaffected

### v1.32 — Code cleanup: remove OLS sign correction + RNA root baseline + window-validation boundary fix (2026-06)

**Background**: after v1.30 replaced multi-TF joint Ridge with per-TF independent Ridge, TF-TF collinearity was eliminated at the root. The v1.25 OLS sign correction and the v1.31 RNA root-baseline test had both become redundant patches, and the former could additionally introduce an asymmetric bias (a tendency to flip signs positive) that interfered with window validation.

**Change A — remove OLS sign correction (v1.25)**:

Per-TF independent Ridge removes the root cause of sign flipping (collinearity) entirely, so the three-tier decision strategy of OLS + condition number + F-test + peak open-state fallback is no longer needed. Removed ~165 lines of code and 2 parameters (`sign_correction`, `min_tf_expression_ratio`).

**Change B — remove the RNA root-baseline test (v1.31)**:

The RNA root-baseline test never actually modified model output (it was statistics only), and its logic differs fundamentally from the ATAC root baseline — ATAC has a physical [0,1] boundary while RNA has no upper bound. Removed the `_compute_root_rna_baseline` function and its end-to-end plumbing, ~200 lines.

**Change C — window-validation boundary fix**:

Window validation originally forced a sign-consistency check on **every** best_lag, misreading the normal sign differences between lags in a distributed-lag model (for example lag=0 instantaneous positive correlation vs lag=2 delayed negative feedback) as "boundary effects that need correcting". As a result 81% of edges were flipped (17266/21203).

Fix: trigger window validation only when best_lag is near the max_lag boundary (`best_lag >= max_lag - 1`, i.e. best_lag ∈ {4,5}). Sign differences at non-boundary best_lag are normal biological phenomena and should not be forced into agreement.

```
Before: every best_lag triggered it
  future = group[group["tf_lag"] > best_lag]  ← unconditional

After: triggered only when best_lag ∈ {4,5}
  if best_lag >= ridge_max_lag - 1:
      future = group[...]
  else:
      _n_skipped_boundary += 1
```

| Change | Location | Description |
|------|------|------|
| **Remove OLS sign correction** | `kinetics.py:_fit_tf_peak_regression` | ~165 lines of three-tier decision logic (OLS+F-test+peak-state fallback), including parameter cleanup |
| **Remove RNA root baseline** | `kinetics.py`, `perturbation.py` | `_compute_root_rna_baseline` + sign test + end-to-end plumbing, ~200 lines |
| **Window-validation boundary guard** | `perturbation.py:_simulate_knockout_bin_forward` | `best_lag >= max_lag - 1` condition; non-boundary cases are skipped and a skip counter is added to the log |
| **Config cleanup** | `config.yaml` | Removed `sign_correction`, `min_tf_expression_ratio`, and the RNA-root-baseline items |

**Effect** (RUNX1 KO): apparent direction consistency drops (spurious consistency fades) while Top N hit rate and other metrics rise slightly — removing the overfitting patches improves generalization.

### v1.31 — RNA root-baseline sign test: independent direction validation (2026-06) ⚠️ Removed

**Problem**: model direction predictions carry a systematic UP bias (71% predicted up), with only 35.2% accuracy on DOWN genes (truly downregulated). Moreover the entire sign chain `sign(Δgene) = -sign(w) × sign(pearson_r)` is determined purely by two linear correlation coefficients, with no independent biological validation.

**Approach — RNA root-cell baseline test** (`kinetics.py` + `perturbation.py`):

Following the idea of the ATAC root-cell baseline, a "ground state" reference is also established for RNA expression: take the mean of the earliest pseudotime bin as each gene's root-baseline expression, compare it with the late-stage expression to obtain a deviation direction, and use that to independently validate the model's predicted perturbation direction.

```
baseline = mean(RNA[earliest 3 bins])
deviation = mean(RNA[latest 3 bins]) - baseline
expected direction (activator KO) = -sign(deviation)  // reversal of the natural trend
```

- **TF role inference**: infer from the TF→peak weight distribution whether the KO target is an activator (positive weights > 60%) or a repressor (positive weights < 40%), and use that to decide the expected direction
- **Statistics only**: the sign-test result is emitted only to the log and as a confidence reference; it never forces a flip of the model prediction

| Change | Location | Description |
|------|------|------|
| **Baseline computation** | `kinetics.py:_compute_root_rna_baseline` | New function taking the mean of the first N bins as each gene's root baseline |
| **Sign test** | `perturbation.py:_simulate_knockout_bin_forward` | Runs the root-baseline sign-consistency test after delta_rna is computed |
| **Return-value plumbing** | `perturbation.py` | `root_rna`, `rna_deviation`, and `tf_role` are plumbed through the whole chain |
| **Log output** | `perturbation.py` | Agreement count, conflict count, agreement rate, and TF role inference |

**Simulation validation** (KLF1 KO):
- When the root baseline conflicts with the model on DOWN genes: root-baseline accuracy **74.5%** (model only 35.2%)
- When the model is wrong, the root baseline identifies **50.6%** of the wrong predictions
- When the root baseline agrees with the model, accuracy is 62.2%

### v1.30 — Per-TF independent Ridge + Pearson r sign validation: direction accuracy improvement (2026-06)

**Problem 1 — TF-TF collinearity makes Ridge coefficient signs random**: v1.22's sign_correction audited signs for negative weights caused by TF-TF collinearity in the multi-TF joint Ridge, but it did not address the joint Ridge's own coefficient-sharing problem. When several correlated TFs predict the same peak, RidgeCV distributes the regression coefficients arbitrarily across the collinear TFs, so an individual TF's coefficient sign can flip at random.

**Problem 2 — Granger β₂ (atac_coef) sign is close to random**: in the Granger model `Y(t) = β₀ + β₁·Y(t-1) + β₂·X(t-1)`, the gene-expression autocorrelation term Y(t-1) dominates the fit, so β₂ is estimated from the residual. With only 50 time bins, the sign of β₂ stays near 50% (random) at every statistical significance level. Yet `_verify_gain_sign` used the β₂ sign to correct the NN gain, so the NN layer inherited a random sign.

**Approach A — per-TF independent Ridge** (`kinetics.py:_fit_tf_peak_regression`):

```python
# Old: all TFs stacked into one design matrix, a single RidgeCV
X = [TF1_lag0..L, TF2_lag0..L, ...]
RidgeCV.fit(X, Y)  # coefficients shared arbitrarily across TFs

# New: fit each TF independently
for tf in valid_tfs:
    X = [tf_lag0..L]  # that TF only
    RidgeCV.fit(X, Y)  # sign is not disturbed by other TFs
```

**Approach B — Pearson r sign validation** (`granger.py` + `nn_transfer.py`):

The Granger test also computes `pearson_r = corr(X(t-1), Y(t))`, which measures the correlation direction between peak accessibility and gene expression directly and is not disturbed by competition from the Y(t-1) autoregressive term. `_verify_gain_sign` now uses pearson_r instead of atac_coef as the NN gain sign reference:

```
Old: gain_sign != atac_coef_sign → flip
New: gain_sign != pearson_r_sign → flip
```

**Approach C — NN raw gain experiment**: we tried trusting the NN autograd gradient sign directly (skipping external validation), but NN gain sign accuracy was lower than pearson_r. The pearson_r validation was kept as the best approach.

| Change | Location | Description |
|------|------|------|
| **Per-TF independent Ridge** | `kinetics.py:_fit_tf_peak_regression` | Build a single-TF design matrix for each candidate TF and fit RidgeCV, eliminating contamination of coefficient signs by TF-TF collinearity |
| **Pearson r computation** | `granger.py:granger_test` | New `pearson_r` result column: `corr(X(t-1), Y(t))`, used for downstream sign validation |
| **Pearson r sign validation** | `nn_transfer.py:_verify_gain_sign` | Uses pearson_r instead of atac_coef to decide whether to flip the NN gain |
| **NN raw gain experiment** | `nn_transfer.py` | `_verify_gain_sign` was briefly commented out to test raw NN gain accuracy; results were worse than pearson_r, so it was restored |

**Effect** (KLF1 KO, 5 hematopoietic clusters):

| Metric | Before | v1.30 |
|------|------|-------|
| Direction accuracy | 56.9% (250/439) | **60.8%** (287/472) |
| TF→peak positive-weight share | 24.7% | **41.4%** |
| Causal edge count | 439 genes | 472 genes (+33) |
| atac_coef sign accuracy | ~50% (random) | — |
| pearson_r sign accuracy | — | 55.8% (causal edges) |

**Remaining bottleneck**: for genes whose direction is predicted wrongly (~39%), the edge-level pearson_r sign accuracy is only 40.4% — the correlation direction between peak and gene expression is itself opposite to the true causal direction, possibly due to confounders, feedback loops, or limited pseudotime resolution. Around 60% is the direction-accuracy ceiling for the model under the current data conditions.

### v1.29 — Per-cluster pseudotime inference + adaptive thresholds + family-expansion fix (2026-06)

**Problem 1: global DPT design bias**. Previously pseudotime was inferred on all cells globally (`run_pipeline` Step 3) and then subset by cluster. Global DPT mixes cells from different lineages (erythroid/myeloid/megakaryocytic), so two cells at the same pseudotime may sit at completely unrelated differentiation stages. Granger causal testing relies on a consistent temporal direction, and mixing lineages produces false-positive causal edges.

**Problem 2: insufficient statistical power in small clusters**. Fixed `reference_bins=50` and `n_neighbors=15` produced an overly aggressive degrees-of-freedom penalty in small clusters (~60 cells): `correction = (47/9)² = 27.3x`, dropping the causal rate from 77% to 25%. The BFU-E cluster even produced zero Granger causal edges.

**Problem 3: incomplete TF family motif matching**. The first version only passively borrowed family-member matches already present in the cache (limited by `max_tfs_per_peak=5`), which severely restricted candidate peaks for the target TF — GATA1 fell from 48K peaks to 737, and RUNX1 hit only 21 peaks. Active DNA-sequence scanning had to be restored, but only for target TFs in the KO list, to avoid exploding large families such as C2H2_ZF.

**Approach**:

| Change | Location | Description |
|------|------|------|
| **Per-cluster pseudotime inference** | `run.py:_run_modeling_for_subset` | Removed global Step 3; pseudotime inference moved inside `_run_modeling_for_subset` so each cluster runs DPT independently |
| **Adaptive n_neighbors** | `run.py:_run_modeling_for_subset` | `n_neighbors = min(default, max(5, n_cells // 3))`, reducing KNN graph complexity in small clusters |
| **Adaptive reference_bins** | `run.py:_run_modeling_for_subset` | `reference_bins = min(50, max(est_bins, 10))`, matched to the actual bin count to remove the excessive df penalty; second calibration lowers it further when the actual bin count < 0.8× the estimate |
| **Adaptive min_composite_score** | `run.py:_run_modeling_for_subset` | Small clusters relax the composite-score threshold in proportion to `n_cells/200` |
| **Family expansion rewritten (two stages)** | `kinetics.py:_supplement_family_proxies` | v1.29 final version: **Stage A** passive borrowing — peaks of family members already present in the cache join the target TF directly (0 cost); **Stage B** forced full-family PWM scan — every target TF with family information scans the DNA sequence of all causal peaks with all of its family members' PWMs (Gumbel p < 0.1), and any match is added. Only target TFs in the KO list are expanded; non-target families are not unfolded. |
| **Removal of the family-expansion threshold** | `config.yaml` | Removed the `family_proxy_min_matches` parameter; all KO-TFs forcibly run the full-family PWM scan instead of being judged by a match-count threshold |
| **Cluster label tracking** | `run.py:_run_modeling_for_subset` | Results gain a `cluster` field so the source can be traced during cross-cluster merging |

**DPT ordering change**:
```
Old: global DPT → split data by cluster → Granger → perturb
New: split data by cluster → per-cluster DPT → Granger → perturb (keeps every lineage independent)
```

**Effect** (v1.29, 5 hematopoietic clusters):
| Metric | Old run | v1.29 |
|------|--------|-------|
| Causal rate | ~25% | **72–91%** |
| Clusters with empty tf_weights | 2/5 (BFU-E, Basophilic) | **0/5** |
| RUNX1 TF→peak edges | 0–15 per cluster | **greatly increased after the full-family PWM scan** (RUNX family PWMs cover the whole genome) |
| GATA1 TF→peak edges | ~50K peaks (old family proxy) | **restored by the full-family PWM scan** (GATA family PWMs cover ~48K peaks) |

### v1.28 — Code cleanup: remove the nn_gp / nn_sc paths (2026-05)

**Background**:
- `nn_gp` (NN mean + GP residual): the NN+GP hybrid was never fully validated and keeping it added maintenance burden
- `nn_sc` (learning gain from single-cell data): the design has a fundamental flaw — single-cell data measure the at-the-same-time ATAC→RNA correlation, whereas the model needs the lagged effect (ATAC[t-1]→RNA[t]); controlling for pseudotime cancels exactly the temporal-lag signal the model depends on
- Both had already been reverted to the V5 configuration (`method: "nn"`)

**Removal scope**:
| File | Removed content |
|------|---------|
| `nn_transfer.py` | `PairEmbeddingNN_SC` class, `_fit_atac_to_rna_nn_gp/sc`, `_nn_gp/sc_predict_*`, `_train_nn_sc`, `_collect_training_data_sc`, `_compute_pair_metrics_sc`, `_binned_verify_sign` |
| `kinetics.py` | The `nn_gp` / `nn_sc` method dispatch blocks |
| `perturbation.py` | `_atac_to_rna_step_nn_gp`, the `use_nn_gp` logic, `nn_sc` model loading |
| `io.py` | nn_sc default parameters |
| `config.yaml` | Method options changed from `"gp" \| "nn" \| "nn_gp"` to `"gp" \| "nn"`, and nn_sc-specific parameters were removed |

### v1.27.5 — TF expression-trend filter: excluding inactive TFs (2026-05)

> **⚠ Disabled in v1.31**: both the median filter (rule a) and the Spearman trend filter (rule b) were too aggressive and were removed during the v1.31 cleanup. A TF with median=0 may still be active in a specific subpopulation, and a downward trend may be a downstream effect driven by the target gene, so killing it would cut off the propagation path. The current `_filter_tfs_by_expression` only verifies gene names (TFs absent from the RNA matrix are skipped).

**Approach** (the original v1.27.5 design, now deprecated): insert a two-rule filter before Ridge regression (stage 1.6):

| Rule | Condition | Meaning |
|------|---------|------|
| (a) Median filter | Single-cell expression median = 0 | Non-functional TF, not expressed in this cell type |
| (b) Trend filter | Spearman ρ < 0, p < 0.05 | TF expression decreases with pseudotime, i.e. it is being "switched off" |

| Change | Location | Description |
|------|------|------|
| **TF expression filter** | `kinetics.py:_filter_tfs_by_expression` | Two-rule filter function; simplified in v1.31 to gene-name verification only |
| **Stage 1.6 integration** | `kinetics.py:fit_rna_to_atac` | Calls the filter before Ridge regression |
| **New configuration** | `config.yaml` | `rna_to_atac.tf_expression_filter: true`, `rna_to_atac.tf_trend_pvalue: 0.05` |
| **Cluster-key compatibility** | `preprocess.py`, `run.py` | Cluster-label sorting supports both integer (Leiden) and string (author annotation) types |

### v1.27 — Configurable NN capacity: atac_hidden_dims (2026-05)

**Problem**: the monotonic constraint (v1.26) restricts atac_net weights to >= 0, halving effective capacity. The median R² dropped from 0.225 to 0.049 and the R²>=0.3 share from 41.7% to 23.4%.

**Approach**: turn the atac_net hidden-layer dimensions from the hardcoded [32,64,128] into the configurable `nn_atac_hidden_dims`. The default is [64,128,256,256,128] (~146K parameters, 10.5x the old default). Old checkpoints remain compatible automatically (falling back to [32,64,128] when the key is absent).

### v1.26 — NN monotonic constraint: fixing the gain-sign contradiction (2026-05)

**Problem**: after the shared neural network PairEmbeddingNN replaced GP, the gain sign became close to random (53.8% positive / 46.2% negative, vs 77.5% positive for GP). The root cause is that atac_net's ReLU weights can be positive or negative, so each basis function f_i(x) can oscillate and the sign of the dot product between the embedding weights and the basis-function derivatives is random.

31.4% of activating edges (positive Spearman correlation) were given a negative gain by the NN, because `gain = mean(d_pred/d_atac)` averages autograd local derivatives over 50 sample points and is sensitive to the kinks of a piecewise-linear ReLU function. Even when the curve trends upward overall (Spearman > 0), derivatives from locally descending segments can dominate the mean.

**Approach**: apply a nonnegative weight constraint `weight.data.clamp_(min=0.0)` to every Linear layer of atac_net, executed after each optimizer step.

```
Before: atac_net weights can be positive or negative → f_i(x) can oscillate → gain sign 50/50
After:  atac_net weights >= 0 → f_i(x) monotonically increasing → gain sign = sign(Σw_i)
```

- `atac_net`: all weights nonnegative → each basis function f_i(x) increases monotonically with ATAC. Nonlinearity is retained (saturation and differing growth rates remain possible)
- `pair_head`: weights are free → positive = activating pair, negative = repressing pair
- Gain = Σ w_i × f'_i(x), where f'_i(x) >= 0 → **the gain sign is uniquely determined by Σw_i** and no longer contradicts Spearman

**Design idea**: the basis functions provide a monotonically increasing basis for "how much RNA signal can be extracted from ATAC", while the pair weights determine "whether this pair is activating or repressing, and how strongly". Biologically, peak opening → more opportunities for TF binding → increased transcription, so the basis functions should increase monotonically; whether the effect is activating depends on the specific TF-peak-gene context and is encoded by the weights.

| Change | Location | Description |
|------|------|------|
| **clamp_atac_weights** | `nn_transfer.py:PairEmbeddingNN` | New method: walk atac_net and clamp every Linear.weight to >= 0 |
| **Training-loop call** | `nn_transfer.py:_train_nn` | Call `model.clamp_atac_weights()` after `optimizer.step()` |
| **Documentation update** | `nn_transfer.py` | Class documentation describes the monotonic-constraint design rationale |

**Expected effect**: gain signs become 100% consistent with response_type (activating positive / repressing negative), and the up/down ratio of perturbations should improve substantially.

### v1.25 — Sign Correction v2: OLS + condition-number detection + peak open-state fallback (2026-05) ⚠️ Removed in v1.32

**Problem**: v1.22's sign_correction used the same-time `pearson_r(TF(t), peak(t))` to check the Ridge coefficient direction at all lags. In multi-lag mode, the lag=5 coefficient models TF(t-5)→peak(t), while the same-time correlation measures TF(t)→peak(t) — the two do not match. In GATA1 KO simulations, target genes such as ALAS2 were instead **upregulated** (delta_rna=+0.055).

**Root-cause chain**: Ridge learned a negative coefficient because of TF-TF collinearity → sign_correction did not correct it (the same-time Pearson r is insensitive to lagged relationships) → the multi-lag sum further amplified the negative weight → the perturbation propagation direction was inverted.

**Exploration**:

1. **Lag-offset Pearson r** (tried, then abandoned): align `corr(TF[t-lag], peak[t])` using the `tf_lag` offset. Why it failed — GATA1 has extremely small between-bin expression variance (std=0.11), so the Pearson r flips negative at random in the noise and cannot reliably determine direction.

2. **Distributed-lag OLS** (tried, then abandoned): fit `peak(t) = Σβₗ·TF(t-l)` for each (TF,peak) using all lags of that single TF. Why it failed — the TF's own lags are collinear (condition number 1527), 30 data points are used to fit 7 parameters, β oscillates within ±11, and the sign of sum(β) is meaningless.

**Final approach — three-tier decision strategy**:

For every (TF, peak) pair with a negative weight:

```
1. Single-TF distributed-lag OLS → check the condition number + F-test
   ├─ cond < 30 and F-test p < 0.05 (OLS is reliable)
   │   ├─ sum(β) > 0 → activator → flip the negative weight positive
   │   └─ sum(β) < 0 → repressor → keep the negative weight
   │
   └─ cond ≥ 30 or the F-test is not significant (TF persistently high/low
       expressed, not enough fluctuation)
       └─ fall back to the peak open-state decision
           ├─ peak_mean > 0 (Scheme B space, above the root-cell baseline)
           │   → an activator maintaining its openness → flip positive
           └─ peak_mean ≤ 0 → direction unreliable → filter all edges of this (TF,peak)
```

**Design idea**: for a master regulator such as GATA1 that is persistently highly expressed in the target cells, expression hardly changes, and no statistical method can learn "the direction of change" from "no change". The downstream peak's open state is then the most reliable signal — if the TF is persistently highly expressed and the peak is persistently open, the TF maintaining that openness is the only reasonable explanation.

**Companion change — remove the multi-lag sum** (`perturbation.py`): drop `net_weight = sum(group["weight"])` and propagate using the `best_lag` weight directly. v1.22's summation was a fallback patch for when sign_correction failed — positive and negative weights cancel each other out and reduce the impact. Once each lag coefficient's direction is trustworthy, the summation becomes a noise source instead.

| Change | Location | Description |
|------|------|------|
| **Sign Correction v2** | `kinetics.py:_fit_tf_peak_regression` | Rewritten as single-TF OLS + condition-number detection (cond<30) + peak open-state fallback (mean>0), with unreliable edges filtered |
| **Remove multi-lag sum** | `perturbation.py:_simulate_knockout_bin_forward` | Drops the groupby→sum aggregation so the best_lag weight is used directly for perturbation propagation |

**Effect** (BMMC GATA1 KO): the upregulated share falls from 52% to 41%, the spurious upregulation of ALAS2 in cluster 1 is filtered, and cluster 2 is correctly downregulated (-0.008).

**Comparison with CellOracle**: CellOracle's edge weights come from co-expression correlation (so their direction is inherently consistent with biology) and it has only a single TF→gene layer. In CausalBridge's two-layer Ridge+NN architecture, Ridge regression coefficients do not themselves encode regulatory direction (they are affected by TF collinearity), and sign_correction v2 is the key bridge that closes the gap between "regression coefficient" and "biological direction". For persistently highly expressed master regulators CellOracle also handles this correctly — because its correlation depends on co-expression variance in expressing cells rather than on the TF's own variance, it is inherently robust to "constant high expression".

### v1.24 — Cross-cluster merge fix: absolute-value summation avoids direction cancellation (2026-05)

**Problem**: `delta_rna` in `perturbation_merged.csv` merged cluster results with a signed weighted mean, so when the same gene moved in opposite directions in different clusters the effects cancelled out and the true perturbation effect was diluted in cross-cluster analysis.

**Approach**: change `delta_rna` to the absolute-value weighted sum `sum(|delta_rna_i| × cell_count_i) / total_cells`, and add a `delta_rna_signed` column that retains the signed weighted mean for direction judgment.

| Change | Location | Description |
|------|------|------|
| **Cross-cluster merge fix** | `run.py` | `delta_rna` becomes the sum of `abs(delta_rna) * weight / total_cells`, and `delta_rna_signed` is added |
| **Merged-result sorting** | `io.py:save_results` | `perturbation_merged.csv` is sorted by absolute magnitude `delta_rna` in descending order |

### v1.23 — Candidate-pool reduction + RNA log1p normalization: lowering causal-edge density and unit bias (2026-05)

**Problem 1 — too many causal edges**: `distance_thresh=1,000,000` let an average of 146 candidate peaks per gene enter the Granger test, and Granger filtered only ~16% (122/146 passed). The composite-score ceiling effect (effect_score=1.0 for all edges) produced 694,303 causal edges and an average of 138 peaks per gene — biologically implausible.

**Problem 2 — GP gain unit issue**: GP was trained in raw count space, so `gain = d(raw_RNA)/d(scaled_ATAC)` had a median of only 0.0004. After binning, RNA std is ~0.1-1 and the per-peak partial contribution ~1%, so in the perturbation simulation the gain is compressed a second time by the Ridge coefficients.

**Approach A — reduce the candidate pool**: `distance_thresh: 1,000,000 → 200,000`, sharply cutting the Granger candidate peak pool.

**Approach B — RNA log1p normalization**: apply a `log1p` transformation to the binned RNA before GP training, so the gain becomes `d(log1p(RNA))/d(scaled_ATAC)` ≈ log-fold-change per unit ATAC. The perturbation module gains a new `_delta_r_log_to_raw` function that converts the log-space delta back to raw count space via `expm1(log1p(r_orig) + delta_r_log) - r_orig` before accumulation.

| Change | Location | Description |
|------|------|------|
| **distance_thresh reduction** | `config.yaml` | `granger.distance_thresh: 1000000 → 200000` |
| **RNA log1p normalization** | `kinetics.py:fit_atac_to_rna` | `rna_binned = np.log1p(rna_binned)` before GP training, so the gain becomes a log-fold-change |
| **rna_log_normalized flag** | `kinetics.py:fit_atac_to_rna` | Writes `rna_log_normalized: True` into the gp_models dictionary, which the perturbation module uses to decide whether a log→raw conversion is needed |
| **log→raw conversion** | `perturbation.py:_delta_r_log_to_raw` | New function, `expm1(log1p(r_orig) + delta_log) - r_orig` |
| **Automatic detection in perturbation** | `perturbation.py:_simulate_knockout_bin_forward` | Scans gp_models for the `rna_log_normalized` flag and enables the log→raw conversion automatically |
| **New configuration** | `config.yaml` | `atac_to_rna.log_normalize_rna: true` |

**Design principle**: unit consistency — GP is trained in log space and its predictions are in log space, but the `delta_cum` accumulator and the `rna_pert` matrix are in raw count space. `_delta_r_log_to_raw` converts before accumulation to keep the units consistent. A `distance_thresh=200kb` is enough to cover the typical enhancer-gene interaction range (median ~50kb) while reducing useless candidates.

### v1.22 — Sign correction + multi-lag aggregation: removing the perturbation upward bias caused by TF collinearity (2026-05)

**Problem**: in the distributed-lag Ridge, collinearity between different TFs (for example GATA1 and GATA2 are highly correlated in expression) flipped Ridge coefficient signs — the weights of positively regulating TFs were turned negative. 55% of GATA1's (a known transcriptional activator) TF→peak weights were negative, producing a 2:1 upward bias in KO simulations.

**Root cause**: under multicollinearity Ridge's L2 regularization assigns a coefficient to an arbitrary one of the collinear variables, so a TF that receives a negative coefficient gets a negative weight even when it correlates positively with the peak. Multi-lag aggregation (net weight = Σw_lag) fixed the across-lag collinearity within a single TF, but collinearity between different TFs remained.

**Approach A — Pearson r sign correction**: for every TF→peak edge with a negative weight, compute the Pearson correlation between TF expression and peak ATAC. If r > 0 (TF correlates positively with the peak), the negative weight is a collinearity artifact and is replaced by the univariate OLS coefficient:

```
w_corrected = r × σ(peak) / σ(TF)
```

Ridge's original negative weight is kept if and only if r ≤ 0 (the TF truly correlates negatively with the peak and may be a repressor).

**Approach B — multi-lag aggregation**: in the bin_forward perturbation simulation, combine the distributed-lag coefficients per (TF, peak) into a net effect `w_net = Σβ_lag`, and pick the lag with the largest |w_lag| as the propagation delay. This avoids spurious upregulation signals from negative lag coefficients when different lags of the same TF propagate independently.

| Change | Location | Description |
|------|------|------|
| **Pearson r sign correction** | `kinetics.py:_fit_tf_peak_regression` | Computes corr(TF, peak) for negative-weight edges; replaces with `w = r × σ(y)/σ(x)` when r>0 and keeps the original negative weight when r≤0. Low-expression TFs (non-zero bins < min_tf_expression_ratio) skip the audit |
| **Multi-lag aggregation** | `perturbation.py:_simulate_knockout_bin_forward` | Group by (TF, peak), w_net = Σw_lag, best_lag = argmax\|w_lag\|. The zero-string lookup iteration path is updated accordingly |
| **KeyError fix** | `perturbation.py:_propagate_bin_forward` | When all genes of a joint KO are skipped, return `{"perturbation_results": pd.DataFrame(), "pathway_edges": pd.DataFrame()}` instead of a bare DataFrame |
| **New configuration** | `config.yaml` | `rna_to_atac.sign_correction: true`, `rna_to_atac.min_tf_expression_ratio: 0.15` |

**Effect** (BMMC GATA1 KO): the negative-weight share falls from 55% to 47%, the mean positive weight is +0.137, and the perturbation up/down ratio recovers from 2:1 to 53:47 (nearly balanced).

### v1.21 — Scheme B [-1,1] piecewise scaling: removing the systematic positive bias (2026-05)

**Problem**: v1.20's atac_pert clipping to [0,1] was asymmetric — upward there is 1-baseline of room, downward only baseline of room. Most peaks have a very low baseline (84% < 0.05), so downward perturbation had almost no room and the perturbation engine was forced to output mostly upregulation (69.8% delta_rna>0).

**Approach**: use the mean ATAC of the root cells (earliest 5% of pseudotime) as the baseline and map ATAC values piecewise into a symmetric [-1,1] space:

```
atac_scaled = (atac - baseline) / baseline          (downward, atac ≤ baseline)
atac_scaled = (atac - baseline) / (1 - baseline)    (upward, atac > baseline)
```

0 = the normal baseline state, and upward/downward perturbations are symmetric and equidistant. floor=0.05 prevents noise amplification at low-baseline peaks.

**Data-flow change**: `root_atac` is computed in `fit_atac_to_rna` → passed through `_run_modeling_for_subset` to `fit_rna_to_atac` and `propagate_perturbation` → the perturbation engine applies `_unscale_atac` in reverse to output results in the original space.

| Change | Location | Description |
|------|------|------|
| **New `_compute_root_atac`** | `kinetics.py` | Takes the mean binarized ATAC of the earliest 5% of pseudotime cells, floor=0.05 |
| **New `_scale_atac`** | `kinetics.py` | Piecewise mapping of ATAC→[-1,1] with asymmetric denominators for the up/down directions |
| **New `_unscale_atac`** | `kinetics.py` | Inverse mapping [-1,1]→original ATAC [0,1] |
| **GP training uses scaled ATAC** | `kinetics.py:fit_atac_to_rna` | X=scaled_ATAC, Y=RNA (unchanged), gain=d(RNA)/d(scaled_ATAC) |
| **Ridge training uses the scaled target** | `kinetics.py:fit_rna_to_atac` | X=TF (unchanged), Y=scaled_ATAC, weight=d(scaled_ATAC)/d(TF) |
| **Symmetric clip in the perturbation engine** | `perturbation.py:_simulate_knockout_bin_forward` | clip[-1,1] replaces clip[0,1]; atac_pert operates in the scaled space |
| **Inverse scaling for output** | `perturbation.py` | `_unscale_atac` restores the original [0,1] space before delta_atac is computed |
| **root_atac passed across steps** | `run.py` | Stored/loaded with checkpoints and passed through Step5→Step6→Step8 |

### v1.20 — ATAC perturbation-range fix + GP extrapolation safety net (2026-05)

**Problem**: in the RNA→ATAC propagation step, `atac_pert[t, peak] += weight × delta_tf` accumulated without bound. delta_tf can reach ~1.2 (complete KO), Ridge weights can reach ±2~5, and a single contribution can reach ±4~6 — far outside ATAC's normal [0,1] range. This caused:
- 240 thousand atac_pert values pushed to negative numbers (physically impossible)
- 170 thousand atac_pert values above 1.0 (>100% open)
- 550 thousand GP extrapolations to positions 4x beyond the training range
- Uncontrollable RBF extrapolation behavior, loss of weight sign information, and a uniform positive bias on activating/repressing edges

**Approach**: two defensive fixes:
1. **atac_pert clipping**: once all TF contributions to the same peak have been accumulated, clip to [0, 1]
2. **GP OOB safety net**: when a_new is outside the training range, fall back to the linear `gain × delta_a` (guaranteeing the correct sign), while interpolation still uses the GP path integral

| Change | Location | Description |
|------|------|------|
| **GP gradient unit restoration** | `perturbation.py:_gp_posterior_gradient` | With `sklearn normalize_y=True` the alpha lives in standardized space, so the gradient must be multiplied by `_y_train_std` to restore units |
| **atac_pert clipping** | `perturbation.py:_simulate_knockout_bin_forward` | `np.clip(atac_pert[t], 0, 1)` after the RNA→ATAC step |
| **GP OOB safety net** | `perturbation.py:_simulate_knockout_bin_forward` | Fall back to linear when `a_new < X_min or a_new > X_max` |

**Effect** (sign accuracy on ATF4-only peaks):

| Category | Before fix | After fix |
|------|--------|--------|
| weight>0, gain>0 (expect NEG) | 20.8% | **77.8%** |
| weight>0, gain<0 (expect POS) | 34.2% | **85.5%** |
| weight<0, gain>0 (expect POS) | 76.8% | **99.6%** |
| weight<0, gain<0 (expect NEG) | 68.8% | **99.3%** |

The global delta_rna mean fell from 0.60 to 0.118 and the positive-bias share from 81.8% to 69.8% (the residual positive bias reflects that ATF4 genuinely acts mainly as a repressor in this system).

### v1.19 — Path-integral delta_r computation: removing GP clip/fallback bias (2026-05)

**Problem**: when v1.18's GP point-prediction difference method handled an ATAC perturbation beyond the training range, it either clipped a_new to the boundary (losing the effect) or fell back to the linear gain×delta_a (75x too small an effect, median 0.0016 vs GP's 0.14). The root cause is that the "point-prediction difference" `f(a_new) - f(a_orig)` is fragile in boundary cases.

**Approach**: add a path-integral method — instead of computing the difference directly, integrate the GP derivative along the ATAC change path (the RBF kernel derivative has a closed form):

```
delta_r = ∫[a_orig → a_new] f'(a) da ≈ Σ f'(a_mid_i) · Δa
```

It is naturally symmetric, needs no clipping, needs no fallback, and handles nonlinearity automatically.

| Change | Location | Description |
|------|------|------|
| **New GP gradient function** | `perturbation.py:_gp_posterior_gradient` | Analytically computes the RBF-kernel GP posterior-mean derivative ∂f/∂a |
| **New path-integral function** | `perturbation.py:_compute_delta_r_path_integral` | Midpoint Riemann integration, 10 steps by default |
| **New point-prediction function** | `perturbation.py:_compute_delta_r_point_prediction` | Retains the original GP difference+clipping logic for backward compatibility |
| **Method dispatch** | `perturbation.py` bin_forward + iterative | Selects the method through the `delta_r_method` configuration |
| **Configuration** | `config.yaml` | Adds `delta_r_method: "path_integral"` (default) and `path_integral_steps: 10` |

Performance: the hand-written gradient computation is 6x faster than sklearn predict (9μs vs 54μs), and a 10-step path integral costs ≈2× predict, i.e. almost no extra overhead.

### v1.18 — GP asymmetric-clipping fix + duplicate pathway records fix (2026-05)

**Problem 1**: during GP point prediction, `a_new` outside the training range was `np.clip`ped to the boundary. ATAC values are concentrated in the low range, so downward perturbations easily punched through x_min and were clipped to zero → negative delta_r was systematically weakened → perturbation results were positively biased (71.5% delta_rna>0).

**Problem 2**: `pathway_records.append` sat outside the `fired_edges` check, so every edge wrote both a gp and a linear record (249,238 rows vs the actual 93,745 unique edges).

| Change | Location | Description |
|------|------|------|
| **GP extrapolation falls back to linear** | `perturbation.py:_simulate_knockout_bin_forward` | Falls back to `gain * delta_a` when a_new is outside [x_min, x_max], avoiding asymmetric clipping |
| **Duplicate pathway fix** | Same as above | `pathway_records.append` moved inside the `if edge_key not in fired_edges` block |

### v1.17 — Fixing the upward bias of perturbations: separating TF and non-TF effects (2026-05)

**Problem**: perturbation results were systematically biased upward. There were two root causes:

1. **Asymmetric `np.maximum(0, ...)` truncation**: when inheriting across bins, `rna_pert = max(0, orig + delta_cum)` truncated negative values (downregulation) but let positive values (upregulation) through, so downregulated genes were clamped at 0 while upregulation had no ceiling → positive bias.

2. **Cross-bin inheritance is meaningless for non-TF genes**: the same gene has different baseline expression in different bins (for example bin1=100, bin2=2), so applying the same `delta_cum` (for example +10) produces completely different biological effects in different bins (bin1 10% vs bin2 500%). For non-TF genes, cross-bin inheritance violates biological reality.

**Approach**: separate TF cascade propagation from non-TF output computation.

| Change | Location | Description |
|------|------|------|
| **TF/non-TF masks** | `perturbation.py:_simulate_knockout_bin_forward` | Adds three masks: `is_tf` (genes with TF→peak edges), `is_ko_target` (KO target genes), and `rna_pert_modified` (genes whose output must be read from rna_pert) |
| **Only TFs inherit delta_cum** | Same as above, top of the bin loop | `active = (abs(delta_cum) > 1e-10) & is_tf`; non-TF genes do not inherit cross-bin state |
| **Remove the max(0,) truncation** | Same as above | `rna_pert[t] = orig + delta_cum` replaces `np.maximum(0, orig + delta_cum)`, so downregulation is no longer clamped at 0 |
| **Only TFs update rna_pert** | Same as above, ATAC→RNA step | `if is_tf[gene_idx]: rna_pert[t, gene_idx] = orig + delta_cum`; a non-TF gene's delta_r is recorded only in delta_cum and not written into rna_pert |
| **Equivalent computation for non-TF output** | Same as above, aggregation step | For non-TF/KO genes `mean_pert = mean_orig + delta_cum`, using the accumulated effect directly rather than the rna_pert mean. TF/KO genes are still read from rna_pert |

**Design principle**: TFs need cross-bin inheritance because their expression change cascades into downstream peak→gene pathways and must persist. Non-TF genes are merely the endpoint of a pathway, and their effect should be expressed directly through `delta_cum` (the accumulated contribution of all upstream edges) rather than through cross-bin state passing.

**Parameter changes**: `min_cells_per_bin: 15→10`, `smooth_overlap_factor: 3.0→2.0`. Lower bin overlap makes bins more independent (improving the validity of the F-test assumptions) while the lower sigma-amplification threshold reduces information borrowed from neighboring bins (user priority: false-positive control > false negatives).

### v1.16 — Granger reverts to lag=1: suppressing false-positive inflation (2026-05)

The distributed-lag Granger introduced in v1.15 (joint F test `ATAC(t-1..t-L)`) raised the F statistic's numerator degrees of freedom from 1 to `n_lags`, lowering the significance bar and greatly inflating the number of causal edges. Since the user prioritizes false-positive control over false negatives, the code reverts to the standard lag=1 F test (F(df=1, n-3)).

| Change | Location | Description |
|------|------|------|
| **Remove distributed lags** | `granger.py` | Removes the `_branch_aware_multi_lag` function; pre-slicing reverts to `_branch_aware_lag_pairs(lag=1)` |
| **Simplify the full model** | `granger.py:granger_test` | Reverts to `Y(t) ~ Y(t-1) + X(t-1)`, F(df=1, n-3) |
| **Configuration update** | `config.yaml` | `granger.max_lag: 2` → `1` |

**Design principle**: the Granger stage uses a single lag to detect causal regulatory relationships (strict false-positive control), while the distributed-lag Ridge for TF→peak is kept (RidgeCV has per-lag weights and does not inflate false positives).

### v1.14 — Removing the log2FC hard cap + cross-cluster weighted-mean output (2026-05)

**Historical problem (fixed)**: the old `_simulate_knockout_bin_forward` applied `np.nan_to_num(..., posinf=5.0, neginf=-5.0)` and `.clip(-5.0, 5.0)` to log2FC. The current implementation no longer treats ±5 as an end-of-pipeline output cap; invalid simulated endpoints are retained as `NaN`.

**Problem 2**: after per-cluster modeling there were only per-cluster results (`perturbation_results.csv`) and no cross-cluster summary. Users had to manually merge each cluster's delta_rna weighted by cell count to obtain the whole-tissue mean effect.

**Approach A — remove the hard cap**:

| Change | Location | Description |
|------|------|------|
| **Remove nan_to_num posinf/neginf** | `perturbation.py:_simulate_knockout_bin_forward` | `np.nan_to_num(log2fc, nan=0.0, posinf=5.0, neginf=-5.0)` → `np.nan_to_num(log2fc, nan=0.0)`; inf values are no longer truncated |
| **Remove .clip(-5.0, 5.0)** | Same as above (two places) | The per-round delta for propagation_rounds>1 and the final delta_rna both lose their clip |

**Approach B — cross-cluster weighted mean**:

| Change | Location | Description |
|------|------|------|
| **Cell-count tracking** | `run.py` | Adds a `cluster_cell_counts` dict recording the valid cell count per cluster |
| **Weighted-mean computation** | `run.py` | `delta_merged = Σ(delta_i × n_cells_i) / total_cells`, grouped by (target_gene, affected_gene) |
| **Merged-result saving** | `io.py:save_results` | Adds the `perturbation_merged.csv` output, including an `n_clusters_observed` column |

The merged result contains an `n_clusters_observed` column indicating in how many clusters the gene pair was observed, which can be used to assess cross-cluster consistency of the effect.

### v1.13 — Additive accumulation: perturbation effects persist across bins (2026-05)

**Problem**: with propagation_rounds=1 each bin computed its perturbation delta independently, so the effect of the same regulatory edge firing repeatedly in different bins could not accumulate. The pseudotime-binning mean aggregation (log2(mean(orig+delta)/mean(orig))) further diluted the multi-bin delta into a denominator of 50, making perturbation magnitudes systematically too small.

**Approach**: persist on first firing — each `(peak, gene)` edge contributes its delta once, in the bin where it first fires, and that delta persists into all subsequent bins. Effects of different edges on the same gene can still add up.

| Change | Location | Description |
|------|------|------|
| **delta_cum initialization** | `perturbation.py:_propagate_bin_forward` | Adds the `delta_cum` array and the `fired_edges` set to track accumulated state |
| **Per-bin persistence** | Same as above, top of the bin loop | With `propagation_rounds=1`, each bin applies `delta_cum` to `rna_pert[t]` so the accumulated effect continues |
| **Accumulate on first firing** | Same as above, ATAC→RNA step | `rna_pert[t, gene] += delta_r` → `delta_cum[gene] += delta_r` (only when `(peak,gene)` has not fired before), replacing the independent per-bin accumulation |

**Design principle**: a regulatory relationship is a persistent biological connection and should not be "re-perturbed" at every time point. The same edge firing repeatedly = the same effect counted N times → rejected; different edges firing = independent contributions from different regulatory pathways → allowed to add up.

### v1.12 — Removing the orphan-peak fallback mechanism (2026-05)

**Motivation**: `_orphan_peak_fallback` used genomic distance plus pseudotime Spearman correlation to force-add downstream gene connections for "orphan peaks" that failed the Granger causal test (they have TF→peak links but no peak→gene causal edge). That logic is identical to CellOracle's and has fundamental flaws:

- **False-positive connections**: a peak that is open while gene expression does not change is biologically more likely to be a poised enhancer, enhancer redundancy, an insulator, or a bystander region than a missed causal regulation
- **Repression confusion**: the fallback does not distinguish "no regulation" from "repressive regulation", so `abs(corr)` ranking wrongly admits negatively correlated peak→gene pairs
- **Introduced noise**: replacing a causal test with a correlation essentially lowers the minimum confidence standard for an edge

**Change**: delete the `_orphan_peak_fallback` function and all its call sites entirely, and remove the `enable_fallback` and `orphan_fallback` configuration items. Perturbation propagation uses only peak→gene edges that passed the Granger causal test.

### v1.11 — Per-cluster adaptive binning fix: eliminating the Granger false-positive explosion in small clusters (2026-05)

**Problem**: an earlier refactor centralized the `_adaptive_rebin` call at the `infer_pseudotime` entry point, but that entry point used the global cell count (915) to compute the bin count, so every cluster (regardless of its own size) received 50 bins. Small clusters (such as 117 cells) were forced into 50 bins, Granger testing overfit badly, and a single cluster produced 220 thousand causal edges → 2400+ perturbation results.

**Approach**: revert to calling `_adaptive_rebin` per module, so each downstream module computes its adaptive bin count independently from the current cluster's actual cell count `rna_adata.n_obs`. In Gaussian soft-binning mode small clusters automatically use fewer bins (for example 117 cells → 23 bins) while large clusters keep 50.

| Change | Location | Description |
|------|------|------|
| **Restore the per-module call** | `granger.py:granger_test` | Switches from `infer_pseudotime`'s output bins to calling `_adaptive_rebin(pseudotime_df, rna_adata.n_obs, ...)` |
| **Restore the per-module call** | `kinetics.py:fit_atac_to_rna` | Same as above; per-cluster adaptive binning |
| **Restore the per-module call** | `kinetics.py:fit_rna_to_atac` | Same as above |
| **Restore the per-module call** | `perturbation.py:propagate_perturbation` | Same as above; bin-forward propagation uses the per-cluster bin count |

**Design principle**: each downstream module decides its bin count independently from the current cluster's actual size, and Gaussian soft binning uses the `sigma ≤ k × bin_width` constraint to keep adjacent bins non-redundant. Binning at the `infer_pseudotime` entry point is only for visualization/initial reference, and each module rebins on demand before computing.

### v1.10 — Stronger TF→peak edge filtering: Bagging stability + effect-size trimming (2026-05)

Ridge regression's L2 regularization keeps many weak edges whose coefficients are near zero. The effect size of these weak edges (|weight| × std(X)) can be as low as 1e-5, which becomes noise once it enters the perturbation simulation.

**Approach A — Bagging stability assessment**: bootstrap-resample N rounds (default 0 = disabled), refit Ridge in each round, and count each edge's selection frequency (stability = selections / total rounds). Only edges with stability ≥ bagging_stability_threshold (0.5) are kept. When enabled it affects only the filtering logic, not the Ridge coefficient computation.

**Approach B — effect-size trimming**: compute `abs_contribution = |weight| × std(X)` and remove edges below min_abs_contribution (0.01). This avoids residual weak edges after variance rescaling.

| Change | Location | Description |
|------|------|------|
| **Bagging stability filter** | `kinetics.py:_fit_tf_peak_regression` | Bootstrap resampling → stability statistics → edges below threshold are filtered |
| **Effect-size trimming** | `kinetics.py:_fit_tf_peak_regression` | `abs_contribution = \|weight\| × std(X)`; filter below min_abs_contribution |
| **New configuration** | `config.yaml` | `rna_to_atac.bagging_n_estimators: 0` (disabled by default), `bagging_stability_threshold: 0.5`, `min_abs_contribution: 0.01` |

### v1.9 — Distributed-lag Ridge + variance rescaling: fixing two-layer decay and lag information loss (2026-05)

**Problem 1 — single-lag selection discards information**: v1.8's per-TF lag selection (winner voting) kept only one optimal lag per TF, but the same TF may have different response delays for different peaks. Single-lag selection discarded all regulatory edges at the lags that lost the vote — and those may be real relationships with slightly weaker signal.

**Problem 2 — two-layer multiplicative decay**: the TF→Peak Ridge coefficient (~0.01 after regularization) × the Peak→Gene GP gain (~2e-4) ≈ 2e-6, so the perturbation effect vanishes as it passes through the two layers. Ridge's L2 regularization systematically shrinks coefficients toward zero, and Ridge's `score` method returns the training-set R² (not CV R²), underestimating the shrinkage.

**Problem 3 — lag=0 was excluded**: `_select_optimal_lag` scanned `range(1, ...)`, excluding lag=0, but some TFs (such as pioneer factors) may act through a transient interaction with chromatin (the same pseudotime bin).

**Approach**: three changes together.

**Approach A — distributed-lag Ridge (replacing single-lag selection)**:

The design matrix expands to `[TF₁(t-0), TF₁(t-1), ..., TF₁(t-L), TF₂(t-0), ..., TF_K(t-L)]`, and RidgeCV learns the weights for each lag automatically. L2 regularization naturally pushes the coefficients of irrelevant lags toward zero while retaining lags with a causal effect.

```
X_design = [TF₁_t, TF₁_{t-1}, ..., TF₁_{t-L}, TF₂_t, ..., TF_K_{t-L}]
RidgeCV.fit(X_design, peak_t)
keep every (TF, peak, lag) combination with a nonzero coefficient as an edge
```

**Approach B — variance rescaling**:

After fitting, rescale all Ridge coefficients with `scale = clip(std(Y)/std(pred), 0.3, 5.0)`:

```python
y_std = np.std(y_true)          # true std of bin-averaged ATAC
pred_std = np.std(pred)         # std of the Ridge prediction (compressed by regularization)
scale = float(np.clip(y_std / pred_std, 0.3, 5.0))
coef *= scale
```

Effect: Ridge coefficients that were ~0.01 are amplified to ~0.3-0.5, restoring the two-layer product from 2e-6 to 6e-5~1e-4, an improvement of roughly 30-50x.

**Approach C — lag=0 support**:

- `_branch_aware_lag_pairs`: at lag=0, return the unsliced matrices directly
- `_select_optimal_lag`: scan range becomes `range(0, max_allowed+1)`
- The distributed-lag design matrix naturally includes a lag=0 column

| Change | Location | Description |
|------|------|------|
| **Distributed-lag design matrix** | `kinetics.py:_fit_tf_peak_regression` | A wholly new distributed-lag Ridge path (`multi_lag=True`) whose design matrix expands to `[TF(t-0), ..., TF(t-L)]` and is selected automatically by RidgeCV. The original single-lag path (`multi_lag=False`) is retained for backward compatibility |
| **Variance rescaling** | `kinetics.py:_fit_tf_peak_regression` | With `variance_rescale=True`, scales Ridge coefficients by `clip(std(Y)/std(pred), 0.3, 5.0)` |
| **per-edge lag label** | `kinetics.py:_fit_tf_peak_regression` | The result DataFrame gains a `tf_lag` column, keeping each (TF, peak, lag) as an independent row |
| **lag=0 support** | `kinetics.py:_branch_aware_lag_pairs` | At lag=0, `return mat_before, mat_after` (no slicing) |
| **lag=0 support** | `kinetics.py:_select_optimal_lag` | Scan range `range(0, max_allowed+1)` (previously `range(1, ...)`) |
| **Updated call signature** | `kinetics.py:fit_rna_to_atac` | Adds `max_lag`, `multi_lag`, and `variance_rescale` parameter passing |
| **Perturbation compatibility** | `perturbation.py` | No change needed: the perturbation engine already uses the `_tf_lags` set and per-edge lag tuples, so it is naturally compatible with multi-lag edges |
| **New configuration** | `config.yaml` | `rna_to_atac.max_lag: 5`, `rna_to_atac.multi_lag: true`, `rna_to_atac.variance_rescale: true` |

### v1.8 — Fixing the TF-specific lag-selection metric: winner voting + median rank (2026-05)

**Problem**: the per-TF lag selection introduced in v1.0 used the **mean R²** as its metric (`max(mean(R²_per_peak))`). For binarized ATAC a univariate Ridge has a very low R² and differences between lags are tiny (~0.001), so mean-based selection amounts to a coin flip. After trying 5-fold CV R² we found that at 50 bins 110/111 TFs still had CV R² ≤0, i.e. CV was too harsh.

**Approach**: use **winner voting + median rank** uniformly instead of the mean. For each peak, vote for the lag with the highest R² (winner voting); when the top vote share is < 30%, fall back to the global best lag; break ties with the median rank.

| Change | Location | Description |
|------|------|------|
| **Winner-vote selection** | `kinetics.py:_select_by_vote_and_rank` | New function: each peak votes for the lag with the highest R²; the highest vote wins, ties are broken by median rank, and weak signals (<30%) return None |
| **Remove mean-based selection** | `kinetics.py:_select_optimal_lag` | Drops `max(mean(r2))` and the CV branch in favor of vote+rank; removes the `cv_min_bins` parameter |
| **Remove CV support** | `kinetics.py` | Deletes the `_cv_r2_single` function and the `KFold` import |
| **Config cleanup** | `config.yaml` | Removes `rna_to_atac.lag_cv_min_bins` |

**Effect**: in a single run, 111 TFs now cover 5 different lags, with only 7 weak-signal TFs falling back to the global lag (vs CV mode where 110 TFs with CV≤0 all fell back to lag=1).

### v1.7 — Granger degrees-of-freedom correction: suppressing small-cluster false positives (2026-05)

**Problem**: the adaptive bin-count constraint lowers the bin count of small clusters (for example 117 cells → only 23 bins), which lowers the effective observation count of the Granger test (df=n_bins-3). The same peak-gene pair gives ΔR²=0.07 in cluster 0 (50 bins) but ΔR²=0.49 in cluster 19 (23 bins), a 6.6x difference, and 88% of pairs had higher ΔR² in small clusters — a systematic false positive caused by linear-regression overfitting with few observations.

**Approach**: apply a quadratic degrees-of-freedom correction to both the significance threshold and the effect-size threshold of the Granger test.

```
correction = (df_ref / df_actual)²
  df_ref = reference_bins - 3 = 47
  df_actual = n_bins - 3

p_threshold   ← sig_threshold / correction    # stricter with fewer bins
effect_threshold ← min_effect × correction    # larger effect required with fewer bins
```

| Example cluster | cells | bins | df | correction² | Pass rate before | Pass rate after |
|--------|-------|------|----|-------------|-------------|-------------|
| 0 | 913 | 50 | 47 | 1.00 | 17.4% | 17.4% |
| 16 | 212 | 42 | 39 | 1.45 | 58.1% | 50.3% |
| 19 | 117 | 23 | 20 | 5.52 | 79.8% | 52.3% |

Large clusters (≥50 bins) are unaffected; the small-cluster false-positive rate drops sharply and the std of the pass rate falls from 20.5% to 13.5%.

| Change | Location | Description |
|------|------|------|
| **Quadratic df correction** | `granger.py:granger_test` | `correction = (df_ref/df_actual)²`, scaling both `corrected_sig` and `corrected_effect` in place of the original thresholds in composite_score |
| **Hard-threshold branch synced** | `granger.py:granger_test` | The non-soft-threshold mode also uses the corrected thresholds |
| **New configuration** | `config.yaml` | `granger.reference_bins: 50`, the reference bin count for the df correction |

### v1.6 — Fixing the GP training target: from synthesis rate to direct RNA expression (2026-05)

**Problem**: after v1.5 unified β, perturbation results were still unchanged. The root cause is that β cancels mathematically between the GP training target (`α = dR/dt + β·R`) and the perturbation formula (`ΔR = Δα/β`) — changing β does not affect ΔR.

**Deeper problem**: on pseudotime-binned data the `dR/dt` term is extremely noisy, so the GP-predicted α spans −2 to +4 (for example Gm8251), and dividing by β=0.074 can push ΔR above 80.

| Change | Location | Description |
|------|------|------|
| **GP training-target switch** | `kinetics.py:_fit_single_gp` | From `alpha_est = dR/dt + β·R` to fitting `R_seq` directly; the GP learns R=f(A) instead of α=f(A) |
| **Remove the β dependency** | `kinetics.py:fit_atac_to_rna` | No longer computes `degradation_rates` or `beta_g`, does not pass them into GP fitting, and does not store them in results |
| **Steady-state gain redefined** | `kinetics.py:_fit_single_gp` | `steady_state_gain` changes from `dα/dA` to `dR/dA`, which is more intuitive |
| **Remove β division in perturbation** | `perturbation.py:_simulate_knockout_bin_forward` | GP path: `ΔR = R_new − R_orig` (direct); linear path: `ΔR = gain × ΔA` (no /β) |
| **Legacy iterative mode updated too** | `perturbation.py` | Also removes the β_g parameter and the β division (the iterative mode and its functions were removed in v1.28) |
| **Config deprecated** | `config.yaml` | `default_degradation_rate` marked as deprecated in v1.6 |

**Core change**: the model no longer depends on a degradation-rate assumption. The GP fits the observed mapping from chromatin accessibility → RNA expression directly, and during perturbation propagation ΔATAC maps directly to ΔRNA without being derived indirectly through a synthesis rate.

### v1.5 — Simplified degradation-rate strategy + perturbation output filtering (2026-05)

**Problem**: degradation rates estimated by the pseudotime-gradient method were systematically biased for differentiation-induced genes — a high synthesis rate was misread as a low degradation rate, so some genes had β_g near the 0.01 lower bound, the 1/β amplification factor reached 100x, and delta_rna was severely distorted (upregulated genes with delta_rna > 10).

**Root cause**: `_pseudotime_degradation_estimate` inferred the degradation rate from the late/early expression ratio, but a high ratio can come from either a long half-life or sustained high transcription, and the two are indistinguishable.

| Change | Location | Description |
|------|------|------|
| **Remove pseudotime degradation estimation** | `kinetics.py:estimate_degradation_rates` | Deletes the `_pseudotime_degradation_estimate` call; all genes uniformly use `default_degradation_rate: 0.074` |
| **New perturbation output filter** | `perturbation.py:propagate_perturbation`, `config.yaml` | `min_delta_threshold: 0.7` (≈\|log2FC\|>1); perturbation relationships below the threshold are not output |
| **Config cleanup** | `config.yaml` | `min_degradation_rate` marked as deprecated |

### v1.4 — Perturbation-simulation accuracy fix: R² hard filtering + bin-aware KO + state inheritance (2026-05)

**Problem**: perturbation results were abnormal — delta RNA values were too large (±12.7), up/down was roughly 50/50 (15056 up, 12939 down), and a single-gene KO (Rfx8) affected ~28,000 genes.

**Root-cause analysis**:
1. R² soft weighting (floor=0.1) retained low-confidence peak→gene edges, and the noise in those edges produced the 50/50 random up/down distribution
2. The KO was applied to every pseudotime bin, including bins where the target gene is not expressed or the downstream pathway is unreachable
3. Downstream bins were reset to the original state each round, so the KO effect could not accumulate along the pseudotime axis

| Change | Location | Description |
|------|------|------|
| **R² hard filtering replaces soft weighting** | `perturbation.py:_simulate_knockout_bin_forward` | `min_r2_threshold=0.3`; peak→gene edges below the threshold are dropped outright and floor=0.1 soft weighting is no longer used. `gain *= confidence` is also removed from the linear fallback |
| **Bin-aware KO** | `perturbation.py:_simulate_knockout_bin_forward` §2 | Apply the KO only in bins where the target gene is expressed > 0 **and** at least one TF→peak edge satisfies `t + lag < n_bins` |
| **Bin-aware proxy TF** | `perturbation.py:_simulate_knockout_bin_forward` §3 | A proxy TF is repressed only in bins where the KO actually happens (`ko_active` mask) rather than globally |
| **State inheritance (fold-change carrying)** | `perturbation.py:_simulate_knockout_bin_forward` §8 | Maintain `rna_fold` / `atac_fold` vectors and inherit the upstream accumulated fold-change at the start of each bin: `rna_pert[t] = rna_orig[t] * rna_fold`. The fold-change vectors are updated after the bin is processed |
| **New configuration** | `config.yaml` | `perturbation.min_r2_threshold: 0.3` |

**State-inheritance principle**: after a gene/peak is perturbed at bin t, its fold-change (perturbed/original) is written into the fold-change vector. When bin t+1 initializes, every gene/peak original value is multiplied by that vector so the upstream perturbation effect persists. Feedback loops are preserved naturally: if the perturbed gene is a TF, its fold-change keeps propagating through the TF→peak→gene chain in later bins.

> **⚠ Found later**: state inheritance (fold-change accumulation) has a logical flaw in single-round propagation — passing fold between bins makes the perturbation effect diverge monotonically along pseudotime. The code is retained but controlled by `propagation_rounds`: the default `propagation_rounds=1` (single round, no fold accumulation, symmetric propagation) enables multi-round fold inheritance only when `propagation_rounds>1`.

### v1.3 — Perturbation-simulation performance optimization: string-free hot loop (2026-05)

After the thresholds were tightened, perturbation was still slow, with the root cause being heavy use of `list.index()` for name→index mapping in the hot loop of `_simulate_knockout_bin_forward`. RNA has 33k genes and ATAC has 117k peaks, so each `list.index()` scans 117k elements in O(n), and it was called ~126,000 times in the bin loop.

| Optimization | Location | Description |
|------|------|------|
| **Prebuilt name→idx dicts** | `perturbation.py:_simulate_knockout_bin_forward` | The `rna_name_to_idx` / `atac_name_to_idx` dicts replace the O(n) scan with O(1) lookup |
| **Edge indices pre-parsed to integers** | Same as above | `tf_to_peaks`: `{str: [(str,...)]}` → `{int: [(int,...)]}`; `peak_to_genes`: `{str: [dict]}` → `{int: [dict]}`, so the bin loop does no string lookup |
| **GP models pre-deserialized** | Same as above | All GP models are `pickle.loads`ed once before entering the bin loop, avoiding repeated deserialization in the hot loop |
| **Proxy-TF matrix reuse** | `perturbation.py:_propagate_bin_forward` | The RNA matrix is densified once in advance and passed to every non-TF target gene's `_find_proxy_tfs` query, avoiding a repeated `toarray()` per gene |
### v1.2 — Dependency trimming and threshold recalibration (2026-05)

After Gaussian soft binning reduced noise, signal quality improved, so the filtering thresholds were adjusted and redundant dependencies removed.

| Change | Location | Description |
|------|------|------|
| **Remove the scVelo steady-state model** | `kinetics.py:_try_scvelo_degradation` | Deletes the whole function and cuts the scVelo dependency |
| **Simplify the degradation-rate strategy** | `kinetics.py:estimate_degradation_rates` | Three tiers reduced to two: pseudotime-gradient method → default value |
| **Delete CSV loading** | `kinetics.py:_load_degradation_csv` | The pseudotime-gradient method is reliable enough, so the whole function is deleted |
| **Config cleanup** | `config.yaml`, `io.py`, `run.py` | Removes the `degradation_from_scvelo` and `degradation_csv` parameters and their remaining references |
| **Tighter Granger thresholds** | `config.yaml` | `significance_threshold: 0.05→0.01`, `min_composite_score: 0.2→0.3` |
| **Reduced subnetwork depth** | `config.yaml` | `subnetwork_depth: 3→2`; the network is better connected after Gaussian soft binning, so 2 layers are enough to cover the core regulatory neighborhood |

### v1.1 — Gaussian soft binning: denoising small clusters (2026-05)

**Problem**: hard binning averages independently within each bin. A small cluster (for example 200 cells) binned by `min_cells_per_bin=15` yields only ~13 bins — enough cells per bin but too few bins; forcing more bins leaves only 5 cells per bin, so the mean is noisy and the downstream GP fits spurious "repressive" transfer functions (37% misclassified as repressive, R² ~0.07).

**Approach**: Gaussian-kernel soft binning — replace the hard mean with a weighted average so each bin borrows information from neighboring bins through a Gaussian kernel.

| Change | Location | Description |
|------|------|------|
| **Gaussian soft binning** | `granger.py:_bin_expression` | Adds a `smooth="gaussian"` mode. Builds a weight matrix W(n_bins×n_cells): `w_ij = exp(-0.5*(d_ij/σ)²)`, where d_ij is the distance from cell i's pseudotime to bin j's center and σ is adapted automatically |
| **Matrix-level Gaussian binning** | `kinetics.py:_bin_matrix` | The same logic operating directly on numpy matrices (for GP fitting and Ridge regression binning) |
| **Adaptive sigma** | `granger.py`, `kinetics.py` | base σ = pseudotime_span / n_bins. If `avg_cells_per_bin < min_cells_per_bin`, σ is scaled up proportionally: `σ *= min_cells_per_bin / avg_cells_per_bin`, letting small clusters borrow from farther bins |
| **Adaptive bin-count constraint** | `kinetics.py:_adaptive_rebin` | In Gaussian mode the bin count is constrained by `k × n_cells / min_cells_per_bin`, capped at `n_bins`. k is controlled by `smooth_overlap_factor` (default 3.0) to guarantee `σ ≤ k × bin_width`, preventing small clusters from forcing so many bins that adjacent bins become highly redundant |
| **New configuration** | `config.yaml` | `pseudotime.n_bins: 50`, `pseudotime.min_cells_per_bin: 15`, `pseudotime.smooth_kernel: "gaussian"`, `pseudotime.smooth_overlap_factor: 3.0` |

**Large-cluster behavior**: a large cluster (for example 3000 cells) already has ~60 cells per bin, far above `min_cells_per_bin=15`, so σ stays at its base value and is not scaled up. When σ is very small the Gaussian kernel approximates a δ function and soft binning degenerates to hard binning — large clusters are unaffected.

### v1.0 — Perturbation-engine rewrite: TF-specific lag + bin-wise forward simulation (2026-05)

**Design correction**: fixes the deep temporal contradiction between modeling (TF(t-lag)→peak(t)) and perturbation propagation (where the TF effect takes hold instantaneously). The perturbation engine was rewritten completely to implement per-TF time lags and bin-wise forward simulation along pseudotime.

| Change | Location | Description |
|------|------|------|
| **TF-specific lag selection** | `kinetics.py:_select_optimal_lag` | From a single global lag to selecting the optimal lag independently for each TF, returning `Dict[tf_name → lag]` |
| **Multi-lag design matrix** | `kinetics.py:_fit_tf_peak_regression` | Ridge fitting supports a different lag per TF: TF(t - lag[tf]) → peak(t) |
| **tf_lag column output** | `kinetics.py:fit_rna_to_atac` | The `tf_peak_weights` DataFrame gains a `tf_lag` column that is passed to the perturbation engine |
| **Bin-wise forward simulation** | `perturbation.py:simulate_knockout` | Abandons the iterative propagation loop in favor of a forward traversal bin by bin along the pseudotime axis |
| **Delete iterative workarounds** | `perturbation.py` | Removes `_rna_to_atac_step`, `_atac_to_rna_step_gp/linear`, `used_pathways`, the convergence criterion, prev_new_rna, and the temporal-direction check — all replaced by lags plus forward direction |
| **Multi-gene joint KO** | `perturbation.py:simulate_knockout` | Bin-wise mode naturally supports knocking out several genes at once and handles cross/stacking/compensatory effects automatically |
| **New configuration** | `config.yaml` | `perturbation.perturbation_mode: "bin_forward"` (iterative mode was removed in v1.28) |

**Core principle**: traverse each bin forward along pseudotime; a TF change at bin(t-lag) affects the peak at bin(t) after that TF's specific lag, and the peak change immediately affects gene expression in the same bin. If a downstream gene is a TF, its change affects more peaks in later bins — feedback loops are preserved naturally without explicit iteration. A multi-gene KO only needs all KOs applied together when the bin matrix is initialized.

### v0.9 — Filtering-architecture rewrite: 1 hard + 1 soft (2026-05)

**Filtering strategy**: statistical-test hard filtering for TF→peak, R²-confidence soft weighting for ATAC→RNA. No repeat filtering during perturbation propagation.

| Change | Location | Description |
|------|------|------|
| **TF→peak p-value hard filtering** | `kinetics.py:_fit_tf_peak_regression` | `pvalue_threshold=0.05`; edges whose Ridge coefficient is not significant are dropped at the fitting stage |
| **ATAC→RNA R² soft weighting** | `perturbation.py:simulate_knockout` | The gain of a low-R² transfer function is suppressed: `confidence = clip(r2/r2_median, 0.1, 1.0)` |
| **Remove the perturbation safety net** | `perturbation.py:propagate_perturbation` | Drops the second p-value/R² filtering so multiple filtering layers do not push downstream effects beyond the biological range |
| **Remove the R² hard threshold** | `kinetics.py:fit_atac_to_rna` | `gp_quality_threshold` reverts to only triggering deeper optimization and no longer filters results |
| **New configuration** | `config.yaml`, `io.py` | `rna_to_atac.tf_peak_pvalue_threshold: 0.05`, `atac_to_rna.min_degradation_rate: 0.01` |
| **Gain extreme-value clipping** | `kinetics.py:_fit_single_gp` | `steady_state_gain` clipped to ±5.0 to keep extreme GP gradients (max 52.6 → 5.0) out of perturbation propagation |
| **GP-path β safety floor** | `perturbation.py:_atac_to_rna_step_gp` | `max(beta_g, 1e-6)` → `max(beta_g, 0.01)`, consistent with kinetics' `min_degradation_rate` |
| **Tighter propagation upper bound** | `perturbation.py:_atac_to_rna_step_*` | The delta_rna upper bound tightens from `rna_steady*10` to `rna_steady*3` to keep iterative accumulation from growing beyond the biological range |

The GP-fit median R² is ~0.046 (fitting the derived quantity α=dR/dt+βR), so a hard threshold of 0.3 discarded >99% of transfer functions. Soft weighting preserves network connectivity while suppressing the gain of low-confidence edges.
All three independent sources of perturbation amplification are now constrained: β_g ≥ 0.01 (half-life ≤3 days), |gain| ≤ 5.0 (dα/dA slope), and a single-step delta ≤ 3× the expression level.

### v0.8 — Adaptive time-lag Ridge regression (2026-05)

| Change | Location | Description |
|------|------|------|
| **Adaptive lag selection** | `kinetics.py:_select_optimal_lag` | For the subset of high-variance peaks that have candidate TFs, fit Ridge at different lags: TF(t-lag)→peak(t), and pick the lag with the highest R². No prior threshold is needed ("how fast = 1, how slow = 5"); the data decides the optimal lag |
| **Time-lag Ridge** | `kinetics.py:_fit_tf_peak_regression` | From static TF(t)→peak(t) to TF(t-lag)→peak(t) with the lag chosen adaptively. The cell-state parallax filters out pure co-expression noise |
| **Remove Bagging** | `kinetics.py`, `io.py` | Removes the Bagging bootstrap parameters and logic; in io.py's defaults `time_lag: true` replaces `bagging_n_estimators` |

### v0.7 — Network sparsity and propagation-correctness fixes (2026-05)

| Change | Location | Description |
|------|------|------|
| **TF candidate deduplication** | `kinetics.py:_fit_tf_peak_regression` | Deduplicates the candidate TF list (`dict.fromkeys`) so duplicated TFs introduced upstream by supplement/fallback cannot make Ridge regression emit duplicate rows and create a spuriously dense network |
| **TF cap on the fallback path** | `kinetics.py:_assign_tfs_fallback` | Adds a `max_tfs_per_peak` parameter so the fallback path used when motifs are unavailable also respects the per-peak TF cap (before the fix there was no cap and a peak could be assigned every TF) |
| **Family expansion not bound by the cap** | `kinetics.py:_supplement_family_proxies` | Family relationships are biological facts (a shared DNA-binding domain) and are not subject to the top-N limit of motif scanning. v1.29 rewrote this into a two-stage strategy: passively borrow from the cache + actively scan DNA with all family PWMs |
| **Small-cluster Palantir fallback** | `run.py` | When a cluster's cell count < `min_cells_per_cluster`, automatically fall back to global Palantir pseudotime (`fallback_pseudotime_method`), avoiding silently dropping cells or forcing small clusters whose pseudotime inference would be unreliable |

### v0.6 — Logic fixes and statistical improvements (2026-05)

| Change | Location | Description |
|------|------|------|
| **Root-cell raw counts** | `granger.py:430` | Root-cell inference now uses `layers["raw"]` original UMI counts, fixing the log-norm bias |
| **Ridge SE correction** | `kinetics.py:1661` | p-values now use the true `(XᵀX + αI)⁻¹` diagonal, fixing the orthogonality-assumption bug |
| **Multi-cluster edge merge output** | `run.py:417` | causal_edges / tf_peak_weights / transfer_functions merge all clusters and record their source, fixing the bug where only the first cluster was saved |
| **GP steady-state relaxation** | `kinetics.py:257` | α_est = dR/dt + β·R; the pseudotime gradient supplies the dynamic term, relaxing the steady-state assumption |
| **Motif Gumbel correction** | `kinetics.py:1024` | An empirical null distribution plus a Gumbel extreme-value distribution replace the normal approximation, making p-values more accurate |
| **Motif cache threshold validation** | `kinetics.py:973` | Cache loading verifies that motif_pval_threshold matches, preventing silent reuse of an old threshold |

### v0.5 — Degradation rates and output completeness (2026-05)

| Change | Location | Description |
|------|------|------|
| **CSV degradation-rate loading** | `kinetics.py` | Loads per-gene degradation rates from a user-provided scVelo CSV when available |
| **Configurable default degradation rate** | `config.yaml` | `atac_to_rna.default_degradation_rate: 0.074`, replacing the hardcoded value |
| **Clustered h5ad export** | `run.py` | Exports `rna_clustered.h5ad` (with PCA/KNN/Leiden/UMAP) automatically after clustering |
| **UMAP coordinates** | `preprocess.py` | `cluster_cells` gains UMAP dimensionality reduction and writes the coordinates into obsm |

### v0.4 — Reducing false negatives (2026-05)

| Change | Location | Description |
|------|------|------|
| **Soft-threshold Granger** | `granger.py` | composite_score = p_score × 0.4 + effect_score × 0.6, replacing the chained hard thresholds (p < α AND ΔR² > min) |
| **Bagging information output** | `kinetics.py` | 50 bootstrap Ridge rounds provide weight_std / bagging_stability as reference, **without filtering edges** |
| **Family-expansion motifs** | `kinetics.py` | Two-stage strategy: stage A passively borrows from the cache + stage B actively scans DNA sequences with all family PWMs. Only target TFs in the KO list are expanded (rewritten in v1.29) |
| **GP triage optimization** | `kinetics.py` | A first fast fit followed by a layered deeper optimization improves fit quality |

### v0.3 — Configuration completion and dead-parameter cleanup

- Removed `n_branches` (DPT uses a hardcoded `n_branchings=0`, and the parameter was never read)
- config.yaml fully completed to cover every io.py default
- Documentation of pseudotime-method switching (when to use Palantir / scvelo / MultiVelo)

---

## 9. Complete configuration reference

> Every “Default” in this section is the value in the shipped `config.yaml`. Different code fallbacks used when a user configuration omits a key are listed separately at the end of this section.

### input — input data

| Parameter | Type | Default | Description |
|------|------|--------|------|
| rna_h5ad | path | required | RNA AnnData; .var must contain chr, start, end |
| atac_h5ad | path | required | ATAC AnnData; .var must contain chr, start, end |
| genome_fasta | path | — | Reference genome, used for motif scanning |
| target_genes | path | required | KO target-gene list file |

### preprocess.rna — RNA preprocessing

| Parameter | Default | Description |
|------|--------|------|
| min_cells_per_gene | 50 | Gene must be detected in at least N cells |
| min_umi_per_cell | 500 | Cell must have at least N UMIs |
| n_highly_variable_genes | 3000 | Number of highly variable genes retained |

### preprocess.atac — ATAC preprocessing

| Parameter | Default | Description |
|------|--------|------|
| min_cells_per_peak | 10 | Peak must be open in at least N cells |
| min_peaks_per_cell | 1000 | Cell must have at least N peaks |
| binarize | true | Binarize ATAC counts (0/1) |

### pseudotime — pseudotime inference

| Parameter | Default | Description |
|------|--------|------|
| method | dpt | dpt / palantir |
| n_neighbors | 20 | Number of KNN graph neighbors |
| root_cell_marker | null | Marker gene for the developmental start; null=infer automatically |
| root_cell_method | cytotrace | Root-cell strategy: `cytotrace` (default, Scheme B: global CytoTRACE 2020 stemness score → cached argmax within each cluster's obs) / `marker` / `min_umi` / `auto` (falls back to cytotrace when the marker is missing) |
| cytotrace_n_top_genes | 200 | CytoTRACE 2020 takes each cell's top-N highly expressed genes and computes the mean pairwise Pearson correlation (mpc) as the coordination signal |
| n_bins | 50 | Maximum bin count; the actual value adapts per cluster: `min(cells//min_cells_per_bin, n_bins)` |
| min_cells_per_bin | 10 | Target cells per bin, controlling both the bin count and the Gaussian sigma |
| smooth_kernel | gaussian | gaussian / hard. gaussian=soft binning (small clusters borrow from neighboring bins to denoise), hard=conventional equal-count hard binning |
| smooth_overlap_factor | 2.0 | The k in sigma≤k×bin_width for Gaussian mode; controls adjacent-bin overlap tolerance |
| ablation_shuffle_pseudotime | false | Ablation: randomly shuffle the pseudotime bin order to destroy the time-axis information (affects the whole pipeline) |

**Gaussian soft-binning principle**: each bin's value = the weighted mean over all cells, with weights `exp(-0.5*(Δpseudotime/σ)²)`. Large clusters have a small σ → approximately hard binning, so behavior is unchanged; small clusters get an automatically enlarged σ → information borrowed from neighboring bins → noise suppressed. See the v1.1 change record for details.

> **Method choice**: use `dpt` (default) or `palantir`. When neither is available the pipeline uses its own diffusion-pseudotime fallback.

### granger — Granger causal testing

| Parameter | Default | Description |
|------|--------|------|
| distance_thresh | 200000 | Maximum peak-gene genomic distance (bp) |
| significance_threshold | 0.01 | F-test p-value threshold |
| correction_method | fdr_bh | Multiple-testing correction method |
| min_effect_size | 0.1 | Minimum ΔR² effect size |
| soft_threshold | true | Enable soft-threshold composite scoring |
| min_composite_score | 0.3 | Minimum composite score |
| composite_weights | [0.4, 0.6] | [p_adj weight, effect_size weight] |
| max_lag | 1 | Number of Granger lags (lag=1) |
| reference_bins | 50 | (v1.7) Reference bin count for the df correction, used to scale the thresholds |
| ablation_lag0 | false | Ablation: remove the temporal lag and run the pure correlation test Y(t)~X(t) |

### atac_to_rna — ATAC→RNA transfer function

| Parameter | Default | Description |
|------|--------|------|
| log_normalize_rna | true | (v1.23) RNA log1p normalization; the gain becomes a log-fold-change and perturbation converts back to raw space automatically |
| min_observations | 5 | Minimum valid observations required for fitting (5 is more suitable after adaptive binning) |
| root_cell_quantile | 0.05 | (v1.21) Root-cell quantile; the earliest N% of pseudotime cells are used to compute the ATAC baseline |
| root_cell_min_count | 5 | (v1.21) Minimum root-cell count; the quantile expands automatically when insufficient (up to 50%) |
| root_atac_floor | 0.05 | (v1.21) Minimum root-cell ATAC baseline, preventing noise amplification at low-baseline peaks |
| nn_d_embed | 4 | Embedding dimension; a small dimension forces the NN to depend on the ATAC signal |
| nn_d_atac | 16 | ATAC feature-vector dimension, i.e. the number of basis functions |
| nn_atac_hidden_dims | [32,64,128] | atac_net hidden-layer dimension list, controlling model capacity |
| nn_dropout | 0.1 | Dropout ratio |
| nn_batch_size | 4096 | Training batch size |
| nn_learning_rate | 0.001 | Initial learning rate |
| nn_max_epochs | 200 | Maximum training epochs |
| nn_early_stopping_patience | 25 | Early-stopping patience |
| nn_device | cuda | Training device: cuda / cpu |

### rna_to_atac — RNA→ATAC regulatory learning

| Parameter | Default | Description |
|------|--------|------|
| motif_db | jaspar2024 | TF motif database |
| motif_pval_threshold | 0.1 | Motif-scan p-value threshold (candidate prefilter; the Pearson test filters again) |
| max_tfs_per_peak | 5 | Maximum TFs retained per peak |
| tf_peak_pvalue_threshold | 0.05 | TF→peak single-feature Pearson-test p-value threshold |
| time_lag | true | Whether to enable the time lag (independent single-feature Pearson per lag + early-stop). false=only same-time TF(t)→peak(t), and max_lag is forced to 0 |
| max_lag | "auto" | Maximum lag count. "auto"=n_bins//4, 0=instantaneous effect only, N=fixed value. Forced to 0 when time_lag=false |
| min_abs_contribution | 0.01 | (v1.11) Minimum \|weight\| × std(X), filtering TF→peak bindings with too small an effect size |
| tf_expression_filter | true | TF-expression filter switch: verifies that gene names exist in the RNA matrix |

### perturbation — perturbation simulation

| Parameter | Default | Description |
|------|--------|------|
| perturbation_mode | bin_forward | Perturbation mode: bin_forward (bin-wise forward + TF-specific lag) |
| joint_knockout | true | Whether to knock out all target_genes jointly (all targets in one simulation) |
| propagation_rounds | 1 | Propagation rounds. 1=single round with no fold accumulation (default); >1=multiple rounds with state inheritance enabled |
| ko_strength | 1.0 | Knockout strength (1.0=complete knockout) |
| subnetwork_depth | unlimited | BFS subnetwork extraction depth; gene-KO bin_forward uses a strictly downstream executable TF→peak→gene closure after R²/lag/endpoint filtering |
| subnetwork_unlimited_max_edges | 125000 | Maximum simulated edges retained in unlimited mode |
| subnetwork_unlimited_max_work_edges | 250000 | Safety limit for candidate rows traversed during unlimited closure |
| min_lag_per_peak_filter | true | Retain only TFs with the minimum lag per peak; retain all TFs tied at that lag |

`subnetwork_depth: "unlimited"` applies only to gene-KO `bin_forward` and does not expand the whole network without limit:
it only keeps traversing along executable TF→peak→gene directions. The executable simulation edges finally retained are limited by
`subnetwork_unlimited_max_edges` (shipped config.yaml sets 125000); closure traversal is additionally protected by
`subnetwork_unlimited_max_work_edges` (which transparently becomes
`max(max_edges*10, max_edges+100)` when unset). The latter is a conservative safety limit on candidate rows scanned, not equal to the
number of edges finally retained, and it raises an error before continuing to expand. Exceeding either budget raises an error
directly; set `subnetwork_depth` to a finite integer when the closure needs a stricter size limit.
| min_r2_threshold | 0.3 | Hard R² filter threshold for peak→gene edges; edges below it are dropped outright |
| min_delta_threshold | 0 | Perturbation output filter: minimum \|delta_rna\|; 0=no filtering |
| log_accumulation | true | (v1.36) Accumulation space. **true** (default) = accumulate delta_r in log1p-log1p (native NN L2) space (+δ/−δ cancel symmetrically, removing the exponential positive bias of raw accumulation), then back-convert to L1 at the end and compute `log2(L1_pert/L1_orig)` to restore the legacy signal amplitude; **false** = legacy raw path (kept for regression comparison) |
| atac_significance_zscore | 2.0 | (v1.36.1) ATAC-change significance: \|z\| = \|delta_accessibility\| / WT between-bin std. Based on the WT null distribution (the natural between-bin fluctuation along pseudotime of the input atac_adata); \|z\| >= this value is significant. Produces `atac_changes_significant.bed` |
| knockout_type | gene | (v1.37) Perturbation mode switch: **"gene"** (default, TF/gene KO, existing flow unchanged) / **"peak"** (peak KO; reads the peak_ko section below, and target_genes.txt is ignored) |
| peak_ko.peak_id | — | (v1.37) Target peak, e.g. `chr10:1008923-1009780`; supports exact/overlap/nearest matching |
| peak_ko.mode | relative | (v1.37) Perturbation definition: **relative** `A′=A(1+s)` (recommended; s=-1 fully closes) / absolute `A′=s` (raw [0,1] target value) |
| peak_ko.strength | -1.0 | (v1.37) Perturbation strength s ∈ [-1,+1]: -1=fully closed, -0.5=50% reduction, +0.5=enhanced |
| peak_ko.depth | 2 | (v1.37/1.38) Propagation depth: 1=direct (direct target genes only) / 2/3=cascade (v1.38: TF→secondary peak→secondary gene, layer-limited BFS) |
| peak_ko.state | all | (v1.38) Leiden cluster label (run a single cluster only) or "all" (all clusters + merged weighted by cluster cell count) |
| peak_ko.match | overlap | (v1.37) Peak matching strategy: exact / overlap / nearest (distance is output explicitly; never silently substituted) |
| cell_projection.enabled | false | Enable optional RNA reference projection |
| cell_projection.n_components | null | Retain all estimable PCA components |
| cell_projection.n_neighbors | 10 | Reference neighbors for inverse-distance projection |
| cell_projection.history_delta_threshold | 1e-8 | Sparse-history nonzero cutoff |
| cell_projection.state_unit | rna_log1p_cp10k | Unit of projected RNA state values |
### clustering — cell clustering

| Parameter | Default | Description |
|------|--------|------|
| enabled | true | Whether to enable clustering. false=global pseudotime mode, true=per-cluster DPT mode |
| cluster_key | leiden | Column name for cluster labels in obs |
| n_neighbors | 20 | Number of KNN graph neighbors |
| n_pcs | 20 | Number of PCA components |
| resolution | 0.5 | Leiden resolution |
| random_state | 42 | Random seed |
| min_cells_per_cluster | 50 | Minimum cells per cluster (smaller clusters trigger fallback) |
| fallback_pseudotime_method | palantir | Pseudotime method for the global fallback (Palantir recommended) |
| clustering.paga.enabled | false | Whether to enable PAGA lineage merging: identify topological connectivity between clusters and merge connected clusters into one lineage before inferring pseudotime |
| clustering.paga.connectivity_threshold | 0.08 | PAGA connectivity threshold; between-cluster links below this value are not treated as the same lineage |
| clustering.paga.min_cells_per_lineage | 50 | Minimum cells in a merged lineage (smaller lineages merge into the nearest lineage) |

### output — output

| Parameter | Default | Description |
|------|--------|------|
| dir | results/ | Output directory |
| cache_dir | cache/ | Cache directory (motif scans, TF databases, etc.) |
| checkpoint_dir | null | Checkpoint directory (null=use output.dir/checkpoints) |
| save_intermediate | true | Whether to retain intermediate files

### Code fallbacks (only when a user configuration omits a key)

The following values come from `atac_bridge/io.py` and apply only when the corresponding key is absent from a user configuration; they do not replace the shipped `config.yaml`.

| Parameter | Code fallback |
|------|------------|
| pseudotime.n_neighbors | 30 |
| pseudotime.n_bins | 100 |
| granger.distance_thresh | 1000000 |
| granger.significance_threshold | 0.10 |
| granger.min_effect_size | 0.02 |
| rna_to_atac.motif_pval_threshold | 1e-4 |
| rna_to_atac.max_lag | 0 |
| perturbation.subnetwork_unlimited_max_work_edges | null (then derives `max(max_edges*10, max_edges+100)`) |
| clustering.resolution | 0.8 |
| clustering.paga.enabled | true |
| clustering.paga.connectivity_threshold | 0.1 |

---

## 10. Ablation study design

Ablation experiments verify how much each CausalBridge component contributes to prediction performance. By disabling core mechanisms one at a time, the individual importance of pseudotime ordering, temporal lag, and binning-based denoising can be quantified.

### Ablation dimensions

| Ablation | Configuration | What it breaks | What it keeps |
|----------|---------|---------|---------|
| **Remove the Granger lag** | `granger.ablation_lag0: true` | The 1-step temporal lag of peak→gene | Pseudotime ordering, bin aggregation |
| **Remove the TF→peak lag** | `rna_to_atac.time_lag: false` | The time-offset search for TF→peak | Pseudotime ordering, motif information |
| **Shuffle pseudotime bins** | `pseudotime.ablation_shuffle_pseudotime: true` | The temporal order of bins | Bin contents (the same cells aggregate) |
| **All off (pure-correlation baseline)** | The three above all enabled | Temporal lag + temporal ordering | Binning denoising, motif bridging |

### Expected conclusions

| Ablation outcome | Meaning |
|----------|------|
| **Performance collapses after shuffling pseudotime** | Pseudotime ordering is the main source of advantage — covariation along the developmental trajectory provides a stronger causal signal than cross-cell static correlation. This also explains why CausalBridge outperforms CellOracle |
| **Still better than CellOracle after shuffling pseudotime** | The advantage comes from binning aggregation denoising + motif bridging rather than from the time axis itself |
| **Little effect from removing the lag** | Gaussian soft binning makes adjacent bins highly overlapping (`sigma ≤ 2×bin_width`), so lag=0 and lag=1 data are nearly equivalent and the temporal-lag information has already been smoothed |
| **Large effect from removing the lag** | Bins are strongly independent and there is a genuine causal-delay signal (such as chromatin remodeling preceding transcriptional activation) |

### Usage

```yaml
# config.yaml — example ablation combinations

# 1. Pure-correlation baseline (all ablations on)
granger:
  ablation_lag0: true                    # Y(t)~X(t) replaces Granger

pseudotime:
  ablation_shuffle_pseudotime: true      # shuffle the pseudotime axis

rna_to_atac:
  time_lag: false                        # TF(t)→peak(t) with no lag

# 2. Shuffle pseudotime only (keep lags)
pseudotime:
  ablation_shuffle_pseudotime: true      # the core validation item

# 3. Remove lags only (keep pseudotime ordering)
granger:
  ablation_lag0: true
  ablation_shuffle_pseudotime: false
```

### Caveats

- `ablation_shuffle_pseudotime` uses a fixed seed `seed=42`, consistent within a cluster, so results are reproducible
- After shuffling, `branch_boundaries` is reset to zero automatically (branch-aware features depend on time ordering)
- `time_lag: false` disables the temporal lag directly (only same-time TF(t)→peak(t) regression)
- Enable only one ablation at a time and compare against the original results to separate each component's contribution

---

## 11. Core differences from CellOracle

| Aspect | CausalBridge | CellOracle |
|------|-------------|------------|
| Edge inference | Granger causal testing (temporal lag) | Co-expression correlation |
| Prior information | motif + genomic distance | motif only |
| Transfer function | Monotonic-constrained NN (nonlinear) | Linear steady-state matrix |
| Perturbation propagation | Bin-wise forward propagation + TF-specific lag | Static matrix solve |
| Chromatin modeling | End-to-end (peak-level opening/closing → expression) | TF→gene layer only |
| Cell-type modeling | Independent modeling per cluster | Global modeling |
| TF scope | Determined by JASPAR motif scanning | Hardcoded TF list |
| Non-TF targets | Co-expression proxy-TF mapping | Not supported (must KO a TF) |
| Edge reliability | Temporal-lag causal filtering + motif prior + per-lag single-feature Pearson | A single correlation/regression coefficient |
| Network completion | None | None |
