# CausalBridge Project Skills File

> GitHub: [Senki-zip/CausalBridge](https://github.com/Senki-zip/CausalBridge)

## Session Information

- **Session ID**: `ses_1498167f0ffe8BNHU4EKwnJ8MP`
- **Creation Time**: 2026-06-11
- **AI Agent**: Sisyphus (OhMyOpenCode)
- **Model ID**: `xiaomi/mimo-v2.5-pro`

## Project Overview

CausalBridge is an **ATAC-bridged causal perturbation prediction framework**. Using single-cell multi-omics data (scRNA-seq + scATAC-seq), it uses chromatin accessibility as a mechanistic bridge to construct a **TF → Peak → Gene** three-layer causal regulatory network and predict whole-transcriptome + chromatin-state shifts after in silico gene knockout.

Core insight: ATAC signals capture the earlier state of chromatin opening in the same cell, while RNA signals capture the later state of transcription—the “cell-state parallax” makes Granger causal testing possible.

## Project Structure

```
atac_bridge/                  # Main code directory
├── run.py                    # Pipeline entry point; runs the full workflow and resumes from checkpoints
├── io.py                     # Configuration parsing, data loading, result saving, validation
├── preprocess.py             # RNA/ATAC QC normalization, PCA+KNN+Leiden clustering+UMAP
├── granger.py                # Pseudotime inference, adaptive Gaussian soft binning, Granger causal testing
├── kinetics.py               # NN transfer-function fitting, motif scanning, independent regression per lag
├── nn_transfer.py            # NN transfer function (PairEmbeddingNN)
├── perturbation.py           # Perturbation simulation engine, subnetwork extraction, forward simulation per bin
config.yaml                     # User configuration file
target_genes.txt                # KO target-gene list
run_peak_perturbation.py        # Standalone CLI: peak perturbation trial (reuses checkpoints)
convert_10x_to_atac_bridge.py   # Converts 10x output to atac_bridge input format
visualize_perturbation.R        # R visualization: Volcano + Pathway Sankey plots
plot_causal_network.py          # Python visualization: causal network plot
```

Data/output/cache paths are specified by the `input` / `output` sections in `config.yaml`, decoupled from the project code.

## Analysis Workflow

```
Step 1: Load configuration + data (RNA + ATAC AnnData)
Step 2: Preprocess (QC/normalization/highly variable genes/binarized ATAC) + PCA + KNN + Leiden clustering
Step 3: Infer pseudotime by cell cluster (Palantir/DPT, v1.29+ independently per cluster) + adaptive Gaussian soft binning
Step 4: Granger causal testing by cell cluster (with degrees-of-freedom correction; ablation_lag0 switches to pure correlation testing)
Step 5: Fit NN transfer functions by cluster
Step 6: Learn RNA→ATAC TF regulatory weights by cluster (motif-selected candidate TFs + single-feature Pearson per TF/per lag; first significant lag wins)
Step 7: Integrate causal GRN
Step 8: Simulate perturbations by cluster (forward per bin + additive accumulation)
Step 9: Aggregate and save results
```

## Core Design Principles

### 1. Changes Must Follow Real Biological Logic
Every algorithmic change must answer: “Does this change conform to real gene-regulation/knockout logic?” Changes that are purely mathematical optimizations but violate biological intuition should not be implemented.

### 2. Cluster-Specific Modeling
Cells within a cluster are more homogeneous, NN fitting noise is easier to control, and TF regulation is cell-type-specific.

### 3. Per-cluster Adaptive Binning
Each downstream module independently calculates the adaptive number of bins from the actual cell count of the current cluster. Gaussian soft binning constrains adjacent bins from being redundant through `sigma ≤ k × bin_width`. Small clusters automatically use fewer bins; large clusters retain 50 bins.

### 4. Filtering Must Be Completed Inside the Result-Producing Module
TF→peak uses hard statistical-test filtering (in kinetics.py), and ATAC→RNA uses hard R² filtering (in perturbation.py). Do not filter again late in the pipeline. **If a p-value is calculated, it must actually be used for filtering** (a lesson from experience).

### 5. Zero Impact on the DPT+Clustering Path
All branch-aware changes (Palantir lineage splitting) must leave DPT+clustering-path behavior completely unchanged. When `branch_labels is None`, everything must fall back to the existing logic.

### 6. Separate Propagation_rounds=1 and >1 Paths
- `propagation_rounds=1` (default): single round, persistent additive accumulation, symmetric propagation
- `propagation_rounds>1`: multiple rounds, multiplicative fold inheritance (with a risk of monotonic divergence)

The two paths are clearly separated by `if/else` in `_propagate_bin_forward`.

### 7. Global CytoTRACE Cache Contract (v1.35+)
`run_pipeline` Step 2.5 writes the global stemness score (Scheme B) to `rna_adata.obs["cytotrace_score"]`.
When slicing by cluster, the `.obs` column is naturally retained; `_find_root_cell` obtains `iroot` by taking the `argmax` of this column within the cluster.
If the column is missing (when called externally on a single file), `_find_root_cell` computes it within the subcluster (Scheme A, equivalent but without the global advantage).
`root_cell_method` options: `cytotrace` (default) / `marker` / `min_umi` / `auto`.
When modifying `_compute_cytotrace_scores`, also verify that the `obs` column is written and propagated through slicing—any lost column will silently downgrade to Scheme A.

### 8. Accumulation-Space Contract (v1.36+)
When `perturbation.log_accumulation=True` (default and requires `atac_to_rna.log_normalize_rna=True`):
`delta_cum` directly accumulates the NN delta_r in log1p-log1p (L2) space (**skip** `_delta_r_log_to_raw`).
At the end, do **not** use `delta_cum/log(2)`: first back-convert to L1 via `expm1(log1p(L1_orig) + delta_cum)`,
then calculate log2FC with `log2(L1_pert/L1_orig)`, and truncate with
`np.nan_to_num(log2fc, nan=0.0, neginf=-5.0, posinf=5.0)`
(perturbation.py ~L1754-1758).
When modifying edge firing / per-bin TF inheritance / endpoint aggregation, all three must remain coordinated:
- edge-firing accumulation layer: delta_r must enter delta_cum directly, **without** `_delta_r_log_to_raw`
- per-bin TF inheritance: use `expm1(log1p(L1_orig) + delta_cum)` to back-convert rna_pert to L1 (for downstream TF→peak delta_tf)
- endpoint output: back-convert to L1, then calculate `log2(L1_pert/L1_orig)` + ±5 truncation (since v1.38.2, TF output is weighted by first-trigger event time; non-TF output reconstructs the existing delta_cum endpoint path)
Otherwise, any mismatch among the three reintroduces positive bias from raw accumulation or unit inconsistency. `log_accumulation: false` switches back to the legacy raw path for regression comparison.

## Key Version-Change Summary

| Version | Date | Core change |
|------|------|---------|
| v1.39 | 2026-09 | **NN-only transfer functions + removal of lag modes 2/3, linear decay, and dead code**: removed GP fitting/serialization/distribution and `method`/`gp_kernel`/`gp_quality_threshold`/`gp_n_restarts`/`delta_r_method`/`path_integral_steps`; removed Bagging (`bagging_n_estimators`/`bagging_stability_threshold`); removed the `multi_lag`/`auxiliary_lags` multi-column design matrix and `lag_penalty`/`lag_penalty_mode`; removed the unreachable RidgeCV branch and ineffective `variance_rescale`; added explicit empty-training errors, linear fallback on inference failure, `rna_log_normalized` metadata validation, rejection of old Step 5/6 checkpoints, and small-sample NN split fixes |
| v1.36 | 2026-07 | **Whether Log-space delta accumulation resolves upward bias**: delta_cum accumulated in log1p-log1p (NN-native L2) space with symmetric +δ/−δ; endpoint delta_rna=delta_cum/log(2); removed the false-symmetry neginf=-5/posinf=+5 truncation; config.perturbation.log_accumulation defaults to true |
| v1.35 | 2026-07 | **CytoTRACE 2020 root-cell selection** (Scheme B): global gc + mpc stemness score → obs cache → within-cluster argmax mapping; replaced min_umi default |
| v1.34 | 2026-06 | **Motif deduplication**: reused tf_best logic + all_factors merging; internal Cofactor identification + proxy-TF weight/threshold optimization |
| v1.33 | 2026-06 | **Per-lag indentation bug fix**: Pearson replaces RidgeCV (single feature per lag); fallback Granger lag=1 strategy |
| v1.32 | 2026-06 | **Code cleanup**: fixed window-validation boundaries; removed the obsolete TF-expression-trend filter (Spearman + median) |
| v1.31 | 2026-06 | **RNA root-baseline sign test**: independent direction validation, not dependent on the Granger beta_2 sign |
| v1.30 | 2026-06 | **Per-TF independent Ridge + Pearson r sign verification** |
| v1.29 | 2026-06 | **Pseudotime cluster inference + adaptive thresholds + two-stage family expansion rewrite**: passive borrowing + forced scanning of all family PWMs; causal rate 72-91% |
| v1.28 | 2026-05 | **Code cleanup**: removed nn_gp / nn_sc paths |
| v1.27.5 | 2026-05 | **TF-expression-trend filter**: excluded inactive TFs and reduced false TF→peak edges |
| v1.27 | 2026-05 | **Configurable NN capacity**: atac_hidden_dims configurable, default [64,128,256,256,128] |
| v1.26 | 2026-05 | **NN monotonic constraint**: PairEmbeddingNN atac_net weights >= 0, fixing gain-sign contradictions |
| v1.25 | 2026-05 | **Sign Correction v2**: three-level OLS + condition number + peak fallback strategy, eliminating TF→peak sign flips; removed multi-lag summation |
| v1.24 | 2026-05 | **Cross-cluster absolute-value weighted merging**: delta_rna changed to sum(\|delta\| * weight), eliminating cross-cluster directional cancellation |
| v1.23 | 2026-05 | **Reduced candidate-gene pool + RNA log1p normalization** |
| v1.22 | 2026-05 | **Sign correction + multi-lag aggregation**: fixed sign bias caused by TF collinearity |
| v1.21 | 2026-05 | **Scheme B [-1,1] piecewise ATAC scaling + configurable root-cell parameters**: symmetric perturbation space |
| v1.20 | 2026-05 | **ATAC perturbation clipping + GP OOB safety net**: falls back to linear prediction outside the training range |
| v1.19 | 2026-05 | **Path-integral delta_r**: replaced point prediction and eliminated GP clipping/fallback bias |
| v1.18 | 2026-05 | **Fixed duplicate pathway_records writes**: one gp/linear record per edge |
| v1.17 | 2026-05 | **Separated TF/non-TF effects**: eliminated perturbation upward bias + tightened parameters |
| v1.16 | 2026-05 | **Fallback from distributed-lag Granger to lag=1** |
| v1.15 | 2026-05 | **Distributed-lag Granger causal testing** |
| v1.14 | 2026-05 | **Removed log2FC hard upper limit + cross-cluster weighted average**: removed ±5.0 clip, added perturbation_merged.csv |
| v1.13 | 2026-05 | **Persistent additive accumulation**: each (peak,gene) edge’s delta persists in all subsequent bins after first firing |
| v1.12 | 2026-05 | **Removed orphan-peak fallback**: use only Granger causal-test edges |
| v1.11 | 2026-05 | **Fixed per-cluster adaptive binning**: reverted to per-module `_adaptive_rebin`, eliminating small-cluster false positives |
| v1.10 | 2026-05 | **Bagging stability + effect-size clipping**: dual filtering of weak TF→peak edges |
| v1.9 | 2026-05 | **Distributed-lag Ridge + variance rescaling**: fixed double decay (2e-6 → 1e-4) |
| v1.8 | 2026-05 | **TF-specific lag winner voting**: replaced mean-R² selection and corrected random selection in small samples |
| v1.7 | 2026-05 | **Second Granger degrees-of-freedom correction**: scaled threshold by (df_ref/df_actual)² |
| v1.6 | 2026-05 | **GP training-target switch**: changed from synthetic rate α to directly fitting RNA expression R |
| v1.5 | 2026-05 | **Simplified degradation-rate strategy**: removed pseudotime degradation-rate estimation |
| v1.4 | 2026-05 | **R² hard filtering + bin-aware KO + state inheritance** |
| v1.3 | 2026-05 | **Zero-string lookup in hot loops**: prebuilt name→idx dict |
| v1.2 | 2026-05 | **Reduced dependencies + recalibrated thresholds** |
| v1.1 | 2026-05 | **Gaussian soft binning**: small clusters borrow information from neighboring bins to reduce noise |
| v1.0 | 2026-05 | **Perturbation-engine refactor**: TF-specific lag + forward simulation per bin |
| v0.9 | 2026-05 | **Filtering-architecture refactor**: 1 hard + 1 soft |
| v0.8 | 2026-05 | **Adaptive time-lag Ridge** |
| v0.7 | 2026-05 | **Network-sparsity fix** |
| v0.6 | 2026-05 | **Statistical corrections**: Ridge SE, motif Gumbel |
| v0.5 | 2026-05 | **Degradation-rate CSV + clustered h5ad export + UMAP** |

## Key Code Patterns

### Bin-Aware Operations
Modules involving pseudotime binning must call `_adaptive_rebin` per module and must not depend on global binning at the entry point:
```python
pseudotime_df = _adaptive_rebin(pseudotime_df, adata.n_obs, ...)
```

### Perturbation Accumulation Pattern (v1.13, updated through v1.36+)
```python
delta_cum = np.zeros(n_rna)
fired_edges = set()  # (peak_idx, gene_idx)

# ATAC→RNA step: add delta once when each (peak,gene) edge fires for the first time
# (v1.36+ accumulates directly in log1p-log1p L2 space; see §8 Accumulation-Space Contract)
edge_key = (peak_idx, gene_idx)
if edge_key not in fired_edges:
    delta_cum[gene_idx] += delta_r
    fired_edges.add(edge_key)

# Per-bin TF inheritance (only TFs are written back to the propagation matrix; the np.maximum(0,...) clip was removed in v1.17):
# rna_pert[t, active] = expm1(log1p(rna_binned_orig[t, active]) + delta_cum[active])
```
> Note: the asymmetric `np.maximum(0, rna_binned_orig + delta_cum)` clip in the original v1.13 code was removed in v1.17.

### Branch-Aware Operations (Implemented)
All time-lag operations must be handled through `_branch_aware_lag_pairs` and must respect branch boundaries:
```python
mat_before_lagged, mat_after_lagged = _branch_aware_lag_pairs(
    binned_mat, binned_mat, lag, branch_boundaries
)
```

## Branch-Aware Binning (Palantir Lineage-Specific Binning) — Implemented

**Status**: Implemented (adaptive per-branch qcut rebinning + branch-aware lag pairs)

**Implementation**: when `branch_labels`/`branch_boundaries` exist, `_adaptive_rebin` performs qcut binning independently by branch (globally unique bin IDs), and `branch_boundaries` is passed through `pseudotime_df.attrs` with zero signature changes;
`_branch_aware_lag_pairs` ensures lag slices do not cross branch boundaries.

**Wired modules**: granger.py (lag pairs), nn_transfer.py (lag pairs), perturbation.py (forward-propagation branch traversal).

**Key constraint**: when `branch_labels is None`, everything falls back to the existing logic, with zero impact on the DPT+clustering path.

## GitHub Repository Information

| Item | Value |
|------|-----|
| Repository name | **CausalBridge** |
| Address (HTTPS) | `https://github.com/Senki-zip/CausalBridge.git` |
| Address (SSH) | `git@github.com:Senki-zip/CausalBridge.git` |
| Default branch | `main` |
| Local path | `/home/huangtao/Desktop/huangchengqi/code_1/atac_bridge` |
| Update date | 2026-07-13 |

## Common Git Commands

```bash
# View status and changes
git status
git diff --stat
git log --oneline -10

# Stage and commit
git add <file1> <file2> ...
git commit -m "v1.XX: <brief description>"

# Push to GitHub
git push origin main

# Undo
git restore <file>              # discard unstaged changes
git reset HEAD <file>           # unstage

# View history
git diff HEAD~1                 # changes in the last commit
git log --oneline --since="2026-06-01"  # filter by month
```

## Notes

1. **Run the pipeline to confirm baseline behavior before modifying**: the code is highly coupled, and small changes may be amplified downstream
2. **Logs first**: every key branch should have logger.info/warning for tracing the execution path
3. **Configuration synchronization**: new parameters must also update config.yaml + io.py defaults + the README parameter table
4. **Backward compatibility**: if old configuration parameters must be deprecated, mark them `[deprecated]` but retain their defaults; do not delete them directly
5. **Threshold adjustments require data support**: do not change thresholds by intuition; use the statistical distribution of actual run results
6. **NN and Pearson paths are independent**: NN is used for the ATAC→RNA transfer function, Pearson for RNA→ATAC regulatory weights; modifying one must not affect the other
7. **Verify that p-values are actually used for filtering**: this issue has occurred before
8. **Granger ablation_lag0 mode**: set `ablation_lag0: true` in the `granger` section of `config.yaml` to switch to pure correlation testing Y(t)~X(t) for ablation comparison
9. **TF→peak time_lag control**: set `time_lag: false` in the `rna_to_atac` section of `config.yaml` to disable time lag (then max_lag is forced to 0, with only same-time TF(t)→peak(t) regression)
10. **Ablation: randomly shuffle pseudotime values**: set `ablation_shuffle_pseudotime: true` in the `pseudotime` section of `config.yaml` to randomly shuffle each cell’s pseudotime value (seed=42). Downstream `_adaptive_rebin` uses `pd.qcut` to cut the shuffled values by quantile, aggregating random cells into each bin and fundamentally destroying time-axis ordering. This affects the entire workflow (Granger / kinetics / perturbation). It tests “whether pseudotime ordering itself is the core information source for causal inference”—if performance drops sharply after shuffling, the advantage comes from covariation along the trajectory rather than bin aggregation

## Ablation Experiment Design

| Ablation dimension | Configuration | What is disrupted | What is retained |
|----------|------|---------|---------|
| Remove time lag (Granger) | `ablation_lag0: true` | 1-step peak→gene lag | Pseudotime ordering, bin aggregation |
| Remove time lag (TF→peak) | `time_lag: false` | TF→peak time shift | Pseudotime ordering, motif information |
| **Shuffle pseudotime binning** | `pseudotime.ablation_shuffle_pseudotime: true` | Pseudotime ordering | Bin contents (same cells aggregated) |
| All off (pure correlation baseline) | Enable all three above | Time lag + time ordering | Bin denoising, motif information |

Expected core conclusions of the ablation experiment:
- If **pseudotime shuffling** causes a substantial performance drop → pseudotime ordering is the core advantage and outperforms CellOracle’s cross-cell static correlation
- If performance after **pseudotime shuffling** remains better than CellOracle → the advantage comes from bin denoising + the motif-bridging mechanism
- If **removing lag** has little impact → adjacent bins overlap heavily (Gaussian soft binning), so lag information has already been smoothed

## Core Differences from CellOracle

| Aspect | CausalBridge | CellOracle |
|------|-------------|------------|
| Edge inference | Granger causal testing (time lag + degrees-of-freedom correction) | Coexpression correlation |
| Transfer function | NN (nonlinear, R=f(A)) | Linear steady-state matrix |
| Perturbation propagation | Forward per bin + TF-specific lag + persistent additive accumulation | Static matrix solution |
| Chromatin modeling | End-to-end (peak-level opening/closing → expression) | TF→gene layer only |
| TF scope | JASPAR motif scanning + family proxies | Hard-coded TF list |
| Non-TF targets | Coexpression proxy-TF mapping | Not supported |
