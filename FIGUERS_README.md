# TF KO Figure Guide

Companion to the [CausalBridge v2.0 guide](README.md). See the corresponding README sections for installation, execution, output contracts, and known limitations. This document explains the fields, units, and selection criteria used in the current figures.

This directory contains visualizations of CausalBridge results for single-TF KO or joint multi-TF KO. Red generally indicates activation or upregulation, blue indicates repression or downregulation, and gray indicates stable, missing, or unobserved results; consult each figure's description for its precise meaning. Each output file retains only one main title at the top. Methods and selection notes are documented here.

## Script entry point

- Single-TF KO: `Rscript visualize_perturbation.R RESULT_DIR`

Input and output directories, specified genes, and top-N settings follow the script configuration and can be overridden through command-line arguments and environment variables. Internal joins, grouping, and filtering use stable gene IDs, while figures display gene symbols; distinguish genes with identical symbols by ID. Path labels such as `cluster_0` align only when the results table contains the corresponding `0`; they do not justify merging otherwise distinct cell clusters.

### Current model and result conventions (shared by all figures)

In `saturating_hybrid`, the TF→peak and peak→gene layers select `rna_to_atac.response_model` and `atac_to_rna.response_model`, respectively, supporting `spline`, `nn`, and `saturation`. The default is saturation when unspecified; the configuration and exported fields used to generate that batch of CSV files are authoritative. Single-input relationships are response functions with physical lower and upper output bounds: spline uses a bounded spline, and NN uses a separate bounded network. In the model's native RNA state, peak→gene uses gene L2 response units, TF→peak takes the TF's native RNA state as input, and ATAC output uses native accessibility units. These values cannot be directly added to or compared in magnitude with the final cellular CP10k log2FC.

After WT fitting, the per-relationship mixing weights α remain fixed and are not renormalized using changed inputs during perturbation. Optional two- or three-input groups replace the corresponding batches of single relationships only when training quality, local geometry, effective cell support, and support for both WT and intervention queries pass their checks. Otherwise, single-relationship responses with fixed α are retained. Training `R²` is evidence of fit, not independently validated accuracy; plotting does not refit or modify the model.

Coverage and status notes previously shown at the bottom of figures are recorded here. Read each gene's actual values from the CSV files for that run rather than applying values from another run:

- `aggregation_scope`: `full` means complete coverage; `partial` represents only the included subset of valid cells; `unavailable` means no aggregate is available.
- `aggregation_cell_fraction`: fraction of cells included in aggregation; `aggregation_used_contexts` / `aggregation_total_contexts`: used/total context counts; `aggregation_context_exclusions`: reasons for exclusion.
- `status`: whether the effect is valid. Plotting uses finite effects with status `ok`; missing, quality-filtered, or unavailable effects must not be interpreted as zero.
- `display_clipped`: whether the displayed effect reaches the reporting cap. When present, `l0_log2fc_unclipped` retains the result before clipping. The display cap differs from the model's physical response bounds.

Global cards and global effect plots use the saved `perturbation_merged.csv::delta_rna_signed`. For `valid_cell_means`, this is computed as `log2(perturbed mean / WT mean)` from cell-count-weighted CP10k means for WT and perturbation over the same subset of valid cells. It is neither the arithmetic mean of cluster log2FC values nor the sum of displayed path contributions. Handling of missing contexts follows that batch's aggregation strategy and coverage fields.

## 1. `volcano.png`: Perturbation effects and model fit quality

This is not a traditional volcano plot with `-log10(p)` on the y-axis. It is a two-dimensional scatter plot showing gene perturbation effects alongside support from the ATAC→RNA model fit.

- X-axis: mean RNA change, `ΔRNA (log2 fold change)`.
- Y-axis: mean fit quality `R²` of the corresponding gene's ATAC→RNA transfer functions.
- Red points: `log2FC > 1`, defined as strong upregulation.
- Blue points: `log2FC < -1`, defined as strong downregulation.
- Gray points: finite values with `|log2FC| ≤ 1`, defined as stable or weaker effects; missing values or `NaN` must not be interpreted as weak effects.
- Vertical dashed lines: effect classification thresholds at `log2FC = ±1`.
- Horizontal dashed line: reference at `R² = 0.5`.
- Gene labels: prioritize genes with larger `|log2FC| × R²`, combining larger effects with better model support.

Focus on the upper-right and upper-left regions: these genes have both substantial expression changes and higher ATAC→RNA fit quality. This plot does not express statistical significance, and its y-axis must not be interpreted as a p-value.

## 2. `distribution.png`: Perturbation effect distribution and classification

This figure has two panels.

### Left: Effect distribution

- X-axis: mean `ΔRNA` for all affected genes.
- Y-axis: number of genes in each interval.
- Blue bars: negative effects, or downregulation.
- Red bars: positive effects, or upregulation.
- Solid black line: zero effect.
- Dashed red line: median effect across all genes.

This panel shows whether the global response after KO tends toward upregulation or downregulation, and whether the effect distribution has long tails or pronounced skew.

### Right: Effect classification

- Up: `log2FC > 1`.
- Down: `log2FC < -1`.
- Stable: `|log2FC| ≤ 1`.

Classification uses effect-size thresholds only, not a statistical significance test, hence the title **Effect Classification**.

For the corresponding ATAC accessibility plot, see `atac_distribution.png` in Section 9.

## 3. `cluster_heatmap.png`: Differential effects across cell clusters

This heatmap shows the direction and magnitude of gene responses to the current target TF KO across cell clusters.

- Rows: affected genes.
- Columns: cell clusters.
- Red: RNA upregulation.
- Blue: RNA downregulation.
- White: observed effects close to 0.
- Gray: no available result or no observation for that gene in the corresponding cluster.
- Color range: plotting may cap the color mapping at `[-3, 3]` to prevent extreme values from compressing the main color differences. This is a **visualization-only colour cap**; it neither modifies CSV effect values nor clips the data.

The heatmap prioritizes genes whose effects differ more strongly across clusters. Genes are ranked by the following composite score. The ranking subtitle is no longer displayed in the figure; this document defines the calculation. Gray cells represent unobserved results and are excluded from `effect_range` and `effect_sd` calculations across that gene's observed clusters.

```text
heterogeneity_score = coverage × (effect_range + effect_sd) × direction_bonus
```

Here:

- `coverage`: fraction of all cell clusters with results, reducing the influence of isolated extremes in only a few clusters.
- `effect_range`: difference between the maximum and minimum cluster `ΔRNA` values.
- `effect_sd`: standard deviation of cluster `ΔRNA` values.
- `direction_bonus`: `1.25` if a gene shows both upregulation and downregulation across clusters; otherwise `1`.

Genes near the top therefore tend to show larger differences in magnitude or direction reversals across clusters. Gray and white have different meanings: missing results must not be treated as zero effects.

## 4. `pathway_specified.png`: Regulatory paths for specified genes

This plot uses the same **radial cluster hierarchy** as the automatic pathway plots in Section 5. A single gene in `PATHWAY_GENES` produces one radial panel; multiple genes each receive a separate panel, stacked vertically. Paths for different genes are never merged.

From the center outward, each panel represents:

```text
KO TF → Target peak → Intermediate TF (indirect paths) → TF peak → Terminal effect within cluster
                                                               ↓
                                                     GLOBAL WEIGHTED OUTCOME
```

- Center circle: KO TF; labels such as `C0` and `C1` near its inner edge identify the cluster sectors.
- Each cluster occupies a sector with a base area. Its angle and radial depth increase with a logarithmically compressed value of that cluster's `|ΔRNA|`. Weak-effect, zero-effect, and missing clusters remain readable, while stronger-effect clusters occupy more area.
- Successive block layers represent target peaks, intermediate TFs, TF peaks, and terminal effects. Channels between peak/TF blocks accommodate arrows. Direct paths that cross intermediate-layer blocks are curved; unobstructed direct paths and indirect paths use straight lines.
- Terminal-effect blocks fill the outermost layer of their sectors. Neutral connectors outside the sectors connect observed cluster terminal effects to the `GLOBAL WEIGHTED OUTCOME` below. This card takes its value directly from `perturbation_merged.csv::delta_rna_signed`; it is a weighted aggregate, not a simple sum of the displayed path contributions.
- Terminal-effect blocks are colored according to the cluster's `delta_rna` in `perturbation_results.csv`; observed responses close to zero appear as neutral blocks.
- If a cluster has retained paths but no corresponding finite `delta_rna` in `perturbation_results.csv` (including `NaN`), its terminal effect is gray/missing. It must not be rendered as near-zero (`ΔRNA +0.00`).
- An outlined `Missing` placeholder appears only when a cluster has neither a terminal-effect result nor any constructible path. A cluster with paths but an invalid terminal effect must be explicitly distinguished as missing/NaN.

Visual encoding:

- Orange center: KO TF.
- Purple arc blocks: intermediate TFs, with TF names written directly inside the blocks.
- Light gray arc blocks: peaks.
- Letters in peak blocks correspond to the annotation column on the right, which lists full peak coordinates (`chr:start-end`).
- Terminal effects and global card: light red indicates RNA upregulation, light blue downregulation, and light gray zero or near-zero effects. White outlined blocks indicate only the `Missing` state with neither a terminal-effect result nor retained paths.
- Edge colors indicate intrinsic regulatory direction only: red for Positive, blue for Negative, and gray for Uncertain. They do not indicate the direction of the final RNA response.

The former footer “terminal/card fill = actual RNA response” means that the outermost terminal-effect blocks and global card use saved actual model responses for their fills. Cluster terminal effects use fields selected from `perturbation_results.csv` by the effect adapter: endpoint log2FC first, then an explicit native RNA effect, then legacy `delta_rna`. They can therefore be native responses and cannot universally be called log2FC. The global card still uses the merged table's signed display log2FC. Colors express each response's positive/negative/near-zero/missing state; they do not imply identical units.

Radial edge widths and opacity remain fixed by design: ordinary straight edges have width `.78`, and curved direct paths crossing layers have width `.66`. They do not encode effect size or statistical confidence. Arrows indicate propagation direction along model paths. Edge colors depend on TF→peak weights and peak→gene correlation direction; in hybrid models, these filtering/direction metadata must not be treated as linear coefficients in the simulation. The Node key on the right retains full peak coordinates and node identities.

Distinguish two sources of path contributions. Legacy allocated results use `delta_r_raw_sum` (`delta_r_total` for hierarchy aggregation). If current temporal-link exports lack both columns and explicitly set `shared_peak_contribution=True`, saved `peak_input_contribution_mean` is used to rank paths and hierarchies only when `peak_input_unit=delta_log1p_rna_log1p_cp10k` and the response is valid. The former footer “Shared peak→RNA mean contributions … temporal links, not TF-allocated RNA effects” refers to this shared downstream peak→RNA mean contribution. Upstream TF→ATAC inputs do not substitute for RNA responses, and missing TF-specific allocation columns remain missing. A shared peak response may appear in multiple TF links. Pathway plots show candidate temporal associations; they must not be treated as additive TF-specific causal attributions of RNA effects, and path scores must not be treated as final gene endpoints.

Indirect paths require a complete `KO TF → peak A → intermediate TF → peak B → affected gene` chain within the same cluster; direct paths are `KO TF → peak → affected gene`. Path topology is no longer filtered by magnitude: any finite aggregated edge effect can participate in direct or indirect path construction. The only count limit is a maximum of 3 displayed paths per cluster, ranked by path contribution. Weak paths are therefore retained and are not classified as `No route` merely because their contributions are small.

## 5. Automatically selected positive- and negative-effect pathways

When no genes are specified, the specified-plot entry point in Section 4 instead outputs `pathway_topN.png` (default `pathway_top5.png`), automatically selecting valid genes by saved `|delta_rna_signed|`. Each gene still receives a separate radial panel. The four plot categories below independently select from their own candidate sets.

Alongside `pathway_specified.png` for specified genes, the script selects the target genes with the largest positive and negative effects for each of two pathway categories, automatically producing four plots. These use exactly the same radial cluster hierarchy, response colors, path selection, and global aggregation rules as Section 4. When all four candidate categories are available, the total should therefore be **5 pathway plots** (1 specified-gene plot + 4 automatically selected plots):

- `pathway_direct_top_positive.png`: the positive-effect gene with the largest `delta_rna_signed` among targets with a direct regulatory edge from the target TF.
- `pathway_direct_top_negative.png`: the negative-effect gene with the smallest `delta_rna_signed` among targets with a direct regulatory edge from the target TF.
- `pathway_indirect_top_positive.png`: the positive-effect gene with the largest `delta_rna_signed` among purely indirect targets.
- `pathway_indirect_top_negative.png`: the negative-effect gene with the smallest `delta_rna_signed` among purely indirect targets.

If a category has no candidate gene in the corresponding direction, its plot is skipped, so the actual total may be fewer than 5.

The `direct regulation` category requires at least one `target TF→peak→gene` edge to the final target gene. If that gene also has indirect paths, those are displayed as well.

The `purely indirect regulation` category requires no direct `target TF→peak→gene` edge but at least one complete indirect chain:

```text
Target TF → Intermediate TF → peak → Downstream target gene
```

The upstream and downstream chain must be verifiable within the same cluster. All propagation paths in purely indirect plots therefore pass through an intermediate TF and contain no direct regulatory edge from the target TF to the final target gene.
To keep figures uncluttered, full selection rules and visual encoding are documented in Section 4 and this section.

## 6. `tf_hierarchy.png`: TF regulatory hierarchy

This ring plot summarizes the regulatory hierarchy through which the current target TF KO propagates via intermediate TFs to downstream genes.

- Center: current target TF KO.
- Inner ring: the 5 intermediate TFs with the highest joint contributions.
- Outer ring: major downstream genes for each intermediate TF, up to 4 per TF.
- Red: positive displayed contribution.
- Blue: negative displayed contribution; the outer ring uses lighter corresponding colors.
- `*` after an outer-ring gene name: the gene is regulated by multiple intermediate TFs shown in the figure.

An intermediate TF's joint contribution is defined as:

```text
joint_contribution = sqrt(upstream_contribution × downstream_contribution)
```

Here:

- `upstream_contribution`: absolute aggregated contribution from the target TF to the intermediate TF.
- `downstream_contribution`: sum of that TF's absolute effects on all downstream genes.

Inner-ring sector widths are proportional to `joint_contribution`; `J` in TF labels is the joint contribution score. The geometric mean requires strong contributions on both upstream and downstream sides, preventing a large effect on only one side from dominating the ranking. Within each TF, outer-ring sectors are allocated according to downstream genes' absolute effects.

The default display counts above are controlled by `TOP_TF_N` and `TOP_DOWNSTREAM_N`; downstream display also uses the `MIN_EFFECT_TF` threshold. Intermediate TFs with no downstream genes passing the threshold are omitted. Contribution sources follow Section 4: in temporal-link plots, J, sector widths, ring-label values, and red/blue colors summarize shared downstream peak→RNA responses. They are neither TF-specific RNA allocations nor complete endpoint log2FC values for outer-ring genes. The `*` on repeated targets indicates association with multiple displayed TFs; these shared contributions must not be summed as independent contributions. This paragraph and Section 4 replace the former footer notes about shared responses, native units, and the absence of TF-specific allocation.

## 7. `umap_cluster_bin_effect.png`: Relative KO effect by cluster/bin

This plot uses UMAP coordinates from `rna_clustered.h5ad` and assigns each modeled cell a KO effect intensity based on its cluster and pseudotime bin. Cells within the same `(cluster, bin)` share one intensity value.

**Each point represents a cell, not a bin.**

- Point position: the cell's UMAP coordinates in `rna_clustered.h5ad`.
- Point color: relative KO effect for the cell's `(cluster, bin)`.
- All cells in the same cluster and bin have the same color value.
- This is therefore a **cell-level UMAP display with cluster-bin-level effect coloring**, not an independently computed KO effect for every cell.

In path-level records, `bin_min` identifies the bin where an effect is first triggered. Because effects persist in later bins during perturbation propagation, each bin accumulates contributions from all paths with `bin_min ≤ current bin`. Net effects are first summed within each downstream gene, then:

```text
cluster_bin_effect = sum_over_genes(abs(net_gene_effect))
relative_effect = cluster_bin_effect / max(cluster_bin_effect)
```

- Light gray: unmodeled cells or no available perturbation effect for that cluster/bin (including missing/NaN); gray does not mean zero effect.
- Light yellow to dark red: weaker to stronger relative KO effects across all clusters/bins.
- `Cl n`: cell cluster labels from the original h5ad.

This score measures total downstream transcriptome perturbation intensity and does not retain upregulation/downregulation direction. It supports comparisons of relative KO responses across clusters and pseudotime stages and must not be interpreted as an independent single-cell effect estimate.

## 8. `umap_cluster_bin_mean_effect.png`: Mean KO effect by cluster/bin

This plot is produced alongside Section 7 using exactly the same UMAP coordinates, cell coloring scheme, and palette, but a different definition of intensity.

Section 7 uses **cumulative total burden** (more affected genes increase intensity). This plot uses the **mean absolute effect per affected gene**, removing amplification caused by a larger number of affected genes:

```text
mean_abs_downstream_effect = downstream_abs_sum / n_affected_genes
mean_relative_effect = mean_abs_downstream_effect / max(mean_abs_downstream_effect)
```

- `downstream_abs_sum`: sum of absolute net effects across all downstream genes in that cluster/bin.
- `n_affected_genes`: number of downstream genes in that cluster/bin with `|net effect| > 1e-10`; when this count is 0, the mean effect is set to 0.
- `mean_relative_effect`: normalized to `[0, 1]` across all clusters/bins.

The two plots can rank intensities differently: some clusters score higher in Section 7 because they cover more genes, while their mean per-gene effect is less pronounced, and vice versa. Use Section 7 to compare total perturbation coverage and this section to compare mean per-gene perturbation intensity. This score also does not retain upregulation/downregulation direction.

## 9. `atac_accessibility_chromosome.png`: Peak accessibility changes by chromosome

This plot shows genome-wide ATAC peak opening/closing after perturbation. Data come from `<RESULT_DIR>/atac_changes_merged.csv` (columns: `peak_id, chr, start, end, delta_accessibility_signed, delta_accessibility, n_clusters_observed, z_score, is_significant`), using all peaks without significance filtering.

- X-axis: chromosomes in natural order `chr1…chr22, chrX, chrY`, followed by noncanonical chromosomes.
- Y-axis: genomic peak coordinates (bp).
- Red: increased accessibility, or opening.
- Blue: decreased accessibility, or closing.
- Color intensity: darker for larger `|delta_accessibility_signed|`.
- Neutral colors: changes close to 0.

Because individual peak spans (roughly hundreds of bp) are invisible relative to chromosome lengths (roughly 250 Mb), the plot uses **fixed-height bars**. All bars have the same height, while their positions remain at the genomic midpoints of the peaks. Color intensity uses a square-root mapping:

```text
effect_color = sign(delta_accessibility_signed) × sqrt(|delta_accessibility_signed| / max|delta_accessibility_signed|)
```

This mapping keeps the many weak effects visible (approximately 51.6% of peaks in D3 have `|Δ| < 0.01`); the legend identifies the nonlinear √ scale. If `atac_changes_merged.csv` is missing or lacks required columns, this plot issues a warning and is skipped without affecting other plots.

## 10. `atac_distribution.png`: ATAC accessibility effect distribution and classification

This is the ATAC counterpart to the RNA distribution plot in Section 2, with two panels. **The unit of analysis is a peak, not a gene.**

### Left: Effect distribution

- X-axis: signed `delta_accessibility_signed` (positive for opening, negative for closing).
- Y-axis: number of peaks in each interval.
- Red bars: positive effects, or opening.
- Blue bars: negative effects, or closing.
- Solid black line: zero effect.
- Dashed red line: median accessibility change across all peaks.

### Right: Effect classification

Classification uses the change-magnitude threshold `|delta_accessibility_signed| > 0.01`:

- Open: `Δ > 0.01`.
- Close: `Δ < -0.01`.
- Stable: `-0.01 ≤ Δ ≤ 0.01`.

This classification likewise uses magnitude thresholds only, not a statistical significance test. The pipeline separately determines significance using `atac_significance_zscore` (default `2.0`), which differs from this plot's threshold; the significant set in D3 contains only approximately 257 peaks. If `atac_changes_merged.csv` is missing or lacks required columns, this plot is skipped together with Section 9.

## 11. Projection plots (when projection results are available)

`projection_overview.png` retains the full UMAP embedding. Colors encode predicted projected pseudotime minus source pseudotime. The former footer “Clean UMAP view” only described the plotting view; colors do not represent RNA log2FC, and UMAP coordinates themselves do not represent time.

`projection_pseudotime_ordered.png` arranges cells by source context, with source pseudotime on the x-axis. An arrow marks the direction of increasing source pseudotime. It uses ordered trajectory coordinates and must not be interpreted as UMAP geometry. Colors again represent predicted projected pseudotime minus source pseudotime. Both plots retain their axis titles, color bars, and original titles.

## Interpretation notes

1. All results are model-predicted in silico KO effects and are not equivalent to experimentally validated causal conclusions.
2. `log2FC` represents predicted expression change; `R²` represents model fit quality. They have different meanings.
3. Red/blue colors indicate regulatory direction, not statistical significance.
4. Pathway topology is no longer filtered by magnitude, but at most 3 paths per cluster are displayed, ranked by contribution. Other undisplayed paths may still contribute to the final effect.
5. Gray in the cross-cluster heatmap means missing, unobserved, or NaN and must not be treated as zero. White indicates an observed effect close to 0.
6. In pathway plots, `Missing` means neither a terminal-effect result nor retained paths are available. If paths exist but the terminal effect is missing or NaN, show a gray/missing state rather than rendering it as near-zero.
7. In ATAC plots, red/blue means **accessibility opening (positive)/closing (negative)**. This corresponds to upregulation/downregulation in RNA plots but uses different units, so numerical magnitudes cannot be directly compared.
8. Colors in `atac_accessibility_chromosome.png` use a nonlinear square-root mapping. Bars have fixed height, and only their positions correspond to genomic coordinates; bar height does not represent effect size.
9. The ATAC classification threshold `|Δ| > 0.01` differs from the pipeline's `atac_significance_zscore` significance criterion and must not be used interchangeably with it.
