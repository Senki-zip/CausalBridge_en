#!/usr/bin/env Rscript
# =============================================================================
# Perturbation Result Visualization
#   - Volcano plot: per-gene t-test for delta_rna significance
#   - Pathway schematic: TF -> Peak -> Gene causal chain for specified genes
#
# Usage:
#   Rscript visualize_perturbation.R [result_dir] [--genes GENE1,GENE2,...]
#
# Examples:
#   Rscript visualize_perturbation.R
#   Rscript visualize_perturbation.R /path/to/results/ --genes RASSF2,CBWD2,CDH1
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(ggrepel)
  library(patchwork)
  library(igraph)
})

# The UMAP panel is optional. Keep the core CSV-based visualizations runnable
# when the R anndata bridge is not installed.
HAS_ANNDATA <- requireNamespace("anndata", quietly = TRUE)

# ==============================================================================
# Configuration — modify these variables directly before running
# ==============================================================================

# Default result directory (can be overridden by command-line argument)
RESULT_DIR <- Sys.getenv(
  "CAUSALBRIDGE_RESULT_DIR",
  unset = "/home/huangtao/Desktop/huangchengqi/code_1/CausalBridge/results/test_2/D3/"
)

# Genes to draw pathway schematics for.
#   NULL         → auto-select top 5 by |delta_rna_signed|
#   c("A","B")   → draw specified genes only
# Can be overridden by:  --genes GENE1,GENE2,...
pathway_genes_env <- Sys.getenv("CAUSALBRIDGE_PATHWAY_GENES", unset = "")
PATHWAY_GENES <- if (identical(pathway_genes_env, "__AUTO__")) {
  NULL
} else if (nzchar(pathway_genes_env)) {
  trimws(strsplit(pathway_genes_env, ",")[[1]])
} else {
  NULL
  }

# ==============================================================================
# Parse command-line arguments (override config above)
# ==============================================================================
args <- commandArgs(trailingOnly = TRUE)

specified_genes <- PATHWAY_GENES
positional <- character(0)
i <- 1
while (i <= length(args)) {
  if (args[i] == "--genes" && i < length(args)) {
    specified_genes <- strsplit(args[i + 1], ",")[[1]]
    specified_genes <- trimws(specified_genes)
    i <- i + 2
  } else if (!startsWith(args[i], "--")) {
    positional <- c(positional, args[i])
    i <- i + 1
  } else {
    i <- i + 1
  }
}

if (length(positional) > 0) {
  RESULT_DIR <- positional[1]
}

OUTPUT_DIR <- Sys.getenv(
  "CAUSALBRIDGE_OUTPUT_DIR",
  unset = file.path(RESULT_DIR, "figures_R")
)
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# Visual style — presentation-only settings
# ==============================================================================

COLORS <- list(
  up       = "#E74C3C",      # Dark red, activation
  down     = "#3498DB",      # Dark blue, inhibition
  stable   = "#BDC3C7",      # Gray, stable
  ink      = "#2C3E50",      # Dark text
  muted    = "#7F8C8D",      # Light text
  grid     = "#ECF0F1",      # Grid lines
  panel    = "#FFFFFF",      # Panel background
  peak     = "#F8F9FA",      # Peak node background
  target   = "#F39C12",      # Target node (orange)
  target_2 = "#E67E22",      # Target border
  tf       = "#9B59B6",      # TF node (purple)
  tf_bg    = "#F5EEF8",      # TF node background
  gene     = "#27AE60",      # Gene node (green)
  gene_bg  = "#EAFAF1",      # Gene node background
  edge_pos = "#E74C3C",      # Positive edge (red)
  edge_neg = "#3498DB"       # Negative edge (blue)
)

theme_pub <- function(base_size = 14) {
  theme_minimal(base_size = base_size) +
    theme(
      text = element_text(color = COLORS$ink),
      plot.title = element_text(face = "bold", color = COLORS$ink, hjust = 0),
      axis.title = element_text(face = "bold", color = COLORS$ink),
      axis.text = element_text(color = COLORS$ink),
      panel.grid.major = element_line(color = COLORS$grid, linewidth = 0.25),
      panel.grid.minor = element_blank(),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      legend.background = element_rect(fill = "white", color = NA),
      legend.key = element_rect(fill = "white", color = NA),
      plot.margin = margin(10, 12, 10, 12)
    )
}

cat(sprintf("Data dir:    %s\n", RESULT_DIR))
cat(sprintf("Output dir:  %s\n", OUTPUT_DIR))
if (!is.null(specified_genes)) {
  cat(sprintf("Genes:       %s\n", paste(specified_genes, collapse = ", ")))
}

# ==============================================================================
# Load data
# ==============================================================================
cat("\n--- Loading data ---\n")

pert_res <- read.csv(file.path(RESULT_DIR, "perturbation_results.csv"))
cat(sprintf("perturbation_results: %d rows, %d unique genes\n",
            nrow(pert_res), length(unique(pert_res$affected_gene))))

pert_merged <- read.csv(file.path(RESULT_DIR, "perturbation_merged.csv"))
cat(sprintf("perturbation_merged: %d rows\n", nrow(pert_merged)))

pathways <- read.csv(file.path(RESULT_DIR, "perturbation_pathways.csv"))
cat(sprintf("pathways: %d rows\n", nrow(pathways)))

# Load TF→peak weights to get lag information (per TF, per peak, per cluster)
tf_weights <- read.csv(file.path(RESULT_DIR, "tf_peak_weights.csv"))
cat(sprintf("tf_peak_weights: %d rows\n", nrow(tf_weights)))

# Load Peak→Gene causal edges (atac_coef per peak, gene, cluster)
peak_gene_edges <- read.csv(file.path(RESULT_DIR, "causal_peak_gene_edges.csv"))
cat(sprintf("peak_gene_edges: %d rows\n", nrow(peak_gene_edges)))

# Join TF→Peak weight into pathways (match on tf_name=tf_gene, peak_id, cluster)
# Dedup first to avoid many-to-many joins
tf_weights_dedup <- tf_weights %>%
  group_by(tf_gene, peak_id, cluster) %>%
  summarise(tf_peak_weight = mean(weight, na.rm = TRUE), .groups = "drop") %>%
  rename(tf_name = tf_gene)

pathways <- pathways %>%
  left_join(tf_weights_dedup, by = c("tf_name", "peak_id", "cluster"))

# Join Peak→Gene atac_coef into pathways (match on peak_id, affected_gene=gene, cluster)
pg_dedup <- peak_gene_edges %>%
  group_by(peak_id, gene, cluster) %>%
  summarise(
    peak_gene_coef = mean(atac_coef, na.rm = TRUE),
    peak_gene_pearson_r = if (all(is.na(pearson_r))) NA_real_ else
      mean(pearson_r, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  rename(affected_gene = gene)

pathways <- pathways %>%
  left_join(pg_dedup, by = c("peak_id", "affected_gene", "cluster"))

cat(sprintf("Joined tf_peak_weight + peak_gene_coef + peak_gene_pearson_r into pathways\n"))

# Join tf_lag from tf_weights into pathways (match on tf_name=tf_gene, peak_id, cluster)
if ("tf_lag_first" %in% colnames(pathways)) {
  # New pipeline output already has tf_lag
  cat("Using tf_lag from perturbation_pathways.csv\n")
  # Rename for downstream compatibility
  pathways$tf_lag <- pathways$tf_lag_first
} else {
  # Backward compat: join from tf_peak_weights
  lag_info <- tf_weights %>%
    select(tf_gene, peak_id, tf_lag, cluster) %>%
    rename(tf_name = tf_gene) %>%
    mutate(cluster = as.integer(cluster))

  pathways <- pathways %>%
    left_join(lag_info, by = c("tf_name", "peak_id", "cluster"),
              relationship = "many-to-many")

  n_with_lag <- sum(!is.na(pathways$tf_lag))
  cat(sprintf("Joined tf_lag: %d/%d rows have lag info\n", n_with_lag, nrow(pathways)))
}

# Auto-detect one or more jointly perturbed TFs. Combined-KO outputs encode the
# intervention as a comma- or plus-delimited string (for example KLF1,GATA1),
# while pathway tf_name values remain individual gene symbols.
target_gene_raw <- unique(pert_merged$target_gene)[1]
target_gene_names <- trimws(unlist(strsplit(target_gene_raw, "[,+]")))
target_gene_names <- unique(target_gene_names[nzchar(target_gene_names)])
target_gene_name <- paste(target_gene_names, collapse = " + ")

if (identical(Sys.getenv("CAUSALBRIDGE_REQUIRE_MULTI_TF"), "1") &&
    length(target_gene_names) < 2) {
  stop("The joint KO visualization script requires target_gene to contain at least two TFs")
}

cat(sprintf("Target gene%s: %s\n",
            if (length(target_gene_names) > 1) "s" else "",
            target_gene_name))

# ==============================================================================
# 1. Effect–fit quality plot — x=log2FC, y=mean R² (credibility)
# ==============================================================================
cat("\n--- Computing volcano data ---\n")

# Per-gene mean delta_rna from perturbation_results (per-cluster average)
delta_df <- pert_res %>%
  group_by(affected_gene) %>%
  summarise(
    mean_delta = mean(delta_rna),
    n_clusters = n(),
    .groups    = "drop"
  )

# Per-gene mean R² from pathway edges (ATAC→RNA transfer function fit quality)
r2_df <- pathways %>%
  group_by(affected_gene) %>%
  summarise(
    mean_r2 = mean(r2_score_first),
    n_edges  = n(),
    .groups  = "drop"
  )

# Join
volcano_df <- inner_join(delta_df, r2_df, by = "affected_gene") %>%
  mutate(
    change = case_when(
      mean_delta > 1   ~ "Up",
      mean_delta < -1  ~ "Down",
      TRUE             ~ "Stable"
    ),
    change = factor(change, levels = c("Up", "Down", "Stable")),
    # Joint label priority: large perturbation with strong model support.
    label_score = abs(mean_delta) * pmax(mean_r2, 0),
    label = ifelse(abs(mean_delta) > 1 & rank(-label_score) <= 15,
                   affected_gene, "")
  )

n_up   <- sum(volcano_df$change == "Up")
n_down <- sum(volcano_df$change == "Down")
cat(sprintf("Genes with R² data: %d\n", nrow(volcano_df)))
cat(sprintf("Large-effect genes (|log2FC|>1): Up=%d, Down=%d\n", n_up, n_down))
cat(sprintf("Mean R² range: %.3f – %.3f, median: %.3f\n",
            min(volcano_df$mean_r2), max(volcano_df$mean_r2),
            median(volcano_df$mean_r2)))

# Effect–fit quality plot: x = log2FC, y = mean R²
x_max <- max(abs(volcano_df$mean_delta), na.rm = TRUE) * 1.1

p_volcano <- ggplot(volcano_df, aes(x = mean_delta, y = mean_r2)) +
  geom_point(aes(color = change), alpha = 0.72, size = 1.9, stroke = 0) +
  scale_color_manual(
    values = c("Up" = COLORS$up, "Down" = COLORS$down, "Stable" = COLORS$stable),
    name   = NULL
  ) +
  geom_label_repel(
    data = subset(volcano_df, label != ""),
    aes(label = label),
    size = 4.2, fontface = "bold.italic", max.overlaps = 20,
    box.padding = 0.55, point.padding = 0.35, label.padding = unit(0.12, "lines"),
    min.segment.length = 0,
    segment.size = 0.28, segment.alpha = 0.55,
    label.size = NA, fill = scales::alpha("white", 0.82),
    color = COLORS$ink
  ) +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = COLORS$muted,
             linewidth = 0.4) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = COLORS$muted,
             linewidth = 0.4) +
  scale_x_continuous(limits = c(-x_max, x_max)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = paste0(target_gene_name, " KO: Perturbation Effect vs Model Fit"),
    x = expression(Delta * " RNA (log"[2] * " fold change)"),
    y = expression("Mean " * italic(R)^2 * " (ATAC→RNA fit quality)")
  ) +
  theme_pub(base_size = 18) +
    theme(
      plot.title        = element_text(size = 18, hjust = 0.5),
      legend.position   = c(0.86, 0.18),
    legend.background = element_rect(fill = scales::alpha("white", 0.9), color = COLORS$grid),
    legend.key.size   = unit(0.6, "cm"),
    legend.text       = element_text(size = 13, color = COLORS$ink),
    panel.grid.major  = element_line(color = COLORS$grid, linewidth = 0.22)
  )

ggsave(
  file.path(OUTPUT_DIR, "volcano.png"),
  p_volcano, width = 12, height = 9, dpi = 320, bg = "white"
)
cat(sprintf("[OK] Effect–fit plot saved: %s\n", file.path(OUTPUT_DIR, "volcano.png")))

# ==============================================================================
# 2. Pathway schematics for specified genes (or top N if not specified)
# ==============================================================================
cat("\n--- Drawing pathway schematics ---\n")

# Determine genes to plot
if (!is.null(specified_genes)) {
  # Validate specified genes
  valid_genes <- specified_genes[specified_genes %in% pert_merged$affected_gene]
  invalid_genes <- setdiff(specified_genes, valid_genes)
  if (length(invalid_genes) > 0) {
    cat(sprintf("WARNING: genes not found in data: %s\n",
                paste(invalid_genes, collapse = ", ")))
  }
  if (length(valid_genes) == 0) {
    stop("None of the specified genes were found in perturbation_merged.csv")
  }
  # Order by |delta_rna_signed|
  gene_order <- pert_merged %>%
    filter(affected_gene %in% valid_genes) %>%
    arrange(desc(abs(delta_rna_signed)))
  pathway_genes <- gene_order$affected_gene
  cat(sprintf("Specified genes: %s\n", paste(pathway_genes, collapse = ", ")))
} else {
  # Default: top 5 by |delta_rna_signed|
  pathway_genes <- pert_merged %>%
    slice_max(order_by = abs(delta_rna_signed), n = 5) %>%
    pull(affected_gene)
  cat(sprintf("Top 5 genes (auto): %s\n", paste(pathway_genes, collapse = ", ")))
}

# ---- Single pathway plot function (Target → Intermediate TF → Peak → Gene) ----
plot_single_pathway <- function(gene_name, pw_df, merged_df,
                                show_main_title = TRUE, cluster_filter = NULL,
                                cluster_effect = NULL, panel_title = NULL) {
  target_gene_raw <- unique(merged_df$target_gene)[1]
  target_gene_names <- unique(trimws(unlist(strsplit(target_gene_raw, "[,+]"))))
  target_gene_names <- target_gene_names[nzchar(target_gene_names)]
  target_gene_name <- paste(target_gene_names, collapse = " + ")
  delta_total <- merged_df$delta_rna_signed[merged_df$affected_gene == gene_name]
  if (length(delta_total) == 0) delta_total <- 0
  # Cluster panels use the observed cluster perturbation; the unfiltered plot
  # continues to use the merged response.
  displayed_gene_delta <- if (!is.null(cluster_filter) &&
                              length(cluster_effect) && is.finite(cluster_effect[1])) {
    cluster_effect[1]
  } else delta_total[1]

  # All pathway edges for this affected gene
  gene_pw <- pw_df %>% filter(affected_gene == !!gene_name)
  if (!is.null(cluster_filter)) {
    gene_pw <- gene_pw %>% filter(as.character(cluster) == as.character(cluster_filter))
  }

  # Exclude clusters where the affected gene had zero perturbation effect
  # (delta_rna_log2fc == 0 means the gene wasn't perturbed in that cluster,
  #  so those edges are noise — they don't contribute to the actual effect)
  gene_pw <- gene_pw %>% filter(abs(delta_rna_log2fc) > 1e-10)

  # Will be recomputed after aggregation from displayed edges
  gene_pathway_sum <- 0

  if (nrow(gene_pw) == 0) {
    p <- ggplot() +
      annotate("text", x = 0.5, y = 0.5, size = 8.5, color = "grey40",
               label = paste(gene_name, "\n(no pathway data)")) +
      theme_void()
    return(p)
  }

  shorten_peak <- function(pid) NULL  # Chromatin coordinates are no longer shown; retain the function signature for compatibility

  # --- Split edges by membership in the joint-KO target set ---
  direct_pw <- gene_pw %>% filter(tf_name %in% target_gene_names)
  indirect_pw <- gene_pw %>% filter(!tf_name %in% target_gene_names)

  # Identify intermediate TFs (non-target TFs that regulate this gene)
  intermediate_tfs <- unique(indirect_pw$tf_name)

  # For each intermediate TF, find target→TF connections
  # (edges where target binds peaks and the affected_gene is the intermediate TF)
  target_to_tf_pw <- pw_df %>%
    filter(tf_name %in% target_gene_names,
           affected_gene %in% intermediate_tfs,
           !affected_gene %in% target_gene_names)
  if (!is.null(cluster_filter)) {
    target_to_tf_pw <- target_to_tf_pw %>%
      filter(as.character(cluster) == as.character(cluster_filter))
  }

  # --- Cluster consistency filtering ---
  # For each intermediate TF, retain only downstream edges whose cluster
  # matches the cluster of an upstream regulated edge.
  if (nrow(target_to_tf_pw) > 0 && "cluster" %in% colnames(target_to_tf_pw)) {
    # Upstream: (TF, cluster) pair set
    upstream_pairs <- target_to_tf_pw %>%
      select(affected_gene, cluster) %>%
      distinct() %>%
      rename(tf_name = affected_gene)

    # Retain only downstream edges whose cluster matches an upstream cluster.
    if (nrow(indirect_pw) > 0 && "cluster" %in% colnames(indirect_pw)) {
      indirect_pw <- indirect_pw %>%
        semi_join(upstream_pairs, by = c("tf_name", "cluster"))
    }
  }

  # =========================================================================
  # Build 5-row layout (vertical, top to bottom):
  #   Target
  #   Peak_A
  #   Intermediate TF
  #   Peak_B
  #   Affected Gene
  # =========================================================================
  y_target   <- 0.93
  y_peak_a   <- 0.72
  y_tf_mid   <- 0.50
  y_peak_b   <- 0.30
  y_gene     <- 0.10

  x_spread <- function(n) {
    if (n <= 1) return(0.5)
    seq(0.25, 0.95, length.out = n)
  }

  # Unweighted cross-cluster coefficient means can reverse a displayed edge
  # even when the clusters with the largest pathway contributions agree.
  weighted_reg_direction <- function(coefficient, contribution) {
    usable <- is.finite(coefficient) & is.finite(contribution)
    if (!any(usable)) return(NA_real_)
    weights <- abs(contribution[usable])
    if (sum(weights) <= 0) return(NA_real_)
    direction <- weighted.mean(coefficient[usable], w = weights)
    if (!is.finite(direction) || abs(direction) <= 1e-8) NA_real_ else direction
  }

  weighted_reg_weight <- function(coefficient, contribution) {
    usable <- is.finite(coefficient) & is.finite(contribution)
    if (!any(usable)) 0 else sum(abs(contribution[usable]))
  }

  # --- Aggregate direct edges: Target → Peak_B → Gene ---
  #   Aggregate the effects of the same peak across all clusters without
  #   manually filtering by positive or negative direction.
  if (nrow(direct_pw) > 0) {
    direct_agg <- direct_pw %>%
      group_by(peak_id) %>%
      summarise(
        total_effect = sum(delta_r_raw_sum, na.rm = TRUE),
        tf_peak_weight = weighted_reg_direction(tf_peak_weight, delta_r_raw_sum),
        tf_peak_direction_weight = weighted_reg_weight(tf_peak_weight, delta_r_raw_sum),
        peak_gene_pearson_r = weighted_reg_direction(peak_gene_pearson_r, delta_r_raw_sum),
        peak_gene_direction_weight = weighted_reg_weight(peak_gene_pearson_r, delta_r_raw_sum),
        r2_mean = if ("r2_score_first" %in% colnames(direct_pw))
                  mean(r2_score_first) else NA_real_,
        tf_lag_mean = if ("tf_lag" %in% colnames(direct_pw))
                      round(mean(tf_lag, na.rm = TRUE), 1) else NA_real_,
        clusters = if ("cluster" %in% colnames(direct_pw))
                   paste(sort(unique(cluster)), collapse = ",") else "",
        .groups = "drop"
      ) %>%
      arrange(desc(abs(total_effect)))
  } else {
    direct_agg <- data.frame(
      peak_id = character(0), total_effect = numeric(0),
      r2_mean = numeric(0), tf_lag_mean = numeric(0),
      clusters = character(0), stringsAsFactors = FALSE
    )
  }

  # --- Aggregate indirect edges: TF → Peak_B → Gene ---
  #   Aggregate the effects of the same (TF, peak) across all clusters without
  #   manually filtering by positive or negative direction.
  if (nrow(indirect_pw) > 0) {
    indirect_agg <- indirect_pw %>%
      group_by(tf_name, peak_id) %>%
      summarise(
        total_effect = sum(delta_r_raw_sum, na.rm = TRUE),
        tf_peak_weight = weighted_reg_direction(tf_peak_weight, delta_r_raw_sum),
        tf_peak_direction_weight = weighted_reg_weight(tf_peak_weight, delta_r_raw_sum),
        peak_gene_pearson_r = weighted_reg_direction(peak_gene_pearson_r, delta_r_raw_sum),
        peak_gene_direction_weight = weighted_reg_weight(peak_gene_pearson_r, delta_r_raw_sum),
        r2_mean = if ("r2_score_first" %in% colnames(indirect_pw))
                  mean(r2_score_first) else NA_real_,
        tf_lag_mean = if ("tf_lag" %in% colnames(indirect_pw))
                      round(mean(tf_lag, na.rm = TRUE), 1) else NA_real_,
        clusters = if ("cluster" %in% colnames(indirect_pw))
                   paste(sort(unique(cluster)), collapse = ",") else "",
        .groups = "drop"
      ) %>%
      ungroup() %>%
      arrange(desc(abs(total_effect)))
  } else {
    indirect_agg <- data.frame(
      tf_name = character(0), peak_id = character(0),
      total_effect = numeric(0), r2_mean = numeric(0),
      tf_lag_mean = numeric(0), clusters = character(0),
      stringsAsFactors = FALSE
    )
  }

  # --- Aggregate target→TF edges: Target → Peak_A → Intermediate TF ---
  #   Aggregate the effects of the same (TF, peak) across all clusters without
  #   manually filtering by positive or negative direction.
  if (nrow(target_to_tf_pw) > 0 && length(intermediate_tfs) > 0) {
    t2tf_agg <- target_to_tf_pw %>%
      group_by(affected_gene, peak_id) %>%
      summarise(
        total_effect = sum(delta_r_raw_sum, na.rm = TRUE),
        tf_peak_weight = weighted_reg_direction(tf_peak_weight, delta_r_raw_sum),
        tf_peak_direction_weight = weighted_reg_weight(tf_peak_weight, delta_r_raw_sum),
        peak_gene_pearson_r = weighted_reg_direction(peak_gene_pearson_r, delta_r_raw_sum),
        peak_gene_direction_weight = weighted_reg_weight(peak_gene_pearson_r, delta_r_raw_sum),
        clusters = if ("cluster" %in% colnames(target_to_tf_pw))
                   paste(sort(unique(cluster)), collapse = ",") else "",
        .groups = "drop"
      ) %>%
      ungroup()
  } else {
    t2tf_agg <- data.frame(
      affected_gene = character(0), peak_id = character(0),
      total_effect = numeric(0), stringsAsFactors = FALSE
    )
  }

  # --- Determine which nodes to show (limit for readability) ---
  # Only keep intermediate TFs that have BOTH:
  #   (a) a finite regulatory effect on this gene
  #   (b) verified target→TF connection (target regulates this TF)
  valid_tfs <- intersect(unique(indirect_agg$tf_name), unique(t2tf_agg$affected_gene))
  # Rank TFs by their joint upstream and downstream contribution.  The
  # geometric mean favors TFs that are strong on both target→TF and TF→gene,
  # instead of allowing one large side to dominate the ordering.
  tf_joint_rank <- full_join(
    t2tf_agg %>%
      filter(affected_gene %in% valid_tfs) %>%
      group_by(tf_name = affected_gene) %>%
      summarise(upstream_contribution = sum(abs(total_effect)), .groups = "drop"),
    indirect_agg %>%
      filter(tf_name %in% valid_tfs) %>%
      group_by(tf_name) %>%
      summarise(downstream_contribution = sum(abs(total_effect)), .groups = "drop"),
    by = "tf_name"
  ) %>%
    mutate(
      upstream_contribution = coalesce(upstream_contribution, 0),
      downstream_contribution = coalesce(downstream_contribution, 0),
      joint_contribution = sqrt(upstream_contribution * downstream_contribution)
    ) %>%
    arrange(desc(joint_contribution), tf_name)
  # Keep at most three indirect routes so complete paths remain legible.
  top_tfs <- head(tf_joint_rank$tf_name, 3)

  # Top peaks from direct edges (by absolute effect magnitude)
  direct_peaks <- direct_agg %>%
    slice_max(order_by = abs(total_effect), n = 3, with_ties = FALSE) %>%
    pull(peak_id)

  # Top peaks from indirect edges (take the top 1 for each TF to cover all TFs)
  indirect_peaks <- indirect_agg %>%
    filter(tf_name %in% top_tfs) %>%
    group_by(tf_name) %>%
    slice_max(order_by = abs(total_effect), n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    pull(peak_id) %>% unique()

  # Top peaks from target→TF edges (take the top 1 for each TF so every TF has
  # at least one upstream edge)
  t2tf_peaks <- t2tf_agg %>%
    filter(affected_gene %in% top_tfs) %>%
    group_by(affected_gene) %>%
    slice_max(order_by = abs(total_effect), n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    pull(peak_id) %>% unique()

  all_peak_b <- unique(c(direct_peaks, indirect_peaks))
  all_peak_a <- t2tf_peaks

  # If no intermediate TFs, use simplified 3-column layout
  has_indirect <- length(top_tfs) > 0
  has_direct   <- nrow(direct_agg) > 0

  if (!has_indirect && !has_direct) {
    p <- ggplot() +
      annotate("text", x = 0.5, y = 0.5, size = 8.5, color = "grey40",
               label = paste(gene_name, "\n(no valid edges)")) +
      theme_void()
    return(p)
  }

  # --- Node positions (vertical layout) ---
  # Target and gene stay at fixed positions; intermediate nodes spread
  # horizontally.
  target_x <- 0.55
  gene_x   <- 0.55

  if (has_direct && has_indirect) {
    n_tf <- length(top_tfs)

    # Separate lanes: indirect routes on the left, direct routes on the right.
    n_dir <- length(direct_peaks)
    direct_pk_x <- setNames(
      seq(0.66, 0.90, length.out = max(n_dir, 1)),
      direct_peaks
    )

    # Give intermediate-TF labels enough horizontal breathing room while
    # keeping the direct-peak lane clear. This is intentionally wider than
    # the old 0.20–0.46 lane for three long pathway labels.
    tf_x_vals <- setNames(seq(0.14, 0.54, length.out = max(n_tf, 1)), top_tfs)

    # Peak_A follows the sector of its connected TF.
    peak_a_x_vals <- sapply(all_peak_a, function(pk) {
      tfs <- unique(t2tf_agg$affected_gene[t2tf_agg$peak_id == pk])
      tfs <- intersect(tfs, names(tf_x_vals))
      if (length(tfs) > 0) mean(tf_x_vals[tfs]) else 0.5
    })

    # Indirect Peak_B follows the sector of its connected TF.
    indirect_pk_b_ids <- setdiff(all_peak_b, direct_peaks)
    indirect_pk_b_x <- sapply(indirect_pk_b_ids, function(pk) {
      tfs <- unique(indirect_agg$tf_name[indirect_agg$peak_id == pk])
      tfs <- intersect(tfs, names(tf_x_vals))
      if (length(tfs) > 0) mean(tf_x_vals[tfs]) else 0.34
    })

    tf_x     <- tf_x_vals
    peak_a_x <- peak_a_x_vals
    peak_b_x <- unlist(c(direct_pk_x, indirect_pk_b_x), use.names = TRUE)

  } else if (has_direct && !has_indirect) {
    # --- Direct only: spread evenly across the middle ---
    tf_x     <- numeric(0)
    peak_a_x <- numeric(0)
    peak_b_x <- setNames(
      seq(0.30, 0.92, length.out = max(length(all_peak_b), 1)),
      all_peak_b
    )

  } else {
    # --- Indirect only: establish vertical channels by TF to avoid crossings
    #     caused by independent sorting. ---
    tf_positions <- if (length(top_tfs) == 1) 0.55 else
      seq(0.22, 0.78, length.out = length(top_tfs))
    tf_x <- setNames(tf_positions, top_tfs)

    # Upstream peaks follow the intermediate TFs they regulate; shared peaks
    # are placed at the midpoint of the related TFs.
    peak_a_x <- setNames(sapply(all_peak_a, function(pk) {
      connected_tfs <- t2tf_agg$affected_gene[t2tf_agg$peak_id == pk]
      connected_tfs <- intersect(unique(connected_tfs), names(tf_x))
      if (length(connected_tfs) > 0) mean(tf_x[connected_tfs]) else 0.55
    }), all_peak_a)

    # Downstream peaks likewise follow their source TF, keeping TF→peak paths
    # essentially vertical.
    peak_b_x <- setNames(sapply(all_peak_b, function(pk) {
      connected_tfs <- indirect_agg$tf_name[indirect_agg$peak_id == pk]
      connected_tfs <- intersect(unique(connected_tfs), names(tf_x))
      if (length(connected_tfs) > 0) mean(tf_x[connected_tfs]) else 0.55
    }), all_peak_b)
  }

  # --- Per-node net effect for up/down arrow annotation ---
  arrow_up   <- " ▲"
  arrow_down <- " ▼"

  # Target KO: always down (loss of function)
  target_arrow <- arrow_down

  # Keep the displayed contribution summary consistent with the subset of edges
  # drawn below. This is a pathway magnitude, not the source of node direction.
  direct_display <- direct_agg %>%
    filter(peak_id %in% direct_peaks)
  indirect_display <- indirect_agg %>%
    filter(tf_name %in% top_tfs, peak_id %in% indirect_peaks)
  gene_pathway_sum <- sum(direct_display$total_effect, na.rm = TRUE) +
    sum(indirect_display$total_effect, na.rm = TRUE)
  # Node direction is the final merged RNA response, not the truncated pathway
  # subset. The same value is printed in the gene log2FC label below.
  gene_arrow <- if (is.finite(displayed_gene_delta) && displayed_gene_delta > 1e-10) arrow_up else
    if (is.finite(displayed_gene_delta) && displayed_gene_delta < -1e-10) arrow_down else ""

  # The intermediate-TF node reports the actual signed contribution of its
  # displayed Target → Peak_A → TF paths, not a synthetic sign product.
  displayed_route_response <- t2tf_agg %>%
    filter(affected_gene %in% top_tfs, peak_id %in% all_peak_a) %>%
    group_by(affected_gene) %>%
    summarise(pathway_delta = sum(total_effect, na.rm = TRUE), .groups = "drop")

  # Per-peak net effect for Peak_A (target peaks: sum effects going through each peak)
  peak_a_net <- if (nrow(t2tf_agg) > 0) {
    t2tf_agg %>%
      group_by(peak_id) %>%
      summarise(net_effect = sum(total_effect), .groups = "drop")
  } else {
    data.frame(peak_id = character(0), net_effect = numeric(0))
  }
  peak_a_arrow <- setNames(
    rep("", nrow(peak_a_net)),
    peak_a_net$peak_id
  )

  # Per-peak net effect for Peak_B (combined direct + indirect effects to gene)
  peak_b_effects <- data.frame(
    peak_id = character(0), total_effect = numeric(0),
    peak_gene_pearson_r = numeric(0),
    peak_gene_direction_weight = numeric(0)
  )
  if (has_direct && nrow(direct_agg) > 0) {
    peak_b_effects <- rbind(peak_b_effects,
      direct_agg %>% select(peak_id, total_effect, peak_gene_pearson_r,
                            peak_gene_direction_weight))
  }
  if (has_indirect && nrow(indirect_agg) > 0) {
    peak_b_effects <- rbind(peak_b_effects,
      indirect_agg %>% select(peak_id, total_effect, peak_gene_pearson_r,
                              peak_gene_direction_weight))
  }
  peak_b_net <- if (nrow(peak_b_effects) > 0) {
    peak_b_effects %>%
      group_by(peak_id) %>%
      summarise(
        net_effect = sum(total_effect, na.rm = TRUE),
        peak_gene_pearson_r = weighted_reg_direction(
          peak_gene_pearson_r, peak_gene_direction_weight
        ),
        .groups = "drop"
      )
  } else {
    data.frame(
      peak_id = character(0), net_effect = numeric(0),
      peak_gene_pearson_r = numeric(0)
    )
  }
  peak_b_arrow <- setNames(
    rep("", nrow(peak_b_net)),
    peak_b_net$peak_id
  )

  # =========================================================================
  # Build edge dataframes (vertical layout)
  # =========================================================================

  # --- Edge set 1: Target → Peak_A (TF→Peak edge) ---
  edge_t_pa <- list()
  if (length(all_peak_a) > 0) {
    for (j in seq_len(nrow(t2tf_agg))) {
      e <- t2tf_agg[j, ]
      pk <- e$peak_id
      if (!pk %in% names(peak_a_x)) next
      edge_t_pa[[length(edge_t_pa) + 1]] <- data.frame(
        x_from = target_x, y_from = y_target,
        x_to   = peak_a_x[[pk]], y_to   = y_peak_a,
        effect = e$total_effect,
        reg_direction = e$tf_peak_weight,
        clusters = if (!is.null(e$clusters)) e$clusters else "",
        edge_type = "target_to_peak",
        stringsAsFactors = FALSE
      )
    }
  }

  # --- Edge set 2: Peak_A → Intermediate TF ---
  edge_pa_tf <- list()
  if (length(all_peak_a) > 0 && length(top_tfs) > 0) {
    for (j in seq_len(nrow(t2tf_agg))) {
      e <- t2tf_agg[j, ]
      pk <- e$peak_id
      tf <- as.character(e$affected_gene)
      if (!pk %in% names(peak_a_x) || !tf %in% names(tf_x)) next
      edge_pa_tf[[length(edge_pa_tf) + 1]] <- data.frame(
        x_from = peak_a_x[[pk]], y_from = y_peak_a,
        x_to   = tf_x[[tf]], y_to   = y_tf_mid,
        effect = e$total_effect,
        reg_direction = e$peak_gene_pearson_r,
        clusters = if (!is.null(e$clusters)) e$clusters else "",
        edge_type = "peak_to_tf",
        stringsAsFactors = FALSE
      )
    }
  }

  # --- Edge set 3: Intermediate TF → Peak_B ---
  edge_tf_pb <- list()
  if (has_indirect) {
    for (j in seq_len(nrow(indirect_agg))) {
      e <- indirect_agg[j, ]
      tf <- e$tf_name
      pk <- e$peak_id
      if (!tf %in% names(tf_x) || !pk %in% names(peak_b_x)) next
      edge_tf_pb[[length(edge_tf_pb) + 1]] <- data.frame(
        x_from = tf_x[[tf]], y_from = y_tf_mid,
        x_to   = peak_b_x[[pk]], y_to   = y_peak_b,
        effect = e$total_effect,
        reg_direction = e$tf_peak_weight,
        r2     = e$r2_mean,
        tf_lag = if (!is.null(e$tf_lag_mean) && !is.na(e$tf_lag_mean))
                 e$tf_lag_mean else NA_real_,
        clusters = if (!is.null(e$clusters)) e$clusters else "",
        edge_type = "tf_to_peak",
        stringsAsFactors = FALSE
      )
    }
  }

  # --- Edge set 4: Target → Peak_B (direct, skip intermediate TF) ---
  edge_t_pb <- list()
  if (has_direct) {
    for (j in seq_len(nrow(direct_agg))) {
      e <- direct_agg[j, ]
      pk <- e$peak_id
      if (!pk %in% names(peak_b_x)) next
      edge_t_pb[[length(edge_t_pb) + 1]] <- data.frame(
        x_from = target_x, y_from = y_target,
        x_to   = peak_b_x[[pk]], y_to   = y_peak_b,
        effect = e$total_effect,
        reg_direction = e$tf_peak_weight,
        r2     = e$r2_mean,
        tf_lag = if (!is.null(e$tf_lag_mean) && !is.na(e$tf_lag_mean))
                 e$tf_lag_mean else NA_real_,
        clusters = if (!is.null(e$clusters)) e$clusters else "",
        edge_type = "target_to_peak_direct",
        stringsAsFactors = FALSE
      )
    }
  }

  # --- Edge set 5: Peak_B → Affected Gene ---
  edge_pb_g <- list()
  for (pk in all_peak_b) {
    # Sum effects from both direct and indirect contributions
    pk_eff <- 0
    if (has_direct) {
      pk_eff <- pk_eff + sum(direct_agg$total_effect[direct_agg$peak_id == pk])
    }
    if (has_indirect) {
      pk_eff <- pk_eff + sum(indirect_agg$total_effect[indirect_agg$peak_id == pk])
    }
    edge_pb_g[[length(edge_pb_g) + 1]] <- data.frame(
      x_from = peak_b_x[[pk]], y_from = y_peak_b,
      x_to   = gene_x, y_to   = y_gene,
      effect = pk_eff,
      reg_direction = peak_b_net$peak_gene_pearson_r[
        match(pk, peak_b_net$peak_id)
      ],
      edge_type = "peak_to_gene",
      stringsAsFactors = FALSE
    )
  }

  empty_edge_df <- data.frame(
    x_from = numeric(0), y_from = numeric(0),
    x_to = numeric(0), y_to = numeric(0),
    effect = numeric(0), reg_direction = numeric(0),
    r2 = numeric(0), tf_lag = numeric(0),
    clusters = character(0), edge_type = character(0),
    edge_color = character(0), edge_alpha = numeric(0), edge_lw = numeric(0),
    stringsAsFactors = FALSE
  )

  # Combine all edges
  edge_t_pa_df   <- if (length(edge_t_pa) > 0)   do.call(rbind, edge_t_pa)   else empty_edge_df
  edge_pa_tf_df  <- if (length(edge_pa_tf) > 0)  do.call(rbind, edge_pa_tf)  else empty_edge_df
  edge_tf_pb_df  <- if (length(edge_tf_pb) > 0)  do.call(rbind, edge_tf_pb)  else empty_edge_df
  edge_t_pb_df   <- if (length(edge_t_pb) > 0)   do.call(rbind, edge_t_pb)   else empty_edge_df
  edge_pb_g_df   <- if (length(edge_pb_g) > 0)   do.call(rbind, edge_pb_g)   else empty_edge_df

  # Stop arrow tips at node boundaries instead of drawing through node centers.
  trim_edge_ends <- function(df, start_pad = 0.010, end_pad = 0.016) {
    if (is.null(df) || nrow(df) == 0) return(df)
    dx <- df$x_to - df$x_from
    dy <- df$y_to - df$y_from
    distance <- pmax(sqrt(dx^2 + dy^2), 1e-6)
    df$x_from <- df$x_from + dx / distance * start_pad
    df$y_from <- df$y_from + dy / distance * start_pad
    df$x_to <- df$x_to - dx / distance * end_pad
    df$y_to <- df$y_to - dy / distance * end_pad
    df
  }

  edge_t_pa_df  <- trim_edge_ends(edge_t_pa_df, 0.018, 0.020)
  edge_pa_tf_df <- trim_edge_ends(edge_pa_tf_df, 0.016, 0.022)
  edge_tf_pb_df <- trim_edge_ends(edge_tf_pb_df, 0.022, 0.020)
  edge_t_pb_df  <- trim_edge_ends(edge_t_pb_df, 0.018, 0.022)
  edge_pb_g_df  <- trim_edge_ends(edge_pb_g_df, 0.012, 0.022)

  # Collect all effects for scaling
  all_effects <- c(
    if (!is.null(edge_t_pa_df))  edge_t_pa_df$effect  else 0,
    if (!is.null(edge_pa_tf_df)) edge_pa_tf_df$effect else 0,
    if (!is.null(edge_tf_pb_df)) edge_tf_pb_df$effect else 0,
    if (!is.null(edge_t_pb_df))  edge_t_pb_df$effect  else 0,
    if (!is.null(edge_pb_g_df))  edge_pb_g_df$effect  else 0
  )
  abs_max <- max(abs(all_effects), 1e-6)

  style_edges <- function(df) {
    if (is.null(df) || nrow(df) == 0) return(empty_edge_df)
    df %>%
      mutate(
        edge_color = case_when(
          !is.na(reg_direction) & reg_direction > 1e-8 ~ COLORS$edge_pos,
          !is.na(reg_direction) & reg_direction < -1e-8 ~ COLORS$edge_neg,
          TRUE ~ "#9CA3AF"
        ),
        edge_alpha = pmin(0.72, pmax(0.28, sqrt(abs(effect) / abs_max) * 0.72)),
        edge_lw    = pmax(0.7, pmin(3.0, sqrt(abs(effect) / abs_max) * 3.0))
      )
  }

  edge_t_pa_df  <- style_edges(edge_t_pa_df)
  edge_pa_tf_df <- style_edges(edge_pa_tf_df)
  edge_tf_pb_df <- style_edges(edge_tf_pb_df)
  edge_t_pb_df  <- style_edges(edge_t_pb_df)
  edge_pb_g_df  <- style_edges(edge_pb_g_df)

  # =========================================================================
  # Node dataframes (vertical layout)
  # =========================================================================
  node_target <- data.frame(
    x = target_x, y = y_target,
    label = paste0(target_gene_name, " (KO)", target_arrow),
    stringsAsFactors = FALSE
  )
  response_fill <- function(value) {
    ifelse(!is.finite(value) | abs(value) <= 1e-10, "#E5E7EB",
           ifelse(value > 0, "#FADBD8", "#D5E5F2"))
  }

  node_tfs <- if (length(top_tfs) > 0) {
    tf_labels <- names(tf_x)
    # The intermediate-TF node describes only the displayed upstream pathway,
    # not the TF's global KO response. Use the direction-consistent route
    # response computed from the per-cluster pathway rows above.
    tf_pathway_delta <- setNames(
      sapply(tf_labels, function(tf) {
        val <- displayed_route_response$pathway_delta[
          displayed_route_response$affected_gene == tf
        ]
        if (length(val) > 0) val[1] else NA_real_
      }),
      tf_labels
    )
    tf_arrow <- setNames(
      ifelse(tf_pathway_delta > 0, arrow_up,
             ifelse(tf_pathway_delta < 0, arrow_down, "")),
      tf_labels
    )
    tf_label_text <- paste0(tf_labels, tf_arrow[tf_labels],
                            "\ndisplayed path ΔRNA = ",
                            ifelse(is.na(tf_pathway_delta[tf_labels]),
                                   "—",
                                   sprintf("%+.3f", tf_pathway_delta[tf_labels])))
    # Place labels above the nodes, beyond the incoming edge endpoints. Raise
    # alternating labels when several TFs share the row to avoid collisions.
    tf_label_y <- y_tf_mid + 0.09 +
      ifelse(length(tf_labels) > 1 & seq_along(tf_labels) %% 2 == 0, 0.055, 0)
    tf_center <- mean(unname(tf_x))
    tf_label_x <- ifelse(
      unname(tf_x) < tf_center, unname(tf_x) - 0.075,
      ifelse(unname(tf_x) > tf_center, unname(tf_x) + 0.075,
             unname(tf_x) + 0.10)
    )
    data.frame(
      x = unname(tf_x), y = y_tf_mid,
      label_x = tf_label_x, label_y = tf_label_y,
      label = tf_label_text,
      fill = unname(response_fill(ifelse(is.na(tf_pathway_delta),
                                         0, tf_pathway_delta))),
      stringsAsFactors = FALSE
    )
  } else NULL

  node_peak_a <- if (length(all_peak_a) > 0) {
    pk_labels <- names(peak_a_x)
    peak_effect <- abs(peak_a_net$net_effect[match(pk_labels, peak_a_net$peak_id)])
    peak_effect[is.na(peak_effect)] <- 0
    data.frame(
      x = unname(peak_a_x), y = y_peak_a,
      label = rep("", length(pk_labels)),
      node_size = 3.2 + 3.0 * sqrt(peak_effect / abs_max),
      stringsAsFactors = FALSE
    )
  } else NULL

  node_peak_b <- if (length(all_peak_b) > 0) {
    pk_labels <- names(peak_b_x)
    peak_effect <- abs(peak_b_net$net_effect[match(pk_labels, peak_b_net$peak_id)])
    peak_effect[is.na(peak_effect)] <- 0
    data.frame(
      x = unname(peak_b_x), y = y_peak_b,
      label = rep("", length(pk_labels)),
      node_size = 3.2 + 3.0 * sqrt(peak_effect / abs_max),
      stringsAsFactors = FALSE
    )
  } else NULL

  node_gene <- data.frame(
    x = gene_x, y = y_gene,
    label = if (is.null(cluster_filter)) {
      sprintf("%s%s\nglobal KO log2FC = %+.2f", gene_name, gene_arrow, delta_total)
    } else {
      sprintf("%s%s\nCluster %s ΔRNA = %+.2f", gene_name, gene_arrow,
              cluster_filter, ifelse(is.null(cluster_effect), NA_real_, cluster_effect))
    },
    # Response fill is the cluster RNA effect, not intrinsic edge direction.
    fill  = response_fill(displayed_gene_delta),
    stringsAsFactors = FALSE
  )

  # =========================================================================
  # Draw (vertical layout)
  # =========================================================================
  # Row background rects
  rects <- list(
    annotate("rect", xmin = 0.03, xmax = 0.97, ymin = 0.87, ymax = 0.99,
             fill = "white", color = NA),
    annotate("rect", xmin = 0.03, xmax = 0.97, ymin = 0.66, ymax = 0.78,
             fill = "white", color = NA),
    annotate("rect", xmin = 0.03, xmax = 0.97, ymin = 0.42, ymax = 0.58,
             fill = "white", color = NA),
    annotate("rect", xmin = 0.03, xmax = 0.97, ymin = 0.22, ymax = 0.34,
             fill = "white", color = NA),
    annotate("rect", xmin = 0.03, xmax = 0.97, ymin = 0.01, ymax = 0.13,
             fill = "white", color = NA)
  )

  p <- ggplot() +
    rects +
    # --- Edges: Target → Peak_A ---
    geom_segment(
      data = edge_t_pa_df,
      aes(x = x_from, y = y_from, xend = x_to, yend = y_to,
          color = edge_color, linewidth = edge_lw, alpha = edge_alpha),
      arrow = arrow(length = unit(0.06, "inches"), type = "closed"),
      lineend = "round",
      show.legend = FALSE
    ) +
    # --- Edges: Peak_A → Intermediate TF ---
    geom_segment(
      data = edge_pa_tf_df,
      aes(x = x_from, y = y_from, xend = x_to, yend = y_to,
          color = edge_color, linewidth = edge_lw, alpha = edge_alpha),
      arrow = arrow(length = unit(0.06, "inches"), type = "closed"),
      lineend = "round",
      show.legend = FALSE
    ) +
    # --- Edges: Target → Peak_B (direct, dashed to distinguish) ---
    geom_segment(
      data = edge_t_pb_df,
      aes(x = x_from, y = y_from, xend = x_to, yend = y_to,
          color = edge_color, linewidth = edge_lw, alpha = edge_alpha),
      arrow = arrow(length = unit(0.06, "inches"), type = "closed"),
      lineend = "round", show.legend = FALSE
    ) +
    # --- Edges: Intermediate TF → Peak_B ---
    geom_segment(
      data = edge_tf_pb_df,
      aes(x = x_from, y = y_from, xend = x_to, yend = y_to,
          color = edge_color, linewidth = edge_lw, alpha = edge_alpha),
      arrow = arrow(length = unit(0.06, "inches"), type = "closed"),
      lineend = "round",
      show.legend = FALSE
    ) +
    # --- Edges: Peak_B → Gene (arrow) ---
    geom_segment(
      data = edge_pb_g_df,
      aes(x = x_from, y = y_from, xend = x_to, yend = y_to,
          color = edge_color, linewidth = edge_lw, alpha = edge_alpha),
      arrow = arrow(length = unit(0.07, "inches"), type = "closed"),
      lineend = "round",
      show.legend = FALSE
    ) +
    scale_color_identity() +
    scale_alpha_identity() +
    scale_linewidth_identity() +
    scale_size_identity() +
    scale_fill_identity()

  # --- Target node (KO target gene) ---
  p <- p +
    geom_point(
      data = node_target, aes(x, y), size = 10,
      shape = 21, fill = COLORS$target, color = COLORS$target_2,
      stroke = 1.3, show.legend = FALSE
    ) +
    geom_label(
      data = node_target, aes(x, y, label = label),
      size = 4.4, fontface = "bold", color = COLORS$ink,
      fill = scales::alpha("white", 0.90), linewidth = 0,
      label.r = unit(0.14, "lines"), label.padding = unit(0.10, "lines"),
      nudge_y = 0.045, lineheight = 0.85
    )

  # --- Intermediate TF nodes ---
  if (!is.null(node_tfs) && nrow(node_tfs) > 0) {
    p <- p +
      geom_point(
        data = node_tfs, aes(x, y, fill = fill), size = 7.0,
        shape = 21, color = COLORS$tf,
        stroke = 1.0, show.legend = FALSE
      ) +
      geom_label(
        data = node_tfs, aes(label_x, label_y, label = label),
        size = if (nrow(node_tfs) > 2) 3.5 else 3.8,
        fontface = "italic", color = COLORS$ink,
        fill = scales::alpha("white", 0.88), linewidth = 0,
        label.r = unit(0.10, "lines"), label.padding = unit(0.055, "lines"),
        hjust = 0.5
      )
  }

  # --- Peak A nodes (target-regulated peaks) ---
  if (!is.null(node_peak_a) && nrow(node_peak_a) > 0) {
    p <- p +
      geom_point(
        data = node_peak_a, aes(x, y, size = node_size),
        shape = 21, fill = COLORS$peak, color = "#95A5A6",
        stroke = 0.70, show.legend = FALSE
      )
  }

  # --- Peak B nodes (TF-regulated peaks) ---
  if (!is.null(node_peak_b) && nrow(node_peak_b) > 0) {
    p <- p +
      geom_point(
        data = node_peak_b, aes(x, y, size = node_size),
        shape = 21, fill = COLORS$peak, color = "#95A5A6",
        stroke = 0.70, show.legend = FALSE
      )
  }

  # --- Affected gene node ---
  p <- p +
    geom_label(
      data = node_gene, aes(x, y, label = label, fill = fill),
      size = 4.6, fontface = "bold", color = COLORS$ink,
      linewidth = 0.40,
      label.r = unit(0.16, "lines"),
      label.padding = unit(0.18, "lines"), lineheight = 0.9,
      show.legend = FALSE
    )

  # --- Row headers (vertical layout) ---
  # Keep headers in a real left gutter, outside even the widest intermediate
  # TF/peak lane. The expanded x range prevents text clipping at export.
  header_x <- -0.20
  p <- p +
    annotate("text", x = header_x, y = y_target, label = "KO Target",
             fontface = "bold", size = 5.0, color = COLORS$muted, hjust = 0) +
    annotate("text", x = header_x, y = y_peak_a, label = "Target Peak",
             fontface = "bold", size = 5.0, color = COLORS$muted, hjust = 0) +
    annotate("text", x = header_x, y = y_tf_mid, label = "Intermediate TF",
             fontface = "bold", size = 5.0, color = COLORS$muted, hjust = 0) +
    annotate("text", x = header_x, y = y_peak_b, label = "TF Peak",
             fontface = "bold", size = 5.0, color = COLORS$muted, hjust = 0) +
    annotate("text", x = header_x, y = y_gene,   label = "Affected Gene",
             fontface = "bold", size = 5.0, color = COLORS$muted, hjust = 0)

  if (has_direct && has_indirect) {
    p <- p +
      annotate("segment", x = 0.57, xend = 0.57, y = 0.18, yend = 0.84,
               linewidth = 0.35, linetype = "dotted", color = COLORS$grid)
  }

  # --- Place the legend at the bottom center ---
  p <- p +
    annotate("point", x = 0.35, y = 0.02, size = 2.0, color = COLORS$edge_pos) +
    annotate("text", x = 0.38, y = 0.02, label = "Positive",
             size = 3.5, color = COLORS$edge_pos, hjust = 0) +
    annotate("point", x = 0.55, y = 0.02, size = 2.0, color = COLORS$edge_neg) +
    annotate("text", x = 0.58, y = 0.02, label = "Negative",
             size = 3.5, color = COLORS$edge_neg, hjust = 0) +
    annotate("point", x = 0.76, y = 0.02, size = 2.0, color = "#9CA3AF") +
    annotate("text", x = 0.79, y = 0.02, label = "Uncertain",
             size = 3.5, color = "#6B7280", hjust = 0)

  p <- p +
    xlim(-0.25, 1) + ylim(0, 1) +
    labs(
      title = if (show_main_title) {
        if (is.null(panel_title)) paste0(target_gene_name, " KO — Regulatory Pathway: ", gene_name)
        else panel_title
      } else {
        NULL
      },
      subtitle = paste(
        "Edge color = intrinsic regulatory direction;",
        "intermediate labels = displayed-path contributions;",
        "final gene = global merged RUNX1-KO RNA response"
      )
    ) +
    theme_void() +
    theme(
      plot.title    = element_text(face = "bold", size = 20,
                                   hjust = 0.5, color = COLORS$ink, margin = margin(b = 4)),

      plot.subtitle = element_text(size = 11, hjust = 0.5, color = COLORS$muted),
	      plot.background = element_rect(fill = "white", color = NA, linewidth = 0),
	      plot.margin   = margin(8, 12, 8, 28)
	    )

  return(p)
}

# Compose the original vertical pathway plot once per cluster, followed by a
# compact global result card; the familiar pathway diagram remains unchanged.
plot_cluster_pathway_figure <- function(gene_name, pw_df, cluster_res, merged_df) {
  effects <- cluster_res %>% filter(affected_gene == gene_name, is.finite(delta_rna)) %>%
    mutate(cluster = as.character(cluster)) %>% group_by(cluster) %>%
    summarise(cluster_effect = mean(delta_rna), .groups = "drop")
  available <- pw_df %>% filter(affected_gene == gene_name, !is.na(cluster),
    is.finite(delta_r_raw_sum), abs(delta_rna_log2fc) > 1e-10) %>%
    mutate(cluster = as.character(cluster)) %>% pull(cluster) %>% unique()
  clusters <- sort(intersect(effects$cluster, available))
  if (!length(clusters)) clusters <- sort(effects$cluster)
  panels <- lapply(clusters, function(cl) {
    effect <- effects$cluster_effect[match(cl, effects$cluster)]
    plot_single_pathway(gene_name, pw_df, merged_df, show_main_title = TRUE,
      cluster_filter = cl, cluster_effect = effect,
      panel_title = paste0(target_gene_name, " KO — Cluster ", cl,
                           " Regulatory Pathway: ", gene_name))
  })
  global_value <- merged_df$delta_rna_signed[merged_df$affected_gene == gene_name]
  global_label <- paste0(gene_name, "\nperturbation_merged.csv\nΔRNA = ",
    if (length(global_value) && is.finite(global_value[1]))
      sprintf("%+.2f", global_value[1]) else "—")
  global_panel <- ggplot() +
    annotate("text", .5, .78, label = "GLOBAL WEIGHTED OUTCOME",
      size = 5.0, fontface = "bold", color = COLORS$ink) +
    annotate("text", .5, .45, label = global_label, size = 4.6,
      fontface = "bold", color = COLORS$ink, lineheight = .92) +
    annotate("text", .5, .10,
      label = "Weighted merged result from perturbation_merged.csv",
      size = 3.2, color = COLORS$muted) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), clip = "off") +
    theme_void() + theme(plot.background = element_rect(fill = "#D5E5F2", color = NA),
                          plot.margin = margin(8, 12, 8, 12))
  wrap_plots(c(panels, list(global_panel)), ncol = 1,
             heights = c(rep(1, length(panels)), .48))
}

# Explicit Cartesian radial hierarchy. Angles are used only to construct
# annular polygons; this deliberately does not use coord_polar().
plot_radial_cluster_pathway <- function(gene_name, pw_df, cluster_res, merged_df,
                                         max_routes_per_cluster = 3) {
  epsilon <- 1e-10
  target <- unique(merged_df$target_gene)[1]
  target_names <- unique(trimws(unlist(strsplit(target, "[,+]"))))
  target_names <- target_names[nzchar(target_names)]
  target_label <- paste(target_names, collapse = " + ")
  response_fill <- function(x) {
    if (!is.finite(x) || abs(x) <= epsilon) "#E5E7EB" else
      if (x > 0) "#FADBD8" else "#D5E5F2"
  }
  direction_name <- function(x) {
    if (!is.finite(x) || abs(x) <= epsilon) "Uncertain" else
      if (x > 0) "Positive" else "Negative"
  }
  weighted_direction <- function(value, weight) {
    ok <- is.finite(value) & is.finite(weight)
    if (!any(ok) || sum(abs(weight[ok])) <= 0) return(NA_real_)
    weighted.mean(value[ok], w = abs(weight[ok]))
  }
  polar_xy <- function(r, theta, cy = 0.34) {
    data.frame(x = r * cos(theta), y = cy + r * sin(theta))
  }
  annulus <- function(r0, r1, a0, a1, id, fill, border = NA, n = 24) {
    a <- seq(a0, a1, length.out = n)
    xy <- rbind(polar_xy(r1, a), polar_xy(r0, rev(a)))
    data.frame(x = xy$x, y = xy$y, block = id, fill = fill, border = border)
  }

  # Route ranking is by pathway magnitude; coefficient signs are used only
  # for intrinsic edge colors. Cluster response is never inferred from routes.
  # Path topology is not magnitude-thresholded: every finite aggregated edge
  # remains eligible to form a direct or indirect route; the display cap below
  # is the only route-count limit.
  # Direct routes retain peak identity. Indirect routes are complete joins of
  # same-cluster KO→peak A→TF and TF→peak B→gene halves, with all four edge
  # coefficients retained for intrinsic direction coloring.
  edge_summary <- function(df, by) {
    df %>% group_by(across(all_of(by))) %>% summarise(
      effect = sum(delta_r_raw_sum, na.rm = TRUE),
      contribution = sum(abs(delta_r_raw_sum), na.rm = TRUE),
      tf_coef = weighted_direction(tf_peak_weight, delta_r_raw_sum),
      gene_coef = weighted_direction(peak_gene_pearson_r, delta_r_raw_sum),
      .groups = "drop") %>% filter(is.finite(effect), is.finite(contribution))
  }
  target_edges <- pw_df %>% filter(tf_name %in% target_names,
    affected_gene != gene_name, !is.na(cluster), is.finite(delta_r_raw_sum)) %>%
    mutate(cluster = as.character(cluster), tf_name = as.character(tf_name))
  direct <- pw_df %>% filter(affected_gene == gene_name,
    tf_name %in% target_names, !is.na(cluster), is.finite(delta_r_raw_sum)) %>%
    mutate(cluster = as.character(cluster)) %>%
    edge_summary(c("cluster", "peak_id")) %>% mutate(
      tf_name = target_names[1], direct = TRUE, upstream_tf_coef = tf_coef,
      upstream_gene_coef = NA_real_, downstream_tf_coef = tf_coef,
      downstream_gene_coef = gene_coef, route_kind = "direct")
  upstream <- target_edges %>% edge_summary(c("cluster", "affected_gene", "peak_id")) %>%
    rename(intermediate = affected_gene, peak_a = peak_id,
           upstream_effect = effect, upstream_contribution = contribution,
           upstream_tf_coef = tf_coef, upstream_gene_coef = gene_coef)
  downstream <- pw_df %>% filter(affected_gene == gene_name,
    !tf_name %in% target_names, !is.na(cluster), is.finite(delta_r_raw_sum)) %>%
    mutate(cluster = as.character(cluster), tf_name = as.character(tf_name)) %>%
    edge_summary(c("cluster", "tf_name", "peak_id")) %>%
    rename(intermediate = tf_name, peak_b = peak_id,
           downstream_effect = effect, downstream_contribution = contribution,
           downstream_tf_coef = tf_coef, downstream_gene_coef = gene_coef)
  # Multiple upstream and downstream peak routes per intermediate TF form the
  # intended complete set of indirect paths.
  indirect <- upstream %>% inner_join(downstream,
    by = c("cluster", "intermediate"), relationship = "many-to-many") %>%
    mutate(tf_name = intermediate, direct = FALSE, route_kind = "indirect",
      route_effect = (upstream_effect + downstream_effect) / 2,
      contribution = sqrt(upstream_contribution * downstream_contribution)) %>%
    filter(is.finite(route_effect), is.finite(contribution)) %>%
    group_by(cluster) %>% arrange(desc(contribution), tf_name, peak_a, peak_b) %>%
    slice_head(n = max_routes_per_cluster) %>% ungroup()
  direct <- direct %>% transmute(cluster, tf_name, direct, route_kind,
    route_effect = effect, contribution, peak_a = NA_character_, peak_b = peak_id,
    upstream_tf_coef, upstream_gene_coef, downstream_tf_coef, downstream_gene_coef)
  routes <- bind_rows(direct, indirect) %>% group_by(cluster) %>%
    arrange(desc(contribution), route_kind, tf_name, peak_a, peak_b) %>%
    slice_head(n = max_routes_per_cluster) %>% ungroup()
  # One authoritative lane order per cluster. Every later node layer uses the
  # route membership lane position rather than independently sorting labels.
  routes <- routes %>% group_by(cluster) %>% arrange(desc(contribution),
    route_kind, tf_name, peak_a, peak_b, .by_group = TRUE) %>%
    mutate(route_id = paste(cluster, row_number(), sep = "::"),
      lane_rank = row_number(), lane_pos = if (n() == 1) 0 else
        seq(-1, 1, length.out = n())) %>% ungroup()
  cluster_levels <- union(as.character(cluster_res$cluster), as.character(pw_df$cluster))
  cluster_levels <- cluster_levels[!is.na(cluster_levels) & nzchar(cluster_levels)]
  cluster_levels <- unique(cluster_levels[order(suppressWarnings(as.numeric(cluster_levels)), cluster_levels,
                                                na.last = TRUE)])
  n_cluster <- length(cluster_levels)
  if (!n_cluster) {
    return(ggplot() + annotate("text", 0.5, 0.5,
      label = paste0(gene_name, "\nNo modeled clusters"),
      fontface = "bold", size = 6, color = COLORS$muted) +
      theme_void())
  }
  cy <- 0.34
  center_r <- .22
  cluster_effects <- cluster_res %>% filter(affected_gene == gene_name,
    is.finite(delta_rna)) %>% mutate(cluster = as.character(cluster)) %>%
    group_by(cluster) %>% summarise(cluster_delta = mean(delta_rna), .groups = "drop")
  positive_effects <- abs(cluster_effects$cluster_delta[is.finite(cluster_effects$cluster_delta) &
    abs(cluster_effects$cluster_delta) > epsilon])
  median_positive_effect <- if (length(positive_effects)) median(positive_effects) else 1
  effect_weight <- vapply(cluster_levels, function(cl) {
    z <- cluster_effects$cluster_delta[match(cl, cluster_effects$cluster)]
    if (length(z) && is.finite(z) && abs(z) > epsilon)
      log1p(abs(z) / median_positive_effect) else 0
  }, numeric(1))
  q <- if (max(effect_weight) > 0) effect_weight / max(effect_weight) else
    rep(0, n_cluster)
  alpha <- .35; beta <- .90
  sector_baseline <- min(.72, (2 * pi / n_cluster) * .72)
  raw_sector_angle <- sector_baseline * (1 + alpha * q)
  sector_angle <- raw_sector_angle * (2 * pi / sum(raw_sector_angle))
  sector_end <- pi / 2 + sector_angle[1] / 2 - c(0, cumsum(head(sector_angle, -1)))
  sector_start <- sector_end - sector_angle
  centers <- (sector_start + sector_end) / 2
  names(centers) <- cluster_levels; names(sector_start) <- cluster_levels
  names(sector_end) <- cluster_levels; names(sector_angle) <- cluster_levels
  # Fit center labels against their actual sector arcs. Prefer full names,
  # then compact identities, and grow the center only while preserving the
  # fixed target-peak ring clearance.
  center_label_r <- center_r * .82
  center_label_size <- 1.90
  center_label_df <- NULL
  for (attempt in 1:1) {
    center_label_df <- data.frame(cluster = cluster_levels,
      theta = unname(centers), sector = unname(sector_angle),
      stringsAsFactors = FALSE) %>% mutate(
        label = vapply(seq_along(cluster), function(i) {
          candidates <- c(paste0("Cluster ", cluster[i]), paste0("C", cluster[i]), "C")
          usable <- sector[i] * .82 * center_label_r
          fits <- vapply(candidates, function(s) nchar(s) * .0075 *
            center_label_size <= usable, logical(1))
          if (any(fits)) candidates[which(fits)[1]] else candidates[length(candidates)]
        }, character(1)),
        tangential = nchar(label) * .0075 * center_label_size,
        permitted = sector * .82 * center_label_r,
        label_size = pmax(.72, pmin(center_label_size,
          center_label_size * permitted / pmax(tangential, 1e-6))))
    if (all(center_label_df$tangential <= center_label_df$permitted + 1e-9)) break
  }
  center_label_df <- center_label_df %>% mutate(
    label_angle = (theta * 180 / pi + 90) %% 360,
    label_angle = ifelse(label_angle > 90 & label_angle < 270,
      label_angle + 180, label_angle),
    x = polar_xy(center_label_r, theta)$x,
    y = polar_xy(center_label_r, theta)$y)
  stopifnot(all(center_label_df$tangential <= center_label_df$permitted + 1e-9),
    all(center_label_df$tangential / center_label_r <= center_label_df$sector * .82 + 1e-9),
    all(center_label_r + .012 * center_label_df$label_size < center_r),
    all(sqrt(center_label_df$x^2 + (center_label_df$y - cy)^2) < center_r),
    all(diff(center_label_df$theta) <= 0))
  center_circle <- data.frame(polar_xy(center_r, seq(0, 2 * pi, length.out = 160)))
  observed_weight <- pmax(effect_weight, 0)
  A0 <- .20
  target_envelope_area <- A0 * (1 + beta * q)
  cluster_outer <- sqrt(center_r^2 + 2 * target_envelope_area / sector_angle)
  effect_strength <- q
  names(cluster_outer) <- cluster_levels; names(effect_strength) <- cluster_levels
  layer_names <- c("target_peak", "intermediate_tf", "tf_peak", "terminal")
  base_fraction <- c(target_peak = .12, intermediate_tf = .12,
                    tf_peak = .12, terminal = .17)
  lift_fraction <- c(target_peak = .02, intermediate_tf = .02,
                    tf_peak = .02, terminal = .03)
  layer_bounds <- do.call(rbind, lapply(cluster_levels, function(cl) {
    fractions <- base_fraction + lift_fraction * effect_strength[cl]
    depth <- cluster_outer[cl] - center_r
    corridor <- depth * (1 - sum(fractions)) / (length(layer_names) + 1)
    heights <- depth * fractions
    starts <- center_r + corridor + c(0, cumsum(head(heights + corridor, -1)))
    data.frame(cluster = cl, layer = layer_names, r0 = starts,
      r1 = starts + heights, height = heights, corridor = corridor,
      stringsAsFactors = FALSE)
  }))
  layer_bounds$radius <- (layer_bounds$r0 + layer_bounds$r1) / 2
  layer_r <- setNames(layer_bounds$radius, paste(layer_bounds$cluster, layer_bounds$layer, sep = "||"))
  layer_inner <- function(cl, layer) layer_bounds$r0[layer_bounds$cluster == cl & layer_bounds$layer == layer]
  layer_outer <- function(cl, layer) layer_bounds$r1[layer_bounds$cluster == cl & layer_bounds$layer == layer]
  max_body_r <- max(cluster_outer)
  stopifnot(all(layer_bounds$r0 < layer_bounds$r1),
    all(layer_bounds$r0 > center_r),
    all(layer_bounds$r1 <= cluster_outer[layer_bounds$cluster] + 1e-9),
    all(vapply(split(layer_bounds, layer_bounds$cluster), function(z)
      all(z$r1[-nrow(z)] + z$corridor[-nrow(z)] <= z$r0[-1] + 1e-9), logical(1))))
  if (n_cluster > 1) stopifnot(all(sector_end[-n_cluster] >=
    sector_start[-1] - 1e-9), all(sector_start >= sector_end - sector_angle - 1e-9))
  effect_order <- order(effect_strength)
  stopifnot(all(diff(sector_angle[effect_order]) >= -1e-9),
    all(diff(layer_bounds$height[layer_bounds$layer == "target_peak"][effect_order]) >= -1e-9),
    all(diff(layer_bounds$height[layer_bounds$layer == "terminal"][effect_order]) >= -1e-9))
  envelope_area <- sector_angle * (cluster_outer^2 - center_r^2) / 2
  stopifnot(max(abs(envelope_area / target_envelope_area - 1)) < 1e-8,
    all(diff(sector_angle[effect_order]) >= -1e-9),
    all(diff(cluster_outer[effect_order]) >= -1e-9),
    all(diff(target_envelope_area[effect_order]) >= -1e-9))

  dividers <- do.call(rbind, lapply(seq_len(n_cluster), function(i) {
    b <- sector_start[i]; xy <- rbind(polar_xy(center_r, b), polar_xy(max_body_r, b))
    data.frame(x = xy$x[1], y = xy$y[1], xend = xy$x[2], yend = xy$y[2])
  }))
  layer_guides <- do.call(rbind, lapply(seq_len(nrow(layer_bounds)), function(i) {
    z <- layer_bounds[i, ]; a <- seq(sector_start[z$cluster], sector_end[z$cluster], length.out = 80)
    xy <- polar_xy(z$r1, a); data.frame(x = xy$x, y = xy$y, r = z$r1, cluster = z$cluster)
  }))

  terminal_blocks <- do.call(rbind, lapply(seq_along(cluster_levels), function(i) {
    cl <- cluster_levels[i]; eff <- cluster_effects$cluster_delta[
      match(cl, cluster_effects$cluster)]
    has_route <- cl %in% routes$cluster
    if (!length(eff) || !is.finite(eff)) {
      # A retained route is evidence of a modeled terminal even when the
      # observed delta_rna result is unavailable; render that terminal at
      # ordinary near-zero rather than labeling it Missing.
      if (has_route) {
        eff <- 0
        fill <- response_fill(eff); border <- COLORS$ink
        status <- "near-zero"
      } else {
        fill <- "white"; border <- COLORS$muted
        status <- "missing"
      }
      terminal_r0 <- layer_inner(cl, "terminal"); outer_r <- layer_outer(cl, "terminal")
    } else {
      if (abs(eff) <= epsilon) {
        fill <- response_fill(eff); border <- COLORS$ink
        terminal_r0 <- layer_inner(cl, "terminal"); outer_r <- layer_outer(cl, "terminal")
      } else {
        terminal_r0 <- layer_inner(cl, "terminal"); outer_r <- layer_outer(cl, "terminal")
        fill <- response_fill(eff); border <- COLORS$ink
      }
      status <- if (abs(eff) <= epsilon) "near-zero" else "observed"
    }
    a <- centers[cl]; block <- annulus(terminal_r0, outer_r,
      sector_start[cl], sector_end[cl],
      paste0("terminal_", cl), fill, border)
    block$cluster <- cl; block$delta <- eff; block$status <- status
    block$span <- sector_angle[cl] / 2; block$r0 <- terminal_r0
    block$r1 <- outer_r; block$a0 <- sector_start[cl]; block$a1 <- sector_end[cl]
    block
  }))
  terminal_centers <- terminal_blocks %>% group_by(cluster) %>% summarise(
    delta = first(delta), status = first(status), span = first(span),
    r0 = first(r0), r1 = first(r1), a0 = first(a0), a1 = first(a1), .groups = "drop") %>%
    mutate(has_route = cluster %in% routes$cluster) %>%
    mutate(theta = unname(centers[cluster]), r = (r0 + r1) / 2) %>%
    cbind(polar_xy(.$r, .$theta))
  terminal_labels <- terminal_centers %>% mutate(
    arc_length = (a1 - a0) * r,
    label_r = (r0 + r1) / 2,
    label_x = polar_xy(label_r, theta)$x, label_y = polar_xy(label_r, theta)$y,
    label_angle = (theta * 180 / pi + 90) %% 360,
    label_angle = ifelse(label_angle > 90 & label_angle < 270, label_angle + 180, label_angle),
    label = ifelse(status == "missing", "Missing",
      ifelse(!has_route, paste0("ΔRNA ", sprintf("%+.2f", delta), "\nNo route"),
        paste0("ΔRNA ", sprintf("%+.2f", delta)))),
    label_lines = vapply(strsplit(as.character(label), "\n", fixed = TRUE), length, integer(1)),
    label_chars = vapply(strsplit(as.character(label), "\n", fixed = TRUE), function(z) max(nchar(z)), integer(1)),
    safe_arc = pmax(arc_length - .024, .001), safe_radial = pmax(r1 - r0 - .018, .001),
    label_size = pmax(.35, pmin(2.5,
      safe_arc / pmax(label_chars * .0065, .001),
      safe_radial / pmax(label_lines * .012, .001))))
  stopifnot(all(terminal_labels$label_chars * .0065 * terminal_labels$label_size <=
      terminal_labels$safe_arc + 1e-9),
    all(terminal_labels$label_lines * .012 * terminal_labels$label_size <=
      terminal_labels$safe_radial + 1e-9))
  # Canonicalize biological nodes before assigning any coordinates. Full peak
  # IDs and full TF symbols define identity; compact display labels are never
  # used as keys. This guarantees that every route sharing a node converges
  # on the same point, rather than on a coordinate later removed by dedup.
  node_key <- function(cluster, node_type, full_id) {
    paste(cluster, node_type, full_id, sep = "||")
  }
  canonical_nodes <- bind_rows(
    routes %>% filter(!direct) %>% transmute(cluster, node_type = "target_peak",
      full_id = as.character(peak_a), lane_pos),
    routes %>% filter(!direct) %>% transmute(cluster, node_type = "intermediate_tf",
      full_id = as.character(tf_name), lane_pos),
    routes %>% transmute(cluster, node_type = "tf_peak",
      full_id = as.character(peak_b), lane_pos)
  ) %>% filter(!is.na(full_id), nzchar(full_id)) %>%
    group_by(cluster, node_type, full_id) %>%
    summarise(lane_pos = mean(lane_pos), .groups = "drop") %>%
    group_by(cluster) %>%
    mutate(max_node_count = n()) %>% group_by(cluster, node_type) %>%
    arrange(lane_pos, full_id, .by_group = TRUE) %>%
    mutate(node_rank = row_number(), node_count = n(),
      layer_gutter = min(.012, sector_angle[cluster] * .035 / max(node_count, 1)),
      arc_width = pmax(.004, (sector_angle[cluster] - 2 * layer_gutter -
        (node_count - 1) * layer_gutter) / node_count),
      theta = sector_start[cluster] + layer_gutter +
        (node_rank - .5) * (arc_width + layer_gutter),
      radius = vapply(seq_len(n()), function(i) {
        layer_r[[paste(cluster[i], node_type[i], sep = "||")]]
      }, numeric(1)),
      xy = lapply(seq_len(n()), function(i) polar_xy(radius[i], theta[i]))) %>%
    ungroup()
  canonical_nodes$x <- vapply(canonical_nodes$xy, function(z) z$x, numeric(1))
  canonical_nodes$y <- vapply(canonical_nodes$xy, function(z) z$y, numeric(1))
  wrap_peak_coordinate <- function(x) {
    x <- as.character(x)
    x <- sub(":", ":\n", x, fixed = TRUE)
    sub("-", "-\n", x, fixed = TRUE)
  }
  next_letter_id <- function(i) {
    out <- character(length(i))
    for (j in seq_along(i)) {
      n <- i[j]; s <- ""
      repeat {
        s <- paste0(LETTERS[(n - 1) %% 26 + 1], s)
        n <- (n - 1) %/% 26
        if (!n) break
      }
      out[j] <- s
    }
    out
  }
  role_order <- c(target_peak = "Peak A", intermediate_tf = "TF", tf_peak = "Peak B")
  id_table <- canonical_nodes %>% distinct(node_type, full_id) %>%
    arrange(factor(node_type, names(role_order)), full_id) %>%
    mutate(display_id = next_letter_id(row_number()), role = role_order[node_type])
  canonical_nodes <- canonical_nodes %>% left_join(id_table,
    by = c("node_type", "full_id"))
  canonical_nodes$label <- ifelse(canonical_nodes$node_type == "intermediate_tf",
    canonical_nodes$full_id, canonical_nodes$display_id)
  canonical_nodes$lookup <- node_key(canonical_nodes$cluster,
    canonical_nodes$node_type, canonical_nodes$full_id)
  canonical_lookup <- split(seq_len(nrow(canonical_nodes)), canonical_nodes$lookup)
  get_node <- function(cluster, node_type, full_id) {
    idx <- canonical_lookup[[node_key(cluster, node_type, as.character(full_id))]]
    if (is.null(idx)) stop("Missing canonical radial node: ", node_key(cluster, node_type, full_id))
    canonical_nodes[idx[1], ]
  }

  node_bounds <- canonical_nodes %>% transmute(lookup,
    raw_a0 = theta - arc_width / 2, raw_a1 = theta + arc_width / 2,
    r0 = vapply(seq_len(n()), function(i) layer_inner(cluster[i], node_type[i]), numeric(1)),
    r1 = vapply(seq_len(n()), function(i) layer_outer(cluster[i], node_type[i]), numeric(1))) %>%
    mutate(gutter = pmin(.018, (raw_a1 - raw_a0) / 4),
      a0 = raw_a0 + gutter, a1 = raw_a1 - gutter,
      r0 = r0 + .006, r1 = r1 - .006)
  canonical_nodes <- canonical_nodes %>% left_join(node_bounds, by = "lookup")
  node_blocks <- if (nrow(canonical_nodes)) {
    do.call(rbind, lapply(seq_len(nrow(canonical_nodes)), function(i) {
      b <- canonical_nodes[i, ]
      block <- annulus(b$r0, b$r1, b$a0, b$a1,
        b$lookup, ifelse(b$node_type == "intermediate_tf", COLORS$tf_bg, COLORS$peak),
        ifelse(b$node_type == "intermediate_tf", COLORS$tf, "#95A5A6"), n = 20)
      block$node_type <- b$node_type
      block$cluster <- b$cluster
      block$r0 <- b$r0; block$r1 <- b$r1; block$a0 <- b$a0; block$a1 <- b$a1
      block
    }))
  } else data.frame(x = numeric(), y = numeric(), block = character(),
    fill = character(), border = character(), node_type = character(),
    cluster = character(), r0 = numeric(), r1 = numeric(), a0 = numeric(), a1 = numeric())
  if (nrow(canonical_nodes)) {
    stopifnot(all(routes$lane_rank == ave(routes$lane_rank, routes$cluster,
      FUN = seq_along)))
    stopifnot(all(canonical_nodes$a0 >= sector_start[canonical_nodes$cluster] - 1e-9),
      all(canonical_nodes$a1 <= sector_end[canonical_nodes$cluster] + 1e-9))
    for (i in seq_len(nrow(routes))) {
      z <- routes[i, ]
      widths <- if (z$direct) {
        c(get_node(z$cluster, "tf_peak", z$peak_b)$a1 - get_node(z$cluster, "tf_peak", z$peak_b)$a0,
          sector_angle[z$cluster])
      } else {
        c(get_node(z$cluster, "target_peak", z$peak_a)$a1 - get_node(z$cluster, "target_peak", z$peak_a)$a0,
          get_node(z$cluster, "intermediate_tf", z$tf_name)$a1 - get_node(z$cluster, "intermediate_tf", z$tf_name)$a0,
          get_node(z$cluster, "tf_peak", z$peak_b)$a1 - get_node(z$cluster, "tf_peak", z$peak_b)$a0,
          sector_angle[z$cluster])
      }
    }
  }

  # Build every route segment from canonical block boundaries.
  path_segments <- list()
  for (cl in cluster_levels) {
    rr <- routes %>% filter(cluster == cl)
    if (!nrow(rr)) next
    for (j in seq_len(nrow(rr))) {
      z <- rr[j, ]
      terminal <- terminal_centers %>% filter(cluster == cl)
      route_nodes <- if (z$direct) {
        list(get_node(cl, "tf_peak", z$peak_b))
      } else {
        list(get_node(cl, "target_peak", z$peak_a),
          get_node(cl, "intermediate_tf", z$tf_name),
          get_node(cl, "tf_peak", z$peak_b))
      }
      cols <- if (z$direct) c(z$upstream_tf_coef, z$downstream_gene_coef) else
        c(z$upstream_tf_coef, z$upstream_gene_coef,
          z$downstream_tf_coef, z$downstream_gene_coef)
      for (k in seq_len(length(route_nodes) + 1)) {
        is_terminal_edge <- k == length(route_nodes) + 1
        p0 <- if (k == 1) polar_xy(center_r, route_nodes[[1]]$theta) else
          polar_xy(route_nodes[[k - 1]]$r1, route_nodes[[k - 1]]$theta)
        p1 <- if (is_terminal_edge) polar_xy(
          terminal$r0, centers[cl]) else
          polar_xy(route_nodes[[k]]$r0, route_nodes[[k]]$theta)
        expected_p0_r <- if (k == 1) center_r else route_nodes[[k - 1]]$r1
        expected_p1_r <- if (is_terminal_edge) terminal$r0 else
          route_nodes[[k]]$r0
        stopifnot(abs(sqrt(p0$x^2 + (p0$y - cy)^2) - expected_p0_r) < 1e-8,
          abs(sqrt(p1$x^2 + (p1$y - cy)^2) - expected_p1_r) < 1e-8)
        path_segments[[length(path_segments) + 1]] <- data.frame(
          x = p0$x, y = p0$y, xend = p1$x, yend = p1$y,
          direction = direction_name(cols[k]),
          direct = z$direct, is_terminal_edge = is_terminal_edge, route_id = z$route_id,
          from_block = if (k == 1) "__center__" else route_nodes[[k - 1]]$lookup,
          to_block = if (k == length(route_nodes) + 1) paste0("terminal_", cl) else
            route_nodes[[k]]$lookup, stringsAsFactors = FALSE)
      }
    }
  }
  segments <- if (length(path_segments)) do.call(rbind, path_segments) else
    data.frame(x = numeric(), y = numeric(), xend = numeric(), yend = numeric(),
      direction = character(), direct = logical(), is_terminal_edge = logical(), route_id = character(),
      from_block = character(), to_block = character())
  block_records <- bind_rows(
    canonical_nodes %>% transmute(block = lookup, r0, r1, a0, a1),
    terminal_centers %>% transmute(block = paste0("terminal_", cluster), r0, r1, a0, a1))
  inside_block <- function(x, y, b) {
    rr <- sqrt(x^2 + (y - cy)^2)
    aa <- atan2(y - cy, x)
    while (aa < b$a0) aa <- aa + 2 * pi
    while (aa > b$a1) aa <- aa - 2 * pi
    rr > b$r0 + 1e-8 && rr < b$r1 - 1e-8 &&
      aa > b$a0 + 1e-8 && aa < b$a1 - 1e-8
  }
  # A direct edge is curved only when its ordinary straight segment is
  # geometrically obstructed by a non-endpoint annular block.
  segments$obstructed <- if (nrow(segments)) rep(FALSE, nrow(segments)) else logical()
  if (nrow(segments) && nrow(block_records)) for (i in which(segments$direct & !segments$is_terminal_edge)) {
    s <- segments[i, ]; tt <- seq(0, 1, length.out = 241)
    xx <- s$x + tt * (s$xend - s$x); yy <- s$y + tt * (s$yend - s$y)
    candidates <- which(!block_records$block %in% c(s$from_block, s$to_block))
    segments$obstructed[i] <- any(vapply(candidates, function(j)
      any(vapply(seq_along(tt), function(k)
        inside_block(xx[k], yy[k], block_records[j, ]), logical(1))), logical(1)))
  }
  if (nrow(segments) && nrow(block_records)) for (i in which(segments$is_terminal_edge)) {
    s <- segments[i, ]; tt <- seq(0, 1, length.out = 401)
    xx <- s$x + tt * (s$xend - s$x); yy <- s$y + tt * (s$yend - s$y)
    source_record <- block_records[block_records$block == s$from_block, , drop = FALSE]
    candidates <- which(!block_records$block %in% c(s$from_block, s$to_block) &
      if (nrow(source_record))
        abs(block_records$r0 - source_record$r0[1]) > 1e-8 |
          abs(block_records$r1 - source_record$r1[1]) > 1e-8
      else TRUE)
    terminal_hits <- candidates[vapply(candidates, function(j)
      any(vapply(seq_along(tt), function(k)
        inside_block(xx[k], yy[k], block_records[j, ]), logical(1))), logical(1))]
    if (length(terminal_hits)) stop("Terminal edge enters a third block: ", s$route_id,
      " / ", paste(block_records$block[terminal_hits], collapse = ","))
  }
  straight_segments <- segments %>% filter(!direct | is_terminal_edge | !obstructed)
  direct_segments <- segments %>% filter(direct & obstructed)
  nodes <- canonical_nodes
  # Anchor labels inside their own block. Peak coordinates are never compacted:
  # only the existing ':' and '-' delimiters may introduce line breaks.
  node_labels <- nodes %>% mutate(
    arc_length = ((r0 + r1) / 2) * pmax(a1 - a0, 0),
    label_r = (r0 + r1) / 2,
    label_x = polar_xy(label_r, theta)$x, label_y = polar_xy(label_r, theta)$y,
    angle = (theta * 180 / pi + 90) %% 360,
    angle = ifelse(angle > 90 & angle < 270, angle + 180, angle),
    safe_arc = pmax(arc_length - .024, .001),
    safe_radial = pmax(r1 - r0 - .018, .001),
    label_lines = vapply(strsplit(as.character(label), "\n", fixed = TRUE), length, integer(1)),
    label_chars = vapply(strsplit(as.character(label), "\n", fixed = TRUE), function(z) max(nchar(z)), integer(1)),
    label_size = pmax(.65, pmin(1.85,
      safe_arc / pmax(label_chars * .0065, .001),
      safe_radial / pmax(label_lines * .012, .001)))
  )
  stopifnot(all(node_labels$label_size >= .65),
    all(node_labels$label_chars * .0065 * node_labels$label_size <= node_labels$safe_arc + 1e-9),
    all(node_labels$label_lines * .012 * node_labels$label_size <= node_labels$safe_radial + 1e-9),
    all(ifelse(node_labels$node_type == "intermediate_tf",
      node_labels$label == node_labels$full_id,
      node_labels$label == node_labels$display_id)),
    all(node_labels$label_chars * .0065 * node_labels$label_size <=
      (node_labels$a1 - node_labels$a0) * node_labels$label_r + 1e-9),
    all(node_labels$label_lines * .012 * node_labels$label_size <=
      node_labels$r1 - node_labels$r0 + 1e-9))
  # Deterministic pairwise tangent-envelope assertion for the retained full
  # labels. The extra scale factor above leaves a device-independent gutter.
  label_groups <- split(node_labels, interaction(node_labels$cluster,
    node_labels$node_type, drop = TRUE))
  for (g in label_groups) if (nrow(g) > 1) {
    g <- g[order(g$theta), , drop = FALSE]
    half_width <- g$label_chars * .0065 * g$label_size / g$label_r
    stopifnot(all(g$theta[-nrow(g)] + half_width[-nrow(g)] <=
      g$theta[-1] - half_width[-1] + 1e-9))
  }
  global <- merged_df$delta_rna_signed[merged_df$affected_gene == gene_name]
  global <- if (length(global) && is.finite(global[1])) global[1] else NA_real_
  # Only observed responses connect to the merged card. Route them radially
  # outward, then along an external lower bus with left-to-right ordered ports;
  # this keeps aggregation lines outside the radial network and non-crossing.
  observed_terminals <- terminal_centers %>% filter(status == "observed") %>%
    mutate(theta = as.numeric(theta), outer_r = r1,
           outer_x = polar_xy(outer_r, theta)$x) %>%
    arrange(outer_x)
  connector_paths <- list()
  body_outer_r <- max_body_r + .04
  bus_y <- cy - body_outer_r - .16
  card_top <- bus_y - .04
  card_label_lines <- c("GLOBAL WEIGHTED OUTCOME",
    paste0("ΔRNA = ", if (is.na(global)) "—" else sprintf("%+.2f", global)))
  # Size the rectangle from the actual multiline content. The conservative
  # glyph-width and line-step constants are in plot data units for the fixed
  # ggsave device; wrapping is deterministic so a future source/explanation
  # line cannot silently escape the card.
  card_text_size <- 3.2
  card_max_chars <- 26L
  card_lines <- unlist(lapply(card_label_lines, function(line)
    strwrap(line, width = card_max_chars, simplify = TRUE)), use.names = FALSE)
  card_text <- paste(card_lines, collapse = "\n")
  card_char_width <- .0075 * card_text_size
  card_line_step <- .015 * card_text_size
  card_text_width <- max(nchar(card_lines)) * card_char_width
  card_half_width <- max(.52, card_text_width / 2 + .10)
  card_height <- max(.42, length(card_lines) * card_line_step + .10)
  card_bottom <- card_top - card_height
  stopifnot(all(nchar(card_lines) * card_char_width <=
      2 * card_half_width - .20 + 1e-9),
    length(card_lines) * card_line_step <= card_height - .10 + 1e-9)
  side_x_abs <- body_outer_r + .16
  port_span <- min(.40, max(.20, .055 * max(nrow(observed_terminals), 1)))
  legend_y <- card_bottom - .055
  response_note_y <- card_bottom - .145
  legend_x_pos <- -body_outer_r
  legend_x_neg <- legend_x_pos + .36
  legend_x_unc <- legend_x_neg + .36
  if (nrow(observed_terminals)) {
    # First travel radially to the outside of the circular body. Only then
    # use side lanes and the lower bus, so no connector cuts through sectors.
    for (i in seq_len(nrow(observed_terminals))) {
      z <- observed_terminals[i, ]
      outer <- polar_xy(max(z$outer_r, body_outer_r), z$theta)
      side_x <- if (outer$x >= 0) side_x_abs else -side_x_abs
      # Preserve left/right ordering at the card so side-lane paths do not
      # cross each other on the way to the weighted global outcome.
      left_n <- sum(observed_terminals$outer_x < 0)
      if (outer$x < 0) {
        left_rank <- sum(observed_terminals$outer_x[seq_len(i)] < 0)
        port_x <- seq(-port_span, -.03, length.out = max(left_n, 1))[left_rank]
      } else {
        right_rank <- sum(observed_terminals$outer_x[seq_len(i)] >= 0)
        right_n <- nrow(observed_terminals) - left_n
        port_x <- seq(.03, port_span, length.out = max(right_n, 1))[right_rank]
      }
      connector_paths[[i]] <- data.frame(
        connector = i,
        x = c(polar_xy(z$outer_r, z$theta)$x, outer$x, side_x, side_x,
              port_x, port_x),
        y = c(polar_xy(z$outer_r, z$theta)$y, outer$y, outer$y, bus_y,
              bus_y, card_top)
      )
    }
  }
  connector_paths <- if (length(connector_paths)) do.call(rbind, connector_paths) else
    data.frame(connector = integer(), x = numeric(), y = numeric())
  radial_plot <- ggplot() +
    geom_polygon(data = layer_guides, aes(x, y, group = r),
                 fill = NA, color = COLORS$grid, linewidth = .25, linetype = "dashed") +
    geom_segment(data = dividers, aes(x, y, xend = xend, yend = yend),
                 color = "white", linewidth = 2.0) +
    geom_segment(data = dividers, aes(x, y, xend = xend, yend = yend),
                 color = COLORS$grid, linewidth = .35) +
    geom_segment(data = straight_segments, aes(x, y, xend = xend, yend = yend,
                                      color = direction), linewidth = .78,
                 alpha = .82, lineend = "round",
                 arrow = arrow(length = unit(.07, "cm"), type = "closed")) +
    geom_curve(data = direct_segments, aes(x, y, xend = xend, yend = yend,
      color = direction), curvature = .34, angle = 90, ncp = 24,
      linewidth = .66, alpha = .58, lineend = "round",
      arrow = arrow(length = unit(.065, "cm"), type = "closed")) +
    geom_polygon(data = terminal_blocks,
                 aes(x, y, group = block, fill = fill, color = border), linewidth = .55) +
    geom_polygon(data = node_blocks, aes(x, y, group = block,
                                         fill = fill, color = border),
                 linewidth = .65) +
    # Re-draw route strokes above blocks: radial corridors, not node fills,
    # own the visual continuity of the fan.
    geom_segment(data = straight_segments, aes(x, y, xend = xend, yend = yend,
                                      color = direction), linewidth = .78,
                 alpha = .82, lineend = "round",
                 arrow = arrow(length = unit(.07, "cm"), type = "closed")) +
    geom_curve(data = direct_segments, aes(x, y, xend = xend, yend = yend,
      color = direction), curvature = .34, angle = 90, ncp = 24,
      linewidth = .66, alpha = .58, lineend = "round",
      arrow = arrow(length = unit(.065, "cm"), type = "closed")) +
    geom_text(data = node_labels,
              aes(label_x, label_y, label = label, angle = angle,
                  size = label_size, group = interaction(cluster, node_type, label)),
              hjust = .5, color = COLORS$ink, lineheight = .8,
              check_overlap = FALSE, show.legend = FALSE) +
    geom_path(data = connector_paths, aes(x, y, group = connector),
              color = COLORS$muted, linewidth = .65, alpha = .75,
              lineend = "round", arrow = arrow(length = unit(.08, "cm"),
                                                type = "closed")) +
    geom_polygon(data = center_circle, aes(x, y), fill = COLORS$target,
                 color = COLORS$target_2, linewidth = 1.1) +
    annotate("text", 0, cy, label = paste0(target_label, " KO"),
             fontface = "bold", size = 2.80, color = COLORS$ink) +
    geom_text(data = center_label_df,
      aes(x, y, label = label, angle = label_angle, size = label_size),
      fontface = "bold", color = COLORS$ink, show.legend = FALSE) +
    geom_text(data = terminal_labels,
      aes(label_x, label_y, label = label, angle = label_angle,
          size = label_size, group = cluster),
      hjust = .5, color = COLORS$ink, fontface = "bold", lineheight = .8,
      check_overlap = TRUE, show.legend = FALSE) +
    geom_rect(aes(xmin = -card_half_width, xmax = card_half_width,
                   ymin = card_bottom, ymax = card_top),
              fill = response_fill(global), color = COLORS$ink, linewidth = .65) +
    annotate("text", 0, (card_bottom + card_top) / 2,
             label = card_text, fontface = "bold", size = card_text_size,
             lineheight = .85, color = COLORS$ink) +
    annotate("point", legend_x_pos, legend_y, size = 2.1, color = COLORS$edge_pos) +
    annotate("text", legend_x_pos + .04, legend_y, label = "Positive", hjust = 0,
             size = 3.2, color = COLORS$edge_pos) +
    annotate("point", legend_x_neg, legend_y, size = 2.1, color = COLORS$edge_neg) +
    annotate("text", legend_x_neg + .04, legend_y, label = "Negative", hjust = 0,
             size = 3.2, color = COLORS$edge_neg) +
    annotate("point", legend_x_unc, legend_y, size = 2.1, color = COLORS$muted) +
    annotate("text", legend_x_unc + .04, legend_y, label = "Uncertain", hjust = 0,
             size = 3.2, color = COLORS$muted) +
    annotate("text", body_outer_r, response_note_y, label = "terminal/card fill = actual RNA response",
             hjust = 1, size = 2.9, color = COLORS$muted) +
    scale_color_manual(values = c(Positive = COLORS$edge_pos,
      Negative = COLORS$edge_neg, Uncertain = COLORS$muted,
      `#2C3E50` = COLORS$ink, `#7F8C8D` = COLORS$muted,
      `#C0392B` = "#C0392B", `#2874A6` = "#2874A6"), guide = "none") +
    scale_fill_identity() +
    scale_size_identity() +
    coord_fixed(xlim = c(-body_outer_r - .20, body_outer_r + .20),
                 ylim = c(card_bottom - .20, cy + max_body_r + .10), clip = "off") +
    labs(title = paste0(target_label, " KO — Radial Cluster Pathway: ", gene_name),
         subtitle = "Direct KO→gene propagation uses curved edges; intrinsic direction is separate from cluster RNA response") +
    theme_void() + theme(plot.title = element_text(face = "bold", size = 20,
      hjust = .5, color = COLORS$ink), plot.subtitle = element_text(size = 11,
      hjust = .5, color = COLORS$muted), plot.margin = margin(10, 14, 12, 14),
      plot.background = element_rect(fill = "white", color = NA))
  legend_df <- id_table %>% filter(node_type != "intermediate_tf") %>% transmute(
    id = display_id,
    y = rev(seq_len(n())),
    label = paste0(display_id, "  ", role, " · ", full_id))
  if (!nrow(legend_df)) legend_df <- data.frame(id = character(), y = numeric(), label = character())
  legend_plot <- ggplot(legend_df, aes(x = 0, y = y, label = label)) +
    geom_text(hjust = 0, vjust = .5, size = 2.9, color = COLORS$ink) +
    labs(title = "Node key\n(full identities)") +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, max(1, nrow(legend_df) + 1)),
      clip = "off") +
    theme_void() + theme(plot.title = element_text(face = "bold", size = 12,
      color = COLORS$ink), plot.margin = margin(18, 8, 18, 2))
  radial_plot + legend_plot + patchwork::plot_layout(widths = c(4.5, 1.7))
}

# Generate independent radial panels. Each gene is rendered against its own
# complete route table and cluster response; combining multiple genes' rows
# would create invalid cross-gene pathways, so panels are stacked vertically.
pathway_plots <- lapply(pathway_genes, function(gene) {
  plot_radial_cluster_pathway(gene, pathways, pert_res, pert_merged)
})

n_genes <- length(pathway_plots)
p_pathways <- wrap_plots(pathway_plots, ncol = 1, nrow = n_genes)

pathway_file <- file.path(OUTPUT_DIR,
  if (is.null(specified_genes)) "pathway_top5.png" else "pathway_specified.png")

ggsave(
  pathway_file,
  p_pathways, width = 10, height = max(10, n_genes * 10), dpi = 320,
  bg = "white", limitsize = FALSE
)
cat(sprintf("[OK] Pathway schematics saved: %s\n", pathway_file))

# --- Strongest positive/negative targets by pathway class ---
# Keep the user-specified plot above, and additionally select effect extremes
# for (1) targets with a direct target-TF edge and (2) purely indirect targets.
target_regulated_tfs <- pathways %>%
  filter(tf_name %in% target_gene_names,
         !affected_gene %in% target_gene_names) %>%
  transmute(intermediate_tf = affected_gene, cluster) %>%
  distinct()

direct_target_genes <- pathways %>%
  filter(tf_name %in% target_gene_names,
         !affected_gene %in% target_gene_names) %>%
  pull(affected_gene) %>%
  unique()

direct_candidates <- data.frame(
  affected_gene = direct_target_genes,
  stringsAsFactors = FALSE
) %>%
  inner_join(
    pert_merged %>% select(affected_gene, delta_rna_signed),
    by = "affected_gene"
  )

pure_indirect_candidates <- pathways %>%
  filter(!tf_name %in% target_gene_names,
         !affected_gene %in% target_gene_names,
         !affected_gene %in% direct_target_genes) %>%
  rename(intermediate_tf = tf_name) %>%
  semi_join(target_regulated_tfs, by = c("intermediate_tf", "cluster")) %>%
  distinct(affected_gene) %>%
  inner_join(
    pert_merged %>%
      select(affected_gene, delta_rna_signed),
    by = "affected_gene"
  )

save_extreme_pathway <- function(candidates, pathway_class, direction,
                                 output_name) {
  if (direction == "positive") {
    selected <- candidates %>%
      filter(delta_rna_signed > 0) %>%
      arrange(desc(delta_rna_signed), affected_gene) %>%
      slice_head(n = 1)
  } else {
    selected <- candidates %>%
      filter(delta_rna_signed < 0) %>%
      arrange(delta_rna_signed, affected_gene) %>%
      slice_head(n = 1)
  }

  if (nrow(selected) == 0) {
    cat(sprintf(
      "WARNING: no %s %s-effect target found; plot skipped\n",
      pathway_class, direction
    ))
    return(invisible(NULL))
  }

  selected_gene <- selected$affected_gene[1]
  selected_effect <- selected$delta_rna_signed[1]
  output_file <- file.path(OUTPUT_DIR, output_name)
  ggsave(
    output_file,
    plot_radial_cluster_pathway(selected_gene, pathways, pert_res, pert_merged),
    width = 10, height = 10,
    dpi = 320, bg = "white"
  )
  cat(sprintf(
    "[OK] %s %s-effect pathway: %s (log2FC=%+.3f) -> %s\n",
    pathway_class, direction, selected_gene, selected_effect, output_file
  ))
  invisible(selected)
}

save_extreme_pathway(
  direct_candidates, "direct-supported", "positive",
  "pathway_direct_top_positive.png"
)
save_extreme_pathway(
  direct_candidates, "direct-supported", "negative",
  "pathway_direct_top_negative.png"
)
save_extreme_pathway(
  pure_indirect_candidates, "pure-indirect", "positive",
  "pathway_indirect_top_positive.png"
)
save_extreme_pathway(
  pure_indirect_candidates, "pure-indirect", "negative",
  "pathway_indirect_top_negative.png"
)

# Remove the obsolete single absolute-effect output to avoid stale ambiguity.
legacy_indirect_file <- file.path(OUTPUT_DIR, "pathway_indirect_top1.png")
if (file.exists(legacy_indirect_file)) unlink(legacy_indirect_file)

# ==============================================================================
# 3. Supplemental: effect distribution + significance summary
# ==============================================================================

p_hist <- ggplot(volcano_df, aes(x = mean_delta)) +
  geom_histogram(aes(fill = after_stat(x >= 0)), bins = 80,
                 color = "white", alpha = 0.88) +
  scale_fill_manual(values = c(`TRUE` = COLORS$up, `FALSE` = COLORS$down),
                    guide = "none") +
  geom_vline(xintercept = 0, linewidth = 0.5, color = COLORS$ink) +
  geom_vline(
    xintercept = median(volcano_df$mean_delta),
    linetype = "dashed", color = COLORS$up, linewidth = 0.8
  ) +
  annotate("text", x = 0, y = Inf, label = "zero", vjust = 1.4,
           hjust = 1.15, size = 3.4, color = COLORS$ink) +
  annotate("text", x = median(volcano_df$mean_delta), y = Inf,
           label = sprintf("median = %.2f", median(volcano_df$mean_delta)),
           vjust = 2.8, hjust = -0.08, size = 3.4, color = COLORS$up) +
  labs(
    x = expression(Delta * " RNA (mean log-fold change)"),
    y = "Number of genes"
  ) +
  theme_pub(base_size = 16) +
  theme(
    panel.grid.major.x = element_blank()
  )

p_bar <- volcano_df %>%
  count(change) %>%
  mutate(change = factor(change, levels = c("Up", "Down", "Stable"))) %>%
  ggplot(aes(x = change, y = n, fill = change)) +
  geom_col(color = "white", linewidth = 0.6, width = 0.66) +
  geom_text(aes(label = n), vjust = -0.45, size = 4.9, fontface = "bold",
            color = COLORS$ink) +
  scale_fill_manual(
    values = c("Up" = COLORS$up, "Down" = COLORS$down, "Stable" = COLORS$stable),
    guide = "none"
  ) +
  labs(
    x = NULL, y = "Number of genes"
  ) +
  theme_pub(base_size = 16) +
  theme(
    panel.grid.major.x   = element_blank(),
    axis.text.x          = element_text(angle = 30, hjust = 1),
    plot.title           = element_blank()
  )

p_dist <- p_hist + p_bar +
  plot_layout(widths = c(2, 1)) +
  plot_annotation(
    title = paste0(target_gene_name, " KO: Perturbation Effect Distribution"),
    theme = theme(
      plot.title = element_text(
        face = "bold", size = 18, hjust = 0.5, color = COLORS$ink
      )
    )
  )

ggsave(
  file.path(OUTPUT_DIR, "distribution.png"),
  p_dist, width = 14, height = 5.5, dpi = 320, bg = "white"
)
cat(sprintf("[OK] Distribution plot saved: %s\n",
            file.path(OUTPUT_DIR, "distribution.png")))

# ==============================================================================
# 4. Cross-cluster differential-effect heatmap
# ==============================================================================
cat("\n--- Cross-cluster heatmap ---\n")

# Aggregate once per gene/cluster while retaining missing combinations as NA.
cluster_long <- pert_res %>%
  group_by(affected_gene, cluster) %>%
  summarise(delta_rna = mean(delta_rna), .groups = "drop")

# Rank genes by cross-cluster heterogeneity while penalizing sparse coverage.
# Direction switches receive a modest bonus because they indicate the clearest
# cluster-specific regulatory response.
n_cluster_total <- n_distinct(cluster_long$cluster)
gene_heterogeneity <- cluster_long %>%
  group_by(affected_gene) %>%
  summarise(
    n_observed = n_distinct(cluster),
    coverage = n_observed / n_cluster_total,
    effect_range = ifelse(n_observed > 1,
                          max(delta_rna) - min(delta_rna), 0),
    effect_sd = ifelse(n_observed > 1, sd(delta_rna), 0),
    direction_switch = any(delta_rna > 0) & any(delta_rna < 0),
    heterogeneity_score = coverage * (effect_range + effect_sd) *
                          ifelse(direction_switch, 1.25, 1),
    .groups = "drop"
  ) %>%
  arrange(desc(heterogeneity_score), desc(effect_range),
          desc(n_observed), affected_gene)

top_genes <- head(gene_heterogeneity$affected_gene,
                  min(40, nrow(gene_heterogeneity)))

# Pivot: genes × clusters matrix of delta_rna. Missing stays NA for display.
cluster_mat <- cluster_long %>%
  filter(affected_gene %in% top_genes) %>%
  tidyr::pivot_wider(
    names_from  = cluster,
    values_from = delta_rna,
    values_fn   = mean  # if duplicate, take mean
  ) %>%
  tibble::column_to_rownames("affected_gene")

mat_top <- cluster_mat[top_genes, , drop = FALSE]

# Cluster columns using a temporary zero-imputed copy; NA remains gray in the plot.
mat_for_clustering <- mat_top
mat_for_clustering[is.na(mat_for_clustering)] <- 0
if (ncol(mat_for_clustering) > 1) {
  col_hc <- hclust(dist(t(mat_for_clustering)), method = "ward.D2")
  cluster_levels <- col_hc$labels[col_hc$order]
} else {
  cluster_levels <- colnames(mat_for_clustering)
}

mat_melt <- mat_top %>%
  tibble::rownames_to_column("gene") %>%
  tidyr::pivot_longer(-gene, names_to = "cluster", values_to = "delta_rna") %>%
  mutate(
    gene    = factor(gene, levels = rev(top_genes)),
    cluster = factor(paste0("Cl ", cluster),
                     levels = paste0("Cl ", cluster_levels))
  )

# Cap extreme values for color scale
cap_val <- 3
mat_melt$delta_capped <- pmax(-cap_val, pmin(cap_val, mat_melt$delta_rna))

p_heat <- ggplot(mat_melt, aes(x = cluster, y = gene, fill = delta_capped)) +
  geom_tile(color = "white", linewidth = 0.45) +
  scale_fill_gradient2(
    low = COLORS$down, mid = "white", high = COLORS$up,
    midpoint = 0, limits = c(-cap_val, cap_val), oob = scales::squish,
    na.value = "#D9DDE1",
    name = expression(Delta * " RNA"),
    guide = guide_colorbar(barwidth = 0.65, barheight = 8.5, frame.colour = COLORS$grid)
  ) +
  labs(
    title = paste0(target_gene_name, " KO: Cross-Cluster Differential Effects"),
    x = NULL, y = NULL
  ) +
  theme_pub(base_size = 14) +
  theme(
    axis.text.x      = element_text(size = 13, face = "bold"),
    axis.text.y      = element_text(size = 10.5, face = "italic"),
    panel.grid       = element_blank(),
    plot.title       = element_text(size = 18, face = "bold", hjust = 0.5),
    legend.title     = element_text(size = 11, face = "bold"),
    legend.text      = element_text(size = 10),
    panel.background = element_rect(fill = "white", color = NA)
  )

ggsave(
  file.path(OUTPUT_DIR, "cluster_heatmap.png"),
  p_heat, width = 6.6, height = 11, dpi = 320, bg = "white"
)
cat(sprintf("[OK] Cross-cluster heatmap saved: %s\n",
            file.path(OUTPUT_DIR, "cluster_heatmap.png")))

# ==============================================================================
# 5. Chromosome-wide ATAC accessibility change
# ==============================================================================
cat("\n--- Chromosome-wide ATAC accessibility change ---\n")

atac_changes_path <- file.path(RESULT_DIR, "atac_changes_merged.csv")
if (!file.exists(atac_changes_path)) {
  warning(sprintf(
    "Missing optional ATAC accessibility file: %s; skipping chromosome-wide plot",
    atac_changes_path
  ))
} else {
  atac_changes <- read.csv(atac_changes_path, stringsAsFactors = FALSE)
  required_atac_cols <- c(
    "peak_id", "chr", "start", "end", "delta_accessibility_signed",
    "delta_accessibility", "n_clusters_observed", "z_score", "is_significant"
  )
  missing_atac_cols <- setdiff(required_atac_cols, names(atac_changes))

  if (length(missing_atac_cols) > 0) {
    warning(sprintf(
      "ATAC accessibility file is missing columns (%s); skipping chromosome-wide plot",
      paste(missing_atac_cols, collapse = ", ")
    ))
  } else if (nrow(atac_changes) == 0) {
    warning("ATAC accessibility file has no rows; skipping chromosome-wide plot")
  } else {
    # Keep every source row. Sorting weak segments first lets stronger effects
    # remain visible where dense peaks overlap, without applying significance
    # filtering.
    canonical_chr <- c(paste0("chr", 1:22), "chrX", "chrY")
    observed_chr <- unique(as.character(atac_changes$chr))
    observed_chr <- observed_chr[!is.na(observed_chr) & nzchar(observed_chr)]
    noncanonical_chr <- sort(setdiff(observed_chr, canonical_chr))
    chr_levels <- c(intersect(canonical_chr, observed_chr), noncanonical_chr)

    atac_plot <- atac_changes %>%
      mutate(
        chr = factor(as.character(chr), levels = chr_levels),
        start = as.numeric(start),
        end = as.numeric(end),
        delta_accessibility_signed = as.numeric(delta_accessibility_signed),
        peak_position = (start + end) / 2,
        effect_abs = abs(delta_accessibility_signed)
      ) %>%
      arrange(effect_abs)

    max_effect <- max(atac_plot$effect_abs, na.rm = TRUE)
    if (!is.finite(max_effect) || max_effect == 0) max_effect <- 1
    # Peak spans are sub-pixel at chromosome scale, so use a constant-height
    # bar centered at each peak coordinate. The position is still genomic;
    # only the mark height is expanded for visibility.
    fixed_bar_height <- max(250000, max(atac_plot$end, na.rm = TRUE) * 0.004)
    atac_plot <- atac_plot %>%
      mutate(
        bar_ymin = peak_position - fixed_bar_height / 2,
        bar_ymax = peak_position + fixed_bar_height / 2,
        # A square-root transform keeps small non-zero effects visibly tinted
        # while retaining the signed red/blue direction.
        effect_color = sign(delta_accessibility_signed) *
          sqrt(effect_abs / max_effect)
      )
    effect_breaks <- sort(unique(c(
      -max_effect, -0.10, -0.01, 0, 0.01, 0.10, max_effect
    )))
    effect_breaks <- effect_breaks[effect_breaks >= -max_effect &
                                   effect_breaks <= max_effect]
    effect_break_positions <- sign(effect_breaks) *
      sqrt(abs(effect_breaks) / max_effect)
    effect_break_labels <- ifelse(
      effect_breaks == 0, "0", sprintf("%+.2f", effect_breaks)
    )
    # Use deeper, more saturated endpoints than the general-purpose palette so
    # weak square-root-scaled effects remain legible on the white background.
    atac_opening_color <- "#C62828"
    atac_closing_color <- "#1565C0"
    atac_neutral_color <- "#E2E6EB"
    n_invalid_coords <- sum(
      is.na(atac_plot$chr) | !is.finite(atac_plot$start) |
        !is.finite(atac_plot$end) |
        is.na(atac_plot$delta_accessibility_signed)
    )
    if (n_invalid_coords > 0) {
      warning(sprintf(
        "%d ATAC rows have missing/invalid chromosome, coordinates, or signed effect and will not be drawable",
        n_invalid_coords
      ))
    }

    p_atac_chr <- ggplot(
      atac_plot,
      aes(
        x = chr, xend = chr, y = bar_ymin, yend = bar_ymax,
        color = effect_color
      )
    ) +
      geom_segment(linewidth = 1.4, alpha = 0.92, lineend = "butt",
                   na.rm = TRUE) +
      scale_color_gradient2(
        low = atac_closing_color, mid = atac_neutral_color,
        high = atac_opening_color,
        midpoint = 0, limits = c(-1, 1),
        breaks = effect_break_positions, labels = effect_break_labels,
        oob = scales::squish,
        name = "ATAC change (signed Δ)\n√ intensity; closing ← 0 → opening",
        guide = guide_colorbar(
          barheight = 8.5, barwidth = 0.7, frame.colour = COLORS$grid
        )
      ) +
      labs(
        title = paste0(target_gene_name, " KO — ATAC Accessibility Change by Chromosome"),
        x = "Chromosome", y = "Genomic coordinate (bp)"
      ) +
      theme_pub(base_size = 14) +
      theme(
        plot.title = element_text(size = 18, hjust = 0.5),
        axis.text.x = element_text(face = "bold", angle = 45, hjust = 1),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        legend.title = element_text(size = 10.5, face = "bold"),
        legend.text = element_text(size = 10)
      )

    atac_chr_file <- file.path(OUTPUT_DIR, "atac_accessibility_chromosome.png")
    ggsave(
      atac_chr_file, p_atac_chr,
      width = 15, height = 9, dpi = 320, bg = "white"
    )
    cat(sprintf(
      "[OK] Chromosome-wide ATAC accessibility plot saved: %s (%d peaks)\n",
      atac_chr_file, nrow(atac_plot)
    ))

    # --------------------------------------------------------------------------
    # ATAC counterpart to the RNA effect distribution summary above.
    # Each source row represents one peak; no significance filtering is applied.
    atac_distribution <- atac_changes %>%
      mutate(
        delta_accessibility_signed = as.numeric(delta_accessibility_signed),
        change = case_when(
          delta_accessibility_signed > 0.01 ~ "Open",
          delta_accessibility_signed < -0.01 ~ "Close",
          TRUE ~ "Stable"
        ),
        change = factor(change, levels = c("Open", "Close", "Stable"))
      )
    atac_counts <- atac_distribution %>% count(change, .drop = FALSE)
    cat(sprintf(
      "ATAC peak counts: Open=%d, Close=%d, Stable=%d\n",
      atac_counts$n[atac_counts$change == "Open"],
      atac_counts$n[atac_counts$change == "Close"],
      atac_counts$n[atac_counts$change == "Stable"]
    ))
    atac_median <- median(
      atac_distribution$delta_accessibility_signed, na.rm = TRUE
    )

    p_atac_hist <- ggplot(
      atac_distribution,
      aes(x = delta_accessibility_signed)
    ) +
      geom_histogram(
        aes(fill = after_stat(x >= 0)), bins = 80,
        color = "white", alpha = 0.88
      ) +
      scale_fill_manual(
        values = c(`TRUE` = COLORS$up, `FALSE` = COLORS$down),
        guide = "none"
      ) +
      geom_vline(xintercept = 0, linewidth = 0.5, color = COLORS$ink) +
      geom_vline(
        xintercept = atac_median,
        linetype = "dashed", color = COLORS$up, linewidth = 0.8
      ) +
      annotate("text", x = 0, y = Inf, label = "zero", vjust = 1.4,
               hjust = 1.15, size = 3.4, color = COLORS$ink) +
      annotate(
        "text", x = atac_median, y = Inf,
        label = sprintf("median = %.3f", atac_median),
        vjust = 2.8, hjust = -0.08, size = 3.4, color = COLORS$up
      ) +
      labs(
        x = expression(Delta * " ATAC accessibility (signed)"),
        y = "Number of peaks"
      ) +
      theme_pub(base_size = 16) +
      theme(panel.grid.major.x = element_blank())

    p_atac_bar <- atac_counts %>%
      ggplot(aes(x = change, y = n, fill = change)) +
      geom_col(color = "white", linewidth = 0.6, width = 0.66) +
      geom_text(aes(label = n), vjust = -0.45, size = 4.9,
                fontface = "bold", color = COLORS$ink) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
      scale_fill_manual(
        values = c("Open" = COLORS$up, "Close" = COLORS$down,
                   "Stable" = COLORS$stable),
        guide = "none"
      ) +
      labs(x = NULL, y = "Number of peaks") +
      theme_pub(base_size = 16) +
      theme(
        panel.grid.major.x = element_blank(),
        axis.text.x = element_text(angle = 30, hjust = 1),
        plot.title = element_blank()
      )

    p_atac_distribution <- p_atac_hist + p_atac_bar +
      plot_layout(widths = c(2, 1)) +
      plot_annotation(
        title = paste0(target_gene_name,
                       " KO: ATAC Accessibility Effect Distribution"),
        theme = theme(
          plot.title = element_text(
            face = "bold", size = 18, hjust = 0.5, color = COLORS$ink
          )
        )
      )

    atac_distribution_file <- file.path(OUTPUT_DIR, "atac_distribution.png")
    ggsave(
      atac_distribution_file, p_atac_distribution,
      width = 14, height = 5.5, dpi = 320, bg = "white"
    )
    cat(sprintf("[OK] ATAC distribution plot saved: %s\n",
                atac_distribution_file))
  }
}

# ==============================================================================
# 6. TF regulatory hierarchy network (target gene → intermediate TFs → genes)
cat("\n--- TF regulatory hierarchy ---\n")

# Identify intermediate TFs: affected_genes that also act as TFs
all_tfs <- unique(tf_weights$tf_gene)
pathways_tf <- pathways %>%
  filter(tf_name %in% all_tfs | affected_gene %in% all_tfs)

# target → intermediate TF: aggregate across all clusters
target_to_tf <- pathways %>%
  filter(tf_name %in% target_gene_names, affected_gene %in% all_tfs,
         !affected_gene %in% target_gene_names) %>%
  group_by(affected_gene) %>%
  summarise(
    total_effect = sum(delta_r_total),
    n_peaks      = n_distinct(peak_id),
    mean_r2      = mean(r2_score_first),
    .groups      = "drop"
  ) %>%
  arrange(desc(abs(total_effect)))

# Rank intermediate TFs by joint target→TF and TF→gene contribution.
# First aggregate every candidate TF's downstream effects, before truncating to top N.
downstream_contribution <- pathways %>%
  filter(tf_name %in% target_to_tf$affected_gene,
         !affected_gene %in% target_gene_names) %>%
  group_by(tf_name, affected_gene) %>%
  summarise(gene_effect = sum(delta_r_total), .groups = "drop") %>%
  group_by(tf_name) %>%
  summarise(
    downstream_contribution = sum(abs(gene_effect)),
    n_downstream_genes = n(),
    .groups = "drop"
  )

target_to_tf <- target_to_tf %>%
  mutate(upstream_contribution = abs(total_effect)) %>%
  left_join(downstream_contribution,
            by = c("affected_gene" = "tf_name")) %>%
  mutate(
    downstream_contribution = coalesce(downstream_contribution, 0),
    n_downstream_genes = coalesce(n_downstream_genes, 0L),
    joint_contribution = sqrt(upstream_contribution * downstream_contribution)
  ) %>%
  arrange(desc(joint_contribution), desc(upstream_contribution), affected_gene)

# Keeping only the strongest joint contributors prevents tiny sunburst sectors
# from becoming unreadable.
TOP_TF_N <- 5
top_tfs <- target_to_tf %>% head(TOP_TF_N)

cat(sprintf("Top %d intermediate TFs by joint contribution: %s\n", TOP_TF_N,
            paste(sprintf("%s (%.4g)", top_tfs$affected_gene,
                          top_tfs$joint_contribution), collapse = ", ")))

# For each top TF, get its top downstream genes
MIN_EFFECT_TF <- 0.005  # remove downstream genes with negligible effect
downstream <- lapply(seq_len(nrow(top_tfs)), function(i) {
  tf <- top_tfs$affected_gene[i]
  pathways %>%
    filter(tf_name == tf, !affected_gene %in% target_gene_names) %>%
    group_by(affected_gene) %>%
    summarise(
      total_effect = sum(delta_r_total),
      mean_r2      = mean(r2_score_first),
      .groups      = "drop"
    ) %>%
    filter(abs(total_effect) > MIN_EFFECT_TF) %>%   # drop negligible-effect genes
    arrange(desc(abs(total_effect))) %>%
    head(4) %>%
    mutate(source_tf = tf)
})

downstream_df <- bind_rows(downstream)

# Remove TFs that lost all downstream genes after effect filtering
top_tfs <- top_tfs %>%
  filter(affected_gene %in% unique(downstream_df$source_tf))

cat(sprintf("Downstream genes: %d unique (after |effect| > %.3f filter)\n",
            length(unique(downstream_df$affected_gene)), MIN_EFFECT_TF))

# --- Draw TF hierarchy sunburst ---
# Center: KO target; inner ring: intermediate TFs; outer ring: downstream genes.
make_color <- function(eff) ifelse(eff > 0, COLORS$up, COLORS$down)
lighten_color <- function(col, amount = 0.45) {
  rgb_mat <- grDevices::col2rgb(col) / 255
  rgb_new <- rgb_mat + (1 - rgb_mat) * amount
  grDevices::rgb(rgb_new[1, ], rgb_new[2, ], rgb_new[3, ])
}
label_angle <- function(ymid) {
  angle <- 90 - 360 * ymid
  ifelse(angle < -90, angle + 180, angle)
}
label_hjust <- function(ymid) {
  angle <- 90 - 360 * ymid
  ifelse(angle < -90, 1, 0)
}
TF_LABEL_MIN_ARC <- 0.070
GENE_LABEL_MIN_ARC <- 0.050

if (nrow(top_tfs) == 0 || nrow(downstream_df) == 0) {
  p_net <- ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = "No TF hierarchy after filtering",
             size = 8, color = COLORS$muted, fontface = "bold") +
    labs(title = paste0(target_gene_name, " KO — TF Regulatory Hierarchy")) +
    theme_void() +
    theme(
      plot.title = element_text(
        face = "bold", size = 20, hjust = 0.5, color = COLORS$ink
      )
    )
} else {
  hierarchy_df <- downstream_df %>%
    group_by(affected_gene) %>%
    mutate(shared_target = n_distinct(source_tf) > 1) %>%
    ungroup() %>%
    mutate(
      gene_effect = total_effect,
      gene_weight = pmax(abs(total_effect), 1e-6)
    ) %>%
    left_join(
      top_tfs %>%
        transmute(
          source_tf = affected_gene,
          tf_effect = total_effect,
          tf_peaks  = n_peaks,
          tf_r2     = mean_r2,
          tf_joint  = joint_contribution
        ),
      by = "source_tf"
    )

	  tf_ring <- hierarchy_df %>%
	    group_by(source_tf) %>%
	    summarise(
	      tf_effect = first(tf_effect),
	      tf_peaks  = first(tf_peaks),
	      tf_r2     = first(tf_r2),
	      tf_joint  = first(tf_joint),
	      ring_weight = pmax(first(tf_joint), 1e-6),
	      n_genes = n(),
	      .groups = "drop"
	    ) %>%
	    arrange(desc(tf_joint), desc(abs(tf_effect)), source_tf) %>%
	    mutate(
	      frac = ring_weight / sum(ring_weight),
	      ymax = cumsum(frac),
      ymin = dplyr::lag(ymax, default = 0),
	      ymid = (ymin + ymax) / 2,
	      arc = ymax - ymin,
	      xmin = 1.22,
	      xmax = 2.05,
	      fill = make_color(tf_effect),
	      label = sprintf("%s\nJ=%.2f", source_tf, tf_joint),
	      angle = 0,
	      hjust = 0.5,
	      label_size = pmax(3.0, pmin(5.0, 5.0 * sqrt(arc / 0.10)))
	    )

	  gene_ring <- hierarchy_df %>%
	    mutate(source_tf = factor(source_tf, levels = tf_ring$source_tf)) %>%
	    arrange(source_tf, desc(gene_weight)) %>%
	    left_join(tf_ring %>% select(source_tf, tf_ymin = ymin, tf_ymax = ymax, tf_fill = fill),
	              by = "source_tf") %>%
	    group_by(source_tf) %>%
	    mutate(
	      gene_rank = row_number(),
	      gene_frac = gene_weight / sum(gene_weight),
      ymax = tf_ymin + cumsum(gene_frac) * (tf_ymax - tf_ymin),
      ymin = tf_ymin + (cumsum(gene_frac) - gene_frac) * (tf_ymax - tf_ymin),
	      ymid = (ymin + ymax) / 2,
	      xmin = 2.05,
	      xmax = 3.05,
	      fill = lighten_color(make_color(gene_effect), 0.42),
	      arc = ymax - ymin,
	      label = ifelse(shared_target, paste0(affected_gene, "*"), affected_gene),
	      angle = 0,
	      hjust = 0.5,
	      label_size = pmax(2.0, pmin(3.2, 3.2 * sqrt(arc / 0.035)))
	    ) %>%
	    ungroup()

	  p_sunburst <- ggplot() +
	    geom_rect(
	      data = data.frame(xmin = 0.00, xmax = 1.22, ymin = 0, ymax = 1),
	      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
	      fill = "white", color = "white", linewidth = 0
	    ) +
	    geom_rect(
	      data = tf_ring,
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = fill),
      color = "white", linewidth = 1.0
    ) +
    geom_rect(
      data = gene_ring,
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = fill),
      color = "white", linewidth = 0.85
    ) +
	    scale_fill_identity() +
	    scale_size_identity() +
	    coord_polar(theta = "y", start = 0, direction = -1, clip = "off") +
		    xlim(0.00, 3.35) +
	    geom_text(
	      data = tf_ring,
	      aes(x = 1.64, y = ymid, label = label, angle = angle, hjust = hjust,
	          size = label_size),
	      color = "white", fontface = "bold", lineheight = 0.90,
	      check_overlap = TRUE
	    ) +
	    geom_text(
	      data = gene_ring,
	      aes(x = 2.58, y = ymid, label = label, angle = angle, hjust = hjust,
	          size = label_size),
	      color = COLORS$ink, fontface = "bold", lineheight = 0.86,
	      check_overlap = TRUE
	    ) +
		    annotate("text", x = 0.00, y = 0,
		             label = paste0(target_gene_name, "\nKO"),
		             size = 6.8, color = COLORS$ink, fontface = "bold", lineheight = 0.9) +
    labs(
      title = paste0(target_gene_name, " KO — TF Regulatory Hierarchy")
    ) +
    theme_void() +
    theme(
      plot.title = element_text(face = "bold", size = 20, hjust = 0.5, color = COLORS$ink),
	      plot.background = element_rect(fill = "white", color = NA),
	      plot.margin = margin(12, 24, 12, 24)
	    )
	  p_net <- p_sunburst
	}

ggsave(
  file.path(OUTPUT_DIR, "tf_hierarchy.png"),
  p_net, width = 12, height = 12, dpi = 320, bg = "white"
)
cat(sprintf("[OK] TF hierarchy network saved: %s\n",
            file.path(OUTPUT_DIR, "tf_hierarchy.png")))

# ==============================================================================
# 7. UMAP of relative KO effect by cluster and pseudotime bin
# ==============================================================================
cat("\n--- UMAP cluster-bin KO effect ---\n")

rna_h5ad_path <- file.path(RESULT_DIR, "rna_clustered.h5ad")
pseudotime_path <- file.path(RESULT_DIR, "intermediate", "pseudotime.csv")

if (file.exists(rna_h5ad_path) && file.exists(pseudotime_path)) {
  if (!HAS_ANNDATA) {
    stop(
      "The UMAP effect plot requires the R package anndata. After installation, set RETICULATE_PYTHON to an environment containing Python anndata."
    )
  }
  adata <- anndata::read_h5ad(rna_h5ad_path)
  obs_df <- as.data.frame(adata$obs)
  cell_barcodes <- rownames(obs_df)

  umap_key <- NULL
  for (candidate in c("X_umap", "umap", "X_UMAP")) {
    if (candidate %in% names(adata$obsm)) {
      umap_key <- candidate
      break
    }
  }
  if (is.null(umap_key)) {
    stop("UMAP coordinates were not found in rna_clustered.h5ad")
  }

  umap_coords <- adata$obsm[[umap_key]][, 1:2]
  umap_df <- data.frame(
    cell_barcode = cell_barcodes,
    UMAP_1 = umap_coords[, 1],
    UMAP_2 = umap_coords[, 2],
    stringsAsFactors = FALSE
  )

  pseudotime_cells <- read.csv(pseudotime_path, stringsAsFactors = FALSE) %>%
    mutate(cluster = as.character(cluster))

  # Find the obs cluster column whose per-cluster counts exactly match the
  # modeled pseudotime table. This avoids assuming a fixed metadata key.
  cluster_candidates <- unique(c(
    "seurat_clusters", "leiden", "louvain",
    grep("cluster|leiden|louvain", names(obs_df),
         value = TRUE, ignore.case = TRUE)
  ))
  cluster_candidates <- cluster_candidates[cluster_candidates %in% names(obs_df)]
  modeled_counts <- table(pseudotime_cells$cluster)
  cluster_col <- NULL
  for (candidate in cluster_candidates) {
    obs_counts <- table(as.character(obs_df[[candidate]]))
    modeled_labels <- names(modeled_counts)
    if (all(modeled_labels %in% names(obs_counts)) &&
        all(as.integer(obs_counts[modeled_labels]) == as.integer(modeled_counts))) {
      cluster_col <- candidate
      break
    }
  }
  if (is.null(cluster_col)) {
    stop("Could not reliably align the pseudotime cluster with an h5ad obs cluster column")
  }

  obs_cluster <- as.character(obs_df[[cluster_col]])
  umap_df$cluster <- obs_cluster

  # Current pipeline CSV omits its cell-barcode index. Reconstruct barcodes
  # using the exact within-cluster cell order used by run.py, with strict count
  # validation above. Future files with cell_barcode are used directly.
  if (!"cell_barcode" %in% names(pseudotime_cells)) {
    pseudotime_cells$cell_barcode <- NA_character_
    for (cl in unique(pseudotime_cells$cluster)) {
      pt_idx <- which(pseudotime_cells$cluster == cl)
      cell_idx <- which(obs_cluster == cl)
      if (length(pt_idx) != length(cell_idx)) {
        stop(sprintf(
          "The pseudotime row count for cluster %s (%d) does not match the h5ad cell count (%d)",
          cl, length(pt_idx), length(cell_idx)
        ))
      }
      pseudotime_cells$cell_barcode[pt_idx] <- cell_barcodes[cell_idx]
    }
  }

  # Pathway records store the first bin in which each persistent downstream
  # effect fires. Reconstruct the cumulative downstream gene effect at every
  # later bin, then sum absolute net effects across genes.
  pathway_effect_col <- if ("delta_r_raw_sum" %in% names(pathways)) {
    "delta_r_raw_sum"
  } else if ("delta_r_raw" %in% names(pathways)) {
    "delta_r_raw"
  } else {
    stop("perturbation_pathways.csv is missing the delta_r_raw(_sum) column")
  }
  pathway_bin_col <- if ("bin_min" %in% names(pathways)) {
    "bin_min"
  } else if ("bin" %in% names(pathways)) {
    "bin"
  } else {
    stop("perturbation_pathways.csv is missing the bin_min/bin column")
  }

  cluster_bins <- pseudotime_cells %>%
    transmute(cluster, bin = as.integer(bin)) %>%
    distinct()

  gene_onsets <- pathways %>%
    mutate(
      cluster = as.character(cluster),
      onset_bin = as.integer(.data[[pathway_bin_col]]),
      pathway_effect = as.numeric(.data[[pathway_effect_col]])
    ) %>%
    filter(!affected_gene %in% target_gene_names, !is.na(onset_bin)) %>%
    group_by(cluster, affected_gene, onset_bin) %>%
    summarise(onset_effect = sum(pathway_effect, na.rm = TRUE),
              .groups = "drop")

  cluster_bin_effect <- cluster_bins %>%
    inner_join(gene_onsets, by = "cluster", relationship = "many-to-many") %>%
    filter(onset_bin <= bin) %>%
    group_by(cluster, bin, affected_gene) %>%
    summarise(gene_effect = sum(onset_effect), .groups = "drop") %>%
    group_by(cluster, bin) %>%
    summarise(
      downstream_abs_sum = sum(abs(gene_effect)),
      n_affected_genes = sum(abs(gene_effect) > 1e-10),
      .groups = "drop"
    ) %>%
    right_join(cluster_bins, by = c("cluster", "bin")) %>%
    mutate(
      downstream_abs_sum = coalesce(downstream_abs_sum, 0),
      n_affected_genes = coalesce(n_affected_genes, 0L),
      relative_effect = if (max(downstream_abs_sum) > 0) {
        downstream_abs_sum / max(downstream_abs_sum)
      } else 0
    ) %>%
    mutate(
      mean_abs_downstream_effect = if_else(
        n_affected_genes > 0,
        downstream_abs_sum / n_affected_genes,
        0
      ),
      mean_relative_effect = if (max(mean_abs_downstream_effect) > 0) {
        mean_abs_downstream_effect / max(mean_abs_downstream_effect)
      } else 0
    )

  umap_effect <- umap_df %>%
    left_join(
      pseudotime_cells %>% select(cell_barcode, cluster, bin),
      by = c("cell_barcode", "cluster")
    ) %>%
    left_join(cluster_bin_effect, by = c("cluster", "bin"))

  # Draw unmodeled cells first in gray, then modeled cells from weak to strong
  # so high-effect regions remain visible.
  modeled_umap <- umap_effect %>%
    filter(!is.na(relative_effect)) %>%
    arrange(relative_effect)
  cluster_labels <- umap_effect %>%
    group_by(cluster) %>%
    summarise(
      UMAP_1 = median(UMAP_1),
      UMAP_2 = median(UMAP_2),
      .groups = "drop"
    )

  p_umap_effect <- ggplot() +
    geom_point(
      data = umap_effect,
      aes(UMAP_1, UMAP_2),
      color = "#D1D5DB", size = 0.55, alpha = 0.65
    ) +
    geom_point(
      data = modeled_umap,
      aes(UMAP_1, UMAP_2, color = relative_effect),
      size = 0.75, alpha = 0.90
    ) +
    geom_label(
      data = cluster_labels,
      aes(UMAP_1, UMAP_2, label = paste0("Cl ", cluster)),
      size = 3.2, fontface = "bold", color = COLORS$ink,
      fill = scales::alpha("white", 0.78), linewidth = 0,
      label.padding = unit(0.10, "lines")
    ) +
    scale_color_gradientn(
      colors = c("#E8EDF2", "#FEE08B", "#F46D43", "#9E0142"),
      limits = c(0, 1),
      name = "Relative\nKO effect",
      guide = guide_colorbar(
        barheight = 7.5, barwidth = 0.65,
        frame.colour = COLORS$grid
      )
    ) +
    coord_equal() +
    labs(
      title = paste0(target_gene_name,
                     " KO: Relative Downstream Effect by Cluster and Bin"),
      x = "UMAP 1", y = "UMAP 2"
    ) +
    theme_pub(base_size = 15) +
    theme(
      plot.title = element_text(size = 18, hjust = 0.5),
      panel.grid = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      legend.title = element_text(size = 11, face = "bold"),
      legend.text = element_text(size = 10)
    )

  umap_effect_file <- file.path(
    OUTPUT_DIR, "umap_cluster_bin_effect.png"
  )
  ggsave(
    umap_effect_file, p_umap_effect,
    width = 9, height = 7.5, dpi = 320, bg = "white"
  )

  modeled_mean_umap <- umap_effect %>%
    filter(!is.na(mean_relative_effect)) %>%
    arrange(mean_relative_effect)
  p_umap_mean_effect <- ggplot() +
    geom_point(
      data = umap_effect,
      aes(UMAP_1, UMAP_2),
      color = "#D1D5DB", size = 0.55, alpha = 0.65
    ) +
    geom_point(
      data = modeled_mean_umap,
      aes(UMAP_1, UMAP_2, color = mean_relative_effect),
      size = 0.75, alpha = 0.90
    ) +
    geom_label(
      data = cluster_labels,
      aes(UMAP_1, UMAP_2, label = paste0("Cl ", cluster)),
      size = 3.2, fontface = "bold", color = COLORS$ink,
      fill = scales::alpha("white", 0.78), linewidth = 0,
      label.padding = unit(0.10, "lines")
    ) +
    scale_color_gradientn(
      colors = c("#E8EDF2", "#FEE08B", "#F46D43", "#9E0142"),
      limits = c(0, 1),
      name = "Mean absolute\ndownstream KO effect\nper affected gene",
      guide = guide_colorbar(
        barheight = 7.5, barwidth = 0.65,
        frame.colour = COLORS$grid
      )
    ) +
    coord_equal() +
    labs(
      title = paste0(
        target_gene_name,
        " KO: Mean Absolute Downstream Effect\nper Affected Gene by Cluster and Bin"
      ),
      x = "UMAP 1", y = "UMAP 2"
    ) +
    theme_pub(base_size = 15) +
    theme(
      plot.title = element_text(size = 16, hjust = 0.5),
      plot.margin = margin(18, 12, 10, 12),
      panel.grid = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      legend.title = element_text(size = 11, face = "bold"),
      legend.text = element_text(size = 10)
    )

  umap_mean_effect_file <- file.path(
    OUTPUT_DIR, "umap_cluster_bin_mean_effect.png"
  )
  ggsave(
    umap_mean_effect_file, p_umap_mean_effect,
    width = 9, height = 7.5, dpi = 320, bg = "white"
  )
  cat(sprintf(
    "[OK] Cluster-bin effect UMAPs saved: %s and %s (%d modeled cells, %d bins)\n",
    umap_effect_file, umap_mean_effect_file,
    nrow(modeled_umap), nrow(cluster_bin_effect)
  ))

  rm(adata, umap_coords)
  gc()
} else {
  warning("rna_clustered.h5ad or intermediate/pseudotime.csv is missing; cannot generate the UMAP effect plot")
}

# ==============================================================================
# Figure 1 — perturbation projection overview
# ===============================================================================
# Show the projection shift on the existing cell embedding; colour carries the
# signed change in pseudotime while source contexts remain separately readable.
projection_file <- file.path(RESULT_DIR, "perturbation_cell_projection.csv")
if (!file.exists(projection_file)) {
  warning(sprintf(
    "%s is missing; skipping the projection overview plot",
    projection_file
  ))
} else {
  projection_df <- read.csv(projection_file, stringsAsFactors = FALSE)
  required_projection_cols <- c(
    "source_context", "source_pseudotime", "projected_pseudotime"
  )
  missing_projection_cols <- setdiff(
    required_projection_cols, names(projection_df)
  )

  if (length(missing_projection_cols) > 0) {
    warning(sprintf(
      "The projection overview is missing columns (%s); skipping the plot",
      paste(missing_projection_cols, collapse = ", ")
    ))
  } else {
    n_projection_total <- nrow(projection_df)
    if ("projection_status" %in% names(projection_df)) {
      projection_df$projection_status <- as.character(
        projection_df$projection_status
      )
    } else {
      warning("The projection_status column does not exist; filtering only by numeric columns and source_context")
      projection_df$projection_status <- "ok"
    }
    projection_df <- projection_df %>%
      mutate(
        source_context = as.character(source_context),
        source_pseudotime = suppressWarnings(as.numeric(source_pseudotime)),
        projected_pseudotime = suppressWarnings(as.numeric(projected_pseudotime)),
        valid_projection =
          nzchar(coalesce(source_context, "")) &
          is.finite(source_pseudotime) & is.finite(projected_pseudotime) &
          !is.na(projection_status) & projection_status == "ok"
      )

    n_projection_excluded <- sum(!projection_df$valid_projection)
    cat(sprintf(
      "Projection overview input: %d rows; excluded %d invalid rows; plotting %d cells\n",
      n_projection_total, n_projection_excluded,
      n_projection_total - n_projection_excluded
    ))

    projection_df <- projection_df %>%
      filter(valid_projection) %>%
      mutate(
        shift = projected_pseudotime - source_pseudotime
      )

    if (nrow(projection_df) == 0) {
      warning("No valid projection rows; skipping the projection overview plot")
    } else {
      if (!"cell_id" %in% names(projection_df)) {
        warning("The projection overview is missing cell_id and cannot be aligned with UMAP coordinates; skipping the plot")
      } else {
        # Reuse the embedding made by the existing UMAP section when present.
        # If that section was skipped, load only the coordinates needed here.
        projection_umap <- NULL
        if (exists("umap_df") &&
            all(c("cell_barcode", "UMAP_1", "UMAP_2") %in% names(umap_df))) {
          projection_umap <- umap_df %>%
            transmute(
              cell_id = as.character(cell_barcode),
              UMAP_1 = as.numeric(UMAP_1),
              UMAP_2 = as.numeric(UMAP_2)
            )
        } else if (HAS_ANNDATA) {
          rna_h5ad_path <- file.path(RESULT_DIR, "rna_clustered.h5ad")
          if (file.exists(rna_h5ad_path)) {
            adata_projection <- anndata::read_h5ad(rna_h5ad_path)
            umap_key_projection <- NULL
            for (candidate in c("X_umap", "umap", "X_UMAP")) {
              if (candidate %in% names(adata_projection$obsm)) {
                umap_key_projection <- candidate
                break
              }
            }
            if (!is.null(umap_key_projection)) {
              umap_coords_projection <- adata_projection$obsm[[
                umap_key_projection
              ]][, 1:2]
              projection_umap <- data.frame(
                cell_id = rownames(as.data.frame(adata_projection$obs)),
                UMAP_1 = as.numeric(umap_coords_projection[, 1]),
                UMAP_2 = as.numeric(umap_coords_projection[, 2]),
                stringsAsFactors = FALSE
              )
            }
            rm(adata_projection)
          }
        }

        if (is.null(projection_umap)) {
          warning(
            "No usable UMAP coordinates (umap_df or UMAP in rna_clustered.h5ad is required); skipping the projection overview plot"
          )
        } else {
          projection_umap <- projection_umap %>%
            filter(
              !is.na(cell_id), nzchar(cell_id),
              is.finite(UMAP_1), is.finite(UMAP_2)
            ) %>%
            distinct(cell_id, .keep_all = TRUE)
          n_projection_before_umap <- nrow(projection_df)
          projection_df <- projection_df %>%
            inner_join(projection_umap, by = "cell_id")
          n_projection_missing_umap <- n_projection_before_umap - nrow(projection_df)
          cat(sprintf(
            "Projection overview UMAP alignment: excluded %d cells without coordinates; plotting %d cells\n",
            n_projection_missing_umap, nrow(projection_df)
          ))

          if (nrow(projection_df) == 0) {
            warning("No projection rows with matching UMAP coordinates; skipping the plot")
          } else {
      context_labels <- projection_df %>%
        group_by(source_context) %>%
        summarise(
          UMAP_1 = median(UMAP_1),
          UMAP_2 = median(UMAP_2),
          .groups = "drop"
        ) %>%
        mutate(label = paste0("Cl ", source_context))

      shift_limit <- max(abs(projection_df$shift), na.rm = TRUE)
      if (!is.finite(shift_limit) || shift_limit == 0) shift_limit <- 1e-6

      # Build supported trajectory guides from ordered local UMAP centroids.
      # Large centroid jumps and missing bins split paths rather than joining
      # disconnected populations with an invented trajectory.
      if ("source_bin" %in% names(projection_df)) {
        projection_df$guide_bin <- suppressWarnings(
          as.numeric(projection_df$source_bin)
        )
      } else {
        warning("projection file lacks source_bin; deriving 40 pseudotime guide bins")
        projection_df <- projection_df %>%
          group_by(source_context) %>%
          mutate(guide_bin = pmin(39L, floor(source_pseudotime * 40))) %>%
          ungroup()
      }

      guide_centroids <- projection_df %>%
        filter(is.finite(guide_bin), is.finite(UMAP_1), is.finite(UMAP_2)) %>%
        group_by(source_context, guide_bin) %>%
        summarise(
          UMAP_1 = median(UMAP_1),
          UMAP_2 = median(UMAP_2),
          n_cells = n(),
          .groups = "drop"
        ) %>%
        filter(n_cells >= 4) %>%
        arrange(source_context, guide_bin)

      guide_segments <- list()
      guide_id <- 0L
      for (context in unique(guide_centroids$source_context)) {
        context_points <- guide_centroids %>%
          filter(source_context == context) %>%
          arrange(guide_bin)
        if (nrow(context_points) < 6) next
        centroid_step <- sqrt(
          diff(context_points$UMAP_1)^2 + diff(context_points$UMAP_2)^2
        )
        step_median <- median(centroid_step, na.rm = TRUE)
        step_mad <- mad(centroid_step, na.rm = TRUE)
        step_cutoff <- max(
          quantile(centroid_step, 0.90, na.rm = TRUE, names = FALSE),
          step_median + 4 * step_mad,
          na.rm = TRUE
        )
        if (!is.finite(step_cutoff)) step_cutoff <- Inf
        breaks <- which(
          diff(context_points$guide_bin) > 1 | centroid_step > step_cutoff
        )
        segment_id <- cumsum(c(TRUE, seq_len(nrow(context_points) - 1) %in% breaks))
        for (segment in split(context_points, segment_id)) {
          if (nrow(segment) < 6) next
          guide_id <- guide_id + 1L
          t <- seq_len(nrow(segment))
          n_smooth <- max(40L, nrow(segment) * 5L)
          smooth_x <- tryCatch(
            smooth.spline(t, segment$UMAP_1, spar = 0.55),
            error = function(e) NULL
          )
          smooth_y <- tryCatch(
            smooth.spline(t, segment$UMAP_2, spar = 0.55),
            error = function(e) NULL
          )
          t_out <- seq(min(t), max(t), length.out = n_smooth)
          if (is.null(smooth_x) || is.null(smooth_y)) {
            x_out <- approx(t, segment$UMAP_1, xout = t_out)$y
            y_out <- approx(t, segment$UMAP_2, xout = t_out)$y
          } else {
            x_out <- predict(smooth_x, t_out)$y
            y_out <- predict(smooth_y, t_out)$y
          }
          guide_segments[[length(guide_segments) + 1L]] <- data.frame(
            guide_id = paste0(as.character(context), "_", guide_id),
            source_context = as.character(context),
            guide_order = seq_along(t_out),
            UMAP_1 = x_out,
            UMAP_2 = y_out,
            support_cells = sum(segment$n_cells),
            n_bins = nrow(segment),
            stringsAsFactors = FALSE
          )
        }
      }
      trajectory_guides <- if (length(guide_segments)) {
        bind_rows(guide_segments)
      } else {
        data.frame(
          guide_id = character(), source_context = character(),
          guide_order = integer(), UMAP_1 = numeric(), UMAP_2 = numeric(),
          support_cells = integer(), n_bins = integer()
        )
      }

      # Retain only major, spatially distinct guides.  The support threshold
      # removes small context-specific fragments; the greedy separation step
      # keeps one dominant trunk and at most two genuine spatial branches.
      if (nrow(trajectory_guides)) {
        guide_summary <- trajectory_guides %>%
          group_by(guide_id, source_context) %>%
          summarise(
            support_cells = first(support_cells),
            n_bins = first(n_bins),
            center_x = median(UMAP_1),
            center_y = median(UMAP_2),
            .groups = "drop"
          ) %>%
          filter(support_cells >= max(100, 0.05 * nrow(projection_df))) %>%
          arrange(desc(support_cells), desc(n_bins))
        umap_span <- max(
          diff(range(projection_df$UMAP_1, na.rm = TRUE)),
          diff(range(projection_df$UMAP_2, na.rm = TRUE))
        )
        guide_separation <- max(0.8, 0.08 * umap_span)
        selected_guides <- character()
        if (nrow(guide_summary)) {
          for (i in seq_len(nrow(guide_summary))) {
            candidate <- guide_summary[i, ]
            sufficiently_separate <- if (!length(selected_guides)) {
              TRUE
            } else {
              chosen <- guide_summary %>%
                filter(guide_id %in% selected_guides)
              all(sqrt(
                (candidate$center_x - chosen$center_x)^2 +
                  (candidate$center_y - chosen$center_y)^2
              ) >= guide_separation)
            }
            if (sufficiently_separate) {
              selected_guides <- c(selected_guides, candidate$guide_id)
            }
            if (length(selected_guides) >= 3L) break
          }
        }
        trajectory_guides <- trajectory_guides %>%
          filter(guide_id %in% selected_guides)
        guide_summary <- guide_summary %>%
          filter(guide_id %in% selected_guides)
      } else {
        guide_summary <- data.frame()
      }
      arrow_guides <- if (nrow(trajectory_guides)) {
        bind_rows(lapply(split(trajectory_guides, trajectory_guides$guide_id),
          function(path) {
            arrow_idx <- unique(pmax(
              2L,
              pmin(nrow(path), round(seq(
                nrow(path) * 0.25, nrow(path) * 0.82, length.out = 3
              )))
            ))
            data.frame(
              UMAP_1 = path$UMAP_1[arrow_idx - 1L],
              UMAP_2 = path$UMAP_2[arrow_idx - 1L],
              xend = path$UMAP_1[arrow_idx],
              yend = path$UMAP_2[arrow_idx]
            )
          }
        ))
      } else {
        data.frame(
          UMAP_1 = numeric(), UMAP_2 = numeric(),
          xend = numeric(), yend = numeric()
        )
      }
      cat(sprintf(
        "Projection overview guides: %d major spatially distinct paths, %d arrowheads\n",
        n_distinct(trajectory_guides$guide_id), nrow(arrow_guides)
      ))

      p_projection_overview <- ggplot(projection_df)
      p_projection_overview <- p_projection_overview +
        geom_point(
          aes(x = UMAP_1, y = UMAP_2, color = shift),
          size = 0.72, alpha = 0.82
        ) +
        geom_label(
          data = context_labels,
          aes(x = UMAP_1, y = UMAP_2, label = label),
          inherit.aes = FALSE,
          size = 3.2,
          fontface = "bold",
          color = COLORS$ink,
          fill = scales::alpha("white", 0.78),
          linewidth = 0,
          label.padding = grid::unit(0.10, "lines")
        )
      p_projection_overview <- p_projection_overview +
        scale_color_gradient2(
          low = COLORS$down,
          mid = "#F7F7F7",
          high = COLORS$up,
          midpoint = 0,
          limits = c(-shift_limit, shift_limit),
          oob = scales::squish,
          name = "Pseudotime shift\n(projected − source)",
          guide = guide_colorbar(
            barheight = 7.5,
            barwidth = 0.65,
            frame.colour = COLORS$grid
          )
        ) +
        coord_equal() +
        labs(
          title = "Perturbation Projection on UMAP",
          subtitle = "Full embedding; colour shows projected pseudotime − source pseudotime",
          x = "UMAP 1",
          y = "UMAP 2",
          caption = "Clean UMAP view; colour encodes projected pseudotime − source pseudotime"
        ) +
        theme_pub(base_size = 15) +
        theme(
          plot.title = element_text(size = 18, hjust = 0.5),
          plot.subtitle = element_text(color = COLORS$muted, size = 11),
          plot.caption = element_text(
            color = COLORS$muted, size = 9, hjust = 0
          ),
          panel.grid = element_blank(),
          axis.text = element_blank(),
          axis.ticks = element_blank(),
          legend.title = element_text(size = 10, face = "bold"),
          legend.text = element_text(size = 9)
        )

      projection_overview_file <- file.path(
        OUTPUT_DIR, "projection_overview.png"
      )
      ggsave(
        projection_overview_file,
        p_projection_overview,
        width = 9, height = 7.5, dpi = 320, bg = "white"
      )
      cat(sprintf(
        "[OK] Projection overview saved: %s\n",
        projection_overview_file
      ))

      # A separate ordered view makes the trajectory variable explicit without
      # implying that pseudotime displacement is a geometric UMAP vector.
      # Lane order is frozen by cluster id: ordering by the number of valid
      # (non-OOD) cells made each KO produce a different row order, because
      # each KO excludes a different number of out-of-distribution cells.
      context_ids <- unique(as.character(projection_df$source_context))
      context_num <- suppressWarnings(as.numeric(
        sub("^[^0-9]*([0-9]+)$", "\\1", context_ids)
      ))
      context_order <- context_ids[order(is.na(context_num), context_num, context_ids)]
      ordered_df <- projection_df %>%
        mutate(
          context = factor(source_context, levels = rev(context_order)),
          context_y = as.numeric(context)
        ) %>%
        arrange(context, source_pseudotime) %>%
        mutate(
          lane_y = context_y + ((seq_len(n()) %% 7) - 3) / 24
        )
      ordered_lanes <- ordered_df %>%
        group_by(context, context_y) %>%
        summarise(
          x_start = min(source_pseudotime),
          x_end = max(source_pseudotime),
          .groups = "drop"
        )
      p_projection_ordered <- ggplot(ordered_df) +
        geom_segment(
          data = ordered_lanes,
          aes(x = x_start, xend = x_end, y = context_y + 0.28,
              yend = context_y + 0.28),
          inherit.aes = FALSE,
          color = COLORS$muted,
          linewidth = 0.55,
          alpha = 0.7,
          arrow = grid::arrow(
            length = grid::unit(0.10, "inches"), type = "closed"
          )
        ) +
        geom_point(
          aes(x = source_pseudotime, y = lane_y, color = shift),
          size = 0.9,
          alpha = 0.78
        ) +
        scale_color_gradient2(
          low = COLORS$down,
          mid = "#F7F7F7",
          high = COLORS$up,
          midpoint = 0,
          limits = c(-shift_limit, shift_limit),
          oob = scales::squish,
          name = "Pseudotime shift\n(projected − source)",
          guide = guide_colorbar(
            barheight = 7.5,
            barwidth = 0.65,
            frame.colour = COLORS$grid
          )
        ) +
        scale_y_continuous(
          breaks = seq_along(rev(context_order)),
          labels = rev(context_order),
          limits = c(0.45, length(context_order) + 0.55),
          expand = c(0, 0)
        ) +
        labs(
          title = "Perturbation Projection",
          subtitle = "Rows are ordered by cluster ID; arrowheads mark increasing source pseudotime",
          x = "Source pseudotime",
          y = "Cluster",
          caption = "Ordered trajectory view, not UMAP geometry; rows ordered by cluster ID; colour encodes projected − source pseudotime"
        ) +
        theme_pub(base_size = 14) +
        theme(
          plot.title = element_text(size = 18, hjust = 0.5),
          plot.subtitle = element_text(color = COLORS$muted, size = 11),
          plot.caption = element_text(color = COLORS$muted, size = 9, hjust = 0),
          panel.grid.major.y = element_blank(),
          panel.grid.minor = element_blank(),
          axis.text.y = element_text(face = "bold"),
          legend.title = element_text(size = 10, face = "bold"),
          legend.text = element_text(size = 9)
        )
      projection_ordered_file <- file.path(
        OUTPUT_DIR, "projection_pseudotime_ordered.png"
      )
      ggsave(
        projection_ordered_file,
        p_projection_ordered,
        width = 11, height = 7.5, dpi = 320, bg = "white"
      )
      cat(sprintf(
        "[OK] Ordered projection saved: %s\n",
        projection_ordered_file
      ))
          }
        }
      }
    }
  }
}

cat(sprintf("\nAll figures saved to: %s/\n", OUTPUT_DIR))
