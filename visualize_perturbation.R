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

# ==============================================================================
# 用户配置 / Configuration — modify these variables directly before running
# 优先级：命令行 > 环境变量 > 下方手动配置。
# ==============================================================================
script_file <- grep("^--file=", commandArgs(), value = TRUE)
PROJECT_DIR <- if (length(script_file)) dirname(normalizePath(sub("^--file=", "", script_file[1]))) else getwd()

# 输入结果目录：保留适配前的 test_2/D3 默认值；可改为任意一次运行的目录。
RESULT_DIR <- "/home/huangtao/Desktop/huangchengqi/code_1/CausalBridge/results/test/KLF1_test/"
# 输出图片目录：NULL 表示使用最终输入目录下的 figures_R；也可填写独立路径。
OUTPUT_DIR <- NULL  # e.g. "/path/to/my_figures"

# 通路图下游基因：NULL → 自动按 |delta_rna_signed| 取前 TOP_PATHWAY_N 个。
# 手动选择示例：PATHWAY_GENES <- c("RASSF2", "CBWD2", "CDH1")
# 支持稳定 gene ID 或显示名称；同名基因请用 ID 区分。
# --genes 或 CAUSALBRIDGE_PATHWAY_GENES 可覆盖；__AUTO__ 切换回自动选择。
PATHWAY_GENES <- NULL
TOP_PATHWAY_N <- 5

# TF 层级图：按上下游联合贡献选择 TF，再按下游贡献的绝对值选择基因。
# 这些参数只控制展示，不改变模型或保存的扰动结果。
TOP_TF_N <- 5
TOP_DOWNSTREAM_N <- 4
MIN_EFFECT_TF <- 0.005  # 去掉 |effect| 不超过此值的下游基因

# Source only these adapters with CAUSALBRIDGE_VIS_ADAPTERS_ONLY=1.
VIS_LABELS <- new.env(parent = emptyenv())
vis_label <- function(ids) vapply(as.character(ids), function(id) {
  if (is.na(id)) return(NA_character_)
  if (exists(id, envir = VIS_LABELS, inherits = FALSE)) get(id, envir = VIS_LABELS) else id
}, character(1), USE.NAMES = FALSE)

vis_identity <- function(df, roles) {
  for (role in intersect(roles, names(df))) {
    label <- if (paste0(role, "_name") %in% names(df)) df[[paste0(role, "_name")]] else df[[role]]
    if (paste0(role, "_id") %in% names(df)) {
      ids <- df[[paste0(role, "_id")]]
      use <- !is.na(ids) & nzchar(ids)
      df[[role]][use] <- ids[use]
    }
    df[[paste0(role, "_label")]] <- label
    for (i in seq_len(nrow(df))) if (!is.na(df[[role]][i]) && !is.na(label[i]))
      assign(as.character(df[[role]][i]), as.character(label[i]), envir = VIS_LABELS)
  }
  df
}

vis_targets <- function(df) {
  if (!nrow(df)) return(list(ids = character(), names = character()))
  if ("target_gene_ids_json" %in% names(df) && !is.na(df$target_gene_ids_json[1])) {
    if (!requireNamespace("jsonlite", quietly = TRUE)) stop("JSON target arrays require R package jsonlite")
    ids <- jsonlite::fromJSON(df$target_gene_ids_json[1])
    labels <- if ("target_gene_names_json" %in% names(df))
      jsonlite::fromJSON(df$target_gene_names_json[1]) else vis_label(ids)
    if (!is.character(ids) || !is.character(labels) || length(ids) != length(labels))
      stop("Target JSON arrays must contain matching gene IDs and names")
    return(list(ids = ids, names = labels))
  }
  ids <- unique(trimws(unlist(strsplit(as.character(df$target_gene[1]), "[,+]"))))
  ids <- ids[nzchar(ids)]
  list(ids = ids, names = vis_label(ids))
}

vis_effects <- function(df, merged = FALSE) {
  df <- vis_identity(df, c("target_gene", "affected_gene"))
  candidates <- c(if (merged) "delta_rna_signed", "endpoint_delta_rna_log2fc",
    "delta_rna_log2fc", "delta_log1p_rna_log1p_cp10k", "delta_rna_signed", "delta_rna")
  column <- intersect(candidates, names(df))[1]
  if (is.na(column)) stop("No supported RNA effect column")
  df$plot_delta <- as.numeric(df[[column]])
  unit <- "RNA effect (legacy units)"
  if ((merged && column == "delta_rna_signed") || column %in% c("endpoint_delta_rna_log2fc", "delta_rna_log2fc")) {
    unit <- "log2FC (display)"
  } else if (column == "delta_log1p_rna_log1p_cp10k") {
    unit <- "Δ log1p RNA (CP10k)"
  } else if (all(c("control_mean_cp10k", "perturbed_mean_cp10k") %in% names(df))) {
    unit <- "log2FC (CP10k endpoint)"
  } else if (all(c("control_mean_l1", "perturbed_mean_l1") %in% names(df))) {
    unit <- "log2FC (RNA state endpoint)"
  } else if ("native_effect_unit" %in% names(df)) {
    units <- unique(na.omit(df$native_effect_unit))
    if (length(units) == 1) unit <- units
  }
  df$effect_unit <- rep(unit, nrow(df))
  valid <- is.finite(df$plot_delta)
  if ("status" %in% names(df)) valid <- valid & !is.na(df$status) & df$status == "ok"
  if ("aggregation_scope" %in% names(df)) valid <- valid & (is.na(df$aggregation_scope) | df$aggregation_scope != "unavailable")
  df$plot_available <- valid
  # Aliases for existing plots, never reconstructed global effects.
  df$delta_rna <- ifelse(valid, df$plot_delta, NA_real_)
  if (merged) df$delta_rna_signed <- df$delta_rna
  df
}

vis_note <- function(df, gene = NULL) {
  if (!is.null(gene)) df <- df[df$affected_gene == gene, , drop = FALSE]
  fields <- intersect(c("aggregation_scope", "aggregation_cell_fraction",
    "aggregation_used_contexts", "aggregation_total_contexts", "aggregation_context_exclusions",
    "display_clipped", "status"), names(df))
  paste(vapply(fields, function(f) paste0(f, "=", paste(unique(df[[f]]), collapse = ",")), character(1)), collapse = "; ")
}

vis_pathways <- function(df) {
  df <- vis_identity(df, c("target_gene", "tf_name", "affected_gene"))
  # Saturating links have no allocated RNA effect. Peak-input contributions
  # cannot be substituted for missing RNA pathway contributions.
  aliases <- list(r2_score_first = c("train_r2", "r2_score"),
    tf_lag_first = c("source_tf_lag", "tf_lag"),
    delta_r_raw_sum = c("delta_r_raw"),
    delta_rna_log2fc = c("endpoint_delta_rna_log2fc"))
  for (field in names(aliases)) if (!field %in% names(df)) {
    source <- intersect(aliases[[field]], names(df))[1]
    df[[field]] <- if (!is.na(source)) df[[source]] else rep(NA_real_, nrow(df))
  }
  valid <- is.finite(df$delta_r_raw_sum)
  if ("pathway_allocation_status" %in% names(df))
    valid <- valid & !is.na(df$pathway_allocation_status) & df$pathway_allocation_status != "unavailable"
  if ("status" %in% names(df)) valid <- valid & !is.na(df$status) & df$status == "ok"
  df$delta_r_raw_sum[!valid] <- NA_real_
  df$pathway_plot_available <- valid
  df
}

vis_join_pathways <- function(pathways, tf_weights, peak_gene_edges) {
  # Canonical IDs are installed by vis_identity before any grouping or join.
  for (field in c("atac_coef", "pearson_r")) if (!field %in% names(peak_gene_edges))
    peak_gene_edges[[field]] <- rep(NA_real_, nrow(peak_gene_edges))
  if (!"cluster" %in% names(pathways) && !nrow(pathways)) pathways$cluster <- character()
  for (field in c("tf_name", "affected_gene", "peak_id"))
    if (!field %in% names(pathways) && !nrow(pathways)) pathways[[field]] <- character()
  pathways$cluster <- as.character(pathways$cluster)
  tf_weights$cluster <- as.character(tf_weights$cluster)
  peak_gene_edges$cluster <- as.character(peak_gene_edges$cluster)
  # Event-link exports use cluster_0 while fitted edges / RNA use 0. Align
  # only to an existing reference ID, and reject ambiguous mixed spellings.
  original_cluster <- pathways$cluster
  reference_clusters <- union(tf_weights$cluster, peak_gene_edges$cluster)
  short_cluster <- sub("^cluster_", "", original_cluster)
  remap <- !is.na(original_cluster) & grepl("^cluster_[0-9]+$", original_cluster) &
    !original_cluster %in% reference_clusters & short_cluster %in% reference_clusters
  pathways$cluster[remap] <- short_cluster[remap]
  cluster_map <- unique(data.frame(original = original_cluster, canonical = pathways$cluster))
  if (anyDuplicated(cluster_map$canonical))
    stop("Ambiguous pathway cluster IDs: both prefixed and unprefixed labels refer to one cluster")
  if (any(remap)) pathways$cluster_source_label <- original_cluster
  finite_mean <- function(x) if (any(is.finite(x))) mean(x[is.finite(x)]) else NA_real_
  tf <- dplyr::summarise(dplyr::group_by(tf_weights, tf_gene, peak_id, cluster),
    tf_peak_weight = finite_mean(weight),
    tf_lag_joined = if ("tf_lag" %in% names(tf_weights)) finite_mean(tf_lag) else NA_real_,
    .groups = "drop")
  names(tf)[names(tf) == "tf_gene"] <- "tf_name"
  pg <- dplyr::summarise(dplyr::group_by(peak_gene_edges, peak_id, gene, cluster),
    peak_gene_coef = finite_mean(atac_coef), peak_gene_pearson_r = finite_mean(pearson_r),
    .groups = "drop")
  names(pg)[names(pg) == "gene"] <- "affected_gene"
  pathways <- dplyr::left_join(pathways, tf, by = c("tf_name", "peak_id", "cluster"))
  pathways <- dplyr::left_join(pathways, pg, by = c("peak_id", "affected_gene", "cluster"))
  pathways$tf_lag <- if ("tf_lag_first" %in% names(pathways))
    dplyr::coalesce(as.numeric(pathways$tf_lag_first), pathways$tf_lag_joined) else pathways$tf_lag_joined
  pathways
}

# Adapt saved contributions for the original plotting geometry only. A hybrid
# temporal link carries a shared peak→RNA response, not a TF-allocated response.
# Keep both original allocation columns untouched, including their missingness.
vis_layout_pathways <- function(df) {
  df$plot_pathway_weight <- df$delta_r_raw_sum
  df$plot_hierarchy_weight <- if ("delta_r_total" %in% names(df))
    df$delta_r_total else df$delta_r_raw_sum
  df$plot_shared_peak_response <- rep(FALSE, nrow(df))
  fields <- c("pathway_type", "shared_peak_contribution",
    "peak_input_contribution_mean", "peak_input_unit")
  if (all(fields %in% names(df))) {
    shared <- !is.finite(df$delta_r_raw_sum) &
      !is.na(df$pathway_type) & df$pathway_type == "temporal_event_link" &
      tolower(as.character(df$shared_peak_contribution)) %in% c("true", "1") &
      is.finite(df$peak_input_contribution_mean) &
      !is.na(df$peak_input_unit) &
      df$peak_input_unit == "delta_log1p_rna_log1p_cp10k"
    if ("status" %in% names(df)) shared <- shared & !is.na(df$status) & df$status == "ok"
    df$plot_pathway_weight[shared] <- df$peak_input_contribution_mean[shared]
    df$plot_hierarchy_weight[shared] <- df$peak_input_contribution_mean[shared]
    df$plot_shared_peak_response[shared] <- TRUE
  }
  df
}

vis_config <- function(args = commandArgs(trailingOnly = TRUE),
                       result_dir = RESULT_DIR, output_dir = OUTPUT_DIR,
                       pathway_genes = PATHWAY_GENES) {
  # Resolve overrides without creating directories or loading plotting packages.
  result_dir <- Sys.getenv("CAUSALBRIDGE_RESULT_DIR", unset = result_dir)
  output_env <- Sys.getenv("CAUSALBRIDGE_OUTPUT_DIR", unset = "")
  if (nzchar(output_env)) output_dir <- output_env
  genes_env <- Sys.getenv("CAUSALBRIDGE_PATHWAY_GENES", unset = "")
  parse_genes <- function(x) if (identical(x, "__AUTO__")) NULL else trimws(strsplit(x, ",")[[1]])
  if (nzchar(genes_env)) pathway_genes <- parse_genes(genes_env)
  positional <- character()
  result_arg <- NULL
  i <- 1
  while (i <= length(args)) {
    if (args[i] %in% c("--result-dir", "--output-dir", "--genes")) {
      if (i == length(args)) stop(paste("Missing value for", args[i]))
      if (args[i] == "--result-dir") result_arg <- args[i + 1]
      if (args[i] == "--output-dir") output_dir <- args[i + 1]
      if (args[i] == "--genes") pathway_genes <- parse_genes(args[i + 1])
      i <- i + 2
    } else {
      if (!startsWith(args[i], "--")) positional <- c(positional, args[i])
      i <- i + 1
    }
  }
  if (length(positional)) result_dir <- positional[1]
  if (!is.null(result_arg)) result_dir <- result_arg
  if (is.null(output_dir)) output_dir <- file.path(result_dir, "figures_R")
  list(result_dir = result_dir, output_dir = output_dir, pathway_genes = pathway_genes)
}

vis_select_pathway_genes <- function(merged, specified = NULL, top_n = TOP_PATHWAY_N) {
  if (!is.null(specified)) {
    valid <- merged$affected_gene[merged$affected_gene %in% specified |
      vis_label(merged$affected_gene) %in% specified]
    invalid <- setdiff(specified, c(valid, vis_label(valid)))
    if (length(invalid)) warning(paste("genes not found in data:", paste(invalid, collapse = ", ")))
    if (!length(valid)) stop("None of the specified genes were found in perturbation_merged.csv")
    # Preserve manual selection even when its saved effect is unavailable.
    selected <- dplyr::filter(merged, affected_gene %in% valid)
    return(dplyr::pull(dplyr::arrange(selected, dplyr::desc(abs(delta_rna_signed))), affected_gene))
  }
  if (length(top_n) != 1 || !is.finite(top_n) || top_n < 1 || top_n != as.integer(top_n))
    stop("TOP_PATHWAY_N must be a positive integer")
  available <- dplyr::filter(merged, is.finite(delta_rna_signed))
  dplyr::pull(dplyr::slice_max(available, order_by = abs(delta_rna_signed), n = top_n), affected_gene)
}

if (!identical(Sys.getenv("CAUSALBRIDGE_VIS_ADAPTERS_ONLY"), "1")) {
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
# 应用配置 / Apply environment and command-line overrides to the settings above
# ==============================================================================

resolved_config <- vis_config()
RESULT_DIR <- resolved_config$result_dir
OUTPUT_DIR <- resolved_config$output_dir
specified_genes <- resolved_config$pathway_genes
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# Visual style — presentation-only settings
# ==============================================================================

COLORS <- list(
  up       = "#E74C3C",      # 深红色，激活
  down     = "#3498DB",      # 深蓝色，抑制
  stable   = "#BDC3C7",      # 灰色，稳定
  ink      = "#2C3E50",      # 深色文字
  muted    = "#7F8C8D",      # 浅色文字
  grid     = "#ECF0F1",      # 网格线
  panel    = "#FFFFFF",      # 面板背景
  peak     = "#F8F9FA",      # Peak 节点背景
  target   = "#F39C12",      # Target 节点（橙色）
  target_2 = "#E67E22",      # Target 边框
  tf       = "#9B59B6",      # TF 节点（紫色）
  tf_bg    = "#F5EEF8",      # TF 节点背景
  gene     = "#27AE60",      # Gene 节点（绿色）
  gene_bg  = "#EAFAF1",      # Gene 节点背景
  edge_pos = "#E74C3C",      # 正向边（红色）
  edge_neg = "#3498DB"       # 负向边（蓝色）
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

pert_res <- vis_effects(read.csv(file.path(RESULT_DIR, "perturbation_results.csv")))
cat(sprintf("perturbation_results: %d rows, %d unique genes\n",
            nrow(pert_res), length(unique(pert_res$affected_gene))))

pert_merged <- vis_effects(read.csv(file.path(RESULT_DIR, "perturbation_merged.csv")), merged = TRUE)
cat(sprintf("perturbation_merged: %d rows\n", nrow(pert_merged)))

pathways <- vis_pathways(read.csv(file.path(RESULT_DIR, "perturbation_pathways.csv")))
cat(sprintf("pathways: %d rows\n", nrow(pathways)))

# Load TF→peak weights to get lag information (per TF, per peak, per cluster)
tf_weights <- vis_identity(read.csv(file.path(RESULT_DIR, "tf_peak_weights.csv")), "tf_gene")
cat(sprintf("tf_peak_weights: %d rows\n", nrow(tf_weights)))

# Load Peak→Gene causal edges (atac_coef per peak, gene, cluster)
peak_gene_edges <- vis_identity(read.csv(file.path(RESULT_DIR, "causal_peak_gene_edges.csv")), "gene")
for (field in c("atac_coef", "pearson_r")) if (!field %in% names(peak_gene_edges))
  peak_gene_edges[[field]] <- rep(NA_real_, nrow(peak_gene_edges))
cat(sprintf("peak_gene_edges: %d rows\n", nrow(peak_gene_edges)))

pathways <- vis_join_pathways(pathways, tf_weights, peak_gene_edges)
cat("Joined canonical TF/peak/gene weights and lag into pathways\n")

# Auto-detect one or more jointly perturbed TFs. Combined-KO outputs encode the
# intervention as a comma- or plus-delimited string (for example KLF1,GATA1),
# while pathway tf_name values remain individual gene symbols.
target_gene_names <- vis_targets(pert_merged)$ids
target_gene_name <- paste(vis_targets(pert_merged)$names, collapse = " + ")

if (identical(Sys.getenv("CAUSALBRIDGE_REQUIRE_MULTI_TF"), "1") &&
    length(target_gene_names) < 2) {
  stop("联合 KO 可视化脚本要求 target_gene 至少包含两个 TF")
}

cat(sprintf("Target gene%s: %s\n",
            if (length(target_gene_names) > 1) "s" else "",
            target_gene_name))

# ==============================================================================
# 1. Effect–fit quality plot — x=log2FC, y=mean R² (credibility)
# ==============================================================================
cat("\n--- Computing volcano data ---\n")

# The saved merged endpoint is authoritative, including partial coverage.
delta_df <- pert_merged %>% filter(plot_available) %>%
  select(affected_gene, mean_delta = delta_rna_signed)

# Per-gene mean R² from pathway edges (ATAC→RNA transfer function fit quality)
r2_df <- pathways %>%
  group_by(affected_gene) %>%
  summarise(
    mean_r2 = mean(r2_score_first),
    n_edges  = n(),
    .groups  = "drop"
  )

# Join
volcano_df <- left_join(delta_df, r2_df, by = "affected_gene") %>%
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
                   vis_label(affected_gene), "")
  )

n_up   <- sum(volcano_df$change == "Up")
n_down <- sum(volcano_df$change == "Down")
cat(sprintf("Genes with R² data: %d\n", nrow(volcano_df)))
cat(sprintf("Large-effect genes (|log2FC|>1): Up=%d, Down=%d\n", n_up, n_down))
cat(sprintf("Mean R² range: %.3f – %.3f, median: %.3f\n",
            min(volcano_df$mean_r2), max(volcano_df$mean_r2),
            median(volcano_df$mean_r2)))

# Effect–fit quality plot: x = log2FC, y = mean R²
x_max <- max(c(1, abs(volcano_df$mean_delta)), na.rm = TRUE) * 1.1

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
    x = unique(pert_merged$effect_unit)[1],
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

# 手动指定或按保存的有符号效应绝对值自动选择；内部始终使用稳定 ID。
pathway_genes <- vis_select_pathway_genes(pert_merged, specified_genes)
cat(sprintf("%s genes: %s\n", if (is.null(specified_genes)) "Auto-selected" else "Specified",
            paste(pathway_genes, collapse = ", ")))

# Explicit Cartesian radial hierarchy. Angles are used only to construct
# annular polygons; this deliberately does not use coord_polar().
plot_radial_cluster_pathway <- function(gene_name, pw_df, cluster_res, merged_df,
                                         max_routes_per_cluster = 3) {
  pw_df <- vis_layout_pathways(pw_df)
  epsilon <- 1e-10
  target_names <- vis_targets(merged_df)$ids
  target_label <- paste(vis_targets(merged_df)$names, collapse = " + ")
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
      effect = sum(plot_pathway_weight, na.rm = TRUE),
      contribution = sum(abs(plot_pathway_weight), na.rm = TRUE),
      tf_coef = weighted_direction(tf_peak_weight, plot_pathway_weight),
      gene_coef = weighted_direction(peak_gene_pearson_r, plot_pathway_weight),
      .groups = "drop") %>% filter(is.finite(effect), is.finite(contribution))
  }
  target_edges <- pw_df %>% filter(tf_name %in% target_names,
    affected_gene != gene_name, !is.na(cluster), is.finite(plot_pathway_weight)) %>%
    mutate(cluster = as.character(cluster), tf_name = as.character(tf_name))
  direct <- pw_df %>% filter(affected_gene == gene_name,
    tf_name %in% target_names, !is.na(cluster), is.finite(plot_pathway_weight)) %>%
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
    !tf_name %in% target_names, !is.na(cluster), is.finite(plot_pathway_weight)) %>%
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
      fill <- "white"; border <- COLORS$muted
      status <- "missing"
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
    vis_label(canonical_nodes$full_id), canonical_nodes$display_id)
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
      node_labels$label == vis_label(node_labels$full_id),
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
    paste0(unique(merged_df$effect_unit)[1], " = ", if (is.na(global)) "—" else sprintf("%+.2f", global)))
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
    scale_color_manual(values = c(Positive = COLORS$edge_pos,
      Negative = COLORS$edge_neg, Uncertain = COLORS$muted,
      `#2C3E50` = COLORS$ink, `#7F8C8D` = COLORS$muted,
      `#C0392B` = "#C0392B", `#2874A6` = "#2874A6"), guide = "none") +
    scale_fill_identity() +
    scale_size_identity() +
    coord_fixed(xlim = c(-body_outer_r - .20, body_outer_r + .20),
                 ylim = c(card_bottom - .20, cy + max_body_r + .10), clip = "off") +
    labs(title = paste0(target_label, " KO — Radial Cluster Pathway: ", vis_label(gene_name))) +
    theme_void() + theme(plot.title = element_text(face = "bold", size = 20,
      hjust = .5, color = COLORS$ink), plot.subtitle = element_text(size = 11,
      hjust = .5, color = COLORS$muted), plot.margin = margin(10, 14, 12, 14),
      plot.background = element_rect(fill = "white", color = NA))
  legend_df <- id_table %>% filter(node_type != "intermediate_tf") %>% transmute(
    id = display_id,
    y = rev(seq_len(n())),
    label = paste0(display_id, "  ", role, " · ", vis_label(full_id)))
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
  if (is.null(specified_genes)) paste0("pathway_top", TOP_PATHWAY_N, ".png") else "pathway_specified.png")

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
  scale_y_discrete(labels = vis_label) +
  scale_fill_gradient2(
    low = COLORS$down, mid = "white", high = COLORS$up,
    midpoint = 0, limits = c(-cap_val, cap_val), oob = scales::squish,
    na.value = "#D9DDE1",
    name = expression(Delta * " RNA"),
    guide = guide_colorbar(barwidth = 0.65, barheight = 8.5, frame.colour = COLORS$grid)
  ) +
  labs(
    title = paste0(target_gene_name, " KO: Cross-Cluster Differential Effects"),
    subtitle = unique(pert_res$effect_unit)[1],
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
    "delta_accessibility", "n_clusters_observed", "z_score"
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
          is.na(delta_accessibility_signed) ~ "Missing",
          is.finite(delta_accessibility_signed) &
            delta_accessibility_signed > 0.01 ~ "Open",
          is.finite(delta_accessibility_signed) &
            delta_accessibility_signed < -0.01 ~ "Close",
          TRUE ~ "Stable"
        ),
        change = factor(change, levels = c("Open", "Close", "Stable", "Missing"))
      )
    # Share the exact plotting rules and styling between all and significant peaks.
    save_atac_distribution <- function(atac_distribution, output_filename, title) {
    atac_counts <- atac_distribution %>%
      filter(change %in% c("Open", "Close", "Stable")) %>%
      mutate(change = factor(change, levels = c("Open", "Close", "Stable"))) %>%
      count(change, .drop = FALSE)
    cat(sprintf(
      "ATAC peak counts: Open=%d, Close=%d, Stable=%d\n",
      atac_counts$n[atac_counts$change == "Open"],
      atac_counts$n[atac_counts$change == "Close"],
      atac_counts$n[atac_counts$change == "Stable"]
    ))

    # Stable peaks do not participate in the left histogram; missing and
    # non-finite signed effects are excluded there. Compress the excluded stable
    # interval only for plotting so Open/Close effects meet around zero.
    atac_hist_data <- atac_distribution %>%
      filter(
        change %in% c("Open", "Close"),
        is.finite(delta_accessibility_signed)
      ) %>%
      mutate(
        hist_effect = sign(delta_accessibility_signed) *
          (abs(delta_accessibility_signed) - 0.01)
      )
    atac_median <- if (nrow(atac_hist_data) > 0) {
      median(atac_hist_data$hist_effect)
    } else {
      NA_real_
    }

    p_atac_hist <- ggplot(
      atac_hist_data,
      aes(x = hist_effect)
    ) +
      labs(
        x = expression(Delta * " ATAC accessibility (signed)"),
        y = "Number of peaks"
      ) +
      theme_pub(base_size = 16) +
      theme(panel.grid.major.x = element_blank())

    if (nrow(atac_hist_data) > 0) {
      p_atac_hist <- p_atac_hist +
        geom_histogram(
          aes(fill = after_stat(x >= 0)), bins = 80,
          color = "white", alpha = 0.88
        ) +
        scale_fill_manual(
          values = c(`TRUE` = COLORS$up, `FALSE` = COLORS$down),
          guide = "none"
        )
    }
    if (is.finite(atac_median)) {
      p_atac_hist <- p_atac_hist +
        geom_vline(
          xintercept = atac_median,
          linetype = "dashed", color = COLORS$up, linewidth = 0.8
        ) +
        annotate(
          "text", x = atac_median, y = Inf,
          label = sprintf("median = %.3f", atac_median),
          vjust = 2.8, hjust = -0.08, size = 3.4, color = COLORS$up
        )
    }

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
        title = title,
        theme = theme(
          plot.title = element_text(
            face = "bold", size = 18, hjust = 0.5, color = COLORS$ink
          )
        )
      )

    atac_distribution_file <- file.path(OUTPUT_DIR, output_filename)
    ggsave(
      atac_distribution_file, p_atac_distribution,
      width = 14, height = 5.5, dpi = 320, bg = "white"
    )
    cat(sprintf("[OK] ATAC distribution plot saved: %s\n",
                atac_distribution_file))
    }

    save_atac_distribution(
      atac_distribution, "atac_distribution.png",
      paste0(target_gene_name, " KO: ATAC Accessibility Effect Distribution")
    )

    if (!"is_significant" %in% names(atac_distribution)) {
      warning("ATAC accessibility file is missing is_significant; skipping significant ATAC distribution plot")
    } else {
      # read.csv may infer logical, character, or numeric flags. Missing and
      # unrecognized values are not significant; preserve the source column.
      significant_rows <- tolower(trimws(as.character(
        atac_distribution$is_significant
      ))) %in% c("true", "1", "yes")
      cat(sprintf(
        "Significant ATAC peaks: %d of %d source rows\n",
        sum(significant_rows), nrow(atac_distribution)
      ))
      save_atac_distribution(
        atac_distribution[significant_rows, , drop = FALSE],
        "atac_distribution_significant.png",
        paste0(target_gene_name,
               " KO: Significant ATAC Accessibility Effect Distribution")
      )
    }
  }
}

# ==============================================================================
# 6. TF regulatory hierarchy network (target gene → intermediate TFs → genes)
cat("\n--- TF regulatory hierarchy ---\n")

# Identify intermediate TFs: affected_genes that also act as TFs
hierarchy_pathways <- vis_layout_pathways(pathways)
all_tfs <- unique(tf_weights$tf_gene)
pathways_tf <- hierarchy_pathways %>%
  filter(tf_name %in% all_tfs | affected_gene %in% all_tfs)

# target → intermediate TF: aggregate across all clusters
target_to_tf <- hierarchy_pathways %>%
  filter(tf_name %in% target_gene_names, affected_gene %in% all_tfs,
         !affected_gene %in% target_gene_names) %>%
  group_by(affected_gene) %>%
  summarise(
    total_effect = sum(plot_hierarchy_weight),
    n_peaks      = n_distinct(peak_id),
    mean_r2      = mean(r2_score_first),
    .groups      = "drop"
  ) %>%
  arrange(desc(abs(total_effect)))

# Rank intermediate TFs by joint target→TF and TF→gene contribution.
# First aggregate every candidate TF's downstream effects, before truncating to top N.
downstream_contribution <- hierarchy_pathways %>%
  filter(tf_name %in% target_to_tf$affected_gene,
         !affected_gene %in% target_gene_names) %>%
  group_by(tf_name, affected_gene) %>%
  summarise(gene_effect = sum(plot_hierarchy_weight), .groups = "drop") %>%
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
top_tfs <- target_to_tf %>% head(TOP_TF_N)

cat(sprintf("Top %d intermediate TFs by joint contribution: %s\n", TOP_TF_N,
            paste(sprintf("%s (%.4g)", top_tfs$affected_gene,
                          top_tfs$joint_contribution), collapse = ", ")))

# For each top TF, get its top downstream genes
downstream <- lapply(seq_len(nrow(top_tfs)), function(i) {
  tf <- top_tfs$affected_gene[i]
  hierarchy_pathways %>%
    filter(tf_name == tf, !affected_gene %in% target_gene_names) %>%
    group_by(affected_gene) %>%
    summarise(
      total_effect = sum(plot_hierarchy_weight),
      mean_r2      = mean(r2_score_first),
      .groups      = "drop"
    ) %>%
    filter(abs(total_effect) > MIN_EFFECT_TF) %>%   # drop negligible-effect genes
    arrange(desc(abs(total_effect))) %>%
    head(TOP_DOWNSTREAM_N) %>%
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
    annotate("text", x = 0.5, y = 0.5, label = if (!any(is.finite(pathways$delta_r_raw_sum)))
               "No saved RNA pathway or shared peak response contributions"
             else "No TF hierarchy after filtering",
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
	      label = sprintf("%s\nJ=%.2f", vis_label(source_tf), tf_joint),
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
	      label = ifelse(shared_target, paste0(vis_label(affected_gene), "*"), vis_label(affected_gene)),
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
      "UMAP effect 图需要 R 包 anndata。安装后请设置 RETICULATE_PYTHON 指向包含 Python anndata 的环境。"
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
    stop("rna_clustered.h5ad 中未找到 UMAP 坐标")
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
    stop("无法将 pseudotime cluster 与 h5ad obs 聚类列可靠对齐")
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
          "cluster %s 的 pseudotime 行数 (%d) 与 h5ad 细胞数 (%d) 不一致",
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
    stop("perturbation_pathways.csv 缺少 delta_r_raw(_sum) 列")
  }
  pathway_bin_col <- if ("bin_min" %in% names(pathways)) {
    "bin_min"
  } else if ("bin" %in% names(pathways)) {
    "bin"
  } else {
    stop("perturbation_pathways.csv 缺少 bin_min/bin 列")
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
  warning("缺少 rna_clustered.h5ad 或 intermediate/pseudotime.csv，无法生成 UMAP effect 图")
}

# ==============================================================================
# Figure 1 — perturbation projection overview
# ===============================================================================
# Show the projection shift on the existing cell embedding; colour carries the
# signed change in pseudotime while source contexts remain separately readable.
projection_file <- file.path(RESULT_DIR, "perturbation_cell_projection.csv")
if (!file.exists(projection_file)) {
  warning(sprintf(
    "缺少 %s，跳过 projection overview 图",
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
      "projection overview 缺少列 (%s)，跳过该图",
      paste(missing_projection_cols, collapse = ", ")
    ))
  } else {
    n_projection_total <- nrow(projection_df)
    if ("projection_status" %in% names(projection_df)) {
      projection_df$projection_status <- as.character(
        projection_df$projection_status
      )
    } else {
      warning("projection_status 列不存在；将仅按数值列和 source_context 过滤")
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
      warning("没有有效 projection rows，跳过 projection overview 图")
    } else {
      if (!"cell_id" %in% names(projection_df)) {
        warning("projection overview 缺少 cell_id，无法与 UMAP 坐标对齐；跳过该图")
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
            "缺少可用 UMAP 坐标（需要 umap_df 或 rna_clustered.h5ad 中的 UMAP），跳过 projection overview 图"
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
            warning("没有 projection rows with matching UMAP coordinates，跳过该图")
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
          y = "UMAP 2"
        ) +
        theme_pub(base_size = 15) +
        theme(
          plot.title = element_text(size = 18, hjust = 0.5),
          plot.subtitle = element_text(color = COLORS$muted, size = 11),
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
      context_ids <- unique(as.character(projection_df$source_context))
      context_num <- suppressWarnings(as.numeric(
        sub("^[^0-9]*([0-9]+)$", "\\1", context_ids)
      ))
      context_order <- context_ids[order(is.na(context_num), context_num, context_ids)]
      ordered_df <- projection_df %>%
        mutate(
          context = factor(source_context, levels = rev(context_order)),
          context_y = as.numeric(context),
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
          title = "Perturbation Projection Ordered by Source Pseudotime",
          subtitle = "Cells are arranged by source context; arrowheads mark increasing source pseudotime",
          x = "Source pseudotime",
          y = "Source context"
        ) +
        theme_pub(base_size = 14) +
        theme(
          plot.title = element_text(size = 18, hjust = 0.5),
          plot.subtitle = element_text(color = COLORS$muted, size = 11),
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
}
