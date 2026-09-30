#!/usr/bin/env Rscript
# =============================================================================
# Joint multi-TF KO perturbation visualization
#
# Produces the same figure set and filenames as visualize_perturbation.R, while
# requiring a comma- or plus-delimited multi-TF target_gene value.
#
# Usage:
#   Rscript visualize_multi_tf_perturbation.R [result_dir] [--genes GENE1,GENE2]
#
# =============================================================================

# ==============================================================================
# Configuration — modify these variables directly before running
# ==============================================================================

# Directory containing perturbation_results.csv, perturbation_merged.csv,
# perturbation_pathways.csv, tf_peak_weights.csv, and related input files.
RESULT_DIR <- "/home/huangtao/Desktop/huangchengqi/code_1/atac_bridge/results/test/GATA1_KLF1"

# Directory in which all PNG figures will be written. This can be changed to an
# arbitrary path; it does not have to be located inside RESULT_DIR.
OUTPUT_DIR <- file.path(RESULT_DIR, "figures_R")

# Genes shown in pathway_specified.png.
#   c("DLGAP1")       -> draw one specified gene
#   c("DLGAP1", "BEX1") -> draw multiple specified genes in the same PNG
#   NULL              -> automatically select the top five genes by |effect|
PATHWAY_GENES <- c("DLGAP1")

# ==============================================================================
# Command-line overrides and shared implementation
# ==============================================================================

args <- commandArgs(trailingOnly = TRUE)

# Determine whether the caller supplied a result directory. The value following
# --genes is not positional and must be skipped.
positional <- character(0)
i <- 1
while (i <= length(args)) {
  if (args[i] == "--genes" && i < length(args)) {
    i <- i + 2
  } else if (!startsWith(args[i], "--")) {
    positional <- c(positional, args[i])
    i <- i + 1
  } else {
    i <- i + 1
  }
}

script_flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_flag) == 0) {
  stop("Could not determine the path of visualize_multi_tf_perturbation.R")
}
script_path <- normalizePath(sub("^--file=", "", script_flag[1]))
script_dir <- dirname(script_path)

# A positional result directory still overrides RESULT_DIR. When OUTPUT_DIR has
# its default relationship to RESULT_DIR, keep that relationship after override.
output_follows_result <- identical(
  normalizePath(OUTPUT_DIR, mustWork = FALSE),
  normalizePath(file.path(RESULT_DIR, "figures_R"), mustWork = FALSE)
)
if (length(positional) > 0) {
  RESULT_DIR <- positional[1]
  if (output_follows_result) {
    OUTPUT_DIR <- file.path(RESULT_DIR, "figures_R")
  }
}

# The shared implementation now understands a vector of KO TFs. This dedicated
# entry point additionally rejects accidental use with a single-TF result.
Sys.setenv(
  ATAC_BRIDGE_REQUIRE_MULTI_TF = "1",
  ATAC_BRIDGE_RESULT_DIR = RESULT_DIR,
  ATAC_BRIDGE_OUTPUT_DIR = OUTPUT_DIR,
  ATAC_BRIDGE_PATHWAY_GENES = if (is.null(PATHWAY_GENES)) {
    "__AUTO__"
  } else {
    paste(PATHWAY_GENES, collapse = ",")
  }
)

core_script <- file.path(script_dir, "visualize_perturbation.R")
if (!file.exists(core_script)) {
  stop(sprintf("Shared plotting implementation not found: %s", core_script))
}

source(core_script, chdir = TRUE)
