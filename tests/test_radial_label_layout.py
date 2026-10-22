"""Crowded radial labels use their half widths for center-to-center clearance."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_crowded_tf_labels_fit_without_false_overlap(tmp_path):
    rscript = shutil.which('Rscript')
    if not rscript:
        pytest.skip('Rscript unavailable')
    code = r'''
    source(Sys.getenv("VIS_SOURCE"))
    suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(patchwork)})
    expressions <- parse(Sys.getenv("VIS_SOURCE"))
    main <- expressions[[length(expressions)]][[3]]
    for (expr in as.list(main)[-1]) {
      if (is.call(expr) && identical(expr[[1]],as.name("<-")) &&
          as.character(expr[[2]])[1] %in% c("COLORS","plot_radial_cluster_pathway")) eval(expr)
    }
    # Three full TF names in each narrow sector exercise text-size adaptation.
    edges <- bind_rows(lapply(0:4,function(cl) bind_rows(lapply(1:3,function(i) {
      tf <- paste0("LONG_TF_NAME_",i)
      data.frame(target_gene="KO",tf_name=c("KO",tf),affected_gene=c(tf,"G"),
        peak_id=c(paste0("up",i),paste0("down",i)),cluster=as.character(cl),
        delta_r_raw_sum=c(-.2,-.1)/i,tf_peak_weight=.5,peak_gene_pearson_r=.4)
    }))))
    endpoint <- data.frame(target_gene="KO",affected_gene="G",delta_rna_signed=-.5)
    contexts <- data.frame(affected_gene="G",cluster=as.character(0:4),delta_rna=rep(-.5,5))
    plot <- plot_radial_cluster_pathway("G",edges,contexts,endpoint)[[1]]
    layers <- Filter(function(layer) is.data.frame(layer$data) &&
                      "node_type" %in% names(layer$data) && "label_chars" %in% names(layer$data),plot$layers)
    stopifnot(length(layers)==1L)
    labels <- layers[[1]]$data
    stopifnot(all(c("LONG_TF_NAME_1","LONG_TF_NAME_2","LONG_TF_NAME_3") %in% labels$label))
    groups <- split(labels,interaction(labels$cluster,labels$node_type,drop=TRUE))
    caught_old_error <- FALSE
    for (g in groups) if(nrow(g)>1) {
      g <- g[order(g$theta),]
      width <- g$label_chars*.0065*g$label_size/g$label_r
      gap <- diff(g$theta)
      stopifnot(all(gap >= (head(width,-1)+tail(width,-1))/2-1e-9))
      caught_old_error <- caught_old_error || any(gap < head(width,-1)+tail(width,-1)-1e-9)
    }
    stopifnot(caught_old_error)
    '''
    env = dict(os.environ,CAUSALBRIDGE_VIS_ADAPTERS_ONLY='1',VIS_SOURCE=str(ROOT/'visualize_perturbation.R'))
    result = subprocess.run([rscript,'-e',code],env=env,capture_output=True,text=True,timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
