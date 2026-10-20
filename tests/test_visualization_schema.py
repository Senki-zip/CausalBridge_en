"""Visualization-only contracts; fixtures never require HVG or model fitting."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_r_identity_dedup_keeps_last_valid_label(tmp_path):
    rscript = shutil.which('Rscript')
    if not rscript:
        pytest.skip('Rscript unavailable')
    code = '''
source("visualize_perturbation.R")
d <- data.frame(gene=c("old","old","old",NA,"other"),
  gene_id=c("G1","G1","G1",NA,"G2"),
  gene_name=c("first","last",NA,"missing","second"))
result <- vis_identity(d, "gene")
stopifnot(identical(result$gene, c("G1","G1","G1",NA,"G2")),
  identical(result$gene_label, d$gene_name),
  identical(vis_label(c("G1","G2")), c("last","second")))
empty <- vis_identity(d[FALSE,], "gene")
stopifnot(nrow(empty)==0, identical(vis_label("G1"), "last"))
'''
    env = dict(os.environ, CAUSALBRIDGE_VIS_ADAPTERS_ONLY='1')
    subprocess.run([rscript, '-e', code], cwd=ROOT, env=env, check=True,
                   capture_output=True, text=True)


def test_r_heatmap_ranking_rewards_signed_sum_balance(tmp_path):
    rscript = shutil.which("Rscript")
    if not rscript:
        pytest.skip("Rscript unavailable")
    script = tmp_path / "heatmap_ranking.R"
    script.write_text(r'''
source("visualize_perturbation.R")
x <- data.frame(affected_gene=rep(c("balanced", "shifted", "zero", "tiny", "sparse"), each=2),
  cluster=rep(c("0", "1"), 5),
  delta_rna=c(-1, 1, -.5, 1.5, 0, 0, -1e-6, 1e-6, -1, 1))
x <- x[!(x$affected_gene == "sparse" & x$cluster == "1"), ]
r <- as.data.frame(vis_heatmap_gene_ranking(x))
rownames(r) <- r$affected_gene
stopifnot(r["balanced", "effect_sum"] == 0, r["shifted", "effect_sum"] == 1,
  r["balanced", "balance_score"] == 1, r["shifted", "balance_score"] == .5,
  r["balanced", "effect_range"] == r["shifted", "effect_range"],
  isTRUE(all.equal(r["balanced", "effect_sd"], r["shifted", "effect_sd"])),
  r["balanced", "heterogeneity_score"] > r["shifted", "heterogeneity_score"],
  r["zero", "heterogeneity_score"] == 0,
  r["tiny", "heterogeneity_score"] < r["shifted", "heterogeneity_score"],
  r["sparse", "coverage"] == .5, r["sparse", "n_observed"] == 1,
  r["sparse", "effect_sum"] == -1,
  all(is.finite(r$heterogeneity_score)))
base <- r$coverage * (r$effect_range + r$effect_sd) * ifelse(r$direction_switch, 1.25, 1)
stopifnot(isTRUE(all.equal(r$heterogeneity_score, base * (1 + r$balance_score))),
  all(r$heterogeneity_score <= 2 * base),
  identical(x$delta_rna, c(-1, 1, -.5, 1.5, 0, 0, -1e-6, 1e-6, -1)))
''')
    result = subprocess.run([rscript, str(script)], cwd=ROOT, text=True,
        capture_output=True, env={**os.environ, "CAUSALBRIDGE_VIS_ADAPTERS_ONLY": "1"},
        timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_r_plot_signed_scales_distribution_and_bin_states(tmp_path):
    """Exercise actual R color mapping and state identity, including missing bins."""
    rscript = shutil.which("Rscript")
    if not rscript:
        pytest.skip("Rscript unavailable")
    script = tmp_path / "plot_contracts.R"
    script.write_text(r'''
source("visualize_perturbation.R")
suppressPackageStartupMessages(library(dplyr))
stopifnot(identical(vis_distribution_change(c(-.10001, -.1, 0, .1, .10001, NA, Inf)),
  c("Down", "Stable", "Stable", "Stable", "Up", NA_character_, NA_character_)))
spec <- vis_heatmap_spec(c(-.2, .8, NA, Inf), "#2166AC", "#B2182B")
stopifnot(isTRUE(all.equal(spec$limits, c(-.2, .8))))
scale <- ggplot2::scale_fill_gradientn(colors=spec$colors, values=spec$values,
  limits=spec$limits, breaks=spec$breaks)
mapped <- toupper(scale$map(c(-.2, -.1, 0, .4, .8)))
stopifnot(mapped[1] == "#2166AC", mapped[3] == "#FFFFFF", mapped[5] == "#B2182B")
stopifnot(identical(mapped[c(2,4)], toupper(scales::gradient_n_pal(
  c("#2166AC", "white", "#B2182B"))(c(.25,.75)))))
rounded <- vis_heatmap_spec(c(-.371, .821), "blue", "red")
stopifnot(rounded$limits[1] <= -.371, rounded$limits[2] >= .821,
  isTRUE(all.equal(rounded$limits, c(-.38, .83))))
for (x in list(c(0,0,NA), c(.03,.2), c(-.3,-.1), c(NA,Inf))) {
  s <- vis_heatmap_spec(x, "blue", "red")
  stopifnot(all(is.finite(s$values)), !anyDuplicated(s$values),
    !anyDuplicated(s$breaks), all(x[is.finite(x)] >= s$limits[1]),
    all(x[is.finite(x)] <= s$limits[2]))
}
stopifnot(identical(vis_cluster_ids(c("cluster_0", "PAGA_A", "my_cluster_0"),
  c("0", "PAGA_A")), c("0", "PAGA_A", "my_cluster_0")))
h <- data.frame(source_context="cluster_0", branch=0, bin=0,
  gene=c("display", "same", "target"), gene_id=c("G1", "G2", "T"),
  control_state=c(1,2,9), state_value=c(1.2,1.6,0),
  state_unit="rna_log1p_cp10k", native_effect_unit="delta_log1p_rna_log1p_cp10k")
cells <- data.frame(cluster="0", bin=0:1)
b <- vis_history_bin_effects(h, cells, "T")
stopifnot(isTRUE(all.equal(b$downstream_abs_sum[1], .6)),
  b$n_affected_genes[1] == 2, b$relative_effect[1] == 1,
  is.na(b$relative_effect[2]), is.na(b$mean_relative_effect[2]))
h$state_value[1:2] <- NA
stopifnot(is.na(vis_history_bin_effects(h, cells, "T")$relative_effect[1]))
h$state_value[1:2] <- h$control_state[1:2]
stopifnot(vis_history_bin_effects(h, cells, "T")$relative_effect[1] == 0)
h2 <- rbind(h, transform(h, branch=1))
stopifnot(inherits(try(vis_history_bin_effects(h2, cells, "T"), silent=TRUE), "try-error"))
branched <- vis_history_bin_effects(h2, transform(cells, branch=0), "T")
stopifnot(nrow(branched) == 2, branched$relative_effect[1] == 0)
''')
    result = subprocess.run([rscript, str(script)], cwd=ROOT, text=True,
        capture_output=True, env={**os.environ, "CAUSALBRIDGE_VIS_ADAPTERS_ONLY": "1"},
        timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


spec = importlib.util.spec_from_file_location("visualization_network", ROOT / "plot_causal_network.py")
assert spec is not None and spec.loader is not None
vis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vis)


@pytest.fixture
def merged():
    return pd.DataFrame({
        "target_gene": ['A_B,+Q"', 'A_B,+Q"', 'A_B,+Q"'],
        "target_gene_id": ["T1,T2"] * 3,
        "target_gene_ids_json": [json.dumps(["T1", "T2"])] * 3,
        "target_gene_names_json": [json.dumps(["A_B,+", 'Q"\\Z'])] * 3,
        "affected_gene": ["DUP [G1]", "DUP [G2]", "BAD"],
        "affected_gene_id": ["G1", "G2", "G3"],
        "affected_gene_name": ["DUP", "DUP", "BAD"],
        "delta_rna": [10., 2., np.nan],
        "delta_rna_signed": [-10., 2., np.nan],
        "l0_log2fc_unclipped": [-18., 2., np.nan],
        "display_clipped": [True, False, False],
        "status": ["ok", "ok", "quality_filtered"],
        "aggregation_scope": ["partial", "full", "unavailable"],
        "aggregation_cell_fraction": [.6, 1., 0.],
        "aggregation_context_exclusions": ['{"other":"unavailable"}', "{}", "{}"],
    })


def test_merged_signed_display_and_partial_coverage(merged):
    data = vis.adapt_effects(merged, merged=True)
    assert list(data.affected_gene) == ["G1", "G2", "G3"]
    assert list(data.affected_gene_label) == ["DUP", "DUP", "BAD"]
    assert list(data.plot_delta[:2]) == [-10., 2.]
    assert data.l0_log2fc_unclipped[0] == -18.
    assert list(data.plot_available) == [True, True, False]
    assert np.isnan(data.plot_delta[2])
    assert "partial" in data.effect_note[0] and "0.6" in data.effect_note[0]
    assert "unavailable" in data.effect_note[0]


def test_combined_json_preserves_escaped_and_delimited_names(merged):
    row = vis.adapt_effects(merged, merged=True).iloc[0]
    assert vis.target_members(row) == (["T1", "T2"], ["A_B,+", 'Q"\\Z'])
    assert vis.target_members(pd.Series({"target_gene": "A_B+C_D"})) == (
        ["A_B", "C_D"], ["A_B", "C_D"])


def test_invalid_json_does_not_silently_split_labels(merged):
    merged.loc[0, "target_gene_ids_json"] = "not JSON"
    with pytest.raises(ValueError):
        vis.target_members(merged.iloc[0])


def test_finite_excluded_effect_is_not_eligible(merged):
    merged.loc[0, "status"] = "quality_filtered"
    result = vis.adapt_effects(merged, merged=True)
    assert not result.plot_available[0]
    assert result.plot_delta[0] == -10.


def test_schema_identifies_merged_effects_independent_of_filename(merged):
    result = vis.adapt_effects(merged)
    assert result.plot_delta[0] == -10.
    assert result.effect_unit[0] == "log2FC (display)"
    empty = vis.adapt_effects(merged.iloc[:0])
    assert empty.empty and "plot_available" in empty


@pytest.mark.parametrize("column,extra,unit", [
    ("delta_log1p_rna_log1p_cp10k", {}, "Δ log1p RNA (CP10k)"),
    ("delta_rna", {"native_effect_unit": "delta_log1p_rna_log1p_cp10k"}, "delta_log1p_rna_log1p_cp10k"),
    ("delta_rna", {"control_mean_cp10k": 4., "perturbed_mean_cp10k": 2.}, "log2FC (CP10k endpoint)"),
    ("delta_rna", {"control_mean_l1": 4., "perturbed_mean_l1": 2.}, "log2FC (RNA state endpoint)"),
    ("endpoint_delta_rna_log2fc", {}, "log2FC (display)"),
    ("delta_rna", {}, "RNA effect (legacy units)"),
])
def test_effect_units_follow_explicit_semantics(column, extra, unit):
    frame = pd.DataFrame([dict(target_gene="A", affected_gene="B", **{column: -.75}, **extra)])
    result = vis.adapt_effects(frame)
    assert result.plot_delta[0] == -.75
    assert result.effect_unit[0] == unit


@pytest.mark.parametrize("flags,expected", [
    (["False", "True", "0", "1"], [False, True, False, True]),
    ([False, True, 0, 1], [False, True, False, True]),
])
def test_fallback_flags_are_values_not_string_truthiness(flags, expected):
    assert list(vis.boolean_flags(pd.Series(flags))) == expected
    with pytest.raises(ValueError, match="invalid is_fallback"):
        vis.boolean_flags(pd.Series(["not false"]))


def test_stable_ids_keep_duplicate_symbols_and_combined_network(tmp_path):
    tf = pd.DataFrame({"tf_gene": ["DUP [T1]", "DUP [T2]"], "tf_gene_id": ["T1", "T2"],
                       "tf_gene_name": ["DUP", "DUP"], "peak_id": ["p1", "p2"], "weight": [.5, -.4]})
    edges = pd.DataFrame({"gene": ["DUP [G1]", "DUP [G2]"], "gene_id": ["G1", "G2"],
                          "gene_name": ["DUP", "DUP"], "peak_id": ["p1", "p2"]})
    tf.to_csv(tmp_path / "tf.csv", index=False)
    edges.to_csv(tmp_path / "edges.csv", index=False)
    graph = vis.build_full_network(tmp_path / "tf.csv", tmp_path / "edges.csv")
    assert graph.has_edge("T1", "p1") and graph.has_edge("T2", "p2")
    assert "G1" in graph and "G2" in graph and "DUP" not in graph
    assert graph.nodes["G1"]["label"] == "DUP"
    subgraph, _ = vis.extract_subgraph(graph, ("T1", "T2"), {"G1", "G2"}, set(),
        vis.identity_frame(tf, ("tf_gene",)), vis.identity_frame(edges, ("gene",)))
    assert subgraph.has_edge("T1", "p1") and subgraph.has_edge("T2", "p2")
    assert ("T1", "T2") not in subgraph
    assert not subgraph.has_edge("T1", "p2") and not subgraph.has_edge("T2", "p1")


def test_agg_plot_smoke_legacy_multiple_targets_and_nonnumeric_clusters(tmp_path, monkeypatch):
    tf = pd.DataFrame({"tf_gene": ["A", "X"], "peak_id": ["pA", "pX"], "weight": [.5, -.4], "cluster": ["alpha"] * 2})
    edges = pd.DataFrame({"gene": ["B", "Y"], "peak_id": ["pA", "pX"], "cluster": ["alpha"] * 2})
    pert = pd.DataFrame({"target_gene": ["A", "X"], "affected_gene": ["B", "Y"],
                         "delta_rna": [-2., 3.], "cluster": ["alpha"] * 2})
    for name, frame in (("tf", tf), ("edges", edges), ("pert", pert)):
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    captured = []
    draw = vis.draw_network
    def capture(ax, graph, target, cluster, delta_map, orphan_set):
        captured.append((target, dict(delta_map), set(orphan_set)))
        return draw(ax, graph, target, cluster, delta_map, orphan_set)
    monkeypatch.setattr(vis, "draw_network", capture)
    vis.main(tmp_path / "pert.csv", tmp_path / "tf.csv", tmp_path / "edges.csv")
    assert all(not orphan for _, _, orphan in captured)
    assert any(delta == {"B": -2.} for _, delta, _ in captured)
    assert (tmp_path / "causal_network_top5.png").is_file()
    assert not any(path.is_dir() for path in tmp_path.iterdir())
    with pytest.raises(ValueError, match="positive"):
        vis.main("missing", "missing", "missing", top_n=0)


def test_agg_merged_combined_ids_status_and_output_dir(tmp_path, monkeypatch, merged):
    merged["is_fallback"] = ["False", "False", "True"]
    merged.to_csv(tmp_path / "custom.csv", index=False)
    pd.DataFrame({"tf_gene": ["A", "B"], "tf_gene_id": ["T1", "T2"],
        "tf_gene_name": ["A", "B"], "peak_id": ["p1", "p2"], "weight": [.5, .4]
    }).to_csv(tmp_path / "tf.csv", index=False)
    pd.DataFrame({"gene": ["DUP [G1]", "DUP [G2]"], "gene_id": ["G1", "G2"],
        "gene_name": ["DUP", "DUP"], "peak_id": ["p1", "p2"]
    }).to_csv(tmp_path / "edges.csv", index=False)
    captured = []
    draw = vis.draw_network
    def capture(ax, graph, target, cluster, delta_map, orphan_set):
        captured.append((graph, target, dict(delta_map), orphan_set))
        return draw(ax, graph, target, cluster, delta_map, orphan_set)
    monkeypatch.setattr(vis, "draw_network", capture)
    output = tmp_path / "plots"
    vis.main(tmp_path / "custom.csv", tmp_path / "tf.csv", tmp_path / "edges.csv", output_dir=output)
    assert captured
    for graph, target, delta, orphan in captured:
        assert target == ("T1", "T2") and not orphan
        assert delta == {"G1": -10., "G2": 2.}
        assert "G3" not in graph
        assert graph.nodes["G1"]["label"] == graph.nodes["G2"]["label"] == "DUP"
        assert graph.nodes["T1"]["label"] == 'A_B,+'
        assert graph.nodes["T2"]["label"] == 'Q"\\Z'
        assert "partial" in graph.graph["effect_note"]
        assert "0.6" in graph.graph["effect_note"]
    assert (output / "causal_network_top5.png").is_file()
    assert not any(path.is_dir() for path in output.iterdir())


def test_r_csv_loading_adapters_and_id_joins(tmp_path, merged):
    rscript = shutil.which("Rscript")
    if not rscript:
        pytest.skip("Rscript unavailable")
    assert rscript is not None
    probe = subprocess.run([rscript, "-e", 'quit(status=if(all(vapply(c("dplyr","jsonlite"), requireNamespace, logical(1), quietly=TRUE))) 0 else 42)'], capture_output=True, text=True)
    if probe.returncode == 42:
        pytest.skip("R adapter fixture requires dplyr and jsonlite")
    assert probe.returncode == 0, probe.stderr
    merged.to_csv(tmp_path / "merged.csv", index=False)
    pd.DataFrame({"target_gene": ["T1"] * 2, "affected_gene": ["DUP [G1]", "DUP [G2]"],
        "affected_gene_id": ["G1", "G2"], "affected_gene_name": ["DUP", "DUP"],
        "delta_rna": [-.25, .5], "native_effect_unit": ["delta_log1p_rna_log1p_cp10k"] * 2,
        "cluster": ["alpha"] * 2}).to_csv(tmp_path / "native.csv", index=False)
    pd.DataFrame({"tf_name": ["DUP [T1]", "DUP [T2]"], "tf_name_id": ["T1", "T2"],
        "tf_name_name": ["DUP"] * 2, "peak_id": ["p1"] * 2, "cluster": ["alpha"] * 2,
        "affected_gene": ["DUP [G1]", "DUP [G2]"], "affected_gene_id": ["G1", "G2"],
        "affected_gene_name": ["DUP"] * 2, "delta_r_raw": [1., -2.], "r2_score": [.8, .9]
    }).to_csv(tmp_path / "pathways.csv", index=False)
    pd.DataFrame({"tf_gene": ["DUP"] * 2, "tf_gene_id": ["T1", "T2"], "tf_gene_name": ["DUP"] * 2,
        "peak_id": ["p1"] * 2, "cluster": ["alpha"] * 2, "weight": [.3, -.7], "tf_lag": [1., 2.]
    }).to_csv(tmp_path / "tf.csv", index=False)
    pd.DataFrame({"gene": ["DUP"] * 2, "gene_id": ["G1", "G2"], "gene_name": ["DUP"] * 2,
        "peak_id": ["p1"] * 2, "cluster": ["alpha"] * 2, "atac_coef": [.4, -.8], "pearson_r": [.6, -.9]
    }).to_csv(tmp_path / "edges.csv", index=False)
    code = f'''
      source({json.dumps(str(ROOT / "visualize_perturbation.R"))})
      root <- {json.dumps(str(tmp_path))}
      load <- function(name) read.csv(file.path(root, name), stringsAsFactors=FALSE)
      m <- vis_effects(load("merged.csv"), merged=TRUE)
      n <- vis_effects(load("native.csv"))
      t <- vis_targets(m)
      stopifnot(identical(t$ids, c("T1", "T2")), length(t$names)==2,
        t$names[1]=="A_B,+", m$delta_rna_signed[1]==-10,
        m$l0_log2fc_unclipped[1]==-18, is.na(m$delta_rna_signed[3]),
        identical(m$plot_available, c(TRUE,TRUE,FALSE)),
        grepl("partial", vis_note(m)), grepl("0.6", vis_note(m)),
        n$delta_rna[1]==-.25, !grepl("log2", n$effect_unit[1]),
        identical(vis_label(c("G1","G2")), c("DUP","DUP")))
      p <- vis_pathways(load("pathways.csv"))
      tf <- vis_identity(load("tf.csv"), "tf_gene")
      pg <- vis_identity(load("edges.csv"), "gene")
      joined <- vis_join_pathways(p, tf, pg)
      stopifnot(nrow(joined)==2, identical(joined$tf_peak_weight, c(.3,-.7)),
        identical(joined$peak_gene_coef, c(.4,-.8)),
        identical(joined$tf_lag, c(1,2)), identical(joined$delta_r_raw_sum,c(1,-2)))
      prefixed <- p
      prefixed$cluster <- "cluster_0"
      numeric_tf <- tf; numeric_tf$cluster <- "0"
      numeric_pg <- pg; numeric_pg$cluster <- "0"
      aligned <- vis_join_pathways(prefixed, numeric_tf, numeric_pg)
      stopifnot(identical(aligned$cluster,c("0","0")),
        identical(aligned$cluster_source_label,c("cluster_0","cluster_0")),
        identical(aligned$tf_peak_weight,c(.3,-.7)),
        identical(aligned$peak_gene_coef,c(.4,-.8)))
      ambiguous <- prefixed; ambiguous$cluster[2] <- "0"
      stopifnot(inherits(try(vis_join_pathways(ambiguous,numeric_tf,numeric_pg),
        silent=TRUE),"try-error"))
      saturation <- load("pathways.csv")
      saturation$delta_r_raw <- NA_real_
      saturation$train_r2 <- c(.7,.8)
      saturation$r2_score <- NULL
      saturation$endpoint_delta_rna_log2fc <- c(-1,2)
      saturation$peak_input_contribution_sum <- c(10,20)
      saturation$pathway_allocation_status <- "unavailable"
      s <- vis_pathways(saturation)
      stopifnot(nrow(s)==2, all(is.na(s$delta_r_raw_sum)),
        !any(s$pathway_plot_available), identical(s$r2_score_first,c(.7,.8)),
        identical(s$delta_rna_log2fc,c(-1,2)))
       # Layout weights use saved downstream peak RNA responses, never upstream
       # TF-to-ATAC inputs (which can have the opposite sign and different units).
      library(dplyr)
      s$tf_input_contribution_mean <- c(-.3, .8)
      s$tf_input_unit <- "delta_normalized_atac_log1p"
       s$pathway_type <- "temporal_event_link"
       s$shared_peak_contribution <- TRUE
       s$peak_input_contribution_mean <- c(.02,-.04)
       s$peak_input_unit <- "delta_log1p_rna_log1p_cp10k"
       input <- vis_layout_pathways(s)
       stopifnot(nrow(input)==2,
         identical(input$plot_pathway_weight,c(.02,-.04)),
         all(input$plot_shared_peak_response),
         all(is.na(input$delta_r_raw_sum)), !any(input$pathway_plot_available),
         identical(vis_layout_pathways(p)$plot_pathway_weight,p$delta_r_raw_sum))
       mixed <- s; mixed$peak_input_unit[2] <- "other_unit"
       stopifnot(is.na(vis_layout_pathways(mixed)$plot_pathway_weight[2]),
         identical(joined$delta_r_raw_sum,c(1,-2)))
      explicit <- vis_effects(data.frame(target_gene="T", affected_gene="G",
        endpoint_delta_rna_log2fc=-2, delta_rna=5))
      endpoint <- vis_effects(data.frame(target_gene="T", affected_gene="G",
        delta_rna=-1, control_mean_cp10k=4, perturbed_mean_cp10k=2))
      failed <- load("merged.csv")
      failed$status[1] <- "quality_filtered"
      failed <- vis_effects(failed, merged=TRUE)
      stopifnot(explicit$delta_rna==-2, grepl("log2",endpoint$effect_unit),
        is.na(failed$delta_rna_signed[1]), !failed$plot_available[1])
      if (requireNamespace("ggplot2",quietly=TRUE) && requireNamespace("patchwork",quietly=TRUE)) {{
        library(ggplot2); library(patchwork)
        expressions <- parse({json.dumps(str(ROOT / 'visualize_perturbation.R'))})
        main <- expressions[[length(expressions)]][[3]]
        for (expression in as.list(main)[-1]) {{
          if (is.call(expression) && identical(expression[[1]],as.name("<-")) &&
              as.character(expression[[2]])[1] %in% c("COLORS","plot_radial_cluster_pathway"))
            eval(expression)
        }}
        edges <- data.frame(target_gene="A",tf_name="A",affected_gene="G",
          peak_id=c("chr1:100-200","chr1:300-400"),cluster=c("0","1"),
          delta_r_raw_sum=c(-.4,.2),tf_peak_weight=c(.5,-.6),
          peak_gene_pearson_r=c(.4,-.2))
        endpoint <- data.frame(target_gene="A",affected_gene="G",delta_rna_signed=-.5)
        contexts <- data.frame(affected_gene="G",cluster=c("0","1"),delta_rna=c(-.5,.5))
        radial <- plot_radial_cluster_pathway("G",edges,contexts,endpoint)[[1]]
        segments <- Filter(function(layer) inherits(layer$geom,"GeomSegment") &&
          isTRUE(layer$aes_params$linewidth==.78),radial$layers)
        curves <- Filter(function(layer) inherits(layer$geom,"GeomCurve"),radial$layers)
        stopifnot(length(segments)==2,length(curves)==2,
          all(vapply(segments,function(layer) layer$aes_params$linewidth==.78,logical(1))),
          all(vapply(segments,function(layer) {{
            arrow <- layer$geom_params$arrow
            if (is.null(arrow)) arrow <- layer$aes_params$arrow
            isTRUE(arrow$type==2)
          }},logical(1))),
          all(vapply(curves,function(layer) layer$aes_params$linewidth==.66,logical(1))),
          inherits(radial$coordinates,"CoordCartesian"),
          COLORS$edge_pos=="#E74C3C",COLORS$edge_neg=="#3498DB",
          radial$labels$title=="A KO — Radial Cluster Pathway: G")
         # Shared native peak RNA responses retain the original radial geometry.
         edges$delta_r_raw_sum <- NA_real_
         edges$tf_input_contribution_mean <- c(.8,-.9)
         edges$tf_input_unit <- "delta_normalized_atac_log1p"
         edges$pathway_type <- "temporal_event_link"
         edges$shared_peak_contribution <- TRUE
         edges$peak_input_contribution_mean <- c(-.4,.2)
         edges$peak_input_unit <- "delta_log1p_rna_log1p_cp10k"
         native <- plot_radial_cluster_pathway("G",edges,contexts,endpoint)[[1]]
        stopifnot(identical(vapply(radial$layers,function(layer) class(layer$geom)[1],character(1)),
          vapply(native$layers,function(layer) class(layer$geom)[1],character(1))),
           is.null(native$labels$caption), is.null(native$labels$subtitle),
           !any(vapply(native$layers,function(layer)
             identical(layer$aes_params$label,"terminal/card fill = actual RNA response"),logical(1))))
        source_text <- readLines({json.dumps(str(ROOT / 'visualize_perturbation.R'))})
        stopifnot(!any(grepl("plot_single_pathway|plot_cluster_pathway_figure|plot_input_pathway|caption[[:space:]]*=|annotate_footer",
          source_text, perl=TRUE)))
      }}
      cat("R schema fixtures passed\\n")
    '''
    result = subprocess.run([rscript, "-e", code], capture_output=True, text=True,
        env={**os.environ, "CAUSALBRIDGE_VIS_ADAPTERS_ONLY": "1"}, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "R schema fixtures passed" in result.stdout
    assert not (tmp_path / "figures_R").exists()


def test_r_user_configuration_precedence_and_gene_selection(tmp_path, merged):
    rscript = shutil.which("Rscript")
    if not rscript:
        pytest.skip("Rscript unavailable")
    merged.to_csv(tmp_path / "merged.csv", index=False)
    code = f'''
      source({json.dumps(str(ROOT / "visualize_perturbation.R"))})
      if (!requireNamespace("dplyr", quietly=TRUE)) quit(status=42)
      Sys.unsetenv(c("CAUSALBRIDGE_RESULT_DIR", "CAUSALBRIDGE_OUTPUT_DIR",
        "CAUSALBRIDGE_PATHWAY_GENES"))
      RESULT_DIR <- "manual/input"
      OUTPUT_DIR <- "manual/output"
      PATHWAY_GENES <- c("G2")
      cfg <- vis_config(character())
      stopifnot(cfg$result_dir=="manual/input", cfg$output_dir=="manual/output",
        identical(cfg$pathway_genes, "G2"))
      cfg <- vis_config(c("positional/input"), output_dir=NULL)
      stopifnot(cfg$output_dir=="positional/input/figures_R")
      Sys.setenv(CAUSALBRIDGE_RESULT_DIR="env/input", CAUSALBRIDGE_OUTPUT_DIR="env/output",
        CAUSALBRIDGE_PATHWAY_GENES="DUP,G1")
      cfg <- vis_config(character())
      stopifnot(cfg$result_dir=="env/input", cfg$output_dir=="env/output",
        identical(cfg$pathway_genes, c("DUP","G1")))
      cfg <- vis_config(c("positional/input", "--result-dir", "cli/input",
        "--output-dir", "cli/output", "--genes", "G2"))
      stopifnot(cfg$result_dir=="cli/input", cfg$output_dir=="cli/output",
        identical(cfg$pathway_genes, "G2"))
      stopifnot(is.null(vis_config(c("--genes", "__AUTO__"))$pathway_genes))
      Sys.setenv(CAUSALBRIDGE_PATHWAY_GENES="__AUTO__")
      stopifnot(is.null(vis_config(character())$pathway_genes))
      stopifnot(inherits(try(vis_config("--output-dir"), silent=TRUE), "try-error"))
      m <- vis_effects(read.csv({json.dumps(str(tmp_path / 'merged.csv'))}), merged=TRUE)
      # Signed magnitude ranks G1 first; IDs disambiguate duplicate symbols.
      stopifnot(identical(vis_select_pathway_genes(m, top_n=1), "G1"),
        identical(vis_select_pathway_genes(m, "G2"), "G2"),
        identical(vis_select_pathway_genes(m, "DUP"), c("G1","G2")),
        identical(vis_select_pathway_genes(m, top_n=5), c("G1","G2")),
        identical(vis_select_pathway_genes(m, "G3"), "G3"))
      legacy <- vis_effects(data.frame(target_gene="T", affected_gene=c("A","B"),
        delta_rna=c(4,1), delta_rna_signed=c(-4,1)), merged=TRUE)
      stopifnot(identical(vis_select_pathway_genes(legacy, top_n=1), "A"),
        legacy$delta_rna_signed[1]==-4)
      m$status[1] <- "quality_filtered"
      m <- vis_effects(m, merged=TRUE)
      stopifnot(identical(vis_select_pathway_genes(m, top_n=5), "G2"))
      cat("R customization fixtures passed\\n")
    '''
    result = subprocess.run([rscript, "-e", code], capture_output=True, text=True,
        env={**os.environ, "CAUSALBRIDGE_VIS_ADAPTERS_ONLY": "1"}, timeout=60)
    if result.returncode == 42:
        pytest.skip("R selection fixture requires dplyr")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "R customization fixtures passed" in result.stdout
    assert not (tmp_path / "figures_R").exists()
