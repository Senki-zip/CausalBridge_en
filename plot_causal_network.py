#!/usr/bin/env python3
"""
CausalBridge 四层因果网络可视化
展示完整的扰动传播路径: Target Gene → TF → Peak → Affected Gene

数据来源:
  - tf_peak_weights.csv        : TF → Peak 调控权重 (逐 lag 单特征 Pearson)
  - causal_peak_gene_edges.csv : Peak → Gene 因果边 (Granger)
  - perturbation_results.csv   : Target → Affected 最终效应
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from collections import deque
import os
import json
import argparse
import re

# ==============================================================================
# Color scheme
# ==============================================================================
C_BG          = "#FAFAFA"
C_TARGET      = "#E74C3C"     # Target (KO) gene  — red
C_TF_NODE     = "#E67E22"     # Intermediate TF   — orange
C_PEAK        = "#27AE60"     # Peak               — green
C_AFFECTED    = "#8E44AD"     # Affected gene      — purple
C_ORPHAN      = "#E91E90"     # Orphan fallback    — pink
C_TF_PEAK     = "#D35400"     # TF→peak edge       — dark orange
C_PEAK_GENE   = "#16A085"     # Peak→gene edge     — teal
C_ORPHAN_EDGE = "#E91E90"     # Orphan edge        — pink dashed
C_TEXT        = "#2C3E50"

# ==============================================================================
# Network construction
# ==============================================================================


def identity_frame(frame, roles):
    """Use canonical IDs internally; keep presentation names on each row."""
    frame = frame.copy()
    for role in roles:
        if role not in frame:
            continue
        frame[role + "_label"] = frame.get(role + "_name", frame[role])
        if role + "_id" in frame:
            frame[role] = frame[role + "_id"].where(
                frame[role + "_id"].notna(), frame[role])
    return frame


def target_members(row):
    """JSON is authoritative: symbols may themselves contain delimiters."""
    if pd.notna(row.get("target_gene_ids_json")):
        ids = json.loads(row["target_gene_ids_json"])
        names = json.loads(row.get("target_gene_names_json", row["target_gene_ids_json"]))
        if (not isinstance(ids, list) or not isinstance(names, list) or not ids
                or len(ids) != len(names)
                or not all(isinstance(value, str) and value for value in ids + names)):
            raise ValueError("target gene JSON arrays must have matching lengths")
        return ids, names
    ids = [v.strip() for v in re.split(r"[,+]", str(row["target_gene"])) if v.strip()]
    names = [v.strip() for v in re.split(r"[,+]", str(row.get("target_gene_label", row["target_gene"]))) if v.strip()]
    return ids, names if len(names) == len(ids) else ids


def boolean_flags(values):
    def parse(value):
        if pd.isna(value):
            return None
        text = str(value).lower().strip()
        if text in ("true", "1", "1.0"):
            return True
        if text in ("false", "0", "0.0"):
            return False
        raise ValueError(f"invalid is_fallback flag: {value!r}")
    return values.map(parse)


def adapt_effects(frame, merged=False):
    """Select signed display effects, never replace missing effects with zero."""
    frame = identity_frame(frame, ("target_gene", "affected_gene"))
    merged = merged or any(c in frame for c in (
        "l0_log2fc_unclipped", "aggregation_method", "aggregation_scope"))
    candidates = (["delta_rna_signed"] if merged else []) + [
        "endpoint_delta_rna_log2fc", "delta_rna_log2fc",
        "delta_log1p_rna_log1p_cp10k", "delta_rna_signed", "delta_rna"]
    column = next((c for c in candidates if c in frame), None)
    if column is None:
        raise ValueError("No supported RNA effect column")
    frame["plot_delta"] = pd.to_numeric(frame[column], errors="coerce")
    unit = "RNA effect (legacy units)"
    if column in ("endpoint_delta_rna_log2fc", "delta_rna_log2fc") or (merged and column == "delta_rna_signed"):
        unit = "log2FC (display)"
    elif column == "delta_log1p_rna_log1p_cp10k":
        unit = "Δ log1p RNA (CP10k)"
    elif {"control_mean_cp10k", "perturbed_mean_cp10k"}.issubset(frame):
        unit = "log2FC (CP10k endpoint)"
    elif {"control_mean_l1", "perturbed_mean_l1"}.issubset(frame):
        unit = "log2FC (RNA state endpoint)"
    elif "native_effect_unit" in frame:
        units = frame.native_effect_unit.dropna().unique()
        if len(units) == 1:
            unit = str(units[0])
    frame["effect_unit"] = unit
    valid = np.isfinite(frame.plot_delta)
    if "status" in frame:
        valid &= frame.status.eq("ok")
    if "aggregation_scope" in frame:
        valid &= ~frame.aggregation_scope.eq("unavailable")
    frame["plot_available"] = valid
    frame["effect_note"] = ["; ".join(
        f"{c}={r[c]}" for c in ("aggregation_scope", "aggregation_cell_fraction",
        "aggregation_used_contexts", "aggregation_total_contexts", "aggregation_context_exclusions",
        "display_clipped", "status") if c in r and pd.notna(r[c])) for _, r in frame.iterrows()]
    if "is_fallback" in frame:
        frame["is_fallback"] = boolean_flags(frame.is_fallback)
    return frame


def build_full_network(tf_csv, edges_csv):
    """Build bipartite TF↔peak↔gene network from all available edges."""
    tf = identity_frame(pd.read_csv(tf_csv), ("tf_gene",))
    edges = identity_frame(pd.read_csv(edges_csv), ("gene",))
    return network_from_frames(tf, edges)


def context_rows(frame, cluster):
    if "cluster" in frame and cluster != "global":
        return frame[frame.cluster.astype(str).eq(str(cluster))]
    return frame


def network_from_frames(tf, edges):
    G = nx.DiGraph()

    for _, r in tf.iterrows():
        G.add_node(r["tf_gene"], ntype="tf", label=r["tf_gene_label"])
        G.add_node(r["peak_id"], ntype="peak")
        G.add_edge(r["tf_gene"], r["peak_id"],
                   etype="TF_PEAK", weight=abs(r["weight"]),
                   raw_weight=r["weight"])

    for _, r in edges.iterrows():
        G.add_node(r["peak_id"], ntype="peak")
        G.add_node(r["gene"], ntype="gene", label=r["gene_label"])
        G.add_edge(r["peak_id"], r["gene"],
                   etype="PEAK_GENE",
                   weight=r.get("composite_score", r.get("delta_r2", 0.1)),
                   p_adj=r.get("p_adj", 1.0),
                   is_causal=r.get("is_causal", True))

    return G


def _short_peak_label(peak_id: str, max_len: int = 14) -> str:
    """Abbreviate peak ID for display: chr:start-end → chr:start..end"""
    if len(peak_id) <= max_len:
        return peak_id
    parts = peak_id.replace(":", "-").split("-")
    if len(parts) >= 3:
        # chr:start-end format
        chr_part = parts[0]
        start = parts[1]
        end = parts[-1]
        return f"{chr_part}:{start}-{end}"
    return peak_id[:max_len-2] + ".."


def _truncate_label(name: str, max_len: int = 14) -> str:
    if len(name) <= max_len:
        return name
    return name[:max_len-2] + ".."


def extract_subgraph(G, target_gene, affected_genes, orphan_genes,
                     tf_weights_df, causal_edges_df):
    """
    Extract layered subgraph showing:
      target → peaks → intermediate TFs → peaks → affected genes
      target → orphan fallback → affected genes (dashed)

    Returns (sg, orphan_set) where orphan_set are genes WITHOUT causal path.
    """
    sg = nx.DiGraph()
    sg.graph["target"] = target_gene

    targets = target_gene if isinstance(target_gene, tuple) else (target_gene,)
    sg.add_node(target_gene, ntype="target", layer=0)

    # Precompute lookups
    tf_all = set(tf_weights_df["tf_gene"])
    affected_set = set(affected_genes)
    orphan_set = set(orphan_genes)

    # Fast path: no causal genes, just orphan genes
    if not affected_set:
        for og in orphan_genes:
            sg.add_node(og, ntype="orphan", layer=4)
        return sg, orphan_set

    # Which peaks causally affect our target affected genes
    affected_upstream_peaks = set()
    for ag in affected_genes:
        affected_upstream_peaks |= set(
            causal_edges_df[causal_edges_df["gene"] == ag]["peak_id"]
        )

    # --- Layer 1: peaks regulated by target ---
    tf_peaks = set(tf_weights_df[tf_weights_df["tf_gene"].isin(targets)]["peak_id"])
    peaks_layer1 = set()

    for p in tf_peaks:
        if p not in G:
            continue
        p_downstream = set(causal_edges_df[causal_edges_df["peak_id"] == p]["gene"])
        hits_affected = bool(p_downstream & affected_set)
        downstream_tfs = p_downstream & tf_all
        can_reach = bool(downstream_tfs)
        if hits_affected or can_reach:
            sg.add_node(p, ntype="peak", layer=1)
            sg.add_edge(target_gene, p, etype="TF_PEAK")
            peaks_layer1.add(p)

    # --- Layer 2: genes downstream of layer-1 peaks ---
    genes_layer2 = set()
    peak_gene_map = {}

    for p in peaks_layer1:
        downstream = set(causal_edges_df[causal_edges_df["peak_id"] == p]["gene"])
        for g in downstream:
            if g not in G:
                continue
            if g in affected_set:
                genes_layer2.add(g)
                peak_gene_map.setdefault(p, set()).add(g)
            elif g in tf_all:
                tf_peaks_down = set(tf_weights_df[tf_weights_df["tf_gene"] == g]["peak_id"])
                if tf_peaks_down & affected_upstream_peaks:
                    genes_layer2.add(g)
                    peak_gene_map.setdefault(p, set()).add(g)

    for g in genes_layer2:
        ntype = "tf" if g in tf_all else "gene"
        sg.add_node(g, ntype=ntype, layer=2)

    for p, genes in peak_gene_map.items():
        for g in genes:
            if g in sg:
                edge_data = causal_edges_df[
                    (causal_edges_df["peak_id"] == p) &
                    (causal_edges_df["gene"] == g)
                ]
                if len(edge_data) > 0:
                    row = edge_data.iloc[0]
                    sg.add_edge(p, g, etype="PEAK_GENE",
                                weight=row.get("composite_score", 0.1))

    # --- Layer 3: peaks regulated by layer-2 TFs (only those hitting affected) ---
    layer2_tfs = {g for g in genes_layer2 if g in tf_all}
    peaks_layer3 = set()

    # All peaks upstream of affected genes
    downstream_peak_set = set()
    for ag in affected_genes:
        downstream_peak_set |= set(
            causal_edges_df[causal_edges_df["gene"] == ag]["peak_id"]
        )

    for tf_g in layer2_tfs:
        tf_peaks_2 = set(tf_weights_df[tf_weights_df["tf_gene"] == tf_g]["peak_id"])
        connecting_peaks = tf_peaks_2 & downstream_peak_set
        for p in connecting_peaks:
            if p in G and p not in sg:
                sg.add_node(p, ntype="peak", layer=3)
                sg.add_edge(tf_g, p, etype="TF_PEAK")
                peaks_layer3.add(p)

    # --- Layer 4: affected genes downstream of layer-3 peaks ---
    for p in peaks_layer3:
        downstream = set(causal_edges_df[causal_edges_df["peak_id"] == p]["gene"])
        for g in downstream:
            if g in affected_set and g in G:
                if g not in sg:
                    sg.add_node(g, ntype="affected", layer=4)
                edge_data = causal_edges_df[
                    (causal_edges_df["peak_id"] == p) &
                    (causal_edges_df["gene"] == g)
                ]
                if len(edge_data) > 0:
                    row = edge_data.iloc[0]
                    sg.add_edge(p, g, etype="PEAK_GENE",
                                weight=row.get("composite_score", 0.1))

    # Affected genes reachable from layer 2 or layer 1
    for p in peaks_layer1:
        downstream = set(causal_edges_df[causal_edges_df["peak_id"] == p]["gene"])
        for ag in affected_set:
            if ag in downstream and ag in G:
                if ag not in sg:
                    sg.add_node(ag, ntype="affected", layer=2)
                edge_data = causal_edges_df[
                    (causal_edges_df["peak_id"] == p) &
                    (causal_edges_df["gene"] == ag)
                ]
                if len(edge_data) > 0:
                    row = edge_data.iloc[0]
                    sg.add_edge(p, ag, etype="PEAK_GENE",
                                weight=row.get("composite_score", 0.1))

    # --- Orphan fallback genes: direct dashed edge from target ---
    for og in orphan_genes:
        if og not in sg:
            sg.add_node(og, ntype="orphan", layer=4)
            # No real edge — handled in drawing

    return sg, orphan_set


# ==============================================================================
# Drawing
# ==============================================================================


def _node_style(ntype: str):
    styles = {
        "target":   (C_TARGET,   1800, ""),
        "tf":       (C_TF_NODE,  1100, ""),
        "peak":     (C_PEAK,     500,  ""),
        "gene":     ("#95A5A6",  500,  ""),
        "affected": (C_AFFECTED, 1300, ""),
        "orphan":   (C_ORPHAN,   1200, ""),
    }
    return styles.get(ntype, ("#BDC3C7", 500, ""))


def _layout_adaptive(G):
    """
    Adaptive layered layout with collision-aware vertical spacing.
    """
    # Group nodes by layer
    layers = {}
    for node, data in G.nodes(data=True):
        layer = data.get("layer", 0)
        layers.setdefault(layer, []).append(node)

    pos = {}
    layer_x = {0: 0.05, 1: 0.25, 2: 0.45, 3: 0.62, 4: 0.80}

    for layer_idx in sorted(layers.keys()):
        nodes = layers[layer_idx]
        n = len(nodes)
        x = layer_x.get(layer_idx, 0.5)

        if n == 0:
            continue
        elif n == 1:
            pos[nodes[0]] = (x, 0.50)
        else:
            # Adaptive spacing: more nodes → tighter, but cap at min spacing
            total_span = min(0.85, n * 0.08)
            top = 0.50 + total_span / 2
            bottom = 0.50 - total_span / 2
            positions_y = np.linspace(top, bottom, n)
            for node, y in zip(nodes, positions_y):
                pos[node] = (x, y)

    return pos


def draw_network(ax, G, target_gene, cluster_label, delta_map=None,
                 orphan_set=None):
    """Draw the layered causal network."""
    ax.set_facecolor(C_BG)
    ax.axis("off")

    if G.number_of_nodes() <= 1:
        ax.text(0.5, 0.5, f"No causal subnetwork\ntop genes are orphan fallback",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#95A5A6")
        return

    orphan_set = orphan_set or set()
    pos = _layout_adaptive(G)

    # ── Draw causal edges ──
    for u, v, data in G.edges(data=True):
        etype = data.get("etype", "PEAK_GENE")
        if etype == "TF_PEAK":
            edge_color, lw = C_TF_PEAK, 0.8
        else:
            edge_color, lw = C_PEAK_GENE, 0.8
        weight = data.get("weight", 0.5)
        lw = 0.5 + np.clip(weight * 2.5, 0, 3.5)

        ax.annotate("", xy=pos[v], xytext=pos[u],
                    arrowprops=dict(
                        arrowstyle="->,head_width=0.15,head_length=0.2",
                        color=edge_color, lw=lw, alpha=0.65,
                        connectionstyle="arc3,rad=0.02",
                    ))

    # ── Draw orphan fallback edges (target → orphan gene, dashed) ──
    for node, ndata in G.nodes(data=True):
        if ndata.get("ntype") == "orphan":
            tpos = pos[target_gene]
            opos = pos[node]
            ax.annotate("", xy=opos, xytext=tpos,
                        arrowprops=dict(
                            arrowstyle="->,head_width=0.15,head_length=0.2",
                            color=C_ORPHAN_EDGE, lw=1.0, alpha=0.5,
                            linestyle="dashed",
                            connectionstyle="arc3,rad=0.1",
                        ))

    # ── Draw nodes ──
    drawn_labels = {}  # (x, y) → label to check overlaps

    for node, (x, y) in pos.items():
        ndata = G.nodes[node]
        ntype = ndata.get("ntype", "gene")
        color, size, _ = _node_style(ntype)

        if ntype == "peak":
            w, h = 0.022, 0.015
            rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                                  boxstyle="round,pad=0.008",
                                  facecolor=color, edgecolor="#ffffff",
                                  alpha=0.85, linewidth=0.6, zorder=3)
            ax.add_patch(rect)
        elif ntype == "orphan":
            # Diamond shape for orphan
            diamond = plt.Polygon(
                [(x, y+0.02), (x+0.02, y), (x, y-0.02), (x-0.02, y)],
                facecolor=C_ORPHAN, edgecolor="white", alpha=0.75,
                linewidth=0.8, zorder=3, linestyle="dashed"
            )
            ax.add_patch(diamond)
        else:
            radius = 0.018 if ntype in ("target", "affected") else 0.013
            lw = 1.0 if ntype in ("target", "affected", "tf") else 0.6
            circle = plt.Circle((x, y), radius, facecolor=color,
                                edgecolor="white", alpha=0.9,
                                linewidth=lw, zorder=3)
            ax.add_patch(circle)

        # Label
        if ntype == "peak":
            label = _short_peak_label(node, 13)
            fontsize = 4.2
            fw = "normal"
            offset_y = -0.025
            va = "top"
            rotation = 0  # No rotation to reduce overlap
        elif ntype == "orphan":
            label = _truncate_label(str(ndata.get("label", node)), 12)
            fontsize = 5.5
            fw = "bold"
            offset_y = 0.030
            va = "bottom"
            rotation = 0
        elif ntype in ("target", "affected"):
            label = _truncate_label(str(ndata.get("label", node)), 14)
            fontsize = 5.8
            fw = "bold"
            offset_y = 0.026
            va = "bottom"
            rotation = 0
        else:  # tf, gene
            label = _truncate_label(str(ndata.get("label", node)), 12)
            fontsize = 4.8
            fw = "bold" if ntype == "tf" else "normal"
            offset_y = 0.024
            va = "bottom"
            rotation = 0

        # Append delta for affected and orphan genes
        if ntype in ("affected", "orphan", "gene", "tf") and delta_map and node in delta_map:
            dv = delta_map[node]
            if abs(dv) >= 1.0:
                ds = f"{dv:+.2f}"
            elif abs(dv) >= 0.01:
                ds = f"{dv:+.3f}"
            else:
                ds = f"{dv:+.2e}"
            label += f"\n({ds})"
            fontsize = 5.0

        # Slight horizontal jitter for labels to avoid overlap in same layer
        x_offset = 0.0
        if rotation == 0:
            # Alternate label x-offset for nodes in same layer
            x_offset = 0.004 * (hash(node) % 5 - 2)

        ax.text(x + x_offset, y + offset_y, label, fontsize=fontsize,
                fontweight=fw, ha="center", va=va, color=C_TEXT, zorder=4)

    # ── Layer headers ──
    layer_names = {
        0: "Target", 1: "TF→Peak\n(Pearson)",
        2: "Peak→Gene\n(Granger)", 3: "TF→Peak\n(Pearson)",
        4: "Affected /\nOrphan"
    }
    for li, name in layer_names.items():
        x = {0: 0.05, 1: 0.25, 2: 0.45, 3: 0.62, 4: 0.80}.get(li, 0.5)
        ax.text(x, 1.01, name, ha="center", va="bottom", fontsize=5.0,
                fontweight="bold", color="#7F8C8D", transform=ax.transAxes)

    # ── Title ──
    n_causal = sum(1 for _, d in G.nodes(data=True)
                   if d.get("ntype") == "affected")
    n_orphan = sum(1 for _, d in G.nodes(data=True)
                   if d.get("ntype") == "orphan")
    n_total = n_causal + n_orphan

    title = f"Cluster {cluster_label}  |  {G.nodes[target_gene].get('label', target_gene)} → {n_total} affected"
    if G.graph.get("effect_note"):
        title += "\n" + G.graph["effect_note"]
    if n_orphan > 0:
        title += f" ({n_orphan} orphan fallback)"
    ax.set_title(title, fontsize=9, fontweight="bold", color=C_TEXT, pad=6)

    # ── Legend ──
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_TARGET,
               markersize=7, label="Target (KO)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_TF_NODE,
               markersize=6, label="TF"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_PEAK,
               markersize=5, label="Peak"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_AFFECTED,
               markersize=7, label="Affected (causal)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_ORPHAN,
               markersize=6, label="Orphan fallback"),
        Line2D([0], [0], color=C_TF_PEAK, lw=1.2, label="TF→Peak"),
        Line2D([0], [0], color=C_PEAK_GENE, lw=1.2, label="Peak→Gene"),
        Line2D([0], [0], color=C_ORPHAN_EDGE, lw=1.2, linestyle="dashed",
               label="Orphan"),
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              fontsize=4.5, framealpha=0.7, ncol=4,
              borderpad=0.3, labelspacing=0.15, handletextpad=0.3,
              bbox_to_anchor=(0.5, -0.14))


# ==============================================================================
# Main
# ==============================================================================


def main(pert_csv, tf_csv, edges_csv, top_n=5, output_dir=None):
    if top_n < 1:
        raise ValueError("top_n must be positive")
    for path in (pert_csv, tf_csv, edges_csv):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    pert = adapt_effects(pd.read_csv(pert_csv), merged=Path(pert_csv).name == "perturbation_merged.csv")
    print(f"Excluded {int((~pert.plot_available).sum())} unavailable RNA effects")
    pert = pert[pert.plot_available].copy()
    if "cluster" not in pert:
        pert["cluster"] = "global"
    tf_df = identity_frame(pd.read_csv(tf_csv), ("tf_gene",))
    edges_df = identity_frame(pd.read_csv(edges_csv), ("gene",))

    if output_dir is None:
        output_dir = str(Path(pert_csv).parent)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Building full network...")
    G = build_full_network(tf_csv, edges_csv)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    pert["abs_delta"] = pert["plot_delta"].abs()
    target_genes = pert["target_gene"].unique()
    clusters = sorted(pert["cluster"].dropna().unique(), key=str)

    # Use is_fallback column if available, otherwise compute via BFS
    if "is_fallback" not in pert.columns or pert["is_fallback"].isna().any():
        print("  Computing reachability (is_fallback column not found)...")
        def _reachable_genes(G, target, max_depth=5):
            reached = set()
            queue = deque([(target, 0)])
            visited = {target}
            while queue:
                curr, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                for _, nbr in G.out_edges(curr):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append((nbr, depth + 1))
                        ndata = G.nodes[nbr]
                        if ndata.get("ntype") in ("gene", "tf", "affected"):
                            reached.add(nbr)
                        if ndata.get("ntype") in ("gene", "tf", "target"):
                            for _, nbr2 in G.out_edges(nbr):
                                if nbr2 not in visited:
                                    visited.add(nbr2)
                                    queue.append((nbr2, depth + 1))
            return reached
        if "is_fallback" not in pert:
            pert["is_fallback"] = None
        for tg in target_genes:
            for cl in clusters:
                mask = pert.target_gene.eq(tg) & pert.cluster.eq(cl) & pert.is_fallback.isna()
                if not mask.any():
                    continue
                row = pert[mask].iloc[0]
                local_graph = network_from_frames(context_rows(tf_df, cl), context_rows(edges_df, cl))
                reachable = set().union(*(_reachable_genes(local_graph, member) if member in local_graph else set()
                                          for member in target_members(row)[0]))
                pert.loc[mask, "is_fallback"] = ~pert.loc[mask, "affected_gene"].isin(reachable)
    pert["is_fallback"] = pert["is_fallback"].astype(bool)

    all_plots = []
    for tg in target_genes:
        n_fallback = (pert["is_fallback"] & (pert["target_gene"] == tg)).sum()
        n_causal_tot = (~pert["is_fallback"] & (pert["target_gene"] == tg)).sum()
        print(f"  {tg}: {n_causal_tot} causal + {n_fallback} fallback genes")

        dft = pert[pert["target_gene"] == tg]
        for cl in clusters:
            dfc = dft[dft["cluster"] == cl]
            if dfc.empty:
                continue

            # Top N by |delta| — split by is_fallback flag from CSV
            top = dfc.nlargest(top_n, "abs_delta")
            causal_genes = top[~top["is_fallback"]]
            orphan_genes = top[top["is_fallback"]]

            if causal_genes.empty and orphan_genes.empty:
                continue

            affected_set = set(causal_genes["affected_gene"])
            orphan_set = set(orphan_genes["affected_gene"])
            delta_map = {}
            for _, row in top.iterrows():
                delta_map[row["affected_gene"]] = row["plot_delta"]

            # Build subgraph with both causal and orphan genes
            members, member_names = target_members(dfc.iloc[0])
            target_key = members[0] if len(members) == 1 else tuple(members)
            # Context-specific tables prevent paths assembled across clusters.
            local_tf = context_rows(tf_df, cl)
            local_edges = context_rows(edges_df, cl)
            sg, _ = extract_subgraph(
                G, target_key,
                set(causal_genes["affected_gene"]),
                orphan_set,
                local_tf, local_edges
            )
            labels = dict(zip(pert.affected_gene, pert.affected_gene_label))
            for node in sg:
                sg.nodes[node]["label"] = labels.get(node, G.nodes[node].get("label", node) if node in G else node)
            sg.nodes[target_key]["label"] = " + ".join(member_names)
            sg.graph["effect_note"] = "; ".join(dict.fromkeys(
                list(dfc.effect_unit) + [note for note in dfc.effect_note if note]))

            # Also add orphan nodes if they weren't already added
            # (some may have been added via causal edges if they happen to have partial paths)
            for og in orphan_set:
                if og not in sg:
                    sg.add_node(og, ntype="orphan", layer=4)

            n_causal = len(affected_set)
            n_orphan = len(orphan_set)
            if n_orphan > 0:
                print(f"  Cluster {cl}: {n_causal} causal + {n_orphan} orphan")

            all_plots.append((target_key, cl, sg, delta_map, orphan_set,
                              len(dfc), n_causal, n_orphan))

    if not all_plots:
        print("No connected subgraphs found.")
        return

    print(f"Found {len(all_plots)} valid subgraphs")

    # ── Individual cluster figures (better quality, no overlap) ──
    for tg, cl, sg, delta_map, orphan_set, n_cells, n_causal, n_orphan in all_plots:
        fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=C_BG)
        draw_network(ax, sg, tg, str(cl), delta_map, orphan_set)

        fpath = out / f"causal_network_cluster_{cl}_{tg}_top{top_n}.png"
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        fig.savefig(fpath, dpi=200, bbox_inches="tight",
                    facecolor=C_BG, edgecolor="none")
        plt.close(fig)

        n_peaks = sum(1 for _, d in sg.nodes(data=True) if d.get("ntype") == "peak")
        n_tfs = sum(1 for _, d in sg.nodes(data=True) if d.get("ntype") == "tf")
        n_aff = sum(1 for _, d in sg.nodes(data=True)
                    if d.get("ntype") in ("affected", "orphan"))
        orphan_str = f" ({n_orphan} orphan)" if n_orphan > 0 else ""
        print(f"  Cluster {cl}: {n_tfs} TFs, {n_peaks} peaks, {n_aff} affected"
              f"{orphan_str} — {fpath.name}")

    # ── Combined figure (3 columns to avoid crowding) ──
    cols = 3
    rows = int(np.ceil(len(all_plots) / cols))

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 5.2, rows * 4.5),
                             facecolor=C_BG, squeeze=False)

    for idx, (tg, cl, sg, delta_map, orphan_set, n_cells, n_causal,
              n_orphan) in enumerate(all_plots):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        draw_network(ax, sg, tg, str(cl), delta_map, orphan_set)

    # Hide unused subplots
    for idx in range(len(all_plots), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    orphan_info = ""
    total_orphan = sum(p[-1] for p in all_plots)
    if total_orphan > 0:
        orphan_info = (
            f"\n[Pink diamonds = orphan fallback genes (largest |Δ|, "
            f"no Granger causal path in network)]"
        )

    fig.suptitle(
        f"CausalBridge Causal Network — Target → TF → Peak → Affected Gene\n"
        f"Top {top_n} affected genes per cluster by |ΔRNA| magnitude"
        f"{orphan_info}",
        fontsize=11, fontweight="bold", color=C_TEXT, y=1.005
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    combined_path = out / f"causal_network_top{top_n}.png"
    fig.savefig(combined_path, dpi=200, bbox_inches="tight",
                facecolor=C_BG, edgecolor="none")
    print(f"Combined: {combined_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("top", nargs="?", type=int, default=5)
    parser.add_argument("--result-dir", type=Path, default=Path(os.environ.get(
        "CAUSALBRIDGE_RESULT_DIR", Path(__file__).resolve().parent / "results/test/RUNX1")))
    parser.add_argument("--pert-csv", type=Path)
    parser.add_argument("--tf-csv", type=Path)
    parser.add_argument("--edges-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=os.environ.get("CAUSALBRIDGE_OUTPUT_DIR"))
    options = parser.parse_args()
    if options.top < 1:
        parser.error("top must be positive")
    main(options.pert_csv or options.result_dir / "perturbation_results.csv",
         options.tf_csv or options.result_dir / "tf_peak_weights.csv",
         options.edges_csv or options.result_dir / "causal_peak_gene_edges.csv",
         top_n=options.top, output_dir=options.output_dir)
