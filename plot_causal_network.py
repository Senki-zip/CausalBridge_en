#!/usr/bin/env python3
"""
CausalBridge four-layer causal network visualization
Show the complete perturbation propagation path: Target Gene → TF → Peak → Affected Gene

Data sources:
  - tf_peak_weights.csv        : TF → Peak regulatory weights (single-feature Pearson per lag)
  - causal_peak_gene_edges.csv : Peak → Gene causal edges (Granger)
  - perturbation_results.csv   : final Target → Affected effects
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
import sys

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


def build_full_network(tf_csv, edges_csv):
    """Build bipartite TF↔peak↔gene network from all available edges."""
    tf = pd.read_csv(tf_csv)
    edges = pd.read_csv(edges_csv)

    G = nx.DiGraph()

    for _, r in tf.iterrows():
        G.add_node(r["tf_gene"], ntype="tf")
        G.add_node(r["peak_id"], ntype="peak")
        G.add_edge(r["tf_gene"], r["peak_id"],
                   etype="TF_PEAK", weight=abs(r["weight"]),
                   raw_weight=r["weight"])

    for _, r in edges.iterrows():
        G.add_node(r["peak_id"], ntype="peak")
        G.add_node(r["gene"], ntype="gene")
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
    tf_peaks = set(tf_weights_df[tf_weights_df["tf_gene"] == target_gene]["peak_id"])
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
            label = _truncate_label(node, 12)
            fontsize = 5.5
            fw = "bold"
            offset_y = 0.030
            va = "bottom"
            rotation = 0
        elif ntype in ("target", "affected"):
            label = _truncate_label(node, 14)
            fontsize = 5.8
            fw = "bold"
            offset_y = 0.026
            va = "bottom"
            rotation = 0
        else:  # tf, gene
            label = _truncate_label(node, 12)
            fontsize = 4.8
            fw = "bold" if ntype == "tf" else "normal"
            offset_y = 0.024
            va = "bottom"
            rotation = 0

        # Append delta for affected and orphan genes
        if ntype in ("affected", "orphan") and delta_map and node in delta_map:
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

    title = f"Cluster {cluster_label}  |  {target_gene} → {n_total} affected"
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
    pert = pd.read_csv(pert_csv)
    tf_df = pd.read_csv(tf_csv)
    edges_df = pd.read_csv(edges_csv)

    if output_dir is None:
        output_dir = str(Path(pert_csv).parent)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Building full network...")
    G = build_full_network(tf_csv, edges_csv)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    pert["abs_delta"] = pert["delta_rna"].abs()
    target_genes = pert["target_gene"].unique()
    clusters = sorted(pert["cluster"].unique(), key=int)

    # Use is_fallback column if available, otherwise compute via BFS
    if "is_fallback" not in pert.columns:
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
        reachable = _reachable_genes(G, target_genes[0], max_depth=5)
        pert["is_fallback"] = ~pert["affected_gene"].isin(reachable)

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
                delta_map[row["affected_gene"]] = row["delta_rna"]

            # Build subgraph with both causal and orphan genes
            sg, _ = extract_subgraph(
                G, tg,
                set(causal_genes["affected_gene"]),
                orphan_set,
                tf_df, edges_df
            )

            # Also add orphan nodes if they weren't already added
            # (some may have been added via causal edges if they happen to have partial paths)
            for og in orphan_set:
                if og not in sg:
                    sg.add_node(og, ntype="orphan", layer=4)

            n_causal = len(affected_set)
            n_orphan = len(orphan_set)
            if n_orphan > 0:
                print(f"  Cluster {cl}: {n_causal} causal + {n_orphan} orphan")

            all_plots.append((tg, cl, sg, delta_map, orphan_set,
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
    base = "/home/huangtao/Desktop/huangchengqi/code_1/atac_bridge/results"
    pert_csv = f"{base}/perturbation_results.csv"
    tf_csv   = f"{base}/tf_peak_weights.csv"
    edges_csv = f"{base}/causal_peak_gene_edges.csv"
    top = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(pert_csv, tf_csv, edges_csv, top_n=top)
