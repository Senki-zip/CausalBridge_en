"""Output-only temporal links; no independent TF allocation of RNA effects."""
from bisect import bisect_right

import numpy as np
import pandas as pd


KEY = "saturating_pathways"
COLUMNS = [
    "target_gene", "tf_name", "peak_id", "affected_gene", "cluster", "source_context",
    "pathway_type", "causal_claim", "branch", "branch_boundary_check", "tf_lag",
    "tf_lag_first", "bin_min", "bin_max", "peak_bin_start", "peak_bin_end",
    "source_bin_min", "source_bin_max", "n_bins", "n_active_bins", "event_count",
    "tf_input_contribution_mean", "tf_input_contribution_sum", "tf_input_unit",
    "peak_input_contribution_mean", "peak_input_contribution_sum", "peak_input_unit",
    "tf_contribution_methods", "peak_contribution_methods", "peak_clipping_status",
    "rna_clipping_status", "peak_state_check", "shared_peak_contribution",
    "endpoint_delta_rna_log2fc", "delta_rna_log2fc", "endpoint_effect_scope",
    "train_r2", "quality_scope", "delta_r_raw_sum", "delta_r_total",
    "pathway_allocation_status",
]


def empty_pathways():
    return pd.DataFrame(columns=COLUMNS)


def _records(tables):
    """Accept frames or frame batches without materializing a joined event table."""
    if isinstance(tables, pd.DataFrame):
        tables = (tables,)
    for table in tables:
        if not table.empty:
            yield from (
                dict(zip(table.columns, values)) for values in table.itertuples(index=False, name=None))


def _identity(row):
    return str(row.get("source_context", row.get("cluster", "global"))), str(row["target_gene"])


def _integer(value):
    try:
        number = float(value)
        return int(number) if np.isfinite(number) and number >= 0 and number.is_integer() else None
    except (TypeError, ValueError):
        return None


def _nonzero(value):
    try:
        return np.isfinite(float(value)) and float(value) != 0
    except (TypeError, ValueError):
        return False


def build_pathways(tf_contributions, peak_attributions, endpoints, *, branch_boundaries=None,
                   control_atac_states=None, atac_state_history=None, peak_ids=None,
                   atac_unavailable_mask=None, local_events=None, rna_clips=None):
    """Summarize TF(s-lag)→peak(s)→gene(s+1) candidate event co-occurrences.

    Each row is one context/intervention/TF/peak/gene/lag/branch. The peak→gene
    contribution and complete gene endpoint can be shared by several incoming
    TFs. Neither is allocated to those TFs, and legacy allocation columns are NaN.
    Frame batches produce the same summaries as whole frames.
    """
    boundaries = sorted(tuple(map(int, pair)) for pair in (branch_boundaries or ()))
    if any(a < 0 or b <= a for a, b in boundaries) or any(
            boundaries[i][1] > boundaries[i + 1][0] for i in range(len(boundaries) - 1)):
        raise ValueError("invalid pathway branch boundaries")
    starts = [a for a, _ in boundaries]
    def branch(bin_number):
        if branch_boundaries is None:
            return "unspecified"
        i = bisect_right(starts, bin_number) - 1
        return str(i) if i >= 0 and bin_number < boundaries[i][1] else None

    state_check = control_atac_states is not None or atac_state_history is not None
    wt = final = np.empty((0, 0))
    bad = np.empty((0, 0), dtype=bool)
    index = {}
    if state_check:
        wt, final = np.asarray(control_atac_states), np.asarray(atac_state_history)
        if wt.ndim != 2 or final.shape != wt.shape or peak_ids is None or len(peak_ids) != wt.shape[1]:
            raise ValueError("pathway ATAC history/axis mismatch")
        index = {str(peak): i for i, peak in enumerate(peak_ids)}
        bad = np.zeros(wt.shape, bool) if atac_unavailable_mask is None else np.asarray(atac_unavailable_mask)
        if bad.shape != wt.shape:
            raise ValueError("pathway unavailable-mask mismatch")

    endpoint_index = {}
    for row in _records(endpoints):
        # An unavailable/filtered endpoint cannot become a reported pathway.
        if row.get("status") != "ok" or row.get("direct_target", False) or not row.get("selected", True):
            continue
        effect = row.get("delta_rna")
        if effect is None or not np.isfinite(effect):
            continue
        endpoint_index[(*_identity(row), str(row["affected_gene"]))] = row
    if not endpoint_index:
        return empty_pathways()

    def clip_index(tables, name, default):
        output = {}
        if tables is not None:
            for row in _records(tables):
                b = _integer(row.get("bin"))
                identity = row.get(name)
                if b is not None and identity is not None and not pd.isna(identity):
                    output[(*_identity(row), str(identity), b)] = str(row.get("status", default))
        return output
    peak_clips = clip_index(local_events, "peak_id", "not_reported")
    gene_clips = clip_index(rna_clips, "gene", "not_reported")

    tf_tables = (tf_contributions,) if isinstance(tf_contributions, pd.DataFrame) else tuple(tf_contributions)
    peak_tables = (peak_attributions,) if isinstance(peak_attributions, pd.DataFrame) else tuple(peak_attributions)
    bins = sorted({b for table in tf_tables if "bin" in table for value in table["bin"].unique()
                   if (b := _integer(value)) is not None})
    def bin_records(tables, b):
        for table in tables:
            if "bin" in table:
                yield from _records(table.loc[table["bin"] == b])
    # Keep only one bin of event indexes. Stream links directly into unique
    # pathway summaries; a full event Cartesian dataframe is never created.
    summaries = {}
    for b in bins:
        peak_events = {}
        for row in bin_records(peak_tables, b + 1):
            if not _nonzero(row.get("delta_l2")) or row.get("status", "ok") != "ok":
                continue
            context, target = _identity(row)
            gene, peak = str(row["gene"]), str(row["peak_id"])
            if (context, target, gene) not in endpoint_index:
                continue
            event = peak_events.setdefault((context, target, peak), {}).setdefault(gene, [0., set()])
            event[0] += float(row["delta_l2"])
            event[1].add(str(row.get("method", "unspecified")))
        inputs = {}
        for row in bin_records(tf_tables, b):
            source = _integer(row.get("source_bin"))
            if source is None or source > b or not _nonzero(row.get("delta_atac_raw")) or row.get("status", "ok") != "ok":
                continue
            context, target = _identity(row)
            peak, tf = str(row["peak_id"]), str(row["tf_gene"])
            if (context, target, peak) not in peak_events:
                continue
            br = branch(b)
            if br is None or branch(source) != br or branch(b + 1) != br:
                continue
            if state_check:
                j = index.get(peak)
                if j is None or b >= len(wt) or bad[b, j] or not np.isfinite([wt[b, j], final[b, j]]).all() or wt[b, j] == final[b, j]:
                    continue
            unit = str(row.get("atac_state_unit", "raw_atac"))
            event = inputs.setdefault((context, target, tf, peak, source, br), [0., set(), unit])
            if event[2] != unit:
                raise ValueError("mixed pathway ATAC contribution units")
            event[0] += float(row["delta_atac_raw"])
            event[1].add(str(row.get("method", "unspecified")))
        for (context, target, tf, peak, source, br), (tf_value, methods, unit) in sorted(inputs.items()):
            if not _nonzero(tf_value):
                continue
            for gene, (peak_value, peak_methods) in sorted(peak_events[context, target, peak].items()):
                if not _nonzero(peak_value):
                    continue
                key = (context, target, tf, peak, gene, b - source, br)
                acc = summaries.setdefault(key, dict(bin_min=b, bin_max=b, source_min=source, source_max=source,
                    tf_sum=0., peak_sum=0., count=0,
                    tf_methods=set(), peak_methods=set(), peak_clips=set(), gene_clips=set(), unit=unit))
                if acc["unit"] != unit:
                    raise ValueError("mixed pathway ATAC contribution units")
                acc["bin_min"] = min(acc["bin_min"], b); acc["bin_max"] = max(acc["bin_max"], b)
                acc["source_min"] = min(acc["source_min"], source); acc["source_max"] = max(acc["source_max"], source)
                acc["tf_sum"] += tf_value; acc["peak_sum"] += peak_value; acc["count"] += 1
                acc["tf_methods"].update(methods); acc["peak_methods"].update(peak_methods)
                acc["peak_clips"].add(peak_clips.get((context, target, peak, b), "not_reported"))
                acc["gene_clips"].add(gene_clips.get((context, target, gene, b + 1), "not_reported"))
    rows = []
    for (context, target, tf, peak, gene, lag, br), acc in sorted(summaries.items()):
        endpoint = endpoint_index[context, target, gene]
        count = acc["count"]
        rows.append(dict(target_gene=target, tf_name=tf, peak_id=peak, affected_gene=gene,
            cluster=context, source_context=context, pathway_type="temporal_event_link", causal_claim=False,
            branch=br, branch_boundary_check=branch_boundaries is not None, tf_lag=lag, tf_lag_first=lag,
            bin_min=acc["bin_min"] + 1, bin_max=acc["bin_max"] + 1,
            peak_bin_start=acc["bin_min"], peak_bin_end=acc["bin_max"],
            source_bin_min=acc["source_min"], source_bin_max=acc["source_max"],
            n_bins=count, n_active_bins=count, event_count=count,
            tf_input_contribution_mean=acc["tf_sum"] / count, tf_input_contribution_sum=acc["tf_sum"],
            tf_input_unit="delta_" + acc["unit"], peak_input_contribution_mean=acc["peak_sum"] / count,
            peak_input_contribution_sum=acc["peak_sum"], peak_input_unit="delta_log1p_rna_log1p_cp10k",
            tf_contribution_methods=";".join(sorted(acc["tf_methods"])),
            peak_contribution_methods=";".join(sorted(acc["peak_methods"])),
            peak_clipping_status=";".join(sorted(acc["peak_clips"])),
            rna_clipping_status=";".join(sorted(acc["gene_clips"])),
            peak_state_check="changed_available" if state_check else "not_checked",
            shared_peak_contribution=True, endpoint_delta_rna_log2fc=endpoint["delta_rna"],
            delta_rna_log2fc=endpoint["delta_rna"], endpoint_effect_scope="complete_gene_endpoint_shared_not_allocated",
            train_r2=endpoint.get("train_r2", np.nan), quality_scope=endpoint.get("quality_scope", "target_full_training"),
            delta_r_raw_sum=np.nan, delta_r_total=np.nan, pathway_allocation_status="unavailable"))
    return pd.DataFrame(rows, columns=COLUMNS)
