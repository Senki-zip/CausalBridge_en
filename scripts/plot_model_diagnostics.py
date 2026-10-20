#!/usr/bin/env python3
"""Independent saved-result plots. No fitting, checkpoint migration or result writes.

Legacy registries are read as authenticated single-curve archives only. This is
deliberately not a replacement for production registry validation or prediction.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import pickle
import re
import sys
import traceback
import textwrap
import time
import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

os.environ.setdefault("MPLCONFIGDIR", "/tmp/opencode/CausalBridge_new_visualizations/matplotlib")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from CausalBridge import saturating_hybrid as core
from CausalBridge.nn_hybrid import NNSingleModel
from CausalBridge.spline_hybrid import SplineSingleModel
from plot_causal_network import intervention_members


def note(message):
    print(f"[new visualizations] {message}", flush=True)


@contextmanager
def timed_step(name):
    started = time.perf_counter()
    note(f"starting {name}")
    try:
        yield
    finally:
        note(f"{name}: {time.perf_counter() - started:.2f}s")


def context_id(value):
    return re.sub(r"^cluster_", "", str(value))


def slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def read_table(root, name):
    for file in (root / name, root / "intermediate" / name):
        if file.exists():
            return pd.read_csv(file, low_memory=False)
    return pd.DataFrame()


def identity(frame, role):
    field = role + "_id"
    if field in frame:
        return frame[field].fillna(frame[role]).astype(str)
    return frame[role].astype(str)


def load_pathways(root, labels):
    """Read actual exported paths and resolve result-local gene aliases."""
    # Prefer the canonical export even when it is empty.
    name = next((name for name in ("perturbation_pathways.csv", "saturating_pathways.csv")
                 if any(file.exists() for file in (root / name, root / "intermediate" / name))), None)
    paths = read_table(root, name) if name else pd.DataFrame()
    aliases = {}

    def add(gene, alias):
        if pd.notna(gene) and pd.notna(alias):
            aliases.setdefault(str(alias), set()).add(str(gene))
            aliases.setdefault(str(gene), set()).add(str(gene))

    for gene, label in labels.items():
        add(gene, label)
    genes = read_table(root, "gene_identity.csv")
    if "gene_id" in genes:
        for field in ("gene_id", "gene_name", "display_name"):
            if field in genes:
                for gene, alias in zip(genes.gene_id, genes[field]):
                    add(gene, alias)
    for frame in (read_table(root, "perturbation_results.csv"), paths):
        for role in ("affected_gene", "target_gene", "tf_name", "tf_gene"):
            if role + "_id" in frame:
                for field in (role, role + "_name"):
                    if field in frame:
                        pairs = frame[[role + "_id", field]].drop_duplicates()
                        for gene, alias in pairs.itertuples(index=False, name=None):
                            add(gene, alias)
    if paths.empty:
        return paths, aliases

    canonical_ids = {gene for matches in aliases.values() for gene in matches}

    def canonical(value):
        value = str(value)
        if value in canonical_ids:
            return value
        matches = aliases.get(value, {value})
        if len(matches) != 1:
            raise ValueError(f"ambiguous saved pathway gene {value!r}; use canonical IDs")
        return next(iter(matches))

    paths = paths.copy()
    paths["gene_id"] = identity(paths, "affected_gene").map(canonical)
    tf_ids = identity(paths, "tf_name")
    paths["tf_id"] = tf_ids.map({value: canonical(value) for value in tf_ids.unique()})
    context = paths["source_context"] if "source_context" in paths else paths["cluster"]
    if "cluster" in paths:
        context = context.fillna(paths["cluster"])
    paths["context"] = context.map({value: context_id(value) for value in context.unique()})
    for gene in paths.gene_id.unique():
        add(gene, gene)
    return paths, aliases


def requested_pathways(paths, requested, aliases):
    genes = set()
    canonical_ids = {gene for matches in aliases.values() for gene in matches}
    for value in requested:
        matches = {value} if value in canonical_ids else aliases.get(value, set())
        if len(matches) > 1:
            raise ValueError(f"ambiguous downstream gene {value!r}: {sorted(matches)}; use a stable ID or unique display name")
        if matches:
            genes.update(matches)
        else:
            note(f"no saved downstream gene matches {value!r}")
    return paths[paths.gene_id.isin(genes)].copy() if not paths.empty else paths.copy()


def requested_peak_pathways(paths, peaks):
    """Use exact saved identities; never silently replace with a nearby peak."""
    if not peaks:
        return paths
    available = set(paths.peak_id.astype(str)) if 'peak_id' in paths else set()
    missing = sorted(set(peaks) - available)
    if missing:
        raise ValueError('No saved pathway for requested peak IDs (within selected genes): '
                         + ', '.join(missing))
    return paths[paths.peak_id.astype(str).isin(peaks)].copy()


def pathway_edges(paths):
    """All saved TF/peak/context/lag relations, independent of trace top-paths."""
    edges = set()
    for _, row in paths.iterrows():
        lag = None
        if "tf_lag" in paths:
            value = pd.to_numeric(row.tf_lag, errors="coerce")
            if not np.isfinite(value) or value < 0 or value != int(value):
                continue
            lag = int(value)
        edges.add((row.context, row.tf_id, str(row.peak_id), lag))
    return edges


def authenticated(payload, label):
    if not isinstance(payload, dict) or payload.get("fingerprint") != core._digest(
            {k: v for k, v in payload.items() if k != "fingerprint"}):
        raise ValueError(f"{label}: saved fingerprint mismatch")


def decode_single(component, family, input_unit, output_unit):
    """Use the saved family's own validated inference object, never refit."""
    item = copy.deepcopy(component)
    if family in ("spline", "nn"):
        if item.pop("response_type", None) != family:
            raise ValueError("saved component response family mismatch")
        for key, unit in (("input_unit", input_unit), ("output_unit", output_unit)):
            if item.pop(key, None) != unit:
                raise ValueError("saved component unit mismatch")
        return (SplineSingleModel if family == "spline" else NNSingleModel)(**item)
    if family != "saturation":
        raise ValueError("unknown saved response family")
    if "normalized_atac_log1p" in (input_unit, output_unit):
        if item.pop("schema", None) != core.CONTINUOUS_SINGLE_SCHEMA:
            raise ValueError("unknown continuous saturation component schema")
        if item.pop("input_unit", None) != input_unit or item.pop("output_unit", None) != output_unit:
            raise ValueError("saturation component unit mismatch")
    return core.SingleModel(**item)


def curve_archive(registry, selected_tfs=None):
    """Authenticate archive and expose singles, not whole-model predictions.

    Current contract 2 gets the full production validator. Pre-contract archives
    retain their own fingerprints/schema and validated component math; they are
    never passed off as runnable current production models.
    """
    # Current registries are authenticated (including every target bundle) by
    # validate_registry below. Legacy archives require our explicit checks.
    current = isinstance(registry, dict) and "model_contract_version" in registry
    if not current:
        authenticated(registry, "registry")
    family = registry.get("context", {}).get("response_family", "saturation")
    expected = {"spline": "hybrid_spline_tf_peak_registry_v2",
                "nn": "hybrid_nn_tf_peak_registry_v1"}
    if family == "saturation":
        from CausalBridge.saturating_transfer import TF_SCHEMA, TF_SCHEMA_V2
        schemas = (TF_SCHEMA, TF_SCHEMA_V2)
    else:
        schemas = (expected.get(family),)
    schemas = (*schemas, f"hybrid_{family}_tf_peak_local_registry_v3")
    if (registry.get("type") != "saturating_hybrid_tf_peak" or
            registry.get("schema") not in schemas or
            registry.get("schema_version") != registry.get("schema") or
            registry.get("model_mode") != "saturating_hybrid"):
        raise ValueError("unknown TF response archive schema")
    if current:
        from CausalBridge.saturating_transfer import validate_registry
        validate_registry(registry)
        archive_mode = "current_contract_2"
    else:
        archive_mode = "legacy_saved_singles_only"
    target_schemas = {"spline": "saturating_hybrid_spline_target_v2",
                      "nn": "saturating_hybrid_nn_target_v1",
                      "saturation": (core.SCHEMA, core.CONTINUOUS_SCHEMA)}
    result = []
    for peak, bundle in registry["targets"].items():
        if selected_tfs is not None and not any(key[0] in selected_tfs for key in bundle["input_keys"]):
            continue
        if not current:
            authenticated(bundle, f"target {peak}")
        allowed = target_schemas[family]
        if bundle["schema"] not in ((allowed,) if isinstance(allowed, str) else allowed):
            raise ValueError("unknown target bundle schema")
        if (bundle["target_id"] != peak or bundle["input_unit"] != registry["input_unit"] or
                bundle["output_unit"] != registry["output_unit"] or
                bundle["provenance"].get("response_family", "saturation") != family):
            raise ValueError("target archive axis/family/unit mismatch")
        keys = bundle["input_keys"]
        if len(keys) != len(bundle["singles"]) or len(keys) != len(bundle["weights"]):
            raise ValueError("target input/single/weight dimensions mismatch")
        for key, component, alpha in zip(keys, bundle["singles"], bundle["weights"]):
            if selected_tfs is not None and key[0] not in selected_tfs:
                continue
            if (len(key) != 2 or not isinstance(key[0], str) or
                    (key[1] != "local" and (not isinstance(key[1], int) or key[1] < 0)) or not np.isfinite(alpha) or alpha < 0):
                raise ValueError("invalid saved TF input/lag/weight")
            model = decode_single(component, family, bundle["input_unit"], bundle["output_unit"])
            result.append(dict(peak=peak, tf=key[0], lag=key[1], alpha=float(alpha),
                model=model, bundle=bundle, family=family, archive_mode=archive_mode,
                input_unit=bundle["input_unit"], output_unit=bundle["output_unit"]))
    return result


def _archive_worker(job):
    file, selected_tfs, output = job
    # Validate in the worker; transfer only selected, plain saved bundles via
    # disk. Immutable inference objects contain mappingproxy and are rebuilt
    # in the parent without repeating full-registry validation.
    archives = load_archives(file.parents[2], selected_tfs, files=[file])
    plain = {}
    for ctx, (registry, curves, weights) in archives.items():
        compact = {key: registry[key] for key in ('context', 'input_unit', 'output_unit')}
        plain[ctx] = (compact, [{k:v for k,v in curve.items() if k != 'model'}
                               for curve in curves], weights)
    with open(output, 'wb') as stream:
        pickle.dump(plain, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return output


def _plot_worker_init():
    from threadpoolctl import threadpool_limits
    global _PLOT_THREAD_LIMIT
    _PLOT_THREAD_LIMIT = threadpool_limits(limits=1)


def load_archives(root, selected_tfs=None, *, workers=1, files=None):
    files = sorted((root / "checkpoints").glob("cluster_*/step6_rna_to_atac.pkl")) if files is None else files
    if workers > 1 and len(files) > 1:
        archives = {}
        with tempfile.TemporaryDirectory(prefix='causalbridge-plot-') as directory:
            jobs = [(file, selected_tfs, str(Path(directory) / f'{i}.pkl')) for i, file in enumerate(files)]
            with ProcessPoolExecutor(max_workers=min(workers, len(files)),
                    mp_context=multiprocessing.get_context('spawn'), initializer=_plot_worker_init) as pool:
                for output in pool.map(_archive_worker, jobs):
                    with open(output, 'rb') as stream:
                        part = pickle.load(stream)
                    for ctx, (registry, curves, weights) in part.items():
                        for curve in curves:
                            bundle = curve['bundle']
                            index = next(i for i,key in enumerate(bundle['input_keys'])
                                         if key == [curve['tf'], curve['lag']] or key == (curve['tf'], curve['lag']))
                            curve['model'] = decode_single(bundle['singles'][index], curve['family'],
                                                          curve['input_unit'], curve['output_unit'])
                        archives[ctx] = (registry, curves, weights)
                    Path(output).unlink()
        return archives
    archives = {}
    for file in files:
        try:
            # Checkpoints are local trusted pipeline artifacts, like the pipeline's loader.
            with timed_step(f"{file.parent.name} checkpoint read"):
                with file.open("rb") as stream:
                    checkpoint = pickle.load(stream)
            registry = checkpoint["tf_model_registry"]
            with timed_step(f"{file.parent.name} model validation and response selection"):
                curves = curve_archive(registry, selected_tfs)
            for curve in curves:
                curve["context"] = context_id(file.parent.name)
            # Full registry has been validated. Plots consume only its context,
            # units and the selected curves; release unrelated target bundles.
            compact = {key: registry[key] for key in ('context', 'input_unit', 'output_unit')}
            archives[context_id(file.parent.name)] = (compact, curves, checkpoint["tf_weights"])
            note(f"{file.parent.name}: {len(curves)} authenticated saved singles ({curves[0]['archive_mode'] if curves else 'empty'})")
        except (ValueError, KeyError, TypeError, OSError, pickle.UnpicklingError) as exc:
            note(f"skip {file}: {exc}")
    return archives


def labels_and_targets(root):
    labels = {}
    genes = read_table(root, "gene_identity.csv")
    if {"gene_id", "gene_name"}.issubset(genes):
        labels.update(zip(genes.gene_id.astype(str), genes.gene_name.astype(str)))
    endpoints = read_table(root, "perturbation_results.csv")
    members, target_labels = intervention_members(endpoints)
    targets = set(members)
    labels.update(target_labels)
    for _, row in endpoints.iterrows():
        for role in ("affected_gene", "target_gene"):
            if pd.notna(row.get(role + "_id")) and pd.notna(row.get(role + "_name")):
                labels[str(row[role + "_id"])] = str(row[role + "_name"])
    return labels, targets


def intervention_context(targets, labels):
    if len(targets) < 2:
        return ""
    return "\nJoint KO: " + " + ".join(labels.get(tf, tf) for tf in sorted(targets))


def gene_curve_archive(registry, selected_edges):
    """Read only requested Step 5 singles, including historical source signatures.

    This authenticates saved data and component math, not production resumability.
    No production validator or checkpoint acceptance rule is changed.
    """
    authenticated(registry, "gene registry")
    schemas = {"hybrid_nn_peak_gene_registry_v1": "nn",
               "hybrid_spline_peak_gene_registry_v2": "spline",
               "saturating_peak_gene_registry_v1": "saturation",
               "saturating_peak_gene_registry_v2": "saturation"}
    schema = registry.get("schema")
    family = schemas.get(schema)
    if (family is None or registry.get("schema_version") != schema or
            registry.get("type") != "saturating_hybrid_peak_gene" or
            registry.get("model_mode") != "saturating_hybrid" or
            registry.get("model_contract_version", 0) not in (0, 2) or
            registry.get("context", {}).get("response_family", family) != family):
        raise ValueError("unknown peak→gene archive schema/family/contract")
    input_unit, output_unit = registry["input_unit"], registry["output_unit"]
    if input_unit not in ("raw_atac", "normalized_atac_log1p") or output_unit != "log1p_rna_log1p_cp10k":
        raise ValueError("peak→gene archive unit mismatch")
    target_schemas = {"nn": ("saturating_hybrid_nn_target_v1",),
                      "spline": ("saturating_hybrid_spline_target_v2",),
                      "saturation": (core.SCHEMA, core.CONTINUOUS_SCHEMA)}
    curves = []
    for gene in sorted({gene for gene, _, _ in selected_edges}):
        bundle = registry["targets"].get(gene)
        if bundle is None:
            continue
        authenticated(bundle, f"target {gene}")
        if (bundle.get("schema") not in target_schemas[family] or bundle.get("target_id") != gene or
                bundle.get("input_unit") != input_unit or bundle.get("output_unit") != output_unit or
                bundle.get("provenance", {}).get("response_family", family) != family):
            raise ValueError("peak→gene target axis/family/unit mismatch")
        keys = bundle["input_keys"]
        if len(keys) != len(bundle["singles"]) or len(keys) != len(bundle["weights"]):
            raise ValueError("gene input/single/weight dimensions mismatch")
        for key, component, alpha in zip(keys, bundle["singles"], bundle["weights"]):
            if (len(key) != 2 or not isinstance(key[0], str) or
                    not isinstance(key[1], int) or key[1] != 1 or not np.isfinite(alpha) or alpha < 0):
                raise ValueError("invalid saved peak input/lag/weight")
            if (gene, key[0], key[1]) not in selected_edges:
                continue
            model = decode_single(component, family, input_unit, output_unit)
            curves.append(dict(gene=gene, peak=key[0], lag=key[1], alpha=float(alpha),
                model=model, family=family, input_unit=input_unit, output_unit=output_unit,
                archive_mode="authenticated_saved_gene_singles_only"))
    return curves


def _peak_plot_worker(job):
    root, paths, targets, labels, output, top_n, min_r2, file, style = job
    plt.rcParams.update(style)
    output.mkdir(parents=True, exist_ok=True)
    records = peak_gene_response_plots(root, paths, targets, labels, output, top_n, min_r2, files=[file])
    if records:
        pd.DataFrame(records).to_pickle(output / 'selected.pkl')
    return str(output)


def peak_gene_response_plots(root, paths, targets, labels, output, top_n, min_r2,
                             *, workers=1, files=None):
    """One figure per requested gene/context; load and release each Step 5 file."""
    if paths.empty or top_n == 0:
        return
    files = sorted((root / "checkpoints").glob("cluster_*/step5_atac_to_rna.pkl")) if files is None else files
    if workers > 1 and len(files) > 1:
        tables = []
        # Render on the destination filesystem so atomic replace works when
        # system /tmp and the user's output directory are different mounts.
        with tempfile.TemporaryDirectory(prefix='.causalbridge-gene-plots-', dir=output) as directory:
            style = {key: plt.rcParams[key] for key in ('font.size', 'axes.spines.top', 'axes.spines.right')}
            jobs = [(root, paths[paths.context == context_id(file.parent.name)], targets, labels,
                     Path(directory) / str(i), top_n, min_r2, file, style) for i,file in enumerate(files)]
            with ProcessPoolExecutor(max_workers=min(workers, len(files)),
                    mp_context=multiprocessing.get_context('spawn'), initializer=_plot_worker_init) as pool:
                for folder in pool.map(_peak_plot_worker, jobs):
                    folder = Path(folder)
                    for image in folder.glob('*.png'):
                        image.replace(output / image.name)
                    table = folder / 'selected.pkl'
                    if table.exists():
                        tables.append(pd.read_pickle(table))
        if tables:
            pd.concat(tables, ignore_index=True).to_csv(output / 'peak_gene_response_selected.csv', index=False)
        return
    paths = paths[paths.tf_id.isin(targets)]
    selected_records = []
    for file in files:
        ctx = context_id(file.parent.name)
        group = paths[paths.context == ctx]
        edges = set()
        for _, row in group.iterrows():
            lag = pd.to_numeric(row.get("peak_lag", 1), errors="coerce")
            if np.isfinite(lag) and lag == 1:
                edges.add((row.gene_id, str(row.peak_id), 1))
        if not edges:
            continue
        with file.open("rb") as stream:
            checkpoint = pickle.load(stream)  # Trusted local pipeline artifacts only.
        curves = gene_curve_archive(checkpoint["transfer_models"], edges)
        evidence = {}
        screening = checkpoint.get("causal_edges", pd.DataFrame())
        if isinstance(screening, pd.DataFrame) and {"gene", "peak_id"}.issubset(screening):
            screening = screening.copy()
            screening["gene_id"] = identity(screening, "gene")
            screening["_lag"] = screening.get("lag", 1)
            screening["_p"] = pd.to_numeric(screening.get("p_value", np.nan), errors="coerce")
            screening["_fdr"] = pd.to_numeric(screening.get("p_adj", screening.get("fdr", np.nan)), errors="coerce")
            evidence = screening.groupby(["gene_id", "peak_id", "_lag"])[["_p", "_fdr"]].min().to_dict("index")
        del checkpoint  # Do not retain full registries across contexts.
        records = []
        for curve in curves:
            m = curve["model"].metrics
            r2 = m.get("training_r2")
            if (curve["alpha"] > 0 and m.get("status") == "ok" and r2 is not None and
                    np.isfinite(r2) and r2 >= min_r2):
                saved = evidence.get((curve["gene"], curve["peak"], curve["lag"]), {})
                records.append(dict(curve, context=ctx, training_r2=float(r2),
                    screening_p=saved.get("_p", np.nan), screening_fdr=saved.get("_fdr", np.nan),
                    solver_status=m.get("training_status", m.get("termination_status", "unreported"))))
        for gene in sorted({r["gene"] for r in records}):
            def rank(r):
                return (-r["training_r2"],
                        r["screening_fdr"] if np.isfinite(r["screening_fdr"]) else np.inf,
                        r["screening_p"] if np.isfinite(r["screening_p"]) else np.inf,
                        r["peak"], r["lag"])
            chosen = sorted((r for r in records if r["gene"] == gene),
                            key=rank)[:top_n]
            fig, ax = plt.subplots(figsize=(10, 6))
            high = max(r["model"].x_range[1] for r in chosen)
            low = min(0., min(r["model"].x_range[0] for r in chosen))
            x = np.linspace(low, high if high > 0 else 1., 250)
            for i, r in enumerate(chosen):
                model = r["model"]
                y = model.predict(x)
                if not np.isfinite(y).all():
                    raise ValueError("nonfinite saved peak→gene response evaluation")
                lo, hi = model.x_range
                supported = (x >= lo) & (x <= hi)
                color = plt.get_cmap("tab10")(i % 10)
                ax.plot(x, np.where(supported, y, np.nan), color=color, lw=1.5,
                        label=f"{r['peak']}  lag={r['lag']}  train R²={r['training_r2']:.2f}")
                ax.plot(x, np.where(~supported, y, np.nan), color=color, ls="--", lw=1., alpha=.65)
                ax.scatter([lo, hi], model.predict(np.array([lo, hi])), color=color, s=10)
                ax.scatter([0.], model.predict(np.array([0.])), color=color, marker="x", s=22)
                selected_records.append({k: v for k, v in r.items() if k != "model"})
            ax.axvline(0, color="#b33333", lw=1, ls=":", label="Zero peak input")
            ax.set(xlabel=f"Peak input ({chosen[0]['input_unit']})",
                   ylabel=f"Fitted {labels.get(gene, gene)} response ({chosen[0]['output_unit']})",
                   title=f"Peaks → {labels.get(gene, gene)} · Cluster {ctx}\nHigh training-fit rank; no validation confidence probability" + intervention_context(targets, labels))
            figure_footer(fig, "Solid: saved WT input range; dashed: extrapolation; dots: WT endpoints; ×: input zero.\n"
                    "Ranking: single training R², then saved screening FDR/p if available. No refitting or whole-model prediction.",
                    )
            ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
            ax.grid(alpha=.16)
            save_figure(fig, output, f"peak_gene_response_{slug(labels.get(gene, gene))}_{slug(gene)}_context_{slug(ctx)}")
    if selected_records:
        pd.DataFrame(selected_records).to_csv(output / "peak_gene_response_selected.csv", index=False)
    else:
        note("skip peak→gene responses: no saved pathway singles pass status / training R2 filters")
    return selected_records


def load_history(root, wanted, targets=None):
    file = root / "perturbation_projection_bin_history.csv"
    if not file.exists():
        file = root / "intermediate" / "projection_history.csv"
    if not file.exists():
        return pd.DataFrame()
    parts = []
    observed_targets = set(targets) if targets is not None else None
    for part in pd.read_csv(file, chunksize=200000, low_memory=False):
        if not {"control_state", "state_value", "bin", "branch", "pseudotime", "source_context"}.issubset(part):
            note("skip RNA history: complete WT/KO state schema unavailable")
            return pd.DataFrame()
        if "target_gene" in part:
            members, _ = intervention_members(part)
            if observed_targets is not None and set(members) != observed_targets:
                raise ValueError("RNA history intervention differs from endpoint intervention")
            observed_targets = set(members)
        part["gene_id"] = identity(part, "gene")
        part = part[part.gene_id.isin(wanted)].copy()
        part["context"] = part.source_context.map(context_id)
        parts.append(part)
    history = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not history.empty and history.duplicated(["context", "branch", "bin", "gene_id"]).any():
        raise ValueError("RNA history contains multiple interventions/duplicate states; choose a single-run result directory")
    return history


def validate_history_axes(history, archives):
    """Require saved history and checkpoint to describe the same bin/branch axes."""
    if history.empty:
        return history
    keep = []
    for ctx, group in history.groupby("context"):
        if ctx not in archives:
            continue
        metadata = archives[ctx][0]["context"]
        centers = np.asarray(metadata["bin_centers"], dtype=float)
        branches = np.asarray(metadata["bin_branch_ids"]).astype(str)
        bins = group.bin.to_numpy(int)
        if ((bins < 0).any() or (bins >= len(centers)).any() or
                not np.allclose(group.pseudotime, centers[bins], rtol=1e-9, atol=1e-12) or
                not np.array_equal(group.branch.astype(str).to_numpy(), branches[bins]) or
                set(group.state_unit.dropna()) != {archives[ctx][0]["input_unit"]}):
            note(f"skip RNA history context {ctx}: saved checkpoint bin/branch/time/unit axes mismatch")
            continue
        keep.append(group)
    return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()


def support_distance(values, x_range):
    lo, hi = x_range
    span = hi - lo
    outside = np.maximum(np.maximum(lo - values, values - hi), 0.)
    if span == 0:
        # A constant WT input supplies no scale for nonzero extrapolation.
        return np.where(outside == 0, 0., np.nan)
    return outside / span


def wrapped_text(value, width):
    """Wrap prose without cutting gene symbols, stable IDs or peak coordinates."""
    return "\n".join(textwrap.fill(line, width=width, break_long_words=False,
                                  break_on_hyphens=False) for line in value.split("\n"))


def figure_footer(fig, text):
    # Figure coordinates keep notes out of neighboring axes' white patches.
    existing = getattr(fig, "_diagnostic_footer", None)
    if existing is None:
        fig._diagnostic_footer = fig.text(.02, .02, text, fontsize=8, va="bottom")


def layout_figure(fig):
    """Reserve internal space before tight export (which only fixes outer bounds)."""
    for ax in fig.axes:
        ax.set_title(wrapped_text(ax.get_title(), 62), pad=10)
        for axis in (ax.xaxis, ax.yaxis):
            label = axis.label.get_text()
            # Put physical units on their own line, keeping the unit token intact.
            label = re.sub(r" (?=\([^()]+\)$)", "\n", label)
            label = re.sub(r"^(Peak:) (chr\S+)", r"\1\n\2", label)
            axis.label.set_text(wrapped_text(label, 42))
    if fig._suptitle is not None:
        fig._suptitle.set_text(wrapped_text(fig._suptitle.get_text(), 72))
    footer = getattr(fig, "_diagnostic_footer", None)
    bottom = 0.
    if footer is not None:
        footer.set_text(wrapped_text(footer.get_text(), max(45, int(fig.get_figwidth() * 15))))
        fig.canvas.draw()
        bottom = (footer.get_window_extent(fig.canvas.get_renderer()).height + 20) / fig.bbox.height
    fig.set_layout_engine("constrained", rect=(0, bottom, 1, 1 - bottom),
                          h_pad=.12, w_pad=.12, hspace=.08, wspace=.08)
    fig.canvas.draw()
    # A rotated stable ID cannot be word-wrapped. Grow the canvas until each
    # complete label fits its own panel instead of intruding on its neighbors.
    for _ in range(3):
        renderer = fig.canvas.get_renderer()
        scale = max((ax.yaxis.label.get_window_extent(renderer).height /
                     max(1., .90 * ax.get_window_extent(renderer).height)
                     for ax in fig.axes if ax.get_ylabel()), default=1.)
        if scale <= 1:
            break
        fig.set_figheight(fig.get_figheight() * scale)
        fig.canvas.draw()


def save_figure(fig, output, name):
    layout_figure(fig)
    fig.savefig(output / (name + ".png"), dpi=220, bbox_inches="tight", pad_inches=.2, facecolor="white")
    plt.close(fig)
    note(f"saved {name}.png")


def response_plots(archives, targets, labels, output, top_n, min_r2, related_edges=None):
    if top_n == 0:
        return
    records = []
    for ctx, (_, curves, weights) in archives.items():
        weights = weights.copy()
        weights["tf_id"] = identity(weights, "tf_gene")
        weights["_lag"] = weights.get("tf_lag", weights.get("lag", 0))
        if "lag_mode" in weights:
            weights.loc[weights.lag_mode == "local", "_lag"] = "local"
        weights["_p"] = pd.to_numeric(weights.get("p_value", np.nan), errors="coerce")
        weights["_fdr"] = pd.to_numeric(weights.get("p_adj", weights.get("fdr", np.nan)), errors="coerce")
        edge_evidence = weights.groupby(["tf_id", "peak_id", "_lag"])[["_p", "_fdr"]].min().to_dict("index")
        for curve in curves:
            edge = (context_id(ctx), curve["tf"], str(curve["peak"]))
            if related_edges is not None and not (
                    (*edge, curve["lag"]) in related_edges or (*edge, None) in related_edges or
                    curve["lag"] == "local" and any(item[:3] == edge for item in related_edges)):
                continue
            m = curve["model"].metrics
            r2 = m.get("training_r2")
            eligible = (curve["tf"] in targets and curve["alpha"] > 0 and m.get("status") == "ok"
                        and r2 is not None and np.isfinite(r2) and r2 >= min_r2)
            if not eligible:
                continue
            evidence = edge_evidence.get((curve["tf"], curve["peak"], curve["lag"]), {})
            # Saved Pearson screening significance is evidence, never relabeled as FDR.
            records.append(dict(curve, training_r2=float(r2),
                screening_p=evidence.get("_p", np.nan),
                screening_fdr=evidence.get("_fdr", np.nan),
                solver_status=m.get("training_status", m.get("termination_status", "unreported"))))
    if not records:
        note("skip TF responses: no KO-TF singles pass saved pathway / status / training R2 filters")
        return
    table = pd.DataFrame([{k: v for k, v in row.items() if k not in ("model", "bundle")} for row in records])
    selected_records = []
    for (tf, ctx), group in table.groupby(["tf", "context"], sort=True):
        group = group.sort_values(["training_r2", "screening_fdr", "screening_p", "peak", "lag"],
                                  ascending=[False, True, True, True, True]).drop_duplicates("peak").head(top_n)
        chosen = [r for r in records if r["tf"] == tf and r["context"] == ctx and
                  (r["peak"], r["lag"]) in set(zip(group.peak, group.lag))]
        # Common physical TF axis. Each peak is evaluated from its saved model.
        low = min(r["model"].x_range[0] for r in chosen)
        high = max(r["model"].x_range[1] for r in chosen)
        x = np.linspace(min(0., low), high if high > 0 else 1., 250)
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, r in enumerate(chosen):
            y = r["model"].predict(x)
            if not np.isfinite(y).all():
                raise ValueError("nonfinite saved response evaluation")
            lo, hi = r["model"].x_range
            supported = (x >= lo) & (x <= hi)
            color = plt.get_cmap("tab10")(i % 10)
            ax.plot(x, np.where(supported, y, np.nan), color=color, lw=1.5,
                    label=f"{r['peak']}  lag={r['lag']}  train R²={r['training_r2']:.2f}")
            ax.plot(x, np.where(~supported, y, np.nan), color=color, ls="--", lw=1., alpha=.65)
            ax.scatter([lo, hi], r["model"].predict(np.array([lo, hi])), color=color, s=10)
            ax.scatter([0.], r["model"].predict(np.array([0.])), color=color, marker="x", s=22)
            selected_records.append({k: v for k, v in r.items() if k not in ("model", "bundle")})
        ax.axvline(0, color="#b33333", lw=1, ls=":", label="Complete KO input")
        ax.set(xlabel=f"{labels.get(tf, tf)} input ({chosen[0]['input_unit']})",
               ylabel=f"Fitted peak response ({chosen[0]['output_unit']})",
               title=f"{labels.get(tf, tf)} → peaks · Cluster {ctx}\nHigh training-fit rank; no validation confidence probability" + intervention_context(targets, labels))
        figure_footer(fig, "Solid: saved WT input range; dashed: extrapolation; dots: WT endpoints; ×: input zero.\n"
                "Ranking: single training R², then saved screening FDR/p if available. No refitting or whole-model prediction.",
                )
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.grid(alpha=.16)
        save_figure(fig, output, f"tf_response_{slug(labels.get(tf, tf))}_{slug(tf)}_context_{slug(ctx)}")
    pd.DataFrame(selected_records).to_csv(output / "tf_response_selected.csv", index=False)


def diagnostic_scatter(archives, history, targets, labels, output):
    if history.empty:
        note("skip input support: complete RNA WT/KO bin history unavailable")
        return
    points = []
    inputs = {(str(ctx), str(tf)): dict(bin=group.bin.to_numpy(int),
        wt=group.control_state.to_numpy(float), ko=group.state_value.to_numpy(float),
        branch=group.branch.astype(str).to_numpy())
        for (ctx, tf), group in history.groupby(["context", "gene_id"])}
    for ctx, (_, curves, _) in archives.items():
        for curve in curves:
            if curve["tf"] not in targets or curve["alpha"] <= 0:
                continue
            rows = inputs.get((ctx, curve["tf"]))
            if rows is None:
                continue
            allowed = np.asarray(curve["bundle"]["source_bins"])
            # source_bins is target-row × input-column; keep the saved lag column.
            index = curve["bundle"]["input_keys"].index([curve["tf"], curve["lag"]])
            if curve["lag"] == "local":
                bundle = curve["bundle"]
                branch_offsets = {}
                for start, end in archives[ctx][0]["context"]["branch_boundaries"]:
                    branch = archives[ctx][0]["context"]["bin_branch_ids"][start]
                    branch_offsets[branch] = start
                supported = np.asarray(bundle["route_supported"],dtype=bool)[:,index]
                source = allowed[:,index] + np.array([branch_offsets[b] for b in bundle["branch_ids"]])
                positions = {int(b):i for i,b in enumerate(rows["bin"])}
                # Preserve one observation per supported target, including
                # repeated source bins when neighboring target lags differ.
                selected = np.array([positions[int(b)] for b in source[supported] if int(b) in positions],dtype=int)
            else:
                selected = np.isin(rows["bin"], allowed[:, index])
            if not len(selected) or selected.dtype == bool and not selected.any():
                continue
            wt = rows["wt"][selected]
            ko = rows["ko"][selected]
            branches = rows["branch"][selected]
            distance = support_distance(ko, curve["model"].x_range)
            effect = curve["alpha"] * (curve["model"].predict(ko) - curve["model"].predict(wt))
            for branch in np.unique(branches):
                use = branches == branch
                d, e = distance[use], effect[use]
                points.append(dict(context=ctx, branch=branch, n_bins=int(use.sum()),
                    tf=curve["tf"], peak=curve["peak"], lag=curve["lag"],
                    extrapolation_wt_spans=float(np.max(d)) if np.isfinite(d).all() else np.nan,
                    effect=float(np.mean(e)), effect_min=float(np.min(e)), effect_max=float(np.max(e)),
                    effect_unit=curve["output_unit"], alpha=curve["alpha"],
                    archive_mode=curve["archive_mode"],
                    scope="branch_mean_frozen_alpha_single_response_not_total_peak_effect"))
    frame = pd.DataFrame(points)
    if frame.empty:
        note("skip input support: no KO-TF input rows match saved training-bin axes")
        return
    frame.to_csv(output / "effect_input_support_points.csv", index=False)
    frame = frame[np.isfinite(frame.extrapolation_wt_spans) & np.isfinite(frame.effect)]
    if frame.empty:
        note("skip scatter: only constant-input WT ranges with undefined extrapolation scale")
        return
    units = list(frame.effect_unit.unique())
    fig, axes = plt.subplots(1, len(units), figsize=(9 * len(units), 6), squeeze=False)
    for ax, unit in zip(axes[0], units):
        unit_frame = frame[frame.effect_unit == unit]
        for i, ((tf, ctx), group) in enumerate(unit_frame.groupby(["tf", "context"])):
            # Fixed-size transparent points, not a bubble plot.
            ax.scatter(group.extrapolation_wt_spans, group.effect, s=9, alpha=.22,
                       color=plt.get_cmap("tab10")(sorted(targets).index(tf) % 10),
                       marker=("o", "s", "^", "D", "v", "P", "X")[sorted(unit_frame.context.unique()).index(ctx) % 7],
                       label=f"{labels.get(tf, tf)} · Cluster {ctx}", rasterized=True)
        ax.axhline(0, lw=.8, color="#555")
        ax.axvline(0, lw=.8, color="#888", ls=":")
        ax.set(xlabel="Maximum KO input distance outside saved WT range / WT span",
               ylabel=f"Mean frozen-alpha single response change ({unit})",
               title="Effect vs input extrapolation\nSaved TF→peak singles; training support diagnostic" + intervention_context(targets, labels))
        ax.legend(title="KO TF", fontsize=8)
        ax.grid(alpha=.16)
        figure_footer(fig, "Each point: TF–peak–lag–branch; mean effect over saved training bins, maximum extrapolation.\n"
                "These additive single contributions can differ from a routed interaction group's total effect.",
                )
    save_figure(fig, output, "effect_vs_input_support")


def ranked_response_curves(curves, limit=5):
    """Rank finite single training R²; no extra display-quality cutoff."""
    eligible = [c for c in curves if c['alpha'] > 0 and
                c['model'].metrics.get('training_r2') is not None and
                np.isfinite(c['model'].metrics['training_r2'])]
    return sorted(eligible, key=lambda c: (-c['model'].metrics['training_r2'],
                  c.get('tf', c.get('gene', '')), c['peak'], str(c['lag'])))[:limit]


def individual_response(curve, ctx, labels, output, name, layer):
    model = curve['model']
    lo, hi = model.x_range
    x = np.linspace(0., max(hi, 1e-8), 250)
    y = model.predict(x)
    supported = (x >= lo) & (x <= hi)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, np.where(supported, y, np.nan), color='#346ca0', lw=2)
    ax.plot(x, np.where(~supported, y, np.nan), color='#346ca0', ls='--', lw=1)
    ax.scatter([lo, hi], model.predict(np.array([lo, hi])), color='#346ca0')
    ax.scatter([0.], model.predict(np.array([0.])), color='#ce5145', marker='x')
    source = curve['tf'] if layer == 'tf_peak' else curve['peak']
    target = curve['peak'] if layer == 'tf_peak' else curve['gene']
    r2 = model.metrics.get('training_r2')
    r2_label = f'{r2:.3f}' if r2 is not None and np.isfinite(r2) else 'undefined'
    ax.set(xlabel=f"{labels.get(source, source)} input ({curve['input_unit']})",
           ylabel=f"Fitted response ({curve['output_unit']})",
           title=f"{labels.get(source, source)} → {labels.get(target, target)} · Cluster {ctx}\n"
                 f"Single training R²={r2_label} · alpha={curve['alpha']:.4g} · lag={curve['lag']}")
    figure_footer(fig, 'Saved single response, before alpha weighting; no refitting.\n'
                  'Solid: WT range; dashed: extrapolation; ×: zero input. Training R² is not validation.')
    save_figure(fig, output, name)


def cluster_gene_top_worker(job):
    file, labels, output = job
    ctx = context_id(file.parent.name)
    with file.open('rb') as stream:
        registry = pickle.load(stream)['transfer_models']
    edges = {(gene, key[0], key[1]) for gene, bundle in registry['targets'].items()
             for key in bundle['input_keys']}
    chosen = ranked_response_curves(gene_curve_archive(registry, edges))
    records = []
    for rank, curve in enumerate(chosen, 1):
        name = f"top5_peak_gene_context_{slug(ctx)}_rank_{rank}_{slug(curve['gene'])}_{slug(curve['peak'])}"
        individual_response(curve, ctx, labels, output, name, 'peak_gene')
        records.append(dict(context=ctx, rank=rank, gene=curve['gene'], peak=curve['peak'],
                            lag=curve['lag'], alpha=curve['alpha'], training_r2=curve['model'].metrics['training_r2']))
    return records


def cluster_top_responses(root, archives, labels, output, workers=1):
    records = []
    for ctx, (_, curves, _) in archives.items():
        for rank, curve in enumerate(ranked_response_curves(curves), 1):
            name = f"top5_tf_response_context_{slug(ctx)}_rank_{rank}_{slug(curve['tf'])}_{slug(curve['peak'])}"
            individual_response(curve, ctx, labels, output, name, 'tf_peak')
            records.append(dict(context=ctx, rank=rank, tf=curve['tf'], peak=curve['peak'],
                                lag=curve['lag'], alpha=curve['alpha'], training_r2=curve['model'].metrics['training_r2']))
    pd.DataFrame(records).to_csv(output / 'top5_tf_response_selected.csv', index=False)
    jobs = [(file, labels, output) for file in sorted((root/'checkpoints').glob('cluster_*/step5_atac_to_rna.pkl'))]
    if not jobs:
        return
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)),
                mp_context=multiprocessing.get_context('spawn'), initializer=_plot_worker_init) as pool:
            parts = list(pool.map(cluster_gene_top_worker, jobs))
    else:
        parts = [cluster_gene_top_worker(job) for job in jobs]
    pd.DataFrame([row for part in parts for row in part]).to_csv(output/'top5_peak_gene_selected.csv', index=False)


def trajectory_companion_responses(root, path, curve, ctx, labels, output, name):
    """One saved single per layer, exactly matching a successfully drawn trace."""
    individual_response(curve, ctx, labels, output, name+'_tf_response', 'tf_peak')
    file = root/'checkpoints'/f'cluster_{ctx}'/'step5_atac_to_rna.pkl'
    if not file.exists():
        note(f"skip companion peak→gene response: missing {file}")
        return
    with file.open('rb') as stream:
        registry = pickle.load(stream)['transfer_models']
    lag = path.get('peak_lag', 1)
    lag = int(lag) if pd.notna(lag) else 1
    curves = gene_curve_archive(registry, {(path.gene_id, str(path.peak_id), lag)})
    if not curves:
        raise ValueError(f"trajectory companion missing saved peak→gene model: {ctx}/{path.peak_id}/{path.gene_id}")
    individual_response(curves[0], ctx, labels, output, name+'_peak_gene_response', 'peak_gene')


def validated_peak_baseline(atac_file, registry, peaks):
    """Reconstruct WT bins only from the exact archived full-input count source."""
    import anndata as ad
    from CausalBridge import atac_normalization as atac_norm
    meta = registry["context"].get("atac_representation", {})
    if meta.get("unit") != "normalized_atac_log1p" or registry["context"].get("root_scaling_applied"):
        raise ValueError("peak baseline currently requires saved normalized ATAC without root scaling")
    adata = ad.read_h5ad(atac_file)
    counts = atac_norm._counts(adata.X)
    if (list(adata.shape) != list(meta["source_shape"]) or
            atac_norm._axis_hash(adata.obs_names) != meta["source_full_cell_axis_sha256"] or
            atac_norm._axis_hash(adata.var_names) != meta["source_full_peak_axis_sha256"] or
            atac_norm._matrix_hash(counts) != meta["source_counts_sha256"]):
        raise ValueError("ATAC file differs from saved full-input source counts/axes")
    depth = np.asarray(counts.sum(axis=1)).ravel()
    if atac_norm._digest(depth.tolist()) != meta["source_depth_sha256"]:
        raise ValueError("ATAC full-input depth fingerprint mismatch")
    ctx = registry["context"]
    cells = adata.obs_names.get_indexer(ctx["cell_ids"])
    cols = adata.var_names.get_indexer(peaks)
    if (cells < 0).any() or (cols < 0).any():
        raise ValueError("archived ATAC cells/peaks absent")
    selected = counts[cells][:, cols].toarray()
    normalized = np.log1p(selected * np.divide(float(meta["target_sum"]), depth[cells],
                          out=np.zeros(len(cells)), where=depth[cells] > 0)[:, None])
    operator = np.asarray(ctx["operator"], dtype=float)
    if operator.shape[1] != len(cells) or not np.isfinite(operator).all():
        raise ValueError("invalid saved bin operator")
    return operator @ normalized


def trajectory_plots(root, archives, history, requested, targets, labels, output, top_paths, atac_file,
                     *, selected_pathways=None):
    if not requested:
        note("skip triple traces: TRAJECTORY_GENES is NULL")
        return
    if selected_pathways is None:
        pathways, aliases = load_pathways(root, labels)
        pathways = requested_pathways(pathways, requested, aliases)
    else:
        pathways = selected_pathways
    event_file = root / "perturbation_projection_events.csv"
    event_parts = []
    if event_file.exists() and not pathways.empty:
        for part in pd.read_csv(event_file, chunksize=200000, low_memory=False):
            if "target_gene" in part:
                members, _ = intervention_members(part)
                if set(members) != set(targets):
                    raise ValueError("Peak event intervention differs from endpoint intervention")
            event_parts.append(part[part.peak_id.isin(pathways.peak_id) &
                part.relation_type.eq("saturating_hybrid_tf_peak")])
    events = pd.concat(event_parts, ignore_index=True) if event_parts else pd.DataFrame()
    if pathways.empty or history.empty or events.empty:
        note("skip triple traces: saved pathways, complete RNA history or peak events unavailable")
        return
    events["context"] = events.source_context.map(context_id)
    exported = []
    for gene in sorted(pathways.gene_id.unique()):
        requested_gene = gene
        paths = pathways[(pathways.gene_id == gene) & pathways.tf_id.isin(targets)].copy()
        if paths.empty:
            note(f"skip {requested_gene}: no saved KO-TF paths")
            continue
        effect_field = next((c for c in ("peak_input_contribution_mean", "delta_r_raw_sum") if c in paths), None)
        if effect_field:
            paths["rank"] = pd.to_numeric(paths[effect_field], errors="coerce").abs()
            paths = paths.sort_values("rank", ascending=False)
        paths = paths.drop_duplicates(["context", "tf_id", "peak_id"])
        # The display cap applies independently to each jointly knocked-out TF
        # and context, so a strong TF cannot remove every path of its partner.
        paths = (paths.groupby(["context", "tf_id"], sort=False).head(top_paths)
                 if len(targets) > 1 else paths.head(top_paths))
        for _, path in paths.iterrows():
            ctx, tf, peak = path.context, path.tf_id, path.peak_id
            if ctx not in archives:
                note(f"skip {requested_gene}/{ctx}/{peak}: no authenticated TF archive")
                continue
            registry, curves, _ = archives[ctx]
            edges = pathway_edges(pd.DataFrame([path]))
            matching = [c for c in curves if c["peak"] == peak and c["tf"] == tf and
                        (c["lag"] == "local" or (ctx, tf, str(peak), c["lag"]) in edges or (ctx, tf, str(peak), None) in edges)]
            if not matching:
                note(f"skip {requested_gene}/{ctx}/{peak}: saved TF path absent from fitted archive")
                continue
            if not atac_file:
                raise ValueError(f"{requested_gene}/{ctx}/{peak}: set --trajectory-atac-file to exact original ATAC counts")
            try:
                baseline = validated_peak_baseline(atac_file, registry, [peak])[:, 0]
            except (ValueError, KeyError, OSError, ImportError) as exc:
                raise ValueError(f"{requested_gene}/{ctx}/{peak}: {exc}") from exc
            peak_events = events[(events.context == ctx) & (events.peak_id == peak) &
                                 (events.relation_type == "saturating_hybrid_tf_peak")]
            if peak_events.duplicated("bin").any():
                note(f"skip {peak}: ambiguous duplicate peak events")
                continue
            saved_delta = peak_events.set_index("bin")
            upper = matching[0]["bundle"]["bounds"][1]
            if "bin_to_support" in registry["context"] and "support_rows" in matching[0]["bundle"]:
                inverse = {int(s):b for b,s in enumerate(registry["context"]["bin_to_support"]) if s>=0}
                global_rows = [inverse[int(s)] for s in matching[0]["bundle"]["support_rows"]]
            else:
                global_rows = matching[0]["bundle"]["target_rows"]
            expected_rows = set(global_rows)
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            data_parts = []
            for role, feature in (("TF", tf), ("Gene", gene)):
                rows = history[(history.context == ctx) & (history.gene_id == feature)].copy()
                rows["role"] = role
                data_parts.append(rows[["context", "branch", "bin", "pseudotime", "role", "control_state", "state_value", "state_unit"]])
            peak_rows = []
            for b, wt in enumerate(baseline):
                ko = np.nan if b in expected_rows else wt
                if b in saved_delta.index:
                    event = saved_delta.loc[b]
                    ko = float(np.clip(wt + event.delta_native, 0, upper)) if event.status in ("ok", "ATAC_physical_clip") else np.nan
                peak_rows.append(dict(context=ctx, branch=registry["context"]["bin_branch_ids"][b], bin=b,
                    pseudotime=registry["context"]["bin_centers"][b], role="Peak", control_state=wt,
                    state_value=ko, state_unit=registry["output_unit"]))
            data_parts.append(pd.DataFrame(peak_rows))
            data = pd.concat(data_parts, ignore_index=True)
            for ax, role, feature in zip(axes, ("TF", "Peak", "Gene"), (tf, peak, gene)):
                panel = data[data.role == role]
                for _, branch in panel.groupby("branch"):
                    branch = branch.sort_values("bin")
                    # Missing bins create gaps; separate branch lines never join.
                    first, last = int(branch.bin.min()), int(branch.bin.max())
                    branch = branch.set_index("bin").reindex(range(first, last + 1))
                    for field, color, style, label in (("control_state", "#346ca0", "-", "WT"), ("state_value", "#ce5145", "--", "KO")):
                        ax.plot(branch.pseudotime, branch[field], color=color, ls=style, lw=1.8, label=label)
                unit = panel.state_unit.dropna().unique()
                ax.set(ylabel=f"{role}: {labels.get(feature, feature)}\n{unit[0] if len(unit) else 'unavailable'}")
                ax.grid(alpha=.16)
            handles, names = axes[0].get_legend_handles_labels()
            axes[0].legend(dict(zip(names, handles)).values(), dict(zip(names, handles)).keys(), loc="best")
            lag = matching[0]["lag"]
            lag_label = "TF lag varies by bin" if lag == "local" else f"TF lag {lag} bin(s)"
            axes[0].set_title(f"{labels.get(tf, tf)} → {peak} → {labels.get(gene, gene)} · Cluster {ctx}\nSaved temporal path; {lag_label}; no allocated causal claim" + intervention_context(targets, labels))
            axes[-1].set_xlabel("Native pseudotime (lag retained; branches shown separately)")
            name = f"trajectory_{slug(labels.get(gene, gene))}_{slug(gene)}_{slug(tf)}_{slug(peak)}_context_{slug(ctx)}"
            save_figure(fig, output, name)
            # Companion plots are tied to the trace actually emitted, and are
            # never removed by the unrelated response-gallery R² threshold.
            trajectory_companion_responses(root, path, matching[0], ctx, labels, output, name)
            data["downstream_gene"] = gene
            data["tf"] = tf
            data["peak"] = peak
            if lag == "local":
                bundle = matching[0]["bundle"]
                column = bundle["input_keys"].index([tf,"local"])
                lags = {b:int(bundle["selected_lag"][i][column]) for i,b in enumerate(global_rows)}
                data["tf_lag"] = data.bin.map(lags)
                data["tf_source_bin"] = np.where(data.tf_lag>=0,data.bin-data.tf_lag,-1)
                data["lag_mode"] = "local"
            else:
                data["tf_lag"] = lag
            data["target_gene_ids_json"] = json.dumps(sorted(targets))
            exported.append(data)
    if exported:
        pd.concat(exported, ignore_index=True).to_csv(output / "trajectory_trace_data.csv", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-genes", default="")
    parser.add_argument("--trajectory-peaks", default="", help="Comma-separated exact saved peak IDs; intersect with requested genes")
    parser.add_argument("--trajectory-top-paths", type=int, default=3)
    parser.add_argument("--trajectory-atac-file", type=Path)
    parser.add_argument("--tf-response-top-n", type=int, default=10)
    parser.add_argument("--tf-response-min-r2", type=float, default=.5)
    parser.add_argument("--model-diagnostic-plots", choices=("true", "false"), default="true")
    parser.add_argument("--workers", type=int, default=2, help="Independent cluster plot workers (default: 2)")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be a positive integer")
    if args.tf_response_top_n < 0 or args.trajectory_top_paths < 1 or not np.isfinite(args.tf_response_min_r2):
        parser.error("top-n must be nonnegative, top-paths positive, and min-r2 finite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    with timed_step("labels and intervention"):
        labels, targets = labels_and_targets(args.result_dir)
    requested = [x.strip() for x in args.trajectory_genes.split(",") if x.strip()]
    peaks = [x.strip() for x in args.trajectory_peaks.split(',') if x.strip()]
    selected_tfs = set(targets)
    related_edges = None
    if requested or peaks:
        with timed_step("pathway loading and selection"):
            paths, aliases = load_pathways(args.result_dir, labels)
            if requested:
                paths = requested_pathways(paths, requested, aliases)
            paths = requested_peak_pathways(paths, peaks)
            if not requested:
                paths = paths[paths.tf_id.isin(targets)].copy()
                requested = sorted(paths.gene_id.unique())
                if not requested:
                    raise ValueError('Requested peaks have no saved paths from the knocked-out TFs')
        related_edges = pathway_edges(paths)
        if not paths.empty:
            selected_tfs.update(paths.tf_id)
        note(f"TF responses restricted to {len(related_edges)} saved pathway relations for requested downstream genes")
    with timed_step("TF model archives"):
        note(f"model loading: workers={args.workers}, numerical threads per worker=1")
        archives = load_archives(args.result_dir, selected_tfs if requested else None, **({'workers': args.workers} if args.workers > 1 else {}))
    if not archives:
        note("no compatible authenticated TF checkpoints; TF plots unavailable")
    wanted = set(targets)
    for _, curves, _ in archives.values():
        wanted.update(c["tf"] for c in curves) if requested else None
    wanted.update(requested)
    wanted.update(key for key, value in labels.items() if value in requested)
    if requested and not paths.empty:
        wanted.update(paths.gene_id)
    failures = []

    def attempt(name, action):
        try:
            with timed_step(name):
                return action()
        except Exception:
            failures.append(name)
            note(f"FAILED {name}; see traceback below; continuing other plot types")
            traceback.print_exc()
        finally:
            plt.close("all")

    def read_history():
        history = load_history(args.result_dir, wanted, targets) if requested or args.model_diagnostic_plots == "true" else pd.DataFrame()
        return validate_history_axes(history, archives)

    history = attempt("saved RNA history validation", read_history)
    if requested:
        attempt("TF response plots", lambda: response_plots(
            archives, targets, labels, args.output_dir, args.tf_response_top_n, args.tf_response_min_r2,
            related_edges))
    else:
        attempt("cluster top 5 responses", lambda: cluster_top_responses(
            args.result_dir, archives, labels, args.output_dir, workers=args.workers))
    if requested:
        attempt("peak→gene response plots", lambda: peak_gene_response_plots(
            args.result_dir, paths, targets, labels, args.output_dir,
            args.tf_response_top_n, args.tf_response_min_r2, workers=args.workers))
    if args.model_diagnostic_plots == "true" and history is not None:
        attempt("effect vs input support", lambda: diagnostic_scatter(
            archives, history, targets, labels, args.output_dir))
    atac_file = args.trajectory_atac_file
    if requested and atac_file is None:
        try:
            import yaml
            # Prefer this run's frozen configuration. Never treat the mutable
            # project configuration as historical provenance.
            for filename in ("run_config.yaml", "config.yaml", "config_peak.yaml"):
                saved = args.result_dir / filename
                if saved.exists():
                    config = yaml.safe_load(saved.read_text())
                    configured = config.get("input", {}).get("atac_h5ad")
                    if configured:
                        atac_file = Path(configured)
                        if not atac_file.is_absolute():
                            atac_file = saved.parent / atac_file
                        note(f"ATAC provenance: {saved}: {atac_file}")
                        break
        except (OSError, ImportError, ValueError):
            pass
    if history is not None:
        attempt("WT/KO trajectory plots", lambda: trajectory_plots(
            args.result_dir, archives, history, requested, targets, labels, args.output_dir,
            args.trajectory_top_paths, atac_file,
            **({"selected_pathways": paths} if requested else {})))
    if failures:
        note("Failed plot steps: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
