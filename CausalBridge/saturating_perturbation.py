"""Branch-causal saturating simulation, with cell-level CP10k KO and fixed routes.

No old linear weights execute here. Group routes use one weighted preview and
one fixed-route final pass; any lost group support triggers a whole-intervention
weighted replay (at most three passes, never a splice of group/single histories).
"""
from dataclasses import replace
import json
import logging

import numpy as np
import pandas as pd

from . import saturating_hybrid as core
from .saturating_transfer import (GENE_TYPE,TF_TYPE,validate_saturating_config,
    validate_registry,load_models,_snapshot,_dense,_net_metrics,_lag)
from .saturating_projection import SCHEMA,cell_l0_from_saturating_payload


logger = logging.getLogger(__name__)


def _log_intervention_summary(result, cluster_label, threshold):
    """Summarize public endpoints and actual model-decomposition contributions."""
    public = result["perturbation_results"]
    affected = public[(public.status == "ok") & public.selected & ~public.direct_target &
        np.isfinite(public.delta_rna) & (public.delta_rna.abs() > max(0, threshold))]
    unavailable_genes = int(result["rna_unavailable_mask"].any(axis=0).sum())
    unavailable_peaks = int(result["atac_unavailable_mask"].any(axis=0).sum())
    context = "/".join(str(v) for v in (cluster_label, result["target_gene"]) if v)

    def changed_relations(table_name, source, target, value, kind):
        table = result[table_name]
        if table.empty:
            return 0
        changed = table[np.isfinite(table[value]) & (table[value] != 0)].copy()
        if "status" in changed:
            changed = changed[changed.status == "ok"]
        # Contributions have no status column: use the matching final-pass
        # support records, never failed event rows or preview routes.
        support = result["saturating_support_diagnostics"]
        valid = support[(support.relation_type == kind) & (support.status == "ok")]
        changed = changed.merge(valid[["bin", "target_id"]].drop_duplicates(),
            left_on=["bin", target], right_on=["bin", "target_id"], how="inner")
        return len(changed[[source, target]].dropna().drop_duplicates())

    tf_peak = changed_relations("saturating_tf_peak_contributions",
        "tf_gene", "peak_id", "delta_atac_raw", TF_TYPE)
    peak_gene = changed_relations("saturating_peak_attributions",
        "peak_id", "gene", "delta_l2", GENE_TYPE)
    logger.info("扰动完成 [%s]: 扰动基因=%d; 质量排除=%d; 不可用=%d（基因）/%d（peak）; 结果行=%d",
        context, affected.affected_gene.nunique(), len(result["excluded_gene_ids"]),
        unavailable_genes, unavailable_peaks, len(public))
    logger.info("变化关系（跨bin去重，模型分解）: TF→peak=%d; peak→gene=%d", tf_peak, peak_gene)


class _Replay(RuntimeError):
    pass


def _validate_screening_topology(edges, tf_weights, gene_registry, tf_registry):
    """Screen tables are identity/lag evidence only, never executable coefficients."""
    for frame, registry, target_column, input_column, lag_column in (
        (edges, gene_registry, "gene", "peak_id", "lag"),
        (tf_weights, tf_registry, "peak_id", "tf_gene", "tf_lag")):
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("saturating simulation requires its fitted screening tables")
        relations = {}
        if not frame.empty:
            if not {target_column, input_column}.issubset(frame):
                raise ValueError("saturating screening table identity columns missing")
            for row in frame.to_dict("records"):
                target = str(row.get(target_column + "_id", row[target_column]))
                source = str(row.get(input_column + "_id", row[input_column]))
                lag = _lag(row.get(lag_column, 1 if registry["type"]==GENE_TYPE else 0))
                relations.setdefault(target, set()).add((source, lag))
        if set(relations) != set(registry["targets"]) | set(registry["failures"]):
            raise ValueError("saturating screening topology differs from fitted registry; refit Step 5/6")
        for target, bundle in registry["targets"].items():
            excluded = bundle["provenance"].get("excluded_single_relations", {}).get("records", ())
            expected = {tuple(key) for key in bundle["input_keys"]} | {tuple(record["input_key"]) for record in excluded}
            if relations[target] != expected:
                raise ValueError("saturating screening input keys/lags differ from fitted registry")
        for target, failure in registry["failures"].items():
            if "excluded_single_relations" in failure:
                expected = {tuple(record["input_key"]) for record in failure["excluded_single_relations"]["records"]}
                if relations[target] != expected:
                    raise ValueError("saturating failed screening input keys/lags differ from fitted registry")


def _rows(model,context):
    inverse={int(s):b for b,s in enumerate(context["bin_to_support"]) if s>=0}
    return np.array([inverse[int(s)] for s in model.support_rows],dtype=int)


def _inputs(model,rows,states,index):
    return np.column_stack([states[rows-lag,index[key]] for key,lag in model.input_keys])


def _verify_wt_metrics(registry,models,snapshot):
    """The live frozen WT context verifies stored scores/quality, not just hashes."""
    context=snapshot["context"]
    gene_index={g:i for i,g in enumerate(context["gene_ids"])}
    peak_index={p:i for i,p in enumerate(context["peak_ids"])}
    for target,model in models.items():
        rows=_rows(model,context)
        if registry["type"]==GENE_TYPE:
            X=_inputs(model,rows,snapshot["A"],peak_index)
            y=np.log1p(snapshot["R"][rows,gene_index[target]])
            if model.bounds!=core.rna_output_bounds(y,model.config):
                raise ValueError("RNA model cap differs from frozen WT range")
        else:
            X=_inputs(model,rows,snapshot["R"],gene_index)
            y=snapshot["A"][rows,peak_index[target]]
            from .atac_normalization import atac_output_bounds
            if model.bounds!=atac_output_bounds(y,context.get("atac_representation",{"unit":"raw_atac"})):
                raise ValueError("TF output cap differs from frozen WT representation/range")
        predictions=np.column_stack([single.predict(X[:,i]) for i,single in enumerate(model.singles)])
        adjacent=(np.diff(model.target_rows)==1)&(np.array(model.branch_ids[1:])==np.array(model.branch_ids[:-1]))
        adjacent &= np.all(np.diff(model.source_bins,axis=0)==1,axis=1)
        scores=np.mean(abs(np.diff(predictions,axis=0)[adjacent]),axis=0) if adjacent.any() else np.zeros(len(model.input_keys))
        if not np.array_equal(scores,np.asarray(model.scores)):
            raise ValueError("frozen WT response scores differ from fitted registry")
        metrics=_net_metrics(model,X,y)
        stored=registry["target_metrics"][target]
        for field in ("status","n_rows","single_fit_statuses","quality_scope","validation_performed"):
            if metrics[field]!=stored[field]:
                raise ValueError("stored target quality evidence differs from frozen WT")
        for field in ("train_r2","training_rmse"):
            value,reference=metrics[field],stored[field]
            if value is None or reference is None:
                if value is not reference:raise ValueError("stored target quality differs from frozen WT")
            elif not np.isclose(value,reference,rtol=1e-10,atol=1e-12):
                raise ValueError("stored target quality differs from frozen WT")


def _closure(targets,seed_peak,genes,tfmodels,depth,*,tf_keys=None,work_budget=None):
    selected_genes=set(targets);selected_peaks=set(() if seed_peak is None else (seed_peak,))
    if depth is None or depth=="unlimited":
        rounds=len(genes)+len(tfmodels)+1
    else:
        rounds=core._integer(depth,"subnetwork_depth")
    keys={p:m.input_keys for p,m in tfmodels.items()} if tf_keys is None else tf_keys
    gene_to_peaks={};peak_to_genes={}
    # Keep one entry per input row, including duplicate/lagged relations.
    for peak,inputs in keys.items():
        for gene,_ in inputs:gene_to_peaks.setdefault(gene,[]).append(peak)
    for gene,model in genes.items():
        for peak,_ in model.input_keys:peak_to_genes.setdefault(peak,[]).append(gene)
    work=0
    gene_frontier=set(selected_genes);peak_frontier=set(selected_peaks)
    def visit(frontier,index):
        nonlocal work
        reached=set()
        for source in frontier:
            for target in index.get(source,()):
                work+=1
                if work_budget is not None and work>work_budget:
                    raise ValueError(f"saturating subnetwork closure work-edge budget exceeded: "
                        f"budget={work_budget}, scanned={work} reachable relation rows")
                reached.add(target)
        return reached
    for _ in range(rounds):
        new_peaks=visit(gene_frontier,gene_to_peaks)-selected_peaks
        selected_peaks.update(new_peaks)
        new_genes=visit(peak_frontier|new_peaks,peak_to_genes)-selected_genes
        selected_genes.update(new_genes)
        gene_frontier=new_genes;peak_frontier=set()
        if not gene_frontier:break
    return selected_genes,selected_peaks


def _contributions(model,old,new,route,group_indices):
    if route=="group":
        group=next(g for g in model.groups if g.indices==group_indices)
        value=old[:,group.indices].copy();previous=group.predict(value)
        output=[]
        for position,index in enumerate(group.indices):
            value[:,position]=new[:,index]
            current=group.predict(value);output.append((index,float(current[0]-previous[0])))
            previous=current
        return output,"ordered_telescoping_not_causal"
    return [(i,float(weight*(single.predict(new[:,i])[0]-single.predict(old[:,i])[0])))
            for i,(weight,single) in enumerate(zip(model.weights,model.singles))],"frozen_alpha_single_response"


def _simulate(rna,atac,pt,config,gene_registry,tf_registry,*,targets=(),seed_peak=None,bin_plan=None,cluster_label=""):
    cfg=validate_saturating_config(config)
    gmodels,gsupport=load_models(gene_registry,GENE_TYPE)
    tmodels,tsupport=load_models(tf_registry,TF_TYPE)
    snapshot=_snapshot(rna,atac,pt,config,bin_plan)
    context=snapshot["context"]
    representation=context.get("atac_representation",{"unit":"raw_atac"})
    continuous=representation["unit"]=="normalized_atac_log1p"
    ceiling=float(np.log1p(representation["target_sum"])) if continuous else 1.
    for registry in (gene_registry,tf_registry):
        from .saturating_transfer import response_context
        expected_context = response_context(context, config, registry["type"])
        if registry["context"]["fingerprint"]!=expected_context["fingerprint"] or core.HybridConfig.from_dict(registry["config"])!=cfg:
            raise ValueError("saturating simulation differs from fitted WT context/config")
    _verify_wt_metrics(gene_registry,gmodels,snapshot)
    _verify_wt_metrics(tf_registry,tmodels,snapshot)
    genes,peaks=context["gene_ids"],context["peak_ids"]
    gi,ai={v:i for i,v in enumerate(genes)},{v:i for i,v in enumerate(peaks)}
    if not targets and seed_peak is None:
        raise ValueError("saturating intervention requires a target")
    if any(g not in gi for g in targets) or seed_peak is not None and seed_peak not in ai:
        raise ValueError("intervention target absent from fitted axes")
    p=config.get("perturbation",{})
    min_r2=core._number(p.get("min_r2_threshold",.3),"min_r2_threshold")
    excluded={g for g in gmodels if gene_registry["target_metrics"][g]["status"]!="ok" or
        gene_registry["target_metrics"][g]["train_r2"] is None or gene_registry["target_metrics"][g]["train_r2"]<min_r2}
    excluded.update(gene_registry["failures"])
    eligible={g:m for g,m in gmodels.items() if g not in excluded}
    eligible_tf={a:m for a,m in tmodels.items() if tf_registry["target_metrics"][a]["status"]!="budget_exhausted"}
    tf_keys={a:tuple(key for key in m.input_keys if not p.get("min_lag_per_peak_filter",False)
                    or key[1]==min(lag for _,lag in m.input_keys)) for a,m in eligible_tf.items()}
    depth = p.get("peak_ko", {}).get("depth", 1) if seed_peak is not None else p.get("subnetwork_depth","unlimited")
    work_budget=p.get("subnetwork_unlimited_max_work_edges")
    if work_budget is not None:work_budget=core._integer(work_budget,"subnetwork_unlimited_max_work_edges",1)
    selected_genes,selected_peaks=_closure(targets,seed_peak,eligible,eligible_tf,depth,
        tf_keys=tf_keys,work_budget=work_budget)
    memberships=sum(len(m.input_keys) for g,m in eligible.items() if g in selected_genes)+sum(len(m.input_keys) for a,m in eligible_tf.items() if a in selected_peaks)
    if memberships>core._integer(p.get("subnetwork_unlimited_max_edges",100000),"subnetwork_unlimited_max_edges",1):
        raise MemoryError("saturating executable subnetwork edge budget exceeded")
    active_g={g:m for g,m in eligible.items() if g in selected_genes and g not in targets}
    active_t={a:m for a,m in eligible_tf.items() if a in selected_peaks and a!=seed_peak}
    models={(GENE_TYPE,g):m for g,m in active_g.items()}|{(TF_TYPE,a):m for a,m in active_t.items()}
    declared={key:tuple(k for k in m.input_keys if k[0] in (selected_peaks if key[0]==GENE_TYPE else selected_genes)
                       and (key[0]==GENE_TYPE or k in tf_keys[key[1]])) for key,m in models.items()}
    cells=_dense(rna.X).astype(float)
    factors=np.ones_like(cells)
    strength=core._number(p.get("ko_strength",1.),"ko_strength")
    if not 0<=strength<=1:
        raise ValueError("saturating gene ko_strength must lie in [0,1]")
    for gene in targets:factors[:,gi[gene]]=1-strength
    # Untargeted cells/bins must remain bit-identical to WT. Applying the
    # expm1/log1p round trip to factors==1 introduces ulp changes in unrelated
    # TF inputs, outside the whole-intervention declared topology. Reaggregating
    # every column can also differ from legacy WT gaussian aggregation.
    target_columns=[gi[gene] for gene in targets]
    direct_cells=cells.copy()
    direct_R=snapshot["R"].copy()
    with np.errstate(over="raise",invalid="raise"):
        if strength != 0 and target_columns:
            direct_cells[:,target_columns]=np.log1p(np.expm1(cells[:,target_columns])*factors[:,target_columns])
            direct_R[:,target_columns]=np.asarray(snapshot["operator"]@direct_cells[:,target_columns])
    WT_R,WT_A=snapshot["R"],snapshot["A"]
    seed_values=None
    direct_peak_clip_count=0
    if seed_peak is not None:
        peak_cfg=p.get("peak_ko",{})
        amount=core._number(peak_cfg.get("strength",-1.),"peak strength")
        mode=peak_cfg.get("mode","relative")
        if mode=="relative":
            if amount < -1 or (not continuous and amount>1):raise ValueError("relative peak strength outside supported range")
            if continuous:
                original_cells=_dense(atac.X)[:,ai[seed_peak]]
                with np.errstate(over="ignore"):
                    desired=np.expm1(original_cells)*(1+amount)
                direct_peak_clip_count=int((desired>representation["target_sum"]).sum())
                perturbed_cells=np.log1p(np.clip(desired,0,representation["target_sum"]))
                seed_values=WT_A[:,ai[seed_peak]].copy() if amount==0 else np.asarray(snapshot["operator"]@perturbed_cells).ravel()
            else:
                seed_values=np.clip(WT_A[:,ai[seed_peak]]*(1+amount),0,1)
        elif mode=="absolute":
            if not 0<=amount<=ceiling:raise ValueError("absolute peak strength outside declared ATAC state ceiling")
            seed_values=np.asarray(snapshot["operator"]@np.full(atac.n_obs,amount)).ravel() if continuous else np.full(len(WT_A),amount)
        else:raise ValueError("unsupported saturating peak intervention mode")
    row_lookup={key:set(_rows(m,context).tolist()) for key,m in models.items()}
    def run(locked=None):
        R=direct_R.copy();A=WT_A.copy()
        if seed_peak is not None:A[:,ai[seed_peak]]=seed_values
        rbad=np.zeros(R.shape,bool);abad=np.zeros(A.shape,bool)
        rorigins={};aorigins={}
        events=[];clips=[];support_rows=[];attributions=[];tf_contributions=[]
        def record_failure(kind,target,m,b,enabled,status,reason,inherited):
            audit=dict(bin=b,branch=context["bin_branch_ids"][b],target_id=target,
                layer="peak_gene" if kind==GENE_TYPE else "tf_peak",relation_type=kind,
                gene=target if kind==GENE_TYPE else None,peak_id=target if kind==TF_TYPE else None,
                status=status,reason=reason,route="unavailable",
                input_features_json=json.dumps(m.input_keys),
                enabled_input_features_json=json.dumps(enabled),
                unavailable_origins_json=json.dumps(sorted(inherited)),
                changed_inputs=declared[(kind,target)])
            support_rows.append(audit)
            events.append(dict(audit,diagnostic=reason))
        locked={} if locked is None else locked
        locked_models={key:replace(models[key],groups=(next(g for g in models[key].groups if g.indices==indices),)) for key,indices in locked.items()}
        for start,end in snapshot["boundaries"]:
            for b in range(start,end):
                # RNA lag1 is finalized before all same-bin TF lag0 queries.
                for kind,table,WT,current,index,bad,inputbad,inputidx,shared,origins,inputorigins in (
                    (GENE_TYPE,active_g,WT_A,A,ai,rbad,abad,ai,gsupport,rorigins,aorigins),
                    (TF_TYPE,active_t,WT_R,R,gi,abad,rbad,gi,tsupport,aorigins,rorigins)):
                    for target,m in table.items():
                        key=(kind,target)
                        if b not in row_lookup[key]:continue
                        enabled=m.input_keys if kind==GENE_TYPE else tf_keys[target]
                        sources=[(b-lag,name) for name,lag in enabled]
                        target_index=gi[target] if kind==GENE_TYPE else ai[target]
                        if any(inputbad[s,inputidx[name]] for s,name in sources):
                            bad[b,target_index]=True
                            inherited=set()
                            for s,name in sources:
                                if inputbad[s,inputidx[name]]:inherited.update(inputorigins.get((s,inputidx[name]),{name+":unavailable"}))
                            origins[b,target_index]=inherited
                            record_failure(kind,target,m,b,enabled,"unavailable_upstream",
                                "enabled input state is unavailable",inherited)
                            continue
                        old=_inputs(m,np.array([b]),WT,index)
                        new=_inputs(m,np.array([b]),current,index)
                        if kind==TF_TYPE:
                            for i,input_key in enumerate(m.input_keys):
                                if input_key not in enabled:new[:,i]=old[:,i]
                        selected=locked_models.get(key,m)
                        try:
                            answer=core.predict_delta(selected,old,new,changed_inputs=declared[key],
                                branch_ids=(context["bin_branch_ids"][b],),
                                cell_support=shared if key in locked else None)
                            if key in locked and answer.route not in ("group","no_change","no_information"):
                                raise _Replay(f"{kind}/{target}: locked group lost support")
                            delta=float(answer.values[0])
                            if not np.isfinite(delta):raise FloatingPointError("nonfinite response")
                        except (FloatingPointError,OverflowError) as exc:
                            bad[b,target_index]=True
                            origins[b,target_index]={target+":numerical_failure"}
                            record_failure(kind,target,m,b,enabled,"numerical_failure",str(exc),origins[b,target_index])
                            continue
                        support_rows.append(dict(bin=b,target_id=target,relation_type=kind,route="group" if key in locked else "weighted_singles",
                            gene=target if kind==GENE_TYPE else None, peak_id=target if kind==TF_TYPE else None,
                            branch=context["bin_branch_ids"][b],status="ok",
                            group_indices=answer.group_indices,changed_inputs=declared[key],
                            saturation_out_of_WT_support=answer.diagnostics["saturation_out_of_WT_support"]))
                        event = dict(bin=b, gene=target if kind==GENE_TYPE else None,
                            peak_id=target if kind==TF_TYPE else None, target_id=target,
                            relation_type=kind, status="ok", delta_native=delta,
                            route=answer.route, group_indices=answer.group_indices)
                        events.append(event)
                        if delta==0:continue
                        contributions,method=_contributions(selected,old,new,answer.route,answer.group_indices)
                        if kind==GENE_TYPE:
                            before=np.log1p(direct_R[b,target_index])+delta
                            after=float(np.clip(before,0,m.bounds[1]))
                            R[b,target_index]=np.expm1(after)
                            if before!=after:clips.append(dict(bin=b,gene=target,pre_clip_L2=before,post_clip_L2=after,
                                status="lower_floor" if before<0 else "upper_cap",state_unit="log1p_rna_log1p_cp10k"))
                            for i,value in contributions:
                                attributions.append(dict(bin=b,gene=target,peak_id=m.input_keys[i][0],delta_l2=value,
                                    method=method,causal_claim=False))
                        else:
                            before=WT_A[b,target_index]+delta
                            A[b,target_index]=np.clip(before,0,m.bounds[1] if continuous else 1)
                            if before!=A[b,target_index]:event["status"]="ATAC_physical_clip"
                            for i,value in contributions:
                                if m.input_keys[i] not in enabled:continue
                                tf_contributions.append(dict(bin=b,source_bin=b-m.input_keys[i][1],tf_gene=m.input_keys[i][0],
                                    peak_id=target,delta_atac_raw=value,method=method,causal_claim=False))
        return dict(R=R,A=A,rbad=rbad,abad=abad,events=events,clips=clips,support=support_rows,
                    attributions=attributions,tf_contributions=tf_contributions,rorigins=rorigins,aorigins=aorigins)
    preview=run()
    routes={}
    for key,m in models.items():
        if not m.groups or not declared[key]:continue
        rows=_rows(m,context)
        old_states,new_states,index,shared=(WT_A,preview["A"],ai,gsupport) if key[0]==GENE_TYPE else (WT_R,preview["R"],gi,tsupport)
        old=_inputs(m,rows,old_states,index);new=_inputs(m,rows,new_states,index)
        if key[0]==TF_TYPE:
            for i,input_key in enumerate(m.input_keys):
                if input_key not in tf_keys[key[1]]:new[:,i]=old[:,i]
        answer=core.predict_delta(m,old,new,changed_inputs=declared[key],branch_ids=m.branch_ids,cell_support=shared)
        if answer.route=="group":routes[key]=answer.group_indices
    replay_count=0;replay_reason=None
    if routes:
        try:final=run(routes)
        except _Replay as exc:
            final=run();routes={};replay_count=1;replay_reason=str(exc)
    else:final=preview
    label=",".join(targets) if targets else "PEAK:"+seed_peak
    native=np.log1p(final["R"])-np.log1p(direct_R)
    payload=dict(effect_schema_version=SCHEMA,intervention_unit="cp10k_before_log1p_and_aggregation",
        target_genes=tuple(targets),control_cell_states=cells,direct_cell_states=direct_cells,
        direct_cell_cp10k_factors=factors,control_bin_states=WT_R,direct_bin_states=direct_R,
        perturbed_bin_states=final["R"],native_bin_effect=native,
        cell_bin_weights=_dense(snapshot["cell_bin_weights"]),bin_cell_operator=_dense(snapshot["operator"]),
        features=tuple(genes),cell_ids=np.asarray(rna.obs_names),bins=pt.bin.to_numpy(),
        pseudotime=pt.pseudotime.to_numpy(),branch=np.asarray(context["cell_branch_ids"]),
        bin_branch_ids=tuple(context["bin_branch_ids"]),bin_plan_centers=np.asarray(context["bin_centers"]),
        bin_plan_fingerprint=context["bin_plan_fingerprint"],state_unit="rna_log1p_cp10k",
        native_effect_unit="delta_log1p_rna_log1p_cp10k",bin_state_mode="frozen_shared_operator",
        invalid_state=bool(final["rbad"].any()),local_events=final["events"],
        target_gene=label, reconstruction_policy="interpolate_native_L2_on_direct_cell_L1_then_decode_CP10k",
        reconstruction_limitation="interpolated_cell_states_are_not_exact_bin_inverse_or_raw_UMI_counts",
        registry_fingerprint=gene_registry["fingerprint"],tf_registry_fingerprint=tf_registry["fingerprint"],
        route_replay_count=replay_count,route_replay_reason=replay_reason)
    if continuous:
        payload["atac_representation"]=representation
    # Unavailable columns abstain; do not contaminate unrelated valid endpoints.
    # The public payload remains invalid and cannot be silently projected.
    _,_,pert_cp=cell_l0_from_saturating_payload(dict(payload,invalid_state=False))
    pert_cp[:,final["rbad"].any(axis=0)]=np.nan
    control_cp=np.expm1(cells)
    endpoints=[];effects={}
    for gene,j in gi.items():
        control=float(control_cp[:,j].mean());pert=float(pert_cp[:,j].mean())
        status="unavailable" if final["rbad"][:,j].any() else "zero_control" if control==0 else "zero_perturbed_floor" if pert==0 else "ok"
        if gene in excluded and gene not in targets:status="quality_filtered"
        effect=float(np.log2(pert/control)) if status=="ok" else np.nan
        effects[gene]=effect
        endpoints.append(dict(target_gene=label,affected_gene=gene,control_mean_cp10k=control,
            perturbed_mean_cp10k=pert,delta_rna=effect,status=status,direct_target=gene in targets,
            selected=gene in selected_genes,converged=status=="ok",quality_scope="target_full_training",
            fit_status=gene_registry["target_metrics"].get(gene,gene_registry["failures"].get(gene,{})).get("status"),
            train_r2=gene_registry["target_metrics"].get(gene,{}).get("train_r2")))
    endpoint_table=pd.DataFrame(endpoints)
    threshold=core._number(p.get("min_delta_threshold",0.),"min_delta_threshold")
    public=endpoint_table[(endpoint_table.status=="ok")&endpoint_table.selected&~endpoint_table.direct_target&
        (endpoint_table.delta_rna.abs()>max(0,threshold))].copy()
    public["mechanism"]="ATAC-mediated";public["mediated_by_atac"]=True
    public["perturbation_mode"]="saturating_hybrid";public["cluster"]=cluster_label
    public["is_fallback"]=False;public["propagation_depth"]=p.get("subnetwork_depth","unlimited")
    da=final["A"].mean(axis=0)-WT_A.mean(axis=0);std=WT_A.std(axis=0)
    z=np.divide(da,std,out=np.zeros_like(da),where=std>1e-10)
    unavailable_peaks=final["abad"].any(axis=0)
    da[unavailable_peaks]=np.nan;z[unavailable_peaks]=np.nan
    delta_atac=pd.Series(da,index=peaks);zscores=pd.Series(z,index=peaks)
    from .perturbation import _build_atac_changes_df
    changes=_build_atac_changes_df(delta_atac,atac,atac_z_score=zscores,z_threshold=p.get("atac_significance_zscore",2.))
    if unavailable_peaks.any():
        failed_ids=[peak for peak,unavailable in zip(peaks,unavailable_peaks) if unavailable]
        failures=atac.var.reindex(columns=["chr","start","end"]).loc[failed_ids].copy()
        failures["peak_id"]=failures.index
        failures["delta_accessibility"]=np.nan;failures["z_score"]=np.nan
        failures["is_significant"]=False;failures["status"]="unavailable"
        failures["unavailable_bin_count"]=[int(final["abad"][:,ai[peak]].sum()) for peak in failed_ids]
        failures["unavailable_origins_json"]=[json.dumps(sorted(set().union(*(
            final["aorigins"].get((b,ai[peak]),{peak+":unavailable"})
            for b in np.flatnonzero(final["abad"][:,ai[peak]]))))) for peak in failed_ids]
        changes["status"]="ok"
        changes=pd.concat([changes,failures.reset_index(drop=True)],ignore_index=True)
    if continuous:
        changes["atac_state_unit"]=representation["unit"]
        changes["atac_intervention_unit"]="normalized_counts_before_log1p" if seed_peak is not None and p.get("peak_ko",{}).get("mode","relative")=="relative" else representation["unit"]
    tables=dict(saturating_endpoints=endpoint_table,
        saturating_peak_attributions=pd.DataFrame(final["attributions"]),
        saturating_tf_peak_contributions=pd.DataFrame(final["tf_contributions"]),
        saturating_support_diagnostics=pd.DataFrame(final["support"]),
        l1_floor_diagnostics=pd.DataFrame(final["clips"]))
    for table in tables.values():table["target_gene"]=label
    result=dict(target_gene=label,target_genes=list(targets),ko_genes_applied=list(targets),
        perturbation_results=public,delta_rna=effects,delta_atac=delta_atac,atac_z_score=zscores,
        atac_changes=changes,prediction_mode="saturating_hybrid",effect_schema_version=SCHEMA,
        rna_state_history=final["R"],atac_state_history=final["A"],control_rna_states=WT_R,
        control_atac_states=WT_A,saturating_native=native,projection_inputs=payload,
        quality_policy="pruned_gene_models",excluded_gene_ids=sorted(excluded),
        rna_unavailable_mask=final["rbad"],atac_unavailable_mask=final["abad"],local_events=pd.DataFrame(final["events"]),
        rna_unavailable_origins={f"{b}:{genes[j]}":sorted(v) for (b,j),v in final["rorigins"].items()},
        atac_unavailable_origins={f"{b}:{peaks[j]}":sorted(v) for (b,j),v in final["aorigins"].items()},
        pathway_edges=pd.DataFrame(),aggregated_tf_peak_edges=pd.DataFrame(),upstream_tfs=pd.DataFrame(),
        saturating_gene_metrics=pd.DataFrame([dict(gene=g,**v) for g,v in gene_registry["target_metrics"].items()]),
        saturating_context_metadata=dict(model_mode="saturating_hybrid",effect_schema_version=SCHEMA,
            registry_fingerprint=gene_registry["fingerprint"],tf_registry_fingerprint=tf_registry["fingerprint"],
            context_fingerprint=context["fingerprint"],alpha_frozen=True,active_input_renormalization=False,
            route_policy="preview_locked_final_whole_intervention_replay",route_passes=1 if not routes and not replay_count else 3 if replay_count else 2,
            route_replay_count=replay_count,route_replay_reason=replay_reason,
            locked_groups={kind+":"+target:list(indices) for (kind,target),indices in routes.items()},
            plateau_and_support_thresholds="experimental",raw_RNA_KO=True),**tables)
    if seed_peak is not None:
        original=float(WT_A[:,ai[seed_peak]].mean());changed=float(final["A"][:,ai[seed_peak]].mean())
        result.update(peak_id=seed_peak, prediction_mode="peak_saturating_hybrid",
            baseline_accessibility=original, perturbed_accessibility=changed,
            delta_accessibility=changed-original, n_affected_edges=len(active_g))
    if continuous:
        result["saturating_context_metadata"]["atac_representation"]=representation
        result["saturating_context_metadata"]["direct_peak_physical_clip_cells"]=direct_peak_clip_count
        result["saturating_context_metadata"]["atac_intervention_unit"]="normalized_counts_before_log1p" if seed_peak is not None and p.get("peak_ko",{}).get("mode","relative")=="relative" else representation["unit"]
        result["atac_state_unit"]=representation["unit"]
        for frame in (result["saturating_support_diagnostics"],result["saturating_tf_peak_contributions"],result["saturating_peak_attributions"]):
            frame["atac_state_unit"]=representation["unit"]
    from .saturating_pathways import build_pathways, KEY
    for frame in tables.values():
        frame["source_context"] = str(cluster_label or "global")
    path_events = result["local_events"].copy()
    path_events["target_gene"] = label
    path_events["source_context"] = str(cluster_label or "global")
    result[KEY] = build_pathways(result["saturating_tf_peak_contributions"],
        result["saturating_peak_attributions"], result["saturating_endpoints"],
        branch_boundaries=snapshot["boundaries"], control_atac_states=WT_A,
        atac_state_history=final["A"], peak_ids=peaks, atac_unavailable_mask=final["abad"],
        local_events=path_events, rna_clips=result["l1_floor_diagnostics"])
    result["pathway_edges"] = result[KEY].copy()
    _log_intervention_summary(result, cluster_label, threshold)
    return result


def propagate_saturating(target_genes,rna,atac,pt,edges,transfers,tf_weights,config,*,
        transfer_models=None,tf_model_registry=None,gene_types=None,bin_plan=None,cluster_label=""):
    validate_saturating_config(config)
    if any(str((gene_types or {}).get(g,"TF")).upper()!="TF" for g in target_genes):
        raise ValueError("saturating_hybrid does not support TARGET proxy interventions")
    validate_registry(transfer_models,expected_type=GENE_TYPE);validate_registry(tf_model_registry,expected_type=TF_TYPE)
    _validate_screening_topology(edges,tf_weights,transfer_models,tf_model_registry)
    groups=[list(target_genes)] if config.get("perturbation",{}).get("joint_knockout",False) else [[g] for g in target_genes]
    if not groups or any(not g for g in groups):raise ValueError("empty saturating gene intervention")
    results=[_simulate(rna,atac,pt,config,transfer_models,tf_model_registry,targets=group,
        bin_plan=bin_plan,cluster_label=cluster_label) for group in groups]
    output=dict(results[0])
    for name in ("perturbation_results","atac_changes","saturating_endpoints","saturating_peak_attributions",
                 "saturating_tf_peak_contributions","saturating_support_diagnostics","l1_floor_diagnostics","saturating_gene_metrics",
                 "saturating_pathways", "pathway_edges"):
        output[name]=pd.concat([r[name] for r in results],ignore_index=True)
    output["projection_inputs"]=[r["projection_inputs"] for r in results]
    output["saturating_interventions"]=results
    output["saturating_context_metadata"] = dict(output["saturating_context_metadata"],
        interventions=[dict(r["saturating_context_metadata"], target_gene=r["target_gene"]) for r in results])
    return output


def perturb_peak_saturating(peak_id,rna,atac,pt,edges,transfers,tf_weights,config,*,
        transfer_models=None,tf_model_registry=None,bin_plan=None,cluster_label=""):
    validate_saturating_config(config)
    validate_registry(transfer_models,expected_type=GENE_TYPE);validate_registry(tf_model_registry,expected_type=TF_TYPE)
    _validate_screening_topology(edges,tf_weights,transfer_models,tf_model_registry)
    from .perturbation import _match_peak
    matched, info = _match_peak(peak_id, atac, config.get("perturbation", {}).get("peak_ko", {}).get("match", "exact"))
    if matched is None:
        return dict(error="peak_not_found", requested_peak=peak_id, match_info=info)
    output = _simulate(rna,atac,pt,config,transfer_models,tf_model_registry,seed_peak=matched,
        bin_plan=bin_plan,cluster_label=cluster_label)
    output.update(requested_peak=peak_id, match_info=info)
    return output
