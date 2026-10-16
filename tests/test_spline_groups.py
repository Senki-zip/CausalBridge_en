"""Natural cubic 2/3 input interactions, whole-response routing and strict evidence."""
import copy
from dataclasses import replace
import numpy as np
import pytest
from scipy.special import expit
from CausalBridge import spline_hybrid as s, saturating_hybrid as c
from CausalBridge._wt_aggregation import _weights,_full_wt_aggregates,validate_model_mode


def interaction(d=3,n=180,seed=12):
    rng=np.random.default_rng(seed);X=rng.uniform(.05,.95,(n,d));u=X-.5
    latent=3*u[:,0]+2*u[:,1]+8*u[:,0]*u[:,1]
    if d==3:latent+=u[:,2]+7*u[:,0]*u[:,1]*u[:,2]
    y=expit(latent)
    return c.TargetData('g',tuple((f'p{i}',0) for i in range(d)),X,y,
        np.tile(np.arange(n)[:,None],(1,d)),np.arange(n),('b',)*n,'raw_atac','raw_atac',
        dict(response_family='spline',spline_response_config=s.SplineConfig().to_dict()),cell_weights=np.eye(n))


@pytest.fixture(scope='module')
def fitted():
    data=interaction();model=c.fit_target(data,(0,1),c.HybridConfig(min_query_cell_ess=1))
    assert {len(g.indices) for g in model.groups}=={2,3}
    return data,model


@pytest.mark.parametrize('d',[2,3])
def test_real_interaction_group_accepted_natural_basis_penalty_and_C1(d):
    data=interaction(d);model=c.fit_target(data,(0,1));g=next(g for g in model.groups if len(g.indices)==d)
    assert isinstance(g,s.SplineGroupModel) and g.n_parameters==(6 if d==2 else 11)
    assert np.linalg.norm(g.coefficients[1+2*d:])>1e-4
    assert g.diagnostics['accepted'] and g.diagnostics['penalized_gain']>=model.config.min_penalized_gain
    P=s._group_penalty(g.medians,g.centers,g.qcenters,.5)
    shape=np.sum((P@g.coefficients)**2)
    assert shape==pytest.approx(g.diagnostics['shape_penalty'])
    baseline=np.column_stack([q.predict(data.X[:,j]) for j,q in enumerate(model.singles)])@model.weights
    mse=np.mean((g.predict(data.X[:,g.indices])-data.y)**2);base=np.mean((baseline-data.y)**2)
    J=mse+g.config['smoothness']*shape+g.config['group_interaction_penalty']*np.sum(g.coefficients[1+2*d:]**2)+g.n_parameters*base/len(data.y)
    assert J==pytest.approx(g.diagnostics['penalized_group_mse'])
    # Corner combinations and far mixed tails remain bounded and finite.
    import itertools
    z=np.array(list(itertools.product([-1e10,0,.5,1,1e10],repeat=d)))
    X=g.x_ranges[:,0]+z*(g.x_ranges[:,1]-g.x_ranges[:,0]);pred=g.predict(X)
    assert np.isfinite(pred).all() and (pred>=0).all() and (pred<=1).all()
    for j in range(d):
        for edge in [0.,1.]:
            z=np.full((3,d),.4);eps=1e-7;z[:,j]=[edge-eps,edge,edge+eps]
            X=g.x_ranges[:,0]+z*(g.x_ranges[:,1]-g.x_ranges[:,0]);v=g.predict(X)
            assert abs((v[1]-v[0])/eps-(v[2]-v[1])/eps)<1e-3
    # Natural boundary second derivative of each univariate interior is zero.
    for median in g.medians:
        np.testing.assert_allclose(s._natural(np.array([0.,1.]),median,.5,2),0,atol=1e-12)


def test_group_replaces_entire_response_unsupported_falls_back_without_reweighting(fitted):
    data,model=fitted;g=next(g for g in model.groups if len(g.indices)==3)
    support=c.CellSupport(data.cell_weights)
    # Query an actual supported training row with a small changed coordinate.
    old=data.X[50:51].copy();new=old.copy();new[0,0]+=1e-5
    result=c.predict_delta(model,old,new,branch_ids=('b',),cell_support=support)
    assert result.route=='group'
    chosen=next(g for g in model.groups if g.indices==result.group_indices)
    np.testing.assert_allclose(result.values,chosen.predict(new[:,chosen.indices])-chosen.predict(old[:,chosen.indices]),atol=1e-15)
    # Off-support query uses all original weighted singles, no additive correction.
    new=np.zeros_like(old)
    result=c.predict_delta(model,old,new,branch_ids=('b',),cell_support=support)
    assert result.route=='singles'
    expected=sum(w*(q.predict(new[:,j])-q.predict(old[:,j])) for j,(w,q) in enumerate(zip(model.weights,model.singles)))
    np.testing.assert_allclose(result.values,expected,atol=1e-15)
    assert result.diagnostics['active_input_renormalization'] is False
    assert result.diagnostics['group_corrections_added'] is False


@pytest.mark.parametrize('mode',['gain','rank','rows','celless','kernel','spread','solver','knot'])
def test_group_gate_rejections(mode,monkeypatch):
    data=interaction(2,n=80);cfg=c.HybridConfig();sc=s.SplineConfig();baseline=data.y.copy()
    if mode=='rank':data=replace(data,X=np.column_stack([data.X[:,0],data.X[:,0]]))
    elif mode=='rows':cfg=replace(cfg,min_group_rows=100)
    elif mode=='celless':cfg=replace(cfg,min_group_cell_ess=100)
    elif mode=='kernel':cfg=replace(cfg,min_group_effective_rows=100)
    elif mode=='spread':
        baseline=np.full_like(data.y,data.y.mean());cfg=replace(cfg,restart_prediction_spread_max=0)
    elif mode=='solver':sc=replace(sc,max_nfev=1)
    elif mode=='knot':
        X=data.X.copy();X[:50,0]=X[:,0].min();data=replace(data,X=X)
    group,diag=s.fit_group(data,(0,1),baseline,(0,1),cfg,sc)
    assert group is None and diag['accepted'] is False
    expected={'gain':'insufficient_penalized_gain','rank':'rank_deficient_or_ill_conditioned_geometry',
        'rows':'insufficient_group_rows','celless':'training_cell_ess','kernel':'kernel_effective_rows',
        'spread':'restart_prediction_spread','solver':'group_solver_not_converged','knot':'degenerate_natural_knot'}
    assert diag['reason']==expected[mode]


@pytest.mark.parametrize('field',['coefficients','medians','centers','qcenters','x_ranges','config','shape_penalty','interaction_square','n_parameters','solver_statuses','kernel_effective_rows','schema'])
def test_resigned_bundle_tampering_rejected(fitted,field):
    data,model=fitted;bundle=model.to_bundle();g=bundle['groups'][0]
    clone=c.TargetModel.from_bundle(bundle,cell_support=c.CellSupport(data.cell_weights))
    np.testing.assert_array_equal(clone.groups[0].predict(data.X[:,clone.groups[0].indices]),model.groups[0].predict(data.X[:,model.groups[0].indices]))
    if field in ('coefficients','medians','qcenters'):g[field][0]+=.1
    elif field in ('centers','x_ranges'):g[field][0][0]+=.1
    elif field=='config':g[field]['group_interaction_penalty']=.02
    elif field=='schema':g[field]='bounded_spline_group_unknown'
    elif field=='solver_statuses':g['diagnostics'][field]=[0]*3
    else:g['diagnostics'][field]=0
    bundle['fingerprint']=c._digest({k:v for k,v in bundle.items() if k!='fingerprint'})
    with pytest.raises(ValueError):c.TargetModel.from_bundle(bundle,cell_support=c.CellSupport(data.cell_weights))


def test_old_spline_target_v1_rejected(fitted):
    _,m=fitted;b=m.to_bundle();b['schema']='saturating_hybrid_spline_target_v1';b['fingerprint']=c._digest({k:v for k,v in b.items() if k!='fingerprint'})
    with pytest.raises(ValueError,match='schema'):c.TargetModel.from_bundle(b)


def test_removed_joint_mode_rejected_public_boundaries(tmp_path):
    from CausalBridge import run,io,kinetics,perturbation
    cfg={'atac_to_rna':{'model_mode':'joint_gene'},'output':{'dir':str(tmp_path)}}
    for fn in [validate_model_mode,run._joint_runtime_config,io._validate_config]:
        with pytest.raises(ValueError,match='removed'):fn(cfg)
    with pytest.raises(RuntimeError,match='removed'):run._validate_step5_checkpoint({},'joint_gene')
    # Reject before any NN fitting or simulation can occur.
    with pytest.raises(ValueError,match='removed'):
        perturbation.propagate_perturbation([],None,None,None,None,None,None,cfg)


@pytest.mark.parametrize('smooth',['hard','gaussian'])
def test_shared_WT_aggregation_matches_manual_legacy(smooth):
    import pandas as pd
    from scipy import sparse
    n=30;times=np.linspace(0,1,n);labels=np.repeat(np.arange(6),5)
    pt=pd.DataFrame({'pseudotime':times,'bin':labels});cfg={'pseudotime':{'smooth_kernel':smooth,'min_cells_per_bin':5}}
    kernel,_=_weights(pt,6,cfg,None);x=np.arange(n*2).reshape(n,2)/10
    A,R,valid=_full_wt_aggregates(kernel,sparse.csr_matrix(x),x,None,smooth=smooth)
    manual=kernel.toarray()/(np.asarray(kernel.sum(1))+(1e-10 if smooth=='gaussian' else 0))
    np.testing.assert_allclose(A,manual@x,atol=1e-14);np.testing.assert_allclose(R,A,atol=1e-14)
    assert valid.all()


@pytest.mark.parametrize('kind',['saturating_hybrid_peak_gene','saturating_hybrid_tf_peak'])
def test_real_group_jobs_spawn_serial_equivalent(kind):
    from dataclasses import fields
    from CausalBridge import saturating_transfer as t
    data=interaction();support=c.CellSupport(data.cell_weights);cfg=c.HybridConfig(min_query_cell_ess=1)
    payload=c._plain({f.name:getattr(data,f.name) for f in fields(data) if f.name not in ('cell_weights','cell_support_ref')})
    jobs=[('g',3,dict(data=payload,bounds=(0,1)),None)]
    serial=t._execute_target_jobs(iter(jobs),support,cfg,1,1,3,kind)
    parallel=t._execute_target_jobs(iter(jobs),support,cfg,2,1,3,kind)
    assert parallel==serial
    bundle=serial['g'][0]
    assert {len(g['indices']) for g in bundle['groups']}=={2,3}
    restored=c.TargetModel.from_bundle(bundle,cell_support=support)
    assert all(isinstance(g,s.SplineGroupModel) for g in restored.groups)


def test_removed_modules_absent_and_public_no_import_dependency():
    import importlib.util
    for name in ('joint_transfer','joint_estimator','joint_perturbation'):
        assert importlib.util.find_spec('CausalBridge.'+name) is None


def test_narrow_original_range_fit_and_strict_bundle_roundtrip():
    X=.5+1e-6*np.random.default_rng(0).uniform(size=(50,3))
    z=(X-X.min(0))/np.ptp(X,axis=0)
    y=expit(3*(z[:,0]-.5)+2*(z[:,1]-.5)+8*(z[:,0]-.5)*(z[:,1]-.5))
    data=replace(interaction(n=50),X=X,y=y)
    model=c.fit_target(data,(0,1))
    assert all(single.metrics['status']=='ok' for single in model.singles)
    assert model.groups  # Exercise the formerly fatal constructor, not just rejection.
    bundle=model.to_bundle();clone=c.TargetModel.from_bundle(bundle,cell_support=c.CellSupport(data.cell_weights))
    for original,restored in zip(model.groups,clone.groups):
        np.testing.assert_array_equal(original.predict(X[:,original.indices]),restored.predict(X[:,restored.indices]))
    g=model.groups[0]
    tolerance=s._standardization_tolerance(g.mean,g.scale,g.x_ranges,len(X))
    assert np.all(tolerance<1e-6) and np.max(np.abs(g.reference_u.mean(0)))>1e-10
    for field in ['reference_u','mean','scale','centers']:
        bad=copy.deepcopy(bundle);group=bad['groups'][0]
        if field=='reference_u':group[field][0][0]+=.001
        elif field=='centers':group[field][0][0]+=.001
        elif field=='mean':group[field][0]+=1e-8
        else:group[field][0]*=1.01
        bad['fingerprint']=c._digest({k:v for k,v in bad.items() if k!='fingerprint'})
        with pytest.raises(ValueError):c.TargetModel.from_bundle(bad)


@pytest.mark.parametrize('failure',[FloatingPointError,OverflowError,np.linalg.LinAlgError])
def test_expected_group_numerical_failure_keeps_executable_singles(monkeypatch,failure):
    data=interaction()
    def unusable(*args,**kwargs):raise failure('forced optional group numerical failure')
    monkeypatch.setattr(s,'_fit_group_impl',unusable)
    model=c.fit_target(data,(0,1))
    assert model.status=='ok' and not model.groups
    assert len(model.singles)==3 and all(single.metrics['status']=='ok' for single in model.singles)
    assert all(a['reason']=='group_numerical_failure' for a in model.diagnostics['group_attempts'])
    restored=c.TargetModel.from_bundle(model.to_bundle())
    old=data.X[:1];new=old.copy();new[0,0]=0
    result=c.predict_delta(restored,old,new,branch_ids=('b',))
    assert result.route=='singles' and np.isfinite(result.values).all()


def test_group_programming_errors_not_swallowed(monkeypatch):
    def bug(*args,**kwargs):raise TypeError('implementation defect')
    monkeypatch.setattr(s,'_fit_group_impl',bug)
    with pytest.raises(TypeError,match='implementation defect'):c.fit_target(interaction(),(0,1))


def test_unresolvable_coordinate_precision_rejects_optional_group():
    data=interaction(n=50);X=.5+1e-12*data.X
    data=replace(data,X=X)
    group,diagnostic=s.fit_group(data,(0,1),np.full(50,.5),(0,1),c.HybridConfig(),s.SplineConfig())
    assert group is None and diagnostic['reason']=='group_numerical_failure'
