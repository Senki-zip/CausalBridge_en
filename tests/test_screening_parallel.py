"""Parallel screening preserves ordered statistics, routes and fit contracts."""
from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from CausalBridge import granger, kinetics, run, screening_parallel as screen
from test_local_tf_batch_pearson import assert_exact


@pytest.mark.parametrize('key', ['granger_workers', 'local_lag_workers'])
@pytest.mark.parametrize('value', [0,-1,True,False,'4',1.5,None])
def test_bad_worker_options(key,value):
    with pytest.raises(ValueError,match='positive integer'):
        screen.screening_workers({'execution':{key:value}},key)


@pytest.mark.parametrize('key', ['granger_workers', 'local_lag_workers'])
def test_worker_defaults(key):
    assert screen.screening_workers({},key)==1
    assert screen.screening_workers({'execution':{key:4}},key)==4
    with pytest.raises(ValueError,match='mapping'):
        screen.screening_workers({'execution':[]},key)


@pytest.mark.parametrize('lag0', [False,True])
@pytest.mark.parametrize('branches', [False,True])
def test_actual_granger_parallel_is_bitwise_equal(monkeypatch,lag0,branches,capsys):
    rng=np.random.default_rng(91)
    n,ng,npk=50,30,40
    obs=pd.DataFrame(index=[f'c{i}' for i in range(n)])
    def data(matrix,prefix):
        count=matrix.shape[1]
        return ad.AnnData(matrix,obs=obs.copy(),var=pd.DataFrame({
            'chr':['1']*count,'start':np.arange(count)*100,'end':np.arange(count)*100+50},
            index=[f'{prefix}{i}' for i in range(count)]))
    rna=data(rng.normal(size=(n,ng)),'G');atac=data(rng.normal(size=(n,npk)),'P')
    rna.X[:,0]=0;atac.X[:,0]=1
    # Rank-deficient full designs and low-variance signals retain skip semantics.
    atac.X[:,1]=rna.X[:,1]
    candidates=[(p,g) for p in atac.var_names for g in rna.var_names]
    candidates += [('absent','G0')]
    pt=pd.DataFrame({'pseudotime':np.linspace(0,1,n)},index=obs.index)
    if branches:pt.attrs['branch_boundaries']=[(0,20),(20,50)]
    cfg={'pseudotime':{'n_bins':n,'min_cells_per_bin':1},
         'granger':dict(distance_thresh=10000,max_lag=1,ablation_lag0=lag0,correction_method='fdr_bh',
            significance_threshold=.05,min_effect_size=.1,reference_bins=50,soft_threshold=True,min_composite_score=.3)}
    monkeypatch.setattr(granger,'_adaptive_rebin',lambda *a,**k:(np.arange(n),n))
    monkeypatch.setattr(granger,'_bin_expression',lambda data,*a,**k:data.X)
    monkeypatch.setattr(granger,'_generate_candidate_pairs',lambda *a:candidates)
    before=deepcopy(cfg)
    serial=granger.granger_test(rna,atac,pt.copy(deep=True),cfg)
    cfg['execution']={'granger_workers':2}
    parallel=granger.granger_test(rna,atac,pt.copy(deep=True),cfg)
    pd.testing.assert_frame_equal(serial,parallel,check_exact=True)
    assert cfg['granger']==before['granger']
    output=capsys.readouterr().err
    assert output.count('Step 4 Granger: 0/1201')==2
    assert output.count('Step 4 Granger: 1201/1201')==2


@pytest.mark.parametrize('mode', ['finite','float32','nonfinite','branches','plan'])
def test_local_lag_parallel_exact_routes(mode,capsys):
    rng=np.random.default_rng(11)
    n,count=18,260
    x=rng.normal(size=(n,4));y=rng.normal(size=(n,count))
    y[:,0]=x[:,0];x[:,1]=1
    bounds=None;plan=None
    if mode=='float32':x=x.astype('float32');y=y.astype('float32')
    if mode=='nonfinite':x[3,0]=np.nan;y[4,2]=np.inf
    if mode=='branches':bounds=[(0,3),(3,18)]
    if mode=='plan':
        from CausalBridge.adaptive_binning import fit_policy, BinningConfig
        ids=[f'c{i}' for i in range(120)]
        policy=fit_policy(np.linspace(0,1,120),ids,embedding=rng.normal(size=(120,3)),
                          config=BinningConfig(max_bins=18,min_cells_per_bin=3))
        plan=policy.initial_plan
        n=plan.n_bins
        x=rng.normal(size=(n,4));y=rng.normal(size=(n,count))
    genes=[f'TF{i}' for i in range(4)];peaks=[f'P{i}' for i in range(count)]
    mapping={p:[genes[i%4]] for i,p in enumerate(peaks)}
    mapping['P0']=['tf0','TF0','unknown'];mapping['missing']=['TF0']
    args=[mapping,SimpleNamespace(var_names=genes),SimpleNamespace(var_names=peaks),x,y]
    options=dict(max_lag=4,pvalue_threshold=.05,branch_boundaries=bounds,
                 min_abs_contribution=.01,local_lag_window_rows=6,bin_plan=plan)
    expected=kinetics._fit_tf_peak_regression(*args,**options)
    serial=screen.local_lag_candidates(*args,workers=1,**options)
    parallel=screen.local_lag_candidates(*args,workers=2,**options)
    assert_exact(expected,serial);assert_exact(expected,parallel)
    text=capsys.readouterr().err
    assert text.count('Step 6 局部 lag: 260/260')==2


def test_empty_invalid_local_options_still_raise():
    with pytest.raises(ValueError,match='at least three'):
        screen.local_lag_candidates({},SimpleNamespace(var_names=[]),SimpleNamespace(var_names=[]),
            np.empty((5,0)),np.empty((5,0)),local_lag_window_rows=2,workers=2)


def test_parallel_helper_edits_invalidate_reviewed_alias(monkeypatch,tmp_path):
    original=screen.__file__
    changed=tmp_path/'changed.py';changed.write_bytes(Path(original).read_bytes()+b'\n# changed worker implementation\n')
    before=run._numerical_source_hash(granger.granger_test)
    monkeypatch.setattr(screen,'__file__',str(changed))
    assert run._numerical_source_hash(granger.granger_test)!=before


def test_execution_counts_do_not_change_checkpoint_contracts(tmp_path,monkeypatch):
    from test_pipeline_bin_plan import inputs, install_stubs
    rna,atac,cfg=inputs(tmp_path)
    seen=install_stubs(monkeypatch)
    run._run_modeling_for_subset(rna,atac,['KO'],{'KO':'TF'},cfg)
    stages=['step4_granger','step5_atac_to_rna','step6_rna_to_atac']
    before={stage:run._load_checkpoint(cfg,stage)['adaptive_checkpoint'] for stage in stages}
    calls=[stage for stage,_,_ in seen if stage in (4,5,6)]
    cfg['execution']={'granger_workers':4,'local_lag_workers':4}
    run._run_modeling_for_subset(rna,atac,['KO'],{'KO':'TF'},cfg)
    after={stage:run._load_checkpoint(cfg,stage)['adaptive_checkpoint'] for stage in stages}
    assert before==after
    assert [stage for stage,_,_ in seen if stage in (4,5,6)]==calls
