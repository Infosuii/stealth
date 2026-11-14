"""Nested boundaries, calibrated covariance objective, persistent search and reader exclusion."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from stealth._runtime import basis_design
from stealth.campaign_models import inner_windows,ridge_path,fit_trees
from stealth.research_models import era_target


def test_inner_windows_are_disjoint_chronological_and_purged():
    era=np.repeat(np.arange(2800,3960),3)
    cfg={'windows':3,'width':200,'gap':40,'min_train':400}
    seen=[]
    for train,valid in inner_windows(era,cfg):
        assert era[valid].min()-era[train].max()>40
        assert not (train&valid).any()
        assert len(np.unique(era[valid]))==200
        seen.extend(np.unique(era[valid]))
    assert len(seen)==len(set(seen))==600
    with pytest.raises(ValueError,match='insufficient'):
        list(inner_windows(era[:900],cfg))


def test_calibrated_path_matches_centered_normal_equations():
    rng=np.random.default_rng(1)
    X=rng.integers(0,10,size=(180,4)).astype(float)
    X[::9,0]=np.nan
    era=np.repeat(np.arange(20),9)
    y=rng.normal(size=len(X))
    coefs,penalties=ridge_path(X,y,era,[.01,1,100])
    d=basis_design(X,'linear')
    for e in np.unique(era):d[era==e]-=d[era==e].mean(axis=0)
    target,weights,_,_=era_target(y,era)
    gram=d.T@(weights[:,None]*d)
    for j,alpha in enumerate(penalties):
        expected=np.linalg.solve(gram+alpha*np.eye(d.shape[1]),d.T@(weights*target))
        np.testing.assert_allclose(coefs[:,j],expected,atol=1e-10,rtol=1e-8)
    changed,_=ridge_path(X,y+era*100,era,[.01,1,100])
    np.testing.assert_allclose(changed,coefs,atol=1e-11)
    assert np.linalg.norm(coefs[:,-1])<np.linalg.norm(coefs[:,0])


def test_tree_search_persists_and_reuses_completed_trials(tmp_path):
    rng=np.random.default_rng(2)
    X=rng.integers(0,10,size=(1000,4)).astype(float)
    era=np.repeat(np.arange(100),10)
    y=X[:,0]*.1+rng.normal(size=len(X))
    spec={'trials':2,'cap':8,'patience':3,'top_k':2}
    inner={'windows':2,'width':10,'gap':2,'min_train':20,'stability_penalty':.25}
    storage=tmp_path/'search.sqlite'
    models,info=fit_trees(spec,X,y,era,inner,seed=17,threads=1,storage=storage)
    again,reused=fit_trees(spec,X,y,era,inner,seed=17,threads=1,storage=storage)
    assert info==reused and info['trials_completed']==2 and len(models)==2
    for a,b in zip(models,again):np.testing.assert_allclose(a.predict(X,num_threads=1),b.predict(X,num_threads=1))


def test_discovery_reader_cannot_load_confirmation_labels(tmp_path,monkeypatch):
    import yaml
    path=Path(__file__).resolve().parents[1]/'scripts/campaign.py'
    spec=importlib.util.spec_from_file_location('campaign_test',path)
    runner=importlib.util.module_from_spec(spec)
    # Sibling script imports are an explicit part of the CLI environment.
    monkeypatch.syspath_prepend(str(path.parent))
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner,'REPO',tmp_path)
    monkeypatch.setattr(runner.r,'sha',lambda p:'fixed')
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/campaign.yaml').write_text(yaml.safe_dump({'name':'test','train_min_era':2800,'confirmation':{'train_end':5300},'excluded_from_research':[5925,6524]}))
    calls=[]
    def reader(path,**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({'exped':['exped_4000']})
    monkeypatch.setattr(runner.pd,'read_parquet',reader)
    runner.load()
    assert calls[-1]=={'filters':[('exped','<','exped_5300')]}
    runner.load('confirmation')
    assert calls[-1]=={'filters':[('exped','<','exped_5925')]}
    monkeypatch.setattr(runner.pd,'read_parquet',lambda *a,**k:pd.DataFrame({'exped':['exped_5340']}))
    with pytest.raises(RuntimeError,match='excluded eras'):
        runner.load()
