"""Availability, exact score semantics, surrogate isolation and actual serialized refits."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stealth import metrics
from stealth.models import lgbm
from stealth.temporal_models import blocks, composite, core_mean, fit, fixed_fit, rank_by_era, score_target, stopping_scorer


def test_sequential_blocks_cover_once_with_available_targets_only():
    windows = [dict(name='early',test_start=4000,test_end=4499),dict(name='late',test_start=5000,test_end=5299)]
    bs = blocks(windows,100,40)
    assert len(bs)==8
    eras = [e for b in bs for e in range(b['test_start'],b['test_end']+1)]
    assert len(eras)==len(set(eras))==800
    assert all(b['test_start']-b['train_end']==40 for b in bs)
    assert bs[1]['train_end']==4060  # first block's final 40 targets are still unavailable
    with pytest.raises(ValueError): blocks(windows,100,20)


def test_composite_matches_sdk_wrappers_with_ties_zero_rules_missing_core_and_unsorted_rows():
    rng = np.random.default_rng(70)
    era = np.repeat(np.arange(4),13)
    order = rng.permutation(len(era))
    y = rng.choice([0,.25,.5,.75,1],len(era))
    prediction,bench = rng.integers(0,5,len(era)),rng.normal(size=len(era))
    X = rng.integers(-1,10,(len(era),5)).astype(float)
    X[::5] = -1
    core = core_mean(X)
    bound = 2/np.pi
    result = composite(prediction[order],y[order],era[order],bench[order],core[order],bound=bound)
    for e in np.unique(era):
        ids = era==e
        f = metrics.fit20(pd.Series(prediction[ids]),pd.Series(y[ids]))
        u = metrics.unq20(pd.Series(prediction[ids]),pd.Series(bench[ids]),pd.Series(y[ids]))
        i = metrics.inov(pd.Series(prediction[ids]),pd.DataFrame(X[ids]),pd.Series(y[ids]))
        np.testing.assert_allclose(result.loc[e,['fit','unq','inov']],[f,u,i],atol=1e-12)
        assert result.loc[e,'score']==pytest.approx(bound*np.arctan((f+u+i)/bound))
    fast = stopping_scorer(y[order],era[order],bench[order],core[order],bound)
    assert fast(prediction[order])==pytest.approx(result.score.mean(),abs=1e-14)
    assert abs(result.score.mean()-bound*np.arctan(result[['fit','unq','inov']].mean().sum()/bound))>1e-7


def test_score_target_does_not_mix_eras_or_use_future_labels():
    y = np.array([0.,.25,.5,.75,1]*3)
    era = np.repeat(np.arange(3),5)
    core = np.tile(np.arange(5),3)
    original = score_target(y,era,core)
    # Rescaling is fit-global, but the normalized direction within earlier eras is unchanged.
    changed = y.copy();changed[era==2] = [1,0,1,0,1]
    modified = score_target(changed,era,core)
    np.testing.assert_allclose(rank_by_era(original,era)[era<2],rank_by_era(modified,era)[era<2])
    shifted = score_target(y+100*era,era,core)
    np.testing.assert_allclose(shifted,original,atol=1e-12)
    assert np.isfinite(original).all() and original.std()==pytest.approx((y-.5).std())
    assert np.nanmax(np.abs(core_mean(np.array([[-1,4,6],[np.nan,-1,7]]))-[5,7]))==0


@pytest.mark.parametrize('score_aware',[False,True])
def test_real_refit_and_serialized_replay_use_purged_own_inner_count(score_aware):
    rng = np.random.default_rng(3)
    era = np.repeat(np.arange(400),5)
    X = rng.integers(0,10,(len(era),6)).astype(float)
    y = .5+.02*(X[:,0]-4.5)+rng.normal(0,.1,len(era))
    spec = dict(history=350,half_life=100,score_aware=score_aware)
    model,info = fit(spec,dict(num_leaves=3,min_data_in_leaf=30,cap=8,early_stopping_rounds=3),X,y,era,
                     inner_width=40,gap=40,benchmark_trees=3,bound=2/np.pi,seed=17,threads=1)
    assert info['train_min']==50 and info['train_max']==399
    assert info['inner_valid_min']-info['inner_fit_max']>40
    assert 1<=info['n_iter']<=8
    np.testing.assert_allclose(lgbm.predict(model,X[:35]),lgbm.predict(lgbm.from_bytes(lgbm.to_bytes(model)),X[:35]),atol=1e-12)
    controlled = fixed_fit(spec,dict(num_leaves=3,min_data_in_leaf=30,cap=8,early_stopping_rounds=3),X,y,era,info['n_iter'],seed=17,threads=1)
    np.testing.assert_allclose(lgbm.predict(model,X[:35]),lgbm.predict(controlled,X[:35]),atol=1e-12)
    assert info['benchmark_fit_max']==info['inner_fit_max'] if score_aware else info['benchmark_fit_max'] is None


def runner(monkeypatch):
    path = Path(__file__).resolve().parents[1]/'scripts/temporal.py'
    monkeypatch.syspath_prepend(str(path.parent))
    module_spec = importlib.util.spec_from_file_location('temporal_test',path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_reader_fails_closed_before_confirmation_and_source_changes(tmp_path,monkeypatch):
    import yaml
    mod = runner(monkeypatch)
    monkeypatch.setattr(mod,'REPO',tmp_path)
    monkeypatch.setattr(mod.r,'sha',lambda path:'fixed')
    (tmp_path/'configs').mkdir()
    cfg = dict(name='toy',train_min_era=2800,reader_ceiling=5300,excluded_from_research=[5925,6524])
    (tmp_path/'configs/temporal.yaml').write_text(yaml.safe_dump(cfg))
    calls = []
    def read(path,**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(dict(exped=['exped_4000']))
    monkeypatch.setattr(mod.pd,'read_parquet',read)
    mod.load()
    assert calls==[{'filters':[('exped','<','exped_5300')]}]
    (mod.RUN/'selection.json').write_text(json.dumps(dict(plan_sha256=mod.PLAN,selected=None)))
    with pytest.raises(RuntimeError,match='no frozen'): mod.load(True)
    assert len(calls)==1
    (mod.RUN/'selection.json').write_text(json.dumps(dict(plan_sha256=mod.PLAN,selected={'name':'rolling'})))
    mod.load(True)
    assert calls[-1]=={'filters':[('exped','<','exped_5925')]}
    cfg['reader_ceiling']=5400
    (tmp_path/'configs/temporal.yaml').write_text(yaml.safe_dump(cfg))
    with pytest.raises(RuntimeError,match='changed temporal plan'): mod.load()


def test_neutralization_policy_is_fixed_when_scoring_references_change(monkeypatch):
    mod = runner(monkeypatch)
    era = np.repeat([1,2],8)
    raw = np.tile(np.arange(8),2).astype(float)
    core = np.tile([2,4,0,7,5,1,3,6],2)
    z = mod.neutralized(raw,era,core,.25)
    # Policy is established once. Subsequent metrics use these identical predictions for every reference.
    recorded = []
    def spy(pred,*args,**kwargs):
        recorded.append(np.array(pred))
        return pd.DataFrame(dict(fit=[.1],unq=[0.],inov=[.01],score=[.11]))
    monkeypatch.setattr(mod,'composite',spy)
    monkeypatch.setattr(mod,'candidate_prediction',lambda *a:z)
    monkeypatch.setattr(mod,'CFG',dict(score_bound=2/np.pi))
    base = pd.DataFrame(dict(era=era,prediction=raw))
    frame = pd.DataFrame(dict(target_everest=raw))
    mod.evaluate('rolling',.25,(base,base,frame,[core,core[::-1],core+1]))
    for p in recorded[::2]:np.testing.assert_array_equal(p,recorded[0])
