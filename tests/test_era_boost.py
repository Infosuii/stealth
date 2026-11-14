"""Era transforms, inner-only stopping, and actual booster invariance."""
import numpy as np
import pandas as pd
import pytest

from stealth import era_boost, metrics


def test_target_transform_equalises_era_mass_and_removes_offsets():
    y=np.array([1.,3.,5.,8.,11.])
    era=np.array([1,1,2,2,2])
    centered,w=era_boost.centred_target(y,era)
    for e in np.unique(era):
        assert centered[era==e].mean()==pytest.approx(0)
        assert w[era==e].sum()==pytest.approx(len(y)/2)
    shifted,_=era_boost.centred_target(y+era*100,era)
    np.testing.assert_allclose(shifted,centered)


@pytest.mark.parametrize('stopping',['mse','fit'])
def test_era_centered_booster_is_invariant_to_fit_era_offsets(stopping):
    rng=np.random.default_rng(3)
    X=rng.normal(size=(3000,4))
    era=np.repeat(np.arange(100),30)
    y=X[:,0]+rng.normal(0,.2,len(X))
    params={'cap':12,'num_leaves':4,'min_data_in_leaf':20,'max_bin':16}
    spec={'centred':True,'stopping':stopping}
    a,na=era_boost.fit(spec,params,X,y,era,seed=17,threads=1)
    b,nb=era_boost.fit(spec,params,X,y+era*100,era,seed=17,threads=1)
    assert na==nb
    np.testing.assert_allclose(a.predict(X,num_threads=1),b.predict(X,num_threads=1),atol=1e-7)


def test_fit_stopping_uses_raw_targets_and_a_purged_inner_partition(monkeypatch):
    import lightgbm as lgb
    rng=np.random.default_rng(8)
    X=np.arange(300).reshape(-1,1)
    era=np.repeat(np.arange(100),3)
    y=rng.normal(size=len(X))+era*100
    seen={}
    def train(params,dataset,**kwargs):
        seen['metric']=params['metric']
        # Training eras end strictly before cut - 20, not just before validation.
        np.testing.assert_array_equal(dataset.data[:,0],X[era<60,0])
        for e in range(60):
            assert np.mean(dataset.label[era[era<60]==e])==pytest.approx(0,abs=1e-10)
        val=kwargs['valid_sets'][0]
        np.testing.assert_array_equal(val.data[:,0],X[era>=80,0])
        prediction=rng.normal(size=sum(era>=80))
        name,value,maximize=kwargs['feval'](prediction,val)
        expected=metrics.per_era_fit(pd.Series(prediction),pd.Series(y[era>=80]),pd.Series(era[era>=80])).mean()
        assert name=='fit20' and maximize
        assert value==pytest.approx(expected,abs=1e-12)
        return type('Booster',(),{'best_iteration':7})()
    def refit(params,X,y,w,**kwargs):
        assert kwargs['n_iter']==7
        assert len(y)==300
        for e in range(100):
            assert y[era==e].mean()==pytest.approx(0,abs=1e-10)
        return 'refit'
    monkeypatch.setattr(lgb,'train',train)
    monkeypatch.setattr(era_boost.lgbm,'fit_fixed',refit)
    model,count=era_boost.fit({'centred':True,'stopping':'fit'},{'cap':20},X,y,era,seed=17,threads=1)
    assert seen['metric']=='None' and model=='refit' and count==7


def test_short_windows_fail_closed():
    with pytest.raises(ValueError,match='30 eras'):
        era_boost.fit({'centred':True,'stopping':'fit'},{'cap':10},np.ones((20,2)),np.ones(20),np.arange(20),seed=17,threads=1)
