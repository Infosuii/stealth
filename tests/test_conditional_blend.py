import numpy as np
import pytest
from stealth.conditional_blend import feature_state,prediction_state,predict


def inputs():
    rng=np.random.default_rng(45);era=np.repeat(np.arange(1000,1500),7)
    P=rng.normal(size=(len(era),5));y=.1*P[:,2]+rng.normal(size=len(era));core=rng.normal(size=len(era));core[234]=np.nan
    S=rng.normal(size=(500,8))
    cfg=dict(prior=[.5,.5,0,0,0],history=512,gap=40,half_life=256,ridge=1.,min_eras=200,anchor=.25,bandwidth=1.,context_mass=.5)
    return P,y,era,core,S,cfg


@pytest.mark.parametrize('conditional',[False,True])
def test_daily_future_label_and_state_invariance(conditional):
    P,y,era,core,S,cfg=inputs();a,trace=predict(P,y,era,core,S,cfg,conditional)
    other=y.copy();other[era>=1360]=10000
    b,_=predict(P,other,era,core,S,cfg,conditional)
    np.testing.assert_array_equal(a[era<=1400],b[era<=1400])
    future=S.copy();future[np.arange(1000,1500)>=1300]=1e6
    d,_=predict(P,y,era,core,future,cfg,conditional)
    np.testing.assert_array_equal(a[era<1300],d[era<1300])
    for row in trace:
        assert row['train_max'] is None or row['train_max']<row['era']-40
        assert sum(row['weights'])==pytest.approx(1) and min(row['weights'])>=0 and row['weights'][0]>=.25


def test_constant_state_reduces_to_unconditional_and_warmup_is_prior():
    P,y,era,core,S,cfg=inputs();S[:]=1
    a,_=predict(P,y,era,core,S,cfg,False);b,_=predict(P,y,era,core,S,cfg,True)
    np.testing.assert_allclose(a,b,atol=1e-12)
    warm=era<1240
    np.testing.assert_allclose(a[warm],.625*P[warm,0]+.375*P[warm,1],atol=1e-14)


def test_current_feature_state_is_label_free_finite_and_order_invariant():
    rng=np.random.default_rng(3);X=rng.integers(-1,10,(40,12)).astype(float)
    a=feature_state(X);b=feature_state(X[::-1,::-1])
    np.testing.assert_allclose(a,b,atol=1e-12)
    assert np.isfinite(feature_state(np.full((10,12),-1.))).all()
    P=np.zeros((40,5));s=prediction_state(P,[(np.ones(40),np.ones(40))]*3)
    np.testing.assert_array_equal(s,np.zeros(8))
