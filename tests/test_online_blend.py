import numpy as np
import pytest
from stealth.online_blend import predict,utility,weights


def example():
    rng=np.random.default_rng(4);era=np.repeat(np.arange(1000,1500),8)
    P=rng.normal(size=(len(era),5));y=.5+.1*P[:,2]+rng.normal(0,.2,len(era));core=rng.normal(size=len(era))
    blocks=[dict(test_start=e,test_end=e+99) for e in range(1000,1500,100)]
    cfg=dict(prior=[.5,.5,0,0,0],gap=40,history=512,half_life=256,ridge=1.,min_eras=200,anchor=.25)
    return P,y,era,core,blocks,cfg


@pytest.mark.parametrize('objective',['fit','score'])
def test_future_labels_cannot_change_current_or_previous_weight_decisions(objective):
    P,y,era,core,blocks,cfg=example();a,trace=predict(P,y,era,core,blocks,cfg,objective)
    changed=y.copy();changed[era>=1360]=np.random.default_rng(8).normal(100,100,(era>=1360).sum())
    b,other=predict(P,changed,era,core,blocks,cfg,objective)
    np.testing.assert_array_equal(a,b)  # final block starts 1400, cutoff 1360; every changed label is unavailable
    assert trace==other
    for row in trace:
        assert row['train_max'] is None or row['train_max']<row['start']-40
        assert min(row['weights'])>=0 and sum(row['weights'])==pytest.approx(1)
        assert row['weights'][0]>=.25
    warm=era<1300
    np.testing.assert_allclose(a[warm],.625*P[warm,0]+.375*P[warm,1],atol=1e-14)


def test_covariance_solver_matches_single_signal_limit_and_learns_useful_component():
    P,y,era,core,blocks,cfg=example()
    w=weights(P,y,era,np.array(cfg['prior']),half_life=256,ridge=1.)
    assert np.argmax(w)==2 and min(w)>=0 and sum(w)==pytest.approx(1)
    singleton=weights(P[:,:1],y,era,np.ones(1),half_life=256,ridge=1.)
    np.testing.assert_array_equal(singleton,[1.])
    # The negative-reference zero rule leaves only FIT when both reference covariances are negative.
    t=np.tile(np.arange(8,dtype=float),3);e=np.repeat(np.arange(3),8)
    np.testing.assert_allclose(utility(t,e,-t,-t),t-3.5)
