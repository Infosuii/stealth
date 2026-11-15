import numpy as np
import pytest
from stealth import online_blend as old
from stealth import online_missing as new


def test_complete_reference_behavior_is_identical():
    rng=np.random.default_rng(10);era=np.repeat(np.arange(20),8)
    y=rng.normal(size=len(era));b=rng.normal(size=len(era));c=rng.normal(size=len(era))
    np.testing.assert_array_equal(new.utility(y,era,b,c),old.utility(y,era,b,c))


def test_partial_core_uses_valid_subset_without_inventing_a_reference():
    from stealth.temporal_models import rank_by_era
    y=np.arange(8,dtype=float);era=np.zeros(8,dtype=int);core=y.copy();core[-1]=np.nan
    result=new.utility(y,era,-y,core)
    r=rank_by_era(core,era)[:-1];tc=y[:-1]-y[:-1].mean()
    expected=y-y.mean();expected[:-1]+=8/7*(tc-r*(r@tc)/(r@r))
    np.testing.assert_allclose(result,expected,atol=1e-14)
    assert result[-1]==pytest.approx(y[-1]-y.mean())
    assert np.isfinite(result).all()


def test_partial_reference_does_not_weaken_maturity_guard():
    rng=np.random.default_rng(11);era=np.repeat(np.arange(1000,1500),8)
    P=rng.normal(size=(len(era),5));y=rng.normal(size=len(era));core=rng.normal(size=len(era));core[123]=np.nan
    cfg=dict(prior=[.5,.5,0,0,0],gap=40,history=512,half_life=256,ridge=1.,min_eras=200,anchor=.25)
    blocks=[dict(test_start=e,test_end=e+99) for e in range(1000,1500,100)]
    a,ta=new.predict(P,y,era,core,blocks,cfg,'score')
    y[era>=1360]=10000
    b,tb=new.predict(P,y,era,core,blocks,cfg,'score')
    np.testing.assert_array_equal(a,b);assert ta==tb
