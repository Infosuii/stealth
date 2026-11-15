import numpy as np
import pytest
from stealth.conditional_supported import constrained_mass,predict


def test_concentrated_kernel_is_bounded_without_using_any_target():
    age=np.ones(512)/512;kernel=np.zeros(512);kernel[0]=1
    mass,alpha,floor=constrained_mass(age,kernel,.5,.5,200)
    assert floor==200 and 0<alpha<.5 and 1/(mass@mass)>=200-1e-9
    same,a,_=constrained_mass(age,age,.5,.5,200)
    np.testing.assert_array_equal(same,age);assert a==.5


def test_supported_daily_policy_preserves_future_invariance_and_support():
    rng=np.random.default_rng(8);era=np.repeat(np.arange(1000,1500),6)
    P=rng.normal(size=(len(era),5));y=rng.normal(size=len(era));core=rng.normal(size=len(era));S=rng.normal(size=(500,8));S[300]=10000
    cfg=dict(prior=[.5,.5,0,0,0],history=512,gap=40,half_life=256,ridge=1.,min_eras=200,anchor=.25,bandwidth=1.,context_mass=.5,support_fraction=.5,support_cap=200)
    a,trace=predict(P,y,era,core,S,cfg,True)
    y[era>=1360]=1000;S[450:]=1000
    b,_=predict(P,y,era,core,S,cfg,True)
    np.testing.assert_array_equal(a[era<=1400],b[era<=1400])
    assert all(r['effective_eras']>=r['required_effective_eras']-1e-8 for r in trace)
