import numpy as np
from stealth.stability_models import moments,coefficient_path,settings,fit_stability


def test_signal_moments_match_direct_era_centering():
    rng=np.random.default_rng(4);d=rng.normal(size=(120,5));y=rng.normal(size=120);era=np.repeat(np.arange(20),6)
    unique,signal,var=moments(d,y,era)
    for j,e in enumerate(unique):
        xx=d[era==e];yy=y[era==e]
        np.testing.assert_allclose(signal[j],((xx-xx.mean(axis=0))*(yy-yy.mean())[:,None]).mean(axis=0),atol=1e-15)
        np.testing.assert_allclose(var[j],xx.var(axis=0),atol=1e-15)
    _,changed,_=moments(d,y+era*100,era)
    np.testing.assert_allclose(changed,signal,atol=1e-12)


def test_covariance_paths_are_finite_and_invariant_to_era_target_offsets():
    rng=np.random.default_rng(5);d=rng.normal(size=(300,5));era=np.repeat(np.arange(50),6);y=d[:,0]+rng.normal(size=300)
    for adaptive in [False,True]:
        spec={'adaptive':adaptive,'histories':[None,30],'half_lives':[16,32],'trends':[0.,1.],'lambdas':[.1,100],
              'shrinks':[0.,1.],'block':5,'forecast_horizon':10}
        a=coefficient_path(spec,d,y,era);b=coefficient_path(spec,d,y+era*100,era)
        assert a.shape==(5,len(settings(spec))) and np.isfinite(a).all()
        np.testing.assert_allclose(a,b,rtol=1e-8,atol=1e-8)


def test_stability_fit_exports_a_finite_portable_linear_model():
    rng=np.random.default_rng(7);X=rng.integers(0,10,size=(1200,3)).astype(float);era=np.repeat(np.arange(120),10)
    y=.1*X[:,0]+rng.normal(size=1200)
    spec={'adaptive':False,'histories':[None,40],'shrinks':[0.,1.],'lambdas':[1,100],'block':5}
    model,info=fit_stability(spec,X,y,era,list('abc'),{'windows':2,'width':10,'gap':2,'min_train':20,'stability_penalty':.25})
    from stealth.research_models import to_bytes
    from stealth._runtime import load_basis,basis_forward
    actual=basis_forward(load_basis(to_bytes(model)),X)
    assert np.isfinite(actual).all() and len(info['path'])==len(settings(spec))
    np.testing.assert_allclose(np.linalg.norm(model['coef']),1.)
