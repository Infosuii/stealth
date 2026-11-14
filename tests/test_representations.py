"""Cross-sectional transform parity and actual isolated-runtime portability of latent trees."""
import numpy as np
import pytest
from stealth import _runtime as rt,bundle,pkl_check
from stealth.models import lgbm
from stealth.research_models import to_bytes
from stealth.representation_models import projection,covariance,design_path
from pathlib import Path


def latent_model(X,y,era,columns,context,add_raw):
    model=projection(X,era,columns,n=4)
    model.update(context=context,add_raw=add_raw)
    design=rt.latent_design(model,X,era)
    booster=lgbm.fit_fixed({'max_bin':63,'num_leaves':4,'min_data_in_leaf':5},design,y,None,n_iter=4,seed=17,threads=1)
    model['tree_blob']=np.frombuffer(lgbm.to_bytes(booster),dtype=np.uint8)
    return model,booster,design


@pytest.mark.parametrize('context',[False,True])
@pytest.mark.parametrize('add_raw',[False,True])
def test_latent_training_transform_equals_serialized_cross_section(context,add_raw):
    rng=np.random.default_rng(2)
    X=rng.integers(0,10,size=(240,5)).astype(float);X[::7,0]=np.nan
    y=rng.normal(size=len(X));era=np.repeat(np.arange(20),12)
    model,booster,design=latent_model(X,y,era,list('abcde'),context,add_raw)
    decoded=rt.load_basis(to_bytes(model))
    expected=booster.predict(design,num_threads=1)
    for e in np.unique(era):
        keep=era==e
        actual=rt.basis_forward(decoded,X[keep])
        np.testing.assert_allclose(actual,expected[keep],atol=1e-12)
        np.testing.assert_allclose(rt.basis_forward(decoded,X[keep][::-1])[::-1],actual,atol=1e-12)
    model['context']=.5
    with pytest.raises(ValueError,match='scalar booleans'):rt.load_basis(to_bytes(model))


def test_covariance_removes_era_offsets_and_ridge_path_is_correct():
    rng=np.random.default_rng(8);d=rng.normal(size=(120,7));era=np.repeat(np.arange(20),6);y=rng.normal(size=120)
    gram,w=covariance(d,era)
    changed,_=covariance(d+era[:,None],era)
    np.testing.assert_allclose(gram,changed,atol=1e-10)
    coef,alpha=design_path(d,y,era,[.1,10])
    from stealth.research_models import era_target
    target=era_target(y,era)[0]
    for j,a in enumerate(alpha):np.testing.assert_allclose(coef[:,j],np.linalg.solve(gram+a*np.eye(7),d.T@(w*target)),atol=1e-12)


def test_latent_pickle_in_clean_interpreter(live_frame,data_dir,tmp_path):
    clean=Path(__file__).resolve().parents[1]/'artifacts/pklcheck_venv/bin/python'
    if not clean.exists():pytest.skip('clean interpreter unavailable')
    columns=[c for c in live_frame if c.startswith('feature_')]
    X=live_frame[columns].to_numpy(dtype=float,copy=True);X[X<0]=np.nan
    era=np.repeat(np.arange(3),30)
    model,_,_=latent_model(X,np.random.default_rng(7).normal(size=len(X)),era,columns,True,True)
    b={'version':'test','run_id':'latent-portability','feature_cols':columns,'missing_cols':[],
       'members':[{'name':'latent','family':'basis','input':'nan','feature_cols':columns,'blobs':[to_bytes(model)]}],
       'weights':{'latent':1},'p_b':0.,'p_c':0.,'core_proxy_cols':[]}
    path=tmp_path/'latent.pkl';bundle.write_pickle(b,path)
    result=pkl_check.run_check(path,python=clean,data_dir=data_dir,eras=0)
    assert result['passed'],pkl_check.format_report(result)
