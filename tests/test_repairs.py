import numpy as np
import pandas as pd
import pytest
from stealth import _runtime as rt,bundle,pkl_check
from stealth.representation_models import group_head,design_path
from stealth.research_models import to_bytes
from pathlib import Path


def test_group_balancing_is_invariant_to_duplicate_raw_columns():
    rng=np.random.default_rng(6);raw=rng.normal(size=(180,5));latent=rng.normal(size=(180,3));era=np.repeat(np.arange(30),6)
    y=latent[:,0]+.1*raw[:,0]+rng.normal(size=180)
    coef,scale,penalty=group_head(raw,latent,y,era,[.01,1],.5)
    doubled=np.concatenate([raw,raw],axis=1)
    other,new_scale,new_penalty=group_head(doubled,latent,y,era,[.01,1],.5)
    np.testing.assert_allclose(np.concatenate([raw*scale,latent],axis=1)@coef,np.concatenate([doubled*new_scale,latent],axis=1)@other,atol=1e-11)
    np.testing.assert_allclose(penalty,new_penalty,atol=1e-12)


def test_zero_raw_strength_matches_latent_only_ridge_even_with_padding():
    rng=np.random.default_rng(8);raw=rng.normal(size=(120,30));latent=rng.normal(size=(120,4));era=np.repeat(np.arange(20),6);y=rng.normal(size=120)
    coef,scale,_=group_head(raw,latent,y,era,[.1,10],0)
    direct,_=design_path(latent,y,era,[.1,10],penalty_mode='trace')
    assert scale==0
    np.testing.assert_allclose(coef[:30],0,atol=1e-13)
    np.testing.assert_allclose(latent@coef[30:],latent@direct,atol=1e-11)


def test_blob_rank_rule_preserves_affine_scale_and_rejects_unknown():
    scores=[np.array([1.,3.,2.,5.,4.]),np.array([5.,4.,3.,2.,1.])]
    expected=rt.average_blob_scores(scores,'gaussian_mean')
    np.testing.assert_allclose(rt.average_blob_scores([scores[0]*100+9,scores[1]],'gaussian_mean'),expected)
    np.testing.assert_array_equal(rt.average_blob_scores(scores),np.mean(np.vstack(scores),axis=0))
    with pytest.raises(ValueError,match='unsupported'):rt.average_blob_scores(scores,'unknown')


def test_rank_blob_rule_portable_pickle(live_frame,data_dir,tmp_path):
    clean=Path(__file__).resolve().parents[1]/'artifacts/pklcheck_venv/bin/python'
    if not clean.exists():pytest.skip('clean interpreter unavailable')
    columns=[c for c in live_frame if c.startswith('feature_')];width=2*len(columns)
    a=np.zeros(width);a[0]=1;b=np.zeros(width);b[1]=-100
    models=[{'kind':'linear','columns':columns,'coef':v} for v in [a,b]]
    member={'name':'ranked','family':'basis','input':'nan','feature_cols':columns,'blobs':[to_bytes(m) for m in models],'blob_rule':'gaussian_mean'}
    bag={'version':'test','run_id':'ranked','feature_cols':columns,'missing_cols':[],'members':[member],'weights':{'ranked':1},'p_b':0.,'p_c':0.,'core_proxy_cols':[]}
    prediction=rt.make_predict(bag)(live_frame)
    X=live_frame[columns].to_numpy(dtype=float,copy=True);X[X<0]=np.nan
    expected=rt.rank01(rt.average_blob_scores([rt.basis_forward(m,X) for m in models],'gaussian_mean'))
    np.testing.assert_allclose(prediction.prediction,expected,atol=1e-12)
    path=tmp_path/'ranked.pkl';bundle.write_pickle(bag,path)
    result=pkl_check.run_check(path,python=clean,data_dir=data_dir,eras=0)
    assert result['passed'],pkl_check.format_report(result)
    member['blob_rule']='unsupported'
    with pytest.raises(ValueError,match='averaging rule'):rt.make_predict(bag)
