"""Nonnegative online ridge weights fit only to matured out-of-sample predictions."""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from stealth.temporal_models import rank_by_era


def utility(y,era,benchmark,core):
    y,era=np.asarray(y,float),np.asarray(era)
    refs=[rank_by_era(benchmark,era),rank_by_era(core,era)]
    if not all(np.isfinite(r).all() for r in refs):raise ValueError('complete calibration references required')
    out=np.empty(len(y))
    for e in np.unique(era):
        m=era==e;tc=y[m]-y[m].mean();t=tc.copy()
        for reference in refs:
            r=reference[m];mm=r@r;cov=r@tc
            if len(r)>=5 and mm>0 and cov>=0:t+=tc-r*cov/mm
        out[m]=t
    return out


def weights(P,y,era,prior,*,half_life,ridge):
    P,y,era,prior=np.asarray(P,float),np.asarray(y,float),np.asarray(era),np.asarray(prior,float)
    unique=np.unique(era);age=np.exp2((unique-unique.max())/half_life);age/=age.sum()
    C=np.zeros((P.shape[1],P.shape[1]));mu=np.zeros(P.shape[1])
    for e,w in zip(unique,age):
        mask=era==e;z=P[mask]-P[mask].mean(axis=0)
        C+=w*(z.T@z)/mask.sum();mu+=w*(z.T@y[mask])/mask.sum()
    scale=max(float(prior@mu)/max(float(prior@C@prior),1e-12),0.)
    M=C+ridge*np.eye(len(prior));rhs=mu+ridge*scale*prior
    L=np.linalg.cholesky(M)
    coef,_=nnls(L.T,np.linalg.solve(L,rhs))
    return coef/coef.sum() if coef.sum()>1e-12 else prior.copy()


def predict(P,y,era,core,blocks,cfg,objective):
    """Every block's target access is restricted before estimating even normalization or priors."""
    P,y,era,core=np.asarray(P,float),np.asarray(y,float),np.asarray(era),np.asarray(core,float)
    prior=np.asarray(cfg['prior'],float)
    if P.shape[1]!=len(prior) or not np.isclose(prior.sum(),1) or (prior<0).any():raise ValueError('invalid online prior')
    if not np.isfinite(P).all():raise ValueError('finite OOS component predictions required')
    output=np.full(len(y),np.nan);trace=[]
    for block in blocks:
        start,end=block['test_start'],block['test_end'];cut=start-cfg['gap']
        valid=(era>=start)&(era<=end)
        if not valid.any():continue
        train=(era<cut)&(era>=cut-cfg['history'])
        count=len(np.unique(era[train]));w=prior.copy()
        if count>=cfg['min_eras']:
            target=y[train]
            if objective=='score':target=utility(target,era[train],P[train,0],core[train])
            elif objective!='fit':raise ValueError('unknown online objective')
            w=weights(P[train],target,era[train],prior,half_life=cfg['half_life'],ridge=cfg['ridge'])
        w=(1-cfg['anchor'])*w;w[0]+=cfg['anchor']
        output[valid]=P[valid]@w
        trace.append(dict(start=start,end=end,train_max=int(era[train].max()) if train.any() else None,
                          fitting_eras=count,weights=w.tolist(),warmup=count<cfg['min_eras']))
    if not np.isfinite(output).all():raise ValueError('uncovered online prediction rows')
    return output,trace
