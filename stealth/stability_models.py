"""Temporal covariance-risk weighting. Every estimate belongs to the supplied fitting eras."""
import numpy as np
import pandas as pd
from stealth import metrics
from stealth._runtime import basis_design
from stealth.campaign_models import inner_windows


def moments(design,y,era):
    order=np.argsort(era,kind='stable');d=design[order];y=np.asarray(y)[order]
    unique,count=np.unique(np.asarray(era)[order],return_counts=True)
    starts=np.r_[0,np.cumsum(count)[:-1]]
    mean=np.add.reduceat(d,starts,axis=0)/count[:,None]
    ym=np.add.reduceat(y,starts)/count
    signal=np.add.reduceat(d*y[:,None],starts,axis=0)/count[:,None]-mean*ym[:,None]
    variance=np.maximum(np.add.reduceat(d*d,starts,axis=0)/count[:,None]-mean*mean,0)
    return unique,signal,variance


def settings(spec):
    if spec['adaptive']:
        return [(h,t,s,l) for h in spec['half_lives'] for t in spec['trends'] for s in spec['shrinks'] for l in spec['lambdas']]
    return [(h,0.,s,l) for h in spec['histories'] for s in spec['shrinks'] for l in spec['lambdas']]


def coefficient_path(spec,design,y,era):
    unique,signal,variance=moments(design,y,era)
    out=[]
    histories=spec['half_lives'] if spec['adaptive'] else spec['histories']
    for history in histories:
        keep=np.ones(len(unique),dtype=bool) if spec['adaptive'] or history is None else unique>unique[-1]-history
        ee,ss,vv=unique[keep],signal[keep],variance[keep]
        scale=np.maximum(np.sqrt(vv.mean(axis=0)),.1)
        ss=ss/scale
        block=spec['block'];count=len(ss)//block
        if count<3:raise ValueError('insufficient signal blocks')
        blocks=ss[-count*block:].reshape(count,block,-1).mean(axis=1)
        times=ee[-count*block:].reshape(count,block).mean(axis=1)
        if spec['adaptive']:
            w=np.exp2((ee-ee[-1])/history);w/=w.sum()
            mean=w@ss;tt=ee-ee[-1];mt=float(w@tt)
            slope=(w*(tt-mt))@ss/max(float(w@((tt-mt)**2)),1e-12)
            means=[mean+t*(spec['forecast_horizon']-mt)*slope for t in spec['trends']]
            bw=np.exp2((times-times[-1])/history);bw/=bw.sum()
        else:
            means=[ss.mean(axis=0)];bw=np.full(count,1/count)
        centered=blocks-bw@blocks
        covariance=centered.T@(bw[:,None]*centered)
        diagonal=np.diag(np.diag(covariance))
        strength=max(float(np.trace(covariance)/len(scale)),1e-12)
        paths={}
        for shrink in spec['shrinks']:
            risk=(1-shrink)*covariance+shrink*diagonal
            eigen,vectors=np.linalg.eigh((risk+risk.T)*.5)
            for j,mean in enumerate(means):
                coef=vectors@((vectors.T@mean)[:,None]/(np.maximum(eigen,0)[:,None]+strength*np.asarray(spec['lambdas'])[None,:]))
                paths[(j,shrink)]=coef/scale[:,None]
        for j in range(len(means)):
            for shrink in spec['shrinks']:
                out.extend(paths[(j,shrink)].T)
    result=np.asarray(out).T
    if not np.isfinite(result).all():raise RuntimeError('nonfinite temporal coefficients')
    return result


def fit_stability(spec,X,y,era,columns,inner):
    design=basis_design(X,'linear')
    choices=settings(spec);scores=np.empty((len(choices),inner['windows']))
    for fold,(train,valid) in enumerate(inner_windows(era,inner)):
        coef=coefficient_path(spec,design[train],y[train],np.asarray(era)[train])
        prediction=design[valid]@coef
        for j in range(len(choices)):
            scores[j,fold]=metrics.per_era_fit_fast(pd.Series(prediction[:,j]),pd.Series(y[valid]),pd.Series(np.asarray(era)[valid])).mean()
    objective=scores.mean(axis=1)-inner['stability_penalty']*scores.std(axis=1)
    best=int(np.argmax(objective))
    coefficients=coefficient_path(spec,design,y,era)[:,best]
    # Overall scale is irrelevant to ranks, but bounded numeric weights aid portability.
    coefficients=coefficients/max(float(np.linalg.norm(coefficients)),1e-12)
    return {'kind':'linear','columns':columns,'coef':coefficients},{'selected':choices[best],
        'inner_scores':scores[best].tolist(),'path':[{'setting':c,'objective':float(v),'scores':s.tolist()} for c,v,s in zip(choices,objective,scores)]}
