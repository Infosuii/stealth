"""Daily causal covariance updates, optionally conditioned on observable current state."""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from stealth.online_missing import utility


def correlation(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    x=x-x.mean();y=y-y.mean();den=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/den) if den>1e-12 else 0.


def feature_state(X):
    """One current cross-section only; no fitted statistics from other eras."""
    X=np.asarray(X,float);n,d=X.shape;missing=~np.isfinite(X)|(X<0)
    z=np.where(missing,4.5,X);z-=z.mean(axis=0)
    sd=np.sqrt(np.mean(z*z,axis=0));valid=sd>1e-8
    z=z[:,valid]/sd[valid];dim=z.shape[1]
    if dim<2:return np.array([np.log(n),missing.mean(),0,0,0,0,0,0.],float)
    spectrum=np.maximum(np.linalg.eigvalsh((z@z.T)/(n*dim)),0)
    spectrum/=max(spectrum.sum(),1e-12)
    positive=spectrum[spectrum>1e-12]
    entropy=-float(positive@np.log(positive))/max(np.log(min(n,dim)),1.)
    off_mean=(float(np.sum(np.sum(z,axis=1)**2)/n)-dim)/(dim*(dim-1))
    off_second=max((dim*dim*float(spectrum@spectrum)-dim)/(dim*(dim-1)),0.)
    return np.array([np.log(n),missing.mean(),spectrum[-1],spectrum[-3:].sum(),entropy,
                     off_mean,np.sqrt(max(off_second-off_mean**2,0)),np.var(z.mean(axis=1))])


def prediction_state(P,seeds):
    """Agreement is observable without any realized target."""
    P=np.asarray(P,float)
    return np.array([correlation(P[:,0],P[:,j]) for j in range(1,P.shape[1])]
                    +[correlation(a,b) for a,b in seeds]+[float(np.mean(np.std(P,axis=1)))])


def solve(C,mu,prior,ridge):
    scale=max(float(prior@mu)/max(float(prior@C@prior),1e-12),0.)
    L=np.linalg.cholesky(C+ridge*np.eye(len(prior)))
    coef,_=nnls(L.T,np.linalg.solve(L,mu+ridge*scale*prior))
    return coef/coef.sum() if coef.sum()>1e-12 else prior.copy()


def predict(P,y,era,core,states,cfg,conditional):
    """Targets enter cached moments only when their era has matured; state scaling is past-only."""
    P,y,era,core=np.asarray(P,float),np.asarray(y,float),np.asarray(era),np.asarray(core,float)
    states=np.asarray(states,float);unique=np.unique(era);groups=[np.flatnonzero(era==e) for e in unique]
    if len(states)!=len(unique) or not np.isfinite(states).all() or not np.isfinite(P).all():raise ValueError('invalid state/prediction layout')
    prior=np.asarray(cfg['prior'],float);d=P.shape[1]
    if len(prior)!=d or np.min(prior)<0 or not np.isclose(prior.sum(),1.):raise ValueError('invalid prior')
    C=np.empty((len(unique),d,d));mu=np.empty((len(unique),d));next_matured=0
    output=np.full(len(y),np.nan);trace=[]
    for j,(e,ids) in enumerate(zip(unique,groups)):
        cut=e-cfg['gap']
        while next_matured<len(unique) and unique[next_matured]<cut:
            k=next_matured;m=groups[k];z=P[m]-P[m].mean(axis=0)
            t=utility(y[m],era[m],P[m,0],core[m])
            C[k]=z.T@z/len(m);mu[k]=z.T@t/len(m);next_matured+=1
        keep=np.flatnonzero((unique[:next_matured]>=cut-cfg['history']))
        w=prior.copy();ess=0.;distance=None
        if len(keep)>=cfg['min_eras']:
            age=np.exp2((unique[keep]-unique[keep].max())/cfg['half_life']);age/=age.sum()
            mass=age
            if conditional:
                center=age@states[keep]
                sd=np.sqrt(age@((states[keep]-center)**2));sd=np.maximum(sd,1e-6)
                distances=np.mean(((states[keep]-states[j])/sd)**2,axis=1)
                log_kernel=-distances/(2*cfg['bandwidth']**2);log_kernel-=log_kernel.max()
                kernel=age*np.exp(log_kernel);kernel/=kernel.sum()
                mass=(1-cfg['context_mass'])*age+cfg['context_mass']*kernel
                distance=float(distances.min())
            ess=float(1/(mass@mass))
            w=solve(np.einsum('i,ijk->jk',mass,C[keep]),mass@mu[keep],prior,cfg['ridge'])
        w=(1-cfg['anchor'])*w;w[0]+=cfg['anchor']
        output[ids]=P[ids]@w
        trace.append(dict(era=int(e),train_max=int(unique[next_matured-1]) if next_matured else None,
                          fitting_eras=len(keep),effective_eras=ess,nearest_state_distance=distance,weights=w.tolist()))
    if not np.isfinite(output).all():raise RuntimeError('uncovered daily predictions')
    return output,trace
