"""Complete-case reference repair; complete-reference behavior is preserved exactly."""
import numpy as np
from stealth.temporal_models import rank_by_era
from stealth.online_blend import weights


def utility(y,era,benchmark,core):
    y,era=np.asarray(y,float),np.asarray(era)
    refs=[rank_by_era(benchmark,era),rank_by_era(core,era)]
    if not np.isfinite(refs[0]).all():raise ValueError('complete OOS benchmark required')
    out=np.empty(len(y))
    for e in np.unique(era):
        m=era==e;yy=y[m];t=yy-yy.mean()
        for reference in refs:
            r=reference[m];ok=np.isfinite(r);n=int(ok.sum())
            if n<5:continue
            rr=r[ok];tc=yy[ok]-yy[ok].mean();mm=rr@rr;cov=rr@tc
            if mm>0 and cov>=0:t[ok]+=len(yy)/n*(tc-rr*cov/mm)
        out[m]=t
    return out


def predict(P,y,era,core,blocks,cfg,objective):
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
