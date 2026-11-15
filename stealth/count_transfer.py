"""Matched era-count controls for stopping-count transfer; no outer-period outcomes."""
from __future__ import annotations

import numpy as np

from stealth.models import lgbm
from stealth.robust_temporal import fit as prior_fit,median_count
from stealth.temporal_models import core_mean,score_target,stopping_scorer


def partitions(era, *, history, width, gap, floor, policy):
    era = np.asarray(era)
    outer = (era>=floor) & (era>era.max()-history)
    unique = np.unique(era[outer])
    if len(unique)!=history:
        raise ValueError('complete production history required')
    candidates = []
    for i in range(3):
        valid = np.isin(era,unique[len(unique)-(3-i)*width:len(unique)-(2-i)*width])
        available = outer & (era<era[valid].min()-gap)
        candidates.append((available,valid))
    common = min(len(np.unique(era[m])) for m,_ in candidates)
    full = history-width-gap
    out = []
    for available,valid in candidates:
        values = np.unique(era[available])
        if policy=='spread':
            values = values[np.rint(np.linspace(0,len(values)-1,common)).astype(int)]
        elif policy=='recent':
            values = values[-common:]
        elif policy=='full_span':
            values = np.unique(era[era<era[valid].min()-gap])[-full:]
            if len(values)!=full:raise ValueError('insufficient earlier context for equal full histories')
        elif policy!='ragged':raise ValueError('unknown count policy')
        train = np.isin(era,values)
        if era[valid].min()-era[train].max()<=gap:
            raise RuntimeError('inner availability gap violated')
        out.append((train,valid))
    return outer,out


def fit(spec,params,X,y,era,*,inner_width,gap,benchmark_trees,bound,seed,threads):
    import lightgbm as lgb
    X,y,era = np.asarray(X,np.float32),np.asarray(y,float),np.asarray(era)
    if 'count_policy' not in spec:
        keep = era>=spec.get('production_floor',2800)
        return prior_fit(spec,params,X[keep],y[keep],era[keep],inner_width=inner_width,gap=gap,
                         benchmark_trees=benchmark_trees,bound=bound,seed=seed,threads=threads)
    outer,folds = partitions(era,history=spec['history'],width=inner_width,gap=gap,
                            floor=spec.get('production_floor',2800),policy=spec['count_policy'])
    core = core_mean(X)
    p = dict(params);cap = p.pop('cap')
    sources,counts = [],[]
    for train,valid in folds:
        ref_model = lgbm.fit_fixed(p,X[train],y[train],None,n_iter=benchmark_trees,seed=seed,threads=threads)
        ref = lgbm.predict(ref_model,X[valid])
        score = stopping_scorer(y[valid],era[valid],ref,core[valid],bound)
        built,patience = lgbm._build_params({**p,'metric':'None'},seed=seed,threads=threads)
        dtr = lgb.Dataset(X[train],label=score_target(y[train],era[train],core[train]),params={'max_bin':built['max_bin']})
        dv = lgb.Dataset(X[valid],label=y[valid],reference=dtr)
        curve = []
        def feval(prediction,dataset):
            value = score(prediction);curve.append(value)
            return 'bounded_composite_proxy',value,True
        model = lgb.train(built,dtr,num_boost_round=cap,valid_sets=[dv],feval=feval,
                          callbacks=[lgb.early_stopping(patience,first_metric_only=True,verbose=False)])
        count = max(1,int(model.best_iteration or cap));counts.append(count)
        sources.append(dict(fit_min=int(era[train].min()),fit_max=int(era[train].max()),
                            fit_eras=len(np.unique(era[train])),fit_rows=int(train.sum()),
                            valid_min=int(era[valid].min()),valid_max=int(era[valid].max()),count=count,
                            curve=curve))
    count = median_count(counts)
    model = lgbm.fit_fixed(p,X[outer],score_target(y[outer],era[outer],core[outer]),None,
                          n_iter=count,seed=seed,threads=threads)
    return model,dict(n_iter=count,counts=counts,count_sources=sources,count_policy=spec['count_policy'],
                      train_min=int(era[outer].min()),train_max=int(era[outer].max()),training_rows=int(outer.sum()))
