"""Fixed core-reference diversity and own-history, multi-window stopping controls."""
from __future__ import annotations

import numpy as np

from stealth.models import lgbm
from stealth.temporal_models import core_mean, fit as original_fit, score_target, stopping_scorer


def core_columns(spec, X):
    which = spec.get('training_core','all')
    if which=='all': return X
    if which=='even': return X[:,::2]
    if which=='odd': return X[:,1::2]
    raise ValueError('unknown core proxy')


def median_count(counts):
    if not len(counts) or any(int(c)<1 for c in counts):
        raise ValueError('positive own-inner counts required')
    return int(np.median(counts))


def fit(spec, params, X, y, era, *, inner_width, gap, benchmark_trees, bound, seed, threads):
    if spec.get('training_core','all')=='all' and spec.get('stopping_windows',1)==1:
        return original_fit(spec,params,X,y,era,inner_width=inner_width,gap=gap,
                            benchmark_trees=benchmark_trees,bound=bound,seed=seed,threads=threads)
    import lightgbm as lgb
    X,y,era = np.asarray(X,np.float32),np.asarray(y,float),np.asarray(era)
    keep = np.ones(len(era),bool) if spec['history'] is None else era > era.max()-spec['history']
    X,y,era = X[keep],y[keep],era[keep]
    unique = np.unique(era)
    windows = spec.get('stopping_windows',1)
    if len(unique)<windows*inner_width+gap+200:
        raise ValueError('insufficient own-inner history')
    target_core = core_mean(core_columns(spec,X))
    scoring_core = core_mean(X)  # the stopping objective does not change with the target intervention
    p = dict(params)
    cap = p.pop('cap')
    counts,sources = [],[]
    for i in range(windows):
        start = len(unique)-(windows-i)*inner_width
        valid_eras = unique[start:start+inner_width]
        inner,valid = era < valid_eras[0]-gap,np.isin(era,valid_eras)
        benchmark = lgbm.fit_fixed(p,X[inner],y[inner],None,n_iter=benchmark_trees,seed=seed,threads=threads)
        ref = lgbm.predict(benchmark,X[valid])
        score = stopping_scorer(y[valid],era[valid],ref,scoring_core[valid],bound)
        train_y = score_target(y[inner],era[inner],target_core[inner])
        built,patience = lgbm._build_params({**p,'metric':'None'},seed=seed,threads=threads)
        train = lgb.Dataset(X[inner],label=train_y,params={'max_bin':built['max_bin']})
        validation = lgb.Dataset(X[valid],label=y[valid],reference=train)
        def feval(prediction,dataset):return 'bounded_composite_proxy',score(prediction),True
        model = lgb.train(built,train,num_boost_round=cap,valid_sets=[validation],feval=feval,
                          callbacks=[lgb.early_stopping(patience,first_metric_only=True,verbose=False)])
        count = max(1,int(model.best_iteration or cap))
        counts.append(count)
        sources.append(dict(fit_max=int(era[inner].max()),valid_min=int(valid_eras.min()),
                            valid_max=int(valid_eras.max()),count=count))
    count = median_count(counts)
    model = lgbm.fit_fixed(p,X,score_target(y,era,target_core),None,n_iter=count,seed=seed,threads=threads)
    return model,dict(n_iter=count,counts=counts,count_sources=sources,training_core=spec.get('training_core','all'),
                      train_min=int(era.min()),train_max=int(era.max()),training_rows=len(era),
                      stopping='median_own_inner_composite_proxy')
