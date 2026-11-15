"""GPU CatBoost ranking controls with identical composite-proxy checkpoint selection."""
from __future__ import annotations

import numpy as np

from stealth.models import lgbm
from stealth.temporal_models import core_mean,stopping_scorer


def ordered(X,y,era):
    X,y,era=np.asarray(X,np.float32),np.asarray(y,float),np.asarray(era)
    order=np.argsort(era,kind='stable')
    return X[order],y[order],era[order]


def pool(X,y,era,kind):
    from catboost import Pool
    X,y,era=ordered(X,y,era)
    return Pool(X,label=y,group_id=era if kind!='RMSE' else None)


def params(spec, *, seed, device='GPU'):
    return dict(loss_function=spec['loss'],iterations=spec['cap'],learning_rate=spec['learning_rate'],
                depth=spec['depth'],l2_leaf_reg=spec['l2_leaf_reg'],border_count=16,
                bootstrap_type='Bayesian',bagging_temperature=.5,random_strength=1.,
                task_type=device,**(dict(devices='0',gpu_ram_part=.8) if device=='GPU' else {}),
                thread_count=spec['threads'],random_seed=seed,metric_period=spec['checkpoint_step'],
                verbose=False,allow_writing_files=False)


def checkpoint_counts(cap,step):
    return sorted(set([1,min(4,cap),min(16,cap),*range(step,cap+1,step),cap]))


def cpu_predict(model,X,*,count=0,threads=4):
    # Ranker's narrower convenience signature omits task_type; the shared public base API exposes it.
    from catboost import CatBoost
    return np.asarray(CatBoost.predict(model,X,prediction_type='RawFormulaVal',ntree_end=count,task_type='CPU',thread_count=threads),float)


def checkpoint_predictions(model,X,cap,step,threads):
    """Cumulative CPU inference avoids rescoring every tree prefix from scratch."""
    from catboost import CatBoost
    allowed=set(checkpoint_counts(cap,step));seen=set()
    for count in sorted({1,min(4,cap),min(16,cap)}):
        seen.add(count);yield count,cpu_predict(model,X,count=count,threads=threads)
    for i,prediction in enumerate(CatBoost.staged_predict(model,X,eval_period=step,thread_count=threads)):
        count=min((i+1)*step,cap)
        if count in allowed and count not in seen:
            seen.add(count);yield count,np.asarray(prediction,float)


def fit(spec,baseline,X,y,era,*,seed,gap,inner_width,bound):
    from catboost import CatBoostRegressor,CatBoostRanker
    from catboost.utils import get_gpu_device_count
    if get_gpu_device_count()<1:raise RuntimeError('registered GPU ranking fit requires CUDA; no CPU fallback')
    X,y,era=ordered(X,y,era)
    keep=(era>=2800)&(era>era.max()-spec['history'])
    X,y,era=X[keep],y[keep],era[keep]
    unique=np.unique(era)
    if len(unique)<inner_width+gap+200:raise ValueError('insufficient ranking fit history')
    cut=unique[-inner_width];inner,valid=era<cut-gap,era>=cut
    cls=CatBoostRegressor if spec['loss']=='RMSE' else CatBoostRanker
    p=params(spec,seed=seed)
    # Fully train the inner trajectory; every loss uses the SAME explicit selection metric and count grid.
    model=cls(**p)
    model.fit(pool(X[inner],y[inner],era[inner],spec['loss']))
    bp=dict(baseline);bp.pop('cap');bp.pop('early_stopping_rounds',None)
    reference=lgbm.fit_fixed(bp,X[inner],y[inner],None,n_iter=200,seed=17,threads=spec['threads'])
    score=stopping_scorer(y[valid],era[valid],lgbm.predict(reference,X[valid]),core_mean(X[valid]),bound)
    validation=pool(X[valid],y[valid],era[valid],spec['loss'])
    curve=[]
    for count,prediction in checkpoint_predictions(model,validation,spec['cap'],spec['checkpoint_step'],spec['threads']):
        curve.append(dict(count=count,score=score(prediction)))
    curve.sort(key=lambda row:row['count'])
    count=max(curve,key=lambda row:row['score'])['count']
    full=cls(**{**p,'iterations':count})
    full.fit(pool(X,y,era,spec['loss']))
    info=dict(n_iter=count,inner_cap=spec['cap'],curve=curve,train_min=int(era.min()),train_max=int(era.max()),
              inner_fit_max=int(era[inner].max()),inner_valid_min=int(era[valid].min()),
              inner_valid_max=int(era[valid].max()),loss=spec['loss'],seed=seed,task_type='GPU',
              nondeterminism='GPU native training may vary; resume the byte-bound actual model, never silently retrain')
    return full,info
