"""Utility-target native fits with a strictly out-of-time training reference."""
import numpy as np
from stealth.models import lgbm
from stealth.online_missing import utility
from stealth.temporal_models import core_mean,stopping_scorer
from stealth.ranking_models import params as cat_params,pool,checkpoint_predictions


def labels(kind,y,era,core,benchmark):
    y,era=np.asarray(y,float),np.asarray(era)
    if kind=='raw':return y.copy()
    ref=np.zeros(len(y)) if kind=='core' else np.asarray(benchmark,float)
    if kind not in ('core','full'):raise ValueError('unknown supervised target')
    if not np.isfinite(ref).all():raise ValueError('complete OOS training benchmark required')
    target=utility(y,era,ref,core);centered=np.empty(len(y))
    for e in np.unique(era):
        mask=era==e;centered[mask]=y[mask]-y[mask].mean();target[mask]-=target[mask].mean()
    sd=target.std()
    return target*(centered.std()/sd) if sd>0 else target


def prepare(spec,X,y,era,teacher,width,gap):
    X,y,era,teacher=np.asarray(X,np.float32),np.asarray(y,float),np.asarray(era),np.asarray(teacher,float)
    order=np.argsort(era,kind='stable');X,y,era,teacher=X[order],y[order],era[order],teacher[order]
    keep=np.ones(len(era),bool) if spec['history'] is None else era>era.max()-spec['history']
    X,y,era,teacher=X[keep],y[keep],era[keep],teacher[keep]
    unique=np.unique(era)
    if len(unique)<width+gap+200:raise ValueError('insufficient own-inner history')
    cut=unique[-width];inner,valid=era<cut-gap,era>=cut
    if spec['target']=='full' and not np.isfinite(teacher).all():raise ValueError('missing OOS teacher in final fitting partition')
    return X,y,era,teacher,inner,valid,core_mean(X)


def scorer(base,X,y,era,core,inner,valid,threads,bound):
    p=dict(base);p.pop('cap');p.pop('early_stopping_rounds',None)
    ref=lgbm.fit_fixed(p,X[inner],y[inner],None,n_iter=200,seed=17,threads=threads)
    return stopping_scorer(y[valid],era[valid],lgbm.predict(ref,X[valid]),core[valid],bound)


def fit_cpu(spec,base,X,y,era,teacher,*,width,gap,seed,threads,bound):
    import lightgbm as lgb
    X,y,era,teacher,inner,valid,core=prepare(spec,X,y,era,teacher,width,gap)
    p=dict(base);cap=p.pop('cap');built,patience=lgbm._build_params({**p,'metric':'None'},seed=seed,threads=threads)
    score=scorer(base,X,y,era,core,inner,valid,threads,bound);curve=[]
    def feval(prediction,dataset):
        value=score(prediction);curve.append(value);return 'bounded_composite_proxy',value,True
    dtr=lgb.Dataset(X[inner],label=labels(spec['target'],y[inner],era[inner],core[inner],teacher[inner]),params={'max_bin':16})
    dv=lgb.Dataset(X[valid],label=y[valid],reference=dtr)
    model=lgb.train(built,dtr,num_boost_round=cap,valid_sets=[dv],feval=feval,
                    callbacks=[lgb.early_stopping(patience,first_metric_only=True,verbose=False)])
    count=max(1,int(model.best_iteration or cap))
    full=lgbm.fit_fixed(p,X,labels(spec['target'],y,era,core,teacher),None,n_iter=count,seed=seed,threads=threads)
    return full,dict(n_iter=count,curve=curve,target=spec['target'],train_min=int(era.min()),train_max=int(era.max()),
                     inner_fit_max=int(era[inner].max()),inner_valid_min=int(era[valid].min()))


def fit_gpu(spec,base,X,y,era,teacher,*,width,gap,seed,bound):
    from catboost import CatBoostRegressor
    from catboost.utils import get_gpu_device_count
    if get_gpu_device_count()<1:raise RuntimeError('utility-target GPU fit requires visible CUDA')
    X,y,era,teacher,inner,valid,core=prepare(spec,X,y,era,teacher,width,gap)
    p=cat_params({**spec,'loss':'RMSE'},seed=seed)
    model=CatBoostRegressor(**p);model.fit(pool(X[inner],labels(spec['target'],y[inner],era[inner],core[inner],teacher[inner]),era[inner],'RMSE'))
    score=scorer(base,X,y,era,core,inner,valid,spec['threads'],bound)
    validation=pool(X[valid],y[valid],era[valid],'RMSE');curve=[]
    for count,prediction in checkpoint_predictions(model,validation,spec['cap'],spec['checkpoint_step'],spec['threads']):
        curve.append(dict(count=count,score=score(prediction)))
    curve.sort(key=lambda r:r['count']);count=max(curve,key=lambda r:r['score'])['count']
    full=CatBoostRegressor(**{**p,'iterations':count});full.fit(pool(X,labels(spec['target'],y,era,core,teacher),era,'RMSE'))
    if full.get_all_params()['task_type']!='GPU':raise RuntimeError('GPU objective fit silently changed device')
    return full,dict(n_iter=count,inner_cap=spec['cap'],curve=curve,target=spec['target'],seed=seed,native_task_type='GPU',
                     train_min=int(era.min()),train_max=int(era.max()),inner_fit_max=int(era[inner].max()),inner_valid_min=int(era[valid].min()))
