"""Target-free projections, arbitrary-design ridge paths and nested latent-tree fitting."""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from stealth import metrics
from stealth._runtime import basis_design,basis_forward,latent_design,average_blob_scores
from stealth.campaign_models import inner_windows
from stealth.research_models import era_target

log=logging.getLogger(__name__)


def covariance(design,era):
    _,group,count=np.unique(era,return_inverse=True,return_counts=True)
    weights=len(era)/len(count)/count[group]
    order=np.argsort(era,kind='stable')
    means=np.add.reduceat(design[order],np.r_[0,np.cumsum(count)[:-1]],axis=0)/count[:,None]
    weighted=design*np.sqrt(weights[:,None])
    gram=weighted.T@weighted-len(era)/len(count)*(means.T@means)
    return (gram+gram.T)*.5,weights


def design_path(design,y,era,lambdas,*,penalty_mode='mean'):
    gram,weights=covariance(design,era)
    target=era_target(np.asarray(y),np.asarray(era))[0]
    rhs=design.T@(weights*target)
    if penalty_mode not in ('mean','trace'):raise ValueError('unknown ridge penalty mode')
    scale=max(float(np.trace(gram)/(len(rhs) if penalty_mode=='mean' else 1)),1e-12)
    eigen,vectors=np.linalg.eigh(gram)
    penalties=np.asarray(lambdas)*scale
    coef=vectors@((vectors.T@rhs)[:,None]/(np.maximum(eigen,0)[:,None]+penalties[None,:]))
    return coef,penalties


def group_head(raw,latent,y,era,lambdas,raw_strength):
    """Equalize group variance and use a dimension-invariant total-trace penalty."""
    if not np.isfinite(raw_strength) or raw_strength<0:raise ValueError('invalid raw-group strength')
    def trace_variance(values):
        _,count=np.unique(era,return_counts=True);order=np.argsort(era,kind='stable');d=values[order]
        starts=np.r_[0,np.cumsum(count)[:-1]]
        mean=np.add.reduceat(d,starts,axis=0)/count[:,None]
        var=np.add.reduceat(d*d,starts,axis=0)/count[:,None]-mean*mean
        return float(np.maximum(var,0).mean(axis=0).sum())
    raw_var,latent_var=trace_variance(raw),trace_variance(latent)
    scale=raw_strength*np.sqrt(latent_var/raw_var) if raw_var>1e-12 else 0.
    combined=np.concatenate([raw*scale,latent],axis=1)
    coef,penalties=design_path(combined,y,era,lambdas,penalty_mode='trace')
    return coef,scale,penalties


def projection(X,era,columns,n=64):
    design=basis_design(X,'linear')
    gram,_=covariance(design,era)
    _,vectors=np.linalg.eigh(gram)
    return {'kind':'latent_tree','columns':columns,'x_mean':design.mean(axis=0),
            'projection':vectors[:,-min(n,len(vectors)):][:,::-1].copy(),'context':False,'add_raw':False}


def grouped_prediction(models,X,era,rule='raw_mean'):
    result=np.empty(len(X))
    for e in np.unique(era):
        keep=np.asarray(era)==e
        result[keep]=average_blob_scores([basis_forward(m,X[keep]) for m in models],rule)
    return result


def fit_latent(spec,X,y,era,columns,inner,*,seed,threads,storage):
    import lightgbm as lgb
    import optuna
    from stealth.era_boost import centred_target
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    splits=list(inner_windows(era,inner))
    # Projection contains no labels and is fitted separately for every chronological inner split.
    bases=[projection(X[tr],np.asarray(era)[tr],columns) for tr,va in splits]
    def transform(base,n,context,add_raw,xx,ee):
        model={**base,'projection':base['projection'][:,:n],'context':context,'add_raw':add_raw}
        return latent_design(model,xx,ee).astype(np.float32),model
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=seed),
        storage='sqlite:///'+str(storage),study_name='latent',load_if_exists=True)
    def objective(trial):
        p={'components':trial.suggest_categorical('components',[8,16,32,64]),
           'context':trial.suggest_categorical('context',[False,True]),'add_raw':trial.suggest_categorical('add_raw',[True] if spec.get('force_raw') else [False,True]),
           'num_leaves':trial.suggest_categorical('num_leaves',[4,8,16,31]),'learning_rate':trial.suggest_float('learning_rate',.005,.03,log=True),
           'leaf_fraction':trial.suggest_float('leaf_fraction',.01,.1,log=True),'lambda_l2':trial.suggest_float('lambda_l2',10,1000,log=True),
           'feature_fraction':trial.suggest_float('feature_fraction',.3,1)}
        scores,counts=[],[]
        for (tr,va),base in zip(splits,bases):
            xx,_=transform(base,p['components'],p['context'],p['add_raw'],X[tr],np.asarray(era)[tr])
            xv,_=transform(base,p['components'],p['context'],p['add_raw'],X[va],np.asarray(era)[va])
            yy,w=centred_target(y[tr],np.asarray(era)[tr])
            params=params_for(p,len(yy),seed,threads)
            target,eras=pd.Series(y[va]),pd.Series(np.asarray(era)[va])
            def evaluate(prediction,dataset):return 'fit20',float(metrics.per_era_fit_fast(pd.Series(prediction),target,eras).mean()),True
            ds=lgb.Dataset(xx,label=yy,weight=w,params={'max_bin':63})
            model=lgb.train(params,ds,num_boost_round=spec['cap'],valid_sets=[lgb.Dataset(xv,label=y[va],reference=ds)],feval=evaluate,
                callbacks=[lgb.early_stopping(spec['patience'],first_metric_only=True,verbose=False)])
            counts.append(max(1,int(model.best_iteration or spec['cap'])))
            scores.append(float(model.best_score['valid_0']['fit20']))
        trial.set_user_attr('counts',counts);trial.set_user_attr('scores',scores)
        value=float(np.mean(scores)-inner['stability_penalty']*np.std(scores))
        log.info('latent trial %d/%d objective %.5f counts %s',trial.number+1,spec['trials'],value,counts)
        return value
    complete=lambda:[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    study.optimize(objective,n_trials=max(0,spec['trials']-len(complete())))
    trials=sorted(complete(),key=lambda t:t.value,reverse=True)[:spec['top_k']]
    base=projection(X,era,columns)
    yy,w=centred_target(y,era)
    models,records=[],[]
    for trial in trials:
        p=trial.params
        xx,model=transform(base,p['components'],p['context'],p['add_raw'],X,era)
        count=max(1,int(np.median(trial.user_attrs['counts'])))
        booster=lgb.train(params_for(p,len(y),seed,threads),lgb.Dataset(xx,label=yy,weight=w,params={'max_bin':63}),num_boost_round=count)
        model['tree_blob']=np.frombuffer(booster.model_to_string().encode(),dtype=np.uint8)
        models.append(model)
        records.append({'trial':trial.number,'params':p,'count':count,'objective':trial.value,'scores':trial.user_attrs['scores']})
    return models,{'trials_completed':len(complete()),'selected':records}


def params_for(p,n,seed,threads):
    return {'objective':'regression','metric':'None','verbosity':-1,'deterministic':True,'force_row_wise':True,
            'num_threads':threads,'seed':seed,'max_bin':63,'num_leaves':p['num_leaves'],'learning_rate':p['learning_rate'],
            'min_data_in_leaf':max(100,int(n*p['leaf_fraction'])),'lambda_l2':p['lambda_l2'],'feature_fraction':p['feature_fraction']}
