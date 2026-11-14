"""Nested training algorithms for the long-compute campaign; no outer-test labels enter a fit."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from stealth import metrics
from stealth._runtime import basis_design
from stealth.research_models import era_target

log = logging.getLogger(__name__)


def inner_windows(era, config):
    unique = np.unique(era)
    width, n = int(config['width']), int(config['windows'])
    if len(unique) < n*width+config['gap']+config['min_train']:
        raise ValueError('insufficient eras for registered nested windows')
    for i in range(n):
        first = len(unique)-(n-i)*width
        valid_eras = unique[first:first+width]
        train = np.asarray(era) < valid_eras[0]-config['gap']
        valid = np.isin(era,valid_eras)
        if len(np.unique(np.asarray(era)[train])) < config['min_train']:
            raise ValueError('inner fit below minimum history')
        yield train,valid


def history_mask(era, history):
    era=np.asarray(era)
    return np.ones(len(era),dtype=bool) if history is None else era>era.max()-int(history)


def ridge_path(X,y,era,lambdas):
    """One covariance factorization for the whole dimensionless regularization path."""
    design=basis_design(X,'linear')
    target,weights,_,counts=era_target(np.asarray(y,dtype=float),np.asarray(era))
    order=np.argsort(era,kind='stable')
    starts=np.r_[0,np.cumsum(counts)[:-1]]
    means=np.add.reduceat(design[order],starts,axis=0)/counts[:,None]
    weighted=design*np.sqrt(weights[:,None])
    gram=weighted.T@weighted-len(y)/len(counts)*(means.T@means)
    gram=(gram+gram.T)*.5
    rhs=design.T@(weights*target)
    scale=max(float(np.trace(gram)/len(rhs)),1e-12)
    eigen,vectors=np.linalg.eigh(gram)
    eigen=np.maximum(eigen,0)
    penalties=np.asarray(lambdas,dtype=float)*scale
    if np.any(penalties<=0) or not np.isfinite(penalties).all():
        raise ValueError('ridge path requires finite positive regularization')
    coef=vectors@((vectors.T@rhs)[:,None]/(eigen[:,None]+penalties[None,:]))
    return coef,penalties


def fit_calibrated(spec,X,y,era,columns,inner):
    settings=[(h,float(lam)) for h in spec['histories'] for lam in spec['lambdas']]
    scores=np.zeros((len(settings),inner['windows']))
    for fold,(train,valid) in enumerate(inner_windows(era,inner)):
        start=0
        for history in spec['histories']:
            ids=np.flatnonzero(train)
            ids=ids[history_mask(np.asarray(era)[ids],history)]
            coef,_=ridge_path(X[ids],y[ids],np.asarray(era)[ids],spec['lambdas'])
            prediction=basis_design(X[valid],'linear')@coef
            for j in range(len(spec['lambdas'])):
                scores[start+j,fold]=metrics.per_era_fit_fast(pd.Series(prediction[:,j]),pd.Series(y[valid]),pd.Series(np.asarray(era)[valid])).mean()
            start+=len(spec['lambdas'])
    objective=scores.mean(axis=1)-inner['stability_penalty']*scores.std(axis=1)
    best=int(np.argmax(objective))
    history,lam=settings[best]
    rows=history_mask(era,history)
    coef,penalties=ridge_path(X[rows],y[rows],np.asarray(era)[rows],[lam])
    info={'history':history,'lambda':lam,'alpha':float(penalties[0]),'inner_scores':scores[best].tolist(),
          'path':[{'history':h,'lambda':l,'objective':float(v),'scores':s.tolist()} for (h,l),v,s in zip(settings,objective,scores)]}
    return {'kind':'linear','columns':columns,'coef':coef[:,0]},info


def tree_arrays(X,y,era,params):
    from stealth.era_boost import centred_target
    keep=history_mask(era,params['history'])
    X,y,era=X[keep],y[keep],np.asarray(era)[keep]
    yy,w=centred_target(y,era) if params['centred'] else (np.asarray(y),np.ones(len(y)))
    if params['half_life'] is not None:
        w=w*np.exp2((era-era.max())/params['half_life'])
    w=w/np.mean(w)
    return X,yy,w


def booster_params(p,n,seed,threads):
    return {'objective':'regression','metric':'None','verbosity':-1,'num_threads':threads,'seed':seed,
            'deterministic':True,'force_row_wise':True,'num_leaves':p['num_leaves'],
            'learning_rate':p['learning_rate'],'min_data_in_leaf':max(100,int(n*p['leaf_fraction'])),
            'feature_fraction':p['feature_fraction'],'bagging_fraction':p['bagging_fraction'],'bagging_freq':1,
            'lambda_l2':p['lambda_l2'],'lambda_l1':p['lambda_l1'],'max_bin':16,'extra_trees':p['extra_trees']}


def tree_setting(trial):
    return {'num_leaves':trial.suggest_categorical('num_leaves',[4,8,16,31,63]),
            'learning_rate':trial.suggest_float('learning_rate',.002,.05,log=True),
            'leaf_fraction':trial.suggest_float('leaf_fraction',.005,.1,log=True),
            'feature_fraction':trial.suggest_float('feature_fraction',.1,1.),
            'bagging_fraction':trial.suggest_float('bagging_fraction',.5,1.),
            'lambda_l2':trial.suggest_float('lambda_l2',1,10000,log=True),
            'lambda_l1':trial.suggest_float('lambda_l1',.01,100,log=True),
            'extra_trees':trial.suggest_categorical('extra_trees',[False,True]),
            'history':trial.suggest_categorical('history',[None,1024,512]),
            'half_life':trial.suggest_categorical('half_life',[None,1024,512,256]),
            'centred':trial.suggest_categorical('centred',[False,True])}


def fit_trees(spec,X,y,era,inner,*,seed,threads,storage):
    import lightgbm as lgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X=np.asarray(X,dtype=np.float32)
    splits=list(inner_windows(era,inner))
    sampler=optuna.samplers.RandomSampler(seed=seed) if spec.get('sampler')=='random' else optuna.samplers.TPESampler(seed=seed)
    study=optuna.create_study(direction='maximize',sampler=sampler,
                              storage='sqlite:///'+str(storage),study_name='nested',load_if_exists=True)
    if not study.trials:
        for params in prior_proposals(spec['trials'],seed) if spec.get('sampler')=='random' else initial_proposals():
            study.enqueue_trial(params)
    if spec.get('sampler')=='random':
        # Persist every prior proposal up front; a process interruption cannot change the random pool.
        for trial in study.trials:
            if trial.state==optuna.trial.TrialState.RUNNING:
                fixed=trial.system_attrs.get('fixed_params')
                if not fixed:raise RuntimeError('interrupted prior trial lacks frozen parameters')
                study.tell(trial.number,state=optuna.trial.TrialState.FAIL)
                study.enqueue_trial(fixed)
    def objective(trial):
        p=tree_setting(trial)
        scores,counts=[],[]
        for train,valid in splits:
            xx,yy,w=tree_arrays(X[train],y[train],np.asarray(era)[train],p)
            params=booster_params(p,len(yy),seed,threads)
            target,eras=pd.Series(y[valid]),pd.Series(np.asarray(era)[valid])
            def evaluate(prediction,dataset):
                return 'fit20',float(metrics.per_era_fit_fast(pd.Series(prediction),target,eras).mean()),True
            ds=lgb.Dataset(xx,label=yy,weight=w,params={'max_bin':16})
            val=lgb.Dataset(X[valid],label=y[valid],reference=ds)
            model=lgb.train(params,ds,num_boost_round=spec['cap'],valid_sets=[val],feval=evaluate,
                callbacks=[lgb.early_stopping(spec['patience'],first_metric_only=True,verbose=False)])
            counts.append(max(1,int(model.best_iteration or spec['cap'])))
            scores.append(float(model.best_score['valid_0']['fit20']))
        value=float(np.mean(scores)-inner['stability_penalty']*np.std(scores))
        trial.set_user_attr('counts',counts)
        trial.set_user_attr('scores',scores)
        log.info('tree trial %d/%d objective %.5f scores %s counts %s',trial.number+1,spec['trials'],value,np.round(scores,5).tolist(),counts)
        return value
    # Completed trials are never silently repeated after an interruption.
    complete=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    study.optimize(objective,n_trials=max(0,spec['trials']-len(complete)))
    trials=sorted([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE],key=lambda t:t.value,reverse=True)[:spec['top_k']]
    models,info=[],[]
    for t in trials:
        xx,yy,w=tree_arrays(X,y,era,t.params)
        count=max(1,int(np.median(t.user_attrs['counts'])))
        model=lgb.train(booster_params(t.params,len(yy),seed,threads),lgb.Dataset(xx,label=yy,weight=w,params={'max_bin':16}),num_boost_round=count)
        models.append(model)
        info.append({'trial':t.number,'params':t.params,'count':count,'objective':t.value,'scores':t.user_attrs['scores']})
    return models,{'trials_completed':len([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]),'selected':info}


def initial_proposals():
    return [{'num_leaves':8,'learning_rate':.03,'leaf_fraction':.02,'feature_fraction':.3,'bagging_fraction':.7,
        'lambda_l2':10,'lambda_l1':.01,'extra_trees':False,'history':None,'half_life':None,'centred':False},
        {'num_leaves':8,'learning_rate':.005,'leaf_fraction':.03,'feature_fraction':.3,'bagging_fraction':.8,
        'lambda_l2':100,'lambda_l1':1,'extra_trees':False,'history':None,'half_life':512,'centred':True}]


def prior_proposals(count,seed):
    """Generate a frozen prior pool without looking at any target or objective value."""
    import optuna
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.RandomSampler(seed=seed))
    for p in initial_proposals():study.enqueue_trial(p)
    result=[]
    for _ in range(count):
        trial=study.ask();result.append(tree_setting(trial));study.tell(trial,0.)
    return result
