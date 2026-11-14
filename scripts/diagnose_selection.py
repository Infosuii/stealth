#!/usr/bin/env python3
"""Read-only original trials, matched refits and model-selection calibration diagnostics."""
import argparse,json,logging,sqlite3
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
import lightgbm as lgb
import diagnose_models as d
from stealth import _runtime as rt
from stealth.campaign_models import tree_arrays,booster_params
from stealth.era_boost import centred_target
from stealth.representation_models import projection,params_for

FRAME=None;COLUMNS=None;OUT=None;THREADS=6;LIMIT=None
log=logging.getLogger('selection-forensics')


def decode_parameter(value,distribution_json):
    dist=json.loads(distribution_json);name=dist['name']
    if name=='CategoricalDistribution':return dist['attributes']['choices'][int(value)]
    if name=='IntDistribution':return int(value)
    if name=='FloatDistribution':return float(value)
    raise ValueError(f'unsupported stored distribution {name}')


def read_trials(path):
    before=d.sha(path);result=[]
    with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as db:
        rows=db.execute("select trial_id,number from trials where state='COMPLETE' order by number").fetchall()
        for ident,number in rows:
            params={key:decode_parameter(value,dist) for key,value,dist in db.execute('select param_name,param_value,distribution_json from trial_params where trial_id=?',(ident,))}
            attrs={key:json.loads(value) for key,value in db.execute('select key,value_json from trial_user_attributes where trial_id=?',(ident,))}
            value=db.execute('select value from trial_values where trial_id=? and objective=0',(ident,)).fetchone()[0]
            if len(attrs['counts'])!=3:raise RuntimeError('invalid original count vector')
            result.append({'number':number,'params':params,'counts':attrs['counts'],'inner_scores':attrs['scores'],'objective':value})
    if d.sha(path)!=before:raise RuntimeError('source trial database changed')
    return result,before


def contextual(values,era):
    result=values.copy()
    for e in np.unique(era):
        keep=np.asarray(era)==e;z=values[keep]
        result[keep]=(z-z.mean(axis=0))/np.maximum(z.std(axis=0),.1)
    return result


def run_window(window):
    with threadpool_limits(limits=THREADS):
        tr,te,X,Xt=d.arrays(FRAME,COLUMNS,window);era=tr.era.to_numpy();test_era=te.era.to_numpy();y=tr.target_everest.to_numpy()
        base=projection(X,era,COLUMNS)
        dense=rt.basis_design(X,'linear');dense_test=rt.basis_design(Xt,'linear')
        latent=(dense-base['x_mean'])@base['projection'];latent_test=(dense_test-base['x_mean'])@base['projection']
        context,context_test=contextual(latent,era),contextual(latent_test,test_era)
        for source,name in [('campaign03','nested_trees'),('representations04','latent_trees'),('repairs07','preserved_latent_trees')]:
            folder,done,old,cached=d.checkpoint(source,window['name'],name)
            trials,db_sha=read_trials(folder/'search.sqlite')
            records=json.loads((folder/'training.json').read_text())['selected']
            selected=[r['trial'] for r in records];predictions={};rows=[];fits={}
            for trial in trials[:LIMIT]:
                p=trial['params'];count=max(1,int(np.median(trial['counts'])))
                if source=='campaign03':
                    xx,yy,w=tree_arrays(X.astype(np.float32),y,era,p)
                    params=booster_params(p,len(yy),17,THREADS);bins=16;xv=Xt.astype(np.float32)
                else:
                    n=p['components'];z,zv=(context,context_test) if p['context'] else (latent,latent_test)
                    xx=np.concatenate([dense,z[:,:n]],axis=1) if p['add_raw'] else z[:,:n]
                    xv=np.concatenate([dense_test,zv[:,:n]],axis=1) if p['add_raw'] else zv[:,:n]
                    xx=xx.astype(np.float32);yy,w=centred_target(y,era);params=params_for(p,len(y),17,THREADS);bins=63
                model=lgb.train(params,lgb.Dataset(xx,label=yy,weight=w,params={'max_bin':bins}),num_boost_round=count)
                prediction=np.asarray(model.predict(xv,num_threads=1),dtype=float)
                fit=d.score(prediction,te);fits[f'trial{trial["number"]}']=fit
                if trial['number'] in selected:predictions[trial['number']]=prediction
                rows.append({'trial':trial['number'],'inner_objective':trial['objective'],'inner_mean':float(np.mean(trial['inner_scores'])),
                    'outer_fit':float(fit.mean()),'requested_count':count,'actual_count':model.current_iteration(),'selected':trial['number'] in selected})
                if len(rows)%10==0:log.info('%s/%s refit %d/%d',source,window['name'],len(rows),len(trials))
            verified=False
            if all(number in predictions for number in selected):
                values=np.array([predictions[number] for number in selected])
                combined=d.rank_ensemble(values,test_era) if done.get('blob_rule')=='gaussian_mean' else values.mean(axis=0)
                np.testing.assert_allclose(combined,cached.prediction.to_numpy(),atol=1e-10)
                verified=True
            a=np.array([r['inner_objective'] for r in rows]);b=np.array([r['outer_fit'] for r in rows])
            selected_values=[r['outer_fit'] for r in rows if r['selected']]
            correlation=float(spearmanr(a,b).statistic) if len(rows)>2 and np.std(a)>0 and np.std(b)>0 else None
            summary={'source_run':source,'window':window['name'],'trials':len(rows),'source_database_sha256':db_sha,
                'selected_ensemble_replayed':verified,'inner_outer_spearman':correlation,
                'outer_fit_min':float(b.min()),'outer_fit_median':float(np.median(b)),'outer_fit_max_oracle_only':float(b.max()),
                'selected_individual_fit_mean':float(np.mean(selected_values)) if selected_values else None,
                'selected_outer_percentiles':[float(np.mean(b<=v)) for v in selected_values]}
            dst=OUT/source/window['name'];dst.mkdir(parents=True,exist_ok=True)
            d.write(dst/'summary.json',summary);d.write(dst/'trials.json',rows);pd.DataFrame(fits).to_parquet(dst/'fits.parquet')
            log.info('%s/%s finished; inner/outer rank correlation %s',source,window['name'],correlation)
    return window['name']


def main():
    global FRAME,COLUMNS,OUT,THREADS,LIMIT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--threads',type=int,default=6);parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--run-id',default='forensics08');parser.add_argument('--limit',type=int)
    args=parser.parse_args();THREADS=args.threads;LIMIT=args.limit
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    OUT=d.REPO/'artifacts'/args.run_id;OUT.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),Path(d.__file__),d.REPO/'stealth/_runtime.py',d.REPO/'stealth/campaign_models.py',d.REPO/'stealth/representation_models.py']
    plan={'data_sha256':d.sha(d.REPO/'data/futures_train.parquet'),'code_sha256':{str(p.relative_to(d.REPO)):d.sha(p) for p in files},'reader_ceiling':5300,'trial_limit':LIMIT}
    if (OUT/'plan.json').exists() and json.loads((OUT/'plan.json').read_text())!=plan:raise RuntimeError('diagnostic run changed')
    for source in ['campaign03','representations04','repairs07']:
        old=json.loads((d.REPO/'artifacts'/source/'plan.json').read_text())
        if old['data_sha256']!=plan['data_sha256']:raise RuntimeError('source data changed')
    d.write(OUT/'plan.json',plan);FRAME,COLUMNS=d.frame_data()
    windows=json.loads((d.REPO/'artifacts/campaign03/plan.json').read_text())['config']['discovery']
    # Linux fork shares immutable input while keeping each source database read-only.
    import multiprocessing
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        for future in as_completed([pool.submit(run_window,w) for w in windows]):log.info('%s window completed',future.result())
    d.write(OUT/'done.json',{'status':'done','plan_sha256':d.sha(OUT/'plan.json')})


if __name__=='__main__':main()
