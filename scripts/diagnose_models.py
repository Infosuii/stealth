#!/usr/bin/env python3
"""Controlled forensic ablations of existing models; discovery labels only, no promotion or upload."""
from __future__ import annotations
import argparse,copy,hashlib,json,logging,sqlite3,sys,time
from pathlib import Path
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from stealth import metrics,cv
from stealth import _runtime as rt
from stealth.campaign_models import tree_arrays,booster_params
from stealth.representation_models import params_for
from stealth.research_models import to_bytes
from stealth.stability_models import moments

log=logging.getLogger('forensics')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def rank_ensemble(predictions,era):
    values=np.asarray(predictions,dtype=float)
    if values.ndim!=2 or values.shape[1]!=len(era) or not np.isfinite(values).all():
        raise ValueError('ensemble requires finite model-by-row predictions')
    result=np.empty(values.shape[1])
    for e in np.unique(era):
        keep=np.asarray(era)==e
        result[keep]=np.mean([rt.gaussianise(p[keep]) for p in values],axis=0)
    return result


def predict(family,model,X,era):
    if family=='basis' and model['kind']=='latent_tree':
        result=np.empty(len(X))
        for e in np.unique(era):
            keep=np.asarray(era)==e;result[keep]=rt.basis_forward(model,X[keep])
        return result
    return rt.predict_member(family,model,X.astype(np.float32) if family=='lgbm' else X)


def checkpoint(run,window,name):
    root=REPO/'artifacts'/run
    plan=json.loads((root/'plan.json').read_text())
    path=root/window/name;done=json.loads((path/'done.json').read_text())
    if done['plan_sha256']!=plan['plan_sha256']:raise RuntimeError('foreign checkpoint')
    for f,digest in done['files'].items():
        if sha(path/f)!=digest:raise RuntimeError('corrupt saved model or predictions')
    if done.get('training_sha256') and sha(path/'training.json')!=done['training_sha256']:
        raise RuntimeError('changed training record')
    blobs=[(path/f).read_bytes() for f in done['files'] if f.startswith('model')]
    models=[rt.load_member(done['family'],blob) for blob in blobs]
    return path,done,models,pd.read_parquet(path/'predictions.parquet')


def frame_data():
    frame=pd.read_parquet(REPO/'data/futures_train.parquet',filters=[('exped','<','exped_5300')])
    frame['era']=frame.exped.str.rsplit('_',n=1).str[-1].astype(np.int32)
    frame=frame.loc[frame.era>=2800].copy()
    assert frame.era.max()<5300 and frame.index.is_unique
    columns=[x for x in frame if x.startswith('feature_')]
    return frame,columns


def arrays(frame,columns,window):
    tr=frame.loc[frame.era<window['train_end']]
    te=frame.loc[frame.era.between(window['test_start'],window['test_end'])]
    X=tr[columns].to_numpy(dtype=float,copy=True);X[X<0]=np.nan
    Xt=te[columns].to_numpy(dtype=float,copy=True);Xt[Xt<0]=np.nan
    return tr,te,X,Xt


def score(prediction,te):
    return metrics.per_era_fit_fast(pd.Series(prediction),pd.Series(te.target_everest.to_numpy()),pd.Series(te.era.to_numpy()))


def score_cases(cases,baseline,te):
    result={};fits={}
    base=score(baseline,te);fits['baseline']=base
    for name,prediction in cases.items():
        for blend in (False,True):
            value=rank_ensemble([prediction,baseline],te.era.to_numpy()) if blend else prediction
            key=name+('/baseline_blend' if blend else '')
            fit=score(value,te);fits[key]=fit
            result[key]={'fit':float(fit.mean()),'baseline':float(base.mean()),'gain':float((fit-base).mean())}
    return result,fits


def inner_counts(folder,trial):
    database=folder/'search.sqlite';before=sha(database)
    with sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True) as db:
        row=db.execute("select a.value_json from trial_user_attributes a join trials t on t.trial_id=a.trial_id where t.number=? and a.key='counts'",(int(trial),)).fetchone()
    if row is None or sha(database)!=before:raise RuntimeError('missing or changing inner-count evidence')
    counts=json.loads(row[0])
    if len(counts)!=3 or min(counts)<1:raise RuntimeError('invalid own-inner counts')
    return counts,before


def drift_summary(coef,X,y,era,Xt,yt,et):
    _,history,var=moments(rt.basis_design(X,'linear'),y,era)
    recent=np.asarray(era)>np.max(era)-512
    _,near,nv=moments(rt.basis_design(X[recent],'linear'),np.asarray(y)[recent],np.asarray(era)[recent])
    _,future,fv=moments(rt.basis_design(Xt,'linear'),yt,et)
    mu,latest,test=history.mean(axis=0),near.mean(axis=0),future.mean(axis=0)
    mass=np.abs(coef*mu)
    difference=np.abs((np.nanmean(Xt,axis=0)-np.nanmean(X[recent],axis=0))/np.maximum(np.nanstd(X[recent],axis=0),.1))
    return {'full_history_test_covariance_correlation':float(np.corrcoef(mu,test)[0,1]),
        'recent_history_test_covariance_correlation':float(np.corrcoef(latest,test)[0,1]),
        'weighted_sign_flip_fraction':float(mass[mu*test<0].sum()/max(mass.sum(),1e-12)),
        'median_marginal_standardized_mean_change':float(np.median(difference)),
        'max_marginal_standardized_mean_change':float(np.max(difference)),
        'mean_training_feature_variance':float(var.mean()),'mean_test_feature_variance':float(fv.mean()),
        'train_missing_fraction':float(np.isnan(X).mean()),'test_missing_fraction':float(np.isnan(Xt).mean())}


def tree_ablation(run,window,frame,columns,out,threads):
    import lightgbm as lgb
    name='nested_trees' if run=='campaign03' else 'latent_trees'
    folder,done,models,cached=checkpoint(run,window['name'],name)
    _,_,base_models,base_cached=checkpoint(run,window['name'],'baseline')
    tr,te,X,Xt=arrays(frame,columns,window)
    assert cached.index.equals(te.index) and base_cached.index.equals(te.index)
    era=te.era.to_numpy();train_era=tr.era.to_numpy();y=tr.target_everest.to_numpy()
    original=np.array([predict(done['family'],m,Xt,era) for m in models])
    np.testing.assert_allclose(original.mean(axis=0),cached.prediction.to_numpy(),atol=1e-10)
    records=json.loads((folder/'training.json').read_text())['selected']
    changed=[];details=[]
    destination=out/run/window['name'];destination.mkdir(parents=True,exist_ok=True)
    projection_records=[]
    if run=='representations04':
        train_design=rt.basis_design(X,'linear');test_design=rt.basis_design(Xt,'linear')
        _,sig,var=moments(train_design,y,train_era)
        _,test_sig,test_var=moments(test_design,te.target_everest.to_numpy(),era)
        mu,tm=sig.mean(axis=0),test_sig.mean(axis=0)
        for i,model in enumerate(models):
            P=model['projection']
            projected_variance=moments(train_design@P,y,train_era)[2].mean(axis=0).sum()
            projection_records.append({'model':i,'dimensions':P.shape[1],'add_raw':bool(model['add_raw']),'context':bool(model['context']),
                'retained_feature_variance_fraction':float(projected_variance/max(var.mean(axis=0).sum(),1e-12)),
                'retained_mean_linear_covariance_fraction_train':float(np.sum((P.T@mu)**2)/max(mu@mu,1e-12)),
                'retained_mean_linear_covariance_fraction_test':float(np.sum((P.T@tm)**2)/max(tm@tm,1e-12))})
    for i,(record,model) in enumerate(zip(records,models)):
        counts,database_sha=inner_counts(folder,record['trial'])
        if record['count']!=int(np.median(counts)):raise RuntimeError('original refit is not the recorded median')
        p=record['params'];count=counts[-1]
        if run=='campaign03':
            xx,yy,w=tree_arrays(X.astype(np.float32),y,train_era,p)
            params=booster_params(p,len(yy),17,threads)
            booster=lgb.train(params,lgb.Dataset(xx,label=yy,weight=w,params={'max_bin':16}),num_boost_round=count)
            new=booster;blob=booster.model_to_string().encode();old_count=model.current_iteration()
        else:
            from stealth.era_boost import centred_target
            new={k:v for k,v in model.items() if not k.startswith('_')}
            xx=rt.latent_design(new,X,train_era).astype(np.float32);yy,w=centred_target(y,train_era)
            booster=lgb.train(params_for(p,len(y),17,threads),lgb.Dataset(xx,label=yy,weight=w,params={'max_bin':63}),num_boost_round=count)
            new['tree_blob']=np.frombuffer(booster.model_to_string().encode(),dtype=np.uint8)
            blob=to_bytes(new);old_count=model['_tree'].current_iteration()
        (destination/f'recent_model{i}.bin').write_bytes(blob)
        pred=predict(done['family'],new,Xt,era);changed.append(pred)
        train_pred=predict(done['family'],model,X,train_era)
        details.append({'model':i,'counts':counts,'original_requested_count':record['count'],'original_actual_count':old_count,
            'recent_requested_count':count,'recent_actual_count':booster.current_iteration(),'train_prediction_std':float(np.std(train_pred)),
            'test_prediction_std':float(np.std(original[i])),'original_fit':float(score(original[i],te).mean()),
            'recent_fit':float(score(pred,te).mean()),'database_sha256':database_sha,'refit_sha256':sha(destination/f'recent_model{i}.bin')})
    cases={'median/raw':original.mean(axis=0),'median/rank':rank_ensemble(original,era),
           'recent/raw':np.mean(changed,axis=0),'recent/rank':rank_ensemble(changed,era)}
    results,fits=score_cases(cases,base_cached.prediction.to_numpy(),te)
    pd.DataFrame(fits).to_parquet(destination/'fits.parquet')
    write(destination/'ablation.json',{'source_run':run,'window':window['name'],'members':details,'results':results,'projection_diagnostics':projection_records})
    log.info('%s/%s controlled tree ablations complete',run,window['name'])


def encoder_ablation(window,frame,columns,out):
    folder,done,models,cached=checkpoint('representations04',window['name'],'denoised_ridge')
    _,_,_,base=checkpoint('representations04',window['name'],'baseline')
    tr,te,X,Xt=arrays(frame,columns,window)
    assert cached.index.equals(te.index)
    design=rt.basis_design(Xt,'linear')
    full=np.mean([rt.basis_forward(m,Xt) for m in models],axis=0)
    raw=np.mean([design@m['coef'] for m in models],axis=0)
    np.testing.assert_allclose(full,cached.prediction.to_numpy(),atol=1e-10)
    results,fits=score_cases({'complete':full,'raw_term':raw,'encoder_term':full-raw},base.prediction.to_numpy(),te)
    destination=out/'denoising'/window['name'];destination.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(fits).to_parquet(destination/'fits.parquet')
    coef=np.mean([m['coef'] for m in models],axis=0)
    drift=drift_summary(coef,X,tr.target_everest.to_numpy(),tr.era.to_numpy(),Xt,te.target_everest.to_numpy(),te.era.to_numpy())
    write(destination/'ablation.json',{'window':window['name'],'results':results,'drift':drift})
    log.info('%s encoder/raw-term ablations complete',window['name'])


def stability_drift(window,frame,columns,out):
    tr,te,X,Xt=arrays(frame,columns,window);result={}
    for name in ['stable_covariance','adaptive_covariance']:
        _,done,models,cached=checkpoint('stability05',window['name'],name)
        prediction=predict(done['family'],models[0],Xt,te.era.to_numpy())
        np.testing.assert_allclose(prediction,cached.prediction.to_numpy(),atol=1e-10)
        result[name]=drift_summary(models[0]['coef'],X,tr.target_everest.to_numpy(),tr.era.to_numpy(),Xt,te.target_everest.to_numpy(),te.era.to_numpy())
    write(out/'stability'/window['name']/'drift.json',result)


def curves(out):
    records=[]
    for path in (REPO/'artifacts/campaign03').glob('*/*/seed*/training.json'):
        info=json.loads(path.read_text())['inner'];curve=info['curves']
        loss=[r['train_mse'] for r in curve if r['train_mse'] is not None]
        records.append({'window':path.parts[-4],'recipe':path.parts[-3],'seed_label':path.parts[-2],
            'selected_epoch':info['best_epoch'],'best_inner_fit':max(r['inner_fit'] for r in curve),
            'last_inner_fit':curve[-1]['inner_fit'],'first_train_mse':loss[0],'last_train_mse':loss[-1]})
    write(out/'learning-curves.json',records)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--threads',type=int,default=6)
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    from threadpoolctl import threadpool_limits
    out=REPO/'artifacts/forensics06';out.mkdir(parents=True,exist_ok=True)
    data_sha=sha(REPO/'data/futures_train.parquet');plans={}
    for run in ['campaign03','representations04','stability05']:
        p=json.loads((REPO/'artifacts'/run/'plan.json').read_text())
        if p['data_sha256']!=data_sha:raise RuntimeError('data differs from original experiment')
        plans[run]=p['plan_sha256']
    plan={'data_sha256':data_sha,'sources':plans,'code_sha256':sha(Path(__file__)),'reader_ceiling':5300,'controls':['median/raw','median/rank','recent/raw','recent/rank']}
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=plan:raise RuntimeError('forensic run changed')
    write(out/'plan.json',plan)
    frame,columns=frame_data();curves(out)
    windows=json.loads((REPO/'artifacts/campaign03/plan.json').read_text())['config']['discovery']
    with threadpool_limits(limits=args.threads):
        for w in windows:
            encoder_ablation(w,frame,columns,out)
            stability_drift(w,frame,columns,out)
            for run in ['campaign03','representations04']:tree_ablation(run,w,frame,columns,out,args.threads)
    write(out/'done.json',{'status':'done','finished_at':pd.Timestamp.now(tz='UTC').isoformat(),'plan_sha256':sha(out/'plan.json')})


if __name__=='__main__':main()
