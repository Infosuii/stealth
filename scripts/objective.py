#!/usr/bin/env python3
"""Local OOS teacher generation, matched supervised-target fits and historical replication."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,importlib.metadata,json,logging,multiprocessing
from pathlib import Path
import sys,time
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import research_run as r
from stealth import cv
from stealth.models import lgbm,cat
from stealth.temporal_models import blocks,core_mean,composite,rank_by_era,fit as raw_fit
from stealth.objective_models import fit_cpu,fit_gpu
from stealth.ranking_models import cpu_predict

CFG=FRAME=RUN=PLAN=TEACHER=TEACHER_SHA=None
log=logging.getLogger('objective')


def load():
    global CFG,FRAME,RUN,PLAN
    CFG=yaml.safe_load((REPO/'configs/objective.yaml').read_text());RUN=REPO/'artifacts'/CFG['name'];RUN.mkdir(parents=True,exist_ok=True)
    files=['scripts/objective.py','scripts/research_run.py','stealth/objective_models.py','stealth/temporal_models.py',
           'stealth/online_missing.py','stealth/online_blend.py','stealth/ranking_models.py','stealth/metrics.py','stealth/cv.py',
           'stealth/models/lgbm.py','stealth/models/cat.py','docs/OBJECTIVE-16.md']
    plan=dict(protocol='honest-supervised-utility-v1',config=CFG,data_sha256=r.sha(REPO/'data/futures_train.parquet'),
              code_sha256={f:r.sha(REPO/f) for f in files},
              prior_baseline_plan_sha256=r.sha(REPO/'artifacts/online14r/plan.json'),
              versions={p:importlib.metadata.version(p) for p in ('numpy','pandas','lightgbm','catboost','scipy')})
    PLAN=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest();plan['plan_sha256']=PLAN
    path=RUN/'plan.json'
    if path.exists() and json.loads(path.read_text())['plan_sha256']!=PLAN:raise RuntimeError('objective experiment changed; use a new run ID')
    if not path.exists():r.write_json(path,plan)
    FRAME=pd.read_parquet(REPO/'data/futures_train.parquet',filters=[('exped','<','exped_5925')])
    FRAME['era']=FRAME.exped.str.rsplit('_',n=1).str[-1].astype(np.int32)
    FRAME=FRAME.loc[FRAME.era>=CFG['train_min_era']]
    if FRAME.era.max()>=5925 or not FRAME.index.is_unique:raise RuntimeError('objective reader boundary/index')
    log.info('objective reader %d..%d; plan %s',FRAME.era.min(),FRAME.era.max(),PLAN)


def checked(path):
    doc=json.loads(path.read_text())
    if doc['plan_sha256']!=PLAN:raise RuntimeError('objective checkpoint plan changed')
    for f,digest in doc.get('files',{}).items():
        if r.sha(path.parent/f)!=digest:raise RuntimeError('objective checkpoint bytes changed')
    return doc


def splits(frame,block):
    tr=frame.loc[frame.era<block['train_end']];te=frame.loc[frame.era.between(block['test_start'],block['test_end'])]
    if tr.empty or te.empty or te.era.min()-tr.era.max()<=40:raise RuntimeError('objective label availability')
    return tr,te


def arrays(tr,te):
    columns=[x for x in FRAME if x.startswith('feature_')]
    X,Xt=(f[columns].to_numpy(np.float32,copy=True) for f in (tr,te));X[X<0]=np.nan;Xt[Xt<0]=np.nan
    return X,Xt


def finish(directory,model,prediction,Xt,te,info,seconds,family,seed=None):
    directory.mkdir(parents=True,exist_ok=True);suffix='' if seed is None else str(seed)
    blob=lgbm.to_bytes(model) if family=='lgbm' else cat.to_bytes(model)
    replay=lgbm.predict(lgbm.from_bytes(blob),Xt) if family=='lgbm' else cat.predict(cat.from_bytes(blob),Xt)
    np.testing.assert_allclose(replay,prediction,atol=1e-12,rtol=1e-12)
    if not np.isfinite(prediction).all():raise RuntimeError('nonfinite objective prediction')
    names=[f'model{suffix}.bin',f'prediction{suffix}.parquet',f'training{suffix}.json']
    (directory/names[0]).write_bytes(blob)
    pd.DataFrame(dict(era=te.era,prediction=prediction),index=te.index).to_parquet(directory/names[1]);r.write_json(directory/names[2],info)
    marker='done.json' if seed is None else f'seed{seed}.done.json'
    record=dict(plan_sha256=PLAN,seconds=seconds,family=family,teacher_manifest_sha256=info.get('teacher_manifest_sha256'),serialized_replay_max_error=float(np.max(np.abs(replay-prediction))),files={f:r.sha(directory/f) for f in names})
    r.write_json(directory/marker,record);return record


def teacher_blocks():
    return blocks([dict(name='teacher',test_start=CFG['teacher_start'],test_end=CFG['teacher_end'])],CFG['cadence'],CFG['gap'])


def teacher_job(block):
    directory=RUN/block['name']/'teacher'
    if (directory/'done.json').exists():checked(directory/'done.json');return f'{block["name"]} reused'
    start=time.perf_counter();tr,te=splits(FRAME,block);X,Xt=arrays(tr,te)
    spec=dict(history=1024,half_life=None,score_aware=False)
    with threadpool_limits(limits=CFG['threads']):
        model,info=raw_fit(spec,CFG['baseline'],X,tr.target_everest.to_numpy(),tr.era.to_numpy(),inner_width=CFG['inner_width'],gap=CFG['gap'],benchmark_trees=200,bound=CFG['score_bound'],seed=17,threads=CFG['threads'])
        prediction=lgbm.predict(model,Xt)
    info.update(prediction_start=block['test_start'],prediction_end=block['test_end'])
    finish(directory,model,prediction,Xt,te,info,time.perf_counter()-start,'lgbm')
    return f'{block["name"]} done {time.perf_counter()-start:.1f}s'


def parallel(jobs,fn):
    with ProcessPoolExecutor(max_workers=CFG['workers'],mp_context=multiprocessing.get_context('fork')) as pool:
        for future in as_completed([pool.submit(fn,*args) for args in jobs]):log.info('%s',future.result())


def teachers():
    parallel([(b,) for b in teacher_blocks()],teacher_job)
    parts=[];provenance=[]
    for b0 in teacher_blocks():
        d=RUN/b0['name']/'teacher';checked(d/'done.json');info=json.loads((d/'training.json').read_text())
        if info['train_max']>=b0['test_start']-40:raise RuntimeError('teacher reference reached prediction horizon')
        parts.append(pd.read_parquet(d/'prediction.parquet'));provenance.append(dict(block=b0,train_max=info['train_max'],done_sha256=r.sha(d/'done.json')))
    pred=pd.concat(parts)
    if not pred.index.is_unique:raise RuntimeError('overlapping teacher predictions')
    pred.to_parquet(RUN/'teacher-predictions.parquet');r.write_json(RUN/'teacher-provenance.json',provenance)
    r.write_json(RUN/'teacher-done.json',dict(plan_sha256=PLAN,files={f:r.sha(RUN/f) for f in ['teacher-predictions.parquet','teacher-provenance.json']}))


def teacher_load():
    global TEACHER,TEACHER_SHA
    checked(RUN/'teacher-done.json');TEACHER_SHA=r.sha(RUN/'teacher-done.json')
    for item in json.loads((RUN/'teacher-provenance.json').read_text()):
        path=RUN/item['block']['name']/'teacher'/'done.json'
        if r.sha(path)!=item['done_sha256']:raise RuntimeError('teacher provenance marker changed')
        checked(path)
    TEACHER=pd.read_parquet(RUN/'teacher-predictions.parquet').prediction


def native_cached(path):
    doc=checked(path)
    if doc.get('teacher_manifest_sha256')!=TEACHER_SHA:raise RuntimeError('cached native fit used a different teacher')
    return doc


def cpu_job(block,spec):
    directory=RUN/block['name']/spec['name']
    if (directory/'done.json').exists():native_cached(directory/'done.json');return f'{block["name"]}/{spec["name"]} reused'
    start=time.perf_counter();tr,te=splits(FRAME,block);X,Xt=arrays(tr,te);y,e=tr.target_everest.to_numpy(),tr.era.to_numpy()
    with threadpool_limits(limits=CFG['threads']):
        if spec['name']=='baseline':
            model,info=raw_fit(dict(history=None,half_life=None,score_aware=False),CFG['baseline'],X,y,e,inner_width=CFG['inner_width'],gap=40,benchmark_trees=200,bound=CFG['score_bound'],seed=17,threads=CFG['threads'])
        else:model,info=fit_cpu(spec,CFG['baseline'],X,y,e,TEACHER.reindex(tr.index).to_numpy(),width=CFG['inner_width'],gap=40,seed=17,threads=CFG['threads'],bound=CFG['score_bound'])
        prediction=lgbm.predict(model,Xt)
    info['teacher_manifest_sha256']=TEACHER_SHA
    finish(directory,model,prediction,Xt,te,info,time.perf_counter()-start,'lgbm')
    return f'{block["name"]}/{spec["name"]} {time.perf_counter()-start:.1f}s, trees={info["n_iter"]}'


def gpu_job(block,spec):
    directory=RUN/block['name']/spec['name'];directory.mkdir(parents=True,exist_ok=True)
    if (directory/'done.json').exists():native_cached(directory/'done.json');log.info('%s/%s reused',block['name'],spec['name']);return
    tr,te=splits(FRAME,block);X,Xt=arrays(tr,te);scores=[];records=[]
    for seed in spec['seeds']:
        marker=directory/f'seed{seed}.done.json'
        if marker.exists():record=native_cached(marker)
        else:
            start=time.perf_counter();log.info('GPU %s/%s seed %d starting',block['name'],spec['name'],seed)
            with threadpool_limits(limits=spec['threads']):
                model,info=fit_gpu(spec,CFG['baseline'],X,tr.target_everest.to_numpy(),tr.era.to_numpy(),TEACHER.reindex(tr.index).to_numpy(),width=CFG['inner_width'],gap=40,seed=seed,bound=CFG['score_bound'])
                prediction=cpu_predict(model,Xt,threads=spec['threads'])
                info['teacher_manifest_sha256']=TEACHER_SHA
                record=finish(directory,model,prediction,Xt,te,info,time.perf_counter()-start,'catboost',seed)
            log.info('GPU %s/%s seed %d %.1fs, trees=%d',block['name'],spec['name'],seed,record['seconds'],info['n_iter'])
        p=pd.read_parquet(directory/f'prediction{seed}.parquet').reindex(te.index)
        if not np.array_equal(p.era,te.era):raise RuntimeError('objective GPU seed alignment')
        scores.append(rank_by_era(p.prediction,te.era));records.append(record)
    pd.DataFrame(dict(era=te.era,prediction=np.mean(scores,axis=0)),index=te.index).to_parquet(directory/'prediction.parquet')
    files=['prediction.parquet']+[f for seed in spec['seeds'] for f in [f'model{seed}.bin',f'prediction{seed}.parquet',f'training{seed}.json',f'seed{seed}.done.json']]
    r.write_json(directory/'done.json',dict(plan_sha256=PLAN,seconds=sum(d['seconds'] for d in records),family='catboost_group',teacher_manifest_sha256=TEACHER_SHA,
                                           serialized_replay_max_error=max(d['serialized_replay_max_error'] for d in records),files={f:r.sha(directory/f) for f in files}))


def predictions(name):
    pieces=[]
    for b0 in blocks(CFG['intervals'],CFG['cadence'],40):
        directory=RUN/b0['name']/name;native_cached(directory/'done.json');pieces.append(pd.read_parquet(directory/'prediction.parquet'))
    return pd.concat(pieces)


def evaluate(pred,base,teacher,frame,cores,w,comparison=None):
    mask=base.era.between(w['test_start'],w['test_end']).to_numpy();scenario={};a=bb=None
    comparison=base.prediction.to_numpy() if comparison is None else np.asarray(comparison)
    for label,ref in [('expanding',base.prediction.to_numpy()),('rolling',teacher)]:
        for i,core in enumerate(cores):
            p=composite(pred[mask],frame.target_everest.to_numpy()[mask],base.era.to_numpy()[mask],ref[mask],core[mask],bound=CFG['score_bound'])
            q=composite(comparison[mask],frame.target_everest.to_numpy()[mask],base.era.to_numpy()[mask],ref[mask],core[mask],bound=CFG['score_bound'])
            scenario[f'{label}/core_{i}']=float((p.score-q.score).mean())
            if label=='expanding' and i==0:a,bb=p,q
    delta=a-bb;segments=np.array_split(delta.index.to_numpy(),3)
    return dict(metrics=a.mean().to_dict(),baseline=bb.mean().to_dict(),scenario_score_gains=scenario,
                paired={k:cv.paired_block_bootstrap(delta[k],**CFG['bootstrap']).as_dict() for k in ['fit','score']},
                windows={str(i):{k:float(delta.loc[e,k].mean()) for k in ['fit','score']} for i,e in enumerate(segments)},
                block_sensitivity={str(b):cv.paired_block_bootstrap(delta.score,**{**CFG['bootstrap'],'block':b}).as_dict() for b in [40,60]})


def summarize():
    path=RUN/'results.json'
    if path.exists():return checked(path)
    teacher_load();base=predictions('baseline');frame=FRAME.loc[base.index];X=frame[[x for x in frame if x.startswith('feature_')]].to_numpy(float)
    cores=[core_mean(X),core_mean(X[:,::2]),core_mean(X[:,1::2])];teacher=TEACHER.reindex(base.index).to_numpy()
    component={s['name']:rank_by_era(predictions(s['name']).reindex(base.index).prediction,base.era) for s in CFG['cpu_recipes']+CFG['gpu_recipes']}
    component['full_committee']=(component['cpu_full']+component['gpu_full'])/2
    for n in ['cpu_full','gpu_full','full_committee']:component[n+'_blend']=(rank_by_era(component[n],base.era)+component['baseline'])/2
    root=REPO/'artifacts/online14r';old_plan=json.loads((root/'plan.json').read_text())
    if old_plan['data_sha256']!=json.loads((RUN/'plan.json').read_text())['data_sha256']:raise RuntimeError('prior baseline data differ')
    prior_parts=[]
    for marker in root.glob('*_*/baseline/done.json'):
        if marker.parent.parent.name.startswith('static_'):continue
        doc=json.loads(marker.read_text())
        if doc['plan_sha256']!=old_plan['plan_sha256']:raise RuntimeError('prior baseline checkpoint plan')
        for f,digest in doc['files'].items():
            if r.sha(marker.parent/f)!=digest:raise RuntimeError('prior baseline bytes changed')
        prior_parts.append(pd.read_parquet(marker.parent/'predictions.parquet'))
    prior=pd.concat(prior_parts).reindex(base.index).prediction.to_numpy()
    rows={};controls=[];prior_comparisons=[]
    for interval in CFG['intervals']:
        rr=[]
        for name in CFG['comparisons']:rr.append(dict(name=name,**evaluate(component[name],base,teacher,frame,cores,interval)))
        g=CFG['promotion'];stopped=False
        for i,row in enumerate(sorted(rr,key=lambda r:r['paired']['score']['p_value'])):
            p=row['paired'];limit=g['alpha']/(len(rr)-i);ok=not stopped and p['score']['inference_valid'] and p['score']['p_value']<=limit;stopped=stopped or not ok
            row['holm_pass']=bool(ok);row['holm_threshold']=limit
            row['passes_historical_gates']=bool(ok and p['fit']['diff_mean']>=g['min_fit_gain'] and p['score']['diff_mean']>=g['min_score_gain'] and min(p[k]['p_pos'] for k in ['fit','score'])>=g['p_pos'] and min(row['scenario_score_gains'].values())>=0 and all(min(v[k] for v in row['windows'].values())>=g['worst_window_gain'] for k in ['fit','score']) and sum(v['score']>0 for v in row['windows'].values())>=g['min_positive_windows'])
        rows[interval['name']]=rr
        for name in ['cpu_core','gpu_raw']:controls.append(dict(interval=interval['name'],name=name,**evaluate(component[name],base,teacher,frame,cores,interval)))
        if interval['name']!='earlier_replication':
            for name in CFG['comparisons']:
                prior_comparisons.append(dict(interval=interval['name'],name=name,**evaluate(component[name],base,teacher,frame,cores,interval,comparison=prior)))
    records=[checked(p) for p in RUN.glob('*/*/done.json')]
    doc=dict(plan_sha256=PLAN,run_id=CFG['name'],label='historical earlier replication and viewed later intervals; local score proxies',comparisons=rows,controls=controls,prior_floor2800_comparisons=prior_comparisons,promoted=False,
             checkpoint_groups=len(records),serialized_replay_max_error=max(d['serialized_replay_max_error'] for d in records))
    r.write_json(path,doc);return doc


def report(doc):
    out=REPO/'handoffs/results';r.write_json(out/'objective16-metrics.json',doc)
    lines=['# Honest supervised utility and historical replication 16','',doc['label'],'',
           '| interval | algorithm | FIT | baseline FIT | proxy score | baseline proxy | historical gates |','|---|---|---|---|---|---|---|']
    for phase,rows in doc['comparisons'].items():
        for row in rows:lines.append(f'| {phase} | {row["name"]} | {row["metrics"]["fit"]:.8f} | {row["baseline"]["fit"]:.8f} | {row["metrics"]["score"]:.8f} | {row["baseline"]["score"]:.8f} | {row["passes_historical_gates"]} |')
    lines+=['',f'{doc["checkpoint_groups"]} checkpoint groups; actual serialization replay max error {doc["serialized_replay_max_error"]:g}.',
            'No native candidate was promoted. All baselines use the new, consistently declared era-400 training floor.',
            'These are historical proxy scores, not platform scores or independent future evidence. Baseline01 unchanged; no platform writes.']
    (out/'objective16-results.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,choices=['init','teacher','cpu','gpu','summarize','report','rehearse']);args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s');load()
    if args.stage=='teacher':teachers()
    elif args.stage=='cpu':teacher_load();parallel([(bb,s) for bb in blocks(CFG['intervals'],100,40) for s in CFG['cpu_recipes']],cpu_job)
    elif args.stage=='gpu':
        teacher_load()
        for bb in blocks(CFG['intervals'],100,40):
            for s in CFG['gpu_recipes']:gpu_job(bb,s)
    elif args.stage=='summarize':print(json.dumps(summarize(),indent=2))
    elif args.stage=='report':report(summarize())
    elif args.stage=='rehearse':
        rng=np.random.default_rng(17);e=np.repeat(np.arange(400,1000),6);X=rng.integers(0,10,(len(e),8)).astype(np.float32);y=rng.integers(0,5,len(e))/4;ref=rng.normal(size=len(e))
        for s in CFG['gpu_recipes']:
            model,info=fit_gpu({**s,'cap':4,'depth':3,'checkpoint_step':2},CFG['baseline'],X,y,e,ref,width=40,gap=40,seed=17,bound=CFG['score_bound'])
            np.testing.assert_allclose(cpu_predict(model,X[:40]),cat.predict(cat.from_bytes(cat.to_bytes(model)),X[:40]),atol=1e-12)
            log.info('GPU objective rehearsal passed: %s',s['target'])


if __name__=='__main__':main()
