#!/usr/bin/env python3
"""GPU ranking campaign with source-verified native ensemble dependencies."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import yaml

import robust_temporal as b
from stealth.models import cat
from stealth.ranking_models import cpu_predict,fit as gpu_fit

c=b.c
REPO=b.REPO
log=logging.getLogger('ranking')


def load(confirmation=False):
    c.CFG=yaml.safe_load((REPO/'configs/ranking.yaml').read_text())
    c.RUN=REPO/'artifacts'/c.CFG['name'];c.RUN.mkdir(parents=True,exist_ok=True)
    files=['scripts/ranking.py','scripts/robust_temporal.py','scripts/temporal.py','scripts/research_run.py',
           'stealth/ranking_models.py','stealth/robust_temporal.py','stealth/temporal_models.py',
           'stealth/metrics.py','stealth/cv.py','stealth/models/lgbm.py','stealth/models/cat.py','docs/RANKING-13.md']
    plan=dict(protocol='gpu-ranking-v1',config=c.CFG,data_sha256=c.r.sha(REPO/'data/futures_train.parquet'),
              code_sha256={f:c.r.sha(REPO/f) for f in files},
              source_plan_sha256=c.r.sha(REPO/'artifacts/temporal11/plan.json'),
              versions={p:importlib.metadata.version(p) for p in ('numpy','pandas','lightgbm','scipy','catboost')})
    c.PLAN=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest();plan['plan_sha256']=c.PLAN
    path=c.RUN/'plan.json'
    if path.exists() and json.loads(path.read_text())['plan_sha256']!=c.PLAN:raise RuntimeError('ranking plan changed; use a new run ID')
    if not path.exists():c.r.write_json(path,plan)
    ceiling=c.CFG['reader_ceiling']
    if confirmation:
        if c.checked_json(c.RUN/'selection.json')['selected'] is None:raise RuntimeError('no frozen ranking winner')
        ceiling=c.CFG['excluded_from_research'][0]
    c.FRAME=pd.read_parquet(REPO/'data/futures_train.parquet',filters=[('exped','<',f'exped_{ceiling:04d}')])
    c.FRAME['era']=c.FRAME.exped.str.rsplit('_',n=1).str[-1].astype(np.int32)
    c.FRAME=c.FRAME.loc[c.FRAME.era>=c.CFG['train_min_era']].copy()
    if c.FRAME.era.max()>=ceiling or not c.FRAME.index.is_unique:raise RuntimeError('ranking reader boundary')
    log.info('loaded ranking eras %d..%d; plan %s',c.FRAME.era.min(),c.FRAME.era.max(),c.PLAN)


def cpu_fit(spec,params,X,y,era,**kwargs):
    return b.fit(spec,params,X,y,era,**kwargs)


def copy_controls():
    root=REPO/'artifacts/temporal11';source=json.loads((root/'plan.json').read_text())
    current=json.loads((c.RUN/'plan.json').read_text())
    if source['data_sha256']!=current['data_sha256']:raise RuntimeError('ranking dependency data changed')
    for f,digest in source['code_sha256'].items():
        if c.r.sha(REPO/f)!=digest:raise RuntimeError('ranking dependency implementation changed')
    for key in ('baseline','seed','threads','inner_width','gap','cadence','inner_benchmark_trees','train_min_era'):
        if source['config'][key]!=c.CFG[key]:raise RuntimeError(f'ranking dependency {key} changed')
    for name,group in source['config']['ensembles'].items():
        if c.CFG['ensembles'][name]!=group:raise RuntimeError('ranking dependency group changed')
    for src in root.glob('*/*/done.json'):
        doc=json.loads(src.read_text())
        if doc['plan_sha256']!=source['plan_sha256']:raise RuntimeError('dependency checkpoint plan changed')
        for f,digest in doc['files'].items():
            if c.r.sha(src.parent/f)!=digest:raise RuntimeError('dependency bytes changed')
        dest=c.RUN/src.parent.parent.name/src.parent.name;dest.mkdir(parents=True,exist_ok=True)
        target=dest/'done.json'
        if target.exists():c.checked_json(target);continue
        for f in doc['files']:shutil.copyfile(src.parent/f,dest/f)
        c.r.write_json(target,{**doc,'plan_sha256':c.PLAN,'seconds':0.,'copied_from':'temporal11',
                               'source_done_sha256':c.r.sha(src),'source_plan_sha256':source['plan_sha256']})


def gpu_job(block,spec):
    from threadpoolctl import threadpool_limits
    directory=c.RUN/block['name']/spec['name'];directory.mkdir(parents=True,exist_ok=True)
    done=directory/'done.json'
    if done.exists():c.checked_json(done);log.info('%s/%s reused',block['name'],spec['name']);return
    columns=[x for x in c.FRAME if x.startswith('feature_')]
    tr=c.FRAME.loc[c.FRAME.era<block['train_end']]
    te=c.FRAME.loc[c.FRAME.era.between(block['test_start'],block['test_end'])]
    if te.era.min()-tr.era.max()<=c.CFG['gap']:raise RuntimeError('ranking target availability violated')
    X,Xt=(f[columns].to_numpy(np.float32,copy=True) for f in (tr,te));X[X<0]=np.nan;Xt[Xt<0]=np.nan
    seed_predictions=[];records=[]
    for seed in spec['seeds']:
        marker=directory/f'seed{seed}.done.json'
        if marker.exists():record=c.checked_json(marker)
        else:
            log.info('GPU fitting %s/%s seed %d, inner cap %d',block['name'],spec['name'],seed,spec['cap'])
            started=time.perf_counter()
            with threadpool_limits(limits=spec['threads']):
                model,info=gpu_fit(spec,c.CFG['baseline'],X,tr.target_everest.to_numpy(),tr.era.to_numpy(),
                                   seed=seed,gap=c.CFG['gap'],inner_width=c.CFG['inner_width'],bound=c.CFG['score_bound'])
                info['native_task_type']=model.get_all_params()['task_type']
                if info['native_task_type']!='GPU':raise RuntimeError('native model was not fitted on GPU')
                prediction=cpu_predict(model,Xt,threads=spec['threads'])
                blob=cat.to_bytes(model)
                replay=cat.predict(cat.from_bytes(blob),Xt)
                np.testing.assert_allclose(replay,prediction,atol=1e-12,rtol=1e-12)
                if not np.isfinite(prediction).all():raise RuntimeError('nonfinite GPU model prediction')
            (directory/f'model{seed}.cbm').write_bytes(blob)
            pd.DataFrame(dict(era=te.era,prediction=prediction),index=te.index).to_parquet(directory/f'prediction{seed}.parquet')
            c.r.write_json(directory/f'training{seed}.json',info)
            files=[f'model{seed}.cbm',f'prediction{seed}.parquet',f'training{seed}.json']
            record=dict(plan_sha256=c.PLAN,seconds=time.perf_counter()-started,
                        serialized_replay_max_error=float(np.max(np.abs(replay-prediction))),
                        files={f:c.r.sha(directory/f) for f in files})
            c.r.write_json(marker,record)
            log.info('GPU finished %s/%s seed %d in %.1fs, selected %d',block['name'],spec['name'],seed,record['seconds'],info['n_iter'])
            del model;gc.collect()
        pp=pd.read_parquet(directory/f'prediction{seed}.parquet').reindex(te.index)
        if not np.array_equal(pp.era,te.era):raise RuntimeError('GPU seed prediction rows/eras differ')
        seed_predictions.append(c.rank_by_era(pp.prediction,te.era));records.append(record)
    prediction=np.mean(seed_predictions,axis=0)
    pd.DataFrame(dict(era=te.era,prediction=prediction),index=te.index).to_parquet(directory/'predictions.parquet')
    info=dict(seeds=spec['seeds'],counts={str(s):json.loads((directory/f'training{s}.json').read_text())['n_iter'] for s in spec['seeds']},
              aggregation='equal per-era Gaussian ranks',native_task_type='GPU')
    c.r.write_json(directory/'training.json',info)
    files=['predictions.parquet','training.json']
    for seed in spec['seeds']:files.extend([f'model{seed}.cbm',f'prediction{seed}.parquet',f'training{seed}.json',f'seed{seed}.done.json'])
    c.r.write_json(done,dict(plan_sha256=c.PLAN,block=block,recipe=spec['name'],family='catboost',
                           seconds=sum(r['seconds'] for r in records),
                           serialized_replay_max_error=max(r['serialized_replay_max_error'] for r in records),
                           files={f:c.r.sha(directory/f) for f in files}))


def train(confirmation=False):
    from catboost.utils import get_gpu_device_count
    if get_gpu_device_count()<1:raise RuntimeError('GPU ranking campaign requires visible CUDA')
    if not confirmation:copy_controls()
    needed={s['name'] for s in c.CFG['recipes']}
    if confirmation:
        chosen=c.checked_json(c.RUN/'selection.json')['selected']['name'];needed=b.dependencies(chosen)|{'baseline'}
    for block in c.blocks(c.windows(confirmation),c.CFG['cadence'],c.CFG['gap']):
        for spec in c.CFG['recipes']:
            if spec['name'] not in needed:continue
            if spec['name'] in c.CFG['gpu_recipes']:gpu_job(block,spec)
            elif confirmation:log.info('%s',c.fit_job(block,spec))
    if confirmation:
        for w in c.windows(True):
            log.info('%s',c.fit_job({**w,'name':f'static_{w["name"]}','train_end':w['test_start']-c.CFG['gap']},c.CFG['recipes'][0]))
    c.r.write_json(c.RUN/('confirmation-training.json' if confirmation else 'discovery-training.json'),dict(plan_sha256=c.PLAN,status='done'))


def rehearse():
    rng=np.random.default_rng(17);era=np.repeat(np.arange(2800,3400),6)
    X=rng.integers(0,10,(len(era),8)).astype(np.float32);y=np.tile(np.linspace(0,1,6),600)
    for spec in [s for s in c.CFG['recipes'] if s['name'] in c.CFG['gpu_recipes']]:
        spec={**spec,'cap':4,'depth':3,'checkpoint_step':2,'history':512}
        model,info=gpu_fit(spec,c.CFG['baseline'],X,y,era,seed=17,gap=40,inner_width=40,bound=c.CFG['score_bound'])
        np.testing.assert_allclose(cpu_predict(model,X[:30]),cat.predict(cat.from_bytes(cat.to_bytes(model)),X[:30]),atol=1e-12)
        log.info('synthetic GPU rehearsal passed: %s, native task %s',spec['loss'],model.get_all_params()['task_type'])


def report():
    doc=b.report()
    records=[c.checked_json(p) for p in c.RUN.glob('*/*/done.json')]
    doc['training']['copied_checkpoints']=sum('copied_from' in r for r in records)
    doc['training']['new_gpu_seed_models']=sum(len(s['seeds']) for s in c.CFG['recipes'] if s['name'] in c.CFG['gpu_recipes'])*len(c.blocks(c.CFG['discovery'],c.CFG['cadence'],c.CFG['gap']))
    c.r.write_json(REPO/'handoffs/results'/f'{c.CFG["name"]}-metrics.json',doc)
    path=REPO/'handoffs/results'/f'{c.CFG["name"]}-results.md'
    path.write_text(path.read_text().replace('# Robust temporal learning 11','# GPU ranking and model diversity 13',1)+
                    f'\nCopied source-verified checkpoints: {doc["training"]["copied_checkpoints"]}; newly fitted GPU seed models: {doc["training"]["new_gpu_seed_models"]}.\n')
    return doc


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',required=True,choices=['init','rehearse','train','select','confirm','report'])
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    b.wire();c.load=load;c.train=train;c.fit=cpu_fit
    load()
    if args.stage=='train':train()
    elif args.stage=='rehearse':rehearse()
    elif args.stage=='select':print(json.dumps(b.select(),indent=2))
    elif args.stage=='confirm':print(json.dumps(c.confirm(),indent=2))
    elif args.stage=='report':report()


if __name__=='__main__':main()
