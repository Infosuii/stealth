#!/usr/bin/env python3
"""Source-bound, nested model optimization. No platform client or old-audit scoring."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
import logging
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import research_run as r
from stealth import bundle,cv
from stealth._runtime import basis_forward,make_predict
from stealth.campaign_models import fit_calibrated,fit_trees
from stealth.research_models import to_bytes

log=logging.getLogger('campaign')


def load(phase='discovery',*,config_path=None,additional_sources=()):
    r.CONFIG=yaml.safe_load(Path(config_path or REPO/'configs/campaign.yaml').read_text())
    r.RUN=REPO/'artifacts'/r.CONFIG['name']
    r.RUN.mkdir(parents=True,exist_ok=True)
    files=['scripts/campaign.py','scripts/campaign_gpu.py','scripts/research_run.py','scripts/research_gpu.py',
           'stealth/campaign_models.py','stealth/research_models.py','stealth/models/lgbm.py','stealth/era_boost.py',
           'stealth/metrics.py','stealth/cv.py','stealth/_runtime.py','stealth/bundle.py',*additional_sources]
    source=REPO/'data/futures_train.parquet'
    plan={'protocol':'sustained-campaign-v1','config':r.CONFIG,'data_sha256':r.sha(source),
          'code_sha256':{f:r.sha(REPO/f) for f in files}}
    r.PLAN_SHA=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    plan['plan_sha256']=r.PLAN_SHA
    path=r.RUN/'plan.json'
    if path.exists() and json.loads(path.read_text())['plan_sha256']!=r.PLAN_SHA:
        raise RuntimeError('campaign source/data/spec changed; a new run ID is required')
    if not path.exists():
        r.write_json(path,plan)
    ceiling=r.CONFIG['confirmation']['train_end'] if phase=='discovery' else r.CONFIG['excluded_from_research'][0]
    r.FRAME=pd.read_parquet(source,filters=[('exped','<',f'exped_{ceiling:04d}')])
    r.FRAME['era']=r.FRAME.exped.str.rsplit('_',n=1).str[-1].astype(np.int32)
    r.FRAME=r.FRAME.loc[r.FRAME.era>=r.CONFIG['train_min_era']].copy()
    if r.FRAME.era.max()>=ceiling:
        raise RuntimeError('excluded eras reached campaign reader')
    log.info('%s reader loaded %d rows, eras %d..%d; plan %s',phase,len(r.FRAME),r.FRAME.era.min(),r.FRAME.era.max(),r.PLAN_SHA)


def fitted(spec,X,y,era,columns,directory):
    if spec['kind']=='calibrated':
        model,info=fit_calibrated(spec,X,y,era,columns,r.CONFIG['inner'])
        return 'basis',[to_bytes(model)],lambda xx,eras=None:basis_forward(model,xx),info
    if spec['kind']=='tuned_trees':
        from stealth.models import lgbm
        models,info=fit_trees(spec,X,y,era,r.CONFIG['inner'],seed=r.CONFIG['seed'],threads=r.CONFIG['threads'],storage=directory/'search.sqlite')
        return 'lgbm',[lgbm.to_bytes(m) for m in models],lambda xx,eras=None:np.mean([lgbm.predict(m,xx) for m in models],axis=0),info
    if spec['kind']=='latent_tree':
        from stealth.representation_models import fit_latent,grouped_prediction
        models,info=fit_latent(spec,X,y,era,columns,r.CONFIG['inner'],seed=r.CONFIG['seed'],threads=r.CONFIG['threads'],storage=directory/'search.sqlite')
        return 'basis',[to_bytes(m) for m in models],lambda xx,eras:grouped_prediction(models,xx,eras),info
    if spec['kind']=='stable_signal':
        from stealth.stability_models import fit_stability
        model,info=fit_stability(spec,X,y,era,columns,r.CONFIG['inner'])
        return 'basis',[to_bytes(model)],lambda xx,eras=None:basis_forward(model,xx),info
    raise ValueError('external GPU recipe requires campaign_gpu.py')


def fit_job(window,spec):
    if spec['name']=='baseline':
        return r.fit_job(window,spec)
    if r.cached(window,spec['name']):
        return f'{window["name"]}/{spec["name"]}: reused'
    from threadpoolctl import threadpool_limits
    start=time.perf_counter()
    directory=r.cache_dir(window,spec['name'])
    directory.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=r.CONFIG['threads']):
        columns,tr,te,X,Xt=r.arrays(window)
        family,blobs,predict,info=fitted(spec,X,tr.target_everest.to_numpy(),tr.era.to_numpy(),columns,directory)
        prediction=predict(Xt,te.era.to_numpy())
        r.write_json(directory/'training.json',info)
        r.save_fit(window,spec['name'],family,blobs,prediction,te,time.perf_counter()-start,{'training_sha256':r.sha(directory/'training.json')})
    return f'{window["name"]}/{spec["name"]}: done in {time.perf_counter()-start:.1f}s'


def cpu():
    jobs=[(w,s) for w in r.CONFIG['discovery'] for s in [{'name':'baseline'},*r.CONFIG['candidates']] if s.get('kind') not in ('long_net','denoised')]
    with ProcessPoolExecutor(max_workers=r.CONFIG['workers'],mp_context=multiprocessing.get_context('fork')) as pool:
        futures=[pool.submit(fit_job,w,s) for w,s in jobs]
        for future in as_completed(futures):
            log.info('%s',future.result())
    r.write_json(r.RUN/'cpu-discovery.json',{'status':'done','plan_sha256':r.PLAN_SHA})


def selection():
    # Shared eight-comparison Holm/effect/stability gates. No outer test drives an inner choice.
    return r.summarise()


def confirm():
    selected=selection()['selected']
    path=r.RUN/'confirmation.json'
    if path.exists():
        return json.loads(path.read_text())
    if selected is None:
        result={'promoted':False,'major_improvement':False,'reason':'no eligible discovery challenger','plan_sha256':r.PLAN_SHA}
    else:
        window=r.CONFIG['confirmation']
        spec=next(s for s in r.CONFIG['candidates'] if s['name']==selected['name'])
        fit_job(window,{'name':'baseline'})
        if spec['kind'] in ('long_net','denoised'):
            if not r.cached(window,spec['name']):
                raise RuntimeError('train the frozen neural winner with campaign_gpu.py --confirmation first')
        else:
            fit_job(window,spec)
        score,base=r.comparison(window,spec['name'],selected['blend_with_baseline'])
        paired=cv.paired_block_bootstrap(score-base,**r.CONFIG['bootstrap']).as_dict()
        passed=paired['inference_valid'] and paired['diff_mean']>0 and paired['p_pos']>=r.CONFIG['promotion']['p_pos']
        relative=float(paired['diff_mean']/base.mean())
        discovery_relative=selected['paired']['diff_mean']/selected['baseline_mean']
        result={'selected':spec['name'],'blend_with_baseline':selected['blend_with_baseline'],'promoted':bool(passed),
                'fit_mean':float(score.mean()),'baseline_mean':float(base.mean()),'paired':paired,'relative_gain':relative,
                'major_improvement':bool(passed and relative>=r.CONFIG['major_relative_gain'] and discovery_relative>=r.CONFIG['major_relative_gain']),
                'label':'historically viewed, round-locked confirmation; not independent future evidence','plan_sha256':r.PLAN_SHA}
    r.write_json(path,result)
    return result


def package():
    doc=confirm()
    if not doc['promoted']:
        return None
    from threadpoolctl import threadpool_limits
    from stealth.models import lgbm
    spec=next(s for s in r.CONFIG['candidates'] if s['name']==doc['selected'])
    columns=[c for c in r.FRAME if c.startswith('feature_')]
    X=r.FRAME[columns].to_numpy(dtype=float,copy=True)
    X[X<0]=np.nan
    full=r.RUN/'full'/spec['name']
    full.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=r.CONFIG['threads']):
        if spec['kind'] in ('long_net','denoised'):
            record=json.loads((full/'done.json').read_text())
            if record['plan_sha256']!=r.PLAN_SHA:
                raise RuntimeError('full neural fit has another plan')
            for f,digest in record['files'].items():
                if r.sha(full/f)!=digest:
                    raise RuntimeError('corrupt full neural checkpoint')
            family='basis'
            blobs=[(full/f).read_bytes() for f in record['files'] if f.startswith('model')]
        else:
            family,blobs,_,info=fitted(spec,X,r.FRAME.target_everest.to_numpy(),r.FRAME.era.to_numpy(),columns,full)
            r.write_json(full/'training.json',info)
        members=[{'name':spec['name'],'family':family,'input':'nan','feature_cols':columns,'blobs':blobs,
                  'target':'target_everest','arm':'campaign','group':'everest'}]
        weights={spec['name']:1.}
        if doc['blend_with_baseline']:
            base,_=r.baseline_fit(None,X,r.FRAME.target_everest.to_numpy(),r.FRAME.era.to_numpy())
            members.append({'name':'baseline','family':'lgbm','input':'nan','feature_cols':columns,'blobs':[lgbm.to_bytes(base)],
                            'target':'target_everest','arm':'baseline','group':'everest'})
            weights={spec['name']:.5,'baseline':.5}
    b={'version':'campaign-v1','run_id':r.CONFIG['name'],'git_sha':bundle.git_sha(),'built_at':datetime.now(timezone.utc).isoformat(),
       'feature_cols':columns,'missing_cols':[],'members':members,'weights':weights,'p_b':0.,'p_c':0.,'p_source':'override',
       'p_reason':'raw frozen campaign; no audit tuning','core_proxy_cols':[],'min_bench_coverage':.5,'blob_files':{},
       'admitted_arms':[m['arm'] for m in members],'blend_rule':'fixed_equal' if len(members)>1 else 'single',
       'artifact_source':'research_refit','source_run_fingerprint':r.PLAN_SHA,'source_universe':'e2800-none-d0',
       'weights_source':'frozen campaign decision'}
    make_predict(b)
    path=REPO/'models'/f'{r.CONFIG["name"]}.pkl'
    digest=bundle.write_pickle(b,path)
    bundle.write_manifest(path,b,digest)
    r.write_json(r.RUN/'packaging.json',{'sha256':digest,'plan_sha256':r.PLAN_SHA,'selection_sha256':r.sha(r.RUN/'selection.json'),
                                     'confirmation_sha256':r.sha(r.RUN/'confirmation.json')})
    return digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['init','cpu','select','confirm','package','report'])
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    load('confirmation' if args.stage in ('confirm','package') else 'discovery')
    if args.stage=='cpu':cpu()
    elif args.stage=='select':print(json.dumps(selection(),indent=2))
    elif args.stage=='confirm':print(json.dumps(confirm(),indent=2))
    elif args.stage=='package':print(package())
    elif args.stage=='report':r.report()


if __name__=='__main__':main()
