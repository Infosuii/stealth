#!/usr/bin/env python3
"""Fixed objective/stopping research, with source-bound checkpoints and no platform access."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
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
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import research_run as r
from stealth import bundle, cv
from stealth.era_boost import fit
from stealth.models import lgbm

log = logging.getLogger('research-objectives')


def load():
    """Bind the additional runner and fitter as well as all shared research source."""
    r.CONFIG = yaml.safe_load((REPO / 'configs/research-objectives.yaml').read_text())
    r.RUN = REPO / 'artifacts' / r.CONFIG['name']
    r.RUN.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), REPO/'scripts/research_run.py', REPO/'stealth/era_boost.py',
             REPO/'stealth/models/lgbm.py', REPO/'stealth/metrics.py', REPO/'stealth/cv.py', REPO/'stealth/_runtime.py']
    source = REPO/'data/futures_train.parquet'
    plan = {'protocol':'era-objective-v1', 'config':r.CONFIG, 'data_sha256':r.sha(source),
            'code_sha256':{str(p.relative_to(REPO)):r.sha(p) for p in paths}}
    r.PLAN_SHA = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan['plan_sha256'] = r.PLAN_SHA
    path = r.RUN/'plan.json'
    if path.exists() and json.loads(path.read_text())['plan_sha256'] != r.PLAN_SHA:
        raise RuntimeError('research specification changed; do not reuse this run ID')
    if not path.exists():
        r.write_json(path, plan)
    r.FRAME = pd.read_parquet(source, filters=[('exped','<','exped_5925')])
    r.FRAME['era'] = r.FRAME.exped.str.rsplit('_',n=1).str[-1].astype(np.int32)
    r.FRAME = r.FRAME.loc[r.FRAME.era >= r.CONFIG['train_min_era']].copy()
    if r.FRAME.era.max() >= r.CONFIG['excluded_from_research'][0]:
        raise RuntimeError('old audit reached the research data loader')
    log.info('loaded %d rows; old audit excluded; plan %s', len(r.FRAME), r.PLAN_SHA)


def fit_models(spec, X, y, era):
    models, counts = [], []
    for seed in spec['seeds']:
        model, count = fit(spec, r.CONFIG['baseline'], X, y, era, seed=seed, threads=r.CONFIG['threads'])
        models.append(model)
        counts.append(count)
    return models, counts


def fit_job(window, spec):
    if spec['name'] == 'baseline':
        return r.fit_job(window, spec)
    if r.cached(window, spec['name']):
        return f'{window["name"]}/{spec["name"]}: reused'
    start = time.perf_counter()
    with threadpool_limits(limits=r.CONFIG['threads']):
        _, tr, te, X, Xt = r.arrays(window)
        models, counts = fit_models(spec, X, tr.target_everest.to_numpy(), tr.era.to_numpy())
        prediction = np.mean([lgbm.predict(m,Xt) for m in models],axis=0)
        seconds = time.perf_counter()-start
        r.save_fit(window,spec['name'],'lgbm',[lgbm.to_bytes(m) for m in models],prediction,te,seconds,{'counts':counts})
    return f'{window["name"]}/{spec["name"]}: done in {seconds:.1f}s; own-inner counts {counts}'


def discovery():
    with ProcessPoolExecutor(max_workers=r.CONFIG['workers'],mp_context=multiprocessing.get_context('fork')) as pool:
        jobs = [pool.submit(fit_job,w,s) for w in r.CONFIG['discovery']
                for s in [{'name':'baseline'},*r.CONFIG['candidates']]]
        for job in as_completed(jobs):
            log.info('%s',job.result())
    r.write_json(r.RUN/'cpu-discovery.json',{'status':'done','plan_sha256':r.PLAN_SHA})


def confirm():
    selected = r.summarise()['selected']
    path = r.RUN/'confirmation.json'
    if path.exists():
        return json.loads(path.read_text())
    if selected is None:
        result = {'promoted':False,'reason':'no candidate passed discovery gates; baseline retained','plan_sha256':r.PLAN_SHA}
    else:
        w = r.CONFIG['confirmation']
        spec = next(s for s in r.CONFIG['candidates'] if s['name']==selected['name'])
        fit_job(w,{'name':'baseline'})
        fit_job(w,spec)
        score,base = r.comparison(w,spec['name'],selected['blend_with_baseline'])
        paired = cv.paired_block_bootstrap(score-base,**r.CONFIG['bootstrap']).as_dict()
        passed = paired['inference_valid'] and paired['diff_mean']>0 and paired['p_pos']>=r.CONFIG['promotion']['p_pos']
        result = {'plan_sha256':r.PLAN_SHA,'selected':spec['name'],'blend_with_baseline':selected['blend_with_baseline'],
                  'fit_mean':float(score.mean()),'baseline_mean':float(base.mean()),'paired':paired,'promoted':bool(passed),
                  'label':'historically viewed confirmation; not independent future evidence',
                  'reason':'confirmation passed' if passed else 'confirmation failed; baseline retained'}
    r.write_json(path,result)
    return result


def package():
    confirmation = confirm()
    if not confirmation['promoted']:
        log.info('no candidate promoted; baseline retained')
        return None
    spec = next(s for s in r.CONFIG['candidates'] if s['name']==confirmation['selected'])
    columns = [c for c in r.FRAME if c.startswith('feature_')]
    X = r.FRAME[columns].to_numpy(dtype=np.float64,copy=True)
    X[X<0]=np.nan
    y,eras = r.FRAME.target_everest.to_numpy(),r.FRAME.era.to_numpy()
    with threadpool_limits(limits=r.CONFIG['threads']):
        models,counts = fit_models(spec,X,y,eras)
        members = [{'name':spec['name'],'family':'lgbm','input':'nan','feature_cols':columns,
                    'target':'target_everest','arm':'objective','group':'everest','blobs':[lgbm.to_bytes(m) for m in models]}]
        weights = {spec['name']:1.0}
        if confirmation['blend_with_baseline']:
            base,_ = r.baseline_fit(None,X,y,eras)
            members.append({'name':'baseline','family':'lgbm','input':'nan','feature_cols':columns,
                            'target':'target_everest','arm':'baseline','group':'everest','blobs':[lgbm.to_bytes(base)]})
            weights = {spec['name']:.5,'baseline':.5}
    b = {'version':'objective-v1','git_sha':bundle.git_sha(),'built_at':datetime.now(timezone.utc).isoformat(),
         'run_id':r.CONFIG['name'],'feature_cols':columns,'missing_cols':[],'members':members,'weights':weights,
         'p_b':0.0,'p_c':0.0,'p_source':'override','p_reason':'raw frozen objective research; no audit tuning',
         'core_proxy_cols':[],'min_bench_coverage':.5,'blend_rule':'fixed_equal' if len(members)>1 else 'single',
         'admitted_arms':[m['arm'] for m in members],'artifact_source':'research_refit','weights_source':'frozen research decision',
         'blob_files':{},'source_run_fingerprint':r.PLAN_SHA,'source_universe':f'e{r.CONFIG["train_min_era"]}-none-d0'}
    from stealth._runtime import make_predict
    make_predict(b)
    path = REPO/'models'/f'{r.CONFIG["name"]}.pkl'
    digest = bundle.write_pickle(b,path)
    bundle.write_manifest(path,b,digest)
    r.write_json(r.RUN/'packaging.json',{'sha256':digest,'counts':counts,'plan_sha256':r.PLAN_SHA,
                 'selection_sha256':r.sha(r.RUN/'selection.json'),'confirmation_sha256':r.sha(r.RUN/'confirmation.json')})
    return digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['init','cpu','select','confirm','package','report'])
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    load()
    if args.stage=='cpu':
        discovery()
    elif args.stage=='select':
        print(json.dumps(r.summarise(),indent=2))
    elif args.stage=='confirm':
        print(json.dumps(confirm(),indent=2))
    elif args.stage=='package':
        print(package())
    elif args.stage=='report':
        r.report()


if __name__=='__main__':
    main()
