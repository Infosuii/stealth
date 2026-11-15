#!/usr/bin/env python3
"""Causal OOS family-weight learning with frozen native dependencies."""
import argparse
import hashlib
import json
import logging
import shutil

import numpy as np
import pandas as pd
import yaml

import ranking as parent
from stealth.online_blend import predict as online_predict

c=parent.c;b=parent.b;REPO=parent.REPO
CACHE={}


def load(confirmation=False):
    parent.load()  # verifies the immutable original source and supplies its discovery-only reader
    source=json.loads((c.RUN/'plan.json').read_text())
    cfg=yaml.safe_load((REPO/'configs/online.yaml').read_text())
    c.CFG=cfg;c.RUN=REPO/'artifacts'/cfg['name'];c.RUN.mkdir(parents=True,exist_ok=True)
    plan=dict(protocol='prequential-family-weights-v1',config=cfg,data_sha256=source['data_sha256'],
              source_plan_sha256=c.r.sha(REPO/'artifacts/ranking13/plan.json'),versions=source['versions'],
              code_sha256={f:c.r.sha(REPO/f) for f in ['scripts/online.py','stealth/online_blend.py','docs/ONLINE-14.md']})
    c.PLAN=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest();plan['plan_sha256']=c.PLAN
    path=c.RUN/'plan.json'
    if path.exists() and json.loads(path.read_text())['plan_sha256']!=c.PLAN:raise RuntimeError('online plan changed; use a new ID')
    if not path.exists():c.r.write_json(path,plan)
    if confirmation:
        if c.checked_json(c.RUN/'selection.json')['selected'] is None:raise RuntimeError('no frozen online winner')
        ceiling=cfg['excluded_from_research'][0]
        c.FRAME=pd.read_parquet(REPO/'data/futures_train.parquet',filters=[('exped','<',f'exped_{ceiling:04d}')])
        c.FRAME['era']=c.FRAME.exped.str.rsplit('_',n=1).str[-1].astype(np.int32)
        c.FRAME=c.FRAME.loc[c.FRAME.era>=cfg['train_min_era']]
        if c.FRAME.era.max()>=ceiling:raise RuntimeError('online confirmation reader boundary')
    CACHE.clear()
    logging.info('online reader eras %d..%d; plan %s',c.FRAME.era.min(),c.FRAME.era.max(),c.PLAN)


def copy_source():
    root=REPO/'artifacts/ranking13';plan=json.loads((root/'plan.json').read_text())
    for key in ('baseline','seed','threads','gap','cadence','discovery','train_min_era'):
        if c.CFG[key]!=plan['config'][key]:raise RuntimeError('online native dependency specification changed')
    for name,group in plan['config']['ensembles'].items():
        if c.CFG['ensembles'][name]!=group:raise RuntimeError('online source group changed')
    for path in root.glob('*/*/done.json'):
        doc=json.loads(path.read_text())
        if doc['plan_sha256']!=plan['plan_sha256']:raise RuntimeError('online source plan mismatch')
        for f,digest in doc['files'].items():
            if c.r.sha(path.parent/f)!=digest:raise RuntimeError('online source bytes changed')
        dest=c.RUN/path.parent.parent.name/path.parent.name;dest.mkdir(parents=True,exist_ok=True)
        marker=dest/'done.json'
        if marker.exists():c.checked_json(marker);continue
        for f in doc['files']:shutil.copyfile(path.parent/f,dest/f)
        c.r.write_json(marker,{**doc,'plan_sha256':c.PLAN,'seconds':0.,'copied_from':'ranking13',
                              'source_done_sha256':c.r.sha(path)})


def prediction(name,base,confirmation=False):
    if name not in ('adaptive_fit','adaptive_score'):return b.prediction(name,base,confirmation)
    key=(name,confirmation)
    if key not in CACHE:
        past=c.predictions('baseline')
        bases=[past,base] if confirmation else [base]
        tables=[]
        for i,bb in enumerate(bases):
            phase=confirmation and i==1
            pp=np.column_stack([c.rank_by_era(b.prediction(n,bb,phase),bb.era) for n in c.CFG['online']['components']])
            tables.append(pd.DataFrame(pp,index=bb.index))
        all_base=pd.concat(bases);P=pd.concat(tables).reindex(all_base.index).to_numpy()
        frame=c.FRAME.loc[all_base.index]
        core=c.core_mean(frame[[x for x in frame if x.startswith('feature_')]].to_numpy())
        blocks=c.blocks(c.CFG['discovery']+([c.CFG['confirmation']] if confirmation else []),c.CFG['cadence'],c.CFG['gap'])
        pred,trace=online_predict(P,frame.target_everest.to_numpy(),all_base.era.to_numpy(),core,blocks,c.CFG['online'],name.split('_')[1])
        c.r.write_json(c.RUN/f'{name}_{"confirmation" if confirmation else "discovery"}_trace.json',dict(plan_sha256=c.PLAN,trace=trace))
        CACHE[key]=pd.Series(pred,index=all_base.index)
    return CACHE[key].reindex(base.index).to_numpy()


def select():
    path=c.RUN/'selection.json'
    if path.exists():return c.checked_json(path)
    doc=b.select()
    fixed=next(row for row in doc['comparisons'] if row['name']=='fixed_prior')
    for row in doc['comparisons']:
        row['gain_vs_fixed']={k:row['metrics'][k]-fixed['metrics'][k] for k in ('fit','score')}
        row['eligible']=bool(row['eligible'] and (row['name']=='fixed_prior' or min(row['gain_vs_fixed'].values())>0))
    eligible=sorted([row for row in doc['comparisons'] if row['eligible']],key=lambda row:row['paired']['score']['diff_mean'],reverse=True)
    doc['selected']=eligible[0] if eligible else None
    c.r.write_json(path,doc)
    return doc


def train(confirmation=False):
    if confirmation:parent.train(True)
    else:copy_source()


def report():
    selection=select();confirmation=c.checked_json(c.RUN/'confirmation.json')
    traces={n:c.checked_json(c.RUN/f'{n}_discovery_trace.json')['trace'] for n in ('adaptive_fit','adaptive_score')}
    doc=dict(run_id=c.CFG['name'],label='adaptive historical OOS weight learning; composite metrics use proxies',
             selection=selection,confirmation=confirmation,
             weight_summary={n:{w['name']:np.mean([r['weights'] for r in trace if w['test_start']<=r['start']<=w['test_end']],axis=0).tolist()
                                for w in c.CFG['discovery']} for n,trace in traces.items()})
    out=REPO/'handoffs/results';c.r.write_json(out/'online14-metrics.json',doc)
    lines=['# Prequential model-family weighting 14','',doc['label'],'',
           '| policy | FIT | proxy score | FIT gain | score gain | Holm | eligible |','|---|---|---|---|---|---|---|']
    for row in selection['comparisons']:
        lines.append(f'| {row["name"]} | {row["metrics"]["fit"]:.8f} | {row["metrics"]["score"]:.8f} | '
                     f'{row["paired"]["fit"]["diff_mean"]:+.8f} | {row["paired"]["score"]["diff_mean"]:+.8f} | {row["holm_pass"]} | {row["eligible"]} |')
    lines+=['',f'Confirmation: {json.dumps(confirmation,sort_keys=True)}','',
            'Native discovery models were immutable source-verified ranking13 dependencies; none was retrained.',
            'Every weight fit excluded labels at or after block_start minus 40. Private traces retain exact boundaries and weights.',
            'Baseline01 remains unchanged. No uploads, submissions, staking or deployment changes.']
    (out/'online14-results.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',required=True,choices=['init','train','select','confirm','report'])
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    b.wire();c.load=load;c.train=train;c.fit=parent.cpu_fit;c.candidate_prediction=prediction;c.select=select
    load()
    if args.stage=='train':train()
    elif args.stage=='select':print(json.dumps(select(),indent=2))
    elif args.stage=='confirm':print(json.dumps(c.confirm(),indent=2))
    elif args.stage=='report':report()


if __name__=='__main__':main()
