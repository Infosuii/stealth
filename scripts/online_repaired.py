#!/usr/bin/env python3
"""Missing-core technical repair with unchanged discovery and native models."""
import argparse
import hashlib
import json
import logging
import shutil

import numpy as np
import pandas as pd
import yaml

import online as old
from stealth.online_missing import predict

c=old.c;b=old.b;REPO=old.REPO


def load(confirmation=False):
    old.load()
    source=json.loads((c.RUN/'plan.json').read_text())
    c.CFG=yaml.safe_load((REPO/'configs/online-repaired.yaml').read_text())
    c.RUN=REPO/'artifacts'/c.CFG['name'];c.RUN.mkdir(parents=True,exist_ok=True)
    plan=dict(protocol='online-missing-core-technical-repair-v1',config=c.CFG,data_sha256=source['data_sha256'],
              source_plan_sha256=c.r.sha(REPO/'artifacts/online14/plan.json'),
              source_selection_sha256=c.r.sha(REPO/'artifacts/online14/selection.json'),
              code_sha256={f:c.r.sha(REPO/f) for f in ['scripts/online_repaired.py','stealth/online_missing.py','docs/ONLINE-14-REPAIR.md']})
    c.PLAN=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest();plan['plan_sha256']=c.PLAN
    path=c.RUN/'plan.json'
    if path.exists() and json.loads(path.read_text())['plan_sha256']!=c.PLAN:raise RuntimeError('repair plan changed')
    if not path.exists():c.r.write_json(path,plan)
    if confirmation:
        decision=c.checked_json(c.RUN/'selection.json')
        if decision['selected'] is None or not decision['discovery_replay_passed']:raise RuntimeError('repair replay/winner guard')
        ceiling=c.CFG['excluded_from_research'][0]
        c.FRAME=pd.read_parquet(REPO/'data/futures_train.parquet',filters=[('exped','<',f'exped_{ceiling:04d}')])
        c.FRAME['era']=c.FRAME.exped.str.rsplit('_',n=1).str[-1].astype(np.int32)
        c.FRAME=c.FRAME.loc[c.FRAME.era>=c.CFG['train_min_era']]
        if c.FRAME.era.max()>=ceiling:raise RuntimeError('repair confirmation boundary')
    old.CACHE.clear()


def copy_source():
    root=REPO/'artifacts/online14';plan=json.loads((root/'plan.json').read_text())
    for key,value in plan['config'].items():
        if key!='name' and c.CFG[key]!=value:raise RuntimeError('repair altered original specification')
    for path in root.glob('*/*/done.json'):
        doc=json.loads(path.read_text())
        if doc['plan_sha256']!=plan['plan_sha256']:raise RuntimeError('repair source native plan')
        for f,digest in doc['files'].items():
            if c.r.sha(path.parent/f)!=digest:raise RuntimeError('repair source native bytes')
        dest=c.RUN/path.parent.parent.name/path.parent.name;dest.mkdir(parents=True,exist_ok=True)
        marker=dest/'done.json'
        if marker.exists():c.checked_json(marker);continue
        for f in doc['files']:shutil.copyfile(path.parent/f,dest/f)
        c.r.write_json(marker,{**doc,'plan_sha256':c.PLAN,'seconds':0.,'copied_from':'online14','source_done_sha256':c.r.sha(path)})


def select():
    path=c.RUN/'selection.json'
    if path.exists():return c.checked_json(path)
    doc=old.select()
    source=json.loads((REPO/'artifacts/online14/selection.json').read_text())
    if doc['selected']['name']!=source['selected']['name']:raise RuntimeError('repair changed frozen winner')
    max_error=0.
    for name in c.CFG['comparisons']:
        a=pd.read_parquet(c.RUN/f'scores_{name}_0.parquet')
        b0=pd.read_parquet(REPO/f'artifacts/online14/scores_{name}_0.parquet').reindex(a.index)
        np.testing.assert_allclose(a.to_numpy(),b0.to_numpy(),atol=1e-12,rtol=1e-12)
        max_error=max(max_error,float(np.max(np.abs(a.to_numpy()-b0.to_numpy()))))
    for name in ('adaptive_fit','adaptive_score'):
        a=c.checked_json(c.RUN/f'{name}_discovery_trace.json')['trace']
        prior=json.loads((REPO/f'artifacts/online14/{name}_discovery_trace.json').read_text())['trace']
        np.testing.assert_allclose([x['weights'] for x in a],[x['weights'] for x in prior],atol=1e-12,rtol=1e-12)
    doc['discovery_replay_passed']=True;doc['discovery_score_replay_max_error']=max_error
    c.r.write_json(path,doc)
    return doc


def train(confirmation=False):
    if confirmation:old.train(True)  # every native checkpoint was copied and must be reused byte-for-byte
    else:copy_source()


def report():
    decision=select();confirmation=c.checked_json(c.RUN/'confirmation.json')
    out=REPO/'handoffs/results'
    doc=dict(run_id='online14r',label='technical missing-core repair; historically viewed, locked confirmation',
             selection=decision,confirmation=confirmation)
    c.r.write_json(out/'online14r-metrics.json',doc)
    lines=['# Online calibration missing-core repair','',doc['label'],'',
           f'Discovery replay max error: {decision["discovery_score_replay_max_error"]:g}; frozen winner unchanged.',
           '',f'Confirmation: {json.dumps(confirmation,sort_keys=True)}','',
           'All native models and original gates were preserved. Only missing-core calibration semantics changed.',
           'No confirmation performance verdict existed before the repair. Baseline01 remains unchanged.',
           'No platform uploads, submissions, staking or deployment changes.']
    (out/'online14r-results.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',required=True,choices=['init','train','select','confirm','report'])
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    b.wire();old.online_predict=predict;c.load=load;c.select=select;c.train=train;c.fit=old.parent.cpu_fit;c.candidate_prediction=old.prediction
    load()
    if args.stage=='train':train()
    elif args.stage=='select':print(json.dumps(select(),indent=2))
    elif args.stage=='confirm':print(json.dumps(c.confirm(),indent=2))
    elif args.stage=='report':report()


if __name__=='__main__':main()
