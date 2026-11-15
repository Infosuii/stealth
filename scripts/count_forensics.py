#!/usr/bin/env python3
"""Source-bound matched stopping-count controls, sharing frozen temporal evaluation."""
import argparse
import hashlib
import importlib.metadata
import json
import logging

import numpy as np
import pandas as pd
import yaml

import robust_temporal as b
from stealth.count_transfer import fit

c = b.c
REPO = b.REPO


def load(confirmation=False):
    c.CFG = yaml.safe_load((REPO/'configs/count-forensics.yaml').read_text())
    c.RUN = REPO/'artifacts'/c.CFG['name'];c.RUN.mkdir(parents=True,exist_ok=True)
    files = ['scripts/count_forensics.py','scripts/robust_temporal.py','scripts/temporal.py','scripts/research_run.py',
             'stealth/count_transfer.py','stealth/robust_temporal.py','stealth/temporal_models.py',
             'stealth/metrics.py','stealth/cv.py','stealth/models/lgbm.py','docs/COUNT-12.md']
    plan = dict(protocol='count-transfer-v1',config=c.CFG,data_sha256=c.r.sha(REPO/'data/futures_train.parquet'),
                code_sha256={f:c.r.sha(REPO/f) for f in files},
                control_sha256={r:c.r.sha(REPO/f'artifacts/{r}/plan.json') for r in ('temporal10','temporal11')},
                versions={p:importlib.metadata.version(p) for p in ('numpy','pandas','lightgbm','scipy')})
    c.PLAN = hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest();plan['plan_sha256']=c.PLAN
    path = c.RUN/'plan.json'
    if path.exists() and json.loads(path.read_text())['plan_sha256']!=c.PLAN:raise RuntimeError('count plan changed; use a new ID')
    if not path.exists():c.r.write_json(path,plan)
    ceiling = c.CFG['reader_ceiling']
    if confirmation:
        if c.checked_json(c.RUN/'selection.json')['selected'] is None:raise RuntimeError('no frozen count winner')
        ceiling=c.CFG['excluded_from_research'][0]
    c.FRAME=pd.read_parquet(REPO/'data/futures_train.parquet',filters=[('exped','>=','exped_2400'),('exped','<',f'exped_{ceiling:04d}')])
    c.FRAME['era']=c.FRAME.exped.str.rsplit('_',n=1).str[-1].astype(np.int32)
    if c.FRAME.era.min()<2400 or c.FRAME.era.max()>=ceiling or not c.FRAME.index.is_unique:raise RuntimeError('count reader boundary')
    logging.info('count reader loaded eras %d..%d; plan %s',c.FRAME.era.min(),c.FRAME.era.max(),c.PLAN)


def ragged_control():
    root=REPO/'artifacts/temporal11';plan=json.loads((root/'plan.json').read_text())
    pieces=[]
    for block in c.blocks(c.CFG['discovery'],c.CFG['cadence'],c.CFG['gap']):
        d=root/block['name']/'score_multistop';doc=json.loads((d/'done.json').read_text())
        if doc['plan_sha256']!=plan['plan_sha256']:raise RuntimeError('ragged source plan')
        for f,digest in doc['files'].items():
            if c.r.sha(d/f)!=digest:raise RuntimeError('ragged source bytes')
        pieces.append(pd.read_parquet(d/'predictions.parquet'))
    new=c.predictions('ragged_control');old=pd.concat(pieces).reindex(new.index)
    if not np.array_equal(old.era,new.era):raise RuntimeError('ragged rows differ')
    np.testing.assert_allclose(old.prediction,new.prediction,atol=1e-12,rtol=1e-12)
    return float(np.abs(old.prediction-new.prediction).max())


def select():
    path=c.RUN/'selection.json'
    if path.exists():return c.checked_json(path)
    replay=ragged_control()
    doc=b.select();doc['ragged_replay_max_error']=replay;c.r.write_json(path,doc)
    return doc


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',required=True,choices=['init','train','select','confirm','report'])
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    b.wire();c.load=load;c.fit=fit;c.select=select
    load()
    if args.stage=='train':b.train()
    elif args.stage=='select':print(json.dumps(select(),indent=2))
    elif args.stage=='confirm':print(json.dumps(c.confirm(),indent=2))
    elif args.stage=='report':
        select();b.report()
        p=REPO/'handoffs/results'/f'{c.CFG["name"]}-results.md'
        p.write_text(p.read_text().replace('# Robust temporal learning 11','# Count-transfer controls 12',1))


if __name__=='__main__':main()
