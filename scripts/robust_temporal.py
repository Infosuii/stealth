#!/usr/bin/env python3
"""Fixed group consensus, history diversity and multi-window stopping; local-only research."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import logging
import multiprocessing

import numpy as np
import pandas as pd
import yaml

import temporal as c
from stealth.robust_temporal import fit

REPO = c.REPO
log = logging.getLogger('robust_temporal')


def load(confirmation=False):
    c.CFG = yaml.safe_load((REPO/'configs/robust-temporal.yaml').read_text())
    c.RUN = REPO/'artifacts'/c.CFG['name']
    c.RUN.mkdir(parents=True,exist_ok=True)
    sources = ['scripts/robust_temporal.py','scripts/temporal.py','scripts/research_run.py',
               'stealth/robust_temporal.py','stealth/temporal_models.py','stealth/metrics.py',
               'stealth/cv.py','stealth/models/lgbm.py','docs/TEMPORAL-11.md']
    plan = dict(protocol='robust-temporal-v1',config=c.CFG,data_sha256=c.r.sha(REPO/'data/futures_train.parquet'),
                code_sha256={f:c.r.sha(REPO/f) for f in sources},
                versions={p:importlib.metadata.version(p) for p in ('numpy','pandas','lightgbm','scipy')},
                control_plan_sha256=c.r.sha(REPO/'artifacts/temporal10/plan.json'))
    c.PLAN = hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    plan['plan_sha256'] = c.PLAN
    path = c.RUN/'plan.json'
    if path.exists() and json.loads(path.read_text())['plan_sha256']!=c.PLAN:
        raise RuntimeError('robust temporal source/data/config changed; use a new run ID')
    if not path.exists():c.r.write_json(path,plan)
    ceiling = c.CFG['reader_ceiling']
    if confirmation:
        if c.checked_json(c.RUN/'selection.json')['selected'] is None:
            raise RuntimeError('no frozen robust temporal winner')
        ceiling = c.CFG['excluded_from_research'][0]
    c.FRAME = pd.read_parquet(REPO/'data/futures_train.parquet',filters=[('exped','<',f'exped_{ceiling:04d}')])
    c.FRAME['era'] = c.FRAME.exped.str.rsplit('_',n=1).str[-1].astype(np.int32)
    c.FRAME = c.FRAME.loc[c.FRAME.era>=c.CFG['train_min_era']].copy()
    if c.FRAME.era.max()>=ceiling or not c.FRAME.index.is_unique:
        raise RuntimeError('reader boundary/index violation')
    log.info('loaded eras %d..%d; plan %s',c.FRAME.era.min(),c.FRAME.era.max(),c.PLAN)


def dependencies(name):
    if name not in c.CFG['ensembles']:return {name}
    return set().union(*(dependencies(n) for n in c.CFG['ensembles'][name]))


def prediction(name,base,confirmation=False):
    if name not in c.CFG['ensembles']:
        return c.rank_by_era(c.predictions(name,confirmation).reindex(base.index).prediction,base.era)
    group = c.CFG['ensembles'][name]
    return sum(w*c.rank_by_era(prediction(n,base,confirmation),base.era) for n,w in group.items())/sum(group.values())


def train(confirmation=False):
    specs = c.CFG['recipes']
    if confirmation:
        chosen = c.checked_json(c.RUN/'selection.json')['selected']['name']
        needed = dependencies(chosen)|{'baseline'}
        specs = [s for s in specs if s['name'] in needed]
    jobs = [(b,s) for b in c.blocks(c.windows(confirmation),c.CFG['cadence'],c.CFG['gap']) for s in specs]
    for w in c.windows(confirmation):
        jobs.append(({**w,'name':f'static_{w["name"]}','train_end':w['test_start']-c.CFG['gap']},c.CFG['recipes'][0]))
    with ProcessPoolExecutor(max_workers=c.CFG['workers'],mp_context=multiprocessing.get_context('fork')) as pool:
        pending = [pool.submit(c.fit_job,*job) for job in jobs]
        for f in as_completed(pending):log.info('%s',f.result())
    c.r.write_json(c.RUN/('confirmation-training.json' if confirmation else 'discovery-training.json'),
                   dict(plan_sha256=c.PLAN,jobs=len(jobs)))


def check_control():
    source = REPO/'artifacts/temporal10'
    plan = json.loads((source/'plan.json').read_text())
    current = json.loads((c.RUN/'plan.json').read_text())
    if plan['data_sha256']!=current['data_sha256']:
        raise RuntimeError('control data changed')
    for key in ('baseline','seed','threads','inner_width','gap','cadence','inner_benchmark_trees'):
        if plan['config'][key]!=c.CFG[key]:raise RuntimeError(f'control {key} changed')
    errors = {}
    for name in ('baseline','score_rolling'):
        new = c.predictions(name)
        parts = []
        for b in c.blocks(c.CFG['discovery'],c.CFG['cadence'],c.CFG['gap']):
            directory = source/b['name']/name
            done = json.loads((directory/'done.json').read_text())
            if done['plan_sha256']!=plan['plan_sha256']:
                raise RuntimeError('source control has another plan')
            for f,digest in done['files'].items():
                if c.r.sha(directory/f)!=digest:raise RuntimeError('source control bytes changed')
            parts.append(pd.read_parquet(directory/'predictions.parquet'))
        old = pd.concat(parts).reindex(new.index)
        if not np.array_equal(old.era,new.era):raise RuntimeError('control row/era mismatch')
        np.testing.assert_allclose(old.prediction,new.prediction,atol=1e-12,rtol=1e-12)
        errors[name] = float(np.abs(old.prediction-new.prediction).max())
    return dict(source_plan_sha256=plan['plan_sha256'],prediction_max_error=errors)


def select():
    path = c.RUN/'selection.json'
    if path.exists():return c.checked_json(path)
    control = check_control()
    inputs = c.comparison_inputs()
    rows = []
    for name in c.CFG['comparisons']:
        for proportion in c.CFG['core_neutralization']:
            row,a,b = c.result_row(name,proportion,inputs)
            rows.append(row)
            a.to_parquet(c.RUN/f'scores_{name}_{proportion:g}.parquet')
    gates = c.CFG['promotion']
    stopped = False
    for i,row in enumerate(sorted(rows,key=lambda row:row['paired']['score']['p_value'])):
        pair = row['paired']
        threshold = gates['alpha']/(len(rows)-i)
        ok = not stopped and pair['score']['inference_valid'] and pair['score']['p_value']<=threshold
        stopped = stopped or not ok
        row['holm_threshold'],row['holm_pass'] = threshold,bool(ok)
        row['eligible'] = bool(ok and pair['score']['diff_mean']>=gates['min_score_gain']
                               and pair['score']['p_pos']>=gates['p_pos']
                               and pair['fit']['diff_mean']>=gates['min_fit_gain']
                               and pair['fit']['p_pos']>=gates['p_pos']
                               and min(row['scenario_score_gains'].values())>=0
                               and all(min(v[k] for v in row['windows'].values())>=gates['worst_window_gain'] for k in ('fit','score'))
                               and sum(v['score']>0 for v in row['windows'].values())>=gates['min_positive_windows'])
    eligible = sorted([row for row in rows if row['eligible']],key=lambda row:row['paired']['score']['diff_mean'],reverse=True)
    doc = dict(plan_sha256=c.PLAN,label='adaptive historical selection estimates; all composite metrics use proxies',
               comparisons=rows,selected=eligible[0] if eligible else None,control_replay=control,confirmation_scored=False)
    c.r.write_json(path,doc)
    return doc


def report():
    decision,confirmation = select(),c.checked_json(c.RUN/'confirmation.json')
    inputs = c.comparison_inputs()
    components = []
    for spec in c.CFG['recipes']:
        a,b,sensitivity = c.evaluate(spec['name'],0.,inputs)
        components.append(dict(name=spec['name'],metrics=a.mean().to_dict(),
                               windows={w['name']:a.loc[w['test_start']:w['test_end']].mean().to_dict() for w in c.CFG['discovery']},
                               scenario_score_gains=sensitivity,diagnostic_only=True))
    done = [c.checked_json(p) for p in c.RUN.glob('*/*/done.json')]
    doc = dict(run_id=c.CFG['name'],selection=decision,confirmation=confirmation,components=components,
               training=dict(checkpoints=len(done),summed_fit_seconds=sum(d['seconds'] for d in done),
                             serialized_replay_max_error=max(d['serialized_replay_max_error'] for d in done)))
    out = REPO/'handoffs/results'
    c.r.write_json(out/f'{c.CFG["name"]}-metrics.json',doc)
    lines = ['# Robust temporal learning 11','',decision['label'],'',
             '| algorithm | core proportion | FIT | proxy score | FIT gain | proxy-score gain | Holm | eligible |',
             '|---|---|---|---|---|---|---|---|']
    for row in sorted(decision['comparisons'],key=lambda row:row['paired']['score']['diff_mean'],reverse=True):
        lines.append(f'| {row["name"]} | {row["core_neutralization"]:g} | {row["metrics"]["fit"]:.8f} | {row["metrics"]["score"]:.8f} | '
                     f'{row["paired"]["fit"]["diff_mean"]:+.8f} | {row["paired"]["score"]["diff_mean"]:+.8f} | {row["holm_pass"]} | {row["eligible"]} |')
    lines += ['',f'Control replay: {json.dumps(decision["control_replay"])}',
              '',f'Confirmation: {json.dumps(confirmation,sort_keys=True)}',
              '',f'{len(done)} model checkpoints replayed their actual serialized boosters; max error {doc["training"]["serialized_replay_max_error"]:g}.',
              '', 'Components are descriptive, cannot replace a failed comparison, and were all retained in their registered groups.',
              '', '| component | FIT | proxy score | early FIT | middle FIT | late FIT |',
              '|---|---|---|---|---|---|']
    for row in components:
        lines.append(f'| {row["name"]} | {row["metrics"]["fit"]:.8f} | {row["metrics"]["score"]:.8f} | '
                     f'{row["windows"]["early"]["fit"]:.8f} | {row["windows"]["middle"]["fit"]:.8f} | {row["windows"]["late"]["fit"]:.8f} |')
    lines += ['', 'Baseline01 remains unchanged. No deployment, old-audit reads, uploads, submissions or staking.']
    (out/f'{c.CFG["name"]}-results.md').write_text('\n'.join(lines)+'\n')
    return doc


def wire():
    c.load,c.train,c.select = load,train,select
    c.fit,c.candidate_prediction = fit,prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['init','train','select','confirm','report'])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    wire()
    load()
    if args.stage=='train':train()
    elif args.stage=='select':print(json.dumps(select(),indent=2))
    elif args.stage=='confirm':print(json.dumps(c.confirm(),indent=2))
    elif args.stage=='report':report()


if __name__=='__main__':main()
