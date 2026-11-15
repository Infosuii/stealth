#!/usr/bin/env python3
"""Source-bound sequential research. Local-only; no deployment or platform imports."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import research_run as r
from stealth import cv
from stealth.models import lgbm
from stealth.temporal_models import blocks, composite, core_mean, fit, fixed_fit, rank_by_era

log = logging.getLogger('temporal')
CFG = FRAME = RUN = PLAN = None


def load(confirmation=False):
    global CFG, FRAME, RUN, PLAN
    CFG = yaml.safe_load((REPO/'configs/temporal.yaml').read_text())
    RUN = REPO/'artifacts'/CFG['name']
    RUN.mkdir(parents=True, exist_ok=True)
    files = ['scripts/temporal.py', 'scripts/research_run.py', 'stealth/temporal_models.py',
             'stealth/metrics.py', 'stealth/cv.py', 'stealth/models/lgbm.py', 'docs/TEMPORAL-10.md']
    plan = dict(protocol='sequential-composite-proxy-v1', config=CFG,
                data_sha256=r.sha(REPO/'data/futures_train.parquet'),
                code_sha256={f:r.sha(REPO/f) for f in files},
                versions={p:importlib.metadata.version(p) for p in ('numpy','pandas','lightgbm','scipy')})
    PLAN = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan['plan_sha256'] = PLAN
    path = RUN/'plan.json'
    if path.exists() and json.loads(path.read_text())['plan_sha256'] != PLAN:
        raise RuntimeError('changed temporal plan: use a new run ID')
    if not path.exists():
        r.write_json(path, plan)
    ceiling = CFG['excluded_from_research'][0] if confirmation else CFG['reader_ceiling']
    if confirmation:
        decision = checked_json(RUN/'selection.json')
        if decision['selected'] is None:
            raise RuntimeError('no frozen discovery winner; confirmation reader refused')
    FRAME = pd.read_parquet(REPO/'data/futures_train.parquet', filters=[('exped','<',f'exped_{ceiling:04d}')])
    FRAME['era'] = FRAME.exped.str.rsplit('_', n=1).str[-1].astype(np.int32)
    FRAME = FRAME.loc[FRAME.era >= CFG['train_min_era']].copy()
    if FRAME.era.max() >= ceiling or not FRAME.index.is_unique:
        raise RuntimeError('reader boundary/index violation')
    log.info('loaded %d rows, eras %d..%d; plan %s', len(FRAME), FRAME.era.min(), FRAME.era.max(), PLAN)


def checked_json(path):
    doc = json.loads(path.read_text())
    if doc['plan_sha256'] != PLAN:
        raise RuntimeError(f'wrong checkpoint plan: {path}')
    for file, digest in doc.get('files', {}).items():
        if r.sha(path.parent/file) != digest:
            raise RuntimeError(f'corrupt checkpoint: {path.parent/file}')
    return doc


def windows(confirmation=False):
    return [CFG['confirmation']] if confirmation else CFG['discovery']


def fit_job(block, spec, count_donor=None):
    from threadpoolctl import threadpool_limits
    directory = RUN/block['name']/spec['name']
    done = directory/'done.json'
    if done.exists():
        checked_json(done)
        return f'{block["name"]}/{spec["name"]}: reused'
    directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    r.write_json(directory/'started.json', dict(plan_sha256=PLAN, at=datetime.now(timezone.utc).isoformat()))
    columns = [c for c in FRAME if c.startswith('feature_')]
    tr = FRAME.loc[FRAME.era < block['train_end']]
    te = FRAME.loc[FRAME.era.between(block['test_start'], block['test_end'])]
    if tr.empty or te.empty or te.era.min()-tr.era.max() <= CFG['gap']:
        raise RuntimeError('availability boundary violated')
    X, Xt = (a[columns].to_numpy(dtype=np.float32, copy=True) for a in (tr,te))
    X[X<0], Xt[Xt<0] = np.nan, np.nan
    with threadpool_limits(limits=CFG['threads']):
        if count_donor is None:
            model, info = fit(spec, CFG['baseline'], X, tr.target_everest.to_numpy(), tr.era.to_numpy(),
                              inner_width=CFG['inner_width'], gap=CFG['gap'], benchmark_trees=CFG['inner_benchmark_trees'],
                              bound=CFG['score_bound'], seed=CFG['seed'], threads=CFG['threads'])
        else:
            donor = RUN/block['name']/count_donor
            checked_json(donor/'done.json')
            count = json.loads((donor/'training.json').read_text())['n_iter']
            model = fixed_fit(spec,CFG['baseline'],X,tr.target_everest.to_numpy(),tr.era.to_numpy(),count,
                              seed=CFG['seed'],threads=CFG['threads'])
            info = dict(n_iter=count,count_donor=count_donor,donor_training_sha256=r.sha(donor/'training.json'),
                        train_max=int(tr.era.max()),diagnostic_only=True)
        prediction = lgbm.predict(model, Xt)
        blob = lgbm.to_bytes(model)
        replay = lgbm.predict(lgbm.from_bytes(blob), Xt)
        np.testing.assert_allclose(replay, prediction, atol=1e-12, rtol=1e-12)
        if not np.isfinite(prediction).all():
            raise RuntimeError('nonfinite prediction')
    (directory/'model.bin').write_bytes(blob)
    pd.DataFrame(dict(era=te.era, prediction=prediction), index=te.index).to_parquet(directory/'predictions.parquet')
    r.write_json(directory/'training.json', info)
    r.write_json(done, dict(plan_sha256=PLAN, block=block, recipe=spec['name'], seconds=time.perf_counter()-start,
                           serialized_replay_max_error=float(np.abs(replay-prediction).max()),
                           files={f:r.sha(directory/f) for f in ('model.bin','predictions.parquet','training.json')}))
    return f'{block["name"]}/{spec["name"]}: {time.perf_counter()-start:.1f}s, trees={info["n_iter"]}'


def train(confirmation=False):
    bs = blocks(windows(confirmation), CFG['cadence'], CFG['gap'])
    specs = CFG['recipes']
    if confirmation:
        chosen = checked_json(RUN/'selection.json')['selected']['name']
        specs = [s for s in specs if s['name'] in ('baseline',chosen) or chosen=='consensus']
    jobs = [(b,s) for b in bs for s in specs]
    # Static controls deliberately use identical stopping rules and only window-start data.
    for w in windows(confirmation):
        jobs.append(({**w,'name':f'static_{w["name"]}','train_end':w['test_start']-CFG['gap']}, CFG['recipes'][0]))
    with ProcessPoolExecutor(max_workers=CFG['workers'], mp_context=multiprocessing.get_context('fork')) as pool:
        pending = [pool.submit(fit_job,b,s) for b,s in jobs]
        for f in as_completed(pending):
            log.info('%s', f.result())
    r.write_json(RUN/('confirmation-training.json' if confirmation else 'discovery-training.json'),
                 dict(plan_sha256=PLAN, jobs=len(jobs)))


def predictions(name, confirmation=False, static=False):
    if static:
        bs = [dict(name=f'static_{w["name"]}') for w in windows(confirmation)]
    else:
        bs = blocks(windows(confirmation), CFG['cadence'], CFG['gap'])
    parts = []
    for b in bs:
        directory = RUN/b['name']/name
        checked_json(directory/'done.json')
        parts.append(pd.read_parquet(directory/'predictions.parquet'))
    out = pd.concat(parts)
    if not out.index.is_unique:
        raise RuntimeError('overlapping prediction rows')
    return out


def comparison_inputs(confirmation=False):
    base = predictions('baseline', confirmation)
    static = predictions('baseline', confirmation, static=True).reindex(base.index)
    frame = FRAME.loc[base.index]
    X = frame[[c for c in frame if c.startswith('feature_')]].to_numpy(dtype=float)
    return base, static, frame, [core_mean(X), core_mean(X[:,::2]), core_mean(X[:,1::2])]


def neutralized(prediction, era, core, proportion):
    z = rank_by_era(prediction, era)
    c = rank_by_era(core, era)
    for _, ids in r.metrics.era_groups(pd.Series(np.asarray(era))):
        cc = np.nan_to_num(c[ids])
        mm = cc @ cc
        if mm>0:
            z[ids] -= proportion*cc*(cc@z[ids])/mm
    return z


def candidate_prediction(name, base, confirmation=False):
    names = [s['name'] for s in CFG['recipes']] if name=='consensus' else [name]
    return np.mean([rank_by_era(predictions(n,confirmation).reindex(base.index).prediction.to_numpy(),base.era)
                    for n in names], axis=0)


def evaluate(name, proportion, inputs, confirmation=False):
    base, static, frame, cores = inputs
    raw = candidate_prediction(name,base,confirmation)
    # The *policy* always neutralizes against all features; changing the scoring scenario cannot change predictions.
    prediction = neutralized(raw,base.era,cores[0],proportion)
    scenarios = {}
    primary = baseline = None
    for ref_name, ref in [('refreshed',base.prediction),('static',static.prediction)]:
        for i,core in enumerate(cores):
            a = composite(prediction,frame.target_everest,base.era,ref,core,bound=CFG['score_bound'])
            b = composite(base.prediction,frame.target_everest,base.era,ref,core,bound=CFG['score_bound'])
            scenarios[f'{ref_name}/core_{i}'] = float((a.score-b.score).mean())
            if ref_name=='refreshed' and i==0:
                primary,baseline = a,b
    return primary,baseline,scenarios


def result_row(name, proportion, inputs, confirmation=False):
    a,b,sensitivity = evaluate(name,proportion,inputs,confirmation)
    deltas = a-b
    groups = windows(confirmation)
    if confirmation:
        lo,hi = CFG['confirmation']['test_start'],CFG['confirmation']['test_end']
        midpoint = (lo+hi)//2
        groups = [dict(name='first_half',test_start=lo,test_end=midpoint),dict(name='second_half',test_start=midpoint+1,test_end=hi)]
    by_window = {w['name']:{k:float(deltas.loc[w['test_start']:w['test_end'],k].mean()) for k in ('fit','score')}
                 for w in groups}
    paired = {k:cv.paired_block_bootstrap(deltas[k],**CFG['bootstrap']).as_dict() for k in ('fit','score')}
    block_sensitivity = {str(block):cv.paired_block_bootstrap(deltas.score,**{**CFG['bootstrap'],'block':block}).as_dict()
                         for block in (40,60)}
    out = dict(name=name, core_neutralization=proportion, metrics=a.mean().to_dict(), baseline=b.mean().to_dict(),
               fit_sharpe=float(a.fit.mean()/a.fit.std()), proxy_score_sharpe=float(a.score.mean()/a.score.std()),
               paired=paired, windows=by_window, scenario_score_gains=sensitivity, block_sensitivity=block_sensitivity)
    return out, a, b


def select():
    path = RUN/'selection.json'
    if path.exists():
        return checked_json(path)
    inputs = comparison_inputs()
    comparisons = []
    names = [s['name'] for s in CFG['recipes'] if s['name']!='baseline'] + ['consensus']
    for name in names + ['baseline']:
        for p in CFG['core_neutralization']:
            if name=='baseline' and p==0:
                continue
            row,a,b = result_row(name,p,inputs)
            a.to_parquet(RUN/f'scores_{name}_{p:g}.parquet')
            comparisons.append(row)
    gates = CFG['promotion']
    stopped = False
    for i,row in enumerate(sorted(comparisons,key=lambda row:row['paired']['score']['p_value'])):
        paired = row['paired']
        threshold = gates['alpha']/(len(comparisons)-i)
        ok = not stopped and paired['score']['inference_valid'] and paired['score']['p_value']<=threshold
        stopped = stopped or not ok
        row['holm_threshold'],row['holm_pass'] = threshold,bool(ok)
        row['eligible'] = bool(ok and paired['score']['diff_mean']>=gates['min_score_gain']
                               and paired['score']['p_pos']>=gates['p_pos']
                               and paired['fit']['diff_mean']>=gates['min_fit_gain']
                               and paired['fit']['p_pos']>=gates['p_pos']
                               and min(row['scenario_score_gains'].values())>=0
                               and all(min(v[k] for v in row['windows'].values())>=gates['worst_window_gain'] for k in ('fit','score'))
                               and sum(v['score']>0 for v in row['windows'].values())>=gates['min_positive_windows'])
    eligible = sorted([v for v in comparisons if v['eligible']],key=lambda v:v['paired']['score']['diff_mean'],reverse=True)
    base,static,frame,cores = inputs
    static_scores = composite(static.prediction,frame.target_everest,base.era,base.prediction,cores[0],bound=CFG['score_bound'])
    refreshed_scores = composite(base.prediction,frame.target_everest,base.era,base.prediction,cores[0],bound=CFG['score_bound'])
    control = dict(static=static_scores.mean().to_dict(),refreshed=refreshed_scores.mean().to_dict(),
                   paired_fit=cv.paired_block_bootstrap(refreshed_scores.fit-static_scores.fit,**CFG['bootstrap']).as_dict(),
                   windows={w['name']:float((refreshed_scores.fit-static_scores.fit).loc[w['test_start']:w['test_end']].mean()) for w in CFG['discovery']})
    doc = dict(plan_sha256=PLAN, label='adaptive historical development estimates; composite scores use proxies',
               comparisons=comparisons, selected=eligible[0] if eligible else None, refresh_control=control,
               confirmation_scored=False)
    r.write_json(path,doc)
    return doc


def confirm():
    path = RUN/'confirmation.json'
    if path.exists():
        return checked_json(path)
    selection = select()
    chosen = selection['selected']
    if chosen is None:
        doc = dict(plan_sha256=PLAN, qualified=False, confirmation_scored=False, reason='no eligible discovery candidate')
    else:
        load(confirmation=True)
        train(confirmation=True)
        row,_,_ = result_row(chosen['name'],chosen['core_neutralization'],comparison_inputs(True),True)
        qualified = all(row['paired'][k]['diff_mean']>0 and row['paired'][k]['p_pos']>=CFG['promotion']['p_pos'] for k in ('fit','score'))
        qualified = qualified and all(min(v[k] for v in row['windows'].values())>=CFG['promotion']['worst_window_gain'] for k in ('fit','score'))
        qualified = qualified and min(row['scenario_score_gains'].values())>=0
        relative = row['paired']['fit']['diff_mean']/row['baseline']['fit']
        discovery_relative = chosen['paired']['fit']['diff_mean']/chosen['baseline']['fit']
        doc = dict(plan_sha256=PLAN, qualified=bool(qualified), confirmation_scored=True, result=row,
                   major_improvement=bool(qualified and min(relative,discovery_relative)>=CFG['major_relative_gain']),
                   label='historically viewed, round-locked confirmation; no prospective or platform evidence')
    r.write_json(path,doc)
    return doc


def diagnose():
    path = RUN/'diagnosis.json'
    if path.exists():
        return checked_json(path)
    select()  # diagnostic fits are never selection candidates
    jobs,variants = [],[]
    for raw_name,score_name in [('baseline','score_expanding'),('rolling','score_rolling')]:
        raw = next(s for s in CFG['recipes'] if s['name']==raw_name)
        scored = next(s for s in CFG['recipes'] if s['name']==score_name)
        for original,donor in [(raw,score_name),(scored,raw_name)]:
            spec = {**original,'name':f'diag_{original["name"]}_count_{donor}'}
            variants.append(dict(name=spec['name'],target='score' if original['score_aware'] else 'raw',
                                 history=original['history'],count_donor=donor))
            jobs.extend((b,spec,donor) for b in blocks(CFG['discovery'],CFG['cadence'],CFG['gap']))
    with ProcessPoolExecutor(max_workers=CFG['workers'],mp_context=multiprocessing.get_context('fork')) as pool:
        pending = [pool.submit(fit_job,*job) for job in jobs]
        for f in as_completed(pending):log.info('%s',f.result())
    inputs = comparison_inputs()
    results = []
    for variant in variants:
        a,b,sensitivity = evaluate(variant['name'],0.,inputs)
        results.append({**variant,'metrics':a.mean().to_dict(),
                        'windows':{w['name']:a.loc[w['test_start']:w['test_end']].mean().to_dict() for w in CFG['discovery']},
                        'scenario_score_gains':sensitivity})
    doc = dict(plan_sha256=PLAN,label='descriptive target/count controls; cannot select or promote',variants=results)
    r.write_json(path,doc)
    return doc


def report():
    selection,confirmation = select(),checked_json(RUN/'confirmation.json')
    records = list(RUN.glob('*/*/done.json'))
    times = [checked_json(p) for p in records]
    models = dict(checkpoints=len(times),serialized_replay_max_error=max(d['serialized_replay_max_error'] for d in times),
                  summed_fit_seconds=sum(d['seconds'] for d in times))
    diagnosis = checked_json(RUN/'diagnosis.json')
    doc = dict(run_id=CFG['name'],label=selection['label'],selection=selection,confirmation=confirmation,training=models,diagnosis=diagnosis)
    out = REPO/'handoffs/results'
    r.write_json(out/f'{CFG["name"]}-metrics.json',doc)
    lines = ['# Temporal adaptation and composite score 10','',selection['label'],
             '','All composite metrics use local reference proxies, not the platform benchmark or known core set.',
             '','| recipe | core proportion | FIT | proxy score | FIT gain | proxy-score gain | Holm | eligible |',
             '|---|---|---|---|---|---|---|---|']
    for row in sorted(selection['comparisons'],key=lambda v:v['paired']['score']['diff_mean'],reverse=True):
        lines.append(f'| {row["name"]} | {row["core_neutralization"]:g} | {row["metrics"]["fit"]:.8f} | {row["metrics"]["score"]:.8f} | '
                     f'{row["paired"]["fit"]["diff_mean"]:+.8f} | {row["paired"]["score"]["diff_mean"]:+.8f} | {row["holm_pass"]} | {row["eligible"]} |')
    control = selection['refresh_control']
    lines += ['',f'Refreshed baseline FIT: {control["refreshed"]["fit"]:.8f}; static baseline: {control["static"]["fit"]:.8f}.',
              f'Refresh paired FIT gain: {control["paired_fit"]["diff_mean"]:+.8f}; p_pos {control["paired_fit"]["p_pos"]:.4f}.',
              '',f'Confirmation: {json.dumps(confirmation,sort_keys=True)}',
              '',f'{models["checkpoints"]} byte-bound fit checkpoints; serialization max error {models["serialized_replay_max_error"]:g}.',
              '', 'Baseline01 retained. No platform writes, old-audit reads or deployment changes.']
    lines += ['', 'Matched target/count controls (descriptive; no selection):', '',
              '| variant | FIT | proxy score | early FIT | middle FIT | late FIT |',
              '|---|---|---|---|---|---|']
    for row in diagnosis['variants']:
        lines.append(f'| {row["name"]} | {row["metrics"]["fit"]:.8f} | {row["metrics"]["score"]:.8f} | '
                     f'{row["windows"]["early"]["fit"]:.8f} | {row["windows"]["middle"]["fit"]:.8f} | {row["windows"]["late"]["fit"]:.8f} |')
    (out/f'{CFG["name"]}-results.md').write_text('\n'.join(lines)+'\n')
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['init','train','select','confirm','diagnose','report'])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    load()
    if args.stage=='train': train()
    elif args.stage=='select': print(json.dumps(select(),indent=2))
    elif args.stage=='confirm': print(json.dumps(confirm(),indent=2))
    elif args.stage=='diagnose': diagnose()
    elif args.stage=='report': report()


if __name__=='__main__': main()
