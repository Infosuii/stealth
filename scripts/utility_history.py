#!/usr/bin/env python3
"""Long-history utility targets and raw-count controls; historical research only."""
import argparse
import hashlib
import json
import logging
import shutil
import time

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

import history_context as h
from stealth.objective_models import prepare, labels, fit_cpu, fit_gpu
from stealth.ranking_models import params as cat_params, pool, cpu_predict

s = h.s
SOURCE = PARENT = None


def filtered(frame, spec):
    return frame.loc[frame.era >= spec['floor']]


def load():
    global SOURCE, PARENT
    h.load()
    SOURCE = s.RUN
    PARENT = json.loads((SOURCE / 'plan.json').read_text())
    cfg = yaml.safe_load((s.REPO / 'configs/utility-history.yaml').read_text())
    for key in ('baseline', 'seed', 'gap', 'cadence', 'inner_width', 'intervals', 'bootstrap', 'promotion'):
        if cfg[key] != PARENT['config'][key]:
            raise RuntimeError('utility-history matched configuration changed: ' + key)
    for recipe in cfg['gpu_recipes']:
        raw = next(r for r in PARENT['config']['gpu_recipes'] if r['name'] == f"gpu_expanding{recipe['floor']}")
        if {k:v for k,v in raw.items() if k not in ('name', 'target')} != {k:v for k,v in recipe.items() if k not in ('name', 'target')}:
            raise RuntimeError('utility GPU parameters/history changed')
    markers = [SOURCE / b['name'] / f'{family}_expanding{floor}' / 'done.json'
               for b in s.blocks(cfg['intervals'], 100, 40) for family in ('cpu', 'gpu') for floor in (400, 3)]
    for p in markers:
        s.native_cached(p)
    plan = dict(protocol='long-history-core-count-v1', config=cfg,
                parent_plan_sha256=s.r.sha(SOURCE / 'plan.json'), teacher_sha256=s.TEACHER_SHA,
                sources={str(p.relative_to(SOURCE)):s.r.sha(p) for p in markers},
                code_sha256={f:s.r.sha(s.REPO / f) for f in ['scripts/utility_history.py', 'docs/UTILITY-HISTORY-19.md']})
    s.CFG = cfg
    s.RUN = s.REPO / 'artifacts' / cfg['name']
    s.RUN.mkdir(parents=True, exist_ok=True)
    s.PLAN = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan['plan_sha256'] = s.PLAN
    p = s.RUN / 'plan.json'
    if p.exists() and json.loads(p.read_text())['plan_sha256'] != s.PLAN:
        raise RuntimeError('utility-history experiment changed')
    if not p.exists():
        s.r.write_json(p, plan)


def copy_controls():
    for block in s.blocks(s.CFG['intervals'], 100, 40):
        for family in ('cpu', 'gpu'):
            for floor in (400, 3):
                src = SOURCE / block['name'] / f'{family}_expanding{floor}'
                doc = json.loads((src / 'done.json').read_text())
                if doc['plan_sha256'] != PARENT['plan_sha256']:
                    raise RuntimeError('raw control source plan changed')
                dest = s.RUN / block['name'] / f'{family}_raw{floor}'
                if (dest / 'done.json').exists():
                    s.native_cached(dest / 'done.json')
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                for f, digest in doc['files'].items():
                    if s.r.sha(src / f) != digest:
                        raise RuntimeError('raw control source bytes changed')
                    shutil.copyfile(src / f, dest / f)
                s.r.write_json(dest / 'done.json', {**doc, 'plan_sha256':s.PLAN, 'seconds':0.,
                              'copied_from':'history18', 'source_done_sha256':s.r.sha(src / 'done.json')})


def inputs(block, spec):
    tr, te = s.splits(s.FRAME, block)
    tr = filtered(tr, spec)
    X, Xt = s.arrays(tr, te)
    return X, tr.target_everest.to_numpy(), tr.era.to_numpy(), s.TEACHER.reindex(tr.index).to_numpy(), Xt, te


def fixed_model(block, spec, count, seed=17, gpu=False):
    X, y, era, teacher, Xt, te = inputs(block, spec)
    X, y, era, teacher, inner, valid, core = prepare(spec, X, y, era, teacher, 200, 40)
    target = labels(spec['target'], y, era, core, teacher)
    with threadpool_limits(limits=4 if gpu else 6):
        if gpu:
            from catboost import CatBoostRegressor
            from catboost.utils import get_gpu_device_count
            if get_gpu_device_count() < 1:
                raise RuntimeError('fixed-count GPU fit requires CUDA')
            model = CatBoostRegressor(**{**cat_params({**spec, 'loss':'RMSE'}, seed=seed), 'iterations':count})
            model.fit(pool(X, target, era, 'RMSE'))
            if model.get_all_params()['task_type'] != 'GPU':
                raise RuntimeError('fixed count silently changed device')
            prediction = cpu_predict(model, Xt, threads=4)
        else:
            params = dict(s.CFG['baseline'])
            params.pop('cap')
            model = s.lgbm.fit_fixed(params, X, target, None, n_iter=count, seed=seed, threads=6)
            prediction = s.lgbm.predict(model, Xt)
    info = dict(n_iter=int(count), target=spec['target'], train_min=int(era.min()), train_max=int(era.max()),
                inner_fit_max=int(era[inner].max()), inner_valid_min=int(era[valid].min()),
                count_source='frozen raw inner selection', native_task_type='GPU' if gpu else 'CPU')
    return model, prediction, Xt, te, info


def cpu_job(block, spec):
    directory = s.RUN / block['name'] / spec['name']
    if (directory / 'done.json').exists():
        s.native_cached(directory / 'done.json')
        return f'{block["name"]}/{spec["name"]} reused'
    start = time.perf_counter()
    if spec.get('fixed_raw_count'):
        donor = s.RUN / block['name'] / f'cpu_raw{spec["floor"]}'
        s.native_cached(donor / 'done.json')
        count = json.loads((donor / 'training.json').read_text())['n_iter']
        model, prediction, Xt, te, info = fixed_model(block, spec, count)
        info['count_source_done_sha256'] = s.r.sha(donor / 'done.json')
    else:
        X, y, era, teacher, Xt, te = inputs(block, spec)
        with threadpool_limits(limits=6):
            model, info = fit_cpu(spec, s.CFG['baseline'], X, y, era, teacher,
                                  width=200, gap=40, seed=17, threads=6, bound=s.CFG['score_bound'])
            prediction = s.lgbm.predict(model, Xt)
    info.update(declared_floor=spec['floor'], teacher_manifest_sha256=s.TEACHER_SHA)
    s.finish(directory, model, prediction, Xt, te, info, time.perf_counter()-start, 'lgbm')
    return f'{block["name"]}/{spec["name"]} {time.perf_counter()-start:.1f}s trees={info["n_iter"]}'


def gpu_fit(spec, base, X, y, era, teacher, **kwargs):
    return fit_gpu(spec, base, *h.filter_rows(spec, X, y, era, teacher), **kwargs)


def fixed_gpu_job(block, spec):
    directory = s.RUN / block['name'] / (spec['name'] + '_rawcount')
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'done.json').exists():
        s.native_cached(directory / 'done.json')
        return
    donor = s.RUN / block['name'] / f'gpu_raw{spec["floor"]}'
    s.native_cached(donor / 'done.json')
    scores, records = [], []
    for seed in spec['seeds']:
        marker = directory / f'seed{seed}.done.json'
        if marker.exists():
            record = s.native_cached(marker)
        else:
            count = json.loads((donor / f'training{seed}.json').read_text())['n_iter']
            start = time.perf_counter()
            model, pred, Xt, te, info = fixed_model(block, spec, count, seed, gpu=True)
            info.update(teacher_manifest_sha256=s.TEACHER_SHA, count_source_done_sha256=s.r.sha(donor / 'done.json'))
            record = s.finish(directory, model, pred, Xt, te, info, time.perf_counter()-start, 'catboost', seed)
            s.log.info('fixed-count GPU %s/%s seed %d trees=%d', block['name'], spec['name'], seed, count)
        p = pd.read_parquet(directory / f'prediction{seed}.parquet')
        if len(scores) and not p.index.equals(index):
            raise RuntimeError('fixed GPU seed index changed')
        index, era = p.index, p.era
        scores.append(s.rank_by_era(p.prediction, era))
        records.append(record)
    pd.DataFrame(dict(era=era, prediction=np.mean(scores, axis=0)), index=index).to_parquet(directory / 'prediction.parquet')
    files = ['prediction.parquet'] + [f for seed in spec['seeds'] for f in
            [f'model{seed}.bin', f'prediction{seed}.parquet', f'training{seed}.json', f'seed{seed}.done.json']]
    s.r.write_json(directory / 'done.json', dict(plan_sha256=s.PLAN, family='catboost_group',
                  teacher_manifest_sha256=s.TEACHER_SHA, seconds=sum(r['seconds'] for r in records),
                  serialized_replay_max_error=max(r['serialized_replay_max_error'] for r in records),
                  files={f:s.r.sha(directory / f) for f in files}))


def rehearse():
    errors = []
    for start in (2000, 4000, 5340):
        block = next(b for b in s.blocks(s.CFG['intervals'], 100, 40) if b['test_start'] == start)
        for floor in (400, 3):
            donor = s.RUN / block['name'] / f'cpu_raw{floor}'
            info = json.loads((donor / 'training.json').read_text())
            _, prediction, _, te, _ = fixed_model(block, dict(target='raw', history=None, floor=floor), info['n_iter'])
            old = pd.read_parquet(donor / 'prediction.parquet').reindex(te.index).prediction.to_numpy()
            np.testing.assert_allclose(prediction, old, atol=1e-12, rtol=1e-12)
            errors.append(float(np.abs(prediction-old).max()))
    s.r.write_json(s.RUN / 'cpu-own-count-replay.json', dict(plan_sha256=s.PLAN, replays=6, max_error=max(errors)))


def summarize():
    path = s.RUN / 'results.json'
    if path.exists():
        return s.checked(path)
    replay = s.checked(s.RUN / 'cpu-own-count-replay.json')
    base = s.predictions('cpu_raw400')
    frame = s.FRAME.loc[base.index]
    X = frame[[x for x in frame if x.startswith('feature_')]].to_numpy(float)
    cores = [s.core_mean(X), s.core_mean(X[:, ::2]), s.core_mean(X[:, 1::2])]
    teacher = s.TEACHER.reindex(base.index).to_numpy()
    names = [f'{family}_{target}{floor}{suffix}' for family in ('cpu','gpu') for floor in (400,3)
             for target,suffix in [('raw',''),('core',''),('core','_rawcount')]]
    components = {n:s.rank_by_era(s.predictions(n).reindex(base.index).prediction, base.era) for n in names}
    for r in s.CFG['cpu_recipes']+s.CFG['gpu_recipes']:
        components[r['name']+'_blend'] = (components[r['name']]+components['cpu_raw400'])/2
    comparisons, controls = {}, []
    for w in s.CFG['intervals']:
        rows = [dict(name=n, **s.evaluate(components[n],base,teacher,frame,cores,w)) for n in s.CFG['comparisons']]
        g = s.CFG['promotion']; stopped = False
        for i,row in enumerate(sorted(rows,key=lambda r:r['paired']['score']['p_value'])):
            pair = row['paired']; limit = g['alpha']/(len(rows)-i)
            ok = not stopped and pair['score']['inference_valid'] and pair['score']['p_value']<=limit
            stopped = stopped or not ok
            row['holm_pass'], row['holm_threshold'] = bool(ok),limit
            row['passes_historical_gates'] = bool(ok and pair['fit']['diff_mean']>=g['min_fit_gain'] and
                pair['score']['diff_mean']>=g['min_score_gain'] and min(pair[k]['p_pos'] for k in ('fit','score'))>=g['p_pos'] and
                min(row['scenario_score_gains'].values())>=0 and all(min(v[k] for v in row['windows'].values())>=g['worst_window_gain'] for k in ('fit','score')) and
                sum(v['score']>0 for v in row['windows'].values())>=g['min_positive_windows'])
        comparisons[w['name']] = rows
        for family in ('cpu','gpu'):
            for floor in (400,3):
                raw,own,fixed = (f'{family}_{name}{floor}{suffix}' for name,suffix in [('raw',''),('core',''),('core','_rawcount')])
                for a,b in [(fixed,raw),(own,fixed)]:
                    controls.append(dict(interval=w['name'],challenger=a,comparator=b,
                        **s.evaluate(components[a],base,teacher,frame,cores,w,comparison=components[b])))
    records = [s.native_cached(p) for p in s.RUN.glob('*/*/done.json')]
    doc = dict(plan_sha256=s.PLAN,label='viewed historical long-history core utility and count controls',
               comparisons=comparisons,matched_controls=controls,cpu_own_count_replay=replay,promoted=False,
               checkpoint_groups=len(records),copied_groups=sum('copied_from' in r for r in records),
               serialized_replay_max_error=max(r['serialized_replay_max_error'] for r in records))
    s.r.write_json(path,doc)
    return doc


def report(doc):
    out=s.REPO/'handoffs/results'
    s.r.write_json(out/'utility19-metrics.json',doc)
    lines=['# Long-history core utility 19','',doc['label'],'',
           '| interval | policy | FIT | baseline FIT | proxy score | baseline proxy | historical gates |',
           '|---|---|---|---|---|---|---|']
    for phase,rows in doc['comparisons'].items():
        for r in rows:
            lines.append(f'| {phase} | {r["name"]} | {r["metrics"]["fit"]:.8f} | {r["baseline"]["fit"]:.8f} | {r["metrics"]["score"]:.8f} | {r["baseline"]["score"]:.8f} | {r["passes_historical_gates"]} |')
    lines+=['',f'{doc["checkpoint_groups"]} groups; {doc["copied_groups"]} copied controls; serialization error {doc["serialized_replay_max_error"]:g}.',
            'Matched-count controls are diagnostic only. No automatic promotion or platform write.']
    (out/'utility19-results.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['init','copy','rehearse','gpu-rehearse','cpu','gpu','summarize','report'])
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    load();s.fit_gpu=gpu_fit
    if args.stage=='copy':copy_controls()
    elif args.stage=='rehearse':rehearse()
    elif args.stage=='cpu':
        recipes=s.CFG['cpu_recipes']+[{**r,'name':r['name']+'_rawcount','fixed_raw_count':True} for r in s.CFG['cpu_recipes']]
        s.parallel([(b,r) for b in s.blocks(s.CFG['intervals'],100,40) for r in recipes],cpu_job)
    elif args.stage=='gpu':
        for b in s.blocks(s.CFG['intervals'],100,40):
            for r in s.CFG['gpu_recipes']:
                s.gpu_job(b,r);fixed_gpu_job(b,r)
    elif args.stage=='gpu-rehearse':
        rng=np.random.default_rng(17);era=np.repeat(np.arange(3,1003),5)
        X=rng.integers(0,10,(len(era),8)).astype(np.float32);y=rng.integers(0,5,len(era))/4
        for r in s.CFG['gpu_recipes']:
            model,info=gpu_fit({**r,'cap':4,'depth':3,'checkpoint_step':2},s.CFG['baseline'],X,y,era,np.zeros(len(era)),width=40,gap=40,seed=17,bound=s.CFG['score_bound'])
            np.testing.assert_allclose(cpu_predict(model,X[:40]),s.cat.predict(s.cat.from_bytes(s.cat.to_bytes(model)),X[:40]),atol=1e-12)
            s.log.info('GPU core long-history rehearsal floor %d passed',r['floor'])
    elif args.stage=='summarize':print(json.dumps(summarize(),indent=2))
    elif args.stage=='report':report(summarize())


if __name__=='__main__':main()
