#!/usr/bin/env python3
"""Frozen research-policy full refit and lane packaging; never uploads or scores."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import history_context as h
from stealth import bundle
from stealth.objective_models import labels
from stealth.temporal_models import core_mean
from stealth.ranking_models import params as cat_params, pool, cpu_predict


def components(policy, run):
    name = policy.removesuffix('_blend')
    if run == 'history18':
        if policy != 'cpu_expanding3':
            raise ValueError('history18 upload fallback is frozen to pure CPU expanding3')
        return [name]
    allowed = ('cpu_core400', 'cpu_core3', 'gpu_core400', 'gpu_core3')
    if name not in allowed:
        raise ValueError('not a registered utility19 policy')
    return [name, 'cpu_raw400'] if policy.endswith('_blend') else [name]


def source_recipe(name):
    floor = 400 if name.endswith('400') else 3
    return dict(floor=floor, target='core' if '_core' in name else 'raw', gpu=name.startswith('gpu'))


def payload(members, columns, plan):
    names = [m['name'] for m in members]
    return dict(version=bundle.VERSION, git_sha=bundle.git_sha(), built_at=datetime.now(timezone.utc).isoformat(),
        run_id='upload06', feature_cols=columns, missing_cols=[], members=members,
        weights={n:1/len(names) for n in names}, p_b=0., p_c=0., p_source='frozen research policy',
        p_reason='no post-research neutralisation or validation tuning', blend_rule='fixed_equal',
        admitted_arms=sorted({m['arm'] for m in members}), core_proxy_cols=[], min_bench_coverage=1.,
        member_subset=names, weights_source='registered historical policy',
        artifact_source='frozen research policy; final full-data refit', source_run_fingerprint=plan)


def source_equivalence(s, names, columns):
    from stealth._runtime import make_predict, gaussianise, rank01
    parts = {n:s.predictions(n) for n in names}  # verifies every chosen native source checkpoint
    maximum, replays = 0., 0
    for start in (2000, 4000, 5340):
        block = next(b for b in s.blocks(s.CFG['intervals'],100,40) if b['test_start']==start)
        _, te = s.splits(s.FRAME,block)
        members=[]
        for name in names:
            spec=source_recipe(name);d=s.RUN/block['name']/name
            files=[f'model{seed}.bin' for seed in (17,29)] if spec['gpu'] else ['model.bin']
            members.append(dict(name=name,family='cat' if spec['gpu'] else 'lgbm',target='target_everest',
                group='everest',arm='cat' if spec['gpu'] else 'lgbm',input='nan',feature_cols=columns,
                blobs=[(d/f).read_bytes() for f in files],blob_rule='gaussian_mean' if spec['gpu'] else 'raw_mean'))
        predict=make_predict(payload(members,columns,s.PLAN))
        for _, live in te.groupby('era',sort=True):
            expected=rank01(sum(gaussianise(parts[n].reindex(live.index).prediction.to_numpy()) for n in names)/len(names))
            actual=predict(live[columns]).iloc[:,0].to_numpy()
            np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
            maximum=max(maximum,float(np.abs(actual-expected).max()));replays+=1
    return dict(era_replays=replays,max_error=maximum)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decision', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    decision = json.loads(args.decision.read_text())
    if decision['source_run'] == 'history18':
        h.load()
    elif decision['source_run'] == 'utility19':
        import utility_history
        utility_history.load()
    else:
        raise ValueError('unknown frozen source')
    s = h.s
    source = s.RUN
    if decision['source_plan_sha256'] != s.PLAN:
        raise RuntimeError('selected source plan changed')
    if args.out.exists() or bundle.manifest_path(args.out).exists():
        raise RuntimeError('artifact already exists; verify existing provenance rather than silently refitting')
    names = components(decision['policy'], decision['source_run'])
    columns = [c for c in s.FRAME if c.startswith('feature_')]
    equivalence = source_equivalence(s,names,columns)
    blocks = sorted(s.blocks(s.CFG['intervals'], 100, 40), key=lambda b:b['test_start'])[-3:]
    if [b['test_start'] for b in blocks] != [5640, 5740, 5840]:
        raise RuntimeError('unexpected source count blocks')
    # Training labels after 5924 are used ONLY in this frozen final fit, never in model/count selection.
    frame = pd.read_parquet(s.REPO/'data/futures_train.parquet')
    era = frame.exped.str.rsplit('_', n=1).str[-1].astype(np.int32).to_numpy()
    if columns != [c for c in frame if c.startswith('feature_')] or len(columns) != 178 or not frame.index.is_unique:
        raise RuntimeError('full-refit feature/index contract changed')
    X = frame[columns].to_numpy(np.float32); X[X < 0] = np.nan
    y = frame.target_everest.to_numpy()
    if not np.isfinite(y).all():
        raise RuntimeError('missing final train labels')
    members, provenance = [], []
    dest = args.decision.parent
    for name in names:
        spec = source_recipe(name)
        seeds = [17,29] if spec['gpu'] else [17]
        keep = era >= spec['floor']
        xx, yy, ee = X[keep], y[keep], era[keep]
        transformed = labels(spec['target'], yy, ee, core_mean(xx), np.full(len(yy), np.nan))
        blobs = []
        for seed in seeds:
            suffix = str(seed) if spec['gpu'] else ''
            counts, markers = [], {}
            for block in blocks:
                d = source/block['name']/name
                s.native_cached(d/'done.json')
                info = json.loads((d/f'training{suffix}.json').read_text())
                if info['train_max'] >= block['test_start']-40 or info['train_min'] != spec['floor']:
                    raise RuntimeError('source fitting boundary changed')
                counts.append(info['n_iter'])
                markers[str((d/'done.json').relative_to(source))] = s.r.sha(d/'done.json')
            count = int(np.median(counts))
            with threadpool_limits(limits=4 if spec['gpu'] else 6):
                if spec['gpu']:
                    from catboost import CatBoostRegressor
                    from catboost.utils import get_gpu_device_count
                    if get_gpu_device_count() < 1:raise RuntimeError('final GPU fit requires CUDA')
                    recipe = next(r for r in s.CFG['gpu_recipes'] if r['floor']==spec['floor'])
                    model = CatBoostRegressor(**{**cat_params({**recipe,'loss':'RMSE'}, seed=seed), 'iterations':count})
                    model.fit(pool(xx, transformed, ee, 'RMSE'))
                    blob = s.cat.to_bytes(model)
                    pred = cpu_predict(model, xx[:1000]); replay = s.cat.predict(s.cat.from_bytes(blob), xx[:1000])
                else:
                    params = dict(s.CFG['baseline']); params.pop('cap')
                    model = s.lgbm.fit_fixed(params, xx, transformed, None, n_iter=count, seed=seed, threads=6)
                    blob = s.lgbm.to_bytes(model)
                    pred = s.lgbm.predict(model, xx[:1000]); replay = s.lgbm.predict(s.lgbm.from_bytes(blob), xx[:1000])
            np.testing.assert_allclose(pred, replay, atol=1e-12, rtol=1e-12)
            filename = f'{name}-seed{seed}.bin'
            (dest/filename).write_bytes(blob); blobs.append(blob)
            provenance.append(dict(name=name, seed=seed, target=spec['target'], floor=spec['floor'],
                source_counts=counts, count=count, source_markers=markers, train_min=int(ee.min()),
                train_max=int(ee.max()), rows=len(ee), file=filename,
                sha256=hashlib.sha256(blob).hexdigest(), serialization_max_error=float(np.abs(pred-replay).max())))
        members.append(dict(name=name, family='cat' if spec['gpu'] else 'lgbm', target='target_everest',
            group='everest', arm='cat' if spec['gpu'] else 'lgbm', input='nan', feature_cols=columns,
            blobs=blobs, blob_rule='gaussian_mean' if spec['gpu'] else 'raw_mean'))
    result = payload(members,columns,s.PLAN)
    sha = bundle.write_pickle(result, args.out)
    bundle.write_manifest(args.out, result, sha)
    record = dict(decision=decision, decision_sha256=s.r.sha(args.decision),
        training_data_sha256=s.r.sha(s.REPO/'data/futures_train.parquet'),
        final_refit_uses_later_train_labels=True, later_labels_used_for_selection=False,
        source_runtime_equivalence=equivalence,native_fits=provenance, pkl_sha256=sha, pkl=str(args.out), source_plan_sha256=s.PLAN)
    s.r.write_json(dest/'refit-provenance.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':main()
