#!/usr/bin/env python3
"""Pre-registered architecture research. Local files only; no platform client or audit rescoring."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import logging
import multiprocessing
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.special import ndtri
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from stealth import bundle, cv, metrics
from stealth._runtime import basis_forward, load_basis
from stealth.research_models import fit_basis, to_bytes

log = logging.getLogger('research')
FRAME = None
CONFIG = None
RUN = None
PLAN_SHA = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load(config_path=REPO / 'configs/research.yaml'):
    global FRAME, CONFIG, RUN, PLAN_SHA
    CONFIG = yaml.safe_load(Path(config_path).read_text())
    RUN = REPO / 'artifacts' / CONFIG['name']
    RUN.mkdir(parents=True, exist_ok=True)
    source = REPO / 'data/futures_train.parquet'
    plan = {'protocol': 'era-architecture-v1', 'config': CONFIG, 'data_sha256': sha(source)}
    plan['code_sha256'] = {str(p.relative_to(REPO)): sha(p) for p in
                          [Path(__file__), REPO / 'scripts/research_gpu.py', REPO / 'stealth/research_models.py',
                           REPO / 'stealth/_runtime.py', REPO / 'stealth/metrics.py', REPO / 'stealth/cv.py']}
    PLAN_SHA = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan['plan_sha256'] = PLAN_SHA
    old = RUN / 'plan.json'
    if old.exists() and json.loads(old.read_text()).get('plan_sha256') != PLAN_SHA:
        raise RuntimeError('research specification or data changed: do not reuse this run ID')
    if not old.exists():
        write_json(old, plan)
    # Exclude the already-viewed old audit at the parquet reader, not merely at fitting time.
    FRAME = pd.read_parquet(source, filters=[('exped', '<', 'exped_5925')])
    FRAME['era'] = FRAME['exped'].str.rsplit('_', n=1).str[-1].astype(np.int32)
    FRAME = FRAME.loc[FRAME.era >= CONFIG['train_min_era']].copy()
    if FRAME.era.max() >= CONFIG['excluded_from_research'][0]:
        raise RuntimeError('old audit reached the research data loader')
    log.info('loaded %d rows, eras %d..%d; old audit excluded', len(FRAME), FRAME.era.min(), FRAME.era.max())
    return CONFIG, RUN


def arrays(window):
    columns = [c for c in FRAME if c.startswith('feature_')]
    train = FRAME.era < window['train_end']
    test = FRAME.era.between(window['test_start'], window['test_end'])
    if window['test_start'] - window['train_end'] < 40 or not train.any() or not test.any():
        raise ValueError('invalid chronological split')
    tr, te = FRAME.loc[train], FRAME.loc[test]
    X = tr[columns].to_numpy(dtype=np.float64, copy=True)
    Xt = te[columns].to_numpy(dtype=np.float64, copy=True)
    X[X < 0] = np.nan
    Xt[Xt < 0] = np.nan
    return columns, tr, te, X, Xt


def cache_dir(window, name):
    return RUN / window['name'] / name


def cached(window, name):
    directory = cache_dir(window, name)
    record = directory / 'done.json'
    if not record.exists():
        return False
    doc = json.loads(record.read_text())
    if doc['plan_sha256'] != PLAN_SHA:
        raise RuntimeError('cached research model has another specification')
    for file, digest in doc['files'].items():
        if not (directory / file).exists() or sha(directory / file) != digest:
            raise RuntimeError(f'incomplete/corrupt research checkpoint {directory / file}')
    return True


def save_fit(window, name, family, blobs, prediction, te, seconds, extra=None):
    directory = cache_dir(window, name)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for i, blob in enumerate(blobs):
        path = directory / f'model{i}.bin'
        path.write_bytes(blob)
        files[path.name] = sha(path)
    result = pd.DataFrame({'era': te.era, 'prediction': prediction}, index=te.index)
    result.to_parquet(directory / 'predictions.parquet')
    files['predictions.parquet'] = sha(directory / 'predictions.parquet')
    write_json(directory / 'done.json', {'plan_sha256': PLAN_SHA, 'name': name, 'family': family,
                'seconds': seconds, 'files': files, 'rows': len(te), 'train_end': window['train_end'],
                'test_eras': [window['test_start'], window['test_end']], **(extra or {})})


def baseline_fit(window, X, y, eras):
    from stealth.models import lgbm

    params = dict(CONFIG['baseline'])
    cap = params.pop('cap')
    unique = np.unique(eras)
    cut = int(unique[max(1, int(len(unique) * 0.8))])
    fit, valid = eras < cut - 20, eras >= cut
    model, count = lgbm.fit_with_early_stopping(params, X[fit], y[fit], None, X[valid], y[valid],
                           cap=cap, seed=CONFIG['seed'], threads=CONFIG['threads'])
    return lgbm.fit_fixed(params, X, y, None, n_iter=count, seed=CONFIG['seed'], threads=CONFIG['threads']), count


def fit_job(window, spec):
    from threadpoolctl import threadpool_limits

    name = spec['name']
    if cached(window, name):
        return f'{window["name"]}/{name}: reused'
    start = time.perf_counter()
    with threadpool_limits(limits=CONFIG['threads']):
        columns, tr, te, X, Xt = arrays(window)
        y, era = tr.target_everest.to_numpy(), tr.era.to_numpy()
        if name == 'baseline':
            from stealth.models import lgbm

            model, count = baseline_fit(window, X, y, era)
            blob = lgbm.to_bytes(model)
            prediction = lgbm.predict(model, Xt)
            family, extra = 'lgbm', {'n_iter': count}
        else:
            model = fit_basis(spec, X, y, era, columns, seed=CONFIG['seed'], threads=CONFIG['threads'])
            prediction = basis_forward(model, Xt)
            blob, family, extra = to_bytes(model), 'basis', {}
        seconds = time.perf_counter() - start
        save_fit(window, name, family, [blob], prediction, te, seconds, extra)
    return f'{window["name"]}/{name}: done in {seconds:.1f}s'


def cpu_discovery():
    jobs = [(w, spec) for w in CONFIG['discovery']
            for spec in [{'name': 'baseline'}, *CONFIG['candidates']] if spec.get('kind') != 'net']
    with ProcessPoolExecutor(max_workers=CONFIG['workers'], mp_context=multiprocessing.get_context('fork')) as pool:
        futures = [pool.submit(fit_job, w, spec) for w, spec in jobs]
        for future in as_completed(futures):
            log.info('%s', future.result())
    write_json(RUN / 'cpu-discovery.json', {'status': 'done', 'plan_sha256': PLAN_SHA})


def ranked(prediction, era):
    ranks = pd.Series(prediction).groupby(np.asarray(era)).rank(method='average').to_numpy()
    count = pd.Series(era).groupby(np.asarray(era)).transform('size').to_numpy()
    return ndtri((ranks - 0.5) / count)


def read_prediction(window, name):
    if not cached(window, name):
        raise RuntimeError(f'{window["name"]}/{name} is not complete')
    return pd.read_parquet(cache_dir(window, name) / 'predictions.parquet')


def comparison(window, name, mix):
    base = read_prediction(window, 'baseline')
    candidate = read_prediction(window, name).reindex(base.index)
    target = FRAME.loc[base.index, 'target_everest']
    z = candidate.prediction
    if mix:
        z = pd.Series(0.5 * ranked(base.prediction, base.era) + 0.5 * ranked(z, base.era), index=base.index)
    fit = metrics.per_era_fit_fast(z, target, base.era)
    base_fit = metrics.per_era_fit_fast(base.prediction, target, base.era)
    return fit, base_fit


def summarise():
    if (RUN / 'selection.json').exists():
        return json.loads((RUN / 'selection.json').read_text())
    rows = []
    bs, gates = CONFIG['bootstrap'], CONFIG['promotion']
    for spec in CONFIG['candidates']:
        for mix in (False, True):
            parts = [comparison(w, spec['name'], mix) for w in CONFIG['discovery']]
            fit = pd.concat([p[0] for p in parts]).sort_index()
            base = pd.concat([p[1] for p in parts]).sort_index()
            delta = fit - base
            paired = cv.paired_block_bootstrap(delta, **bs).as_dict()
            window_diffs = {w['name']: float((p[0] - p[1]).mean()) for w, p in zip(CONFIG['discovery'], parts)}
            rows.append({'name': spec['name'], 'blend_with_baseline': mix, 'fit_mean': float(fit.mean()),
                         'baseline_mean': float(base.mean()), 'sharpe': float(fit.mean() / fit.std()),
                         'paired': paired, 'windows': window_diffs})
    stopped = False
    for j, row in enumerate(sorted(rows, key=lambda r: r['paired']['p_value'])):
        row['holm_threshold'] = gates['alpha'] / (len(rows) - j)
        ok = not stopped and row['paired']['inference_valid'] and row['paired']['p_value'] <= row['holm_threshold']
        stopped = stopped or not ok
        row['holm_pass'] = bool(ok)
        row['eligible'] = bool(ok and row['paired']['diff_mean'] >= gates['min_discovery_gain']
                  and row['paired']['p_pos'] >= gates['p_pos']
                  and sum(d > 0 for d in row['windows'].values()) >= gates['min_positive_windows']
                  and min(row['windows'].values()) >= gates['worst_window_gain'])
    eligible = sorted([r for r in rows if r['eligible']], key=lambda r: r['paired']['diff_mean'], reverse=True)
    doc = {'plan_sha256': PLAN_SHA, 'label': 'historical development selection estimates; not a fresh holdout',
           'comparisons': rows, 'selected': eligible[0] if eligible else None,
           'confirmation_scored': False}
    write_json(RUN / 'selection.json', doc)
    log.info('discovery decision FROZEN: %s', None if not eligible else eligible[0]['name'])
    return doc


def confirm():
    selection = summarise()
    if (RUN / 'confirmation.json').exists():
        return json.loads((RUN / 'confirmation.json').read_text())
    selected = selection['selected']
    if selected is None:
        doc = {'promoted': False, 'reason': 'no candidate passed discovery gates; baseline retained', 'plan_sha256': PLAN_SHA}
    else:
        window = CONFIG['confirmation']
        fit_job(window, {'name': 'baseline'})
        spec = next(s for s in CONFIG['candidates'] if s['name'] == selected['name'])
        if spec['kind'] == 'net':
            if not cached(window, spec['name']):
                raise RuntimeError('fit the frozen neural confirmation with research_gpu.py --confirmation first')
        else:
            fit_job(window, spec)
        fit, base = comparison(window, selected['name'], selected['blend_with_baseline'])
        paired = cv.paired_block_bootstrap(fit - base, **CONFIG['bootstrap']).as_dict()
        promoted = paired['inference_valid'] and paired['diff_mean'] > 0 and paired['p_pos'] >= CONFIG['promotion']['p_pos']
        doc = {'plan_sha256': PLAN_SHA, 'selected': selected['name'], 'blend_with_baseline': selected['blend_with_baseline'],
               'label': 'locked for this round, historically viewed; not independent future evidence',
               'fit_mean': float(fit.mean()), 'baseline_mean': float(base.mean()), 'paired': paired,
               'promoted': bool(promoted), 'reason': 'confirmation gate passed' if promoted else 'confirmation gate failed; baseline retained'}
    write_json(RUN / 'confirmation.json', doc)
    return doc


def package():
    from threadpoolctl import threadpool_limits

    confirmation = confirm()
    if not confirmation['promoted']:
        log.info('no candidate promoted; existing baseline remains unchanged')
        return None
    selected = json.loads((RUN / 'selection.json').read_text())['selected']
    spec = next(s for s in CONFIG['candidates'] if s['name'] == selected['name'])
    columns = [c for c in FRAME if c.startswith('feature_')]
    X = FRAME[columns].to_numpy(dtype=np.float64, copy=True)
    X[X < 0] = np.nan
    y, eras = FRAME.target_everest.to_numpy(), FRAME.era.to_numpy()
    full = RUN / 'full'
    full.mkdir(exist_ok=True)
    with threadpool_limits(limits=CONFIG['threads']):
        if spec['kind'] == 'net':
            location = full / spec['name']
            done = json.loads((location / 'done.json').read_text())
            if done['plan_sha256'] != PLAN_SHA:
                raise RuntimeError('neural refit has another plan')
            blobs = [(location / f).read_bytes() for f in done['files'] if f.startswith('model')]
        else:
            fitted = fit_basis(spec, X, y, eras, columns, seed=CONFIG['seed'], threads=CONFIG['threads'])
            blobs = [to_bytes(fitted)]
        members = [{'name': spec['name'], 'family': 'basis', 'target': 'target_everest', 'group': 'everest',
                    'arm': 'architecture', 'input': 'nan', 'feature_cols': columns, 'blobs': blobs}]
        weights = {spec['name']: 1.0}
        if selected['blend_with_baseline']:
            model, _ = baseline_fit(None, X, y, eras)
            members.append({'name': 'baseline', 'family': 'lgbm', 'target': 'target_everest', 'group': 'everest',
                            'arm': 'lgbm', 'input': 'nan', 'feature_cols': columns, 'blobs': [lgbm_bytes(model)]})
            weights = {spec['name']: 0.5, 'baseline': 0.5}
    result = {'version': 'research-v1', 'git_sha': bundle.git_sha(), 'built_at': datetime.now(timezone.utc).isoformat(),
              'run_id': CONFIG['name'], 'feature_cols': columns, 'missing_cols': [], 'members': members,
              'weights': weights, 'p_b': 0.0, 'p_c': 0.0, 'p_source': 'override',
              'p_reason': 'raw pre-registered architecture research; no audit tuning', 'blend_rule': 'fixed_equal' if len(members)>1 else 'single',
              'admitted_arms': [m['arm'] for m in members], 'core_proxy_cols': [], 'min_bench_coverage': 0.5,
              'blob_files': {}, 'weights_source': 'frozen research decision', 'artifact_source': 'research_refit',
              'source_run_fingerprint': PLAN_SHA, 'source_universe': f'e{CONFIG["train_min_era"]}-none-d0'}
    from stealth._runtime import make_predict

    make_predict(result)
    path = REPO / 'models' / f'{CONFIG["name"]}.pkl'
    digest = bundle.write_pickle(result, path)
    bundle.write_manifest(path, result, digest)
    write_json(RUN / 'packaging.json', {'pkl': path.name, 'sha256': digest, 'plan_sha256': PLAN_SHA,
                                       'selection_sha256': sha(RUN / 'selection.json'),
                                       'confirmation_sha256': sha(RUN / 'confirmation.json')})
    return digest


def lgbm_bytes(model):
    from stealth.models import lgbm

    return lgbm.to_bytes(model)


def report():
    selection = summarise()
    confirmation = json.loads((RUN / 'confirmation.json').read_text()) if (RUN / 'confirmation.json').exists() else None
    public = {'run_id': CONFIG['name'], 'label': selection['label'], 'comparisons': selection['comparisons'],
              'selected': selection['selected'], 'confirmation': confirmation}
    output = REPO / 'handoffs/results'
    write_json(output / f'{CONFIG["name"]}-metrics.json', public)
    lines = [f'# Research {CONFIG["name"]}', '', selection['label'], '',
             '| recipe | blend | FIT | baseline FIT | gain | p_pos | Holm pass | eligible |',
             '|---|---|---|---|---|---|---|---|']
    for row in sorted(selection['comparisons'], key=lambda r:r['paired']['diff_mean'], reverse=True):
        p = row['paired']
        lines.append(f'| {row["name"]} | {row["blend_with_baseline"]} | {row["fit_mean"]:.6f} | {row["baseline_mean"]:.6f} | '
                     f'{p["diff_mean"]:+.6f} | {p["p_pos"]:.4f} | {row["holm_pass"]} | {row["eligible"]} |')
    lines += ['', 'Frozen discovery winner: '+str(None if selection['selected'] is None else selection['selected']['name']),
              '', 'Confirmation:', '```json', json.dumps(confirmation, indent=2), '```', '',
              'No old audit eras, platform writes or staking. No candidate is an independently proven live improvement.']
    (output / f'{CONFIG["name"]}-results.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['init', 'cpu', 'select', 'confirm', 'package', 'report'], required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    load()
    if args.stage == 'init':
        log.info('immutable research plan initialised; no model scored')
    elif args.stage == 'cpu':
        cpu_discovery()
    elif args.stage == 'select':
        print(json.dumps(summarise(), indent=2))
    elif args.stage == 'confirm':
        print(json.dumps(confirm(), indent=2))
    elif args.stage == 'package':
        print(package())
    else:
        report()


if __name__ == '__main__':
    main()
