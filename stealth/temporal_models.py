"""Chronological adaptation and explicitly labeled composite-score proxies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stealth import metrics
from stealth.models import lgbm


def blocks(windows, cadence, gap):
    if cadence < 1 or gap < 40:
        raise ValueError('positive cadence and at least 40-era availability gap required')
    out = []
    for w in windows:
        for start in range(w['test_start'], w['test_end'] + 1, cadence):
            out.append(dict(name=f'{w["name"]}_{start}', window=w['name'], train_end=start-gap,
                            test_start=start, test_end=min(start+cadence-1, w['test_end'])))
    return out


def rank_by_era(values, era):
    """SDK-compatible ranks, preserving caller row order and tied/constant scores."""
    values = np.asarray(values, dtype=float)
    out = np.empty(len(values))
    for _, ids in metrics.era_groups(pd.Series(np.asarray(era))):
        out[ids] = metrics.rank_gauss_rows(values[ids][None, :])[0]
    return out


def core_mean(X):
    X = np.asarray(X, dtype=float)
    ok = np.isfinite(X) & (X >= 0)
    n = ok.sum(axis=1)
    return np.divide(np.where(ok, X, 0).sum(axis=1), n, out=np.full(len(X), np.nan), where=n>0)


def score_target(y, era, core):
    """FIT+INOV linear utility direction; all operations are within available fit eras."""
    y, era, core = np.asarray(y, float), np.asarray(era), np.asarray(core, float)
    target = np.empty(len(y))
    centered = np.empty(len(y))
    for _, ids in metrics.era_groups(pd.Series(era)):
        tc = y[ids] - y[ids].mean()
        centered[ids] = tc
        c = rank_by_era(core[ids], era[ids])
        c = np.nan_to_num(c)
        mm, cov = c @ c, c @ tc
        target[ids] = tc + (tc - c * cov / mm if mm > 0 and cov >= 0 else 0)
    sd = float(target.std())
    return target * (float(centered.std()) / sd) if sd > 0 else target


def composite(prediction, target, era, benchmark, core, *, bound, neutralize=0.):
    """Exact metric kernels for an OOS benchmark and chosen core proxy; no simulated benchmark labels."""
    if not np.isfinite(bound) or bound <= 0 or not 0 <= neutralize <= 1:
        raise ValueError('invalid score bound/neutralization')
    era = pd.Series(np.asarray(era))
    z, eras = metrics.pad_by_era(np.asarray(prediction), era)
    y, _ = metrics.pad_by_era(np.asarray(target), era)
    b, _ = metrics.pad_by_era(np.asarray(benchmark), era)
    c, _ = metrics.pad_by_era(np.asarray(core), era)
    if not np.isfinite(np.asarray(target)).all() or not np.isfinite(np.asarray(prediction)).all() or not np.isfinite(np.asarray(benchmark)).all():
        raise ValueError('complete finite predictions, targets and benchmark required')
    z, b, c = (metrics.rank_gauss_rows(a) for a in (z, b, c))
    z = metrics.rank_gauss_rows(metrics.neutralise_rows(z, c, neutralize))
    tc = y - np.nanmean(y, axis=1, keepdims=True)
    fit, unq = metrics.fit_rows(z, tc), metrics.unq_rows(z, b, tc)
    ok = ~np.isnan(c)
    yi = np.where(ok, y, np.nan)
    # Avoid warning-prone all-missing core eras; INOV is unavailable there, not invented as zero.
    if np.any(ok.sum(axis=1) == 0):
        raise ValueError('core proxy unavailable for a complete era')
    P = dict(core_ok=ok, core_partial=np.any(~ok & ~np.isnan(y), axis=1), c_inov=c,
             tc_inov=yi-np.nanmean(yi, axis=1, keepdims=True))
    inov = metrics.inov_rows(z, P)
    score = bound * np.arctan((fit+unq+inov)/bound)
    return pd.DataFrame(dict(fit=fit, unq=unq, inov=inov, score=score), index=eras)


def stopping_scorer(target, era, benchmark, core, bound):
    """Cache the target/reference layout, so boosting callbacks only rank new predictions."""
    era = np.asarray(era)
    unique, group, counts = np.unique(era, return_inverse=True, return_counts=True)
    offsets = np.empty(len(era), dtype=int)
    for i in range(len(unique)):
        offsets[group==i] = np.arange(counts[i])
    def pack(values):
        out = np.full((len(unique), counts.max()), np.nan)
        out[group,offsets] = values
        return out
    y = pack(target)
    tc = y-np.nanmean(y,axis=1,keepdims=True)
    b,c = metrics.rank_gauss_rows(pack(benchmark)),metrics.rank_gauss_rows(pack(core))
    ok = ~np.isnan(c)
    if np.any(ok.sum(axis=1)==0):
        raise ValueError('core proxy unavailable for a complete era')
    yi = np.where(ok,y,np.nan)
    P = dict(core_ok=ok,core_partial=np.any(~ok & ~np.isnan(y),axis=1),c_inov=c,
             tc_inov=yi-np.nanmean(yi,axis=1,keepdims=True))
    def score(prediction):
        z = metrics.rank_gauss_rows(pack(prediction))
        total = metrics.fit_rows(z,tc)+metrics.unq_rows(z,b,tc)+metrics.inov_rows(z,P)
        return float(np.mean(bound*np.arctan(total/bound)))
    return score


def fixed_fit(spec, params, X, y, era, count, *, seed, threads):
    """Matched target/count ablation: same available history, a frozen donor stopping count."""
    X,y,era = np.asarray(X,np.float32),np.asarray(y,float),np.asarray(era)
    keep = np.ones(len(era),bool) if spec['history'] is None else era > era.max()-spec['history']
    X,y,era = X[keep],y[keep],era[keep]
    target = score_target(y,era,core_mean(X)) if spec['score_aware'] else y
    weight = None if spec['half_life'] is None else np.exp2((era-era.max())/spec['half_life'])
    p = dict(params)
    p.pop('cap',None)
    return lgbm.fit_fixed(p,X,target,weight,n_iter=count,seed=seed,threads=threads)


def fit(spec, params, X, y, era, *, inner_width, gap, benchmark_trees, bound, seed, threads):
    """Own-inner stopping, then available-history refit. No prediction-period targets accepted."""
    import lightgbm as lgb
    X, y, era = np.asarray(X, np.float32), np.asarray(y, float), np.asarray(era)
    keep = np.ones(len(era), bool) if spec['history'] is None else era > era.max()-spec['history']
    X, y, era = X[keep], y[keep], era[keep]
    unique = np.unique(era)
    if len(unique) < inner_width + gap + 200:
        raise ValueError('insufficient history for purged stopping')
    cut = unique[-inner_width]
    inner, valid = era < cut-gap, era >= cut
    core = core_mean(X)
    def weights(e):
        return None if spec['half_life'] is None else np.exp2((e-e.max())/spec['half_life'])
    p = dict(params)
    cap = int(p.pop('cap'))
    train_y = score_target(y[inner], era[inner], core[inner]) if spec['score_aware'] else y[inner]
    feval = None
    benchmark = None
    if spec['score_aware']:
        benchmark = lgbm.fit_fixed(p, X[inner], y[inner], None, n_iter=benchmark_trees, seed=seed, threads=threads)
        ref = lgbm.predict(benchmark, X[valid])
        scorer = stopping_scorer(y[valid],era[valid],ref,core[valid],bound)
        p['metric'] = 'None'
        def feval(prediction, dataset):
            return 'bounded_composite_proxy', scorer(prediction), True
    built, patience = lgbm._build_params(p, seed=seed, threads=threads)
    dtr = lgb.Dataset(X[inner], label=train_y, weight=weights(era[inner]), params={'max_bin':built['max_bin']})
    dv = lgb.Dataset(X[valid], label=y[valid], reference=dtr)
    m = lgb.train(built, dtr, num_boost_round=cap, valid_sets=[dv], feval=feval,
                  callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)])
    count = max(1, int(m.best_iteration or cap))
    full_y = score_target(y, era, core) if spec['score_aware'] else y
    model = lgbm.fit_fixed(p, X, full_y, weights(era), n_iter=count, seed=seed, threads=threads)
    info = dict(n_iter=count, train_min=int(era.min()), train_max=int(era.max()),
                inner_fit_max=int(era[inner].max()), inner_valid_min=int(era[valid].min()),
                inner_valid_max=int(era[valid].max()), stopping='composite_proxy' if feval else 'mse',
                best_score=float(next(iter(m.best_score['valid_0'].values()))),
                training_rows=len(y), benchmark_fit_max=int(era[inner].max()) if benchmark else None)
    return model, info
