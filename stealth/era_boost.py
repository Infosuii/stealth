"""Fixed tree-training ablations; all target transforms and stopping are local to the fit partition."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stealth import metrics
from stealth.models import lgbm


def centred_target(y, era):
    y = np.asarray(y, dtype=np.float64)
    _, group, counts = np.unique(era, return_inverse=True, return_counts=True)
    means = np.bincount(group, weights=y) / counts
    return y - means[group], len(y) / len(counts) / counts[group]


def fit(spec, params, X, y, era, *, seed, threads):
    """Choose a count on a purged inner window, then refit the outer training rows only."""
    import lightgbm as lgb

    X, y, era = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float64), np.asarray(era)
    unique = np.unique(era)
    if len(unique) < 30:
        raise ValueError('at least 30 eras required for the purged inner split')
    cut = unique[int(len(unique) * .8)]
    inner, valid = era < cut - 20, era >= cut
    if not inner.any() or not valid.any():
        raise ValueError('empty inner fit or validation partition')
    yi, wi = centred_target(y[inner], era[inner]) if spec['centred'] else (y[inner], None)
    yv, wv = centred_target(y[valid], era[valid]) if spec['centred'] else (y[valid], None)
    p = dict(params)
    cap = int(p.pop('cap'))
    p, patience = lgbm._build_params(p, seed=seed, threads=threads)
    feval = None
    if spec['stopping'] == 'fit':
        p['metric'] = 'None'
        target, eras = pd.Series(y[valid]), pd.Series(era[valid])
        def feval(prediction, dataset):
            score = float(metrics.per_era_fit_fast(pd.Series(prediction), target, eras).mean())
            return 'fit20', score, True
    elif spec['stopping'] != 'mse':
        raise ValueError('unknown stopping metric')
    train = lgb.Dataset(X[inner], label=yi, weight=wi, params={'max_bin':p['max_bin']})
    validation = lgb.Dataset(X[valid], label=yv, weight=wv, reference=train)
    model = lgb.train(p, train, num_boost_round=cap, valid_sets=[validation], feval=feval,
                      callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)])
    count = max(1, int(model.best_iteration or cap))
    full_y, full_w = centred_target(y, era) if spec['centred'] else (y, None)
    refit_params = dict(params)
    refit_params.pop('cap')
    model = lgbm.fit_fixed(refit_params, X, full_y, full_w, n_iter=count, seed=seed, threads=threads)
    return model, count
