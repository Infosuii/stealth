#!/usr/bin/env python3
"""Era-aware residual neural research in the separate GPU venv; export numeric arrays only."""
from __future__ import annotations

import argparse
import copy
import logging
import os
from pathlib import Path
import sys
import time

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research_run as research
from stealth._runtime import basis_design, basis_forward
from stealth.research_models import fit_basis, to_bytes, era_target
from stealth.metrics import per_era_fit_fast

log = logging.getLogger('research-gpu')


def net(spec, n):
    hidden = int(spec['hidden'])
    result = nn.Sequential(nn.Linear(n, hidden), nn.GELU(approximate='tanh'), nn.Dropout(spec['dropout']),
                           nn.Linear(hidden, hidden // 2), nn.GELU(approximate='tanh'), nn.Dropout(spec['dropout']),
                           nn.Linear(hidden // 2, 1))
    nn.init.zeros_(result[-1].weight)
    nn.init.zeros_(result[-1].bias)
    return result.cuda()


def padded(X, y, era, skip, mean, std):
    order = np.argsort(era, kind='stable')
    X, y, era = X[order], y[order], era[order]
    unique, inv, count = np.unique(era, return_inverse=True, return_counts=True)
    width = int(count.max())
    positions = np.arange(len(era)) - np.repeat(np.cumsum(count) - count, count)
    features = np.zeros((len(unique), width, X.shape[1]), dtype=np.float32)
    targets, base, mask = [np.zeros((len(unique), width), dtype=np.float32) for _ in range(3)]
    features[inv, positions] = (X - mean) / std
    targets[inv, positions] = era_target(y, era)[0]
    base[inv, positions] = X @ skip
    mask[inv, positions] = 1
    return [torch.as_tensor(a, device='cuda') for a in (features, targets, base, mask)]


def export(model, skip, mean, std, columns):
    result = {'kind': 'net', 'columns': columns, 'coef': skip, 'x_mean': mean, 'x_std': std}
    for i, layer in enumerate([model[0], model[3], model[6]]):
        result[f'W{i}'] = layer.weight.detach().cpu().numpy().T.copy().astype(np.float64)
        result[f'b{i}'] = layer.bias.detach().cpu().numpy().copy().astype(np.float64)
    return result


def fit_one(spec, X, y, era, columns, seed, epochs, valid=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    design = basis_design(X, 'linear')
    skip = fit_basis({'kind': 'linear', 'alpha': spec['alpha']}, X, y, era, columns, seed=seed)['coef']
    mean, std = design.mean(axis=0), design.std(axis=0)
    std = np.maximum(std, 0.1)
    batches = padded(design, y, era, skip, mean, std)
    model = net(spec, design.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['lr'], weight_decay=spec['weight_decay'])
    best, best_epoch, state, stale = -float('inf'), 1, None, 0
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        for ids in np.array_split(rng.permutation(len(batches[0])), max(1, int(np.ceil(len(batches[0]) / spec['batch_eras'])))):
            inputs, target, base, mask = [a[ids] for a in batches]
            pred = model(inputs).squeeze(-1) + base
            count = mask.sum(dim=1).clamp_min(1)
            pred = (pred - (pred * mask).sum(dim=1)[:, None] / count[:, None]) * mask
            covariance = (pred * target).sum(dim=1) / count
            scale = ((pred ** 2).sum(dim=1) / count + 1e-6).sqrt()
            # Equal era mass, direct correlation reward, and a small scale anchor.
            loss = -(covariance / scale).mean() + 0.01 * (((pred - target) ** 2).sum(dim=1) / count).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if valid is not None:
            model.eval()
            current = export(model, skip, mean, std, columns)
            prediction = basis_forward(current, valid[0])
            score = float(per_era_fit_fast(pd.Series(prediction), pd.Series(valid[1]), pd.Series(valid[2])).mean())
            if score > best + 1e-8:
                best, best_epoch, stale = score, epoch, 0
                state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if epoch % 5 == 0 or stale >= spec['patience']:
                log.info('seed %d inner epoch %d best epoch %d FIT %.5f', seed, epoch, best_epoch, best)
            if stale >= spec['patience']:
                break
    if valid is not None:
        model.load_state_dict(state)
    model.eval()
    result = export(model, skip, mean, std, columns)
    # Seed outputs use a scale fitted only on training features, never on outer-test scores.
    scale = max(float(np.std(basis_forward(result, X))), 1e-6)
    result['coef'] = result['coef'] / scale
    result['W2'] = result['W2'] / scale
    result['b2'] = result['b2'] / scale
    # Verify the exported arithmetic before scoring an outer window.
    with torch.no_grad():
        xx = torch.as_tensor((design[:256] - mean) / std, dtype=torch.float32, device='cuda')
        reference = (model(xx).squeeze(-1).cpu().numpy() + design[:256] @ skip) / scale
    exported = basis_forward(result, X[:256])
    error = float(np.max(np.abs(exported - reference)))
    if error > 1e-4:
        raise RuntimeError(f'neural export parity failure {error}')
    del batches, optimizer, model, design
    torch.cuda.empty_cache()
    return result, best_epoch


def fit_window(window, spec):
    if research.cached(window, spec['name']):
        log.info('%s/%s reused', window['name'], spec['name'])
        return
    start = time.perf_counter()
    columns, tr, te, X, Xt = research.arrays(window)
    y, era = tr.target_everest.to_numpy(), tr.era.to_numpy()
    eras = np.unique(era)
    cut = int(eras[int(len(eras) * 0.8)])
    inner, valid = era < cut - 20, era >= cut
    blobs, predictions, counts = [], [], []
    for seed in spec['seeds']:
        _, count = fit_one(spec, X[inner], y[inner], era[inner], columns, seed, spec['epochs'],
                          valid=(X[valid], y[valid], era[valid]))
        model, _ = fit_one(spec, X, y, era, columns, seed, count)
        counts.append(count)
        predictions.append(basis_forward(model, Xt))
        blobs.append(to_bytes(model))
        log.info('%s seed %d done: frozen epochs %d', window['name'], seed, count)
    research.save_fit(window, spec['name'], 'basis', blobs, np.mean(predictions, axis=0), te,
                      time.perf_counter() - start, {'epochs': counts, 'device': torch.cuda.get_device_name(0)})
    log.info('%s/%s complete in %.1fs', window['name'], spec['name'], time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmation', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU fallback')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(2)
    if args.smoke:
        rng=np.random.default_rng(17)
        X=rng.integers(0,10,size=(256,4)).astype(float)
        y=rng.normal(size=256)
        era=np.repeat(np.arange(16),16)
        spec={'kind':'net','alpha':100,'hidden':16,'dropout':.25,'weight_decay':.01,'lr':.0003,'batch_eras':8,'patience':2}
        model,_=fit_one(spec,X,y,era,['a','b','c','d'],17,2)
        assert np.isfinite(basis_forward(model,X)).all()
        log.info('CUDA research smoke PASSED: numeric export matches inference')
        return
    research.load()
    spec = next(s for s in research.CONFIG['candidates'] if s['kind'] == 'net')
    if args.full:
        import json

        confirmation = json.loads((research.RUN / 'confirmation.json').read_text())
        if not confirmation['promoted'] or confirmation['selected'] != spec['name']:
            raise RuntimeError('neural candidate was not promoted')
        columns = [c for c in research.FRAME if c.startswith('feature_')]
        frame = research.FRAME
        X = frame[columns].to_numpy(dtype=np.float64, copy=True)
        X[X < 0] = np.nan
        y, era = frame.target_everest.to_numpy(), frame.era.to_numpy()
        unique = np.unique(era)
        cut = unique[int(len(unique)*0.8)]
        inner, valid = era < cut-20, era >= cut
        blobs, counts = [], []
        for seed in spec['seeds']:
            _, count = fit_one(spec, X[inner], y[inner], era[inner], columns, seed, spec['epochs'],
                               valid=(X[valid], y[valid], era[valid]))
            model, _ = fit_one(spec, X, y, era, columns, seed, count)
            blobs.append(to_bytes(model))
            counts.append(count)
        research.save_fit({'name':'full','train_end':5925,'test_start':5925,'test_end':5924}, spec['name'], 'basis',
                          blobs, np.empty(0), frame.iloc[:0], 0, {'epochs':counts,'test_eras':None,'purpose':'refit only'})
        return
    if args.confirmation:
        selected = research.summarise()['selected']
        if not selected or selected['name'] != spec['name']:
            raise RuntimeError('neural candidate is not the frozen discovery winner')
        windows = [research.CONFIG['confirmation']]
    else:
        windows = research.CONFIG['discovery']
    for window in windows:
        fit_window(window, spec)


if __name__ == '__main__':
    main()
