"""Era-balanced fixed-basis research fits. Prediction and feature transforms live in the isolated runtime."""
from __future__ import annotations

import io
import logging

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

from stealth._runtime import basis_design, basis_forward

log = logging.getLogger(__name__)


def to_bytes(model: dict) -> bytes:
    arrays = {k: np.asarray(v) for k, v in model.items() if not k.startswith('_')}
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def era_target(y: np.ndarray, era: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Within-FIT-row era standardisation and equal era mass, with mean sample weight one."""
    _, group, counts = np.unique(era, return_inverse=True, return_counts=True)
    mean = np.bincount(group, weights=y) / counts
    centered = y - mean[group]
    std = np.sqrt(np.bincount(group, weights=centered ** 2) / counts)
    target = centered / np.maximum(std[group], 1e-8)
    weights = len(y) / len(counts) / counts[group]
    return target, weights, group, counts


def gam_design(X: np.ndarray) -> sparse.csr_matrix:
    valid = np.isfinite(X) & (X >= 0) & (X <= 9)
    codes = np.where(valid, np.rint(np.where(valid, X, 0)), 10).astype(np.int32)
    codes += np.arange(X.shape[1], dtype=np.int32)[None, :] * 11
    return sparse.csr_matrix((np.ones(X.size), codes.ravel(), np.arange(0, X.size + 1, X.shape[1])),
                             shape=(len(X), X.shape[1] * 11))


def fit_basis(spec: dict, X: np.ndarray, y: np.ndarray, era: np.ndarray, columns: list[str], *, seed: int = 17,
              threads: int = 8) -> dict:
    """Solve the centred, equally era-weighted regularised objective without densifying categorical input."""
    kind, alpha = spec['kind'], float(spec['alpha'])
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError('basis alpha must be finite and positive')
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    target, weights, group, counts = era_target(y, np.asarray(era))
    model = {'kind': kind, 'columns': list(columns)}
    if kind == 'quadratic':
        model['projection'] = np.random.default_rng(seed).normal(size=(X.shape[1], int(spec['projections']))) / np.sqrt(X.shape[1])
    design = gam_design(X) if kind == 'gam' else basis_design(X, kind, model.get('projection'))
    n_features = design.shape[1]
    smooth = float(spec.get('smooth', 0.0))

    def penalty(v):
        result = alpha * v
        if smooth:
            table, output = v.reshape(-1, 11), result.reshape(-1, 11)
            d = table[:, :8] - 2 * table[:, 1:9] + table[:, 2:10]
            output[:, :8] += smooth * d
            output[:, 1:9] -= 2 * smooth * d
            output[:, 2:10] += smooth * d
        return result

    def matvec(v):
        prediction = np.asarray(design @ v).ravel()
        means = np.bincount(group, weights=prediction) / counts
        return np.asarray(design.T @ (weights * (prediction - means[group]))).ravel() + penalty(v)

    rhs = np.asarray(design.T @ (weights * target)).ravel()
    if sparse.issparse(design):
        diagonal = np.asarray(design.T @ weights).ravel() + alpha
    else:
        diagonal = np.einsum('ij,i,ij->j', design, weights, design, optimize=True) + alpha
    if smooth:
        diagonal.reshape(-1, 11)[:, :10] += smooth * np.array([1, 5, 6, 6, 6, 6, 6, 6, 5, 1])
    operator = LinearOperator((n_features, n_features), matvec=matvec)
    preconditioner = LinearOperator((n_features, n_features), matvec=lambda v: v / diagonal)
    coef, info = cg(operator, rhs, M=preconditioner, rtol=1e-7, atol=0.0, maxiter=300)
    if info != 0 or not np.isfinite(coef).all():
        raise RuntimeError(f'basis solver did not converge (info={info})')
    model['coef'] = coef.reshape(-1, 11) if kind == 'gam' else coef
    if kind == 'residual':
        import lightgbm as lgb

        residual = target - design @ coef
        ds = lgb.Dataset(X.astype(np.float32), label=residual, weight=weights)
        booster = lgb.train({'objective': 'regression', 'verbosity': -1, 'num_leaves': 4, 'learning_rate': 0.03,
                             'min_data_in_leaf': 5000, 'feature_fraction': 0.5, 'lambda_l2': 100,
                             'max_bin': 16, 'deterministic': True, 'force_row_wise': True,
                             'num_threads': threads, 'seed': seed}, ds, num_boost_round=int(spec['rounds']))
        model['residual_blob'] = np.frombuffer(booster.model_to_string().encode(), dtype=np.uint8)
    return model
