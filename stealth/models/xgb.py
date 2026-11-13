"""XGBoost member (CPU ``hist``): inner early stopping, fixed-count refit, ``save_raw('ubj')`` bytes.

Contract (docs/INTERFACES.md section 5): regression (squared error) on the raw target, NaN = missing,
``threads`` -> ``nthread``, ``seed`` -> ``seed``. Trained on the float32 numpy view of ``X`` (no feature
names in the booster); ``predict`` needs the training column order.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

log = logging.getLogger(__name__)

DEFAULT_EARLY_STOPPING_ROUNDS = 50


def _as_f32(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    return X.to_numpy(dtype=np.float32, copy=False) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)


def _build_params(params: dict, *, seed: int, threads: int) -> tuple[dict, int]:
    p = dict(params)
    p.pop("n_estimators", None)
    es_rounds = int(p.pop("early_stopping_rounds", DEFAULT_EARLY_STOPPING_ROUNDS))
    base = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cpu",
        "max_bin": 16,
        "nthread": int(threads),
        "seed": int(seed),
        "verbosity": 0,
    }
    base.update(p)
    return base, es_rounds


def _dmatrix(X: pd.DataFrame | np.ndarray, y: Any = None, w: np.ndarray | None = None) -> xgb.DMatrix:
    return xgb.DMatrix(
        _as_f32(X),
        label=None if y is None else np.asarray(y, dtype=np.float32),
        weight=w,
        missing=np.nan,
    )


def fit_with_early_stopping(
    params: dict,
    X_tr: pd.DataFrame,
    y_tr: Any,
    w_tr: np.ndarray | None,
    X_val: pd.DataFrame,
    y_val: Any,
    *,
    cap: int,
    seed: int,
    threads: int,
) -> tuple[Any, int]:
    """Boost up to ``cap`` rounds with early stopping on the inner validation RMSE; (booster, best_iter >= 1)."""
    p, es_rounds = _build_params(params, seed=seed, threads=threads)
    dtr = _dmatrix(X_tr, y_tr, w_tr)
    dval = _dmatrix(X_val, y_val)
    booster = xgb.train(
        p,
        dtr,
        num_boost_round=int(cap),
        evals=[(dval, "inner")],
        early_stopping_rounds=es_rounds,
        verbose_eval=False,
    )
    best = int(getattr(booster, "best_iteration", cap - 1)) + 1  # best_iteration is 0-based
    best = max(min(best, int(cap)), 1)
    if best < booster.num_boosted_rounds():
        booster = booster[:best]  # drop the post-best rounds so predict/to_bytes agree
    return booster, best


def fit_fixed(
    params: dict,
    X: pd.DataFrame,
    y: Any,
    w: np.ndarray | None,
    *,
    n_iter: int | None,
    seed: int,
    threads: int,
) -> Any:
    """Boost exactly ``n_iter`` rounds (frozen count); None -> ``params['n_estimators']`` or 100."""
    p, _ = _build_params(params, seed=seed, threads=threads)
    rounds = int(n_iter if n_iter is not None else params.get("n_estimators", 100))
    return xgb.train(p, _dmatrix(X, y, w), num_boost_round=max(rounds, 1))


def to_bytes(model: Any) -> bytes:
    """Booster -> UBJSON bytes (early-stopped boosters are already sliced to their best iteration)."""
    return bytes(model.save_raw("ubj"))


def from_bytes(blob: bytes) -> Any:
    booster = xgb.Booster()
    booster.load_model(bytearray(blob))
    return booster


def predict(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """float64 raw scores, len(X)."""
    return np.asarray(model.predict(_dmatrix(X)), dtype=np.float64)
