"""LightGBM member: inner early stopping, fixed-count refit, ``model_to_string`` bytes.

Contract (docs/INTERFACES.md section 5): regression (L2) on the raw 5-level target, native NaN missing
(the caller passes ``data.features_nan`` frames, -1 already -> NaN), ``max_bin`` from the config
(default 16), ``threads`` -> ``num_threads``, ``seed`` -> ``seed``. Models are trained on the numpy
float32 view of ``X`` so feature names never travel with the booster; ``predict`` therefore requires the
SAME column order as training (``spec.feature_cols``).
"""
from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_EARLY_STOPPING_ROUNDS = 50


def _as_f32(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    return X.to_numpy(dtype=np.float32, copy=False) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)


def _build_params(params: dict, *, seed: int, threads: int) -> tuple[dict, int]:
    """Merge config params with the fixed defaults; returns (lgb params, early_stopping_rounds)."""
    p = dict(params)
    p.pop("n_estimators", None)
    es_rounds = int(p.pop("early_stopping_rounds", DEFAULT_EARLY_STOPPING_ROUNDS))
    base = {
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
        "max_bin": 16,
        "num_threads": int(threads),
        "seed": int(seed),
        "deterministic": True,
        "force_row_wise": True,
    }
    base.update(p)
    return base, es_rounds


def fit_with_early_stopping(
    params: dict,
    X_tr: pd.DataFrame,
    y_tr: pd.Series | np.ndarray,
    w_tr: np.ndarray | None,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    *,
    cap: int,
    seed: int,
    threads: int,
) -> tuple[Any, int]:
    """Fit up to ``cap`` trees, stopping on the inner validation L2; returns (booster, best_iter >= 1)."""
    p, es_rounds = _build_params(params, seed=seed, threads=threads)
    dtr = lgb.Dataset(_as_f32(X_tr), label=np.asarray(y_tr, dtype=np.float32), weight=w_tr, params={"max_bin": p["max_bin"]})
    dval = lgb.Dataset(_as_f32(X_val), label=np.asarray(y_val, dtype=np.float32), reference=dtr)
    booster = lgb.train(
        p,
        dtr,
        num_boost_round=int(cap),
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(es_rounds, verbose=False)],
    )
    best = int(booster.best_iteration) if booster.best_iteration else int(cap)
    return booster, max(best, 1)


def fit_fixed(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    w: np.ndarray | None,
    *,
    n_iter: int | None,
    seed: int,
    threads: int,
) -> Any:
    """Fit exactly ``n_iter`` trees (frozen count). ``n_iter`` None -> ``params['n_estimators']`` or 100."""
    p, _ = _build_params(params, seed=seed, threads=threads)
    rounds = int(n_iter if n_iter is not None else params.get("n_estimators", 100))
    dtr = lgb.Dataset(_as_f32(X), label=np.asarray(y, dtype=np.float32), weight=w, params={"max_bin": p["max_bin"]})
    return lgb.train(p, dtr, num_boost_round=max(rounds, 1))


def to_bytes(model: Any) -> bytes:
    """Booster -> UTF-8 bytes of ``model_to_string()`` (best iteration is baked in)."""
    return model.model_to_string(num_iteration=model.best_iteration or None).encode("utf-8")


def from_bytes(blob: bytes) -> Any:
    return lgb.Booster(model_str=blob.decode("utf-8"))


def predict(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """float64 raw scores, len(X). NaN inputs are routed by LightGBM's native missing handling."""
    return np.asarray(model.predict(_as_f32(X)), dtype=np.float64)
