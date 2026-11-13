"""CatBoost member (CPU): inner early stopping, fixed-count refit, ``.cbm`` bytes. Every fit is timed.

Contract (docs/INTERFACES.md section 5): RMSE regression on the raw target, NaN = missing (``nan_mode``
Min), ``threads`` -> ``thread_count``, ``seed`` -> ``random_seed``. ``border_count`` from the config
(default 16). ``fit_seconds`` of the last fit is exposed via :func:`last_fit_seconds` so train.py can
apply the ``timing.catboost_max_fit_seconds`` keep/drop rule.
"""
from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

log = logging.getLogger(__name__)

DEFAULT_EARLY_STOPPING_ROUNDS = 50
_LAST_FIT_SECONDS: float = 0.0


def last_fit_seconds() -> float:
    """Wall time of the most recent fit in this process (0.0 before any fit)."""
    return _LAST_FIT_SECONDS


def _as_f32(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    return X.to_numpy(dtype=np.float32, copy=False) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)


def _build_params(params: dict, *, seed: int, threads: int, iterations: int) -> tuple[dict, int]:
    p = dict(params)
    p.pop("n_estimators", None)
    p.pop("iterations", None)
    es_rounds = int(p.pop("early_stopping_rounds", DEFAULT_EARLY_STOPPING_ROUNDS))
    base = {
        "loss_function": "RMSE",
        "iterations": int(iterations),
        "border_count": 16,
        "thread_count": int(threads),
        "random_seed": int(seed),
        "verbose": False,
        "allow_writing_files": False,
        "nan_mode": "Min",
        "task_type": "CPU",
    }
    base.update(p)
    return base, es_rounds


def _pool(X: pd.DataFrame | np.ndarray, y: Any = None, w: np.ndarray | None = None) -> Pool:
    return Pool(_as_f32(X), label=None if y is None else np.asarray(y, dtype=np.float32), weight=w)


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
    """Fit up to ``cap`` iterations with early stopping on the inner validation; (model, best_iter >= 1)."""
    global _LAST_FIT_SECONDS
    p, es_rounds = _build_params(params, seed=seed, threads=threads, iterations=int(cap))
    model = CatBoostRegressor(**p)
    t0 = time.perf_counter()
    model.fit(_pool(X_tr, y_tr, w_tr), eval_set=_pool(X_val, y_val), early_stopping_rounds=es_rounds, use_best_model=True)
    _LAST_FIT_SECONDS = time.perf_counter() - t0
    best = model.get_best_iteration()
    best = int(best) + 1 if best is not None else int(cap)
    return model, max(min(best, int(cap)), 1)


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
    """Fit exactly ``n_iter`` iterations; None -> ``params['iterations']`` / ``params['n_estimators']`` / 100."""
    global _LAST_FIT_SECONDS
    rounds = n_iter if n_iter is not None else params.get("iterations", params.get("n_estimators", 100))
    p, _ = _build_params(params, seed=seed, threads=threads, iterations=max(int(rounds), 1))
    model = CatBoostRegressor(**p)
    t0 = time.perf_counter()
    model.fit(_pool(X, y, w))
    _LAST_FIT_SECONDS = time.perf_counter() - t0
    return model


def to_bytes(model: Any) -> bytes:
    """CatBoost model -> ``.cbm`` bytes (via a temporary file; CatBoost has no in-memory save)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.cbm"
        model.save_model(str(path), format="cbm")
        return path.read_bytes()


def from_bytes(blob: bytes) -> Any:
    model = CatBoostRegressor()
    model.load_model(blob=blob)
    return model


def predict(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """float64 raw scores, len(X)."""
    return np.asarray(model.predict(_as_f32(X)), dtype=np.float64)
