"""Ridge member on the linear input (value/9, -1 -> 0.5, missing indicators, n_missing); npz bytes.

Contract (docs/INTERFACES.md section 5): the model object is a plain dict
``{"coef": float64[n], "intercept": float64, "columns": list[str]}`` (exactly what ``np.savez`` stores and
what ``_runtime.load_member('ridge', blob)`` reconstructs). ``alpha`` (lambda) comes from the config
params (default 1e5). Ridge has no iterations: ``fit_with_early_stopping`` raises NotImplementedError.
"""
from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

log = logging.getLogger(__name__)

DEFAULT_ALPHA = 1.0e5


def fit_with_early_stopping(*args: Any, **kwargs: Any) -> tuple[Any, int]:
    raise NotImplementedError("ridge has no iterations; use fit_fixed (set n_estimators: null, es_cap: null)")


def fit_fixed(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    w: np.ndarray | None,
    *,
    n_iter: int | None = None,
    seed: int = 0,
    threads: int = 1,
) -> dict:
    """Weighted ridge with intercept; returns the coef dict. ``n_iter``/``seed``/``threads`` are ignored."""
    alpha = float(params.get("alpha", params.get("lambda", DEFAULT_ALPHA)))
    cols = [str(c) for c in X.columns]
    Xf = X.to_numpy(dtype=np.float32, copy=False)
    model = Ridge(alpha=alpha, fit_intercept=True, solver="auto", random_state=int(seed))
    model.fit(Xf, np.asarray(y, dtype=np.float64), sample_weight=None if w is None else np.asarray(w, dtype=np.float64))
    return {
        "coef": np.asarray(model.coef_, dtype=np.float64).ravel(),
        "intercept": float(np.asarray(model.intercept_).ravel()[0]),
        "columns": cols,
    }


def to_bytes(model: dict) -> bytes:
    """npz with keys coef (float64[n]), intercept (float64 scalar), columns (str[n])."""
    buf = io.BytesIO()
    np.savez(
        buf,
        coef=np.asarray(model["coef"], dtype=np.float64),
        intercept=np.float64(model["intercept"]),
        columns=np.asarray(list(model["columns"]), dtype=str),
    )
    return buf.getvalue()


def from_bytes(blob: bytes) -> dict:
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        return {
            "coef": np.asarray(z["coef"], dtype=np.float64),
            "intercept": float(z["intercept"]),
            "columns": [str(c) for c in z["columns"].tolist()],
        }


def predict(model: dict, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """float64 scores. A DataFrame is checked against the stored column list (order must match)."""
    if isinstance(X, pd.DataFrame):
        cols = [str(c) for c in X.columns]
        if cols != list(model["columns"]):
            raise ValueError("ridge predict: column mismatch vs training columns")
        Xf = X.to_numpy(dtype=np.float32, copy=False)
    else:
        Xf = np.asarray(X, dtype=np.float32)
    return Xf.astype(np.float64) @ model["coef"] + model["intercept"]
