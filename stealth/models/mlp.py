"""MLP member (family ``mlp``): numpy-only serialisation + prediction; training happens OUTSIDE this package.

Contract (docs/INTERFACES.md sections 14, 18, 19):

* the model object is the plain dict ``stealth._runtime.load_member("mlp", blob)`` returns:
  ``{"W": [float64 (in x out), ...], "b": [...], "x_mean", "x_std", "columns": [names], "act": "gelu_tanh",
  "n_layers": int}``; the bytes are the npz written by ``_runtime.mlp_to_bytes`` (``W0..``, ``b0..``, ``x_mean``,
  ``x_std`` float32, ``columns`` str, ``act``, ``n_layers``; loaded with ``allow_pickle=False``);
* ``predict`` is ``_runtime.mlp_forward`` -- the SAME function the pickled bundle runs, so the train venv,
  the GPU venv's parity check and the lane all share one forward pass;
* ``fit_with_early_stopping`` / ``fit_fixed`` raise NotImplementedError: the MLP is trained by
  ``scripts/mlp_train.py`` in the optional GPU venv (``.venv-gpu``); the deep-learning library is never imported
  by anything in this package.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from stealth import _runtime

log = logging.getLogger(__name__)

EXTERNAL_MSG = "mlp is trained externally by scripts/mlp_train.py"


def fit_with_early_stopping(*args: Any, **kwargs: Any) -> tuple[Any, int]:
    raise NotImplementedError(EXTERNAL_MSG)


def fit_fixed(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(EXTERNAL_MSG)


def to_bytes(model: dict) -> bytes:
    """npz bytes of the model dict (see module docstring)."""
    return _runtime.mlp_to_bytes(model)


def from_bytes(blob: bytes) -> dict:
    """Model dict from npz bytes (shape-checked; ``ValueError`` on a malformed blob)."""
    return _runtime.load_member("mlp", blob)


def predict(model: dict, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """float64 scores; a DataFrame has ``model["columns"]`` selected BY NAME, a matrix must be in that order."""
    return _runtime.mlp_forward(model, X)
