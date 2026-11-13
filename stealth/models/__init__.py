"""Member families (lgbm, xgb, cat, ridge, mlp): one uniform fit / to_bytes / from_bytes / predict contract.

Dispatch on ``family`` (docs/INTERFACES.md section 5). Every family module exposes
``fit_with_early_stopping``, ``fit_fixed``, ``to_bytes``, ``from_bytes``, ``predict``. The ``Member`` class
is a thin object wrapper over the same functions (``fit(X, y, w, X_es, y_es) -> self``, ``predict``,
``to_bytes`` / ``from_bytes``) for callers that prefer an object.

``mlp`` is an EXTERNAL family: it is trained by ``scripts/mlp_train.py`` in the optional GPU venv and only
serialised / predicted here (numpy forward in ``stealth._runtime``); its fit functions raise NotImplementedError.
"""
from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

Family = Literal["lgbm", "xgb", "cat", "ridge", "mlp"]
FAMILIES: tuple[str, ...] = ("lgbm", "xgb", "cat", "ridge", "mlp")
EXTERNAL_FAMILIES: tuple[str, ...] = ("mlp",)  # trained outside this package (scripts/mlp_train.py)
_MODULES = {"lgbm": "stealth.models.lgbm", "xgb": "stealth.models.xgb", "cat": "stealth.models.cat", "ridge": "stealth.models.linear",
            "mlp": "stealth.models.mlp"}
_EXT = {"lgbm": "lgbm.txt", "xgb": "ubj", "cat": "cbm", "ridge": "npz", "mlp": "npz"}


def family_module(family: str) -> ModuleType:
    """Import and return the family module; ValueError on an unknown family."""
    if family not in _MODULES:
        raise ValueError(f"unknown model family {family!r}; expected one of {FAMILIES}")
    return importlib.import_module(_MODULES[family])


def family_extension(family: str) -> str:
    """Conventional file extension for a serialised member of this family."""
    return _EXT[family]


@dataclass
class FittedMember:
    """A fitted member: serialised bytes plus the metadata the bundle needs."""

    name: str
    family: str
    target: str
    group: str
    input: str
    feature_cols: list[str]
    blob: bytes
    n_iter: int | None
    fit_seconds: float
    model: Any = field(default=None, repr=False, compare=False)  # in-memory handle, optional


def fit_member(
    spec: dict,
    X_tr: pd.DataFrame,
    y_tr: pd.Series | np.ndarray,
    w_tr: np.ndarray | None,
    *,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | np.ndarray | None = None,
    seed: int,
    threads: int,
) -> FittedMember:
    """Fit one member from its (resolved) spec.

    ``spec`` keys: name, family, target, group, input, params (dict), n_estimators (int|None), es_cap
    (int|None), feature_cols (list; defaults to ``X_tr.columns``). ``n_estimators`` set -> ``fit_fixed``;
    otherwise inner early stopping with ``es_cap`` (``X_val``/``y_val`` required). Ridge always uses
    ``fit_fixed``. Returns a :class:`FittedMember` with ``blob`` from ``to_bytes``.
    """
    family = spec["family"]
    mod = family_module(family)
    params = dict(spec.get("params") or {})
    n_est = spec.get("n_estimators")
    t0 = time.perf_counter()
    if family in EXTERNAL_FAMILIES:
        raise NotImplementedError(f"member {spec.get('name')}: family {family} is trained externally by scripts/mlp_train.py")
    if family == "ridge" or n_est is not None:
        n_iter = int(n_est) if n_est is not None else None
        model = mod.fit_fixed(params, X_tr, y_tr, w_tr, n_iter=n_iter, seed=seed, threads=threads)
    else:
        if X_val is None or y_val is None:
            raise ValueError(f"member {spec.get('name')}: early stopping needs X_val/y_val (or set n_estimators)")
        cap = spec.get("es_cap")
        if cap is None:
            raise ValueError(f"member {spec.get('name')}: es_cap is required when n_estimators is null")
        model, n_iter = mod.fit_with_early_stopping(params, X_tr, y_tr, w_tr, X_val, y_val, cap=int(cap), seed=seed, threads=threads)
    secs = time.perf_counter() - t0
    return FittedMember(
        name=str(spec.get("name", family)),
        family=family,
        target=str(spec.get("target", "")),
        group=str(spec.get("group", "")),
        input=str(spec.get("input", "nan")),
        feature_cols=[str(c) for c in spec.get("feature_cols", list(X_tr.columns))],
        blob=mod.to_bytes(model),
        n_iter=None if family == "ridge" else int(n_iter),
        fit_seconds=secs,
        model=model,
    )


def predict_member(fm: FittedMember | tuple[str, bytes], X: pd.DataFrame) -> np.ndarray:
    """Raw float64 scores for ``X`` (columns == the member's feature columns, same order)."""
    if isinstance(fm, FittedMember):
        family, model = fm.family, fm.model
        if model is None:
            model = family_module(family).from_bytes(fm.blob)
    else:
        family, blob = fm
        model = family_module(family).from_bytes(blob)
    return family_module(family).predict(model, X)


def member_to_bytes(family: str, model: Any) -> bytes:
    """Serialise an in-memory model of ``family``."""
    return family_module(family).to_bytes(model)


def member_from_bytes(family: str, blob: bytes) -> Any:
    """Deserialise a model of ``family``."""
    return family_module(family).from_bytes(blob)


class Member:
    """Object wrapper: ``Member(spec, seed=0, threads=2).fit(X, y, w, X_es, y_es).predict(X)``."""

    def __init__(self, spec: dict, *, seed: int = 0, threads: int = 1) -> None:
        self.spec = dict(spec)
        self.seed = int(seed)
        self.threads = int(threads)
        self.fitted: FittedMember | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        w: np.ndarray | None = None,
        X_es: pd.DataFrame | None = None,
        y_es: pd.Series | np.ndarray | None = None,
    ) -> "Member":
        self.fitted = fit_member(self.spec, X, y, w, X_val=X_es, y_val=y_es, seed=self.seed, threads=self.threads)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.fitted is None:
            raise RuntimeError("Member.predict called before fit")
        return predict_member(self.fitted, X)

    @property
    def n_iter(self) -> int | None:
        return None if self.fitted is None else self.fitted.n_iter

    def to_bytes(self) -> bytes:
        if self.fitted is None:
            raise RuntimeError("Member.to_bytes called before fit")
        return self.fitted.blob

    @classmethod
    def from_bytes(cls, spec: dict, blob: bytes, *, n_iter: int | None = None) -> "Member":
        m = cls(spec)
        m.fitted = FittedMember(
            name=str(spec.get("name", spec["family"])), family=spec["family"], target=str(spec.get("target", "")),
            group=str(spec.get("group", "")), input=str(spec.get("input", "nan")),
            feature_cols=[str(c) for c in spec.get("feature_cols", [])], blob=blob, n_iter=n_iter, fit_seconds=0.0,
            model=member_from_bytes(spec["family"], blob),
        )
        return m


__all__ = [
    "EXTERNAL_FAMILIES", "FAMILIES", "Family", "FittedMember", "Member", "family_extension", "family_module", "fit_member",
    "member_from_bytes", "member_to_bytes", "predict_member",
]
