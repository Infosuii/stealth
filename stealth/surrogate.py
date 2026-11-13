"""Ridge surrogate features -> v1_sherpa fitted on eras < 2800 (direction only, for the regime simulation).

Contracts: see docs/INTERFACES.md section 4. The surrogate is NEVER used for model selection, blending or
as a feature: it only supplies the benchmark *direction* that ``metrics.regime_simulation`` mixes with
noise and a target component. Eras < 2800 are never scored, so the fit is leak-free w.r.t. every fold.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Surrogate:
    """Linear map ``X_lin[columns] @ coef + intercept`` approximating v1_sherpa."""

    columns: list[str]
    coef: np.ndarray
    intercept: float

    def predict(self, X_lin: pd.DataFrame) -> pd.Series:
        """Direction only (raw linear score on ``X_lin.index``); never used for selection."""
        missing = [c for c in self.columns if c not in X_lin.columns]
        if missing:
            raise KeyError(f"surrogate: missing columns {missing[:5]}")
        x = X_lin[self.columns].to_numpy(dtype=np.float64)
        return pd.Series(x @ self.coef + self.intercept, index=X_lin.index, name="surrogate")

    def to_dict(self) -> dict:
        return {"columns": list(self.columns), "coef": self.coef.tolist(), "intercept": float(self.intercept)}

    @classmethod
    def from_dict(cls, d: dict) -> "Surrogate":
        return cls(list(d["columns"]), np.asarray(d["coef"], dtype=np.float64), float(d["intercept"]))


def fit_surrogate(
    X_lin: pd.DataFrame, bench: pd.Series, era: pd.Series, *, max_era: int = 2800, alpha: float = 1e4
) -> Surrogate:
    """Ridge(``alpha``) of ``bench`` (v1_sherpa) on the linear feature frame, rows with ``era < max_era`` only.

    Rows with a NaN benchmark are dropped. ``X_lin`` is ``data.features_linear(...)`` (all its columns are used).
    """
    from sklearn.linear_model import Ridge

    e = era.reindex(X_lin.index)
    b = bench.reindex(X_lin.index)
    keep = ((e < max_era) & b.notna()).to_numpy()
    if keep.sum() < 10:
        raise ValueError(f"fit_surrogate: only {int(keep.sum())} usable rows with era < {max_era}")
    x = X_lin.loc[keep].to_numpy(dtype=np.float64)
    y = b.loc[keep].to_numpy(dtype=np.float64)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x, y)
    r2 = float(model.score(x, y))
    log.info("fit_surrogate: %d rows, %d eras < %d, alpha=%g, in-sample R2=%.3f", len(y), int(e.loc[keep].nunique()),
             max_era, alpha, r2)
    return Surrogate(list(X_lin.columns), np.asarray(model.coef_, dtype=np.float64), float(model.intercept_))
