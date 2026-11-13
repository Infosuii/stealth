"""Per-fold feature subsets (|sharpe|>0.1), core-proxy list from v1_sherpa loadings, random column bags.

Contracts: see docs/INTERFACES.md section 4. Nothing here may see test-fold eras: callers pass the
training-era rows (``sharpe_subset``) or the function restricts itself to ``era < max_era``
(``core_proxy_columns``; eras < 2800 are never scored).

Core-proxy method (documented choice): **per-era Spearman** of each feature with ``v1_sherpa``, averaged
over eras < ``max_era``, top-k by |mean|. Per-era rank correlation is preferred over a ridge fit because the
INOV reference is an *equal-weight average of raw features*: the ranking by univariate loading is what the
proxy must reproduce, and it is immune to the early-era missingness pattern (-1 is dropped per pair).
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from stealth.metrics import era_groups

log = logging.getLogger(__name__)

_MISSING = -1


def _nan_spearman_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman of each column of ``x`` [n, F] (NaN allowed) with ``y`` [n]; NaN where < 3 valid pairs."""
    n, f = x.shape
    out = np.full(f, np.nan)
    ok_y = ~np.isnan(y)
    if ok_y.sum() < 3:
        return out
    valid = ~np.isnan(x) & ok_y[:, None]
    cnt = valid.sum(axis=0)
    rx = stats.rankdata(np.where(valid, x, np.nan), axis=0, method="average", nan_policy="omit")
    rx = np.where(valid, rx, 0.0)
    # y re-ranked within each column's valid set: invalid rows are pushed to +inf so they rank last and
    # never tie with a valid row; their ranks are then zeroed out.
    ry = stats.rankdata(np.where(valid, y[:, None], np.inf), axis=0, method="average")
    ry = np.where(valid, ry, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = rx.sum(axis=0) / cnt
        my = ry.sum(axis=0) / cnt
        dx = np.where(valid, rx - mx, 0.0)
        dy = np.where(valid, ry - my, 0.0)
        cov = (dx * dy).sum(axis=0)
        den = np.sqrt((dx * dx).sum(axis=0) * (dy * dy).sum(axis=0))
        rho = np.where(den > 0, cov / np.where(den > 0, den, 1.0), np.nan)
    rho[cnt < 3] = np.nan
    return rho


def per_era_feature_spearman(x: pd.DataFrame, y: pd.Series, era: pd.Series) -> pd.DataFrame:
    """``[era x feature]`` per-era Spearman of every column of ``x`` with ``y`` (NaN/-1 in ``x`` dropped per pair).

    ``x`` may be the raw int8 frame (-1 treated as missing) or the NaN frame.
    """
    xv = x.to_numpy(dtype=np.float64, copy=True)
    xv[xv == _MISSING] = np.nan
    yv = y.reindex(x.index).to_numpy(dtype=np.float64)
    rows, eras = [], []
    for e, idx in era_groups(era.reindex(x.index)):
        rows.append(_nan_spearman_matrix(xv[idx], yv[idx]))
        eras.append(e)
    return pd.DataFrame(np.vstack(rows), index=pd.Index(np.asarray(eras, dtype=np.int32), name="era"),
                        columns=list(x.columns))


def sharpe_subset(X_nan: pd.DataFrame, y: pd.Series, era: pd.Series, threshold: float = 0.1) -> list[str]:
    """Features whose per-era Spearman with ``y`` has ``|mean/std| > threshold`` on the GIVEN rows only.

    The caller passes training-era rows (nested inside the fold). Returns names in ``X_nan`` column order.
    """
    corr = per_era_feature_spearman(X_nan, y, era)
    mean = corr.mean(axis=0, skipna=True)
    std = corr.std(axis=0, ddof=1, skipna=True)
    sharpe = (mean / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    keep = [c for c in X_nan.columns if abs(float(sharpe[c])) > threshold]
    log.info("sharpe_subset(threshold=%.2f): %d/%d features kept on %d eras", threshold, len(keep),
             X_nan.shape[1], corr.shape[0])
    return keep


def feature_loadings(X_nan: pd.DataFrame, bench: pd.Series, era: pd.Series, *, max_era: int = 2800) -> pd.Series:
    """Mean per-era Spearman(feature, bench) over eras < ``max_era`` (sorted by |value| desc)."""
    e = era.reindex(X_nan.index)
    keep = (e < max_era).to_numpy()
    if keep.sum() == 0:
        raise ValueError(f"no rows with era < {max_era}")
    corr = per_era_feature_spearman(X_nan.loc[keep], bench, e.loc[keep])
    mean = corr.mean(axis=0, skipna=True).fillna(0.0)
    return mean.reindex(mean.abs().sort_values(ascending=False, kind="stable").index).rename("loading")


def core_proxy_columns(
    X_nan: pd.DataFrame, bench: pd.Series, era: pd.Series, *, k: int = 10, max_era: int = 2800
) -> list[str]:
    """Top-``k`` features by |mean per-era Spearman(feature, v1_sherpa)| on eras < ``max_era`` ONLY.

    Those eras are never scored, so the list is leak-free with respect to every CV fold. Returned in
    descending |loading| order (names only).
    """
    load = feature_loadings(X_nan, bench, era, max_era=max_era)
    cols = list(load.index[:k])
    log.info("core_proxy_columns(k=%d, max_era=%d): %s", k, max_era,
             ", ".join(f"{c.removeprefix('feature_')}{load[c]:+.3f}" for c in cols))
    return cols


def core_proxy_series(features: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Row mean over ``cols`` ignoring -1/NaN (same rule as ``scoring.core_feature_average``); all-missing -> NaN."""
    x = features[list(cols)].to_numpy(dtype=np.float64, copy=True)
    x[x == _MISSING] = np.nan
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-missing rows -> NaN by design
        m = np.nanmean(x, axis=1)
    return pd.Series(m, index=features.index, name="core_proxy")


def random_column_bags(cols: list[str], *, n_bags: int = 4, fraction: float = 0.5, seed: int = 0) -> list[list[str]]:
    """``n_bags`` seeded random subsets of ``round(fraction*len(cols))`` columns, each in original column order."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0,1]")
    rng = np.random.default_rng(seed)
    n = max(1, int(round(fraction * len(cols))))
    bags = []
    for _ in range(n_bags):
        pick = np.sort(rng.choice(len(cols), size=n, replace=False))
        bags.append([cols[i] for i in pick])
    return bags


def resolve_feature_set(
    spec: str | list[str],
    all_cols: list[str],
    *,
    sharpe_cols: list[str] | None = None,
    bags: list[list[str]] | None = None,
) -> list[str]:
    """``"all" | "sharpe_subset" | "bag:<i>" | [names]`` -> concrete column list (validated against ``all_cols``)."""
    all_cols = list(all_cols)
    if isinstance(spec, str):
        if spec == "all":
            return all_cols
        if spec == "sharpe_subset":
            if sharpe_cols is None:
                raise ValueError("features: 'sharpe_subset' requested but sharpe_cols not supplied")
            cols = list(sharpe_cols)
        elif spec.startswith("bag:"):
            if bags is None:
                raise ValueError("features: 'bag:<i>' requested but bags not supplied")
            i = int(spec.split(":", 1)[1])
            if not 0 <= i < len(bags):
                raise ValueError(f"features: bag index {i} out of range (have {len(bags)})")
            cols = list(bags[i])
        else:
            raise ValueError(f"features: unknown spec {spec!r}")
    else:
        cols = list(spec)
    unknown = [c for c in cols if c not in set(all_cols)]
    if unknown:
        raise ValueError(f"features: unknown columns {unknown[:5]}{'...' if len(unknown) > 5 else ''}")
    if not cols:
        raise ValueError("features: empty column list")
    return cols
