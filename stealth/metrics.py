"""fit20 / unq20 / inov wrappers per era, summaries, leak + winner's-curse gates, regime simulation.

Contracts: see docs/INTERFACES.md section 3.

* ``per_era_*`` call ``everestapi.scoring`` once per era (one call = one cross-section). If an SDK call
  raises, the documented local re-implementation (``_fit20_local`` / ``_unq20_local``) is used for that era
  and a warning is logged once; ``tests/test_metrics.py`` asserts parity between the two.
* Zero rule: ``scoring.unq20`` (0.4.1) already returns 0 on an era where the benchmark's own covariance with
  the target is negative. ``per_era_unq`` does NOT re-apply it. The simulation's vectorised kernel applies
  the same rule (``bench_fit < 0 -> 0``).
* The regime simulation is a vectorised numpy re-implementation of the same kernels over a padded
  ``[n_eras, max_rows]`` array (an SDK call per era x draw x p_b would be millions of calls).
* ``regime_grid`` (INTERFACES section 17) runs the simulation over ``p_b_grid x p_c_grid`` with COMMON RANDOM
  NUMBERS (one benchmark draw serves every cell); ``regime_simulation(p_c=x)`` is its single-``p_c`` slice in the
  M1 shape. Every regime block carries ``score_sharpe`` (staking view: mean over draws of the across-era Sharpe of
  fit + unq + inov, benchmark noise kept in the variance) and its era block-bootstrap SE.
* ``paired_sharpe_bootstrap`` is the Sharpe analogue of ``cv.paired_block_bootstrap`` (identical block indices for
  both series) used by the admission ladder's Sharpe route.
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from stealth.cv import block_bootstrap_means

log = logging.getLogger(__name__)

try:  # the SDK is pinned in the venv; keep the import guarded so offline unit tests still run
    from everestapi import scoring as _sdk
except Exception:  # pragma: no cover - import guard
    _sdk = None  # type: ignore[assignment]

_CLIP = (0.001, 0.999)
_fallback_warned: set[str] = set()


# --------------------------------------------------------------------------------------------------
# local re-implementations (documented fallbacks, parity-tested against the SDK)
# --------------------------------------------------------------------------------------------------
def _rank_gauss_1d(x: np.ndarray) -> np.ndarray:
    """``norm.ppf(clip((rank_avg - 0.5)/n, .001, .999))`` -- exactly the SDK's transform."""
    r = (stats.rankdata(x, method="average") - 0.5) / len(x)
    return stats.norm.ppf(np.clip(r, *_CLIP))


def _fit20_local(pred: np.ndarray, target: np.ndarray) -> float:
    """Local FIT20: covariance of rank-gaussianised ``pred`` with the centred target (SDK semantics)."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    ok = ~(np.isnan(p) | np.isnan(t))
    p, t = p[ok], t[ok]
    if len(p) < 3 or np.std(p, ddof=1) == 0.0:
        return 0.0
    z = _rank_gauss_1d(p)
    cov = float(np.dot(z - z.mean(), t - t.mean()) / len(p))
    return 0.0 if np.isnan(cov) else cov


def _unq20_local(pred: np.ndarray, bench: np.ndarray, target: np.ndarray) -> float:
    """Local UNQ20 incl. the zero rule (benchmark covariance with target < 0 -> 0)."""
    p = np.asarray(pred, dtype=np.float64)
    m = np.asarray(bench, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    ok = ~(np.isnan(p) | np.isnan(m) | np.isnan(t))
    p, m, t = p[ok], m[ok], t[ok]
    n = len(p)
    if n < 5:
        return 0.0
    pg, mg = _rank_gauss_1d(p), _rank_gauss_1d(m)
    tc = t - t.mean()
    if float((mg - mg.mean()) @ tc) / n < 0.0:
        return 0.0
    mm = mg @ mg
    if mm == 0.0:
        return 0.0
    ortho = pg - mg * (mg @ pg) / mm
    return float(tc @ ortho) / n


def _warn_fallback(which: str, exc: Exception) -> None:
    if which not in _fallback_warned:
        _fallback_warned.add(which)
        log.warning("everestapi.scoring.%s failed (%r); using the local implementation", which, exc)


def fit20(pred: pd.Series, target: pd.Series) -> float:
    """One-era FIT20 via the SDK (local fallback on failure)."""
    if _sdk is not None:
        try:
            return float(_sdk.fit20(pred, target))
        except Exception as exc:  # noqa: BLE001
            _warn_fallback("fit20", exc)
    return _fit20_local(pred.to_numpy(), target.to_numpy())


def unq20(pred: pd.Series, bench: pd.Series, target: pd.Series) -> float:
    """One-era UNQ20 via the SDK (zero rule inside the SDK; local fallback on failure)."""
    if _sdk is not None:
        try:
            return float(_sdk.unq20(pred, bench, target))
        except Exception as exc:  # noqa: BLE001
            _warn_fallback("unq20", exc)
    return _unq20_local(pred.to_numpy(), bench.to_numpy(), target.to_numpy())


def inov(pred: pd.Series, core_frame: pd.DataFrame, target: pd.Series) -> float:
    """One-era INOV via the SDK: UNQ with the core-feature average (-1/NaN dropped per row) as reference."""
    if _sdk is not None:
        try:
            return float(_sdk.inov(pred, core_frame, target))
        except Exception as exc:  # noqa: BLE001
            _warn_fallback("inov", exc)
    num = core_frame.apply(pd.to_numeric, errors="coerce")
    avg = num.mask(num == -1).mean(axis=1, skipna=True)
    return _unq20_local(pred.to_numpy(), avg.to_numpy(), target.to_numpy())


# --------------------------------------------------------------------------------------------------
# per-era grouping
# --------------------------------------------------------------------------------------------------
def era_groups(era: pd.Series) -> list[tuple[int, np.ndarray]]:
    """``[(era_value, positional_indices)]`` sorted by era; positions refer to ``era``'s order."""
    e = era.to_numpy()
    order = np.argsort(e, kind="stable")
    uniq, starts = np.unique(e[order], return_index=True)
    bounds = list(starts) + [len(e)]
    return [(int(uniq[i]), order[bounds[i]:bounds[i + 1]]) for i in range(len(uniq))]


def _aligned(s: pd.Series, index: pd.Index, name: str) -> pd.Series:
    """Reindex ``s`` to ``index`` (contract: same ids), raising if nothing lines up."""
    if s.index.equals(index):
        return s
    out = s.reindex(index)
    if out.notna().sum() == 0:
        raise ValueError(f"{name} shares no ids with the era index")
    return out


def _era_series(values: list[float], eras: list[int], name: str) -> pd.Series:
    idx = pd.Index(np.asarray(eras, dtype=np.int32), name="era")
    return pd.Series(np.asarray(values, dtype=np.float64), index=idx, name=name)


def per_era_fit(pred: pd.Series, target: pd.Series, era: pd.Series) -> pd.Series:
    """``scoring.fit20`` per era -> Series indexed by era (int32, sorted), name ``fit``."""
    pred = _aligned(pred, era.index, "pred")
    target = _aligned(target, era.index, "target")
    eras, vals = [], []
    for e, idx in era_groups(era):
        eras.append(e)
        vals.append(fit20(pred.iloc[idx], target.iloc[idx]))
    return _era_series(vals, eras, "fit")


def per_era_unq(pred: pd.Series, bench: pd.Series, target: pd.Series, era: pd.Series) -> pd.Series:
    """``scoring.unq20`` per era (zero rule applied INSIDE the SDK) -> Series named ``unq``."""
    pred = _aligned(pred, era.index, "pred")
    bench = _aligned(bench, era.index, "bench")
    target = _aligned(target, era.index, "target")
    eras, vals = [], []
    for e, idx in era_groups(era):
        eras.append(e)
        vals.append(unq20(pred.iloc[idx], bench.iloc[idx], target.iloc[idx]))
    return _era_series(vals, eras, "unq")


def per_era_inov(pred: pd.Series, core_frame: pd.DataFrame, target: pd.Series, era: pd.Series) -> pd.Series:
    """``scoring.inov`` per era with ``core_frame`` = raw int8 core columns (-1 ok) -> Series named ``inov``."""
    pred = _aligned(pred, era.index, "pred")
    target = _aligned(target, era.index, "target")
    core = core_frame if core_frame.index.equals(era.index) else core_frame.reindex(era.index)
    eras, vals = [], []
    for e, idx in era_groups(era):
        eras.append(e)
        vals.append(inov(pred.iloc[idx], core.iloc[idx], target.iloc[idx]))
    return _era_series(vals, eras, "inov")


def per_era_bench_fit(bench: pd.Series, target: pd.Series, era: pd.Series) -> pd.Series:
    """FIT20 of the benchmark itself per era (the quantity the zero rule gates on), name ``bench_fit``."""
    return per_era_fit(bench, target, era).rename("bench_fit")


def per_era_spearman(pred: pd.Series, target: pd.Series, era: pd.Series) -> pd.Series:
    """Spearman(pred, target) per era (NaN pairs dropped; 0.0 when undefined), name ``spearman``."""
    pred = _aligned(pred, era.index, "pred")
    target = _aligned(target, era.index, "target")
    p_all = pred.to_numpy(dtype=np.float64)
    t_all = target.to_numpy(dtype=np.float64)
    eras, vals = [], []
    for e, idx in era_groups(era):
        p, t = p_all[idx], t_all[idx]
        ok = ~(np.isnan(p) | np.isnan(t))
        p, t = p[ok], t[ok]
        if len(p) < 3 or np.std(p) == 0.0 or np.std(t) == 0.0:
            rho = 0.0
        else:
            rho = float(stats.spearmanr(p, t).statistic)
            if np.isnan(rho):
                rho = 0.0
        eras.append(e)
        vals.append(rho)
    return _era_series(vals, eras, "spearman")


def per_era_corr(a: pd.Series, b: pd.Series, era: pd.Series) -> pd.Series:
    """Per-era Spearman between two prediction series (e.g. ours vs v1_sherpa on validation = rho_b)."""
    return per_era_spearman(a, b, era).rename("corr")


# --------------------------------------------------------------------------------------------------
# summaries and gates
# --------------------------------------------------------------------------------------------------
def summarise(s: pd.Series, *, block: int = 20, n_boot: int = 2000, seed: int = 0) -> dict:
    """``{"mean","std","sharpe","pct_pos","frac_roll20_neg","ci90":[lo,hi],"se_block","n_eras"}`` of a per-era series.

    ``sharpe`` = mean/std per era (NOT annualised); ``frac_roll20_neg`` = fraction of 20-era rolling means
    below zero (the payout unit); ``ci90`` = unpaired block-bootstrap interval of the mean; ``se_block`` = the
    standard deviation of the ``block``-era block-bootstrap means -- the SE of the mean that respects the ~20-era
    autocorrelation of a 20-day target. The naive ``std/sqrt(n_eras)`` is NOT reported anywhere (it is ~3x too small).
    """
    v = pd.Series(s).dropna().sort_index()
    n = int(len(v))
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "sharpe": float("nan"), "pct_pos": float("nan"),
                "frac_roll20_neg": float("nan"), "ci90": [float("nan"), float("nan")], "se_block": float("nan"), "n_eras": 0}
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else float("nan")
    sharpe = mean / std if std and std > 0 else float("nan")
    roll = v.rolling(20, min_periods=20).mean().dropna()
    frac_neg = float((roll < 0).mean()) if len(roll) else float("nan")
    if n > 1 and n_boot > 1:
        means = block_bootstrap_means(v.to_numpy(dtype=np.float64), block=block, n_boot=n_boot, seed=seed)
        lo, hi = np.percentile(means, [5.0, 95.0])
        se_block = float(means.std(ddof=1))
    else:
        lo, hi, se_block = mean, mean, float("nan")
    return {
        "mean": mean,
        "std": std,
        "sharpe": float(sharpe),
        "pct_pos": float((v > 0).mean()),
        "frac_roll20_neg": frac_neg,
        "ci90": [float(lo), float(hi)],
        "se_block": se_block,
        "n_eras": n,
    }


def leak_check(spearman: pd.Series, threshold: float = 0.18) -> None:
    """Raise ``RuntimeError("LEAK: ...")`` if the mean OOF per-era Spearman exceeds ``threshold``."""
    mean = float(pd.Series(spearman).dropna().mean())
    if mean > threshold:
        raise RuntimeError(f"LEAK: mean OOF Spearman {mean:.4f} > {threshold} (honest ceiling ~0.15)")
    log.info("leak_check pass: mean OOF Spearman %.4f <= %.2f", mean, threshold)


def suspect_gain(fit_mean: float, baseline_mean: float, margin: float = 0.004) -> bool:
    """Winner's-curse flag: True when ``fit_mean > baseline_mean + margin`` (PLAN: 'suspect', not success)."""
    return bool(fit_mean > baseline_mean + margin)


# --------------------------------------------------------------------------------------------------
# vectorised kernels over a padded [n_eras, max_rows] array (NaN = padding)
# --------------------------------------------------------------------------------------------------
def pad_by_era(values: pd.Series | np.ndarray, era: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Scatter a row series into ``[n_eras, max_rows]`` (NaN padded). Returns (matrix, sorted unique eras)."""
    v = np.asarray(values, dtype=np.float64)
    groups = era_groups(era)
    width = max(len(idx) for _, idx in groups)
    out = np.full((len(groups), width), np.nan)
    for i, (_, idx) in enumerate(groups):
        out[i, : len(idx)] = v[idx]
    return out, np.asarray([e for e, _ in groups], dtype=np.int32)


_PPF_TABLES: dict[int, np.ndarray] = {}


def _ppf_table(width: int) -> np.ndarray:
    """``T[n, k] = norm.ppf(clip(((k/2 + 1) - 0.5)/n))``: every value an average rank (a half-integer ``k/2 + 1``) of a
    row with ``n`` non-NaN entries can map to. Same float expression as the direct path, so a lookup is bit-identical."""
    t = _PPF_TABLES.get(width)
    if t is None:
        n = np.arange(width + 1, dtype=np.float64)[:, None]
        r = np.arange(max(2 * width - 1, 1), dtype=np.float64)[None, :] / 2 + 1
        with np.errstate(invalid="ignore", divide="ignore"):
            u = (r - 0.5) / n
        t = stats.norm.ppf(np.clip(u, *_CLIP))
        _PPF_TABLES[width] = t
    return t


def rank_gauss_rows(a: np.ndarray) -> np.ndarray:
    """Row-wise SDK transform (average ranks over the non-NaN entries, clip, ppf); NaN stays NaN.

    The ppf is read from :func:`_ppf_table` (bit-identical to ``norm.ppf`` on the same input, ~30% faster: the
    regime grid calls this once per draw x era-matrix x cell)."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("rank_gauss_rows expects a 2-D [rows, width] array")
    nan = np.isnan(a)
    r = stats.rankdata(a, axis=1, method="average", nan_policy="omit")
    n = np.sum(~nan, axis=1)
    k = np.where(nan, 0.0, np.nan_to_num(2.0 * r - 2.0)).astype(np.intp)
    z = _ppf_table(a.shape[1])[n[:, None], k]
    z[nan] = np.nan
    return z


def _nanmean_rows(a: np.ndarray) -> np.ndarray:
    return np.nanmean(a, axis=1, keepdims=True)


def fit_rows(z: np.ndarray, tc: np.ndarray) -> np.ndarray:
    """FIT per row for already rank-gaussianised ``z`` and centred target ``tc`` (NaN padded)."""
    n = np.sum(~np.isnan(tc), axis=1)
    return np.nansum((z - _nanmean_rows(z)) * tc, axis=1) / n


def unq_rows(z: np.ndarray, m: np.ndarray, tc: np.ndarray) -> np.ndarray:
    """UNQ per row with the zero rule: ``m`` gaussianised reference; rows where ``cov(m, t) < 0`` -> 0."""
    n = np.sum(~np.isnan(tc), axis=1)
    ref_fit = np.nansum((m - _nanmean_rows(m)) * tc, axis=1) / n
    mm = np.nansum(m * m, axis=1)
    mz = np.nansum(m * z, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(mm > 0, mz / np.where(mm > 0, mm, 1.0), 0.0)
    ortho = z - m * beta[:, None]
    unq = np.nansum(tc * ortho, axis=1) / n
    return np.where((ref_fit < 0) | (mm == 0), 0.0, unq)


def neutralise_rows(z: np.ndarray, m: np.ndarray, p: float) -> np.ndarray:
    """Row-wise ``z - p * m * (m.z)/(m.m)`` with NaN as 0 (same algebra as ``_runtime.neutralise``)."""
    if p == 0.0:
        return z
    zz = np.nan_to_num(z)
    mmv = np.nan_to_num(m)
    mm = np.sum(mmv * mmv, axis=1)
    mz = np.sum(mmv * zz, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(mm > 0, mz / np.where(mm > 0, mm, 1.0), 0.0)
    out = zz - p * mmv * beta[:, None]
    out[np.isnan(z)] = np.nan
    return out


def _standardise_rows(a: np.ndarray) -> np.ndarray:
    """Row-wise zero mean / unit variance over non-NaN entries (constant rows -> 0)."""
    c = a - _nanmean_rows(a)
    sd = np.sqrt(np.nanmean(c * c, axis=1, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(sd > 0, c / np.where(sd > 0, sd, 1.0), 0.0)
    out[np.isnan(a)] = np.nan
    return out


def _orthogonalise_rows(a: np.ndarray, *refs: np.ndarray) -> np.ndarray:
    """Remove the projection of ``a`` on each (already orthogonal, standardised) reference, then re-standardise."""
    out = a
    for r in refs:
        rr = np.nansum(r * r, axis=1, keepdims=True)
        ar = np.nansum(out * r, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = out - np.where(rr > 0, ar / np.where(rr > 0, rr, 1.0), 0.0) * r
    return _standardise_rows(out)


# --------------------------------------------------------------------------------------------------
# regime simulation
# --------------------------------------------------------------------------------------------------
_REGIME_SE_BLOCK, _REGIME_SE_BOOT = 20, 300


def _regime_stats(values: np.ndarray, mask: np.ndarray) -> dict:
    """Mean over all (draw, era) cells in ``mask`` plus a 20-era block-bootstrap SE of the draw-averaged per-era values
    (the eras are kept in time order so the blocks respect the target's ~20-era autocorrelation)."""
    m = mask.astype(np.float64)
    cnt = m.sum()
    out: dict[str, float] = {}
    if cnt == 0:
        return {"mean": float("nan"), "se": float("nan"), "n_eras": 0}
    per_era_cnt = m.sum(axis=0)
    per_era = np.where(per_era_cnt > 0, (values * m).sum(axis=0) / np.where(per_era_cnt > 0, per_era_cnt, 1.0), np.nan)
    ok = ~np.isnan(per_era)
    n_eras = int(ok.sum())
    out["mean"] = float((values * m).sum() / cnt)
    if n_eras > 1:
        means = block_bootstrap_means(per_era[ok], block=_REGIME_SE_BLOCK, n_boot=_REGIME_SE_BOOT, seed=0)
        out["se"] = float(means.std(ddof=1))
    else:
        out["se"] = float("nan")
    out["n_eras"] = n_eras
    return out


def _block_bootstrap_index(n: int, *, block: int, n_boot: int, seed: int) -> np.ndarray:
    """``[n_boot, n]`` era positions of a circular moving-block bootstrap: the same generator as
    ``cv.block_bootstrap_means`` (for equal ``seed`` the very same resamples), exposed so that SEVERAL series can be
    resampled with identical indices (paired Sharpe test, score-Sharpe SE)."""
    if n == 0:
        raise ValueError("empty series")
    if block < 1:
        raise ValueError("block must be >= 1")
    block = min(block, n)
    k = math.ceil(n / block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, k * block) % n
    return idx[:, :n]


def _bootstrap_counts(n: int, *, block: int = _REGIME_SE_BLOCK, n_boot: int = _REGIME_SE_BOOT, seed: int = 0) -> np.ndarray:
    """``[n_boot, n]`` multiplicity of every era in every block-bootstrap resample (float64; rows sum to ``n``)."""
    idx = _block_bootstrap_index(n, block=block, n_boot=n_boot, seed=seed)
    flat = (np.arange(n_boot)[:, None] * n + idx).ravel()
    return np.bincount(flat, minlength=n_boot * n).reshape(n_boot, n).astype(np.float64)


def _sharpe_from_moments(n: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """``mean / std(ddof=1)`` from count, sum and sum of squares; NaN where ``n < 2`` or the variance is 0."""
    n = np.asarray(n, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        var = (s2 - s1 * s1 / n) / (n - 1.0)
        ok = (n >= 2) & (var > 0)
        return np.where(ok, mean / np.sqrt(np.where(ok, var, 1.0)), np.nan)


def _nanmean_quiet(a: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """``np.nanmean`` without the all-NaN RuntimeWarning (an all-NaN slice -> NaN)."""
    a = np.asarray(a, dtype=np.float64)
    cnt = np.sum(~np.isnan(a), axis=axis)
    tot = np.nansum(a, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, tot / np.where(cnt > 0, cnt, 1), np.nan)


def _score_sharpe(score: np.ndarray, mask: np.ndarray, counts: np.ndarray | None) -> tuple[float, float]:
    """``(score_sharpe, score_sharpe_se)`` of a ``[draws, eras]`` score array restricted to ``mask``.

    ``score_sharpe`` = mean over draws of that draw's across-era ``mean / std`` (eras where ``mask`` is true; a draw with
    fewer than 2 such eras is skipped). The benchmark noise of every draw therefore stays in the variance -- what a
    staker experiences round by round. ``score_sharpe_se`` = std of the same statistic over 20-era block-bootstrap
    resamples of the ERAS (``counts``, identical era indices for every draw); NaN without ``counts``.
    """
    m = mask.astype(np.float64)
    a = np.where(mask, score, 0.0)
    a2 = a * a
    sh = _sharpe_from_moments(m.sum(axis=1), a.sum(axis=1), a2.sum(axis=1))
    s_mean = float(_nanmean_quiet(sh))
    if counts is None or counts.shape[0] < 2 or not np.isfinite(s_mean):
        return s_mean, float("nan")
    ct = counts.T
    shb = _sharpe_from_moments(m @ ct, a @ ct, a2 @ ct)       # [draws, n_boot]
    per_boot = np.asarray(_nanmean_quiet(shb, axis=0))
    per_boot = per_boot[np.isfinite(per_boot)]
    se = float(per_boot.std(ddof=1)) if len(per_boot) > 1 else float("nan")
    return s_mean, se


def _block(fit: np.ndarray, unq: np.ndarray, inv: np.ndarray, mask: np.ndarray, counts: np.ndarray | None = None) -> dict:
    """``{fit, unq, inov, score, *_se, n_eras, frac, score_sharpe, score_sharpe_se}`` for one regime mask over [draws, eras]."""
    res: dict[str, float | int] = {}
    score = fit + unq + inv
    for key, arr in (("fit", fit), ("unq", unq), ("inov", inv), ("score", score)):
        st = _regime_stats(arr, mask)
        res[key] = st["mean"]
        res[f"{key}_se"] = st["se"]
        res["n_eras"] = st["n_eras"]
    res["frac"] = float(mask.mean())
    res["score_sharpe"], res["score_sharpe_se"] = _score_sharpe(score, mask, counts)
    return res


def _validate_regime_args(p_b_grid: Sequence[float], p_c_grid: Sequence[float], direction_weight: float, n_draws: int) -> None:
    if not len(p_c_grid):
        raise ValueError("p_c_grid must not be empty")
    if any(p < 0.0 or p > 0.5 for p in p_c_grid):
        raise ValueError(f"p_c values must be in [0, 0.5], got {list(p_c_grid)}")
    if not len(p_b_grid):
        raise ValueError("p_b_grid must not be empty")
    if any(p < 0.0 or p > 0.75 for p in p_b_grid):
        raise ValueError(f"p_b values must be in [0, 0.75], got {list(p_b_grid)}")
    for name, g in (("p_b_grid", p_b_grid), ("p_c_grid", p_c_grid)):
        keys = [f"{float(p):g}" for p in g]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{name} has duplicate values: {list(g)}")
    if not 0.0 <= direction_weight <= 1.0:
        raise ValueError("direction_weight must be in [0,1]")
    if n_draws < 1:
        raise ValueError("n_draws must be >= 1")


def _regime_prepare(z_pred, target, era, direction, core_proxy) -> dict:
    """Padded ``[n_eras, max_rows]`` inputs shared by every draw and cell (rows with a NaN target dropped)."""
    index = era.index
    z_pred = _aligned(z_pred, index, "z_pred")
    target = _aligned(target, index, "target")
    direction = _aligned(direction, index, "direction")
    core_proxy = _aligned(core_proxy, index, "core_proxy")
    ok = target.notna().to_numpy()
    if not ok.all():
        era, z_pred, target = era[ok], z_pred[ok], target[ok]
        direction, core_proxy = direction[ok], core_proxy[ok]
    z, eras = pad_by_era(z_pred.to_numpy(), era)
    t, _ = pad_by_era(target.to_numpy(), era)
    d, _ = pad_by_era(direction.to_numpy(), era)
    c, _ = pad_by_era(core_proxy.to_numpy(), era)
    pad = np.isnan(t)
    z = np.where(pad, np.nan, np.nan_to_num(z))  # NaN members already handled upstream -> mid
    d = np.where(pad, np.nan, np.nan_to_num(d))
    c = np.where(pad, np.nan, np.nan_to_num(c))  # all-missing core rows -> mid (as _runtime does)
    tc = t - _nanmean_rows(t)
    t_s = _standardise_rows(t)
    return {
        "pad": pad, "eras": eras, "n_rows": np.sum(~pad, axis=1), "tc": tc,
        "t_sd": np.sqrt(np.nanmean(tc * tc, axis=1)), "t_s": t_s,
        "d_perp": _orthogonalise_rows(rank_gauss_rows(d), t_s),
        "c_g": rank_gauss_rows(c), "z_g": rank_gauss_rows(z),
    }


def regime_grid(
    z_pred: pd.Series,
    target: pd.Series,
    era: pd.Series,
    direction: pd.Series,
    core_proxy: pd.Series,
    *,
    p_b_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75),
    p_c_grid: Sequence[float] = (0.0, 0.25, 0.5),
    bench_fit_means: Sequence[float] = (0.0, 0.009, 0.02),
    bench_fit_sd: float = 0.035,
    n_draws: int = 500,
    seed: int = 0,
    direction_weight: float = 0.8,
) -> dict:
    """Monte-Carlo score of neutralising ``z_pred`` over the ``p_b_grid x p_c_grid`` cells (INTERFACES section 17).

    Per draw and era the benchmark is ``b = r*·t_s + sqrt(1-r*²)·(w·d_perp + sqrt(1-w²)·e_perp)`` where
    ``t_s`` is the standardised target, ``d_perp`` the surrogate ``direction`` (gaussianised, orthogonalised
    to the target), ``e_perp`` fresh noise orthogonal to both, ``w = direction_weight`` and ``r*`` chosen so
    that the benchmark's FIT ~ N(mean, bench_fit_sd²) (``r* = f*/std(t)``, clipped to |r*| <= 0.95). The
    realised benchmark FIT is reported under ``meta["bench"]`` so the calibration can be checked.
    COMMON RANDOM NUMBERS: one benchmark draw ``m`` serves EVERY ``(p_b, p_c)`` cell, so cell differences carry no
    benchmark-sampling noise and any one ``p_c`` slice equals :func:`regime_simulation` at that ``p_c`` exactly.
    Per cell: ``z <- neutralise(z, gauss(b), p_b)``, ``z <- neutralise(z, gauss(core), p_c)``, re-rank-gaussianise,
    record ``fit``, ``unq`` (vs b, zero rule), ``inov`` (vs core proxy, zero rule), ``score = fit + unq + inov``
    (not arctan-bounded). ``p_b = 0`` cells do not depend on the draw except through ``unq``; their prediction is
    computed once (bit-identical to recomputing it per draw).

    Returns ``{"<mean>": {"p_c=<y>": {"p_b=<x>": {"bpos","bneg","mix"}}}, "meta"}``; ``bpos``/``bneg`` split
    (draw, era) cells by the sign of the realised benchmark FIT. Each regime block holds ``fit/unq/inov/score``
    means, ``*_se`` (20-era block-bootstrap SE of the draw-averaged per-era values), ``n_eras``, ``frac``,
    ``score_sharpe`` (mean over draws of the across-era mean/std of that draw's score) and ``score_sharpe_se``.
    """
    _validate_regime_args(p_b_grid, p_c_grid, direction_weight, n_draws)
    t_start = time.perf_counter()
    P = _regime_prepare(z_pred, target, era, direction, core_proxy)
    pad, tc, t_s, t_sd, d_perp, c_g, z_g = P["pad"], P["tc"], P["t_s"], P["t_sd"], P["d_perp"], P["c_g"], P["z_g"]
    n_eras, width = pad.shape
    p_b_list = [float(p) for p in p_b_grid]
    p_c_list = [float(p) for p in p_c_grid]
    counts = _bootstrap_counts(n_eras) if n_eras > 1 else None
    # draw-independent part of every p_b = 0 cell (prediction, fit, inov); only unq needs the draw
    static = {}
    for p_c in p_c_list:
        zg0 = rank_gauss_rows(neutralise_rows(z_g, c_g, p_c))
        static[p_c] = (zg0, fit_rows(zg0, tc), unq_rows(zg0, c_g, tc))

    w = float(direction_weight)
    w_noise = float(np.sqrt(max(0.0, 1.0 - w * w)))
    rng = np.random.default_rng(seed)
    out: dict = {}
    bench_meta: dict = {}
    cells = [(p_b, p_c) for p_b in p_b_list for p_c in p_c_list]
    for mean in bench_fit_means:
        f_star = rng.normal(float(mean), bench_fit_sd, size=(n_draws, n_eras))
        acc = {cell: {k: np.empty((n_draws, n_eras)) for k in ("fit", "unq", "inov")} for cell in cells}
        realised = np.empty((n_draws, n_eras))
        for k in range(n_draws):
            e = rng.standard_normal((n_eras, width))
            e[pad] = np.nan
            e_perp = _orthogonalise_rows(_standardise_rows(e), t_s, d_perp)
            with np.errstate(invalid="ignore", divide="ignore"):
                r_star = np.clip(f_star[k] / np.where(t_sd > 0, t_sd, np.nan), -0.95, 0.95)
            r_star = np.nan_to_num(r_star)[:, None]
            b_lin = r_star * t_s + np.sqrt(1.0 - r_star**2) * (w * d_perp + w_noise * e_perp)
            m = rank_gauss_rows(b_lin)
            realised[k] = fit_rows(m, tc)
            for p_b in p_b_list:
                zb = None if p_b == 0.0 else neutralise_rows(z_g, m, p_b)
                for p_c in p_c_list:
                    a = acc[(p_b, p_c)]
                    if zb is None:
                        zg, f, i = static[p_c]
                    else:
                        zg = rank_gauss_rows(neutralise_rows(zb, c_g, p_c))
                        f, i = fit_rows(zg, tc), unq_rows(zg, c_g, tc)
                    a["fit"][k] = f
                    a["inov"][k] = i
                    a["unq"][k] = unq_rows(zg, m, tc)
        key = f"{float(mean):g}"
        bpos = realised > 0
        everything = np.ones_like(bpos, dtype=bool)
        bench_meta[key] = {
            "requested_mean": float(mean),
            "realised_fit_mean": float(realised.mean()),
            "realised_fit_sd": float(realised.std()),
            "frac_pos": float(bpos.mean()),
        }
        out[key] = {}
        for p_c in p_c_list:
            by_pb = out[key][f"p_c={p_c:g}"] = {}
            for p_b in p_b_list:
                a = acc[(p_b, p_c)]
                by_pb[f"p_b={p_b:g}"] = {
                    "bpos": _block(a["fit"], a["unq"], a["inov"], bpos, counts),
                    "bneg": _block(a["fit"], a["unq"], a["inov"], ~bpos, counts),
                    "mix": _block(a["fit"], a["unq"], a["inov"], everything, counts),
                }
            log.info("regime mean=%s p_c=%g: %s", key, p_c, "; ".join(
                f"p_b={p:g} mix score {by_pb[f'p_b={p:g}']['mix']['score']:.4f} sharpe {by_pb[f'p_b={p:g}']['mix']['score_sharpe']:.3f}"
                for p in p_b_list))
        del acc
    out["meta"] = {
        "p_c_grid": p_c_list,
        "p_b_grid": p_b_list,
        "n_draws": int(n_draws),
        "n_eras": int(n_eras),
        "rows_per_era": [int(P["n_rows"].min()), int(P["n_rows"].max())],
        "bench_fit_sd": float(bench_fit_sd),
        "direction_weight": w,
        "seed": int(seed),
        "bench": bench_meta,
        "common_random_numbers": True,
        "note": "score = fit + unq + inov (not arctan-bounded); unq/inov use the zero rule; inov vs core PROXY; "
                "score_sharpe = mean over draws of the across-era mean/std of the score",
    }
    log.info("regime grid: %d draws x %d means x %d cells (%d p_b x %d p_c) on %d eras in %.1fs", n_draws, len(bench_fit_means),
             len(cells), len(p_b_list), len(p_c_list), n_eras, time.perf_counter() - t_start)
    return out


def _grid_key(by_value: dict, prefix: str, value: float) -> str:
    """The ``"<prefix>=<x>"`` key of ``by_value`` whose number equals ``value`` (KeyError listing the keys otherwise)."""
    for k in by_value:
        if k.startswith(prefix + "=") and abs(float(k.split("=", 1)[1]) - float(value)) < 1e-12:
            return k
    raise KeyError(f"{prefix}={value:g} not in {list(by_value)}")


def regime_slice(grid: dict, p_c: float) -> dict:
    """One ``p_c`` of a :func:`regime_grid` result in the :func:`regime_simulation` shape
    (``{"<mean>": {"p_b=<x>": {bpos,bneg,mix}}, "meta"}``; ``meta`` carries the old keys with ``p_c``)."""
    out: dict = {}
    for mean_key, by_pc in grid.items():
        if mean_key == "meta":
            continue
        out[mean_key] = by_pc[_grid_key(by_pc, "p_c", p_c)]
    meta = grid.get("meta") or {}
    out["meta"] = {"p_c": float(p_c), **{k: meta[k] for k in ("p_b_grid", "n_draws", "n_eras", "rows_per_era", "bench_fit_sd",
                                                              "direction_weight", "seed", "bench") if k in meta},
                   "note": "score = fit + unq + inov (not arctan-bounded); unq/inov use the zero rule; inov vs core PROXY"}
    return out


def regime_simulation(
    z_pred: pd.Series,
    target: pd.Series,
    era: pd.Series,
    direction: pd.Series,
    core_proxy: pd.Series,
    *,
    p_b_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75),
    p_c: float,
    bench_fit_means: Sequence[float] = (0.0, 0.009, 0.02),
    bench_fit_sd: float = 0.035,
    n_draws: int = 500,
    seed: int = 0,
    direction_weight: float = 0.8,
) -> dict:
    """Single-``p_c`` regime simulation: :func:`regime_grid` with ``p_c_grid=(p_c,)`` in the M1 output shape
    ``{"<mean>": {"p_b=<x>": {"bpos": {...}, "bneg": {...}, "mix": {...}}}, "meta": {...}}`` (same random stream, so the
    numbers are identical to the pre-grid implementation; blocks additionally carry ``score_sharpe``/``score_sharpe_se``)."""
    if not 0.0 <= p_c <= 0.5:
        raise ValueError(f"p_c must be in [0, 0.5], got {p_c}")
    grid = regime_grid(z_pred, target, era, direction, core_proxy, p_b_grid=p_b_grid, p_c_grid=(float(p_c),),
                       bench_fit_means=bench_fit_means, bench_fit_sd=bench_fit_sd, n_draws=n_draws, seed=seed,
                       direction_weight=direction_weight)
    return regime_slice(grid, float(p_c))


def regime_table(result: dict, regime: str = "mix") -> pd.DataFrame:
    """Flatten a :func:`regime_simulation` result into a DataFrame (rows = bench mean x p_b)."""
    rows = []
    for mean_key, by_pb in result.items():
        if mean_key == "meta":
            continue
        for pb_key, regimes in by_pb.items():
            r = regimes[regime]
            rows.append({"bench_mean": float(mean_key), "p_b": float(pb_key.split("=")[1]), "regime": regime,
                         **{k: r[k] for k in ("fit", "unq", "inov", "score", "n_eras", "frac")}})
    return pd.DataFrame(rows)


def regime_grid_table(grid: dict, regime: str = "mix") -> pd.DataFrame:
    """Flatten a :func:`regime_grid` result (rows = bench mean x p_c x p_b) for logging."""
    rows = []
    for mean_key, by_pc in grid.items():
        if mean_key == "meta":
            continue
        for pc_key, by_pb in by_pc.items():
            for pb_key, regimes in by_pb.items():
                r = regimes[regime]
                rows.append({"bench_mean": float(mean_key), "p_c": float(pc_key.split("=")[1]), "p_b": float(pb_key.split("=")[1]),
                             "regime": regime, **{k: r.get(k) for k in ("fit", "unq", "inov", "score", "score_se", "score_sharpe",
                                                                         "n_eras", "frac")}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# paired Sharpe test
# --------------------------------------------------------------------------------------------------
def _sharpe_rows(a: np.ndarray) -> np.ndarray:
    """Row-wise ``mean / std(ddof=1)``; NaN where the std is 0."""
    mean = a.mean(axis=-1)
    sd = a.std(axis=-1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, mean / np.where(sd > 0, sd, 1.0), np.nan)


def paired_sharpe_bootstrap(a: pd.Series, b: pd.Series, *, block: int = 20, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired test of a per-era Sharpe difference ``sharpe(a) - sharpe(b)`` (``sharpe = mean/std`` per era, NOT annualised).

    ``a`` and ``b`` are per-era series on the same era index (inner-joined, NaN eras dropped, sorted by era). Circular
    moving-block bootstrap with IDENTICAL era indices for both series (``_block_bootstrap_index``, the generator of
    ``cv.paired_block_bootstrap``). Returns ``{"dsharpe", "se", "p_pos", "ci90", "n_eras", "sharpe_a", "sharpe_b"}``;
    ``p_pos`` = fraction of resamples with a positive difference (a resample with an undefined Sharpe counts as not
    positive).
    """
    df = pd.concat([pd.Series(a, dtype=np.float64), pd.Series(b, dtype=np.float64)], axis=1, keys=["a", "b"], join="inner")
    df = df.dropna().sort_index()
    n = int(len(df))
    if n < 3:
        raise ValueError(f"paired_sharpe_bootstrap needs >= 3 common eras, got {n}")
    va, vb = df["a"].to_numpy(), df["b"].to_numpy()
    sa, sb = float(_sharpe_rows(va)), float(_sharpe_rows(vb))
    idx = _block_bootstrap_index(n, block=block, n_boot=n_boot, seed=seed)
    d = _sharpe_rows(va[idx]) - _sharpe_rows(vb[idx])
    fin = d[np.isfinite(d)]
    if len(fin):
        lo, hi = np.percentile(fin, [5.0, 95.0])
    else:
        lo = hi = float("nan")
    return {
        "dsharpe": sa - sb,
        "se": float(fin.std(ddof=1)) if len(fin) > 1 else float("nan"),
        "p_pos": float(np.mean(np.nan_to_num(d, nan=0.0) > 0)),
        "ci90": [float(lo), float(hi)],
        "n_eras": n,
        "sharpe_a": sa,
        "sharpe_b": sb,
    }


__all__: Iterable[str] = [
    "fit20", "unq20", "inov", "per_era_fit", "per_era_unq", "per_era_inov", "per_era_bench_fit",
    "per_era_spearman", "per_era_corr", "summarise", "leak_check", "suspect_gain", "regime_simulation",
    "regime_table", "era_groups", "pad_by_era", "rank_gauss_rows", "fit_rows", "unq_rows", "neutralise_rows",
    "regime_grid", "regime_slice", "regime_grid_table", "paired_sharpe_bootstrap",
]
