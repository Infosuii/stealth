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
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from stealth.cv import block_bootstrap_ci

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
    """``{"mean","std","sharpe","pct_pos","frac_roll20_neg","ci90":[lo,hi],"n_eras"}`` of a per-era series.

    ``sharpe`` = mean/std per era (NOT annualised); ``frac_roll20_neg`` = fraction of 20-era rolling means
    below zero (the payout unit); ``ci90`` = unpaired block-bootstrap interval of the mean.
    """
    v = pd.Series(s).dropna().sort_index()
    n = int(len(v))
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "sharpe": float("nan"), "pct_pos": float("nan"),
                "frac_roll20_neg": float("nan"), "ci90": [float("nan"), float("nan")], "n_eras": 0}
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else float("nan")
    sharpe = mean / std if std and std > 0 else float("nan")
    roll = v.rolling(20, min_periods=20).mean().dropna()
    frac_neg = float((roll < 0).mean()) if len(roll) else float("nan")
    lo, hi = block_bootstrap_ci(v, block=block, n_boot=n_boot, seed=seed, level=0.90) if n > 1 else (mean, mean)
    return {
        "mean": mean,
        "std": std,
        "sharpe": float(sharpe),
        "pct_pos": float((v > 0).mean()),
        "frac_roll20_neg": frac_neg,
        "ci90": [float(lo), float(hi)],
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


def rank_gauss_rows(a: np.ndarray) -> np.ndarray:
    """Row-wise SDK transform (average ranks over the non-NaN entries, clip, ppf); NaN stays NaN."""
    r = stats.rankdata(a, axis=1, method="average", nan_policy="omit")
    n = np.sum(~np.isnan(a), axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        u = (r - 0.5) / n
    z = stats.norm.ppf(np.clip(u, *_CLIP))
    z[np.isnan(a)] = np.nan
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
def _regime_stats(values: np.ndarray, mask: np.ndarray) -> dict:
    """Mean over all (draw, era) cells in ``mask`` plus an across-era SE of the draw-averaged values."""
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
    out["se"] = float(np.std(per_era[ok], ddof=1) / np.sqrt(n_eras)) if n_eras > 1 else float("nan")
    out["n_eras"] = n_eras
    return out


def _block(fit: np.ndarray, unq: np.ndarray, inv: np.ndarray, mask: np.ndarray) -> dict:
    """``{fit, unq, inov, score, *_se, n_eras, frac}`` for one regime mask over [draws, eras]."""
    res: dict[str, float | int] = {}
    for key, arr in (("fit", fit), ("unq", unq), ("inov", inv), ("score", fit + unq + inv)):
        st = _regime_stats(arr, mask)
        res[key] = st["mean"]
        res[f"{key}_se"] = st["se"]
        res["n_eras"] = st["n_eras"]
    res["frac"] = float(mask.mean())
    return res


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
    """Monte-Carlo score of neutralising ``z_pred`` against a synthetic live-like benchmark.

    Per draw and era the benchmark is ``b = r*·t_s + sqrt(1-r*²)·(w·d_perp + sqrt(1-w²)·e_perp)`` where
    ``t_s`` is the standardised target, ``d_perp`` the surrogate ``direction`` (gaussianised, orthogonalised
    to the target), ``e_perp`` fresh noise orthogonal to both, ``w = direction_weight`` and ``r*`` chosen so
    that the benchmark's FIT ~ N(mean, bench_fit_sd²) (``r* = f*/std(t)``, clipped to |r*| <= 0.95). The
    realised benchmark FIT is reported under ``meta["bench"]`` so the calibration can be checked.
    For every ``p_b`` we apply ``z <- neutralise(z, gauss(b), p_b)`` then ``z <- neutralise(z, gauss(core), p_c)``,
    re-rank-gaussianise, and record ``fit``, ``unq`` (vs b, zero rule), ``inov`` (vs core proxy, zero rule)
    and ``score = fit + unq + inov`` (not arctan-bounded).

    Returns ``{"<mean>": {"p_b=<x>": {"bpos": {...}, "bneg": {...}, "mix": {...}}}, "meta": {...}}``;
    ``bpos``/``bneg`` split (draw, era) cells by the sign of the realised benchmark FIT. Each regime block
    holds ``fit/unq/inov/score`` means, ``*_se`` (across-era SE of draw-averaged values), ``n_eras``, ``frac``.
    """
    if not 0.0 <= p_c <= 0.5:
        raise ValueError(f"p_c must be in [0, 0.5], got {p_c}")
    if any(p < 0.0 or p > 0.75 for p in p_b_grid):
        raise ValueError(f"p_b values must be in [0, 0.75], got {list(p_b_grid)}")
    if not 0.0 <= direction_weight <= 1.0:
        raise ValueError("direction_weight must be in [0,1]")
    if n_draws < 1:
        raise ValueError("n_draws must be >= 1")
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
    n_eras, width = t.shape
    n_rows = np.sum(~pad, axis=1)

    tc = t - _nanmean_rows(t)
    t_sd = np.sqrt(np.nanmean(tc * tc, axis=1))
    t_s = _standardise_rows(t)
    d_g = rank_gauss_rows(d)
    d_perp = _orthogonalise_rows(d_g, t_s)
    c_g = rank_gauss_rows(c)
    z_g = rank_gauss_rows(z)
    inov_ref = c_g  # constant across draws

    w = float(direction_weight)
    w_noise = float(np.sqrt(max(0.0, 1.0 - w * w)))
    rng = np.random.default_rng(seed)
    out: dict = {}
    bench_meta: dict = {}
    p_b_list = [float(p) for p in p_b_grid]
    for mean in bench_fit_means:
        f_star = rng.normal(float(mean), bench_fit_sd, size=(n_draws, n_eras))
        acc = {p: {k: np.empty((n_draws, n_eras)) for k in ("fit", "unq", "inov")} for p in p_b_list}
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
                zn = neutralise_rows(z_g, m, p_b)
                zn = neutralise_rows(zn, c_g, p_c)
                zg = rank_gauss_rows(zn)
                acc[p_b]["fit"][k] = fit_rows(zg, tc)
                acc[p_b]["unq"][k] = unq_rows(zg, m, tc)
                acc[p_b]["inov"][k] = unq_rows(zg, inov_ref, tc)
        key = f"{float(mean):g}"
        bpos = realised > 0
        bench_meta[key] = {
            "requested_mean": float(mean),
            "realised_fit_mean": float(realised.mean()),
            "realised_fit_sd": float(realised.std()),
            "frac_pos": float(bpos.mean()),
        }
        out[key] = {}
        for p_b in p_b_list:
            a = acc[p_b]
            out[key][f"p_b={p_b:g}"] = {
                "bpos": _block(a["fit"], a["unq"], a["inov"], bpos),
                "bneg": _block(a["fit"], a["unq"], a["inov"], ~bpos),
                "mix": _block(a["fit"], a["unq"], a["inov"], np.ones_like(bpos, dtype=bool)),
            }
        log.info("regime mean=%s: %s", key, "; ".join(
            f"p_b={p:g} mix score {out[key][f'p_b={p:g}']['mix']['score']:.4f}" for p in p_b_list))
    out["meta"] = {
        "p_c": float(p_c),
        "p_b_grid": p_b_list,
        "n_draws": int(n_draws),
        "n_eras": int(n_eras),
        "rows_per_era": [int(n_rows.min()), int(n_rows.max())],
        "bench_fit_sd": float(bench_fit_sd),
        "direction_weight": w,
        "seed": int(seed),
        "bench": bench_meta,
        "note": "score = fit + unq + inov (not arctan-bounded); unq/inov use the zero rule; inov vs core PROXY",
    }
    return out


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


__all__: Iterable[str] = [
    "fit20", "unq20", "inov", "per_era_fit", "per_era_unq", "per_era_inov", "per_era_bench_fit",
    "per_era_spearman", "per_era_corr", "summarise", "leak_check", "suspect_gain", "regime_simulation",
    "regime_table", "era_groups", "pad_by_era", "rank_gauss_rows", "fit_rows", "unq_rows", "neutralise_rows",
]
