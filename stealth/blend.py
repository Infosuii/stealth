"""Blend of member OOF scores: arms, redundancy, admission ladder, nested NNLS gate, staking-aware neutralisation.

Contracts: docs/INTERFACES.md sections 8, 10, 16 and 17; protocol: docs/PLAN.md "Neutralisation / post-processing".

* every member's RAW OOF score is gaussianised per era with ``_runtime.gaussianise`` (the same transform the
  pickle applies at predict time);
* equal weights are HIERARCHICAL: group (everest ``blend.everest_weight`` / aux the rest) -> equal per arm in the
  group -> equal per member in the arm (``arm_of``: the ``arm`` key, else ``"aux"`` for the aux group, else the member
  name -- so a config without ``arm`` keys gets exactly the M1 flat weights);
* ``member_corr_matrix`` (mean per-era Pearson of the gaussianised scores) -> ``blend.json.member_corr`` and
  ``member_corr.csv``;
* with ``blend.admission`` set, the pre-registered ADMISSION LADDER visits base arms, then ``admission.order``
  (config order inside an arm), drops members correlated above ``gates.member_corr_max`` with an already-kept member
  (the earlier/simpler one stays), and tests each arm ONCE against the incumbent (paired fit20 bootstrap on CV OOF,
  forward diff, paired Sharpe bootstrap, correlation with the incumbent). Without ``admission`` every member is in;
* NNLS weights (sum to 1, floor 0.02) on the ADMITTED set are computed NESTED: fitted on CV folds ``0..K-2``,
  evaluated on fold ``K-1`` (paired block bootstrap vs hierarchical equal weight) and on the forward block; adopted
  only when ``p_pos >= gates.p_pos_adopt`` and the forward difference is ``>= 0``. When NNLS is adopted the bundle
  gets ``full_weights`` (fitted on all OOF eras) but every REPORTED number (``chosen`` fit20, forward fit, the
  SUSPECT and leak checks, the regime grid, the neutralisation choice) comes from CROSS-FITTED weights;
* the chosen blend's OOF ``z`` goes through ``metrics.regime_grid`` (p_b x p_c, common random numbers) and
  ``select_neutralisation`` freezes ``p_b``/``p_c`` from the simulation (section 16.3; never from the diag lane);
* gates: ``metrics.leak_check`` on the blend's per-era Spearman; a "SUSPECT" banner when the blend's OOF fit20
  exceeds the baseline (``gates.baseline_fit`` if set, else the in-run baseline member) by more than
  ``gates.suspect_gain_over_baseline``.
Nothing here fits, blends or selects on ``v1_sherpa``: the benchmark in the simulation is synthetic.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from stealth import _runtime, cv, data, features, metrics, surrogate
from stealth.train import (ARTIFACTS, REPO_ROOT, baseline_fit, baseline_member, per_era_fit_fast, per_era_spearman_fast, setup_logging,
                           write_json)

log = logging.getLogger("stealth.blend")  # explicit: under `python -m` __name__ is __main__

WEIGHT_FLOOR = 0.02
FORWARD_GAP = 20  # eras between the OOF rows that fit forward-block NNLS weights and the forward test block
SUSPECT_BANNER = "SUSPECT — winner's curse or leak: blended OOF fit20 {blend:.4f} > {base} {bmean:.4f} + {margin}"
DEFAULT_MEMBER_CORR_MAX = 0.95
DEFAULT_SHARPE_GAIN = 0.05
NEUTRAL_DEFAULTS = {"select": False, "central_mean": 0.009, "pessimistic_mean": 0.02, "fit_cost_max": 0.003,
                    "sharpe_tol": 0.02, "worst_tol": 0.0005}
P_B_CAP, P_C_CAP = 0.75, 0.5
_EPS = 1e-12


# --------------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------------
def load_run(run_dir: Path) -> dict:
    """Everything blend.py needs from ``artifacts/<run>``: cfg, oof, forward, member specs, features.json, metrics."""
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    oof = pd.read_parquet(run_dir / "oof.parquet")
    fwd_path = run_dir / "forward.parquet"
    fwd = pd.read_parquet(fwd_path) if fwd_path.exists() else None
    specs = {}
    for p in sorted((run_dir / "members").glob("*/spec.json")):
        s = json.loads(p.read_text())
        if f"m__{s['name']}" in oof.columns:
            specs[s["name"]] = s
    feats = json.loads((run_dir / "features.json").read_text())
    mpath = run_dir / "metrics.json"
    mets = json.loads(mpath.read_text()) if mpath.exists() else {}
    return {"run_dir": run_dir, "cfg": cfg, "oof": oof, "fwd": fwd, "specs": specs, "features": feats, "metrics": mets}


def member_columns(df: pd.DataFrame) -> list[str]:
    return [c[3:] for c in df.columns if c.startswith("m__")]


def config_order(names: list[str], cfg: dict) -> list[str]:
    """``names`` in ``cfg['members']`` order (names absent from the config keep their relative order at the end)."""
    pos = {str(m.get("name")): i for i, m in enumerate(cfg.get("members") or [])}
    return sorted(names, key=lambda n: (pos.get(n, len(pos)), names.index(n)))


# --------------------------------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------------------------------
def gaussianise_per_era(raw: pd.DataFrame, era: pd.Series) -> pd.DataFrame:
    """Per-era ``_runtime.gaussianise`` of every column (NaN stays NaN so a missing member drops out of the row)."""
    out = np.full(raw.shape, np.nan, dtype=np.float64)
    vals = raw.to_numpy(dtype=np.float64)
    for _, idx in metrics.era_groups(era):
        block = vals[idx]
        for j in range(block.shape[1]):
            col = block[:, j]
            ok = ~np.isnan(col)
            if ok.sum() >= 2:
                z = np.full(len(col), np.nan)
                z[ok] = _runtime.gaussianise(col[ok])
                out[idx, j] = z
    return pd.DataFrame(out, index=raw.index, columns=raw.columns)


def blend_scores(Z: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted mean of the gaussianised member columns; NaN members drop out of a row; all-NaN -> 0."""
    names = [n for n in weights if n in Z.columns and weights[n] > 0]
    w = np.array([weights[n] for n in names], dtype=np.float64)
    M = Z[names].to_numpy(dtype=np.float64)
    ok = ~np.isnan(M)
    num = np.nansum(M * w, axis=1)
    den = (ok * w).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(den > 0, num / den, 0.0)
    return pd.Series(z, index=Z.index, name="blend")


def arm_of(spec: dict) -> str:
    """Member arm (INTERFACES section 11/16): ``spec["arm"]``, else ``"aux"`` for the aux group, else the member name."""
    return str(spec.get("arm") or ("aux" if spec.get("group") == "aux" else spec.get("name")))


def blend_weights(groups: dict[str, str], everest_weight: float, arms: dict[str, str] | None = None) -> dict[str, float]:
    """Equal weights. Everest group gets ``everest_weight`` (all of it without an aux group, none without everest).

    ``arms=None``: equal per member inside a group (M1). Else HIERARCHICAL: the group weight is split equally over the
    arms present in the group, then equally over that arm's members (a member missing from ``arms`` is its own arm).
    """
    ev = [n for n, g in groups.items() if g == "everest"]
    aux = [n for n, g in groups.items() if g != "everest"]
    if not ev and not aux:
        raise ValueError("no members to blend")
    w_ev = float(everest_weight) if aux else 1.0
    if not ev:
        w_ev = 0.0
    if arms is None:
        weights = {n: w_ev / len(ev) for n in ev}
        weights.update({n: (1.0 - w_ev) / len(aux) for n in aux})
        return weights
    weights = {}
    for members, gw in ((ev, w_ev), (aux, 1.0 - w_ev)):
        by_arm: dict[str, list[str]] = {}
        for n in members:
            by_arm.setdefault(str(arms.get(n, n)), []).append(n)
        for arm_members in by_arm.values():
            for n in arm_members:
                weights[n] = gw / len(by_arm) / len(arm_members)
    return weights


def nnls_weights(Z: pd.DataFrame, y: pd.Series, era: pd.Series, *, floor: float = WEIGHT_FLOOR) -> dict[str, float]:
    """Non-negative least squares of the per-era gaussianised target on the member z's; sum to 1 with a floor."""
    from scipy.optimize import nnls

    yz = gaussianise_per_era(y.to_frame("y"), era)["y"]
    M = Z.to_numpy(dtype=np.float64)
    ok = ~np.isnan(M).any(axis=1) & yz.notna().to_numpy()
    coef, _ = nnls(M[ok], yz.to_numpy()[ok])
    if coef.sum() <= 0:
        coef = np.ones(len(coef))
    w = coef / coef.sum()
    w = np.maximum(w, floor)
    w = w / w.sum()
    return {n: float(x) for n, x in zip(Z.columns, w)}


def fit_series(z: pd.Series, y: pd.Series, era: pd.Series) -> pd.Series:
    """Per-era fit20 (``metrics.per_era_fit`` semantics) via the vectorised kernel ``train.per_era_fit_fast``
    (parity with the SDK loop to ~1e-16, tests/test_train_m2.py); the SDK loop cost ~10 s per ladder step."""
    return per_era_fit_fast(z, y, era)


def paired_vs_baseline(z: pd.Series, oof: pd.DataFrame, fwd: pd.DataFrame | None, fwd_fit: float | None,
                       cfg: dict, names: list[str]) -> dict | None:
    """Paired 20-era block bootstrap of the REPORTED blend z (cross-fitted when NNLS) minus the baseline member
    (``train.baseline_member``) on CV OOF, plus the forward-block diff. Informational: a tie or a loss means the
    ladder kept (or should keep) the simpler incumbent. None when the baseline member is not in the run."""
    base = baseline_member(cfg)
    if base not in names:
        return None
    bs = cfg["bootstrap"]
    y, era = oof[data.PRIMARY_TARGET], oof["era"]
    d = (fit_series(z, y, era) - fit_series(oof[f"m__{base}"], y, era)).dropna()
    res = cv.paired_block_bootstrap(d, block=int(bs["block"]), n_boot=int(bs["n_boot"]), seed=int(bs["seed"])).as_dict()
    res["fwd_diff"] = None
    if fwd is not None and fwd_fit is not None:
        res["fwd_diff"] = float(fwd_fit - fit_series(fwd[f"m__{base}"], fwd[data.PRIMARY_TARGET], fwd["era"]).mean())
    res["baseline"] = base
    log.info("blend vs baseline %s: diff_mean=%+.4f se=%.4f p_pos=%.3f forward diff=%s", base, res["diff_mean"], res["diff_se"],
             res["p_pos"], None if res["fwd_diff"] is None else round(res["fwd_diff"], 4))
    return res


# --------------------------------------------------------------------------------------------------
# correlation
# --------------------------------------------------------------------------------------------------
def _pad_stack(Z: pd.DataFrame, era: pd.Series) -> np.ndarray:
    """``[n_members, n_eras, max_rows]`` NaN-padded per-era blocks of every column of ``Z`` (one era grouping)."""
    groups = metrics.era_groups(era)
    width = max(len(idx) for _, idx in groups)
    vals = Z.to_numpy(dtype=np.float64)
    out = np.full((vals.shape[1], len(groups), width), np.nan)
    for i, (_, idx) in enumerate(groups):
        out[:, i, : len(idx)] = vals[idx].T
    return out


def _era_pearson(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-row Pearson correlation of two NaN-padded ``[n_eras, width]`` arrays over the jointly non-NaN entries
    (NaN where fewer than 3 such entries or a constant side)."""
    ok = ~(np.isnan(A) | np.isnan(B))
    cnt = ok.sum(axis=1)
    a = np.where(ok, A, 0.0)
    b = np.where(ok, B, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ac = np.where(ok, a - (a.sum(axis=1) / cnt)[:, None], 0.0)
        bc = np.where(ok, b - (b.sum(axis=1) / cnt)[:, None], 0.0)
        den = np.sqrt((ac * ac).sum(axis=1) * (bc * bc).sum(axis=1))
        return np.where((cnt >= 3) & (den > 0), (ac * bc).sum(axis=1) / np.where(den > 0, den, 1.0), np.nan)


def _nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def mean_era_corr(a: pd.Series, b: pd.Series, era: pd.Series) -> float:
    """Mean over eras of the per-era Pearson correlation of two row series on ``era``'s index."""
    st = _pad_stack(pd.concat([a.reindex(era.index), b.reindex(era.index)], axis=1), era)
    return _nanmean(_era_pearson(st[0], st[1]))


def member_corr_matrix(Z: pd.DataFrame, era: pd.Series) -> pd.DataFrame:
    """Mean per-era Pearson correlation of the gaussianised member scores (columns of ``Z``); diagonal 1."""
    names = list(Z.columns)
    st = _pad_stack(Z, era)
    C = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            C[i, j] = C[j, i] = _nanmean(_era_pearson(st[i], st[j]))
    return pd.DataFrame(C, index=names, columns=names)


# --------------------------------------------------------------------------------------------------
# nested NNLS gate
# --------------------------------------------------------------------------------------------------
def nested_nnls(Z: pd.DataFrame, oof: pd.DataFrame, Zf: pd.DataFrame | None, fwd: pd.DataFrame | None,
                ew: dict[str, float], cfg: dict) -> dict:
    """Fit NNLS on folds 0..K-2, compare with equal weight on fold K-1 (paired bootstrap) and on the forward block.

    Returns ``{"nested_weights", "full_weights", "paired_vs_ew", "forward_diff", "adopt", "reason"}``.
    """
    y, era, fold = oof[data.PRIMARY_TARGET], oof["era"], oof["fold"]
    bs, gates = cfg["bootstrap"], cfg["gates"]
    last = int(fold.max())
    fit_mask = (fold < last).to_numpy()
    eval_mask = (fold == last).to_numpy()
    w_nested = nnls_weights(Z.loc[fit_mask], y.loc[fit_mask], era.loc[fit_mask])
    z_n = blend_scores(Z.loc[eval_mask], w_nested)
    z_e = blend_scores(Z.loc[eval_mask], ew)
    diff = fit_series(z_n, y.loc[eval_mask], era.loc[eval_mask]) - fit_series(z_e, y.loc[eval_mask], era.loc[eval_mask])
    paired = cv.paired_block_bootstrap(diff.dropna(), block=int(bs["block"]), n_boot=int(bs["n_boot"]), seed=int(bs["seed"])).as_dict()
    fwd_diff = None
    if Zf is not None and fwd is not None:
        fy, fe = fwd[data.PRIMARY_TARGET], fwd["era"]
        fwd_diff = float(fit_series(blend_scores(Zf, w_nested), fy, fe).mean() - fit_series(blend_scores(Zf, ew), fy, fe).mean())
    adopt = paired["p_pos"] >= float(gates["p_pos_adopt"]) and (fwd_diff is None or fwd_diff >= 0.0)
    reason = (f"p_pos={paired['p_pos']:.3f} {'>=' if paired['p_pos'] >= float(gates['p_pos_adopt']) else '<'} {gates['p_pos_adopt']}"
              + ("" if fwd_diff is None else f", forward diff={fwd_diff:+.4f}"))
    w_full = nnls_weights(Z, y, era)
    log.info("nested NNLS: nested weights %s; eval fold%d diff_mean=%+.4f se=%.4f p_pos=%.3f; %s -> %s", _fmt_w(w_nested), last,
             paired["diff_mean"], paired["diff_se"], paired["p_pos"], reason, "ADOPT nnls" if adopt else "keep equal weights")
    return {"nested_weights": w_nested, "full_weights": w_full, "paired_vs_ew": paired, "forward_diff": fwd_diff,
            "adopt": bool(adopt), "reason": reason, "eval_fold": last}


def crossfit_nnls(Z: pd.DataFrame, oof: pd.DataFrame) -> tuple[pd.Series, dict[str, dict[str, float]]]:
    """Cross-fitted NNLS blend: for every CV fold ``k`` fit the weights on the OOF rows of folds ``!= k`` and apply them to
    fold ``k``. Returns ``(z, {fold: weights})``; ``z`` is out-of-sample w.r.t. the weights on every row."""
    y, era, fold = oof[data.PRIMARY_TARGET], oof["era"], oof["fold"]
    z = pd.Series(np.nan, index=Z.index, name="blend", dtype=np.float64)
    per_fold: dict[str, dict[str, float]] = {}
    for k in sorted(int(x) for x in fold.unique()):
        fit_mask = (fold != k).to_numpy()
        apply_mask = (fold == k).to_numpy()
        w = nnls_weights(Z.loc[fit_mask], y.loc[fit_mask], era.loc[fit_mask])
        z.loc[apply_mask] = blend_scores(Z.loc[apply_mask], w).to_numpy()
        per_fold[f"fold{k}"] = w
    return z, per_fold


def forward_nnls_weights(Z: pd.DataFrame, oof: pd.DataFrame, fwd: pd.DataFrame, nested: dict[str, float]) -> tuple[dict[str, float], str]:
    """NNLS weights that never saw the forward test eras: fitted on the OOF rows ``FORWARD_GAP`` eras or more before the
    forward block (the forward test eras overlap the last CV test block, so the full-OOF weights are in-sample there).
    Falls back to the nested (folds 0..K-2) weights when fewer than ``2 * FORWARD_GAP`` such eras exist."""
    cut = int(fwd["era"].min()) - FORWARD_GAP
    mask = (oof["era"] < cut).to_numpy()
    if int(oof.loc[mask, "era"].nunique()) < 2 * FORWARD_GAP:
        return dict(nested), "nested weights (folds 0..K-2): too few OOF eras before the forward block"
    w = nnls_weights(Z.loc[mask], oof.loc[mask, data.PRIMARY_TARGET], oof.loc[mask, "era"])
    return w, f"NNLS on OOF eras < {cut} (>= {FORWARD_GAP} eras before the forward block)"


def _fmt_w(w: dict[str, float]) -> str:
    return "{" + ", ".join(f"{k}: {v:.3f}" for k, v in w.items()) + "}"


def evaluate_blend(z: pd.Series, frame: pd.DataFrame, cfg: dict) -> tuple[dict, pd.Series, pd.Series]:
    """``(summarise(fit), fit series, spearman series)`` of a blended z on an oof/forward frame."""
    bs = cfg["bootstrap"]
    y, era = frame[data.PRIMARY_TARGET], frame["era"]
    fit = fit_series(z, y, era)
    sp = per_era_spearman_fast(z, y, era)
    return metrics.summarise(fit, block=int(bs["block"]), n_boot=int(bs["n_boot"]), seed=int(bs["seed"])), fit, sp


# --------------------------------------------------------------------------------------------------
# admission ladder (16.1)
# --------------------------------------------------------------------------------------------------
def ladder_plan(names: list[str], specs: dict[str, dict], cfg: dict) -> tuple[list[tuple[str, list[str], bool]], list[str]]:
    """``([(arm, members in config order, is_base)], unregistered arms)`` in pre-registered ladder order: the
    ``admission.base`` arms, then ``admission.order``. Arms named but absent from the run are skipped (logged); arms of
    the run named in neither list are returned separately (their members are NOT admitted)."""
    adm = (cfg.get("blend") or {}).get("admission") or {}
    ordered = config_order(list(names), cfg)
    by_arm: dict[str, list[str]] = {}
    for n in ordered:
        by_arm.setdefault(arm_of(specs[n]), []).append(n)
    plan: list[tuple[str, list[str], bool]] = []
    seen: set[str] = set()
    for is_base, key in ((True, "base"), (False, "order")):
        for a in adm.get(key) or []:
            a = str(a)
            if a in seen:
                continue
            seen.add(a)
            if a not in by_arm:
                log.info("admission ladder: arm %r has no member in this run (absent, dropped or pending) -> skipped", a)
                continue
            plan.append((a, by_arm[a], is_base))
    return plan, [a for a in by_arm if a not in seen]


def admit_decision(row: dict, *, corr_max: float, p_pos_adopt: float, sharpe_gain: float) -> tuple[bool, str]:
    """16.1 rule: admit iff ``corr_with_incumbent <= corr_max`` AND ``fwd_diff >= 0`` (or no forward block) AND
    (``p_pos >= p_pos_adopt`` OR (``dsharpe >= sharpe_gain`` AND ``p_pos_sharpe >= p_pos_adopt``))."""
    def _ok(x) -> bool:
        return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))

    corr, fwd = row.get("corr_with_incumbent"), row.get("fwd_diff")
    p_pos, ds, pps = row.get("p_pos"), row.get("dsharpe"), row.get("p_pos_sharpe")
    corr_ok = _ok(corr) and float(corr) <= corr_max + _EPS
    fwd_ok = fwd is None or (_ok(fwd) and float(fwd) >= 0.0)
    fit_route = _ok(p_pos) and float(p_pos) >= p_pos_adopt - _EPS
    sharpe_route = _ok(ds) and _ok(pps) and float(ds) >= sharpe_gain - _EPS and float(pps) >= p_pos_adopt - _EPS
    admitted = bool(corr_ok and fwd_ok and (fit_route or sharpe_route))
    parts = [f"corr {corr:.3f} {'<=' if corr_ok else '>'} {corr_max}" if _ok(corr) else "corr undefined",
             "no forward block" if fwd is None else f"fwd diff {fwd:+.4f}{'' if fwd_ok else ' < 0'}",
             f"p_pos {p_pos:.3f} {'>=' if fit_route else '<'} {p_pos_adopt}" if _ok(p_pos) else "p_pos undefined",
             (f"dsharpe {ds:+.3f} {'>=' if _ok(ds) and float(ds) >= sharpe_gain - _EPS else '<'} {sharpe_gain}, "
              f"p_pos_sharpe {pps:.3f}" if _ok(ds) and _ok(pps) else "sharpe undefined")]
    route = ("fit + Sharpe routes" if fit_route and sharpe_route else "fit route" if fit_route
             else "Sharpe route" if sharpe_route else "no route")
    return admitted, ("ADMIT (" + route + "): " if admitted else "REJECT: ") + "; ".join(parts)


def _ladder_row(arm: str, members: list[str], admitted: bool, reason: str, dropped: list[str], **stats) -> dict:
    row = {"arm": arm, "members": list(members), "admitted": bool(admitted), "reason": reason, "diff_mean": None, "diff_se": None,
           "p_pos": None, "fwd_diff": None, "dsharpe": None, "p_pos_sharpe": None, "corr_with_incumbent": None,
           "dropped_redundant": list(dropped)}
    row.update(stats)
    return row


def admission_ladder(Z: pd.DataFrame, oof: pd.DataFrame, Zf: pd.DataFrame | None, fwd: pd.DataFrame | None,
                     specs: dict[str, dict], cfg: dict, *, corr: pd.DataFrame | None = None,
                     redundant: dict[str, dict] | None = None) -> tuple[list[str], list[dict]]:
    """Section 16.1: redundancy drop + pre-registered admission ladder, one look per arm.

    Members are visited in ladder order (base arms, then ``admission.order``, config order inside an arm). A member
    whose correlation with an already-kept member (the incumbent's members and the earlier kept members of its own
    arm) exceeds ``gates.member_corr_max`` is dropped as ``redundant(<other>)``. Incumbent ``I`` = the base arms. For
    each later arm ``A``: ``C = I + A`` (hierarchical EW over C vs over I); per-era fit20 difference on CV OOF ->
    ``cv.paired_block_bootstrap``; ``fwd_diff`` on the forward block; ``metrics.paired_sharpe_bootstrap(fit_C, fit_I)``;
    ``corr_with_incumbent`` = mean per-era corr of A's mean z with I's blend z. Admitted arms join I.
    Returns ``(admitted member names, ladder rows)``; ``redundant`` (optional dict) receives
    ``{member: {"other", "corr"}}``.
    """
    bcfg, gates, bs = cfg["blend"], cfg["gates"], cfg["bootstrap"]
    adm = bcfg.get("admission") or {}
    corr_max = float(gates.get("member_corr_max", DEFAULT_MEMBER_CORR_MAX))
    p_pos_adopt = float(gates["p_pos_adopt"])
    sharpe_gain = float(adm.get("sharpe_gain", DEFAULT_SHARPE_GAIN))
    ew_w = float(bcfg["everest_weight"])
    block, n_boot, seed = int(bs["block"]), int(bs["n_boot"]), int(bs["seed"])
    names = [n for n in Z.columns if n in specs]
    if corr is None:
        corr = member_corr_matrix(Z[names], oof["era"])
    redundant = {} if redundant is None else redundant
    groups = {n: specs[n]["group"] for n in names}
    arms = {n: arm_of(specs[n]) for n in names}
    y, era = oof[data.PRIMARY_TARGET], oof["era"]

    def ew_of(members: list[str]) -> dict[str, float]:
        return blend_weights({n: groups[n] for n in members}, ew_w, {n: arms[n] for n in members})

    def fit_of(members: list[str], Zx: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
        return fit_series(blend_scores(Zx, ew_of(members)), frame[data.PRIMARY_TARGET], frame["era"])

    plan, unregistered = ladder_plan(names, specs, cfg)
    incumbent: list[str] = []
    rows: list[dict] = []
    for arm, members, is_base in plan:
        keep: list[str] = []
        dropped: list[str] = []
        for n in members:
            ref = incumbent + keep
            if ref:
                c = corr.loc[n, ref].astype(float)
                if float(c.max()) > corr_max:
                    other = str(c.idxmax())
                    redundant[n] = {"other": other, "corr": float(c.max()), "arm": arm}
                    dropped.append(n)
                    log.info("admission: %s dropped as redundant(%s), corr %.3f > %s", n, other, float(c.max()), corr_max)
                    continue
            keep.append(n)
        red_note = "" if not dropped else "; dropped redundant: " + ", ".join(
            f"{n} (redundant({redundant[n]['other']}) corr {redundant[n]['corr']:.3f})" for n in dropped)
        if not keep:
            rows.append(_ladder_row(arm, members, False, "REJECT: every member redundant" + red_note, dropped))
            continue
        if is_base or not incumbent:
            if not is_base:
                log.warning("admission: no base arm present in the run; %r becomes the incumbent", arm)
            incumbent += keep
            rows.append(_ladder_row(arm, keep, True, ("base arm (incumbent)" if is_base else "incumbent (no base arm present)") + red_note,
                                    dropped))
            continue
        cand = incumbent + keep
        fit_c, fit_i = fit_of(cand, Z, oof), fit_of(incumbent, Z, oof)
        paired = cv.paired_block_bootstrap((fit_c - fit_i).dropna(), block=block, n_boot=n_boot, seed=seed).as_dict()
        fwd_diff = None
        if Zf is not None and fwd is not None:
            fwd_diff = float(fit_of(cand, Zf, fwd).mean() - fit_of(incumbent, Zf, fwd).mean())
        ps = metrics.paired_sharpe_bootstrap(fit_c, fit_i, block=block, n_boot=n_boot, seed=seed)
        z_arm = blend_scores(Z, {n: 1.0 / len(keep) for n in keep})
        z_inc = blend_scores(Z, ew_of(incumbent))
        c_inc = mean_era_corr(z_arm, z_inc, era)
        stats = {"diff_mean": paired["diff_mean"], "diff_se": paired["diff_se"], "p_pos": paired["p_pos"], "fwd_diff": fwd_diff,
                 "dsharpe": ps["dsharpe"], "p_pos_sharpe": ps["p_pos"], "corr_with_incumbent": c_inc}
        ok, reason = admit_decision(stats, corr_max=corr_max, p_pos_adopt=p_pos_adopt, sharpe_gain=sharpe_gain)
        rows.append(_ladder_row(arm, keep, ok, reason + red_note, dropped, **stats))
        log.info("admission ladder %s %s: diff_mean=%+.4f se=%.4f p_pos=%.3f fwd=%s dsharpe=%+.3f p_pos_sharpe=%.3f corr=%.3f -> %s",
                 arm, keep, paired["diff_mean"], paired["diff_se"], paired["p_pos"], None if fwd_diff is None else round(fwd_diff, 4),
                 ps["dsharpe"], ps["p_pos"], c_inc, reason)
        if ok:
            incumbent += keep
    for arm in unregistered:
        members = [n for n in config_order(names, cfg) if arms[n] == arm]
        log.warning("admission: arm %r (%s) is not pre-registered in blend.admission base/order -> not admitted", arm, members)
        rows.append(_ladder_row(arm, members, False, "REJECT: arm not pre-registered in blend.admission", []))
    admitted = [n for n in names if n in set(incumbent)]
    return admitted, rows


# --------------------------------------------------------------------------------------------------
# neutralisation constants (16.3)
# --------------------------------------------------------------------------------------------------
def neutralisation_config(cfg: dict) -> dict:
    """``blend.neutralisation`` with the section-11 defaults filled in."""
    return {**NEUTRAL_DEFAULTS, **((cfg.get("blend") or {}).get("neutralisation") or {})}


def _mean_key(grid: dict, mean: float) -> str | None:
    for k in grid:
        if k != "meta" and abs(float(k) - float(mean)) < _EPS:
            return k
    return None


def _cells(grid: dict, mean_key: str) -> list[tuple[float, float, str, str]]:
    """``[(p_b, p_c, pc_key, pb_key)]`` of one bench-mean slice of a regime grid."""
    out = []
    for pc_key, by_pb in grid[mean_key].items():
        for pb_key in by_pb:
            out.append((float(pb_key.split("=", 1)[1]), float(pc_key.split("=", 1)[1]), pc_key, pb_key))
    return out


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _num(x) -> str:
    return f"{float(x):.4f}" if _finite(x) else str(x)


def _neutralisation_scan(grid: dict, ncfg: dict) -> tuple[list[dict], dict, dict]:
    """Eligibility, tie set and worst-regime score of every cell -> ``(table, chosen row, summary)`` (section 16.3)."""
    ck, pk = _mean_key(grid, ncfg["central_mean"]), _mean_key(grid, ncfg["pessimistic_mean"])
    if ck is None or pk is None:
        raise ValueError(f"regime grid has no bench mean {ncfg['central_mean']} / {ncfg['pessimistic_mean']} "
                         f"(means: {[k for k in grid if k != 'meta']})")
    mean_keys = [k for k in grid if k != "meta"]
    cells = _cells(grid, ck)
    ref = [c for c in cells if c[0] == 0.0 and c[1] == 0.0]
    if not ref:
        raise ValueError("regime grid needs the (p_b=0, p_c=0) reference cell: p_b_grid and p_c_grid must contain 0")
    _, _, rpc, rpb = ref[0]
    ref_fit = grid[ck][rpc][rpb]["mix"]["fit"]
    ref_pess, ref_pess_se = grid[pk][rpc][rpb]["mix"]["score"], grid[pk][rpc][rpb]["mix"].get("score_se")
    se = float(ref_pess_se) if _finite(ref_pess_se) else 0.0
    fit_cost_max = float(ncfg["fit_cost_max"])
    table: list[dict] = []
    for p_b, p_c, pc_key, pb_key in cells:
        mix = grid[ck][pc_key][pb_key]["mix"]
        worst = [grid[m][pc_key][pb_key][r]["score"] for m in mean_keys for r in ("bpos", "bneg")]
        worst = [float(v) for v in worst if _finite(v)]
        W = min(worst) if worst else None
        is_ref = p_b == 0.0 and p_c == 0.0
        capped = p_b <= P_B_CAP + _EPS and p_c <= P_C_CAP + _EPS
        pess = grid[pk][pc_key][pb_key]["mix"]["score"]
        fit_ok = _finite(mix["fit"]) and _finite(ref_fit) and float(mix["fit"]) >= float(ref_fit) - fit_cost_max - _EPS
        pess_ok = _finite(pess) and _finite(ref_pess) and float(pess) >= float(ref_pess) - se - _EPS
        table.append({"p_b": p_b, "p_c": p_c, "fit": mix["fit"], "score": mix["score"], "score_sharpe": mix.get("score_sharpe"),
                      "W": W, "eligible": bool(is_ref or (capped and fit_ok and pess_ok)), "in_tie_set": False})
    elig = [r for r in table if r["eligible"]]
    s_val = {id(r): float(r["score_sharpe"]) if _finite(r["score_sharpe"]) else -math.inf for r in table}
    w_val = {id(r): float(r["W"]) if r["W"] is not None else -math.inf for r in table}
    s_best = max(s_val[id(r)] for r in elig)
    tie = [r for r in elig if s_val[id(r)] >= s_best - float(ncfg["sharpe_tol"]) - _EPS]
    for r in tie:
        r["in_tie_set"] = True
    w_best = max(w_val[id(r)] for r in tie)
    final = [r for r in tie if w_val[id(r)] >= w_best - float(ncfg["worst_tol"]) - _EPS]
    chosen = min(final, key=lambda r: (round(r["p_b"] + r["p_c"], 12), r["p_b"]))
    summary = {"n_cells": len(table), "n_eligible": len(elig), "n_tie": len(tie), "n_final": len(final),
               "sharpe_best": s_best, "W_best": w_best, "central_key": ck, "pessimistic_key": pk}
    return table, chosen, summary


def select_neutralisation(grid: dict, cfg: dict, *, z_source: str | None = None) -> dict:
    """Section 16.3: freeze ``p_b``/``p_c`` from the OOF regime grid (staking-aware; never from the diag lane).

    Eligible iff (a) mix FIT at ``central_mean`` >= the (0,0) cell's minus ``fit_cost_max`` and (b) the mix score at
    ``pessimistic_mean`` >= the (0,0) cell's score minus its ``score_se``; caps 0.75 / 0.5; (0,0) always eligible.
    Among eligible cells: best mix ``score_sharpe`` at ``central_mean``; tie set within ``sharpe_tol``; inside it maximise
    ``W`` = min over {bpos, bneg} x every bench mean of the score; ``W`` within ``worst_tol`` -> smaller ``p_b + p_c``,
    then smaller ``p_b``. ``select: false`` -> the config constants with ``p_source: "config"`` (table for context).
    """
    ncfg = neutralisation_config(cfg)
    bcfg = cfg.get("blend") or {}
    criterion = (f"eligible: mix fit >= fit(0,0) - {ncfg['fit_cost_max']} at bench mean {ncfg['central_mean']} and mix score at "
                 f"{ncfg['pessimistic_mean']} >= score(0,0) - score_se(0,0); caps p_b <= {P_B_CAP}, p_c <= {P_C_CAP}; (0,0) always "
                 f"eligible -> max mix score_sharpe at {ncfg['central_mean']} -> tie set within {ncfg['sharpe_tol']} -> max W = min over "
                 f"(bpos, bneg) x all bench means of score -> W within {ncfg['worst_tol']}: smaller p_b + p_c, then smaller p_b")
    if not ncfg.get("select"):
        table: list[dict] = []
        would = None
        try:
            table, ch, _ = _neutralisation_scan(grid, ncfg)
            would = {"p_b": ch["p_b"], "p_c": ch["p_c"]}
        except (ValueError, KeyError) as exc:
            log.info("neutralisation table skipped: %s", exc)
        return {"p_b": float(bcfg.get("p_b", 0.5)), "p_c": float(bcfg.get("p_c", 0.25)), "p_source": "config", "criterion": None,
                "table": table, "reason": "blend.neutralisation.select is false -> config constants (simulation table for context only)",
                "would_choose": would, "z_source": z_source}
    table, ch, sm = _neutralisation_scan(grid, ncfg)
    ref = next(r for r in table if r["p_b"] == 0.0 and r["p_c"] == 0.0)
    if sm["n_eligible"] == 1:
        reason = "only the (0,0) cell is eligible -> no neutralisation"
    else:
        reason = (f"{sm['n_eligible']}/{sm['n_cells']} cells eligible; best mix score_sharpe {sm['sharpe_best']:.3f} at bench mean "
                  f"{ncfg['central_mean']}; {sm['n_tie']} in the tie set (within {ncfg['sharpe_tol']}); W best {sm['W_best']:.4f}, "
                  f"{sm['n_final']} within {ncfg['worst_tol']} -> simplest")
    reason += (f"; chosen p_b={ch['p_b']:g} p_c={ch['p_c']:g}: fit {_num(ch['fit'])} (vs {_num(ref['fit'])} at 0,0), score_sharpe "
               f"{_num(ch['score_sharpe'])} (vs {_num(ref['score_sharpe'])}), W {_num(ch['W'])}")
    log.info("neutralisation: %s", reason)
    return {"p_b": float(ch["p_b"]), "p_c": float(ch["p_c"]), "p_source": "simulation", "criterion": criterion, "table": table,
            "reason": reason, "z_source": z_source}


# --------------------------------------------------------------------------------------------------
# regime grid inputs
# --------------------------------------------------------------------------------------------------
def regime_inputs(run: dict, ids: pd.Index) -> tuple[pd.Series, pd.Series]:
    """(surrogate direction, core-proxy mean) on the OOF rows, from the raw train split + features.json."""
    cfg, feats = run["cfg"], run["features"]
    data_dir = Path(cfg["run"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir
    cols = feats["feature_cols"]
    raw = data.load_split("train", data_dir, columns=cols).reindex(ids)
    sur = surrogate.Surrogate.from_dict(feats["surrogate"])
    direction = sur.predict(data.features_linear(raw, cols, feats["missing_cols"]))
    core = features.core_proxy_series(raw, feats["core_proxy_cols"])
    return direction, core


def simulation_grids(cfg: dict) -> tuple[list[float], list[float]]:
    """``(p_b_grid, p_c_grid)`` for the run: ``simulation.p_c_grid`` (default ``[blend.p_c]``); with ``select: false`` the
    config ``p_c`` is added when missing so the old-shape ``regime`` slice at the config value exists."""
    bcfg = cfg["blend"]
    sim = bcfg.get("simulation") or {}
    p_b = [float(p) for p in sim.get("p_b_grid", (0.0, 0.25, 0.5, 0.75))]
    p_c = [float(p) for p in sim.get("p_c_grid", [float(bcfg["p_c"])])]
    if not neutralisation_config(cfg).get("select") and not any(abs(x - float(bcfg["p_c"])) < _EPS for x in p_c):
        p_c.append(float(bcfg["p_c"]))
    return p_b, p_c


def run_regime_grid(run: dict, z: pd.Series, oof: pd.DataFrame) -> dict:
    """``metrics.regime_grid`` of the chosen blend's OOF z; logs the mix table and the 500-draw cost extrapolation."""
    cfg = run["cfg"]
    sim = dict(cfg["blend"].get("simulation") or {})
    p_b_grid, p_c_grid = simulation_grids(cfg)
    direction, core = regime_inputs(run, oof.index)
    n_draws = int(sim.get("n_draws", 500))
    t0 = time.perf_counter()
    grid = metrics.regime_grid(z, oof[data.PRIMARY_TARGET], oof["era"], direction, core, p_b_grid=p_b_grid, p_c_grid=p_c_grid,
                               bench_fit_means=tuple(sim.get("bench_fit_means", (0.0, 0.009, 0.02))),
                               bench_fit_sd=float(sim.get("bench_fit_sd", 0.035)), n_draws=n_draws, seed=int(sim.get("seed", 0)))
    secs = time.perf_counter() - t0
    log.info("regime grid: %d draws x %d cells on %d eras in %.0fs (extrapolated 500 draws: %.0fs)", n_draws,
             len(p_b_grid) * len(p_c_grid), grid["meta"]["n_eras"], secs, secs * 500.0 / max(n_draws, 1))
    tbl = metrics.regime_grid_table(grid, "mix")
    log.info("regime grid (mix):\n%s", tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    grid["meta"]["seconds"] = round(secs, 1)
    return grid


# --------------------------------------------------------------------------------------------------
# final selection (16.1-16.4) + driver
# --------------------------------------------------------------------------------------------------
def final_selection(run: dict) -> dict:
    """Orchestrates 16.1-16.3 on a loaded run: correlation matrix, admission ladder (or everyone), hierarchical EW vs
    nested NNLS on the admitted set, cross-fitted headline, regime grid, neutralisation constants, ``final``.

    Returns the ``blend.json`` dict plus three private keys popped by :func:`blend_run`: ``_z`` (the reported OOF z,
    cross-fitted when NNLS is adopted), ``_corr`` (the correlation DataFrame) and ``_spearman`` (per-era Spearman of
    ``_z``, for the leak check). Nothing here reads ``v1_sherpa``: the regime grid's benchmark is synthetic.
    """
    cfg, oof, fwd, specs = run["cfg"], run["oof"], run["fwd"], run["specs"]
    bcfg, gates = cfg["blend"], cfg["gates"]
    names = [n for n in member_columns(oof) if n in specs]
    if not names:
        raise ValueError("no blendable member: oof.parquet has no m__<member> column with a finished spec.json")
    groups = {n: specs[n]["group"] for n in names}
    arms = {n: arm_of(specs[n]) for n in names}
    Z = gaussianise_per_era(oof[[f"m__{n}" for n in names]].rename(columns=lambda c: c[3:]), oof["era"])
    Zf = gaussianise_per_era(fwd[[f"m__{n}" for n in names]].rename(columns=lambda c: c[3:]), fwd["era"]) if fwd is not None else None
    corr = member_corr_matrix(Z, oof["era"])
    corr_max = float(gates.get("member_corr_max", DEFAULT_MEMBER_CORR_MAX))
    redundant: dict[str, dict] = {}
    if bcfg.get("admission"):
        admitted, ladder = admission_ladder(Z, oof, Zf, fwd, specs, cfg, corr=corr, redundant=redundant)
    else:
        admitted, ladder = list(names), []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if float(corr.loc[a, b]) > corr_max:
                    log.warning("members %s and %s correlate %.3f > %s (no blend.admission: both stay in)", a, b, float(corr.loc[a, b]), corr_max)
    if not admitted:
        raise ValueError("admission ladder admitted no member")
    ew_adm = blend_weights({n: groups[n] for n in admitted}, float(bcfg["everest_weight"]), {n: arms[n] for n in admitted})
    # keep blend_weights' key order (everest members first): blend_scores sums in dict order, so the order is part of
    # the bit-for-bit M1 reproduction; members not admitted get weight 0 (the bundle drops them)
    ew = {**ew_adm, **{n: 0.0 for n in names if n not in ew_adm}}
    log.info("members %s; admitted %s; hierarchical equal weights %s", names, admitted, _fmt_w(ew))
    out: dict = {"rule": "ew", "weights": ew, "ew_weights": ew, "p_b": float(bcfg["p_b"]), "p_c": float(bcfg["p_c"]),
                 "core_proxy_cols": run["features"]["core_proxy_cols"], "paired_vs_ew": None, "nnls": None,
                 "arms": arms, "admitted": admitted, "ladder": ladder, "redundant": redundant,
                 "member_corr": {"names": names, "matrix": [[float(v) for v in row] for row in corr.to_numpy()]}}
    Za = Z[admitted]
    Zfa = Zf[admitted] if Zf is not None else None
    if bcfg.get("nnls") and len(admitted) > 1 and int(oof["fold"].max()) >= 1:
        nn = nested_nnls(Za, oof, Zfa, fwd, ew_adm, cfg)
        out["nnls"], out["paired_vs_ew"] = nn, nn["paired_vs_ew"]
        if nn["adopt"]:
            out["rule"], out["weights"] = "nnls", {**nn["full_weights"], **{n: 0.0 for n in names if n not in nn["full_weights"]}}
    # z used for EVERY reported number: equal weights as is; adopted NNLS -> cross-fitted weights (out-of-sample)
    ew_summary, _, _ = evaluate_blend(blend_scores(Z, ew), oof, cfg)
    out["chosen_is_crossfit"] = False
    fwd_w = out["weights"]
    if out["rule"] == "nnls":
        z, cf_weights = crossfit_nnls(Za, oof)
        out["crossfit_weights"] = cf_weights
        out["chosen_is_crossfit"] = True
        insample_summary, _, _ = evaluate_blend(blend_scores(Z, out["weights"]), oof, cfg)
        out["chosen_insample_full_weights"] = insample_summary  # context only: in-sample w.r.t. the K weights
        log.info("NNLS adopted: reported fit20 uses cross-fitted weights (per fold %s); full-OOF weights (bundle) score %.4f in-sample",
                 {k: _fmt_w(w) for k, w in cf_weights.items()}, insample_summary["mean"])
        if fwd is not None:
            fwd_w, out["forward_weights_source"] = forward_nnls_weights(Za, oof, fwd, out["nnls"]["nested_weights"])
            out["forward_weights"] = fwd_w
    else:
        z = blend_scores(Z, out["weights"])
    chosen_summary, _, sp = evaluate_blend(z, oof, cfg)
    fwd_fit = None
    if Zf is not None and fwd is not None:
        fwd_fit = float(fit_series(blend_scores(Zf, fwd_w), fwd[data.PRIMARY_TARGET], fwd["era"]).mean())
    out["blend"] = {"ew": ew_summary, "chosen": chosen_summary, "rule": out["rule"], "forward_fit": fwd_fit, "spearman_mean": float(sp.mean()),
                    "chosen_is_crossfit": out["chosen_is_crossfit"]}
    log.info("blend[%s]: fit20 mean=%.4f se_block=%.4f sharpe=%.3f pct_pos=%.2f forward=%s (ew mean=%.4f)%s", out["rule"], chosen_summary["mean"],
             chosen_summary.get("se_block", float("nan")), chosen_summary["sharpe"], chosen_summary["pct_pos"],
             None if fwd_fit is None else round(fwd_fit, 4), ew_summary["mean"], " [cross-fitted weights]" if out["chosen_is_crossfit"] else "")
    out["paired_vs_baseline"] = paired_vs_baseline(z, oof, fwd, fwd_fit, cfg, names)
    # 16.3: regime grid on the reported z, neutralisation constants from the simulation only
    grid = run_regime_grid(run, z, oof)
    z_source = "crossfit" if out["chosen_is_crossfit"] else "ew"
    neu = select_neutralisation(grid, cfg, z_source=z_source)
    out["neutralisation"] = neu
    out["p_b"], out["p_c"], out["p_source"] = float(neu["p_b"]), float(neu["p_c"]), neu["p_source"]
    out["regime_grid"] = grid
    out["regime"] = metrics.regime_slice(grid, out["p_c"])
    row = next((r for r in neu["table"] if abs(r["p_b"] - out["p_b"]) < _EPS and abs(r["p_c"] - out["p_c"]) < _EPS), None)
    ck = _mean_key(grid, neutralisation_config(cfg)["central_mean"])
    cell = None
    if ck is not None:
        try:
            cell = grid[ck][metrics._grid_key(grid[ck], "p_c", out["p_c"])]
            cell = cell[metrics._grid_key(cell, "p_b", out["p_b"])]["mix"]
        except KeyError:
            cell = None
    adm_arms = []
    for r in ladder:
        if r["admitted"] and r["arm"] not in adm_arms:
            adm_arms.append(r["arm"])
    if not ladder:
        adm_arms = list(dict.fromkeys(arms[n] for n in admitted))
    out["final"] = {"admitted_arms": adm_arms, "rule": out["rule"], "p_b": out["p_b"], "p_c": out["p_c"], "p_source": out["p_source"],
                    "oof": chosen_summary, "forward_fit": fwd_fit,
                    "score_mix_central": None if cell is None else cell.get("score"),
                    "score_sharpe_central": None if cell is None else cell.get("score_sharpe"),
                    "W": None if row is None else row["W"]}
    log.info("final: arms %s rule %s p_b=%g p_c=%g (%s); OOF fit20 %s forward %s; central mix score %s score_sharpe %s",
             adm_arms, out["rule"], out["p_b"], out["p_c"], out["p_source"], _num(chosen_summary["mean"]), _num(fwd_fit),
             _num(out["final"]["score_mix_central"]), _num(out["final"]["score_sharpe_central"]))
    out["_z"], out["_corr"], out["_spearman"] = z, corr, sp
    return out


def blend_run(run_dir: Path) -> tuple[dict, int]:
    """Blend a finished training run; writes ``blend.json`` (+ ``member_corr.csv``) and extends ``metrics.json``.
    Returns (blend dict, exit code)."""
    run_dir = Path(run_dir)
    run = load_run(run_dir)
    cfg = run["cfg"]
    gates = cfg["gates"]
    out = final_selection(run)
    corr, sp = out.pop("_corr"), out.pop("_spearman")
    out.pop("_z")
    chosen_summary = out["blend"]["chosen"]
    notes: list[str] = []
    code = 0
    member_means = {n: (m.get("fit") or {}).get("mean") for n, m in (run["metrics"].get("members") or {}).items()}
    bmean, base = baseline_fit(cfg, {n: v for n, v in member_means.items() if v is not None})
    margin = float(gates["suspect_gain_over_baseline"])
    out["suspect"] = bool(bmean is not None and metrics.suspect_gain(chosen_summary["mean"], bmean, margin))
    if out["suspect"]:
        banner = SUSPECT_BANNER.format(blend=chosen_summary["mean"], base=base, bmean=bmean, margin=margin)
        log.warning(banner)
        notes.append(banner)
    try:
        metrics.leak_check(sp, float(gates["leak_spearman"]))
        out["leak_check"] = "pass"
    except RuntimeError as exc:
        out["leak_check"] = "FAIL"
        notes.append(str(exc))
        log.error("%s", exc)
        code = 2
    for n, r in out["redundant"].items():
        notes.append(f"{n}: dropped as redundant({r['other']}) corr {r['corr']:.3f} > gates.member_corr_max")
    corr.to_csv(run_dir / "member_corr.csv", float_format="%.4f", index_label="member")
    write_json(run_dir / "blend.json", out)
    mets = run["metrics"]
    mets["blend"] = {**out["blend"], "weights": out["weights"], "suspect": out["suspect"], "final": out["final"],
                     "admitted": out["admitted"],
                     "neutralisation": {k: out["neutralisation"].get(k) for k in ("p_b", "p_c", "p_source", "reason", "z_source")}}
    mets["regime"] = out["regime"]
    paired = mets.setdefault("paired", {})
    if out["paired_vs_ew"] is not None:
        paired["blend_nnls_vs_ew"] = out["paired_vs_ew"]
    if out.get("paired_vs_baseline") is not None:
        pvb = out["paired_vs_baseline"]
        paired[f"blend_vs_{pvb['baseline']}"] = {k: v for k, v in pvb.items() if k != "baseline"}
    for r in out["ladder"]:
        if r.get("diff_mean") is not None:
            paired[f"{r['arm']}_ladder"] = {k: r[k] for k in ("diff_mean", "diff_se", "p_pos", "fwd_diff", "dsharpe", "p_pos_sharpe",
                                                               "corr_with_incumbent", "admitted")}
    mets["notes"] = list(dict.fromkeys(list(mets.get("notes") or []) + notes))  # re-running blend (pipeline resume) adds nothing twice
    if out["leak_check"] == "FAIL":
        mets["leak_check"], mets["status"] = "FAIL", "failed"
    write_json(run_dir / "metrics.json", mets)
    log.info("wrote %s, member_corr.csv and updated metrics.json", run_dir / "blend.json")
    return out, code


def main(argv: list[str] | None = None) -> int:
    """``python -m stealth.blend --run-id ID [--artifacts DIR]``."""
    parser = argparse.ArgumentParser(prog="python -m stealth.blend")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    args = parser.parse_args(argv)
    run_dir = Path(args.artifacts) / args.run_id
    if not (run_dir / "oof.parquet").exists():
        print(f"{run_dir}/oof.parquet missing; run stealth.train first", file=sys.stderr)
        return 1
    fh = setup_logging(run_dir)
    try:
        _, code = blend_run(run_dir)
        return code
    except Exception:  # noqa: BLE001 - CLI boundary
        log.exception("blend failed")
        return 1
    finally:
        logging.getLogger("stealth").removeHandler(fh)
        fh.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
