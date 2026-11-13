"""Equal-weight / nested NNLS blend of member OOF scores, neutralisation regime simulation, ``blend.json``.

Contracts: docs/INTERFACES.md sections 8 and 10; protocol: docs/PLAN.md "Neutralisation / post-processing".

* every member's RAW OOF score is gaussianised per era with ``_runtime.gaussianise`` (the same transform the
  pickle applies at predict time);
* default rule = equal weight within target group, everest group ``blend.everest_weight`` / aux the rest;
* NNLS weights (sum to 1, floor 0.02) are computed NESTED: fitted on CV folds ``0..K-2``, evaluated on fold
  ``K-1`` (paired block bootstrap vs equal weight) and on the forward block; adopted only when
  ``p_pos >= gates.p_pos_adopt`` and the forward difference is ``>= 0``. Otherwise equal weights stay. When NNLS is
  adopted the bundle gets ``full_weights`` (fitted on all OOF eras) but every REPORTED number (``chosen`` fit20, the
  SUSPECT and leak checks, the regime simulation) comes from CROSS-FITTED weights: for each fold ``k`` the weights are
  fitted on folds ``!= k`` and applied to fold ``k`` (``crossfit_nnls``), so the headline stays out-of-sample;
* the blended OOF ``z`` then goes through ``metrics.regime_simulation`` with the surrogate direction and the
  core-proxy frame recorded by ``train.py`` in ``features.json``;
* gates: ``metrics.leak_check`` on the blend's per-era Spearman; a "SUSPECT" banner when the blend's OOF fit20
  exceeds the baseline (``gates.baseline_fit`` if set, else the in-run baseline member) by more than
  ``gates.suspect_gain_over_baseline``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from stealth import _runtime, cv, data, features, metrics, surrogate
from stealth.train import ARTIFACTS, REPO_ROOT, baseline_fit, setup_logging, write_json

log = logging.getLogger("stealth.blend")  # explicit: under `python -m` __name__ is __main__

WEIGHT_FLOOR = 0.02
SUSPECT_BANNER = "SUSPECT — winner's curse or leak: blended OOF fit20 {blend:.4f} > {base} {bmean:.4f} + {margin}"


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


def blend_weights(groups: dict[str, str], everest_weight: float) -> dict[str, float]:
    """Equal weight within group; everest group gets ``everest_weight`` (all of it when there is no aux group)."""
    ev = [n for n, g in groups.items() if g == "everest"]
    aux = [n for n, g in groups.items() if g != "everest"]
    if not ev and not aux:
        raise ValueError("no members to blend")
    w_ev = float(everest_weight) if aux else 1.0
    if not ev:
        w_ev = 0.0
    weights = {n: w_ev / len(ev) for n in ev}
    weights.update({n: (1.0 - w_ev) / len(aux) for n in aux})
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
    return metrics.per_era_fit(z, y, era)


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


def _fmt_w(w: dict[str, float]) -> str:
    return "{" + ", ".join(f"{k}: {v:.3f}" for k, v in w.items()) + "}"


# --------------------------------------------------------------------------------------------------
# evaluation + regime simulation
# --------------------------------------------------------------------------------------------------
def evaluate_blend(z: pd.Series, frame: pd.DataFrame, cfg: dict) -> tuple[dict, pd.Series, pd.Series]:
    """``(summarise(fit), fit series, spearman series)`` of a blended z on an oof/forward frame."""
    bs = cfg["bootstrap"]
    y, era = frame[data.PRIMARY_TARGET], frame["era"]
    fit = fit_series(z, y, era)
    sp = metrics.per_era_spearman(z, y, era)
    return metrics.summarise(fit, block=int(bs["block"]), n_boot=int(bs["n_boot"]), seed=int(bs["seed"])), fit, sp


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


def run_regime(run: dict, z: pd.Series, oof: pd.DataFrame) -> dict:
    sim = dict(run["cfg"]["blend"].get("simulation") or {})
    direction, core = regime_inputs(run, oof.index)
    t0 = time.perf_counter()
    res = metrics.regime_simulation(z, oof[data.PRIMARY_TARGET], oof["era"], direction, core,
                                    p_b_grid=tuple(sim.get("p_b_grid", (0.0, 0.25, 0.5, 0.75))), p_c=float(run["cfg"]["blend"]["p_c"]),
                                    bench_fit_means=tuple(sim.get("bench_fit_means", (0.0, 0.009, 0.02))),
                                    bench_fit_sd=float(sim.get("bench_fit_sd", 0.035)), n_draws=int(sim.get("n_draws", 500)),
                                    seed=int(sim.get("seed", 0)))
    log.info("regime simulation: %d draws in %.0fs", int(sim.get("n_draws", 500)), time.perf_counter() - t0)
    for regime in ("bpos", "bneg", "mix"):
        tbl = metrics.regime_table(res, regime)
        log.info("regime %s:\n%s", regime, tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return res


# --------------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------------
def blend_run(run_dir: Path) -> tuple[dict, int]:
    """Blend a finished training run; writes ``blend.json`` and extends ``metrics.json``. Returns (blend dict, exit code)."""
    run = load_run(run_dir)
    cfg, oof, fwd, specs = run["cfg"], run["oof"], run["fwd"], run["specs"]
    bcfg, gates = cfg["blend"], cfg["gates"]
    names = [n for n in member_columns(oof) if n in specs]
    groups = {n: specs[n]["group"] for n in names}
    Z = gaussianise_per_era(oof[[f"m__{n}" for n in names]].rename(columns=lambda c: c[3:]), oof["era"])
    Zf = gaussianise_per_era(fwd[[f"m__{n}" for n in names]].rename(columns=lambda c: c[3:]), fwd["era"]) if fwd is not None else None
    ew = blend_weights(groups, float(bcfg["everest_weight"]))
    log.info("members %s; equal weights %s", names, _fmt_w(ew))
    out: dict = {"rule": "ew", "weights": ew, "ew_weights": ew, "p_b": float(bcfg["p_b"]), "p_c": float(bcfg["p_c"]),
                 "core_proxy_cols": run["features"]["core_proxy_cols"], "paired_vs_ew": None, "nnls": None}
    if bcfg.get("nnls") and len(names) > 1 and int(oof["fold"].max()) >= 1:
        nn = nested_nnls(Z, oof, Zf, fwd, ew, cfg)
        out["nnls"], out["paired_vs_ew"] = nn, nn["paired_vs_ew"]
        if nn["adopt"]:
            out["rule"], out["weights"] = "nnls", nn["full_weights"]
    # z used for EVERY reported number: equal weights as is; adopted NNLS -> cross-fitted weights (out-of-sample)
    ew_summary, _, _ = evaluate_blend(blend_scores(Z, ew), oof, cfg)
    out["chosen_is_crossfit"] = False
    if out["rule"] == "nnls":
        z, cf_weights = crossfit_nnls(Z, oof)
        out["crossfit_weights"] = cf_weights
        out["chosen_is_crossfit"] = True
        insample_summary, _, _ = evaluate_blend(blend_scores(Z, out["weights"]), oof, cfg)
        out["chosen_insample_full_weights"] = insample_summary  # context only: in-sample w.r.t. the K weights
        log.info("NNLS adopted: reported fit20 uses cross-fitted weights (per fold %s); full-OOF weights (bundle) score %.4f in-sample",
                 {k: _fmt_w(w) for k, w in cf_weights.items()}, insample_summary["mean"])
    else:
        z = blend_scores(Z, out["weights"])
    chosen_summary, fit, sp = evaluate_blend(z, oof, cfg)
    fwd_fit = None
    if Zf is not None and fwd is not None:
        fwd_fit = float(fit_series(blend_scores(Zf, out["weights"]), fwd[data.PRIMARY_TARGET], fwd["era"]).mean())
    out["blend"] = {"ew": ew_summary, "chosen": chosen_summary, "rule": out["rule"], "forward_fit": fwd_fit, "spearman_mean": float(sp.mean()),
                    "chosen_is_crossfit": out["chosen_is_crossfit"]}
    log.info("blend[%s]: fit20 mean=%.4f se_block=%.4f sharpe=%.3f pct_pos=%.2f forward=%s (ew mean=%.4f)%s", out["rule"], chosen_summary["mean"],
             chosen_summary.get("se_block", float("nan")), chosen_summary["sharpe"], chosen_summary["pct_pos"],
             None if fwd_fit is None else round(fwd_fit, 4), ew_summary["mean"], " [cross-fitted weights]" if out["chosen_is_crossfit"] else "")
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
    out["regime"] = run_regime(run, z, oof)
    write_json(run_dir / "blend.json", out)
    mets = run["metrics"]
    mets["blend"] = {**out["blend"], "weights": out["weights"], "suspect": out["suspect"]}
    mets["regime"] = out["regime"]
    if out["paired_vs_ew"] is not None:
        mets.setdefault("paired", {})["blend_nnls_vs_ew"] = out["paired_vs_ew"]
    mets.setdefault("notes", []).extend(notes)
    if out["leak_check"] == "FAIL":
        mets["leak_check"], mets["status"] = "FAIL", "failed"
    write_json(run_dir / "metrics.json", mets)
    log.info("wrote %s and updated metrics.json", run_dir / "blend.json")
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
