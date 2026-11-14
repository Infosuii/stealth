"""Reserved-partition audit of the FROZEN blend: ``python -m stealth.audit --run-id R [--artifacts DIR]``.

Contract: docs/INTERFACES.md section 22 (C, D). Runs after ``stealth.blend`` and before ``stealth.bundle``:

* inputs: ``artifacts/<R>/audit.parquet`` (written by ``stealth.train``: the reserved eras predicted by EXACTLY the
  full-refit blobs the bundle averages, one ``m__<member>`` column each, plus ``baseline__<member>`` = the unsearched,
  default-universe baseline refit on the same pre-audit eras) and ``blend.json`` (weights, rule, p_b / p_c frozen by
  the development selection). Every development choice -- CV, forward block, knobs, search, admission ladder, NNLS,
  count sources, neutralisation grid -- and every fit used eras before ``cv.audit.train_end`` only;
* output: ``audit.json`` (schema ``stealth.audit/1``): the frozen blend's UNNEUTRALISED per-era fit20 on the reserved
  eras (summary, Sharpe), the paired block bootstrap against the baseline (diff, p_pos, null-centred p_value, validity),
  a paired Sharpe bootstrap, the cumulative-fit20 drawdown (labelled: NOT platform P&L), chronological stability
  segments, bootstrap block-length sensitivity, the frozen core-proxy step alone (the train-split benchmark is in-sample,
  so the benchmark step is evaluated only in the synthetic ``scenario``) and a verdict that separates technical
  completion from an audit win;
* it NEVER writes ``blend.json``, never changes a weight or a constant, and nothing reads ``audit.json`` to make a
  choice. Re-running it recomputes the same numbers. The audit is independent of THIS run's selection only: earlier
  handoffs (02) have seen these eras.

Exit 0 ok, 1 error (audit disabled, inputs missing or inconsistent).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from stealth import cv, data, metrics
from stealth.blend import blend_scores, gaussianise_per_era, neutralisation_config, regime_inputs
from stealth.train import ARTIFACTS, audit_config, per_era_fit_fast, per_era_spearman_fast, setup_logging, write_json

log = logging.getLogger("stealth.audit")  # explicit: under `python -m` __name__ is __main__

SCHEMA = "stealth.audit/1"
DRAWDOWN_LABEL = "cumulative per-era fit20 drawdown (NOT platform P&L: no stake, no arctan bound, no UNQ/INOV)"
SENSITIVITY_BLOCKS = (10, 20, 40)
N_SEGMENTS = 4
EXIT_OK, EXIT_ERROR = 0, 1


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def drawdown(fit: pd.Series) -> dict:
    """Max drawdown of the cumulative per-era fit20 (peak includes the 0 start) and the longest run of eras below the
    running peak. A descriptive staking-risk figure, NOT platform P&L."""
    cum = pd.Series(fit).sort_index().cumsum().to_numpy(dtype=np.float64)
    if cum.size == 0:
        return {"max_drawdown": None, "longest_underwater_eras": None, "final_cum": None}
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    dd = peak - cum
    longest = run = 0
    for under in dd > 1e-15:
        run = run + 1 if under else 0
        longest = max(longest, run)
    return {"max_drawdown": float(dd.max()), "longest_underwater_eras": int(longest), "final_cum": float(cum[-1])}


def segments(fit_blend: pd.Series, fit_base: pd.Series | None, n: int = N_SEGMENTS) -> dict:
    """``n`` chronological, contiguous era segments: mean fit20 of the blend, the baseline and their difference."""
    eras = np.sort(fit_blend.index.to_numpy())
    rows = []
    for part in np.array_split(eras, n):
        if part.size == 0:
            continue
        fb = float(fit_blend.loc[part].mean())
        fbase = None if fit_base is None else float(fit_base.reindex(part).mean())
        rows.append({"eras": [int(part.min()), int(part.max())], "n_eras": int(part.size), "blend_fit": fb,
                     "baseline_fit": fbase, "diff": None if fbase is None else fb - fbase})
    return {"segments": rows, "n_segments": len(rows), "blend_positive_segments": int(sum(r["blend_fit"] > 0 for r in rows)),
            "diff_positive_segments": None if fit_base is None else int(sum((r["diff"] or 0.0) > 0 for r in rows))}


def _summary(s: pd.Series, bs: dict, block: int | None = None) -> dict:
    return metrics.summarise(s, block=int(block or bs["block"]), n_boot=int(bs["n_boot"]), seed=int(bs["seed"]))


def _mix(cell: dict) -> dict:
    return {k: cell.get(k) for k in ("fit", "unq", "inov", "score", "score_sharpe", "score_se")}


def _grid_cell(grid: dict, mean_key: str, p_b: float, p_c: float) -> dict:
    """``{bpos, bneg, mix}`` of one (p_b, p_c) cell of a :func:`metrics.regime_grid` result at one bench mean."""
    by_pc = grid[mean_key]
    by_pb = by_pc[metrics._grid_key(by_pc, "p_c", p_c)]
    return by_pb[metrics._grid_key(by_pb, "p_b", p_b)]


def audit_run(run_dir: Path) -> tuple[dict, int]:
    """Score the frozen blend once on the reserved eras; writes ``audit.json``; returns (doc, exit code)."""
    run_dir = Path(run_dir)
    t0 = time.perf_counter()
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    acfg = audit_config(cfg)
    if acfg is None:
        raise ValueError("cv.audit is not enabled in this run's config.yaml: nothing to audit")
    for need in ("audit.parquet", "blend.json", "run_fingerprint.json", "metrics.json", "features.json"):
        if not (run_dir / need).exists():
            raise FileNotFoundError(f"{run_dir / need} missing: run stealth.train and stealth.blend first")
    fingerprint = json.loads((run_dir / "run_fingerprint.json").read_text())["fingerprint"]
    mets = json.loads((run_dir / "metrics.json").read_text())
    if mets.get("run_fingerprint") != fingerprint:
        raise ValueError("metrics.json and run_fingerprint.json disagree: re-run stealth.train --resume")
    blend_path = run_dir / "blend.json"
    blend = json.loads(blend_path.read_text())
    weights = {str(k): float(v) for k, v in (blend.get("weights") or {}).items() if float(v) > 0.0}
    if not weights:
        raise ValueError("blend.json has no member with a positive weight")
    audit = pd.read_parquet(run_dir / "audit.parquet")
    missing = [n for n in weights if f"m__{n}" not in audit.columns]
    if missing:
        raise ValueError(f"audit.parquet lacks the weighted member(s) {missing}: the blend and the audit predictions disagree "
                         "(re-run stealth.train --resume, then stealth.blend)")
    era, y = audit["era"], audit[data.PRIMARY_TARGET]
    bs = cfg["bootstrap"]
    gates = cfg["gates"]
    # the frozen blend exactly as the pickle builds it before neutralisation: per-era gaussianised member scores, weighted
    Z = gaussianise_per_era(audit[[f"m__{n}" for n in weights]].rename(columns=lambda c: c[3:]), era)
    z = blend_scores(Z, weights)
    fit_blend = per_era_fit_fast(z, y, era)
    base_cols = [c for c in audit.columns if c.startswith("baseline__")]
    base_col = base_cols[0] if base_cols else None
    fit_base = per_era_fit_fast(audit[base_col], y, era) if base_col else None
    doc: dict = {"schema": SCHEMA, "run_id": run_dir.name, "status": "ok", "created_at": _now(), "run_fingerprint": fingerprint,
                 "blend_sha256": _sha256(blend_path),
                 "partition": {"train_end": int(acfg["train_end"]), "test_start": int(acfg["test_start"]),
                               "eras": [int(era.min()), int(era.max())], "n_eras": int(era.nunique()), "n_rows": int(len(audit))},
                 "frozen": {"rule": blend.get("rule"), "members": sorted(weights),
                            "admitted_arms": (blend.get("final") or {}).get("admitted_arms"), "p_b": blend.get("p_b"),
                            "p_c": blend.get("p_c"), "p_source": blend.get("p_source")},
                 "blend": _summary(fit_blend, bs),
                 "blend_spearman_mean": float(per_era_spearman_fast(z, y, era).mean())}
    notes = [f"reserved eras {int(era.min())}..{int(era.max())}: no fit, early stop, knob, search, ladder, NNLS or "
             f"neutralisation choice of this run saw them (development and full refits stop before {int(acfg['train_end'])})",
             "independent of THIS run's selection only: handoff 02 (test01) has already looked at these eras",
             "scored once after the blend was frozen; audit.json never feeds back into any choice",
             "unneutralised: the frozen blend before the benchmark / core-proxy steps (the train-split benchmark is in-sample)"]
    if base_col is not None:
        doc["baseline"] = {"name": base_col[len("baseline__"):], "label": "unsearched config params, default universe, refit on "
                           "the pre-audit eras", **_summary(fit_base, bs)}
        diff = (fit_blend - fit_base).dropna()
        doc["paired_vs_baseline"] = cv.paired_block_bootstrap(diff, block=int(bs["block"]), n_boot=int(bs["n_boot"]),
                                                              seed=int(bs["seed"])).as_dict()
        doc["sharpe_vs_baseline"] = metrics.paired_sharpe_bootstrap(fit_blend, fit_base, block=int(bs["block"]),
                                                                    n_boot=int(bs["n_boot"]), seed=int(bs["seed"]))
    else:
        doc["baseline"] = None
        doc["paired_vs_baseline"] = doc["sharpe_vs_baseline"] = None
        notes.append("audit.parquet has no baseline__ column: no paired comparison")
    doc["drawdown"] = {"label": DRAWDOWN_LABEL, "blend": drawdown(fit_blend),
                       "baseline": None if fit_base is None else drawdown(fit_base)}
    doc["stability"] = segments(fit_blend, fit_base)
    sens = []
    for b in SENSITIVITY_BLOCKS:
        row = {"block": int(b), "se_block": _summary(fit_blend, bs, b)["se_block"]}
        if fit_base is not None:
            pr = cv.paired_block_bootstrap((fit_blend - fit_base).dropna(), block=int(b), n_boot=int(bs["n_boot"]),
                                           seed=int(bs["seed"]))
            row.update({"p_pos": pr.p_pos, "p_value": pr.p_value, "ci90": list(pr.ci90), "n_blocks": pr.n_blocks,
                        "inference_valid": pr.inference_valid})
        sens.append(row)
    doc["bootstrap_sensitivity"] = sens
    # the frozen constants only: core-proxy step (real features) and the synthetic-benchmark scenario at (p_b, p_c)
    p_b, p_c = float(blend.get("p_b") or 0.0), float(blend.get("p_c") or 0.0)
    run = {"cfg": cfg, "features": json.loads((run_dir / "features.json").read_text())}
    direction, core = regime_inputs(run, audit.index)
    P = metrics._regime_prepare(z, y, era, direction, core)
    _, f_pc, _, i_pc = metrics.cell_scores(P, None, 0.0, p_c)
    _, f_00, _, i_00 = metrics.cell_scores(P, None, 0.0, 0.0)
    doc["frozen_core_step"] = {"p_c": p_c, "fit": float(np.mean(f_pc)), "inov_proxy": float(np.mean(i_pc)),
                               "fit_unneutralised": float(np.mean(f_00)), "inov_proxy_unneutralised": float(np.mean(i_00)),
                               "note": "INOV vs the core PROXY (true core set unpublished); benchmark step not applied"}
    sim = (cfg.get("blend") or {}).get("simulation") or {}
    ncfg = neutralisation_config(cfg)
    means = tuple(float(m) for m in sim.get("bench_fit_means", (0.0, 0.009, 0.02)))
    central = float(ncfg["central_mean"]) if any(abs(float(ncfg["central_mean"]) - m) < 1e-12 for m in means) else means[0]
    grid = metrics.regime_grid(z, y, era, direction, core, p_b_grid=sorted({0.0, p_b}), p_c_grid=sorted({0.0, p_c}),
                               bench_fit_means=means, bench_fit_sd=float(sim.get("bench_fit_sd", 0.035)),
                               n_draws=int(sim.get("n_draws", 500)), seed=int(sim.get("seed", 0)))
    ck = next(k for k in grid if k != "meta" and abs(float(k) - central) < 1e-12)
    chosen, ref = _grid_cell(grid, ck, p_b, p_c), _grid_cell(grid, ck, 0.0, 0.0)
    doc["scenario"] = {"note": "synthetic benchmark (N(mean, sd) per era, surrogate direction) on the reserved eras at the FROZEN "
                               "constants only; not a selection input", "p_b": p_b, "p_c": p_c, "bench_fit_mean": central,
                       "n_draws": int(sim.get("n_draws", 500)), "mix": _mix(chosen["mix"]), "bneg": _mix(chosen["bneg"]),
                       "reference_00": _mix(ref["mix"])}
    pv = doc["paired_vs_baseline"]
    thr = float(gates["p_pos_adopt"])
    strength = None
    if pv is None:
        win, text = None, "no baseline column: technical completion only, no audit comparison"
    elif pv["inference_valid"] and pv["diff_mean"] > 0 and pv["p_pos"] >= thr:
        win = True
        strength = "strong" if pv["p_value"] < 0.05 else "weak"
        text = (f"audit win by the pre-registered paired rule: the frozen blend beats the unsearched baseline on the "
                f"reserved eras (diff {pv['diff_mean']:+.4f}, p_pos {pv['p_pos']:.3f} >= {thr}); one-sided p_value "
                f"{pv['p_value']:.3f}" + ("" if strength == "strong" else " -- not significant at 5%: weak evidence"))
    else:
        win, text = False, (f"no audit win: tie or loss against the unsearched baseline on the reserved eras (diff "
                            f"{pv['diff_mean']:+.4f}, p_pos {pv['p_pos']:.3f}, p_value {pv['p_value']:.3f}"
                            + ("" if pv["inference_valid"] else ", too few blocks for inference") + "): technical completion only")
    doc["verdict"] = {"technical": "complete", "audit_win": win, "strength": strength, "text": text}
    doc["notes"] = notes
    doc["seconds"] = round(time.perf_counter() - t0, 1)
    write_json(run_dir / "audit.json", doc)
    b = doc["blend"]
    log.info("audit %s eras %s..%s: blend fit20 %.4f (se %.4f, sharpe %.3f)%s; drawdown %.4f; %s", run_dir.name,
             doc["partition"]["eras"][0], doc["partition"]["eras"][1], b["mean"], b["se_block"], b["sharpe"],
             "" if pv is None else f" vs baseline {doc['baseline']['mean']:.4f} (diff {pv['diff_mean']:+.4f}, p_pos {pv['p_pos']:.3f})",
             doc["drawdown"]["blend"]["max_drawdown"], text)
    return doc, EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """``python -m stealth.audit --run-id ID [--artifacts DIR]``."""
    parser = argparse.ArgumentParser(prog="python -m stealth.audit", description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    args = parser.parse_args(argv)
    run_dir = Path(args.artifacts) / args.run_id
    if not run_dir.exists():
        print(f"{run_dir} does not exist; run stealth.train first", file=sys.stderr)
        return EXIT_ERROR
    fh = setup_logging(run_dir)
    try:
        _, code = audit_run(run_dir)
        return code
    except Exception as exc:  # noqa: BLE001 - CLI boundary: a failed audit is recorded, never silently skipped
        log.exception("audit failed")
        write_json(run_dir / "audit.json", {"schema": SCHEMA, "run_id": run_dir.name, "status": "failed", "created_at": _now(),
                                            "error": f"{type(exc).__name__}: {exc}"})
        return EXIT_ERROR
    finally:
        logging.getLogger("stealth").removeHandler(fh)
        fh.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
