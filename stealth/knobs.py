"""Training-universe knobs (M2): era start, n_rows filter/weight, time decay -- decided ONCE on the forward block.

Contract: docs/INTERFACES.md sections 11-12. Module-level imports are stdlib, numpy, pandas, ``stealth.data`` and
``stealth.cv`` ONLY: the optional GPU venv (``scripts/mlp_train.py``) imports this module to apply the same training
mask and weights. ``stealth.metrics`` / ``stealth.train`` are imported inside :func:`select_universe`.

* ``run.min_era`` stays the SCORING universe (CV test blocks + the forward test block). A :class:`Universe` only
  masks and weights TRAINING rows (fold train / inner-train fits and full refits); test and prediction rows are never
  touched, so every comparison between universes is paired on identical eras.
* candidates = the default plus ONE knob changed at a time (no factorial), in config order.
* decision: ``knobs.member`` at its BASE params under every candidate; per alternative the forward-block paired block
  bootstrap of per-era fit20 (alternative minus default); Holm step-down at ``alpha_total`` on ``1 - p_pos_fwd``; an
  alternative passes iff Holm passes AND the forward diff is ``>= 0`` AND the mean per-era diff over the CV folds whose
  test blocks lie ``>= 20`` eras from the forward test block is ``>= 0`` (independent confirmation). At most one winner
  per dimension (largest forward diff); the combination is reported, never re-tested.
* ``knobs.json`` is written once per run-id; a config whose candidates differ raises ``ValueError``.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stealth import cv as cvmod
from stealth import data

log = logging.getLogger(__name__)

ROWS_RULE_RE = re.compile(r"^(none|filter(\d+)|weight(\d+))$")
DIMENSIONS = ("era_start", "rows_rule", "time_decay")
REDECIDE_MSG = "knobs already decided for this run-id; start a new run-id"


def parse_rows_rule(rule: str) -> tuple[str, int | None]:
    """``"none" -> ("none", None)``, ``"filter80" -> ("filter", 80)``, ``"weight80" -> ("weight", 80)``."""
    m = ROWS_RULE_RE.match(str(rule))
    if not m:
        raise ValueError(f"rows_rule {rule!r} must be none | filter<N> | weight<N>")
    if m.group(2):
        return "filter", int(m.group(2))
    if m.group(3):
        return "weight", int(m.group(3))
    return "none", None


@dataclass(frozen=True)
class Universe:
    """Training-row universe: rows of eras ``>= era_start``, an n_rows rule, optional linear time decay."""

    era_start: int
    rows_rule: str = "none"
    time_decay: bool = False
    rows_low_weight: float = 0.5

    def __post_init__(self) -> None:
        parse_rows_rule(self.rows_rule)
        if not 0.0 < float(self.rows_low_weight) <= 1.0:
            raise ValueError("rows_low_weight must be in (0, 1]")

    def key(self) -> str:
        """Stable cache key, e.g. ``e2800-none-d0`` / ``e2800-weight80w0.5-d1``."""
        kind, _ = parse_rows_rule(self.rows_rule)
        rows = self.rows_rule + (f"w{float(self.rows_low_weight):g}" if kind == "weight" else "")
        return f"e{int(self.era_start)}-{rows}-d{int(bool(self.time_decay))}"

    def to_dict(self) -> dict:
        return {"era_start": int(self.era_start), "rows_rule": str(self.rows_rule), "time_decay": bool(self.time_decay),
                "rows_low_weight": float(self.rows_low_weight)}

    @classmethod
    def from_dict(cls, d: dict) -> "Universe":
        return cls(era_start=int(d["era_start"]), rows_rule=str(d.get("rows_rule", "none")),
                   time_decay=bool(d.get("time_decay", False)), rows_low_weight=float(d.get("rows_low_weight", 0.5)))

    def train_mask(self, era: np.ndarray, era_n_rows: np.ndarray) -> np.ndarray:
        """Boolean per TRAINING row: ``era >= era_start`` and, for ``filter<N>``, the row's era has ``>= N`` rows."""
        era = np.asarray(era)
        mask = era >= int(self.era_start)
        kind, n = parse_rows_rule(self.rows_rule)
        if kind == "filter":
            mask &= np.asarray(era_n_rows) >= int(n)
        return mask

    def rows_factor(self, era_n_rows: np.ndarray) -> np.ndarray | None:
        """Per-row weight of the n_rows rule alone (``weight<N>``: eras with ``< N`` rows get ``rows_low_weight``)."""
        kind, n = parse_rows_rule(self.rows_rule)
        if kind != "weight":
            return None
        nr = np.asarray(era_n_rows)
        return np.where(nr < int(n), float(self.rows_low_weight), 1.0).astype(np.float64)

    def weights(self, era: np.ndarray, era_n_rows: np.ndarray) -> np.ndarray | None:
        """Sample weights for the rows of ONE fit (already masked): rows-rule weight x ``data.time_decay_weights`` over the
        eras of this fit; ``None`` when neither is active (unweighted fit, the M1 behaviour)."""
        w = self.rows_factor(era_n_rows)
        if self.time_decay:
            td = data.time_decay_weights(pd.Series(np.asarray(era)), True)
            w = td if w is None else w * td
        return w


def default_universe(cfg: dict) -> Universe:
    """``era_start = run.min_era`` (3 when null), rows rule ``none``, ``time_decay = run.time_decay``."""
    run = cfg["run"]
    kn = cfg.get("knobs") or {}
    start = run.get("min_era")
    return Universe(era_start=int(start) if start is not None else 3, rows_rule="none", time_decay=bool(run.get("time_decay")),
                    rows_low_weight=float(kn.get("rows_low_weight", 0.5)))


def candidates(cfg: dict) -> list[Universe]:
    """``[default] + one-knob-at-a-time alternatives`` in config order (era_start, rows_rule, time_decay); no factorial."""
    d = default_universe(cfg)
    kn = cfg.get("knobs") or {}
    out = [d]
    for v in kn.get("era_start") or []:
        out.append(dataclasses.replace(d, era_start=int(v)))
    for v in kn.get("rows_rule") or []:
        out.append(dataclasses.replace(d, rows_rule=str(v)))
    for v in kn.get("time_decay") or []:
        out.append(dataclasses.replace(d, time_decay=bool(v)))
    seen: set[str] = set()
    uniq = []
    for u in out:
        if u.key() not in seen:
            seen.add(u.key())
            uniq.append(u)
    return uniq


def pool_min_era(cfg: dict) -> int | None:
    """Lowest era any candidate trains on: ``min(run.min_era, min(knobs.era_start))`` (None = whole split)."""
    start = cfg["run"].get("min_era")
    kn = cfg.get("knobs") or {}
    if start is None:
        return None
    return int(min([int(start)] + [int(v) for v in kn.get("era_start") or []]))


def dimension_of(u: Universe, default: Universe) -> str | None:
    """The single knob in which ``u`` differs from ``default`` (None for the default itself)."""
    diff = [k for k in DIMENSIONS if getattr(u, k) != getattr(default, k)]
    if not diff:
        return None
    if len(diff) > 1:
        raise ValueError(f"candidate {u.key()} changes more than one knob vs {default.key()}")
    return diff[0]


def equivalent(u: Universe, default: Universe, era: np.ndarray, era_n_rows: np.ndarray) -> bool:
    """True when ``u`` selects and weights exactly the same pool rows as ``default`` (its fits would be identical)."""
    if u.time_decay != default.time_decay:
        return False
    if not np.array_equal(u.train_mask(era, era_n_rows), default.train_mask(era, era_n_rows)):
        return False
    m = default.train_mask(era, era_n_rows)
    fu, fd = u.rows_factor(era_n_rows), default.rows_factor(era_n_rows)
    fu = np.ones(len(era)) if fu is None else fu
    fd = np.ones(len(era)) if fd is None else fd
    return bool(np.array_equal(fu[m], fd[m]))


def holm_decide(rows: list[dict], alpha: float) -> list[dict]:
    """Holm step-down over the alternative rows (``is_default`` False) on ``1 - p_pos_fwd``; mutates and returns ``rows``.

    Sorted by ``1 - p_pos_fwd`` (ties: config order); rank ``j`` (1-based) of ``m`` passes Holm iff
    ``1 - p_pos_fwd <= alpha / (m - j + 1)`` and every earlier rank passed (stop at the first failure). ``pass`` =
    Holm AND ``diff_fwd >= 0`` AND ``diff_cv_indep >= 0`` (None -> no independent confirmation -> fail).
    """
    alts = [(i, r) for i, r in enumerate(rows) if not r.get("is_default")]
    m = len(alts)
    order = sorted(alts, key=lambda ir: (1.0 - float(ir[1]["p_pos_fwd"]), ir[0]))
    stopped = False
    for j, (_, r) in enumerate(order, start=1):
        thr = float(alpha) / (m - j + 1)
        holm_ok = (not stopped) and (1.0 - float(r["p_pos_fwd"]) <= thr + 1e-12)
        stopped = stopped or not holm_ok
        cv_ok = r.get("diff_cv_indep") is not None and float(r["diff_cv_indep"]) >= 0.0
        fwd_ok = float(r["diff_fwd"]) >= 0.0
        r.update({"holm_rank": j, "holm_threshold": thr, "holm_pass": bool(holm_ok), "cv_indep_ok": bool(cv_ok),
                  "pass": bool(holm_ok and cv_ok and fwd_ok)})
        if r["pass"]:
            r["reason"] = "Holm pass + forward diff >= 0 + independent CV folds confirm"
        elif not holm_ok:
            r["reason"] = f"Holm: 1-p_pos {1.0 - float(r['p_pos_fwd']):.3f} > {thr:.3f}" + (" (step-down stopped)" if 1.0 - float(r["p_pos_fwd"]) <= thr else "")
        elif not fwd_ok:
            r["reason"] = "forward diff < 0"
        else:
            r["reason"] = "independent CV folds do not confirm (diff_cv_indep < 0 or unavailable)"
    for r in rows:
        if r.get("is_default"):
            r.update({"holm_rank": None, "holm_threshold": None, "holm_pass": None, "cv_indep_ok": None, "pass": None,
                      "reason": "default"})
    return rows


def choose_universe(default: Universe, rows: list[dict]) -> tuple[Universe, dict[str, str]]:
    """Default with each dimension's winner (passing row with the largest ``diff_fwd``); returns (chosen, {dim: key})."""
    winners: dict[str, dict] = {}
    for r in rows:
        if not r.get("pass"):
            continue
        dim = r["dimension"]
        if dim not in winners or float(r["diff_fwd"]) > float(winners[dim]["diff_fwd"]):
            winners[dim] = r
    chosen = default
    for dim, r in winners.items():
        chosen = dataclasses.replace(chosen, **{dim: r[dim]})
    return chosen, {dim: r["key"] for dim, r in winners.items()}


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def decision_signature(member: str, alpha: float, cands: list[Universe]) -> dict:
    """What a ``knobs.json`` decision is keyed on: the knob member, the Holm level and the ordered candidate keys."""
    return {"member": str(member), "alpha_total": float(alpha), "keys": [u.key() for u in cands]}


def check_previous(path: Path, sig: dict) -> dict | None:
    """The decided ``knobs.json`` when it matches ``sig``; None when absent/unfinished; ValueError on a re-decision."""
    path = Path(path)
    if not path.exists():
        return None
    prev = json.loads(path.read_text())
    if prev.get("status") != "done":
        return None
    prev_sig = {"member": prev.get("member"), "alpha_total": float(prev.get("alpha_total", -1.0)),
                "keys": [c["key"] for c in prev.get("candidates", [])]}
    if prev_sig != sig:
        raise ValueError(f"{REDECIDE_MSG} (decided {prev_sig}, config now {sig})")
    return prev


def knob_member(cfg: dict) -> str:
    """``knobs.member`` or the in-run baseline member (``lgbm_everest`` / first everest LGBM / first member)."""
    kn = cfg.get("knobs") or {}
    if kn.get("member"):
        return str(kn["member"])
    names = [m["name"] for m in cfg["members"]]
    if "lgbm_everest" in names:
        return "lgbm_everest"
    for m in cfg["members"]:
        if m["group"] == "everest" and m["family"] == "lgbm":
            return m["name"]
    return names[0]


def _compare(rd: Any, cfg: dict, cands: list[Universe], default: Universe, results: dict[str, Any]) -> tuple[list[dict], list[str]]:
    """Per-candidate rows (fit, paired forward diff, independent CV diff) vs the default; Holm applied by the caller."""
    from stealth import train as trainmod  # per_era_fit_fast = metrics.per_era_fit semantics on vectorised kernels

    bs = cfg["bootstrap"]
    y, era = rd.df[data.PRIMARY_TARGET], rd.era

    def per_era(pred: pd.Series) -> pd.Series:
        return trainmod.per_era_fit_fast(pred, y.loc[pred.index], era.loc[pred.index])

    fw = rd.forward
    indep = [f for f in rd.folds if cvmod._gap_to_block(f.test_eras, fw.test_eras) >= 20]
    indep_eras = np.concatenate([f.test_eras for f in indep]) if indep else np.array([], dtype=np.int32)
    base = results[default.key()]
    base_cv, base_fwd = per_era(base.oof), per_era(base.forward)
    rows = []
    for u in cands:
        r = results[u.key()]
        f_cv, f_fwd = per_era(r.oof), per_era(r.forward)
        pr = cvmod.paired_block_bootstrap((f_fwd - base_fwd).dropna(), block=int(bs["block"]), n_boot=int(bs["n_boot"]),
                                          seed=int(bs["seed"]))
        d_cv = (f_cv - base_cv).dropna()
        d_ind = d_cv.loc[d_cv.index.isin(indep_eras)]
        is_def = u.key() == default.key()
        rows.append({"key": u.key(), **u.to_dict(), "dimension": dimension_of(u, default), "is_default": is_def,
                     "equivalent_to_default": (not is_def) and r is base,
                     "fit_cv": float(f_cv.mean()), "fit_fwd": float(f_fwd.mean()), "diff_fwd": float(pr.diff_mean),
                     "diff_se_fwd": float(pr.diff_se), "p_pos_fwd": float(pr.p_pos),
                     "diff_cv_indep": float(d_ind.mean()) if len(d_ind) else None, "n_eras_cv_indep": int(len(d_ind))})
    return rows, [f.name for f in indep]


def select_universe(rd: Any, cfg: dict, run_dir: Path, *, prog: Any = None) -> tuple[Universe, dict]:
    """Decide the training universe once (docs/INTERFACES.md section 12); writes / reuses ``run_dir/knobs.json``.

    ``rd`` is a ``train.RunData`` (``universe``, ``folds``, ``forward``, ``df``, ``era``, ``era_n_rows``). Without a
    ``knobs`` block the default universe is returned and nothing is written. ``rd.universe`` is restored on exit.
    Candidates whose training rows and weights equal the default's (e.g. ``filter80`` on eras that all have >= 80
    rows) reuse the default cell (diff 0, p_pos 0: they stay in Holm's ``m`` -- pre-registered, never dropped).
    """
    from stealth import train as trainmod

    kn = cfg.get("knobs") or {}
    default = default_universe(cfg)
    if not kn:
        return default, {"status": "none", "default": default.key(), "chosen": {**default.to_dict(), "key": default.key()}}
    if rd.forward is None:
        raise ValueError("knobs need cv.forward.enabled (the decision is made on the forward block)")
    cands = candidates(cfg)
    member = knob_member(cfg)
    alpha = float(kn.get("alpha_total", 1.0 - float(cfg["gates"]["p_pos_adopt"])))
    path = Path(run_dir) / "knobs.json"
    prev = check_previous(path, decision_signature(member, alpha, cands))
    if prev is not None:
        chosen = Universe.from_dict(prev["chosen"])
        log.info("knobs: reusing the decision in %s -> %s", path, chosen.key())
        return chosen, prev
    mspec = next(m for m in cfg["members"] if m["name"] == member)
    params = dict(mspec.get("params") or {})
    seed, threads = int(cfg["run"]["seed"]), int(cfg["run"]["threads"])
    era_all, nrows_all = rd.era.to_numpy(), np.asarray(rd.era_n_rows)
    results: dict[str, Any] = {}
    saved = rd.universe
    try:
        for i, u in enumerate(cands):
            key = u.key()
            if i > 0 and equivalent(u, default, era_all, nrows_all):
                log.info("knobs: %s selects the same training rows as the default -> reusing the default cell", key)
                results[key] = results[default.key()]
                continue
            d = Path(run_dir) / "knobs" / key
            rd.universe = u
            res = trainmod._load_protocol(d)
            if res is None:
                if prog is not None:
                    prog.set(f"knobs {i + 1}/{len(cands)} {key}")
                res = trainmod.run_protocol(rd, mspec, params, seed=seed, threads=threads, prog=prog, label=f"knobs[{key}]")
                trainmod._save_protocol(d, res)
                trainmod.write_json(d / "cell.json", {"member": member, "universe": u.to_dict(), "key": key,
                                                      "params_hash": trainmod.cell_hash(params, u)})
            results[key] = res
    finally:
        rd.universe = saved
    rows, indep = _compare(rd, cfg, cands, default, results)
    holm_decide(rows, alpha)
    chosen, winners = choose_universe(default, rows)
    out = {"status": "done", "member": member, "alpha_total": alpha, "default": default.key(), "indep_cv_folds": indep,
           "candidates": rows, "winners": winners, "chosen": {**chosen.to_dict(), "key": chosen.key()},
           "combination_retested": False, "decided_at": _now()}
    trainmod.write_json(path, out)
    for r in rows:
        log.info("knobs %-26s fit_cv=%.4f fit_fwd=%.4f diff_fwd=%+.4f (se %.4f, p_pos %.2f) cv_indep=%s -> %s", r["key"],
                 r["fit_cv"], r["fit_fwd"], r["diff_fwd"], r["diff_se_fwd"], r["p_pos_fwd"],
                 None if r["diff_cv_indep"] is None else round(r["diff_cv_indep"], 4), r["reason"])
    log.info("knobs: chosen universe %s (winners %s; the combination is reported, not re-tested)", chosen.key(), winners or "none")
    return chosen, out


__all__ = ["DIMENSIONS", "REDECIDE_MSG", "Universe", "candidates", "check_previous", "choose_universe", "decision_signature",
           "default_universe", "dimension_of", "equivalent", "holm_decide", "knob_member", "parse_rows_rule", "pool_min_era",
           "select_universe"]
