"""CLI: config -> OOF parquet + member bytes + metrics.json (``--resume``); writes ``artifacts/<run>/run.log`` itself.

Contracts: docs/INTERFACES.md sections 7 (config), 8 (artifacts), 10 (CLI). Protocol (docs/PLAN.md):

* chronological exped-grouped folds with purge + embargo (``cv.purged_folds``) and the optional strict
  forward block (``cv.forward_block``);
* tree members with ``n_estimators: null`` pick their count on the INNER split of every training fold
  (CV folds AND the forward block). Fold ``k`` is then refit on its full training eras with a frozen count =
  the MEDIAN of the inner best-iterations of the folds whose inner-validation eras lie >= 20 eras away from
  fold ``k``'s test block (``cv.independent_inner_folds``; with chronological folds every non-last fold's
  inner_val sits at the end of the era range, i.e. inside the last fold's and the forward block's test eras,
  so those two use only independent sources -- in practice their own inner split). The scored fold never
  influences its own tree count. The full-train refit uses the median over all CV folds;
* searched members (``search.<member>``) evaluate each grid cell with the same protocol; the member's OOF
  score is the equal-weight mean of the ``top_k`` cells, and the full-train refits of those cells are
  written as ``full_s<k>.bin`` so the bundle averages them. The member's BASE ``params`` cell is always
  evaluated too and ``search/<member>.gain.json`` records top-k minus base on OOF and on the forward block
  (winner's-curse check: an OOF gain above ``gates.suspect_gain_over_baseline`` flags the member suspect);
* the suspect baseline is ``gates.baseline_fit`` (a fixed M1 number) when set, else the in-run ``lgbm_everest``;
* per member: ``oof_<member>.parquet`` (resume unit), ``members/<member>/{spec.json,fold<k>.bin,forward.bin,
  full_s<seed>.bin}``; at the end ``oof.parquet`` / ``forward.parquet`` (``m__<member>`` raw scores),
  ``features.json`` (feature_cols, missing_cols, core_proxy_cols, surrogate), ``folds.json``, ``metrics.json``.

M2-M6 additions (docs/INTERFACES.md sections 11-15):

* two universes: ``run.min_era`` = SCORING eras (test blocks, unchanged); the TRAINING pool reaches down to
  ``min(knobs.era_start)`` and joins train/inner_train via ``cv.extend_train``. ``stealth.knobs.select_universe`` decides
  the training universe ONCE on the forward block (``knobs.json``) before any member trains; ``_fit_rows`` / sample
  weights / full refits apply it, cache keys (``cell_hash``, sharpe cache, ``spec.json["universe"]``) include it;
* the search adoption gate: the top-k average is adopted iff paired CV ``p_pos >= gates.p_pos_adopt`` AND the forward gain
  is ``>= 0``; otherwise the member is its base cell (``search/<m>.adopted.json``; ``best_of`` resolves through it);
* ``folds_eras.json`` for ``scripts/mlp_train.py``; ``family: mlp`` members are EXTERNAL (loaded when finished, else
  pending with a WARNING; exit code unchanged); CatBoost ``dropped_timing`` on a slow first fit; ``rho_c`` on validation.

Hyperparameter VALUES stay under ``artifacts/`` (never in handoff results).
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from stealth import cv as cvmod
from stealth import data, features, metrics, models, surrogate
from stealth import knobs as knobsmod
from stealth.cv import Fold

log = logging.getLogger("stealth.train")  # explicit: under `python -m` __name__ is __main__

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts"
_FAMILIES = ("lgbm", "xgb", "cat", "ridge", "mlp")  # mlp = EXTERNAL member (scripts/mlp_train.py), never fitted here
EXTERNAL_FAMILIES = ("mlp",)
_GROUPS = ("everest", "aux")
_INPUTS = ("nan", "linear")
_REQUIRED_TOP = ("name", "run", "cv", "members", "search", "blend", "bootstrap", "gates", "timing")
_RUN_KEYS = ("seed", "threads", "min_era", "max_era", "min_rows", "time_decay", "data_dir")
_CV_KEYS = ("n_folds", "purge", "embargo", "inner_frac", "inner_gap", "forward")
_MEMBER_KEYS = ("name", "family", "target", "group", "features", "input", "params", "n_estimators", "es_cap")
_MEMBER_OPTIONAL = ("arm", "seeds")
_KNOB_KEYS = ("member", "era_start", "rows_rule", "rows_low_weight", "time_decay", "alpha_total")
_NEUTRAL_KEYS = ("select", "central_mean", "pessimistic_mean", "fit_cost_max", "sharpe_tol", "worst_tol")
DEFAULT_MAX_SEARCH_TRIALS = 25
P_B_CAP, P_C_CAP = 0.75, 0.5
EXIT_OK, EXIT_ERROR, EXIT_GATE = 0, 1, 2
MIN_GAP = 20  # floor on every train/test and early-stopping gap (PLAN: purge + embargo >= 20 eras)
# Part of every run fingerprint, cell id and study name: bump whenever folds, count sources, cell semantics or the
# artifacts a finished member must carry change, so nothing cached under the old protocol can be resumed.
PROTOCOL_VERSION = "m2m6-fix1-audit"
FINGERPRINT_DATA = ("train", "train_benchmark")  # files whose bytes training depends on (validation: side metrics only)
SELECTION_NOTE = ("development figures (CV OOF, forward block, knobs, search adoption, admission ladder, NNLS, neutralisation "
                  "grid) are SELECTION ESTIMATES: the same eras choose and score, so they are optimistic. The independent "
                  "check is audit.json on the reserved eras, which never feeds back into any choice.")


# --------------------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    """Load + validate a run config (docs/INTERFACES.md section 7). Raises ``ValueError`` listing every problem."""
    path = Path(path)
    with path.open() as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: config must be a mapping")
    problems: list[str] = []
    for k in _REQUIRED_TOP:
        if k not in cfg:
            problems.append(f"missing top-level key {k!r}")
    run = cfg.get("run") or {}
    for k in _RUN_KEYS:
        if k not in run:
            problems.append(f"run.{k} missing")
    cvc = cfg.get("cv") or {}
    for k in _CV_KEYS:
        if k not in cvc:
            problems.append(f"cv.{k} missing")
    if isinstance(cvc.get("forward"), dict):
        for k in ("enabled", "train_end", "test_start"):
            if k not in cvc["forward"]:
                problems.append(f"cv.forward.{k} missing")
    elif "forward" in cvc:
        problems.append("cv.forward must be a mapping")
    if cvc.get("n_folds", 2) < 2:
        problems.append("cv.n_folds must be >= 2")
    _validate_audit(cfg, problems)
    if all(k in cvc for k in ("purge", "embargo")) and int(cvc["purge"]) + int(cvc["embargo"]) < MIN_GAP:
        problems.append(f"cv.purge + cv.embargo must be >= {MIN_GAP} eras (target autocorrelation horizon)")
    if "inner_gap" in cvc and int(cvc["inner_gap"]) < MIN_GAP:
        problems.append(f"cv.inner_gap must be >= {MIN_GAP} eras")
    members = cfg.get("members") or []
    if not isinstance(members, list) or not members:
        problems.append("members must be a non-empty list")
    names: list[str] = []
    for i, m in enumerate(members if isinstance(members, list) else []):
        if not isinstance(m, dict):
            problems.append(f"members[{i}] must be a mapping")
            continue
        for k in _MEMBER_KEYS:
            if k not in m:
                problems.append(f"members[{i}].{k} missing")
        names.append(str(m.get("name")))
        if m.get("family") not in _FAMILIES:
            problems.append(f"members[{i}].family {m.get('family')!r} not in {_FAMILIES}")
        if m.get("group") not in _GROUPS:
            problems.append(f"members[{i}].group {m.get('group')!r} not in {_GROUPS}")
        if m.get("input") not in _INPUTS:
            problems.append(f"members[{i}].input {m.get('input')!r} not in {_INPUTS}")
        if not str(m.get("target", "")).startswith("target_"):
            problems.append(f"members[{i}].target must be a target_* column")
        if m.get("family") == "ridge" and m.get("input") != "linear":
            problems.append(f"members[{i}]: ridge needs input: linear")
        if m.get("family") != "ridge" and m.get("n_estimators") is None and m.get("es_cap") is None:
            problems.append(f"members[{i}]: n_estimators null needs es_cap")
        params = m.get("params")
        if params is not None and not isinstance(params, dict):
            problems.append(f"members[{i}].params must be a mapping")
        if isinstance(params, dict) and "best_of" in params:
            all_names = [str(x.get("name")) for x in members if isinstance(x, dict)]
            if params["best_of"] not in all_names or params["best_of"] == m.get("name"):
                problems.append(f"members[{i}].params.best_of must name another member")
    if len(set(names)) != len(names):
        problems.append("member names must be unique")
    search = cfg.get("search")
    if search is None:
        cfg["search"] = {}
    elif not isinstance(search, dict):
        problems.append("search must be a mapping (may be {})")
    else:
        for k, s in search.items():
            if k not in names:
                problems.append(f"search.{k} does not name a member")
            if not isinstance(s, dict) or not isinstance(s.get("grid"), dict) or "n_trials" not in s or "top_k" not in s:
                problems.append(f"search.{k} needs grid/n_trials/top_k")
    blend = cfg.get("blend") or {}
    if not 0.0 <= float(blend.get("p_b", 0.0)) <= 0.75:
        problems.append("blend.p_b must be in [0, 0.75]")
    if not 0.0 <= float(blend.get("p_c", 0.0)) <= 0.5:
        problems.append("blend.p_c must be in [0, 0.5]")
    for k in ("everest_weight", "nnls", "simulation"):
        if k not in blend:
            problems.append(f"blend.{k} missing")
    for k in ("block", "n_boot", "seed"):
        if k not in (cfg.get("bootstrap") or {}):
            problems.append(f"bootstrap.{k} missing")
    for k in ("leak_spearman", "suspect_gain_over_baseline", "p_pos_adopt"):
        if k not in (cfg.get("gates") or {}):
            problems.append(f"gates.{k} missing")
    bf = (cfg.get("gates") or {}).get("baseline_fit")
    if bf is not None and not isinstance(bf, (int, float)):
        problems.append("gates.baseline_fit must be a number (the fixed M1 OOF fit20) or null")
    for k in ("catboost_max_fit_seconds", "progress_every_seconds"):
        if k not in (cfg.get("timing") or {}):
            problems.append(f"timing.{k} missing")
    _validate_m2(cfg, problems)
    if problems:
        raise ValueError(f"{path}: invalid config:\n  - " + "\n  - ".join(problems))
    cfg.setdefault("refit", {"seeds": [0]})
    cfg.setdefault("knobs", {})
    cfg["gates"].setdefault("max_search_trials", DEFAULT_MAX_SEARCH_TRIALS)
    return cfg


def audit_config(cfg: dict) -> dict | None:
    """``cv.audit`` when enabled (``{"enabled": True, "train_end": int, "test_start": int}``), else None."""
    a = (cfg.get("cv") or {}).get("audit")
    return a if isinstance(a, dict) and a.get("enabled") else None


def _validate_audit(cfg: dict, problems: list[str]) -> None:
    """``cv.audit`` (reserved partition, docs/INTERFACES.md section 22): development uses eras < train_end only."""
    cvc = cfg.get("cv") or {}
    if "audit" not in cvc or cvc["audit"] is None:
        return
    a = cvc["audit"]
    if not isinstance(a, dict):
        problems.append("cv.audit must be a mapping {enabled, train_end, test_start}")
        return
    for k in sorted(set(a) - {"enabled", "train_end", "test_start"}):
        problems.append(f"cv.audit.{k}: unknown key")
    if not isinstance(a.get("enabled"), bool):
        problems.append("cv.audit.enabled must be true/false")
    if not a.get("enabled"):
        return
    te, ts = a.get("train_end"), a.get("test_start")
    if not (_num(te) and _num(ts)):
        problems.append("cv.audit needs integer train_end and test_start")
        return
    if int(ts) - int(te) < MIN_GAP:
        problems.append(f"cv.audit: test_start - train_end must be >= {MIN_GAP} (the target horizon)")
    run = cfg.get("run") or {}
    if run.get("min_era") is not None and int(run["min_era"]) >= int(te):
        problems.append("cv.audit.train_end must lie above run.min_era (development needs scoring eras)")
    if run.get("max_era") is not None and int(run["max_era"]) < int(ts):
        problems.append("cv.audit.test_start lies above run.max_era: the audit partition would be empty")
    fw = cvc.get("forward") or {}
    if isinstance(fw, dict) and fw.get("enabled"):
        if _num(fw.get("test_start")) and int(fw["test_start"]) >= int(te):
            problems.append("cv.forward.test_start must lie below cv.audit.train_end (the forward block is development data)")
    for v in ((cfg.get("knobs") or {}).get("era_start") or []):
        if _num(v) and int(v) >= int(te):
            problems.append(f"knobs.era_start {v} must lie below cv.audit.train_end")


def arm_of(m: dict) -> str:
    """Member arm (INTERFACES section 11): ``arm`` key, else ``"aux"`` for the aux group, else the member name."""
    return str(m.get("arm") or ("aux" if m.get("group") == "aux" else m.get("name")))


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _validate_members_m2(cfg: dict, problems: list[str]) -> None:
    members = [m for m in (cfg.get("members") or []) if isinstance(m, dict)]
    by_name = {str(m.get("name")): m for m in members}
    for i, m in enumerate(members):
        extra = sorted(set(m) - set(_MEMBER_KEYS) - set(_MEMBER_OPTIONAL))
        if extra:
            problems.append(f"members[{i}]: unknown keys {extra} (optional keys: {list(_MEMBER_OPTIONAL)})")
        if "arm" in m and (not isinstance(m["arm"], str) or not m["arm"]):
            problems.append(f"members[{i}].arm must be a non-empty string")
        if "seeds" in m and (not isinstance(m["seeds"], list) or not m["seeds"] or not all(isinstance(s, int) for s in m["seeds"])):
            problems.append(f"members[{i}].seeds must be a non-empty list of ints")
        if m.get("family") == "mlp":
            if "seeds" not in m:
                problems.append(f"members[{i}]: family mlp (external member) needs seeds")
            if m.get("input") != "linear":
                problems.append(f"members[{i}]: family mlp needs input: linear")
            if m.get("n_estimators") is not None:
                problems.append(f"members[{i}]: family mlp takes its epochs from inner early stopping (n_estimators: null)")
            if "best_of" in (m.get("params") or {}):
                problems.append(f"members[{i}]: family mlp cannot use params.best_of")
        dep = (m.get("params") or {}).get("best_of") if isinstance(m.get("params"), dict) else None
        if dep in by_name and by_name[dep].get("family") in EXTERNAL_FAMILIES:
            problems.append(f"members[{i}].params.best_of names an external (mlp) member")
    for k, s in (cfg.get("search") or {}).items():
        if k in by_name and by_name[k].get("family") in EXTERNAL_FAMILIES:
            problems.append(f"search.{k}: external (mlp) members are not searched by train.py")
        if isinstance(s, dict) and "adopt_gate" in s and not isinstance(s["adopt_gate"], bool):
            problems.append(f"search.{k}.adopt_gate must be true/false")


def _validate_knobs(cfg: dict, problems: list[str]) -> None:
    kn = cfg.get("knobs")
    if kn is None or kn == {}:
        return
    if not isinstance(kn, dict):
        problems.append("knobs must be a mapping")
        return
    run = cfg.get("run") or {}
    if "min_rows" in kn:
        problems.append("knobs.min_rows is not supported: use knobs.rows_rule (none | filter<N> | weight<N>)")
    for k in sorted(set(kn) - set(_KNOB_KEYS) - {"min_rows"}):
        problems.append(f"knobs.{k}: unknown key (allowed: {list(_KNOB_KEYS)})")
    if run.get("min_rows") is not None:
        problems.append("run.min_rows must be null when knobs are set (it filters the SCORING eras; use knobs.rows_rule)")
    if run.get("min_era") is None:
        problems.append("knobs need a numeric run.min_era (the scoring universe)")
    if not ((cfg.get("cv") or {}).get("forward") or {}).get("enabled"):
        problems.append("knobs need cv.forward.enabled: true (the decision is made on the forward block)")
    es = kn.get("era_start", [run.get("min_era")])
    if not isinstance(es, list) or not es or not all(isinstance(v, int) and not isinstance(v, bool) for v in es):
        problems.append("knobs.era_start must be a non-empty list of ints")
    elif run.get("min_era") is not None and int(run["min_era"]) not in es:
        problems.append("knobs.era_start must contain run.min_era (the default)")
    rr = kn.get("rows_rule", ["none"])
    if not isinstance(rr, list) or not rr:
        problems.append("knobs.rows_rule must be a non-empty list")
    else:
        for v in rr:
            if not isinstance(v, str) or not knobsmod.ROWS_RULE_RE.match(v):
                problems.append(f"knobs.rows_rule entry {v!r} must be none | filter<N> | weight<N>")
        if "none" not in rr:
            problems.append("knobs.rows_rule must contain 'none' (the default)")
    if "rows_low_weight" in kn and (not _num(kn["rows_low_weight"]) or not 0.0 < float(kn["rows_low_weight"]) <= 1.0):
        problems.append("knobs.rows_low_weight must be in (0, 1]")
    td = kn.get("time_decay", [bool(run.get("time_decay"))])
    if not isinstance(td, list) or not td or not all(isinstance(v, bool) for v in td):
        problems.append("knobs.time_decay must be a non-empty list of booleans")
    elif bool(run.get("time_decay")) not in td:
        problems.append("knobs.time_decay must contain run.time_decay (the default)")
    if "alpha_total" in kn and (not _num(kn["alpha_total"]) or not 0.0 < float(kn["alpha_total"]) < 1.0):
        problems.append("knobs.alpha_total must be in (0, 1)")
    members = {str(m.get("name")): m for m in (cfg.get("members") or []) if isinstance(m, dict)}
    km = kn.get("member")
    if km is not None:
        if km not in members:
            problems.append(f"knobs.member {km!r} does not name a member")
        elif members[km].get("family") in EXTERNAL_FAMILIES or "best_of" in (members[km].get("params") or {}):
            problems.append("knobs.member must be a member trained here with explicit params (no best_of, not mlp)")


def _validate_blend_gates(cfg: dict, problems: list[str]) -> None:
    blend = cfg.get("blend") or {}
    sim = blend.get("simulation") or {}
    for v in sim.get("p_b_grid") or []:
        if not _num(v) or not 0.0 <= float(v) <= P_B_CAP:
            problems.append(f"blend.simulation.p_b_grid value {v!r} outside [0, {P_B_CAP}]")
    if "p_c_grid" in sim:
        if not isinstance(sim["p_c_grid"], list) or not sim["p_c_grid"]:
            problems.append("blend.simulation.p_c_grid must be a non-empty list")
        else:
            for v in sim["p_c_grid"]:
                if not _num(v) or not 0.0 <= float(v) <= P_C_CAP:
                    problems.append(f"blend.simulation.p_c_grid value {v!r} outside [0, {P_C_CAP}]")
    neu = blend.get("neutralisation")
    if neu is not None:
        if not isinstance(neu, dict):
            problems.append("blend.neutralisation must be a mapping")
        else:
            for k in sorted(set(neu) - set(_NEUTRAL_KEYS)):
                problems.append(f"blend.neutralisation.{k}: unknown key (allowed: {list(_NEUTRAL_KEYS)})")
            if not isinstance(neu.get("select", False), bool):
                problems.append("blend.neutralisation.select must be true/false")
            means = [float(x) for x in sim.get("bench_fit_means") or [] if _num(x)]
            for k in ("central_mean", "pessimistic_mean"):
                if k in neu and (not _num(neu[k]) or not any(abs(float(neu[k]) - m) < 1e-12 for m in means)):
                    problems.append(f"blend.neutralisation.{k} must be one of blend.simulation.bench_fit_means {means}")
            for k in ("fit_cost_max", "sharpe_tol", "worst_tol"):
                if k in neu and (not _num(neu[k]) or float(neu[k]) < 0.0):
                    problems.append(f"blend.neutralisation.{k} must be a number >= 0")
            if neu.get("select"):
                for k in ("central_mean", "pessimistic_mean"):
                    if k not in neu:
                        problems.append(f"blend.neutralisation.select needs {k}")
                if "p_c_grid" not in sim:
                    problems.append("blend.neutralisation.select needs blend.simulation.p_c_grid")
                for g in ("p_b_grid", "p_c_grid"):  # the (p_b=0, p_c=0) reference cell of select_neutralisation
                    if g in sim and isinstance(sim[g], list) and not any(_num(v) and float(v) == 0.0 for v in sim[g]):
                        problems.append(f"blend.neutralisation.select needs 0.0 in blend.simulation.{g} (the reference cell)")
    adm = blend.get("admission")
    if adm is not None:
        if not isinstance(adm, dict):
            problems.append("blend.admission must be a mapping")
        else:
            for k in ("base", "order"):
                if not isinstance(adm.get(k), list) or not all(isinstance(a, str) for a in adm.get(k) or []):
                    problems.append(f"blend.admission.{k} must be a list of arm names")
            if "sharpe_gain" in adm and (not _num(adm["sharpe_gain"]) or float(adm["sharpe_gain"]) < 0.0):
                problems.append("blend.admission.sharpe_gain must be a number >= 0")
            arms = {arm_of(m) for m in (cfg.get("members") or []) if isinstance(m, dict)}
            for a in list(adm.get("base") or []) + list(adm.get("order") or []):
                if isinstance(a, str) and a not in arms:
                    log.warning("blend.admission names arm %r that no member of this config has: it is skipped", a)
    gates = cfg.get("gates") or {}
    if "member_corr_max" in gates and (not _num(gates["member_corr_max"]) or not 0.0 < float(gates["member_corr_max"]) <= 1.0):
        problems.append("gates.member_corr_max must be in (0, 1]")
    mst = gates.get("max_search_trials", DEFAULT_MAX_SEARCH_TRIALS)
    if not isinstance(mst, int) or isinstance(mst, bool) or mst < 1:
        problems.append("gates.max_search_trials must be an int >= 1")
    else:
        for k, s in (cfg.get("search") or {}).items():
            if isinstance(s, dict) and isinstance(s.get("n_trials"), int) and s["n_trials"] > mst:
                problems.append(f"search.{k}.n_trials {s['n_trials']} > gates.max_search_trials {mst} (PLAN C6: <= 25 trials)")


def _validate_m2(cfg: dict, problems: list[str]) -> None:
    """INTERFACES section 11 additions (knobs, arms, external mlp members, admission, neutralisation, trial cap)."""
    _validate_members_m2(cfg, problems)
    _validate_knobs(cfg, problems)
    _validate_blend_gates(cfg, problems)


# --------------------------------------------------------------------------------------------------
# logging / host / progress
# --------------------------------------------------------------------------------------------------
def setup_logging(run_dir: Path) -> logging.Handler:
    """Attach a ``run.log`` FileHandler (+ stderr stream) to the ``stealth`` logger; returns the file handler."""
    run_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("stealth")
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    return fh


def git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def host_info() -> dict:
    """``{"os","cpu","ram_gb","python","libs"}`` for metrics.json."""
    ram = None
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    ram = round(int(line.split()[1]) / 1024**2, 1)
                    break
    except OSError:
        pass
    libs = {}
    for name in ("lightgbm", "xgboost", "catboost", "scikit-learn", "numpy", "pandas", "scipy", "optuna", "everesteer-api"):
        try:
            from importlib.metadata import version

            libs[name] = version(name)
        except Exception:  # noqa: BLE001
            libs[name] = None
    return {"os": f"{platform.system()} {platform.release()}", "cpu": f"{platform.processor() or platform.machine()} x{os.cpu_count()}",
            "ram_gb": ram, "python": platform.python_version(), "libs": libs}


class Progress:
    """Background thread that logs a progress line every ``every`` seconds (handoffs rely on it)."""

    def __init__(self, every: float) -> None:
        self.every = max(float(every), 1.0)
        self.status = "starting"
        self.t0 = time.perf_counter()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="progress", daemon=True)

    def set(self, status: str) -> None:
        self.status = status

    def _loop(self) -> None:
        while not self._stop.wait(self.every):
            log.info("progress: %s (elapsed %.0fs)", self.status, time.perf_counter() - self.t0)

    def __enter__(self) -> "Progress":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
    tmp.replace(path)


# --------------------------------------------------------------------------------------------------
# run fingerprint (statistics review finding 2): config + data + protocol, checked BEFORE any mutation
# --------------------------------------------------------------------------------------------------
class FingerprintMismatch(ValueError):
    """The run-id already holds artifacts made with another config, data set or protocol (or none recorded)."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_for_fingerprint(cfg: dict) -> dict:
    """The config the fingerprint covers: everything except ``run.threads`` (the one edit a handoff permits) and
    ``timing.progress_every_seconds`` (log cadence). Neither changes a single prediction."""
    c = copy.deepcopy(cfg)
    (c.get("run") or {}).pop("threads", None)
    (c.get("timing") or {}).pop("progress_every_seconds", None)
    return c


def run_fingerprint(cfg: dict, data_dir: Path) -> dict:
    """``run_fingerprint.json`` content (docs/INTERFACES.md section 22.A)."""
    files = {"train": data.split_path("train", data_dir), "train_benchmark": data.benchmark_path("train", data_dir)}
    data_sha = {k: (sha256_file(p) if p.exists() else None) for k, p in files.items() if k in FINGERPRINT_DATA}
    config_sha = hashlib.sha256(json.dumps(config_for_fingerprint(cfg), sort_keys=True, default=str).encode()).hexdigest()
    body = {"config": config_sha, "data": data_sha, "protocol": PROTOCOL_VERSION}
    fp = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return {"schema": "stealth.fingerprint/1", "fingerprint": fp, "config_sha": config_sha, "data": data_sha,
            "protocol": PROTOCOL_VERSION, "created_at": _now()}


_LEGACY_MARKERS = ("metrics.json", "oof.parquet", "knobs.json", "folds.json", "folds_eras.json", "members")


def check_fingerprint(run_dir: Path, fp: dict) -> bool:
    """True when ``run_dir`` already carries this fingerprint (resume is safe), False for a fresh run dir.

    Raises :class:`FingerprintMismatch` -- before anything is written -- when the run dir was made with a different
    config / data / protocol, or holds artifacts from before fingerprints were recorded."""
    path = Path(run_dir) / "run_fingerprint.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("fingerprint") == fp["fingerprint"]:
            return True
        what = [k.replace("_sha", "") for k in ("config_sha", "protocol") if old.get(k) != fp.get(k)]
        if old.get("data") != fp.get("data"):
            what.append("data")
        raise FingerprintMismatch(f"run-id {Path(run_dir).name} was created with a different {'/'.join(what) or 'fingerprint'}; "
                                  "start a new run-id (nothing in the old run dir was changed)")
    legacy = [n for n in _LEGACY_MARKERS if (Path(run_dir) / n).exists()]
    if legacy:
        raise FingerprintMismatch(f"run-id {Path(run_dir).name} holds artifacts without a run fingerprint ({', '.join(legacy)}); "
                                  "start a new run-id (nothing in the old run dir was changed)")
    return False


# --------------------------------------------------------------------------------------------------
# data context
# --------------------------------------------------------------------------------------------------
class RunData:
    """Loaded train split (int8 features kept as-is), run-era subset, folds and per-run feature metadata.

    ``df`` holds the eras in play (after ``run.min_era/max_era/min_rows``) with int8 features, float32
    targets and the int32 ``era`` column. Input matrices are built per fold from the int8 frame
    (``matrix``) so no float32 copy of the whole split is ever kept.
    """

    def __init__(self, cfg: dict, data_dir: Path, *, fingerprint: str = "") -> None:
        run = cfg["run"]
        self.cfg = cfg
        self.fingerprint = str(fingerprint)  # run_fingerprint.json["fingerprint"]: part of every cell id / member spec
        full = data.load_split("train", data_dir)
        self.feature_cols = data.feature_columns(full)
        self.target_cols = data.target_columns(full)
        # leak-free side products from eras < 2800 (never scored): missing columns, core proxy, surrogate
        bench = data.load_benchmark("train", data_dir)
        early = full.loc[full["era"] < 2800]
        if len(early) < 1000:  # tiny/filtered data sets: fall back to the first third of the eras, never past min_era
            cut = int(np.quantile(full["era"].unique(), 0.34))
            if run.get("min_era") is not None:
                cut = min(cut, int(run["min_era"]))
            early = full.loc[full["era"] < cut]
            if len(early) == 0:
                cut = int(np.quantile(full["era"].unique(), 0.34))
                early = full.loc[full["era"] < cut]
            log.warning("fewer than 1000 rows before era 2800: side products (missing cols, core proxy, surrogate) fitted on "
                        "eras < %d%s -- the leak-free guarantee of the side products is weakened", cut,
                        "" if run.get("min_era") is not None and cut <= int(run["min_era"]) else " (these eras are also scored)")
        self.missing_cols = data.top_missing_columns(early, self.feature_cols, k=15)
        X_nan_early = data.features_nan(early, self.feature_cols)
        self.core_proxy_cols = features.core_proxy_columns(X_nan_early, bench.reindex(early.index), early["era"])
        del X_nan_early
        X_lin_early = data.features_linear(early, self.feature_cols, self.missing_cols)
        self.surrogate = surrogate.fit_surrogate(X_lin_early, bench.reindex(early.index), early["era"],
                                                 max_era=int(early["era"].max()) + 1)
        del X_lin_early, early, bench
        # two universes (INTERFACES section 12): SCORING eras >= run.min_era (CV + forward test blocks, unchanged) and the
        # TRAINING pool >= min(run.min_era, knobs.era_start); a Universe masks/weights training rows inside the pool only
        self.has_knobs = bool(cfg.get("knobs"))
        if self.has_knobs:
            pool_min = knobsmod.pool_min_era(cfg)
            self.df = data.filter_eras(full, min_era=pool_min, max_era=run.get("max_era"))
        else:  # M1 configs: exactly the old frame (run.min_rows still filters the scoring eras there)
            self.df = data.filter_eras(full, min_era=run.get("min_era"), max_era=run.get("max_era"), min_rows=run.get("min_rows"))
        del full
        self.era = self.df["era"]
        counts = data.era_counts(self.df)
        self.era_n_rows = counts.reindex(self.era.to_numpy()).to_numpy(dtype=np.int64)  # n_rows of each row's era
        self.n_rows_by_era = counts
        self.pool_eras = np.unique(self.era.to_numpy()).astype(np.int32)
        min_era = run.get("min_era")
        self.eras = self.pool_eras[self.pool_eras >= int(min_era)] if min_era is not None else self.pool_eras
        # reserved audit partition (docs/INTERFACES.md section 22.C): every development choice -- CV, forward, knobs,
        # search, admission, NNLS, count sources -- AND the deployment full refits see only eras < train_end; the audit
        # eras (>= test_start) are predicted by the finished full-refit blobs and scored once, after the blend is frozen
        self.audit_cfg = audit_config(cfg)
        if self.audit_cfg:
            te, ts = int(self.audit_cfg["train_end"]), int(self.audit_cfg["test_start"])
            self.audit_eras = self.pool_eras[self.pool_eras >= ts]
            if self.audit_eras.size == 0:
                raise ValueError(f"cv.audit: the data has no eras >= test_start {ts}")
            self.eras = self.eras[self.eras < te]
            self.full_train_eras = self.pool_eras[self.pool_eras < te]
        else:
            self.audit_eras = np.array([], dtype=np.int32)
            self.full_train_eras = self.pool_eras
        self.scoring_min_era = int(self.eras.min())
        extra = self.pool_eras[self.pool_eras < self.scoring_min_era]
        c = cfg["cv"]
        gap = int(c["purge"]) + int(c["embargo"])
        self.folds: list[Fold] = cvmod.purged_folds(self.eras, int(c["n_folds"]), purge=int(c["purge"]), embargo=int(c["embargo"]),
                                                   inner_frac=float(c["inner_frac"]), inner_gap=int(c["inner_gap"]))
        self.forward: Fold | None = None
        fw = c.get("forward") or {}
        if fw.get("enabled"):
            self.forward = cvmod.forward_block(self.eras, train_end=int(fw["train_end"]), test_start=int(fw["test_start"]),
                                               inner_frac=float(c["inner_frac"]), inner_gap=int(c["inner_gap"]))
        if extra.size:  # pool eras below the scoring universe join train + inner_train (purge gap > purge + embargo)
            self.folds = [cvmod.extend_train(f, extra, gap=gap) for f in self.folds]
            if self.forward is not None:
                self.forward = cvmod.extend_train(self.forward, extra, gap=gap)
        self.min_gap = max(MIN_GAP, min(gap, int(c["inner_gap"])))
        cvmod.assert_fold_gaps(self.folds + ([self.forward] if self.forward else []), min_gap=self.min_gap)
        if self.audit_cfg:
            self.assert_development_only()
        self.universe: knobsmod.Universe = knobsmod.default_universe(cfg)
        self.sharpe_cols_cache: dict[tuple[str, str, str], list[str]] = {}
        bags = cfg.get("feature_bags") or {}
        self.bags = features.random_column_bags(self.feature_cols, n_bags=int(bags.get("n_bags", 4)),
                                                fraction=float(bags.get("fraction", 0.5)), seed=int(bags.get("seed", 0)))
        log.info("run data: %d rows, %d scoring eras (%d..%d), training pool %d eras from %d, %d folds%s%s", len(self.df), len(self.eras),
                 self.eras.min(), self.eras.max(), len(self.pool_eras), self.pool_eras.min(), len(self.folds),
                 ", forward block" if self.forward else "",
                 f", RESERVED audit {len(self.audit_eras)} eras ({int(self.audit_eras.min())}..{int(self.audit_eras.max())}; "
                 f"development + full refits use eras < {int(self.audit_cfg['train_end'])})" if self.audit_cfg else "")

    def assert_development_only(self) -> None:
        """Raise ``AssertionError`` if any fold / forward / full-refit era reaches the reserved audit partition."""
        te = int(self.audit_cfg["train_end"])
        for f in self.folds + ([self.forward] if self.forward is not None else []):
            for part in ("train_eras", "test_eras", "inner_train_eras", "inner_val_eras"):
                arr = getattr(f, part)
                if arr.size and int(arr.max()) >= te:
                    raise AssertionError(f"{f.name}.{part} reaches era {int(arr.max())} >= audit train_end {te}")
        if self.full_train_eras.size and int(self.full_train_eras.max()) >= te:
            raise AssertionError("full-refit eras reach the audit partition")
        if np.intersect1d(self.full_train_eras, self.audit_eras).size:
            raise AssertionError("full-refit eras overlap the audit eras")

    # -- rows -------------------------------------------------------------------------------------------
    def rows(self, eras: np.ndarray) -> np.ndarray:
        """Boolean row mask for the given eras."""
        return np.isin(self.era.to_numpy(), eras)

    def matrix(self, mask: np.ndarray, cols: list[str], input_kind: str) -> pd.DataFrame:
        """Float32 input frame for the rows in ``mask`` (``nan`` -> features_nan; ``linear`` -> features_linear)."""
        sub = self.df.loc[mask]
        if input_kind == "linear":
            return data.features_linear(sub, cols, self.missing_cols)
        return data.features_nan(sub, cols)

    def target(self, name: str, mask: np.ndarray) -> np.ndarray:
        return self.df.loc[mask, name].to_numpy(dtype=np.float32)

    def train_mask(self) -> np.ndarray:
        """Per-row training mask of the current universe (TRAINING rows only; never applied to test/prediction rows)."""
        return self.universe.train_mask(self.era.to_numpy(), self.era_n_rows)

    def weights(self, mask: np.ndarray) -> np.ndarray | None:
        """Sample weights of the fit rows in ``mask`` under the current universe (None = unweighted, the M1 default)."""
        return self.universe.weights(self.era.to_numpy()[mask], self.era_n_rows[mask])

    def resolve_features(self, spec: str | list[str], fold: Fold | None, target: str) -> list[str]:
        """Concrete feature list; ``sharpe_subset`` is computed on the fold's universe-masked TRAINING rows only (cached
        per fold, target and universe key)."""
        sharpe_cols = None
        if spec == "sharpe_subset":
            key = (fold.name if fold else "full", target, self.universe.key())
            if key not in self.sharpe_cols_cache:
                thr = float((self.cfg.get("sharpe_subset") or {}).get("threshold", 0.1))
                mask = (self.rows(fold.train_eras) if fold else self.rows(self.full_train_eras)) & self.train_mask()
                X = self.matrix(mask, self.feature_cols, "nan")
                y = self.df.loc[mask, target]
                ok = y.notna().to_numpy()
                cols = features.sharpe_subset(X.loc[ok], y.loc[ok], self.era.loc[mask].loc[ok], threshold=thr)
                self.sharpe_cols_cache[key] = cols or list(self.feature_cols)
                log.info("sharpe_subset[%s]: %d features", key, len(self.sharpe_cols_cache[key]))
            sharpe_cols = self.sharpe_cols_cache[key]
        return features.resolve_feature_set(spec, self.feature_cols, sharpe_cols=sharpe_cols, bags=self.bags)

    def folds_summary(self) -> list[dict]:
        return [f.summary() for f in self.folds] + ([self.forward.summary()] if self.forward else [])


def _fit_rows(rd: RunData, eras: np.ndarray, target: str) -> np.ndarray:
    """Training mask: rows in ``eras`` whose ``target`` is not NaN, restricted by the current universe's train mask."""
    mask = rd.rows(eras)
    mask &= rd.df[target].notna().to_numpy()
    mask &= rd.train_mask()
    if not mask.any():
        raise ValueError(f"universe {rd.universe.key()} leaves no training rows for {target} in eras {int(np.min(eras))}..{int(np.max(eras))}")
    return mask


# --------------------------------------------------------------------------------------------------
# the fold protocol for one (member spec, params) pair
# --------------------------------------------------------------------------------------------------
class ProtocolResult:
    """OOF predictions + fold blobs of one config run through the frozen-count protocol."""

    def __init__(self) -> None:
        self.oof: pd.Series | None = None          # raw scores on every CV test row (index id)
        self.oof_fold: pd.Series | None = None     # int8 fold number per OOF row
        self.forward: pd.Series | None = None      # raw scores on the forward test rows
        self.fold_blobs: list[bytes] = []
        self.forward_blob: bytes | None = None
        self.inner_iters: list[int] = []               # inner best-iteration per CV fold
        self.inner_iter_forward: int | None = None     # inner best-iteration of the forward block's own split
        self.frozen_n_iter: int | None = None          # median over CV folds: used for the FULL-train refit only
        self.frozen_n_iter_per_fold: list[int] = []    # count actually used to fit each CV fold (independent sources)
        self.frozen_n_iter_forward: int | None = None  # count used for the forward block
        self.count_sources: dict[str, list[str]] = {}  # fold name -> names of the folds whose inner ES set its count
        self.fit_seconds: float = 0.0
        self.max_fit_seconds: float = 0.0
        self.feature_cols: dict[str, list[str]] = {}


def _es_needed(spec: dict) -> bool:
    return spec["family"] != "ridge" and spec.get("n_estimators") is None


def _fit(rd: RunData, spec: dict, params: dict, cols: list[str], train_eras: np.ndarray, n_iter: int | None,
         *, seed: int, threads: int, es: tuple[np.ndarray, np.ndarray] | None = None) -> models.FittedMember:
    """Fit on the rows of ``train_eras`` (NaN-target rows dropped). ``es`` = (inner_train_eras, inner_val_eras)
    switches to early stopping with ``spec['es_cap']``; otherwise a fixed ``n_iter`` fit."""
    target = spec["target"]
    s = {**spec, "params": params, "feature_cols": cols}
    if es is not None:
        s["n_estimators"] = None
        m_tr, m_val = _fit_rows(rd, es[0], target), _fit_rows(rd, es[1], target)
        return models.fit_member(s, rd.matrix(m_tr, cols, spec["input"]), rd.target(target, m_tr), rd.weights(m_tr),
                                 X_val=rd.matrix(m_val, cols, spec["input"]), y_val=rd.target(target, m_val), seed=seed, threads=threads)
    s["n_estimators"] = n_iter if spec["family"] != "ridge" else None
    mask = _fit_rows(rd, train_eras, target)
    return models.fit_member(s, rd.matrix(mask, cols, spec["input"]), rd.target(target, mask), rd.weights(mask), seed=seed, threads=threads)


def _predict_rows(rd: RunData, fm: models.FittedMember, cols: list[str], eras: np.ndarray, input_kind: str) -> pd.Series:
    mask = rd.rows(eras)
    X = rd.matrix(mask, cols, input_kind)
    return pd.Series(models.predict_member(fm, X), index=X.index, dtype=np.float64)


class TimingDrop(Exception):
    """A member's FIRST fit exceeded its wall-time cap (``timing.catboost_max_fit_seconds`` for CatBoost)."""

    def __init__(self, seconds: float, cap: float) -> None:
        super().__init__(f"first fit {seconds:.1f}s > cap {cap:.0f}s")
        self.seconds, self.cap = float(seconds), float(cap)


def run_protocol(rd: RunData, spec: dict, params: dict, *, seed: int, threads: int, prog: Progress | None = None,
                 label: str = "", max_first_fit_seconds: float | None = None) -> ProtocolResult:
    """Inner early stopping per fold -> frozen median count -> fixed-count fold fits -> OOF (+ forward) predictions.

    ``max_first_fit_seconds`` (CatBoost cost guard, INTERFACES section 15): the very first fit of this run is timed and
    :class:`TimingDrop` is raised immediately when it took longer.
    """
    res = ProtocolResult()
    t0 = time.perf_counter()
    first = [True]

    def _guard(seconds: float) -> None:
        if first[0]:
            first[0] = False
            if max_first_fit_seconds is not None and seconds > float(max_first_fit_seconds):
                raise TimingDrop(seconds, float(max_first_fit_seconds))
    input_kind = spec["input"]
    cols_by_fold = {f.name: rd.resolve_features(spec["features"], f, spec["target"]) for f in rd.folds}
    if rd.forward is not None:
        cols_by_fold["forward"] = rd.resolve_features(spec["features"], rd.forward, spec["target"])
    res.feature_cols = cols_by_fold
    all_folds = rd.folds + ([rd.forward] if rd.forward is not None else [])
    counts: dict[str, int | None] = {f.name: None for f in all_folds}
    if _es_needed(spec):
        es_iters: dict[str, int] = {}
        for f in all_folds:
            if prog:
                prog.set(f"{label}: inner early stopping {f.name}")
            t1 = time.perf_counter()
            fm = _fit(rd, spec, params, cols_by_fold[f.name], f.train_eras, None, seed=seed, threads=threads,
                      es=(f.inner_train_eras, f.inner_val_eras))
            res.max_fit_seconds = max(res.max_fit_seconds, time.perf_counter() - t1)
            _guard(time.perf_counter() - t1)
            es_iters[f.name] = int(fm.n_iter)
            log.info("%s %s: inner best_iter=%d (%.1fs)", label, f.name, fm.n_iter, fm.fit_seconds)
        res.inner_iters = [es_iters[f.name] for f in rd.folds]
        res.inner_iter_forward = es_iters.get("forward")
        res.frozen_n_iter = int(round(float(np.median(res.inner_iters))))  # full-train refit only (nothing scored)
        counts, res.count_sources = frozen_counts(rd, es_iters)
        res.frozen_n_iter_per_fold = [int(counts[f.name]) for f in rd.folds]
        res.frozen_n_iter_forward = counts.get("forward")
        log.info("%s: frozen n_iter per fold=%s forward=%s full-refit=%d (inner iters %s; sources %s)", label,
                 res.frozen_n_iter_per_fold, res.frozen_n_iter_forward, res.frozen_n_iter, es_iters, res.count_sources)
    elif spec["family"] != "ridge":
        res.frozen_n_iter = int(spec["n_estimators"])
        counts = {f.name: res.frozen_n_iter for f in all_folds}
        res.frozen_n_iter_per_fold = [res.frozen_n_iter for _ in rd.folds]
        res.frozen_n_iter_forward = res.frozen_n_iter if rd.forward is not None else None
    parts, fold_ids = [], []
    for k, f in enumerate(all_folds):
        if prog:
            prog.set(f"{label}: fitting {f.name} with n_iter={counts[f.name]}")
        t1 = time.perf_counter()
        fm = _fit(rd, spec, params, cols_by_fold[f.name], f.train_eras, counts[f.name], seed=seed, threads=threads)
        res.max_fit_seconds = max(res.max_fit_seconds, time.perf_counter() - t1)
        _guard(time.perf_counter() - t1)
        pred = _predict_rows(rd, fm, cols_by_fold[f.name], f.test_eras, input_kind)
        if f.name == "forward":
            res.forward, res.forward_blob = pred, fm.blob
        else:
            parts.append(pred)
            fold_ids.append(pd.Series(np.int8(k), index=pred.index))
            res.fold_blobs.append(fm.blob)
        log.info("%s %s: fit %.1fs, %d test rows", label, f.name, fm.fit_seconds, len(pred))
        del fm
    res.oof = pd.concat(parts)
    res.oof_fold = pd.concat(fold_ids).astype(np.int8)
    res.fit_seconds = time.perf_counter() - t0
    return res


def frozen_counts(rd: RunData, es_iters: dict[str, int]) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Per-fold frozen tree count from INDEPENDENT early-stopping sources (PLAN C8, leakage review item 1).

    CV fold ``k`` uses the median of the inner best-iterations of the CV folds whose inner_val eras are >= ``rd.min_gap``
    from fold ``k``'s test block (its own split always qualifies). The forward block uses the median over its own inner
    split plus the CV folds that are independent of its test block. ``cv.assert_count_sources`` guards every choice.
    Returns ``({fold_name: count}, {fold_name: [source fold names]})``.
    """
    counts: dict[str, int] = {}
    sources: dict[str, list[str]] = {}
    for f in rd.folds:
        idx = cvmod.independent_inner_folds(f, rd.folds, min_gap=rd.min_gap)
        src = [rd.folds[j] for j in idx]
        cvmod.assert_count_sources(f, src, min_gap=rd.min_gap)
        counts[f.name] = int(round(float(np.median([es_iters[x.name] for x in src]))))
        sources[f.name] = [x.name for x in src]
    if rd.forward is not None:
        fw = rd.forward
        idx = cvmod.independent_inner_folds(fw, rd.folds, min_gap=rd.min_gap)
        src = [rd.folds[j] for j in idx] + [fw]
        cvmod.assert_count_sources(fw, src, min_gap=rd.min_gap)
        counts[fw.name] = int(round(float(np.median([es_iters[x.name] for x in src]))))
        sources[fw.name] = [x.name for x in src]
    return counts, sources


def refit_full(rd: RunData, spec: dict, params: dict, n_iter: int | None, *, seeds: list[int], threads: int,
               prog: Progress | None = None, label: str = "") -> tuple[list[bytes], list[str]]:
    """Refit on the full-refit eras (``rd.full_train_eras``: the whole training pool, or only the eras before the reserved
    audit partition when ``cv.audit`` is on) under the current universe with the frozen count, once per seed; returns
    (blobs, feature_cols). Without knobs or audit the pool is exactly the run eras (M1 behaviour)."""
    cols = rd.resolve_features(spec["features"], None, spec["target"])
    blobs = []
    for s in seeds:
        if prog:
            prog.set(f"{label}: full refit seed={s}")
        fm = _fit(rd, spec, params, cols, rd.full_train_eras, n_iter, seed=int(s), threads=threads)
        log.info("%s: full refit seed=%d n_iter=%s (%.1fs, %d bytes)", label, s, fm.n_iter, fm.fit_seconds, len(fm.blob))
        blobs.append(fm.blob)
    return blobs, cols


def validation_side_metrics(spec: dict, blobs: list[bytes], cols: list[str], missing_cols: list[str], data_dir: Path,
                            core_proxy_cols: list[str] | None = None) -> dict:
    """``{"corr_with_bench_val": rho_b, "rho_c": rho_c}`` on the VALIDATION split (INTERFACES section 15).

    ``rho_b`` = mean per-era corr(prediction, v1_sherpa) (exact, no targets needed); ``rho_c`` = mean per-era
    corr(prediction, core proxy) with the proxy = row mean of ``core_proxy_cols`` ignoring -1. Either is None when the
    files / columns are missing.
    """
    out: dict = {"corr_with_bench_val": None, "rho_c": None}
    core = [c for c in (core_proxy_cols or [])]
    try:
        val = data.load_split("validation", data_dir, columns=list(dict.fromkeys(list(cols) + list(missing_cols) + core)))
    except FileNotFoundError:
        return out
    X = data.features_linear(val, cols, missing_cols) if spec["input"] == "linear" else data.features_nan(val, cols)
    if spec["family"] in EXTERNAL_FAMILIES:  # mlp: the base columns of the linear frame, selected by name
        X = X[list(cols)]
    pred = pd.Series(np.mean([models.predict_member((spec["family"], b), X) for b in blobs], axis=0), index=X.index)
    try:
        bench = data.load_benchmark("validation", data_dir)
        out["corr_with_bench_val"] = float(per_era_spearman_fast(pred, bench.reindex(X.index), val["era"]).mean())
    except FileNotFoundError:
        pass
    if core:
        cp = features.core_proxy_series(val, core)
        out["rho_c"] = float(per_era_spearman_fast(pred, cp.reindex(X.index), val["era"]).mean())
    return out


def validation_bench_corr(spec: dict, blobs: list[bytes], cols: list[str], missing_cols: list[str], data_dir: Path) -> float | None:
    """Back-compat wrapper: rho_b only (see :func:`validation_side_metrics`)."""
    return validation_side_metrics(spec, blobs, cols, missing_cols, data_dir)["corr_with_bench_val"]


def _params_hash(params: dict) -> str:
    import hashlib

    return hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:12]


def cell_hash(params: dict, universe: knobsmod.Universe, fingerprint: str = "") -> str:
    """Search/knob cell identity: params, the training universe AND the run fingerprint (config + data + protocol), so a
    resumed run never reuses a cell made under another universe, config, data set or fold protocol."""
    key = {**params, "__universe__": universe.key()}
    if fingerprint:
        key["__run__"] = str(fingerprint)
    return _params_hash(key)


def _predict_full(rd: RunData, spec: dict, blobs: list[bytes], cols: list[str], eras: np.ndarray) -> pd.Series:
    """Equal-weight mean over ``blobs`` (the member's full refits, exactly the bytes the bundle averages) on the rows of
    ``eras``. Used for the reserved audit partition: no label is read here."""
    mask = rd.rows(eras)
    X = rd.matrix(mask, cols, spec["input"])
    pred = np.mean(np.vstack([models.predict_member((spec["family"], b), X) for b in blobs]), axis=0)
    return pd.Series(pred.astype(np.float64), index=X.index, name="pred")


def _write_audit_pred(path: Path, rd: RunData, pred: pd.Series) -> None:
    df = pd.DataFrame({"era": rd.era.loc[pred.index].astype(np.int32), "pred": pred.astype(np.float64)}, index=pred.index)
    df.index.name = "id"
    df.to_parquet(path)


def _save_protocol(d: Path, res: ProtocolResult, cell_id: str | None = None) -> None:
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"pred": res.oof.astype(np.float32), "fold": res.oof_fold})
    if res.forward is not None:
        df = pd.concat([df, pd.DataFrame({"pred": res.forward.astype(np.float32), "fold": np.int8(-1)})])
    df.index.name = "id"
    df.to_parquet(d / "oof.parquet")
    for k, b in enumerate(res.fold_blobs):
        (d / f"fold{k}.bin").write_bytes(b)
    if res.forward_blob is not None:
        (d / "forward.bin").write_bytes(res.forward_blob)
    write_json(d / "result.json", {"inner_iters": res.inner_iters, "frozen_n_iter": res.frozen_n_iter, "fit_seconds": res.fit_seconds,
                                   "max_fit_seconds": res.max_fit_seconds, "feature_cols": res.feature_cols,
                                   "inner_iter_forward": res.inner_iter_forward, "frozen_n_iter_per_fold": res.frozen_n_iter_per_fold,
                                   "frozen_n_iter_forward": res.frozen_n_iter_forward, "count_sources": res.count_sources,
                                   "cell_id": cell_id, "protocol": PROTOCOL_VERSION})


def _load_protocol(d: Path, cell_id: str | None = None) -> ProtocolResult | None:
    """A cached cell, or None when absent, incomplete, from an older protocol or (``cell_id`` given) another cell."""
    if not (d / "oof.parquet").exists() or not (d / "result.json").exists():
        return None
    r = json.loads((d / "result.json").read_text())
    if r.get("protocol") != PROTOCOL_VERSION or (cell_id is not None and r.get("cell_id") != cell_id):
        log.info("cell cache %s: protocol/cell id %s/%s != %s/%s -> recomputed", d, r.get("protocol"), r.get("cell_id"),
                 PROTOCOL_VERSION, cell_id)
        return None
    res = ProtocolResult()
    df = pd.read_parquet(d / "oof.parquet")
    cvm = df["fold"] >= 0
    res.oof, res.oof_fold = df.loc[cvm, "pred"].astype(np.float64), df.loc[cvm, "fold"].astype(np.int8)
    if (~cvm).any():
        res.forward = df.loc[~cvm, "pred"].astype(np.float64)
    res.inner_iters, res.frozen_n_iter = r["inner_iters"], r["frozen_n_iter"]
    res.fit_seconds, res.max_fit_seconds, res.feature_cols = r["fit_seconds"], r["max_fit_seconds"], r["feature_cols"]
    if "frozen_n_iter_per_fold" not in r:  # cached cell from before the per-fold protocol: do not reuse it
        return None
    res.inner_iter_forward, res.frozen_n_iter_per_fold = r.get("inner_iter_forward"), list(r["frozen_n_iter_per_fold"])
    res.frozen_n_iter_forward, res.count_sources = r.get("frozen_n_iter_forward"), dict(r.get("count_sources") or {})
    res.fold_blobs = [p.read_bytes() for p in sorted(d.glob("fold*.bin"), key=lambda p: int(p.stem[4:]))]
    res.forward_blob = (d / "forward.bin").read_bytes() if (d / "forward.bin").exists() else None
    return res


# --------------------------------------------------------------------------------------------------
# member driver (search, resume, artifacts)
# --------------------------------------------------------------------------------------------------
def _resolve_params(mspec: dict, run_dir: Path) -> dict:
    """Member params; ``params.best_of: <m>`` resolves to ``search/<m>.adopted.json["best"]`` (the search top-1 when the
    search was adopted, else <m>'s base params -- INTERFACES section 13). Legacy run dirs without an adoption record
    fall back to the search top-1."""
    params = dict(mspec.get("params") or {})
    if "best_of" in params:
        src_name = params.pop("best_of")
        adopted = run_dir / "search" / f"{src_name}.adopted.json"
        src = run_dir / "search" / f"{src_name}.json"
        if adopted.exists():
            rec = json.loads(adopted.read_text())
            best, how = rec["best"], "adopted search top-1" if rec.get("adopted") else "base params (search not adopted)"
        elif src.exists():
            best, how = json.loads(src.read_text())[0]["params"], "search top-1 (no adoption record)"
        else:
            raise FileNotFoundError(f"{mspec['name']}: params.best_of needs {adopted} (run the searched member first)")
        params = {**best, **params}
        log.info("%s: params from %s of %s", mspec["name"], how, src_name)
    return params


def member_order(cfg: dict) -> list[dict]:
    """Config order, except that a member is deferred until the member its ``params.best_of`` names has run."""
    by_name = {m["name"]: m for m in cfg["members"]}
    order: list[dict] = []
    seen: set[str] = set()

    def visit(m: dict, stack: tuple[str, ...]) -> None:
        if m["name"] in seen:
            return
        dep = (m.get("params") or {}).get("best_of")
        if dep:
            if dep in stack:
                raise ValueError(f"best_of cycle: {' -> '.join(stack + (dep,))}")
            visit(by_name[dep], stack + (m["name"],))
        seen.add(m["name"])
        order.append(m)

    for m in cfg["members"]:
        visit(m, ())
    return order


def _reusable_spec(rd: RunData, name: str, spec_path: Path, oof_path: Path, resume: bool) -> dict | None:
    """A finished (or timing-dropped) member's spec.json when ``resume``, its universe AND run fingerprint match and every
    artifact it promises exists; else None (retrain). A spec left ``running`` by an interrupted (re)train is never
    reused."""
    if not resume or not spec_path.exists():
        return None
    spec = json.loads(spec_path.read_text())
    if spec.get("universe") != rd.universe.key():
        log.info("member %s: spec.json universe %s != run universe %s -> retraining", name, spec.get("universe"), rd.universe.key())
        return None
    if spec.get("run_fingerprint") != rd.fingerprint:
        log.info("member %s: spec.json was written under another run fingerprint -> retraining", name)
        return None
    if spec.get("status") == "dropped_timing":
        log.info("member %s: resumed as dropped_timing (first fit %.0fs > cap) -- the drop is kept", name,
                 float(spec.get("first_fit_seconds") or 0.0))
        return spec
    if spec.get("status") != "done" or not oof_path.exists():
        return None
    mdir = spec_path.parent
    missing = [f["file"] for f in spec.get("full_blobs") or [] if not (mdir / f["file"]).exists()]
    if not spec.get("full_blobs") or missing:
        log.info("member %s: declared full refits missing (%s) -> retraining", name, missing or "none declared")
        return None
    if rd.audit_eras.size and not (spec_path.parents[2] / f"audit_{name}.parquet").exists():
        log.info("member %s: audit predictions missing -> retraining", name)
        return None
    log.info("member %s: resumed from %s", name, spec_path.parent)
    return spec


def per_era_fit_fast(pred: pd.Series, target: pd.Series, era: pd.Series) -> pd.Series:
    """``metrics.per_era_fit`` semantics (SDK ``fit20`` per era: NaN pairs dropped, < 3 rows or constant predictions ->
    0.0) computed with the vectorised kernels ``metrics.pad_by_era`` / ``rank_gauss_rows`` / ``fit_rows`` (parity-tested
    against the SDK to 1e-12 in tests/test_metrics.py and tests/test_train_m2.py); ~50x faster than the per-era SDK loop,
    which dominated search objectives, knob comparisons and member metrics."""
    import warnings

    p = pred.reindex(era.index).to_numpy(dtype=np.float64)
    t = target.reindex(era.index).to_numpy(dtype=np.float64)
    bad = np.isnan(p) | np.isnan(t)
    p, t = np.where(bad, np.nan, p), np.where(bad, np.nan, t)
    P, eras = metrics.pad_by_era(p, era)
    T, _ = metrics.pad_by_era(t, era)
    n = np.sum(~np.isnan(P), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN eras -> 0.0 below
        Z = metrics.rank_gauss_rows(P)
        tc = T - np.nanmean(T, axis=1, keepdims=True)
        f = metrics.fit_rows(Z, tc)
        sd = np.nanstd(P, axis=1, ddof=1)
    f = np.where((n < 3) | ~(sd > 0.0) | np.isnan(f), 0.0, f)
    return pd.Series(f.astype(np.float64), index=pd.Index(eras.astype(np.int32), name="era"), name="fit")


def per_era_spearman_fast(pred: pd.Series, target: pd.Series, era: pd.Series) -> pd.Series:
    """``metrics.per_era_spearman`` semantics (NaN pairs dropped; < 3 rows or a constant side -> 0.0), vectorised:
    Pearson correlation of the per-era average ranks on a padded ``[n_eras, max_rows]`` array."""
    from scipy import stats

    p = pred.reindex(era.index).to_numpy(dtype=np.float64)
    t = target.reindex(era.index).to_numpy(dtype=np.float64)
    bad = np.isnan(p) | np.isnan(t)
    p, t = np.where(bad, np.nan, p), np.where(bad, np.nan, t)
    P, eras = metrics.pad_by_era(p, era)
    T, _ = metrics.pad_by_era(t, era)
    nan = np.isnan(P)
    n = np.sum(~nan, axis=1)
    rp = np.where(nan, np.nan, stats.rankdata(P, axis=1, method="average", nan_policy="omit"))
    rt = np.where(nan, np.nan, stats.rankdata(T, axis=1, method="average", nan_policy="omit"))
    with np.errstate(invalid="ignore", divide="ignore"):
        dp = np.where(nan, 0.0, rp - np.nanmean(np.where(nan, np.nan, rp), axis=1, keepdims=True))
        dt = np.where(nan, 0.0, rt - np.nanmean(np.where(nan, np.nan, rt), axis=1, keepdims=True))
        den = np.sqrt((dp * dp).sum(axis=1) * (dt * dt).sum(axis=1))
        rho = np.where(den > 0, (dp * dt).sum(axis=1) / np.where(den > 0, den, 1.0), 0.0)
    rho = np.where((n < 3) | np.isnan(rho), 0.0, rho)
    return pd.Series(rho.astype(np.float64), index=pd.Index(eras.astype(np.int32), name="era"), name="spearman")


def _mean_fit_series(rd: RunData, pred: pd.Series) -> pd.Series:
    return per_era_fit_fast(pred, rd.df.loc[pred.index, data.PRIMARY_TARGET], rd.era.loc[pred.index])


def _average_results(runs: list[ProtocolResult]) -> ProtocolResult:
    """Equal-weight mean of several cells' OOF/forward predictions; fold blobs and counts of the first cell (top-1)."""
    base = runs[0]
    res = ProtocolResult()
    res.oof = pd.concat([r.oof for r in runs], axis=1).mean(axis=1)
    res.oof_fold = base.oof_fold
    res.forward = pd.concat([r.forward for r in runs], axis=1).mean(axis=1) if base.forward is not None else None
    res.fold_blobs, res.forward_blob = base.fold_blobs, base.forward_blob
    res.inner_iters, res.frozen_n_iter = base.inner_iters, base.frozen_n_iter
    res.inner_iter_forward, res.frozen_n_iter_per_fold = base.inner_iter_forward, base.frozen_n_iter_per_fold
    res.frozen_n_iter_forward, res.count_sources = base.frozen_n_iter_forward, base.count_sources
    res.fit_seconds = sum(r.fit_seconds for r in runs)
    res.max_fit_seconds, res.feature_cols = max(r.max_fit_seconds for r in runs), base.feature_cols
    return res


def adoption_decision(paired_cv: dict, forward_gain: float | None, *, p_pos_adopt: float, adopt_gate: bool = True) -> tuple[bool, str]:
    """INTERFACES section 13: adopt the top-k average iff ``paired_cv.p_pos >= p_pos_adopt`` AND ``forward_gain >= 0``.

    ``adopt_gate: false`` keeps the M1 behaviour (top-k always). Without a forward block only the CV gate applies. A paired
    result without enough bootstrap blocks (``inference_valid`` False) fails closed: the base cell is kept.
    """
    if not adopt_gate:
        return True, "adopt_gate: false (top-k average always used, M1 behaviour)"
    if paired_cv.get("inference_valid") is False:
        return False, f"tie: only {paired_cv.get('n_blocks')} bootstrap block(s) -> no inference -> base cell (simpler)"
    p = float(paired_cv["p_pos"])
    if p < float(p_pos_adopt):
        return False, f"tie: paired CV p_pos {p:.3f} < {p_pos_adopt} -> base cell (simpler)"
    if forward_gain is None:
        return True, f"paired CV p_pos {p:.3f} >= {p_pos_adopt}; no forward block (CV gate only)"
    if float(forward_gain) < 0.0:
        return False, f"tie: forward gain {float(forward_gain):+.4f} < 0 -> base cell (simpler)"
    return True, f"paired CV p_pos {p:.3f} >= {p_pos_adopt} and forward gain {float(forward_gain):+.4f} >= 0"


def _knob_cell(rd: RunData, run_dir: Path, name: str, h: str) -> ProtocolResult | None:
    """The knob decision's cell for this member + universe when its hash matches (saves one protocol run)."""
    d = run_dir / "knobs" / rd.universe.key()
    cj = d / "cell.json"
    if not cj.exists():
        return None
    meta = json.loads(cj.read_text())
    if meta.get("member") != name or meta.get("params_hash") != h:
        return None
    res = _load_protocol(d, h)
    if res is not None:
        log.info("%s: base cell reused from the knob decision (%s)", name, d.name)
    return res


def search_member(rd: RunData, mspec: dict, params: dict, cfg: dict, run_dir: Path, *, prog: Progress | None,
                  cap: float | None) -> tuple[ProtocolResult, list[dict], list[int | None], dict]:
    """Search + winner's-curse record + adoption gate (INTERFACES sections 6, 8, 13).

    Returns ``(result, params_list, frozen_n_iter per params, gain record)``. The base cell is ALWAYS evaluated; the
    top-k average is adopted only if the paired CV gate and the forward gain agree, otherwise the member IS the base
    cell. Writes ``search/<m>.json``, ``search/<m>.gain.json`` and ``search/<m>.adopted.json``.
    """
    from stealth import search as searchmod

    name = mspec["name"]
    search_cfg = cfg["search"][name]
    seed, threads = int(cfg["run"]["seed"]), int(cfg["run"]["threads"])
    bs, gates = cfg["bootstrap"], cfg["gates"]
    sdir = run_dir / "search" / name
    cache: dict[str, ProtocolResult] = {}

    def h_of(p: dict) -> str:
        return cell_hash(p, rd.universe, rd.fingerprint)

    def cell(p: dict, label: str) -> ProtocolResult:
        h = h_of(p)
        if h in cache:
            return cache[h]
        res = _load_protocol(sdir / h, h) or _knob_cell(rd, run_dir, name, h)  # trial cache on disk = the resume unit
        if res is None:
            res = run_protocol(rd, mspec, p, seed=seed, threads=threads, prog=prog, label=label, max_first_fit_seconds=cap)
        if _load_protocol(sdir / h, h) is None:
            _save_protocol(sdir / h, res, h)
        cache[h] = res
        return res

    def objective(c: dict) -> float:
        # a grid cell overrides the member's base params; everything not in the grid (bagging, max_bin, ...) stays
        p = {**params, **c}
        return float(_mean_fit_series(rd, cell(p, f"{name}[{h_of(p)}]").oof).mean())

    n_trials = min(int(search_cfg["n_trials"]), int(gates.get("max_search_trials", DEFAULT_MAX_SEARCH_TRIALS)))
    # the Optuna study is namespaced by run fingerprint + universe: a changed protocol never resumes an old study
    study = f"{name}@{rd.fingerprint[:16] or 'nofp'}-{rd.universe.key()}"
    results = searchmod.run_search(name, search_cfg["grid"], objective, n_trials=n_trials,
                                   storage=run_dir / "optuna.db", seed=seed, study_name=study)
    # search/<name>.json carries the FULL params (base + cell) so params.best_of dependents inherit everything
    results = [({**params, **c}, v) for c, v in results]
    write_json(run_dir / "search" / f"{name}.json", [{"params": p, "value": v} for p, v in results])
    top = searchmod.top_k_params(results, int(search_cfg["top_k"]))
    runs = [cell(p, name) for p in top]
    topk_res = _average_results(runs)
    base_h = h_of(params)
    base_res = cell(params, f"{name}[base]")
    gain = _search_gain(rd, topk_res, base_res, [h_of(p) for p in top], base_h,
                        float(gates["suspect_gain_over_baseline"]))
    d = (_mean_fit_series(rd, topk_res.oof) - _mean_fit_series(rd, base_res.oof)).dropna()
    paired = cvmod.paired_block_bootstrap(d, block=int(bs["block"]), n_boot=int(bs["n_boot"]), seed=int(bs["seed"])).as_dict()
    adopted, reason = adoption_decision(paired, gain["forward_gain"], p_pos_adopt=float(gates["p_pos_adopt"]),
                                        adopt_gate=bool(search_cfg.get("adopt_gate", True)))
    gain.update({"paired_cv": paired, "adopted": adopted, "adopt_reason": reason, "n_trials": n_trials,
                 "universe": rd.universe.key()})
    write_json(run_dir / "search" / f"{name}.gain.json", gain)
    if adopted:
        res, params_list, frozen = topk_res, top, [r.frozen_n_iter for r in runs]
    else:
        res, params_list, frozen = base_res, [params], [base_res.frozen_n_iter]
    write_json(run_dir / "search" / f"{name}.adopted.json", {"adopted": adopted, "params_list": params_list,
                                                             "best": params_list[0], "reason": reason})
    log.info("%s search: top-%d OOF fit %.4f vs base cell %.4f (gain %+.4f, paired p_pos %.3f%s); forward gain %s -> %s", name,
             len(top), gain["topk_fit"], gain["base_params_fit"], gain["gain"], paired["p_pos"], ", SUSPECT" if gain["suspect"] else "",
             None if gain["forward_gain"] is None else round(gain["forward_gain"], 4), "ADOPTED" if adopted else f"NOT adopted ({reason})")
    return res, params_list, frozen, gain


def _mark_running(rd: RunData, mspec: dict, spec_path: Path, stale: list[Path]) -> None:
    """spec.json ``status: running`` (+ fingerprint) and the member's assembled outputs removed, before any (re)fit."""
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    for p in stale:
        p.unlink(missing_ok=True)
    write_json(spec_path, {"name": mspec["name"], "family": mspec["family"], "status": "running", "universe": rd.universe.key(),
                           "run_fingerprint": rd.fingerprint, "protocol": PROTOCOL_VERSION, "started_at": _now()})


def _drop_for_timing(rd: RunData, mspec: dict, spec_path: Path, oof_path: Path, exc: TimingDrop) -> dict:
    """Write the ``dropped_timing`` spec.json (INTERFACES section 15): no oof file, excluded from assembly and blend."""
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    for stale in [oof_path, oof_path.with_name(f"audit_{mspec['name']}.parquet"), *spec_path.parent.glob("*.bin")]:
        stale.unlink(missing_ok=True)
    spec = {**mspec, "arm": arm_of(mspec), "status": "dropped_timing", "first_fit_seconds": exc.seconds, "max_fit_seconds_cap": exc.cap,
            "universe": rd.universe.key(), "run_fingerprint": rd.fingerprint, "protocol": PROTOCOL_VERSION, "dropped_at": _now()}
    write_json(spec_path, spec)
    log.warning("member %s: CatBoost dropped (%.0fs > %.0fs cap on its first fit)", mspec["name"], exc.seconds, exc.cap)
    return spec


def _data_dir(cfg: dict) -> Path:
    d = Path(cfg["run"]["data_dir"])
    return d if d.is_absolute() else REPO_ROOT / d


def train_member(rd: RunData, mspec: dict, cfg: dict, run_dir: Path, *, resume: bool, prog: Progress) -> dict:
    """Run one member end-to-end under ``rd.universe``; returns its spec.json dict. ``resume`` reuses finished
    artifacts of the SAME universe (and keeps a timing drop)."""
    name = mspec["name"]
    mdir = run_dir / "members" / name
    spec_path, oof_path = mdir / "spec.json", run_dir / f"oof_{name}.parquet"
    audit_path = run_dir / f"audit_{name}.parquet"
    spec = _reusable_spec(rd, name, spec_path, oof_path, resume)
    if spec is not None:
        return spec
    seed, threads = int(cfg["run"]["seed"]), int(cfg["run"]["threads"])
    seeds = [int(s) for s in (cfg.get("refit") or {}).get("seeds", [0])]
    params = _resolve_params(mspec, run_dir)
    cap = float(cfg["timing"]["catboost_max_fit_seconds"]) if mspec["family"] == "cat" else None
    # Mark the member incomplete BEFORE any artifact is replaced (statistics review 2, contract review 3): an interrupted
    # retrain then leaves status "running" (never reused, refused by the bundle) instead of an old "done" spec next to
    # partially replaced blobs. The member's OOF / audit files go first so they cannot be assembled meanwhile.
    _mark_running(rd, mspec, spec_path, [oof_path, audit_path])
    t0 = time.perf_counter()
    try:
        if cfg["search"].get(name):
            res, top_params, frozen_per_cfg, search_gain = search_member(rd, mspec, params, cfg, run_dir, prog=prog, cap=cap)
        else:
            res = run_protocol(rd, mspec, params, seed=seed, threads=threads, prog=prog, label=name, max_first_fit_seconds=cap)
            top_params, frozen_per_cfg, search_gain = [params], [res.frozen_n_iter], None
    except TimingDrop as exc:
        return _drop_for_timing(rd, mspec, spec_path, oof_path, exc)
    # artifacts
    mdir.mkdir(parents=True, exist_ok=True)
    for stale in mdir.glob("*.bin"):  # a non-adopted search may have fewer blobs than an earlier attempt
        stale.unlink()
    for k, b in enumerate(res.fold_blobs):
        (mdir / f"fold{k}.bin").write_bytes(b)
    if res.forward_blob is not None:
        (mdir / "forward.bin").write_bytes(res.forward_blob)
    full_files = []
    full_cols: list[str] = []
    for rank, (p, n_iter) in enumerate(zip(top_params, frozen_per_cfg)):
        blobs, full_cols = refit_full(rd, mspec, p, n_iter, seeds=seeds, threads=threads, prog=prog, label=f"{name} rank{rank}")
        for j, b in enumerate(blobs):
            fname = f"full_s{rank * len(seeds) + j}.bin"
            (mdir / fname).write_bytes(b)
            full_files.append({"file": fname, "rank": rank, "seed": seeds[j], "n_iter": n_iter})
    prog.set(f"{name}: validation benchmark / core-proxy correlation")
    full_bytes = [(mdir / f["file"]).read_bytes() for f in full_files]
    side = validation_side_metrics(mspec, full_bytes, full_cols, rd.missing_cols, _data_dir(cfg), rd.core_proxy_cols)
    if rd.audit_eras.size:  # the reserved eras, predicted by exactly the blobs the bundle averages; labels never read
        prog.set(f"{name}: audit predictions (reserved eras)")
        _write_audit_pred(audit_path, rd, _predict_full(rd, mspec, full_bytes, full_cols, rd.audit_eras))
    df = pd.DataFrame({"era": rd.era.loc[res.oof.index].astype(np.int32), "fold": res.oof_fold.loc[res.oof.index], "pred": res.oof.astype(np.float32)})
    if res.forward is not None:
        df = pd.concat([df, pd.DataFrame({"era": rd.era.loc[res.forward.index].astype(np.int32), "fold": np.int8(-1),
                                          "pred": res.forward.astype(np.float32)})])
    df.index.name = "id"
    df.to_parquet(oof_path)
    spec = {**mspec, "arm": arm_of(mspec), "params": top_params[0], "top_params": top_params, "feature_cols": full_cols,
            "all_feature_cols": rd.feature_cols, "missing_cols": rd.missing_cols, "feature_cols_per_fold": res.feature_cols,
            "inner_iters": res.inner_iters, "inner_iter_forward": res.inner_iter_forward, "frozen_n_iter": res.frozen_n_iter,
            "frozen_n_iter_per_fold": res.frozen_n_iter_per_fold, "frozen_n_iter_forward": res.frozen_n_iter_forward,
            "count_sources": res.count_sources, "frozen_n_iter_per_config": frozen_per_cfg, "fit_seconds": res.fit_seconds,
            "max_fit_seconds": res.max_fit_seconds, "member_seconds": time.perf_counter() - t0, "full_blobs": full_files,
            "corr_with_bench_val": side["corr_with_bench_val"], "rho_c": side["rho_c"], "search_gain": search_gain,
            "search_adopted": None if search_gain is None else bool(search_gain["adopted"]),
            "universe": rd.universe.key(), "universe_dict": rd.universe.to_dict(), "run_fingerprint": rd.fingerprint,
            "protocol": PROTOCOL_VERSION, "audit_file": audit_path.name if rd.audit_eras.size else None,
            "full_train_eras": [int(rd.full_train_eras.min()), int(rd.full_train_eras.max()), int(len(rd.full_train_eras))],
            "finished_at": _now(), "status": "done"}
    write_json(spec_path, spec)  # LAST: "done" only once every blob, OOF and audit file is in place
    log.info("member %s done in %.0fs (frozen n_iter=%s, rho_b(val)=%s, rho_c(val)=%s)", name, spec["member_seconds"], res.frozen_n_iter,
             None if side["corr_with_bench_val"] is None else round(side["corr_with_bench_val"], 4),
             None if side["rho_c"] is None else round(side["rho_c"], 4))
    return spec


def _search_gain(rd: RunData, topk: ProtocolResult, base: ProtocolResult, top_hashes: list[str], base_hash: str,
                 margin: float) -> dict:
    """Top-k-average minus base-params-cell fit20 on OOF (and the forward block); ``suspect`` when the OOF gain > margin.

    Hashes identify grid cells without exposing hyperparameter values (the file lands under artifacts/ anyway).
    """
    def _mean_fit(pred: pd.Series | None) -> float | None:
        return None if pred is None else float(_mean_fit_series(rd, pred).mean())

    topk_fit, base_fit = _mean_fit(topk.oof), _mean_fit(base.oof)
    topk_fwd, base_fwd = _mean_fit(topk.forward), _mean_fit(base.forward)
    gain = float(topk_fit - base_fit)
    fwd_gain = None if topk_fwd is None or base_fwd is None else float(topk_fwd - base_fwd)
    return {"topk_fit": topk_fit, "base_params_fit": base_fit, "gain": gain, "margin": margin, "suspect": bool(gain > margin),
            "topk_forward_fit": topk_fwd, "base_params_forward_fit": base_fwd, "forward_gain": fwd_gain,
            "forward_confirms": None if fwd_gain is None else bool(fwd_gain >= 0.0),
            "topk_cells": top_hashes, "base_cell": base_hash, "base_cell_in_topk": base_hash in top_hashes}


# --------------------------------------------------------------------------------------------------
# external members (family mlp) + folds_eras.json (INTERFACES section 14)
# --------------------------------------------------------------------------------------------------
def count_sources(rd: RunData) -> dict[str, list[str]]:
    """Fold name -> the folds whose inner early stopping may set its frozen count (same rule as :func:`frozen_counts`)."""
    out: dict[str, list[str]] = {}
    for f in rd.folds:
        out[f.name] = [rd.folds[j].name for j in cvmod.independent_inner_folds(f, rd.folds, min_gap=rd.min_gap)]
    if rd.forward is not None:
        fw = rd.forward
        out[fw.name] = [rd.folds[j].name for j in cvmod.independent_inner_folds(fw, rd.folds, min_gap=rd.min_gap)] + [fw.name]
    return out


def folds_eras_doc(rd: RunData) -> dict:
    """``folds_eras.json``: the exact fold era lists (pool-extended; the universe's row mask applies on top, exactly as in
    train.py's ``_fit_rows``), count sources and the full-refit eras, for ``scripts/mlp_train.py``."""
    src = count_sources(rd)

    def fold_doc(f: Fold) -> dict:
        return {"name": f.name, "train_eras": [int(e) for e in f.train_eras], "test_eras": [int(e) for e in f.test_eras],
                "inner_train_eras": [int(e) for e in f.inner_train_eras], "inner_val_eras": [int(e) for e in f.inner_val_eras],
                "count_sources": src[f.name]}

    audit = None
    if rd.audit_eras.size:
        audit = {"train_end": int(rd.audit_cfg["train_end"]), "test_start": int(rd.audit_cfg["test_start"]),
                 "eras": [int(e) for e in rd.audit_eras], "file": "audit_<member>.parquet (index id; era int32; pred float64)",
                 "note": "predict these eras with the exported full-refit blobs only; never early-stop, select or score on them"}
    return {"universe": rd.universe.to_dict(), "universe_key": rd.universe.key(), "scoring_min_era": int(rd.scoring_min_era),
            "run_fingerprint": rd.fingerprint, "protocol": PROTOCOL_VERSION,
            "folds": [fold_doc(f) for f in rd.folds], "forward": fold_doc(rd.forward) if rd.forward is not None else None,
            "full_train_eras": [int(e) for e in rd.full_train_eras], "full_refit_count": "median over CV folds' inner best iterations",
            "audit": audit, "min_gap": int(rd.min_gap)}


def _blob_files(spec: dict) -> list[str]:
    return [f["file"] if isinstance(f, dict) else str(f) for f in spec.get("full_blobs") or []]


def _external_blobs(mdir: Path, spec: dict) -> list[bytes]:
    return [(mdir / f).read_bytes() for f in _blob_files(spec)]


def external_problem(rd: RunData, mspec: dict, spec: dict, run_dir: Path) -> str | None:
    """Why a finished-looking external spec must NOT be loaded (statistics review 2): another run fingerprint, universe,
    seeds or epoch cap than this run's config, undeclared / missing full refits, or missing OOF / audit predictions.
    None when it is this run's member."""
    name = mspec["name"]
    if spec.get("status") != "done":
        return f"spec status {spec.get('status')!r}"
    if spec.get("universe") != rd.universe.key():
        return f"spec universe {spec.get('universe')!r} != run universe {rd.universe.key()!r}"
    if spec.get("run_fingerprint") != rd.fingerprint:
        return "trained for another run fingerprint (config / data / protocol changed): retrain it with scripts/mlp_train.py"
    want_seeds = [int(s) for s in mspec.get("seeds") or []]
    if [int(s) for s in spec.get("seeds") or []] != want_seeds:
        return f"trained with seeds {spec.get('seeds')} but the config says {want_seeds}"
    if mspec.get("es_cap") is not None and spec.get("epochs_cap") is not None and int(spec["epochs_cap"]) != int(mspec["es_cap"]):
        return f"trained with an epoch cap of {spec['epochs_cap']} but the config says {mspec['es_cap']} (--epochs-cap override)"
    files = _blob_files(spec)
    if not files:
        return "spec declares no full_blobs"
    missing = [f for f in files if not (run_dir / "members" / name / f).exists()]
    if missing:
        return f"declared full refit(s) missing: {missing}"
    if not (run_dir / f"oof_{name}.parquet").exists():
        return f"oof_{name}.parquet missing"
    if rd.audit_eras.size and not (run_dir / f"audit_{name}.parquet").exists():
        return f"audit_{name}.parquet missing (the audit partition is on)"
    return None


def load_external_member(rd: RunData, mspec: dict, cfg: dict, run_dir: Path) -> tuple[dict | None, str | None]:
    """``(spec, None)`` for a finished external member of THIS run, else ``(None, reason)`` (it stays pending).

    Finished = ``members/<m>/spec.json`` passing :func:`external_problem` (status done, same universe, same run
    fingerprint, the config's seeds and epoch cap, declared full refits present, OOF + audit predictions present).
    Missing ``corr_with_bench_val`` / ``rho_c`` are computed via ``models.predict_member`` when the family module can
    load the blobs (best effort: a failure only logs a WARNING).
    """
    name = mspec["name"]
    mdir = run_dir / "members" / name
    spec_path = mdir / "spec.json"
    if not spec_path.exists():
        return None, "no members/<m>/spec.json yet"
    spec = json.loads(spec_path.read_text())
    why = external_problem(rd, mspec, spec, run_dir)
    if why is not None:
        return None, why
    for k in ("name", "family", "target", "group", "input", "features"):
        spec.setdefault(k, mspec[k])
    spec.setdefault("arm", arm_of(mspec))
    spec.setdefault("feature_cols", list(rd.feature_cols))
    spec.setdefault("missing_cols", list(rd.missing_cols))
    if spec.get("corr_with_bench_val") is None or spec.get("rho_c") is None:
        try:
            side = validation_side_metrics(spec, _external_blobs(mdir, spec), list(spec["feature_cols"]), list(spec["missing_cols"]),
                                           _data_dir(cfg), rd.core_proxy_cols)
            for k, v in side.items():
                if spec.get(k) is None:
                    spec[k] = v
            write_json(spec_path, spec)
        except Exception as exc:  # noqa: BLE001 - side metrics are informational
            log.warning("external member %s: validation side metrics unavailable (%s)", name, exc)
    log.info("external member %s: loaded (%s)", name, spec_path)
    return spec, None


# --------------------------------------------------------------------------------------------------
# assembly + metrics
# --------------------------------------------------------------------------------------------------
def assemble_oof(rd: RunData, names: list[str], run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Combine ``oof_<member>.parquet`` files into ``oof.parquet`` / ``forward.parquet`` (INTERFACES section 8).

    A row can carry both a CV prediction (fold >= 0) and a forward-block prediction (fold -1) because the
    forward test eras overlap the last CV test block; the two are kept in separate frames.
    """
    def _frame(part: str) -> pd.DataFrame | None:
        frames = {}
        for n in names:
            p = pd.read_parquet(run_dir / f"oof_{n}.parquet")
            sel = p.loc[p["fold"] >= 0] if part == "oof" else p.loc[p["fold"] == -1]
            if len(sel):
                frames[n] = sel
        if not frames:
            return None
        ids = pd.Index(sorted(set().union(*[set(f.index) for f in frames.values()])), name="id")
        first = next(iter(frames.values()))
        out = pd.DataFrame({"era": rd.era.reindex(ids).astype(np.int32),
                            "fold": first["fold"].reindex(ids).fillna(-1 if part == "forward" else 0).astype(np.int8),
                            data.PRIMARY_TARGET: rd.df[data.PRIMARY_TARGET].reindex(ids).astype(np.float32)}, index=ids)
        for n, f in frames.items():
            out[f"m__{n}"] = f["pred"].reindex(ids).astype(np.float32)
        return out

    oof = _frame("oof")
    fwd = _frame("forward")
    if oof is None:
        raise RuntimeError("no OOF predictions produced")
    oof.to_parquet(run_dir / "oof.parquet")
    if fwd is not None:
        fwd.to_parquet(run_dir / "forward.parquet")
    return oof, fwd


# --------------------------------------------------------------------------------------------------
# reserved audit partition (docs/INTERFACES.md section 22.C)
# --------------------------------------------------------------------------------------------------
BASELINE_DIR = "baseline"


def audit_baseline(rd: RunData, cfg: dict, run_dir: Path, *, prog: Progress | None = None) -> dict | None:
    """The comparison line of the audit: ``knobs.member`` (``lgbm_everest``) at its CONFIG params under the DEFAULT
    universe -- never searched, never knob-tuned -- with the frozen count from its own CV inner splits, refit on the
    pre-audit eras with the refit seeds, predicting the audit eras (``audit_baseline.parquet``). Fitted here, before any
    audit label is read; resumable (``baseline/spec.json`` status done + the run fingerprint). None without an audit."""
    if not rd.audit_eras.size:
        return None
    name = knobsmod.knob_member(cfg)
    mspec = next(m for m in cfg["members"] if m["name"] == name)
    if mspec["family"] in EXTERNAL_FAMILIES or "best_of" in (mspec.get("params") or {}):
        log.warning("audit baseline: %s has no explicit params (best_of / external) -> no baseline column", name)
        return None
    params = dict(mspec.get("params") or {})
    u = knobsmod.default_universe(cfg)
    cid = cell_hash(params, u, rd.fingerprint)
    bdir = run_dir / BASELINE_DIR
    spec_path, out_path = bdir / "spec.json", run_dir / "audit_baseline.parquet"
    if spec_path.exists():
        old = json.loads(spec_path.read_text())
        if old.get("status") == "done" and old.get("cell_id") == cid and out_path.exists():
            log.info("audit baseline: resumed (%s, universe %s)", name, u.key())
            return old
    seed, threads = int(cfg["run"]["seed"]), int(cfg["run"]["threads"])
    seeds = [int(s) for s in (cfg.get("refit") or {}).get("seeds", [0])]
    saved = rd.universe
    rd.universe = u
    try:
        res = _knob_cell(rd, run_dir, name, cid) or _load_protocol(bdir / "cell", cid)
        if res is None:
            res = run_protocol(rd, mspec, params, seed=seed, threads=threads, prog=prog, label=f"audit baseline {name}")
            _save_protocol(bdir / "cell", res, cid)
        bdir.mkdir(parents=True, exist_ok=True)
        _mark_running(rd, {"name": f"baseline__{name}", "family": mspec["family"]}, spec_path, [out_path])
        for stale in bdir.glob("full_s*.bin"):
            stale.unlink()
        blobs, cols = refit_full(rd, mspec, params, res.frozen_n_iter, seeds=seeds, threads=threads, prog=prog,
                                 label=f"audit baseline {name}")
        files = []
        for j, b in enumerate(blobs):
            (bdir / f"full_s{j}.bin").write_bytes(b)
            files.append({"file": f"full_s{j}.bin", "seed": seeds[j], "n_iter": res.frozen_n_iter})
        _write_audit_pred(out_path, rd, _predict_full(rd, mspec, blobs, cols, rd.audit_eras))
    finally:
        rd.universe = saved
    spec = {"name": name, "column": f"baseline__{name}", "family": mspec["family"], "target": mspec["target"],
            "label": f"{name}: config params (unsearched), default universe {u.key()}, refit on eras < audit train_end",
            "universe": u.key(), "cell_id": cid, "frozen_n_iter": res.frozen_n_iter, "inner_iters": res.inner_iters,
            "full_blobs": files, "feature_cols": cols, "run_fingerprint": rd.fingerprint, "protocol": PROTOCOL_VERSION,
            "finished_at": _now(), "status": "done"}
    write_json(spec_path, spec)
    log.info("audit baseline %s: refit (n_iter=%s, %d seeds) and audit predictions written", name, res.frozen_n_iter, len(seeds))
    return spec


def assemble_audit(rd: RunData, names: list[str], run_dir: Path, baseline: dict | None) -> pd.DataFrame:
    """``audit.parquet``: index id over EVERY row of the audit eras; ``era`` int32, ``target_everest`` float32 (copied, never
    scored here), ``m__<member>`` float64 per assembled member and ``baseline__<name>``. Refuses incomplete coverage."""
    ids = rd.df.index[rd.rows(rd.audit_eras)]
    out = pd.DataFrame({"era": rd.era.loc[ids].astype(np.int32),
                        data.PRIMARY_TARGET: rd.df.loc[ids, data.PRIMARY_TARGET].astype(np.float32)}, index=ids)
    out.index.name = "id"
    sources = [(f"m__{n}", run_dir / f"audit_{n}.parquet") for n in names]
    if baseline is not None:
        sources.append((str(baseline["column"]), run_dir / "audit_baseline.parquet"))
    for col, path in sources:
        if not path.exists():
            raise RuntimeError(f"audit: {path.name} missing for {col}")
        p = pd.read_parquet(path)
        if len(p) != len(ids) or not p.index.isin(ids).all():
            raise RuntimeError(f"audit: {path.name} covers {len(p)} rows, expected exactly the {len(ids)} audit rows")
        out[col] = p["pred"].reindex(ids).astype(np.float64)
        if out[col].isna().any():
            raise RuntimeError(f"audit: {path.name} has missing predictions")
    out.to_parquet(run_dir / "audit.parquet")
    log.info("audit.parquet: %d rows, %d eras (%d..%d), columns %s -- scored only by stealth.audit after the blend is frozen",
             len(out), out["era"].nunique(), int(out["era"].min()), int(out["era"].max()), [c for c, _ in sources])
    return out


def knobs_summary(doc: dict) -> dict:
    """Config-name-level summary of ``knobs.json`` for metrics.json (keys, diffs, p_pos, pass; no hyperparameters)."""
    if doc.get("status") != "done":
        return {"status": doc.get("status", "none"), "chosen": (doc.get("chosen") or {}).get("key"), "candidates": []}
    keep = ("key", "dimension", "fit_cv", "fit_fwd", "diff_fwd", "diff_se_fwd", "p_pos_fwd", "diff_cv_indep", "holm_rank",
            "holm_threshold", "pass", "reason", "equivalent_to_default")
    return {"status": "done", "member": doc.get("member"), "alpha_total": doc.get("alpha_total"), "default": doc.get("default"),
            "chosen": doc["chosen"]["key"], "winners": doc.get("winners"),
            "candidates": [{k: c.get(k) for k in keep} for c in doc.get("candidates", [])]}


def baseline_fit(cfg: dict, member_fit_means: dict[str, float], base_cell_fits: dict[str, float] | None = None) -> tuple[float | None, str]:
    """``(baseline fit20, label)`` for the suspect flag: ``gates.baseline_fit`` (a FIXED M1 number recorded from an unsearched
    run on the SAME scoring layout) when set; else the in-run :func:`baseline_member`'s UNSEARCHED base cell
    (``base_cell_fits[base]`` = its ``search_gain.base_params_fit``) when it was searched; else its OOF mean. Returns
    ``(None, label)`` when none exists.

    A searched in-run baseline is itself selected on the scored folds, so the plan's "OOF FIT > M1 + 0.004 -> suspect"
    banner can never fire against it; the unsearched base cell (or a fixed M1 number) is the honest reference.
    """
    bf = (cfg.get("gates") or {}).get("baseline_fit")
    if bf is not None:
        return float(bf), f"gates.baseline_fit={float(bf):.4f}"
    base = baseline_member(cfg)
    cell = (base_cell_fits or {}).get(base)
    if cell is not None:
        return float(cell), f"{base} base cell (unsearched params, in-run)"
    return member_fit_means.get(base), base


def base_cell_fits(members: dict[str, dict]) -> dict[str, float]:
    """``{member: search_gain.base_params_fit}`` for every searched member of a metrics.json ``members`` block."""
    out = {}
    for n, m in (members or {}).items():
        sg = (m or {}).get("search_gain") if isinstance(m, dict) else None
        if isinstance(sg, dict) and sg.get("base_params_fit") is not None:
            out[n] = float(sg["base_params_fit"])
    return out


def baseline_member(cfg: dict) -> str:
    """The in-run baseline member for the suspect flag: ``lgbm_everest`` if present, else the first everest tree member, else the first."""
    names = [m["name"] for m in cfg["members"]]
    if "lgbm_everest" in names:
        return "lgbm_everest"
    for m in cfg["members"]:
        if m["group"] == "everest" and m["family"] == "lgbm":
            return m["name"]
    return names[0]


def member_metrics(cfg: dict, oof: pd.DataFrame, fwd: pd.DataFrame | None, specs: dict[str, dict]) -> tuple[dict, dict, str, list[str]]:
    """Per-member fit summaries, paired tests vs the baseline, leak check status, notes."""
    bs, gates = cfg["bootstrap"], cfg["gates"]
    y, era = oof[data.PRIMARY_TARGET], oof["era"]
    fits: dict[str, pd.Series] = {}
    out: dict[str, dict] = {}
    notes: list[str] = []
    leak = "pass"
    for n, spec in specs.items():
        pred = oof[f"m__{n}"]
        fit = per_era_fit_fast(pred, y, era)
        fits[n] = fit
        sp = per_era_spearman_fast(pred, y, era)
        try:
            metrics.leak_check(sp, float(gates["leak_spearman"]))
        except RuntimeError as exc:
            leak = "FAIL"
            notes.append(f"{n}: {exc}")
            log.error("%s", exc)
        fwd_fit = None
        if fwd is not None:
            fwd_fit = float(per_era_fit_fast(fwd[f"m__{n}"], fwd[data.PRIMARY_TARGET], fwd["era"]).mean())
        out[n] = {"fit": metrics.summarise(fit, block=int(bs["block"]), n_boot=int(bs["n_boot"]), seed=int(bs["seed"])),
                  "spearman_mean": float(sp.mean()), "forward_fit": fwd_fit, "corr_with_bench_val": spec.get("corr_with_bench_val"),
                  "n_iter": spec.get("frozen_n_iter"), "inner_iters": spec.get("inner_iters"), "fit_seconds": spec.get("fit_seconds"),
                  "max_fit_seconds": spec.get("max_fit_seconds"), "group": spec["group"], "target": spec["target"], "family": spec["family"],
                  "arm": spec.get("arm") or arm_of(spec), "rho_c": spec.get("rho_c"), "universe": spec.get("universe"),
                  "search_adopted": spec.get("search_adopted"), "suspect": False}
        if spec["family"] == "cat" and spec.get("max_fit_seconds", 0) > float(cfg["timing"]["catboost_max_fit_seconds"]):
            notes.append(f"{n}: catboost fit {spec['max_fit_seconds']:.0f}s > {cfg['timing']['catboost_max_fit_seconds']}s -> drop candidate")
        sg = spec.get("search_gain")
        out[n]["search_gain"] = sg
        if sg:
            if sg.get("suspect"):
                out[n]["suspect"] = True
                notes.append(f"SUSPECT: {n} search top-k OOF fit {sg['topk_fit']:.4f} > base-params cell {sg['base_params_fit']:.4f} "
                             f"+ {sg['margin']} (winner's curse; forward gain {sg['forward_gain']})")
            if sg.get("forward_confirms") is False:
                notes.append(f"{n}: search gain {sg['gain']:+.4f} on OOF is NOT confirmed on the forward block "
                             f"(forward gain {sg['forward_gain']:+.4f} < 0) -> treat the search as a tie")
    base = baseline_member(cfg)
    margin = float(gates["suspect_gain_over_baseline"])
    bmean, blabel = baseline_fit(cfg, {n: m["fit"]["mean"] for n, m in out.items()}, base_cell_fits(out))
    paired: dict[str, dict] = {}
    if bmean is not None:
        for n in out:
            if metrics.suspect_gain(out[n]["fit"]["mean"], bmean, margin):
                out[n]["suspect"] = True
                notes.append(f"SUSPECT: {n} OOF fit {out[n]['fit']['mean']:.4f} > {blabel} + {margin}")
    if base in out:
        for n in out:
            if n != base:
                d = (fits[n] - fits[base]).dropna()
                paired[f"{n}_vs_{base}"] = cvmod.paired_block_bootstrap(d, block=int(bs["block"]), n_boot=int(bs["n_boot"]), seed=int(bs["seed"])).as_dict()
    return out, paired, leak, notes


# --------------------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------------------
def train_run(cfg: dict, run_id: str, *, resume: bool = False, config_path: Path | None = None,
              artifacts: Path = ARTIFACTS, skip_external: bool = False, require_external: bool = False) -> tuple[dict, int]:
    """Train every member of ``cfg`` into ``artifacts/<run_id>``; returns (metrics dict, exit code).

    The run fingerprint (config + data + protocol) is checked FIRST: a run-id created with anything else is refused with
    exit 1 before ``config.yaml`` / ``metrics.json`` or any artifact is touched. ``skip_external`` never loads a
    ``family: mlp`` member (it is recorded as ``skipped_external``); ``require_external`` turns an external member that
    cannot be loaded into an error instead of ``pending_external`` (pipeline retrain after a finished MLP)."""
    run_dir = Path(artifacts) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = setup_logging(run_dir)
    started = _now()
    t0 = time.perf_counter()
    mpath = run_dir / "metrics.json"
    data_dir = _data_dir(cfg)
    try:
        fp = run_fingerprint(cfg, data_dir)
        check_fingerprint(run_dir, fp)
    except (FingerprintMismatch, OSError) as exc:
        log.error("train refused before touching %s: %s", run_dir, exc)
        logging.getLogger("stealth").removeHandler(fh)
        fh.close()
        return {"run_id": run_id, "config_name": cfg.get("name"), "status": "failed", "started_at": started, "finished_at": _now(),
                "notes": [f"error: {exc}"], "members": {}, "leak_check": None}, EXIT_ERROR
    result: dict = {"run_id": run_id, "config_name": cfg["name"], "git_sha": git_sha(), "status": "partial", "started_at": started,
                    "finished_at": None, "host": host_info(), "timings": {"total_s": None, "per_member_fit_s": {}}, "members": {},
                    "blend": {}, "paired": {}, "leak_check": "pass", "notes": [], "run_fingerprint": fp["fingerprint"],
                    "protocol": PROTOCOL_VERSION, "external_members": {}}
    code = EXIT_OK
    try:
        if not (run_dir / "run_fingerprint.json").exists():
            write_json(run_dir / "run_fingerprint.json", fp)
        if config_path is not None:
            shutil.copyfile(config_path, run_dir / "config.yaml")
        else:
            (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        write_json(mpath, result)
        log.info("run %s: config %s, resume=%s, skip_external=%s, require_external=%s, data_dir=%s, fingerprint %s", run_id,
                 cfg["name"], resume, skip_external, require_external, data_dir, fp["fingerprint"][:16])
        with Progress(float(cfg["timing"]["progress_every_seconds"])) as prog:
            prog.set("loading data")
            rd = RunData(cfg, data_dir, fingerprint=fp["fingerprint"])
            write_json(run_dir / "folds.json", rd.folds_summary())
            write_json(run_dir / "features.json", {"feature_cols": rd.feature_cols, "missing_cols": rd.missing_cols,
                                                   "core_proxy_cols": rd.core_proxy_cols, "surrogate": rd.surrogate.to_dict()})
            prog.set("deciding the training universe (knobs)")
            universe, knobs_doc = knobsmod.select_universe(rd, cfg, run_dir, prog=prog)
            rd.universe = universe
            write_json(run_dir / "folds_eras.json", folds_eras_doc(rd))
            result["universe"] = {"key": universe.key(), **universe.to_dict()}
            result["knobs"] = knobs_summary(knobs_doc)
            baseline = None
            if rd.audit_eras.size:
                prog.set("audit baseline (unsearched, default universe)")
                baseline = audit_baseline(rd, cfg, run_dir, prog=prog)
            specs: dict[str, dict] = {}
            dropped: list[dict] = []
            run_notes: list[str] = []
            order = member_order(cfg)
            for i, m in enumerate(order):
                prog.set(f"member {i + 1}/{len(order)} {m['name']}")
                if m["family"] in EXTERNAL_FAMILIES:
                    if skip_external:
                        log.info("external member %s: not loaded (--skip-external)", m["name"])
                        result["external_members"][m["name"]] = "skipped_external"
                        dropped.append({"member": m["name"], "status": "skipped_external", "reason": "--skip-external"})
                        continue
                    spec, why = load_external_member(rd, m, cfg, run_dir)
                    if spec is None:
                        if require_external:
                            raise RuntimeError(f"--require-external: external member {m['name']} cannot be loaded ({why})")
                        msg = f"external member {m['name']} pending: run scripts/mlp_train.py --run-id {run_id} ({why})"
                        log.warning("%s", msg)
                        run_notes.append(msg)
                        result["external_members"][m["name"]] = "pending"
                        dropped.append({"member": m["name"], "status": "pending_external", "reason": why})
                        continue
                    result["external_members"][m["name"]] = "loaded"
                else:
                    spec = train_member(rd, m, cfg, run_dir, resume=resume, prog=prog)
                if spec.get("status") == "dropped_timing":
                    cap = float(cfg["timing"]["catboost_max_fit_seconds"])
                    run_notes.append(f"{m['name']}: CatBoost dropped ({float(spec['first_fit_seconds']):.0f}s > {cap:.0f}s cap)")
                    dropped.append({"member": m["name"], "status": "dropped_timing", "first_fit_seconds": spec["first_fit_seconds"]})
                    continue
                specs[m["name"]] = spec
                result["timings"]["per_member_fit_s"][m["name"]] = spec.get("fit_seconds")
                result["timings"]["total_s"] = time.perf_counter() - t0
                write_json(mpath, result)
            result["dropped"] = dropped
            prog.set("assembling OOF + metrics")
            oof, fwd = assemble_oof(rd, list(specs), run_dir)
            if rd.audit_eras.size:
                assemble_audit(rd, list(specs), run_dir, baseline)
                result["audit"] = {"enabled": True, "train_end": int(rd.audit_cfg["train_end"]),
                                   "test_start": int(rd.audit_cfg["test_start"]), "n_eras": int(len(rd.audit_eras)),
                                   "eras": [int(rd.audit_eras.min()), int(rd.audit_eras.max())], "file": "audit.parquet",
                                   "baseline": None if baseline is None else baseline["column"],
                                   "note": "predictions of the reserved eras; scored only by stealth.audit after blend"}
            elif (run_dir / "audit.parquet").exists():
                (run_dir / "audit.parquet").unlink()
            members_m, paired, leak, notes = member_metrics(cfg, oof, fwd, specs)
            notes = run_notes + notes + [SELECTION_NOTE]
        result.update({"members": members_m, "paired": paired, "leak_check": leak, "notes": notes,
                       "frozen_n_iter": {n: s.get("frozen_n_iter") for n, s in specs.items()},
                       "frozen_n_iter_per_fold": {n: s.get("frozen_n_iter_per_fold") for n, s in specs.items()},
                       "frozen_n_iter_forward": {n: s.get("frozen_n_iter_forward") for n, s in specs.items()},
                       "baseline": dict(zip(("fit", "label"), baseline_fit(cfg, {n: m["fit"]["mean"] for n, m in members_m.items()},
                                                                           base_cell_fits(members_m)))),
                       "folds": rd.folds_summary(), "n_rows_oof": int(len(oof)), "n_eras_oof": int(oof["era"].nunique())})
        result["status"] = "ok" if leak == "pass" else "failed"
        code = EXIT_OK if leak == "pass" else EXIT_GATE
        for n, mm in members_m.items():
            f = mm["fit"]
            log.info("member %-22s fit20 mean=%.4f se_block=%.4f sharpe=%.3f pct_pos=%.2f forward=%s n_iter=%s%s", n, f["mean"],
                     f.get("se_block", float("nan")), f["sharpe"], f["pct_pos"],
                     None if mm["forward_fit"] is None else round(mm["forward_fit"], 4), mm["n_iter"], "  SUSPECT" if mm["suspect"] else "")
    except Exception as exc:  # noqa: BLE001 - CLI boundary: traceback goes to run.log + metrics.json
        log.error("train failed: %s\n%s", exc, traceback.format_exc())
        result["status"] = "failed"
        result["notes"].append(f"error: {exc!r}")
        result["traceback"] = traceback.format_exc()
        code = EXIT_ERROR
    finally:
        result["finished_at"] = _now()
        result["timings"]["total_s"] = time.perf_counter() - t0
        write_json(mpath, result)
        log.info("run %s finished: status=%s exit=%d total=%.0fs", run_id, result["status"], code, result["timings"]["total_s"])
        logging.getLogger("stealth").removeHandler(fh)
        fh.close()
    return result, code


def main(argv: list[str] | None = None) -> int:
    """``python -m stealth.train --config configs/X.yaml --run-id ID [--resume] [--refit]``."""
    parser = argparse.ArgumentParser(prog="python -m stealth.train")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true", help="skip members whose artifacts exist")
    parser.add_argument("--refit", action="store_true", help="accepted for compatibility; full refits are always written")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    ext = parser.add_mutually_exclusive_group()
    ext.add_argument("--skip-external", action="store_true",
                     help="never load family: mlp members (assemble OOF/forward/audit without them)")
    ext.add_argument("--require-external", action="store_true",
                     help="an external member that cannot be loaded (missing / stale) is an error, not 'pending'")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except (ValueError, OSError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    _, code = train_run(cfg, args.run_id, resume=args.resume, config_path=args.config, artifacts=args.artifacts,
                        skip_external=args.skip_external, require_external=args.require_external)
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
