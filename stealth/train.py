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

Hyperparameter VALUES stay under ``artifacts/`` (never in handoff results).
"""
from __future__ import annotations

import argparse
import datetime as _dt
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
from stealth.cv import Fold

log = logging.getLogger("stealth.train")  # explicit: under `python -m` __name__ is __main__

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts"
_FAMILIES = ("lgbm", "xgb", "cat", "ridge")
_GROUPS = ("everest", "aux")
_INPUTS = ("nan", "linear")
_REQUIRED_TOP = ("name", "run", "cv", "members", "search", "blend", "bootstrap", "gates", "timing")
_RUN_KEYS = ("seed", "threads", "min_era", "max_era", "min_rows", "time_decay", "data_dir")
_CV_KEYS = ("n_folds", "purge", "embargo", "inner_frac", "inner_gap", "forward")
_MEMBER_KEYS = ("name", "family", "target", "group", "features", "input", "params", "n_estimators", "es_cap")
EXIT_OK, EXIT_ERROR, EXIT_GATE = 0, 1, 2
MIN_GAP = 20  # floor on every train/test and early-stopping gap (PLAN: purge + embargo >= 20 eras)


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
    if problems:
        raise ValueError(f"{path}: invalid config:\n  - " + "\n  - ".join(problems))
    cfg.setdefault("refit", {"seeds": [0]})
    cfg.setdefault("knobs", {})
    return cfg


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
# data context
# --------------------------------------------------------------------------------------------------
class RunData:
    """Loaded train split (int8 features kept as-is), run-era subset, folds and per-run feature metadata.

    ``df`` holds the eras in play (after ``run.min_era/max_era/min_rows``) with int8 features, float32
    targets and the int32 ``era`` column. Input matrices are built per fold from the int8 frame
    (``matrix``) so no float32 copy of the whole split is ever kept.
    """

    def __init__(self, cfg: dict, data_dir: Path) -> None:
        run = cfg["run"]
        self.cfg = cfg
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
        self.df = data.filter_eras(full, min_era=run.get("min_era"), max_era=run.get("max_era"), min_rows=run.get("min_rows"))
        del full
        self.era = self.df["era"]
        self.eras = np.unique(self.era.to_numpy()).astype(np.int32)
        c = cfg["cv"]
        self.folds: list[Fold] = cvmod.purged_folds(self.eras, int(c["n_folds"]), purge=int(c["purge"]), embargo=int(c["embargo"]),
                                                   inner_frac=float(c["inner_frac"]), inner_gap=int(c["inner_gap"]))
        self.forward: Fold | None = None
        fw = c.get("forward") or {}
        if fw.get("enabled"):
            self.forward = cvmod.forward_block(self.eras, train_end=int(fw["train_end"]), test_start=int(fw["test_start"]),
                                               inner_frac=float(c["inner_frac"]), inner_gap=int(c["inner_gap"]))
        self.min_gap = max(MIN_GAP, min(int(c["purge"]) + int(c["embargo"]), int(c["inner_gap"])))
        cvmod.assert_fold_gaps(self.folds + ([self.forward] if self.forward else []), min_gap=self.min_gap)
        self.sharpe_cols_cache: dict[tuple[str, str], list[str]] = {}
        bags = cfg.get("feature_bags") or {}
        self.bags = features.random_column_bags(self.feature_cols, n_bags=int(bags.get("n_bags", 4)),
                                                fraction=float(bags.get("fraction", 0.5)), seed=int(bags.get("seed", 0)))
        log.info("run data: %d rows, %d eras (%d..%d), %d folds%s", len(self.df), len(self.eras), self.eras.min(), self.eras.max(),
                 len(self.folds), ", forward block" if self.forward else "")

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

    def weights(self, mask: np.ndarray) -> np.ndarray | None:
        if not self.cfg["run"].get("time_decay"):
            return None
        return data.time_decay_weights(self.era.loc[mask], True)

    def resolve_features(self, spec: str | list[str], fold: Fold | None, target: str) -> list[str]:
        """Concrete feature list; ``sharpe_subset`` is computed on the fold's TRAINING eras only (cached)."""
        sharpe_cols = None
        if spec == "sharpe_subset":
            key = (fold.name if fold else "full", target)
            if key not in self.sharpe_cols_cache:
                thr = float((self.cfg.get("sharpe_subset") or {}).get("threshold", 0.1))
                mask = self.rows(fold.train_eras) if fold else np.ones(len(self.df), dtype=bool)
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
    """Training mask: rows in ``eras`` whose ``target`` is not NaN."""
    mask = rd.rows(eras)
    mask &= rd.df[target].notna().to_numpy()
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


def run_protocol(rd: RunData, spec: dict, params: dict, *, seed: int, threads: int, prog: Progress | None = None,
                 label: str = "") -> ProtocolResult:
    """Inner early stopping per fold -> frozen median count -> fixed-count fold fits -> OOF (+ forward) predictions."""
    res = ProtocolResult()
    t0 = time.perf_counter()
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
    """Refit on ALL run eras with the frozen count, once per seed; returns (blobs, feature_cols)."""
    cols = rd.resolve_features(spec["features"], None, spec["target"])
    blobs = []
    for s in seeds:
        if prog:
            prog.set(f"{label}: full refit seed={s}")
        fm = _fit(rd, spec, params, cols, rd.eras, n_iter, seed=int(s), threads=threads)
        log.info("%s: full refit seed=%d n_iter=%s (%.1fs, %d bytes)", label, s, fm.n_iter, fm.fit_seconds, len(fm.blob))
        blobs.append(fm.blob)
    return blobs, cols


def validation_bench_corr(spec: dict, blobs: list[bytes], cols: list[str], missing_cols: list[str], data_dir: Path) -> float | None:
    """Mean per-era corr(prediction, v1_sherpa) on the VALIDATION split (exact rho_b; None if files are missing)."""
    try:
        val = data.load_split("validation", data_dir, columns=list(cols) + list(missing_cols))
        bench = data.load_benchmark("validation", data_dir)
    except FileNotFoundError:
        return None
    X = data.features_linear(val, cols, missing_cols) if spec["input"] == "linear" else data.features_nan(val, cols)
    pred = np.mean([models.predict_member((spec["family"], b), X) for b in blobs], axis=0)
    s = metrics.per_era_corr(pd.Series(pred, index=X.index), bench.reindex(X.index), val["era"])
    return float(s.mean())


def _params_hash(params: dict) -> str:
    import hashlib

    return hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _save_protocol(d: Path, res: ProtocolResult) -> None:
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
                                   "frozen_n_iter_forward": res.frozen_n_iter_forward, "count_sources": res.count_sources})


def _load_protocol(d: Path) -> ProtocolResult | None:
    if not (d / "oof.parquet").exists() or not (d / "result.json").exists():
        return None
    res = ProtocolResult()
    df = pd.read_parquet(d / "oof.parquet")
    cvm = df["fold"] >= 0
    res.oof, res.oof_fold = df.loc[cvm, "pred"].astype(np.float64), df.loc[cvm, "fold"].astype(np.int8)
    if (~cvm).any():
        res.forward = df.loc[~cvm, "pred"].astype(np.float64)
    r = json.loads((d / "result.json").read_text())
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
    params = dict(mspec.get("params") or {})
    if "best_of" in params:
        src = run_dir / "search" / f"{params.pop('best_of')}.json"
        if not src.exists():
            raise FileNotFoundError(f"{mspec['name']}: params.best_of needs {src} (run the searched member first)")
        best = json.loads(src.read_text())[0]["params"]
        params = {**best, **params}
        log.info("%s: params from search top-1 of %s", mspec["name"], src.stem)
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


def train_member(rd: RunData, mspec: dict, cfg: dict, run_dir: Path, *, resume: bool, prog: Progress) -> dict:
    """Run one member end-to-end; returns its spec.json dict. ``resume`` reuses finished artifacts."""
    name = mspec["name"]
    mdir = run_dir / "members" / name
    spec_path, oof_path = mdir / "spec.json", run_dir / f"oof_{name}.parquet"
    if resume and spec_path.exists() and oof_path.exists():
        spec = json.loads(spec_path.read_text())
        if spec.get("status") == "done":
            log.info("member %s: resumed from %s", name, mdir)
            return spec
    seed, threads = int(cfg["run"]["seed"]), int(cfg["run"]["threads"])
    seeds = [int(s) for s in (cfg.get("refit") or {}).get("seeds", [0])]
    params = _resolve_params(mspec, run_dir)
    search_cfg = cfg["search"].get(name)
    t0 = time.perf_counter()
    if search_cfg:
        from stealth import search as searchmod

        sdir = run_dir / "search" / name
        cache: dict[str, ProtocolResult] = {}

        def objective(cell: dict) -> float:
            # a grid cell overrides the member's base params; everything not in the grid (bagging, max_bin, ...) stays
            p = {**params, **cell}
            h = _params_hash(p)
            res = _load_protocol(sdir / h)  # trial cache on disk doubles as the resume unit
            if res is None:
                res = run_protocol(rd, mspec, p, seed=seed, threads=threads, prog=prog, label=f"{name}[{h}]")
                _save_protocol(sdir / h, res)
            cache[h] = res
            fit = metrics.per_era_fit(res.oof, rd.df.loc[res.oof.index, data.PRIMARY_TARGET], rd.era.loc[res.oof.index])
            return float(fit.mean())

        results = searchmod.run_search(name, search_cfg["grid"], objective, n_trials=int(search_cfg["n_trials"]),
                                       storage=run_dir / "optuna.db", seed=seed)
        # search/<name>.json carries the FULL params (base + cell) so params.best_of dependents inherit everything
        results = [({**params, **cell}, v) for cell, v in results]
        write_json(run_dir / "search" / f"{name}.json", [{"params": p, "value": v} for p, v in results])
        top = searchmod.top_k_params(results, int(search_cfg["top_k"]))
        runs = []
        for p in top:
            h = _params_hash(p)
            runs.append(cache.get(h) or _load_protocol(sdir / h) or run_protocol(rd, mspec, p, seed=seed, threads=threads, prog=prog, label=name))
        base = runs[0]
        oof = pd.concat([r.oof for r in runs], axis=1).mean(axis=1)
        forward = pd.concat([r.forward for r in runs], axis=1).mean(axis=1) if base.forward is not None else None
        res = ProtocolResult()
        res.oof, res.oof_fold, res.forward = oof, base.oof_fold, forward
        res.fold_blobs, res.forward_blob = base.fold_blobs, base.forward_blob
        res.inner_iters, res.frozen_n_iter = base.inner_iters, base.frozen_n_iter
        res.inner_iter_forward, res.frozen_n_iter_per_fold = base.inner_iter_forward, base.frozen_n_iter_per_fold
        res.frozen_n_iter_forward, res.count_sources = base.frozen_n_iter_forward, base.count_sources
        res.fit_seconds = sum(r.fit_seconds for r in runs)
        res.max_fit_seconds, res.feature_cols = max(r.max_fit_seconds for r in runs), base.feature_cols
        top_params, frozen_per_cfg = top, [r.frozen_n_iter for r in runs]
        # winner's-curse check: the BASE params cell is always evaluated; top-k minus base on OOF and forward
        base_h = _params_hash(params)
        base_res = cache.get(base_h) or _load_protocol(sdir / base_h)
        if base_res is None:
            base_res = run_protocol(rd, mspec, params, seed=seed, threads=threads, prog=prog, label=f"{name}[base]")
            _save_protocol(sdir / base_h, base_res)
        search_gain = _search_gain(rd, res, base_res, [_params_hash(p) for p in top], base_h,
                                   float(cfg["gates"]["suspect_gain_over_baseline"]))
        write_json(run_dir / "search" / f"{name}.gain.json", search_gain)
        log.info("%s search gain: top-%d OOF fit %.4f vs base-params cell %.4f (gain %+.4f%s); forward %s vs %s (gain %s)%s", name,
                 len(top), search_gain["topk_fit"], search_gain["base_params_fit"], search_gain["gain"],
                 ", SUSPECT" if search_gain["suspect"] else "", search_gain["topk_forward_fit"], search_gain["base_params_forward_fit"],
                 search_gain["forward_gain"], "" if search_gain["forward_confirms"] in (True, None) else " NOT CONFIRMED on the forward block")
    else:
        res = run_protocol(rd, mspec, params, seed=seed, threads=threads, prog=prog, label=name)
        top_params, frozen_per_cfg = [params], [res.frozen_n_iter]
        search_gain = None
    # artifacts
    mdir.mkdir(parents=True, exist_ok=True)
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
    prog.set(f"{name}: validation benchmark correlation")
    corr = validation_bench_corr(mspec, [(mdir / f["file"]).read_bytes() for f in full_files], full_cols, rd.missing_cols,
                                 Path(cfg["run"]["data_dir"]) if Path(cfg["run"]["data_dir"]).is_absolute() else REPO_ROOT / cfg["run"]["data_dir"])
    df = pd.DataFrame({"era": rd.era.loc[res.oof.index].astype(np.int32), "fold": res.oof_fold.loc[res.oof.index], "pred": res.oof.astype(np.float32)})
    if res.forward is not None:
        df = pd.concat([df, pd.DataFrame({"era": rd.era.loc[res.forward.index].astype(np.int32), "fold": np.int8(-1),
                                          "pred": res.forward.astype(np.float32)})])
    df.index.name = "id"
    df.to_parquet(oof_path)
    spec = {**mspec, "params": top_params[0], "top_params": top_params, "feature_cols": full_cols, "all_feature_cols": rd.feature_cols,
            "missing_cols": rd.missing_cols, "feature_cols_per_fold": res.feature_cols, "inner_iters": res.inner_iters,
            "inner_iter_forward": res.inner_iter_forward, "frozen_n_iter": res.frozen_n_iter,
            "frozen_n_iter_per_fold": res.frozen_n_iter_per_fold, "frozen_n_iter_forward": res.frozen_n_iter_forward,
            "count_sources": res.count_sources, "frozen_n_iter_per_config": frozen_per_cfg, "fit_seconds": res.fit_seconds,
            "max_fit_seconds": res.max_fit_seconds, "member_seconds": time.perf_counter() - t0, "full_blobs": full_files,
            "corr_with_bench_val": corr, "search_gain": search_gain, "status": "done"}
    write_json(spec_path, spec)
    log.info("member %s done in %.0fs (frozen n_iter=%s, rho_b(val)=%s)", name, spec["member_seconds"], res.frozen_n_iter,
             None if corr is None else round(corr, 4))
    return spec


def _search_gain(rd: RunData, topk: ProtocolResult, base: ProtocolResult, top_hashes: list[str], base_hash: str,
                 margin: float) -> dict:
    """Top-k-average minus base-params-cell fit20 on OOF (and the forward block); ``suspect`` when the OOF gain > margin.

    Hashes identify grid cells without exposing hyperparameter values (the file lands under artifacts/ anyway).
    """
    def _mean_fit(pred: pd.Series | None) -> float | None:
        if pred is None:
            return None
        return float(metrics.per_era_fit(pred, rd.df.loc[pred.index, data.PRIMARY_TARGET], rd.era.loc[pred.index]).mean())

    topk_fit, base_fit = _mean_fit(topk.oof), _mean_fit(base.oof)
    topk_fwd, base_fwd = _mean_fit(topk.forward), _mean_fit(base.forward)
    gain = float(topk_fit - base_fit)
    fwd_gain = None if topk_fwd is None or base_fwd is None else float(topk_fwd - base_fwd)
    return {"topk_fit": topk_fit, "base_params_fit": base_fit, "gain": gain, "margin": margin, "suspect": bool(gain > margin),
            "topk_forward_fit": topk_fwd, "base_params_forward_fit": base_fwd, "forward_gain": fwd_gain,
            "forward_confirms": None if fwd_gain is None else bool(fwd_gain >= 0.0),
            "topk_cells": top_hashes, "base_cell": base_hash, "base_cell_in_topk": base_hash in top_hashes}


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


def baseline_fit(cfg: dict, member_fit_means: dict[str, float]) -> tuple[float | None, str]:
    """``(baseline fit20, label)`` for the suspect flag: ``gates.baseline_fit`` (a FIXED M1 number recorded from an unsearched
    run) when set, else the in-run :func:`baseline_member`'s OOF mean. Returns ``(None, label)`` when neither exists.

    A searched in-run baseline is itself selected on the scored folds, so the plan's "OOF FIT > M1 + 0.004 -> suspect"
    banner can never fire against it; ``gates.baseline_fit`` is the honest reference once M1 (handoff 02) is known.
    """
    bf = (cfg.get("gates") or {}).get("baseline_fit")
    if bf is not None:
        return float(bf), f"gates.baseline_fit={float(bf):.4f}"
    base = baseline_member(cfg)
    return member_fit_means.get(base), base


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
        fit = metrics.per_era_fit(pred, y, era)
        fits[n] = fit
        sp = metrics.per_era_spearman(pred, y, era)
        try:
            metrics.leak_check(sp, float(gates["leak_spearman"]))
        except RuntimeError as exc:
            leak = "FAIL"
            notes.append(f"{n}: {exc}")
            log.error("%s", exc)
        fwd_fit = None
        if fwd is not None:
            fwd_fit = float(metrics.per_era_fit(fwd[f"m__{n}"], fwd[data.PRIMARY_TARGET], fwd["era"]).mean())
        out[n] = {"fit": metrics.summarise(fit, block=int(bs["block"]), n_boot=int(bs["n_boot"]), seed=int(bs["seed"])),
                  "spearman_mean": float(sp.mean()), "forward_fit": fwd_fit, "corr_with_bench_val": spec.get("corr_with_bench_val"),
                  "n_iter": spec.get("frozen_n_iter"), "inner_iters": spec.get("inner_iters"), "fit_seconds": spec.get("fit_seconds"),
                  "max_fit_seconds": spec.get("max_fit_seconds"), "group": spec["group"], "target": spec["target"], "family": spec["family"],
                  "suspect": False}
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
    bmean, blabel = baseline_fit(cfg, {n: m["fit"]["mean"] for n, m in out.items()})
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
              artifacts: Path = ARTIFACTS) -> tuple[dict, int]:
    """Train every member of ``cfg`` into ``artifacts/<run_id>``; returns (metrics dict, exit code)."""
    run_dir = Path(artifacts) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = setup_logging(run_dir)
    started = _now()
    t0 = time.perf_counter()
    mpath = run_dir / "metrics.json"
    result: dict = {"run_id": run_id, "config_name": cfg["name"], "git_sha": git_sha(), "status": "partial", "started_at": started,
                    "finished_at": None, "host": host_info(), "timings": {"total_s": None, "per_member_fit_s": {}}, "members": {},
                    "blend": {}, "paired": {}, "leak_check": "pass", "notes": []}
    code = EXIT_OK
    try:
        if config_path is not None:
            shutil.copyfile(config_path, run_dir / "config.yaml")
        else:
            (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        write_json(mpath, result)
        data_dir = Path(cfg["run"]["data_dir"])
        if not data_dir.is_absolute():
            data_dir = REPO_ROOT / data_dir
        log.info("run %s: config %s, resume=%s, data_dir=%s", run_id, cfg["name"], resume, data_dir)
        with Progress(float(cfg["timing"]["progress_every_seconds"])) as prog:
            prog.set("loading data")
            rd = RunData(cfg, data_dir)
            write_json(run_dir / "folds.json", rd.folds_summary())
            write_json(run_dir / "features.json", {"feature_cols": rd.feature_cols, "missing_cols": rd.missing_cols,
                                                   "core_proxy_cols": rd.core_proxy_cols, "surrogate": rd.surrogate.to_dict()})
            specs: dict[str, dict] = {}
            order = member_order(cfg)
            for i, m in enumerate(order):
                prog.set(f"member {i + 1}/{len(order)} {m['name']}")
                specs[m["name"]] = train_member(rd, m, cfg, run_dir, resume=resume, prog=prog)
                result["timings"]["per_member_fit_s"][m["name"]] = specs[m["name"]].get("fit_seconds")
                result["timings"]["total_s"] = time.perf_counter() - t0
                write_json(mpath, result)
            prog.set("assembling OOF + metrics")
            oof, fwd = assemble_oof(rd, list(specs), run_dir)
            members_m, paired, leak, notes = member_metrics(cfg, oof, fwd, specs)
        result.update({"members": members_m, "paired": paired, "leak_check": leak, "notes": notes,
                       "frozen_n_iter": {n: s.get("frozen_n_iter") for n, s in specs.items()},
                       "frozen_n_iter_per_fold": {n: s.get("frozen_n_iter_per_fold") for n, s in specs.items()},
                       "frozen_n_iter_forward": {n: s.get("frozen_n_iter_forward") for n, s in specs.items()},
                       "baseline": {"label": baseline_fit(cfg, {n: m["fit"]["mean"] for n, m in members_m.items()})[1],
                                    "fit": baseline_fit(cfg, {n: m["fit"]["mean"] for n, m in members_m.items()})[0]},
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
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except (ValueError, OSError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    _, code = train_run(cfg, args.run_id, resume=args.resume, config_path=args.config, artifacts=args.artifacts)
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
