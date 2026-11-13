"""Results reporter: ``artifacts/<run>/metrics.json`` (+ ``blend.json``, ``run.log``) -> ``handoffs/results/NN-*``.

CLI contract (docs/INTERFACES.md section 10)::

    python -m stealth.report --run-id R --handoff NN [--no-git] [--no-push] [--log-lines 60]
                             [--diag-json artifacts/diag/<file>.upload.json] [--no-recompute]

Writes three files under ``handoffs/results/``:

* ``NN-results.md``   env table, metrics table (incl. paired diff_mean/diff_se/p_pos), regime table, wall times,
                      last 60 lines of run.log, traceback when the run failed
* ``NN-metrics.json`` machine-readable schema (docs/PLAN.md "Handoff protocol" item 7): run_id, git_sha, status,
                      host{os,cpu,ram_gb,gpu,driver,python,libs,tz}, timings, configs[]{name, fit_mean, fit_se, sharpe,
                      forward_fit, rho_b_val, rho_c, regime{bpos,bneg,mix}, ci90, paired_vs_baseline{diff,se,p_pos}}
* ``NN-log-tail.txt`` the log tail on its own

M2-M6 additions (docs/INTERFACES.md section 20), present only when the run produced them: ``universe``, ``knobs``
(candidate keys + diff/p_pos/pass, chosen key), ``search`` (per searched member: gain, forward_gain, paired_cv,
adopted), ``ladder``, ``member_corr_max_pair`` (+ ``member_corr`` matrix), ``neutralisation`` (table + chosen +
p_source), ``final``, ``dropped`` (timing / pending external / redundant / not admitted), ``diag`` (with
``--diag-json``: the validation-diagnostics panel + the diag/OOF fit ratio, informational only). The blend row
reports the CHOSEN blend (cross-fitted when NNLS was adopted). When metrics.json has no ``blend_vs_<baseline>``
paired row (M1 runs), it is recomputed from ``oof.parquet`` with the blend.json weights and labelled as such.

Then ``git add handoffs/results && git commit -m "results: NN <slug>" && git push``. On push failure it prints
``PUSH FAILED — results are saved locally in handoffs/results/; ask the human to run git push`` and exits 1
(the files are already written). ``--no-git`` skips commit+push (sandbox testing); ``--no-push`` commits only.

Rules honoured here: config NAMES only (hyperparameter values stay in ``artifacts/``); the reporter never reads
``members/*/spec.json`` or ``config.yaml`` beyond its ``name``; every text that goes into a results file passes
through :func:`scrub` (API keys / tokens / tokenised remote URLs are redacted); git output is scrubbed before it
is printed. Works when ``metrics.json`` is missing (status ``failed``).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from stealth import selfcheck

log = logging.getLogger("stealth.report")  # explicit: under `python -m` __name__ is __main__

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
HANDOFFS = REPO / "handoffs"
RESULTS = HANDOFFS / "results"

PUSH_FAILED_MSG = "PUSH FAILED — results are saved locally in handoffs/results/; ask the human to run git push"

#: same families of secrets the pre-commit hook blocks; anything matching is replaced before it is written
_SECRET_PATTERNS = [
    re.compile(r"EIQ_API_KEY\s*=\s*\S{8,}"),
    re.compile(r"EVEREST_API_KEY\s*=\s*\S{8,}"),
    re.compile(r"X-API-Key\s*[:=]\s*\S{8,}", re.IGNORECASE),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"gho_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"https://[^/\s]*@github\.com"),
    re.compile(r"https://[^/\s]*:[^/\s]*@[^/\s]+"),
]
_MEAN_PREFERRED = 0.009  # live-record benchmark FIT mean used for the headline regime row


class GitError(RuntimeError):
    """A git step failed; ``.stage`` is 'add' | 'commit' | 'push'."""

    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(f"git {stage} failed: {detail}")
        self.stage = stage


# --------------------------------------------------------------------------------------------- helpers


def scrub(text: str) -> str:
    """Redact anything that looks like a credential or a tokenised URL. Idempotent."""
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return None


def _config_name(run_dir: Path, metrics: dict[str, Any] | None) -> str | None:
    """Config NAME only; the YAML's other keys (hyperparameters) are never surfaced."""
    if metrics and metrics.get("config_name"):
        return str(metrics["config_name"])
    cfg = run_dir / "config.yaml"
    if cfg.exists():
        try:
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            return str(data.get("name")) if data.get("name") else None
        except yaml.YAMLError:
            return None
    return None


def log_tail(run_log: Path, n: int = 60) -> list[str]:
    if not run_log.exists():
        return []
    lines = run_log.read_text(encoding="utf-8", errors="replace").splitlines()
    return [scrub(line) for line in lines[-n:]]


def extract_traceback(run_log: Path) -> str | None:
    """Last Python traceback in ``run_log`` (from the last 'Traceback (most recent call last)' to the end)."""
    if not run_log.exists():
        return None
    text = run_log.read_text(encoding="utf-8", errors="replace")
    idx = text.rfind("Traceback (most recent call last)")
    if idx < 0:
        return None
    return scrub(text[idx:].strip()[-6000:])


def handoff_slug(handoff: str, handoffs_dir: Path = HANDOFFS, fallback: str | None = None) -> str:
    """Slug from ``handoffs/NN-<slug>.md`` when it exists, else ``fallback``, else 'run'."""
    for p in sorted(handoffs_dir.glob(f"{handoff}-*.md")):
        return p.stem[len(handoff) + 1:]
    return (fallback or "run").replace(" ", "-")


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "-"
        return f"{x:.{nd}f}" if isinstance(x, float) else str(x)
    return str(x)


def _se(summary: dict[str, Any] | None) -> float | None:
    """Block-bootstrap SE of the mean: ``se_block`` when the summary carries it, else ``(ci90_hi - ci90_lo) / (2 * 1.645)``.

    Never the naive ``std / sqrt(n_eras)``: with a 20-day target the eras are autocorrelated and that figure is
    ~3x too small (PLAN noise floor 0.0033 per config).
    """
    if not summary:
        return None
    se = summary.get("se_block")
    if isinstance(se, (int, float)) and math.isfinite(float(se)):
        return float(se)
    ci = summary.get("ci90")
    if isinstance(ci, (list, tuple)) and len(ci) == 2 and all(isinstance(x, (int, float)) for x in ci):
        lo, hi = float(ci[0]), float(ci[1])
        if math.isfinite(lo) and math.isfinite(hi) and hi >= lo:
            return (hi - lo) / (2.0 * 1.645)
    return None


def _paired_for(name: str, paired: dict[str, Any]) -> dict[str, Any] | None:
    """The paired test row where ``name`` is the A side (``<A>_vs_<B>``)."""
    for key, val in paired.items():
        if key.startswith(f"{name}_vs_") and isinstance(val, dict):
            return {
                "baseline": key[len(name) + 4:],
                "diff": val.get("diff_mean"),
                "se": val.get("diff_se"),
                "p_pos": val.get("p_pos"),
                "ci90": val.get("ci90"),
            }
    return None


def _headline_regime(blend: dict[str, Any] | None) -> dict[str, Any] | None:
    """``{bpos, bneg, mix}`` at the blend's chosen p_b and the live-record benchmark mean (0.009 or nearest)."""
    if not blend or not isinstance(blend.get("regime"), dict) or not blend["regime"]:
        return None
    regime = {k: v for k, v in blend["regime"].items() if k != "meta" and isinstance(v, dict)}
    if not regime:
        return None

    def _as_float(key: str) -> float:
        try:
            return float(str(key).split("=")[-1])
        except ValueError:
            return math.inf

    mean_key = min(regime, key=lambda k: abs(_as_float(k) - _MEAN_PREFERRED))
    by_pb = regime.get(mean_key) or {}
    if not isinstance(by_pb, dict) or not by_pb:
        return None
    target_pb = blend.get("p_b")
    if isinstance(target_pb, (int, float)):
        pb_key = min(by_pb, key=lambda k: abs(_as_float(k) - float(target_pb)))
    else:
        pb_key = next(iter(by_pb))
    cell = by_pb.get(pb_key) or {}
    if not isinstance(cell, dict):
        return None
    out = {k: cell.get(k) for k in ("bpos", "bneg", "mix")}
    out["at"] = {"bench_fit_mean": mean_key, "p_b": pb_key}
    return out


# --------------------------------------------------------------------------------------------- M2-M6 sections
# Every helper below copies an explicit allow-list of keys: metrics, config/arm/member/universe NAMES, decisions and
# reasons. Hyperparameter values never get here (search/<m>.json and spec.json are never read).

_SUMMARY_KEYS = ("mean", "se_block", "std", "sharpe", "pct_pos", "frac_roll20_neg", "ci90", "n_eras")
_PAIRED_KEYS = ("diff_mean", "diff_se", "p_pos", "ci90", "n_eras")
_KNOB_KEYS = ("key", "dimension", "fit_cv", "fit_fwd", "diff_fwd", "diff_se_fwd", "p_pos_fwd", "diff_cv_indep", "holm_rank",
              "holm_threshold", "pass", "reason", "equivalent_to_default")
_SEARCH_KEYS = ("n_trials", "topk_fit", "base_params_fit", "gain", "margin", "suspect", "topk_forward_fit",
                "base_params_forward_fit", "forward_gain", "forward_confirms", "base_cell_in_topk", "adopted", "adopt_reason",
                "universe")
_LADDER_KEYS = ("arm", "members", "admitted", "reason", "diff_mean", "diff_se", "p_pos", "fwd_diff", "dsharpe", "p_pos_sharpe",
                "corr_with_incumbent", "dropped_redundant")
_NEUTRAL_ROW_KEYS = ("p_b", "p_c", "fit", "score", "score_sharpe", "W", "eligible", "in_tie_set")
_UNIVERSE_KEYS = ("key", "era_start", "rows_rule", "time_decay", "rows_low_weight")


def _pick(d: Any, keys: tuple[str, ...]) -> dict[str, Any] | None:
    if not isinstance(d, dict):
        return None
    return {k: d.get(k) for k in keys if k in d}


def _summary(d: Any) -> dict[str, Any] | None:
    out = _pick(d, _SUMMARY_KEYS)
    if out is not None:
        out["fit_se"] = _se(d)
    return out


def knobs_doc(run_dir: Path, metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    """The training-universe decision (``metrics.json.knobs``, else ``knobs.json``): candidate keys + diff/p_pos/pass."""
    src = (metrics or {}).get("knobs")
    if not isinstance(src, dict) or not src.get("candidates"):
        raw = _load_json(run_dir / "knobs.json")
        if isinstance(raw, dict) and raw.get("candidates"):
            src = {**raw, "chosen": (raw.get("chosen") or {}).get("key") if isinstance(raw.get("chosen"), dict) else raw.get("chosen")}
    if not isinstance(src, dict) or not (src.get("candidates") or src.get("chosen")):
        return None
    return {"status": src.get("status"), "member": src.get("member"), "alpha_total": src.get("alpha_total"),
            "default": src.get("default"), "chosen": src.get("chosen"), "winners": src.get("winners"),
            "candidates": [_pick(c, _KNOB_KEYS) for c in src.get("candidates") or [] if isinstance(c, dict)]}


def _search_row(name: str, g: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"member": name, **(_pick(g, _SEARCH_KEYS) or {})}
    row["paired_cv"] = _pick(g.get("paired_cv"), _PAIRED_KEYS)
    return row


def search_doc(run_dir: Path, metrics: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Per searched member: gain, forward_gain, paired_cv, adopted (``members.<m>.search_gain``, else ``*.gain.json``)."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, m in ((metrics or {}).get("members") or {}).items():
        if isinstance(m, dict) and isinstance(m.get("search_gain"), dict):
            rows.append(_search_row(name, m["search_gain"]))
            seen.add(name)
    sdir = run_dir / "search"
    if sdir.is_dir():
        for p in sorted(sdir.glob("*.gain.json")):
            name = p.name[: -len(".gain.json")]
            g = _load_json(p) if name not in seen else None
            if isinstance(g, dict):
                rows.append(_search_row(name, g))
    return rows


def ladder_doc(blend: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [_pick(r, _LADDER_KEYS) for r in (blend or {}).get("ladder") or [] if isinstance(r, dict)]


def member_corr_doc(blend: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """``(max-pair summary incl. the top 5 pairs, {names, matrix rounded to 4 dp})`` from ``blend.json.member_corr``."""
    mc = (blend or {}).get("member_corr")
    if not isinstance(mc, dict) or not mc.get("names") or not mc.get("matrix"):
        return None, None
    names, mat = list(mc["names"]), mc["matrix"]
    pairs = []
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            try:
                v = float(mat[i][j])
            except (IndexError, TypeError, ValueError):
                continue
            if math.isfinite(v):
                pairs.append({"a": a, "b": names[j], "corr": v})
    pairs.sort(key=lambda p: -p["corr"])
    rounded = [[round(float(v), 4) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None for v in row]
               for row in mat]
    summary = {"max_pair": pairs[0] if pairs else None, "top_pairs": pairs[:5], "n_members": len(names)}
    return summary, {"names": names, "matrix": rounded}


def neutralisation_doc(blend: dict[str, Any] | None) -> dict[str, Any] | None:
    n = (blend or {}).get("neutralisation")
    if not isinstance(n, dict):
        return None
    return {"p_b": n.get("p_b"), "p_c": n.get("p_c"), "p_source": n.get("p_source"), "z_source": n.get("z_source"),
            "reason": n.get("reason"), "criterion": n.get("criterion"), "would_choose": n.get("would_choose"),
            "table": [_pick(r, _NEUTRAL_ROW_KEYS) for r in n.get("table") or [] if isinstance(r, dict)]}


def final_doc(blend: dict[str, Any] | None) -> dict[str, Any] | None:
    f = (blend or {}).get("final")
    if not isinstance(f, dict):
        return None
    out = {k: f.get(k) for k in ("admitted_arms", "rule", "p_b", "p_c", "p_source", "forward_fit", "score_mix_central",
                                 "score_sharpe_central", "W") if k in f}
    out["oof"] = _summary(f.get("oof"))
    return out


def dropped_doc(metrics: dict[str, Any] | None, blend: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Members that are not in the bundle and why: timing / pending external (train), redundant / not admitted (blend)."""
    out: list[dict[str, Any]] = []
    for d in (metrics or {}).get("dropped") or []:
        if isinstance(d, dict) and d.get("member"):
            detail = d.get("reason")
            if d.get("first_fit_seconds") is not None:
                detail = f"first fit {_fmt(float(d['first_fit_seconds']), 0)} s > cap"
            out.append({"member": d["member"], "status": d.get("status"), "detail": detail})
    for name, r in ((blend or {}).get("redundant") or {}).items():
        if isinstance(r, dict):
            out.append({"member": name, "status": "redundant", "detail": f"corr {_fmt(r.get('corr'), 3)} with {r.get('other')}"})
    for r in (blend or {}).get("ladder") or []:
        if isinstance(r, dict) and not r.get("admitted"):
            for m in r.get("members") or []:
                out.append({"member": m, "status": "not_admitted", "detail": f"arm {r.get('arm')}: {r.get('reason')}"})
    seen: set[str] = set()
    return [d for d in out if not (d["member"] in seen or seen.add(d["member"]))]


def diag_reference(diag_json: Path, configs: list[dict[str, Any]], models_dir: Path | None = None) -> tuple[float | None, float | None, str]:
    """``(oof_fit, forward_fit, label)`` the diag ratio is taken against: the uploaded pickle's single member when its
    manifest (found in ``models/`` by the sha256 the diag sidecars record) bundles exactly one member of this run
    (e.g. ``bundle --members lgbm_everest``), else the blend row."""
    blend_fit = next((c.get("fit_mean") for c in configs if c["kind"] == "blend"), None)
    blend_fwd = next((c.get("forward_fit") for c in configs if c["kind"] == "blend"), None)
    try:
        path = Path(diag_json)
        sides = [path]
        if path.name.endswith(".upload.json"):
            sides.append(path.with_name(path.name[: -len(".upload.json")] + ".meta.json"))
        shas: set[str] = set()
        for side in sides:
            rec = _load_json(side)
            if isinstance(rec, dict):
                shas |= {str(rec[k]) for k in ("built_from_pkl_sha256", "pkl_sha256") if rec.get(k)}
        mdir = Path(models_dir) if models_dir is not None else Path(__file__).resolve().parents[1] / "models"
        manifests = sorted(mdir.glob("*.manifest.json")) if shas and mdir.is_dir() else []
        for mf in manifests:
            man = _load_json(mf)
            if not isinstance(man, dict) or str(man.get("sha256")) not in shas:
                continue
            names = [str(m.get("name")) for m in man.get("members") or [] if isinstance(m, dict)]
            row = next((c for c in configs if c.get("kind") == "member" and len(names) == 1 and c.get("name") == names[0]), None)
            if row is not None:
                return row.get("fit_mean"), row.get("forward_fit"), f"member {names[0]} ({mf.name})"
    except Exception as exc:  # noqa: BLE001 - a reporter never fails on a side computation
        log.warning("diag reference lookup skipped: %s", exc)
    return blend_fit, blend_fwd, "blend row"


def _diag_with_reference(diag_json: Path, configs: list[dict[str, Any]], models_dir: Path | None) -> dict[str, Any]:
    fit, fwd, label = diag_reference(diag_json, configs, models_dir)
    d = diag_doc(diag_json, fit, fwd)
    if d.get("present"):
        d["oof_reference"] = label
    return d


def diag_doc(path: Path, oof_fit: float | None, forward_fit: float | None) -> dict[str, Any]:
    """Validation-diagnostics panel from ``<file>.upload.json`` (written by ``stealth.diag --upload``) + the OOF ratio.

    Informational only (PLAN C13): the number is never used to tune p_b / p_c or to select members.
    """
    path = Path(path)
    rec = _load_json(path)
    if not isinstance(rec, dict):
        return {"present": False, "path": path.name, "note": f"{path.name} missing or unreadable (no upload recorded)"}
    panel = rec.get("panel")
    if not isinstance(panel, dict) and isinstance(rec.get("result"), dict):
        from stealth import diag as diagmod  # lazy: only for an upload json written before the panel was stored

        panel = diagmod.panel_from_result(rec["result"])
    panel = panel if isinstance(panel, dict) else {}
    fit = panel.get("fit20")

    def _ratio(ref: float | None) -> float | None:
        if isinstance(fit, (int, float)) and isinstance(ref, (int, float)) and ref:
            return float(fit) / float(ref)
        return None

    return {"present": True, "path": path.name, "model_id": rec.get("model_id"), "upload_id": rec.get("upload_id"),
            "file": rec.get("file"), "rows": rec.get("rows"), "uploaded_at": rec.get("uploaded_at"),
            "result_at": rec.get("result_at"), "attached_pkl": rec.get("attached_pkl"),
            "built_from_pkl_sha256": rec.get("built_from_pkl_sha256"), "status": panel.get("status"),
            "panel": {k: v for k, v in panel.items() if k != "status"},
            "oof_fit": oof_fit, "forward_fit": forward_fit,
            "ratio_diag_vs_oof": _ratio(oof_fit), "ratio_diag_vs_forward": _ratio(forward_fit),
            "note": "informational only: the diag lane never tunes p_b/p_c or selects members"}


def _baseline_name(paired: dict[str, Any], members: dict[str, Any]) -> str | None:
    """The B side most member paired keys (``<member>_vs_<B>``) share; else ``lgbm_everest`` when it is a member."""
    counts: dict[str, int] = {}
    for key in paired:
        a, sep, b = key.partition("_vs_")
        if sep and a in members and b in members:
            counts[b] = counts.get(b, 0) + 1
    if counts:
        return max(counts, key=lambda k: counts[k])
    return "lgbm_everest" if "lgbm_everest" in members else None


def blend_vs_baseline(run_dir: Path, blend: dict[str, Any] | None, metrics: dict[str, Any] | None) -> tuple[str, dict] | None:
    """``("blend_vs_<B>", paired)`` recomputed from ``oof.parquet`` when metrics.json lacks a ``blend_vs_*`` row.

    Per-era fit20 of the reported blend z (blend.json weights; cross-fitted weights when NNLS was adopted, exactly as
    blend.py reports it) minus the baseline member's, 20-era paired block bootstrap (2000 draws, seed 0) + forward
    diff. Returns None when the row already exists or the inputs are missing; never raises.
    """
    paired = (metrics or {}).get("paired") or {}
    members = (metrics or {}).get("members") or {}
    if any(k.startswith("blend_vs_") for k in paired) or not blend or not isinstance(blend.get("weights"), dict):
        return None
    base = _baseline_name(paired, members)
    if base is None or not (run_dir / "oof.parquet").exists():
        return None
    try:
        import pandas as pd

        from stealth import blend as blendmod
        from stealth import cv as cvmod
        from stealth import metrics as metmod
        from stealth.data import PRIMARY_TARGET

        w = {k: float(v) for k, v in blend["weights"].items() if isinstance(v, (int, float)) and float(v) > 0}
        cols = sorted({f"m__{n}" for n in [*w, base]})

        def fits(frame: pd.DataFrame, crossfit: bool) -> tuple[pd.Series, pd.Series]:
            Z = blendmod.gaussianise_per_era(frame[cols].rename(columns=lambda c: c[3:]), frame["era"])
            if crossfit:
                z, _ = blendmod.crossfit_nnls(Z[list(w)], frame)
            else:
                z = blendmod.blend_scores(Z, w)
            y, era = frame[PRIMARY_TARGET], frame["era"]
            return metmod.per_era_fit(z, y, era), metmod.per_era_fit(frame[f"m__{base}"], y, era)

        oof = pd.read_parquet(run_dir / "oof.parquet", columns=["era", "fold", PRIMARY_TARGET, *cols])
        nnls = blend.get("rule") == "nnls"
        fb, fbase = fits(oof, nnls)
        res = cvmod.paired_block_bootstrap((fb - fbase).dropna(), block=20, n_boot=2000, seed=0).as_dict()
        res["fwd_diff"] = None
        fwd_path = run_dir / "forward.parquet"
        if fwd_path.exists() and not nnls:
            fwd = pd.read_parquet(fwd_path, columns=["era", PRIMARY_TARGET, *cols])
            gb, gbase = fits(fwd, False)
            res["fwd_diff"] = float(gb.mean() - gbase.mean())
        res["source"] = ("report.py: recomputed from oof.parquet with blend.json weights"
                         + (" (cross-fitted NNLS)" if nnls else " (equal weights)"))
        return f"blend_vs_{base}", res
    except Exception as exc:  # noqa: BLE001 - a reporter never fails on a side computation
        log.warning("blend_vs_baseline recompute skipped: %s", exc)
        return None


# --------------------------------------------------------------------------------------------- JSON


def build_metrics_json(run_id: str, handoff: str, slug: str, run_dir: Path, metrics: dict[str, Any] | None,
                       blend: dict[str, Any] | None, env: dict[str, Any], *, diag_json: Path | None = None,
                       recompute: bool = True, models_dir: Path | None = None) -> dict[str, Any]:
    """Assemble the ``NN-metrics.json`` document. Config names + metrics only; never hyperparameter values.

    ``diag_json`` adds the validation-diagnostics panel (``stealth.diag`` upload record); ``recompute`` allows the
    ``blend_vs_<baseline>`` paired row to be recomputed from ``oof.parquet`` when metrics.json lacks it.
    """
    status = (metrics or {}).get("status") or "failed"
    host_src = (metrics or {}).get("host") or {}
    host = {
        "os": host_src.get("os", env.get("os")),
        "cpu": host_src.get("cpu", env.get("cpu")),
        "ram_gb": host_src.get("ram_gb", env.get("ram_gb")),
        "gpu": host_src.get("gpu", env.get("gpu")),
        "driver": host_src.get("driver", env.get("driver")),
        "python": host_src.get("python", env.get("python")),
        "libs": host_src.get("libs", env.get("libs")),
        "tz": host_src.get("tz", env.get("tz")),
    }
    paired = dict((metrics or {}).get("paired") or {})
    recomputed = blend_vs_baseline(run_dir, blend, metrics) if recompute else None
    if recomputed is not None:
        paired[recomputed[0]] = recomputed[1]
    admitted = (blend or {}).get("admitted")
    weights = (blend or {}).get("weights") if isinstance((blend or {}).get("weights"), dict) else {}
    configs: list[dict[str, Any]] = []
    for name, m in ((metrics or {}).get("members") or {}).items():
        fit = m.get("fit") or {}
        configs.append({
            "arm": m.get("arm"),
            "admitted": (name in admitted) if isinstance(admitted, list) else None,
            "weight": weights.get(name),
            "name": name,
            "kind": "member",
            "fit_mean": fit.get("mean"),
            "fit_se": _se(fit),
            "fit_std": fit.get("std"),
            "sharpe": fit.get("sharpe"),
            "pct_pos": fit.get("pct_pos"),
            "frac_roll20_neg": fit.get("frac_roll20_neg"),
            "n_eras": fit.get("n_eras"),
            "ci90": fit.get("ci90"),
            "spearman_mean": m.get("spearman_mean"),
            "forward_fit": m.get("forward_fit"),
            "rho_b_val": m.get("corr_with_bench_val"),
            "rho_c": m.get("rho_c"),
            "regime": None,
            "n_iter": m.get("n_iter"),
            "suspect": m.get("suspect"),
            "paired_vs_baseline": _paired_for(name, paired),
        })
    blend_m = (metrics or {}).get("blend") or {}
    if blend_m or blend:
        # the blend row is the CHOSEN blend (cross-fitted when NNLS was adopted); M1 metrics.json only has "ew"
        rule = str((blend or {}).get("rule") or blend_m.get("rule") or "ew")
        chosen = blend_m.get("chosen") or ((blend or {}).get("blend") or {}).get("chosen") or blend_m.get("ew") or {}
        configs.append({
            "name": "blend_" + rule,
            "kind": "blend",
            "fit_mean": chosen.get("mean"),
            "fit_se": _se(chosen),
            "fit_std": chosen.get("std"),
            "sharpe": chosen.get("sharpe"),
            "pct_pos": chosen.get("pct_pos"),
            "frac_roll20_neg": chosen.get("frac_roll20_neg"),
            "n_eras": chosen.get("n_eras"),
            "ci90": chosen.get("ci90"),
            "forward_fit": blend_m.get("forward_fit"),
            "rho_b_val": blend_m.get("corr_with_bench_val"),
            "rho_c": blend_m.get("rho_c"),
            "regime": _headline_regime(blend),
            "chosen_is_crossfit": bool(blend_m.get("chosen_is_crossfit") or (blend or {}).get("chosen_is_crossfit")),
            "suspect": blend_m.get("suspect", (blend or {}).get("suspect")),
            "paired_vs_baseline": _paired_for("blend", paired),
        })
        ew = blend_m.get("ew")
        if rule != "ew" and isinstance(ew, dict):
            configs.append({"name": "blend_ew", "kind": "blend_context", "fit_mean": ew.get("mean"), "fit_se": _se(ew),
                            "fit_std": ew.get("std"), "sharpe": ew.get("sharpe"), "pct_pos": ew.get("pct_pos"),
                            "frac_roll20_neg": ew.get("frac_roll20_neg"), "n_eras": ew.get("n_eras"), "ci90": ew.get("ci90"),
                            "paired_vs_baseline": None})
    corr_summary, corr_matrix = member_corr_doc(blend)
    doc: dict[str, Any] = {
        "schema": "stealth.results/1",
        "handoff": handoff,
        "slug": slug,
        "run_id": run_id,
        "config_name": _config_name(run_dir, metrics),
        "git_sha": (metrics or {}).get("git_sha") or env.get("git_sha"),
        "status": status,
        "started_at": (metrics or {}).get("started_at"),
        "finished_at": (metrics or {}).get("finished_at"),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": host,
        "timings": (metrics or {}).get("timings") or {},
        "configs": configs,
        "paired": paired,
        "blend": {
            "rule": (blend or {}).get("rule"),
            "weights": (blend or {}).get("weights"),
            "p_b": (blend or {}).get("p_b"),
            "p_c": (blend or {}).get("p_c"),
            "p_source": ((blend or {}).get("neutralisation") or {}).get("p_source") or (blend or {}).get("p_source"),
            "paired_vs_ew": (blend or {}).get("paired_vs_ew"),
            "regime": (blend or {}).get("regime"),
            "admitted": admitted,
            "chosen_is_crossfit": (blend or {}).get("chosen_is_crossfit"),
        } if blend else None,
        "universe": _pick((metrics or {}).get("universe"), _UNIVERSE_KEYS),
        "knobs": knobs_doc(run_dir, metrics),
        "search": search_doc(run_dir, metrics),
        "ladder": ladder_doc(blend),
        "member_corr_max_pair": corr_summary,
        "member_corr": corr_matrix,
        "neutralisation": neutralisation_doc(blend),
        "final": final_doc(blend),
        "dropped": dropped_doc(metrics, blend),
        "diag": _diag_with_reference(diag_json, configs, models_dir) if diag_json is not None else None,
        "leak_check": (metrics or {}).get("leak_check"),
        "notes": list((metrics or {}).get("notes") or []),
        "artifacts_present": sorted(p.name for p in run_dir.iterdir()) if run_dir.exists() else [],
    }
    if recomputed is not None:
        doc["notes"].append(f"{recomputed[0]}: not in metrics.json; {recomputed[1]['source']}")
    if metrics is None:
        doc["notes"].append(f"metrics.json missing in {run_dir.relative_to(REPO) if run_dir.is_relative_to(REPO) else run_dir}")
    return doc


# --------------------------------------------------------------------------------------------- Markdown


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_none_"
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _env_section(doc: dict[str, Any], env: dict[str, Any]) -> str:
    h = doc["host"]
    libs = h.get("libs") or {}
    rows = [
        ["os", _fmt(h.get("os"))], ["cpu", _fmt(h.get("cpu"))], ["ram_gb", _fmt(h.get("ram_gb"), 1)],
        ["gpu", _fmt(h.get("gpu"))], ["driver", _fmt(h.get("driver"))], ["python", _fmt(h.get("python"))],
        ["timezone", _fmt(h.get("tz"))], ["git_sha", _fmt(doc.get("git_sha"))],
        ["core.hooksPath (reporting host)", _fmt(env.get("hooks_path")) or "UNSET"],
        ["pins (reporting host)", "OK" if env.get("pins_ok") else "MISMATCH/UNKNOWN"],
        ["libs", ", ".join(f"{k} {v}" for k, v in sorted(libs.items())) or "-"],
    ]
    return _md_table(["item", "value"], rows)


def _metrics_section(doc: dict[str, Any]) -> str:
    rows = []
    for c in doc["configs"]:
        p = c.get("paired_vs_baseline") or {}
        ci = c.get("ci90")
        ci_s = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if isinstance(ci, (list, tuple)) and len(ci) == 2 else "-"
        kind = c["kind"] + (" (cross-fitted)" if c.get("chosen_is_crossfit") else "")
        rows.append([
            c["name"], kind, _fmt(c.get("arm")), _fmt(c.get("weight"), 3), _fmt(c.get("fit_mean")), _fmt(c.get("fit_se")),
            _fmt(c.get("sharpe"), 2), _fmt(c.get("pct_pos"), 2), ci_s, _fmt(c.get("forward_fit")),
            _fmt(c.get("rho_b_val"), 3), _fmt(c.get("rho_c"), 3), _fmt(c.get("n_iter")), _fmt(c.get("suspect")),
            _fmt(p.get("diff")), _fmt(p.get("se")), _fmt(p.get("p_pos"), 2), _fmt(p.get("baseline")),
        ])
    return _md_table(
        ["config", "kind", "arm", "weight", "fit_mean", "fit_se", "sharpe", "pct_pos", "ci90", "forward_fit", "rho_b_val",
         "rho_c", "n_iter", "suspect", "paired diff_mean", "diff_se", "p_pos", "vs"], rows)


def _knobs_section(k: dict[str, Any]) -> str:
    rows = [[_fmt(c.get("key")), _fmt(c.get("dimension")), _fmt(c.get("fit_cv")), _fmt(c.get("fit_fwd")),
             _fmt(c.get("diff_fwd")), _fmt(c.get("diff_se_fwd")), _fmt(c.get("p_pos_fwd"), 2), _fmt(c.get("diff_cv_indep")),
             _fmt(c.get("holm_rank")), _fmt(c.get("holm_threshold"), 3), _fmt(c.get("pass"))] for c in k.get("candidates") or []]
    head = (f"chosen universe: `{k.get('chosen')}` (default `{k.get('default')}`; knob member {k.get('member')}; "
            f"Holm alpha {_fmt(k.get('alpha_total'), 2)}; status {k.get('status')})")
    return head + "\n\n" + _md_table(["candidate", "dimension", "fit_cv", "fit_fwd", "diff_fwd", "diff_se", "p_pos_fwd",
                                      "diff_cv_indep", "holm_rank", "holm_thr", "pass"], rows)


def _search_section(rows_in: list[dict[str, Any]]) -> str:
    rows = []
    for r in rows_in:
        pc = r.get("paired_cv") or {}
        rows.append([r["member"], _fmt(r.get("n_trials")), _fmt(r.get("topk_fit")), _fmt(r.get("base_params_fit")),
                     _fmt(r.get("gain")), _fmt(r.get("forward_gain")), _fmt(pc.get("p_pos"), 2), _fmt(r.get("adopted")),
                     _fmt(r.get("suspect")), scrub(str(r.get("adopt_reason") or "-"))])
    return _md_table(["member", "n_trials", "topk_fit", "base_fit", "gain", "forward_gain", "paired p_pos", "adopted",
                      "suspect", "reason"], rows)


def _ladder_section(rows_in: list[dict[str, Any]]) -> str:
    rows = [[_fmt(r.get("arm")), ", ".join(r.get("members") or []) or "-", _fmt(r.get("admitted")), _fmt(r.get("diff_mean")),
             _fmt(r.get("diff_se")), _fmt(r.get("p_pos"), 2), _fmt(r.get("fwd_diff")), _fmt(r.get("dsharpe"), 3),
             _fmt(r.get("p_pos_sharpe"), 2), _fmt(r.get("corr_with_incumbent"), 3), scrub(str(r.get("reason") or "-"))]
            for r in rows_in]
    return _md_table(["arm", "members", "admitted", "diff_mean", "diff_se", "p_pos", "fwd_diff", "dsharpe", "p_pos_sharpe",
                      "corr_inc", "reason"], rows)


def _corr_section(s: dict[str, Any]) -> str:
    mp = s.get("max_pair") or {}
    head = (f"max pair: {mp.get('a', '-')} / {mp.get('b', '-')} corr {_fmt(mp.get('corr'), 3)} "
            f"({s.get('n_members')} members; full matrix in NN-metrics.json `member_corr`)")
    rows = [[p["a"], p["b"], _fmt(p["corr"], 3)] for p in s.get("top_pairs") or []]
    return head + "\n\n" + _md_table(["member a", "member b", "corr"], rows)


def _neutral_section(n: dict[str, Any]) -> str:
    head = [f"chosen p_b = {_fmt(n.get('p_b'), 2)}, p_c = {_fmt(n.get('p_c'), 2)}; p_source = **{n.get('p_source')}**"
            f" (z: {n.get('z_source') or '-'})", f"reason: {scrub(str(n.get('reason') or '-'))}"]
    if n.get("would_choose"):
        head.append(f"simulation would choose: {n['would_choose']} (not used: select is false)")
    rows = [[_fmt(r.get("p_b"), 2), _fmt(r.get("p_c"), 2), _fmt(r.get("fit")), _fmt(r.get("score")),
             _fmt(r.get("score_sharpe"), 3), _fmt(r.get("W")), _fmt(r.get("eligible")), _fmt(r.get("in_tie_set"))]
            for r in n.get("table") or []]
    return "\n".join(f"- {h}" for h in head) + "\n\n" + _md_table(
        ["p_b", "p_c", "mix fit", "mix score", "score_sharpe", "W (worst regime)", "eligible", "tie set"], rows)


def _final_section(f: dict[str, Any]) -> str:
    oof = f.get("oof") or {}
    rows = [["admitted arms", ", ".join(f.get("admitted_arms") or []) or "-"], ["rule", _fmt(f.get("rule"))],
            ["p_b / p_c (p_source)", f"{_fmt(f.get('p_b'), 2)} / {_fmt(f.get('p_c'), 2)} ({f.get('p_source')})"],
            ["OOF fit20 mean (se)", f"{_fmt(oof.get('mean'))} ({_fmt(oof.get('fit_se'))})"],
            ["OOF sharpe / pct_pos", f"{_fmt(oof.get('sharpe'), 3)} / {_fmt(oof.get('pct_pos'), 2)}"],
            ["forward fit20", _fmt(f.get("forward_fit"))], ["central mix score", _fmt(f.get("score_mix_central"))],
            ["central score Sharpe", _fmt(f.get("score_sharpe_central"), 3)], ["W (worst regime score)", _fmt(f.get("W"))]]
    return _md_table(["item", "value"], rows)


def _dropped_section(rows_in: list[dict[str, Any]]) -> str:
    return _md_table(["member", "status", "detail"],
                     [[d["member"], _fmt(d.get("status")), scrub(str(d.get("detail") or "-"))] for d in rows_in])


def _diag_section(d: dict[str, Any]) -> str:
    if not d.get("present"):
        return f"_{d.get('note')}_"
    p = d.get("panel") or {}
    rows = [["model_id", _fmt(d.get("model_id"))], ["upload_id", _fmt(d.get("upload_id"))],
            ["file / rows", f"{d.get('file')} / {d.get('rows')}"], ["uploaded_at", _fmt(d.get("uploaded_at"))],
            ["status", _fmt(d.get("status"))], ["attached_pkl", _fmt(d.get("attached_pkl"))]]
    rows += [[k, _fmt(v)] for k, v in p.items()]
    rows += [[f"OOF fit20 ({d.get('oof_reference') or 'blend row'})", _fmt(d.get("oof_fit"))], ["diag fit20 / OOF fit20", _fmt(d.get("ratio_diag_vs_oof"), 3)],
             ["diag fit20 / forward fit20", _fmt(d.get("ratio_diag_vs_forward"), 3)]]
    return _md_table(["item", "value"], rows) + f"\n\n- {d.get('note')}"


def _paired_section(paired: dict[str, Any]) -> str:
    rows = []
    for key, v in paired.items():
        if not isinstance(v, dict):
            continue
        ci = v.get("ci90")
        ci_s = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if isinstance(ci, (list, tuple)) and len(ci) == 2 else "-"
        verdict = "-"
        if isinstance(v.get("p_pos"), (int, float)):
            verdict = "adopt A (p_pos >= 0.8)" if v["p_pos"] >= 0.8 else "tie — keep the simpler one"
        rows.append([key, _fmt(v.get("diff_mean")), _fmt(v.get("diff_se")), _fmt(v.get("p_pos"), 2), ci_s,
                     _fmt(v.get("n_eras")), verdict])
    return _md_table(["A_vs_B", "diff_mean", "diff_se", "p_pos", "ci90", "n_eras", "verdict"], rows)


def _regime_section(blend: dict[str, Any] | None) -> str:
    regime = (blend or {}).get("regime")
    if not isinstance(regime, dict) or not regime:
        return "_no blend.json / regime simulation present_"
    rows: list[list[str]] = []
    metric_keys: list[str] = []
    for mean_key, by_pb in regime.items():
        if mean_key == "meta" or not isinstance(by_pb, dict):
            continue
        for pb_key, cell in by_pb.items():
            if not isinstance(cell, dict):
                continue
            for reg in ("bpos", "bneg", "mix"):
                vals = cell.get(reg)
                if not isinstance(vals, dict):
                    continue
                if not metric_keys:
                    metric_keys = [k for k in vals if isinstance(vals[k], (int, float))]
                rows.append([str(mean_key), str(pb_key), reg] + [_fmt(vals.get(k)) for k in metric_keys])
    header = ["bench_fit_mean", "p_b", "regime"] + metric_keys
    lines = [_md_table(header, rows)]
    extra = []
    meta = regime.get("meta")
    if isinstance(meta, dict):
        extra.append("simulation: " + ", ".join(f"{k} {meta[k]}" for k in ("n_draws", "n_eras", "p_c", "bench_fit_sd")
                                                if k in meta))
    if blend:
        extra.append(f"chosen p_b = {_fmt(blend.get('p_b'), 2)}, p_c = {_fmt(blend.get('p_c'), 2)}, rule = {blend.get('rule')}")
        if isinstance(blend.get("weights"), dict):
            extra.append("weights: " + ", ".join(f"{k} {_fmt(v, 3)}" for k, v in blend["weights"].items()))
        pv = blend.get("paired_vs_ew")
        if isinstance(pv, dict):
            extra.append(f"paired NNLS vs EW: diff_mean {_fmt(pv.get('diff_mean'))}, diff_se {_fmt(pv.get('diff_se'))}, "
                         f"p_pos {_fmt(pv.get('p_pos'), 2)}")
    if extra:
        lines.append("")
        lines.extend(f"- {e}" for e in extra)
    return "\n".join(lines)


def _timings_section(timings: dict[str, Any]) -> str:
    rows = []
    if "total_s" in timings:
        rows.append(["TOTAL", _fmt(timings.get("total_s"), 1)])
    for name, s in sorted((timings.get("per_member_fit_s") or {}).items()):
        rows.append([name, _fmt(s, 1)])
    for key, val in timings.items():
        if key not in {"total_s", "per_member_fit_s"} and isinstance(val, (int, float)):
            rows.append([key, _fmt(val, 1)])
    return _md_table(["step / member", "wall seconds"], rows)


def render_markdown(doc: dict[str, Any], env: dict[str, Any], blend: dict[str, Any] | None, tail: list[str],
                    traceback_text: str | None) -> str:
    status = doc["status"]
    banner = "" if status == "ok" else f"\n> **STATUS: {status.upper()}** — see the traceback / log tail below.\n"
    suspects = [c["name"] for c in doc["configs"] if c.get("suspect")]
    if suspects:
        banner += ("\n> **SUSPECT — winner's curse or leak:** " + ", ".join(suspects)
                   + " exceed(s) the baseline by more than the suspect gain; do not celebrate.\n")
    if doc.get("leak_check") and str(doc["leak_check"]).upper() == "FAIL":
        banner += "\n> **LEAK CHECK FAILED** (OOF Spearman above the gate). The run is void.\n"
    parts = [
        f"# Results {doc['handoff']} — {doc['slug']}",
        "",
        f"- run_id: `{doc['run_id']}`  config: `{doc.get('config_name') or '-'}`  status: **{status}**",
        f"- git_sha: `{doc.get('git_sha') or '-'}`  started: {doc.get('started_at') or '-'}  finished: {doc.get('finished_at') or '-'}",
        f"- generated: {doc['generated_at']} by `python -m stealth.report`",
        banner,
        "## Environment",
        "",
        _env_section(doc, env),
        "",
        "## Metrics (per-era fit20 on OOF; fit_se = 20-era block-bootstrap SE; config names only, no hyperparameter values)",
        "",
        _metrics_section(doc),
        "",
        "## Paired block-bootstrap tests",
        "",
        _paired_section(doc.get("paired") or {}),
        "",
        "## Regime simulation (blend.json)",
        "",
        _regime_section(blend),
        "",
    ]
    # M2-M6 sections appear only when the run produced them (an M1 run's report keeps its old shape)
    optional = [
        ("Validation diagnostics (upload; informational only, never tunes p)", doc.get("diag"), _diag_section),
        ("Training universe (knobs, decided once on the forward block)", doc.get("knobs"), _knobs_section),
        ("Search adoption (top-k vs base cell; paired gate)", doc.get("search"), _search_section),
        ("Admission ladder (pre-registered order, one look per arm)", doc.get("ladder"), _ladder_section),
        ("Member correlation", doc.get("member_corr_max_pair"), _corr_section),
        ("Neutralisation constants (simulation, staking-aware)", doc.get("neutralisation"), _neutral_section),
        ("Final selection", doc.get("final"), _final_section),
        ("Dropped members", doc.get("dropped"), _dropped_section),
    ]
    for title, value, fn in optional:
        if value:
            parts += [f"## {title}", "", fn(value), ""]
    parts += [
        "## Wall times",
        "",
        _timings_section(doc.get("timings") or {}),
        "",
        "## Notes",
        "",
        "\n".join(f"- {scrub(str(n))}" for n in doc.get("notes") or []) or "_none_",
        "",
    ]
    if traceback_text:
        parts += ["## Traceback", "", "```text", traceback_text, "```", ""]
    parts += ["## run.log (last %d lines)" % len(tail), "", "```text", "\n".join(tail) if tail else "(run.log missing)",
              "```", ""]
    return "\n".join(parts)


# --------------------------------------------------------------------------------------------- write + git


def write_results(run_id: str, handoff: str, *, repo: Path = REPO, artifacts_dir: Path | None = None,
                  results_dir: Path | None = None, log_lines: int = 60, diag_json: Path | None = None,
                  recompute: bool = True) -> dict[str, Any]:
    """Write ``NN-results.md``, ``NN-metrics.json``, ``NN-log-tail.txt``.

    Returns ``{"paths": [Path, ...], "status": str, "slug": str, "doc": dict}``. Never raises on missing
    artifacts: a missing ``metrics.json`` yields status ``failed`` with whatever log tail exists. ``diag_json``
    (relative paths resolve against ``repo``) adds the validation-diagnostics panel.
    """
    if diag_json is not None and not Path(diag_json).is_absolute():
        diag_json = repo / diag_json
    artifacts_dir = artifacts_dir or (repo / "artifacts")
    results_dir = results_dir or (repo / "handoffs" / "results")
    run_dir = artifacts_dir / run_id
    metrics = _load_json(run_dir / "metrics.json")
    blend = _load_json(run_dir / "blend.json")
    env = selfcheck.collect_env(repo / "requirements-lock.txt", repo)
    slug = handoff_slug(handoff, repo / "handoffs", fallback=_config_name(run_dir, metrics) or run_id)
    doc = build_metrics_json(run_id, handoff, slug, run_dir, metrics, blend, env, diag_json=diag_json,
                             recompute=recompute, models_dir=repo / "models")
    tail = log_tail(run_dir / "run.log", log_lines)
    tb = extract_traceback(run_dir / "run.log") if doc["status"] != "ok" else None
    results_dir.mkdir(parents=True, exist_ok=True)
    md = results_dir / f"{handoff}-results.md"
    js = results_dir / f"{handoff}-metrics.json"
    lt = results_dir / f"{handoff}-log-tail.txt"
    md.write_text(scrub(render_markdown(doc, env, blend, tail, tb)), encoding="utf-8")
    js.write_text(scrub(json.dumps(doc, indent=2, default=str)) + "\n", encoding="utf-8")
    lt.write_text("\n".join(tail) + ("\n" if tail else ""), encoding="utf-8")
    log.info("wrote %s, %s, %s (status=%s)", md.name, js.name, lt.name, doc["status"])
    return {"paths": [md, js, lt], "status": doc["status"], "slug": slug, "doc": doc}


def _git(repo: Path, *args: str, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=timeout,
                          check=False)


def git_commit_push(repo: Path, rel_path: str, message: str, *, push: bool = True) -> str:
    """``git add <rel_path> && git commit -m <message> && git push``.

    Returns a short status string ('committed+pushed' | 'committed' | 'nothing-to-commit[+pushed]').
    Raises :class:`GitError` (stage 'add'/'commit'/'push') with scrubbed detail text.
    """
    add = _git(repo, "add", "--", rel_path)
    if add.returncode != 0:
        raise GitError("add", scrub(add.stderr.strip() or add.stdout.strip()))
    staged = _git(repo, "diff", "--cached", "--quiet", "--", rel_path)
    committed = False
    if staged.returncode == 1:
        commit = _git(repo, "commit", "-m", message, "--", rel_path)
        if commit.returncode != 0:
            raise GitError("commit", scrub(commit.stderr.strip() or commit.stdout.strip()))
        committed = True
        log.info("committed: %s", message)
    else:
        log.info("nothing new to commit under %s", rel_path)
    if not push:
        return "committed" if committed else "nothing-to-commit"
    pushed = _git(repo, "push", timeout=300.0)
    if pushed.returncode != 0:
        raise GitError("push", scrub(pushed.stderr.strip() or pushed.stdout.strip()))
    return ("committed" if committed else "nothing-to-commit") + "+pushed"


def main(argv: list[str] | None = None) -> int:
    """CLI entry. Exit 0 when files are written (and pushed when requested); 1 on commit/push failure."""
    parser = argparse.ArgumentParser(prog="python -m stealth.report", description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-id", required=True, help="artifacts/<run-id>")
    parser.add_argument("--handoff", required=True, help="two-digit handoff number, e.g. 02")
    parser.add_argument("--no-git", action="store_true", help="write files only; no commit, no push")
    parser.add_argument("--no-push", action="store_true", help="commit but do not push")
    parser.add_argument("--log-lines", type=int, default=60)
    parser.add_argument("--diag-json", type=Path, default=None,
                        help="<validation file>.upload.json from `stealth.diag --upload`: adds the diagnostics panel")
    parser.add_argument("--no-recompute", action="store_true",
                        help="never recompute the blend_vs_<baseline> paired row from oof.parquet")
    parser.add_argument("--repo", type=Path, default=REPO, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    res = write_results(args.run_id, args.handoff, repo=args.repo, log_lines=args.log_lines, diag_json=args.diag_json,
                        recompute=not args.no_recompute)
    for p in res["paths"]:
        sys.stdout.write(f"wrote {p}\n")
    sys.stdout.write(f"status: {res['status']}\n")
    if args.no_git:
        sys.stdout.write("git: skipped (--no-git)\n")
        return 0
    message = f"results: {args.handoff} {res['slug']}"
    try:
        outcome = git_commit_push(args.repo, "handoffs/results", message, push=not args.no_push)
    except GitError as exc:
        detail = scrub(str(exc))
        if exc.stage == "push":
            sys.stdout.write(f"{PUSH_FAILED_MSG}\n({detail})\n")
        else:
            sys.stdout.write(f"GIT {exc.stage.upper()} FAILED — results are saved locally in handoffs/results/; "
                             f"ask the human to commit and push.\n({detail})\n")
        return 1
    sys.stdout.write(f"git: {outcome}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
