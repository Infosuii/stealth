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

Fix-pass additions (docs/INTERFACES.md section 22): the ``status`` is decided by evidence, never by one module's own
status field: ``pipeline.json`` (a running / interrupted / failed pipeline is never ``ok``; the failing step's
scrubbed log tail and the attempts' per-step seconds are reported, so the first train's time survives a retrain),
``metrics.json`` status + leak check, and -- for every step the pipeline reports as ok -- the files behind it: the
manifest and its byte-bound ``pkl_check`` record (``passed`` AND ``sha256`` == the manifest's == the pickle on disk),
the diagnostics FILE's ``.meta.json`` (coverage, built from the bundled pickle; built, NOT uploaded) and
``audit.json``. A required item that failed makes the status ``failed``; one that is missing makes it ``incomplete``.
Sections ``pipeline``, ``evidence``, ``packaging``, ``diag_file``, ``audit`` (reserved eras, frozen model; technical
completion is reported apart from ``audit_win``) appear when present; development figures are labelled
:data:`SELECTION_LABEL`. Without ``pipeline.json`` (handoffs 00-03) the evidence is informational only.

Then ``git add handoffs/results && git commit -m "results: NN <slug>" && git push``. On push failure it prints
``PUSH FAILED — results are saved locally in handoffs/results/; ask the human to run git push`` and exits 1
(the files are already written). ``--no-git`` skips commit+push (sandbox testing); ``--no-push`` commits only.

Rules honoured here: config NAMES only (hyperparameter values stay in ``artifacts/``); the reporter never reads
``members/*/spec.json`` or ``config.yaml`` beyond its ``name``; every text that goes into a results file passes
through :func:`scrub` (API keys / tokens / tokenised remote URLs / presigned-URL queries and signatures are
redacted) and every captured log tail through :func:`scrub_params` (hyperparameter name = value pairs dropped); git
output is scrubbed before it is printed. Local absolute paths (the pipeline's pkl path) are reduced to file names.
Works when ``metrics.json`` is missing (status ``failed``).
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
    # bare signature / credential parameters of presigned storage URLs (S3 / GCS), wherever they appear
    re.compile(r"(?i)\b(X-(?:Amz|Goog)-(?:Signature|Credential|Security-Token))=[^\s&\"'<>]+"),
]
#: a URL query is redacted as a whole when it carries any signing / credential key (presigned download / upload URLs)
_URL_QUERY = re.compile(r"(https?://[^\s?#\"'<>]+)\?([^\s\"'<>]*)")
_SIGNED_QUERY_KEY = re.compile(r"(?i)(?:^|[&;])(?:x-amz-[a-z-]+|x-goog-[a-z-]+|signature|sig|token|expires|credential|"
                               r"awsaccesskeyid|security-token|policy|key-pair-id)=")
#: hyperparameter NAME = VALUE pairs that a captured log line may carry (search / trainer logs): name and value are both
#: dropped from every log tail that reaches a results file (hyperparameter values stay in artifacts/, PLAN C3)
_HP_NAMES = ("num_leaves", "learning_rate", "min_data_in_leaf", "feature_fraction", "lambda_l1", "lambda_l2",
             "bagging_fraction", "bagging_freq", "max_bin", "max_depth", "min_child_weight", "subsample",
             "colsample_bytree", "reg_lambda", "reg_alpha", "depth", "l2_leaf_reg", "iterations", "border_count", "alpha",
             "hidden", "dropout", "aux_weight", "lr", "weight_decay", "batch_size", "patience", "params", "parameters")
_HP_RE = re.compile(r"(?<![A-Za-z0-9_])['\"]?(?:%s)['\"]?\s*[:=]\s*(?:\{[^}]*\}|\[[^\]]*\]|\([^)]*\)|'[^']*'|\"[^\"]*\"|"
                    r"[^\s,;}{)]+)" % "|".join(_HP_NAMES))
_MEAN_PREFERRED = 0.009  # live-record benchmark FIT mean used for the headline regime row
SELECTION_LABEL = "selection estimates (reused for selection; optimistic)"
AUDIT_LABEL = "reserved audit eras: the frozen model scored once after selection; informational, never fed back"
PIPELINE_REQUIRED_STEPS = ("bundle", "pkl_check", "diag", "audit")  # steps whose files must back a pipeline 'ok'
_TAIL_LINES = 40


class GitError(RuntimeError):
    """A git step failed; ``.stage`` is 'add' | 'commit' | 'push'."""

    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(f"git {stage} failed: {detail}")
        self.stage = stage


# --------------------------------------------------------------------------------------------- helpers


def _scrub_query(m: re.Match[str]) -> str:
    return f"{m.group(1)}?[REDACTED]" if _SIGNED_QUERY_KEY.search(m.group(2)) else m.group(0)


def scrub(text: str) -> str:
    """Redact anything that looks like a credential, a tokenised URL or a presigned URL's query. Idempotent."""
    text = _URL_QUERY.sub(_scrub_query, text)
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def scrub_params(text: str) -> str:
    """Drop ``<hyperparameter name> = <value>`` pairs from captured log text (e.g. a trainer's 'hidden=[256, 64]')."""
    return _HP_RE.sub("[hyperparameter]", text)


_HOME_RE = re.compile(r"/(?:home|Users)/[^/\s'\"]+/")


def _clean_log_line(line: str) -> str:
    """A captured log line as it may appear in a public results file: credentials and presigned queries redacted,
    hyperparameter values dropped, home directories shortened to ``~/`` (no local user names)."""
    return _HOME_RE.sub("~/", scrub_params(scrub(line)))


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
    return [_clean_log_line(line) for line in lines[-n:]]


def extract_traceback(run_log: Path) -> str | None:
    """Last Python traceback in ``run_log`` (from the last 'Traceback (most recent call last)' to the end)."""
    if not run_log.exists():
        return None
    text = run_log.read_text(encoding="utf-8", errors="replace")
    idx = text.rfind("Traceback (most recent call last)")
    if idx < 0:
        return None
    return scrub_params(scrub(text[idx:].strip()[-6000:]))


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
_PAIRED_KEYS = ("diff_mean", "diff_se", "p_pos", "ci90", "n_eras", "p_value", "n_blocks", "inference_valid")
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


# --------------------------------------------------------------------------------------------- evidence
# What a results file may claim is decided by the artifacts, never by one module's own status field: the driver's
# pipeline.json (attempts / steps / exit codes / seconds), the manifest's byte-bound pkl_check record, the diagnostics
# file's meta sidecar and the audit's audit.json. Every helper copies an allow-list of keys.

_STEP_KEYS = ("name", "status", "exit_code", "seconds", "started_at", "finished_at", "note")
_AUDIT_SUMMARY_KEYS = _SUMMARY_KEYS
_AUDIT_SHARPE_KEYS = ("dsharpe", "se", "p_pos", "ci90", "n_eras", "sharpe_a", "sharpe_b", "n_blocks", "inference_valid")
_AUDIT_SEGMENT_KEYS = ("eras", "n_eras", "blend_fit", "baseline_fit", "diff")
_AUDIT_SENS_KEYS = ("block", "se_block", "p_pos", "p_value", "ci90", "n_blocks", "inference_valid")
_PKL_CHECK_KEYS = ("passed", "status", "at", "sha256", "python", "eras", "seconds_per_era_max", "isolation", "checker_libs_ok")


def _read_tail(path: Path, n: int = _TAIL_LINES) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [_clean_log_line(line) for line in lines[-n:]]


def _latest_step_status(pipe: dict[str, Any] | None, name: str) -> str | None:
    """Status of the most recent run of ``name`` over every attempt (None when it never ran)."""
    out = None
    for att in (pipe or {}).get("attempts") or []:
        for s in att.get("steps") or []:
            if isinstance(s, dict) and s.get("name") == name:
                out = s.get("status")
    return out


def pipeline_doc(run_dir: Path) -> dict[str, Any] | None:
    """``artifacts/<run>/pipeline.json`` (written by scripts/run_pipeline.sh): status, attempts, per-step seconds of
    EVERY attempt (so the first attempt's train time survives a retrain), the failing step with its scrubbed log tail,
    and the MLP step's tail when it was skipped. None when the run was not driven by the pipeline."""
    pj = _load_json(run_dir / "pipeline.json")
    if not isinstance(pj, dict):
        return None
    attempts: list[dict[str, Any]] = []
    step_seconds: dict[str, list[dict[str, Any]]] = {}
    for i, att in enumerate(pj.get("attempts") or [], 1):
        if not isinstance(att, dict):
            continue
        steps = [_pick(s, _STEP_KEYS) for s in att.get("steps") or [] if isinstance(s, dict)]
        for s in steps:
            step_seconds.setdefault(str(s.get("name")), []).append(
                {"attempt": i, "seconds": s.get("seconds"), "status": s.get("status")})
        attempts.append({"attempt": i, "from_step": att.get("from_step"), "status": att.get("status"),
                         "exit_code": att.get("exit_code"), "seconds": att.get("seconds"), "started_at": att.get("started_at"),
                         "finished_at": att.get("finished_at"), "failed_step": att.get("failed_step"), "mlp": att.get("mlp"),
                         "note": att.get("note"), "steps": steps, "report_step": _pick(att.get("report_step"), _STEP_KEYS)})
    status = pj.get("status")
    doc: dict[str, Any] = {
        "status": status, "exit_code": pj.get("exit_code"), "failed_step": pj.get("failed_step"),
        "last_step": pj.get("last_step"), "finished_at": pj.get("finished_at"), "mlp": pj.get("mlp"),
        "pkl": Path(str(pj["pkl"])).name if pj.get("pkl") else None,  # never an absolute local path
        "n_attempts": len(attempts), "attempts": attempts, "step_seconds": step_seconds,
        "total_seconds": sum(int(a["seconds"]) for a in attempts if isinstance(a.get("seconds"), (int, float))),
        "report_step": _pick(pj.get("report_step"), _STEP_KEYS), "failing_step": None, "mlp_step": None,
    }
    last = (pj.get("attempts") or [{}])[-1] if pj.get("attempts") else {}
    for s in reversed(last.get("steps") or []):
        if isinstance(s, dict) and s.get("status") in ("failed", "interrupted", "running"):
            doc["failing_step"] = {**(_pick(s, _STEP_KEYS) or {}), "log_tail": _read_tail(run_dir / str(s.get("log_file") or ""))
                                   if s.get("log_file") else []}
            break
    for s in reversed(last.get("steps") or []):
        if isinstance(s, dict) and s.get("name") == "mlp" and s.get("status") != "ok":
            doc["mlp_step"] = {**(_pick(s, _STEP_KEYS) or {}),
                               "log_tail": _read_tail(run_dir / str(s.get("log_file") or ""), 20) if s.get("log_file") else []}
            break
    return doc


def _sha256(path: Path) -> str | None:
    import hashlib

    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _manifest_path(run_id: str, models_dir: Path, pipe: dict[str, Any] | None) -> Path:
    pkl_name = (pipe or {}).get("pkl") or f"{run_id}.pkl"
    stem = Path(str(pkl_name)).stem
    return models_dir / f"{stem}.manifest.json"


def packaging_doc(run_id: str, models_dir: Path, pipe: dict[str, Any] | None) -> dict[str, Any]:
    """The bundle's manifest and its byte-bound pkl_check record: does the record name the bytes the manifest
    describes, and (when the pickle is on disk) are those the file's bytes? Never the pickle itself."""
    mpath = _manifest_path(run_id, models_dir, pipe)
    man = _load_json(mpath)
    if not isinstance(man, dict):
        return {"present": False, "manifest": mpath.name}
    rec = man.get("pkl_check") if isinstance(man.get("pkl_check"), dict) else {}
    pkl_path = mpath.with_name(str(man.get("pkl") or f"{mpath.name[: -len('.manifest.json')]}.pkl"))
    file_sha = _sha256(pkl_path) if pkl_path.exists() else None
    size = man.get("size_bytes")
    from stealth.pkl_check import check_evidence_problems

    return {
        "present": True, "manifest": mpath.name, "pkl": pkl_path.name, "sha256": man.get("sha256"),
        "file_present": pkl_path.exists(), "file_sha_matches": None if file_sha is None else file_sha == man.get("sha256"),
        "size_mb": round(float(size) / 1e6, 3) if isinstance(size, (int, float)) else None, "python": man.get("python"),
        "run_id": man.get("run_id"), "p_b": man.get("p_b"), "p_c": man.get("p_c"), "p_source": man.get("p_source"),
        "blend_rule": man.get("blend_rule"), "admitted_arms": man.get("admitted_arms"),
        "members": [str(m.get("name")) for m in man.get("members") or [] if isinstance(m, dict)],
        "pkl_check": {**(_pick(rec, _PKL_CHECK_KEYS) or {}), "problems": list(rec.get("problems") or [])[:5]} if rec else None,
        "pkl_check_bound": bool(rec.get("sha256")) and rec.get("sha256") == man.get("sha256") if rec else False,
        "pkl_check_evidence_problems": check_evidence_problems(rec, man.get("sha256")),
    }


def diag_file_doc(run_id: str, artifacts_dir: Path, packaging: dict[str, Any] | None) -> dict[str, Any]:
    """``artifacts/diag/<run>_validation.parquet.meta.json``: coverage of the diagnostics FILE (built, not uploaded) and
    whether it was built from the bundled pickle. An ``.upload.json`` next to it would mean an upload happened."""
    out_file = artifacts_dir / "diag" / f"{run_id}_validation.parquet"
    meta = _load_json(out_file.with_name(out_file.name + ".meta.json"))
    upload = _load_json(out_file.with_name(out_file.name + ".upload.json"))
    if not isinstance(meta, dict):
        return {"present": False, "file": out_file.name, "upload_recorded": isinstance(upload, dict)}
    full = (meta.get("eras_limit") is None and meta.get("rows") == meta.get("validation_rows")
            and meta.get("eras") == meta.get("validation_eras"))
    pk_sha = (packaging or {}).get("sha256")
    return {"present": True, "file": out_file.name, "file_present": out_file.exists(), "rows": meta.get("rows"),
            "eras": meta.get("eras"), "validation_rows": meta.get("validation_rows"), "validation_eras": meta.get("validation_eras"),
            "eras_limit": meta.get("eras_limit"), "with_bench": meta.get("with_bench"), "bench_used_eras": meta.get("bench_used_eras"),
            "full_coverage": bool(full), "built_from_bundled_pkl": None if not pk_sha else meta.get("pkl_sha256") == pk_sha,
            "upload_recorded": isinstance(upload, dict), "created_at": meta.get("created_at")}


def audit_doc(run_dir: Path) -> dict[str, Any] | None:
    """``audit.json`` (stealth.audit, after the frozen blend selection): the reserved-era comparison of the bundled blend
    with the unsearched baseline. Informational: it never fed back into any choice."""
    a = _load_json(run_dir / "audit.json")
    if not isinstance(a, dict):
        return {"present": False, "label": AUDIT_LABEL} if (run_dir / "audit.parquet").exists() else None
    drawdown = a.get("drawdown") if isinstance(a.get("drawdown"), dict) else {}
    stab = a.get("stability") if isinstance(a.get("stability"), dict) else {}
    scen = a.get("scenario") if isinstance(a.get("scenario"), dict) else None
    return {
        "present": True, "label": AUDIT_LABEL, "schema": a.get("schema"), "status": a.get("status"),
        "created_at": a.get("created_at"),
        "partition": _pick(a.get("partition"), ("train_end", "test_start", "eras", "n_eras", "n_rows")),
        "frozen": _pick(a.get("frozen"), ("rule", "members", "admitted_arms", "p_b", "p_c", "p_source")),
        "blend": _summary(a.get("blend")),
        "baseline": {**(_summary(a.get("baseline")) or {}), **(_pick(a.get("baseline"), ("name", "label")) or {})}
        if isinstance(a.get("baseline"), dict) else None,
        "paired_vs_baseline": _pick(a.get("paired_vs_baseline"), _PAIRED_KEYS),
        "sharpe_vs_baseline": _pick(a.get("sharpe_vs_baseline"), _AUDIT_SHARPE_KEYS),
        "drawdown": {"label": drawdown.get("label"),
                     "blend": _pick(drawdown.get("blend"), ("max_drawdown", "longest_underwater_eras")),
                     "baseline": _pick(drawdown.get("baseline"), ("max_drawdown", "longest_underwater_eras"))} if drawdown else None,
        "stability": {"segments": [_pick(s, _AUDIT_SEGMENT_KEYS) for s in stab.get("segments") or [] if isinstance(s, dict)],
                      **(_pick(stab, ("n_segments", "blend_positive_segments", "diff_positive_segments")) or {})} if stab else None,
        "bootstrap_sensitivity": [_pick(r, _AUDIT_SENS_KEYS) for r in a.get("bootstrap_sensitivity") or [] if isinstance(r, dict)],
        "scenario": None if scen is None else {**(_pick(scen, ("note", "p_b", "p_c", "bench_fit_mean")) or {}),
                                               "mix": _pick(scen.get("mix"), ("fit", "unq", "inov", "score", "score_sharpe"))},
        "verdict": _pick(a.get("verdict"), ("technical", "audit_win", "text")),
        "notes": [str(n) for n in a.get("notes") or []][:20],
    }


def evidence_items(pipe: dict[str, Any] | None, packaging: dict[str, Any], diag_file: dict[str, Any],
                   audit: dict[str, Any] | None) -> list[dict[str, Any]]:
    """``[{item, status: ok|fail|missing|warn|info, required, detail}]``. Without pipeline.json (handoffs 00-03 style)
    everything is informational; with it, the files behind every step the pipeline reports as ok are REQUIRED."""
    items: list[dict[str, Any]] = []

    def add(item: str, status: str, detail: str, required: bool) -> None:
        items.append({"item": item, "status": status, "required": required, "detail": detail})

    req = {s: pipe is not None and _latest_step_status(pipe, s) == "ok" for s in PIPELINE_REQUIRED_STEPS}
    need_bundle = req["bundle"] or req["pkl_check"]
    if not packaging.get("present"):
        if need_bundle:
            add("manifest", "missing", f"{packaging.get('manifest')} not found although bundle/pkl_check reported ok", True)
    else:
        sm = packaging.get("file_sha_matches")
        add("manifest", "ok" if sm is not False else "fail",
            f"{packaging['manifest']} sha256 {str(packaging.get('sha256'))[:12]}"
            + ("" if sm is None else " = pickle bytes" if sm else " != the pickle on disk"), need_bundle)
        rec = packaging.get("pkl_check") or {}
        if not rec:
            add("pkl_check", "missing" if req["pkl_check"] else "info", "manifest has no pkl_check record", req["pkl_check"])
        elif not rec.get("passed"):
            add("pkl_check", "fail", f"pkl_check {rec.get('status') or 'FAILED'}: " + "; ".join(rec.get("problems") or [])[:300],
                req["pkl_check"])
        elif not packaging.get("pkl_check_bound"):
            add("pkl_check", "fail", "the pass record does not name these bytes (sha256 absent or different): re-run pkl_check",
                req["pkl_check"])
        elif packaging.get("pkl_check_evidence_problems"):
            add("pkl_check", "fail", "; ".join(packaging["pkl_check_evidence_problems"]), req["pkl_check"])
        else:
            add("pkl_check", "ok", f"PASSED on these bytes (python {rec.get('python')}, {rec.get('eras')} eras, slowest call "
                f"{_fmt(rec.get('seconds_per_era_max'), 2)} s)", req["pkl_check"])
    if not diag_file.get("present"):
        if req["diag"]:
            add("diag_file", "missing", f"{diag_file.get('file')}.meta.json not found although diag reported ok", True)
    else:
        built = diag_file.get("built_from_bundled_pkl")
        if built is False:
            add("diag_file", "fail", "built from a different pickle than the manifest's", req["diag"])
        elif not diag_file.get("full_coverage"):
            add("diag_file", "warn", f"partial coverage: rows {diag_file.get('rows')}/{diag_file.get('validation_rows')}, eras "
                f"{diag_file.get('eras')}/{diag_file.get('validation_eras')} (--eras smoke)", req["diag"])
        else:
            add("diag_file", "ok", f"{diag_file.get('rows')} rows / {diag_file.get('eras')} eras, full coverage; built, not uploaded",
                req["diag"])
        if diag_file.get("upload_recorded"):
            add("diag_upload", "info", "an .upload.json exists next to the file: an upload was recorded for it", False)
    if audit is None:
        if req["audit"]:
            add("audit", "missing", "audit.json not found although the audit step reported ok", True)
    elif not audit.get("present"):
        add("audit", "missing", "audit.parquet exists but audit.json does not (run stealth.audit)", req["audit"] or pipe is not None)
    elif audit.get("status") != "ok":
        add("audit", "fail", f"audit.json status {audit.get('status')}", True)
    else:
        add("audit", "ok", "frozen model scored on the reserved eras (informational)", req["audit"])
    if pipe is not None:
        add("mlp", "info", str(pipe.get("mlp") or "not recorded"), False)
    return items


def overall_status(metrics: dict[str, Any] | None, pipe: dict[str, Any] | None,
                   evidence: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """``(status, reasons)``: the pipeline's own outcome first (a running / interrupted / failed pipeline is never 'ok'),
    then train/blend metrics status + leak check, then every REQUIRED evidence item (fail -> failed, missing ->
    incomplete)."""
    reasons: list[str] = []
    if pipe is not None and pipe.get("status") != "ok":
        ps = str(pipe.get("status") or "unknown")
        step = pipe.get("failed_step") or pipe.get("last_step") or "?"
        if ps == "running":
            reasons.append(f"pipeline.json says running at step {step}: still running, or the driver was killed before it "
                           "recorded an end (re-run the same pipeline command)")
        else:
            reasons.append(f"pipeline {ps} at step {step} (exit {pipe.get('exit_code')})")
        status = ps if ps in ("failed", "interrupted", "running") else "failed"
    else:
        status = "ok"
    m_status = (metrics or {}).get("status") or "failed"
    if metrics is None:
        reasons.append("metrics.json missing")
    elif m_status != "ok":
        reasons.append(f"metrics.json status {m_status}")
    if str((metrics or {}).get("leak_check") or "").upper() == "FAIL":
        reasons.append("leak check FAILED")
        m_status = "failed"
    if status == "ok" and m_status != "ok":
        status = m_status
    fails = [e for e in evidence if e["required"] and e["status"] == "fail"]
    missing = [e for e in evidence if e["required"] and e["status"] == "missing"]
    reasons += [f"{e['item']}: {e['detail']}" for e in fails + missing]
    if status == "ok" and fails:
        status = "failed"
    elif status == "ok" and missing:
        status = "incomplete"
    return status, reasons


# --------------------------------------------------------------------------------------------- JSON


def build_metrics_json(run_id: str, handoff: str, slug: str, run_dir: Path, metrics: dict[str, Any] | None,
                       blend: dict[str, Any] | None, env: dict[str, Any], *, diag_json: Path | None = None,
                       recompute: bool = True, models_dir: Path | None = None) -> dict[str, Any]:
    """Assemble the ``NN-metrics.json`` document. Config names + metrics only; never hyperparameter values.

    ``diag_json`` adds the validation-diagnostics panel (``stealth.diag`` upload record); ``recompute`` allows the
    ``blend_vs_<baseline>`` paired row to be recomputed from ``oof.parquet`` when metrics.json lacks it.
    ``status`` comes from :func:`overall_status` (pipeline.json, metrics.json + leak check, the files behind every step
    the pipeline reports as ok), never from one module's own status field alone.
    """
    models_dir = models_dir if models_dir is not None else run_dir.parent.parent / "models"
    pipe = pipeline_doc(run_dir)
    packaging = packaging_doc(run_id, models_dir, pipe)
    diag_file = diag_file_doc(run_id, run_dir.parent, packaging)
    audit = audit_doc(run_dir)
    evidence = evidence_items(pipe, packaging, diag_file, audit)
    status, status_reasons = overall_status(metrics, pipe, evidence)
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
        "status_reasons": status_reasons,
        "labels": {"development": SELECTION_LABEL, "audit": AUDIT_LABEL},
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
        "pipeline": pipe,
        "packaging": packaging,
        "diag_file": diag_file,
        "audit": audit,
        "evidence": evidence,
        "external_members": (metrics or {}).get("external_members"),
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
        if v.get("inference_valid") is False:
            verdict = "no valid inference (fewer than 2 bootstrap blocks)"
        elif isinstance(v.get("p_pos"), (int, float)):
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


def _timings_section(timings: dict[str, Any], pipe: dict[str, Any] | None = None) -> str:
    """metrics.json timings (the LAST train invocation: a retrain overwrites them) plus, when the pipeline drove the
    run, every attempt's per-step seconds from pipeline.json (the first train's time survives a retrain)."""
    rows = []
    for name, runs in ((pipe or {}).get("step_seconds") or {}).items():
        for r in runs:
            rows.append([f"pipeline step {name} (attempt {r.get('attempt')}, {r.get('status')})", _fmt(r.get("seconds"))])
    if pipe:
        rows.append(["pipeline total (all attempts)", _fmt(pipe.get("total_seconds"))])
    if "total_s" in timings:
        rows.append(["TOTAL (metrics.json, last train invocation)" if pipe else "TOTAL", _fmt(timings.get("total_s"), 1)])
    for name, s in sorted((timings.get("per_member_fit_s") or {}).items()):
        rows.append([name, _fmt(s, 1)])
    for key, val in timings.items():
        if key not in {"total_s", "per_member_fit_s"} and isinstance(val, (int, float)):
            rows.append([key, _fmt(val, 1)])
    return _md_table(["step / member", "wall seconds"], rows)


def _pipeline_section(pd_: dict[str, Any]) -> str:
    head = (f"pipeline status **{pd_.get('status')}** (exit {_fmt(pd_.get('exit_code'))}"
            + (f", failed step {pd_['failed_step']}" if pd_.get("failed_step") else "")
            + f"); {pd_.get('n_attempts')} attempt(s), {_fmt(pd_.get('total_seconds'))} s of driver wall time; "
            f"mlp: {pd_.get('mlp') or '-'}")
    rows = []
    for a in pd_.get("attempts") or []:
        rows.append([str(a.get("attempt")), "(attempt)", _fmt(a.get("from_step")), _fmt(a.get("status")), _fmt(a.get("exit_code")),
                     _fmt(a.get("seconds")), scrub(str(a.get("note") or "-"))])
        for st in a.get("steps") or []:
            rows.append([str(a.get("attempt")), _fmt(st.get("name")), "", _fmt(st.get("status")), _fmt(st.get("exit_code")),
                         _fmt(st.get("seconds")), scrub(str(st.get("note") or "-"))])
        rep = a.get("report_step")
        if rep:
            rows.append([str(a.get("attempt")), "report", "", _fmt(rep.get("status")), _fmt(rep.get("exit_code")),
                         _fmt(rep.get("seconds")), "report step (recorded apart from the pipeline result)"])
    out = head + "\n\n" + _md_table(["attempt", "step", "from", "status", "exit", "seconds", "note"], rows)
    for key, title in (("failing_step", "failing step"), ("mlp_step", "mlp step (optional member)")):
        st = pd_.get(key)
        if st and st.get("log_tail"):
            out += (f"\n\n{title} `{st.get('name')}` ({st.get('status')}, exit {st.get('exit_code')}), last "
                    f"{len(st['log_tail'])} log lines:\n\n```text\n" + "\n".join(st["log_tail"]) + "\n```")
    return out


def _evidence_section(items: list[dict[str, Any]]) -> str:
    return _md_table(["item", "status", "required", "detail"],
                     [[e["item"], e["status"].upper() if e["status"] in ("fail", "missing") else e["status"],
                       _fmt(e.get("required")), scrub(str(e.get("detail") or "-"))] for e in items])


def _packaging_section(p: dict[str, Any]) -> str:
    if not p.get("present"):
        return f"_no manifest ({p.get('manifest')})_"
    rec = p.get("pkl_check") or {}
    iso = rec.get("isolation") if isinstance(rec.get("isolation"), dict) else {}
    rows = [["manifest", _fmt(p.get("manifest"))], ["pickle file present / bytes = manifest sha256",
             f"{_fmt(p.get('file_present'))} / {_fmt(p.get('file_sha_matches'))}"],
            ["size MB", _fmt(p.get("size_mb"), 2)], ["built under python", _fmt(p.get("python"))],
            ["members", ", ".join(p.get("members") or []) or "-"], ["admitted arms", ", ".join(p.get("admitted_arms") or []) or "-"],
            ["rule / p_b / p_c (p_source)", f"{_fmt(p.get('blend_rule'))} / {_fmt(p.get('p_b'), 2)} / {_fmt(p.get('p_c'), 2)} "
             f"({p.get('p_source')})"],
            ["pkl_check", "PASSED" if rec.get("passed") else ("not run" if not rec else f"FAILED ({rec.get('status') or '-'})")],
            ["pkl_check bound to these bytes", _fmt(p.get("pkl_check_bound"))],
            ["checker python / eras / slowest call s", f"{_fmt(rec.get('python'))} / {_fmt(rec.get('eras'))} / "
             f"{_fmt(rec.get('seconds_per_era_max'), 2)}"],
            ["isolation (stealth / torch importable)", f"{_fmt(iso.get('stealth_importable'))} / {_fmt(iso.get('torch_importable'))}"]]
    rows += [["pkl_check problem", scrub(str(x))] for x in rec.get("problems") or []]
    return _md_table(["item", "value"], rows)


def _diag_file_section(d: dict[str, Any]) -> str:
    if not d.get("present"):
        return f"_not built ({d.get('file')} has no .meta.json)_"
    rows = [["file", _fmt(d.get("file"))], ["rows", f"{d.get('rows')} / {d.get('validation_rows')}"],
            ["eras", f"{d.get('eras')} / {d.get('validation_eras')}"], ["full coverage", _fmt(d.get("full_coverage"))],
            ["benchmark used (eras)", _fmt(d.get("bench_used_eras"))], ["built from the bundled pickle",
             _fmt(d.get("built_from_bundled_pkl"))], ["upload recorded", _fmt(d.get("upload_recorded"))]]
    return _md_table(["item", "value"], rows) + "\n\n- built only; uploading it is a separate USER MUST APPROVE handoff"


def _audit_section(a: dict[str, Any]) -> str:
    if not a.get("present"):
        return "_audit.parquet exists but audit.json is missing: run `python -m stealth.audit --run-id <run>`_"
    part = a.get("partition") or {}
    fr = a.get("frozen") or {}
    b, base = a.get("blend") or {}, a.get("baseline") or {}
    pv = a.get("paired_vs_baseline") or {}
    sh = a.get("sharpe_vs_baseline") or {}
    dd = a.get("drawdown") or {}
    ver = a.get("verdict") or {}
    eras = part.get("eras")
    head = [f"{AUDIT_LABEL}.",
            f"status **{a.get('status')}**; eras {eras[0] if isinstance(eras, list) and eras else '-'}.."
            f"{eras[-1] if isinstance(eras, list) and eras else '-'} ({_fmt(part.get('n_eras'))} eras, {_fmt(part.get('n_rows'))} rows; "
            f"development used eras < {_fmt(part.get('train_end'))})",
            f"frozen blend: rule {_fmt(fr.get('rule'))}, arms {', '.join(fr.get('admitted_arms') or []) or '-'}, "
            f"p_b {_fmt(fr.get('p_b'), 2)}, p_c {_fmt(fr.get('p_c'), 2)} ({fr.get('p_source')}); unneutralised FIT below",
            f"verdict: technical **{ver.get('technical') or '-'}**; audit win **{_fmt(ver.get('audit_win'))}** — "
            f"{scrub(str(ver.get('text') or '-'))}"]
    rows = [["fit20 mean (se)", f"{_fmt(b.get('mean'))} ({_fmt(b.get('fit_se'))})",
             f"{_fmt(base.get('mean'))} ({_fmt(base.get('fit_se'))})"],
            ["sharpe", _fmt(b.get("sharpe"), 3), _fmt(base.get("sharpe"), 3)],
            ["pct_pos", _fmt(b.get("pct_pos"), 2), _fmt(base.get("pct_pos"), 2)],
            ["frac 20-era means < 0", _fmt(b.get("frac_roll20_neg"), 2), _fmt(base.get("frac_roll20_neg"), 2)],
            ["max drawdown (cum. fit20)", _fmt((dd.get("blend") or {}).get("max_drawdown")),
             _fmt((dd.get("baseline") or {}).get("max_drawdown"))],
            ["longest underwater (eras)", _fmt((dd.get("blend") or {}).get("longest_underwater_eras")),
             _fmt((dd.get("baseline") or {}).get("longest_underwater_eras"))]]
    out = "\n".join(f"- {h}" for h in head) + "\n\n" + _md_table(["metric", "blend", f"baseline {base.get('name') or ''}"], rows)
    out += (f"\n\n- paired fit20 blend - baseline: diff {_fmt(pv.get('diff_mean'))} (se {_fmt(pv.get('diff_se'))}), "
            f"p_pos {_fmt(pv.get('p_pos'), 2)}, p_value {_fmt(pv.get('p_value'), 3)}, {_fmt(pv.get('n_blocks'))} blocks, "
            f"inference valid {_fmt(pv.get('inference_valid'))}"
            f"\n- paired Sharpe: dsharpe {_fmt(sh.get('dsharpe'), 3)} (se {_fmt(sh.get('se'), 3)}), p_pos {_fmt(sh.get('p_pos'), 2)}"
            f"\n- drawdown: {scrub(str(dd.get('label') or 'cumulative per-era fit20 drawdown (NOT platform P&L)'))}")
    stab = a.get("stability") or {}
    if stab.get("segments"):
        out += "\n\nchronological stability:\n\n" + _md_table(
            ["eras", "n_eras", "blend fit", "baseline fit", "diff"],
            [[f"{s['eras'][0]}..{s['eras'][-1]}" if isinstance(s.get("eras"), list) and s["eras"] else "-", _fmt(s.get("n_eras")),
              _fmt(s.get("blend_fit")), _fmt(s.get("baseline_fit")), _fmt(s.get("diff"))] for s in stab["segments"]])
    if a.get("bootstrap_sensitivity"):
        out += "\n\nbootstrap block-length sensitivity (paired diff):\n\n" + _md_table(
            ["block", "se", "p_pos", "p_value", "ci90", "blocks", "valid"],
            [[_fmt(r.get("block")), _fmt(r.get("se_block")), _fmt(r.get("p_pos"), 2), _fmt(r.get("p_value"), 3),
              f"[{_fmt(r['ci90'][0])}, {_fmt(r['ci90'][1])}]" if isinstance(r.get("ci90"), list) and len(r["ci90"]) == 2 else "-",
              _fmt(r.get("n_blocks")), _fmt(r.get("inference_valid"))] for r in a["bootstrap_sensitivity"]])
    sc = a.get("scenario")
    if sc and sc.get("mix"):
        m = sc["mix"]
        out += (f"\n\n- scenario (frozen constants only, simulated benchmark mean {_fmt(sc.get('bench_fit_mean'), 3)}): mix fit "
                f"{_fmt(m.get('fit'))}, unq {_fmt(m.get('unq'))}, inov {_fmt(m.get('inov'))}, score {_fmt(m.get('score'))}, "
                f"score Sharpe {_fmt(m.get('score_sharpe'), 3)}; {scrub(str(sc.get('note') or ''))}")
    for n in a.get("notes") or []:
        out += f"\n- note: {scrub(str(n))}"
    return out


def render_markdown(doc: dict[str, Any], env: dict[str, Any], blend: dict[str, Any] | None, tail: list[str],
                    traceback_text: str | None) -> str:
    status = doc["status"]
    banner = "" if status == "ok" else f"\n> **STATUS: {status.upper()}** — see the reasons and the traceback / log tail below.\n"
    for r in doc.get("status_reasons") or []:
        if status != "ok":
            banner += f"> - {scrub(str(r))}\n"
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
        f"## Metrics — development {SELECTION_LABEL} (per-era fit20 on OOF; fit_se = 20-era block-bootstrap SE; config names "
        "only, no hyperparameter values)",
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
        ("Pipeline (scripts/run_pipeline.sh attempts and steps)", doc.get("pipeline"), _pipeline_section),
        ("Evidence behind the status", doc.get("evidence"), _evidence_section),
        ("Audit — reserved eras, frozen model (informational; never fed back)", doc.get("audit"), _audit_section),
        ("Packaging evidence (manifest + pkl_check)", doc.get("packaging") if (doc.get("packaging") or {}).get("present") else None,
         _packaging_section),
        ("Diagnostics FILE (built, NOT uploaded)", doc.get("diag_file") if (doc.get("diag_file") or {}).get("present") else None,
         _diag_file_section),
        ("Validation diagnostics (upload; informational only, never tunes p)", doc.get("diag"), _diag_section),
        (f"Training universe (knobs, decided once on the development forward block) — {SELECTION_LABEL}", doc.get("knobs"),
         _knobs_section),
        (f"Search adoption (top-k vs base cell; paired gate) — {SELECTION_LABEL}", doc.get("search"), _search_section),
        (f"Admission ladder (pre-registered order, one look per arm) — {SELECTION_LABEL}", doc.get("ladder"), _ladder_section),
        ("Member correlation", doc.get("member_corr_max_pair"), _corr_section),
        ("Neutralisation constants (simulation, staking-aware)", doc.get("neutralisation"), _neutral_section),
        (f"Final selection — {SELECTION_LABEL}", doc.get("final"), _final_section),
        ("Dropped members", doc.get("dropped"), _dropped_section),
    ]
    for title, value, fn in optional:
        if value:
            parts += [f"## {title}", "", fn(value), ""]
    parts += [
        "## Wall times",
        "",
        _timings_section(doc.get("timings") or {}, doc.get("pipeline")),
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
    tb = extract_traceback(run_dir / "run.log") if doc["status"] not in ("ok", "incomplete") else None
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
