"""Results reporter: ``artifacts/<run>/metrics.json`` (+ ``blend.json``, ``run.log``) -> ``handoffs/results/NN-*``.

CLI contract (docs/INTERFACES.md section 10)::

    python -m stealth.report --run-id R --handoff NN [--no-git] [--no-push] [--log-lines 60]

Writes three files under ``handoffs/results/``:

* ``NN-results.md``   env table, metrics table (incl. paired diff_mean/diff_se/p_pos), regime table, wall times,
                      last 60 lines of run.log, traceback when the run failed
* ``NN-metrics.json`` machine-readable schema (docs/PLAN.md "Handoff protocol" item 7): run_id, git_sha, status,
                      host{os,cpu,ram_gb,gpu,driver,python,libs,tz}, timings, configs[]{name, fit_mean, fit_se, sharpe,
                      forward_fit, rho_b_val, rho_c, regime{bpos,bneg,mix}, ci90, paired_vs_baseline{diff,se,p_pos}}
* ``NN-log-tail.txt`` the log tail on its own

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

log = logging.getLogger(__name__)

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
    if not summary:
        return None
    std, n = summary.get("std"), summary.get("n_eras")
    if isinstance(std, (int, float)) and isinstance(n, (int, float)) and n > 0:
        return float(std) / math.sqrt(float(n))
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


# --------------------------------------------------------------------------------------------- JSON


def build_metrics_json(run_id: str, handoff: str, slug: str, run_dir: Path, metrics: dict[str, Any] | None,
                       blend: dict[str, Any] | None, env: dict[str, Any]) -> dict[str, Any]:
    """Assemble the ``NN-metrics.json`` document. Config names + metrics only; never hyperparameter values."""
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
    paired = (metrics or {}).get("paired") or {}
    configs: list[dict[str, Any]] = []
    for name, m in ((metrics or {}).get("members") or {}).items():
        fit = m.get("fit") or {}
        configs.append({
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
        ew = blend_m.get("ew") or {}
        configs.append({
            "name": "blend_" + str((blend or {}).get("rule") or "ew"),
            "kind": "blend",
            "fit_mean": ew.get("mean"),
            "fit_se": _se(ew),
            "fit_std": ew.get("std"),
            "sharpe": ew.get("sharpe"),
            "pct_pos": ew.get("pct_pos"),
            "frac_roll20_neg": ew.get("frac_roll20_neg"),
            "n_eras": ew.get("n_eras"),
            "ci90": ew.get("ci90"),
            "forward_fit": blend_m.get("forward_fit"),
            "rho_b_val": blend_m.get("corr_with_bench_val"),
            "rho_c": blend_m.get("rho_c"),
            "regime": _headline_regime(blend),
            "paired_vs_baseline": _paired_for("blend", paired),
        })
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
            "paired_vs_ew": (blend or {}).get("paired_vs_ew"),
            "regime": (blend or {}).get("regime"),
        } if blend else None,
        "leak_check": (metrics or {}).get("leak_check"),
        "notes": list((metrics or {}).get("notes") or []),
        "artifacts_present": sorted(p.name for p in run_dir.iterdir()) if run_dir.exists() else [],
    }
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
        rows.append([
            c["name"], c["kind"], _fmt(c.get("fit_mean")), _fmt(c.get("fit_se")), _fmt(c.get("sharpe"), 2),
            _fmt(c.get("pct_pos"), 2), ci_s, _fmt(c.get("forward_fit")), _fmt(c.get("rho_b_val"), 3),
            _fmt(c.get("n_iter")), _fmt(c.get("suspect")),
            _fmt(p.get("diff")), _fmt(p.get("se")), _fmt(p.get("p_pos"), 2), _fmt(p.get("baseline")),
        ])
    return _md_table(
        ["config", "kind", "fit_mean", "fit_se", "sharpe", "pct_pos", "ci90", "forward_fit", "rho_b_val", "n_iter",
         "suspect", "paired diff_mean", "diff_se", "p_pos", "vs"], rows)


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
                   + " exceed the baseline by more than the suspect gain; do not celebrate.\n")
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
        "## Metrics (per-era fit20 on OOF; config names only, no hyperparameter values)",
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
                  results_dir: Path | None = None, log_lines: int = 60) -> dict[str, Any]:
    """Write ``NN-results.md``, ``NN-metrics.json``, ``NN-log-tail.txt``.

    Returns ``{"paths": [Path, ...], "status": str, "slug": str, "doc": dict}``. Never raises on missing
    artifacts: a missing ``metrics.json`` yields status ``failed`` with whatever log tail exists.
    """
    artifacts_dir = artifacts_dir or (repo / "artifacts")
    results_dir = results_dir or (repo / "handoffs" / "results")
    run_dir = artifacts_dir / run_id
    metrics = _load_json(run_dir / "metrics.json")
    blend = _load_json(run_dir / "blend.json")
    env = selfcheck.collect_env(repo / "requirements-lock.txt", repo)
    slug = handoff_slug(handoff, repo / "handoffs", fallback=_config_name(run_dir, metrics) or run_id)
    doc = build_metrics_json(run_id, handoff, slug, run_dir, metrics, blend, env)
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
    parser.add_argument("--repo", type=Path, default=REPO, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    res = write_results(args.run_id, args.handoff, repo=args.repo, log_lines=args.log_lines)
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
