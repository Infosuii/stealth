"""stealth.report + stealth.selfcheck: offline, against synthetic artifacts in a throwaway git repo."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from stealth import report, selfcheck

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------------- selfcheck


def test_selfcheck_pins_ok_in_train_venv() -> None:
    rows = selfcheck.check_pins(REPO / "requirements-lock.txt")
    names = {r["package"] for r in rows}
    assert {"lightgbm", "xgboost", "pandas", "numpy", "everesteer-api", "cloudpickle"} <= names
    bad = [r for r in rows if r["status"] != "OK"]
    assert not bad, bad


def test_selfcheck_detects_mismatch(tmp_path: Path) -> None:
    lock = tmp_path / "lock.txt"
    lock.write_text("numpy==0.0.1\n# comment\nnot-installed-pkg==1.0\n")
    rows = selfcheck.check_pins(lock)
    assert {r["package"]: r["status"] for r in rows} == {"numpy": "MISMATCH", "not-installed-pkg": "MISSING"}
    assert selfcheck.main(["--lock", str(lock)]) == 1


def test_selfcheck_main_and_env(capsys: pytest.CaptureFixture[str]) -> None:
    assert selfcheck.main([]) == 0
    out = capsys.readouterr().out
    for word in ("python", "core.hooksPath", "EIQ_API_KEY", "PINS OK", "timezone"):
        assert word in out
    env = selfcheck.collect_env()
    assert env["python"].startswith("3.11")
    assert isinstance(env["cpu_count"], int) and env["cpu_count"] >= 1
    assert env["ram_gb"] is None or env["ram_gb"] > 0
    assert "lightgbm" in env["libs"]
    # the key VALUE must never appear anywhere in the output
    key = os.environ.get("EIQ_API_KEY")
    if key:
        assert key not in out and key not in json.dumps(env)


# --------------------------------------------------------------------------------------------- report


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


@pytest.fixture()
def fake_repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    (r / "handoffs" / "results").mkdir(parents=True)
    (r / "handoffs" / "02-test-run.md").write_text("# handoff\n")
    (r / "requirements-lock.txt").write_text((REPO / "requirements-lock.txt").read_text())
    assert _git(r, "init", "-q").returncode == 0
    _git(r, "config", "user.email", "t@example.invalid")
    _git(r, "config", "user.name", "t")
    (r / "README.md").write_text("x\n")
    _git(r, "add", "README.md")
    _git(r, "commit", "-q", "-m", "init")
    return r


def _make_run(repo: Path, run_id: str, *, status: str = "ok", with_blend: bool = True) -> Path:
    run = repo / "artifacts" / run_id
    (run / "members" / "lgbm_everest").mkdir(parents=True)
    (run / "config.yaml").write_text("name: test\nmembers:\n  - {name: lgbm_everest, params: {num_leaves: 8}}\n")
    summ = {"mean": 0.0261, "std": 0.045, "sharpe": 0.58, "pct_pos": 0.71, "frac_roll20_neg": 0.1,
            "ci90": [0.0195, 0.0327], "n_eras": 3700}
    metrics = {
        "run_id": run_id, "config_name": "test", "git_sha": "a" * 40, "status": status,
        "started_at": "2026-11-14T10:00:00+00:00", "finished_at": "2026-11-14T10:40:00+00:00",
        "host": {"os": "Linux-6.1", "cpu": "fake cpu", "ram_gb": 32.0, "python": "3.11.9",
                 "libs": {"lightgbm": "4.7.0"}},
        "timings": {"total_s": 2400.0, "per_member_fit_s": {"lgbm_everest": 60.0, "cat_everest_l2_30": 110.0}},
        "members": {
            "lgbm_everest": {"fit": summ, "spearman_mean": 0.118, "forward_fit": 0.021,
                             "corr_with_bench_val": 0.55, "n_iter": 350, "suspect": False},
            "ridge_everest": {"fit": {**summ, "mean": 0.0240}, "spearman_mean": 0.11, "forward_fit": 0.019,
                              "corr_with_bench_val": 0.5, "n_iter": None, "suspect": False},
        },
        "blend": {"ew": {**summ, "mean": 0.0275}, "forward_fit": 0.022},
        "paired": {"blend_vs_lgbm_everest": {"diff_mean": 0.0014, "diff_se": 0.0015, "p_pos": 0.83,
                                             "ci90": [-0.001, 0.004], "n_eras": 3700}},
        "leak_check": "pass", "notes": ["cat fit 110 s <= 120 s cap: keep"],
    }
    (run / "metrics.json").write_text(json.dumps(metrics))
    if with_blend:
        cell = {"fit": 0.026, "unq": 0.004, "inov": 0.003, "score": 0.033, "score_se": 0.001}
        regime = {f"{m:g}": {f"p_b={p:g}": {"bpos": cell, "bneg": {**cell, "unq": 0.0}, "mix": cell}
                             for p in (0.0, 0.25, 0.5, 0.75)} for m in (0.0, 0.009, 0.02)}
        regime["meta"] = {"p_c": 0.25, "n_draws": 500, "n_eras": 3700, "bench_fit_sd": 0.035,
                          "bench": {"0.009": {"realised_fit_mean": 0.0091}}, "note": "proxy"}
        (run / "blend.json").write_text(json.dumps({
            "weights": {"lgbm_everest": 0.6, "ridge_everest": 0.4}, "rule": "ew", "p_b": 0.5, "p_c": 0.25,
            "core_proxy_cols": ["feature_A"], "regime": regime,
            "paired_vs_ew": {"diff_mean": 0.0, "diff_se": 0.001, "p_pos": 0.5}}))
    lines = [f"2026-11-14 10:{i:02d} INFO progress line {i}" for i in range(80)]
    lines.append("EIQ_API_KEY=" + "s" * 24 + "  <- must be scrubbed")
    if status != "ok":
        lines += ["Traceback (most recent call last):", '  File "x.py", line 1, in <module>', "ValueError: boom"]
    (run / "run.log").write_text("\n".join(lines) + "\n")
    return run


def test_write_results_ok(fake_repo: Path) -> None:
    _make_run(fake_repo, "r1")
    res = report.write_results("r1", "02", repo=fake_repo)
    md, js, lt = res["paths"]
    assert res["status"] == "ok" and res["slug"] == "test-run"
    assert md.name == "02-results.md" and js.name == "02-metrics.json" and lt.name == "02-log-tail.txt"
    text = md.read_text()
    for s in ("## Environment", "## Metrics", "## Paired", "## Regime", "## Wall times", "run.log (last 60 lines)",
              "lgbm_everest", "0.0261", "0.0014", "0.83", "p_b=0.5", "bneg", "110.0", "fake cpu"):
        assert s in text, s
    assert "num_leaves" not in text and "num_leaves" not in js.read_text(), "hyperparameter values leaked"
    assert "s" * 24 not in text and "[REDACTED]" in text, "secret in log tail not scrubbed"
    assert len(lt.read_text().splitlines()) == 60
    doc = json.loads(js.read_text())
    assert doc["status"] == "ok" and doc["git_sha"] == "a" * 40 and doc["host"]["cpu"] == "fake cpu"
    names = {c["name"]: c for c in doc["configs"]}
    assert names["lgbm_everest"]["fit_mean"] == 0.0261 and names["lgbm_everest"]["rho_b_val"] == 0.55
    assert abs(names["lgbm_everest"]["fit_se"] - 0.045 / 3700**0.5) < 1e-9
    assert names["blend_ew"]["paired_vs_baseline"]["p_pos"] == 0.83
    assert names["blend_ew"]["regime"]["bneg"]["unq"] == 0.0
    assert names["blend_ew"]["regime"]["at"] == {"bench_fit_mean": "0.009", "p_b": "p_b=0.5"}
    assert "simulation: n_draws 500" in text and "| 0 | p_b=0 | bpos |" in text
    assert "| meta |" not in text
    assert "Traceback" not in text


def test_write_results_failed_run_has_traceback(fake_repo: Path) -> None:
    _make_run(fake_repo, "r2", status="failed", with_blend=False)
    res = report.write_results("r2", "02", repo=fake_repo)
    text = res["paths"][0].read_text()
    assert "STATUS: FAILED" in text and "## Traceback" in text and "ValueError: boom" in text
    assert "no blend.json" in text


def test_write_results_without_metrics_json(fake_repo: Path) -> None:
    run = fake_repo / "artifacts" / "r3"
    run.mkdir(parents=True)
    run.joinpath("run.log").write_text("started\nTraceback (most recent call last):\nKeyError: 'x'\n")
    res = report.write_results("r3", "05", repo=fake_repo)
    assert res["status"] == "failed" and res["slug"] == "r3"
    doc = json.loads(res["paths"][1].read_text())
    assert doc["configs"] == [] and any("metrics.json missing" in n for n in doc["notes"])
    assert "KeyError" in res["paths"][0].read_text()
    # even a run dir that does not exist yields files
    res2 = report.write_results("nope", "06", repo=fake_repo)
    assert res2["status"] == "failed" and all(p.exists() for p in res2["paths"])


def test_main_no_git_and_commit(fake_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _make_run(fake_repo, "r1")
    assert report.main(["--run-id", "r1", "--handoff", "02", "--no-git", "--repo", str(fake_repo)]) == 0
    assert "git: skipped" in capsys.readouterr().out
    assert report.main(["--run-id", "r1", "--handoff", "02", "--no-push", "--repo", str(fake_repo)]) == 0
    assert "git: committed" in capsys.readouterr().out
    last = _git(fake_repo, "log", "-1", "--pretty=%s").stdout.strip()
    assert last == "results: 02 test-run"
    # re-running with nothing new is fine
    assert report.main(["--run-id", "r1", "--handoff", "02", "--no-push", "--repo", str(fake_repo)]) == 0
    assert "nothing-to-commit" in capsys.readouterr().out


def test_main_push_failure_is_loud(fake_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _make_run(fake_repo, "r1")
    _git(fake_repo, "remote", "add", "origin", "https://x-access-token:" + "t" * 20 + "@github.invalid/nope/nope.git")
    _git(fake_repo, "config", "branch.master.remote", "origin")
    _git(fake_repo, "config", "branch.main.remote", "origin")
    env_backup = os.environ.get("GIT_TERMINAL_PROMPT")
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    try:
        rc = report.main(["--run-id", "r1", "--handoff", "02", "--repo", str(fake_repo)])
    finally:
        if env_backup is None:
            del os.environ["GIT_TERMINAL_PROMPT"]
        else:
            os.environ["GIT_TERMINAL_PROMPT"] = env_backup
    out = capsys.readouterr().out
    assert rc == 1
    assert report.PUSH_FAILED_MSG in out
    assert "t" * 20 not in out, "token from the remote URL leaked into the output"
    assert (fake_repo / "handoffs" / "results" / "02-results.md").exists()
    assert _git(fake_repo, "log", "-1", "--pretty=%s").stdout.strip() == "results: 02 test-run"


def test_scrub() -> None:
    s = ("EIQ_API_KEY" + "=abcdefgh12345 " + "X-API-" + "Key: zzzzzzzzzzzz " + "ghp_" + "A" * 30
         + " https://u:p@" + "github.com/x fine=ok")
    out = report.scrub(s)
    assert "abcdefgh" not in out and "zzzz" not in out and "A" * 30 not in out and "u:p@" not in out
    assert "fine=ok" in out
