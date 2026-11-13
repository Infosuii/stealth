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
    # block SE, not std/sqrt(n): the synthetic summary has no se_block, so the ci90 half-width / 1.645 is used
    assert abs(names["lgbm_everest"]["fit_se"] - (0.0327 - 0.0195) / (2 * 1.645)) < 1e-9
    assert names["lgbm_everest"]["fit_se"] > 2 * 0.045 / 3700**0.5
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


# --------------------------------------------------------------------------------------------- M2-M6 report sections

_HASH = "c0ffee" * 6  # a search-cell hash: must never reach the results files


def _make_m2_run(repo: Path, run_id: str) -> Path:
    """A fake full/minifull-style run: knobs, search adoption, ladder, correlation, neutralisation, final, drops."""
    run = _make_run(repo, run_id)
    (run / "config.yaml").write_text("name: minifull\nmembers:\n  - {name: lgbm_everest, params: {learning_rate: 0.03}}\n")
    summ = {"mean": 0.0261, "std": 0.045, "sharpe": 0.58, "pct_pos": 0.71, "ci90": [0.0195, 0.0327], "n_eras": 3700,
            "se_block": 0.0033}
    metrics = json.loads((run / "metrics.json").read_text())
    metrics["config_name"] = "minifull"
    metrics["paired"] = {"ridge_everest_vs_lgbm_everest": {"diff_mean": -0.002, "diff_se": 0.001, "p_pos": 0.02}}
    metrics["members"]["lgbm_everest"].update({"arm": "lgbm", "search_gain": {
        "topk_fit": 0.0262, "base_params_fit": 0.0255, "gain": 0.0007, "margin": 0.004, "suspect": False,
        "topk_forward_fit": 0.020, "base_params_forward_fit": 0.021, "forward_gain": -0.001, "forward_confirms": False,
        "topk_cells": [_HASH], "base_cell": _HASH, "base_cell_in_topk": False, "n_trials": 3,
        "paired_cv": {"diff_mean": 0.0007, "diff_se": 0.0006, "p_pos": 0.86, "ci90": [-0.0002, 0.0017], "n_eras": 3700},
        "adopted": False, "adopt_reason": "forward gain -0.0010 < 0 -> base cell", "universe": "e5000-none-d0"}})
    metrics["members"]["ridge_everest"]["arm"] = "ridge"
    metrics["universe"] = {"key": "e5000-filter80-d0", "era_start": 5000, "rows_rule": "filter80", "time_decay": False,
                           "rows_low_weight": 0.5}
    metrics["knobs"] = {"status": "done", "member": "lgbm_everest", "alpha_total": 0.2, "default": "e5000-none-d0",
                        "chosen": "e5000-filter80-d0", "winners": {"rows_rule": "e5000-filter80-d0"}, "candidates": [
                            {"key": "e5000-none-d0", "dimension": None, "fit_cv": 0.025, "fit_fwd": 0.020, "pass": None},
                            {"key": "e5000-filter80-d0", "dimension": "rows_rule", "fit_cv": 0.026, "fit_fwd": 0.023,
                             "diff_fwd": 0.003, "diff_se_fwd": 0.001, "p_pos_fwd": 0.97, "diff_cv_indep": 0.001,
                             "holm_rank": 1, "holm_threshold": 0.04, "pass": True}]}
    metrics["dropped"] = [{"member": "cat_everest_tiny", "status": "dropped_timing", "first_fit_seconds": 150.0},
                          {"member": "mlp_everest", "status": "pending_external", "reason": "spec.json missing"}]
    metrics["blend"] = {"ew": {**summ, "mean": 0.0270}, "chosen": {**summ, "mean": 0.0280}, "rule": "nnls",
                        "forward_fit": 0.022, "chosen_is_crossfit": True, "suspect": False}
    (run / "metrics.json").write_text(json.dumps(metrics))
    blend = json.loads((run / "blend.json").read_text())
    table = [{"p_b": pb, "p_c": pc, "fit": 0.026 - pb / 100, "score": 0.04 + pb / 100, "score_sharpe": 0.5 + pb / 10,
              "W": 0.03, "eligible": pb <= 0.5, "in_tie_set": pb == 0.5} for pb in (0.0, 0.25, 0.5, 0.75) for pc in (0.0, 0.25)]
    blend.update({
        "rule": "nnls", "chosen_is_crossfit": True, "p_b": 0.5, "p_c": 0.0, "p_source": "simulation",
        "weights": {"lgbm_everest": 0.7, "ridge_everest": 0.3, "xgb_everest": 0.0},
        "admitted": ["lgbm_everest", "ridge_everest"],
        "ladder": [{"arm": "xgb", "members": ["xgb_everest"], "admitted": False, "reason": "REJECT: p_pos 0.40 < 0.8",
                    "diff_mean": -0.0001, "diff_se": 0.0004, "p_pos": 0.40, "fwd_diff": 0.001, "dsharpe": 0.01,
                    "p_pos_sharpe": 0.6, "corr_with_incumbent": 0.96, "dropped_redundant": []},
                   {"arm": "ridge", "members": ["ridge_everest"], "admitted": True, "reason": "ADMIT: sharpe route",
                    "diff_mean": 0.0004, "diff_se": 0.0005, "p_pos": 0.7, "fwd_diff": 0.0005, "dsharpe": 0.06,
                    "p_pos_sharpe": 0.85, "corr_with_incumbent": 0.80, "dropped_redundant": ["lgbm_everest_bag1"]}],
        "redundant": {"lgbm_everest_bag1": {"other": "lgbm_everest", "corr": 0.97}},
        "member_corr": {"names": ["lgbm_everest", "ridge_everest", "xgb_everest"],
                        "matrix": [[1.0, 0.81, 0.962], [0.81, 1.0, 0.79], [0.962, 0.79, 1.0]]},
        "neutralisation": {"p_b": 0.5, "p_c": 0.0, "p_source": "simulation", "z_source": "crossfit", "criterion": "rule text",
                           "reason": "7/8 eligible -> simplest", "table": table},
        "final": {"admitted_arms": ["lgbm", "ridge"], "rule": "nnls", "p_b": 0.5, "p_c": 0.0, "p_source": "simulation",
                  "oof": {**summ, "mean": 0.0280}, "forward_fit": 0.022, "score_mix_central": 0.041,
                  "score_sharpe_central": 0.55, "W": 0.03}})
    (run / "blend.json").write_text(json.dumps(blend))
    (run / "search").mkdir()
    (run / "search" / "lgbm_everest.json").write_text(json.dumps([{"params": {"learning_rate": 0.03}, "value": 0.02}]))
    (run / "search" / "xgb_everest.gain.json").write_text(json.dumps(
        {"gain": 0.0002, "forward_gain": 0.0001, "topk_cells": [_HASH], "adopted": True, "adopt_reason": "gate passed",
         "paired_cv": {"diff_mean": 0.0002, "diff_se": 0.0001, "p_pos": 0.9}}))
    return run


def test_write_results_m2_sections_and_diag(fake_repo: Path) -> None:
    _make_m2_run(fake_repo, "m2")
    diag_dir = fake_repo / "artifacts" / "diag"
    diag_dir.mkdir(parents=True)
    (diag_dir / "m2_validation.parquet.upload.json").write_text(json.dumps({
        "model_id": "slot-x", "upload_id": "up-9", "file": "m2_validation.parquet", "rows": 125931, "uploaded_at": "t",
        "attached_pkl": False, "accept": {"presigned": "https://u:p" + "@example.invalid/x"},
        "result": {"status": "done", "fit20": 0.021, "unq": 0.004, "inov": 0.015, "sharpe_ratio": 0.45}}))
    res = report.write_results("m2", "04", repo=fake_repo, diag_json=Path("artifacts/diag/m2_validation.parquet.upload.json"))
    md, js, _ = res["paths"]
    text, jtext = md.read_text(), js.read_text()
    doc = json.loads(jtext)
    names = {c["name"]: c for c in doc["configs"]}
    assert names["blend_nnls"]["fit_mean"] == 0.0280 and names["blend_nnls"]["chosen_is_crossfit"] is True
    assert names["blend_ew"]["kind"] == "blend_context" and names["blend_ew"]["fit_mean"] == 0.0270
    assert names["lgbm_everest"]["arm"] == "lgbm" and names["lgbm_everest"]["admitted"] is True
    assert names["lgbm_everest"]["weight"] == 0.7
    assert doc["universe"]["key"] == "e5000-filter80-d0"
    assert doc["knobs"]["chosen"] == "e5000-filter80-d0" and doc["knobs"]["candidates"][1]["pass"] is True
    search = {r["member"]: r for r in doc["search"]}
    assert search["lgbm_everest"]["adopted"] is False and search["lgbm_everest"]["paired_cv"]["p_pos"] == 0.86
    assert search["xgb_everest"]["adopted"] is True  # from search/xgb_everest.gain.json
    assert [r["arm"] for r in doc["ladder"]] == ["xgb", "ridge"]
    mp = doc["member_corr_max_pair"]["max_pair"]
    assert (mp["a"], mp["b"]) == ("lgbm_everest", "xgb_everest") and abs(mp["corr"] - 0.962) < 1e-12
    assert doc["member_corr"]["names"][0] == "lgbm_everest"
    assert doc["neutralisation"]["p_source"] == "simulation" and len(doc["neutralisation"]["table"]) == 8
    assert doc["blend"]["p_source"] == "simulation"
    assert doc["final"]["admitted_arms"] == ["lgbm", "ridge"] and doc["final"]["oof"]["fit_se"] == 0.0033
    dropped = {d["member"]: d["status"] for d in doc["dropped"]}
    assert dropped == {"cat_everest_tiny": "dropped_timing", "mlp_everest": "pending_external",
                       "lgbm_everest_bag1": "redundant", "xgb_everest": "not_admitted"}
    d = doc["diag"]
    assert d["present"] and d["upload_id"] == "up-9" and d["panel"]["fit20"] == 0.021 and d["panel"]["sharpe"] == 0.45
    assert abs(d["ratio_diag_vs_oof"] - 0.021 / 0.0280) < 1e-12 and abs(d["ratio_diag_vs_forward"] - 0.021 / 0.022) < 1e-12
    for title in ("## Validation diagnostics", "## Training universe", "## Search adoption", "## Admission ladder",
                  "## Member correlation", "## Neutralisation constants", "## Final selection", "## Dropped members"):
        assert title in text, title
    assert "p_source = **simulation**" in text and "lgbm_everest / xgb_everest corr 0.962" in text
    assert "(cross-fitted)" in text
    # never: hyperparameter values, cell hashes, server presigned URLs / credentials
    for bad in ("learning_rate", "num_leaves", _HASH, "u:p@"):
        assert bad not in text and bad not in jtext, bad
    assert "presigned" not in jtext  # only the allow-listed upload fields are copied


def test_m1_report_has_no_m2_sections_and_missing_diag_json(fake_repo: Path) -> None:
    _make_run(fake_repo, "r1")
    res = report.write_results("r1", "03", repo=fake_repo, diag_json=fake_repo / "artifacts" / "diag" / "nope.upload.json")
    text = res["paths"][0].read_text()
    for title in ("## Training universe", "## Search adoption", "## Admission ladder", "## Neutralisation constants"):
        assert title not in text
    assert res["doc"]["diag"]["present"] is False and "no upload recorded" in text


def _write_synthetic_oof(run: Path, n_eras: int = 80, n_rows: int = 40) -> None:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    era = np.repeat(np.arange(5000, 5000 + n_eras), n_rows).astype("int32")
    y = rng.integers(0, 5, size=era.size) / 4.0
    a = y + rng.normal(0, 3.0, size=era.size)
    b = y + rng.normal(0, 3.0, size=era.size)
    idx = pd.Index([f"id{i}" for i in range(era.size)], name="id")
    df = pd.DataFrame({"era": era, "fold": (np.arange(era.size) * 3 // era.size).astype("int8"),
                       "target_everest": y.astype("float32"), "m__a": a.astype("float32"), "m__b": b.astype("float32")},
                      index=idx)
    df.to_parquet(run / "oof.parquet")
    df.assign(fold=np.int8(-1)).iloc[: 20 * n_rows].to_parquet(run / "forward.parquet")


def test_blend_vs_baseline_recomputed_when_missing(fake_repo: Path) -> None:
    run = fake_repo / "artifacts" / "rb"
    run.mkdir(parents=True)
    summ = {"mean": 0.02, "std": 0.04, "sharpe": 0.5, "pct_pos": 0.7, "ci90": [0.01, 0.03], "n_eras": 80}
    (run / "metrics.json").write_text(json.dumps({
        "run_id": "rb", "config_name": "test", "status": "ok",
        "members": {"a": {"fit": summ}, "b": {"fit": summ}},
        "paired": {"b_vs_a": {"diff_mean": 0.0, "diff_se": 0.001, "p_pos": 0.5}, "blend_nnls_vs_ew": {"p_pos": 0.4}},
        "blend": {"ew": summ, "forward_fit": 0.02}}))
    (run / "blend.json").write_text(json.dumps({"rule": "ew", "weights": {"a": 0.5, "b": 0.5}, "p_b": 0.5, "p_c": 0.25}))
    _write_synthetic_oof(run)
    doc = report.write_results("rb", "02", repo=fake_repo)["doc"]
    row = doc["paired"]["blend_vs_a"]
    assert row["n_eras"] == 80 and 0.0 <= row["p_pos"] <= 1.0 and row["fwd_diff"] is not None
    assert "recomputed from oof.parquet" in row["source"]
    blend_row = next(c for c in doc["configs"] if c["kind"] == "blend")
    assert blend_row["paired_vs_baseline"]["baseline"] == "a" and blend_row["paired_vs_baseline"]["p_pos"] == row["p_pos"]
    assert any("blend_vs_a" in n for n in doc["notes"])
    # the blend of two noisy copies beats one of them on average
    assert row["diff_mean"] > 0
    # opt-out, and an existing blend_vs_* row is never overwritten
    assert "blend_vs_a" not in report.write_results("rb", "02", repo=fake_repo, recompute=False)["doc"]["paired"]
    m = json.loads((run / "metrics.json").read_text())
    m["paired"]["blend_vs_a"] = {"diff_mean": 9.0, "diff_se": 1.0, "p_pos": 1.0}
    (run / "metrics.json").write_text(json.dumps(m))
    assert report.write_results("rb", "02", repo=fake_repo)["doc"]["paired"]["blend_vs_a"]["diff_mean"] == 9.0


def test_main_diag_json_flag(fake_repo: Path) -> None:
    _make_run(fake_repo, "r1")
    up = fake_repo / "x.upload.json"
    up.write_text(json.dumps({"upload_id": "u1", "panel": {"status": "done", "fit20": 0.02}}))
    assert report.main(["--run-id", "r1", "--handoff", "03", "--no-git", "--repo", str(fake_repo), "--diag-json", str(up)]) == 0
    doc = json.loads((fake_repo / "handoffs" / "results" / "03-metrics.json").read_text())
    assert doc["diag"]["upload_id"] == "u1" and abs(doc["diag"]["ratio_diag_vs_oof"] - 0.02 / 0.0275) < 1e-12


def test_diag_ratio_uses_the_uploaded_single_member(fake_repo: Path) -> None:
    """A diag upload of a ``bundle --members lgbm_everest`` pickle is compared with that member, not the blend row."""
    _make_run(fake_repo, "r1")
    (fake_repo / "models").mkdir(exist_ok=True)
    sha = "ab" * 32
    (fake_repo / "models" / "r1_lgbm.manifest.json").write_text(json.dumps({"sha256": sha, "members": [{"name": "lgbm_everest"}]}))
    up = fake_repo / "artifacts" / "diag" / "r1_lgbm_validation.parquet.upload.json"
    up.parent.mkdir(parents=True, exist_ok=True)
    up.write_text(json.dumps({"upload_id": "u2", "built_from_pkl_sha256": sha, "panel": {"status": "done", "fit20": 0.02}}))
    doc = report.write_results("r1", "03", repo=fake_repo, diag_json=up)["doc"]
    member_fit = json.loads((fake_repo / "artifacts" / "r1" / "metrics.json").read_text())["members"]["lgbm_everest"]["fit"]["mean"]
    assert doc["diag"]["oof_reference"].startswith("member lgbm_everest")
    assert abs(doc["diag"]["ratio_diag_vs_oof"] - 0.02 / member_fit) < 1e-12
    # an unknown pickle sha keeps the blend row as the reference
    up.write_text(json.dumps({"upload_id": "u2", "built_from_pkl_sha256": "cd" * 32, "panel": {"status": "done", "fit20": 0.02}}))
    doc = report.write_results("r1", "03", repo=fake_repo, diag_json=up)["doc"]
    assert doc["diag"]["oof_reference"] == "blend row" and abs(doc["diag"]["ratio_diag_vs_oof"] - 0.02 / 0.0275) < 1e-12

