"""stealth.report against the fix-pass evidence (handoff review findings 1, 4): pipeline.json status and step tails,
the manifest's byte-bound pkl_check record, the diagnostics FILE meta, audit.json, retained per-attempt timings,
selection-estimate labels and the signed-URL / hyperparameter scrubbing. Synthetic artifacts only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from stealth import report
from tests.test_report_selfcheck import _make_run, fake_repo  # noqa: F401  (fixture re-export)

SHA = "ab" * 32
SIGNED = ("https://bucket.s3.amazonaws.com/futures_train.parquet?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential="
          + "AKIA" + "Z" * 16 + "%2F20261114&X-Amz-Date=20261114T000000Z&X-Amz-Expires=3600&X-Amz-Signature=" + "f" * 64)


def _pipeline(run: Path, *, status: str = "ok", steps: list[tuple[str, str, int]] | None = None,
              attempts: list[dict] | None = None, report_step: dict | None = None, mlp: str = "ok") -> None:
    steps = steps if steps is not None else [("train", "ok", 0), ("mlp", "ok", 0), ("retrain", "ok", 0), ("blend", "ok", 0),
                                             ("audit", "ok", 0), ("bundle", "ok", 0), ("pkl_check", "ok", 0), ("diag", "ok", 0)]
    (run / "pipeline_steps").mkdir(exist_ok=True)
    sts = []
    for i, (name, st, rc) in enumerate(steps):
        lf = f"pipeline_steps/2026-{i}-{name}.log"
        (run / lf).write_text(f"line of {name}\n" * 50 + f"last line of {name}\n")
        sts.append({"name": name, "status": st, "exit_code": rc, "seconds": 10 * (i + 1), "log_file": lf})
    att = attempts if attempts is not None else [{"started_at": "t0", "from_step": "train", "status": status, "seconds": 400,
                                                  "steps": sts, "mlp": mlp, "report_step": report_step}]
    failed = next((s["name"] for s in sts if s["status"] in ("failed", "interrupted")), None)
    doc = {"schema": "stealth.pipeline/1", "run_id": run.name, "config": "full.yaml",
           "pkl": f"/home/someone/private/path/models/{run.name}.pkl", "status": status,
           "exit_code": 0 if status == "ok" else (1 if status == "failed" else None), "failed_step": failed,
           "last_step": sts[-1]["name"] if sts else None, "mlp": mlp, "attempts": att, "report_step": report_step}
    (run / "pipeline.json").write_text(json.dumps(doc))


def _manifest(repo: Path, run_id: str, *, passed: bool = True, check_sha: str | None = SHA, pkl_bytes: bytes | None = None) -> None:
    (repo / "models").mkdir(exist_ok=True)
    sha = SHA if pkl_bytes is None else hashlib.sha256(pkl_bytes).hexdigest()
    if pkl_bytes is not None:
        (repo / "models" / f"{run_id}.pkl").write_bytes(pkl_bytes)
        check_sha = sha if check_sha == SHA else check_sha
    rec = {"passed": passed, "at": "t", "python": "3.11.2", "eras": 50, "seconds_per_era_max": 0.41,
           "isolation": {"stealth_importable": False, "torch_importable": False}, "problems": [] if passed else ["boom"]}
    if check_sha is not None:
        rec["sha256"] = check_sha
    (repo / "models" / f"{run_id}.manifest.json").write_text(json.dumps({
        "pkl": f"{run_id}.pkl", "sha256": sha, "size_bytes": 2_500_000, "python": "3.11.2", "run_id": run_id,
        "p_b": 0.25, "p_c": 0.0, "p_source": "simulation", "blend_rule": "ew", "admitted_arms": ["lgbm", "aux"],
        "members": [{"name": "lgbm_everest"}, {"name": "ridge_everest"}], "weights": {"lgbm_everest": 1.0}, "pkl_check": rec}))


def _diag_meta(repo: Path, run_id: str, *, rows: int = 125931, eras_limit: int | None = None, pkl_sha: str = SHA) -> None:
    d = repo / "artifacts" / "diag"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{run_id}_validation.parquet").write_bytes(b"x")
    (d / f"{run_id}_validation.parquet.meta.json").write_text(json.dumps({
        "file": f"{run_id}_validation.parquet", "pkl_sha256": pkl_sha, "rows": rows, "eras": 1392 if rows == 125931 else 5,
        "eras_limit": eras_limit, "with_bench": True, "bench_used_eras": 1392, "validation_rows": 125931,
        "validation_eras": 1392, "created_at": "t"}))


def _audit(run: Path, *, status: str = "ok", win: bool | None = False) -> None:
    summ = {"mean": 0.0195, "se_block": 0.0041, "std": 0.04, "sharpe": 0.49, "pct_pos": 0.68, "frac_roll20_neg": 0.12,
            "ci90": [0.013, 0.026], "n_eras": 580}
    (run / "audit.parquet").write_bytes(b"x")
    (run / "audit.json").write_text(json.dumps({
        "schema": "stealth.audit/1", "run_id": run.name, "status": status, "created_at": "t", "run_fingerprint": "f" * 16,
        "blend_sha256": "c" * 64, "partition": {"train_end": 5925, "test_start": 5945, "eras": [5945, 6524], "n_eras": 580,
                                                "n_rows": 41000},
        "frozen": {"rule": "ew", "members": ["lgbm_everest"], "admitted_arms": ["lgbm"], "p_b": 0.25, "p_c": 0.0,
                   "p_source": "simulation", "weights": {"lgbm_everest": 1.0}},
        "blend": summ, "baseline": {"name": "lgbm_everest", "label": "unsearched base params, default universe",
                                    **{**summ, "mean": 0.0190}},
        "paired_vs_baseline": {"diff_mean": 0.0005, "diff_se": 0.0009, "p_pos": 0.71, "p_value": 0.29, "ci90": [-0.001, 0.002],
                               "n_eras": 580, "n_blocks": 29, "inference_valid": True},
        "sharpe_vs_baseline": {"dsharpe": 0.01, "se": 0.03, "p_pos": 0.6, "ci90": [-0.04, 0.06], "n_eras": 580, "sharpe_a": 0.49,
                               "sharpe_b": 0.48, "n_blocks": 29, "inference_valid": True},
        "drawdown": {"label": "cumulative per-era fit20 drawdown (NOT platform P&L)",
                     "blend": {"max_drawdown": -0.21, "longest_underwater_eras": 61},
                     "baseline": {"max_drawdown": -0.24, "longest_underwater_eras": 70}},
        "stability": {"segments": [{"eras": [5945, 6234], "n_eras": 290, "blend_fit": 0.02, "baseline_fit": 0.019, "diff": 0.001},
                                   {"eras": [6235, 6524], "n_eras": 290, "blend_fit": 0.019, "baseline_fit": 0.019, "diff": 0.0}],
                      "n_segments": 2, "blend_positive_segments": 2, "diff_positive_segments": 1},
        "bootstrap_sensitivity": [{"block": b, "se_block": 0.0009, "p_pos": 0.7, "p_value": 0.3, "ci90": [-0.001, 0.002],
                                   "n_blocks": 580 // b, "inference_valid": True} for b in (10, 20, 40)],
        "scenario": {"note": "frozen constants only", "p_b": 0.25, "p_c": 0.0, "bench_fit_mean": 0.009,
                     "mix": {"fit": 0.019, "unq": 0.004, "inov": 0.012, "score": 0.035, "score_sharpe": 0.5}},
        "verdict": {"technical": "complete", "audit_win": win, "text": "tie on the reserved eras: keep the baseline in mind"},
        "notes": ["reported after the frozen selection"]}))


def _full_ok(repo: Path, run_id: str) -> Path:
    run = _make_run(repo, run_id)
    _pipeline(run)
    _manifest(repo, run_id)
    _diag_meta(repo, run_id)
    _audit(run)
    return run


def test_all_evidence_ok_and_sections(fake_repo: Path) -> None:  # noqa: F811
    _full_ok(fake_repo, "f1")
    res = report.write_results("f1", "04", repo=fake_repo)
    md, js, _ = res["paths"]
    text, jtext = md.read_text(), js.read_text()
    doc = json.loads(jtext)
    assert res["status"] == "ok", doc["status_reasons"]
    assert {e["item"]: e["status"] for e in doc["evidence"]} == {"manifest": "ok", "pkl_check": "ok", "diag_file": "ok",
                                                                 "audit": "ok", "mlp": "info"}
    assert doc["pipeline"]["pkl"] == "f1.pkl" and "/home/someone" not in jtext + text  # never a local absolute path
    assert doc["packaging"]["pkl_check_bound"] is True and doc["diag_file"]["full_coverage"] is True
    assert doc["audit"]["verdict"]["technical"] == "complete" and doc["audit"]["verdict"]["audit_win"] is False
    assert "weights" not in doc["audit"]["frozen"]  # allow-listed keys only
    for title in ("## Pipeline", "## Evidence behind the status", "## Audit — reserved eras", "## Packaging evidence",
                  "## Diagnostics FILE (built, NOT uploaded)", report.SELECTION_LABEL):
        assert title in text, title
    assert "NOT platform P&L" in text and "audit win **no**" in text and "technical **complete**" in text
    assert "bootstrap block-length sensitivity" in text and "chronological stability" in text
    assert doc["labels"]["development"] == report.SELECTION_LABEL and "STATUS:" not in text


def test_failed_pipeline_step_tail_is_scrubbed(fake_repo: Path) -> None:  # noqa: F811
    run = _make_run(fake_repo, "f2")
    _pipeline(run, status="failed", steps=[("train", "ok", 0), ("blend", "ok", 0), ("bundle", "ok", 0), ("pkl_check", "failed", 1)])
    tail = run / "pipeline_steps" / "2026-3-pkl_check.log"
    tail.write_text("downloading " + SIGNED + "\nmlp: patience=5 hidden=[256, 64] lr=0.001\nPROBLEM: torch imported\n")
    _manifest(fake_repo, "f2", passed=False)
    doc = report.write_results("f2", "04", repo=fake_repo)["doc"]
    text = (fake_repo / "handoffs" / "results" / "04-results.md").read_text()
    assert doc["status"] == "failed" and any("pipeline failed at step pkl_check" in r for r in doc["status_reasons"])
    fs = doc["pipeline"]["failing_step"]
    assert fs["name"] == "pkl_check" and "PROBLEM: torch imported" in fs["log_tail"]
    joined = "\n".join(fs["log_tail"]) + text
    assert "f" * 64 not in joined and "Z" * 16 not in joined and "?[REDACTED]" in joined
    assert "256, 64" not in joined and "patience=5" not in joined and "[hyperparameter]" in joined
    assert "STATUS: FAILED" in text and "failing step `pkl_check`" in text


@pytest.mark.parametrize("status", ["running", "interrupted"])
def test_unfinished_pipeline_is_never_ok(fake_repo: Path, status: str) -> None:  # noqa: F811
    run = _full_ok(fake_repo, "f3")
    _pipeline(run, status=status, steps=[("train", "ok", 0), ("blend", status, 130 if status == "interrupted" else 0)])
    doc = report.write_results("f3", "04", repo=fake_repo)["doc"]
    assert doc["status"] == status
    if status == "running":
        assert any("killed before it recorded an end" in r for r in doc["status_reasons"])


def test_report_inside_pipeline_sees_compute_result(fake_repo: Path) -> None:  # noqa: F811
    run = _full_ok(fake_repo, "f4")
    _pipeline(run, status="ok", report_step={"name": "report", "status": "running"})
    assert report.write_results("f4", "04", repo=fake_repo)["status"] == "ok"


@pytest.mark.parametrize("case", ["no_sha", "other_sha", "file_bytes_differ", "check_failed"])
def test_pkl_check_record_must_name_these_bytes(fake_repo: Path, case: str) -> None:  # noqa: F811
    run = _full_ok(fake_repo, "f5")
    if case == "no_sha":
        _manifest(fake_repo, "f5", check_sha=None)
    elif case == "other_sha":
        _manifest(fake_repo, "f5", check_sha="cd" * 32)
    elif case == "check_failed":
        _manifest(fake_repo, "f5", passed=False)
    else:
        _manifest(fake_repo, "f5", pkl_bytes=b"real pickle bytes")
        (fake_repo / "models" / "f5.pkl").write_bytes(b"tampered bytes")
    _diag_meta(fake_repo, "f5", pkl_sha=json.loads((fake_repo / "models" / "f5.manifest.json").read_text())["sha256"])
    doc = report.write_results("f5", "04", repo=fake_repo)["doc"]
    assert doc["status"] == "failed", doc["evidence"]
    assert run.exists()


def test_missing_files_behind_ok_steps_make_it_incomplete(fake_repo: Path) -> None:  # noqa: F811
    run = _full_ok(fake_repo, "f6")
    (fake_repo / "artifacts" / "diag" / "f6_validation.parquet.meta.json").unlink()
    (run / "audit.json").unlink()
    doc = report.write_results("f6", "04", repo=fake_repo)["doc"]
    assert doc["status"] == "incomplete"
    ev = {e["item"]: e["status"] for e in doc["evidence"]}
    assert ev["diag_file"] == "missing" and ev["audit"] == "missing"
    text = (fake_repo / "handoffs" / "results" / "04-results.md").read_text()
    assert "STATUS: INCOMPLETE" in text and "## Traceback" not in text


def test_audit_failed_and_partial_diag_file(fake_repo: Path) -> None:  # noqa: F811
    run = _full_ok(fake_repo, "f7")
    _audit(run, status="failed")
    _diag_meta(fake_repo, "f7", rows=450, eras_limit=5)
    doc = report.write_results("f7", "04", repo=fake_repo)["doc"]
    ev = {e["item"]: e["status"] for e in doc["evidence"]}
    assert doc["status"] == "failed" and ev["audit"] == "fail" and ev["diag_file"] == "warn"


def test_retry_keeps_every_attempts_step_seconds_and_mlp_tail(fake_repo: Path) -> None:  # noqa: F811
    run = _full_ok(fake_repo, "f8")
    (run / "pipeline_steps").mkdir(exist_ok=True)
    (run / "pipeline_steps" / "a1-mlp.log").write_text("NO USABLE DEVICE -> skip the MLP member\n")
    a1 = {"started_at": "t0", "from_step": "train", "status": "failed", "seconds": 600, "failed_step": "bundle", "steps": [
        {"name": "train", "status": "ok", "exit_code": 0, "seconds": 497},
        {"name": "mlp", "status": "skipped", "exit_code": 4, "seconds": 3, "log_file": "pipeline_steps/a1-mlp.log",
         "note": "optional member"},
        {"name": "blend", "status": "ok", "exit_code": 0, "seconds": 208},
        {"name": "bundle", "status": "failed", "exit_code": 1, "seconds": 5}]}
    a2 = {"started_at": "t1", "from_step": "bundle", "status": "ok", "seconds": 90, "steps": [
        {"name": "bundle", "status": "ok", "exit_code": 0, "seconds": 20},
        {"name": "pkl_check", "status": "ok", "exit_code": 0, "seconds": 40},
        {"name": "diag", "status": "ok", "exit_code": 0, "seconds": 30}]}
    _pipeline(run, attempts=[a1, a2], mlp="skipped (mlp_train exit 4)")
    doc = report.write_results("f8", "04", repo=fake_repo)["doc"]
    assert doc["status"] == "ok", doc["status_reasons"]
    ss = doc["pipeline"]["step_seconds"]
    assert ss["train"] == [{"attempt": 1, "seconds": 497, "status": "ok"}]
    assert [r["attempt"] for r in ss["bundle"]] == [1, 2] and doc["pipeline"]["total_seconds"] == 690
    text = (fake_repo / "handoffs" / "results" / "04-results.md").read_text()
    assert "pipeline step train (attempt 1, ok) | 497" in text and "pipeline total (all attempts) | 690" in text
    assert "TOTAL (metrics.json, last train invocation)" in text
    assert doc["pipeline"]["mlp_step"] is None  # the latest attempt did not run the mlp step
    _pipeline(run, attempts=[{**a1, "status": "ok", "failed_step": None, "steps": a1["steps"][:3]}],
              mlp="skipped (mlp_train exit 4)")
    doc = report.write_results("f8", "04", repo=fake_repo)["doc"]
    assert "NO USABLE DEVICE" in "\n".join(doc["pipeline"]["mlp_step"]["log_tail"])


def test_legacy_run_without_pipeline_keeps_metrics_status(fake_repo: Path) -> None:  # noqa: F811
    _make_run(fake_repo, "f9")
    _manifest(fake_repo, "f9", passed=False)
    doc = report.write_results("f9", "02", repo=fake_repo)["doc"]
    assert doc["status"] == "ok" and doc["pipeline"] is None  # informational only without pipeline.json
    assert all(not e["required"] for e in doc["evidence"])


def test_scrub_signed_urls_and_is_idempotent() -> None:
    gcs = ("https://storage.googleapis.com/b/o?X-Goog-Algorithm=GOOG4&X-Goog-Credential=svc%40p&X-Goog-Signature="
           + "0a" * 40)
    azure = "https://acct.blob.core.windows.net/c/f.parquet?sv=2024&sig=" + "Q" * 30 + "&se=2026"
    tok = "https://api.example.com/dl/train.parquet?token=" + "t" * 32
    plain = "https://api.everesteer.ai/api/v1/models?page=2&limit=5"
    bare = "header X-Amz-Signature=" + "e" * 40 + " end"
    for url in (SIGNED, gcs, azure, tok):
        out = report.scrub(f"GET {url} 200")
        assert out.endswith("?[REDACTED] 200") and url.split("?", 1)[1] not in out
        assert report.scrub(out) == out
    assert report.scrub(plain) == plain
    assert "e" * 40 not in report.scrub(bare) and report.scrub(bare).endswith(" end")
    js = json.dumps({"url": SIGNED})
    assert json.loads(report.scrub(js))["url"].endswith("?[REDACTED]")  # stays valid JSON


def test_scrub_params_drops_hyperparameter_values() -> None:
    line = ("torch 2.15, device cpu (--device/params: cpu), 2 threads; member mlp_everest seeds=[0] epochs_cap=3 patience=2 "
            "hidden=[32, 16] aux=['a']; parameters: {'num_leaves': 8, 'learning_rate': 0.03}; fit20 mean=0.0261")
    out = report.scrub_params(line)
    for bad in ("32, 16", "patience=2", "num_leaves", "learning_rate", "0.03"):
        assert bad not in out, bad
    assert "fit20 mean=0.0261" in out and "epochs_cap=3" in out


def test_log_tail_hides_home_directories(fake_repo: Path) -> None:  # noqa: F811
    run = _make_run(fake_repo, "f10")
    with (run / "run.log").open("a") as fh:
        fh.write("wrote /home/someuser/Projects/eversteer/stealth/artifacts/f10/blend.json\n")
    tail = report.write_results("f10", "02", repo=fake_repo)["paths"][2].read_text()
    assert "someuser" not in tail and "~/Projects/eversteer/stealth/artifacts/f10/blend.json" in tail
