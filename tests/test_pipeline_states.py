"""scripts/run_pipeline.sh state machine (handoff review findings 1-3): pipeline.json attempts + steps, per-step
logs with stderr, signal handling on the step's own process group, MLP exit codes, retrain / external-member
rules, the --no-mlp --from blend consistency check, audit placement and the --report ordering.

Every step is a FAKE interpreter (a bash stub that records its argv); nothing trains, nothing touches the network."""
from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_pipeline.sh"
PY = sys.executable


def _config(tmp_path: Path, *, audit: bool, mlp: bool = True, name: str = "cfg.yaml") -> Path:
    """minifull with cv.audit forced on/off (and optionally without its mlp member), independent of later edits."""
    cfg = yaml.safe_load((REPO / "configs" / "minifull.yaml").read_text())
    cfg.setdefault("cv", {})["audit"] = {"enabled": bool(audit), "train_end": 5925, "test_start": 5945}
    if not mlp:
        cfg["members"] = [m for m in cfg["members"] if m.get("family") != "mlp"]
    p = tmp_path / name
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def _fakes(tmp_path: Path) -> tuple[Path, Path, Path]:
    log = tmp_path / "calls.log"
    fake_py = tmp_path / "fakepy"
    fake_py.write_text(
        "#!/usr/bin/env bash\n"
        f'if [ "${{1:-}}" = "-" ]; then exec "{PY}" "$@"; fi\n'
        'echo "PY $*" >> "$FAKE_LOG"\n'
        'args="$*"\n'
        'if [ -n "${FAKE_OBSERVE:-}" ]; then case "$args" in *"stealth.report"*)\n'
        f'  "{PY}" -c \'import json,sys; d=json.load(open(sys.argv[1])); r=d.get("report_step") or {{}}; '
        'print(d["status"], r.get("status"))\' "$FAKE_PJSON" > "$FAKE_OBSERVE";; esac; fi\n'
        'if [ -n "${FAKE_SLEEP_MATCH:-}" ]; then case "$args" in *"$FAKE_SLEEP_MATCH"*)\n'
        '  sleep 300 & echo $! > "$FAKE_PIDS.child"; echo $$ > "$FAKE_PIDS.leader"; wait; exit 0;; esac; fi\n'
        'if [ -n "${FAKE_FAIL_MATCH:-}" ]; then case "$args" in *"$FAKE_FAIL_MATCH"*)\n'
        '  echo "BOOM on stderr for: $args" >&2; exit "${FAKE_FAIL_RC:-1}";; esac; fi\n'
        'echo "fake stdout for: $args"\n'
        "exit 0\n")
    fake_gpu = tmp_path / "fakegpu"
    fake_gpu.write_text('#!/usr/bin/env bash\necho "GPU $*" >> "$FAKE_LOG"\necho "mlp stderr line" >&2\n'
                        'exit "${FAKE_GPU_RC:-4}"\n')
    for p in (fake_py, fake_gpu):
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return fake_py, fake_gpu, log


@pytest.fixture()
def run_id() -> str:
    rid = f"pytest_states_{os.getpid()}"
    shutil.rmtree(REPO / "artifacts" / rid, ignore_errors=True)
    yield rid
    shutil.rmtree(REPO / "artifacts" / rid, ignore_errors=True)


def _run(*args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, cwd=REPO, timeout=120,
                          env={**os.environ, **(env or {})}, check=False, start_new_session=True)


def _calls(log: Path) -> list[str]:
    return log.read_text().splitlines() if log.exists() else []


def _modules(log: Path) -> list[str]:
    return [c.split()[2] for c in _calls(log) if c.startswith("PY -m ")]


def _pjson(run_id: str) -> dict:
    return json.loads((REPO / "artifacts" / run_id / "pipeline.json").read_text())


def _alive(pid: int) -> bool:
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat_line.rsplit(")", 1)[1].split()[0] != "Z"  # a zombie is dead for our purpose


def test_dry_run_audit_between_blend_and_bundle(tmp_path: Path) -> None:
    on = _run(str(_config(tmp_path, audit=True)), "pytest_dry", "--py", PY, "--gpu-py", "/opt/ml/bin/python3", "--dry-run")
    off = _run(str(_config(tmp_path, audit=False, name="off.yaml")), "pytest_dry", "--py", PY, "--no-mlp", "--dry-run")
    assert on.returncode == 0 and off.returncode == 0, on.stderr + off.stderr
    steps_on = [ln.split("]", 1)[0] + "]" for ln in on.stdout.splitlines() if ln.startswith("DRY-RUN [")]
    assert steps_on == ["DRY-RUN [train]", "DRY-RUN [mlp]", "DRY-RUN [retrain]", "DRY-RUN [blend]", "DRY-RUN [audit]",
                        "DRY-RUN [bundle]", "DRY-RUN [pkl_check]", "DRY-RUN [diag]"]
    assert "stealth.audit --run-id pytest_dry" in on.stdout and "[audit]" not in off.stdout
    train = next(ln for ln in on.stdout.splitlines() if ln.startswith("DRY-RUN [train]"))
    retrain = next(ln for ln in on.stdout.splitlines() if ln.startswith("DRY-RUN [retrain]"))
    assert train.endswith("--resume --skip-external") and retrain.endswith("--resume --require-external")
    mlp = next(ln for ln in on.stdout.splitlines() if ln.startswith("DRY-RUN [mlp]"))
    assert "--device cuda" in mlp  # explicit CUDA by default: no silent CPU fallback on the laptop
    assert _run(str(_config(tmp_path, audit=True)), "pytest_dry", "--py", PY, "--mlp-device", "gpu", "--dry-run").returncode == 64


def test_success_records_steps_logs_and_tolerated_mlp_skip(tmp_path: Path, run_id: str) -> None:
    fake_py, fake_gpu, log = _fakes(tmp_path)
    res = _run(str(_config(tmp_path, audit=True)), run_id, "--py", str(fake_py), "--gpu-py", str(fake_gpu),
               env={"FAKE_LOG": str(log)})
    assert res.returncode == 0, res.stdout + res.stderr
    assert _modules(log) == ["stealth.train", "stealth.blend", "stealth.audit", "stealth.bundle", "stealth.pkl_check",
                             "stealth.diag"]
    doc = _pjson(run_id)
    assert doc["schema"] == "stealth.pipeline/1" and doc["status"] == "ok" and doc["exit_code"] == 0
    att = doc["attempts"][-1]
    names = [s["name"] for s in att["steps"]]
    assert names == ["train", "mlp", "blend", "audit", "bundle", "pkl_check", "diag"]
    by = {s["name"]: s for s in att["steps"]}
    assert by["mlp"]["status"] == "skipped" and by["mlp"]["exit_code"] == 4 and "tolerated" in by["mlp"]["note"]
    assert all(by[n]["status"] == "ok" and isinstance(by[n]["seconds"], int) for n in names if n != "mlp")
    assert doc["mlp"] == "skipped (mlp_train exit 4)" and att["options"]["no_mlp"] == "0"
    mlp_log = (REPO / "artifacts" / run_id / by["mlp"]["log_file"]).read_text()
    assert "mlp stderr line" in mlp_log  # stderr is captured too
    assert "fake stdout for: -m stealth.blend" in (REPO / "artifacts" / run_id / by["blend"]["log_file"]).read_text()
    assert "--upload" not in log.read_text() and "--submit" not in log.read_text()


def test_failure_then_retry_two_attempts_and_stderr_in_step_log(tmp_path: Path, run_id: str) -> None:
    fake_py, _, log = _fakes(tmp_path)
    cfg = str(_config(tmp_path, audit=False))
    env = {"FAKE_LOG": str(log)}
    res = _run(cfg, run_id, "--py", str(fake_py), "--no-mlp", env={**env, "FAKE_FAIL_MATCH": "stealth.bundle"})
    assert res.returncode == 1
    assert "PIPELINE FAILED at step bundle (exit 1)" in res.stdout
    doc = _pjson(run_id)
    assert doc["status"] == "failed" and doc["failed_step"] == "bundle" and doc["exit_code"] == 1
    bundle = doc["attempts"][-1]["steps"][-1]
    assert bundle["name"] == "bundle" and bundle["status"] == "failed"
    assert "BOOM on stderr" in (REPO / "artifacts" / run_id / bundle["log_file"]).read_text()
    res = _run(cfg, run_id, "--py", str(fake_py), "--no-mlp", "--from", "bundle", env=env)
    assert res.returncode == 0, res.stdout + res.stderr
    doc = _pjson(run_id)
    assert [a["status"] for a in doc["attempts"]] == ["failed", "ok"] and doc["status"] == "ok"
    assert doc["failed_step"] is None and doc["attempts"][1]["from_step"] == "bundle"
    assert [s["name"] for s in doc["attempts"][0]["steps"]] == ["train", "blend", "bundle"]  # first attempt kept


@pytest.mark.parametrize("rc", [130, 143, 137])
def test_mlp_signal_exit_stops_the_pipeline(tmp_path: Path, run_id: str, rc: int) -> None:
    fake_py, fake_gpu, log = _fakes(tmp_path)
    res = _run(str(_config(tmp_path, audit=False)), run_id, "--py", str(fake_py), "--gpu-py", str(fake_gpu),
               env={"FAKE_LOG": str(log), "FAKE_GPU_RC": str(rc)})
    assert res.returncode == rc
    assert "MLP skipped" not in res.stdout and f"PIPELINE INTERRUPTED at step mlp (exit {rc})" in res.stdout
    assert _modules(log) == ["stealth.train"]  # nothing after the mlp step ran
    doc = _pjson(run_id)
    assert doc["status"] == "interrupted" and doc["failed_step"] == "mlp"
    assert doc["attempts"][-1]["steps"][-1]["status"] == "interrupted"


def test_mlp_other_error_code_fails(tmp_path: Path, run_id: str) -> None:
    fake_py, fake_gpu, log = _fakes(tmp_path)
    res = _run(str(_config(tmp_path, audit=False)), run_id, "--py", str(fake_py), "--gpu-py", str(fake_gpu),
               env={"FAKE_LOG": str(log), "FAKE_GPU_RC": "2"})
    assert res.returncode == 2 and "PIPELINE FAILED at step mlp (exit 2)" in res.stdout
    assert _pjson(run_id)["status"] == "failed"


def test_retrain_rules_for_external_members(tmp_path: Path, run_id: str) -> None:
    fake_py, fake_gpu, log = _fakes(tmp_path)
    cfg = str(_config(tmp_path, audit=False))
    env = {"FAKE_LOG": str(log)}
    # mlp ok -> retrain WITH the member, provenance required
    res = _run(cfg, run_id, "--py", str(fake_py), "--gpu-py", str(fake_gpu), "--no-diag", env={**env, "FAKE_GPU_RC": "0"})
    assert res.returncode == 0, res.stdout + res.stderr
    trains = [c for c in _calls(log) if " -m stealth.train" in c]
    assert trains[0].endswith("--resume --skip-external") and trains[1].endswith("--resume --require-external")
    # --from mlp and the mlp is skipped -> re-assemble WITHOUT external members (a stale MLP cannot survive)
    log.unlink()
    res = _run(cfg, run_id, "--py", str(fake_py), "--gpu-py", str(fake_gpu), "--from", "mlp", "--no-diag", env=env)
    assert res.returncode == 0, res.stdout + res.stderr
    trains = [c for c in _calls(log) if " -m stealth.train" in c]
    assert len(trains) == 1 and trains[0].endswith("--resume --skip-external")
    assert "re-assembling WITHOUT external members" in res.stdout
    # --from retrain --no-mlp -> the same
    log.unlink()
    res = _run(cfg, run_id, "--py", str(fake_py), "--no-mlp", "--from", "retrain", "--no-diag", env=env)
    assert res.returncode == 0 and _modules(log)[0] == "stealth.train"
    assert [c for c in _calls(log) if " -m stealth.train" in c][0].endswith("--skip-external")
    # train ran in this attempt and the mlp failed -> no retrain at all
    log.unlink()
    res = _run(cfg, run_id, "--py", str(fake_py), "--gpu-py", str(fake_gpu), "--no-diag", env={**env, "FAKE_GPU_RC": "1"})
    assert res.returncode == 0 and _modules(log).count("stealth.train") == 1


def test_no_mlp_from_blend_refuses_a_stale_external_member(tmp_path: Path, run_id: str) -> None:
    import pandas as pd

    fake_py, _, log = _fakes(tmp_path)
    cfg = str(_config(tmp_path, audit=False))
    run_dir = REPO / "artifacts" / run_id
    run_dir.mkdir(parents=True)
    oof = pd.DataFrame({"era": [1], "fold": [0], "target_everest": [0.5], "m__lgbm_everest": [0.1], "m__mlp_everest": [0.2]})
    oof.to_parquet(run_dir / "oof.parquet")
    res = _run(cfg, run_id, "--py", str(fake_py), "--no-mlp", "--from", "blend", env={"FAKE_LOG": str(log)})
    assert res.returncode == 64 and "mlp_everest" in res.stderr and "--from retrain --no-mlp" in res.stderr
    assert not log.exists()  # nothing ran
    # without the external column it proceeds; without --no-mlp the member may stay
    oof.drop(columns=["m__mlp_everest"]).to_parquet(run_dir / "oof.parquet")
    assert _run(cfg, run_id, "--py", str(fake_py), "--no-mlp", "--from", "blend", "--no-diag",
                env={"FAKE_LOG": str(log)}).returncode == 0
    oof.to_parquet(run_dir / "oof.parquet")
    assert _run(cfg, run_id, "--py", str(fake_py), "--from", "blend", "--no-diag", env={"FAKE_LOG": str(log)}).returncode == 0


def test_report_observes_a_finished_pipeline_and_its_failure_is_separate(tmp_path: Path, run_id: str) -> None:
    fake_py, _, log = _fakes(tmp_path)
    cfg = str(_config(tmp_path, audit=False))
    obs = tmp_path / "observed.txt"
    env = {"FAKE_LOG": str(log), "FAKE_OBSERVE": str(obs), "FAKE_PJSON": str(REPO / "artifacts" / run_id / "pipeline.json")}
    res = _run(cfg, run_id, "--py", str(fake_py), "--no-mlp", "--report", "99", env=env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert obs.read_text().split() == ["ok", "running"]  # the compute result was recorded BEFORE the report ran
    doc = _pjson(run_id)
    assert doc["status"] == "ok" and doc["report_step"]["status"] == "ok"
    assert "report" not in [s["name"] for s in doc["attempts"][-1]["steps"]]
    # a failing report: non-zero exit, the pipeline result itself stays ok, the report step is recorded as failed
    res = _run(cfg, run_id, "--py", str(fake_py), "--no-mlp", "--from", "report", "--report", "99",
               env={**env, "FAKE_FAIL_MATCH": "stealth.report", "FAKE_FAIL_RC": "1"})
    assert res.returncode == 1 and "REPORT FAILED at step report (exit 1)" in res.stdout
    doc = _pjson(run_id)
    assert doc["status"] == "ok" and doc["report_step"]["status"] == "failed" and doc["report_step"]["exit_code"] == 1


@pytest.mark.parametrize("sig, rc", [(signal.SIGTERM, 143), (signal.SIGINT, 130)])
def test_signal_stops_the_step_process_group_only(tmp_path: Path, run_id: str, sig: signal.Signals, rc: int) -> None:
    fake_py, _, log = _fakes(tmp_path)
    pids = tmp_path / "pids"
    env = {**os.environ, "FAKE_LOG": str(log), "FAKE_SLEEP_MATCH": "stealth.blend", "FAKE_PIDS": str(pids)}
    bystander = subprocess.Popen(["sleep", "300"])
    try:
        proc = subprocess.Popen(["bash", str(SCRIPT), str(_config(tmp_path, audit=False)), run_id, "--py", str(fake_py),
                                 "--no-mlp"], cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                start_new_session=True)
        leader_f, child_f = Path(f"{pids}.leader"), Path(f"{pids}.child")
        deadline = time.monotonic() + 60
        while not (leader_f.exists() and child_f.exists() and child_f.read_text().strip()) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert leader_f.exists(), "the blend stub never started"
        leader, child = int(leader_f.read_text()), int(child_f.read_text())
        assert _alive(leader) and _alive(child)
        os.kill(proc.pid, sig)
        out, err = proc.communicate(timeout=60)
        assert proc.returncode == rc, out + err
        for _ in range(50):
            if not (_alive(leader) or _alive(child)):
                break
            time.sleep(0.1)
        assert not _alive(leader) and not _alive(child), "the step's process group survived the interrupt"
        assert _alive(bystander.pid), "cleanup killed a process outside the step's process group"
        assert "PIPELINE INTERRUPTED at step blend" in out
        doc = _pjson(run_id)
        assert doc["status"] == "interrupted" and doc["failed_step"] == "blend" and doc["exit_code"] == rc
        assert doc["attempts"][-1]["steps"][-1]["status"] == "interrupted"
        assert _modules(log) == ["stealth.train", "stealth.blend"]
        # the next attempt resumes; the interrupted one stays on record
        res = _run(str(_config(tmp_path, audit=False)), run_id, "--py", str(fake_py), "--no-mlp", "--no-diag",
                   env={"FAKE_LOG": str(log)})
        assert res.returncode == 0
        assert [a["status"] for a in _pjson(run_id)["attempts"]] == ["interrupted", "ok"]
    finally:
        bystander.kill()
        bystander.wait()


def test_killed_driver_leaves_running_attempt_marked_interrupted_next_time(tmp_path: Path, run_id: str) -> None:
    fake_py, _, log = _fakes(tmp_path)
    run_dir = REPO / "artifacts" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "pipeline.json").write_text(json.dumps({"schema": "stealth.pipeline/1", "status": "running", "attempts": [
        {"started_at": "t", "from_step": "train", "status": "running", "steps": [{"name": "train", "status": "running"}]}]}))
    res = _run(str(_config(tmp_path, audit=False)), run_id, "--py", str(fake_py), "--no-mlp", "--no-diag",
               env={"FAKE_LOG": str(log)})
    assert res.returncode == 0
    doc = _pjson(run_id)
    assert doc["attempts"][0]["status"] == "interrupted" and "no end record" in doc["attempts"][0]["note"]
    assert doc["attempts"][1]["status"] == "ok"


def test_setup_gpu_script_usage_and_cpu_build_reinstall() -> None:
    gpu = REPO / "scripts" / "setup_gpu.sh"
    assert subprocess.run(["bash", "-n", str(gpu)], capture_output=True, check=False).returncode == 0
    assert subprocess.run(["bash", str(gpu), "--help"], capture_output=True, check=False).returncode == 0
    assert subprocess.run(["bash", str(gpu), "unexpected"], capture_output=True, check=False).returncode == 64
    text = gpu.read_text()
    # exit codes the handoff branches on, the CUDA 12.8 index (Blackwell sm_120), a CPU-only torch never kept
    for needle in ("echo \"GPU OK\"", "echo \"GPU FAIL -> skip MLP\"", "exit 3", "SETUP ERROR", "whl/cu128",
                   "torch.version.cuda is None", "--reinstall-package torch", "--force-reinstall"):
        assert needle in text, needle
