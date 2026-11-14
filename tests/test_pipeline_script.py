"""scripts/run_pipeline.sh (docs/INTERFACES.md section 20): bash syntax, dry-run command list, step control, and
the failure semantics (MLP failure tolerated, any other failing step stops with its exit code) with fake interpreters.
Nothing here trains a model or touches the network."""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_pipeline.sh"
PY = sys.executable  # the train venv (has yaml); the script only reads the config with it in --dry-run mode
STEP_TAGS = ["[train]", "[mlp]", "[retrain]", "[blend]", "[bundle]", "[pkl_check]", "[diag]", "[report]"]


def _minifull_tags() -> list[str]:
    """STEP_TAGS plus [audit] after [blend] when configs/minifull.yaml enables the reserved audit partition."""
    import yaml

    cfg = yaml.safe_load((REPO / "configs" / "minifull.yaml").read_text())
    tags = list(STEP_TAGS)
    if ((cfg.get("cv") or {}).get("audit") or {}).get("enabled"):
        tags.insert(tags.index("[blend]") + 1, "[audit]")
    return tags


def _run(*args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, cwd=REPO, timeout=120,
                          env={**os.environ, **(env or {})}, check=False)


def _dry_steps(out: str) -> list[str]:
    return [line[len("DRY-RUN "):].split("]", 1)[0] + "]" for line in out.splitlines() if line.startswith("DRY-RUN [")]


def _dry_line(out: str, tag: str) -> str:
    return next(line for line in out.splitlines() if line.startswith(f"DRY-RUN {tag}"))


def test_bash_syntax() -> None:
    res = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr
    text = SCRIPT.read_text()
    assert text.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in text


def test_dry_run_minifull_every_step(tmp_path: Path) -> None:
    res = _run("configs/minifull.yaml", "pytest_dry", "--py", PY, "--gpu-py", "/opt/ml/bin/python3", "--mlp-device", "cpu",
               "--mlp-seeds", "0 1", "--pkl-check-python", "artifacts/pklcheck_venv/bin/python", "--diag-eras", "5",
               "--report", "99", "--dry-run")
    assert res.returncode == 0, res.stdout + res.stderr
    out = res.stdout
    assert _dry_steps(out) == _minifull_tags()
    assert "--resume" in _dry_line(out, "[train]") and str(REPO / "configs" / "minifull.yaml") in _dry_line(out, "[train]")
    mlp = _dry_line(out, "[mlp]")
    assert "scripts/mlp_train.py --run-id pytest_dry --resume --device cpu --seeds 0 1" in mlp
    assert _dry_line(out, "[bundle]").endswith("--require-frozen")  # minifull has blend.neutralisation.select: true
    assert str(REPO / "models" / "pytest_dry.pkl") in _dry_line(out, "[bundle]")
    assert "--venv-python" in _dry_line(out, "[pkl_check]")
    diag = _dry_line(out, "[diag]")
    assert "artifacts/diag/pytest_dry_validation.parquet" in diag and diag.endswith("--eras 5")
    assert _dry_line(out, "[report]").endswith("--handoff 99 --no-git")
    for forbidden in ("--upload", "--submit", "--i-have-approval", "STEALTH_CONFIRM", "stake"):
        assert forbidden not in out, forbidden
    assert "DRY RUN done: nothing was executed" in out
    assert not (REPO / "artifacts" / "pytest_dry").exists()  # a dry run writes nothing


def test_dry_run_m1_config_and_no_mlp() -> None:
    res = _run("configs/test.yaml", "pytest_dry", "--py", PY, "--gpu-py", "/opt/ml/bin/python3", "--dry-run")
    assert res.returncode == 0, res.stderr
    # no mlp member -> no mlp/retrain; no neutralisation selection -> no --require-frozen; no --report -> no report
    assert _dry_steps(res.stdout) == ["[train]", "[blend]", "[bundle]", "[pkl_check]", "[diag]"]
    assert "--require-frozen" not in res.stdout
    res = _run("configs/minifull.yaml", "pytest_dry", "--py", PY, "--no-mlp", "--no-diag", "--dry-run")
    assert res.returncode == 0, res.stderr
    expect = [t for t in _minifull_tags() if t not in ("[mlp]", "[retrain]", "[diag]", "[report]")]
    assert _dry_steps(res.stdout) == expect
    assert "MLP skipped: skipped (--no-mlp)" in res.stdout and "step diag: skipped (--no-diag)" in res.stdout
    res = _run("configs/minifull.yaml", "pytest_dry", "--py", PY, "--dry-run")  # no --gpu-py
    assert "MLP skipped: skipped (no --gpu-py)" in res.stdout and "[mlp]" not in res.stdout


def test_from_step_and_bad_arguments() -> None:
    res = _run("configs/minifull.yaml", "pytest_dry", "--py", PY, "--from", "blend", "--dry-run")
    tags = _minifull_tags()
    assert res.returncode == 0 and _dry_steps(res.stdout) == tags[tags.index("[blend]"):-1]
    res = _run("configs/minifull.yaml", "pytest_dry", "--py", PY, "--from", "retrain", "--dry-run")
    assert _dry_steps(res.stdout)[0] == "[retrain]"
    assert _run("configs/minifull.yaml", "pytest_dry", "--py", PY, "--from", "nope", "--dry-run").returncode == 64
    assert _run("configs/minifull.yaml", "pytest_dry", "--py", PY, "--bogus").returncode == 64
    assert _run("configs/does_not_exist.yaml", "pytest_dry", "--py", PY).returncode == 64
    assert _run("configs/minifull.yaml", "bad/id", "--py", PY).returncode == 64
    assert _run("configs/minifull.yaml").returncode == 64
    assert _run("configs/minifull.yaml", "pytest_dry", "--py", "/nonexistent/python", "--dry-run").returncode == 64


def _fake_interpreters(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A fake train-venv python (real python only for the script's yaml read on stdin) and a fake GPU python."""
    log = tmp_path / "calls.log"
    fake_py = tmp_path / "fakepy"
    fake_py.write_text(
        "#!/usr/bin/env bash\n"
        f'if [ "${{1:-}}" = "-" ]; then exec "{PY}" "$@"; fi\n'
        'echo "PY $*" >> "$FAKE_LOG"\n'
        'if [ -n "${FAKE_FAIL_MATCH:-}" ]; then case "$*" in *"$FAKE_FAIL_MATCH"*) exit "${FAKE_FAIL_RC:-1}";; esac; fi\n'
        "exit 0\n")
    fake_gpu = tmp_path / "fakegpu"
    fake_gpu.write_text('#!/usr/bin/env bash\necho "GPU $*" >> "$FAKE_LOG"\nexit "${FAKE_GPU_RC:-4}"\n')
    for p in (fake_py, fake_gpu):
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return fake_py, fake_gpu, log


@pytest.fixture()
def run_id() -> str:
    rid = f"pytest_pipeline_{os.getpid()}"
    yield rid
    shutil.rmtree(REPO / "artifacts" / rid, ignore_errors=True)


def test_mlp_failure_is_tolerated_and_other_failures_stop(tmp_path: Path, run_id: str) -> None:
    fake_py, fake_gpu, log = _fake_interpreters(tmp_path)
    env = {"FAKE_LOG": str(log)}
    res = _run("configs/minifull.yaml", run_id, "--py", str(fake_py), "--gpu-py", str(fake_gpu), env=env)
    assert res.returncode == 0, res.stdout + res.stderr
    calls = log.read_text().splitlines()
    assert calls[0].startswith("PY -m stealth.train") and calls[1].startswith("GPU ")
    modules = [c.split()[2] for c in calls if c.startswith("PY -m ") and "stealth.audit" not in c]  # audit: config-dependent
    # mlp exit 4 -> logged, retrain skipped, the rest runs; diag never uploads
    assert modules == ["stealth.train", "stealth.blend", "stealth.bundle", "stealth.pkl_check", "stealth.diag"]
    assert "MLP skipped: skipped (mlp_train exit 4)" in res.stdout and "PIPELINE OK" in res.stdout
    assert not any("--upload" in c for c in calls)
    plog = (REPO / "artifacts" / run_id / "pipeline.log").read_text()
    assert "step blend: exit 0" in plog and "PIPELINE OK" in plog
    # mlp ok -> retrain runs
    log.unlink()
    res = _run("configs/minifull.yaml", run_id, "--py", str(fake_py), "--gpu-py", str(fake_gpu), "--no-diag",
               env={**env, "FAKE_GPU_RC": "0"})
    assert res.returncode == 0, res.stdout + res.stderr
    modules = [c.split()[2] for c in log.read_text().splitlines() if c.startswith("PY -m ") and "stealth.audit" not in c]
    assert modules == ["stealth.train", "stealth.train", "stealth.blend", "stealth.bundle", "stealth.pkl_check"]
    # a failing blend (exit 2 = leak gate) stops the pipeline with that exit code; later steps never run
    log.unlink()
    res = _run("configs/minifull.yaml", run_id, "--py", str(fake_py), "--no-mlp",
               env={**env, "FAKE_FAIL_MATCH": "stealth.blend", "FAKE_FAIL_RC": "2"})
    assert res.returncode == 2
    assert "PIPELINE FAILED at step blend (exit 2)" in res.stdout
    modules = [c.split()[2] for c in log.read_text().splitlines() if c.startswith("PY -m ") and "stealth.audit" not in c]
    assert modules == ["stealth.train", "stealth.blend"]
