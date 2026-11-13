"""scripts/hooks/pre-commit: a sample results file + sha256 file must pass; secrets and forbidden paths must fail.

The hook is executed directly (not via ``git commit``) against a throwaway git repo so the test needs no
user.name/email and never touches the real index. Fake secrets are built by concatenation so that this
test file itself never contains a token-shaped string.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "pre-commit"
SHA40 = "e" * 40
SHA64 = "0123456789abcdef" * 4


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    assert _git(r, "init", "-q").returncode == 0
    _git(r, "config", "user.email", "t@example.invalid")
    _git(r, "config", "user.name", "t")
    hooks = r / "scripts" / "hooks"
    hooks.mkdir(parents=True)
    shutil.copy(HOOK, hooks / "pre-commit")
    (hooks / "pre-commit").chmod(0o755)
    return r


def _stage(repo: Path, rel: str, content: str | bytes) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        p.write_bytes(content)
    else:
        p.write_text(content, encoding="utf-8")
    assert _git(repo, "add", "-f", "--", rel).returncode == 0


def _run_hook(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(repo / "scripts" / "hooks" / "pre-commit")], cwd=repo, capture_output=True,
                          text=True, check=False, env={**os.environ, "GIT_DIR": str(repo / ".git")})


def _unstage_all(repo: Path) -> None:
    _git(repo, "reset", "-q")


def test_hook_is_executable_bash() -> None:
    assert HOOK.exists()
    assert os.access(HOOK, os.X_OK), "scripts/hooks/pre-commit must be chmod +x"
    assert HOOK.read_text().startswith("#!/usr/bin/env bash")


def test_clean_results_and_sha_pass(repo: Path) -> None:
    results = (
        "# Results 01 — smoke\n\n| config | fit_mean |\n|---|---|\n| lgbm_everest | 0.0261 |\n\n"
        f"- git_sha: `{SHA40}`\n- EIQ_API_KEY present: yes\n- header name X-API-Key is used by the SDK\n"
        "- regex list: ghp_[A-Za-z0-9]{20,}, -----BEGIN, https://[^/\\s]*@github\\.com\n"
    )
    _stage(repo, "handoffs/results/01-results.md", results)
    _stage(repo, "models/stealth_v1.pkl.sha256", SHA64 + "\n")
    _stage(repo, "models/stealth_v1.manifest.json", '{"sha256": "%s", "git_sha": "%s"}\n' % (SHA64, SHA40))
    _stage(repo, ".env.example", "EIQ_API_KEY=\nEIQ_BASE_URL=https://api.everesteer.ai\n")
    out = _run_hook(repo)
    assert out.returncode == 0, out.stdout + out.stderr


@pytest.mark.parametrize(
    "rel, line",
    [
        ("notes.md", "EIQ_API_KEY=" + "sk" + "x" * 30),
        ("notes.md", 'EIQ_API_KEY="' + "abcdefgh12345678" + '"'),
        ("notes.md", "EVEREST_API_KEY=" + "qwertyuiop1234"),
        ("curl.sh", "curl -H 'X-API-Key: " + "k" * 24 + "'"),
        ("notes.md", "token " + "ghp_" + "A" * 36),
        ("notes.md", "token " + "github_pat_" + "B" * 40),
        ("notes.md", "token " + "gho_" + "C" * 30),
        ("notes.md", "aws " + "AKIA" + "Q" * 16),
        ("id_rsa", "-----BEGIN " + "OPENSSH PRIVATE KEY-----\nabc\n-----END OPENSSH PRIVATE KEY-----\n"),
        ("remote.txt", "url = https://x-access-token:" + "z" * 20 + "@github.com/Infosuii/stealth.git"),
    ],
)
def test_secrets_fail(repo: Path, rel: str, line: str) -> None:
    _stage(repo, rel, line + "\n")
    out = _run_hook(repo)
    assert out.returncode == 1, out.stdout + out.stderr
    assert "BLOCKED" in out.stdout
    # the hook must not echo the secret value itself
    assert line.split()[-1].strip("'") not in out.stdout


@pytest.mark.parametrize("rel", ["data/futures_train.parquet", ".env", ".env.local", "models/stealth_v1.pkl",
                                 "artifacts/run1/metrics.json", "sub/data/x.csv"])
def test_forbidden_paths_fail(repo: Path, rel: str) -> None:
    _stage(repo, rel, b"\x00binary or text\n" if rel.endswith((".parquet", ".pkl")) else "harmless\n")
    out = _run_hook(repo)
    assert out.returncode == 1, out.stdout + out.stderr
    assert "BLOCKED" in out.stdout


@pytest.mark.parametrize(
    "line",
    [
        "EIQ_API_KEY=  EIQ_BASE_URL=https://api.everesteer.ai   # docs mention",
        "env.write_text('EIQ_API_KEY=' + 'dummy-test-key')",
        "Auth header **`X-API-Key: <key>`**",
        "export EIQ_API_KEY=your-key-here-please",
        "X-API-Key: placeholder_value_1234",
        "regex: EIQ_API_KEY\\s*=\\s*\\S{8,}",
    ],
)
def test_mentions_and_placeholders_pass(repo: Path, line: str) -> None:
    _stage(repo, "docs/notes.md", line + "\n")
    out = _run_hook(repo)
    assert out.returncode == 0, out.stdout + out.stderr


def test_only_staged_content_is_scanned(repo: Path) -> None:
    """A secret in the working tree but NOT staged must not block; a staged clean version passes."""
    _stage(repo, "notes.md", "clean\n")
    (repo / "notes.md").write_text("EIQ_API_KEY=" + "x" * 20 + "\n")
    out = _run_hook(repo)
    assert out.returncode == 0, out.stdout + out.stderr


def test_binary_blob_is_skipped(repo: Path) -> None:
    _stage(repo, "models/blob.bin", bytes(range(256)) * 4)
    out = _run_hook(repo)
    assert out.returncode == 0, out.stdout + out.stderr


def test_nothing_staged_passes(repo: Path) -> None:
    out = _run_hook(repo)
    assert out.returncode == 0, out.stdout + out.stderr
