"""Environment self-check: pinned-version compliance, OS/CPU/RAM/GPU/TZ table, ``core.hooksPath`` check.

CLI contract (docs/INTERFACES.md section 10)::

    python -m stealth.selfcheck [--lock requirements-lock.txt] [--json]

Prints a table (python, every pin in ``requirements-lock.txt`` vs the installed version with OK/MISMATCH/MISSING,
OS, CPU, RAM, GPU + driver via ``nvidia-smi`` when present, timezone, ``git config core.hooksPath``, whether
``EIQ_API_KEY`` is present). Exit code 1 on any pin mismatch, 0 otherwise. The API key VALUE is never read into
the output; only its presence is reported. Fully offline.

``collect_env`` is reused by ``stealth.report`` for the ``host`` block of the results JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
LOCK_FILE = REPO / "requirements-lock.txt"
#: pins that the pickle lane cares about; shown first in the table (the rest of the lock is still checked)
LANE_PINS = (
    "lightgbm", "xgboost", "scikit-learn", "pandas", "numpy", "scipy", "catboost", "pyarrow", "cloudpickle",
    "everesteer-api",
)
#: the lock may contain alternative spellings; both compare on this normal form
_NORMALISE = str.maketrans({"_": "-", ".": "-"})


def _norm(name: str) -> str:
    return name.strip().lower().translate(_NORMALISE)


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    """Run ``cmd`` and return stripped stdout, or None on any failure (missing binary, non-zero, timeout)."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


# --------------------------------------------------------------------------------------------- pins


def parse_lock(lock_file: Path = LOCK_FILE) -> dict[str, str]:
    """Return ``{normalised_name: pinned_version}`` from a ``name==version`` lock file (comments/blank lines skipped).

    Raises ``FileNotFoundError`` when the lock is absent.
    """
    pins: dict[str, str] = {}
    for raw in lock_file.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "==" not in line:
            continue
        name, version = line.split("==", 1)
        pins[_norm(name)] = version.strip()
    return pins


def installed_version(name: str) -> str | None:
    """Installed distribution version for ``name`` (any spelling), or None when not installed."""
    for candidate in {name, name.replace("-", "_"), name.replace("-", ".")}:
        try:
            return metadata.version(candidate)
        except metadata.PackageNotFoundError:
            continue
    return None


def check_pins(lock_file: Path = LOCK_FILE) -> list[dict[str, str]]:
    """Compare every lock pin with the installed version.

    Returns rows ``{"package", "pinned", "installed", "status"}`` with status in {"OK", "MISMATCH", "MISSING"},
    lane pins first in ``LANE_PINS`` order, then the rest alphabetically.
    """
    pins = parse_lock(lock_file)
    order = [p for p in LANE_PINS if p in pins] + sorted(p for p in pins if p not in LANE_PINS)
    rows: list[dict[str, str]] = []
    for name in order:
        pinned = pins[name]
        inst = installed_version(name)
        if inst is None:
            status = "MISSING"
        elif inst == pinned:
            status = "OK"
        else:
            status = "MISMATCH"
        rows.append({"package": name, "pinned": pinned, "installed": inst or "-", "status": status})
    return rows


def pins_ok(rows: list[dict[str, str]]) -> bool:
    return all(r["status"] == "OK" for r in rows)


# --------------------------------------------------------------------------------------------- host


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown"


def _ram_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _gpu() -> tuple[str | None, str | None]:
    """(gpu name, driver version) from ``nvidia-smi``; (None, None) when absent or failing."""
    if shutil.which("nvidia-smi") is None:
        return None, None
    out = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if not out:
        return None, None
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return parts[0], None


def _timezone() -> str:
    tz = datetime.now().astimezone().tzinfo
    name = os.environ.get("TZ") or (tz.tzname(datetime.now()) if tz is not None else None)
    etc_tz = Path("/etc/timezone")
    if etc_tz.exists():
        name = etc_tz.read_text().strip() or name
    else:
        localtime = Path("/etc/localtime")
        if localtime.is_symlink():
            target = os.readlink(localtime)
            if "zoneinfo/" in target:
                name = target.split("zoneinfo/", 1)[1]
    offset = datetime.now().astimezone().strftime("%z")
    return f"{name or 'unknown'} (UTC{offset[:3]}:{offset[3:]})"


def hooks_path(repo: Path = REPO) -> str | None:
    """``git config core.hooksPath`` for ``repo`` (only this one key is read; never ``--list``)."""
    return _run(["git", "-C", str(repo), "config", "--get", "core.hooksPath"]) or None


def git_sha(repo: Path = REPO) -> str | None:
    return _run(["git", "-C", str(repo), "rev-parse", "HEAD"]) or None


def collect_env(lock_file: Path = LOCK_FILE, repo: Path = REPO) -> dict[str, Any]:
    """Host + pin picture as one dict (the ``host`` block of results JSON plus ``pins``/``hooks_path``/``api_key``).

    Keys: os, cpu, cpu_count, ram_gb, gpu, driver, python, libs {name: installed version}, tz, hooks_path,
    git_sha, api_key_present (bool), pins (rows from :func:`check_pins`), pins_ok (bool).
    """
    rows = check_pins(lock_file) if lock_file.exists() else []
    gpu, driver = _gpu()
    return {
        "os": platform.platform(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "gpu": gpu,
        "driver": driver,
        "python": platform.python_version(),
        "python_exe": sys.executable,
        "libs": {r["package"]: r["installed"] for r in rows if r["installed"] != "-"},
        "tz": _timezone(),
        "hooks_path": hooks_path(repo),
        "git_sha": git_sha(repo),
        "api_key_present": bool(os.environ.get("EIQ_API_KEY") or os.environ.get("EVEREST_API_KEY")),
        "pins": rows,
        "pins_ok": pins_ok(rows) and bool(rows),
    }


# --------------------------------------------------------------------------------------------- render


def _table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    lines = [fmt.format(*header), fmt.format(*["-" * w for w in widths])]
    lines += [fmt.format(*[str(c) for c in r]) for r in rows]
    return "\n".join(lines)


def render(env: dict[str, Any]) -> str:
    """Human-readable report (also embedded verbatim in results files by ``stealth.report``)."""
    host_rows = [
        ["python", f"{env['python']}  ({env['python_exe']})"],
        ["os", env["os"]],
        ["cpu", f"{env['cpu']}  x{env['cpu_count']}"],
        ["ram_gb", str(env["ram_gb"])],
        ["gpu", env["gpu"] or "none (nvidia-smi absent or no device)"],
        ["driver", env["driver"] or "-"],
        ["timezone", env["tz"]],
        ["git_sha", env["git_sha"] or "-"],
        ["core.hooksPath", env["hooks_path"] or "UNSET  <- run: git config core.hooksPath scripts/hooks"],
        ["EIQ_API_KEY", "present" if env["api_key_present"] else "ABSENT (copy .env.example to .env; API steps will fail)"],
    ]
    out = ["== host ==", _table(host_rows, ["item", "value"]), "", "== pins (requirements-lock.txt) =="]
    if env["pins"]:
        out.append(_table([[r["package"], r["pinned"], r["installed"], r["status"]] for r in env["pins"]],
                          ["package", "pinned", "installed", "status"]))
    else:
        out.append("requirements-lock.txt not found")
    out.append("")
    out.append("PINS OK" if env["pins_ok"] else "PINS MISMATCH -> run: $PY -m pip install -r requirements-lock.txt")
    if not env["hooks_path"]:
        out.append("WARNING: core.hooksPath unset; the secret-scanning pre-commit hook is not active")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """CLI entry. Exit 1 on pin mismatch/missing, 0 otherwise."""
    parser = argparse.ArgumentParser(prog="python -m stealth.selfcheck", description=__doc__.split("\n\n")[0])
    parser.add_argument("--lock", type=Path, default=LOCK_FILE, help="lock file to compare against")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of the table")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    env = collect_env(args.lock)
    text = json.dumps(env, indent=2, default=str) if args.json else render(env)
    sys.stdout.write(text + "\n")
    return 0 if env["pins_ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
