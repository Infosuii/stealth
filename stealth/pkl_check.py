"""Portability + contract gate for a ``.pkl``: run in a throwaway venv WITHOUT ``stealth`` (``python -I``, cwd=tmp).

What it proves (docs/PLAN.md "Packaging", Fix C1):

* the pickle loads in an interpreter that has only the lane pins (no ``stealth`` package on sys.path) and whose
  ``major.minor`` equals the version declared at upload (3.11; ``--expect-python``). The checker VERIFIES this itself
  before loading anything: every lane pin's installed version (``importlib.metadata``) must equal the lock file, and
  neither ``stealth`` nor ``torch`` may be importable (``importlib.util.find_spec``) -- the training venv, which has
  ``stealth`` installed editable, therefore fails, whatever ``--venv-python`` or the venv marker file claim;
* after ``predict()`` neither ``stealth`` nor ``torch`` is in ``sys.modules``;
* the raw bytes contain no ``b"torch"`` and no ``b"stealth."`` reference other than the by-value module
  qualname ``stealth._runtime`` and its source path (``co_filename``; a ``stealth`` directory that is a parent of
  ``.../stealth/_runtime.py`` -- e.g. a git worktree under the checkout -- is part of that path); bare
  ``b"stealth"`` occurrences are reported, not failed. The MLP member is numpy arrays in an npz, so it adds
  nothing to either scan (never put library names in member metadata);
* ``predict(live)``, ``predict(live, bench)``, ``predict(live, bench_misaligned)``, ``predict(live, bench_constant)``,
  ``predict(live, None)``, ``predict(live, bench.to_frame())`` and ``predict(live_shuffled_extra_cols)`` on the
  live parquet and ``--eras`` random validation eras all return an exact ``pd.DataFrame``, one float column, the
  input index, strictly inside (0, 1), no NaN, deterministic across two calls, < 5 s per era; file < 50 MB;
  a frame missing a feature column raises.

The checker itself is a self-contained script (``CHECKER_SOURCE``) that imports only numpy/pandas/pickle; it is
written to a temp dir and executed with ``python -I`` so neither the repo nor PYTHONPATH can leak in.

CLI: ``python -m stealth.pkl_check models/X.pkl [--venv artifacts/pklcheck_venv] [--venv-python PATH] [--eras 50]
[--data-dir data] [--seed 0]``. Exit 0 = passed, 1 = failed/error. ``<stem>.manifest.json``'s ``pkl_check`` is set to
a non-passing ``running`` record before each attempt and then to :func:`check_record` -- ``{passed, status, at, sha256,
python, libs, checker, eras, seconds_per_era_max, isolation, problems}`` bound to the sha256 of the bytes checked (the
upload gates in ``stealth.predict`` require that hash to equal the file's).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("stealth.pkl_check")  # explicit: under `python -m` __name__ is __main__

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
LOCK_FILE = REPO_ROOT / "requirements-lock.txt"
DEFAULT_VENV = REPO_ROOT / "artifacts" / "pklcheck_venv"
LANE_PACKAGES = ("catboost", "cloudpickle", "lightgbm", "numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "xgboost")
MAX_PKL_MB = 50.0
EXPECT_PYTHON = "3.11"  # == predict.PKL_PYTHON_VERSION: the interpreter minor declared at upload
MAX_SECONDS_PER_ERA = 5.0

# --------------------------------------------------------------------------------------------------
# the standalone checker (NO stealth import; runs under `python -I` in a temp cwd)
# --------------------------------------------------------------------------------------------------
CHECKER_SOURCE = r'''
"""pkl contract checker: standalone, imports only stdlib + numpy + pandas (+ pyarrow via pandas)."""
import argparse, hashlib, importlib.metadata, importlib.util, json, pickle, sys, time, traceback
import numpy as np
import pandas as pd

MAX_MB = %(max_mb)r
MAX_S = %(max_s)r
# What by-value pickling of stealth._runtime legitimately embeds (observed, documented in pkl_check.py):
#   * the qualname/module string      b"stealth._runtime"
#   * code objects' co_filename       b".../stealth/stealth/_runtime.py"  (repo dir + package dir)
#   * the function's __package__      b"stealth" followed by a pickle framing byte (no '.' or '/')
ALLOWED_AFTER = (b"._runtime", b"/_runtime.py", b"/stealth/_runtime.py", b"\\_runtime.py", b"\\stealth\\_runtime.py")
IDENT = set(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./\\")
# a 'stealth' directory that is a PARENT of the runtime file in co_filename (e.g. a checkout under
# /home/u/work/stealth/.claude/worktrees/x/stealth/_runtime.py) is allowed: the rest of that path must contain the file
PATH_CHARS = IDENT | set(b"-~+ ")
RUNTIME_FILE = (b"/stealth/_runtime.py", b"\\stealth\\_runtime.py")


def _path_run(raw, j):
    k, n = j, len(raw)
    while k < n and raw[k] in PATH_CHARS and k - j < 4096:
        k += 1
    return raw[j:k]


def scan_bytes(raw):
    """-> (torch_count, stealth_bad_contexts, stealth_total). 'bad' = any stealth.<x> / stealth/<x> that is not _runtime."""
    torch_count = raw.count(b"torch")
    bad, total, start = [], 0, 0
    while True:
        i = raw.find(b"stealth", start)
        if i < 0:
            break
        total += 1
        tail = raw[i + 7: i + 8]
        ok = any(raw.startswith(a, i + 7) for a in ALLOWED_AFTER) or not tail or tail[0] not in IDENT
        if not ok and tail in (b"/", b"\\"):
            ok = any(r in _path_run(raw, i + 7) for r in RUNTIME_FILE)
        if not ok:
            bad.append(raw[max(0, i - 20): i + 40].decode("latin-1"))
        start = i + 7
    return torch_count, bad, total


def check_frame(out, index, label, problems):
    if type(out) is not pd.DataFrame:
        problems.append(f"{label}: returned {type(out).__name__}, not an exact pandas.DataFrame"); return None
    if out.shape[1] != 1:
        problems.append(f"{label}: {out.shape[1]} columns (want 1)")
    if not out.index.equals(index):
        problems.append(f"{label}: index differs from live_features.index")
    if out.shape[1] == 0:
        return None
    v = out.iloc[:, 0]
    if not (isinstance(v.dtype, np.dtype) and np.issubdtype(v.dtype, np.floating)):
        problems.append(f"{label}: dtype {v.dtype} is not float"); return None
    a = v.to_numpy(dtype=np.float64)
    if a.size and not np.isfinite(a).all():
        problems.append(f"{label}: {int((~np.isfinite(a)).sum())} non-finite values")
    elif a.size and not ((a > 0.0).all() and (a < 1.0).all()):
        problems.append(f"{label}: values outside (0,1): min={a.min():.6f} max={a.max():.6f}")
    return a


def spearman(a, b):
    if a is None or b is None or len(a) < 3:
        return None
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def run_era(predict, frame, bench, label, rng, problems, timings, stats):
    index = frame.index
    n = len(index)
    t0 = time.perf_counter()
    p0 = check_frame(predict(frame), index, f"{label} predict(F)", problems)
    timings.append(time.perf_counter() - t0)
    p0b = check_frame(predict(frame), index, f"{label} predict(F) #2", problems)
    if p0 is not None and p0b is not None and not np.array_equal(p0, p0b):
        problems.append(f"{label}: predict(F) not deterministic across two calls")
    pn = check_frame(predict(frame, None), index, f"{label} predict(F, None)", problems)
    if p0 is not None and pn is not None and not np.array_equal(p0, pn):
        problems.append(f"{label}: predict(F, None) != predict(F)")
    pb = None
    if bench is not None:
        t1 = time.perf_counter()
        pb = check_frame(predict(frame, bench), index, f"{label} predict(F, bench)", problems)
        timings.append(time.perf_counter() - t1)
        pbb = check_frame(predict(frame, bench), index, f"{label} predict(F, bench) #2", problems)
        if pb is not None and pbb is not None and not np.array_equal(pb, pbb):
            problems.append(f"{label}: predict(F, bench) not deterministic")
        pdf = check_frame(predict(frame, bench.to_frame()), index, f"{label} predict(F, bench.to_frame())", problems)
        if pb is not None and pdf is not None and not np.array_equal(pb, pdf):
            problems.append(f"{label}: 1-col DataFrame benchmark != Series benchmark")
        # misaligned: drop ~15%% of ids, add 10 foreign ids, shuffle
        keep = rng.random(n) > 0.15
        mis = bench[keep].sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        extra = pd.Series(rng.random(10), index=pd.Index([f"zz{i:04d}" for i in range(10)], name=bench.index.name))
        mis = pd.concat([mis, extra])
        check_frame(predict(frame, mis), index, f"{label} predict(F, bench_misaligned)", problems)
        const = pd.Series(0.5, index=index, name=bench.name)
        pc = check_frame(predict(frame, const), index, f"{label} predict(F, bench_constant)", problems)
        if p0 is not None and pc is not None and not np.array_equal(p0, pc):
            problems.append(f"{label}: constant benchmark changed the output (should be skipped)")
        tiny = bench.iloc[: max(1, n // 4)]
        pt = check_frame(predict(frame, tiny), index, f"{label} predict(F, bench_25pct)", problems)
        if p0 is not None and pt is not None and not np.array_equal(p0, pt):
            problems.append(f"{label}: 25%%-coverage benchmark changed the output (should be skipped)")
        rho = spearman(pb, bench.reindex(index).to_numpy())
        rho0 = spearman(p0, bench.reindex(index).to_numpy())
        if rho is not None:
            stats["rho_bench_with"].append(rho)
        if rho0 is not None:
            stats["rho_bench_without"].append(rho0)
        if p0 is not None and pb is not None:
            r = spearman(p0, pb)
            if r is not None:
                stats["rho_p0_pb"].append(r)
    # shuffled column order + extra column -> identical to predict(F)
    shuffled = frame.sample(frac=1.0, axis=1, random_state=int(rng.integers(0, 2**31 - 1))).copy()
    shuffled["feature_zz_extra"] = 3
    ps = check_frame(predict(shuffled), index, f"{label} predict(F shuffled cols + extra)", problems)
    if p0 is not None and ps is not None and not np.array_equal(p0, ps):
        problems.append(f"{label}: column order / extra column changed the output")
    return p0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--live", required=True)
    ap.add_argument("--live-bench", default=None)
    ap.add_argument("--validation", default=None)
    ap.add_argument("--validation-bench", default=None)
    ap.add_argument("--eras", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", required=True)
    ap.add_argument("--expect-python", default=None, help="major.minor the pickle was built under; mismatch = problem")
    ap.add_argument("--pins", default=None, help="JSON list of name==version lane pins THIS interpreter must have installed")
    args = ap.parse_args()
    report = {"passed": False, "problems": [], "notes": [], "stats": {}}
    problems, notes = report["problems"], report["notes"]
    stats = {"rho_bench_with": [], "rho_bench_without": [], "rho_p0_pb": []}
    timings = []
    report["python_version"] = sys.version
    try:
        if sys.flags.isolated != 1:
            problems.append("checker not running under python -I")
        running = f"{sys.version_info.major}.{sys.version_info.minor}"
        report["python"] = sys.version.split()[0]
        if args.expect_python and running != args.expect_python:
            problems.append(f"checker python {running} != expected {args.expect_python} (build/declare/check under the same minor)")
        # the interpreter itself is part of what is certified: a clean lane venv has the exact pins, no stealth package
        # (the pickle must carry its code by value) and no torch (the lane has none). Checked BEFORE the pickle loads.
        iso = {"stealth_importable": importlib.util.find_spec("stealth") is not None,
               "torch_importable": importlib.util.find_spec("torch") is not None}
        report["isolation"] = iso
        if iso["stealth_importable"]:
            problems.append("the stealth package is importable in the checker interpreter: not a clean lane venv "
                            "(use artifacts/pklcheck_venv, never the train venv)")
        if iso["torch_importable"]:
            problems.append("torch is importable in the checker interpreter: not a clean lane venv")
        pins = json.loads(args.pins) if args.pins else []
        if not pins:
            problems.append("no lane pins given: the checker cannot certify its environment")
        libs = {}
        for pin in pins:
            name, want = str(pin).split("==", 1)
            try:
                have = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                have = None
            libs[name] = have
            if have != want:
                problems.append(f"lane pin {name}=={want} but the checker interpreter has {have or 'nothing installed'}")
        report["libs"] = libs
        raw = open(args.pkl, "rb").read()
        report["sha256"] = hashlib.sha256(raw).hexdigest()
        size_mb = len(raw) / 1e6
        report["size_mb"] = size_mb
        if size_mb >= MAX_MB:
            problems.append(f"pickle is {size_mb:.1f} MB (limit {MAX_MB} MB)")
        torch_count, bad_stealth, total_stealth = scan_bytes(raw)
        report["byte_scan"] = {"torch": torch_count, "stealth_total": total_stealth, "stealth_non_runtime": bad_stealth[:10]}
        if torch_count:
            problems.append(f"raw bytes contain b'torch' {torch_count}x")
        if bad_stealth:
            problems.append(f"raw bytes contain {len(bad_stealth)} 'stealth' reference(s) other than stealth._runtime: {bad_stealth[:3]}")
        notes.append(f"byte scan: 'stealth' occurrences={total_stealth} (all stealth._runtime={not bad_stealth}), torch={torch_count}")
        before = set(sys.modules)
        predict = pickle.loads(raw)
        if not callable(predict):
            problems.append(f"unpickled object is {type(predict).__name__}, not callable")
        if "stealth" in sys.modules or any(m.startswith("stealth.") for m in sys.modules):
            problems.append("'stealth' was imported while loading the pickle (not pickled by value)")
        live = pd.read_parquet(args.live)
        if live.index.name != "id" and "id" in live.columns:
            live = live.set_index("id")
        feats = [c for c in live.columns if c.startswith("feature_")]
        live_f = live[feats]
        bench = None
        if args.live_bench:
            b = pd.read_parquet(args.live_bench)
            if b.index.name != "id" and "id" in b.columns:
                b = b.set_index("id")
            col = "v1_sherpa" if "v1_sherpa" in b.columns else [c for c in b.columns if c != "exped"][0]
            bench = b[col].astype("float64").reindex(live_f.index).rename("eiq_minera_model")
        rng = np.random.default_rng(args.seed)
        run_era(predict, live_f, bench, "live", rng, problems, timings, stats)
        # missing feature column must raise
        try:
            predict(live_f.drop(columns=[feats[0]]))
            problems.append("predict() did not raise on a missing feature column")
        except Exception:
            notes.append("missing feature column raises (ok)")
        n_eras_done = 1
        if args.validation and args.eras > 0:
            import pyarrow.parquet as pq, pyarrow.compute as pc
            expeds = pq.read_table(args.validation, columns=["exped"]).column("exped").unique().to_pylist()
            expeds = sorted(expeds)
            pick = sorted(rng.choice(expeds, size=min(args.eras, len(expeds)), replace=False).tolist())
            tbl = pq.read_table(args.validation, filters=pc.field("exped").isin(pick))
            val = tbl.to_pandas()
            if val.index.name != "id" and "id" in val.columns:
                val = val.set_index("id")
            vbench = None
            if args.validation_bench:
                vb = pq.read_table(args.validation_bench, filters=pc.field("exped").isin(pick)).to_pandas()
                if vb.index.name != "id" and "id" in vb.columns:
                    vb = vb.set_index("id")
                vcol = "v1_sherpa" if "v1_sherpa" in vb.columns else [c for c in vb.columns if c != "exped"][0]
                vbench = vb[vcol].astype("float64")
            vfeats = [c for c in val.columns if c.startswith("feature_")]
            for e in pick:
                fr = val.loc[val["exped"] == e, vfeats]
                be = None if vbench is None else vbench.reindex(fr.index).rename("eiq_minera_model")
                run_era(predict, fr, be, f"validation {e}", rng, problems, timings, stats)
                n_eras_done += 1
                if len(problems) > 50:
                    problems.append("too many problems; stopping early")
                    break
        after = set(sys.modules)
        new_mods = sorted(m for m in after - before if not m.startswith("_"))
        report["modules_loaded"] = [m for m in new_mods if "." not in m][:60]
        if "torch" in sys.modules:
            problems.append("'torch' is in sys.modules after predict()")
        if "stealth" in sys.modules or any(m.startswith("stealth.") for m in sys.modules):
            problems.append("'stealth' is in sys.modules after predict()")
        report["n_eras"] = n_eras_done
        report["seconds_per_era_max"] = max(timings) if timings else None
        report["seconds_per_era_mean"] = float(np.mean(timings)) if timings else None
        if timings and max(timings) >= MAX_S:
            problems.append(f"slowest predict call {max(timings):.2f}s >= {MAX_S}s")
        report["stats"] = {k: (float(np.mean(v)) if v else None) for k, v in stats.items()}
        report["stats"]["n_bench_eras"] = len(stats["rho_bench_with"])
        report["last_info"] = getattr(predict, "last_info", None)
        report["bundle_meta"] = getattr(predict, "bundle_meta", None)
    except Exception as exc:  # report, never crash silently
        problems.append(f"checker exception: {type(exc).__name__}: {exc}")
        report["traceback"] = traceback.format_exc()
    report["passed"] = not problems
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(json.dumps({"passed": report["passed"], "n_problems": len(problems)}))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
'''


# --------------------------------------------------------------------------------------------------
# venv management
# --------------------------------------------------------------------------------------------------
def base_interpreter() -> str:
    """The non-venv Python this process derives from (``sys._base_executable``), used to create the venv."""
    base = getattr(sys, "_base_executable", None) or sys.executable
    return str(base)


def lane_pins(lock_file: Path = LOCK_FILE) -> list[str]:
    """``name==version`` lines from the lock file restricted to the lane packages (no stealth, no pytest/optuna)."""
    pins: list[str] = []
    for line in Path(lock_file).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name = line.split("==", 1)[0].strip().lower()
        if name in LANE_PACKAGES:
            pins.append(line)
    missing = set(LANE_PACKAGES) - {p.split("==", 1)[0].strip().lower() for p in pins}
    if missing:
        raise ValueError(f"{lock_file} lacks lane pins for {sorted(missing)}")
    return pins


def venv_python(venv_dir: Path) -> Path:
    """Interpreter path inside a venv (posix or windows layout)."""
    posix = Path(venv_dir) / "bin" / "python"
    return posix if posix.exists() or os.name != "nt" else Path(venv_dir) / "Scripts" / "python.exe"


def ensure_check_venv(venv_dir: Path = DEFAULT_VENV, lock_file: Path = LOCK_FILE, *, timeout: int = 540) -> Path:
    """Create (or reuse) the throwaway venv with ONLY the lane pins installed; returns its python.

    Reuse requires a ``.pins.txt`` marker equal to the pins we would install; otherwise the venv is rebuilt.
    ``stealth`` is never installed here. Creation can take minutes (pip downloads); ``timeout`` bounds it.
    """
    venv_dir = Path(venv_dir)
    pins = lane_pins(lock_file)
    marker = venv_dir / ".pins.txt"
    py = venv_python(venv_dir)
    if py.exists() and marker.exists() and marker.read_text().split() == pins:
        log.info("reusing check venv %s", venv_dir)
        return py
    if venv_dir.exists():
        log.info("rebuilding check venv %s (stale or incomplete)", venv_dir)
        shutil.rmtree(venv_dir)
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    base = base_interpreter()
    ver = subprocess.run([base, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], capture_output=True, text=True,
                         timeout=30, check=True).stdout.strip()
    if ver != EXPECT_PYTHON:
        raise ValueError(f"base interpreter {base} is Python {ver}, not {EXPECT_PYTHON}; pass --venv-python <clean python{EXPECT_PYTHON}>")
    log.info("creating check venv %s from %s (python %s)", venv_dir, base, ver)
    subprocess.run([base, "-m", "venv", str(venv_dir)], check=True, timeout=120)
    py = venv_python(venv_dir)
    t0 = time.perf_counter()
    log.info("pip install %d lane pins (this can take a few minutes)", len(pins))
    subprocess.run([str(py), "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *pins],
                   check=True, timeout=timeout)
    marker.write_text("\n".join(pins) + "\n")
    log.info("check venv ready in %.0fs", time.perf_counter() - t0)
    return py


# --------------------------------------------------------------------------------------------------
# running the checker
# --------------------------------------------------------------------------------------------------
def _abs(path: Path | str) -> Path:
    """Absolute path WITHOUT resolving symlinks (a venv's ``bin/python`` is a symlink to the base interpreter: resolving
    it would silently swap the clean venv for the base installation)."""
    return Path(os.path.abspath(os.fspath(path)))


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_check(
    pkl: Path,
    *,
    python: Path,
    data_dir: Path = DATA_DIR,
    eras: int = 50,
    seed: int = 0,
    timeout: int = 540,
    expect_python: str | None = EXPECT_PYTHON,
    pins: list[str] | None = None,
) -> dict:
    """Run the standalone checker under ``python -I`` with cwd in a temp dir; returns the report dict.

    ``expect_python`` (default ``"3.11"``, the version declared at upload) is passed to the checker, which records
    a problem when its own ``major.minor`` differs -- the pickle must be built, checked and declared under one minor.
    ``pins`` (default :func:`lane_pins`) are verified INSIDE the checker interpreter against what is actually installed
    there, and the checker refuses an interpreter in which ``stealth`` or ``torch`` is importable: the subprocess, not
    the venv's marker file or the ``--venv-python`` argument, certifies the environment. ``pkl``, ``python`` and
    ``data_dir`` are made absolute before the subprocess starts in its temp cwd. The checked bytes are hashed by the
    checker and compared with the file before and after the run (``report["sha256"]``).

    The report always carries ``passed`` (bool), ``problems`` (list[str]), ``notes``, ``size_mb``, ``byte_scan``,
    ``seconds_per_era_max``, ``stats``, ``isolation``, ``libs``, ``sha256`` and ``stdout``/``stderr`` tails. Needs
    ``data/futures_live.parquet``; validation files are optional (eras skipped when absent).
    """
    pkl, python, data_dir = _abs(pkl), _abs(python), _abs(data_dir)
    pins = lane_pins() if pins is None else list(pins)
    live = data_dir / "futures_live.parquet"
    if not live.exists():
        raise FileNotFoundError(f"{live} missing; run `python -m stealth.data download`")
    if not python.exists():
        raise FileNotFoundError(f"checker interpreter {python} does not exist")
    live_bench = data_dir / "futures_live_benchmark_models.parquet"
    val = data_dir / "futures_validation.parquet"
    val_bench = data_dir / "futures_validation_benchmark_models.parquet"
    sha_before = _sha256(pkl)
    with tempfile.TemporaryDirectory(prefix="pklcheck_") as tmp:
        tmpdir = Path(tmp)
        script = tmpdir / "checker.py"
        script.write_text(CHECKER_SOURCE % {"max_mb": MAX_PKL_MB, "max_s": MAX_SECONDS_PER_ERA})
        report_path = tmpdir / "report.json"
        cmd = [str(python), "-I", str(script), "--pkl", str(pkl), "--live", str(live), "--eras", str(eras),
               "--seed", str(seed), "--report", str(report_path), "--pins", json.dumps(pins)]
        if expect_python:
            cmd += ["--expect-python", str(expect_python)]
        if live_bench.exists():
            cmd += ["--live-bench", str(live_bench)]
        if val.exists():
            cmd += ["--validation", str(val)]
            if val_bench.exists():
                cmd += ["--validation-bench", str(val_bench)]
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        env["PYTHONPATH"] = ""
        env.setdefault("OMP_NUM_THREADS", "2")
        log.info("running checker: %s -I checker.py (cwd=%s, eras=%d)", python, tmpdir, eras)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, cwd=tmpdir, env=env, capture_output=True, text=True, timeout=timeout)
            stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout, stderr, rc = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""), \
                                 (exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""), -9
        elapsed = time.perf_counter() - t0
        if report_path.exists():
            report = json.loads(report_path.read_text())
        else:
            report = {"passed": False, "problems": [f"checker produced no report (rc={rc})"], "notes": []}
        report["returncode"] = rc
        report["elapsed_s"] = elapsed
        report["checker"] = str(python)
        report["eras_requested"] = int(eras)
        report["stdout_tail"] = stdout[-2000:]
        report["stderr_tail"] = stderr[-4000:]
        if rc == -9:
            report["passed"] = False
            report["problems"].append(f"checker timed out after {timeout}s")
        if rc != 0 and report.get("passed"):
            report["passed"] = False
            report["problems"].append(f"checker exited {rc}")
    sha_after = _sha256(pkl)
    checked = report.get("sha256")
    if checked != sha_before or sha_after != sha_before:
        report["passed"] = False
        report["problems"].append(f"checked bytes {str(checked)[:12]} / file before {sha_before[:12]} / after {sha_after[:12]} "
                                  "differ: the pickle changed during the check")
    report["sha256"] = sha_before
    return report


def check_record(pkl: Path, passed: bool, report: dict | None = None, *, status: str | None = None,
                 error: str | None = None) -> dict:
    """The manifest's ``pkl_check`` record (docs/INTERFACES.md section 22.F), bound to the sha256 of the bytes checked.

    ``{"passed", "status", "at", "sha256", "python", "libs", "checker", "eras", "seconds_per_era_max", "isolation",
    "problems"}``; without a ``report`` (the ``running`` marker, an error before the checker ran, a legacy caller) only
    the first four (+ ``error``) are set and ``sha256`` is the file's current hash. ``passed`` is True only when the
    checked hash equals the file's hash right now.
    """
    pkl = Path(pkl)
    sha = _sha256(pkl) if pkl.exists() else None
    rec: dict = {"passed": False, "status": status or ("passed" if passed else "failed"),
                 "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sha256": sha}
    if report is not None:
        rec.update({
            "sha256": report.get("sha256") or sha,
            "python": report.get("python") or (str(report.get("python_version") or "").split(" ")[0] or None),
            "libs": report.get("libs"),
            "checker": report.get("checker"),
            "eras": report.get("n_eras"),
            "seconds_per_era_max": report.get("seconds_per_era_max"),
            "isolation": report.get("isolation"),
            "problems": list(report.get("problems") or [])[:20],
        })
    rec["passed"] = bool(passed) and rec["sha256"] is not None and rec["sha256"] == sha
    if passed and not rec["passed"]:
        rec["status"] = "failed"
        rec["problems"] = [*rec.get("problems", []), "pickle bytes differ from the checked bytes"]
    if error is not None:
        rec["error"] = str(error)[:500]
    return rec


def update_manifest(pkl: Path, passed: bool, report: dict | None = None, *, status: str | None = None,
                    error: str | None = None) -> Path | None:
    """Set ``pkl_check`` (:func:`check_record`) in ``<stem>.manifest.json`` next to the pickle (if it exists)."""
    pkl = Path(pkl)
    mpath = pkl.with_name(f"{pkl.stem}.manifest.json")
    if not mpath.exists():
        log.warning("no manifest at %s; pkl_check result not recorded", mpath)
        return None
    manifest = json.loads(mpath.read_text())
    manifest["pkl_check"] = check_record(pkl, passed, report, status=status, error=error)
    tmp = mpath.with_name(mpath.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, mpath)
    return mpath


def format_report(report: dict) -> str:
    """Human-readable multi-line summary of a checker report."""
    pyv = str(report.get("python") or report.get("python_version") or "?").split()[0]
    lines = [f"pkl_check: {'PASSED' if report.get('passed') else 'FAILED'}",
             f"  python: {report.get('checker')} ({pyv})  size: {report.get('size_mb', float('nan')):.2f} MB  eras: {report.get('n_eras')}"
             f"  slowest call: {report.get('seconds_per_era_max')}s  elapsed: {report.get('elapsed_s', 0):.1f}s",
             f"  sha256: {report.get('sha256')}"]
    iso = report.get("isolation") or {}
    libs = report.get("libs") or {}
    lines.append(f"  environment: stealth importable={iso.get('stealth_importable')} torch importable={iso.get('torch_importable')} "
                 f"lane pins verified={len(libs)}")
    bs = report.get("byte_scan") or {}
    lines.append(f"  byte scan: torch={bs.get('torch')} stealth_total={bs.get('stealth_total')} "
                 f"non-runtime stealth refs={len(bs.get('stealth_non_runtime') or [])}")
    bm = report.get("bundle_meta") or {}
    if bm:
        lines.append(f"  bundle: families={bm.get('families') or bm.get('members')} p_b={bm.get('p_b')} p_c={bm.get('p_c')} "
                     f"p_source={bm.get('p_source')}")
    st = report.get("stats") or {}
    if st:
        lines.append("  stats: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in st.items()))
    for n in report.get("notes", []):
        lines.append(f"  note: {n}")
    for p in report.get("problems", []):
        lines.append(f"  PROBLEM: {p}")
    if report.get("traceback"):
        lines.append(report["traceback"])
    if not report.get("passed") and report.get("stderr_tail"):
        lines.append("  stderr tail:\n" + report["stderr_tail"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``python -m stealth.pkl_check models/X.pkl [--venv DIR] [--venv-python PATH] [--eras N] [--data-dir D] [--seed S] [--report OUT.json]``.

    The manifest's ``pkl_check`` is replaced by a non-passing ``running`` record BEFORE anything else happens, so any
    failure on the way (missing data, a venv that cannot be built, an exception) leaves a non-passing record behind,
    never an earlier attempt's pass. Relative paths are resolved against the current directory.
    """
    parser = argparse.ArgumentParser(prog="python -m stealth.pkl_check", description=__doc__.splitlines()[0])
    parser.add_argument("pkl", type=Path)
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV, help="throwaway venv dir (created from the lane pins)")
    parser.add_argument("--venv-python", type=Path, default=None,
                        help="use this clean interpreter instead of creating the venv (verified inside the checker: exact "
                             "lane pins, no stealth, no torch)")
    parser.add_argument("--eras", type=int, default=50, help="random validation eras to test (0 = live only)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, default=None, help="also write the JSON report here")
    parser.add_argument("--timeout", type=int, default=540)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pkl = _abs(args.pkl)
    if not pkl.exists():
        log.error("%s does not exist", pkl)
        return 1
    update_manifest(pkl, False, status="running")  # invalidate any earlier pass before this attempt can fail
    try:
        python = _abs(args.venv_python) if args.venv_python else ensure_check_venv(_abs(args.venv), timeout=args.timeout)
        report = run_check(pkl, python=python, data_dir=_abs(args.data_dir), eras=args.eras, seed=args.seed,
                           timeout=args.timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, ValueError, OSError) as exc:
        log.error("pkl_check could not run: %s", exc)
        update_manifest(pkl, False, status="error", error=f"{type(exc).__name__}: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI boundary: never leave the running marker without an outcome
        log.exception("pkl_check failed")
        update_manifest(pkl, False, status="error", error=f"{type(exc).__name__}: {exc}")
        return 1
    if args.report:
        report_path = _abs(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    update_manifest(pkl, report["passed"], report)
    for line in format_report(report).splitlines():
        log.log(logging.INFO if report["passed"] else logging.ERROR, "%s", line)
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
