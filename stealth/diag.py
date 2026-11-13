"""Validation-split predictions file for the diagnostics lane; upload only behind --upload (USER-GATED).

CLI (docs/INTERFACES.md section 10; handoff 03):

    python -m stealth.diag --pkl models/X.pkl [--out artifacts/diag/<run>_validation.parquet] [--eras N]
    python -m stealth.diag --pkl models/X.pkl --out F --preflight                  # offline readiness summary
    python -m stealth.diag --pkl models/X.pkl --out F --upload --i-have-approval --wait   # + STEALTH_CONFIRM_DIAG=1
    python -m stealth.diag --out F --fetch-result [--wait]                         # read-only re-poll, never uploads

Runs the pickle era by era over the full validation split (1392 expeds, 125,931 ids), handing it the validation
benchmark (``v1_sherpa`` as ``eiq_minera_model``) exactly as the lane would, and writes a parquet with the ``id``
index and a ``prediction`` column covering every validation id (the file ``submit_validation_diagnostics`` takes),
plus ``<out>.meta.json`` (pkl sha256, rows, eras, coverage). ``--eras N`` limits to the first N eras (smoke only;
such a file is refused by ``--upload``: the lane needs full coverage).

``--preflight`` (no network) prints what an upload would send -- model slot, pkl sha256 vs its manifest and
``pkl_check``, file coverage, any earlier upload of the same file -- and exits 0 only when everything is ready, so
the operator can show it to the user before asking for approval.

``--upload`` is USER-GATED (``--i-have-approval`` AND ``STEALTH_CONFIRM_DIAG=1``): the user approved at most five
diagnostics uploads, each behind a USER MUST APPROVE handoff (docs/DECISIONS.md). It reuses ``--out`` when its
meta matches this pickle with full coverage (otherwise rebuilds it), refuses when ``<out>.upload.json`` already
records an upload (``--new-upload`` overrides; every upload consumes one of the five), logs the ``upload_id`` and
writes ``<out>.upload.json``. ``--wait`` polls ``get_diagnostics_result`` and stores the panel (fit20 / unq / inov /
sharpe / ...) in the same json. ``--attach-pkl`` (default off) also sends the pickle as ``model_pkl`` with
``model_pkl_python_version='3.11'`` -- only if the server answered that a model .pkl is required and the handoff
says so. ``--fetch-result`` re-reads the result of the recorded upload (read-only; no approval needed).

Exit codes: 0 ok, 1 error / refused gate / scoring failed, 2 uploaded (or recorded) but scoring not finished
within ``--wait-timeout`` -- never re-upload then; run ``--fetch-result --wait`` later.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stealth import data as sdata
from stealth.predict import (BANNER, BENCH_SERIES_NAME, ENV_DIAG, PKL_PYTHON_VERSION, approval_gate, load_predict,
                             resolve_model_id, validate_predictions)

log = logging.getLogger("stealth.diag")  # explicit: under `python -m` __name__ is __main__

REPO_ROOT = Path(__file__).resolve().parents[1]
DIAG_DIR = REPO_ROOT / "artifacts" / "diag"
DEFAULT_OUT = DIAG_DIR / "validation_preds.parquet"
PROGRESS_EVERY = 200
EXIT_OK, EXIT_ERROR, EXIT_PENDING = 0, 1, 2
DEFAULT_WAIT_TIMEOUT = 1800.0
BUDGET_NOTE = "each diagnostics upload consumes one of the 5 pre-registered uploads (docs/DECISIONS.md #4)"
_FAILED = {"failed", "error", "cancelled", "canceled"}

#: normalised panel name -> accepted spellings in a get_diagnostics_result payload (SDK format_diagnostics_table)
PANEL_KEYS: dict[str, tuple[str, ...]] = {
    "fit20": ("fit20", "corr20", "mean_fit"),
    "unq": ("unq", "aimc", "bmc", "mean_unq"),
    "inov": ("inov", "ncorr", "fnc", "mean_inov"),
    "sharpe": ("sharpe_ratio", "sharpe", "val_sharpe"),
    "std_dev": ("std_dev", "std"),
    "max_drawdown": ("max_drawdown",),
    "autocorrelation": ("autocorrelation",),
    "feature_exposure": ("feature_exposure",),
    "example_preds_corr": ("example_preds_corr",),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def meta_path(out: Path) -> Path:
    out = Path(out)
    return out.with_name(out.name + ".meta.json")


def upload_json_path(out: Path) -> Path:
    out = Path(out)
    return out.with_name(out.name + ".upload.json")


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, doc: dict) -> None:
    Path(path).write_text(json.dumps(doc, indent=2, default=str) + "\n")


# --------------------------------------------------------------------------------------------------
# predictions file
# --------------------------------------------------------------------------------------------------
def make_validation_predictions(
    pkl: Path,
    *,
    data_dir: Path = sdata.DATA_DIR,
    out: Path = DEFAULT_OUT,
    eras: int | None = None,
    with_bench: bool = True,
) -> pd.DataFrame:
    """Predict every validation era with the pickle; write + return the ``id``-indexed ``prediction`` frame.

    Each era is a separate ``predict`` call (one cross-section), with that era's benchmark rows when
    ``with_bench``. Every per-era output is validated (exact DataFrame, index, finite, (0, 1)); the concatenated
    frame covers the validation ids in file order. ``eras`` limits to the first N eras (smoke only). Also writes
    ``<out>.meta.json`` (pkl sha256, rows, eras, coverage) which ``--upload`` / ``--preflight`` check.
    """
    pkl = Path(pkl)
    predict = load_predict(pkl)
    val = sdata.load_split("validation", Path(data_dir))
    feats = sdata.feature_columns(val)
    bench = sdata.load_benchmark("validation", Path(data_dir)) if with_bench else None
    era_values = val["era"].to_numpy()
    all_eras = np.unique(era_values)
    uniq = all_eras if eras is None else all_eras[: int(eras)]
    log.info("diag: %d eras, %d rows, bench=%s", len(uniq), int(np.isin(era_values, uniq).sum()), with_bench)
    parts: list[pd.Series] = []
    t0 = time.perf_counter()
    bench_used = 0
    for i, e in enumerate(uniq, 1):
        fr = val.loc[era_values == e, feats]
        b = None if bench is None else bench.reindex(fr.index).rename(BENCH_SERIES_NAME)
        o = predict(fr, b)
        validate_predictions(o, fr.index)
        bench_used += int(bool(getattr(predict, "last_info", {}).get("bench_used")))
        parts.append(o.iloc[:, 0].rename("prediction"))
        if i % PROGRESS_EVERY == 0 or i == len(uniq):
            el = time.perf_counter() - t0
            log.info("diag progress %d/%d eras (%.0fs, %.2fs/era, bench_used=%d)", i, len(uniq), el, el / i, bench_used)
    pred = pd.concat(parts)
    pred = pred.reindex(val.index[np.isin(era_values, uniq)])
    assert pred.notna().all() and pred.index.is_unique
    df = pred.to_frame("prediction")
    df.index.name = "id"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, use_dictionary=False)
    meta = {"file": out.name, "pkl": pkl.name, "pkl_sha256": sha256_file(pkl), "rows": int(len(df)), "eras": int(len(uniq)),
            "eras_limit": eras, "with_bench": bool(with_bench), "bench_used_eras": int(bench_used),
            "validation_rows": int(len(val)), "validation_eras": int(len(all_eras)), "file_sha256": sha256_file(out),
            "created_at": _now()}
    _write_json(meta_path(out), meta)
    log.info("wrote %s (%d rows, %d eras) + %s", out, len(df), len(uniq), meta_path(out).name)
    return df


def file_status(out: Path, pkl: Path) -> tuple[bool, str, dict | None]:
    """``(ready, reason, meta)``: ready iff ``out`` + its meta exist, the meta's pkl sha256 equals ``pkl``'s, the file
    is unchanged since, the benchmark was used, and coverage is the full validation split."""
    out, pkl = Path(out), Path(pkl)
    meta = _read_json(meta_path(out))
    if not out.exists():
        return False, f"{out.name} does not exist", meta
    if meta is None:
        return False, f"{meta_path(out).name} missing (file not built by this tool)", None
    if not pkl.exists():
        return False, f"{pkl} does not exist", meta
    if meta.get("pkl_sha256") != sha256_file(pkl):
        return False, f"{out.name} was built from a different pickle (sha {str(meta.get('pkl_sha256'))[:12]})", meta
    if meta.get("file_sha256") and meta["file_sha256"] != sha256_file(out):
        return False, f"{out.name} changed after it was built", meta
    if meta.get("eras_limit") is not None or meta.get("rows") != meta.get("validation_rows") \
            or meta.get("eras") != meta.get("validation_eras"):
        return False, (f"partial coverage: rows {meta.get('rows')}/{meta.get('validation_rows')}, eras "
                       f"{meta.get('eras')}/{meta.get('validation_eras')} (an --eras smoke file)"), meta
    if not meta.get("with_bench"):
        return False, "built with --no-bench (the lane hands predict the benchmark)", meta
    return True, "full coverage, built from this pickle", meta


# --------------------------------------------------------------------------------------------------
# platform: upload (USER-GATED) and read-only result polling
# --------------------------------------------------------------------------------------------------
def panel_from_result(run: dict | None) -> dict:
    """Normalised ``{fit20, unq, inov, sharpe, ...}`` from a ``get_diagnostics_result`` payload (top level, or under
    ``metrics`` / ``panel`` / ``result``); missing values are None. Only these numbers are ever surfaced."""
    run = run if isinstance(run, dict) else {}
    sources = [s for s in (run.get("metrics"), run.get("panel"), run.get("result"), run) if isinstance(s, dict)]
    out: dict[str, Any] = {"status": run.get("status")}
    for name, aliases in PANEL_KEYS.items():
        val = None
        for src in sources:
            for a in aliases:
                v = src.get(a)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    val = float(v)
                    break
            if val is not None:
                break
        out[name] = val
    return out


def _poll(client: Any, upload_id: str, timeout: float) -> dict:
    """Poll ``get_diagnostics_result`` until done / failed or ``timeout`` seconds (0 = one read). Returns the last run."""
    deadline = time.monotonic() + max(float(timeout), 0.0)
    delay = 5.0
    while True:
        run = client.get_diagnostics_result(upload_id) or {}
        st = str(run.get("status") or "").lower()
        if st == "done" or st in _FAILED:
            return run
        if time.monotonic() + delay > deadline:
            return run
        log.info("diagnostics %s (queue=%s elapsed=%s est=%s); waiting %.0fs", st or "?", run.get("queue_position"),
                 run.get("elapsed_seconds"), run.get("estimate_seconds"), delay)
        time.sleep(delay)
        delay = min(delay * 1.5, 30.0)


def _record_result(rec: dict, run: dict, path: Path) -> int:
    """Store ``run`` + its panel in the upload record; returns the exit code it implies (0 done, 1 failed, 2 pending)."""
    st = str(run.get("status") or "").lower()
    rec["result"], rec["panel"], rec["result_at"] = run, panel_from_result(run), _now()
    _write_json(path, rec)
    p = rec["panel"]
    if st == "done":
        log.info("diagnostics done: fit20=%s unq=%s inov=%s sharpe=%s (upload_id=%s)", p["fit20"], p["unq"], p["inov"],
                 p["sharpe"], rec.get("upload_id"))
        return EXIT_OK
    if st in _FAILED:
        log.error("diagnostics run %s: %s (upload_id=%s)", st, run.get("error") or run.get("error_message"),
                  rec.get("upload_id"))
        return EXIT_ERROR
    log.warning("diagnostics not finished (status=%s, upload_id=%s). Do NOT upload again: run "
                "`python -m stealth.diag --out <same file> --fetch-result --wait` later.", st or "?", rec.get("upload_id"))
    return EXIT_PENDING


def _pkl_required(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return ".pkl" in msg and "required" in msg


def upload_diagnostics(client: Any, model_id: str, path: Path, *, wait: bool = False,
                       timeout: float = DEFAULT_WAIT_TIMEOUT, model_pkl: Path | None = None) -> dict:
    """``submit_validation_diagnostics`` for ``path`` (+ ``model_pkl`` when attached); writes ``<path>.upload.json``.

    Returns the accept dict (+ ``result`` and ``exit_code`` when ``wait``). A server answer that a model .pkl is
    required raises ``RuntimeError('MODEL PKL REQUIRED ...')``: the handoff stops and reports, nothing is retried.
    """
    path = Path(path)
    n = len(pd.read_parquet(path, columns=[]))
    kwargs: dict[str, Any] = {}
    if model_pkl is not None:
        kwargs = {"model_pkl": str(model_pkl), "model_pkl_python_version": PKL_PYTHON_VERSION}
    try:
        res = client.submit_validation_diagnostics(model_id, str(path), tournament="futures", target="target_everest",
                                                   wait=False, **kwargs) or {}
    except Exception as exc:  # noqa: BLE001 - the SDK raises its own error types
        if _pkl_required(exc) and model_pkl is None:
            raise RuntimeError(f"MODEL PKL REQUIRED: the server refused the upload without a model .pkl ({exc}). "
                               "Stop and report; do not retry.") from exc
        raise
    upload_id = res.get("upload_id")
    log.warning("DIAGNOSTICS UPLOADED model=%s file=%s rows=%d upload_id=%s status=%s attached_pkl=%s", model_id,
                path.name, n, upload_id, res.get("status"), model_pkl is not None)
    meta = _read_json(meta_path(path)) or {}
    rec: dict[str, Any] = {"model_id": model_id, "upload_id": upload_id, "file": path.name, "rows": n,
                           "file_sha256": sha256_file(path), "built_from_pkl_sha256": meta.get("pkl_sha256"),
                           "attached_pkl": model_pkl is not None,
                           "pkl_sha256": None if model_pkl is None else sha256_file(Path(model_pkl)),
                           "uploaded_at": _now(), "accept": res}
    jpath = upload_json_path(path)
    _write_json(jpath, rec)
    if wait and upload_id:
        run = _poll(client, upload_id, timeout)
        res["result"] = run
        res["exit_code"] = _record_result(rec, run, jpath)
    return res


def fetch_result(client: Any, out: Path, *, wait: bool = False,
                 timeout: float = DEFAULT_WAIT_TIMEOUT) -> tuple[dict, int]:
    """Re-read the result of the upload recorded in ``<out>.upload.json`` (read-only); returns (record, exit code)."""
    jpath = upload_json_path(out)
    rec = _read_json(jpath)
    if not rec or not rec.get("upload_id"):
        raise FileNotFoundError(f"{jpath.name} missing or without an upload_id: nothing was uploaded for {Path(out).name}")
    run = _poll(client, rec["upload_id"], timeout if wait else 0.0)
    return rec, _record_result(rec, run, jpath)


def preflight(pkl: Path, out: Path, model_id_arg: str | None = None) -> tuple[bool, list[str]]:
    """Offline readiness summary of an upload (no network). Returns (ready, lines)."""
    pkl, out = Path(pkl), Path(out)
    ok, lines = True, ["DIAG PREFLIGHT (offline; nothing is sent)"]
    try:
        lines.append(f"  model_id        {resolve_model_id(model_id_arg)}")
    except RuntimeError:
        ok = False
        lines.append("  model_id        MISSING (EIQ_MODEL_ID not set in .env) -> HUMAN STEP")
    if pkl.exists():
        sha = sha256_file(pkl)
        man = _read_json(pkl.with_name(f"{pkl.stem}.manifest.json"))
        if man is None:
            ok = False
            lines.append(f"  pkl             {pkl.name} sha256 {sha[:16]}  manifest MISSING")
        else:
            match = man.get("sha256") == sha
            passed = bool((man.get("pkl_check") or {}).get("passed"))
            py = str(man.get("python") or "?")
            ok = ok and match and passed and py.startswith("3.11")
            lines.append(f"  pkl             {pkl.name} sha256 {sha[:16]}  manifest {'match' if match else 'MISMATCH'}, "
                         f"pkl_check {'passed' if passed else 'NOT passed'}, python {py}, run {man.get('run_id')}")
    else:
        ok = False
        lines.append(f"  pkl             {pkl} MISSING")
    ready, reason, meta = file_status(out, pkl)
    ok = ok and ready
    rows = (f"rows {meta.get('rows')}/{meta.get('validation_rows')} eras {meta.get('eras')}/{meta.get('validation_eras')}"
            if meta else "")
    lines.append(f"  file            {out.name} {rows}  -> {'READY' if ready else 'NOT READY'}: {reason}")
    prev = _read_json(upload_json_path(out))
    if prev and prev.get("upload_id"):
        ok = False
        lines.append(f"  previous upload ALREADY UPLOADED upload_id={prev['upload_id']} at {prev.get('uploaded_at')} "
                     "-> use --fetch-result, do not upload again")
    else:
        lines.append("  previous upload none")
    lines.append(f"  budget          {BUDGET_NOTE}")
    lines.append("PREFLIGHT OK -- ask the user for approval in chat" if ok else "PREFLIGHT NOT READY -- do not upload")
    return ok, lines


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def _upload_file(args: argparse.Namespace) -> pd.DataFrame:
    """The validation file an upload sends: reused when built from this pickle with full coverage, else rebuilt."""
    ready, reason, _ = file_status(args.out, args.pkl)
    if ready:
        log.info("reusing %s (%s)", args.out, reason)
        return pd.read_parquet(args.out)
    log.info("building %s (%s)", args.out, reason)
    df = make_validation_predictions(args.pkl, data_dir=args.data_dir, out=args.out, with_bench=not args.no_bench)
    ready, reason, _ = file_status(args.out, args.pkl)
    if not ready:
        raise RuntimeError(f"{args.out.name} not uploadable: {reason}")
    return df


def main(argv: list[str] | None = None) -> int:
    """CLI entry; see module docstring."""
    p = argparse.ArgumentParser(prog="python -m stealth.diag", description=__doc__.splitlines()[0])
    p.add_argument("--pkl", type=Path, default=None, help="the pickle (required except with --fetch-result)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--data-dir", type=Path, default=sdata.DATA_DIR)
    p.add_argument("--eras", type=int, default=None, help="first N eras only (smoke; never upload such a file)")
    p.add_argument("--no-bench", action="store_true", help="predict without the validation benchmark")
    p.add_argument("--preflight", action="store_true", help="offline readiness summary of an upload; sends nothing")
    p.add_argument("--upload", action="store_true", help=f"USER-GATED: needs --i-have-approval + {ENV_DIAG}=1")
    p.add_argument("--i-have-approval", action="store_true")
    p.add_argument("--wait", action="store_true", help="with --upload / --fetch-result: poll the run to completion")
    p.add_argument("--wait-timeout", type=float, default=DEFAULT_WAIT_TIMEOUT)
    p.add_argument("--model-id", default=None)
    p.add_argument("--attach-pkl", action="store_true", help="also send the pickle as model_pkl (python 3.11); default off")
    p.add_argument("--new-upload", action="store_true", help="upload although <out>.upload.json records one (uses budget)")
    p.add_argument("--fetch-result", action="store_true", help="read-only: re-poll the upload recorded next to --out")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if args.fetch_result:
            rec, code = fetch_result(sdata.make_client(), args.out, wait=args.wait, timeout=args.wait_timeout)
            log.info("fetch-result: upload_id=%s panel=%s", rec.get("upload_id"), rec.get("panel"))
            return code
        if args.pkl is None:
            raise ValueError("--pkl is required (except with --fetch-result)")
        if args.preflight:
            ready, lines = preflight(args.pkl, args.out, args.model_id)
            sys.stdout.write("\n".join(lines) + "\n")
            return EXIT_OK if ready else EXIT_ERROR
        if not args.upload:
            df = make_validation_predictions(args.pkl, data_dir=args.data_dir, out=args.out, eras=args.eras,
                                             with_bench=not args.no_bench)
            log.info("diag ok: %d rows -> %s", len(df), args.out)
            return EXIT_OK
        if args.eras is not None:
            raise ValueError("--upload requires full coverage; drop --eras")
        model_id = resolve_model_id(args.model_id)
        approval_gate("upload validation diagnostics", ENV_DIAG, args.i_have_approval)
        prev = _read_json(upload_json_path(args.out))
        if prev and prev.get("upload_id") and not args.new_upload:
            raise RuntimeError(f"{args.out.name} was already uploaded (upload_id={prev['upload_id']}); use --fetch-result. "
                               f"Uploading again needs --new-upload and the user's approval: {BUDGET_NOTE}")
        df = _upload_file(args)
        client = sdata.make_client()
        log.warning(BANNER, "UPLOAD VALIDATION DIAGNOSTICS", model_id, args.out.name, ENV_DIAG)
        res = upload_diagnostics(client, model_id, args.out, wait=args.wait, timeout=args.wait_timeout,
                                 model_pkl=args.pkl if args.attach_pkl else None)
        log.info("diag upload done: %d rows, upload_id=%s (%s)", len(df), res.get("upload_id"), upload_json_path(args.out).name)
        return int(res.get("exit_code", EXIT_OK))
    except PermissionError as exc:
        log.error("%s", exc)
        return EXIT_ERROR
    except (RuntimeError, ValueError, FileNotFoundError, TypeError) as exc:
        log.error("%s", exc)
        return EXIT_ERROR
    except Exception:  # noqa: BLE001 - CLI boundary
        log.exception("diag failed")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
