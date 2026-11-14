"""Daily path: round check -> download live + benchmark -> predict -> validate -> (--submit | --upload) -> status.

CLI (docs/INTERFACES.md section 10):

    python -m stealth.predict --pkl models/X.pkl [--offline] [--submit] [--upload] [--data-dir D] [--model-id M]

* Dry run by default: checks the round is open (SDK ``get_current_round``; skipped with ``--offline``), downloads
  the live split + live benchmark into ``data_dir`` (``--offline`` reads what is already there), runs the pickle,
  validates (coverage == live ids, finite, strictly inside (0, 1)), writes ``artifacts/predictions/<exped>.parquet``
  and, online, calls the platform's free ``validate_submission`` pre-flight.
* ``--submit`` and ``--upload`` are USER-GATED: both refuse unless the ``--i-have-approval`` flag is given AND the
  matching environment variable is ``1`` (``STEALTH_CONFIRM_SUBMIT`` / ``STEALTH_CONFIRM_UPLOAD``; ``STEALTH_APPROVED``
  is accepted for either). A banner is logged before any write.
* online, the live file's ``exped`` must equal the round's ``exped`` (hard stop otherwise: a stale live file must never
  be submitted).
* ``--submit`` sends the literal ``exped`` read from the live file (never "current") and logs ``replaced`` /
  ``late`` / ``scored`` plus the recorded coverage. ``--upload`` uploads the pickle to the model slot with
  ``python_version="3.11"``, polls ``get_upload_status`` until ``sandbox_ready == 1`` (or an ``error_message``), then
  records ``models/active.json`` and the manifest's ``upload_id``. It refuses without a manifest next to the pickle,
  without ``manifest.pkl_check.passed``, on a sha mismatch, or when the manifest's ``python`` is not 3.11.x.

Exit codes: 0 ok, 3 submission window closed (incl. the SDK's 'no active round' 404), 1 error (incl. refused gates,
validation failures, live/round exped mismatch).
Needs ``EIQ_API_KEY`` for anything online (``RuntimeError`` otherwise); ``--offline`` never touches the network.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from stealth import data as sdata

log = logging.getLogger("stealth.predict")  # explicit: under `python -m` __name__ is __main__

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = REPO_ROOT / "artifacts" / "predictions"
MODELS_DIR = REPO_ROOT / "models"
ACTIVE_JSON = MODELS_DIR / "active.json"
PKL_PYTHON_VERSION = "3.11"
BENCH_SERIES_NAME = "eiq_minera_model"
EXIT_OK, EXIT_ERROR, EXIT_WINDOW_CLOSED = 0, 1, 3
ENV_SUBMIT, ENV_UPLOAD, ENV_DIAG, ENV_ANY = "STEALTH_CONFIRM_SUBMIT", "STEALTH_CONFIRM_UPLOAD", "STEALTH_CONFIRM_DIAG", "STEALTH_APPROVED"

BANNER = (
    "=================================================================\n"
    "  USER-GATED PLATFORM WRITE: %s\n"
    "  model_id=%s  target=%s\n"
    "  This is irreversible on the platform side. Proceeding because\n"
    "  --i-have-approval was passed AND %s=1 is set.\n"
    "================================================================="
)


# --------------------------------------------------------------------------------------------------
# gating / config
# --------------------------------------------------------------------------------------------------
def approval_gate(action: str, env_var: str, flag: bool) -> None:
    """Refuse (``PermissionError``) unless ``flag`` is True AND ``env_var`` (or ``STEALTH_APPROVED``) == "1".

    Nothing is written to the platform without a handoff carrying a USER MUST APPROVE banner (docs/DECISIONS.md).
    """
    env_ok = os.environ.get(env_var) == "1" or os.environ.get(ENV_ANY) == "1"
    if not flag or not env_ok:
        raise PermissionError(
            f"{action} refused: requires --i-have-approval AND {env_var}=1 in the environment "
            f"(flag={'set' if flag else 'missing'}, env={'ok' if env_ok else 'missing'}). "
            "Only run this from a handoff carrying the USER MUST APPROVE banner."
        )


def resolve_model_id(cli_value: str | None) -> str:
    """``--model-id`` else ``EIQ_MODEL_ID`` from the environment or ``.env``; raises RuntimeError when absent."""
    if cli_value:
        return cli_value
    mid = os.environ.get("EIQ_MODEL_ID") or sdata._read_dotenv(sdata.DOTENV_PATH).get("EIQ_MODEL_ID")
    if not mid:
        raise RuntimeError("EIQ_MODEL_ID not set (pass --model-id or add it to .env)")
    return mid


# --------------------------------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------------------------------
def load_predict(pkl: Path) -> Callable[..., pd.DataFrame]:
    """Unpickle the cloudpickled ``predict`` callable (plain ``pickle.load`` suffices for cloudpickle output)."""
    pkl = Path(pkl)
    if not pkl.exists():
        raise FileNotFoundError(f"{pkl} does not exist; build it with `python -m stealth.bundle`")
    with pkl.open("rb") as fh:
        predict = pickle.load(fh)
    if not callable(predict):
        raise TypeError(f"{pkl} does not contain a callable (got {type(predict).__name__})")
    return predict


def live_exped(live: pd.DataFrame) -> str:
    """The single literal ``exped`` of a live frame (ValueError if absent or not unique)."""
    if "exped" not in live.columns:
        raise ValueError("live frame has no 'exped' column")
    vals = live["exped"].astype(str).unique().tolist()
    if len(vals) != 1:
        raise ValueError(f"live frame spans {len(vals)} expeds: {vals[:5]}")
    return vals[0]


def validate_predictions(out: pd.DataFrame, live_index: pd.Index) -> dict[str, Any]:
    """Local pre-flight: exact DataFrame, one float column on ``live_index``, finite, strictly in (0, 1).

    Returns summary stats; raises ``ValueError`` listing every problem.
    """
    problems: list[str] = []
    if type(out) is not pd.DataFrame:
        raise ValueError(f"predict returned {type(out).__name__}, not pandas.DataFrame")
    if out.shape[1] != 1:
        problems.append(f"{out.shape[1]} columns (want 1)")
    if out.shape[1] == 0:
        raise ValueError("prediction validation failed: " + "; ".join(problems))
    if not out.index.equals(live_index):
        problems.append("index != live ids")
    if out.index.has_duplicates:
        problems.append("duplicate ids")
    col = out.iloc[:, 0]
    # same rule as pkl_check: a NumPy float column only (numeric strings, ints, pandas extension dtypes are refused)
    if not (isinstance(col.dtype, np.dtype) and np.issubdtype(col.dtype, np.floating)):
        raise ValueError("prediction validation failed: " + "; ".join([*problems, f"dtype {col.dtype} is not float"]))
    v = col.to_numpy(dtype=np.float64)
    if not np.isfinite(v).all():
        problems.append(f"{int((~np.isfinite(v)).sum())} non-finite values")
    elif v.size and not ((v > 0.0).all() and (v < 1.0).all()):
        problems.append(f"values outside (0,1): min={v.min():.6f} max={v.max():.6f}")
    if problems:
        raise ValueError("prediction validation failed: " + "; ".join(problems))
    return {"n": int(v.size), "min": float(v.min()) if v.size else None, "max": float(v.max()) if v.size else None,
            "mean": float(v.mean()) if v.size else None, "n_unique": int(np.unique(v).size)}


def predictions_dict(out: pd.DataFrame) -> dict[str, float]:
    """``{instrument_id: prediction}`` with native floats, as the SDK wants."""
    return {str(k): float(v) for k, v in out.iloc[:, 0].items()}


def read_live(data_dir: Path) -> tuple[pd.DataFrame, pd.Series | None]:
    """Live features (+ ``exped``) and the live benchmark as ``eiq_minera_model`` (None if the file is absent)."""
    live = sdata.load_split("live", data_dir)
    bench: pd.Series | None = None
    if sdata.benchmark_path("live", data_dir).exists():
        bench = sdata.load_benchmark("live", data_dir).reindex(live.index).rename(BENCH_SERIES_NAME)
    else:
        log.warning("no live benchmark file; predicting without benchmark neutralisation")
    return live, bench


def _is_no_round_error(exc: Exception) -> bool:
    """True for the SDK's 'no active round' failure (``EverestError`` 404 / a 'no active round' body)."""
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return status == 404 or "no active round" in text or "no current round" in text


def check_round_open(client: Any) -> dict:
    """``get_current_round('futures')``; raises ``RuntimeError`` unless ``submission_window == 'open'``.

    When no active round exists (weekend / between rounds) the SDK raises ``EverestError`` 404; that is mapped
    to the same "submission window is not open" ``RuntimeError`` so the CLI exits ``EXIT_WINDOW_CLOSED``.
    """
    try:
        rnd = client.get_current_round("futures")
    except Exception as exc:  # noqa: BLE001 - only the no-round case is translated; anything else propagates
        if _is_no_round_error(exc):
            raise RuntimeError(f"submission window is not open (no active round: {str(exc)[:200]})") from exc
        raise
    if not isinstance(rnd, dict):
        raise RuntimeError(f"submission window is not open (unexpected get_current_round payload: {type(rnd).__name__})")
    window = rnd.get("submission_window") or rnd.get("status")
    log.info("current round: number=%s exped=%s window=%s close=%s", rnd.get("round_number"), rnd.get("exped"),
             window, rnd.get("round_close") or rnd.get("close_at"))
    if window != "open":
        raise RuntimeError(f"submission window is not open (submission_window={window!r}); next window: "
                           f"{rnd.get('round_open')} .. {rnd.get('round_close')}")
    return rnd


def download_live(client: Any, data_dir: Path) -> None:
    """Refresh ``futures_live.parquet`` and ``futures_live_benchmark_models.parquet`` in ``data_dir`` via the SDK."""
    data_dir.mkdir(parents=True, exist_ok=True)
    client.download_dataset(universe="futures", split="live", output_path=str(sdata.split_path("live", data_dir)))
    try:
        client.download_benchmark(universe="futures", split="live", output_path=str(sdata.benchmark_path("live", data_dir)))
    except Exception as exc:  # noqa: BLE001 - benchmark is optional at predict time
        log.warning("live benchmark download failed (%s); continuing without it", exc)


def predict_live(
    pkl: Path,
    *,
    data_dir: Path = sdata.DATA_DIR,
    offline: bool = False,
    client: Any = None,
    out_dir: Path = PREDICTIONS_DIR,
) -> dict[str, Any]:
    """Run the pickle on the live split; returns ``{exped, n, path, predictions, stats, round}``.

    Online (``offline=False``): requires the round to be open (``RuntimeError`` otherwise), downloads live + bench
    first, and runs the platform ``validate_submission`` pre-flight (``ValueError`` on reported issues). Offline:
    reads ``data_dir`` as is. ``client`` defaults to ``stealth.data.make_client()`` (needs EIQ_API_KEY).
    """
    pkl = Path(pkl)
    rnd: dict = {}
    if not offline:
        client = client or sdata.make_client()
        rnd = check_round_open(client)
        download_live(client, Path(data_dir))
    live, bench = read_live(Path(data_dir))
    exped = live_exped(live)
    if rnd.get("exped") and str(rnd["exped"]) != exped:
        # PLAN "Daily operations": proceed only if the live parquet's exped equals the round's exped. A stale or
        # partially refreshed live file would otherwise yield a late/unscored or 409 submission with no error.
        raise RuntimeError(f"live exped {exped} != round exped {rnd['exped']}; refusing to submit "
                           "(stale futures_live.parquet? re-run after the round data is published)")
    predict = load_predict(pkl)
    t0 = time.perf_counter()
    out = predict(live, bench)
    stats = validate_predictions(out, live.index)
    stats["seconds"] = time.perf_counter() - t0
    info = getattr(predict, "last_info", {})
    log.info("predicted exped=%s n=%d in %.2fs bench_used=%s core_used=%s unique=%d", exped, stats["n"], stats["seconds"],
             info.get("bench_used"), info.get("core_used"), stats["n_unique"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{exped}.parquet"
    out.rename(columns={out.columns[0]: "prediction"}).to_parquet(path)
    preds = predictions_dict(out)
    if not offline:
        res = client.validate_submission(preds, exped=exped)
        issues = res.get("issues") or res.get("errors") or []
        if issues or res.get("valid") is False:
            raise ValueError(f"platform validate_submission reported issues: {issues}")
        log.info("platform validate_submission: ok (%s)", {k: v for k, v in res.items() if k != "issues"})
    return {"exped": exped, "n": stats["n"], "path": path, "predictions": preds, "stats": stats, "round": rnd}


# --------------------------------------------------------------------------------------------------
# platform writes (gated by the caller)
# --------------------------------------------------------------------------------------------------
def submit(client: Any, model_id: str, preds: dict[str, float], exped: str) -> dict:
    """``submit_futures_predictions`` with the literal exped; logs replaced/late/scored and the recorded coverage."""
    res = client.submit_futures_predictions(model_id, preds, exped=exped)
    log.warning("SUBMITTED model=%s exped=%s n=%s replaced=%s late=%s scored=%s submission_id=%s", model_id, exped,
                res.get("instrument_count"), res.get("replaced"), res.get("late"), res.get("scored", "n/a"),
                res.get("submission_id"))
    if res.get("late") and res.get("scored") is False:
        log.error("submission recorded as LATE and will NOT be scored")
    try:
        status = client.get_submission_status(tournament="futures", round=exped, model_id=model_id)
        log.info("submission status: %s", json.dumps(status, default=str)[:800])
        res["status"] = status
    except Exception as exc:  # noqa: BLE001 - status is informational
        log.warning("get_submission_status failed: %s", exc)
    return res


def poll_upload_status(client: Any, model_id: str, *, poll_seconds: float = 10.0, timeout: float = 900.0) -> dict:
    """Poll ``get_upload_status`` until ``sandbox_ready == 1`` (returns it) or ``error_message`` / timeout (RuntimeError)."""
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        last = client.get_upload_status(model_id) or {}
        if last.get("error_message"):
            raise RuntimeError(f"upload sandbox smoke FAILED: {last['error_message']}")
        if int(last.get("sandbox_ready") or 0) == 1:
            log.info("upload sandbox_ready=1 status=%s lane_active=%s", last.get("status"), last.get("lane_active"))
            return last
        log.info("upload smoke still running (sandbox_ready=%s status=%s); waiting %.0fs", last.get("sandbox_ready"),
                 last.get("status"), poll_seconds)
        time.sleep(poll_seconds)
    raise RuntimeError(f"upload status did not reach sandbox_ready=1 within {timeout:.0f}s (last={last})")


def write_active(model_id: str, upload_id: str | None, sha256: str, pkl: Path, path: Path = ACTIVE_JSON) -> Path:
    """Write ``models/active.json`` ``{model_id, upload_id, sha256, pkl, uploaded_at}`` (committed by the orchestrator)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"model_id": model_id, "upload_id": upload_id, "sha256": sha256, "pkl": Path(pkl).name,
           "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path.write_text(json.dumps(rec, indent=2) + "\n")
    return path


def check_upload_artifact(pkl: Path) -> dict[str, Any]:
    """THE artifact gate of every upload path (``predict --upload``, ``diag --upload``, ``diag --attach-pkl``).

    Raises ``RuntimeError`` naming the failed condition unless ALL hold: the pickle exists; ``<stem>.manifest.json``
    exists next to it; the manifest's ``sha256`` equals the file's; ``manifest.pkl_check.passed`` is true AND its
    ``sha256`` equals the file's (a pass recorded for other bytes, or a legacy pass without a hash, does not count);
    the manifest's ``python`` (and the checker's, when recorded) is 3.11.x, the version every upload declares.
    Purely local: callers run it BEFORE creating a client, so a refusal sends nothing.
    Returns ``{"pkl", "sha256", "manifest", "manifest_path"}``.
    """
    from stealth.bundle import manifest_path, sha256_file

    pkl = Path(pkl)
    if not pkl.is_file():
        raise RuntimeError(f"{pkl} does not exist; build it with `python -m stealth.bundle`")
    sha = sha256_file(pkl)
    mpath = manifest_path(pkl)
    if not mpath.exists():
        # pkl_check is the final gate before any upload (PLAN "Packaging"); no manifest = no gate record.
        raise RuntimeError(f"no manifest next to {pkl.name} ({mpath.name} missing); run `python -m stealth.bundle` "
                           "and `python -m stealth.pkl_check` first")
    try:
        manifest = json.loads(mpath.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{mpath.name} is unreadable ({exc}); rebuild it with `python -m stealth.bundle`") from exc
    if manifest.get("sha256") != sha:
        raise RuntimeError(f"{pkl.name} sha256 {sha[:12]} != manifest {str(manifest.get('sha256'))[:12]}; rebuild the manifest")
    check = manifest.get("pkl_check") or {}
    if not check.get("passed"):
        raise RuntimeError(f"manifest.pkl_check has not passed (status {check.get('status', 'none')}); run "
                           "`python -m stealth.pkl_check` first")
    if check.get("sha256") != sha:
        raise RuntimeError(f"manifest.pkl_check passed for bytes {str(check.get('sha256'))[:12]}, not for {pkl.name} "
                           f"({sha[:12]}); re-run `python -m stealth.pkl_check` on this pickle")
    built_py = str(manifest.get("python") or "")
    if not built_py.startswith(PKL_PYTHON_VERSION + "."):
        raise RuntimeError(f"manifest says the pickle was built under Python {built_py or '?'}, but the upload declares "
                           f"python_version={PKL_PYTHON_VERSION}; rebuild the pickle in the 3.11 train venv")
    checked_py = check.get("python")
    if checked_py is not None and not str(checked_py).startswith(PKL_PYTHON_VERSION + "."):
        raise RuntimeError(f"pkl_check ran under Python {checked_py}, not {PKL_PYTHON_VERSION}.x; re-run it in the lane venv")
    return {"pkl": pkl, "sha256": sha, "manifest": manifest, "manifest_path": mpath}


def upload(client: Any, model_id: str, pkl: Path, *, poll_seconds: float = 10.0, timeout: float = 900.0,
           active_path: Path = ACTIVE_JSON) -> dict:
    """Upload the pickle (``python_version="3.11"``), poll until sandbox-ready, record active.json + manifest upload_id.

    :func:`check_upload_artifact` runs first; nothing reaches ``client`` when it refuses."""
    gate = check_upload_artifact(pkl)
    pkl, sha, mpath, manifest = Path(pkl), gate["sha256"], gate["manifest_path"], gate["manifest"]
    res = client.upload_model(model_id, str(pkl), python_version=PKL_PYTHON_VERSION)
    upload_id = res.get("upload_id") or res.get("id")
    log.warning("UPLOADED %s (%.2f MB, sha256 %s) to model %s: upload_id=%s", pkl.name, pkl.stat().st_size / 1e6, sha[:12],
                model_id, upload_id)
    status = poll_upload_status(client, model_id, poll_seconds=poll_seconds, timeout=timeout)
    upload_id = upload_id or status.get("upload_id") or status.get("id")
    write_active(model_id, upload_id, sha, pkl, active_path)
    manifest["upload_id"] = upload_id
    manifest["uploaded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("recorded %s (upload_id=%s)", active_path, upload_id)
    return {"upload": res, "status": status, "upload_id": upload_id, "sha256": sha}


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m stealth.predict", description=__doc__.splitlines()[0])
    p.add_argument("--pkl", type=Path, required=True)
    p.add_argument("--offline", action="store_true", help="no network: read data/ as is, skip round check + platform validate")
    p.add_argument("--data-dir", type=Path, default=sdata.DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=PREDICTIONS_DIR)
    p.add_argument("--model-id", default=None, help="platform model slot (default: EIQ_MODEL_ID)")
    p.add_argument("--submit", action="store_true", help=f"USER-GATED: submit predictions (needs --i-have-approval AND {ENV_SUBMIT}=1 or {ENV_ANY}=1 in the environment)")
    p.add_argument("--upload", action="store_true", help=f"USER-GATED: upload the pkl to the model slot (needs --i-have-approval AND {ENV_UPLOAD}=1 or {ENV_ANY}=1 in the environment)")
    p.add_argument("--i-have-approval", action="store_true", help="assert a USER MUST APPROVE handoff covers this write")
    p.add_argument("--poll-seconds", type=float, default=10.0)
    p.add_argument("--poll-timeout", type=float, default=900.0)
    p.add_argument("--active", type=Path, default=ACTIVE_JSON, help="where --upload records the accepted upload (models/active.json)")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry; see module docstring for the flow and exit codes."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if (args.submit or args.upload) and args.offline:
        log.error("--submit/--upload cannot be combined with --offline")
        return EXIT_ERROR
    try:
        model_id = resolve_model_id(args.model_id) if (args.submit or args.upload) else (args.model_id or "n/a")
        if args.submit:
            approval_gate("submit predictions", ENV_SUBMIT, args.i_have_approval)
        if args.upload:
            approval_gate("upload model", ENV_UPLOAD, args.i_have_approval)
            check_upload_artifact(args.pkl)  # local artifact gate before any client exists (upload() repeats it)
        client = None if args.offline else sdata.make_client()
        try:
            result = predict_live(args.pkl, data_dir=args.data_dir, offline=args.offline, client=client, out_dir=args.out_dir)
        except RuntimeError as exc:
            if "submission window is not open" in str(exc):
                log.warning("%s", exc)
                return EXIT_WINDOW_CLOSED
            raise
        log.info("dry run ok: exped=%s n=%d -> %s", result["exped"], result["n"], result["path"])
        if args.submit:
            log.warning(BANNER, "SUBMIT PREDICTIONS", model_id, result["exped"], ENV_SUBMIT)
            submit(client, model_id, result["predictions"], result["exped"])
        if args.upload:
            log.warning(BANNER, "UPLOAD MODEL PICKLE", model_id, args.pkl.name, ENV_UPLOAD)
            upload(client, model_id, args.pkl, poll_seconds=args.poll_seconds, timeout=args.poll_timeout, active_path=args.active)
        return EXIT_OK
    except PermissionError as exc:
        log.error("%s", exc)
        return EXIT_ERROR
    except (RuntimeError, ValueError, FileNotFoundError, TypeError) as exc:
        log.error("%s", exc)
        return EXIT_ERROR
    except Exception:  # noqa: BLE001 - CLI boundary
        log.exception("predict failed")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
