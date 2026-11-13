"""Validation-split predictions file for the diagnostics lane; upload only behind --upload (USER-GATED).

CLI (docs/INTERFACES.md section 10):

    python -m stealth.diag --pkl models/X.pkl [--out artifacts/diag/validation_preds.parquet] [--eras N] [--upload]

Runs the pickle era by era over the full validation split (1392 expeds), handing it the validation benchmark
(``v1_sherpa`` as ``eiq_minera_model``) exactly as the lane would, and writes a parquet with the ``id`` index and a
``prediction`` column covering every validation id (the file ``submit_validation_diagnostics`` takes). ``--eras N``
limits to the first N eras (smoke only; such a file must NOT be uploaded, the lane needs full coverage).

``--upload`` is USER-GATED (``--i-have-approval`` AND ``STEALTH_CONFIRM_DIAG=1``): the user approved at most five
diagnostics uploads, each behind a USER MUST APPROVE handoff (docs/DECISIONS.md). On upload the ``upload_id`` is
logged and written next to the file as ``<out>.upload.json``; ``--wait`` polls ``get_diagnostics_result`` to
completion and logs the panel (fit20 / unq / inov / sharpe).
Exit codes: 0 ok, 1 error / refused gate.
"""
from __future__ import annotations

import argparse
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
from stealth.predict import ENV_DIAG, BANNER, BENCH_SERIES_NAME, approval_gate, load_predict, resolve_model_id, validate_predictions

log = logging.getLogger("stealth.diag")  # explicit: under `python -m` __name__ is __main__

REPO_ROOT = Path(__file__).resolve().parents[1]
DIAG_DIR = REPO_ROOT / "artifacts" / "diag"
DEFAULT_OUT = DIAG_DIR / "validation_preds.parquet"
PROGRESS_EVERY = 200


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
    frame covers the validation ids in file order. ``eras`` limits to the first N eras (smoke only).
    """
    pkl = Path(pkl)
    predict = load_predict(pkl)
    val = sdata.load_split("validation", Path(data_dir))
    feats = sdata.feature_columns(val)
    bench = sdata.load_benchmark("validation", Path(data_dir)) if with_bench else None
    era_values = val["era"].to_numpy()
    uniq = np.unique(era_values)
    if eras is not None:
        uniq = uniq[: int(eras)]
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
    log.info("wrote %s (%d rows, %d eras)", out, len(df), len(uniq))
    return df


def upload_diagnostics(client: Any, model_id: str, path: Path, *, wait: bool = False, timeout: float = 1200.0) -> dict:
    """``submit_validation_diagnostics`` for ``path``; returns the accept dict (+ ``result`` when ``wait``)."""
    path = Path(path)
    n = len(pd.read_parquet(path, columns=[]))
    res = client.submit_validation_diagnostics(model_id, str(path), tournament="futures", target="target_everest", wait=False)
    upload_id = res.get("upload_id")
    log.warning("DIAGNOSTICS UPLOADED model=%s file=%s rows=%d upload_id=%s status=%s", model_id, path.name, n, upload_id,
                res.get("status"))
    rec = {"model_id": model_id, "upload_id": upload_id, "file": path.name, "rows": n,
           "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "accept": res}
    path.with_name(path.name + ".upload.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")
    if wait and upload_id:
        deadline = time.monotonic() + timeout
        delay = 5.0
        while time.monotonic() < deadline:
            run = client.get_diagnostics_result(upload_id)
            st = run.get("status")
            if st == "done":
                m = run.get("metrics") or run
                log.info("diagnostics done: fit20=%s unq=%s inov=%s sharpe=%s", m.get("fit20"), m.get("unq"), m.get("inov"),
                         m.get("sharpe_ratio", m.get("sharpe")))
                res["result"] = run
                break
            if st == "failed":
                raise RuntimeError(f"diagnostics run failed: {run.get('error')}")
            log.info("diagnostics %s (queue=%s elapsed=%s est=%s); waiting %.0fs", st, run.get("queue_position"),
                     run.get("elapsed_seconds"), run.get("estimate_seconds"), delay)
            time.sleep(delay)
            delay = min(delay * 1.5, 30.0)
    return res


def main(argv: list[str] | None = None) -> int:
    """CLI entry; see module docstring."""
    p = argparse.ArgumentParser(prog="python -m stealth.diag", description=__doc__.splitlines()[0])
    p.add_argument("--pkl", type=Path, required=True)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--data-dir", type=Path, default=sdata.DATA_DIR)
    p.add_argument("--eras", type=int, default=None, help="first N eras only (smoke; never upload such a file)")
    p.add_argument("--no-bench", action="store_true", help="predict without the validation benchmark")
    p.add_argument("--upload", action="store_true", help=f"USER-GATED: needs --i-have-approval + {ENV_DIAG}=1")
    p.add_argument("--i-have-approval", action="store_true")
    p.add_argument("--wait", action="store_true", help="with --upload: poll the run to completion")
    p.add_argument("--model-id", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if args.upload:
            if args.eras is not None:
                raise ValueError("--upload requires full coverage; drop --eras")
            model_id = resolve_model_id(args.model_id)
            approval_gate("upload validation diagnostics", ENV_DIAG, args.i_have_approval)
        df = make_validation_predictions(args.pkl, data_dir=args.data_dir, out=args.out, eras=args.eras, with_bench=not args.no_bench)
        if args.upload:
            client = sdata.make_client()
            log.warning(BANNER, "UPLOAD VALIDATION DIAGNOSTICS", model_id, args.out.name, ENV_DIAG)
            upload_diagnostics(client, model_id, args.out, wait=args.wait)
        log.info("diag ok: %d rows -> %s", len(df), args.out)
        return 0
    except PermissionError as exc:
        log.error("%s", exc)
        return 1
    except (RuntimeError, ValueError, FileNotFoundError, TypeError) as exc:
        log.error("%s", exc)
        return 1
    except Exception:  # noqa: BLE001 - CLI boundary
        log.exception("diag failed")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
