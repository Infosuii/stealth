"""Monitor: today's exped has an accepted submission; the lane's active upload matches ``models/active.json`` /
the manifest sha; otherwise write ``handoffs/ALERT.md`` and exit 1.

CLI (docs/INTERFACES.md section 10):

    python -m stealth.monitor [--offline] [--scores] [--model-id M] [--pkl models/X.pkl]

Checks
1. local consistency (always): ``<stem>.manifest.json`` exists, ``<stem>.pkl.sha256`` equals its ``sha256``, the
   pickle (if present) hashes to it, ``manifest.pkl_check.passed`` is true, and ``models/active.json`` (if present)
   carries the same ``sha256``;
2. online: ``get_upload_status(model_id)`` is ``sandbox_ready=1`` without ``error_message`` and, where the response
   exposes an upload id / sha (field names are not published -- read defensively, (unverified)), it equals
   ``active.json``;
3. online: ``get_current_round`` -> if a round is open/known for today, ``get_submission_status(round=exped)`` holds
   an accepted entry with full coverage. Before 13:30 UTC on an open round this is informational only (the lane
   fires at ~13:02 UTC); after that, a missing entry is an alert;
4. ``--scores``: logs ``get_scores(model_id, days=30)`` (informational).

``--offline`` runs only check 1 (no network, no API key needed). Exit 0 = all good, 1 = alert written, 2 = cannot
run (no model id / no manifest). The alert file is written; committing/pushing it is the orchestrator's job.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stealth import data as sdata
from stealth.bundle import manifest_path, sha256_file
from stealth.predict import ACTIVE_JSON, MODELS_DIR, resolve_model_id

log = logging.getLogger("stealth.monitor")  # explicit: under `python -m` __name__ is __main__

REPO_ROOT = Path(__file__).resolve().parents[1]
ALERT_PATH = REPO_ROOT / "handoffs" / "ALERT.md"
LANE_GRACE_UTC = (13, 30)  # (hour, minute): after this, an open round without an accepted entry is an alert
_UPLOAD_ID_KEYS = ("upload_id", "active_upload_id", "id")
_SHA_KEYS = ("sha256", "file_sha256", "artifact_sha256", "pkl_sha256")


# --------------------------------------------------------------------------------------------------
# local
# --------------------------------------------------------------------------------------------------
def load_active(path: Path = ACTIVE_JSON) -> dict | None:
    """``models/active.json`` as a dict, or None when absent."""
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def find_manifest(models_dir: Path = MODELS_DIR, pkl: Path | None = None, active: dict | None = None) -> Path:
    """Manifest to check: from ``--pkl``, else ``active.json``'s pkl, else the single ``*.manifest.json`` in models/."""
    if pkl is not None:
        return manifest_path(Path(pkl))
    if active and active.get("pkl"):
        return manifest_path(Path(models_dir) / active["pkl"])
    found = sorted(Path(models_dir).glob("*.manifest.json"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected exactly one manifest in {models_dir} (found {len(found)}); pass --pkl")
    return found[0]


def check_local(manifest_file: Path, active: dict | None) -> tuple[dict, list[str]]:
    """Local consistency problems (empty list = ok); returns (manifest, problems)."""
    problems: list[str] = []
    manifest_file = Path(manifest_file)
    if not manifest_file.exists():
        return {}, [f"manifest {manifest_file} missing"]
    manifest = json.loads(manifest_file.read_text())
    sha = manifest.get("sha256")
    pkl = manifest_file.with_name(manifest.get("pkl") or f"{manifest_file.name.removesuffix('.manifest.json')}.pkl")
    sha_file = pkl.with_name(pkl.name + ".sha256")
    if sha_file.exists():
        recorded = sha_file.read_text().split()[0] if sha_file.read_text().split() else ""
        if recorded != sha:
            problems.append(f"{sha_file.name} ({recorded[:12]}) != manifest sha256 ({str(sha)[:12]})")
    else:
        problems.append(f"{sha_file.name} missing")
    if pkl.exists():
        actual = sha256_file(pkl)
        if actual != sha:
            problems.append(f"{pkl.name} hashes to {actual[:12]} but manifest says {str(sha)[:12]} (rebuild manifest or pickle)")
    else:
        log.info("%s not present locally (expected on a machine that did not build it)", pkl.name)
    check = manifest.get("pkl_check") or {}
    if not check.get("passed"):
        problems.append("manifest.pkl_check has not passed")
    elif check.get("sha256") != sha:  # the pass must be bound to these bytes (contract review 2)
        problems.append(f"manifest.pkl_check passed for other bytes ({str(check.get('sha256'))[:12]}) than the manifest's "
                        f"sha256 ({str(sha)[:12]}): re-run stealth.pkl_check")
    if active is not None:
        if active.get("sha256") != sha:
            problems.append(f"active.json sha256 ({str(active.get('sha256'))[:12]}) != manifest sha256 ({str(sha)[:12]})")
        if manifest.get("upload_id") and active.get("upload_id") and manifest["upload_id"] != active["upload_id"]:
            problems.append("active.json upload_id != manifest upload_id")
    return manifest, problems


# --------------------------------------------------------------------------------------------------
# online
# --------------------------------------------------------------------------------------------------
def _first(d: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return None


def check_active_upload(client: Any, model_id: str, manifest: dict, active: dict | None) -> list[str]:
    """Problems with the lane's active upload vs our records (empty = ok)."""
    problems: list[str] = []
    status = client.get_upload_status(model_id) or {}
    log.info("upload status: %s", json.dumps({k: v for k, v in status.items() if k != "feature_manifest"}, default=str)[:600])
    if status.get("error_message"):
        problems.append(f"active upload has error_message: {status['error_message']}")
    if int(status.get("sandbox_ready") or 0) != 1:
        problems.append(f"active upload sandbox_ready={status.get('sandbox_ready')} (not 1)")
    if status.get("lane_active") is False:
        problems.append(f"lane_active is false: {status.get('note') or status.get('lane_note')}")
    remote_id = _first(status, _UPLOAD_ID_KEYS)
    remote_sha = _first(status, _SHA_KEYS)
    expected_id = (active or {}).get("upload_id") or manifest.get("upload_id")
    expected_sha = (active or {}).get("sha256") or manifest.get("sha256")
    if remote_sha is not None and expected_sha and str(remote_sha).lower() != str(expected_sha).lower():
        problems.append(f"active upload sha {str(remote_sha)[:12]} != recorded {str(expected_sha)[:12]}")
    elif remote_id is not None and expected_id and str(remote_id) != str(expected_id):
        problems.append(f"active upload_id {remote_id} != recorded {expected_id}")
    elif remote_sha is None and remote_id is None:
        log.warning("upload status exposes neither an upload id nor a sha (unverified field names); identity not compared")
    return problems


def _entries(resp: Any) -> list[dict]:
    if isinstance(resp, list):
        return [e for e in resp if isinstance(e, dict)]
    if isinstance(resp, dict):
        for k in ("submissions", "items", "results", "data"):
            if isinstance(resp.get(k), list):
                return [e for e in resp[k] if isinstance(e, dict)]
        if "accepted" in resp or "submitted" in resp:
            return [resp]
    return []


def _full_coverage(cov: Any) -> bool:
    """``coverage_pct`` None = field not published (assume full); any number < 100 (incl. 0) = NOT full."""
    if cov is None:
        return True
    try:
        return float(cov) >= 100.0
    except (TypeError, ValueError):
        return False


def check_submission(client: Any, model_id: str, *, now: datetime | None = None) -> list[str]:
    """Problems with today's submission (empty = ok or nothing to check yet)."""
    now = now or datetime.now(timezone.utc)
    try:
        rnd = client.get_current_round("futures")
    except Exception as exc:  # noqa: BLE001 - 404 = no active round (weekend / between rounds)
        log.info("no current round (%s); submission check skipped", str(exc)[:200])
        return []
    exped = rnd.get("exped")
    window = rnd.get("submission_window") or rnd.get("status")
    log.info("round %s exped=%s window=%s", rnd.get("round_number"), exped, window)
    if not exped:
        return []
    resp = client.get_submission_status(tournament="futures", round=exped, model_id=model_id)
    entries = _entries(resp)
    accepted = [e for e in entries if e.get("accepted") or e.get("status") == "accepted"]
    full = [e for e in accepted if _full_coverage(e.get("coverage_pct"))]
    if full:
        e = full[0]
        log.info("accepted submission for %s: n=%s coverage=%s at %s", exped, e.get("n_predictions"), e.get("coverage_pct"),
                 e.get("submitted_at"))
        return []
    if accepted:
        return [f"submission for {exped} accepted but coverage {accepted[0].get('coverage_pct')}% < 100"]
    grace = now.replace(hour=LANE_GRACE_UTC[0], minute=LANE_GRACE_UTC[1], second=0, microsecond=0)
    if window == "open" and now < grace:
        log.info("no submission yet for %s; window open and before %02d:%02d UTC lane grace -> not an alert",
                 exped, *LANE_GRACE_UTC)
        return []
    if window not in ("open", "closed", "scoring", "resolving", None):
        return []
    return [f"NO accepted submission for {exped} (window={window}, {now.strftime('%H:%M')} UTC); run `predict --submit` if the window is open"]


def log_scores(client: Any, model_id: str, days: int = 30) -> None:
    """Informational: recent FIT/UNQ/INOV/payout rows for the model."""
    resp = client.get_scores(model_id, days=days)
    rows = _entries(resp) or (resp.get("scores") if isinstance(resp, dict) and isinstance(resp.get("scores"), list) else [])
    if not rows:
        log.info("get_scores: no rows (%s)", json.dumps(resp, default=str)[:300])
        return
    for r in rows[-20:]:
        log.info("score %s: fit=%s unq=%s inov=%s payout=%s partial=%s", r.get("exped") or r.get("round"), r.get("fit"),
                 r.get("unq"), r.get("inov"), r.get("payout"), r.get("is_partial"))


# --------------------------------------------------------------------------------------------------
# alert
# --------------------------------------------------------------------------------------------------
def write_alert(problems: list[str], *, path: Path = ALERT_PATH, context: dict | None = None) -> Path:
    """Write ``handoffs/ALERT.md`` listing the problems (the orchestrator commits/pushes it)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [f"# ALERT — monitor found {len(problems)} problem(s) at {now}", "",
             "**USER MUST REVIEW.** Nothing was submitted or uploaded by the monitor.", ""]
    for p in problems:
        lines.append(f"- {p}")
    if context:
        lines += ["", "Context:", "```json", json.dumps(context, indent=2, default=str), "```"]
    lines += ["", "Suggested actions: if the window is open run `python -m stealth.predict --pkl models/<X>.pkl` (dry run) and,",
              "with a USER MUST APPROVE handoff, `--submit`; if the active upload is wrong, re-run `predict --upload`."]
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entry; see module docstring for checks and exit codes."""
    p = argparse.ArgumentParser(prog="python -m stealth.monitor", description=__doc__.splitlines()[0])
    p.add_argument("--offline", action="store_true", help="local consistency only (no network)")
    p.add_argument("--scores", action="store_true", help="also log recent scores")
    p.add_argument("--model-id", default=None)
    p.add_argument("--pkl", type=Path, default=None, help="pickle whose manifest to check (default: active.json / the only manifest)")
    p.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    p.add_argument("--active", type=Path, default=ACTIVE_JSON)
    p.add_argument("--alert", type=Path, default=ALERT_PATH)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    problems: list[str] = []
    context: dict[str, Any] = {}
    try:
        active = load_active(args.active)
        manifest_file = find_manifest(args.models_dir, args.pkl, active)
        manifest, local = check_local(manifest_file, active)
        problems += local
        context["manifest"] = str(manifest_file)
        context["active"] = active
        if not args.offline:
            model_id = resolve_model_id(args.model_id or (active or {}).get("model_id"))
            client = sdata.make_client()
            problems += check_active_upload(client, model_id, manifest, active)
            problems += check_submission(client, model_id)
            if args.scores:
                log_scores(client, model_id)
    except (FileNotFoundError, RuntimeError) as exc:
        log.error("%s", exc)
        return 2
    except Exception:  # noqa: BLE001 - CLI boundary
        log.exception("monitor failed")
        problems.append("monitor crashed; see log")
    if problems:
        path = write_alert(problems, path=args.alert, context=context)
        for pr in problems:
            log.error("ALERT: %s", pr)
        log.error("wrote %s", path)
        return 1
    log.info("monitor: all checks passed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
