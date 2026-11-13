"""Build the cloudpickled ``predict`` closure BY VALUE from ``artifacts/<run_id>``; write sha256 + manifest.

Contract (docs/INTERFACES.md section 9, docs/PLAN.md "Packaging"):

* ``build_bundle(run_dir)`` reads ``blend.json`` (weights, p_b, p_c, core_proxy_cols) and, for every member
  with a positive weight, ``members/<name>/spec.json`` plus its serialised bytes: ``full_s<seed>.bin``
  (full-train refits, averaged) or, when no refit exists, ``fold<k>.bin`` (fold models, averaged).
* ``write_pickle`` registers ``stealth._runtime`` (and ``stealth``) for by-value pickling so the lane
  sandbox, which has no ``stealth`` package, can unpickle the closure; returns the file's sha256.
* ``write_manifest`` writes ``<stem>.manifest.json`` and ``<stem>.pkl.sha256`` next to the pickle. The
  ``.pkl`` itself is never committed; the manifest and sha256 are.

Defaults: ``p_b`` 0.5 (cap 0.75), ``p_c`` 0.25 (cap 0.5) when ``blend.json`` carries none.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from stealth import VERSION
from stealth._runtime import P_B_CAP, P_C_CAP, DEFAULT_MIN_BENCH_COVERAGE

log = logging.getLogger("stealth.bundle")  # explicit: under `python -m` __name__ is __main__

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODELS_DIR = REPO_ROOT / "models"
DEFAULT_P_B = 0.5
DEFAULT_P_C = 0.25
N_FEATURES_EXPECTED = 178
PKL_PYTHON = (3, 11)  # the python_version declared at upload (predict.PKL_PYTHON_VERSION); write_pickle refuses others
MANIFEST_LIBS = ("lightgbm", "xgboost", "catboost", "scikit-learn", "numpy", "pandas", "scipy", "cloudpickle")
_FULL_RE = re.compile(r"^full_s(\d+)\.bin$")
_FOLD_RE = re.compile(r"^fold(\d+)\.bin$")


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def git_sha(repo: Path = REPO_ROOT) -> str:
    """``git rev-parse HEAD`` of ``repo`` (``"unknown"`` if git is unavailable). Never touches remotes."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                             timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", sha) else "unknown"


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file (streamed)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lib_versions(names: tuple[str, ...] = MANIFEST_LIBS) -> dict[str, str | None]:
    """Installed versions of the lane libraries (``None`` when not installed)."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


def _clamp(value: float | None, default: float, cap: float, name: str) -> float:
    v = default if value is None else float(value)
    if v < 0.0:
        raise ValueError(f"{name} must be >= 0 (got {v})")
    if v > cap:
        log.warning("%s=%.3f exceeds the hard cap %.2f; capping", name, v, cap)
        v = cap
    return v


def member_blobs(member_dir: Path) -> tuple[list[bytes], str]:
    """Serialised bytes of a member: ``full_s*.bin`` (sorted by seed) else ``fold*.bin`` (sorted by fold).

    Returns ``(blobs, source)`` with source ``"full"`` or ``"folds"``; raises FileNotFoundError when neither exists.
    """
    full = sorted((p for p in member_dir.iterdir() if _FULL_RE.match(p.name)),
                  key=lambda p: int(_FULL_RE.match(p.name).group(1)))  # type: ignore[union-attr]
    if full:
        return [p.read_bytes() for p in full], "full"
    folds = sorted((p for p in member_dir.iterdir() if _FOLD_RE.match(p.name)),
                   key=lambda p: int(_FOLD_RE.match(p.name).group(1)))  # type: ignore[union-attr]
    if folds:
        return [p.read_bytes() for p in folds], "folds"
    raise FileNotFoundError(f"{member_dir}: no full_s*.bin or fold*.bin")


def _ridge_missing_cols(blob: bytes) -> list[str]:
    """Recover ``missing_cols`` from a ridge npz's stored column names (``miss_<feature>``)."""
    import io

    import numpy as np

    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        cols = [str(c) for c in z["columns"].tolist()]
    return [c[len("miss_"):] for c in cols if c.startswith("miss_")]


def _resolve_feature_cols(run_dir: Path, members: list[dict]) -> tuple[list[str], list[str]]:
    """The run's full feature list (file order) and ``missing_cols``.

    Sources, in order: ``features.json`` (keys ``feature_cols`` / ``missing_cols``), ``spec.json`` fields
    ``all_feature_cols`` / ``missing_cols``, the longest member ``feature_cols`` list (a ``features: all``
    member carries all 178), then the ordered union of member lists. ``missing_cols`` falls back to the
    names recovered from a ridge member's npz columns.
    """
    feature_cols: list[str] = []
    missing_cols: list[str] = []
    fj = run_dir / "features.json"
    if fj.exists():
        d = json.loads(fj.read_text())
        feature_cols = [str(c) for c in d.get("feature_cols", [])]
        missing_cols = [str(c) for c in d.get("missing_cols", [])]
    if not feature_cols:
        for m in members:
            spec = m["spec"]
            if spec.get("all_feature_cols"):
                feature_cols = [str(c) for c in spec["all_feature_cols"]]
                break
    if not feature_cols:
        longest = max(members, key=lambda m: len(m["feature_cols"]))["feature_cols"]
        union = list(dict.fromkeys(c for m in members for c in m["feature_cols"]))
        feature_cols = list(longest) if len(longest) == len(union) else union
        log.warning("feature_cols derived from member lists (%d names); add features.json to the run for an exact list",
                    len(feature_cols))
    if not missing_cols:
        for m in members:
            if m["spec"].get("missing_cols"):
                missing_cols = [str(c) for c in m["spec"]["missing_cols"]]
                break
    if not missing_cols:
        for m in members:
            if m["family"] == "ridge":
                missing_cols = _ridge_missing_cols(m["blobs"][0])
                break
    return feature_cols, missing_cols


# --------------------------------------------------------------------------------------------------
# bundle
# --------------------------------------------------------------------------------------------------
def build_bundle(run_dir: Path, *, p_b: float | None = None, p_c: float | None = None, version: str = VERSION) -> dict:
    """Assemble the bundle dict ``stealth._runtime.make_predict`` consumes from ``artifacts/<run_id>``.

    Reads ``blend.json`` (``weights``, ``p_b``, ``p_c``, ``core_proxy_cols``) and each positively weighted
    member's ``spec.json`` + bytes. ``p_b`` / ``p_c`` arguments override blend.json; both are capped
    (0.75 / 0.5). Weights are renormalised to sum to 1 over the members actually bundled.
    """
    run_dir = Path(run_dir)
    blend_path = run_dir / "blend.json"
    if not blend_path.exists():
        raise FileNotFoundError(f"{blend_path} missing; run `python -m stealth.blend --run-id {run_dir.name}` first")
    blend = json.loads(blend_path.read_text())
    weights_in: dict[str, float] = {str(k): float(v) for k, v in blend.get("weights", {}).items()}
    kept = {k: v for k, v in weights_in.items() if v > 0.0}
    if not kept:
        raise ValueError(f"{blend_path}: no member with positive weight")
    dropped = sorted(set(weights_in) - set(kept))
    if dropped:
        log.info("dropping zero-weight members: %s", dropped)

    members: list[dict] = []
    for name in sorted(kept):
        mdir = run_dir / "members" / name
        spec_path = mdir / "spec.json"
        if not spec_path.exists():
            raise FileNotFoundError(f"{spec_path} missing for weighted member {name!r}")
        spec = json.loads(spec_path.read_text())
        blobs, source = member_blobs(mdir)
        members.append({
            "name": name,
            "family": str(spec["family"]),
            "target": str(spec.get("target", "target_everest")),
            "group": str(spec.get("group", "everest")),
            "input": str(spec.get("input", "linear" if spec["family"] == "ridge" else "nan")),
            "feature_cols": [str(c) for c in spec["feature_cols"]],
            "blobs": blobs,
            "spec": spec,
            "_source": source,
        })
        log.info("member %s: family=%s input=%s %d blobs (%s), %d features", name, spec["family"],
                 members[-1]["input"], len(blobs), source, len(spec["feature_cols"]))

    feature_cols, missing_cols = _resolve_feature_cols(run_dir, members)
    if len(feature_cols) != N_FEATURES_EXPECTED:
        log.warning("bundle has %d feature columns (expected %d)", len(feature_cols), N_FEATURES_EXPECTED)
    total = sum(kept.values())
    weights = {k: v / total for k, v in kept.items()}
    core_cols = [str(c) for c in blend.get("core_proxy_cols", [])]
    if not core_cols:
        log.warning("blend.json has no core_proxy_cols; core-proxy neutralisation will be skipped")
    bundle = {
        "version": version,
        "git_sha": git_sha(),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_dir.name,
        "feature_cols": feature_cols,
        "missing_cols": missing_cols,
        "members": [{k: v for k, v in m.items() if k not in ("spec", "_source")} for m in members],
        "weights": weights,
        "p_b": _clamp(p_b if p_b is not None else blend.get("p_b"), DEFAULT_P_B, P_B_CAP, "p_b"),
        "p_c": _clamp(p_c if p_c is not None else blend.get("p_c"), DEFAULT_P_C, P_C_CAP, "p_c"),
        "core_proxy_cols": core_cols,
        "min_bench_coverage": DEFAULT_MIN_BENCH_COVERAGE,
        "blob_source": {m["name"]: m["_source"] for m in members},
    }
    return bundle


def write_pickle(bundle: dict, out: Path) -> str:
    """cloudpickle ``make_predict(bundle)`` BY VALUE to ``out``; returns the file's sha256 hex digest.

    Registers ``stealth._runtime`` and ``stealth`` with ``cloudpickle.register_pickle_by_value`` so the
    pickle carries the code and never imports ``stealth`` on load. Fails fast if the interpreter is not
    Python 3.11 (the version declared at upload), if the closure cannot be built, or if torch is loaded in
    this interpreter (the lane has no torch).
    """
    import cloudpickle

    import stealth
    import stealth._runtime as runtime

    if sys.version_info[:2] != PKL_PYTHON:
        raise RuntimeError(f"pickle must be built under Python {PKL_PYTHON[0]}.{PKL_PYTHON[1]} (the declared lane version); "
                           f"this interpreter is {platform.python_version()} -- use the train venv")
    if "torch" in sys.modules:
        raise RuntimeError("torch is loaded in this interpreter; build the pickle from the train venv")
    predict = runtime.make_predict(bundle)
    cloudpickle.register_pickle_by_value(runtime)
    cloudpickle.register_pickle_by_value(stealth)
    try:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as fh:
            cloudpickle.dump(predict, fh, protocol=5)
    finally:
        cloudpickle.unregister_pickle_by_value(runtime)
        cloudpickle.unregister_pickle_by_value(stealth)
    sha = sha256_file(out)
    log.info("wrote %s (%.2f MB) sha256=%s", out, out.stat().st_size / 1e6, sha)
    return sha


def manifest_path(out_pkl: Path) -> Path:
    """``models/<stem>.manifest.json`` for ``models/<stem>.pkl``."""
    out_pkl = Path(out_pkl)
    return out_pkl.with_name(f"{out_pkl.stem}.manifest.json")


def write_manifest(out_pkl: Path, bundle: dict, sha256: str) -> Path:
    """Write ``<stem>.manifest.json`` and ``<stem>.pkl.sha256`` next to the pickle; returns the manifest path.

    Existing ``pkl_check`` / ``sandbox_sha256`` / ``upload_id`` values are preserved only when the sha256 is
    unchanged (a new pickle invalidates them).
    """
    out_pkl = Path(out_pkl)
    mpath = manifest_path(out_pkl)
    previous: dict = {}
    if mpath.exists():
        try:
            previous = json.loads(mpath.read_text())
        except json.JSONDecodeError:
            previous = {}
    same = previous.get("sha256") == sha256
    manifest = {
        "pkl": out_pkl.name,
        "sha256": sha256,
        "size_bytes": out_pkl.stat().st_size,
        "version": bundle.get("version", VERSION),
        "git_sha": bundle.get("git_sha", "unknown"),
        "run_id": bundle.get("run_id"),
        "built_at": bundle.get("built_at"),
        "python": platform.python_version(),
        "libs": lib_versions(),
        "members": [{"name": m["name"], "family": m["family"], "target": m["target"], "group": m["group"],
                     "n_blobs": len(m["blobs"])} for m in bundle["members"]],
        "weights": bundle["weights"],
        "p_b": bundle["p_b"],
        "p_c": bundle["p_c"],
        "core_proxy_cols": bundle.get("core_proxy_cols", []),
        "n_features": len(bundle["feature_cols"]),
        "pkl_check": previous.get("pkl_check") if same else None,
        "sandbox_sha256": previous.get("sandbox_sha256") if same else None,
        "upload_id": previous.get("upload_id") if same else None,
    }
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    out_pkl.with_name(out_pkl.name + ".sha256").write_text(f"{sha256}  {out_pkl.name}\n")
    log.info("wrote %s and %s", mpath, out_pkl.name + ".sha256")
    return mpath


def build_and_write(run_dir: Path, out: Path, *, p_b: float | None = None, p_c: float | None = None) -> tuple[dict, str, Path]:
    """build_bundle -> write_pickle -> write_manifest; returns (bundle, sha256, manifest_path)."""
    bundle = build_bundle(run_dir, p_b=p_b, p_c=p_c)
    sha = write_pickle(bundle, out)
    mpath = write_manifest(out, bundle, sha)
    return bundle, sha, mpath


def main(argv: list[str] | None = None) -> int:
    """``python -m stealth.bundle --run-id ID --out models/stealth_v1.pkl [--p-b x] [--p-c y] [--artifacts-dir D]``."""
    parser = argparse.ArgumentParser(prog="python -m stealth.bundle", description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True, help="output .pkl path (e.g. models/stealth_v1.pkl)")
    parser.add_argument("--p-b", type=float, default=None, help="benchmark neutralisation strength (cap 0.75)")
    parser.add_argument("--p-c", type=float, default=None, help="core-proxy neutralisation strength (cap 0.5)")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        bundle, sha, mpath = build_and_write(args.artifacts_dir / args.run_id, args.out, p_b=args.p_b, p_c=args.p_c)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        log.error("bundle failed: %s", exc)
        return 1
    except Exception:  # noqa: BLE001 - CLI boundary
        log.exception("bundle failed")
        return 1
    log.info("bundle ok: run=%s members=%s p_b=%.2f p_c=%.2f sha256=%s manifest=%s", bundle["run_id"],
             [m["name"] for m in bundle["members"]], bundle["p_b"], bundle["p_c"], sha, mpath)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
