"""Build the cloudpickled ``predict`` closure BY VALUE from ``artifacts/<run_id>``; write sha256 + manifest.

Contract (docs/INTERFACES.md section 9, docs/PLAN.md "Packaging"):

* ``build_bundle(run_dir)`` reads ``blend.json`` (weights, p_b, p_c, core_proxy_cols) and, for every member
  with a positive weight, ``members/<name>/spec.json`` plus its serialised bytes: ``full_s<seed>.bin``
  (full-train refits, averaged) or, when no refit exists, ``fold<k>.bin`` (fold models, averaged).
* ``write_pickle`` registers ``stealth._runtime`` (and ``stealth``) for by-value pickling so the lane
  sandbox, which has no ``stealth`` package, can unpickle the closure; returns the file's sha256.
* ``write_manifest`` writes ``<stem>.manifest.json`` and ``<stem>.pkl.sha256`` next to the pickle. The
  ``.pkl`` itself is never committed; the manifest and sha256 are.

Neutralisation constants (docs/INTERFACES.md section 18), resolved in this order and recorded as ``p_source``:
explicit ``p_b``/``p_c`` arguments (``"override"``, WARNING) > ``blend.json["neutralisation"]`` with
``p_source: "simulation"`` (``"simulation"``, the frozen staking-aware choice) > ``blend.json`` ``p_b``/``p_c``
(``"config"``) > ``DEFAULT_P_B``/``DEFAULT_P_C`` (``"default"``, WARNING). A member subset (``--members``) whose
positive-weight set or relative weights differ from the selected blend gets ``"inherited"`` (the constants were
chosen for another prediction) unless explicit overrides are given. ``require_frozen=True`` (CLI
``--require-frozen``) refuses anything but ``"simulation"``. Caps 0.75 / 0.5 always apply; NaN/inf constants and
non-finite or negative weights are refused.

Blobs: a spec that DECLARES its full refits (``full_blobs``) contributes exactly those files (missing or extra
undeclared ``full_s*.bin`` -> error). Legacy specs fall back to every ``full_s*.bin``, else the fold models; the
fold-blob fallback is refused unless every LOADED fold was fitted on ``feature_cols`` (``check_fold_fallback``).
``assert_plain`` guarantees the bundle holds only plain Python values before it is pickled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from stealth import VERSION
from stealth._runtime import DEFAULT_MIN_BENCH_COVERAGE, LINEAR_FAMILIES, P_B_CAP, P_C_CAP, SUPPORTED_FAMILIES

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
P_SOURCES = ("override", "simulation", "config", "default", "inherited")
_WEIGHT_TOL = 1e-9  # relative-weight equality of a member subset with the selected blend
_PLAIN_SCALARS = (str, bytes, bool, int, float, type(None))


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
    if not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number (got {value!r})")
    if v < 0.0:
        raise ValueError(f"{name} must be >= 0 (got {v})")
    if v > cap:
        log.warning("%s=%.3f exceeds the hard cap %.2f; capping", name, v, cap)
        v = cap
    return v


def assert_plain(obj: object, path: str = "bundle") -> None:
    """Raise ``TypeError`` unless ``obj`` is built only from dict (str keys) / list / str / bytes / int / float / bool /
    None -- EXACT types, so numpy scalars/arrays, tuples, tensors or any other object are refused before pickling."""
    t = type(obj)
    if t in _PLAIN_SCALARS:
        return
    if t is dict:
        for k, v in obj.items():  # type: ignore[union-attr]
            if type(k) is not str:
                raise TypeError(f"{path}: dict key {k!r} is {type(k).__name__}, not str")
            assert_plain(v, f"{path}.{k}")
        return
    if t is list:
        for i, v in enumerate(obj):  # type: ignore[arg-type]
            assert_plain(v, f"{path}[{i}]")
        return
    raise TypeError(f"{path}: {t.__module__}.{t.__name__} is not a plain value (dict/list/str/bytes/int/float/bool/None)")


def resolve_p(blend: dict, p_b: float | None = None, p_c: float | None = None) -> tuple[float, float, str, str]:
    """``(p_b, p_c, p_source, reason)`` before capping, by the precedence in the module docstring.

    An override of only one constant keeps the other from the next source down; ``p_source`` is then still
    ``"override"`` (the pair as a whole was not frozen by the simulation).
    """
    neu = blend.get("neutralisation") if isinstance(blend.get("neutralisation"), dict) else {}
    # keyed on p_source, never on the mere presence of `neutralisation` (blend.py always writes it; "config" when
    # blend.neutralisation.select is false). Fallback: the top-level p_source mirror.
    source = neu.get("p_source") or blend.get("p_source")
    frozen = None
    if source == "simulation":
        sb = neu.get("p_b") if neu.get("p_b") is not None else blend.get("p_b")
        sc = neu.get("p_c") if neu.get("p_c") is not None else blend.get("p_c")
        if sb is not None and sc is not None:
            frozen = (float(sb), float(sc))
        top = (blend.get("p_b"), blend.get("p_c"))
        if frozen and top[0] is not None and top[1] is not None and (abs(float(top[0]) - frozen[0]) > 1e-12 or abs(float(top[1]) - frozen[1]) > 1e-12):
            log.warning("blend.json top-level p_b/p_c %s differ from neutralisation %s; using the neutralisation (simulation) values",
                        top, frozen)
    cfg_b = blend.get("p_b") if blend.get("p_b") is not None else neu.get("p_b")
    cfg_c = blend.get("p_c") if blend.get("p_c") is not None else neu.get("p_c")
    if frozen is not None:
        base = (frozen[0], frozen[1], "simulation", "blend.json neutralisation (frozen from the regime simulation)")
    elif cfg_b is not None or cfg_c is not None:
        cb = float(cfg_b) if cfg_b is not None else DEFAULT_P_B
        cc = float(cfg_c) if cfg_c is not None else DEFAULT_P_C
        why = "blend.json p_b/p_c (config constants, not frozen by the simulation)"
        if cfg_b is None or cfg_c is None:
            log.warning("blend.json carries only one of p_b/p_c; the other falls back to its default")
            why += "; one of them defaulted"
        base = (cb, cc, "config", why)
    else:
        log.warning("blend.json has no neutralisation and no p_b/p_c; using defaults p_b=%.2f p_c=%.2f", DEFAULT_P_B, DEFAULT_P_C)
        base = (DEFAULT_P_B, DEFAULT_P_C, "default", "no neutralisation constants in blend.json -> module defaults")
    if p_b is None and p_c is None:
        return base
    rb = float(p_b) if p_b is not None else base[0]
    rc = float(p_c) if p_c is not None else base[1]
    log.warning("neutralisation OVERRIDE p_b=%s p_c=%s (would have been %.3f / %.3f from %s); never ship an override without a handoff "
                "that says why", p_b, p_c, base[0], base[1], base[2])
    return rb, rc, "override", f"explicit override (p_b={p_b}, p_c={p_c}) of {base[2]} values {base[0]:g}/{base[1]:g}"


def member_arm(name: str, spec: dict, blend: dict) -> str:
    """``spec["arm"]`` > ``blend.json["arms"][name]`` > ``"aux"`` for group aux > the member name (INTERFACES section 11)."""
    if spec.get("arm"):
        return str(spec["arm"])
    arms = blend.get("arms") or {}
    if isinstance(arms, dict) and arms.get(name):
        return str(arms[name])
    return "aux" if spec.get("group") == "aux" else str(name)


_PER_FOLD_EXTRA = ("forward",)  # per-fold metadata keys that never correspond to a bundled fold blob


def check_fold_fallback(name: str, spec: dict, folds: list[str] | int) -> None:
    """Refuse averaging ``fold<k>.bin`` under ONE column list unless every LOADED fold blob was fitted on exactly
    ``spec["feature_cols"]`` (``sharpe_subset`` members select columns per fold).

    ``folds`` = the names of the fold blobs actually loaded (``["fold0", "fold2"]``; an int ``n`` means
    ``fold0..fold{n-1}``, the legacy contiguous form). ``spec["feature_cols_per_fold"]`` is ``{fold_name: [cols]}``
    (train.py) or a list indexed by fold number. With per-fold metadata present, every loaded fold needs an entry
    equal to ``feature_cols``, and the metadata may not name a CV fold that was NOT loaded (a sparse fold set, e.g. a
    deleted ``fold1.bin``) nor carry unknown keys -- otherwise the fallback could feed a fold model the wrong columns.
    Without per-fold metadata (legacy specs) there is nothing to compare and the call returns.
    """
    per_fold = spec.get("feature_cols_per_fold")
    if not per_fold:
        return
    loaded = [f"fold{k}" for k in range(folds)] if isinstance(folds, int) else [str(f) for f in folds]
    bad_names = [f for f in loaded if not re.fullmatch(r"fold\d+", f)]
    if bad_names:
        raise ValueError(f"member {name!r}: unexpected fold blob name(s) {bad_names}")
    cols = [str(c) for c in spec["feature_cols"]]
    if isinstance(per_fold, dict):
        meta = {str(k): v for k, v in per_fold.items()}
        unknown = sorted(k for k in meta if not re.fullmatch(r"fold\d+", k) and k not in _PER_FOLD_EXTRA)
    elif isinstance(per_fold, list):
        meta = {f"fold{k}": v for k, v in enumerate(per_fold)}
        unknown = []
    else:
        raise ValueError(f"member {name!r}: feature_cols_per_fold must be a mapping or a list (got {type(per_fold).__name__})")
    if unknown:
        raise ValueError(f"member {name!r}: feature_cols_per_fold has unknown key(s) {unknown}; refusing the fold-blob fallback")
    without = [f for f in loaded if meta.get(f) is None]
    not_loaded = sorted((k for k in meta if re.fullmatch(r"fold\d+", k) and k not in loaded), key=lambda k: int(k[4:]))
    differ = [f for f in loaded if meta.get(f) is not None and [str(c) for c in meta[f]] != cols]
    problems = []
    if without:
        problems.append(f"loaded fold blob(s) {without} have no feature list in feature_cols_per_fold")
    if not_loaded:
        problems.append(f"feature_cols_per_fold names fold(s) {not_loaded} whose blobs are missing (sparse fold set)")
    if differ:
        problems.append(f"the fold feature lists of {differ} differ from feature_cols (sharpe_subset)")
    if problems:
        raise ValueError(f"member {name!r}: no full refit and " + "; ".join(problems) + "; refusing the fold-blob fallback -- "
                         "re-run `stealth.train --resume` to write the full refits")


def declared_full_files(spec: dict | None) -> list[str] | None:
    """File names of the full refits a spec DECLARES (``spec["full_blobs"]``: ``[{"file": "full_s0.bin", ...}]`` from
    train.py / mlp_train.py, or plain names), in declared order; ``None`` when nothing is declared (legacy spec)."""
    if spec is None or "full_blobs" not in spec:
        return None
    declared = spec["full_blobs"]
    if not isinstance(declared, list):
        raise ValueError(f"spec full_blobs must be a list (got {type(declared).__name__})")
    if not declared:
        raise ValueError("spec full_blobs is empty: declared refits are incomplete; refusing legacy fallback")
    names: list[str] = []
    for entry in declared:
        fname = entry.get("file") if isinstance(entry, dict) else entry
        if not isinstance(fname, str) or not _FULL_RE.match(fname):
            raise ValueError(f"spec full_blobs entry {entry!r} is not a full_s<k>.bin file name")
        if fname in names:
            raise ValueError(f"spec full_blobs declares {fname} twice")
        names.append(fname)
    return names


def member_blob_files(member_dir: Path, spec: dict | None = None) -> tuple[list[str], str]:
    """``(file names, source)`` of the blobs a member contributes; source ``"full"`` or ``"folds"``.

    * ``spec["full_blobs"]`` declared -> EXACTLY those files, in declared order: a missing declared file and an extra
      undeclared ``full_s*.bin`` are both errors (an interrupted refit must never change the average silently, and a
      member whose declared refits are gone must never fall back to its fold models unannounced);
    * nothing declared (legacy spec) -> every ``full_s*.bin`` sorted by number, else every ``fold*.bin`` sorted by fold.
    Raises ``FileNotFoundError`` when nothing usable exists.
    """
    member_dir = Path(member_dir)
    present_full = sorted((p.name for p in member_dir.iterdir() if _FULL_RE.match(p.name)),
                          key=lambda n: int(_FULL_RE.match(n).group(1)))  # type: ignore[union-attr]
    declared = declared_full_files(spec)
    if declared is not None:
        missing = [f for f in declared if not (member_dir / f).is_file()]
        if missing:
            raise FileNotFoundError(f"{member_dir}: declared full refit(s) {missing} missing (spec.full_blobs declares {declared}); "
                                    "re-run `stealth.train --resume` (or scripts/mlp_train.py --resume) for this member")
        extra = [f for f in present_full if f not in declared]
        if extra:
            raise ValueError(f"{member_dir}: full refit file(s) {extra} are not declared in spec.full_blobs {declared} "
                             "(stale artifacts of an earlier attempt?); refusing to guess which refits to average")
        return list(declared), "full"
    if present_full:
        return present_full, "full"
    folds = sorted((p.name for p in member_dir.iterdir() if _FOLD_RE.match(p.name)),
                   key=lambda n: int(_FOLD_RE.match(n).group(1)))  # type: ignore[union-attr]
    if folds:
        return folds, "folds"
    raise FileNotFoundError(f"{member_dir}: no full_s*.bin or fold*.bin")


def member_blobs(member_dir: Path, spec: dict | None = None) -> tuple[list[bytes], str]:
    """Serialised bytes of a member per :func:`member_blob_files`; returns ``(blobs, source)``."""
    files, source = member_blob_files(member_dir, spec)
    return [(Path(member_dir) / f).read_bytes() for f in files], source


def _validated_weights(weights: dict, where: str) -> dict[str, float]:
    """``{member: float}`` with every weight finite and >= 0 (NaN would drop a member silently, inf poisons the blend)."""
    out: dict[str, float] = {}
    for k, v in (weights or {}).items():
        try:
            f = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}: weight of {k!r} is not a number ({v!r})") from exc
        if not math.isfinite(f) or f < 0.0:
            raise ValueError(f"{where}: weight of {k!r} must be finite and >= 0 (got {v!r})")
        out[str(k)] = f
    return out


def _same_relative_weights(a: dict[str, float], b: dict[str, float]) -> bool:
    """Same positive-weight member set and the same normalised weights (within ``_WEIGHT_TOL``)."""
    pa = {k: v for k, v in a.items() if v > 0.0}
    pb = {k: v for k, v in b.items() if v > 0.0}
    if set(pa) != set(pb) or not pa:
        return False
    sa, sb = sum(pa.values()), sum(pb.values())
    return all(abs(pa[k] / sa - pb[k] / sb) <= _WEIGHT_TOL for k in pa)


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
def build_bundle(run_dir: Path, *, p_b: float | None = None, p_c: float | None = None, version: str = VERSION,
                 require_frozen: bool = False, members: list[str] | None = None) -> dict:
    """Assemble the bundle dict ``stealth._runtime.make_predict`` consumes from ``artifacts/<run_id>``.

    Reads ``blend.json`` (``weights``, ``rule``, ``neutralisation``/``p_b``/``p_c``, ``core_proxy_cols``, ``arms``,
    ``final.admitted_arms``) and each positively weighted member's ``spec.json`` + bytes. ``p_b`` / ``p_c`` resolve
    via :func:`resolve_p` (``require_frozen`` -> ``ValueError`` unless the source is the simulation); both are capped
    (0.75 / 0.5). Weights are renormalised to sum to 1 over the members actually bundled. The result passes
    :func:`assert_plain` and ``make_predict`` (subset / family checks) before it is returned.

    ``members`` (CLI ``--members a b``) bundles exactly that subset -- e.g. the single-LGBM baseline for a diag
    upload: blend.json weights renormalised over the subset when every requested member has a positive weight, else
    EQUAL weights (WARNING; e.g. a member the ladder did not admit). ``weights_source`` / ``member_subset`` record it.
    """
    from stealth._runtime import make_predict

    run_dir = Path(run_dir)
    blend_path = run_dir / "blend.json"
    if not blend_path.exists():
        raise FileNotFoundError(f"{blend_path} missing; run `python -m stealth.blend --run-id {run_dir.name}` first")
    blend = json.loads(blend_path.read_text())
    rb, rc, p_source, p_reason = resolve_p(blend, p_b, p_c)
    weights_in = _validated_weights(blend.get("weights", {}), f"{blend_path.name} weights")
    subset = None if members is None else list(dict.fromkeys(str(m) for m in members))
    if subset is not None:
        if not subset:
            raise ValueError("--members: empty member list")
        absent = [m for m in subset if not (run_dir / "members" / m / "spec.json").exists()]
        if absent:
            raise ValueError(f"--members: no members/<m>/spec.json for {absent} in {run_dir}")
        if all(weights_in.get(m, 0.0) > 0.0 for m in subset):
            kept, weights_source = {m: weights_in[m] for m in subset}, "blend (member subset, renormalised)"
        else:
            kept, weights_source = {m: 1.0 for m in subset}, "equal (member subset; not every member has a blend weight)"
            log.warning("--members %s: not every member has a positive blend.json weight (%s); using EQUAL weights", subset,
                        {m: weights_in.get(m) for m in subset})
        log.info("member subset %s (blend weights ignored for %s)", subset, sorted(set(weights_in) - set(subset)))
    else:
        kept, weights_source = {k: v for k, v in weights_in.items() if v > 0.0}, "blend"
        if not kept:
            raise ValueError(f"{blend_path}: no member with positive weight")
        dropped = sorted(set(weights_in) - set(kept))
        if dropped:
            log.info("dropping zero-weight members: %s", dropped)
    if subset is not None and p_source != "override" and not _same_relative_weights(kept, weights_in):
        # the constants were chosen (simulated, or configured) for the SELECTED blend's prediction; a different member
        # set or relative weighting is a different prediction, so they are only inherited here, never frozen for it
        inherited_from, p_source = p_source, "inherited"
        p_reason = (f"member subset {subset} (weights {weights_source}) differs from the selected blend "
                    f"{sorted(k for k, v in weights_in.items() if v > 0.0)}: p_b/p_c inherited from its {inherited_from} "
                    f"constants ({p_reason}); not frozen for this prediction -- pass explicit --p-b/--p-c or simulate it")
        log.warning("neutralisation constants INHERITED by a member subset: %s", p_reason)
    if require_frozen and p_source != "simulation":
        raise ValueError(f"--require-frozen: neutralisation constants come from {p_source!r} ({p_reason}), not from the regime "
                         "simulation of this exact blend; run `python -m stealth.blend` with blend.neutralisation.select: true, "
                         "bundle the selected blend and drop any --p-b/--p-c")
    if not math.isfinite(sum(kept.values())) or sum(kept.values()) <= 0.0:
        raise ValueError(f"{blend_path}: member weights {kept} have no positive finite sum")

    members_out: list[dict] = []
    for name in sorted(kept):
        mdir = run_dir / "members" / name
        spec_path = mdir / "spec.json"
        if not spec_path.exists():
            raise FileNotFoundError(f"{spec_path} missing for weighted member {name!r}")
        spec = json.loads(spec_path.read_text())
        family = str(spec["family"])
        if family not in SUPPORTED_FAMILIES:
            raise ValueError(f"member {name!r}: family {family!r} is not supported by the runtime {SUPPORTED_FAMILIES}")
        if spec.get("status") not in (None, "done"):
            raise ValueError(f"member {name!r}: spec status {spec.get('status')!r} is not 'done' but blend.json weights it")
        files, source = member_blob_files(mdir, spec)
        if source == "folds":
            check_fold_fallback(name, spec, [f[: -len(".bin")] for f in files])
            log.warning("member %s: no full refit; bundling the %d fold models %s (averaged)", name, len(files), files)
        blobs = [(mdir / f).read_bytes() for f in files]
        members_out.append({
            "name": name,
            "family": family,
            "target": str(spec.get("target", "target_everest")),
            "group": str(spec.get("group", "everest")),
            "arm": member_arm(name, spec, blend),
            "input": str(spec.get("input", "linear" if family in LINEAR_FAMILIES else "nan")),
            "feature_cols": [str(c) for c in spec["feature_cols"]],
            "blobs": blobs,
            "spec": spec,
            "_source": source,
            "_files": files,
        })
        log.info("member %s: family=%s arm=%s input=%s %d blobs (%s), %d features", name, family, members_out[-1]["arm"],
                 members_out[-1]["input"], len(blobs), source, len(spec["feature_cols"]))

    feature_cols, missing_cols = _resolve_feature_cols(run_dir, members_out)
    if len(feature_cols) != N_FEATURES_EXPECTED:
        log.warning("bundle has %d feature columns (expected %d)", len(feature_cols), N_FEATURES_EXPECTED)
    total = sum(kept.values())
    weights = {k: v / total for k, v in kept.items()}
    core_cols = [str(c) for c in blend.get("core_proxy_cols", [])]
    if not core_cols:
        log.warning("blend.json has no core_proxy_cols; core-proxy neutralisation will be skipped")
    final = blend.get("final") if isinstance(blend.get("final"), dict) else {}
    admitted_arms = [str(a) for a in (final.get("admitted_arms") or [])]
    if subset is not None or not admitted_arms:  # a member subset ships only its own arms
        admitted_arms = list(dict.fromkeys(m["arm"] for m in members_out))
    bundle = {
        "version": version,
        "git_sha": git_sha(),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_dir.name,
        "feature_cols": feature_cols,
        "missing_cols": missing_cols,
        "members": [{k: v for k, v in m.items() if k not in ("spec", "_source", "_files")} for m in members_out],
        "weights": weights,
        "p_b": _clamp(rb, DEFAULT_P_B, P_B_CAP, "p_b"),
        "p_c": _clamp(rc, DEFAULT_P_C, P_C_CAP, "p_c"),
        "p_source": p_source,
        "p_reason": p_reason,
        "blend_rule": str(blend.get("rule", "ew")),
        "admitted_arms": admitted_arms,
        "core_proxy_cols": core_cols,
        "min_bench_coverage": DEFAULT_MIN_BENCH_COVERAGE,
        "blob_source": {m["name"]: m["_source"] for m in members_out},
        "blob_files": {m["name"]: list(m["_files"]) for m in members_out},
        "member_subset": subset,
        "weights_source": weights_source,
    }
    assert_plain(bundle)
    make_predict(bundle)  # fail here (subset / family / input checks), not inside the lane
    log.info("neutralisation: p_b=%.3f p_c=%.3f p_source=%s (%s); rule=%s; arms=%s", bundle["p_b"], bundle["p_c"], p_source,
             p_reason, bundle["blend_rule"], admitted_arms)
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
    assert_plain(bundle)
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
                     "arm": m.get("arm", "aux" if m.get("group") == "aux" else m["name"]), "input": m.get("input", "nan"),
                     "n_features": len(m["feature_cols"]), "n_blobs": len(m["blobs"])} for m in bundle["members"]],
        "weights": bundle["weights"],
        "p_b": bundle["p_b"],
        "p_c": bundle["p_c"],
        "p_source": bundle.get("p_source"),
        "p_reason": bundle.get("p_reason"),
        "blend_rule": bundle.get("blend_rule"),
        "admitted_arms": bundle.get("admitted_arms"),
        "member_subset": bundle.get("member_subset"),
        "weights_source": bundle.get("weights_source"),
        "artifact_source": bundle.get("artifact_source", "selected_blend"),
        "source_run_fingerprint": bundle.get("source_run_fingerprint"),
        "source_universe": bundle.get("source_universe"),
        "blob_files": bundle.get("blob_files"),
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


def build_audit_baseline_bundle(run_dir: Path) -> dict:
    """Export the exact unsearched, pre-audit comparator refits, with neutralisation off. Never trains or selects.

    This is distinct from ``--members lgbm_everest``: that member may use a knob-selected universe and searched
    params. Retaining the pre-registered comparator after a challenger loses does not make its already-seen
    audit scores fresh evidence for future tuning.
    """
    import yaml
    from stealth._runtime import make_predict
    from stealth.knobs import default_universe

    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "baseline" / "spec.json").read_text())
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    fingerprint = json.loads((run_dir / "run_fingerprint.json").read_text())["fingerprint"]
    if not ((cfg.get("cv") or {}).get("audit") or {}).get("enabled"):
        raise ValueError("--audit-baseline requires a run with a reserved audit partition")
    if spec.get("status") != "done" or spec.get("run_fingerprint") != fingerprint:
        raise ValueError("audit baseline is incomplete or belongs to another run fingerprint")
    if spec.get("universe") != default_universe(cfg).key():
        raise ValueError("audit baseline does not use the unsearched default universe")
    if spec.get("family") != "lgbm" or not any(m.get("name") == spec.get("name") and m.get("family") == "lgbm"
                                               for m in cfg.get("members", [])):
        raise ValueError("--audit-baseline requires the registered LightGBM comparator")
    if declared_full_files(spec) is None:
        raise ValueError("audit baseline must declare its complete full-refit blobs")
    files, source = member_blob_files(run_dir / "baseline", spec)
    name, cols = spec["name"], list(spec["feature_cols"])
    result = {"version": VERSION, "git_sha": git_sha(), "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "run_id": run_dir.name, "feature_cols": cols, "missing_cols": [],
              "members": [{"name": name, "family": "lgbm", "target": spec.get("target", "target_everest"),
                           "group": "everest", "arm": "lgbm", "input": "nan", "feature_cols": cols,
                           "blobs": [(run_dir / "baseline" / f).read_bytes() for f in files]}],
              "weights": {name: 1.0}, "p_b": 0.0, "p_c": 0.0, "p_source": "override",
              "p_reason": "exact unneutralised unsearched audit comparator; no post-audit tuning or refit",
              "blend_rule": "audit_baseline", "admitted_arms": ["lgbm"], "core_proxy_cols": [],
              "min_bench_coverage": DEFAULT_MIN_BENCH_COVERAGE, "blob_source": {name: source}, "blob_files": {name: files},
              "member_subset": [name], "weights_source": "unsearched audit baseline",
              "artifact_source": "audit_baseline", "source_run_fingerprint": fingerprint, "source_universe": spec["universe"]}
    assert_plain(result)
    make_predict(result)
    return result


def build_and_write(run_dir: Path, out: Path, *, p_b: float | None = None, p_c: float | None = None,
                    require_frozen: bool = False, members: list[str] | None = None,
                    audit_baseline: bool = False) -> tuple[dict, str, Path]:
    """build_bundle -> write_pickle -> write_manifest; returns (bundle, sha256, manifest_path)."""
    if audit_baseline and (p_b is not None or p_c is not None or require_frozen or members is not None):
        raise ValueError("--audit-baseline exports the exact raw comparator; do not combine it with member or neutralisation overrides")
    bundle = build_audit_baseline_bundle(run_dir) if audit_baseline else build_bundle(
        run_dir, p_b=p_b, p_c=p_c, require_frozen=require_frozen, members=members)
    sha = write_pickle(bundle, out)
    mpath = write_manifest(out, bundle, sha)
    return bundle, sha, mpath


def main(argv: list[str] | None = None) -> int:
    """``python -m stealth.bundle --run-id ID --out models/stealth_v1.pkl [--p-b x] [--p-c y] [--require-frozen]
    [--members m1 m2 ...] [--artifacts-dir D]``. Exit 0 ok, 1 error (including a --require-frozen refusal)."""
    parser = argparse.ArgumentParser(prog="python -m stealth.bundle", description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True, help="output .pkl path (e.g. models/stealth_v1.pkl)")
    parser.add_argument("--p-b", type=float, default=None, help="benchmark neutralisation strength (cap 0.75)")
    parser.add_argument("--p-c", type=float, default=None, help="core-proxy neutralisation strength (cap 0.5)")
    parser.add_argument("--require-frozen", action="store_true",
                        help="refuse unless p_b/p_c were frozen by the regime simulation (blend.json neutralisation)")
    parser.add_argument("--members", nargs="+", default=None,
                        help="bundle exactly these members (e.g. the single-LGBM baseline for a diag upload)")
    parser.add_argument("--audit-baseline", action="store_true",
                        help="export the exact unsearched pre-audit LightGBM comparator, with neutralisation off; no retraining")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        bundle, sha, mpath = build_and_write(args.artifacts_dir / args.run_id, args.out, p_b=args.p_b, p_c=args.p_c,
                                             require_frozen=args.require_frozen, members=args.members, audit_baseline=args.audit_baseline)
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        log.error("bundle failed: %s", exc)
        return 1
    except Exception:  # noqa: BLE001 - CLI boundary
        log.exception("bundle failed")
        return 1
    log.info("bundle ok: run=%s members=%s p_b=%.2f p_c=%.2f p_source=%s rule=%s sha256=%s manifest=%s", bundle["run_id"],
             [m["name"] for m in bundle["members"]], bundle["p_b"], bundle["p_c"], bundle["p_source"], bundle["blend_rule"], sha, mpath)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
