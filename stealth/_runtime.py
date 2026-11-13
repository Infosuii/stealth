"""The ONLY module the pickle may reference (pickled BY VALUE by the bundle module).

Rules (docs/PLAN.md "Packaging", docs/INTERFACES.md section 9):

* module-level imports: stdlib + ``numpy`` + ``pandas`` only. ``scipy``, ``lightgbm``, ``xgboost`` and
  ``catboost`` are imported INSIDE function bodies so the pickle never forces an import at load time
  and never references a library the bundle does not use;
* no imports from the rest of this package (the lane sandbox does not have it);
* no torch, anywhere;
* every function is pure and deterministic.

:func:`make_predict` builds the ``predict(live_features, live_benchmark_models=None)`` closure that the
platform calls once per round (one call == one era / cross-section). Everything the closure needs is
captured in the ``bundle`` dict (feature names, serialised member bytes, blend weights, neutralisation
constants). Members are deserialised lazily on the first call and cached.

Families (docs/INTERFACES.md section 18): ``lgbm``/``xgb``/``cat`` take the NaN tree matrix indexed by the
member's ``feature_cols``; ``ridge``/``mlp`` take the linear frame and select the npz ``columns`` BY NAME.
The MLP forward pass (:func:`mlp_forward`, GELU tanh approximation) is plain numpy and lives here so the
pickle still references this single by-value module. NOTE: function docstrings and string constants of
everything the closure references end up inside the pickle, whose bytes are scanned for the name of the
deep-learning library and for package references -- keep both words out of them.
"""
from __future__ import annotations

import io
import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MISSING = -1.0
DEFAULT_MIN_BENCH_COVERAGE = 0.5
P_B_CAP = 0.75
P_C_CAP = 0.5
SUPPORTED_FAMILIES = ("lgbm", "xgb", "cat", "ridge", "mlp")
LINEAR_FAMILIES = ("ridge", "mlp")   # input: linear, npz blobs with a by-name ``columns`` list
TREE_FAMILIES = ("lgbm", "xgb", "cat")  # input: nan, positions of ``feature_cols`` in the bundle list
MLP_ACT = "gelu_tanh"
_GELU_C = 0.7978845608028654  # sqrt(2 / pi)


# --------------------------------------------------------------------------------------------------
# rank / gaussianise / neutralise (pure numpy)
# --------------------------------------------------------------------------------------------------
def rank_avg(x: np.ndarray) -> np.ndarray:
    """1-based average ranks (ties share the mean rank) of a 1-D float array with NO NaN.

    Equivalent to ``scipy.stats.rankdata(x, method="average")`` / ``pd.Series.rank(method="average")``
    but implemented with numpy only (stable argsort, so it is deterministic).
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    n = v.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(v, kind="stable")
    sv = v[order]
    _, inv, counts = np.unique(sv, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts  # 0-based start of each tie group
    group_rank = starts + (counts - 1) / 2.0 + 1.0  # 1-based average rank per group
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = group_rank[np.asarray(inv).ravel()]
    return ranks


def rank01(x: np.ndarray) -> np.ndarray:
    """``(rank_avg - 0.5) / n`` over the non-NaN entries, strictly inside (0, 1); NaN -> 0.5.

    This is the final mapping of the blended score to the submitted prediction (FIT only sees ranks).
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    out = np.full(v.shape, 0.5, dtype=np.float64)
    ok = np.isfinite(v)
    n = int(ok.sum())
    if n == 0:
        return out
    out[ok] = (rank_avg(v[ok]) - 0.5) / n
    return out


def gaussianise(x: np.ndarray) -> np.ndarray:
    """Per-cross-section gaussian scores ``z = norm.ppf((rank_avg - 0.5) / n)`` over the non-NaN entries.

    NaN (a member that produced no score for that row) -> 0.0, the centre of the distribution, which is
    equivalent to filling with the cross-sectional mean. A single-row cross-section maps to 0.0.
    """
    from scipy.special import ndtri  # scipy imported lazily (module rule)

    v = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(v.shape, dtype=np.float64)
    ok = np.isfinite(v)
    n = int(ok.sum())
    if n == 0:
        return out
    out[ok] = ndtri((rank_avg(v[ok]) - 0.5) / n)
    return out


def neutralise(z: np.ndarray, m: np.ndarray, p: float) -> np.ndarray:
    """Partial orthogonalisation ``z - p * m * (m . z) / (m . m)``.

    ``m`` is the (already gaussianised, hence centred) reference direction. Guards: ``p <= 0``,
    non-finite ``m``, or ``m . m == 0`` (constant reference) -> ``z`` is returned unchanged (a copy).
    With ``p = 1`` the result is exactly orthogonal to ``m`` (the SDK's ``unq20`` residual).
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    m = np.asarray(m, dtype=np.float64).ravel()
    if p <= 0.0 or z.size == 0 or m.shape != z.shape or not np.all(np.isfinite(m)):
        return z.copy()
    mm = float(m @ m)
    if not np.isfinite(mm) or mm <= 0.0:
        return z.copy()
    return z - float(p) * m * (float(m @ z) / mm)


# --------------------------------------------------------------------------------------------------
# feature frames
# --------------------------------------------------------------------------------------------------
def raw_matrix(live_features: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """``feature_cols`` selected BY NAME as a float32 matrix; ``KeyError`` listing the missing names.

    Extra columns are ignored, column order in ``live_features`` is irrelevant. NaN in the served
    frame is treated like -1 (missing) downstream.
    """
    missing = [c for c in feature_cols if c not in live_features.columns]
    if missing:
        raise KeyError(f"live_features is missing {len(missing)} feature column(s): {missing[:10]}"
                       + (" ..." if len(missing) > 10 else ""))
    return live_features.loc[:, list(feature_cols)].to_numpy(dtype=np.float32, copy=True)


def nan_matrix(raw: np.ndarray) -> np.ndarray:
    """Tree input: a copy of ``raw`` with -1 -> NaN (float32)."""
    x = np.array(raw, dtype=np.float32, copy=True)
    x[x == MISSING] = np.nan
    return x


def linear_frame(raw: np.ndarray, feature_cols: list[str], missing_cols: list[str]) -> pd.DataFrame:
    """Linear input, identical to the training-side ``features_linear``: value/9 with missing -> 0.5, one
    ``miss_<feature>`` 0/1 indicator per ``missing_cols`` name and ``n_missing`` = share of missing
    features per row. Columns: ``feature_cols`` + ``miss_*`` + ``n_missing`` (float32)."""
    miss = (raw == MISSING) | ~np.isfinite(raw)
    x = np.where(miss, np.float32(0.5), raw / np.float32(9.0)).astype(np.float32)
    cols = list(feature_cols)
    blocks = [x]
    pos = {c: i for i, c in enumerate(feature_cols)}
    if missing_cols:
        idx = [pos[c] for c in missing_cols if c in pos]
        ind = miss[:, idx].astype(np.float32)
        blocks.append(ind)
        cols += [f"miss_{c}" for c in missing_cols if c in pos]
    n_missing = (miss.sum(axis=1) / np.float32(max(len(feature_cols), 1))).astype(np.float32)
    blocks.append(n_missing[:, None])
    cols.append("n_missing")
    return pd.DataFrame(np.hstack(blocks), columns=cols)


def linear_columns(feature_cols: list[str], missing_cols: list[str]) -> list[str]:
    """Column names :func:`linear_frame` produces for these lists (no data needed)."""
    known = set(feature_cols)
    return list(feature_cols) + [f"miss_{c}" for c in missing_cols if c in known] + ["n_missing"]


def core_proxy_values(raw: np.ndarray, feature_cols: list[str], core_cols: list[str]) -> np.ndarray:
    """Row mean of ``core_cols`` ignoring -1/NaN (the ``scoring.core_feature_average`` rule).

    Rows with every core feature missing -> NaN (the caller's ``gaussianise`` maps them to 0.0).
    Unknown ``core_cols`` names are ignored; no usable column -> all-NaN.
    """
    pos = {c: i for i, c in enumerate(feature_cols)}
    idx = [pos[c] for c in core_cols if c in pos]
    n = raw.shape[0]
    if not idx:
        return np.full(n, np.nan, dtype=np.float64)
    x = raw[:, idx].astype(np.float64)
    x[(x == MISSING) | ~np.isfinite(x)] = np.nan
    with np.errstate(invalid="ignore"):
        cnt = np.sum(np.isfinite(x), axis=1)
        s = np.nansum(x, axis=1)
    out = np.full(n, np.nan, dtype=np.float64)
    ok = cnt > 0
    out[ok] = s[ok] / cnt[ok]
    return out


# --------------------------------------------------------------------------------------------------
# MLP: npz format + numpy forward (section 18)
# --------------------------------------------------------------------------------------------------
def npz_columns(blob: bytes) -> list[str]:
    """The ``columns`` list stored in a ridge/mlp npz blob (cheap: no other array is read)."""
    try:
        with np.load(io.BytesIO(bytes(blob)), allow_pickle=False) as z:
            return [str(c) for c in z["columns"].tolist()]
    except Exception as exc:  # noqa: BLE001 - surface every malformed blob as one error type
        raise ValueError(f"npz blob has no readable columns list: {type(exc).__name__}: {exc}") from exc


def _load_mlp(blob: bytes) -> dict:
    """npz -> ``{"W": [float64 (in x out)], "b": [...], "x_mean", "x_std", "columns", "act", "n_layers"}``, shape-checked."""
    try:
        with np.load(io.BytesIO(bytes(blob)), allow_pickle=False) as z:
            n_layers = int(np.asarray(z["n_layers"]).ravel()[0])
            act = str(np.asarray(z["act"]).ravel()[0])
            W = [np.asarray(z[f"W{i}"], dtype=np.float64) for i in range(n_layers)]
            b = [np.asarray(z[f"b{i}"], dtype=np.float64).ravel() for i in range(n_layers)]
            x_mean = np.asarray(z["x_mean"], dtype=np.float64).ravel()
            x_std = np.asarray(z["x_std"], dtype=np.float64).ravel()
            columns = [str(c) for c in z["columns"].tolist()]
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"mlp: malformed npz blob ({type(exc).__name__}: {exc})") from exc
    if n_layers < 1:
        raise ValueError("mlp: n_layers must be >= 1")
    if act != MLP_ACT:
        raise ValueError(f"mlp: activation {act!r} is not supported (want {MLP_ACT!r})")
    n_in = len(columns)
    if x_mean.size != n_in or x_std.size != n_in or W[0].ndim != 2 or W[0].shape[0] != n_in:
        raise ValueError(f"mlp: input size mismatch (columns={n_in}, x_mean={x_mean.size}, x_std={x_std.size}, "
                         f"W0={W[0].shape})")
    for i in range(n_layers):
        if W[i].ndim != 2 or b[i].shape != (W[i].shape[1],):
            raise ValueError(f"mlp: layer {i}: W{W[i].shape} vs b{b[i].shape}")
        if i + 1 < n_layers and W[i].shape[1] != W[i + 1].shape[0]:
            raise ValueError(f"mlp: layer {i} output {W[i].shape[1]} != layer {i + 1} input {W[i + 1].shape[0]}")
    if W[-1].shape[1] != 1:
        raise ValueError(f"mlp: last layer must have one output (got {W[-1].shape[1]})")
    if not all(np.isfinite(a).all() for a in [*W, *b, x_mean, x_std]):
        raise ValueError("mlp: non-finite weights")
    return {"W": W, "b": b, "x_mean": x_mean, "x_std": x_std, "columns": columns, "act": act, "n_layers": n_layers}


def mlp_to_bytes(model: dict) -> bytes:
    """Serialise an MLP dict (``W``/``b`` lists, in x out) to the npz member format: ``W0..``, ``b0..``, ``x_mean``,
    ``x_std`` float32; ``columns`` str[n_in]; ``act``; ``n_layers``. Round trip: ``load_member("mlp", blob)``."""
    W, b = list(model["W"]), list(model["b"])
    if len(W) != len(b) or not W:
        raise ValueError("mlp: W and b must be non-empty lists of equal length")
    arrays: dict[str, np.ndarray] = {}
    for i, (w_i, b_i) in enumerate(zip(W, b)):
        arrays[f"W{i}"] = np.ascontiguousarray(np.asarray(w_i, dtype=np.float32))
        arrays[f"b{i}"] = np.ascontiguousarray(np.asarray(b_i, dtype=np.float32).ravel())
    arrays["x_mean"] = np.asarray(model["x_mean"], dtype=np.float32).ravel()
    arrays["x_std"] = np.asarray(model["x_std"], dtype=np.float32).ravel()
    arrays["columns"] = np.asarray([str(c) for c in model["columns"]], dtype=str)
    arrays["act"] = np.asarray(str(model.get("act", MLP_ACT)), dtype=str)
    arrays["n_layers"] = np.asarray(len(W), dtype=np.int64)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    blob = buf.getvalue()
    _load_mlp(blob)  # validate the exported shapes once
    return blob


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """GELU, tanh approximation: ``0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`` (float64)."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + 0.044715 * (x * x * x))))


def mlp_forward(model: dict, X: Any) -> np.ndarray:
    """MLP inference in numpy: ``h = (X - x_mean) / x_std``; hidden layers ``h = gelu_tanh(h @ W + b)``; last layer
    linear (dropout is off at inference; only the main head is exported). ``X``: a DataFrame (``model["columns"]``
    selected BY NAME, ``KeyError`` when absent) or a 2-D matrix already in that column order. Returns float64[n]."""
    cols = model["columns"]
    if isinstance(X, pd.DataFrame):
        absent = [c for c in cols if c not in X.columns]
        if absent:
            raise KeyError(f"mlp: input frame is missing {len(absent)} column(s): {absent[:5]}")
        x = X.loc[:, list(cols)].to_numpy(dtype=np.float64)
    else:
        x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(cols):
        raise ValueError(f"mlp: expected a 2-D input with {len(cols)} columns, got shape {x.shape}")
    if x.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    std = np.asarray(model["x_std"], dtype=np.float64)
    std = np.where(np.isfinite(std) & (std > 0.0), std, 1.0)
    h = (x - np.asarray(model["x_mean"], dtype=np.float64)) / std
    W, b = model["W"], model["b"]
    last = len(W) - 1
    for i in range(len(W)):
        h = h @ W[i] + b[i]
        if i < last:
            h = gelu_tanh(h)
    return np.ascontiguousarray(h[:, 0], dtype=np.float64)


# --------------------------------------------------------------------------------------------------
# members
# --------------------------------------------------------------------------------------------------
def load_member(family: str, blob: bytes) -> Any:
    """Deserialise one member from its bytes (the family module's ``to_bytes`` format).

    lgbm: ``Booster(model_str=...)``; xgb: ``Booster().load_model(bytearray)``; cat:
    ``CatBoostRegressor().load_model(blob=...)``; ridge: dict ``{coef, intercept, columns}`` from npz;
    mlp: dict ``{W, b, x_mean, x_std, columns, act, n_layers}`` from npz (``allow_pickle=False``).
    """
    if family == "lgbm":
        import lightgbm as lgb

        return lgb.Booster(model_str=bytes(blob).decode("utf-8"))
    if family == "xgb":
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(bytearray(blob))
        return booster
    if family == "cat":
        from catboost import CatBoostRegressor

        model = CatBoostRegressor()
        model.load_model(blob=bytes(blob))
        return model
    if family == "ridge":
        with np.load(io.BytesIO(bytes(blob)), allow_pickle=False) as z:
            return {
                "coef": np.asarray(z["coef"], dtype=np.float64).ravel(),
                "intercept": float(np.asarray(z["intercept"]).ravel()[0]),
                "columns": [str(c) for c in z["columns"].tolist()],
            }
    if family == "mlp":
        return _load_mlp(blob)
    raise ValueError(f"unknown member family {family!r}")


def predict_member(family: str, model: Any, X: np.ndarray) -> np.ndarray:
    """Raw float64 scores for a float32 2-D matrix ``X`` whose columns are in the member's order."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("predict_member expects a 2-D matrix")
    if X.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    if family == "lgbm":
        return np.asarray(model.predict(X), dtype=np.float64).ravel()
    if family == "xgb":
        import xgboost as xgb

        return np.asarray(model.predict(xgb.DMatrix(X, missing=np.nan)), dtype=np.float64).ravel()
    if family == "cat":
        return np.asarray(model.predict(X), dtype=np.float64).ravel()
    if family == "ridge":
        coef = np.asarray(model["coef"], dtype=np.float64)
        if X.shape[1] != coef.size:
            raise ValueError(f"ridge: {X.shape[1]} input columns vs {coef.size} coefficients")
        return X.astype(np.float64) @ coef + float(model["intercept"])
    if family == "mlp":
        return mlp_forward(model, X)
    raise ValueError(f"unknown member family {family!r}")


# --------------------------------------------------------------------------------------------------
# benchmark coercion
# --------------------------------------------------------------------------------------------------
BENCH_PREFERRED_COLUMNS = ("eiq_minera_model", "v1_sherpa", "prediction", "benchmark")


def select_benchmark_column(frame: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    """Pick the benchmark Series out of a DataFrame: a preferred name (``BENCH_PREFERRED_COLUMNS``) if present,
    else the first column with non-zero finite numeric coverage. Returns ``(series, column_name)`` or ``(None, None)``
    (an empty frame or one with no numeric column). Never raises."""
    try:
        if frame.shape[1] == 0:
            return None, None
        cols = [str(c) for c in frame.columns]
        for name in BENCH_PREFERRED_COLUMNS:
            if name in cols:
                return frame.iloc[:, cols.index(name)], name
        for i, name in enumerate(cols):
            num = pd.to_numeric(frame.iloc[:, i], errors="coerce")
            if int(np.isfinite(num.to_numpy(dtype=np.float64)).sum()) > 0:
                return frame.iloc[:, i], name
    except Exception:  # noqa: BLE001
        return None, None
    return None, None


def coerce_benchmark(bench: Any, index: pd.Index, min_coverage: float = DEFAULT_MIN_BENCH_COVERAGE) -> np.ndarray | None:
    """``live_benchmark_models`` (Series | DataFrame | None | same-length array) -> float64 array
    aligned to ``index`` (NaN where absent), or ``None`` when it must be ignored. A DataFrame goes through
    :func:`select_benchmark_column` (preferred name, else first numeric column) -- never blindly ``iloc[:, 0]``.

    ``None`` when: input is None/unusable, finite coverage < ``min_coverage`` of ``index``, or the covered
    values have zero variance. Never raises.
    """
    if bench is None or len(index) == 0:
        return None
    try:
        if isinstance(bench, pd.DataFrame):
            bench, _ = select_benchmark_column(bench)
            if bench is None:
                return None
        if isinstance(bench, pd.Series):
            s = bench[~bench.index.duplicated(keep="first")]
            arr = pd.to_numeric(s, errors="coerce").reindex(index).to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(bench, dtype=np.float64).ravel()
            if arr.shape[0] != len(index):
                return None
    except Exception:  # noqa: BLE001 - "never raises on a weird benchmark"
        return None
    ok = np.isfinite(arr)
    n_ok = int(ok.sum())
    if n_ok < 2 or n_ok / len(index) < min_coverage:
        return None
    vals = arr[ok]
    if float(np.ptp(vals)) <= 1e-12 * max(1.0, float(np.abs(vals).max())):  # constant (zero variance)
        return None
    return arr


# --------------------------------------------------------------------------------------------------
# the closure
# --------------------------------------------------------------------------------------------------
def _weighted_blend(Z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Row-wise weighted mean of ``Z`` (n x k) dropping NaN entries per row; all-NaN row -> 0.0."""
    ok = np.isfinite(Z)
    Zf = np.where(ok, Z, 0.0)
    num = Zf @ w
    den = ok.astype(np.float64) @ w
    out = np.zeros(Z.shape[0], dtype=np.float64)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


def make_predict(bundle: dict) -> Callable[..., pd.DataFrame]:
    """Build ``predict(live_features, live_benchmark_models=None) -> pd.DataFrame`` from a bundle dict.

    Bundle keys (docs/INTERFACES.md section 9): ``feature_cols`` (all 178 names, file order),
    ``missing_cols``, ``members`` (each: name, family, target, group, input 'nan'|'linear', feature_cols,
    blobs [bytes, ...] averaged), ``weights`` {member: w}, ``p_b``, ``p_c``, ``core_proxy_cols``,
    ``min_bench_coverage``. Steps per call (one era): select columns by name (KeyError if missing) ->
    per-member raw score -> gaussianise -> weighted blend (NaN member dropped per row) -> optional
    benchmark neutralisation (skipped on coverage < min or zero variance) -> core-proxy neutralisation
    -> ``rank01``. Returns an exact ``pd.DataFrame`` with one float64 column ``prediction`` on
    ``live_features.index``, all values strictly in (0, 1), no NaN. The last call's diagnostics are
    exposed as ``predict.last_info`` (dict).

    Member inputs: ``input: nan`` members (lgbm/xgb/cat) index the NaN matrix by the positions of their
    ``feature_cols``; ``input: linear`` members (ridge/mlp) select each blob's npz ``columns`` BY NAME from the
    linear frame. A member whose columns are not all in the bundle lists, an unsupported family or a
    family/input mismatch raises ``ValueError`` here, never at predict time.
    """
    feature_cols: list[str] = [str(c) for c in bundle["feature_cols"]]
    missing_cols: list[str] = [str(c) for c in bundle.get("missing_cols", [])]
    members: list[dict] = list(bundle["members"])
    weights_in: dict[str, float] = dict(bundle["weights"])
    p_b = float(min(max(bundle.get("p_b", 0.0) or 0.0, 0.0), P_B_CAP))
    p_c = float(min(max(bundle.get("p_c", 0.0) or 0.0, 0.0), P_C_CAP))
    core_cols: list[str] = [str(c) for c in bundle.get("core_proxy_cols", [])]
    min_cov = float(bundle.get("min_bench_coverage", DEFAULT_MIN_BENCH_COVERAGE))

    active = [m for m in members if float(weights_in.get(m["name"], 0.0)) > 0.0 and m.get("blobs")]
    if not active:
        raise ValueError("bundle has no member with positive weight and at least one blob")
    w = np.asarray([float(weights_in[m["name"]]) for m in active], dtype=np.float64)
    w = w / w.sum()
    col_pos = {c: i for i, c in enumerate(feature_cols)}
    lin_cols = linear_columns(feature_cols, missing_cols)
    lin_pos = {c: i for i, c in enumerate(lin_cols)}
    # per member: positions into the nan matrix (trees) or, per blob, into the linear frame (ridge / mlp, BY NAME)
    member_idx: list[np.ndarray | None] = []
    blob_idx: list[list[np.ndarray] | None] = []
    for m in active:
        fam, inp = m["family"], m.get("input", "nan")
        if fam not in SUPPORTED_FAMILIES:
            raise ValueError(f"member {m['name']}: unsupported family {fam!r} (supported: {SUPPORTED_FAMILIES})")
        unknown = [c for c in m["feature_cols"] if c not in col_pos]
        if unknown:
            raise ValueError(f"member {m['name']}: feature(s) not in bundle feature_cols: {unknown[:5]}")
        if inp == "linear":
            if fam not in LINEAR_FAMILIES:
                raise ValueError(f"member {m['name']}: input 'linear' is only supported for {LINEAR_FAMILIES}")
            per_blob = []
            for j, blob in enumerate(m["blobs"]):
                cols_j = npz_columns(blob)
                bad = [c for c in cols_j if c not in lin_pos]
                if bad:
                    raise ValueError(f"member {m['name']} blob {j}: column(s) not in the bundle's linear frame: {bad[:5]}")
                per_blob.append(np.asarray([lin_pos[c] for c in cols_j], dtype=np.int64))
            member_idx.append(None)
            blob_idx.append(per_blob)
        elif inp == "nan":
            if fam in LINEAR_FAMILIES:
                raise ValueError(f"member {m['name']}: family {fam!r} needs input 'linear'")
            member_idx.append(np.asarray([col_pos[c] for c in m["feature_cols"]], dtype=np.int64))
            blob_idx.append(None)
        else:
            raise ValueError(f"member {m['name']}: unknown input kind {inp!r}")

    cache: dict[tuple[str, int], Any] = {}

    def _models(i: int) -> list[Any]:
        m = active[i]
        out = []
        for j, blob in enumerate(m["blobs"]):
            key = (m["name"], j)
            if key not in cache:
                cache[key] = load_member(m["family"], blob)
            out.append(cache[key])
        return out

    def _member_score(i: int, nan_mat: np.ndarray, lin: np.ndarray | None) -> np.ndarray:
        m = active[i]
        models = _models(i)
        per_blob = blob_idx[i]
        if per_blob is not None:
            if lin is None:
                raise RuntimeError("linear frame missing")
            scores = [predict_member(m["family"], model, lin[:, per_blob[j]]) for j, model in enumerate(models)]
        else:
            X = nan_mat[:, member_idx[i]]
            scores = [predict_member(m["family"], model, X) for model in models]
        return np.mean(np.vstack(scores), axis=0)

    needs_linear = any(b is not None for b in blob_idx)

    def predict(live_features: pd.DataFrame, live_benchmark_models: Any = None) -> pd.DataFrame:
        index = live_features.index
        info: dict[str, Any] = {"n_rows": int(len(index)), "bench_used": False, "bench_reason": "none",
                                "core_used": False, "p_b": p_b, "p_c": p_c}
        raw = raw_matrix(live_features, feature_cols)
        n = raw.shape[0]
        if n == 0:
            predict.last_info = info  # type: ignore[attr-defined]
            return pd.DataFrame({"prediction": np.empty(0, dtype=np.float64)}, index=index)
        nan_mat = nan_matrix(raw)
        lin = linear_frame(raw, feature_cols, missing_cols).to_numpy(dtype=np.float32) if needs_linear else None
        Z = np.empty((n, len(active)), dtype=np.float64)
        for i in range(len(active)):
            s = _member_score(i, nan_mat, lin)
            Z[:, i] = gaussianise(s) if np.isfinite(s).any() else np.nan
            if not np.isfinite(s).all():
                Z[~np.isfinite(s), i] = np.nan
        z = _weighted_blend(Z, w)
        if p_b > 0.0:
            if isinstance(live_benchmark_models, pd.DataFrame):
                _, info["bench_column"] = select_benchmark_column(live_benchmark_models)
            b = coerce_benchmark(live_benchmark_models, index, min_cov)
            if b is None:
                info["bench_reason"] = "skipped" if live_benchmark_models is not None else "none"
            else:
                cov = np.isfinite(b)
                z_cov = neutralise(z[cov], gaussianise(b[cov]), p_b)
                z = z.copy()
                z[cov] = z_cov
                info["bench_used"] = True
                info["bench_reason"] = f"coverage={cov.mean():.3f}"
        if p_c > 0.0 and core_cols:
            c = core_proxy_values(raw, feature_cols, core_cols)
            if np.isfinite(c).sum() >= 2:
                z = neutralise(z, gaussianise(c), p_c)
                info["core_used"] = True
        pred = rank01(z)
        predict.last_info = info  # type: ignore[attr-defined]
        return pd.DataFrame({"prediction": pred}, index=index)

    predict.last_info = {}  # type: ignore[attr-defined]
    predict.bundle_meta = {  # type: ignore[attr-defined]
        "version": bundle.get("version"), "run_id": bundle.get("run_id"), "git_sha": bundle.get("git_sha"),
        "members": [m["name"] for m in active], "families": {m["name"]: m["family"] for m in active},
        "p_b": p_b, "p_c": p_c, "p_source": bundle.get("p_source"),
    }
    return predict
