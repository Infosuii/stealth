"""Tests for stealth._runtime: pure functions + make_predict on tiny fake bundles (offline)."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from stealth import _runtime as rt


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def ridge_blob(columns: list[str], coef: np.ndarray, intercept: float = 0.0) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, coef=np.asarray(coef, dtype=np.float64), intercept=np.float64(intercept),
             columns=np.asarray(columns, dtype=str))
    return buf.getvalue()


def lgbm_blob(X: np.ndarray, y: np.ndarray, seed: int = 0) -> bytes:
    import lightgbm as lgb

    ds = lgb.Dataset(X.astype(np.float32), label=y, params={"max_bin": 16})
    booster = lgb.train({"objective": "regression", "verbosity": -1, "num_leaves": 4, "seed": seed,
                         "num_threads": 1, "min_data_in_leaf": 5, "deterministic": True}, ds, num_boost_round=5)
    return booster.model_to_string().encode("utf-8")


def make_frame(n: int = 90, n_feat: int = 12, seed: int = 0, missing_rate: float = 0.1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = [f"feature_{i:03d}" for i in range(n_feat)]
    x = rng.integers(0, 10, size=(n, n_feat)).astype(np.int8)
    x[rng.random((n, n_feat)) < missing_rate] = -1
    idx = pd.Index([f"{i:016x}" for i in rng.integers(0, 2**60, size=n)], name="id")
    df = pd.DataFrame(x, index=idx, columns=cols)
    df.insert(0, "exped", "exped_0001")
    return df


def fake_bundle(df: pd.DataFrame, *, p_b: float = 0.5, p_c: float = 0.25, with_lgbm: bool = True, seed: int = 0) -> dict:
    cols = [c for c in df.columns if c.startswith("feature_")]
    missing_cols = cols[:3]
    rng = np.random.default_rng(seed)
    lin_cols = cols + [f"miss_{c}" for c in missing_cols] + ["n_missing"]
    ridge = ridge_blob(lin_cols, rng.normal(size=len(lin_cols)), 0.1)
    members = [{"name": "ridge_everest", "family": "ridge", "target": "target_everest", "group": "everest",
                "input": "linear", "feature_cols": cols, "blobs": [ridge]}]
    weights = {"ridge_everest": 0.5}
    if with_lgbm:
        X = rt.nan_matrix(rt.raw_matrix(df, cols))
        y = rng.normal(size=len(df))
        members.append({"name": "lgbm_everest", "family": "lgbm", "target": "target_everest", "group": "everest",
                        "input": "nan", "feature_cols": cols[2:], "blobs": [lgbm_blob(X[:, 2:], y, 0), lgbm_blob(X[:, 2:], y, 1)]})
        weights["lgbm_everest"] = 0.5
    return {"version": "test", "git_sha": "deadbeef", "built_at": "now", "run_id": "t",
            "feature_cols": cols, "missing_cols": missing_cols, "members": members, "weights": weights,
            "p_b": p_b, "p_c": p_c, "core_proxy_cols": cols[-4:], "min_bench_coverage": 0.5}


def check_output(out: pd.DataFrame, index: pd.Index) -> None:
    assert type(out) is pd.DataFrame
    assert out.shape == (len(index), 1)
    assert out.index.equals(index)
    assert out.iloc[:, 0].dtype == np.float64
    v = out.iloc[:, 0].to_numpy()
    assert np.isfinite(v).all()
    assert (v > 0).all() and (v < 1).all()


# --------------------------------------------------------------------------------------------------
# pure functions
# --------------------------------------------------------------------------------------------------
def test_rank_avg_matches_pandas():
    rng = np.random.default_rng(1)
    v = rng.integers(0, 5, size=200).astype(float)
    expect = pd.Series(v).rank(method="average").to_numpy()
    np.testing.assert_allclose(rt.rank_avg(v), expect)
    assert rt.rank_avg(np.empty(0)).shape == (0,)


def test_rank01_range_and_nan():
    v = np.array([3.0, np.nan, 1.0, 2.0, 2.0])
    r = rt.rank01(v)
    assert r[1] == 0.5
    ok = np.delete(r, 1)
    assert (ok > 0).all() and (ok < 1).all()
    assert abs(ok.mean() - 0.5) < 1e-12
    assert np.all(rt.rank01(np.array([np.nan, np.nan])) == 0.5)
    assert rt.rank01(np.array([7.0])) == pytest.approx(0.5)


def test_gaussianise_properties():
    rng = np.random.default_rng(2)
    v = rng.normal(size=91)
    z = rt.gaussianise(v)
    assert abs(z.mean()) < 1e-9  # antisymmetric ranks -> exactly centred (no ties)
    assert np.all(np.diff(z[np.argsort(v)]) > 0)  # monotone in v
    assert np.isfinite(z).all()
    v2 = v.copy()
    v2[[3, 10]] = np.nan
    z2 = rt.gaussianise(v2)
    assert z2[3] == 0.0 and z2[10] == 0.0
    assert rt.gaussianise(np.array([1.0])) == pytest.approx(0.0)
    assert np.all(rt.gaussianise(np.array([np.nan, np.nan])) == 0.0)
    # ties share a value
    zt = rt.gaussianise(np.array([1.0, 1.0, 2.0, 3.0]))
    assert zt[0] == zt[1]


def test_neutralise_guards_and_full_projection():
    rng = np.random.default_rng(3)
    z = rt.gaussianise(rng.normal(size=90))
    m = rt.gaussianise(rng.normal(size=90))
    z1 = rt.neutralise(z, m, 1.0)
    assert abs(m @ z1) < 1e-9  # fully orthogonal
    z_half = rt.neutralise(z, m, 0.5)
    assert abs(m @ z_half - 0.5 * (m @ z)) < 1e-9  # halves the projection
    np.testing.assert_array_equal(rt.neutralise(z, m, 0.0), z)
    np.testing.assert_array_equal(rt.neutralise(z, np.zeros(90), 0.5), z)  # constant reference -> no-op
    np.testing.assert_array_equal(rt.neutralise(z, m[:10], 0.5), z)  # shape mismatch -> no-op
    bad = m.copy()
    bad[0] = np.nan
    np.testing.assert_array_equal(rt.neutralise(z, bad, 0.5), z)
    out = rt.neutralise(z, m, 0.5)
    assert out is not z  # copy semantics


def test_coerce_benchmark_cases():
    idx = pd.Index([f"id{i}" for i in range(20)], name="id")
    rng = np.random.default_rng(4)
    s = pd.Series(rng.random(20), index=idx, name="eiq_minera_model")
    assert rt.coerce_benchmark(None, idx) is None
    arr = rt.coerce_benchmark(s, idx)
    assert arr is not None and np.isfinite(arr).all()
    np.testing.assert_allclose(arr, s.to_numpy())
    # 1-col DataFrame, reordered + extra ids + 5 missing ids -> aligned, coverage 0.75
    mis = s.sample(frac=1.0, random_state=1).iloc[:15]
    mis = pd.concat([mis, pd.Series([0.1, 0.2], index=pd.Index(["zz1", "zz2"], name="id"))])
    arr2 = rt.coerce_benchmark(mis.to_frame(), idx)
    assert arr2 is not None and np.isfinite(arr2).sum() == 15
    # coverage below 50% -> None
    assert rt.coerce_benchmark(s.iloc[:8], idx) is None
    # constant -> None
    assert rt.coerce_benchmark(pd.Series(0.3, index=idx), idx) is None
    # garbage -> None, never raises
    assert rt.coerce_benchmark("nonsense", idx) is None
    assert rt.coerce_benchmark(pd.Series(["a", "b"], index=idx[:2]), idx) is None
    # duplicate ids tolerated
    dup = pd.concat([s, s.iloc[:3]])
    assert rt.coerce_benchmark(dup, idx) is not None
    # positional array of the right length is accepted
    assert rt.coerce_benchmark(s.to_numpy(), idx) is not None
    assert rt.coerce_benchmark(s.to_numpy()[:5], idx) is None


def test_core_proxy_values_and_linear_frame():
    df = make_frame(n=30, n_feat=6, missing_rate=0.3)
    cols = [c for c in df.columns if c.startswith("feature_")]
    raw = rt.raw_matrix(df, cols)
    c = rt.core_proxy_values(raw, cols, cols[:3])
    x = raw[:, :3].astype(float)
    x[x == -1] = np.nan
    expect = np.nanmean(x, axis=1)
    np.testing.assert_allclose(c[np.isfinite(expect)], expect[np.isfinite(expect)])
    assert np.isnan(c[~np.isfinite(expect)]).all()
    assert np.isnan(rt.core_proxy_values(raw, cols, ["nope"])).all()
    lin = rt.linear_frame(raw, cols, cols[:2])
    assert list(lin.columns) == cols + [f"miss_{cols[0]}", f"miss_{cols[1]}", "n_missing"]
    assert lin.dtypes.eq(np.float32).all()
    assert float(lin[cols].to_numpy().min()) >= 0.0 and float(lin[cols].to_numpy().max()) <= 1.0
    assert np.all(lin.loc[raw[:, 0] == -1, cols[0]].to_numpy() == np.float32(0.5))
    assert np.all(lin.loc[raw[:, 0] == -1, f"miss_{cols[0]}"].to_numpy() == 1.0)


def test_raw_matrix_missing_column_raises():
    df = make_frame(n=10, n_feat=4)
    with pytest.raises(KeyError, match="feature_003"):
        rt.raw_matrix(df.drop(columns=["feature_003"]), [f"feature_{i:03d}" for i in range(4)])


def test_load_and_predict_member_roundtrip_ridge_lgbm():
    df = make_frame(n=40, n_feat=5)
    cols = [c for c in df.columns if c.startswith("feature_")]
    raw = rt.raw_matrix(df, cols)
    lin = rt.linear_frame(raw, cols, [])
    blob = ridge_blob(list(lin.columns), np.arange(len(lin.columns), dtype=float), 1.5)
    model = rt.load_member("ridge", blob)
    assert model["columns"] == list(lin.columns)
    pred = rt.predict_member("ridge", model, lin.to_numpy(dtype=np.float32))
    expect = lin.to_numpy(dtype=np.float32).astype(np.float64) @ np.arange(len(lin.columns), dtype=float) + 1.5
    np.testing.assert_allclose(pred, expect)
    with pytest.raises(ValueError):
        rt.predict_member("ridge", model, lin.to_numpy(dtype=np.float32)[:, :2])
    X = rt.nan_matrix(raw)
    booster = rt.load_member("lgbm", lgbm_blob(X, np.arange(40, dtype=float)))
    p = rt.predict_member("lgbm", booster, X)
    assert p.shape == (40,) and np.isfinite(p).all()
    assert rt.predict_member("lgbm", booster, X[:0]).shape == (0,)
    with pytest.raises(ValueError):
        rt.load_member("mlp", b"")  # malformed npz -> ValueError (mlp is a supported family now)
    with pytest.raises(ValueError, match="unknown member family"):
        rt.load_member("nope", b"")
    with pytest.raises(ValueError, match="unknown member family"):
        rt.predict_member("nope", None, X)


# --------------------------------------------------------------------------------------------------
# make_predict
# --------------------------------------------------------------------------------------------------
def test_make_predict_contract_fake_bundle():
    df = make_frame(n=90, n_feat=12, seed=5)
    bundle = fake_bundle(df)
    predict = rt.make_predict(bundle)
    out = predict(df)
    check_output(out, df.index)
    assert predict.last_info["bench_used"] is False and predict.last_info["core_used"] is True
    # deterministic
    pd.testing.assert_frame_equal(predict(df), out)
    # extra columns + shuffled column order -> identical
    shuffled = df.sample(frac=1.0, axis=1, random_state=3).copy()
    shuffled["feature_extra"] = 7
    pd.testing.assert_frame_equal(predict(shuffled), out)
    # row order irrelevant for the values (ranks are per row)
    perm = df.sample(frac=1.0, random_state=9)
    out_perm = predict(perm)
    pd.testing.assert_series_equal(out_perm["prediction"].reindex(df.index), out["prediction"])
    # missing feature -> KeyError
    with pytest.raises(KeyError):
        predict(df.drop(columns=[bundle["feature_cols"][0]]))
    # uniform ranks (no ties expected with ridge + lgbm mix)
    v = np.sort(out["prediction"].to_numpy())
    np.testing.assert_allclose(v, (np.arange(90) + 0.5) / 90)


def test_make_predict_benchmark_handling():
    df = make_frame(n=90, n_feat=12, seed=6)
    bundle = fake_bundle(df, p_b=0.75, p_c=0.0)
    predict = rt.make_predict(bundle)
    rng = np.random.default_rng(7)
    bench = pd.Series(rng.random(90), index=df.index, name="eiq_minera_model")
    base = predict(df)
    with_b = predict(df, bench)
    check_output(with_b, df.index)
    assert predict.last_info["bench_used"] is True
    assert not with_b.equals(base)
    # p_b = 1 is capped to 0.75, so correlation with the benchmark must be *reduced* but need not be 0
    rho0 = np.corrcoef(rt.gaussianise(base["prediction"].to_numpy()), rt.gaussianise(bench.to_numpy()))[0, 1]
    rho1 = np.corrcoef(rt.gaussianise(with_b["prediction"].to_numpy()), rt.gaussianise(bench.to_numpy()))[0, 1]
    assert abs(rho1) < abs(rho0) + 1e-9
    # 1-col DataFrame == Series
    pd.testing.assert_frame_equal(predict(df, bench.to_frame()), with_b)
    # misaligned: reordered, 20 ids dropped, 10 foreign ids added -> still a valid output
    mis = bench.sample(frac=1.0, random_state=1).iloc[:70]
    mis = pd.concat([mis, pd.Series(rng.random(10), index=pd.Index([f"x{i}" for i in range(10)], name="id"))])
    out_mis = predict(df, mis)
    check_output(out_mis, df.index)
    assert predict.last_info["bench_used"] is True
    # coverage < 50% -> skipped == no-benchmark result
    pd.testing.assert_frame_equal(predict(df, bench.iloc[:30]), base)
    assert predict.last_info["bench_used"] is False and predict.last_info["bench_reason"] == "skipped"
    # constant -> skipped
    pd.testing.assert_frame_equal(predict(df, pd.Series(0.4, index=df.index)), base)
    # None -> skipped
    pd.testing.assert_frame_equal(predict(df, None), base)
    # garbage never raises
    check_output(predict(df, "garbage"), df.index)
    check_output(predict(df, pd.DataFrame(index=df.index)), df.index)


def test_make_predict_caps_and_weights():
    df = make_frame(n=50, n_feat=8, seed=8)
    bundle = fake_bundle(df, p_b=5.0, p_c=5.0, with_lgbm=False)
    predict = rt.make_predict(bundle)
    assert predict.last_info == {}
    predict(df)
    assert predict.last_info["p_b"] == rt.P_B_CAP and predict.last_info["p_c"] == rt.P_C_CAP
    # zero-weight member is dropped; all-zero -> ValueError
    bad = fake_bundle(df, with_lgbm=False)
    bad["weights"] = {"ridge_everest": 0.0}
    with pytest.raises(ValueError):
        rt.make_predict(bad)
    # unnormalised weights are fine
    b2 = fake_bundle(df)
    b2["weights"] = {"ridge_everest": 3.0, "lgbm_everest": 1.0}
    check_output(rt.make_predict(b2)(df), df.index)


def test_make_predict_refuses_non_finite_numbers():
    """contract review 7: NaN weights used to drop a member silently, inf weights turned every score into NaN (then a
    constant 0.5 that passes the range checks), and a NaN p_b skipped neutralisation without a word."""
    df = make_frame(n=40, n_feat=8, seed=9)
    for upd in ({"weights": {"ridge_everest": float("nan"), "lgbm_everest": 0.5}},
                {"weights": {"ridge_everest": float("inf"), "lgbm_everest": 0.5}},
                {"weights": {"ridge_everest": -0.5, "lgbm_everest": 1.5}},
                {"p_b": float("nan")}, {"p_c": float("inf")}, {"p_b": -0.1}, {"min_bench_coverage": float("nan")}):
        with pytest.raises(ValueError):
            rt.make_predict({**fake_bundle(df), **upd})
    ok = rt.make_predict({**fake_bundle(df), "p_b": None})  # None = no neutralisation, as before
    check_output(ok(df), df.index)
    assert ok.last_info["p_b"] == 0.0


def test_make_predict_edge_sizes():
    df = make_frame(n=90, n_feat=8, seed=10)
    predict = rt.make_predict(fake_bundle(df))
    one = predict(df.iloc[:1])
    check_output(one, df.index[:1])
    assert one.iloc[0, 0] == pytest.approx(0.5)
    empty = predict(df.iloc[:0])
    assert type(empty) is pd.DataFrame and empty.shape == (0, 1)
    # a frame served as float with NaN instead of -1 behaves like -1
    as_float = df.copy()
    feats = [c for c in df.columns if c.startswith("feature_")]
    as_float[feats] = as_float[feats].astype("float32").replace(-1.0, np.nan)
    pd.testing.assert_frame_equal(predict(as_float), predict(df))


def test_make_predict_on_live_frame(live_frame: pd.DataFrame, live_bench: pd.Series):
    """Real live parquet (90 rows x 178 features) with a fake ridge+lgbm bundle over all features."""
    cols = [c for c in live_frame.columns if c.startswith("feature_")]
    assert len(cols) == 178
    bundle = fake_bundle(live_frame.loc[:, ["exped", *cols]], p_b=0.5, p_c=0.25)
    predict = rt.make_predict(bundle)
    out = predict(live_frame, live_bench)
    check_output(out, live_frame.index)
    assert predict.last_info["bench_used"] is True and predict.last_info["core_used"] is True
    pd.testing.assert_frame_equal(predict(live_frame, live_bench), out)


def test_coerce_benchmark_multi_column_frame_prefers_named_column():
    """Contract review item 4: a multi-column frame (e.g. the downloaded ``exped`` + ``v1_sherpa`` file) must not be
    reduced to its first column blindly; the benchmark column is picked by name, else the first numeric one."""
    idx = pd.Index([f"id{i}" for i in range(20)], name="id")
    rng = np.random.default_rng(5)
    s = pd.Series(rng.random(20), index=idx)
    frame = pd.DataFrame({"exped": "exped_8055", "v1_sherpa": s})
    arr = rt.coerce_benchmark(frame, idx)
    assert arr is not None
    np.testing.assert_allclose(arr, s.to_numpy())
    assert rt.select_benchmark_column(frame)[1] == "v1_sherpa"
    frame2 = pd.DataFrame({"zzz_text": "x", "eiq_minera_model": s, "other": s * 2})
    assert rt.select_benchmark_column(frame2)[1] == "eiq_minera_model"
    frame3 = pd.DataFrame({"a_text": "x", "scores": s})
    assert rt.select_benchmark_column(frame3)[1] == "scores"  # first numeric column
    assert rt.select_benchmark_column(pd.DataFrame({"only_text": ["x"] * 20}, index=idx)) == (None, None)
    assert rt.coerce_benchmark(pd.DataFrame(index=idx), idx) is None


# --------------------------------------------------------------------------------------------------
# M5/M6 (INTERFACES section 18): mlp family, per-member subsets, linear members BY NAME
# --------------------------------------------------------------------------------------------------
def mlp_blob(columns: list[str], hidden=(6, 3), seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    dims = [len(columns), *hidden, 1]
    W = [rng.normal(scale=1.0 / np.sqrt(dims[i]), size=(dims[i], dims[i + 1])) for i in range(len(dims) - 1)]
    b = [rng.normal(scale=0.1, size=dims[i + 1]) for i in range(len(dims) - 1)]
    return rt.mlp_to_bytes({"W": W, "b": b, "x_mean": np.full(len(columns), 0.5), "x_std": np.full(len(columns), 0.3),
                            "columns": columns, "act": "gelu_tanh"})


def mixed_bundle(df: pd.DataFrame, *, p_b: float = 0.5, p_c: float = 0.25) -> dict:
    """lgbm on a column BAG (subset, non-contiguous) + ridge (linear, with miss_ columns in shuffled order) + mlp
    (linear base columns, two seeds, columns stored in REVERSED order)."""
    cols = [c for c in df.columns if c.startswith("feature_")]
    missing_cols = cols[:3]
    rng = np.random.default_rng(11)
    bag = cols[1::3] + cols[::4]  # arbitrary order, not a prefix
    X = rt.nan_matrix(rt.raw_matrix(df, cols))
    y = rng.normal(size=len(df))
    pos = [cols.index(c) for c in bag]
    lin_cols = cols + [f"miss_{c}" for c in missing_cols] + ["n_missing"]
    ridge_cols = list(rng.permutation(lin_cols))
    members = [
        {"name": "lgbm_everest_bag0", "family": "lgbm", "target": "target_everest", "group": "everest", "arm": "bags", "input": "nan",
         "feature_cols": bag, "blobs": [lgbm_blob(X[:, pos], y, 0)]},
        {"name": "ridge_everest", "family": "ridge", "target": "target_everest", "group": "everest", "arm": "ridge", "input": "linear",
         "feature_cols": cols, "blobs": [ridge_blob(ridge_cols, rng.normal(size=len(ridge_cols)), 0.2)]},
        {"name": "mlp_everest", "family": "mlp", "target": "target_everest", "group": "everest", "arm": "mlp", "input": "linear",
         "feature_cols": cols, "blobs": [mlp_blob(list(reversed(cols)), seed=1), mlp_blob(list(reversed(cols)), seed=2)]},
    ]
    return {"version": "test", "git_sha": "x", "built_at": "now", "run_id": "t", "feature_cols": cols, "missing_cols": missing_cols,
            "members": members, "weights": {"lgbm_everest_bag0": 0.4, "ridge_everest": 0.3, "mlp_everest": 0.3},
            "p_b": p_b, "p_c": p_c, "core_proxy_cols": cols[-4:], "min_bench_coverage": 0.5, "p_source": "simulation"}


def test_mixed_bundle_bag_ridge_mlp_by_name_and_shuffle_invariant():
    df = make_frame(n=90, n_feat=12, seed=21)
    bundle = mixed_bundle(df)
    predict = rt.make_predict(bundle)
    out = predict(df)
    check_output(out, df.index)
    assert predict.bundle_meta["families"] == {"lgbm_everest_bag0": "lgbm", "ridge_everest": "ridge", "mlp_everest": "mlp"}
    assert predict.bundle_meta["p_source"] == "simulation"
    shuffled = df.sample(frac=1.0, axis=1, random_state=5).copy()
    shuffled["feature_zz_extra"] = 4
    pd.testing.assert_frame_equal(predict(shuffled), out)
    # the members really see their own columns BY NAME: recompute every raw score by hand and blend
    cols = bundle["feature_cols"]
    raw = rt.raw_matrix(df, cols)
    lin = rt.linear_frame(raw, cols, bundle["missing_cols"])
    m_bag, m_ridge, m_mlp = bundle["members"]
    s_bag = rt.predict_member("lgbm", rt.load_member("lgbm", m_bag["blobs"][0]), rt.nan_matrix(raw)[:, [cols.index(c) for c in m_bag["feature_cols"]]])
    ridge = rt.load_member("ridge", m_ridge["blobs"][0])
    s_ridge = lin[ridge["columns"]].to_numpy(dtype=np.float64) @ ridge["coef"] + ridge["intercept"]
    s_mlp = np.mean([rt.mlp_forward(rt.load_member("mlp", b), lin) for b in m_mlp["blobs"]], axis=0)
    Z = np.column_stack([rt.gaussianise(s) for s in (s_bag, s_ridge, s_mlp)])
    z = Z @ np.array([0.4, 0.3, 0.3])
    z = rt.neutralise(z, rt.gaussianise(rt.core_proxy_values(raw, cols, bundle["core_proxy_cols"])), 0.25)
    np.testing.assert_allclose(out["prediction"].to_numpy(), rt.rank01(z), atol=1e-12)


def test_make_predict_subset_violations_raise_valueerror():
    df = make_frame(n=40, n_feat=8, seed=22)
    cols = [c for c in df.columns if c.startswith("feature_")]
    base = mixed_bundle(df)

    def with_member(i: int, **upd) -> dict:
        b = {**base, "members": [dict(m) for m in base["members"]]}
        b["members"][i].update(upd)
        return b

    with pytest.raises(ValueError, match="not in the bundle's linear frame"):  # mlp npz column unknown to the bundle
        rt.make_predict(with_member(2, blobs=[mlp_blob(cols[:-1] + ["feature_zzz"])]))
    with pytest.raises(ValueError, match="not in the bundle's linear frame"):  # ridge miss_ column not in missing_cols
        rt.make_predict({**base, "missing_cols": cols[:2]})
    with pytest.raises(ValueError, match="not in bundle feature_cols"):
        rt.make_predict(with_member(0, feature_cols=["feature_nope", *cols[:2]]))
    with pytest.raises(ValueError, match="needs input 'linear'"):
        rt.make_predict(with_member(2, input="nan"))
    with pytest.raises(ValueError, match="only supported for"):
        rt.make_predict(with_member(0, input="linear"))
    with pytest.raises(ValueError, match="unsupported family"):
        rt.make_predict(with_member(0, family="tabnet"))
    with pytest.raises(ValueError, match="unknown input kind"):
        rt.make_predict(with_member(0, input="raw"))
    with pytest.raises(ValueError, match="npz blob"):  # never unpickled: allow_pickle=False refuses it
        rt.make_predict(with_member(2, blobs=[b"not an npz"]))
    # an mlp-only bundle works (no tree family is ever imported for it)
    only = {**base, "weights": {"mlp_everest": 1.0}}
    check_output(rt.make_predict(only)(df), df.index)


def test_mlp_member_on_live_frame(live_frame: pd.DataFrame, live_bench: pd.Series):
    cols = [c for c in live_frame.columns if c.startswith("feature_")]
    bundle = mixed_bundle(live_frame.loc[:, ["exped", *cols]])
    predict = rt.make_predict(bundle)
    out = predict(live_frame, live_bench)
    check_output(out, live_frame.index)
    assert predict.last_info["bench_used"] is True
    pd.testing.assert_frame_equal(predict(live_frame.sample(frac=1.0, axis=1, random_state=1), live_bench), out)
