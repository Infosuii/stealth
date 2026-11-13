"""Tests for stealth.metrics: SDK parity per era, zero rule, summaries, gates, regime simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from everestapi import scoring

from stealth import metrics as M
from stealth.data import feature_columns

TARGET = "target_everest"


@pytest.fixture(scope="module")
def noisy_pred(train_slice: pd.DataFrame) -> pd.Series:
    """A weak signal: 0.3*feature + noise (deterministic)."""
    rng = np.random.default_rng(0)
    f = train_slice["feature_Agouti"].astype(float).replace(-1.0, 4.5)
    return pd.Series(rng.standard_normal(len(train_slice)) + 0.3 * f, index=train_slice.index, name="pred")


def _random_eras(train_slice: pd.DataFrame, n: int = 8, seed: int = 11) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(e) for e in rng.choice(train_slice["era"].unique(), size=n, replace=False)]


# ------------------------------------------------------------------------------------------- SDK parity
def test_fit20_parity_vs_sdk_random_eras(train_slice, noisy_pred) -> None:
    fit = M.per_era_fit(noisy_pred, train_slice[TARGET], train_slice["era"])
    assert fit.name == "fit" and fit.index.name == "era" and fit.index.dtype == np.int32
    assert fit.index.is_monotonic_increasing and len(fit) == 300
    for e in _random_eras(train_slice):
        m = (train_slice["era"] == e).to_numpy()
        expect = scoring.fit20(noisy_pred[m], train_slice.loc[m, TARGET])
        assert fit.loc[e] == pytest.approx(expect, abs=1e-12)
        # documented local fallback must match the SDK bit-for-bit (modulo float rounding)
        local = M._fit20_local(noisy_pred[m].to_numpy(), train_slice.loc[m, TARGET].to_numpy())
        assert local == pytest.approx(expect, abs=1e-12)
    assert 0.005 < fit.mean() < 0.03  # a 0.3*feature signal is weak but positive


def test_fit20_degenerate_inputs() -> None:
    idx = pd.Index([f"r{i}" for i in range(10)])
    const = pd.Series(np.ones(10), index=idx)
    t = pd.Series(np.linspace(0, 1, 10), index=idx)
    assert M.fit20(const, t) == 0.0 and M._fit20_local(const.to_numpy(), t.to_numpy()) == 0.0
    assert M.fit20(t.iloc[:2], t.iloc[:2]) == 0.0


def test_unq20_parity_and_zero_rule(train_slice, noisy_pred, bench_slice) -> None:
    unq = M.per_era_unq(noisy_pred, bench_slice, train_slice[TARGET], train_slice["era"])
    assert unq.name == "unq" and len(unq) == 300
    bench_fit = M.per_era_bench_fit(bench_slice, train_slice[TARGET], train_slice["era"])
    for e in _random_eras(train_slice):
        m = (train_slice["era"] == e).to_numpy()
        expect = scoring.unq20(noisy_pred[m], bench_slice[m], train_slice.loc[m, TARGET])
        assert unq.loc[e] == pytest.approx(expect, abs=1e-12)
        local = M._unq20_local(noisy_pred[m].to_numpy(), bench_slice[m].to_numpy(), train_slice.loc[m, TARGET].to_numpy())
        assert local == pytest.approx(expect, abs=1e-12)
    # zero rule (inside the SDK, not re-applied by us): eras where the benchmark lost have UNQ == 0 exactly
    neg = bench_fit.index[bench_fit < 0]
    assert len(neg) > 0, "fixture should contain at least one benchmark-negative era"
    assert (unq.loc[neg] == 0.0).all()
    # ... and in benchmark-positive eras UNQ is generally non-zero
    pos = bench_fit.index[bench_fit > 0]
    assert (unq.loc[pos] != 0.0).mean() > 0.95


def test_zero_rule_synthetic() -> None:
    """Flip the benchmark's sign -> its own covariance turns negative -> UNQ must be exactly 0."""
    rng = np.random.default_rng(5)
    idx = pd.Index([f"r{i}" for i in range(90)])
    t = pd.Series(rng.choice([0, .25, .5, .75, 1], size=90, p=[.05, .2, .5, .2, .05]), index=idx)
    pred = pd.Series(t.to_numpy() + rng.standard_normal(90), index=idx)
    good_bench = pd.Series(t.to_numpy() + 0.5 * rng.standard_normal(90), index=idx)
    assert M.unq20(pred, good_bench, t) != 0.0
    assert M.unq20(pred, -good_bench, t) == 0.0
    assert M._unq20_local(pred.to_numpy(), -good_bench.to_numpy(), t.to_numpy()) == 0.0
    # vectorised kernel agrees on both regimes
    z = M.rank_gauss_rows(pred.to_numpy()[None, :])
    tc = (t - t.mean()).to_numpy()[None, :]
    mg = M.rank_gauss_rows(good_bench.to_numpy()[None, :])
    assert M.unq_rows(z, mg, tc)[0] == pytest.approx(M.unq20(pred, good_bench, t), abs=1e-12)
    assert M.unq_rows(z, -mg, tc)[0] == 0.0


def test_inov_parity_vs_sdk(train_slice, noisy_pred) -> None:
    core_cols = feature_columns(train_slice)[:10]
    core = train_slice[core_cols]
    inv = M.per_era_inov(noisy_pred, core, train_slice[TARGET], train_slice["era"])
    assert inv.name == "inov" and len(inv) == 300
    for e in _random_eras(train_slice, n=5):
        m = (train_slice["era"] == e).to_numpy()
        expect = scoring.inov(noisy_pred[m], core[m], train_slice.loc[m, TARGET])
        assert inv.loc[e] == pytest.approx(expect, abs=1e-12)


def test_per_era_spearman_and_corr(train_slice, noisy_pred) -> None:
    from scipy.stats import spearmanr

    sp = M.per_era_spearman(noisy_pred, train_slice[TARGET], train_slice["era"])
    assert sp.name == "spearman" and len(sp) == 300
    e = _random_eras(train_slice, n=1)[0]
    m = (train_slice["era"] == e).to_numpy()
    assert sp.loc[e] == pytest.approx(spearmanr(noisy_pred[m], train_slice.loc[m, TARGET]).statistic, abs=1e-12)
    # FIT ~ 0.226 x Spearman on this data (platform_contract.md)
    fit = M.per_era_fit(noisy_pred, train_slice[TARGET], train_slice["era"])
    ratio = fit.mean() / sp.mean()
    assert 0.18 < ratio < 0.27
    c = M.per_era_corr(noisy_pred, noisy_pred, train_slice["era"])
    assert c.name == "corr" and np.allclose(c.to_numpy(), 1.0)


def test_alignment_reindexes_shuffled_inputs(train_slice, noisy_pred) -> None:
    shuffled = noisy_pred.sample(frac=1.0, random_state=0)
    a = M.per_era_fit(noisy_pred, train_slice[TARGET], train_slice["era"])
    b = M.per_era_fit(shuffled, train_slice[TARGET], train_slice["era"])
    pd.testing.assert_series_equal(a, b)
    with pytest.raises(ValueError):
        M.per_era_fit(pd.Series([1.0, 2.0], index=["x", "y"]), train_slice[TARGET], train_slice["era"])


# --------------------------------------------------------------------------------------- summaries/gates
def test_summarise_keys_and_values() -> None:
    rng = np.random.default_rng(6)
    s = pd.Series(0.02 + 0.04 * rng.standard_normal(400), index=pd.Index(np.arange(400), name="era"))
    d = M.summarise(s, n_boot=500)
    assert set(d) == {"mean", "std", "sharpe", "pct_pos", "frac_roll20_neg", "ci90", "se_block", "n_eras"}
    # the reported SE is the block-bootstrap SE, consistent with ci90 (~ +/- 1.645 se), never the naive std/sqrt(n)
    assert d["se_block"] == pytest.approx((d["ci90"][1] - d["ci90"][0]) / (2 * 1.645), rel=0.25)
    assert d["n_eras"] == 400 and d["mean"] == pytest.approx(s.mean())
    assert d["std"] == pytest.approx(s.std(ddof=1)) and d["sharpe"] == pytest.approx(d["mean"] / d["std"])
    assert d["pct_pos"] == pytest.approx((s > 0).mean())
    roll = s.rolling(20).mean().dropna()
    assert d["frac_roll20_neg"] == pytest.approx((roll < 0).mean())
    assert d["ci90"][0] < d["mean"] < d["ci90"][1]
    empty = M.summarise(pd.Series([], dtype=float))
    assert empty["n_eras"] == 0 and np.isnan(empty["mean"])


def test_leak_check_and_suspect_flag() -> None:
    ok = pd.Series(np.full(100, 0.12))
    M.leak_check(ok)  # no raise
    with pytest.raises(RuntimeError, match="LEAK"):
        M.leak_check(pd.Series(np.full(100, 0.25)))
    with pytest.raises(RuntimeError):
        M.leak_check(ok, threshold=0.10)
    assert M.suspect_gain(0.031, 0.026) is True
    assert M.suspect_gain(0.029, 0.026) is False
    assert M.suspect_gain(0.030, 0.026, margin=0.005) is False


# ------------------------------------------------------------------------------------ vectorised kernels
def test_rank_gauss_rows_matches_sdk_transform() -> None:
    rng = np.random.default_rng(7)
    a = rng.standard_normal((3, 12))
    a[0, 10:] = np.nan
    a[1, 3] = a[1, 4]  # a tie
    z = M.rank_gauss_rows(a)
    for i in range(3):
        row = a[i][~np.isnan(a[i])]
        expect = scoring._rank_and_gaussianize(pd.Series(row))
        np.testing.assert_allclose(z[i][~np.isnan(a[i])], expect, atol=1e-12)
    assert np.isnan(z[0, 10:]).all()


def test_fit_rows_parity_with_sdk(train_slice, noisy_pred) -> None:
    era = train_slice["era"]
    zp, eras = M.pad_by_era(noisy_pred.to_numpy(), era)
    t, _ = M.pad_by_era(train_slice[TARGET].to_numpy(), era)
    z = M.rank_gauss_rows(zp)
    tc = t - np.nanmean(t, axis=1, keepdims=True)
    fit_vec = pd.Series(M.fit_rows(z, tc), index=eras)
    fit_sdk = M.per_era_fit(noisy_pred, train_slice[TARGET], era)
    np.testing.assert_allclose(fit_vec.to_numpy(), fit_sdk.to_numpy(), atol=1e-12)


def test_neutralise_rows_algebra() -> None:
    rng = np.random.default_rng(8)
    z = rng.standard_normal((2, 50))
    m = rng.standard_normal((2, 50))
    m -= m.mean(axis=1, keepdims=True)
    out = M.neutralise_rows(z, m, 1.0)
    assert np.allclose((out * m).sum(axis=1), 0.0, atol=1e-10)  # full neutralisation -> orthogonal
    half = M.neutralise_rows(z, m, 0.5)
    np.testing.assert_allclose(half, 0.5 * z + 0.5 * out, atol=1e-12)
    assert M.neutralise_rows(z, m, 0.0) is z
    zero_m = M.neutralise_rows(z, np.zeros_like(m), 0.5)
    np.testing.assert_allclose(zero_m, z)
    try:
        from stealth import _runtime

        rt = _runtime.neutralise(z[0], m[0], 0.5)
        np.testing.assert_allclose(rt, half[0], atol=1e-10)
    except NotImplementedError:
        pytest.skip("_runtime.neutralise not implemented yet (other group)")


# ----------------------------------------------------------------------------------- regime simulation
@pytest.fixture(scope="module")
def sim_inputs(train_slice, noisy_pred, bench_slice):
    """Small slice (60 eras) with a gaussianised prediction, a direction and a core proxy."""
    from stealth.features import core_proxy_series

    sl = train_slice[train_slice["era"] >= 6465]
    era = sl["era"]
    zp, eras = M.pad_by_era(noisy_pred.reindex(sl.index).to_numpy(), era)
    zg = M.rank_gauss_rows(zp)
    z = np.empty(len(sl))
    for i, (_, idx) in enumerate(M.era_groups(era)):
        z[idx] = zg[i, : len(idx)]
    z_pred = pd.Series(z, index=sl.index)
    direction = bench_slice.reindex(sl.index)  # the real (in-sample) benchmark as the direction
    core = core_proxy_series(sl, feature_columns(sl)[:10])
    return sl, z_pred, direction, core


def test_regime_simulation_structure_and_pb0_identity(sim_inputs) -> None:
    sl, z_pred, direction, core = sim_inputs
    res = M.regime_simulation(z_pred, sl[TARGET], sl["era"], direction, core, p_c=0.0, n_draws=6, seed=0,
                              bench_fit_means=(0.0, 0.009), p_b_grid=(0.0, 0.5))
    assert set(res) == {"0", "0.009", "meta"}
    for mean_key in ("0", "0.009"):
        assert set(res[mean_key]) == {"p_b=0", "p_b=0.5"}
        for pb in res[mean_key].values():
            assert set(pb) == {"bpos", "bneg", "mix"}
            for blk in pb.values():
                assert {"fit", "unq", "inov", "score", "fit_se", "unq_se", "inov_se", "score_se", "n_eras", "frac"} <= set(blk)
            assert pb["bpos"]["frac"] + pb["bneg"]["frac"] == pytest.approx(1.0)
            assert pb["mix"]["frac"] == 1.0 and pb["mix"]["n_eras"] == 60
            assert pb["mix"]["score"] == pytest.approx(pb["mix"]["fit"] + pb["mix"]["unq"] + pb["mix"]["inov"])
    # p_b = 0 and p_c = 0 leave the prediction untouched -> FIT equals the SDK per-era FIT exactly
    fit = M.per_era_fit(z_pred, sl[TARGET], sl["era"])
    assert res["0.009"]["p_b=0"]["mix"]["fit"] == pytest.approx(fit.mean(), abs=1e-9)
    assert res["0"]["p_b=0"]["mix"]["fit"] == pytest.approx(fit.mean(), abs=1e-9)
    meta = res["meta"]
    assert meta["n_draws"] == 6 and meta["n_eras"] == 60 and meta["p_c"] == 0.0
    assert set(meta["bench"]) == {"0", "0.009"}
    table = M.regime_table(res)
    assert len(table) == 4 and set(table.columns) >= {"bench_mean", "p_b", "fit", "unq", "inov", "score"}


def test_regime_simulation_benchmark_calibration_and_determinism(sim_inputs) -> None:
    sl, z_pred, direction, core = sim_inputs
    kw = dict(p_c=0.25, n_draws=40, seed=3, bench_fit_means=(0.0, 0.02), p_b_grid=(0.0, 0.5))
    res = M.regime_simulation(z_pred, sl[TARGET], sl["era"], direction, core, **kw)
    b0, b2 = res["meta"]["bench"]["0"], res["meta"]["bench"]["0.02"]
    # realised benchmark FIT tracks the requested N(mean, 0.035^2) (SE over 40x60 cells ~ 0.0007)
    assert abs(b0["realised_fit_mean"] - 0.0) < 0.004
    assert abs(b2["realised_fit_mean"] - 0.02) < 0.004
    assert 0.025 < b0["realised_fit_sd"] < 0.045
    assert 0.4 < b0["frac_pos"] < 0.6 and b2["frac_pos"] > b0["frac_pos"]
    # by construction bpos eras have unq possibly non-zero; bneg eras have unq == 0 (zero rule)
    assert res["0.02"]["p_b=0.5"]["bneg"]["unq"] == 0.0
    # neutralising vs a benchmark that is right on average costs FIT when its mean is 0.02 ...
    assert res["0.02"]["p_b=0.5"]["bpos"]["fit"] < res["0.02"]["p_b=0"]["bpos"]["fit"]
    # ... and helps in benchmark-negative eras
    assert res["0.02"]["p_b=0.5"]["bneg"]["fit"] > res["0.02"]["p_b=0"]["bneg"]["fit"]
    again = M.regime_simulation(z_pred, sl[TARGET], sl["era"], direction, core, **kw)
    assert again == res


def test_regime_simulation_validates_args(sim_inputs) -> None:
    sl, z_pred, direction, core = sim_inputs
    with pytest.raises(ValueError):
        M.regime_simulation(z_pred, sl[TARGET], sl["era"], direction, core, p_c=0.9, n_draws=1)
    with pytest.raises(ValueError):
        M.regime_simulation(z_pred, sl[TARGET], sl["era"], direction, core, p_c=0.1, p_b_grid=(0.9,), n_draws=1)
    with pytest.raises(ValueError):
        M.regime_simulation(z_pred, sl[TARGET], sl["era"], direction, core, p_c=0.1, n_draws=0)


# ------------------------------------------------------------------------- M5 additions (INTERFACES 17)
def _rank_gauss_rows_direct(a: np.ndarray) -> np.ndarray:
    """The pre-lookup implementation: rankdata + norm.ppf computed directly."""
    from scipy import stats

    r = stats.rankdata(a, axis=1, method="average", nan_policy="omit")
    n = np.sum(~np.isnan(a), axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        u = (r - 0.5) / n
    z = stats.norm.ppf(np.clip(u, 0.001, 0.999))
    z[np.isnan(a)] = np.nan
    return z


def test_rank_gauss_rows_lookup_is_bit_identical() -> None:
    rng = np.random.default_rng(21)
    a = rng.integers(0, 6, size=(40, 93)).astype(float)  # heavy ties
    a[:, 70:] = np.nan                                    # padding
    a[3, :] = np.nan                                      # an all-NaN row
    a[5, 10] = np.nan                                     # an interior NaN
    b = rng.standard_normal((7, 2))                       # tiny width
    for x in (a, b, rng.standard_normal((5, 91))):
        np.testing.assert_array_equal(M.rank_gauss_rows(x), _rank_gauss_rows_direct(x))


def test_block_bootstrap_index_matches_cv_generator() -> None:
    from stealth.cv import block_bootstrap_means

    v = np.random.default_rng(3).standard_normal(157)
    idx = M._block_bootstrap_index(len(v), block=20, n_boot=50, seed=4)
    np.testing.assert_allclose(v[idx].mean(axis=1), block_bootstrap_means(v, block=20, n_boot=50, seed=4), atol=0, rtol=0)
    counts = M._bootstrap_counts(len(v), block=20, n_boot=50, seed=4)
    assert counts.shape == (50, 157) and np.all(counts.sum(axis=1) == 157)
    np.testing.assert_allclose(counts @ v / 157, v[idx].mean(axis=1), atol=1e-12)


def test_score_sharpe_helper_matches_naive_loop() -> None:
    rng = np.random.default_rng(9)
    score = 0.02 + 0.05 * rng.standard_normal((6, 80))
    mask = rng.random((6, 80)) > 0.4
    mask[2] = False
    mask[2, 5] = True  # a draw with a single era is skipped
    counts = M._bootstrap_counts(80, n_boot=100)
    s, se = M._score_sharpe(score, mask, counts)
    naive = [score[k][mask[k]].mean() / score[k][mask[k]].std(ddof=1) for k in range(6) if mask[k].sum() >= 2]
    assert s == pytest.approx(float(np.mean(naive)), abs=1e-12)
    assert np.isfinite(se) and 0.0 < se < 0.5
    s_all, _ = M._score_sharpe(score, np.ones_like(mask), None)
    assert s_all == pytest.approx(float(np.mean(score.mean(axis=1) / score.std(axis=1, ddof=1))), abs=1e-12)


def test_regime_grid_structure_crn_and_single_pc_consistency(sim_inputs) -> None:
    sl, z_pred, direction, core = sim_inputs
    kw = dict(n_draws=5, seed=2, bench_fit_means=(0.0, 0.02), p_b_grid=(0.0, 0.5))
    grid = M.regime_grid(z_pred, sl[TARGET], sl["era"], direction, core, p_c_grid=(0.0, 0.25, 0.5), **kw)
    assert set(grid) == {"0", "0.02", "meta"}
    for mean_key in ("0", "0.02"):
        assert set(grid[mean_key]) == {"p_c=0", "p_c=0.25", "p_c=0.5"}
        for by_pb in grid[mean_key].values():
            assert set(by_pb) == {"p_b=0", "p_b=0.5"}
            for regs in by_pb.values():
                assert set(regs) == {"bpos", "bneg", "mix"}
                for blk in regs.values():
                    assert {"fit", "unq", "inov", "score", "score_se", "score_sharpe", "score_sharpe_se", "n_eras", "frac"} <= set(blk)
                mix = regs["mix"]
                assert np.isfinite(mix["score_sharpe"]) and mix["score_sharpe_se"] > 0
    meta = grid["meta"]
    assert meta["p_c_grid"] == [0.0, 0.25, 0.5] and meta["p_b_grid"] == [0.0, 0.5] and meta["common_random_numbers"] is True
    # single-p_c simulation == the grid's slice (same draws: common random numbers across cells)
    for p_c in (0.0, 0.25):
        single = M.regime_simulation(z_pred, sl[TARGET], sl["era"], direction, core, p_c=p_c, **kw)
        assert single == M.regime_slice(grid, p_c)
        assert single["meta"]["p_c"] == p_c and single["meta"]["bench"] == meta["bench"]
    # a cell's numbers do not depend on which other cells are simulated alongside it
    small = M.regime_grid(z_pred, sl[TARGET], sl["era"], direction, core, p_c_grid=(0.5,), **kw)
    assert small["0.02"]["p_c=0.5"] == grid["0.02"]["p_c=0.5"] and small["meta"]["bench"] == meta["bench"]
    # p_b = 0 cells: fit/inov do not depend on the benchmark draw; p_c = 0 leaves the prediction untouched
    fit = M.per_era_fit(z_pred, sl[TARGET], sl["era"])
    for mean_key in ("0", "0.02"):
        assert grid[mean_key]["p_c=0"]["p_b=0"]["mix"]["fit"] == pytest.approx(fit.mean(), abs=1e-9)
    assert grid["0"]["p_c=0.5"]["p_b=0"]["mix"]["inov"] == pytest.approx(grid["0.02"]["p_c=0.5"]["p_b=0"]["mix"]["inov"], abs=1e-15)
    table = M.regime_grid_table(grid)
    assert len(table) == 2 * 3 * 2 and {"p_c", "p_b", "score_sharpe"} <= set(table.columns)
    with pytest.raises(ValueError):
        M.regime_grid(z_pred, sl[TARGET], sl["era"], direction, core, p_c_grid=(0.0, 0.75), **kw)
    with pytest.raises(ValueError):
        M.regime_grid(z_pred, sl[TARGET], sl["era"], direction, core, p_c_grid=(0.0, 0.0), **kw)
    with pytest.raises(ValueError):
        M.regime_grid(z_pred, sl[TARGET], sl["era"], direction, core, p_c_grid=(), **kw)


def test_paired_sharpe_bootstrap_sign_and_p_pos() -> None:
    rng = np.random.default_rng(12)
    idx = pd.Index(np.arange(600, dtype=np.int32), name="era")
    common = rng.standard_normal(600)
    b = pd.Series(0.02 + 0.05 * common + 0.03 * rng.standard_normal(600), index=idx)
    a = pd.Series(0.02 + 0.02 * common + 0.01 * rng.standard_normal(600), index=idx)  # same mean, ~2.5x less risk
    res = M.paired_sharpe_bootstrap(a, b, n_boot=500)
    assert set(res) >= {"dsharpe", "se", "p_pos", "ci90", "n_eras"} and res["n_eras"] == 600
    assert res["dsharpe"] > 0.2 and res["p_pos"] > 0.95 and res["se"] > 0
    assert res["ci90"][0] < res["dsharpe"] < res["ci90"][1]
    assert res["dsharpe"] == pytest.approx(a.mean() / a.std() - b.mean() / b.std(), abs=1e-12)
    rev = M.paired_sharpe_bootstrap(b, a, n_boot=500)
    assert rev["dsharpe"] == pytest.approx(-res["dsharpe"]) and rev["p_pos"] < 0.05
    same = M.paired_sharpe_bootstrap(a, a, n_boot=200)
    assert same["dsharpe"] == 0.0 and same["p_pos"] == 0.0
    # inner join on the era index, NaN eras dropped, order irrelevant
    shuffled = M.paired_sharpe_bootstrap(a.sample(frac=1.0, random_state=1).iloc[:550], b.where(b.index != 3), n_boot=200)
    assert shuffled["n_eras"] == 550 - int(3 in set(a.sample(frac=1.0, random_state=1).iloc[:550].index))
    # deterministic for a seed
    assert M.paired_sharpe_bootstrap(a, b, n_boot=300, seed=5) == M.paired_sharpe_bootstrap(a, b, n_boot=300, seed=5)
    with pytest.raises(ValueError):
        M.paired_sharpe_bootstrap(a.iloc[:2], b.iloc[:2])
