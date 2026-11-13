"""Tests for stealth.features and stealth.surrogate: nested subsets, core proxy (eras < max_era only), bags."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from stealth import features as F
from stealth import surrogate as S
from stealth.data import feature_columns, features_linear, features_nan, top_missing_columns

TARGET = "target_everest"


@pytest.fixture(scope="module")
def nan_frame(train_slice: pd.DataFrame) -> pd.DataFrame:
    return features_nan(train_slice, feature_columns(train_slice))


def test_per_era_feature_spearman_matches_scipy(train_slice, nan_frame) -> None:
    cols = ["feature_Agouti", "feature_Azilal", "feature_Azzaden"]
    corr = F.per_era_feature_spearman(nan_frame[cols], train_slice[TARGET], train_slice["era"])
    assert corr.shape == (300, 3) and corr.index.name == "era" and corr.index.dtype == np.int32
    rng = np.random.default_rng(0)
    for e in rng.choice(corr.index.to_numpy(), size=6, replace=False):
        m = (train_slice["era"] == e).to_numpy()
        for c in cols:
            x = nan_frame.loc[m, c].to_numpy()
            y = train_slice.loc[m, TARGET].to_numpy()
            ok = ~np.isnan(x)
            expect = spearmanr(x[ok], y[ok]).statistic if ok.sum() >= 3 else np.nan
            if np.isnan(expect):
                assert np.isnan(corr.loc[e, c])
            else:
                assert corr.loc[e, c] == pytest.approx(expect, abs=1e-12)
    # raw int8 input (with -1) gives the same answer as the NaN frame
    corr_raw = F.per_era_feature_spearman(train_slice[cols], train_slice[TARGET], train_slice["era"])
    pd.testing.assert_frame_equal(corr, corr_raw)


def test_sharpe_subset_uses_given_rows_only(train_slice, nan_frame) -> None:
    era = train_slice["era"]
    y = train_slice[TARGET]
    keep = F.sharpe_subset(nan_frame, y, era, threshold=0.1)
    assert 0 < len(keep) < nan_frame.shape[1]
    assert keep == [c for c in nan_frame.columns if c in set(keep)]  # column order preserved
    strict = F.sharpe_subset(nan_frame, y, era, threshold=0.3)
    assert set(strict) <= set(keep) and len(strict) < len(keep)
    # only the passed rows matter: scrambling the target on rows NOT passed changes nothing
    train_rows = (era < 6400).to_numpy()
    base = F.sharpe_subset(nan_frame.loc[train_rows], y.loc[train_rows], era.loc[train_rows])
    y2 = y.copy()
    y2.loc[~train_rows] = y2.loc[~train_rows].sample(frac=1.0, random_state=1).to_numpy()
    assert F.sharpe_subset(nan_frame.loc[train_rows], y2.loc[train_rows], era.loc[train_rows]) == base
    # the hand computation agrees
    corr = F.per_era_feature_spearman(nan_frame, y, era)
    sharpe = corr.mean() / corr.std(ddof=1)
    assert keep == [c for c in nan_frame.columns if abs(sharpe[c]) > 0.1]


def test_core_proxy_columns_only_uses_eras_below_max_era(train_slice, nan_frame, bench_slice) -> None:
    era = train_slice["era"]
    cols = F.core_proxy_columns(nan_frame, bench_slice, era, k=10, max_era=6400)
    assert len(cols) == 10 and len(set(cols)) == 10 and all(c.startswith("feature_") for c in cols)
    # scramble the benchmark on eras >= max_era: result must be identical
    bad = bench_slice.copy()
    late = (era >= 6400).to_numpy()
    bad.loc[late] = bad.loc[late].sample(frac=1.0, random_state=2).to_numpy()
    assert F.core_proxy_columns(nan_frame, bad, era, k=10, max_era=6400) == cols
    # scrambling eras < max_era DOES change it (sanity that the test has teeth)
    bad2 = bench_slice.copy()
    bad2.loc[~late] = bad2.loc[~late].sample(frac=1.0, random_state=3).to_numpy()
    assert F.core_proxy_columns(nan_frame, bad2, era, k=10, max_era=6400) != cols
    # loadings sorted by |value| desc and the benchmark's known top features show up
    load = F.feature_loadings(nan_frame, bench_slice, era, max_era=6400)
    assert np.all(np.diff(load.abs().to_numpy()) <= 1e-12)
    assert {"feature_Agouti", "feature_Tazzarine", "feature_Azzaden"} <= set(cols)
    with pytest.raises(ValueError):
        F.core_proxy_columns(nan_frame, bench_slice, era, max_era=2800)  # fixture has no eras < 2800


def test_core_proxy_columns_on_real_eras_below_2800(data_dir: Path) -> None:
    """The production call: TRAIN benchmark file, eras < 2800 only. Must overlap the known v1_sherpa loadings."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from stealth.data import add_era

    df = pq.read_table(data_dir / "futures_train.parquet", filters=pc.field("exped") < "exped_2800").to_pandas()
    add_era(df)
    assert df["era"].max() < 2800
    bench = pd.read_parquet(data_dir / "futures_train_benchmark_models.parquet", columns=["v1_sherpa"])["v1_sherpa"]
    cols = F.core_proxy_columns(features_nan(df, feature_columns(df)), bench.reindex(df.index), df["era"], k=10)
    expected_core = {"feature_Tazzarine", "feature_Agouti", "feature_Azzaden", "feature_Ighzer", "feature_Tafouk",
                     "feature_Amazigh", "feature_Takerkoust", "feature_Ouarzazate", "feature_Mrirt", "feature_Azaghar"}
    assert len(set(cols) & expected_core) >= 6, cols


def test_core_proxy_series(train_slice) -> None:
    cols = feature_columns(train_slice)[:5]
    s = F.core_proxy_series(train_slice, cols)
    assert s.index.equals(train_slice.index) and s.name == "core_proxy"
    x = train_slice[cols].to_numpy(dtype=float, copy=True)
    x[x == -1] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-missing rows -> NaN by design
        expect = np.nanmean(x, axis=1)
    np.testing.assert_allclose(s.to_numpy(), expect, equal_nan=True)
    # SDK parity
    from everestapi.scoring import core_feature_average

    np.testing.assert_allclose(s.to_numpy(), core_feature_average(train_slice[cols]).to_numpy(), equal_nan=True)
    df = pd.DataFrame({"a": [-1, 3], "b": [-1, -1]}, index=["x", "y"]).astype("int8")
    s2 = F.core_proxy_series(df, ["a", "b"])
    assert np.isnan(s2["x"]) and s2["y"] == 3.0


def test_random_column_bags() -> None:
    cols = [f"feature_{i:03d}" for i in range(178)]
    bags = F.random_column_bags(cols, n_bags=4, fraction=0.5, seed=0)
    assert len(bags) == 4 and all(len(b) == 89 for b in bags)
    for b in bags:
        assert b == [c for c in cols if c in set(b)]  # original order
        assert len(set(b)) == 89
    assert bags != F.random_column_bags(cols, n_bags=4, fraction=0.5, seed=1)
    assert bags == F.random_column_bags(cols, n_bags=4, fraction=0.5, seed=0)
    assert len({tuple(b) for b in bags}) == 4
    with pytest.raises(ValueError):
        F.random_column_bags(cols, fraction=0.0)


def test_resolve_feature_set() -> None:
    cols = [f"feature_{i}" for i in range(6)]
    bags = F.random_column_bags(cols, n_bags=2, fraction=0.5, seed=0)
    assert F.resolve_feature_set("all", cols) == cols
    assert F.resolve_feature_set("sharpe_subset", cols, sharpe_cols=cols[:2]) == cols[:2]
    assert F.resolve_feature_set("bag:1", cols, bags=bags) == bags[1]
    assert F.resolve_feature_set(cols[2:4], cols) == cols[2:4]
    for bad in ("sharpe_subset", "bag:0", "bag:7", "nonsense"):
        with pytest.raises(ValueError):
            F.resolve_feature_set(bad, cols, bags=bags if bad == "bag:7" else None)
    with pytest.raises(ValueError):
        F.resolve_feature_set(["feature_99"], cols)
    with pytest.raises(ValueError):
        F.resolve_feature_set([], cols)


def test_surrogate_fit_predict_and_era_restriction(train_slice, bench_slice) -> None:
    cols = feature_columns(train_slice)
    miss = top_missing_columns(train_slice, cols)
    X_lin = features_linear(train_slice, cols, miss)
    era = train_slice["era"]
    sur = S.fit_surrogate(X_lin, bench_slice, era, max_era=6400, alpha=1e4)
    assert sur.columns == list(X_lin.columns) and sur.coef.shape == (X_lin.shape[1],)
    pred = sur.predict(X_lin)
    assert pred.index.equals(X_lin.index) and np.isfinite(pred.to_numpy()).all()
    # a direction, not a toy: correlates with the benchmark on the held-out eras >= 6400
    from stealth.metrics import per_era_corr

    held = (era >= 6400).to_numpy()
    rho = per_era_corr(pred[held], bench_slice[held], era[held]).mean()
    assert rho > 0.5
    # rows with era >= max_era never influence the fit
    bad = bench_slice.copy()
    bad.loc[held] = bad.loc[held].sample(frac=1.0, random_state=4).to_numpy()
    sur2 = S.fit_surrogate(X_lin, bad, era, max_era=6400, alpha=1e4)
    np.testing.assert_allclose(sur2.coef, sur.coef)
    assert sur2.intercept == pytest.approx(sur.intercept)
    # round trip + column check
    back = S.Surrogate.from_dict(sur.to_dict())
    pd.testing.assert_series_equal(back.predict(X_lin), pred)
    with pytest.raises(KeyError):
        sur.predict(X_lin.iloc[:, :10])
    with pytest.raises(ValueError):
        S.fit_surrogate(X_lin, bench_slice, era, max_era=2800)
