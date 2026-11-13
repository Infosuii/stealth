"""Tests for stealth.data: era helpers, loaders, input frames, filters, download gating (offline)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stealth import data as D


def test_constants() -> None:
    assert D.PRIMARY_TARGET == "target_everest"
    assert len(D.AUX_TARGETS) == 6 and all(t.startswith("target_") for t in D.AUX_TARGETS)
    assert D.N_FEATURES == 178
    assert D.DATA_DIR.name == "data"


def test_era_int() -> None:
    assert D.era_int("exped_0003") == 3
    assert D.era_int("exped_8055") == 8055
    with pytest.raises((ValueError, IndexError)):
        D.era_int("garbage")


def test_add_era_int32() -> None:
    df = pd.DataFrame({"exped": ["exped_0003", "exped_6524", "exped_8055"]})
    out = D.add_era(df)
    assert out is df
    assert str(df["era"].dtype) == "int32"
    assert df["era"].tolist() == [3, 6524, 8055]


def test_feature_and_target_columns(train_slice: pd.DataFrame) -> None:
    cols = D.feature_columns(train_slice)
    assert len(cols) == D.N_FEATURES
    assert cols == [c for c in train_slice.columns if c.startswith("feature_")]  # file order
    tcols = D.target_columns(train_slice)
    assert len(tcols) == 15 and D.PRIMARY_TARGET in tcols and set(D.AUX_TARGETS) <= set(tcols)


def test_load_split_live_and_benchmark(data_dir) -> None:
    live = D.load_split("live", data_dir)
    assert live.index.name == "id" and len(live) == 90
    assert str(live["era"].dtype) == "int32" and live["era"].unique().tolist() == [8055]
    assert str(live["feature_Adfel"].dtype) == "int8"
    assert not any(c in ("id", "index", "target") for c in live.columns)
    sub = D.load_split("live", data_dir, columns=["feature_Adfel", "target_everest"])
    assert set(sub.columns) == {"exped", "feature_Adfel", "target_everest", "era"}
    bench = D.load_benchmark("live", data_dir)
    assert bench.name == "v1_sherpa" and bench.dtype == np.float64 and bench.index.name == "id"
    assert set(bench.index) == set(live.index)


def test_load_split_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        D.load_split("train", tmp_path)
    with pytest.raises(FileNotFoundError):
        D.load_benchmark("train", tmp_path)


def test_filter_eras(train_slice: pd.DataFrame) -> None:
    out = D.filter_eras(train_slice, min_era=6300, max_era=6400)
    assert out["era"].min() == 6300 and out["era"].max() == 6400
    counts = D.era_counts(train_slice)
    assert counts.index.dtype == np.int32 and counts.sum() == len(train_slice)
    thresh = int(counts.median())
    out2 = D.filter_eras(train_slice, min_rows=thresh)
    assert D.era_counts(out2).min() >= thresh
    assert set(out2["era"]) == set(counts.index[counts >= thresh])


def test_features_nan(train_slice: pd.DataFrame) -> None:
    cols = D.feature_columns(train_slice)
    x = D.features_nan(train_slice, cols)
    assert x.shape == (len(train_slice), len(cols)) and x.index.equals(train_slice.index)
    assert all(str(dt) == "float32" for dt in x.dtypes)
    raw = train_slice[cols].to_numpy()
    assert np.isnan(x.to_numpy()).sum() == (raw == -1).sum()
    assert np.nanmin(x.to_numpy()) == 0.0 and np.nanmax(x.to_numpy()) == 9.0
    assert (train_slice[cols].to_numpy() == raw).all()  # input untouched


def test_features_linear(train_slice: pd.DataFrame) -> None:
    cols = D.feature_columns(train_slice)
    miss = D.top_missing_columns(train_slice, cols, k=15)
    assert len(miss) == 15 and len(set(miss)) == 15
    rates = D.missing_rate(train_slice, cols)
    assert rates[miss].min() >= rates.drop(miss).max()
    x = D.features_linear(train_slice, cols, miss)
    assert x.shape == (len(train_slice), len(cols) + 15 + 1)
    assert all(str(dt) == "float32" for dt in x.dtypes)
    assert list(x.columns[: len(cols)]) == cols
    assert [c for c in x.columns if c.startswith("miss_")] == [f"miss_{c}" for c in miss]
    v = x.to_numpy()
    assert np.isfinite(v).all() and v.min() >= 0.0 and v.max() <= 1.0
    raw = train_slice[cols].to_numpy()
    expect = np.where(raw == -1, 0.5, raw / 9.0)
    np.testing.assert_allclose(x[cols].to_numpy(), expect, atol=1e-6)
    np.testing.assert_allclose(x["n_missing"].to_numpy(), (raw == -1).mean(axis=1), atol=1e-6)
    j = miss[0]
    assert (x[f"miss_{j}"].to_numpy() == (train_slice[j].to_numpy() == -1)).all()


def test_time_decay_weights() -> None:
    era = pd.Series([10, 20, 30, 30, 40], dtype="int32")
    w = D.time_decay_weights(era, True)
    np.testing.assert_allclose(w, [0.5, 0.6666667, 0.8333333, 0.8333333, 1.0], atol=1e-6)
    assert (D.time_decay_weights(era, False) == 1.0).all()
    assert (D.time_decay_weights(pd.Series([7, 7]), True) == 1.0).all()


def test_download_requires_key(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("EIQ_API_KEY", raising=False)
    monkeypatch.delenv("EVEREST_API_KEY", raising=False)
    monkeypatch.setattr(D, "DOTENV_PATH", tmp_path / "nonexistent.env")
    with pytest.raises(RuntimeError, match="EIQ_API_KEY not set"):
        D.download(tmp_path)
    assert D.main(["download", "--data-dir", str(tmp_path)]) == 1
    assert not list(tmp_path.glob("*.parquet"))


def test_dotenv_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("EIQ_API_KEY", raising=False)
    monkeypatch.delenv("EVEREST_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("# comment\nEIQ_API_KEY=dummy-test-key\nEIQ_BASE_URL=https://example.invalid\n")
    assert D.resolve_api_key(env) == "dummy-test-key"
