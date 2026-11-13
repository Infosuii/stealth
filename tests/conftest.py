"""Shared pytest fixtures.

``train_slice`` loads only the LAST ``N_ERAS`` eras of data/futures_train.parquet (plus the matching
benchmark rows) so that tests stay fast (< 2 s load, ~27k rows). Tests needing data are skipped when
the parquet files are absent (CI without data). Everything here is offline: no API key is needed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
N_ERAS = 300  # last 300 eras of train (exped_6225..exped_6524)


def _era_int(exped: str) -> int:
    return int(exped.rsplit("_", 1)[1])


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="session")
def data_dir() -> Path:
    if not (DATA_DIR / "futures_train.parquet").exists():
        pytest.skip("data/futures_train.parquet missing; run `python -m stealth.data download`")
    return DATA_DIR


@pytest.fixture(scope="session")
def train_slice(data_dir: Path) -> pd.DataFrame:
    """Last N_ERAS eras of the train split with an int32 ``era`` column added; index 'id'."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(data_dir / "futures_train.parquet", columns=["exped"])
    eras = sorted({_era_int(e) for e in table.column("exped").unique().to_pylist()})
    keep = eras[-N_ERAS:]
    first = f"exped_{keep[0]:04d}"
    df = pq.read_table(
        data_dir / "futures_train.parquet",
        filters=pc.field("exped") >= first,
    ).to_pandas()
    df["era"] = df["exped"].map(_era_int).astype("int32")
    assert df.index.name == "id"
    return df


@pytest.fixture(scope="session")
def bench_slice(data_dir: Path, train_slice: pd.DataFrame) -> pd.Series:
    """v1_sherpa (float64) aligned to ``train_slice.index``."""
    b = pd.read_parquet(data_dir / "futures_train_benchmark_models.parquet", columns=["v1_sherpa"])
    return b["v1_sherpa"].reindex(train_slice.index)


@pytest.fixture(scope="session")
def live_frame(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "futures_live.parquet")


@pytest.fixture(scope="session")
def live_bench(data_dir: Path) -> pd.Series:
    b = pd.read_parquet(data_dir / "futures_live_benchmark_models.parquet")
    return b["v1_sherpa"].rename("eiq_minera_model")


@pytest.fixture(scope="session")
def smoke_config(repo_root: Path) -> dict:
    import yaml

    with (repo_root / "configs" / "smoke.yaml").open() as fh:
        return yaml.safe_load(fh)
