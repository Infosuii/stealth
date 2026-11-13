"""Data access: SDK download, parquet load, -1 -> NaN, era ints, era/n_rows filters, missing indicators.

Contracts: see docs/INTERFACES.md section 1. Everything except :func:`download` runs OFFLINE from
``data/``. Conventions enforced here:

* index name ``id`` (opaque strings, never renumbered); any *column* literally named ``id``, ``index``
  or ``target`` is dropped on load;
* ``era`` is an ``int32`` column derived from ``exped`` (``exped_0003`` -> 3);
* features are ``int8`` on disk, ``-1`` = missing. :func:`features_nan` (trees) maps -1 -> NaN float32;
  :func:`features_linear` (ridge) maps to value/9 with -1 -> 0.5 plus missing indicators.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

Split = Literal["train", "validation", "live"]

PRIMARY_TARGET = "target_everest"
AUX_TARGETS = [
    "target_Tougroute",
    "target_Ghat",
    "target_Meltsene",
    "target_Timesguida",
    "target_Ighil",
    "target_Tarhat",
]
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DOTENV_PATH = REPO_ROOT / ".env"
N_FEATURES = 178
BENCHMARK_COL = "v1_sherpa"
DEFAULT_BASE_URL = "https://api.everesteer.ai"
SPLITS: tuple[Split, ...] = ("train", "validation", "live")
_DROP_COLUMNS = ("id", "index", "target")
_MISSING = -1


# --------------------------------------------------------------------------------------------------
# paths / era helpers
# --------------------------------------------------------------------------------------------------
def split_path(split: Split, data_dir: Path = DATA_DIR) -> Path:
    """Path of the feature/target parquet for ``split``."""
    return Path(data_dir) / f"futures_{split}.parquet"


def benchmark_path(split: Split, data_dir: Path = DATA_DIR) -> Path:
    """Path of the benchmark-model parquet for ``split``."""
    return Path(data_dir) / f"futures_{split}_benchmark_models.parquet"


def era_int(exped: str) -> int:
    """``"exped_0003" -> 3``. Splits on the last underscore; raises ``ValueError`` on garbage."""
    return int(str(exped).rsplit("_", 1)[1])


def add_era(df: pd.DataFrame) -> pd.DataFrame:
    """Add/overwrite the ``int32`` ``era`` column from ``exped``; returns the same object."""
    if "exped" not in df.columns:
        raise KeyError("add_era needs an 'exped' column")
    df["era"] = df["exped"].astype(str).str.rsplit("_", n=1).str[-1].astype("int32")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """``feature_*`` columns in FILE order (the bundle stores this list verbatim)."""
    return [c for c in df.columns if c.startswith("feature_")]


def target_columns(df: pd.DataFrame) -> list[str]:
    """``target_*`` columns in file order."""
    return [c for c in df.columns if c.startswith("target_")]


# --------------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------------
def _ensure_id_index(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee the frame is indexed by ``id`` and carries no stray id/index/target columns."""
    if df.index.name != "id":
        if "id" in df.columns:
            df = df.set_index("id")
        else:
            raise ValueError(f"parquet has no 'id' index/column (index name={df.index.name!r})")
    drop = [c for c in df.columns if c in _DROP_COLUMNS]
    if drop:
        log.warning("dropping stray columns %s", drop)
        df = df.drop(columns=drop)
    if not df.index.is_unique:
        raise ValueError("'id' index is not unique")
    return df


def load_split(
    split: Split,
    data_dir: Path = DATA_DIR,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load ``data/futures_<split>.parquet`` and add the ``int32`` ``era`` column.

    ``columns`` restricts the columns read (``exped`` is always read so ``era`` can be built).
    Index is ``id``; features stay ``int8`` (-1 = missing); targets stay ``float32``.
    """
    path = split_path(split, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m stealth.data download`")
    cols = None
    if columns is not None:
        cols = list(dict.fromkeys(["exped", *columns]))
    df = pd.read_parquet(path, columns=cols)
    df = _ensure_id_index(df)
    add_era(df)
    log.info("loaded %s: %d rows, %d eras (%d..%d)", path.name, len(df), df["era"].nunique(),
             int(df["era"].min()), int(df["era"].max()))
    return df


def load_benchmark(split: Split, data_dir: Path = DATA_DIR) -> pd.Series:
    """``v1_sherpa`` as a float64 Series named ``v1_sherpa`` indexed by ``id``."""
    path = benchmark_path(split, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m stealth.data download`")
    df = pd.read_parquet(path, columns=[BENCHMARK_COL])
    df = _ensure_id_index(df)
    return df[BENCHMARK_COL].astype("float64").rename(BENCHMARK_COL)


def era_counts(df: pd.DataFrame) -> pd.Series:
    """Rows per era, indexed by era (int32, sorted)."""
    counts = df["era"].value_counts().sort_index()
    counts.index = counts.index.astype("int32")
    return counts.rename("n_rows")


def filter_eras(
    df: pd.DataFrame,
    *,
    min_era: int | None = None,
    max_era: int | None = None,
    min_rows: int | None = None,
) -> pd.DataFrame:
    """Keep eras in ``[min_era, max_era]`` and drop eras with fewer than ``min_rows`` rows."""
    mask = np.ones(len(df), dtype=bool)
    era = df["era"].to_numpy()
    if min_era is not None:
        mask &= era >= min_era
    if max_era is not None:
        mask &= era <= max_era
    if min_rows is not None:
        counts = era_counts(df)
        small = counts.index[counts.to_numpy() < min_rows].to_numpy()
        if len(small):
            mask &= ~np.isin(era, small)
    out = df.loc[mask]
    log.info("filter_eras(min_era=%s, max_era=%s, min_rows=%s): %d -> %d rows", min_era, max_era, min_rows,
             len(df), len(out))
    return out


# --------------------------------------------------------------------------------------------------
# model inputs
# --------------------------------------------------------------------------------------------------
def features_nan(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Tree input: ``cols`` as float32 with -1 -> NaN (LightGBM/XGBoost/CatBoost native missing)."""
    x = df[cols].to_numpy(dtype=np.float32, copy=True)
    x[x == _MISSING] = np.nan
    return pd.DataFrame(x, index=df.index, columns=list(cols))


def missing_rate(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Fraction of -1 per feature over the rows of ``df`` (caller restricts to training eras)."""
    x = df[cols].to_numpy()
    return pd.Series((x == _MISSING).mean(axis=0), index=list(cols), name="missing_rate")


def top_missing_columns(df: pd.DataFrame, cols: list[str], k: int = 15) -> list[str]:
    """The ``k`` features with the highest -1 rate on ``df`` (ties broken by file order)."""
    rate = missing_rate(df, cols)
    order = np.lexsort((np.arange(len(cols)), -rate.to_numpy()))
    return [cols[i] for i in order[:k]]


def features_linear(df: pd.DataFrame, cols: list[str], missing_cols: list[str]) -> pd.DataFrame:
    """Linear input (float32): ``cols`` as value/9 with -1 -> 0.5, one ``miss_<feature>`` 0/1 indicator per
    name in ``missing_cols`` and ``n_missing`` = (# -1 across ``cols``)/len(cols)."""
    raw = df[cols].to_numpy(dtype=np.float32, copy=False)
    miss = raw == _MISSING
    x = np.where(miss, np.float32(0.5), raw / np.float32(9.0)).astype(np.float32)
    out = pd.DataFrame(x, index=df.index, columns=list(cols))
    pos = {c: i for i, c in enumerate(cols)}
    for c in missing_cols:
        if c in pos:
            ind = miss[:, pos[c]]
        else:
            ind = df[c].to_numpy() == _MISSING
        out[f"miss_{c}"] = ind.astype(np.float32)
    out["n_missing"] = (miss.sum(axis=1) / np.float32(len(cols))).astype(np.float32)
    return out


def time_decay_weights(era: pd.Series, enabled: bool) -> np.ndarray:
    """Linear sample weights 0.5 (oldest era) .. 1.0 (newest era) by era value; all ones if disabled."""
    n = len(era)
    if not enabled or n == 0:
        return np.ones(n, dtype=np.float64)
    e = era.to_numpy(dtype=np.float64)
    lo, hi = e.min(), e.max()
    if hi == lo:
        return np.ones(n, dtype=np.float64)
    return 0.5 + 0.5 * (e - lo) / (hi - lo)


# --------------------------------------------------------------------------------------------------
# download (needs EIQ_API_KEY)
# --------------------------------------------------------------------------------------------------
def _read_dotenv(path: Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines of a .env file (values are never logged)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def resolve_api_key(dotenv_path: Path | None = None) -> str:
    """``EIQ_API_KEY`` (fallback ``EVEREST_API_KEY``) from the environment, else from ``.env``.

    Raises ``RuntimeError("EIQ_API_KEY not set; copy .env.example to .env")`` when absent.
    """
    key = os.environ.get("EIQ_API_KEY") or os.environ.get("EVEREST_API_KEY")
    if not key:
        env = _read_dotenv(dotenv_path if dotenv_path is not None else DOTENV_PATH)
        key = env.get("EIQ_API_KEY") or env.get("EVEREST_API_KEY") or ""
        if env.get("EIQ_BASE_URL") and not os.environ.get("EIQ_BASE_URL"):
            os.environ["EIQ_BASE_URL"] = env["EIQ_BASE_URL"]
    if not key:
        raise RuntimeError("EIQ_API_KEY not set; copy .env.example to .env")
    return key


def make_client(dotenv_path: Path | None = None):  # type: ignore[no-untyped-def]
    """Authenticated ``EverestAPI`` futures client (base URL from ``EIQ_BASE_URL``)."""
    key = resolve_api_key(dotenv_path)
    from everestapi import EverestAPI

    # HTTP request logs include signed download URLs. Keep credentials out of CLI
    # logs; our own download messages already identify the files and their sizes.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    base_url = os.environ.get("EIQ_BASE_URL", DEFAULT_BASE_URL)
    log.info("EverestAPI client -> %s (tournament=futures)", base_url)
    return EverestAPI(api_key=key, base_url=base_url, tournament="futures")


def download(
    data_dir: Path = DATA_DIR,
    splits: tuple[Split, ...] = SPLITS,
    benchmarks: bool = True,
) -> list[Path]:
    """Download the futures splits (+ benchmark files) into ``data_dir`` via the SDK. Needs EIQ_API_KEY."""
    client = make_client()
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for split in splits:
        out = split_path(split, data_dir)
        client.download_dataset(universe="futures", split=split, output_path=str(out))
        log.info("downloaded %s (%.1f MB)", out, out.stat().st_size / 1e6)
        written.append(out)
        if benchmarks:
            outb = benchmark_path(split, data_dir)
            client.download_benchmark(universe="futures", split=split, output_path=str(outb))
            log.info("downloaded %s (%.1f MB)", outb, outb.stat().st_size / 1e6)
            written.append(outb)
    return written


def main(argv: list[str] | None = None) -> int:
    """``python -m stealth.data download [--data-dir D] [--splits train validation live] [--no-benchmarks]``."""
    parser = argparse.ArgumentParser(prog="python -m stealth.data")
    sub = parser.add_subparsers(dest="cmd", required=True)
    dl = sub.add_parser("download", help="download futures splits + benchmark files via the SDK")
    dl.add_argument("--data-dir", type=Path, default=DATA_DIR)
    dl.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    dl.add_argument("--no-benchmarks", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "download":
        try:
            paths = download(args.data_dir, tuple(args.splits), benchmarks=not args.no_benchmarks)
        except RuntimeError as exc:
            log.error("%s", exc)
            return 1
        except Exception:  # noqa: BLE001 - CLI boundary
            log.exception("download failed")
            return 1
        log.info("wrote %d files", len(paths))
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
