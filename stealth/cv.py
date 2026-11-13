"""Purged/embargoed chronological folds, inner early-stopping split, forward block, paired block bootstrap.

Contracts: see docs/INTERFACES.md section 2.

Era arithmetic (all eras are ``int32`` era numbers, see ``data.era_int``):

* outer fold: test block = one of ``n_folds`` contiguous equal slices of the sorted unique eras; train =
  every era with ``era < test_min - purge - embargo`` or ``era > test_max + purge + embargo`` (both sides);
* inner split (for early stopping only): ``inner_val`` = the last ``inner_frac`` of the fold's train eras
  by time; ``inner_train`` = train eras ``< inner_val_min - inner_gap``. The scored test block never sees
  either of them;
* forward block: train ``< train_end``, test ``>= test_start`` (gap asserted ``>= 20``).

:func:`assert_fold_gaps` is the single place that checks the gaps; tests call it on every config.
:func:`independent_inner_folds` / :func:`assert_count_sources` decide which folds' inner early-stopping results may
set a given fold's frozen tree count (their inner_val eras must be >= 20 from that fold's test block).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    """One CV fold expressed as era sets (int32, sorted)."""

    name: str
    train_eras: np.ndarray
    test_eras: np.ndarray
    inner_train_eras: np.ndarray
    inner_val_eras: np.ndarray

    def summary(self) -> dict:
        """JSON-friendly summary used by ``folds.json``."""
        return {
            "name": self.name,
            "train_eras": [int(self.train_eras.min()), int(self.train_eras.max()), int(len(self.train_eras))],
            "test_eras": [int(self.test_eras.min()), int(self.test_eras.max())],
            "inner_train_eras": [int(self.inner_train_eras.min()), int(self.inner_train_eras.max()),
                                 int(len(self.inner_train_eras))],
            "inner_val_eras": [int(self.inner_val_eras.min()), int(self.inner_val_eras.max())],
        }


def _as_eras(eras: np.ndarray | pd.Series | list) -> np.ndarray:
    """Unique, sorted int32 era array; raises on empty input."""
    arr = np.unique(np.asarray(eras).astype(np.int64)).astype(np.int32)
    if arr.size == 0:
        raise ValueError("no eras given")
    return arr


def inner_split(
    train_eras: np.ndarray, *, inner_frac: float = 0.15, inner_gap: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Split a fold's train eras into (inner_train, inner_val) for early stopping.

    ``inner_val`` = last ``ceil(inner_frac * n)`` eras by time (at least 1); ``inner_train`` = train eras
    strictly below ``inner_val_min - inner_gap`` (distance to the validation block >= inner_gap + 1).
    """
    if not 0.0 < inner_frac < 1.0:
        raise ValueError(f"inner_frac must be in (0,1), got {inner_frac}")
    tr = _as_eras(train_eras)
    n_val = max(1, math.ceil(inner_frac * len(tr)))
    val = tr[-n_val:]
    inner_tr = tr[tr < int(val.min()) - inner_gap]
    if inner_tr.size == 0:
        raise ValueError("inner split leaves no inner-train eras; fewer folds or a smaller inner_gap needed")
    return inner_tr.astype(np.int32), val.astype(np.int32)


def purged_folds(
    eras: np.ndarray,
    n_folds: int,
    *,
    purge: int = 20,
    embargo: int = 20,
    inner_frac: float = 0.15,
    inner_gap: int = 20,
) -> list[Fold]:
    """Chronological exped-grouped K folds with purge + embargo on BOTH sides of each test block.

    ``eras`` = the eras in play (any order/duplicates; made unique + sorted). Test blocks are
    ``n_folds`` contiguous equal slices; train eras satisfy ``era < test_min - purge - embargo`` or
    ``era > test_max + purge + embargo``. Each fold also carries its inner early-stopping split.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if purge < 0 or embargo < 0:
        raise ValueError("purge/embargo must be >= 0")
    arr = _as_eras(eras)
    if len(arr) < n_folds * 2:
        raise ValueError(f"{len(arr)} eras is too few for {n_folds} folds")
    gap = purge + embargo
    folds: list[Fold] = []
    for k, test in enumerate(np.array_split(arr, n_folds)):
        lo, hi = int(test.min()), int(test.max())
        train = arr[(arr < lo - gap) | (arr > hi + gap)]
        if train.size == 0:
            raise ValueError(f"fold{k}: no training eras left after purge+embargo={gap}")
        inner_tr, inner_val = inner_split(train, inner_frac=inner_frac, inner_gap=inner_gap)
        folds.append(Fold(f"fold{k}", train.astype(np.int32), test.astype(np.int32), inner_tr, inner_val))
        log.debug("fold%d: train %d eras, test %d..%d (%d eras)", k, len(train), lo, hi, len(test))
    assert_fold_gaps(folds, min_gap=min(gap, inner_gap))
    return folds


def forward_block(
    eras: np.ndarray,
    *,
    train_end: int = 5925,
    test_start: int = 5945,
    **inner_kwargs,
) -> Fold:
    """Strict walk-forward block: train eras ``< train_end``, test eras ``>= test_start`` (gap >= 20)."""
    arr = _as_eras(eras)
    train = arr[arr < train_end]
    test = arr[arr >= test_start]
    if train.size == 0 or test.size == 0:
        raise ValueError(f"forward block empty: {len(train)} train / {len(test)} test eras")
    gap = int(test.min()) - int(train.max())
    if gap < 20:
        raise AssertionError(f"forward block gap {gap} < 20 (train_end={train_end}, test_start={test_start})")
    inner_tr, inner_val = inner_split(train, **inner_kwargs)
    fold = Fold("forward", train.astype(np.int32), test.astype(np.int32), inner_tr, inner_val)
    assert_fold_gaps([fold], min_gap=20)
    return fold


def extend_train(fold: Fold, extra_eras: np.ndarray, *, gap: int) -> Fold:
    """Add the eras of ``extra_eras`` lying MORE than ``gap`` eras before ``fold``'s test block to its training sets.

    Used for the training pool below the scoring universe (``run.min_era``, docs/INTERFACES.md section 12): the added
    eras join ``train_eras`` AND ``inner_train_eras``; ``test_eras`` and ``inner_val_eras`` are unchanged, so a training
    universe that masks the pool eras out reproduces the unextended fold exactly. Extra eras that would sit inside the
    inner split's gap (``>= inner_val_min - gap``) or anywhere after the test block are never added. Re-asserts
    :func:`assert_fold_gaps` with ``min_gap=min(gap, 20)``.
    """
    if gap < 0:
        raise ValueError("gap must be >= 0")
    extra = np.unique(np.asarray(extra_eras).astype(np.int64)).astype(np.int32)
    lo = int(fold.test_eras.min())
    limit = lo - int(gap)  # era < limit  <=>  test_min - era > gap
    limit = min(limit, int(fold.inner_val_eras.min()) - max(int(gap), 20))
    add = extra[extra < limit]
    add = add[~np.isin(add, fold.train_eras)]
    if add.size == 0:
        return fold
    train = np.union1d(fold.train_eras, add).astype(np.int32)
    inner_tr = np.union1d(fold.inner_train_eras, add).astype(np.int32)
    out = Fold(fold.name, train, fold.test_eras, inner_tr, fold.inner_val_eras)
    assert_fold_gaps([out], min_gap=min(int(gap), 20))
    return out


def _gap_to_block(train: np.ndarray, block: np.ndarray) -> int:
    """Smallest distance (in eras) from any ``train`` era to the contiguous span of ``block``."""
    lo, hi = int(block.min()), int(block.max())
    inside = (train >= lo) & (train <= hi)
    if inside.any():
        return 0
    below = train[train < lo]
    above = train[train > hi]
    gaps = []
    if below.size:
        gaps.append(lo - int(below.max()))
    if above.size:
        gaps.append(int(above.min()) - hi)
    return min(gaps) if gaps else 10**9


def assert_fold_gaps(folds: list[Fold], min_gap: int = 20) -> None:
    """Raise ``AssertionError`` unless every fold has train/test gap >= ``min_gap`` (and inner split too).

    Also checks: train and test disjoint, test block contiguous within the fold's era universe,
    inner_train/inner_val subsets of train, inner_val after inner_train.
    """
    for f in folds:
        if np.intersect1d(f.train_eras, f.test_eras).size:
            raise AssertionError(f"{f.name}: train and test eras overlap")
        g = _gap_to_block(f.train_eras, f.test_eras)
        if g < min_gap:
            raise AssertionError(f"{f.name}: train/test gap {g} < {min_gap}")
        if not (np.isin(f.inner_train_eras, f.train_eras).all() and np.isin(f.inner_val_eras, f.train_eras).all()):
            raise AssertionError(f"{f.name}: inner split eras are not a subset of train eras")
        if np.intersect1d(f.inner_train_eras, f.inner_val_eras).size:
            raise AssertionError(f"{f.name}: inner train/val eras overlap")
        if int(f.inner_train_eras.max()) >= int(f.inner_val_eras.min()):
            raise AssertionError(f"{f.name}: inner_val must come after inner_train")
        gi = _gap_to_block(f.inner_train_eras, f.inner_val_eras)
        if gi < min_gap:
            raise AssertionError(f"{f.name}: inner train/val gap {gi} < {min_gap}")
        if _gap_to_block(f.inner_val_eras, f.test_eras) < min_gap or _gap_to_block(f.inner_train_eras, f.test_eras) < min_gap:
            raise AssertionError(f"{f.name}: inner split touches the test block")


def independent_inner_folds(target: Fold, candidates: list[Fold], *, min_gap: int = 20) -> list[int]:
    """Indices of ``candidates`` whose INNER VALIDATION eras lie >= ``min_gap`` eras away from ``target.test_eras``.

    The tree count of a scored fold may only be chosen from early-stopping results whose validation eras do not
    touch that fold's test block (PLAN C8: "the scored fold never influences its own tree count"). With chronological
    folds the inner_val of every non-last fold sits at the END of the era range, i.e. inside the last fold's (and the
    forward block's) test eras -- so those folds' counts are excluded for the last fold / the forward block. A fold's
    own inner split is always eligible (inner_val is a subset of its train eras, which are purged from its test).
    """
    out = []
    for j, c in enumerate(candidates):
        if _gap_to_block(c.inner_val_eras, target.test_eras) >= min_gap:
            out.append(j)
    return out


def assert_count_sources(target: Fold, sources: list[Fold], *, min_gap: int = 20) -> None:
    """Raise ``AssertionError`` unless every source fold's inner_val eras are >= ``min_gap`` from ``target.test_eras``."""
    if not sources:
        raise AssertionError(f"{target.name}: no early-stopping source fold is independent of its test block")
    for s in sources:
        g = _gap_to_block(s.inner_val_eras, target.test_eras)
        if g < min_gap:
            raise AssertionError(f"{target.name}: count source {s.name} inner_val eras are {g} < {min_gap} from the test block")


# --------------------------------------------------------------------------------------------------
# block bootstrap
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PairedResult:
    """Paired block-bootstrap summary of a per-era difference series (A - B)."""

    diff_mean: float
    diff_se: float
    p_pos: float
    ci90: tuple[float, float]
    n_eras: int

    def as_dict(self) -> dict:
        return {"diff_mean": self.diff_mean, "diff_se": self.diff_se, "p_pos": self.p_pos,
                "ci90": [self.ci90[0], self.ci90[1]], "n_eras": self.n_eras}


def _clean_series(series: pd.Series) -> np.ndarray:
    """Values sorted by era index with NaNs dropped."""
    s = pd.Series(series).dropna()
    s = s.sort_index()
    return s.to_numpy(dtype=np.float64)


def block_bootstrap_means(values: np.ndarray, *, block: int = 20, n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    """Circular moving-block bootstrap: ``n_boot`` resampled means of the era series ``values``.

    Each replicate concatenates ``ceil(n/block)`` blocks of ``block`` consecutive eras (wrapping around),
    truncated to ``n`` eras. Deterministic for a given ``seed``.
    """
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n == 0:
        raise ValueError("empty series")
    if block < 1:
        raise ValueError("block must be >= 1")
    block = min(block, n)
    k = math.ceil(n / block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, k * block) % n
    idx = idx[:, :n]
    return v[idx].mean(axis=1)


def paired_block_bootstrap(
    diff: pd.Series, *, block: int = 20, n_boot: int = 2000, seed: int = 0
) -> PairedResult:
    """Paired test of a per-era metric difference ``A - B`` (indexed by era int).

    Returns the observed mean, the bootstrap SE of the mean, ``p_pos`` = fraction of bootstrap means > 0
    and the 5%-95% percentile interval. Gate convention (PLAN): adopt A iff ``p_pos >= 0.8`` on CV AND
    diff >= 0 on the forward block.
    """
    v = _clean_series(diff)
    means = block_bootstrap_means(v, block=block, n_boot=n_boot, seed=seed)
    lo, hi = np.percentile(means, [5.0, 95.0])
    return PairedResult(
        diff_mean=float(v.mean()),
        diff_se=float(means.std(ddof=1)) if n_boot > 1 else float("nan"),
        p_pos=float((means > 0).mean()),
        ci90=(float(lo), float(hi)),
        n_eras=int(len(v)),
    )


def block_bootstrap_ci(
    series: pd.Series, *, block: int = 20, n_boot: int = 2000, seed: int = 0, level: float = 0.90
) -> tuple[float, float]:
    """Unpaired percentile block-bootstrap CI of the mean of a per-era series (context only, not a gate)."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0,1)")
    v = _clean_series(series)
    means = block_bootstrap_means(v, block=block, n_boot=n_boot, seed=seed)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)
