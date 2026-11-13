"""Tests for stealth.cv: fold gaps >= 20 (outer + inner) on every config, forward block bounds, bootstraps."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from stealth import cv

ALL_TRAIN_ERAS = np.arange(3, 6525, dtype=np.int32)  # exped_0003..exped_6524


def _folds_for_config(cfg: dict) -> list[cv.Fold]:
    run, c = cfg["run"], cfg["cv"]
    eras = ALL_TRAIN_ERAS
    if run.get("min_era") is not None:
        eras = eras[eras >= run["min_era"]]
    if run.get("max_era") is not None:
        eras = eras[eras <= run["max_era"]]
    folds = cv.purged_folds(eras, c["n_folds"], purge=c["purge"], embargo=c["embargo"],
                            inner_frac=c["inner_frac"], inner_gap=c["inner_gap"])
    if c["forward"]["enabled"]:
        folds.append(cv.forward_block(eras, train_end=c["forward"]["train_end"],
                                      test_start=c["forward"]["test_start"],
                                      inner_frac=c["inner_frac"], inner_gap=c["inner_gap"]))
    return folds


@pytest.mark.parametrize("name", ["smoke", "test", "full", "minifull"])
def test_every_config_fold_gap_ge_20(repo_root: Path, name: str) -> None:
    cfg = yaml.safe_load((repo_root / "configs" / f"{name}.yaml").read_text())
    folds = _folds_for_config(cfg)
    cv.assert_fold_gaps(folds, min_gap=20)
    assert len(folds) == cfg["cv"]["n_folds"] + int(cfg["cv"]["forward"]["enabled"])
    for f in folds:
        # explicit gap arithmetic, independent of assert_fold_gaps
        lo, hi = int(f.test_eras.min()), int(f.test_eras.max())
        below = f.train_eras[f.train_eras < lo]
        above = f.train_eras[f.train_eras > hi]
        assert not ((f.train_eras >= lo) & (f.train_eras <= hi)).any()
        if below.size:
            assert lo - int(below.max()) >= 20
        if above.size:
            assert int(above.min()) - hi >= 20
        # inner split: val after train with a >= 20 gap, both inside train eras, away from the test block
        assert int(f.inner_val_eras.min()) - int(f.inner_train_eras.max()) >= 20
        assert set(f.inner_val_eras) <= set(f.train_eras) and set(f.inner_train_eras) <= set(f.train_eras)
        assert not (set(f.inner_val_eras) & set(f.test_eras))


def test_purged_folds_structure() -> None:
    eras = ALL_TRAIN_ERAS[ALL_TRAIN_ERAS >= 2800]
    folds = cv.purged_folds(eras, 5)
    assert [f.name for f in folds] == [f"fold{k}" for k in range(5)]
    # test blocks partition the eras, contiguous and chronological
    cat = np.concatenate([f.test_eras for f in folds])
    assert np.array_equal(cat, eras)
    for f in folds:
        assert f.test_eras.dtype == np.int32 and f.train_eras.dtype == np.int32
        assert np.array_equal(f.test_eras, np.arange(f.test_eras.min(), f.test_eras.max() + 1))
        assert np.all(np.diff(f.train_eras) > 0)
        assert abs(len(f.test_eras) - len(eras) / 5) <= 1
    # middle fold trains on BOTH sides of the test block
    mid = folds[2]
    assert (mid.train_eras < mid.test_eras.min()).any() and (mid.train_eras > mid.test_eras.max()).any()
    # exact purge+embargo arithmetic from INTERFACES: era < test_min - 40 or era > test_max + 40
    assert int(mid.train_eras[mid.train_eras < mid.test_eras.min()].max()) == int(mid.test_eras.min()) - 41
    assert int(mid.train_eras[mid.train_eras > mid.test_eras.max()].min()) == int(mid.test_eras.max()) + 41
    # inner val is ~15% of train eras, at the end by time
    n_val = len(mid.inner_val_eras)
    assert n_val == int(np.ceil(0.15 * len(mid.train_eras)))
    assert int(mid.inner_val_eras.max()) == int(mid.train_eras.max())
    assert int(mid.inner_train_eras.max()) < int(mid.inner_val_eras.min()) - 20
    cv.assert_fold_gaps(folds)


def test_purged_folds_accepts_unsorted_duplicates() -> None:
    eras = np.array([5003, 5001, 5002, 5001] + list(range(5004, 5600)))
    folds = cv.purged_folds(eras, 3)
    assert folds[0].test_eras.min() == 5001


def test_purged_folds_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        cv.purged_folds(ALL_TRAIN_ERAS, 1)
    with pytest.raises(ValueError):
        cv.purged_folds(np.array([], dtype=np.int32), 3)
    with pytest.raises(ValueError):
        cv.purged_folds(np.arange(10), 3, purge=100, embargo=100)


def test_forward_block_bounds() -> None:
    eras = ALL_TRAIN_ERAS[ALL_TRAIN_ERAS >= 2800]
    f = cv.forward_block(eras)
    assert f.name == "forward"
    assert int(f.train_eras.min()) == 2800 and int(f.train_eras.max()) == 5924
    assert int(f.test_eras.min()) == 5945 and int(f.test_eras.max()) == 6524
    assert int(f.test_eras.min()) - int(f.train_eras.max()) >= 20
    assert int(f.inner_val_eras.min()) - int(f.inner_train_eras.max()) >= 20
    assert int(f.inner_val_eras.max()) <= 5924
    with pytest.raises(AssertionError):
        cv.forward_block(eras, train_end=5940, test_start=5945)
    with pytest.raises(ValueError):
        cv.forward_block(eras, train_end=100, test_start=5945)


def test_assert_fold_gaps_catches_violations() -> None:
    good = cv.purged_folds(ALL_TRAIN_ERAS[ALL_TRAIN_ERAS >= 5000], 3)[0]
    bad_train = np.append(good.train_eras, good.test_eras.min() - 5).astype(np.int32)
    bad = cv.Fold("bad", np.sort(bad_train), good.test_eras, good.inner_train_eras, good.inner_val_eras)
    with pytest.raises(AssertionError, match="gap"):
        cv.assert_fold_gaps([bad])
    leak = cv.Fold("leak", good.train_eras, good.test_eras, good.inner_train_eras,
                   np.append(good.inner_val_eras, good.test_eras[0]).astype(np.int32))
    with pytest.raises(AssertionError):
        cv.assert_fold_gaps([leak])
    overlap = cv.Fold("ov", np.union1d(good.train_eras, good.test_eras[:1]), good.test_eras,
                      good.inner_train_eras, good.inner_val_eras)
    with pytest.raises(AssertionError, match="overlap"):
        cv.assert_fold_gaps([overlap])


def test_fold_summary_json_friendly() -> None:
    f = cv.purged_folds(ALL_TRAIN_ERAS[ALL_TRAIN_ERAS >= 5000], 3)[1]
    s = f.summary()
    assert s["name"] == "fold1" and len(s["train_eras"]) == 3 and len(s["test_eras"]) == 2
    assert all(isinstance(v, int) for v in s["train_eras"] + s["test_eras"] + s["inner_val_eras"])


# ---------------------------------------------------------------------------------------------- bootstrap
def _era_series(values: np.ndarray, start: int = 3000) -> pd.Series:
    return pd.Series(values, index=pd.Index(np.arange(start, start + len(values)), name="era"))


def test_paired_block_bootstrap_clear_winner() -> None:
    rng = np.random.default_rng(1)
    diff = _era_series(0.004 + 0.01 * rng.standard_normal(1000))
    r = cv.paired_block_bootstrap(diff, n_boot=2000, seed=0)
    assert r.n_eras == 1000
    assert abs(r.diff_mean - diff.mean()) < 1e-12
    assert r.p_pos > 0.99 and r.ci90[0] > 0 and r.ci90[0] < r.diff_mean < r.ci90[1]
    assert 0.0002 < r.diff_se < 0.0006  # ~ sd/sqrt(n) = 0.01/31.6 = 0.00032, inflated by blocks
    d = r.as_dict()
    assert set(d) == {"diff_mean", "diff_se", "p_pos", "ci90", "n_eras"}


def test_paired_block_bootstrap_tie_and_determinism() -> None:
    rng = np.random.default_rng(2)
    diff = _era_series(0.01 * rng.standard_normal(600))
    diff = diff - diff.mean()  # exactly zero mean -> p_pos ~ 0.5
    r1 = cv.paired_block_bootstrap(diff, n_boot=2000, seed=7)
    r2 = cv.paired_block_bootstrap(diff, n_boot=2000, seed=7)
    assert r1 == r2
    assert 0.4 < r1.p_pos < 0.6
    r3 = cv.paired_block_bootstrap(diff, n_boot=2000, seed=8)
    assert r3 != r1


def test_block_bootstrap_respects_autocorrelation() -> None:
    """Block SE on an AR(1)-like series must exceed the iid SE."""
    rng = np.random.default_rng(3)
    e = rng.standard_normal(2000)
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, len(e)):
        x[i] = 0.9 * x[i - 1] + e[i]
    s = _era_series(x)
    se_block = cv.paired_block_bootstrap(s, block=20, n_boot=1000, seed=0).diff_se
    se_iid = cv.paired_block_bootstrap(s, block=1, n_boot=1000, seed=0).diff_se
    assert se_block > 2 * se_iid


def test_block_bootstrap_ci_and_nan_handling() -> None:
    rng = np.random.default_rng(4)
    s = _era_series(0.02 + 0.03 * rng.standard_normal(500))
    s.iloc[[5, 50]] = np.nan
    lo, hi = cv.block_bootstrap_ci(s, n_boot=1000, seed=0, level=0.90)
    assert lo < s.mean() < hi
    lo2, hi2 = cv.block_bootstrap_ci(s, n_boot=1000, seed=0, level=0.50)
    assert lo < lo2 < hi2 < hi
    with pytest.raises(ValueError):
        cv.block_bootstrap_ci(s, level=1.5)
    with pytest.raises(ValueError):
        cv.paired_block_bootstrap(pd.Series([], dtype=float))


def test_block_bootstrap_means_shape_and_short_series() -> None:
    m = cv.block_bootstrap_means(np.arange(7, dtype=float), block=20, n_boot=50, seed=0)
    assert m.shape == (50,) and np.all((m >= 0) & (m <= 6))


# ---------------------------------------------------------------------------------------------- count sources
def test_independent_inner_folds_test_layout() -> None:
    """test.yaml layout: the last fold and the forward block may only take their tree count from independent sources."""
    eras = ALL_TRAIN_ERAS[ALL_TRAIN_ERAS >= 2800]
    folds = cv.purged_folds(eras, 5)
    fw = cv.forward_block(eras)
    # folds 0..3 have their inner_val at the end of the era range -> inside fold4's AND the forward's test block
    for k in range(4):
        assert int(folds[k].inner_val_eras.min()) >= int(folds[4].test_eras.min())
        assert int(folds[k].inner_val_eras.min()) >= int(fw.test_eras.min())
    assert cv.independent_inner_folds(folds[4], folds) == [4]
    assert cv.independent_inner_folds(fw, folds) == [4]
    # an early fold may use every fold's inner split (all inner_val blocks are >= 20 eras after its test block)
    assert cv.independent_inner_folds(folds[0], folds) == [0, 1, 2, 3, 4]
    # every admitted source is >= 20 eras from the target's test block; a contaminated source is rejected
    for k, f in enumerate(folds):
        src = [folds[j] for j in cv.independent_inner_folds(f, folds)]
        assert k in cv.independent_inner_folds(f, folds)  # own split always eligible
        cv.assert_count_sources(f, src)
        for s in src:
            assert cv._gap_to_block(s.inner_val_eras, f.test_eras) >= 20
    cv.assert_count_sources(fw, [folds[4], fw])
    with pytest.raises(AssertionError, match="count source"):
        cv.assert_count_sources(folds[4], [folds[0]])
    with pytest.raises(AssertionError, match="no early-stopping source"):
        cv.assert_count_sources(folds[4], [])


def test_independent_inner_folds_smoke_layout() -> None:
    eras = ALL_TRAIN_ERAS[ALL_TRAIN_ERAS >= 5000]
    folds = cv.purged_folds(eras, 3)
    assert cv.independent_inner_folds(folds[2], folds) == [2]
    assert cv.independent_inner_folds(folds[0], folds) == [0, 1, 2]


# ------------------------------------------------------------------------------------- extend_train (M2 pool)
def test_extend_train_adds_pool_eras_with_gap_and_keeps_inner_val() -> None:
    scoring = ALL_TRAIN_ERAS[ALL_TRAIN_ERAS >= 5000]
    pool_extra = ALL_TRAIN_ERAS[(ALL_TRAIN_ERAS >= 4600) & (ALL_TRAIN_ERAS < 5000)]
    folds = cv.purged_folds(scoring, 3)
    fw = cv.forward_block(scoring)
    for f in folds + [fw]:
        g = cv.extend_train(f, pool_extra, gap=40)
        # test and inner_val untouched -> a universe masking the pool out reproduces the M1 fold exactly
        assert np.array_equal(g.test_eras, f.test_eras) and np.array_equal(g.inner_val_eras, f.inner_val_eras)
        added = np.setdiff1d(g.train_eras, f.train_eras)
        assert np.array_equal(added, np.setdiff1d(g.inner_train_eras, f.inner_train_eras))
        # every added era lies MORE than 40 eras before the test block (purge + embargo)
        assert added.size and int(added.max()) < int(f.test_eras.min()) - 40
        assert np.isin(f.train_eras, g.train_eras).all() and np.isin(f.inner_train_eras, g.inner_train_eras).all()
        assert g.train_eras.dtype == np.int32 and np.all(np.diff(g.train_eras) > 0)
        cv.assert_fold_gaps([g], min_gap=20)
    # fold0's test block starts at 5000: the 40 eras just below are purged, the rest join train AND inner_train
    g0 = cv.extend_train(folds[0], pool_extra, gap=40)
    assert int(g0.train_eras.min()) == 4600 and not ((g0.train_eras >= 4960) & (g0.train_eras < 5000)).any()
    assert int(np.setdiff1d(g0.train_eras, folds[0].train_eras).max()) == 4959


def test_extend_train_noop_and_validation() -> None:
    eras = ALL_TRAIN_ERAS[ALL_TRAIN_ERAS >= 5000]
    f = cv.purged_folds(eras, 3)[0]
    assert cv.extend_train(f, np.array([4980, 4990], dtype=np.int32), gap=40) is f  # all inside the purge gap
    assert cv.extend_train(f, np.array([], dtype=np.int32), gap=40) is f
    with pytest.raises(ValueError):
        cv.extend_train(f, eras, gap=-1)
    # eras after the test block are never added (the pool only extends BELOW the scoring universe)
    g = cv.extend_train(f, np.array([4000, 9000], dtype=np.int32), gap=40)
    assert 4000 in g.train_eras and 9000 not in g.train_eras
