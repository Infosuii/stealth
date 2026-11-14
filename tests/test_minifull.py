"""configs/minifull.yaml (docs/INTERFACES.md section 20): the sandbox tier that exercises every M2-M6 path.

The schema checks need no data; the ``slow`` test builds the run's folds on the real train split (skipped without
data). The end-to-end run itself is ``scripts/run_pipeline.sh configs/minifull.yaml <run_id>`` (<= 8 min on 2 cores).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stealth import cv, knobs
from stealth import train as trainmod


@pytest.fixture(scope="module")
def mini(repo_root: Path) -> dict:
    return trainmod.load_config(repo_root / "configs" / "minifull.yaml")


def test_minifull_loads_and_matches_section_20(mini: dict) -> None:
    run, c, kn = mini["run"], mini["cv"], mini["knobs"]
    assert run["min_era"] == 5000 and run["threads"] == 2 and knobs.pool_min_era(mini) == 4600
    assert c["n_folds"] == 3 and c["forward"] == {"enabled": True, "train_end": 5600, "test_start": 5640}
    assert c["audit"] == {"enabled": True, "train_end": 5925, "test_start": 5945}  # same reserved partition as full.yaml
    assert kn["era_start"] == [5000, 4600] and kn["rows_rule"] == ["none", "filter80", "weight80"]
    assert kn["time_decay"] == [False, True]
    assert len(knobs.candidates(mini)) == 5  # default + 4 one-at-a-time alternatives
    by = {m["name"]: m for m in mini["members"]}
    s = mini["search"]
    assert s["lgbm_everest"]["n_trials"] == 3 and s["lgbm_everest"]["top_k"] == 2 and s["lgbm_everest"]["adopt_gate"] is True
    grid = s["lgbm_everest"]["grid"]
    assert int(np.prod([len(v) for v in grid.values()])) == 4  # TPE: 3 trials over 4 cells
    assert s["xgb_everest"]["n_trials"] == 2 == int(np.prod([len(v) for v in s["xgb_everest"]["grid"].values()]))
    aux = [m for m in mini["members"] if m["group"] == "aux"]
    assert len(aux) == 2 and all(m["family"] == "lgbm" and m["params"].get("best_of") == "lgbm_everest" for m in aux)
    cats = [m for m in mini["members"] if m["family"] == "cat"]
    assert len(cats) == 1 and cats[0]["n_estimators"] == 50
    assert by["ridge_everest"]["family"] == "ridge" and by["lgbm_everest_sharpe"]["features"] == "sharpe_subset"
    assert sorted(m["features"] for m in mini["members"] if trainmod.arm_of(m) == "bags") == ["bag:0", "bag:1"]
    mlp = by["mlp_everest"]
    assert mlp["family"] == "mlp" and mlp["seeds"] == [0] and mlp["es_cap"] == 3 and mlp["params"]["hidden"] == [32, 16]
    for m in mini["members"]:
        if m["family"] in ("mlp", "ridge"):
            continue
        assert (m["n_estimators"] is not None and m["n_estimators"] <= 80) or (m["es_cap"] is not None and m["es_cap"] <= 120), m["name"]
    b = mini["blend"]
    assert b["admission"]["order"] == ["aux", "xgb", "cat", "ridge", "sharpe", "bags", "mlp"] and b["admission"]["base"] == ["lgbm"]
    assert b["simulation"]["n_draws"] == 30 and b["simulation"]["p_c_grid"] == [0.0, 0.25, 0.5]
    assert b["neutralisation"]["select"] is True and mini["refit"]["seeds"] == [0]
    arms = {trainmod.arm_of(m) for m in mini["members"]}
    assert set(b["admission"]["order"]) | set(b["admission"]["base"]) == arms  # every pre-registered arm is present


@pytest.mark.slow
def test_minifull_folds_extend_into_the_pool(mini: dict, data_dir: Path) -> None:
    rd = trainmod.RunData(mini, data_dir)
    assert rd.scoring_min_era == 5000 and int(rd.pool_eras.min()) == 4600
    assert rd.universe == knobs.default_universe(mini)
    cv.assert_fold_gaps(rd.folds + [rd.forward], min_gap=20)
    for f in rd.folds + [rd.forward]:
        pool = f.train_eras[f.train_eras < 5000]
        assert pool.size and int(pool.max()) < int(f.test_eras.min()) - 40 and int(f.test_eras.min()) >= 5000
        assert np.isin(pool, f.inner_train_eras).all() and int(f.inner_val_eras.min()) >= 5000
    # the default universe masks the pool out of every training fit (M1-identical rows); the 4600 candidate keeps it
    era = rd.era.to_numpy()
    assert not rd.train_mask()[era < 5000].any() and rd.train_mask()[era >= 5000].all()
    alt = knobs.Universe(4600)
    assert alt.train_mask(era, rd.era_n_rows).all()
    assert knobs.equivalent(knobs.Universe(5000, "filter80"), rd.universe, era, rd.era_n_rows)
    # the reserved audit partition: development and full refits stop before 5925, the audit eras are held apart
    rd.assert_development_only()
    assert int(rd.full_train_eras.max()) < 5925 and int(rd.audit_eras.min()) == 5945 and int(rd.audit_eras.max()) == 6524
    assert int(rd.forward.test_eras.min()) == 5640 and int(rd.forward.test_eras.max()) == 5924


def test_full_config_audit_layout(repo_root: Path, data_dir: Path) -> None:
    """full.yaml (handoff 04): development CV 2800..5924, dev forward 5340..5924, reserved audit 5945..6524, measurement
    budget 5000 bootstrap / 1000 simulation draws, <= 25 trials, the historical baseline_fit is context only (null)."""
    full = trainmod.load_config(repo_root / "configs" / "full.yaml")
    assert full["cv"]["audit"] == {"enabled": True, "train_end": 5925, "test_start": 5945}
    assert full["cv"]["forward"] == {"enabled": True, "train_end": 5300, "test_start": 5340}
    assert full["bootstrap"]["n_boot"] == 5000 and full["blend"]["simulation"]["n_draws"] == 1000
    assert full["gates"]["baseline_fit"] is None and full["search"]["lgbm_everest"]["n_trials"] <= 25
    assert full["refit"]["seeds"] == [0, 1] and full["run"]["threads"] == 16
    eras = np.arange(2800, 5925, dtype=np.int32)
    folds = cv.purged_folds(eras, 5)
    fw = cv.forward_block(eras, train_end=5300, test_start=5340)
    cv.assert_fold_gaps(folds + [fw], min_gap=20)
    assert max(int(f.test_eras.max()) for f in folds) == 5924 and int(fw.test_eras.min()) == 5340
