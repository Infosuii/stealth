"""Tests for stealth.knobs: universe masks/weights, candidates, Holm ordering, re-decision refusal, cache keys.

Synthetic arrays plus the ``train_slice`` fixture (last 300 train eras). The end-to-end decision inside a training
run is covered by tests/test_train_m2.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stealth import data, knobs, train
from stealth.knobs import Universe


def _cfg(**kn) -> dict:
    return {"run": {"min_era": 2800, "time_decay": False}, "gates": {"p_pos_adopt": 0.8},
            "members": [{"name": "lgbm_everest", "family": "lgbm", "group": "everest"}],
            "knobs": {"era_start": [2800, 2000, 3], "rows_rule": ["none", "filter80", "weight80"], "time_decay": [False, True],
                      **kn}}


# ------------------------------------------------------------------------------------------ Universe
def test_universe_key_roundtrip_and_validation() -> None:
    u = Universe(2800)
    assert u.key() == "e2800-none-d0"
    assert Universe(2000, "filter80", True).key() == "e2000-filter80-d1"
    assert Universe(2800, "weight80", False, 0.5).key() == "e2800-weight80w0.5-d0"
    assert Universe(2800, "weight80", False, 0.25).key() != Universe(2800, "weight80", False, 0.5).key()
    for v in (u, Universe(3, "weight80", True, 0.25)):
        assert Universe.from_dict(json.loads(json.dumps(v.to_dict()))) == v
    with pytest.raises(ValueError):
        Universe(2800, "rows80")
    with pytest.raises(ValueError):
        Universe(2800, "weight80", rows_low_weight=0.0)
    assert knobs.parse_rows_rule("filter80") == ("filter", 80) and knobs.parse_rows_rule("none") == ("none", None)


def test_train_mask_and_weights_synthetic() -> None:
    era = np.array([10, 10, 20, 20, 30, 30], dtype=np.int32)
    n_rows = np.array([50, 50, 90, 90, 85, 85])
    assert Universe(3).train_mask(era, n_rows).all()
    assert Universe(3).weights(era, n_rows) is None  # default universe: unweighted fit (M1 behaviour)
    np.testing.assert_array_equal(Universe(20).train_mask(era, n_rows), [False, False, True, True, True, True])
    np.testing.assert_array_equal(Universe(3, "filter80").train_mask(era, n_rows), [False, False, True, True, True, True])
    np.testing.assert_array_equal(Universe(3, "filter86").train_mask(era, n_rows), [False, False, True, True, False, False])
    # the weight rule never drops rows; low-row eras get rows_low_weight
    assert Universe(3, "weight80").train_mask(era, n_rows).all()
    np.testing.assert_allclose(Universe(3, "weight80", rows_low_weight=0.3).weights(era, n_rows), [0.3, 0.3, 1, 1, 1, 1])
    # time decay over the eras of THIS fit: 0.5 (oldest) .. 1.0 (newest), identical to data.time_decay_weights
    td = Universe(3, time_decay=True).weights(era, n_rows)
    np.testing.assert_allclose(td, data.time_decay_weights(pd.Series(era), True))
    np.testing.assert_allclose(td, [0.5, 0.5, 0.75, 0.75, 1.0, 1.0])
    both = Universe(3, "weight80", True, 0.5).weights(era, n_rows)
    np.testing.assert_allclose(both, [0.25, 0.25, 0.75, 0.75, 1.0, 1.0])


def test_universe_on_train_slice(train_slice: pd.DataFrame) -> None:
    era = train_slice["era"].to_numpy()
    n_rows = data.era_counts(train_slice).reindex(era).to_numpy()
    assert (n_rows >= 80).all()  # late eras: ~90 instruments each
    d = Universe(6225)
    assert d.train_mask(era, n_rows).all()
    m = Universe(6400).train_mask(era, n_rows)
    assert set(np.unique(era[m])) == set(range(6400, 6525))
    assert knobs.equivalent(Universe(6225, "filter80"), d, era, n_rows)
    assert knobs.equivalent(Universe(6225, "weight80"), d, era, n_rows)
    above_min = int(n_rows.min()) + 1  # a filter that drops the smallest eras of the slice
    assert not knobs.equivalent(Universe(6225, f"filter{above_min}"), d, era, n_rows)
    assert not knobs.equivalent(Universe(6225, f"weight{above_min}"), d, era, n_rows)
    assert not Universe(6225, f"filter{above_min}").train_mask(era, n_rows).all()
    assert not knobs.equivalent(Universe(6225, time_decay=True), d, era, n_rows)
    assert not knobs.equivalent(Universe(6400), d, era, n_rows)


# ------------------------------------------------------------------------------------------ candidates
def test_default_candidates_one_knob_at_a_time() -> None:
    cfg = _cfg()
    d = knobs.default_universe(cfg)
    assert d == Universe(2800, "none", False)
    keys = [u.key() for u in knobs.candidates(cfg)]
    assert keys == ["e2800-none-d0", "e2000-none-d0", "e3-none-d0", "e2800-filter80-d0", "e2800-weight80w0.5-d0", "e2800-none-d1"]
    dims = [knobs.dimension_of(u, d) for u in knobs.candidates(cfg)]
    assert dims == [None, "era_start", "era_start", "rows_rule", "rows_rule", "time_decay"]
    with pytest.raises(ValueError):
        knobs.dimension_of(Universe(2000, "filter80"), d)  # two knobs at once is never a candidate
    assert knobs.pool_min_era(cfg) == 3
    assert knobs.candidates({"run": {"min_era": 5000, "time_decay": False}}) == [Universe(5000)]
    assert knobs.knob_member(cfg) == "lgbm_everest" and knobs.knob_member(_cfg(member="x")) == "x"


def test_full_and_minifull_candidates(repo_root: Path) -> None:
    full = train.load_config(repo_root / "configs" / "full.yaml")
    assert len(knobs.candidates(full)) == 6  # default + 5 alternatives
    mini = train.load_config(repo_root / "configs" / "minifull.yaml")
    assert [u.key() for u in knobs.candidates(mini)][:2] == ["e5000-none-d0", "e4600-none-d0"]
    assert knobs.pool_min_era(mini) == 4600


# ------------------------------------------------------------------------------------------ Holm + choice
def _row(key: str, dim: str | None, p: float, diff: float, cv: float | None = 0.001, **kw) -> dict:
    return {"key": key, "dimension": dim, "is_default": dim is None, "p_pos_fwd": p, "diff_fwd": diff, "diff_cv_indep": cv, **kw}


def test_holm_step_down_ordering_and_stop() -> None:
    rows = [_row("D", None, 0.0, 0.0),
            _row("a", "era_start", 0.97, 0.004, era_start=2000),     # 1-p = .03 -> sorted 2nd (thr .2/4 = .05) pass
            _row("b", "era_start", 0.99, 0.003, era_start=3),        # 1-p = .01 -> 1st (thr .2/5 = .04) pass
            _row("c", "rows_rule", 0.90, 0.002, rows_rule="filter80"),  # 1-p = .10 -> 3rd (thr .2/3 = .067) FAIL, stop
            _row("d", "rows_rule", 0.95, 0.001, cv=-0.001, rows_rule="weight80"),  # .05 -> 3rd? no: .05 < .10 -> 3rd
            _row("e", "time_decay", 0.95, 0.002, cv=None, time_decay=True)]
    knobs.holm_decide(rows, 0.2)
    by = {r["key"]: r for r in rows}
    # order by 1-p (ties in config order): b(.01) a(.03) d(.05) e(.05) c(.10)
    assert [by[k]["holm_rank"] for k in ("b", "a", "d", "e", "c")] == [1, 2, 3, 4, 5]
    assert by["b"]["holm_threshold"] == pytest.approx(0.2 / 5) and by["c"]["holm_threshold"] == pytest.approx(0.2 / 1)
    assert by["b"]["holm_pass"] and by["a"]["holm_pass"] and by["d"]["holm_pass"]  # .05 <= .2/3
    assert by["e"]["holm_pass"]  # .05 <= .2/2
    assert by["c"]["holm_pass"]  # .10 <= .2/1
    assert by["b"]["pass"] and by["a"]["pass"]
    assert not by["d"]["pass"] and "non-overlapping CV folds disagree" in by["d"]["reason"]
    assert not by["e"]["pass"]  # no independent confirmation available
    assert by["D"]["pass"] is None and by["D"]["holm_rank"] is None
    # step-down stops at the first failure: once rank j fails every later rank fails even if it is below its threshold
    rows2 = [_row("D", None, 0, 0), _row("x", "era_start", 0.85, 0.01), _row("y", "rows_rule", 0.84, 0.01)]
    knobs.holm_decide(rows2, 0.2)  # x: .15 > .2/2 fails at rank 1; y: .16 <= .2/1 but the step-down has stopped
    assert rows2[1]["holm_rank"] == 1 and not rows2[1]["holm_pass"]
    assert rows2[2]["holm_rank"] == 2 and not rows2[2]["holm_pass"] and "stopped" in rows2[2]["reason"]
    # a negative forward diff never passes
    rows3 = [_row("D", None, 0, 0), _row("z", "time_decay", 0.999, -0.0001, time_decay=True)]
    knobs.holm_decide(rows3, 0.2)
    assert rows3[1]["holm_pass"] and not rows3[1]["pass"] and rows3[1]["reason"] == "forward diff < 0"


def test_holm_uses_null_centred_p_value_and_fails_closed() -> None:
    """Review finding 6: Holm runs on p_value_fwd (null-centred), not 1 - p_pos; too few blocks never pass."""
    rows = [_row("D", None, 0.0, 0.0),
            # high p_pos but a weak null-centred p-value: must NOT pass at rank 1 (thr .2/3)
            _row("a", "era_start", 0.99, 0.004, p_value_fwd=0.09, inference_valid_fwd=True, era_start=2000),
            # modest p_pos but a strong p-value: sorted first, passes
            _row("b", "rows_rule", 0.90, 0.003, p_value_fwd=0.01, inference_valid_fwd=True, rows_rule="filter80"),
            # one bootstrap block: no inference even with p_pos 1 and p_value 0.0005
            _row("c", "time_decay", 1.0, 0.01, p_value_fwd=0.0005, inference_valid_fwd=False, n_blocks_fwd=1, time_decay=True)]
    knobs.holm_decide(rows, 0.2)
    by = {r["key"]: r for r in rows}
    assert by["b"]["holm_rank"] == 1 and by["b"]["holm_p"] == pytest.approx(0.01) and by["b"]["pass"]
    assert by["a"]["holm_rank"] == 2 and by["a"]["holm_p"] == pytest.approx(0.09)
    assert by["a"]["holm_pass"]  # .09 <= .2/2 = .1 at rank 2
    assert by["c"]["holm_rank"] == 3 and by["c"]["holm_p"] == 1.0 and not by["c"]["holm_pass"] and not by["c"]["pass"]
    assert "no inference" in by["c"]["reason"]
    rows2 = [_row("D", None, 0.0, 0.0), _row("a", "era_start", 0.99, 0.004, p_value_fwd=0.15, inference_valid_fwd=True)]
    knobs.holm_decide(rows2, 0.2)  # m = 1: threshold .2 -> passes on p .15 even though 1-p_pos would be .01 either way
    assert rows2[1]["holm_pass"]
    rows3 = [_row("D", None, 0.0, 0.0), _row("a", "era_start", 0.99, 0.004, p_value_fwd=0.25, inference_valid_fwd=True)]
    knobs.holm_decide(rows3, 0.2)  # 1 - p_pos = .01 would pass; the calibrated p-value .25 does not
    assert not rows3[1]["holm_pass"]


def test_choose_universe_one_winner_per_dimension() -> None:
    d = Universe(2800)
    rows = [{"key": "a", "dimension": "era_start", "era_start": 2000, "diff_fwd": 0.002, "pass": True},
            {"key": "b", "dimension": "era_start", "era_start": 3, "diff_fwd": 0.004, "pass": True},
            {"key": "c", "dimension": "rows_rule", "rows_rule": "weight80", "diff_fwd": 0.009, "pass": False},
            {"key": "t", "dimension": "time_decay", "time_decay": True, "diff_fwd": 0.001, "pass": True}]
    chosen, winners = knobs.choose_universe(d, rows)
    assert chosen == Universe(3, "none", True) and winners == {"era_start": "b", "time_decay": "t"}
    assert knobs.choose_universe(d, [dict(r, **{"pass": False}) for r in rows]) == (d, {})


# ------------------------------------------------------------------------------------------ once only
def test_redecision_refused(tmp_path: Path) -> None:
    cfg = _cfg()
    cands = knobs.candidates(cfg)
    sig = knobs.decision_signature("lgbm_everest", 0.2, cands)
    path = tmp_path / "knobs.json"
    assert knobs.check_previous(path, sig) is None
    path.write_text(json.dumps({"status": "running"}))
    assert knobs.check_previous(path, sig) is None  # unfinished decision: decided afresh
    doc = {"status": "done", "member": "lgbm_everest", "alpha_total": 0.2, "candidates": [{"key": u.key()} for u in cands],
           "chosen": Universe(2800).to_dict()}
    path.write_text(json.dumps(doc))
    assert knobs.check_previous(path, sig)["status"] == "done"
    for other in (knobs.decision_signature("lgbm_everest", 0.2, cands[:-1]),
                  knobs.decision_signature("xgb_everest", 0.2, cands),
                  knobs.decision_signature("lgbm_everest", 0.1, cands)):
        with pytest.raises(ValueError, match="knobs already decided for this run-id"):
            knobs.check_previous(path, other)


def test_cache_keys_separate_universes() -> None:
    params = {"num_leaves": 8, "learning_rate": 0.03}
    h = {u.key(): train.cell_hash(params, u) for u in knobs.candidates(_cfg())}
    assert len(set(h.values())) == len(h)  # every universe gets its own search/knob cell
    assert train.cell_hash(params, Universe(2800)) == train.cell_hash(dict(reversed(list(params.items()))), Universe(2800))
    assert train.cell_hash(params, Universe(2800)) != train._params_hash(params)


def test_knobs_module_imports_only_light_deps() -> None:
    """The GPU venv imports stealth.knobs: module level may use stdlib, numpy, pandas, stealth.data, stealth.cv only."""
    import ast

    src = Path(knobs.__file__).read_text()
    top = [n for n in ast.parse(src).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    mods = set()
    for n in top:
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        else:
            mods |= {f"{n.module}.{a.name}" if n.module == "stealth" else n.module for a in n.names}
    allowed = {"__future__", "dataclasses", "datetime", "json", "logging", "re", "pathlib", "typing", "numpy", "pandas",
               "stealth.cv", "stealth.data"}
    assert mods <= allowed, mods - allowed
