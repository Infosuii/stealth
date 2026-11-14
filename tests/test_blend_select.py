"""Tests for the M3-M5 blend additions (INTERFACES section 16): arms + hierarchical equal weights, member correlation,
redundancy drop, the pre-registered admission ladder (fit and Sharpe routes), and the staking-aware neutralisation
selection on handcrafted regime grids. Synthetic data only (no parquet files needed)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from stealth import blend as B
from stealth import data

TARGET = "target_everest"


# --------------------------------------------------------------------------------------------- fixtures
def _bins(x: np.ndarray, era: np.ndarray) -> np.ndarray:
    """Per-era 5-level bell bins (0, .25, .5, .75, 1) of a latent, like target_everest."""
    out = np.empty(len(x))
    for e in np.unique(era):
        m = era == e
        r = pd.Series(x[m]).rank(pct=True).to_numpy()
        out[m] = np.select([r <= .05, r <= .25, r <= .75, r <= .95], [0.0, 0.25, 0.5, 0.75], 1.0)
    return out


def synthetic_run(seed: int = 0, n_eras: int = 240, rows: int = 60) -> dict:
    """OOF/forward frames with three independent signal sources (lgbm ~ a, aux ~ b, xgb ~ c), a pure-noise ridge,
    two near-copies of lgbm (bags) and two noisy lgbm copies (cat: individually ~0.75 corr with lgbm)."""
    rng = np.random.default_rng(seed)
    n = n_eras * rows
    era = np.repeat(np.arange(1000, 1000 + n_eras), rows).astype(np.int32)
    a, b, c = rng.standard_normal((3, n))
    y = _bins(0.35 * a + 0.35 * b + 0.35 * c + rng.standard_normal(n), era)
    lg = a + 0.6 * rng.standard_normal(n)
    cols = {
        "lgbm_everest": lg,
        "lgbm_aux1": b + 0.6 * rng.standard_normal(n),
        "xgb_everest": c + 0.6 * rng.standard_normal(n),
        "ridge_everest": rng.standard_normal(n),
        "lgbm_everest_bag0": lg + 0.05 * rng.standard_normal(n),
        "lgbm_everest_bag1": lg + 0.05 * rng.standard_normal(n),
        "cat_everest_a": lg + 1.03 * rng.standard_normal(n),
        "cat_everest_b": lg + 1.03 * rng.standard_normal(n),
    }
    ids = pd.Index([f"r{i:06d}" for i in range(n)], name="id")
    fold = np.minimum((era - 1000) * 3 // n_eras, 2).astype(np.int8)
    oof = pd.DataFrame({"era": era, "fold": fold, TARGET: y.astype(np.float32)}, index=ids)
    for k, v in cols.items():
        oof[f"m__{k}"] = v.astype(np.float32)
    fwd = oof.loc[oof["era"] >= 1000 + n_eras - 60].copy()
    fwd["fold"] = np.int8(-1)
    arms = {"lgbm_everest": "lgbm", "lgbm_aux1": "aux", "xgb_everest": "xgb", "ridge_everest": "ridge",
            "lgbm_everest_bag0": "bags", "lgbm_everest_bag1": "bags", "cat_everest_a": "cat", "cat_everest_b": "cat"}
    specs = {k: {"name": k, "group": "aux" if k == "lgbm_aux1" else "everest", "arm": arms[k], "target": TARGET,
                 "family": "ridge" if k.startswith("ridge") else k.split("_")[0], "status": "done"} for k in cols}
    direction = pd.Series(a + rng.standard_normal(n), index=ids)
    core = pd.Series(0.5 * a + rng.standard_normal(n), index=ids)
    return {"oof": oof, "fwd": fwd, "specs": specs, "direction": direction, "core": core}


def ladder_cfg(order=("aux", "xgb", "ridge", "bags"), corr_max: float = 0.95, n_boot: int = 300) -> dict:
    members = [{"name": n, "group": g, "family": "lgbm", "arm": a} for n, g, a in (
        ("lgbm_everest", "everest", "lgbm"), ("lgbm_aux1", "aux", "aux"), ("xgb_everest", "everest", "xgb"),
        ("cat_everest_a", "everest", "cat"), ("cat_everest_b", "everest", "cat"), ("ridge_everest", "everest", "ridge"),
        ("lgbm_everest_bag0", "everest", "bags"), ("lgbm_everest_bag1", "everest", "bags"))]
    return {"members": members,
            "blend": {"everest_weight": 0.5, "nnls": False, "p_b": 0.5, "p_c": 0.25,
                      "admission": {"base": ["lgbm"], "order": list(order), "sharpe_gain": 0.05}},
            "gates": {"p_pos_adopt": 0.8, "member_corr_max": corr_max, "leak_spearman": 0.9, "suspect_gain_over_baseline": 0.004},
            "bootstrap": {"block": 20, "n_boot": n_boot, "seed": 0}}


@pytest.fixture(scope="module")
def syn() -> dict:
    s = synthetic_run()
    names = B.member_columns(s["oof"])
    s["Z"] = B.gaussianise_per_era(s["oof"][[f"m__{n}" for n in names]].rename(columns=lambda c: c[3:]), s["oof"]["era"])
    s["Zf"] = B.gaussianise_per_era(s["fwd"][[f"m__{n}" for n in names]].rename(columns=lambda c: c[3:]), s["fwd"]["era"])
    return s


# ------------------------------------------------------------------------------------- arms + weights
def test_arm_of_defaults_and_explicit() -> None:
    assert B.arm_of({"name": "lgbm_everest", "group": "everest"}) == "lgbm_everest"
    assert B.arm_of({"name": "xgb_ghat", "group": "aux"}) == "aux"
    assert B.arm_of({"name": "lgbm_everest_bag0", "group": "everest", "arm": "bags"}) == "bags"
    assert B.arm_of({"name": "x", "group": "everest", "arm": ""}) == "x"


def test_blend_weights_hierarchical_vs_m1() -> None:
    groups = {"lgbm_everest": "everest", "bag0": "everest", "bag1": "everest", "bag2": "everest", "bag3": "everest",
              "aux1": "aux", "aux2": "aux"}
    arms = {"lgbm_everest": "lgbm", **{f"bag{i}": "bags" for i in range(4)}, "aux1": "aux", "aux2": "aux"}
    h = B.blend_weights(groups, 0.5, arms)
    assert h["lgbm_everest"] == pytest.approx(0.25) and h["bag2"] == pytest.approx(0.0625)
    assert h["aux1"] == pytest.approx(0.25) and sum(h.values()) == pytest.approx(1.0)
    flat = B.blend_weights(groups, 0.5)
    assert flat["lgbm_everest"] == pytest.approx(0.1) and flat["aux2"] == pytest.approx(0.25)
    # default arms (no `arm` keys): one arm per everest member, all aux in one arm -> exactly the M1 weights, same key order
    default = {n: B.arm_of({"name": n, "group": g}) for n, g in groups.items()}
    assert list(B.blend_weights(groups, 0.5, default).items()) == list(flat.items())
    # no aux group -> everest takes everything; only aux -> aux takes everything
    ev_only = B.blend_weights({"a": "everest", "b": "everest"}, 0.5, {"a": "x", "b": "x"})
    assert ev_only == {"a": 0.5, "b": 0.5}
    assert B.blend_weights({"q": "aux"}, 0.5, {"q": "aux"}) == {"q": 1.0}
    with pytest.raises(ValueError):
        B.blend_weights({}, 0.5, {})


def test_member_corr_matrix(syn) -> None:
    C = B.member_corr_matrix(syn["Z"], syn["oof"]["era"])
    names = list(syn["Z"].columns)
    assert list(C.index) == names and list(C.columns) == names
    np.testing.assert_allclose(np.diag(C.to_numpy()), 1.0)
    np.testing.assert_allclose(C.to_numpy(), C.to_numpy().T)
    assert C.loc["lgbm_everest", "lgbm_everest_bag0"] > 0.99
    assert abs(C.loc["lgbm_everest", "ridge_everest"]) < 0.05
    assert 0.65 < C.loc["lgbm_everest", "cat_everest_a"] < 0.85
    # pairwise-complete per era: a member missing on some rows still gets a correlation
    Z = syn["Z"].copy()
    Z.iloc[::7, 0] = np.nan
    C2 = B.member_corr_matrix(Z, syn["oof"]["era"])
    assert np.isfinite(C2.to_numpy()).all() and C2.iloc[0, 4] > 0.99
    assert B.mean_era_corr(syn["Z"]["lgbm_everest"], syn["Z"]["lgbm_everest"], syn["oof"]["era"]) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------------- ladder
def test_admit_decision_routes() -> None:
    kw = dict(corr_max=0.95, p_pos_adopt=0.8, sharpe_gain=0.05)
    base = {"corr_with_incumbent": 0.7, "fwd_diff": 0.001, "p_pos": 0.9, "dsharpe": 0.0, "p_pos_sharpe": 0.5}
    ok, why = B.admit_decision(base, **kw)
    assert ok and why.startswith("ADMIT (fit route)")
    sharpe = {**base, "p_pos": 0.55, "dsharpe": 0.06, "p_pos_sharpe": 0.85}
    ok, why = B.admit_decision(sharpe, **kw)
    assert ok and "Sharpe route" in why
    assert not B.admit_decision({**sharpe, "dsharpe": 0.04}, **kw)[0]          # Sharpe gain too small
    assert not B.admit_decision({**sharpe, "p_pos_sharpe": 0.7}, **kw)[0]      # Sharpe gain not significant
    assert not B.admit_decision({**base, "fwd_diff": -0.0001}, **kw)[0]        # forward block disagrees
    assert not B.admit_decision({**sharpe, "fwd_diff": -0.0001}, **kw)[0]      # ... on the Sharpe route too
    assert not B.admit_decision({**base, "corr_with_incumbent": 0.96}, **kw)[0]  # redundant with the incumbent
    assert not B.admit_decision({**base, "corr_with_incumbent": float("nan")}, **kw)[0]
    assert B.admit_decision({**base, "fwd_diff": None}, **kw)[0]               # no forward block -> CV gate alone
    assert B.admit_decision({**base, "fwd_diff": 0.0, "p_pos": 0.8}, **kw)[0]   # boundaries are inclusive


def test_ladder_plan_order_and_unregistered(syn) -> None:
    cfg = ladder_cfg(order=("aux", "mlp", "xgb", "bags", "ridge"))
    plan, unreg = B.ladder_plan(list(syn["Z"].columns), syn["specs"], cfg)
    assert [(a, base) for a, _, base in plan] == [("lgbm", True), ("aux", False), ("xgb", False), ("bags", False), ("ridge", False)]
    assert dict((a, m) for a, m, _ in plan)["bags"] == ["lgbm_everest_bag0", "lgbm_everest_bag1"]  # config order
    assert unreg == ["cat"]  # "mlp" (absent from the run) is skipped; "cat" is in the run but not pre-registered


def by_valid(rows: list[dict]) -> bool:
    """Every tested arm carries a valid bootstrap (240 eras = 12 blocks) and a null-centred p-value in (0, 1]."""
    tested = [r for r in rows if r["diff_mean"] is not None]
    return bool(tested) and all(r["inference_valid"] and 0.0 < r["p_value"] <= 1.0 and r["n_blocks"] == 12 for r in tested)


def test_admit_decision_fails_closed_without_inference() -> None:
    kw = {"corr_max": 0.95, "p_pos_adopt": 0.8, "sharpe_gain": 0.05}
    row = {"corr_with_incumbent": 0.5, "fwd_diff": 0.001, "p_pos": 1.0, "dsharpe": 0.3, "p_pos_sharpe": 1.0}
    assert B.admit_decision(row, **kw)[0]
    assert not B.admit_decision({**row, "inference_valid": False, "sharpe_inference_valid": False}, **kw)[0]
    assert B.admit_decision({**row, "inference_valid": False}, **kw)[0]  # the Sharpe route alone still has evidence
    assert not B.admit_decision({**row, "inference_valid": False, "dsharpe": 0.01}, **kw)[0]


def test_nested_nnls_meta_fit_is_purged(syn) -> None:
    """Review finding 5: the NNLS meta-fit rows lie >= 20 eras from the eval fold AND the forward block; the forward
    diff of the gate uses those same weights; too few purged eras fail closed."""
    cfg = ladder_cfg()
    oof, fwd = syn["oof"], syn["fwd"]
    names = ["lgbm_everest", "lgbm_aux1", "xgb_everest"]
    ew = B.blend_weights({n: syn["specs"][n]["group"] for n in names}, 0.5, {n: syn["specs"][n]["arm"] for n in names})
    nn = B.nested_nnls(syn["Z"][names], oof, syn["Zf"][names], fwd, ew, cfg)
    lo_eval = int(oof.loc[oof["fold"] == oof["fold"].max(), "era"].min())
    assert nn["fit_eras"][1] <= min(lo_eval, int(fwd["era"].min())) - B.FORWARD_GAP
    # unpurged, the last fold-1 eras (adjacent to the eval fold) would have entered the fit
    assert int(oof.loc[oof["fold"] == oof["fold"].max() - 1, "era"].max()) > nn["fit_eras"][1]
    mask = B.purged_fit_mask(oof["era"], [np.array([1100, 1150])], gap=20)
    kept = np.unique(oof["era"].to_numpy()[mask])
    assert 1080 in kept and 1081 not in kept and 1169 not in kept and 1170 in kept
    # a fold layout with almost no purged meta rows -> no NNLS
    tiny = oof.loc[oof["era"] < 1000 + 100].copy()
    tiny["fold"] = np.where(tiny["era"] < 1000 + 50, 0, 1).astype(np.int8)
    nn2 = B.nested_nnls(syn["Z"].loc[tiny.index, names], tiny, None, None, ew, cfg)
    assert nn2["adopt"] is False and nn2["nested_weights"] is None and "fail closed" in nn2["reason"]


def test_admission_ladder_admits_signal_rejects_noise_and_redundant(syn) -> None:
    cfg = ladder_cfg()
    red: dict = {}
    admitted, rows = B.admission_ladder(syn["Z"], syn["oof"], syn["Zf"], syn["fwd"], syn["specs"], cfg, redundant=red)
    by = {r["arm"]: r for r in rows}
    keys = ["arm", "members", "admitted", "reason", "diff_mean", "diff_se", "p_pos", "fwd_diff", "dsharpe", "p_pos_sharpe",
            "corr_with_incumbent", "dropped_redundant", "p_value", "n_blocks", "inference_valid", "sharpe_inference_valid"]
    assert all(list(r) == keys for r in rows)
    assert by_valid(rows)
    assert [r["arm"] for r in rows] == ["lgbm", "aux", "xgb", "ridge", "bags", "cat"]
    assert by["lgbm"]["admitted"] and by["lgbm"]["reason"].startswith("base")
    for arm in ("aux", "xgb"):  # independent signal -> clear paired gain on CV and forward
        r = by[arm]
        assert r["admitted"], r["reason"]
        assert r["p_pos"] > 0.95 and r["fwd_diff"] > 0 and r["corr_with_incumbent"] < 0.6
        assert np.isfinite(r["dsharpe"]) and 0.0 <= r["p_pos_sharpe"] <= 1.0
    assert not by["ridge"]["admitted"] and by["ridge"]["p_pos"] < 0.2  # pure noise hurts
    # both bags are > 0.99 correlated with lgbm_everest (kept first) -> dropped as redundant, arm never tested
    assert not by["bags"]["admitted"] and by["bags"]["dropped_redundant"] == ["lgbm_everest_bag0", "lgbm_everest_bag1"]
    assert by["bags"]["p_pos"] is None and "every member redundant" in by["bags"]["reason"]
    assert red["lgbm_everest_bag0"]["other"] == "lgbm_everest" and red["lgbm_everest_bag0"]["corr"] > 0.99
    assert not by["cat"]["admitted"] and "not pre-registered" in by["cat"]["reason"]
    assert admitted == ["lgbm_everest", "lgbm_aux1", "xgb_everest"]


def test_admission_ladder_redundancy_keeps_earlier_member_and_corr_gate(syn) -> None:
    # bag1 vs bag0 when lgbm is not the base: the earlier member of the arm survives
    cfg = ladder_cfg(order=("aux",))
    cfg["blend"]["admission"]["base"] = ["bags"]
    admitted, rows = B.admission_ladder(syn["Z"], syn["oof"], syn["Zf"], syn["fwd"], syn["specs"], cfg)
    assert rows[0]["arm"] == "bags" and rows[0]["members"] == ["lgbm_everest_bag0"]
    assert rows[0]["dropped_redundant"] == ["lgbm_everest_bag1"] and "lgbm_everest_bag0" in admitted
    # the cat arm: members ~0.75 correlated with lgbm (kept at member_corr_max 0.8) but their MEAN correlates ~0.85 with
    # the incumbent -> rejected on corr_with_incumbent
    cfg = ladder_cfg(order=("cat",), corr_max=0.8)
    admitted, rows = B.admission_ladder(syn["Z"], syn["oof"], syn["Zf"], syn["fwd"], syn["specs"], cfg)
    cat = next(r for r in rows if r["arm"] == "cat")
    assert cat["arm"] == "cat" and cat["members"] == ["cat_everest_a", "cat_everest_b"] and cat["dropped_redundant"] == []
    assert cat["corr_with_incumbent"] > 0.8 and not cat["admitted"] and "corr" in cat["reason"]
    assert admitted == ["lgbm_everest"]


# ---------------------------------------------------------------------------------- neutralisation (16.3)
_DEFAULT_CELL = {"fit": 0.020, "score": 0.030, "pess": 0.025, "se": 0.004, "sh": 0.30, "W": 0.010}
NEU = {"select": True, "central_mean": 0.009, "pessimistic_mean": 0.02, "fit_cost_max": 0.003, "sharpe_tol": 0.02, "worst_tol": 0.0005}


def make_grid(spec: dict, p_bs=(0.0, 0.25, 0.5, 0.75), p_cs=(0.0, 0.25, 0.5)) -> dict:
    """Handcrafted regime grid: ``spec[(p_b, p_c)]`` overrides the default cell; W is the bneg score in every mean."""
    grid: dict = {}
    for mk in ("0", "0.009", "0.02"):
        grid[mk] = {}
        for p_c in p_cs:
            by = grid[mk][f"p_c={p_c:g}"] = {}
            for p_b in p_bs:
                s = {**_DEFAULT_CELL, **spec.get((p_b, p_c), {})}
                mix = {"fit": s["fit"], "score": s["pess"] if mk == "0.02" else s["score"], "score_se": s["se"], "score_sharpe": s["sh"]}
                by[f"p_b={p_b:g}"] = {"mix": mix, "bpos": {"score": s["W"] + 0.02, "fit": s["fit"]},
                                      "bneg": {"score": s["W"] + (0.001 if mk == "0" else 0.0), "fit": s["fit"]}}
    grid["meta"] = {}
    return grid


def _cfg(**neu) -> dict:
    return {"blend": {"p_b": 0.5, "p_c": 0.25, "neutralisation": {**NEU, **neu}}}


def _pick(spec: dict, **kw) -> dict:
    return B.select_neutralisation(make_grid(spec, **{k: v for k, v in kw.items() if k in ("p_bs", "p_cs")}),
                                   _cfg(**{k: v for k, v in kw.items() if k not in ("p_bs", "p_cs")}), z_source="ew")


def test_select_neutralisation_flat_grid_picks_simplest() -> None:
    sel = _pick({})
    assert (sel["p_b"], sel["p_c"]) == (0.0, 0.0) and sel["p_source"] == "simulation" and sel["z_source"] == "ew"
    assert len(sel["table"]) == 12 and all(r["eligible"] and r["in_tie_set"] for r in sel["table"])
    assert set(sel["table"][0]) == {"p_b", "p_c", "fit", "score", "score_sharpe", "W", "eligible", "in_tie_set"}
    assert sel["criterion"] and "chosen p_b=0 p_c=0" in sel["reason"]


def test_select_neutralisation_best_sharpe_then_worst_regime_then_simplest() -> None:
    # a clear Sharpe winner
    assert (lambda s: (s["p_b"], s["p_c"]))(_pick({(0.5, 0.25): {"sh": 0.40}})) == (0.5, 0.25)
    # two cells in the tie set (0.40 vs 0.39, tol 0.02): the higher worst-regime score W wins
    sel = _pick({(0.25, 0.0): {"sh": 0.40, "W": 0.010}, (0.5, 0.0): {"sh": 0.39, "W": 0.015}})
    assert (sel["p_b"], sel["p_c"]) == (0.5, 0.0)
    tie = {(r["p_b"], r["p_c"]) for r in sel["table"] if r["in_tie_set"]}
    assert tie == {(0.25, 0.0), (0.5, 0.0)}
    # W within worst_tol (0.0004 < 0.0005) -> smaller p_b + p_c
    sel = _pick({(0.25, 0.0): {"sh": 0.40, "W": 0.0100}, (0.5, 0.25): {"sh": 0.40, "W": 0.0104}})
    assert (sel["p_b"], sel["p_c"]) == (0.25, 0.0)
    # equal p_b + p_c -> smaller p_b
    sel = _pick({(0.25, 0.25): {"sh": 0.40}, (0.5, 0.0): {"sh": 0.40}})
    assert (sel["p_b"], sel["p_c"]) == (0.25, 0.25)
    # a cell just outside the tie set loses even with a much better W
    sel = _pick({(0.25, 0.0): {"sh": 0.40, "W": 0.0}, (0.75, 0.5): {"sh": 0.379, "W": 0.05}})
    assert (sel["p_b"], sel["p_c"]) == (0.25, 0.0)


def test_select_neutralisation_eligibility_rules() -> None:
    # (a) FIT cost above 0.003 at the central mean -> ineligible however good its Sharpe
    sel = _pick({(0.75, 0.0): {"sh": 0.60, "fit": 0.0165}, (0.5, 0.0): {"sh": 0.35, "fit": 0.0171}})
    assert (sel["p_b"], sel["p_c"]) == (0.5, 0.0)
    row = next(r for r in sel["table"] if (r["p_b"], r["p_c"]) == (0.75, 0.0))
    assert not row["eligible"] and not row["in_tie_set"]
    # (b) pessimistic-world score more than one SE below the (0,0) cell's -> ineligible
    sel = _pick({(0.5, 0.0): {"sh": 0.60, "pess": 0.025 - 0.0041}, (0.25, 0.0): {"sh": 0.35, "pess": 0.025 - 0.0039}})
    assert (sel["p_b"], sel["p_c"]) == (0.25, 0.0)


def test_select_neutralisation_falls_back_to_zero_cell() -> None:
    bad = {(pb, pc): {"sh": 0.9, "fit": 0.010} for pb in (0.0, 0.25, 0.5, 0.75) for pc in (0.0, 0.25, 0.5) if (pb, pc) != (0.0, 0.0)}
    sel = _pick(bad)
    assert (sel["p_b"], sel["p_c"]) == (0.0, 0.0) and "only the (0,0) cell is eligible" in sel["reason"]
    assert sum(r["eligible"] for r in sel["table"]) == 1
    # even when the (0,0) cell itself has an undefined Sharpe it remains the fallback
    bad[(0.0, 0.0)] = {"sh": float("nan")}
    assert (lambda s: (s["p_b"], s["p_c"]))(_pick(bad)) == (0.0, 0.0)


def test_select_neutralisation_caps_and_validation() -> None:
    # cells beyond the hard caps (p_b > 0.75, p_c > 0.5) are never eligible
    sel = _pick({(1.0, 0.0): {"sh": 0.9}, (0.25, 0.75): {"sh": 0.9}, (0.25, 0.0): {"sh": 0.35}},
                p_bs=(0.0, 0.25, 1.0), p_cs=(0.0, 0.75))
    assert (sel["p_b"], sel["p_c"]) == (0.25, 0.0)
    assert not any(r["eligible"] for r in sel["table"] if r["p_b"] > 0.75 or r["p_c"] > 0.5)
    with pytest.raises(ValueError, match="reference cell"):
        _pick({}, p_bs=(0.25, 0.5))
    with pytest.raises(ValueError, match="bench mean"):
        _pick({}, central_mean=0.005)


def test_select_neutralisation_off_uses_config() -> None:
    sel = B.select_neutralisation(make_grid({(0.5, 0.25): {"sh": 0.4}}), {"blend": {"p_b": 0.25, "p_c": 0.0}}, z_source="crossfit")
    assert sel["p_source"] == "config" and (sel["p_b"], sel["p_c"]) == (0.25, 0.0)
    assert sel["would_choose"] == {"p_b": 0.5, "p_c": 0.25} and len(sel["table"]) == 12
    # a grid without the reference cell (M1: p_c_grid = [p_c]) still works when select is off
    sel = B.select_neutralisation(make_grid({}, p_cs=(0.25,)), {"blend": {"p_b": 0.5, "p_c": 0.25}})
    assert sel["p_source"] == "config" and sel["table"] == [] and sel["would_choose"] is None


# ------------------------------------------------------------------------------- end to end (blend_run)
def _write_run(tmp: Path, syn: dict, cfg_blend: dict, names: list[str], gates: dict | None = None) -> Path:
    run_dir = tmp / "synrun"
    (run_dir / "members").mkdir(parents=True)
    keep = ["era", "fold", TARGET] + [f"m__{n}" for n in names]
    syn["oof"][keep].to_parquet(run_dir / "oof.parquet")
    syn["fwd"][keep].to_parquet(run_dir / "forward.parquet")
    for n in names:
        (run_dir / "members" / n).mkdir()
        (run_dir / "members" / n / "spec.json").write_text(json.dumps(syn["specs"][n]))
    members = [m for m in ladder_cfg()["members"] if m["name"] in names]
    cfg = {"name": "syn", "run": {"data_dir": "data"}, "members": members, "blend": cfg_blend,
           "bootstrap": {"block": 20, "n_boot": 200, "seed": 0},
           "gates": gates or {"p_pos_adopt": 0.8, "member_corr_max": 0.95, "leak_spearman": 0.9, "suspect_gain_over_baseline": 0.004}}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    (run_dir / "features.json").write_text(json.dumps({"core_proxy_cols": [f"feature_c{i}" for i in range(10)]}))
    (run_dir / "metrics.json").write_text(json.dumps({"run_id": "synrun", "notes": []}))
    return run_dir


@pytest.fixture()
def patched_inputs(syn, monkeypatch):
    monkeypatch.setattr(B, "regime_inputs", lambda run, ids: (syn["direction"].reindex(ids), syn["core"].reindex(ids)))


SIM = {"n_draws": 4, "p_b_grid": [0.0, 0.5], "bench_fit_means": [0.0, 0.009, 0.02], "bench_fit_sd": 0.035, "seed": 0}


def test_blend_run_full_selection(tmp_path, syn, patched_inputs) -> None:
    names = ["lgbm_everest", "lgbm_aux1", "xgb_everest", "ridge_everest", "lgbm_everest_bag0", "lgbm_everest_bag1"]
    bcfg = {"everest_weight": 0.5, "nnls": False, "p_b": 0.5, "p_c": 0.25,
            "admission": {"base": ["lgbm"], "order": ["aux", "xgb", "cat", "ridge", "sharpe", "bags", "mlp"], "sharpe_gain": 0.05},
            "simulation": {**SIM, "p_c_grid": [0.0, 0.25]},
            "neutralisation": {"select": True, "central_mean": 0.009, "pessimistic_mean": 0.02}}
    run_dir = _write_run(tmp_path, syn, bcfg, names)
    out, code = B.blend_run(run_dir)
    assert code == 0
    saved = json.loads((run_dir / "blend.json").read_text())
    for k in ("arms", "member_corr", "ladder", "admitted", "neutralisation", "regime_grid", "final", "regime", "p_source", "redundant"):
        assert k in saved, k
    assert not any(k.startswith("_") for k in saved)
    assert saved["admitted"] == ["lgbm_everest", "lgbm_aux1", "xgb_everest"]
    w = saved["weights"]
    assert w["ridge_everest"] == 0.0 and w["lgbm_everest_bag0"] == 0.0 and w["lgbm_everest_bag1"] == 0.0
    # hierarchical EW on the admitted set: everest 0.5 over arms lgbm + xgb, aux 0.5
    assert w["lgbm_everest"] == pytest.approx(0.25) and w["xgb_everest"] == pytest.approx(0.25) and w["lgbm_aux1"] == pytest.approx(0.5)
    assert sum(w.values()) == pytest.approx(1.0)
    neu = saved["neutralisation"]
    assert neu["p_source"] == "simulation" and neu["z_source"] == "ew" and len(neu["table"]) == 4
    assert (saved["p_b"], saved["p_c"]) == (neu["p_b"], neu["p_c"]) and saved["p_source"] == "simulation"
    assert saved["p_b"] in (0.0, 0.5) and saved["p_c"] in (0.0, 0.25)
    # old-shape regime slice at the chosen p_c (report.py reads it) + the full grid
    assert set(saved["regime"]) == {"0", "0.009", "0.02", "meta"} and saved["regime"]["meta"]["p_c"] == saved["p_c"]
    assert set(saved["regime"]["0.009"]) == {"p_b=0", "p_b=0.5"}
    assert set(saved["regime_grid"]["0.009"]) == {"p_c=0", "p_c=0.25"}
    fin = saved["final"]
    assert fin["admitted_arms"] == ["lgbm", "aux", "xgb"] and fin["rule"] == "ew" and fin["p_source"] == "simulation"
    assert np.isfinite(fin["score_sharpe_central"]) and np.isfinite(fin["forward_fit"]) and fin["oof"]["mean"] > 0
    assert [r["arm"] for r in saved["ladder"]] == ["lgbm", "aux", "xgb", "ridge", "bags"]
    corr = pd.read_csv(run_dir / "member_corr.csv", index_col=0)
    assert list(corr.index) == names and corr.loc["lgbm_everest", "lgbm_everest_bag0"] > 0.99
    assert saved["member_corr"]["names"] == names and len(saved["member_corr"]["matrix"]) == len(names)
    mets = json.loads((run_dir / "metrics.json").read_text())
    assert mets["blend"]["final"]["p_source"] == "simulation" and mets["blend"]["neutralisation"]["p_b"] == saved["p_b"]
    assert "xgb_ladder" in mets["paired"] and mets["paired"]["xgb_ladder"]["admitted"] is True
    assert "ridge_ladder" in mets["paired"] and "bags_ladder" not in mets["paired"]  # bags never tested (all redundant)
    assert any("redundant(lgbm_everest)" in n for n in mets["notes"])
    # the chosen blend vs the baseline member (handoff 02's missing row): persisted, same numbers in both files
    pvb = saved["paired_vs_baseline"]
    assert pvb["baseline"] == "lgbm_everest" and 0.0 <= pvb["p_pos"] <= 1.0 and np.isfinite(pvb["fwd_diff"])
    row = mets["paired"]["blend_vs_lgbm_everest"]
    assert row["diff_mean"] == pvb["diff_mean"] and row["p_pos"] == pvb["p_pos"] and "baseline" not in row
    oof = pd.read_parquet(run_dir / "oof.parquet")
    base_fit = B.fit_series(oof["m__lgbm_everest"], oof[data.PRIMARY_TARGET], oof["era"]).mean()
    assert pvb["diff_mean"] == pytest.approx(fin["oof"]["mean"] - base_fit, abs=1e-12)


def test_blend_run_m1_config_keeps_old_behaviour(tmp_path, syn, patched_inputs) -> None:
    names = ["lgbm_everest", "lgbm_aux1", "ridge_everest"]
    bcfg = {"everest_weight": 0.5, "nnls": True, "p_b": 0.5, "p_c": 0.25, "simulation": dict(SIM)}
    run_dir = _write_run(tmp_path, syn, bcfg, names)
    # M1 specs have no `arm` key: default arms -> flat M1 weights
    for n in names:
        p = run_dir / "members" / n / "spec.json"
        s = json.loads(p.read_text())
        s.pop("arm")
        p.write_text(json.dumps(s))
    out, code = B.blend_run(run_dir)
    assert code == 0
    assert out["admitted"] == names and out["ladder"] == []
    assert list(out["ew_weights"].items()) == list(B.blend_weights({n: syn["specs"][n]["group"] for n in names}, 0.5).items())
    assert out["p_source"] == "config" and (out["p_b"], out["p_c"]) == (0.5, 0.25)
    assert out["neutralisation"]["table"] == []  # no (0,0) cell when p_c_grid defaults to [p_c]
    assert set(out["regime"]) == {"0", "0.009", "0.02", "meta"} and out["regime"]["meta"]["p_c"] == 0.25
    assert set(out["regime_grid"]["0"]) == {"p_c=0.25"}
    if out["rule"] == "nnls":  # NNLS: every reported number cross-fitted, the forward fit with the gate's own weights
        assert out["chosen_is_crossfit"] and out["neutralisation"]["z_source"] == "crossfit"
        assert out["forward_weights"] == out["nnls"]["nested_weights"]
        assert "gate weights" in out["forward_weights_source"]
    assert out["final"]["admitted_arms"] == ["lgbm_everest", "aux", "ridge_everest"]
