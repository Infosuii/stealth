"""M2-M6 training paths on a tiny run (< 1 min): knobs decided once, adoption-gate fallback, folds_eras.json, CatBoost
timing drop, external (mlp) member pending vs loaded, re-decision refusal, config validation, fast per-era kernels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from stealth import knobs, metrics
from stealth import train as trainmod

RUN = "pytest_m2"
LGBM = {"num_leaves": 8, "learning_rate": 0.1, "min_data_in_leaf": 500, "feature_fraction": 0.3, "lambda_l2": 10,
        "bagging_fraction": 0.7, "bagging_freq": 1, "max_bin": 16}


def _tiny_cfg_dict(repo_root: Path) -> dict:
    cfg = yaml.safe_load((repo_root / "configs" / "smoke.yaml").read_text())
    cfg["run"]["min_era"] = 6100
    cfg["cv"]["n_folds"] = 2
    cfg["cv"]["forward"] = {"enabled": True, "train_end": 6400, "test_start": 6420}
    cfg["knobs"] = {"member": "lgbm_everest", "era_start": [6100, 6000], "rows_rule": ["none", "filter80"],
                    "time_decay": [False, True], "alpha_total": 0.2}
    cfg["members"] = [
        {"name": "lgbm_everest", "family": "lgbm", "target": "target_everest", "group": "everest", "arm": "lgbm", "features": "all",
         "input": "nan", "params": dict(LGBM), "n_estimators": None, "es_cap": 30},
        {"name": "lgbm_tougroute", "family": "lgbm", "target": "target_Tougroute", "group": "aux", "features": "all", "input": "nan",
         "params": {"best_of": "lgbm_everest"}, "n_estimators": 20, "es_cap": None},
        {"name": "cat_everest", "family": "cat", "target": "target_everest", "group": "everest", "arm": "cat", "features": "all",
         "input": "nan", "params": {"depth": 3, "learning_rate": 0.1, "iterations": 20, "border_count": 16}, "n_estimators": 20,
         "es_cap": None},
        {"name": "ridge_everest", "family": "ridge", "target": "target_everest", "group": "everest", "arm": "ridge", "features": "all",
         "input": "linear", "params": {"alpha": 1.0e5}, "n_estimators": None, "es_cap": None},
        {"name": "mlp_everest", "family": "mlp", "target": "target_everest", "group": "everest", "arm": "mlp", "features": "all",
         "input": "linear", "params": {"hidden": [8], "dropout": 0.3, "device": "cpu"}, "seeds": [0], "n_estimators": None, "es_cap": 2},
    ]
    cfg["search"] = {"lgbm_everest": {"grid": {"num_leaves": [4, 8]}, "n_trials": 2, "top_k": 2, "adopt_gate": True}}
    cfg["gates"]["p_pos_adopt"] = 1.01  # unreachable: the search can never be adopted -> the member must be its BASE cell
    cfg["bootstrap"]["n_boot"] = 200
    cfg["timing"] = {"catboost_max_fit_seconds": 0.0001, "progress_every_seconds": 5}  # any CatBoost fit exceeds this
    return cfg


def _load(tmp: Path, cfg: dict) -> dict:
    p = tmp / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return trainmod.load_config(p)


@pytest.fixture(scope="module")
def m2_run(repo_root: Path, data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    artifacts = tmp_path_factory.mktemp("m2_artifacts")
    cfg = _load(tmp_path_factory.mktemp("m2_cfg"), _tiny_cfg_dict(repo_root))
    res, code = trainmod.train_run(cfg, RUN, artifacts=artifacts)
    assert code == 0, res.get("traceback")
    run_dir = artifacts / RUN
    oof_cols = list(pd.read_parquet(run_dir / "oof.parquet").columns)
    return {"cfg": cfg, "metrics": res, "run_dir": run_dir, "artifacts": artifacts, "oof_cols": oof_cols}


def test_folds_eras_schema(m2_run: dict) -> None:
    run_dir = m2_run["run_dir"]
    doc = json.loads((run_dir / "folds_eras.json").read_text())
    assert set(doc) >= {"universe", "scoring_min_era", "folds", "forward", "full_train_eras"}
    assert doc["scoring_min_era"] == 6100 and knobs.Universe.from_dict(doc["universe"]).key() == m2_run["metrics"]["universe"]["key"]
    assert [f["name"] for f in doc["folds"]] == ["fold0", "fold1"] and doc["forward"]["name"] == "forward"
    for f in doc["folds"] + [doc["forward"]]:
        assert set(f) == {"name", "train_eras", "test_eras", "inner_train_eras", "inner_val_eras", "count_sources"}
        assert all(isinstance(e, int) for e in f["train_eras"]) and min(f["test_eras"]) >= 6100
        assert set(f["inner_train_eras"]) <= set(f["train_eras"]) and set(f["inner_val_eras"]) <= set(f["train_eras"])
        # the training pool below the scoring universe (6000..6099) joins train + inner_train, > 40 eras before the test block
        pool = [e for e in f["train_eras"] if e < 6100]
        assert pool and max(pool) < min(f["test_eras"]) - 40 and set(pool) <= set(f["inner_train_eras"])
        assert min(f["inner_val_eras"]) >= 6100
    assert doc["folds"][1]["count_sources"] == ["fold1"] and doc["forward"]["count_sources"][-1] == "forward"
    assert doc["full_train_eras"][0] == 6000 and doc["full_train_eras"][-1] == 6524
    # the fold era sets equal what the trees used (spec.count_sources agrees)
    spec = json.loads((run_dir / "members" / "lgbm_everest" / "spec.json").read_text())
    assert spec["count_sources"] == {f["name"]: f["count_sources"] for f in doc["folds"] + [doc["forward"]]}


def test_knobs_decided_once_and_recorded(m2_run: dict) -> None:
    run_dir = m2_run["run_dir"]
    kj = json.loads((run_dir / "knobs.json").read_text())
    assert kj["status"] == "done" and kj["member"] == "lgbm_everest" and kj["alpha_total"] == 0.2
    keys = [c["key"] for c in kj["candidates"]]
    assert keys == ["e6100-none-d0", "e6000-none-d0", "e6100-filter80-d0", "e6100-none-d1"] and kj["default"] == keys[0]
    by = {c["key"]: c for c in kj["candidates"]}
    assert by["e6100-filter80-d0"]["equivalent_to_default"] is True  # late eras all have >= 80 rows
    assert by["e6100-filter80-d0"]["diff_fwd"] == 0.0 and by["e6100-filter80-d0"]["pass"] is False
    alts = [c for c in kj["candidates"] if not c["is_default"]]
    assert sorted(c["holm_rank"] for c in alts) == [1, 2, 3]
    for c in alts:
        for k in ("fit_cv", "fit_fwd", "diff_fwd", "diff_se_fwd", "p_pos_fwd", "holm_threshold"):
            assert np.isfinite(c[k]), (c["key"], k)
        assert c["pass"] == (c["holm_pass"] and c["diff_fwd"] >= 0 and c["diff_cv_indep"] is not None and c["diff_cv_indep"] >= 0)
    assert kj["indep_cv_folds"] == ["fold0"]  # fold1's test block overlaps the forward test block
    chosen = knobs.Universe.from_dict(kj["chosen"])
    assert kj["chosen"]["key"] == chosen.key() == m2_run["metrics"]["universe"]["key"]
    assert (run_dir / "knobs" / "e6100-none-d0" / "oof.parquet").exists() and not (run_dir / "knobs" / "e6100-filter80-d0").exists()
    mk = m2_run["metrics"]["knobs"]
    assert mk["chosen"] == chosen.key() and len(mk["candidates"]) == 4 and "num_leaves" not in json.dumps(mk)
    for name in ("lgbm_everest", "ridge_everest"):
        assert json.loads((run_dir / "members" / name / "spec.json").read_text())["universe"] == chosen.key()


def test_adoption_gate_falls_back_to_base_cell(m2_run: dict) -> None:
    run_dir = m2_run["run_dir"]
    gain = json.loads((run_dir / "search" / "lgbm_everest.gain.json").read_text())
    assert gain["adopted"] is False and "base cell" in gain["adopt_reason"]
    assert 0.0 <= gain["paired_cv"]["p_pos"] <= 1.0 and set(gain["paired_cv"]) >= {"diff_mean", "diff_se", "p_pos", "ci90"}
    assert gain["paired_cv"]["diff_mean"] == pytest.approx(gain["gain"], abs=1e-12)
    adopted = json.loads((run_dir / "search" / "lgbm_everest.adopted.json").read_text())
    assert adopted["adopted"] is False and adopted["best"] == LGBM and adopted["params_list"] == [LGBM]
    spec = json.loads((run_dir / "members" / "lgbm_everest" / "spec.json").read_text())
    assert spec["search_adopted"] is False and spec["top_params"] == [LGBM] and len(spec["full_blobs"]) == 1
    assert not (run_dir / "members" / "lgbm_everest" / "full_s1.bin").exists()
    # the member's OOF IS the base cell's OOF
    base = pd.read_parquet(run_dir / "search" / "lgbm_everest" / gain["base_cell"] / "oof.parquet")
    oof = pd.read_parquet(run_dir / "oof_lgbm_everest.parquet")
    for cv_part in (True, False):  # ids repeat across the CV and forward parts: compare each part on its own
        o = oof.loc[(oof["fold"] >= 0) == cv_part, "pred"].sort_index()
        b = base.loc[(base["fold"] >= 0) == cv_part, "pred"].sort_index()
        assert o.index.equals(b.index)
        np.testing.assert_allclose(o.to_numpy(), b.to_numpy())
    # best_of dependents inherit the BASE params when the search is not adopted
    aux = json.loads((run_dir / "members" / "lgbm_tougroute" / "spec.json").read_text())
    assert aux["params"] == LGBM
    # the knob decision's cell for the chosen universe doubles as the search's base cell (no second protocol run)
    if (run_dir / "knobs" / m2_run["metrics"]["universe"]["key"] / "cell.json").exists():
        assert "base cell reused from the knob decision" in (run_dir / "run.log").read_text()


def test_adoption_decision_rule() -> None:
    d = trainmod.adoption_decision
    assert d({"p_pos": 0.85}, 0.001, p_pos_adopt=0.8)[0] is True
    assert d({"p_pos": 0.80}, 0.0, p_pos_adopt=0.8)[0] is True  # boundaries are inclusive
    assert d({"p_pos": 0.79}, 0.01, p_pos_adopt=0.8)[0] is False
    assert d({"p_pos": 0.95}, -0.0001, p_pos_adopt=0.8)[0] is False
    assert d({"p_pos": 0.95}, None, p_pos_adopt=0.8)[0] is True  # no forward block: CV gate only
    assert d({"p_pos": 0.1}, -1.0, p_pos_adopt=0.8, adopt_gate=False)[0] is True  # M1 behaviour


def test_cat_timing_drop(m2_run: dict) -> None:
    run_dir, m = m2_run["run_dir"], m2_run["metrics"]
    spec = json.loads((run_dir / "members" / "cat_everest" / "spec.json").read_text())
    assert spec["status"] == "dropped_timing" and spec["first_fit_seconds"] > 0.0001
    assert not (run_dir / "oof_cat_everest.parquet").exists() and not list((run_dir / "members" / "cat_everest").glob("*.bin"))
    assert "m__cat_everest" not in m2_run["oof_cols"] and "cat_everest" not in m["members"]
    assert any(n.startswith("cat_everest: CatBoost dropped") for n in m["notes"])
    assert {"member": "cat_everest", "status": "dropped_timing"}.items() <= next(d for d in m["dropped"] if d["member"] == "cat_everest").items()


def test_external_member_pending(m2_run: dict) -> None:
    m = m2_run["metrics"]
    assert "m__mlp_everest" not in m2_run["oof_cols"] and "mlp_everest" not in m["members"]
    note = next(n for n in m["notes"] if n.startswith("external member mlp_everest pending"))
    assert f"scripts/mlp_train.py --run-id {RUN}" in note
    assert next(d for d in m["dropped"] if d["member"] == "mlp_everest")["status"] == "pending_external"
    assert m["status"] == "ok"  # exit code unchanged
    assert m["members"]["lgbm_everest"]["rho_c"] is not None and np.isfinite(m["members"]["lgbm_everest"]["rho_c"])
    assert m["members"]["ridge_everest"]["arm"] == "ridge" and m["members"]["lgbm_tougroute"]["arm"] == "aux"


# ------------------------------------------------------------------------------------------ resume with a finished external
@pytest.fixture(scope="module")
def m2_resumed(m2_run: dict) -> dict:
    """Fake a finished scripts/mlp_train.py member (spec + oof under the run's universe), then resume the run."""
    run_dir = m2_run["run_dir"]
    key = m2_run["metrics"]["universe"]["key"]
    src = pd.read_parquet(run_dir / "oof_lgbm_everest.parquet")
    rng = np.random.default_rng(0)
    fake = src.assign(pred=(src["pred"] + rng.normal(0, 0.05, len(src))).astype(np.float32))
    fake.to_parquet(run_dir / "oof_mlp_everest.parquet")
    feats = json.loads((run_dir / "features.json").read_text())
    mdir = run_dir / "members" / "mlp_everest"
    mdir.mkdir(parents=True, exist_ok=True)
    spec = {"name": "mlp_everest", "family": "mlp", "status": "done", "universe": key, "feature_cols": feats["feature_cols"],
            "missing_cols": feats["missing_cols"], "inner_iters": [2, 2], "frozen_n_iter": 2, "fit_seconds": 1.0,
            "full_blobs": [{"file": "full_s0.bin", "seed": 0}], "corr_with_bench_val": 0.5, "rho_c": 0.1, "seeds": [0],
            "epochs_cap": 2, "device": "cpu", "run_fingerprint": m2_run["metrics"]["run_fingerprint"]}
    (mdir / "full_s0.bin").write_bytes(b"placeholder: side metrics are given, the blob is never read here")
    (mdir / "spec.json").write_text(json.dumps(spec))
    res, code = trainmod.train_run(m2_run["cfg"], RUN, resume=True, artifacts=m2_run["artifacts"])
    assert code == 0, res.get("traceback")
    return {"metrics": res, "run_dir": run_dir, "fake": fake}


def test_external_member_loaded_on_resume(m2_resumed: dict) -> None:
    run_dir, m = m2_resumed["run_dir"], m2_resumed["metrics"]
    oof = pd.read_parquet(run_dir / "oof.parquet")
    assert "m__mlp_everest" in oof.columns and oof["m__mlp_everest"].notna().all()
    fake = m2_resumed["fake"]
    cvp = fake.loc[fake["fold"] >= 0]
    np.testing.assert_allclose(oof.loc[cvp.index, "m__mlp_everest"].to_numpy(), cvp["pred"].to_numpy())
    assert "m__mlp_everest" in pd.read_parquet(run_dir / "forward.parquet").columns
    mm = m["members"]["mlp_everest"]
    assert mm["family"] == "mlp" and mm["arm"] == "mlp" and mm["group"] == "everest" and mm["rho_c"] == 0.1
    assert not any("mlp_everest pending" in n for n in m["notes"])
    log_text = (run_dir / "run.log").read_text()
    assert "external member mlp_everest: loaded" in log_text
    assert "resumed as dropped_timing" in log_text  # resume keeps the CatBoost drop
    assert "knobs: reusing the decision" in log_text  # the knob is never re-decided on resume
    assert "cat_everest" not in m["members"]


def test_universe_mismatch_is_not_reused(m2_resumed: dict, m2_run: dict) -> None:
    run_dir = m2_resumed["run_dir"]
    rd = type("RD", (), {})()
    rd.universe = knobs.Universe(1234)
    rd.feature_cols, rd.missing_cols, rd.core_proxy_cols = [], [], []
    spec, why = trainmod.load_external_member(rd, m2_run["cfg"]["members"][-1], m2_run["cfg"], run_dir)
    assert spec is None and "universe" in why
    # a tree member trained under another universe is retrained, not resumed
    assert trainmod._reusable_spec(rd, "ridge_everest", run_dir / "members" / "ridge_everest" / "spec.json",
                                   run_dir / "oof_ridge_everest.parquet", True) is None


def test_redecision_refused_in_run(m2_resumed: dict, m2_run: dict, tmp_path: Path, repo_root: Path) -> None:
    """A changed config under the same run-id is refused by the run fingerprint BEFORE anything is written."""
    run_dir = m2_run["run_dir"]
    before = {p: p.read_bytes() for p in (run_dir / "metrics.json", run_dir / "config.yaml", run_dir / "knobs.json")}
    cfg = _tiny_cfg_dict(repo_root)
    cfg["knobs"]["time_decay"] = [False]  # candidate set changed under the same run-id
    res, code = trainmod.train_run(_load(tmp_path, cfg), RUN, resume=True, artifacts=m2_run["artifacts"])
    assert code == trainmod.EXIT_ERROR and res["status"] == "failed"
    assert any("different config" in n and "start a new run-id" in n for n in res["notes"])
    assert {p: p.read_bytes() for p in before} == before  # metrics.json / config.yaml / knobs.json untouched
    # the one permitted handoff edit (run.threads) keeps the fingerprint
    same = _tiny_cfg_dict(repo_root)
    same["run"]["threads"] = 1
    fp_a = trainmod.run_fingerprint(m2_run["cfg"], trainmod._data_dir(m2_run["cfg"]))["fingerprint"]
    fp_b = trainmod.run_fingerprint(_load(tmp_path, same), trainmod._data_dir(m2_run["cfg"]))["fingerprint"]
    assert fp_a == fp_b == m2_run["metrics"]["run_fingerprint"]


def test_legacy_run_dir_without_fingerprint_refused(m2_run: dict, tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    (legacy / "members").mkdir(parents=True)
    (legacy / "metrics.json").write_text("{}")
    res, code = trainmod.train_run(m2_run["cfg"], "legacy", resume=True, artifacts=tmp_path)
    assert code == trainmod.EXIT_ERROR and any("without a run fingerprint" in n for n in res["notes"])
    assert (legacy / "metrics.json").read_text() == "{}" and not (legacy / "config.yaml").exists()


def test_external_member_provenance_checks(m2_resumed: dict, m2_run: dict) -> None:
    """Review finding 2: an external member is loaded only with THIS run's fingerprint, seeds, cap and blobs."""
    run_dir = m2_resumed["run_dir"]
    mspec = next(m for m in m2_run["cfg"]["members"] if m["name"] == "mlp_everest")
    good = json.loads((run_dir / "members" / "mlp_everest" / "spec.json").read_text())
    rd = type("RD", (), {})()
    rd.universe = knobs.Universe.from_dict(m2_run["metrics"]["universe"])
    rd.fingerprint = m2_run["metrics"]["run_fingerprint"]
    rd.audit_eras = np.array([], dtype=np.int32)
    assert trainmod.external_problem(rd, mspec, good, run_dir) is None
    cases = [({"run_fingerprint": "0" * 64}, "another run fingerprint"), ({"seeds": [0, 1]}, "seeds"),
             ({"epochs_cap": 1}, "epoch cap"), ({"full_blobs": []}, "no full_blobs"),
             ({"full_blobs": [{"file": "full_s9.bin"}]}, "missing"), ({"status": "running"}, "status")]
    for patch, needle in cases:
        why = trainmod.external_problem(rd, mspec, {**good, **patch}, run_dir)
        assert why is not None and needle in why, (patch, why)
    rd.audit_eras = np.array([6500], dtype=np.int32)  # with an audit partition the member must carry its audit predictions
    assert "audit_mlp_everest.parquet missing" in trainmod.external_problem(rd, mspec, good, run_dir)


def test_skip_and_require_external(m2_resumed: dict, m2_run: dict) -> None:
    res, code = trainmod.train_run(m2_run["cfg"], RUN, resume=True, artifacts=m2_run["artifacts"], skip_external=True)
    assert code == 0, res.get("traceback")
    assert res["external_members"] == {"mlp_everest": "skipped_external"}
    assert "m__mlp_everest" not in pd.read_parquet(m2_resumed["run_dir"] / "oof.parquet").columns
    assert next(d for d in res["dropped"] if d["member"] == "mlp_everest")["status"] == "skipped_external"
    res, code = trainmod.train_run(m2_run["cfg"], RUN, resume=True, artifacts=m2_run["artifacts"], require_external=True)
    assert code == 0 and res["external_members"] == {"mlp_everest": "loaded"}
    spec_path = m2_resumed["run_dir"] / "members" / "mlp_everest" / "spec.json"
    saved = spec_path.read_text()
    try:
        spec_path.write_text(json.dumps({**json.loads(saved), "seeds": [7]}))  # stale: trained with other seeds
        res, code = trainmod.train_run(m2_run["cfg"], RUN, resume=True, artifacts=m2_run["artifacts"], require_external=True)
        assert code == trainmod.EXIT_ERROR and any("--require-external" in n for n in res["notes"])
        res, code = trainmod.train_run(m2_run["cfg"], RUN, resume=True, artifacts=m2_run["artifacts"])
        assert code == 0 and res["external_members"] == {"mlp_everest": "pending"}
        assert "m__mlp_everest" not in pd.read_parquet(m2_resumed["run_dir"] / "oof.parquet").columns
    finally:
        spec_path.write_text(saved)


def test_interrupted_retrain_never_exposes_a_done_member(m2_run: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review finding 2 / contract 3: spec.json says 'running' from before the first new blob until 'done' is written last."""
    run_dir = m2_run["run_dir"]
    spec_path = run_dir / "members" / "ridge_everest" / "spec.json"
    assert json.loads(spec_path.read_text())["status"] == "done"

    def boom(*a, **k):
        raise KeyboardInterrupt("simulated Ctrl-C during the full refit")

    monkeypatch.setattr(trainmod, "refit_full", boom)
    rd = trainmod.RunData(m2_run["cfg"], trainmod._data_dir(m2_run["cfg"]), fingerprint=m2_run["metrics"]["run_fingerprint"])
    rd.universe = knobs.Universe.from_dict(m2_run["metrics"]["universe"])
    mspec = next(m for m in m2_run["cfg"]["members"] if m["name"] == "ridge_everest")
    with trainmod.Progress(60) as prog, pytest.raises(KeyboardInterrupt):
        trainmod.train_member(rd, mspec, m2_run["cfg"], run_dir, resume=False, prog=prog)
    spec = json.loads(spec_path.read_text())
    assert spec["status"] == "running" and not (run_dir / "oof_ridge_everest.parquet").exists()
    assert trainmod._reusable_spec(rd, "ridge_everest", spec_path, run_dir / "oof_ridge_everest.parquet", True) is None
    monkeypatch.undo()
    with trainmod.Progress(60) as prog:  # a normal resume retrains it and ends with "done"
        spec = trainmod.train_member(rd, mspec, m2_run["cfg"], run_dir, resume=True, prog=prog)
    assert spec["status"] == "done" and (run_dir / "oof_ridge_everest.parquet").exists()


def test_cell_cache_identity(m2_run: dict, tmp_path: Path) -> None:
    """A cached search/knob cell is reused only under its own cell id (params + universe + run fingerprint + protocol)."""
    run_dir = m2_run["run_dir"]
    d = next(p.parent for p in (run_dir / "search" / "lgbm_everest").glob("*/result.json"))
    cid = d.name
    assert trainmod._load_protocol(d, cid) is not None
    assert trainmod._load_protocol(d, "deadbeef0000") is None
    u = knobs.Universe.from_dict(m2_run["metrics"]["universe"])
    assert trainmod.cell_hash(LGBM, u, "a" * 64) != trainmod.cell_hash(LGBM, u, "b" * 64) != trainmod.cell_hash(LGBM, u)
    import optuna

    names = [s.study_name for s in optuna.get_all_study_summaries(f"sqlite:///{run_dir / 'optuna.db'}")]
    assert names and all(n.startswith("lgbm_everest@" + m2_run["metrics"]["run_fingerprint"][:16]) for n in names)


def test_fold_count_does_not_depend_on_its_own_test_labels(m2_run: dict) -> None:
    """Review finding 1 regression: permuting fold0's TEST labels leaves fold0's tree count and predictions unchanged."""
    cfg = m2_run["cfg"]
    rd = trainmod.RunData(cfg, trainmod._data_dir(cfg), fingerprint="x")
    rd.universe = knobs.Universe.from_dict(m2_run["metrics"]["universe"])
    mspec = {**next(m for m in cfg["members"] if m["name"] == "lgbm_everest"), "params": dict(LGBM)}
    a = trainmod.run_protocol(rd, mspec, dict(LGBM), seed=0, threads=1)
    f0 = rd.folds[0]
    rows = rd.rows(f0.test_eras)
    rng = np.random.default_rng(5)
    t = rd.df[trainmod.data.PRIMARY_TARGET].to_numpy().copy()
    t[rows] = rng.permutation(t[rows])
    rd.df[trainmod.data.PRIMARY_TARGET] = t
    b = trainmod.run_protocol(rd, mspec, dict(LGBM), seed=0, threads=1)
    assert a.count_sources["fold0"] == ["fold0"] and a.frozen_n_iter_per_fold[0] == b.frozen_n_iter_per_fold[0]
    ids = a.oof.index[a.oof_fold.to_numpy() == 0]
    np.testing.assert_array_equal(a.oof.loc[ids].to_numpy(), b.oof.loc[ids].to_numpy())


# ------------------------------------------------------------------------------------------ config validation (section 11)
_BAD_CASES = [
    (lambda c: c["knobs"].update(min_rows=[None, 80]), "use knobs.rows_rule"),
    (lambda c: c["run"].update(min_rows=80), "run.min_rows must be null"),
    (lambda c: c["knobs"].update(rows_rule=["none", "rows80"]), "rows_rule entry"),
    (lambda c: c["knobs"].update(era_start=[6000]), "must contain run.min_era"),
    (lambda c: c["knobs"].update(bogus=1), "knobs.bogus"),
    (lambda c: c["cv"]["forward"].update(enabled=False), "knobs need cv.forward.enabled"),
    (lambda c: c["search"]["lgbm_everest"].update(n_trials=26), "max_search_trials"),
    (lambda c: c["members"][-1].pop("seeds"), "needs seeds"),
    (lambda c: c["members"][-1].update(input="nan"), "family mlp needs input: linear"),
    (lambda c: c["members"][0].update(colour="red"), "unknown keys"),
    (lambda c: c["blend"]["simulation"].update(p_c_grid=[0.0, 0.6]), "p_c_grid value"),
    (lambda c: c["blend"]["simulation"].update(p_b_grid=[0.0, 0.8]), "p_b_grid value"),
    (lambda c: c["blend"].update(neutralisation={"select": True, "central_mean": 0.01, "pessimistic_mean": 0.02}),
     "central_mean must be one of"),
    (lambda c: c["blend"].update(neutralisation={"select": True, "central_mean": 0.009, "pessimistic_mean": 0.02}),
     "needs blend.simulation.p_c_grid"),
    (lambda c: (c["blend"]["simulation"].update(p_c_grid=[0.25, 0.5]),
                c["blend"].update(neutralisation={"select": True, "central_mean": 0.009, "pessimistic_mean": 0.02})),
     "needs 0.0 in blend.simulation.p_c_grid"),
    (lambda c: c["blend"].update(admission={"base": "lgbm", "order": []}), "admission.base"),
    (lambda c: c["gates"].update(member_corr_max=1.5), "member_corr_max"),
]


def test_load_config_m2_validation(tmp_path: Path, repo_root: Path) -> None:
    import re

    good = _tiny_cfg_dict(repo_root)
    assert _load(tmp_path, good)["gates"]["max_search_trials"] == 25
    for name in ("smoke", "test", "full", "minifull"):
        trainmod.load_config(repo_root / "configs" / f"{name}.yaml")
    for mutate, needle in _BAD_CASES:
        bad = json.loads(json.dumps(good))
        mutate(bad)
        with pytest.raises(ValueError, match=re.escape(needle)):
            _load(tmp_path, bad)


# ------------------------------------------------------------------------------------------ fast per-era kernels
def test_fast_kernels_match_sdk_path(train_slice: pd.DataFrame) -> None:
    sub = train_slice.loc[train_slice["era"] >= 6400]
    rng = np.random.default_rng(1)
    pred = pd.Series(rng.normal(size=len(sub)) + sub["target_everest"].to_numpy(), index=sub.index)
    pred.iloc[:7] = np.nan  # NaN pairs dropped
    pred[(sub["era"] == sub["era"].min()).to_numpy()] = 1.0  # constant predictions in one era -> 0.0
    tgt = sub["target_everest"].astype(np.float64).copy()
    tgt.iloc[-3:] = np.nan
    a = metrics.per_era_fit(pred, tgt, sub["era"])
    b = trainmod.per_era_fit_fast(pred, tgt, sub["era"])
    assert a.index.equals(b.index) and b.index.dtype == np.int32 and b.name == "fit"
    np.testing.assert_allclose(b.to_numpy(), a.to_numpy(), atol=1e-12)
    assert b.iloc[0] == 0.0
    sa, sb = metrics.per_era_spearman(pred, tgt, sub["era"]), trainmod.per_era_spearman_fast(pred, tgt, sub["era"])
    assert sa.index.equals(sb.index)
    np.testing.assert_allclose(sb.to_numpy(), sa.to_numpy(), atol=1e-12)
