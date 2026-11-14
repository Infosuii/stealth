"""Reserved audit partition (docs/INTERFACES.md section 22): development / full refits stop before cv.audit.train_end,
audit predictions come from the exact bundled blobs, stealth.audit scores the FROZEN blend once, and the audit labels
never reach any adaptive decision (permuting them leaves knobs, search, ladder, NNLS, weights and constants unchanged).
Tiny real-data run (~1-2 min on 2 cores); skipped without data/."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from stealth import _runtime, audit, blend, bundle, data, knobs
from stealth import train as trainmod

LGBM = {"num_leaves": 8, "learning_rate": 0.1, "min_data_in_leaf": 500, "feature_fraction": 0.3, "lambda_l2": 10,
        "bagging_fraction": 0.7, "bagging_freq": 1, "max_bin": 16}
TRAIN_END, TEST_START = 6300, 6320


def _cfg(repo_root: Path) -> dict:
    cfg = yaml.safe_load((repo_root / "configs" / "smoke.yaml").read_text())
    cfg["name"] = "audit_tiny"
    cfg["run"].update(min_era=6000, threads=2)
    cfg["cv"].update(n_folds=2, forward={"enabled": True, "train_end": 6150, "test_start": 6170},
                     audit={"enabled": True, "train_end": TRAIN_END, "test_start": TEST_START})
    cfg["knobs"] = {"member": "lgbm_everest", "era_start": [6000, 5900], "rows_rule": ["none"], "time_decay": [False],
                    "alpha_total": 0.2}
    cfg["members"] = [
        {"name": "lgbm_everest", "family": "lgbm", "target": "target_everest", "group": "everest", "arm": "lgbm",
         "features": "all", "input": "nan", "params": dict(LGBM), "n_estimators": None, "es_cap": 30},
        {"name": "lgbm_tougroute", "family": "lgbm", "target": "target_Tougroute", "group": "aux", "arm": "aux",
         "features": "all", "input": "nan", "params": {"best_of": "lgbm_everest"}, "n_estimators": 20, "es_cap": None},
        {"name": "ridge_everest", "family": "ridge", "target": "target_everest", "group": "everest", "arm": "ridge",
         "features": "all", "input": "linear", "params": {"alpha": 1.0e5}, "n_estimators": None, "es_cap": None},
    ]
    cfg["search"] = {"lgbm_everest": {"grid": {"num_leaves": [4, 8]}, "n_trials": 2, "top_k": 2, "adopt_gate": True}}
    cfg["refit"] = {"seeds": [0]}
    cfg["blend"].update(nnls=True, admission={"base": ["lgbm"], "order": ["aux", "ridge"], "sharpe_gain": 0.05},
                        simulation={"n_draws": 4, "p_b_grid": [0.0, 0.5], "p_c_grid": [0.0, 0.25],
                                    "bench_fit_means": [0.0, 0.009, 0.02], "bench_fit_sd": 0.035, "seed": 0},
                        neutralisation={"select": True, "central_mean": 0.009, "pessimistic_mean": 0.02})
    cfg["bootstrap"]["n_boot"] = 200
    cfg["timing"] = {"catboost_max_fit_seconds": 120, "progress_every_seconds": 30}
    # eras 6000..6299 are a high-signal regime: feature_Agouti ALONE has a mean per-era Spearman of +0.20 there (+0.07 on
    # 5000..5999), so the full-history leak gate (0.18) would void this toy run without any leak
    cfg["gates"]["leak_spearman"] = 0.5
    return cfg


def _run(cfg_path: Path, artifacts: Path, run_id: str) -> Path:
    cfg = trainmod.load_config(cfg_path)
    res, code = trainmod.train_run(cfg, run_id, artifacts=artifacts)
    assert code == 0, res.get("traceback")
    out, code = blend.blend_run(artifacts / run_id)
    assert code == 0
    return artifacts / run_id


@pytest.fixture(scope="module")
def runs(repo_root: Path, data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("audit")
    cfg_path = tmp / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg(repo_root), sort_keys=False))
    a = _run(cfg_path, tmp / "a", "audit_a")
    # run B: identical except that every TARGET of the reserved audit eras is permuted inside its era
    orig = data.load_split

    def permuted(split, data_dir=data.DATA_DIR, columns=None):
        df = orig(split, data_dir, columns)
        if split != "train":
            return df
        rng = np.random.default_rng(99)
        tcols = [c for c in df.columns if c.startswith("target_")]
        sel = (df["era"] >= TEST_START).to_numpy()
        for c in tcols:
            v = df[c].to_numpy().copy()
            idx = np.flatnonzero(sel)
            v[idx] = v[rng.permutation(idx)]
            df[c] = v
        return df

    mp = pytest.MonkeyPatch()
    mp.setattr(trainmod.data, "load_split", permuted)
    try:
        b = _run(cfg_path, tmp / "b", "audit_b")
    finally:
        mp.undo()
    return {"a": a, "b": b, "cfg": trainmod.load_config(cfg_path)}


def test_partition_boundaries_and_audit_file(runs: dict) -> None:
    run_dir = runs["a"]
    fe = json.loads((run_dir / "folds_eras.json").read_text())
    dev_eras = {e for f in [*fe["folds"], fe["forward"]] for k in ("train_eras", "test_eras", "inner_train_eras", "inner_val_eras")
                for e in f[k]}
    assert max(dev_eras) < TRAIN_END and max(fe["full_train_eras"]) < TRAIN_END
    assert fe["audit"]["train_end"] == TRAIN_END and min(fe["audit"]["eras"]) == TEST_START
    for name in ("lgbm_everest", "ridge_everest", "lgbm_tougroute"):
        spec = json.loads((run_dir / "members" / name / "spec.json").read_text())
        assert spec["status"] == "done" and spec["full_train_eras"][1] < TRAIN_END and spec["audit_file"] == f"audit_{name}.parquet"
    aud = pd.read_parquet(run_dir / "audit.parquet")
    assert int(aud["era"].min()) == TEST_START and aud.index.is_unique and aud.notna().all().all()
    assert {"m__lgbm_everest", "m__ridge_everest", "m__lgbm_tougroute", "baseline__lgbm_everest"} <= set(aud.columns)
    train = data.load_split("train", columns=[data.PRIMARY_TARGET])
    want = train.index[train["era"].to_numpy() >= TEST_START]
    assert set(aud.index) == set(want)
    base = json.loads((run_dir / "baseline" / "spec.json").read_text())
    assert base["status"] == "done" and base["universe"] == knobs.default_universe(runs["cfg"]).key()
    mets = json.loads((run_dir / "metrics.json").read_text())
    assert mets["audit"]["enabled"] and mets["audit"]["baseline"] == "baseline__lgbm_everest"
    assert any("SELECTION ESTIMATES" in n for n in mets["notes"])


def test_audit_predictions_are_the_bundled_blobs(runs: dict) -> None:
    """Rank-for-rank: the audit's unneutralised blend equals the pickle's predict(...) with p_b = p_c = 0."""
    run_dir = runs["a"]
    b = bundle.build_bundle(run_dir, p_b=0.0, p_c=0.0)
    predict = _runtime.make_predict(b)
    aud = pd.read_parquet(run_dir / "audit.parquet")
    bl = json.loads((run_dir / "blend.json").read_text())
    w = {k: float(v) for k, v in bl["weights"].items() if float(v) > 0}
    feats = data.load_split("train", columns=b["feature_cols"])
    for e in sorted(aud["era"].unique())[:5]:
        rows = aud.index[aud["era"].to_numpy() == e]
        pred = predict(feats.loc[rows, b["feature_cols"]])["prediction"].to_numpy()
        Z = blend.gaussianise_per_era(aud.loc[rows, [f"m__{n}" for n in w]].rename(columns=lambda c: c[3:]), aud.loc[rows, "era"])
        z = blend.blend_scores(Z, w).to_numpy()
        np.testing.assert_allclose(pred, _runtime.rank01(z), atol=1e-12)


def test_audit_labels_never_reach_a_decision(runs: dict) -> None:
    """Every adaptive choice of run B (audit labels permuted) equals run A's; only the audit targets differ."""
    a, b = runs["a"], runs["b"]
    ka, kb = (json.loads((d / "knobs.json").read_text()) for d in (a, b))
    assert ka["chosen"] == kb["chosen"] and [c["p_value_fwd"] for c in ka["candidates"]] == [c["p_value_fwd"] for c in kb["candidates"]]
    ga, gb = (json.loads((d / "search" / "lgbm_everest.gain.json").read_text()) for d in (a, b))
    assert ga["adopted"] == gb["adopted"] and ga["topk_cells"] == gb["topk_cells"] and ga["paired_cv"] == gb["paired_cv"]
    ba, bb = (json.loads((d / "blend.json").read_text()) for d in (a, b))
    for k in ("rule", "weights", "admitted", "p_b", "p_c", "p_source"):
        assert ba[k] == bb[k], k
    assert ba["ladder"] == bb["ladder"] and ba["neutralisation"]["table"] == bb["neutralisation"]["table"]
    oa, ob = (pd.read_parquet(d / "oof.parquet") for d in (a, b))
    pd.testing.assert_frame_equal(oa, ob)
    aa, ab = (pd.read_parquet(d / "audit.parquet") for d in (a, b))
    pred_cols = [c for c in aa.columns if c.startswith(("m__", "baseline__"))]
    pd.testing.assert_frame_equal(aa[pred_cols], ab[pred_cols])  # predictions do not depend on audit labels
    assert not np.array_equal(aa[data.PRIMARY_TARGET].to_numpy(), ab[data.PRIMARY_TARGET].to_numpy())


def test_stealth_audit_scores_the_frozen_blend_once(runs: dict) -> None:
    run_dir = runs["a"]
    before = (run_dir / "blend.json").read_bytes()
    assert audit.main(["--run-id", run_dir.name, "--artifacts", str(run_dir.parent)]) == 0
    assert (run_dir / "blend.json").read_bytes() == before  # never touches the frozen selection
    doc = json.loads((run_dir / "audit.json").read_text())
    for k in ("schema", "run_id", "status", "run_fingerprint", "blend_sha256", "partition", "frozen", "blend", "baseline",
              "paired_vs_baseline", "sharpe_vs_baseline", "drawdown", "stability", "bootstrap_sensitivity", "scenario",
              "verdict", "notes", "frozen_core_step"):
        assert k in doc, k
    assert doc["schema"] == "stealth.audit/1" and doc["status"] == "ok"
    assert doc["partition"]["eras"][0] == TEST_START and doc["partition"]["n_eras"] == 6524 - TEST_START + 1
    assert doc["verdict"]["technical"] == "complete" and doc["verdict"]["audit_win"] in (True, False)
    assert "NOT platform P&L" in doc["drawdown"]["label"] and doc["drawdown"]["blend"]["max_drawdown"] >= 0
    assert [r["block"] for r in doc["bootstrap_sensitivity"]] == [10, 20, 40]
    assert doc["stability"]["n_segments"] == 4 and sum(s["n_eras"] for s in doc["stability"]["segments"]) == doc["partition"]["n_eras"]
    pv = doc["paired_vs_baseline"]
    assert pv["inference_valid"] and 0.0 < pv["p_value"] <= 1.0
    aud = pd.read_parquet(run_dir / "audit.parquet")
    fit_b = trainmod.per_era_fit_fast(aud["baseline__lgbm_everest"], aud[data.PRIMARY_TARGET], aud["era"]).mean()
    assert doc["baseline"]["mean"] == pytest.approx(float(fit_b), abs=1e-12)
    assert doc["blend_sha256"] == __import__("hashlib").sha256(before).hexdigest()
    # a second run of the module reproduces the numbers (it changes nothing it reads)
    doc2, _ = audit.audit_run(run_dir)
    assert doc2["blend"] == doc["blend"] and doc2["paired_vs_baseline"] == doc["paired_vs_baseline"]
    # the label-permuted run scores differently on the audit (the module does read the audit labels)
    doc_b, _ = audit.audit_run(runs["b"])
    assert doc_b["blend"]["mean"] != doc["blend"]["mean"]


def test_audit_refuses_inconsistent_inputs(runs: dict, tmp_path: Path) -> None:
    import shutil

    run_dir = tmp_path / "copy"
    shutil.copytree(runs["a"], run_dir)
    bl = json.loads((run_dir / "blend.json").read_text())
    bl["weights"]["not_a_member"] = 0.5
    (run_dir / "blend.json").write_text(json.dumps(bl))
    with pytest.raises(ValueError, match="lacks the weighted member"):
        audit.audit_run(run_dir)
    assert audit.main(["--run-id", "copy", "--artifacts", str(tmp_path)]) == 1
    assert json.loads((run_dir / "audit.json").read_text())["status"] == "failed"


def test_audit_config_validation(repo_root: Path, tmp_path: Path) -> None:
    bad = [({"train_end": 6300, "test_start": 6310}, "test_start - train_end"),
           ({"train_end": 6100, "test_start": 6320}, "cv.forward.test_start must lie below"),
           ({"train_end": 6300, "test_start": 6320, "extra": 1}, "cv.audit.extra")]
    for audit_cfg, needle in bad:
        cfg = _cfg(repo_root)
        cfg["cv"]["audit"] = {"enabled": True, **audit_cfg}
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(cfg))
        with pytest.raises(ValueError, match=needle):
            trainmod.load_config(p)
