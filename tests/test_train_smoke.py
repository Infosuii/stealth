"""End-to-end smoke of stealth.train + stealth.blend on a tiny override of configs/smoke.yaml (< 2 min).

Eras >= 6000, 2 purged folds + a small forward block, ~50 trees, a 2-cell search with top_k=2 (exercises the
top-k averaging path), 10 regime draws. Asserts the artifact layout of docs/INTERFACES.md section 8, finite
metrics, a sane OOF fit20 (|mean| < 0.1), the frozen tree count, resume behaviour and the blend outputs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stealth import blend as blendmod
from stealth import train as trainmod


def _tiny_config(repo_root: Path) -> dict:
    cfg = trainmod.load_config(repo_root / "configs" / "smoke.yaml")
    cfg["run"]["min_era"] = 6000
    cfg["cv"]["n_folds"] = 2
    cfg["cv"]["forward"] = {"enabled": True, "train_end": 6400, "test_start": 6420}
    cfg["members"][0]["n_estimators"] = None
    cfg["members"][0]["es_cap"] = 60
    cfg["members"][1]["n_estimators"] = 50
    cfg["search"] = {"lgbm_everest": {"grid": {"num_leaves": [4, 8]}, "n_trials": 2, "top_k": 2}}
    cfg["blend"]["nnls"] = True
    cfg["blend"]["simulation"]["n_draws"] = 10
    cfg["bootstrap"]["n_boot"] = 200
    cfg["timing"]["progress_every_seconds"] = 5
    return cfg


@pytest.fixture(scope="module")
def smoke_run(repo_root: Path, data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    artifacts = tmp_path_factory.mktemp("artifacts")
    cfg = _tiny_config(repo_root)
    res, code = trainmod.train_run(cfg, "pytest_smoke", artifacts=artifacts)
    assert code == 0, res.get("traceback")
    run_dir = artifacts / "pytest_smoke"
    blend, bcode = blendmod.blend_run(run_dir)
    assert bcode == 0
    return {"cfg": cfg, "metrics": res, "blend": blend, "run_dir": run_dir, "artifacts": artifacts}


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def test_artifacts_exist(smoke_run: dict) -> None:
    run_dir = smoke_run["run_dir"]
    for f in ("config.yaml", "run.log", "folds.json", "features.json", "oof.parquet", "forward.parquet", "metrics.json", "blend.json",
              "optuna.db", "search/lgbm_everest.json"):
        assert (run_dir / f).exists(), f
    for m in smoke_run["cfg"]["members"]:
        mdir = run_dir / "members" / m["name"]
        assert (mdir / "spec.json").exists() and (mdir / "fold0.bin").exists() and (mdir / "fold1.bin").exists()
        assert (mdir / "forward.bin").exists() and (mdir / "full_s0.bin").exists()
        assert (run_dir / f"oof_{m['name']}.parquet").exists()
    # searched member: top-2 refits -> two full blobs the bundle will average
    assert (run_dir / "members" / "lgbm_everest" / "full_s1.bin").exists()
    log_text = (run_dir / "run.log").read_text()
    assert "progress:" in log_text and "frozen n_iter" in log_text


def test_oof_schema_and_folds(smoke_run: dict) -> None:
    oof = pd.read_parquet(smoke_run["run_dir"] / "oof.parquet")
    assert oof.index.name == "id" and oof.index.is_unique
    assert {"era", "fold", "target_everest"} <= set(oof.columns)
    members = [c for c in oof.columns if c.startswith("m__")]
    assert len(members) == 3
    assert str(oof["era"].dtype) == "int32" and str(oof["fold"].dtype) == "int8"
    assert set(oof["fold"].unique()) == {0, 1}
    assert oof[members].notna().all().all()
    folds = json.loads((smoke_run["run_dir"] / "folds.json").read_text())
    assert [f["name"] for f in folds] == ["fold0", "fold1", "forward"]
    fwd = pd.read_parquet(smoke_run["run_dir"] / "forward.parquet")
    assert (fwd["fold"] == -1).all() and fwd["era"].min() >= 6420


def test_member_metrics_sane(smoke_run: dict) -> None:
    m = smoke_run["metrics"]
    assert m["status"] == "ok" and m["leak_check"] == "pass"
    assert _finite(m["timings"]["total_s"]) and m["host"]["python"]
    for name, mm in m["members"].items():
        fit = mm["fit"]
        for k in ("mean", "std", "sharpe", "pct_pos"):
            assert _finite(fit[k]), (name, k)
        assert abs(fit["mean"]) < 0.1, (name, fit["mean"])
        assert fit["n_eras"] > 100
        assert _finite(mm["spearman_mean"]) and abs(mm["spearman_mean"]) < 0.18
        assert _finite(mm["forward_fit"])
        assert _finite(mm["corr_with_bench_val"])
    # frozen tree counts: early-stopped member records inner iterations + a median; fixed member records its count
    assert m["frozen_n_iter"]["lgbm_everest"] is not None and 1 <= m["frozen_n_iter"]["lgbm_everest"] <= 60
    spec = json.loads((smoke_run["run_dir"] / "members" / "lgbm_everest" / "spec.json").read_text())
    assert len(spec["inner_iters"]) == 2 and spec["frozen_n_iter"] == int(round(float(np.median(spec["inner_iters"]))))
    assert len(spec["top_params"]) == 2 and len(spec["full_blobs"]) == 2
    assert m["frozen_n_iter"]["lgbm_tougroute"] == 50
    assert m["frozen_n_iter"]["ridge_everest"] is None
    assert "lgbm_tougroute_vs_lgbm_everest" in m["paired"]
    assert 0.0 <= m["paired"]["lgbm_tougroute_vs_lgbm_everest"]["p_pos"] <= 1.0


def test_blend_outputs(smoke_run: dict) -> None:
    b = smoke_run["blend"]
    w = b["weights"]
    assert set(w) == {"lgbm_everest", "lgbm_tougroute", "ridge_everest"}
    assert abs(sum(w.values()) - 1.0) < 1e-9 and min(w.values()) > 0
    assert b["rule"] in ("ew", "nnls")
    assert b["ew_weights"]["lgbm_tougroute"] == pytest.approx(0.5) and b["ew_weights"]["lgbm_everest"] == pytest.approx(0.25)
    assert b["nnls"] is not None and 0.0 <= b["paired_vs_ew"]["p_pos"] <= 1.0
    assert _finite(b["blend"]["chosen"]["mean"]) and abs(b["blend"]["chosen"]["mean"]) < 0.1
    assert _finite(b["blend"]["forward_fit"])
    assert b["leak_check"] == "pass"
    assert len(b["core_proxy_cols"]) == 10
    regime = b["regime"]
    cell = regime["0.009"]["p_b=0.5"]["mix"]
    for k in ("fit", "unq", "inov", "score"):
        assert _finite(cell[k]), k
    mets = json.loads((smoke_run["run_dir"] / "metrics.json").read_text())
    assert mets["blend"]["rule"] == b["rule"] and "regime" in mets
    assert "blend_nnls_vs_ew" in mets["paired"]


def test_resume_skips_members(smoke_run: dict) -> None:
    res, code = trainmod.train_run(smoke_run["cfg"], "pytest_smoke", resume=True, artifacts=smoke_run["artifacts"])
    assert code == 0
    log_text = (smoke_run["run_dir"] / "run.log").read_text()
    assert log_text.count("resumed from") >= 3
    assert res["members"]["lgbm_everest"]["fit"]["mean"] == pytest.approx(smoke_run["metrics"]["members"]["lgbm_everest"]["fit"]["mean"])


def test_load_config_rejects_bad(tmp_path: Path, repo_root: Path) -> None:
    import yaml

    cfg = yaml.safe_load((repo_root / "configs" / "smoke.yaml").read_text())
    cfg["members"][0]["family"] = "torch"
    del cfg["bootstrap"]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError) as exc:
        trainmod.load_config(p)
    assert "family" in str(exc.value) and "bootstrap" in str(exc.value)
    for name in ("smoke", "test", "full"):
        trainmod.load_config(repo_root / "configs" / f"{name}.yaml")
