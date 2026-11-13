"""Bundle -> cloudpickle -> load in a `python -I` subprocess with NO stealth import; plus the pkl_check gate.

The fake run directory mimics artifacts/<run_id> (blend.json + members/<name>/{spec.json,full_s0.bin}).
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stealth import _runtime as rt
from stealth import bundle as bd
from stealth import pkl_check


def _ridge_blob(columns: list[str], coef: np.ndarray, intercept: float) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, coef=np.asarray(coef, dtype=np.float64), intercept=np.float64(intercept),
             columns=np.asarray(columns, dtype=str))
    return buf.getvalue()


def _lgbm_blob(X: np.ndarray, y: np.ndarray, seed: int) -> bytes:
    import lightgbm as lgb

    ds = lgb.Dataset(X.astype(np.float32), label=y, params={"max_bin": 16})
    booster = lgb.train({"objective": "regression", "verbosity": -1, "num_leaves": 4, "seed": seed, "num_threads": 1,
                         "min_data_in_leaf": 5, "deterministic": True}, ds, num_boost_round=5)
    return booster.model_to_string().encode("utf-8")


def _xgb_blob(X: np.ndarray, y: np.ndarray, seed: int) -> bytes:
    import xgboost as xgb

    d = xgb.DMatrix(X.astype(np.float32), label=y, missing=np.nan)
    booster = xgb.train({"objective": "reg:squarederror", "max_depth": 2, "seed": seed, "nthread": 1, "verbosity": 0,
                         "tree_method": "hist", "max_bin": 16}, d, num_boost_round=5)
    return bytes(booster.save_raw("ubj"))


def make_fake_run(run_dir: Path, live: pd.DataFrame, *, p_b: float = 0.5, p_c: float = 0.25, use_folds: bool = False) -> list[str]:
    """Write a minimal artifacts/<run> with ridge (linear) + lgbm (nan) + xgb (nan) members; returns feature cols."""
    cols = [c for c in live.columns if c.startswith("feature_")]
    missing_cols = cols[:15]
    rng = np.random.default_rng(0)
    raw = rt.raw_matrix(live, cols)
    X = rt.nan_matrix(raw)
    y = rng.normal(size=len(live))
    lin_cols = cols + [f"miss_{c}" for c in missing_cols] + ["n_missing"]
    members = {
        "ridge_everest": ("ridge", "linear", cols, [_ridge_blob(lin_cols, rng.normal(size=len(lin_cols)) * 0.1, 0.5)]),
        "lgbm_everest": ("lgbm", "nan", cols, [_lgbm_blob(X, y, 0), _lgbm_blob(X, y, 1)]),
        "xgb_aux": ("xgb", "nan", cols[::2], [_xgb_blob(X[:, ::2], y, 0)]),
        "dropped_member": ("lgbm", "nan", cols, [_lgbm_blob(X, y, 3)]),
    }
    for name, (family, inp, fcols, blobs) in members.items():
        mdir = run_dir / "members" / name
        mdir.mkdir(parents=True)
        spec = {"name": name, "family": family, "target": "target_everest", "group": "everest", "input": inp,
                "feature_cols": fcols, "missing_cols": missing_cols if family == "ridge" else [], "n_iter": 5}
        (mdir / "spec.json").write_text(json.dumps(spec))
        for i, blob in enumerate(blobs):
            (mdir / (f"fold{i}.bin" if use_folds else f"full_s{i}.bin")).write_bytes(blob)
        if use_folds:
            (mdir / "forward.bin").write_bytes(blobs[0])  # must be ignored
    blend = {"weights": {"ridge_everest": 0.25, "lgbm_everest": 0.5, "xgb_aux": 0.25, "dropped_member": 0.0},
             "rule": "ew", "p_b": p_b, "p_c": p_c, "core_proxy_cols": cols[-10:]}
    (run_dir / "blend.json").write_text(json.dumps(blend))
    return cols


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory, live_frame: pd.DataFrame) -> dict:
    root = tmp_path_factory.mktemp("bundle")
    run_dir = root / "artifacts" / "run_fake"
    cols = make_fake_run(run_dir, live_frame)
    out = root / "models" / "stealth_test.pkl"
    bundle, sha, mpath = bd.build_and_write(run_dir, out)
    return {"root": root, "run_dir": run_dir, "out": out, "bundle": bundle, "sha": sha, "manifest": mpath, "cols": cols}


def test_build_bundle_contents(built: dict):
    b = built["bundle"]
    assert [m["name"] for m in b["members"]] == ["lgbm_everest", "ridge_everest", "xgb_aux"]  # sorted, zero weight dropped
    assert abs(sum(b["weights"].values()) - 1.0) < 1e-12
    assert b["weights"]["lgbm_everest"] == pytest.approx(0.5)
    assert len(b["feature_cols"]) == 178 and len(b["missing_cols"]) == 15
    assert b["p_b"] == 0.5 and b["p_c"] == 0.25 and len(b["core_proxy_cols"]) == 10
    assert b["blob_source"] == {"lgbm_everest": "full", "ridge_everest": "full", "xgb_aux": "full"}
    assert len(b["members"][0]["blobs"]) == 2
    assert "spec" not in b["members"][0]
    manifest = json.loads(built["manifest"].read_text())
    assert manifest["sha256"] == built["sha"] == bd.sha256_file(built["out"])
    assert manifest["n_features"] == 178 and manifest["pkl_check"] is None
    assert [m["n_blobs"] for m in manifest["members"]] == [2, 1, 1]
    assert manifest["libs"]["lightgbm"] == "4.7.0"
    sha_file = built["out"].with_name(built["out"].name + ".sha256").read_text().split()
    assert sha_file == [built["sha"], built["out"].name]


def test_build_bundle_fold_fallback_and_caps(tmp_path: Path, live_frame: pd.DataFrame):
    run_dir = tmp_path / "run_folds"
    make_fake_run(run_dir, live_frame, p_b=0.9, p_c=0.9, use_folds=True)
    b = bd.build_bundle(run_dir)
    assert b["blob_source"]["lgbm_everest"] == "folds"
    assert len(b["members"][0]["blobs"]) == 2  # fold0, fold1 (forward.bin ignored)
    assert b["p_b"] == rt.P_B_CAP and b["p_c"] == rt.P_C_CAP
    b2 = bd.build_bundle(run_dir, p_b=0.1, p_c=0.0)
    assert b2["p_b"] == 0.1 and b2["p_c"] == 0.0
    with pytest.raises(FileNotFoundError):
        bd.build_bundle(tmp_path / "nope")
    # missing_cols recovered from the ridge npz when spec.json lacks them
    spec_path = run_dir / "members" / "ridge_everest" / "spec.json"
    spec = json.loads(spec_path.read_text())
    spec["missing_cols"] = []
    spec_path.write_text(json.dumps(spec))
    assert len(bd.build_bundle(run_dir)["missing_cols"]) == 15


def test_pickle_loads_in_isolated_subprocess_without_stealth(built: dict, data_dir: Path):
    """python -I, cwd=tmp, PYTHONPATH empty: load + predict twice, assert stealth/torch never imported."""
    code = r"""
import json, pickle, sys
import numpy as np, pandas as pd
raw = open(sys.argv[1], "rb").read()
assert b"torch" not in raw
ns = {}
exec(sys.argv[4], ns)  # the checker's own scan_bytes (stdlib only), so this test and pkl_check agree on the whitelist
_, bad, total = ns["scan_bytes"](raw)
assert not bad and total > 0, bad
predict = pickle.loads(raw)
assert "stealth" not in sys.modules and not any(m.startswith("stealth.") for m in sys.modules)
live = pd.read_parquet(sys.argv[2])
bench = pd.read_parquet(sys.argv[3])["v1_sherpa"].reindex(live.index).rename("eiq_minera_model")
feats = [c for c in live.columns if c.startswith("feature_")]
out1 = predict(live[feats], bench); out2 = predict(live[feats], bench); out3 = predict(live[feats])
assert type(out1) is pd.DataFrame and out1.shape == (len(live), 1) and out1.index.equals(live.index)
v = out1.iloc[:, 0].to_numpy()
assert np.isfinite(v).all() and (v > 0).all() and (v < 1).all()
assert out1.equals(out2) and not out1.equals(out3)
assert "stealth" not in sys.modules and "torch" not in sys.modules
print(json.dumps({"ok": True, "n": int(len(out1)), "modules": sorted(m for m in sys.modules if m in ("lightgbm","xgboost","catboost","scipy","sklearn"))}))
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONPATH"] = ""
    src = pkl_check.CHECKER_SOURCE
    scanner = src[src.index("ALLOWED_AFTER ="): src.index("\ndef check_frame")]
    proc = subprocess.run(
        [sys.executable, "-I", "-c", code, str(built["out"]), str(data_dir / "futures_live.parquet"),
         str(data_dir / "futures_live_benchmark_models.parquet"), scanner],
        cwd="/tmp", env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    assert res["ok"] and res["n"] == 90
    # lazy imports: only the families the bundle uses are loaded (lgbm + xgb + scipy for ndtri), never catboost
    assert "catboost" not in res["modules"] and "lightgbm" in res["modules"] and "xgboost" in res["modules"]


def test_run_check_gate_passes_on_fake_bundle(built: dict, data_dir: Path):
    """Full pkl_check.run_check with the current interpreter (stealth importable, but must not be imported)."""
    report = pkl_check.run_check(built["out"], python=Path(sys.executable), data_dir=data_dir, eras=3, seed=1, timeout=400)
    assert report["passed"], pkl_check.format_report(report)
    assert report["n_eras"] == 4 and report["byte_scan"]["torch"] == 0
    assert report["byte_scan"]["stealth_non_runtime"] == []
    assert report["byte_scan"]["stealth_total"] > 0  # the by-value qualname legitimately appears
    assert report["stats"]["n_bench_eras"] == 4
    assert report["seconds_per_era_max"] < pkl_check.MAX_SECONDS_PER_ERA
    mpath = pkl_check.update_manifest(built["out"], report["passed"])
    assert json.loads(mpath.read_text())["pkl_check"]["passed"] is True


def test_run_check_fails_on_bad_pickle(tmp_path: Path, data_dir: Path):
    """A by-reference pickle of a plain function that returns a Series must fail the gate."""
    import cloudpickle

    def bad_predict(live_features, live_benchmark_models=None):
        return pd.Series(np.linspace(0, 1, len(live_features)), index=live_features.index)

    bad = tmp_path / "bad.pkl"
    with bad.open("wb") as fh:
        cloudpickle.dump(bad_predict, fh)
    report = pkl_check.run_check(bad, python=Path(sys.executable), data_dir=data_dir, eras=0)
    assert not report["passed"]
    assert any("not an exact pandas.DataFrame" in p for p in report["problems"])
    torchy = tmp_path / "torchy.pkl"
    torchy.write_bytes(b"\x80\x05" + b"import torch" + b".")
    report2 = pkl_check.run_check(torchy, python=Path(sys.executable), data_dir=data_dir, eras=0)
    assert not report2["passed"] and any("torch" in p for p in report2["problems"])


def test_byte_scan_whitelist_is_path_aware():
    """co_filename of the by-value module may sit under a checkout whose own path contains 'stealth' (a git worktree
    under /home/u/work/stealth/.claude/...); any other stealth module reference must still fail."""
    src = pkl_check.CHECKER_SOURCE
    ns: dict = {}
    exec(src[src.index("ALLOWED_AFTER ="): src.index("\ndef check_frame")], ns)  # noqa: S102 - our own checker code
    scan = ns["scan_bytes"]
    framed = lambda s: b"\x8c" + bytes([len(s) % 256]) + s + b"\x94"  # noqa: E731 - SHORT_BINUNICODE + MEMOIZE framing
    ok = [b"stealth._runtime", b"/home/u/stealth/stealth/_runtime.py", b"/home/u/work/stealth/.claude/worktrees/m2-m6/stealth/_runtime.py",
          b"C:\\Users\\u\\stealth\\stealth\\_runtime.py", b"stealth", b"/opt/my stealth/x-y/stealth/_runtime.py"]
    for s in ok:
        assert scan(framed(s))[1] == [], s
    bad = [b"stealth.models.mlp", b"/home/u/stealth/stealth/models/mlp.py", b"/home/u/work/stealth/.claude/worktrees/x/stealth/bundle.py",
           b"stealthy_thing", b"stealth/models/__init__.py"]
    for s in bad:
        assert scan(framed(s))[1], s
    assert scan(b"xx torch yy")[0] == 1


def test_lane_pins_and_cli_errors(tmp_path: Path):
    pins = pkl_check.lane_pins()
    assert len(pins) == 9 and all("==" in p for p in pins)
    assert not any(p.startswith(("pytest", "optuna", "everesteer")) for p in pins)
    assert pkl_check.main([str(tmp_path / "missing.pkl")]) == 1
    assert bd.main(["--run-id", "nope", "--out", str(tmp_path / "x.pkl"), "--artifacts-dir", str(tmp_path)]) == 1


def test_bundle_cli_end_to_end(tmp_path: Path, live_frame: pd.DataFrame):
    run_dir = tmp_path / "artifacts" / "run_cli"
    make_fake_run(run_dir, live_frame)
    out = tmp_path / "models" / "stealth_cli.pkl"
    rc = bd.main(["--run-id", "run_cli", "--out", str(out), "--artifacts-dir", str(tmp_path / "artifacts"), "--p-b", "0.3"])
    assert rc == 0 and out.exists()
    manifest = json.loads(bd.manifest_path(out).read_text())
    assert manifest["p_b"] == 0.3 and manifest["pkl"] == "stealth_cli.pkl"
