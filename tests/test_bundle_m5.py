"""M5/M6 bundle side (docs/INTERFACES.md section 18): p resolution + p_source, --require-frozen, manifest additions,
fold-blob fallback refusal, assert_plain, and a bag-subset lgbm + ridge + mlp bundle through the real pkl_check
gate in the clean lane venv (``artifacts/pklcheck_venv``, never rebuilt here)."""
from __future__ import annotations

import io
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stealth import _runtime as rt
from stealth import bundle as bd
from stealth import pkl_check
from tests.test_runtime import lgbm_blob, mlp_blob, ridge_blob

REPO = Path(__file__).resolve().parents[1]
PKLCHECK_PY = REPO / "artifacts" / "pklcheck_venv" / "bin" / "python"
NEU_SIM = {"p_b": 0.25, "p_c": 0.5, "p_source": "simulation", "criterion": "test", "table": [], "reason": "test", "z_source": "ew"}


def write_member(run_dir: Path, name: str, spec: dict, blobs: list[bytes], *, kind: str = "full") -> None:
    mdir = run_dir / "members" / name
    mdir.mkdir(parents=True)
    (mdir / "spec.json").write_text(json.dumps({"name": name, "status": "done", **spec}))
    for i, b in enumerate(blobs):
        (mdir / (f"full_s{i}.bin" if kind == "full" else f"fold{i}.bin")).write_bytes(b)


def make_m5_run(run_dir: Path, live: pd.DataFrame, *, neutralisation: dict | None = NEU_SIM, top_p: tuple | None = (0.25, 0.5),
                sharpe_folds: str | None = None) -> list[str]:
    """features.json + blend.json + members: lgbm on a column bag (subset), ridge, mlp (2 seeds), aux lgbm at weight 0.
    ``sharpe_folds`` = "differ" | "same" adds a sharpe-subset lgbm member that has ONLY fold blobs."""
    cols = [c for c in live.columns if c.startswith("feature_")]
    missing_cols = cols[:15]
    rng = np.random.default_rng(0)
    X = rt.nan_matrix(rt.raw_matrix(live, cols))
    y = rng.normal(size=len(live))
    bag = cols[1::2]
    pos = [cols.index(c) for c in bag]
    lin_cols = cols + [f"miss_{c}" for c in missing_cols] + ["n_missing"]
    run_dir.mkdir(parents=True)
    (run_dir / "features.json").write_text(json.dumps({"feature_cols": cols, "missing_cols": missing_cols, "core_proxy_cols": cols[-10:]}))
    write_member(run_dir, "lgbm_everest_bag0", {"family": "lgbm", "target": "target_everest", "group": "everest", "arm": "bags",
                                                "input": "nan", "feature_cols": bag,
                                                "feature_cols_per_fold": {"fold0": bag, "fold1": bag, "forward": bag}},
                 [lgbm_blob(X[:, pos], y, 0)])
    write_member(run_dir, "ridge_everest", {"family": "ridge", "target": "target_everest", "group": "everest", "input": "linear",
                                            "feature_cols": cols, "missing_cols": missing_cols},
                 [ridge_blob(list(rng.permutation(lin_cols)), rng.normal(size=len(lin_cols)) * 0.1, 0.5)])
    write_member(run_dir, "mlp_everest", {"family": "mlp", "target": "target_everest", "group": "everest", "arm": "mlp",
                                          "input": "linear", "feature_cols": cols},
                 [mlp_blob(cols, hidden=(16, 8), seed=1), mlp_blob(cols, hidden=(16, 8), seed=2)])
    write_member(run_dir, "lgbm_ghat", {"family": "lgbm", "target": "target_Ghat", "group": "aux", "input": "nan", "feature_cols": cols},
                 [lgbm_blob(X, y, 3)])
    weights = {"lgbm_everest_bag0": 0.4, "ridge_everest": 0.2, "mlp_everest": 0.4, "lgbm_ghat": 0.0}
    if sharpe_folds:
        sub = [cols[3:40], cols[3:40] if sharpe_folds == "same" else cols[5:50]]
        full_sub = cols[3:40]
        blobs = [lgbm_blob(X[:, [cols.index(c) for c in s]], y, 4 + k) for k, s in enumerate(sub)]
        write_member(run_dir, "lgbm_everest_sharpe", {"family": "lgbm", "target": "target_everest", "group": "everest", "arm": "sharpe",
                                                      "input": "nan", "feature_cols": full_sub,
                                                      "feature_cols_per_fold": {"fold0": sub[0], "fold1": sub[1], "forward": cols}},
                     blobs, kind="folds")
        weights["lgbm_everest_sharpe"] = 0.2
    blend = {"weights": weights, "rule": "nnls", "core_proxy_cols": cols[-10:],
             "arms": {"lgbm_everest_bag0": "bags", "ridge_everest": "ridge", "mlp_everest": "mlp", "lgbm_ghat": "aux"},
             "final": {"admitted_arms": ["bags", "ridge", "mlp"], "rule": "nnls"}}
    if top_p is not None:
        blend["p_b"], blend["p_c"] = top_p
    if neutralisation is not None:
        blend["neutralisation"] = neutralisation
    (run_dir / "blend.json").write_text(json.dumps(blend))
    return cols


@pytest.fixture(scope="module")
def m5(tmp_path_factory: pytest.TempPathFactory, live_frame: pd.DataFrame) -> dict:
    root = tmp_path_factory.mktemp("m5")
    run_dir = root / "artifacts" / "run_m5"
    cols = make_m5_run(run_dir, live_frame)
    out = root / "models" / "stealth_m5.pkl"
    bundle, sha, mpath = bd.build_and_write(run_dir, out, require_frozen=True)
    return {"root": root, "run_dir": run_dir, "cols": cols, "out": out, "bundle": bundle, "sha": sha, "manifest": mpath}


def test_bundle_contents_and_manifest(m5: dict):
    b = m5["bundle"]
    assert [m["name"] for m in b["members"]] == ["lgbm_everest_bag0", "mlp_everest", "ridge_everest"]  # zero-weight aux dropped
    assert {m["name"]: m["arm"] for m in b["members"]} == {"lgbm_everest_bag0": "bags", "mlp_everest": "mlp", "ridge_everest": "ridge"}
    assert (b["p_b"], b["p_c"], b["p_source"]) == (0.25, 0.5, "simulation")
    assert b["blend_rule"] == "nnls" and b["admitted_arms"] == ["bags", "ridge", "mlp"]
    assert abs(sum(b["weights"].values()) - 1.0) < 1e-12
    bd.assert_plain(b)
    man = json.loads(m5["manifest"].read_text())
    assert man["p_source"] == "simulation" and man["blend_rule"] == "nnls" and man["admitted_arms"] == ["bags", "ridge", "mlp"]
    assert {m["name"]: (m["arm"], m["n_features"], m["n_blobs"]) for m in man["members"]} == {
        "lgbm_everest_bag0": ("bags", 89, 1), "mlp_everest": ("mlp", 178, 2), "ridge_everest": ("ridge", 178, 1)}
    assert man["n_features"] == 178 and man["sha256"] == m5["sha"]


def test_pickle_predicts_identically_on_shuffled_live_columns(m5: dict, live_frame: pd.DataFrame, live_bench: pd.Series):
    predict = pickle.loads(m5["out"].read_bytes())
    feats = [c for c in live_frame.columns if c.startswith("feature_")]
    out = predict(live_frame[feats], live_bench)
    assert type(out) is pd.DataFrame and out.shape == (90, 1) and out.index.equals(live_frame.index)
    v = out.iloc[:, 0].to_numpy()
    assert np.isfinite(v).all() and (v > 0).all() and (v < 1).all()
    shuffled = live_frame.sample(frac=1.0, axis=1, random_state=7).copy()
    shuffled["feature_zz_extra"] = 1
    pd.testing.assert_frame_equal(predict(shuffled, live_bench), out)
    assert predict.bundle_meta["families"]["mlp_everest"] == "mlp" and predict.bundle_meta["p_source"] == "simulation"
    raw = m5["out"].read_bytes()
    assert b"torch" not in raw
    _, bad, total = pkl_check_scan(raw)
    assert bad == [] and total > 0


def pkl_check_scan(raw: bytes):
    src = pkl_check.CHECKER_SOURCE
    ns: dict = {}
    exec(src[src.index("ALLOWED_AFTER ="): src.index("\ndef check_frame")], ns)  # noqa: S102 - our own checker code
    return ns["scan_bytes"](raw)


@pytest.mark.skipif(not PKLCHECK_PY.exists(), reason="artifacts/pklcheck_venv missing (clean lane venv; never rebuilt by tests)")
def test_pkl_check_gate_passes_in_clean_lane_venv(m5: dict, data_dir: Path):
    report = pkl_check.run_check(m5["out"], python=PKLCHECK_PY, data_dir=data_dir, eras=3, seed=2, timeout=400)
    assert report["passed"], pkl_check.format_report(report)
    assert report["byte_scan"]["torch"] == 0 and report["byte_scan"]["stealth_non_runtime"] == []
    assert report["bundle_meta"]["families"] == {"lgbm_everest_bag0": "lgbm", "mlp_everest": "mlp", "ridge_everest": "ridge"}
    assert "torch" not in report.get("modules_loaded", [])
    assert report["isolation"] == {"stealth_importable": False, "torch_importable": False}
    assert report["libs"] == {p.split("==")[0]: p.split("==")[1] for p in pkl_check.lane_pins()}
    assert report["sha256"] == bd.sha256_file(m5["out"])
    mpath = pkl_check.update_manifest(m5["out"], report["passed"], report)
    rec = json.loads(mpath.read_text())["pkl_check"]
    assert rec["passed"] is True and rec["sha256"] == m5["sha"] and rec["python"].startswith("3.11.")
    assert rec["isolation"]["stealth_importable"] is False and rec["libs"]["numpy"] == "2.4.6"


def test_resolve_p_precedence(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="stealth.bundle"):
        assert bd.resolve_p({"neutralisation": NEU_SIM, "p_b": 0.25, "p_c": 0.5})[:3] == (0.25, 0.5, "simulation")
        assert not caplog.records
        assert bd.resolve_p({"p_b": 0.4, "p_c": 0.1})[:3] == (0.4, 0.1, "config")
        assert bd.resolve_p({"neutralisation": {**NEU_SIM, "p_source": "config"}, "p_b": 0.5, "p_c": 0.25})[:3] == (0.5, 0.25, "config")
        assert not caplog.records
        assert bd.resolve_p({})[:3] == (bd.DEFAULT_P_B, bd.DEFAULT_P_C, "default")
        assert any("defaults" in r.getMessage() for r in caplog.records)
        caplog.clear()
        assert bd.resolve_p({"neutralisation": NEU_SIM}, p_b=0.1)[:3] == (0.1, 0.5, "override")
        assert any("OVERRIDE" in r.getMessage() for r in caplog.records)
        assert bd.resolve_p({"p_b": 0.4, "p_c": 0.1}, p_c=0.0)[:3] == (0.4, 0.0, "override")
        caplog.clear()
        # top-level values that disagree with the frozen ones: the simulation wins, loudly
        assert bd.resolve_p({"neutralisation": NEU_SIM, "p_b": 0.75, "p_c": 0.0})[:3] == (0.25, 0.5, "simulation")
        assert any("differ" in r.getMessage() for r in caplog.records)


def test_resolve_p_keys_on_p_source_not_on_presence():
    """blend.py always writes `neutralisation`; with select false its p_source is "config" (request blend_select)."""
    neu_cfg = {"p_b": 0.5, "p_c": 0.25, "p_source": "config", "table": []}
    assert bd.resolve_p({"neutralisation": neu_cfg, "p_b": 0.5, "p_c": 0.25, "p_source": "config"})[:3] == (0.5, 0.25, "config")
    assert bd.resolve_p({"neutralisation": neu_cfg})[:3] == (0.5, 0.25, "config")  # values from neutralisation when no mirror
    # top-level p_source mirror as the fallback when neutralisation is absent / lacks p_source
    assert bd.resolve_p({"p_b": 0.25, "p_c": 0.0, "p_source": "simulation"})[:3] == (0.25, 0.0, "simulation")
    assert bd.resolve_p({"neutralisation": {"table": []}, "p_b": 0.25, "p_c": 0.0, "p_source": "simulation"})[:3] == (0.25, 0.0, "simulation")


def test_member_subset_bundle(tmp_path: Path, live_frame: pd.DataFrame, caplog: pytest.LogCaptureFixture):
    run = tmp_path / "artifacts" / "run_sub"
    make_m5_run(run, live_frame, neutralisation={"p_b": 0.5, "p_c": 0.25, "p_source": "config"}, top_p=(0.5, 0.25))
    b = bd.build_bundle(run, members=["lgbm_everest_bag0"])
    assert [m["name"] for m in b["members"]] == ["lgbm_everest_bag0"] and b["weights"] == {"lgbm_everest_bag0": 1.0}
    assert b["member_subset"] == ["lgbm_everest_bag0"] and b["weights_source"].startswith("blend")
    # the constants were configured for the selected 3-member blend, not for this single member: inherited, never frozen
    assert b["admitted_arms"] == ["bags"] and b["p_source"] == "inherited" and "inherited" in b["p_reason"]
    assert (b["p_b"], b["p_c"]) == (0.5, 0.25)
    two = bd.build_bundle(run, members=["mlp_everest", "lgbm_everest_bag0"])
    assert two["weights"] == pytest.approx({"lgbm_everest_bag0": 0.5, "mlp_everest": 0.5})  # 0.4 / 0.4 renormalised
    with caplog.at_level(logging.WARNING, logger="stealth.bundle"):
        zero = bd.build_bundle(run, members=["lgbm_ghat", "ridge_everest"])  # lgbm_ghat has blend weight 0
    assert zero["weights"] == {"lgbm_ghat": 0.5, "ridge_everest": 0.5} and zero["weights_source"].startswith("equal")
    assert any("EQUAL weights" in r.getMessage() for r in caplog.records)
    with pytest.raises(ValueError, match="no members/<m>/spec.json"):
        bd.build_bundle(run, members=["nope"])
    out = tmp_path / "models" / "baseline.pkl"
    rc = bd.main(["--run-id", "run_sub", "--out", str(out), "--artifacts-dir", str(tmp_path / "artifacts"), "--members", "lgbm_everest_bag0"])
    assert rc == 0
    man = json.loads(bd.manifest_path(out).read_text())
    assert man["member_subset"] == ["lgbm_everest_bag0"] and [m["name"] for m in man["members"]] == ["lgbm_everest_bag0"]
    assert man["p_source"] == "inherited" and "inherited" in man["p_reason"]
    assert bd.main(["--run-id", "run_sub", "--out", str(out), "--artifacts-dir", str(tmp_path / "artifacts"), "--members",
                    "lgbm_everest_bag0", "--require-frozen"]) == 1  # config constants are never "frozen"
    # handoff 03's path: explicit zero overrides on a member subset stay "override"
    ov = bd.build_bundle(run, members=["lgbm_everest_bag0"], p_b=0.0, p_c=0.0)
    assert (ov["p_b"], ov["p_c"], ov["p_source"]) == (0.0, 0.0, "override")


def test_member_subset_inherits_simulated_constants_unless_it_is_the_selected_blend(tmp_path: Path, live_frame: pd.DataFrame):
    """contract review 5: a subset of a SIMULATED blend never ships its constants as frozen; the selected blend does."""
    run = tmp_path / "artifacts" / "run_sim_sub"
    make_m5_run(run, live_frame)
    sub = bd.build_bundle(run, members=["mlp_everest"])
    assert sub["p_source"] == "inherited" and (sub["p_b"], sub["p_c"]) == (0.25, 0.5)
    with pytest.raises(ValueError, match="require-frozen.*inherited"):
        bd.build_bundle(run, members=["mlp_everest"], require_frozen=True)
    # different relative weights over the selected members (equal weights: lgbm_ghat has weight 0) -> another prediction
    eq = bd.build_bundle(run, members=["lgbm_everest_bag0", "mlp_everest", "ridge_everest", "lgbm_ghat"])
    assert eq["weights_source"].startswith("equal") and eq["p_source"] == "inherited"
    # exactly the selected positive-weight set (any order) with the blend's relative weights: still the simulated blend
    same = bd.build_bundle(run, members=["ridge_everest", "mlp_everest", "lgbm_everest_bag0"], require_frozen=True)
    assert same["p_source"] == "simulation" and same["weights"] == pytest.approx(
        {"lgbm_everest_bag0": 0.4, "mlp_everest": 0.4, "ridge_everest": 0.2})
    with pytest.raises(ValueError, match="override"):
        bd.build_bundle(run, members=["mlp_everest"], p_b=0.0, p_c=0.0, require_frozen=True)


def test_require_frozen_refuses_non_simulation_sources(tmp_path: Path, live_frame: pd.DataFrame):
    cfg_run = tmp_path / "artifacts" / "run_cfg"
    make_m5_run(cfg_run, live_frame, neutralisation=None, top_p=(0.5, 0.25))
    b = bd.build_bundle(cfg_run)
    assert b["p_source"] == "config" and (b["p_b"], b["p_c"]) == (0.5, 0.25)
    with pytest.raises(ValueError, match="require-frozen"):
        bd.build_bundle(cfg_run, require_frozen=True)
    out = tmp_path / "models" / "x.pkl"
    assert bd.main(["--run-id", "run_cfg", "--out", str(out), "--artifacts-dir", str(tmp_path / "artifacts"), "--require-frozen"]) == 1
    assert not out.exists()
    def_run = tmp_path / "artifacts" / "run_def"
    make_m5_run(def_run, live_frame, neutralisation=None, top_p=None)
    assert bd.build_bundle(def_run)["p_source"] == "default"
    with pytest.raises(ValueError, match="require-frozen"):
        bd.build_bundle(def_run, require_frozen=True)
    sim_run = tmp_path / "artifacts" / "run_sim"
    make_m5_run(sim_run, live_frame)
    with pytest.raises(ValueError, match="override"):  # an override is never "frozen"
        bd.build_bundle(sim_run, p_b=0.5, require_frozen=True)
    b2 = bd.build_bundle(sim_run, p_b=0.9)  # capped, recorded as override
    assert b2["p_source"] == "override" and b2["p_b"] == rt.P_B_CAP and b2["p_c"] == 0.5
    assert bd.main(["--run-id", "run_sim", "--out", str(out), "--artifacts-dir", str(tmp_path / "artifacts"), "--require-frozen"]) == 0
    assert json.loads(bd.manifest_path(out).read_text())["p_source"] == "simulation"


def test_fold_fallback_refused_when_fold_feature_lists_differ(tmp_path: Path, live_frame: pd.DataFrame):
    bad = tmp_path / "run_differ"
    make_m5_run(bad, live_frame, sharpe_folds="differ")
    with pytest.raises(ValueError, match="refusing the fold-blob fallback"):
        bd.build_bundle(bad)
    ok = tmp_path / "run_same"
    make_m5_run(ok, live_frame, sharpe_folds="same")  # only the forward list differs, and forward.bin is never bundled
    b = bd.build_bundle(ok)
    assert b["blob_source"]["lgbm_everest_sharpe"] == "folds"
    assert len(next(m for m in b["members"] if m["name"] == "lgbm_everest_sharpe")["blobs"]) == 2
    bd.check_fold_fallback("m", {"feature_cols": ["a"], "feature_cols_per_fold": [["a"], ["a"]]}, ["fold0", "fold1"])  # list form
    with pytest.raises(ValueError, match="differ"):
        bd.check_fold_fallback("m", {"feature_cols": ["a"], "feature_cols_per_fold": [["a"], ["b"]]}, ["fold0", "fold1"])
    with pytest.raises(ValueError, match="sparse"):  # metadata for fold1 but fold1.bin was not loaded
        bd.check_fold_fallback("m", {"feature_cols": ["a"], "feature_cols_per_fold": [["a"], ["b"]]}, ["fold0"])


def test_fold_fallback_validates_the_exact_loaded_fold_names(tmp_path: Path, live_frame: pd.DataFrame):
    """contract review 6: fold0.bin + fold2.bin must be checked as fold0 / fold2, never as fold0 / fold1."""
    cols = ["a", "b", "c"]
    spec = {"feature_cols": cols, "feature_cols_per_fold": {"fold0": cols, "fold1": cols, "fold2": ["a"], "forward": ["b"]}}
    with pytest.raises(ValueError, match=r"fold2"):
        bd.check_fold_fallback("m", spec, ["fold0", "fold2"])
    # the old count-based form compared fold0 + fold1 only; it now also sees that fold2's metadata has no loaded blob
    with pytest.raises(ValueError, match="sparse"):
        bd.check_fold_fallback("m", spec, 2)
    with pytest.raises(ValueError, match="no feature list"):
        bd.check_fold_fallback("m", {"feature_cols": cols, "feature_cols_per_fold": {"fold0": cols}}, ["fold0", "fold3"])
    with pytest.raises(ValueError, match="unknown key"):
        bd.check_fold_fallback("m", {"feature_cols": cols, "feature_cols_per_fold": {"fold0": cols, "holdout": cols}}, ["fold0"])
    bd.check_fold_fallback("m", {"feature_cols": cols, "feature_cols_per_fold": {"fold0": cols, "forward": ["x"]}}, ["fold0"])
    bd.check_fold_fallback("m", {"feature_cols": cols}, ["fold0", "fold2"])  # legacy spec without per-fold metadata
    # end to end: fold1.bin of a sharpe member renamed to fold2.bin -> the sparse set is refused at build time
    run = tmp_path / "run_sparse"
    make_m5_run(run, live_frame, sharpe_folds="same")
    mdir = run / "members" / "lgbm_everest_sharpe"
    (mdir / "fold1.bin").rename(mdir / "fold2.bin")
    with pytest.raises(ValueError, match="refusing the fold-blob fallback"):
        bd.build_bundle(run)


def test_declared_full_blobs_are_enforced(tmp_path: Path, live_frame: pd.DataFrame):
    """contract review 3: spec.full_blobs is the exact list of refits the bundle averages."""
    run = tmp_path / "run_declared"
    cols = make_m5_run(run, live_frame)
    mdir = run / "members" / "mlp_everest"
    spec_path = mdir / "spec.json"
    spec = json.loads(spec_path.read_text())
    declared = [{"file": "full_s1.bin", "seed": 2}, {"file": "full_s0.bin", "seed": 1}]
    spec_path.write_text(json.dumps({**spec, "full_blobs": declared}))
    b = bd.build_bundle(run)
    mlp = next(m for m in b["members"] if m["name"] == "mlp_everest")
    assert mlp["blobs"] == [(mdir / "full_s1.bin").read_bytes(), (mdir / "full_s0.bin").read_bytes()]  # declared order
    assert b["blob_files"]["mlp_everest"] == ["full_s1.bin", "full_s0.bin"]
    # one declared seed missing -> refused (the average would silently change)
    (mdir / "full_s1.bin").rename(mdir / "full_s1.bak")
    with pytest.raises(FileNotFoundError, match=r"full_s1\.bin.*missing"):
        bd.build_bundle(run)
    (mdir / "full_s1.bak").rename(mdir / "full_s1.bin")
    # an undeclared extra refit (stale artifact of an interrupted retrain) -> refused
    (mdir / "full_s2.bin").write_bytes((mdir / "full_s0.bin").read_bytes())
    with pytest.raises(ValueError, match="not declared"):
        bd.build_bundle(run)
    (mdir / "full_s2.bin").unlink()
    # every declared refit gone + fold blobs present -> refused, never an unannounced switch to the fold models
    for f in ("full_s0.bin", "full_s1.bin"):
        (mdir / f).unlink()
    (mdir / "fold0.bin").write_bytes(mlp_blob(cols, hidden=(16, 8), seed=3))
    with pytest.raises(FileNotFoundError, match="declared full refit"):
        bd.build_bundle(run)
    # a legacy spec (nothing declared) still falls back to the fold models
    spec_path.write_text(json.dumps(spec))
    assert bd.build_bundle(run)["blob_source"]["mlp_everest"] == "folds"
    for bad in ([{"file": "fold0.bin"}], ["full_s0.bin", "full_s0.bin"], "full_s0.bin"):
        with pytest.raises(ValueError):
            bd.declared_full_files({"full_blobs": bad})
    for empty in ([], None, ""):
        with pytest.raises(ValueError):
            bd.declared_full_files({"full_blobs": empty})
    assert bd.declared_full_files({}) is None


def test_non_finite_constants_and_weights_are_refused(tmp_path: Path, live_frame: pd.DataFrame):
    """contract review 7: NaN / inf / negative numbers never reach the closure."""
    run = tmp_path / "run_nan"
    make_m5_run(run, live_frame)
    blend_path = run / "blend.json"
    good = json.loads(blend_path.read_text())
    for weights in ({**good["weights"], "ridge_everest": float("nan")}, {**good["weights"], "ridge_everest": float("inf")},
                    {**good["weights"], "lgbm_ghat": -0.1}):
        blend_path.write_text(json.dumps({**good, "weights": weights}))
        with pytest.raises(ValueError, match="finite and >= 0"):
            bd.build_bundle(run)
    blend_path.write_text(json.dumps({**good, "neutralisation": {**NEU_SIM, "p_b": float("nan")}}))
    with pytest.raises(ValueError, match="finite"):
        bd.build_bundle(run)
    blend_path.write_text(json.dumps(good))
    with pytest.raises(ValueError, match="finite"):
        bd.build_bundle(run, p_c=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        bd.build_bundle(run, p_b=float("nan"))
    assert bd.build_bundle(run)["p_source"] == "simulation"


def test_assert_plain_and_bad_member_specs(tmp_path: Path, live_frame: pd.DataFrame):
    bd.assert_plain({"a": [1, 2.5, "x", b"y", None, True, {"b": []}]})
    for bad in ({"a": np.float64(1.0)}, {"a": np.arange(3)}, {"a": (1, 2)}, {1: "x"}, {"a": [object()]}, {"a": io.BytesIO()}):
        with pytest.raises(TypeError):
            bd.assert_plain(bad)
    run = tmp_path / "run_bad"
    make_m5_run(run, live_frame)
    spec_path = run / "members" / "mlp_everest" / "spec.json"
    spec = json.loads(spec_path.read_text())
    spec_path.write_text(json.dumps({**spec, "status": "running"}))
    with pytest.raises(ValueError, match="not 'done'"):
        bd.build_bundle(run)
    spec_path.write_text(json.dumps({**spec, "family": "tabnet"}))
    with pytest.raises(ValueError, match="not supported"):
        bd.build_bundle(run)
    spec_path.write_text(json.dumps(spec))
    (run / "members" / "mlp_everest" / "full_s0.bin").write_bytes(mlp_blob(["feature_not_in_run"] * 3))
    with pytest.raises(ValueError, match="linear frame"):  # subset violation caught at build time, not in the lane
        bd.build_bundle(run)
