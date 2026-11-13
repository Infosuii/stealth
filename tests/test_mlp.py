"""MLP member: npz format, numpy forward vs a hand-computed reference, and scripts/mlp_train.py end to end.

The training script needs torch, which the train venv must NOT have; it is run in a subprocess under
``$STEALTH_GPU_PY`` (laptop: ``.venv-gpu/bin/python``) or the sandbox's ``/opt/ml/bin/python3`` (CPU torch) with
``PYTHONPATH=<repo>``, on a tiny synthetic run directory. Skipped when neither interpreter has torch.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from stealth import _runtime as rt
from stealth import knobs, models
from stealth.models import mlp as mlp_mod

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "mlp_train.py"
AUX = "target_Tougroute"


def _gpu_py() -> str | None:
    for cand in (os.environ.get("STEALTH_GPU_PY"), "/opt/ml/bin/python3"):
        if cand and Path(cand).exists():
            ok = subprocess.run([cand, "-c", "import torch, numpy, pandas, yaml, pyarrow"], capture_output=True, timeout=120)
            if ok.returncode == 0:
                return cand
    return None


GPU_PY = _gpu_py()
needs_torch_py = pytest.mark.skipif(GPU_PY is None, reason="no interpreter with torch (set STEALTH_GPU_PY or /opt/ml/bin/python3)")


def random_mlp(n_in: int, hidden: list[int], seed: int = 0, cols: list[str] | None = None) -> dict:
    rng = np.random.default_rng(seed)
    dims = [n_in, *hidden, 1]
    W = [rng.normal(scale=1.0 / math.sqrt(dims[i]), size=(dims[i], dims[i + 1])).astype(np.float32) for i in range(len(dims) - 1)]
    b = [rng.normal(scale=0.1, size=dims[i + 1]).astype(np.float32) for i in range(len(dims) - 1)]
    return {"W": W, "b": b, "x_mean": rng.normal(size=n_in).astype(np.float32) * 0.1,
            "x_std": (0.5 + rng.random(n_in)).astype(np.float32),
            "columns": cols or [f"feature_{i:03d}" for i in range(n_in)], "act": "gelu_tanh"}


# --------------------------------------------------------------------------------------------------
# numpy side (train venv)
# --------------------------------------------------------------------------------------------------
def test_npz_roundtrip_and_format():
    m = random_mlp(7, [5, 3], seed=1)
    blob = mlp_mod.to_bytes(m)
    import io

    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        keys = set(z.files)
        assert keys == {"W0", "W1", "W2", "b0", "b1", "b2", "x_mean", "x_std", "columns", "act", "n_layers"}
        assert z["W0"].dtype == np.float32 and z["W0"].shape == (7, 5) and z["W2"].shape == (3, 1)
        assert z["x_mean"].dtype == np.float32 and int(z["n_layers"]) == 3 and str(z["act"]) == "gelu_tanh"
    back = mlp_mod.from_bytes(blob)
    assert back["columns"] == m["columns"] and back["n_layers"] == 3
    for a, b in zip(back["W"], m["W"]):
        np.testing.assert_array_equal(a, b.astype(np.float64))
    assert rt.npz_columns(blob) == m["columns"]
    assert b"torch" not in blob and b"stealth" not in blob
    # family dispatch
    assert "mlp" in models.FAMILIES and "mlp" in rt.SUPPORTED_FAMILIES and models.family_extension("mlp") == "npz"
    with pytest.raises(NotImplementedError):
        mlp_mod.fit_fixed({}, None, None, None, n_iter=1, seed=0, threads=1)
    with pytest.raises(NotImplementedError):
        models.fit_member({"name": "m", "family": "mlp"}, pd.DataFrame({"a": [0.0]}), np.zeros(1), None, seed=0, threads=1)


def test_malformed_mlp_blobs_raise_valueerror():
    with pytest.raises(ValueError):
        rt.load_member("mlp", b"")
    m = random_mlp(4, [3])
    m["x_std"] = m["x_std"][:3]
    with pytest.raises(ValueError, match="input size"):
        rt.mlp_to_bytes(m)
    m = random_mlp(4, [3])
    m["act"] = "relu"
    with pytest.raises(ValueError, match="activation"):
        rt.mlp_to_bytes(m)
    m = random_mlp(4, [3])
    m["W"][-1] = np.zeros((3, 2), np.float32)
    m["b"][-1] = np.zeros(2, np.float32)
    with pytest.raises(ValueError, match="one output"):
        rt.mlp_to_bytes(m)


def test_gelu_tanh_reference_values():
    x = np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    ref = np.array([0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3))) for v in x])
    np.testing.assert_allclose(rt.gelu_tanh(x), ref, rtol=0, atol=1e-15)
    assert rt.gelu_tanh(np.array([0.0]))[0] == 0.0
    # known value of the tanh approximation at 1.0
    assert rt.gelu_tanh(np.array([1.0]))[0] == pytest.approx(0.8411919906082768, abs=1e-12)


def test_mlp_forward_matches_hand_computation():
    m = random_mlp(4, [3, 2], seed=3)
    rng = np.random.default_rng(4)
    X = rng.random((6, 4)).astype(np.float32)
    model = rt.load_member("mlp", rt.mlp_to_bytes(m))
    got = rt.mlp_forward(model, X)
    # explicit loops, no matrix products
    W = [w.astype(np.float64) for w in m["W"]]
    b = [v.astype(np.float64) for v in m["b"]]
    expect = []
    for r in range(X.shape[0]):
        h = [(float(X[r, i]) - float(m["x_mean"][i])) / float(m["x_std"][i]) for i in range(4)]
        for layer in range(3):
            out = [sum(h[i] * W[layer][i, j] for i in range(len(h))) + b[layer][j] for j in range(W[layer].shape[1])]
            if layer < 2:
                out = [0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3))) for v in out]
            h = out
        expect.append(h[0])
    np.testing.assert_allclose(got, np.array(expect), rtol=0, atol=1e-10)
    assert got.dtype == np.float64 and got.shape == (6,)
    # DataFrame input: columns BY NAME (shuffled + extra), same values; missing column -> KeyError
    df = pd.DataFrame(X, columns=m["columns"])
    df["extra"] = 1.0
    shuffled = df[list(reversed(df.columns))]
    np.testing.assert_array_equal(mlp_mod.predict(model, shuffled), got)
    with pytest.raises(KeyError):
        rt.mlp_forward(model, df.drop(columns=[m["columns"][1]]))
    with pytest.raises(ValueError):
        rt.mlp_forward(model, X[:, :3])
    assert rt.mlp_forward(model, X[:0]).shape == (0,)
    # predict_member dispatch == mlp_forward; zero std is guarded (never inf)
    np.testing.assert_array_equal(rt.predict_member("mlp", model, X), got)
    m0 = dict(model)
    m0["x_std"] = np.zeros(4)
    assert np.isfinite(rt.mlp_forward(m0, X)).all()


# --------------------------------------------------------------------------------------------------
# scripts/mlp_train.py on a tiny synthetic run (subprocess, interpreter with torch)
# --------------------------------------------------------------------------------------------------
N_FEAT = 12
ERAS = list(range(100, 140))


def _rng_ids(rng: np.random.Generator, n: int) -> list[str]:
    return [f"{v:016x}" for v in rng.choice(2**60, size=n, replace=False)]


def make_fake_data(data_dir: Path, seed: int = 0) -> list[str]:
    """futures_train.parquet: 40 eras x 40..80 rows, 12 int8 features (-1 = missing), everest + one aux target."""
    rng = np.random.default_rng(seed)
    cols = [f"feature_{i:03d}" for i in range(N_FEAT)]
    frames = []
    for e in ERAS:
        n = int(rng.integers(40, 81))
        x = rng.integers(0, 10, size=(n, N_FEAT)).astype(np.int8)
        x[rng.random((n, N_FEAT)) < 0.05] = -1
        lin = np.where(x == -1, 0.5, x / 9.0)
        signal = lin[:, 0] - lin[:, 1] + 0.5 * lin[:, 2] * lin[:, 3] + rng.normal(scale=0.3, size=n)
        q = pd.Series(signal).rank(pct=True).to_numpy()
        y = (np.ceil(q * 5) - 1) / 4.0
        aux = (np.ceil(pd.Series(signal + rng.normal(scale=0.5, size=n)).rank(pct=True).to_numpy() * 5) - 1) / 4.0
        aux[rng.random(n) < 0.2] = np.nan
        df = pd.DataFrame(x, columns=cols, index=pd.Index(_rng_ids(rng, n), name="id"))
        df.insert(0, "exped", f"exped_{e:04d}")
        df.insert(1, "data_type", "train")
        df["target_everest"] = y.astype(np.float32)
        df[AUX] = aux.astype(np.float32)
        frames.append(df)
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_parquet(data_dir / "futures_train.parquet")
    return cols


def make_fake_run(root: Path, *, hidden=(8, 4), seeds=(0,), universe: knobs.Universe | None = None) -> tuple[Path, list[str]]:
    data_dir = root / "data"
    cols = make_fake_data(data_dir)
    run_dir = root / "artifacts" / "run_mlp"
    run_dir.mkdir(parents=True)
    member = {"name": "mlp_everest", "family": "mlp", "target": "target_everest", "group": "everest", "arm": "mlp", "features": "all",
              "input": "linear", "params": {"hidden": list(hidden), "dropout": 0.3, "aux_targets": [AUX], "aux_weight": 0.5, "lr": 3e-3,
                                            "weight_decay": 1e-4, "batch_size": 128, "patience": 1, "device": "cpu"},
              "seeds": list(seeds), "n_estimators": None, "es_cap": 3}
    cfg = {"name": "fake", "run": {"seed": 0, "threads": 1, "min_era": 100, "data_dir": str(data_dir)}, "members": [member]}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "features.json").write_text(json.dumps({"feature_cols": cols, "missing_cols": cols[:2], "core_proxy_cols": cols[-3:]}))
    u = universe or knobs.Universe(era_start=100, rows_rule="weight50", time_decay=True)
    rng_ = lambda a, b: list(range(a, b + 1))  # noqa: E731
    folds = [
        {"name": "fold0", "test_eras": rng_(100, 109), "train_eras": rng_(120, 139), "inner_train_eras": rng_(120, 131),
         "inner_val_eras": rng_(134, 139), "count_sources": ["fold0"]},
        {"name": "fold1", "test_eras": rng_(130, 139), "train_eras": rng_(100, 119), "inner_train_eras": rng_(100, 111),
         "inner_val_eras": rng_(114, 119), "count_sources": ["fold1", "fold0"]},
    ]
    fwd = {"name": "forward", "test_eras": rng_(135, 139), "train_eras": rng_(100, 124), "inner_train_eras": rng_(100, 116),
           "inner_val_eras": rng_(119, 124), "count_sources": ["fold1", "forward"]}
    fe = {"universe": u.to_dict(), "universe_key": u.key(), "scoring_min_era": 100, "folds": folds, "forward": fwd,
          "full_train_eras": ERAS, "min_gap": 1}
    (run_dir / "folds_eras.json").write_text(json.dumps(fe))
    return run_dir, cols


def run_script(run_dir: Path, *extra: str, timeout: int = 400) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(REPO)
    env.setdefault("OMP_NUM_THREADS", "1")
    cmd = [GPU_PY, str(SCRIPT), "--run-id", run_dir.name, "--artifacts", str(run_dir.parent), "--threads", "1", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=str(REPO))


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict:
    if GPU_PY is None:
        pytest.skip("no interpreter with torch")
    root = tmp_path_factory.mktemp("mlp")
    run_dir, cols = make_fake_run(root)
    proc = run_script(run_dir, "--device", "cpu", "--epochs-cap", "2", "--seeds", "0")
    return {"root": root, "run_dir": run_dir, "cols": cols, "proc": proc}


@needs_torch_py
def test_mlp_train_artifacts_schema_and_parity(trained: dict):
    proc, run_dir, cols = trained["proc"], trained["run_dir"], trained["cols"]
    assert proc.returncode == 0, proc.stderr[-4000:]
    mdir = run_dir / "members" / "mlp_everest"
    spec = json.loads((mdir / "spec.json").read_text())
    assert spec["status"] == "done" and spec["family"] == "mlp" and spec["input"] == "linear" and spec["arm"] == "mlp"
    fe = json.loads((run_dir / "folds_eras.json").read_text())
    assert spec["universe"] == fe["universe_key"]
    for k in ("inner_iters", "frozen_n_iter_per_fold", "frozen_n_iter_forward", "frozen_n_iter", "count_sources", "fit_seconds",
              "max_fit_seconds", "member_seconds", "full_blobs", "feature_cols", "missing_cols", "universe", "seeds", "device",
              "torch_version", "parity_max_abs"):
        assert k in spec, k
    assert spec["feature_cols"] == cols and spec["seeds"] == [0] and spec["device"].startswith("cpu")
    assert 0.0 <= spec["parity_max_abs"] < 1e-4
    assert len(spec["inner_iters"]) == 2 and all(1 <= e <= 2 for e in spec["inner_iters"])
    assert spec["count_sources"]["forward"] == ["fold1", "forward"]
    assert spec["full_blobs"] == [{"file": "full_s0.bin", "seed": 0, "n_iter": spec["frozen_n_iter"]}]
    assert sorted(p.name for p in mdir.glob("*.bin")) == ["fold0.bin", "fold1.bin", "forward.bin", "full_s0.bin"]
    assert b"torch" not in json.dumps({k: v for k, v in spec.items() if k not in ("torch_version",)}).encode()
    oof = pd.read_parquet(run_dir / "oof_mlp_everest.parquet")
    assert oof.index.name == "id" and list(oof.columns) == ["era", "fold", "pred"]
    assert oof["era"].dtype == np.int32 and oof["fold"].dtype == np.int8 and oof["pred"].dtype == np.float32
    assert set(oof["fold"].unique()) == {0, 1, -1}
    sizes = oof.groupby("fold").size()
    data = pd.read_parquet(trained["root"] / "data" / "futures_train.parquet")
    era = data["exped"].str.slice(6).astype(int)
    assert sizes[0] == int(era.between(100, 109).sum()) and sizes[1] == int(era.between(130, 139).sum())
    assert sizes[-1] == int(era.between(135, 139).sum())
    assert np.isfinite(oof["pred"].to_numpy()).all() and oof["pred"].std() > 0
    # the train venv reads the blobs with the numpy forward (models.predict_member, as train.py's side metrics do)
    from stealth import data as sdata

    X = sdata.features_linear(data, cols, [])[cols]
    for name in ("fold0.bin", "forward.bin", "full_s0.bin"):
        pred = models.predict_member(("mlp", (mdir / name).read_bytes()), X)
        assert pred.shape == (len(X),) and np.isfinite(pred).all()
    # OOF fold0 rows == the fold0 blob's prediction on those rows (one seed -> no averaging)
    f0 = oof.loc[oof["fold"] == 0]
    p0 = models.predict_member(("mlp", (mdir / "fold0.bin").read_bytes()), X.loc[f0.index])
    np.testing.assert_allclose(p0, f0["pred"].to_numpy(dtype=np.float64), atol=2e-5)
    assert (mdir / "_work" / "meta.json").exists()


@needs_torch_py
def test_mlp_train_resume_is_a_noop_and_refusals(trained: dict, tmp_path: Path):
    run_dir = trained["run_dir"]
    assert trained["proc"].returncode == 0
    spec_before = (run_dir / "members" / "mlp_everest" / "spec.json").read_text()
    proc = run_script(run_dir, "--device", "cpu", "--epochs-cap", "2", "--seeds", "0", "--resume")
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "already done" in proc.stderr
    assert (run_dir / "members" / "mlp_everest" / "spec.json").read_text() == spec_before
    # no folds_eras.json -> exit 1 with the fix in the message
    empty = tmp_path / "artifacts" / "run_empty"
    empty.mkdir(parents=True)
    proc = run_script(empty)
    assert proc.returncode == 1 and "run stealth.train first" in proc.stderr


@needs_torch_py
def test_mlp_train_no_device_exit_4(tmp_path: Path):
    probe = subprocess.run([GPU_PY, "-c", "import torch; print(int(torch.cuda.is_available()))"], capture_output=True, text=True,
                           timeout=120)
    if probe.stdout.strip() == "1":
        pytest.skip("this interpreter has a GPU; the no-device path cannot be exercised")
    run_dir, _ = make_fake_run(tmp_path)
    proc = run_script(run_dir, "--device", "cuda")
    assert proc.returncode == 4, proc.stderr[-3000:]
    assert "NO USABLE DEVICE" in proc.stderr
    spec = json.loads((run_dir / "members" / "mlp_everest" / "spec.json").read_text())
    assert spec["status"] == "no_device"
