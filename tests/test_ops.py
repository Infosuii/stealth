"""Offline tests for stealth.predict / stealth.diag / stealth.monitor with a fake SDK client (no network, no writes)."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stealth import bundle as bd
from stealth import diag, monitor, predict
from stealth import pkl_check
from tests.test_pkl_contract import make_fake_run
from tests.contract_helpers import record_test_check


class FakeClient:
    """Records every call; mimics the subset of EverestAPI the ops modules use."""

    def __init__(self, data_dir: Path, *, window: str = "open", exped: str = "exped_8055", sandbox_ready_after: int = 1):
        self.data_dir = data_dir
        self.window = window
        self.exped = exped
        self.calls: list[tuple[str, tuple, dict]] = []
        self.sandbox_ready_after = sandbox_ready_after
        self.status_polls = 0
        self.submission_entries: list[dict] = []
        self.upload_status_extra: dict = {}

    def _rec(self, name, *a, **k):
        self.calls.append((name, a, k))

    def get_current_round(self, tournament="futures"):
        self._rec("get_current_round", tournament)
        return {"round_number": 37, "exped": self.exped, "submission_window": self.window, "round_close": "2026-11-13T16:00:00Z"}

    def download_dataset(self, universe, split, output_path):
        self._rec("download_dataset", universe, split, output_path)
        shutil.copy(self.data_dir / f"futures_{split}.parquet", output_path)
        return output_path

    def download_benchmark(self, universe, split, output_path):
        self._rec("download_benchmark", universe, split, output_path)
        shutil.copy(self.data_dir / f"futures_{split}_benchmark_models.parquet", output_path)
        return output_path

    def validate_submission(self, predictions, exped=None):
        self._rec("validate_submission", len(predictions), exped)
        return {"valid": True, "issues": [], "n": len(predictions)}

    def submit_futures_predictions(self, model_id, predictions, exped=None):
        self._rec("submit_futures_predictions", model_id, len(predictions), exped)
        self.submission_entries = [{"accepted": True, "coverage_pct": 100.0, "n_predictions": len(predictions),
                                    "submitted_at": "now", "round": exped}]
        return {"submission_id": "sub1", "model_id": model_id, "exped": exped, "instrument_count": len(predictions),
                "late": False, "replaced": False}

    def get_submission_status(self, tournament="futures", round=None, model_id=None):
        self._rec("get_submission_status", tournament, round, model_id)
        return {"submissions": self.submission_entries}

    def upload_model(self, model_id, file_path, python_version=None):
        self._rec("upload_model", model_id, Path(file_path).name, python_version)
        return {"upload_id": "up-123", "sandbox_ready": 0}

    def get_upload_status(self, model_id):
        self._rec("get_upload_status", model_id)
        self.status_polls += 1
        ready = 1 if self.status_polls >= self.sandbox_ready_after else 0
        return {"sandbox_ready": ready, "error_message": None, "status": "pending", "upload_id": "up-123",
                "lane_active": True, **self.upload_status_extra}

    diag_statuses: list[str] = ["done"]  # successive get_diagnostics_result statuses (last one repeats)
    diag_error: Exception | None = None  # raised by submit_validation_diagnostics when set

    def submit_validation_diagnostics(self, model_id, predictions, tournament="futures", target="target_everest", wait=False,
                                      **kwargs):
        self._rec("submit_validation_diagnostics", model_id, Path(predictions).name, tournament, target, **kwargs)
        if self.diag_error is not None:
            raise self.diag_error
        return {"upload_id": "diag-1", "status": "pending"}

    def get_diagnostics_result(self, upload_id):
        self._rec("get_diagnostics_result", upload_id)
        statuses = list(self.diag_statuses)
        st = statuses.pop(0) if len(statuses) > 1 else statuses[0]
        self.diag_statuses = statuses
        if st != "done":
            return {"status": st, "queue_position": 1, "elapsed_seconds": 3, "estimate_seconds": 30}
        return {"status": "done", "metrics": {"fit20": 0.02, "unq": 0.01, "inov": 0.01, "sharpe_ratio": 0.5}}

    def get_scores(self, model_id, days=30):
        self._rec("get_scores", model_id, days)
        return {"scores": [{"exped": "exped_8050", "fit": 0.01, "unq": 0.0, "inov": 0.0, "payout": 0.01, "is_partial": True}]}


@pytest.fixture(scope="module")
def pkl(tmp_path_factory: pytest.TempPathFactory, live_frame: pd.DataFrame) -> Path:
    root = tmp_path_factory.mktemp("ops")
    run_dir = root / "artifacts" / "run_ops"
    make_fake_run(run_dir, live_frame)
    out = root / "models" / "stealth_ops.pkl"
    bd.build_and_write(run_dir, out)
    return out


@pytest.fixture(autouse=True)
def _no_approval_env(monkeypatch: pytest.MonkeyPatch):
    for var in (predict.ENV_SUBMIT, predict.ENV_UPLOAD, predict.ENV_DIAG, predict.ENV_ANY):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------------------------------
# predict
# --------------------------------------------------------------------------------------------------
def test_predict_live_offline(pkl: Path, data_dir: Path, tmp_path: Path):
    res = predict.predict_live(pkl, data_dir=data_dir, offline=True, out_dir=tmp_path / "preds")
    assert res["exped"] == "exped_8055" and res["n"] == 90
    assert res["path"].name == "exped_8055.parquet" and res["path"].exists()
    saved = pd.read_parquet(res["path"])
    assert list(saved.columns) == ["prediction"] and saved.index.name == "id" and len(saved) == 90
    assert len(res["predictions"]) == 90 and all(0 < v < 1 for v in res["predictions"].values())
    assert res["stats"]["n_unique"] == 90


def test_approval_gate(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(PermissionError):
        predict.approval_gate("x", predict.ENV_SUBMIT, False)
    with pytest.raises(PermissionError):
        predict.approval_gate("x", predict.ENV_SUBMIT, True)
    monkeypatch.setenv(predict.ENV_SUBMIT, "1")
    with pytest.raises(PermissionError):
        predict.approval_gate("x", predict.ENV_SUBMIT, False)
    predict.approval_gate("x", predict.ENV_SUBMIT, True)
    monkeypatch.delenv(predict.ENV_SUBMIT)
    monkeypatch.setenv(predict.ENV_ANY, "1")
    predict.approval_gate("x", predict.ENV_UPLOAD, True)


def test_predict_cli_refuses_without_gate(pkl: Path, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    base = ["--pkl", str(pkl), "--data-dir", str(data_dir), "--out-dir", str(tmp_path), "--model-id", "m"]
    assert predict.main([*base, "--offline"]) == predict.EXIT_OK
    assert predict.main([*base, "--offline", "--submit"]) == predict.EXIT_ERROR  # offline + submit
    fake = FakeClient(data_dir)
    monkeypatch.setattr(predict.sdata, "make_client", lambda *a, **k: fake)
    assert predict.main([*base, "--submit"]) == predict.EXIT_ERROR  # no flag, no env
    assert predict.main([*base, "--submit", "--i-have-approval"]) == predict.EXIT_ERROR  # env missing
    monkeypatch.setenv(predict.ENV_SUBMIT, "1")
    assert predict.main([*base, "--submit"]) == predict.EXIT_ERROR  # flag missing
    assert not any(c[0] == "submit_futures_predictions" for c in fake.calls)


def test_predict_cli_online_dry_run_and_window_closed(pkl: Path, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ddir = tmp_path / "data"
    ddir.mkdir()
    fake = FakeClient(data_dir, window="closed")
    monkeypatch.setattr(predict.sdata, "make_client", lambda *a, **k: fake)
    base = ["--pkl", str(pkl), "--data-dir", str(ddir), "--out-dir", str(tmp_path / "p")]
    assert predict.main(base) == predict.EXIT_WINDOW_CLOSED
    assert not (ddir / "futures_live.parquet").exists()
    fake.window = "open"
    assert predict.main(base) == predict.EXIT_OK
    names = [c[0] for c in fake.calls]
    assert names.count("download_dataset") == 1 and "download_benchmark" in names and "validate_submission" in names
    assert "submit_futures_predictions" not in names and "upload_model" not in names
    assert (ddir / "futures_live.parquet").exists() and (tmp_path / "p" / "exped_8055.parquet").exists()


def test_predict_submit_and_upload_with_fake_client(pkl: Path, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = FakeClient(data_dir, sandbox_ready_after=2)
    monkeypatch.setattr(predict.sdata, "make_client", lambda *a, **k: fake)
    monkeypatch.setattr(predict.time, "sleep", lambda s: None)
    monkeypatch.setenv(predict.ENV_SUBMIT, "1")
    monkeypatch.setenv(predict.ENV_UPLOAD, "1")
    ddir = tmp_path / "data"
    ddir.mkdir()
    base = ["--pkl", str(pkl), "--data-dir", str(ddir), "--out-dir", str(tmp_path / "p"), "--model-id", "slot-1", "--i-have-approval",
            "--active", str(tmp_path / "active.json")]
    # upload refuses while the manifest's pkl_check has not passed
    assert predict.main([*base, "--upload"]) == predict.EXIT_ERROR
    assert not any(c[0] == "upload_model" for c in fake.calls)
    record_test_check(pkl)
    rc = predict.main([*base, "--submit", "--upload"])
    assert rc == predict.EXIT_OK
    sub = [c for c in fake.calls if c[0] == "submit_futures_predictions"]
    assert len(sub) == 1 and sub[0][1] == ("slot-1", 90, "exped_8055")  # literal exped from the live file
    up = [c for c in fake.calls if c[0] == "upload_model"]
    assert len(up) == 1 and up[0][1] == ("slot-1", pkl.name, "3.11")
    assert fake.status_polls == 2
    active = json.loads((tmp_path / "active.json").read_text())
    assert active["model_id"] == "slot-1" and active["upload_id"] == "up-123" and active["sha256"] == bd.sha256_file(pkl)
    manifest = json.loads(bd.manifest_path(pkl).read_text())
    assert manifest["upload_id"] == "up-123"


def test_poll_upload_status_error(data_dir: Path, monkeypatch: pytest.MonkeyPatch):
    fake = FakeClient(data_dir)
    fake.upload_status_extra = {"error_message": "boom"}
    with pytest.raises(RuntimeError, match="boom"):
        predict.poll_upload_status(fake, "m", poll_seconds=0, timeout=1)
    fake2 = FakeClient(data_dir, sandbox_ready_after=10**6)
    monkeypatch.setattr(predict.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="did not reach"):
        predict.poll_upload_status(fake2, "m", poll_seconds=0, timeout=0.2)


def test_validate_predictions_errors(live_frame: pd.DataFrame):
    idx = live_frame.index
    good = pd.DataFrame({"prediction": np.linspace(0.01, 0.99, len(idx))}, index=idx)
    assert predict.validate_predictions(good, idx)["n"] == 90
    with pytest.raises(ValueError, match="not pandas.DataFrame"):
        predict.validate_predictions(good["prediction"], idx)
    with pytest.raises(ValueError, match="outside"):
        predict.validate_predictions(good.assign(prediction=lambda d: d["prediction"] * 2), idx)
    with pytest.raises(ValueError, match="index"):
        predict.validate_predictions(good.iloc[:-1], idx)
    with pytest.raises(ValueError, match="non-finite"):
        predict.validate_predictions(good.assign(prediction=np.nan), idx)
    # contract review 7: numeric strings and integers are not a float column (pkl_check refuses them too)
    with pytest.raises(ValueError, match="not float"):
        predict.validate_predictions(good.assign(prediction=good["prediction"].map(lambda v: f"{v:.6f}")), idx)
    with pytest.raises(ValueError, match="not float"):
        predict.validate_predictions(good.assign(prediction=1), idx)
    with pytest.raises(ValueError, match="0 columns"):
        predict.validate_predictions(good.iloc[:, :0], idx)
    assert predict.validate_predictions(good.astype("float32"), idx)["n"] == 90


# --------------------------------------------------------------------------------------------------
# diag
# --------------------------------------------------------------------------------------------------
def test_diag_first_eras(pkl: Path, data_dir: Path, tmp_path: Path):
    out = tmp_path / "diag" / "validation_preds.parquet"
    df = diag.make_validation_predictions(pkl, data_dir=data_dir, out=out, eras=3)
    assert out.exists() and df.index.name == "id" and list(df.columns) == ["prediction"]
    val = pd.read_parquet(data_dir / "futures_validation.parquet", columns=["exped"])
    first3 = sorted(val["exped"].unique())[:3]
    assert len(df) == int(val["exped"].isin(first3).sum()) > 200
    assert df.index.is_unique and ((df["prediction"] > 0) & (df["prediction"] < 1)).all()
    back = pd.read_parquet(out)
    assert back.index.name == "id" and len(back) == len(df)
    # per-era ranks are uniform
    one = df.loc[val.index[val["exped"] == first3[0]], "prediction"].sort_values().to_numpy()
    np.testing.assert_allclose(one, (np.arange(len(one)) + 0.5) / len(one))


def test_diag_cli_gate_and_upload(pkl: Path, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "vp.parquet"
    base = ["--pkl", str(pkl), "--data-dir", str(data_dir), "--out", str(out), "--eras", "2"]
    assert diag.main(base) == 0 and out.exists()
    assert diag.main([*base, "--upload", "--i-have-approval", "--model-id", "m"]) == 1  # --eras with --upload refused
    fake = FakeClient(data_dir)
    monkeypatch.setattr(diag.sdata, "make_client", lambda *a, **k: fake)
    assert diag.main(["--pkl", str(pkl), "--data-dir", str(data_dir), "--out", str(out), "--eras", "2"]) == 0
    assert diag.main(["--pkl", str(pkl), "--data-dir", str(data_dir), "--out", str(out), "--upload", "--model-id", "m"]) == 1
    assert not fake.calls
    res = diag.upload_diagnostics(fake, "m", out, wait=True)
    assert res["upload_id"] == "diag-1" and res["result"]["status"] == "done"
    assert (out.with_name(out.name + ".upload.json")).exists()
    rec = json.loads(out.with_name(out.name + ".upload.json").read_text())
    assert rec["panel"]["fit20"] == 0.02 and rec["panel"]["sharpe"] == 0.5 and rec["result_at"]


@pytest.fixture()
def small_val_dir(data_dir: Path, tmp_path: Path) -> Path:
    """A data dir whose validation split is the first 3 validation eras (so 'full coverage' is cheap)."""
    d = tmp_path / "small_data"
    d.mkdir()
    val = pd.read_parquet(data_dir / "futures_validation.parquet")
    keep = sorted(val["exped"].unique())[:3]
    val = val[val["exped"].isin(keep)]
    val.to_parquet(d / "futures_validation.parquet")
    bench = pd.read_parquet(data_dir / "futures_validation_benchmark_models.parquet")
    bench.reindex(val.index).to_parquet(d / "futures_validation_benchmark_models.parquet")
    return d


@pytest.fixture()
def checked_pkl(pkl: Path, tmp_path: Path) -> Path:
    """A copy of the fixture pickle whose manifest records a pkl_check pass bound to its bytes (sha256)."""
    d = tmp_path / "models"
    d.mkdir()
    out = d / "m01.pkl"
    shutil.copy(pkl, out)
    man = json.loads(pkl.with_name(f"{pkl.stem}.manifest.json").read_text())
    man.update({"pkl": out.name, "pkl_check": None})
    out.with_name("m01.manifest.json").write_text(json.dumps(man))
    record_test_check(out)
    return out


def test_diag_meta_file_status_and_preflight(checked_pkl: Path, small_val_dir: Path, tmp_path: Path,
                                             capsys: pytest.CaptureFixture[str]):
    out = tmp_path / "diag" / "m01_validation.parquet"
    diag.make_validation_predictions(checked_pkl, data_dir=small_val_dir, out=out, eras=2)
    meta = json.loads(diag.meta_path(out).read_text())
    assert meta["eras"] == 2 and meta["validation_eras"] == 3 and meta["eras_limit"] == 2
    ready, reason, _ = diag.file_status(out, checked_pkl)
    assert not ready and "partial coverage" in reason
    df = diag.make_validation_predictions(checked_pkl, data_dir=small_val_dir, out=out)
    meta = json.loads(diag.meta_path(out).read_text())
    assert meta["rows"] == meta["validation_rows"] == len(df) and meta["eras"] == 3 and meta["bench_used_eras"] == 3
    assert diag.file_status(out, checked_pkl)[0]
    ok, lines = diag.preflight(checked_pkl, out, "slot-x")
    assert ok, lines
    text = "\n".join(lines)
    assert "slot-x" in text and "manifest match" in text and "pkl_check passed" in text and "PREFLIGHT OK" in text
    assert diag.main(["--pkl", str(checked_pkl), "--out", str(out), "--preflight", "--model-id", "slot-x"]) == 0
    assert "PREFLIGHT OK" in capsys.readouterr().out
    # a different pickle -> the file is not reusable; a changed file -> not reusable
    other = checked_pkl.with_name("other.pkl")
    other.write_bytes(checked_pkl.read_bytes() + b"\0")
    assert "different pickle" in diag.file_status(out, other)[1]
    assert not diag.preflight(other, out, "slot-x")[0]  # no manifest next to other.pkl either
    # an earlier upload of the same file blocks the preflight
    diag.upload_json_path(out).write_text(json.dumps({"upload_id": "u-old", "uploaded_at": "x"}))
    ok, lines = diag.preflight(checked_pkl, out, "slot-x")
    assert not ok and any("ALREADY UPLOADED upload_id=u-old" in ln for ln in lines)


def test_diag_preflight_missing_model_id(checked_pkl: Path, small_val_dir: Path, tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EIQ_MODEL_ID", raising=False)
    monkeypatch.setattr(predict.sdata, "DOTENV_PATH", tmp_path / "no.env")
    out = tmp_path / "v.parquet"
    diag.make_validation_predictions(checked_pkl, data_dir=small_val_dir, out=out)
    ok, lines = diag.preflight(checked_pkl, out, None)
    assert not ok and any("MISSING" in ln and "HUMAN STEP" in ln for ln in lines)


def test_diag_upload_reuses_file_guards_budget_and_fetches(checked_pkl: Path, small_val_dir: Path, tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "diag" / "m01_validation.parquet"
    base = ["--pkl", str(checked_pkl), "--data-dir", str(small_val_dir), "--out", str(out)]
    assert diag.main(base) == 0
    built = json.loads(diag.meta_path(out).read_text())["created_at"]
    fake = FakeClient(small_val_dir)
    monkeypatch.setattr(diag.sdata, "make_client", lambda *a, **k: fake)
    assert diag.main([*base, "--upload", "--i-have-approval", "--model-id", "m"]) == 1  # env gate missing
    assert not fake.calls
    monkeypatch.setenv(predict.ENV_DIAG, "1")
    assert diag.main([*base, "--upload", "--i-have-approval", "--model-id", "m", "--wait"]) == 0
    assert json.loads(diag.meta_path(out).read_text())["created_at"] == built  # reused, not rebuilt
    subs = [c for c in fake.calls if c[0] == "submit_validation_diagnostics"]
    assert len(subs) == 1 and "model_pkl" not in subs[0][2]  # --attach-pkl is opt-in
    rec = json.loads(diag.upload_json_path(out).read_text())
    assert rec["upload_id"] == "diag-1" and rec["panel"]["fit20"] == 0.02 and rec["attached_pkl"] is False
    assert rec["built_from_pkl_sha256"] == diag.sha256_file(checked_pkl)
    # a second upload of the same file is refused (budget guard) and sends nothing
    assert diag.main([*base, "--upload", "--i-have-approval", "--model-id", "m", "--wait"]) == 1
    assert len([c for c in fake.calls if c[0] == "submit_validation_diagnostics"]) == 1
    # read-only re-poll needs no approval and no pickle
    monkeypatch.delenv(predict.ENV_DIAG)
    n_reads = len([c for c in fake.calls if c[0] == "get_diagnostics_result"])
    assert diag.main(["--out", str(out), "--fetch-result"]) == 0
    assert len([c for c in fake.calls if c[0] == "get_diagnostics_result"]) == n_reads + 1


def test_diag_upload_pending_then_fetch(checked_pkl: Path, small_val_dir: Path, tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "v.parquet"
    fake = FakeClient(small_val_dir)
    fake.diag_statuses = ["running", "running", "done"]
    monkeypatch.setattr(diag.sdata, "make_client", lambda *a, **k: fake)
    monkeypatch.setenv(predict.ENV_DIAG, "1")
    args = ["--pkl", str(checked_pkl), "--data-dir", str(small_val_dir), "--out", str(out), "--upload",
            "--i-have-approval", "--model-id", "m", "--wait", "--wait-timeout", "0"]
    assert diag.main(args) == 2  # uploaded, scoring not finished: never re-upload
    rec = json.loads(diag.upload_json_path(out).read_text())
    assert rec["upload_id"] == "diag-1" and rec["panel"]["status"] == "running" and rec["panel"]["fit20"] is None
    assert diag.main(["--out", str(out), "--fetch-result"]) == 2
    assert diag.main(["--out", str(out), "--fetch-result"]) == 0
    assert json.loads(diag.upload_json_path(out).read_text())["panel"]["fit20"] == 0.02
    assert diag.main(["--out", str(tmp_path / "never.parquet"), "--fetch-result"]) == 1


def test_diag_pkl_required_and_attach(checked_pkl: Path, small_val_dir: Path, tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    out = tmp_path / "v.parquet"
    fake = FakeClient(small_val_dir)
    fake.diag_error = RuntimeError("400: A model .pkl file is required for hackathon submissions.")
    monkeypatch.setattr(diag.sdata, "make_client", lambda *a, **k: fake)
    monkeypatch.setenv(predict.ENV_DIAG, "1")
    args = ["--pkl", str(checked_pkl), "--data-dir", str(small_val_dir), "--out", str(out), "--upload",
            "--i-have-approval", "--model-id", "m"]
    with caplog.at_level("ERROR"):
        assert diag.main(args) == 1
    assert "MODEL PKL REQUIRED" in caplog.text
    assert not diag.upload_json_path(out).exists()
    fake.diag_error = None
    assert diag.main([*args, "--attach-pkl"]) == 0
    kw = [c for c in fake.calls if c[0] == "submit_validation_diagnostics"][-1][2]
    assert Path(kw["model_pkl"]).name == checked_pkl.name
    assert kw["model_pkl_python_version"] == "3.11"
    rec = json.loads(diag.upload_json_path(out).read_text())
    assert rec["attached_pkl"] is True and rec["pkl_sha256"] == diag.sha256_file(checked_pkl)


def test_diag_panel_from_result_aliases():
    p = diag.panel_from_result({"status": "done", "corr20": 0.03, "aimc": 0.004, "fnc": 0.01, "sharpe_ratio": 0.6,
                                "max_drawdown": -0.2, "secret_field": "x"})
    assert p == {"status": "done", "fit20": 0.03, "unq": 0.004, "inov": 0.01, "sharpe": 0.6, "std_dev": None,
                 "max_drawdown": -0.2, "autocorrelation": None, "feature_exposure": None, "example_preds_corr": None}
    assert diag.panel_from_result(None)["fit20"] is None


# --------------------------------------------------------------------------------------------------
# monitor
# --------------------------------------------------------------------------------------------------
def test_monitor_offline_local_checks(pkl: Path, tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for f in pkl.parent.iterdir():
        shutil.copy(f, models_dir / f.name)
    mpath = models_dir / bd.manifest_path(pkl).name
    manifest = json.loads(mpath.read_text())
    manifest["pkl_check"] = None
    mpath.write_text(json.dumps(manifest))
    alert = tmp_path / "ALERT.md"
    rc = monitor.main(["--offline", "--models-dir", str(models_dir), "--active", str(tmp_path / "none.json"), "--alert", str(alert)])
    assert rc == 1 and alert.exists() and "pkl_check has not passed" in alert.read_text()
    record_test_check(models_dir / pkl.name)
    alert.unlink()
    rc = monitor.main(["--offline", "--models-dir", str(models_dir), "--active", str(tmp_path / "none.json"), "--alert", str(alert)])
    assert rc == 0 and not alert.exists()
    # active.json with a stale sha -> alert
    predict.write_active("m", "up-0", "0" * 64, pkl, tmp_path / "active.json")
    rc = monitor.main(["--offline", "--models-dir", str(models_dir), "--active", str(tmp_path / "active.json"), "--alert", str(alert)])
    assert rc == 1 and "active.json sha256" in alert.read_text()
    # corrupted pickle bytes -> alert
    (models_dir / pkl.name).write_bytes(b"garbage")
    _, problems = monitor.check_local(mpath, None)
    assert any("hashes to" in p for p in problems)
    # no manifest -> exit 2
    assert monitor.main(["--offline", "--models-dir", str(tmp_path / "empty"), "--active", str(tmp_path / "none.json")]) == 2


def test_monitor_refuses_a_pass_bound_to_other_bytes(pkl: Path, tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for f in pkl.parent.iterdir():
        shutil.copy(f, models_dir / f.name)
    record_test_check(models_dir / pkl.name)
    mpath = models_dir / bd.manifest_path(pkl).name
    manifest = json.loads(mpath.read_text())
    assert monitor.check_local(mpath, None)[1] == []
    manifest["pkl_check"]["sha256"] = "f" * 64  # a pass recorded for different bytes (or a legacy record without a hash)
    mpath.write_text(json.dumps(manifest))
    assert any("passed for other bytes" in p for p in monitor.check_local(mpath, None)[1])
    manifest["pkl_check"].pop("sha256")
    mpath.write_text(json.dumps(manifest))
    assert any("passed for other bytes" in p for p in monitor.check_local(mpath, None)[1])


def test_monitor_online_checks_with_fake_client(pkl: Path, data_dir: Path, tmp_path: Path):
    fake = FakeClient(data_dir)
    manifest = json.loads(bd.manifest_path(pkl).read_text())
    active = {"model_id": "m", "upload_id": "up-123", "sha256": manifest["sha256"]}
    assert monitor.check_active_upload(fake, "m", manifest, active) == []
    fake.upload_status_extra = {"upload_id": "up-999"}
    assert any("upload_id" in p for p in monitor.check_active_upload(fake, "m", manifest, active))
    fake.upload_status_extra = {"error_message": "smoke failed"}
    assert any("error_message" in p for p in monitor.check_active_upload(fake, "m", manifest, active))
    # submission: open window before grace -> no alert; after grace -> alert; accepted -> ok
    early = datetime(2026, 11, 13, 13, 10, tzinfo=timezone.utc)
    late = datetime(2026, 11, 13, 14, 0, tzinfo=timezone.utc)
    assert monitor.check_submission(fake, "m", now=early) == []
    assert any("NO accepted submission" in p for p in monitor.check_submission(fake, "m", now=late))
    fake.submit_futures_predictions("m", {"a": 0.5}, exped="exped_8055")
    assert monitor.check_submission(fake, "m", now=late) == []
    fake.submission_entries = [{"accepted": True, "coverage_pct": 50.0}]
    assert any("coverage" in p for p in monitor.check_submission(fake, "m", now=late))

    class NoRound(FakeClient):
        def get_current_round(self, tournament="futures"):
            raise RuntimeError("404 no active round")

    assert monitor.check_submission(NoRound(data_dir), "m", now=late) == []
    monitor.log_scores(fake, "m")
    path = monitor.write_alert(["a", "b"], path=tmp_path / "A.md", context={"x": 1})
    assert "USER MUST REVIEW" in path.read_text()


# --------------------------------------------------------------------------------------------------
# contract review follow-ups
# --------------------------------------------------------------------------------------------------
def test_predict_refuses_live_round_exped_mismatch(pkl: Path, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A stale live file (exped != the round's exped) must be a hard stop, not a warning."""
    fake = FakeClient(data_dir, exped="exped_9999")
    monkeypatch.setattr(predict.sdata, "make_client", lambda *a, **k: fake)
    ddir = tmp_path / "data"
    ddir.mkdir()
    with pytest.raises(RuntimeError, match="refusing to submit"):
        predict.predict_live(pkl, data_dir=ddir, offline=False, client=fake, out_dir=tmp_path / "p")
    assert not any(c[0] == "validate_submission" for c in fake.calls)
    rc = predict.main(["--pkl", str(pkl), "--data-dir", str(ddir), "--out-dir", str(tmp_path / "p")])
    assert rc == predict.EXIT_ERROR


def test_predict_no_active_round_exits_window_closed(pkl: Path, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from everestapi.client import EverestError

    class NoRound(FakeClient):
        def get_current_round(self, tournament="futures"):
            raise EverestError(404, {"detail": "No active round"})

    fake = NoRound(data_dir)
    monkeypatch.setattr(predict.sdata, "make_client", lambda *a, **k: fake)
    rc = predict.main(["--pkl", str(pkl), "--data-dir", str(tmp_path / "d"), "--out-dir", str(tmp_path / "p")])
    assert rc == predict.EXIT_WINDOW_CLOSED
    assert not any(c[0] == "download_dataset" for c in fake.calls)

    class Broken(FakeClient):
        def get_current_round(self, tournament="futures"):
            raise EverestError(500, "server error")

    monkeypatch.setattr(predict.sdata, "make_client", lambda *a, **k: Broken(data_dir))
    assert predict.main(["--pkl", str(pkl), "--data-dir", str(tmp_path / "d"), "--out-dir", str(tmp_path / "p")]) == predict.EXIT_ERROR


def test_upload_requires_manifest_and_python_311(pkl: Path, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = FakeClient(data_dir)
    monkeypatch.setattr(predict.time, "sleep", lambda s: None)
    # a pickle renamed/moved away from its manifest -> refused before any platform call
    loose = tmp_path / "loose.pkl"
    shutil.copy(pkl, loose)
    with pytest.raises(RuntimeError, match="no manifest"):
        predict.upload(fake, "m", loose, active_path=tmp_path / "a.json")
    assert not any(c[0] == "upload_model" for c in fake.calls)
    # manifest built under another python minor -> refused
    mdir = tmp_path / "models"
    mdir.mkdir()
    for f in pkl.parent.iterdir():
        shutil.copy(f, mdir / f.name)
    record_test_check(mdir / pkl.name)
    mpath = mdir / bd.manifest_path(pkl).name
    manifest = json.loads(mpath.read_text())
    assert manifest["python"].startswith("3.11.")
    manifest["python"] = "3.12.4"
    mpath.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="Python 3.12.4"):
        predict.upload(fake, "m", mdir / pkl.name, active_path=tmp_path / "a.json")
    assert not any(c[0] == "upload_model" for c in fake.calls)


def test_monitor_zero_coverage_is_not_full(data_dir: Path):
    fake = FakeClient(data_dir)
    late = datetime(2026, 11, 13, 14, 0, tzinfo=timezone.utc)
    fake.submission_entries = [{"accepted": True, "coverage_pct": 0}]
    assert any("coverage" in p for p in monitor.check_submission(fake, "m", now=late))
    fake.submission_entries = [{"accepted": True}]  # field not published -> accepted counts as full
    assert monitor.check_submission(fake, "m", now=late) == []
    assert monitor._full_coverage(None) and monitor._full_coverage(100) and not monitor._full_coverage(0) and not monitor._full_coverage("x")


def test_bundle_write_pickle_python_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(bd, "PKL_PYTHON", (3, 99))
    with pytest.raises(RuntimeError, match="Python 3.99"):
        bd.write_pickle({"feature_cols": [], "members": [], "weights": {}}, tmp_path / "x.pkl")
