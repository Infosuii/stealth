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

    def submit_validation_diagnostics(self, model_id, predictions, tournament="futures", target="target_everest", wait=False):
        self._rec("submit_validation_diagnostics", model_id, Path(predictions).name, tournament, target)
        return {"upload_id": "diag-1", "status": "pending"}

    def get_diagnostics_result(self, upload_id):
        self._rec("get_diagnostics_result", upload_id)
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
    pkl_check.update_manifest(pkl, True)
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
    pkl_check.update_manifest(models_dir / pkl.name, True)
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
