"""Contract review finding 1: ONE artifact gate (``predict.check_upload_artifact``) in front of every upload path.

``predict --upload``, ``diag --upload`` (with or without ``--attach-pkl``) and ``diag.upload_diagnostics(model_pkl=...)``
must refuse a pickle whose manifest is missing or stale, whose pkl_check did not pass, whose pass was recorded for
other bytes (or carries no hash at all, the pre-fix shape), or that was built / checked under another Python minor --
and they must refuse BEFORE a client exists, so nothing is ever sent. Offline: fake clients only.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from stealth import bundle as bd
from stealth import diag, pkl_check, predict
from tests.test_ops import FakeClient
from tests.test_pkl_contract import make_fake_run


@pytest.fixture(scope="module")
def good_pkl(tmp_path_factory: pytest.TempPathFactory, live_frame: pd.DataFrame) -> Path:
    root = tmp_path_factory.mktemp("gates")
    run_dir = root / "artifacts" / "run_gates"
    make_fake_run(run_dir, live_frame)
    out = root / "models" / "gates.pkl"
    bd.build_and_write(run_dir, out)
    pkl_check.update_manifest(out, True)  # a pass bound to these bytes
    return out


@pytest.fixture()
def small_val_dir(data_dir: Path, tmp_path: Path) -> Path:
    d = tmp_path / "small_data"
    d.mkdir()
    val = pd.read_parquet(data_dir / "futures_validation.parquet")
    keep = sorted(val["exped"].unique())[:2]
    val = val[val["exped"].isin(keep)]
    val.to_parquet(d / "futures_validation.parquet")
    pd.read_parquet(data_dir / "futures_validation_benchmark_models.parquet").reindex(val.index).to_parquet(
        d / "futures_validation_benchmark_models.parquet")
    return d


def _copy(good_pkl: Path, dest: Path) -> tuple[Path, Path, dict]:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "cand.pkl"
    shutil.copy(good_pkl, out)
    man = json.loads(bd.manifest_path(good_pkl).read_text())
    man["pkl"] = out.name
    mpath = bd.manifest_path(out)
    mpath.write_text(json.dumps(man))
    return out, mpath, man


def _no_manifest(out: Path, mpath: Path, man: dict) -> None:
    mpath.unlink()


def _stale_manifest(out: Path, mpath: Path, man: dict) -> None:
    out.write_bytes(out.read_bytes() + b"\0")  # bytes changed after the manifest (and the check) were written


def _failed_check(out: Path, mpath: Path, man: dict) -> None:
    mpath.write_text(json.dumps({**man, "pkl_check": {**man["pkl_check"], "passed": False, "status": "failed"}}))


def _running_check(out: Path, mpath: Path, man: dict) -> None:
    mpath.write_text(json.dumps({**man, "pkl_check": {"passed": False, "status": "running", "sha256": man["sha256"]}}))


def _pass_for_other_bytes(out: Path, mpath: Path, man: dict) -> None:
    mpath.write_text(json.dumps({**man, "pkl_check": {**man["pkl_check"], "sha256": "f" * 64}}))


def _legacy_pass_without_hash(out: Path, mpath: Path, man: dict) -> None:
    mpath.write_text(json.dumps({**man, "pkl_check": {"passed": True, "at": "2026-11-13T20:31:31+00:00"}}))


def _built_under_312(out: Path, mpath: Path, man: dict) -> None:
    mpath.write_text(json.dumps({**man, "python": "3.12.4"}))


def _checked_under_312(out: Path, mpath: Path, man: dict) -> None:
    mpath.write_text(json.dumps({**man, "pkl_check": {**man["pkl_check"], "python": "3.12.4"}}))


CASES = {
    "no manifest": (_no_manifest, "no manifest"),
    "stale manifest sha": (_stale_manifest, "!= manifest"),
    "failed check": (_failed_check, "has not passed"),
    "running marker": (_running_check, "has not passed"),
    "pass for other bytes": (_pass_for_other_bytes, "passed for bytes"),
    "legacy pass without sha": (_legacy_pass_without_hash, "passed for bytes"),
    "built under 3.12": (_built_under_312, "Python 3.12.4"),
    "checked under 3.12": (_checked_under_312, "pkl_check ran under Python 3.12.4"),
}


class Recorder:
    """``make_client`` stand-in that records whether a client was ever created."""

    def __init__(self, data_dir: Path) -> None:
        self.created = 0
        self.fake = FakeClient(data_dir)

    def __call__(self, *a, **k) -> FakeClient:
        self.created += 1
        return self.fake


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch):
    for var in (predict.ENV_SUBMIT, predict.ENV_UPLOAD, predict.ENV_DIAG, predict.ENV_ANY):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(predict.time, "sleep", lambda s: None)
    monkeypatch.setattr(diag.time, "sleep", lambda s: None)


def test_gate_accepts_a_byte_bound_pass(good_pkl: Path):
    gate = predict.check_upload_artifact(good_pkl)
    assert gate["sha256"] == bd.sha256_file(good_pkl) and gate["manifest"]["pkl_check"]["passed"] is True


@pytest.mark.parametrize("case", list(CASES))
def test_gate_refuses(case: str, good_pkl: Path, tmp_path: Path):
    out, mpath, man = _copy(good_pkl, tmp_path / "m")
    mutate, msg = CASES[case]
    mutate(out, mpath, man)
    with pytest.raises(RuntimeError, match=msg.replace("(", r"\(").replace(".", r"\.")):
        predict.check_upload_artifact(out)


@pytest.mark.parametrize("case", list(CASES))
def test_predict_upload_refused_before_any_client(case: str, good_pkl: Path, data_dir: Path, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch):
    out, mpath, man = _copy(good_pkl, tmp_path / "m")
    CASES[case][0](out, mpath, man)
    rec = Recorder(data_dir)
    monkeypatch.setattr(predict.sdata, "make_client", rec)
    monkeypatch.setenv(predict.ENV_UPLOAD, "1")
    rc = predict.main(["--pkl", str(out), "--data-dir", str(tmp_path / "d"), "--out-dir", str(tmp_path / "p"), "--model-id", "slot",
                       "--upload", "--i-have-approval", "--active", str(tmp_path / "active.json")])
    assert rc == predict.EXIT_ERROR
    assert rec.created == 0 and rec.fake.calls == []
    assert not (tmp_path / "active.json").exists()


@pytest.mark.parametrize("attach", [False, True])
@pytest.mark.parametrize("case", list(CASES))
def test_diag_upload_refused_before_any_client(case: str, attach: bool, good_pkl: Path, small_val_dir: Path, tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch):
    out, mpath, man = _copy(good_pkl, tmp_path / "m")
    CASES[case][0](out, mpath, man)
    rec = Recorder(small_val_dir)
    monkeypatch.setattr(diag.sdata, "make_client", rec)
    monkeypatch.setenv(predict.ENV_DIAG, "1")
    vfile = tmp_path / "diag" / "v.parquet"
    args = ["--pkl", str(out), "--data-dir", str(small_val_dir), "--out", str(vfile), "--upload", "--i-have-approval",
            "--model-id", "slot", "--wait"] + (["--attach-pkl"] if attach else [])
    assert diag.main(args) == diag.EXIT_ERROR
    assert rec.created == 0 and rec.fake.calls == []
    assert not vfile.exists() and not diag.upload_json_path(vfile).exists()  # gated before the file is even built


def test_upload_diagnostics_gates_an_attached_pickle(good_pkl: Path, small_val_dir: Path, tmp_path: Path):
    """The library entry point (not only the CLI) refuses an attached pickle that fails the gate: zero client calls."""
    vfile = tmp_path / "v.parquet"
    diag.make_validation_predictions(good_pkl, data_dir=small_val_dir, out=vfile)
    out, mpath, man = _copy(good_pkl, tmp_path / "m")
    _pass_for_other_bytes(out, mpath, man)
    fake = FakeClient(small_val_dir)
    with pytest.raises(RuntimeError, match="passed for bytes"):
        diag.upload_diagnostics(fake, "slot", vfile, model_pkl=out)
    assert fake.calls == []
    res = diag.upload_diagnostics(fake, "slot", vfile, model_pkl=good_pkl)  # positive control
    assert res["upload_id"] == "diag-1"
    kw = [c for c in fake.calls if c[0] == "submit_validation_diagnostics"][0][2]
    assert kw == {"model_pkl": str(good_pkl), "model_pkl_python_version": "3.11"}


def test_positive_controls_upload_through_the_gate(good_pkl: Path, data_dir: Path, small_val_dir: Path, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch):
    out, _, _ = _copy(good_pkl, tmp_path / "m")
    pkl_check.update_manifest(out, True)
    rec = Recorder(data_dir)
    monkeypatch.setattr(predict.sdata, "make_client", rec)
    monkeypatch.setenv(predict.ENV_UPLOAD, "1")
    ddir = tmp_path / "d"
    ddir.mkdir()
    assert predict.main(["--pkl", str(out), "--data-dir", str(ddir), "--out-dir", str(tmp_path / "p"), "--model-id", "slot",
                         "--upload", "--i-have-approval", "--active", str(tmp_path / "active.json")]) == predict.EXIT_OK
    assert [c for c in rec.fake.calls if c[0] == "upload_model"][0][1] == ("slot", out.name, "3.11")
    drec = Recorder(small_val_dir)
    monkeypatch.setattr(diag.sdata, "make_client", drec)
    monkeypatch.setenv(predict.ENV_DIAG, "1")
    vfile = tmp_path / "diag" / "v.parquet"
    assert diag.main(["--pkl", str(out), "--data-dir", str(small_val_dir), "--out", str(vfile), "--upload", "--i-have-approval",
                      "--model-id", "slot", "--wait", "--attach-pkl"]) == diag.EXIT_OK
    assert len([c for c in drec.fake.calls if c[0] == "submit_validation_diagnostics"]) == 1


def test_preflight_reports_the_gate(good_pkl: Path, small_val_dir: Path, tmp_path: Path):
    out, mpath, man = _copy(good_pkl, tmp_path / "m")
    vfile = tmp_path / "v.parquet"
    diag.make_validation_predictions(out, data_dir=small_val_dir, out=vfile)
    ok, lines = diag.preflight(out, vfile, "slot")
    assert ok and any("artifact gate OK" in ln for ln in lines)
    _legacy_pass_without_hash(out, mpath, man)
    ok, lines = diag.preflight(out, vfile, "slot")
    assert not ok and any("artifact gate REFUSES" in ln for ln in lines)
    assert any("NOT passed for these bytes" in ln for ln in lines)
