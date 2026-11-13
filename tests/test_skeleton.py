"""Skeleton tests: package imports, config schema, fixture sanity. No model training here."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

MODULES = [
    "stealth", "stealth.data", "stealth.cv", "stealth.metrics", "stealth.features", "stealth.surrogate",
    "stealth.models", "stealth.models.lgbm", "stealth.models.xgb", "stealth.models.cat", "stealth.models.linear",
    "stealth.search", "stealth.knobs", "stealth.models.mlp", "stealth.train", "stealth.blend", "stealth._runtime", "stealth.bundle",
    "stealth.pkl_check", "stealth.predict", "stealth.diag", "stealth.monitor", "stealth.report", "stealth.selfcheck",
]
REQUIRED_TOP_KEYS = {"name", "run", "cv", "members", "search", "blend", "bootstrap", "gates", "timing"}
MEMBER_KEYS = {"name", "family", "target", "group", "features", "input", "params", "n_estimators", "es_cap"}
OPTIONAL_MEMBER_KEYS = {"arm", "seeds"}  # INTERFACES section 11


@pytest.mark.parametrize("mod", MODULES)
def test_importable(mod: str) -> None:
    importlib.import_module(mod)


def test_version() -> None:
    import stealth

    assert isinstance(stealth.VERSION, str) and stealth.VERSION.count(".") == 2


def test_no_torch_in_train_venv() -> None:
    """The pickle lane rejects torch; the train venv must not even have it importable."""
    assert importlib.util.find_spec("torch") is None


@pytest.mark.parametrize("name", ["smoke", "test", "full", "minifull"])
def test_config_schema(repo_root: Path, name: str) -> None:
    cfg = yaml.safe_load((repo_root / "configs" / f"{name}.yaml").read_text())
    assert REQUIRED_TOP_KEYS <= set(cfg), set(cfg) ^ REQUIRED_TOP_KEYS
    assert cfg["cv"]["purge"] >= 20 and cfg["cv"]["embargo"] >= 20
    assert 0.0 <= cfg["blend"]["p_b"] <= 0.75 and 0.0 <= cfg["blend"]["p_c"] <= 0.5
    names = [m["name"] for m in cfg["members"]]
    assert len(names) == len(set(names)), "member names must be unique"
    for m in cfg["members"]:
        assert MEMBER_KEYS <= set(m) <= MEMBER_KEYS | OPTIONAL_MEMBER_KEYS, (m["name"], set(m) ^ MEMBER_KEYS)
        assert m["family"] in {"lgbm", "xgb", "cat", "ridge", "mlp"}
        assert m["group"] in {"everest", "aux"}
        assert m["input"] in {"nan", "linear"}
        assert m["target"].startswith("target_")
        if m["family"] == "mlp":  # external member (scripts/mlp_train.py)
            assert m["input"] == "linear" and m["seeds"] and m["n_estimators"] is None
    if name in ("smoke", "test"):  # M1 configs are unchanged: no arms, no knobs, no external members
        assert all(set(m) == MEMBER_KEYS for m in cfg["members"]) and "knobs" not in cfg
    for key, spec in cfg["search"].items():
        assert key in names
        grid_size = 1
        for values in spec["grid"].values():
            grid_size *= len(values)
        assert spec["n_trials"] <= grid_size and spec["top_k"] <= spec["n_trials"]
        assert spec["n_trials"] <= cfg["gates"].get("max_search_trials", 25) <= 25


def test_train_slice_fixture(train_slice) -> None:
    assert train_slice.index.name == "id"
    assert train_slice["era"].nunique() == 300
    assert train_slice["era"].max() == 6524
    assert str(train_slice["feature_Adfel"].dtype) == "int8"
    assert train_slice["target_everest"].isna().sum() == 0


def test_live_fixture(live_frame, live_bench) -> None:
    assert len(live_frame) == 90 and live_frame["exped"].iloc[0] == "exped_8055"
    assert live_bench.index.equals(live_frame.index) or set(live_bench.index) == set(live_frame.index)
