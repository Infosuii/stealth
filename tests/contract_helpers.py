"""Synthetic checker evidence for offline operations tests; real checker integration tests run the subprocess."""
from pathlib import Path

from stealth import pkl_check
from stealth.bundle import sha256_file


def valid_check_report(pkl: Path) -> dict:
    return {"passed": True, "sha256": sha256_file(pkl), "python": "3.11.18",
            "libs": dict(pin.split("==", 1) for pin in pkl_check.lane_pins()),
            "isolation": {"stealth_importable": False, "torch_importable": False},
            "problems": [], "n_eras": 51, "seconds_per_era_max": 0.1, "checker": "offline-test-fixture"}


def record_test_check(pkl: Path):
    return pkl_check.update_manifest(pkl, True, valid_check_report(pkl))
