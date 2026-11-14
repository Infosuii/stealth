"""No torch anywhere in the package (docs/INTERFACES.md section 18): the lane has no torch, and the sandbox's
CPU torch would hide a stray import. Only the explicitly listed external GPU training scripts may import it."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "stealth"
_IMPORT_RE = re.compile(r"^\s*(import\s+torch\b|from\s+torch(\.|\s)|.*\bimportlib\.import_module\(\s*['\"]torch)", re.M)

CODE = r"""
import importlib, importlib.abc, json, pkgutil, sys

attempts = []

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            attempts.append(name)
            raise ImportError("torch is blocked in this test")
        return None

sys.meta_path.insert(0, Block())
import stealth
names = ["stealth"] + [m.name for m in pkgutil.walk_packages(stealth.__path__, "stealth.")]
failed = {}
for n in names:
    try:
        importlib.import_module(n)
    except Exception as exc:
        failed[n] = f"{type(exc).__name__}: {exc}"
print(json.dumps({"names": names, "failed": failed, "attempts": sorted(set(attempts)),
                  "torch_loaded": sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))}))
"""


def test_importing_every_stealth_module_never_imports_torch():
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONPATH"] = str(REPO)
    proc = subprocess.run([sys.executable, "-c", CODE], cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    assert {"stealth._runtime", "stealth.bundle", "stealth.models.mlp", "stealth.train", "stealth.pkl_check"} <= set(res["names"])
    assert res["failed"] == {}, res["failed"]
    assert res["torch_loaded"] == []
    # nothing in the package (or what it imports) even TRIED to import torch
    assert res["attempts"] == [], res["attempts"]


def test_no_torch_import_statement_in_the_package():
    offenders = []
    for path in sorted(PKG.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "import torch" in text or _IMPORT_RE.search(text):
            offenders.append(str(path.relative_to(REPO)))
    assert offenders == [], offenders


def test_only_external_gpu_training_scripts_import_torch():
    allowed = {REPO / "scripts" / name for name in ("mlp_train.py", "research_gpu.py", "campaign_gpu.py", "denoise_gpu.py")}
    offenders = [str(p.relative_to(REPO)) for p in sorted((REPO / "scripts").rglob("*.py"))
                 if p not in allowed and ("import torch" in p.read_text(encoding="utf-8") or _IMPORT_RE.search(p.read_text(encoding="utf-8")))]
    assert offenders == [], offenders
    # the script imports torch lazily (inside a function), so a broken GPU venv maps to exit 4, not a traceback
    src = (REPO / "scripts" / "mlp_train.py").read_text(encoding="utf-8")
    assert not re.search(r"^import torch|^from torch", src, re.M)
    assert "        import torch" in src
