#!/usr/bin/env python3
"""Train the EXTERNAL MLP member in the optional GPU venv; export numpy npz blobs (docs/INTERFACES.md 14, 18, 19).

This script lives OUTSIDE the ``stealth`` package and is the ONLY file in the repo that imports torch. It runs
under ``.venv-gpu`` (``scripts/setup_gpu.sh``) -- never the train venv -- and imports only torch, numpy, pandas,
yaml, ``stealth.data``, ``stealth.knobs`` and ``stealth._runtime`` (npz format + the numpy forward used for the
parity check, i.e. the SAME forward pass the pickled bundle runs in the lane).

    $GPU_PY scripts/mlp_train.py --run-id R [--artifacts DIR] [--member mlp_everest] [--device auto|cuda|cpu]
                                 [--seeds 0 1 ...] [--epochs-cap N] [--threads N] [--data-dir D] [--resume]

Inputs (written by ``python -m stealth.train``): ``artifacts/R/config.yaml``, ``folds_eras.json`` (exact era lists,
count sources, the decided training universe) and ``features.json`` (the 178 base columns). Refuses without
``folds_eras.json``.

Protocol (identical to the tree members): for every seed and fold (CV folds + forward block) an inner fit on
``inner_train_eras`` early-stopped on ``inner_val_eras`` (mean per-era Pearson of the main head, ``patience``,
cap ``es_cap``) gives a best epoch; the frozen epoch count of a fold is the median over its ``count_sources`` of
the seed-median best epochs; each seed is refit on the fold's ``train_eras`` for that count and predicts the
fold's test eras; OOF = mean over seeds. Full refit per seed on ``full_train_eras`` with the median CV count.
Training rows = fold eras & target not NaN & ``Universe.train_mask``; sample weights = ``Universe.weights`` of
the rows of that fit (exactly train.py's ``_fit_rows`` / ``RunData.weights``). Test rows are never masked.
Loss = weighted MSE(main) + ``aux_weight`` x masked MSE(aux heads; NaN aux targets masked). Input = the linear
frame's 178 base columns, standardised with each fit's training mean/std (stored in the npz).

Outputs: ``oof_<m>.parquet`` (index id; era int32, fold int8 (-1 = forward), pred float32) and
``members/<m>/{fold<k>.bin, forward.bin, full_s<j>.bin, spec.json}`` (``status: done`` written LAST). Every
exported blob is checked against the torch eval forward on <= 2000 rows: ``max_abs > 1e-4`` -> exit 1.
Resumable per (seed, fold) under ``members/<m>/_work/`` with ``--resume``. Exit 0 ok, 1 error, 4 no usable device
(torch not importable, or ``--device cuda`` without a working GPU).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS (must precede CUDA init)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # works without `pip install -e .` (sandbox: PYTHONPATH=<repo>)
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from stealth import _runtime, data, knobs  # noqa: E402

log = logging.getLogger("mlp_train")

EXIT_OK, EXIT_ERROR, EXIT_NO_DEVICE = 0, 1, 4
PARITY_TOL = 1e-4
PARITY_ROWS = 2000
PRED_BATCH = 65536
TARGET_SCALING = "standardised_v1"  # loss on (y - mean) / sd of the fit rows; bump to invalidate _work/ caches
DEFAULT_PARAMS: dict[str, Any] = {"hidden": [256, 64], "dropout": 0.3, "aux_targets": [], "aux_weight": 0.5, "lr": 1.0e-3,
                                  "weight_decay": 1.0e-4, "batch_size": 4096, "patience": 5, "device": "auto"}


class NoDevice(RuntimeError):
    """No usable compute device (exit 4: the handoff skips the member)."""


# --------------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, obj: Any) -> None:
    """Atomic JSON write (tmp + replace) so a reader never sees half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n")
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def median_round(values: list[float]) -> int:
    """``int(round(median))`` -- the tree members' rule (``train.run_protocol``), floored at 1 epoch."""
    return max(1, int(round(float(np.median(np.asarray(values, dtype=np.float64))))))


def per_era_pearson(pred: np.ndarray, y: np.ndarray, era: np.ndarray) -> float:
    """Mean over eras of Pearson(pred, y); eras with < 2 rows or zero variance count as 0."""
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    era = np.asarray(era)
    if pred.size == 0:
        return 0.0
    order = np.argsort(era, kind="stable")
    e, p, t = era[order], pred[order], y[order]
    starts = np.flatnonzero(np.r_[True, e[1:] != e[:-1]])
    n = np.diff(np.r_[starts, e.size]).astype(np.float64)
    sp, st = np.add.reduceat(p, starts), np.add.reduceat(t, starts)
    spp, stt, spt = np.add.reduceat(p * p, starts), np.add.reduceat(t * t, starts), np.add.reduceat(p * t, starts)
    cov = spt - sp * st / n
    vp = spp - sp * sp / n
    vt = stt - st * st / n
    ok = (n >= 2) & (vp > 1e-12 * np.maximum(1.0, spp)) & (vt > 1e-12 * np.maximum(1.0, stt))
    corr = np.zeros_like(cov)
    corr[ok] = cov[ok] / np.sqrt(vp[ok] * vt[ok])
    return float(corr.mean())


def sha_of(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=_json_default).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------------------
def load_run(run_dir: Path, member: str | None) -> tuple[dict, dict, dict, dict]:
    """``(cfg, member spec, folds_eras doc, features doc)``; ``FileNotFoundError`` / ``ValueError`` with the fix."""
    fe_path = run_dir / "folds_eras.json"
    if not fe_path.exists():
        raise FileNotFoundError(f"{fe_path} missing: run stealth.train first (python -m stealth.train --config C --run-id "
                                f"{run_dir.name})")
    for name in ("config.yaml", "features.json"):
        if not (run_dir / name).exists():
            raise FileNotFoundError(f"{run_dir / name} missing: run stealth.train first")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    mlps = [m for m in cfg.get("members") or [] if m.get("family") == "mlp"]
    if member is None:
        if not mlps:
            raise ValueError(f"{run_dir}/config.yaml has no family: mlp member")
        mspec = mlps[0]
    else:
        found = [m for m in cfg.get("members") or [] if m.get("name") == member]
        if not found:
            raise ValueError(f"member {member!r} not in {run_dir}/config.yaml")
        mspec = found[0]
        if mspec.get("family") != "mlp":
            raise ValueError(f"member {member!r} has family {mspec.get('family')!r}, not mlp")
    if mspec.get("input", "linear") != "linear":
        raise ValueError(f"member {mspec['name']}: family mlp needs input: linear")
    fe = json.loads(fe_path.read_text())
    feats = json.loads((run_dir / "features.json").read_text())
    if not feats.get("feature_cols"):
        raise ValueError(f"{run_dir}/features.json has no feature_cols")
    return cfg, mspec, fe, feats


def resolve_params(mspec: dict) -> dict:
    p = {**DEFAULT_PARAMS, **(mspec.get("params") or {})}
    p["hidden"] = [int(h) for h in p["hidden"]]
    p["aux_targets"] = [str(t) for t in p.get("aux_targets") or []]
    for k in ("dropout", "aux_weight", "lr", "weight_decay"):
        p[k] = float(p[k])
    p["batch_size"], p["patience"] = int(p["batch_size"]), int(p["patience"])
    return p


class TrainData:
    """The training pool as float32 linear-frame base columns (value/9, -1 -> 0.5) + targets + era bookkeeping."""

    def __init__(self, data_dir: Path, feature_cols: list[str], target: str, aux_targets: list[str], pool_eras: np.ndarray,
                 universe: knobs.Universe) -> None:
        cols = list(dict.fromkeys([*feature_cols, target, *aux_targets]))
        df = data.load_split("train", data_dir, columns=cols)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"train split lacks column(s) {missing}")
        counts = data.era_counts(df)  # per-era n_rows over the WHOLE split (== train.py's pool counts)
        df = df.loc[np.isin(df["era"].to_numpy(), pool_eras)]
        if df.empty:
            raise ValueError("no training rows in the folds' eras")
        self.ids = df.index
        self.era = df["era"].to_numpy(dtype=np.int32)
        self.era_n_rows = counts.reindex(self.era).to_numpy(dtype=np.int64)
        self.X = data.features_linear(df, list(feature_cols), [])[list(feature_cols)].to_numpy(dtype=np.float32)
        self.y = df[target].to_numpy(dtype=np.float32)
        self.Yaux = df[aux_targets].to_numpy(dtype=np.float32) if aux_targets else np.zeros((len(df), 0), np.float32)
        self.universe = universe
        self.columns = list(feature_cols)
        self._train_mask = universe.train_mask(self.era, self.era_n_rows)
        log.info("data: %d rows, %d eras (%d..%d), %d features, aux targets %s, universe %s (%d training-eligible rows)",
                 len(df), len(np.unique(self.era)), self.era.min(), self.era.max(), self.X.shape[1], aux_targets,
                 universe.key(), int(self._train_mask.sum()))

    def rows(self, eras: list[int] | np.ndarray) -> np.ndarray:
        return np.isin(self.era, np.asarray(eras, dtype=np.int32))

    def fit_mask(self, eras: list[int] | np.ndarray, label: str) -> np.ndarray:
        """train.py ``_fit_rows``: rows of ``eras`` with a non-NaN main target, restricted by the universe mask."""
        m = self.rows(eras) & np.isfinite(self.y) & self._train_mask
        if not m.any():
            raise ValueError(f"{label}: universe {self.universe.key()} leaves no training rows")
        return m

    def weights(self, mask: np.ndarray) -> np.ndarray | None:
        return self.universe.weights(self.era[mask], self.era_n_rows[mask])


# --------------------------------------------------------------------------------------------------
# torch side (imported lazily so a broken GPU venv maps to exit 4, not a traceback)
# --------------------------------------------------------------------------------------------------
def import_torch() -> Any:
    try:
        import torch  # the ONLY torch import in the repo (never inside the stealth package)
    except Exception as exc:  # noqa: BLE001
        raise NoDevice(f"torch is not importable in {sys.executable}: {type(exc).__name__}: {exc}") from exc
    return torch


def pick_device(torch: Any, requested: str) -> tuple[str, str]:
    """``(device, description)``. auto -> cuda when available AND a small matmul runs, else cpu; cuda -> NoDevice if not."""
    requested = (requested or "auto").lower()
    if requested not in ("auto", "cuda", "cpu"):
        raise ValueError(f"--device must be auto|cuda|cpu (got {requested!r})")
    if requested == "cpu":
        return "cpu", "cpu (requested)"
    reason = "torch.cuda.is_available() is False"
    if torch.cuda.is_available():
        try:
            a = torch.randn(256, 256, device="cuda")
            float((a @ a).sum().item())
            torch.cuda.synchronize()
            cap = torch.cuda.get_device_capability(0)
            return "cuda", f"cuda:0 {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})"
        except Exception as exc:  # noqa: BLE001 - e.g. "no kernel image is available" on an unsupported arch
            reason = f"CUDA smoke failed: {type(exc).__name__}: {exc}"
    if requested == "cuda":
        raise NoDevice(f"--device cuda requested but unusable ({reason})")
    log.warning("no usable GPU (%s); training on CPU -- expect it to be slow on the full config", reason)
    return "cpu", f"cpu (auto; {reason})"


def configure_torch(torch: Any, threads: int) -> None:
    torch.set_num_threads(max(1, int(threads)))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False  # parity with the float64 numpy forward
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def make_net(torch: Any, n_in: int, hidden: list[int], dropout: float, n_aux: int) -> Any:
    """in -> hidden... -> 1 (main head), aux head (n_aux outputs) off the last hidden layer; GELU(tanh); dropout after
    every hidden activation."""
    nn = torch.nn

    class MLPNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dims = [n_in, *hidden]
            self.hidden = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden))])
            self.act = nn.GELU(approximate="tanh")
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(dims[-1], 1)
            self.aux = nn.Linear(dims[-1], n_aux) if n_aux > 0 else None

        def forward(self, x: Any) -> tuple[Any, Any]:
            h = x
            for lin in self.hidden:
                h = self.drop(self.act(lin(h)))
            return self.head(h).squeeze(-1), (self.aux(h) if self.aux is not None else None)

    return MLPNet()


class Fitter:
    """One fit = standardise on the fit rows -> fresh seeded net -> AdamW epochs (optionally early-stopped)."""

    def __init__(self, torch: Any, device: str, params: dict, columns: list[str]) -> None:
        self.torch, self.device, self.p, self.columns = torch, device, params, list(columns)

    def _tensor(self, a: np.ndarray) -> Any:
        return self.torch.from_numpy(np.ascontiguousarray(a)).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None, Yaux: np.ndarray, *, seed: int, epochs: int,
            val: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None, patience: int | None = None) -> dict:
        """Returns ``{"net", "x_mean", "x_std", "best_epoch", "curve", "epochs_run", "seconds"}``. With ``val`` =
        (X_val, y_val, era_val) the run stops after ``patience`` epochs without a better per-era Pearson; ``best_epoch``
        is 1-based. Without ``val`` exactly ``epochs`` epochs are run."""
        torch, p = self.torch, self.p
        t0 = time.perf_counter()
        x64 = X.astype(np.float64)
        mean = x64.mean(axis=0).astype(np.float32)
        std = x64.std(axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
        del x64
        Xs = self._tensor(((X - mean) / std).astype(np.float32))
        # Targets are rank-scaled to [0, 1] (mean 0.5, sd ~0.22): fitting them raw spends the first epochs learning the
        # intercept (AdamW moves the head bias ~lr per step), so inner early stopping picked noise at epochs 1-3. The
        # prediction is only ever used through per-era ranks, so standardising the fit rows' targets changes nothing
        # downstream except that the loss now measures the signal (TARGET_SCALING in the signature).
        y_c = (y.astype(np.float64) - np.nanmean(y)) / max(float(np.nanstd(y)), 1e-6)
        yt = self._tensor(y_c.astype(np.float32))
        w_arr = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64) / float(np.mean(w))
        wt = self._tensor(w_arr.astype(np.float32))
        n_aux = Yaux.shape[1]
        if n_aux:
            amask = np.isfinite(Yaux)
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)  # an all-NaN aux column -> mean/sd fallback
                a_mu = np.nan_to_num(np.nanmean(Yaux.astype(np.float64), axis=0), nan=0.0)
                a_sd = np.nan_to_num(np.nanstd(Yaux.astype(np.float64), axis=0), nan=1.0)
            a_sd = np.where(a_sd > 1e-6, a_sd, 1.0)
            At = self._tensor(np.where(amask, (Yaux - a_mu) / a_sd, 0.0).astype(np.float32))
            Mt = self._tensor(amask.astype(np.float32))
        torch.manual_seed(int(seed))
        np.random.seed(int(seed) % (2**32))
        net = make_net(torch, X.shape[1], p["hidden"], p["dropout"], n_aux).to(self.device)
        opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        n, bs = len(y), max(1, min(int(p["batch_size"]), len(y)))
        curve: list[float] = []
        best, best_epoch, epochs_run = -np.inf, 1, 0
        for epoch in range(1, int(epochs) + 1):
            net.train()
            perm = torch.randperm(n, generator=gen).to(self.device)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                main, aux = net(Xs[idx])
                wb = wt[idx]
                loss = (wb * (main - yt[idx]) ** 2).sum() / wb.sum()
                if n_aux:
                    m = Mt[idx] * wb[:, None]
                    loss = loss + p["aux_weight"] * ((m * (aux - At[idx]) ** 2).sum() / m.sum().clamp_min(1e-12))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            epochs_run = epoch
            if val is not None:
                score = per_era_pearson(self.predict_std(net, ((val[0] - mean) / std).astype(np.float32)), val[1], val[2])
                curve.append(score)
                if score > best + 1e-12:
                    best, best_epoch = score, epoch
                elif patience is not None and epoch - best_epoch >= int(patience):
                    break
        return {"net": net, "x_mean": mean, "x_std": std, "best_epoch": int(best_epoch if val is not None else epochs_run),
                "curve": curve, "epochs_run": epochs_run, "seconds": time.perf_counter() - t0}

    def predict_std(self, net: Any, Xs: np.ndarray) -> np.ndarray:
        """Main-head eval forward on ALREADY standardised float32 rows; returns float64."""
        torch = self.torch
        net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(Xs), PRED_BATCH):
                xb = self._tensor(np.asarray(Xs[s:s + PRED_BATCH], dtype=np.float32))
                out.append(net(xb)[0].detach().cpu().numpy().astype(np.float64))
        return np.concatenate(out) if out else np.empty(0, dtype=np.float64)

    def predict(self, res: dict, X: np.ndarray) -> np.ndarray:
        return self.predict_std(res["net"], ((X - res["x_mean"]) / res["x_std"]).astype(np.float32))

    def export(self, res: dict) -> bytes:
        """npz blob of the main path (hidden layers + head; the aux head is NOT exported); W stored in x out."""
        net = res["net"]
        W = [lin.weight.detach().cpu().numpy().T.astype(np.float32) for lin in net.hidden]
        b = [lin.bias.detach().cpu().numpy().astype(np.float32) for lin in net.hidden]
        W.append(net.head.weight.detach().cpu().numpy().T.astype(np.float32))
        b.append(net.head.bias.detach().cpu().numpy().astype(np.float32))
        return _runtime.mlp_to_bytes({"W": W, "b": b, "x_mean": res["x_mean"], "x_std": res["x_std"], "columns": self.columns,
                                      "act": _runtime.MLP_ACT})

    def parity(self, res: dict, blob: bytes, X: np.ndarray) -> float:
        """max |numpy forward(blob) - torch eval forward| on <= PARITY_ROWS rows of raw inputs ``X``."""
        Xp = X[:PARITY_ROWS]
        if len(Xp) == 0:
            return 0.0
        ref = self.predict(res, Xp)
        got = _runtime.mlp_forward(_runtime.load_member("mlp", blob), Xp)
        return float(np.max(np.abs(ref - got)))


class ParityError(RuntimeError):
    """numpy forward and torch eval forward disagree by more than PARITY_TOL."""


# --------------------------------------------------------------------------------------------------
# resumable work cache: members/<m>/_work/
# --------------------------------------------------------------------------------------------------
class Work:
    """Per-(seed, fold) cache. Reused only with ``--resume`` AND an identical signature; otherwise wiped."""

    def __init__(self, mdir: Path, signature: dict, resume: bool) -> None:
        self.dir = mdir / "_work"
        meta = self.dir / "meta.json"
        sig = sha_of(signature)
        if self.dir.exists():
            old = json.loads(meta.read_text()).get("signature") if meta.exists() else None
            if not resume or old != sig:
                log.info("work cache %s: %s -> starting fresh", self.dir, "no --resume" if not resume else "signature changed")
                shutil.rmtree(self.dir)
            else:
                log.info("work cache %s: resuming", self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        if not meta.exists():
            write_json(meta, {"signature": sig, "signature_doc": signature, "created_at": _now()})

    def load(self, key: str) -> tuple[dict, bytes | None, np.ndarray | None] | None:
        j = self.dir / f"{key}.json"
        if not j.exists():
            return None
        doc = json.loads(j.read_text())
        blob = (self.dir / f"{key}.bin").read_bytes() if (self.dir / f"{key}.bin").exists() else None
        pred = np.load(self.dir / f"{key}.npy", allow_pickle=False) if (self.dir / f"{key}.npy").exists() else None
        return doc, blob, pred

    def save(self, key: str, doc: dict, blob: bytes | None = None, pred: np.ndarray | None = None) -> None:
        if blob is not None:
            (self.dir / f"{key}.bin").write_bytes(blob)
        if pred is not None:
            np.save(self.dir / f"{key}.npy", np.asarray(pred, dtype=np.float64), allow_pickle=False)
        write_json(self.dir / f"{key}.json", doc)  # last: its presence marks the item complete


# --------------------------------------------------------------------------------------------------
# the protocol
# --------------------------------------------------------------------------------------------------
def config_signature(mspec: dict, params: dict, seeds: list[int], cap: int, fe: dict, columns: list[str]) -> dict:
    p = {k: v for k, v in params.items() if k != "device"}
    folds = [{k: f[k] for k in ("name", "train_eras", "test_eras", "inner_train_eras", "inner_val_eras")} | {"src": f.get("count_sources")}
             for f in [*fe["folds"], *([fe["forward"]] if fe.get("forward") else [])]]
    return {"member": mspec["name"], "target": mspec["target"], "params": p, "seeds": seeds, "epochs_cap": cap,
            "universe": fe["universe"], "folds": sha_of(folds), "full_train_eras": sha_of(fe["full_train_eras"]),
            "columns": sha_of(columns), "target_scaling": TARGET_SCALING}


def train_member(run_dir: Path, cfg: dict, mspec: dict, fe: dict, feats: dict, args: argparse.Namespace) -> dict:
    """Run the protocol; write artifacts; return the spec.json dict (status done)."""
    t_start = time.perf_counter()
    name = mspec["name"]
    mdir = run_dir / "members" / name
    spec_path, oof_path = mdir / "spec.json", run_dir / f"oof_{name}.parquet"
    params = resolve_params(mspec)
    seeds = [int(s) for s in (args.seeds if args.seeds else mspec.get("seeds") or [0])]
    cap = int(mspec.get("es_cap") or 60)
    if args.epochs_cap:
        cap = min(cap, int(args.epochs_cap))
    universe = knobs.Universe.from_dict(fe["universe"])
    if fe.get("universe_key") and fe["universe_key"] != universe.key():
        raise ValueError(f"folds_eras.json universe_key {fe['universe_key']!r} != Universe.key() {universe.key()!r}")
    columns = [str(c) for c in feats["feature_cols"]]
    signature = config_signature(mspec, params, seeds, cap, fe, columns)
    sig = sha_of(signature)
    if args.resume and spec_path.exists() and oof_path.exists():
        old = json.loads(spec_path.read_text())
        if old.get("status") == "done" and old.get("signature") == sig and old.get("universe") == universe.key():
            log.info("%s already done for this signature (%s); nothing to do", name, spec_path)
            return old
    folds, fw = list(fe["folds"]), fe.get("forward")
    all_folds = folds + ([fw] if fw else [])
    pool = sorted({int(e) for f in all_folds for k in ("train_eras", "test_eras", "inner_train_eras", "inner_val_eras")
                   for e in f[k]} | {int(e) for e in fe["full_train_eras"]})
    torch = import_torch()
    requested = args.device or params.get("device") or "auto"
    device, device_desc = pick_device(torch, requested)
    threads = int(args.threads or (cfg.get("run") or {}).get("threads") or os.cpu_count() or 1)
    configure_torch(torch, threads)
    log.info("torch %s, device %s (--device/params: %s), %d threads; member %s seeds=%s epochs_cap=%d patience=%d hidden=%s "
             "aux=%s", torch.__version__, device_desc, requested, threads, name, seeds, cap, params["patience"], params["hidden"],
             params["aux_targets"])
    data_dir = Path(args.data_dir) if args.data_dir else Path((cfg.get("run") or {}).get("data_dir") or "data")
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir
    td = TrainData(data_dir, columns, mspec["target"], params["aux_targets"], np.asarray(pool, dtype=np.int32), universe)
    mdir.mkdir(parents=True, exist_ok=True)
    work = Work(mdir, {**signature, "n_rows": int(len(td.y))}, args.resume)
    write_json(spec_path, {"name": name, "family": "mlp", "status": "running", "universe": universe.key(), "signature": sig,
                           "started_at": _now()})
    fitter = Fitter(torch, device, params, columns)
    max_fit = 0.0

    # 1) inner early stopping per seed and fold
    best: dict[int, dict[str, int]] = {s: {} for s in seeds}
    for seed in seeds:
        for f in all_folds:
            key = f"es_s{seed}_{f['name']}"
            hit = work.load(key)
            if hit is None:
                tr = td.fit_mask(f["inner_train_eras"], f"{f['name']} inner_train")
                va = td.fit_mask(f["inner_val_eras"], f"{f['name']} inner_val")
                res = fitter.fit(td.X[tr], td.y[tr], td.weights(tr), td.Yaux[tr], seed=seed, epochs=cap,
                                 val=(td.X[va], td.y[va], td.era[va]), patience=params["patience"])
                doc = {"best_epoch": res["best_epoch"], "curve": res["curve"], "epochs_run": res["epochs_run"], "seconds": res["seconds"]}
                work.save(key, doc)
                del res
                log.info("seed %d %s: inner best epoch %d of %d run (val per-era corr %s) %.1fs", seed, f["name"], doc["best_epoch"],
                         doc["epochs_run"], [round(c, 4) for c in doc["curve"]], doc["seconds"])
            else:
                doc = hit[0]
            best[seed][f["name"]] = int(doc["best_epoch"])
            max_fit = max(max_fit, float(doc["seconds"]))
    med = {f["name"]: float(np.median([best[s][f["name"]] for s in seeds])) for f in all_folds}
    counts: dict[str, int] = {}
    sources: dict[str, list[str]] = {}
    for f in all_folds:
        src = [str(x) for x in (f.get("count_sources") or [f["name"]])]
        unknown = [x for x in src if x not in med]
        if unknown:
            raise ValueError(f"{f['name']}: count_sources {unknown} are not folds of this run")
        counts[f["name"]], sources[f["name"]] = median_round([med[x] for x in src]), src
    frozen_full = median_round([med[f["name"]] for f in folds])
    log.info("frozen epochs per fold %s (seed-median inner bests %s; sources %s); full refit %d", counts, med, sources, frozen_full)
    return _fit_and_export(run_dir, td, fitter, work, mspec, params, seeds, cap, universe, fe, feats, all_folds, folds, fw,
                           best, med, counts, sources, frozen_full, max_fit, sig, device_desc, torch.__version__, t_start)


def _fit_and_export(run_dir: Path, td: TrainData, fitter: Fitter, work: Work, mspec: dict, params: dict, seeds: list[int], cap: int,
                    universe: knobs.Universe, fe: dict, feats: dict, all_folds: list[dict], folds: list[dict], fw: dict | None,
                    best: dict, med: dict, counts: dict, sources: dict, frozen_full: int, max_fit: float, sig: str,
                    device_desc: str, torch_version: str, t_start: float) -> dict:
    """2) frozen-count refits per seed and fold -> OOF (mean over seeds); 3) full refits; 4) artifacts + spec.json."""
    name = mspec["name"]
    mdir = run_dir / "members" / name
    t_proto = time.perf_counter()
    parity_max = 0.0

    def _check(par: float, label: str) -> None:
        nonlocal parity_max
        parity_max = max(parity_max, par)
        if not np.isfinite(par) or par > PARITY_TOL:
            raise ParityError(f"{label}: numpy forward vs torch max_abs={par:.3g} > {PARITY_TOL:g}")

    parts: list[pd.DataFrame] = []
    first_blob: dict[str, bytes] = {}
    for k, f in enumerate(all_folds):
        is_fwd = fw is not None and f is fw
        test = td.rows(f["test_eras"])
        if not test.any():
            raise ValueError(f"{f['name']}: no rows in its test eras")
        preds = []
        for j, seed in enumerate(seeds):
            key = f"fit_s{seed}_{f['name']}"
            hit = work.load(key)
            if hit is None or hit[1] is None or hit[2] is None or len(hit[2]) != int(test.sum()):
                tr = td.fit_mask(f["train_eras"], f"{f['name']} train")
                res = fitter.fit(td.X[tr], td.y[tr], td.weights(tr), td.Yaux[tr], seed=seed, epochs=counts[f["name"]])
                blob = fitter.export(res)
                pred = fitter.predict(res, td.X[test])
                par = fitter.parity(res, blob, td.X[test])
                doc = {"n_epochs": counts[f["name"]], "seconds": res["seconds"], "parity_max_abs": par, "n_train": int(tr.sum())}
                del res
                _check(par, f"seed {seed} {f['name']}")
                work.save(key, doc, blob, pred)
                log.info("seed %d %s: %d epochs on %d rows %.1fs, %d test rows, parity %.2e", seed, f["name"], doc["n_epochs"],
                         doc["n_train"], doc["seconds"], len(pred), par)
            else:
                doc, blob, pred = hit
                _check(float(doc["parity_max_abs"]), f"cached seed {seed} {f['name']}")
            max_fit = max(max_fit, float(doc["seconds"]))
            preds.append(np.asarray(pred, dtype=np.float64))
            if j == 0:
                first_blob[f["name"]] = blob
        parts.append(pd.DataFrame({"era": td.era[test].astype(np.int32), "fold": np.int8(-1 if is_fwd else k),
                                   "pred": np.mean(np.vstack(preds), axis=0).astype(np.float32)}, index=td.ids[test]))
    fit_seconds = time.perf_counter() - t_proto

    full = td.fit_mask(fe["full_train_eras"], "full refit")
    probe = np.flatnonzero(full)[:PARITY_ROWS]
    full_files: list[dict] = []
    full_blobs: list[bytes] = []
    for j, seed in enumerate(seeds):
        key = f"full_s{seed}"
        hit = work.load(key)
        if hit is None or hit[1] is None:
            res = fitter.fit(td.X[full], td.y[full], td.weights(full), td.Yaux[full], seed=seed, epochs=frozen_full)
            blob = fitter.export(res)
            par = fitter.parity(res, blob, td.X[probe])
            doc = {"n_epochs": frozen_full, "seconds": res["seconds"], "parity_max_abs": par, "n_train": int(full.sum())}
            del res
            _check(par, f"full refit seed {seed}")
            work.save(key, doc, blob)
            log.info("full refit seed %d: %d epochs on %d rows %.1fs, parity %.2e", seed, frozen_full, doc["n_train"], doc["seconds"], par)
        else:
            doc, blob, _ = hit
            _check(float(doc["parity_max_abs"]), f"cached full refit seed {seed}")
        max_fit = max(max_fit, float(doc["seconds"]))
        full_blobs.append(blob)
        full_files.append({"file": f"full_s{j}.bin", "seed": int(seed), "n_iter": int(frozen_full)})

    # artifacts: blobs + OOF first, spec.json (status done) LAST
    for stale in mdir.glob("*.bin"):
        stale.unlink()
    for k, f in enumerate(folds):
        (mdir / f"fold{k}.bin").write_bytes(first_blob[f["name"]])
    if fw is not None:
        (mdir / "forward.bin").write_bytes(first_blob[fw["name"]])
    for ff, blob in zip(full_files, full_blobs):
        (mdir / ff["file"]).write_bytes(blob)
    oof = pd.concat(parts)
    oof.index.name = "id"
    oof.to_parquet(run_dir / f"oof_{name}.parquet")
    columns = list(fitter.columns)
    spec = {**{k: v for k, v in mspec.items()}, "arm": mspec.get("arm") or ("aux" if mspec.get("group") == "aux" else name),
            "params": params, "feature_cols": columns, "all_feature_cols": columns,
            "missing_cols": [str(c) for c in feats.get("missing_cols") or []],
            "feature_cols_per_fold": {f["name"]: columns for f in all_folds},
            "inner_iters": [int(round(med[f["name"]])) for f in folds],
            "inner_iter_forward": None if fw is None else int(round(med[fw["name"]])),
            "inner_best_epochs": {str(s): best[s] for s in seeds},
            "frozen_n_iter": int(frozen_full), "frozen_n_iter_per_fold": [int(counts[f["name"]]) for f in folds],
            "frozen_n_iter_forward": None if fw is None else int(counts[fw["name"]]), "count_sources": sources,
            "fit_seconds": fit_seconds, "max_fit_seconds": max_fit, "member_seconds": time.perf_counter() - t_start,
            "full_blobs": full_files, "fold_blob_seed": int(seeds[0]), "seeds": seeds, "epochs_cap": int(cap),
            "device": device_desc, "torch_version": torch_version, "parity_max_abs": parity_max, "parity_tol": PARITY_TOL,
            "corr_with_bench_val": None, "rho_c": None, "universe": universe.key(), "universe_dict": universe.to_dict(),
            "signature": sig, "external": True, "trained_by": "scripts/mlp_train.py", "finished_at": _now(), "status": "done"}
    write_json(mdir / "spec.json", spec)
    log.info("%s done: frozen epochs %s / forward %s / full %d; OOF %d rows; parity max %.2e; %.0fs", name,
             spec["frozen_n_iter_per_fold"], spec["frozen_n_iter_forward"], frozen_full, len(oof), parity_max, spec["member_seconds"])
    return spec


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def _setup_logging(run_dir: Path | None) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    if run_dir is not None and run_dir.exists():
        fh = logging.FileHandler(run_dir / "mlp_train.log")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def _mark(spec_path: Path, status: str, msg: str, *, keep_done: bool) -> None:
    """Record a failure in spec.json without ever clobbering a finished member when ``keep_done``."""
    try:
        old = json.loads(spec_path.read_text()) if spec_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        old = {}
    if keep_done and old.get("status") == "done":
        return
    if spec_path.parents[2].exists():  # artifacts/<run> exists: members/<m>/ may be created
        base = {"name": spec_path.parent.name, "family": "mlp", **{k: old[k] for k in ("name", "universe", "signature") if k in old}}
        write_json(spec_path, {**base, "status": status, "error": msg, "at": _now()})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scripts/mlp_train.py", description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts")
    ap.add_argument("--member", default=None, help="mlp member name (default: the config's first family: mlp member)")
    ap.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"], help="default: the member's params.device")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override the member's seeds")
    ap.add_argument("--epochs-cap", type=int, default=None, help="lower the member's es_cap (smoke tests)")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: run.threads)")
    ap.add_argument("--data-dir", type=Path, default=None, help="default: the config's run.data_dir")
    ap.add_argument("--resume", action="store_true", help="reuse finished (seed, fold) items under members/<m>/_work")
    args = ap.parse_args(argv)
    run_dir = Path(args.artifacts) / args.run_id
    _setup_logging(run_dir)
    spec_path: Path | None = None
    try:
        cfg, mspec, fe, feats = load_run(run_dir, args.member)
        spec_path = run_dir / "members" / mspec["name"] / "spec.json"
        spec = train_member(run_dir, cfg, mspec, fe, feats, args)
        log.info("MLP OK: %s status=%s parity_max_abs=%s device=%s", spec["name"], spec["status"], spec.get("parity_max_abs"),
                 spec.get("device"))
        return EXIT_OK
    except NoDevice as exc:
        log.error("NO USABLE DEVICE -> skip the MLP member: %s", exc)
        if spec_path is not None:
            _mark(spec_path, "no_device", str(exc), keep_done=True)
        return EXIT_NO_DEVICE
    except ParityError as exc:
        log.error("PARITY FAILED: %s", exc)
        if spec_path is not None:
            _mark(spec_path, "failed_parity", str(exc), keep_done=False)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        log.error("mlp_train failed: %s\n%s", exc, traceback.format_exc())
        if spec_path is not None:
            _mark(spec_path, "failed", f"{type(exc).__name__}: {exc}", keep_done=True)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
