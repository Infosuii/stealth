"""Optuna search over coarse categorical grids, SQLite storage in artifacts/<run>/optuna.db (resumable).

Contract (docs/INTERFACES.md section 6)::

    run_search(name, grid, objective, *, n_trials, storage, seed) -> [(params, value), ...]  # sorted desc
    top_k_params(results, k) -> [params, ...]

``grid`` maps param name -> list of allowed values (categorical choices only). ``n_trials >= grid size``
means exhaustive (``GridSampler``); otherwise TPE. One study per ``name`` in ``sqlite:///<storage>``; a
re-run with the same storage resumes and only runs the missing trials. A proposed point that was already
evaluated is answered from the stored value without calling ``objective`` again, and the returned list is
de-duplicated on params.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import optuna

log = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def grid_size(grid: dict[str, list]) -> int:
    """Number of cells in the full grid."""
    return math.prod(len(v) for v in grid.values()) if grid else 0


def _key(params: dict) -> tuple:
    return tuple(sorted((k, repr(v)) for k, v in params.items()))


def _cast(value: Any, choices: list) -> Any:
    """Optuna stores categorical values as-is, but SQLite round-trips may change int/float; match to the grid."""
    for c in choices:
        if c == value and type(c) is type(value):
            return c
    for c in choices:
        if c == value:
            return c
    return value


def _completed(study: optuna.Study, grid: dict[str, list]) -> dict[tuple, tuple[dict, float]]:
    done: dict[tuple, tuple[dict, float]] = {}
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        params = {k: _cast(t.params[k], grid[k]) for k in grid if k in t.params}
        if len(params) != len(grid) or t.value is None:
            continue
        done[_key(params)] = (params, float(t.value))
    return done


def run_search(
    name: str,
    grid: dict[str, list],
    objective: Callable[[dict], float],
    *,
    n_trials: int,
    storage: Path,
    seed: int,
    study_name: str | None = None,
) -> list[tuple[dict, float]]:
    """Maximise ``objective(params)`` over ``grid``; see the module docstring. Returns (params, value) sorted desc.

    ``study_name`` (default ``name``) is the Optuna study identity; train.py namespaces it with the run fingerprint and
    universe so a study recorded under another config / data / protocol is never resumed."""
    if not grid:
        raise ValueError(f"search {name!r}: empty grid")
    size = grid_size(grid)
    n_trials = int(min(n_trials, size))
    storage = Path(storage)
    storage.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{storage.as_posix()}"
    exhaustive = n_trials >= size
    sampler: optuna.samplers.BaseSampler
    if exhaustive:
        sampler = optuna.samplers.GridSampler(grid, seed=seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(10, max(n_trials // 2, 1)), multivariate=True)
    study = optuna.create_study(study_name=study_name or name, storage=url, direction="maximize", sampler=sampler,
                                load_if_exists=True)
    done = _completed(study, grid)
    log.info("search %s: grid=%d cells, target=%d distinct trials, %d already done (%s)",
             name, size, n_trials, len(done), "exhaustive" if exhaustive else "TPE")

    def _obj(trial: optuna.Trial) -> float:
        params = {k: trial.suggest_categorical(k, list(v)) for k, v in grid.items()}
        params = {k: _cast(v, grid[k]) for k, v in params.items()}
        key = _key(params)
        if key in done:
            log.info("search %s: duplicate point, reusing stored value %.5f", name, done[key][1])
            return done[key][1]
        value = float(objective(params))
        if not math.isfinite(value):
            value = -1e9
        done[key] = (params, value)
        log.info("search %s: trial %d value=%.5f (%d/%d distinct)", name, trial.number, value, len(done), n_trials)
        return value

    budget = 0
    max_total = 4 * n_trials + 10  # guard against TPE re-proposing duplicates forever
    while len(done) < n_trials and budget < max_total:
        remaining = n_trials - len(done)
        study.optimize(_obj, n_trials=remaining, gc_after_trial=True)
        budget += remaining
        if exhaustive and len(study.trials) >= size:
            break
    results = sorted(done.values(), key=lambda pv: pv[1], reverse=True)
    log.info("search %s: done, best=%.5f over %d distinct points", name, results[0][1] if results else float("nan"), len(results))
    return results


def top_k_params(results: list[tuple[dict, float]], k: int) -> list[dict]:
    """The ``k`` best param dicts (results already sorted desc); fewer if the search produced fewer."""
    return [dict(p) for p, _ in results[: max(int(k), 1)]]


def average_predictions(preds: list[Any]) -> Any:
    """Equal-weight mean of the top-k members' raw predictions (arrays of equal length)."""
    import numpy as np

    if not preds:
        raise ValueError("average_predictions: nothing to average")
    return np.mean(np.vstack([np.asarray(p, dtype=np.float64) for p in preds]), axis=0)


__all__ = ["average_predictions", "grid_size", "run_search", "top_k_params"]
