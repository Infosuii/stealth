# Pins — installed versions (train venv, Python 3.11.2, sandbox 2026-11-13)

All lane pins installed **exactly** as specified; no deviations.

| package | pin | installed |
|---|---|---|
| lightgbm | 4.7.0 | 4.7.0 |
| xgboost | 3.2.0 | 3.2.0 |
| scikit-learn | 1.9.1 | 1.9.1 |
| pandas | 3.0.6 | 3.0.6 |
| numpy | 2.4.6 | 2.4.6 |
| scipy | 1.17.1 | 1.17.1 |
| catboost | 1.2.10 | 1.2.10 |
| pyarrow | 24.0.0 | 24.0.0 |
| cloudpickle | 3.1.2 | 3.1.2 |
| everesteer-api | 0.4.1 | 0.4.1 (import name `everestapi`) |

Unpinned-by-plan, resolved and frozen in `requirements-lock.txt`: optuna 5.0.0, PyYAML 6.0.3, pytest 9.2.0.

Notes
- `torch` is deliberately NOT installed in `.venv` (tests assert `find_spec("torch") is None`). The optional MLP (M6) uses a separate GPU venv.
- `everesteer-api` 0.4.1 `unq20` already applies the zero rule (see docs/INTERFACES.md §0).
- Recreate: `python3.11 -m venv .venv && .venv/bin/python -m pip install -r requirements-lock.txt -e .`
