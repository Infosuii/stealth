# stealth

Prediction bot for the Everesteer **Himalayas** (futures) tournament: a purged-CV ensemble of small gradient-boosted
trees (LightGBM / XGBoost / optional CatBoost) on the main and auxiliary targets plus ridge, equal-weight blended,
with partial benchmark / core-feature neutralisation *inside* the prediction function. Packaged as a by-value
cloudpickle `predict(live_features, live_benchmark_models=None)` for the platform's auto-run lane. CPU only;
Python 3.11 with the exact lane pins. Plan: `docs/PLAN.md`; module contracts: `docs/INTERFACES.md`; decisions:
`docs/DECISIONS.md`.

## Quickstart (Linux)

```bash
git clone https://github.com/Infosuii/stealth "$HOME/stealth" && cd "$HOME/stealth"   # the handoffs assume $HOME/stealth
bash scripts/setup.sh                 # python 3.11 venv (with pip), exact pins, pre-commit hook, selfcheck; prints PY=...
cp -n .env.example .env               # paste EIQ_API_KEY (never committed, never printed)
PY="$HOME/stealth/.venv/bin/python"
"$PY" -m stealth.data download        # data/ (gitignored)
"$PY" -m pytest -q
```
Then follow **`handoffs/INDEX.md`** — every step the laptop runs is a numbered, self-contained runbook that reports
back via `"$PY" -m stealth.report --run-id <id> --handoff NN` (writes `handoffs/results/NN-*`, commits, pushes).

## The three configs

| config | where | what | time |
|---|---|---|---|
| `configs/smoke.yaml` | sandbox / laptop | eras ≥ 5000, 3 folds, 2 LGBM + ridge, 50-draw simulation — proves the pipeline | ≤ 8 min on 2 cores |
| `configs/test.yaml` | laptop | eras ≥ 2800, 5 folds + forward block, 12-cell LGBM grid, 3 aux, 4 XGB, 1 timed CatBoost, ridge — reproduces the baseline (OOF fit20 0.026 ± 0.004) | 30–45 min |
| `configs/full.yaml` | laptop | ≤ 25 TPE trials, 6 aux × 2 families, feature bags, era/decay knobs, refit seeds — the model that gets uploaded | 2.5–3.5 h |

Pipeline per run: `train --config C --run-id R --resume` → `blend --run-id R` → `bundle --run-id R --out models/X.pkl`
→ `pkl_check models/X.pkl` (throwaway venv without `stealth`) → `report --run-id R --handoff NN`.
Daily: the platform lane runs the uploaded pickle; `scripts/daily_task.sh` (cron) is the fallback; `monitor` checks.

## Safety rules

- **Nothing is submitted, uploaded or staked** without a handoff carrying a **USER MUST APPROVE** banner;
  `predict --submit/--upload` additionally require `--i-have-approval` AND an environment variable set to `1`:
  `STEALTH_CONFIRM_SUBMIT` (submit) / `STEALTH_CONFIRM_UPLOAD` (upload), or `STEALTH_APPROVED` which covers either
  (the code accepts both spellings; `--help` lists them).
- `.env`, `data/`, `artifacts/`, `*.pkl` are never committed; `scripts/hooks/pre-commit` (activated by
  `git config core.hooksPath scripts/hooks`) blocks them and any secret-shaped text. Never `cat .env`, never
  `git remote -v`, never `git config --list`.
- The repo is public: results files carry metrics and config *names* only; hyperparameter values stay in `artifacts/`.
- Never fit to, blend with or select on `v1_sherpa` (the train-split benchmark is in-sample).
- Offline numbers above ~0.034 fit20 are a leak or winner's curse until the forward block and live rounds agree.
