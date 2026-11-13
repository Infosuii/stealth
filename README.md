# stealth

Prediction bot for the Everesteer **Himalayas** (futures) tournament. It is a purged-CV ensemble of:
- small gradient-boosted trees (LightGBM / XGBoost / CatBoost) on the main and auxiliary targets;
- ridge;
- an optional MLP, trained in a separate GPU venv and exported to numpy.

Members join through a pre-registered, paired-bootstrap admission ladder and are blended with hierarchical equal
weights (or nested NNLS if it passes its gate). Partial benchmark / core-feature neutralisation runs *inside* the
prediction function. Its constants `p_b` / `p_c` are frozen from a staking-aware regime simulation, never from the
diagnostics lane.

The model ships as a by-value cloudpickle `predict(live_features, live_benchmark_models=None)` for the platform's
auto-run lane. It runs on CPU only, under Python 3.11 with the exact lane pins; torch never enters the pickle.

Docs: plan in `docs/PLAN.md`; module contracts in `docs/INTERFACES.md` (sections 11–20 cover M2–M6); decisions in
`docs/DECISIONS.md`.

## Quickstart (Linux)

```bash
git clone https://github.com/Infosuii/stealth "$HOME/stealth" && cd "$HOME/stealth"   # any path works; handoffs call it REPO
bash scripts/setup.sh                 # python 3.11 venv (with pip), exact pins, pre-commit hook, selfcheck; prints PY=...
cp -n .env.example .env               # paste EIQ_API_KEY (never committed, never printed)
PY="$HOME/stealth/.venv/bin/python"
"$PY" -m stealth.data download        # data/ (gitignored)
"$PY" -m pytest -q
bash scripts/setup_gpu.sh             # OPTIONAL: .venv-gpu with torch cu128 (RTX 50xx / sm_120) for the MLP member only;
                                      #   ends with "GPU OK" (exit 0) or "GPU FAIL -> skip MLP" (exit 3)
```
Then follow **`handoffs/INDEX.md`**. Every step the laptop runs is a numbered, self-contained runbook. Each one
reports back via `"$PY" -m stealth.report --run-id <id> --handoff NN`, which writes `handoffs/results/NN-*`, commits
and pushes.

## The configs

| config | where | what | time |
|---|---|---|---|
| `configs/smoke.yaml` | sandbox / laptop | eras ≥ 5000, 3 folds, 2 LGBM + ridge, 50-draw simulation — proves the M1 pipeline | ≤ 8 min on 2 cores |
| `configs/minifull.yaml` | sandbox | every M2–M6 path at toy scale: knobs on a pool from 4600, TPE search + adoption gate, aux, XGB grid, tiny CatBoost, ridge, sharpe arm, 2 bags, MLP on CPU torch, ladder, 4×3 (p_b, p_c) grid, `bundle --require-frozen` | ≤ 8 min on 2 cores |
| `configs/test.yaml` | laptop | eras ≥ 2800, 5 folds + forward block, 12-cell LGBM grid, 3 aux, 4 XGB, 1 timed CatBoost, ridge — the M1 baseline (handoff 02: OOF fit20 0.0238, forward 0.0313) | ~12 min on 16 threads |
| `configs/full.yaml` | laptop (+ GPU for the MLP) | knobs, ≤ 25 TPE trials, 6 aux × 2 families, XGB 12, Cat 3, ridge, sharpe + 4 bags, MLP, admission ladder, simulation-frozen `p_b`/`p_c` — the model that gets uploaded | ~1–2.5 h |

The pipeline runs these steps per run:
1. `train --config C --run-id R --resume`
2. (MLP) `scripts/mlp_train.py --run-id R --resume` in `.venv-gpu`
3. `train --resume`, which loads the external member
4. `blend --run-id R`
5. `bundle --run-id R --out models/X.pkl [--require-frozen]`
6. `pkl_check models/X.pkl` (throwaway venv without `stealth`)
7. `diag --pkl models/X.pkl --out artifacts/diag/R_validation.parquet` (file only)
8. `report --run-id R --handoff NN`

One driver runs all of it, is idempotent, and resumes when re-run with the same command:
```bash
bash scripts/run_pipeline.sh configs/full.yaml full01 --gpu-py "$HOME/stealth/.venv-gpu/bin/python"   # or --no-mlp
bash scripts/run_pipeline.sh configs/minifull.yaml mf01 --dry-run    # print the commands, run nothing
```
An MLP failure is logged as "MLP skipped" and the pipeline continues. Any other failing step stops it with that
step's exit code.

The validation-diagnostics lane is a USER-GATED upload. It is free but limited to 5 pre-registered uploads:
`stealth.diag --preflight`, then `--upload --i-have-approval --wait` with `STEALTH_CONFIRM_DIAG=1`, then
`--fetch-result` if scoring is slow.

Daily: the platform lane runs the uploaded pickle; `scripts/daily_task.sh` (cron) is the fallback; `monitor` checks.

## Safety rules

- **Nothing is submitted, uploaded or staked** without a handoff carrying a **USER MUST APPROVE** banner. On top of
  that, `predict --submit/--upload` and `diag --upload` require both `--i-have-approval` and an environment variable
  set to `1`:
  - `STEALTH_CONFIRM_SUBMIT` (submit);
  - `STEALTH_CONFIRM_UPLOAD` (upload);
  - `STEALTH_CONFIRM_DIAG` (diagnostics);
  - or `STEALTH_APPROVED`, which covers any of them.

  `diag --upload` also refuses to upload the same file twice.
- `.env`, `data/`, `artifacts/`, `*.pkl` are never committed. `scripts/hooks/pre-commit` (activated by
  `git config core.hooksPath scripts/hooks`) blocks them and any secret-shaped text. Never `cat .env`, never
  `git remote -v`, never `git config --list`.
- The repo is public: results files carry metrics and config / arm *names* only; hyperparameter values stay in
  `artifacts/`.
- Never fit to, blend with or select on `v1_sherpa` (the train-split benchmark is in-sample). Never tune `p_b` /
  `p_c` on the diagnostics lane.
- Offline numbers above M1 + 0.004 fit20 (0.0278) are a leak or winner's curse until the forward block and live
  rounds agree.
