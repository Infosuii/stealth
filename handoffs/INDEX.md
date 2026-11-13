# Handoffs — index

Each `handoffs/NN-<slug>.md` is a self-contained runbook. A context-free AI (the user's GPT) executes it on the
laptop (Linux, Europe/London), always calling the absolute venv interpreter `$PY`. It then reports back by
committing `handoffs/results/NN-{results.md,metrics.json,log-tail.txt}` and pushing.

The loop:
1. Claude (sandbox) builds and smoke-tests the code, writes the next handoff, and pushes.
2. The user runs the handoff via GPT.
3. Claude pulls, reads `NN-metrics.json`, and iterates.

Status values: `draft` → `ready` → `running` → `reported` → `accepted` | `failed`. "Waiting on" names who acts next.

## Status table

| NN | slug | purpose | status | waiting on | result sha | verdict |
|---|---|---|---|---|---|---|
| 00 | env-setup | venv + pins + hook + data download + pytest on the laptop | accepted | – | b96dd66, ee2afbd | PINS OK, hooksPath set, 6 parquet files, 163 passed |
| 01 | smoke | `configs/smoke.yaml` end to end: train → blend → bundle → pkl_check → report | accepted | – | c33bff6, acfaf33 | pkl_check PASSED (0.38 s/era), predict --offline ok, 163 passed |
| 02 | test-run | `configs/test.yaml` (M1): baseline reproduction, CatBoost timing | accepted | – | 34c722a, 86c1c5c, 88b4e7e | fit 0.0238, fwd 0.0313, sharpe 0.51, Cat 18 s kept, blend 0.0226 (EW; NNLS tie p_pos 0.44), 8.3 min train + 3.5 min simulation; blend-vs-baseline p_pos not recorded (reporter gap, filled in by 03's report) |
| 03 | diag-upload-1 | **USER MUST APPROVE** — validation-diagnostics upload #1 of 5: the M1 single-LightGBM baseline (`bundle --members lgbm_everest` of test01 -> `models/test01_lgbm.pkl`; the EW blend lost to it in 02) | **ready** | GPT, then the user's `yes, upload` in chat | – | – |
| 04 | full-run | `configs/full.yaml` (M2–M6): knobs, TPE search + adoption gate, aux, families, ladder, MLP (GPU venv), neutralisation constants from the simulation, `bundle --require-frozen`, diag **file** (no upload) | draft | Claude (finalise from 02: Cat kept, baseline_fit 0.0238, threads 16, timings) | – | – |
| 05 | upload-and-arm | **USER MUST APPROVE** — `predict --upload`, `monitor`, `models/active.json`, cron fallback (M7) | draft | Claude | – | – |

Fill in `result sha` with the commit(s) that added `handoffs/results/NN-*`. Fill in `verdict` with one line from
the success criterion.

The diagnostics uploads after 03 (#2–#5: the M3/M5 checks and the final model) are each a separate
USER MUST APPROVE handoff. They are numbered when they are written. 04 builds the next diagnostics file but never
uploads it.

## Decision log

Full table with rationale: `docs/DECISIONS.md`. Summary of the decisions that shape these handoffs:

| date | decision | effect on handoffs |
|---|---|---|
| 2026-11-13 | Repo stays **public** | results files carry metrics + config names only; hyperparameter values live in `artifacts/` (gitignored); `.pkl` never committed |
| 2026-11-13 | Laptop = **Linux**, **Europe/London** | bash-only blocks; cron lines in `scripts/daily_task.sh` fire 14:30/15:30/16:30 local, gated on 13:05–15:55 UTC |
| 2026-11-13 | Pre-armed hosted baseline **disabled** | no live canary; first live calibration comes from our own upload (handoff 05) |
| 2026-11-13 | **≤ 5** diagnostics uploads approved, pre-registered (M1, M3, M5 ×2, final) | each is its own handoff with a USER MUST APPROVE banner (03 is the first) |
| 2026-11-13 | Laptop **always on** | `scripts/daily_task.sh` cron fallback is viable; primary path remains the platform's auto-run lane |
| 2026-11-13 | GPT has its own GitHub access | handoffs just say "commit and push"; no `gh auth` HUMAN STEP |
| 2026-11-13 | Staking **possibly yes** | selection weights Sharpe and the benchmark-negative regime, not mean FIT alone (04: Sharpe admission route, staking-aware `p_b`/`p_c`) |
| 2026-11-13 | M2–M6 merge into `main` approved in advance (after tests + reviews) | 03/04 run on the merged code; both re-run `pytest` first |
| 2026-11-13 | Handoff 03 approved in advance | the GPT **still** asks the user in chat right before the upload step and needs an explicit `yes, upload` |
| 2026-11-13 | 02 ran 3–4x faster than estimated | spare compute goes to measurement, slower trees, admission candidates and the MLP — not to bigger searches or seed bagging (04 finalisation) |

## Standing rules (every handoff repeats them)

- Nothing is submitted, uploaded or staked without a handoff carrying a **USER MUST APPROVE** banner, and then only
  after the user approves **in chat** at that step.
- Never commit `data/`, `.env`, `*.pkl`, `artifacts/`. The pre-commit hook (`core.hooksPath scripts/hooks`) blocks
  them, and any secret-shaped text.
- Never `cat`/`echo` `.env`, never `git remote -v`, never `git config --list`.
- `report.py` always writes the results files first. If the push fails it says `PUSH FAILED — ...` and the human
  pushes.
- Use the same clone for every handoff. Its path is `REPO` in each handoff's section 4; 00–02 ran in a clone that is
  not `$HOME/stealth`, so set `REPO` to that path.
