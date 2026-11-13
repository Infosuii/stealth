# Handoffs — index

Each `handoffs/NN-<slug>.md` is a self-contained runbook that a context-free AI (the user's GPT) executes on the
laptop (Linux, Europe/London) using the absolute venv interpreter `$PY`, then reports back by committing
`handoffs/results/NN-{results.md,metrics.json,log-tail.txt}` and pushing. Claude (sandbox) builds and smoke-tests,
writes the next handoff, pushes; the user runs it via GPT; Claude pulls, reads `NN-metrics.json`, iterates.

Status values: `draft` → `ready` → `running` → `reported` → `accepted` | `failed`. "Waiting on" names who acts next.

## Status table

| NN | slug | purpose | status | waiting on | result sha | verdict |
|---|---|---|---|---|---|---|
| 00 | env-setup | venv + pins + hook + data download + pytest on the laptop | ready | GPT | – | – |
| 01 | smoke | `configs/smoke.yaml` end to end: train → blend → bundle → pkl_check → report (~10 min) | ready (after 00) | GPT | – | – |
| 02 | test-run | `configs/test.yaml` (M1): OOF fit 0.026 ± 0.004, forward ≥ 0.020, Sharpe ≥ 0.5, CatBoost timing | ready (after 01) | GPT | – | – |
| 03 | diag-upload-1 | **USER MUST APPROVE** — first of 5 validation-diagnostics uploads (M1 baseline) | draft | Claude | – | – |
| 04 | full-run | `configs/full.yaml` (M2–M5): grid + knobs + aux + families + neutralisation constants (2.5–3.5 h) | draft | Claude | – | – |
| 05 | upload-and-arm | **USER MUST APPROVE** — `predict --upload`, `monitor`, `models/active.json`, cron fallback (M7) | draft | Claude | – | – |

Fill in `result sha` with the commit that added `handoffs/results/NN-*` and `verdict` with one line from the
success criterion (e.g. "fit 0.0263, fwd 0.021, sharpe 0.55, Cat 95 s kept").

## Decision log

Full table with rationale: `docs/DECISIONS.md`. Summary that shapes these handoffs:

| date | decision | effect on handoffs |
|---|---|---|
| 2026-11-13 | Repo stays **public** | results files carry metrics + config names only; hyperparameter values live in `artifacts/` (gitignored); `.pkl` never committed |
| 2026-11-13 | Laptop = **Linux**, **Europe/London** | bash-only blocks; cron lines in `scripts/daily_task.sh` fire 14:30/15:30/16:30 local, gated on 13:05–15:55 UTC |
| 2026-11-13 | Pre-armed hosted baseline **disabled** | no live canary; first live calibration comes from our own upload (handoff 05) |
| 2026-11-13 | **≤ 5** diagnostics uploads approved, pre-registered (M1, M3, M5 ×2, final) | each is its own handoff with a USER MUST APPROVE banner (03 is the first) |
| 2026-11-13 | Laptop **always on** | `scripts/daily_task.sh` cron fallback is viable; primary path remains the platform's auto-run lane |
| 2026-11-13 | GPT has its own GitHub access | handoffs just say "commit and push"; no `gh auth` HUMAN STEP |
| 2026-11-13 | Staking **possibly yes** | selection weights Sharpe and the benchmark-negative regime, not mean FIT alone (02 records both) |

## Standing rules (every handoff repeats them)

- Nothing is submitted, uploaded or staked without a handoff carrying a **USER MUST APPROVE** banner.
- Never commit `data/`, `.env`, `*.pkl`, `artifacts/`; the pre-commit hook (`core.hooksPath scripts/hooks`) blocks them and any secret-shaped text.
- Never `cat`/`echo` `.env`, never `git remote -v`, never `git config --list`.
- `report.py` always writes the results files first; if the push fails it says `PUSH FAILED — ...` and the human pushes.
