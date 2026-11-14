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
| 00 | env-setup | venv + pins + hook + data download + pytest on the laptop | accepted | – | b96dd66, ee2afbd | PINS OK, hooksPath set, 6 parquet files, 163 passed; the 51 MB train file is correct (the 80–90 MB figure was a doc error, fixed) |
| 01 | smoke | `configs/smoke.yaml` end to end: train → blend → bundle → pkl_check → report | accepted | – | c33bff6, acfaf33 | pkl_check PASSED (0.38 s/era), predict --offline ok, 163 passed |
| 02 | test-run | `configs/test.yaml` (M1): baseline reproduction, CatBoost timing | accepted | – | 34c722a, 86c1c5c, 88b4e7e | fit 0.0238, fwd 0.0313, sharpe 0.51 → M1 met; EW blend 0.0226 lost to the single LGBM → baseline kept; NNLS tie (p_pos 0.44); Cat 18 s kept but weak; search a tie; whole run 12.5 min |
| 03 | diag-upload-1 | **USER MUST APPROVE** — validation-diagnostics upload #1 of 5: the M1 single-LightGBM baseline (`bundle --run-id test01 --members lgbm_everest --p-b 0 --p-c 0` → `models/m1_lgbm.pkl`) | accepted | – | 56819a6 | upload 35bfe266: fit20 0.0189, unq 0.0039, inov 0.0141, sharpe 0.32, round score ~0.0368 → rank 89/150 (board median 0.0388); OOF→validation FIT ratio 0.79; offline simulation predicted 0.0383 (calibrated) |
| 04 | full-run | M2–M6 model, reserved audit, local packaging and diagnostics file | accepted (technical) | – | c66d0cd, da2ff1c, a329cb3 | status ok; lgbm + ridge NNLS; audit FIT 0.0266 vs baseline 0.0308, p_pos 0.0412: no audit win; retain baseline; 47m21s pipeline |
| 05 | diag-upload-2 | **USER MUST APPROVE** — optional calibration of the unchanged full01 challenger; no automatic promotion | ready (optional) | user decision | – | baseline remains the conservative candidate; audit result preserved; no upload performed |
| 06 | refit-and-arm | **USER MUST APPROVE** — separately specified M7 full-data refit, lane upload and monitoring | draft (not written) | development protocol and user decision | – | not implied by handoff 04 or diagnostics #2 |

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
