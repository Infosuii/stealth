# Decisions log

Decisions made by the user that shape the build. Newest at the bottom.

## 2026-11-13 — plan approval

| # | Question | Decision |
|---|---|---|
| 1 | Repo visibility | **Stays public.** User accepts that configs/results are readable. The `.pkl` is still never committed. |
| 2 | Laptop OS / timezone | **Linux**, timezone **Europe/London** (UK; observes DST). |
| 3 | Pre-armed hosted LightGBM baseline on `mandelbrot-bayesian-payoff` | **Disabled** (auto-submit off). No canary; live calibration will come from our own first upload instead. |
| 4 | Validation-diagnostics uploads | **Approved: up to 5**, pre-registered per the plan (M1, M3, M5 ×2, final). Each still needs a USER MUST APPROVE handoff. |
| 5 | Laptop availability | **Basically always on** → scheduled daily fallback (`predict --submit` if no accepted submission) is viable. |
| 6 | GPT push auth | User handles it; the GPT has its own GitHub access. Handoffs just say "commit and push". |
| 7 | Staking intent | **Possibly yes** → model selection weights Sharpe / benchmark-negative-regime robustness more heavily, not just mean FIT. |

## 2026-11-13 — after handoff 02

| # | Question | Decision |
|---|---|---|
| 8 | Merge the M2–M6 branch into `main` | **Approved in advance**, once tests and reviews pass. |
| 9 | Handoff 03 (diag upload #1 of the M1 LightGBM baseline) | **Approved in advance.** The GPT still asks for confirmation in its own chat before the upload step. |
| 10 | Compute budget | The laptop ran handoff 02 in ~12.5 min (est. 30–45). Spend the extra on measurement (fold offsets, bootstrap draws, stability checks), slower trees, admission candidates and the MLP — not on bigger searches or seed bagging. |

## 2026-11-14 — fix pass after the three independent reviews

| # | Question | Decision |
|---|---|---|
| 11 | Independent evidence for 04 | **Reserved audit partition** (user: finish it before calling results independent; keep audit labels out of every adaptive decision). Eras 5945–6524 are reserved; development CV, the dev forward block (now 5300/5340), knobs, search, ladder, NNLS, count sources and the 04 deployment full refits all stop before 5925. `stealth.audit` scores the frozen blend once. A later full-data refit (M7) is a separate decision, never done silently after reading the audit. |
| 12 | Spare laptop compute (DECISIONS #10, restated by the user) | **Measurement and stability only**: paired-bootstrap draws 2000 → 5000, regime-simulation draws 500 → 1000, audit stability segments + block-length sensitivity. Unchanged: ≤ 25 LGBM TPE trials, the fixed 12-cell XGB grid, refit seeds [0, 1]; no added seed bagging. |
| 13 | Historical SUSPECT baseline 0.0229 | **Context only** (`gates.baseline_fit: null`): it was scored on folds over 2800–6524. The SUSPECT bar is the in-run unsearched LightGBM base cell + 0.004. |
| 14 | Handoff 03 | Accepted (upload `35bfe266…`); 04 must not repeat it. Four diagnostics uploads remain, each its own USER MUST APPROVE handoff. |
| 15 | Laptop for 04 | Linux, 20 logical threads with `run.threads: 16`, RTX 5070 Ti (MLP via `--device cuda`, no silent CPU fallback), ~30.8 GB RAM, Europe/London. |

## 2026-11-14 — after laptop 04 and delegated follow-up

The user delegated the development follow-up to the laptop agent. Handoff 04 completed technically but reported
no audit win. Preserve the pre-registered unsearched comparator as the conservative candidate; do not promote
the larger blend from its optimistic development scores. `bundle --audit-baseline` exports its existing refits
without changing any model decisions or fitting on the audit. The baseline differs from the knob-selected
`--members lgbm_everest` export.

All newly inspected diagnostics would be additional selection evidence, never a new untouched holdout. Optional
diagnostics #2 gets handoff 05; M7 refit/deployment remains separately specified and authorised. This follow-up
performs no uploads, submissions, model-slot changes or staking. See `docs/reviews/bot-followup-2026-11-14.md`.

## 2026-11-14 — autonomous model research and compute preference

The user explicitly authorised substantial model improvements, code changes and autonomous local research,
and asked that powerful local compute be treated as abundant. For new research, this supersedes the earlier
handoff-only restriction on code changes and compute expansion. Prioritise predictive quality, reproducibility
and robust evaluation over local compute cost. Use CPU and GPU concurrently when useful, measure available
memory, and freeze recipes/seeds/gates before looking at their scores. The machine has a 20-core Core Ultra 7
255HX and an RTX 5070 Ti Laptop GPU with 12 GB VRAM. Compute is not a substitute for independent data.

Two registered research rounds tested 13 recipes and their 13 fixed baseline blends. Neither round passed
discovery. Preserve the exact unsearched baseline; no confirmation was scored and no new candidate was promoted.
Protocols: `docs/RESEARCH-01.md`, `docs/RESEARCH-02.md`. Complete results: `handoffs/results/research01-*` and
`handoffs/results/research02-*`. These are historical selection estimates. New research remains separate from
platform uploads, live submissions, model-slot changes, staking and any separately specified M7 refit.

The user subsequently renewed the major-improvement request and explicitly suggested much longer training.
Campaigns 03/04 used sustained nested searches, complete long neural schedules and target-free representation
learning. Their best observed historical FIT gain was 11.5%, with a material loss in the latest window; no
candidate qualified for promotion. The 15% major-improvement threshold was the agent's predeclared operational
criterion, not a promise from or a threshold supplied by the user. The requested reliable major gain remains
unachieved. Preserve all failed/positive evidence and do not relax completed gates to manufacture a success.

The subsequent stability05 experiment narrowed the latest-window loss to 2.51% and raised the observed average
gain to 12.89%, but missed its corrected gate and major-improvement criterion. The incumbent is retained.
Do not describe the largest historical average as a verified future gain or substitute it for the unmet goal.

## Standing rules (unchanged)

- The user explicitly requires investigation of promising models that underperform. Before retiring an idea,
  distinguish implementation/numerical errors, optimization failure, representation limits, ensemble or
  objective mismatch, overfitting and temporal shift. Use matched controls where possible; retain contradicted
  hypotheses and adverse windows, and mark unresolved causes honestly. A weak score alone is not an explanation.
  Register ensuing repairs before scoring, preserve old artifacts, and do not relax a completed gate.

- Nothing is submitted, uploaded, or staked without a handoff carrying a **USER MUST APPROVE** banner.
- `.env`, `data/`, `artifacts/`, `models/*.pkl` are never committed. Pre-commit hook blocks secrets.
- Never fit to, blend with, or select on `v1_sherpa` (the train-split benchmark is in-sample).
