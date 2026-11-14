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

## Standing rules

- Nothing is submitted, uploaded, or staked without a handoff carrying a **USER MUST APPROVE** banner.
- `.env`, `data/`, `artifacts/`, `models/*.pkl` are never committed. Pre-commit hook blocks secrets.
- Never fit to, blend with, or select on `v1_sherpa` (the train-split benchmark is in-sample).
