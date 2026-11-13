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

## Standing rules

- Nothing is submitted, uploaded, or staked without a handoff carrying a **USER MUST APPROVE** banner.
- `.env`, `data/`, `artifacts/`, `models/*.pkl` are never committed. Pre-commit hook blocks secrets.
- Never fit to, blend with, or select on `v1_sherpa` (the train-split benchmark is in-sample).
