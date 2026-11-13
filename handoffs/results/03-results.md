# Results 03 — diag-upload-1

status: ok
git_sha: e69b848ecd804a901f5e76865843dcc551f278ff
pickle: models/m1_lgbm.pkl sha256 7ea75e30f8ca9ea85dae3f75b4cceee9ceb1a3cea9d9cb46493a21a4d46d94e6  pkl_check: PASSED  slowest call: 0.37641512299887836 s
validation file rows: 125931
upload_id: 35bfe266-286a-4464-a532-e4dca0d7899b
diagnostics status: done
fit20: 0.0189   unq: 0.0039   inov: 0.014054225   sharpe: 0.3237

Other panel metrics:
- std_dev: 0.0583
- max_drawdown: -7.5462
- autocorrelation: 0.9375
- feature_exposure: 0.6853
- example_preds_corr: 0.7334

Operator notes:
- Merged-code pytest: 249 passed, 3 skipped in 48.47s.
- Baseline: only lgbm_everest from test01; neutralisation off (p_b=0, p_c=0), exactly as handoff 03 specifies. The bundle's neutralisation override warning is expected for this calibration.
- Contract check: Python 3.11.18, 0.27 MB, 51 eras, torch=0, non-runtime stealth refs=0; missing feature column raises (ok).
- Validation generation: diag ok: 125931 rows, 1392 eras. bench_used=0 is expected with neutralisation off; the validation benchmark was supplied.
- Human explicitly approved diagnostics upload #1 in chat after step 4's question. No prior upload record existed.
- Exactly one upload to mandelbrot-bayesian-payoff; attached_pkl=False. Accepted at 2026-11-13T23:07:30+00:00; scoring done at 2026-11-13T23:08:38+00:00. Polling only after the upload ID was returned; no retries or duplicate uploads.
- Metrics copied from the saved upload record's panel. result.metrics is absent; no additional metrics were printed by diagnostics done.
- .env presence verified only with test -s .env; contents never displayed. No code/config edits, live submission, staking, or auto-submit setting changes. Pickles, predictions, and the upload JSON remain uncommitted.
