# Handoff 06 — research bot upload

Status: **ready for the authorized upload; no platform write yet**.

Selected experimental policy: history18 CPU expanding-from-3, applying the selection rule committed before
utility19 scores. Its viewed-transfer FIT is 0.01819329 vs baseline0.01526832 (+19.16%); proxy0.02531301 vs
0.01868202 (+35.49%). Earlier FIT is worse (0.01402049 vs0.01690444), and development does not pass.
No broadly qualified improvement is claimed. Utility19's stronger recent GPU maximum also fails earlier
periods; it did not replace the frozen upload fallback. Baseline01 remains preserved locally.

The historical figures describe rolling source models, **not a measured score of the final full-data refit**.
The final model uses all 468,236 provided training rows, eras3–6524, with a frozen count86 from the median
of source counts[86,231,39] and CPU seed17. Later training labels chose no count, weights, target or policy.
There was no post-refit audit. All research readers remained below5925.

Artifact: `models/research06.pkl`, **142,069 bytes**, SHA256
`50c1301f73605c48067cbed6ee5789b341afe0c1e3cd3ea2cd5164901ae05a48`. Feature order, raw/core target handling and per-era rank aggregation were checked against
300 source eras with zero error. Final native serialization replay error was zero.

`pkl_check: PASSED` in the isolated exact-pin Python3.11.18 lane venv, 51 eras, slowest call
0.4134s, no stealth/torch importability, no checker problems. The upload
artifact gate passed for these exact bytes. Offline prediction passed on90 live-feature rows (exped8055),
90 distinct finite predictions strictly inside(0,1). Full suite before the packaging correction:500 passed;
its targeted regression/upload tests:5 passed. The final full suite passed **501 tests without skips** in110.03s.

The concrete existing slot is `f06a70bf-ef8b-483d-b2a9-d4b9ef3e7043` (`mandelbrot-bayesian-payoff`).
Auto-submit was re-read as OFF. The user explicitly authorized this upload; no further approval question.
The one-shot receipt will be persisted before sending and the returned ID before polling. No direct
submissions, diagnostics uploads, auto-submit toggles, staking, model creation or renaming.

Operational correction: the first packaging attempt hit pandas read-only NumPy memory before fitting or
writing any bot; fd4e68a adds an owned copy and regression test. The repeated successful package used the
same frozen choice and count. Preparation source1d7e2b4; history report0bc8a89; utility19 reportfc56cca.

Upload ID, remote hash verification and final platform state will be added after the single upload.
