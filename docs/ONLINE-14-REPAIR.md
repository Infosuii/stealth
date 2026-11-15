# Technical missing-reference repair to the frozen online policy

Original online14 cleared discovery and froze adaptive_score, but confirmation failed before producing any
performance score. One legitimate row in exped_5582 has all 178 features set to -1. The strict calibration
reference check rejected its missing core mean. The row IDs are unique and source data/model hashes match;
this is genuine missingness, not index misalignment. No confirmation performance was inspected to choose
this repair. Preserve original artifacts and its failed-input report.

New run ID online14r. Keep every native model byte, seed, count, source data, policy choice, weight parameter,
metric, scenario and statistical gate. Do not retrain native models or select a runner-up. Extend only the
calibration utility's missing-core case to follow complete-case reference semantics: FIT and benchmark
utility remain on all instruments; core residual utility uses valid-core instruments, centered on that
subset, with N/M scaling for its different covariance denominator. Rows without a core reference receive
no core contribution, never a fabricated zero reference. Fewer than five valid reference instruments gives
no reference contribution. The benchmark must remain complete. The core's negative-reference zero rule is
evaluated on its valid subset. Prediction re-ranking and arctan still make this a directional surrogate;
the final scoring kernels, including exact complete-case INOV, are unchanged.

Verify exact complete-reference utility behavior and future-label invariance with partial core observations.
Recompute discovery using copied native models; every comparison and weight trace must replay online14 to
1e-12 and the frozen winner must remain adaptive_score. Only then resume the same historical confirmation
using the already trained, byte-verified confirmation components. A new run binds the repaired code and old
native caches. No original run ID or source file is overwritten; no gate, hyperparameter or decision is
changed after a performance failure. The original failure was an input-contract failure, not a performance
verdict. Record the initial GPU-context launch and premature-report errors as operator issues too.

If replay fails, stop this repair. If confirmation fails its existing gates, retain that result without
further policy changes. Historical confirmation remains previously viewed project data, not independent
future evidence. No deployment, uploads, submissions, staking or `.env` access.
