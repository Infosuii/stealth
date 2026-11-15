# Prequential model-family weighting, round 14

Registered after GPU ranking13, before any online-blend scores. GPU regression was weak early and strong
late in both seeds, while the prior stable group lost late. Fixed averaging diluted aggregate performance.
Test whether learning weights from genuinely earlier OOS outcomes can exploit complementary model families.
Do not switch retrospectively at an observed regime boundary or select the better seed.

Use five frozen components: refreshed baseline; temporal11 stable group; two-seed GPU RMSE, QueryRMSE and
YetiRank groups from ranking13. Gaussian-rank every component within each era. Discovery models are copied
only after source-plan/data/code/file verification; no native model is retrained or selected for discovery.

Every 100-era prediction block, fit nonnegative ridge coefficients to prior OOS prediction/target covariance.
Calibration uses only eras strictly before block_start-40, within the last 512 calendar eras, exponentially
weighted with 256-era half-life and equal mass per era. At least 200 distinct fitting eras are required.
Before that, use the fixed prior: half baseline, half stable group, zero GPU weight. Ridge strength is 1 in
Gaussian-prediction covariance units; shrink toward that prior's nonnegative optimal amplitude estimated on
the available history only. Normalize nonnegative coefficients to sum one, then anchor 25% to the baseline.
All choices are fixed now, not tuned on this campaign. Signed unconstrained weights and leverage are excluded.

Two learning objectives: centered target covariance (FIT direction); full FIT+UNQ+INOV linear utility using
the matured era's honest OOS baseline and all-feature core proxy, including both negative-reference zero
rules. The utility is a directional surrogate for Gaussian scores, not the exact rank/arctan objective.
Calibration references must be complete; the final evaluation still uses all six proxy scenarios. No fitting
operation, including scaling, covariance, prior amplitude or normalization, can read unmatured labels.

A fixed control has exactly the same anchored prior weights: 0.625 baseline and 0.375 stable group. Three
primary comparisons (two adaptive policies plus fixed control) versus the same refreshed baseline, without
post-neutralization. Retain the existing Holm, FIT/score effect floors, p_pos, window guards and six-scenario
requirements. Additionally require each adaptive policy's aggregate FIT and score to exceed the fixed
control. Changing this new family does not alter any prior failed gate or supply global repeated-test control.

Freeze at most one winner before confirmation. If eligible, train only its component dependencies in new
online14 caches on the unchanged sequential confirmation schedule (5340–5924). Calibration starts with all
available discovery OOS history and then incorporates only matured earlier confirmation predictions. Gaps
without native predictions contribute no calibration observations. Never reread eras >=5925. These historical
confirmation eras were viewed in older project work, so this remains round-locked evidence, not independent
future proof. No automatic deployment on proxy evidence.

Verify future-label invariance: permuting any labels at/after a block's availability cutoff cannot change that
block or earlier predictions. Check exact prior behavior, nonnegative normalized weights, baseline anchoring,
and covariance-solver agreement. Report all policies, warmup and weight changes; no oracle family switch or
rejection of weak components after scoring. Save source-bound private traces. Baseline01 stays unchanged.
No platform calls, uploads, submissions, staking or `.env` access.
