# Matched CPU target/count decomposition for objective16

Registered before objective16 aggregate results were read. Utility labels and their selected stopping counts
change together; a weak or strong score alone cannot identify which intervention contributed. Use the
completed objective16 CPU checkpoints without changing its native fits, comparisons or gates.

For each of its 39 blocks, retain original core-label/core-count and full-label/full-count predictions.
Refit full labels at the core donor's count, and core labels at the full donor's count, using the identical
final fitting rows, preprocessing, labels, teacher file, parameters, seed and six threads. Counts are each
source model's own purged-inner selections. Equal-count cases may reuse original bytes, with that reuse
explicitly recorded. First replay each target at its own count in the first block of each interval; require
native predictions to match within 1e-12 before interpreting any changed-count result.

Report all four cells for each historical interval, label effect at fixed count, count effect at fixed label
and their interaction. Nonlinear rank scoring means these are score-level factorial contrasts, not an
additive decomposition of model coefficients. GPU label fits have stochastic training, so do not call a
fresh stochastic refit an exact counterfactual; inspect their fixed-seed primary replicates separately.

This is descriptive diagnosis, never a new eligible arm, oracle count choice or candidate substitution.
No old gate is relaxed; all intervals remain historical (later intervals viewed). Bind source model markers,
teacher manifest, data and code. Exclude >=5925, retain every failure and perform no platform write, upload,
submission, staking or `.env` access.
