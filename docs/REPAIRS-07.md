# Cause-driven repairs 07

Frozen after the controlled forensics06/06b results, before repaired-algorithm scores. Do not reject an idea
solely from a weak aggregate: record the mechanism, change one supported cause, and retain the adverse evidence.
The count-transfer underfitting hypothesis was contradicted; median counts remain. Confirmed partial issues:
unequal raw-score scales, irreversible projection loss, and dilution of a useful denoised representation.

Two fixed algorithms in `configs/repairs.yaml`:

1. Preserve raw inputs in every latent-tree trial and refit. Keep the same nested search budget, remaining
   parameter space and median count rule. Average each selected member's Gaussian rank score within a
   cross-section, so arbitrary positive score-scale changes do not alter equal-model weighting. Use the
   exact same two-stage prediction rule in discovery and portable inference.
2. Keep the denoising encoder architecture and schedule. Fit raw and latent prediction groups with separate
   group strengths and label-free total within-era variance normalization. Include a latent-only option and
   raw fallback; select group strength and ridge strength only on the purged inner target-validation split.
   Scale the ridge penalty by total covariance trace, so duplicating raw feature columns or padding zero
   columns cannot silently change the penalty. Refit the chosen head rule on the outer fitting partition.

Reuse a target-free encoder from representations04 only when its data hash, training boundaries, feature
order, normalization arrays and serialized bytes match the exact fitting partition. Discard its previous
prediction head completely. Inner encoders were not persisted and must be retrained with the same fixed
600-epoch target-free schedule; outer encoders can be reused. Confirmation/full-data encoders, if absent,
receive the full registered schedule. Record actual trained and reused epochs separately. Reconstruction
performance and a label-invariance check do not establish predictive performance.

Chronological readers, splits, bootstrap, effect/stability/worst-window gates and the operational 15% major
criterion remain fixed. Four outer comparisons: each algorithm and its fixed baseline blend. Freeze one
eligible winner before loading confirmation; no runner-up substitution. These are adaptive historical
research estimates, not independent future evidence or a globally controlled sequence of tests. Report all
repairs, including failures; do not relax a gate to claim a recovery. No old-audit rescoring or platform writes.
Only a positively confirmed candidate can be packaged, and its own bytes must pass clean-interpreter checks.
