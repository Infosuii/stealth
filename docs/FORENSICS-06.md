# Underperformance investigation 06

The user explicitly requested investigation and repair of promising ideas, rather than discarding them after
a weak score. Preserve every completed result. This investigation uses only the historical discovery eras
below 5300. It does not open round-locked confirmation or the old audit, alter old caches, change completed
gates, or make any platform write.

Register controlled diagnostics before calculating their new held-out scores:

1. Verify saved model bytes reproduce their original cached predictions. Quantify per-model learning curves,
   train/validation divergence and selected checkpoint behaviour for the supervised networks.
2. For the denoising predictor, keep all learned weights frozen and separately score the raw-feature term,
   encoder term and complete predictor, both directly and in the same fixed baseline blend. Measure which
   term loses in the late window. These are mechanistic ablations, not new independent promotion evidence.
3. For both raw-feature and latent-tree searches, retrieve each selected trial's actual inner count vector
   from its immutable SQLite record. Compare the original median-count refit with a refit using the latest
   chronological inner count, holding parameters, data, weights, seed and latent transform fixed.
4. For both count choices, compare the original raw-score ensemble with an equal per-cross-section Gaussian
   rank ensemble. Record each model's train and test prediction scale, effective raw-average dominance,
   actual fitted tree count and individual FIT. A lower mean does not by itself establish the cause.
5. Measure feature/target covariance changes between fit history, recent fit history and the outer test;
   distinguish label-relationship change from missingness/marginal-feature changes. Project covariance
   vectors into retained PCA directions to assess discarded linear covariance, labelled as that quantity
   rather than a claim to measure all predictive information.

Every diagnostic will be reported, including failed repair hypotheses and adverse windows. Compare one change
at a time and their registered combination. A repair is supported only where its controlled ablation improves
the relevant failure, with explicit uncertainty and temporal caveats. Do not select parameters from an outer
diagnostic or declare a globally validated major gain from repeatedly viewed data. Any ensuing model campaign
needs a new source-bound ID, frozen algorithm and existing chronological gates. Raw parameters, weights and
data stay private; public reporting carries metric/diagnostic summaries and config names.
