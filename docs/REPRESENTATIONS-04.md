# Representation campaign 04

Registered after campaign03, before fitting this campaign. Campaign03 improved historical discovery FIT by
10.1%, but did not pass its corrected gate or 15% major-improvement target. Its confirmation was never loaded.
Its supervised networks overfit despite complete long schedules. Do not relax those completed gates.

Test two complete training algorithms that learn feature structure before predicting targets:

* Nested latent-tree optimization: fit target-free within-era PCA on each inner training partition; optimize
  latent dimension, inclusion of original inputs, label-free cross-sectional normalization, and tree capacity,
  regularization and learning rate. Three purged inner windows, 48 trials per outer fit, top-three refits.
* Masked denoising encoder followed by a calibrated ridge head. Train encoders against feature reconstruction
  for 600 epochs, with artificial missingness and an information bottleneck. Use no target in encoder updates.
  Select ridge strength on a purged chronological inner target-validation partition. Refit encoder and head
  on the outer fit partition, with the same frozen schedule and selected dimensionless penalty. Three fixed
  seeds. Export only encoder and primary head to the existing NumPy neural runtime.

All PCA/encoder fits and preprocessing statistics belong to their training partition. Cross-sectional
normalization uses available features only, separately by era in training/evaluation and by the live frame
at inference. Test the actual per-cross-section portable prediction, not a different offline transform.

Discovery, reader exclusions, one-winner confirmation, major-improvement definition, bootstrap and effect
gates follow `configs/representations.yaml`. Two algorithms tested standalone and in fixed baseline blends
give four corrected comparisons. These are adaptive historical development results, with no global error-rate
guarantee or untouched future holdout. A major claim requires at least 15% paired mean-FIT lift in discovery
and locked confirmation, plus both gates. If unsuccessful, report it and preserve the incumbent. No platform
upload, live submission, model-slot change, old-audit rescoring or staking. Do not substitute a runner-up after
confirmation. A successful candidate must pass actual serialized-model parity and clean-interpreter checks.

Any runtime extension occurs after campaign03 finished; replay that campaign at its recorded implementation
commit, since its source-bound plan intentionally refuses later source changes.
