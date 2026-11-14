# Matched projection rescue

Follow-up to the completed forensics06 controls, registered before the new intervention scores. The late
latent-tree models used only 16 PCA coordinates. Their retained mean linear feature/target-covariance vector
fraction fell from about 83% in fit history to about 50% in the test window. This does not measure all
predictive information or establish causation. More trees made the loss worse, so a count-only fix is unsupported.

Controlled intervention: keep the learned PCA, context rule, selected tree parameters, data, target centring,
weights, seed and original median count. Preserve the original raw Gaussian-bin inputs alongside the latent
coordinates. For early/middle models that already preserve raw inputs, verify identical output as the control.
Refit only the late members whose input availability changes. Compare original and rescued predictors with
both registered raw/rank averaging rules and fixed baseline blends. Do not choose new parameters or counts
using the test labels. Write to a new forensic artifact directory; preserve old caches and all adverse results.

This tests whether the irreversible projection contributes to the late failure. It is historical causal
diagnostic evidence, not independent model promotion. No confirmation/audit labels or platform writes.
