# Stability campaign 05

Registered after the complete 03/04 reports, without loading confirmation labels. Longer supervised models
overfit; denoising improved the historical average by 11.5% but lost in the latest discovery window. Test a
different training objective: emphasize feature signals that are stable through time, and estimate recent
signal changes using only past labels. This is a new adaptive historical research round, not a relaxation of
any previous gate and not a globally independent test.

Two frozen learning algorithms in `configs/stability.yaml`:

* Stable covariance: calculate equal-era, within-era feature/target covariance vectors on training rows.
  Estimate their covariance across contiguous 20-era blocks. Apply diagonal shrinkage and a positive ridge
  path to obtain regularized signal weights, jointly choosing history and regularization on the three inner
  chronological windows.
* Adaptive covariance: estimate exponentially weighted recent feature signals, optionally extrapolating a
  damped linear trend to the registered horizon. Apply the same regularized covariance objective. Choose
  half-life, damping, covariance shrinkage and strength only on the three inner windows.

Standardize feature scale using within-era feature variance from the fitting partition; map coefficients back
to the existing portable Gaussian-bin linear form. No prediction-time target, era ID, state or mutable history.
No new encoder training is needed. Use CPU parallelism for independent outer fits and exact matrix paths.

Readers, outer splits, inner purge, bootstrap, worst-window/effect/stability gates, four-comparison Holm
correction and the 15% operational major-improvement criterion are frozen in the config. Test each algorithm
alone and in the same fixed baseline blend. Freeze one eligible winner before any confirmation read; no
runner-up substitution. Preserve the incumbent if unsuccessful. Report the whole result, then stop this
campaign. No old-audit rescoring, platform writes, live predictions, model-slot changes or staking.
