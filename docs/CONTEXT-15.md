# Current-state conditioning and daily updates, round 15

Registered after online14r's failed confirmation, before any new policy scores. Learned weights lagged a
reversal in model-family value; initial GPU exposure hurt, and the baseline eventually recovered most
weight. This does not establish the economic cause of the reversal. Test two possible mechanisms separately:
infrequent updates and absence of current, label-free state information.

Reuse the five frozen component groups and exact native model bytes from online14r. Preserve the original
40-era outcome-availability delay, 512-era calibration history, 256-era half-life, ridge 1, minimum 200 eras,
half-baseline/half-stable prior and 25% baseline anchor. Use the repaired complete-case utility and unchanged
bounded evaluation metrics. No new native model, target, seed, mixture group or hyperparameter search.

Three fixed policies:
1. Daily unconditional score-utility covariance/ridge updates. This isolates 100-era versus daily updating.
2. Daily updates with prediction-state similarity: four component correlations to baseline, three within-GPU
   seed rank-Gaussian correlations and mean cross-component prediction dispersion. Every value is computed from the current
   cross-section only. Constant predictions have correlation zero.
3. Daily updates with feature-state similarity: log instrument count, missing fraction, leading covariance
   spectral shares, spectral entropy, off-diagonal correlation mean/dispersion and mean-feature variance.
   Fill missing feature values with their fixed bin midpoint only for state construction and explicitly
   retain missingness. Center/scale within that cross-section; no future cross-sections fit a transform.

Conditioned calibration weights mix 50% ordinary time-decay mass with 50% time-decay times a Gaussian state
kernel. Kernel width is 1 using mean squared standardized state distance; center/scales use only matured
past states, with a numerical scale floor 1e-6. Normalize kernel weights after subtracting the maximum log
weight. This is a fixed similarity heuristic, not an economic-regime identifier or a guarantee of reliability.
Report effective calibration eras and state distances. Daily unconditional uses exactly the same engine,
with context weighting disabled. No state dimensions, bandwidth, history or thresholds are selected by scores.

Compute target-dependent moments incrementally only when an era matures. Future-label and future-state
permutations must leave earlier predictions unchanged. Native models use their original rolling availability
rules; only the daily blend is new. Missing-state and constant-state behavior must remain finite and causal.

The discovery interval remains 4000–5299; the transfer interval is 5340–5924. BOTH HAVE NOW BEEN VIEWED.
Report this explicitly as adaptive historical research, never fresh confirmation or prospective evidence.
Keep native-prediction gaps empty: 5300–5339 have no OOS calibration observations. Compare three policies to
the equally refreshed baseline with the existing effect/window/bootstrap/Holm rules; show the original
adaptive_score and fixed-prior policies as descriptive controls, not eligible replacement candidates.
Also report all three policies' viewed transfer results, without choosing a winner on that interval or
promoting any model on it. Previous failures and gates remain unchanged. Additional earlier-period OOS
replication or genuinely prospective labels would be needed before a deployment claim.

Diagnose whether increased update frequency helps, whether similarity adds anything beyond frequency, and
whether prediction agreement or feature state has stable association with relative model performance.
Keep adverse periods and failures. Exact discovery replay of the source controls is required. Bind config,
sources, source plans and native file bytes. Exclude eras >=5925 at every reader. Never access `.env` or make
platform writes, uploads, submissions, model-slot changes or staking. Baseline01 remains unchanged.
