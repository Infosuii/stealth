# Daily and observable-state weighting 15

adaptive historical research; discovery and transfer both viewed

| interval | policy | FIT | baseline FIT | proxy score | baseline proxy | historical gates |
|---|---|---|---|---|---|---|
| discovery | daily | 0.02616325 | 0.02429122 | 0.03956263 | 0.03376212 | True |
| discovery | prediction_state | 0.02613330 | 0.02429122 | 0.03947700 | 0.03376212 | True |
| discovery | feature_state | 0.02617538 | 0.02429122 | 0.03958763 | 0.03376212 | True |
| viewed_transfer | daily | 0.01286781 | 0.01498275 | 0.01439370 | 0.01753069 | False |
| viewed_transfer | prediction_state | 0.01279259 | 0.01498275 | 0.01429061 | 0.01753069 | False |
| viewed_transfer | feature_state | 0.01284852 | 0.01498275 | 0.01426094 | 0.01753069 | False |

Source block-boundary weight replay max error: 6.66134e-16.

Native models are source-verified copies, not new fits. Label/state maturity guards are unchanged.
No model was promoted. Baseline01 stays selected; no platform writes, uploads, submissions or staking.

The frequency control did not repair transfer: daily FIT 0.01286781 vs the original 100-era controller's
0.01287388. First-half loss narrowed from -0.00395822 to -0.00350157, but second-half loss worsened from
-0.00025317 to -0.00072357. Faster updates alone therefore did not solve this reversal.

Conditioning added little in discovery and slightly reduced transfer FIT. Agreement was not a stable
reliability indicator in these data: the descriptive correlation of GPU RMSE seed agreement with relative
FIT was -0.2868 early, -0.0758 middle, +0.0759 late and -0.0052 in viewed transfer. Feature-factor and
baseline-agreement associations also varied. These uncorrected descriptive associations are not economic
causal evidence, do not exhaust useful state information, and were not used to select a state dimension.

One implementation-level modeling issue is supported: similarity weighting sometimes concentrated its
nominal calibration mass severely. Minimum effective era counts were 4.68 for prediction state and 27.58
for feature state versus 195.25 for unconditional time weighting. These are weight-based effective counts,
not independent observations; serially overlapping outcomes reduce information further. Mixing 50% ordinary
time weights did not guarantee adequate effective support. This motivates a separately registered support
constraint while keeping all original scores, failures and thresholds unchanged. It does not prove this
concentration is the only or dominant reason for failure.

Validation: four new state/maturity tests passed; full suite 489 passed with no skips. The new daily engine
replayed source weights at original update boundaries to 6.7e-16. No new native model was trained. Source
58c309a. Both intervals were already viewed; there is no fresh confirmation or deployment qualification.
