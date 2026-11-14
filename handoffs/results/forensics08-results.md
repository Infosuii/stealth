# Why the nested searches selected weak later models

Every completed original trial was refitted on its original outer training partition with its recorded parameters and median inner count. No new search was run. All nine original selected ensembles reproduced their cached predictions. Source studies were read-only and hash-checked. Only historical discovery labels below 5300 were read.

| family | window | trials | inner/later rank correlation | median individual FIT | selected individual FIT mean | unavailable oracle max FIT |
|---|---|---|---|---|---|---|
| campaign03 | early | 64 | 0.022 | 0.005782 | 0.006065 | 0.012159 |
| campaign03 | late | 64 | -0.274 | 0.025550 | 0.024272 | 0.029497 |
| campaign03 | middle | 64 | -0.039 | 0.032260 | 0.030154 | 0.037732 |
| repairs07 | early | 48 | -0.512 | 0.002812 | 0.003391 | 0.009031 |
| repairs07 | late | 48 | -0.335 | 0.026335 | 0.022487 | 0.029252 |
| repairs07 | middle | 48 | -0.157 | 0.029359 | 0.022361 | 0.033288 |
| representations04 | early | 48 | -0.065 | 0.003360 | 0.008188 | 0.012777 |
| representations04 | late | 48 | -0.420 | 0.022154 | 0.020385 | 0.029615 |
| representations04 | middle | 48 | 0.106 | 0.028466 | 0.028535 | 0.032585 |

## Interpretation

The inner ranking was negative in seven of nine cases and only weakly positive in the remaining two. In several windows, the selected individual members were below the median later performer even though stronger candidate models existed. This supports a model-selection/generalization problem in addition to the representation defects found earlier. It contradicts the assumption that increasing search budget or choosing the highest inner score automatically produces a better next-period model.

This does not prove that the model class is universally good or that all underperformance comes from selection. In the repaired latent-tree middle window, even the best individual trial fell below the baseline. Model populations also share data and correlated forecasts; the table is descriptive, not a naive significance test across independent trials. Individual score means are not ensemble FIT.

The oracle maximum is strictly diagnostic. Choosing those trial IDs using later labels would leak the test period and is not an authorised repair. No oracle-selected model was packaged or promoted. A supported next test is to soften or remove winner-take-all selection, using an algorithm registered before computing its ensemble scores and the unchanged chronological gates.

A total of 480 matched full-partition refits completed. Detailed trial scores remain private under artifacts/forensics08. Final suite at this checkpoint: **459 passed, 1 skipped in 124.08s**, with the expected no-device path skipped on a CUDA-capable host. Protocol and implementation: 3679880. No confirmation or audit read, uploads, submissions, model-slot changes or staking.
