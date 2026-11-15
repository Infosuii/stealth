# Support-constrained state conditioning

adaptive historical research; discovery and transfer both viewed

| interval | policy | FIT | proxy score | historical gates |
|---|---|---|---|---|
| discovery | daily | 0.02616325 | 0.03956263 | True |
| discovery | prediction_state | 0.02613311 | 0.03947658 | True |
| discovery | feature_state | 0.02618303 | 0.03960918 | True |
| viewed_transfer | daily | 0.01286781 | 0.01439370 | False |
| viewed_transfer | prediction_state | 0.01278598 | 0.01419802 | False |
| viewed_transfer | feature_state | 0.01288537 | 0.01433111 | False |

Native models and previous gates are unchanged. No promotion or platform writes.

The support constraint worked numerically: all recorded support ratios were >=1 within floating-point
tolerance. Minimum nominal ESS rose to 146.74 prediction / 146.66 feature; mixture mass was reduced on
48 of 1,645 calibrated prediction dates in each variant. The unconditional policy replayed context15.

It did not repair transfer. Prediction-state FIT moved from 0.01279259 to 0.01278598; feature-state FIT
from 0.01284852 to 0.01288537, both below baseline 0.01498275. Fixing the demonstrated support collapse
therefore does not establish useful state predictability or solve the family reversal. The concentration
problem was real, but a pure support-collapse explanation of failure is contradicted by this intervention.
The prior unstable state/value associations remain relevant; richer state information is unresolved.

Two new support/maturity tests passed. Original states, targets, native bytes, time weighting and gates
were preserved under source 614dd3d. The earlier full suite passed 489 tests; subsequent native-target
work gets its own tests and final suite. Both intervals are viewed research data, never fresh confirmation.
