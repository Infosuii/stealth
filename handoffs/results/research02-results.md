# Research round 02 — training objective and stopping

historical development selection estimates; not a fresh holdout

| recipe | blend | FIT | baseline FIT | gain | p_pos | Holm pass | eligible |
|---|---|---|---|---|---|---|---|
| era_bag | True | 0.021777 | 0.021811 | -0.000033 | 0.4936 | False | False |
| era_bag | False | 0.021350 | 0.021811 | -0.000461 | 0.3768 | False | False |
| era_mse | True | 0.021315 | 0.021811 | -0.000496 | 0.0054 | False | False |
| era_fit | True | 0.021268 | 0.021811 | -0.000543 | 0.1776 | False | False |
| fit_stop | True | 0.020822 | 0.021811 | -0.000989 | 0.0414 | False | False |
| era_mse | False | 0.020726 | 0.021811 | -0.001085 | 0.0028 | False | False |
| era_fit | False | 0.020285 | 0.021811 | -0.001526 | 0.1008 | False | False |
| fit_stop | False | 0.019594 | 0.021811 | -0.002217 | 0.0244 | False | False |

Frozen discovery winner: None

Confirmation:
```json
{
  "promoted": false,
  "reason": "no candidate passed discovery gates; baseline retained",
  "plan_sha256": "439e48dee94a1126e9234cbac0d08c9b9556adbf20f1544b85ee6344bc5fe8fe"
}
```

No old audit eras, platform writes or staking. No candidate is an independently proven live improvement.

## Verification and final model decision

Baseline discovery FIT: 0.02181070. Best challenger: era_bag with the fixed baseline blend, FIT 0.02177738, paired gain -0.00003332, p_pos 0.4936. No candidate passed discovery. Confirmation was not scored; the packaging command retained the existing baseline and exported no new pickle.

All 15 saved fold ensembles reproduced their cached predictions exactly, with finite outputs. Combined with round 01, serialization parity passed for 45 fold ensembles (57 individual serialized models). The full final suite ran with the absolute GPU interpreter supplied: **438 passed, 1 skipped in 111.26s**. The skip is the expected no-device failure path on a host with a working GPU. No failures occurred in this round.

This round is an adaptive follow-up after the first failed experiment. Its corrected discovery statistics are exploratory and its historical confirmation would not constitute independent future evidence. Across both rounds, 13 fixed recipes plus their 13 baseline blends failed to establish a gain. The exact unsearched baseline01 artifact remains the conservative candidate. No platform upload, live submission, model-slot change or stake occurred.

The objective protocol was frozen in 7162318 and its implementation in 0f77fff. CPU fits ran two at a time, with eight threads per fit. The complete objective round finished in about 23 seconds of training; the earlier neural round used the RTX 5070 Ti and its registered three seeds. The user's standing preference for extensive powerful local compute is recorded in docs/DECISIONS.md and docs/NEXT-SESSION.md. Additional compute was used for convergence, model diversity, chronological comparisons, serialization parity and the full integration suite.
