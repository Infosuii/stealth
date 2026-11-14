# Campaign 03 — sustained optimization and long training

historical development selection estimates; not a fresh holdout

| recipe | blend | FIT | baseline FIT | gain | p_pos | Holm pass | eligible |
|---|---|---|---|---|---|---|---|
| calibrated_ridge | True | 0.024017 | 0.021811 | +0.002207 | 0.9718 | False | False |
| calibrated_ridge | False | 0.023970 | 0.021811 | +0.002160 | 0.8509 | False | False |
| long_mse | True | 0.023329 | 0.021811 | +0.001518 | 0.8895 | False | False |
| long_multi | True | 0.023234 | 0.021811 | +0.001423 | 0.8788 | False | False |
| nested_trees | True | 0.023113 | 0.021811 | +0.001303 | 0.9272 | False | False |
| long_multi | False | 0.022439 | 0.021811 | +0.000628 | 0.6163 | False | False |
| long_mse | False | 0.022410 | 0.021811 | +0.000600 | 0.6079 | False | False |
| nested_trees | False | 0.021759 | 0.021811 | -0.000052 | 0.4902 | False | False |

Frozen discovery winner: None

Confirmation:
```json
{
  "promoted": false,
  "major_improvement": false,
  "reason": "no eligible discovery challenger",
  "plan_sha256": "1756906af970219193bac5f895b5a38eb621c9c10ea681968ecf80dc8db122d7"
}
```

No old audit eras, platform writes or staking. No candidate is an independently proven live improvement.

## Measured result

Best discovery result: calibrated_ridge with the fixed baseline blend, FIT 0.02401723 versus 0.02181070, a gain of 10.12%. The paired p_pos is 0.9718; the null-centred p-value is 0.02700. It did not pass Holm correction across eight comparisons. The major-improvement target was not met. No candidate was promoted, and confirmation labels were not loaded or scored.

Completed 192 tree-search trials, each evaluated on three purged inner chronological windows (576 inner fits), plus refits and baseline fits. Completed all 18 neural seed schedules, totaling 10,800 inner epochs; final refits used only their inner-selected checkpoint counts. Several seeds selected epoch zero, so their exported primary predictor remains the ridge skip. Longer supervised fitting substantially reduced training error while often degrading inner FIT. Full curves and checkpoints remain private under artifacts/campaign03.

All 15 saved fold ensembles (33 serialized models) reproduced their predictions. Pure and baseline-blended portable inference passed rank parity, shuffled-feature-column and reversed-row checks. Full integration suite: **442 passed, 1 skipped in 143.38s**; the skip is the no-device failure path on a CUDA-capable host. No platform uploads, live submissions, model changes or stakes occurred.

The data reader excluded eras 5300 onward throughout discovery and always excluded the old audit. Plan hash: `1756906af970219193bac5f895b5a38eb621c9c10ea681968ecf80dc8db122d7`. Protocol commit: `3aadfbc`; implementation: `a8a3d00`. Training ran concurrently on CPU and GPU, 10:23:17–10:34:45 UTC. The observed 10.1% gain is exploratory historical evidence, not independent future performance.
