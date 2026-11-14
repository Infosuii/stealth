# Stability campaign 05 — measured result

historical development selection estimates; not a fresh holdout

| recipe | blend | FIT | baseline FIT | gain | p_pos | Holm pass | eligible |
|---|---|---|---|---|---|---|---|
| stable_covariance | True | 0.024621 | 0.021811 | +0.002810 | 0.9679 | False | False |
| adaptive_covariance | True | 0.024619 | 0.021811 | +0.002808 | 0.9716 | False | False |
| adaptive_covariance | False | 0.023747 | 0.021811 | +0.001937 | 0.7591 | False | False |
| stable_covariance | False | 0.023662 | 0.021811 | +0.001851 | 0.7422 | False | False |

Frozen discovery winner: None

Confirmation:
```json
{
  "promoted": false,
  "major_improvement": false,
  "reason": "no eligible discovery challenger",
  "plan_sha256": "e0d8b8277d05fa968014f18b96563b07e4bd80091747110017e5778ca77053b1"
}
```

No old audit eras, platform writes or staking. No candidate is an independently proven live improvement.

## Result and decision

The fixed stable-covariance/baseline blend reached discovery FIT **0.02462116**, versus **0.02181070**, an observed **12.89%** improvement. Its p_pos was 0.9679 and null-centred p-value 0.02960. It missed Holm correction across the four comparisons and the registered 15% major-improvement target. No candidate was promoted; confirmation labels were not loaded or scored. No new deployment pickle was packaged.

| window | baseline FIT | stable blend FIT | relative gain |
|---|---|---|---|
| early | 0.005826 | 0.009068 | +55.65% |
| middle | 0.034366 | 0.038845 | +13.03% |
| late | 0.027527 | 0.026837 | -2.51% |

This improves the historical average and narrows the latest-window shortfall, but does not establish a reliable major improvement. The last window still favours the baseline. These data were already viewed and used in adaptive research; no global false-positive guarantee or independent future performance is claimed. The original performance objective remains unfinished.

The two algorithms evaluated 2,835 inner coefficient paths, using direct regularized matrix solutions and the three purged chronological inner windows. All nine saved fold ensembles replayed exactly, and pure/blended runtime predictions passed rank, feature-order and row-permutation checks. Final full suite: **451 passed, 1 skipped in 120.04s**; the skip is the no-device failure path on the CUDA-capable host. Training ran 11:09:40–11:09:54 UTC.

Protocol `c14cdb9`; implementation `21da749`; plan hash `e0d8b8277d05fa968014f18b96563b07e4bd80091747110017e5778ca77053b1`. Private checkpoints and selected settings are under artifacts/stability05. No old audit was rescored. No uploads, live submissions, model-slot changes or stakes occurred. Reports for campaigns03/04 preserve the longer tree/neural work and all earlier failures.
