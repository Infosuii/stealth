# Representation campaign 04 — measured result

historical development selection estimates; not a fresh holdout

| recipe | blend | FIT | baseline FIT | gain | p_pos | Holm pass | eligible |
|---|---|---|---|---|---|---|---|
| denoised_ridge | False | 0.024319 | 0.021811 | +0.002508 | 0.8818 | False | False |
| denoised_ridge | True | 0.024250 | 0.021811 | +0.002440 | 0.9807 | False | False |
| latent_trees | True | 0.021822 | 0.021811 | +0.000011 | 0.5002 | False | False |
| latent_trees | False | 0.019227 | 0.021811 | -0.002583 | 0.1063 | False | False |

Frozen discovery winner: None

Confirmation:
```json
{
  "promoted": false,
  "major_improvement": false,
  "reason": "no eligible discovery challenger",
  "plan_sha256": "51f70c2356cb1135823232c5e6120d77239f95e57cd46588607c6c1cf271af8f"
}
```

No old audit eras, platform writes or staking. No candidate is an independently proven live improvement.

## Measured gain and its limits

The best observed historical discovery FIT was 0.02431878, versus baseline 0.02181070: **+11.50%**. It did not meet the registered 15% major-improvement target or its corrected promotion gate. The baseline remains selected. No new deployable pickle was packaged.

| window | baseline FIT | denoised FIT | fixed blend FIT |
|---|---|---|---|
| early | 0.005826 | 0.011895 | 0.009661 |
| middle | 0.034366 | 0.037376 | 0.037693 |
| late | 0.027527 | 0.023264 | 0.026162 |

The standalone denoised model improved the first two windows but lost materially in the latest one. Its baseline blend gained 11.19% overall with p_pos 0.9807 and null-centred p-value 0.02270, but failed the four-comparison Holm gate and the registered worst-window bound. Confirmation labels were not loaded or scored; no runner-up was substituted. These are historically viewed selection estimates, not independent future evidence.

## Actual compute and verification

Completed 144 latent-tree search trials across three outer fits, each evaluated on three purged inner windows (432 inner tree fits). Nine denoising seed pairs each completed 600 target-free inner encoder epochs and 600 target-free refit epochs: **10,800 encoder epochs**. Full reconstruction curves are in the companion metrics JSON. A synthetic label-permutation check produced identical encoder weights; targets influence the ridge head, not feature learning.

All nine fold ensembles (21 serialized models) reproduced cached predictions. Actual portable inference passed rank parity, shuffled feature columns, reversed rows and cross-sectional-transform checks. Full suite: **448 passed, 1 skipped in 141.99s**, with the absolute GPU interpreter supplied; the skip is the expected no-device path on a CUDA-capable host. The actual latent-tree pickle also passed the clean-interpreter contract in regression coverage. No unpromoted artifact is claimed to have passed a final full-data deployment check.

Training ran concurrently on CPU and GPU, 10:48:16–10:57:24 UTC. Across campaigns 03/04, 336 search trials produced 1,008 inner tree fits, plus refits, calibrated paths and more than 21,600 GPU training epochs. Longer supervised training exposed overfitting; representation learning increased the best observed development gain from 10.1% to 11.5%. A reliable major improvement has not been established.

Protocol commit `f57037d`; implementation `3ad0a71`; plan hash `51f70c2356cb1135823232c5e6120d77239f95e57cd46588607c6c1cf271af8f`. Old audit eras were excluded. No platform uploads, live submissions, model-slot changes or stakes occurred. Private checkpoints remain in artifacts/representations04; weights and data are not committed.
