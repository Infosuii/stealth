# Research research01

historical development selection estimates; not a fresh holdout

| recipe | blend | FIT | baseline FIT | gain | p_pos | Holm pass | eligible |
|---|---|---|---|---|---|---|---|
| bin_shrink | True | 0.021586 | 0.021811 | -0.000225 | 0.4196 | False | False |
| gaussian_shrink | True | 0.020079 | 0.021811 | -0.001731 | 0.0822 | False | False |
| bin_shrink | False | 0.019481 | 0.021811 | -0.002330 | 0.1262 | False | False |
| smooth_additive | True | 0.019463 | 0.021811 | -0.002348 | 0.0598 | False | False |
| era_neural | True | 0.018735 | 0.021811 | -0.003076 | 0.0346 | False | False |
| bin_additive | True | 0.017898 | 0.021811 | -0.003912 | 0.0028 | False | False |
| gaussian_ridge | True | 0.017828 | 0.021811 | -0.003983 | 0.0040 | False | False |
| cubic_additive | True | 0.017789 | 0.021811 | -0.004022 | 0.0078 | False | False |
| ridge_residual | True | 0.017770 | 0.021811 | -0.004040 | 0.0042 | False | False |
| quadratic_sketch | True | 0.016951 | 0.021811 | -0.004860 | 0.0032 | False | False |
| gaussian_shrink | False | 0.016933 | 0.021811 | -0.004878 | 0.0196 | False | False |
| smooth_additive | False | 0.014483 | 0.021811 | -0.007327 | 0.0024 | False | False |
| era_neural | False | 0.012448 | 0.021811 | -0.009363 | 0.0010 | False | False |
| gaussian_ridge | False | 0.012145 | 0.021811 | -0.009666 | 0.0002 | False | False |
| ridge_residual | False | 0.011980 | 0.021811 | -0.009831 | 0.0002 | False | False |
| cubic_additive | False | 0.011002 | 0.021811 | -0.010809 | 0.0002 | False | False |
| bin_additive | False | 0.010957 | 0.021811 | -0.010854 | 0.0002 | False | False |
| quadratic_sketch | False | 0.008839 | 0.021811 | -0.012972 | 0.0000 | False | False |

Frozen discovery winner: None

Confirmation:
```json
{
  "promoted": false,
  "reason": "no candidate passed discovery gates; baseline retained",
  "plan_sha256": "4923bab5c3ed5962d7195f6bfc9987804a8ce25bcbc57ec7e8c084543e977286"
}
```

No old audit eras, platform writes or staking. No candidate is an independently proven live improvement.

## Verification and operator notes

All 27 CPU fits and all three GPU fold ensembles completed; each neural ensemble used the three registered seeds. All 30 serialized fold models reproduced their cached predictions (maximum absolute error 7.22e-16). The shared FIT kernel matched the SDK path, including missing/constant cases. Full suite: 430 passed, 3 skipped in 116.83s with the optional GPU interpreter unset; the seven GPU-training tests then passed with its absolute path supplied.

No candidate was eligible, so confirmation was not scored and no challenger pickle was exported. Baseline discovery FIT was 0.02181061; the best challenger blend reached 0.02158644, a negative paired gain of -0.00022417. These are selection estimates on historical development data.

The first background launch was cleaned up by its short-lived shell before training began. A persistent supervisor fixed orchestration. The first supervised CPU run hit its numerical iteration ceiling; that run remains private in artifacts/research01-solver-limit. After increasing only the solver iteration limit, every fit was rerun under a new source-bound plan before any discovery comparison. No failed fit was treated as a valid result.

Implementation commits: 71f481c (protocol), 8270334 (models and runtime), aefeb6e (numerical correction and compute preference). Hardware: 20-core Core Ultra 7 255HX; RTX 5070 Ti Laptop GPU, 12 GB VRAM. Model training ran concurrently on CPU and GPU.
