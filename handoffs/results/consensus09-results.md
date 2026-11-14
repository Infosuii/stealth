# Selection-policy repair 09 — broad consensus

historical development selection estimates; not a fresh holdout

| recipe | blend | FIT | baseline FIT | gain | p_pos | Holm pass | eligible |
|---|---|---|---|---|---|---|---|
| search_population | True | 0.023151 | 0.021811 | +0.001340 | 0.9342 | False | False |
| prior_population | True | 0.023142 | 0.021811 | +0.001332 | 0.9234 | False | False |
| search_population | False | 0.022784 | 0.021811 | +0.000973 | 0.7281 | False | False |
| prior_population | False | 0.022391 | 0.021811 | +0.000581 | 0.6404 | False | False |

Frozen discovery winner: None

Confirmation:
```json
{
  "promoted": false,
  "major_improvement": false,
  "reason": "no eligible discovery challenger",
  "plan_sha256": "9092490dad2b0f2ebc7a38f2b0dc106f784747f35b5331b380cea4d6b57740e2"
}
```

No old audit eras, platform writes or staking. No candidate is an independently proven live improvement.

## What this intervention established

The all-search-population predictor improved standalone tree FIT from the original selected-top-three value of 0.021759 to 0.022784. Its baseline blend reached 0.023151 (+6.14% versus baseline). The independent fixed-prior blend reached 0.023142 (+6.10%). Neither passed its promotion gates or major-improvement target. No confirmation labels were loaded and no artifact was promoted or packaged.

| window | policy | baseline FIT | blend FIT | gain |
|---|---|---|---|---|
| early | search_population | 0.005826 | 0.005945 | +0.000119 |
| early | prior_population | 0.005826 | 0.005524 | -0.000302 |
| middle | search_population | 0.034366 | 0.037834 | +0.003468 |
| middle | prior_population | 0.034366 | 0.038047 | +0.003682 |
| late | search_population | 0.027527 | 0.027355 | -0.000172 |
| late | prior_population | 0.027527 | 0.027664 | +0.000137 |

Removing sharp trial selection partially recovered the native-tree class. Removing adaptive proposals also narrowed the late-window loss, but weakened the early window. This supports selection instability as a contributor, not a complete explanation or a universal reason to average every model. It does not overturn the stronger 14.13% observed average gain from the balanced denoising repair, which still failed its temporal guard. The remaining temporal generalization problem is unresolved. No oracle trial was selected.

The search-population studies were cloned only after data/seed/window/inner-layout/budget checks; original database hashes were retained and source files were never modified. Prior proposals were frozen without labels before fitting. All 384 pooled model blobs plus three baseline blobs replayed across nine ensembles; rank rules, final blends, feature ordering and row permutation matched. Representative fold-pool bytes and latency were within lane bounds; this is not a check on an unbuilt final deployment artifact. Final suite: **461 passed, 1 skipped in 136.54s**, the skip being the expected CUDA no-device path.

Protocol 7046ada; implementation 90fabd1; plan `9092490dad2b0f2ebc7a38f2b0dc106f784747f35b5331b380cea4d6b57740e2`. Training 13:09:40–13:21:33 UTC. All data are historically viewed discovery data; no global independent significance claim. No audit rescoring, uploads, live submissions, model-slot changes or staking.
