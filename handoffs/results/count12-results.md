# Count-transfer controls 12

adaptive historical selection estimates; all composite metrics use proxies

| algorithm | core proportion | FIT | proxy score | FIT gain | proxy-score gain | Holm | eligible |
|---|---|---|---|---|---|---|---|
| count_recent | 0.25 | 0.02482201 | 0.04445058 | +0.00053078 | +0.01068846 | False | False |
| count_recent | 0 | 0.02479900 | 0.04417233 | +0.00050778 | +0.01041021 | False | False |
| guarded_full_span | 0 | 0.02497829 | 0.03824125 | +0.00068707 | +0.00447913 | False | False |
| guarded_full_span | 0.25 | 0.02485913 | 0.03805302 | +0.00056791 | +0.00429090 | False | False |
| count_full_span | 0.25 | 0.02183630 | 0.03736625 | -0.00245493 | +0.00360413 | False | False |
| count_full_span | 0 | 0.02177650 | 0.03716742 | -0.00251473 | +0.00340530 | False | False |
| count_spread | 0.25 | 0.02143106 | 0.03652108 | -0.00286016 | +0.00275896 | False | False |
| count_spread | 0 | 0.02140618 | 0.03633419 | -0.00288505 | +0.00257207 | False | False |

Control replay: {"source_plan_sha256": "218ed9a03aee83acac7022d89409608b67c3fa536763c0a4e6aaeec420c2d507", "prediction_max_error": {"baseline": 0.0, "score_rolling": 0.0}}

Confirmation: {"confirmation_scored": false, "plan_sha256": "604f5ebd67307d3642038362ed34f691f1e49170090439f3bca130ea126f0e0a", "qualified": false, "reason": "no eligible discovery candidate"}

81 model checkpoints replayed their actual serialized boosters; max error 0.

Components are descriptive, cannot replace a failed comparison, and were all retained in their registered groups.

| component | FIT | proxy score | early FIT | middle FIT | late FIT |
|---|---|---|---|---|---|
| baseline | 0.02429122 | 0.03376212 | 0.00997155 | 0.03852949 | 0.02442690 |
| score_rolling | 0.02506814 | 0.04417933 | 0.01117136 | 0.04078286 | 0.02203824 |
| ragged_control | 0.02267523 | 0.03983575 | 0.01018863 | 0.03644566 | 0.02053551 |
| count_spread | 0.02140618 | 0.03633419 | 0.00988874 | 0.03383442 | 0.01988817 |
| count_recent | 0.02479900 | 0.04417233 | 0.01133299 | 0.04079469 | 0.02058287 |
| count_full_span | 0.02177650 | 0.03716742 | 0.00920709 | 0.03481174 | 0.02100012 |

Baseline01 remains unchanged. No deployment, old-audit reads, uploads, submissions or staking.

Matched interpretation:

- The ragged three-window control reproduced temporal11 exactly, including its actual predictions (max error 0).
  Refreshed baseline and single-inner controls likewise reproduced temporal10 with zero error.
- At the same 384 fitting eras per inner source, recent contiguous history reached FIT 0.02479900 versus
  0.02140618 for evenly spaced history across the original available spans. The final fitting partition,
  validation windows and count-transfer rule were identical. Time coverage therefore changed downstream
  performance substantially at fixed era count. Instrument counts and selected observations still differ;
  this is not proof of an economic regime mechanism or an instrument-level sample-count effect.
- Giving every inner source the latest source's full 784-era history reached only 0.02177650, below the
  ragged control's 0.02267523. More earlier fitting history did not repair the failure. A pure "older folds
  just need more data" explanation is contradicted for this intervention; temporal relevance/composition
  remains a plausible contributor. Full-span versus recent also changes size and span, so those two alone
  cannot isolate a pure sample-size effect.
- The recent-history intervention recovered most of the rolling control's average but still lost late FIT
  by 0.00384403 vs the refreshed baseline. The fixed full-span/baseline blend bounded late loss to 0.00049467,
  but its FIT improvement was only 0.00068707 and it failed the corrected score gate. No candidate qualified;
  confirmation remained unread and all previous gates remain unchanged.

Validation: count/temporal/ranking targeted tests passed 16/16 before the GPU campaign. All 81 saved boosters
replayed exactly. Source implementation d9f3f52. Full inner stopping curves and source boundaries remain
private under artifacts/count12; no deployment artifact was exported.
