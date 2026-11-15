# Robust temporal learning 11

adaptive historical selection estimates; all composite metrics use proxies

| algorithm | core proportion | FIT | proxy score | FIT gain | proxy-score gain | Holm | eligible |
|---|---|---|---|---|---|---|---|
| core_consensus | 0 | 0.02556789 | 0.04403967 | +0.00127666 | +0.01027755 | False | False |
| core_consensus | 0.25 | 0.02549819 | 0.04399903 | +0.00120696 | +0.01023691 | False | False |
| history_consensus | 0 | 0.02534744 | 0.04303056 | +0.00105622 | +0.00926844 | False | False |
| history_consensus | 0.25 | 0.02528986 | 0.04300410 | +0.00099864 | +0.00924198 | False | False |
| stable_consensus | 0 | 0.02624674 | 0.04299203 | +0.00195552 | +0.00922991 | False | False |
| stable_consensus | 0.25 | 0.02614103 | 0.04287976 | +0.00184980 | +0.00911764 | False | False |
| score_multistop | 0.25 | 0.02278377 | 0.04025920 | -0.00150745 | +0.00649708 | False | False |
| score_multistop | 0 | 0.02267523 | 0.03983575 | -0.00161599 | +0.00607363 | False | False |

Control replay: {"source_plan_sha256": "218ed9a03aee83acac7022d89409608b67c3fa536763c0a4e6aaeec420c2d507", "prediction_max_error": {"baseline": 0.0, "score_rolling": 0.0}}

Confirmation: {"confirmation_scored": false, "plan_sha256": "f7391b84877f11f2ebf0980f867ba9122b3a61366e96ca04c214186c1ae280e2", "qualified": false, "reason": "no eligible discovery candidate"}

94 model checkpoints replayed their actual serialized boosters; max error 0.

Components are descriptive, cannot replace a failed comparison, and were all retained in their registered groups.

| component | FIT | proxy score | early FIT | middle FIT | late FIT |
|---|---|---|---|---|---|
| baseline | 0.02429122 | 0.03376212 | 0.00997155 | 0.03852949 | 0.02442690 |
| score_rolling | 0.02506814 | 0.04417933 | 0.01117136 | 0.04078286 | 0.02203824 |
| score_short | 0.01875908 | 0.03336052 | 0.00904728 | 0.02657533 | 0.02191831 |
| score_long | 0.02297039 | 0.03669478 | 0.00667708 | 0.03892651 | 0.02353235 |
| score_core_even | 0.02312118 | 0.03977900 | 0.01093775 | 0.03663688 | 0.02090074 |
| score_core_odd | 0.02371638 | 0.03950877 | 0.01389427 | 0.03594611 | 0.01970367 |
| score_multistop | 0.02267523 | 0.03983575 | 0.01018863 | 0.03644566 | 0.02053551 |

Baseline01 remains unchanged. No deployment, old-audit reads, uploads, submissions or staking.

Interpretation and failure investigation:

- Fixed group consensus reached FIT 0.02624674 vs refreshed baseline 0.02429122 (+8.05%), with UNQ proxy
  0.00550716, INOV proxy 0.01227907 and bounded score proxy 0.04299203 vs 0.03376212 (+27.34%). All six
  aggregate scenario score differences were positive (range +0.007525 to +0.009575), and score differences
  were positive in all three time windows. These are conditional local-reference results, not live scores.
- Its primary score p-value was 0.006999, missing the first eight-comparison Holm threshold of 0.00625.
  FIT p_pos was 0.8574 vs required 0.95, and late FIT fell 0.00144052 vs the allowed 0.001. Thus the result
  fails for multiple reasons, not only a borderline corrected score test. No threshold was relaxed and no
  confirmation labels were loaded. The predeclared 40/60-era score-bootstrap sensitivity p-values were
  0.005699/0.010299; these do not replace the primary 20-era decision.
- Core consensus FIT (0.02556789) exceeded each of its individual constituents (0.02506814, 0.02312118,
  0.02371638), evidence that weaker individual recipes can supply complementary prediction directions.
  Its late FIT still lost 0.00347480, so diversity did not solve temporal stability.
- History consensus FIT was 0.02534744, with late FIT loss narrowed to 0.00127872 from the rolling control's
  0.00238866 loss. The short and long components were individually weaker on average; deleting them on
  that basis would miss the complementarity the fixed mixture tested. The short component lost most in
  the middle window, while the long component improved late FIT over the 1,024-era model. This establishes
  horizon sensitivity, not the economic cause of the changing useful horizon.
- Multi-window stopping reduced FIT from 0.02506814 to 0.02267523. Seven blocks with unchanged final counts
  reproduced the single-window control's predictions exactly (max error 0). The other six blocks changed
  only the final stopping count; this identifies count transfer as the operational source of their changed
  predictions. Late counts changed 27→8, 15→20 and 29→6. Older inner windows use smaller available fit
  partitions and different periods, so sample-size transfer and temporal shift remain competing causes.
  More historical validation windows did not automatically improve this rule. No older-window model was
  silently dropped from the registered consensus.

Validation: 10 temporal/consensus tests passed. Full CPU suite: 469 passed, 3 optional-torch-interpreter skips.
Re-running the MLP suite with the absolute GPU interpreter configured passed all 7 tests, covering those
three skipped paths. All 94 saved boosters replayed with zero error, and baseline/rolling controls matched
round 10 exactly. Implementation commit: 62212fb. No full-data deployment pickle was exported.
