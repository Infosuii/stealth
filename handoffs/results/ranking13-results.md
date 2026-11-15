# GPU ranking and model diversity 13

adaptive historical selection estimates; all composite metrics use proxies

| algorithm | core proportion | FIT | proxy score | FIT gain | proxy-score gain | Holm | eligible |
|---|---|---|---|---|---|---|---|
| cross_family | 0 | 0.02392945 | 0.03857805 | -0.00036177 | +0.00481593 | False | False |
| cat_rmse_blend | 0 | 0.02454753 | 0.03699045 | +0.00025631 | +0.00322832 | False | False |
| cat_query_blend | 0 | 0.02442262 | 0.03680781 | +0.00013139 | +0.00304569 | False | False |
| cat_rmse | 0 | 0.02137180 | 0.03524416 | -0.00291942 | +0.00148204 | False | False |
| cat_query | 0 | 0.02058446 | 0.03383017 | -0.00370677 | +0.00006805 | False | False |
| cat_rank_blend | 0 | 0.02305636 | 0.03336038 | -0.00123486 | -0.00040174 | False | False |
| ranking_consensus | 0 | 0.02011840 | 0.03199935 | -0.00417282 | -0.00176277 | False | False |
| cat_rank | 0 | 0.01879643 | 0.02920880 | -0.00549480 | -0.00455332 | False | False |

Control replay: {"source_plan_sha256": "218ed9a03aee83acac7022d89409608b67c3fa536763c0a4e6aaeec420c2d507", "prediction_max_error": {"baseline": 0.0, "score_rolling": 0.0}}

Confirmation: {"confirmation_scored": false, "plan_sha256": "61dc954ab152e88f7b45bd61e7d4a9edc0c7b6b67511cfbfcfd4cf9c104f3265", "qualified": false, "reason": "no eligible discovery candidate"}

133 checkpoint groups comprise 94 copied source-verified groups and 39 new two-seed GPU groups. The 78
new native GPU models replayed CPU predictions after serialization with max error 0; copied groups retain
their prior exact-replay evidence and their actual bytes were verified again.

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
| cat_rmse | 0.02137180 | 0.03524416 | 0.00143613 | 0.03644789 | 0.02947111 |
| cat_query | 0.02058446 | 0.03383017 | 0.00175662 | 0.03588064 | 0.02647054 |
| cat_rank | 0.01879643 | 0.02920880 | 0.00522616 | 0.02821108 | 0.02572246 |

Baseline01 remains unchanged. No deployment, old-audit reads, uploads, submissions or staking.

Copied source-verified checkpoints: 94; newly fitted GPU seed models: 78.

Failure investigation and useful diversity:

- All 78 native fits reported task_type GPU. Inner trajectories trained 159,744 boosting iterations in total,
  plus their selected-count refits. No seed had its best inner checkpoint at the 2,048-iteration cap. Selected
  counts ranged 1–2016 for RMSE, 1–1792 for QueryRMSE and 1–1568 for YetiRank. Mean best-to-endpoint validation
  proxy-score declines were 0.02924, 0.02767 and 0.01483 respectively. These curves show that the full-budget
  endpoint was worse than an earlier checkpoint; they do not prove a longer trajectory can never recover.
- Raw GPU regression improved late FIT by 0.00504421 (+20.65%) but lost early by 0.00853542. Both seeds showed
  the late strength independently (FIT 0.02885106/0.02872171 vs baseline 0.02442690); neither matched the
  early baseline. Thus the reversal is not explained solely by one fortunate seed. GPU nondeterminism and
  seed sensitivity remain real; no best seed was selected retrospectively.
- Equal seed averaging improved RMSE FIT over either individual seed's whole-period FIT (0.02137180 vs
  0.01985259/0.02112432). Query-centered regression and ranking did not beat regression overall. This
  contradicts the hypothesis that ranking loss alone would solve the performance gap in this setup.
  It does not establish which native loss mechanics or exposure differences caused the gap.
- The fixed cross-family consensus helped late but diluted the prior stable group's aggregate FIT from
  0.02624674 to 0.02392945. Rejecting the GPU family solely on its aggregate would miss a potential hedge
  for the existing late weakness. An honest next intervention must learn from already-matured previous
  results, not retrospectively switch families at the known early/late boundary.

No primary comparison qualified; no confirmation labels were read. The incumbent and prior failed gates
are unchanged. Full suite: 478 passed without skips with the absolute torch interpreter configured.
An additional source/destination corruption regression passed in the 5-test ranking suite. Source 6ec06f9.
