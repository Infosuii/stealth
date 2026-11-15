# Utility target/count controls 17

descriptive matched CPU target/count controls; no selection or promotion

| interval | cell | FIT | proxy score |
|---|---|---|---|
| earlier_replication | full_at_core | 0.00630757 | 0.01165111 |
| earlier_replication | core_at_full | 0.00859536 | 0.01543199 |
| earlier_replication | cpu_core | 0.00802678 | 0.01433139 |
| earlier_replication | cpu_full | 0.00694412 | 0.01305257 |
| viewed_development | full_at_core | 0.02432342 | 0.04142183 |
| viewed_development | core_at_full | 0.02563523 | 0.04474117 |
| viewed_development | cpu_core | 0.02475115 | 0.04208874 |
| viewed_development | cpu_full | 0.02409077 | 0.04201138 |
| viewed_transfer | full_at_core | 0.00717651 | 0.01349071 |
| viewed_transfer | core_at_full | 0.00379531 | 0.00781607 |
| viewed_transfer | cpu_core | 0.00340346 | 0.00655334 |
| viewed_transfer | cpu_full | 0.00420219 | 0.00922545 |

Own-count replay max error: 0.
Diagnostics cannot select a new candidate or change previous gates. No platform writes.

The six prespecified own-count replays matched exactly. There were 54 changed-count refits and 24 explicit
equal-count reuses. These are deterministic CPU controls; no GPU stochastic draw is treated as an exact
counterfactual.

At the core model's fixed count, changing core labels to full labels lowered FIT in earlier replication
(0.00802678 → 0.00630757) and development (0.02475115 → 0.02432342), but improved viewed transfer
(0.00340346 → 0.00717651). Full labels at their own count reached only 0.00420219 transfer FIT, so the
new target and selected count interact materially. The full-label own-count effect relative to the borrowed
core count was -0.00297432 in transfer. None of these diagnostic cells beat transfer baseline 0.01526832.

The result contradicts both a uniformly beneficial full-label hypothesis and a uniformly bad-label account:
effects differed by period and count. It supports investigating count robustness and training context, but
does not identify the economic cause of the reversal or authorise deploying a diagnostic count swap.
All intervals are viewed historical evidence. Source 9e03a5b; no candidate substitution or gate changes.
