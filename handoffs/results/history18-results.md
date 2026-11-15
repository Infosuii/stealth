# Raw-target training history 18

adaptive historical raw-history research; all intervals viewed

The all-history CPU model improved viewed-transfer FIT from **0.01526832 to 0.01819329 (+19.16%)**
and primary composite proxy from **0.01868202 to 0.02531301 (+35.49%)**. It and its fixed baseline blend
passed that interval's unchanged eight-family Holm/effect/window/reference checks. Neither passed earlier
replication or development. No candidate passed all three intervals, no model was promoted, and the requested
reliable major improvement remains unestablished. All intervals are viewed historical data.

| interval | policy | FIT | baseline FIT | proxy score | baseline proxy | historical gates |
|---|---|---|---|---|---|---|
| earlier_replication | cpu_expanding3 | 0.01402049 | 0.01690444 | 0.02107137 | 0.02369472 | False |
| earlier_replication | cpu_expanding3_blend | 0.01623850 | 0.01690444 | 0.02359611 | 0.02369472 | False |
| earlier_replication | gpu_expanding400 | 0.01264043 | 0.01690444 | 0.01716363 | 0.02369472 | False |
| earlier_replication | gpu_expanding400_blend | 0.01550088 | 0.01690444 | 0.02120134 | 0.02369472 | False |
| earlier_replication | gpu_expanding3 | 0.01269416 | 0.01690444 | 0.01759788 | 0.02369472 | False |
| earlier_replication | gpu_expanding3_blend | 0.01542831 | 0.01690444 | 0.02122456 | 0.02369472 | False |
| earlier_replication | balanced_history | 0.01521444 | 0.01690444 | 0.02291716 | 0.02369472 | False |
| earlier_replication | balanced_history_blend | 0.01654311 | 0.01690444 | 0.02393006 | 0.02369472 | False |
| viewed_development | cpu_expanding3 | 0.02679559 | 0.02702630 | 0.03852906 | 0.03790877 | False |
| viewed_development | cpu_expanding3_blend | 0.02707247 | 0.02702630 | 0.03846471 | 0.03790877 | False |
| viewed_development | gpu_expanding400 | 0.02560661 | 0.02702630 | 0.03554853 | 0.03790877 | False |
| viewed_development | gpu_expanding400_blend | 0.02682449 | 0.02702630 | 0.03744631 | 0.03790877 | False |
| viewed_development | gpu_expanding3 | 0.02619614 | 0.02702630 | 0.03757216 | 0.03790877 | False |
| viewed_development | gpu_expanding3_blend | 0.02729248 | 0.02702630 | 0.03852812 | 0.03790877 | False |
| viewed_development | balanced_history | 0.02694417 | 0.02702630 | 0.03902887 | 0.03790877 | False |
| viewed_development | balanced_history_blend | 0.02713880 | 0.02702630 | 0.03854957 | 0.03790877 | False |
| viewed_transfer | cpu_expanding3 | 0.01819329 | 0.01526832 | 0.02531301 | 0.01868202 | True |
| viewed_transfer | cpu_expanding3_blend | 0.01689213 | 0.01526832 | 0.02221009 | 0.01868202 | True |
| viewed_transfer | gpu_expanding400 | 0.01584654 | 0.01526832 | 0.02400987 | 0.01868202 | False |
| viewed_transfer | gpu_expanding400_blend | 0.01661269 | 0.01526832 | 0.02271508 | 0.01868202 | False |
| viewed_transfer | gpu_expanding3 | 0.01589544 | 0.01526832 | 0.02490748 | 0.01868202 | False |
| viewed_transfer | gpu_expanding3_blend | 0.01692614 | 0.01526832 | 0.02350689 | 0.01868202 | False |
| viewed_transfer | balanced_history | 0.01532587 | 0.01526832 | 0.02200418 | 0.01868202 | False |
| viewed_transfer | balanced_history_blend | 0.01568438 | 0.01526832 | 0.02088351 | 0.01868202 | False |

234 groups; 78 copied immutable controls; serialization error 0.
No candidate was promoted and no platform state changed. Historical proxy scores do not establish future performance.

## What the controlled histories show

| interval | CPU rolling | CPU expanding 400 | CPU expanding 3 | GPU rolling | GPU expanding 400 | GPU expanding 3 |
|---|---|---|---|---|---|---|
| earlier_replication | 0.00987382 | 0.01690444 | 0.01402049 | 0.00824086 | 0.01264043 | 0.01269416 |
| viewed_development | 0.02039527 | 0.02702630 | 0.02679559 | 0.02137180 | 0.02560661 | 0.02619614 |
| viewed_transfer | 0.00666251 | 0.01526832 | 0.01819329 | 0.00491660 | 0.01584654 | 0.01589544 |

These controls isolate a material weakness in the short-history recipes. Expanding from era 400 improves
CPU FIT over the rolling 1,024-era recipe in all three intervals; descriptive paired FIT p_pos is
0.9926/0.9986/0.9835. GPU FIT also rises in all three, with p_pos 0.9336/0.9375/0.9930. In transfer GPU FIT
rises 0.00491660→0.01584654; the paired 90% FIT-gain interval is [0.00356839, 0.01813875]. This supports
history truncation as a contributor to prior raw-family failures, rather than concluding the family has
no useful signal. GPU randomness and history-dependent effective regularization prevent a pure attribution
to the economic relevance of old rows. These descriptive contrasts have not been multiplicity-adjusted and
do not replace the registered comparison gates.

Adding the very earliest eras has a different CPU effect by period: FIT delta -0.00288395 earlier,
-0.00023071 development, +0.00292496 transfer. Its earlier middle-segment loss is -0.00899156. The transfer
success is therefore period-dependent, even though all its transfer segment FIT/score gains and six reference
scenario score gains are positive. Additional GPU data from eras 3–399 has no established incremental FIT
benefit over expanding-from-400: paired p_pos 0.5197/0.7030/0.5215. Do not infer that maximum history is
universally optimal or select a historical period's best history after seeing its result. The mechanism
behind earliest-era relevance versus sample-size/regularization effects remains unresolved.

## Training length and integrity

There are 234 checkpoint groups (78 copied byte-verified controls) covering 351 selected native models.
234 models were newly fitted: 78 CPU and 156 GPU seed models. Each new GPU inner trajectory completed all
2,048 iterations, totalling 319,488 inner boosting iterations plus selected-count refits. Expanding-from-400
selected a median 32 trees; expanding-from-3 selected a median 48. In 154/156 fits the full-cap endpoint was
worse than the selected inner proxy optimum; only two selected the cap. Merely extending every schedule is
not supported as the main repair. These stopping diagnostics describe the inner periods, not proof of the
best future training length.

All 351 fit/inner-split histories passed the strict 40-era boundaries and declared history/floor checks.
All checkpoint files, plans, teacher dependencies and copied controls verified; actual saved-model replay
max error was zero. The full suite passed 495 tests without skips. No reader admitted labels >=5925.

Operational issue: the first GPU shell command returned 143 after logging its last fit and writing every
checkpoint. The cause is unresolved. Re-running the identical resume stage verified and reused all 117 GPU
groups, trained nothing new, and exited 0; no source, configuration, gates or native bytes were changed to
handle it. Both the nonzero exit and successful recovery are retained in the machine-readable report.

This completes context15/15r, objective16/17 and history18. The support repair alone did not recover transfer;
utility targets interacted with stopping counts and did not generalize; raw-history retention is the strongest
new mechanistic lead. Baseline01 and platform state remain unchanged. No uploads, submissions or staking.
