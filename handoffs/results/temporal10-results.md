# Temporal adaptation and composite score 10

adaptive historical development estimates; composite scores use proxies

All composite metrics use local reference proxies, not the platform benchmark or known core set.

| recipe | core proportion | FIT | proxy score | FIT gain | proxy-score gain | Holm | eligible |
|---|---|---|---|---|---|---|---|
| score_rolling | 0.25 | 0.02510301 | 0.04442926 | +0.00081178 | +0.01066714 | False | False |
| score_rolling | 0 | 0.02506814 | 0.04417933 | +0.00077692 | +0.01041721 | False | False |
| consensus | 0 | 0.02536313 | 0.03972384 | +0.00107190 | +0.00596172 | False | False |
| consensus | 0.25 | 0.02525716 | 0.03957673 | +0.00096594 | +0.00581461 | False | False |
| decay | 0 | 0.02366854 | 0.03779707 | -0.00062268 | +0.00403495 | False | False |
| decay | 0.25 | 0.02355350 | 0.03768571 | -0.00073772 | +0.00392359 | False | False |
| score_expanding | 0 | 0.02199896 | 0.03445793 | -0.00229227 | +0.00069581 | False | False |
| score_expanding | 0.25 | 0.02184408 | 0.03423267 | -0.00244715 | +0.00047055 | False | False |
| baseline | 0.25 | 0.02423284 | 0.03373847 | -0.00005838 | -0.00002365 | False | False |
| rolling | 0 | 0.02039527 | 0.03225023 | -0.00389596 | -0.00151189 | False | False |
| rolling | 0.25 | 0.02035770 | 0.03203865 | -0.00393353 | -0.00172347 | False | False |

Refreshed baseline FIT: 0.02429122; static baseline: 0.02225898.
Refresh paired FIT gain: +0.00203224; p_pos 0.8649.

Confirmation: {"confirmation_scored": false, "plan_sha256": "218ed9a03aee83acac7022d89409608b67c3fa536763c0a4e6aaeec420c2d507", "qualified": false, "reason": "no eligible discovery candidate"}

120 byte-bound fit checkpoints; serialization max error 0.

Baseline01 retained. No platform writes, old-audit reads or deployment changes.

Matched target/count controls (descriptive; no selection):

| variant | FIT | proxy score | early FIT | middle FIT | late FIT |
|---|---|---|---|---|---|
| diag_baseline_count_score_expanding | 0.02350539 | 0.03515893 | 0.01059953 | 0.03667974 | 0.02305789 |
| diag_score_expanding_count_baseline | 0.02370887 | 0.03550355 | 0.00795860 | 0.04022033 | 0.02244024 |
| diag_rolling_count_score_rolling | 0.02172813 | 0.03607923 | 0.00709037 | 0.03478630 | 0.02436077 |
| diag_score_rolling_count_rolling | 0.02220476 | 0.03616811 | 0.00607925 | 0.03955710 | 0.02016004 |

Interpretation: the rolling score-aware algorithm's unneutralized FIT was 0.02506814. Raw rolling was
0.02039527; transferring only the score-aware count recovered 0.02172813, while changing only the target at
the raw count reached 0.02220476. Both interventions together worked better than either isolated intervention;
this supports a target/count interaction, not a claim that stopping alone caused the improvement. The same
interventions did not improve expanding-history FIT, where the complete score-aware model was 0.02199896.
The short-window counts varied from 1 to 319 across recipes and dates. Count noise remains a hypothesis,
not an established cause; the next protocol tests multiple inner windows without changing this round's gates.

The best primary score p-value was 0.01240, above the first eleven-comparison Holm threshold of 0.00455.
That candidate's FIT gain was only 0.000812 and late-window FIT difference was -0.002010, failing the
prespecified effect and temporal guards. All six aggregate proxy scenarios improved, but that sensitivity
result cannot establish performance against the actual platform benchmark. No candidate qualified.

Validation: 7 new tests passed; full suite 466 passed, 3 skipped in the CPU venv. All 120 actual booster
checkpoints replayed their predictions with zero serialization error. Source commit: 97c82c3.
