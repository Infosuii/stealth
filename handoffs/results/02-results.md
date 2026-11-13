# Results 02 — test-run

- run_id: `test01`  config: `test`  status: **ok**
- git_sha: `acfaf3394247783fc1dff24b8a7849b445e91542`  started: 2026-11-13T20:19:10+00:00  finished: 2026-11-13T20:27:27+00:00
- generated: 2026-11-13T20:31:47+00:00 by `python -m stealth.report`

## Environment

| item | value |
|---|---|
| os | Linux 7.3.3-1-cachyos |
| cpu | x86_64 x20 |
| ram_gb | 30.8 |
| gpu | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| driver | 620.70.06 |
| python | 3.11.18 |
| timezone | Europe/London (UTC+00:00) |
| git_sha | acfaf3394247783fc1dff24b8a7849b445e91542 |
| core.hooksPath (reporting host) | scripts/hooks |
| pins (reporting host) | OK |
| libs | catboost 1.2.10, everesteer-api 0.4.1, lightgbm 4.7.0, numpy 2.4.6, optuna 5.0.0, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.17.1, xgboost 3.2.0 |

## Metrics (per-era fit20 on OOF; fit_se = 20-era block-bootstrap SE; config names only, no hyperparameter values)

| config | kind | fit_mean | fit_se | sharpe | pct_pos | ci90 | forward_fit | rho_b_val | n_iter | suspect | paired diff_mean | diff_se | p_pos | vs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm_everest | member | 0.0238 | 0.0026 | 0.51 | 0.71 | [0.0194, 0.0280] | 0.0313 | 0.739 | 119 | no | - | - | - | - |
| lgbm_tougroute | member | 0.0203 | 0.0025 | 0.46 | 0.69 | [0.0161, 0.0242] | 0.0225 | 0.690 | 57 | no | -0.0035 | 0.0008 | 0.00 | lgbm_everest |
| lgbm_ghat | member | 0.0119 | 0.0024 | 0.29 | 0.63 | [0.0080, 0.0158] | 0.0243 | 0.610 | 116 | no | -0.0119 | 0.0026 | 0.00 | lgbm_everest |
| lgbm_meltsene | member | 0.0230 | 0.0026 | 0.51 | 0.71 | [0.0187, 0.0270] | 0.0308 | 0.679 | 64 | no | -0.0008 | 0.0007 | 0.15 | lgbm_everest |
| xgb_everest_d3_lr05 | member | 0.0234 | 0.0027 | 0.50 | 0.71 | [0.0189, 0.0276] | 0.0303 | 0.717 | 39 | no | -0.0004 | 0.0005 | 0.21 | lgbm_everest |
| xgb_everest_d4_lr05 | member | 0.0222 | 0.0027 | 0.47 | 0.70 | [0.0177, 0.0264] | 0.0278 | 0.735 | 35 | no | -0.0017 | 0.0006 | 0.01 | lgbm_everest |
| xgb_everest_d3_lr02 | member | 0.0232 | 0.0027 | 0.50 | 0.70 | [0.0187, 0.0274] | 0.0310 | 0.734 | 113 | no | -0.0006 | 0.0004 | 0.05 | lgbm_everest |
| xgb_everest_d6_lr02 | member | 0.0218 | 0.0027 | 0.46 | 0.69 | [0.0173, 0.0260] | 0.0265 | 0.740 | 60 | no | -0.0020 | 0.0007 | 0.00 | lgbm_everest |
| cat_everest_l2_30 | member | 0.0167 | 0.0025 | 0.37 | 0.67 | [0.0125, 0.0208] | 0.0112 | 0.764 | 400 | no | -0.0072 | 0.0016 | 0.00 | lgbm_everest |
| ridge_everest | member | 0.0215 | 0.0026 | 0.46 | 0.69 | [0.0171, 0.0256] | 0.0252 | 0.705 | - | no | -0.0024 | 0.0010 | 0.01 | lgbm_everest |
| blend_ew | blend | 0.0226 | 0.0026 | 0.50 | 0.70 | [0.0182, 0.0267] | 0.0284 | - | - | - | - | - | - | - |

## Paired block-bootstrap tests

| A_vs_B | diff_mean | diff_se | p_pos | ci90 | n_eras | verdict |
|---|---|---|---|---|---|---|
| lgbm_tougroute_vs_lgbm_everest | -0.0035 | 0.0008 | 0.00 | [-0.0049, -0.0021] | 3725 | tie — keep the simpler one |
| lgbm_ghat_vs_lgbm_everest | -0.0119 | 0.0026 | 0.00 | [-0.0164, -0.0077] | 3725 | tie — keep the simpler one |
| lgbm_meltsene_vs_lgbm_everest | -0.0008 | 0.0007 | 0.15 | [-0.0020, 0.0004] | 3725 | tie — keep the simpler one |
| xgb_everest_d3_lr05_vs_lgbm_everest | -0.0004 | 0.0005 | 0.21 | [-0.0013, 0.0004] | 3725 | tie — keep the simpler one |
| xgb_everest_d4_lr05_vs_lgbm_everest | -0.0017 | 0.0006 | 0.01 | [-0.0027, -0.0006] | 3725 | tie — keep the simpler one |
| xgb_everest_d3_lr02_vs_lgbm_everest | -0.0006 | 0.0004 | 0.05 | [-0.0013, 0.0000] | 3725 | tie — keep the simpler one |
| xgb_everest_d6_lr02_vs_lgbm_everest | -0.0020 | 0.0007 | 0.00 | [-0.0032, -0.0009] | 3725 | tie — keep the simpler one |
| cat_everest_l2_30_vs_lgbm_everest | -0.0072 | 0.0016 | 0.00 | [-0.0098, -0.0045] | 3725 | tie — keep the simpler one |
| ridge_everest_vs_lgbm_everest | -0.0024 | 0.0010 | 0.01 | [-0.0040, -0.0007] | 3725 | tie — keep the simpler one |
| blend_nnls_vs_ew | -0.0001 | 0.0007 | 0.44 | [-0.0013, 0.0011] | 745 | tie — keep the simpler one |

## Regime simulation (blend.json)

| bench_fit_mean | p_b | regime | fit | fit_se | n_eras | unq | unq_se | inov | inov_se | score | score_se | frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | p_b=0 | bpos | 0.0226 | 0.0024 | 3725 | 0.0063 | 0.0024 | 0.0134 | 0.0016 | 0.0423 | 0.0061 | 0.5020 |
| 0 | p_b=0 | bneg | 0.0222 | 0.0024 | 3725 | 0.0000 | 0.0000 | 0.0132 | 0.0016 | 0.0354 | 0.0038 | 0.4980 |
| 0 | p_b=0 | mix | 0.0224 | 0.0024 | 3725 | 0.0032 | 0.0012 | 0.0133 | 0.0016 | 0.0389 | 0.0050 | 1.0000 |
| 0 | p_b=0.25 | bpos | 0.0207 | 0.0026 | 3725 | 0.0070 | 0.0025 | 0.0128 | 0.0017 | 0.0404 | 0.0065 | 0.5020 |
| 0 | p_b=0.25 | bneg | 0.0269 | 0.0025 | 3725 | 0.0000 | 0.0000 | 0.0160 | 0.0017 | 0.0430 | 0.0040 | 0.4980 |
| 0 | p_b=0.25 | mix | 0.0238 | 0.0025 | 3725 | 0.0035 | 0.0013 | 0.0144 | 0.0017 | 0.0417 | 0.0053 | 1.0000 |
| 0 | p_b=0.5 | bpos | 0.0176 | 0.0027 | 3725 | 0.0077 | 0.0027 | 0.0117 | 0.0018 | 0.0370 | 0.0069 | 0.5020 |
| 0 | p_b=0.5 | bneg | 0.0328 | 0.0026 | 3725 | 0.0000 | 0.0000 | 0.0196 | 0.0019 | 0.0524 | 0.0042 | 0.4980 |
| 0 | p_b=0.5 | mix | 0.0252 | 0.0027 | 3725 | 0.0039 | 0.0014 | 0.0156 | 0.0018 | 0.0447 | 0.0056 | 1.0000 |
| 0 | p_b=0.75 | bpos | 0.0135 | 0.0028 | 3725 | 0.0083 | 0.0028 | 0.0100 | 0.0018 | 0.0318 | 0.0072 | 0.5020 |
| 0 | p_b=0.75 | bneg | 0.0387 | 0.0027 | 3725 | 0.0000 | 0.0000 | 0.0231 | 0.0020 | 0.0619 | 0.0044 | 0.4980 |
| 0 | p_b=0.75 | mix | 0.0261 | 0.0028 | 3725 | 0.0041 | 0.0014 | 0.0166 | 0.0019 | 0.0468 | 0.0057 | 1.0000 |
| 0.009 | p_b=0 | bpos | 0.0226 | 0.0024 | 3725 | 0.0043 | 0.0024 | 0.0134 | 0.0016 | 0.0403 | 0.0061 | 0.6031 |
| 0.009 | p_b=0 | bneg | 0.0222 | 0.0024 | 3725 | 0.0000 | 0.0000 | 0.0131 | 0.0016 | 0.0353 | 0.0038 | 0.3969 |
| 0.009 | p_b=0 | mix | 0.0224 | 0.0024 | 3725 | 0.0026 | 0.0014 | 0.0133 | 0.0016 | 0.0383 | 0.0052 | 1.0000 |
| 0.009 | p_b=0.25 | bpos | 0.0202 | 0.0026 | 3725 | 0.0048 | 0.0025 | 0.0125 | 0.0017 | 0.0375 | 0.0065 | 0.6031 |
| 0.009 | p_b=0.25 | bneg | 0.0266 | 0.0025 | 3725 | 0.0000 | 0.0000 | 0.0158 | 0.0017 | 0.0424 | 0.0040 | 0.3969 |
| 0.009 | p_b=0.25 | mix | 0.0227 | 0.0025 | 3725 | 0.0029 | 0.0015 | 0.0138 | 0.0017 | 0.0395 | 0.0055 | 1.0000 |
| 0.009 | p_b=0.5 | bpos | 0.0166 | 0.0027 | 3725 | 0.0054 | 0.0027 | 0.0111 | 0.0018 | 0.0331 | 0.0069 | 0.6031 |
| 0.009 | p_b=0.5 | bneg | 0.0320 | 0.0027 | 3725 | 0.0000 | 0.0000 | 0.0191 | 0.0019 | 0.0511 | 0.0042 | 0.3969 |
| 0.009 | p_b=0.5 | mix | 0.0227 | 0.0027 | 3725 | 0.0033 | 0.0016 | 0.0143 | 0.0018 | 0.0403 | 0.0058 | 1.0000 |
| 0.009 | p_b=0.75 | bpos | 0.0117 | 0.0028 | 3725 | 0.0059 | 0.0028 | 0.0091 | 0.0018 | 0.0267 | 0.0071 | 0.6031 |
| 0.009 | p_b=0.75 | bneg | 0.0375 | 0.0027 | 3725 | 0.0000 | 0.0000 | 0.0225 | 0.0020 | 0.0599 | 0.0044 | 0.3969 |
| 0.009 | p_b=0.75 | mix | 0.0219 | 0.0028 | 3725 | 0.0035 | 0.0017 | 0.0144 | 0.0018 | 0.0399 | 0.0060 | 1.0000 |
| 0.02 | p_b=0 | bpos | 0.0226 | 0.0024 | 3725 | 0.0013 | 0.0024 | 0.0134 | 0.0016 | 0.0373 | 0.0061 | 0.7163 |
| 0.02 | p_b=0 | bneg | 0.0221 | 0.0024 | 3725 | 0.0000 | 0.0000 | 0.0130 | 0.0016 | 0.0351 | 0.0038 | 0.2837 |
| 0.02 | p_b=0 | mix | 0.0224 | 0.0024 | 3725 | 0.0009 | 0.0017 | 0.0133 | 0.0016 | 0.0366 | 0.0055 | 1.0000 |
| 0.02 | p_b=0.25 | bpos | 0.0195 | 0.0026 | 3725 | 0.0017 | 0.0025 | 0.0122 | 0.0017 | 0.0334 | 0.0065 | 0.7163 |
| 0.02 | p_b=0.25 | bneg | 0.0261 | 0.0025 | 3725 | 0.0000 | 0.0000 | 0.0156 | 0.0017 | 0.0417 | 0.0040 | 0.2837 |
| 0.02 | p_b=0.25 | mix | 0.0214 | 0.0026 | 3725 | 0.0012 | 0.0018 | 0.0131 | 0.0017 | 0.0357 | 0.0058 | 1.0000 |
| 0.02 | p_b=0.5 | bpos | 0.0150 | 0.0027 | 3725 | 0.0021 | 0.0027 | 0.0103 | 0.0018 | 0.0274 | 0.0068 | 0.7163 |
| 0.02 | p_b=0.5 | bneg | 0.0311 | 0.0027 | 3725 | 0.0000 | 0.0000 | 0.0187 | 0.0019 | 0.0498 | 0.0043 | 0.2837 |
| 0.02 | p_b=0.5 | mix | 0.0196 | 0.0027 | 3725 | 0.0015 | 0.0019 | 0.0127 | 0.0018 | 0.0338 | 0.0061 | 1.0000 |
| 0.02 | p_b=0.75 | bpos | 0.0092 | 0.0028 | 3725 | 0.0024 | 0.0028 | 0.0077 | 0.0018 | 0.0194 | 0.0071 | 0.7163 |
| 0.02 | p_b=0.75 | bneg | 0.0361 | 0.0027 | 3725 | 0.0000 | 0.0000 | 0.0218 | 0.0020 | 0.0578 | 0.0044 | 0.2837 |
| 0.02 | p_b=0.75 | mix | 0.0168 | 0.0028 | 3725 | 0.0017 | 0.0020 | 0.0117 | 0.0018 | 0.0303 | 0.0063 | 1.0000 |

- simulation: n_draws 500, n_eras 3725, p_c 0.25, bench_fit_sd 0.035
- chosen p_b = 0.50, p_c = 0.25, rule = ew
- weights: lgbm_everest 0.071, xgb_everest_d3_lr05 0.071, xgb_everest_d4_lr05 0.071, xgb_everest_d3_lr02 0.071, xgb_everest_d6_lr02 0.071, cat_everest_l2_30 0.071, ridge_everest 0.071, lgbm_tougroute 0.167, lgbm_ghat 0.167, lgbm_meltsene 0.167
- paired NNLS vs EW: diff_mean -0.0001, diff_se 0.0007, p_pos 0.44

## Wall times

| step / member | wall seconds |
|---|---|
| TOTAL | 496.6 |
| cat_everest_l2_30 | 18.1 |
| lgbm_everest | 60.0 |
| lgbm_ghat | 16.9 |
| lgbm_meltsene | 16.1 |
| lgbm_tougroute | 16.1 |
| ridge_everest | 2.1 |
| xgb_everest_d3_lr02 | 14.4 |
| xgb_everest_d3_lr05 | 11.6 |
| xgb_everest_d4_lr05 | 11.0 |
| xgb_everest_d6_lr02 | 16.6 |

## Notes

- lgbm_everest: search gain +0.0009 on OOF is NOT confirmed on the forward block (forward gain -0.0006 < 0) -> treat the search as a tie

## run.log (last 60 lines)

```text
2026-11-13 20:27:27,091 INFO stealth.train: member lgbm_meltsene          fit20 mean=0.0230 se_block=0.0026 sharpe=0.512 pct_pos=0.71 forward=0.0308 n_iter=64
2026-11-13 20:27:27,091 INFO stealth.train: member xgb_everest_d3_lr05    fit20 mean=0.0234 se_block=0.0027 sharpe=0.502 pct_pos=0.71 forward=0.0303 n_iter=39
2026-11-13 20:27:27,091 INFO stealth.train: member xgb_everest_d4_lr05    fit20 mean=0.0222 se_block=0.0027 sharpe=0.474 pct_pos=0.70 forward=0.0278 n_iter=35
2026-11-13 20:27:27,091 INFO stealth.train: member xgb_everest_d3_lr02    fit20 mean=0.0232 se_block=0.0027 sharpe=0.496 pct_pos=0.70 forward=0.031 n_iter=113
2026-11-13 20:27:27,091 INFO stealth.train: member xgb_everest_d6_lr02    fit20 mean=0.0218 se_block=0.0027 sharpe=0.464 pct_pos=0.69 forward=0.0265 n_iter=60
2026-11-13 20:27:27,091 INFO stealth.train: member cat_everest_l2_30      fit20 mean=0.0167 se_block=0.0025 sharpe=0.367 pct_pos=0.67 forward=0.0112 n_iter=400
2026-11-13 20:27:27,091 INFO stealth.train: member ridge_everest          fit20 mean=0.0215 se_block=0.0026 sharpe=0.460 pct_pos=0.69 forward=0.0252 n_iter=None
2026-11-13 20:27:27,092 INFO stealth.train: run test01 finished: status=ok exit=0 total=497s
2026-11-13 20:27:33,609 INFO stealth.blend: members ['lgbm_everest', 'lgbm_tougroute', 'lgbm_ghat', 'lgbm_meltsene', 'xgb_everest_d3_lr05', 'xgb_everest_d4_lr05', 'xgb_everest_d3_lr02', 'xgb_everest_d6_lr02', 'cat_everest_l2_30', 'ridge_everest']; equal weights {lgbm_everest: 0.071, xgb_everest_d3_lr05: 0.071, xgb_everest_d4_lr05: 0.071, xgb_everest_d3_lr02: 0.071, xgb_everest_d6_lr02: 0.071, cat_everest_l2_30: 0.071, ridge_everest: 0.071, lgbm_tougroute: 0.167, lgbm_ghat: 0.167, lgbm_meltsene: 0.167}
2026-11-13 20:27:35,816 INFO stealth.blend: nested NNLS: nested weights {lgbm_everest: 0.433, lgbm_tougroute: 0.018, lgbm_ghat: 0.018, lgbm_meltsene: 0.166, xgb_everest_d3_lr05: 0.161, xgb_everest_d4_lr05: 0.018, xgb_everest_d3_lr02: 0.018, xgb_everest_d6_lr02: 0.018, cat_everest_l2_30: 0.018, ridge_everest: 0.133}; eval fold4 diff_mean=-0.0001 se=0.0007 p_pos=0.443; p_pos=0.443 < 0.8, forward diff=+0.0021 -> keep equal weights
2026-11-13 20:27:48,655 INFO stealth.blend: blend[ew]: fit20 mean=0.0226 se_block=0.0026 sharpe=0.496 pct_pos=0.70 forward=0.0284 (ew mean=0.0226)
2026-11-13 20:27:48,656 INFO stealth.metrics: leak_check pass: mean OOF Spearman 0.0994 <= 0.18
2026-11-13 20:27:49,089 INFO stealth.data: loaded futures_train.parquet: 468236 rows, 6522 eras (3..6524)
2026-11-13 20:28:59,657 INFO stealth.metrics: regime mean=0: p_b=0 mix score 0.0389; p_b=0.25 mix score 0.0417; p_b=0.5 mix score 0.0447; p_b=0.75 mix score 0.0468
2026-11-13 20:30:08,413 INFO stealth.metrics: regime mean=0.009: p_b=0 mix score 0.0383; p_b=0.25 mix score 0.0395; p_b=0.5 mix score 0.0403; p_b=0.75 mix score 0.0399
2026-11-13 20:31:16,989 INFO stealth.metrics: regime mean=0.02: p_b=0 mix score 0.0366; p_b=0.25 mix score 0.0357; p_b=0.5 mix score 0.0338; p_b=0.75 mix score 0.0303
2026-11-13 20:31:16,992 INFO stealth.blend: regime simulation: 500 draws in 208s
2026-11-13 20:31:17,001 INFO stealth.blend: regime bpos:
 bench_mean    p_b regime    fit    unq   inov  score  n_eras   frac
     0.0000 0.0000   bpos 0.0226 0.0063 0.0134 0.0423    3725 0.5020
     0.0000 0.2500   bpos 0.0207 0.0070 0.0128 0.0404    3725 0.5020
     0.0000 0.5000   bpos 0.0176 0.0077 0.0117 0.0370    3725 0.5020
     0.0000 0.7500   bpos 0.0135 0.0083 0.0100 0.0318    3725 0.5020
     0.0090 0.0000   bpos 0.0226 0.0043 0.0134 0.0403    3725 0.6031
     0.0090 0.2500   bpos 0.0202 0.0048 0.0125 0.0375    3725 0.6031
     0.0090 0.5000   bpos 0.0166 0.0054 0.0111 0.0331    3725 0.6031
     0.0090 0.7500   bpos 0.0117 0.0059 0.0091 0.0267    3725 0.6031
     0.0200 0.0000   bpos 0.0226 0.0013 0.0134 0.0373    3725 0.7163
     0.0200 0.2500   bpos 0.0195 0.0017 0.0122 0.0334    3725 0.7163
     0.0200 0.5000   bpos 0.0150 0.0021 0.0103 0.0274    3725 0.7163
     0.0200 0.7500   bpos 0.0092 0.0024 0.0077 0.0194    3725 0.7163
2026-11-13 20:31:17,002 INFO stealth.blend: regime bneg:
 bench_mean    p_b regime    fit    unq   inov  score  n_eras   frac
     0.0000 0.0000   bneg 0.0222 0.0000 0.0132 0.0354    3725 0.4980
     0.0000 0.2500   bneg 0.0269 0.0000 0.0160 0.0430    3725 0.4980
     0.0000 0.5000   bneg 0.0328 0.0000 0.0196 0.0524    3725 0.4980
     0.0000 0.7500   bneg 0.0387 0.0000 0.0231 0.0619    3725 0.4980
     0.0090 0.0000   bneg 0.0222 0.0000 0.0131 0.0353    3725 0.3969
     0.0090 0.2500   bneg 0.0266 0.0000 0.0158 0.0424    3725 0.3969
     0.0090 0.5000   bneg 0.0320 0.0000 0.0191 0.0511    3725 0.3969
     0.0090 0.7500   bneg 0.0375 0.0000 0.0225 0.0599    3725 0.3969
     0.0200 0.0000   bneg 0.0221 0.0000 0.0130 0.0351    3725 0.2837
     0.0200 0.2500   bneg 0.0261 0.0000 0.0156 0.0417    3725 0.2837
     0.0200 0.5000   bneg 0.0311 0.0000 0.0187 0.0498    3725 0.2837
     0.0200 0.7500   bneg 0.0361 0.0000 0.0218 0.0578    3725 0.2837
2026-11-13 20:31:17,002 INFO stealth.blend: regime mix:
 bench_mean    p_b regime    fit    unq   inov  score  n_eras   frac
     0.0000 0.0000    mix 0.0224 0.0032 0.0133 0.0389    3725 1.0000
     0.0000 0.2500    mix 0.0238 0.0035 0.0144 0.0417    3725 1.0000
     0.0000 0.5000    mix 0.0252 0.0039 0.0156 0.0447    3725 1.0000
     0.0000 0.7500    mix 0.0261 0.0041 0.0166 0.0468    3725 1.0000
     0.0090 0.0000    mix 0.0224 0.0026 0.0133 0.0383    3725 1.0000
     0.0090 0.2500    mix 0.0227 0.0029 0.0138 0.0395    3725 1.0000
     0.0090 0.5000    mix 0.0227 0.0033 0.0143 0.0403    3725 1.0000
     0.0090 0.7500    mix 0.0219 0.0035 0.0144 0.0399    3725 1.0000
     0.0200 0.0000    mix 0.0224 0.0009 0.0133 0.0366    3725 1.0000
     0.0200 0.2500    mix 0.0214 0.0012 0.0131 0.0357    3725 1.0000
     0.0200 0.5000    mix 0.0196 0.0015 0.0127 0.0338    3725 1.0000
     0.0200 0.7500    mix 0.0168 0.0017 0.0117 0.0303    3725 1.0000
2026-11-13 20:31:17,004 INFO stealth.blend: wrote /home/om/Projects/eversteer/stealth/artifacts/test01/blend.json and updated metrics.json
```
