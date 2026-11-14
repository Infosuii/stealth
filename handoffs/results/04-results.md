# Results 04 — full-run

- run_id: `full01`  config: `full`  status: **ok**
- git_sha: `1a0a9693f5b4fdf269d92f8b8bbb64085f8cb841`  started: 2026-11-14T01:54:29+00:00  finished: 2026-11-14T01:54:51+00:00
- generated: 2026-11-14T02:08:07+00:00 by `python -m stealth.report`

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
| git_sha | 1a0a9693f5b4fdf269d92f8b8bbb64085f8cb841 |
| core.hooksPath (reporting host) | scripts/hooks |
| pins (reporting host) | OK |
| libs | catboost 1.2.10, everesteer-api 0.4.1, lightgbm 4.7.0, numpy 2.4.6, optuna 5.0.0, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.17.1, xgboost 3.2.0 |

## Metrics — development selection estimates (reused for selection; optimistic) (per-era fit20 on OOF; fit_se = 20-era block-bootstrap SE; config names only, no hyperparameter values)

| config | kind | arm | weight | fit_mean | fit_se | sharpe | pct_pos | ci90 | forward_fit | rho_b_val | rho_c | n_iter | suspect | paired diff_mean | diff_se | p_pos | vs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm_everest | member | lgbm | 0.255 | 0.0211 | 0.0029 | 0.45 | 0.68 | [0.0163, 0.0260] | 0.0204 | 0.630 | 0.163 | 43 | no | - | - | - | - |
| lgbm_tougroute | member | aux | 0.000 | 0.0192 | 0.0027 | 0.44 | 0.67 | [0.0147, 0.0237] | 0.0138 | 0.531 | 0.120 | 47 | no | -0.0019 | 0.0009 | 0.02 | lgbm_everest |
| lgbm_ghat | member | aux | 0.000 | 0.0141 | 0.0022 | 0.39 | 0.66 | [0.0106, 0.0177] | 0.0161 | 0.573 | 0.102 | 8 | no | -0.0070 | 0.0023 | 0.00 | lgbm_everest |
| lgbm_meltsene | member | aux | 0.000 | 0.0196 | 0.0029 | 0.43 | 0.66 | [0.0149, 0.0243] | 0.0168 | 0.616 | 0.095 | 42 | no | -0.0016 | 0.0009 | 0.05 | lgbm_everest |
| lgbm_timesguida | member | aux | 0.000 | 0.0199 | 0.0027 | 0.46 | 0.67 | [0.0156, 0.0243] | 0.0168 | 0.544 | 0.086 | 29 | no | -0.0012 | 0.0010 | 0.11 | lgbm_everest |
| lgbm_ighil | member | aux | 0.000 | 0.0196 | 0.0029 | 0.43 | 0.66 | [0.0148, 0.0244] | 0.0148 | 0.619 | 0.173 | 34 | no | -0.0015 | 0.0015 | 0.16 | lgbm_everest |
| lgbm_tarhat | member | aux | 0.000 | 0.0153 | 0.0026 | 0.36 | 0.62 | [0.0110, 0.0195] | 0.0128 | 0.510 | 0.149 | 6 | no | -0.0058 | 0.0019 | 0.00 | lgbm_everest |
| xgb_everest | member | xgb | 0.000 | 0.0204 | 0.0029 | 0.45 | 0.68 | [0.0158, 0.0252] | 0.0197 | 0.630 | 0.124 | 30 | no | -0.0007 | 0.0006 | 0.15 | lgbm_everest |
| xgb_tougroute | member | aux | 0.000 | 0.0176 | 0.0027 | 0.40 | 0.67 | [0.0131, 0.0222] | 0.0156 | 0.529 | 0.095 | 36 | no | -0.0035 | 0.0010 | 0.00 | lgbm_everest |
| xgb_ghat | member | aux | 0.000 | 0.0121 | 0.0024 | 0.30 | 0.62 | [0.0082, 0.0160] | 0.0102 | 0.530 | 0.024 | 16 | no | -0.0090 | 0.0021 | 0.00 | lgbm_everest |
| xgb_meltsene | member | aux | 0.000 | 0.0194 | 0.0027 | 0.45 | 0.67 | [0.0151, 0.0238] | 0.0169 | 0.605 | 0.101 | 16 | no | -0.0018 | 0.0009 | 0.03 | lgbm_everest |
| xgb_timesguida | member | aux | 0.000 | 0.0175 | 0.0026 | 0.41 | 0.65 | [0.0132, 0.0219] | 0.0157 | 0.547 | 0.041 | 19 | no | -0.0036 | 0.0011 | 0.00 | lgbm_everest |
| xgb_ighil | member | aux | 0.000 | 0.0179 | 0.0027 | 0.41 | 0.66 | [0.0134, 0.0223] | 0.0149 | 0.603 | 0.123 | 14 | no | -0.0033 | 0.0013 | 0.01 | lgbm_everest |
| xgb_tarhat | member | aux | 0.000 | 0.0140 | 0.0024 | 0.35 | 0.64 | [0.0100, 0.0179] | 0.0138 | 0.449 | 0.163 | 5 | no | -0.0071 | 0.0015 | 0.00 | lgbm_everest |
| cat_everest_l2_3 | member | cat | 0.000 | 0.0190 | 0.0030 | 0.40 | 0.65 | [0.0141, 0.0239] | 0.0159 | 0.648 | 0.051 | 400 | no | -0.0021 | 0.0013 | 0.06 | lgbm_everest |
| cat_everest_l2_30 | member | cat | 0.000 | 0.0200 | 0.0030 | 0.41 | 0.66 | [0.0151, 0.0249] | 0.0160 | 0.649 | 0.056 | 400 | no | -0.0011 | 0.0013 | 0.18 | lgbm_everest |
| cat_everest_l2_100 | member | cat | 0.000 | 0.0195 | 0.0030 | 0.41 | 0.65 | [0.0146, 0.0244] | 0.0181 | 0.646 | 0.065 | 400 | no | -0.0016 | 0.0013 | 0.10 | lgbm_everest |
| ridge_everest | member | ridge | 0.745 | 0.0228 | 0.0031 | 0.47 | 0.69 | [0.0177, 0.0278] | 0.0217 | 0.601 | 0.071 | - | no | 0.0016 | 0.0011 | 0.93 | lgbm_everest |
| lgbm_everest_sharpe | member | sharpe | 0.000 | 0.0217 | 0.0029 | 0.47 | 0.68 | [0.0169, 0.0265] | 0.0170 | 0.616 | 0.098 | 105 | no | 0.0006 | 0.0006 | 0.81 | lgbm_everest |
| lgbm_everest_bag0 | member | bags | 0.000 | 0.0207 | 0.0030 | 0.43 | 0.66 | [0.0158, 0.0257] | 0.0166 | 0.592 | 0.153 | 68 | no | -0.0004 | 0.0010 | 0.34 | lgbm_everest |
| lgbm_everest_bag1 | member | bags | 0.000 | 0.0224 | 0.0030 | 0.46 | 0.70 | [0.0174, 0.0273] | 0.0209 | 0.644 | 0.121 | 169 | no | 0.0013 | 0.0012 | 0.85 | lgbm_everest |
| lgbm_everest_bag2 | member | bags | 0.000 | 0.0209 | 0.0030 | 0.44 | 0.66 | [0.0161, 0.0259] | 0.0175 | 0.592 | 0.175 | 63 | no | -0.0002 | 0.0010 | 0.42 | lgbm_everest |
| lgbm_everest_bag3 | member | bags | 0.000 | 0.0209 | 0.0029 | 0.44 | 0.69 | [0.0161, 0.0257] | 0.0168 | 0.570 | -0.043 | 72 | no | -0.0002 | 0.0011 | 0.41 | lgbm_everest |
| mlp_everest | member | mlp | 0.000 | 0.0092 | 0.0026 | 0.22 | 0.59 | [0.0049, 0.0135] | 0.0073 | 0.394 | -0.002 | 2 | no | -0.0119 | 0.0025 | 0.00 | lgbm_everest |
| blend_nnls | blend (cross-fitted) | - | - | 0.0226 | 0.0030 | 0.46 | 0.69 | [0.0175, 0.0276] | 0.0219 | - | - | - | no | 0.0014 | 0.0009 | 0.95 | lgbm_everest |
| blend_ew | blend_context | - | - | 0.0226 | 0.0030 | 0.47 | 0.68 | [0.0175, 0.0276] | - | - | - | - | - | - | - | - | - |

## Paired block-bootstrap tests

| A_vs_B | diff_mean | diff_se | p_pos | ci90 | n_eras | verdict |
|---|---|---|---|---|---|---|
| lgbm_tougroute_vs_lgbm_everest | -0.0019 | 0.0009 | 0.02 | [-0.0035, -0.0004] | 3125 | tie — keep the simpler one |
| lgbm_ghat_vs_lgbm_everest | -0.0070 | 0.0023 | 0.00 | [-0.0108, -0.0032] | 3125 | tie — keep the simpler one |
| lgbm_meltsene_vs_lgbm_everest | -0.0016 | 0.0009 | 0.05 | [-0.0031, -0.0000] | 3125 | tie — keep the simpler one |
| lgbm_timesguida_vs_lgbm_everest | -0.0012 | 0.0010 | 0.11 | [-0.0028, 0.0004] | 3125 | tie — keep the simpler one |
| lgbm_ighil_vs_lgbm_everest | -0.0015 | 0.0015 | 0.16 | [-0.0040, 0.0010] | 3125 | tie — keep the simpler one |
| lgbm_tarhat_vs_lgbm_everest | -0.0058 | 0.0019 | 0.00 | [-0.0090, -0.0028] | 3125 | tie — keep the simpler one |
| xgb_everest_vs_lgbm_everest | -0.0007 | 0.0006 | 0.15 | [-0.0017, 0.0003] | 3125 | tie — keep the simpler one |
| xgb_tougroute_vs_lgbm_everest | -0.0035 | 0.0010 | 0.00 | [-0.0052, -0.0018] | 3125 | tie — keep the simpler one |
| xgb_ghat_vs_lgbm_everest | -0.0090 | 0.0021 | 0.00 | [-0.0124, -0.0055] | 3125 | tie — keep the simpler one |
| xgb_meltsene_vs_lgbm_everest | -0.0018 | 0.0009 | 0.03 | [-0.0033, -0.0002] | 3125 | tie — keep the simpler one |
| xgb_timesguida_vs_lgbm_everest | -0.0036 | 0.0011 | 0.00 | [-0.0055, -0.0017] | 3125 | tie — keep the simpler one |
| xgb_ighil_vs_lgbm_everest | -0.0033 | 0.0013 | 0.01 | [-0.0055, -0.0011] | 3125 | tie — keep the simpler one |
| xgb_tarhat_vs_lgbm_everest | -0.0071 | 0.0015 | 0.00 | [-0.0097, -0.0046] | 3125 | tie — keep the simpler one |
| cat_everest_l2_3_vs_lgbm_everest | -0.0021 | 0.0013 | 0.06 | [-0.0043, 0.0000] | 3125 | tie — keep the simpler one |
| cat_everest_l2_30_vs_lgbm_everest | -0.0011 | 0.0013 | 0.18 | [-0.0033, 0.0010] | 3125 | tie — keep the simpler one |
| cat_everest_l2_100_vs_lgbm_everest | -0.0016 | 0.0013 | 0.10 | [-0.0037, 0.0005] | 3125 | tie — keep the simpler one |
| ridge_everest_vs_lgbm_everest | 0.0016 | 0.0011 | 0.93 | [-0.0002, 0.0034] | 3125 | adopt A (p_pos >= 0.8) |
| lgbm_everest_sharpe_vs_lgbm_everest | 0.0006 | 0.0006 | 0.81 | [-0.0005, 0.0017] | 3125 | adopt A (p_pos >= 0.8) |
| lgbm_everest_bag0_vs_lgbm_everest | -0.0004 | 0.0010 | 0.34 | [-0.0021, 0.0012] | 3125 | tie — keep the simpler one |
| lgbm_everest_bag1_vs_lgbm_everest | 0.0013 | 0.0012 | 0.85 | [-0.0007, 0.0032] | 3125 | adopt A (p_pos >= 0.8) |
| lgbm_everest_bag2_vs_lgbm_everest | -0.0002 | 0.0010 | 0.42 | [-0.0019, 0.0015] | 3125 | tie — keep the simpler one |
| lgbm_everest_bag3_vs_lgbm_everest | -0.0002 | 0.0011 | 0.41 | [-0.0021, 0.0017] | 3125 | tie — keep the simpler one |
| mlp_everest_vs_lgbm_everest | -0.0119 | 0.0025 | 0.00 | [-0.0161, -0.0079] | 3125 | tie — keep the simpler one |
| blend_nnls_vs_ew | 0.0007 | 0.0004 | 0.93 | [-0.0001, 0.0013] | 625 | adopt A (p_pos >= 0.8) |
| blend_vs_lgbm_everest | 0.0014 | 0.0009 | 0.95 | [0.0000, 0.0029] | 3125 | adopt A (p_pos >= 0.8) |
| aux_ladder | -0.0002 | 0.0003 | 0.25 | - | - | tie — keep the simpler one |
| xgb_ladder | -0.0001 | 0.0003 | 0.38 | - | - | tie — keep the simpler one |
| cat_ladder | -0.0001 | 0.0007 | 0.41 | - | - | tie — keep the simpler one |
| ridge_ladder | 0.0015 | 0.0006 | 0.99 | - | - | adopt A (p_pos >= 0.8) |
| sharpe_ladder | -0.0000 | 0.0002 | 0.45 | - | - | tie — keep the simpler one |
| bags_ladder | 0.0003 | 0.0002 | 0.91 | - | - | adopt A (p_pos >= 0.8) |
| mlp_ladder | -0.0020 | 0.0008 | 0.01 | - | - | tie — keep the simpler one |

## Regime simulation (blend.json)

| bench_fit_mean | p_b | regime | fit | fit_se | n_eras | unq | unq_se | inov | inov_se | score | score_se | frac | score_sharpe | score_sharpe_se |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | p_b=0 | bpos | 0.0228 | 0.0028 | 3125 | 0.0063 | 0.0028 | 0.0144 | 0.0020 | 0.0434 | 0.0072 | 0.5013 | 0.3456 | 0.0582 |
| 0 | p_b=0 | bneg | 0.0223 | 0.0028 | 3125 | 0.0000 | 0.0000 | 0.0141 | 0.0020 | 0.0365 | 0.0045 | 0.4987 | 0.4641 | 0.0581 |
| 0 | p_b=0 | mix | 0.0226 | 0.0028 | 3125 | 0.0031 | 0.0014 | 0.0143 | 0.0020 | 0.0400 | 0.0059 | 1.0000 | 0.3808 | 0.0564 |
| 0 | p_b=0.25 | bpos | 0.0208 | 0.0030 | 3125 | 0.0069 | 0.0030 | 0.0138 | 0.0021 | 0.0415 | 0.0077 | 0.5013 | 0.3076 | 0.0578 |
| 0 | p_b=0.25 | bneg | 0.0273 | 0.0030 | 3125 | 0.0000 | 0.0000 | 0.0171 | 0.0021 | 0.0444 | 0.0048 | 0.4987 | 0.5308 | 0.0593 |
| 0 | p_b=0.25 | mix | 0.0240 | 0.0030 | 3125 | 0.0035 | 0.0015 | 0.0154 | 0.0021 | 0.0429 | 0.0063 | 1.0000 | 0.3822 | 0.0566 |
| 0 | p_b=0.5 | bpos | 0.0177 | 0.0032 | 3125 | 0.0076 | 0.0032 | 0.0128 | 0.0022 | 0.0382 | 0.0082 | 0.5013 | 0.2648 | 0.0574 |
| 0 | p_b=0.5 | bneg | 0.0330 | 0.0031 | 3125 | 0.0000 | 0.0000 | 0.0205 | 0.0022 | 0.0535 | 0.0050 | 0.4987 | 0.6050 | 0.0606 |
| 0 | p_b=0.5 | mix | 0.0254 | 0.0031 | 3125 | 0.0038 | 0.0016 | 0.0166 | 0.0022 | 0.0458 | 0.0066 | 1.0000 | 0.3820 | 0.0566 |
| 0 | p_b=0.75 | bpos | 0.0135 | 0.0033 | 3125 | 0.0081 | 0.0033 | 0.0113 | 0.0022 | 0.0330 | 0.0086 | 0.5013 | 0.2178 | 0.0568 |
| 0 | p_b=0.75 | bneg | 0.0390 | 0.0032 | 3125 | 0.0000 | 0.0000 | 0.0240 | 0.0023 | 0.0630 | 0.0052 | 0.4987 | 0.6826 | 0.0616 |
| 0 | p_b=0.75 | mix | 0.0262 | 0.0033 | 3125 | 0.0041 | 0.0017 | 0.0176 | 0.0022 | 0.0479 | 0.0068 | 1.0000 | 0.3794 | 0.0561 |
| 0.009 | p_b=0 | bpos | 0.0228 | 0.0028 | 3125 | 0.0042 | 0.0028 | 0.0144 | 0.0020 | 0.0415 | 0.0072 | 0.6026 | 0.3300 | 0.0580 |
| 0.009 | p_b=0 | bneg | 0.0222 | 0.0028 | 3125 | 0.0000 | 0.0000 | 0.0140 | 0.0020 | 0.0362 | 0.0045 | 0.3974 | 0.4611 | 0.0582 |
| 0.009 | p_b=0 | mix | 0.0226 | 0.0028 | 3125 | 0.0026 | 0.0017 | 0.0143 | 0.0020 | 0.0394 | 0.0061 | 1.0000 | 0.3599 | 0.0566 |
| 0.009 | p_b=0.25 | bpos | 0.0203 | 0.0030 | 3125 | 0.0048 | 0.0030 | 0.0136 | 0.0021 | 0.0387 | 0.0077 | 0.6026 | 0.2871 | 0.0576 |
| 0.009 | p_b=0.25 | bneg | 0.0268 | 0.0030 | 3125 | 0.0000 | 0.0000 | 0.0168 | 0.0021 | 0.0436 | 0.0048 | 0.3974 | 0.5215 | 0.0593 |
| 0.009 | p_b=0.25 | mix | 0.0229 | 0.0030 | 3125 | 0.0029 | 0.0018 | 0.0149 | 0.0021 | 0.0407 | 0.0065 | 1.0000 | 0.3468 | 0.0566 |
| 0.009 | p_b=0.5 | bpos | 0.0167 | 0.0032 | 3125 | 0.0053 | 0.0032 | 0.0123 | 0.0022 | 0.0344 | 0.0082 | 0.6026 | 0.2386 | 0.0571 |
| 0.009 | p_b=0.5 | bneg | 0.0321 | 0.0031 | 3125 | 0.0000 | 0.0000 | 0.0200 | 0.0022 | 0.0521 | 0.0050 | 0.3974 | 0.5885 | 0.0606 |
| 0.009 | p_b=0.5 | mix | 0.0228 | 0.0032 | 3125 | 0.0032 | 0.0019 | 0.0154 | 0.0022 | 0.0414 | 0.0069 | 1.0000 | 0.3306 | 0.0563 |
| 0.009 | p_b=0.75 | bpos | 0.0119 | 0.0033 | 3125 | 0.0058 | 0.0033 | 0.0104 | 0.0022 | 0.0281 | 0.0085 | 0.6026 | 0.1855 | 0.0565 |
| 0.009 | p_b=0.75 | bneg | 0.0376 | 0.0032 | 3125 | 0.0000 | 0.0000 | 0.0233 | 0.0023 | 0.0609 | 0.0052 | 0.3974 | 0.6587 | 0.0615 |
| 0.009 | p_b=0.75 | mix | 0.0221 | 0.0033 | 3125 | 0.0035 | 0.0020 | 0.0155 | 0.0022 | 0.0411 | 0.0072 | 1.0000 | 0.3113 | 0.0555 |
| 0.02 | p_b=0 | bpos | 0.0227 | 0.0028 | 3125 | 0.0012 | 0.0027 | 0.0143 | 0.0020 | 0.0382 | 0.0072 | 0.7161 | 0.3049 | 0.0578 |
| 0.02 | p_b=0 | bneg | 0.0222 | 0.0028 | 3125 | 0.0000 | 0.0000 | 0.0140 | 0.0020 | 0.0362 | 0.0045 | 0.2839 | 0.4608 | 0.0583 |
| 0.02 | p_b=0 | mix | 0.0226 | 0.0028 | 3125 | 0.0008 | 0.0020 | 0.0143 | 0.0020 | 0.0377 | 0.0064 | 1.0000 | 0.3301 | 0.0568 |
| 0.02 | p_b=0.25 | bpos | 0.0196 | 0.0030 | 3125 | 0.0015 | 0.0029 | 0.0132 | 0.0021 | 0.0343 | 0.0077 | 0.7161 | 0.2547 | 0.0573 |
| 0.02 | p_b=0.25 | bneg | 0.0265 | 0.0030 | 3125 | 0.0000 | 0.0000 | 0.0166 | 0.0021 | 0.0431 | 0.0048 | 0.2839 | 0.5142 | 0.0593 |
| 0.02 | p_b=0.25 | mix | 0.0215 | 0.0030 | 3125 | 0.0011 | 0.0021 | 0.0142 | 0.0021 | 0.0368 | 0.0069 | 1.0000 | 0.3005 | 0.0565 |
| 0.02 | p_b=0.5 | bpos | 0.0151 | 0.0032 | 3125 | 0.0019 | 0.0031 | 0.0114 | 0.0022 | 0.0284 | 0.0082 | 0.7161 | 0.1979 | 0.0567 |
| 0.02 | p_b=0.5 | bneg | 0.0313 | 0.0031 | 3125 | 0.0000 | 0.0000 | 0.0196 | 0.0022 | 0.0510 | 0.0050 | 0.2839 | 0.5736 | 0.0605 |
| 0.02 | p_b=0.5 | mix | 0.0197 | 0.0032 | 3125 | 0.0014 | 0.0023 | 0.0138 | 0.0022 | 0.0348 | 0.0073 | 1.0000 | 0.2662 | 0.0559 |
| 0.02 | p_b=0.75 | bpos | 0.0093 | 0.0033 | 3125 | 0.0022 | 0.0033 | 0.0090 | 0.0022 | 0.0205 | 0.0085 | 0.7161 | 0.1359 | 0.0560 |
| 0.02 | p_b=0.75 | bneg | 0.0363 | 0.0032 | 3125 | 0.0000 | 0.0000 | 0.0227 | 0.0023 | 0.0590 | 0.0052 | 0.2839 | 0.6360 | 0.0614 |
| 0.02 | p_b=0.75 | mix | 0.0170 | 0.0033 | 3125 | 0.0016 | 0.0024 | 0.0129 | 0.0022 | 0.0314 | 0.0075 | 1.0000 | 0.2278 | 0.0550 |

- simulation: n_draws 1000, n_eras 3125, p_c 0.0, bench_fit_sd 0.035
- chosen p_b = 0.00, p_c = 0.00, rule = nnls
- weights: lgbm_everest 0.255, ridge_everest 0.745, lgbm_tougroute 0.000, lgbm_ghat 0.000, lgbm_meltsene 0.000, lgbm_timesguida 0.000, lgbm_ighil 0.000, lgbm_tarhat 0.000, xgb_everest 0.000, xgb_tougroute 0.000, xgb_ghat 0.000, xgb_meltsene 0.000, xgb_timesguida 0.000, xgb_ighil 0.000, xgb_tarhat 0.000, cat_everest_l2_3 0.000, cat_everest_l2_30 0.000, cat_everest_l2_100 0.000, lgbm_everest_sharpe 0.000, lgbm_everest_bag0 0.000, lgbm_everest_bag1 0.000, lgbm_everest_bag2 0.000, lgbm_everest_bag3 0.000, mlp_everest 0.000
- paired NNLS vs EW: diff_mean 0.0007, diff_se 0.0004, p_pos 0.93

## Pipeline (scripts/run_pipeline.sh attempts and steps)

pipeline status **ok** (exit 0); 1 attempt(s), 2841 s of driver wall time; mlp: ok

| attempt | step | from | status | exit | seconds | note |
|---|---|---|---|---|---|---|
| 1 | (attempt) | train | ok | 0 | 2841 | - |
| 1 | train |  | ok | 0 | 1998 | - |
| 1 | mlp |  | ok | 0 | 73 | - |
| 1 | retrain |  | ok | 0 | 23 | - |
| 1 | blend |  | ok | 0 | 730 | - |
| 1 | audit |  | ok | 0 | 12 | - |
| 1 | bundle |  | ok | 0 | 0 | - |
| 1 | pkl_check |  | ok | 0 | 1 | - |
| 1 | diag |  | ok | 0 | 4 | - |

## Evidence behind the status

| item | status | required | detail |
|---|---|---|---|
| manifest | ok | yes | full01.manifest.json sha256 c693300d81d1 = pickle bytes |
| pkl_check | ok | yes | PASSED on these bytes (python 3.11.18, 51 eras, slowest call 0.32 s) |
| diag_file | ok | yes | 125931 rows / 1392 eras, full coverage; built, not uploaded |
| audit | ok | yes | frozen model scored on the reserved eras (informational) |
| mlp | info | no | ok |

## Audit — reserved eras, frozen model (informational; never fed back)

- reserved audit eras: the frozen model scored once after selection; informational, never fed back.
- status **ok**; eras 5945..6524 (580 eras, 53292 rows; development used eras < 5925)
- frozen blend: rule nnls, arms lgbm, ridge, p_b 0.00, p_c 0.00 (simulation); unneutralised FIT below
- verdict: technical **complete**; audit win **no** — no audit win: tie or loss against the unsearched baseline on the reserved eras (diff -0.0042, p_pos 0.041, p_value 0.948): technical completion only

| metric | blend | baseline lgbm_everest |
|---|---|---|
| fit20 mean (se) | 0.0266 (0.0070) | 0.0308 (0.0062) |
| sharpe | 0.557 | 0.696 |
| pct_pos | 0.74 | 0.77 |
| frac 20-era means < 0 | 0.23 | 0.17 |
| max drawdown (cum. fit20) | 1.5745 | 1.5016 |
| longest underwater (eras) | 98 | 120 |

- paired fit20 blend - baseline: diff -0.0042 (se 0.0025), p_pos 0.04, p_value 0.948, 29 blocks, inference valid yes
- paired Sharpe: dsharpe -0.139 (se 0.068), p_pos 0.02
- drawdown: cumulative per-era fit20 drawdown (NOT platform P&L: no stake, no arctan bound, no UNQ/INOV)

chronological stability:

| eras | n_eras | blend fit | baseline fit | diff |
|---|---|---|---|---|
| 5945..6089 | 145 | 0.0209 | 0.0388 | -0.0180 |
| 6090..6234 | 145 | 0.0464 | 0.0454 | 0.0010 |
| 6235..6379 | 145 | 0.0407 | 0.0390 | 0.0017 |
| 6380..6524 | 145 | -0.0017 | -0.0001 | -0.0015 |

bootstrap block-length sensitivity (paired diff):

| block | se | p_pos | p_value | ci90 | blocks | valid |
|---|---|---|---|---|---|---|
| 10 | 0.0055 | 0.02 | 0.980 | [-0.0075, -0.0009] | 58 | yes |
| 20 | 0.0070 | 0.04 | 0.948 | [-0.0085, -0.0002] | 29 | yes |
| 40 | 0.0080 | 0.08 | 0.912 | [-0.0093, 0.0006] | 14 | yes |

- scenario (frozen constants only, simulated benchmark mean 0.009): mix fit 0.0266, unq 0.0045, inov 0.0143, score 0.0454, score Sharpe 0.435; synthetic benchmark (N(mean, sd) per era, surrogate direction) on the reserved eras at the FROZEN constants only; not a selection input
- note: reserved eras 5945..6524: no fit, early stop, knob, search, ladder, NNLS or neutralisation choice of this run saw them (development and full refits stop before 5925)
- note: independent of THIS run's selection only: handoff 02 (test01) has already looked at these eras
- note: scored once after the blend was frozen; audit.json never feeds back into any choice
- note: unneutralised: the frozen blend before the benchmark / core-proxy steps (the train-split benchmark is in-sample)

## Packaging evidence (manifest + pkl_check)

| item | value |
|---|---|
| manifest | full01.manifest.json |
| pickle file present / bytes = manifest sha256 | yes / yes |
| size MB | 0.19 |
| built under python | 3.11.18 |
| members | lgbm_everest, ridge_everest |
| admitted arms | lgbm, ridge |
| rule / p_b / p_c (p_source) | nnls / 0.00 / 0.00 (simulation) |
| pkl_check | PASSED |
| pkl_check bound to these bytes | yes |
| checker python / eras / slowest call s | 3.11.18 / 51 / 0.32 |
| isolation (stealth / torch importable) | no / no |

## Diagnostics FILE (built, NOT uploaded)

| item | value |
|---|---|
| file | full01_validation.parquet |
| rows | 125931 / 125931 |
| eras | 1392 / 1392 |
| full coverage | yes |
| benchmark used (eras) | 0 |
| built from the bundled pickle | yes |
| upload recorded | no |

- built only; uploading it is a separate USER MUST APPROVE handoff

## Training universe (knobs, decided once on the development forward block) — selection estimates (reused for selection; optimistic)

chosen universe: `e3-none-d0` (default `e2800-none-d0`; knob member lgbm_everest; Holm alpha 0.20; status done)

| candidate | dimension | fit_cv | fit_fwd | diff_fwd | diff_se | p_pos_fwd | diff_cv_indep | holm_rank | holm_thr | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| e2800-none-d0 | - | 0.0191 | 0.0146 | 0.0000 | 0.0000 | 0.00 | 0.0000 | - | - | - |
| e2000-none-d0 | era_start | 0.0207 | 0.0183 | 0.0036 | 0.0019 | 0.97 | 0.0012 | 2 | 0.050 | yes |
| e3-none-d0 | era_start | 0.0211 | 0.0204 | 0.0058 | 0.0021 | 1.00 | 0.0015 | 1 | 0.040 | yes |
| e2800-filter80-d0 | rows_rule | 0.0167 | 0.0082 | -0.0064 | 0.0023 | 0.00 | -0.0017 | 5 | 0.200 | no |
| e2800-weight80w0.5-d0 | rows_rule | 0.0180 | 0.0130 | -0.0016 | 0.0011 | 0.08 | -0.0008 | 4 | 0.100 | no |
| e2800-none-d1 | time_decay | 0.0182 | 0.0138 | -0.0009 | 0.0008 | 0.14 | -0.0009 | 3 | 0.067 | no |

## Search adoption (top-k vs base cell; paired gate) — selection estimates (reused for selection; optimistic)

| member | n_trials | topk_fit | base_fit | gain | forward_gain | paired p_pos | adopted | suspect | reason |
|---|---|---|---|---|---|---|---|---|---|
| lgbm_everest | 25 | 0.0233 | 0.0211 | 0.0022 | -0.0010 | 1.00 | no | no | tie: forward gain -0.0010 < 0 -> base cell (simpler) |
| xgb_everest | 12 | 0.0218 | 0.0204 | 0.0013 | -0.0025 | 1.00 | no | no | tie: forward gain -0.0025 < 0 -> base cell (simpler) |

## Admission ladder (pre-registered order, one look per arm) — selection estimates (reused for selection; optimistic)

| arm | members | admitted | diff_mean | diff_se | p_pos | fwd_diff | dsharpe | p_pos_sharpe | corr_inc | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | lgbm_everest | yes | - | - | - | - | - | - | - | base arm (incumbent) |
| aux | lgbm_tougroute, lgbm_ghat, lgbm_meltsene, lgbm_timesguida, lgbm_ighil, lgbm_tarhat, xgb_tougroute, xgb_ghat, xgb_meltsene, xgb_timesguida, xgb_ighil, xgb_tarhat | no | -0.0002 | 0.0003 | 0.25 | -0.0014 | 0.005 | 0.73 | 0.925 | REJECT: corr 0.925 <= 0.95; fwd diff -0.0014 < 0; p_pos 0.250 < 0.8; dsharpe +0.005 < 0.05, p_pos_sharpe 0.730 |
| xgb | xgb_everest | no | -0.0001 | 0.0003 | 0.38 | -0.0003 | 0.001 | 0.57 | 0.925 | REJECT: corr 0.925 <= 0.95; fwd diff -0.0003 < 0; p_pos 0.375 < 0.8; dsharpe +0.001 < 0.05, p_pos_sharpe 0.574 |
| cat | cat_everest_l2_3 | no | -0.0001 | 0.0007 | 0.41 | -0.0018 | -0.013 | 0.22 | 0.798 | REJECT: corr 0.798 <= 0.95; fwd diff -0.0018 < 0; p_pos 0.410 < 0.8; dsharpe -0.013 < 0.05, p_pos_sharpe 0.220; dropped redundant: cat_everest_l2_30 (redundant(cat_everest_l2_3) corr 0.967), cat_everest_l2_100 (redundant(cat_everest_l2_3) corr 0.963) |
| ridge | ridge_everest | yes | 0.0015 | 0.0006 | 0.99 | 0.0013 | 0.019 | 0.92 | 0.856 | ADMIT (fit route): corr 0.856 <= 0.95; fwd diff +0.0013; p_pos 0.995 >= 0.8; dsharpe +0.019 < 0.05, p_pos_sharpe 0.920 |
| sharpe | lgbm_everest_sharpe | no | -0.0000 | 0.0002 | 0.45 | -0.0014 | 0.003 | 0.74 | 0.938 | REJECT: corr 0.938 <= 0.95; fwd diff -0.0014 < 0; p_pos 0.445 < 0.8; dsharpe +0.003 < 0.05, p_pos_sharpe 0.737 |
| bags | lgbm_everest_bag0, lgbm_everest_bag1, lgbm_everest_bag2, lgbm_everest_bag3 | no | 0.0003 | 0.0002 | 0.91 | -0.0006 | 0.003 | 0.70 | 0.950 | REJECT: corr 0.950 <= 0.95; fwd diff -0.0006 < 0; p_pos 0.914 >= 0.8; dsharpe +0.003 < 0.05, p_pos_sharpe 0.702 |
| mlp | mlp_everest | no | -0.0020 | 0.0008 | 0.01 | -0.0028 | -0.038 | 0.02 | 0.464 | REJECT: corr 0.464 <= 0.95; fwd diff -0.0028 < 0; p_pos 0.006 < 0.8; dsharpe -0.038 < 0.05, p_pos_sharpe 0.017 |

## Member correlation

max pair: cat_everest_l2_3 / cat_everest_l2_30 corr 0.967 (24 members; full matrix in NN-metrics.json `member_corr`)

| member a | member b | corr |
|---|---|---|
| cat_everest_l2_3 | cat_everest_l2_30 | 0.967 |
| cat_everest_l2_30 | cat_everest_l2_100 | 0.966 |
| cat_everest_l2_3 | cat_everest_l2_100 | 0.963 |
| lgbm_everest | lgbm_everest_sharpe | 0.939 |
| lgbm_everest | xgb_everest | 0.925 |

## Neutralisation constants (simulation, staking-aware)

- chosen p_b = 0.00, p_c = 0.00; p_source = **simulation** (z: crossfit)
- reason: 12/12 cells eligible; best mix score_sharpe 0.361 at bench mean 0.009; 6 in the tie set (within 0.02); W best 0.0363, 3 within 0.0005 -> simplest; chosen p_b=0 p_c=0: fit 0.0226 (vs 0.0226 at 0,0), score_sharpe 0.3599 (vs 0.3599), W 0.0362

| p_b | p_c | mix fit | mix score | score_sharpe | W (worst regime) | eligible | tie set |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.0226 | 0.0394 | 0.360 | 0.0362 | yes | yes |
| 0.25 | 0.00 | 0.0229 | 0.0407 | 0.347 | 0.0343 | yes | yes |
| 0.50 | 0.00 | 0.0228 | 0.0414 | 0.331 | 0.0284 | yes | no |
| 0.75 | 0.00 | 0.0221 | 0.0411 | 0.311 | 0.0205 | yes | no |
| 0.00 | 0.25 | 0.0225 | 0.0395 | 0.361 | 0.0363 | yes | yes |
| 0.25 | 0.25 | 0.0228 | 0.0408 | 0.347 | 0.0343 | yes | yes |
| 0.50 | 0.25 | 0.0228 | 0.0416 | 0.332 | 0.0285 | yes | no |
| 0.75 | 0.25 | 0.0220 | 0.0413 | 0.313 | 0.0206 | yes | no |
| 0.00 | 0.50 | 0.0223 | 0.0394 | 0.361 | 0.0363 | yes | yes |
| 0.25 | 0.50 | 0.0226 | 0.0408 | 0.348 | 0.0342 | yes | yes |
| 0.50 | 0.50 | 0.0226 | 0.0416 | 0.333 | 0.0284 | yes | no |
| 0.75 | 0.50 | 0.0219 | 0.0414 | 0.315 | 0.0206 | yes | no |

## Final selection — selection estimates (reused for selection; optimistic)

| item | value |
|---|---|
| admitted arms | lgbm, ridge |
| rule | nnls |
| p_b / p_c (p_source) | 0.00 / 0.00 (simulation) |
| OOF fit20 mean (se) | 0.0226 (0.0030) |
| OOF sharpe / pct_pos | 0.464 / 0.69 |
| forward fit20 | 0.0219 |
| central mix score | 0.0394 |
| central score Sharpe | 0.360 |
| W (worst regime score) | 0.0362 |

## Dropped members

| member | status | detail |
|---|---|---|
| cat_everest_l2_30 | redundant | corr 0.967 with cat_everest_l2_3 |
| cat_everest_l2_100 | redundant | corr 0.963 with cat_everest_l2_3 |
| lgbm_tougroute | not_admitted | arm aux: REJECT: corr 0.925 <= 0.95; fwd diff -0.0014 < 0; p_pos 0.250 < 0.8; dsharpe +0.005 < 0.05, p_pos_sharpe 0.730 |
| lgbm_ghat | not_admitted | arm aux: REJECT: corr 0.925 <= 0.95; fwd diff -0.0014 < 0; p_pos 0.250 < 0.8; dsharpe +0.005 < 0.05, p_pos_sharpe 0.730 |
| lgbm_meltsene | not_admitted | arm aux: REJECT: corr 0.925 <= 0.95; fwd diff -0.0014 < 0; p_pos 0.250 < 0.8; dsharpe +0.005 < 0.05, p_pos_sharpe 0.730 |
| lgbm_timesguida | not_admitted | arm aux: REJECT: corr 0.925 <= 0.95; fwd diff -0.0014 < 0; p_pos 0.250 < 0.8; dsharpe +0.005 < 0.05, p_pos_sharpe 0.730 |
| lgbm_ighil | not_admitted | arm aux: REJECT: corr 0.925 <= 0.95; fwd diff -0.0014 < 0; p_pos 0.250 < 0.8; dsharpe +0.005 < 0.05, p_pos_sharpe 0.730 |
| lgbm_tarhat | not_admitted | arm aux: REJECT: corr 0.925 <= 0.95; fwd diff -0.0014 < 0; p_pos 0.250 < 0.8; dsharpe +0.005 < 0.05, p_pos_sharpe 0.730 |
| xgb_tougroute | not_admitted | arm aux: REJECT: corr 0.925 <= 0.95; fwd diff -0.0014 < 0; p_pos 0.250 < 0.8; dsharpe +0.005 < 0.05, p_pos_sharpe 0.730 |
| xgb_ghat | not_admitted | arm aux: REJECT: corr 0.925 <= 0.95; fwd diff -0.0014 < 0; p_pos 0.250 < 0.8; dsharpe +0.005 < 0.05, p_pos_sharpe 0.730 |
| xgb_meltsene | not_admitted | arm aux: REJECT: corr 0.925 <= 0.95; fwd diff -0.0014 < 0; p_pos 0.250 < 0.8; dsharpe +0.005 < 0.05, p_pos_sharpe 0.730 |
| xgb_timesguida | not_admitted | arm aux: REJECT: corr 0.925 <= 0.95; fwd diff -0.0014 < 0; p_pos 0.250 < 0.8; dsharpe +0.005 < 0.05, p_pos_sharpe 0.730 |
| xgb_ighil | not_admitted | arm aux: REJECT: corr 0.925 <= 0.95; fwd diff -0.0014 < 0; p_pos 0.250 < 0.8; dsharpe +0.005 < 0.05, p_pos_sharpe 0.730 |
| xgb_tarhat | not_admitted | arm aux: REJECT: corr 0.925 <= 0.95; fwd diff -0.0014 < 0; p_pos 0.250 < 0.8; dsharpe +0.005 < 0.05, p_pos_sharpe 0.730 |
| xgb_everest | not_admitted | arm xgb: REJECT: corr 0.925 <= 0.95; fwd diff -0.0003 < 0; p_pos 0.375 < 0.8; dsharpe +0.001 < 0.05, p_pos_sharpe 0.574 |
| cat_everest_l2_3 | not_admitted | arm cat: REJECT: corr 0.798 <= 0.95; fwd diff -0.0018 < 0; p_pos 0.410 < 0.8; dsharpe -0.013 < 0.05, p_pos_sharpe 0.220; dropped redundant: cat_everest_l2_30 (redundant(cat_everest_l2_3) corr 0.967), cat_everest_l2_100 (redundant(cat_everest_l2_3) corr 0.963) |
| lgbm_everest_sharpe | not_admitted | arm sharpe: REJECT: corr 0.938 <= 0.95; fwd diff -0.0014 < 0; p_pos 0.445 < 0.8; dsharpe +0.003 < 0.05, p_pos_sharpe 0.737 |
| lgbm_everest_bag0 | not_admitted | arm bags: REJECT: corr 0.950 <= 0.95; fwd diff -0.0006 < 0; p_pos 0.914 >= 0.8; dsharpe +0.003 < 0.05, p_pos_sharpe 0.702 |
| lgbm_everest_bag1 | not_admitted | arm bags: REJECT: corr 0.950 <= 0.95; fwd diff -0.0006 < 0; p_pos 0.914 >= 0.8; dsharpe +0.003 < 0.05, p_pos_sharpe 0.702 |
| lgbm_everest_bag2 | not_admitted | arm bags: REJECT: corr 0.950 <= 0.95; fwd diff -0.0006 < 0; p_pos 0.914 >= 0.8; dsharpe +0.003 < 0.05, p_pos_sharpe 0.702 |
| lgbm_everest_bag3 | not_admitted | arm bags: REJECT: corr 0.950 <= 0.95; fwd diff -0.0006 < 0; p_pos 0.914 >= 0.8; dsharpe +0.003 < 0.05, p_pos_sharpe 0.702 |
| mlp_everest | not_admitted | arm mlp: REJECT: corr 0.464 <= 0.95; fwd diff -0.0028 < 0; p_pos 0.006 < 0.8; dsharpe -0.038 < 0.05, p_pos_sharpe 0.017 |

## Wall times

| step / member | wall seconds |
|---|---|
| pipeline step train (attempt 1, ok) | 1998 |
| pipeline step mlp (attempt 1, ok) | 73 |
| pipeline step retrain (attempt 1, ok) | 23 |
| pipeline step blend (attempt 1, ok) | 730 |
| pipeline step audit (attempt 1, ok) | 12 |
| pipeline step bundle (attempt 1, ok) | 0 |
| pipeline step pkl_check (attempt 1, ok) | 1 |
| pipeline step diag (attempt 1, ok) | 4 |
| pipeline total (all attempts) | 2841 |
| TOTAL (metrics.json, last train invocation) | 22.7 |
| cat_everest_l2_100 | 26.6 |
| cat_everest_l2_3 | 26.6 |
| cat_everest_l2_30 | 26.7 |
| lgbm_everest | 28.6 |
| lgbm_everest_bag0 | 17.2 |
| lgbm_everest_bag1 | 20.6 |
| lgbm_everest_bag2 | 15.3 |
| lgbm_everest_bag3 | 16.5 |
| lgbm_everest_sharpe | 37.0 |
| lgbm_ghat | 13.7 |
| lgbm_ighil | 23.4 |
| lgbm_meltsene | 22.2 |
| lgbm_tarhat | 17.4 |
| lgbm_timesguida | 18.7 |
| lgbm_tougroute | 25.3 |
| mlp_everest | 24.8 |
| ridge_everest | 2.4 |
| xgb_everest | 16.9 |
| xgb_ghat | 12.0 |
| xgb_ighil | 14.1 |
| xgb_meltsene | 17.8 |
| xgb_tarhat | 12.1 |
| xgb_timesguida | 14.2 |
| xgb_tougroute | 16.3 |

## Notes

- lgbm_everest: search gain +0.0022 on OOF is NOT confirmed on the forward block (forward gain -0.0010 < 0) -> treat the search as a tie
- xgb_everest: search gain +0.0013 on OOF is NOT confirmed on the forward block (forward gain -0.0025 < 0) -> treat the search as a tie
- development figures (CV OOF, forward block, knobs, search adoption, admission ladder, NNLS, neutralisation grid) are SELECTION ESTIMATES: the same eras choose and score, so they are optimistic. The independent check is audit.json on the reserved eras, which never feeds back into any choice.
- cat_everest_l2_30: dropped as redundant(cat_everest_l2_3) corr 0.967 > gates.member_corr_max
- cat_everest_l2_100: dropped as redundant(cat_everest_l2_3) corr 0.963 > gates.member_corr_max

## run.log (last 60 lines)

```text
2026-11-14 01:54:59,451 INFO stealth.data: loaded futures_train.parquet: 468236 rows, 6522 eras (3..6524)
2026-11-14 01:58:58,143 INFO stealth.metrics: regime mean=0 p_c=0: p_b=0 mix score 0.0400 sharpe 0.381; p_b=0.25 mix score 0.0429 sharpe 0.382; p_b=0.5 mix score 0.0458 sharpe 0.382; p_b=0.75 mix score 0.0479 sharpe 0.379
2026-11-14 01:58:59,246 INFO stealth.metrics: regime mean=0 p_c=0.25: p_b=0 mix score 0.0401 sharpe 0.381; p_b=0.25 mix score 0.0431 sharpe 0.383; p_b=0.5 mix score 0.0460 sharpe 0.384; p_b=0.75 mix score 0.0482 sharpe 0.382
2026-11-14 01:59:00,357 INFO stealth.metrics: regime mean=0 p_c=0.5: p_b=0 mix score 0.0400 sharpe 0.381; p_b=0.25 mix score 0.0431 sharpe 0.384; p_b=0.5 mix score 0.0461 sharpe 0.385; p_b=0.75 mix score 0.0483 sharpe 0.384
2026-11-14 02:02:58,016 INFO stealth.metrics: regime mean=0.009 p_c=0: p_b=0 mix score 0.0394 sharpe 0.360; p_b=0.25 mix score 0.0407 sharpe 0.347; p_b=0.5 mix score 0.0414 sharpe 0.331; p_b=0.75 mix score 0.0411 sharpe 0.311
2026-11-14 02:02:59,022 INFO stealth.metrics: regime mean=0.009 p_c=0.25: p_b=0 mix score 0.0395 sharpe 0.361; p_b=0.25 mix score 0.0408 sharpe 0.347; p_b=0.5 mix score 0.0416 sharpe 0.332; p_b=0.75 mix score 0.0413 sharpe 0.313
2026-11-14 02:03:00,041 INFO stealth.metrics: regime mean=0.009 p_c=0.5: p_b=0 mix score 0.0394 sharpe 0.361; p_b=0.25 mix score 0.0408 sharpe 0.348; p_b=0.5 mix score 0.0416 sharpe 0.333; p_b=0.75 mix score 0.0414 sharpe 0.315
2026-11-14 02:06:59,026 INFO stealth.metrics: regime mean=0.02 p_c=0: p_b=0 mix score 0.0377 sharpe 0.330; p_b=0.25 mix score 0.0368 sharpe 0.300; p_b=0.5 mix score 0.0348 sharpe 0.266; p_b=0.75 mix score 0.0314 sharpe 0.228
2026-11-14 02:07:00,122 INFO stealth.metrics: regime mean=0.02 p_c=0.25: p_b=0 mix score 0.0378 sharpe 0.331; p_b=0.25 mix score 0.0369 sharpe 0.301; p_b=0.5 mix score 0.0350 sharpe 0.267; p_b=0.75 mix score 0.0316 sharpe 0.229
2026-11-14 02:07:01,182 INFO stealth.metrics: regime mean=0.02 p_c=0.5: p_b=0 mix score 0.0377 sharpe 0.331; p_b=0.25 mix score 0.0368 sharpe 0.301; p_b=0.5 mix score 0.0350 sharpe 0.268; p_b=0.75 mix score 0.0316 sharpe 0.230
2026-11-14 02:07:01,182 INFO stealth.metrics: regime grid: 1000 draws x 3 means x 12 cells (4 p_b x 3 p_c) on 3125 eras in 721.5s
2026-11-14 02:07:01,188 INFO stealth.blend: regime grid: 1000 draws x 12 cells on 3125 eras in 721s (extrapolated 500 draws: 361s)
2026-11-14 02:07:01,190 INFO stealth.blend: regime grid (mix):
 bench_mean    p_c    p_b regime    fit    unq   inov  score  score_se  score_sharpe  n_eras   frac
     0.0000 0.0000 0.0000    mix 0.0226 0.0031 0.0143 0.0400    0.0059        0.3808    3125 1.0000
     0.0000 0.0000 0.2500    mix 0.0240 0.0035 0.0154 0.0429    0.0063        0.3822    3125 1.0000
     0.0000 0.0000 0.5000    mix 0.0254 0.0038 0.0166 0.0458    0.0066        0.3820    3125 1.0000
     0.0000 0.0000 0.7500    mix 0.0262 0.0041 0.0176 0.0479    0.0068        0.3794    3125 1.0000
     0.0000 0.2500 0.0000    mix 0.0225 0.0031 0.0145 0.0401    0.0059        0.3815    3125 1.0000
     0.0000 0.2500 0.2500    mix 0.0239 0.0034 0.0157 0.0431    0.0063        0.3830    3125 1.0000
     0.0000 0.2500 0.5000    mix 0.0253 0.0038 0.0169 0.0460    0.0066        0.3837    3125 1.0000
     0.0000 0.2500 0.7500    mix 0.0262 0.0041 0.0179 0.0482    0.0068        0.3818    3125 1.0000
     0.0000 0.5000 0.0000    mix 0.0223 0.0031 0.0146 0.0400    0.0059        0.3814    3125 1.0000
     0.0000 0.5000 0.2500    mix 0.0238 0.0034 0.0159 0.0431    0.0063        0.3836    3125 1.0000
     0.0000 0.5000 0.5000    mix 0.0251 0.0038 0.0171 0.0461    0.0066        0.3850    3125 1.0000
     0.0000 0.5000 0.7500    mix 0.0260 0.0041 0.0181 0.0483    0.0068        0.3837    3125 1.0000
     0.0090 0.0000 0.0000    mix 0.0226 0.0026 0.0143 0.0394    0.0061        0.3599    3125 1.0000
     0.0090 0.0000 0.2500    mix 0.0229 0.0029 0.0149 0.0407    0.0065        0.3468    3125 1.0000
     0.0090 0.0000 0.5000    mix 0.0228 0.0032 0.0154 0.0414    0.0069        0.3306    3125 1.0000
     0.0090 0.0000 0.7500    mix 0.0221 0.0035 0.0155 0.0411    0.0072        0.3113    3125 1.0000
     0.0090 0.2500 0.0000    mix 0.0225 0.0025 0.0145 0.0395    0.0062        0.3606    3125 1.0000
     0.0090 0.2500 0.2500    mix 0.0228 0.0029 0.0151 0.0408    0.0066        0.3474    3125 1.0000
     0.0090 0.2500 0.5000    mix 0.0228 0.0032 0.0156 0.0416    0.0069        0.3320    3125 1.0000
     0.0090 0.2500 0.7500    mix 0.0220 0.0035 0.0158 0.0413    0.0072        0.3133    3125 1.0000
     0.0090 0.5000 0.0000    mix 0.0223 0.0025 0.0146 0.0394    0.0062        0.3605    3125 1.0000
     0.0090 0.5000 0.2500    mix 0.0226 0.0028 0.0153 0.0408    0.0066        0.3477    3125 1.0000
     0.0090 0.5000 0.5000    mix 0.0226 0.0032 0.0158 0.0416    0.0069        0.3330    3125 1.0000
     0.0090 0.5000 0.7500    mix 0.0219 0.0035 0.0160 0.0414    0.0071        0.3148    3125 1.0000
     0.0200 0.0000 0.0000    mix 0.0226 0.0008 0.0143 0.0377    0.0064        0.3301    3125 1.0000
     0.0200 0.0000 0.2500    mix 0.0215 0.0011 0.0142 0.0368    0.0069        0.3005    3125 1.0000
     0.0200 0.0000 0.5000    mix 0.0197 0.0014 0.0138 0.0348    0.0073        0.2662    3125 1.0000
     0.0200 0.0000 0.7500    mix 0.0170 0.0016 0.0129 0.0314    0.0075        0.2278    3125 1.0000
     0.0200 0.2500 0.0000    mix 0.0225 0.0008 0.0145 0.0378    0.0064        0.3308    3125 1.0000
     0.0200 0.2500 0.2500    mix 0.0214 0.0011 0.0144 0.0369    0.0069        0.3009    3125 1.0000
     0.0200 0.2500 0.5000    mix 0.0196 0.0014 0.0140 0.0350    0.0073        0.2672    3125 1.0000
     0.0200 0.2500 0.7500    mix 0.0169 0.0016 0.0131 0.0316    0.0075        0.2293    3125 1.0000
     0.0200 0.5000 0.0000    mix 0.0223 0.0008 0.0146 0.0377    0.0064        0.3309    3125 1.0000
     0.0200 0.5000 0.2500    mix 0.0212 0.0010 0.0146 0.0368    0.0069        0.3009    3125 1.0000
     0.0200 0.5000 0.5000    mix 0.0194 0.0014 0.0142 0.0350    0.0072        0.2678    3125 1.0000
     0.0200 0.5000 0.7500    mix 0.0167 0.0016 0.0133 0.0316    0.0075        0.2304    3125 1.0000
2026-11-14 02:07:01,190 INFO stealth.blend: neutralisation: 12/12 cells eligible; best mix score_sharpe 0.361 at bench mean 0.009; 6 in the tie set (within 0.02); W best 0.0363, 3 within 0.0005 -> simplest; chosen p_b=0 p_c=0: fit 0.0226 (vs 0.0226 at 0,0), score_sharpe 0.3599 (vs 0.3599), W 0.0362
2026-11-14 02:07:01,190 INFO stealth.blend: final: arms ['lgbm', 'ridge'] rule nnls p_b=0 p_c=0 (simulation); OOF fit20 0.0226 forward 0.0219; central mix score 0.0394 score_sharpe 0.3599
2026-11-14 02:07:01,191 INFO stealth.metrics: leak_check pass: mean OOF Spearman 0.1010 <= 0.18
2026-11-14 02:07:01,196 INFO stealth.blend: wrote ~/Projects/eversteer/stealth/artifacts/full01/blend.json, member_corr.csv and updated metrics.json
2026-11-14 02:07:02,417 INFO stealth.data: loaded futures_train.parquet: 468236 rows, 6522 eras (3..6524)
2026-11-14 02:07:06,061 INFO stealth.metrics: regime mean=0 p_c=0: p_b=0 mix score 0.0457 sharpe 0.456
2026-11-14 02:07:09,603 INFO stealth.metrics: regime mean=0.009 p_c=0: p_b=0 mix score 0.0454 sharpe 0.435
2026-11-14 02:07:13,087 INFO stealth.metrics: regime mean=0.02 p_c=0: p_b=0 mix score 0.0439 sharpe 0.403
2026-11-14 02:07:13,087 INFO stealth.metrics: regime grid: 1000 draws x 3 means x 1 cells (1 p_b x 1 p_c) on 580 eras in 10.6s
2026-11-14 02:07:13,088 INFO stealth.audit: audit full01 eras 5945..6524: blend fit20 0.0266 (se 0.0070, sharpe 0.557) vs baseline 0.0308 (diff -0.0042, p_pos 0.041); drawdown 1.5745; no audit win: tie or loss against the unsearched baseline on the reserved eras (diff -0.0042, p_pos 0.041, p_value 0.948): technical completion only
```

## Operator notes

```text
Summary:
Step 3b: GPU rehearsal passed every stage; stale-MLP refusal exit 64 as expected; TERM interruption exit 143, no stray processes; resume exit 0 with attempts [interrupted, ok], final ok.
GPU: OK on NVIDIA GeForce RTX 5070 Ti Laptop GPU (sm_120). MLP: trained successfully on CUDA in 73 s, parity max 8.742886604129474e-07; not admitted by the ladder.
Pytest: 360 passed, 1 skipped in 108.14s (0:01:48); exit 0.
Chosen knobs universe: e3-none-d0.
LightGBM search: 25 distinct trials, not adopted; negative development forward gain, base cell retained. XGBoost search: 12 distinct trials, not adopted for the same gate outcome.
Admitted arms: lgbm, ridge; members lgbm_everest and ridge_everest; rule nnls with cross-fitted development OOF predictions.
p_b: 0.0; p_c: 0.0; p_source: simulation.
Selection estimates (reused for selection; optimistic): final OOF fit20 0.022570161315259105, block SE 0.003047143311984256; forward fit 0.021879666795741777; central score Sharpe 0.35988967073801414; W 0.03621411516307345.
verdict: no audit win: tie or loss against the unsearched baseline on the reserved eras (diff -0.0042, p_pos 0.041, p_value 0.948): technical completion only
Report status: ok; leak_check pass; pkl_check PASSED on the actual pickle bytes; validation diagnostics file 125931 rows, 1392 eras, not uploaded.
Total wall: real pipeline 47 min 21 s (2841 s, 01:19:57Z to 02:07:18Z); pipeline launch through offline dry run 47 min 56 s; preparation, re-checks and pipeline through offline dry run 57 min 01 s (3421 s), excluding reporting/commits.
Every fix made: none. No source-code or config edits; no unexpected failures. The deliberate stale-MLP refusal and TERM interruption are the required successful re-checks. Main pipeline completed in one attempt with zero resumes.

Detailed evidence:
Code commit: 1a0a969 (main at launch); same persistent Bash shell for all runbook blocks; nproc 20, configured threads 16; mains connected; selfcheck exit 0, PINS OK.
GPU setup tail:
device: NVIDIA GeForce RTX 5070 Ti Laptop GPU  compute capability sm_120  arch list ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
4096^2 matmul 3.444s (incl. transfer + first-call init), rel max err vs CPU 2.58e-06

GPU_PY=/home/om/Projects/eversteer/stealth/.venv-gpu/bin/python
GPU OK

Step 3b printed lines:
rehearsal exit 0
2026-11-14T01:17:38Z pipeline[recheck01]: PIPELINE OK run_id=recheck01 pkl=/home/om/Projects/eversteer/stealth/artifacts/recheck01.pkl mlp: ok
pipeline ok | mlp ok | audit ok [5945, 6524] | p_source simulation | pkl_check True bound True isolation {'stealth_importable': False, 'torch_importable': False}
stale-MLP check exit 64 (expect 64)
run_pipeline: --no-mlp --from blend: /home/om/Projects/eversteer/stealth/artifacts/recheck01/oof.parquet still contains external member(s) mlp_everest from an earlier attempt; re-run with --from retrain --no-mlp so it is re-assembled without them
interrupted attempt exit 143 (expect 143)
no stray processes
resume exit 0 (expect 0)
attempts ['interrupted', 'ok'] | final ok

Pipeline step exits:
2026-11-14T01:53:15Z pipeline[full01]: step train: exit 0 after 1998 s
2026-11-14T01:54:28Z pipeline[full01]: step mlp: exit 0 after 73 s
2026-11-14T01:54:51Z pipeline[full01]: step retrain: exit 0 after 23 s
2026-11-14T02:07:01Z pipeline[full01]: step blend: exit 0 after 730 s
2026-11-14T02:07:13Z pipeline[full01]: step audit: exit 0 after 12 s
2026-11-14T02:07:13Z pipeline[full01]: step bundle: exit 0 after 0 s
2026-11-14T02:07:14Z pipeline[full01]: step pkl_check: exit 0 after 1 s
2026-11-14T02:07:18Z pipeline[full01]: step diag: exit 0 after 4 s
2026-11-14T02:07:18Z pipeline[full01]: PIPELINE OK run_id=full01 pkl=/home/om/Projects/eversteer/stealth/models/full01.pkl mlp: ok

Step 7 packaging and audit:
p_source simulation | p_b 0.0 | p_c 0.0 | rule nnls | arms ['lgbm', 'ridge'] | MB 0.19
pkl_check passed True | on these bytes True | slowest s 0.3216879949977738 | isolation {'stealth_importable': False, 'torch_importable': False}
audit ok [5945, 6524] | blend fit20 0.0266 | baseline 0.0308 | diff -0.004206266635333212 | p_pos 0.0412
verdict: no audit win: tie or loss against the unsearched baseline on the reserved eras (diff -0.0042, p_pos 0.041, p_value 0.948): technical completion only

Step 8 offline dry run:
2026-11-14 02:07:53,565 INFO stealth.data: loaded futures_live.parquet: 90 rows, 1 eras (8055..8055)
2026-11-14 02:07:53,898 INFO stealth.predict: predicted exped=exped_8055 n=90 in 0.33s bench_used=False core_used=False unique=90
2026-11-14 02:07:53,900 INFO stealth.predict: dry run ok: exped=exped_8055 n=90 -> /home/om/Projects/eversteer/stealth/artifacts/predictions/exped_8055.parquet

Deviations: none. Read-only progress-log checks supplemented the scheduled status polls. .env presence checked only with test -s .env; its contents were never displayed. Nothing was uploaded, submitted or staked; no model slots or auto-submit settings changed. Re-check artifacts remain local and uncommitted.
```
