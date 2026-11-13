# Results 01 — smoke

- run_id: `smoke01`  config: `smoke`  status: **ok**
- git_sha: `ee2afbd5195504bab8b8cb23ff3475abe1686719`  started: 2026-11-13T20:16:25+00:00  finished: 2026-11-13T20:16:46+00:00
- generated: 2026-11-13T20:18:27+00:00 by `python -m stealth.report`

> **SUSPECT — winner's curse or leak:** ridge_everest exceed(s) the baseline by more than the suspect gain; do not celebrate.

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
| git_sha | ee2afbd5195504bab8b8cb23ff3475abe1686719 |
| core.hooksPath (reporting host) | scripts/hooks |
| pins (reporting host) | OK |
| libs | catboost 1.2.10, everesteer-api 0.4.1, lightgbm 4.7.0, numpy 2.4.6, optuna 5.0.0, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.17.1, xgboost 3.2.0 |

## Metrics (per-era fit20 on OOF; fit_se = 20-era block-bootstrap SE; config names only, no hyperparameter values)

| config | kind | fit_mean | fit_se | sharpe | pct_pos | ci90 | forward_fit | rho_b_val | n_iter | suspect | paired diff_mean | diff_se | p_pos | vs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm_everest | member | 0.0152 | 0.0039 | 0.34 | 0.64 | [0.0091, 0.0218] | - | 0.794 | 200 | no | - | - | - | - |
| lgbm_tougroute | member | 0.0087 | 0.0039 | 0.20 | 0.60 | [0.0022, 0.0151] | - | 0.660 | 200 | no | -0.0065 | 0.0021 | 0.00 | lgbm_everest |
| ridge_everest | member | 0.0241 | 0.0039 | 0.53 | 0.71 | [0.0177, 0.0302] | - | 0.742 | - | yes | 0.0089 | 0.0026 | 1.00 | lgbm_everest |
| blend_ew | blend | 0.0151 | 0.0040 | 0.34 | 0.63 | [0.0086, 0.0216] | - | - | - | - | - | - | - | - |

## Paired block-bootstrap tests

| A_vs_B | diff_mean | diff_se | p_pos | ci90 | n_eras | verdict |
|---|---|---|---|---|---|---|
| lgbm_tougroute_vs_lgbm_everest | -0.0065 | 0.0021 | 0.00 | [-0.0102, -0.0029] | 1525 | tie — keep the simpler one |
| ridge_everest_vs_lgbm_everest | 0.0089 | 0.0026 | 1.00 | [0.0047, 0.0129] | 1525 | adopt A (p_pos >= 0.8) |

## Regime simulation (blend.json)

| bench_fit_mean | p_b | regime | fit | fit_se | n_eras | unq | unq_se | inov | inov_se | score | score_se | frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | p_b=0 | bpos | 0.0154 | 0.0041 | 1525 | 0.0011 | 0.0041 | 0.0071 | 0.0025 | 0.0236 | 0.0103 | 0.5011 |
| 0 | p_b=0 | bneg | 0.0147 | 0.0041 | 1525 | 0.0000 | 0.0000 | 0.0068 | 0.0025 | 0.0215 | 0.0063 | 0.4989 |
| 0 | p_b=0 | mix | 0.0150 | 0.0041 | 1525 | 0.0005 | 0.0020 | 0.0069 | 0.0025 | 0.0225 | 0.0083 | 1.0000 |
| 0 | p_b=0.25 | bpos | 0.0132 | 0.0043 | 1525 | 0.0013 | 0.0043 | 0.0063 | 0.0026 | 0.0208 | 0.0108 | 0.5011 |
| 0 | p_b=0.25 | bneg | 0.0184 | 0.0043 | 1525 | 0.0000 | 0.0000 | 0.0089 | 0.0026 | 0.0273 | 0.0066 | 0.4989 |
| 0 | p_b=0.25 | mix | 0.0158 | 0.0043 | 1525 | 0.0007 | 0.0022 | 0.0076 | 0.0026 | 0.0240 | 0.0087 | 1.0000 |
| 0 | p_b=0.5 | bpos | 0.0101 | 0.0045 | 1525 | 0.0016 | 0.0045 | 0.0051 | 0.0027 | 0.0168 | 0.0113 | 0.5011 |
| 0 | p_b=0.5 | bneg | 0.0230 | 0.0045 | 1525 | 0.0000 | 0.0000 | 0.0114 | 0.0028 | 0.0344 | 0.0069 | 0.4989 |
| 0 | p_b=0.5 | mix | 0.0165 | 0.0045 | 1525 | 0.0008 | 0.0023 | 0.0082 | 0.0027 | 0.0256 | 0.0091 | 1.0000 |
| 0 | p_b=0.75 | bpos | 0.0063 | 0.0047 | 1525 | 0.0018 | 0.0046 | 0.0036 | 0.0028 | 0.0117 | 0.0117 | 0.5011 |
| 0 | p_b=0.75 | bneg | 0.0276 | 0.0045 | 1525 | 0.0000 | 0.0000 | 0.0141 | 0.0029 | 0.0417 | 0.0070 | 0.4989 |
| 0 | p_b=0.75 | mix | 0.0170 | 0.0046 | 1525 | 0.0009 | 0.0023 | 0.0088 | 0.0028 | 0.0267 | 0.0093 | 1.0000 |
| 0.009 | p_b=0 | bpos | 0.0151 | 0.0041 | 1525 | -0.0008 | 0.0040 | 0.0069 | 0.0025 | 0.0213 | 0.0103 | 0.6045 |
| 0.009 | p_b=0 | bneg | 0.0149 | 0.0041 | 1525 | 0.0000 | 0.0000 | 0.0070 | 0.0025 | 0.0218 | 0.0063 | 0.3955 |
| 0.009 | p_b=0 | mix | 0.0150 | 0.0041 | 1525 | -0.0005 | 0.0025 | 0.0069 | 0.0025 | 0.0215 | 0.0087 | 1.0000 |
| 0.009 | p_b=0.25 | bpos | 0.0126 | 0.0043 | 1525 | -0.0006 | 0.0043 | 0.0059 | 0.0026 | 0.0178 | 0.0108 | 0.6045 |
| 0.009 | p_b=0.25 | bneg | 0.0183 | 0.0043 | 1525 | 0.0000 | 0.0000 | 0.0089 | 0.0026 | 0.0272 | 0.0066 | 0.3955 |
| 0.009 | p_b=0.25 | mix | 0.0148 | 0.0043 | 1525 | -0.0004 | 0.0026 | 0.0071 | 0.0026 | 0.0216 | 0.0091 | 1.0000 |
| 0.009 | p_b=0.5 | bpos | 0.0090 | 0.0045 | 1525 | -0.0004 | 0.0045 | 0.0044 | 0.0027 | 0.0131 | 0.0113 | 0.6045 |
| 0.009 | p_b=0.5 | bneg | 0.0225 | 0.0045 | 1525 | 0.0000 | 0.0000 | 0.0113 | 0.0028 | 0.0338 | 0.0069 | 0.3955 |
| 0.009 | p_b=0.5 | mix | 0.0144 | 0.0045 | 1525 | -0.0002 | 0.0027 | 0.0072 | 0.0027 | 0.0213 | 0.0096 | 1.0000 |
| 0.009 | p_b=0.75 | bpos | 0.0048 | 0.0046 | 1525 | -0.0002 | 0.0046 | 0.0027 | 0.0027 | 0.0072 | 0.0116 | 0.6045 |
| 0.009 | p_b=0.75 | bneg | 0.0267 | 0.0046 | 1525 | 0.0000 | 0.0000 | 0.0138 | 0.0029 | 0.0405 | 0.0071 | 0.3955 |
| 0.009 | p_b=0.75 | mix | 0.0135 | 0.0046 | 1525 | -0.0001 | 0.0028 | 0.0071 | 0.0028 | 0.0204 | 0.0098 | 1.0000 |
| 0.02 | p_b=0 | bpos | 0.0151 | 0.0041 | 1525 | -0.0034 | 0.0040 | 0.0069 | 0.0025 | 0.0185 | 0.0102 | 0.7193 |
| 0.02 | p_b=0 | bneg | 0.0150 | 0.0041 | 1525 | 0.0000 | 0.0000 | 0.0071 | 0.0025 | 0.0220 | 0.0063 | 0.2807 |
| 0.02 | p_b=0 | mix | 0.0150 | 0.0041 | 1525 | -0.0025 | 0.0029 | 0.0069 | 0.0025 | 0.0195 | 0.0091 | 1.0000 |
| 0.02 | p_b=0.25 | bpos | 0.0119 | 0.0043 | 1525 | -0.0034 | 0.0042 | 0.0056 | 0.0026 | 0.0142 | 0.0107 | 0.7193 |
| 0.02 | p_b=0.25 | bneg | 0.0182 | 0.0043 | 1525 | 0.0000 | 0.0000 | 0.0089 | 0.0026 | 0.0270 | 0.0066 | 0.2807 |
| 0.02 | p_b=0.25 | mix | 0.0137 | 0.0043 | 1525 | -0.0024 | 0.0030 | 0.0065 | 0.0026 | 0.0178 | 0.0096 | 1.0000 |
| 0.02 | p_b=0.5 | bpos | 0.0077 | 0.0045 | 1525 | -0.0033 | 0.0044 | 0.0038 | 0.0027 | 0.0082 | 0.0112 | 0.7193 |
| 0.02 | p_b=0.5 | bneg | 0.0220 | 0.0045 | 1525 | 0.0000 | 0.0000 | 0.0111 | 0.0028 | 0.0331 | 0.0069 | 0.2807 |
| 0.02 | p_b=0.5 | mix | 0.0117 | 0.0045 | 1525 | -0.0024 | 0.0032 | 0.0058 | 0.0027 | 0.0152 | 0.0100 | 1.0000 |
| 0.02 | p_b=0.75 | bpos | 0.0026 | 0.0046 | 1525 | -0.0032 | 0.0046 | 0.0016 | 0.0027 | 0.0010 | 0.0115 | 0.7193 |
| 0.02 | p_b=0.75 | bneg | 0.0259 | 0.0046 | 1525 | 0.0000 | 0.0000 | 0.0134 | 0.0029 | 0.0393 | 0.0071 | 0.2807 |
| 0.02 | p_b=0.75 | mix | 0.0091 | 0.0046 | 1525 | -0.0023 | 0.0033 | 0.0049 | 0.0027 | 0.0117 | 0.0102 | 1.0000 |

- simulation: n_draws 50, n_eras 1525, p_c 0.25, bench_fit_sd 0.035
- chosen p_b = 0.50, p_c = 0.25, rule = ew
- weights: lgbm_everest 0.250, ridge_everest 0.250, lgbm_tougroute 0.500

## Wall times

| step / member | wall seconds |
|---|---|
| TOTAL | 21.8 |
| lgbm_everest | 4.2 |
| lgbm_tougroute | 4.2 |
| ridge_everest | 0.4 |

## Notes

- SUSPECT: ridge_everest OOF fit 0.0241 > lgbm_everest + 0.004

## run.log (last 60 lines)

```text
2026-11-13 20:16:42,071 INFO stealth.data: loaded futures_validation.parquet: 125931 rows, 1392 eras (6566..7957)
2026-11-13 20:16:42,407 INFO stealth.train: member ridge_everest done in 1s (frozen n_iter=None, rho_b(val)=0.7418)
2026-11-13 20:16:44,166 INFO stealth.metrics: leak_check pass: mean OOF Spearman 0.0744 <= 0.18
2026-11-13 20:16:45,506 INFO stealth.metrics: leak_check pass: mean OOF Spearman 0.0408 <= 0.18
2026-11-13 20:16:46,852 INFO stealth.metrics: leak_check pass: mean OOF Spearman 0.1108 <= 0.18
2026-11-13 20:16:46,865 INFO stealth.train: member lgbm_everest           fit20 mean=0.0152 se_block=0.0039 sharpe=0.342 pct_pos=0.64 forward=None n_iter=200
2026-11-13 20:16:46,865 INFO stealth.train: member lgbm_tougroute         fit20 mean=0.0087 se_block=0.0039 sharpe=0.200 pct_pos=0.60 forward=None n_iter=200
2026-11-13 20:16:46,865 INFO stealth.train: member ridge_everest          fit20 mean=0.0241 se_block=0.0039 sharpe=0.534 pct_pos=0.71 forward=None n_iter=None  SUSPECT
2026-11-13 20:16:46,865 INFO stealth.train: run smoke01 finished: status=ok exit=0 total=22s
2026-11-13 20:16:53,561 INFO stealth.blend: members ['lgbm_everest', 'lgbm_tougroute', 'ridge_everest']; equal weights {lgbm_everest: 0.250, ridge_everest: 0.250, lgbm_tougroute: 0.500}
2026-11-13 20:16:57,389 INFO stealth.blend: blend[ew]: fit20 mean=0.0151 se_block=0.0040 sharpe=0.340 pct_pos=0.63 forward=None (ew mean=0.0151)
2026-11-13 20:16:57,390 INFO stealth.metrics: leak_check pass: mean OOF Spearman 0.0707 <= 0.18
2026-11-13 20:16:57,871 INFO stealth.data: loaded futures_train.parquet: 468236 rows, 6522 eras (3..6524)
2026-11-13 20:17:00,912 INFO stealth.metrics: regime mean=0: p_b=0 mix score 0.0225; p_b=0.25 mix score 0.0240; p_b=0.5 mix score 0.0256; p_b=0.75 mix score 0.0267
2026-11-13 20:17:03,531 INFO stealth.metrics: regime mean=0.009: p_b=0 mix score 0.0215; p_b=0.25 mix score 0.0216; p_b=0.5 mix score 0.0213; p_b=0.75 mix score 0.0204
2026-11-13 20:17:06,073 INFO stealth.metrics: regime mean=0.02: p_b=0 mix score 0.0195; p_b=0.25 mix score 0.0178; p_b=0.5 mix score 0.0152; p_b=0.75 mix score 0.0117
2026-11-13 20:17:06,073 INFO stealth.blend: regime simulation: 50 draws in 8s
2026-11-13 20:17:06,075 INFO stealth.blend: regime bpos:
 bench_mean    p_b regime    fit     unq   inov  score  n_eras   frac
     0.0000 0.0000   bpos 0.0154  0.0011 0.0071 0.0236    1525 0.5011
     0.0000 0.2500   bpos 0.0132  0.0013 0.0063 0.0208    1525 0.5011
     0.0000 0.5000   bpos 0.0101  0.0016 0.0051 0.0168    1525 0.5011
     0.0000 0.7500   bpos 0.0063  0.0018 0.0036 0.0117    1525 0.5011
     0.0090 0.0000   bpos 0.0151 -0.0008 0.0069 0.0213    1525 0.6045
     0.0090 0.2500   bpos 0.0126 -0.0006 0.0059 0.0178    1525 0.6045
     0.0090 0.5000   bpos 0.0090 -0.0004 0.0044 0.0131    1525 0.6045
     0.0090 0.7500   bpos 0.0048 -0.0002 0.0027 0.0072    1525 0.6045
     0.0200 0.0000   bpos 0.0151 -0.0034 0.0069 0.0185    1525 0.7193
     0.0200 0.2500   bpos 0.0119 -0.0034 0.0056 0.0142    1525 0.7193
     0.0200 0.5000   bpos 0.0077 -0.0033 0.0038 0.0082    1525 0.7193
     0.0200 0.7500   bpos 0.0026 -0.0032 0.0016 0.0010    1525 0.7193
2026-11-13 20:17:06,075 INFO stealth.blend: regime bneg:
 bench_mean    p_b regime    fit    unq   inov  score  n_eras   frac
     0.0000 0.0000   bneg 0.0147 0.0000 0.0068 0.0215    1525 0.4989
     0.0000 0.2500   bneg 0.0184 0.0000 0.0089 0.0273    1525 0.4989
     0.0000 0.5000   bneg 0.0230 0.0000 0.0114 0.0344    1525 0.4989
     0.0000 0.7500   bneg 0.0276 0.0000 0.0141 0.0417    1525 0.4989
     0.0090 0.0000   bneg 0.0149 0.0000 0.0070 0.0218    1525 0.3955
     0.0090 0.2500   bneg 0.0183 0.0000 0.0089 0.0272    1525 0.3955
     0.0090 0.5000   bneg 0.0225 0.0000 0.0113 0.0338    1525 0.3955
     0.0090 0.7500   bneg 0.0267 0.0000 0.0138 0.0405    1525 0.3955
     0.0200 0.0000   bneg 0.0150 0.0000 0.0071 0.0220    1525 0.2807
     0.0200 0.2500   bneg 0.0182 0.0000 0.0089 0.0270    1525 0.2807
     0.0200 0.5000   bneg 0.0220 0.0000 0.0111 0.0331    1525 0.2807
     0.0200 0.7500   bneg 0.0259 0.0000 0.0134 0.0393    1525 0.2807
2026-11-13 20:17:06,076 INFO stealth.blend: regime mix:
 bench_mean    p_b regime    fit     unq   inov  score  n_eras   frac
     0.0000 0.0000    mix 0.0150  0.0005 0.0069 0.0225    1525 1.0000
     0.0000 0.2500    mix 0.0158  0.0007 0.0076 0.0240    1525 1.0000
     0.0000 0.5000    mix 0.0165  0.0008 0.0082 0.0256    1525 1.0000
     0.0000 0.7500    mix 0.0170  0.0009 0.0088 0.0267    1525 1.0000
     0.0090 0.0000    mix 0.0150 -0.0005 0.0069 0.0215    1525 1.0000
     0.0090 0.2500    mix 0.0148 -0.0004 0.0071 0.0216    1525 1.0000
     0.0090 0.5000    mix 0.0144 -0.0002 0.0072 0.0213    1525 1.0000
     0.0090 0.7500    mix 0.0135 -0.0001 0.0071 0.0204    1525 1.0000
     0.0200 0.0000    mix 0.0150 -0.0025 0.0069 0.0195    1525 1.0000
     0.0200 0.2500    mix 0.0137 -0.0024 0.0065 0.0178    1525 1.0000
     0.0200 0.5000    mix 0.0117 -0.0024 0.0058 0.0152    1525 1.0000
     0.0200 0.7500    mix 0.0091 -0.0023 0.0049 0.0117    1525 1.0000
2026-11-13 20:17:06,078 INFO stealth.blend: wrote /home/om/Projects/eversteer/stealth/artifacts/smoke01/blend.json and updated metrics.json
```
