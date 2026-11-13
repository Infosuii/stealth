# Everesteer "atlas" futures dataset — data profile (scout report)

Generated 2026-11-13 on the sandbox (2 cores / 9 GB). All numbers below are measured, not assumed.
Scripts: /home/climber/.claude/jobs/8bc89d82/tmp/profile{1,2,3_lgbm,4_variants,5}.py ; raw outputs profile*.out ;
per-feature table feature_target_corr.csv ; baseline JSON lgbm_baseline.json, variants.json.

## 0. Headline findings (read this first)

1. Data is SMALL: train 468,236 rows x 178 int8 features = 83 MB in RAM (333 MB as float32). 6,522 eras, 44-92 instruments/era.
   LightGBM 300 trees on 323k rows takes 17 s on 2 threads. A multi-hour training run is only justified by large ensembles /
   CV sweeps, not by data size. GPU is irrelevant for tree models here.
2. Signal is WEAK and overfits fast: honest OOS per-era Spearman for a tuned LightGBM is ~0.12 (sharpe ~0.57); the default
   300-tree config scores 0.08 OOS vs 0.54 in-sample. Fewer trees / shallower trees / stronger regularization all help.
   5-seed bagging adds nothing (seed preds corr 0.985). A linear sign-sum of the top-20 features gets 0.120 — trees barely beat it.
3. The published benchmark v1_sherpa scores 0.35 per-era Spearman on the LAST 25% of train (0.44 in the last 500 eras),
   3x any model we can fit honestly, and it is a 0.66-R2 linear function of the features. This is almost certainly IN-SAMPLE
   (the benchmark was fit on the train split). Consequences: (a) do NOT use v1_sherpa over train as a feature/target/blend
   weight calibrator — it leaks; (b) any UNQ proxy computed on train is biased NEGATIVE (all our models show UNQ proxy
   ~ -0.10) and does not reflect live UNQ; (c) its true live strength is unknown — get_benchmarks returns an empty list.
4. Targets are ~20-era overlapping forward returns: ACF of the per-era corr series is 0.90 at lag 1, 0.47 at lag 10,
   0.08 at lag 20, ~0.06 after. Era-purged CV with a >=20-era embargo is mandatory; naive K-fold will look great and lie.
5. Instrument identity is hidden (ids are per-row opaque hex) BUT instruments can be re-identified across adjacent eras by
   nearest-neighbour on the 178 binned features: mutual-best-match rate 0.99-1.00 in every era pair tested (train and
   validation), mean Hamming distance 10-45/178 vs ~140 for random pairs. Lag/delta features and target-lag ideas are
   feasible (a "tracking" preprocessing step), though confidence drops for lag >= 5 (Hamming 66).
6. Early data is degraded: first decile of eras has 63% feature values missing and ~98 of 178 features entirely missing per
   era; from exped_2804 onward no feature is fully missing. Instrument count grows 44 -> 92. Consider down-weighting or
   dropping eras < ~2000-2800.
7. Era clock gap: train ends exped_6524, validation is 6566-7957 (1,392 eras), live round is exped_8055. If eras are
   trading days the training data ends ~6 years before live. Validation diagnostics (server-side scoring of validation
   preds) is the only honest recent-period feedback available; it is a submission-like action and was NOT performed here.

## 1. Files

| split | path | bytes | rows | cols |
|---|---|---|---|---|
| train | /home/climber/work/stealth/data/futures_train.parquet | 50,886,382 | 468,236 | 196 (178 feature + 15 target + exped + data_type + id index) |
| validation | /home/climber/work/stealth/data/futures_validation.parquet | 13,685,903 | 125,931 | 196 (targets all NaN) |
| live | /home/climber/work/stealth/data/futures_live.parquet | 131,356 | 90 | 196 (targets all NaN); exped_8055, round WAS open |
| train_benchmark_models | /home/climber/work/stealth/data/futures_train_benchmark_models.parquet | 8,778,492 | 468,236 | 3: exped, v1_sherpa (float64), id index |

Download via the preinstalled `everestapi` SDK (`EverestAPI().download_dataset(universe='futures', split=..., output_path=...)`),
already authenticated on the box; no URL/credential handling needed. Parquet: single row group, SNAPPY, pandas metadata
with index_columns=['id'].

Python on sandbox: 3.11.2 with pandas 3.0.6, pyarrow 24.0.0, numpy 2.4.6, scipy 1.17.1, lightgbm 4.7.0, scikit-learn 1.9.1
(all importable, no venv needed). NOTE pandas 3.x: strings load as dtype `str`; `groupby.apply` needs `include_groups=False`.

## 2. Schema / structure (train)

- Index: name `id`, 468,236 unique 16-char lowercase hex strings (e.g. `1a5165855490f41a`, `b18e66590dfa893a`). One row per
  (instrument, era). Ids do NOT encode instrument or era (8-char prefixes: 468,201 unique of 468,236). Zero id overlap
  between train/validation/live.
- Non-feature columns: `exped` (str, era label), `data_type` (str, constant 'train'/'validation'/'live'), 15 `target_*`
  (float32). No climb_difficulty column exists in this dataset. No `id` column (only the index).
- Features: 178 columns `feature_<MoroccanPlaceName>`, dtype int8, values -1 (missing) and 0..9. Within an era each
  feature is ~uniformly binned into deciles (e.g. mid era exped_3264, n=71: 6-7 rows per bin). Feature-set metadata: only
  'all' (verbose schema confirms; no core/small/medium sets exposed, so the INOV "core features" are not identifiable).
- Eras: 6,522 labels `exped_0003` .. `exped_6524`, zero-padded 4-digit, contiguous except 0001-0002 absent, file is sorted
  by era. Instruments/era: mean 71.8, median 70, min 44 (first ~10 eras), max 92 (last eras). Mean by time decile:
  47, 56, 61, 66, 70, 72, 81, 85, 88, 92 — the universe grows monotonically.
- Value distribution (all cells): -1: 10.8%, bins 0..9 each 8.4-9.5%.

## 3. Missingness

- Overall -1 rate 10.8%; 58.4% of rows have at least one -1; all 178 features have some missing.
- Worst features: Azilal 38.1%, Ahermoumou 36.3%, Zaouiat 36.3%, Moulouya 36.2%, Imlil 35.9%, Tislit 29.3%, Tinejdad 26.5%,
  Lalla 25.1%, Tanougha 24.5%, Hassane 21.7%, Ansous 21.0%, Amsouzert 20.2%, Argana/Tighassaline/Agadir 20.1%.
- Strong TIME trend (mean cell missing rate by era decile): 62.8%, 23.0%, 11.3%, 9.6%, 5.1%, 3.4%, 6.6%, 4.9%, 4.4%, 2.9%.
- Features entirely missing within an era, mean count by decile: 98.0, 18.9, 7.3, 5.0, 0.3, 0.1, 0, 0.1, 0.1, 0.1.
  First era with zero fully-missing features: exped_2804. 1,044 eras have >10% of features more than half missing.
- Validation missing rate 0.72%; live 3.06%. So the live regime is "almost complete data" — early train eras are a
  different regime.

## 4. Targets

- 15 targets, all 5-level {0, .25, .5, .75, 1} with per-era bell mass 5.1/20.2/49.5/20.2/5.1% (exact per era: per-era mean
  of target_everest = 0.5000 with std 0.0001 across eras, per-era std 0.2266) -> targets are per-era rank-binned.
- NaN rates: target_everest 0.00% (never NaN, no NaN eras); Meltsene 0.03, Ighil 0.06, Bougafer 0.31, Azurki/Tarhat 0.57,
  Igli/Timesguida 0.63, Ghat/Tougroute 0.69, Anghomar/Ayachi 0.90, Gourza 3.84, Saghro/Tiskiouine 4.39%.
- Per-era Spearman with target_everest (mean, std over 6.5k eras): Tougroute 0.859 (0.054), Ghat 0.600, Meltsene 0.590,
  Timesguida 0.568, Ighil 0.484, Tarhat 0.468, Igli 0.412, Bougafer 0.406, Anghomar 0.389, Azurki 0.334, Ayachi 0.291,
  Gourza 0.257, Saghro 0.250, Tiskiouine 0.186. Tougroute looks like the same return at a slightly different horizon;
  others are different horizons/definitions. In the baseline, training on Tougroute instead of everest HURT (0.090 vs 0.118).
- Horizon evidence: ACF of per-era benchmark-corr series at lags 1/2/3/5/10/15/19/20/21/25/30/40 =
  0.90/0.84/0.78/0.68/0.47/0.27/0.12/0.08/0.07/0.06/0.05/0.06; same shape for LGBM per-era corr (0.90 at lag 1, 0.04 at
  lag 20). -> ~20-era overlapping forward-return target, consistent with the 20-trading-day scoring horizon. Use embargo >= 20.

## 5. Feature -> target_everest (per-era Spearman, -1 treated as NaN, 6,522 eras)

Top 20 by |mean| (mean, std, sharpe=mean/std, missing rate, fraction of eras with the modal sign):

| feature | mean | std | sharpe | miss | sign-agree |
|---|---|---|---|---|---|
| Agouti | +0.0791 | 0.211 | +0.375 | 0.030 | 0.644 |
| Azzaden | -0.0766 | 0.205 | -0.373 | 0.044 | 0.631 |
| Zerrouka | +0.0752 | 0.252 | +0.299 | 0.052 | 0.596 |
| Ighzer | -0.0751 | 0.200 | -0.376 | 0.026 | 0.636 |
| Tazzarine | +0.0748 | 0.195 | +0.384 | 0.027 | 0.667 |
| Boutferda | +0.0747 | 0.212 | +0.352 | 0.021 | 0.644 |
| Amane | +0.0742 | 0.234 | +0.317 | 0.199 | 0.636 |
| Midelt | +0.0728 | 0.257 | +0.283 | 0.051 | 0.598 |
| Amazigh | -0.0699 | 0.199 | -0.351 | 0.141 | 0.587 |
| Tanaghmelt | +0.0663 | 0.256 | +0.259 | 0.091 | 0.602 |
| Takerkoust | +0.0654 | 0.251 | +0.261 | 0.091 | 0.602 |
| Tarhzirt | -0.0649 | 0.213 | -0.304 | 0.037 | 0.617 |
| Ouarzazate | -0.0616 | 0.226 | -0.272 | 0.196 | 0.516 |
| Tagmat | -0.0596 | 0.228 | -0.261 | 0.091 | 0.583 |
| Azaghar | -0.0579 | 0.216 | -0.269 | 0.048 | 0.603 |
| Tagoudiche | +0.0571 | 0.242 | +0.236 | 0.173 | 0.584 |
| Rich | -0.0554 | 0.187 | -0.297 | 0.147 | 0.563 |
| Tafraout | -0.0554 | 0.268 | -0.206 | 0.017 | 0.579 |
| Tafouk | +0.0549 | 0.203 | +0.270 | 0.075 | 0.578 |
| Imilchil | +0.0527 | 0.212 | +0.249 | 0.147 | 0.534 |

- Distribution of |mean| over 178 features: p10 0.0024, p25 0.0046, median 0.0128, p75 0.0237, p90 0.055, max 0.079.
  Counts: |mean|>0.01: 106; >0.02: 60; >0.03: 40. |sharpe|>0.1: 53; >0.2: 27; >0.3: 8. Sign-agreement >0.6: 11; >0.55: 33.
- Best single-feature sharpe is 0.38 (Tazzarine). Per-era std of any feature's corr is ~0.2 — the per-era noise floor.
- Time stability: correlation between 1st-half and 2nd-half mean feature corrs = 0.57 (moderate drift). Signs of the top
  20 are stable across halves (all 20 same sign), magnitudes wander: e.g. Takerkoust 0.024 -> 0.104, Tagoudiche 0.013 ->
  0.099, Midelt 0.092 -> 0.055, Zerrouka 0.094 -> 0.058. By time quintile, Agouti: .044/.105/.022/.125/.099;
  Azzaden: -.085/-.078/-.056/-.066/-.099; Midelt: .098/.095/.062/.094/.019 (decaying); Amazigh: -.017/-.090/-.041/-.065/-.111 (growing).
- OOS check of feature selection: picking top-K by train-period corr and sign-summing on the held-out quarter gives
  K=1: 0.055, K=5: 0.106, K=10: 0.111, K=20: 0.120, K=40: 0.111; all-178 weighted by train sharpe: 0.115 (sharpe 0.51).

## 6. Feature redundancy (global corr of era-ranked features, NaN->0.5)

- Pairs |corr|>0.95: 0; >0.9: 0; >0.8: 4; >0.5: 74 of 15,753 pairs. Max |corr| 0.853. No feature has a >0.9 partner.
- PCA: top 5/10/20/40 components explain 16.4/24.5/36.5/52.8% of variance -> features are genuinely diverse, not clones.
- Among the top-20 target features, mean pairwise |corr| 0.18 (max 0.85).
- Feature-fraction/colsample around 0.2-0.3 is appropriate; no need for dedup.

## 7. Benchmark v1_sherpa (train split)

- Columns: exped, v1_sherpa; index id; 100% join coverage to train, exped matches row-for-row. Values in [0.2065, 0.80],
  mean 0.504, std 0.173, 4,307 unique values; per-era mean 0.504 / std 0.174 (not rank-normalized; looks like a shrunk score).
- Per-era Spearman with target_everest over ALL train: mean 0.2553, std 0.2057, sharpe 1.24, positive in 88.4% of eras.
  By time quintile: 0.204, 0.251, 0.196, 0.245, 0.381. Last 500 eras: 0.4385 (std 0.164). Last 25% test block: 0.354,
  sharpe 1.74, positive 93.7%, 20-era rolling mean negative in only 2.8% of windows.
- Against other targets: Ighil 0.285, Meltsene 0.282, everest 0.255, Tarhat 0.248, Timesguida 0.247, Tougroute 0.220,
  Igli 0.154, Azurki 0.153, Ghat 0.138, Bougafer 0.124, Anghomar 0.107, Gourza 0.078, Ayachi 0.069, Saghro 0.067, Tiskiouine 0.042.
- Benchmark vs features (per-era Spearman mean): Tazzarine +0.470, Agouti +0.463, Azzaden -0.418, Ighzer -0.379,
  Tafouk +0.379, Amazigh -0.375, Takerkoust +0.375, Ouarzazate -0.367, Mrirt +0.350, Azaghar -0.340, Zerrouka +0.318,
  Tagoudiche +0.312, Nkob -0.310, Tagmat -0.309, Tazarine +0.289. The benchmark loads on exactly the same features our
  correlation screen finds, with the same signs. Linear fit of era-ranked benchmark on 178 era-ranked features: R2 = 0.66.
- Interpretation: a model trained on these features WILL be highly correlated with the benchmark (our LGBM: 0.46-0.63
  per-era corr with v1_sherpa; a sign-sum of top-20 features: 0.62). UNQ will be hard-won; it requires signal the
  benchmark lacks (non-linear interactions, instrument-tracking/lag features, alternative targets, or deliberate
  orthogonalization).
- WARNING (in-sample suspicion): 0.35-0.44 per-era Spearman is far above anything fit honestly here (max ~0.12), its
  strength rises over time, and a model trained to MIMIC the benchmark (LGBM with v1_sherpa as target) scores 0.161 on
  our held-out block — better than fitting the real target (0.118) — which is the signature of label leakage through an
  in-sample teacher. Treat v1_sherpa-on-train as contaminated for anything other than a rough "what does the benchmark
  look at" study.

## 8. Realism baseline (train = eras 1..4891, 20-era embargo, test = eras 4914..6524, 1,611 eras / 143,504 rows)

Metric: per-era Spearman(pred, target_everest), mean / std / sharpe / fraction of eras > 0.

| model | corr | std | sharpe | pos | corr w/ bench | UNQ proxy* | 20-era rolling <0 |
|---|---|---|---|---|---|---|---|
| LGBM as specified (300t, lr .05, nl31, ff .3, min_leaf 1000) | 0.0821 | 0.188 | 0.436 | 0.677 | 0.456 | -0.078 | 28.5% |
|   same at 50 / 100 / 200 trees | 0.1164 / 0.1052 / 0.0900 | | 0.577 / 0.538 / 0.479 | | | | |
|   in-sample (train rows) | 0.5432 | 0.152 | 3.56 | 0.996 | | | |
| LGBM shallow (nl7, lr .02, ff .2, min_leaf 3000, l2 10), 300t | 0.1175 | 0.206 | 0.571 | 0.717 | 0.607 | -0.108 | 24.9% |
|   same at 100 / 200 / 1000 trees | 0.1210 / 0.1208 / 0.0982 | | 0.568 / 0.576 / 0.508 | | | | |
|   shallow on era-ranked inputs | 0.1157 | | 0.571 | | 0.603 | | 23.6% |
|   shallow, last 1500 train eras only | 0.1180 | | 0.585 | | 0.595 | -0.095 | 23.7% |
|   shallow, last 3000 train eras only | 0.1159 | | 0.571 | | 0.605 | | |
|   shallow, target_Tougroute as label | 0.0900 | | 0.475 | | 0.493 | | |
|   shallow, 5-seed average | 0.1180 | | 0.575 | | | | |
| ridge on era-ranked feats, lambda 1e2 / 1e4 / 1e5 | 0.070 / 0.091 / 0.116 | | 0.36 / 0.45 / 0.52 | | 0.41-0.62 | | |
| sign-sum of top-20 features (chosen on train) | 0.1203 | 0.217 | 0.554 | 0.735 | 0.617 | -0.126 | 25.7% |
| benchmark v1_sherpa (likely in-sample) | 0.3536 | 0.203 | 1.738 | 0.937 | 1.0 | 0 | 2.8% |
| 0.5 gauss(shallow LGBM) + 0.5 gauss(bench) (leaky) | 0.2603 | 0.199 | 1.306 | 0.886 | 0.888 | -0.115 | 5.5% |
| LGBM fit to BENCHMARK as target (leaky) | 0.1610 | 0.217 | 0.741 | 0.764 | 0.838 | -0.212 | 17.0% |

*UNQ proxy = per-era Spearman of (rank-gaussianized preds residualized per era against rank-gaussianized benchmark) with
target. Negative for every honest model because the in-sample benchmark "explains" more target than it will live; do not
read it as live UNQ. Live UNQ uses a stake-weighted crowd/consensus, not v1_sherpa-on-train.

- Shallow-LGBM test corr by 4 consecutive ~400-era blocks: 0.148, 0.102, 0.108, 0.112 (sharpe 0.67, 0.51, 0.57, 0.53);
  benchmark on the same blocks: 0.221, 0.390, 0.388, 0.415.
- FIT as the platform defines it (covariance of rank-gaussianized preds with centred target) for the shallow LGBM:
  mean 0.0255/era, std 0.0442, sharpe 0.58; FIT/Spearman ratio = 0.217 (~ target std 0.227). So a 0.12 Spearman model
  is a ~0.026 FIT per era; a 20-day window averages the per-era noise.
- Crude INOV proxy (orthogonalize preds vs equal-weight avg of OUR top-10 features, stand-in for the unknown core set):
  the top-10 average itself scores 0.112 (sharpe 0.52); our preds correlate 0.78 with it; residual retains 0.048 corr
  (sharpe 0.31). I.e. a plain tree model is ~80% "core-feature average"; INOV will also need deliberate orthogonal signal.
- Feature importance (gain share, 300-tree base): Amane 1.75%, Zerrouka 1.42, Agouti 1.35, Tagmat 1.34, Tafoughalt 1.32,
  Takerkoust 1.28, Akchour 1.27, Iferd 1.27, Tarhzirt 1.25, Tinghir 1.22; top-10/20/50 = 13.5/24.1/47.8% of gain; no
  feature has zero gain. Importance is flat -> the model spreads over many weak features (overfitting risk).
- Prediction scale (regression on 0..1 target): mean 0.4997, std 0.044, range [0.256, 0.673]. Must be rank-normalized
  to [0,1] per era before submission anyway.

## 9. Validation & live

- validation: 125,931 rows, 1,392 eras `exped_6566`..`exped_7955`/`7957` (gap of 42 eras after the last train era 6524),
  89-92 instruments/era (mean 90.5), feature columns identical to train in name and order (178), feature value set
  {-1,0..9}, missing 0.72%, all 15 targets 100% NaN, ids unique 16-hex (e.g. `1250fe6ab2501ae4`, `4fbab755bfcf465f`,
  `4d6304cc98b81a8c`), zero id overlap with train.
- live (downloaded while round open): 90 rows, 1 era `exped_8055` (98 eras after validation ends, 1,531 after train ends),
  identical 178 feature columns, missing 3.06%, targets NaN, ids e.g. `0e70e422087ffa87`, `24415f9c1a376b9b`,
  `bf7fadcc2582fd1f`. Submission must map these 90 index ids -> prediction in [0,1].
- Instrument tracking across eras works in validation too: mutual-best-match 1.000 at eras 6567, 7266, 7957 (Hamming 34-70).

## 10. Instrument persistence (hidden-id re-identification)

For each row in era t, nearest row in era t-1 by Hamming distance over the 178 binned features:
mean min-Hamming 10.5 (era 53), 29 (1003), 39 (3003), 35 (5003), 21 (6003), 45 (6524) out of 178, vs ~140 for a random
era 500 apart; best matches are one-to-one and mutual in 98.9-100% of rows at lag 1 and lag 2 (Hamming 36), still 98.9%
mutual at lag 5 (Hamming 66). Universe sizes are nearly constant between adjacent eras (e.g. 91 -> 92). This is a usable
"instrument chain" for lagged/delta features and for per-instrument target autocorrelation studies, with a
confidence-threshold needed for lags > 2.

## 11. Compute / memory estimates

Measured on sandbox (2 threads, lightgbm 4.7): 300 trees nl31 on 323k x 178 = 15-17 s; 300 trees nl7 = 11 s; 1000 trees
nl7 = 28 s; 5x300-tree shallow fits = ~55 s. Data: train float32 333 MB; int8 83 MB; validation 90 MB float32; a full
era-ranked copy adds 333 MB. Peak RSS of the heaviest script here was < 3 GB.

Laptop (Core Ultra 7, ~16 threads, 32 GB): expect LightGBM ~4-6x faster than the sandbox: ~3-4 s per 300-tree model on
full train (468k rows), ~10-15 s per 1000-tree model. A 5-fold purged CV x 5 seeds x 1000 trees sweep over 10 configs is
~20-40 min; a 50-model final ensemble ~10 min. RAM needs < 4 GB throughout. GPU: no benefit for LightGBM/XGBoost at this
size (CPU hist is faster at 468k rows); only relevant for neural nets, which cannot ship as torch in the .pkl anyway
(would need weights exported to numpy). A "multi-hour" run therefore only makes sense as a broad hyper-parameter /
feature-engineering / instrument-chain search with repeated purged CV, not as a single long fit.

## 12. Design implications & open questions for the planner

- Validation protocol: era-grouped, purged, embargo >= 20 eras (target ACF hits ~0 at lag 20), walk-forward or
  grouped-k-fold over eras, metric = per-era Spearman mean & sharpe plus 20-era rolling windows (the payout unit).
  Expect honest numbers around 0.10-0.13 corr, sharpe 0.5-0.6; anything much higher is leakage.
- Model: heavily regularized GBMs (few leaves, large min_data_in_leaf, feature_fraction 0.2-0.3, <= 200-300 trees at
  lr 0.02-0.05, or more trees at lower lr with early stopping on purged CV); linear/ridge on era-ranked features is a
  strong, nearly-free ensemble member (0.116). Seed bagging alone is useless; diversity must come from different
  feature subsets / targets / lag features / model families.
- Era weighting: missingness and universe regime change over time; recent-only (last 1500 eras) matched full-history
  OOS, so time-decay weighting or dropping eras < ~2800 is a legitimate knob to test.
- UNQ/INOV: every feature-driven model is ~0.6 correlated with the benchmark and ~0.8 with a top-feature average. Need
  explicit mechanisms: instrument-chain lag/delta features (not in the 178 cross-sectional bins), feature-neutralization
  against the top-K feature axis, or training on target residuals vs a feature-average proxy. The exact core-feature
  set for INOV is not exposed by the schema.
- Missing handling: -1 -> NaN for GBMs (LightGBM native); -> 0.5 (median rank) for linear models. Keep int8 on disk.
- OPEN: is v1_sherpa-on-train in-sample? (Strongly suspected.) Only server-side validation diagnostics can answer; that
  is a submission-type action and is out of scope for this scout.
- OPEN: era-to-calendar mapping (is one exped one trading day? then train ends ~6 years before live).
- OPEN: how FIT is pooled over the 20-day scored set (per-era then averaged vs pooled covariance) — affects whether
  per-era rank-gaussianization is sufficient.
