# Prior art: what wins in Numerai-style era-based cross-sectional ranking tournaments

Scope: evidence gathered 2026-11-13 from Numerai docs/forum/example code, numerai-tools source, Kaggle Jane Street, library release notes. Mapped onto the Everesteer "Himalayas" futures tournament (dataset `atlas`: 178 features in 10 uniform bins 0..9, -1 = missing; 15 targets, primary `target_everest` with 5 levels and 5/20/50/20/5 bell mass; daily expeds; ~20-exped gap between train and validation matching the 20-day horizon; tens to low hundreds of instruments per exped).

Evidence grades: **A** = official docs/code or reproducible notebook; **B** = well-argued forum post with numbers; **C** = anecdote / theory.

---

## 0. The score formula and what it implies (read first)

Live formula (eiq_explain_scoring, read 2026-11-13):
`payout = stake * score`, `score = b*arctan((FIT+UNQ+INOV)/b)`, `b = 2/pi = 0.63662`, weights 1/1/1.

- **FIT** = covariance of rank-gaussianized predictions with the centred realized target.
- **UNQ** = same covariance after the (rank-gaussianized) benchmark model's direction is removed from your predictions; **set to 0 on rounds where the benchmark's own covariance with the target is negative**.
- **INOV** = same as UNQ but the reference is the equal-weight average of 10 frozen "core" features; 0 when that average's own covariance is negative.

This is Numerai's CORR + MMC + a "feature-neutral" term (numerai-tools `correlation_contribution`, [scoring.py](https://raw.githubusercontent.com/numerai/numerai-tools/master/numerai_tools/scoring.py)), except (a) no power-1.5 tail emphasis, (b) covariance not correlation, (c) all three terms summed with weight 1.

**Algebra.** Let p = gaussianized prediction (unit variance), b = gaussianized benchmark, c = standardized core-mean, t = centred target. With rho_b = corr(p,b), rho_c = corr(p,c), B = cov(b,t), C = cov(c,t):

```
FIT  = cov(p,t)
UNQ  = cov(p - rho_b*b, t) = FIT - rho_b*B        (if B > 0, else 0)
INOV = cov(p - rho_c*c, t) = FIT - rho_c*C        (if C > 0, else 0)
sum  = 3*FIT - rho_b*B - rho_c*C                  (both references positive)
```
Decompose p = rho_b*b + s*o with s = sqrt(1-rho_b^2), o orthogonal to b, O = cov(o,t):
`FIT + UNQ = rho_b*B + 2*s*O`. **Signal orthogonal to the benchmark is paid twice (three times if also orthogonal to the core-mean); benchmark-aligned signal is paid once.** Consequences:

1. Partially neutralizing predictions toward the benchmark and toward the core-feature mean is not a pure FIT cost here: it is directly rewarded. Optimal proportion is a tunable with a real objective (maximize mean per-era `3*FIT - rho_b*B - rho_c*C` on the honest validation folds), not a hunch.
2. The pkl contract allows `predict(live_features, live_benchmark_models)` (context brief), so the live benchmark series is available at inference time: neutralization to the benchmark can be done inside `predict`. Core-mean neutralization needs the 10 core feature names; `get_dataset_schema` only reports `core_feature_overlap: all = 10`, so the design must check whether the names are discoverable (e.g. `eiq_train` metrics expose `inov_feature_coverage`) and otherwise approximate.
3. The zero-gating is asymmetric: being correlated with the benchmark is never penalized on rounds the benchmark loses. Being anti-correlated with it is paid only through FIT. So "benchmark + orthogonal alpha" dominates "anti-benchmark".
4. **The arctan is irrelevant in practice.** Target std is sqrt(0.1*0.25 + 0.4*0.0625) = 0.224, so FIT = corr * 0.224; corr 0.05 gives FIT ~ 0.011, and even the triple sum is ~0.03-0.07. `b*arctan(x/b)` is linear to <1% below |x| ~ 0.1 (3% compression at 0.2, 15% at 0.5, saturates at 1.0). Treat the objective as the plain sum; variance matters only through ranking stability and stake risk, not through the bound. (Assumption: the server does not rescale the target; if it gaussianizes it too the point still holds, x stays ~ corr.)

---

## 1. Model families and settings that work on binned tabular era data

### 1.1 Gradient-boosted trees (the backbone; grade A)
Numerai's official recipes are consistently "many slow shallow trees, 10% column subsampling":

| Source | n_estimators | lr | depth / leaves | colsample | min_data_in_leaf |
|---|---|---|---|---|---|
| hello_numerai.ipynb (v6, demo) | 300 | 0.05 | 5 / 31 | 0.1 | default |
| example_model.ipynb "standard" | 2000 | 0.01 | 5 / 31 | 0.1 | default |
| example_model.ipynb "deeper" | 30000 | 0.001 | 10 / 1024 | 0.1 | 10000 |
| forum "super massive LGBM grid search" (6463) | 20k-60k | 0.001-0.01 | 5-7 / 2^d-1 | 0.1 | n/a |

Sources: [example_model.ipynb](https://raw.githubusercontent.com/numerai/example-scripts/master/numerai/example_model.ipynb), [hello_numerai.ipynb](https://raw.githubusercontent.com/numerai/example-scripts/master/numerai/hello_numerai.ipynb), [grid search thread](https://forum.numer.ai/t/super-massive-lgbm-grid-search/6463).

Findings from the grid search: going from 20k to 30k-60k trees bought ~+0.001 corr for 6 h of compute; "20k trees at lr 0.01 is very competitive in Sharpe space". `min_data_in_leaf`, L2 and `subsample` were not explored by Numerai. Small-budget thread ([6693](https://forum.numer.ai/t/hyperparameters-optimization-for-small-lgbm-models/6693)): 1885 trees, lr 0.019, depth 5, colsample 0.13 reached top-5%; CatBoost best at 6685 iters, lr 0.027; "you can choose whatever GB method and you will not gain any significant edge" (grade B). Signal Miner's search space ([7922](https://forum.numer.ai/t/signal-miner-find-unique-alpha-beat-the-benchmark/7922)) spans colsample 0.001-1, reg_lambda 0-100k, lr 1e-5-0.3, depth 5-16, leaves 4-2048, min_child_samples 1-15k, up to 30k trees.

**Scaling to `atlas`.** Numerai's rows ~ 5000 stocks x ~1000 eras (millions). `atlas` is ~50-200 instruments x daily eras, likely 0.1-1 M rows total but only ~100 per cross-section. Scale leaf sizes to rows, not copy Numerai's absolute numbers. Recommended starting ranges (to be tuned on honest CV, section 3):

- **LightGBM** (CPU is fine at this size): `objective=regression` (L2 on the 0..1 target; also try `huber`), `num_leaves` 15-63, `max_depth` 4-6, `learning_rate` 0.005-0.02, `n_estimators` 2000-10000 with no early stopping on the test fold (pick by CV), `colsample_bytree` 0.1-0.3 (178 features -> 18-53 per tree), `min_data_in_leaf` 0.1-0.5% of training rows (200-2000 on ~400k rows), `lambda_l2` 1-100, `bagging_fraction` 0.5-0.8 with `bagging_freq=1`, `max_bin=16` (features are already 10-bin integers, so this is lossless and faster), `min_sum_hessian_in_leaf` default. Treat -1 as NaN (`use_missing=True`, default) or keep -1 as its own bin; test both.
- **XGBoost** (GPU, `tree_method=hist, device=cuda`): `max_depth` 4-6 (or `grow_policy=lossguide, max_leaves` 15-63), `eta` 0.005-0.02, 2000-10000 rounds, `colsample_bytree` 0.1-0.3 (also try `colsample_bynode`), `min_child_weight` 100-1000 (hessian = row count for squared error), `subsample` 0.5-0.8, `lambda` 1-100, `max_bin` 16.
- **CatBoost**: `depth` 4-6, `learning_rate` 0.02-0.05, `iterations` 3000-10000, `l2_leaf_reg` 3-30, `rsm` 0.1-0.3 (verify GPU support of rsm for RMSE in your version), `border_count` 16, `bootstrap_type=Bernoulli, subsample 0.5-0.8`, `loss_function=RMSE` (MultiRMSE available for multi-target).
- Use `RMSE/L2` regression on the 5-level target, not classification; Numerai's bell-binned target is deliberately regularized ([hello_numerai](https://raw.githubusercontent.com/numerai/example-scripts/master/numerai/hello_numerai.ipynb)).

### 1.2 Learning-to-rank objectives (grade B, mixed)
[Learning to Rank thread (454)](https://forum.numer.ai/t/learning-to-rank/454): XGBRanker `rank:pairwise` grouped by era was roughly at parity with regression after tuning; LGBMRanker lagged badly; pairwise costs much more compute. Most robust finding there: **neutralizing the target before training** was the single most important hyperparameter for the ranking models. Recommendation: regression first; LTR as a diversity member only.

### 1.3 Neural nets: supervised autoencoder + multitask MLP (grade B)
Jane Street 1st place (Yirun) re-implemented for Numerai in [forum 4338](https://forum.numer.ai/t/autoencoder-and-multitask-mlp-on-new-dataset-from-kaggle-jane-street/4338): Input -> BatchNorm -> GaussianNoise(0.035) -> Dense(96) -> BN -> Swish; decoder reconstructs inputs; AE head Dense(96)->BN->Swish->Dropout(0.4)->Dense(6 targets); MLP on concat(features, encoding): BN -> Dropout(0.1) -> Dense(896,448,448,256) with BN/Swish/progressive dropout -> Dense(6 targets); three MSE losses, Adam 1e-4; early stop on MLP loss only; average the 6 target heads. Caveats recorded: the AE must be fit inside each CV fold (leakage otherwise), several people could not reproduce gains, and DAE-only features underperformed. Value in our setting: diversity member orthogonal to GBMs (the "vibesciencing" thread's MLP ranking model was explicitly the orthogonal component: [8214](https://forum.numer.ai/t/vibesciencing-my-way-through-v5-2-data-faith-ii/8214)). Shipping constraint: **no torch in the pkl** -> train in torch, export weights to numpy and implement forward pass in numpy inside `predict`.

### 1.4 Tabular foundation models (grade B, negative)
TabPFN v2.5 / TabICL v2 on Numerai ([8269](https://forum.numer.ai/t/foundation-models-on-numerai-data-tabpfn-v2-5-and-tabicl-v2/8269)): NC 0.0010-0.0039 vs a standard tree model at ~0.0170; distillation abandoned. Skip.

### 1.5 GPU stack on an RTX 5070 Ti (Blackwell, sm_120) as of Nov 2026
- **PyTorch**: stable 2.15.0 (2026-10-28) ships default CUDA 13.2 wheels and cu134; CUDA 12.6/13.0 wheels dropped (stay on 2.14 for those). Blackwell has been supported since 2.7 with cu128 wheels (`pip install torch --index-url https://download.pytorch.org/whl/cu128`); the index still carries cu126/cu128/cu129/cu130/cu132/cu134. Minimum Python for 2.15 binaries is 3.11. Sources: [v2.15.0 notes](https://github.com/pytorch/pytorch/releases/tag/v2.15.0), [PyTorch 2.7 blog](https://pytorch.org/blog/pytorch-2-7/), [wheel index](https://download.pytorch.org/whl/).
- **XGBoost**: 3.4.1 (2026-08-15). Default PyPI wheels (Linux and Windows) build against CUDA 13.3; `xgboost-cu12` targets CUDA 12.9; GPU docs state "CUDA >= 12.9 is required"; 3.4.0 fixed an SM120 (Blackwell) SHAP issue, implying sm_120 is a supported target. **3.4.x requires Python >= 3.12** (PyPI `requires_python`). The platform runs Python 3.11, so any xgboost Booster inside the pkl must be from a release that still supports 3.11 (3.3.x, Jun 2026, or earlier) and the sandbox must have xgboost installed -- verify, or ship xgboost only via a numpy re-implementation / exclude from the pkl. Sources: [install](https://xgboost.readthedocs.io/en/stable/install.html), [GPU](https://xgboost.readthedocs.io/en/stable/gpu/index.html), [3.4.0 notes](https://xgboost.readthedocs.io/en/latest/changes/v3.4.0.html), [3.1.0 notes](https://xgboost.readthedocs.io/en/latest/changes/v3.1.0.html), [PyPI](https://pypi.org/project/xgboost/), [PyPI cu12](https://pypi.org/project/xgboost-cu12/).
- **LightGBM**: 4.7.0 (2026-07-18). PyPI wheels are CPU-only. GPU needs a source build: OpenCL (`pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_GPU=ON`, Windows+Linux, needs OpenCL+Boost) or CUDA (`...USE_CUDA=ON`, **Linux only**, Blackwell supported since 4.6.0, set `CMAKE_CUDA_ARCHITECTURES` e.g. `'100;120'`). OpenCL build only accelerates histograms and needs `max_bin<=63`. At `atlas` size CPU LightGBM is fast enough; do not spend handoff time on a GPU build. Sources: [PyPI](https://pypi.org/project/lightgbm/), [install guide](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html), [v4.6.0](https://github.com/microsoft/LightGBM/releases/tag/v4.6.0), [GPU tutorial](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html).
- **CatBoost**: 1.2.10 (2026-02-18), wheels for CPython 3.8-3.14, Windows and Linux GPU builds. Explicit sm_100/sm_120 architecture flags were only merged in [PR 3108](https://github.com/catboost/catboost/pull/3108) (June 2026) and are **not in any release**; [issue 2967](https://github.com/catboost/catboost/issues/2967) reports CUDA error 800 crashes on a Blackwell (sm_121) GPU with 1.2.8 and no maintainer response. GPU training is also non-deterministic. Plan: smoke-test `task_type='GPU'` on a tiny fit in the first laptop handoff and fall back to CPU (fast at this size). Sources: [releases](https://github.com/catboost/catboost/releases), [PyPI](https://pypi.org/project/catboost/), [GPU training](https://catboost.ai/docs/en/features/training-on-gpu).
- Practical: with ~100 rows/era the whole dataset fits in RAM; GPU mostly helps XGBoost hyperparameter sweeps and the torch MLP. CPU core count matters most for LightGBM/CatBoost. Provide both Windows (PowerShell) and Linux commands; prefer `uv`/venv with pinned versions, Python 3.11 locally too so pickles round-trip.

---

## 2. Multi-target training and blending (grade A/B)

- Numerai ships ~20 auxiliary targets with 0.3-0.9 correlation to the primary and officially recommends "training models to be good at multiple targets and/or ensembling models trained on different targets", warning about selection bias: cross-validate target choice rather than pick by validation score ([Super Massive Data Release deep dive](https://forum.numer.ai/t/super-massive-data-release-deep-dive/4053); [v6 data docs](https://docs.numer.ai/numerai-tournament/data): "models trained on other targets can outperform or ensemble well").
- Which target to train on: over 9 years the primary (Cyrus) model won on Sharpe/corr, but over the last year auxiliary-target models matched it; Numerai's own conclusion: "almost certainly better to build multiple models on multiple targets and ensemble them" ([6331](https://forum.numer.ai/t/v4-vs-v4-1-nomi-vs-ralph-vs-cyrus/6331)).
- Vector-leaf multi-output trees (XGBoost `multi_strategy="multi_output_tree"`, production-ready in 3.4): +0.0003 corr for a two-target model, helps weak targets, marginal/negative on the primary -- "different models, not better ones" ([8355](https://forum.numer.ai/t/xgboost-vector-leaf-output-two-targets-experiment/8355)). Multi-task MLP heads (6 targets averaged) are the NN analogue (section 1.3).
- Recipe for `atlas` (15 targets): (1) compute per-exped corr of each auxiliary target with `target_everest` and its own predictability; (2) train one GBM per target with identical features/params; (3) score every member on `target_everest` per exped; (4) ensemble by per-exped rank-average with weights tilted toward the primary; (5) select members by CV, not by the final fold. Diversity across targets has historically mattered more for MMC-type metrics than seeds or hyperparameters ([8214](https://forum.numer.ai/t/vibesciencing-my-way-through-v5-2-data-faith-ii/8214): "you win correlation contribution by producing correct predictions, without building another boosted-tree regressor that converges to the same exposures").

---

## 3. Era-aware validation (grade A)

- Rows within an era are not independent and consecutive eras share overlapping target windows: Numerai tells you to "treat each era as a single data point" and that overlapping targets require careful CV ([data docs](https://docs.numer.ai/numerai-tournament/data)); for a 20-day target, sample every 4th weekly era or purge overlapping eras from folds ([4053](https://forum.numer.ai/t/super-massive-data-release-deep-dive/4053)); hello_numerai uses a 60-business-day embargo equal to the target horizon ([hello_numerai](https://raw.githubusercontent.com/numerai/example-scripts/master/numerai/hello_numerai.ipynb)). Random KFold leaks because rows from era t sit in train while era t+1 (same return window) sits in test; GroupKFold without a gap leaks for the same reason.
- For `atlas` expeds are daily and the horizon is 20 trading days, so **adjacent eras overlap 19/20**: use grouped, chronological folds with a purge of >= 20 expeds on both sides of each test block plus an embargo (>= 20 after the block). The platform itself leaves a ~20-exped gap between train and validation (schema notes). Expanding-window walk-forward was preferred by the community because the final training set approaches the full data ([791](https://forum.numer.ai/t/era-wise-time-series-cross-validation/791)); the platform's hosted trainer uses an "exped_purged" scheme with embargo defaulting to the label horizon (eiq_train docs) -- mirror it locally.
- Metrics: compute the platform metric per exped (covariance of rank-gaussianized preds with centred target, i.e. `numerai_tools.tie_kept_rank -> gaussian -> cov`), then mean, std, mean/std ("Sharpe"), max drawdown of the cumulative series, and the full `3*FIT - rho_b*B - rho_c*C` proxy (section 0). Always score each era independently ("per date metrics are what matter", hello_numerai). Because of the 20-day overlap the effective number of independent eras is roughly N_eras/20: use block bootstrap with ~20-exped blocks for confidence intervals, and compare models by paired per-era differences, not by headline means. Reference for purging/embargo: Lopez de Prado, *Advances in Financial ML* ch. 7; sklearn `TimeSeriesSplit(gap=...)`.
- Recency weighting: thin evidence. One thread reports sample weights that favour eras "similar" to live sometimes weight older eras higher ([4753](https://forum.numer.ai/t/what-do-the-top-models-do-that-lesser-models-dont/4753), grade C). Treat exponential time-decay weights as a CV-tested option, not a default; the platform exposes era ids (`eiq.exped_train`) for exactly this.
- Era boosting (retrain extra trees on the worst 50% of eras, 10 trees/iter, ~20 iters): in-sample Sharpe exploded (2.3 -> 22) while out-of-sample stayed ~0.025 corr -- strong overfit signal ([189](https://forum.numer.ai/t/era-boosted-models/189)). Avoid, or only as a CV-validated member.
- Feature stability screens: dropping 10 features whose target correlation flipped sign over time raised validation corr 0.023 -> 0.025 ([5627](https://forum.numer.ai/t/removing-dangerous-features/5627)). Compute per-era feature-target corr on train folds; drop or down-weight features with unstable sign; keep decisions inside the CV loop.

---

## 4. Neutralization and exposure control, mapped to UNQ / INOV (grade A/B)

- Mechanics (numerai-tools): per era, rank -> gaussianize -> `scores -= proportion * X @ pinv(X) @ scores` (with intercept) -> re-standardize ([scoring.py](https://raw.githubusercontent.com/numerai/numerai-tools/master/numerai_tools/scoring.py), [FNC docs](https://docs.numer.ai/numerai-tournament/scoring/feature-neutral-correlation)). Full (p=1) kills linear feature exposure but keeps interactions.
- Evidence on proportion: bootstrap bias-variance study finds variance falls and bias rises with p, sweet spot ~0.5, lower for low-variance models (RF), higher for high-variance ones (NNs) ([7486](https://forum.numer.ai/t/feature-neutralization-increases-bias-and-reduces-variance/7486)); the feature-exposure post shows lower max exposure correlates with better live results across 80+ models and recommends p in (0,1) tuned, top-k riskiest features, per era ([899](https://forum.numer.ai/t/model-diagnostics-feature-exposure/899)); counter-evidence: plain pinv neutralization optimized to p=0 for one user, fixed by an L1+L2-regularized exposure fit ([1015](https://forum.numer.ai/t/how-to-safely-perform-feature-neutralization/1015)), and others found near-zero p best ([4595](https://forum.numer.ai/t/way-too-early-comparison-of-legacy-new-models/4595)). Net: use **ridge-regularized, per-era, partial** neutralization, tune p in {0, 0.25, 0.5, 0.75, 1} by CV.
- Direct mapping for Everesteer:
  - **UNQ**: neutralize predictions to the benchmark direction inside `predict(live_features, live_benchmark_models)` with proportion p_b. Section 0 shows the payoff: `rho_b*B + 2*s*O`, so the tuned p_b will typically be > 0 if your residual carries signal. Also train "residual" members on a target residualized to the benchmark (the hosted trainer offers `transforms.target.type = residualize_to_benchmark`; the forum "residual benchmark models" produced the MMC signal in [8214](https://forum.numer.ai/t/vibesciencing-my-way-through-v5-2-data-faith-ii/8214)).
  - **INOV**: neutralize to the equal-weight mean of the 10 core features (if names are discoverable) or, failing that, to a ridge fit on all 178 features with small p; FNC-style validation ([FNC](https://docs.numer.ai/numerai-tournament/scoring/feature-neutral-correlation)) is the local proxy. Training members on fewer/other features (the forum's low-CWMM models) and on auxiliary targets lowers rho_c without a custom loss.
  - Training-time alternative: differentiable neutralized-corr loss with torchsort soft ranks and exposure penalties ([5132](https://forum.numer.ai/t/optimizing-for-fnc-and-tb-scores/5132)); per-era batching required; TB200-type tail metrics overfit (irrelevant here, no power 1.5).
- Warning from the meta-model correlation study: validation corr and correlation with the consensus can move inversely; any neutralization reduced both ([6825](https://forum.numer.ai/t/initial-mm-correlation-tests-with-rain-dataset/6825)). Optimize the full proxy sum, not FIT alone.
- Max feature exposure / exposure RMS are cheap diagnostics; keep |max exposure| under ~0.2-0.3 as a guardrail (numbers from Numerai's 5000-stock universe; expect larger sampling noise at N~100).

---

## 5. Overfitting control, ensembles, rank-normalization (grade A/B)

- Rank per era with ties kept: `(rank(method='average') - 0.5)/n` -> [0,1] (numerai-tools `tie_kept_rank`); the platform gaussianizes ranks before scoring so prediction scale/distribution is irrelevant ("you are only evaluated on your prediction's ranks", [CORR docs](https://docs.numer.ai/numerai-tournament/scoring/correlation-corr)). Rank-average members per era, then re-rank, then clip to [0,1] (the gate rejects anything outside; no clipping server-side). For live, the frame is one exped, so rank within it.
- Ensemble composition: official target ensembles rank-average per-target models; Jane Street winners averaged seeds x folds; community evidence says more GBMs with the same features/targets add little, diversity by target > by feature subset > by model family > by seed. Expect diminishing returns past ~5-10 members; weight by CV with a sparsity prior (greedy forward selection as in [8214](https://forum.numer.ai/t/vibesciencing-my-way-through-v5-2-data-faith-ii/8214)). Orthogonalizing features via PCA did not de-correlate era-wise performance (still ~0.5) ([7459](https://forum.numer.ai/t/orthogonal-model-performance-across-eras/7459)) -- do not expect large Sharpe gains from "orthogonal" members.
- Regularization knobs with evidence: low colsample (0.1), slow learning, large leaves, L2; multi-target averaging; partial neutralization; drop unstable features. Avoid: early stopping on an un-embargoed fold, era boosting, picking targets/neutralization proportions on the final fold.
- Selection discipline: fix the search space, run the purged CV once per config, select on mean per-era proxy minus a penalty on std, then refit on all train and predict validation for the platform's out-of-sample diagnostics.

---

## 6. Small cross-sections (tens to low hundreds of instruments per exped)

Little published work; Numerai Crypto (~50-100 tokens/day, 5-bin gaussianized 30-day target, [data](https://docs.numer.ai/numerai-crypto/data)) is the closest public analogue and its example notebook is a plain LightGBM with a 70/30 chronological split ([crypto example](https://raw.githubusercontent.com/numerai/example-scripts/master/crypto/example_model.ipynb)). What changes, from first principles plus Numerai experience:

- Per-era Spearman noise is ~1/sqrt(N-1): N=50 -> 0.14, N=100 -> 0.10, vs 0.014 at N=5000. A model with true per-era corr 0.03 will post negative eras ~40% of the time. Validation needs hundreds of expeds and block-bootstrap CIs; small differences between configs are undetectable, so prefer simple, strongly regularized choices over fine tuning.
- Ties and ranks: with 5-level targets and ~100 rows, tied targets are the norm; use tie-kept average ranks for both predictions and diagnostics, and rank within exped, never globally.
- Capacity: min_data_in_leaf must be measured against total rows across eras (hundreds of thousands), but each leaf should still span many eras; prefer shallow trees (depth 4-6), heavy bagging, and more members over deeper trees.
- Cross-sectional de-meaning is already done by binning; do not add per-era standardization of features (they are bins). Consider era-level context features (e.g. per-exped mean of a feature) only inside CV -- they are cheap to overfit with 100 rows.
- Benchmark correlation: with few instruments, rho_b estimated per era is itself noisy; shrink the neutralization coefficient (ridge) and estimate it pooled over trailing eras rather than per live frame when possible.
- Missing values: -1 is MISSING; with small cross-sections an instrument with many missing features can dominate tail ranks. Treat -1 as NaN for GBMs (native missing handling) and cap predictions for rows with high missingness toward 0.5 rank as a tested option.

---

## 7. Bounded arctan payout: mean vs variance

- Numerai's analogue is a per-round clip (now +200%/-100% of stake, previously +-1) with multipliers on CORR and MMC ([staking docs](https://docs.numer.ai/numerai-tournament/staking)); Everesteer uses `b*arctan(x/b)`, odd and concave for x > 0, convex for x < 0, saturating at +-1.
- Because realistic |x| is ~0.01-0.07 (section 0), the transform is numerically linear: expected payout is proportional to mean(FIT+UNQ+INOV); Jensen-type variance penalties are below 1%. Variance still matters for (a) the leaderboard/track record and (b) stake sizing once staking starts, so report Sharpe and drawdown, but **do not sacrifice mean for variance to game the bound**. The actionable lever is the triple-counting of orthogonal signal, not the arctan.
- Payout over 20 trading days per round means consecutive rounds overlap; compounding stake across overlapping rounds is where variance bites -- a later concern.

---

## 8. Concrete recommendations (summary)

1. Backbone: LightGBM regression per target on all 178 features, `num_leaves 31, lr 0.01, 5000 trees, colsample 0.15, min_data_in_leaf ~0.3% rows, lambda_l2 10, bagging 0.7, max_bin 16`; sweep leaves {15,31,63}, lr {0.005,0.01,0.02}, colsample {0.1,0.2,0.3}, min_data_in_leaf {0.1%,0.3%,1%}. Then XGBoost (GPU) and CatBoost (GPU if the Blackwell smoke test passes) with equivalent settings as family diversity.
2. Targets: train on `target_everest` plus the 4-8 auxiliary targets most predictive of `target_everest` in CV; rank-average with primary weight 0.4-0.6.
3. Validation: expanding-window exped-grouped folds, purge 20 and embargo 20 expeds, >= 4 folds; per-exped platform metric plus the `3*FIT - rho_b*B - rho_c*C` proxy using `train_benchmark_models` (v1_sherpa) as the benchmark proxy; block-bootstrap CIs.
4. Post-processing inside `predict`: per-exped rank -> gaussianize -> ridge-neutralize to benchmark (p_b in {0,0.25,0.5,0.75}) and to core-mean / feature set (p_c in {0,0.25,0.5}) -> re-rank to [0,1]. Tune (p_b, p_c) on CV against the proxy.
5. Diversity members (optional, later handoffs): residualized-target GBMs, a numpy-exported multitask MLP (AE optional), an XGBRanker. Skip TabPFN, era boosting, and heavy LTR.
6. Shipping: build and pickle under Python 3.11; confirm which libraries exist in the platform sandbox before relying on xgboost/catboost objects inside the pkl (xgboost 3.4+ cannot even install on 3.11); never embed torch.

## 9. Open questions for the design phase

- Are the 10 core feature names obtainable (schema verbose, train-job metrics `inov_feature_coverage`)? If not, INOV neutralization must be approximated.
- Does the sandbox that runs `predict` have lightgbm / xgboost / catboost / sklearn, and which versions? (Determines what can live in the pkl.)
- Actual exped count and instruments per exped in `atlas` train (sets min_data_in_leaf, fold sizes, CI widths).
- Is the realized target centred only, or also gaussianized server-side (affects absolute FIT scale, not conclusions)?
- User's laptop OS (Windows vs Linux) decides LightGBM GPU feasibility (irrelevant at this scale) and CatBoost GPU wheel path.

## Source list
- https://raw.githubusercontent.com/numerai/example-scripts/master/numerai/example_model.ipynb
- https://raw.githubusercontent.com/numerai/example-scripts/master/numerai/hello_numerai.ipynb
- https://raw.githubusercontent.com/numerai/example-scripts/master/crypto/example_model.ipynb
- https://raw.githubusercontent.com/numerai/numerai-tools/master/numerai_tools/scoring.py
- https://docs.numer.ai/numerai-tournament/data
- https://docs.numer.ai/numerai-tournament/scoring/correlation-corr
- https://docs.numer.ai/numerai-tournament/scoring/meta-model-contribution-mmc
- https://docs.numer.ai/numerai-tournament/scoring/feature-neutral-correlation
- https://docs.numer.ai/numerai-tournament/scoring/definitions
- https://docs.numer.ai/numerai-tournament/staking
- https://docs.numer.ai/numerai-crypto/data
- https://forum.numer.ai/t/super-massive-lgbm-grid-search/6463
- https://forum.numer.ai/t/hyperparameters-optimization-for-small-lgbm-models/6693
- https://forum.numer.ai/t/era-boosted-models/189
- https://forum.numer.ai/t/era-wise-time-series-cross-validation/791
- https://forum.numer.ai/t/super-massive-data-release-deep-dive/4053
- https://forum.numer.ai/t/v4-vs-v4-1-nomi-vs-ralph-vs-cyrus/6331
- https://forum.numer.ai/t/xgboost-vector-leaf-output-two-targets-experiment/8355
- https://forum.numer.ai/t/autoencoder-and-multitask-mlp-on-new-dataset-from-kaggle-jane-street/4338
- https://forum.numer.ai/t/foundation-models-on-numerai-data-tabpfn-v2-5-and-tabicl-v2/8269
- https://forum.numer.ai/t/learning-to-rank/454
- https://forum.numer.ai/t/model-diagnostics-feature-exposure/899
- https://forum.numer.ai/t/how-to-safely-perform-feature-neutralization/1015
- https://forum.numer.ai/t/feature-neutralization-increases-bias-and-reduces-variance/7486
- https://forum.numer.ai/t/an-introduction-to-feature-neutralization-exposure/4955
- https://forum.numer.ai/t/optimizing-for-fnc-and-tb-scores/5132
- https://forum.numer.ai/t/initial-mm-correlation-tests-with-rain-dataset/6825
- https://forum.numer.ai/t/removing-dangerous-features/5627
- https://forum.numer.ai/t/vibesciencing-my-way-through-v5-2-data-faith-ii/8214
- https://forum.numer.ai/t/signal-miner-find-unique-alpha-beat-the-benchmark/7922
- https://forum.numer.ai/t/what-do-the-top-models-do-that-lesser-models-dont/4753
- https://forum.numer.ai/t/orthogonal-model-performance-across-eras/7459
- https://github.com/pytorch/pytorch/releases/tag/v2.15.0
- https://pytorch.org/blog/pytorch-2-7/
- https://download.pytorch.org/whl/
- https://xgboost.readthedocs.io/en/stable/install.html
- https://xgboost.readthedocs.io/en/stable/gpu/index.html
- https://xgboost.readthedocs.io/en/latest/changes/v3.4.0.html
- https://xgboost.readthedocs.io/en/latest/changes/v3.1.0.html
- https://xgboost.readthedocs.io/en/latest/changes/v3.0.0.html
- https://pypi.org/project/xgboost/ , https://pypi.org/project/xgboost-cu12/
- https://pypi.org/project/lightgbm/
- https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html
- https://github.com/microsoft/LightGBM/releases/tag/v4.6.0
- https://pypi.org/project/catboost/
- https://github.com/catboost/catboost/releases
- https://github.com/catboost/catboost/pull/3108
- https://github.com/catboost/catboost/issues/2967
- https://catboost.ai/docs/en/features/training-on-gpu
