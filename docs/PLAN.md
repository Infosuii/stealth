# Design review and final build plan — Everesteer "Himalayas" bot (repo `stealth`) — revision 2

Revision notes: this version resolves all 12 blocker/major items and the 6 minor items from the critique. Each resolution is marked **Fix (Cn)** where *n* is the critique item. Claims that could not be re-verified in this pass are marked (unverified). Verified on the sandbox during this revision: SDK `everesteer-api 0.4.1` with `fit20(predictions, target)`, `unq20(predictions, ai_model, target)`, `inov(predictions, features, target)` (no `zero_rule` kwarg — the previous draft's call signature was wrong); `cloudpickle 3.1.2` has `register_pickle_by_value`; sandbox has `torch 2.15.0+cpu` installed, so a stray torch import would pass locally and die in the lane.

## Part 1 — Scores (1–10)

Note: the `score-aware` and `robust-ops` proposals arrived truncated (sections 1–5 missing for score-aware, 1–6 for robust-ops); they are scored on what is visible plus what their later sections imply.

| Criterion | fit-max | score-aware | robust-ops |
|---|---|---|---|
| Expected tournament score | **8** — wide honest search, multi-target, family diversity, neutralisation in `predict` | **8** — same backbone plus a *scenario* UNQ/score estimator and a benchmark surrogate that make neutralisation tunable offline | 6 — ops-first; modelling ambition unclear from visible text, lag features deferred |
| Robustness vs overfitting/leakage | 7 — nested blend, leak assertion, "never fit to v1_sherpa" rule; but unpaired admission gates and large Optuna budgets (fixed below) | **9** — pre-registered diag budget, two fold offsets, strict member gate (gate now re-specified as a paired test) | 8 — stateless fallback design, λ_c cap |
| Feasibility on the laptop | 6 — leans on XGB-GPU/CatBoost-GPU on Blackwell with known sm_120 breakage and a py3.11 vs xgboost≥3.4 conflict | 8 — GPU only for a 2-min torch smoke + optional MLP | **9** — GPU avoided by design; CPU trees on 83 MB data |
| Handoff / operational simplicity | 7 — good template, dual-shell blocks, `report.py` always pushes | **8** — adds INDEX.md status table, pre-commit secret hook, "rules for the executing AI", resumable steps | **9** — 15:30 UTC monitor, scheduled-task fallback, designed degradation |
| Contract compliance (.pkl, [0,1], lane pins) | 8 — lazy-deserialised model bytes, verify in fresh subprocess; missed pickle-by-reference | 8 — `pkl_check` clean subprocess, manifest + sha256, determinism check; missed pickle-by-reference | 7 — same rules, less detail visible |
| **Total** | 36 | **41** | 39 |

**Decision (unchanged):** backbone = fit-max's modelling plan, *constrained* by score-aware's scenario-score selection and robust-ops' CPU-only/stateless operational design, with the statistical and packaging corrections below.

---

## Part 2 — Final build plan

### Executive summary (plain language)

- The dataset is tiny (83 MB) and the signal is weak: the best honest model gets a per-round rank correlation of ~0.12 with the target. Anything scoring much higher offline is a leak or selection noise, not a breakthrough. Our edge comes from *discipline*, not model size.
- The score pays three times for the same signal if it is also different from the benchmark and from a 10-feature average. The bot (a) predicts as well as it can, then (b) partially removes the benchmark direction *inside the prediction function* (the platform hands us the benchmark at predict time) and (c) partially removes a proxy of the 10 "core" features. Honest expectation after re-doing the algebra with the zero rule: about **+0.004 per round net**, coming mostly from down-weighting a low-Sharpe benchmark direction in rounds where the benchmark is negative — not from "free UNQ credit". The sign flips if the live benchmark is actually much stronger than its 36-round record, so the strength of this knob is set conservatively and re-read from live data after ~10 rounds.
- The model is an ensemble of many small, heavily-regularised gradient-boosted trees (LightGBM, plus XGBoost on CPU for variety), trained on the main target and 6 related targets, plus a ridge model. Equal-weight blend by default. CatBoost is demoted to 3 fixed configs that are kept only if the test run measures them as cheap.
- Validation is strict: time-ordered folds with a 20-era purge and embargo, tree counts chosen on an *inner* split (never on the scored fold), feature selection and surrogates fitted only on training eras, a final walk-forward block, and **paired** block-bootstrap tests between configs. Two configs whose paired difference has P(diff>0) < 0.8 are a tie.
- Hyperparameter search is deliberately small (≤25 trials over a coarse grid, top-k averaged) because the best of 150 noisy trials is ~+0.008 of pure winner's-curse bias. The multi-hour budget goes into *members* (targets, feature bags, families), not trial count.
- **GPU is not required.** Trees train in seconds on CPU. The GPU is used only for an optional neural member late in the plan (weights exported to numpy — the pickle may not contain torch).
- The pickle is built by a standalone script with cloudpickle pickling our code **by value**, and tested in a throwaway venv where the `stealth` package is *not installed*, because the lane sandbox doesn't have it either.
- Daily operation is hands-off: we upload the pickle once (laptop → platform directly; the pickle is never committed to the repo); the platform's auto-run lane submits every weekday. A laptop scheduled task is the fallback.
- Every step the laptop runs is a self-contained `handoffs/NN-*.md` that a context-free GPT can execute using **absolute venv python paths** (no activation state assumed); it reports back by committing `handoffs/results/NN-*.{md,json}` and pushing (and says loudly if the push failed).
- The repo is currently public. **We recommend flipping it to private before the first results commit** (Open question 1); otherwise the winning config and OOF results are free for every competitor.

### Data facts that matter

- Train 468,236 rows × 178 int8 features, 6,522 eras (`exped_0003..6524`), 44–92 instruments/era (most of history 44–72; validation/live ~90). Validation 1,392 eras, targets blanked, 42-era gap after train, ends 98 eras before live. Live = 1 era = 90 rows. Rows are (instrument, era); ids opaque; no instrument identity across eras (stateless `predict`).
- Features: per-era decile bins 0..9, −1 = missing (10.8% overall, 63% in the first era-decile; from `exped_2804` no feature is fully missing; live is 3% missing). Diverse (0 pairs >0.9 corr). Importance flat.
- Target `target_everest`: 5-level per-era bell bin of a ~20-day forward return; target ACF ~0 beyond lag 20 → **purge + embargo ≥ 20 eras**. 14 aux targets correlate 0.19–0.86 with everest; training *on* Tougroute hurt (0.090 vs 0.118) → aux targets are blend members, never replacements.
- Honest baseline: shallow LGBM per-era Spearman 0.118–0.121 (FIT ≈ 0.226×Spearman ≈ 0.027), Sharpe ~0.57; ridge λ=1e5 0.116; top-20 sign-sum 0.120. Seed bagging worthless (0.985 corr). 300-tree nl31 default overfits (0.082 OOS vs 0.543 in-sample).
- Noise floor: per-config FIT SE ≈ 0.0033 (≈186 effective 20-era blocks). Paired SE of a *difference* between two highly-correlated configs (r≈0.9) ≈ 0.0015. Expected tuning gain ≤0.002. These three numbers drive every gate below.
- `v1_sherpa` on train is in-sample (0.354 vs any honest 0.12; live FIT only 0.009, per-round std ≈0.03–0.04, B<0 in ~40% of rounds) → never fit to, blend with, or select on it. Offline UNQ against it on train is biased negative for every honest model. The validation-split `v1_sherpa` file exists (no targets needed) and is usable for exact ρ_b. Validation-board unq/fit ≈0.2–0.3 vs live ≈0.5–0.6 → v1_sherpa is likely still partly in-sample on validation too (unverified; inferred from board ratios), so even the diag lane overstates benchmark strength.
- **UNQ reference:** `download_dataset(split='minera_model')` for exped_8055 is Spearman 1.000 with the live `v1_sherpa` file (verified by the critic on `/home/climber/.claude/jobs/8bc89d82/tmp/futures_minera_model.parquet`). The benchmark handed to `predict` *is* the UNQ reference; the reference-ambiguity risk is minor. The `p_b ≤ 0.75` cap stays for the FIT-cost reason only.
- The 10 INOV core features are unpublished and the hosted job's `inov_feature_coverage` always reports `{resolved:10, frozen:10}` regardless of training features — subset probing reveals nothing. Live benchmark INOV is 0.0016 vs FIT 0.009, so the core average nearly spans v1_sherpa → **core proxy = equal-weight mean of v1_sherpa's top-10 feature loadings** (**verified at M0 on eras <2800 from the train benchmark file**, per-era Spearman loading on v1_sherpa: Agouti +.47, Tazzarine +.45, Azzaden −.42, Tagmat −.39, Zaouiat +.39, Ighzer −.38, Amazigh −.36, Tafouk +.35, Zerrouka +.33, Mrirt +.32 — 7/10 overlap with the critique's all-era list; Tagmat/Zaouiat/Zerrouka replace Takerkoust/Ouarzazate/Azaghar. `train.py` recomputes it per run and stores it in `features.json`), not target-univariate top-10. **Fix (C12)**.
- Competitive bar (live, ≥10 rounds): median FIT 0.011, P75 0.0185; top-10 ≈ 0.05/round (FIT ≥0.025, UNQ ≥0.012, INOV ≥0.012). Validation board P75 fit20 = 0.0214.

### Model architecture

**Members (all CPU; data is 83 MB):**

| Member | Spec | Role |
|---|---|---|
| LGBM-everest ×K | coarse grid: `num_leaves` {4,8,16,31}, lr {0.01,0.03}, `min_data_in_leaf` {500,2000,5000}, `feature_fraction` {0.15,0.3}, `lambda_l2` {10,100}, bagging 0.7, `max_bin=16`, −1→NaN native missing; tree count from inner-split early stopping (cap 1500), then frozen | main lever; **top-3 grid configs averaged**, not argmax (C6) |
| LGBM-aux ×6 | best everest params, targets Tougroute, Ghat, Meltsene, Timesguida, Ighil, Tarhat (NaN-target rows dropped) | target diversity (largest documented lever); everest group weight 0.5 |
| XGB-everest | xgboost **3.2.0 CPU hist**, depth {3,4,6} × lr {0.02,0.05} × `min_child_weight` {200,1000} = 12 fixed configs, top-2 averaged | family diversity |
| Cat-everest (conditional) | catboost 1.2.10 CPU, **3 fixed configs** (depth 4, 400 iters, lr 0.05, `l2_leaf_reg` {3,30,100}); **kept only if the test run measures ≤2 min per fit**, otherwise dropped — XGB-hist + ridge already give family diversity | **Fix (C11)**: previous 20-trial estimate was 5–10× too cheap |
| Ridge | λ≈1e5 on value/9 with −1→0.5 + 15 missing indicators + `n_missing` | near-free, different failure mode |
| Feature-subset arms | LGBM on features with \|sharpe\|>0.1 **selected per fold on training eras only**; LGBM on 4 random 50% column bags (seeded) | tests whether near-noise features hurt; **Fix (C7)** |
| MLP (optional, M6) | 178→256→64→1 GELU, dropout 0.3, aux-target head, 5 seeds; torch on GPU, **weights exported to numpy**, forward pass reimplemented in numpy inside the pickle | admitted only via the paired gate and corr with GBDT mean < 0.95 |

Era universe knobs chosen by the forward block: start era ∈ {≥2000, ≥2800 (default), all}; linear time-decay weight on/off; **`n_rows ≥ 80` era filter or weight** (universe-size shift: most of train has 44–72 instruments, live has 90) **Fix (C14)**.

**Resolved — GPU trees:** all tree members train on CPU with the exact lane-pinned versions (Blackwell sm_120 needs xgboost ≥3.4 / Python ≥3.12, incompatible with the 3.11 pickle pin). GPU = optional MLP only.
**Resolved — search style (C6):** fixed coarse grids above (LGBM 96 cells → Optuna TPE with **≤25 trials** over that grid, SQLite, resumable; XGB 12 fixed; Cat 3 fixed). Winner = **average of top-k** configs' predictions, confirmed **once** on the forward block. Any OOF FIT > M1 baseline + 0.004 is flagged "suspect — winner's curse or leak" rather than celebrated. The old `mean − 0.5·SE` objective is dropped: SE is ~constant across configs, so it never changed the argmax (C5).
**Resolved — lag/delta features:** `predict` is stateless and the lane has no history. Not in v1; logged as a v2 lever for the manual daily path only.

### Validation protocol (offline FIT/UNQ/INOV proxies)

- 5-fold chronological exped-grouped CV on eras ≥2800 (~745-era test blocks), **purge 20 + embargo 20** both sides; training folds use both sides. Plus one strict **forward block**: train <5925, test 5945–6524. `tests/` asserts every fold gap ≥20.
- **Tree count (C8):** inside each training fold, an inner purged split (last 15% of training eras, 20-era gap) picks the early-stopping iteration. The **median inner best-iteration across folds** is frozen; test-fold predictions are made with that frozen count only. The scored fold never influences its own tree count.
- **Nested feature work (C7):** the \|sharpe\|>0.1 subset, the core-proxy list, and the benchmark surrogate are each recomputed inside every fold from its training eras (or from eras <2800, which are never scored). Nothing fitted on a test fold's eras touches that fold.
- Primary metric: `everestapi.scoring.fit20(predictions, target)` per exped on OOF preds → mean, std, Sharpe, % positive eras, fraction of 20-era rolling means <0, 20-era block-bootstrap 90% CI.
- **Admission / comparison gate (C5):** every "A beats B" decision uses a **paired 20-era block bootstrap of the per-era FIT difference** (2,000 resamples). Adopt A iff `P(diff > 0) ≥ 0.8` on CV **and** diff ≥ 0 on the forward block. Otherwise "tie — keep the simpler one". Reports always print `diff_mean, diff_se, p_pos`. Unpaired CIs are shown for context only.
- **Scenario score (C4, C9):** replaces the closed-form `3·FIT − ρ_b·B − ρ_c·C`. Per OOF era we simulate a benchmark `b = surrogate_direction·s + ε` with per-era FIT drawn from `N(0.009, 0.035²)` (36-round live record; also run at means 0.0 and 0.02), apply neutralisation at each `p_b ∈ {0, .25, .5, .75}`, and compute `fit20`, `unq20(pred, b, target)` **with the zero rule** (UNQ := 0 in eras where the benchmark's own covariance with the target is < 0; verified at M0: `everestapi/scoring.py` 0.4.1 applies it inside `unq20`, and the vectorised simulation kernel `metrics.unq_rows` re-implements the same `< 0` rule — note `< 0`, not `≤ 0`; `metrics.per_era_unq` therefore does NOT re-apply it) and `inov(pred, core_proxy_frame, target)`. We report three regimes separately: benchmark-positive eras, benchmark-negative eras, and the mixture, each with 500 Monte-Carlo draws. Expected shape (from the critic's algebra at ρ=0.6, O=0.027, p_b=0.5): ≈−0.001 in B>0 eras, ≈+0.010 in B<0 eras, net ≈+0.004.
- **ρ_b exactly, not via surrogate (C7):** per-era correlation of our predictions with `v1_sherpa` is computed directly on the **validation split** (benchmark file available, no targets needed). The ridge surrogate (features→v1_sherpa, fitted on eras <2800) is kept only to generate the *direction* in the simulation above.
- Direct `unq20` against v1_sherpa-on-train is logged, labelled "contaminated — not for selection".
- INOV proxy: `inov()` vs the v1_sherpa-loading top-10 mean, labelled "proxy; true core set unpublished".
- Leak assertion: OOF Spearman >0.18 → hard failure. Winner's-curse flag: OOF FIT > M1 + 0.004 → "suspect" banner in the report.
- Exact OOS oracle: validation-diagnostics lane (free; fit20/unq/inov/sharpe over 1,392 eras vs the *true* core set). **Budget: 5 uploads total, pre-registered** (M1 baseline, M3, M5 ×2, final), each a USER MUST APPROVE handoff. **Use (C9, C13):** diag results answer *direction* questions only ("does neutralisation cost FIT? is INOV > 0 vs the real core set?"). `p_b` is **not** tuned on the diag lane because v1_sherpa is stronger there than live; it comes from the live-calibrated simulation and is revised only after reading the canary's live UNQ/FIT ratio over ≥10 rounds.

### Neutralisation / post-processing (inside `predict`, identical offline)

1. Each member → raw score on the live rows → gaussianise per era: `z = norm.ppf((rank_avg − 0.5)/n)`.
2. Blend: equal weight within target group, everest group 0.5 / aux group 0.5. NNLS weights (sum-to-1, floor 0.02) computed nested (folds 1–4 → fold 5 + forward) and adopted only if they pass the paired gate vs equal-weight. Default is equal-weight.
3. **Benchmark handling (C15):** if `live_benchmark_models` is `None` → skip. Else coerce Series/single-column DataFrame to a Series, **reindex to `live_features.index`**; if coverage <50% or variance 0 → skip neutralisation and set a flag; otherwise gaussianise `m` and apply `z ← z − p_b·m(m·z)/(m·m)` over the covered rows. Default **`p_b = 0.5`** from the simulation; hard cap 0.75; revisited after ≥10 live rounds via the canary's UNQ/FIT ratio (raise only if the live ratio stays ≥0.5 and our live FIT cost is <0.002).
4. Core-proxy partial neutralisation: same with the v1_sherpa-loading top-10 feature mean (−1→NaN→per-era mean fill), default `p_c = 0.25`, **hard cap 0.5**.
5. Final: `pred = (rank_avg(z) − 0.5)/n` → strictly in (0,1), no NaN (NaN member dropped per row; all-NaN row → 0.5). Return `pd.DataFrame({"prediction": pred}, index=live_features.index)`.

Offline, FIT with/without steps 3–4 is reported per regime so the neutralisation cost is quantified before any constant is frozen.

### Packaging (.pkl contract)

- **Fix (C1) — pickle by value.** `stealth/bundle.py` is invoked as a script that: (a) builds the closure `predict(live_features, live_benchmark_models=None)` capturing feature names, each member as **serialised bytes** (LGBM `model_to_string`, XGB `save_raw("ubj")`, CatBoost `.cbm` bytes, ridge/MLP as numpy arrays), blend weights, `p_b`, `p_c`, core-proxy names, `VERSION`, git sha; (b) places every helper the closure needs (`gaussianise`, member loaders, numpy MLP forward, neutralise) in **one module `stealth/_runtime.py` that imports only stdlib + numpy + pandas + scipy + lightgbm + xgboost + catboost inside function bodies**; (c) calls `cloudpickle.register_pickle_by_value(stealth._runtime)` (and `stealth` itself as belt-and-braces) before `cloudpickle.dump`. Members deserialised lazily on first call. Columns selected **by name**; missing column → raise.
- Built only in the pinned **Python 3.11** venv (`requirements-lock.txt`, exact versions verified on the sandbox this pass: lightgbm==4.7.0, xgboost==3.2.0, scikit-learn==1.9.1, pandas==3.0.6, numpy==2.4.6, scipy==1.17.1, catboost==1.2.10, pyarrow==24.0.0, cloudpickle==3.1.2, **everesteer-api==0.4.1** (C18)). Declare `python_version="3.11"`. The optional GPU/torch venv never touches the pickle.
- **Fix (C1) — `pkl_check` proves portability.** `python -m stealth.pkl_check models/stealth_vX.pkl` creates (or reuses) a throwaway venv `artifacts/pklcheck_venv` from `requirements-lock.txt` **without `stealth` installed**, then runs the checker as `python -I` with `cwd=/tmp` (Windows: `%TEMP%`) in a subprocess. It: loads the pickle; asserts `"stealth" not in sys.modules` and `"torch" not in sys.modules` after `predict()`; scans the raw bytes for `b"torch"` **and** `b"stealth"` (the by-value module name may legitimately appear in pickled code metadata — the check whitelists only `stealth._runtime` qualname strings and fails on any other `stealth.` reference, (unverified) that this whitelist is needed; M0 decides); calls `predict(live)`, `predict(live, bench)`, `predict(live, bench_misaligned)` (extra + missing ids, reordered), `predict(live, bench_constant)`, and `predict(live, None)` on the live parquet and 50 random validation eras (C15); asserts exact `pd.DataFrame`, one column, index equality, float dtype, all in (0,1), no NaN, deterministic across two calls, <5 s/era, file <50 MB. Writes `models/stealth_vX.pkl.sha256` and `models/stealth_vX.manifest.json` (sha256, git sha, members, `p_b`, `p_c`, python/lib versions, build time). The sandbox (identical pins) runs this as the final gate before any upload handoff.
- **Fix (C3) — the `.pkl` is never committed.** `.gitignore` contains `models/*.pkl`. Only `*.sha256` and `*.manifest.json` are committed. The pickle travels laptop → platform directly via `predict --upload`; the sandbox verifies a *sandbox-built* copy of the same commit (same pins → same bytes modulo tree-training nondeterminism; the manifest records both sha256s).

### Repo layout + exact commands

```
stealth/
  README.md                      # one-screen quickstart → handoffs/INDEX.md
  pyproject.toml                 # package "stealth" (installed editable in the train venv ONLY)
  requirements-lock.txt          # exact lane pins (py3.11) incl. everesteer-api==0.4.1
  requirements-gpu.txt           # OPTIONAL torch cu128 venv for the MLP member only
  .env.example                   # EIQ_API_KEY=  EIQ_BASE_URL=https://api.everesteer.ai
  .gitignore                     # data/ artifacts/ .env *.log models/*.pkl reports/**/preds_*.parquet
  configs/smoke.yaml  configs/test.yaml  configs/full.yaml
  stealth/
    data.py        # SDK download, load, −1→NaN, era ints, era/n_rows filters, missing indicators
    cv.py          # purged/embargoed grouped folds + inner ES split + forward block + paired block bootstrap
    metrics.py     # fit20/unq20(+zero rule)/inov wrappers, regime simulation, report dict
    features.py    # per-fold subsets, sign-flip screen, core-proxy list (from v1_sherpa loadings)
    surrogate.py   # features→v1_sherpa ridge on eras<2800 (direction only, for the simulation)
    models/        # lgbm.py xgb.py cat.py linear.py mlp.py (torch→numpy export)
    search.py      # Optuna over coarse grids, SQLite in artifacts/<run>/optuna.db (resumable), top-k averaging
    train.py       # config → OOF parquet + member bytes + metrics.json (--resume); writes artifacts/<run>/run.log itself
    blend.py       # EW/NNLS nested blend, neutralisation regime simulation on OOF
    _runtime.py    # ONLY code the pickle may reference (stdlib/numpy/pandas/scipy/lgbm/xgb/cat imports inside functions)
    bundle.py      # build pkl by value + manifest + sha256
    pkl_check.py   # throwaway venv (no stealth), python -I, cwd=/tmp, contract + portability test
    predict.py     # daily: round check → download live+bench → predict → validate → (--submit) → status
    diag.py        # validation-split predictions file; upload behind --upload
    monitor.py     # submission exists for today's exped; active upload sha == manifest sha (C16); exit code
    report.py      # artifacts/<run>/metrics.json → handoffs/results/NN-*.{md,json,txt}; commit+push; loud on push failure
    selfcheck.py   # pinned-version compliance, OS/CPU/GPU/TZ table, hooksPath check
  scripts/setup.sh  scripts/setup.ps1  scripts/hooks/pre-commit  scripts/daily_task.{sh,ps1}
  handoffs/INDEX.md  handoffs/00-env-setup.md  01-... NN-...
  handoffs/results/  # NN-results.md NN-metrics.json NN-log-tail.txt (committed)
  models/            # *.sha256 + *.manifest.json only (pkl gitignored)
  tests/             # cv gaps ≥20; inner-ES isolation; fit20 parity vs everestapi.scoring; pkl contract; hook regex; no-secrets
```

Commands (identical on Windows/Linux; `$PY` is the **absolute** venv interpreter, e.g. `/home/<u>/stealth/.venv/bin/python` or `C:\Users\<u>\stealth\.venv\Scripts\python.exe` — every handoff spells it out, never relies on activation) **Fix (C10a)**:

```
$PY -m stealth.selfcheck
$PY -m stealth.data download                                   # train/validation/live + benchmark files → data/
$PY -m stealth.train --config configs/<smoke|test|full>.yaml --run-id <id> --resume   # logs to artifacts/<id>/run.log itself (C10d)
$PY -m stealth.blend --run-id <id>
$PY -m stealth.bundle --run-id <id> --out models/stealth_v1.pkl
$PY -m stealth.pkl_check models/stealth_v1.pkl
$PY -m stealth.report --run-id <id> --handoff NN
$PY -m stealth.predict --pkl models/stealth_v1.pkl             # dry run (validate_submission only)
$PY -m stealth.predict --pkl models/stealth_v1.pkl --submit    # USER-GATED
$PY -m stealth.predict --pkl models/stealth_v1.pkl --upload    # USER-GATED: push to auto-run lane
$PY -m stealth.diag --pkl models/stealth_v1.pkl [--upload]     # USER-GATED upload
$PY -m stealth.monitor
$PY -m pytest -q
```

### Handoff protocol

`handoffs/NN-<slug>.md`, fixed section order, fully self-contained:

1. **Purpose + success criterion** (numbers expected, e.g. `oof_fit_mean` 0.024–0.030; paired `p_pos` thresholds where relevant).
2. **Preconditions**: repo at commit `<sha>` (`git pull`), `.env` present with `EIQ_API_KEY`, ~5 GB disk, `uv` installed.
3. **HUMAN STEP block (C10b)** — present in 00 and repeated verbatim in any handoff that pushes: the human (not the GPT) must (i) run `gh auth login` (or have SSH keys) so `git push` works, (ii) create `.env` from `.env.example` and paste the key, (iii) if the repo is private, confirm `gh repo view Infosuii/stealth` succeeds. The GPT verifies with `git push --dry-run` and `test -s .env` / `Test-Path .env`, never by printing contents.
4. **Environment (C10a)**: two blocks — bash and PowerShell — `uv python install 3.11`, `uv venv .venv --python 3.11`, `uv pip install -r requirements-lock.txt -e .` (train venv only), `git config core.hooksPath scripts/hooks` **(C2)**, then `$PY -m stealth.selfcheck` (which also asserts `core.hooksPath` is set). `$PY` is defined as an absolute path at the top of each block.
5. **Steps**: numbered; each with the exact command, expected laptop runtime, 2–3-line expected output signature, idempotent/resumable. No `tee`/shell redirection: `train.py` writes `run.log` itself and prints a progress line every minute (C10d).
6. **If it fails**: per-step fallback (e.g. "set `threads: 8`", "skip step 4 and continue").
7. **Report back (mandatory, also on failure) (C10c)**: `$PY -m stealth.report --run-id <id> --handoff NN` → `handoffs/results/NN-results.md` (env table, metrics table incl. paired `diff_mean/diff_se/p_pos`, regime table, wall times, last 60 log lines, traceback), `NN-metrics.json` (schema: `run_id, git_sha, status, host{os,cpu,ram_gb,gpu,driver,python,libs,tz}, timings, configs[]{name, fit_mean, fit_se, sharpe, forward_fit, rho_b_val, rho_c, regime{bpos,bneg,mix}, ci90, paired_vs_baseline{diff,se,p_pos}}`), `NN-log-tail.txt`; then `git add handoffs/results && git commit -m "results: NN <slug>" && git push`. If the push fails, `report.py` **exits non-zero and prints `PUSH FAILED — results are saved locally in handoffs/results/; ask the human to run git push`**; the results files are already written at that point. **Hyperparameter values are written to `artifacts/` only, not to results files** (C3) — results carry metrics and config *names*.
8. **Rules for the executing AI**: no code/config edits unless a step says so; never `--submit`, `--upload`, or diag upload unless the handoff carries a **USER MUST APPROVE** banner; never commit `data/`, `.env`, `*.pkl`; never `cat`/`echo` `.env`, never `git remote -v`, never `git config --list`; stop and write a results file on any judgement call.

**Pre-commit secret hook (C2):** `scripts/hooks/pre-commit` (bash; on Windows runs under Git-for-Windows' bundled sh) blocks staged content matching `EIQ_API_KEY\s*=\s*\S{8,}`, `X-API-Key`, `ghp_[A-Za-z0-9]{20,}`, `github_pat_`, `gho_`, `AKIA[0-9A-Z]{16}`, `-----BEGIN`, and `https://[^/\s]*@github\.com`; lines that are exactly `[0-9a-f]{40}` or `[0-9a-f]{64}` (git shas, sha256 files) are explicitly allowed. `tests/test_hook.py` feeds it a sample results file + sha256 file (must pass) and a fake key (must fail). Activated by `core.hooksPath` in handoff 00; `selfcheck` warns if unset.

`handoffs/INDEX.md` = status table (NN, slug, status, waiting-on Claude/GPT/user, result sha, decision log). Claude's loop: build + smoke on sandbox → write handoff → push → user runs via GPT → `git pull` → read `NN-metrics.json` → iterate.

### Daily operations

- **Primary path (zero daily effort):** after the final run, a user-approved handoff runs `predict --upload` from the laptop; poll `get_upload_status` until `sandbox_ready=1`; `monitor` records the accepted upload id/sha into `models/active.json` (committed). The lane runs the pickle at ~13:02 UTC every weekday and submits. Re-upload only after retrains.
- **Monitor (C16):** `monitor` checks (i) `get_submission_status` has an accepted entry for today's exped, and (ii) the lane's active upload (`get_upload_status`) matches `models/stealth_vX.manifest.json` sha / recorded upload id. A mismatch (e.g. a failed re-upload leaving the old pickle live) writes `handoffs/ALERT.md` and pushes.
- **Fallback:** `scripts/daily_task.{sh,ps1}` registered as a Task Scheduler / cron job. **Clock (C17):** the scheduler fires in local time; the script itself converts and only proceeds if `datetime.now(UTC)` is within 13:05–15:55 UTC, *and* `get_current_round.submission_window == "open"`, *and* the live parquet's `exped` equals the round's `exped`. Setup instructions give the local-time equivalent of 14:30 UTC for the user's timezone (asked in Open question 2) and note the DST shift. If no accepted entry exists, it runs `predict --submit` (a direct submission replaces a lane entry; the lane never overwrites a direct one). A second run at 15:30 UTC alerts if still nothing. If the laptop is off, the lane alone carries the day.
- **What the user does each day: nothing.** Weekly: glance at `get_scores` via a tiny `monitor --scores` handoff, or the dashboard.
- **Before Monday 2026-11-16 13:00 UTC** the already-armed hosted baseline (cv FIT 0.016) will submit for round 38. **Recommendation: leave it armed as a live canary** — it yields a live_FIT/cv_FIT calibration ratio and, crucially for C9, a live **UNQ/FIT ratio** that tells us how strong the benchmark really is out of sample before we touch `p_b`. User decision (Open question 3).

### Compute plan

| Tier | Where | Content | Runtime / RAM |
|---|---|---|---|
| Smoke (every iteration) | sandbox, 2 cores | `smoke.yaml`: eras ≥5000, 3 purged folds with inner ES split, LGBM everest + Tougroute (200 trees), ridge, EW blend, regime simulation (50 draws), bundle, `pkl_check` in throwaway venv on live parquet, `pytest` | ≤8 min, <3 GB |
| Test run (handoff 02) | laptop CPU, 16 threads | `test.yaml`: eras ≥2800, 5 folds + forward; LGBM everest 12 grid cells; 3 aux targets at best params; XGB-CPU 4 fixed configs; **Cat 1 config, timed** (decides C11 keep/drop); ridge; blend + regime simulation (500 draws); bundle; `pkl_check`; 2-min GPU smoke (torch cu128 matmul only) | 30–45 min, <6 GB. Calibrates per-fit wall time and member correlation matrix; must reproduce OOF FIT 0.026±0.004 |
| Final run (handoff NN) | laptop CPU (+GPU for MLP) | LGBM everest ≤25 Optuna trials over the coarse grid (~1 min/trial → ≤30 min), top-3 averaged; XGB 12 fixed configs (~25 min); Cat 3 configs if kept (≤20 min by construction); 6 aux × 2 families at family-best params (~40 min); subset + 4 column-bag arms (15 min); era-start / decay / `n_rows≥80` knobs (20 min); MLP 5 seeds on GPU (~20 min, optional); nested blend + regime simulation + paired bootstraps (10 min); full-train refit of admitted members (2 seeds, frozen tree counts), bundle, `pkl_check`, diag file (20 min) | **2.5–3.5 h** wall, <8 GB; resumable via Optuna SQLite + member cache; the budget is spent on members, not trials (C6) |

### Milestones

| # | Milestone | Success criterion |
|---|---|---|
| M0 | Repo skeleton, data loader, purged CV with inner ES split, paired bootstrap, metric parity tests, hook + hook test, `_runtime.py` + by-value bundle + throwaway-venv `pkl_check`, core-proxy list recomputed from v1_sherpa loadings on eras <2800, smoke config, handoffs 00/01 | `pytest` green on sandbox and laptop; `pkl_check` passes in a venv without `stealth`; `selfcheck` shows pinned versions + hooksPath; results file pushed by the GPT |
| M1 | Shallow LGBM everest + ridge baseline under full CV (handoff 02 test run) | OOF FIT 0.026±0.004, forward FIT ≥0.020, Sharpe ≥0.5; per-fit wall times recorded (Cat decision); diag upload #1 (approved). **Diag fit20 is informational (C13)**: record the OOF→validation ratio; pass = ratio within the range implied by the canary's live/cv ratio once available, else simply logged |
| M2 | Coarse-grid tuning + era-start/decay/`n_rows` knobs | paired `P(diff>0) ≥ 0.8` vs M1 on CV and diff ≥0 on forward; **any gain >0.004 is flagged suspect** |
| M3 | Multi-target blend (6 aux) | nested blend passes the paired gate vs everest-only; aux members individually scored on everest; diag #2 |
| M4 | Family diversity (XGB, Cat-if-kept, subset/bag arms), blend rule decided | member corr matrix; paired gate or Sharpe +0.05 with `p_pos ≥ 0.8`; EW vs NNLS decision logged |
| M5 | Neutralisation constants + bundle v1 | regime simulation table; `p_b`, `p_c` frozen from the simulation (not from diag); diag #3–4 confirm direction: FIT cost ≤0.003 and inov > 0 vs the true core set, unq ≥0 |
| M6 | MLP-numpy member (optional) | paired gate and corr <0.95; otherwise cut |
| M7 | Final run, pkl v-final, `pkl_check` on sandbox, user-approved upload, daily fallback dry-run, `models/active.json` | OOF FIT ≥0.027 **and ≤ M1+0.004 unless the forward block independently confirms**; diag #5 fit20 ≥0.021 (validation P75 0.0214); lane `sandbox_ready=1` with sha matching manifest; live goal FIT ≥0.025 / UNQ ≥0.012 / INOV ≥0.012 (~0.05/round) assessed after ≥20 rounds |

Honest ceiling: Spearman 0.14–0.15 (FIT 0.032–0.034). Higher offline = leak or winner's curse until proven otherwise on the forward block and live.

### Risks & mitigations

1. **Benchmark contamination on train and (partly) validation** → never fit to, blend with, or select on v1_sherpa; `p_b` from live-calibrated simulation, revised only from live canary UNQ/FIT; diag lane for direction only.
2. **Regime gap** (train ends ~6 years before live; validation 98 eras stale; universe 44→92) → era-start/decay/`n_rows≥80` knobs chosen by the forward block; prefer configs stable across blocks; live canary ratio.
3. **Selection noise / winner's curse** → ≤25 trials, top-k averaging, paired bootstrap gates, pre-registered 5 diag uploads, "suspect" banner above M1+0.004, ties declared as ties.
4. **UNQ reference** → verified identical to v1_sherpa live file (Spearman 1.000); residual risk is the zero rule, handled in the simulation; `p_b ≤ 0.75` for FIT-cost reasons.
5. **Unknown core set** → v1_sherpa-loading proxy; `p_c ≤ 0.5`; hosted INOV subset probes **dropped** (coverage field is uninformative, C12).
6. **Pickle/lane mismatch** → by-value pickling, throwaway-venv `python -I` check on sandbox before every upload, byte scan for `torch`/`stealth`.
7. **Blackwell GPU stack** → avoided for trees; torch only for the optional MLP; GPU smoke in handoff 02 decides.
8. **Secrets in a public repo** → `.env` gitignored, versioned hook via `core.hooksPath` with a tested regex set, logs scrubbed, no `remote -v`/`config --list` anywhere.
9. **Model theft from a public repo** → `.pkl` never committed; hyperparameter values kept out of results files; recommend private repo (Open question 1).
10. **Missed/malformed submission or stale artifact** → reindex to live ids, hard asserts, `validate_submission` before submit, `get_submission_status` after, sha-vs-manifest check, UTC-gated scheduled task.
11. **Windows friction / GPT shell state** → absolute `$PY` paths, `uv` for Python 3.11, no `tee`, `python -m` entry points, pathlib, resumable runs, `report.py` always writes and is loud on push failure.
12. **CatBoost cost** → 1 timed fit in the test run gates 3 fixed configs or drop.
13. **Over-engineering for a 0.12 signal** → every member must pass the paired gate; equal weights by default.

### Open questions for the user

1. **Repo visibility** — may we flip `Infosuii/stealth` to **private** now? (Strongly recommended: otherwise every competitor can read the winning config and OOF results. Your GPT works identically after `gh auth login`.) If it must stay public, confirm you accept that leak knowingly; the `.pkl` is never committed either way.
2. **Laptop OS and timezone** — Windows or Linux, and your local timezone (for converting the 14:30/15:30 UTC fallback schedule and documenting the DST shift)?
3. **Keep the already-armed hosted baseline auto-submitting from Monday** as a live canary (recommended — it is also how we calibrate `p_b` safely), or disable it?
4. **Approve ≤5 free validation-diagnostics uploads** across the milestones? (The hosted-compute INOV probe is withdrawn; no spend requested.)
5. **Daily fallback**: is the laptop normally on at 14:30 UTC on weekdays so a scheduled `monitor`/`predict --submit` task is viable, or should we rely on the auto-run lane alone?
6. **Push auth on the laptop**: will you run `gh auth login` (HTTPS) or set up an SSH key for the GPT's `git push`? Handoff 00 will carry a HUMAN STEP block for whichever you choose.
7. **Staking intent**: do you plan to stake later? (If yes, Sharpe and the benchmark-negative-regime robustness weigh more in final selection; the arctan bound means one bad round can cost the whole stake.)

---

## Build notes (M0 fixer pass, 2026-11-13) — decisions that changed during the build

Appended rather than rewritten; each item names the review finding it resolves and the code that implements it.

1. **Tree count per fold from independent sources (leakage review 1).** The "median inner best-iteration across folds" rule above is kept for the *full-train refit* only. For a scored fold `k` the frozen count is now the median over the folds whose inner-validation eras lie ≥ 20 eras from fold `k`'s test block (`cv.independent_inner_folds`, asserted by `cv.assert_count_sources`). With chronological folds every non-last fold's inner_val sits at the end of the era range, i.e. inside the last fold's and the forward block's test eras, so those two draw only from independent splits (in the test/full layout: their own). The forward block now runs its own inner early stop (`inner_iter_forward`). `spec.json` records `frozen_n_iter_per_fold`, `frozen_n_iter_forward`, `count_sources`; cached search cells written before this rule are not reused.
2. **Suspect baseline (leakage review 2).** `gates.baseline_fit` (optional, a fixed M1 OOF fit20 recorded from the unsearched handoff-02 run) replaces the in-run `lgbm_everest` as the "M1 + 0.004 → suspect" reference whenever it is set; until then the in-run baseline is used and labelled as such. Searched members additionally always evaluate their base `params` cell and write `search/<member>.gain.json` (top-k minus base, on OOF and on the forward block); an OOF gain above the margin flags the member suspect, a negative forward gain is noted as "tie".
3. **Standard errors (leakage review 3).** Every reported `fit_se` is the 20-era block-bootstrap SE (`summarise(...)["se_block"]`, ≈ ci90 half-width / 1.645); the naive `std/sqrt(n_eras)` is gone from logs, results and the regime table `*_se`. Expect ≈ 0.0035–0.004 per config on the test layout, i.e. the plan's noise floor.
4. **NNLS headline (leakage review 4).** When nested NNLS is adopted, the bundle keeps the full-OOF weights, but the reported `chosen` fit20, the SUSPECT/leak checks and the regime simulation use *cross-fitted* weights (fit on folds ≠ k, applied to fold k; `blend.crossfit_nnls`). `blend.json.chosen_is_crossfit` says which.
5. **Gap floor (leakage review 5, 6).** `load_config` rejects `purge + embargo < 20` or `inner_gap < 20`; `RunData` asserts gaps with `max(20, …)`. The tiny-data fallback for the eras<2800 side products never reaches past `run.min_era` and logs a WARNING.
6. **Zero rule wording (leakage review 7).** The 0.4.1 SDK *does* apply the zero rule inside `unq20` (`< 0 → 0`); `metrics.per_era_unq` relies on it and the simulation kernel re-implements the same `< 0` rule. The sentence in "Validation protocol" is corrected above.
7. **Daily path hard stops (contract review 1–3, 6).** `predict` refuses to continue when the live file's `exped` differs from `get_current_round().exped` (no more "warn and submit under the live exped"); a 404 / "no active round" from the SDK maps to exit 3 (window closed). `predict --upload` refuses without a manifest next to the pickle, without `pkl_check.passed`, on a sha mismatch, or when the manifest's `python` is not 3.11.x; `bundle.write_pickle` refuses to run under any interpreter other than 3.11; `pkl_check` passes `--expect-python 3.11` to the checker and refuses a non-3.11 base interpreter for the throwaway venv.
8. **Benchmark frame handling (contract review 4).** A DataFrame handed to `predict` is reduced by `_runtime.select_benchmark_column` (preferred names `eiq_minera_model`/`v1_sherpa`, else the first numeric column) and `last_info["bench_column"]` records the choice — never a blind `iloc[:, 0]`.
9. **Monitor (contract review 5; handoff review 11).** `coverage_pct == 0` is no longer "full"; `None` (field unpublished) still counts as full. `scripts/daily_task.sh` treats monitor exit 2 (nothing configured yet) as "nothing to do, exit 0" so the cron fallback is silent until handoff 05 fills `EIQ_MODEL_ID`/`models/active.json`; whether `handoffs/ALERT.md` is auto-pushed is decided at handoff 05.
10. **Laptop setup (handoff review 1–3).** `selfcheck` reports `EIQ_API_KEY` presence from the environment *or* `.env` (presence only). `scripts/setup.sh` creates the venv with `uv venv --seed` (pip present), backfills pip if missing, and prints `GIT IDENTITY MISSING -> HUMAN STEP` when `user.name`/`user.email` are unset; handoffs 00–02 carry the matching HUMAN STEP, the `GIT COMMIT FAILED — …` wording, the pytest-tail file (no silent re-run of the suite) and the `pkl_check: PASSED` operator-notes token that handoff 02 depends on.

## Build notes (M2–M6 fix pass, 2026-11-14) — protocol changes after the three independent reviews

Appended; details and as-built schemas in `docs/INTERFACES.md` section 22, findings and their regression tests in
`docs/reviews/m2-m6-*.md` ("Fix verification"). These supersede the corresponding statements above.

1. **The forward block is selection data, not confirmation.** It feeds knobs, search adoption, the ladder and the NNLS
   gate, so its sign is reported as a selection estimate. Independent evidence comes from a **reserved audit**
   (`cv.audit`, full: eras 5945–6524, DECISIONS #11): development CV runs on 2800–5924, the development forward block is
   5340–5924, and the deployment full refits stop at 5924 too. After the blend is frozen, `stealth.audit` scores the
   exact bundled blobs once against an unsearched default-universe LightGBM baseline (paired bootstrap, Sharpe,
   cumulative-fit drawdown, four chronological segments, block-length sensitivity). An audit "win" is information,
   never a gate. Handoff 02 has already looked at those eras, so the audit is independent of this run's selection only.
2. **Tree counts (C8 tightened).** A fold's frozen count may only come from folds whose ENTIRE training provenance
   (inner fit, inner validation, feature selection) is >= 20 eras from its test block: in practice each fold's own
   inner split. The forward block also uses the last fold.
3. **Gates.** Every paired gate needs >= 2 whole bootstrap blocks (`inference_valid`), otherwise it fails closed. The
   pre-registered `p_pos >= 0.8` rules are unchanged. Holm for the knobs uses a null-centred one-sided bootstrap
   p-value with the `(count + 1) / (n_boot + 1)` correction. No exact family-wise control is claimed under the
   time-series dependence.
4. **NNLS.** Meta-fit rows are purged >= 20 eras from the evaluation fold and the forward block. The gate and the
   reported forward fit use the same weights; too few purged eras means no NNLS.
5. **Regime simulation = the deployed recipe.** The weighted member blend is neutralised directly (no second
   gaussianisation); missing core-proxy rows are z = 0 after ranking the valid ones; INOV is complete-case like the
   SDK. Parity with `_runtime` + `everestapi.scoring` is tested to 1e-10.
6. **Provenance.** A run fingerprint (config minus threads + data sha256 + protocol version) is checked before
   anything is written; cells, studies and specs carry it. Members are `running` until every artifact is written.
   External (MLP) members must match the run's fingerprint, seeds and epoch cap.
7. **Suspect bar.** The in-run unsearched base cell (+ 0.004); `gates.baseline_fit` only for the same scoring layout
   (full.yaml: null).
8. **Packaging (contract review).** One upload gate for the lane and diagnostics uploads: a pkl_check pass bound to the
   exact bytes, run in a checker that verifies its own pins and isolation. Declared full refits are enforced. Member
   subsets inherit, but never freeze, the neutralisation constants. Numbers must be finite.
9. **Pipeline + report (handoff review).** Atomic `pipeline.json` and per-step logs; signal-safe process groups; the
   MLP runs on CUDA explicitly. Stale external members are refused. The report status comes from the evidence
   (pipeline, pkl_check bound to the bytes, diag-file coverage, audit). Signed URLs are redacted.
10. **Measurement budget for 04 (DECISIONS #12).** 5,000 paired-bootstrap draws and 1,000 simulation draws; search
    trials, the XGB grid and seeds are unchanged.
