# M2–M6 selection statistics review

Reviewed `worktree-m2-m6` independently on 2026-11-13, before the fix pass. No external submissions or uploads. References below identify the reviewed code; subsequent fixes may move the lines.

1. **P1 — shared early stopping still uses scored labels.** `cv.py:204–227` checks only a source's inner validation eras; `train.py:713–735` pools its best iteration even when its inner training set includes the destination test labels. On the full 2800–6524 five-fold layout, fold0 accepts fold1–fold4 as count sources; their inner training sets contain 705, 745, 745, and 745 of fold0's 745 test eras. Changing fold0 labels can therefore change fold0's count and prediction. The MLP inherits these sources (`scripts/mlp_train.py:514–522`). Require the source's complete training/feature-selection and validation provenance to exclude the destination test block plus purge; own-fold stopping is the safe default. Regression: inspect both source sets and perturb scored labels while asserting unchanged fit decisions.

2. **P1 — cached artifacts have insufficient provenance and can be partly overwritten while marked done.** Finished trees are reused on universe equality alone (`train.py:883–897`); knob identity omits member params, folds, data, and seed (`knobs.py:224–241`); protocol hashes omit everything except params/universe (`train.py:794–834`); Optuna study identity is only the member name (`train.py:1036–1037`). `train_run` overwrites the recorded config before validating reuse (`1390–1394`). Thus changed folds/params/data may publish old OOF and models as a new run. During retraining the old done spec survives deletion/replacement of blobs (`1106–1140`). External members are also accepted without checking their training signature (`1205–1239`). Freeze a run/protocol fingerprint before mutation, validate artifact identity/completeness, namespace studies, and mark running before replacing blobs. Regression: resume after each relevant change; interrupted refits must never expose a done model.

3. **P1 — simulation and deployed neutralisation disagree.** `_regime_prepare` re-Gaussianises the weighted member blend (`metrics.py:529`), while runtime projects that blend directly (`_runtime.py:538–555`). It also fills missing raw core values with zero before ranking (`metrics.py:522`), while runtime ranks valid core values and maps missing values to zero afterwards. Seed 742, 60 eras × 90 rows, a two-member blend, `p_b=0,p_c=.5`: 4,136/5,400 ranks differ even without missing core values. With 20% missing core rows, simulation FIT is 0.231728 versus runtime 0.228832, a +0.002896 discrepancy; missing core z is −1.28155 in simulation versus 0 at runtime. Preserve the weighted blend and reproduce runtime missing-value semantics; test complete postprocessing parity, not only projection algebra or the `(0,0)` identity.

4. **P1 — forward is selection data, not independent confirmation.** All 580 forward test eras are also in the search objective's OOF universe (`train.py:508–514,1030–1051`). Knobs select on the forward scores (`knobs.py:268–286`), then search adoption, successive admission arms, and NNLS reuse the same labels. The supposed independent CV confirmation excludes overlapping test sets but its models train on forward labels. Moreover, top-k is selected and bootstrapped on the same OOF scores. These estimates are conditional on selection and optimistic; fixed order and bootstrapping do not repair the reuse. A new untouched audit partition, excluded from every development fit and adaptive decision, is needed for an independent performance claim. Until then label the figures as development/selection estimates and do not describe the forward sign as independent confirmation. This is a protocol decision, not a local numerical patch.

5. **P1 — NNLS fitting is not purged and can directly consume forward labels.** `nested_nnls` uses every row with `fold < last` (`blend.py:260–271`); `crossfit_nnls` similarly uses every other fold (`288–292`). Adjacent target windows therefore cross the meta-fit/evaluation boundary without the required 20-era gap. On minifull, the nested gate includes 72 forward test eras (5945–6016) in its weight fit. The reporting path later fits different forward weights (`698–706`), so adoption and the reported forward score can even evaluate different rules. Purge meta-fit rows against evaluation eras, use the same forward-safe weights for gate/report, and fail closed when none exist. Full independence also requires nesting the base model fits; purging meta rows alone does not solve finding 4.

6. **P2 — short-series bootstrap can manufacture certainty; Holm inputs are heuristic.** When `n_eras <= block`, the circular bootstrap uses the entire series in every draw (`cv.py:267–273`). For `[-1,1.1]` with block 20 it returns SE 0, `p_pos=1`, CI `[.05,.05]`. Such evidence can pass search/knob gates despite being only one effective block. Refuse adoption with insufficient blocks. Holm arithmetic itself implements step-down correctly, but `1-p_pos` from an uncentered percentile bootstrap is not a generally calibrated one-sided null p-value (`knobs.py:171–189`), and zero Monte Carlo tails are treated as exact zero. Use a separately named, null-centered one-sided block p-value with finite-draw correction for Holm; retain `p_pos` as a descriptive stability statistic. Do not claim exact family-wise control under unverified time-series assumptions.

Other checks: MLP standardisation and target scaling use fit rows only; target-dependent feature subsets exclude the outer test rows. The feature subset is nevertheless selected using inner validation labels before early stopping, so independent-source checks should conservatively cover the complete source fold training set. The simulation uses common draws across neutralisation cells, and the paired Sharpe bootstrap uses common era indices. Existing tests confirm those mechanics but miss the provenance and complete runtime-parity defects above.

Fix pass requested by the coordinating agent after this independent review; implementation and regression results will be appended below.

## Paused checkpoint — partial implementation, not validated

The user requested a pause during the fix pass. Stop point: one atomic code patch completed; no tests have run after it. This report and the changes below are WIP and must not be treated as ready to merge/deploy.

Completed code edits:

- `cv.independent_inner_folds` / `assert_count_sources` now check the entire source `train_eras`, covering inner fit, inner validation, and target-dependent feature selection. Existing tests that expect early folds to share all other folds' counts now need correction.
- `PairedResult` gains `p_value`, `n_blocks`, and `inference_valid`. The p-value is the null-centered one-sided bootstrap tail with a `(count + 1)/(n_boot + 1)` correction. Fewer than two effective blocks returns `p_value=1` and `p_pos=.5`; `block_bootstrap_means` rejects fewer than two draws. Holm has **not yet** been changed to consume `p_value` or the validity flag.
- `metrics._regime_prepare` preserves the weighted blend before projection, and ranks valid core values before assigning missing core rows z=0. `paired_sharpe_bootstrap` gains the block-validity fields and returns `p_pos=.5` for fewer than two blocks.

Unimplemented: every provenance/cache/status fix; search study fingerprinting; external-member validation; CLI `--skip-external`; NNLS purge and consistent forward gate/report weights; Holm wiring and fail-closed adoption; audit partition/prediction generation; all new regressions and changed-test expectations. No `train.py`, `knobs.py`, `search.py`, `blend.py`, or `scripts/mlp_train.py` edits were made by this reviewer before the pause.

Remaining simulation edge case to inspect: although missing-core prediction semantics are now fixed, INOV scoring with missing core rows should match the SDK's complete-case ranking/centering. The simulator's `unq_rows` currently treats missing core z=0 as present; if the SDK drops those rows, use separate core-valid masks, rerank the prediction on valid rows, and recenter targets for proxy INOV. This is not yet patched or validated.

Agreed audit interface with the coordinating agent (not implemented here):

- Optional `cv.audit: {enabled: true, train_end: 5925, test_start: 5945}`. Development CV and development forward use only eras `<5925`; proposed development forward is `train_end:5300,test_start:5340`.
- `RunData` keeps a separate `rd.audit`; all CV/forward/count sources, knobs, search, and admission exclude audit labels. For full04, full deployment refits also use only pre-audit eras. A later M7 may refit on all data after this evaluation.
- Generate audit predictions from the **same completed full-refit blobs**, averaged over selected params/seeds. Write `audit_<member>.parquet` with `id` index, `era`, `pred`; assemble `audit.parquet` with `era`, `target_everest`, `m__<member>` columns, and `baseline__lgbm_everest` from the unsearched default-universe baseline refit.
- `folds_eras.json` gains a separate audit entry for the MLP, excluded from its early stopping; it predicts that entry from its full-refit blobs. Main train and MLP need matching data/config/protocol fingerprints.
- Root owns standalone `stealth.audit` after blend freezing, evaluating the frozen weights against `audit.parquet`, reporting paired baseline/drawdown/stability, and never tuning on audit. Audit is reserved from **this run's** selection; historical handoff 02 already viewed these eras.
- Historical baseline `.0229` becomes context only because the development layout changes; use an in-run comparable baseline for suspect checks. Root owns full.yaml and handoff/documentation updates.

Coordination promises still pending: handoff reviewer expects `train_run(..., skip_external=False)` and CLI `--skip-external` for initial tree-only/fallback pipeline runs. Contract reviewer owns runtime/bundle and will require declared `full_blobs`; training still needs to mark a member running before blob replacement.

Next steps: inspect the WIP diff; implement partition/provenance/status first; wire the statistical changes and NNLS purge; add complete runtime-parity tests; update the count-source tests; run focused tests, then the integrated suite/pipeline. Reproduce the numerical findings above against the fixed implementation. Do not reuse pre-fix artifacts as validated evidence.

## Fix verification (2026-11-14)

Interfaces: `docs/INTERFACES.md` section 22. Every finding is fixed; each line names the regression test(s).

1. **Count sources.** `cv.independent_inner_folds` / `assert_count_sources` check the source's whole `train_eras`. Every
   CV fold now uses only its own inner split; the dev forward block uses its own plus fold4 (training ends >= 20 eras
   before it). Tests: `test_cv.py::test_independent_inner_folds_test_layout` (each other fold is shown to train on the
   target's test eras), `::test_independent_inner_folds_full_audit_layout`, `::test_independent_inner_folds_smoke_layout`,
   and the requested label perturbation `test_train_m2.py::test_fold_count_does_not_depend_on_its_own_test_labels`
   (fold0's test labels permuted: its count and its predictions are bit-identical).
2. **Provenance.** Run fingerprint (config + data + protocol) checked before ANY write; cell ids, Optuna study names
   and member specs carry it; `status: running` before any blob is replaced, `done` last; external members validated
   (fingerprint, universe, seeds, epoch cap, declared blobs, OOF + audit files); `--skip-external` /
   `--require-external`; blend refuses stale / incomplete members. Tests: `test_train_m2.py::
   test_redecision_refused_in_run` (metrics.json, config.yaml, knobs.json byte-identical after the refusal; threads-only
   change keeps the fingerprint), `::test_legacy_run_dir_without_fingerprint_refused`, `::test_cell_cache_identity`,
   `::test_external_member_provenance_checks`, `::test_skip_and_require_external`,
   `::test_interrupted_retrain_never_exposes_a_done_member` (KeyboardInterrupt inside the refit leaves "running"; resume
   retrains to "done").
3. **Simulation vs runtime.** `metrics.cell_scores` (no re-gaussianisation, runtime missing-core semantics, SDK
   complete-case INOV via `inov_rows`) is the grid's per-cell kernel. `test_metrics.py::
   test_cell_scores_match_runtime_and_sdk[4 cells]` matches `_runtime` + `everestapi.scoring` per era to 1e-10 with 20%
   missing-core rows; `::test_regime_grid_uses_the_parity_kernel`. On that fixture the pre-fix semantics overstated
   FIT by +0.0022 and INOV by +0.0038 (per-era errors up to 0.016 / 0.044); the fixed kernel differs by < 1e-16.
4. **Forward reuse.** Protocol change, not a patch: `cv.audit {train_end 5925, test_start 5945}` reserves 580 eras from
   every choice and fit of the run (development CV 2800..5924, dev forward 5340..5924, full refits < 5925); the frozen
   blend is scored once by `stealth.audit`. Development figures are labelled selection estimates in metrics.json,
   blend.json and the report. Tests: `test_audit.py::test_partition_boundaries_and_audit_file`,
   `::test_audit_predictions_are_the_bundled_blobs` (rank-for-rank equal to the pickle at p = 0),
   `::test_audit_labels_never_reach_a_decision` (audit targets permuted: knobs, search gain, ladder, weights, p_b/p_c,
   oof.parquet and every audit prediction identical), `::test_stealth_audit_scores_the_frozen_blend_once`.
   The audit is independent of THIS run's selection only: handoff 02 already looked at those eras.
5. **NNLS purge.** `purged_fit_mask`; nested fit rows >= 20 eras from the eval fold AND the forward block; the gate's
   forward diff and the reported forward fit use the same nested weights; < 40 purged eras -> no NNLS; cross-fit rows
   purged per fold. Tests: `test_blend_select.py::test_nested_nnls_meta_fit_is_purged`,
   `::test_blend_run_m1_config_keeps_old_behaviour` (forward weights == gate weights).
6. **Short series / Holm.** `inference_valid` (>= 2 whole blocks) with neutral p_pos 0.5 / p_value 1 otherwise; every
   gate fails closed; Holm on the null-centred `p_value_fwd`; p_pos kept as a descriptive statistic; no exact FWER
   claim. Tests: `test_cv.py::test_paired_bootstrap_short_series_is_not_certain` (the `[-1, 1.1]` case),
   `::test_paired_bootstrap_p_value_is_null_centred` (~uniform under the null), `test_knobs.py::
   test_holm_uses_null_centred_p_value_and_fails_closed`, `test_blend_select.py::test_admit_decision_fails_closed_without_inference`,
   `test_metrics.py::test_paired_sharpe_bootstrap_sign_and_p_pos` (one-block case).

Also: the SUSPECT reference of a searched in-run baseline is its unsearched base cell (`train.baseline_fit`), and
`full.yaml` sets `gates.baseline_fit: null` (0.0229 was scored on another layout). Integration evidence (full suite,
real-data minifull with the CPU MLP, frozen bundle in the clean lane venv) is recorded in `docs/NEXT-SESSION.md`.
