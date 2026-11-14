# Bot follow-up — contract, statistics and execution review

Primary-agent follow-up on 2026-11-14 after the completed laptop handoff 04. This is a fresh solo code review and
regression verification, **not** a claim that the previously stopped independent reviewer agents reported.

## Model decision

Keep the pre-registered unsearched LightGBM comparator as the conservative deployment candidate. Do not promote
full01 just because it is larger or technically complete. Its frozen blend lost on the reserved audit: FIT
0.0265698 versus 0.0307761, difference -0.0042063, paired p_pos 0.0412; Sharpe 0.5571 versus 0.6958.

This comparator is `artifacts/full01/baseline/`, default universe `e2800-none-d0`. It is **not**
`artifacts/full01/members/lgbm_everest/`, which uses the knob-selected `e3-none-d0` universe, and it is not the old
top-three test01 model used for diagnostics #1. The explicit `bundle --audit-baseline` path preserves those exact
existing pre-audit refits with neutralisation off. It makes no new fit, search or selection and records its source
fingerprint/universe in the manifest. It refuses incomplete, stale or empty declared refits and cannot inherit the
challenger's simulated constants.

The retained comparator was specified before the audit. Its audit score has nevertheless been seen; reusing these
eras for new tuning would no longer be independent. Neither audit candidate has demonstrated future live returns.

## Outstanding contract findings addressed

1. A checker boolean or legacy partial record could mint/accept a pass. Passing evidence now requires status
   `passed`, the checked byte hash, Python 3.11, all nine exact runtime pins, both import-isolation flags false,
   no problems, at least one checked era and finite timing below the lane limit. A boolean without a successful
   complete checker report cannot create a pass. Uploads, monitoring and technical reporting share these checks.
2. The SDK re-read mutable original paths after the artifact gate. Lane and diagnostics uploads now hand it private
   snapshots whose hashes are verified before any write. The diagnostics record hashes the files actually sent;
   an attached pickle must match the source predictions. A lane upload does not overwrite a newer local manifest.
3. Explicit `full_blobs: []`/null could fall back to legacy blobs. These now fail; legacy behavior applies only
   when the field is absent.
4. A bare `--venv-python python` was turned into a nonexistent local absolute filename. PATH lookup now works
   without resolving away the venv's interpreter symlink.
5. Checker setup could delete an unrelated directory. It now refuses symlinks and directories without checker
   ownership evidence. Newly created incomplete environments are marked as owned, so a failed setup can resume.

The platform's authenticated Python 3.11 lane-pins endpoint was read without exposing credentials: all nine
runtime pins exactly matched the local lock. The public unauthenticated URL returned 401; the authenticated
endpoint serves requirements text rather than JSON. No package changes were needed.

## Statistics verification

- Reviewed full training/count/feature-selection provenance and the own-fold count-source checks.
- Reviewed separation of development eras, dev forward rows and the reserved audit, including full refit limits.
- Ran the audit-label perturbation regression: changing reserved targets leaves every development decision,
  prediction and frozen constant unchanged.
- Reviewed purged NNLS meta-fit/evaluation rows and consistent forward gate/report weights. Cross-fitted meta
  weights do not remove reuse of labels for upstream selection; the development results remain optimistic
  selection estimates, not nested-CV evidence for the entire adaptive procedure.
- Reviewed the short-series fail-closed checks, finite-draw null-centred Holm inputs and the distinction between
  descriptive p_pos and a null p-value. No exact finite-sample time-series family-wise guarantee is claimed.
- Reviewed and regression-tested runtime/simulation parity, including missing-core semantics.
- Completed audits are now reused without rescoring their labels. Different frozen-model or recorded audit-input
  hashes are refused, and a failed later attempt cannot erase the original completed evaluation.

Full01's earlier audit predates the audit-input hash field; its recorded model hash and fingerprint are preserved
and bind the cached result, but that legacy record cannot retrospectively prove the original audit-input hash.

## Execution verification and limits

The real laptop 04 evidence is retained without rerunning or overwriting it: GPU rehearsal and stale-MLP refusal,
TERM exit 143 with no stray processes, resume `[interrupted, ok]`, full pipeline exit 0 in 47m21s, MLP trained on
CUDA in 73s and rejected by the ladder, byte-bound clean-venv checker, and a complete local diagnostics file.
Regression coverage also exercises process-group signals, late-stage failures, optional-MLP exclusion, retries,
report status and signed-URL redaction. The full integration suite uses fake clients for every platform write.

No larger search, extra seed bagging, post-audit parameter tuning, full-data M7 refit, platform upload, submission,
model-slot change or stake is part of this follow-up. Diagnostics #2 is optional calibration under handoff 05;
it does not automatically promote the challenger. A fresh assessment requires an agreed prospective protocol.

Verification counts and the exact exported baseline's packaging evidence are recorded after execution below.
