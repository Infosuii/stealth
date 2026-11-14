# M2–M6 pickle contract review

Reviewed independently on `worktree-m2-m6`, 2026-11-13. Scope: packaging/build notes in `PLAN.md`, `INTERFACES.md`
sections 9/18 and integration notes, `DECISIONS.md`, runtime, bundle, checker, prediction/diagnostics entry points,
model serializers, MLP export and their tests. No platform submission or upload was performed.

**Checkpoint: paused at the user's request.** Review is complete; no source or test fixes have been started by
this reviewer. The findings below remain open. Only this review document was edited, so there is no partial
implementation to repair before resuming.

## Findings

1. **P1 — diagnostics bypass the pickle gate.** `diag.main` calls `_upload_file` then `upload_diagnostics` without
   checking manifest hash, `pkl_check.passed`, or Python version. Those checks exist only in the optional
   `--preflight` branch. `--attach-pkl` consequently declares Python 3.11 for an unchecked artifact. Share one
   artifact validation helper with `predict.upload`, and test missing/stale manifests, failed checks and a 3.12
   manifest with fake clients; every case must fail before a client write.

2. **P1 — the checker can certify the wrong environment and retain a stale pass.** `ensure_check_venv` trusts
   `.pins.txt`; `--venv-python` bypasses even that. The subprocess checks Python 3.11 and imported modules, but
   never compares installed library versions or checks whether `stealth`/`torch` are importable. Confirmed:
   `run_check(..., python=<training .venv>, eras=0)` returned `passed=True` in the environment containing both.
   `main` also returns early on setup/data errors without clearing an existing pass: after a successful marker,
   a mocked missing-data exception returned exit 1 while the manifest still said `passed=True`. Record the
   checked byte hash and environment, invalidate before each attempt, verify actual pins/isolation in the
   subprocess, and require the hash in both upload paths.

3. **P1 — declared full refits are not enforced.** `bundle.member_blobs` globs `full_s*.bin` and ignores
   `spec.full_blobs`. A missing declared seed/config silently changes the average; all full blobs missing causes
   an unannounced change to fold models. MLP fold blobs contain only the first seed, while its OOF prediction
   averages seeds. Normal writers remove obsolete blobs, but an interrupted retrain can leave an old `done`
   spec with partial new artifacts. Validate the declared full-blob set and refuse incompleteness; keep legacy
   fallback only for specs with no declared full refits. Training should mark the member incomplete before
   replacing artifacts (coordinated with the training reviewer).

4. **P2 — relative checker paths break after the isolation chdir.** `run_check` resolves the pickle, but not
   `python` or `data_dir`. Confirmed: absolute interpreter plus `data_dir=Path('data')` passed the existence check
   then failed in the temporary cwd with `FileNotFoundError: data/futures_live.parquet`. Resolve all input paths
   before creating the subprocess; test the CLI with relative `--data-dir` and `--venv-python`.

5. **P2 — member subsets inherit another blend's frozen neutralisation.** `build_bundle(..., members=[...],
   require_frozen=True)` still marks the original blend's constants as `simulation`, even though member removal
   changes the prediction the simulation evaluated. Confirmed with the M5 fixture: the MLP-only subset retains
   `p_b=.25`, `p_c=.5`, `p_source='simulation'`. Refuse `--require-frozen` if the requested positive-weight set or
   relative weights differ from the selected blend; ordinary subsets should record that their constants were
   inherited and require explicit overrides or a matching simulation before shipping as frozen. Handoff 03's
   explicit zero overrides remain valid.

6. **P2 — fallback fold validation assumes contiguous filenames.** `check_fold_fallback` checks `fold0..N-1`
   from the count rather than the filenames actually loaded. A sparse set such as `fold0.bin, fold2.bin` can
   evade a different `fold2` feature list and feed wrong columns. Validate the exact loaded fold names and
   refuse unknown per-fold metadata.

7. **P2 — numeric contract inputs are not checked for finiteness.** `_clamp` accepts NaN, and positive infinity
   weights are selected before normalization. Infinite weights become NaN; runtime normalization can then
   silently produce constant predictions that pass range checks. Reject non-finite constants/weights and
   negative weights at construction. `predict.validate_predictions` also accepts numeric strings despite its
   documented float-column requirement; align it with the checker.

## Positive evidence and limits

The runtime contains no package imports; MLP serialization is `npz` with `allow_pickle=False`, float32 exported
weights and the specified GELU tanh approximation. The NumPy inference uses float64 and the external trainer
checks exported predictions against evaluation-mode torch within `1e-4`. Members are chosen by name, zero-weight
members are omitted, column bags and linear-model column order are respected, and benchmark alignment/coverage
guards are implemented. Explicit neutralisation overrides cannot pass `--require-frozen`.

Initial targeted run: **57 passed, 2 failed** across runtime, bundle, checker, no-torch and ops tests. One failure
was the clean checker venv still being installed concurrently (NumPy absent); the other was an optional torch
probe from the inherited training environment, with no torch module loaded. The root agent subsequently
reported both environments repaired; this reviewer did not rerun tests before the pause. Existing tests cover
runtime shape/range/index, shuffled columns, benchmark variants,
by-value loading and all supported model input formats. They did not cover the failures above.

## Fix verification

Not started. The next implementer should own `_runtime.py`, `bundle.py`, `pkl_check.py`, `predict.py`, `diag.py`
and the corresponding `test_runtime.py`, `test_bundle_m5.py`, `test_pkl_contract.py`, `test_ops.py` tests. Coordinate
training artifact status changes with the statistics reviewer and diagnostics handoff changes with the handoff
reviewer. Do not weaken the no-torch tests to accommodate the original inherited environment.

Proposed order: (1) actual checker pins/isolation and byte-bound pass records, invalidation before each attempt,
relative paths; (2) common artifact gate for both upload paths; (3) declared refits/exact fold names and subset
neutralisation provenance; (4) finite numeric inputs and float-output validation. Then run the focused suite and
the real checker in `artifacts/pklcheck_venv/bin/python`, including an MLP bundle and 50 validation eras. Root
prepared `.venv/bin/python` for tests and the isolated checker interpreter; no dependency installation is needed
unless subsequent inspection shows drift.

## Fix verification (2026-11-14)

All seven findings are fixed; interfaces are in `docs/INTERFACES.md` section 22.K. Each item names its tests, mostly in
`tests/test_contract_gates.py`.
1. **One upload gate.** `predict.check_upload_artifact` guards every upload path: lane upload, `diag --upload`,
   `--attach-pkl` and `--preflight`. It runs before any client exists. Eight bad states are tested across predict,
   diag and attach, each asserting zero client creations: no manifest, stale sha, failed, running, a pass recorded for
   other bytes, a legacy pass without a hash, built under 3.12, checked under 3.12.
2. **Checker trust and stale passes.**
   - The checker subprocess verifies the installed lane pins and that neither stealth nor torch is importable.
   - It hashes the bytes it checked, before and after the run.
   - `main` writes a non-passing `running` record first, and an `error` record on any exception.
   - All paths are made absolute before the subprocess starts.
   - The training venv now fails.
   - Tests: `test_run_check_refuses_the_training_venv`, `test_run_check_verifies_installed_pins`,
     `test_run_check_resolves_relative_paths` (CLI too), `test_pkl_check_main_never_leaves_a_stale_pass`,
     `test_check_record_is_bound_to_the_checked_bytes`.
   - `monitor` now also requires the pass to be bound to the manifest sha
     (`test_ops.py::test_monitor_refuses_a_pass_bound_to_other_bytes`).
3. **Declared full refits.** `bundle.member_blob_files` uses exactly `spec.full_blobs`. A missing declared refit or
   an extra undeclared one is an error, and the legacy glob applies only when the key is absent
   (`test_declared_full_blobs_are_enforced`).
4. **Relative paths.** Resolved before the chdir and the subprocess (see item 2).
5. **Subset neutralisation.** A new `p_source` value, `inherited`, which `--require-frozen` refuses; explicit
   overrides stay `override` (handoff 03's command still works). Test:
   `test_member_subset_inherits_simulated_constants_unless_it_is_the_selected_blend`.
6. **Sparse fold names.** `check_fold_fallback` validates the exact loaded fold names and refuses unknown metadata
   (`test_fold_fallback_validates_the_exact_loaded_fold_names`).
7. **Finite numbers.** Non-finite or negative weights and constants are refused in `build_bundle` and `make_predict`.
   `validate_predictions` requires a numpy float column, matching the checker. Tests:
   `test_non_finite_constants_and_weights_are_refused`, `test_make_predict_refuses_non_finite_numbers`.

**Real-data evidence on the sandbox**, using the clean lane venv `artifacts/pklcheck_venv` and the minifull run `mf01`:
- The frozen bundle (`--require-frozen`, `p_source simulation`) passed `pkl_check` on 51 eras, bound to its sha256.
- A bundle of the same run that includes the trained MLP (`--members lgbm_everest ridge_everest mlp_everest --p-b 0.5
  --p-c 0.25`) passed on 51 eras with the benchmark neutralisation active. Correlation with the benchmark fell from
  0.648 to 0.405. The byte scan found `torch` 0 times and only `stealth._runtime` references; all 9 pins were verified;
  slowest call 0.69 s.
- Handoff 03's legacy `m1_lgbm` manifest pass has no hash, so it no longer satisfies any upload gate. A future upload
  re-runs `pkl_check` first.
