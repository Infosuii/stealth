# M2–M6 handoff executability review — paused checkpoint

Status: **unfinished; do not treat handoff 04 as reviewed or ready**. The user paused work during the fix pass.
No upload, submission, stake, or platform mutation was performed.

Reviewed the session/plan/decisions documents, handoff index and 04, full config, pipeline/GPU/MLP scripts,
reporter, related tests, and handoff 02 results. Laptop assumptions: Linux, Europe/London, 20 CPU threads
(16 used), RTX 5070 Ti Laptop GPU, 30.8 GB RAM. Handoff 02 measured 497 s training and 208 s regime simulation.

## Findings

1. **High: failed pipelines can report success.** `stealth/report.py` takes status from train/blend
   `metrics.json` and only reads `run.log`. A later bundle, pickle-check, or diagnostics failure leaves status
   `ok`; an exception in blend can do the same. The reporter omits pipeline outcomes, MLP errors, pickle-check
   evidence, diagnostics coverage, and actual pipeline wall time. Retraining overwrites the original training
   timing. Fix: consume authoritative pipeline state, capture failing-step output, expose checked bundle and
   diagnostics metadata, distinguish incomplete/interrupted/failed attempts, and retain timing across retries.
2. **High: optional MLP handling can hide interruptions or reuse an excluded member.** The original pipeline
   tolerated every nonzero MLP exit, including signal exits. `--no-mlp` only skipped its trainer: initial
   `train --resume` could still load a previously completed MLP. Fix: stop on interrupts; explicitly exclude
   external members until their training/provenance checks succeed; cover suffix resumes as well as fresh runs.
3. **Medium: GPU run can unexpectedly become CPU training.** Handoff 04 did not force CUDA, while the member's
   `device: auto` falls back to CPU if CUDA stops working after setup. Use `--device cuda` for the laptop GPU
   path; keep explicit CPU mode for sandbox tests. The setup smoke checks device capability and actual CUDA
   computation, but has not been tested on this laptop in this review.
4. **Medium: report redaction misses signed download URLs.** The current secret patterns do not redact query
   signatures/tokens in presigned URLs. Root observed HTTPX INFO output containing these URLs and owns
   suppression in `stealth/data.py`. Reporter redaction and synthetic tests are still required before publishing
   captured logs. Preserve the existing public-results rule that hyperparameter values stay out of reports;
   captured search logs also need review before inclusion.
5. **Handoff corrections required.** SUSPECT threshold contradicts itself: full config/checklist gives
   `0.0229 + 0.004 = 0.0269`, success criteria says `0.0278`. Both mutually exclusive pipeline commands appear
   in one executable block. Selfcheck/pytest piped to `tail` hide their exit codes without pipefail. Failure
   reporting must not unconditionally commit missing manifests or write success-shaped operator notes.
   Preserve preexisting config edits when restoring threads. Reconcile the stated 1–2 h, 1–2.5 h, and
   config/plan 2.5–3.5 h estimates after measurement; the old small sandbox run does not establish GPU timings.

The original pipeline's diagnostics command builds a local file and does not pass upload/submit flags.
The handoff's no-upload scope is explicit and should remain unchanged.

## Changes already applied when paused

Only `scripts/run_pipeline.sh` was edited during this fix pass. The unverified patch:

- Adds atomic `pipeline.json` attempt/step outcomes and timings, plus per-step stdout/stderr logs under
  `pipeline_steps/`.
- Adds process-group execution with `setsid`, output capture with `tee`, and INT/TERM/EXIT handling.
- Stops on unexpected or signal MLP exits; retains optional skips for exits 1 and 4.
- Defaults the MLP subprocess to CUDA, with `--mlp-device cpu` retained for explicit sandbox runs.
- Passes `--skip-external` to the first train and to `--from retrain --no-mlp`.
- Adds optional `audit` after blend when `cv.audit.enabled`, invoking `python -m stealth.audit --run-id R`.

**This patch has not been syntax-checked or tested.** Reporter, GPU setup script, and tests remain unchanged.
The statistics reviewer confirmed that train CLI `--skip-external`, `train_run(skip_external=False)`, and
external artifact provenance checks **were not implemented before pause**. The modified pipeline's first train
command therefore currently fails argument parsing. Implement this interface before running the pipeline.

Known follow-ups in the partial pipeline implementation:

- Starting at `--from mlp` then skipping/failing the MLP still needs a retrain excluding external members if
  OOF was previously assembled with an MLP.
- `--no-mlp --from blend` or a later stage needs a consistency check against already assembled members.
- `--report` executes before the final success state. When report state ingestion is implemented, arrange
  completion recording so a successful report is not incorrectly marked incomplete.
- Test output capture and signal forwarding, including subprocess termination and repeated attempts; ensure
  cleanup never kills processes outside the step's dedicated process group.

Root owns handoff 04, full config, documentation finalization, and a new audit module. The proposed audit
reserves eras 5945–6524, moves development selection earlier, and scores frozen blend weights once. Its JSON
public whitelist and reporter integration were not settled at pause. Root also owns HTTPX log suppression.

## Verification and continuation

Before edits, `.venv/bin/python -m pytest -q tests/test_pipeline_script.py tests/test_report_selfcheck.py`
passed **19 tests in 6.27 s**. These are baseline results only; none verifies the partial patch.

After completing the fixes, add meaningful coverage for late-stage failure reporting, missing terminal state,
failure followed by successful retry, MLP exit 130/143, direct INT/TERM, stale external-member exclusion,
optional audit ordering, captured stderr, signed URL scrubbing, and `--report` completion ordering. Then run
the targeted tests and the coordinated minifull pipeline before declaring handoff 04 ready.

## Fix verification (2026-11-14)

The code findings are fixed (`docs/INTERFACES.md` section 22.L). The tests named below are in
`tests/test_pipeline_states.py`, `tests/test_report_m7.py`, `tests/test_pipeline_script.py` and `tests/test_hook.py`.

1. **Failed pipelines reported success.** The report status now comes from `pipeline.json`, metrics and the leak
   check, the manifest's pkl_check (bound to the bytes), diag-file coverage and `audit.json`. Failed evidence gives
   `failed`; missing evidence gives `incomplete`. The failing step's log tail is included (scrubbed). Timings come from
   every attempt, so the first train's time survives a retrain. A finished pipeline records its compute result before
   the report step; the report's own outcome goes in `report_step`. Tests:
   - `test_all_evidence_ok_and_sections`, `test_failed_pipeline_step_tail_is_scrubbed`,
     `test_unfinished_pipeline_is_never_ok`;
   - `test_pkl_check_record_must_name_these_bytes`, `test_missing_files_behind_ok_steps_make_it_incomplete`,
     `test_retry_keeps_every_attempts_step_seconds_and_mlp_tail`;
   - `test_report_observes_a_finished_pipeline_and_its_failure_is_separate`,
     `test_failure_then_retry_two_attempts_and_stderr_in_step_log`,
     `test_killed_driver_leaves_running_attempt_marked_interrupted_next_time`.
2. **MLP interruptions and stale members.** MLP exits 1 and 4 are a tolerated skip; exits 130, 143 and 137 mean
   interrupted; any other code fails the pipeline. The first `train` runs with `--skip-external`. `retrain` uses
   `--require-external` after a finished MLP and `--skip-external` otherwise. `--no-mlp --from blend` refuses (exit
   64) while `oof.parquet` still holds the MLP. Independently, `blend.load_run` refuses stale external members. Tests:
   `test_mlp_signal_exit_stops_the_pipeline`, `test_retrain_rules_for_external_members`,
   `test_no_mlp_from_blend_refuses_a_stale_external_member`, `test_dry_run_audit_between_blend_and_bundle`.
3. **GPU silently becoming CPU training.** The MLP defaults to `--device cuda` (no silent CPU fallback). `setup_gpu.sh`
   reinstalls a CPU-only torch from the cu128 index (`test_setup_gpu_script_usage_and_cpu_build_reinstall`).
4. **Redaction.** Signed-URL queries (X-Amz-*, X-Goog-*, signature/token/expires/credential) are redacted; log tails
   lose hyperparameter `name=value` pairs and home paths; the pre-commit hook blocks signed URLs. Tests:
   `test_scrub_signed_urls_and_is_idempotent`, `test_scrub_params_drops_hyperparameter_values`,
   `test_log_tail_hides_home_directories`, new `test_hook.py` cases.
5. **Handoff text.** Handoff 04 was rewritten:
   - **Commands:** one GPU branch per block (4a / 4b); `pipefail` and exit codes captured from files; the pipeline
     runs under `nohup` with `pipeline.json` polling, so a tool timeout cannot kill it; the only edit is restoring
     `run.threads` itself.
   - **Expectations:** time estimates scaled from handoff 02; technical success and the audit verdict are separate;
     the SUSPECT bar is the in-run unsearched base cell.
   - **Failure safety:** the manifest is committed only after a pass bound to its bytes; operator notes record what
     happened, never success-shaped placeholders.

**Real-data evidence on the sandbox** (2 cores, minifull with an external CPU-torch MLP):
- Pipeline `mf01` ran OK end to end in 154 s: train 100, mlp 10, retrain 14, blend 16, audit 3, bundle 0,
  pkl_check 3, diag 8 s, then report. Report status `ok`, with every evidence item bound to real files.
- A same-command resume reused every cache and finished OK.
- `--no-mlp --from blend` refused with its instruction.
- Run `mf02` received TERM on the driver mid-train. The attempt was recorded `interrupted` (143), no process
  survived, and the in-flight member was left `running`, never `done`.
- The same-command re-run finished OK with both attempts recorded. Its blend decisions and audit numbers are
  identical to the uninterrupted `mf01`.
