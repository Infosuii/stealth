# Next session — PAUSED review checkpoint (2026-11-13)

**The user explicitly paused this session and requested a checkpoint for another agent. The working branch is
`worktree-m2-m6`. This checkpoint is incomplete, has NOT passed the fix-pass tests, and must NOT be merged into
`main` or used to run handoff 04 yet.** The original task is still to finish the three reviews/fixes, finalise 04
for Linux / 20 logical threads / RTX 5070 Ti, verify it, publish to main, and provide the laptop GPT prompt.

Read this section first, then `docs/PLAN.md`, `docs/DECISIONS.md`, `handoffs/INDEX.md`, and the three reports in
`docs/reviews/`. The older session notes below are history; this checkpoint takes precedence where they differ.

## Repository and publication state

- This session started with an empty `/home/climber/work`; the repository was cloned to
  `/home/climber/work/stealth` from `https://github.com/Infosuii/stealth.git`.
- At the last remote read, **both** `origin/main` and `origin/worktree-m2-m6` were
  `0d185c878472fbc36aecdf502b39053359a7a109`. The M2–M6 implementation and accepted handoff 03 were already on main.
- GitHub reads work, but pushes fail: `fatal: could not read Username for 'https://github.com': No such device
  or address`. There is no `gh`, configured credential helper, GitHub token environment variable, or SSH agent
  available in this workspace. This is missing authentication, not missing user authorization or an approval
  rejection. Do not claim this checkpoint reached GitHub unless a later push succeeds.
- A local WIP commit and a portable git bundle are prepared when pausing. The bundle is
  `/home/climber/work/stealth-review-checkpoint.bundle`; it contains branch `worktree-m2-m6`. In a fresh clone,
  fetch that branch from the supplied bundle and fast-forward the matching local branch before continuing.
  Once credentials are available, publish the checkpoint branch normally; do not force-push.
- User already authorized getting the completed, tested/reviewed work onto main. Do not ask for that approval
  again. The current checkpoint is not complete enough to merge.

## Three independent reviews: completed, fixes only partly started

The user explicitly authorized independent agent reviews. Three reviewers ran in parallel and all have stopped.
Each report contains concrete evidence, priorities, and its pause status:

1. `docs/reviews/m2-m6-statistics.md`: six findings covering early-stopping provenance, stale caches,
   simulation/runtime mismatch, reused selection/forward data, unpurged NNLS, and bootstrap/Holm validity.
2. `docs/reviews/m2-m6-contract.md`: seven findings covering diagnostics upload bypasses, checker environment
   and stale pass markers, declared full blobs, relative paths, subset neutralisation provenance, sparse fold
   filenames, and non-finite numeric inputs. **No contract source/test fixes have started.**
3. `docs/reviews/m2-m6-handoff.md`: executability, truthful pipeline status, failed-step logs, GPU fallback,
   stale MLP inclusion, interruption handling, and instructions/reporting inconsistencies.

Changes present in the checkpoint:

- `stealth/cv.py`: early-stopping count sources now check their entire training-era provenance; paired bootstrap
  exposes a null-centred finite-draw `p_value`, `n_blocks`, and `inference_valid`; inadequate blocks give neutral
  `p_pos`. **Callers/tests still need updating**, notably Holm still uses `1-p_pos`.
- `stealth/metrics.py`: simulation no longer re-Gaussianises the weighted member blend; missing core values
  are ranked before zero filling; Sharpe bootstrap gains inadequate-block flags. Full SDK/runtime parity
  regressions and missing-core INOV verification remain to be written/run.
- `stealth/data.py`: `make_client` suppresses HTTP INFO/DEBUG logs that otherwise print signed download URLs.
- `scripts/run_pipeline.sh`: an **untested partial patch** adds atomic `pipeline.json` attempts, per-step logs,
  process-group/signal handling, explicit CUDA by default, and an optional audit stage. It now calls
  `train --skip-external`, but that CLI option **does not exist yet**. It can call `stealth.audit` when enabled,
  but that module **does not exist yet**. Do not run this driver until those interfaces are completed.
- `report.py`, `setup_gpu.sh`, all contract sources, full/minifull configs, laptop runbooks and kickoff prompt
  are unchanged. Handoff 04 remains `draft`.

## Agreed statistical repair to implement next

The old forward block is part of CV search and is repeatedly reused for knobs/search/admission/NNLS. Its sign
cannot be called independent confirmation. Preserve the development gates but label their numbers as selection
estimates, and add an audit reserved from **this run's** selection (historical handoff 02 has seen those eras).

Agreed interface, **not yet implemented**:

- Add `cv.audit: {enabled: true, train_end: 5925, test_start: 5945}` to `full.yaml`.
- Development CV, development forward, knobs, search, admission, NNLS, count sources, and full-refit training
  use only eras strictly before 5925. Move development `cv.forward` to train_end 5300 / test_start 5340.
- Keep audit rows 5945–6524 out of every adaptive choice. For 04, deployment full refits also use pre-audit
  eras only, so their identical blobs can produce honest audit predictions. A later M7 full-data refit is a
  separate decision/work item, not something to silently do after viewing this audit.
- `RunData` gains separate `audit`; completed members write `audit_<member>.parquet` (`id` index, `era`, `pred`)
  from the exact averaged full-refit blobs. Assemble `audit.parquet` containing `era`, `target_everest`,
  `m__<member>` and `baseline__lgbm_everest` (unsearched/default-universe baseline, fitted before audit).
- MLP training gets separate audit metadata in `folds_eras.json` and predicts it from the same exported full
  blobs. Audit is not an early-stopping fold and must never provide count sources or model selection scores.
- New `stealth.audit --run-id R` runs **after frozen blend selection**, reads `audit.parquet` and fixed weights,
  and writes `audit.json`. Report unneutralised audit FIT/Sharpe, paired baseline difference, cumulative-FIT
  drawdown (clearly labelled, not platform P&L), chronological stability, and bootstrap block sensitivity.
  Optional scenario metrics may evaluate only the frozen constants. It must not admit/tune/replace anything.
- Set `full.yaml` historical `gates.baseline_fit` to null because 0.0229 used a different scoring layout;
  retain the old baseline as context, not a directly comparable pass/fail threshold.
- Planned measurement budget: 5,000 bootstrap draws and 1,000 simulation draws, unchanged ≤25 LGBM trials,
  unchanged fixed XGB grid, no added seed bagging. These config edits have NOT been made.
- Add audit coverage to minifull/integration tests and public report fields. Agree the `audit.json` schema
  before wiring `report.py` (no schema was finalized before the pause).

Other required fixes: run/data/config/protocol fingerprints checked before mutation, no stale done specs during
refits, external-member signature validation, `--skip-external`, purged and consistent NNLS gate/report weights,
all contract findings, truthful report status from pipeline/manifest/diag evidence, and final runbook corrections.
The handoff reviewer also identified remaining `--from mlp` / `--no-mlp --from blend` stale-member cases and the
need to make `--report` observe a finished pipeline rather than its own still-running state.

## Environment and verification actually performed

- Six fresh futures parquet files are in ignored `data/`; no hosted training, model creation, upload, live
  submission, auto-submit change, or staking occurred. Handoff 03 remains the only diagnostics upload.
- `.venv/bin/python` is Python 3.11.2 with exact project requirements. Its ignored
  `site-packages/ml_environment.pth` points at the isolated check venv's pinned libraries; pytest/Optuna/SDK
  extras are in `.venv`. **Torch is not importable** there. This is a local convenience, not a laptop setup change.
- `artifacts/pklcheck_venv/bin/python` was created with the exact lane packages, without stealth or torch.
- `/opt/ml/bin/python3.11` has CPU torch for the MLP smoke. Use it only for the external MLP, with repo on
  `PYTHONPATH`; it is not a clean pickle checker or train environment.
- An initial full pytest run against the wrongly inherited ML environment: 249 passed, 1 skipped, 2 failures
  (both caused by importable torch). After correcting the environment, `tests/test_no_torch.py` and
  `tests/test_skeleton.py` passed: **35 passed**.
- Contract reviewer initial targeted suite: 57 passed, 2 environment failures (checker installation race and
  torch). Handoff reviewer initial targeted suite: 19 passed. These all precede the new source patches.
- **No post-patch suite or end-to-end minifull run has been executed.** Count-source tests need updating;
  the partial driver is expected to fail until its missing interfaces are implemented. `git diff --check`
  and shell syntax were checked during checkpoint preparation; they do not establish functional correctness.
- Git hook is enabled with `core.hooksPath=scripts/hooks`. Never commit `.env`, `data/`, `artifacts/`, `.venv*`,
  pickles, or credential-bearing logs. Use a new integration run ID after protocol changes.

## Next completion sequence

1. Complete the fixes in all three review reports plus the agreed audit interface and regression tests.
2. Run the full suite, real-data minifull including CPU MLP, frozen bundle and isolated checker; exercise
   failure/resume/report paths. Do not mark ready from syntax checks or old test results.
3. Finalise handoff 04 and `handoffs/GPT-KICKOFF.md` for the laptop: Linux, 20 logical threads (keep 16 worker
   threads), RTX 5070 Ti, ~30.8 GB RAM, Europe/London. Handoff 02 took 12.5 minutes (training 497 s, simulation
   208 s). Calibrate estimates, keep GPU optional/explicit, and distinguish technical completion from an audit win.
4. Mark 04 ready only after tests and reviews pass; update INDEX, PLAN/INTERFACES/DECISIONS as needed. Keep all
   upload/submission/staking operations out of 04. Four diagnostics uploads remain, each separately approved.
5. Publish completed work to main using existing user authorization, verify the remote SHA, and give the user
   the actual laptop GPT prompt for 04. If authentication is still missing, report that accurately and preserve
   a portable patch/bundle instead of claiming publication.

---

# Earlier session notes (historical; written 2026-11-13 22:15 UTC)

Branch `worktree-m2-m6` holds the M2–M6 build. It is a **work-in-progress checkpoint, not yet reviewed — do not merge as-is.**

## Where it stopped

The build workflow reached its integration stage:

- Done: audit, plus four implementers (`train_knobs`, `blend_select`, `nn_runtime`, `docs_handoffs`).
- Interrupted: integration (pytest, smoke, `configs/minifull.yaml` end to end, `pkl_check`).
- Never ran: three adversarial reviews (selection statistics, pickle contract, handoff 03/04 executability) and the fix pass.

## Laptop handoff 02 findings the integration must address

Source: `handoffs/results/02-results.md` on `main`.

1. **blend.py / report.py:** compute and report the paired 20-era block bootstrap of the chosen blend vs the baseline member (`lgbm_everest`), on CV OOF plus the forward diff. The laptop AI stopped because this was missing.
2. **Member/group admission:** select the final blend among candidates via the paired gate (p_pos ≥ 0.8 on CV and forward diff ≥ 0; ties → simpler), and drop failing members. Equal weights over all 10 members scored 0.0226 against 0.0238 for the single LGBM; `lgbm_ghat` (0.0119) and CatBoost (0.0167, forward 0.0112) diluted it.
3. **bundle.py:** support `--members lgbm_everest` so diag upload #1 (handoff 03) is the single-LGBM M1 baseline.
4. **Handoff 04:** recalibrate runtimes. On the laptop (20 threads, 16 used) the whole test run took ~12.5 min (train 497 s, regime simulation 208 s). Spend the extra compute on measurement (fold offsets, bootstrap draws, stability checks), slower trees, admission candidates and the MLP, not on bigger searches or seed bagging.
5. **handoffs/02-test-run.md:** make the blend-vs-baseline p_pos informational, so a tie or loss means "keep the baseline".
6. **handoffs/INDEX.md:**
   - 00: accepted (b96dd66 / ee2afbd).
   - 01: accepted (c33bff6 / acfaf33).
   - 02: accepted (34c722a / 86c1c5c) — lgbm 0.0238 / fwd 0.0313 / sharpe 0.51; EW blend worse; Cat 18 s but weak; search tie.

## Update 22:35 UTC — hotfix landed on `main` (d2f1d0b)

To unblock the laptop before the session ended, a small hotfix went straight to `main`:

- `bundle --members` (equal weights over an explicit subset), with a test.
- `handoffs/03-diag-upload-1.md`: diag upload #1 of the single-LGBM baseline from `test01`, with `--p-b 0 --p-c 0`.
- The handoff 00 file-size fix and the handoff 02 "p_pos informational" fix.
- INDEX: 00–02 accepted, 03 ready.

Items 3, 5 and 6 above are therefore done on `main`. When merging this branch, expect conflicts in `stealth/bundle.py`
(this branch has its own `--members`; keep the branch version and make sure the main test
`tests/test_train_smoke.py::test_bundle_members_subset` still passes), `handoffs/03-diag-upload-1.md` (keep main's
version unless the branch's is better reviewed), `handoffs/00-env-setup.md`, `handoffs/02-test-run.md` and
`handoffs/INDEX.md`. The user may already have run 03 by then: check `handoffs/results/03-results.md` on `main`, and
if an `upload_id` is recorded, fetch the score with the MCP tool `eiq_get_diagnostics_result`.

## Update 22:50 UTC — branch integrated and merged with main; push to main blocked

Done solo by Claude (no subagents):
- `main` merged into this branch, with conflicts resolved (merge commit on `worktree-m2-m6`). **252 tests pass.**
- The integrator's `minifull` pipeline run (`artifacts/mfpipe`) is green end to end, including the MLP on CPU torch.
- Partial review:
  - Contract: `pkl_check` passes on a pickle that includes the MLP member, with torch=0 and only `stealth._runtime` by value.
  - Branch code is backward-compatible with the laptop's old-format `test01` artifacts, so handoff 03 works either way.
  - The admission ladder rejects weak arms on the paired gate; the staking-aware neutralisation selection is as designed.
  - MLP training uses inner early stopping, independent count sources and train-only standardisation, with a numpy/torch parity check.
- Handoff 04 clone path fixed to `~/Projects/eversteer/stealth`.

Claude's own push to `main` was refused by the permission classifier, so **the user pushed it themselves at
~22:55 UTC: `main` = e69b848, which includes all of the M2–M6 work.** Next session:
1. Run the three independent reviews on `main`: selection-stats/leakage of knobs + search adoption, contract,
   handoff 04 executability. Fix what they find.
2. Note: Claude cannot push to `main` without a Bash permission rule. Work on a branch and ask the user to push or
   merge, or ask them to add the rule.

Finding to tell the user: applied to the laptop's 02 regime table, the staking-aware rule picks **p_b = 0**: worst-case
score 0.0351 against 0.0274 at p_b = 0.5, because neutralising hurts in benchmark-right rounds.

## Update 23:15 UTC — diagnostics upload #1 scored (M1 baseline, neutralisation off)

Upload `35bfe266-286a-4464-a532-e4dca0d7899b` (results on main: 56819a6). Validation board, 150 models:

| metric | ours | board median | board P75 | models scoring > 0.045 (n = 23, median) |
|---|---|---|---|---|
| fit20 | 0.0189 | 0.0193 | 0.0214 | – |
| unq | 0.0039 | – | – | – |
| inov | 0.0141 | – | – | – |
| round score | ~0.0368 (rank 89) | 0.0388 | 0.0423 | – |
| sharpe | 0.32 | 0.344 | 0.381 | 0.40 |
| max drawdown | −7.55 (worse than 67%) | −5.98 | −4.40 | −3.76 |
| feature exposure | 0.685 (higher than 73%) | 0.659 | – | 0.686 |

Takeaways:
- **The offline framework is calibrated.** The regime simulation at p_b = 0 predicted a score of 0.0383
  (fit 0.0224 / unq 0.0026 / inov 0.0133); the true figures are 0.0368 (0.0189 / 0.0039 / 0.0141). The INOV proxy
  is close to the true core set, so M2–M6 selection can trust the offline numbers. Discount OOF FIT by about 0.79 for
  validation; the forward block (0.0313) was optimistic.
- **The gap to strong models is FIT, Sharpe and drawdown**, not UNQ/INOV. Long autocorrelated losing stretches
  (autocorrelation 0.94) cause the drawdown, which matters for staking. M2–M6 levers (knobs incl. recency, admission,
  family diversity) and the staking-aware W criterion target exactly this. Consider adding drawdown on OOF as a
  reported metric in the final selection.
- The #1 board model (0.136, Sharpe 1.33) is an outlier, plausibly exploiting the partly in-sample `v1_sherpa` on
  validation. Don't chase it.
- 4 diagnostics uploads remain (M3, M5 ×2, final).

## Steps

1. Check out this branch. Symlink `data/` and `artifacts/pklcheck_venv` from a checkout that has them, or re-download the data and recreate the venv.
2. Finish the integration and apply items 1–6.
3. Run the three adversarial reviews and fix what they find.
4. Merge into `main`. The user pre-approved this once tests and reviews pass (`docs/DECISIONS.md` #8).
5. Give the user the GPT prompt for handoff 03. The user pre-approved it (#9); the GPT still asks for confirmation in its own chat before uploading.
6. Finalise handoff 04 scaled to the laptop's real speed.
