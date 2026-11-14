# Next session — state after the M2–M6 fix pass (2026-11-14)

## Current follow-up (read before the historical checkpoint below)

**Newest: diagnose before discarding.** The user explicitly requires causal investigation of promising ideas
that underperform. `forensics06-*` and `forensics08-*` report matched controls, adverse results and unresolved
causes. More trees was a contradicted repair hypothesis. Preserving raw inputs recovered late latent-tree
FIT from 0.020151 to 0.024301 with frozen parameters; scale-consistent ensembling helped some windows. A
balanced denoising-head repair then reached 0.02489356 (+14.13% over the 0.02181070 baseline). Its baseline
blend passed the within-round corrected aggregate check, but failed the late-window guard. Thus no model was
promoted, confirmation remained unread, and baseline01 remains selected. This is not a reliable major gain.

The 480-trial replication found inner rankings weak/negative in most cases, with all nine original ensembles
replaying. Broad-consensus controls partially repaired that selection failure: the existing-search blend gained
6.14% overall and the fixed-prior blend 6.10%; the latter was slightly positive late but weaker early. The
remaining temporal generalization problem is unresolved. Do not retire these ideas solely on their scores,
claim every cause is proved, select an oracle later winner, or relax a completed gate. Keep supported,
contradicted and unresolved mechanisms separate and register new interventions before their scores.

Latest reports: `forensics06-*`, `repairs07-*`, `forensics08-*`, `consensus09-*` in `handoffs/results`.
Final suite: 461 passed, one expected CUDA no-device-path skip. Research code and private source-bound caches
are preserved. Reproduce each campaign at its recorded implementation/report commit; current source is newer
than older plan hashes. The following checkpoints are historical where they conflict with this update.

**Latest, after stability05:** the observed best discovery FIT is now 0.02462116 versus baseline 0.02181070,
or +12.89%, from a fixed stable-covariance/baseline blend. The latest window remains down 2.51%. It did not pass
the four-comparison Holm gate or the predeclared 15% major-improvement criterion. No candidate was promoted,
confirmation was not loaded, and baseline01 remains selected. The requested reliable major gain is still
unproven. Final suite: 451 passed, 1 expected CUDA no-device-path skip. Full evidence:
`handoffs/results/stability05-*`; protocol `docs/STABILITY-05.md`; implementation `21da749`; checkpoints
`artifacts/stability05`. This targeted follow-up uses temporal signal-risk weighting and optional recent-signal
forecasting; its direct linear solves require no additional neural training. The following paragraphs preserve
the preceding 03/04 checkpoint. Reproduce each source-bound experiment at its recorded version.

The user renewed the request for a **major predictive improvement**, including much longer training. Sustained
campaigns 03/04 are complete; reports are `handoffs/results/campaign03-*` and `representations04-*`. Across them,
336 nested search trials evaluated 1,008 inner tree fits, with more than 21,600 GPU training epochs plus refits.
The best historical discovery FIT increased from 0.02181070 to 0.02431878 (+11.50%), but the latest discovery
window fell from 0.02752725 to 0.02326433. Neither campaign cleared its predeclared promotion gates or 15%
major-improvement criterion. The user's performance objective remains unmet. Do not present tests, epochs,
code volume or this exploratory average gain as a reliable major improvement. Baseline01 remains selected;
neither campaign loaded confirmation labels, exported a new deployment pickle or made a platform write.

Long supervised fitting overfit. Target-free denoising improved the average, but did not solve temporal
instability. All saved fold models replayed correctly, and the final suite passed 448 tests with one expected
CUDA no-device-path skip. Encoder weights were unchanged after permuting targets in a regression check.
Protocols: `docs/CAMPAIGN-03.md`, `docs/REPRESENTATIONS-04.md`. Source-bound private checkpoints live under
`artifacts/campaign03` and `artifacts/representations04`. Reproduce campaign03 at `a8a3d00`/`bbbb29b`, before the
runtime extension; reproduce representations04 at `3ad0a71` or its report commit. Changing source requires a
new run ID; do not bypass the plan guard or retune on the old audit. Further development needs a new explicit
experiment design and honest historical/prospective evidence, not a relaxed gate after seeing these scores.

The user's latest instruction authorises autonomous model development and prioritises powerful local CPU/GPU
compute over compute cost. This supersedes the old laptop-operator-only prohibition on code changes for new
research. The preference is recorded in `docs/DECISIONS.md` and `docs/RESEARCH-01.md`. Always use the absolute
training and GPU venv interpreters, and never display `.env`.

Research rounds `research01` and `research02` are complete: nine architectures plus four objective/stopping
recipes, each tested alone and in a fixed baseline blend. Every challenger failed discovery; no confirmation
was scored, no challenger was promoted, and the exact unsearched `baseline01` artifact remains the conservative
candidate. Results and full failed-candidate tables are in `handoffs/results/research01-*` and `research02-*`.
Source-bound private checkpoints live in `artifacts/research01` and `artifacts/research02`; the first numerical
failure is preserved in `artifacts/research01-solver-limit`. Do not silently reuse a run ID after changing code,
data, recipes or seeds. Historical discovery/confirmation data are not a fresh holdout. Neither round made a
platform write. Their reports describe the actual tests and serialization checks.

Handoff 04 completed on the laptop and was pushed as `c66d0cd`, `da2ff1c`, `a329cb3`: technical status ok;
**no audit win**. Primary-agent follow-up verification and the contract fixes are documented in
`docs/reviews/bot-followup-2026-11-14.md`. This solo follow-up does not claim the stopped independent agents finished.
Do not rerun the full search or rescore an altered model on the audit.

The conservative candidate is the exact unsearched `full01/baseline` comparator, exported with
`stealth.bundle --audit-baseline`, not a member subset of the selected blend. No M7 full-data refit is authorised
by this artifact export. The optional next calibration step is `handoffs/05-diag-upload-2.md`; it needs its own
explicit approval and never automatically promotes full01. No new platform writes have occurred.

The remaining sections below describe the earlier checkpoint and are historical where they conflict with this update.

Read this first, then `docs/DECISIONS.md` (#11–#15 are new), `handoffs/INDEX.md`, `docs/INTERFACES.md` section 22 and
the "Fix verification" sections of `docs/reviews/m2-m6-{statistics,contract,handoff}.md`. Older history is in git
(`git show 6c719d6:docs/NEXT-SESSION.md`).

## Where things stand

- The three independent reviews found 6 statistics, 7 contract and 5 handoff issues. All of them are fixed, with
  regression tests. Three fresh, read-only reviewer agents then verified the fixes (results under "Independent
  verification" in each review file).
- The reserved audit partition is implemented end to end (`cv.audit`, `stealth.audit`):
  - eras 5945–6524 are never read by a fit or a choice of the run;
  - the frozen blend is scored once against an unsearched default-universe LightGBM baseline;
  - a permuted-label regression proves the audit labels reach no decision.
- `train --skip-external / --require-external` exist and the pipeline uses them.
- **Handoff 04 is finalised for the laptop**: Linux, 20 threads (16 used), RTX 5070 Ti, ~30.8 GB, Europe/London.
  Estimates are scaled from handoff 02 (training 497 s, simulation 208 s). The measurement budget is 5,000 bootstrap
  and 1,000 simulation draws; there are no extra trials or seeds. The run uploads, submits and stakes nothing.
- Handoff 03 is accepted (upload `35bfe266`). Four diagnostics uploads remain, each needing its own USER MUST APPROVE
  handoff. The `m1_lgbm` manifest from 03 holds a legacy pkl_check pass without a hash, so it no longer satisfies any
  upload gate: re-run `pkl_check` before any future upload of it.

## Independent verification status (session wrapped up early at the user's request)

- **Contract reviewer: finished.** No bypass and no P1. All seven findings were confirmed fixed.
  - **Open P2:** `predict.check_upload_artifact` should also require `status == "passed"`, a checker Python of
    3.11.x, both isolation flags false and libs equal to the lane pins. Also, `pkl_check.update_manifest(pkl, True)`
    with no report should not create a pass. The real CLI always passes a report, so this is not live.
  - **Open P3s:** the diag/predict re-read race after the gate; `full_blobs: []` treated as legacy; the
    `--venv-python` bare-name regression; `ensure_check_venv` deletes any `--venv` dir without `.pins.txt`.
- **Statistics and handoff verification reviewers: stopped before they reported.** Re-run them next session before
  diagnostics upload #2.

## Verification actually performed on the sandbox (2 cores, Python 3.11.2, exact pins)

- Full pytest in `.venv` (no torch importable): see the final count in the commit message of the merge to main.
  Before the docs commits it was 358 passed + 2 expectation updates, then green.
- Real-data minifull pipeline `mf01` with the external MLP on CPU torch (`/opt/ml/bin/python3`, `--mlp-device cpu`):
  - **Outcome:** `PIPELINE OK` in 154 s: train 100, mlp 10, retrain 14, blend 16, audit 3, bundle 0, pkl_check 3,
    diag 8 s, then report. Report status `ok` with all evidence bound to the files.
  - **Bundle:** frozen (`--require-frozen`, `p_source simulation`); `pkl_check` PASSED in `artifacts/pklcheck_venv`,
    51 eras; diagnostics file 125,931 rows / 1,392 eras.
- **MLP portability:** a real-data bundle including the trained MLP (`--members ... mlp_everest --p-b 0.5 --p-c 0.25`)
  passed `pkl_check` in the clean venv on 51 eras: torch=0, pins verified, benchmark neutralisation active.
- **Failure paths:** a same-command resume reused every cache. `--no-mlp --from blend` refused while the OOF holds the
  MLP. Run `mf02`: TERM mid-train recorded `interrupted` (143) with no stray processes and the member left `running`;
  the re-run finished OK and reproduces `mf01`'s blend and audit exactly.
- **Real `full.yaml` layout** (RunData on the real data, pipeline `--dry-run`): development 2800–5924 (3,125 eras),
  audit 5945–6524 (580 eras, 53,292 rows), full refits ≤ 5924, own-fold count sources, documented commands.
- **Leak gate on a toy window:** eras 6000–6299 are a high-signal regime. `feature_Agouti` alone has a per-era
  Spearman of +0.20 there, so a toy test on that window raises `leak_spearman` locally (`tests/test_audit.py`).
  The full-history gate (0.18) is unchanged.

## Environment notes (this sandbox; nothing here is a laptop change)

- `.venv` = `/usr/bin/python3.11 -m venv` + `requirements-lock.txt` + `-e .` (no torch). `artifacts/pklcheck_venv` was
  built by `pkl_check.ensure_check_venv`. `/opt/ml/bin/python3` has CPU torch and is used only as `--gpu-py` with
  `--mlp-device cpu`.
- Data: the SDK on the box is signed in, so a small script calling `EverestAPI(tournament="futures")` downloaded the
  six parquet files into ignored `data/`. No model, upload, submission, auto-submit change or stake.
- GitHub reads work. **Pushes need credentials this box does not have** (no gh, helper, token or SSH key). See the
  publication note in the session report: the branch/main state was either pushed and verified, or left as a local
  fast-forward plus a git bundle for the user to push.
- Commits use `git -c user.name=Claude -c user.email=noreply@anthropic.com` (no repo config change). The hook runs via
  `core.hooksPath=scripts/hooks`.

## Next steps

1. The user runs handoff 04 on the laptop with the kickoff prompt in `handoffs/GPT-KICKOFF.md`.
2. Pull and read `handoffs/results/04-*`:
   - a technical success needs report status `ok`, `pkl_check` bound to the bytes, the diag file and the audit;
   - record the audit verdict (blend vs the unsearched baseline on 5945–6524) separately;
   - compare dev selection estimates with the audit and handoff 02's numbers;
   - check the GPU MLP timings against the 5–25 min estimate.
3. Decide (user) on diagnostics upload #2 of the full01 bundle: a new USER MUST APPROVE handoff (05 or 06) that
   re-runs `pkl_check`, `diag --preflight`, and uploads once.
4. M7 (separate decision, never silently after reading the audit): whether to refit on all eras up to 6524 for the
   lane upload, then the lane upload + monitor (handoff 05, USER MUST APPROVE).
