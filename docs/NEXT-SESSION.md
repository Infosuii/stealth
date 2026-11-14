# Next session — state after the M2–M6 fix pass (2026-11-14)

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
