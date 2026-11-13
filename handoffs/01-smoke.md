# Handoff 01 — smoke run on the laptop (prove the whole pipeline end to end)

## 1. Purpose + success criterion

Run the smallest config (`configs/smoke.yaml`: eras ≥ 5000, 3 purged folds, 2 small LightGBM members + ridge,
equal-weight blend, 50-draw regime simulation), build the pickle, and prove it loads and predicts in a throwaway
venv **without** `stealth` installed. This checks the environment, not the model. ~10 minutes total.

Success = ALL of:
- `pytest`: 0 failed.
- `train` exits 0; `artifacts/smoke01/metrics.json` has `status: ok`, `leak_check: pass`, and
  `members.lgbm_everest.fit.mean` between **0.010 and 0.040** (3 folds on ~1,500 eras is noisy; the point is the
  sign and the magnitude). A value **> 0.045** is a suspected leak: still report, flag it, do not celebrate.
  `members.ridge_everest.fit.mean` between 0.005 and 0.040.
- `blend` exits 0 and writes `artifacts/smoke01/blend.json` with a `regime` table.
- `bundle` writes `models/smoke01.pkl` (+ `.sha256`, `.manifest.json`); `pkl_check` prints `PASS` and exits 0
  (all 90 live rows predicted in (0,1), deterministic, no `stealth`/`torch` in the checker's `sys.modules`).
- `predict --offline` prints a 90-row prediction summary and exits 0 (no network, nothing submitted).
- `handoffs/results/01-results.md` + `01-metrics.json` pushed.

## 2. Preconditions

- Handoff 00 done: `.venv` present, `data/` has the 6 parquet files, `core.hooksPath` set, `.env` present.
- `git -C "$HOME/stealth" pull` (repo at the latest `main`).
- ~2 GB free disk (`artifacts/pklcheck_venv` is created once, ~500 MB).

## 3. HUMAN STEP

Only if `test -s .env` fails: the human copies `.env.example` to `.env` and pastes `EIQ_API_KEY` (see handoff 00).
Nothing in this handoff needs the key — the whole run is offline from `data/`.

## 4. Environment

```bash
REPO="$HOME/stealth"                    # absolute clone path; change if different
PY="$REPO/.venv/bin/python"             # absolute venv interpreter used by EVERY command
cd "$REPO"
"$PY" -m stealth.selfcheck | tail -3    # expect PINS OK
```

## 5. Steps

All steps are idempotent. `train` writes its own `artifacts/smoke01/run.log` and prints one progress line per
minute — do not add `tee` or redirections.

**Step 1 — tests (1–3 min)**
```bash
"$PY" -m pytest -q
```
Expected last line: `NN passed, K skipped` with 0 failed.

**Step 2 — train the smoke config (laptop: 2–5 min; sandbox with 2 cores: ≤ 8 min; resumable)**
```bash
"$PY" -m stealth.train --config configs/smoke.yaml --run-id smoke01 --resume
```
Expected output signature (log lines, exact wording may differ):
```
INFO stealth.train: run smoke01 config=smoke eras>=5000 n_folds=3 members=3
INFO stealth.train: fold0 lgbm_everest fit ... s
...
INFO stealth.train: leak_check pass; metrics -> artifacts/smoke01/metrics.json; exit 0
```
Exit code 0. Exit 2 = leak/gate failure (report it); exit 1 = error (traceback is in `artifacts/smoke01/run.log`).

**Step 3 — blend + regime simulation (< 1 min)**
```bash
"$PY" -m stealth.blend --run-id smoke01
```
Expected: a regime table in the log (rows for bench means 0.0 / 0.009 / 0.02 × p_b 0, .25, .5, .75 × bpos/bneg/mix)
and `artifacts/smoke01/blend.json` written.

**Step 4 — bundle the pickle (< 1 min)**
```bash
"$PY" -m stealth.bundle --run-id smoke01 --out models/smoke01.pkl
ls -l models/smoke01.pkl models/smoke01.pkl.sha256 models/smoke01.manifest.json
```
Expected: a pickle of a few MB, its sha256 file and manifest.

**Step 5 — portability check in a throwaway venv (first time 2–4 min to build the venv, then < 1 min)**
```bash
"$PY" -m stealth.pkl_check models/smoke01.pkl
```
Expected last lines: a checklist of `ok` items (loads, exact DataFrame, (0,1), no NaN, deterministic, misaligned /
constant / None benchmark, no stealth/torch in sys.modules, byte scan clean, < 5 s/era, < 50 MB) and `PASS`. Exit 0.

**Step 6 — offline dry-run prediction on the live parquet (seconds; no network, nothing submitted)**
```bash
"$PY" -m stealth.predict --pkl models/smoke01.pkl --offline
```
Expected: `90 predictions, min ... max ... in (0,1)`, exit 0. **Do not add `--submit` or `--upload`.**

## 6. If it fails

- Step 1: a failing test → stop, report (step 7). Do not edit code.
- Step 2: `MemoryError`/killed → re-run the same command (it resumes). Exit 2 with `LEAK` → report; the run is void.
  Exit 1 → the traceback is in `artifacts/smoke01/run.log`; `report.py` copies it; stop.
- Step 5: `FAIL` lines → copy them into the results notes; do not try to fix the pickle. If venv creation fails for
  lack of network, say so and report; everything else can still be reported.
- Any step: a wall time > 3× the expected time → still finish, note the time; the test run in handoff 02 is sized from it.

## 7. Report back (mandatory, also on failure)

```bash
"$PY" -m stealth.report --run-id smoke01 --handoff 01
```
This writes `handoffs/results/01-{results.md,metrics.json,log-tail.txt}` and runs
`git add handoffs/results && git commit -m "results: 01 smoke" && git push`. Exit 0 = pushed.
If it prints **PUSH FAILED — results are saved locally in handoffs/results/; ask the human to run git push**, repeat
that sentence to the user verbatim and stop. If step 5/6 produced output not captured in `metrics.json`, append it:
```bash
printf '\n## Operator notes\n\n```text\n%s\n```\n' "<paste the pkl_check tail and the predict --offline summary here>" >> handoffs/results/01-results.md
git add handoffs/results && git commit -m "results: 01 smoke (notes)" && git push
```
Do **not** `git add models/` in this handoff (the smoke manifest is throwaway).

## 8. Rules for the executing AI

- No code or config edits. Only the commands above, in order.
- Never `--submit`, never `--upload`, never a diagnostics upload: this handoff has no USER MUST APPROVE banner.
- Never commit `data/`, `.env`, `*.pkl`, `artifacts/`; only `git add handoffs/results`.
- Never `cat`/`echo`/`grep` `.env`; never `git remote -v`; never `git config --list`; never print the API key.
- Stop and write the results file on any judgement call; do not improvise fixes.
