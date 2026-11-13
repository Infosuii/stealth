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
  `members.ridge_everest.fit.mean` between 0.005 and 0.040. Sandbox reference (same seed, deterministic):
  lgbm_everest 0.0152, lgbm_tougroute 0.0087, ridge_everest 0.0241, blend 0.0151. **Expected and harmless:**
  `ridge_everest` carries `suspect: true` (and the results file shows a SUSPECT banner) because the smoke LightGBM
  is deliberately under-trained (200 fixed trees) and ridge beats it by more than the 0.004 margin. Report it as
  "expected in smoke"; it is not a leak.
- `blend` exits 0 and writes `artifacts/smoke01/blend.json` with a `regime` table.
- `bundle` writes `models/smoke01.pkl` (+ `.sha256`, `.manifest.json`); `pkl_check` logs `pkl_check: PASSED` and
  exits 0 (all 90 live rows + 50 validation eras predicted in (0,1), deterministic, no `stealth`/`torch` in the
  checker's `sys.modules`, byte scan `torch=0`, `non-runtime stealth refs=0`).
- `predict --offline` logs `predicted exped=exped_8055 n=90 ... unique=90` then `dry run ok: exped=exped_8055 n=90`
  and exits 0 (no network, nothing submitted).
- `handoffs/results/01-results.md` + `01-metrics.json` pushed.

## 2. Preconditions

- Handoff 00 done: `.venv` present, `data/` has the 6 parquet files, `core.hooksPath` set, `.env` present.
- `git -C "$HOME/stealth" pull` (repo at the latest `main`).
- ~3 GB free disk (`artifacts/pklcheck_venv` is created once from the lane pins, ~1.6 GB; reused afterwards).

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

**Step 1 — tests (1–3 min; 65 s on the 2-core sandbox; prints only dots until the end)**
```bash
"$PY" -m pytest -q 2>&1 | tail -3
```
Expected last line: `N passed in …s` (a `, K skipped` suffix is fine; `failed` / `error` is not).

**Step 2 — train the smoke config (laptop: 2–5 min; sandbox with 2 cores: ≤ 8 min; resumable)**
```bash
"$PY" -m stealth.train --config configs/smoke.yaml --run-id smoke01 --resume
```
Expected output signature (real sandbox log lines; timestamps omitted):
```
INFO stealth.train: run smoke01: config smoke, resume=True, data_dir=.../data
INFO stealth.train: run data: 136246 rows, 1525 eras (5000..6524), 3 folds
INFO stealth.train: lgbm_everest fold0: fit 3.3s, 43764 test rows
INFO stealth.train: progress: lgbm_everest: fitting fold2 with n_iter=200 (elapsed 60s)
INFO stealth.train: member lgbm_everest done in 16s (frozen n_iter=200, rho_b(val)=0.7944)
INFO stealth.metrics: leak_check pass: mean OOF Spearman 0.0744 <= 0.18
INFO stealth.train: member lgbm_everest           fit20 mean=0.0152 se_block=0.0039 sharpe=0.342 pct_pos=0.64 forward=None n_iter=200
INFO stealth.train: member ridge_everest          fit20 mean=0.0241 se_block=0.0039 sharpe=0.534 pct_pos=0.71 forward=None n_iter=None  SUSPECT
INFO stealth.train: run smoke01 finished: status=ok exit=0 total=49s
```
(`se_block` is the 20-era block-bootstrap SE — ~0.004 here; it is deliberately ~3× the naive std/sqrt(n).)
Sandbox wall time: 49 s on 2 cores (peak RSS 1.22 GB). Exit code 0. Exit 2 = leak/gate failure (report it);
exit 1 = error (traceback is in `artifacts/smoke01/run.log`).

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
Expected: a small pickle (sandbox: 0.45 MB for the 3 smoke members), its sha256 file and manifest; last log line
`bundle ok: run=smoke01 members=['lgbm_everest', 'lgbm_tougroute', 'ridge_everest'] p_b=0.50 p_c=0.25 sha256=...`.

**Step 5 — portability check in a throwaway venv (first time 2–4 min to build the venv, then < 1 min)**
```bash
"$PY" -m stealth.pkl_check models/smoke01.pkl
```
Expected last lines (sandbox: venv built in 29 s, check itself 4 s):
```
INFO stealth.pkl_check: pkl_check: PASSED
INFO stealth.pkl_check:   python: .../artifacts/pklcheck_venv/bin/python (3.11.x)  size: 0.45 MB  eras: 51  slowest call: 0.8s  elapsed: 8.3s
INFO stealth.pkl_check:   byte scan: torch=0 stealth_total=6 non-runtime stealth refs=0
INFO stealth.pkl_check:   note: missing feature column raises (ok)
```
(The `(3.11.x)` after the interpreter path is the checker's own Python; a `PROBLEM: checker python 3.12 != expected 3.11`
line means the throwaway venv was built from the wrong interpreter — see section 6.)
`pkl_check: PASSED` and exit 0 is the success signal; `pkl_check: FAILED` lists the problems. Exit 0.

**Step 6 — offline dry-run prediction on the live parquet (seconds; no network, nothing submitted)**
```bash
"$PY" -m stealth.predict --pkl models/smoke01.pkl --offline
```
Expected (2 s): `predicted exped=exped_8055 n=90 in 0.8s bench_used=True core_used=True unique=90` then
`dry run ok: exped=exped_8055 n=90 -> .../artifacts/predictions/exped_8055.parquet`, exit 0.
**Do not add `--submit` or `--upload`.**

## 6. If it fails

- Any step: a wall time > 3× the expected time → still finish, note the time in the results; the test run in handoff 02
  is sized from it. Do not abort a step that is still printing progress lines.
- Step 1: a failing test → stop, report (step 7). Do not edit code.
- Step 2: `MemoryError`/killed → re-run the same command (it resumes). Exit 2 with `LEAK` → report; the run is void.
  Exit 1 → the traceback is in `artifacts/smoke01/run.log`; `report.py` copies it; stop.
- Step 5: `FAIL` lines → copy them into the results notes; do not try to fix the pickle. If venv creation fails for
  lack of network, say so and report; everything else can still be reported. If it fails with `No module named pip`
  / `ensurepip` or `base interpreter ... is not 3.11`, create the throwaway venv yourself and point the checker at it:
  `uv venv artifacts/pklcheck_venv --python 3.11 --seed` then re-run step 5 (the checker installs the lane pins into
  it), or `"$PY" -m stealth.pkl_check models/smoke01.pkl --venv-python <any clean python3.11 without stealth>`.

## 7. Report back (mandatory, also on failure)

```bash
"$PY" -m stealth.report --run-id smoke01 --handoff 01
```
This writes `handoffs/results/01-{results.md,metrics.json,log-tail.txt}` and runs
`git add handoffs/results && git commit -m "results: 01 smoke" && git push`. Exit 0 = pushed.
If it prints **PUSH FAILED — results are saved locally in handoffs/results/; ask the human to run git push**, repeat
that sentence to the user verbatim and stop. If it prints **GIT COMMIT FAILED — …** (usually a missing git identity:
`Author identity unknown`), repeat that verbatim too — it is the HUMAN STEP from handoff 00 — and stop.
Then append the operator notes (mandatory: handoff 02 reads `pkl_check: PASSED` from them):
```bash
printf '\n## Operator notes\n\n```text\n%s\n```\n' "pkl_check: PASSED|FAILED (<paste the 4 pkl_check tail lines>); predict --offline: <paste the predicted exped=... line>; wall times: train <s>, pkl_check <s>" >> handoffs/results/01-results.md
git add handoffs/results && git commit -m "results: 01 smoke (notes)" && git push
```
Do **not** `git add models/` in this handoff (the smoke manifest is throwaway).

## 8. Rules for the executing AI

- No code or config edits. Only the commands above, in order.
- Never `--submit`, never `--upload`, never a diagnostics upload: this handoff has no USER MUST APPROVE banner.
- Never commit `data/`, `.env`, `*.pkl`, `artifacts/`; only `git add handoffs/results`.
- Never `cat`/`echo`/`grep` `.env`; never `git remote -v`; never `git config --list`; never print the API key.
- Stop and write the results file on any judgement call; do not improvise fixes.
