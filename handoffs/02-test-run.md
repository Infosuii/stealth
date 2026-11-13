# Handoff 02 — test run (`configs/test.yaml`): calibrate the laptop and reproduce the baseline (M1)

## 1. Purpose + success criterion

Run the full validation protocol at test scale: eras ≥ 2800, 5 purged folds (purge 20 + embargo 20) plus the
forward block (train < 5925, test 5945–6524); 12-cell exhaustive LightGBM grid with top-3 averaging, 3 aux-target
members, 4 fixed XGBoost configs, **1 timed CatBoost config**, ridge; nested blend + 500-draw regime simulation;
bundle + portability check. This is milestone **M1**: it must reproduce the honest baseline and record per-fit wall
times (the CatBoost keep/drop decision). Expected 30–45 min on 16 threads, < 6 GB RAM.

Success = ALL of (read from `handoffs/results/02-metrics.json` → `configs[]`):
- `lgbm_everest`: `fit_mean` in **[0.022, 0.030]** (0.026 ± 0.004), `forward_fit` **≥ 0.020**, `sharpe` **≥ 0.5**.
- `blend_*` (the blend row): `fit_mean` ≥ `lgbm_everest.fit_mean` − 0.002 (the blend must not be worse than its
  best member beyond noise). A paired `p_pos` of the blend vs `lgbm_everest` is **informational only**: the current
  code does not compute it yet (M2–M6 adds it), so its absence is NOT a failure. Do not stop on it; a tie or a
  loss simply means "keep the baseline".
- `leak_check` = `pass`; no config has `suspect: true` (any `fit_mean` > 0.030 is flagged by `train` and must be
  reported as suspect, not as a win). `lgbm_everest` is a searched member: `artifacts/test01/search/lgbm_everest.gain.json`
  records top-3 minus the base-params cell on OOF (`gain`) and on the forward block (`forward_gain`); a `gain` > 0.004 or a
  negative `forward_gain` is noted by `train` in `metrics.json.notes` — copy such a note into the operator notes verbatim.
  `fit_se` in the results is the 20-era block-bootstrap SE (expect ≈ 0.003–0.004), not std/sqrt(n).
- `timings.per_member_fit_s.cat_everest_l2_30` is present. **≤ 120 s** → "CatBoost kept"; **> 120 s** →
  "CatBoost dropped" — write the verdict in the operator notes.
- `pkl_check` logs `pkl_check: PASSED` and exits 0; `models/test01.manifest.json` exists.
- `02-results.md`, `02-metrics.json`, `02-log-tail.txt` pushed.

## 2. Preconditions

- Handoffs 00 and 01 reported success: the **Operator notes** section of `handoffs/results/01-results.md` contains
  `pkl_check: PASSED` (the reporter itself does not record pkl_check; the previous GPT pasted it there).
- `git -C "$HOME/stealth" pull`; `.venv` present; `data/` complete; ~3 GB free disk.
- Laptop on mains power, not expected to sleep for an hour.

## 3. HUMAN STEP

Only if `test -s .env` fails: the human copies `.env.example` to `.env` and pastes `EIQ_API_KEY` (handoff 00).
This run itself is offline; the key is not used.

## 4. Environment

```bash
REPO="$HOME/stealth"                    # absolute clone path; change if different
PY="$REPO/.venv/bin/python"             # absolute venv interpreter used by EVERY command
cd "$REPO"
"$PY" -m stealth.selfcheck | tail -3    # expect PINS OK
nproc                                   # logical cores; see step 1
```

## 5. Steps

**Step 1 — thread count (seconds; the only permitted edit)**
`configs/test.yaml` has `run.threads: 16`. If `nproc` printed a smaller number, set `run.threads` to that number
(this one edit is explicitly allowed by this handoff; do not commit it — `git checkout configs/test.yaml` after
the run if `git status` shows it modified). Do not change anything else.

**Step 2 — train (30–40 min on 16 threads; resumable — re-run the same command after any interruption)**
```bash
"$PY" -m stealth.train --config configs/test.yaml --run-id test01 --resume
```
Expected output signature (wording from the real smoke run; the numbers will differ):
```
INFO stealth.train: run test01: config test, resume=True, data_dir=.../data
INFO stealth.train: run data: ... rows, 3725 eras (2800..6524), 5 folds, forward block
INFO stealth.search: search lgbm_everest: grid=12 cells, target=12 distinct trials, 0 already done (exhaustive)
INFO stealth.train: lgbm_everest[<hash>] fold0: inner best_iter=...
INFO stealth.train: lgbm_everest[<hash>] forward: inner best_iter=...
INFO stealth.train: lgbm_everest[<hash>]: frozen n_iter per fold=[...] forward=... full-refit=... (inner iters {...}; sources {...})
INFO stealth.train: lgbm_everest search gain: top-3 OOF fit 0.02.. vs base-params cell 0.02.. (gain +0.00..); forward ... (gain ...)
INFO stealth.train: progress: member 6/10 lgbm_meltsene (elapsed 720s)
INFO stealth.train: member lgbm_everest done in ...s (frozen n_iter=..., rho_b(val)=0.7...)
INFO stealth.metrics: leak_check pass: mean OOF Spearman 0.1... <= 0.18
INFO stealth.train: member lgbm_everest  fit20 mean=0.02.. se_block=0.003. sharpe=0.5.. pct_pos=0.6. forward=0.02.. n_iter=...
INFO stealth.train: run test01 finished: status=ok exit=0 total=...s
```
Exit 0 expected. Exit 2 = leak/gate failure; exit 1 = error (traceback in `artifacts/test01/run.log`).
The log prints a progress line every 60 s; silence > 10 min means a hung process — note the time, kill it with
Ctrl-C, re-run the same command (it resumes from the member cache / Optuna SQLite).

**Step 3 — nested blend + regime simulation (2–5 min; 500 draws)**
```bash
"$PY" -m stealth.blend --run-id test01
```
Expected: log shows `nested NNLS: nested weights {...}; eval fold4 diff_mean=... se=... p_pos=...; ... -> ADOPT nnls` (or
`-> keep equal weights`), then `blend[ew|nnls]: fit20 mean=... sharpe=... pct_pos=... forward=...`, the regime table,
and `artifacts/test01/blend.json` written.

**Step 4 — bundle (1–2 min)**
```bash
"$PY" -m stealth.bundle --run-id test01 --out models/test01.pkl
ls -l models/test01.pkl models/test01.pkl.sha256 models/test01.manifest.json
```

**Step 5 — portability check (1–3 min; reuses `artifacts/pklcheck_venv` from handoff 01)**
```bash
"$PY" -m stealth.pkl_check models/test01.pkl
```
Expected: `pkl_check: PASSED` plus the `size / eras / slowest call` and `byte scan: torch=0 ... non-runtime stealth refs=0`
lines, exit 0. Note the `slowest call` seconds figure (must stay < 5 s/era).

**Step 6 — offline dry run (seconds; nothing submitted)**
```bash
"$PY" -m stealth.predict --pkl models/test01.pkl --offline
```
Expected: `predicted exped=exped_8055 n=90 ... unique=90` then `dry run ok: exped=exped_8055 n=90 -> ...`, exit 0.
**No `--submit`, no `--upload`.**

## 6. If it fails

- Step 2 `MemoryError` / OOM-killed: re-run with `run.threads` halved (edit permitted as in step 1); it resumes.
- Step 2 exit 2 (`LEAK`): the run is void; report and stop. Exit 1: report (the reporter copies the traceback).
- Step 2 wall time > 75 min: let it finish if progress lines keep coming; record the time — the full run is sized
  from it. If a single CatBoost fit exceeds 10 min, kill, re-run (resume skips completed members) and note
  "CatBoost dropped (timeout)".
- Step 3 fails but step 2 succeeded: still run step 7 — the member metrics are the main deliverable.
- Step 5 `FAIL`: copy the failing lines into the operator notes; do not fix.

## 7. Report back (mandatory, also on failure)

```bash
"$PY" -m stealth.report --run-id test01 --handoff 02
```
Writes `handoffs/results/02-{results.md,metrics.json,log-tail.txt}`, then
`git add handoffs/results && git commit -m "results: 02 test-run" && git push`. Exit 0 = pushed.
On **PUSH FAILED — results are saved locally in handoffs/results/; ask the human to run git push**, repeat that to
the user verbatim and stop. On **GIT COMMIT FAILED — …** (missing git identity), repeat it verbatim and stop (HUMAN STEP
in handoff 00 section 3).

Then append the operator notes (CatBoost verdict, pkl_check tail, predict summary, total wall time, `nproc`) and push:
```bash
printf '\n## Operator notes\n\n```text\n%s\n```\n' "<CatBoost: kept|dropped (NNN s per fit); pkl_check: PASSED|FAILED, N s/era; predict --offline: ok; total wall: MM min; nproc: N>" >> handoffs/results/02-results.md
git add handoffs/results && git commit -m "results: 02 test-run (notes)" && git push
```
Also commit the manifest + sha (never the pickle): `git add models/test01.manifest.json models/test01.pkl.sha256 && git commit -m "models: test01 manifest" && git push`.
Results files carry metrics and config NAMES only; hyperparameter values stay in `artifacts/` (the reporter enforces this).

## 8. Rules for the executing AI

- No code edits. The only permitted config edit is `run.threads` in `configs/test.yaml` (step 1); revert it afterwards.
- Never `--submit`, never `--upload`, never a diagnostics upload: this handoff has **no USER MUST APPROVE banner**.
  (The M1 diagnostics upload is a separate, later handoff that will carry one.)
- Never commit `data/`, `.env`, `*.pkl`, `artifacts/`. Only `handoffs/results/` and `models/*.{manifest.json,sha256}`.
- Never `cat`/`echo`/`grep` `.env`; never `git remote -v`; never `git config --list`; never print the API key.
- Stop and write the results file on any judgement call.
