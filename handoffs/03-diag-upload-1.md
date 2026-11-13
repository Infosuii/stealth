# Handoff 03 — diagnostics upload #1: the M1 LightGBM baseline

> **USER MUST APPROVE.** Step 4 uploads validation predictions to the Everesteer platform. Stop at step 4 and ask
> the human in this chat. Upload only after they explicitly say yes. This is upload **1 of the 5** pre-registered
> diagnostics uploads (`docs/DECISIONS.md`), so never run it twice.

## 1. Purpose + success criterion

Score the M1 baseline out of sample on the platform's validation diagnostics lane. The baseline is the single
`lgbm_everest` model from handoff 02's run `test01` (OOF fit20 0.0238, forward 0.0313, Sharpe 0.51), uploaded with
neutralisation **off** (`p_b 0`, `p_c 0`) so the number calibrates raw FIT. This does **not** submit to a live round,
stake anything, or change the model's auto-submit setting (which stays off).

Success = ALL of:
- `pkl_check: PASSED` for `models/m1_lgbm.pkl`.
- `diag ok: 125931 rows` for the validation predictions file.
- After the human approves: one upload accepted, with an `upload_id` logged, and, if scoring finishes within the
  wait window, a line `diagnostics done: fit20=… unq=… inov=… sharpe=…`.
- `handoffs/results/03-results.md` committed and pushed.

The numbers themselves are informational; there is no pass/fail threshold on fit20 here.

## 2. Preconditions

- Handoffs 00, 01 and 02 completed on this laptop; `artifacts/test01/` from handoff 02 still exists
  (`test -d artifacts/test01/members/lgbm_everest && echo ok`).
- Repo up to date: `git pull`.
- `.env` present (`test -s .env && echo ".env present"`); never print it.

## 3. HUMAN STEP

Only step 4, the approval. No other human action is needed.

## 4. Environment

```bash
REPO="$HOME/Projects/eversteer/stealth"   # the clone used for 00-02; change if different
PY="$REPO/.venv/bin/python"
cd "$REPO"
```

## 5. Steps

**Step 1 — build the baseline pickle (seconds)**
```bash
"$PY" -m stealth.bundle --run-id test01 --members lgbm_everest --p-b 0 --p-c 0 --out models/m1_lgbm.pkl
```
Expected last line: `bundle ok: run=test01 members=['lgbm_everest'] p_b=0.00 p_c=0.00 sha256=…`.

**Step 2 — contract check (under 1 min)**
```bash
"$PY" -m stealth.pkl_check models/m1_lgbm.pkl
```
Expected: `pkl_check: PASSED`, `torch=0`, `non-runtime stealth refs=0`.

**Step 3 — validation predictions, no upload (1–3 min)**
```bash
"$PY" -m stealth.diag --pkl models/m1_lgbm.pkl --out artifacts/diag/m1_lgbm_validation.parquet
```
Expected last line: `diag ok: 125931 rows -> …/m1_lgbm_validation.parquet`.

**Step 4 — STOP: ask the human**

Ask exactly:
> "Ready for diagnostics upload #1 of 5: the M1 LightGBM baseline's validation predictions, to model slot
> `mandelbrot-bayesian-payoff`. It is not a live submission and stakes nothing. Approve the upload? (yes/no)"

If the answer is anything other than an explicit yes, skip to section 7 and report "upload not approved".

**Step 5 — upload (after a yes; seconds to upload, then up to 20 min of scoring)**
```bash
STEALTH_CONFIRM_DIAG=1 "$PY" -m stealth.diag --pkl models/m1_lgbm.pkl --out artifacts/diag/m1_lgbm_validation.parquet \
  --upload --i-have-approval --wait --model-id mandelbrot-bayesian-payoff
```
Expected: `DIAGNOSTICS UPLOADED model=mandelbrot-bayesian-payoff … upload_id=<id>`, then a few `diagnostics running …`
lines, then `diagnostics done: fit20=… unq=… inov=… sharpe=…`. The record is written to
`artifacts/diag/m1_lgbm_validation.parquet.upload.json`.

## 6. If it fails

- Step 1 says `no members/<m>/spec.json`: `artifacts/test01` is missing; stop and report. Do not re-run handoff 02 on your own.
- Step 2 or 3 fails: copy the error lines into the results and stop. Do not upload.
- Step 5 refused before uploading (missing approval flag or env var): you mistyped the command; fix the typing only.
- Step 5 says the model is unknown: re-run once with `--model-id f06a70bf-ef8b-483d-b2a9-d4b9ef3e7043` (the same slot by id).
- **If an `upload_id` was printed, the upload happened. Never upload again**, even if waiting timed out or failed.
  Record the `upload_id` and report it; Claude will fetch the score.

## 7. Report back (mandatory, including on failure)

Create `handoffs/results/03-results.md` by hand with these fields. Copy them from the step outputs and from
`artifacts/diag/m1_lgbm_validation.parquet.upload.json`, but **do not commit that JSON file itself**.

```text
# Results 03 — diag-upload-1
status: ok | failed | upload not approved
git_sha: <git rev-parse HEAD>
pickle: models/m1_lgbm.pkl sha256 <from step 1>  pkl_check: PASSED|FAILED  slowest call: <s>
validation file rows: <n>
upload_id: <id or none>
diagnostics status: done | running | failed | not uploaded
fit20: <v>   unq: <v>   inov: <v>   sharpe: <v>
any other metric lines printed by `diagnostics done` or present under "result" -> "metrics" in the JSON: <copy>
operator notes: <anything unexpected, verbatim error lines>
```
Then:
```bash
git add handoffs/results/03-results.md && git commit -m "results: 03 diag-upload-1" && git push
```
If the push fails, say `PUSH FAILED` and ask the human to push.

## 8. Rules for the executing AI

- Upload at most once, and only after the human's explicit yes in step 4. Never use `--submit` or `predict --upload`.
- No code or config edits. Never commit `data/`, `.env`, `*.pkl`, `artifacts/` or the `.upload.json` file.
- Never `cat`/`echo` `.env`; never `git remote -v` or `git config --list`.
- On any judgement call, stop and write the results file.
