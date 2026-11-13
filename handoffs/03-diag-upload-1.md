# Handoff 03 — validation-diagnostics upload #1 of 5: the M1 single-LightGBM baseline (`models/test01_lgbm.pkl`)

> ## USER MUST APPROVE
> This handoff makes **one platform write**: it uploads a predictions file for the validation split to the free
> validation-diagnostics lane. That uses **upload #1 of the 5 pre-registered diagnostics uploads** (M1, M3, M5 x2,
> final; docs/DECISIONS.md #4). The upload cannot be undone.
>
> The AI runs step 5 **only after the user replies `yes, upload` in this chat**, directly after seeing the step 3
> preflight block. DECISIONS #9 approved this handoff in advance, but that does not replace the reply in chat.
> Nothing else is written to the platform: no pickle upload to the auto-run lane, no round submission, no staking,
> no model created or renamed, and auto-submit stays off.

## 1. Purpose + success criterion

The M1 baseline is the single LightGBM member `lgbm_everest` of handoff 02's run `test01` (OOF fit20 0.0238,
forward 0.0313, Sharpe 0.51). Handoff 02 showed that the 10-member equal-weight blend in `models/test01.pkl` is
worse (OOF 0.0226), so that pickle is **not** uploaded. Step 1 bundles `lgbm_everest` alone from the 02 artifacts
into `models/test01_lgbm.pkl` (`bundle --members lgbm_everest`, same config constants `p_b` 0.5 / `p_c` 0.25 as
test01.pkl) and checks it with `pkl_check`. The pickle predicts every validation era
(1,392 expeds, 125,931 ids), with the validation benchmark passed in exactly as the lane would pass it. Those
predictions are scored server-side against the hidden answers. The panel that comes back (fit20 / unq / inov /
sharpe) is **informational only** (PLAN C13):
- It records the OOF -> validation fit ratio.
- It shows whether INOV is positive against the true core set.
- It never tunes `p_b` / `p_c` and never selects members.

Success = ALL of:
- Step 2 writes `artifacts/diag/test01_lgbm_validation.parquet` with **125,931 rows / 1,392 eras**. The step 3 preflight
  shows `rows 125931/125931 eras 1392/1392 -> READY` and ends `PREFLIGHT OK`.
- The user's approval reply is quoted verbatim in the operator notes. Without it, the handoff ends after step 3 with
  "not approved", which is a valid outcome.
- Step 5 logs `DIAGNOSTICS UPLOADED ... upload_id=<id>` once, then `diagnostics done: fit20=... unq=... inov=...
  sharpe=...`, and exits 0. If it exits 2 instead, step 6 brings it to exit 0.
- `handoffs/results/03-results.md` has a **Validation diagnostics** section containing:
  - `upload_id`;
  - the panel;
  - `diag fit20 / OOF fit20`.
- The same report also carries the `blend_vs_lgbm_everest` paired row (`p_pos`) that 02 could not record. The
  reporter now recomputes it from `artifacts/test01/oof.parquet` and labels it as recomputed.

For context (not gates): `lgbm_everest` has OOF fit20 0.0238 and forward 0.0313 (the EW blend: 0.0226 / 0.0284).
The report compares the diag fit20 with `lgbm_everest`, found through the pickle's manifest. The validation board's fit20
quartiles are 0.0167 / 0.0194 / 0.0214. A diag fit20 anywhere between about 0.010 and 0.030 is unremarkable. Record
the number; do not act on it.

## 2. Preconditions

- Handoffs 00–02 are reported on `main`.
  - 02 is accepted.
  - The 02 operator note "blend_ew versus lgbm_everest paired p_pos is absent" was a reporter gap, not a run failure.
    This handoff's report fills it in. Nothing from 02 is re-run.
- `git -C "$REPO" pull` brings in this handoff and the M2–M6 code. `.venv` is unchanged (same pins).
- These handoff-02 outputs are still on this laptop, untouched (they are gitignored, so a pull does not remove them):
  `artifacts/test01/` (`metrics.json`, `blend.json`, `oof.parquet`, `features.json`, `members/lgbm_everest/`).
  `models/test01.pkl` (the 10-member blend) is not used by this handoff.
- `models/test01.manifest.json` and `models/test01.pkl.sha256` (handoff 02) stay as they are; never edit them.
- `.env` is present with `EIQ_API_KEY` (`test -s .env`). `EIQ_MODEL_ID` is checked by the preflight (step 3).
- The laptop is online. The upload itself is a few MB, and scoring takes seconds to ~20 min.

## 3. HUMAN STEP (the human does this, not the AI)

**Model slot (only if the step 3 preflight prints `model_id MISSING`).** The diagnostics upload is filed under a
platform model slot. That slot is the `EIQ_MODEL_ID` value in `.env`.
- The human chooses it: a slot they already own, or a new one they create themselves on the Everesteer website.
- The human writes the line `EIQ_MODEL_ID=<slot name or id>` into `.env` with an editor.
- The AI never creates, renames or archives models. It never prints, cats, greps or sources `.env`. It verifies only
  by re-running the step 3 preflight, which prints the resolved slot name.

The AI also stops and asks the human if `test -s .env` fails (handoff 00 section 3).

## 4. Environment

```bash
REPO="$HOME/stealth"                    # absolute clone path: use the SAME clone that ran handoffs 00-02
PY="$REPO/.venv/bin/python"             # absolute venv interpreter used by EVERY command
cd "$REPO"
git log -1 --oneline                    # note the commit in the operator notes
"$PY" -m stealth.selfcheck | tail -3    # expect PINS OK
"$PY" -m pytest -q 2>&1 | tail -3       # new code since 02: expect "N passed" (", K skipped" is fine); <= 5 min
test -s .env && echo ".env present" || echo ".env MISSING -> HUMAN STEP"
```
If pytest reports `failed` or `error`, stop and go to section 7. Copy the tail into the operator notes.

## 5. Steps

**Step 1 — bundle the single-LightGBM baseline from the 02 artifacts and check it (< 2 min; no network)**
```bash
"$PY" -m stealth.bundle --run-id test01 --out models/test01_lgbm.pkl --members lgbm_everest
"$PY" -m stealth.pkl_check models/test01_lgbm.pkl          # expect pkl_check: PASSED, exit 0
"$PY" -c "import json; m=json.load(open('models/test01_lgbm.manifest.json')); n=[x['name'] for x in m['members']]; ok=n==['lgbm_everest'] and m['p_b']==0.5 and m['p_c']==0.25 and m['pkl_check']['passed'] is True; print('BASELINE OK' if ok else 'BASELINE MISMATCH', n, m['p_b'], m['p_c'], m['p_source'], m['sha256'][:16])"
```
Expected:
- bundle: one WARNING that the other members' blend weights are ignored (expected), then the last line
  `bundle ok: run=test01 members=['lgbm_everest'] p_b=0.50 p_c=0.25 p_source=config rule=ew sha256=...`;
- `pkl_check: PASSED` and `byte scan: torch=0 ... non-runtime stealth refs=0`;
- `BASELINE OK ['lgbm_everest'] 0.5 0.25 config <sha16>`. Record the sha16 in the operator notes.
Sandbox analogue (an M1 smoke run): 0.23 MB pickle, pkl_check PASSED in seconds, validation file in 6 s.
Anything else -> section 6, first bullet.

**Step 2 — build the validation predictions file (1–5 min; no network; idempotent)**
```bash
"$PY" -m stealth.diag --pkl models/test01_lgbm.pkl --out artifacts/diag/test01_lgbm_validation.parquet
```
Expected output signature:
```
INFO stealth.diag: diag: 1392 eras, 125931 rows, bench=True
INFO stealth.diag: diag progress 200/1392 eras (...s, 0.0Xs/era, bench_used=200)
...
INFO stealth.diag: wrote .../artifacts/diag/test01_lgbm_validation.parquet (125931 rows, 1392 eras) + test01_lgbm_validation.parquet.meta.json
INFO stealth.diag: diag ok: 125931 rows -> artifacts/diag/test01_lgbm_validation.parquet
```
Exit 0. Record the `s/era` figure from the last progress line.

**Step 3 — preflight (seconds; offline; sends nothing)**
```bash
"$PY" -m stealth.diag --pkl models/test01_lgbm.pkl --out artifacts/diag/test01_lgbm_validation.parquet --preflight
```
Expected (exit 0):
```
DIAG PREFLIGHT (offline; nothing is sent)
  model_id        <your slot>
  pkl             test01_lgbm.pkl sha256 <sha16 of step 1>  manifest match, pkl_check passed, python 3.11.x, run test01
  file            test01_lgbm_validation.parquet rows 125931/125931 eras 1392/1392  -> READY: full coverage, built from this pickle
  previous upload none
  budget          each diagnostics upload consumes one of the 5 pre-registered uploads (docs/DECISIONS.md #4)
PREFLIGHT OK -- ask the user for approval in chat
```
- `model_id MISSING` -> HUMAN STEP (section 3), then re-run step 3.
- `ALREADY UPLOADED upload_id=...` -> do **not** upload. Go to step 6 to read that upload's result.
- Any other `NOT READY` / `MISMATCH` / `NOT passed` -> stop and go to section 7.

**Step 4 — APPROVAL GATE (the user, in this chat)**

Paste the whole preflight block to the user and ask, word for word:

> Handoff 03 will upload `artifacts/diag/test01_lgbm_validation.parquet` (125,931 validation predictions of the M1
> single-LightGBM baseline, `lgbm_everest` of run test01) to the validation-diagnostics lane under model slot `<model_id>`. This is diagnostics upload #1
> of the 5 pre-registered uploads and cannot be undone. No pickle is uploaded, nothing is submitted to a round,
> nothing is staked. Reply `yes, upload` to proceed.

Continue to step 5 only after an explicit `yes, upload` (or an equally unambiguous yes) **in this chat**. Any other
answer, no answer, or a question means **do not upload**. Write "not approved: <what the user said>" in the operator
notes and go to section 7.

**Step 5 — the upload (USER-GATED; one platform write; 1–30 min)**
```bash
STEALTH_CONFIRM_DIAG=1 "$PY" -m stealth.diag --pkl models/test01_lgbm.pkl --out artifacts/diag/test01_lgbm_validation.parquet --upload --i-have-approval --wait
```
Expected:
```
INFO stealth.diag: reusing artifacts/diag/test01_lgbm_validation.parquet (full coverage, built from this pickle)
WARNING stealth.diag: =========  USER-GATED PLATFORM WRITE: UPLOAD VALIDATION DIAGNOSTICS ... =========
WARNING stealth.diag: DIAGNOSTICS UPLOADED model=<slot> file=test01_lgbm_validation.parquet rows=125931 upload_id=<id> status=... attached_pkl=False
INFO stealth.diag: diagnostics running (queue=... elapsed=... est=...); waiting 5s
INFO stealth.diag: diagnostics done: fit20=0.0... unq=0.0... inov=0.0... sharpe=0.... (upload_id=<id>)
```
It writes `artifacts/diag/test01_lgbm_validation.parquet.upload.json`, which holds `upload_id` and the panel.
- **Exit 0**: done.
- **Exit 2**: the file WAS uploaded but is still scoring after 30 min. Never run step 5 again; use step 6.
- **Exit 1**: read the last ERROR line and follow section 6.

**Run step 5 exactly once.** A second run is refused anyway, because the `.upload.json` exists. Never pass
`--new-upload` or `--attach-pkl`.

**Step 6 — only if step 5 exited 2, or step 3 said ALREADY UPLOADED: read the result (read-only)**
```bash
"$PY" -m stealth.diag --out artifacts/diag/test01_lgbm_validation.parquet --fetch-result --wait
```
Read-only: no approval and no environment variable are needed.
- Exit 0 with `diagnostics done: fit20=...` -> finished.
- Exit 2 -> still scoring. Repeat every ~10 min. After 2 h, stop and report.
- Exit 1 -> the run failed server-side: record the ERROR line.

## 6. If it fails

- **Step 1 not OK.**
  - `bundle` fails (for example `artifacts/test01/members/lgbm_everest` is missing): stop and report. Do not re-run
    handoff 02, and never upload `models/test01.pkl` instead.
  - `pkl_check: FAILED` or `BASELINE MISMATCH`: stop and report. Never upload an unchecked pickle.
- **Step 2 error** (traceback, `KeyError` on a feature column): stop and report. Do not edit code.
- **Step 5 `MODEL PKL REQUIRED`**: nothing was stored. Stop and report. Claude decides whether a later approved run
  uses `--attach-pkl`. Do not retry with it now.
- **Step 5 `PermissionError ... refused`**: the `STEALTH_CONFIRM_DIAG=1` prefix or `--i-have-approval` is missing.
  Nothing was sent. Fix the command exactly as written above, still within this one approval.
- **Step 5 error with no `DIAGNOSTICS UPLOADED` line** (401/403, network, timeout): treat it as *possibly
  received*. Do **not** retry. Stop and report: Claude checks the platform read-only.
- **Step 5 error after `DIAGNOSTICS UPLOADED` appeared**: the upload exists. Use step 6, never step 5.
- **Any "run already in flight for this model" message**: stop and report.

## 7. Report back (mandatory, also on failure and on "not approved")

```bash
"$PY" -m stealth.report --run-id test01 --handoff 03 --diag-json artifacts/diag/test01_lgbm_validation.parquet.upload.json
```
This writes `handoffs/results/03-{results.md,metrics.json,log-tail.txt}`, then
`git add handoffs/results && git commit -m "results: 03 diag-upload-1" && git push`. Exit 0 = pushed.
- With no upload, the diagnostics section says `no upload recorded`. That is expected; run the report anyway.
- On **PUSH FAILED — results are saved locally in handoffs/results/; ask the human to run git push**: repeat it to
  the user verbatim and stop.
- On **GIT COMMIT FAILED — …**: repeat it verbatim and stop.

Then append the operator notes and push:
```bash
printf '\n## Operator notes\n\n```text\n%s\n```\n' "<commit: <sha>; pytest: N passed; step 1: BASELINE OK <sha16>, pkl_check PASSED; step 2: 125931 rows, N s/era, M min; preflight: <paste the block>; approval: \"<user's words>\" at <time>; upload_id: <id>; panel: fit20 x unq y inov z sharpe s; diag/OOF ratio r; wall: MM min; deviations: none>" >> handoffs/results/03-results.md
git add handoffs/results && git commit -m "results: 03 diag-upload-1 (notes)" && git push
```
Do not commit `artifacts/diag/*`; it is gitignored. Do not commit `models/test01_lgbm.*` and do not edit `models/test01.manifest.json`.

## 8. Rules for the executing AI

- This handoff has a **USER MUST APPROVE** banner for **step 5 only**, and only after the user's `yes, upload` in
  this chat. Exactly one upload; never `--new-upload`; never `--attach-pkl`.
- Never `predict --submit`, never `predict --upload`, never stake, never create, rename or archive a model, never
  change auto-submit. No platform write other than step 5.
- No code or config edits. Stop and write the results file on any judgement call.
- Never commit `data/`, `.env`, `*.pkl`, `artifacts/`. Only `handoffs/results/`.
- Never `cat`/`echo`/`grep`/`source` `.env`; never `git remote -v`; never `git config --list`; never print the API key.
- Do not run any step for more than 3x its stated time without telling the user.
