# Handoff 05 — optional diagnostics upload #2: the frozen full01 challenger

> **USER MUST APPROVE.** This handoff authorises preparation only until the human explicitly approves this
> particular diagnostics upload in chat. It consumes upload **2 of 5**. It never submits live predictions, stakes,
> changes auto-submit, or uploads a model pickle. Do not repeat handoff 03.

## 1. Decision and scope

The full01 blend did **not** beat its unsearched audit baseline: audit FIT 0.0266 versus 0.0308, paired p_pos
0.0412. The deployment candidate remains the unsearched baseline. This optional upload measures the frozen
challenger on the platform's existing validation lane; its purpose is calibration, not automatic promotion.

Do not rebuild, refit, change weights, or tune neutralisation in this handoff. Do not search on the returned
validation scores. They become additional selection evidence as soon as they are inspected, not a fresh holdout.
There is no requirement to spend this upload before doing more development-only stability work.

## 2. Preconditions and environment

Use the SAME clone as 00–04. All Python commands use its absolute train interpreter.

```bash
set -o pipefail
REPO=/home/om/Projects/eversteer/stealth
PY="$REPO/.venv/bin/python"
cd "$REPO"
git pull --ff-only
test -s .env && echo '.env present'
```

If `.env` is missing, stop at the human step from handoff 00. Never display, grep, source, or export it.

## 3. Prepare and verify locally

The approved artifact is `models/full01.pkl`, SHA256
`c693300d81d1ff102cf684b8451836355ba82b216005ae42977dcf2c6fc3ab03`.
The predictions file is `artifacts/diag/full01_validation.parquet`, SHA256
`1c2ab0af9bb51332bbabfb1c6d493127ae8b91db2ebcc93ff2b940e12fd3d3e2`.

```bash
"$PY" - <<'PY'
import hashlib, json
from pathlib import Path
pkl = Path('models/full01.pkl')
pred = Path('artifacts/diag/full01_validation.parquet')
assert hashlib.sha256(pkl.read_bytes()).hexdigest() == 'c693300d81d1ff102cf684b8451836355ba82b216005ae42977dcf2c6fc3ab03'
assert hashlib.sha256(pred.read_bytes()).hexdigest() == '1c2ab0af9bb51332bbabfb1c6d493127ae8b91db2ebcc93ff2b940e12fd3d3e2'
record = pred.with_name(pred.name + '.upload.json')
if record.exists():
    previous = json.loads(record.read_text())
    raise SystemExit('STOP: existing upload record; inspect upload_id and fetch its result, never upload again')
print('Frozen artifact hashes match; no earlier upload record')
PY
"$PY" -m stealth.pkl_check models/full01.pkl
"$PY" -m stealth.diag --pkl models/full01.pkl --out artifacts/diag/full01_validation.parquet --preflight --model-id mandelbrot-bayesian-payoff
```

Require exit 0 throughout, `pkl_check: PASSED` with exact lane pins and import isolation, and `PREFLIGHT OK`:
125,931 rows / 1,392 eras, full coverage, no previous upload. Any discrepancy means stop and report; do not
regenerate predictions or substitute `--members lgbm_everest` or the separately exported audit baseline.

## 4. Human decision

Ask: **Approve diagnostics upload #2 of 5 of the unchanged full01 challenger to
`mandelbrot-bayesian-payoff`? Its audit did not beat the baseline. This is validation calibration only, with no
live submission, staking, or automatic promotion. (yes/no)**

An explicit yes covering this file and this upload is required. Otherwise report `upload not approved` and stop.

## 5. One upload, after approval

```bash
STEALTH_CONFIRM_DIAG=1 "$PY" -m stealth.diag --pkl models/full01.pkl --out artifacts/diag/full01_validation.parquet \
  --upload --i-have-approval --wait --model-id mandelbrot-bayesian-payoff
```

Never add `--new-upload` or `--attach-pkl`. Never re-run this command after an upload ID is returned. If the
request fails after it might have reached the server but before an ID is recorded, stop and report the uncertainty;
do not retry an ambiguous write. If waiting times out, use only the read-only result path:

```bash
"$PY" -m stealth.diag --out artifacts/diag/full01_validation.parquet --fetch-result --wait
```

## 6. Report and stop (also on failure)

Write `handoffs/results/05-results.md` with status, code commit, the two artifact hashes, checker/preflight results,
rows/eras, upload ID (or none), completion state, every returned numeric panel metric, wall time and exact errors.
Compare descriptively with diagnostics #1: FIT 0.0189, UNQ 0.0039, INOV 0.014054225, Sharpe 0.3237. The models and
training protocols differ; do not call that comparison independent proof of an upgrade. Preserve the audit verdict.

```bash
git add handoffs/results/05-results.md
git commit -m 'results: 05 diag-upload-2' && git push
git ls-remote origin main
```

Never commit the pickle, data, `.env`, artifacts, or the upload JSON. On commit/push failure, report the failure and
stop. No live refit, lane upload or auto-submit step follows this handoff. M7 needs its own concrete protocol and
separate authorisation; do not improvise it after seeing the diagnostics result.
