# Handoff 04 — full run (`configs/full.yaml`): the M2–M6 model with a reserved audit (no upload)

> **No USER MUST APPROVE banner.** Nothing is uploaded, submitted or staked. No model is created or renamed and
> auto-submit is never touched. The validation diagnostics **file** is built and **not** uploaded. The next
> diagnostics upload and the lane upload are later, separate USER MUST APPROVE handoffs.

## 1. Purpose + success criterion

One resumable pipeline on the laptop (Linux, 20 logical threads, 16 used; RTX 5070 Ti; ~30.8 GB RAM;
Europe/London):
1. **Development** (eras 2800–5924 only): the training-universe knobs (decided once on the dev forward block
   5340–5924, Holm), LightGBM TPE search (≤ 25 trials) with the top-k adoption gate, 12 fixed XGBoost cells, 3 CatBoost
   configs, 6 aux targets × 2 families, ridge, the |sharpe| subset arm, 4 feature bags, the MLP on the GPU. Then the
   pre-registered admission ladder, equal weights vs purged nested NNLS, and the regime grid (1,000 draws) that freezes
   `p_b` / `p_c`.
2. **Reserved audit** (eras 5945–6524): no fit, early stop or choice of this run reads these eras. Full refits stop
   at 5924 too. After the blend is frozen, `stealth.audit` scores it once against an unsearched baseline (LightGBM at
   its config params, default universe).
3. Bundle with `--require-frozen`, `pkl_check` in the clean lane venv, and the diagnostics **file**.

Every development number (OOF, forward, knobs, search, ladder, NNLS, neutralisation) is a **selection estimate**: the
same eras chose and scored, so it is optimistic. The audit is the only number that no choice of this run has seen.
It is still not "fresh": handoff 02 looked at those eras.

**Technical success** = ALL of the following, read from `handoffs/results/04-metrics.json` and the step outputs:
- The pipeline ends with `PIPELINE OK run_id=full01 ...`, exit 0. `04-metrics.json` `status` is `ok` (the reporter
  derives it from the evidence) and `leak_check` is `pass`.
- `knobs.status` is `done`, with a `chosen` key and one candidate row per alternative.
- `search` has rows for `lgbm_everest` (`n_trials` ≤ 25) and `xgb_everest`, each with `adopted` true or false.
- `ladder` has one row per arm present, each `admitted` true or false with a reason.
- `neutralisation.p_source` is **`simulation`**, `p_b` ≤ 0.75 and `p_c` ≤ 0.5. The manifest agrees (step 7).
- The audit section shows `status ok`, partition 5945..6524 with 580 eras, and a verdict.
- `pkl_check: PASSED` on the pickle's own sha256, file < 50 MB, slowest call < 5 s.
- `artifacts/diag/full01_validation.parquet` has 125,931 rows and 1,392 eras, and is **not** uploaded.
- `04-results.md`, `04-metrics.json`, `04-log-tail.txt`, `models/full01.manifest.json` and `models/full01.pkl.sha256`
  are committed and pushed.

**Audit verdict (informational, never a pass/fail of this handoff):** `audit.verdict.audit_win` true means the frozen
blend beat the unsearched baseline on the reserved eras (paired p_pos ≥ 0.8). False means a tie or a loss.
Report it verbatim either way. Technical completion is not an audit win, and a dev-block gain is not one either.

Expected ranges (selection estimates; for sanity only):
- Final OOF fit20 roughly 0.020–0.028.
- The run itself prints a SUSPECT banner when the blend beats the in-run unsearched LightGBM base cell by more than
  0.004. Report it as suspect, never as a win.
- Handoff 02's single LightGBM scored 0.0313 on the eras that are now the audit, a strong regime.

## 2. Preconditions

- Handoffs 00–03 are done on this laptop. 03 is accepted; **do not repeat its upload**.
- The same clone as 00–03, after `git pull`. `.venv` and `data/` are present (handoff 00).
- ~6 GB free disk (torch wheels ~2.5 GB; artifacts ~1–2 GB). The laptop is on mains power and set not to sleep for
  3 hours. No other heavy job is running: the run uses 16 threads and up to ~8 GB RAM.

## 3. HUMAN STEP

None, unless `test -s .env` fails (handoff 00). The run is offline apart from the torch download in step 2; it never
reads `.env`.

## 4. Environment

Run every block in ONE bash shell, in order. `pipefail` makes a failing command fail its pipe.
```bash
set -o pipefail
REPO="$HOME/Projects/eversteer/stealth"   # the SAME clone that ran handoffs 00-03; change if different
PY="$REPO/.venv/bin/python"               # train venv: EVERY stealth command
GPU_PY="$REPO/.venv-gpu/bin/python"       # GPU venv (step 2): the MLP member ONLY
cd "$REPO" && git pull && git log -1 --oneline
"$PY" -m stealth.selfcheck > /tmp/04-selfcheck.txt 2>&1; echo "selfcheck exit $?"; tail -3 /tmp/04-selfcheck.txt
nproc; free -g | head -2
```
Expected: `selfcheck exit 0` with `PINS OK`, `nproc` 20.

## 5. Steps

**Step 1 — threads (seconds; the only permitted edit)**

`configs/full.yaml` has `run.threads: 16`. With `nproc` ≥ 16 (this laptop: 20) change nothing. Only if `nproc`
printed less than 16, edit that single line to the `nproc` value, and afterwards restore only that line. Then
`git diff --stat configs/full.yaml` must show no change before you commit anything.

**Step 2 — GPU venv for the MLP (5–15 min the first time, seconds afterwards; idempotent)**
```bash
bash scripts/setup_gpu.sh > /tmp/04-gpu.txt 2>&1; GPU_RC=$?; echo "setup_gpu exit $GPU_RC"; tail -5 /tmp/04-gpu.txt
```
It installs torch from the CUDA 12.8 wheel index into `.venv-gpu`. This is never the train venv, and torch never
reaches the pickle. It ends with a GPU matmul smoke test.
- Exit 0 with last line `GPU OK`: use **step 4a**.
- Exit 3 (`GPU FAIL -> skip MLP`), exit 1 (`SETUP ERROR`), or any other exit: use **step 4b**. Write
  "MLP skipped: <last 5 lines of /tmp/04-gpu.txt>" in the operator notes.

**Step 3 — test suite (2–6 min)**
```bash
if [ "$GPU_RC" -eq 0 ]; then export STEALTH_GPU_PY="$GPU_PY"; fi
"$PY" -m pytest -q -p no:cacheprovider > /tmp/04-pytest.txt 2>&1; echo "pytest exit $?"; tail -3 /tmp/04-pytest.txt
```
Expected: `pytest exit 0` and `N passed` (a few skips are fine). On a non-zero exit, stop, copy the last 30 lines of
`/tmp/04-pytest.txt` into the operator notes, and go to section 7.

**Step 4 — start the pipeline in the background (seconds). Run exactly ONE of 4a / 4b.**

It survives a closed terminal or a tool timeout: you poll it in step 5. Every stage resumes from its cache.

4a — GPU OK (step 2 exit 0):
```bash
mkdir -p artifacts
nohup bash scripts/run_pipeline.sh configs/full.yaml full01 --gpu-py "$GPU_PY" > artifacts/full01_pipeline.out 2>&1 &
echo $! > artifacts/full01_pipeline.pid; echo "started pid $(cat artifacts/full01_pipeline.pid)"
```

4b — no usable GPU (any other step 2 result):
```bash
mkdir -p artifacts
nohup bash scripts/run_pipeline.sh configs/full.yaml full01 --no-mlp > artifacts/full01_pipeline.out 2>&1 &
echo $! > artifacts/full01_pipeline.pid; echo "started pid $(cat artifacts/full01_pipeline.pid)"
```
The MLP defaults to `--device cuda`. If CUDA stops working mid-run, the MLP step exits 4, the pipeline logs
`MLP skipped: ...` and continues without it. It never silently trains on the CPU.

**Step 5 — poll until it finishes (every 5–10 minutes; ~1–2 h in total)**
```bash
tail -n 3 artifacts/full01/pipeline.log; tail -n 1 artifacts/full01/run.log
"$PY" -c "import json; d = json.load(open('artifacts/full01/pipeline.json')); print('STATUS', d['status'], '| step', d.get('last_step'), '| exit', d.get('exit_code'), '| mlp', d.get('mlp'))"
```
Repeat while it prints `STATUS running`. Stop polling at `STATUS ok`, `failed` or `interrupted`. The steps, in order:
1. `train` (`--skip-external`)
2. `mlp` (4a only)
3. `retrain` (assembles the MLP after it finished, `--require-external`)
4. `blend`
5. `audit`
6. `bundle --require-frozen`
7. `pkl_check`
8. `diag` (builds the file only)

Each logs `step <name>: start` / `exit 0 after N s` in `artifacts/full01/pipeline.log`. Each step's full output is in
`artifacts/full01/pipeline_steps/`.

Estimated wall time on 16 threads, scaled from handoff 02 (training 497 s, regime simulation 208 s):

| part | estimate | what you see |
|---|---|---|
| train: knobs (6 cells), audit baseline, LGBM ≤25 TPE trials + base, XGB 12, 12 aux, 3 CatBoost, ridge, sharpe, 4 bags, full refits ×2 seeds, audit predictions | 25–45 min | `knobs: ... chosen <key>`, `search lgbm_everest: trial N`, `progress: member i/N ...` every 60 s |
| mlp: 5 seeds × (5 folds + forward + full) on the GPU | 5–25 min | `seed s foldK: inner best epoch ...`, `parity ...` |
| retrain: re-assemble with the MLP | 2–5 min | `external member mlp_everest: loaded` |
| blend: ladder, NNLS, regime grid (1,000 draws × 12 cells × 3 means) | 10–25 min | **can print nothing for ~15 min** during the grid |
| audit: frozen blend on the reserved eras + 1,000-draw scenario | 1–3 min | `audit full01 eras 5945..6524: blend fit20 ...` |
| bundle + pkl_check (51 eras) | 1–5 min | `pkl_check: PASSED` |
| diag file (1,392 eras) | 2–8 min | `diag progress ...` |
| **total** | **~50–120 min** | `PIPELINE OK run_id=full01 ... mlp: ok` |

Tell the user, and keep it running, if the total passes 4 h or a step passes 3× its upper estimate while progress
lines still appear.

**Step 6 — the result line (seconds)**
```bash
tail -n 2 artifacts/full01/pipeline.log; grep -E 'step .*: exit|MLP skipped|PIPELINE' artifacts/full01/pipeline.log
```
Expected: `PIPELINE OK run_id=full01 ...`. A failure ends with `PIPELINE FAILED at step <name> (exit N)`; an
interruption ends with `PIPELINE INTERRUPTED ...`. Both go to section 6.

**Step 7 — packaging evidence (seconds; only after PIPELINE OK)**
```bash
"$PY" -c "import json; m = json.load(open('models/full01.manifest.json')); c = m.get('pkl_check') or {}; print('p_source', m.get('p_source'), '| p_b', m.get('p_b'), '| p_c', m.get('p_c'), '| rule', m.get('blend_rule'), '| arms', m.get('admitted_arms'), '| MB', round(m['size_bytes'] / 1e6, 2)); print('pkl_check passed', c.get('passed'), '| on these bytes', c.get('sha256') == m.get('sha256'), '| slowest s', c.get('seconds_per_era_max'), '| isolation', c.get('isolation'))"
"$PY" -c "import json; a = json.load(open('artifacts/full01/audit.json')); p = a.get('paired_vs_baseline') or {}; print('audit', a['status'], a['partition']['eras'], '| blend fit20', round(a['blend']['mean'], 4), '| baseline', round((a.get('baseline') or {}).get('mean', float('nan')), 4), '| diff', p.get('diff_mean'), '| p_pos', p.get('p_pos')); print('verdict:', a['verdict']['text'])"
```
Expected: `p_source simulation`, `p_b` ≤ 0.75, `p_c` ≤ 0.5, `pkl_check passed True | on these bytes True`; audit `ok`
`[5945, 6524]` and a verdict line (either outcome is fine).

**Step 8 — offline dry run (seconds; nothing is sent)**
```bash
"$PY" -m stealth.predict --pkl models/full01.pkl --offline 2>&1 | tail -3
```
Expected: `predicted exped=... n=90 ...` then `dry run ok`. **No `--submit`, no `--upload`.**

## 6. If it fails

- **Interrupted** (reboot, crash, `STATUS interrupted`): re-run the SAME step 4 command (4a or 4b, whichever you
  used). Train resumes from its caches and the MLP from `members/mlp_everest/_work/`; later steps repeat. To stop it
  yourself: `kill -TERM "$(cat artifacts/full01_pipeline.pid)"`. That stops the running step's whole process group
  and records the attempt as interrupted. Never `kill -9`.
- **`run-id full01 was created with a different config/data/protocol`, or `holds artifacts without a run
  fingerprint`**: an older `artifacts/full01` exists. Stop and report it. Do not delete it, and do not pick another
  run-id on your own.
- **`MemoryError` / OOM-killed**: set `run.threads` to 8 (the step 1 edit), re-run the same step 4 command, and
  restore the line afterwards.
- **train or blend exit 2 (`LEAK`)**: the run is void. Report and stop.
- **`knobs already decided for this run-id`**: the config changed under `full01`. Stop and report. Never delete
  `knobs.json`.
- **`--require-frozen` refusal at bundle** (`p_source` is not `simulation`): stop and report. Never pass `--p-b` or
  `--p-c`.
- **The MLP step fails** with exit 1 or 4: the pipeline continues on its own. Note "MLP skipped (exit N)" and the last 20
  lines of the newest `artifacts/full01/pipeline_steps/*-mlp.log`.
- **Any step exits ≥ 128**: it was interrupted. Re-run the same step 4 command.
- **`pkl_check` FAIL** or any other step failure: copy the step's last 30 lines (its `pipeline_steps/*-<step>.log`)
  into the operator notes. Fix nothing.
- **No new line in `artifacts/full01/run.log` or `pipeline.log` for > 30 min** outside the blend's regime grid: note
  the time, `kill -TERM` as above, and re-run the same step 4 command (it resumes).

## 7. Report back (mandatory, also on failure)

```bash
"$PY" -m stealth.report --run-id full01 --handoff 04
```
This writes `handoffs/results/04-{results.md,metrics.json,log-tail.txt}` and commits and pushes them. The status comes
from the pipeline, manifest, pkl_check, diag-file and audit evidence (`failed` / `incomplete` when one is missing).
The results carry these sections:
- pipeline attempts and step times;
- knobs, search adoption and admission ladder (labelled selection estimates);
- member correlation;
- the neutralisation table with `p_b` / `p_c` / `p_source`;
- final selection;
- packaging evidence and diag-file coverage;
- the audit and its verdict;
- dropped members and wall times.

On **PUSH FAILED — …** or **GIT COMMIT FAILED — …**, repeat the message verbatim to the user and stop.

Then append the operator notes and push them. Write only what happened; leave a field `n/a` rather than guess.
```bash
printf '\n## Operator notes\n\n```text\n%s\n```\n' "<commit: <git log -1 --oneline>; GPU: OK|FAIL (<step 2 tail>); pytest: <N passed / failed>; MLP: ok|skipped (why); step exits + seconds: <step 6 grep>; step 7: <paste both lines>; step 8: <ok|error>; total wall: H h MM min; nproc: N, threads: 16; deviations: none|<what>>" >> handoffs/results/04-results.md
git add handoffs/results && git commit -m "results: 04 full-run (notes)" && git push
```
Commit the manifest and its sha only if step 7 showed `pkl_check passed True | on these bytes True`. Never commit the
pickle.
```bash
git add models/full01.manifest.json models/full01.pkl.sha256 && git commit -m "models: full01 manifest" && git push
```
Results files carry metrics and config/arm NAMES only. Hyperparameter values stay in `artifacts/`.

## 8. Rules for the executing AI

- No code edits. The only permitted config edit is `run.threads` in `configs/full.yaml` (step 1); restore it
  afterwards.
- This handoff has **no USER MUST APPROVE banner**:
  - never `diag --upload`, never `predict --submit` / `--upload`, never stake;
  - never create, rename or archive a model, never change auto-submit;
  - never repeat handoff 03's upload. `run_pipeline.sh` builds the diagnostics file only.
- Never commit `data/`, `.env`, `*.pkl`, `artifacts/` or `.venv*`. Commit only `handoffs/results/` and
  `models/full01.{manifest.json,pkl.sha256}`.
- Never `cat`/`echo`/`grep`/`source` `.env`; never `git remote -v`; never `git config --list`; never print the API key.
- Stop and write the results file on any judgement call. Do not run a step for more than 3x its stated time without
  telling the user.
