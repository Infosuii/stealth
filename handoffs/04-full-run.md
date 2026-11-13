# Handoff 04 — full run (`configs/full.yaml`): the M2–M6 model, end to end (no upload)

> **DRAFT — Claude finalises it from handoff 02 results. Do not run it until `handoffs/INDEX.md` marks 04 `ready`.**
> This handoff has **no USER MUST APPROVE banner**. Nothing is uploaded, submitted or staked. The validation
> diagnostics file is built but **not** uploaded. The next diagnostics upload and the lane upload are later,
> separate USER MUST APPROVE handoffs.

### Finalisation checklist (Claude; delete this block when marking 04 ready)

| item | from handoff 02 (`handoffs/results/02-*`) | action |
|---|---|---|
| CatBoost keep/drop | `cat_everest_l2_30` took 18.1 s for the whole member (fold fits ~3 s), well under the 120 s cap | **kept**: `full.yaml` keeps the 3 `cat_*` members; `timing.catboost_max_fit_seconds: 120` still enforces `dropped_timing` |
| `gates.baseline_fit` | `lgbm_everest` OOF fit20 **0.0238** (top-3 of the 12-cell grid; its base cell was ≈ 0.0229) | DONE (integration): `gates.baseline_fit: 0.0229` in `full.yaml` = the UNSEARCHED base cell (build note 2: a searched baseline can never trip its own SUSPECT bar); SUSPECT bar 0.0269 |
| thread count | `nproc` 20, `run.threads` 16, RAM 30.8 GB, no memory pressure | keep `run.threads: 16` |
| per-fit timings | train 497 s in total (LGBM ≈ 20 s per grid cell incl. 5 folds + forward; aux ≈ 16 s, XGB 11–17 s, ridge 2 s); regime simulation 208 s | replace the estimates in section 5 with these scaled numbers (done below; refresh them after the minifull run) |
| compute budget (DECISIONS #10) | 02 ran 3–4x faster than estimated | decide any extra measurement (bootstrap draws, fold offsets, slower trees, admission candidates) in `full.yaml` **before** `ready`; never more trials or seed bagging |
| sandbox proof | `bash scripts/run_pipeline.sh configs/minifull.yaml <id>` green on the sandbox, plus the full `pytest` | green at integration 2026-11-13: run `mfpipe`, 194 s end to end incl. the MLP on CPU torch (`setup_gpu.sh` with the CPU index: exit 3 as expected); record the merge commit here |

## 1. Purpose + success criterion

This is the final multi-hour run (milestones M2–M6) on the laptop CPU, plus the optional MLP on the GPU. In order:
1. Training-universe knobs are decided once on the forward block (Holm).
2. LightGBM TPE search (≤ 25 trials) with the top-k adoption gate; 12 fixed XGBoost configs (top-2 averaged, same
   gate); 3 CatBoost configs; 6 aux targets × 2 families; ridge; the |sharpe| subset arm; 4 feature bags; the MLP
   member.
3. The pre-registered admission ladder, then EW vs nested NNLS.
4. The regime grid (p_b × p_c, 500 draws) and the staking-aware choice of the neutralisation constants.
5. Bundle with `--require-frozen`, `pkl_check`, and the validation diagnostics **file**.

Success = ALL of (read from `handoffs/results/04-metrics.json`):
- The pipeline ends with `PIPELINE OK run_id=full01`; exit 0; `leak_check` = `pass`.
- `knobs.status` = `done`, with a `chosen` universe key and one candidate row per alternative (diff / p_pos / pass).
- `search` has rows for `lgbm_everest` (`n_trials` ≤ 25) and `xgb_everest`, each with `paired_cv.p_pos` and
  `adopted` true/false. Either outcome is fine; it is recorded, not judged.
- `ladder` has one row per arm present (aux, xgb, cat, ridge, sharpe, bags, mlp if trained), each `admitted`
  true/false with a reason. `member_corr_max_pair` is filled.
- `neutralisation.p_source` = **`simulation`**, with `p_b` ≤ 0.75 and `p_c` ≤ 0.5. The manifest agrees (step 4).
- `final` carries OOF fit20 (cross-fitted when NNLS is adopted), `forward_fit`, `score_sharpe_central` and `W`.
  - The realistic band for the final OOF fit20 is **0.024–0.028**; forward ≥ 0.020.
  - A blend above **0.0278** gets a SUSPECT banner from the run itself. Report it as suspect, never as a win.
- `pkl_check: PASSED`, file < 50 MB, slowest call < 5 s/era.
- `artifacts/diag/full01_validation.parquet` is built with 125,931 rows. It is **not** uploaded.
- `04-results.md`, `04-metrics.json`, `04-log-tail.txt`, `models/full01.manifest.json` and `.pkl.sha256` are pushed.

## 2. Preconditions

- Handoff 03 is reported (or explicitly skipped by the user). `git -C "$REPO" pull`; `.venv` present; `data/`
  complete.
- ~6 GB free disk: the GPU venv's torch wheels are ~2.5 GB, and the artifacts are ~1–2 GB.
- The laptop is on mains power and set not to sleep for 3 hours.
- No other heavy job is running: the run uses 16 threads and up to ~8 GB RAM.

## 3. HUMAN STEP

None, unless `test -s .env` fails (handoff 00). This run is offline apart from the pip/torch downloads in step 2.

## 4. Environment

```bash
REPO="$HOME/stealth"                    # absolute clone path: the SAME clone that ran handoffs 00-03
PY="$REPO/.venv/bin/python"             # train venv: EVERY stealth command
GPU_PY="$REPO/.venv-gpu/bin/python"     # GPU venv (step 2): the MLP member ONLY
cd "$REPO"
git log -1 --oneline
"$PY" -m stealth.selfcheck | tail -3    # expect PINS OK
"$PY" -m pytest -q 2>&1 | tail -3       # expect "N passed"; failures -> stop, report
nproc
```

## 5. Steps

**Step 1 — thread count (seconds; the only permitted edit)**

`configs/full.yaml` has `run.threads: 16`. If `nproc` printed less than 16, set it to that number. Do not commit the
change; revert it after the run with `git checkout configs/full.yaml`. Do not change anything else.

**Step 2 — GPU venv for the MLP (5–15 min the first time, seconds afterwards; idempotent)**
```bash
bash scripts/setup_gpu.sh
```
It installs torch from the CUDA 12.8 wheel index (the RTX 5070 Ti is Blackwell sm_120) into `.venv-gpu`. That venv
is never the train venv, and torch never reaches the pickle. It ends with a 4096² matmul smoke test.
- Last line `GPU OK` (exit 0) -> use step 3a.
- `GPU FAIL -> skip MLP` (exit 3), `SETUP ERROR` (exit 1), or any other non-zero exit -> use step 3b. Write
  "MLP skipped: <last 5 lines of setup_gpu.sh>" in the operator notes.

**Step 3 — the pipeline (expected 1–2 h on 16 threads, see the table; resumable: after ANY interruption re-run the
SAME command)**
```bash
bash scripts/run_pipeline.sh configs/full.yaml full01 --gpu-py "$GPU_PY"     # 3a: GPU OK
bash scripts/run_pipeline.sh configs/full.yaml full01 --no-mlp               # 3b: GPU failed
```
The script runs, in order:
1. `train --resume`
2. `mlp_train.py --resume`
3. `train --resume` again, to load the MLP member
4. `blend`
5. `bundle --require-frozen`
6. `pkl_check`
7. `diag`, which builds the file only

Each step logs `pipeline[full01]: step <name>: start` / `exit 0 after N s`, and the same lines go to
`artifacts/full01/pipeline.log`. If the MLP step fails, the pipeline logs `MLP skipped: ...` and continues without
it. That is expected and fine.

Estimated wall time, scaled from handoff 02 (497 s of training at 16 threads):

| part | estimate | what you see |
|---|---|---|
| data + knobs (default + 5 alternatives at base params) | 3–8 min | `knobs: ... chosen <key>` |
| LGBM search 25 TPE trials + base cell (5 folds + forward each) | 10–30 min | `search lgbm_everest: trial N ...`, then `... ADOPTED` or `NOT adopted` |
| XGB 12 cells, CatBoost 3, 12 aux, sharpe + 4 bags, ridge, refits | 15–30 min | `progress: member i/N ...` every 60 s |
| MLP 5 seeds × (5 folds + forward + full) on the GPU | 10–30 min | `mlp_train` epoch / parity lines |
| retrain (re-assemble with the MLP) + validation side metrics | 3–8 min | `external member mlp_everest: loaded` |
| blend: ladder + NNLS + regime grid (500 draws × 12 cells × 3 means) | 10–20 min | **can be silent ~15 min** during the grid |
| bundle + pkl_check + diag file (1,392 eras) | 8–20 min | `pkl_check: PASSED`, `diag progress ...` |
| **total** | **~1–2.5 h** (pre-02 estimate was 3–4.5 h) | `PIPELINE OK run_id=full01 ... mlp: ok` |

Exit 0 expected. A non-zero exit is the exit code of the failing step; the log ends with
`PIPELINE FAILED at step <name> (exit N)` -> section 6.

**Step 4 — the frozen constants reached the bundle (seconds)**
```bash
"$PY" -c "import json; m=json.load(open('models/full01.manifest.json')); print('p_source', m.get('p_source'), '| p_b', m.get('p_b'), '| p_c', m.get('p_c'), '| rule', m.get('blend_rule'), '| arms', m.get('admitted_arms'), '| pkl_check', m.get('pkl_check'), '| MB', round(m['size_bytes']/1e6, 2))"
```
Expected: `p_source simulation`, `p_b` ≤ 0.75, `p_c` ≤ 0.5, `pkl_check {'passed': True, ...}`.

**Step 5 — offline dry run (seconds; nothing submitted)**
```bash
"$PY" -m stealth.predict --pkl models/full01.pkl --offline
```
Expected: `predicted exped=exped_8055 n=90 ... unique=90`, then `dry run ok`; exit 0. **No `--submit`, no `--upload`.**

## 6. If it fails

- **Any interruption** (crash, reboot, Ctrl-C): re-run the same step 3 command. Train resumes from the Optuna SQLite
  and the member cache, the MLP resumes from `members/mlp_everest/_work/`, and the later steps repeat. After fixing a
  later-step problem you may add `--from <step>` (`blend`, `bundle`, `pkl_check`, `diag`).
- **`MemoryError` / OOM-killed**: halve `run.threads` (the step 1 edit) and re-run the same command.
- **train exit 2 (`LEAK`)**: the run is void. Report and stop.
- **`knobs already decided for this run-id; start a new run-id`**: the config changed under `full01`. Stop and report.
  Never delete `knobs.json`.
- **`--require-frozen` refusal at bundle** (`p_source` is not `simulation`): stop and report. Never pass
  `--p-b` / `--p-c`.
- **The MLP step fails** (exit 1 or 4): the pipeline continues on its own. Note "MLP skipped (exit N)" and the last
  20 lines of the `mlp_train` output.
- **`pkl_check` FAIL**: copy the failing lines into the operator notes; do not fix anything.
- **No new log line for > 20 min outside the blend's regime grid**: note the time, press Ctrl-C, re-run (it
  resumes). Past 4.5 h total, tell the user; let it continue while progress lines still come.

## 7. Report back (mandatory, also on failure)

```bash
"$PY" -m stealth.report --run-id full01 --handoff 04
```
This writes `handoffs/results/04-{results.md,metrics.json,log-tail.txt}`; then `git add handoffs/results && git commit
-m "results: 04 full-run" && git push`. The results file carries these sections:
- knobs decision;
- search adoption;
- admission ladder;
- member correlation (max pair);
- neutralisation table with the chosen `p_b` / `p_c` and `p_source`;
- final selection: OOF / forward / score Sharpe;
- dropped members;
- wall times.

On **PUSH FAILED — …** or **GIT COMMIT FAILED — …**, repeat the message verbatim to the user and stop.

Then append the operator notes and push, and commit the manifest and sha (never the pickle):
```bash
grep -E 'step .*: exit|MLP skipped|PIPELINE' artifacts/full01/pipeline.log
printf '\n## Operator notes\n\n```text\n%s\n```\n' "<commit: <sha>; GPU: OK|FAIL (<line>); MLP: ok|skipped (why); step exits + seconds: <paste the grep>; step 4: <paste>; predict --offline: ok; total wall: H h MM min; nproc: N, threads: 16; deviations: none>" >> handoffs/results/04-results.md
git add handoffs/results && git commit -m "results: 04 full-run (notes)" && git push
git add models/full01.manifest.json models/full01.pkl.sha256 && git commit -m "models: full01 manifest" && git push
```
Results files carry metrics and config/arm NAMES only; hyperparameter values stay in `artifacts/`.

## 8. Rules for the executing AI

- No code edits. The only permitted config edit is `run.threads` in `configs/full.yaml` (step 1); revert it afterwards.
- This handoff has **no USER MUST APPROVE banner**:
  - never `diag --upload`, never `predict --submit` / `--upload`, never stake;
  - never create, rename or archive a model, never change auto-submit.
  - `run_pipeline.sh` builds the diagnostics file only.
- Never commit `data/`, `.env`, `*.pkl`, `artifacts/`. Only `handoffs/results/` and `models/*.{manifest.json,sha256}`.
- Never `cat`/`echo`/`grep`/`source` `.env`; never `git remote -v`; never `git config --list`; never print the API key.
- Stop and write the results file on any judgement call. Do not run a step for more than 3x its stated time
  without telling the user.
