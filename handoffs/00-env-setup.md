# Handoff 00 — environment setup on the laptop (Linux, Europe/London)

## 1. Purpose + success criterion

Create the pinned Python 3.11 training venv on the laptop, activate the secret-scanning git hook, download the
data, and prove it with `selfcheck` + `pytest`. Nothing is trained and nothing touches the platform except the
read-only dataset download.

Success = ALL of:
- `$PY -m stealth.selfcheck` prints `PINS OK` (13 rows OK, 0 MISMATCH/MISSING), `core.hooksPath` = `scripts/hooks`,
  `EIQ_API_KEY present (environment or .env)` (selfcheck reads `.env` itself; nothing needs exporting), and exits 0.
- `git config --get user.email` prints an address (a git identity exists; otherwise `git commit` fails -> HUMAN STEP).
- `data/` holds 6 parquet files: `futures_{train,validation,live}.parquet` (468,236 / 125,931 / 90 rows) and
  `futures_{train,validation,live}_benchmark_models.parquet`.
- `$PY -m pytest -q` ends with `N passed in …s` (a `, K skipped` suffix is fine; `failed`/`error` is not).
- `handoffs/results/00-results.md` + `00-metrics.json` are committed and pushed.

## 2. Preconditions

- Linux laptop, ~5 GB free disk, network access (pip wheels ~600 MB; data ~120 MB).
- Repo cloned at `$HOME/stealth` (adjust `REPO` below if it lives elsewhere) and up to date: `git -C "$HOME/stealth" pull`.
- `git push` works from this clone (the GPT has its own GitHub access to `Infosuii/stealth`).
- `.env` present with `EIQ_API_KEY` — see the HUMAN STEP below. Without it every offline step still works; only the
  data download (step 4) needs it.

## 3. HUMAN STEP (the human does this, not the AI)

```
cd ~/stealth
cp -n .env.example .env
# open .env in an editor and paste the Everesteer API key after  EIQ_API_KEY=
# leave EIQ_BASE_URL=https://api.everesteer.ai ; EIQ_MODEL_ID can stay empty for now
```
The AI verifies only with `test -s .env && echo ".env present"` — it never prints, cats, greps, sources or exports `.env`
(`selfcheck`, `data download`, `predict` and `monitor` read it themselves).

Also a HUMAN STEP if step 2 prints `GIT IDENTITY MISSING`: the human runs, inside the clone,
`git config user.name "<name>"` and `git config user.email "<email>"` (any values GitHub accepts). Without them no
`git commit` in any handoff can succeed.

## 4. Environment

Every command below uses the absolute interpreter `$PY`; no venv activation is assumed anywhere.

```bash
REPO="$HOME/stealth"                    # absolute clone path; change if different
PY="$REPO/.venv/bin/python"             # absolute venv interpreter used by EVERY command
cd "$REPO"
```

## 5. Steps

**Step 1 — create the venv, install pins, activate the hook (3–6 min first time, seconds afterwards; idempotent)**
```bash
bash scripts/setup.sh
```
Expected tail of output:
```
PINS OK
SETUP OK. Use this interpreter in every command:
PY=/home/<you>/stealth/.venv/bin/python
```
If the printed `PY=` path differs from your `$PY`, use the printed one from now on.
(`setup.sh` installs `uv` to `~/.local/bin` when missing, or falls back to `python3.11 -m venv`.)

**Step 2 — confirm the hook, the git identity and the .env presence (seconds)**
```bash
git config --get core.hooksPath          # expect: scripts/hooks
git config --get user.email >/dev/null && git config --get user.name >/dev/null && echo "git identity ok" || echo "GIT IDENTITY MISSING -> HUMAN STEP"
test -s .env && echo ".env present" || echo ".env MISSING -> HUMAN STEP"
```

**Step 3 — selfcheck (seconds)**
```bash
"$PY" -m stealth.selfcheck
```
Expected: a `== host ==` table (python 3.11.x, os, cpu, ram_gb, gpu or `none`, timezone `Europe/London (...)`,
`core.hooksPath scripts/hooks`, `EIQ_API_KEY present (environment or .env)`), then 13 pin rows all `OK`, then
`PINS OK`. Exit 0. (`ABSENT` here means `.env` is missing or its `EIQ_API_KEY=` line is empty -> HUMAN STEP.)

**Step 4 — download the data (1–3 min; needs `.env`; re-run safe, skips nothing but overwrites identical files)**
```bash
"$PY" -m stealth.data download
ls -l data/
```
Expected: 6 files in `data/`; `futures_train.parquet` ≈ 80–90 MB. The log prints each file path once.

**Step 5 — test suite (1–3 min on the laptop, 65 s on the 2-core sandbox; offline; prints only dots until the end)**
```bash
"$PY" -m pytest -q 2>&1 | tee /tmp/pytest_full.txt | tail -3
tail -3 /tmp/pytest_full.txt > /tmp/pytest_tail.txt
```
Expected last line: `N passed in …s` (N ≈ 160; a `, K skipped` suffix is fine; `failed` / `error` is not).
(`tee` is used here only because pytest writes nothing to disk itself; the tail file is pasted into the results in step 7.)

## 6. If it fails

- Any step: a wall time > 3× the expected time → still finish, note the time in the results; do not abort.
- Step 1: `uv` download blocked → the script falls back to `python3.11`; if neither exists:
  `sudo apt install python3.11 python3.11-venv` (Ubuntu ≤22.04 needs the deadsnakes PPA) and re-run.
  A `MISMATCH` row in selfcheck → re-run `bash scripts/setup.sh` (idempotent); manual equivalent:
  `uv pip install --python "$PY" -r requirements-lock.txt -e .` (or `"$PY" -m pip install -r requirements-lock.txt -e .`;
  the venv is created with `--seed`, so pip exists; if `No module named pip` ever appears, use the `uv pip` form).
- Step 2: hook path empty → `git config core.hooksPath scripts/hooks`. `GIT IDENTITY MISSING` → HUMAN STEP (section 3).
- Step 4: `RuntimeError: EIQ_API_KEY not set` → `.env` missing or empty → HUMAN STEP, then re-run.
  HTTP 401/403 → the key is wrong; stop and report. Network error → retry once, then report.
- Step 5: any `failed` → do NOT fix code. Copy the failing test names into the results file (step 7 below) and stop.

## 7. Report back (mandatory, also on failure)

There is no training run here, so the reporter is pointed at a run id that does not exist; it still writes the
environment table and marks the run `failed` (that is expected and fine for 00). Append the selfcheck tail and the
pytest tail saved in step 5 (seconds; no re-run of the test suite), then commit and push:

```bash
"$PY" -m stealth.report --run-id env-setup --handoff 00 --no-git
"$PY" -m stealth.selfcheck 2>&1 | tail -20 > /tmp/selfcheck_tail.txt
printf '\n## Operator notes\n\n```text\n' >> handoffs/results/00-results.md
cat /tmp/selfcheck_tail.txt /tmp/pytest_tail.txt >> handoffs/results/00-results.md
printf '```\n' >> handoffs/results/00-results.md
git add handoffs/results && git commit -m "results: 00 env-setup" && git push
```
Expected: three files `handoffs/results/00-{results.md,metrics.json,log-tail.txt}` committed; `git push` prints the
new commit on `main`. If `git commit` fails with `Author identity unknown` / `Please tell me who you are`, that is the
GIT IDENTITY HUMAN STEP (section 3): say so and stop. If `git push` fails, say loudly **PUSH FAILED — results are
saved locally in handoffs/results/; ask the human to run git push** and stop. (When `report.py` is run WITHOUT
`--no-git` in later handoffs it prints either that sentence or `GIT COMMIT FAILED — …` itself; both mean "stop,
tell the human".)

## 8. Rules for the executing AI

- No code or config edits in this handoff. Only the commands above.
- Never run `--submit`, `--upload`, or any diagnostics upload: this handoff carries no USER MUST APPROVE banner.
- Never commit `data/`, `.env`, `*.pkl`, `artifacts/`. Only `git add handoffs/results`.
- Never `cat`/`echo`/`grep` `.env`; never `git remote -v`; never `git config --list`; never print the API key.
- On any judgement call (unexpected output, partial success), write the results file with what you saw, push, stop.
