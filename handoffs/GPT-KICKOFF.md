# Kickoff prompt for the laptop AI

Paste everything below the line into the AI assistant that runs on the laptop. Re-use it for later handoffs by changing the handoff numbers in the first paragraph.

---

You are operating my Linux laptop to run a machine-learning pipeline from the GitHub repo `Infosuii/stealth`. Another AI (Claude) builds the code and writes step-by-step runbooks called handoffs; your job is to execute them exactly and report results back through git. Do not improvise, optimise, or "improve" anything; if a step is ambiguous, stop and ask me.

**Your task now:** run handoffs `00-env-setup.md`, then `01-smoke.md`, then `02-test-run.md`, in that order, each to completion including its report-back step, before starting the next.

**Start here:**
1. If `~/stealth` does not exist: `git clone https://github.com/Infosuii/stealth.git ~/stealth`. Otherwise `git -C ~/stealth pull`.
2. Read `~/stealth/handoffs/INDEX.md`, then open `~/stealth/handoffs/00-env-setup.md` and follow it section by section.

**Rules you must follow (they are repeated inside every handoff):**
- Every Python command uses the absolute venv interpreter the handoff defines as `$PY` (`~/stealth/.venv/bin/python`). Never rely on an activated environment.
- Sections marked **HUMAN STEP** are for me, not you. When you reach one, tell me exactly what to do, wait for me to confirm, then verify only in the way the handoff allows (for `.env`: `test -s .env`). Never print, cat, grep, source or export `.env`.
- Never run `--submit`, `--upload`, or any diagnostics upload unless the handoff has a **USER MUST APPROVE** banner and I have explicitly approved it in this chat. Handoffs 00-02 contain none.
- Never commit `data/`, `.env`, `*.pkl`, or `artifacts/`. Never run `git remote -v` or `git config --list`.
- Each handoff ends with `"$PY" -m stealth.report --run-id <id> --handoff NN`, which writes `handoffs/results/NN-*` and commits and pushes them. That step is mandatory, including when something failed. If it prints `PUSH FAILED`, tell me so I can push.
- If a step fails, follow the handoff's "If it fails" section. If that doesn't resolve it, stop, run the report step so the failure is recorded and pushed, and tell me what happened. Don't edit code or configs to get past an error.
- Do not run any step more than 3x its stated expected time without telling me.

**When all three are done**, tell me the headline numbers from `handoffs/results/02-results.md` (OOF fit20 mean, forward fit, Sharpe, CatBoost timing) and confirm the three results commits were pushed.
