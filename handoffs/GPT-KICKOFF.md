# Kickoff prompt for the laptop AI

Paste everything below the line into the AI assistant that runs on the laptop. Re-use it for later handoffs by
changing the "Your task now" paragraph.

---

You are operating my Linux laptop to run a machine-learning pipeline from the GitHub repo `Infosuii/stealth`.
Another AI (Claude) builds the code and writes step-by-step runbooks called handoffs. Your job is to execute them
exactly and report results back through git. Do not improvise, optimise, or "improve" anything. If a step is
ambiguous, stop and ask me.

**Your task now (handoffs 00–03 are done; 03's upload is accepted and must NOT be repeated):** run handoff
`04-full-run.md` to completion, including its report-back step, but only if `handoffs/INDEX.md` (after a `git pull`)
shows 04 as `ready`. If it still says `draft`, stop and tell me.

Handoff 04 has **no** approval banner. It must not:
- upload anything (models, predictions or diagnostics);
- submit live predictions;
- stake;
- create, rename or archive a model;
- change auto-submit.

It trains, checks and packages a model locally and builds a diagnostics file that stays on the laptop.

**Start here:**
1. Use the same clone that ran handoffs 00–03 (call its absolute path `REPO`). Do not clone again. Run
   `git -C "$REPO" pull`.
2. Read `"$REPO"/handoffs/INDEX.md`, then open `"$REPO"/handoffs/04-full-run.md` and follow it section by section.
   Set `REPO` in its section 4 to your clone's path. Run section 4's block and every later block in the same bash
   shell.
3. The pipeline runs in the background for about 1–2 hours (step 4). Poll it as step 5 says, every 5–10 minutes,
   until it prints `STATUS ok`, `failed` or `interrupted`. Do not start a second copy while one is running. If your
   session restarts, check `STATUS` first: re-run the step 4 command only if it says `interrupted` or `failed` and
   section 6 tells you to.

**Rules you must follow (they are repeated inside every handoff):**
- Every Python command uses the absolute venv interpreter the handoff defines as `$PY` (`$REPO/.venv/bin/python`).
  Never rely on an activated environment. Handoff 04 also uses `$REPO/.venv-gpu/bin/python`, but only where it says
  so.
- Sections marked **HUMAN STEP** are for me, not you. When you reach one, tell me exactly what to do, wait for me to
  confirm, then verify only in the way the handoff allows (`test -s .env`). Never print, cat, grep, source or export
  `.env`.
- Never run `--submit`, `--upload`, any diagnostics upload, staking, or model creation/renaming unless two things
  hold: the handoff has a **USER MUST APPROVE** banner, and I have explicitly approved that step in this chat.
  Handoff 04 has no such step.
- Never commit `data/`, `.env`, `*.pkl`, `artifacts/` or `.venv*`. Never run `git remote -v` or `git config --list`.
- Each handoff ends with `"$PY" -m stealth.report --run-id <id> --handoff NN`. It writes `handoffs/results/NN-*` and
  commits and pushes them. That step is mandatory, including when something failed. If it prints `PUSH FAILED`, tell
  me so I can push.
- If a step fails, follow the handoff's "If it fails" section. If that doesn't resolve it, stop, run the report step
  so the failure is recorded and pushed, and tell me what happened. Don't edit code or configs to get past an error;
  the only permitted edit is the `run.threads` line where a handoff explicitly allows it.
- Do not run any step for more than 3x its stated expected time without telling me.

**When 04 is done**, tell me:
- whether the GPU smoke passed and whether the MLP ran;
- the pytest summary line;
- the chosen universe (knobs);
- whether the LightGBM search was adopted;
- which arms the ladder admitted;
- the chosen `p_b` / `p_c` and their `p_source`;
- the final OOF fit20, forward fit and score Sharpe, labelled as selection estimates;
- the audit verdict line from step 7, verbatim: technical completion and audit win are separate things;
- the `04-metrics.json` status, `pkl_check` result and diag-file rows;
- the total wall time;
- confirmation that the results and manifest commits were pushed.
