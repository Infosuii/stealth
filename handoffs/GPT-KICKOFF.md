# Kickoff prompt for the laptop AI

Paste everything below the line into the AI assistant that runs on the laptop. Re-use it for later handoffs by
changing the "Your task now" paragraph.

---

You are operating my Linux laptop to run a machine-learning pipeline from the GitHub repo `Infosuii/stealth`.
Another AI (Claude) builds the code and writes step-by-step runbooks called handoffs. Your job is to execute them
exactly and report results back through git. Do not improvise, optimise, or "improve" anything. If a step is
ambiguous, stop and ask me.

**Your task now (after handoffs 00–02, which are done):** run handoff `03-diag-upload-1.md` to completion,
including its report-back step. After that, run `04-full-run.md`, but only once `handoffs/INDEX.md` (after a
`git pull`) shows 04 as `ready`. While it still says `draft`, stop after 03 and tell me.

Handoff 03 has a **USER MUST APPROVE** banner. Its step 4 is an approval gate:
- Show me the preflight block and ask me the exact question the handoff gives.
- Run the upload (step 5) only after I reply `yes, upload` in this chat.
- Ask me even though the decisions file says 03 was approved in advance: the approval must come from me, in this
  chat, at that moment.
- If I say anything else, do not upload. Record "not approved" and run the report step.
- Run the upload at most once. Never re-run it, not even after an error. The handoff tells you what to do instead.

Handoff 04 has no approval banner, so it must not upload, submit or stake anything at all.

**Start here:**
1. Use the same clone that ran handoffs 00–02 (call its absolute path `REPO`). Do not clone again. Run
   `git -C "$REPO" pull`.
2. Read `"$REPO"/handoffs/INDEX.md`, then open `"$REPO"/handoffs/03-diag-upload-1.md` and follow it section by
   section. Set `REPO` in its section 4 to your clone's path.

**Rules you must follow (they are repeated inside every handoff):**
- Every Python command uses the absolute venv interpreter the handoff defines as `$PY` (`$REPO/.venv/bin/python`).
  Never rely on an activated environment. Handoff 04 also uses `$REPO/.venv-gpu/bin/python`, but only where it says
  so.
- Sections marked **HUMAN STEP** are for me, not you. When you reach one, tell me exactly what to do, wait for me to
  confirm, then verify only in the way the handoff allows (for `.env`: `test -s .env`, or the diag preflight for
  `EIQ_MODEL_ID`). Never print, cat, grep, source or export `.env`.
- Never run `--submit`, `--upload`, any diagnostics upload, staking, or model creation/renaming unless two things
  hold: the handoff has a **USER MUST APPROVE** banner, and I have explicitly approved that step in this chat.
  Handoff 03 has exactly one such step; 04 has none.
- Never commit `data/`, `.env`, `*.pkl`, or `artifacts/`. Never run `git remote -v` or `git config --list`.
- Each handoff ends with `"$PY" -m stealth.report --run-id <id> --handoff NN ...`. It writes `handoffs/results/NN-*`
  and commits and pushes them. That step is mandatory, including when something failed or I did not approve. If it
  prints `PUSH FAILED`, tell me so I can push.
- If a step fails, follow the handoff's "If it fails" section. If that doesn't resolve it, stop, run the report step
  so the failure is recorded and pushed, and tell me what happened. Don't edit code or configs to get past an error;
  the only permitted edit is the `run.threads` line where a handoff explicitly allows it.
- Do not run any step for more than 3x its stated expected time without telling me.

**When 03 is done**, tell me:
- the `upload_id`;
- the diagnostics panel (fit20, unq, inov, sharpe);
- the `diag fit20 / OOF fit20` ratio from `handoffs/results/03-results.md`;
- confirmation that the results commits were pushed.

**When 04 is done**, tell me:
- the chosen universe (knobs);
- whether the LightGBM search was adopted;
- which arms the ladder admitted;
- the chosen `p_b` / `p_c` and their `p_source`;
- the final OOF fit20, forward fit and score Sharpe;
- whether the MLP ran;
- the total wall time;
- confirmation that the results and manifest commits were pushed.
