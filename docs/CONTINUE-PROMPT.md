# Development continuation prompt (after the M2–M6 fix pass)

**Latest checkpoint:** the user subsequently authorised autonomous model development and extensive local
CPU/GPU compute. Read the current section of `docs/NEXT-SESSION.md` first. Handoff 04 and research rounds
01/02 have finished; neither research round promoted a challenger. The historical prompt below must not
trigger a rerun or override the latest compute/development preference. Platform-write restrictions remain.

This prompt is for the next **development agent**, not for the laptop GPT (that one gets `handoffs/GPT-KICKOFF.md`).

---

Continue the Everesteer bot in GitHub `Infosuii/stealth` (clone it, or use an existing clone; work on a branch, not
on `main`). The M2–M6 fix pass is merged:
- the three independent reviews are fixed and independently verified;
- the reserved audit partition (`cv.audit`, `stealth.audit`) is in place;
- handoff 04 is `ready` for the laptop.

First read `docs/NEXT-SESSION.md`, `handoffs/INDEX.md`, `docs/DECISIONS.md`, `docs/INTERFACES.md` section 22, and, if
present, `handoffs/results/04-*`.

If handoff 04 has reported:
- verify its technical evidence (report status, the pkl_check pass bound to the manifest sha, diag-file coverage,
  audit status);
- summarise the development selection estimates and the audit verdict separately for me; an audit "win" is
  information, not a gate.

Then propose, but do not run without my approval:
- diagnostics upload #2 of the `full01` bundle (its own USER MUST APPROVE handoff: re-run `pkl_check`, then
  `diag --preflight`, then upload once);
- the M7 decision on whether to refit on all eras before the lane upload.

Rules:
- Never fit, select or tune on audit labels, the diag lane or `v1_sherpa`.
- Use a new run-id after any protocol change.
- Never commit data, artifacts, venvs, pickles, secrets or credential-bearing logs.
- No upload, submission, auto-submit change or stake without a USER MUST APPROVE handoff and my approval in chat.
- If GitHub push credentials are missing, say so, leave a portable git bundle, and do not claim a push.
