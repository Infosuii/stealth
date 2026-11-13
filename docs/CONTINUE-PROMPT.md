# Development continuation prompt (paused M2–M6 review)

This prompt is for another **development agent**, not for the laptop GPT running handoff 04.

---

Continue the Everesteer bot in `/home/climber/work/stealth` (GitHub `Infosuii/stealth`) on `worktree-m2-m6`.
The previous agent paused at my request during the three independent reviews/fix pass. Its WIP checkpoint is
not merge-ready and handoff 04 is still draft. If GitHub still points to `0d185c8`, the previous workspace lacked
push credentials: import the supplied `stealth-review-checkpoint.bundle` (branch `worktree-m2-m6`) before proceeding,
or use the existing local checkpoint. Do not discard its changes or assume they are already on GitHub.

First read `docs/NEXT-SESSION.md`, `docs/PLAN.md`, `docs/DECISIONS.md`, `handoffs/INDEX.md`, and all three
`docs/reviews/m2-m6-*.md` reports. `NEXT-SESSION.md` describes the exact partial implementation, environment,
verified checks, missing interfaces, and the agreed separate development/audit protocol.

Finish the statistical/leakage, pickle-contract, and handoff-executability fixes with regression tests. Independent
review agents are explicitly authorized if helpful. Complete the reserved audit partition and frozen-model audit
reporting before calling the forward results independent; keep audit labels out of every adaptive decision.
The WIP pipeline currently calls an unimplemented `train --skip-external` and conditionally an unimplemented
`stealth.audit`, so complete those interfaces before attempting the end-to-end driver.

Then finalise handoff 04 for my Linux laptop: 20 logical threads (16 workers), RTX 5070 Ti, Europe/London,
~30.8 GB RAM. Handoff 02 took 12.5 minutes: training 497 s, regime simulation 208 s. Spend spare compute on
measurement and stability, not extra search trials or added seed bagging. Run the complete test suite, real-data
minifull including the external CPU-MLP test, and frozen-bundle portability checks in the isolated lane venv.
Keep 04 draft until those pass. Handoff 03 is already accepted; do not repeat its upload.

Once fixes, reviews, tests, and docs are complete, get the finished work onto `main`, verify the remote commit,
and give me the laptop GPT prompt to run 04. I already approved that merge/push; do not ask again. If GitHub
authentication is unavailable, report the specific blocker and preserve a portable checkpoint. Do not claim a
push succeeded without checking it. Handoff 04 must not upload predictions/models, submit live predictions,
change auto-submit, or stake. Never commit data, artifacts, virtual environments, pickles, or secrets.
