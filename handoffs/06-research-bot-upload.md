# Handoff 06 — checked research bot upload

> **USER MUST APPROVE — authorization already received.** The user instructed: “continue this time after
> you finish and are satisfied upload a bot to the platform”. This covers the eventual model-pickle upload
> after research and artifact checks. No additional approval question is required. Existing auto-submit
> remains OFF; no direct submissions, diagnostics uploads, staking, model creation or renaming.

## 1. Finish the research and freeze the decision

Finish utility19, including its matched-count controls, report, commit and push. All periods are viewed
historical research. Preserve every completed gate and every adverse interval. Do not change a completed
criterion to call the research successful.

If any utility19 comparison passes its unchanged historical gates in ALL THREE intervals, choose the one
with highest era-count-weighted primary proxy mean across those intervals (ties by policy name). Otherwise
upload history18's pure `cpu_expanding3` as an EXPERIMENTAL research artifact: its viewed-transfer FIT is
0.01819329 vs 0.01526832, but earlier replication loses and development does not pass. This fallback is
registered before reading utility19's scores. It is a choice for an unstaked, auto-submit-OFF upload, not
promotion of a broadly qualified winner. Keep baseline01 unchanged as the conservative local incumbent.

Write the concrete chosen source, policy, source plan, known results and rationale to
`artifacts/upload06/decision.json` before final training. No choice may depend on new platform scores.

## 2. Final refit and package

Use `/home/om/Projects/eversteer/stealth/.venv/bin/python` throughout. Research readers still exclude >=5925.
The final deployment refit is separate: after freezing policy/parameters/counts, fit on the provided TRAIN
split, including its later labels, without scoring or using them for stopping, tuning, selection or weights.
This handoff interprets final refitting as necessary preparation for the user's requested ready bot upload.
Validation/live labels are never available or used. Do not repeat an audit after full-data fitting.

For each constituent and seed, freeze the tree count to the integer median of the native selected counts in
the final THREE source blocks (5640, 5740, 5840), never a count chosen using post-5924 labels. Retain source
floor 3/400, parameters, feature order, raw/core target, CPU seed17 and GPU seeds17/29. No extra seed search.
For any eligible equal blend, retain exactly its existing equal weights and Gaussian-rank aggregation.
Neutralisation remains zero. Raw controls and core labels use the same frozen helpers as research; full
benchmark-utility labels are not eligible because no honest benchmark covers the earliest rows.

`scripts/package_research_bot.py` must verify all chosen source checkpoints and plans, save a byte-bound
refit provenance record, replay serialization, and use the established by-value callable bundle writer.
Build `models/research06.pkl`; never commit its bytes or training data/artifacts. Run the full hash-bound
`stealth.pkl_check` in the clean lane venv, then offline prediction and `predict.check_upload_artifact`.
Verify count/history/seed/target/weight correspondence and actual callable outputs against source native
models on a source prediction block before accepting packaging. Any discrepancy stops the upload.

## 3. Concrete review and one upload

Add the chosen policy, pickle SHA256/size, portability evidence and exact slot to this report, then commit
and push the preparation before the write. Target the EXISTING slot:
`f06a70bf-ef8b-483d-b2a9-d4b9ef3e7043` (`mandelbrot-bayesian-payoff`).
Verify `.env` only with `test -s .env`; never display, grep, source or export it. The existing app client
loads its credentials internally. Run:

```bash
STEALTH_CONFIRM_UPLOAD=1 /home/om/Projects/eversteer/stealth/.venv/bin/python scripts/upload_checked.py \
  --pkl models/research06.pkl --model-id f06a70bf-ef8b-483d-b2a9-d4b9ef3e7043 \
  --receipt artifacts/upload06/receipt.json --i-have-approval
```

The helper writes an intent receipt before sending bytes, then persists the returned ID before polling.
Do not repeat a write after any ambiguous result. Reconcile using ONLY:

```bash
/home/om/Projects/eversteer/stealth/.venv/bin/python scripts/upload_checked.py \
  --receipt artifacts/upload06/receipt.json --verify
```

Verify remote content_hash, upload ID, file size and declared Python 3.11 against local checked bytes. Preserve
auto-submit OFF. `status=pending` with that setting is an accepted stored file, not proof of platform execution.
Report execution and sandbox fields exactly; never claim the bot is running or invent platform metrics.
Do not activate the lane merely to make the upload status advance.

## 4. Report and push

Write `handoffs/results/06-results.md` and `06-metrics.json`: selected policy and its adverse evidence,
source/preparation commits, final fit boundaries/counts, exact artifact hash, checker/offline results, upload
ID, remote hash/size/interpreter evidence, sandbox/execution status, auto-submit state and every operational
issue. Preserve all prior failure reports. Commit only reports, source, manifests/SHA files and active upload
metadata; never the pickle, `.env`, data, private receipt or artifacts. Push and verify via git ls-remote.

## Concrete checked preparation (2026-11-15)

Utility19 report fc56cca has no all-interval qualifier. The frozen fallback is history18 CPU expanding3;
baseline01 remains the conservative local incumbent. Prepared file `models/research06.pkl`, 142,069 bytes,
SHA256 `50c1301f73605c48067cbed6ee5789b341afe0c1e3cd3ea2cd5164901ae05a48`.
Full-data refit:468,236 train rows, eras3–6524, raw target, CPU seed17,86 trees (frozen median[86,231,39]).
Source runtime replay:300 eras, error0. Native serialization replay:error0. `pkl_check: PASSED`,51 eras,
exact Python3.11.18 pins, isolation verified, slowest0.4134s. Offline prediction passed90/90 rows.
The pandas mutable-array correction is fd4e68a; it changed no model decision or count.

The selection rule was registered in1d7e2b4 before utility19 scores; this appended preparation record only
identifies its concrete output. See `handoffs/results/06-results.md` and `06-metrics.json` for final tests
and the single-upload outcome. The explicit user authorization at the top covers this prepared artifact.
