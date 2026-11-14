# Continuation note for the laptop GPT (after the M2–M6 fix pass, 2026-11-14)

The development agent will probably not be contacted again. You (the laptop GPT) finish two things: publishing the
fix-pass commits, then running handoff 04. Writing new code is out of scope.

## A. Publish (one time; the user's prompt gives the patch URL and its sha256)

The fix-pass commits were built on `worktree-m2-m6` at `6c719d6` and travel as a `git am` patch. After applying it:
- `git log --oneline -1` shows the commit titled "handoff 04: step 3b re-checks (GPU rehearsal, stale-MLP refusal,
  interrupt+resume) before the real run";
- push `worktree-m2-m6`;
- fast-forward `main` to it with `git push origin worktree-m2-m6:main` (a fast-forward; never force);
- verify with `git ls-remote origin main worktree-m2-m6`: both must show the same sha.

If the sha256 check or `git am` fails, stop and tell the user. Do not edit the patch.

## B. Run handoff 04

In the laptop clone that ran 00–03, `git pull` on `main`, then follow `handoffs/GPT-KICKOFF.md` exactly. 04 is
`ready` and uploads, submits and stakes nothing. Its step 3b re-checks the fix pass on the laptop BEFORE the real
training; the training starts only when every re-check passes.

## C. What is open (for the user, not for you to fix)

- The statistics and handoff fix-verification reviews were stopped before they reported. The contract
  verification finished with no P1. Open items are in `docs/NEXT-SESSION.md`:
  - P2: tighten `predict.check_upload_artifact` to require a full pkl_check record;
  - P3: the post-gate re-read races in diag and predict;
  - P3: `full_blobs: []` is treated as legacy;
  - P3: `--venv-python` with a bare name;
  - P3: `ensure_check_venv` deletes a foreign `--venv` dir.
  None affects handoff 04, which uploads nothing. Fix them before any later upload handoff.
- After 04 reports, the next steps each need the user's decision and a USER MUST APPROVE handoff written by a
  development agent: diagnostics upload #2 of the `full01` bundle, then the M7 full-data refit and lane upload.
  Do not improvise them.
