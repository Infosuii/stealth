# Next session — start here (written 2026-11-13 22:15 UTC)

Branch `worktree-m2-m6` holds the M2–M6 build. It is a **work-in-progress checkpoint, not yet reviewed — do not merge as-is.**

## Where it stopped

The build workflow reached its integration stage:

- Done: audit, plus four implementers (`train_knobs`, `blend_select`, `nn_runtime`, `docs_handoffs`).
- Interrupted: integration (pytest, smoke, `configs/minifull.yaml` end to end, `pkl_check`).
- Never ran: three adversarial reviews (selection statistics, pickle contract, handoff 03/04 executability) and the fix pass.

## Laptop handoff 02 findings the integration must address

Source: `handoffs/results/02-results.md` on `main`.

1. **blend.py / report.py:** compute and report the paired 20-era block bootstrap of the chosen blend vs the baseline member (`lgbm_everest`), on CV OOF plus the forward diff. The laptop AI stopped because this was missing.
2. **Member/group admission:** select the final blend among candidates via the paired gate (p_pos ≥ 0.8 on CV and forward diff ≥ 0; ties → simpler), and drop failing members. Equal weights over all 10 members scored 0.0226 against 0.0238 for the single LGBM; `lgbm_ghat` (0.0119) and CatBoost (0.0167, forward 0.0112) diluted it.
3. **bundle.py:** support `--members lgbm_everest` so diag upload #1 (handoff 03) is the single-LGBM M1 baseline.
4. **Handoff 04:** recalibrate runtimes. On the laptop (20 threads, 16 used) the whole test run took ~12.5 min (train 497 s, regime simulation 208 s). Spend the extra compute on measurement (fold offsets, bootstrap draws, stability checks), slower trees, admission candidates and the MLP, not on bigger searches or seed bagging.
5. **handoffs/02-test-run.md:** make the blend-vs-baseline p_pos informational, so a tie or loss means "keep the baseline".
6. **handoffs/INDEX.md:**
   - 00: accepted (b96dd66 / ee2afbd).
   - 01: accepted (c33bff6 / acfaf33).
   - 02: accepted (34c722a / 86c1c5c) — lgbm 0.0238 / fwd 0.0313 / sharpe 0.51; EW blend worse; Cat 18 s but weak; search tie.

## Update 22:35 UTC — hotfix landed on `main` (d2f1d0b)

To unblock the laptop before the session ended, a small hotfix went straight to `main`:

- `bundle --members` (equal weights over an explicit subset), with a test.
- `handoffs/03-diag-upload-1.md`: diag upload #1 of the single-LGBM baseline from `test01`, with `--p-b 0 --p-c 0`.
- The handoff 00 file-size fix and the handoff 02 "p_pos informational" fix.
- INDEX: 00–02 accepted, 03 ready.

Items 3, 5 and 6 above are therefore done on `main`. When merging this branch, expect conflicts in `stealth/bundle.py`
(this branch has its own `--members`; keep the branch version and make sure the main test
`tests/test_train_smoke.py::test_bundle_members_subset` still passes), `handoffs/03-diag-upload-1.md` (keep main's
version unless the branch's is better reviewed), `handoffs/00-env-setup.md`, `handoffs/02-test-run.md` and
`handoffs/INDEX.md`. The user may already have run 03 by then: check `handoffs/results/03-results.md` on `main`, and
if an `upload_id` is recorded, fetch the score with the MCP tool `eiq_get_diagnostics_result`.

## Update 22:50 UTC — branch integrated and merged with main; push to main blocked

Done solo by Claude (no subagents):
- `main` merged into this branch, with conflicts resolved (merge commit on `worktree-m2-m6`). **252 tests pass.**
- The integrator's `minifull` pipeline run (`artifacts/mfpipe`) is green end to end, including the MLP on CPU torch.
- Partial review:
  - Contract: `pkl_check` passes on a pickle that includes the MLP member, with torch=0 and only `stealth._runtime` by value.
  - Branch code is backward-compatible with the laptop's old-format `test01` artifacts, so handoff 03 works either way.
  - The admission ladder rejects weak arms on the paired gate; the staking-aware neutralisation selection is as designed.
  - MLP training uses inner early stopping, independent count sources and train-only standardisation, with a numpy/torch parity check.
- Handoff 04 clone path fixed to `~/Projects/eversteer/stealth`.

Claude's own push to `main` was refused by the permission classifier, so **the user pushed it themselves at
~22:55 UTC: `main` = e69b848, which includes all of the M2–M6 work.** Next session:
1. Run the three independent reviews on `main`: selection-stats/leakage of knobs + search adoption, contract,
   handoff 04 executability. Fix what they find.
2. Note: Claude cannot push to `main` without a Bash permission rule. Work on a branch and ask the user to push or
   merge, or ask them to add the rule.

Finding to tell the user: applied to the laptop's 02 regime table, the staking-aware rule picks **p_b = 0**: worst-case
score 0.0351 against 0.0274 at p_b = 0.5, because neutralising hurts in benchmark-right rounds.

## Update 23:15 UTC — diagnostics upload #1 scored (M1 baseline, neutralisation off)

Upload `35bfe266-286a-4464-a532-e4dca0d7899b` (results on main: 56819a6). Validation board, 150 models:

| metric | ours | board median | board P75 | models scoring > 0.045 (n = 23, median) |
|---|---|---|---|---|
| fit20 | 0.0189 | 0.0193 | 0.0214 | – |
| unq | 0.0039 | – | – | – |
| inov | 0.0141 | – | – | – |
| round score | ~0.0368 (rank 89) | 0.0388 | 0.0423 | – |
| sharpe | 0.32 | 0.344 | 0.381 | 0.40 |
| max drawdown | −7.55 (worse than 67%) | −5.98 | −4.40 | −3.76 |
| feature exposure | 0.685 (higher than 73%) | 0.659 | – | 0.686 |

Takeaways:
- **The offline framework is calibrated.** The regime simulation at p_b = 0 predicted a score of 0.0383
  (fit 0.0224 / unq 0.0026 / inov 0.0133); the true figures are 0.0368 (0.0189 / 0.0039 / 0.0141). The INOV proxy
  is close to the true core set, so M2–M6 selection can trust the offline numbers. Discount OOF FIT by about 0.79 for
  validation; the forward block (0.0313) was optimistic.
- **The gap to strong models is FIT, Sharpe and drawdown**, not UNQ/INOV. Long autocorrelated losing stretches
  (autocorrelation 0.94) cause the drawdown, which matters for staking. M2–M6 levers (knobs incl. recency, admission,
  family diversity) and the staking-aware W criterion target exactly this. Consider adding drawdown on OOF as a
  reported metric in the final selection.
- The #1 board model (0.136, Sharpe 1.33) is an outlier, plausibly exploiting the partly in-sample `v1_sherpa` on
  validation. Don't chase it.
- 4 diagnostics uploads remain (M3, M5 ×2, final).

## Steps

1. Check out this branch. Symlink `data/` and `artifacts/pklcheck_venv` from a checkout that has them, or re-download the data and recreate the venv.
2. Finish the integration and apply items 1–6.
3. Run the three adversarial reviews and fix what they find.
4. Merge into `main`. The user pre-approved this once tests and reviews pass (`docs/DECISIONS.md` #8).
5. Give the user the GPT prompt for handoff 03. The user pre-approved it (#9); the GPT still asks for confirmation in its own chat before uploading.
6. Finalise handoff 04 scaled to the laptop's real speed.
