# Next session — state after the M2–M6 fix pass (2026-11-14)

## Current follow-up (read before the historical checkpoint below)

**Newest, utility19 complete; upload06 authorized and pending:** source 53ad07a, report
`handoffs/results/utility19-*`, protocol `docs/UTILITY-HISTORY-19.md`. All 702 native models in 468 groups
verified; fixed raw-count donors and six CPU own-count replays exact, serialization error0. Both training
stages exit0. Full suite500 passed without skips. Research readers remain <5925.

GPU core400 reaches transfer FIT0.01999896 vs baseline0.01526832 (+30.98%), proxy0.03250764 vs0.01868202
(+74.01%); it, its baseline blend and GPU-core3 blend pass viewed-transfer gates. ALL comparisons fail earlier
and development; no all-interval qualifier. CPU core3 at fixed raw counts helps earlier FIT but hurts transfer;
GPU core400 transfer benefits from both labels and own counts, with individually uncertain attribution.
Preserve these target/count/history interactions rather than dismissing the family or claiming broad success.

The latest user explicitly authorized uploading a bot after checks, without another approval question. Follow
`handoffs/06-research-bot-upload.md` (1d7e2b4). With no all-interval qualifier its frozen fallback is history18
pure CPU expanding3, an EXPERIMENTAL artifact; baseline01 remains the conservative local incumbent. Preserve
auto-submit OFF. Package through `scripts/package_research_bot.py`, run full pkl_check/offline checks, and
upload once via `scripts/upload_checked.py` with its durable receipt. Final full-data refit is separately
specified there; no later training label may choose a count or policy. A pending stored file is not an executed
bot. No direct submissions, diagnostics uploads or staking. Commit/push this completed report before final
refit, and commit/push the concrete checked manifest before uploading. Final artifact research06.pkl is built and checked (SHA50c1301f7360…,142069 bytes);300 source-era and native serialization replays exact, pkl_check51 eras passed, offline90 rows passed, final full suite501 passed. Packaging pandas copy correction fd4e68a is committed/pushed. No platform write has occurred yet.

**Newest, history18:** raw training-history controls complete; source 158512a, protocol
`docs/HISTORY-18.md`, reports `handoffs/results/history18-*`. This isolates short history as a material
contributor to prior failures. CPU expanding-from-400 beats rolling-1024 FIT in all three intervals
(descriptive p_pos 0.9926/0.9986/0.9835). GPU expanding-from-400 improves transfer FIT 0.00491660→0.01584654
(p_pos 0.9930), with positive point gains in the other two periods too. Preserve these family controls.

CPU expanding-from-3 reaches transfer FIT 0.01819329 vs 0.01526832 (+19.16%), proxy 0.02531301 vs
0.01868202 (+35.49%). It and its equal baseline blend pass the unchanged viewed-transfer historical gates.
However, its FIT loses earlier (-0.00288395) and is near-flat/slightly down in development (-0.00023071).
All comparisons fail earlier/development; no candidate passes all three intervals. This is useful historical
progress, not a reliable major future gain. No promotion; baseline01/platform state remain unchanged.

234 checkpoint groups, 351 native models including copied controls; 234 new selected models (78 CPU,
156 GPU seeds). All strict boundaries and saved bytes verified, serialization replay error zero. Full suite
495 passed, no skips. All new GPU trajectories completed 2,048 inner iterations; 154/156 cap endpoints were
worse than selected inner optima. Only two hit the cap, so blindly longer training is not the supported repair.
First GPU command exited 143 after all checkpoints completed; exact resume reused all 117 GPU groups and
exited 0 without retraining. Exit cause unresolved and recorded. Every reader excludes >=5925.

Next research must preserve these fixed comparisons and investigate why the earliest rows help transfer but
hurt earlier periods (representation relevance versus sample-size/regularization effects is unresolved).
Do not retrospectively choose each period's best history, call any viewed interval independent, weaken
completed gates, or assume diagnostics establish a platform leaderboard score. Utility tests below used a
shorter history than the now-improved raw controls; future matched tests must account for this limitation.

**Newest, objective16/17:** native supervised utility and matched CPU target/count controls complete.
Protocols `docs/OBJECTIVE-16.md`, `docs/OBJECTIVE-17-COUNTS.md`; reports `handoffs/results/objective16-*`
and `objective17-*`; sources 7535903 / 9e03a5b. All six objective16 comparisons failed historical gates;
no promotion. Earlier replication 2000–3999 is now scored too. Full GPU utility lost to raw GPU in all
three intervals. The isolated development CPU-full blend FIT 0.02815804 did not transfer (0.01180987 vs
baseline 0.01526832); do not present that maximum as achievement of a major improvement.

Count controls: six own-count replays exact, 54 changed-count refits, 24 equal-count reuses. Full labels at
core counts hurt earlier/development FIT but helped transfer (0.00717651 vs core 0.00340346), while full
labels at their own counts lost much of that recovery (0.00420219). This is a target/count interaction;
all diagnostic cells still lost to baseline, and none is eligible for substitution.

A raw-history lead is unresolved: expanding baseline floor400 vs old floor2800 improved observed development
FIT 0.02429122→0.02702630 and transfer 0.01498275→0.01526832, but paired p_pos 0.9132/0.5482 is insufficient.
History18 above completes the direct history/backbone controls. Do not claim a reliable gain from those means.
All 324 objective16 saved native models (51 teacher, 117 CPU, 156 GPU seeds) replayed exactly; 246 checkpoint
groups account for grouped seeds. Full suite 494 passed without skips. OOS teacher provenance and changed-
teacher cache refusal are enforced. Every reader still excludes >=5925; baseline01 and platform state unchanged.

**Newest, context15r:** effective-support repair complete; source 614dd3d, protocol
`docs/CONTEXT-15-REPAIR.md`, results `handoffs/results/context15r-*`. Nominal state-weight ESS is now bounded
as specified (minimum actual ~146.7; floor respected throughout), but viewed transfer still loses:
prediction-state FIT 0.01278598, feature-state 0.01288537 vs baseline 0.01498275. Only 48/1645 calibrated
dates needed tempering. This contradicts support collapse as a sufficient explanation; state associations
were unstable too. Preserve the repair and adverse controls; no candidate promoted or gate changed.

Native supervised-target experiment objective16, now completed above, creates an honest rolling OOS
benchmark teacher (never the provided train benchmark), tests full-utility labels with matched raw/core
controls on CPU/GPU, and adds earlier-period 2000–3999 replication. Training context >=400 is explicitly
declared; every reader still excludes >=5925. All intervals are historical and later intervals viewed;
no prospective or deployment claim is allowed solely from these scores.

**Newest, context15:** daily updates and current prediction/feature state conditioning are complete. Source
58c309a; protocol `docs/CONTEXT-15.md`; results `handoffs/results/context15-*`. Discovery remained positive,
but viewed-transfer FIT was 0.01286781 daily, 0.01279259 prediction state and 0.01284852 feature state vs
baseline 0.01498275. Faster updates alone did not repair transfer. Prediction-state seed agreement's
association with relative value changed sign across periods and was near zero in transfer. This does not
rule out richer predictive state information. Original boundary weights replayed within 6.7e-16.

Similarity weighting sometimes concentrated on very few eras: minimum nominal weight ESS 4.68 prediction,
27.58 feature, versus 195.25 unconditional. This is a supported variance/support concern, not proof of the
failure's sole cause. A predeclared ESS-constrained repair can isolate that mechanism; preserve this round's
adverse evidence. A separate supervised-target intervention with an honest OOS benchmark and earlier-period
replication is being prepared. No existing gate changed, no candidate was promoted and no native models
were retrained in context15. Full suite 489 passed without skips. All evaluated intervals are viewed data.

**Newest, online14r:** technical repair completed and confirmation evaluated; **no qualification**. The one
all-feature-missing row at 5582 is handled by valid-core residuals with N/M scaling. Source `4fe11ce`, protocol
`docs/ONLINE-14-REPAIR.md`, results `handoffs/results/online14r-*`. Original discovery scores replayed with
zero error, frozen winner and all native model bytes were unchanged. Confirmation FIT 0.01287388 vs baseline
0.01498275; composite proxy 0.01463712 vs 0.01753069; FIT/score p_pos 0.0070/0.0366 and all six aggregate
score sensitivities negative. The policy's positive discovery did not transfer. Baseline01 remains selected.

Post-decision controls show fixed prior also lost (FIT 0.01388173), stable group 0.01041563, GPU RMSE
0.00491660, QueryRMSE 0.00415384 and YetiRank 0.00913890. Family edges reversed in this period, and learned
weights added another 0.00100785 FIT loss vs fixed prior. Initial GPU weight 21.8% reflected earlier matured
strength; baseline weight eventually rose to 82.4%. It reacted too late to protect the first half. Specific
economic causes remain unresolved. Do not discard complementary families solely on averages, but do not
equate earlier late strength with robust future signal. No runner-up was promoted after confirmation.

Old and repaired actual predictions through 5639 were identical, proving the main early loss preceded any
effect of the missing-reference repair. The original input failure and GPU-launch/premature-report errors
are preserved in online14. Full suite 485 passed without skips. No uploads, submissions, staking, deployment
refit or old-audit reads occurred. Eras 5340–5924 have now been scored for this candidate and are consumed;
any future adaptation on them is research on viewed data, not fresh confirmation. Keep all failed gates and
source-bound caches. A reliable major predictive gain remains unproven.

**Newest, online14:** first policy to clear the new discovery gates. Adaptive_score FIT 0.02602050 vs
baseline 0.02429122 (+7.12%); proxy score 0.03933109 vs 0.03376212 (+16.49%); FIT/score gains positive in
all three windows. Fixed anchored prior also cleared gates, so do not attribute the whole gain to learning
weights. Adaptive incremental FIT vs fixed is not statistically established (descriptive p=0.1300).
Source 1d3512b; `docs/ONLINE-14.md`; `handoffs/results/online14-*`.

Frozen confirmation refits completed but scoring stopped on exactly one all-feature-missing row at exped_5582.
There is no original confirmation performance verdict. Preserve failed_input_contract and old caches.
A separately registered technical repair uses valid-core reference residuals with N/M covariance scaling,
retains all FIT/benchmark contributions, and must replay discovery exactly before resuming the frozen policy
on the unchanged native model bytes. No gate or model parameter may be relaxed. Full suite pre-repair 482
passed. Initial GPU-hidden launch and premature-report errors are documented in the report. No platform writes.

**Newest, ranking13:** GPU campaign completed; source `6ec06f9`, protocol `docs/RANKING-13.md`, reports
`handoffs/results/ranking13-*`. All 78 new native seed fits used GPU and replayed their serialized CPU
predictions exactly; 94 older checkpoint groups were byte-verified dependencies, not newly trained models.
The full 2,048-iteration inner schedules totalled 159,744 boosting iterations plus refits. No inner optimum
hit the cap, and endpoint validation proxies generally deteriorated. Do not promise longer fitting fixes this.

Raw GPU CatBoost was weak early (FIT 0.00143613 vs baseline 0.00997155), but strong late (0.02947111 vs
0.02442690, +20.65%). Both seeds independently showed late strength. QueryRMSE and YetiRank underperformed
raw regression overall. Fixed cross-family averaging diluted aggregate FIT but improved late; all eight
comparisons failed, so confirmation stayed unread. Baseline01 remains unchanged. This motivates a new,
preregistered prequential weight-learning test using ONLY earlier OOS predictions with labels matured by
the 40-era delay; never switch using the known period's future outcome. Full suite 478 passed with no skips;
the added dependency-corruption test also passed. CUDA is visible only outside the default sandbox.

**Newest, count12:** matched stopping-count controls are complete. Source `d9f3f52`; protocol
`docs/COUNT-12.md`; reports `handoffs/results/count12-*`. At the same 384 source eras, recent contiguous
history FIT 0.02479900 beat spread history 0.02140618, while expanding all inner sources to 784 eras gave
only 0.02177650 (ragged control 0.02267523). Time coverage matters at fixed era count; increasing older
history alone did not repair the failure. Different observations/instrument counts and regime composition
remain caveats. All source controls and 81 actual serialized boosters replayed exactly. No challenger
qualified and no confirmation labels were loaded. Original rolling FIT 0.02506814 is still stronger than
these isolated count repairs; do not silently promote a failed control or lower any previous gate.

GPU ranking campaign 13 is being prepared independently of count12's scores: RMSE/QueryRMSE/YetiRank with
the same explicit composite-proxy checkpoint selection, two fixed seeds and actual GPU fitting. CUDA is
hidden in the default sandbox but visible with authorised execution outside it on the RTX 5070 Ti. The
absolute CPU venv contains pinned CatBoost with GPU support; the torch GPU venv does not contain CatBoost.
Do not claim a CPU fallback is GPU training. Native CPU inference/serialized replay is intentional.

**Newest, temporal11 (2026-11-15):** fixed history/core/group consensus and multi-window stopping are complete.
Protocol `docs/TEMPORAL-11.md`; results `handoffs/results/temporal11-*`; implementation `62212fb`. Strongest
registered FIT: stable group consensus 0.02624674 vs equally refreshed baseline 0.02429122 (+8.05% observed),
with bounded composite proxy 0.04299203 vs 0.03376212 (+27.34%). These are adaptive historical estimates;
actual platform benchmark and core membership remain unknown. All six aggregate proxy-score sensitivities
improved and all three primary score windows improved, but the candidate missed Holm (0.006999 > 0.00625),
FIT p_pos (0.8574 < 0.95) and late FIT guard (-0.00144052 < -0.001). No discovery winner, confirmation read,
deployment refit, upload, submission or staking. Baseline01 stays selected; a reliable major gain is unproven.

The weaker individual history/core variants helped their fixed mixtures, so do not discard them solely on
individual averages. Core consensus beat each constituent overall but still lost late. The multi-window
count rule harmed FIT; unchanged-count blocks reproduced predictions exactly. Older inner sources had both
less fitting data and different periods, leaving sample-size transfer versus temporal shift unresolved.
Do not present median historical counts as a solved stability mechanism. Keep round-10 and round-11 gates
frozen. All 214 saved model checkpoints across both rounds replayed exactly; original baseline/rolling
control predictions matched between rounds. Full CPU suite 469 passed, 3 optional-interpreter skips; targeted
MLP suite with STEALTH_GPU_PY set to the absolute GPU venv passed 7/7, covering the skipped paths.

Latest runs use CPU LightGBM intentionally, with no gratuitous longer neural schedule; earlier supervised
training already overfit. Source-bound caches are under `artifacts/temporal10` and `artifacts/temporal11`.
Read these updates before the historical notes below; do not mix FIT-only and composite score units or
claim a leaderboard rank from proxy scores. Original live leader names are opaque generated/chosen labels,
not evidence of their architecture. The 2026-11-14 leaderboard leaders had only 1, 6 and 3 scored rounds.

**2026-11-15, temporal10:** sequential retraining and bounded composite-score proxies are implemented.
Protocol `docs/TEMPORAL-10.md`; results `handoffs/results/temporal10-*`; source `97c82c3`. Refreshed baseline
FIT 0.02429122 vs matched static 0.02225898 (refresh lost late). Rolling score-aware FIT 0.02510301 with
0.25 core-proxy neutralization and composite proxy 0.04442926 vs baseline 0.03376212. None of eleven
comparisons passed the corrected/effect/temporal gates; confirmation stayed unread. All 120 byte-bound
fit checkpoints replayed exactly. Target/count controls support an interaction on rolling history, not an
isolated stopping explanation. Counterfactual counts alone are insufficient to explain the gains. These are
historical estimates with OOS local benchmark references and unknown-core proxies, never platform scores.
Baseline01 remains unchanged. A follow-up tests fixed history/core consensus and multi-window stopping;
do not relax round 10's completed gates. Latest suite: 466 passed, 3 skipped in CPU venv.

**Newest: diagnose before discarding.** The user explicitly requires causal investigation of promising ideas
that underperform. `forensics06-*` and `forensics08-*` report matched controls, adverse results and unresolved
causes. More trees was a contradicted repair hypothesis. Preserving raw inputs recovered late latent-tree
FIT from 0.020151 to 0.024301 with frozen parameters; scale-consistent ensembling helped some windows. A
balanced denoising-head repair then reached 0.02489356 (+14.13% over the 0.02181070 baseline). Its baseline
blend passed the within-round corrected aggregate check, but failed the late-window guard. Thus no model was
promoted, confirmation remained unread, and baseline01 remains selected. This is not a reliable major gain.

The 480-trial replication found inner rankings weak/negative in most cases, with all nine original ensembles
replaying. Broad-consensus controls partially repaired that selection failure: the existing-search blend gained
6.14% overall and the fixed-prior blend 6.10%; the latter was slightly positive late but weaker early. The
remaining temporal generalization problem is unresolved. Do not retire these ideas solely on their scores,
claim every cause is proved, select an oracle later winner, or relax a completed gate. Keep supported,
contradicted and unresolved mechanisms separate and register new interventions before their scores.

Latest reports: `forensics06-*`, `repairs07-*`, `forensics08-*`, `consensus09-*` in `handoffs/results`.
Final suite: 461 passed, one expected CUDA no-device-path skip. Research code and private source-bound caches
are preserved. Reproduce each campaign at its recorded implementation/report commit; current source is newer
than older plan hashes. The following checkpoints are historical where they conflict with this update.

**Latest, after stability05:** the observed best discovery FIT is now 0.02462116 versus baseline 0.02181070,
or +12.89%, from a fixed stable-covariance/baseline blend. The latest window remains down 2.51%. It did not pass
the four-comparison Holm gate or the predeclared 15% major-improvement criterion. No candidate was promoted,
confirmation was not loaded, and baseline01 remains selected. The requested reliable major gain is still
unproven. Final suite: 451 passed, 1 expected CUDA no-device-path skip. Full evidence:
`handoffs/results/stability05-*`; protocol `docs/STABILITY-05.md`; implementation `21da749`; checkpoints
`artifacts/stability05`. This targeted follow-up uses temporal signal-risk weighting and optional recent-signal
forecasting; its direct linear solves require no additional neural training. The following paragraphs preserve
the preceding 03/04 checkpoint. Reproduce each source-bound experiment at its recorded version.

The user renewed the request for a **major predictive improvement**, including much longer training. Sustained
campaigns 03/04 are complete; reports are `handoffs/results/campaign03-*` and `representations04-*`. Across them,
336 nested search trials evaluated 1,008 inner tree fits, with more than 21,600 GPU training epochs plus refits.
The best historical discovery FIT increased from 0.02181070 to 0.02431878 (+11.50%), but the latest discovery
window fell from 0.02752725 to 0.02326433. Neither campaign cleared its predeclared promotion gates or 15%
major-improvement criterion. The user's performance objective remains unmet. Do not present tests, epochs,
code volume or this exploratory average gain as a reliable major improvement. Baseline01 remains selected;
neither campaign loaded confirmation labels, exported a new deployment pickle or made a platform write.

Long supervised fitting overfit. Target-free denoising improved the average, but did not solve temporal
instability. All saved fold models replayed correctly, and the final suite passed 448 tests with one expected
CUDA no-device-path skip. Encoder weights were unchanged after permuting targets in a regression check.
Protocols: `docs/CAMPAIGN-03.md`, `docs/REPRESENTATIONS-04.md`. Source-bound private checkpoints live under
`artifacts/campaign03` and `artifacts/representations04`. Reproduce campaign03 at `a8a3d00`/`bbbb29b`, before the
runtime extension; reproduce representations04 at `3ad0a71` or its report commit. Changing source requires a
new run ID; do not bypass the plan guard or retune on the old audit. Further development needs a new explicit
experiment design and honest historical/prospective evidence, not a relaxed gate after seeing these scores.

The user's latest instruction authorises autonomous model development and prioritises powerful local CPU/GPU
compute over compute cost. This supersedes the old laptop-operator-only prohibition on code changes for new
research. The preference is recorded in `docs/DECISIONS.md` and `docs/RESEARCH-01.md`. Always use the absolute
training and GPU venv interpreters, and never display `.env`.

Research rounds `research01` and `research02` are complete: nine architectures plus four objective/stopping
recipes, each tested alone and in a fixed baseline blend. Every challenger failed discovery; no confirmation
was scored, no challenger was promoted, and the exact unsearched `baseline01` artifact remains the conservative
candidate. Results and full failed-candidate tables are in `handoffs/results/research01-*` and `research02-*`.
Source-bound private checkpoints live in `artifacts/research01` and `artifacts/research02`; the first numerical
failure is preserved in `artifacts/research01-solver-limit`. Do not silently reuse a run ID after changing code,
data, recipes or seeds. Historical discovery/confirmation data are not a fresh holdout. Neither round made a
platform write. Their reports describe the actual tests and serialization checks.

Handoff 04 completed on the laptop and was pushed as `c66d0cd`, `da2ff1c`, `a329cb3`: technical status ok;
**no audit win**. Primary-agent follow-up verification and the contract fixes are documented in
`docs/reviews/bot-followup-2026-11-14.md`. This solo follow-up does not claim the stopped independent agents finished.
Do not rerun the full search or rescore an altered model on the audit.

The conservative candidate is the exact unsearched `full01/baseline` comparator, exported with
`stealth.bundle --audit-baseline`, not a member subset of the selected blend. No M7 full-data refit is authorised
by this artifact export. The optional next calibration step is `handoffs/05-diag-upload-2.md`; it needs its own
explicit approval and never automatically promotes full01. No new platform writes have occurred.

The remaining sections below describe the earlier checkpoint and are historical where they conflict with this update.

Read this first, then `docs/DECISIONS.md` (#11–#15 are new), `handoffs/INDEX.md`, `docs/INTERFACES.md` section 22 and
the "Fix verification" sections of `docs/reviews/m2-m6-{statistics,contract,handoff}.md`. Older history is in git
(`git show 6c719d6:docs/NEXT-SESSION.md`).

## Where things stand

- The three independent reviews found 6 statistics, 7 contract and 5 handoff issues. All of them are fixed, with
  regression tests. Three fresh, read-only reviewer agents then verified the fixes (results under "Independent
  verification" in each review file).
- The reserved audit partition is implemented end to end (`cv.audit`, `stealth.audit`):
  - eras 5945–6524 are never read by a fit or a choice of the run;
  - the frozen blend is scored once against an unsearched default-universe LightGBM baseline;
  - a permuted-label regression proves the audit labels reach no decision.
- `train --skip-external / --require-external` exist and the pipeline uses them.
- **Handoff 04 is finalised for the laptop**: Linux, 20 threads (16 used), RTX 5070 Ti, ~30.8 GB, Europe/London.
  Estimates are scaled from handoff 02 (training 497 s, simulation 208 s). The measurement budget is 5,000 bootstrap
  and 1,000 simulation draws; there are no extra trials or seeds. The run uploads, submits and stakes nothing.
- Handoff 03 is accepted (upload `35bfe266`). Four diagnostics uploads remain, each needing its own USER MUST APPROVE
  handoff. The `m1_lgbm` manifest from 03 holds a legacy pkl_check pass without a hash, so it no longer satisfies any
  upload gate: re-run `pkl_check` before any future upload of it.

## Independent verification status (session wrapped up early at the user's request)

- **Contract reviewer: finished.** No bypass and no P1. All seven findings were confirmed fixed.
  - **Open P2:** `predict.check_upload_artifact` should also require `status == "passed"`, a checker Python of
    3.11.x, both isolation flags false and libs equal to the lane pins. Also, `pkl_check.update_manifest(pkl, True)`
    with no report should not create a pass. The real CLI always passes a report, so this is not live.
  - **Open P3s:** the diag/predict re-read race after the gate; `full_blobs: []` treated as legacy; the
    `--venv-python` bare-name regression; `ensure_check_venv` deletes any `--venv` dir without `.pins.txt`.
- **Statistics and handoff verification reviewers: stopped before they reported.** Re-run them next session before
  diagnostics upload #2.

## Verification actually performed on the sandbox (2 cores, Python 3.11.2, exact pins)

- Full pytest in `.venv` (no torch importable): see the final count in the commit message of the merge to main.
  Before the docs commits it was 358 passed + 2 expectation updates, then green.
- Real-data minifull pipeline `mf01` with the external MLP on CPU torch (`/opt/ml/bin/python3`, `--mlp-device cpu`):
  - **Outcome:** `PIPELINE OK` in 154 s: train 100, mlp 10, retrain 14, blend 16, audit 3, bundle 0, pkl_check 3,
    diag 8 s, then report. Report status `ok` with all evidence bound to the files.
  - **Bundle:** frozen (`--require-frozen`, `p_source simulation`); `pkl_check` PASSED in `artifacts/pklcheck_venv`,
    51 eras; diagnostics file 125,931 rows / 1,392 eras.
- **MLP portability:** a real-data bundle including the trained MLP (`--members ... mlp_everest --p-b 0.5 --p-c 0.25`)
  passed `pkl_check` in the clean venv on 51 eras: torch=0, pins verified, benchmark neutralisation active.
- **Failure paths:** a same-command resume reused every cache. `--no-mlp --from blend` refused while the OOF holds the
  MLP. Run `mf02`: TERM mid-train recorded `interrupted` (143) with no stray processes and the member left `running`;
  the re-run finished OK and reproduces `mf01`'s blend and audit exactly.
- **Real `full.yaml` layout** (RunData on the real data, pipeline `--dry-run`): development 2800–5924 (3,125 eras),
  audit 5945–6524 (580 eras, 53,292 rows), full refits ≤ 5924, own-fold count sources, documented commands.
- **Leak gate on a toy window:** eras 6000–6299 are a high-signal regime. `feature_Agouti` alone has a per-era
  Spearman of +0.20 there, so a toy test on that window raises `leak_spearman` locally (`tests/test_audit.py`).
  The full-history gate (0.18) is unchanged.

## Environment notes (this sandbox; nothing here is a laptop change)

- `.venv` = `/usr/bin/python3.11 -m venv` + `requirements-lock.txt` + `-e .` (no torch). `artifacts/pklcheck_venv` was
  built by `pkl_check.ensure_check_venv`. `/opt/ml/bin/python3` has CPU torch and is used only as `--gpu-py` with
  `--mlp-device cpu`.
- Data: the SDK on the box is signed in, so a small script calling `EverestAPI(tournament="futures")` downloaded the
  six parquet files into ignored `data/`. No model, upload, submission, auto-submit change or stake.
- GitHub reads work. **Pushes need credentials this box does not have** (no gh, helper, token or SSH key). See the
  publication note in the session report: the branch/main state was either pushed and verified, or left as a local
  fast-forward plus a git bundle for the user to push.
- Commits use `git -c user.name=Claude -c user.email=noreply@anthropic.com` (no repo config change). The hook runs via
  `core.hooksPath=scripts/hooks`.

## Next steps

1. The user runs handoff 04 on the laptop with the kickoff prompt in `handoffs/GPT-KICKOFF.md`.
2. Pull and read `handoffs/results/04-*`:
   - a technical success needs report status `ok`, `pkl_check` bound to the bytes, the diag file and the audit;
   - record the audit verdict (blend vs the unsearched baseline on 5945–6524) separately;
   - compare dev selection estimates with the audit and handoff 02's numbers;
   - check the GPU MLP timings against the 5–25 min estimate.
3. Decide (user) on diagnostics upload #2 of the full01 bundle: a new USER MUST APPROVE handoff (05 or 06) that
   re-runs `pkl_check`, `diag --preflight`, and uploads once.
4. M7 (separate decision, never silently after reading the audit): whether to refit on all eras up to 6524 for the
   lane upload, then the lane upload + monitor (handoff 05, USER MUST APPROVE).
