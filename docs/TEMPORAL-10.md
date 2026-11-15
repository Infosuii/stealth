# Temporal adaptation and composite score, round 10

Registered before training or inspecting round-10 scores. The user renewed autonomous development after
reviewing the live leaderboard. Past research improved average historical FIT but repeatedly lost in the
latest discovery window. Supervised overfitting, projection loss, scale-sensitive ensembling and unstable
inner model rankings were investigated in rounds 06–09. More epochs alone has no supporting evidence.

Test an explicit temporal intervention rather than another architecture search. Config: `configs/temporal.yaml`.
Every 100 eras, fit with labels strictly before the first prediction era minus 40. Previous prediction-block
labels may enter later fits only after that delay; this is a sequentially executable historical simulation.
The primary comparator is refreshed at exactly the same cadence and uses exactly the same data-availability
rules. Three window-start-only fits isolate the effect of refreshing from changing the learning recipe.

Five fixed recipes: expanding baseline; rolling 1,024-era history; expanding with 512-era half-life weights;
score-aware expanding; score-aware rolling. No hyperparameter search, adaptive model rejection or seed selection.
Also test a fixed equal Gaussian-rank consensus of all five, never raw-score averaging. Inner stopping uses
the latest 200 available eras, with a 40-era gap. Raw recipes stop on MSE. Score-aware recipes stop on the
mean *bounded composite proxy*, against a separate fixed-count baseline fit exclusively on the inner fitting
partition. Every final fit uses only the outer available partition and its own inner stopping count.

Score-aware training labels are the centered target plus its residual against the per-era Gaussian-ranked
all-feature mean when that mean has nonnegative target covariance. This is a FIT+INOV directional-utility
surrogate, not an exact nonlinear ranking objective. Rescale to the centered target's pooled standard deviation.
Do not train on the provided train-split benchmark, or use an in-sample fitted benchmark as a target. The
score-aware inner stopping criterion additionally measures UNQ against an honest inner out-of-sample reference.

Correct the units mismatch explicitly: report FIT, UNQ, INOV and per-era
`b * arctan((FIT + UNQ + INOV) / b)`, with b frozen from the platform's 2026-11-14 scoring read. Average the
bounded per-era scores, rather than bounding averaged components. All composite metrics are **proxies**:
benchmark references are our refreshed and static out-of-sample baselines, not the unknown platform model;
core references are means of all features and the two fixed alternating-column halves, not known core members.
Six benchmark/core scenarios are sensitivity controls; none is a platform score or evidence of a specific rank.

Eleven prespecified comparisons: four trained challengers and their five-member consensus, each with 0 or
0.25 neutralization against the all-feature core proxy; plus neutralizing the refreshed baseline by 0.25.
Primary endpoint: bounded proxy score vs the refreshed baseline under refreshed-benchmark/all-feature-core.
Use paired 20-era block bootstrap with 10,000 draws and Holm across all eleven primary comparisons. Retain
the existing FIT effect floor (0.001), p_pos >= 0.95 and worst-window guard (-0.001); additionally require
score gain >= 0.002, at least two positive score windows, no score window below -0.001, and nonnegative
aggregate score improvement in *every* sensitivity scenario. Report 40/60-era bootstrap sensitivity without
changing the decision. These are adaptively reused historical development estimates, not globally controlled
future-performance claims. The 15% threshold remains an operational major-gain label, never a promised result.

Discovery reader excludes eras >= 5300 at the parquet read. Freeze one eligible winner before any confirmation
read; no runner-up substitution. Only then may a sequential confirmation simulate 5340–5924, with the same
algorithms, controls and gaps. Confirmation must have positive FIT and primary proxy-score paired gains with
p_pos >= 0.95, both worst-half gains >= -0.001, and nonnegative six-scenario score differences. It is
historically viewed, round-locked evidence, not a fresh holdout. Eras >= 5925 remain unread. No automatic
deployment: unknown platform references still require separate prospective calibration and artifact checks.

Bind checkpoints to config, source, environment versions and data hashes. Check actual serialized boosters
against original predictions before accepting a checkpoint. Resume only byte-verified completed fits; never
reuse a changed plan. Preserve failures and run matched diagnostics before rejecting underperforming ideas.
Keep `baseline01` unchanged. No `.env` reads, network training, platform uploads, submissions, slot changes
or staking. CPU LightGBM is intentional here; further long neural training previously overfit.

## Prespecified cause checks

After selection, run descriptive matched target/count controls for score-aware expanding and rolling models.
For each chronological block, refit the raw target at the score-aware donor's selected count, and the score-aware
target at the raw donor's count. The two original fits complete a two-by-two target/count table. Thus changing
the target can be examined at fixed count, and changing the stopping count at fixed target. No diagnostic fit
can enter selection, confirmation, consensus or deployment; report adverse windows and interactions. The donor
counts were selected exclusively within the same available training partition. These controls isolate effects
of transferred counts, not a complete counterfactual comparison of different stopping objectives on identical
learning trajectories. The raw-MSE count was selected for the raw-target trajectory, not the score-target one.
