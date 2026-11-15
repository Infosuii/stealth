# Honest benchmark-utility supervised targets and earlier-period replication

Registered before any new native fit or earlier-period score. Prior full-score learning was limited to
stopping or meta weights; the GPU models still fitted raw targets. Score-aware core targets had helped a
CPU rolling model in earlier development but failed transfer. Test a full supervised utility direction
that also accounts for an honest out-of-time benchmark, with matched target/backbone controls.

Never use the provided train-split benchmark. Build our own reference every 100 eras, predicting 880–5924.
Each teacher is the fixed raw-target rolling 1,024-era LightGBM, fitted only on eras strictly before its
prediction block minus 40, with a 200-era inner window and 40-era gap. Training starts at 400; the first
teacher has 440 available eras. Native candidates predict from 2000 onward, so every row of their final
1,024-era fitting window has a teacher OOS prediction. Verify teacher availability for every reference row.
Teacher prediction blocks may be more than 40 eras stale; they must never be less. Preserve actual counts,
model bytes, predictions and source hashes. These are local reference models, not the platform benchmark.

Three training targets: raw y; core utility (centered y plus its valid-core residual with the zero rule);
full utility (add the valid-benchmark residual with its zero rule). Utility targets are centered per fitting
era and scaled to the pooled centered target SD using that fitting partition only. Use complete-case N/M
reference scaling. These are linear directional surrogates, not exact Gaussian-rank/arctan objectives.

CPU controls: expanding unsearched baseline with native MSE stopping; rolling core-utility and rolling
full-utility LightGBM with the same tree parameters and composite-proxy stopping. GPU controls: raw-target
and full-utility CatBoost RMSE, same fixed depth 5/L2 20/learning-rate 0.02, two fixed seeds 17/29, full
2,048-iteration inner trajectories and identical composite checkpoint grids. Refit at each own count.
All rolling final fits use 1,024 eras. Inner metric references remain a separate raw 200-tree baseline
trained only on inner fitting data; validation labels are always the original target. No label transform
or scale can read an outer prediction-period outcome. Matched raw GPU fits are retrained as controls;
GPU nondeterminism means seed matching is not byte-identical counterfactual randomness.

Use 39 rolling 100-era blocks over three fixed intervals: earlier replication 2000–3999; viewed development
4000–5299; viewed transfer 5340–5924. Earlier replication has not been outer-scored for these algorithms,
but its labels were present in prior data work, and algorithm choices come from later periods. It is an
additional historical test, not untouched forward confirmation or independent future proof. All readers
exclude eras >=5925; training context >=400 is now explicitly allowed for this experiment. The baseline
uses that same history floor consistently; do not mix its metrics with the old floor-2800 comparator.
Also report the byte-verified old floor-2800 refreshed baseline on the two overlapping later intervals,
using the same new score references for each side. This is a secondary comparison, not a substituted
primary baseline or a newly selected hypothesis. A gain over a weaker new baseline must not be presented
as an improvement over the older comparator.

Six candidate comparisons in each interval: CPU full utility, GPU full utility and their equal Gaussian-rank
committee, each alone and in a fixed equal baseline blend. Core-utility CPU and raw-target GPU are descriptive
matched controls. Keep previous effect/p_pos/temporal/Holm rules; correct within all six comparisons. Split
each interval into three contiguous segments for the same two-positive-window and worst-window checks.
Use original bounded composite kernels, 20-era bootstrap with 10,000 draws, 40/60-era sensitivity and the
two honest local benchmark references (expanding baseline and rolling teacher), crossed with all/even/odd
feature-core proxies. No algorithm or hyperparameter is selected from a later interval. Report all results
and no automatic promotion, even if historical gates pass. A prospective deployment case still needs new
evidence and byte-bound artifact checks.

Bind every stage to code/config/data/library hashes. Consolidate teacher predictions only after all teacher
checkpoints verify. Bind each candidate to the exact teacher prediction file; never fit to an in-sample
reference or silently reuse a modified model. Resume only completed byte-verified jobs. Validate native
serialized CPU predictions for GPU and CPU models. Record all failures, native task type, actual fitted
counts and whether any selected count hits the cap. Preserve baseline01, all previous failed gates and
the distinction between historical proxy scores and platform results. No `.env` access or platform writes,
uploads, submissions, slot changes or staking.
