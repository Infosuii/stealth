# Raw-target history controls, round 18

Registered after objective16/17 and before new history-model scores. Extending the unsearched baseline's
training floor from 2800 to 400 raised observed means, but paired uncertainty remained substantial. Utility
target shaping did not generalize and mixed with stopping-count effects. Test data history directly while
retaining raw supervision and each backbone's existing stopping recipe.

Three fixed histories per backbone: latest 1,024 eras; expanding from era 400; expanding from the first
available era (3). CPU remains the fixed unsearched LightGBM with own latest-200 MSE stopping and 40-era
gap. GPU remains CatBoost RMSE, existing parameters, two fixed seeds 17/29 and full 2,048-iteration inner
trajectories with the same explicit composite-proxy checkpoint grid. No new parameter search, target,
loss or seeds. GPU history changes only which rows precede its unchanged inner split and final refit.
CPU/GPU stopping differs as before; interpret history effects within each backbone, not as isolated
cross-backbone causal effects.

Reuse objective16's expanding-floor400 CPU and rolling raw GPU controls only after source plan, data,
config, effective row ranges and all actual file bytes verify. The rolling GPU's first final history starts
at 936, so reading earlier context does not change any of its fitting rows. Other histories are new fits.
Source GPU realizations remain fixed; seed matching on new GPU fits does not imply bit-deterministic
counterfactual randomness. Teacher predictions are unused by raw supervised fitting but retained unchanged
as an evaluation reference and provenance dependency.

Eight fixed comparisons: CPU expanding-from-3, GPU expanding-from-400, GPU expanding-from-3 and balanced
history consensus, each alone and as an equal baseline blend. The consensus first averages Gaussian scores
across the three histories within each backbone, then Gaussian-ranks and equally averages those two groups.
The expanding-floor400 CPU remains the primary baseline; raw rolling CPU and all component histories are
descriptive controls, never silently substituted for an eligible policy. No history weight is tuned.

Use the same 39 rolling blocks in 2000–3999, 4000–5299 and 5340–5924. ALL ARE NOW VIEWED. Report adaptive
historical threshold checks with the existing eight-family Holm, effects, p_pos and window guards, six
reference scenarios and bootstrap sensitivity. No automatic promotion or prospective-performance claim.
Keep objective16's older floor-2800 baseline comparison as contextual evidence; do not manufacture gains
by switching the primary comparator. Report chronological failures and whether more data helps each
backbone, rather than treating a maximum development score as a reliable major improvement.

Reader permits >=3 and excludes >=5925. Every fit and label-dependent stopping operation is still strictly
before its prediction block minus 40. Bind sources/config/data/library identities, original control markers
and teacher manifest. Verify copied controls, actual serialized predictions and strict row filtering.
Preserve all earlier failures and baseline01. No `.env` access or platform writes, uploads, submissions,
slot changes or staking.
