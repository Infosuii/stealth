# GPU ranking and model-family diversity, round 13

Registered after temporal11 and before any new ranking discovery scores. Count12 is an independent cause
check, with no ranking model, target or gate chosen from its scores. The current historical candidate still
loses late FIT. Existing neural schedules overfit, so extending them is unsupported. Test a new learning
bias: within-era pair/listwise ranking versus CatBoost regression and query-centered regression. Ranking
is not known to optimize FIT exactly; it is a controlled hypothesis about relative-order learning.

A synthetic GPU YetiRank rehearsal passed on the RTX 5070 Ti outside the sandbox. Training uses the absolute
CPU venv interpreter, whose pinned CatBoost package supports GPU; CUDA is hidden inside the default sandbox.
The native training call must require the GPU and fail if it is unavailable. No silent CPU training fallback.
Read-only CPU prediction of a trained GPU model is intentional and is checked after serialization.

Three CatBoost losses share history, parameters, split and selection rule: RMSE; QueryRMSE; YetiRank Classic.
Use two fixed seeds, 17 and 29, with fixed equal per-era Gaussian-rank averaging. No seeds or losses are added
according to scores. Use a rolling 1,024-era final history, latest 200-era inner validation and a 40-era gap.
Train each inner trajectory for 2,048 iterations. Evaluate the same explicit bounded composite proxy on
checkpoints 1, 4, 16 and every 32 iterations, selecting the earliest maximum. All losses use the same fixed
200-tree honest out-of-sample inner benchmark and all-feature core proxy. Native loss-specific validation
metrics never choose a count. Refit the final available training partition at its own count. Keep every inner
score curve and flag a cap optimum as unresolved optimization limits. GPU fitting is not bit-deterministic;
persist and verify actual trained bytes so a resume never swaps in a fresh realization.

Training targets are the raw primary target, and each rank query contains only one exped's instruments.
No train-split platform benchmark is used. Earlier target-free encoders, target variants and tuned arms are
not mixed in. The shared backbone has depth 5, learning rate 0.02, L2 20, 16 borders and Bayesian bootstrap;
it is fixed before any ranking results. This substantial inner fitting budget measures whether longer
ranking optimization helps; do not claim that training duration itself establishes predictive improvement.

Copy temporal11's seven original member checkpoints and static baselines only after verifying their source
plan, data hash, code hashes and actual file bytes. Retain their immutable fitted models and prediction rules.
The copied members are controls/dependencies, not newly trained models. Build the exact original fixed
history/core/stable groups from those copied members, then test a fixed equal blend of the stable group and
a ranking group (QueryRMSE plus YetiRank). Rank groups before combination, preserving equal group influence.

Eight comparisons, all at zero post-neutralization: each of three GPU algorithms alone and in a fixed equal
baseline blend; the QueryRMSE/YetiRank ranking consensus; the cross-family stable/ranking consensus. Preserve
all existing eight-comparison Holm, FIT/score effect floors, temporal guards, bootstrap, six reference
sensitivities and single-winner confirmation gates. No retrospective gate changes. These are adaptive
historical development estimates and proxy scores, not platform results or independent future evidence.

Discovery reader stops before 5300. Freeze one eligible winner before any confirmation reader. Confirmation
fits only that winner's native dependencies and its baseline using the same sequential 5340–5924 schedule;
all readers exclude eras >=5925. Source-bound caches cover config/code/data/library versions and actual GPU
fit metadata. Verify all serialized native models' CPU predictions, preserve all failures and investigate
whether a loss, stopping rule, seed variation or ensemble dilution explains underperformance. Baseline01
remains unchanged. No uploads, submissions, platform slot changes, staking or `.env` access.

Official references: https://catboost.ai/docs/en/concepts/loss-functions-ranking and
https://catboost.ai/docs/en/features/training-on-gpu . Native objectives and GPU support were checked;
their usefulness on this dataset remains an empirical question.
