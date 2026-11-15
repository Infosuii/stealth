# Stabilizing temporal score-aware learning, round 11

Registered after temporal10's frozen results and target/count controls, before computing any new scores.
The score-aware rolling model improved every aggregate composite-proxy scenario but failed statistical,
FIT-effect and latest-window guards. Both target and stopping-count interventions contributed, with an
interaction; simply transferring the raw-model count harmed it. Expanding-history score-aware learning
underperformed. A single recent 200-era stopping sample may be noisy, and a single history/core proxy may
be brittle. These remain hypotheses to test, not established explanations of failure.

Preserve round 10's data reader boundaries, 100-era refresh cadence, 40-era availability gap, raw refreshed
baseline, inner benchmark fitting, composite kernels, six sensitivity scenarios and all effect/stability gates.
The config is `configs/robust-temporal.yaml`. No prior round's gates or candidate labels are changed. This
follow-up is adaptive historical research, not an independent confirmation of round 10 or global error control.

Fit fixed score-aware models on 512-, 1,024- and 2,048-era histories. Retain all three in a fixed equal Gaussian
consensus; do not choose a history using their later performance. Separately fit the 1,024-era algorithm using
the all-feature and two alternating-half core means for its training target, retaining all three in a fixed
equal consensus. Stop every trajectory on the same all-feature composite proxy, so the training-core
intervention does not change the selection reference. These arbitrary halves test core-reference sensitivity;
they are not asserted to identify the hidden platform core or independent economic factors.

Test a 1,024-era score-aware model whose tree count is the median of three disjoint 200-era inner validation
windows, each with its own earlier fit and 40-era gap. All three counts belong to that outer fitting partition;
no later period can supply a count. Keep the original target and final available-data refit. The comparison
isolates this count-transfer policy against the original single-inner 1,024-era control.

Finally test fixed equal *group* consensus of the refreshed baseline, history consensus, core consensus and
multi-window model. Rank each group's output before combining, so arbitrary model scales or duplicate group
sizes do not change its weight. Eight primary comparisons: four algorithms (history/core/group consensus and
multi-window stopping), each with 0 or 0.25 all-feature neutralization. Holm is across all eight; use round
10's remaining gates verbatim. Components and the repeated original rolling control are descriptive only;
none may be selected or substituted for a failed ensemble. The repeated control must replay round 10's
actual model predictions, modulo declared serialization precision, before reporting this round.

Freeze at most one eligible winner before confirmation. Only that winner and its dependency models can train
on the same sequential 5340–5924 confirmation schedule. Apply the existing confirmation gates, keep eras
>=5925 unread, and do not promote to deployment on proxy evidence. Preserve all component results and exact
counts to investigate dilution, horizon sensitivity, core-reference sensitivity and count changes. Report
adverse windows explicitly. No longer neural training, platform calls, uploads, submissions or staking.
