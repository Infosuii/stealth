# Repairing unstable model selection with broad consensus

Registered after forensics08, before computing new ensemble scores. Original inner objectives ranked later
individual performance weakly or negatively, so higher search scores did not reliably identify next-period
winners. Some candidates were useful but were not selected; other populations were weak throughout. Never
choose a diagnostic oracle winner.

Test two frozen native-LightGBM learning algorithms from `configs/consensus.yaml`:

1. Search population: keep the original 64-trial TPE algorithm, own-inner stopping counts and median count
   transfer, but average every trial's per-cross-section Gaussian score instead of selecting the top three.
   Reuse original discovery studies only after data, window, seed, inner layout and training-budget checks;
   clone them into the new run without modifying source files. Missing confirmation/full studies run the
   same registered search. This removes sharp winner selection while retaining adaptive proposal bias.
2. Prior population: use the same parameter distributions and training budget with a fixed random sampler;
   do not use trial objective values to adapt proposals or choose members. Average all registered models.
   This additionally removes adaptive proposal bias while retaining the same own-inner stopping procedure.

Keep the two initial fixed baseline/slow-learning probes from the original study; all remaining proposals
use the frozen sampler seed. Individual constant predictions contribute zero Gaussian score. No model is
dropped according to later FIT. Use the same explicit `gaussian_mean` blob rule in evaluation and portable
inference. The equal model prior is the repair being tested, not a guarantee that averaging weak models helps.

The existing readers, four comparisons, chronological splits, bootstrap, gates, single-winner confirmation
and operational major criterion remain unchanged. This is adaptive historical research, not independent
future evidence. Preserve every failure, validate actual serialized models and byte-bound artifacts if one
qualifies, and perform no audit rescoring, uploads, live submissions, model-slot changes or staking.
