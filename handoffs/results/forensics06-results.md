# Why the promising models underperformed

All findings below use already-viewed discovery data below era 5300. Confirmation and old-audit labels remain unopened in this investigation. Controlled ablations diagnose mechanisms; they do not promote a model or create independent future evidence.

## Supervised networks

Across seeds, median training-loss reductions were about 61% for the single-target model and 55% for the multi-target model, while validation FIT fell about 0.020–0.021 from its peak. Checkpoints selected at epoch zero or very early epochs show that the 600-epoch schedule did not suffer from insufficient optimization. The evidence supports overfitting; longer fitting alone is not the remedy.

## Tree-count transfer: hypothesis contradicted

The inner best counts varied strongly over time, suggesting that transferring their median might underfit the latest period. Keeping everything else fixed and using the latest inner count made late latent-tree FIT worse: 0.020151 → 0.019535. It also failed to fix the late raw-tree shortfall. Do not adopt a latest-count policy from this hypothesis.

## Ensemble scaling: a real but partial issue

Averaging raw model scores implicitly gives greater influence to members with larger output amplitudes. Frozen Gaussian-rank averaging improved aggregate raw-tree FIT from 0.021759 to 0.022174, and latent-tree FIT from 0.019227 to 0.020985. Benefits varied across windows; this does not explain the complete late failure or establish a major improvement.

## Irreversible latent bottleneck: controlled rescue

In the late model, retained PCA directions contained about 83% of the training mean linear feature/target-covariance vector norm, but about 50% in the test period. This is a linear-covariance diagnostic, not a measure of all predictive information. A matched intervention preserved raw inputs while retaining the learned projection, parameters, original counts, weights and seed. Late FIT rose 0.020151 → 0.024301, recovering more than half the gap to the 0.027527 baseline. Early/middle models already preserved raw inputs and were unchanged, providing identity controls. The model still underperformed the baseline, so preserving raw inputs is a supported partial repair, not a solved predictor.

## Denoising: useful representation, diluted prediction head

With all learned weights frozen, the encoder-only term reached aggregate FIT 0.024906 versus 0.024319 for the complete predictor; its baseline blend reached 0.024915. The raw-feature term alone reached 0.022715. The encoder improves the first two windows; in the late window both individual terms lose, while their combination partially diversifies the error. This supports investigating separate group regularization and refitting the latent head; it does not justify simply deleting the raw term based on the aggregate.

## Temporal relationship changes

Feature/target covariance profiles changed materially across time, with roughly one fifth of the raw-head training contribution mass changing covariance sign in the late test. Marginal feature means changed little; missingness also changed, especially in the early window. These diagnostics support temporal relationship instability and missingness as contributors, but do not isolate every causal factor or prove that all covariate shift is absent.

## Next controlled repair

Keep raw inputs available in latent-tree fitting, align equal-model ensembling with rank-based scoring, and test group-balanced denoising heads while retaining a raw-feature fallback. Register a new algorithm before scoring; reuse only verifiably target-free encoders fitted on exactly the same training partition. Keep all completed results and gates unchanged.

The original checkpoint hashes and data hash were verified before all ablations; original predictions replayed. New fits were written only under artifacts/forensics06 and artifacts/forensics06b. No uploads, submissions, model changes or staking. Protocols 834e84f/a448793; implementation acb8351/a448793.
