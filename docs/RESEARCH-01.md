# Research round 01 — architecture and chronological stability

Registered before scores on 2026-11-14 after the user requested a substantial predictive improvement and
authorised extensive local compute. Config: `configs/research.yaml`. CPU work and the GPU neural model may run
concurrently. The old LightGBM/XGBoost grids will not be enlarged.

## Hypotheses

The existing tree ensembles mainly repeat related weak signals. Test models with different inductive biases:
Gaussian-bin ridge, smooth/cubic additive effects, categorical-bin shrinkage, compact quadratic interactions,
ridge-plus-residual trees, and an era-aware neural model trained against within-era correlation. These models
are stateless at prediction time and can be exported to the pinned clean runtime. Every feature transform used
at inference is shared with training.

## Frozen evaluation

Use expanding chronological training windows starting at era 2800, with a 40-era gap before each test window.
Discovery test windows: 4000–4499, 4500–4999, 5000–5299. No discovery fit or early stop sees later labels.
All fitted centering, regularisation, residual models and neural stopping decisions use the fit window only;
an inner chronological split selects epochs/tree counts before the outer fit. Era weighting and target
centering apply only inside each fit. Per-era target-normalisation statistics never come from an outer test.

Compare each registered architectural recipe both directly and as a pre-specified equal Gaussian-rank blend
with the same fold's unsearched LightGBM baseline. No arbitrary mixture-weight search. Rank candidates by the
paired development FIT gain after the registered significance/stability gates; correct the discovery null
p-values across all comparisons by Holm step-down. All failed candidates remain in the results.

Freeze ONE candidate before scoring confirmation eras 5340–5924. Require a positive confirmation difference
and the registered paired stability threshold to promote it. If it fails, retain the baseline; do not choose
the runner-up using the same confirmation labels. No new fit or choice reads eras 5925–6524, and the previous
audit/diagnostics are never used to tune this round.

These historical development/confirmation eras have already been viewed by earlier experiments. “Confirmation”
means locked for this research round, not a fresh independent holdout. A positive result would justify a
prospective evaluation, not a claim of proven future returns. No performance improvement is promised in advance.

## Compute and delivery

Use the laptop CPU for regularised fits and residual trees, and its GPU for the era-aware neural challenger.
The neural recipe averages its three pre-registered initialisations; no seeds are added after seeing a score.
Check stability over all three chronological windows rather than spending the budget on unconstrained tuning.
Checkpoint each model/fold so interrupted work resumes without changing its specification.

Preserve the old baseline and all original audit evidence. Publish code, metrics and the model decision;
keep data, hyperparameter detail in private run artifacts, weights and pickle files out of git. Any promoted
artifact must pass byte-bound clean-venv portability and offline prediction checks. This round performs no
platform upload, live submission, model-slot change or staking.
