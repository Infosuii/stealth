# Campaign 03 — sustained optimization and longer training

The user explicitly requested renewed work after the first limited experiments failed, including substantially
longer training. Objective: a measured major improvement, defined before results as at least 15% relative FIT
gain over the contemporaneous unsearched baseline on discovery and round-locked confirmation, with positive
paired bootstrap stability. A smaller robust gain is incremental progress and must be labelled accordingly.
No gain is promised, and test counts or training duration are not performance achievements.

## Diagnosis and frozen algorithms

Earlier Gaussian-ridge recipes used two fixed penalties after changing feature scale. The effective strength
was not calibrated against feature variance and sample mass. Earlier residual networks stopped on short
inner patience. Longer fitting alone can overfit; this campaign tests it with independent chronological
model selection, strong regularization, multiple targets, weight averaging and multiple initialisations.
Exploratory pre-5300 feature/target profiles also varied markedly across 500-era periods. This motivates
testing recency inside training, not choosing it on an outer test.

Config `configs/campaign.yaml` registers four complete learning algorithms:

1. A ridge regularization path scaled by the fit covariance diagonal and sample mass, jointly choosing
   history length and penalty on three purged inner chronological windows. The entire algorithm, not each
   inner setting, is scored on the outer test.
2. A 64-trial nested LightGBM search per outer fit, with up to 6000 boosting rounds and longer inner patience.
   Parameters include capacity, regularization, subsampling, training history and target centring. Rank
   trials by mean inner FIT minus a fixed between-window instability penalty. Average the top three refits.
3. A strongly regularized single-target neural residual model with a full 600-epoch inner schedule.
4. A larger multi-target model with the same long schedule, shared hidden layers and six auxiliary targets.

Both neural recipes use their three fixed seeds, EMA weights and GPU-resident validation. Train the inner
schedule fully, recording training and inner-validation curves; select the checkpoint there, then refit the
outer training partition with that epoch count and the same learning-rate trajectory. An epoch-zero skip-only
checkpoint may win: if so, report it honestly rather than forcing a longer overfit model. Export only the
primary head to the existing NumPy runtime, with arithmetic parity checked against the actual GPU model.

## Evaluation and stopping

The same three expanding discovery windows remain exploratory and historically viewed. The data reader
excludes confirmation labels during discovery and always excludes eras 5925–6524. All hyperparameters,
history choices, tree counts and checkpoints use only the outer fit partition. No old audit or diagnostics
are rescored or used to tune this campaign. Each algorithm is tested standalone and as the registered equal
Gaussian-rank blend with the baseline: eight outer comparisons, with Holm correction and the fixed effect,
stability and worst-window gates. Freeze one winner before loading confirmation labels. Confirmation is
historically viewed data, locked for this campaign, not independent future evidence. No runner-up fallback.

Export only a positively confirmed candidate; preserve the incumbent. Run the actual artifact through the
byte-bound clean-interpreter check and offline prediction contract. Publish complete results and failures,
including actual epochs/tree counts, wall time, training curves and whether the major-improvement target
was met. Stop this campaign after the registered algorithms and report the measured result. The user's broad
authorization permits local code/training work; this campaign makes no platform upload, live submission,
model-slot change or stake.

References: LightGBM's official parameter and custom-evaluation documentation, and PyTorch's official
`AveragedModel` documentation. These inform implementation; they do not establish performance on this data.
