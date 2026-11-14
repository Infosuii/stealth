# Research round 02 — training objective and stopping metric

Registered after round 01 rejected every architectural challenger, before any round 02 discovery scores.
Round 01 confirmation was not scored. This is an adaptive follow-up on already-viewed discovery data;
its discovery statistics are exploratory, even with correction within this round. They are not a global
guarantee against repeated research or independent future evidence.

## Hypothesis and fixed ablations

The incumbent uses pooled raw-target squared error for training and inner stopping. FIT20 instead averages
cross-sectional covariance across eras. Removing fit-era target means and equalising era mass may reduce
irrelevant variation. Choosing the tree count on inner FIT20 may better match the evaluation objective.

Compare four fixed recipes: era-centred/equal-era training with MSE stopping; raw-target training with FIT
stopping; era-centred/equal-era training with FIT stopping; and a three-seed ensemble of that combined recipe.
All recipes retain the incumbent's tree parameters and the same chronological inner split and purge.
FIT stopping uses only the recipe's own inner validation eras. The full outer fit freezes that count.
Targets are centred separately within each fitting partition, never using outer test labels. MSE validation
labels are centred using only that inner validation era. FIT always scores the original raw target.

Every recipe is compared standalone and as the same fixed equal Gaussian-rank blend with the baseline.
No seed, parameter or mixture weight is added after scores. Config: `configs/research-objectives.yaml`.

## Frozen evaluation and delivery

Use round 01's three expanding chronological windows and exclude eras 5925 onward at the parquet reader.
Require positive paired gains in at least two windows, a minimum mean gain, a bounded worst window, and
Holm correction across all eight comparisons at alpha 0.05, plus bootstrap stability. The bootstrap and
effect thresholds are fixed in the config. Freeze at most one discovery winner, then score confirmation
5340–5924 once. Positive confirmation gain and p_pos at least 0.90 are required; no runner-up substitution.

If no candidate qualifies, retain the baseline and stop this research pass. Confirmation remains historically
viewed data, locked only within this pass. Export only a candidate that passes both gates; check the actual
pickle bytes in the clean interpreter before any later use. Write and push all results, including failures.
No platform uploads, live predictions, model changes or staking.

Implementation follows the official LightGBM custom evaluation interface:
https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.train.html
