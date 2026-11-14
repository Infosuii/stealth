# Cause-driven repairs 07

historical development selection estimates; not a fresh holdout

| recipe | blend | FIT | baseline FIT | gain | p_pos | Holm pass | eligible |
|---|---|---|---|---|---|---|---|
| balanced_denoiser | False | 0.024894 | 0.021811 | +0.003083 | 0.9175 | False | False |
| balanced_denoiser | True | 0.024776 | 0.021811 | +0.002965 | 0.9901 | True | False |
| preserved_latent_trees | True | 0.020742 | 0.021811 | -0.001069 | 0.1301 | False | False |
| preserved_latent_trees | False | 0.017755 | 0.021811 | -0.004056 | 0.0160 | False | False |

Frozen discovery winner: None

Confirmation:
```json
{
  "promoted": false,
  "major_improvement": false,
  "reason": "no eligible discovery challenger",
  "plan_sha256": "2ce5cbfd286faebf9a97a9104ec4f56f9a8361a9c33309c3f6f1f7868cd6f7ac"
}
```

No old audit eras, platform writes or staking. No candidate is an independently proven live improvement.

## Which repairs worked and what remains unresolved

The balanced denoising head increased aggregate historical FIT from 0.024319 to 0.024894 (+14.13% versus baseline). Its fixed baseline blend reached 0.024776 (+13.59%); p_pos 0.9901, null-centred p-value 0.01130, passing this round's four-comparison Holm gate. It still failed the late-window bound: blended late FIT was 0.026073 versus baseline 0.027527. The major-improvement criterion was not met. No candidate was promoted, confirmation was not loaded, and no deployment pickle was exported.

The group repair helped the first two windows; it did not solve temporal instability. Keeping the same encoder and refitting a group-balanced head isolates the prediction-head change. All nine outer encoders were reused only after data/partition/normalization/hash checks; 5,400 inner encoder epochs were trained and 5,400 prior encoder epochs reused. No reused work is counted as new training.

The retuned raw-preserving latent-tree algorithm lost to the baseline, despite the matched frozen-parameter raw-input rescue in forensics06b. Therefore the partial representation rescue does not establish that the nested model-selection procedure chooses a better future model. The next investigation must separate model-class capability from selection failure, and retain the contradicted count-only hypothesis. Do not infer that the idea is useless from its aggregate.

All nine fold ensembles (21 serialized models) replayed, including the exact two-stage Gaussian-rank averaging rule, raw and baseline-blended outputs, shuffled columns and row permutations. Final suite: **457 passed, 1 skipped in 146.27s**, the skip being the no-device path on the CUDA-capable host. Duplication/padding invariance, safe encoder reuse and clean-interpreter rank-rule portability passed.

Protocols 62df73d and the preceding forensic controls; implementation 2d42f94; plan `2ce5cbfd286faebf9a97a9104ec4f56f9a8361a9c33309c3f6f1f7868cd6f7ac`. Source-bound checkpoints under artifacts/repairs07. Data remain historical and repeatedly viewed; corrected within-round statistics are not globally independent evidence. No audit rescoring or platform writes.
