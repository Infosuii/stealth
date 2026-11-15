# Separating count-transfer effects, round 12

Registered after temporal11, before new scores. Multi-window count aggregation harmed FIT. Seven unchanged-count
blocks replayed exactly, while older sources had both fewer eras and older regimes. Earlier results do not tell
us whether reduced training size, time coverage, or temporal shift caused the harmful count choices.

Repeat the refreshed baseline, original single-inner score-aware rolling control and ragged three-window model.
Test three prespecified count-source interventions on the same three inner validation windows:

1. Spread: use the same number of fitting eras as the smallest original inner fit (384 in this layout),
   selecting evenly spaced whole eras across each original source's available time span. No labels determine
   the subset. This holds era count fixed while preserving broad time coverage.
2. Recent: keep the same number of eras, but take the most recent contiguous history of each original source.
   Compare with spread to examine coverage at fixed era count. Instrument counts can differ between eras;
   these are equal-era-count controls, not claims of identical sample counts or distributions.
3. Full span: use 784 contiguous eras before every inner validation window with the same 40-era gap. That is
   the original latest-inner fit length, now supplied to all three sources. This needs earlier context below
   era 2800. Reader loads eras >=2400 and <5300, but final model fitting still starts >=2800 and uses exactly
   the same last 1024 production eras as the original rolling control. Earlier context affects only inner
   stopping, not final model rows. Preserve the caveat that changing history also changes its composition.

Every trajectory retains the original score-aware target, fixed 200-tree OOS inner benchmark and bounded
composite-proxy stopping. Transfer the median of its own three counts to the identical final partition.
Save complete inner stopping curves and exact fit/validation era ranges, counts and row totals. Earlier-data
access is training-only context, never confirmation or old-audit data. No provided train benchmark is used.

Four algorithms are eligible: spread, recent, full span, and a fixed equal Gaussian-rank blend of full span
with the refreshed baseline. Each is tested at 0 and 0.25 all-feature neutralization: eight comparisons.
Keep temporal10/11's composite proxy sensitivity scenarios, effect floors, FIT and score stability guards,
bootstrap, Holm correction and single-winner confirmation rules unchanged. Existing failed gates remain
failed. This remains adaptive historical research, not globally controlled or independent future evidence.

Before accepting results, baseline/single-inner controls must match temporal10 and ragged control must match
the actual temporal11 multi-window predictions. No component may be substituted after scores. Report all
failures and which explanations are supported, contradicted or unresolved. Baseline01 remains unchanged;
no deployment on proxy evidence. No platform calls, uploads, submissions, staking or `.env` access.
