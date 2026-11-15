# Prequential model-family weighting 14

adaptive historical OOS weight learning; composite metrics use proxies

| policy | FIT | proxy score | FIT gain | score gain | Holm | eligible |
|---|---|---|---|---|---|---|
| adaptive_fit | 0.02589410 | 0.03873656 | +0.00160287 | +0.00497443 | True | True |
| adaptive_score | 0.02602050 | 0.03933109 | +0.00172927 | +0.00556897 | True | True |
| fixed_prior | 0.02578142 | 0.03818340 | +0.00149020 | +0.00442128 | True | True |

Confirmation: {"confirmation_scored": false, "plan_sha256": "796d300921e1cb21daf1a4cd92f89e22e5b577dd5c5b5d1de08bdafa840bfa71", "qualified": false, "reason": "one all-feature-missing row at exped_5582 made the calibration core reference unavailable; no confirmation performance verdict was produced", "status": "failed_input_contract"}

Native discovery models were immutable source-verified ranking13 dependencies; none was retrained.
Every weight fit excluded labels at or after block_start minus 40. Private traces retain exact boundaries and weights.
Baseline01 remains unchanged. No uploads, submissions, staking or deployment changes.

Discovery passed all registered gates and froze adaptive_score. Its FIT gain was 7.12% vs the refreshed
baseline and score-proxy gain 16.49%, with positive FIT and score differences in all three windows. The fixed
control also passed versus baseline; most average improvement is attributable to the conservative fixed
mixture. Adaptation's incremental FIT over the fixed control was only 0.00023908 (descriptive paired p_pos
0.8733, p=0.1300); its proxy-score increment was 0.00114769 (p_pos 0.9930, p=0.01240). These secondary tests
were not part of a fresh corrected selection family and do not prove a general adaptive advantage.

Confirmation has no performance verdict: all native refits completed, but calibration rejected one genuine
all-feature-missing row in exped_5582. The strict reference-availability contract failed before scoring the
chosen policy. Original data and row IDs were verified; this was not an alignment error. Original artifacts
and code are preserved. Missing-core semantics will be repaired under a new run ID with exact discovery
replay, frozen winner, identical native bytes and unchanged statistical gates before confirmation resumes.

Operator issues: the first confirmation command ran in the GPU-hidden sandbox and failed before native
training; it was restarted unchanged with GPU access. A report was also invoked prematurely before a
confirmation artifact existed; reporting was deferred. No algorithm or parameter was changed for those
launch issues. The later missing-reference failure is recorded separately as failed_input_contract.

Validation before repair: full suite 482 passed without skips. Source 1d3512b. This is historically reused
research data; no independent future improvement, major gain or platform score is claimed.
