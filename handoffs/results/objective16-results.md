# Honest supervised utility and historical replication 16

historical earlier replication and viewed later intervals; local score proxies

| interval | algorithm | FIT | baseline FIT | proxy score | baseline proxy | historical gates |
|---|---|---|---|---|---|---|
| earlier_replication | cpu_full | 0.00694412 | 0.01690444 | 0.01305257 | 0.02369472 | False |
| earlier_replication | cpu_full_blend | 0.01345197 | 0.01690444 | 0.02081151 | 0.02369472 | False |
| earlier_replication | gpu_full | 0.00621326 | 0.01690444 | 0.01026788 | 0.02369472 | False |
| earlier_replication | gpu_full_blend | 0.01363597 | 0.01690444 | 0.01994706 | 0.02369472 | False |
| earlier_replication | full_committee | 0.00768795 | 0.01690444 | 0.01383555 | 0.02369472 | False |
| earlier_replication | full_committee_blend | 0.01407673 | 0.01690444 | 0.02161124 | 0.02369472 | False |
| viewed_development | cpu_full | 0.02409077 | 0.02702630 | 0.04201138 | 0.03790877 | False |
| viewed_development | cpu_full_blend | 0.02815804 | 0.02702630 | 0.04368427 | 0.03790877 | False |
| viewed_development | gpu_full | 0.01895314 | 0.02702630 | 0.03003085 | 0.03790877 | False |
| viewed_development | gpu_full_blend | 0.02505213 | 0.02702630 | 0.03660777 | 0.03790877 | False |
| viewed_development | full_committee | 0.02294606 | 0.02702630 | 0.03834951 | 0.03790877 | False |
| viewed_development | full_committee_blend | 0.02683755 | 0.02702630 | 0.04053176 | 0.03790877 | False |
| viewed_transfer | cpu_full | 0.00420219 | 0.01526832 | 0.00922545 | 0.01868202 | False |
| viewed_transfer | cpu_full_blend | 0.01180987 | 0.01526832 | 0.01653340 | 0.01868202 | False |
| viewed_transfer | gpu_full | 0.00253305 | 0.01526832 | 0.00255651 | 0.01868202 | False |
| viewed_transfer | gpu_full_blend | 0.01027885 | 0.01526832 | 0.01229874 | 0.01868202 | False |
| viewed_transfer | full_committee | 0.00410076 | 0.01526832 | 0.00723332 | 0.01868202 | False |
| viewed_transfer | full_committee_blend | 0.01059511 | 0.01526832 | 0.01390434 | 0.01868202 | False |

246 checkpoint groups; actual serialization replay max error 0.
No native candidate was promoted. All baselines use the new, consistently declared era-400 training floor.
These are historical proxy scores, not platform scores or independent future evidence. Baseline01 unchanged; no platform writes.

No comparison passed all historical gates. Full-label CPU/GPU fits and their fixed committee failed the
earlier replication and viewed transfer, and blending only partially recovered the baseline. The development
CPU-full blend reached 0.02815804 FIT, but that isolated high is not a general improvement: its transfer FIT
was 0.01180987 vs 0.01526832. Do not promote it or use its development maximum as a success claim.

Matched controls: GPU full utility lost to raw-target GPU FIT in every interval (0.00621326 vs 0.00824086;
0.01895314 vs 0.02137180; 0.00253305 vs 0.00491660). The supervised utility intervention did not solve the
gap in this setup. CPU full/core effects and stopping-count interactions are investigated in objective17;
do not conflate label effects, selected counts and stochastic GPU realization differences.

History lead, not established improvement: the otherwise unchanged expanding baseline with floor 400
achieved development FIT 0.02702630 vs the old floor-2800 baseline 0.02429122 (+11.26% observed), and
transfer 0.01526832 vs 0.01498275 (+1.91%). Paired p_pos was only 0.9132 / 0.5482 respectively; both 90%
intervals included zero. This motivates a direct raw-target history control, not adoption or a claim of a
reliable gain. Older data's usefulness depends on the learner and context; prior count-source failures
do not establish that all older training data are harmful.

All 51 OOS teacher models, 117 CPU primary models and 156 GPU seed models are byte-bound; the 246
checkpoint groups include two seeds per GPU group. All actual native serialization replays had zero error.
No provided train benchmark was accessed. Source 7535903; final existing suite 494 passed without skips.
Earlier replication was historical and has now also been scored; none of these intervals is fresh future
confirmation. No code, count grid, target definition or gate changed during fitting.
