# Long-history core utility 19

viewed historical long-history core utility and count controls

GPU core utility with expanding-from-400 history reaches viewed-transfer FIT **0.01999896** versus
baseline **0.01526832 (+30.98%)**, primary composite proxy **0.03250764** versus **0.01868202 (+74.00%)**.
It and its baseline blend pass that interval's unchanged historical gates; GPU-core3's baseline blend does
also. ALL comparisons fail earlier replication and development. No candidate qualifies in all three intervals;
no broadly reliable major gain or automatic promotion is established. All periods are viewed historical data.

| interval | policy | FIT | baseline FIT | proxy score | baseline proxy | historical gates |
|---|---|---|---|---|---|---|
| earlier_replication | cpu_core400 | 0.01290315 | 0.01690444 | 0.01949276 | 0.02369472 | False |
| earlier_replication | cpu_core400_blend | 0.01558941 | 0.01690444 | 0.02224619 | 0.02369472 | False |
| earlier_replication | cpu_core3 | 0.01575477 | 0.01690444 | 0.02666048 | 0.02369472 | False |
| earlier_replication | cpu_core3_blend | 0.01719032 | 0.01690444 | 0.02649549 | 0.02369472 | False |
| earlier_replication | gpu_core400 | 0.01189108 | 0.01690444 | 0.01762628 | 0.02369472 | False |
| earlier_replication | gpu_core400_blend | 0.01519794 | 0.01690444 | 0.02157952 | 0.02369472 | False |
| earlier_replication | gpu_core3 | 0.01322465 | 0.01690444 | 0.02165974 | 0.02369472 | False |
| earlier_replication | gpu_core3_blend | 0.01594765 | 0.01690444 | 0.02383906 | 0.02369472 | False |
| viewed_development | cpu_core400 | 0.02531065 | 0.02702630 | 0.03875914 | 0.03790877 | False |
| viewed_development | cpu_core400_blend | 0.02679021 | 0.02702630 | 0.03907065 | 0.03790877 | False |
| viewed_development | cpu_core3 | 0.02440578 | 0.02702630 | 0.03663271 | 0.03790877 | False |
| viewed_development | cpu_core3_blend | 0.02663399 | 0.02702630 | 0.03834973 | 0.03790877 | False |
| viewed_development | gpu_core400 | 0.02459493 | 0.02702630 | 0.03558471 | 0.03790877 | False |
| viewed_development | gpu_core400_blend | 0.02636843 | 0.02702630 | 0.03726483 | 0.03790877 | False |
| viewed_development | gpu_core3 | 0.02527741 | 0.02702630 | 0.03665115 | 0.03790877 | False |
| viewed_development | gpu_core3_blend | 0.02672318 | 0.02702630 | 0.03791034 | 0.03790877 | False |
| viewed_transfer | cpu_core400 | 0.01334085 | 0.01526832 | 0.01789126 | 0.01868202 | False |
| viewed_transfer | cpu_core400_blend | 0.01547230 | 0.01526832 | 0.01958587 | 0.01868202 | False |
| viewed_transfer | cpu_core3 | 0.01752898 | 0.01526832 | 0.02731288 | 0.01868202 | False |
| viewed_transfer | cpu_core3_blend | 0.01693263 | 0.01526832 | 0.02336005 | 0.01868202 | False |
| viewed_transfer | gpu_core400 | 0.01999896 | 0.01526832 | 0.03250764 | 0.01868202 | True |
| viewed_transfer | gpu_core400_blend | 0.01875886 | 0.01526832 | 0.02707538 | 0.01868202 | True |
| viewed_transfer | gpu_core3 | 0.01712167 | 0.01526832 | 0.02732195 | 0.01868202 | False |
| viewed_transfer | gpu_core3_blend | 0.01744362 | 0.01526832 | 0.02444912 | 0.01868202 | True |

468 groups; 156 copied controls; serialization error 0.
Matched-count controls are diagnostic only. No automatic promotion or platform write.

## Why the targets underperformed

The longer histories recover a useful recent GPU core-utility result, so the rolling-window failures alone
were insufficient to retire the idea. Matched counts show that neither labels nor stopping are uniformly
responsible. For CPU core3, using raw-model counts improves earlier FIT by +0.00305743 over raw labels
(descriptive paired p_pos 0.9976), but loses -0.00107753 in development and -0.00242171 in transfer. Its own
stopping counts further reduce earlier FIT by -0.00132314 but recover +0.00175741 in transfer. A label change
that helped the earlier period therefore still failed later even with the same raw count.

For GPU core400, matched-raw-count labels improve transfer FIT +0.00227994 over GPU raw400, and core's own
counts add +0.00187248. These individual descriptive p_pos values are 0.9436 and 0.9321; do not claim each
mechanism independently established. Own counts hurt earlier FIT -0.00285930 relative to matched counts.
Count choice interacts with target and period; the GPU contrast is recipe-matched and subject to native
stochastic variation. This improves the explanation beyond simply rejecting a low aggregate score, but the
underlying economic/representation cause of changing target relevance remains unresolved.

The following paired controls are descriptive, unadjusted diagnostics, not additional eligible candidates:

| interval | challenger minus comparator | FIT delta | FIT p_pos | proxy-score delta |
|---|---|---|---|---|
| earlier_replication | cpu_core400_rawcount − cpu_raw400 | -0.00066231 | 0.2815 | +0.00170763 |
| earlier_replication | cpu_core400 − cpu_core400_rawcount | -0.00333897 | 0.0275 | -0.00590960 |
| earlier_replication | cpu_core3_rawcount − cpu_raw3 | +0.00305743 | 0.9976 | +0.00811057 |
| earlier_replication | cpu_core3 − cpu_core3_rawcount | -0.00132314 | 0.1921 | -0.00252146 |
| earlier_replication | gpu_core400_rawcount − gpu_raw400 | +0.00210995 | 0.9099 | +0.00556929 |
| earlier_replication | gpu_core400 − gpu_core400_rawcount | -0.00285930 | 0.0037 | -0.00510665 |
| earlier_replication | gpu_core3_rawcount − gpu_raw3 | +0.00050216 | 0.6386 | +0.00349214 |
| earlier_replication | gpu_core3 − gpu_core3_rawcount | +0.00002833 | 0.5186 | +0.00056972 |
| viewed_development | cpu_core400_rawcount − cpu_raw400 | -0.00050146 | 0.3436 | +0.00114707 |
| viewed_development | cpu_core400 − cpu_core400_rawcount | -0.00121419 | 0.1708 | -0.00029669 |
| viewed_development | cpu_core3_rawcount − cpu_raw3 | -0.00107753 | 0.1918 | -0.00081974 |
| viewed_development | cpu_core3 − cpu_core3_rawcount | -0.00131228 | 0.1561 | -0.00107661 |
| viewed_development | gpu_core400_rawcount − gpu_raw400 | -0.00178427 | 0.0541 | -0.00180889 |
| viewed_development | gpu_core400 − gpu_core400_rawcount | +0.00077259 | 0.9039 | +0.00184507 |
| viewed_development | gpu_core3_rawcount − gpu_raw3 | -0.00173855 | 0.1429 | -0.00207001 |
| viewed_development | gpu_core3 − gpu_core3_rawcount | +0.00081981 | 0.8854 | +0.00114900 |
| viewed_transfer | cpu_core400_rawcount − cpu_raw400 | -0.00094644 | 0.1978 | -0.00005593 |
| viewed_transfer | cpu_core400 − cpu_core400_rawcount | -0.00098104 | 0.3701 | -0.00073482 |
| viewed_transfer | cpu_core3_rawcount − cpu_raw3 | -0.00242171 | 0.0227 | -0.00317249 |
| viewed_transfer | cpu_core3 − cpu_core3_rawcount | +0.00175741 | 0.7947 | +0.00517237 |
| viewed_transfer | gpu_core400_rawcount − gpu_raw400 | +0.00227994 | 0.9436 | +0.00443546 |
| viewed_transfer | gpu_core400 − gpu_core400_rawcount | +0.00187248 | 0.9321 | +0.00406231 |
| viewed_transfer | gpu_core3_rawcount − gpu_raw3 | +0.00041638 | 0.5854 | +0.00070966 |
| viewed_transfer | gpu_core3 − gpu_core3_rawcount | +0.00080986 | 0.7285 | +0.00170481 |

## Integrity, training length and operations

All 468 checkpoint groups, containing 702 selected native models including copied controls, passed their
source/hash/teacher checks and exact history/outer/inner boundaries. All fixed-count models use the verified
raw block/seed donor count; six CPU own-count source replays were exact. Serialization max error was zero.
468 models were newly fitted (156 CPU and 312 GPU, including matched-count refits); 234 native models in
156 groups were copied byte-verified controls. The 156 GPU inner trajectories completed 319,488 iterations
plus refits. Only one selected the 2048 cap, and 155 endpoints were worse than their selected inner optimum.
This does not support universally longer schedules as the main repair. Full suite: **500 passed, no skips**.

Two launch/environment issues were corrected without source/config/gate changes: the torch-only GPU venv
lacks pytest, so tests ran in the absolute CPU venv with STEALTH_GPU_PY configured; detached tool launches
ended before writing any training checkpoints, so both training stages were relaunched in persistent tool
sessions. Both final stages exited 0. No platform writes occurred during utility19; only preparation reads.

The predeclared upload handoff selects history18 CPU expanding3 as its experimental fallback when no
utility19 candidate passes all three intervals. Preserve the new GPU transfer success as a research lead;
do not retrospectively change that upload rule to the largest observed recent number. Baseline01 remains
the conservative local incumbent. Upload preparation/acceptance is reported separately in handoff06.
