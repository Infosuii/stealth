# Online calibration missing-core repair

technical missing-core repair; historically viewed, locked confirmation

Discovery replay max error: 0; frozen winner unchanged.

Confirmation: {"confirmation_scored": true, "label": "historically viewed, round-locked confirmation; no prospective or platform evidence", "major_improvement": false, "plan_sha256": "d532d4e27790cfb4d4c9785f70f1da4ef989085c9872d19bbb23b79f2e53efa2", "qualified": false, "result": {"baseline": {"fit": 0.014982747388714504, "inov": 0.0026395553441138017, "score": 0.01753068673144096, "unq": 0.0}, "block_sensitivity": {"40": {"ci90": [-0.0058965940857759, 4.379818446366574e-05], "diff_mean": -0.002893571653529047, "diff_se": 0.0018163063987612026, "inference_valid": true, "n_blocks": 14, "n_eras": 585, "p_pos": 0.0534, "p_value": 0.9424057594240576}, "60": {"ci90": [-0.006049948851666808, 0.00028498649663929673], "diff_mean": -0.002893571653529047, "diff_se": 0.0019109227441367668, "inference_valid": true, "n_blocks": 9, "n_eras": 585, "p_pos": 0.0674, "p_value": 0.9344065593440656}}, "core_neutralization": 0.0, "fit_sharpe": 0.2787406406547687, "metrics": {"fit": 0.01287388355012923, "inov": 0.002100931877351055, "score": 0.014637115077911914, "unq": -0.00027837669236307886}, "name": "adaptive_score", "paired": {"fit": {"ci90": [-0.0035412706086638723, -0.0006925950426069019], "diff_mean": -0.002108863838585279, "diff_se": 0.0008653300484978852, "inference_valid": true, "n_blocks": 29, "n_eras": 585, "p_pos": 0.007, "p_value": 0.9911008899110089}, "score": {"ci90": [-0.005592175149337365, -0.00020735230425019186], "diff_mean": -0.002893571653529047, "diff_se": 0.0016217742791713524, "inference_valid": true, "n_blocks": 29, "n_eras": 585, "p_pos": 0.0366, "p_value": 0.9615038496150385}}, "proxy_score_sharpe": 0.22295608985636536, "scenario_score_gains": {"refreshed/core_0": -0.002893571653529047, "refreshed/core_1": -0.003294014632055205, "refreshed/core_2": -0.0028406138553957075, "static/core_0": -0.0030645328653041717, "static/core_1": -0.003455724841822439, "static/core_2": -0.0030110343911277093}, "windows": {"first_half": {"fit": -0.0039582194516714995, "score": -0.005943886478572316}, "second_half": {"fit": -0.00025317481586520173, "score": 0.00016718945516163008}}}}

All native models and original gates were preserved. Only missing-core calibration semantics changed.
No confirmation performance verdict existed before the repair. Baseline01 remains unchanged.
No platform uploads, submissions, staking or deployment changes.

The repaired confirmation **failed**. Over 585 eras, candidate FIT was 0.01287388 vs baseline 0.01498275
(difference -0.00210886, p_pos 0.0070), and bounded composite proxy score was 0.01463712 vs 0.01753069
(difference -0.00289357, p_pos 0.0366). All six aggregate score sensitivities were negative. First-half FIT
difference was -0.00395822; second-half difference -0.00025317. No candidate is qualified or promoted, and
the 15% operational major-improvement condition was not met. Discovery's positive result did not transfer.

Post-decision controls (descriptive only; no runner-up substitution):

| component/policy | confirmation FIT | confirmation proxy score |
|---|---|---|
| refreshed baseline | 0.01498275 | 0.01753069 |
| fixed anchored prior | 0.01388173 | 0.01663305 |
| FIT-adaptive policy | 0.01342129 | 0.01547723 |
| selected score-adaptive policy | 0.01287388 | 0.01463712 |
| prior stable group | 0.01041563 | 0.01344507 |
| GPU RMSE group | 0.00491660 | 0.00559058 |
| GPU QueryRMSE group | 0.00415384 | 0.00330147 |
| GPU YetiRank group | 0.00913890 | 0.01344349 |

The alternative families all underperformed the baseline in this period. The fixed blend also lost, so the
failure is not solely caused by learning weights. Adaptation added another FIT loss of 0.00100785 versus the
fixed control. The policy initially carried 21.8% GPU weight from the previously strong late-discovery period;
its baseline allocation rose from 49.5% at 5340 to 82.4% at 5840. It reacted to matured outcomes, but the
delayed reaction and residual weak-family exposures did not protect the first half. This supports temporal
transfer and weight-lag concerns; it does not establish the economic cause of the model-family reversal.

The missing-reference repair did **not** cause the first-half loss. Original and repaired algorithms were
run on the full prefix through 5639, before the missing core observation could enter a calibration fit;
their actual predictions were identical (max error 0). The large loss already existed in that unchanged
prefix. No original confirmation performance result was available before defining the technical repair.

Validation: 485 tests passed with no skips. Complete-reference behavior, partial-core covariance scaling,
future-label invariance, native serialization and copied-source integrity are covered. The discovery replay
error was exactly zero, and every native model byte was preserved. Repair source: 4fe11ce. Original input
failure and operator launch/report issues remain in online14's report and metrics.

These are previously viewed historical confirmation eras, now explicitly consumed by this candidate. Never
call a later retuned result on them untouched confirmation or a prospective result. Preserve the failed
candidate and useful GPU-period evidence for causal follow-up; do not silently tune it on this window,
relax its gates, select the fixed-control runner-up, or claim a reliable major improvement.
