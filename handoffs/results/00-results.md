# Results 00 — env-setup

- run_id: `env-setup`  config: `-`  status: **failed**
- git_sha: `b96dd66f72e106d8ea03a01896d6e6f624a7ee41`  started: -  finished: -
- generated: 2026-11-13T20:15:20+00:00 by `python -m stealth.report`

> **STATUS: FAILED** — see the traceback / log tail below.

## Environment

| item | value |
|---|---|
| os | Linux-7.3.3-1-cachyos-x86_64-with-glibc2.44 |
| cpu | Intel(R) Core(TM) Ultra 7 255HX |
| ram_gb | 30.8 |
| gpu | - |
| driver | - |
| python | 3.11.18 |
| timezone | Europe/London (UTC+00:00) |
| git_sha | b96dd66f72e106d8ea03a01896d6e6f624a7ee41 |
| core.hooksPath (reporting host) | scripts/hooks |
| pins (reporting host) | OK |
| libs | catboost 1.2.10, cloudpickle 3.1.2, everesteer-api 0.4.1, lightgbm 4.7.0, numpy 2.4.6, optuna 5.0.0, pandas 3.0.6, pyarrow 24.0.0, pytest 9.2.0, pyyaml 6.0.3, scikit-learn 1.9.1, scipy 1.17.1, xgboost 3.2.0 |

## Metrics (per-era fit20 on OOF; fit_se = 20-era block-bootstrap SE; config names only, no hyperparameter values)

_none_

## Paired block-bootstrap tests

_none_

## Regime simulation (blend.json)

_no blend.json / regime simulation present_

## Wall times

_none_

## Notes

- metrics.json missing in artifacts/env-setup

## run.log (last 0 lines)

```text
(run.log missing)
```

## Operator notes

```text
EIQ_API_KEY     present (environment or .env)                                  

== pins (requirements-lock.txt) ==
package         pinned  installed  status
--------------  ------  ---------  ------
lightgbm        4.7.0   4.7.0      OK    
xgboost         3.2.0   3.2.0      OK    
scikit-learn    1.9.1   1.9.1      OK    
pandas          3.0.6   3.0.6      OK    
numpy           2.4.6   2.4.6      OK    
scipy           1.17.1  1.17.1     OK    
catboost        1.2.10  1.2.10     OK    
pyarrow         24.0.0  24.0.0     OK    
cloudpickle     3.1.2   3.1.2      OK    
everesteer-api  0.4.1   0.4.1      OK    
optuna          5.0.0   5.0.0      OK    
pytest          9.2.0   9.2.0      OK    
pyyaml          6.0.3   6.0.3      OK    

PINS OK
........................................................................ [ 88%]
...................                                                      [100%]
163 passed in 34.40s
Handoff 00: PASSED. Git identity and scripts/hooks confirmed; all six parquet files downloaded.
Row counts verified: train 468236, validation 125931, live 90.
User clarified that 50,886,382 bytes is the correct training file size; the 80-90 MB handoff figure was a documentation error.
The reporter status failed is expected for env-setup because no training run exists, as documented in handoff 00 section 7.
.env verified only with test -s .env; no submissions, uploads or staking.
```
