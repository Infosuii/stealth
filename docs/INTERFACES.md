# INTERFACES — module contracts for parallel implementation

Authoritative for signatures, schemas and file layouts. If you must deviate, write a request to
`/home/climber/.claude/jobs/8bc89d82/tmp/requests/<group>.md`; do not silently change a signature.
Plan: `docs/PLAN.md`. Python 3.11 venv: `/home/climber/work/stealth/.venv/bin/python`. No `print` in library
code (use `logging.getLogger(__name__)`). Seeds explicit everywhere. **No torch anywhere in this package.**

## 0. SDK facts (everesteer-api 0.4.1, verified in the venv)

| Item | Value |
|---|---|
| pip name / import name | `everesteer-api` / `import everestapi` (deps: httpx, click) |
| client | `from everestapi import EverestAPI`; `EverestAPI(api_key=None, base_url=None, timeout=30.0, tournament="equities", basic_auth=None, cf_access_client_id=None, cf_access_client_secret=None)` — **always pass `tournament="futures"`** |
| auth | header `X-API-Key`; env `EIQ_API_KEY` (fallback `EVEREST_API_KEY`); optional `EIQ_BASIC_AUTH_USER/PASS`, `CF_ACCESS_CLIENT_ID/SECRET` |
| base URL | `EverestAPI.DEFAULT_BASE_URL` = `http://api.eiq-prod.internal:8054` (sandbox-internal!). The SDK does **not** read `EIQ_BASE_URL`; our code must: `base_url=os.environ.get("EIQ_BASE_URL", "https://api.everesteer.ai")` |
| scoring | `from everestapi import scoring`: `fit20(predictions, target) -> float`; `unq20(predictions, ai_model, target) -> float`; `inov(predictions, features, target) -> float`; `core_feature_average(features) -> pd.Series`; `feature_exposure(predictions, features) -> float`; `payout(fit20, unq, *, score_multiple_constant, inov=0.0, stake=1.0) -> float`. Inputs: Series/arrays aligned on index; one call = one era. |
| zero rule | **`unq20` in 0.4.1 ALREADY returns 0 when the benchmark's own covariance with the target is < 0** (contrary to the plan text). `metrics.per_era_unq` must NOT re-apply it. `inov` = `unq20` with `core_feature_average(features)` (−1/NaN dropped per row) as the reference. |
| other | `fit20` returns 0.0 for <3 rows or zero-variance preds; `unq20` needs ≥5 common rows. Lane pickles: `python_version="3.11"`. |

`EIQ_API_KEY` presence in this sandbox: **yes** (never print it). Anything touching the API must fail fast with
`RuntimeError("EIQ_API_KEY not set; copy .env.example to .env")` when absent; everything else runs offline from `data/`.

## 1. Data schemas and conventions (`stealth/data.py`)

Parquets in `data/` (index name `id`, opaque 16-hex strings, never renumber):

| file | rows | columns |
|---|---|---|
| `futures_{train,validation,live}.parquet` | 468,236 / 125,931 / 90 | `exped` (str), `data_type` (str), 178 `feature_*` (**int8**, −1 = missing, 0..9 deciles), 15 `target_*` (**float32**, NaN on validation/live; everest never NaN on train) |
| `futures_{train,validation,live}_benchmark_models.parquet` | same ids | `exped` (str), `v1_sherpa` (float64) |

Era convention: `era_int("exped_0003") == 3` (`int(s.rsplit("_",1)[1])`); `era` column is **int32**. Train eras 3..6524,
validation 6566..7957, live 8055. Constants exported by `data.py`:
`PRIMARY_TARGET = "target_everest"`, `AUX_TARGETS = ["target_Tougroute","target_Ghat","target_Meltsene","target_Timesguida","target_Ighil","target_Tarhat"]`,
`DATA_DIR = Path(__file__).resolve().parents[1] / "data"`, `N_FEATURES = 178`.

```python
def load_split(split: Literal["train","validation","live"], data_dir: Path = DATA_DIR,
               columns: list[str] | None = None) -> pd.DataFrame      # raw parquet + int32 'era' column; index 'id'
def load_benchmark(split, data_dir=DATA_DIR) -> pd.Series             # name 'v1_sherpa', float64, index 'id'
def era_int(exped: str) -> int
def add_era(df: pd.DataFrame) -> pd.DataFrame                         # adds/overwrites int32 'era'; returns same object
def feature_columns(df: pd.DataFrame) -> list[str]                    # 'feature_*' in FILE order (stable; the bundle stores this list)
def target_columns(df: pd.DataFrame) -> list[str]
def filter_eras(df, *, min_era: int | None = None, max_era: int | None = None,
                min_rows: int | None = None) -> pd.DataFrame          # era in [min,max]; drop eras with n_rows < min_rows
def features_nan(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame   # float32, −1 -> NaN  (tree input, "nan")
def features_linear(df: pd.DataFrame, cols: list[str], missing_cols: list[str]) -> pd.DataFrame
    # float32: cols as value/9 with −1 -> 0.5 ; + one 0/1 indicator per name in missing_cols (named 'miss_<feature>') ;
    # + 'n_missing' = (#−1 across cols)/len(cols). missing_cols = the 15 highest −1-rate features on TRAINING eras (caller computes).
def time_decay_weights(era: pd.Series, enabled: bool) -> np.ndarray   # linear 0.5..1.0 oldest->newest; ones if disabled
def download(data_dir: Path = DATA_DIR, splits=("train","validation","live"), benchmarks=True) -> list[Path]  # needs EIQ_API_KEY
def main(argv: list[str] | None = None) -> int                        # `python -m stealth.data download [--data-dir]`
```

## 2. CV (`stealth/cv.py`)

```python
@dataclass(frozen=True)
class Fold:
    name: str                      # "fold0".."fold4" | "forward"
    train_eras: np.ndarray         # int32 sorted, both sides of the test block, purged+embargoed
    test_eras: np.ndarray          # int32 sorted, contiguous
    inner_train_eras: np.ndarray   # subset of train_eras for the inner early-stopping fit
    inner_val_eras: np.ndarray     # last inner_frac of train eras (by time), inner_gap eras after inner_train

def purged_folds(eras: np.ndarray, n_folds: int, *, purge: int = 20, embargo: int = 20,
                 inner_frac: float = 0.15, inner_gap: int = 20) -> list[Fold]
    # eras = unique sorted int32. Test blocks = n_folds contiguous equal slices of eras. Train = all eras with
    # |era - test| > purge+embargo ... precisely: era < test_min - purge - embargo  or era > test_max + purge + embargo.
def forward_block(eras: np.ndarray, *, train_end: int = 5925, test_start: int = 5945, **inner_kwargs) -> Fold
    # train eras < train_end, test eras >= test_start (gap >= 20 asserted)
def assert_fold_gaps(folds: list[Fold], min_gap: int = 20) -> None     # raises AssertionError; tests call it on every config
@dataclass(frozen=True)
class PairedResult: diff_mean: float; diff_se: float; p_pos: float; ci90: tuple[float, float]; n_eras: int
def paired_block_bootstrap(diff: pd.Series, *, block: int = 20, n_boot: int = 2000, seed: int = 0) -> PairedResult
    # diff = per-era (A − B) indexed by era int, sorted. Circular moving-block bootstrap of the era series.
def block_bootstrap_ci(series: pd.Series, *, block=20, n_boot=2000, seed=0, level=0.90) -> tuple[float, float]
```

## 3. Metrics (`stealth/metrics.py`) — thin wrappers over `everestapi.scoring`, grouped per era

All `pred/target/bench` are `pd.Series` on the same `id` index; `era` is an int32 Series on that index.
Returned per-era Series are indexed by era (int32, sorted), name = metric.

```python
def per_era_fit(pred, target, era) -> pd.Series                       # scoring.fit20 per era
def per_era_unq(pred, bench, target, era) -> pd.Series                # scoring.unq20 per era (zero rule inside SDK)
def per_era_inov(pred, core_frame: pd.DataFrame, target, era) -> pd.Series   # scoring.inov with core_frame = raw int8 cols (−1 ok)
def per_era_spearman(pred, target, era) -> pd.Series
def summarise(s: pd.Series, *, block=20, n_boot=2000, seed=0) -> dict
    # {"mean","std","sharpe","pct_pos","frac_roll20_neg","ci90":[lo,hi],"n_eras"}  (sharpe = mean/std, per era, NOT annualised)
def leak_check(spearman: pd.Series, threshold: float = 0.18) -> None  # raises RuntimeError("LEAK: ...") if mean > threshold
def regime_simulation(z_pred: pd.Series, target, era, direction: pd.Series, core_proxy: pd.Series, *,
                      p_b_grid=(0.0,.25,.5,.75), p_c: float, bench_fit_means=(0.0,0.009,0.02), bench_fit_sd=0.035,
                      n_draws=500, seed=0) -> dict
    # z_pred = per-era gaussianised blend; direction = surrogate benchmark direction (Series on ids);
    # per draw & era: b = gaussianise(direction)*s + eps scaled so fit20(b,target) ~ N(mean, sd); apply _runtime.neutralise
    # at each p_b (and p_c vs core_proxy); record fit/unq/inov/score=fit+unq+inov.
    # Returns {"<mean>": {"p_b=<x>": {"bpos": {...}, "bneg": {...}, "mix": {...}}}} each with mean fit/unq/inov/score + se.
```

## 4. Features / surrogate (`stealth/features.py`, `stealth/surrogate.py`)

```python
def sharpe_subset(X_nan: pd.DataFrame, y: pd.Series, era: pd.Series, threshold: float = 0.1) -> list[str]
    # per-era Spearman(feature, y) on the given rows ONLY (caller passes training-era rows); keep |mean/std| > threshold
def core_proxy_columns(X_nan, bench: pd.Series, era, *, k: int = 10, max_era: int = 2800) -> list[str]
    # top-k features by |mean per-era Spearman(feature, v1_sherpa)| on eras < max_era (never scored eras)
def core_proxy_series(features: pd.DataFrame, cols: list[str]) -> pd.Series
    # row mean over cols ignoring −1/NaN (same rule as scoring.core_feature_average); all-missing row -> NaN
def random_column_bags(cols: list[str], *, n_bags=4, fraction=0.5, seed=0) -> list[list[str]]
def resolve_feature_set(spec: str | list[str], all_cols, *, sharpe_cols=None, bags=None) -> list[str]
    # "all" | "sharpe_subset" | "bag:<i>" | explicit list

@dataclass(frozen=True)
class Surrogate: columns: list[str]; coef: np.ndarray; intercept: float
    def predict(self, X_lin: pd.DataFrame) -> pd.Series            # direction only; never used for selection
def fit_surrogate(X_lin, bench, era, *, max_era: int = 2800, alpha: float = 1e4) -> Surrogate
```

## 5. Members (`stealth/models/*`)

`MemberSpec` is the YAML member dict (keys below) plus runtime `feature_cols: list[str]` resolved per fold.
Every family module implements the SAME functions; `models/__init__.py` dispatches on `family`:

```python
Family = Literal["lgbm", "xgb", "cat", "ridge"]
# family modules (lgbm.py / xgb.py / cat.py / linear.py):
def fit_with_early_stopping(params: dict, X_tr, y_tr, w_tr, X_val, y_val, *, cap: int, seed: int, threads: int) -> tuple[Any, int]
    # returns (model, best_iter). linear.py: raises NotImplementedError (ridge has no iterations)
def fit_fixed(params: dict, X, y, w, *, n_iter: int | None, seed: int, threads: int) -> Any
def to_bytes(model) -> bytes       # lgbm: model_to_string().encode(); xgb: save_raw("ubj"); cat: save_model(tmp, "cbm") bytes;
                                   # ridge: np.savez (keys: coef float64[n], intercept float64, columns str[n]) -> bytes
def from_bytes(blob: bytes) -> Any
def predict(model, X: pd.DataFrame) -> np.ndarray   # float64, len(X); X columns == spec.feature_cols in order; NaN input allowed for trees
# models/__init__.py
@dataclass
class FittedMember: name: str; family: Family; target: str; group: str; input: str; feature_cols: list[str]; blob: bytes; n_iter: int | None; fit_seconds: float
def fit_member(spec: dict, X_tr, y_tr, w_tr, *, X_val=None, y_val=None, seed: int, threads: int) -> FittedMember
    # spec["n_estimators"] set -> fit_fixed; else early stopping with cap spec["es_cap"] (X_val/y_val required)
def predict_member(fm: FittedMember | tuple[Family, bytes], X: pd.DataFrame) -> np.ndarray
```
X for `input: nan` = `data.features_nan(df, feature_cols)`; for `input: linear` = `data.features_linear(...)` (ridge uses all
linear columns; its `feature_cols` records the 178 base names and `missing_cols` is stored in `spec.json`).
LightGBM/XGB/Cat objective: regression (L2) on the raw 5-level target; `threads` -> `num_threads`/`n_jobs`/`thread_count`.

## 6. Search (`stealth/search.py`)

```python
def run_search(name: str, grid: dict[str, list], objective: Callable[[dict], float], *, n_trials: int,
               storage: Path, seed: int) -> list[tuple[dict, float]]   # (params, value) sorted desc; Optuna TPE, study=name, sqlite:///storage; resumable
def top_k_params(results, k: int) -> list[dict]
```
`objective(params)` = mean per-era FIT over OOF folds for that config (train.py supplies it). `n_trials == grid size` means exhaustive (GridSampler).

## 7. Config YAML (`configs/{smoke,test,full}.yaml`)

```yaml
name: str
run:   {seed: int, threads: int, min_era: int|null, max_era: int|null, min_rows: int|null, time_decay: bool, data_dir: str}
cv:    {n_folds: int, purge: int, embargo: int, inner_frac: float, inner_gap: int,
        forward: {enabled: bool, train_end: int, test_start: int}}
knobs: {era_start: [int], min_rows: [int|null], time_decay: [bool]}          # optional (full only)
members:                                                                       # list; names unique
  - {name: str, family: lgbm|xgb|cat|ridge, target: target_*, group: everest|aux,
     features: all|sharpe_subset|"bag:<i>"|[names], input: nan|linear,
     params: {...} | {best_of: <member name>},     # best_of -> top-1 params from search.<member> at run time
     n_estimators: int|null,                       # int = fixed count; null = inner early stopping
     es_cap: int|null}
search: {<member name>: {grid: {param: [values]}, n_trials: int, top_k: int}}  # may be {}
feature_bags: {n_bags: int, fraction: float, seed: int}                        # optional
sharpe_subset: {threshold: float}                                              # optional
refit:  {seeds: [int]}                                                         # optional; default [0]
blend:  {everest_weight: float, nnls: bool, p_b: float (<=0.75), p_c: float (<=0.5),
         simulation: {n_draws: int, p_b_grid: [float], bench_fit_means: [float], bench_fit_sd: float, seed: int}}
bootstrap: {block: int, n_boot: int, seed: int}
gates:  {leak_spearman: float, suspect_gain_over_baseline: float, p_pos_adopt: float, member_corr_max: float?}
timing: {catboost_max_fit_seconds: int, progress_every_seconds: int}
```
`train.load_config(path: Path) -> dict` validates these keys (raise `ValueError` listing what is wrong).

## 8. Artifacts layout (`artifacts/<run_id>/`, gitignored)

```
config.yaml            # verbatim copy of the config used
run.log                # written by train.py itself (FileHandler + progress line every timing.progress_every_seconds)
folds.json             # [{name, train_eras:[min,max,n], test_eras:[min,max], inner_val_eras:[min,max]}]
oof.parquet            # index 'id'; columns: era int32, fold int8 (-1 = forward), target_everest float32,
                       #   plus one float32 column per member: "m__<member name>" = RAW member score (not gaussianised); NaN where absent
forward.parquet        # same schema, forward block only (when cv.forward.enabled)
members/<name>/spec.json      # member spec + feature_cols + missing_cols + per-fold n_iter + frozen_n_iter (median) + fit_seconds
members/<name>/fold<k>.bin    # bytes from models.to_bytes for fold k (k = 0..n_folds-1), forward.bin for the forward fold
members/<name>/full_s<seed>.bin   # full-train refit(s) with frozen_n_iter (written by train.py --refit / full tier)
search/<member>.json          # [{params, value}] sorted desc  (hyperparameter VALUES stay here, never in handoffs/results)
optuna.db                     # sqlite storage, one study per searched member
metrics.json                  # see below
blend.json                    # written by blend.py: {weights:{member: w}, rule: "ew"|"nnls", p_b, p_c, core_proxy_cols:[...],
                              #   regime: <regime_simulation output>, paired_vs_ew: {diff_mean, diff_se, p_pos}}
```

`metrics.json` schema:
```json
{"run_id": str, "config_name": str, "git_sha": str, "status": "ok|failed|partial", "started_at": iso, "finished_at": iso,
 "host": {"os":str,"cpu":str,"ram_gb":float,"python":str,"libs":{name:version}},
 "timings": {"total_s": float, "per_member_fit_s": {member: float}},
 "members": {member: {"fit": summarise(...), "spearman_mean": float, "forward_fit": float|null,
                      "corr_with_bench_val": float|null, "n_iter": int|null, "suspect": bool}},
 "blend": {"ew": summarise(...), "forward_fit": float|null},
 "paired": {"<A>_vs_<B>": {"diff_mean","diff_se","p_pos","ci90"}},
 "leak_check": "pass|FAIL", "notes": [str]}
```

## 9. Runtime + bundle (`stealth/_runtime.py`, `stealth/bundle.py`)

`_runtime.py` is the ONLY module the pickle references. Module-level imports: stdlib + `numpy` + `pandas` only;
`scipy`, `lightgbm`, `xgboost`, `catboost` are imported INSIDE function bodies. No `stealth.*` imports.

```python
def gaussianise(x: np.ndarray) -> np.ndarray      # z = norm.ppf((rank_avg - 0.5)/n) over non-NaN; NaN -> 0.0 (mid)
def rank01(x: np.ndarray) -> np.ndarray           # (rank_avg - 0.5)/n in (0,1); NaN -> 0.5
def neutralise(z: np.ndarray, m: np.ndarray, p: float) -> np.ndarray   # z - p*m*(m·z)/(m·m); m already gaussianised & centred; m·m==0 -> z
def load_member(family: str, blob: bytes) -> Any   # lgbm.Booster(model_str=...) / xgb.Booster().load_model(bytearray) / CatBoostRegressor().load_model(blob=...) / dict from npz
def predict_member(family: str, model: Any, X: np.ndarray) -> np.ndarray   # X float32 2-D in bundle column order
def coerce_benchmark(bench, index: pd.Index, min_coverage: float = 0.5) -> np.ndarray | None
    # Series | 1-col DataFrame | None -> float array aligned to index (NaN where absent); None if coverage < min_coverage or var == 0
def make_predict(bundle: dict) -> Callable        # returns predict(live_features, live_benchmark_models=None)
```

Bundle dict (captured by the closure; blobs deserialised lazily on first call and cached in a dict):
```python
{"version": str, "git_sha": str, "built_at": iso, "run_id": str,
 "feature_cols": [178 names, file order],
 "missing_cols": [15 names],                       # for the linear input
 "members": [{"name": str, "family": str, "target": str, "group": "everest"|"aux", "input": "nan"|"linear",
              "feature_cols": [names], "blobs": [bytes, ...]}],   # several blobs = seeds, averaged
 "weights": {member_name: float},                  # sum to 1
 "p_b": float, "p_c": float, "core_proxy_cols": [10 names], "min_bench_coverage": 0.5}
```

`predict(live_features: pd.DataFrame, live_benchmark_models: pd.Series | pd.DataFrame | None = None) -> pd.DataFrame` contract:
1. Select `feature_cols` BY NAME (`KeyError` listing missing names if any; extra columns ignored; order irrelevant).
2. Build the nan frame (−1→NaN float32) and the linear frame; each member -> raw score -> `gaussianise` (per call = one era).
3. Weighted sum with `weights`; member rows that are NaN are dropped from that row's weighted mean; all-NaN row -> 0.
4. If benchmark coerces (`coerce_benchmark`): `z = neutralise(z, gaussianise(bench), p_b)` on covered rows. Else skip.
5. Core proxy: `c = gaussianise(core_proxy_series)` (all-missing rows -> 0) ; `z = neutralise(z, c, p_c)`.
6. Return `pd.DataFrame({"prediction": rank01(z)}, index=live_features.index)` — exact `pd.DataFrame`, 1 column,
   float64, all values strictly in (0,1), no NaN, deterministic. Never raises on a weird benchmark; raises only on missing features.

`bundle.py`:
```python
def build_bundle(run_dir: Path, *, p_b: float | None = None, p_c: float | None = None, version: str = VERSION) -> dict
    # reads blend.json + members/*/full_s*.bin (falls back to fold*.bin averaging if no full refit); caps p_b<=0.75, p_c<=0.5
def write_pickle(bundle: dict, out: Path) -> str     # cloudpickle.register_pickle_by_value(stealth._runtime); dump make_predict(bundle); returns sha256
def write_manifest(out_pkl: Path, bundle: dict, sha256: str) -> Path   # models/<stem>.manifest.json + models/<stem>.pkl.sha256
def main(argv=None) -> int     # python -m stealth.bundle --run-id <id> --out models/stealth_v1.pkl [--p-b x --p-c y]
```
Manifest JSON: `{"pkl": str, "sha256": str, "size_bytes": int, "version": str, "git_sha": str, "run_id": str, "built_at": iso,
"python": "3.11.x", "libs": {lightgbm, xgboost, catboost, scikit-learn, numpy, pandas, scipy, cloudpickle}, "members": [{name, family, target, group, n_blobs}],
"weights": {...}, "p_b": float, "p_c": float, "core_proxy_cols": [...], "n_features": 178, "pkl_check": {"passed": bool, "at": iso} | null,
"sandbox_sha256": str | null, "upload_id": str | null}`. The pickle itself is never committed; only `*.sha256` and `*.manifest.json` are.

## 10. Ops modules (CLI contracts)

| command | behaviour / exit code |
|---|---|
| `python -m stealth.train --config C --run-id R [--resume] [--refit]` | writes `artifacts/R/*` (section 8); 0 ok, 2 leak/gate failure, 1 error (traceback in run.log + metrics.json status=failed) |
| `python -m stealth.blend --run-id R` | reads oof/forward parquet -> `blend.json`; logs regime table |
| `python -m stealth.bundle --run-id R --out models/X.pkl` | section 9 |
| `python -m stealth.pkl_check models/X.pkl [--venv artifacts/pklcheck_venv] [--eras 50]` | creates venv from requirements-lock.txt WITHOUT stealth; runs checker via `python -I` with cwd=tmp; checks per plan (no `stealth`/`torch` in sys.modules after predict; byte scan `b"torch"`, `b"stealth."` except `stealth._runtime`); live + 50 random validation eras; misaligned/constant/None benchmark; exact DataFrame, (0,1), no NaN, deterministic, <5 s/era, <50 MB; updates manifest.pkl_check; exit 0/1 |
| `python -m stealth.predict --pkl models/X.pkl [--submit] [--upload] [--data-dir]` | dry run by default: round open? -> download live+bench -> predict -> `validate_submission`; `--submit`/`--upload` are USER-GATED (refuse unless `--i-have-approval` flag AND env `STEALTH_APPROVED=1`); logs `replaced/late/scored`; exit 0 ok, 3 window closed, 1 error |
| `python -m stealth.diag --pkl models/X.pkl [--out reports/diag_<ts>.parquet] [--upload]` | predictions per validation era (bench from validation benchmark file) -> parquet (`id` index, `prediction`); upload USER-GATED as above |
| `python -m stealth.monitor [--scores]` | (i) accepted submission for today's exped, (ii) active upload sha == manifest sha / `models/active.json` upload_id; mismatch -> `handoffs/ALERT.md`, exit 1 |
| `python -m stealth.report --run-id R --handoff NN [--no-push]` | `handoffs/results/NN-results.md`, `NN-metrics.json`, `NN-log-tail.txt`; git add/commit/push; on push failure prints `PUSH FAILED — ...` and exits 1 (files already written). Metrics + config NAMES only; no hyperparameter values. |
| `python -m stealth.selfcheck` | table: python, pinned libs vs requirements-lock (mismatch -> exit 1), OS/CPU/RAM/GPU/TZ, `git config core.hooksPath` (warn if unset), EIQ_API_KEY present yes/no |

`models/active.json`: `{"model_id": str, "upload_id": str, "sha256": str, "pkl": str, "uploaded_at": iso}` (committed).
