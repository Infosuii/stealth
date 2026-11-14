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
    # {"mean","std","sharpe","pct_pos","frac_roll20_neg","ci90":[lo,hi],"se_block","n_eras"}  (sharpe = mean/std, per era, NOT
    # annualised; se_block = std of the 20-era block-bootstrap means = the SE every report prints as fit_se; never std/sqrt(n))
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
members/<name>/spec.json      # member spec + feature_cols + missing_cols + inner_iters (per CV fold) + inner_iter_forward
                              #   + frozen_n_iter_per_fold / frozen_n_iter_forward (count used per scored fold, from INDEPENDENT
                              #   inner splits only; count_sources names them) + frozen_n_iter (median over CV folds; full refit
                              #   only) + fit_seconds + search_gain (searched members: top-k minus base-params cell, OOF + forward)
members/<name>/fold<k>.bin    # bytes from models.to_bytes for fold k (k = 0..n_folds-1), forward.bin for the forward fold
members/<name>/full_s<seed>.bin   # full-train refit(s) with frozen_n_iter (written by train.py --refit / full tier)
search/<member>.json          # [{params, value}] sorted desc  (hyperparameter VALUES stay here, never in handoffs/results)
search/<member>.gain.json     # {topk_fit, base_params_fit, gain, suspect, topk_forward_fit, base_params_forward_fit, forward_gain,
                              #   forward_confirms, topk_cells (hashes), base_cell}  winner's-curse record of the search
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

---

# M2–M6 additions (architect pass, 2026-11-13)

Everything below is additive: an M1 config (`smoke.yaml`, `test.yaml`) must behave exactly as before. Owners in
brackets (`[train]`, `[blend]`, `[nn]`, `[docs]`) are the work groups; a group never edits another group's files —
contract changes go through a request file as in the header. "Gate" always means: paired 20-era block bootstrap,
adopt iff `p_pos >= gates.p_pos_adopt` on CV **and** forward diff `>= 0`; otherwise tie -> keep the simpler thing.

## 11. Config additions (validated by `train.load_config`) [train]

```yaml
run:
  min_era: 2800        # SCORING universe: CV test blocks and the forward test block use eras >= min_era (as before)
  min_rows: null       # legacy scoring-universe filter; load_config REJECTS a non-null value when `knobs` is set
knobs:                 # optional (full/minifull). Training-universe alternatives, decided ONCE -> knobs.json
  member: lgbm_everest            # member whose BASE params run the comparison (default: train.baseline_member)
  era_start: [2800, 2000, 3]      # training rows start here; the entry == run.min_era is the default
  rows_rule: [none, filter80, weight80]   # none | filter<N>: drop TRAINING eras with n_rows < N |
                                          # weight<N>: those eras get weight `rows_low_weight`
  rows_low_weight: 0.5
  time_decay: [false, true]       # the entry == run.time_decay is the default
  alpha_total: 0.2                # Holm family-wise level on (1 - p_pos); default 1 - gates.p_pos_adopt
  # legacy key knobs.min_rows is rejected with "use knobs.rows_rule"
members:
  - {..., arm: str}               # optional; default = "aux" for group aux, else the member name. Used for hierarchical
                                  #   equal weights and the admission ladder. full.yaml sets: lgbm, xgb, cat, ridge,
                                  #   sharpe, bags, aux, mlp
  - name: mlp_everest             # family mlp == EXTERNAL member: train.py never fits it (section 14)
    family: mlp
    target: target_everest
    group: everest
    arm: mlp
    features: all                 # the 178 base columns of the linear frame
    input: linear
    params: {hidden: [256, 64], dropout: 0.3, aux_targets: [<6 aux names>], aux_weight: 0.5, lr: 1.0e-3,
             weight_decay: 1.0e-4, batch_size: 4096, patience: 5, device: auto}
    seeds: [0, 1, 2, 3, 4]
    n_estimators: null            # epochs come from inner early stopping
    es_cap: 60                    # max epochs
search:
  <member>: {grid: {...}, n_trials: int (<= gates.max_search_trials), top_k: int, adopt_gate: true}
blend:
  p_b: 0.5             # FALLBACK only (blend.neutralisation.select false); never what a frozen bundle uses
  p_c: 0.25
  admission:           # optional; absent -> every trained member is admitted (M1 behaviour)
    base: [lgbm]                  # arms always in (the M2 incumbent)
    order: [aux, xgb, cat, ridge, sharpe, bags, mlp]   # pre-registered; arms absent from the run are skipped
    sharpe_gain: 0.05
  simulation: {n_draws, p_b_grid, bench_fit_means, bench_fit_sd, seed, p_c_grid: [0.0, 0.25, 0.5]}
  neutralisation:      # optional; absent -> select: false
    select: true
    central_mean: 0.009           # must be one of simulation.bench_fit_means
    pessimistic_mean: 0.02        # must be one of simulation.bench_fit_means
    fit_cost_max: 0.003
    sharpe_tol: 0.02
    worst_tol: 0.0005
gates: {..., member_corr_max: 0.95, max_search_trials: 25}
timing: {catboost_max_fit_seconds: 120, progress_every_seconds: 60}   # cap now ENFORCED (section 15)
```
Validation adds: every `p_b_grid`/`p_c_grid` value within the caps 0.75 / 0.5; `central_mean`/`pessimistic_mean` in
`bench_fit_means`; `admission.base`/`order` name arms (unknown arm -> warning, skipped); `mlp` members need `seeds`,
`input: linear`; `n_trials <= gates.max_search_trials` (default 25).

## 12. Training universe + knobs (`stealth/knobs.py` NEW, `cv.py`, `train.py`) [train]

`stealth/knobs.py` module-level imports: stdlib, numpy, pandas, `stealth.data`, `stealth.cv` ONLY (the GPU venv
imports it); `stealth.metrics` is imported inside `select_universe`.
```python
@dataclass(frozen=True)
class Universe:
    era_start: int; rows_rule: str = "none"; time_decay: bool = False; rows_low_weight: float = 0.5
    def key(self) -> str                       # stable, e.g. "e2800-none-d0" (cache keys, spec.json["universe"])
    def to_dict(self) -> dict; @classmethod def from_dict(cls, d) -> "Universe"
    def train_mask(self, era: np.ndarray, era_n_rows: np.ndarray) -> np.ndarray      # bool per row, TRAINING only
    def weights(self, era: np.ndarray, era_n_rows: np.ndarray) -> np.ndarray | None
        # rows rule weight x data.time_decay_weights(era of THIS fit, enabled) ; None when both are off
def default_universe(cfg: dict) -> Universe            # era_start=run.min_era, rows none, time_decay=run.time_decay
def candidates(cfg: dict) -> list[Universe]            # [default] + ONE-knob-at-a-time alternatives, config order (no factorial)
def select_universe(rd, cfg: dict, run_dir: Path, *, prog=None) -> tuple[Universe, dict]   # writes / reuses knobs.json
```
`cv.extend_train(fold: Fold, extra_eras: np.ndarray, *, gap: int) -> Fold` adds the eras of `extra_eras` that lie
`> gap` before the test block to `train_eras` AND `inner_train_eras`; `inner_val_eras` and `test_eras` are unchanged
(so the default universe reproduces the M1 folds exactly); re-asserts `assert_fold_gaps`.

`RunData` loads eras `>= min(run.min_era, min(knobs.era_start))` (the pool); folds/forward are built on scoring eras
`>= run.min_era` exactly as today and then extended with the pool eras `< run.min_era`. A universe never touches
test/prediction rows: `_fit_rows` = fold eras & target notna & `universe.train_mask`; sample weights =
`universe.weights`. Full refits use pool eras under the universe. `sharpe_subset` uses universe-masked training rows;
its cache key, search-cell hashes (`_params_hash({**params, "__universe__": key})`) and member resume all include
`universe.key()` (a member whose spec.json universe differs is retrained).

**Decision rule (once, on the forward block):** run `knobs.member` at its base params under every candidate
(protocol identical to members, cells cached under `knobs/<key>/`). For each alternative `A_i` vs default `D`:
`diff_fwd` / `p_pos_fwd` = paired block bootstrap of per-era fit20 on the forward test eras; `diff_cv_indep` = mean
per-era diff over the CV folds whose test block is >= 20 eras from the forward test block (independent
confirmation). Holm over the m alternatives sorted by `1 - p_pos_fwd`: rank j passes iff
`1 - p_pos_fwd <= alpha_total / (m - j + 1)` (stop at the first failure) AND `diff_cv_indep >= 0`. Per dimension at
most one winner (largest `diff_fwd`); chosen = default with each dimension's winner; the combination is reported,
never re-tested. `knobs.json`:
`{"status":"done","member","alpha_total","default":key,"candidates":[{key, era_start, rows_rule, time_decay, fit_cv,
fit_fwd, diff_fwd, diff_se_fwd, p_pos_fwd, diff_cv_indep, holm_rank, holm_threshold, pass}],"chosen":{...,key},
"decided_at"}`. Resume reuses it; a config whose candidate keys differ raises `ValueError("knobs already decided
for this run-id; start a new run-id")` — the knob is never re-decided silently.

## 13. Search adoption gate (`train.py`; `search.py` unchanged API) [train]

`search/<m>.gain.json` gains `paired_cv` (`paired_block_bootstrap` of per-era fit20, top-k mean minus base cell),
`adopted` and `adopt_reason`. Adopted iff `paired_cv.p_pos >= p_pos_adopt` and `forward_gain >= 0` (or
`adopt_gate: false`). Not adopted -> the member is the BASE cell (its OOF, fold blobs and full refits), suspect flag
unchanged. `search/<m>.adopted.json = {"adopted": bool, "params_list": [...], "best": {...}}`; `params.best_of`
resolves to `best` (search top-1 if adopted, else the base params). spec.json gains `search_adopted`.

## 14. External members + `folds_eras.json` (`train.py`; consumed by `scripts/mlp_train.py`) [train] -> [nn]

Right after the universe is decided train.py writes `artifacts/<run>/folds_eras.json`:
`{"universe": Universe.to_dict(), "scoring_min_era": int, "folds": [{"name", "train_eras": [int], "test_eras": [int],
"inner_train_eras": [int], "inner_val_eras": [int], "count_sources": [fold names]}], "forward": {same}|null,
"full_train_eras": [int]}` (count_sources from `train.frozen_counts`; the full refit uses the median over CV folds).
A member with `family: mlp` is EXTERNAL: `train_member` never fits it. If `members/<m>/spec.json` has
`status: done`, `universe == rd.universe.key()` and `oof_<m>.parquet` exists, it is loaded like a finished member
(`corr_with_bench_val`/`rho_c` computed via `models.predict_member` if absent); otherwise it is skipped with WARNING
`external member <m> pending: run scripts/mlp_train.py --run-id <R>` and a `metrics.json.notes` entry (exit code
unchanged). External artifacts follow section 8 exactly: `oof_<m>.parquet` (index id; `era` int32, `fold` int8,
-1 = forward, `pred` float32), `members/<m>/{fold<k>.bin, forward.bin, full_s<seed>.bin, spec.json}`; spec.json
carries the tree keys (`inner_iters` = best epochs, `frozen_n_iter_per_fold`, `frozen_n_iter_forward`,
`frozen_n_iter`, `count_sources`, `fit_seconds`, `max_fit_seconds`, `member_seconds`, `full_blobs`, `feature_cols`,
`missing_cols`, `universe`, `status`) plus `seeds`, `device`, `torch_version`, `parity_max_abs`.

## 15. Timing drop + validation side metrics (`train.py`) [train]

A `cat` member whose FIRST fit takes longer than `timing.catboost_max_fit_seconds` stops immediately: spec.json
`{"status": "dropped_timing", "first_fit_seconds": s}`, no oof file, excluded from assembly/blend, note
`"<m>: CatBoost dropped (s > cap)"`; resume keeps the drop. `validation_bench_corr` additionally returns
`rho_c` = mean per-era corr(prediction, core proxy) on validation; `metrics.json.members.<m>.rho_c` is filled
(report.py already reads it).

## 16. Blend: arms, correlation, admission ladder, final selection (`blend.py`) [blend]

```python
def arm_of(spec: dict) -> str                          # spec["arm"] or ("aux" if group == "aux" else name)
def blend_weights(groups: dict[str, str], everest_weight: float, arms: dict[str, str] | None = None) -> dict[str, float]
    # arms None -> M1 behaviour. Else hierarchical: group weight -> equal per arm in the group -> equal per member.
def member_corr_matrix(Z: pd.DataFrame, era: pd.Series) -> pd.DataFrame   # mean per-era Pearson corr of gaussianised scores
def admission_ladder(Z, oof, Zf, fwd, specs, cfg) -> tuple[list[str], list[dict]]   # (admitted member names, ladder rows)
def select_neutralisation(grid: dict, cfg: dict) -> dict       # section 16.3
def final_selection(run: dict) -> dict                         # orchestrates 16.1-16.3; called by blend_run
```
**16.1 Redundancy + admission (M3/M4/M6).** Members are visited in ladder order (base arms, then `admission.order`,
config order inside an arm). A member whose `member_corr_matrix` entry with any already-kept member exceeds
`gates.member_corr_max` is dropped as `redundant(<other>)`. Incumbent `I` = base arms. For each arm `A`: candidate
`C` = I plus A (hierarchical EW over C vs over I), per-era fit20 diff on CV OOF -> `paired_block_bootstrap`;
`fwd_diff` on the forward block; `metrics.paired_sharpe_bootstrap(fit_C, fit_I)`; `corr_with_incumbent` = mean
per-era corr of the arm's mean z with I's blend z. Admit iff `corr_with_incumbent <= member_corr_max` AND
`fwd_diff >= 0` AND (`p_pos >= p_pos_adopt` OR (`dsharpe >= admission.sharpe_gain` AND `p_pos_sharpe >=
p_pos_adopt`)). Admitted arms join I (sequential, pre-registered order, one look each). Ladder row: `{arm, members,
admitted, reason, diff_mean, diff_se, p_pos, fwd_diff, dsharpe, p_pos_sharpe, corr_with_incumbent,
dropped_redundant: [...]}`.
**16.2 Rule.** The existing nested NNLS gate runs on the admitted set vs hierarchical EW (unchanged semantics,
cross-fitted headline). Members not admitted get weight 0 (bundle drops them).
**16.3 Neutralisation constants (M5, staking-aware; never from the diag lane).** `z` = the chosen blend's OOF z
(cross-fitted if NNLS). `metrics.regime_grid` over `p_b_grid x p_c_grid` (caps 0.75 / 0.5). For each cell at
`central_mean`: `S` = `mix.score_sharpe`, `fit` = `mix.fit`. Eligible iff (a) `fit >= fit(0,0) - fit_cost_max` and
(b) `mix.score` at `pessimistic_mean` >= the (0,0) cell's `mix.score - mix.score_se` at `pessimistic_mean`; (0,0) is
always eligible. Among eligible: best `S`; tie set = cells with `S >= S_best - sharpe_tol`; inside the tie set
maximise the worst-regime score `W = min over {bpos, bneg} x all bench_fit_means of score`; `W` within `worst_tol`
-> smaller `p_b + p_c`, then smaller `p_b`. Output `blend.json["neutralisation"] = {"p_b", "p_c", "p_source":
"simulation", "criterion", "table": [{p_b, p_c, fit, score, score_sharpe, W, eligible, in_tie_set}], "reason",
"z_source": "crossfit"|"ew"}` and top-level `blend.json.p_b/p_c` are set to the chosen values. `select: false` ->
`p_source: "config"` with the config values. `blend.json["regime"]` = the grid slice at the chosen `p_c` in the OLD
shape (`{"<mean>": {"p_b=<x>": {bpos,bneg,mix}}, "meta"}`) so report.py keeps working; the full grid goes to
`blend.json["regime_grid"]`.
**16.4 blend.json additions:** `arms`, `member_corr` (`{"names": [...], "matrix": [[...]]}`; also
`member_corr.csv`), `ladder`, `admitted`, `neutralisation`, `regime_grid`, `final = {admitted_arms, rule, p_b,
p_c, p_source, oof: summarise(...), forward_fit, score_mix_central, score_sharpe_central, W}`; metrics.json gets
`blend.final` and `paired.<arm>_ladder`.

## 17. Metrics additions (`metrics.py`) [blend]

```python
def regime_grid(z_pred, target, era, direction, core_proxy, *, p_b_grid, p_c_grid, bench_fit_means=(0.0, .009, .02),
                bench_fit_sd=0.035, n_draws=500, seed=0, direction_weight=0.8) -> dict
    # {"<mean>": {"p_c=<y>": {"p_b=<x>": {"bpos","bneg","mix"}}}, "meta"}; the SAME benchmark draws serve every
    # (p_b, p_c) cell (common random numbers). regime_simulation(p_c=scalar) keeps its output shape.
    # Each regime block gains "score_sharpe" = mean over draws of (across-era mean / std of that draw's per-era
    # score) -- the benchmark noise stays in the variance a staker sees -- and "score_sharpe_se".
def paired_sharpe_bootstrap(a: pd.Series, b: pd.Series, *, block=20, n_boot=2000, seed=0) -> dict
    # per-era series on the same era index; circular moving blocks with identical indices for a and b;
    # {"dsharpe", "se", "p_pos", "ci90", "n_eras"}; sharpe = mean/std per era (not annualised)
```

## 18. Runtime + bundle additions (`_runtime.py`, `bundle.py`, `models/mlp.py`) [nn]

* `_runtime.SUPPORTED_FAMILIES = ("lgbm", "xgb", "cat", "ridge", "mlp")`. The MLP numpy forward lives IN
  `stealth/_runtime.py` — the one module `bundle.write_pickle` already registers by value — so the pickle still
  references a single module and the `pkl_check` byte-scan whitelist (`._runtime` only) stays as is. No second
  by-value module, no torch anywhere in the package (a test imports every `stealth.*` module in a subprocess and
  asserts `"torch" not in sys.modules`; another greps the package for `import torch`).
* `load_member("mlp", blob)` -> dict from npz (`allow_pickle=False`): `W0..W{L-1}` float32 (in x out), `b0..`,
  `x_mean`, `x_std` float32[n_in], `columns` str[n_in], `act` = "gelu_tanh", `n_layers` int.
  `mlp_forward(model, X) -> float64[n]`: `h = (X - x_mean) / x_std`; hidden layers `h = gelu_tanh(h @ W + b)` with
  `gelu_tanh(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))` (torch `GELU(approximate="tanh")`); last layer
  linear; dropout is inference-off; the aux head is never exported.
* Per-member feature subsets, by name: `input: nan` members index the nan matrix by the positions of their
  `feature_cols` inside bundle `feature_cols` (unchanged); `input: linear` members (ridge, mlp) select
  `model["columns"]` BY NAME from `linear_frame` (mlp: the 178 base columns). Every member's columns must be a
  subset of bundle `feature_cols` / linear-frame columns (`ValueError` at `make_predict`); live column order is
  irrelevant (pkl_check's shuffled-columns call covers it).
* `stealth/models/mlp.py` (numpy only, family contract): `to_bytes(dict) -> npz bytes`, `from_bytes`,
  `predict = _runtime.mlp_forward`; `fit_with_early_stopping` / `fit_fixed` raise `NotImplementedError("mlp is
  trained externally by scripts/mlp_train.py")`. `models.FAMILIES` gains "mlp".
* `build_bundle(run_dir, *, p_b=None, p_c=None, require_frozen=False)`: p resolution = explicit args
  (`p_source "override"`, WARNING) > `blend.json.neutralisation` (`"simulation"`) > `blend.json.p_b/p_c`
  (`"config"`) > defaults (`"default"`, WARNING). `require_frozen=True` (CLI `--require-frozen`, used by handoff 04)
  raises unless `p_source == "simulation"`. Caps unchanged. Bundle and manifest gain `p_source`, `blend_rule`,
  `admitted_arms`; manifest `members[]` gains `arm`, `n_features`. The fold-blob fallback is refused when
  `spec.feature_cols_per_fold` differs from `spec.feature_cols` (sharpe_subset). `assert_plain(bundle)`: only
  dict/list/str/bytes/int/float/bool/None values (no torch / numpy objects) before pickling.

## 19. MLP training (`scripts/mlp_train.py`, `requirements-gpu.txt`, `scripts/setup_gpu.sh`) [nn]

* GPU venv `.venv-gpu` (never the train venv): `requirements-gpu.txt` = numpy/pandas/pyarrow/PyYAML/scipy at the lane
  pins; torch installed by `scripts/setup_gpu.sh` from `https://download.pytorch.org/whl/cu128` (Blackwell sm_120
  needs CUDA >= 12.8 wheels), then `pip install --no-deps -e .`. setup_gpu.sh ends with a GPU smoke
  (`torch.cuda.is_available()`, capability, 4096^2 matmul) and prints `GPU OK` or `GPU FAIL -> skip MLP`.
  Sandbox analogue: `/opt/ml/bin/python3` (torch 2.15.0+cpu) with `PYTHONPATH=<repo>` and `--device cpu`.
* CLI: `$GPU_PY scripts/mlp_train.py --run-id R [--artifacts DIR] [--member mlp_everest] [--device auto|cuda|cpu]
  [--seeds 0 1 ...] [--epochs-cap N] [--threads N] [--resume]`. Imports only torch, numpy, pandas, yaml,
  `stealth.data`, `stealth.knobs`, `stealth._runtime` (parity). Reads `config.yaml`, `folds_eras.json`,
  `features.json`; refuses if `folds_eras.json` is missing ("run stealth.train first").
* Protocol (identical to trees): per seed and fold, inner early stopping on `inner_val_eras` (metric: mean per-era
  Pearson of the main head; patience from params) -> best epoch; frozen epochs per fold = median over that fold's
  `count_sources` of the seed-median best epochs; refit on the fold's train eras for the frozen count; predict the
  test eras; OOF = mean over seeds. Full refit per seed on `full_train_eras` with the median CV count. Training rows
  = universe mask, sample weights = universe weights. Loss = MSE(main) + `aux_weight` x masked MSE(aux heads, NaN
  aux targets masked). Input = `features_linear` base 178 columns, standardised with that fit's training mean/std.
  `torch.manual_seed(seed)`, deterministic algorithms where available.
* Export per blob to the npz format of section 18; after export the numpy `mlp_forward` is compared with the torch
  eval forward on <= 2000 rows: `max_abs > 1e-4` -> exit 1 (`parity_max_abs` recorded). Artifacts per section 14;
  resumable per (seed, fold) under `members/<m>/_work/`; exit 0 ok, 1 error, 4 no usable device (handoff then
  skips the member: train.py marks it pending, the ladder never sees it).

## 20. Artifacts, pipeline driver, report, minifull [train] [docs]

New files under `artifacts/<run>/`: `knobs.json`, `knobs/<key>/` (cached cells), `folds_eras.json`,
`search/<m>.adopted.json`, `member_corr.csv`; `members/<m>/spec.json` may carry `status: dropped_timing`; blend.json
and metrics.json keys per sections 16-17.

`scripts/run_pipeline.sh <config> <run_id> [--gpu-py PATH] [--no-mlp] [--pkl models/<run_id>.pkl]` [docs]:
`train --resume` -> (config has an mlp member and GPU_PY set) `mlp_train.py --resume` -> `train --resume`
(re-assembles with the external member) -> `blend` -> `bundle --require-frozen` (when
`blend.neutralisation.select`) -> `pkl_check` -> `diag` (file only, never `--upload`) -> `report`. Every step is
idempotent; re-running the same command resumes. MLP failure (exit 1/4) is logged and the pipeline continues
without it.

`report.py` NN-metrics.json additions [docs]: `knobs` (candidate keys + diff/p_pos/pass, chosen key), `search`
(per searched member: gain, forward_gain, paired_cv, adopted), `ladder`, `member_corr_max_pair` (+ matrix),
`neutralisation` (table + chosen + p_source), `final`, `dropped` (timing / redundant / pending external), the blend
row reports `chosen` (not `ew`). Universe keys and arm names are config names; hyperparameter values never appear.

`configs/minifull.yaml` [train] (sandbox, 2 threads, <= 8 min end to end incl. MLP on CPU torch) exercises every
new path: scoring eras >= 5000, pool from 4600; 3 folds + forward (5925/5945); knobs `era_start [5000, 4600]`,
`rows_rule [none, filter80, weight80]`, `time_decay [false, true]`; `lgbm_everest` TPE search 3 trials over a
4-cell grid, top_k 2 (adoption gate); 2 aux (lgbm, best_of); `xgb_everest` exhaustive 2-cell grid top_k 2; 1 tiny
cat (50 iterations); ridge; sharpe arm; 2 bags; `mlp_everest` 1 seed, `es_cap: 3`, hidden [32, 16]; fixed
`n_estimators` <= 80 or `es_cap` <= 120 everywhere; admission order [aux, xgb, cat, ridge, sharpe, bags, mlp];
`simulation.n_draws: 30`, `p_c_grid: [0.0, 0.25, 0.5]`; `neutralisation.select: true`; `refit.seeds: [0]`.
`tests/test_minifull.py` (marked `slow`, skipped without data) checks that load_config accepts it.

## 21. As built (integration pass, 2026-11-13) — what differs from / extends sections 11-20

Implementation facts the groups reported plus the integration fixes. Nothing here changes an M1 config's behaviour.

**train / knobs / cv** [train]
* `train.per_era_fit_fast` / `per_era_spearman_fast`: vectorised `metrics.per_era_fit` / `per_era_spearman` semantics
  (SDK parity ~1e-16, `tests/test_train_m2.py`). Used by knobs, search objectives, member metrics and (integration)
  every fit in `blend.py` (`blend.fit_series`, `evaluate_blend`). `metrics.per_era_*` keep the SDK loop.
* `train.validation_side_metrics(...) -> {"corr_with_bench_val", "rho_c"}` implements section 15's `rho_c`;
  `validation_bench_corr` keeps its float return.
* `knobs.json` extras beyond section 12: `indep_cv_folds`, `winners`, `combination_retested: false`; per candidate
  `dimension`, `is_default`, `equivalent_to_default`, `holm_pass`, `cv_indep_ok`, `reason`, `n_eras_cv_indep`. A
  candidate whose training rows and weights equal the default's reuses the default cell (diff 0, p_pos 0, still in m).
* metrics.json additions: `universe {key, era_start, rows_rule, time_decay, rows_low_weight}`, `knobs {...}`,
  `dropped [{member, status: dropped_timing|pending_external, ...}]`, `members.<m>.{rho_c, arm, universe,
  search_adopted}`, `members.<m>.search_gain.{paired_cv, adopted, adopt_reason, n_trials, universe}`.
* `load_config` rejects `blend.neutralisation.select: true` unless 0.0 is in both `p_b_grid` and `p_c_grid`.
* `configs/test.yaml` sets `search.lgbm_everest.adopt_gate: false` (the gate defaults to true; M1 stays top-k).
* `configs/full.yaml`: `gates.baseline_fit: 0.0229` = handoff 02's UNSEARCHED `lgbm_everest` base cell.

**blend** [blend]
* `blend.paired_vs_baseline` (integration): paired 20-era block bootstrap of the REPORTED blend z (cross-fitted when
  NNLS) minus `train.baseline_member(cfg)` on CV OOF + `fwd_diff`. Written to `blend.json.paired_vs_baseline` and to
  `metrics.json.paired["blend_vs_<baseline>"]`; report.py's blend row reads it (its recompute is then skipped).
  Informational: the ladder already chose; a tie or loss means "keep the baseline". Post-selection, so optimistic.
* `blend.json.forward_weights_source`: NNLS forward fit uses weights fitted on OOF eras >= 20 before the forward block.

**runtime / bundle / MLP** [nn]
* `bundle --members a b ...` bundles exactly that subset (blend weights renormalised when all positive, else equal
  weights + WARNING; manifest `member_subset`, `weights_source`). Handoff 03 uses it for the single-LGBM baseline.
* The pickle bytes do not contain `built_at` (manifest only) but do contain `run_id` and `git_sha`: a rebuild from
  identical artifacts at the same commit reproduces the sha (checked on the sandbox); a new commit or any change to
  `_runtime.py` changes it. Handoff 03 therefore checks manifest CONTENT, never a sha match with handoff 02.
* `scripts/mlp_train.py` (integration): the loss fits STANDARDISED targets ((y - mean) / sd of the fit rows; aux the
  same, NaN-aware). Raw [0, 1] targets spent the first epochs on the intercept, so inner early stopping picked noise
  (minifull OOF fit 0.0003 -> 0.0160 with the same epochs). `TARGET_SCALING = "standardised_v1"` is part of the
  resume signature, so older `_work/` caches retrain.
* `scripts/setup_gpu.sh` (integration): installs torch's pure-Python deps (typing-extensions, filelock, sympy,
  networkx, jinja2, fsspec) from PyPI BEFORE torch, because the CUDA index serves typing_extensions 4.16.0 only as an
  sdist whose build backend it lacks (`No matching distribution found for flit_core`). `--help` prints the header;
  any other argument exits 64 (it used to ignore arguments and run the whole setup).

**ops / report / pipeline** [docs]
* `stealth.diag`: `--preflight` (offline, exit 0 only when ready), `--fetch-result [--wait]` (read-only),
  `--attach-pkl` (opt-in `model_pkl`, python 3.11), `--new-upload` (overrides the upload-once guard),
  `--wait-timeout` (default 1800 s); `--pkl` optional with `--fetch-result`; exit 2 = uploaded, still scoring.
  Sidecars: `<out>.meta.json` (pkl sha256, rows/eras coverage, file sha256); `<out>.upload.json` adds `panel`,
  `result`, `result_at`, `built_from_pkl_sha256`, `attached_pkl`, `pkl_sha256`.
* `stealth.report`: `--diag-json <file>.upload.json`, `--no-recompute`. The diag ratio is taken against the uploaded
  pickle's single member when `models/*.manifest.json` with the recorded sha bundles exactly one member
  (`diag.oof_reference`), else against the blend row (integration).
* `scripts/run_pipeline.sh`: also `--py`, `--from STEP`, `--mlp-device`, `--mlp-epochs-cap`, `--mlp-seeds`,
  `--mlp-threads`, `--pkl-check-python`, `--diag-eras`, `--no-diag`, `--report NN`, `--dry-run`; writes
  `artifacts/<run>/pipeline.log`; exit 64 on bad usage.
* pytest marker `slow` is registered in pyproject.toml.

## 22. Fix pass after the three independent reviews (2026-11-14) — supersedes 12-21 where they differ

Reviews: `docs/reviews/m2-m6-{statistics,contract,handoff}.md` (each ends with a "Fix verification" section naming the
regression tests). `PROTOCOL_VERSION = "m2m6-fix1-audit"` (train.py) is part of every run fingerprint, cell id and
study name: nothing cached under an older protocol can be resumed; use a new run-id after any protocol change.

**A. Run fingerprint** [train]. `artifacts/<run>/run_fingerprint.json = {"schema": "stealth.fingerprint/1",
"fingerprint", "config_sha" (config minus `run.threads` / `timing.progress_every_seconds`), "data" ({train,
train_benchmark}: sha256), "protocol", "created_at"}`. `train_run` checks it BEFORE writing anything: another config /
data / protocol for an existing run-id, or a run dir with artifacts but no fingerprint -> exit 1, `notes` = "start a
new run-id", nothing in the old run dir touched. Run fingerprint is copied into `folds_eras.json`, every member's
`spec.json` (`run_fingerprint`, `protocol`) and every external spec (`scripts/mlp_train.py`).

**B. Cache and status identity** [train]. `cell_hash(params, universe, fingerprint)`; `result.json` of every search /
knob / baseline cell records `cell_id` + `protocol` and `_load_protocol(d, cell_id)` refuses another cell. Optuna
study name `"<member>@<fingerprint[:16]>-<universe key>"` (`search.run_search(..., study_name=)`). `train_member`
writes `spec.json {"status": "running", ...}` and deletes the member's `oof_/audit_` files BEFORE any protocol work, and
writes `status: done` LAST; `_reusable_spec` additionally requires the run fingerprint, the declared `full_blobs` on
disk and (audit on) `audit_<m>.parquet`.

**C. External members** [train] [nn]. `train.external_problem` refuses a spec that is not `done`, has another universe or
run fingerprint, other `seeds` or `epochs_cap` than the config, no / missing declared `full_blobs`, no OOF, or (audit on)
no `audit_<m>.parquet`. CLI: `python -m stealth.train ... [--skip-external | --require-external]` (mutually exclusive):
skip never loads `family: mlp` members (`dropped` status `skipped_external`); require turns a refusal into exit 1.
`metrics.json.external_members = {name: "loaded" | "skipped_external" | "pending"}`. `blend.load_run` refuses an
`oof.parquet` member whose spec is missing, not `done` (`None` = legacy M1 allowed) or of another run fingerprint
(external members must carry it).

**D. Count sources (statistics review 1).** `cv.independent_inner_folds` / `assert_count_sources` check the source fold's
WHOLE `train_eras` (inner fit + inner validation + target-dependent feature selection) against the target's test block
(>= 20 eras). Chronological two-sided folds: every CV fold uses its own inner split only; the forward block uses its
own split plus CV folds whose training ends >= 20 eras before it (full: fold4). The full refit keeps the median over CV
folds.

**E. Paired inference (statistics review 6).** `cv.PairedResult` / `as_dict()` gain `p_value` (null-centred one-sided
bootstrap p-value, `(1 + #{mean* - mean >= mean}) / (n_boot + 1)`), `n_blocks` (whole blocks) and `inference_valid`
(`n_blocks >= cv.MIN_EFFECTIVE_BLOCKS = 2`; else `p_pos = 0.5`, `p_value = 1`). `metrics.paired_sharpe_bootstrap` gains
`n_blocks` / `inference_valid` (p_pos 0.5 when invalid). Gates fail closed without inference: search adoption, NNLS,
both ladder routes (`admit_decision` reads `inference_valid` / `sharpe_inference_valid`), knobs. Holm (`knobs.holm_decide`)
runs on `p_value_fwd` (knobs rows add `p_value_fwd`, `n_blocks_fwd`, `inference_valid_fwd`, `holm_p`); `p_pos` stays a
descriptive statistic and the pre-registered `p_pos >= 0.8` gates are unchanged. No exact family-wise claim.

**F. NNLS (statistics review 5).** `blend.purged_fit_mask(era, blocks, gap=20)`: meta-fit rows lie outside and >= 20
eras from every evaluation block. `nested_nnls`: fit = folds < K-1 purged against fold K-1 AND the forward test block;
the gate's forward diff AND the reported forward fit use these same nested weights (`forward_weights_source` "gate
weights"); `< 40` purged eras -> no NNLS (`nested_weights` None, fail closed). `crossfit_nnls` purges each fold's fit
rows around that fold; a fold without rows raises and the blend falls back to EW. `forward_nnls_weights` is gone.

**G. Simulation == deployed recipe (statistics review 3).** `metrics.cell_scores(P, m, p_b, p_c) -> (zg, fit, unq, inov)`
is the regime grid's per-cell kernel: neutralise the weighted member blend (no re-gaussianisation) against the benchmark,
then the core proxy (valid raw values ranked first, missing rows z = 0 afterwards, as `_runtime`); score with SDK
semantics -- `fit20`, `unq20` (zero rule, >= 5 rows), and `inov` COMPLETE-CASE (`metrics.inov_rows`: rows without a core
average dropped, prediction re-ranked and target re-centred). Parity: `tests/test_metrics.py::test_cell_scores_match_
runtime_and_sdk` (1e-10 vs `_runtime` + `everestapi.scoring`, 20% missing-core rows).

**H. Reserved audit partition.** Config `cv.audit: {enabled, train_end, test_start}` (`load_config` checks the >= 20 gap,
`cv.forward.test_start < train_end`, every `knobs.era_start < train_end`). `RunData.eras` (CV + forward) and
`RunData.full_train_eras` (full refits, `sharpe_subset` for full refits) stop before `train_end`; `RunData.audit_eras`
= eras >= `test_start`; `assert_development_only()` guards it. Per member `audit_<m>.parquet` (index id; `era` int32;
`pred` float64) = mean over the member's declared full-refit blobs (`train._predict_full`; MLP: the exported npz via
`_runtime.mlp_forward`) -- exactly what the bundle averages. `train.audit_baseline` refits `knobs.member` at its CONFIG
params under the DEFAULT universe (count from its own knob/baseline cell) on the pre-audit eras -> `baseline/
{spec.json, full_s*.bin, cell/}` + `audit_baseline.parquet`. `train.assemble_audit` -> `audit.parquet` (index id over
every audit row; `era`, `target_everest` float32 copied, `m__<member>` float64, `baseline__<name>`); incomplete coverage
raises. `folds_eras.json` gains `run_fingerprint`, `protocol`, `audit {train_end, test_start, eras, ...}` and
`full_train_eras` = pre-audit eras. `metrics.json.audit = {enabled, train_end, test_start, n_eras, eras, file, baseline}`.
No module reads an audit label before `stealth.audit` (regression: `tests/test_audit.py::
test_audit_labels_never_reach_a_decision` permutes them and compares every decision artefact).

**I. `python -m stealth.audit --run-id R`** (after blend, before bundle; exit 0 / 1). Writes `audit.json`:
`{schema "stealth.audit/1", run_id, status ok|failed, created_at, run_fingerprint, blend_sha256, partition {train_end,
test_start, eras [min,max], n_eras, n_rows}, frozen {rule, members, admitted_arms, p_b, p_c, p_source}, blend
(summarise of the UNNEUTRALISED frozen blend's per-era fit20), blend_spearman_mean, baseline {name, label, **summarise},
paired_vs_baseline (PairedResult dict), sharpe_vs_baseline, drawdown {label "... NOT platform P&L ...", blend {max_drawdown,
longest_underwater_eras, final_cum}, baseline}, stability {segments [{eras, n_eras, blend_fit, baseline_fit, diff}] x 4,
n_segments, blend_positive_segments, diff_positive_segments}, bootstrap_sensitivity [{block 10|20|40, se_block, p_pos,
p_value, ci90, n_blocks, inference_valid}], frozen_core_step {p_c, fit, inov_proxy, fit_unneutralised,
inov_proxy_unneutralised, note}, scenario {note, p_b, p_c, bench_fit_mean, n_draws, mix, bneg, reference_00}, verdict
{technical "complete", audit_win true|false|null, strength "strong"|"weak"|null, text}, notes, seconds}`. `audit_win` =
`inference_valid` AND diff_mean > 0 AND `p_pos >= gates.p_pos_adopt` (the pre-registered paired rule); `strength` is
"strong" when the one-sided `p_value` < 0.05, else "weak" -- reported as information; technical completion is not an
audit win. On error:
`{schema, run_id, status "failed", error}`. Never writes `blend.json`; nothing reads `audit.json` to choose anything.

**J. Suspect reference.** `train.baseline_fit(cfg, fits, base_cell_fits)`: `gates.baseline_fit` (only for the SAME scoring
layout) > the in-run baseline member's UNSEARCHED base cell (`search_gain.base_params_fit`) > its OOF mean.
`full.yaml` sets `gates.baseline_fit: null` (handoff 02's 0.0229 was scored on folds over 2800..6524).
`blend.json.suspect_reference = {fit, label}`.

**K. Contract (contract review 1-7)** [nn]. `predict.check_upload_artifact(pkl)` is the one upload gate (lane upload,
`diag --upload`, `--attach-pkl`, `--preflight`), run before any client exists: manifest sha256 == file, `pkl_check.passed`
AND `pkl_check.sha256` == file, manifest and checker Python 3.11.x. Manifest `pkl_check = {passed, status
(running|passed|failed|error), at, sha256, python, libs (verified in the checker), checker, eras, seconds_per_era_max,
isolation {stealth_importable, torch_importable}, problems[, error]}`, set to a non-passing `running` record before
every attempt. The checker verifies the installed lane pins and that neither package is importable, and hashes the
bytes it checked; all paths are made absolute before the subprocess. `bundle.member_blob_files` honours `spec.full_blobs`
exactly (missing or undeclared extra refit -> error; legacy glob only without the key); `check_fold_fallback` validates
the loaded fold names. `p_source` gains `"inherited"` (member subset whose positive set or relative weights differ from
the selected blend; refused by `--require-frozen`); explicit `--p-b/--p-c` stay `"override"`. Non-finite / negative
weights and constants are refused in `build_bundle` and `make_predict`; `validate_predictions` requires a numpy float
column. Manifest gains `p_reason`, `blob_files`. `monitor` requires the pass to be bound to the manifest sha256.

**L. Pipeline + report (handoff review)** [docs]. `scripts/run_pipeline.sh`: steps `train` (`--resume
--skip-external`) -> `mlp` (`--device cuda` by default; exit 1/4 = optional skip, >= 128 interrupted, other codes fail)
-> `retrain` (`--require-external` after a finished MLP; `--skip-external` when the MLP did not finish in an attempt
that skipped train) -> `blend` -> `audit` (when `cv.audit.enabled`) -> `bundle` -> `pkl_check` -> `diag` (file only) ->
`report` (`--report NN`). `--no-mlp` from blend onwards refuses (exit 64) while `oof.parquet` still holds an external
member. Every step runs in its own process group (INT/TERM stop that group only); stdout+stderr in
`artifacts/<run>/pipeline_steps/<ts>-<pid>-<step>.log`; `artifacts/<run>/pipeline.json` (schema `stealth.pipeline/1`,
atomic): `{run_id, config, pkl, status running|ok|failed|interrupted, exit_code, failed_step, last_step, mlp,
report_step, attempts [{started_at, from_step, status, steps [{name, status, exit_code, seconds, log_file, ...}], ...}]}`;
the compute result is recorded before the report step, whose own outcome is `report_step`. `stealth.report` derives the
status from that evidence (pipeline.json, metrics/leak, manifest pkl_check bound to the bytes, the diag file's
`.meta.json` coverage, `audit.json`): failed evidence -> `failed`, missing evidence -> `incomplete`; adds pipeline,
evidence, packaging, diagnostics-file and audit sections; labels development figures "selection estimates";
`scrub()` also redacts signed-URL queries (X-Amz-*, X-Goog-*, signature/token/expires), `scrub_params()` drops
hyperparameter name=value pairs from log tails, home directories shorten to `~/`. The pre-commit hook blocks signed URLs.


## Local research basis models (2026-11-14)

The research-only `basis` runtime family consumes the declared raw feature columns with `input: nan` and
numeric/string NPZ blobs (`allow_pickle=False`). Existing linear, polynomial, categorical, quadratic,
residual-tree and numeric neural forms share their transforms between training and inference. Neural weights
are exported to NumPy; no torch import is permitted in the package or clean runtime. External training scripts
are explicitly allowlisted in `tests/test_no_torch.py`.

The `latent_tree` basis form stores a fit-only Gaussian-bin projection, its feature mean, scalar `context` and
`add_raw` flags, and serialized LightGBM bytes. Projection and layer dimensions and finite numeric arrays are
validated before inference. With `context` enabled, latent coordinates are standardized using only the current
cross-section's features. Training and offline evaluation must transform each era separately; portable
`predict` receives one live cross-section. Passing multiple eras together changes the defined model.
`stealth.diag` already invokes prediction separately by era. Column order is resolved from declared feature
names; row permutation must preserve scores after reindexing.

`configs/campaign.yaml` and `configs/representations.yaml` define separate local research workflows, not the
`stealth.train` schema. Their source/data-bound plan hashes reject changed code or configuration under a cached
run ID. Discovery readers exclude confirmation and old-audit labels. Reports retain rejected candidates and
label all historical scores as selection estimates. These workflows perform no platform write; a research
score does not imply a promoted artifact or a clean-check pass on a full-data pickle.
