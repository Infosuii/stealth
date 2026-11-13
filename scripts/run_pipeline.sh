#!/usr/bin/env bash
# stealth -- end-to-end pipeline driver (docs/INTERFACES.md section 20). Idempotent and resumable: after ANY
# interruption re-run the SAME command; every step resumes from its cache or repeats deterministically.
#
#   bash scripts/run_pipeline.sh <config> <run_id> [--gpu-py PATH] [--no-mlp] [--pkl models/<run_id>.pkl]
#        [--py PATH] [--from STEP] [--mlp-device auto|cuda|cpu] [--mlp-epochs-cap N] [--mlp-seeds "0 1"]
#        [--mlp-threads N] [--pkl-check-python PATH] [--diag-eras N] [--no-diag] [--report NN] [--dry-run]
#
# Steps (STEP names for --from):
#   train      $PY -m stealth.train --config C --run-id R --resume
#   mlp        $GPU_PY scripts/mlp_train.py --run-id R --resume      only when the config has a `family: mlp` member,
#              --gpu-py is given and --no-mlp is not; exit != 0 (1 error, 4 no device) is logged "MLP skipped" and the
#              pipeline continues without the member (train marks it pending; the ladder never sees it)
#   retrain    $PY -m stealth.train ... --resume again: re-assembles OOF with the finished external member (only
#              after mlp exited 0, or when started with --from retrain)
#   blend      $PY -m stealth.blend --run-id R                       (the regime grid can be silent for ~15 min)
#   bundle     $PY -m stealth.bundle --run-id R --out PKL [--require-frozen when blend.neutralisation.select]
#   pkl_check  $PY -m stealth.pkl_check PKL [--venv-python P]
#   diag       $PY -m stealth.diag --pkl PKL --out artifacts/diag/<R>_validation.parquet [--eras N]   FILE ONLY:
#              this script never passes --upload / --submit / any platform write
#   report     $PY -m stealth.report --run-id R --handoff NN --no-git   only with --report NN (the handoff's own
#              report step commits and pushes)
#
# Defaults: PY=<repo>/.venv/bin/python, PKL=models/<run_id>.pkl. Relative paths resolve against the repo root.
# Step lines go to stdout and artifacts/<run_id>/pipeline.log; each module writes its own detailed log.
# Exit: 0 when every step succeeded (an MLP skip is tolerated), else the exit code of the first failing step
# (train / blend 2 = leak or gate failure). --dry-run prints the commands and runs nothing. Never reads .env.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEPS=(train mlp retrain blend bundle pkl_check diag report)

usage() { sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "run_pipeline: $*" >&2; exit 64; }

resolve() {  # resolve <path>: absolute path; relative paths are tried against $PWD, then the repo root
  local p="$1"
  case "$p" in
    /*) printf '%s\n' "$p" ;;
    *) if [ -e "$p" ]; then printf '%s/%s\n' "$(cd "$(dirname "$p")" && pwd)" "$(basename "$p")"
       else printf '%s/%s\n' "$REPO" "$p"; fi ;;
  esac
}

[ "$#" -ge 2 ] || { usage; die "need <config> <run_id>"; }
case "$1" in -h|--help) usage; exit 0 ;; esac
CONFIG="$(resolve "$1")"
RUN_ID="$2"
shift 2
[ -f "$CONFIG" ] || die "config not found: $CONFIG"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || die "run_id must match [A-Za-z0-9_.-]+ (got '$RUN_ID')"

PY="$REPO/.venv/bin/python"
GPU_PY=""
NO_MLP=0
PKL="$REPO/models/$RUN_ID.pkl"
FROM="train"
MLP_ARGS=()
PKLCHECK_PY=""
DIAG_ERAS=""
NO_DIAG=0
REPORT_NN=""
DRY_RUN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu-py) GPU_PY="$(resolve "${2:?--gpu-py needs a path}")"; shift 2 ;;
    --no-mlp) NO_MLP=1; shift ;;
    --pkl) PKL="$(resolve "${2:?--pkl needs a path}")"; shift 2 ;;
    --py) PY="$(resolve "${2:?--py needs a path}")"; shift 2 ;;
    --from) FROM="${2:?--from needs a step}"; shift 2 ;;
    --mlp-device) MLP_ARGS+=(--device "${2:?}"); shift 2 ;;
    --mlp-epochs-cap) MLP_ARGS+=(--epochs-cap "${2:?}"); shift 2 ;;
    --mlp-threads) MLP_ARGS+=(--threads "${2:?}"); shift 2 ;;
    --mlp-seeds) read -r -a _seeds <<< "${2:?}"; MLP_ARGS+=(--seeds "${_seeds[@]}"); shift 2 ;;
    --pkl-check-python) PKLCHECK_PY="$(resolve "${2:?--pkl-check-python needs a path}")"; shift 2 ;;
    --diag-eras) DIAG_ERAS="${2:?--diag-eras needs N}"; shift 2 ;;
    --no-diag) NO_DIAG=1; shift ;;
    --report) REPORT_NN="${2:?--report needs NN}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

FROM_IDX=-1
for i in "${!STEPS[@]}"; do [ "${STEPS[$i]}" = "$FROM" ] && FROM_IDX=$i; done
[ "$FROM_IDX" -ge 0 ] || die "--from must be one of: ${STEPS[*]} (got '$FROM')"
[ -x "$PY" ] || die "python not found or not executable: $PY (run scripts/setup.sh, or pass --py)"

cd "$REPO"
RUN_DIR="$REPO/artifacts/$RUN_ID"
PLOG="$RUN_DIR/pipeline.log"
DIAG_OUT="$REPO/artifacts/diag/${RUN_ID}_validation.parquet"

# config facts (yaml read by the train venv's python): has a family: mlp member? neutralisation selection on?
read -r HAS_MLP SELECT < <("$PY" - "$CONFIG" <<'PYEOF'
import sys
import yaml
with open(sys.argv[1]) as fh:
    c = yaml.safe_load(fh) or {}
mlp = any(isinstance(m, dict) and m.get("family") == "mlp" for m in c.get("members") or [])
sel = bool(((c.get("blend") or {}).get("neutralisation") or {}).get("select"))
print(int(mlp), int(sel))
PYEOF
) || true
[[ "${HAS_MLP:-}" =~ ^[01]$ && "${SELECT:-}" =~ ^[01]$ ]] || die "could not read $CONFIG with $PY (yaml error?)"

log() {
  local line
  line="$(date -u +%Y-%m-%dT%H:%M:%SZ) pipeline[$RUN_ID]: $*"
  echo "$line"
  if [ "$DRY_RUN" -eq 0 ]; then mkdir -p "$RUN_DIR"; echo "$line" >> "$PLOG"; fi
}

RC=0
run() {  # run <step> <cmd...>: runs the command (or prints it with --dry-run); sets RC; never exits by itself
  local step="$1" t0
  shift
  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY-RUN [%s]:' "$step"; printf ' %q' "$@"; printf '\n'
    RC=0
    return 0
  fi
  log "step $step: start: $*"
  t0=$SECONDS
  set +e
  "$@"
  RC=$?
  set -e
  log "step $step: exit $RC after $((SECONDS - t0)) s"
}

fail() {  # fail <step> <rc>
  log "PIPELINE FAILED at step $1 (exit $2). Follow the handoff's 'If it fails' section, then re-run the same command (or --from $1)."
  exit "$2"
}

wanted() {  # wanted <step>: is <step> at or after --from?
  local i
  for i in "${!STEPS[@]}"; do [ "${STEPS[$i]}" = "$1" ] && { [ "$i" -ge "$FROM_IDX" ]; return; }; done
  return 1
}

log "config=$CONFIG run_id=$RUN_ID pkl=$PKL py=$PY gpu_py=${GPU_PY:-none} from=$FROM has_mlp=$HAS_MLP select=$SELECT dry_run=$DRY_RUN"

if wanted train; then
  run train "$PY" -m stealth.train --config "$CONFIG" --run-id "$RUN_ID" --resume
  [ "$RC" -eq 0 ] || fail train "$RC"
fi

MLP_STATUS="n/a (no family: mlp member)"
MLP_OK=0
if wanted mlp && [ "$HAS_MLP" -eq 1 ]; then
  if [ "$NO_MLP" -eq 1 ]; then
    MLP_STATUS="skipped (--no-mlp)"
  elif [ -z "$GPU_PY" ]; then
    MLP_STATUS="skipped (no --gpu-py)"
  elif [ ! -x "$GPU_PY" ] && [ "$DRY_RUN" -eq 0 ]; then
    MLP_STATUS="skipped (--gpu-py not executable: $GPU_PY)"
  else
    run mlp "$GPU_PY" "$REPO/scripts/mlp_train.py" --run-id "$RUN_ID" --resume ${MLP_ARGS[@]+"${MLP_ARGS[@]}"}
    if [ "$DRY_RUN" -eq 1 ]; then
      MLP_OK=1
      MLP_STATUS="would run (retrain runs only if mlp_train exits 0)"
    elif [ "$RC" -eq 0 ]; then
      MLP_OK=1
      MLP_STATUS="ok"
    else
      MLP_STATUS="skipped (mlp_train exit $RC)"
    fi
  fi
  [ "$MLP_OK" -eq 1 ] || log "MLP skipped: $MLP_STATUS -- continuing without the mlp member"
fi

if wanted retrain; then
  if [ "$MLP_OK" -eq 1 ] || [ "$FROM" = "retrain" ]; then
    run retrain "$PY" -m stealth.train --config "$CONFIG" --run-id "$RUN_ID" --resume
    [ "$RC" -eq 0 ] || fail retrain "$RC"
  else
    log "step retrain: skipped (no newly finished external member)"
  fi
fi

if wanted blend; then
  run blend "$PY" -m stealth.blend --run-id "$RUN_ID"
  [ "$RC" -eq 0 ] || fail blend "$RC"
fi

if wanted bundle; then
  BUNDLE_ARGS=(--run-id "$RUN_ID" --out "$PKL")
  [ "$SELECT" -eq 1 ] && BUNDLE_ARGS+=(--require-frozen)
  run bundle "$PY" -m stealth.bundle "${BUNDLE_ARGS[@]}"
  [ "$RC" -eq 0 ] || fail bundle "$RC"
fi

if wanted pkl_check; then
  CHECK_ARGS=("$PKL")
  [ -n "$PKLCHECK_PY" ] && CHECK_ARGS+=(--venv-python "$PKLCHECK_PY")
  run pkl_check "$PY" -m stealth.pkl_check "${CHECK_ARGS[@]}"
  [ "$RC" -eq 0 ] || fail pkl_check "$RC"
fi

if wanted diag; then
  if [ "$NO_DIAG" -eq 1 ]; then
    log "step diag: skipped (--no-diag)"
  else
    DIAG_ARGS=(--pkl "$PKL" --out "$DIAG_OUT")
    [ -n "$DIAG_ERAS" ] && DIAG_ARGS+=(--eras "$DIAG_ERAS")
    run diag "$PY" -m stealth.diag "${DIAG_ARGS[@]}"
    [ "$RC" -eq 0 ] || fail diag "$RC"
  fi
fi

if wanted report && [ -n "$REPORT_NN" ]; then
  run report "$PY" -m stealth.report --run-id "$RUN_ID" --handoff "$REPORT_NN" --no-git
  [ "$RC" -eq 0 ] || fail report "$RC"
fi

if [ "$DRY_RUN" -eq 1 ]; then
  log "DRY RUN done: nothing was executed (mlp: $MLP_STATUS)"
else
  log "PIPELINE OK run_id=$RUN_ID pkl=$PKL mlp: $MLP_STATUS"
fi
