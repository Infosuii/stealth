#!/usr/bin/env bash
# stealth -- end-to-end pipeline driver (docs/INTERFACES.md sections 20, 22). Idempotent and resumable: after ANY
# interruption re-run the SAME command; every step resumes from its cache or repeats deterministically.
#
#   bash scripts/run_pipeline.sh <config> <run_id> [--gpu-py PATH] [--no-mlp] [--pkl models/<run_id>.pkl]
#        [--py PATH] [--from STEP] [--mlp-device cuda|cpu|auto] [--mlp-epochs-cap N] [--mlp-seeds "0 1"]
#        [--mlp-threads N] [--pkl-check-python PATH] [--diag-eras N] [--no-diag] [--report NN] [--dry-run]
#
# Steps (STEP names for --from):
#   train      $PY -m stealth.train --config C --run-id R --resume --skip-external   (external members never loaded)
#   mlp        $GPU_PY scripts/mlp_train.py --run-id R --resume --device D   only when the config has a `family: mlp`
#              member, --gpu-py is given and --no-mlp is not. D defaults to cuda (no silent CPU fallback on the
#              laptop; the sandbox passes --mlp-device cpu). Exit 1 (error) or 4 (no usable device) is logged
#              "MLP skipped" and the pipeline continues without the member; a signal exit (>= 128) or any other
#              code stops the pipeline.
#   retrain    after mlp exited 0: $PY -m stealth.train ... --resume --require-external (assemble WITH the member;
#              a member that fails its provenance check is an error). When the MLP did not finish in this attempt
#              and train did not run in it either (--from mlp | retrain): --resume --skip-external, so a previously
#              assembled MLP can never survive. To include an MLP trained earlier use --from mlp (mlp_train --resume
#              returns at once when it is already done).
#   blend      $PY -m stealth.blend --run-id R                       (the regime grid can be silent for a while)
#   audit      $PY -m stealth.audit --run-id R                       only when cv.audit.enabled; frozen blend only
#   bundle     $PY -m stealth.bundle --run-id R --out PKL [--require-frozen when blend.neutralisation.select]
#   pkl_check  $PY -m stealth.pkl_check PKL [--venv-python P]
#   diag       $PY -m stealth.diag --pkl PKL --out artifacts/diag/<R>_validation.parquet [--eras N]   FILE ONLY:
#              this script never passes --upload / --submit / any platform write
#   report     $PY -m stealth.report --run-id R --handoff NN --no-git   only with --report NN. The compute steps'
#              final state is recorded BEFORE it runs, so the report sees a finished pipeline; its own outcome is
#              recorded separately (pipeline.json report_step) and a report failure still exits non-zero.
#
# --no-mlp with --from blend (or later) refuses (exit 64) when oof.parquet still holds an external member: re-run
# with --from retrain --no-mlp first.
# Defaults: PY=<repo>/.venv/bin/python, PKL=models/<run_id>.pkl. Relative paths resolve against the repo root.
# Step lines go to stdout and artifacts/<run_id>/pipeline.log. artifacts/<run_id>/pipeline.json (atomic, schema
# stealth.pipeline/1) records every attempt, its steps, exit codes and seconds; pipeline_steps/ holds each step's
# captured stdout + stderr. Every step runs in its own process group: Ctrl-C / TERM stop that whole group (and
# nothing else), the attempt is recorded as interrupted, and re-running the same command resumes.
# Exit: 0 when every step succeeded (an MLP skip is tolerated), else the exit code of the first failing step
# (train / blend 2 = leak or gate failure; 130 / 143 = interrupted). --dry-run prints the commands and runs nothing.
# Never reads .env.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEPS=(train mlp retrain blend audit bundle pkl_check diag report)

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
MLP_DEVICE="cuda"
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
    --mlp-device)
      MLP_DEVICE="${2:?--mlp-device needs cuda|cpu|auto}"; shift 2
      case "$MLP_DEVICE" in cuda|cpu|auto) ;; *) die "--mlp-device must be cuda, cpu or auto (got '$MLP_DEVICE')" ;; esac ;;
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

step_index() { local i; for i in "${!STEPS[@]}"; do [ "${STEPS[$i]}" = "$1" ] && { echo "$i"; return; }; done; echo -1; }
RETRAIN_IDX="$(step_index retrain)"

cd "$REPO"
RUN_DIR="$REPO/artifacts/$RUN_ID"
PLOG="$RUN_DIR/pipeline.log"
PJSON="$RUN_DIR/pipeline.json"
DIAG_OUT="$REPO/artifacts/diag/${RUN_ID}_validation.parquet"

# config facts (yaml read by the train venv's python): has a family: mlp member (and their names)? neutralisation
# selection on? audit partition on?
read -r HAS_MLP SELECT HAS_AUDIT EXTERNALS < <("$PY" - "$CONFIG" <<'PYEOF'
import sys
import yaml
with open(sys.argv[1]) as fh:
    c = yaml.safe_load(fh) or {}
ext = [str(m.get("name")) for m in c.get("members") or [] if isinstance(m, dict) and m.get("family") == "mlp"]
sel = bool(((c.get("blend") or {}).get("neutralisation") or {}).get("select"))
audit = bool(((c.get("cv") or {}).get("audit") or {}).get("enabled"))
print(int(bool(ext)), int(sel), int(audit), ",".join(ext) or "-")
PYEOF
) || true
[[ "${HAS_MLP:-}" =~ ^[01]$ && "${SELECT:-}" =~ ^[01]$ && "${HAS_AUDIT:-}" =~ ^[01]$ && -n "${EXTERNALS:-}" ]] \
  || die "could not read $CONFIG with $PY (yaml error?)"
[ "$EXTERNALS" != "-" ] || EXTERNALS=""

# --no-mlp from blend onwards must not blend, audit or bundle an external member that an earlier attempt assembled
if [ "$NO_MLP" -eq 1 ] && [ -n "$EXTERNALS" ] && [ "$FROM_IDX" -gt "$RETRAIN_IDX" ] && [ -f "$RUN_DIR/oof.parquet" ]; then
  STALE="$("$PY" - "$RUN_DIR/oof.parquet" "$EXTERNALS" <<'PYEOF'
import sys
import pyarrow.parquet as pq
cols = set(pq.read_schema(sys.argv[1]).names)
print(" ".join(n for n in sys.argv[2].split(",") if n and f"m__{n}" in cols))
PYEOF
)" || die "could not read the schema of $RUN_DIR/oof.parquet with $PY"
  [ -z "$STALE" ] || die "--no-mlp --from $FROM: $RUN_DIR/oof.parquet still contains external member(s) $STALE from an earlier attempt; re-run with --from retrain --no-mlp so it is re-assembled without them"
fi

# A real step gets its own process group so Ctrl-C / TERM also stops its subprocesses. Linux provides setsid.
[ "$DRY_RUN" -eq 1 ] || command -v setsid >/dev/null || die "setsid is required to stop pipeline children safely"

log() {
  local line
  line="$(date -u +%Y-%m-%dT%H:%M:%SZ) pipeline[$RUN_ID]: $*"
  echo "$line"
  if [ "$DRY_RUN" -eq 0 ]; then mkdir -p "$RUN_DIR"; echo "$line" >> "$PLOG"; fi
}

state() {  # state <event> <fields...>; atomic (tmp + rename), local-only attempt history for the reporter
  [ "$DRY_RUN" -eq 0 ] || return 0
  "$PY" - "$PJSON" "$@" <<'PYEOF'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

p, event, *args = sys.argv[1:]
p = Path(p)
p.parent.mkdir(parents=True, exist_ok=True)
try:
    doc = json.loads(p.read_text())
except (FileNotFoundError, json.JSONDecodeError):
    doc = {"schema": "stealth.pipeline/1", "attempts": []}
now = datetime.now(timezone.utc).isoformat(timespec="seconds")
att = doc["attempts"][-1] if doc["attempts"] else None


def status_of(rc):
    return "ok" if rc == 0 else "interrupted" if rc >= 128 else "failed"


def step_entry(name):
    if name == "report":  # recorded apart from the compute steps: the report must observe a FINISHED pipeline
        return att.get("report_step")
    return next(s for s in reversed(att["steps"]) if s["name"] == name)


if event == "start":
    run_id, config, pkl, from_step, *opts = args
    for old in doc["attempts"]:
        if old.get("status") == "running":  # an earlier attempt that never recorded its end (killed / power loss)
            old["status"] = "interrupted"
            old.setdefault("note", "no end record: the driver was killed before it could record one")
    options = dict(o.split("=", 1) for o in opts if "=" in o)
    doc.update(run_id=run_id, config=Path(config).name, pkl=pkl, status="running", finished_at=None, exit_code=None,
               failed_step=None, report_step=None)
    doc["attempts"].append({"started_at": now, "from_step": from_step, "status": "running", "steps": [],
                            "options": options})
elif event == "step-start":
    name, logfile = args
    entry = {"name": name, "status": "running", "started_at": now, "log_file": logfile}
    if name == "report":
        att["report_step"] = doc["report_step"] = entry
    else:
        att["steps"].append(entry)
        doc["last_step"] = name
elif event == "step-end":
    name, rc, seconds = args
    rc = int(rc)
    step = step_entry(name)
    step.update(status=status_of(rc), exit_code=rc, seconds=int(seconds), finished_at=now)
    if name == "report":
        doc["report_step"] = step
elif event == "step-mark":  # e.g. a tolerated MLP exit: the step did not fail the pipeline
    name, status, note = args
    step_entry(name).update(status=status, note=note)
elif event == "mlp":
    att["mlp"] = doc["mlp"] = args[0]
elif event == "compute-done":  # every compute step finished: the pipeline result, recorded BEFORE the report step
    seconds = int(args[0])
    result = {"status": "ok", "exit_code": 0, "finished_at": now}
    doc.update(result, failed_step=None)
    att.update(result, seconds=seconds)
elif event == "finish":
    status, rc, seconds, step = args
    result = {"status": status, "exit_code": int(rc), "finished_at": now}
    doc.update(result, failed_step=step or None)
    att.update(result, seconds=int(seconds), failed_step=step or None)
elif event == "report-finish":  # the report step ended after compute-done: the pipeline result itself stays
    status, rc = args
    rep = att.get("report_step") or {"name": "report"}
    if rep.get("status") == "running":
        rep.update(status=status, exit_code=int(rc), finished_at=now)
    att["report_step"] = doc["report_step"] = rep
else:
    raise SystemExit(f"unknown pipeline state event {event!r}")
tmp = p.with_name(p.name + ".tmp")
tmp.write_text(json.dumps(doc, indent=2) + "\n")
os.replace(tmp, p)
PYEOF
}

RC=0
STEP_PID=""
TEE_PID=""
CURRENT_STEP=""
STEP_T0=0
PIPE_T0=$SECONDS
PIPELINE_FINISHED=0
COMPUTE_DONE=0

finish() {  # finish <status> <rc> [<step>]
  if [ "$COMPUTE_DONE" -eq 1 ]; then
    state report-finish "$1" "$2"
  else
    state finish "$1" "$2" "$((SECONDS - PIPE_T0))" "${3:-}"
  fi
  PIPELINE_FINISHED=1
}

stop_step() {  # TERM the running step's whole process group (never anything outside it), KILL after a grace period
  local pg="$STEP_PID" dog i
  [ -n "$pg" ] || return 0
  kill -TERM -- "-$pg" 2>/dev/null || true
  # watchdog without our stdout/stderr, so a leftover sleep never holds the caller's pipe open
  ( exec </dev/null >/dev/null 2>&1; sleep 15; kill -KILL -- "-$pg" ) &
  dog=$!
  wait "$pg" 2>/dev/null || true
  for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 -- "-$pg" 2>/dev/null || break; sleep 0.5; done
  kill -KILL -- "-$pg" 2>/dev/null || true
  kill "$dog" 2>/dev/null || true
  wait "$dog" 2>/dev/null || true
  STEP_PID=""
  if [ -n "$TEE_PID" ]; then wait "$TEE_PID" 2>/dev/null || true; TEE_PID=""; fi
}

unexpected_exit() {
  local rc=$?
  trap - EXIT INT TERM
  if [ "$DRY_RUN" -eq 0 ] && [ "$PIPELINE_FINISHED" -eq 0 ]; then
    set +e
    [ "$rc" -ne 0 ] || rc=1
    stop_step
    if [ -n "$CURRENT_STEP" ]; then state step-end "$CURRENT_STEP" "$rc" "$((SECONDS - STEP_T0))"; fi
    log "PIPELINE FAILED at step ${CURRENT_STEP:-setup} (exit $rc)"
    finish failed "$rc" "${CURRENT_STEP:-setup}"
  fi
  exit "$rc"
}

interrupted() {
  local signal="$1" rc="$2"
  trap - EXIT INT TERM
  set +e
  # Background commands inherit SIGINT ignored; TERM reliably stops their entire new process group.
  stop_step
  if [ -n "$CURRENT_STEP" ]; then state step-end "$CURRENT_STEP" "$rc" "$((SECONDS - STEP_T0))"; fi
  log "PIPELINE INTERRUPTED at step ${CURRENT_STEP:-setup} ($signal; exit $rc). Re-run the same command to resume."
  finish interrupted "$rc" "${CURRENT_STEP:-setup}"
  exit "$rc"
}

run() {  # run <step> <cmd...>: runs the command (or prints it with --dry-run); sets RC; never exits by itself
  local step="$1" step_log pipe_fd tee_rc
  shift
  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY-RUN [%s]:' "$step"; printf ' %q' "$@"; printf '\n'
    RC=0
    return 0
  fi
  CURRENT_STEP="$step"
  STEP_T0=$SECONDS
  mkdir -p "$RUN_DIR/pipeline_steps"
  step_log="pipeline_steps/$(date -u +%Y%m%dT%H%M%S)-$$-$step.log"
  state step-start "$step" "$step_log"
  log "step $step: start: $*"
  exec {pipe_fd}> >(tee -a "$RUN_DIR/$step_log")
  TEE_PID=$!
  setsid "$@" >&"$pipe_fd" 2>&1 &
  STEP_PID=$!
  exec {pipe_fd}>&-
  set +e
  wait "$STEP_PID"
  RC=$?
  STEP_PID=""
  wait "$TEE_PID"
  tee_rc=$?
  TEE_PID=""
  set -e
  [ "$RC" -ne 0 ] || RC=$tee_rc
  state step-end "$step" "$RC" "$((SECONDS - STEP_T0))"
  log "step $step: exit $RC after $((SECONDS - STEP_T0)) s"
  CURRENT_STEP=""
}

fail() {  # fail <step> <rc>
  if [ "$2" -ge 128 ]; then
    log "PIPELINE INTERRUPTED at step $1 (exit $2). Re-run the same command to resume (or --from $1)."
    finish interrupted "$2" "$1"
  elif [ "$COMPUTE_DONE" -eq 1 ]; then
    log "REPORT FAILED at step $1 (exit $2); the pipeline steps themselves finished ok. Re-run with --from $1."
    finish failed "$2" "$1"
  else
    log "PIPELINE FAILED at step $1 (exit $2). Follow the handoff's 'If it fails' section, then re-run the same command (or --from $1)."
    finish failed "$2" "$1"
  fi
  exit "$2"
}

wanted() {  # wanted <step>: is <step> at or after --from?
  local i
  for i in "${!STEPS[@]}"; do [ "${STEPS[$i]}" = "$1" ] && { [ "$i" -ge "$FROM_IDX" ]; return; }; done
  return 1
}

log "config=$CONFIG run_id=$RUN_ID pkl=$PKL py=$PY gpu_py=${GPU_PY:-none} from=$FROM has_mlp=$HAS_MLP select=$SELECT audit=$HAS_AUDIT dry_run=$DRY_RUN"
state start "$RUN_ID" "$CONFIG" "$PKL" "$FROM" "no_mlp=$NO_MLP" "mlp_device=$MLP_DEVICE" "gpu_py=${GPU_PY:+set}" \
  "report=$REPORT_NN" "diag_eras=$DIAG_ERAS" "no_diag=$NO_DIAG"
trap unexpected_exit EXIT
trap 'interrupted INT 130' INT
trap 'interrupted TERM 143' TERM

TRAIN_RAN=0
if wanted train; then
  run train "$PY" -m stealth.train --config "$CONFIG" --run-id "$RUN_ID" --resume --skip-external
  [ "$RC" -eq 0 ] || fail train "$RC"
  TRAIN_RAN=1
fi

MLP_STATUS="not run in this attempt (from=$FROM)"
[ "$HAS_MLP" -eq 1 ] || MLP_STATUS="n/a (no family: mlp member)"
MLP_OK=0
if wanted mlp && [ "$HAS_MLP" -eq 1 ]; then
  if [ "$NO_MLP" -eq 1 ]; then
    MLP_STATUS="skipped (--no-mlp)"
  elif [ -z "$GPU_PY" ]; then
    MLP_STATUS="skipped (no --gpu-py)"
  elif [ ! -x "$GPU_PY" ] && [ "$DRY_RUN" -eq 0 ]; then
    MLP_STATUS="skipped (--gpu-py not executable: $GPU_PY)"
  else
    run mlp "$GPU_PY" "$REPO/scripts/mlp_train.py" --run-id "$RUN_ID" --resume --device "$MLP_DEVICE" ${MLP_ARGS[@]+"${MLP_ARGS[@]}"}
    if [ "$DRY_RUN" -eq 1 ]; then
      MLP_OK=1
      MLP_STATUS="would run (retrain assembles it only if mlp_train exits 0)"
    elif [ "$RC" -eq 0 ]; then
      MLP_OK=1
      MLP_STATUS="ok"
    elif [ "$RC" -eq 1 ] || [ "$RC" -eq 4 ]; then
      MLP_STATUS="skipped (mlp_train exit $RC)"
      state step-mark mlp skipped "optional member: exit $RC tolerated, pipeline continues without it"
    else
      state mlp "failed (mlp_train exit $RC)"
      fail mlp "$RC"
    fi
  fi
  [ "$MLP_OK" -eq 1 ] || log "MLP skipped: $MLP_STATUS -- continuing without the mlp member"
  state mlp "$MLP_STATUS"
fi

if wanted retrain && [ "$HAS_MLP" -eq 1 ]; then
  if [ "$MLP_OK" -eq 1 ]; then
    run retrain "$PY" -m stealth.train --config "$CONFIG" --run-id "$RUN_ID" --resume --require-external
    [ "$RC" -eq 0 ] || fail retrain "$RC"
  elif [ "$TRAIN_RAN" -eq 0 ]; then
    log "step retrain: re-assembling WITHOUT external members (no MLP finished in this attempt)"
    run retrain "$PY" -m stealth.train --config "$CONFIG" --run-id "$RUN_ID" --resume --skip-external
    [ "$RC" -eq 0 ] || fail retrain "$RC"
  else
    log "step retrain: skipped (train already assembled without external members; no MLP finished in this attempt)"
  fi
elif wanted retrain && [ "$TRAIN_RAN" -eq 0 ]; then  # no external member at all: retrain == train --resume
  run retrain "$PY" -m stealth.train --config "$CONFIG" --run-id "$RUN_ID" --resume --skip-external
  [ "$RC" -eq 0 ] || fail retrain "$RC"
fi

if wanted blend; then
  run blend "$PY" -m stealth.blend --run-id "$RUN_ID"
  [ "$RC" -eq 0 ] || fail blend "$RC"
fi

if wanted audit && [ "$HAS_AUDIT" -eq 1 ]; then
  run audit "$PY" -m stealth.audit --run-id "$RUN_ID"
  [ "$RC" -eq 0 ] || fail audit "$RC"
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
  if [ "$DRY_RUN" -eq 0 ]; then
    state compute-done "$((SECONDS - PIPE_T0))"
    COMPUTE_DONE=1
    log "pipeline steps ok run_id=$RUN_ID (mlp: $MLP_STATUS); writing the report"
  fi
  run report "$PY" -m stealth.report --run-id "$RUN_ID" --handoff "$REPORT_NN" --no-git
  [ "$RC" -eq 0 ] || fail report "$RC"
fi

if [ "$DRY_RUN" -eq 1 ]; then
  log "DRY RUN done: nothing was executed (mlp: $MLP_STATUS)"
else
  log "PIPELINE OK run_id=$RUN_ID pkl=$PKL mlp: $MLP_STATUS"
  finish ok 0
fi
