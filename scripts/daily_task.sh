#!/usr/bin/env bash
# stealth daily fallback (cron-friendly). The platform's auto-run lane is the PRIMARY path; this script only
# steps in when no accepted submission exists for today's round.
#
# Logic (all clocks in UTC regardless of the local timezone the scheduler fires in):
#   0. weekday only; UTC time must be inside 13:05-15:55 (the round is open 13:00-16:00 UTC); else exit 0 quietly
#   1. $PY -m stealth.monitor          -> exit 0 = accepted submission for today's exped AND lane sha == manifest: done
#                                         exit 2 = not configured (no EIQ_MODEL_ID / manifest): log + exit 0 (nothing armed yet)
#   2. otherwise, if STEALTH_DAILY_APPROVED=1 is set in the environment (the user's one-time approval, placed in the
#      crontab), run  $PY -m stealth.predict --pkl <active pkl> --submit --i-have-approval  (predict itself re-checks
#      the round is open and the live exped matches; exit 3 = window closed)
#   3. re-run monitor; if it still fails after 15:25 UTC write an ALERT line (monitor writes handoffs/ALERT.md itself;
#      whether the ALERT is committed/pushed automatically is decided in handoff 05 -- today it is only written)
# Logs: artifacts/daily/YYYY-MM-DD.log (gitignored). Never prints secrets; .env is sourced, never echoed.
#
# ---- crontab (Europe/London LOCAL time; `crontab -e`) -------------------------------------------------------
#   # stealth daily fallback. Target 14:30 and 15:30 UTC.
#   #   GMT (late Oct -> late Mar): 14:30 local == 14:30 UTC, 15:30 local == 15:30 UTC
#   #   BST (late Mar -> late Oct): 15:30 local == 14:30 UTC, 16:30 local == 15:30 UTC
#   # Firing at 14:30, 15:30 AND 16:30 local gives >= two runs inside the 13:05-15:55 UTC window in BOTH regimes
#   # (GMT: 14:30Z, 15:30Z, [16:30Z skipped by the UTC gate]; BST: 13:30Z, 14:30Z, 15:30Z).
#   STEALTH_DAILY_APPROVED=1
#   30 14,15,16 * * 1-5 /home/USER/stealth/scripts/daily_task.sh
#
#   Alternative when the cron daemon supports CRON_TZ (cronie; NOT Debian's vixie cron):
#   CRON_TZ=UTC
#   30 14,15 * * 1-5 /home/USER/stealth/scripts/daily_task.sh
# Replace /home/USER/stealth with the absolute clone path. DST note: the UK switches on the last Sunday of March
# and the last Sunday of October; nothing needs editing at the switch because the script gates on UTC itself.
# -----------------------------------------------------------------------------------------------------------
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
LOG_DIR="$REPO/artifacts/daily"
mkdir -p "$LOG_DIR"
TODAY_UTC="$(date -u +%F)"
LOG="$LOG_DIR/$TODAY_UTC.log"
LOCK="$LOG_DIR/.lock"

log() { printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$LOG"; }

# ---- 0. gates --------------------------------------------------------------------------------------
dow="$(date -u +%u)"                      # 1..7, Mon=1
hhmm="$(date -u +%H%M)"
if [ "$dow" -gt 5 ]; then exit 0; fi      # weekend: dark
if [ "$hhmm" -lt 1305 ] || [ "$hhmm" -gt 1555 ]; then
  log "outside 13:05-15:55 UTC (now ${hhmm} UTC); nothing to do"
  exit 0
fi
if [ ! -x "$PY" ]; then log "ERROR: $PY missing; run scripts/setup.sh"; exit 1; fi

exec 9>"$LOCK"
if ! flock -n 9; then log "another daily_task run holds the lock; exiting"; exit 0; fi

cd "$REPO" || exit 1
if [ -s "$REPO/.env" ]; then
  set -a; . "$REPO/.env"; set +a      # export EIQ_API_KEY / EIQ_BASE_URL / EIQ_MODEL_ID for the python steps
fi
if [ -z "${EIQ_API_KEY:-}" ]; then log "ERROR: EIQ_API_KEY not set (.env missing?)"; exit 1; fi

# active pickle: models/active.json {"pkl": ...} else models/active.pkl
PKL="$("$PY" - <<'EOF' 2>/dev/null
import json, pathlib
p = pathlib.Path("models/active.json")
if p.exists():
    v = json.loads(p.read_text()).get("pkl")
    if v: print(v)
EOF
)"
PKL="${PKL:-models/active.pkl}"
case "$PKL" in /*) : ;; *) PKL="$REPO/$PKL" ;; esac

log "start (UTC ${hhmm}); pkl=$PKL approved=${STEALTH_DAILY_APPROVED:-0}"

# ---- 1. is today already covered? --------------------------------------------------------------------
"$PY" -m stealth.monitor >> "$LOG" 2>&1
mrc=$?
if [ "$mrc" -eq 0 ]; then log "monitor OK: accepted submission exists and lane sha matches; done"; exit 0; fi
if [ "$mrc" -eq 2 ]; then
  log "monitor exit 2: not configured yet (EIQ_MODEL_ID empty or no manifest/active.json) -> nothing to do until handoff 05"
  exit 0
fi
log "monitor reported a gap/mismatch (exit $mrc)"

# ---- 2. fallback submit (user-gated) ------------------------------------------------------------------
if [ "${STEALTH_DAILY_APPROVED:-0}" != "1" ]; then
  log "STEALTH_DAILY_APPROVED!=1: not submitting (dry run only). Set it in the crontab to arm the fallback."
  "$PY" -m stealth.predict --pkl "$PKL" >> "$LOG" 2>&1
  log "dry-run predict exit $?"
  exit 1
fi
if [ ! -s "$PKL" ]; then log "ERROR: active pickle $PKL missing"; exit 1; fi

STEALTH_APPROVED=1 "$PY" -m stealth.predict --pkl "$PKL" --submit --i-have-approval >> "$LOG" 2>&1
rc=$?
case "$rc" in
  0) log "predict --submit exit 0 (submitted/replaced)";;
  3) log "predict --submit exit 3: submission window closed; nothing submitted"; exit 0;;
  *) log "predict --submit exit $rc (error)";;
esac

# ---- 3. verify ------------------------------------------------------------------------------------------
"$PY" -m stealth.monitor >> "$LOG" 2>&1
mrc=$?
if [ $mrc -eq 0 ]; then log "monitor OK after fallback submit"; exit 0; fi
if [ "$hhmm" -ge 1525 ]; then
  log "ALERT: still no accepted submission for today after the fallback (monitor exit $mrc); see handoffs/ALERT.md"
else
  log "monitor still failing (exit $mrc); the later run will retry"
fi
exit 1
