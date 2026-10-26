#!/bin/bash
# EVERY 30 MINUTES: is the box healthy, is the active run advancing, and did the queue hand off?
#
# Usage:  nohup ./shell_scripts/watch_health.sh > training/runs/health.log 2>&1 &
#
# READ-ONLY BY CONSTRUCTION. No kill, no launch, no git. Its entire output is a log line and, on
# a problem, one file -- training/runs/HEALTH_ALERT -- which the session blocks on, so a failure
# at 3am reaches a person instead of sitting in a log nobody is reading.
#
# WHAT "ADVANCING" MEANS: the newest trajectory.csv mtime under the active run moved since the
# last tick. A run that is alive but wedged -- the prefetch deadlock this project has already hit
# once -- keeps its processes and its cards and stops writing, so liveness alone would miss it.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
HOLDERS="306042 306045 306050"
ALERT=training/runs/HEALTH_ALERT
PREV_MTIME=0
PREV_RID=""
GONE_SINCE=0

say() { echo "$(TZ=America/New_York date '+%F %I:%M %p %Z') $*"; }
alert() { say "ALERT: $*"; { say "ALERT: $*"; } >> "$ALERT"; }

say "health watcher armed; 30 min cadence"
while :; do
  # --- the active run: the newest rid that has live ranks on the cards ---
  PIDS=$($PY -I run_logging/cards_busy.py 2>&1 | grep -oE 'pid [0-9]+' | awk '{print $2}' | sort -u)
  RID=""
  for P in $PIDS; do
    [ -d "/proc/$P" ] || continue
    C=$(tr '\0' '\n' < "/proc/$P/cmdline" 2>/dev/null | tr '\n' ' ')
    case "$C" in
      *phase_k_train*)
        # FROM --out, NOT THE FIRST PATH THAT LOOKS LIKE A RID. A resumed run carries
        # "--resume training/runs/<parent>/resume_epN.pt" EARLIER on its command line than its own
        # --out, so the first match names the run it continues rather than the run that is
        # executing -- which is how the first tick reported ip8ga4 as "uhyypy", and would have
        # watched a directory nothing is writing to.
        OUT=$(printf '%s' "$C" | tr ' ' '\n' | grep -A1 -x -- '--out' | tail -1 | sed 's#/*$##')
        # A SHARD IS NOT A RUN. The sharded held-out PAE pass spawns phase_k_train children whose
        # --out is training/runs/<rid>/val_loss_ep<N>/shard<i>, so a naive basename called the
        # active run "shard0" -- and since training/runs/shard0/trajectory.csv does not exist,
        # two consecutive ticks would both read mtime 0 and fire a FALSE wedge alert at whatever
        # hour the PAE pass happened to straddle. Require the out dir to sit DIRECTLY under
        # training/runs with a six-character rid; anything deeper is a child of a run, not a run.
        case "$OUT" in
          */training/runs/??????|training/runs/??????) RID=$(basename "$OUT");;
          *) RID="";;
        esac
        ;;
    esac
    [ -n "$RID" ] && break
  done

  if [ -n "$RID" ]; then
    GONE_SINCE=0
    M=$(stat -c %Y "training/runs/$RID/trajectory.csv" 2>/dev/null || echo 0)
    EP=$(grep -cE "^  epoch [0-9]+ TRAIN:" "training/runs/$RID/run.log" 2>/dev/null); EP=${EP:-0}
    if [ "$RID" = "$PREV_RID" ] && [ "$M" = "$PREV_MTIME" ]; then
      alert "$RID has not written trajectory.csv in 30 min -- wedged, not merely slow. Cards still held."
    else
      say "$RID advancing, $EP evaluated epoch(s)"
    fi
    PREV_MTIME=$M; PREV_RID=$RID
  else
    # --- NO RUN ON THE CARDS. That is the handoff window, and it is allowed to be brief. ---
    NOW=$(date +%s)
    [ "$GONE_SINCE" = "0" ] && GONE_SINCE=$NOW
    IDLE=$(( (NOW - GONE_SINCE) / 60 ))
    # THE CHAIN'S PID, RECORDED AT ARM TIME, NEVER MATCHED BY PATTERN. pgrep -f matches the
    # shell running it, which is how this project self-killed a session twice; the habit is worth
    # keeping even where nothing is being killed.
    CHAIN_PID=$(cat training/runs/copy_chain_pid 2>/dev/null || echo 0)
    CHAIN_UP=0; [ "$CHAIN_PID" != "0" ] && [ -d "/proc/$CHAIN_PID" ] && CHAIN_UP=1
    if [ "$IDLE" -ge 40 ]; then
      alert "no training process has held the cards for ${IDLE} min. chain_alive=$CHAIN_UP. The handoff did not happen; cards are idle."
    else
      say "no run on the cards (${IDLE} min); chain_alive=$CHAIN_UP -- handoff window"
    fi
  fi

  # --- the standing box constraints ---
  for P in $HOLDERS; do [ -d "/proc/$P" ] || alert "GPU holder $P is GONE"; done
  MEM=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
  [ "$MEM" -lt 250 ] && alert "MemAvailable ${MEM} GB, under the 250 GB floor other users need"
  USEP=$(df --output=pcent /home | tail -1 | tr -dc '0-9')
  [ "$USEP" -ge 90 ] && alert "/home is ${USEP}% full"

  sleep 1800
done
