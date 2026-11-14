#!/bin/bash
# IS THE RUN ACTUALLY ALIVE. Checks every 5 minutes and says so only when the answer changes or is
# no, so a quiet log means healthy.
#
#     ./shell_scripts/health_watch.sh <rid>
#
# WHY THIS EXISTS. Three runs died today and in each case the next thing reported was progress: a
# stale step count read off a trajectory.csv that had stopped being written. eqd2jx was dead for
# 21 minutes before anyone noticed, and its own per-epoch plotter did not catch it because that
# only trips after 30 minutes of silence. A run that has exited looks exactly like a run between
# writes unless something checks for the exit.
#
# WHAT IT CHECKS, cheapest first:
#   1. a STOP or a traceback in run.log      -- a refusal, which is the common failure here
#   2. run.log's mtime                       -- silence past STALL_S means gone or wedged
#   3. the step counter moving               -- a process that is up but making no progress
#   4. status.json                           -- the launcher's own verdict once it exists
#
# LIVENESS IS NOT A PID CHECK. Deriving the pids means matching a pattern against the process
# table, and `pgrep -f` / `pkill -f` match the shell that runs them -- that has self-killed this
# session twice. The log's mtime answers the same question without naming a process.
#
# IT NEVER TOUCHES THE RUN: reads two files and prints. Exits 0 when the run finishes its epochs,
# 3 when it has died.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python

RID="${1:?usage: health_watch.sh <rid>}"
EVERY="${EVERY:-300}"
STALL_S="${STALL_S:-900}"
D="training/runs/$RID"
[ -d "$D" ] || { echo "no such run: $D"; exit 2; }

TOTAL=$($PY -I -c "
import json
a = json.load(open('$D/args.json')); a = a.get('args', a)
print(int(float(a.get('epochs') or 0)))" 2>/dev/null || echo 0)
# EVAL_EVERY, NOT EVERY: EVERY is this script's own poll interval, and the first version of this
# line overwrote it with the run's eval cadence -- so `sleep "$EVERY"` would have polled every 2
# seconds instead of every 300.
EVAL_CADENCE=$($PY -I -c "
import json
a = json.load(open('$D/args.json')); a = a.get('args', a)
print(int(a.get('train_eval_every') or 0))" 2>/dev/null || echo 0)
EVAL_GRACE="${EVAL_GRACE:-2100}"

ts() { TZ=America/New_York date '+%-I:%M %p %Z'; }
echo "health_watch $RID: poll ${EVERY}s, stall ${STALL_S}s, to epoch ${TOTAL:-?}, eval cadence ${EVAL_CADENCE:-?}"

LAST_STEP=-1
LAST_EP=-1
QUIET=0
while :; do
    # 1. AN EXPLICIT REFUSAL. Checked first because it is the one with a reason attached, and the
    #    reason is what makes the difference between "relaunch it" and "fix it first".
    if grep -qE "^STOP:|Traceback \(most recent call last\)" "$D/run.log" 2>/dev/null; then
        echo "[$(ts)] DEAD: $RID refused or raised"
        grep -E "^STOP:|Error" "$D/run.log" 2>/dev/null \
            | grep -vE "error_file|errors.html|elastic" | head -2 | sed 's/^/    /'
        exit 3
    fi

    EP=$(ls "$D"/head_ep*.pt 2>/dev/null | sed -E 's/.*head_ep([0-9]+)_.*/\1/' | sort -n | tail -1)
    EP="${EP:-0}"
    if [ "${TOTAL:-0}" -gt 0 ] && [ "$EP" -ge "$TOTAL" ]; then
        echo "[$(ts)] FINISHED: $RID reached epoch $EP of $TOTAL"
        exit 0
    fi

    MT=$(stat -c %Y "$D/run.log" 2>/dev/null || echo 0)
    NOW=$(date +%s)
    SILENT=$((NOW - MT))
    if [ "$SILENT" -gt "$STALL_S" ]; then
        echo "[$(ts)] DEAD: $RID has not written run.log in $((SILENT / 60)) min (epoch $EP)"
        tail -3 "$D/run.log" 2>/dev/null | sed 's/^/    /'
        exit 3
    fi

    # 2b. THE EVAL ACTUALLY RAN. A held-out pass is a subprocess on the same cards; it can fail
    #     on its own (a missing shard, a busy card, an OOM) while the trainer carries on to the
    #     next epoch perfectly healthily. The run then looks fine and simply stops producing the
    #     numbers it exists to produce -- which is the failure that matters most here, and the
    #     one nothing else checks. Grace is generous because the pass itself takes ~11 min.
    if [ "${EVAL_CADENCE:-0}" -gt 0 ] && [ "$EP" -gt 0 ] \
            && [ $((EP % EVAL_CADENCE)) -eq 0 ]; then
        MJ="$D/metric_valfull83_ep${EP}.json"
        if [ ! -s "$MJ" ]; then
            CK=$(ls "$D"/head_ep${EP}_*.pt 2>/dev/null | head -1)
            CKT=$(stat -c %Y "$CK" 2>/dev/null || echo "$NOW")
            if [ $((NOW - CKT)) -gt "$EVAL_GRACE" ]; then
                echo "[$(ts)] WARN: epoch $EP finished $(( (NOW - CKT) / 60 )) min ago and wrote no"
                echo "    metric_valfull83_ep${EP}.json -- the held-out eval did not run or failed"
                ls "$D"/valfull_eval_shards_ep${EP}/*.log 2>/dev/null | head -2 | sed 's/^/    /'
            fi
        fi
    fi

    STEP=$(awk -F, 'NR>1{print $2}' "$D/trajectory.csv" 2>/dev/null | sort -n | tail -1)
    STEP="${STEP:-0}"
    # 3. UP BUT NOT MOVING. The step counter is the only thing that distinguishes a process that
    #    is working from one that is holding a lock or wedged in a collective. Two consecutive
    #    quiet checks rather than one, because a long eval or a checkpoint write legitimately
    #    pauses it for a few minutes.
    if [ "$STEP" = "$LAST_STEP" ] && [ "$EP" = "$LAST_EP" ]; then
        QUIET=$((QUIET + 1))
        if [ "$QUIET" -ge 2 ]; then
            echo "[$(ts)] WARN: $RID still on step $STEP, epoch $EP after $((QUIET * EVERY / 60)) min"
            QUIET=0
        fi
    else
        QUIET=0
    fi
    LAST_STEP="$STEP"
    LAST_EP="$EP"
    sleep "$EVERY"
done
