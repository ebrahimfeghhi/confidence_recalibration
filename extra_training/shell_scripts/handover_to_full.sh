#!/bin/bash
# WAIT FOR xmz9ge's EPOCH 15, STOP IT, DERIVE LAMBDA ON THE FULL FOLD, LAUNCH THE FULL-FOLD ARM.
#
# WHY 15. xmz9ge was launched for 50 epochs, but hronsr, bjwcm5 and 6v6bbf all ran 15, so
# stopping here makes it directly comparable to the three arms it is meant to be read against.
# Epochs 16-50 simply do not run; nothing is deleted and resume_ep15.pt remains.
#
# WHY THIS IS ONE SCRIPT. Doing it by hand cost 23 minutes of idle cards earlier today, between
# an arm ending and the next being launched. Every step here -- stop, probe, derive, substitute,
# launch, watch -- runs back to back with no prompt in between.
#
# PIDS ARE EXPLICIT AND RE-VERIFIED, never matched by pattern. Each is checked to still be a
# phase_k_train whose --out is xmz9ge's directory immediately before it is signalled; if a pid has
# been recycled onto something else the script refuses rather than killing it. pkill/pgrep are
# never used -- they match the shell running them and have self-killed this session before.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RID=xmz9ge
PIDS="2580389 2580390 2580391"
TARGET=15
LOG=training/runs/handover_to_full.log
exec >>"$LOG" 2>&1
echo "=== handover armed $(date '+%F %T %Z'), waiting for $RID epoch $TARGET ==="

for _ in $(seq 100000); do
    [ -f training/runs/CANCEL_HANDOVER ] && { echo "CANCELLED"; exit 0; }
    EP=$(cat "training/runs/${RID}_last_plotted" 2>/dev/null || echo 0)
    [ "${EP:-0}" -ge "$TARGET" ] 2>/dev/null && break
    [ -f "training/runs/$RID/status.json" ] && { echo "$RID ended on its own at epoch $EP"; break; }
    sleep 30
done
echo "$(date '+%T') $RID reached epoch $(cat training/runs/${RID}_last_plotted 2>/dev/null)"

# --- stop it, by verified pid ---
for P in $PIDS; do
    [ -d "/proc/$P" ] || { echo "pid $P already gone"; continue; }
    OUT=$(tr '\0' '\n' < "/proc/$P/cmdline" 2>/dev/null | grep -A1 -x -- '--out' | tail -1)
    case "$OUT" in
        */training/runs/$RID) kill -TERM "$P" && echo "SIGTERM -> $P ($RID)";;
        *) echo "REFUSING to signal $P: --out is '${OUT:-unknown}', not $RID. Stopping."; exit 1;;
    esac
done
for _ in $(seq 60); do
    [ "$($PY -I run_logging/cards_busy.py)" = "0" ] && break
    sleep 10
done
echo "$(date '+%T') cards free: $($PY -I run_logging/cards_busy.py | tail -1)"

# --- derive lambda ON THE FULL FOLD; the overfit30 value does not carry ---
echo "$(date '+%T') probing |g_rank| on the full fold"
./shell_scripts/probe_pt3_full.sh > training/runs/probe_pt3_full.log 2>&1
PR=$(ls -t training/runs | while read -r n; do [ -d "training/runs/$n" ] && echo "$n"; done \
     | grep -E '^[a-z0-9]{6}$' | grep -vx latest | head -1)
G=$(grep -oE '\|g_rank\|[[:space:]]+[0-9.]+e?-?[0-9]*' "training/runs/$PR/run.log" 2>/dev/null \
    | head -1 | awk '{print $NF}')
[ -n "$G" ] || { echo "STOP: no |g_rank| in probe $PR; not launching on a guessed lambda."; exit 1; }
echo "$(date '+%T') probe $PR gave |g_rank| = $G"
LAM=$($PY -I run_logging/l2sp_lambda.py --old-lambda 1e-05 --old-grad 2.4260e-03 \
        --old-params 40960 --new-grad "$G" --new-params 827904 \
      | grep -oE 'nearest value worth using: [0-9.e-]+' | awk '{print $NF}')
[ -n "$LAM" ] || { echo "STOP: l2sp_lambda.py produced no lambda."; exit 1; }
echo "$(date '+%T') derived lambda = $LAM"

sed -i "s/LAMBDA_PLACEHOLDER/$LAM/" shell_scripts/launch_pt3_full.sh
grep -q LAMBDA_PLACEHOLDER shell_scripts/launch_pt3_full.sh && { echo "STOP: placeholder survived"; exit 1; }

# --- launch, and identify the run by SET DIFFERENCE, never by `ls -t` ---
BEFORE=$(ls training/runs | while read -r n; do [ -d "training/runs/$n" ] && echo "$n"; done \
         | grep -E '^[a-z0-9]{6}$' | grep -vx latest | sort)
./shell_scripts/launch_pt3_full.sh > training/runs/pt3_full_launch.log 2>&1 &
LAUNCHER=$!
NEW=""
for _ in $(seq 600); do
    NEW=$(comm -13 <(echo "$BEFORE") <(ls training/runs | while read -r n; do [ -d "training/runs/$n" ] && echo "$n"; done | grep -E '^[a-z0-9]{6}$' | grep -vx latest | sort) | head -1)
    [ -n "$NEW" ] && break
    kill -0 "$LAUNCHER" 2>/dev/null || break
    sleep 2
done
if [ -n "$NEW" ]; then
    echo "$NEW" > training/runs/pt3_full_rid
    echo "$(date '+%T') full-fold arm is $NEW (lambda $LAM)"
    nohup ./shell_scripts/watch_arm.sh "$NEW" >> "training/runs/watch_$NEW.log" 2>&1 &
else
    echo "$(date '+%T') no new run directory appeared; check training/runs/pt3_full_launch.log"
fi
wait "$LAUNCHER"
echo "$(date '+%T') full-fold launcher returned $?"
