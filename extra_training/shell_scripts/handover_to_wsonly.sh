#!/bin/bash
# WAIT FOR oe6sru's EPOCH 15, STOP IT, DECIDE WHETHER PAE HELPED, THEN PROBE AND LAUNCH THE
# WITHIN-SEED-ONLY ARM WITH OR WITHOUT PAE ACCORDINGLY.
#
# THE DECISION RULE IS FIXED HERE, BEFORE THE NUMBER EXISTS. PAE counts as having helped if
# oe6sru's mean paired (trained - shipped) DockQ over epochs 8-15 exceeds hronsr's +0.011556 --
# the best no-PAE arm on this pool and trainable family, computed by the same
# run_logging/paired_mean.py so the two sides cannot be computed differently. Writing the
# threshold down in advance is the point: a rule chosen after seeing the result is not a rule.
#
# WHAT THAT COMPARISON CANNOT SETTLE, recorded so the answer is not overstated. oe6sru differs
# from hronsr in THREE ways at once -- PAE on, pairs cross instead of all at L=8, and 830,976
# trainable parameters instead of 44,032. A win is a win for the package, not for PAE alone. The
# clean isolation would be cross-seed at the 17-tensor set with alpha_pae 0, which has not run.
#
# PIDS ARE EXPLICIT AND RE-VERIFIED immediately before signalling: each must still be a
# phase_k_train whose --out is oe6sru's directory, or the script refuses rather than killing
# something that inherited the pid. pkill/pgrep are never used.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RID=oe6sru
PIDS="2660554 2660555 2660556"
TARGET=15
HRONSR_BASELINE=0.011556
LOG=training/runs/handover_to_wsonly.log
exec >>"$LOG" 2>&1
echo "=== armed $(date '+%F %T %Z'); waiting for $RID epoch $TARGET ==="

for _ in $(seq 100000); do
    [ -f training/runs/CANCEL_WSONLY ] && { echo "CANCELLED"; exit 0; }
    EP=$(cat "training/runs/${RID}_last_plotted" 2>/dev/null || echo 0)
    [ "${EP:-0}" -ge "$TARGET" ] 2>/dev/null && break
    [ -f "training/runs/$RID/status.json" ] && { echo "$RID ended early at epoch $EP"; break; }
    sleep 30
done
echo "$(date '+%T') $RID at epoch $(cat training/runs/${RID}_last_plotted 2>/dev/null)"

for P in $PIDS; do
    [ -d "/proc/$P" ] || { echo "pid $P already gone"; continue; }
    OUT=$(tr '\0' '\n' < "/proc/$P/cmdline" 2>/dev/null | grep -A1 -x -- '--out' | tail -1)
    case "$OUT" in
        */training/runs/$RID) kill -TERM "$P" && echo "SIGTERM -> $P ($RID)";;
        *) echo "REFUSING to signal $P: --out is '${OUT:-unknown}'. Stopping."; exit 1;;
    esac
done
for _ in $(seq 60); do [ "$($PY -I run_logging/cards_busy.py)" = "0" ] && break; sleep 10; done
echo "$(date '+%T') cards free"

# --- the pre-registered decision ---
PM=$($PY -I run_logging/paired_mean.py "$RID" 8 15)
MEAN=$(echo "$PM" | awk '{print $2}')
echo "$(date '+%T') $RID epochs 8-15: $PM   (threshold: hronsr $HRONSR_BASELINE)"
USE_PAE=$($PY -I -c "print(1 if $MEAN > $HRONSR_BASELINE else 0)" 2>/dev/null || echo 0)
echo "$(date '+%T') PAE helped? $USE_PAE"

# --- probe at the within-seed objective; alpha_pae 1 so BOTH norms are reported either way ---
./shell_scripts/probe_wsonly.sh > training/runs/probe_wsonly.log 2>&1
PR=$(ls -t training/runs | while read -r n; do [ -d "training/runs/$n" ] && echo "$n"; done \
     | grep -E '^[a-z0-9]{6}$' | grep -vx latest | head -1)
GR=$(grep -oE '\|g_rank\|[[:space:]]+[0-9.]+e?-?[0-9]*' "training/runs/$PR/run.log" | head -1 | awk '{print $NF}')
GP=$(grep -oE '\|g_PAE\|[[:space:]]+[0-9.]+e?-?[0-9]*' "training/runs/$PR/run.log" | head -1 | awk '{print $NF}')
COS=$(grep -oE 'cos\(PAE,rank\)[[:space:]]+[+-][0-9.]+' "training/runs/$PR/run.log" | head -1 | awk '{print $NF}')
[ -n "$GR" ] || { echo "STOP: no |g_rank| in probe $PR; refusing to launch on a guessed lambda."; exit 1; }
echo "$(date '+%T') probe $PR: |g_rank| $GR  |g_PAE| ${GP:-n/a}  cos ${COS:-n/a}"

if [ "$USE_PAE" = "1" ] && [ -n "$GP" ]; then
    AR=$($PY -I -c "print(f'{$GP/$GR:.3f}')")
    AP=1.0
    GTASK=$($PY -I -c "
import math
gp,gr,c=$GP,$GR,${COS:-0}
k=gp/gr
print(f'{math.sqrt(gp*gp+2*k*gp*gr*c+k*k*gr*gr):.6f}')")
else
    AR=1.0; AP=0.0; GTASK=$GR
fi
LAM=$($PY -I run_logging/l2sp_lambda.py --old-lambda 2.5e-4 --old-grad 4.0207e-02 \
        --old-params 40960 --new-grad "$GTASK" --new-params 40960 \
      | grep -oE 'nearest value worth using: [0-9.e-]+' | awk '{print $NF}')
[ -n "$LAM" ] || { echo "STOP: no lambda from l2sp_lambda.py"; exit 1; }
echo "$(date '+%T') lambda $LAM  alpha_rank $AR  alpha_pae $AP  (|g_task| $GTASK)"

BEFORE=$(ls training/runs | while read -r n; do [ -d "training/runs/$n" ] && echo "$n"; done \
         | grep -E '^[a-z0-9]{6}$' | grep -vx latest | sort)
./shell_scripts/launch_wsonly.sh "$LAM" "$AR" "$AP" > training/runs/wsonly_launch.log 2>&1 &
L=$!
NEW=""
for _ in $(seq 600); do
    NEW=$(comm -13 <(echo "$BEFORE") <(ls training/runs | while read -r n; do [ -d "training/runs/$n" ] && echo "$n"; done | grep -E '^[a-z0-9]{6}$' | grep -vx latest | sort) | head -1)
    [ -n "$NEW" ] && break
    kill -0 "$L" 2>/dev/null || break
    sleep 2
done
if [ -n "$NEW" ]; then
    echo "$NEW" > training/runs/wsonly_rid
    echo "$(date '+%T') within-seed arm is $NEW"
    nohup ./shell_scripts/watch_arm.sh "$NEW" >> "training/runs/watch_$NEW.log" 2>&1 &
else
    echo "$(date '+%T') no new run directory; see training/runs/wsonly_launch.log"
fi
wait "$L"; echo "$(date '+%T') launcher returned $?"
