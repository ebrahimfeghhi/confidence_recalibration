#!/bin/bash
# CHAIN: oe6sru -> readout arm -> within-seed arm, with no idle gap between any of them.
#
#   1. wait for oe6sru epoch 15, stop it by verified pid
#   2. record the PAE verdict NOW, while its shards are fresh, for step 5 to use
#   3. probe at the readout trainable set, derive lambda and alpha_rank, launch the readout arm
#   4. wait for the readout arm to finish its 15 epochs
#   5. probe at the within-seed set, launch it with PAE on or off per the step-2 verdict
#
# THE PAE VERDICT IS COMPUTED AND WRITTEN DOWN AT STEP 2, not at step 5, so the within-seed arm's
# configuration cannot be influenced by anything the readout arm does in between. The rule itself
# was fixed before any of these numbers existed: PAE counts as having helped if oe6sru's mean
# paired (trained - shipped) DockQ over epochs 8-15 exceeds hronsr's +0.011556, the best no-PAE
# arm on this pool, computed by the same run_logging/paired_mean.py on both sides.
#
# WHAT THAT COMPARISON CANNOT SETTLE: oe6sru differs from hronsr in three ways at once -- PAE on,
# pairs cross instead of all at L=8, and 830,976 trainable parameters instead of 44,032. A win is
# a win for the package, not for PAE alone.
#
# EVERY PID IS RE-VERIFIED against its --out immediately before being signalled, and the script
# refuses rather than killing a pid that has been recycled. pkill/pgrep are never used.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
LOG=training/runs/handover_chain.log
exec >>"$LOG" 2>&1
HRONSR_BASELINE=0.011556
echo "=== chain armed $(date '+%F %T %Z') ==="

stop_run () {   # $1 = rid
    local rid="$1" p out
    for p in $(ls /proc | grep -E '^[0-9]+$'); do
        out=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | grep -A1 -x -- '--out' | tail -1)
        case "$out" in
            */training/runs/$rid)
                grep -q phase_k_train "/proc/$p/cmdline" 2>/dev/null \
                    && kill -TERM "$p" 2>/dev/null && echo "  SIGTERM -> $p ($rid)";;
        esac
    done
    for _ in $(seq 60); do [ "$($PY -I run_logging/cards_busy.py)" = "0" ] && break; sleep 10; done
}

wait_epoch () { # $1 = rid, $2 = target epoch
    local rid="$1" tgt="$2" ep
    for _ in $(seq 100000); do
        [ -f training/runs/CANCEL_CHAIN ] && { echo "CANCELLED"; exit 0; }
        ep=$(cat "training/runs/${rid}_last_plotted" 2>/dev/null || echo 0)
        [ "${ep:-0}" -ge "$tgt" ] 2>/dev/null && return 0
        [ -f "training/runs/$rid/status.json" ] && { echo "  $rid ended early at epoch $ep"; return 0; }
        sleep 30
    done
}

launch_and_watch () {  # $1 = script, rest = args ; echoes the new rid
    local script="$1"; shift
    local before new l
    before=$(ls training/runs | while read -r n; do [ -d "training/runs/$n" ] && echo "$n"; done \
             | grep -E '^[a-z0-9]{6}$' | grep -vx latest | sort)
    "$script" "$@" > "training/runs/$(basename "$script" .sh)_launch.log" 2>&1 &
    l=$!
    for _ in $(seq 600); do
        new=$(comm -13 <(echo "$before") <(ls training/runs | while read -r n; do [ -d "training/runs/$n" ] && echo "$n"; done | grep -E '^[a-z0-9]{6}$' | grep -vx latest | sort) | head -1)
        [ -n "$new" ] && break
        kill -0 "$l" 2>/dev/null || break
        sleep 2
    done
    [ -n "$new" ] && nohup ./shell_scripts/watch_arm.sh "$new" >> "training/runs/watch_$new.log" 2>&1 &
    echo "$new"
}

probe_lambda () {  # $1 = probe script, $2 = old lambda, $3 = old grad, $4 = anchored params
                   # echoes "<lambda> <alpha_rank> <alpha_pae>"
    local probe="$1" oldlam="$2" oldg="$3" npar="$4" pr gr gp cos ar gt lam
    "$probe" > "training/runs/$(basename "$probe" .sh).log" 2>&1
    pr=$(ls -t training/runs | while read -r n; do [ -d "training/runs/$n" ] && echo "$n"; done \
         | grep -E '^[a-z0-9]{6}$' | grep -vx latest | head -1)
    gr=$(grep -oE '\|g_rank\|[[:space:]]+[0-9.]+e?-?[0-9]*' "training/runs/$pr/run.log" | head -1 | awk '{print $NF}')
    gp=$(grep -oE '\|g_PAE\|[[:space:]]+[0-9.]+e?-?[0-9]*' "training/runs/$pr/run.log" | head -1 | awk '{print $NF}')
    cos=$(grep -oE 'cos\(PAE,rank\)[[:space:]]+[+-][0-9.]+' "training/runs/$pr/run.log" | head -1 | awk '{print $NF}')
    [ -n "$gr" ] && [ -n "$gp" ] || { echo ""; return 1; }
    ar=$($PY -I -c "print(f'{$gp/$gr:.3f}')")
    gt=$($PY -I -c "
import math
gp,gr,c=$gp,$gr,${cos:-0}
k=gp/gr
print(f'{math.sqrt(gp*gp+2*k*gp*gr*c+k*k*gr*gr):.6f}')")
    lam=$($PY -I run_logging/l2sp_lambda.py --old-lambda "$oldlam" --old-grad "$oldg" \
            --old-params 40960 --new-grad "$gt" --new-params "$npar" \
          | grep -oE 'nearest value worth using: [0-9.e-]+' | awk '{print $NF}')
    [ -n "$lam" ] || { echo ""; return 1; }
    echo "$pr $gr $gp $cos $ar $gt $lam"
}

# ---- 1. oe6sru to 15 -------------------------------------------------------
echo "$(date '+%T') waiting for oe6sru epoch 15"
wait_epoch oe6sru 15
echo "$(date '+%T') stopping oe6sru"; stop_run oe6sru

# ---- 2. the PAE verdict, recorded now --------------------------------------
PM=$($PY -I run_logging/paired_mean.py oe6sru 8 15)
MEAN=$(echo "$PM" | awk '{print $2}')
USE_PAE=$($PY -I -c "print(1 if $MEAN > $HRONSR_BASELINE else 0)" 2>/dev/null || echo 0)
echo "$USE_PAE" > training/runs/pae_verdict
echo "$(date '+%T') oe6sru 8-15: $PM   threshold $HRONSR_BASELINE   -> PAE helped = $USE_PAE"

# ---- 3. readout arm --------------------------------------------------------
R=$(probe_lambda ./shell_scripts/probe_readout.sh 2.5e-4 4.0207e-02 57344) || {
    echo "STOP: readout probe gave no gradient; not launching on a guessed lambda"; exit 1; }
set -- $R; echo "$(date '+%T') readout probe $1: |g_rank| $2 |g_PAE| $3 cos $4 -> alpha_rank $5, |g_task| $6, lambda $7"
RID1=$(launch_and_watch ./shell_scripts/launch_readout.sh "$7" "$5" 1.0)
echo "$(date '+%T') readout arm is ${RID1:-FAILED}"
[ -n "$RID1" ] || exit 1
echo "$RID1" > training/runs/readout_rid

# ---- 4. wait for it --------------------------------------------------------
wait_epoch "$RID1" 15
echo "$(date '+%T') stopping $RID1"; stop_run "$RID1"

# ---- 5. within-seed arm, PAE per the step-2 verdict ------------------------
W=$(probe_lambda ./shell_scripts/probe_wsonly.sh 2.5e-4 4.0207e-02 40960) || {
    echo "STOP: wsonly probe gave no gradient"; exit 1; }
set -- $W
if [ "$USE_PAE" = "1" ]; then AR="$5"; AP=1.0; LAM="$7"
else
    AR=1.0; AP=0.0
    LAM=$($PY -I run_logging/l2sp_lambda.py --old-lambda 2.5e-4 --old-grad 4.0207e-02 \
            --old-params 40960 --new-grad "$2" --new-params 40960 \
          | grep -oE 'nearest value worth using: [0-9.e-]+' | awk '{print $NF}')
fi
echo "$(date '+%T') wsonly probe $1: |g_rank| $2 |g_PAE| $3 -> lambda $LAM alpha_rank $AR alpha_pae $AP"
RID2=$(launch_and_watch ./shell_scripts/launch_wsonly.sh "$LAM" "$AR" "$AP")
echo "$(date '+%T') within-seed arm is ${RID2:-FAILED}"
[ -n "$RID2" ] && echo "$RID2" > training/runs/wsonly_rid
echo "=== chain done $(date '+%F %T %Z') ==="
