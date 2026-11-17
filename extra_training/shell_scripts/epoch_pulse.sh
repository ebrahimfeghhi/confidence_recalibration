#!/bin/bash
# WHAT IS NEW ON THE LIVE ARM SINCE THE LAST PULSE, and redraw. One shot, no waiting.
#
# WHY THIS EXISTS. A background waiter can print numbers but cannot put a figure in front of the
# user; only the session can. So the loop is: a scheduled trigger wakes the session, the session
# runs this, and it says in one place whether anything changed and what the numbers are. Exits 1
# when nothing is new, so the caller can stay quiet instead of sending the same figure again.
#
# TWO CADENCES. The training-loss panels move every epoch; val only on the arm's eval cadence
# (every 2 epochs here) and the slice every 10. Both are reported, because "a new epoch finished"
# and "a new val point exists" are different events and only the second changes the held-out curve.
#
# STATE IS A FILE, NOT AN ARGUMENT. $SD/pulse_<rid>.state holds the last epoch and last val epoch
# reported, so a trigger that fires while nothing has moved is silent rather than re-sending.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
SD=${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}
RID="${RID:-$(cat training/runs/pose5_arm_rid 2>/dev/null)}"
[ -n "$RID" ] || { echo "### no run id"; exit 2; }
ST="$SD/pulse_$RID.state"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }

EP=$(ls training/runs/$RID/head_ep*.pt 2>/dev/null | sed -E 's/.*head_ep([0-9]+)_.*/\1/' | sort -n | tail -1)
EP=${EP:-0}
VE=0
for d in training/runs/$RID/valfull83_eval_shards_ep*; do
    [ -d "$d" ] || continue
    [ "$(ls "$d"/rows_*.csv.meta.json 2>/dev/null | wc -l)" -eq 3 ] || continue
    N=$(basename "$d" | sed -E 's/.*_ep([0-9]+)$/\1/')
    [ "$N" -gt "$VE" ] && VE="$N"
done
# THE SLICE IS ITS OWN EVENT. Tracking only (epoch, val epoch) meant a slice pass that finished
# on a pulse where neither had moved exited 1 and was never reported -- the in-sample point, which
# is the whole memorisation diagnostic, could be skipped silently.
SE=0
for d in training/runs/$RID/slice280_eval_shards_ep*; do
    [ -d "$d" ] || continue
    [ "$(ls "$d"/rows_*.csv.meta.json 2>/dev/null | wc -l)" -eq 3 ] || continue
    N=$(basename "$d" | sed -E 's/.*_ep([0-9]+)$/\1/')
    [ "$N" -gt "$SE" ] && SE="$N"
done
OEP=-1; OVE=-1; OSE=-1
[ -f "$ST" ] && read -r OEP OVE OSE < "$ST"

AGE=$(( $(date +%s) - $(stat -c %Y "training/runs/$RID/run.log" 2>/dev/null || echo 0) ))
if [ "$AGE" -gt "${STALL:-3600}" ]; then
    say "STALLED: $RID run.log untouched for ${AGE}s at epoch $EP"
    echo "$EP $VE $SE" > "$ST"
    exit 0
fi

if [ "$EP" = "$OEP" ] && [ "$VE" = "$OVE" ] && [ "$SE" = "$OSE" ]; then
    say "nothing new: $RID still at epoch $EP (last val $VE, last slice $SE)"
    exit 1
fi

say "$RID epoch $EP trained (was $OEP); newest complete val $VE (was $OVE), slice $SE (was $OSE)"
if [ "$VE" != "$OVE" ] && [ "$VE" != "0" ]; then
    $PY -I - "$RID" "$VE" <<'PYEOF'
import sys
sys.path.insert(0, "run_logging")
import plot_fullset as P
rid, e = sys.argv[1], int(sys.argv[2])
d = P.score_points({e: f"training/runs/{rid}/valfull83_eval_shards_ep{e}"})[e]
print(f"    VAL ep{e}  max : q {d['q']:.4f}  top1 {d['top1']:.1%}  rho {d['rho']:.4f}")
m = f"mean: q {d['q_mean']:.4f}  top1 {d['top1_mean']:.1%}  rho {d['rho_mean']:.4f}" \
    if d['rho_mean'] == d['rho_mean'] else "mean: absent (no iptm_mean column)"
print(f"              {m}")
print(f"    shipped   q {d['s_q']:.4f}  top1 {d['s_top1']:.1%}  rho {d['s_rho']:.4f}")
PYEOF
fi
for d in training/runs/$RID/slice280_eval_shards_ep*; do
    [ -d "$d" ] || continue
    [ "$(ls "$d"/rows_*.csv.meta.json 2>/dev/null | wc -l)" -eq 3 ] || continue
    N=$(basename "$d" | sed -E 's/.*_ep([0-9]+)$/\1/')
    $PY -I - "$RID" "$N" <<'PYEOF'
import sys
sys.path.insert(0, "run_logging")
import plot_fullset as P
rid, e = sys.argv[1], int(sys.argv[2])
d = P.score_points({e: f"training/runs/{rid}/slice280_eval_shards_ep{e}"})[e]
print(f"    SLICE ep{e}  max  {d['top1']:.1%} / {d['q']:.4f}   shipped {d['s_top1']:.1%} / {d['s_q']:.4f}")
if d['top1_mean'] == d['top1_mean']:
    print(f"                mean {d['top1_mean']:.1%} / {d['q_mean']:.4f}   rho max {d['rho']:.4f} mean {d['rho_mean']:.4f}")
PYEOF
done
RID_PLOT="$RID" ./shell_scripts/draw_pose5.sh 2>&1 | tail -1 \
    || say "figure not drawn yet (no epoch loss line)"

# PUSH ON EVERY NEW EVAL PASS, so the review copy outside this repo is never stale. Only when a
# val or slice pass actually landed -- a bare new epoch changes the loss curve but not the results
# table, and a commit per epoch would bury the ones that matter.
if [ "$VE" != "$OVE" ] || [ "$SE" != "$OSE" ]; then
    $PY -I run_logging/export_pose5_results.py "$RID" >/dev/null 2>&1 \
        && git -C /home/ebrahim/confidence_recalibration add \
               extra_training/docs/pose5_results/jyz3g2_val_slice.csv \
               extra_training/figures/pose5_rel_lr1e4.png >/dev/null 2>&1 \
        && git -C /home/ebrahim/confidence_recalibration diff --cached --quiet \
        || {
            git -C /home/ebrahim/confidence_recalibration commit -q -m \
                "$RID results through val $VE, slice $SE" >/dev/null 2>&1 \
                && git -C /home/ebrahim/confidence_recalibration push -q >/dev/null 2>&1 \
                && say "pushed results through val $VE / slice $SE" \
                || say "push FAILED -- results are on disk, commit by hand"
        }
fi
echo "$EP $VE $SE" > "$ST"
