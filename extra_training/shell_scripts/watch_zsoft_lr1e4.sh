#!/bin/bash
# WAIT FOR THE LIVE 1e-4 RANKING ARM'S NEXT COMPLETE VAL PASS, then redraw and print it.
#
# NO REPORT NEEDED. The arm writes its shard rows inside its own run directory and
# run_logging/plot_fullset.py reads those directly, so this waits on the three .meta.json
# completion markers rather than on report_watch.sh having turned them into a table.
#
# COMPLETE, NOT MERELY PRESENT. Each shard writes rows_*.csv and then its .meta.json, so a
# directory counted between the two is short by a third of the pool -- which moves the mean
# without looking wrong. Three metas is the gate.
#
# LIVENESS. Aborts if run.log has not moved in STALL seconds rather than waiting forever on a
# dead arm.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RID="${RID:-$(cat training/runs/zsoft_lr1e4_rid)}"
SEEN="${SEEN:?set SEEN to the highest val epoch already reported}"
STALL="${STALL:-3000}"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }

while :; do
    E=""
    for d in training/runs/$RID/valfull83_eval_shards_ep*; do
        [ -d "$d" ] || continue
        [ "$(ls "$d"/rows_*.csv.meta.json 2>/dev/null | wc -l)" -eq 3 ] || continue
        N=$(basename "$d" | sed -E 's/.*_ep([0-9]+)$/\1/')
        [ "$N" -gt "$SEEN" ] 2>/dev/null && { [ -z "$E" ] || [ "$N" -lt "$E" ]; } && E="$N"
    done
    [ -n "$E" ] && break
    AGE=$(( $(date +%s) - $(stat -c %Y "training/runs/$RID/run.log") ))
    [ "$AGE" -gt "$STALL" ] && { say "### STALLED: run.log untouched for ${AGE}s"; exit 2; }
    sleep 60
done

say "epoch $E val complete"
$PY -I - "$RID" "$E" <<'PYEOF'
import sys
sys.path.insert(0, "run_logging")
import plot_fullset as P
rid, e = sys.argv[1], int(sys.argv[2])
d = P.score_points({e: f"training/runs/{rid}/valfull83_eval_shards_ep{e}"})[e]
print(f"    max : q {d['q']:.4f}  top1 {d['top1']:.1%}  rho {d['rho']:.4f}")
print(f"    mean: q {d['q_mean']:.4f}  top1 {d['top1_mean']:.1%}  rho {d['rho_mean']:.4f}")
print(f"    shipped q {d['s_q']:.4f}  top1 {d['s_top1']:.1%}")
PYEOF
./shell_scripts/draw_three_arms.sh 2>&1 | tail -1
