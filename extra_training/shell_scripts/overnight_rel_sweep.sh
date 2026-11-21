#!/usr/bin/env bash
# SUPERSEDED by supervise_rel_alphas.sh -- DO NOT RUN.
# Why: it waited only on the first sweep's PID and could not see WHY that sweep exited, so a
# dropped alpha in the first half was invisible to the second half.
# Kept for the record only; it refuses below.
echo "REFUSING: this script is superseded by supervise_rel_alphas.sh. See the header." >&2
exit 64
# THE OVERNIGHT CHAIN, ONE PROCESS SO NOTHING RACES FOR THE CARDS.
#
#   1. wait for the running alpha sweep (100, 200, 50) to exit
#   2. run three more 10-epoch arms: alpha 250, 300, 25
#   3. STOP. The six 10-epoch arms are the experiment.
#
# THE 30-EPOCH EXTENSION IS OFF unless EXTEND=1. It was specified and then
# narrowed to the six arms; gated rather than deleted so turning it back on is
# one variable and not a rewrite.
#
# WHY ONE CHAIN AND NOT TWO QUEUES. Two loops both polling cards_busy would both see the cards
# free at the same instant; launch_pae_crop_arm.sh refuses occupied cards, so the loser does not
# corrupt anything, but it DOES exit and silently drop its remaining alphas. Waiting on the first
# sweep's pid is the only ordering that cannot interleave.
#
# THE WINNER IS PICKED FROM metric_valfull83_ep10.json, not from a table reprinted by hand:
# mean_dockq is max-pooled, mean_dockq_select_mean is mean-pooled, and the arm's score is the
# larger of the two. Ties go to the smaller alpha, which is the weaker intervention.
set -u
cd "$(dirname "$0")/.."
SD=/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
PY=/home/ebrahim/Protenix/.venv/bin/python

WAIT_PID="${WAIT_PID:-}"
if [ -n "$WAIT_PID" ]; then
    say "waiting for the running sweep (pid $WAIT_PID) to finish its three alphas"
    while [ -d "/proc/$WAIT_PID" ]; do sleep 60; done
    say "first sweep exited"
fi

say "second sweep: alphas 250 300 25"
ALPHAS="250 300 25" ./shell_scripts/queue_rel_alpha_sweep.sh
rc=$?
[ "$rc" -eq 0 ] || { say "STOPPING: second sweep returned $rc"; exit "$rc"; }

if [ "${EXTEND:-0}" != "1" ]; then
    say "six 10-epoch arms complete. Not extending (EXTEND=1 to pick a winner and run 30 epochs)."
    exit 0
fi

say "picking the winner across all six alphas"
BEST=$("$PY" -I - <<'PYEOF'
import json, os
best = None
for alpha in (25, 50, 100, 200, 250, 300):
    f = f"training/runs/rel_alpha{alpha}_rid"
    if not os.path.exists(f):
        print(f"#  alpha {alpha}: no rid file -- arm never started"); continue
    rid = open(f).read().strip()
    m = f"training/runs/{rid}/metric_valfull83_ep10.json"
    if not os.path.exists(m):
        print(f"#  alpha {alpha} ({rid}): no epoch-10 val"); continue
    d = json.load(open(m))
    mx, mn = d.get("mean_dockq"), d.get("mean_dockq_select_mean")
    score = max(x for x in (mx, mn) if x is not None)
    pool = "max" if score == mx else "mean"
    print(f"#  alpha {alpha:>3} ({rid}): max {mx} / mean {mn}  -> {score:.4f} ({pool})")
    # TIES GO TO THE SMALLER ALPHA: the loop ascends, so a later equal score does not replace.
    if best is None or score > best[0] + 1e-12:
        best = (score, alpha, rid, pool)
print(f"BEST_ALPHA={best[1]}" if best else "BEST_ALPHA=")
PYEOF
)
echo "$BEST"
A=$(echo "$BEST" | grep -oE "BEST_ALPHA=[0-9]+" | cut -d= -f2)
[ -n "$A" ] || { say "STOPPING: no arm produced an epoch-10 val, nothing to extend"; exit 7; }
say "winner: alpha $A -- relaunching from scratch for 30 epochs, evals every 5"

for i in $(seq 1 480); do
    n=$("$PY" -I run_logging/cards_busy.py 2>/dev/null | tail -1)
    case "$n" in ''|*[!0-9]*) n=99;; esac
    [ "$n" -eq 0 ] && break
    sleep 30
done
[ "${n:-99}" -eq 0 ] || { say "STOPPING: cards busy, 30-epoch arm NOT launched"; exit 3; }

SEEN=$SD/rel30_before.txt
ls -d training/runs/*/ | grep -vE "/(latest|archive)/$" | xargs -r -n1 basename | sort > "$SEEN"
LOG=training/runs/rel30_alpha${A}_launch.log
ALPHA_RANK="$A" EPOCHS=30 EVAL_EVERY="${EVAL_EVERY:-5}" \
  EXTRA_EVAL="name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=${EVAL_EVERY:-5}" \
  DRY=0 nohup ./shell_scripts/launch_armE_rel.sh > "$LOG" 2>&1 &
RID=""
for i in $(seq 1 60); do
    RID=$(ls -d training/runs/*/ | grep -vE "/(latest|archive)/$" | xargs -r -n1 basename \
          | sort | comm -13 "$SEEN" - | head -1)
    [ -n "$RID" ] && break
    sleep 10
done
[ -n "$RID" ] || { say "30-epoch arm: no new run directory"; exit 4; }
echo "$RID" > training/runs/rel30_rid
echo "$RID" > training/runs/pose6_norms_rid
printf 'rel a%s 30ep' "$A" > "training/runs/${RID}_name"
echo 0 > "training/runs/${RID}_last_autoplot"
say "30-epoch arm $RID is up at alpha $A"
