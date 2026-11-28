#!/bin/bash
# AT EPOCH 20: if the margin to correct selection is still above 0.04, drop lr to 1e-4 and resume
# from resume_ep20.pt; otherwise leave the arm alone and say so.
#
# THE THRESHOLD IS THE USER'S AND IT IS CHECKED, NOT ASSUMED. The margin is recomputed here from
# the epoch-20 rows rather than read off a figure, so the decision and the number it rests on come
# from the same place.
#
# lr IS IN THE RESUME GUARD, so the resume names it in --resume-override with a note. That is the
# guard working: a silently changed learning rate across a resume is a different experiment
# wearing the same run id.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration/extra_training
RID=ambgom
say() { echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
newest_rid() { ls -td training/runs/*/ | grep -vE "/latest/$" | head -1 | xargs basename; }

say "waiting for the epoch-20 eval"
for i in $(seq 1 180); do
    [ -s "training/runs/$RID/metric_hard15_ep20.json" ] && break
    sleep 20
done
[ -s "training/runs/$RID/metric_hard15_ep20.json" ] || { say "ABORT: no epoch-20 metric"; exit 2; }

M=$($PY -I - <<'PYX'
import csv, glob
A = 0.23
by = {}
for nm in ("hard15", "hard1"):
    for f in glob.glob(f"training/runs/ambgom/{nm}_eval_shards_ep20/rows_*.csv"):
        for r in csv.DictReader(open(f)):
            try:
                by.setdefault(r["iface"], []).append((float(r["iptm"]), float(r["dockq"])))
            except (TypeError, ValueError, KeyError):
                pass
g = []
for c in by.values():
    acc = [x for x in c if x[1] >= A]
    if acc:
        g.append(max(x[0] for x in c) - max(x[0] for x in acc))
print(f"{(sum(g)/len(g)) if g else float('nan'):.6f}")
PYX
)
say "epoch-20 margin = $M (threshold 0.04)"
HOT=$($PY -I -c "print(1 if float('$M') > 0.04 else 0)")
if [ "$HOT" != "1" ]; then
    say "margin is AT OR BELOW 0.04 -- leaving the arm at lr 1e-3, as agreed"
    exit 0
fi

say "margin above 0.04: stopping and resuming at lr 1e-4"
./shell_scripts/stop_run.sh "$RID" --yes
for i in $(seq 1 60); do
    [ "$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)" = "0" ] && break
    sleep 15
done
[ -s "training/runs/$RID/resume_ep20.pt" ] || { say "ABORT: resume_ep20.pt missing"; exit 3; }
PRE=$(newest_rid)
SUBSET_OK=1 \
PREFLIGHT=training/overfit_hard/alignment_preflight.json \
MANIFEST=training/overfit_hard/structures.csv \
SCORED_PAIRS=training/overfit_hard/pool.csv \
PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0.23 \
EXTRA_EVAL="name=hard15,pool=training/overfit_hard/pool_b15.csv,fold=training/fold,seed_base=500000,budget=15,every=5;name=hard1,pool=training/overfit_hard/pool_b1.csv,fold=training/fold,seed_base=500000,budget=1,every=5" \
CARDS=4,5,7 EPOCHS=50 EVAL_EVERY=0 LR=1e-4 \
PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
RESUME="$R/training/runs/$RID/resume_ep20.pt" \
RESUME_OVERRIDE=lr \
RESUME_NOTE="margin to correct selection was +$M at epoch 20, above the 0.04 threshold; lr 1e-3 -> 1e-4 to refine rather than keep overshooting" \
    nohup ./shell_scripts/launch_pae_crop_arm.sh > training/runs/lr1e4_resume_launch.log 2>&1 &
LPID=$!
NEW=""
for i in $(seq 1 90); do
    N=$(newest_rid); [ "$N" != "$PRE" ] && { NEW="$N"; break; }
    [ -d "/proc/$LPID" ] || { say "ABORT: launcher exited before a run appeared";
        grep -E "REFUS|STOP:|not found" training/runs/lr1e4_resume_launch.log | head -5; exit 4; }
    sleep 20
done
[ -n "$NEW" ] || { say "ABORT: no run directory after 30 min"; exit 5; }
echo "$NEW" > training/runs/lr1e4_rid
echo 20 > "training/runs/${NEW}_last_sent"
say "resumed at lr 1e-4 as $NEW"
