#!/bin/bash
# RESUME ambgom FROM EPOCH 50 AT lr 5e-4, running to epoch 70.
#
# TWO CHANGES, BOTH THE USER'S: lr 1e-3 -> 5e-4 and the epoch count 60 -> 70. Everything else is
# ambgom's own: the 151 hard lists, PAE-only with the per-list class balance at DockQ 0.23, no
# dropout, no held-out eval, the two uncropped extra-evals every 5 epochs.
#
# WHY NOW. At epoch 50 frac acceptable hit the run's best (0.1788, against shipped's 0.0861) but
# the margin ROSE to +0.0341 from +0.0253 -- the marginal lists crossed while the hard core moved
# further away. A smaller step is the natural response to that shape.
#
# lr IS IN THE RESUME GUARD, so --resume-override lr carries a note saying why. A learning rate
# that changes across a resume without a record is a different experiment under the same name.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration/extra_training
say() { echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
newest_rid() { ls -td training/runs/*/ | grep -vE "/latest/$" | head -1 | xargs basename; }

[ -s training/runs/ambgom/resume_ep50.pt ] || { say "ABORT: resume_ep50.pt missing"; exit 2; }
for i in $(seq 1 40); do
    [ "$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)" = "0" ] && break
    sleep 15
done
PRE=$(newest_rid)
say "=== resuming ambgom at epoch 51, lr 5e-4, to epoch 70 ==="
SUBSET_OK=1 \
PREFLIGHT=training/overfit_hard/alignment_preflight.json \
MANIFEST=training/overfit_hard/structures.csv \
SCORED_PAIRS=training/overfit_hard/pool.csv \
PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0.23 \
EXTRA_EVAL="name=hard15,pool=training/overfit_hard/pool_b15.csv,fold=training/fold,seed_base=500000,budget=15,every=5;name=hard1,pool=training/overfit_hard/pool_b1.csv,fold=training/fold,seed_base=500000,budget=1,every=5" \
CARDS=4,5,7 EPOCHS=70 EVAL_EVERY=0 LR=5e-4 \
PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
RESUME="$R/training/runs/ambgom/resume_ep50.pt" \
RESUME_OVERRIDE=lr \
RESUME_NOTE="epoch 50 hit the run's best frac acceptable 0.1788 while the margin ROSE to +0.0341 from +0.0253; lr 1e-3 -> 5e-4 to refine the hard core rather than keep trading it for marginal lists. Epochs extended 60 -> 70." \
    nohup ./shell_scripts/launch_pae_crop_arm.sh > training/runs/ambgom_5e4_launch.log 2>&1 &
LPID=$!
RID=""
for i in $(seq 1 90); do
    N=$(newest_rid); [ "$N" != "$PRE" ] && { RID="$N"; break; }
    [ -d "/proc/$LPID" ] || { say "ABORT: launcher exited before a run appeared"
        grep -E "REFUS|STOP:|not found" training/runs/ambgom_5e4_launch.log | head -5; exit 4; }
    sleep 20
done
[ -n "$RID" ] || { say "ABORT: no run directory after 30 min"; exit 5; }
echo "$RID" > training/runs/ambgom_5e4_rid
echo 50 > "training/runs/${RID}_last_sent"
say "resumed as $RID at lr 5e-4, to epoch 70"

# AND RE-ARM THE POSE-INJECTION CHAIN ON THE NEW RUN. It was watching ambgom for epoch 60, which
# will never come now; the resume is a new run id and the queue has to follow it.
PREV="$RID" WANT=70 nohup ./shell_scripts/overfit_poseinject.sh \
    > training/runs/poseinject_chain.log 2>&1 &
say "pose-injection chain re-armed on $RID at epoch 70"
