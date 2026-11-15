#!/bin/bash
# FULL TRAINING SET, SHIPPED INIT, CONSTANT lr -- the arm that is meant to generalise.
#
# WHAT IS DIFFERENT FROM THE OVERFIT ARM, and why each one.
#   THE WHOLE TRAINING SET, pre- and post-cutoff: 853 structures, 1,711 interfaces. The overfit arm
#     trained on 155 hard items and lost 6 points of held-out top-1; the test it could not run is
#     whether the same loss generalises when it sees everything.
#   SHIPPED INIT, which is what --resume absent means here: the launcher loads protenix-v2 and
#     trains the confidence head on top. Nothing carries over from the overfit arm.
#   CONSTANT 1e-3 after the 20-step warmup -- no staircase, no decay window, so a result cannot be
#     attributed to a schedule.
#   PAIR DROPOUT 0.25 during training, OFF for every scoring pass. 0.25 is the rate the best
#     held-out arm to date used (7mo52r, valfull83 0.5963); the overfit arm ran at 0 because it was
#     meant to overfit. CHANGE THIS IF YOU WANTED A DIFFERENT RATE -- it was the one value the
#     specification left blank, and it is the only number here I chose rather than was given.
#
# THE CLASS BALANCE IS UNCHANGED: 0.5/n_good and 0.5/n_bad per item, the mean of the two for a
# straddler, renormalised per structure. An item with no good candidates (or no bad ones) puts
# every candidate in one class, so they all take the same 0.5/n and the normaliser makes that
# exactly uniform -- the undefined case resolves to unweighted, which is what was asked for.
#
# EVAL CADENCE, AND WHY IT IS TWO STAGES. The trainer has ONE held-out trigger,
# (ep+1) %% EVAL_EVERY == 0, so "a val point at epoch 2 and then every 10" is not expressible in a
# single run: with EVAL_EVERY=10 the first val point is epoch 10, and the lr check the protocol
# opens with could never fire. STAGE=open runs 2 epochs at EVAL_EVERY=2 to produce exactly that
# point; STAGE=main resumes from its epoch-2 bundle and runs to EPOCHS at EVAL_EVERY=10.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
newest_rid(){ ls -td training/runs/*/ | grep -vE "/latest/$" | head -1 | xargs basename; }

FS=training/fullset
SLICE=training/slice280
[ -s "$FS/alignment_preflight.json" ] || { say "ABORT: no full-set preflight"; exit 2; }
[ -s "$SLICE/pool.csv" ] || { say "ABORT: no 280-item slice"; exit 3; }
N_SLICE=$(( $(wc -l < "$SLICE/pool.csv") - 1 ))
[ "$N_SLICE" -eq 280 ] || { say "ABORT: slice has $N_SLICE items, expected 280"; exit 4; }

for i in $(seq 1 120); do
    [ "$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)" = "0" ] && break
    sleep 30
done
[ "$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)" = "0" ] || { say "ABORT: cards busy"; exit 5; }

STAGE=${STAGE:-open}
case "$STAGE" in
  open) EP_N=2;  EV=2;  RES=() ;;
  main) EP_N=${EPOCHS:-60}; EV=10
        OPEN_RID=$(cat training/runs/fullset_open_rid 2>/dev/null || true)
        B=$(ls -t training/runs/$OPEN_RID/resume_ep2.pt 2>/dev/null | head -1)
        [ -s "$B" ] || { say "ABORT: no epoch-2 bundle from the opening stage"; exit 7; }
        RES=(RESUME="$PWD/${B#./}" RESUME_OVERRIDE=epochs,train_eval_every \
             RESUME_NOTE="opening stage ran 2 epochs at EVAL_EVERY=2 for the lr check; this is the same arm continuing to $EP_N at EVAL_EVERY=10.") ;;
  *) say "ABORT: STAGE must be open or main"; exit 8 ;;
esac

# THE SLICE NEEDS ITS OWN ROLES FILE. The in-training extra-eval resolves <pool dir>/chain_roles.csv
# and falls back to training/fold/chain_roles.csv, which names 501 structures and misses 46 of the
# slice's 178 -- the pass would STOP at the first scoring epoch, rank 0 would exit inside the eval
# block, and ranks 1 and 2 would sit on the next barrier until the 4-hour dist timeout.
[ -s "$SLICE/chain_roles.csv" ] || { say "ABORT: $SLICE/chain_roles.csv is missing"; exit 9; }

PRE=$(newest_rid)
say "=== stage $STAGE: full training set, shipped init, lr 1e-3, dropout 0.25, $EP_N epoch(s), eval every $EV ==="
PREFLIGHT=$FS/alignment_preflight.json \
MANIFEST=$FS/structures.csv \
SCORED_PAIRS=$FS/pool.csv \
PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0.23 \
EXTRA_EVAL="name=slice280,pool=$SLICE/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=$EV" \
EVAL_POOL=training/valfull83/pool.csv \
CARDS=4,5,7 EPOCHS=$EP_N EVAL_EVERY=$EV LR=1e-3 PAIR_DROPOUT=0.25 \
PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=${DRY:-0} \
    env "${RES[@]}" nohup ./shell_scripts/launch_pae_crop_arm.sh > training/runs/fullset_${STAGE}_launch.log 2>&1 &
LPID=$!
for i in $(seq 1 90); do
    N=$(newest_rid)
    [ "$N" != "$PRE" ] && { echo "$N" > training/runs/fullset_rid
        [ "$STAGE" = open ] && echo "$N" > training/runs/fullset_open_rid
        say "launched as $N"; break; }
    [ -d "/proc/$LPID" ] || { say "ABORT: launcher exited before a run appeared"
        grep -E "REFUS|STOP:|not found" training/runs/fullset_${STAGE}_launch.log | head -5; exit 6; }
    sleep 20
done
say "DONE"
