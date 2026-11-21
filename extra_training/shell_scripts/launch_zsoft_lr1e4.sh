#!/bin/bash
# PAE + HELD-OUT RANKING LOSS (zsoftmax) AT lr 1e-4, 60 epochs, in ONE process.
#
# WHAT CHANGED FROM THE 1e-3 ARM. Only the learning rate and the stopping point: LR 1e-3 -> 1e-4
# and 30 -> 60 epochs. Every other setting is the env block of fullset_chain_zsoft.sh, copied
# rather than retyped -- including SCORED_PAIRS=$OH/pool_labelled.csv, which is the ranking arm's
# pool and NOT the pool.csv the PAE-only arms use.
#
# NO SEGMENT CHAIN. The trainer scores val itself at (ep+1) %% EVAL_EVERY == 0, so EVAL_EVERY=2 is
# "val every 2 epochs" exactly -- epochs 2, 4, 6 ... 60, thirty passes -- and the chain's ~2 min
# resume per segment is gone. The slice rides --extra-eval at every=10, so epochs 10, 20 ... 60,
# six passes. 36 scoring passes in total over the run.
#
# NO RESUME BY DEFAULT. This is a fresh arm from the shipped checkpoint, so RESUME is optional
# here, unlike launch_lr1e4_inproc.sh which exists to continue an arm already in flight.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
OH=training/fullset
EPOCHS="${EPOCHS:-60}"
EVAL_EVERY="${EVAL_EVERY:-2}"
LR="${LR:-1e-4}"
ALPHA_RANK="${ALPHA_RANK:-1}"
RANK_ACCEPT="${RANK_ACCEPT:-0.23}"
RANK_S="${RANK_S:-3}"
EXTRA_EVAL="${EXTRA_EVAL:-name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=10}"
RES=()
if [ -n "${RESUME:-}" ]; then
    [ -s "$RESUME" ] || { echo "REFUSING: RESUME=$RESUME is not a file"; exit 2; }
    RES=(RESUME="$RESUME" RESUME_NOTE="${RESUME_NOTE:-continuing the zsoftmax 1e-4 arm}")
fi
echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') PAE + zsoftmax, lr $LR, to epoch $EPOCHS,"
echo "    val every $EVAL_EVERY epoch(s), slice via EXTRA_EVAL: $EXTRA_EVAL"
env "${RES[@]+"${RES[@]}"}" EXTRA_EVAL="$EXTRA_EVAL" \
        PREFLIGHT=$OH/alignment_preflight.json \
        MANIFEST=$OH/structures.csv \
        SCORED_PAIRS=$OH/pool_labelled.csv \
        PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0.23 \
        RANK_FORM=zsoftmax ALPHA_RANK="$ALPHA_RANK" RANK_ACCEPT="$RANK_ACCEPT" \
        RANK_S="$RANK_S" \
        CARDS=4,5,7 EPOCHS=$EPOCHS EVAL_EVERY=$EVAL_EVERY LR=$LR PAIR_DROPOUT=0.25 \
        PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
    ./shell_scripts/launch_pae_crop_arm.sh
