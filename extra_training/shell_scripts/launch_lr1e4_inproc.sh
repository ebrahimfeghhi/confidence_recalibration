#!/bin/bash
# THE SAME ARM, IN ONE PROCESS, with the trainer scoring val itself at every epoch boundary.
#
# WHY THE CHAIN IS GONE. fullset_chain_lr1e4.sh ran one segment per epoch purely so scoring could
# land on chosen epochs: the trainer's trigger is (ep+1) %% EVAL_EVERY == 0, which cannot express
# "1, 2, 4, 7, 10, then every 10" -- but it expresses "every epoch" exactly, as EVAL_EVERY=1. The
# segments were therefore buying nothing here and costing a ~2 min resume per epoch.
#
# THE SAME MEASUREMENT, VERIFIED. --train-eval-pool training/valfull83/pool.csv with
# --train-eval-fold validation/fold, --train-eval-seed-base 400000 and --train-eval-budget 30 is
# what the launcher already passes, and training/valfull83/chain_roles.csv is byte-identical to the
# validation/fold/chain_roles_all83.csv that score_checkpoint.sh passes. Reviewed against
# phase_k_eval.py: the only argv difference is --pair-iptm-only and --pool-mean, neither of which
# touches the candidate set, iptm, stored_iptm, the DockQ join or the rows_*.csv schema that
# run_logging/score_report.py reads; --pool-mean only appends an iptm_mean column. (--select-by
# mean rides along only when rank_pool=mean AND rank_loss, which is false for this PAE-only arm.)
#
# THE SLICE, ON ITS OWN CADENCE. --extra-eval fires at (ep+1) %% every == 0, so every=10 is epochs
# 10, 20 ... 60 -- the SLICE_AT of the chain it replaces. budget 15, seed_base 500000 and
# fold training/fold are score_checkpoint.sh's slice arguments, and training/slice280/chain_roles
# .csv is byte-identical to the training/fold/chain_roles_all.csv it passes.
#
# THE TABLES. The chain also PRINTED each report; the trainer only writes shard rows.
# shell_scripts/report_watch.sh turns those into $SD/report_<TAGP>_ep<N>.txt on CPU, and
# run_logging/plot_fullset.py reads both harnesses so the curve is continuous across the switch.
#
# THE ENV BLOCK BELOW WAS LIFTED FROM fullset_chain_lr1e4.sh, not retyped: EPOCHS and EVAL_EVERY
# are the only edits. Retyping a launch env cost four failed launches in twenty minutes once.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
OH=training/fullset
EPOCHS="${EPOCHS:-60}"
RESUME="${RESUME:?pass RESUME=<absolute path to a resume_ep<N>.pt>}"
RESUME_NOTE="${RESUME_NOTE:-same arm, now in one process: val every epoch from the trainer own --eval-pool trigger instead of one segment per epoch.}"
EXTRA_EVAL="${EXTRA_EVAL:-name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=10}"
export RESUME RESUME_NOTE EXTRA_EVAL
echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') resuming from $RESUME to epoch $EPOCHS, val every epoch"
env RESUME="$RESUME" RESUME_NOTE="$RESUME_NOTE" EXTRA_EVAL="$EXTRA_EVAL" \
        PREFLIGHT=$OH/alignment_preflight.json \
        MANIFEST=$OH/structures.csv \
        SCORED_PAIRS=$OH/pool.csv \
        PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0.23 \
        CARDS=4,5,7 EPOCHS=$EPOCHS EVAL_EVERY=1 LR=${LR:-1e-4} PAIR_DROPOUT=0.25 \
        PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
    ./shell_scripts/launch_pae_crop_arm.sh
