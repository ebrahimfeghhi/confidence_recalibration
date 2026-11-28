#!/bin/bash
# RESUME THE rel ARM FOR EPOCHS 2..10 WITH THE EVALUATION TURNED OFF.
#
# WHY A RESUME AND NOT A RELAUNCH. The owner asked to skip eval after epoch 1 had already cost 2.3 h
# of training, and eval flags cannot be changed in a live process. Resuming keeps that work: the
# bundle restores the head's weights, AdamW's moments and the step counter, and training picks up at
# epoch N+1. That is only EQUIVALENT to a fresh run because this arm's schedule is FLAT after a
# 20-step warmup -- it passes neither --lr-final nor --lr-steps, so the horizon never enters
# lr_at_step. With a cosine, resuming into a different horizon would silently change the schedule.
#
# WHAT IS LOST BY DROPPING EVAL, stated so nobody is surprised later: the HELD-OUT numbers. No
# valfull Spearman, no selected DockQ, no acceptable-pose fraction at 0.23. What survives is the
# trainer's own per-epoch in-sample diagnostics, which need no eval at all:
#   WITHIN-INTERFACE SPEARMAN (predicted mean ipTM vs DockQ), median over interface-groups
#   SCORE SPREAD, median std(predicted) against std(true iTM) with their ratio
# So the correlation the owner asked to see is still reported every epoch; it is in-sample, and it
# must never be presented as a generalisation result.
#
# THE RESUME REFUSES A MISMATCHED BUNDLE -- world size, seed, accum and nsample all change which
# poses land in which batch, so the optimiser state would belong to a different problem. The values
# here are the launch's, for that reason.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RID="${RID:?set RID to the run to resume}"
FROM_EP="${FROM_EP:-1}"
EPOCHS="${EPOCHS:-10}"
CARDS="${CARDS:-4,5,7}"
DRY="${DRY:-1}"
MANIFEST=train_val_splits/cache_train_structures_extended.csv
POOL=training/combined/pool.csv
MARKER=training/alignment_preflight_sharded.json
RELLAB="${RELLAB:-training/fold/itm_labels_extended}"
R="training/runs/$RID"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }
say "script $(basename "$0") sha $(sha1sum "$0" | cut -c1-12)  $RID ep$FROM_EP -> $EPOCHS, NO EVAL"

B="$R/resume_ep${FROM_EP}.pt"
[ -s "$B" ] || { say "REFUSING: no resume bundle at $B"; exit 1; }
RK=$(grep -oE "\[rank 0\] epoch ${FROM_EP} rank-loss[^|]*" "$R/run.log" 2>/dev/null | head -1)
[ -n "$RK" ] || { say "REFUSING: no epoch-$FROM_EP objective accounting; it did not train."; exit 1; }
say "epoch $FROM_EP trained: $RK"
if grep -qE 'STOP:|Traceback' "$R/run.log" 2>/dev/null; then
  say "REFUSING: $R/run.log holds a STOP or traceback."; exit 1
fi
$PY -I run_logging/build_combined_train_pool.py --expect-manifest "$MANIFEST" \
  --allow-short docs/unrecoverable_interfaces.csv >/dev/null \
  || { say "REFUSING: the pool gate fails"; exit 1; }
$PY -I run_logging/check_rel_label_coverage.py --labels "$RELLAB" --manifest "$MANIFEST" \
  --pool "$POOL" --skip '9mx1,4zto,9q5m' --exclude-ifaces docs/unrecoverable_interfaces.csv \
  >/dev/null || { say "REFUSING: the rel label key set is incomplete"; exit 1; }
BUSY=$($PY -I run_logging/cards_busy.py 2>/dev/null | tail -1)
[ "${BUSY:-1}" != "0" ] && { say "REFUSING: cards_busy reports '$BUSY', not 0."; exit 1; }
M=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
say "gates OK: pool, rel key set, cards_busy 0, MemAvailable ${M} GB"

# NO --eval-pool*, NO --extra-eval. That is the whole difference from the launch.
ARGS=(--rank-loss --rank-form rel --rel-labels "$RELLAB"
      --rank-cross-seed --rank-structs-per-step 2
      --alpha-rank 75 --rank-pool mean
      --fold training/fold
      --indices training/prep/all857_indices.csv
      --manifest-selection "$MANIFEST"
      --manifest-skip '9mx1,4zto,9q5m'
      --pae-scored-pairs "$POOL" --pae-scored-roles training/fold/chain_roles_all.csv
      --pae-only --lean-readouts --minimal-items
      --alignment-preflight "$MARKER"
      --resume "$B"
      --epochs "$EPOCHS" --lr 0.01 --warmup-steps 20 --seed 0 --accum 30 --nsample 5
      --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0
      --prefetch-units 2 --prefetch-workers 4
      --blocks-per-ckpt 1 --ckpt-token-threshold 855
      --dist-backend gloo --dist-timeout-min 240
      --max-rss-gb 80 --preload-min-free-gb 250
      --save-every 250
      --require-replay training/extended_mixed_replay.log
      --allow-code-drift)

NOTE="RESUME $RID FROM EPOCH $FROM_EP TO $EPOCHS, EVALUATION OFF at the owner's request. --rank-form rel at --alpha-rank 75 plus the PAE term, fully uncropped, 851 structures of mixed depth, 6,941 units, 231 steps of 30. A resume rather than a relaunch because eval flags cannot change in a live process and a relaunch would discard epoch 1's 3.9 h; this arm's lr schedule is FLAT after a 20-step warmup so the horizon never enters lr_at_step and epochs $((FROM_EP+1))..$EPOCHS see the learning rates a fresh run would. NO HELD-OUT NUMBERS from here: no valfull Spearman, no selected DockQ, no acceptable fraction at 0.23. The per-epoch WITHIN-INTERFACE SPEARMAN and SCORE SPREAD lines are IN-SAMPLE and must not be read as generalisation."

if [ "$DRY" = "1" ]; then
  say "DRY RUN"
  LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py --cards "$CARDS" \
    --numa-map 0:0,1:0,2:1 --note "$NOTE" --dry-run -- "${ARGS[@]}"
  say "dry run exit $?"; exit 0
fi
say "RESUMING $RID epochs $((FROM_EP+1))..$EPOCHS, no eval, cards $CARDS"
LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py --cards "$CARDS" \
  --numa-map 0:0,1:0,2:1 --note "$NOTE" -- "${ARGS[@]}"
say "launch exit $?"
