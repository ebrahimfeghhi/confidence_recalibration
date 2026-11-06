#!/bin/bash
# EXTEND THE UNCROPPED ALL-PAIRS ARM FROM 1 EPOCH TO 10, by resuming rather than relaunching.
#
# WHY A RESUME IS EXACTLY EQUIVALENT HERE, which is not true in general. --resume restores the
# head's weights, AdamW's moment estimates and the step counter, and picks up at epoch N+1. What
# usually makes a resume-to-a-longer-horizon different from a fresh long run is the LR SCHEDULE: a
# cosine built for 231 steps is not the first tenth of one built for 2,310. This arm passes neither
# --lr-final nor --lr-steps, so its schedule is FLAT after a 20-step warmup -- confirmed from the
# trajectory, lr rising 0.0005 -> 0.003 -> 0.01 and then constant. With a flat schedule the horizon
# does not enter lr_at_step at all, so resuming for epochs 2..10 gives the same sequence of learning
# rates as launching for 10 epochs. The first epoch is therefore kept rather than thrown away.
#
# --epochs IS THE TOTAL, NOT THE REMAINDER. The resume bundle carries its own epoch, training
# continues at N+1, and o2qczq's resume from resume_ep27.pt passed --epochs 60 to mean sixty in all.
# So --epochs 10 here runs epochs 2 through 10.
#
# IT WILL NOT RESUME A RUN THAT DID NOT TRAIN. The owner's condition was "if the first epoch
# trains". Three things have to be true and are checked, not assumed: the resume bundle exists, the
# trainer logged its own per-epoch ranking accounting for epoch 1 (which means pairs were actually
# formed, not merely that the process survived), and no STOP or traceback is in the log.
#
# THE RESUME REFUSES A MISMATCHED BUNDLE by design -- world size, seed, accum and nsample all change
# which poses land in which batch, so the optimiser state would belong to a different problem. This
# passes the same values as the launch for exactly that reason.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RID="${RID:?set RID to the run to extend}"
EPOCHS="${EPOCHS:-10}"
L="${L:-8}"
CARDS="${CARDS:-4,5,7}"
DRY="${DRY:-1}"
MANIFEST=train_val_splits/cache_train_structures_extended.csv
POOL=training/combined/pool.csv
MARKER=training/alignment_preflight_sharded.json
R="training/runs/$RID"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }
say "script $(basename "$0") sha $(sha1sum "$0" | cut -c1-12)  RID=$RID -> $EPOCHS epoch(s), L=$L"

B="$R/resume_ep1.pt"
[ -s "$B" ] || { say "REFUSING: no resume bundle at $B -- epoch 1 has not finished."; exit 1; }
if grep -qE 'STOP:|Traceback' "$R/run.log" 2>/dev/null; then
  say "REFUSING: $R/run.log contains a STOP or traceback."
  grep -E 'STOP:|Traceback' "$R/run.log" | head -3
  exit 1
fi
# THE REAL TEST THAT IT TRAINED: the trainer's own per-epoch ranking accounting for epoch 1. A
# process that survived an epoch without forming a single pair would pass every other check here.
RK=$(grep -oE '\[rank 0\] epoch 1 rank-loss[^|]*' "$R/run.log" 2>/dev/null | head -1)
[ -n "$RK" ] || { say "REFUSING: no epoch-1 ranking accounting in $R/run.log; it did not train."; exit 1; }
say "epoch 1 trained: $RK"
P=$(echo "$RK" | grep -oE '[0-9]+ pair\(s\)' | grep -oE '[0-9]+' || echo 0)
[ "${P:-0}" -gt 0 ] || { say "REFUSING: epoch 1 formed 0 ranking pair(s)."; exit 1; }

$PY -I run_logging/build_combined_train_pool.py --expect-manifest "$MANIFEST" \
  --allow-short docs/unrecoverable_interfaces.csv >/dev/null \
  || { say "REFUSING: the pool gate fails"; exit 1; }
BUSY=$($PY -I run_logging/cards_busy.py 2>/dev/null | tail -1)
[ "${BUSY:-1}" != "0" ] && { say "REFUSING: cards_busy reports '$BUSY', not 0."; exit 1; }
M=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
say "gates OK: pool, cards_busy 0, MemAvailable ${M} GB"

ARGS=(--rank-loss --rank-form softplus --rank-cross-seed --rank-structs-per-step 2
      --rank-pairs all --rank-within-weight "$L"
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
      --eval-pool training/valfull/pool.csv --eval-pool-fold validation/fold
      --eval-pool-seed-base 400000 --eval-pool-budget 30 --eval-pool-every 1
      --allow-code-drift)

NOTE="EXTEND $RID TO $EPOCHS EPOCHS, resuming from its epoch-1 bundle. Uncropped train and eval, --rank-pairs all (within AND cross) at L=$L, 851 structures of mixed depth, 6,941 units, 231 steps of 30 units. A resume rather than a relaunch because this arm's lr schedule is FLAT after a 20-step warmup -- no --lr-final and no --lr-steps -- so the horizon does not enter lr_at_step and epochs 2..10 see the same learning rates a fresh 10-epoch run would. Epoch 1 is kept instead of discarded. Eval every epoch on valfull at budget 30, uncropped, with shipped ipTM on the same poses as the matched baseline."

if [ "$DRY" = "1" ]; then
  say "DRY RUN"
  LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py --cards "$CARDS" \
    --numa-map 0:0,1:0,2:1 --note "$NOTE" --dry-run -- "${ARGS[@]}"
  say "dry run exit $?"
  exit 0
fi
say "RESUMING $RID for epochs 2..$EPOCHS on cards $CARDS"
LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py --cards "$CARDS" \
  --numa-map 0:0,1:0,2:1 --note "$NOTE" -- "${ARGS[@]}"
say "launch exit $?"
