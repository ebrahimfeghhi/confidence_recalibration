#!/bin/bash
# THE MIXED-DEPTH PATH, ON ONE CARD, FOR AS FEW UNITS AS PROVE IT RUNS.
#
# WHAT IT IS FOR. The `seeds` column in the selection manifest has never been exercised by a real
# training run. Everything about it has been argued from reading and from unit tests: the packer,
# the lr horizon, the per-rank quotas, the PAE per-step divisor and the _xs_steps walk all read a
# step's unit count as a per-epoch constant, and on this population a step is a constant number of
# UNITS rather than of structures. A test that passes in python does not prove the trainer loads a
# 15-seed structure and a 1-seed structure in the same step and survives it.
#
# WHAT IT DELIBERATELY DOES NOT USE:
#   --rank-form rel      needs --rel-labels, and the rel label rebuild is blocked on 4zto's broken
#                        per-pose ost_json. softplus exercises the same cross-seed packing without
#                        that dependency, so the GPU does not wait on a CPU repair.
#   --crop-tokens        the union crop mask needs build_union_crop_masks.py taught the seed
#                        manifest, which is still open. Uncropped is slower per unit and is the
#                        honest regime anyway.
#   3 cards              a smoke test proves a code path, not a throughput number. One card leaves
#                        the other two free and is the smallest thing that can fail honestly.
#                        WORLD=1 does mean the per-rank quota split is only exercised trivially --
#                        that is the one thing this does not prove, and it is why this is a smoke
#                        test and not the gate.
#
# --indices all857_indices.csv IS REQUIRED, NOT COSMETIC. phase_k_train and the fingerprint gate
# both DEFAULT to job_indices.csv, which names 501 structures; this population's labels were built
# against all857_indices.csv and the indices path is part of the label fingerprint. A run left on
# the default rejects every label it loads. The gate caught it first: 2,010 labels over 223
# structures reported as mismatched on `bio`+`cif` with `code` and `loss` matching exactly -- the
# recipe agreed and only the per-structure inputs differed, which is the signature of this and not
# of a stale label.
#
# 4zto IS SKIPPED, not silently: its ost_json does not associate samples with poses
# ("2.json describes , not sample 2"), which is why the ceiling pass refused it. It is one structure
# of 853 and it is named here so the skip is a record.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
CARDS="${CARDS:-4}"
UNITS="${UNITS:-120}"
EPOCHS="${EPOCHS:-1}"
DRY="${DRY:-1}"
MANIFEST=train_val_splits/cache_train_structures_extended.csv
POOL=training/combined/pool.csv
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }
say "script $(basename "$0") sha $(sha1sum "$0" | cut -c1-12)  DRY=$DRY cards=$CARDS units=$UNITS"

# THE GATES, BEFORE ANY GPU IS TOUCHED.
[ -s "$MANIFEST" ] || { say "REFUSING: no $MANIFEST"; exit 1; }
[ -s "$POOL" ] || { say "REFUSING: no $POOL"; exit 1; }
say "the pool must agree with the declared depth"
$PY -I run_logging/build_combined_train_pool.py --expect-manifest "$MANIFEST" \
  --allow-short docs/unrecoverable_interfaces.csv >/dev/null \
  || { say "REFUSING: the pool gate fails against $MANIFEST"; exit 1; }
say "  pool gate OK"
B=$($PY -I run_logging/cards_busy.py 2>/dev/null | tail -1)
if [ "${B:-1}" != "0" ]; then say "REFUSING: cards_busy reports '$B', not 0."; exit 1; fi
M=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
say "cards_busy 0, MemAvailable ${M} GB"
[ "${M:-0}" -lt 250 ] && { say "REFUSING: MemAvailable under the 250 GB floor."; exit 1; }

ARGS=(--rank-loss --rank-form softplus --rank-cross-seed --rank-structs-per-step 2
      --alpha-rank 75 --rank-pool mean
      --fold training/fold
      --indices training/prep/all857_indices.csv
      --manifest-selection "$MANIFEST"
      --manifest-skip '9mx1,4zto'
      --pae-scored-pairs "$POOL" --pae-only --lean-readouts --minimal-items
      --limit-units "$UNITS"
      --epochs "$EPOCHS" --lr 0.01 --warmup-steps 5 --seed 0 --accum 30 --nsample 5
      --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400
      --prefetch-units 2 --prefetch-workers 4
      --blocks-per-ckpt 1 --ckpt-token-threshold 855
      --dist-backend gloo --dist-timeout-min 120
      --max-rss-gb 80 --preload-min-free-gb 250
      --save-every 100000 --allow-code-drift)

NOTE="MIXED-DEPTH SMOKE TEST, one card, ${UNITS} unit(s), ${EPOCHS} epoch(s). Proves the trainer actually runs the selection manifest's per-structure \`seeds\` column: 437 structures at 15 seeds beside 416 at 1, packed into cross-seed steps of a constant 30 units by lr_schedule.pack_cross_seed_steps. Everything about that path has so far been argued from reading and unit tests; three downstream sites read a step's unit count as a per-epoch constant and a wrong one deadlocks on an all-reduce rather than failing. --rank-form softplus rather than rel because the rel label rebuild is blocked on 4zto's broken per-pose ost_json and the GPU should not wait on a CPU repair. Uncropped because the union mask does not yet know the seed manifest. NOT a throughput or quality measurement."

if [ "$DRY" = "1" ]; then
  say "DRY RUN: validating the config, no GPU"
  LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py --cards "$CARDS" \
    --note "$NOTE" --dry-run -- "${ARGS[@]}"
  say "dry run exit $? (re-run with DRY=0 to launch)"
  exit 0
fi
say "LAUNCHING on card(s) $CARDS"
LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py --cards "$CARDS" --note "$NOTE" -- "${ARGS[@]}"
say "launch exit $?"
