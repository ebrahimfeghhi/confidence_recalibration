#!/bin/bash
# SCORE ONE HEAD on the fixed 280-item training slice and on val, then print the table.
#
# DROPOUT IS OFF HERE and cannot be otherwise: phase_k_eval builds its own model and never takes
# --pair-dropout. Stated because "scored without dropout" is a claim worth being able to point at.
#
# ONE PASS PER SIDE. Each pass emits our score and the shipped score for the same candidate, so
# the shipped reference row costs no extra GPU.
set -u
# PIPEFAIL, BECAUSE THE REPORT IS PIPED INTO tee. score_report.py asserts its candidate count and
# exits non-zero on a short pass; without this the pipeline took tee's status, the assert went to
# stderr unnoticed and tee had already truncated the report to zero bytes. Ten reports were
# silently emptied that way once.
set -o pipefail
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration/extra_training
mkdir -p "${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}"
SD=${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}
CKPT=${CKPT:?set CKPT to a head_ep*.pt}
TAG=${TAG:?set TAG, e.g. ep20}
SIDES=${SIDES:-both}          # both | val | slice
# RE-SCORING COSTS A FULL PASS PER SIDE. The run already wrote its own shards under
# training/runs/<rid>/{valfull83,slice280}_eval_shards_ep<N>/ at every scoring epoch; point
# run_logging/score_report.py at those instead of running this script, unless the checkpoint was
# never scored in-training (the epoch-2 head of a stage that ended before its trigger, say).
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }

# NO --drop-unmapped-poses HERE. The in-training eval does not forward it, so with it the two
# harnesses measure different candidate sets: a pose OST cannot map is silently dropped in one and
# a hard stop in the other. Loud in both.
# --prefetch: read the next unit's trunk on a background thread while the GPU scores this
# one. Its own help measures load at 4.7 min of a 26.1 min pass, strictly serial with the
# forward -- which is what the idle gaps between units were on the epoch-1 and -2 passes.
# BOTH POOLINGS, ALWAYS. --pool-mean adds an iptm_mean column computed in the same forward, so
# the mean-pooled reading costs no extra GPU -- but it is default off, so every arm scored before
# this has rows that CANNOT answer "was the mean better" offline and would need a full re-score.
# The ranking arms train against --rank-pool mean while every baseline here selects by the max, so
# that question comes up for each of them. Computing both and choosing later is the cheap order.
#
# --pair-iptm-only IS REQUIRED BY --pool-mean (its path already holds the PAE probabilities). It
# does not move the candidate set, iptm, stored_iptm, the DockQ join or the rows schema, and the
# in-training eval has passed it on every pass -- phase_k_train.py calls it measured
# bit-identical. NOT independently re-measured here: the val30_zs_ep{8,10} rows already on disk
# were scored WITHOUT it, so the next pass over either epoch is a free check of that claim.
#
# SELECTION STAYS ON THE MAX. --select-by is not passed, so mean_dockq and every existing column
# keep the shipped convention and stay comparable to every earlier report; the mean-pooled numbers
# arrive as an extra column to be read offline, not as a changed headline.
COMMON="--pose-perblock --pose-perblock-copy --frame-direction --prefetch --pair-iptm-only --pool-mean ${POSE_FLAGS:-}"

if [ "$SIDES" = both ] || [ "$SIDES" = slice ]; then
  say "$TAG: scoring the 280-item training slice (15 seeds)"
  EXTRA="--fold $R/training/fold --results $R/training/fold/results.csv \
--pool $R/training/slice280/pool.csv --roles $R/training/fold/chain_roles_all.csv \
--seed-base 500000 --join-cache $R/training/slice280/join_cache.json $COMMON"
  LAYERNORM_TYPE=openfold $PY -I phases/phase_k_eval_sharded.py \
      --cards 4,5,7 --ckpt "$CKPT" --budget 15 \
      --work "$SD/slice280_$TAG" --out "$SD/slice280_$TAG.json" \
      --extra "$EXTRA" > "$SD/slice280_$TAG.log" 2>&1 \
      || { say "ABORT: slice pass exit $?"; tail -3 "$SD/slice280_$TAG.log"; exit 2; }
fi

if [ "$SIDES" = both ] || [ "$SIDES" = val ]; then
  say "$TAG: scoring val (141 items, 30 seeds)"
  EXTRA="--fold $R/validation/fold --results $R/validation/fold/results.csv \
--pool $R/training/valfull83/pool.csv --roles $R/validation/fold/chain_roles_all83.csv \
--seed-base 400000 --seed-manifest $R/training/valfull83/seed_manifest.json \
--join-cache $R/training/valfull83/join_cache.json $COMMON"
  LAYERNORM_TYPE=openfold $PY -I phases/phase_k_eval_sharded.py \
      --cards 4,5,7 --ckpt "$CKPT" --budget 30 \
      --work "$SD/val30_$TAG" --out "$SD/val30_$TAG.json" \
      --extra "$EXTRA" > "$SD/val30_$TAG.log" 2>&1 \
      || { say "ABORT: val pass exit $?"; tail -3 "$SD/val30_$TAG.log"; exit 3; }
fi

$PY -I run_logging/score_report.py --slice-work "$SD/slice280_$TAG" \
    --val-work "$SD/val30_$TAG" --label "$TAG" | tee "$SD/report_$TAG.txt"
