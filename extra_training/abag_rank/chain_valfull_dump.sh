#!/bin/bash
# AFTER THE TRAINING DUMP: build valfull's residue letters, then dump its PAE. No kill anywhere.
#
# VALIDATION NEEDS ITS OWN DATASET CONFIG. restypes.py defaults to the TRAINING preprocessing
# paths; valfull's structures live under validation/prep and validation/fold, so every path is
# passed explicitly. Building them with the training config would either fail to find the
# structures or, worse, featurise a different molecule under the same pdb id.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
for i in $(seq 1 720); do
  BUSY=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
  [ "$BUSY" = "0" ] && break
  sleep 15
done
BUSY=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards busy ($BUSY)"; exit 1; }

echo "$(TZ=America/New_York date '+%F %-I:%M %p') valfull restypes (8 workers, CPU)"
for s in 0 1 2 3 4 5 6 7; do
  LAYERNORM_TYPE=openfold $PY -I abag_rank/restypes.py \
    --selection training/valfull/structures.csv \
    --fold validation/fold \
    --indices validation/prep/job_indices.csv \
    --bio-dir validation/prep/job_bioassembly \
    --mmcif-dir validation/prep/job_cifs \
    --out validation/abag/restypes --shard $s --nshards 8 > /tmp/claude-1004/rtv_$s.log 2>&1 &
done
wait
N=$(ls validation/abag/restypes/*.npz 2>/dev/null | wc -l)
echo "$(TZ=America/New_York date '+%-I:%M %p') valfull restypes: $N structure(s)"
[ "$N" -ge 53 ] || { echo "REFUSING: expected 53 structures, got $N"; exit 1; }

echo "$(TZ=America/New_York date '+%-I:%M %p') valfull PAE dump, budget 30, 3 cards"
mkdir -p validation/abag/pae_valfull
LAYERNORM_TYPE=openfold timeout 120m $PY -I phases/phase_k_eval_sharded.py \
  --cards 4,5,7 --budget 30 \
  --work validation/abag/pae_valfull_shards \
  --out validation/abag/pae_valfull/metric.json \
  --extra "--fold validation/fold --results validation/fold/results.csv --pool training/valfull/pool.csv --roles validation/fold/chain_roles.csv --seed-base 400000 --pair-iptm-only --prefetch --join-cache training/valfull/join_cache.json --dump-pae $PWD/validation/abag/pae_valfull --dump-pae-restypes $PWD/validation/abag/restypes"
rc=$?
echo "$(TZ=America/New_York date '+%-I:%M %p') valfull dump returned $rc"
exit $rc
