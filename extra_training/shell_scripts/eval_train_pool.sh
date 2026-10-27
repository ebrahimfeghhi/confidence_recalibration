#!/bin/bash
# SCORE A FINISHED ARM'S HEAD ON THE TRAINING POOL, to separate "did not learn" from "learned the
# training poses and did not generalise".
#
# Usage:  ./shell_scripts/eval_train_pool.sh <rid> <head_ckpt> <arch flags...>
#
# BUDGET 10, NOT 30, AND THAT IS FORCED BY THE DATA. Training structures were folded at seeds
# 500000-500009 -- 10 seeds x 5 samples = 50 poses each -- while validation has 30 seeds x 5 = 150.
# So a train-pool number is NOT comparable to the valfull numbers in EXPERIMENTS.md, which are all
# budget 30. The matched comparison is against valfull RE-AGGREGATED at 10 seeds, which costs no
# GPU because the rows already exist; those are shipped 0.5756 / oracle 0.6504.
#
# IN-SAMPLE BY CONSTRUCTION: the pool's 977 interfaces cover 488 structures and the arms trained on
# 487 of them (only 9dzj is outside, and it is excluded from the arms' selection, not from the
# pool). That is the point -- this measures training-set fit.
#
# THE ARCHITECTURE FLAGS ARE NOT OPTIONAL. phase_k_eval builds the head from them, and a mismatch
# either makes its state-dict guard refuse the checkpoint or, worse, scores a different model.
# Pass exactly what the arm's args.json says: share arms need --pose-perblock-share, copy arms
# --pose-perblock-copy.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RID="${1:?usage: eval_train_pool.sh <rid> <ckpt> <arch flags...>}"; shift
CKPT="${1:?need a head checkpoint}"; shift
[ -f "$CKPT" ] || { echo "REFUSING: no checkpoint at $CKPT"; exit 1; }
WORK="training/runs/$RID/train_pool_eval"
OUT="training/runs/$RID/metric_trainpool.json"
mkdir -p "$WORK"
EXTRA="--fold training/fold --results training/fold/results.csv --pool training/full_train/pool.csv --roles training/fold/chain_roles.csv --seed-base 500000 $* --join-cache training/full_train/join_cache.json --pair-iptm-only --prefetch"
echo "$(TZ=America/New_York date '+%F %-I:%M %p %Z') $RID: scoring $(basename $CKPT) on the TRAIN pool at budget 10"
echo "  extra: $EXTRA"
LAYERNORM_TYPE=openfold $PY -I phases/phase_k_eval_sharded.py \
  --cards 4,5,7 --ckpt "$CKPT" --budget 10 --work "$WORK" --out "$OUT" --extra "$EXTRA"
rc=$?
echo "$(TZ=America/New_York date '+%F %-I:%M %p') $RID train-pool eval returned $rc"
[ "$rc" = "0" ] && cat "$OUT"
exit $rc
