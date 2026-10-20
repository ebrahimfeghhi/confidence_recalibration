#!/bin/bash
# val10 mean DockQ and frac-acceptable for x3k2m4's epoch 1, at BUDGET 30.
#
# WHY IT IS MISSING. x3k2m4's epoch boundary runs in a fixed order: write the resume bundle, run
# the held-out PAE monitor, then run the --train-eval-pool DockQ eval. A NameError in the
# monitor's csv append killed the run at step 2, so the DockQ eval at step 4 never started -- zero
# eval lines and no shard directories in that run. The bundle is intact, so the number is simply
# measured after the fact.
#
# BUDGET 30, so the baseline is 0.2909 / 0.5556 and the oracle 0.5968 / 1.0000. The budget-100
# figures (0.2626 / 0.5000, oracle 0.6662 / 1.0000) are a DIFFERENT pool depth; never quote one
# against the other. phase_k_eval derives the baseline from the pool it scored, so its output
# states which it used.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
CK=$(ls -t training/runs/x3k2m4/head_ep1_step*.pt | head -1)
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I phases/phase_k_eval_sharded.py \
  --cards 4,5,7 --ckpt "$CK" --budget 30 \
  --work training/runs/x3k2m4/traineval_ep1 \
  --out training/runs/x3k2m4/metric_train_ep1.json \
  --extra "--fold validation/fold --results validation/fold/results.csv --pool training/val10/pool.csv --roles validation/fold/chain_roles.csv --seed-base 400000 --join-cache training/val10/join_cache_b30.json --pair-iptm-only --prefetch"
