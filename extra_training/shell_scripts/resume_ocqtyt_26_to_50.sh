#!/bin/bash
# RESUME ocqtyt FROM EPOCH 26 TO 50. Full confidence head, NO L2, NO pose injection, full train set.
#
# WHY. This is the only run that shows what the head can fit on 487 structures: training PAE 2.7384
# -> 2.1007 by epoch 26, where every constrained arm plateaus near 2.61. It was stopped at 26 of a
# 500-epoch horizon with the loss still falling, so 2.10 is where it was interrupted, not where it
# converges. The overfit30 chain needed ~136 passes per complex to reach its in-sample result;
# ocqtyt has had 26. Running to 50 nearly doubles that.
#
# TRAINING IS BIT-IDENTICAL TO ocqtyt's OWN SETTINGS -- same lr 0.01, warmup 20, accum 85, nsample
# 5, seeds 500000-500009, crop 3400/10/5, alpha_pae 1.0, clip 25.0, structures.csv (488, including
# 9dzj), same skip list, --blocks-per-ckpt 0. Only --epochs changes, from 500 to 50, so the run
# stops at a stated horizon instead of being interrupted again.
#
# ONE MEASUREMENT CHANGE, AND IT CHANGES NO GRADIENT: --train-eval-pool moves from val10 to
# valfull. ocqtyt scored itself on val10's 18 interfaces, which is not comparable to ANY number in
# paper_results.md -- val10 and valfull are never comparable. Pointing the per-epoch DockQ eval at
# valfull's 92 interfaces makes every epoch from 27 on directly readable against shipped 0.5830 and
# against all four arms. train_eval_pool is not in the resume guard's field set, so this is not
# drift; the held-out PAE monitor stays on val10 so ITS series remains continuous.
#
# THE EVAL IS SHARDED OVER ALL THREE CARDS (--eval-cards 4,5,7), as every other run here does.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
BUNDLE=training/runs/ocqtyt/resume_ep26.pt
[ -f "$BUNDLE" ] || { echo "REFUSING: $BUNDLE does not exist"; exit 1; }
echo "RESUMING ocqtyt from epoch 26 to 50 (full head, no L2, no pose injection)"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --numa-map 0:0,1:0,2:1 --note "RESUME OF ocqtyt AT EPOCH 27 TO A 50-EPOCH HORIZON. Full confidence head, NO L2-SP, NO pose injection, full training set (structures.csv, 488). WHY: this is the only run showing what the head can FIT on 487 structures -- training PAE 2.7384 -> 2.1007 by epoch 26 where every constrained arm plateaus near 2.61 -- and it was stopped at 26 of a 500-epoch horizon with the loss still falling. The overfit30 chain needed ~136 passes per complex for its in-sample result; this has had 26. Training settings are ocqtyt's own, unchanged. ONE MEASUREMENT CHANGE that touches no gradient: --train-eval-pool moves val10 -> valfull so every epoch is comparable to shipped 0.5830 and to the four pose/L2 arms; val10 and valfull are never comparable and ocqtyt's old series was unreadable against anything. The held-out PAE monitor stays on val10 so that series remains continuous. TEN-EPOCH MINIMUM does not apply -- the user retracted it." \
  -- --resume "$BUNDLE" --resume-allow-drift \
  --resume-note "continue ocqtyt to a 50-epoch horizon; --epochs 500 -> 50 and the DockQ eval pool moves val10 -> valfull, neither of which changes a gradient" \
  --fold training/fold --labels training/fold/labels \
  --crop-margin 10 --crop-contact 5 --crop-size 3400 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight.json \
  --manifest-selection training/full_train/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --indices training/prep/job_indices.csv \
  --bio-dir training/prep/job_bioassembly \
  --mmcif-dir training/prep/job_cifs \
  --epochs 50 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 \
  --save-every 244 \
  --pae-scored-pairs training/full_train/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 --prefetch-workers 4 \
  --val-loss-every 1 \
  --val-loss-fold validation/fold \
  --val-loss-selection training/val10/structures.csv \
  --val-loss-pool training/val10/pool.csv \
  --val-loss-preflight training/val10/alignment_preflight.json \
  --val-loss-seeds 400000-400029 \
  --val-loss-prep validation/prep \
  --train-eval-pool training/valfull/pool.csv \
  --train-eval-fold validation/fold \
  --train-eval-seed-base 400000 \
  --train-eval-budget 30 \
  --train-eval-every 1 \
  --eval-cards 4,5,7 \
  --blocks-per-ckpt 0 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
rc=$?
echo "resume_ocqtyt_26_to_50 returned $rc"
exit $rc
