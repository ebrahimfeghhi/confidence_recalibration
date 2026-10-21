#!/bin/bash
# POSE-FOCUSED ADAPTATION ON THE 30-STRUCTURE SET, against val10.
#
# WHY overfit30 AND NOT THE FULL SET: there is already a MATCHED CONTROL on this data. 2u8tf4 is
# full-head FT3 on these exact 30 structures at this exact lr, and it is the run that proved the
# objective overfits -- train PAE 2.3248 -> 1.4324 (-38%) while val10 PAE 2.8290 -> 4.2639 (+51%)
# over 142 epochs, against 2.9756 untrained. So "did restricting WHICH computations change reduce
# that?" is answerable here with one variable and nothing else moved. An epoch is ~2.5 min here
# against 34 min on the full set.
#
# THE HONEST LIMITATION: 30 structures did not transfer for the full head and may not transfer for
# any parameterisation. A null result here does NOT clear the method on 488 structures -- it says
# restriction did not rescue a training set this small. A POSITIVE result is the informative one.
#
# THE ANCHOR'S val10 CURVE IS SPARSE. 2u8tf4 ran with NO validation during training; its val10 PAE
# points were backfilled from resume bundles afterwards, so the control is a handful of epochs
# (ep1 2.8290, ep5 2.7231 best, ep142 4.2639). THIS arm runs --val-loss-every 1 and
# --train-eval-every 1, so it produces a per-epoch curve the control never had. To compare
# properly, backfill more of 2u8tf4's bundles with shell_scripts/backfill_val_pae.sh.
#
# WHAT TRAINS: 26,624 parameters, 0.135% of the head -- linear_no_bias_d (256x39),
# linear_no_bias_d_wo_onehot (256x1) and linear_no_bias_pae (64x256). Pose enters the head exactly
# once, as cdist over the distogram representative atom, and reaches z_pair through BOTH distance
# embeddings added together; it leaves as the PAE projection. The four pairformer blocks, both
# s_inputs embeddings and the pde/plddt/resolved weights are FROZEN, so the pretrained structural
# representation is preserved and only how geometry enters and how PAE is read out can adapt.
# --train-only matches with re.search on the relative name, and the launch prints every matched
# tensor; CHECK THAT PRINT says 3 tensors / 26,624 parameters before trusting the run.
#
# pae_ln IS DELIBERATELY NOT INCLUDED, and the reason is not just "the spec said three layers".
# Its gamma is redundant: W @ (gamma * z) = (W diag(gamma)) @ z, absorbable into
# linear_no_bias_pae, which trains here. Only beta adds anything, and what it adds is an output
# bias that a no-bias projection otherwise cannot express. That is a bias-term question, not a
# pose-pathway one, and it is a separate 256-parameter variant if wanted.
#
# LEARNING RATE IS INHERITED, NOT TUNED -- 1e-2 was chosen for 19.8M parameters and 26,624 is a
# different optimisation problem. Kept identical so this is ONE variable against 2u8tf4. If the
# curve is flat, lr is the FIRST thing to vary before concluding anything about the method.
#
# HOW TO READ IT, AND THE TRAP: a flat val10 curve is as consistent with underfitting as with a
# genuine ceiling. The TRAINING PAE curve is what separates them. If train PAE also barely moves,
# 26,624 parameters were too few and the method was never tested. The hypothesis being tested is
# train improves LESS, val10 improves MORE and stays stable LONGER.
#
# --prefetch-workers 4 RUNS FOR THE FIRST TIME HERE. It has never executed on a GPU. Its worst
# failure is a silent deadlock at an epoch boundary, hanging against --dist-timeout-min 240, so a
# 15-step epoch is the right place to find out: this arm crosses a boundary every ~2.5 min and
# will have exercised _pf_start/_pf_end more times in the first hour than a full-set smoke test
# would in a day. WATCH THE FIRST THREE BOUNDARIES. The known deadlock case is already guarded --
# the queue drains BETWEEN joins, because a worker blocked on a full put() never sees the stop flag.
#
# FRESH FROM THE SHIPPED HEAD (no --resume), as the hypothesis requires.
# 30 structures x floor(10/3) = 90 group(s) / K=6 = 15 step(s)/epoch, 90 unit(s)/rank, 1,350 poses.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
# THE GUARD READS GPU OCCUPANCY, NOT COMMAND-LINE TEXT. A text check cannot work here: the
# string it searches for is present in the SEARCHING command's own argv, so both `grep` and a
# /proc cmdline scan report processes that are nothing but the check itself. That is what made
# the queued launch refuse against an idle box at 07:51:01. cards_busy.py counts compute apps on
# the cards the three GPU holders sit on, excluding the holders.
BUSY=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py)
if [ "$BUSY" != "0" ]; then
    echo "REFUSING: $BUSY process(es) occupy cards 4,5,7 besides the holders."
    exit 1
fi
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "POSE-FOCUSED ADAPTATION on the 30-structure set against val10, fresh from the shipped head. Trains only linear_no_bias_d, linear_no_bias_d_wo_onehot and linear_no_bias_pae -- the complete pose pathway plus the PAE readout, 26,624 parameters, 0.135% of the head; the four pairformer blocks and everything else frozen. ONE VARIABLE against 2u8tf4, full-head FT3 on these same 30 structures at this same lr, which went train -38% / val10 +51% over 142 epochs. Per-epoch val10 PAE and DockQ, which the control never had. lr 1e-2 is inherited from a 19.8M-parameter problem and is NOT tuned for 26,624. ALSO the first run of --prefetch-workers 4." \
  -- --resume training/runs/ggfhj5/resume_ep11.pt \
  --resume-note "resumed after --prefetch-workers 4 deadlocked mid-epoch 12: ranks alive at 0% CPU, all three cards at 0% utilisation, stuck at 3 of 15 steps with every thread in futex_wait. Single-worker prefetch is the path that ran 26 epochs of ocqtyt and 11 here without incident." \
  --train-only 'linear_no_bias_d,linear_no_bias_d_wo_onehot,linear_no_bias_pae' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 150 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 15 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 \
  --val-loss-every 1 \
  --val-loss-fold validation/fold \
  --val-loss-selection training/val10/structures.csv \
  --val-loss-pool training/val10/pool.csv \
  --val-loss-preflight training/val10/alignment_preflight.json \
  --val-loss-seeds 400000-400029 \
  --val-loss-prep validation/prep \
  --train-eval-pool training/val10/pool.csv \
  --train-eval-fold validation/fold \
  --train-eval-seed-base 400000 \
  --train-eval-budget 30 \
  --train-eval-every 1 \
  --eval-cards 4,5,7 \
  --blocks-per-ckpt 0 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
echo "launcher returned $?"
