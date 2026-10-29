#!/bin/bash
# ROW-WEIGHTED PAE + WINNABLE STRUCTURES ONLY + COSINE LR. Fresh run, 60 epochs, 3 cards.
#
# WHAT IS NEW, three changes at once -- stated up front because it means NOTHING in this run is
# cleanly attributable to any one of them:
#
#   1. --pae-row-frac 0.2 --pae-row-mass 0.8. The PAE term is reweighted toward the anchor rows
#      ipTM can read. ipTM is max_i t_i over 29-801 rows (median 332), so selection reads ONE row
#      while the unweighted term grades all of them equally. The top 20% of rows by t_i (floor 10)
#      now carry 80% of the loss and the rest share 20%: a top row weighs 16x a bottom row and
#      effective sample size is ~103 of 332. It does NOT change any label -- every pair is still
#      trained toward its true error bin -- which is why it was preferred to a ranking loss on
#      ipTM. See phases/pae_row_weight.py and run_logging/test_pae_row_weight.py.
#
#   2. structures_winnable.csv: 368 structures, not 487. The 119 dropped have NO pose above DockQ
#      0.23 over all 100 labelled poses, so every cross-chain error exceeds 32 A, the PAE label is
#      the top bin everywhere and "say maximum error" scores well with no error detection at all.
#      MEASURED on 45q96x epochs 55+: those structures sit at PAE 1.1023 against 1.9592 for the
#      rest, and they are 25% of poses -- so the 1.73 the unweighted arms report is 0.21 lower than
#      the number for the structures that can actually be won. EXPECT THIS RUN TO START NEAR 1.96
#      AND READ AGAINST THAT, not against 1.73. run_logging/plot_train_pae.py --winnable-only
#      replots the old arms on the same population.
#
#   3. Cosine lr: 1e-2 held to epoch 20, then down to 1e-4 by epoch 50, flat for 51-60. The
#      hand-stepped version of this (fawgtu 1-40 at 1e-2, 41-50 at 1e-3, 51-60 at 1e-4) is what
#      the schedule is modelled on; each manual step bought one good epoch and then decayed.
#
# WHAT IS DELIBERATELY UNCHANGED from launch_fullhead_noanchor_40ep.sh, so the three changes above
# are the only ones: same fold, skip list, crop 3400, margin 10, contact 5, accum 85, nsample 5,
# seeds 500000-500009, warmup 20 steps, --pae-only, --blocks-per-ckpt 0, no L2, no evals.
#
# --frame-direction IS ON. It went into the lineage at epoch 48 of tvden2 and its read was already
# confounded there; it is kept so this arm matches the configuration 45q96x ended in. 258
# trainable tensors, not 254.
#
# WHY A FRESH RUN rather than a resume from 45q96x/resume_ep60.pt: that model spent 60 epochs
# partly fitting the 119 structures this arm drops, and the cosine schedule is defined from epoch
# 0. Resuming would start the schedule in its own past.
#
# THE LAUNCH LOG MUST PRINT, or this is not the arm it claims to be:
#   - "PAE term ROW-WEIGHTED: top 20% of anchor rows by t_i (floor 10 rows) carry 80% ..."
#   - "368 structure(s)" on the manifest line and 738 interfaces on the PAE mask line
#   - 258 trainable tensors
#   - NO "anchor SNAPSHOT" line (that would mean --l2-sp survived)
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
$PY_BIN -I run_logging/test_pae_row_weight.py || {
  echo "REFUSING: the row-weight correctness gate failed"; exit 1; }
echo "launching row-weighted PAE, 368 winnable structures, cosine 1e-2 -> 1e-4 over epochs 20-50"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --numa-map 0:0,1:0,2:1 --note "ROW-WEIGHTED PAE + WINNABLE STRUCTURES ONLY + COSINE LR, fresh 60-epoch run on cards 4,5,7, no evals, checkpoints only. THREE CHANGES AT ONCE, so nothing here is cleanly attributable to one of them -- said plainly rather than discovered at analysis time. (1) --pae-row-frac 0.2 --pae-row-mass 0.8 --pae-row-floor 10: the PAE term is reweighted toward the anchor rows ipTM can read, because ipTM is max_i t_i over 29-801 rows (median 332) and selection reads ONE row while the unweighted term grades every row equally. Top 20% of rows by t_i carry 80% of the loss, the rest share 20%, so a top row weighs 16x a bottom row and Kish effective sample size is about 103 of 332. Mass is deliberately NOT 1.0: zero-weight rows would be unsupervised and free to drift over-confident, and since ipTM is a max an over-confident row is exactly the one selected. THE LABELS ARE UNTOUCHED -- every pair still trains toward its own true error bin -- which is why this was preferred to a ranking loss on ipTM, which is shift-invariant and so cannot learn absolute confidence level and can be lowered by distorting ipTM away from calibration. Weights are DETACHED. Gate: phases/pae_row_weight.py is tested by run_logging/test_pae_row_weight.py, which asserts max(row_t) equals Protenix's own calculate_iptm (exact, 0.00e+00) and that --pae-row-frac 1.0 reproduces the unweighted term bit for bit; the launch script runs it and refuses on failure. (2) 368 structures, not 487: the 119 dropped have no pose above DockQ 0.23 across all 100 labelled poses, so every cross-chain error exceeds 32 A, the PAE label is the top bin everywhere, and 'say maximum error' scores well with no error detection. Measured on 45q96x epochs 55+ they sit at PAE 1.1023 vs 1.9592 for the rest at 25% of poses, so the 1.73 the unweighted arms report is 0.21 below the number for winnable structures. THIS RUN SHOULD START NEAR 1.96 AND BE READ AGAINST THAT. Cost stated: those 119 are the only structures where the correct answer is 'I am wrong, be unconfident', so this arm cannot speak to out-of-distribution calibration -- it is a selection experiment. (3) Cosine lr 1e-2 held flat to epoch 20, decaying to 1e-4 by epoch 50, flat 51-60, replacing the hand-stepped 1e-2/1e-3/1e-4 of the fawgtu lineage where each manual step bought one good epoch then decayed. --frame-direction stays on so this matches the configuration 45q96x ended in, hence 258 trainable tensors not 254. Everything else is verbatim from launch_fullhead_noanchor_40ep.sh. NO L2, no --val-loss-* monitor, no --train-eval-* pass: nothing in this run can see overfitting while it happens, so the checkpoints must be scored afterwards rather than assumed good. The diagnostic that can answer the premise without waiting for DockQ is pae_rows_rank*.csv, written every 50th pose: if rho between t_i and that row's own PAE loss shows the high-t_i rows are already the best fit, this weighting has no headroom." \
  -- --pose-perblock --pose-perblock-copy \
  --frame-direction \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight_winnable.json \
  --manifest-selection training/full_train/structures_winnable.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 60 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --lr-final 1e-4 --lr-decay-shape cosine \
  --lr-decay-start-epoch 20 --lr-decay-end-epoch 50 \
  --pae-row-frac 0.2 --pae-row-mass 0.8 --pae-row-floor 10 --pae-row-log-every 50 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 250 \
  --pae-scored-pairs training/full_train/pool_winnable.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 --prefetch-workers 4 \
  --blocks-per-ckpt 0 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
rc=$?
echo "launch_rowweight_cosine returned $rc"
exit $rc
