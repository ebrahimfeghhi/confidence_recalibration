#!/bin/bash
# PAE READOUT UNFROZEN, pair_transition FROZEN. One question: is the PAE pathway the limit?
#
# Usage:  ./shell_scripts/launch_readout.sh <lambda> <alpha_rank> <alpha_pae>
#
# WHY. oe6sru ran the identical objective with the readout FROZEN and did not learn the PAE loss
# at all: held-out PAE went 2.9085 -> 2.9029 over seven epochs, a 0.2% change, while bouncing by
# more than that between epochs (2.9863 at epoch 2). Training PAE did the same, 2.29-2.56 with no
# trend and rising at epochs 3, 5 and 7. Both terms were stuck, not just the ranking one.
#
# The structural suspect is that pae_ln and linear_no_bias_pae were not trainable, so the map
# from z_pair to PAE logits was fixed and the only route to the PAE loss was through z -- the
# same narrow pathway the ranking loss has failed to move in every arm of this project. This arm
# opens that map directly.
#
# WHAT CHANGES against oe6sru:
#   pairformer_stack.blocks.3.pair_transition   REMOVED from --train-only  (-786,944 params)
#   pae_ln.weight, pae_ln.bias                  ADDED, trainable                 +512
#   linear_no_bias_pae.weight (64,256)          ADDED, trainable              +16,384
#   15 tensors / 60,928 parameters, against oe6sru's 17 / 830,976.
#
# L2-SP COVERS THE READOUT BUT NOT THE LAYERNORM. --l2-sp-exclude 'frame_dir,pae_ln' leaves
# pae_ln trainable and out of the anchor while linear_no_bias_pae stays anchored, so 9 tensors /
# 57,344 parameters are priced against shipped. The readout is a pretrained 64x256 matrix and
# distance from shipped is a prior worth holding for it; a LayerNorm's scale and bias are 512
# numbers that set the scale the readout sees, and anchoring them fights the very adjustment this
# arm exists to allow. frame_dir stays excluded for the usual reason -- zero-init, so anchoring
# to its own init is plain decay toward zero.
#
# ALL THREE WEIGHTS ARE ARGUMENTS, measured by probe_readout.sh at THIS trainable set and passed
# in. None is defaulted: unfreezing a 16,384-parameter readout changes both |g_PAE| and |g_rank|,
# so neither oe6sru's alpha_rank 5.347 nor its lambda 5e-04 carries over.
#
# WHAT WOULD FALSIFY IT: held-out PAE still flat. That would move the suspicion off the frozen
# readout and onto the objective or the data, since at that point every pathway into the PAE
# head would have been opened and none of them moved it.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
LAM="${1:-}"; AR="${2:-}"; AP="${3:-}"
[ -n "$LAM" ] && [ -n "$AR" ] && [ -n "$AP" ] || {
    echo "usage: $0 <lambda> <alpha_rank> <alpha_pae>   (all measured by probe_readout.sh)"; exit 1; }
BUSY=$($PY_BIN -I run_logging/cards_busy.py)
[ "$BUSY" = "0" ] || { echo "REFUSING: $BUSY process(es) occupy cards 4,5,7."; exit 1; }
echo "launching readout arm: lambda=$LAM alpha_rank=$AR alpha_pae=$AP"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --note "PAE READOUT UNFROZEN, pair_transition FROZEN, lambda $LAM, alpha_rank $AR, alpha_pae $AP, 15 epochs. THE QUESTION: is the PAE pathway the limit? oe6sru ran this exact objective with the readout FROZEN and DID NOT LEARN THE PAE LOSS AT ALL -- held-out PAE 2.9085 -> 2.9029 over seven epochs, a 0.2% change, while bouncing more than that between epochs (2.9863 at epoch 2); training PAE 2.29-2.56 with no trend, rising at epochs 3, 5 and 7. Both terms were stuck, not only the ranking one. The structural suspect is that pae_ln and linear_no_bias_pae were frozen, fixing the map from z_pair to PAE logits so the only route to the PAE loss was through z -- the same narrow pathway every ranking arm in this project has failed to move. WHAT CHANGES against oe6sru: pairformer_stack.blocks.3.pair_transition is REMOVED from --train-only (-786,944 parameters) and pae_ln.weight+bias (+512) and linear_no_bias_pae.weight 64x256 (+16,384) are ADDED, giving 15 tensors / 60,928 parameters against oe6sru's 17 / 830,976. L2-SP COVERS THE READOUT BUT NOT THE LAYERNORM: --l2-sp-exclude 'frame_dir,pae_ln' leaves pae_ln trainable and out of the anchor while linear_no_bias_pae stays anchored, so 9 tensors / 57,344 parameters are priced against shipped. The readout is a pretrained 64x256 matrix and distance from shipped is a prior worth holding for it; a LayerNorm's 512 scale-and-bias numbers set the scale that readout sees, and anchoring them would fight the adjustment this arm exists to allow. frame_dir stays excluded because it is zero-init and anchoring it to its own init is plain decay toward zero. ALL THREE WEIGHTS MEASURED AT THIS TRAINABLE SET by probe_readout.sh, none defaulted: unfreezing a 16,384-parameter readout changes both |g_PAE| and |g_rank|, so neither oe6sru's alpha_rank 5.347 nor its lambda 5e-04 carries over. CONTROLS: oe6sru (same objective, readout frozen, pair_transition trainable) and 24u342 (same pool, cross-seed ranking only, no PAE, 50 epochs). NOTE ntwjfp and qiodxz trained pae_ln and linear_no_bias_pae before, but with the pre-flag LINEAR within-seed loss, no tau, no L2-SP and no orientation input, so neither is a control for this. CRITERION: selected DockQ and fraction above 0.23 over the 18 val10 interfaces against shipped DockQ 0.2909 / frac 0.5556 / oracle 0.5968, from this run's own shards; held-out PAE from val_loss.csv is the secondary diagnostic this arm was built to move. TEN-EPOCH MINIMUM. FALSIFIER: held-out PAE still flat, which would move the suspicion off the frozen readout and onto the objective or the data, every pathway into the PAE head having then been opened." \
  -- --pose-perblock --pose-perblock-copy --frame-direction --l2-sp "$LAM" --l2-sp-exclude 'frame_dir,pae_ln' \
  --rank-loss --alpha-rank "$AR" --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
  --rank-pairs cross \
  --train-only 'linear_no_bias_d,pose_perblock,frame_dir,pae_ln,linear_no_bias_pae' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 15 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae "$AP" --clip-norm 25.0 --crop-size 3400 \
  --save-every 15 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 --prefetch-workers 3 \
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
echo "launch_readout returned $?"
