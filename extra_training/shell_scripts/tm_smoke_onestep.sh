#!/usr/bin/env bash
# ONE REAL TRAINING STEP OF A TM-ONLY ARM, ON A GPU. What the pure-function tests cannot reach.
#
# WHAT THIS PROVES, and why the unit tests do not. phases/test_tm_centered.py exercises the loss as
# a function: d0, disjoint blocks, the every-pose mask, the detached-mean gradient, the rank merge.
# It never runs _xs_prepare -> pass B -> optimizer. External review (ChatGPT, issue #16) found a
# blocker exactly there: pass A fell through into the RANKING group construction, where
# --rank-form's default of 'rel' calls _rel_labels and hard-fails without --rel-labels -- or worse,
# fills _xs_coef and lets pass B add a ranking term to an arm that claims the centred TM error is
# its only loss.
#
# SO THIS RUNS THE REAL PATH with --tm-centered --rank-cross-seed --pae-scored-pairs and WITHOUT
# --rank-loss or --rel-labels, for one epoch over a handful of structures, and asserts from the
# log: a nonzero finite objective, a gradient that reached the trainable tensors, an optimiser step
# taken, and no ranking coefficient created.
#
# IT ALSO SETTLES THE eps QUESTION WITH DATA. The claim that AdamW makes the loss scale irrelevant
# was argued from a GLOBAL gradient norm against a PER-PARAMETER eps, which is the wrong
# comparison -- the review is right about that. --grad-norm-every 1 prints the per-tensor norms, so
# the distribution can be read rather than asserted.
set -euo pipefail
cd "$(dirname "$0")/.."
CARD="${CARD:-4}"
OUT="${OUT:-training/runs/tm_smoke}"
LIMIT="${LIMIT:-6}"

head -1 training/slice280/structures.csv > /tmp/tm_smoke_sel.csv
tail -n +2 training/slice280/structures.csv | head -"$LIMIT" >> /tmp/tm_smoke_sel.csv
echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') TM-ONLY one-step smoke on card $CARD, $LIMIT structure(s)"

rm -rf "$OUT"
LAYERNORM_TYPE=openfold CUDA_VISIBLE_DEVICES="$CARD" \
/home/ebrahim/Protenix/.venv/bin/python -I phases/phase_k_train.py \
  --out "$OUT" --epochs 1 --save-every 1 --warmup-steps 0 --lr 1e-3 \
  --fold training/fold --labels training/fold/labels \
  --manifest-selection /tmp/tm_smoke_sel.csv \
  --alignment-preflight training/slice280/alignment_preflight.json \
  --pae-scored-pairs training/slice280/pool.csv \
  --pae-scored-roles training/slice280/chain_roles.csv \
  --indices "$PWD/training/prep/all857_indices.csv" \
  --bio-dir "$PWD/training/prep/job_bioassembly" \
  --mmcif-dir "$PWD/training/prep/job_cifs" \
  --minimal-items --epoch-cover-all --nsample 5 --blocks-per-ckpt 0 \
  --crop-contact 5 --crop-size 3400 \
  --pae-only --alpha-pae 1.0 --alpha-confidence 1.0 \
  --skip-dead-ops --pose-perblock --pose-perblock-copy --frame-direction --lean-single \
  --pose-readout --pose-readout-prenorm --pose-readout-copy-distance --pose-readout-post-zero \
  --lean-readouts --expect-trainable 119 \
  --train-only '^(?!pairformer_stack\.),blocks\.0\.pair_transition\.,blocks\.1\.tri_mul_out\.,blocks\.1\.tri_mul_in\.,blocks\.2\.tri_mul_out\.,blocks\.2\.tri_mul_in\.,blocks\.2\.pair_transition\.,blocks\.3\.tri_mul_out\.,blocks\.3\.tri_mul_in\.,blocks\.3\.tri_att_start\.,blocks\.3\.pair_transition\.' \
  --pae-split-accept 0.23 \
  --tm-centered --rank-cross-seed --rank-structs-per-step 3 \
  --accum 30 \
  --require-replay training/extended_mixed_replay.log --allow-code-drift "$@"
