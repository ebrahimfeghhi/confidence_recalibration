#!/bin/bash
# BACKFILL a missing held-out PAE point from a resume bundle.   usage: backfill_val_pae.sh <epoch>
#
# WHY EPOCH 7 OF ocqtyt IS MISSING. A mandatory --resume-note was added to the resume guard while
# the trainer was running. The trainer keeps sending the argument list its in-memory code builds,
# but the val subprocess it spawns reads phase_k_train.py FRESH FROM DISK -- so it got the new
# code demanding a note the old argument list does not pass, and refused. --loss-only is exempt
# from that rule now, so epoch 8 onward is fine; epoch 7's point simply never got written.
#
# DO NOT RUN THIS WHILE TRAINING IS MID-EPOCH. The epoch-boundary val pass runs right after
# empty_cache() hands back ~28 GB; mid-epoch the cards sit at ~70 of 97.9 GB with the trainer's
# pool already at its peak, and an OOM there kills the RUN, not just the measurement. Run it when
# the cards are free, or at an epoch boundary.
#
# The settings below are the ones phase_k_train.py's own val_loss_checkpoint passes, so the
# number is comparable to the rest of the curve. A different crop or alpha would not be.
set -u
EP="${1:?usage: backfill_val_pae.sh <epoch>}"
RID="${RID:-ocqtyt}"
cd /home/ebrahim/confidence_recalibration/extra_training
B="training/runs/$RID/resume_ep$EP.pt"
[ -f "$B" ] || { echo "no bundle $B"; exit 1; }
OUT="training/runs/$RID/backfill_ep$EP"
mkdir -p "$OUT"
BUSY=$(ps -eo cmd | grep -cE "python3? -u .*phase_k_train\.py" || true)
if [ "$BUSY" -gt 0 ]; then
    echo "REFUSING: $BUSY training process(es) are live and this would share their cards."
    echo "Run it when they are stopped, or export FORCE=1 if you have checked the headroom."
    [ "${FORCE:-0}" = "1" ] || exit 1
fi
LAYERNORM_TYPE=openfold CUDA_VISIBLE_DEVICES=7 \
/home/ebrahim/Protenix/.venv/bin/python -I phases/phase_k_train.py \
  --loss-only --resume "$B" --resume-allow-drift \
  --resume-note "backfill of the held-out PAE point epoch $EP never got, after a guard change killed the in-flight monitor" \
  --fold validation/fold --labels validation/fold/labels \
  --manifest-selection training/val10/structures.csv \
  --alignment-preflight training/val10/alignment_preflight.json --minimal-items \
  --pae-scored-pairs training/val10/pool.csv \
  --seeds 400000-400029 \
  --crop-margin 10 --crop-contact 5 --crop-size 3400 --accum 85 --nsample 5 \
  --alpha-pae 1.0 --alpha-confidence 1.0 --clip-norm 25.0 --blocks-per-ckpt 0 \
  --epochs 1 --save-every 0 --warmup-steps 0 --pae-only \
  --shuffle structure --out "$OUT" \
  --indices validation/prep/job_indices.csv \
  --bio-dir validation/prep/job_bioassembly \
  --mmcif-dir validation/prep/job_cifs \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift 2>&1 | tee "$OUT/backfill.log"
echo
echo "To put it on the curve, append to training/runs/$RID/val_loss.csv:"
echo "  $EP,<step>,<n_poses>,<val_pae>,resume_ep$EP.pt"
