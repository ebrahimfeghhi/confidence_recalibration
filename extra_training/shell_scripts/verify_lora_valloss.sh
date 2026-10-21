#!/bin/bash
# DOES THE HELD-OUT MONITOR WORK FOR A LoRA BUNDLE? One --loss-only pass over val10 using a LoRA
# resume bundle, which is exactly what the per-epoch monitor does at every boundary.
#
# WHY THIS EXISTS. Arm m8rote trained three epochs and produced ZERO held-out points:
#   STOP: the bundle is missing 248 trainable tensor(s)
# A LoRA bundle holds 298 adapter tensors and no plain head tensors, and the monitor spawns itself
# without --lora-rank, so its trainable set was the 248 plain ones. Identical in shape to the
# --train-only failure that cost ggfhj5 its epoch-1 point earlier the same day; the LoRA exposure
# was not checked when that one was fixed.
#
# The fix is inheritance: under --loss-only, read lora_rank/alpha/targets/dropout from the
# bundle's own guard. This script PROVES it before a 4-hour run is launched on the assumption.
# PASS = the line "--loss-only: inheriting the bundle's LoRA configuration" appears AND a
# "VALLOSS: pae ... poses 1500" line is printed. Anything else is a fail.
set -u
B="${1:?usage: verify_lora_valloss.sh <lora-resume-bundle>}"
cd /home/ebrahim/confidence_recalibration/extra_training
OUT=training/runs/lora_val_verify
mkdir -p "$OUT"
BUSY=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py)
[ "$BUSY" = "0" ] || { echo "REFUSING: $BUSY process(es) on cards 4,5,7 besides the holders"; exit 1; }
LAYERNORM_TYPE=openfold CUDA_VISIBLE_DEVICES=7 \
/home/ebrahim/Protenix/.venv/bin/python -I phases/phase_k_train.py \
  --loss-only --resume "$B" --resume-allow-drift \
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
  --require-replay training/ft3_pilot_replay.log --allow-code-drift 2>&1 | tee "$OUT/verify.log"
echo
echo "=== VERDICT ==="
grep -c "inheriting the bundle's LoRA" "$OUT/verify.log" | sed 's/^/inheritance lines: /'
grep -E "VALLOSS|STOP" "$OUT/verify.log" | tail -3
