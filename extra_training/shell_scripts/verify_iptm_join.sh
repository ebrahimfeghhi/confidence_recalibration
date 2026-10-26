#!/bin/bash
# Re-establish the (iface, seed, sample) JOIN before anything merges the two held-out passes.
#
#   ./shell_scripts/verify_iptm_join.sh <rid> <epoch>
#   e.g. ./shell_scripts/verify_iptm_join.sh dntxmn 16
#
# WHY IT HAS TO BE REDONE. dockq_from_iptm.py says "THE ARITHMETIC IS PROVEN, THE JOIN IS NOT":
# phase_k_eval maps poses to DockQ by COORDINATE IDENTITY against the _sample_<k>.cif files,
# because sample ordering is not guaranteed, while the rebuild uses a naive (iface, seed, sample)
# key. Its docstring claims the two agree on all 13,800 rows of j583i8 epoch 1 -- but THAT WAS THE
# CIRCULAR EVIDENCE the review caught as P1 in f1268fe5: the emitted `sample` column held the pose
# index rather than sample_index, and sample_index is non-identity on 396 of 400 label files, so
# 73.8% of rows would have carried another sample's DockQ. f1268fe5 fixed the column. The join has
# not been checked since, so that claim is stale and cannot carry a deletion.
#
# WHY IT MUST BE UNCROPPED. --compare checks PER-POSE ipTM against the eval's own rows, and the
# eval is uncropped. A cropped pass sees different inputs, so every pose would differ for a
# legitimate reason and a join error would be invisible underneath. With the inputs identical, the
# head identical and the weights identical, ipTM must agree to float precision -- so ANY
# difference is the join. That is what makes this check non-circular.
#
# IT MEASURES, IT CHANGES NOTHING. A --loss-only pass reads the bundle from disk and writes no
# checkpoint, and this runs against an epoch whose eval shards already exist, so no arm is touched
# and nothing is deleted. It needs the cards free: it loads a 464M-parameter model.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
RID="${1:?usage: verify_iptm_join.sh <rid> <epoch>}"
EP="${2:?usage: verify_iptm_join.sh <rid> <epoch>}"

BUNDLE="training/runs/$RID/resume_ep${EP}.pt"
SHARDS="training/runs/$RID/train_eval_shards_ep${EP}"
OUT="training/runs/${RID}_iptm_join_ep${EP}"
[ -f "$BUNDLE" ] || { echo "REFUSING: $BUNDLE does not exist"; exit 1; }
[ -d "$SHARDS" ] || { echo "REFUSING: $SHARDS does not exist -- nothing to compare against"; exit 1; }
ls "$SHARDS"/rows_b30_*.csv >/dev/null 2>&1 || {
  echo "REFUSING: no rows_b30_*.csv under $SHARDS"; exit 1; }
[ -e "$OUT" ] && { echo "REFUSING: $OUT exists; move it or pick another epoch"; exit 1; }

BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY). This loads a second model."; exit 1; }

# The run's own geometry EXCEPT the crop, and the run's own pool and seeds, read from its args.json
# so this cannot drift from what the arm actually measured.
read -r FOLD SEL PRE POOL SEEDS PREP ACCUM NSAMPLE < <($PY_BIN -I -c "
import json; a=json.load(open('training/runs/$RID/args.json'))
print(a['val_loss_fold'], a['val_loss_selection'], a['val_loss_preflight'],
      a['val_loss_pool'], a['val_loss_seeds'], a['val_loss_prep'], a['accum'], a['nsample'])")

# THE ARCHITECTURE FLAGS, MIRRORING WHAT THE TRAINER'S OWN MONITOR PASSES ITS CHILD. Omitting
# them was a review blocker and the worst of the four: dntxmn uses --pose-perblock-copy, so
# without these the bundle's pose tensors would load into a model that has no copied pathway,
# and the ipTM comparison would then measure a DIFFERENT ARCHITECTURE. Every difference would be
# attributed to the pose join, which is the one thing this script exists to isolate.
# --resume-allow-drift is not a substitute for building the same model.
ARCH=$($PY_BIN -I -c "
import json; a=json.load(open('training/runs/$RID/args.json'))
f=[]
if a.get('pae_only'): f.append('--pae-only')
if a.get('drop_pde'): f.append('--drop-pde')
if a.get('frame_direction'): f.append('--frame-direction')
if a.get('pose_perblock'):
    f.append('--pose-perblock')
    if a.get('pose_perblock_tied'): f.append('--pose-perblock-tied')
    if a.get('pose_perblock_share'): f.append('--pose-perblock-share')
    if a.get('pose_perblock_copy'): f.append('--pose-perblock-copy')
if a.get('train_only'): f += ['--train-only', a['train_only']]
if a.get('lora_rank'): f += ['--lora-rank', str(a['lora_rank'])]
print(' '.join(f))")
PREPARGS=""
if [ -n "$PREP" ] && [ "$PREP" != "None" ]; then
  PREPARGS="--indices $PREP/job_indices.csv --bio-dir $PREP/job_bioassembly --mmcif-dir $PREP/job_cifs"
fi
echo "verifying the join for $RID epoch $EP"
echo "  bundle $BUNDLE"
echo "  pool   $POOL  seeds $SEEDS"
echo "  arch   $ARCH"
echo "  UNCROPPED (--crop-margin 0), emitting ipTM rows, writing no checkpoint"
[ -n "$ARCH" ] || echo "  NOTE: args.json records no architecture flags for this run"

mkdir -p "$OUT"
LAYERNORM_TYPE=openfold CUDA_VISIBLE_DEVICES=4,5,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY_BIN -I phases/phase_k_train.py \
  --loss-only --resume "$BUNDLE" --resume-allow-drift \
  --resume-note "join verification of $RID epoch $EP; --loss-only reads the weights and writes no checkpoint" \
  --fold "$FOLD" --labels "$FOLD/labels" \
  --manifest-selection "$SEL" --alignment-preflight "$PRE" --minimal-items \
  --pae-scored-pairs "$POOL" --seeds "$SEEDS" \
  --crop-margin 0 --crop-size 3400 --accum "$ACCUM" --nsample "$NSAMPLE" \
  --alpha-pae 1.0 --emit-iptm-rows "$OUT/iptm_rows.csv" \
  --blocks-per-ckpt 0 --epochs 1 --save-every 0 --warmup-steps 0 --shuffle structure \
  $ARCH $PREPARGS \
  --out "$OUT" > "$OUT/pass.log" 2>&1
rc=$?
echo "  loss-only pass returned $rc"
[ "$rc" = "0" ] || { echo "  see $OUT/pass.log"; exit $rc; }
[ -s "$OUT/iptm_rows.csv" ] || { echo "REFUSING: $OUT/iptm_rows.csv is empty"; exit 1; }

echo
echo "=== join check: rebuilt metric vs the eval's own, and PER-POSE ipTM ==="
# THE EXIT STATUS IS THE RESULT. dockq_from_iptm.py now fails on an ipTM disagreement beyond
# --iptm-tol, on mismatched interface sets, on a selected-pose disagreement or on an aggregate
# mismatch -- and this propagates it. Before, the comparison printed its verdict and the script
# ended on an `echo`, so a FAILED comparison returned 0 and no automated caller could tell a
# verified join from a broken one. Review blocker; set -u does not imply set -e.
$PY_BIN -I run_logging/dockq_from_iptm.py "$OUT/iptm_rows.csv" \
  --results "$FOLD/results.csv" --compare "$SHARDS"
rc=$?
echo
if [ "$rc" = "0" ]; then
  echo "JOIN VERIFIED for $RID epoch $EP."
else
  echo "JOIN NOT VERIFIED (exit $rc). The separate DockQ eval must stay."
fi
exit $rc
