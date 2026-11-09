#!/bin/bash
# MEMORY GATE for the full-head five-site arm, BEFORE committing three cards to it.
#
# WHY THIS CANNOT BE SKIPPED. Every peak figure this project has is from the FROZEN graph, where
# blocks 0-2 build no autograd graph at all: 1,650 tok 54.57 GB, 2,004 tok 79.64 GB, 2,289 OOM.
# This arm trains all four blocks, so those numbers do not transfer. The only full-backward fit we
# have (tvqobh) is peak_GB = 2.02 + 2.896e-05 * tokens^2, which puts 2,004 tokens at 118 GB on a
# 94.97 GB card -- but that fit was measured with checkpointing OFF, and this arm runs
# --blocks-per-ckpt 1 at --ckpt-token-threshold 0, i.e. checkpointed for every unit. Whether
# recompute buys back the difference is the ONE thing this probe answers.
#
# It is read-only with respect to the experiment: one rank, one card, --limit-units, no optimiser
# step that matters, writes only into its own run directory.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
SD=/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad
CARD="${CARD:-4}"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

# ---- GATE: the card must be genuinely idle. Never probe next to a live arm. ----
B=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py) || B=fail
if [ "$B" != "0" ]; then say "REFUSING: cards_busy reports '$B'. The probe would compete with a live run."; exit 3; fi
F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $CARD 2>/dev/null)
case "$F" in ''|*[!0-9]*) say "REFUSING: GPU $CARD memory unreadable"; exit 3;; esac
[ "$F" -lt 80000 ] && { say "REFUSING: GPU $CARD only ${F} MiB free"; exit 3; }
say "card $CARD idle with ${F} MiB free"

OUT=training/runs/memprobe_fullhead
mkdir -p $OUT
say "probing the FULL-HEAD five-site config, one rank on card $CARD, 12 units"
CUDA_VISIBLE_DEVICES=$CARD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ebrahim/Protenix/.venv/bin/python -u phases/phase_k_train.py \
  --rank-loss --rank-form rel --rel-labels training/fold/itm_labels_extended \
  --rank-cross-seed --rank-structs-per-step 2 --alpha-rank 75 --rank-pool mean \
  --fold training/fold --indices training/prep/all857_indices.csv \
  --manifest-selection train_val_splits/cache_train_structures_extended.csv \
  --manifest-skip 9mx1,4zto,9q5m,9dzf,9dzg \
  --pae-scored-pairs training/combined/pool.csv \
  --pae-scored-roles training/fold/chain_roles_all.csv \
  --pae-only --lean-readouts --minimal-items \
  --alignment-preflight training/alignment_preflight_sharded.json \
  --epochs 1 --lr 1e-3 --warmup-steps 20 --seed 0 --accum 30 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 \
  --prefetch-units 2 --prefetch-workers 2 \
  --blocks-per-ckpt 1 --ckpt-token-threshold 0 \
  --dist-backend gloo --dist-timeout-min 240 --max-rss-gb 80 --preload-min-free-gb 100 \
  --allow-code-drift --mem-log --trunk-memmap --lean-single --trunk-cache-gb 4 \
  --skip-dead-ops --pose-perblock --pose-perblock-copy --frame-direction --pose-readout \
  --limit-units 12 \
  --out $(pwd)/$OUT > $SD/mem_probe_fullhead.log 2>&1
RC=$?
say "probe exit $RC"

echo
say "=== what the five sites reported at install ==="
grep -aE "POSE READOUT|pose per-block|pose direction|DEAD OPS|LEAN SINGLE" $SD/mem_probe_fullhead.log | head -6
echo
say "=== TRAINABLE SET (the 96-tensor / 5.33M claim, checked against the trainer) ==="
grep -aE "trainable|head tensor\(s\)" $SD/mem_probe_fullhead.log | head -4
echo
say "=== PEAK BY TOKEN COUNT, pass B (the gate) ==="
grep -a "pass B" $SD/mem_probe_fullhead.log \
  | sed -E 's/.* tokens ([0-9]+):.*peak ([0-9.]+).*/\1 \2/' \
  | awk '{if($2>m[$1])m[$1]=$2} END{for(t in m) printf "%6d tokens   peak %7.2f GB\n",t,m[t]}' | sort -n
echo
say "=== any failure ==="
grep -aE "out of memory|CUDA error|STOP:|Traceback" $SD/mem_probe_fullhead.log | head -5
[ $RC -eq 0 ] && say "VERDICT: the probe completed. Read the peaks above against 94.97 GB." \
              || say "VERDICT: the probe FAILED -- see the lines above."
