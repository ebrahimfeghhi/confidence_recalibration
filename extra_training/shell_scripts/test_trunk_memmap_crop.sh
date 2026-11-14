#!/bin/bash
# DOES MAPPING z_trunk SPEED UP THE CROPPED PATH, and does it change the numbers?
#
# The cropped path read the WHOLE pair representation (~100 MB) and kept a 256x256 corner,
# throwing away ~97% of what it read. cpucrop_slice measures 117 min of thread time per rank per
# epoch, the largest single cost in the path. This runs the real trainer twice, identical but for
# --trunk-memmap, and reports both the timers and whether the loss moved.
#
# THE NUMBERS MUST NOT MOVE. A mapping that reads the same bytes lazily is a performance change,
# not a numerical one; if the loss differs at all, the slicing is wrong.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
PY=/home/ebrahim/Protenix/.venv/bin/python
CARD="${CARD:-0}"
EPOCHS="${EPOCHS:-0.025}"
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }
case ",4,5,7," in *",$CARD,"*) say "REFUSING: card $CARD is the live arm's"; exit 2;; esac
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CARD" 2>/dev/null)
case "$U" in ''|*[!0-9]*) say "REFUSING: card $CARD unreadable"; exit 2;; esac
[ "$U" -gt 2000 ] && { say "REFUSING: card $CARD holds ${U} MiB"; exit 2; }

TRAIN_ONLY='^(?!pairformer_stack\.),blocks\.0\.pair_transition\.,blocks\.1\.tri_mul_out\.,blocks\.1\.tri_mul_in\.,blocks\.2\.tri_mul_out\.,blocks\.2\.tri_mul_in\.,blocks\.2\.pair_transition\.,blocks\.3\.tri_mul_out\.,blocks\.3\.tri_mul_in\.,blocks\.3\.tri_att_start\.,blocks\.3\.pair_transition\.'

run_one () {   # $1 = extra flags, $2 = out
  rm -rf "$2"; mkdir -p "$2"
  CUDA_VISIBLE_DEVICES="$CARD" LAYERNORM_TYPE=openfold \
  $PY -I phases/phase_k_train.py \
    --crop-pool training/crop_final --minimal-items --epoch-cover-all \
    --pae-only --alpha-pae 1.0 --alpha-confidence 1.0 \
    --accum 240 --nsample 5 $1 \
    --alignment-preflight training/alignment_preflight_sharded.json \
    --manifest-selection train_val_splits/cache_train_structures_extended.csv \
    --manifest-skip '4zto,7m7i,9b9j,9dzf,9dzg,9e7x,9mx1,9q5m' \
    --fold training/fold --indices training/prep/all857_indices.csv \
    --pae-scored-pairs training/combined/pool.csv \
    --pae-scored-roles training/fold/chain_roles_all.csv \
    --lean-readouts --train-only "$TRAIN_ONLY" --expect-trainable 113 \
    --skip-dead-ops --pose-perblock --pose-perblock-copy --frame-direction --lean-single \
    --epochs "$EPOCHS" --lr 1e-3 --warmup-steps 0 --seed 0 --clip-norm 25.0 \
    --blocks-per-ckpt 1 --ckpt-token-threshold 1250 \
    --prefetch-units 2 --prefetch-workers 4 \
    --max-rss-gb 80 --preload-min-free-gb 100 --save-every 0 \
    --out "$2" --require-replay skip --allow-code-drift > "$2/log" 2>&1
  echo "  exit $?"
}

say "A: memmap OFF"; run_one "" training/runs/mm_off
say "B: memmap ON";  run_one "--trunk-memmap" training/runs/mm_on

say "RESULT"
for d in mm_off mm_on; do
  echo "--- $d ---"
  grep -ao "epoch 1 took [0-9.]* min.*s/pose" "training/runs/$d/log" 2>/dev/null | head -1
  grep -a "epoch 1 took" "training/runs/$d/log" 2>/dev/null | tail -1 \
    | grep -oE "(cpucrop_slice|cpucrop_trunk|unit_setup|crop_slice|trunk_memmap_fallback) [0-9.]+m/[0-9]+" | tr '\n' '  '
  echo
done
$PY -I - <<'PYG'
import csv
def mean(d):
    v=[]
    try:
        for r in csv.DictReader(open(f"training/runs/{d}/trajectory_rank0.csv")):
            try: v.append(float(r["pae_loss"]))
            except (TypeError,ValueError): pass
    except FileNotFoundError: return None, 0
    return (sum(v)/len(v) if v else None), len(v)
a,na=mean("mm_off"); b,nb=mean("mm_on")
print(f"  pae_loss mean: OFF {a} ({na} rows)   ON {b} ({nb} rows)")
if a and b:
    print(f"  relative diff: {abs(a-b)/abs(a):.3e}   {'IDENTICAL enough' if abs(a-b)/abs(a)<1e-6 else '*** NUMBERS MOVED ***'}")
PYG
