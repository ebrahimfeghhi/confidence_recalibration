#!/bin/bash
# THE PAE-ONLY CROPPED ARM. Our own 256-token crop pool, PAE loss alone, no ranking loss,
# 48 distinct structures per optimiser step, 50 epochs, held-out eval every 5 epochs UNCROPPED.
#
# WHY THIS ARM EXISTS. Every cropped attempt so far carried the relative-iTM ranking loss, and
# that loss is what made cropping fragile: it scores ipTM on a NAMED chain pair, so a crop that
# removes that chain is fatal, and the pool has to be validated against whichever join the
# trainer will use. That is exactly what killed 3pdv7i (pool validated against the static
# chain_roles join, trainer scored the per-pose OST join, they disagree on 90 of 846 structures).
#
# WITH PAE ALONE THAT WHOLE CLASS OF FAILURE NARROWS TO ONE CHECKABLE CONDITION. The PAE term is
# a per token-pair cross-entropy, self-consistent under any crop. It is MASKED to scored chain
# pairs (--pae-scored-pairs), and that mask resolves chains through chain_roles -- the SAME
# STATIC JOIN build_crop_pool.py validated training/crop_final against (phase_k_train.py:2096 vs
# build_crop_pool.py's interface_chain_pairs). Pool and consumer therefore agree by construction,
# and check_crop_pairs.py --join static asserts it below rather than trusting the argument.
#
# WHAT IS DIFFERENT FROM THE RANKING ARMS, and all of it:
#   --pae-only                 pLDDT, PDE and resolved stubbed out; ipTM derives from PAE alone
#   no --rank-loss             no ranking objective, so no pass A: one forward per pose, not two
#   no --rank-cross-seed       frees the epoch builder to shuffle over ALL structures
#   --accum 144 --nsample 3    48 structures x 3 poses per step, 16 structures per rank
#   --epochs 50
#
# THE BATCH, AND WHY IT IS SHAPED THIS WAY. structs_per_step = accum / (world * nsample), so
# 144 / (3 * 3) = 16 structures per rank, 48 distinct structures per gradient -- against 2 under
# --rank-cross-seed, which needed many poses of ONE structure to form a ranking group.
# WHOLE STRUCTURES GO TO ONE RANK. Splitting a structure's poses across ranks would make all
# three load the same ~100 MB trunk, and measured rank imbalance at 16 structures per rank is
# already ~0.5% because the 256-token crop makes almost every unit the same size. There is
# nothing left to balance and three times the bytes to pay for.
#
# HELD-OUT EVAL IS UNCROPPED, every 5 epochs, so every historical number stays comparable.
#
# DRY=1 (default) prints the argv and runs every gate without touching a card.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
PY=/home/ebrahim/Protenix/.venv/bin/python
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

POOL_DIR="${POOL_DIR:-training/crop_final}"
MANIFEST="${MANIFEST:-train_val_splits/cache_train_structures_extended.csv}"
SKIP="${SKIP:-4zto,7m7i,9b9j,9dzf,9dzg,9e7x,9mx1,9q5m}"
CARDS="${CARDS:-4,5,7}"
EPOCHS="${EPOCHS:-50}"
ACCUM="${ACCUM:-144}"
NSAMPLE="${NSAMPLE:-3}"
EVAL_POOL="${EVAL_POOL:-training/valfull83/pool.csv}"
LR="${LR:-1e-3}"
DRY="${DRY:-1}"

say "PAE-only cropped arm: pool=$POOL_DIR cards=$CARDS epochs=$EPOCHS accum=$ACCUM nsample=$NSAMPLE DRY=$DRY"

[ -d "$POOL_DIR" ] || { say "REFUSING: $POOL_DIR is not a directory"; exit 2; }

# POPULATION, AS A SET NOT A COUNT.
$PY -I - "$POOL_DIR" "$MANIFEST" "$SKIP" <<'PYG' || exit 3
import csv, os, sys
pool_dir, manifest, skip_s = sys.argv[1:4]
skip = set(x for x in skip_s.split(",") if x)
man = [r["pdb"] for r in csv.DictReader(open(manifest))]
want = set(p for p in man if p not in skip)
pool = set(f[:-4] for f in os.listdir(pool_dir) if f.endswith(".npz"))
print(f"  manifest {len(man)} rows, SKIP removes {len(skip & set(man))} -> trainable {len(want)}")
print(f"  crop pool {len(pool)}   want == pool: {'YES' if want == pool else 'NO'}")
missing = want - pool
if missing:
    print(f"  REFUSING: {len(missing)} structure(s) absent from the pool: {sorted(missing)[:6]}")
sys.exit(0 if want == pool else 1)
PYG

say "CPU gate: wiring"
$PY -I run_logging/test_crop_pool_wiring.py || { say "REFUSING: wiring gates failed"; exit 4; }

# THE GATE 3pdv7i DID NOT HAVE. --join static because that is the join the PAE mask uses; the
# OST join is the ranking loss's and is NOT what this arm asks of the crop.
say "CPU gate: every crop keeps a scorable cross-chain pair under the STATIC join"
CROP_THREADS="${CROP_THREADS:-3}" $PY -I run_logging/check_crop_pairs.py \
    --pool "$POOL_DIR" --join static --roles training/fold/chain_roles_all.csv \
    --fold training/fold --scored-pairs training/combined/pool.csv \
    || { say "REFUSING: a selected crop leaves a scored pair unscorable"; exit 5; }

# COPIED FROM 87ja6a's args.json, NOT RETYPED. The bare negative lookahead alone gives 19 tensors
# and 0.3M parameters; the comma-separated list re-includes the pairformer blocks. Launched
# truncated once -- 19 tensors against 102 -- and stopped before the first optimiser step.
TRAIN_ONLY='^(?!pairformer_stack\.),blocks\.0\.pair_transition\.,blocks\.1\.tri_mul_out\.,blocks\.1\.tri_mul_in\.,blocks\.2\.tri_mul_out\.,blocks\.2\.tri_mul_in\.,blocks\.2\.pair_transition\.,blocks\.3\.tri_mul_out\.,blocks\.3\.tri_mul_in\.,blocks\.3\.tri_att_start\.,blocks\.3\.pair_transition\.'
ARGS=(--crop-pool "$POOL_DIR"
      --minimal-items
      --pae-only --alpha-pae 1.0 --alpha-confidence 1.0
      --accum "$ACCUM" --nsample "$NSAMPLE"
      --skip-dead-ops --pose-perblock --pose-perblock-copy --frame-direction --lean-single
      --train-eval-pool "$EVAL_POOL" --train-eval-fold validation/fold
      --train-eval-every 5 --train-eval-budget 30 --train-eval-seed-base 400000
      --epochs "$EPOCHS" --lr "$LR")

say "argv the trainer would receive:"
printf '    %s\n' "${ARGS[@]}"
say "NOT passed, deliberately: --rank-loss, --rank-form, --rank-cross-seed, --rel-labels,"
say "  --crop-margin, --crop-tokens, --eval-pool-crop-tokens (eval stays UNCROPPED)"

if [ "$DRY" = "1" ]; then
  say "DRY RUN -- nothing launched. Re-run with DRY=0 to go."
  exit 0
fi

B=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$B" = "0" ] || { say "REFUSING: cards_busy reports '$B'"; exit 3; }
for c in $(echo "$CARDS" | tr ',' ' '); do
  F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$c" 2>/dev/null)
  case "$F" in ''|*[!0-9]*) say "REFUSING: GPU $c unreadable"; exit 3;; esac
  [ "$F" -lt 80000 ] && { say "REFUSING: GPU $c has only ${F} MiB free"; exit 3; }
done
M=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
[ "${M:-0}" -lt 250 ] && { say "REFUSING: MemAvailable ${M} GB is under the 250 GB floor"; exit 3; }
say "gates clear: cards $CARDS free, MemAvailable ${M} GB"

NOTE="PAE-ONLY CROPPED ARM. Our own 256-token crop pool (training/crop_final), PAE loss alone \
with pLDDT/PDE/resolved stubbed out, NO ranking loss and therefore no pass A -- one forward per \
pose instead of two. The epoch builder is free of --rank-cross-seed, so a step draws 48 DISTINCT \
structures (accum 144 / (world 3 x nsample 3) = 16 structures per rank x 3 poses) against 2 \
structures under the ranking arms. Whole structures go to one rank: splitting a structure's \
poses across ranks would make all three load the same ~100 MB trunk, and measured rank imbalance \
at this batch size is already ~0.5% because the 256-token crop makes nearly every unit the same \
size. The PAE term is masked to scored chain pairs via chain_roles -- the SAME static join the \
crop pool was validated against -- so pool and consumer agree by construction, and \
check_crop_pairs.py --join static asserts it before launch. This is the join question that \
killed 3pdv7i, where the pool used static and the RANKING loss used the per-pose OST join. \
Held-out eval every 5 epochs on valfull83, UNCROPPED, so every historical number stays \
comparable. 50 epochs."

say "LAUNCH"
LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py --cards "$CARDS" \
  --numa-map 0:0,1:0,2:1 --note "$NOTE" -- "${ARGS[@]}" \
  --alignment-preflight training/alignment_preflight_sharded.json \
  --manifest-selection "$MANIFEST" --manifest-skip "$SKIP" \
  --fold training/fold --indices training/prep/all857_indices.csv \
  --pae-scored-pairs training/combined/pool.csv \
  --pae-scored-roles training/fold/chain_roles_all.csv \
  --lean-readouts --train-only "$TRAIN_ONLY" \
  --warmup-steps 20 --seed 0 --clip-norm 25.0 \
  --blocks-per-ckpt 1 --ckpt-token-threshold 1250 --mem-log \
  --prefetch-units 2 --prefetch-workers 4 \
  --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 250 --save-every 250 \
  --require-replay training/extended_mixed_replay.log --allow-code-drift
say "launch.py returned $?"
