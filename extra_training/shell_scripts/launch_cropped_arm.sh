#!/bin/bash
# THE CROPPED ARM. Same head, same loss, same population as 87ja6a -- the ONLY change is that each
# structure trains on a 256-token crop drawn from its own pool, one per epoch.
#
# A DEDICATED SCRIPT, NOT `EXTRA=` ON launch_default_arm.sh. That script's banner says FULLY
# UNCROPPED, it defaults RELLAB to itm_labels_extended, it has no --crop-pool, and its coverage
# gates were written for the uncropped population. Passing crop flags through EXTRA would leave a
# run whose own log and note described the opposite of what it did.
#
# WHAT IS DELIBERATELY IDENTICAL TO 87ja6a, so the comparison measures the crop and nothing else:
#   --rank-form rel --rank-pool mean --alpha-rank 75 --rank-cross-seed --alpha-pae 1
#   --skip-dead-ops --pose-perblock --pose-perblock-copy --frame-direction --lean-single
#   the same 846-structure population, the same --train-only pattern, lr 1e-3
#   the held-out eval on valfull83, UNCROPPED, every 5 epochs
#
# WHAT IS DIFFERENT, and all of it:
#   --crop-pool training/crop_final          one 256-token mask per structure per epoch
#   --rel-labels training/fold/itm_labels_crop   targets rebuilt under those masks, d0 pinned
#
# FRESH FROM PRETRAINED WEIGHTS. No resume from 87ja6a's epoch 20: that head was trained on full
# structures, and continuing it under a crop would confound the crop with twenty epochs of
# uncropped training.
#
# THE POPULATION IS THE SAME 846 STRUCTURES, verified as a SET not a count:
#   853 manifest rows - 7 SKIP names present = 846 trainable
#   crop pool 846, crop labels 846, and want == pool == labels exactly.
# THE SKIP LIST IS KEPT EVEN THOUGH TWO ENTRIES ARE UNCROPPED OOM EXCLUSIONS (9dzf/9dzg at 2,289
# tokens, 7m7i at 2,004) which a 256-token crop would make harmless. Keeping them is the
# conservative choice: re-including three structures would change the training set at the same
# time as the crop, and the comparison against 87ja6a is the whole point. SKIP=... overrides.
#
# DRY=1 prints the full argv and every gate without touching a card.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
PY=/home/ebrahim/Protenix/.venv/bin/python
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

POOL_DIR="${POOL_DIR:-training/crop_final}"
RELLAB="${RELLAB:-training/fold/itm_labels_crop}"
MANIFEST="${MANIFEST:-train_val_splits/cache_train_structures_extended.csv}"
SKIP="${SKIP:-4zto,7m7i,9b9j,9dzf,9dzg,9e7x,9mx1,9q5m}"
CARDS="${CARDS:-4,5,7}"
EPOCHS="${EPOCHS:-40}"
EVAL_POOL="${EVAL_POOL:-training/valfull83/pool.csv}"
LR="${LR:-1e-3}"
DRY="${DRY:-1}"

say "cropped arm: pool=$POOL_DIR labels=$RELLAB cards=$CARDS epochs=$EPOCHS lr=$LR DRY=$DRY"

# ---------------------------------------------------------------- gates, before anything else
[ -d "$POOL_DIR" ] || { say "REFUSING: $POOL_DIR is not a directory"; exit 2; }
[ -d "$RELLAB" ]   || { say "REFUSING: $RELLAB is not a directory"; exit 2; }
[ -f "$RELLAB/manifest.json" ] || { say "REFUSING: $RELLAB has no manifest.json"; exit 2; }

# THE LABELS MUST BE CROPPED ONES, AND SAY SO. An uncropped --rel-labels here is the one mistake
# that trains against an unreachable target in silence.
$PY -I - "$POOL_DIR" "$RELLAB" "$MANIFEST" "$SKIP" <<'PYG' || exit 3
import csv, json, os, sys
pool_dir, rellab, manifest, skip_s = sys.argv[1:5]
skip = set(x for x in skip_s.split(",") if x)
man = [r["pdb"] for r in csv.DictReader(open(manifest))]
want = set(p for p in man if p not in skip)
pool = set(f[:-4] for f in os.listdir(pool_dir) if f.endswith(".npz"))
lab = set(f[:-4] for f in os.listdir(rellab) if f.endswith(".npz") and f != "manifest.json")
mj = json.load(open(os.path.join(rellab, "manifest.json")))
ok = True
print(f"  manifest rows {len(man)}, SKIP removes {len(skip & set(man))} -> trainable {len(want)}")
print(f"  crop pool {len(pool)}   crop labels {len(lab)}   label manifest says {mj['structures']}")
for nm, missing in (("pool", want - pool), ("labels", want - lab)):
    if missing:
        ok = False
        print(f"  REFUSING: {len(missing)} structure(s) to train are absent from the {nm}: "
              f"{sorted(missing)[:6]}")
if mj["structures"] != len(lab):
    ok = False
    print(f"  REFUSING: label manifest claims {mj['structures']} but {len(lab)} files exist -- a "
          f"partial build. Rerun build_itm_labels_cropped.py --manifest-only.")
if "OST" not in mj.get("chain_mapping", ""):
    ok = False
    print(f"  REFUSING: label manifest chain_mapping is {mj.get('chain_mapping')!r}; targets must "
          f"come from the PER-POSE OST join the evaluator scores with.")
if "uncropped" not in mj.get("d0", ""):
    ok = False
    print(f"  REFUSING: label manifest d0 is {mj.get('d0')!r}; it must be pinned at the uncropped "
          f"scored chain-pair count or prediction and target use different kernels.")
print(f"  population SET EQUALITY: {'YES' if want == pool == lab else 'NO'}")
sys.exit(0 if ok and want == pool == lab else 1)
PYG

say "CPU gates"
$PY -I run_logging/test_crop_pool_wiring.py || { say "REFUSING: wiring gates failed"; exit 4; }

TRAIN_ONLY='^(?!pairformer_stack\.)'
ARGS=(--crop-pool "$POOL_DIR"
      --rel-labels "$RELLAB"
      --rank-loss --rank-form rel --rank-pool mean --alpha-rank 75
      --rank-cross-seed --rank-structs-per-step 2
      --minimal-items
      --skip-dead-ops --pose-perblock --pose-perblock-copy --frame-direction --lean-single
      --eval-pool "$EVAL_POOL" --eval-pool-every 5
      --epochs "$EPOCHS" --lr "$LR"
      --alpha-confidence 1.0 --alpha-pae 1.0)

say "argv the trainer would receive:"
printf '    %s\n' "${ARGS[@]}"
say "held-out eval: $EVAL_POOL, UNCROPPED -- eval_checkpoint takes crop_tokens separately and it "
say "  is left empty, so every historical number stays comparable"
say "NOT passed, deliberately: --resume (fresh from pretrained), --eval-pool-crop-tokens,"
say "  --crop-margin, --crop-tokens, --rank-pool max (refused under a crop)"

if [ "$DRY" = "1" ]; then
  say "DRY RUN -- nothing launched. Re-run with DRY=0 to go."
  exit 0
fi

say "REFUSING to launch: the full launch path is not wired yet. DRY=0 is reserved until a GPU"
say "  smoke has shown N_head equals the selected mask size on every rank."
exit 5
