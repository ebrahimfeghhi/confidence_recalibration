#!/usr/bin/env bash
# Arm S trains the native confidence head on uncropped structures with at most
# 1,745 tokens, the maximum in Arm L's hard-280 training manifest.
set -euo pipefail
cd "$(dirname "$0")/.."

ARM_S_MANIFEST="${MANIFEST:-train_val_splits/cache_train_structures_extended.csv}"
ARM_S_PREFLIGHT="${PREFLIGHT:-training/alignment_preflight_sharded.json}"
ARM_S_SKIP="$(python3 - "$ARM_S_MANIFEST" "$ARM_S_PREFLIGHT" "${SKIP-}" <<'PYLIMIT'
import csv
import json
from pathlib import Path
import sys

manifest, preflight, extra_skip = sys.argv[1:]
limit = 1745
with open(manifest) as stream:
    sizes = {row["pdb"]: int(row["tokens"]) for row in csv.DictReader(stream)}
with open(preflight) as stream:
    marker = json.load(stream)

skip = set(filter(None, (name.strip() for name in extra_skip.split(","))))
if marker.get("manifest_selection") and Path(marker["manifest_selection"]).resolve() == Path(manifest).resolve():
    skip.update(name.strip() for name in marker.get("manifest_skip", "").split(",") if name.strip())
oversized = {name for name, size in sizes.items() if size > limit}
skip.update(oversized)
retained = {name: size for name, size in sizes.items() if name not in skip}
if not retained:
    sys.exit("Arm S size filter left no training structures")
print(f"Arm S training: {len(retained)}/{len(sizes)} structures; maximum {max(retained.values())} tokens. "
      f"Excluded above {limit}: {', '.join(sorted(oversized)) or 'none'}.", file=sys.stderr)
print(",".join(sorted(skip)))
PYLIMIT
)"

env EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-103}" \
    POSE_PERBLOCK="" POSE_PERBLOCK_COPY="" FRAME_DIRECTION="" \
    POSE_READOUT="" POSE_READOUT_PRENORM="" POSE_READOUT_COPY_DISTANCE="" \
    POSE_READOUT_POST_ZERO="" \
    MANIFEST="$ARM_S_MANIFEST" \
    SKIP="$ARM_S_SKIP" \
    SCORED_PAIRS="${SCORED_PAIRS:-training/combined/pool.csv}" \
    PREFLIGHT="$ARM_S_PREFLIGHT" \
    PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT="${PAE_BALANCE_ACCEPT:-0}" \
    CARDS="${CARDS:-4,5,7}" EPOCHS="${EPOCHS:-20}" EVAL_EVERY="${EVAL_EVERY:-0}" \
    LR="${LR:-1e-3}" PAIR_DROPOUT="${PAIR_DROPOUT:-0.25}" \
    POOL_DIR="${POOL_DIR-}" \
    EXTRA_EVAL="${EXTRA_EVAL-name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=5;name=valfull83,pool=training/valfull83/pool.csv,fold=validation/fold,seed_base=400000,budget=30,every=5}" \
    EVAL_PAE_LOSS="${EVAL_PAE_LOSS:-1}" \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY="${DRY:-1}" \
    ./shell_scripts/launch_pae_crop_arm.sh
