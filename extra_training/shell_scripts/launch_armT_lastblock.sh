#!/usr/bin/env bash
# ARM T = THE LAST BLOCK AND EVERYTHING AFTER IT, PLUS TWO POSE SITES.
#
# WHAT IS TRAINABLE. Block 3 (the final pairformer block: both tri-muls and the pair transition),
# pae_ln, linear_no_bias_pae, and the pose modules at two sites -- immediately BEFORE block 3, and
# AFTER pae_ln but before the linear readout. Blocks 0-2 are frozen, so PyTorch builds no autograd
# graph through them at all and their activations are never stored: that is where the memory and
# the ~40 min/epoch of backward recompute go.
#
# WHY ONLY block 3's POSE SITE IS SELECTED WHILE THREE ARE INSTALLED. pose_perblock installs at
# blocks 1..3 as one axis -- there is no per-block selector -- so the modules at blocks 1 and 2
# exist. They are zero-init and ABSENT from --train-only, so they emit exactly 0.0 forever and
# cannot contribute: the ModuleLists are indexed 0,1,2 over blocks 1,2,3, and only `.2.` is named
# below. This is a selection, not a mode, and it is why the trainable count must be read off the
# startup line rather than assumed.
#
# WHY IT IS WORTH RUNNING. Arm S (103 tensors, every block) reached t=-2.24 BELOW shipped on
# held-out top-1 by epoch 12 while its in-sample numbers climbed -- capacity it could not use
# transferably. This cuts trainable capacity by roughly 5x and adds geometry at the two sites
# closest to the readout, which is the opposite trade.
#
# EXPECT_TRAINABLE=34 IS COUNTED, NOT GUESSED: block 3's tri_mul_out (10 tensors), tri_mul_in
# (10) and pair_transition (5), pae_ln (2) and linear_no_bias_pae (1) -- read off the key list of
# Arm S's head_ep12 checkpoint -- plus 3 pose tensors at the block-3 site and 3 at the readout.
# The trainer asserts it at startup, so a pathway that installs a different number refuses rather
# than training a set nobody chose. DRY cannot check this: --train-only is an inline flag after
# `-- "${ARGS[@]}"`, which DRY never reaches, so the real check is args.json after launch.
#
# EVERYTHING ELSE IS ARM S: 842 structures, all seeds, uncropped, lr 1e-3, dropout 0.25,
# split 0.23, balance 0, dead ops skipped, 20 epochs, passes every 5.
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

env EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-34}" \
    TRAIN_ONLY="${TRAIN_ONLY:-blocks\.3\.tri_mul_out\.,blocks\.3\.tri_mul_in\.,blocks\.3\.pair_transition\.,pae_ln,linear_no_bias_pae,pose_perblock_linear_no_bias_d\.2\.,pose_perblock_linear_no_bias_d_wo_onehot\.2\.,pose_perblock_frame_dir\.2\.,pose_readout_}" \
    POSE_PERBLOCK="" POSE_PERBLOCK_COPY="" FRAME_DIRECTION=1 \
    POSE_BLOCK_TERMS=onehot,dist,orient POSE_BLOCK_INIT=zero \
    POSE_READOUT_SITES=post POSE_READOUT_TERMS=onehot,dist,orient POSE_READOUT_INIT=zero \
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
