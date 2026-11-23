#!/usr/bin/env bash
# ARM O: ARM L (VANILLA PAE, NATIVE HEAD) ON THE TOPPED-UP 345-INTERFACE POOL.
#
# THE ONE CHANGE FROM ARM L is the pool: 280 -> 345 interfaces over the SAME 180 structures. So the
# objective, the head, the lr, the crop setting and the manifest are all Arm L's, and a difference
# is attributable to the data mix.
#
# WHY. Every arm so far trains on a pool that is 74% hard interfaces, while val83's good poses are
# 99.7% from EASY ones -- the arms are fitted to a distribution the metric barely contains, and the
# easy-interface regression (Arm L lost 0.0160 top-1 DockQ there on val) is the visible cost. The
# 65 added interfaces are the entire supply that costs nothing: they sit on structures already in
# the pool, so no new poses are loaded and no new trunk is read; only the PAE mask scores more
# chain pairs per pose. The mix moves from 74% hard to 60% hard.
#
# THE SUPPLY IS WHAT IT IS, AND IT SKEWS EASY: of the 65, ten have one all-good seed, fifteen have
# two or three, forty have four or more. There is no way to add near-hard easy interfaces at scale
# -- only 14 exist outside the slice at all -- so this cannot be a balanced top-up, and the arm
# tests "more easy interfaces" rather than "a matched mix".
#
# THE PREFLIGHT AND chain_roles ARE slice280's, COPIED NOT REGENERATED: the structure set is
# identical and the preflight is per structure (checked_seed = "per-structure"), so the marker
# vouches for this pool as it stands. Verified before the copy.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') Arm L: NATIVE shipped head -- no injected layers, no dropout"
# EPOCHS DEFAULTS TO 18, NOT 3: both eval specs are every=6, and (ep+1) % 6 never fires inside
# three epochs, so at the old default the arm produced no val and no in-sample number at all --
# only loss curves. Raised in review.
#
# THIS NOTE IS OUTSIDE THE env BLOCK ON PURPOSE. Put inside it, the '#' ends the backslash
# continuation, env runs with no command, and the rest of the block becomes a separate invocation
# that silently falls back to every base default -- which is exactly what happened on the first
# launch attempt.
echo "    topped-up 345 (206 hard + 139 easy), uncropped, 3 epochs, NO in-sample pass; dead ops stubbed as always"

env EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-103}" \
    POSE_PERBLOCK="" POSE_PERBLOCK_COPY="" FRAME_DIRECTION="" \
    POSE_READOUT="" POSE_READOUT_PRENORM="" POSE_READOUT_COPY_DISTANCE="" \
    POSE_READOUT_POST_ZERO="" \
    MANIFEST="${MANIFEST:-training/slice345/structures.csv}" \
    SKIP="${SKIP-}" \
    SCORED_PAIRS="${SCORED_PAIRS:-training/slice345/pool.csv}" \
    PREFLIGHT="${PREFLIGHT:-training/slice345/alignment_preflight.json}" \
    PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT="${PAE_BALANCE_ACCEPT:-0}" \
    CARDS="${CARDS:-4,5,7}" EPOCHS="${EPOCHS:-18}" EVAL_EVERY="${EVAL_EVERY:-0}" \
    LR="${LR:-1e-3}" PAIR_DROPOUT="${PAIR_DROPOUT-}" \
    POOL_DIR="${POOL_DIR-}" \
    EXTRA_EVAL="${EXTRA_EVAL-name=slice345,pool=training/slice345/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=6;name=valfull83,pool=training/valfull83/pool.csv,fold=validation/fold,seed_base=400000,budget=30,every=6}" \
    EVAL_PAE_LOSS="${EVAL_PAE_LOSS:-1}" \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY="${DRY:-1}" \
    ./shell_scripts/launch_pae_crop_arm.sh
