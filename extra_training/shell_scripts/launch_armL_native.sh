#!/usr/bin/env bash
# ARM L: THE SHIPPED HEAD, UNMODIFIED. No injected layers, no dropout. Hard 280, uncropped.
#
# THE HYPOTHESIS THIS TESTS. Every arm so far adds parameters to the confidence head -- six
# pose-injection sites, the per-block projections, the frame-direction projection -- and then
# spends its early epochs reconciling the perturbed head with the one the trunk was trained
# against. Arm I removed the crop and unfroze every live tensor and still moved top-1 mean DockQ
# by 0.0001 (0.2504 against shipped's 0.2505), so capacity, data volume and the crop are all ruled
# out. This arm removes the one remaining difference from the shipped network: the parameters we
# added.
#
# AND IT IS NOT A CRIPPLED ARM. The shipped confidence head is ALREADY pose-conditioned --
# run_confidence_head takes x_pred_coords -- so the PAE prediction depends on the pose without any
# of our additions. What goes away is the EXTRA pathway, not the only one.
#
# WHAT IS OFF, and each of these adds parameters:
#   --pose-perblock / --pose-perblock-copy   one projection before each of the four blocks
#   --frame-direction                        frame_dir_proj
#   --pose-readout* (all four)               the pre-norm and post-norm readout sites
# WHAT STAYS: --lean-readouts and --lean-single, which only skip computation the PAE-only objective
# throws away and add nothing; and --skip-dead-ops, because dead neurons are never trained.
#
# IN-SAMPLE EVERY 6 EPOCHS: fires at 6, 12, 18, 24, 30 under the gate (ep + 1) % every == 0.
#
# WHY 6 AND NOT 3. A pass is ~30 min of three cards against a 32 min epoch. At every 3 that is two
# passes per round, +31% wall clock, ~10 extra hours over ten rounds; at every 6 it is ~5 hours and
# still gives five measurements across 30 epochs. The deciding point is that cadence is NOT a
# one-way door -- every epoch's checkpoint is kept, so epochs 3, 9, 15 can be filled in later
# without retraining if something starts moving. Spending ten hours up front to resolve a quantity
# that has moved by under 0.001 in thirteen arms is the wrong bet.
#
# BOTH POOLINGS, not max alone. Mean-pooled ipTM comes out of the SAME forward at no extra cost and
# has been the stronger column every time it was measured -- Arm G reached 44.3% frac acceptable at
# top-5 mean-pooled against 41.4% max, and on the contestable val interfaces its mean-pooled best@5
# also beat max. Dropping it would discard the better signal to save nothing.
#
# TOP-5 NEEDS NO EXTRA GPU. The pass writes per-pose rows; run_logging/topk_table.py reduces them
# to top-1 and top-5 afterwards, so asking for top-5 costs only the reader. Each pass costs ~30 min of three cards against a
# 32 min epoch, so per-epoch checks more than doubled the arm's wall clock for a DockQ number that
# has not moved in any arm yet. The per-epoch signal comes from the TRAINING PAE curve instead,
# read against one fixed shipped baseline measured over the same 180 x 15 units
# (training/runs/shipped_insample280) -- which is what made the per-epoch passes redundant: before
# that baseline existed, every comparison had to be restricted to the 4,275 poses the old
# full-train shipped pass happened to cover.
#
# DROPOUT OFF. PAIR_DROPOUT=0 against the 0.25 every other arm has used. With a 3-epoch budget on
# 180 structures, regularisation costs signal rather than buying generalisation, and it also adds
# variance to anything read per epoch.
#
# TRAINABLE SET: 103 TENSORS, and that number is derived rather than guessed. The launcher's
# default --train-only selects 119 on Arm I's head; exactly 16 of those are tensors WE added
# (frame_dir_proj, 9 pose_perblock_*, 6 pose_readout_*), counted by applying the same regex to Arm
# I's checkpoint on the CPU. 119 - 16 = 103, which is the live shipped set. The gate refuses if the
# count moves, which is how a pose flag creeping back in would be caught -- it would read 119.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') Arm L: NATIVE shipped head -- no injected layers, no dropout"
echo "    hard 280, uncropped, 3 epochs, NO in-sample pass; dead ops stubbed as always"

env EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-103}" \
    POSE_PERBLOCK="" POSE_PERBLOCK_COPY="" FRAME_DIRECTION="" \
    POSE_READOUT="" POSE_READOUT_PRENORM="" POSE_READOUT_COPY_DISTANCE="" \
    POSE_READOUT_POST_ZERO="" \
    MANIFEST="${MANIFEST:-training/slice280/structures.csv}" \
    SKIP="${SKIP-}" \
    SCORED_PAIRS="${SCORED_PAIRS:-training/slice280/pool.csv}" \
    PREFLIGHT="${PREFLIGHT:-training/slice280/alignment_preflight.json}" \
    PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT="${PAE_BALANCE_ACCEPT:-0}" \
    CARDS="${CARDS:-4,5,7}" EPOCHS="${EPOCHS:-3}" EVAL_EVERY="${EVAL_EVERY:-0}" \
    LR="${LR:-1e-3}" PAIR_DROPOUT="${PAIR_DROPOUT-}" \
    POOL_DIR="${POOL_DIR-}" \
    EXTRA_EVAL="${EXTRA_EVAL-name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=6}" \
    EVAL_PAE_LOSS="${EVAL_PAE_LOSS:-1}" \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY="${DRY:-1}" \
    ./shell_scripts/launch_pae_crop_arm.sh
