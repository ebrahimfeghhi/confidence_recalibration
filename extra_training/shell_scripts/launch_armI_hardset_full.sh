#!/usr/bin/env bash
# ARM I: TRAIN ON THE HARD 280 ONLY, WITH THE WHOLE HEAD TRAINABLE. 10 epochs.
#
# WHAT QUESTION THIS ANSWERS. Every arm so far has moved the hard set by almost nothing: shipped
# gets 0.2501 mean DockQ / 30.7% acceptable over its 280 interfaces, the oracle is 0.5520, and the
# best arm of thirteen (rel at alpha 250) reached 0.2607 -- 3.5% of the available headroom, and
# 31.8% acceptable is three interfaces out of 280. Those arms all trained 20-53 tensors on the
# FULL pool and were then asked about this subset. This arm removes both limits at once: it trains
# ON the hard set and it unfreezes EVERYTHING.
#
# SO IT IS A CEILING MEASUREMENT, NOT A CANDIDATE. The metric is in-sample by construction -- the
# arm trains on the very interfaces it is scored on -- so a good number here means the hard set is
# FITTABLE, not that anything generalises. That is exactly the point: if a fully unfrozen head
# trained directly on these 280 interfaces still cannot beat 0.2501 by much, then the limit is not
# capacity and not data volume, it is that max-pooled ipTM over a PAE head cannot order these
# poses -- which would redirect the whole line of work. If it fits easily, every arm so far was
# capacity-starved and the 53-tensor budget is the thing to change.
#
# EVERY LIVE TENSOR, AND THE DEAD ONES STAY STUBBED. The launcher's default --train-only is
# already precisely that set: '^(?!pairformer_stack\.)' takes everything outside the stack and the
# named blocks add only the modules measured to carry signal, so the ten dead ones are excluded by
# construction -- 113 tensors of the head's 264. --skip-dead-ops stays ON and the two agree.
#
# 119, NOT 113: the launcher's default pattern selects 113 tensors on a bare arm, and this arm
# ALSO carries the six pose-injection tensors (pose_readout_*, which '^(?!pairformer_stack\.)'
# matches because they sit outside the stack). The gate caught the difference and named it --
# "--expect-trainable 113 but --train-only left 119 trainable tensor(s)" -- which is exactly the
# truncated-pattern class it exists for, so the count is corrected rather than the gate disabled.
#
# WHY NOT LITERALLY EVERYTHING: a module whose every tensor sits in the band 4.904e-37 has an
# exactly zero gradient, because the product of two such factors underflows fp32. Training it
# allocates AdamW moments and runs a forward that emits zeros; it cannot move. The trainer refuses
# the contradiction outright ("--skip-dead-ops would stub 10 module(s) that --train-only has left
# TRAINABLE"), which is the correct refusal.
#
# ALL SIX POSE-INJECTION SITES, which is Arm E's configuration unchanged: POSE_PERBLOCK puts one
# before each of the four pairformer blocks, POSE_READOUT_PRENORM one before pae_ln (where it can
# only rotate), POSE_READOUT_POST_ZERO one after it (zero-init, so it can scale and the head starts
# unchanged).
#
# IN-SAMPLE METRICS ONLY, EVERY 10 EPOCHS, ON THE HARD 280 -- no validation pass, by instruction.
# EVAL_EVERY=0 switches the val pass off; EXTRA_EVAL carries slice280 at every=10. All four metrics
# come out of that one pass (mean DockQ, frac acceptable, rho, and PAE loss split good/bad/total)
# because EVAL_PAE_LOSS computes the loss from the logits that pass already produced.
#
# THE POOL NEEDED A TRAINING MANIFEST BUILT FOR IT. training/slice280 held only a pool.csv: it has
# always been an EVAL pool, and 46 of its 180 structures are absent from
# training/full_train/structures.csv altogether. All 46 do have labels and trunks, so
# training/slice280/structures.csv was derived from what is on disk, with token counts read from
# each trunk's own ci_feat_asym_id rather than from a manifest that does not list them:
# 180 structures, 2,700 units at 15 seeds each, 13,500 poses. Its preflight marker is built over
# that manifest because the full_train marker does not cover those 46.
# UNCROPPED, WHICH IS NOW THE POINT OF THE ARM. POOL_DIR="" passes no --crop-pool, so every pose
# is scored WHOLE.
#
# WHY THIS CHANGED. Measured over 60 random training structures, the 256-token crop keeps a MEDIAN
# 19.8% of a structure's scored cross-chain pairs (p10 6.0%, worst 3.1% on 10gh at 1,474 tokens)
# while keeping 44% of its TOKENS -- the pair count is quadratic and a cross-chain pair needs both
# partners inside the crop. So every arm run so far has been fit on about a fifth of each
# interface and then selected on all of it, and d0 (calculate_normalization of the pair's token
# count) differs between the two as well. A cropped version of this arm would answer "can a full
# head fit a fifth of the hard set", which is not the question.
#
# IT ALSO REMOVES THE d0 MISMATCH that the ranking loss had to patch around
# (n_pair_tokens=_uncrop_npt, added after pass A was found scoring with a cropped d0); the PAE
# loss has no equivalent fix, and uncropped it needs none.
#
# THE HARD 280 FITS: median 589 tokens, max 1,745, against --crop-size 3400. The cost is a
# full-size forward per pose instead of a 256-token one.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') Arm I: hard 280 only, WHOLE HEAD trainable, 3 epochs"
echo "    180 structures / 2,700 units / 13,500 poses; all 6 pose sites; in-sample metrics every 10, no val"

env EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-119}" \
    POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE=1 \
    POSE_READOUT_POST_ZERO=1 \
    MANIFEST="${MANIFEST:-training/slice280/structures.csv}" \
    SKIP="${SKIP:-}" \
    SCORED_PAIRS="${SCORED_PAIRS:-training/slice280/pool.csv}" \
    PREFLIGHT="${PREFLIGHT:-training/slice280/alignment_preflight.json}" \
    PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT="${PAE_BALANCE_ACCEPT:-0}" \
    CARDS="${CARDS:-4,5,7}" EPOCHS="${EPOCHS:-3}" EVAL_EVERY="${EVAL_EVERY:-0}" \
    LR="${LR:-1e-3}" PAIR_DROPOUT="${PAIR_DROPOUT:-0.25}" \
    EXTRA_EVAL="${EXTRA_EVAL:-name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=3}" \
    EVAL_PAE_LOSS="${EVAL_PAE_LOSS:-1}" \
    POOL_DIR="${POOL_DIR-}" \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY="${DRY:-1}" \
    ./shell_scripts/launch_pae_crop_arm.sh
