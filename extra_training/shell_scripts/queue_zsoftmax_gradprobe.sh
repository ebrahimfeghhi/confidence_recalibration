#!/bin/bash
# ONE EPOCH OF PAE + zsoftmax AT FROZEN WEIGHTS, TO SET alpha_rank. Not an experiment.
#
# WHY alpha CANNOT BE CARRIED OVER FROM rel. The zsoftmax coefficients are (1/d) times a quantity
# bounded by 1, where d is the item's own score spread; rel's are (2/P) r_i on a regression
# residual. Those are different magnitudes with different units, so every alpha_rank this project
# has used describes the wrong term. There is no defensible value to reuse.
#
# THE THREE THINGS THIS IS BUILT TO GET RIGHT, each one a way the measurement would read too low:
#
#   1. AVERAGED OVER MANY ITEMS, WITH SKIPPED ONES COUNTED AS ZERO. The ranking gradient is wildly
#      uneven -- near zero on an item already ranked correctly, large on a wrong one, and exactly
#      absent on an all-acceptable or all-unacceptable item, which this loss skips by design.
#      GRAD_PROBE_EVERY=1 measures EVERY optimiser step of a full epoch, and the figure recorded
#      is the ACCUMULATED gradient over the step, so an item that contributed no coefficient is
#      already in the denominator as a zero. Measuring only where the term is active, or on a
#      handful of steps, would set alpha too high by exactly the fraction of quiet items.
#
#   2. MEASURED AT THE SHIPPED WEIGHTS AND FROZEN. LR=0, so AdamW takes no step and all ~155
#      measurements are at the same weights -- the spread across steps is item composition and
#      dropout, not drift. The reason to freeze rather than re-balance later: the ranking gradient
#      should SHRINK as items get solved, so an alpha retuned mid-training would amplify the term
#      exactly when it has least left to say.
#
#   3. TRAINABLE PARAMETERS ONLY. phase_k_train's probe iterates `trainable_params`, which is
#      named_parameters filtered to the trainable set, so the norms it compares are taken where
#      the two terms actually compete rather than over a frozen trunk that dilutes both.
#
# AND THE COSINE, which is the reason to read this before choosing anything. grad_split's `cos`
# column is the cosine between the PAE and ranking gradients at each step. Clearly negative means
# the two terms are pulling against each other, and parity would then be too much weight on
# ranking rather than the neutral choice it looks like.
#
# ALPHA 1, NOT 75. alpha_for_parity divides the applied alpha back out and is alpha-independent,
# but at alpha 1 the `ratio` column reads directly as |g_rank|/|g_pae| for the bare term, which is
# the number to think with when no prior alpha exists.
#
# NO EVAL FIRES: one epoch against --train-eval-every 5.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

CARDS="${CARDS:-4,5,7}"
ALPHA_RANK="${ALPHA_RANK:-1}"
RANK_ACCEPT="${RANK_ACCEPT:-0.23}"
RANK_S="${RANK_S:-3}"
GRAD_PROBE_EVERY="${GRAD_PROBE_EVERY:-1}"
# THE POOL THE ARM WOULD ACTUALLY RUN ON. The full-set arm trained on this manifest, so the item
# mix -- and therefore the fraction of items this loss skips -- is the mix alpha has to survive.
MANIFEST="${MANIFEST:-training/fullset/structures.csv}"
# THE WHOLE ENV THE ARM RAN WITH, NOT THE PARTS I REMEMBERED. The first launch of this script
# passed MANIFEST alone and died in 0.7 min: the alignment preflight is keyed by a digest of the
# manifest, so the launcher's default artefact "says nothing about the structures this run will
# train on". SCORED_PAIRS is worse than a refusal would be if it defaulted silently -- it names
# the interfaces the PAE mask keeps AND the ones the ranking term scores. These four come from
# fullset_chain.sh's own env block, copied rather than retyped.
PREFLIGHT="${PREFLIGHT:-training/fullset/alignment_preflight.json}"
SCORED_PAIRS="${SCORED_PAIRS:-training/fullset/pool.csv}"
PAE_SPLIT_ACCEPT="${PAE_SPLIT_ACCEPT:-0.23}"
PAE_BALANCE_ACCEPT="${PAE_BALANCE_ACCEPT:-0.23}"
# DROPOUT AS THE ARM RUNS IT, not off. It perturbs both terms, and the ratio under the conditions
# the arm will train in is the quantity alpha has to be right for; a clean-weights measurement
# would describe a run nobody is going to launch.
PAIR_DROPOUT="${PAIR_DROPOUT:-0.25}"
LR="${LR:-0}"
DRY="${DRY:-1}"

say "zsoftmax gradient-split diagnostic at FROZEN weights"
say "  alpha_rank=$ALPHA_RANK  accept=$RANK_ACCEPT  K=$GRAD_PROBE_EVERY  lr=$LR  S=$RANK_S"
say "  manifest=$MANIFEST  pair-dropout=$PAIR_DROPOUT  cards=$CARDS  DRY=$DRY"
say "  preflight=$PREFLIGHT  scored-pairs=$SCORED_PAIRS  pae-accept=$PAE_SPLIT_ACCEPT"
if [ "$LR" != "0" ]; then
  say "NOTE: lr is NOT 0, so the weights move across the epoch and the later steps are no longer"
  say "  a measurement at the shipped weights. That is a different quantity -- set LR=0 unless you"
  say "  meant it."
fi

CARDS="$CARDS" \
EPOCHS=1 \
LR="$LR" \
PAIR_DROPOUT="$PAIR_DROPOUT" \
ALPHA_RANK="$ALPHA_RANK" \
RANK_FORM=zsoftmax \
RANK_ACCEPT="$RANK_ACCEPT" \
RANK_S="$RANK_S" \
GRAD_PROBE_EVERY="$GRAD_PROBE_EVERY" \
MANIFEST="$MANIFEST" \
PREFLIGHT="$PREFLIGHT" \
SCORED_PAIRS="$SCORED_PAIRS" \
PAE_SPLIT_ACCEPT="$PAE_SPLIT_ACCEPT" \
PAE_BALANCE_ACCEPT="$PAE_BALANCE_ACCEPT" \
PREFETCH_WORKERS=16 \
TRUNK_MEMMAP=1 \
EPOCH_BALANCE=1 \
DRY="$DRY" \
  ./shell_scripts/launch_pae_crop_arm.sh
rc=$?
say "launcher returned $rc"
if [ "$DRY" = "0" ] && [ "$rc" = "0" ]; then
  say "when it finishes, read the distribution with:"
  say "  python -I run_logging/grad_split_report.py <rid>"
  say "and read the COSINE column before picking any alpha."
fi
exit $rc
