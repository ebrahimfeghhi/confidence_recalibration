#!/bin/bash
# TRAIN ABAG-RANK ON 70% OF OUR TRAINING STRUCTURES, EVALUATE ON THE OTHER 30%.
#
# WHY THIS RUN EXISTS. Against valfull the reranker settles 0.02-0.04 BELOW shipped while its own
# training loss keeps falling (abag_rank/RESULTS.md), and the confidence-head arms fail the same
# way at +0.0024. valfull was drawn as whole SAbDab2 ab_ag_clusters by
# phases/phase_b_define_val_clusters.py, so a val complex is never from a training cluster. This
# run removes that: a RANDOM 70/30 split of the training structures puts similar complexes on both
# sides. If the model generalises here but not to valfull, the wall is cluster distance rather
# than the model.
#
# THE SPLIT IS BY STRUCTURE, NOT BY INTERFACE. 382 of our 487 training structures carry more than
# one labelled interface, and two interfaces of one pdb share a trunk and most of their poses --
# splitting by interface would put the same structure on both sides and make the held-out score
# meaningless. train.py shuffles the pdb list with --seed and takes --val-frac of it.
#
# WHAT IT REPORTS, per epoch: training loss, held-out regression loss, held-out soft Spearman, and
# HELD-OUT TOP-1 DockQ with its oracle. Checkpoints are chosen on top-1, not on loss, because this
# project has already measured a model improving its loss and its rank correlation while the
# single pick did not move.
#
# NOTE THE BUDGET. Training structures were folded at 10 seeds x 5 samples = 50 poses, so this
# held-out number is at budget 10 and is NOT comparable to the budget-30 valfull figures in
# ../paper_results.md. Its own oracle (~0.49) is the thing to read it against.
#
# REPRODUCIBILITY. The split is a pure function of (seed, val_frac, the eligible interface set),
# so train.py writes the membership to <out>/split.json at startup -- 341 train / 146 val
# structures at seed 0, verified disjoint and sharing no structure with valfull. Check a rerun
# against that file rather than trusting the seed: if a chainmap or a label appears later the
# eligible set grows and the same seed reshuffles the whole boundary.
#
# Usage:  ./abag_rank/run_7030.sh [out dir] [epochs] [card] [seed]
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
OUT="${1:-training/abag/run_7030}"
EPOCHS="${2:-10000}"
CARD="${3:-4}"
SEED="${4:-0}"
mkdir -p "$OUT"
echo "$(TZ=America/New_York date '+%F %-I:%M %p %Z') 70/30 structure split, seed $SEED, "\
"$EPOCHS epochs, card $CARD -> $OUT"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$CARD" $PY -I abag_rank/train.py \
  --out "$OUT" \
  --epochs "$EPOCHS" \
  --val-frac 0.3 \
  --seed "$SEED" \
  --device cuda \
  --pool training/full_train/pool_insample.csv \
  --dump training/abag/pae_train \
  --restypes training/abag/restypes \
  --results training/fold/results.csv
rc=$?
echo "$(TZ=America/New_York date '+%-I:%M %p') run_7030 returned $rc"
exit $rc
