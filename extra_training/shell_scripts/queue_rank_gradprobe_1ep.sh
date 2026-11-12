#!/bin/bash
# ONE EPOCH OF PAE + rel, TO MEASURE THE TWO TERMS' GRADIENT MAGNITUDES. Not an experiment.
#
# WHY THIS BEFORE THE 50-EPOCH ARM. Every alpha_rank this project has used came from ONE
# --grad-probe window: a single accumulation step over 2-3 structures, on a quantity that largely
# cancels. The five such measurements on record span ratio 0.0041 to 0.4728, and most of that
# spread is that they froze different fractions of the head rather than anything about the losses.
# 75 was picked against the mid-range of those. This run measures the ratio at EVERY optimiser
# step of a full epoch on the architecture the arm will actually use, so the next alpha comes from
# a distribution with a visible spread instead of one draw.
#
# WHAT IT WRITES. grad_split.r<rank>.csv, one row per measured step:
#   epoch,step,g_pae,g_rank,ratio,cos,alpha_rank,alpha_for_parity,poses
# ratio is |g_rank|/|g_pae| WITH alpha_rank already applied; alpha_for_parity divides it back out
# and reports the alpha that would equalise the two AT THAT STEP, so the column is comparable
# across runs whatever alpha this one used. Read rank 0's file; all three are written.
#
# READ THE ACCUMULATED NUMBERS, NOT PER-POSE ONES. The rel coefficients sum to zero inside an
# interface group, so poses cancel against each other: 0.0311 per-pose against 0.0041 accumulated
# on 3hzrs8, a factor of 7.6. The accumulated figure is what the optimiser applies.
#
# K=1, SO EVERY STEP IS MEASURED. That is one extra backward per pose, which roughly doubles
# backward time -- about 20 min against 10 for this epoch. For a one-off diagnostic the full
# ~155-step distribution is worth more than the 10 minutes; the 50-epoch arm should run K=50.
#
# NO EVAL FIRES. --train-eval-every is 5 and this runs 1 epoch, so the 12-minute valfull83 pass
# never starts and the whole wall clock is training.
#
# ALPHA 75 IS PASSED DELIBERATELY even though the point is to question it: the gradient the
# optimiser takes is the one with alpha applied, so measuring at alpha 1 would describe a
# trajectory this project is not going to run. alpha_for_parity is the alpha-independent answer.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

CARDS="${CARDS:-4,5,7}"
ALPHA_RANK="${ALPHA_RANK:-75}"
RANK_S="${RANK_S:-3}"
GRAD_PROBE_EVERY="${GRAD_PROBE_EVERY:-1}"
# THE 853-STRUCTURE LABELS, NOT THE LAUNCHER'S DEFAULT. training/fold/itm_labels covers 368
# structures -- the old winnable subset -- and the trainer hard-stops on a scored structure with
# no ranking label, so the default would refuse this arm rather than train a silent subset.
# itm_labels_853 was built 2026-11-12 from gt_iptm_ceiling_853: 71,220 poses over 1,686
# interfaces, 852 of 853 structures. The one gap is 4zto, whose OST chain mapping failed when its
# poses were generated, and the arm already carries it in --manifest-skip.
# CROPPED LABELS, BECAUSE THE ARM IS CROPPED. itm_labels_853 (built today from
# gt_iptm_ceiling_853) holds FULL-STRUCTURE true iTM, and phase_k_train refuses it under
# --crop-pool: rel trains the head to reproduce a pose's true iTM, that iTM is an average over
# the interface's cross-chain pairs, and under a crop the head's prediction averages a SUBSET
# while the target would still be the whole-structure value. The residual would absorb the
# difference and nothing would refuse. itm_labels_crop was built 2026-11-10 against this exact
# pool -- training/crop_final with training/combined/pool.csv -- 846 structures, 7,942,385 rows
# over 94,630 crops, keyed (crop_index, seed, sample, iface), with crop_index -1 kept as the
# uncropped reference. itm_labels_853 is the right input for an UNCROPPED rel arm.
REL_LABELS="${REL_LABELS:-training/fold/itm_labels_crop}"
PAIR_DROPOUT="${PAIR_DROPOUT:-0.25}"
LR="${LR:-1e-3}"
DRY="${DRY:-1}"

say "one-epoch gradient-split diagnostic: alpha_rank=$ALPHA_RANK K=$GRAD_PROBE_EVERY"
say "  cropped pool, pair-dropout=$PAIR_DROPOUT, rel labels over all 853 structures"
say "  cards=$CARDS  DRY=$DRY"

# THE CARDS ARE NOT TAKEN BY THIS SCRIPT WITHOUT A FREE GATE. launch_pae_crop_arm.sh runs
# cards_busy itself and refuses occupied cards, so a queued copy of this cannot displace a live
# arm by being run at the wrong moment.
CARDS="$CARDS" \
EPOCHS=1 \
LR="$LR" \
PAIR_DROPOUT="$PAIR_DROPOUT" \
ALPHA_RANK="$ALPHA_RANK" \
RANK_S="$RANK_S" \
GRAD_PROBE_EVERY="$GRAD_PROBE_EVERY" \
REL_LABELS="$REL_LABELS" \
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
fi
exit $rc
