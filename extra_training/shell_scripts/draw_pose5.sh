#!/bin/bash
# THE LIVE 5-SITE rel ARM'S OWN FIGURE. Separate from draw_three_arms.sh so a pulse does not pay
# for redrawing three finished arms every ten minutes.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
RID="${RID_PLOT:-$(cat ${RIDFILE:-training/runs/pose5_arm_rid})}"
# FIGURE NAMES FROM ENV. A second arm is a second pair of names, not a second script.
FIG="${FIG:-figures/pose5_rel_lr1e4.png}"
SEL="${SEL:-figures/pose5_selection.png}"
TITLE_MAIN="${TITLE_MAIN:-5-SITE POSE INJECTION + rel, lr 1e-4, alpha 50}"
SD="${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}" \
    REF_RID="" RID="$RID" TAGP=pose5 \
    TITLE="5-SITE POSE INJECTION + rel (mean-centred iTM regression), lr 1e-4, alpha 50 ($RID)" \
    /home/ebrahim/Protenix/.venv/bin/python -I run_logging/plot_fullset.py "$FIG"

SD="${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}" \
    REF_RID="" RID="$RID" TAGP=pose5 \
    TITLE="$TITLE_MAIN -- SELECTION AT BUDGET N" \
    /home/ebrahim/Protenix/.venv/bin/python -I run_logging/plot_selection.py \
        "$SEL"
