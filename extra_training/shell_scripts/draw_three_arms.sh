#!/bin/bash
# ALL THREE ARMS, ONE FIGURE EACH. Overlaying them sets the panel limits from whichever arm is
# drawn first and clips the others, so each keeps its own axes; comparing them is reading the same
# panel across three files.
#
# EACH ARM NAMES ITS OWN SOURCE. 1e-3 and 1e-4 PAE-only kept their row directories, so they are
# drawn from rows and carry margin and rho. The PAE + held-out ranking loss arm kept only
# reports, so it is drawn from those plus whatever epochs the re-score has rows for by now.
set -u
set -o pipefail
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
SD=${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}

SD=training/runs/pae_ref_scores TITLE="PAE-ONLY, lr 1e-3 (oowhyw)" REF_RID="" RID=oowhyw TAGP=fs \
    $PY -I run_logging/plot_fullset.py figures/pae_lr1e3.png

SD="$SD" TITLE="PAE-ONLY, lr 1e-4 (6gci5b, stopped at 27)" REF_RID="" RID=6gci5b TAGP=lr4 \
    $PY -I run_logging/plot_fullset.py figures/pae_lr1e4.png

SD="$SD" REPORTS=training/runs/zsoft_scores TAGP=zs REF_RID="" RID=2tr3sj \
    TITLE="PAE + HELD-OUT RANKING LOSS (zsoftmax), lr 1e-3 (2tr3sj, stopped at 24)" \
    $PY -I run_logging/plot_fullset.py figures/pae_zsoft_lr1e3.png
