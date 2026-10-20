#!/bin/bash
# Regenerate the 4-metric plot AND the results table for the full training run.
#
# ONE PLACE, so the watch, the plot and the doc cannot drift from each other -- they all read the
# same run directories through the same two readers. The lineage is joined here and nowhere else:
# x3k2m4 + jhgfxy + 2w5to5 is one experiment restarted twice for code fixes, and anything that
# treats them separately reads three short curves instead of one.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
/home/ebrahim/Protenix/.venv/bin/python -I run_logging/plot_full_train.py \
  "x3k2m4+jhgfxy+2w5to5=full set, lr 1e-2" \
  --out training/runs/full_train_plot.png

/home/ebrahim/Protenix/.venv/bin/python -I run_logging/results_table.py \
  x3k2m4+jhgfxy+2w5to5 \
  --out docs/full_train_results.md
