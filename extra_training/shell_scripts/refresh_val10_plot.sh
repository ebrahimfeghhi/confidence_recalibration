#!/bin/bash
# Regenerate the held-out val10 plot. Run ONCE, from the progress check, never as a daemon --
# a polling watcher here was edited with sed -i while running and wedged, because bash reads a
# script incrementally by byte offset and kept reading a deleted inode at shifted offsets.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
exec /home/ebrahim/Protenix/.venv/bin/python -I run_logging/plot_val10.py \
  "e4tvxe+f8rqck=lr 1e-1" \
  "335psh=lr 1e-2 (anchor)" \
  --out training/runs/val10_plot.png "$@"
