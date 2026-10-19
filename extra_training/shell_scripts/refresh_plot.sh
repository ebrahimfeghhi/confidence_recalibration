#!/bin/bash
# Regenerate the training-pool plot. Run ONCE, from the health check, not as a daemon.
#
# There used to be a polling watcher here. It was edited with `sed -i` while running, and bash
# reads a script incrementally by byte offset -- so the running copy kept reading a deleted inode
# at shifted offsets and wedged, silently, leaving a stale plot. A one-shot script invoked fresh
# each time cannot have that failure mode.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
exec /home/ebrahim/Protenix/.venv/bin/python -I run_logging/plot_train_eval.py \
  "n3s8gb=unmasked PAE" \
  "pykssn+x3888d+tgfiqx+89uwjx+owdyyp=masked, PAE-only from ep27" \
  "krah8z=fresh PAE-only" \
  "dapghi=fresh PAE-only (stopped ep6)" \
  "x5a4uj=PAE masked, lr 1e-2" \
  "q7nyu6=overfit30 probe, lr 1e-1" \
  "hhzddi+jy6sah+uqtnbn=orange forked at ep57, lr 1e-2" \
  --stock-dockq 0.0420 --stock-frac 0.0 \
  --ceiling-dockq 0.4216 --ceiling-frac 0.6883 \
  --oracle-dockq 0.5098 --oracle-frac 0.8831 \
  --out training/runs/train_eval_plot.png "$@"
