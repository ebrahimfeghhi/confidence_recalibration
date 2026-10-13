#!/usr/bin/env bash
# Regenerates the seed-depth figures in extra_training/docs/figures from the pooled score tables:
# one pooled figure per dataset, each as png, pdf and csv. CPU only; safe to run during a fold.
#   bash regen_seed_depth_figures.sh [outdir]
set -euo pipefail
OUT=${1:-/home/ebrahim/tts_abag/extra_training/docs/figures}
PY=${PY:-/home/ebrahim/Protenix/.venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
V="$HERE/val_seed_depth_curve.py"

# ONE POOLED FIGURE PER DATASET is what the docs carry. The per-arm views (--arm "all new" /
# --arm mixed / --compare-arms <class>) are generated on demand and not kept in docs/figures.
$PY "$V" --label "Validation set" --out "$OUT/val_seed_depth"
$PY "$V" --dataset pxm2025 --arm-source sabdab --label "PXM-2025" --out "$OUT/pxm_seed_depth"
