#!/usr/bin/env bash
# Regenerates every seed-depth figure in extra_training/docs/figures from the pooled score tables.
# 12 sets x (png, pdf, csv). CPU only; safe to run while folds are on the GPUs.
#   bash regen_seed_depth_figures.sh [outdir]
set -euo pipefail
OUT=${1:-/home/ebrahim/tts_abag/extra_training/docs/figures}
PY=${PY:-/home/ebrahim/Protenix/.venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
V="$HERE/val_seed_depth_curve.py"

# Validation set (sabdab_val2): arm from the frozen split csv.
$PY "$V" --label "Validation set"                      --out "$OUT/val_seed_depth"
$PY "$V" --label "Validation set" --arm "all new"      --out "$OUT/val_seed_depth_allnew"
$PY "$V" --label "Validation set" --arm mixed          --out "$OUT/val_seed_depth_mixed"
for C in acceptable decoy "no acceptable"; do
  $PY "$V" --label "Validation set" --compare-arms "$C" --out "$OUT/val_${C// /_}_by_arm"
done

# PXM-2025: arm derived from SAbDab2 membership, because there is no split csv for a benchmark.
P=(--dataset pxm2025 --arm-source sabdab --label "PXM-2025")
$PY "$V" "${P[@]}"                      --out "$OUT/pxm_seed_depth"
$PY "$V" "${P[@]}" --arm "all new"      --out "$OUT/pxm_seed_depth_all_new"
$PY "$V" "${P[@]}" --arm mixed          --out "$OUT/pxm_seed_depth_mixed"
for C in acceptable decoy "no acceptable"; do
  $PY "$V" "${P[@]}" --compare-arms "$C" --out "$OUT/pxm_${C// /_}_by_arm"
done
