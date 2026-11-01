#!/usr/bin/env bash
# How much does the PER-POSE interface crop vary between one structure's own candidates?
#
# PINNED so the number is re-derivable: the ORIGINAL thresholds (contact 5 A, margin 10 A), the
# exact evaluated seed range (400000..400029, the same 30 seeds phase_k_eval --budget 30 scores),
# and the same pool/trunks/labels the union masks were built from -- those paths are recorded in
# training/runs/union_crop/valfull_m5.npz's __meta, so the two measurements describe the same data.
#
# CPU only. Takes no cards and is safe to run beside a training arm.
#
#   ./shell_scripts/perpose_crop_spread.sh [margin]        # margin defaults to 10
set -euo pipefail
cd "$(dirname "$0")/.."
MARGIN="${1:-10.0}"
OUT="training/runs/union_crop/perpose_spread_m${MARGIN}.json"
mkdir -p training/runs/union_crop
exec /home/ebrahim/Protenix/.venv/bin/python -I run_logging/perpose_crop_spread.py \
  --pool   training/valfull/pool.csv \
  --trunks validation/fold/trunks \
  --labels validation/fold/labels \
  --contact 5.0 \
  --margin "$MARGIN" \
  --seed-base 400000 \
  --budget 30 \
  --out "$OUT"
