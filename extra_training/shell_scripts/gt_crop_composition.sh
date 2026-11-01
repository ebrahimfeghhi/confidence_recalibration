#!/usr/bin/env bash
# Antibody vs antigen token counts in the ground-truth crop that section 9b used.
# Pinned: contact 5 A, margin 10 A, seed 400000, and the pool/trunks/labels/roles recorded in
# training/runs/union_crop/valfull_m5.npz's __meta. CPU only, takes no cards.
#   ./shell_scripts/gt_crop_composition.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p training/runs/union_crop
exec /home/ebrahim/Protenix/.venv/bin/python -I run_logging/gt_crop_composition.py \
  --pool   training/valfull/pool.csv \
  --trunks validation/fold/trunks \
  --labels validation/fold/labels \
  --roles  validation/fold/chain_roles.csv \
  --contact 5.0 --margin 10.0 --seed-base 400000 \
  --out training/runs/union_crop/gt_crop_composition.json
