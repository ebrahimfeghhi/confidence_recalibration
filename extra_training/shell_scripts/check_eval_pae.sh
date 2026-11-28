#!/usr/bin/env bash
# VERIFY THE IN-PASS PAE LOSS AGAINST THE STANDALONE MEASUREMENT, on a few units.
#
# WHAT IS BEING CHECKED. phase_k_eval --pae-loss now computes the training PAE loss inside the
# DockQ pass's own forward. The quantity must equal what shell_scripts/val_pae_once.sh produced
# with a separate --loss-only pass over the same poses -- shipped good 3.0991 / bad 2.4336 over
# val83 -- or the integration has changed the number while keeping its name.
#
# A FEW UNITS, NOT THE SET. --nshards 300 takes roughly 1/300th of val83, ~8 units, which is
# enough to compare PER POSE against the stored rows. Per-pose equality is a far stronger check
# than two aggregate means agreeing, and it is the join -- not the arithmetic -- that this is for.
#
# SHARES A TRAINING CARD ON PURPOSE. The arms use ~12 GB of a 97 GB card, and this is one more
# model on card 4; it takes no additional GPU beyond the three already in use.
set -euo pipefail
cd "$(dirname "$0")/.."
SD=/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad
mkdir -p "$SD"
NSHARDS="${NSHARDS:-300}"
CARD="${CARD:-4}"

echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') eval --pae-loss smoke test, 1/${NSHARDS} of val83 on card ${CARD}"
CUDA_VISIBLE_DEVICES="$CARD" LAYERNORM_TYPE=openfold \
/home/ebrahim/Protenix/.venv/bin/python -I phases/phase_k_eval.py \
  --budget 30 --out "$SD/smoke_pae.json" \
  --fold validation/fold --results validation/fold/results.csv \
  --pool training/valfull83/pool.csv --roles training/valfull83/chain_roles.csv \
  --seed-base 400000 --seed-manifest training/valfull83/seed_manifest.json \
  --pair-iptm-only --pae-loss --pae-loss-labels validation/fold/labels \
  --rows-out "$SD/smoke_rows.csv" --shard 0 --nshards "$NSHARDS" --allow-partial
