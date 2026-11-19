#!/usr/bin/env bash
# ONE HELD-OUT PAE MEASUREMENT of a single resume bundle, on the val83 population.
#
#   BUNDLE=training/runs/<rid>/resume_ep10.pt OUT=<dir> ./shell_scripts/val_pae_once.sh
#
# WHY A SCRIPT AND NOT A HAND-TYPED ARGV: the argv below is copied from what
# val_loss_checkpoint builds for its own child, so a measurement run by hand measures the same
# thing the in-run monitor would. Retyping a launch argv has cost four failed launches in twenty
# minutes in this project.
#
# --loss-only READS THE WEIGHTS AND WRITES NO CHECKPOINT, so the bundle's schedule is not
# continued and running this cannot disturb the arm it measures.
#
# UNCROPPED, on the user's instruction: --val-loss-uncropped suppresses the --crop-all-poses the
# training side uses, so the held-out loss is over whole structures rather than 256-token crops.
#
# THE PREFLIGHT MARKER IS THE val83 ONE, built 2026-11-19: 83 structures, 0 mismatches, digest
# 43f0f1839c0e6c6e. It is bound to the per-structure seed lists in valfull83/structures.csv --
# the val trunks are a SUBSET of 400000-400099, 30 each but not the same 30, and 8 structures
# have no free_400000 at all, so a flat range here would be refused on the digest.
set -euo pipefail
cd "$(dirname "$0")/.."
# BUNDLE IS OPTIONAL. Empty means the SHIPPED head -- no --resume, so the model is built from
# /home/ebrahim/checkpoint/protenix-v2.pt as it ships. That is the right baseline for an arm whose
# added terms are zero-initialised: at epoch 0 they contribute nothing, so shipped IS the arm's
# starting point, measured on the same labels with the same code path.
BUNDLE="${BUNDLE:-}"
OUT="${OUT:?set OUT to a directory for the result}"
[ -z "$BUNDLE" ] || [ -f "$BUNDLE" ] || { echo "STOP: no bundle at $BUNDLE"; exit 2; }
mkdir -p "$OUT"
PY=/home/ebrahim/Protenix/.venv/bin/python

echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') held-out PAE on val83: $([ -n "$BUNDLE" ] && basename "$BUNDLE" || echo "SHIPPED head (no resume)")"
LAYERNORM_TYPE=openfold CUDA_VISIBLE_DEVICES="${CARD:-4}" \
"$PY" -I phases/phase_k_train.py \
  --loss-only \
  ${BUNDLE:+--resume "$BUNDLE" --resume-allow-drift \
  --resume-note "held-out PAE measurement of $(basename "$BUNDLE") on val83; --loss-only writes no checkpoint"} \
  --fold validation/fold \
  --labels validation/fold/labels \
  --manifest-selection training/valfull83/structures.csv \
  --alignment-preflight training/valfull83/alignment_preflight.json \
  --minimal-items \
  --pae-scored-pairs training/valfull83/pool.csv \
  --pae-scored-roles validation/fold/chain_roles_all83.csv \
  --indices validation/prep/val83_indices.csv \
  --bio-dir validation/prep/job_bioassembly \
  --mmcif-dir validation/prep/job_cifs \
  --out "$OUT" \
  --epochs 1 --save-every 0 --warmup-steps 0 \
  "$@" 2>&1 | tee "$OUT/val_pae.log" | grep -viE "^2026-|INFO|WARNING" | tail -25
