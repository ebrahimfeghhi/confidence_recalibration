#!/usr/bin/env bash
# UNCROPPED PAE LOSS OVER THE TRAINING FOLD, for one checkpoint. One epoch, no step taken.
#
# THE REFERENCE IS training/runs/shipped_train_pae, the only run that has ever measured this. Every
# path and crop setting below is copied from its args.json rather than retyped.
#
# WHAT ACTUALLY MAKES IT UNCROPPED IS THE ABSENCE OF --crop-pool, not the crop size. Arm E trains
# with --crop-pool training/crop_final and a 256-token crop; this pass passes no crop pool at all,
# so each structure is scored whole. crop_size 3400 is wider than the largest structure in the pool
# (2,354 tokens) and crop_contact 5 is the reference's own value.
#
# NOTE the reference args.json also records a crop_margin key, which the trainer NO LONGER ACCEPTS
# -- it was renamed to --crop-contact. Copying that key verbatim killed all three shards instantly
# with "unrecognized arguments: --crop-margin 10". An old args.json is a record of what a run was
# given, not a guarantee the flag still exists.
#
# THE ARCHITECTURE COMES FROM THE ARM'S OWN args.json via run_logging/valpae_flags.py, never from
# memory: a hand-built argv once omitted the pose flags and measured a head without the pathway it
# was trained with. That script also deliberately refuses to copy --pae-balance-accept and
# --pae-row-frac, which are TRAINING choices -- a measurement has to be the plain per-pose mean or
# arms cannot be compared.
#
# --pae-split-accept 0.23 IS ADDED HERE even though the reference run has it unset: it only decides
# how the run REPORTS the good/bad split and takes no gradient, and without it the split has to be
# rebuilt from results.csv afterwards anyway.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BUNDLE:?set BUNDLE to the checkpoint to measure (empty string for the shipped head)}"
: "${OUT:?set OUT to the output run dir}"
CARD="${CARD:-4}"
# NO --seeds BY DEFAULT. A flat lo-hi range demands every seed for every structure, and the
# training fold is deliberately ragged (289 structures keep 15 seeds, 199 keep 1), so a range
# claimed 648 units with no label and the pass refused at launch. The manifest's own `seeds`
# column states each structure's seeds instead -- which is how the val83 measurement has run
# all along. SEEDS is kept as an override for a genuinely rectangular pool.
SEEDS="${SEEDS:-}"
PREFLIGHT="${PREFLIGHT:-training/full_train/alignment_preflight.json}"
FLAGS="${FLAGS:-}"

echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') UNCROPPED training PAE: $([ -n "$BUNDLE" ] && basename "$BUNDLE" || echo 'SHIPPED head')"
echo "    card $CARD -> $OUT"

LAYERNORM_TYPE=openfold CUDA_VISIBLE_DEVICES="$CARD" \
/home/ebrahim/Protenix/.venv/bin/python -I phases/phase_k_train.py \
  --loss-only --epochs 1 --save-every 0 --warmup-steps 0 \
  ${BUNDLE:+--resume "$BUNDLE"} ${BUNDLE:+--resume-allow-drift} \
  ${BUNDLE:+--resume-note "uncropped TRAINING PAE measurement of $(basename "$BUNDLE"); --loss-only writes no checkpoint"} \
  --out "$OUT" \
  --fold training/fold --labels training/fold/labels \
  --manifest-selection training/full_train/structures.csv \
  --alignment-preflight "$PREFLIGHT" \
  --pae-scored-pairs training/full_train/pool.csv \
  --crop-contact 5 --crop-size 3400 \
  --indices "$PWD/training/prep/job_indices.csv" \
  --bio-dir "$PWD/training/prep/job_bioassembly" \
  --mmcif-dir "$PWD/training/prep/job_cifs" \
  ${SEEDS:+--seeds "$SEEDS"} --minimal-items --nsample 5 --blocks-per-ckpt 0 \
  --pae-split-accept 0.23 \
  $FLAGS "$@"
