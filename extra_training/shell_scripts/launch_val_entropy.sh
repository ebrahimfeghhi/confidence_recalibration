#!/bin/bash
# ENTROPY MINIMISATION ON THE VALIDATION SET. Test-time adaptation: no ground truth in the
# objective, adapting on the 83 structures that are then scored.
#
# WHAT THE OBJECTIVE IS. --pae-entropy ALPHA REPLACES the labelled PAE cross-entropy with
# ALPHA * mean Shannon entropy of the head's OWN predicted PAE distribution over the scored
# cross-chain pairs. The label npz is still opened -- the unit pipeline and its coverage gates are
# built around it -- but `ld` is never read, so the gradient depends only on the model's output.
#
# IT IS TRANSDUCTIVE BY CONSTRUCTION AND THAT IS THE POINT. It adapts on valfull83 and is scored
# on valfull83. At inference you hold the complex and its 50 poses and no labels at all, so an
# objective needing none can adapt to the very structures being selected over -- which is the
# deployment setting, not a cheat. But a number from this is NOT comparable with the
# train-on-846/score-on-141 numbers in paper_results.md, and any report of it must say so.
#
# ITS FAILURE MODE IS BUILT IN. Entropy is minimised by putting all mass in ONE bin regardless of
# which bin, so this sharpens a confidently WRONG prediction exactly as happily as a right one,
# and nothing in the objective prefers a good pose's score over a bad one's. The labelled pae_loss
# is still computed as a read-only diagnostic. ENTROPY FALLING WHILE rho FALLS IS THE COLLAPSE,
# not progress -- read them together.
#
# UNCROPPED, unlike every recent arm. The val trunks are full structures at 30 seeds each; there
# is no val crop pool and none is wanted, since the eval is uncropped too.
#
# THE MANIFEST IS BUILT, NOT FOUND. Every val manifest in the repo stops at 53 structures --
# val_all_structures.csv and label_audit_validation.csv both predate the valfull83 pool.
# run_logging/build_val_manifest.py writes all 83 with their 30 seeds, cross-checked against the
# trunks on disk, from training/valfull83/seed_manifest.json. phase_k_train reads only `pdb`,
# `tokens` and `seeds` from a selection csv, so the `instance` column the training manifests carry
# -- and which does not exist for the 30 newer val structures -- is not needed.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

CARDS="${CARDS:-4,5,7}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-4}"
ALPHA="${ALPHA:-1.0}"
NSAMPLE="${NSAMPLE:-5}"
ACCUM="${ACCUM:-240}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-16}"
TRUNK_MEMMAP="${TRUNK_MEMMAP:-1}"
RESUME="${RESUME:-}"
MANIFEST="${MANIFEST:-train_val_splits/val83_manifest.csv}"
EVAL_POOL="${EVAL_POOL:-training/valfull83/pool.csv}"

# THE THREE PATHS THAT MUST LEAVE training/. --indices, --bio-dir and --mmcif-dir all DEFAULT to
# the training prep inside phase_k_train (lines 115-117), and the first draft of this launcher
# passed training/prep/all857_indices.csv explicitly -- which covers 0 of the 83 val structures,
# measured. Nothing downstream would have said "wrong set": a missing index is a per-structure
# skip, so the run would have come up, trained on whatever survived, and reported an epoch.
# validation/prep/job_indices.csv is NOT the file either -- it stops at 53, the same pre-valfull83
# boundary as every stale val manifest in the repo. Gate 4 re-measures all three every launch.
VAL_INDICES="${VAL_INDICES:-validation/prep/val83_indices.csv}"
VAL_BIO="${VAL_BIO:-validation/prep/job_bioassembly}"
VAL_CIF="${VAL_CIF:-validation/prep/job_cifs}"
DRY="${DRY:-1}"

# lr DEFAULTS TO 1e-4, AN ORDER BELOW THE TRAINING ARMS. TTA adapts on 2,490 units rather than
# 6,922 and has no label to anchor it, so the same 1e-3 that those arms used would take the head
# further per epoch on less signal. Raise it deliberately, not by inheriting a default.

# COPIED FROM launch_pae_crop_arm.sh, NOT RETYPED: the frozen architecture must match the arms
# this is meant to be read against, or the comparison is of two different heads.
TRAIN_ONLY='^(?!pairformer_stack\.),blocks\.0\.pair_transition\.,blocks\.1\.tri_mul_out\.,blocks\.1\.tri_mul_in\.,blocks\.2\.tri_mul_out\.,blocks\.2\.tri_mul_in\.,blocks\.2\.pair_transition\.,blocks\.3\.tri_mul_out\.,blocks\.3\.tri_mul_in\.,blocks\.3\.tri_att_start\.,blocks\.3\.pair_transition\.'

say "val entropy TTA: alpha=$ALPHA lr=$LR epochs=$EPOCHS cards=$CARDS resume=${RESUME:-shipped} DRY=$DRY"

# ---- GATE 1: the manifest exists, covers 83, and every seed it names has a trunk --------------
say "CPU gate: the 83-structure val manifest and its trunks"
$PY -I run_logging/build_val_manifest.py "$MANIFEST" || {
    say "REFUSING: the val manifest could not be built"; exit 4; }

# ---- GATE 2: adapt pool and eval pool are THE SAME, and that is deliberate --------------------
say "CPU gate: the adapt pool and the eval pool are the same file (transductive, by design)"
[ "$EVAL_POOL" = "training/valfull83/pool.csv" ] || {
    say "REFUSING: EVAL_POOL is $EVAL_POOL, not the valfull83 pool this adapts on"; exit 4; }

# ---- GATE 3: roles cover every structure the pool names --------------------------------------
say "CPU gate: validation/fold/chain_roles_all83.csv covers every pool structure"
$PY -I -c "
import csv,sys
pool={r['pdb'].strip().lower() for r in csv.DictReader(open('$EVAL_POOL'))}
roles={r['pdb'].strip().lower() for r in csv.DictReader(open('validation/fold/chain_roles_all83.csv'))}
miss=sorted(pool-roles)
print(f'  pool {len(pool)} structure(s), roles {len(roles)}, missing {len(miss)}')
sys.exit(1 if miss else 0)
" || { say "REFUSING: roles do not cover the pool; interface_chain_pairs would SKIP those silently"; exit 4; }

# ---- GATE 4: indices, bioassemblies and cifs each cover all 83 ------------------------------
say "CPU gate: $VAL_INDICES, $VAL_BIO and $VAL_CIF each cover every pool structure"
$PY -I -c "
import csv, glob, os, sys
pool = {r['pdb'].strip().lower() for r in csv.DictReader(open('$EVAL_POOL'))}
rows = list(csv.DictReader(open('$VAL_INDICES')))
key = 'pdb' if (rows and 'pdb' in rows[0]) else 'pdb_id'
idx = {str(r[key]).strip().lower()[:4] for r in rows}
bio = {os.path.basename(p).split('.')[0].lower()[:4] for p in glob.glob('$VAL_BIO/*')}
cif = {os.path.basename(p).split('.')[0].lower()[:4] for p in glob.glob('$VAL_CIF/*')}
bad = 0
for name, have in (('indices', idx), ('bio-dir', bio), ('mmcif-dir', cif)):
    miss = sorted(pool - have)
    print(f'  {name}: covers {len(pool) - len(miss)}/{len(pool)}' + (f', MISSING {miss[:8]}' if miss else ''))
    bad += len(miss)
sys.exit(1 if bad else 0)
" || { say "REFUSING: one of the three val inputs does not cover all 83; missing structures would be SKIPPED, not raised"; exit 4; }

ARGS=(--minimal-items
      --pae-only --alpha-pae 1.0 --alpha-confidence 1.0
      --pae-entropy "$ALPHA"
      --accum "$ACCUM" --nsample "$NSAMPLE"
      --skip-dead-ops --pose-perblock --pose-perblock-copy --frame-direction --lean-single
      --train-eval-pool "$EVAL_POOL" --train-eval-fold validation/fold
      --train-eval-every 5 --train-eval-budget 30 --train-eval-seed-base 400000
      --epochs "$EPOCHS" --lr "$LR")

say "argv the trainer would receive:"
printf '    %s\n' "${ARGS[@]}"
say "NOT passed, deliberately: --crop-pool (the val trunks are uncropped), --rank-loss,"
say "  --l2-sp, --lora-rank. The objective is entropy alone."

if [ "$DRY" = "1" ]; then
  say "DRY RUN -- nothing launched. Re-run with DRY=0 to go."
  exit 0
fi

B=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$B" = "0" ] || { say "REFUSING: cards_busy reports '$B'"; exit 3; }
M=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
[ "${M:-0}" -lt 250 ] && { say "REFUSING: MemAvailable ${M} GB under the 250 GB floor"; exit 3; }
say "gates clear: cards $CARDS free, MemAvailable ${M} GB"

# THE LABEL GATE IS SKIPPED HERE, AND ONLY HERE. Measured 2026-11-12 with the fixed scan
# (phase_k_fingerprint_scan.py --fold validation/fold): 1,590 of 2,490 val labels differ on
# bio+code across 53 structures, because all 83 bioassemblies were re-dumped at 7:34 PM on
# 2026-11-04 roughly five hours after those labels were written. --allow-stale-labels makes the
# trainer accept exactly those four provenance fields, and the trainer REFUSES that flag unless
# --pae-entropy is set, so it cannot leak into a labelled arm. --skip-gate is then needed only
# because the launch-time preflight has no notion of "this objective does not read the label".
#
# WHAT IT COSTS: the labelled columns of trajectory.csv are not a diagnostic for this run. Read
# pae_entropy for the objective and the held-out DockQ/rho for whether it helped.
LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py --cards "$CARDS" --skip-gate \
  --numa-map 0:0,1:0,2:1 --note "val entropy TTA alpha=$ALPHA lr=$LR" -- "${ARGS[@]}" \
  --alignment-preflight training/alignment_preflight_sharded.json \
  --manifest-selection "$MANIFEST" \
  --fold validation/fold --indices "$VAL_INDICES" \
  --bio-dir "$VAL_BIO" --mmcif-dir "$VAL_CIF" \
  --pae-scored-pairs "$EVAL_POOL" \
  --pae-scored-roles validation/fold/chain_roles_all83.csv \
  --lean-readouts --train-only "$TRAIN_ONLY" --expect-trainable 113 \
  --warmup-steps 20 --seed 0 --clip-norm 25.0 \
  --blocks-per-ckpt 1 --ckpt-token-threshold 1250 \
  --prefetch-units 2 --prefetch-workers "$PREFETCH_WORKERS" \
  ${TRUNK_MEMMAP:+--trunk-memmap} \
  ${RESUME:+--resume "$RESUME"} \
  --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 250 --save-every 250 \
  --require-replay training/extended_mixed_replay.log --allow-code-drift \
  --allow-stale-labels
say "launch.py returned $?"
