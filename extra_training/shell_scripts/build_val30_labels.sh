#!/bin/bash
# LABEL CACHE for the 30 wave-1 validation structures, which the 100-seed fold left unlabelled.
#
# WHY NOT build_valfull_labels.sh. That script was written for the original 53 and passes
# --indices validation/prep/job_indices.csv, which names those 53 and NONE of these 30. Pointed at
# the new structures it reports "no dataset datapoint for <pdb>" for every unit and skips all 100 --
# which reads exactly like missing upstream data and is not. The indices for all 83 are in
# validation/prep/val83_indices.csv, written by step 3 of val30_cpu_chain.sh, and the bioassemblies
# are already there as <pdb>.pkl.gz for all 83. Nothing upstream is missing; only this argument was
# wrong. I diagnosed it as a blocker first, so the comment is here to stop the next reader doing
# the same.
#
# WHY ALL 100 SEEDS AND NOT THE SELECTED 30. phase_j_build_cache takes --limit, not a seed list,
# and the anchored-coverage selection is non-contiguous -- 30 of the first 100 in no particular
# order. Building the first 30 by directory order would label the WRONG units. Labels are ~0.16 MB,
# so all 3,000 cost ~0.5 GiB, which is cheaper than the risk of a mismatch; the ones whose trunks
# the trim later removes are simply orphaned.
#
# THE ORDER MATTERS, and the gate enforces it: build labels -> verify_val_manifest -> trim. The
# labels are built FROM the trunks, so trimming first would leave nothing to build them from.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
MANIFEST="${MANIFEST:-validation/fold/seed_manifest_wave1.json}"
JOBS="${JOBS:-4}"
MIN_MEM_GB="${MIN_MEM_GB:-250}"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

# RESOURCE FLOOR, because this box is shared. The user's standing rule is to keep MemAvailable
# above 250 GB for other users, so refuse to start below it rather than discover it halfway.
M=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
say "MemAvailable ${M} GB, ${JOBS} concurrent worker(s)"
if [ "$M" -lt "$MIN_MEM_GB" ]; then
  say "REFUSING: MemAvailable ${M} GB is below the ${MIN_MEM_GB} GB floor."
  exit 1
fi

$PY -I -c "
import json, sys
m = json.load(open('$MANIFEST'))
print('\n'.join(sorted(m['structures'])))
" > /tmp/val30_label_pdbs.$$ || { say "could not read $MANIFEST"; exit 1; }
N=$(wc -l < /tmp/val30_label_pdbs.$$)
say "building labels for $N structure(s) x 100 unit(s)"

nice -n 19 ionice -c 3 xargs -P "$JOBS" -I{} -a /tmp/val30_label_pdbs.$$ \
  env LAYERNORM_TYPE=openfold OMP_NUM_THREADS=2 "$PY" -I phases/phase_j_build_cache.py \
    --fold validation/fold \
    --out-dir validation/fold/labels \
    --indices validation/prep/val83_indices.csv \
    --bio-dir validation/prep/job_bioassembly \
    --mmcif-dir validation/prep/job_cifs \
    --only {} --limit 100 \
    --report validation/fold/label_build_{}.csv
RC=$?
rm -f /tmp/val30_label_pdbs.$$
say "label build exit $RC"

# EXIT 0 IS NOT COMPLETENESS. The builder skips a unit it cannot build and still exits 0, which is
# how 1,290 missing labels once got as far as a launch. The gate below is the real answer.
say "gating the manifest against the labels just built"
$PY -I run_logging/verify_val_manifest.py \
  --manifest "$MANIFEST" \
  --trunks validation/fold/trunks \
  --labels validation/fold/labels \
  --results validation/fold/results.csv
G=$?
say "verify_val_manifest exit $G (0 = every selected unit has a label; the trim may proceed)"
[ $RC -ne 0 ] || [ $G -ne 0 ] && { say "NOT READY TO TRIM"; exit 1; }
say "READY: build the trim target list and run staged_reclaim"
