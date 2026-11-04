#!/bin/bash
# BOTH WAVE 1 DELETIONS, gated, run immediately before the fold queue launches.
#
#   WAVE1_LISTS=<dir> LABEL_PID=<pid> nohup ./shell_scripts/wave1_reclaim_chain.sh \
#       > training/runs/reclaim_chain.log 2>&1 &
#
# WHY THE DELETIONS ARE LAST. The fold needs 619.6 GiB for training plus a transient 445.2 GiB for
# the validation 100-seed folds, against 532 GiB free. Something has to go. But nothing is deleted
# on the strength of a projection: each reclaim inventories, quarantines by rename, runs a REAL
# verifier with the data still recoverable, and only then unlinks -- and the unlink stage reads
# quota -v before and after, refusing to report success unless the freed bytes actually appear.
# A same-filesystem quarantine reclaims NO quota, so quarantine buys reversibility and unlink buys
# space; conflating the two would budget the fold against space that is not there.
#
# WHAT IS DELETED
#   A. training  1,979 trunk units -- the 220 structures whose EVERY interface is degenerate
#                (DockQ sd < 0.05 OR no acceptable pose) go from 10 seeds to 1. Not to zero: 64
#                clusters hold only ONE usable post-cutoff structure in the whole PDB, we folded
#                it, and it came back degenerate -- deleting would strand 73 clusters permanently.
#                One seed still carries PAE supervision, and its 5 poses may still carry
#                within-seed relative signal, so the ranking contribution is not assumed to be
#                zero. Frees ~565 GiB.
#   B. validation 3,709 trunk units -- all 53 structures go from 100 seeds to the 30 named by
#                training/valfull/seed_manifest.json (anchored-coverage-v2, digest
#                d86753b1a5a3d776). Frees ~492 GiB.
#
# WHAT IS NOT DELETED: every selected trunk unit, EVERY label npz (small, and keeping all 100
# means the seed selection can be recomputed later without refolding), results.csv with DockQ and
# ipTM for all 100 seeds, and all poses/refs/prep/jobs.
#
# THE COST, ACCEPTED: trimming any training structure to 1 seed breaks phase_k_train's uniform-seed
# assumption, so j2mnoz stops being resumable from resume_ep28.pt until variable-seed support
# lands. That support is required before the expanded-data training anyway.
#
# IT STOPS SHORT ON PURPOSE. This script never stops a trainer and never launches a fold.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
S=${WAVE1_LISTS:?set WAVE1_LISTS to the directory holding the target lists}
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }
q(){ $PY -I -c "
import re,subprocess
o=subprocess.run(['quota','-v'],capture_output=True,text=True).stdout
m=re.search(r'nvme1n1p1\s+(\d+)\s+\d+\s+(\d+)',o)
print('%.1f GiB free of %.1f'%((int(m.group(2))-int(m.group(1)))/1048576,int(m.group(2))/1048576))"; }

say "waiting for the label build (pid ${LABEL_PID:-none})"
while [ -n "${LABEL_PID:-}" ] && [ -d /proc/$LABEL_PID ]; do sleep 60; done
say "label build done"; tail -3 training/runs/val_labels_100.log

say "quota before: $(q)"

say "GATE 1/2 -- validation manifest"
$PY -I run_logging/verify_val_manifest.py --manifest training/valfull/seed_manifest.json || { say "VALIDATION GATE FAILED -- deleting nothing"; exit 1; }

say "RECLAIM A -- training, 1979 units, 220 structures to 1 seed each"
$PY -I run_logging/staged_reclaim.py --targets $S/delete_train_220.txt --journal training/runs/reclaim_train.json --stage inventory || exit 1
$PY -I run_logging/staged_reclaim.py --targets $S/delete_train_220.txt --journal training/runs/reclaim_train.json --stage quarantine || exit 1
$PY -I run_logging/staged_reclaim.py --targets $S/delete_train_220.txt --journal training/runs/reclaim_train.json --stage verify \
  --verify-cmd "$PY -I run_logging/verify_train_trim.py --trimmed $S/drop_train.txt" || { say "TRAIN VERIFY FAILED -- restoring"; $PY -I run_logging/staged_reclaim.py --targets $S/delete_train_220.txt --journal training/runs/reclaim_train.json --stage restore; exit 1; }
$PY -I run_logging/staged_reclaim.py --targets $S/delete_train_220.txt --journal training/runs/reclaim_train.json --stage unlink --i-have-verified || exit 1
say "quota after training trim: $(q)"

say "RECLAIM B -- validation, 3709 units, 100 -> 30 selected seeds"
$PY -I run_logging/staged_reclaim.py --targets $S/delete_targets.txt --journal training/runs/reclaim_val.json --stage inventory || exit 1
$PY -I run_logging/staged_reclaim.py --targets $S/delete_targets.txt --journal training/runs/reclaim_val.json --stage quarantine || exit 1
$PY -I run_logging/staged_reclaim.py --targets $S/delete_targets.txt --journal training/runs/reclaim_val.json --stage verify \
  --verify-cmd "$PY -I run_logging/verify_val_manifest.py --manifest training/valfull/seed_manifest.json" || { say "VAL VERIFY FAILED -- restoring"; $PY -I run_logging/staged_reclaim.py --targets $S/delete_targets.txt --journal training/runs/reclaim_val.json --stage restore; exit 1; }
$PY -I run_logging/staged_reclaim.py --targets $S/delete_targets.txt --journal training/runs/reclaim_val.json --stage unlink --i-have-verified || exit 1
say "quota after validation trim: $(q)"

say "BOTH RECLAIMS DONE. j2mnoz is still running and the fold has NOT been launched."
