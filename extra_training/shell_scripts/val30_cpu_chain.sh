#!/bin/bash
# CPU prep for the 30 new Wave 1 validation structures: per-job cifs, then bioassemblies, then
# the pool rows the fold needs. Same order as the training chain -- the per-job cif FIRST, because
# a label built from the whole deposited assembly cannot be compared to a pose of the job.
#
# phase_j_prepare_training_data is given the ALL-83 selection, not the 30. It prunes the bio-dir
# to match its selection, and handing it a 30-row selection against a dir holding the existing 53
# is what deleted 501 training bioassemblies. --prune is NOT passed; the 83-row selection covers
# exactly what is on disk plus the 30 new ones, so there is nothing to prune either way.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
MIN_MEM_GB=300
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }
mem(){ awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo; }
folders(){ tmux ls 2>/dev/null | grep -c w1fold; }
watchdog(){ while kill -0 "$1" 2>/dev/null; do
    M=$(mem); F=$(folders)
    [ "$M" -lt "$MIN_MEM_GB" ] && { say "WATCHDOG: MemAvailable ${M}GB -- stopping"; kill -TERM "$1"; return 1; }
    [ "$F" -lt 3 ] && { say "WATCHDOG: only $F training fold worker(s) -- stopping so this cannot be the cause"; kill -TERM "$1"; return 1; }
    sleep 30; done; }

say "MemAvailable $(mem) GB, training fold workers $(folders)"

say "STEP 1/3 -- per-job cifs for the 30 (antibody + polypeptide antigen only)"
nice -n 15 ionice -c 3 $PY -I phases/phase_j_job_cifs.py \
  --selection validation/full_val/val30_structures.csv \
  --sabdab-cifs SAbDab2_data/training_cifs \
  --src-dir SAbDab2_data/rcsb_validation_cifs \
  --out-dir validation/prep/job_cifs \
  --jobs-dir validation/jobs \
  --skipped validation/prep/skipped_job_cifs_val30.csv \
  > training/runs/val30_job_cifs.log 2>&1 &
P=$!; watchdog $P & W=$!; wait $P; RC=$?; kill $W 2>/dev/null
say "step 1 exit $RC; job cifs on disk: $(ls validation/prep/job_cifs | wc -l) of 83 (53 were there)"
tail -8 training/runs/val30_job_cifs.log
[ $RC -ne 0 ] && { say "STOPPING at step 1"; exit 1; }

say "STEP 2/3 -- bioassemblies for all 83 FROM the per-job cifs"
nice -n 15 ionice -c 3 $PY -I phases/phase_j_prepare_training_data.py \
  --selection validation/full_val/val83_structures.csv \
  --cif-dir validation/prep/job_cifs \
  --out-csv validation/prep/val83_indices.csv \
  --bio-dir validation/prep/job_bioassembly \
  --file-list validation/prep/val83_file_list.txt \
  --n-cpu 24 --run > training/runs/val30_prep.log 2>&1 &
P=$!; watchdog $P & W=$!; wait $P; RC=$?; kill $W 2>/dev/null
say "step 2 exit $RC; bioassemblies: $(ls validation/prep/job_bioassembly | wc -l) of 83"
tail -10 training/runs/val30_prep.log
[ $RC -ne 0 ] && { say "STOPPING at step 2 -- preparation exited $RC"; exit 1; }
# exit 0 is not completeness; check one bioassembly per selected structure
$PY -I - <<'EOF' || { say "STOPPING at step 2 -- bioassemblies incomplete"; exit 1; }
import csv, os, sys
sel = [r["pdb"] for r in csv.DictReader(open("validation/full_val/val83_structures.csv"))]
miss = [p for p in sel if not os.path.exists(f"validation/prep/job_bioassembly/{p}.pkl.gz")]
print(f"  completeness: {len(sel) - len(miss)} of {len(sel)} selected structure(s) have a bioassembly")
if miss:
    print(f"  MISSING {len(miss)}: {sorted(miss)[:20]}")
    sys.exit(1)
EOF

say "STEP 3/3 -- pool rows and chain roles for the 30"
nice -n 15 ionice -c 3 $PY -I run_logging/build_wave1_pool.py \
  --queue validation/full_val/val30_structures.csv \
  --src-cifs SAbDab2_data/rcsb_validation_cifs \
  --jobs-dir validation/jobs \
  --refs-dir validation/fold/refs \
  --out-pool validation/full_val/pool_val30.csv \
  --out-roles validation/fold/chain_roles_val30.csv \
  > training/runs/val30_pool.log 2>&1 &
P=$!; watchdog $P & W=$!; wait $P; RC=$?; kill $W 2>/dev/null
say "step 3 exit $RC"
tail -12 training/runs/val30_pool.log
say "MemAvailable $(mem) GB, training fold workers $(folders)"
say "VAL30 CPU CHAIN DONE -- pool NOT yet merged into validation/fold/pool.csv"
