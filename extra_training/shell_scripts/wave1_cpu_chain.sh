#!/bin/bash
# Wave 1 CPU chain, correct order this time: per-job cifs FIRST, then bioassemblies FROM them.
# The whole-entry shortcut is what went wrong -- the assembly is a median 1,095 tokens against the
# job's 644, so "a label built from the assembly cannot be compared to a pose of the job".
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
    [ "$F" -lt 3 ] && { say "WATCHDOG: $F fold worker(s) -- stopping so this cannot be the cause"; kill -TERM "$1"; return 1; }
    sleep 30; done; }

say "STEP 1/2 -- per-job cifs for the 356 (antibody + polypeptide antigen only)"
nice -n 15 ionice -c 3 $PY -I phases/phase_j_job_cifs.py \
  --selection training/full_train/wave1_structures.csv \
  --splits SAbDab2_data/zenodo_splits/splits_final/abag_split.csv \
  --src-dir data/rcsb_training_cifs --out-dir training/prep/job_cifs \
  --jobs-dir training/jobs \
  --skipped training/prep/skipped_job_cifs_wave1.csv > training/runs/wave1_job_cifs.log 2>&1 &
P=$!; watchdog $P & W=$!; wait $P; RC=$?; kill $W 2>/dev/null
say "step 1 exit $RC; job cifs on disk: $(ls training/prep/job_cifs | wc -l) (501 were already there)"
tail -6 training/runs/wave1_job_cifs.log
[ $RC -ne 0 ] && { say "STOPPING at step 1"; exit 1; }

say "STEP 2/2 -- bioassemblies for all 857 FROM the per-job cifs"
nice -n 15 ionice -c 3 $PY -I phases/phase_j_prepare_training_data.py \
  --selection training/full_train/all857_structures.csv \
  --splits SAbDab2_data/zenodo_splits/splits_final/abag_split.csv \
  --cif-dir training/prep/job_cifs \
  --out-csv training/prep/all857_indices.csv \
  --bio-dir training/prep/job_bioassembly \
  --file-list training/prep/all857_file_list.txt \
  --n-cpu 24 --run > training/runs/wave1_prep2.log 2>&1 &
P=$!; watchdog $P & W=$!; wait $P; RC=$?; kill $W 2>/dev/null
say "step 2 exit $RC; bioassemblies: $(ls training/prep/job_bioassembly | wc -l) of 857"
tail -8 training/runs/wave1_prep2.log
say "MemAvailable $(mem) GB, fold workers $(folders)"
say "CPU CHAIN DONE"
