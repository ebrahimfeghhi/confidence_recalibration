#!/bin/bash
# CPU prep for the 30 new Wave 1 validation structures, reproducible end to end from the SAbDab2
# release archive: instance cifs, per-job cifs, bioassemblies, pool rows, staged references, and
# the merge into the fold's pool.
#
# EVERY STEP IS IN HERE BECAUSE NONE OF IT SHOULD BE DONE BY HAND. The first pass at this had the
# extraction, the queue json and the pool merge done interactively, which is not reproducible and
# left the script failing on inputs it could not build for itself.
#
# Each step fails closed: a nonzero exit stops the chain, and because exit 0 is not completeness,
# each step also verifies its own output against the selection and names what is missing.
#
# phase_j_prepare_training_data is given the ALL-83 selection, not the 30. It prunes the bio-dir to
# match its selection, and handing it a 30-row selection against a dir holding the existing 53 is
# what deleted 501 training bioassemblies. --prune is NOT passed.
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
step(){ P=$!; watchdog $P & W=$!; wait $P; RC=$?; kill $W 2>/dev/null; return $RC; }

say "MemAvailable $(mem) GB, training fold workers $(folders)"

say "STEP 1/6 -- instance cifs from the release archive (the corpus is a partial extraction)"
nice -n 15 ionice -c 3 $PY -I run_logging/extract_instance_cifs.py \
  --selection validation/full_val/val30_structures.csv --apply \
  > training/runs/val30_instance_cifs.log 2>&1 &
step || { say "STOPPING at step 1"; tail -5 training/runs/val30_instance_cifs.log; exit 1; }
tail -3 training/runs/val30_instance_cifs.log

say "STEP 2/6 -- per-job cifs for the 30 (antibody + polypeptide antigen only)"
nice -n 15 ionice -c 3 $PY -I phases/phase_j_job_cifs.py \
  --selection validation/full_val/val30_structures.csv \
  --sabdab-cifs SAbDab2_data/training_cifs \
  --src-dir SAbDab2_data/rcsb_validation_cifs \
  --out-dir validation/prep/job_cifs \
  --jobs-dir validation/jobs \
  --skipped validation/prep/skipped_job_cifs_val30.csv \
  > training/runs/val30_job_cifs.log 2>&1 &
step || { say "STOPPING at step 2"; tail -8 training/runs/val30_job_cifs.log; exit 1; }
grep -E "^note|^WARNING|wrote .* job cif" training/runs/val30_job_cifs.log

say "STEP 3/6 -- bioassemblies for all 83 FROM the per-job cifs"
nice -n 15 ionice -c 3 $PY -I phases/phase_j_prepare_training_data.py \
  --selection validation/full_val/val83_structures.csv \
  --cif-dir validation/prep/job_cifs \
  --out-csv validation/prep/val83_indices.csv \
  --bio-dir validation/prep/job_bioassembly \
  --file-list validation/prep/val83_file_list.txt \
  --n-cpu 24 --run > training/runs/val30_prep.log 2>&1 &
step || { say "STOPPING at step 3 -- preparation failed"; tail -8 training/runs/val30_prep.log; exit 1; }
$PY -I - <<'EOF' || { say "STOPPING at step 3 -- bioassemblies incomplete"; exit 1; }
import csv, os, sys
sel = [r["pdb"] for r in csv.DictReader(open("validation/full_val/val83_structures.csv"))]
miss = [p for p in sel if not os.path.exists(f"validation/prep/job_bioassembly/{p}.pkl.gz")]
print(f"  completeness: {len(sel)-len(miss)} of {len(sel)} selected structure(s) have a bioassembly")
if miss:
    print(f"  MISSING {len(miss)}: {sorted(miss)[:20]}"); sys.exit(1)
EOF

say "STEP 4/6 -- pool rows and chain roles for the 30"
# the builder takes a JSON queue, derived here so it cannot drift from the selection
$PY -I - <<'EOF' || { say "STOPPING -- could not build the queue json"; exit 1; }
import csv, json
rows = list(csv.DictReader(open("validation/full_val/val30_structures.csv")))
json.dump({"queue": [{"pdb": r["pdb"], "instance": r["instance"]} for r in rows]},
          open("validation/full_val/val30_queue.json", "w"), indent=1)
print(f"  queue json: {len(rows)} structure(s)")
EOF
# --src-cifs is the INSTANCE cifs. Pointing it at the deposited entries was measured and rejected:
# multi-copy entries lose interfaces 57% of the time and DockQ shifts up to 0.12, because the
# instance cifs are trimmed and DockQ is defined over the reference's interface residues.
nice -n 15 ionice -c 3 $PY -I run_logging/build_wave1_pool.py \
  --queue validation/full_val/val30_queue.json \
  --src-cifs SAbDab2_data/training_cifs \
  --jobs-dir validation/jobs \
  --refs-dir validation/fold/refs \
  --out-pool validation/full_val/pool_val30.csv \
  --out-roles validation/fold/chain_roles_val30.csv \
  > training/runs/val30_pool.log 2>&1 &
step || { say "STOPPING at step 4 -- pool builder failed"; tail -12 training/runs/val30_pool.log; exit 1; }
grep -E "pool rows|role rows|relabelled|problem" training/runs/val30_pool.log
$PY -I - <<'EOF' || { say "STOPPING at step 4 -- pool incomplete"; exit 1; }
import csv, os, sys
sel = [r["pdb"] for r in csv.DictReader(open("validation/full_val/val30_structures.csv"))]
for f in ("validation/full_val/pool_val30.csv", "validation/fold/chain_roles_val30.csv"):
    if not os.path.exists(f):
        print(f"  MISSING OUTPUT: {f}"); sys.exit(1)
pool = {r["pdb"] for r in csv.DictReader(open("validation/full_val/pool_val30.csv"))}
roles = {r["pdb"] for r in csv.DictReader(open("validation/fold/chain_roles_val30.csv"))}
nm, rm = sorted(set(sel)-pool), sorted(set(sel)-roles)
print(f"  pool covers {len(pool)} of {len(sel)}; roles cover {len(roles)}")
if nm: print(f"  NOT IN THE POOL ({len(nm)}): {nm[:20]}")
if rm: print(f"  NO CHAIN ROLES ({len(rm)}): {rm[:20]}")
sys.exit(1 if nm or rm else 0)
EOF

say "STEP 5/6 -- stage the DockQ reference for each structure"
nice -n 15 $PY -I run_logging/stage_val_refs.py --apply > training/runs/val30_refs.log 2>&1 &
step || { say "STOPPING at step 5 -- staging failed"; tail -8 training/runs/val30_refs.log; exit 1; }
grep -E "reference|RELABELLED|staged|PROBLEM" training/runs/val30_refs.log

say "STEP 6/6 -- merge the pool into validation/fold/pool.csv"
$PY -I - <<'EOF' || { say "STOPPING at step 6 -- merge or verification failed"; exit 1; }
import csv, os, sys, shutil
VG = "validation/fold"
base = VG + "/pool.csv.pre_val13"      # the pool as it was before any Wave 1 row was added
if not os.path.exists(base):
    print(f"  MISSING BASELINE {base}; refusing to guess what the original pool was"); sys.exit(1)
old = list(csv.DictReader(open(base)))
new = list(csv.DictReader(open("validation/full_val/pool_val30.csv")))
seen = {(r["pdb"], r["chain1"], r["chain2"]) for r in old}
rows = old + [r for r in new if (r["pdb"], r["chain1"], r["chain2"]) not in seen]
with open(VG + "/pool.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["pdb", "chain1", "chain2"]); w.writeheader()
    w.writerows([{k: r[k] for k in ("pdb", "chain1", "chain2")} for r in rows])
print(f"  pool.csv: {len(rows)} row(s) over {len({r['pdb'] for r in rows})} structure(s)")

# THE GATE THE LAUNCHER WILL APPLY, applied here so the chain does not leave a queue that cannot
# be folded: every queued structure needs pool rows AND a staged reference.
q = set()
for ln in open(VG + "/queue.tsv"):
    p = ln.rstrip("\n").split("\t")
    if len(p) >= 5: q.add(p[1])
pool = {r["pdb"] for r in rows}
nop = sorted(q - pool)
nor = sorted(p for p in q if not os.path.exists(f"validation/refs/{p}.cif"))
print(f"  queued {len(q)} structure(s): {len(nop)} unpooled, {len(nor)} without a reference")
if nop: print(f"  UNPOOLED: {nop[:20]}")
if nor: print(f"  NO REFERENCE: {nor[:20]}")
sys.exit(1 if nop or nor else 0)
EOF

say "MemAvailable $(mem) GB, training fold workers $(folders)"
say "VAL30 CPU CHAIN DONE -- every queued structure has pool rows and a staged reference"
