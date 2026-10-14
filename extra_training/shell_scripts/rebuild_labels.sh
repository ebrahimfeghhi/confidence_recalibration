#!/bin/bash
# Rebuild the label artifacts so the 2026-10-14 geometry fix reaches the data.
#
# WHY: phase_j_job_cifs.py now vetoes a residue placement the geometry refuses
# (drop_geometrically_impossible). Every job cif on disk predates that, so 139 of 501 structures
# carry a chain that Protenix's CA-too-far filter either deletes outright (104) or keeps with a wrong
# residue (34, silently). The fix changes nothing until these are rewritten.
#
# NOTHING IS DELETED. The current artifacts are COPIED aside first, under .pre_geomfix, so the old
# and new can be compared and the old restored.
#
# POSES ARE NOT TOUCHED. The fold reads sequences from the job jsons, which were always correct.
# This is CPU work only; the three GPU workers keep folding throughout.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
E=$R/extra_training
P=$E/training/prep
PY=/home/ebrahim/Protenix/.venv/bin/python
LOG=$P/rebuild_geomfix.log
NCPU=32                      # of 288; the fold's featurisation needs the rest
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "=== step 0: copy the current artifacts aside (nothing deleted) ==="
for d in job_cifs job_bioassembly; do
    [ -e "$P/$d.pre_geomfix" ] || cp -a "$P/$d" "$P/$d.pre_geomfix"
done
[ -e "$P/job_indices.csv.pre_geomfix" ] || cp -a "$P/job_indices.csv" "$P/job_indices.csv.pre_geomfix"
say "backups: $(ls -d $P/*.pre_geomfix | wc -l) item(s)"

say "=== step 1: rewrite all 501 job cifs with the geometric veto ==="
$PY "$E/phases/phase_j_job_cifs.py" >> "$LOG" 2>&1 || { say "STOP: job cifs failed"; exit 1; }
say "job cifs: $(ls $P/job_cifs/*.cif | wc -l)"

say "=== step 2: rebuild the bioassemblies and the indices csv ==="
$PY "$E/phases/phase_j_prepare_training_data.py" --run --n-cpu $NCPU >> "$LOG" 2>&1 \
    || { say "STOP: prepare failed"; exit 1; }
say "bioassemblies: $(ls $P/job_bioassembly/*.pkl.gz | wc -l)  indices rows: $(( $(wc -l < $P/job_indices.csv) - 1 ))"

say "=== step 3: does every label now cover the folded system? ==="
R_ROOT=$R $PY - <<'PYEOF' 2>&1 | tee -a "$LOG"
import csv, collections, glob, json, os
R = os.environ.get("R_ROOT")
fold = {}
for f in glob.glob(f"{R}/extra_training/training/jobs/*-final-updated.json"):
    p = os.path.basename(f).split("-")[0]
    s = [sq["proteinChain"]["sequence"] for t in json.load(open(f)) for sq in t["sequences"]
         if "proteinChain" in sq]
    fold[p] = (len(s), sum(len(x) for x in s))
lab = collections.defaultdict(set); tok = {}
for r in csv.DictReader(open(f"{R}/extra_training/training/prep/job_indices.csv")):
    for k in ("chain_1_id", "chain_2_id"):
        if r[k]: lab[r["pdb_id"]].add(r[k])
    tok[r["pdb_id"]] = int(r["num_tokens"])
bad = [p for p in sorted(fold) if p in lab and (len(lab[p]) != fold[p][0] or tok[p] != fold[p][1])]
print(f"structures compared       : {sum(1 for p in fold if p in lab)}")
print(f"label misses a chain NOW  : {len(bad)}   (was 106)")
for p in bad[:20]:
    print(f"   {p}: fold {fold[p][0]} chains / {fold[p][1]} tokens, label {len(lab[p])} / {tok[p]}")
missing = [p for p in fold if p not in lab]
print(f"structures with NO label  : {len(missing)} {missing[:10]}")
PYEOF
say "=== done ==="
