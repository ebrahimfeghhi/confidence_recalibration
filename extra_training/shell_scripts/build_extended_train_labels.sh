#!/bin/bash
# LABELS for the extended 853-structure training population.
#
# WHY EVERY LABEL IS REBUILT, NOT JUST THE 3,981 MISSING ONES. The population needs
# --indices training/prep/all857_indices.csv: the default job_indices.csv names 501 structures and
# NONE of the 356 from this wave, so pointed at them the builder reports "no dataset datapoint" per
# unit and skips all of them -- the identical trap that made build_valfull_labels.sh unusable for
# the 30 new validation structures. The indices path is part of the unit fingerprint, so switching
# it invalidates the 2,989 labels built under the old one and they rebuild.
#
# THAT REBUILD CHANGES NO SUPERVISION, and it was verified rather than assumed. 10fj/free_500000
# was rebuilt against the saved original: of 18 arrays, 16 are bit-identical and the only two that
# differ are `dataset_index` (the row number, which genuinely differs between the two indices
# files) and `fingerprint` (which encodes the inputs by construction). At 0.74 s/unit the whole
# 6,970-unit population costs ~86 min of one core, so uniform provenance is cheaper than a mixed
# label set whose completeness check would have to know which file each label came from.
#
# WHY --unit-seeds. --only takes a STRUCTURE and then labels every trunk it has. While the seed
# chain is running, that includes the seed being written this second; a torn npz is not reliably a
# load error, so this pass is restricted to seeds that no longer move. Run it again with a wider
# range once the chain reports complete -- it is incremental by fingerprint, so the second pass
# only builds what the first could not see.
#
# NO --require-complete HERE. It asserts the queue's contract, which this population deliberately
# does not match: 416 structures were trimmed to a single trunk, so by queue standards they are
# 9/10 missing. Completeness for training is asserted by the trainer's own manifest, which is the
# thing that must refuse a hole.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
LO="${LO:-500000}"
HI="${HI:-500009}"
JOBS="${JOBS:-6}"
MIN_MEM_GB="${MIN_MEM_GB:-250}"
LIST="${LIST:-docs/extended_train_structures.txt}"
# EXPORTED because the coverage check below is a separate python process that reads them.
export LO HI LIST
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

[ -s "$LIST" ] || { say "REFUSING: $LIST is missing or empty"; exit 1; }
N=$(grep -cv '^#' "$LIST")
say "labels for $N structure(s), seeds $LO-$HI, $JOBS worker(s)"

M=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
say "MemAvailable ${M} GB"
if [ "$M" -lt "$MIN_MEM_GB" ]; then
  say "REFUSING: MemAvailable ${M} GB is under the ${MIN_MEM_GB} GB floor for other users."
  exit 1
fi

mkdir -p training/fold/label_reports
grep -v '^#' "$LIST" | nice -n 19 ionice -c 3 xargs -P "$JOBS" -I{} \
  env LAYERNORM_TYPE=openfold OMP_NUM_THREADS=2 "$PY" -I phases/phase_j_build_cache.py \
    --fold training/fold \
    --out-dir training/fold/labels \
    --indices training/prep/all857_indices.csv \
    --bio-dir training/prep/job_bioassembly \
    --mmcif-dir training/prep/job_cifs \
    --only {} --unit-seeds "$LO-$HI" \
    --report training/fold/label_reports/{}_${LO}_${HI}.csv
RC=$?
say "xargs exit $RC"

# EXIT 0 IS NOT COVERAGE. The builder skips a unit it cannot build and still exits 0, which is how
# 1,290 missing validation labels once reached a launch. Count what is actually on disk against
# what the trunks offer in this range, and report the gap as the answer.
$PY -I - <<'PYEOF'
import os, collections
lo, hi = int(os.environ["LO"]), int(os.environ["HI"])
want = [l.strip() for l in open(os.environ["LIST"]) if l.strip() and l[0] != "#"]
def seeds(root, p):
    d = os.path.join(root, p)
    if not os.path.isdir(d):
        return set()
    return {int(f[5:-4]) for f in os.listdir(d)
            if f.startswith("free_") and f.endswith(".npz")}
miss = collections.Counter()
gaps = []
for p in want:
    tr = {s for s in seeds("training/fold/trunks", p) if lo <= s <= hi}
    lb = {s for s in seeds("training/fold/labels", p) if lo <= s <= hi}
    g = tr - lb
    if g:
        miss[p] = len(g)
        gaps.append((p, sorted(g)[:3]))
print(f"  trunk units in {lo}-{hi} with no label: {sum(miss.values())} "
      f"over {len(miss)} structure(s)")
for p, g in gaps[:8]:
    print(f"    {p}: missing {g}")
raise SystemExit(1 if miss else 0)
PYEOF
G=$?
say "coverage check exit $G (0 = every trunk in $LO-$HI has a label)"
[ $RC -ne 0 ] || [ $G -ne 0 ] && { say "LABELS INCOMPLETE for $LO-$HI"; exit 1; }
say "LABELS COMPLETE for seeds $LO-$HI over $N structure(s)"
