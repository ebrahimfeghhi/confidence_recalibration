#!/bin/bash
# MOVE ASIDE the pose directories whose trunk the DockQ-SD<0.05 prune deleted.
#
# WHY A MOVE AND NOT A DELETE. These 198 directories hold the only copy of the poses that the 2,160
# rows in archive/orphaned_degenerate_rows.csv describe. Nothing can train on them -- the trunk
# features are gone -- but they are the provenance of those labels, so they go beside that csv.
#
# WHY THEY HAVE TO MOVE AT ALL. folder.py:996 checks saved_run_dir BEFORE folding and, finding poses
# with no score rows, rescores them and returns "rescored": no trunk written, not counted in the
# summary. With these dirs in place 198 of the 308 units would silently not fold.
#
# ONLY seeds 500001-500009 OF THE 22. Seed 500000 keeps its poses, its trunk and its rows.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
DEST=training/fold/archive/stale_poses_degenerate_20261114
LIST=docs/degenerate_regen_structures.txt
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }

mapfile -t DIRS < <(ls -d training/fold/poses/*/50000[1-9]-* 2>/dev/null |
                    awk -F/ -v OFS=/ '{print $0, $4}' | while IFS= read -r l; do
                        d=${l%/*}; p=${l##*/}
                        grep -qxF "$p" <(grep -v '^#' $LIST) && echo "$d"
                    done)
N=${#DIRS[@]}
say "$N stale pose dir(s) to move -> $DEST"
[ "$N" -eq 198 ] || { say "ABORT: expected 198, found $N"; exit 2; }
BYTES=$(du -sm "${DIRS[@]}" 2>/dev/null | awk '{s+=$1} END{print s}')
say "  ${BYTES} MB"
mkdir -p "$DEST"
MOVED=0
for d in "${DIRS[@]}"; do
    P=$(echo "$d" | awk -F/ '{print $4}')
    mkdir -p "$DEST/$P"
    # SAME FILESYSTEM, so this is a rename: no copy, no window where both exist.
    mv "$d" "$DEST/$P/" || { say "ABORT: mv failed on $d"; exit 3; }
    MOVED=$((MOVED + 1))
done
say "moved $MOVED dir(s)"
LEFT=$(ls -d training/fold/poses/*/50000[1-9]-* 2>/dev/null | awk -F/ '{print $4}' |
       grep -xFf <(grep -v '^#' $LIST) | wc -l)
GOT=$(ls -d $DEST/*/*/ 2>/dev/null | wc -l)
say "verify: $LEFT left in place (want 0), $GOT in the archive (want $N)"
[ "$LEFT" -eq 0 ] && [ "$GOT" -eq "$N" ] || { say "ABORT: verification failed"; exit 4; }
# AND SEED 500000 IS UNTOUCHED, asserted rather than assumed.
K=$(ls -d training/fold/poses/*/500000-* 2>/dev/null | awk -F/ '{print $4}' |
    grep -xFf <(grep -v '^#' $LIST) | wc -l)
say "seed 500000 pose dirs still in place for $K of 22 structures"
[ "$K" -eq 22 ] || { say "ABORT: seed 500000 lost a pose dir"; exit 5; }
say "DONE"
