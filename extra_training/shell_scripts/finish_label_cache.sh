#!/bin/bash
# WAIT FOR THE RUNNING LABEL BUILD, THEN REPAIR WHAT IT BUILT WITH THE OLD CODE, THEN VERIFY.
#
# WHY THIS IS A SCRIPT AND NOT A REMEMBERED STEP. The full build now in flight carries the masking
# bug from review round 7: check_label queried SORTED coordinates and applied the result to the
# ORIGINAL atom order, so for a structure that takes the masking branch it RETAINED the invalid
# atoms and masked valid ones instead (8av9: 4 atoms still 2.436 A from the job cif). Only
# structures with recorded drops AND a prior atom-check failure reach that branch -- 8av9 and 8pww --
# plus 7stz and 8gpu, whose job cifs were corrected after this build started. Those four must be
# rebuilt with the fixed code, and training must not start until they are.
#
# A corrective pass that depends on someone remembering it is how the 2026-10-10 fold loss happened.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=/home/ebrahim/Protenix/.venv/bin/python
LOG="$R/label_cache_finish.log"
REPAIR="7stz,8gpu,8av9,8pww"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for any running phase_j_build_cache to exit"
while pgrep -f "[p]hase_j_build_cache" > /dev/null 2>&1; do sleep 60; done
say "the build has exited"

say "repairing $REPAIR with the fixed masking code (--overwrite forces it past the fingerprint)"
CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=openfold $PY -I "$R/phases/phase_j_build_cache.py" \
    --only "$REPAIR" --fold "$R/training/fold" --out-dir "$R/training/fold/labels" \
    --overwrite >> "$LOG" 2>&1
rc=$?
say "repair exit=$rc"
[ $rc -ne 0 ] && { say "REPAIR FAILED -- do NOT train. The four structures above have no trusted label."; exit 1; }

# EVERY REMAINING RESOLVED ATOM OF EVERY REPAIRED LABEL MUST PASS THE COORDINATE CHECK. The builder
# asserts this per unit now, but it is asserted again here against the labels ON DISK, because what
# matters is the file training will read, not what the builder believed when it wrote it.
say "verifying the repaired labels on disk"
CUDA_VISIBLE_DEVICES=0 $PY -I - >> "$LOG" 2>&1 <<'PYV'
import numpy as np, gemmi, glob, os, sys
from scipy.spatial import cKDTree
R=os.environ.get("R") or "."
bad=0
for pdb in ("7stz","8gpu","8av9","8pww"):
    cif=f"{R}/training/prep/job_cifs/{pdb}.cif"
    st=gemmi.read_structure(cif); st.setup_entities(); st.remove_hydrogens()
    atoms=np.array([[a.pos.x,a.pos.y,a.pos.z] for ch in st[0] for r in ch for a in r])
    tree=cKDTree(atoms)
    for p in sorted(glob.glob(f"{R}/training/fold/labels/{pdb}/*.npz")):
        z=np.load(p, allow_pickle=True)
        lc,lm=z["label_coordinate"],z["label_coordinate_mask"]
        for i in range(lc.shape[0]):
            m=lm[i].astype(bool)
            d,_=tree.query(np.ascontiguousarray(lc[i][m],dtype=np.float64),k=1)
            n=int((d>0.01).sum())
            if n:
                bad+=1
                print(f"  BAD {os.path.basename(os.path.dirname(p))}/{os.path.basename(p)} pose {i}: "
                      f"{n} atom(s) >0.01 A, worst {d.max():.3f} A")
print(f"on-disk verification: {'ALL CLEAN' if not bad else str(bad)+' POSE(S) BAD'}")
sys.exit(1 if bad else 0)
PYV
rc=$?
say "on-disk verification exit=$rc"
[ $rc -ne 0 ] && { say "VERIFICATION FAILED -- do NOT train."; exit 1; }

say "asserting manifest completeness (501 structures x 10 seeds, 9mx1 excluded as unfoldable)"
CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=openfold $PY -I "$R/phases/phase_j_build_cache.py" \
    --fold "$R/training/fold" --out-dir "$R/training/fold/labels" \
    --require-complete --manifest-skip 9mx1 --limit 1 >> "$LOG" 2>&1
say "completeness exit=$? (see $LOG)"
say "LABEL CACHE FINISHED. Training may start."
