#!/bin/bash
# Build the RCSB-derived training labels and audit them against SAbDab2. Reproducible end to end.
#
# WHY A DRIVER. The two steps have to agree on four paths -- selection, source cifs, jobs, output --
# and the validation side needs a selection file that does not otherwise exist, since the 53
# structures are split across val_split_proposal.csv (40 first representatives) and
# val_split_second_reps.csv (13 seconds). Getting that wrong silently audits the wrong label set.
#
# WHY VALIDATION LABELS EXIST AT ALL. The validation arm's own metrics -- decoy, coverage, regret --
# need only a scalar DockQ per pose, which the SAbDab2 reference supplies, so labels are not required
# for them. They are built because the validation structures are expected to become TRAINING data
# once the training setup is settled, and training needs per-atom coordinates across the whole folded
# chain, which SAbDab2's variable-region crop cannot give.
#
# SOURCE CIFS. Training reads SAbDab2_data/rcsb_training_cifs. Validation reads
# SAbDab2_data/rcsb_validation_cifs, which was populated from a session scratchpad that would have
# been wiped -- the deposits existed nowhere permanent until 2026-10-15.
#
#   bash phase_j_labels_and_audit.sh training
#   bash phase_j_labels_and_audit.sh validation
#   bash phase_j_labels_and_audit.sh both
set -euo pipefail
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-/home/ebrahim/Protenix/.venv/bin/python}
WHICH=${1:-both}

val_selection() {
    # the union of first and second representatives, written fresh so it cannot drift
    $PY -I - <<'PYX'
import csv, os
R = os.environ["R"]
out = f"{R}/train_val_splits/val_all_structures.csv"
rows, seen = [], set()
for f in ("val_split_proposal.csv", "val_split_second_reps.csv"):
    p = f"{R}/train_val_splits/{f}"
    if not os.path.exists(p):
        raise SystemExit(f"STOP: {p} missing; the validation selection cannot be rebuilt")
    for r in csv.DictReader(open(p)):
        if r.get("pdb") and r["pdb"] not in seen:
            seen.add(r["pdb"]); rows.append({"pdb": r["pdb"], "instance": r.get("instance", "")})
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["pdb", "instance"]); w.writeheader()
    for r in rows: w.writerow(r)
print(f"  {out}: {len(rows)} structures")
missing = [r["pdb"] for r in rows if not r["instance"]]
if missing:
    raise SystemExit(f"STOP: {len(missing)} structures have no SAbDab2 instance id: {missing[:5]}")
PYX
}

run_training() {
    echo "== training labels =="
    mkdir -p "$R/training/prep/job_cifs"
    $PY -I "$R/phases/phase_j_job_cifs.py" \
        --selection "$R/train_val_splits/cache_train_structures.csv" \
        --src-dir   "$R/SAbDab2_data/rcsb_training_cifs" \
        --jobs-dir  "$R/training/jobs" \
        --out-dir   "$R/training/prep/job_cifs" \
        --native-gaps "$R/training/prep/native_gaps.csv" \
        --unplaced    "$R/training/prep/unplaced_residues.csv" \
        --skipped     "$R/training/prep/skipped_job_cifs.csv"
    echo "== training audit =="
    $PY -I "$R/phases/phase_j_label_audit.py" --which training \
        --coverage-hist \
        --out "$R/train_val_splits/label_audit_training.csv"
}

run_validation() {
    echo "== validation selection =="
    R="$R" val_selection
    echo "== validation labels =="
    mkdir -p "$R/validation/prep/job_cifs"
    $PY -I "$R/phases/phase_j_job_cifs.py" \
        --selection "$R/train_val_splits/val_all_structures.csv" \
        --src-dir   "$R/SAbDab2_data/rcsb_validation_cifs" \
        --jobs-dir  "$R/validation/jobs" \
        --out-dir   "$R/validation/prep/job_cifs" \
        --native-gaps "$R/validation/prep/native_gaps.csv" \
        --unplaced    "$R/validation/prep/unplaced_residues.csv" \
        --skipped     "$R/validation/prep/skipped_job_cifs.csv"
    echo "== validation audit =="
    $PY -I "$R/phases/phase_j_label_audit.py" --which validation \
        --selection "$R/train_val_splits/val_all_structures.csv" \
        --labels    "$R/validation/prep/job_cifs" \
        --roles     "$R/validation/fold/chain_roles.csv" \
        --coverage-hist \
        --out "$R/train_val_splits/label_audit_validation.csv"
}

case "$WHICH" in
    training)   run_training ;;
    validation) run_validation ;;
    both)       run_training; run_validation ;;
    *) echo "usage: $0 [training|validation|both]"; exit 2 ;;
esac
