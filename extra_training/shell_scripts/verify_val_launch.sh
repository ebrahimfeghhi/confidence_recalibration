#!/bin/bash
# Confirm the validation fold is actually FOLDING AND SCORING, not merely that tmux sessions exist.
#
# A tmux session proves a process started. It does not prove the model loaded, that the worker
# claimed a unit, that it is folding the structures we queued, or that scoring produces rows -- and
# the whole reason this handoff needed care is that a fold can run to completion and score nothing
# when its reference is missing. So this waits for the sessions, then waits for EVIDENCE.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
VG=validation/fold
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

say "waiting for the validation fold to launch"
while [ "$(tmux ls 2>/dev/null | grep -c w1val)" -eq 0 ]; do sleep 60; done
say "w1val sessions present: $(tmux ls 2>/dev/null | grep -c w1val)/3"

say "waiting up to 40 min for all three workers to load the model"
for i in $(seq 1 80); do
  N=$(grep -l "model loaded" $VG/w1val_c*.log 2>/dev/null | wc -l)
  [ "$N" -eq 3 ] && break
  sleep 30
done
say "workers that loaded the model: $N/3"
for c in 4 5 7; do
  echo "  c$c: $(grep -m1 'model loaded' $VG/w1val_c$c.log 2>/dev/null | sed 's/^ *//' | cut -c1-80)"
done

say "waiting up to 40 min for the first scored units"
for i in $(seq 1 80); do
  S=$(grep -hc "interfaces in" $VG/w1val_c*.log 2>/dev/null | paste -sd+ | bc)
  [ "${S:-0}" -ge 3 ] && break
  sleep 30
done
say "units scored so far: ${S:-0}"
for c in 4 5 7; do
  echo "  c$c last: $(grep 'interfaces in' $VG/w1val_c$c.log 2>/dev/null | tail -1 | sed 's/^ *//')"
done

say "verifying the evidence, not the sessions"
$PY -I - <<'EOF'
import csv, glob, os, sys
sys.path.insert(0, "run_logging")
sys.path.insert(0, "/home/ebrahim/confidence_recalibration/run_model")
_argv, sys.argv = sys.argv, ["folder.py"]
import folder
sys.argv = _argv

q = {u[1] for u in folder.read_queue("validation/fold/queue.tsv")}
sel = {r["pdb"] for r in csv.DictReader(open("validation/full_val/val30_structures.csv"))}
print(f"  queued structures match the selection: {q == sel} ({len(q)} queued)")

rows = list(csv.reader(open("validation/fold/results.csv")))
new = [r for r in rows if r and r[0].split("_")[0] in sel]
print(f"  DockQ rows for the NEW 30 structures: {len(new)}")
finite = 0
for r in new:
    try:
        if 0.0 <= float(r[6]) <= 1.0 and 0.0 <= float(r[7]) <= 1.0:
            finite += 1
    except (IndexError, ValueError):
        pass
print(f"    of those, dockq and iptm both in [0,1]: {finite}")
for r in new[:4]:
    print(f"      {','.join(r[:8])}")
tr = len(glob.glob("validation/fold/trunks/*/*.npz"))
print(f"  validation trunk units on disk: {tr} (1,590 belong to the old benchmark)")
folded_new = {os.path.basename(d) for d in glob.glob("validation/fold/trunks/*")} & sel
print(f"  new structures with trunks: {len(folded_new)}")
if not new:
    print("  WARNING: workers are running but no DockQ row exists for any new structure yet.")
    print("  That is the failure the training fold hit -- folding fine, scoring nothing.")
    sys.exit(1)
print("  VALIDATION IS FOLDING AND SCORING")
EOF
say "VERIFY DONE (exit $?)"
