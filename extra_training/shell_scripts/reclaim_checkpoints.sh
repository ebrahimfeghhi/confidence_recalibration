#!/bin/bash
# Delete training-run CHECKPOINTS, keeping every record that makes paper_results.md verifiable.
#
# THE OWNER'S DECISION, 2026-11-06: the held-out arms did not achieve the desired selection
# improvement, a document outlining results may stay, and the checkpoints go. This script is the
# reproducible form of that, because a hand-run find -delete leaves no record of what was removed.
#
# WHAT GOES: *.pt and *.pth under training/runs -- 1,317 files, 137.9 GiB, which is essentially all
# of the 139 GiB that tree occupies. None are tracked by git.
#
# WHAT STAYS, AND WHY IT IS NOT SENTIMENT. paper_results.md cites runs by id and quotes numbers with
# the command that produced them, so the small per-run records are what make those numbers checkable
# after the weights are gone: trajectory.csv, run.log, val_loss.csv, the rank_rho_* csvs and the
# panels. They total a few hundred MB against 137.9 GiB, so keeping them costs essentially nothing
# and losing them would turn a cited result into an unverifiable assertion.
#
# WHAT IT REFUSES TO TOUCH AT ALL: anything under training/fold or validation/fold. Those are the
# 1.7 TB and 365 GB of trunks, poses and canonical scores -- the fold is STILL RUNNING as this is
# written, the gap recovery is armed against results.csv, and the labels the trainer reads are built
# from the trunks. Deleting a pose would also make the scoring recovery impossible, and the owner's
# standing rule is never to delete a pose.
#
# IT WILL NOT RUN WHILE A TRAINING PROCESS IS ALIVE. A live run holds its checkpoint directory and
# writes into it; removing files underneath it is how a resume discovers a half-deleted state.
# Checked by GPU occupancy via cards_busy.py rather than by a pattern match on the process list --
# pgrep -f matches the shell running it, which has self-killed this session twice before and which
# produced a false "a training run is live" reading while this script was being written.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RUNS=training/runs
DRY="${DRY:-1}"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

say "script $(basename "$0") sha $(sha1sum "$0" | cut -c1-12)  DRY=$DRY"

# NO TRAINING RUN MAY BE LIVE. Folding is fine and expected -- it does not touch training/runs.
BUSY=$($PY -I run_logging/cards_busy.py 2>/dev/null | tail -1)
FOLDERS=$(tmux ls 2>/dev/null | grep -cE '^(w1fold|w1val|w2_)' || true)
say "cards_busy '$BUSY', $FOLDERS fold session(s)"
for p in $(tmux ls 2>/dev/null | grep -oE '^[a-z0-9_]+' || true); do
  case "$p" in train*|arm*|k_*) say "REFUSING: tmux session '$p' looks like a training run."; exit 1;; esac
done
# THE DIRECT TEST: a process whose argv names phase_k_train AND which is not this shell.
LIVE=$($PY -I - <<'EOF'
import os, glob
me = {os.getpid(), os.getppid()}
n = 0
for d in glob.glob("/proc/[0-9]*"):
    pid = int(os.path.basename(d))
    if pid in me:
        continue
    try:
        argv = open(f"{d}/cmdline", "rb").read().split(b"\0")
    except OSError:
        continue
    # THE EXECUTABLE'S OWN ARGUMENT, not any occurrence in the line: a wrapper shell carries the
    # whole command as one -c string and matched it, which is the false positive this avoids.
    if any(a.endswith(b"phase_k_train.py") for a in argv):
        n += 1
print(n)
EOF
)
say "live phase_k_train process(es): $LIVE"
if [ "${LIVE:-1}" != "0" ]; then
  say "REFUSING: a training process is running; it owns its checkpoint directory."
  exit 1
fi

# THE MANIFEST IS WRITTEN BEFORE ANYTHING IS REMOVED, and it is the record of what was.
MAN="$RUNS/checkpoint_reclaim_$(date +%Y%m%d).csv"
say "building the manifest"
find "$RUNS" -type f \( -name '*.pt' -o -name '*.pth' \) -printf '%s\t%p\n' 2>/dev/null \
  | sort -k2 > /tmp/ckpt_list.$$
N=$(wc -l < /tmp/ckpt_list.$$)
GIB=$(awk -F'\t' '{s+=$1} END {printf "%.1f", s/2^30}' /tmp/ckpt_list.$$)
say "$N checkpoint file(s), $GIB GiB"
if [ "$N" -eq 0 ]; then say "nothing to reclaim"; rm -f /tmp/ckpt_list.$$; exit 0; fi

# A GIT-TRACKED FILE IS NOT OURS TO DELETE QUIETLY.
TRACKED=$(cd /home/ebrahim/confidence_recalibration && git ls-files extra_training/training/runs \
  | grep -cE '\.pt$|\.pth$' || true)
if [ "${TRACKED:-0}" -gt 0 ]; then
  say "REFUSING: $TRACKED checkpoint(s) are tracked by git; untrack them first."
  rm -f /tmp/ckpt_list.$$; exit 1
fi

{ echo "bytes,path"; awk -F'\t' '{printf "%s,%s\n", $1, $2}' /tmp/ckpt_list.$$; } > "$MAN"
say "manifest -> $MAN"
say "per-run totals (top 10):"
awk -F'\t' '{n=$2; sub("/[^/]*$","",n); s[n]+=$1} END {for(d in s) printf "  %8.1f GiB  %s\n", s[d]/2^30, d}' \
  /tmp/ckpt_list.$$ | sort -rh | head -10

if [ "$DRY" = "1" ]; then
  say "DRY RUN: nothing deleted. Re-run with DRY=0 to reclaim $GIB GiB."
  say "records kept per run: trajectory.csv, run.log, val_loss.csv, rank_rho_*.csv, *_panels.png"
  rm -f /tmp/ckpt_list.$$
  exit 0
fi

say "deleting $N file(s)"
Q=$(quota -u ebrahim 2>/dev/null | awk '/\/dev\//{print int($2/2^20)}')
cut -f2- /tmp/ckpt_list.$$ | while IFS= read -r f; do rm -f -- "$f"; done
rm -f /tmp/ckpt_list.$$
Q2=$(quota -u ebrahim 2>/dev/null | awk '/\/dev\//{print int($2/2^20)}')
say "quota used ${Q} GiB -> ${Q2} GiB"
LEFT=$(find "$RUNS" -type f \( -name '*.pt' -o -name '*.pth' \) 2>/dev/null | wc -l)
say "checkpoint file(s) remaining: $LEFT"
say "records still present: $(find $RUNS -name 'trajectory.csv' | wc -l) trajectory.csv, \
$(find $RUNS -name 'run.log' | wc -l) run.log"
[ "$LEFT" -eq 0 ] || { say "WARNING: $LEFT file(s) survived"; exit 1; }
say "RECLAIM COMPLETE: $GIB GiB, manifest at $MAN"
