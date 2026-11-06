#!/bin/bash
# An 11th seed (500010) for the 437 NON-degenerate training structures.
#
# WHY ONLY THOSE 437. The other 416 are the structures whose every interface is degenerate by the
# project's rule -- DockQ sd < 0.05 or best pose < 0.23 -- and they were trimmed to one seed
# earlier today, reclaiming 290.6 GiB. Giving them an 11th seed would undo that. These 437 are the
# ones that already carry ranking signal, so an extra seed adds 5 more poses to a population that
# can actually be ranked.
#
# WHAT THIS BREAKS, AND IT IS NOT HYPOTHETICAL. After this run the training set has a NON-UNIFORM
# seed count: 437 structures at 11 seeds, 416 at 1. Three things assume a uniform 10-seed range and
# must be taught otherwise before this data is usable:
#   - run_logging/build_combined_train_pool.py requires the exact key set for seeds 500000-500009,
#     so at --seeds 10 it will now report "5 unexpected" for every one of these interfaces and
#     REFUSE them from the pool. It needs --seeds 11 for these, or a per-structure expectation.
#   - phases/phase_i_check.py and phases/phase_l_merge_recovered.py take --seeds/--seed0 and
#     enumerate a contiguous range; they will not see seed 500010 unless told.
#   - phases/phase_k_train.py assumes a uniform seed range under --rank-cross-seed. This is the
#     pruning doc's blocker 3 and it is now live.
# Folding first is deliberate -- it is the critical path and the CPU-side fixes can follow -- but
# nothing should train on this until they do.
#
# THE ENV VARS ARE WHERE THE TRUNKS COME FROM. The original fold's argv carries no trunk flag; the
# 1.5 TB under training/fold/trunks was produced by TRUNK_DUMP_DIR and the two PROTENIX_DUMP_*
# switches, so they are set here to match. 437 units at ~151 MB is about 66 GiB.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration
EG=/home/ebrahim/confidence_recalibration/extra_training
TG=$EG/training/fold
Q="${QUEUE:?pass QUEUE=<path to the seed-500010 queue.tsv>}"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

# ONE LAUNCH, AND NEVER A SECOND. Same discipline as launch_wave1_val_fold.sh, which exists
# because three arming watchers once reached a launcher at the same time and put two folds on each
# card, roughly doubling per-unit time.
mkdir -p "$TG"
exec 9>"$TG/.launch_seed500010.lock"
if ! flock -n 9; then say "REFUSING: another launch holds the lock."; exit 1; fi
S=$(tmux ls 2>/dev/null | grep -cE 'w1fold|w1val' || true)
if [ "$S" -gt 0 ]; then say "REFUSING: $S fold session(s) already exist."; exit 1; fi
B=$($PY -I run_logging/cards_busy.py | tail -1)
if [ "${B:-1}" != "0" ]; then say "REFUSING: cards_busy reports '$B', not 0."; exit 1; fi

# THE POOL GATE, which is the check whose absence silently skipped a whole queue once: folder.py
# builds its interface registry from --pool, so a structure absent from it yields "no interfaces to
# score" and the unit folds for nothing.
say "preflight: every queued structure pooled, with a job json and a reference"
QUEUE="$Q" $PY -I - <<'EOF' || exit 1
import csv, glob, os, sys
q = [l.rstrip("\n").split("\t") for l in open(os.environ["QUEUE"]) if l.strip()]
pdbs = sorted({f[1] for f in q if len(f) >= 5})
pool = {r["pdb"] for r in csv.DictReader(open("training/fold/pool.csv"))}
bad = {
    "not in pool.csv": [p for p in pdbs if p not in pool],
    "no job json": [p for p in pdbs if not glob.glob(f"training/jobs/{p}*.json")],
    "no reference": [p for p in pdbs if not os.path.exists(f"training/refs/{p}.cif")],
}
seeds = {f[4] for f in q if len(f) >= 5}
print(f"  {len(q)} unit(s), {len(pdbs)} structure(s), seed(s) {sorted(seeds)}")
if seeds != {"500010"}:
    print(f"  STOP: expected only seed 500010, got {sorted(seeds)}"); sys.exit(1)
for why, xs in bad.items():
    if xs:
        print(f"  STOP: {len(xs)} structure(s) {why}: {xs[:6]}"); sys.exit(1)
print("  preflight OK")
EOF

# DISK, against the user's quota rather than df. The quota is per uid over the whole device and has
# no grace period, so a fold that runs it out fails on a write rather than warning.
AVAIL=$($PY -I -c "
import subprocess
o = subprocess.run(['quota','-u','ebrahim'],capture_output=True,text=True).stdout.split()
for i,t in enumerate(o):
    if t.startswith('/dev/'):
        print(int((int(o[i+3])-int(o[i+1]))/2**20)); break
")
say "quota headroom ${AVAIL} GiB; this run needs about 66 GiB of trunks"
if [ "${AVAIL:-0}" -lt 150 ]; then
  say "REFUSING: under 150 GiB of quota headroom."; exit 1
fi

say "launching 3 workers on cards 4, 5, 7 -- fold AND score inline"
for c in 4 5 7; do
  tmux new-session -d -s w1fold_$c "cd /home/ebrahim/Protenix && CUDA_VISIBLE_DEVICES=$c \
    LAYERNORM_TYPE=openfold TRUNK_DUMP_DIR=$TG/trunks PROTENIX_DUMP_TRUNK_BF16=1 \
    PROTENIX_DUMP_CONF_INPUTS=1 $PY $R/run_model/folder.py --queue $Q --out $TG/poses \
    --csv $TG/results.csv --claim-dir $TG/claims --pool $TG/pool.csv --dataset sabdab_train \
    --ncycle 10 --nsample 5 --dtype bf16 --overlap-scoring >> $TG/w1fold_seed500010_c$c.log 2>&1"
done
sleep 45
UP=$(tmux ls 2>/dev/null | grep -c w1fold || true)
say "w1fold sessions up: $UP of 3"
if [ "$UP" -lt 3 ]; then
  say "ONLY $UP WORKER(S) CAME UP -- check $TG/w1fold_seed500010_c*.log"
  exit 1
fi
say "SEED 500010 FOLD LAUNCHED: 437 units, 3 workers"
